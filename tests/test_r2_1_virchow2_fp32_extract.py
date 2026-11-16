"""Synthetic tests for ``prosper.evaluation.r2_1_virchow2_fp32_extract_cli`` (Stage V1).

No model load, no WSI, no CUDA.  Fake encoder / opener / GPU-tuple doubles are
injected through ``run_extract``'s seams.  The code root is located via env
``R21_V2_CODE_ROOT`` (default: the package's parent directory) and the frozen
extractor X is asserted at its pinned sha256 before anything else.

The marked real test (``R21_V2_REAL=1``) is skipped by default; see its
docstring for the inputs the analyst must supply.

Layout (candidate 002, M1 design decision (a)): ``$PROSPER_OUT`` is a tmp root,
the run root is ``$PROSPER_OUT/r2_1_virchow2_fp32_20261006/run_001`` and
production writes at the run root itself.  Production runs with the fake
encoder on CPU use the two keyword-only ``run_extract`` test seams
(``test_only_allow_cpu_production``, ``test_only_code_root_anywhere``), which
the CLI refuses unless both encoder and tuple-observer doubles are injected.
``CUDA_VISIBLE_DEVICES`` is set to ``""`` by the ``world`` fixture (L6b).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from prosper.evaluation import r2_1_perturb_extract_cli as X
from prosper.evaluation import r2_1_virchow2_fp32_extract_cli as M

CODE_ROOT = Path(os.environ.get("R21_V2_CODE_ROOT", str(Path(M.__file__).resolve().parents[2]))).resolve()
X_PATH = CODE_ROOT / "prosper" / "evaluation" / "r2_1_perturb_extract_cli.py"
PY = sys.executable


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------

_G = torch.Generator().manual_seed(20261006)
PROJ = torch.randn(3 * 8 * 8, M.MODEL_DIM, generator=_G, dtype=torch.float32)


def fake_transform(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((8, 8), Image.NEAREST), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def fake_forward(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1) @ PROJ


_DEFAULT = object()


def default_weights():
    return {"repo_id": M.HF_REPO_ID, "file": "model.safetensors",
            "snapshot_revision": M.HF_SNAPSHOT_REVISION, "cache_path": "/fake", "blob": "/fake/blob",
            "bytes": M.WEIGHTS_BYTES, "sha256": M.WEIGHTS_SHA256}


def make_factory(*, weights=_DEFAULT, dim=M.MODEL_DIM, amp=False, forward=None, name=M.BANK):
    w = default_weights() if weights is _DEFAULT else weights

    def factory(n, device, a, hash_weights=True):
        return X.EncoderRuntime(name, None, fake_transform, forward or fake_forward, amp, dim,
                                {"recipe": "fake", "transform": "fake_transform", "weights": w,
                                 "autocast_dtype": None, "storage_dtype": "float32"})
    return factory


class FakeSlide:
    """Pixel content is a deterministic function of (x, y); optional random delay
    and a set of failing locations."""

    def __init__(self, path, *, delay=0.0, fail_at=()):
        self.path = path
        self.delay = delay
        self.fail_at = set(fail_at)
        self.sid = Path(path).stem

    def read_region(self, loc, level, size):
        x, y = loc
        if (x, y) in self.fail_at:
            raise OSError(f"fake read failure at {loc}")
        if self.delay:
            time.sleep(random.random() * self.delay)
        h = int(hashlib.sha256(f"{self.sid}:{x}:{y}".encode()).hexdigest()[:6], 16)
        col = (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)
        return Image.new("RGBA", size, col + (255,))


def make_opener(**kw):
    def opener(path):
        return FakeSlide(path, **kw)
    return opener


SMI_RAW = "NVIDIA A100-SXM4-80GB, 81920 MiB, 595.58.03\n"


def make_observer(tup=None, raw=SMI_RAW):
    t = dict(M.SEED_TUPLE) if tup is None else tup

    def observer(device):
        return dict(t), raw
    return observer


OTHER_TUPLE = {**M.SEED_TUPLE, "nvidia_smi_driver_version": "550.127.05", "torch_cuda": "12.1"}


# --------------------------------------------------------------------------
# Synthetic world
# --------------------------------------------------------------------------

SLIDES = {"10002": (5, "float16"), "15678": (12, "float16"), "16839": (3, "float32"), "20001": (7, "float16")}


def coords_for(sid: str, n: int):
    base = int(sid) % 7
    return [{"wsi_loc": [int(512 * (i + base)), int(512 * ((i * 3) % 5))]} for i in range(n)]


RUN_PARENT = "r2_1_virchow2_fp32_20261006"
CONTRACT_CALIBRATION_SLIDES = ("10079", "12543", "14490", "14613", "14628", "15358", "15414", "16839",
                               "17125", "17141", "17147", "17148", "19115", "19131", "7419", "9934")


class World:
    pass


def set_hf_library_offline(monkeypatch, value) -> None:
    """The in-process test session imported huggingface_hub before any fixture
    ran, so its frozen flag reflects the pytest launch environment; the
    synthetic world pins it.  Real-library behaviour is covered by the
    subprocess tests (``test_hf_offline_real_library_*``)."""
    import huggingface_hub.constants as hfc
    monkeypatch.setattr(hfc, "HF_HUB_OFFLINE", value)
    tf = sys.modules.get("transformers.utils.hub")
    if tf is not None:
        monkeypatch.setattr(tf, "_is_offline_mode", value)


def build_world(tmp_path: Path, monkeypatch) -> World:
    w = World()
    w.root = tmp_path
    w.banks = tmp_path / "old_banks"
    (w.banks / "virchow2").mkdir(parents=True)
    (w.banks / "h_optimus_1").mkdir(parents=True)
    w.wsi = tmp_path / "wsi"
    w.wsi.mkdir()
    w.coords = {}
    w.old_feats = {}
    rows = []
    rng = np.random.default_rng(7)
    for sid in sorted(SLIDES):
        n, dt = SLIDES[sid]
        coords = coords_for(sid, n)
        w.coords[sid] = coords
        feats = rng.standard_normal((n, M.WIDTH)).astype(dt)
        w.old_feats[sid] = feats
        p = w.banks / "virchow2" / f"{sid}.pkl"
        p.write_bytes(pickle.dumps((feats, coords), protocol=4))
        hfe = rng.standard_normal((n, 1536)).astype(np.float32)
        hp = w.banks / "h_optimus_1" / f"{sid}.pkl"
        hp.write_bytes(pickle.dumps((hfe, [dict(wsi_loc=list(c["wsi_loc"])) for c in coords]), protocol=4))
        (w.wsi / f"{sid}.svs").write_bytes(b"fake")
        for bank, pp, dtn in (("virchow2", p, dt), ("h_optimus_1", hp, "float32")):
            rows.append({"bank": bank, "slide_id": sid, "canonical_tiles": n, "feature_rows": n, "dtype": dtn,
                         "feature_path": str(pp), "resolved_path": str(pp), "size": pp.stat().st_size,
                         "sha256": sha256_path(pp)})
    w.rows = rows
    w.inventory = tmp_path / "inventory.csv"
    with open(w.inventory, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    w.inventory_sha = sha256_path(w.inventory)
    w.allowlist = tmp_path / "allowlist.json"
    w.allowlist.write_text(json.dumps([M.SEED_TUPLE], indent=2, sort_keys=True) + "\n")
    w.allowlist_sha = sha256_path(w.allowlist)
    w.manifest = tmp_path / "code_manifest.json"
    w.manifest.write_text(json.dumps(M.loaded_code_modules(CODE_ROOT), indent=2, sort_keys=True) + "\n")
    w.manifest_sha = sha256_path(w.manifest)
    w.prosper_out = tmp_path / "prosper_out"
    (w.prosper_out / "features").mkdir(parents=True)
    w.run_parent = w.prosper_out / RUN_PARENT
    w.run_parent.mkdir()
    w.run_root = w.run_parent / "run_001"                  # not created: production writes here (M1)
    w.out_root = w.run_root
    for k in M.FORBIDDEN_ENV:
        monkeypatch.delenv(k, raising=False)
    for k in ("LSB_JOBID", "LSB_JOBINDEX", "LSB_QUEUE", "LSB_DJOB_NUMPROC"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")           # L6b: no CUDA driver init; restored after
    for k in M.HF_OFFLINE_ENV:
        monkeypatch.setenv(k, "1")                           # NM1: must be inherited; restored after
    set_hf_library_offline(monkeypatch, True)                # stands in for "exported before start"
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("COMET_SLIDES_ROOT", str(w.wsi))
    monkeypatch.setenv("PROSPER_OUT", str(w.prosper_out))
    return w


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


def base_args(w: World, *, out_root=None, run_root=None, lo=1, hi=None, pilot=False, num_workers=1, extra=()):
    args = ["--inventory", str(w.inventory), "--inventory-sha256", w.inventory_sha,
            "--weights-sha256", M.WEIGHTS_SHA256,
            "--allowlist", str(w.allowlist), "--allowlist-sha256", w.allowlist_sha,
            "--code-manifest", str(w.manifest), "--code-manifest-sha256", w.manifest_sha,
            "--code-root", str(CODE_ROOT), "--run-root", str(run_root or w.run_root),
            "--out-root", str(out_root or w.out_root),
            "--lo", str(lo), "--hi", str(hi if hi is not None else len(SLIDES)),
            "--num-workers", str(num_workers), "--device", "cpu"]
    if pilot:
        args.append("--pilot-observe-tuple")
    return args + list(extra)


def run(w: World, args, *, factory=None, opener=None, observer=None, cpu_production=True, code_anywhere=True):
    """Synthetic run with injected doubles.  Production on ``--device cpu`` and a
    code root outside ``run_001/code_frozen`` need the two test-only seams."""
    return M.run_extract(args, encoder_factory=factory or make_factory(), opener=opener or make_opener(),
                         tuple_observer=observer or make_observer(),
                         test_only_allow_cpu_production=cpu_production,
                         test_only_code_root_anywhere=code_anywhere)


def expected_features(sid: str, rt=None) -> np.ndarray:
    """Independent re-computation of X's identity path on the fake doubles."""
    rt = rt or make_factory()(M.BANK, "cpu", False)
    slide = FakeSlide(str(sid) + ".svs")
    tens = [rt.transform(slide.read_region(tuple(c["wsi_loc"]), 0, (512, 512)).convert("RGB"))
            for c in coords_for(sid, SLIDES[sid][0])]
    out = X._forward(rt, torch.stack(tens), "cpu")[:, :M.WIDTH]
    return np.ascontiguousarray(out.astype(np.float32))


def load_out(w: World, sid: str, out_root=None):
    pkl, rec, _ = M.output_paths(out_root or w.out_root, sid)
    with open(pkl, "rb") as fh:
        feats, coords = pickle.load(fh)
    return feats, coords, json.loads(rec.read_text()), pkl, rec


def failed_files(w: World, out_root=None):
    d = Path(out_root or w.out_root) / "FAILED"
    return sorted(d.iterdir()) if d.exists() else []


# --------------------------------------------------------------------------
# Pins and import hygiene
# --------------------------------------------------------------------------


def test_x_sha_pinned_at_code_root():
    assert X_PATH.is_file(), X_PATH
    assert sha256_path(X_PATH) == M.X_SHA256_PINNED
    assert Path(X.__file__).resolve() == X_PATH.resolve()


def test_import_does_not_initialise_cuda():
    assert not torch.cuda.is_initialized()


def test_record_keys_frozen_and_unique():
    assert len(M.RECORD_KEYS) == len(set(M.RECORD_KEYS)) == 70
    for k in ("allowlist_mode", "observed_tuple", "matched_tuple", "allowlist_sha256",
              "code_manifest_sha256", "prosper_file", "sys_path_head", "pickle_sha256",
              "hoptimus_coords_source", "old_bank_dtype", "lsf_queue", "argv", "threads",
              "hf_offline_env", "num_workers", "forward_batch", "loader_batch"):
        assert k in M.RECORD_KEYS


# --------------------------------------------------------------------------
# Production extract: records, features, coords
# --------------------------------------------------------------------------


def test_production_run_writes_complete_records(world):
    rc = run(world, base_args(world))
    assert rc == 0
    assert failed_files(world) == []
    runs = list((world.out_root / "runs").glob("*.json"))
    assert len(runs) == 1
    rr = json.loads(runs[0].read_text())
    assert rr["counts"] == {"written": 4, "skipped": 0, "failed": 0}
    for sid in SLIDES:
        feats, coords, rec, pkl, _ = load_out(world, sid)
        assert tuple(sorted(rec.keys())) == tuple(sorted(M.RECORD_KEYS))
        assert rec["allowlist_mode"] == "production"
        assert rec["matched_tuple"] == M.SEED_TUPLE and rec["observed_tuple"] == M.SEED_TUPLE
        assert rec["gpu"]["NVIDIA_TF32_OVERRIDE"] is None and rec["gpu"]["CUBLAS_WORKSPACE_CONFIG"] is None
        assert rec["gpu"]["gpu_model"] == "NVIDIA A100-SXM4-80GB" and rec["gpu"]["nvidia_smi"] == SMI_RAW
        assert rec["dtype"] == "float32" and rec["width"] == 1280 and rec["dim"] == 2560
        assert rec["amp"] is False and rec["c_contiguous"] is True
        assert rec["torch_backend_flags"]["cuda_matmul_allow_tf32"] is False
        assert rec["torch_backend_flags"]["cudnn_benchmark"] is False
        assert "cudnn_allow_tf32" in rec["torch_backend_flags"] and "cudnn_deterministic" in rec["torch_backend_flags"]
        assert rec["pickle_bytes"] == pkl.stat().st_size and rec["pickle_sha256"] == sha256_path(pkl)
        assert rec["pickle_protocol"] == 4
        assert rec["inventory_sha256"] == world.inventory_sha
        assert rec["coords_source"]["sha256"] == rec["coords_source_sha256"]
        assert rec["hoptimus_coords_source"]["path"].endswith(f"h_optimus_1/{sid}.pkl")
        assert rec["old_bank_dtype"] == SLIDES[sid][1]
        assert rec["weights_sha256"] == M.WEIGHTS_SHA256
        assert rec["extractor_code_sha256"] == M.X_SHA256_PINNED
        assert rec["code_sha256"] == sha256_path(Path(M.__file__))
        assert rec["code_manifest_sha256"] == world.manifest_sha
        assert rec["allowlist_sha256"] == world.allowlist_sha
        assert rec["threads"] == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        assert rec["hf_offline_env"]["HF_HUB_OFFLINE"] == "1"
        assert set(rec["versions"]) >= {"torch", "cuda", "cudnn", "timm", "huggingface_hub", "safetensors",
                                        "openslide_python", "openslide_library", "pillow", "numpy"}
        assert rec["prosper_file"].startswith(str(CODE_ROOT)) and len(rec["sys_path_head"]) <= 3
        assert set(rec["code_manifest_excluded_modules"]) <= {"torch.ops", "torch.classes"}
        assert rec["num_workers"] == 1 and rec["forward_batch"] == 256 and rec["loader_batch"] == 8
        assert rec["n_forward_batches"] == 1
        assert rec["slide_path"] == str(world.wsi / f"{sid}.svs")
        assert feats.dtype == np.float32 and feats.shape == (SLIDES[sid][0], 1280)
        assert feats.flags["C_CONTIGUOUS"]
        exp = expected_features(sid)
        assert np.array_equal(feats, exp)
        assert np.ascontiguousarray(feats).tobytes() == exp.tobytes()


def test_coords_written_as_exact_list_object_types(world):
    assert run(world, base_args(world, lo=1, hi=1)) == 0
    sid = sorted(SLIDES)[0]
    feats, coords, rec, _, _ = load_out(world, sid)
    assert coords == world.coords[sid]
    assert isinstance(coords, list)
    for c in coords:
        assert type(c) is dict and type(c["wsi_loc"]) is list
        assert all(type(v) is int for v in c["wsi_loc"])


def test_features_match_x_identity_path_bytewise(world):
    assert run(world, base_args(world)) == 0
    rt = make_factory()(M.BANK, "cpu", False)
    for sid in SLIDES:
        feats, coords, _, _, _ = load_out(world, sid)
        exp = expected_features(sid, rt)
        assert np.array_equal(feats, exp) and feats.tobytes() == exp.tobytes()


def test_three_workers_random_delay_preserve_row_order(world):
    args = base_args(world, num_workers=3)
    assert run(world, args, opener=make_opener(delay=0.02)) == 0
    for sid in SLIDES:
        feats, coords, rec, _, _ = load_out(world, sid)
        assert rec["num_workers"] == 3
        assert np.array_equal(feats, expected_features(sid))


def test_forward_batch_chunking_counts(world, monkeypatch):
    """A slide with more tiles than --forward-batch is encoded in exact 256-tile
    chunks in mosaic order; the fake forward is row-wise so bytes are unchanged."""
    monkeypatch.setattr(M, "FORWARD_BATCH", 4)
    args = base_args(world, lo=2, hi=2, extra=["--forward-batch", "4"])
    assert run(world, args) == 0
    sid = sorted(SLIDES)[1]
    feats, _, rec, _, _ = load_out(world, sid)
    assert rec["n_forward_batches"] == 3 and rec["forward_batch"] == 4
    assert np.array_equal(feats, expected_features(sid))


def test_run_record_tag_naming_outside_lsf(world):
    assert run(world, base_args(world, lo=1, hi=1)) == 0
    runs = list((world.out_root / "runs").glob("*.json"))
    assert len(runs) == 1
    pid = os.getpid()
    assert runs[0].name.startswith(f"local.{pid}.local.{pid}.{pid}.") and runs[0].name.endswith(".json")


# --------------------------------------------------------------------------
# Skip predicate (A-F5, B-N1)
# --------------------------------------------------------------------------


def test_second_run_skips_without_rewrite(world):
    assert run(world, base_args(world)) == 0
    stats = {sid: (load_out(world, sid)[3].stat().st_mtime_ns, load_out(world, sid)[4].stat().st_mtime_ns)
             for sid in SLIDES}
    time.sleep(0.01)
    assert run(world, base_args(world)) == 0
    assert failed_files(world) == []
    for sid in SLIDES:
        _, _, _, pkl, rec = load_out(world, sid)
        assert (pkl.stat().st_mtime_ns, rec.stat().st_mtime_ns) == stats[sid]
    runs = sorted((world.out_root / "runs").glob("*.json"))
    assert len(runs) == 2
    assert json.loads(runs[-1].read_text())["counts"]["skipped"] == 4 or \
        json.loads(runs[0].read_text())["counts"]["skipped"] == 4


def test_orphan_pickle_is_failed_not_overwritten(world):
    sid = sorted(SLIDES)[0]
    pkl, rec, _ = M.output_paths(world.out_root, sid)
    pkl.parent.mkdir(parents=True)
    pkl.write_bytes(b"orphan")
    assert run(world, base_args(world, lo=1, hi=1)) == 1
    assert pkl.read_bytes() == b"orphan" and not rec.exists()
    ff = failed_files(world)
    assert len(ff) == 1
    obj = json.loads(ff[0].read_text())
    assert obj["stage"] == "existing_output" and "orphan" in obj["error"]


def test_orphan_record_is_failed(world):
    sid = sorted(SLIDES)[0]
    pkl, rec, _ = M.output_paths(world.out_root, sid)
    rec.parent.mkdir(parents=True)
    rec.write_text("{}")
    assert run(world, base_args(world, lo=1, hi=1)) == 1
    assert not pkl.exists() and rec.read_text() == "{}"
    assert json.loads(failed_files(world)[0].read_text())["stage"] == "existing_output"


MUTATIONS = {
    "slide_id": lambda r: r.update(slide_id="99999"),
    "encoder": lambda r: r.update(encoder="virchow2"),
    "n_tiles": lambda r: r.update(n_tiles=r["n_tiles"] + 1),
    "pickle_bytes": lambda r: r.update(pickle_bytes=r["pickle_bytes"] + 1),
    "pickle_sha256": lambda r: r.update(pickle_sha256="0" * 64),
    "inventory_sha256": lambda r: r.update(inventory_sha256="1" * 64),
    "coords_source": lambda r: r["coords_source"].update(sha256="2" * 64),
    "hoptimus_coords_source": lambda r: r["hoptimus_coords_source"].update(size=1),
    "code_sha256": lambda r: r.update(code_sha256="3" * 64),
    "extractor_code_sha256": lambda r: r.update(extractor_code_sha256="4" * 64),
    "code_manifest_sha256": lambda r: r.update(code_manifest_sha256="5" * 64),
    "weights_sha256": lambda r: r.update(weights_sha256="6" * 64),
    "width": lambda r: r.update(width=2560),
    "dtype": lambda r: r.update(dtype="float16"),
    "gpu_env_tf32": lambda r: r["gpu"].update(NVIDIA_TF32_OVERRIDE="0"),
    "gpu_env_cublas": lambda r: r["gpu"].update(CUBLAS_WORKSPACE_CONFIG=":4096:8"),
    "allowlist_mode_pilot": lambda r: r.update(allowlist_mode="pilot"),
    "matched_tuple_null": lambda r: r.update(matched_tuple=None),
    "matched_tuple_other": lambda r: r.update(matched_tuple=OTHER_TUPLE),
    "device": lambda r: r.update(device="cuda:0"),           # M2: normalised device is a skip field
}


@pytest.mark.parametrize("field", sorted(MUTATIONS))
def test_record_mismatch_on_each_skip_field_is_failed(world, field):
    assert run(world, base_args(world, lo=1, hi=1)) == 0
    sid = sorted(SLIDES)[0]
    _, _, rec, pkl, recp = load_out(world, sid)
    MUTATIONS[field](rec)
    recp.write_text(json.dumps(rec, sort_keys=True))
    before = (pkl.read_bytes(), recp.read_bytes())
    assert run(world, base_args(world, lo=1, hi=1)) == 1
    assert (pkl.read_bytes(), recp.read_bytes()) == before
    ff = failed_files(world)
    assert len(ff) == 1 and json.loads(ff[0].read_text())["stage"] == "existing_output"


def test_matched_tuple_must_be_in_current_allowlist(world):
    """A complete record whose matched tuple is no longer allowlisted is a conflict."""
    assert run(world, base_args(world, lo=1, hi=1)) == 0
    world.allowlist.write_text(json.dumps([OTHER_TUPLE]) + "\n")
    world.allowlist_sha = sha256_path(world.allowlist)
    assert run(world, base_args(world, lo=1, hi=1), observer=make_observer(OTHER_TUPLE)) == 1
    assert json.loads(failed_files(world)[0].read_text())["stage"] == "existing_output"


# --------------------------------------------------------------------------
# Start-up refusals (B-N9 order) — none may touch a tile or write features
# --------------------------------------------------------------------------


def assert_no_features(w: World, out_root=None):
    d = Path(out_root or w.out_root) / "features"
    assert not d.exists() or not any(d.rglob("*.pkl"))


def test_env_refusal_forbidden_var_even_empty(world, monkeypatch, capsys):
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "")
    assert run(world, base_args(world)) == 2
    assert "[env]" in capsys.readouterr().err
    assert not world.out_root.exists()
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert run(world, base_args(world)) == 2
    assert not world.out_root.exists()


def test_env_refusal_threads_not_one(world, monkeypatch, capsys):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert run(world, base_args(world)) == 2
    assert "OMP_NUM_THREADS" in capsys.readouterr().err
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("MKL_NUM_THREADS")
    assert run(world, base_args(world)) == 2


def test_env_check_precedes_inventory_check(world, monkeypatch, capsys):
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    args = base_args(world)
    args[args.index("--inventory-sha256") + 1] = "0" * 64
    assert run(world, args) == 2
    assert "[env]" in capsys.readouterr().err


def test_code_root_refusal(world, tmp_path, capsys):
    args = base_args(world)
    args[args.index("--code-root") + 1] = str(tmp_path / "elsewhere")
    assert run(world, args) == 2
    assert "[code_root]" in capsys.readouterr().err


def test_inventory_sha_refusal(world, capsys):
    args = base_args(world)
    args[args.index("--inventory-sha256") + 1] = "0" * 64
    assert run(world, args) == 2
    assert "[inventory]" in capsys.readouterr().err
    assert not world.out_root.exists()


def test_inventory_lex_order_refusal(world, tmp_path):
    rows = list(csv.DictReader(world.inventory.read_text().splitlines()))
    v2 = [r for r in rows if r["bank"] == "virchow2"]
    rows = [r for r in rows if r["bank"] != "virchow2"] + v2[::-1]
    p = tmp_path / "inv_bad.csv"
    with open(p, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    args = base_args(world)
    args[args.index("--inventory") + 1] = str(p)
    args[args.index("--inventory-sha256") + 1] = sha256_path(p)
    assert run(world, args) == 2


def test_lo_hi_and_num_workers_refusals(world, monkeypatch):
    assert run(world, base_args(world, lo=0, hi=1)) == 2
    assert run(world, base_args(world, lo=1, hi=99)) == 2
    assert run(world, base_args(world, lo=3, hi=2)) == 2
    assert run(world, base_args(world, num_workers=0)) == 2
    args = base_args(world)
    del args[args.index("--num-workers"):args.index("--num-workers") + 2]
    assert run(world, args) == 2                              # no LSB_DJOB_NUMPROC
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "1")
    assert run(world, args) == 2                              # 1-1 = 0 workers refused
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "2")
    assert run(world, args) == 0
    assert load_out(world, sorted(SLIDES)[0])[2]["num_workers"] == 1


def test_width_and_batch_refusals(world):
    assert run(world, base_args(world, extra=["--width", "2560"])) == 2
    assert run(world, base_args(world, extra=["--forward-batch", "128"])) == 2
    assert run(world, base_args(world, extra=["--loader-batch", "16"])) == 2


@pytest.mark.parametrize("kind", ["comet_virchow2_name", "comet_robustness_symlink", "prosper_out_features",
                                  "inventory_parent"])
def test_forbidden_out_root_realpath(world, tmp_path, kind, capsys):
    if kind == "comet_virchow2_name":
        target = world.run_root / "comet-virchow2" / "x"
    elif kind == "comet_robustness_symlink":
        real = tmp_path / "comet-robustness" / "deep"
        real.mkdir(parents=True)
        world.run_root.mkdir()
        link = world.run_root / "innocent"
        link.symlink_to(real)
        target = link / "x"
    elif kind == "prosper_out_features":
        target = world.prosper_out / "features" / "x"
    else:
        target = world.banks / "virchow2" / "new"
    args = base_args(world, out_root=target)
    assert run(world, args) == 2
    assert "[out_root]" in capsys.readouterr().err
    assert not target.exists()


def test_out_root_must_be_under_run_root(world, tmp_path, capsys):
    """M1 design decision (a): production writes at the run root itself."""
    for bad in (tmp_path / "elsewhere" / "x", world.run_root / "sub", world.run_root / "pilot" / "hostA",
                world.run_root / "chunk_001"):
        assert run(world, base_args(world, out_root=bad)) == 2, bad
        assert "[out_root]" in capsys.readouterr().err
        assert not bad.exists()
    assert not (world.run_root / "features").exists()
    assert run(world, base_args(world, out_root=world.run_root)) == 0
    assert (world.run_root / "features" / "virchow2_fp32").is_dir()


def test_run_root_must_be_pinned_run_001(world, tmp_path, capsys):
    """M1: run root must be realpath($PROSPER_OUT)/r2_1_virchow2_fp32_20261006/run_001."""
    cases = [world.run_parent / "run_002",                          # wrong name
             tmp_path / "other" / RUN_PARENT / "run_001",           # wrong parent
             world.run_parent]                                      # parent itself
    for rr in cases:
        assert run(world, base_args(world, run_root=rr, out_root=rr)) == 2, rr
        assert "[out_root]" in capsys.readouterr().err
        assert not (rr / "features").exists()


def test_production_pointed_at_pilot_directory_refused_before_write(world, capsys):
    """f3 regression (M1, review probe f3): after a pilot into
    ``run_001/pilot/hostA``, a production run aimed at that pilot directory --
    via ``--run-root`` = parent of ``run_001`` (the probe), ``run_001/pilot``,
    ``run_001`` itself, or a run root named run_001 under the pilot dir -- is
    refused before any write; the pilot directory is left untouched."""
    pr = world.run_root / "pilot" / "hostA"
    assert run(world, base_args(world, out_root=pr, lo=1, hi=1, pilot=True)) == 0
    before = sorted(p.relative_to(pr) for p in pr.rglob("*"))
    assert before
    cases = ((world.run_parent, pr),                     # exact review probe f3: --run-root = parent of run_001
             (world.run_parent, pr / "x"),
             (world.run_root / "pilot", pr),                 # --run-root = run_001/pilot
             (world.run_root, pr), (pr / "run_001", pr / "run_001"), (world.run_root, pr / "x"))
    for rr, ro in cases:
        assert run(world, base_args(world, run_root=rr, out_root=ro)) == 2, (rr, ro)
        assert "[out_root]" in capsys.readouterr().err
    assert sorted(p.relative_to(pr) for p in pr.rglob("*")) == before
    assert not (pr / "FAILED").exists() or not any((pr / "FAILED").iterdir())   # no startup.* written
    assert not (pr / "run_001").exists() and not (pr / "x").exists()
    assert not (world.run_root / "features").exists()


def _tree(p: Path) -> list:
    return sorted(str(q.relative_to(p)) for q in p.rglob("*")) if p.exists() else []


def test_features_symlink_below_run_root_refused_review_p2(world, capsys):
    """NL1, review probe P2 verbatim: a pre-existing ``run_001/features ->
    $PROSPER_OUT/features`` (forbidden) symlink is refused before any write;
    nothing appears under the link target."""
    forbidden = world.prosper_out / "features"
    world.run_root.mkdir()
    (world.run_root / "features").symlink_to(forbidden, target_is_directory=True)
    assert run(world, base_args(world, lo=1, hi=1)) == 2
    err = capsys.readouterr().err
    assert "[symlink]" in err and "is a symlink" in err
    assert _tree(forbidden) == []
    assert sorted(os.listdir(world.run_root)) == ["features"]      # no FAILED/runs created


@pytest.mark.parametrize("rel", ["features", "features/virchow2_fp32", "FAILED", "runs"])
@pytest.mark.parametrize("pilot", [False, True])
def test_any_output_dir_symlink_refused_before_write(world, tmp_path, capsys, rel, pilot):
    """NL1: a symlink at any directory component the CLI creates/writes into is
    refused before any write.  In pilot mode the pre-existing tag directory is
    already refused by the single-shot rule ([out_root]); either way nothing is
    written through the link."""
    out_root = world.run_root / "pilot" / "hostA" if pilot else world.run_root
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = out_root / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    before = _tree(world.run_root)
    assert run(world, base_args(world, out_root=out_root, lo=1, hi=1, pilot=pilot)) == 2
    assert ("[out_root]" if pilot else "[symlink]") in capsys.readouterr().err
    assert _tree(target) == []
    assert _tree(world.run_root) == before


def test_pilot_component_symlink_refused(world, tmp_path, capsys):
    """NL1: ``run_001/pilot`` itself a symlink -> refused (realpath confinement)."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    world.run_root.mkdir()
    (world.run_root / "pilot").symlink_to(target, target_is_directory=True)
    assert run(world, base_args(world, out_root=world.run_root / "pilot" / "hostA", lo=1, hi=1, pilot=True)) == 2
    assert "[out_root]" in capsys.readouterr().err
    assert _tree(target) == []


@pytest.mark.parametrize("rel", ["FAILED", "features", "runs"])
@pytest.mark.parametrize("pilot", [False, True])
def test_symlink_created_after_startup_checks_refused_at_mkdir(world, tmp_path, capsys, rel, pilot):
    """NL1: a link appearing between the step-4 check and the mkdirs (here made
    by the encoder factory), in production or pilot mode, is refused; no
    startup FAILED and no slide output is written through it."""
    out_root = world.run_root / "pilot" / "hostA" if pilot else world.run_root
    target = tmp_path / "elsewhere"
    target.mkdir()
    base = make_factory()

    def linking(*a, **kw):
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / rel).symlink_to(target, target_is_directory=True)
        return base(*a, **kw)
    assert run(world, base_args(world, out_root=out_root, lo=1, hi=1, pilot=pilot), factory=linking) == 2
    assert "[symlink]" in capsys.readouterr().err
    assert _tree(target) == []
    assert not any(p.suffix == ".pkl" for p in world.run_root.rglob("*"))


def test_runs_symlink_appearing_mid_run_refuses_run_record(world, tmp_path, capsys):
    """NL1: ``runs`` replaced by a link while slides are processed -> the
    run-record publish is refused; nothing is written through the link."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    base_opener = make_opener()

    def swapping(path):
        runs = world.run_root / "runs"
        if runs.is_dir() and not runs.is_symlink():
            runs.rmdir()
            runs.symlink_to(target, target_is_directory=True)
        return base_opener(path)
    assert run(world, base_args(world, lo=1, hi=1), opener=swapping) == 2
    assert "[symlink]" in capsys.readouterr().err
    assert _tree(target) == []


@pytest.mark.parametrize("leaf", ["pkl", "record"])
def test_slide_output_leaf_symlink_is_slide_failure(world, tmp_path, leaf):
    """NL1: a symlinked ``<sid>.pkl``/``<sid>.record.json`` is a per-slide
    FAILED (stage ``symlink``); the link target is neither read nor written."""
    sid = sorted(SLIDES)[0]
    feat_dir = world.run_root / "features" / M.ENCODER
    feat_dir.mkdir(parents=True)
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"do not touch")
    pkl, rec, _ = M.output_paths(world.run_root, sid)
    (pkl if leaf == "pkl" else rec).symlink_to(victim)
    assert run(world, base_args(world, lo=1, hi=2)) == 1
    assert victim.read_bytes() == b"do not touch"
    fails = [json.loads(f.read_text()) for f in failed_files(world)]
    assert [(f["slide_id"], f["stage"]) for f in fails] == [(sid, "symlink")]
    assert load_out(world, sorted(SLIDES)[1])[0].shape[0] == SLIDES[sorted(SLIDES)[1]][0]


def test_check_no_symlink_below_unit(tmp_path):
    root = tmp_path / "r"
    (root / "a" / "b").mkdir(parents=True)
    M.check_no_symlink_below(root, root / "a" / "b" / "c" / "d")        # nonexistent tail ok
    M.check_no_symlink_below(root, root)
    (root / "a" / "l").symlink_to(root / "a" / "b")
    with pytest.raises(M.Refusal, match="symlink"):
        M.check_no_symlink_below(root, root / "a" / "l" / "x")
    (root / "a" / "dangling").symlink_to(tmp_path / "nowhere")
    with pytest.raises(M.Refusal, match="symlink"):
        M.check_no_symlink_below(root, root / "a" / "dangling")
    with pytest.raises(M.Refusal):
        M.check_no_symlink_below(root, tmp_path / "outside")


def test_weights_info_none_error_and_mismatch_refused(world, capsys):
    for w in (None, {"error": "HfHubHTTPError: offline"}, {**default_weights(), "sha256": None},
              {**default_weights(), "sha256": "f" * 64}, {**default_weights(), "repo_id": "x/y"},
              {**default_weights(), "snapshot_revision": "deadbeef"}):
        rc = run(world, base_args(world), factory=make_factory(weights=w))
        assert rc == 2, w
        assert "[weights]" in capsys.readouterr().err
        assert_no_features(world)
    assert run(world, base_args(world), factory=make_factory(dim=1280)) == 2
    assert run(world, base_args(world), factory=make_factory(amp=True)) == 2
    ff = failed_files(world)
    assert ff and all(f.name.startswith("startup.") for f in ff)


def test_gpu_allowlist_refusal_production(world, capsys):
    assert run(world, base_args(world), observer=make_observer(OTHER_TUPLE)) == 2
    assert "[gpu_tuple]" in capsys.readouterr().err
    assert_no_features(world)
    bad = {**M.SEED_TUPLE, "torch_cudnn": "90100"}            # wrong type
    assert run(world, base_args(world), observer=make_observer(bad)) == 2
    assert run(world, base_args(world), observer=make_observer({**M.SEED_TUPLE, "extra": 1})) == 2


def test_allowlist_file_validation(world):
    world.allowlist.write_text(json.dumps([M.SEED_TUPLE, M.SEED_TUPLE]))
    world.allowlist_sha = sha256_path(world.allowlist)
    assert run(world, base_args(world)) == 2                  # duplicate
    world.allowlist.write_text(json.dumps({"tuples": [M.SEED_TUPLE]}))
    world.allowlist_sha = sha256_path(world.allowlist)
    assert run(world, base_args(world)) == 2                  # not a list
    args = base_args(world)
    args[args.index("--allowlist-sha256") + 1] = "0" * 64
    assert run(world, args) == 2                              # sha mismatch


def test_cuda_device_refused_without_cuda(world):
    """L6b: the world fixture hides CUDA, so this always runs (never skips)."""
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not torch.cuda.is_available()
    for dev in ("cuda", "cuda:0"):
        args = base_args(world)
        args[args.index("--device") + 1] = dev
        assert run(world, args) == 2
        assert_no_features(world)


def test_production_requires_cuda_device(world, capsys):
    """M2: production refuses any device other than cuda / cuda:0 (no seam)."""
    for dev in ("cpu", "cuda:1", "mps", ""):
        args = base_args(world)
        args[args.index("--device") + 1] = dev
        assert run(world, args, cpu_production=False) == 2, dev
        assert "[args]" in capsys.readouterr().err
        assert_no_features(world)
        assert not world.run_root.exists()                   # refused before any write
    # pilot on cpu remains allowed (no device restriction in pilot mode)
    pr = world.run_root / "pilot" / "hostA"
    assert run(world, base_args(world, out_root=pr, lo=1, hi=1, pilot=True), cpu_production=False) == 0


def test_test_seams_refused_without_injected_doubles(world, capsys):
    """M2/L3: the test-only seams themselves refuse unless both encoder and
    tuple-observer doubles are injected."""
    args = base_args(world)
    for kw in ({"test_only_allow_cpu_production": True}, {"test_only_code_root_anywhere": True}):
        for inj in ({}, {"encoder_factory": make_factory()}, {"tuple_observer": make_observer()}):
            rc = M.run_extract(args, opener=make_opener(), **inj, **kw)
            assert rc == 2, (kw, list(inj))
            assert "[test_seam]" in capsys.readouterr().err
            assert_no_features(world)


def test_normalise_device_and_smi_selector(monkeypatch):
    assert M.normalise_device("cpu") == "cpu"
    assert M.normalise_device("cuda") == "cuda:0"
    assert M.normalise_device("cuda:0") == "cuda:0"
    assert M.normalise_device("cuda:3") == "cuda:3"
    for bad in ("", "gpu", "cuda:", "cuda:-1", "cuda:x", "CUDA"):
        with pytest.raises(M.Refusal):
            M.normalise_device(bad)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert M.nvidia_smi_selector(0) == "0" and M.nvidia_smi_selector(2) == "2"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,7")
    assert M.nvidia_smi_selector(0) == "5" and M.nvidia_smi_selector(1) == "7"
    with pytest.raises(M.Refusal):
        M.nvidia_smi_selector(2)


def test_skip_view_and_record_carry_normalised_device(world):
    assert "device" in M.SKIP_FIELDS and "device" in M.RECORD_KEYS
    assert run(world, base_args(world, lo=1, hi=1)) == 0
    rec = load_out(world, sorted(SLIDES)[0])[2]
    assert rec["device"] == "cpu"


def test_code_manifest_mismatch_missing_extra_refused(world, capsys):
    man = json.loads(world.manifest.read_text())
    key = next(k for k in man if k.endswith("r2_1_virchow2_fp32_extract_cli.py"))
    for mutate in (lambda m: m.update({key: "0" * 64}), lambda m: m.pop(key),
                   lambda m: m.update({"prosper/not_loaded.py": "1" * 64})):
        m = dict(man)
        mutate(m)
        world.manifest.write_text(json.dumps(m, sort_keys=True))
        world.manifest_sha = sha256_path(world.manifest)
        assert run(world, base_args(world)) == 2
        assert "[code_manifest]" in capsys.readouterr().err
        assert_no_features(world)


def test_pyc_without_py_refused(world, monkeypatch):
    fake = types.ModuleType("prosper._sourceless_fake")
    fake.__file__ = str(CODE_ROOT / "prosper" / "_sourceless_fake.pyc")
    monkeypatch.setitem(sys.modules, "prosper._sourceless_fake", fake)
    with pytest.raises(M.Refusal, match="non-source"):
        M.loaded_code_modules(CODE_ROOT)
    assert run(world, base_args(world)) == 2
    assert_no_features(world)


def test_relative_file_module_whitelist(world, monkeypatch):
    """L8: only torch.ops / torch.classes may carry a relative ``__file__``."""
    M.loaded_code_modules(CODE_ROOT)                          # baseline (incl. torch.ops/classes) passes
    fake = types.ModuleType("some_relative_fake")
    fake.__file__ = "foo.py"
    monkeypatch.setitem(sys.modules, "some_relative_fake", fake)
    with pytest.raises(M.Refusal, match="relative __file__"):
        M.loaded_code_modules(CODE_ROOT)
    assert run(world, base_args(world)) == 2
    assert_no_features(world)
    monkeypatch.delitem(sys.modules, "some_relative_fake")
    fake_ops = types.ModuleType("torch.ops")
    fake_ops.__file__ = "_ops.py"
    monkeypatch.setitem(sys.modules, "torch.ops", fake_ops)
    M.loaded_code_modules(CODE_ROOT)                          # whitelisted name is excluded, not refused
    assert "torch.ops" in M.NON_FILE_MODULE_EXCLUSIONS


def test_empty_expected_snapshot_revision_refused(world, capsys):
    """L1."""
    assert run(world, base_args(world, extra=["--expected-snapshot-revision", ""])) == 2
    assert "[args]" in capsys.readouterr().err
    assert not world.run_root.exists()


def test_weights_sha_flag_and_bytes_pinned(world, capsys):
    """L2: --weights-sha256 must equal the pinned sha; reported bytes must equal
    the pinned size."""
    args = base_args(world)
    args[args.index("--weights-sha256") + 1] = "f" * 64
    fac = make_factory(weights={**default_weights(), "sha256": "f" * 64})
    assert run(world, args, factory=fac) == 2                # consistent but not pinned
    assert "[args]" in capsys.readouterr().err
    assert not world.run_root.exists()
    for b in (M.WEIGHTS_BYTES + 1, 0):
        assert run(world, base_args(world), factory=make_factory(weights={**default_weights(), "bytes": b})) == 2
        assert "[weights]" in capsys.readouterr().err
        assert_no_features(world)


def test_production_code_root_must_be_run_code_frozen(world, capsys):
    """L3: without the test seam, production refuses a code root that is not
    ``run_001/code_frozen``; pilot mode is not restricted."""
    assert run(world, base_args(world), code_anywhere=False) == 2
    assert "[code_root]" in capsys.readouterr().err
    assert_no_features(world)
    pr = world.run_root / "pilot" / "hostA"
    assert run(world, base_args(world, out_root=pr, lo=1, hi=1, pilot=True), code_anywhere=False) == 0


def test_check_code_frozen_unit(tmp_path):
    rr = tmp_path / "run_001"
    (rr / "code_frozen").mkdir(parents=True)
    M.check_code_frozen(rr / "code_frozen", rr)
    M.check_code_frozen(rr / "code_frozen" / ".." / "code_frozen", rr)
    for bad in (rr, rr / "code_frozen" / "prosper", tmp_path / "code_frozen", CODE_ROOT):
        with pytest.raises(M.Refusal, match="code_root"):
            M.check_code_frozen(bad, rr)


@pytest.mark.parametrize("k", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
@pytest.mark.parametrize("val", [None, "0", "", "true"])
@pytest.mark.parametrize("pilot", [False, True])
def test_hf_offline_env_not_inherited_refused(world, monkeypatch, capsys, k, val, pilot):
    """NM1 (a): the CLI no longer sets the variables; unless both are already
    exactly "1" at start-up, extract/pilot refuse at step 1 before any write
    and before the encoder factory is called."""
    if val is None:
        monkeypatch.delenv(k)
    else:
        monkeypatch.setenv(k, val)
    calls = []
    base = make_factory()

    def counting(*args, **kw):
        calls.append(1)
        return base(*args, **kw)
    out_root = world.run_root / "pilot" / "hostA" if pilot else world.run_root
    assert run(world, base_args(world, out_root=out_root, pilot=pilot), factory=counting) == 2
    err = capsys.readouterr().err
    assert "[env]" in err and k in err and "before the process starts" in err
    assert not calls and not world.run_root.exists()
    assert os.environ.get(k) == (None if val is None else val)      # not overwritten by the CLI


@pytest.mark.parametrize("pilot", [False, True])
def test_hf_offline_library_flag_false_refused(world, monkeypatch, capsys, pilot):
    """NM1 (b): variables set but huggingface_hub's frozen flag False (the
    review P8 situation) -> refused at start-up, nothing written."""
    set_hf_library_offline(monkeypatch, False)
    calls = []
    base = make_factory()

    def counting(*args, **kw):
        calls.append(1)
        return base(*args, **kw)
    out_root = world.run_root / "pilot" / "hostA" if pilot else world.run_root
    assert run(world, base_args(world, out_root=out_root, pilot=pilot), factory=counting) == 2
    err = capsys.readouterr().err
    assert "[hf_offline]" in err and "HF_HUB_OFFLINE is False" in err
    assert not calls and not world.run_root.exists()


def test_hf_offline_transformers_flag_false_refused(world, monkeypatch, capsys):
    """NM1: if transformers is loaded its offline mode must also be True."""
    import types
    fake = types.ModuleType("transformers.utils.hub")
    fake.is_offline_mode = lambda: False
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", fake)
    assert run(world, base_args(world)) == 2
    assert "[hf_offline]" in capsys.readouterr().err and not world.run_root.exists()
    fake.is_offline_mode = lambda: True
    assert run(world, base_args(world, lo=1, hi=1)) == 0
    rec = load_out(world, sorted(SLIDES)[0])[2]
    assert rec["hf_offline_env"]["library_offline_before_encoder"]["transformers_offline_mode"] is True


def test_hf_offline_both_true_proceeds_and_records_observations(world, monkeypatch, capsys):
    """NM1 (c): both true -> runs (mocked encoder); record.json and the run
    record carry the observed env values and the live library flags at
    start-up, immediately before and after building the encoder."""
    import huggingface_hub.constants as hfc
    seen = []
    base = make_factory()

    def observing(*args, **kw):
        seen.append(hfc.HF_HUB_OFFLINE)
        return base(*args, **kw)
    assert run(world, base_args(world, lo=1, hi=1), factory=observing) == 0
    assert seen == [True]
    rec = load_out(world, sorted(SLIDES)[0])[2]
    env = rec["hf_offline_env"]
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    lib = {"huggingface_hub_constants_HF_HUB_OFFLINE": True,
           "transformers_offline_mode": None if "transformers.utils.hub" not in sys.modules else True}
    for k in ("library_offline_at_start", "library_offline_before_encoder", "library_offline_after_encoder"):
        assert env[k] == lib, k
    assert set(env) == {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "library_offline_at_start",
                        "library_offline_before_encoder", "library_offline_after_encoder"}
    runrec = json.loads(next((world.out_root / "runs").iterdir()).read_text())
    assert runrec["hf_offline_env"] == env


def test_hf_offline_changed_by_encoder_build_refused(world, monkeypatch, capsys):
    """NM1 (d): a factory that changes a variable or flips the library flag is
    refused after build, before any slide output."""
    import huggingface_hub.constants as hfc
    base = make_factory()
    for k in M.HF_OFFLINE_ENV:
        def bad_env(n, device, a, hash_weights=True, _k=k):
            os.environ[_k] = "0"
            return base(n, device, a, hash_weights)
        assert run(world, base_args(world, lo=2, hi=2), factory=bad_env) == 2
        assert "[env]" in capsys.readouterr().err
        assert not M.output_paths(world.out_root, sorted(SLIDES)[1])[0].exists()
        monkeypatch.setenv(k, "1")

    def bad_flag(n, device, a, hash_weights=True):
        hfc.HF_HUB_OFFLINE = False
        return base(n, device, a, hash_weights)
    assert run(world, base_args(world, lo=2, hi=2), factory=bad_flag) == 2
    assert "[hf_offline]" in capsys.readouterr().err
    assert not M.output_paths(world.out_root, sorted(SLIDES)[1])[0].exists()


_HF_PROBE = r"""
import json, os, sys
import prosper.evaluation.r2_1_virchow2_fp32_extract_cli as M
for k, v in json.loads(sys.argv[1]).items():
    os.environ[k] = v                         # set in-process, AFTER ``import prosper``
os.environ["PROSPER_OUT"] = sys.argv[2]
try:
    env = M.check_env()
    print(json.dumps({"ok": True, "env": env}))
except M.Refusal as exc:
    print(json.dumps({"ok": False, "stage": exc.stage, "message": exc.message}))
"""


def _hf_probe(tmp_path: Path, start_env: dict, late_env: dict) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in M.HF_OFFLINE_ENV and k not in M.FORBIDDEN_ENV}
    env.update({"PYTHONPATH": str(CODE_ROOT), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", **start_env})
    r = subprocess.run([PY, "-c", _HF_PROBE, json.dumps(late_env), str(tmp_path)], env=env,
                       capture_output=True, text=True, timeout=300, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_hf_offline_real_library_set_after_import_refused(tmp_path):
    """NM1, real huggingface_hub in a fresh interpreter (review probe P8):
    variables exported only after ``import prosper`` -> the gate sees the
    library's frozen False flag and refuses."""
    res = _hf_probe(tmp_path, {}, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    assert res == {"ok": False, "stage": "hf_offline", "message": res["message"]}
    assert "HF_HUB_OFFLINE is False" in res["message"]


def test_hf_offline_real_library_exported_at_start_passes(tmp_path):
    """NM1, real huggingface_hub: both variables exported before Python starts
    -> the real library flag is True and the gate passes."""
    res = _hf_probe(tmp_path, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, {})
    assert res["ok"] is True
    assert res["env"]["library_offline_at_start"]["huggingface_hub_constants_HF_HUB_OFFLINE"] is True
    assert res["env"]["HF_HUB_OFFLINE"] == "1" and res["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_prosper_out_unset_refused(world, monkeypatch, capsys):
    """L5."""
    monkeypatch.delenv("PROSPER_OUT")
    assert run(world, base_args(world)) == 2
    assert "[env]" in capsys.readouterr().err
    assert not world.run_root.exists()
    monkeypatch.setenv("PROSPER_OUT", "")
    assert run(world, base_args(world)) == 2
    with pytest.raises(M.Refusal):
        M.forbidden_parents([], [])


# --------------------------------------------------------------------------
# Pilot mode (R2/R3) and FAILED naming (B-N4)
# --------------------------------------------------------------------------


def test_pilot_mode_records_observed_only_and_confines_root(world):
    pr = world.run_root / "pilot" / "hostA"
    assert run(world, base_args(world, out_root=pr, lo=1, hi=2, pilot=True),
               observer=make_observer(OTHER_TUPLE)) == 0
    for sid in sorted(SLIDES)[:2]:
        _, _, rec, _, _ = load_out(world, sid, out_root=pr)
        assert rec["allowlist_mode"] == "pilot" and rec["matched_tuple"] is None
        assert rec["observed_tuple"] == OTHER_TUPLE
    # seed-tuple pilot arm also runs in pilot mode (R2)
    pr2 = world.run_root / "pilot" / "hostB"
    assert run(world, base_args(world, out_root=pr2, lo=1, hi=1, pilot=True)) == 0
    assert load_out(world, sorted(SLIDES)[0], out_root=pr2)[2]["matched_tuple"] is None
    # single-shot: existing pilot root refused
    assert run(world, base_args(world, out_root=pr, lo=1, hi=1, pilot=True)) == 2
    # confinement: not a direct child of run_root/pilot
    assert run(world, base_args(world, out_root=world.run_root / "pilot" / "deep" / "x", pilot=True)) == 2
    assert run(world, base_args(world, out_root=world.run_root / "chunk_009", pilot=True)) == 2


def test_failed_naming_inside_and_outside_lsf(world, monkeypatch):
    sid = sorted(SLIDES)[0]
    pkl, _, _ = M.output_paths(world.out_root, sid)
    pkl.parent.mkdir(parents=True)
    pkl.write_bytes(b"orphan")
    monkeypatch.setenv("LSB_JOBID", "326099999")
    monkeypatch.setenv("LSB_JOBINDEX", "7")
    assert run(world, base_args(world, lo=1, hi=1)) == 1
    assert [f.name for f in failed_files(world)] == [f"{sid}.326099999.7.json"]
    obj = json.loads(failed_files(world)[0].read_text())
    assert obj["lsf_job"] == "326099999" and obj["lsf_index"] == "7"
    monkeypatch.delenv("LSB_JOBINDEX")                        # either unset -> local.<pid> for both
    assert run(world, base_args(world, lo=1, hi=1)) == 1
    pid = os.getpid()
    assert f"{sid}.local.{pid}.local.{pid}.json" in [f.name for f in failed_files(world)]


def test_failed_file_never_overwritten_on_collision(world):
    sid = sorted(SLIDES)[0]
    pkl, _, _ = M.output_paths(world.out_root, sid)
    pkl.parent.mkdir(parents=True)
    pkl.write_bytes(b"orphan")
    assert run(world, base_args(world, lo=1, hi=1)) == 1
    first = failed_files(world)[0]
    content = first.read_bytes()
    assert run(world, base_args(world, lo=1, hi=1)) == 1
    ff = failed_files(world)
    assert len(ff) == 2 and first.read_bytes() == content


# --------------------------------------------------------------------------
# Per-slide failures
# --------------------------------------------------------------------------


def test_no_replace_publish_when_destination_appears_concurrently(world):
    sid = sorted(SLIDES)[0]
    pkl, rec, _ = M.output_paths(world.out_root, sid)
    state = {"done": False}

    def racing_forward(x):
        if not state["done"]:
            state["done"] = True
            pkl.parent.mkdir(parents=True, exist_ok=True)
            pkl.write_bytes(b"concurrent-writer")
        return fake_forward(x)

    assert run(world, base_args(world, lo=1, hi=1), factory=make_factory(forward=racing_forward)) == 1
    assert pkl.read_bytes() == b"concurrent-writer" and not rec.exists()
    obj = json.loads(failed_files(world)[0].read_text())
    assert obj["stage"] == "publish"
    assert not list(pkl.parent.glob(".*.tmp"))


def test_tile_read_error_is_failed_without_pickle(world):
    sid = sorted(SLIDES)[0]
    loc = tuple(world.coords[sid][2]["wsi_loc"])
    assert run(world, base_args(world, lo=1, hi=1), opener=make_opener(fail_at=[loc])) == 1
    pkl, rec, _ = M.output_paths(world.out_root, sid)
    assert not pkl.exists() and not rec.exists()
    obj = json.loads(failed_files(world)[0].read_text())
    assert obj["stage"] == "tile_read" and obj["n_tile_errors"] == 1


def test_non_finite_features_failed(world):
    def nan_forward(x):
        out = fake_forward(x)
        out[0, 0] = float("nan")
        return out
    assert run(world, base_args(world, lo=1, hi=1), factory=make_factory(forward=nan_forward)) == 1
    assert json.loads(failed_files(world)[0].read_text())["stage"] == "encode"


def test_coords_disagreement_with_hoptimus_failed(world):
    sid = sorted(SLIDES)[0]
    hp = world.banks / "h_optimus_1" / f"{sid}.pkl"
    feats, coords = pickle.loads(hp.read_bytes())
    coords[1]["wsi_loc"][0] += 512
    hp.write_bytes(pickle.dumps((feats, coords), protocol=4))
    rows = list(csv.DictReader(world.inventory.read_text().splitlines()))
    for r in rows:
        if r["bank"] == "h_optimus_1" and r["slide_id"] == sid:
            r["size"] = str(hp.stat().st_size)
            r["sha256"] = sha256_path(hp)
    with open(world.inventory, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    world.inventory_sha = sha256_path(world.inventory)
    assert run(world, base_args(world, lo=1, hi=1)) == 1
    obj = json.loads(failed_files(world)[0].read_text())
    assert obj["stage"] == "coords" and "differ" in obj["error"]


def test_old_pickle_sha_or_size_mismatch_failed(world):
    sid = sorted(SLIDES)[0]
    p = world.banks / "virchow2" / f"{sid}.pkl"
    p.write_bytes(p.read_bytes() + b"\x00")
    assert run(world, base_args(world, lo=1, hi=1)) == 1
    obj = json.loads(failed_files(world)[0].read_text())
    assert obj["stage"] == "coords" and "size mismatch" in obj["error"]


def test_slides_filter_and_missing_wsi(world):
    sid = sorted(SLIDES)[1]
    (world.wsi / f"{sid}.svs").unlink()
    assert run(world, base_args(world, extra=["--slides", sid])) == 1
    assert json.loads(failed_files(world)[0].read_text())["stage"] == "wsi"
    assert run(world, base_args(world, lo=1, hi=1, extra=["--slides", sid])) == 2  # outside lo..hi


# --------------------------------------------------------------------------
# make-params
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_make_params_coverage_and_bound(world, tmp_path, k):
    out = tmp_path / f"params_{k}.txt"
    rc = M.main(["make-params", "--inventory", str(world.inventory), "--inventory-sha256",
                 world.inventory_sha, "--k", str(k), "--out", str(out)])
    assert rc == 0
    lines = [ln.split() for ln in out.read_text().splitlines()]
    assert len(lines) == k
    sizes = [SLIDES[s][0] for s in sorted(SLIDES)]
    covered = []
    for i, (idx, lo, hi, ns, nt) in enumerate(lines):
        assert int(idx) == i + 1
        lo, hi = int(lo), int(hi)
        assert hi - lo + 1 == int(ns) and sum(sizes[lo - 1:hi]) == int(nt)
        covered += list(range(lo, hi + 1))
    assert covered == list(range(1, len(sizes) + 1))
    side = json.loads(out.with_suffix(".json").read_text())
    assert side["max_le_bound"] and side["max_chunk_tiles"] <= -(-sum(sizes) // k) + max(sizes)
    assert side["params_sha256"] == sha256_path(out)
    with pytest.raises(FileExistsError):                    # no-replace: existing params.txt
        M.run_make_params(["--inventory", str(world.inventory), "--inventory-sha256",
                           world.inventory_sha, "--k", str(k), "--out", str(out)])


def test_optimal_partition_is_optimal_on_random_cases():
    rng = random.Random(3)
    for _ in range(50):
        n = rng.randint(1, 9)
        sizes = [rng.randint(1, 50) for _ in range(n)]
        k = rng.randint(1, n)
        chunks, cap = M.optimal_partition(sizes, k)
        assert len(chunks) == k
        got = max(sum(sizes[a:b + 1]) for a, b in chunks)
        assert got <= cap
        # brute force optimum over all contiguous k-partitions
        import itertools
        best = None
        for cuts in itertools.combinations(range(1, n), k - 1):
            bounds = [0, *cuts, n]
            m = max(sum(sizes[bounds[i]:bounds[i + 1]]) for i in range(k))
            best = m if best is None else min(best, m)
        assert got == best == cap


def test_make_params_refuses_k_out_of_range(world, tmp_path):
    assert M.main(["make-params", "--inventory", str(world.inventory), "--inventory-sha256",
                   world.inventory_sha, "--k", "60", "--out", str(tmp_path / "p.txt")]) == 2


# --------------------------------------------------------------------------
# extend-allowlist / seed-allowlist
# --------------------------------------------------------------------------


def test_seed_allowlist_matches_world_allowlist(world, tmp_path, capsys):
    src = tmp_path / "x_first_run_record.json"
    src.write_text(json.dumps({"observed": M.SEED_TUPLE}))
    out = tmp_path / "seed.json"
    with pytest.raises(SystemExit) as ei:                       # L9: --provenance required (argparse)
        M.main(["seed-allowlist", "--out", str(out)])
    assert ei.value.code == 2
    capsys.readouterr()
    assert not out.exists()
    assert M.main(["seed-allowlist", "--out", str(out), "--provenance", str(tmp_path / "missing.json")]) == 2
    assert not out.exists()
    capsys.readouterr()
    assert M.main(["seed-allowlist", "--out", str(out), "--provenance", str(src)]) == 0
    assert json.loads(out.read_text()) == [M.SEED_TUPLE]
    assert capsys.readouterr().out.strip() == sha256_path(out)
    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["out_sha256"] == sha256_path(out) and prov["seed_tuple"] == M.SEED_TUPLE
    assert prov["sources"] == [{"path": str(M.realpath(src)), "bytes": src.stat().st_size,
                                "sha256": sha256_path(src)}]


def test_extend_allowlist_from_pilots(world, tmp_path, capsys):
    pa = world.run_root / "pilot" / "hostA"
    pb = world.run_root / "pilot" / "hostB"
    pc = world.run_root / "pilot" / "hostC"
    assert run(world, base_args(world, out_root=pa, lo=1, hi=2, pilot=True), observer=make_observer(OTHER_TUPLE)) == 0
    assert run(world, base_args(world, out_root=pb, lo=1, hi=1, pilot=True)) == 0
    # hostC: a pilot with a FAILED file
    (world.wsi / f"{sorted(SLIDES)[0]}.svs").unlink()
    assert run(world, base_args(world, out_root=pc, lo=1, hi=1, pilot=True), observer=make_observer(OTHER_TUPLE)) == 1
    out = tmp_path / "allowlist_v2.json"
    base = ["extend-allowlist", "--allowlist", str(world.allowlist), "--allowlist-sha256", world.allowlist_sha,
            "--pilot-root", str(world.run_root / "pilot")]
    # L9: --passed-host-tags required, must be a duplicate-free subset of --host-tags
    with pytest.raises(SystemExit) as ei:
        M.main(base + ["--host-tags", "hostA", "hostB", "--out", str(out)])
    assert ei.value.code == 2
    assert M.main(base + ["--host-tags", "hostA", "--passed-host-tags", "hostB", "--out", str(out)]) == 2
    assert M.main(base + ["--host-tags", "hostA", "hostB", "--passed-host-tags", "hostA", "hostA",
                          "--out", str(out)]) == 2
    assert not out.exists()
    capsys.readouterr()
    # a listed-but-not-passed tag is not admitted
    v_b = tmp_path / "only_b.json"
    assert M.main(base + ["--host-tags", "hostA", "hostB", "--passed-host-tags", "hostB", "--out", str(v_b)]) == 0
    assert json.loads(v_b.read_text()) == [M.SEED_TUPLE]
    capsys.readouterr()
    assert M.main(base + ["--host-tags", "hostA", "hostB", "--passed-host-tags", "hostA", "hostB",
                          "--out", str(out)]) == 0
    new = json.loads(out.read_text())
    assert new == [M.SEED_TUPLE, OTHER_TUPLE]
    assert capsys.readouterr().out.strip() == sha256_path(out)
    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["admitted"] == [OTHER_TUPLE] and len(prov["records"]) == 3
    assert prov["passed_host_tags"] == ["hostA", "hostB"]
    for r in prov["records"]:
        assert r["pickle_sha256"] == sha256_path(Path(r["pickle"]))
    # in-place / existing out refused
    assert M.main(base + ["--host-tags", "hostA", "--passed-host-tags", "hostA", "--out", str(out)]) == 2
    assert M.main(base + ["--host-tags", "hostA", "--passed-host-tags", "hostA",
                          "--out", str(world.allowlist)]) == 2
    # FAILED present refused
    assert M.main(base + ["--host-tags", "hostC", "--passed-host-tags", "hostC",
                          "--out", str(tmp_path / "v3.json")]) == 2
    # L9: tampered pilot pickle (sha no longer matches its record) refused
    pk = M.output_paths(pa, sorted(SLIDES)[0])[0]
    orig = pk.read_bytes()
    pk.write_bytes(orig[:-1] + bytes([orig[-1] ^ 1]))
    assert M.main(base + ["--host-tags", "hostA", "--passed-host-tags", "hostA",
                          "--out", str(tmp_path / "v5.json")]) == 2
    pk.write_bytes(orig)
    # L9: orphan pickle (no record) refused
    orphan = pk.parent / "99999.pkl"
    orphan.write_bytes(orig)
    assert M.main(base + ["--host-tags", "hostA", "--passed-host-tags", "hostA",
                          "--out", str(tmp_path / "v6.json")]) == 2
    orphan.unlink()
    # production-mode record never admitted
    assert run(world, base_args(world, lo=2, hi=2)) == 0
    prod = world.run_root / "pilot" / "hostD"
    (prod / "features").mkdir(parents=True)
    os.rename(world.out_root / "features" / M.ENCODER, prod / "features" / M.ENCODER)
    assert M.main(base + ["--host-tags", "hostD", "--passed-host-tags", "hostD",
                          "--out", str(tmp_path / "v4.json")]) == 2
    for n in ("v3", "v4", "v5", "v6"):
        assert not (tmp_path / f"{n}.json").exists()


# --------------------------------------------------------------------------
# ULP / calibrate / fidelity
# --------------------------------------------------------------------------


def test_ulp_distance_and_histogram():
    a = np.array([1.0, -1.0, 0.0, 2.5, -3.0], dtype=np.float32)
    b = a.copy()
    for _ in range(3):
        b = np.nextafter(b, np.float32(np.inf))
    d = M.ulp_distance(a, b)
    assert d.tolist() == [3, 3, 3, 3, 3]
    # +-0 are numerically equal and share ordinal 0; the smallest subnormal is 1 ULP from either.
    assert M.ulp_distance(np.array([-0.0], np.float32), np.array([0.0], np.float32)).tolist() == [0]
    tiny = np.nextafter(np.float32(0.0), np.float32(np.inf))
    assert M.ulp_distance(np.array([-0.0], np.float32), np.array([tiny], np.float32)).tolist() == [1]
    assert M.ulp_distance(np.array([-tiny], np.float32), np.array([tiny], np.float32)).tolist() == [2]
    h = M.ulp_histogram(np.array([0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 10 ** 6]))
    assert h == {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5-7": 2, "8-15": 2, "16-31": 2, "32-63": 2,
                 "64-127": 2, "128-255": 2, ">=256": 2}
    a16 = np.array([1.0, -2.0], np.float16)
    assert M.ulp_distance(a16, np.nextafter(a16, np.float16(np.inf))).tolist() == [1, 1]
    with pytest.raises(M.Refusal):
        M.ulp_distance(a, a.astype(np.float64))


def write_x_first_run(tmp_path: Path, sid: str, feats2560: np.ndarray, coords, *, name="x_first_run",
                      **rec_over) -> Path:
    """An X first-run style ``features/virchow2`` dir: ``({"identity": ...}, coords)``
    pickle plus a record carrying the keys step 4.0 checks."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    payload = pickle.dumps(({"identity": feats2560}, coords), protocol=4)
    (d / f"{sid}.pkl").write_bytes(payload)
    man = x_stimuli(tmp_path)
    rec = {"schema": M.X_RECORD_SCHEMA, "code_sha256": M.X_SHA256_PINNED, "encoder": "virchow2", "amp": False,
           "slide_id": sid, "n_tiles": int(feats2560.shape[0]), "dim": int(feats2560.shape[1]),
           "pickle_sha256": sha256_bytes(payload), "pickle_bytes": len(payload),
           "lsf_job": X_FIRST_RUN_JOB, "manifest_path": str(man), "manifest_sha256": sha256_path(man)}
    rec.update(rec_over)
    (d / f"{sid}.record.json").write_text(json.dumps(rec))
    return d


X_FIRST_RUN_JOB = "326997606"           # the real first run's lsf_job (string, as in its records)


def x_stimuli(tmp_path: Path) -> Path:
    """Stand-in for X's first-run stimulus manifest (NL3)."""
    man = tmp_path / "x_stimuli.json"
    if not man.exists():
        man.write_text(json.dumps({"stimuli": "synthetic first-run manifest"}) + "\n")
    return man


CAL_SID = next(s for s in SLIDES if SLIDES[s][1] == "float32")


@pytest.fixture
def cal_world(world, monkeypatch):
    """Pin step 4.0 to the one stored-fp32 world slide (production pins the
    specification's 16 slides / 41,714 tiles)."""
    monkeypatch.setattr(M, "CALIBRATION_SLIDES", (CAL_SID,))
    monkeypatch.setattr(M, "CALIBRATION_TILES", SLIDES[CAL_SID][0])
    return world


def _x_full(first1280: np.ndarray) -> np.ndarray:
    tail = np.full((first1280.shape[0], 1280), 7.0, np.float32)    # columns 1280: must be ignored
    return np.ascontiguousarray(np.concatenate([first1280.astype(np.float32), tail], axis=1))


def _cal_args(w: World, xd: Path, out: Path, extra=(), *, job=None, man=None, man_sha=None):
    man = man or x_stimuli(Path(xd).parent)
    return ["calibrate", "--inventory", str(w.inventory), "--inventory-sha256", w.inventory_sha,
            "--x-features-dir", str(xd), "--out", str(out),
            "--x-lsf-job", job or X_FIRST_RUN_JOB, "--x-manifest-path", str(man),
            "--x-manifest-sha256", man_sha or sha256_path(man), *extra]


def _rewrite_bank(w: World, bank: str, sid: str, feats, coords) -> None:
    p = w.banks / bank / f"{sid}.pkl"
    p.write_bytes(pickle.dumps((feats, coords), protocol=4))
    rows = list(csv.DictReader(w.inventory.read_text().splitlines()))
    for r in rows:
        if r["bank"] == bank and r["slide_id"] == sid:
            r["size"] = str(p.stat().st_size)
            r["sha256"] = sha256_path(p)
    with open(w.inventory, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    w.inventory_sha = sha256_path(w.inventory)


def test_calibration_slide_set_pinned_to_contract():
    """B1: the 16 stored-fp32 slides X's first run already extracted."""
    assert tuple(M.CALIBRATION_SLIDES) == CONTRACT_CALIBRATION_SLIDES
    assert len(set(M.CALIBRATION_SLIDES)) == 16 and M.CALIBRATION_TILES == 41714


def test_calibrate_published_vs_x_bit_identical_needs_no_v1_run(cal_world, tmp_path, capsys):
    """B1: published side is the record-less inventory pickle; X side is
    identity[:, :1280]; no V1 output is read (none exists)."""
    w = cal_world
    assert not list((w.banks / "virchow2").glob("*.record.json"))   # published dir is pickles only
    assert not w.run_root.exists()                                   # no V1 run at all
    xd = write_x_first_run(tmp_path, CAL_SID, _x_full(w.old_feats[CAL_SID]), w.coords[CAL_SID])
    out = tmp_path / "cal.json"
    assert M.main(_cal_args(w, xd, out)) == 0
    rep = json.loads(out.read_text())
    assert capsys.readouterr().out.strip() == sha256_path(out)
    assert rep["schema"] == M.CALIBRATE_SCHEMA and rep["step"] == "4.0"
    assert rep["slides"] == [CAL_SID] and rep["n_tiles"] == SLIDES[CAL_SID][0]
    assert rep["all_bit_identical"] and rep["band_abs"] == 1e-6 and rep["band_rel"] == 1e-7
    st = rep["per_slide"][CAL_SID]
    assert st["bit_identical"] and st["max_abs"] == 0 and st["denominator"] == "published"
    assert st["ulp_histogram_elements"]["0"] == SLIDES[CAL_SID][0] * 1280
    assert st["x_code_sha256"] == M.X_SHA256_PINNED and st["x_encoder"] == "virchow2" and st["x_amp"] is False
    # NL2: class (i), no author report
    assert st["class"] == "i" and rep["report_to_author"] is False
    assert rep["class_counts"] == {"i": 1, "ii": 0, "iii": 0}
    # NL3: first-run binding recorded
    man = x_stimuli(tmp_path)
    assert rep["x_first_run"] == {"lsf_job": X_FIRST_RUN_JOB, "manifest_path": str(M.realpath(man)),
                                  "manifest_sha256": sha256_path(man)}
    assert st["x_lsf_job"] == X_FIRST_RUN_JOB and st["x_manifest_sha256"] == sha256_path(man)
    pub = w.banks / "virchow2" / f"{CAL_SID}.pkl"
    assert st["published_pickle"] == {"path": str(pub), "size": pub.stat().st_size, "sha256": sha256_path(pub)}
    # --slides equal to the pinned set is accepted; any other set refused
    assert M.main(_cal_args(w, xd, tmp_path / "cal_s.json", ["--slides", CAL_SID])) == 0
    assert M.main(_cal_args(w, xd, tmp_path / "cal_bad.json", ["--slides", CAL_SID, "10002"])) == 2
    assert not (tmp_path / "cal_bad.json").exists()
    # the removed V1-output flag is no longer accepted
    with pytest.raises(SystemExit):
        M.main(_cal_args(w, xd, tmp_path / "cal_nf.json", ["--new-features-dir", str(tmp_path)]))


def test_calibrate_denominator_is_published_row(cal_world, tmp_path):
    """B1: X row = 2 x published row, so ||X|| != ||published||; rel-L2 with
    the published denominator is 1.0 (the X denominator would give 0.5)."""
    w = cal_world
    pub = w.old_feats[CAL_SID]
    xf = pub.copy()
    xf[0] = pub[0] * np.float32(2)
    xd = write_x_first_run(tmp_path, CAL_SID, _x_full(xf), w.coords[CAL_SID])
    out = tmp_path / "cal.json"
    assert M.main(_cal_args(w, xd, out)) == 0
    st = json.loads(out.read_text())["per_slide"][CAL_SID]
    d = (xf[0].astype(np.float64) - pub[0].astype(np.float64))
    want = np.linalg.norm(d) / np.linalg.norm(pub[0].astype(np.float64))
    other = np.linalg.norm(d) / np.linalg.norm(xf[0].astype(np.float64))
    assert abs(want - 1.0) < 1e-12 and abs(other - 0.5) < 1e-12
    assert abs(st["max_rel_l2"] - want) < 1e-12
    assert abs(st["max_rel_l2"] - other) > 0.4


def test_calibrate_bands_scale_with_delta(cal_world, tmp_path):
    w = cal_world
    pert = w.old_feats[CAL_SID].copy()
    pert[0, 0] += np.float32(3e-6)
    xd = write_x_first_run(tmp_path, CAL_SID, _x_full(pert), w.coords[CAL_SID])
    out = tmp_path / "cal.json"
    assert M.main(_cal_args(w, xd, out)) == 0
    rep = json.loads(out.read_text())
    assert not rep["all_bit_identical"]
    assert 1e-6 < rep["band_abs"] <= 1e-4 and abs(rep["band_abs"] - 10 * rep["max_abs_over_slides"]) < 1e-12
    assert rep["band_rel"] == min(1e-5, max(1e-7, 10 * rep["max_rel_l2_over_slides"]))


@pytest.mark.parametrize("bad", ["x_sha", "x_size", "code_sha256", "encoder", "amp", "schema", "slide_id",
                                 "n_tiles", "no_pickle_sha", "no_record", "x_dtype", "x_width", "x_nonfinite",
                                 "x_coords", "x_zero_row", "pub_zero_row", "pub_sha", "pub_dtype_row",
                                 "tiles_total", "not_inventory_row"])
def test_calibrate_refusals(cal_world, tmp_path, monkeypatch, bad, capsys):
    w = cal_world
    pub = w.old_feats[CAL_SID]
    coords = w.coords[CAL_SID]
    full = _x_full(pub)
    over = {}
    if bad == "code_sha256":
        over = {"code_sha256": "0" * 64}
    elif bad == "encoder":
        over = {"encoder": "virchow2_fp32"}
    elif bad == "amp":
        over = {"amp": True}
    elif bad == "schema":
        over = {"schema": "something_else"}
    elif bad == "slide_id":
        over = {"slide_id": "10002"}
    elif bad == "n_tiles":
        over = {"n_tiles": SLIDES[CAL_SID][0] + 1}
    elif bad == "no_pickle_sha":
        over = {"pickle_sha256": None}
    elif bad == "x_dtype":
        full = full.astype(np.float64)
    elif bad == "x_width":
        full = np.ascontiguousarray(full[:, :1280])
    elif bad == "x_nonfinite":
        full = full.copy()
        full[1, 5] = np.nan
    elif bad == "x_coords":
        coords = [dict(c) for c in coords]
        coords[0] = {"wsi_loc": [coords[0]["wsi_loc"][0] + 512, coords[0]["wsi_loc"][1]]}
    elif bad == "x_zero_row":
        full = full.copy()
        full[0, :1280] = 0
    xd = write_x_first_run(tmp_path, CAL_SID, full, coords, **over)
    if bad == "x_sha":
        pk = xd / f"{CAL_SID}.pkl"
        b = bytearray(pk.read_bytes())
        b[-3] ^= 1
        pk.write_bytes(bytes(b))
    elif bad == "x_size":
        pk = xd / f"{CAL_SID}.pkl"
        pk.write_bytes(pk.read_bytes() + b"\x00")
    elif bad == "no_record":
        (xd / f"{CAL_SID}.record.json").unlink()
    elif bad == "pub_zero_row":
        z = pub.copy()
        z[0] = 0
        _rewrite_bank(w, "virchow2", CAL_SID, z, w.coords[CAL_SID])
    elif bad == "pub_sha":
        p = w.banks / "virchow2" / f"{CAL_SID}.pkl"
        p.write_bytes(p.read_bytes() + b"\x00")
    elif bad == "pub_dtype_row":
        monkeypatch.setattr(M, "CALIBRATION_SLIDES", ("10002",))      # a float16 inventory row
        monkeypatch.setattr(M, "CALIBRATION_TILES", SLIDES["10002"][0])
        xd = write_x_first_run(tmp_path, "10002", _x_full(w.old_feats["10002"].astype(np.float32)),
                               w.coords["10002"], name="x_fp16")
    elif bad == "tiles_total":
        monkeypatch.setattr(M, "CALIBRATION_TILES", SLIDES[CAL_SID][0] + 1)
    elif bad == "not_inventory_row":
        monkeypatch.setattr(M, "CALIBRATION_SLIDES", (CAL_SID, "99999"))
    out = tmp_path / "cal.json"
    assert M.main(_cal_args(w, xd, out)) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("delta,want", [(3e-6, "i"), (0.05, "ii"), (5.0, "iii")])
def test_calibrate_classifies_4_0_and_flags_author_report(cal_world, tmp_path, capsys, delta, want):
    """NL2: each 4.0 slide is classified against the report's own band; any
    class other than (i) sets ``report_to_author`` and prints a warning.
    0.05 is the review P6 case (band capped at 1e-4/1e-5 -> class ii)."""
    w = cal_world
    pert = w.old_feats[CAL_SID].copy()
    pert[0, 0] += np.float32(delta)
    xd = write_x_first_run(tmp_path, CAL_SID, _x_full(pert), w.coords[CAL_SID])
    out = tmp_path / "cal.json"
    assert M.main(_cal_args(w, xd, out)) == 0
    cap = capsys.readouterr()
    rep = json.loads(out.read_text())
    st = rep["per_slide"][CAL_SID]
    assert st["class"] == want == M.classify_fp32(st, rep["band_abs"], rep["band_rel"])
    assert rep["report_to_author"] is (want != "i")
    assert rep["class_counts"][want] == 1 and sum(rep["class_counts"].values()) == 1
    assert ("report_to_author=true" in cap.err) is (want != "i")
    assert cap.out.strip() == sha256_path(out)
    if want == "ii":
        assert rep["band_abs"] == 1e-4 and rep["band_rel"] == 1e-5


@pytest.mark.parametrize("bad", ["job_record", "job_flag", "manifest_path_record", "manifest_sha_record",
                                 "manifest_sha_flag", "manifest_file_changed", "job_int", "manifest_missing_key",
                                 "flag_sha_not_hex", "empty_job"])
def test_calibrate_binds_x_records_to_first_run(cal_world, tmp_path, capsys, bad):
    """NL3: X records must carry the first run's lsf_job / manifest_path /
    manifest_sha256 given on the command line (and the manifest file must hash
    to the given sha); a later X run with the same code is refused."""
    w = cal_world
    man = x_stimuli(tmp_path)
    other = tmp_path / "other_stimuli.json"
    other.write_text("{}\n")
    over, kw = {}, {}
    if bad == "job_record":
        over = {"lsf_job": "327000000"}
    elif bad == "job_flag":
        kw = {"job": "327000000"}
    elif bad == "manifest_path_record":
        over = {"manifest_path": str(other)}
    elif bad == "manifest_sha_record":
        over = {"manifest_sha256": sha256_path(other)}
    elif bad == "manifest_sha_flag":
        kw = {"man": other, "man_sha": sha256_path(other)}
    elif bad == "job_int":
        over = {"lsf_job": int(X_FIRST_RUN_JOB)}
    elif bad == "manifest_missing_key":
        over = {"manifest_sha256": None}
    elif bad == "flag_sha_not_hex":
        kw = {"man_sha": "Z" * 64}
    elif bad == "empty_job":
        kw = {"job": ""}
    xd = write_x_first_run(tmp_path, CAL_SID, _x_full(w.old_feats[CAL_SID]), w.coords[CAL_SID], **over)
    if bad == "manifest_sha_record":
        rec = json.loads((xd / f"{CAL_SID}.record.json").read_text())
        rec["manifest_path"] = str(man)
        (xd / f"{CAL_SID}.record.json").write_text(json.dumps(rec))
    if bad == "manifest_file_changed":
        args = _cal_args(w, xd, tmp_path / "cal.json")
        man.write_text(man.read_text() + " ")
    else:
        args = _cal_args(w, xd, tmp_path / "cal.json", **kw)
    if bad == "empty_job":
        args[args.index("--x-lsf-job") + 1] = ""
    assert M.main(args) == 2
    err = capsys.readouterr().err
    assert "REFUSED [calibrate]" in err and "Traceback" not in err
    assert not (tmp_path / "cal.json").exists()


@pytest.mark.parametrize("flag", ["--x-lsf-job", "--x-manifest-path", "--x-manifest-sha256"])
def test_calibrate_first_run_flags_required(cal_world, tmp_path, flag):
    w = cal_world
    xd = write_x_first_run(tmp_path, CAL_SID, _x_full(w.old_feats[CAL_SID]), w.coords[CAL_SID])
    args = _cal_args(w, xd, tmp_path / "cal.json")
    i = args.index(flag)
    del args[i:i + 2]
    with pytest.raises(SystemExit):
        M.main(args)
    assert not (tmp_path / "cal.json").exists()


@pytest.mark.parametrize("bad", ["bytes_str", "bytes_bool", "bytes_negative", "bytes_float", "sha_upper",
                                 "sha_short", "sha_int", "not_json", "not_utf8", "json_list", "json_null"])
def test_calibrate_malformed_x_record_is_clean_refusal(cal_world, tmp_path, capsys, bad):
    """NL4: malformed X records (review P6 ``pickle_bytes: "x"`` and kin) give
    the CLI's refusal rc 2 with a specific message, never a traceback."""
    w = cal_world
    over = {"bytes_str": {"pickle_bytes": "x"}, "bytes_bool": {"pickle_bytes": True},
            "bytes_negative": {"pickle_bytes": -1}, "bytes_float": {"pickle_bytes": 10.0},
            "sha_upper": {"pickle_sha256": "A" * 64}, "sha_short": {"pickle_sha256": "a" * 63},
            "sha_int": {"pickle_sha256": 7}}.get(bad, {})
    xd = write_x_first_run(tmp_path, CAL_SID, _x_full(w.old_feats[CAL_SID]), w.coords[CAL_SID], **over)
    rp = xd / f"{CAL_SID}.record.json"
    if bad == "not_json":
        rp.write_text("{not json")
    elif bad == "not_utf8":
        rp.write_bytes(b"\xff\xfe{}")
    elif bad == "json_list":
        rp.write_text("[1, 2]")
    elif bad == "json_null":
        rp.write_text("null")
    out = tmp_path / "cal.json"
    assert M.main(_cal_args(w, xd, out)) == M.RC_REFUSED == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err and "REFUSED [calibrate]" in err
    want = {"bytes": "pickle_bytes", "sha": "pickle_sha256", "not": "not valid UTF-8 JSON",
            "json": "not an object"}[bad.split("_")[0]]
    assert want in err, err
    assert not out.exists()


def test_calibrate_malformed_record_via_subprocess_exit_code(cal_world, tmp_path):
    """NL4 end-to-end: the console process exits 2 with no traceback."""
    w = cal_world
    xd = write_x_first_run(tmp_path, CAL_SID, _x_full(w.old_feats[CAL_SID]), w.coords[CAL_SID],
                           pickle_bytes="x")
    env = {**os.environ, "PYTHONPATH": str(CODE_ROOT), "CUDA_VISIBLE_DEVICES": ""}
    code = ("import sys, prosper.evaluation.r2_1_virchow2_fp32_extract_cli as M; "
            f"M.CALIBRATION_SLIDES = ({CAL_SID!r},); M.CALIBRATION_TILES = {SLIDES[CAL_SID][0]}; "
            "sys.exit(M.main(sys.argv[1:]))")
    r = subprocess.run([PY, "-c", code, *_cal_args(w, xd, tmp_path / "cal.json")], env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 2, r.stderr
    assert "Traceback" not in r.stderr and "pickle_bytes" in r.stderr


def _write_cal(tmp_path: Path, band_abs: float, band_rel: float, **over) -> Tuple[Path, str]:
    p = tmp_path / "cal_manual.json"
    obj = {"schema": M.CALIBRATE_SCHEMA, "step": "4.0", "band_abs": band_abs, "band_rel": band_rel}
    obj.update(over)
    p.write_text(json.dumps(obj))
    return p, sha256_path(p)


def _replace_new_pickle(world: World, sid: str, feats: np.ndarray) -> None:
    pkl, recp, _ = M.output_paths(world.out_root, sid)
    coords = pickle.loads(pkl.read_bytes())[1]
    payload = pickle.dumps((np.ascontiguousarray(feats.astype(np.float32)), coords), protocol=4)
    pkl.write_bytes(payload)
    rec = json.loads(recp.read_text())
    rec.update(pickle_sha256=sha256_bytes(payload), pickle_bytes=len(payload))
    recp.write_text(json.dumps(rec))


def test_fidelity_classes_i_ii_iii_and_fp16_descriptives(world, tmp_path):
    assert run(world, base_args(world)) == 0
    fp32_sid = next(s for s in SLIDES if SLIDES[s][1] == "float32")
    old = world.old_feats[fp32_sid]
    cal, cal_sha = _write_cal(tmp_path, 1e-6, 1e-7)
    fdir = world.out_root / "features" / M.ENCODER
    args = ["fidelity", "--inventory", str(world.inventory), "--inventory-sha256", world.inventory_sha,
            "--new-features-dir", str(fdir), "--calibration-report", str(cal),
            "--calibration-report-sha256", cal_sha]
    # class (i): bit-identical to the old fp32 bank
    _replace_new_pickle(world, fp32_sid, old)
    out1 = tmp_path / "fid1.json"
    assert M.main(args + ["--out", str(out1)]) == 0
    rep = json.loads(out1.read_text())
    assert rep["fp32"]["n"] == 1 and rep["fp32"]["counts"] == {"i": 1, "ii": 0, "iii": 0}
    s = rep["fp32"]["per_slide"][fp32_sid]
    assert s["class"] == "i" and s["bit_identical"] and s["rows_out_of_band"] == 0
    assert rep["fp16"]["n"] == 3
    for sid, st in rep["fp16"]["per_slide"].items():
        assert set(st) >= {"ulp_histogram_elements", "min_cosine", "median_rel_l2", "frac_rows_le_2ulp",
                           "per_dim_mean_signed_diff_summary", "rows_out_of_band"}
        assert sum(st["ulp_histogram_elements"].values()) == SLIDES[sid][0] * 1280
    assert len(rep["fp16"]["pooled_per_dim_mean_signed_diff"]) == 1280
    assert rep["fp16"]["bimodality_log10_median_rel_l2"]["n"] == 3
    assert rep["calibration_report_sha256"] == sha256_path(cal) == cal_sha
    # class (ii): small relative perturbation (rel ~1e-4, cosine ~1)
    _replace_new_pickle(world, fp32_sid, old * np.float32(1 + 1e-4))
    out2 = tmp_path / "fid2.json"
    assert M.main(args + ["--slides", fp32_sid, "--out", str(out2)]) == 0
    s = json.loads(out2.read_text())["fp32"]["per_slide"][fp32_sid]
    assert s["class"] == "ii" and s["rows_out_of_band"] == SLIDES[fp32_sid][0]
    # class (iii): sign flip on one row
    bad = old.copy()
    bad[0] = -bad[0]
    _replace_new_pickle(world, fp32_sid, bad)
    out3 = tmp_path / "fid3.json"
    assert M.main(args + ["--slides", fp32_sid, "--out", str(out3)]) == 0
    s = json.loads(out3.read_text())["fp32"]["per_slide"][fp32_sid]
    assert s["class"] == "iii" and s["rows_out_of_class_ii"] == 1 and s["min_cosine"] < 0
    # tampered new pickle (sha != record) -> refuse
    pkl = fdir / f"{fp32_sid}.pkl"
    pkl.write_bytes(pkl.read_bytes() + b"\x00")
    assert M.main(args + ["--slides", fp32_sid, "--out", str(tmp_path / "fid4.json")]) == 2


def test_fidelity_calibration_report_gate_and_production_records_only(world, tmp_path, capsys):
    """L7: the calibration report is sha-pinned and must be a step-4.0 calibrate
    report; pilot-mode records are refused."""
    assert run(world, base_args(world)) == 0
    fdir = world.out_root / "features" / M.ENCODER

    def fid(cal, sha, out, fd=fdir):
        return M.main(["fidelity", "--inventory", str(world.inventory), "--inventory-sha256", world.inventory_sha,
                       "--new-features-dir", str(fd), "--calibration-report", str(cal),
                       "--calibration-report-sha256", sha, "--out", str(out)])

    cal, cal_sha = _write_cal(tmp_path, 1e-6, 1e-7)
    assert fid(cal, cal_sha, tmp_path / "ok.json") == 0
    with pytest.raises(SystemExit):                            # sha flag is required
        M.main(["fidelity", "--inventory", str(world.inventory), "--inventory-sha256", world.inventory_sha,
                "--new-features-dir", str(fdir), "--calibration-report", str(cal), "--out", str(tmp_path / "x.json")])
    assert fid(cal, "0" * 64, tmp_path / "f1.json") == 2
    assert "[calibration_report]" in capsys.readouterr().err
    for over in ({"step": "4.1"}, {"schema": "other"}, {"step": None}):
        c2, s2 = _write_cal(tmp_path, 1e-6, 1e-7, **over)
        assert fid(c2, s2, tmp_path / "f2.json") == 2, over
        assert "[calibration_report]" in capsys.readouterr().err
    c3 = tmp_path / "cal_nostep.json"
    c3.write_text(json.dumps({"schema": M.CALIBRATE_SCHEMA, "band_abs": 1e-6, "band_rel": 1e-7}))
    assert fid(c3, sha256_path(c3), tmp_path / "f3.json") == 2
    # pilot-mode records refused
    pr = world.run_root / "pilot" / "hostA"
    assert run(world, base_args(world, out_root=pr, pilot=True)) == 0
    cal, cal_sha = _write_cal(tmp_path, 1e-6, 1e-7)
    assert fid(cal, cal_sha, tmp_path / "f4.json", fd=pr / "features" / M.ENCODER) == 2
    assert "allowlist_mode" in capsys.readouterr().err
    for n in ("f1", "f2", "f3", "f4"):
        assert not (tmp_path / f"{n}.json").exists()


def test_fidelity_class_boundaries_direct():
    base = {"bit_identical": False, "max_abs": 0.0, "max_rel_l2": 0.0, "min_cosine": 1.0}
    assert M.classify_fp32({**base, "max_abs": 1e-6, "max_rel_l2": 1e-7}, 1e-6, 1e-7) == "i"
    assert M.classify_fp32({**base, "max_abs": 2e-6, "max_rel_l2": 1e-7}, 1e-6, 1e-7) == "ii"
    assert M.classify_fp32({**base, "max_abs": 1.0, "max_rel_l2": 1e-2, "min_cosine": 0.9999}, 1e-6, 1e-7) == "ii"
    assert M.classify_fp32({**base, "max_abs": 1.0, "max_rel_l2": 1.1e-2, "min_cosine": 0.9999}, 1e-6, 1e-7) == "iii"
    assert M.classify_fp32({**base, "max_abs": 1.0, "max_rel_l2": 1e-3, "min_cosine": 0.99989}, 1e-6, 1e-7) == "iii"
    assert M.classify_fp32({**base, "bit_identical": True, "max_abs": 5.0, "max_rel_l2": 5.0}, 1e-6, 1e-7) == "i"


def test_bimodality_summary_separates_two_groups():
    v = np.concatenate([np.full(10, -6.0) + np.linspace(0, 0.1, 10), np.full(5, -2.0) + np.linspace(0, 0.1, 5)])
    s = M.bimodality_summary(v)
    assert s["n_low"] == 10 and s["n_high"] == 5 and s["between_variance_fraction"] > 0.99
    assert -6.0 < s["largest_gap_at"] < -2.0


# --------------------------------------------------------------------------
# check-mosaic (synthetic only)
# --------------------------------------------------------------------------


def test_check_mosaic_synthetic(world, tmp_path):
    assert run(world, base_args(world)) == 0
    fdir = world.out_root / "features" / M.ENCODER
    mosaic = [(f"{sid}.svs", [dict(wsi_loc=list(c["wsi_loc"])) for c in world.coords[sid]]) for sid in sorted(SLIDES)]
    mp = tmp_path / "mosaic.pkl"
    data = pickle.dumps(mosaic, protocol=4)
    mp.write_bytes(data)
    base = ["check-mosaic", "--mosaic", str(mp), "--mosaic-sha256", sha256_bytes(data),
            "--mosaic-bytes", str(len(data)), "--features-dir", str(fdir)]
    assert M.main(base + ["--out", str(tmp_path / "cm1.json")]) == 0
    assert json.loads((tmp_path / "cm1.json").read_text())["all_equal"]
    # wrong sha -> refused; the default expected sha is the real mosaic pin
    assert M.main(["check-mosaic", "--mosaic", str(mp), "--features-dir", str(fdir), "--out", str(tmp_path / "cm2.json")]) == 2
    mosaic[0][1].reverse()
    data2 = pickle.dumps(mosaic, protocol=4)
    mp.write_bytes(data2)
    assert M.main(["check-mosaic", "--mosaic", str(mp), "--mosaic-sha256", sha256_bytes(data2), "--mosaic-bytes",
                   str(len(data2)), "--features-dir", str(fdir), "--out", str(tmp_path / "cm3.json")]) == 1
    rep = json.loads((tmp_path / "cm3.json").read_text())
    assert rep["n_equal"] == 3 and rep["per_slide"][sorted(SLIDES)[0]]["status"] == "mismatch"


def test_main_usage_and_unknown_subcommand(capsys):
    assert M.main([]) == 2
    assert M.main(["frobnicate"]) == 2


# --------------------------------------------------------------------------
# Marked real test (skipped unless R21_V2_REAL=1)
# --------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("R21_V2_REAL") != "1", reason="real-input test; set R21_V2_REAL=1")
def test_real_three_slide_equivalence_with_x_identity():
    """A-F6 / B-N5 real check on slides 15678, 10002, 16839 (analyst-run only).

    Required environment (all paths absolute; nothing is downloaded):
      R21_V2_INVENTORY, R21_V2_INVENTORY_SHA256   inventory.csv (sha 9449de48...) and its sha
      R21_V2_ALLOWLIST, R21_V2_ALLOWLIST_SHA256   allowlist JSON containing the observed tuple
      R21_V2_CODE_MANIFEST, R21_V2_CODE_MANIFEST_SHA256
      R21_V2_RUN_ROOT                               run_001 root; this test writes to
                                                    <run_root>/pilot/<R21_V2_HOST_TAG>
      R21_V2_HOST_TAG                               pilot host tag (fresh, must not exist)
      R21_V2_X_FEATURES_DIR                         X first-run features/virchow2 dir holding
                                                    10002 and 16839 pickles + records
      R21_V2_X_OUT_ROOT                             fresh X run root for slide 15678
      R21_V2_X_MANIFEST (sha 014513af...), R21_V2_X_QUERIES_ONE_ROW + R21_V2_X_QUERIES_SHA256
      (one-row CSV in X's queries schema for 15678 and its sha), R21_V2_X_TEMPLATE,
      R21_V2_X_TORCHSTAIN_PREFIX
      PROSPER_OUT                                   set; R21_V2_RUN_ROOT must be
                                                    realpath($PROSPER_OUT)/r2_1_virchow2_fp32_20261006/run_001 (M1)
      LSB_DJOB_NUMPROC >= 3 (or R21_V2_NUM_WORKERS >= 2): the check runs with >= 2 loader workers
      CUDA device visible; OMP/MKL_NUM_THREADS=1; no NVIDIA_TF32_OVERRIDE/CUBLAS_WORKSPACE_CONFIG.
      HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 exported BEFORE pytest starts (NM1: the
      extractor refuses otherwise; huggingface_hub freezes its flag at import).
    Runs `extract --pilot-observe-tuple` on the three slides, then `X run` for
    15678 (identity included), then (L6) asserts every new record has
    ``observed_tuple == SEED_TUPLE`` and ``num_workers >= 2``, size+sha-checks
    every X pickle against its X record before reading it, asserts
    np.array_equal and tobytes equality (np.ascontiguousarray) of new feats vs X
    identity[:, :1280] and equal coords, and writes a small JSON receipt
    (pickle/record shas + observed tuple) at
    ``<run_root>/pilot/<host_tag>/A_F6_receipt.json`` (no-replace).
    """
    env = {k: os.environ[k] for k in (
        "R21_V2_INVENTORY", "R21_V2_INVENTORY_SHA256", "R21_V2_ALLOWLIST", "R21_V2_ALLOWLIST_SHA256",
        "R21_V2_CODE_MANIFEST", "R21_V2_CODE_MANIFEST_SHA256", "R21_V2_RUN_ROOT", "R21_V2_HOST_TAG",
        "R21_V2_X_FEATURES_DIR", "R21_V2_X_OUT_ROOT", "R21_V2_X_MANIFEST", "R21_V2_X_QUERIES_ONE_ROW",
        "R21_V2_X_TEMPLATE", "R21_V2_X_TORCHSTAIN_PREFIX", "R21_V2_X_QUERIES_SHA256")}
    slides = ["15678", "10002", "16839"]
    out_root = Path(env["R21_V2_RUN_ROOT"]) / "pilot" / env["R21_V2_HOST_TAG"]
    rows = [r for r in csv.DictReader(Path(env["R21_V2_INVENTORY"]).read_text().splitlines()) if r["bank"] == "virchow2"]
    ids = [r["slide_id"] for r in rows]
    lo, hi = min(ids.index(s) for s in slides) + 1, max(ids.index(s) for s in slides) + 1
    nw = os.environ.get("R21_V2_NUM_WORKERS")
    nw_args = ["--num-workers", nw] if nw else []
    rc = M.run_extract([*nw_args, "--inventory", env["R21_V2_INVENTORY"], "--inventory-sha256", env["R21_V2_INVENTORY_SHA256"],
                        "--weights-sha256", M.WEIGHTS_SHA256, "--allowlist", env["R21_V2_ALLOWLIST"],
                        "--allowlist-sha256", env["R21_V2_ALLOWLIST_SHA256"], "--code-manifest", env["R21_V2_CODE_MANIFEST"],
                        "--code-manifest-sha256", env["R21_V2_CODE_MANIFEST_SHA256"], "--code-root", str(CODE_ROOT),
                        "--run-root", env["R21_V2_RUN_ROOT"], "--out-root", str(out_root), "--lo", str(lo), "--hi", str(hi),
                        "--slides", *slides, "--pilot-observe-tuple"])
    assert rc == 0
    rc = X.run(["--manifest", env["R21_V2_X_MANIFEST"], "--manifest-sha256",
                "014513af5920ed621cfad2dc5cf527747936bb21c1db8336d282a187d669d6ce",
                "--queries", env["R21_V2_X_QUERIES_ONE_ROW"], "--queries-sha256", env["R21_V2_X_QUERIES_SHA256"],
                "--encoders", "virchow2", "--no-amp", "virchow2",
                "--template", env["R21_V2_X_TEMPLATE"], "--torchstain-prefix", env["R21_V2_X_TORCHSTAIN_PREFIX"],
                "--out-root", env["R21_V2_X_OUT_ROOT"]])
    assert rc == 0
    xdirs = {"15678": Path(env["R21_V2_X_OUT_ROOT"]) / "features" / "virchow2",
             "10002": Path(env["R21_V2_X_FEATURES_DIR"]), "16839": Path(env["R21_V2_X_FEATURES_DIR"])}
    receipt = {"kind": "r2_1_virchow2_fp32 A-F6 three-slide equivalence", "slides": {},
               "seed_tuple": M.SEED_TUPLE, "x_sha256_pinned": M.X_SHA256_PINNED}
    observed = None
    for sid in slides:
        pkl, recp, _ = M.output_paths(out_root, sid)
        rec = json.loads(recp.read_text())
        assert rec["observed_tuple"] == M.SEED_TUPLE, rec["observed_tuple"]
        assert rec["num_workers"] >= 2, rec["num_workers"]
        assert rec["allowlist_mode"] == "pilot"
        observed = rec["observed_tuple"]
        new_bytes = pkl.read_bytes()
        assert len(new_bytes) == rec["pickle_bytes"] and sha256_bytes(new_bytes) == rec["pickle_sha256"]
        new, ncoords = pickle.loads(new_bytes)
        xpk, xrp = xdirs[sid] / f"{sid}.pkl", xdirs[sid] / f"{sid}.record.json"
        xrec = json.loads(xrp.read_text())
        xbytes = xpk.read_bytes()
        assert len(xbytes) == xrec["pickle_bytes"] and sha256_bytes(xbytes) == xrec["pickle_sha256"]
        assert xrec["code_sha256"] == M.X_SHA256_PINNED and xrec["encoder"] == "virchow2" and xrec["amp"] is False
        xobj = pickle.loads(xbytes)
        xi = np.ascontiguousarray(np.asarray(xobj[0]["identity"])[:, :M.WIDTH])
        assert np.array_equal(new, xi)
        assert np.ascontiguousarray(new).tobytes() == xi.tobytes()
        assert M.locs_of(ncoords) == M.locs_of(xobj[1])
        receipt["slides"][sid] = {
            "new_pickle": str(pkl), "new_pickle_sha256": rec["pickle_sha256"],
            "new_record_sha256": sha256_path(recp), "x_pickle": str(xpk), "x_pickle_sha256": xrec["pickle_sha256"],
            "x_record_sha256": sha256_path(xrp), "n_tiles": int(new.shape[0]), "num_workers": rec["num_workers"],
            "bytes_equal": True, "coords_equal": True}
    receipt["observed_tuple"] = observed
    X._publish_bytes(out_root / "A_F6_receipt.json",
                     (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
