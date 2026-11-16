"""Fast CPU-only tests for the R2.1 perturbation extractor (no model download)."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import functional as F

from prosper.evaluation import r2_1_perturb_extract_cli as M

TEMPLATE = "ab" * 32
DIM = 6


class FakeOpenSlideError(Exception):
    pass


try:  # the real exception type when available
    from openslide import OpenSlideError as _OSE
except Exception:  # pragma: no cover
    _OSE = FakeOpenSlideError


class FakeSlide:
    """Deterministic pixels that depend on absolute coordinates."""

    calls: list = []

    def __init__(self, path, fail_at=()):
        self.path = path
        self.fail_at = set(fail_at)

    def read_region(self, loc, level, size):
        FakeSlide.calls.append((self.path, tuple(loc), level, tuple(size)))
        if tuple(loc) in self.fail_at:
            raise _OSE(f"cannot read {loc}")
        x, y = loc
        w, h = size
        yy, xx = np.mgrid[0:h, 0:w]
        r = ((xx + x) * 7 + (yy + y) * 3) % 256
        g = ((xx + x) * 2 + (yy + y) * 11) % 256
        b = ((xx + x) * 5 + (yy + y) * 1 + 40) % 256
        arr = np.stack([r, g, b, np.full_like(r, 255)], -1).astype(np.uint8)
        return Image.fromarray(arr, "RGBA")

    def close(self):
        pass


def make_opener(fail_at=()):
    def opener(path):
        return FakeSlide(path, fail_at)
    return opener


def fake_reinhard(img):
    return Image.eval(img, lambda v: 255 - v), True


_W = torch.Generator().manual_seed(0)
_PROJ = torch.randn(3 * 8 * 8, DIM, generator=_W)


def fake_transform(img):
    return F.normalize(F.to_tensor(F.resize(img, [8, 8])), [0.5] * 3, [0.25] * 3)


def fake_factory(name, device, amp):
    model = torch.nn.Identity()
    return M.EncoderRuntime(name, model, fake_transform,
                            lambda x: x.flatten(1) @ _PROJ, amp, DIM,
                            {"recipe": "fake", "transform": "fake"})


def stimuli_for(sid, brightness=0.9, shift=(-5, 7)):
    p = {
        "identity": {"treatment": "untreated"},
        "rotate_90": {"angle": 90}, "rotate_180": {"angle": 180}, "rotate_270": {"angle": 270},
        "flip_horizontal": {"axis": "horizontal"}, "flip_vertical": {"axis": "vertical"},
        "brightness": {"factor": brightness}, "saturation": {"factor": 1.1},
        "contrast": {"factor": 0.95}, "spatial_shift": {"x_shift": shift[0], "y_shift": shift[1]},
        "reinhard": {"template_sha256": TEMPLATE},
        "gaussian_blur": {"application_stage": "post_preprocessing", "kernel_size": 3,
                          "sigma": 0.5},
    }
    return [{"condition": c, "level": 0, "parameters": p[c], "patch_size": 512,
             "slide_id": sid} for c in M.CONDITIONS]


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("PROSPER_OUT", str(tmp_path / "out"))
    slides = {"S1": 3, "S2": 2}
    records = stimuli_for("S1", 0.9, (-5, 7)) + stimuli_for("S2", 1.12, (3, -2))
    manifest = tmp_path / "stimuli.json"
    manifest.write_text(json.dumps({"policy": {}, "records": records}))
    rows = []
    for i, (sid, n) in enumerate(slides.items()):
        coords = [{"wsi_loc": [1000 + 600 * k, 2000 + 50 * k]} for k in range(n)]
        bank = tmp_path / "bank" / f"{sid}.pkl"
        bank.parent.mkdir(exist_ok=True)
        data = pickle.dumps((np.zeros((n, 4), np.float32), coords), protocol=4)
        bank.write_bytes(data)
        rows.append({"population_index": i, "metadata_index": i + 1, "slide_id": sid,
                     "patient_id": f"P{i}", "diagnostic_group": "X", "feature_rows": n,
                     "feature_path": str(bank), "resolved_path": str(bank),
                     "feature_size": len(data),
                     "feature_sha256": hashlib.sha256(data).hexdigest()})
    queries = tmp_path / "queries.csv"
    with open(queries, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"tmp": tmp_path, "manifest": manifest, "queries": queries, "rows": rows,
            "out_root": tmp_path / "run", "records": records}


def _argv(world, *extra):
    return ["--manifest", str(world["manifest"]), "--manifest-sha256", "",
            "--queries", str(world["queries"]), "--queries-sha256", "",
            "--out-root", str(world["out_root"]), "--device", "cpu",
            "--num-workers", "0", "--forward-batch", "2", "--loader-batch", "1",
            "--encoders", "h_optimus_1", "--no-hash-weights", *extra]


def _run(world, *extra, opener=None):
    return M.run(_argv(world, *extra), encoder_factory=fake_factory, reinhard=fake_reinhard,
                 reinhard_info={"fake": True}, opener=opener or make_opener(),
                 slide_path_of=lambda sid: Path(f"/fake/{sid}.svs"))


def _stim(world, sid):
    return {r["condition"]: r for r in world["records"] if r["slide_id"] == sid}


# (i) manifest parameters used verbatim -------------------------------------


def test_manifest_factor_and_shift_used_verbatim(world):
    st = _stim(world, "S1")
    region = FakeSlide("x").read_region((0, 0), 0, (64, 64)).convert("RGB")
    imgs, _ = M.condition_images(region, region, st, fake_reinhard)
    assert np.array_equal(np.asarray(imgs["brightness"]),
                          np.asarray(F.adjust_brightness(region, 0.9)))
    assert not np.array_equal(np.asarray(imgs["brightness"]),
                              np.asarray(F.adjust_brightness(region, 1.1)))
    assert np.array_equal(np.asarray(imgs["contrast"]),
                          np.asarray(F.adjust_contrast(region, 0.95)))
    assert np.array_equal(np.asarray(imgs["rotate_90"]),
                          np.asarray(region.rotate(90, expand=True)))
    assert np.array_equal(np.asarray(imgs["flip_horizontal"]),
                          np.asarray(region.transpose(Image.FLIP_LEFT_RIGHT)))

    FakeSlide.calls = []
    ds = M.SlideTiles("/fake/S1.svs", [{"wsi_loc": [1000, 2000]}], st,
                      {"e": fake_transform}, fake_reinhard, make_opener())
    item = ds[0]
    locs = [c[1] for c in FakeSlide.calls]
    assert locs == [(1000, 2000), (995, 2007)]
    assert all(c[2] == 0 and c[3] == (512, 512) for c in FakeSlide.calls)
    assert item["tensors"]["e"].shape == (12, 3, 8, 8)


def test_blur_is_post_preprocessing_and_matches_torchvision_class(world):
    from torchvision.transforms import GaussianBlur
    st = _stim(world, "S1")
    region = FakeSlide("x").read_region((5, 9), 0, (64, 64)).convert("RGB")
    imgs, _ = M.condition_images(region, region, st, fake_reinhard)
    t = M.encode_conditions(imgs, fake_transform, st)
    base = fake_transform(region)
    assert torch.equal(t[0], base)
    assert torch.equal(t[M.CONDITIONS.index("gaussian_blur")],
                       GaussianBlur(kernel_size=3, sigma=0.5)(base))


def test_manifest_validation_rejects_bad_records(world, tmp_path):
    recs = stimuli_for("S9")
    recs[6]["parameters"] = {"factor": 1}            # int, not float
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"records": recs}))
    with pytest.raises(M.InputIntegrityError):
        M.load_manifest(p, None)
    recs = stimuli_for("S9")[:-1]                    # missing a condition
    p.write_text(json.dumps({"records": recs}))
    with pytest.raises(M.InputIntegrityError):
        M.load_manifest(p, None)
    with pytest.raises(M.InputIntegrityError):
        M.load_manifest(world["manifest"], "0" * 64)


# (ii)+(iii) shapes, differences, output layout and record --------------------


def test_end_to_end_layout_and_record(world):
    assert _run(world) == 0
    for row in world["rows"]:
        sid, n = row["slide_id"], int(row["feature_rows"])
        pkl, rec = M.output_paths(world["out_root"], "h_optimus_1", sid)
        feats, coords = pickle.loads(pkl.read_bytes())
        assert list(feats) == list(M.CONDITIONS)
        for c, a in feats.items():
            assert a.shape == (n, DIM) and a.dtype == np.float32
        for c in ("rotate_90", "rotate_180", "rotate_270", "flip_horizontal",
                  "flip_vertical", "brightness", "spatial_shift", "reinhard", "gaussian_blur"):
            assert not np.allclose(feats[c], feats["identity"]), c
        bank_coords = pickle.loads(Path(row["resolved_path"]).read_bytes())[1]
        assert coords == bank_coords
        r = json.loads(rec.read_text())
        assert r["slide_id"] == sid and r["encoder"] == "h_optimus_1" and r["n_tiles"] == n
        assert r["stimuli"] == [s for s in world["records"] if s["slide_id"] == sid]
        assert r["pickle_sha256"] == hashlib.sha256(pkl.read_bytes()).hexdigest()
        assert r["manifest_sha256"] == hashlib.sha256(world["manifest"].read_bytes()).hexdigest()
        assert r["amp"] is True
        for k in ("versions", "host", "gpu", "slide_wall_seconds", "tiles_per_second_slide",
                  "encoder_meta", "reinhard", "lsf_job"):
            assert k in r
    assert not (world["out_root"] / "failures.jsonl").exists()
    assert len(list((world["out_root"] / "runs").glob("*.json"))) == 1


def test_identity_matches_plain_extraction(world):
    """Identity arm == transform(tile) through the same forward (main-bank recipe)."""
    _run(world)
    feats, coords = pickle.loads(M.output_paths(world["out_root"], "h_optimus_1", "S1")[0]
                                 .read_bytes())
    s = FakeSlide("x")
    x = torch.stack([fake_transform(s.read_region(tuple(c["wsi_loc"]), 0, (512, 512))
                                    .convert("RGB")) for c in coords])
    np.testing.assert_array_equal(feats["identity"], (x.flatten(1) @ _PROJ).numpy())


def test_dataloader_workers_and_amp_flags(world):
    argv = _argv(world, "--slides", "S2", "--no-amp", "h_optimus_1")
    argv[argv.index("--num-workers") + 1] = "2"
    rc = M.run(argv, encoder_factory=fake_factory, reinhard=fake_reinhard, reinhard_info={},
               opener=make_opener(), slide_path_of=lambda sid: Path(f"/fake/{sid}.svs"))
    assert rc == 0
    rec = json.loads(M.output_paths(world["out_root"], "h_optimus_1", "S2")[1].read_text())
    assert rec["amp"] is False
    assert not M.output_paths(world["out_root"], "h_optimus_1", "S1")[0].exists()
    assert M.resolve_amp(["h_optimus_1", "uni2", "virchow2"], [], []) == {
        "h_optimus_1": True, "uni2": False, "virchow2": False}


# (iv) resume / never overwrite / no partial pickle --------------------------


def test_resume_skips_and_never_overwrites(world):
    assert _run(world) == 0
    pkl, rec = M.output_paths(world["out_root"], "h_optimus_1", "S1")
    before = (pkl.stat().st_mtime_ns, rec.stat().st_mtime_ns)
    assert _run(world) == 0
    runs = sorted((world["out_root"] / "runs").glob("*.json"))
    summary = json.loads(runs[-1].read_text())
    assert summary["counts"]["h_optimus_1"] == {"written": 0, "skipped": 2, "failed": 0}
    assert (pkl.stat().st_mtime_ns, rec.stat().st_mtime_ns) == before

    # A tampered output is reported, not overwritten.
    pkl.write_bytes(pkl.read_bytes() + b"x")
    tampered = pkl.read_bytes()
    assert _run(world) == 1
    assert pkl.read_bytes() == tampered
    fails = [json.loads(l) for l in (world["out_root"] / "failures.jsonl").read_text().splitlines()]
    assert fails[-1]["stage"] == "existing_output" and fails[-1]["slide_id"] == "S1"


def test_tile_failure_recorded_without_partial_output(world):
    bad = (1600, 2050)                       # S1 tile 1 (S2 shares it -> both fail)
    assert _run(world, "--slides", "S1", opener=make_opener([bad])) == 1
    pkl, rec = M.output_paths(world["out_root"], "h_optimus_1", "S1")
    assert not pkl.exists() and not rec.exists()
    leftovers = list((world["out_root"]).rglob("*.tmp"))
    assert leftovers == []
    fails = [json.loads(l) for l in (world["out_root"] / "failures.jsonl").read_text().splitlines()]
    assert len(fails) == 1
    f = fails[0]
    assert f["stage"] == "tile_read" and f["slide_id"] == "S1" and f["n_failed_tiles"] == 1
    assert f["tile_errors"][0]["index"] == 1 and f["tile_errors"][0]["wsi_loc"] == list(bad)
    # A later clean run fills it in.
    assert _run(world, "--slides", "S1") == 0
    assert pkl.exists() and rec.exists()


def test_publish_refuses_overwrite(tmp_path):
    d = tmp_path / "x.bin"
    M._publish_bytes(d, b"a")
    with pytest.raises(FileExistsError):
        M._publish_bytes(d, b"b")
    assert d.read_bytes() == b"a" and list(tmp_path.glob(".*tmp")) == []


def test_limit_tiles_refused_for_production_root(world, monkeypatch):
    argv = _argv(world, "--limit-tiles", "1")
    i = argv.index("--out-root")
    del argv[i:i + 2]
    with pytest.raises(SystemExit):
        M.run(argv, encoder_factory=fake_factory, reinhard=fake_reinhard,
              opener=make_opener(), slide_path_of=lambda sid: Path("/fake"))


# (v) sha256 verification of the saved bank ----------------------------------


def test_tampered_bank_pickle_rejected(world):
    row = world["rows"][0]
    p = Path(row["resolved_path"])
    original = p.read_bytes()
    feats, coords = pickle.loads(original)
    feats = feats.copy()
    feats[0, 0] += 1.0          # same-size tamper: only sha256 can detect it
    data = pickle.dumps((feats, coords), protocol=4)
    assert len(data) == len(original) and data != original
    p.write_bytes(data)
    with pytest.raises(M.InputIntegrityError):
        M.load_bank_coords(row)
    assert _run(world, "--slides", "S1") == 1
    fails = [json.loads(l) for l in (world["out_root"] / "failures.jsonl").read_text().splitlines()]
    assert fails[-1]["stage"] == "bank_coords" and "sha256" in fails[-1]["reason"]
    assert not M.output_paths(world["out_root"], "h_optimus_1", "S1")[0].exists()


# Reinhard: harness-equivalent bytes (only if the pinned torchstain is present)


def test_reinhard_matches_library_normalize():
    import os
    out = Path(os.environ.get("PROSPER_OUT", "/PROSPER_WORKSPACE/"
                              "prosper_revision/prosper_revision_output"))
    prefix = out / M.TORCHSTAIN_SUBDIR
    if not prefix.exists() or not M.DEFAULT_TEMPLATE.exists():
        pytest.skip("pinned torchstain prefix/template not available")
    rh = M.TorchstainReinhard(M.DEFAULT_TEMPLATE, M.TEMPLATE_SHA256, prefix)
    from torchvision import transforms
    T = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)])
    rng = np.random.default_rng(0)
    for _ in range(2):
        img = Image.fromarray(rng.integers(0, 256, (48, 40, 3), dtype=np.uint8))
        ours, finite = rh(img)
        ref = transforms.ToPILImage()(rh.normalizer.normalize(I=T(img)).permute(2, 0, 1))
        assert finite and np.array_equal(np.asarray(ours), np.asarray(ref))
    flat = Image.new("RGB", (16, 16), (200, 120, 180))
    _, finite = rh(flat)
    assert finite is False
