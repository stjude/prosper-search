"""Synthetic-fixture tests for the R2.1 perturbation suite (PS-1, [decision record] candidate 002).

No GPU, no real WSI/manifest/record reads, no encoder download: tiny fake slides are served
by an injected opener, Reinhard by an injected object carrying the pinned float32 fit, the
encoder by a fixed random projection, and first-run records by ``X.process_slide`` itself
(or synthetic JSON) under ``tmp_path``.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv
import hashlib
import io
import json
import math
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import functional as F

from prosper.evaluation import r2_1_perturb_extract_cli as X
from prosper.evaluation import r2_1_perturb_suite_cli as S

REPO = Path(__file__).resolve().parents[1]
X_SHA256 = "14eb457ebfe2243a924bb7928483538985120b6987de51eb4a6cc6ccf3a8d0bb"
GOLDEN = "3bb4aa71d9861b91fb661b40f1a0892beb55872d9cf96b2c50a67a10ae15f94f"   # specification [A4]
CONTRACT_BLUR_PARAMETERS = {
    "application_stage": "image_rgb_512", "kernel_size": 7, "sigma": 1.1428571428571428,
    "padding": "reflect_101", "arithmetic": "float64_separable_explicit_order",
    "rounding": "rint_half_even_clip_uint8",
    "weights_hex": ["0x1.6d67623129833p-7", "0x1.35b4c9bd6812dp-4",
                    "0x1.e84adbabd1817p-3", "0x1.6604495267dd0p-2", "0x1.e84adbabd1817p-3",
                    "0x1.35b4c9bd6812dp-4", "0x1.6d67623129833p-7"]}
PINNED_MEANS = [67.08672332763672, 23.116905212402344, -15.08007526397705]
PINNED_STDS = [18.064807891845703, 10.140364646911621, 11.612739562988281]
DIRECT_MEANS = [67.08672332763672, 23.11690330505371, -15.080077171325684]
DIM = 6
HOST = "nodegpu247"
CPU_MODEL = "AMD EPYC 7713 64-Core Processor"
# [R4] asserted mapped set entries: realpath, sha256, soname (basename)
FAKE_LIBS = [{"realpath": "/fake/libtorch_cpu.so", "sha256": "1" * 64, "soname": "libtorch_cpu.so"}]


def fake_reflibs(asserted=None, driver=(), ldd_libs=()):
    """What ``S.capture_mapped_libs`` returns (synthetic)."""
    return {"method": "proc_self_maps", "soname_source": S.SONAME_SOURCE,
            "asserted": [dict(d) for d in (FAKE_LIBS if asserted is None else asserted)],
            "recorded_only": {"driver_pattern_matches": [dict(d) for d in driver],
                              "not_hashed": [],
                              "ldd_openslide_closure": {"target": S.OPENSLIDE_LIB,
                                                        "libraries": list(ldd_libs),
                                                        "error": None}},
            "driver_patterns": list(S.DRIVER_LIB_PATTERNS),
            "required_patterns": list(S.REQUIRED_LIB_PATTERNS)}


PROBE_READ = ((0, 0), 0, (512, 512))        # [R4] the one reference-library probe read
SLIDES = {"S1": 5, "S2": 3, "S3": 2}
BASES = {"S1": (1000, 2000), "S2": (5000, 9000), "S3": (300, 700)}
ENC = "h_optimus_1"


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeSlide:
    """Deterministic RGBA pixels depending on absolute coordinates; ``alter`` zeroes an
    8x8 block of reads at the given locations (a changed WSI region)."""

    calls: list = []

    def __init__(self, path, alter=(), fail_at=()):
        self.path = path
        self.alter = {tuple(a) for a in alter}
        self.fail_at = {tuple(a) for a in fail_at}

    def read_region(self, loc, level, size):
        FakeSlide.calls.append((tuple(loc), level, tuple(size)))
        if tuple(loc) in self.fail_at:
            raise OSError(f"cannot read {loc}")
        x, y = loc
        w, h = size
        yy, xx = np.mgrid[0:h, 0:w]
        r = ((xx + x) * 7 + (yy + y) * 3) % 256
        g = ((xx + x) * 2 + (yy + y) * 11) % 256
        b = ((xx + x) * 5 + (yy + y) * 1 + 40) % 256
        arr = np.stack([r, g, b, np.full_like(r, 255)], -1).astype(np.uint8)
        if tuple(loc) in self.alter:
            arr[:8, :8, :3] = 0
        return Image.fromarray(arr)

    def close(self):
        pass


def make_opener(alter=(), fail_at=()):
    def opener(path):
        return FakeSlide(path, alter, fail_at)
    return opener


class FakeRH:
    """Stands in for X.TorchstainReinhard: float32 target tensors, ``info``, callable."""

    def __init__(self, means=PINNED_MEANS, stds=PINNED_STDS, dtype=torch.float32,
                 nonfinite_r0=None):
        self.normalizer = types.SimpleNamespace(
            target_means=torch.tensor(means, dtype=dtype),
            target_stds=torch.tensor(stds, dtype=dtype))
        self.info = {"fake": True, "target_means": [float(v) for v in self.normalizer.target_means],
                     "target_stds": [float(v) for v in self.normalizer.target_stds]}
        self.nonfinite_r0 = nonfinite_r0

    def __call__(self, img):
        finite = not (self.nonfinite_r0 is not None
                      and int(np.asarray(img)[0, 0, 0]) == self.nonfinite_r0)
        return Image.eval(img, lambda v: 255 - v), finite


def rh_factory(template, sha, prefix):
    assert sha == X.TEMPLATE_SHA256
    assert torch.get_num_threads() == 1          # set BEFORE construction (specification A2)
    return FakeRH()


_PROJ = torch.randn(3 * 8 * 8, DIM, generator=torch.Generator().manual_seed(0))


def fake_transform(img):
    return F.normalize(F.to_tensor(F.resize(img, [8, 8])), [0.5] * 3, [0.25] * 3)


FORWARD_CALLS: list = []


def _fake_forward(x):
    FORWARD_CALLS.append(tuple(x.shape))
    return x.flatten(1) @ _PROJ


# what X._hf_weights_info would report (record key encoder_meta.weights) [R1]
FAKE_WEIGHTS = {"repo_id": "fake/encoder", "file": "model.safetensors",
                "snapshot_revision": "0" * 40, "cache_path": "/fake/hf/model.safetensors",
                "blob": "/fake/hf/blobs/ab", "bytes": 123, "sha256": "ab" * 32}


def fake_factory(name, device, amp):
    return X.EncoderRuntime(name, torch.nn.Identity(), fake_transform, _fake_forward, amp, DIM,
                            {"recipe": "fake", "transform": "fake", "weights": dict(FAKE_WEIGHTS)})


def perturbed_factory(name, device, amp):
    """A second encoder whose features are the fake encoder's + 1.0 (same weights info)."""
    return X.EncoderRuntime(name, torch.nn.Identity(), fake_transform,
                            lambda x: _fake_forward(x) + 1.0, amp, DIM,
                            {"recipe": "fake", "transform": "fake", "weights": dict(FAKE_WEIGHTS)})


def probe_values(**over):
    vers = {k: S.EXPECTED_ENV[k] for k in ("python_major_minor", "numpy", "torch", "torchvision",
                                           "pillow", "openslide_python", "openslide_library")}
    vals = dict(
        hostname=lambda: HOST,
        lsf_host_info=lambda h: {"model": "Opteron140", "resources": ["gpu", "driver_595"],
                                 "driver_tags": ["driver_595"]},
        cpu_vendor=lambda: "AuthenticAMD", cpu_model=lambda: CPU_MODEL,
        cpu_capability=lambda: "AVX2", versions=lambda: dict(vers, python="3.12.9"),
        kernel=lambda: "4.18.0", os_release=lambda: "Red Hat Enterprise Linux 8.10",
        reference_libs=lambda: fake_reflibs(),
        torchstain_members=lambda p: dict(X.TORCHSTAIN_MEMBERS),
        extractor_sha256=lambda: S.EXTRACTOR_SHA256)
    vals.update(over)
    return vals


def fake_probe(**over):
    return S.EnvProbe(**probe_values(**over))


def stimuli_for(sid, brightness, shift):
    p = {
        "identity": {"treatment": "untreated"},
        "rotate_90": {"angle": 90}, "rotate_180": {"angle": 180}, "rotate_270": {"angle": 270},
        "flip_horizontal": {"axis": "horizontal"}, "flip_vertical": {"axis": "vertical"},
        "brightness": {"factor": brightness}, "saturation": {"factor": 1.1},
        "contrast": {"factor": 0.95}, "spatial_shift": {"x_shift": shift[0], "y_shift": shift[1]},
        "reinhard": {"template_sha256": X.TEMPLATE_SHA256},
        "gaussian_blur": {"application_stage": "post_preprocessing", "kernel_size": 3,
                          "sigma": 0.5},
    }
    return [{"condition": c, "level": 0, "parameters": p[c], "patch_size": 512,
             "slide_id": sid} for c in X.CONDITIONS]


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def coords_of(sid):
    x0, y0 = BASES[sid]
    return [{"wsi_loc": [x0 + 600 * k, y0 + 50 * k]} for k in range(SLIDES[sid])]


SHORT_TMP_BASE = os.environ.get("PS1_TEST_SHORT_TMP_BASE", "/tmp")


def _ensure_short_tmpdir(monkeypatch, base=None):
    """[D1] worker DataLoaders need TMPDIR <= 75 bytes (AF_UNIX listener). If the runner's
    temp directory is too long (e.g. TMPDIR under a long basetemp), the test sets its own
    short TMPDIR (``mkdtemp`` under ``SHORT_TMP_BASE``); returns the created dir or None.
    ``S.check_worker_tempdir`` itself is not weakened."""
    cur = tempfile.gettempdir()
    if len(os.fsencode(cur)) + S.MP_SOCKET_SUFFIX_LEN <= S.UNIX_SOCKET_PATH_MAX:
        return None
    base = base or SHORT_TMP_BASE
    if not os.path.isdir(base):
        pytest.skip(f"runner TMPDIR {cur!r} too long for worker sockets and short base "
                    f"{base!r} unavailable")
    d = tempfile.mkdtemp(dir=base, prefix="ps1_")
    if len(os.fsencode(d)) + S.MP_SOCKET_SUFFIX_LEN > S.UNIX_SOCKET_PATH_MAX:
        shutil.rmtree(d, ignore_errors=True)
        pytest.skip(f"short TMPDIR base {base!r} is itself too long")
    monkeypatch.setenv("TMPDIR", d)
    monkeypatch.setattr(tempfile, "tempdir", d)
    return d


@pytest.fixture()
def world(tmp_path, monkeypatch):
    short_tmp = _ensure_short_tmpdir(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("PROSPER_OUT", str(tmp_path / "prosper_out_unused"))
    for v in ("LSB_JOBID", "LSB_JOBINDEX", "LSB_QUEUE", "LSB_DJOB_NUMPROC"):
        monkeypatch.delenv(v, raising=False)
    inp = tmp_path / "in"
    inp.mkdir()
    shifts = {"S1": (-5, 7), "S2": (3, -2), "S3": (11, 4)}
    bright = {"S1": 0.9, "S2": 1.12, "S3": 0.97}
    records = []
    for sid in SLIDES:
        records += stimuli_for(sid, bright[sid], shifts[sid])
    v1 = inp / "stimuli.json"
    v1.write_bytes(json.dumps({"policy": {"name": "r2_1_stimulus_v1", "rng": "PCG64", "seed": 7},
                               "records": records}, sort_keys=True,
                              separators=(",", ":")).encode())
    rows = []
    for i, sid in enumerate(SLIDES):
        n = SLIDES[sid]
        bank = inp / "bank" / f"{sid}.pkl"
        bank.parent.mkdir(exist_ok=True)
        data = pickle.dumps((np.zeros((n, 4), np.float32), coords_of(sid)), protocol=4)
        bank.write_bytes(data)
        rows.append({"population_index": i, "metadata_index": i + 1, "slide_id": sid,
                     "patient_id": f"P{i}", "diagnostic_group": "X", "feature_rows": n,
                     "feature_path": str(bank), "resolved_path": str(bank),
                     "feature_size": len(data), "feature_sha256": _sha(data)})
    queries = inp / "queries.csv"
    with open(queries, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    wsi = tmp_path / "wsi"
    wsi.mkdir()
    for sid in SLIDES:
        (wsi / f"{sid}.svs").write_bytes(b"fake wsi " + sid.encode())
    reflibs = inp / "reference_libs.json"
    reflibs.write_text(json.dumps({"schema": S.REFLIBS_SCHEMA, "asserted": FAKE_LIBS}))
    (inp / "torchstain").mkdir()
    out = tmp_path / "suite"
    v1sha = _sha(v1.read_bytes())
    assert S.gen_manifest_v2(["--out-root", str(out), "--v1", str(v1), "--v1-sha256", v1sha]) == 0
    v2 = out / "stimuli_v2.json"
    FakeSlide.calls = []
    FORWARD_CALLS.clear()
    yield types.SimpleNamespace(
        tmp=tmp_path, inp=inp, v1=v1, v1sha=v1sha, v2=v2, v2sha=_sha(v2.read_bytes()),
        queries=queries, qsha=_sha(queries.read_bytes()), rows=rows, records=records,
        out=out, reflibs=reflibs, wsi=wsi, slide_path_of=lambda sid: wsi / f"{sid}.svs")
    if short_tmp is not None:
        shutil.rmtree(short_tmp, ignore_errors=True)


def input_args(w, out=None):
    return ["--out-root", str(out or w.out), "--manifest-v2", str(w.v2),
            "--manifest-v2-sha256", w.v2sha, "--v1", str(w.v1), "--v1-sha256", w.v1sha,
            "--queries", str(w.queries), "--queries-sha256", w.qsha]


def env_args(w, reflibs=True):
    a = ["--template", str(w.inp / "template.png"), "--torchstain-prefix", str(w.inp / "torchstain")]
    return a + (["--reference-libs", str(w.reflibs)] if reflibs else [])


def inject(w, **over):
    kw = dict(opener=make_opener(), slide_path_of=w.slide_path_of, reinhard_factory=rh_factory,
              probe=fake_probe())
    kw.update(over)
    return kw


def fingerprint(w, slides="S1,S2,S3", out=None, processes=0, extra=(), **over):
    return S.fingerprint_main(input_args(w, out) + env_args(w) + ["--slides", slides,
                              "--processes", str(processes), "--loader-batch", "2", *extra],
                              **inject(w, **over))


def stim(w, sid):
    return {r["condition"]: r for r in w.records if r["slide_id"] == sid}


def independent_pictures(w, sid, i, opener=None, rh=None):
    """Independent re-render of tile i: X's two reads, X.condition_images, blur of identity."""
    st = stim(w, sid)
    x, y = coords_of(sid)[i]["wsi_loc"]
    sl = (opener or make_opener())("p")
    dx, dy = X.shift_of(st)
    region = sl.read_region((x, y), 0, (512, 512)).convert("RGB")
    shifted = sl.read_region((x + dx, y + dy), 0, (512, 512)).convert("RGB")
    imgs, _ = X.condition_images(region, shifted, st, rh or FakeRH())
    pics = [imgs[c] for c in X.CONDITIONS]
    pics.append(Image.fromarray(S.gaussian_blur_image(np.asarray(imgs["identity"]))))
    return pics


# --------------------------------------------------------------------------
# Frozen extractor identity
# --------------------------------------------------------------------------


def test_frozen_extractor_hash_and_import_assertion(tmp_path):
    assert _sha(Path(X.__file__).read_bytes()) == X_SHA256 == S.EXTRACTOR_SHA256
    assert S.SUITE_CONDITIONS == tuple(X.CONDITIONS) + ("gaussian_blur_image",)
    assert S.SUITE_CODE_SHA256 == _sha(Path(S.__file__).read_bytes())
    bad = tmp_path / "x.py"
    bad.write_bytes(Path(X.__file__).read_bytes() + b"\n")
    with pytest.raises(S.SuiteError):
        S.check_extractor(bad, X.CONDITIONS)
    with pytest.raises(S.SuiteError):
        S.check_extractor(Path(X.__file__), tuple(X.CONDITIONS)[:-1])


# --------------------------------------------------------------------------
# Item 1: manifest v2
# --------------------------------------------------------------------------


def test_manifest_v2_round_trip(world):
    w = world
    v1doc = json.loads(w.v1.read_bytes())
    v2doc = json.loads(w.v2.read_bytes())
    n1 = len(v1doc["records"])
    assert v2doc["records"][:n1] == v1doc["records"]
    assert json.dumps(v2doc["records"][:n1], sort_keys=True) == json.dumps(v1doc["records"],
                                                                           sort_keys=True)
    extra = v2doc["records"][n1:]
    assert [r["slide_id"] for r in extra] == list(SLIDES)            # v1 slide order
    for r in extra:
        assert r == {"condition": "gaussian_blur_image", "level": 0, "patch_size": 512,
                     "slide_id": r["slide_id"], "parameters": CONTRACT_BLUR_PARAMETERS}
        assert type(r["parameters"]["kernel_size"]) is int
        assert type(r["parameters"]["sigma"]) is float
    assert v2doc["policy"] == {"name": "r2_1_stimulus_v2", "rng": "PCG64", "seed": 7,
                               "supersedes": {"name": "r2_1_stimulus_v1", "sha256": w.v1sha}}
    assert (w.out / "stimuli_v2.json.sha256").read_text() == f"{w.v2sha}  stimuli_v2.json\n"
    m = S.load_manifest_v2(w.v2, w.v2sha, w.v1, w.v1sha)
    assert m.slide_order == list(SLIDES)
    assert set(m.by_slide["S1"]) == set(S.SUITE_CONDITIONS)
    assert m.stimuli_v1("S2") == X.load_manifest(w.v1, w.v1sha)[1]["S2"]
    with pytest.raises(X.InputIntegrityError):          # X itself rejects v2
        X.load_manifest(w.v2, None)
    # idempotent regeneration, never overwritten
    assert S.gen_manifest_v2(["--out-root", str(w.out), "--v1", str(w.v1),
                              "--v1-sha256", w.v1sha]) == 0
    assert _sha(w.v2.read_bytes()) == w.v2sha


def _v2_variant(w, mutate):
    doc = json.loads(w.v2.read_bytes())
    mutate(doc)
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize("name,mutate", [
    ("v1 record changed", lambda d: d["records"][6]["parameters"].update(factor=0.91)),
    ("v1 record dropped", lambda d: d["records"].pop(3)),
    ("policy renamed", lambda d: d["policy"].update(name="r2_1_stimulus_v3")),
    ("policy seed", lambda d: d["policy"].update(seed=8)),
    ("supersedes sha", lambda d: d["policy"]["supersedes"].update(sha256="0" * 64)),
    ("sigma", lambda d: d["records"][-1]["parameters"].update(sigma=1.15)),
    ("kernel float", lambda d: d["records"][-1]["parameters"].update(kernel_size=7.0)),
    ("weights", lambda d: d["records"][-1]["parameters"]["weights_hex"].reverse() or
     d["records"][-1]["parameters"]["weights_hex"].__setitem__(0, "0x1p-7")),
    ("extra key", lambda d: d["records"][-1].update(note="x")),
    ("order", lambda d: d["records"].__setitem__(slice(-3, None), d["records"][-3:][::-1])),
    ("missing new", lambda d: d["records"].pop()),
    ("extra new", lambda d: d["records"].append(dict(d["records"][-1]))),
    ("top-level key", lambda d: d.update(note=1)),
])
def test_manifest_v2_rejections(world, name, mutate):
    raw = _v2_variant(world, mutate)
    with pytest.raises(S.SuiteError):
        S.parse_manifest_v2(raw, world.v1, world.v1sha)


def test_manifest_v2_sha_and_v1_checks(world, tmp_path):
    w = world
    with pytest.raises(S.SuiteError):
        S.load_manifest_v2(w.v2, "0" * 64, w.v1, w.v1sha)
    with pytest.raises(S.SuiteError):
        S.load_manifest_v2(w.v2, "", w.v1, w.v1sha)
    with pytest.raises(X.InputIntegrityError):
        S.load_manifest_v2(w.v2, w.v2sha, w.v1, "0" * 64)
    with pytest.raises(S.SuiteError):
        S.build_manifest_v2(w.v1, "")
    bad = tmp_path / "v1bad.json"
    doc = json.loads(w.v1.read_bytes())
    doc["policy"]["name"] = "other"
    bad.write_bytes(json.dumps(doc).encode())
    with pytest.raises(S.SuiteError):
        S.build_manifest_v2(bad, _sha(bad.read_bytes()))
    # an existing different stimuli_v2.json is never overwritten
    out2 = tmp_path / "o2"
    out2.mkdir()
    (out2 / "stimuli_v2.json").write_bytes(b"{}")
    with pytest.raises(S.SuiteError):
        S.gen_manifest_v2(["--out-root", str(out2), "--v1", str(w.v1), "--v1-sha256", w.v1sha])
    assert (out2 / "stimuli_v2.json").read_bytes() == b"{}"


# --------------------------------------------------------------------------
# Item 2: deterministic blur primitive
# --------------------------------------------------------------------------


def test_blur_golden_literal():
    img = np.random.default_rng(0).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    out = S.gaussian_blur_image(img)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert hashlib.sha256(out.tobytes()).hexdigest() == GOLDEN
    assert S.BLUR_GOLDEN_SHA256 == GOLDEN
    assert S.blur_golden_check() == {"expected": GOLDEN, "observed": GOLDEN, "ok": True}


def test_blur_weights_are_contract_literals_and_math_exp():
    assert list(S.BLUR_WEIGHTS_HEX) == CONTRACT_BLUR_PARAMETERS["weights_hex"]
    assert S.BLUR_PARAMETERS == CONTRACT_BLUR_PARAMETERS
    sigma = 8 / 7
    assert sigma == CONTRACT_BLUR_PARAMETERS["sigma"]
    g = [math.exp(-((i - 3) ** 2) / (2 * sigma * sigma)) for i in range(7)]
    total = 0.0
    for v in g:                                   # sequential sum
        total += v
    expect = [v / total for v in g]
    assert [float.fromhex(h) for h in CONTRACT_BLUR_PARAMETERS["weights_hex"]] == expect
    assert list(S.BLUR_WEIGHTS) == expect


def _reflect(i, n):
    if i < 0:
        return -i
    if i >= n:
        return 2 * (n - 1) - i
    return i


def _hand_pass(rows, w):
    """Pure-Python 7-tap pass over each row, same accumulation order, explicit reflect_101."""
    out = []
    for r in rows:
        n = len(r)
        o = []
        for j in range(n):
            acc = w[0] * r[_reflect(j - 3, n)]
            for k in range(1, 7):
                acc = acc + w[k] * r[_reflect(j - 3 + k, n)]
            o.append(acc)
        out.append(o)
    return out


def test_blur_hand_computed_1x7_pass():
    w = [float.fromhex(h) for h in CONTRACT_BLUR_PARAMETERS["weights_hex"]]
    row = [0.0, 10.0, 20.0, 255.0, 3.0, 99.0, 7.0]
    got = S._blur_pass(np.array(row, np.float64).reshape(1, 7, 1), axis=1)[0, :, 0].tolist()
    assert got == _hand_pass([row], w)[0]
    # explicit first element: padded row = [255, 20, 10, 0, 10, 20, 255] (reflect, no edge repeat)
    e0 = w[0] * 255.0
    for k, v in enumerate([20.0, 10.0, 0.0, 10.0, 20.0, 255.0], 1):
        e0 = e0 + w[k] * v
    assert got[0] == e0


def test_blur_hand_computed_7x7():
    w = [float.fromhex(h) for h in CONTRACT_BLUR_PARAMETERS["weights_hex"]]
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (7, 7, 1), dtype=np.uint8)
    img[3, 3, 0] = 255
    rows = img[:, :, 0].astype(float).tolist()
    h = _hand_pass(rows, w)                                     # horizontal (columns padded)
    v = np.array(_hand_pass(np.array(h).T.tolist(), w)).T        # vertical (rows padded)
    expect = np.clip(np.rint(v), 0, 255).astype(np.uint8)
    got = S.gaussian_blur_image(img)[:, :, 0]
    assert np.array_equal(got, expect)
    # impulse: centre = rint(255 * w3 * w3) = 31; corner gets the reflected tap weights
    imp = np.zeros((7, 7, 3), np.uint8)
    imp[3, 3] = 255
    out = S.gaussian_blur_image(imp)
    assert int(np.rint(255 * w[3] * w[3])) == 31 and out[3, 3, 0] == 31
    corner = 255 * (w[0] + w[6]) * (w[0] + w[6])
    assert out[0, 0, 1] == np.clip(np.rint(corner), 0, 255)
    assert (out[..., 0] == out[..., 2]).all()
    with pytest.raises(ValueError):
        S.gaussian_blur_image(np.zeros((3, 9, 3), np.uint8))
    with pytest.raises(ValueError):
        S.gaussian_blur_image(np.zeros((9, 9, 3), np.float64))


def test_blur_rounding_half_even():
    # a flat image stays flat (weights sum to ~1); 0 and 255 saturate correctly
    for v in (0, 1, 127, 128, 254, 255):
        flat = np.full((9, 9, 3), v, np.uint8)
        assert (S.gaussian_blur_image(flat) == v).all()


def test_blur_equal_with_numpy_cpu_features_disabled():
    code = ("import os, hashlib, numpy as np;"
            "from numpy._core._multiarray_umath import __cpu_features__ as f;"
            "from prosper.evaluation import r2_1_perturb_suite_cli as S;"
            "img = np.random.default_rng(0).integers(0, 256, (512, 512, 3), dtype=np.uint8);"
            "print(f.get('AVX2'), f.get('FMA3'), hashlib.sha256(S.gaussian_blur_image(img).tobytes()).hexdigest())")
    env = dict(os.environ, NPY_DISABLE_CPU_FEATURES="AVX2 FMA3 F16C AVX", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    avx2, fma3, digest = r.stdout.split()[-3:]
    assert (avx2, fma3) == ("False", "False")
    assert digest == GOLDEN


def test_blur_agrees_with_torchvision_within_one():
    for seed in (0, 1, 2):
        img = np.random.default_rng(seed).integers(0, 256, (64, 96, 3), dtype=np.uint8)
        ours = S.gaussian_blur_image(img).astype(np.int16)
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        tv = F.gaussian_blur(t, [7, 7], [8 / 7, 8 / 7]).permute(1, 2, 0).numpy().astype(np.int16)
        assert np.abs(ours - tv).max() <= 1


def test_blur_picture_is_applied_to_identity_picture():
    region = FakeSlide("p").read_region((10, 20), 0, (512, 512)).convert("RGB")
    pic = S.blur_picture(region)
    assert pic.mode == "RGB" and pic.size == (512, 512)
    assert np.array_equal(np.asarray(pic), S.gaussian_blur_image(np.asarray(region)))


# --------------------------------------------------------------------------
# Fixed rendering environment [A2]
# --------------------------------------------------------------------------

ASSERTED_KEYS = {"lsf_model", "cpu_vendor", "cpu_capability", "python_major_minor", "numpy",
                 "torch", "torchvision", "pillow", "openslide_python", "openslide_library",
                 "torchstain_members", "extractor_code_sha256", "threads", "reference_libs"}


def test_expected_env_literals():
    assert S.EXPECTED_ENV == {
        "lsf_model": "Opteron140", "cpu_vendor": "AuthenticAMD", "cpu_capability": "AVX2",
        "python_major_minor": "3.12", "numpy": "2.2.4", "torch": "2.5.1+cu124",
        "torchvision": "0.20.1+cu124", "pillow": "11.3.0", "openslide_python": "1.4.3",
        "openslide_library": "3.4.1"}
    assert S.OPENSLIDE_LIB == "/lib64/libopenslide.so.0"
    assert list(S.PINNED_MEANS) == PINNED_MEANS and list(S.PINNED_STDS) == PINNED_STDS
    assert list(S.DIRECT_MEANS) == DIRECT_MEANS


def test_environment_block_asserted_and_recorded_split(world, monkeypatch):
    monkeypatch.setenv("LSB_JOBID", "4242")
    S.set_single_thread()
    blk = S.environment_block(fake_probe(), torchstain_prefix=world.inp / "torchstain",
                              reference_libs_path=world.reflibs)
    assert blk["ok"] and blk["deviations"] == []
    assert set(blk["asserted"]) == ASSERTED_KEYS
    rec = blk["recorded"]
    # recorded only: CPU model string, kernel, OS release, hostname, LSF job id
    assert rec["cpu_model"] == CPU_MODEL and rec["kernel"] == "4.18.0"
    assert rec["os_release"] == "Red Hat Enterprise Linux 8.10"
    assert rec["hostname"] == HOST and rec["lsf_job"] == "4242"
    assert rec["gpu_driver_tags"] == ["driver_595"]
    assert not set(rec) & ASSERTED_KEYS
    assert blk["asserted"]["reference_libs"]["pinned_file_sha256"] == _sha(world.reflibs.read_bytes())
    # a different CPU model string / kernel / OS / host never aborts
    blk2 = S.environment_block(fake_probe(cpu_model=lambda: "AMD EPYC 9999", kernel=lambda: "5.1",
                                          os_release=lambda: "other", hostname=lambda: "h2"),
                               torchstain_prefix=world.inp / "torchstain",
                               reference_libs_path=world.reflibs)
    assert blk2["ok"]


def _versions_with(**kw):
    v = probe_values()["versions"]()
    v.update(kw)
    return lambda: v


@pytest.mark.parametrize("over", [
    dict(lsf_host_info=lambda h: {"model": "XeonE5", "resources": [], "driver_tags": []}),
    dict(lsf_host_info=lambda h: (_ for _ in ()).throw(S.SuiteError("no lshosts"))),
    dict(cpu_vendor=lambda: "GenuineIntel"),
    dict(cpu_capability=lambda: "AVX512"),
    dict(versions=_versions_with(python_major_minor="3.11")),
    dict(versions=_versions_with(numpy="2.2.5")),
    dict(versions=_versions_with(torch="2.5.1")),
    dict(versions=_versions_with(torchvision="0.20.1")),
    dict(versions=_versions_with(pillow="11.2.0")),
    dict(versions=_versions_with(openslide_python="1.4.2")),
    dict(versions=_versions_with(openslide_library="4.0.0")),
    dict(torchstain_members=lambda p: dict(X.TORCHSTAIN_MEMBERS, **{
        next(iter(X.TORCHSTAIN_MEMBERS)): "0" * 64})),
    dict(extractor_sha256=lambda: "0" * 64),
    dict(reference_libs=lambda: fake_reflibs([dict(FAKE_LIBS[0], sha256="2" * 64)])),
    dict(reference_libs=lambda: fake_reflibs([dict(FAKE_LIBS[0], realpath="/other/libtorch_cpu.so")])),
    dict(reference_libs=lambda: fake_reflibs(FAKE_LIBS + [dict(
        FAKE_LIBS[0], realpath="/fake/extra.so", soname="extra.so")])),
    dict(reference_libs=lambda: fake_reflibs([])),
])
def test_environment_asserted_deviation_aborts(world, over):
    S.set_single_thread()
    with pytest.raises(S.EnvironmentMismatch):
        S.environment_block(fake_probe(**over), torchstain_prefix=world.inp / "torchstain",
                            reference_libs_path=world.reflibs)
    blk = S.environment_block(fake_probe(**over), torchstain_prefix=world.inp / "torchstain",
                              reference_libs_path=world.reflibs, enforce=False)
    assert not blk["ok"] and len(blk["deviations"]) >= 1


def test_environment_threads_asserted(world, monkeypatch):
    S.set_single_thread()
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(S.EnvironmentMismatch, match="threads"):
        S.environment_block(fake_probe(), torchstain_prefix=world.inp / "torchstain",
                            reference_libs_path=world.reflibs)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("MKL_NUM_THREADS")
    with pytest.raises(S.EnvironmentMismatch, match="threads"):
        S.environment_block(fake_probe(), torchstain_prefix=world.inp / "torchstain",
                            reference_libs_path=world.reflibs)
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with pytest.raises(S.EnvironmentMismatch, match="threads"):
            S.environment_block(fake_probe(), torchstain_prefix=world.inp / "torchstain",
                                reference_libs_path=world.reflibs)
    finally:
        torch.set_num_threads(1)
    assert old == 1


def test_reference_libs_required_unless_explicit_waiver(world):
    S.set_single_thread()
    with pytest.raises(S.EnvironmentMismatch, match="reference-libs"):
        S.environment_block(fake_probe(), torchstain_prefix=world.inp / "torchstain",
                            reference_libs_path=None)
    blk = S.environment_block(fake_probe(), torchstain_prefix=world.inp / "torchstain",
                              reference_libs_path=None, allow_missing_reference_libs=True)
    assert blk["ok"] and blk["allow_missing_reference_libs"] is True
    assert blk["asserted"]["reference_libs"]["status"] == "NOT_ASSERTED"
    bad = world.tmp / "bad_reflibs.json"
    bad.write_text(json.dumps({"schema": "other", "asserted": []}))
    with pytest.raises(S.EnvironmentMismatch):
        S.environment_block(fake_probe(), torchstain_prefix=world.inp / "torchstain",
                            reference_libs_path=bad)
    # a candidate-001 (ldd-based, ``libraries``) pin is no longer accepted [R4]
    old = world.tmp / "old_reflibs.json"
    old.write_text(json.dumps({"schema": "r2_1_perturb_suite_reference_libs_v1",
                               "libraries": FAKE_LIBS}))
    with pytest.raises(S.EnvironmentMismatch):
        S.environment_block(fake_probe(), torchstain_prefix=world.inp / "torchstain",
                            reference_libs_path=old)
    # subcommands: fingerprint aborts before rendering without --reference-libs
    a = input_args(world) + env_args(world, reflibs=False) + ["--slides", "S1", "--processes", "0"]
    with pytest.raises(S.EnvironmentMismatch):
        S.fingerprint_main(a, **inject(world))
    assert FakeSlide.calls == []
    assert not (world.out / "fingerprints" / "S1.fp.npy").exists()
    assert S.fingerprint_main(a + ["--allow-missing-reference-libs"], **inject(world)) == 0
    doc = json.loads((world.out / "fingerprints" / "S1.fp.json").read_text())
    assert doc["environment"]["allow_missing_reference_libs"] is True
    assert doc["environment"]["asserted"]["reference_libs"]["status"] == "NOT_ASSERTED"


def test_reference_lib_mismatch_aborts_fingerprint_before_render(world):
    probe = fake_probe(reference_libs=lambda: fake_reflibs([dict(FAKE_LIBS[0], sha256="9" * 64)]))
    with pytest.raises(S.EnvironmentMismatch, match="reference_libs"):
        fingerprint(world, slides="S1", probe=probe)
    # [R4] only the one probe read (queries.csv row 0 = S1) precedes the maps check
    assert FakeSlide.calls == [PROBE_READ]
    assert not (world.out / "fingerprints").exists() or \
        not list((world.out / "fingerprints").glob("*.fp.*"))


def test_parse_ldd():
    text = ("\tlinux-vdso.so.1 (0x00007ffd)\n"
            "\tlibjpeg.so.62 => /lib64/libjpeg.so.62 (0x00007f1a2b000000)\n"
            "\tlibz.so.1 => /lib64/libz.so.1 (0x00007f1a2a000000)\n"
            "\t/lib64/ld-linux-x86-64.so.2 (0x00007f1a2c000000)\n")
    assert S.parse_ldd(text) == [("libjpeg.so.62", "/lib64/libjpeg.so.62"),
                                 ("libz.so.1", "/lib64/libz.so.1"),
                                 ("ld-linux-x86-64.so.2", "/lib64/ld-linux-x86-64.so.2")]
    with pytest.raises(S.SuiteError, match="unresolved"):
        S.parse_ldd("\tlibfoo.so.1 => not found\n")
    with pytest.raises(S.SuiteError, match="unparsed"):
        S.parse_ldd("\tstatically linked weird\n")


REQUIRED_BASENAMES = ["libopenslide.so.0.4.1", "libjpeg.so.62.3.0", "libz.so.1.3.2",
                      "libpng16.so.16.37.0", "libtiff.so.5.3.0", "libopenjp2.so.2.3.1",
                      "_imaging.cpython-312-x86_64-linux-gnu.so", "libtorch_cpu.so"]
DRIVER_BASENAMES = ["libcuda.so.595.58.03", "libnvidia-ml.so.1", "libcudadebugger.so.1",
                    "libGL.so.1", "libEGL.so.1", "libGLX.so.0", "libGLdispatch.so.0",
                    "libOpenGL.so.0", "libdrm.so.2", "libX11.so.6", "libxcb.so.1"]


def _maps_line(path, i=0):
    return f"7f00{i:08x}-7f01{i:08x} r-xp 00000000 fd:00 {1000 + i} {path}\n"


def make_mapped(tmp_path, names, extra=b"", elf=True):
    """Fake mapped files; [R6a] by default they start with the ELF magic (asserted unless a
    driver pattern matches); ``elf=False`` writes non-ELF (recorded-only) files."""
    d = tmp_path / "mapped"
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, n in enumerate(names):
        f = d / n
        f.write_bytes((S.ELF_MAGIC if elf else b"") + n.encode() + extra)
        lines.append(_maps_line(str(f), i))
        lines.append(_maps_line(str(f), i + 500))       # duplicate mapping (deduplicated)
    return d, "".join(lines)


def test_parse_maps_paths_only():
    text = ("00400000-00452000 r-xp 00000000 08:02 173521 /usr/bin/python3\n"
            "7f00-7f01 rw-p 00000000 00:00 0 \n"
            "7f02-7f03 rw-p 00000000 00:00 0 [heap]\n"
            "7f04-7f05 r--p 00000000 08:02 1 /lib64/libz.so.1.2.11\n"
            "7f06-7f07 r--p 00000000 08:02 1 /lib64/libz.so.1.2.11\n"
            "7f08-7f09 r--p 00000000 00:05 2 /dev/shm/x (deleted)\n")
    assert S.parse_maps(text) == ["/usr/bin/python3", "/lib64/libz.so.1.2.11",
                                  "/dev/shm/x (deleted)"]


def test_capture_mapped_libs_asserted_and_recorded_only(tmp_path):
    """[R4] driver/display-pattern files are recorded only; everything else mapped is
    asserted with realpath/sha256/soname(basename); deleted mappings are not hashed; the
    ldd closure is recorded only."""
    d, maps = make_mapped(tmp_path, REQUIRED_BASENAMES + DRIVER_BASENAMES + ["libzstd.so.1"])
    (d / "libz.so.1").symlink_to("libz.so.1.3.2")
    maps += _maps_line(str(d / "libz.so.1"), 900)               # symlink -> realpath dedup
    maps += _maps_line(str(d / "gone.so") + " (deleted)", 901)
    maps += _maps_line(str(d / "missing.so"), 902)
    ldd = f"\tlibjpeg.so.62 => {d}/libjpeg.so.62.3.0 (0x00007f00)\n"
    libs = S.capture_mapped_libs(maps_text=maps, ldd_text=ldd)
    asserted = {e["soname"]: e for e in libs["asserted"]}
    assert set(asserted) == set(REQUIRED_BASENAMES) | {"libzstd.so.1"}
    for n in REQUIRED_BASENAMES:
        assert asserted[n]["realpath"] == str(d / n)
        assert asserted[n]["sha256"] == _sha(S.ELF_MAGIC + n.encode())
    assert libs["soname_source"] == S.SONAME_SOURCE and libs["method"] == "proc_self_maps"
    drv = {e["soname"] for e in libs["recorded_only"]["driver_pattern_matches"]}
    assert drv == set(DRIVER_BASENAMES)
    nh = {e["path"]: e["reason"] for e in libs["recorded_only"]["not_hashed"]}
    assert nh[str(d / "gone.so") + " (deleted)"] == "deleted"
    assert nh[str(d / "missing.so")].startswith("stat:")
    ldd_doc = libs["recorded_only"]["ldd_openslide_closure"]
    assert ldd_doc["error"] is None
    assert ldd_doc["libraries"] == [{"name": "libjpeg.so.62", "realpath": str(d / "libjpeg.so.62.3.0"),
                                     "sha256": _sha(S.ELF_MAGIC + b"libjpeg.so.62.3.0")}]
    assert libs["asserted"] == sorted(libs["asserted"], key=lambda e: e["realpath"])
    m = S.check_required_libs(libs["asserted"])
    assert m["libz.so*"] == [str(d / "libz.so.1.3.2")]          # libzstd does not satisfy zlib
    # an ldd failure is recorded, never raised
    bad = S.capture_mapped_libs(maps_text=maps, ldd_text="\tlibfoo.so.1 => not found\n")
    assert "unresolved" in bad["recorded_only"]["ldd_openslide_closure"]["error"]


@pytest.mark.parametrize("missing", REQUIRED_BASENAMES)
def test_check_required_libs_missing_decode_library_fails(tmp_path, missing):
    names = [n for n in REQUIRED_BASENAMES if n != missing] + ["libzstd.so.1"]
    _, maps = make_mapped(tmp_path, names)
    libs = S.capture_mapped_libs(maps_text=maps, ldd_text="")
    with pytest.raises(S.SuiteError, match="lacks"):
        S.check_required_libs(libs["asserted"])
    # an empty asserted set fails too
    with pytest.raises(S.SuiteError, match="lacks"):
        S.check_required_libs([])


def test_capture_mapped_libs_reads_proc_self_maps():
    """Production path: this test process's own mapped closure includes libtorch_cpu and
    PIL._imaging (imported above), and nothing matching a driver pattern is asserted."""
    libs = S.capture_mapped_libs(ldd_text="")
    names = [e["soname"] for e in libs["asserted"]]
    import fnmatch
    assert any(fnmatch.fnmatchcase(n, "libtorch_cpu*") for n in names)
    assert any(fnmatch.fnmatchcase(n, "_imaging*") for n in names)
    assert not any(fnmatch.fnmatchcase(n, pat) for n in names for pat in S.DRIVER_LIB_PATTERNS)


def test_parse_lshosts():
    text = ("HOST_NAME   type    model  cpuf ncpus maxmem maxswp server RESOURCES\n"
            "nodegpu247  X86_64 Opteron140 60.0 128 1T 4G Yes (mg gpu driver_595 rhel8)\n"
            "nodegpu200  X86_64 Opteron140 60.0 128 1T 4G Yes (mg gpu driver_580)\n")
    assert S.parse_lshosts(text, "nodegpu247") == {
        "model": "Opteron140", "resources": ["mg", "gpu", "driver_595", "rhel8"],
        "driver_tags": ["driver_595"]}
    assert S.parse_lshosts(text, "nodegpu200")["driver_tags"] == ["driver_580"]
    with pytest.raises(S.SuiteError):
        S.parse_lshosts(text, "nodex")
    with pytest.raises(S.SuiteError):
        S.parse_lshosts("garbage\n", "nodegpu247")


def test_pinned_fit_assertion(world):
    S.set_single_thread()
    rh = S.make_pinned_reinhard(world.inp / "template.png", world.inp / "torchstain", rh_factory)
    assert isinstance(rh, FakeRH)
    for means, stds, dtype in [(DIRECT_MEANS, PINNED_STDS, torch.float32),
                               (PINNED_MEANS, [18.0647, 10.140364646911621, 11.612739562988281],
                                torch.float32),
                               (PINNED_MEANS, PINNED_STDS, torch.float64)]:
        with pytest.raises(S.PinnedFitError):
            S.make_pinned_reinhard(world.inp / "t.png", None,
                                   lambda t, sha, p, m=means, s=stds, d=dtype: FakeRH(m, s, d))
    # the factory sees one torch thread even if the process had more
    torch.set_num_threads(3)
    try:
        S.make_pinned_reinhard(world.inp / "template.png", None, rh_factory)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(1)
    # classification of first-run fits
    assert S.classify_fit(PINNED_MEANS, PINNED_STDS) == "pinned_1thread"
    assert S.classify_fit(DIRECT_MEANS, PINNED_STDS) == "direct_121thread"
    assert S.classify_fit([67.0867, 23.1169, -15.08], PINNED_STDS).startswith("other:")
    assert S.classify_fit(DIRECT_MEANS, [18.0, 10.0, 11.0]).startswith("other:")


def test_pinned_fit_mismatch_aborts_fingerprint(world):
    bad = lambda t, sha, p: FakeRH(DIRECT_MEANS)                 # noqa: E731
    with pytest.raises(S.PinnedFitError):
        fingerprint(world, slides="S1", reinhard_factory=bad)
    assert FakeSlide.calls == []


def required_reflibs(tmp_path, extra_names=(), driver_names=(), content=b""):
    """A synthetic [R4] mapped set that satisfies REQUIRED_LIB_PATTERNS."""
    _, maps = make_mapped(tmp_path, list(REQUIRED_BASENAMES) + list(extra_names)
                          + list(driver_names), extra=content)
    return S.capture_mapped_libs(maps_text=maps, ldd_text="")


def capture_args(w, out, probe_wsi=None):
    return ["--out-root", str(out), "--torchstain-prefix", str(w.inp / "torchstain"),
            "--template", str(w.inp / "template.png"),
            "--probe-wsi", str(probe_wsi or w.slide_path_of("S1"))]


def test_env_check_capture_and_pin_reference_libs(world, monkeypatch):
    out = world.out
    good = required_reflibs(world.tmp / "g", driver_names=["libcuda.so.1"])
    pin = world.tmp / "pinned_required.json"
    pin.write_text(json.dumps({"schema": S.REFLIBS_SCHEMA, "asserted": good["asserted"]}))
    probe_good = lambda **kw: fake_probe(reference_libs=lambda: good, **kw)   # noqa: E731
    eargs = ["--out-root", str(out), "--template", str(world.inp / "template.png"),
             "--torchstain-prefix", str(world.inp / "torchstain"), "--reference-libs", str(pin),
             "--probe-wsi", str(world.slide_path_of("S2"))]
    assert S.env_check_main(eargs, probe=probe_good(), reinhard_factory=rh_factory,
                            opener=make_opener()) == 0
    assert FakeSlide.calls == [PROBE_READ]
    assert S.env_check_main(eargs, probe=probe_good(cpu_vendor=lambda: "GenuineIntel"),
                            reinhard_factory=rh_factory, opener=make_opener()) == 1
    assert S.env_check_main(eargs, probe=probe_good(),
                            reinhard_factory=lambda t, s, p: FakeRH(DIRECT_MEANS),
                            opener=make_opener()) == 1
    # env-check needs --probe-wsi [R4]
    with pytest.raises(SystemExit):
        S.env_check_main(eargs[:-2], probe=probe_good(), reinhard_factory=rh_factory,
                         opener=make_opener())
    checks = sorted((out / "environment" / "checks").glob("*.json"))
    docs = [json.loads(c.read_text()) for c in checks]
    assert [d["ok"] for d in docs].count(True) == 1
    okdoc = next(d for d in docs if d["ok"])
    prb = okdoc["environment"]["recorded"]["reference_lib_probe"]
    assert prb["probe_wsi"]["path"] == str(world.slide_path_of("S2"))
    assert prb["probe_wsi"]["st_size"] == world.slide_path_of("S2").stat().st_size
    assert "st_mtime_ns" in prb["probe_wsi"] and prb["read_region"]["size"] == [512, 512]
    assert prb["blur_golden"]["ok"] and prb["pinned_reinhard_fit_asserted"]
    assert okdoc["environment"]["recorded"]["reference_libs_recorded_only"][
        "driver_pattern_matches"][0]["soname"] == "libcuda.so.1"
    FakeSlide.calls = []
    monkeypatch.setenv("LSB_QUEUE", "heavy_io")
    cargs = capture_args(world, out)
    kw = dict(reinhard_factory=rh_factory, opener=make_opener())
    assert S.capture_env_main(cargs, probe=probe_good(), **kw) == 0
    assert FakeSlide.calls == [PROBE_READ]                 # one probe read, no output from it
    monkeypatch.setenv("LSB_QUEUE", "rhel88_gpu")
    assert S.capture_env_main(cargs, probe=probe_good(), **kw) == 0
    caps = sorted((out / "environment" / "captures").glob("*.json"))
    assert len(caps) == 2
    c0 = json.loads(caps[0].read_text())
    assert c0["schema"] == S.ENV_CAPTURE_SCHEMA
    assert c0["blur_golden"]["ok"] and c0["environment"]["deviations"] == []
    assert set(c0["required_library_matches"]) == set(S.REQUIRED_LIB_PATTERNS)
    args = ["--out-root", str(out), "--captures", *map(str, caps)]
    assert S.pin_reference_libs_main(args + ["--expect-queues", "heavy_io,standard"]) == 1
    assert not (out / "environment" / "reference_libs.json").exists()
    assert S.pin_reference_libs_main(args + ["--expect-queues", "heavy_io,rhel88_gpu",
                                             "--expect-driver-tags", "driver_595"]) == 0
    pinned = json.loads((out / "environment" / "reference_libs.json").read_text())
    assert pinned["schema"] == S.REFLIBS_SCHEMA and pinned["asserted"] == good["asserted"]
    assert len(pinned["recorded_only_by_capture"]) == 2
    assert pinned["recorded_only_by_capture"][0]["recorded_only"]["driver_pattern_matches"][0][
        "soname"] == "libcuda.so.1"
    # the pinned file is accepted by every job's env assertion
    assert S.load_reference_libs(out / "environment" / "reference_libs.json")["asserted"] == \
        good["asserted"]
    # a capture whose ASSERTED set differs blocks pinning
    monkeypatch.setenv("LSB_QUEUE", "standard")
    other = required_reflibs(world.tmp / "o", content=b"x")      # same names, other bytes
    assert S.capture_env_main(cargs, probe=fake_probe(reference_libs=lambda: other), **kw) == 0
    caps = sorted((out / "environment" / "captures").glob("*.json"))
    out2 = world.tmp / "pin2"
    assert S.pin_reference_libs_main(["--out-root", str(out2), "--captures", *map(str, caps)]) == 1
    assert not (out2 / "environment" / "reference_libs.json").exists()


def test_capture_env_refuses_missing_decode_library(world, monkeypatch):
    """[R4] the capture FAILS (nothing published) unless every required pattern is mapped."""
    monkeypatch.setenv("LSB_QUEUE", "heavy_io")
    libs = required_reflibs(world.tmp / "m")
    libs["asserted"] = [e for e in libs["asserted"] if not e["soname"].startswith("libopenjp2")]
    with pytest.raises(S.SuiteError, match="libopenjp2"):
        S.capture_env_main(capture_args(world, world.out),
                           probe=fake_probe(reference_libs=lambda: libs),
                           reinhard_factory=rh_factory, opener=make_opener())
    assert not (world.out / "environment" / "captures").exists()


# --------------------------------------------------------------------------
# [R6a] ELF-only asserted set, non-ELF recorded only, REFLIBS v3, locale recorded
# --------------------------------------------------------------------------

NON_ELF_NAMES = ["LC_CTYPE", "gconv-modules.cache"]


def _mapped_set(tmp_path, non_elf=NON_ELF_NAMES):
    """Required ELF libs + one driver lib (ELF) + non-ELF data files, all in one dir."""
    d, maps = make_mapped(tmp_path, list(REQUIRED_BASENAMES) + ["libcuda.so.1"])
    _, maps_ne = make_mapped(tmp_path, list(non_elf), elf=False)
    return d, maps + maps_ne


def test_r6a_capture_asserts_only_elf_and_records_non_elf(tmp_path):
    d, maps = _mapped_set(tmp_path)
    (d / "tiny.bin").write_bytes(b"\x7fE")                   # shorter than the magic
    maps += _maps_line(str(d / "tiny.bin"), 950)
    libs = S.capture_mapped_libs(maps_text=maps, ldd_text="")
    assert {e["soname"] for e in libs["asserted"]} == set(REQUIRED_BASENAMES)
    ne = {e["soname"]: e for e in libs["recorded_only"]["non_elf"]}
    assert set(ne) == set(NON_ELF_NAMES) | {"tiny.bin"}
    for n in NON_ELF_NAMES:
        f = d / n
        assert ne[n]["realpath"] == str(f) and ne[n]["size"] == f.stat().st_size
        assert ne[n]["sha256"] == _sha(f.read_bytes())
    [drv] = libs["recorded_only"]["driver_pattern_matches"]
    assert drv["soname"] == "libcuda.so.1" and drv["size"] == (d / "libcuda.so.1").stat().st_size
    assert drv["sha256"] == _sha((d / "libcuda.so.1").read_bytes())
    # the required-pattern check applies to the asserted (ELF) set: a non-ELF file whose
    # name matches a required pattern does not satisfy it
    names = [n for n in REQUIRED_BASENAMES if n != "libtiff.so.5.3.0"]
    d2, maps2 = make_mapped(tmp_path / "x", names)
    _, maps_ne = make_mapped(tmp_path / "x", ["libtiff.so.5.3.0"], elf=False)
    libs2 = S.capture_mapped_libs(maps_text=maps2 + maps_ne, ldd_text="")
    assert [e["soname"] for e in libs2["recorded_only"]["non_elf"]] == ["libtiff.so.5.3.0"]
    with pytest.raises(S.SuiteError, match="libtiff"):
        S.check_required_libs(libs2["asserted"])


def _pin_from(world, libs, out, monkeypatch):
    monkeypatch.setenv("LSB_QUEUE", "heavy_io")
    kw = dict(reinhard_factory=rh_factory, opener=make_opener())
    assert S.capture_env_main(capture_args(world, out),
                              probe=fake_probe(reference_libs=lambda: libs), **kw) == 0
    caps = sorted((out / "environment" / "captures").glob("*.json"))
    assert S.pin_reference_libs_main(["--out-root", str(out), "--captures", *map(str, caps)]) == 0
    return out / "environment" / "reference_libs.json"


def _env_check(world, out, pin, libs):
    eargs = ["--out-root", str(out), "--template", str(world.inp / "template.png"),
             "--torchstain-prefix", str(world.inp / "torchstain"), "--reference-libs", str(pin),
             "--probe-wsi", str(world.slide_path_of("S1"))]
    rc = S.env_check_main(eargs, probe=fake_probe(reference_libs=lambda: libs),
                          reinhard_factory=rh_factory, opener=make_opener())
    docs = [json.loads(c.read_text())
            for c in sorted((out / "environment" / "checks").glob("*.json"))]
    return rc, docs[-1]


def test_r6a_non_elf_difference_does_not_hold_job(world, monkeypatch):
    """A mapped non-ELF file (locale archive / gconv cache) that differs between the pinned
    capture and the job's observation (other bytes, or absent) is recorded, never refused."""
    d, maps = _mapped_set(world.tmp / "m")
    pinned_libs = S.capture_mapped_libs(maps_text=maps, ldd_text="")
    pin = _pin_from(world, pinned_libs, world.out, monkeypatch)
    pdoc = json.loads(pin.read_text())
    assert pdoc["schema"] == S.REFLIBS_SCHEMA == "r2_1_perturb_suite_reference_libs_v3"
    old_sha = _sha((d / "LC_CTYPE").read_bytes())
    p_ne = {e["soname"]: e for e in pdoc["recorded_only_by_capture"][0]["recorded_only"]["non_elf"]}
    assert p_ne["LC_CTYPE"]["sha256"] == old_sha
    assert not any(e["soname"] == "LC_CTYPE" for e in pdoc["asserted"])
    # the job host's locale archive has other bytes (and is larger)
    (d / "LC_CTYPE").write_bytes(b"other locale archive bytes" * 10)
    obs = S.capture_mapped_libs(maps_text=maps, ldd_text="")
    assert obs["asserted"] == pinned_libs["asserted"]
    rc, doc = _env_check(world, world.out, pin, obs)
    assert rc == 0 and doc["ok"], doc.get("error")
    o_ne = {e["soname"]: e for e in
            doc["environment"]["recorded"]["reference_libs_recorded_only"]["non_elf"]}
    assert o_ne["LC_CTYPE"]["sha256"] == _sha(b"other locale archive bytes" * 10) != old_sha
    assert o_ne["LC_CTYPE"]["size"] == 260
    # the locale archive not mapped at all on the job host: still no refusal
    _, maps_no_locale = _mapped_set(world.tmp / "m", non_elf=["gconv-modules.cache"])
    obs2 = S.capture_mapped_libs(maps_text=maps_no_locale, ldd_text="")
    rc, doc = _env_check(world, world.out, pin, obs2)
    assert rc == 0 and doc["ok"]


def test_r6a_elf_difference_still_refuses(world, monkeypatch):
    d, maps = _mapped_set(world.tmp / "m")
    pin = _pin_from(world, S.capture_mapped_libs(maps_text=maps, ldd_text=""), world.out,
                    monkeypatch)
    (d / "libjpeg.so.62.3.0").write_bytes(S.ELF_MAGIC + b"patched libjpeg")
    obs = S.capture_mapped_libs(maps_text=maps, ldd_text="")
    rc, doc = _env_check(world, world.out, pin, obs)
    assert rc == 1 and not doc["ok"] and "reference_libs" in doc["error"]
    # an extra mapped ELF object also refuses
    (d / "libjpeg.so.62.3.0").write_bytes(S.ELF_MAGIC + b"libjpeg.so.62.3.0")   # restored
    assert _env_check(world, world.out, pin,
                      S.capture_mapped_libs(maps_text=maps, ldd_text=""))[0] == 0
    _, extra = make_mapped(world.tmp / "e", ["libextra.so.1"])
    obs3 = S.capture_mapped_libs(maps_text=maps + extra, ldd_text="")
    rc, doc = _env_check(world, world.out, pin, obs3)
    assert rc == 1 and "reference_libs" in doc["error"]
    # and fingerprint aborts before rendering on an ELF difference
    probe = fake_probe(reference_libs=lambda: obs3)
    FakeSlide.calls = []
    a = input_args(world) + ["--template", str(world.inp / "template.png"),
                             "--torchstain-prefix", str(world.inp / "torchstain"),
                             "--reference-libs", str(pin), "--slides", "S1", "--processes", "0"]
    with pytest.raises(S.EnvironmentMismatch, match="reference_libs"):
        S.fingerprint_main(a, **inject(world, probe=probe))
    assert FakeSlide.calls == [PROBE_READ]


def test_r6a_v2_and_other_schema_pins_refused(world):
    good = required_reflibs(world.tmp / "g")
    for schema in ("r2_1_perturb_suite_reference_libs_v2", "r2_1_perturb_suite_reference_libs_v1",
                   None):
        pin = world.tmp / f"pin_{schema}.json"
        pin.write_text(json.dumps({"schema": schema, "asserted": good["asserted"]}))
        with pytest.raises(S.EnvironmentMismatch, match="refused"):
            S.load_reference_libs(pin)
        with pytest.raises(S.EnvironmentMismatch, match="only r2_1_perturb_suite_reference_libs_v3"):
            S.environment_block(fake_probe(reference_libs=lambda: good),
                                torchstain_prefix=world.inp / "torchstain", reference_libs_path=pin)
    pin = world.tmp / "pin_v3.json"
    pin.write_text(json.dumps({"schema": S.REFLIBS_SCHEMA, "asserted": good["asserted"]}))
    blk = S.environment_block(fake_probe(reference_libs=lambda: good),
                              torchstain_prefix=world.inp / "torchstain", reference_libs_path=pin)
    assert blk["ok"]
    # a v2 capture cannot be pinned into v3
    cap = world.tmp / "v2_capture.json"
    cap.write_text(json.dumps({"schema": "r2_1_perturb_suite_env_capture_v2"}))
    assert S.ENV_CAPTURE_SCHEMA == "r2_1_perturb_suite_env_capture_v3"
    assert S.pin_reference_libs_main(["--out-root", str(world.tmp / "p"),
                                      "--captures", str(cap)]) == 1


@pytest.mark.parametrize("env", [{"LC_ALL": "C", "LANG": "en_US.UTF-8", "LC_CTYPE": None},
                                 {"LC_ALL": None, "LANG": "en_US.UTF-8", "LC_CTYPE": "C.UTF-8"},
                                 {"LC_ALL": "en_US.UTF-8", "LANG": None, "LC_CTYPE": None}])
def test_r6a_locale_recorded_never_asserted(world, monkeypatch, env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    S.set_single_thread()
    blk = S.environment_block(fake_probe(), torchstain_prefix=world.inp / "torchstain",
                              reference_libs_path=world.reflibs)
    assert blk["ok"] and blk["deviations"] == []
    assert blk["recorded"]["locale"] == env
    assert "locale" not in blk["asserted"]
    if env["LC_ALL"] == "C":
        assert "locale_note" not in blk["recorded"]
    else:
        assert "LC_ALL" in blk["recorded"]["locale_note"]
    # env-check passes and the capture record carries the locale
    eargs = ["--out-root", str(world.out), "--template", str(world.inp / "template.png"),
             "--torchstain-prefix", str(world.inp / "torchstain"),
             "--reference-libs", str(world.reflibs), "--probe-wsi", str(world.slide_path_of("S1"))]
    assert S.env_check_main(eargs, probe=fake_probe(), reinhard_factory=rh_factory,
                            opener=make_opener()) == 0
    good = required_reflibs(world.tmp / "g")
    assert S.capture_env_main(capture_args(world, world.out),
                              probe=fake_probe(reference_libs=lambda: good),
                              reinhard_factory=rh_factory, opener=make_opener()) == 0
    [cap] = (world.out / "environment" / "captures").glob("*.json")
    assert json.loads(cap.read_text())["environment"]["recorded"]["locale"] == env
    # capture-env needs --probe-wsi [R4]
    with pytest.raises(SystemExit):
        S.capture_env_main(capture_args(world, world.out)[:-2],
                           probe=fake_probe(reference_libs=lambda: libs),
                           reinhard_factory=rh_factory, opener=make_opener())


def test_pin_reference_libs_recorded_only_difference_does_not_refuse(world, monkeypatch):
    """[R4] captures that differ only in recorded-only objects (driver/display libraries,
    the ldd closure) pin; the per-host recorded-only sets are stored."""
    kw = dict(reinhard_factory=rh_factory, opener=make_opener())
    a = required_reflibs(world.tmp / "a", driver_names=["libcuda.so.595.58.03", "libGL.so.1"])
    b = required_reflibs(world.tmp / "a", driver_names=["libcuda.so.550.127.05", "libX11.so.6"])
    assert a["asserted"] == b["asserted"]
    b["recorded_only"]["ldd_openslide_closure"]["error"] = "SuiteError: ldd unavailable"
    monkeypatch.setenv("LSB_QUEUE", "heavy_io")
    assert S.capture_env_main(capture_args(world, world.out),
                              probe=fake_probe(reference_libs=lambda: a), **kw) == 0
    monkeypatch.setenv("LSB_QUEUE", "rhel88_gpu")
    assert S.capture_env_main(capture_args(world, world.out),
                              probe=fake_probe(hostname=lambda: "nodegpu300",
                                               reference_libs=lambda: b), **kw) == 0
    caps = sorted((world.out / "environment" / "captures").glob("*.json"))
    assert S.pin_reference_libs_main(["--out-root", str(world.out), "--captures",
                                      *map(str, caps)]) == 0
    pinned = json.loads((world.out / "environment" / "reference_libs.json").read_text())
    by_host = {c["host"]: c["recorded_only"] for c in pinned["recorded_only_by_capture"]}
    assert {e["soname"] for e in by_host[HOST]["driver_pattern_matches"]} == \
        {"libcuda.so.595.58.03", "libGL.so.1"}
    assert {e["soname"] for e in by_host["nodegpu300"]["driver_pattern_matches"]} == \
        {"libcuda.so.550.127.05", "libX11.so.6"}
    # an old-schema (candidate 001) capture is refused
    old = world.tmp / "old_capture.json"
    old.write_text(json.dumps({"schema": None, "environment": {}}))
    assert S.pin_reference_libs_main(["--out-root", str(world.tmp / "p3"), "--captures",
                                      str(old)]) == 1


# --------------------------------------------------------------------------
# Item 3: fingerprints
# --------------------------------------------------------------------------


def load_fp(w, sid, out=None):
    d = (out or w.out) / "fingerprints"
    return (np.load(d / f"{sid}.fp.npy", allow_pickle=False),
            json.loads((d / f"{sid}.fp.json").read_text()))


def test_fingerprint_arrays_order_and_digests(world):
    w = world
    assert fingerprint(w) == 0
    # [R4] one probe read of queries.csv row 0 before the maps check, then exactly X's two
    # read_region calls per tile (identity + shifted), level 0, 512x512
    assert FakeSlide.calls[0] == PROBE_READ
    assert len(FakeSlide.calls) == 1 + 2 * sum(SLIDES.values())
    for sid, n in SLIDES.items():
        arr, doc = load_fp(w, sid)
        assert arr.dtype == np.uint8 and arr.shape == (n, 13, 32)
        assert doc["schema"] == "r2_1_perturb_suite_fingerprint_v1"
        assert doc["conditions"] == list(X.CONDITIONS) + ["gaussian_blur_image"]
        assert doc["n_tiles"] == n and doc["slide_id"] == sid
        assert doc["manifest_v2_sha256"] == w.v2sha and doc["manifest_v1_sha256"] == w.v1sha
        assert doc["queries_sha256"] == w.qsha
        assert doc["extractor_code_sha256"] == X_SHA256
        assert doc["suite_code_sha256"] == S.SUITE_CODE_SHA256
        assert doc["array"]["npy_sha256"] == _sha((w.out / "fingerprints" / f"{sid}.fp.npy").read_bytes())
        assert doc["array"]["content_sha256"] == _sha(arr.tobytes())
        assert doc["wsi"]["st_size"] == w.slide_path_of(sid).stat().st_size
        assert doc["wsi"]["st_mtime_ns"] == w.slide_path_of(sid).stat().st_mtime_ns
        assert doc["blur_golden"]["ok"] is True
        assert doc["environment"]["ok"] and doc["environment"]["pinned_reinhard_fit_asserted"]
        assert doc["reinhard"]["target_means"] == PINNED_MEANS
        assert doc["reinhard_nonfinite_tile_indices"] == []
        assert doc["legacy_gaussian_blur_equals_identity"] is True
        for j, c in enumerate(S.SUITE_CONDITIONS):
            assert doc["aggregate_digests"][c] == _sha(np.ascontiguousarray(arr[:, j]).tobytes())
        # independent re-render of every tile, 13 pictures in order
        for i in range(n):
            pics = independent_pictures(w, sid, i)
            for j, pic in enumerate(pics):
                assert arr[i, j].tobytes() == hashlib.sha256(pic.tobytes()).digest(), (sid, i, j)
        gb = S.SUITE_CONDITIONS.index("gaussian_blur")
        assert np.array_equal(arr[:, gb], arr[:, 0])                  # legacy blur == identity
        assert not (arr[:, 12] == arr[:, 0]).all(axis=1).any()        # image blur != identity
        assert not (arr[:, 9] == arr[:, 0]).all(axis=1).any()         # shift differs
    runs = list((w.out / "fingerprints" / "runs").glob("*.json"))
    assert len(runs) == 1 and json.loads(runs[0].read_text())["counts"] == {
        "written": 3, "skipped": 0, "failed": 0}


def test_fingerprint_blur_applied_to_unshifted_identity(world):
    w = world
    assert fingerprint(w, slides="S2") == 0
    arr, _ = load_fp(w, "S2")
    st = stim(w, "S2")
    x, y = coords_of("S2")[0]["wsi_loc"]
    region = FakeSlide("p").read_region((x, y), 0, (512, 512)).convert("RGB")
    dx, dy = X.shift_of(st)
    shifted = FakeSlide("p").read_region((x + dx, y + dy), 0, (512, 512)).convert("RGB")
    assert arr[0, 12].tobytes() == hashlib.sha256(
        S.gaussian_blur_image(np.asarray(region)).tobytes()).digest()
    assert arr[0, 12].tobytes() != hashlib.sha256(
        S.gaussian_blur_image(np.asarray(shifted)).tobytes()).digest()


def test_fingerprint_processes_equal_serial(world):
    w = world
    assert fingerprint(w, slides="S1,S3", processes=0) == 0
    out2 = w.tmp / "suite_parallel"
    out2.mkdir()
    assert fingerprint(w, slides="S1,S3", processes=2, out=out2) == 0
    for sid in ("S1", "S3"):
        a, da = load_fp(w, sid)
        b, db = load_fp(w, sid, out2)
        assert np.array_equal(a, b) and da["aggregate_digests"] == db["aggregate_digests"]
        assert db["render_processes"] == 2


def test_fingerprint_skip_existing_validates_pair(world):
    w = world
    assert fingerprint(w, slides="S1") == 0
    npy = w.out / "fingerprints" / "S1.fp.npy"
    js = w.out / "fingerprints" / "S1.fp.json"
    before = (npy.read_bytes(), js.read_bytes())
    FakeSlide.calls = []
    assert fingerprint(w, slides="S1") == 0                       # skipped, no tile reads
    assert FakeSlide.calls == [PROBE_READ]
    assert (npy.read_bytes(), js.read_bytes()) == before
    st, why, _, _ = S.validate_fp_pair(w.out / "fingerprints", "S1", n_tiles=5,
                                       manifest_v2_sha256=w.v2sha)
    assert (st, why) == ("complete", "")
    for kw, reason in [(dict(n_tiles=4), "n_tiles"), (dict(manifest_v2_sha256="0" * 64),
                                                     "manifest_v2_sha256")]:
        args = dict(n_tiles=5, manifest_v2_sha256=w.v2sha)
        args.update(kw)
        st, why, _, _ = S.validate_fp_pair(w.out / "fingerprints", "S1", **args)
        assert st == "invalid" and reason in why


def _tamper_fp(w, sid, how):
    d = w.out / "fingerprints"
    npy, js = d / f"{sid}.fp.npy", d / f"{sid}.fp.json"
    if how == "npy_byte":
        b = bytearray(npy.read_bytes())
        b[-1] ^= 1
        npy.chmod(0o644)
        npy.write_bytes(bytes(b))
    elif how == "npy_resealed":
        arr = np.load(npy)
        arr[0, 3, 0] ^= 1
        raw = S.npy_bytes(arr)
        doc = json.loads(js.read_text())
        doc["array"].update(npy_sha256=_sha(raw), npy_bytes=len(raw))
        npy.write_bytes(raw)
        js.write_text(json.dumps(doc))
    elif how == "legacy_blur":
        arr = np.load(npy)
        arr[0, 11, 0] ^= 1
        raw = S.npy_bytes(arr)
        doc = json.loads(js.read_text())
        doc["array"].update(npy_sha256=_sha(raw), npy_bytes=len(raw),
                            content_sha256=_sha(arr.tobytes()))
        doc["aggregate_digests"] = S.aggregate_digests(arr)
        npy.write_bytes(raw)
        js.write_text(json.dumps(doc))
    elif how == "orphan":
        js.unlink()
    elif how == "json":
        js.write_text("{")


@pytest.mark.parametrize("how,reason", [("npy_byte", "npy"), ("npy_resealed", "content"),
                                        ("legacy_blur", "legacy"), ("orphan", "orphan"),
                                        ("json", "unreadable")])
def test_fingerprint_invalid_existing_pair_is_failed_not_overwritten(world, how, reason):
    w = world
    assert fingerprint(w, slides="S1") == 0
    _tamper_fp(w, "S1", how)
    d = w.out / "fingerprints"
    before = {p.name: p.read_bytes() for p in d.glob("S1.fp.*")}
    FakeSlide.calls = []
    assert fingerprint(w, slides="S1") == 1
    assert FakeSlide.calls == [PROBE_READ]
    assert {p.name: p.read_bytes() for p in d.glob("S1.fp.*")} == before
    failed = list((d / "failed").glob("S1.*.json"))
    assert len(failed) == 1
    f = json.loads(failed[0].read_text())
    assert f["stage"] == "existing_output" and reason in f["reason"]


def test_publish_is_atomic_never_overwrites_and_confined(world, tmp_path):
    root = tmp_path / "root"
    S.publish(root, root / "a" / "f.bin", b"one")
    assert (root / "a" / "f.bin").read_bytes() == b"one"
    with pytest.raises(FileExistsError):
        S.publish(root, root / "a" / "f.bin", b"two")
    assert (root / "a" / "f.bin").read_bytes() == b"one"
    assert [p.name for p in (root / "a").iterdir()] == ["f.bin"]       # no temp left
    with pytest.raises(S.SuiteError, match="outside"):
        S.publish(root, tmp_path / "elsewhere.bin", b"x")
    with pytest.raises(S.SuiteError, match="outside"):
        S.publish(root, root / ".." / "escape.bin", b"x")
    assert S.publish_idempotent(root, root / "a" / "f.bin", b"one") == "existing"
    with pytest.raises(S.SuiteError):
        S.publish_idempotent(root, root / "a" / "f.bin", b"two")
    assert not (tmp_path / "elsewhere.bin").exists() and not (tmp_path / "escape.bin").exists()


def test_fingerprint_nonfinite_reinhard_indices(world):
    w = world
    # the fake normaliser reports non-finite when the tile's first red value matches
    x, y = coords_of("S1")[2]["wsi_loc"]
    r0 = (x * 7 + y * 3) % 256
    fac = lambda t, sha, p: FakeRH(nonfinite_r0=r0)                # noqa: E731
    assert fingerprint(w, slides="S1", reinhard_factory=fac) == 0
    _, doc = load_fp(w, "S1")
    assert 2 in doc["reinhard_nonfinite_tile_indices"]


def test_fingerprint_tile_error_and_wsi_change_are_failed(world):
    w = world
    bad_loc = tuple(coords_of("S1")[3]["wsi_loc"])
    assert fingerprint(w, slides="S1", opener=make_opener(fail_at=[bad_loc])) == 1
    d = w.out / "fingerprints"
    assert not (d / "S1.fp.npy").exists() and not (d / "S1.fp.json").exists()
    f = json.loads(next((d / "failed").glob("S1.*.json")).read_text())
    assert f["stage"] == "tile_render" and f["n_failed_tiles"] == 1
    assert f["tile_errors"][0]["index"] == 3 and f["tile_errors"][0]["type"] == "OSError"

    def touching_opener(path):
        Path(path).write_bytes(b"changed during render, longer")
        return FakeSlide(path)
    assert fingerprint(w, slides="S2", opener=touching_opener) == 1
    assert not (d / "S2.fp.npy").exists()
    f = json.loads(next((d / "failed").glob("S2.*.json")).read_text())
    assert f["stage"] == "wsi_changed"


def test_fingerprint_rejects_wrong_template_digest_in_manifest(world):
    w = world
    assert S.check_template_digest(stim(w, "S1"), "S1") is None
    st = stim(w, "S1")
    st["reinhard"] = dict(st["reinhard"], parameters={"template_sha256": "0" * 64})
    with pytest.raises(S.SuiteError):
        S.check_template_digest(st, "S1")


def test_plan_tasks_packing():
    rows = [{"slide_id": s, "feature_rows": n} for s, n in
            [("a", 7000), ("b", 16000), ("c", 9000), ("d", 3000), ("e", 6000), ("f", 1000),
             ("g", 15000)]]
    tasks = S.plan_tasks(rows, 15000)
    assert [(t["slides"], t["tiles"], t["solo"]) for t in tasks] == [
        (["b"], 16000, True), (["g"], 15000, True), (["c", "e"], 15000, False),
        (["a", "d", "f"], 11000, False)]
    assert [t["task_index"] for t in tasks] == [1, 2, 3, 4]
    assert sorted(s for t in tasks for s in t["slides"]) == sorted(r["slide_id"] for r in rows)
    assert all(t["tiles"] <= 15000 or t["solo"] for t in tasks)
    assert S.FP_TASK_MAX_TILES == 15000
    # ties broken by slide id
    t2 = S.plan_tasks([{"slide_id": "z", "feature_rows": 10}, {"slide_id": "y", "feature_rows": 10}], 15)
    assert t2[0]["slides"] == ["y"] and t2[1]["slides"] == ["z"]


def test_plan_tasks_main_and_task_index(world):
    w = world
    args = ["--out-root", str(w.out), "--queries", str(w.queries), "--queries-sha256", w.qsha,
            "--max-tiles", "6"]
    assert S.plan_tasks_main(args) == 0
    assert S.plan_tasks_main(args) == 0                         # idempotent
    doc = json.loads((w.out / "fingerprints" / "TASKS.json").read_text())
    assert doc["queries_sha256"] == w.qsha and doc["total_tiles"] == 10
    assert [t["slides"] for t in doc["tasks"]] == [["S1"], ["S2", "S3"]]
    a = input_args(w) + env_args(w) + ["--tasks", str(w.out / "fingerprints" / "TASKS.json"),
                                       "--task-index", "2", "--processes", "0"]
    assert S.fingerprint_main(a, **inject(w)) == 0
    assert (w.out / "fingerprints" / "S2.fp.npy").exists()
    assert (w.out / "fingerprints" / "S3.fp.npy").exists()
    assert not (w.out / "fingerprints" / "S1.fp.npy").exists()
    with pytest.raises(S.SuiteError):
        S.fingerprint_main(input_args(w) + env_args(w) + ["--tasks", str(
            w.out / "fingerprints" / "TASKS.json"), "--task-index", "0"], **inject(w))
    with pytest.raises(S.SuiteError):
        S.fingerprint_main(input_args(w) + env_args(w) + ["--tasks", str(
            w.out / "fingerprints" / "TASKS.json"), "--task-index", "3"], **inject(w))
    with pytest.raises(S.SuiteError):
        S.fingerprint_main(input_args(w) + env_args(w) + ["--slides", "S9"], **inject(w))
    with pytest.raises(S.SuiteError):
        S.plan_tasks_main(args[:4] + ["--queries-sha256", ""])


def finalize_args(w):
    return input_args(w)


def _report_from_stdout(capsys):
    """[C4-1] adaptation: finalize prints a compact summary; the full report (candidate-003
    keys unchanged) is the published report file, whose sha256 must equal the printed one."""
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    raw = Path(summary["report"]["path"]).read_bytes()
    assert _sha(raw) == summary["report"]["sha256"] and len(raw) == summary["report"]["bytes"]
    return json.loads(raw)


def read_complete(w, out=None):
    d = (out or w.out) / "fingerprints"
    return json.loads((d / "COMPLETE").read_text())


def test_finalize_index_and_complete_rules(world):
    w = world
    assert fingerprint(w, slides="S1,S2") == 0
    assert S.finalize_main(finalize_args(w)) == 1                     # S3 missing
    assert not (w.out / "fingerprints" / "INDEX.json").exists()
    assert fingerprint(w, slides="S3") == 0
    assert S.finalize_main(finalize_args(w)) == 0
    idx = json.loads((w.out / "fingerprints" / "INDEX.json").read_text())
    assert idx["n_slides"] == 3 and idx["n_tiles"] == 10
    assert idx["cpu_models"] == [CPU_MODEL]
    assert [e["slide_id"] for e in idx["slides"]] == list(SLIDES)
    for e in idx["slides"]:
        assert e["cpu_model"] == CPU_MODEL and e["host"] == HOST       # [A5]
        assert e["reference_lib_waived"] is False                      # [R5a]
    assert idx["superseded_failures"] == [] and idx["n_reference_lib_waived"] == 0
    comp = read_complete(w)
    assert comp["schema"] == S.COMPLETE_SCHEMA and comp["index_file"] == "INDEX.json"
    assert comp["index_sha256"] == _sha((w.out / "fingerprints" / "INDEX.json").read_bytes())
    assert comp["n_slides"] == 3 and comp["superseded_failures"] == []
    assert comp["reference_lib_waivers"] == {"n_waived": 0, "slides": [], "accepted_reason": None}
    assert S.finalize_main(finalize_args(w)) == 0                      # idempotent


def test_finalize_failed_record_blocks_complete(world):
    """[R3] (rewritten from candidate 001, where ANY FAILED record blocked COMPLETE even after a
    later successful retry): a FAILED record whose slide has NO validated pair blocks and is
    reported."""
    w = world
    bad_loc = tuple(coords_of("S3")[1]["wsi_loc"])
    assert fingerprint(w, slides="S1,S2") == 0
    assert fingerprint(w, slides="S3", opener=make_opener(fail_at=[bad_loc])) == 1
    assert S.finalize_main(finalize_args(w)) == 1                      # S3 has no pair
    assert not (w.out / "fingerprints" / "INDEX.json").exists()
    assert not (w.out / "fingerprints" / "COMPLETE").exists()


def test_finalize_unreadable_or_foreign_failed_record_blocks(world, capsys):
    w = world
    assert fingerprint(w) == 0
    fdir = w.out / "fingerprints" / "failed"
    fdir.mkdir()
    (fdir / "S9.x.json").write_text(json.dumps({"slide_id": "S9", "stage": "tile_render"}))
    capsys.readouterr()
    assert S.finalize_main(finalize_args(w)) == 1
    rep = _report_from_stdout(capsys)                       # [C4-1] adapted (report file)
    assert rep["blocking_failed_records"][0]["reason"] == "slide_id not in the population"
    # [R6b] (adapted from candidate 002, which wrote INDEX.json here): no INDEX when blocked
    assert not (w.out / "fingerprints" / "INDEX.json").exists()
    assert rep["index"] is None and rep["would_be_index"]["n_slides"] == 3
    assert not (w.out / "fingerprints" / "COMPLETE").exists()
    (fdir / "S9.x.json").write_text("{")
    assert S.finalize_main(finalize_args(w)) == 1
    assert not (w.out / "fingerprints" / "COMPLETE").exists()


def test_finalize_transient_failure_then_retry_is_superseded(world):
    """[R3] transient failure -> successful retry -> COMPLETE, the FAILED record kept in place
    and listed under superseded_failures with the superseding pair's digest."""
    w = world
    bad_loc = tuple(coords_of("S3")[1]["wsi_loc"])
    assert fingerprint(w, slides="S1,S2") == 0
    assert fingerprint(w, slides="S3", opener=make_opener(fail_at=[bad_loc])) == 1
    fdir = w.out / "fingerprints" / "failed"
    [frec] = list(fdir.glob("S3.*.json"))
    frec_bytes = frec.read_bytes()
    assert fingerprint(w, slides="S3") == 0                            # later success
    assert S.finalize_main(finalize_args(w)) == 0
    assert frec.read_bytes() == frec_bytes                             # never deleted/moved
    idx = json.loads((w.out / "fingerprints" / "INDEX.json").read_text())
    comp = read_complete(w)
    _, doc = load_fp(w, "S3")
    for sup in (idx["superseded_failures"], comp["superseded_failures"]):
        assert len(sup) == 1
        e = sup[0]
        assert e["record"] == f"failed/{frec.name}" and e["slide_id"] == "S3"
        assert e["stage"] == "tile_render" and e["record_sha256"] == _sha(frec_bytes)
        assert e["superseding_pair"]["content_sha256"] == doc["array"]["content_sha256"]
        assert e["superseding_pair"]["npy_sha256"] == doc["array"]["npy_sha256"]
        assert e["superseding_pair"]["fp_json_sha256"] == _sha(
            (w.out / "fingerprints" / "S3.fp.json").read_bytes())
    assert comp["index_sha256"] == _sha((w.out / "fingerprints" / "INDEX.json").read_bytes())


@pytest.mark.parametrize("lone", ["S3.fp.json", "S3.fp.npy"])
def test_orphan_quarantine_then_rerun_then_complete(world, lone):
    """[R3] orphan -> every rerun refuses (existing behaviour) -> quarantine-orphans (move, never
    delete/overwrite, FAILED record) -> rerun produces a validated pair -> finalize COMPLETE
    with the quarantine FAILED record listed as superseded."""
    w = world
    assert fingerprint(w) == 0
    d = w.out / "fingerprints"
    partner = "S3.fp.npy" if lone == "S3.fp.json" else "S3.fp.json"
    (d / partner).unlink()                                     # crash between the two publishes
    orphan_bytes = (d / lone).read_bytes()
    for _ in range(2):
        assert fingerprint(w, slides="S3") == 1                # refuses, does not overwrite
        assert (d / lone).read_bytes() == orphan_bytes and not (d / partner).exists()
    n_failed_before = len(list((d / "failed").glob("S3.*.json")))
    assert n_failed_before == 2
    assert S.finalize_main(finalize_args(w)) == 1
    assert S.quarantine_orphans_main(["--out-root", str(w.out)]) == 0
    assert not (d / lone).exists()
    [moved] = list((d / "failed" / "quarantine").glob(f"*/{lone}"))
    assert moved.read_bytes() == orphan_bytes
    qrecs = [json.loads(f.read_text()) for f in (d / "failed").glob("S3.*.json")]
    q = [r for r in qrecs if r["stage"] == "orphan_quarantine"]
    assert len(q) == 1 and len(qrecs) == n_failed_before + 1
    assert q[0]["original_path"] == str(d / lone) and q[0]["new_path"] == str(moved)
    assert q[0]["size"] == len(orphan_bytes) and q[0]["sha256"] == _sha(orphan_bytes)
    # idempotent: nothing left to quarantine
    assert S.quarantine_orphans_main(["--out-root", str(w.out)]) == 0
    assert len(list((d / "failed" / "quarantine").glob("*/*"))) == 1
    assert fingerprint(w, slides="S3") == 0                    # rerun now succeeds
    assert S.finalize_main(finalize_args(w)) == 0
    comp = read_complete(w)
    stages = sorted(e["stage"] for e in comp["superseded_failures"])
    assert stages == ["existing_output", "existing_output", "orphan_quarantine"]
    assert all(e["slide_id"] == "S3" for e in comp["superseded_failures"])
    assert moved.read_bytes() == orphan_bytes                  # quarantined file kept


def test_quarantine_orphans_features_tree(world, tmp_path):
    w = world
    feat = w.out / "features" / ENC
    feat.mkdir(parents=True)
    (feat / "S1.pkl").write_bytes(b"pkl S1")
    (feat / "S1.record.json").write_text("{}")                  # complete pair: untouched
    (feat / "S2.pkl").write_bytes(b"lone pkl")
    (feat / "S3.record.json").write_text('{"lone": 1}')
    (feat / ".S2.record.json.123.tmp").write_text("tmp")       # in-flight temp: ignored
    assert S.quarantine_orphans_main(["--out-root", str(w.out), "--tree", "features"]) == 0
    assert (feat / "S1.pkl").exists() and (feat / "S1.record.json").exists()
    assert not (feat / "S2.pkl").exists() and not (feat / "S3.record.json").exists()
    assert (feat / ".S2.record.json.123.tmp").exists()
    q = w.out / "features" / "failed" / "quarantine"
    assert [p.read_bytes() for p in q.glob(f"*/{ENC}/S2.pkl")] == [b"lone pkl"]
    assert len(list(q.glob(f"*/{ENC}/S3.record.json"))) == 1
    recs = [json.loads(f.read_text()) for f in (w.out / "features" / "failed" / ENC).glob("*.json")]
    assert sorted(r["slide_id"] for r in recs) == ["S2", "S3"]
    assert {r["stage"] for r in recs} == {"orphan_quarantine"}
    with pytest.raises(S.SuiteError):
        S.quarantine_orphans_main(["--out-root", str(tmp_path / "nothing"), "--tree", "features"])


def test_finalize_reference_lib_waiver_needs_accept_waivers(world, capsys):
    """[R5a] waived pairs are flagged per slide in INDEX.json; COMPLETE is refused unless
    --accept-waivers <reason>, which is written into COMPLETE."""
    w = world
    assert fingerprint(w, slides="S1,S2") == 0
    a = input_args(w) + env_args(w, reflibs=False) + ["--slides", "S3", "--processes", "0",
                                                      "--allow-missing-reference-libs"]
    assert S.fingerprint_main(a, **inject(w)) == 0
    capsys.readouterr()
    assert S.finalize_main(finalize_args(w)) == 1
    rep = _report_from_stdout(capsys)                       # [C4-1] adapted (report file)
    assert rep["reference_lib_waived"] == ["S3"] and "accept-waivers" in rep["waiver_block"]
    # [R6b] (adapted from candidate 002, which read INDEX.json here): the blocked finalize
    # writes no INDEX; the would-be index is in the report
    assert not (w.out / "fingerprints" / "INDEX.json").exists()
    idx = rep["would_be_index_content"]
    assert {e["slide_id"]: e["reference_lib_waived"] for e in idx["slides"]} == {
        "S1": False, "S2": False, "S3": True}
    assert idx["n_reference_lib_waived"] == 1
    assert rep["would_be_index"]["n_reference_lib_waived"] == 1
    assert not (w.out / "fingerprints" / "COMPLETE").exists()
    with pytest.raises(S.SuiteError, match="reason"):
        S.finalize_main(finalize_args(w) + ["--accept-waivers", "  "])
    reason = "capture host unavailable; lead accepted for S3 only"
    assert S.finalize_main(finalize_args(w) + ["--accept-waivers", reason]) == 0
    comp = read_complete(w)
    assert comp["reference_lib_waivers"] == {"n_waived": 1, "slides": ["S3"],
                                             "accepted_reason": reason}
    idx_disk = json.loads((w.out / "fingerprints" / "INDEX.json").read_text())
    assert idx_disk == idx and comp["index_sha256"] == rep["would_be_index"]["sha256"]


# --------------------------------------------------------------------------
# [R6b] INDEX only with COMPLETE; [R6d] publish-conflict digests
# --------------------------------------------------------------------------


def _finalize_report(w, capsys, extra=()):
    capsys.readouterr()
    rc = S.finalize_main(finalize_args(w) + list(extra))
    return rc, _report_from_stdout(capsys)                  # [C4-1] adapted (report file)


def _waived_world(w):
    assert fingerprint(w, slides="S1,S2") == 0
    a = input_args(w) + env_args(w, reflibs=False) + ["--slides", "S3", "--processes", "0",
                                                      "--allow-missing-reference-libs"]
    assert S.fingerprint_main(a, **inject(w)) == 0


def test_r6b_blocked_finalize_writes_no_index_then_late_failure_then_accept(world, capsys):
    """Adapted from re-review 002 probe I4.index_trap: candidate 002 wrote INDEX.json on a
    waiver-blocked finalize; a later FAILED record of a paired slide changed the would-be
    index and every later finalize raised (INDEX immutable). Now no INDEX is written until
    COMPLETE, and the later finalize succeeds."""
    w = world
    _waived_world(w)
    fp = w.out / "fingerprints"
    rc, rep = _finalize_report(w, capsys)
    assert rc == 1 and rep["complete"] is False and "accept-waivers" in rep["waiver_block"]
    assert rep["index"] is None and "blocked" in rep["index_not_written"]
    assert not (fp / "INDEX.json").exists() and not (fp / "COMPLETE").exists()
    wbi = rep["would_be_index"]
    assert wbi["n_slides"] == 3 and wbi["n_superseded_failures"] == 0
    assert wbi["sha256"] == _sha(S._dump_pretty(rep["would_be_index_content"]))
    # later FAILED records for a paired slide (a transient render failure written by a
    # duplicate task, and an identical-content publish conflict) -- both supersedable
    _, s1 = load_fp(w, "S1")
    S.write_failed(w.out, fp / "failed", "S1", {"stage": "tile_render", "reason": "late"})
    S.write_failed(w.out, fp / "failed", "S1", {
        "stage": "publish", "reason": "refused to overwrite: late duplicate",
        "npy_sha256": s1["array"]["npy_sha256"], "content_sha256": s1["array"]["content_sha256"]})
    rc, rep2 = _finalize_report(w, capsys)                       # still waiver-blocked
    assert rc == 1 and not (fp / "INDEX.json").exists()
    assert rep2["would_be_index"]["sha256"] != wbi["sha256"]     # the trap: index changed
    assert rep2["would_be_index"]["n_superseded_failures"] == 2
    rc, rep3 = _finalize_report(w, capsys, ["--accept-waivers", "lead accepted S3"])
    assert rc == 0 and rep3["complete"] is True
    assert rep3["index"] == "written" and rep3["complete_status"] == "written"
    idx = json.loads((fp / "INDEX.json").read_text())
    comp = read_complete(w)
    assert comp["index_sha256"] == _sha((fp / "INDEX.json").read_bytes())
    assert idx == rep2["would_be_index_content"]
    assert sorted(e["stage"] for e in idx["superseded_failures"]) == ["publish", "tile_render"]
    # idempotent re-finalize after COMPLETE
    rc, rep4 = _finalize_report(w, capsys, ["--accept-waivers", "lead accepted S3"])
    assert rc == 0 and rep4["index"] == "existing" and rep4["complete_status"] == "existing"


def _duplicate_fingerprint(w, monkeypatch, slides="S3", **over):
    """A duplicate task that passed its in-task existence check before the other task
    published (the race), so it reaches publish and is refused."""
    with monkeypatch.context() as m:
        m.setattr(S, "validate_fp_pair", lambda *a, **k: ("absent", "", None, None))
        return fingerprint(w, slides=slides, **over)


def _publish_records(w, sid):
    return [json.loads(f.read_text()) for f in
            sorted((w.out / "fingerprints" / "failed").glob(f"{sid}.*.json"))
            if json.loads(f.read_text()).get("stage") == "publish"]


def test_r6d_identical_duplicate_is_superseded(world, monkeypatch, capsys):
    w = world
    assert fingerprint(w) == 0
    npy, js = (w.out / "fingerprints" / "S3.fp.npy"), (w.out / "fingerprints" / "S3.fp.json")
    before = (npy.read_bytes(), js.read_bytes())
    assert _duplicate_fingerprint(w, monkeypatch) == 1
    assert (npy.read_bytes(), js.read_bytes()) == before         # never overwritten
    [rec] = _publish_records(w, "S3")
    _, doc = load_fp(w, "S3")
    assert rec["npy_sha256"] == doc["array"]["npy_sha256"]
    assert rec["content_sha256"] == doc["array"]["content_sha256"]
    rc, rep = _finalize_report(w, capsys)
    assert rc == 0 and rep["blocking_failed_records"] == []
    [sup] = read_complete(w)["superseded_failures"]
    assert sup["stage"] == "publish" and sup["slide_id"] == "S3"


def test_r6d_different_content_duplicate_blocks(world, monkeypatch, capsys):
    w = world
    assert fingerprint(w) == 0
    loc = tuple(coords_of("S3")[1]["wsi_loc"])
    assert _duplicate_fingerprint(w, monkeypatch, opener=make_opener(alter=[loc])) == 1
    [rec] = _publish_records(w, "S3")
    _, doc = load_fp(w, "S3")
    assert rec["npy_sha256"] != doc["array"]["npy_sha256"]
    assert rec["content_sha256"] != doc["array"]["content_sha256"]
    rc, rep = _finalize_report(w, capsys)
    assert rc == 1 and rep["complete"] is False
    [b] = rep["blocking_failed_records"]
    assert b["reason"] == "duplicate produced different content" and b["slide_id"] == "S3"
    assert b["refused_digest"]["npy_sha256"] == rec["npy_sha256"]
    assert b["pair_digest"]["npy_sha256"] == doc["array"]["npy_sha256"]
    assert rep["superseded_failures"] == []
    fp = w.out / "fingerprints"
    assert not (fp / "COMPLETE").exists() and not (fp / "INDEX.json").exists()
    rc, _ = _finalize_report(w, capsys, ["--accept-waivers", "x"])   # waivers do not unblock
    assert rc == 1 and not (fp / "COMPLETE").exists()


@pytest.mark.parametrize("fields", [{}, {"npy_sha256": "MATCH"}, {"content_sha256": "MATCH"},
                                    {"npy_sha256": None, "content_sha256": "MATCH"}])
def test_r6d_digestless_publish_conflict_blocks(world, capsys, fields):
    """Older-format (or partial) publish-conflict records fail closed."""
    w = world
    assert fingerprint(w) == 0
    _, doc = load_fp(w, "S2")
    rec = {"stage": "publish", "reason": "refused to overwrite: old format"}
    rec.update({k: (doc["array"][k] if v == "MATCH" else v) for k, v in fields.items()})
    S.write_failed(w.out, w.out / "fingerprints" / "failed", "S2", rec)
    rc, rep = _finalize_report(w, capsys)
    assert rc == 1
    [b] = rep["blocking_failed_records"]
    assert b["reason"] == "publish conflict without digest" and b["slide_id"] == "S2"
    assert not (w.out / "fingerprints" / "COMPLETE").exists()
    assert not (w.out / "fingerprints" / "INDEX.json").exists()


# --------------------------------------------------------------------------
# [C4-1] finalize report file + compact summary; INDEX/COMPLETE bytes unchanged from 003
# --------------------------------------------------------------------------

C4_SUMMARY_KEYS = {"complete", "rc", "report", "n_slides", "n_problems", "n_blocking",
                   "blocking_failed_records", "n_superseded_failures", "n_reference_lib_waived"}
C4_REPORT_NAME = re.compile(r"\d{8}T\d{6}Z_\d+_\d{9}\.json")
C003_SUITE_SHA256 = "cc023861719142aeb8993fb8ae4072a607bac2914386043f81c28ad443949879"


def _summary(capsys):
    lines = capsys.readouterr().out.strip().splitlines()
    return json.loads(lines[-1])


def _digest(path):
    raw = Path(path).read_bytes()
    return {"path": str(path), "sha256": _sha(raw), "bytes": len(raw)}


def test_c4_1_report_file_written_sha_equals_printed_and_never_overwritten(world, capsys):
    w = world
    fp = w.out / "fingerprints"
    rdir = fp / "finalize_reports"
    assert fingerprint(w, slides="S1,S2") == 0
    capsys.readouterr()
    assert S.finalize_main(finalize_args(w)) == 1                     # S3 has no pair
    s = _summary(capsys)
    assert set(s) == C4_SUMMARY_KEYS | {"would_be_index"}
    assert s["complete"] is False and s["rc"] == 1 and s["n_slides"] == 3
    assert s["n_problems"] == 1 and s["n_blocking"] == 0 and s["blocking_failed_records"] == []
    assert s["would_be_index"] is None                                # no index computable
    r1 = Path(s["report"]["path"])
    assert r1.parent == rdir and C4_REPORT_NAME.fullmatch(r1.name)
    assert _digest(r1) == s["report"]
    rep = json.loads(r1.read_bytes())
    assert rep["report_schema"] == S.FINALIZE_REPORT_SCHEMA and rep["rc"] == 1
    assert rep["complete"] is False and rep["index"] is None
    assert [p["slide_id"] for p in rep["problems"]] == ["S3"]
    assert "index_published" not in rep and "complete_published" not in rep
    assert rep["manifest_v2_sha256"] == w.v2sha and rep["queries_sha256"] == w.qsha
    assert not (fp / "INDEX.json").exists() and not (fp / "COMPLETE").exists()

    assert fingerprint(w, slides="S3") == 0
    capsys.readouterr()
    assert S.finalize_main(finalize_args(w)) == 0
    s2 = _summary(capsys)
    assert set(s2) == C4_SUMMARY_KEYS | {"index"}
    assert s2["complete"] is True and s2["rc"] == 0 and s2["n_blocking"] == 0
    assert s2["index"] == _digest(fp / "INDEX.json")
    r2 = Path(s2["report"]["path"])
    assert r2.parent == rdir and r2 != r1 and _digest(r2) == s2["report"]
    rep2 = json.loads(r2.read_bytes())
    assert rep2["report_schema"] == S.FINALIZE_REPORT_SCHEMA and rep2["rc"] == 0
    assert rep2["complete"] is True and rep2["index"] == "written"
    assert rep2["complete_status"] == "written"
    assert rep2["index_published"] == s2["index"]                     # both published digests
    assert rep2["complete_published"] == _digest(fp / "COMPLETE")
    assert "would_be_index_content" not in rep2
    # order INDEX -> COMPLETE -> report
    assert ((fp / "INDEX.json").stat().st_mtime_ns <= (fp / "COMPLETE").stat().st_mtime_ns
            <= r2.stat().st_mtime_ns)
    # idempotent rerun: a new uniquely named report; earlier reports untouched
    before = {p: p.read_bytes() for p in rdir.iterdir()}
    capsys.readouterr()
    assert S.finalize_main(finalize_args(w)) == 0
    s3 = _summary(capsys)
    r3 = Path(s3["report"]["path"])
    assert r3 not in before and _digest(r3) == s3["report"]
    assert {p: p.read_bytes() for p in rdir.iterdir() if p != r3} == before
    assert s3["index"] == s2["index"]
    rep3 = json.loads(r3.read_bytes())
    assert rep3["index"] == "existing" and rep3["complete_status"] == "existing"
    assert rep3["complete_published"] == rep2["complete_published"]


def test_c4_1_compact_print_complete_with_60_superseded_and_60_blocking(world, capsys):
    """Re-review 003 R-3: the candidate-003 print was cut at 20,000 characters and could become
    unparseable. The summary carries the COMPLETE blocking list and stays parseable."""
    w = world
    fp = w.out / "fingerprints"
    assert fingerprint(w) == 0
    _, s1 = load_fp(w, "S1")
    pad = "x" * 300
    for i in range(60):                                   # paired slide -> superseded
        S.write_failed(w.out, fp / "failed", "S1", {"stage": "tile_render", "reason": f"{i}{pad}"})
    for i in range(30):                                   # foreign slide -> blocking
        S.write_failed(w.out, fp / "failed", "S9", {"stage": "render", "reason": f"{i}{pad}"})
    for i in range(30):                                   # [R6d] different digest -> blocking
        S.write_failed(w.out, fp / "failed", "S2", {
            "stage": "publish", "reason": f"refused {i}{pad}",
            "npy_sha256": f"{i:064x}", "content_sha256": f"{i + 100:064x}"})
    capsys.readouterr()
    assert S.finalize_main(finalize_args(w)) == 1
    out = capsys.readouterr().out
    line = out.strip().splitlines()[-1]
    s = json.loads(line)                                   # parseable, never cut
    rep = json.loads(Path(s["report"]["path"]).read_bytes())
    assert _digest(s["report"]["path"]) == s["report"]
    full_003_print = json.dumps({k: v for k, v in rep.items()
                                 if k not in ("report_schema", "rc", "written_at",
                                              "suite_code_sha256", "manifest_v2_sha256",
                                              "queries_sha256", "accept_waivers")},
                                sort_keys=True, default=str)
    assert len(full_003_print) > 20000                    # 003 would have cut this
    assert s["n_blocking"] == 60 and len(s["blocking_failed_records"]) == 60
    assert s["blocking_failed_records"] == rep["blocking_failed_records"]
    assert s["n_superseded_failures"] == 60 == len(rep["superseded_failures"])
    reasons = sorted(b["reason"] for b in s["blocking_failed_records"])
    assert reasons == ["duplicate produced different content"] * 30 + \
        ["slide_id not in the population"] * 30
    written = {f"failed/{f.name}" for f in (fp / "failed").glob("S[29].*.json")}
    assert {b["record"] for b in s["blocking_failed_records"]} == written
    assert s["would_be_index"] == {k: rep["would_be_index"][k]
                                   for k in ("sha256", "bytes", "n_slides", "n_tiles")}
    assert s["would_be_index"]["n_slides"] == 3 and s["n_problems"] == 0
    assert _sha(S._dump_pretty(rep["would_be_index_content"])) == s["would_be_index"]["sha256"]
    assert "would_be_index_content" not in s
    assert not (fp / "INDEX.json").exists() and not (fp / "COMPLETE").exists()


def test_c4_1_report_write_failure_is_suite_error(world, capsys):
    w = world
    fp = w.out / "fingerprints"
    assert fingerprint(w, slides="S1,S2") == 0
    (fp / "finalize_reports").write_text("an obstacle, not a directory")
    capsys.readouterr()
    with pytest.raises(S.SuiteError, match="nothing else was written"):
        S.finalize_main(finalize_args(w))                             # blocked branch
    assert capsys.readouterr().out == ""
    assert not (fp / "INDEX.json").exists() and not (fp / "COMPLETE").exists()
    assert fingerprint(w, slides="S3") == 0
    capsys.readouterr()
    with pytest.raises(S.SuiteError, match="OR-8"):
        S.finalize_main(finalize_args(w))                             # complete branch
    assert capsys.readouterr().out == ""
    idx = (fp / "INDEX.json").read_bytes()
    assert json.loads((fp / "COMPLETE").read_text())["index_sha256"] == _sha(idx)
    (fp / "finalize_reports").unlink()                                # obstacle removed
    assert S.finalize_main(finalize_args(w)) == 0
    s = _summary(capsys)
    assert s["index"] == _digest(fp / "INDEX.json")
    assert json.loads(Path(s["report"]["path"]).read_bytes())["index"] == "existing"


def _c003_index_and_complete(w, accept_waivers=None):
    """Transcription of candidate 003's INDEX/COMPLETE construction (finalize_main of suite
    91b26c32..., no blocking records): the byte-identity reference for [C4-1]."""
    fp_dir = w.out / "fingerprints"
    m = S.load_manifest_v2(w.v2, w.v2sha, w.v1, w.v1sha)
    qsha, rows = X.load_queries(w.queries, w.qsha)
    entries, by_sid = [], {}
    for r in rows:
        sid = r["slide_id"]
        st, _, doc, _ = S.validate_fp_pair(fp_dir, sid, n_tiles=int(r["feature_rows"]),
                                           manifest_v2_sha256=m.sha256)
        assert st == "complete"
        rec = doc["environment"]["recorded"]
        e = {"slide_id": sid, "n_tiles": doc["n_tiles"],
             "content_sha256": doc["array"]["content_sha256"],
             "npy_sha256": doc["array"]["npy_sha256"],
             "fp_json_sha256": X.sha256_file(fp_dir / f"{sid}.fp.json"),
             "aggregate_digests": doc["aggregate_digests"],
             "cpu_model": rec.get("cpu_model"), "host": rec.get("hostname"),
             "lsf_job": rec.get("lsf_job"), "lsf_index": rec.get("lsf_index"),
             "reinhard_nonfinite_tile_indices": doc["reinhard_nonfinite_tile_indices"],
             "reference_lib_waived": S._env_waived(doc)}
        entries.append(e)
        by_sid[sid] = e
    superseded = []
    for f in sorted((fp_dir / "failed").glob("*.json")) if (fp_dir / "failed").is_dir() else []:
        fd = json.loads(f.read_text())
        e = by_sid[fd["slide_id"]]
        if fd.get("stage") == "publish":
            assert {k: fd.get(k) for k in ("npy_sha256", "content_sha256")} == \
                {k: e[k] for k in ("npy_sha256", "content_sha256")}
        superseded.append({"record": str(f.relative_to(fp_dir)), "record_sha256": X.sha256_file(f),
                           "slide_id": fd["slide_id"], "stage": fd.get("stage"),
                           "failure_time": fd.get("time"),
                           "superseding_pair": {"content_sha256": e["content_sha256"],
                                                "npy_sha256": e["npy_sha256"],
                                                "fp_json_sha256": e["fp_json_sha256"]}})
    waived = [e["slide_id"] for e in entries if e["reference_lib_waived"]]
    index = {"schema": "r2_1_perturb_suite_fingerprint_index_v2",
             "manifest_v2_sha256": m.sha256, "queries_sha256": qsha,
             "conditions": list(S.SUITE_CONDITIONS), "n_slides": len(entries),
             "n_tiles": sum(e["n_tiles"] for e in entries),
             "cpu_models": sorted({str(e["cpu_model"]) for e in entries}),
             "n_reference_lib_waived": len(waived), "superseded_failures": superseded,
             "slides": entries}
    index_bytes = S._dump_pretty(index)
    complete = {"schema": "r2_1_perturb_suite_fingerprint_complete_v2", "index_file": "INDEX.json",
                "index_sha256": _sha(index_bytes), "n_slides": len(entries),
                "superseded_failures": superseded,
                "reference_lib_waivers": {"n_waived": len(waived), "slides": waived,
                                          "accepted_reason": accept_waivers if waived else None}}
    return index_bytes, S._dump_pretty(complete)


def _c4_shared_tree(w, monkeypatch, waived):
    """Fixture tree with superseded tile_render and identical-duplicate publish records and
    (optionally) a reference-library-waived pair."""
    bad_loc = tuple(coords_of("S3")[1]["wsi_loc"])
    assert fingerprint(w, slides="S1,S2") == 0
    assert fingerprint(w, slides="S3", opener=make_opener(fail_at=[bad_loc])) == 1
    if waived:
        a = input_args(w) + env_args(w, reflibs=False) + ["--slides", "S3", "--processes", "0",
                                                          "--allow-missing-reference-libs"]
        assert S.fingerprint_main(a, **inject(w)) == 0
    else:
        assert fingerprint(w, slides="S3") == 0
    assert _duplicate_fingerprint(w, monkeypatch, slides="S1") == 1
    return ["--accept-waivers", "lead accepted S3"] if waived else []


@pytest.mark.parametrize("waived", [False, True])
def test_c4_1_index_complete_bytes_identical_to_candidate_003(world, monkeypatch, capsys, waived):
    w = world
    extra = _c4_shared_tree(w, monkeypatch, waived)
    capsys.readouterr()
    assert S.finalize_main(finalize_args(w) + extra) == 0
    s = _summary(capsys)
    fp = w.out / "fingerprints"
    idx, comp = (fp / "INDEX.json").read_bytes(), (fp / "COMPLETE").read_bytes()
    ref_idx, ref_comp = _c003_index_and_complete(w, extra[1] if extra else None)
    assert idx == ref_idx and comp == ref_comp
    stages = sorted(e["stage"] for e in json.loads(comp)["superseded_failures"])
    assert stages == ["publish", "tile_render"]
    assert s["n_superseded_failures"] == 2 and s["n_reference_lib_waived"] == int(waived)
    assert s["index"]["sha256"] == _sha(idx)


@pytest.mark.skipif(not os.environ.get("PS1_C003_SUITE_CLI"),
                    reason="set PS1_C003_SUITE_CLI to the candidate-003 module snapshot")
@pytest.mark.parametrize("waived", [False, True])
def test_c4_1_index_complete_bytes_identical_running_candidate_003(world, monkeypatch, capsys,
                                                                   waived):
    """Evidence run: the frozen candidate-003 module itself (sha 91b26c32...) and this module
    finalize copies of the same fixture tree; INDEX.json and COMPLETE must be byte-identical."""
    import importlib.util
    src = Path(os.environ["PS1_C003_SUITE_CLI"])
    assert _sha(src.read_bytes()) == C003_SUITE_SHA256
    name = "prosper.evaluation._ps1_candidate003_suite_cli"
    spec = importlib.util.spec_from_file_location(name, src)
    c003 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, c003)
    spec.loader.exec_module(c003)
    w = world
    extra = _c4_shared_tree(w, monkeypatch, waived)
    out_a, out_b = w.tmp / "c003", w.tmp / "c004"
    shutil.copytree(w.out, out_a, symlinks=True)
    shutil.copytree(w.out, out_b, symlinks=True)
    capsys.readouterr()
    assert c003.finalize_main(input_args(w, out_a) + extra) == 0
    assert S.finalize_main(input_args(w, out_b) + extra) == 0
    capsys.readouterr()
    for f in ("INDEX.json", "COMPLETE"):
        a = (out_a / "fingerprints" / f).read_bytes()
        b = (out_b / "fingerprints" / f).read_bytes()
        assert a == b, f
        with capsys.disabled():
            print(f"[c4-1 byte identity] waived={waived} {f} sha256={_sha(a)} bytes={len(a)}")
    assert not (out_a / "fingerprints" / "finalize_reports").exists()      # 003: print only
    assert len(list((out_b / "fingerprints" / "finalize_reports").iterdir())) == 1


# --------------------------------------------------------------------------
# Item 4: reference PNGs
# --------------------------------------------------------------------------


def test_reference_png(world):
    w = world
    assert fingerprint(w) == 0
    args = input_args(w) + env_args(w) + ["--slides", "S1,S3"]
    with pytest.raises(S.SuiteError):
        S.reference_png_main(input_args(w) + env_args(w), **inject(w))
    assert S.reference_png_main(args, **inject(w)) == 0
    for sid, n_png in (("S1", 4 * 13), ("S3", 2 * 13)):
        d = w.out / "reference_png" / sid
        idx = json.loads((d / "INDEX.json").read_text())
        assert len(idx["pictures"]) == n_png == len(list(d.glob("*.png")))
        arr, _ = load_fp(w, sid)
        for e in idx["pictures"]:
            data = (d / e["png_file"]).read_bytes()
            assert _sha(data) == e["png_sha256"]
            with Image.open(io.BytesIO(data)) as im:
                assert im.mode == "RGB"
                raw = im.tobytes()
            j = S.SUITE_CONDITIONS.index(e["condition"])
            assert hashlib.sha256(raw).digest() == arr[e["tile_index"], j].tobytes()
    FakeSlide.calls = []
    assert S.reference_png_main(args, **inject(w)) == 0              # skipped after validation
    assert FakeSlide.calls == [PROBE_READ]
    p = w.out / "reference_png" / "S1" / "0_identity.png"
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(S.SuiteError):
        S.reference_png_main(args, **inject(w))
    # a re-render that disagrees with the stored fingerprint publishes nothing
    alt = make_opener(alter=[tuple(coords_of("S2")[0]["wsi_loc"])])
    with pytest.raises(S.SuiteError, match="fingerprint"):
        S.reference_png_main(input_args(w) + env_args(w) + ["--slides", "S2"],
                             **inject(w, opener=alt))
    assert not (w.out / "reference_png" / "S2").exists()


# --------------------------------------------------------------------------
# Item 5: pilot slide selection [A5]
# --------------------------------------------------------------------------


def _rec(sid, host, lsf, means, n):
    return {"slide_id": sid, "host": host, "lsf_job": lsf, "n_tiles": n,
            "reinhard": {"target_means": means, "target_stds": PINNED_STDS}}


def test_select_pilot_grouping_fill_and_cpu_models():
    P, D = PINNED_MEANS, DIRECT_MEANS
    records = [_rec("A0", "hA", "1", P, 100), _rec("A1", "hA", "1", P, 600),
               _rec("A2", "hA", "2", P, 700), _rec("B1", "hB", "3", P, 520),
               _rec("D1", "hB", None, D, 800), _rec("D2", "hB", None, D, 400),
               _rec("C1", "hC", "4", P, 100)]
    records += [_rec(f"F{k}", "hA", "9", P, 1000) for k in range(1, 10)]
    tags = {"hA": {"driver_tags": ["driver_595"]}, "hB": {"driver_tags": ["driver_580"]},
            "hC": {"driver_tags": []}}
    index = {"slides": [{"slide_id": r["slide_id"], "n_tiles": r["n_tiles"], "cpu_model": "EPYC 7713",
                         "host": "c1"} for r in records]
             + [{"slide_id": "Z0", "n_tiles": 100, "cpu_model": "EPYC 9654", "host": "c2"},
                {"slide_id": "Z1", "n_tiles": 900, "cpu_model": "EPYC 9654", "host": "c2"}]}
    sel = S.select_pilot(records, tags, index)
    g = {(x["host"], x["run"], x["fit"], x["driver_tag"]): x["selected"] for x in sel["groups"]}
    assert g == {("hA", "lsf", "pinned_1thread", "driver_595"): "A1",
                 ("hB", "direct", "direct_121thread", "driver_580"): "D1",
                 ("hB", "lsf", "pinned_1thread", "driver_580"): "B1",
                 ("hC", "lsf", "pinned_1thread", "none"): None}
    assert [x["host"] for x in sel["uncovered_groups"]] == ["hC"]
    ids = [e["slide_id"] for e in sel["slides"]]
    assert ids == sorted(["A1", "D1", "B1", "A2", "F1", "F2", "F3", "F4", "F5", "F6", "Z1"])
    by = {e["slide_id"]: e for e in sel["slides"]}
    assert by["Z1"]["reasons"] == ["cpu_model:EPYC 9654"]
    assert by["Z1"]["has_first_run_record"] is False and by["Z1"]["first_run_host"] is None
    assert by["A2"]["reasons"] == ["fill"] and by["D1"]["reasons"][0].startswith("group:hB|direct")
    assert sel["cpu_models"] == ["EPYC 7713", "EPYC 9654"]
    assert sel["driver_tags"] == ["driver_580", "driver_595", "none"]
    assert all(e["n_tiles"] >= 500 for e in sel["slides"])
    with pytest.raises(S.SuiteError, match="driver tag"):
        S.select_pilot(records, {"hA": tags["hA"], "hB": tags["hB"]}, index)


def test_select_pilot_and_host_tags_mains(world):
    w = world
    fr = w.tmp / "fr"
    d = fr / "features" / ENC
    d.mkdir(parents=True)
    for sid, host in (("S1", "hA"), ("S2", "hB")):
        r = _rec(sid, host, "1", PINNED_MEANS, 600)
        r.update(schema="r2_1_perturb_record_v1")
        (d / f"{sid}.record.json").write_text(json.dumps(r))
    (d / "S3.record.json").write_text("{")
    seen = []

    def host_info(h):
        seen.append(h)
        return {"model": "Opteron140", "resources": ["driver_595"], "driver_tags": ["driver_595"]}
    assert S.capture_host_tags_main(["--out-root", str(w.out), "--first-run-root", str(fr)],
                                    host_info=host_info) == 0
    assert sorted(seen) == ["hA", "hB"]
    ht = w.out / "environment" / "host_driver_tags.json"
    idx = w.tmp / "INDEX.json"
    idx.write_text(json.dumps({"slides": [{"slide_id": "S1", "n_tiles": 600, "cpu_model": "m",
                                           "host": "c"}]}))
    assert S.select_pilot_main(["--out-root", str(w.out), "--first-run-root", str(fr),
                                "--host-tags", str(ht), "--index", str(idx)]) == 0
    doc = json.loads((w.out / "pilot_bit_identity" / "PILOT_SLIDES.json").read_text())
    assert [e["slide_id"] for e in doc["slides"]] == ["S1", "S2"]
    assert doc["invalid_records"][0]["slide_id"] == "S3"
    assert doc["index_sha256"] == _sha(idx.read_bytes())


# --------------------------------------------------------------------------
# Item 5a: in-line re-render vs fingerprints
# --------------------------------------------------------------------------


def pilot_render(w, slides, **over):
    return S.pilot_render_main(input_args(w) + env_args(w) + ["--slides", slides, "--device", "cpu",
                                                             "--processes", "0"], **inject(w, **over))


def _last_report(d):
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
    return json.loads(files[-1].read_text())


def test_pilot_render_pass_and_single_tile_difference(world):
    w = world
    assert fingerprint(w) == 0
    assert S.reference_png_main(input_args(w) + env_args(w) + ["--slides", "S1"],
                                **inject(w)) == 0
    assert pilot_render(w, "S1,S2") == 0
    rep = _last_report(w.out / "pilot_bit_identity" / "render")
    assert rep["pass"] and rep["blur_golden"]["ok"] and rep["gpu"]["device"] == "cpu"
    assert all(r["exact_all_13"] for r in rep["slides"].values())
    assert set(rep["slides"]["S1"]["comparison"]) == set(S.SUITE_CONDITIONS)
    alt = make_opener(alter=[tuple(coords_of("S1")[0]["wsi_loc"])])
    assert pilot_render(w, "S1", opener=alt) == 1
    rep = _last_report(w.out / "pilot_bit_identity" / "render")
    cmp_ = rep["slides"]["S1"]["comparison"]
    for c in S.SUITE_CONDITIONS:
        want = 0 if c == "spatial_shift" else 1
        assert cmp_[c]["differing_tiles"] == want, c
        if want:
            assert cmp_[c]["first_differing"] == [0]
    # [D5] max-abs is over the saved reference PNG tiles only
    assert cmp_["identity"]["max_abs_pixel_difference_saved_tiles_only"] > 0
    assert cmp_["spatial_shift"]["max_abs_pixel_difference_saved_tiles_only"] is None
    assert "max_abs_pixel_difference" not in cmp_["identity"]
    assert not rep["pass"]
    assert rep["hf_cache_disclosure"]["hf_cache_read_outside_out_root"] is False


# --------------------------------------------------------------------------
# First-run outputs produced by the frozen X.process_slide (for pilot b / extension)
# --------------------------------------------------------------------------


def build_first_run(w, root, slides=("S1", "S2"), num_workers=0):
    _, manifest = X.load_manifest(w.v1, w.v1sha)
    qsha, rows = X.load_queries(w.queries, w.qsha)
    S.set_single_thread()
    rh = FakeRH()
    ctx = X.RunContext(run_root=root, manifest_path=w.v1, manifest_sha256=w.v1sha,
                       queries_path=w.queries, queries_sha256=qsha, device="cpu",
                       forward_batch=256, loader_batch=8, num_workers=num_workers,
                       prefetch_factor=2,
                       reinhard=rh, reinhard_info=rh.info, opener=make_opener(),
                       slide_path_of=w.slide_path_of, limit_tiles=0, job_meta=X._job_meta("cpu"))
    enc = {ENC: fake_factory(ENC, "cpu", X.AMP_DEFAULTS[ENC])}
    for r in rows:
        if r["slide_id"] in slides:
            assert X.process_slide(ctx, r, manifest[r["slide_id"]], enc) == {ENC: "written"}
    return root


def rewrite_first_run(root, sid, *, lsf_job, threads, means, stds=PINNED_STDS, perturb=(),
                      pickle_bytes_only=False):
    d = root / "features" / ENC
    recp, pkl = d / f"{sid}.record.json", d / f"{sid}.pkl"
    rec = json.loads(recp.read_text())
    feats, coords = pickle.loads(pkl.read_bytes())
    for arm in perturb:
        feats[arm] = feats[arm].copy()
        feats[arm][0, 0] += 1.0
    data = pickle.dumps((feats, coords), protocol=4)
    rec["lsf_job"] = lsf_job
    rec["torch_backend_flags"]["main_process_num_threads"] = threads
    rec["reinhard"] = dict(rec["reinhard"], target_means=list(means), target_stds=list(stds))
    if not pickle_bytes_only:
        rec["pickle_sha256"], rec["pickle_bytes"] = _sha(data), len(data)
    pkl.write_bytes(data)
    recp.write_text(json.dumps(rec, indent=1, sort_keys=True))


def standard_first_run(w, root, slides=("S1", "S2"), s1_perturb=(), s2_perturb=("reinhard",),
                       num_workers=0):
    build_first_run(w, root, slides, num_workers=num_workers)
    if "S1" in slides:
        rewrite_first_run(root, "S1", lsf_job="123", threads=8, means=PINNED_MEANS,
                          perturb=s1_perturb)
    if "S2" in slides:   # a 655-style direct-run record (G3): direct fit, no LSF, 121 threads
        rewrite_first_run(root, "S2", lsf_job=None, threads=121, means=DIRECT_MEANS,
                          perturb=s2_perturb)
    return root


# --------------------------------------------------------------------------
# Item 5b: frozen X.process_slide pilot [A1]
# --------------------------------------------------------------------------


PILOT_RUN = f"driver_595_{HOST}_t1"          # <driver_tag>_<host>_<invocation_tag> [R1]


def pilot_b(w, fr, out, slides="S1,S2", extra=(), tag="t1", factory=fake_factory, opener=None,
            probe=None):
    a = ["--out-root", str(out), "--v1", str(w.v1), "--v1-sha256", w.v1sha, "--queries",
         str(w.queries), "--queries-sha256", w.qsha] + env_args(w) + [
        "--slides", slides, "--first-run-root", str(fr), "--encoder", ENC, "--device", "cpu",
        *(["--invocation-tag", tag] if tag is not None else []), *extra]
    return S.pilot_process_slide_main(a, encoder_factory=factory,
                                      opener=opener or make_opener(),
                                      slide_path_of=w.slide_path_of, reinhard_factory=rh_factory,
                                      probe=probe or fake_probe())


def pilot_reports(out):
    return sorted((out / "pilot_bit_identity" / "process_slide_reports").glob("*.json"))


def test_pilot_process_slide_pass_lsf_exact_direct_reinhard_allowed(world, monkeypatch):
    w = world
    monkeypatch.setenv("HF_HUB_OFFLINE", "placeholder")
    monkeypatch.delenv("HF_HUB_OFFLINE")                  # [R1] setdefault -> "1"
    fr = standard_first_run(w, w.tmp / "first_run", num_workers=7)
    out = w.tmp / "pilot_out"
    assert pilot_b(w, fr, out) == 0
    rep = _last_report(out / "pilot_bit_identity" / "process_slide_reports")
    assert rep["pass"] and rep["run_context"]["forward_batch"] == 256
    assert rep["run_context"]["num_workers"] == 7 and rep["run_context"]["loader_batch"] == 8
    assert rep["hf_hub_offline"] == "1" and rep["hf_cache_disclosure"]["HF_HUB_OFFLINE"] == "1"
    assert rep["hf_cache_disclosure"]["hf_cache_read_outside_out_root"] is True
    assert rep["invocation"] == {
        "tag": "t1", "driver_tag": "driver_595", "host": HOST, "run_name": PILOT_RUN,
        "run_roots": {"process_slide": str(out / "pilot_bit_identity" / "process_slide" / PILOT_RUN)}}
    s1 = rep["slides"]["S1"]
    assert s1["first_run"]["class"] == "lsf" and s1["first_run"]["fit"] == "pinned_1thread"
    assert s1["failures"] == [] and s1["pass"]
    run = s1["runs"]["process_slide"]
    assert run["status"] == "written"
    # fresh encode: X._forward runs once per condition for the slide's one tile batch
    assert run["forward_calls"] == len(X.CONDITIONS)
    assert run["forward_images"] == len(X.CONDITIONS) * SLIDES["S1"]
    v = run["vs_first_run"]
    assert v["coords_equal"] and all(a["exact"] for a in v["arms"].values())
    assert s1["weights_check"]["equal"] and s1["weights_check"]["first_run"] == FAKE_WEIGHTS
    assert s1["loader_check"]["equal"]
    assert s1["loader_check"]["first_run"] == {"forward_batch": 256, "loader_batch": 8,
                                               "num_workers": 7, "code_sha256": X_SHA256}
    s2 = rep["slides"]["S2"]
    assert s2["first_run"]["class"] == "direct"
    assert s2["first_run"]["predicate"] == {"means_direct": True, "lsf_job_null": True,
                                            "threads_121": True}
    v = s2["runs"]["process_slide"]["vs_first_run"]
    assert v["arms"]["reinhard"]["exact"] is False and v["unexpected_non_exact_arms"] == []
    assert all(a["exact"] for c, a in v["arms"].items() if c != "reinhard")
    # pilot outputs keep X's schema under pilot_bit_identity/process_slide/<run name>/
    pr = json.loads((out / "pilot_bit_identity" / "process_slide" / PILOT_RUN / "features" / ENC /
                     "S1.record.json").read_text())
    assert pr["schema"] == "r2_1_perturb_record_v1" and pr["num_workers"] == 7
    assert pr["hf_hub_offline"] == "1"
    assert not list(fr.rglob("*.tmp")) and not (fr / "pilot_bit_identity").exists()


def test_pilot_process_slide_detects_non_exact_arm_and_twice(world):
    w = world
    fr = standard_first_run(w, w.tmp / "first_run", slides=("S1",), s1_perturb=("rotate_90",),
                            num_workers=7)
    out = w.tmp / "pilot_out"
    assert pilot_b(w, fr, out, slides="S1", extra=["--twice"]) == 1
    rep = _last_report(out / "pilot_bit_identity" / "process_slide_reports")
    s1 = rep["slides"]["S1"]
    for label in ("process_slide", "twice_run1", "twice_run2"):
        assert s1["runs"][label]["vs_first_run"]["unexpected_non_exact_arms"] == ["rotate_90"]
        assert s1["runs"][label]["forward_calls"] >= 1
    assert s1["failures"] == ["process_slide:arms_not_exact", "twice_run1:arms_not_exact",
                              "twice_run2:arms_not_exact"]
    assert all(a["exact"] for a in s1["twice_run1_vs_run2"]["arms"].values())
    assert rep["invocation"]["run_roots"]["twice_run1"] == str(
        out / "pilot_bit_identity" / "twice" / PILOT_RUN / "run1")
    # an absent first-run record fails the pilot
    assert pilot_b(w, fr, w.tmp / "pilot_out2", slides="S3") == 1
    rep = _last_report(w.tmp / "pilot_out2" / "pilot_bit_identity" / "process_slide_reports")
    assert "first_run_absent" in rep["slides"]["S3"]["failures"]


def test_pilot_process_slide_refuses_rerun_and_detects_perturbed_encoder(world):
    """[R1] I1 regression: (1) a second invocation with the same tag on the same root is
    refused before anything is written (the stale outputs would make X skip the slide);
    (2) a new invocation tag with a perturbed encoder (+1.0) encodes fresh, finds
    non-identity and fails."""
    w = world
    fr = standard_first_run(w, w.tmp / "first_run", num_workers=7)
    out = w.tmp / "pilot_out"
    assert pilot_b(w, fr, out) == 0
    n_rep = len(pilot_reports(out))
    run_dir = out / "pilot_bit_identity" / "process_slide" / PILOT_RUN
    before = sorted(str(q.relative_to(out)) for q in out.rglob("*"))
    with pytest.raises(S.SuiteError, match="refusing to start"):
        pilot_b(w, fr, out, factory=perturbed_factory)
    assert sorted(str(q.relative_to(out)) for q in out.rglob("*")) == before
    assert len(pilot_reports(out)) == n_rep
    # also refused when only one of the selected slides has outputs, or only failures.jsonl
    with pytest.raises(S.SuiteError, match="refusing to start"):
        pilot_b(w, fr, out, slides="S2")
    FORWARD_CALLS.clear()
    assert pilot_b(w, fr, out, tag="t2", factory=perturbed_factory) == 1
    rep = _last_report(out / "pilot_bit_identity" / "process_slide_reports")
    assert rep["invocation"]["run_name"] == f"driver_595_{HOST}_t2"
    for sid in ("S1", "S2"):
        s_ = rep["slides"][sid]
        run = s_["runs"]["process_slide"]
        assert run["status"] == "written" and run["forward_calls"] >= 1
        assert s_["failures"] == ["process_slide:arms_not_exact"] and not s_["pass"]
        v = run["vs_first_run"]
        assert v["coords_equal"]
        assert all(not a["exact"] and a["max_abs_difference"] > 0.99
                   for a in v["arms"].values())
    assert FORWARD_CALLS                                    # the encoder really ran
    assert not rep["pass"] and run_dir.is_dir()


def test_pilot_process_slide_skipped_is_not_encoded(world, monkeypatch):
    """[R1] a 'skipped' process_slide status (no fresh encode) fails the slide."""
    w = world
    fr = standard_first_run(w, w.tmp / "first_run", slides=("S1",), num_workers=7)
    real = X.process_slide
    monkeypatch.setattr(X, "process_slide", lambda ctx, row, st, enc: {ENC: "skipped"})
    out = w.tmp / "pilot_out"
    assert pilot_b(w, fr, out, slides="S1") == 1
    rep = _last_report(out / "pilot_bit_identity" / "process_slide_reports")
    s1 = rep["slides"]["S1"]
    assert s1["runs"]["process_slide"]["status"] == "skipped"
    assert s1["runs"]["process_slide"]["forward_calls"] == 0
    assert s1["failures"] == ["process_slide:not_encoded"] and not rep["pass"]
    assert "vs_first_run" not in s1["runs"]["process_slide"]
    # 'written' without a single forward call is not a fresh encode either
    def written_no_forward(ctx, row, st, enc):
        return real(ctx, row, st, {ENC: fake_factory(ENC, "cpu", False)})
    monkeypatch.setattr(X, "process_slide", written_no_forward)
    out = w.tmp / "pilot_out2"
    assert pilot_b(w, fr, out, slides="S1") == 1
    rep = _last_report(out / "pilot_bit_identity" / "process_slide_reports")
    assert rep["slides"]["S1"]["runs"]["process_slide"]["status"] == "written"
    assert rep["slides"]["S1"]["runs"]["process_slide"]["forward_calls"] == 0
    assert rep["slides"]["S1"]["failures"] == ["process_slide:not_encoded"]


def test_pilot_process_slide_failed_status_and_comparison_error(world, monkeypatch):
    w = world
    fr = standard_first_run(w, w.tmp / "first_run", slides=("S1",), num_workers=7)
    bad = tuple(coords_of("S1")[2]["wsi_loc"])
    out = w.tmp / "pilot_out"
    assert pilot_b(w, fr, out, slides="S1", opener=make_opener(fail_at=[bad])) == 1
    s1 = _last_report(out / "pilot_bit_identity" / "process_slide_reports")["slides"]["S1"]
    assert s1["runs"]["process_slide"]["status"] == "failed"
    assert s1["failures"] == ["process_slide:process_slide_failed"]
    assert s1["runs"]["process_slide"]["failures"]
    # [R2] an exception inside the comparison is a per-slide failure, not a traceback

    def boom(*a, **k):
        raise TypeError("cannot compare")
    monkeypatch.setattr(S, "compare_arms", boom)
    out = w.tmp / "pilot_out2"
    assert pilot_b(w, fr, out, slides="S1") == 1
    s1 = _last_report(out / "pilot_bit_identity" / "process_slide_reports")["slides"]["S1"]
    assert s1["failures"] == ["comparison_error"] and "cannot compare" in s1["comparison_error"]


def test_pilot_process_slide_weights_and_loader_mismatch(world):
    """[R1]/[R5c] the first-run record's encoder weights info and loader settings must equal
    this invocation's."""
    w = world
    fr = standard_first_run(w, w.tmp / "first_run", slides=("S1",), num_workers=7)
    recp = fr / "features" / ENC / "S1.record.json"
    rec = json.loads(recp.read_text())
    rec["encoder_meta"]["weights"]["sha256"] = "cd" * 32           # record-only change
    recp.write_text(json.dumps(rec))
    out = w.tmp / "pilot_out"
    assert pilot_b(w, fr, out, slides="S1") == 1
    s1 = _last_report(out / "pilot_bit_identity" / "process_slide_reports")["slides"]["S1"]
    assert s1["failures"] == ["weights_mismatch"] and not s1["weights_check"]["equal"]
    assert s1["weights_check"]["this_invocation"] == FAKE_WEIGHTS
    # a first run without weights info cannot be shown equal
    rec["encoder_meta"].pop("weights")
    recp.write_text(json.dumps(rec))
    assert pilot_b(w, fr, w.tmp / "pilot_out2", slides="S1") == 1
    s1 = _last_report(w.tmp / "pilot_out2" / "pilot_bit_identity" /
                      "process_slide_reports")["slides"]["S1"]
    assert s1["failures"] == ["weights_mismatch"]
    # loader settings: a first run with num_workers=0 (pilot uses 7)
    fr0 = standard_first_run(w, w.tmp / "first_run0", slides=("S1",), num_workers=0)
    out = w.tmp / "pilot_out3"
    assert pilot_b(w, fr0, out, slides="S1") == 1
    s1 = _last_report(out / "pilot_bit_identity" / "process_slide_reports")["slides"]["S1"]
    assert s1["failures"] == ["loader_mismatch"]
    assert s1["loader_check"]["first_run"]["num_workers"] == 0
    assert s1["loader_check"]["this_invocation"]["num_workers"] == 7


def test_r6c_weights_check_content_keys_only_unit():
    assert S.WEIGHTS_CONTENT_KEYS == ("repo_id", "file", "snapshot_revision", "bytes", "sha256")
    other_paths = dict(FAKE_WEIGHTS, cache_path="/home/other/hf/model.safetensors",
                       blob="/home/other/hf/blobs/ab", extra_path="/x")
    r = S.weights_content_check(dict(FAKE_WEIGHTS), other_paths)
    assert r["equal"] and r["missing_keys"] == [] and r["differing_keys"] == []
    assert r["recorded_not_compared_keys"] == ["blob", "cache_path", "extra_path"]
    for k in S.WEIGHTS_CONTENT_KEYS:
        changed = dict(FAKE_WEIGHTS, **{k: "different"})
        r = S.weights_content_check(dict(FAKE_WEIGHTS), changed)
        assert not r["equal"] and r["differing_keys"] == [k]
        missing = {kk: v for kk, v in FAKE_WEIGHTS.items() if kk != k}
        for a, b in ((missing, dict(FAKE_WEIGHTS)), (dict(FAKE_WEIGHTS), missing)):
            r = S.weights_content_check(a, b)
            assert not r["equal"] and r["missing_keys"] == [k]
    r = S.weights_content_check(dict(FAKE_WEIGHTS), dict(FAKE_WEIGHTS, bytes=123.0))
    assert not r["equal"] and r["differing_keys"] == ["bytes"]           # type-strict
    for a, b in ((None, FAKE_WEIGHTS), (FAKE_WEIGHTS, None), ([], FAKE_WEIGHTS)):
        assert not S.weights_content_check(a, b)["equal"]
    # X's error-form weights info (no content keys) cannot be shown equal
    err = {"repo_id": "fake/encoder", "error": "OSError: offline"}
    assert not S.weights_content_check(err, err)["equal"]


def test_r6c_pilot_weights_path_keys_recorded_not_compared(world):
    """A first run under another HOME/HF_HOME (different cache_path/blob) with identical
    content keys passes; a different sha256 or a missing content key is weights_mismatch."""
    w = world
    fr = standard_first_run(w, w.tmp / "first_run", slides=("S1",), num_workers=7)
    recp = fr / "features" / ENC / "S1.record.json"
    rec = json.loads(recp.read_text())
    base = dict(rec["encoder_meta"]["weights"])
    rec["encoder_meta"]["weights"].update(cache_path="/home/someone/hf/model.safetensors",
                                          blob="/home/someone/hf/blobs/ab")
    recp.write_text(json.dumps(rec))
    out = w.tmp / "pilot_out"
    assert pilot_b(w, fr, out, slides="S1") == 0
    s1 = _last_report(out / "pilot_bit_identity" / "process_slide_reports")["slides"]["S1"]
    wc = s1["weights_check"]
    assert s1["failures"] == [] and wc["equal"]
    assert wc["first_run"]["cache_path"] == "/home/someone/hf/model.safetensors"
    assert wc["this_invocation"]["cache_path"] == FAKE_WEIGHTS["cache_path"]
    assert wc["recorded_not_compared_keys"] == ["blob", "cache_path"]
    for i, (mut, key, field) in enumerate([
            (lambda d: d.update(sha256="cd" * 32), "sha256", "differing_keys"),
            (lambda d: d.pop("snapshot_revision"), "snapshot_revision", "missing_keys")]):
        rec["encoder_meta"]["weights"] = dict(base)
        mut(rec["encoder_meta"]["weights"])
        recp.write_text(json.dumps(rec))
        o = w.tmp / f"pilot_out_m{i}"
        assert pilot_b(w, fr, o, slides="S1") == 1
        s1 = _last_report(o / "pilot_bit_identity" / "process_slide_reports")["slides"]["S1"]
        assert s1["failures"] == ["weights_mismatch"]
        assert s1["weights_check"][field] == [key] and not s1["weights_check"]["equal"]
    # the features themselves still match (only the settings differ)
    assert s1["runs"]["process_slide"]["vs_first_run"]["unexpected_non_exact_arms"] == []


@pytest.mark.parametrize("tag", ["", "../x", "a/b", "_x", ".hidden", "x" * 65, "a b", "a\nb"])
def test_pilot_process_slide_invocation_tag_validated(world, tag):
    w = world
    with pytest.raises(S.SuiteError, match="invocation-tag"):
        pilot_b(w, w.tmp / "fr", w.tmp / "o", tag=tag)
    assert not (w.tmp / "o").exists()


def test_pilot_process_slide_invocation_tag_required(world):
    w = world
    with pytest.raises(SystemExit):
        pilot_b(w, w.tmp / "fr", w.tmp / "o", tag=None)


# --------------------------------------------------------------------------
# Item 7 / [A3]: extension population plan
# --------------------------------------------------------------------------


def plan_ext(w, fr, out, expected=1, extra=(), reason="synthetic fixture population"):
    """``expected=None`` uses the module default EXPECTED_DIRECT_COUNT (655) [R5b]; otherwise an
    override with its required reason."""
    over = ([] if expected is None else
            ["--expected-direct-count", str(expected), "--expected-direct-count-reason", reason])
    return S.plan_extension_main(["--out-root", str(out), "--queries", str(w.queries),
                                  "--queries-sha256", w.qsha, "--first-run-root", str(fr),
                                  "--v1-sha256", w.v1sha, "--encoder", ENC, *over, *extra])


def test_plan_extension_expected_direct_count_default_and_override(world):
    """[R5b] EXPECTED_DIRECT_COUNT=655 is the default; an override needs a reason and both
    are written into PLAN (and into a STOP report)."""
    w = world
    assert S.EXPECTED_DIRECT_COUNT == 655
    fr = standard_first_run(w, w.tmp / "first_run")
    out = w.tmp / "d1"
    assert plan_ext(w, fr, out, expected=None) == 2              # 1 direct != 655 -> STOP
    stop = json.loads(next((out / "extension").glob("STOP_*.json")).read_text())
    assert stop["expected_direct_count"] == 655 and stop["expected_direct_count_override"] is None
    assert "expected 655" in json.dumps(stop["detail"])
    assert not (out / "extension" / "PLAN.json").exists()
    out = w.tmp / "d2"
    assert plan_ext(w, fr, out, expected=1, reason="two-slide synthetic test population") == 0
    doc = json.loads((out / "extension" / "PLAN.json").read_text())
    assert doc["expected_direct_count"] == 1
    assert doc["expected_direct_count_module_constant"] == 655
    assert doc["expected_direct_count_override"] == {
        "value": 1, "reason": "two-slide synthetic test population", "module_constant": 655}
    with pytest.raises(S.SuiteError, match="reason"):
        plan_ext(w, fr, w.tmp / "d3", expected=1, reason="  ")
    with pytest.raises(S.SuiteError, match="without an override"):
        plan_ext(w, fr, w.tmp / "d4", expected=None,
                 extra=["--expected-direct-count-reason", "x"])
    assert not (w.tmp / "d3").exists() and not (w.tmp / "d4").exists()


def test_plan_extension_three_way_predicate(world):
    w = world
    fr = standard_first_run(w, w.tmp / "first_run")
    assert plan_ext(w, fr, w.out) == 0
    plan = json.loads((w.out / "extension" / "PLAN.json").read_text())
    assert (w.out / "extension" / "PLAN.json.sha256").read_text().split()[0] == \
        _sha((w.out / "extension" / "PLAN.json").read_bytes())
    by = {e["slide_id"]: e for e in plan["slides"]}
    assert by["S1"]["conditions"] == ["identity", "gaussian_blur_image"]
    assert by["S1"]["first_run_class"] == "pinned"
    assert by["S2"]["conditions"] == ["identity", "reinhard", "gaussian_blur_image"]
    assert by["S2"]["first_run_class"] == "direct"
    assert by["S3"]["first_run_record"] == "absent"
    assert by["S3"]["conditions"] == list(S.SUITE_CONDITIONS)
    assert plan["counts"] == {"slides": 3, "direct": 1, "pinned": 1, "absent": 1}
    assert plan["manifest_v1_sha256"] == w.v1sha and plan["queries_sha256"] == w.qsha
    assert by["S1"]["record_sha256"] == _sha((fr / "features" / ENC / "S1.record.json").read_bytes())


def _stop_of(out):
    files = list((out / "extension").glob("STOP_*.json"))
    assert len(files) == 1 and not (out / "extension" / "PLAN.json").exists()
    return json.loads(files[0].read_text())["detail"]


def test_plan_extension_stops(world):
    w = world
    fr = standard_first_run(w, w.tmp / "first_run")
    out = w.tmp / "o1"
    assert plan_ext(w, fr, out, expected=2) == 2                   # direct count != expected
    assert "expected 2" in json.dumps(_stop_of(out))
    # predicate disagreement: direct means but an LSF job id
    rewrite_first_run(fr, "S2", lsf_job="77", threads=121, means=DIRECT_MEANS)
    out = w.tmp / "o2"
    assert plan_ext(w, fr, out) == 2
    stop = _stop_of(out)["stops"][0]
    assert stop["slide_id"] == "S2" and "disagree" in stop["stop"]
    assert stop["predicate"] == {"means_direct": True, "lsf_job_null": False, "threads_121": True}
    # threads 121 encoded as a string is not the integer criterion
    rewrite_first_run(fr, "S2", lsf_job=None, threads="121", means=DIRECT_MEANS)
    out = w.tmp / "o3"
    assert plan_ext(w, fr, out) == 2 and "disagree" in json.dumps(_stop_of(out))
    # direct record with stds other than the recorded ones
    rewrite_first_run(fr, "S2", lsf_job=None, threads=121, means=DIRECT_MEANS,
                      stds=[18.0, 10.0, 11.0])
    out = w.tmp / "o4"
    assert plan_ext(w, fr, out) == 2 and "stds" in json.dumps(_stop_of(out))
    # a third Reinhard fit on an LSF record
    rewrite_first_run(fr, "S2", lsf_job=None, threads=121, means=DIRECT_MEANS)
    rewrite_first_run(fr, "S1", lsf_job="123", threads=8, means=[67.0, 23.0, -15.0])
    out = w.tmp / "o5"
    assert plan_ext(w, fr, out) == 2
    stop = _stop_of(out)["stops"]
    assert stop[0]["slide_id"] == "S1" and stop[0]["stop"] == "third Reinhard fit"


def test_plan_extension_invalid_record_is_absent(world):
    w = world
    fr = standard_first_run(w, w.tmp / "first_run")
    pkl = fr / "features" / ENC / "S1.pkl"
    b = bytearray(pkl.read_bytes())
    b[-2] ^= 0xFF                                                  # same size, different sha
    pkl.write_bytes(bytes(b))
    out = w.tmp / "o1"
    assert plan_ext(w, fr, out) == 0
    by = {e["slide_id"]: e for e in json.loads((out / "extension" / "PLAN.json").read_text())["slides"]}
    assert by["S1"]["first_run_record"] == "absent" and "sha256" in by["S1"]["reason"]
    assert by["S1"]["conditions"] == list(S.SUITE_CONDITIONS)
    out = w.tmp / "o2"
    assert plan_ext(w, fr, out, extra=["--size-only"]) == 0
    doc = json.loads((out / "extension" / "PLAN.json").read_text())
    assert doc["pickle_sha256_verified"] is False
    assert {e["slide_id"]: e for e in doc["slides"]}["S1"]["first_run_record"] == "present"
    # wrong manifest sha on the records -> invalid -> absent; a wrong code sha too
    rec = fr / "features" / ENC / "S1.record.json"
    r = json.loads(rec.read_text())
    r["code_sha256"] = "0" * 64
    rec.write_text(json.dumps(r))
    out = w.tmp / "o3"
    assert plan_ext(w, fr, out) == 0
    by = {e["slide_id"]: e for e in json.loads((out / "extension" / "PLAN.json").read_text())["slides"]}
    assert by["S1"]["first_run_record"] == "absent" and "code_sha256" in by["S1"]["reason"]
    out = w.tmp / "o4"
    assert S.plan_extension_main(["--out-root", str(out), "--queries", str(w.queries),
                                  "--queries-sha256", w.qsha, "--first-run-root", str(fr),
                                  "--encoder", ENC, "--expected-direct-count", "0",
                                  "--expected-direct-count-reason", "no valid record"]) == 0
    doc = json.loads((out / "extension" / "PLAN.json").read_text())
    assert doc["counts"]["absent"] == 3                            # default (production) v1 sha


# --------------------------------------------------------------------------
# Item 6: verifying extractor wrapper
# --------------------------------------------------------------------------


def extract(w, extra=(), out=None, **over):
    kw = inject(w, **over)
    kw.setdefault("encoder_factory", fake_factory)
    return S.extract_main(input_args(w, out) + env_args(w) + [
        "--encoder", ENC, "--device", "cpu", "--num-workers", "0", "--loader-batch", "2",
        "--forward-batch", "2", *extra], **kw)


def independent_features(w, sid):
    st = stim(w, sid)
    feats = {c: [] for c in S.SUITE_CONDITIONS}
    for i in range(SLIDES[sid]):
        pics = independent_pictures(w, sid, i)
        stack = X.encode_conditions(dict(zip(X.CONDITIONS, pics[:12])), fake_transform, st)
        for j, c in enumerate(X.CONDITIONS):
            feats[c].append(stack[j])
        feats["gaussian_blur_image"].append(fake_transform(pics[12]))
    return {c: (torch.stack(v).flatten(1) @ _PROJ).numpy() for c, v in feats.items()}


def failed_records(w, sid, out=None):
    d = (out or w.out) / "features" / "failed" / ENC
    return [json.loads(p.read_text()) for p in sorted(d.glob(f"{sid}.*.json"))]


def test_extract_pass_outputs_and_record(world):
    w = world
    assert fingerprint(w) == 0
    assert extract(w, ["--slides", "S1,S2"]) == 0
    for sid in ("S1", "S2"):
        pkl = w.out / "features" / ENC / f"{sid}.pkl"
        rec = json.loads((w.out / "features" / ENC / f"{sid}.record.json").read_text())
        feats, coords = pickle.loads(pkl.read_bytes())
        assert list(feats) == list(S.SUITE_CONDITIONS)
        assert coords == coords_of(sid)
        ind = independent_features(w, sid)
        for c, a in feats.items():
            assert a.dtype == np.float32 and a.shape == (SLIDES[sid], DIM)
            assert np.allclose(a, ind[c], atol=1e-5), c
        assert rec["schema"] == "r2_1_perturb_suite_record_v1"
        assert rec["extractor_code_sha256"] == X_SHA256
        assert rec["suite_code_sha256"] == S.SUITE_CODE_SHA256
        assert rec["conditions_encoded"] == list(S.SUITE_CONDITIONS)
        assert [s["condition"] for s in rec["stimuli"]] == list(S.SUITE_CONDITIONS)
        assert rec["stimuli"][-1]["parameters"] == CONTRACT_BLUR_PARAMETERS
        st = w.slide_path_of(sid).stat()
        assert rec["wsi"] == {"st_size": st.st_size, "st_mtime_ns": st.st_mtime_ns}
        assert rec["reinhard_nonfinite_tile_indices"] == []
        fv = rec["fingerprint_verification"]
        assert fv["n_tiles_verified"] == SLIDES[sid] and fv["n_pictures_verified"] == 13 * SLIDES[sid]
        assert fv["mismatches"] == 0
        assert fv["fp_npy_sha256"] == _sha((w.out / "fingerprints" / f"{sid}.fp.npy").read_bytes())
        assert rec["pickle_sha256"] == _sha(pkl.read_bytes())
        assert rec["pickle_bytes"] == pkl.stat().st_size
        assert rec["manifest_v2_sha256"] == w.v2sha and rec["first_run_record"] == "not_applicable"
        assert rec["environment"]["blur_golden"]["ok"]
        assert rec["forward_batch"] == 2 and rec["amp"] is True
    # skip-existing: only the [R4] reference-library probe read, no tile reads or forwards
    FakeSlide.calls = []
    FORWARD_CALLS.clear()
    assert extract(w, ["--slides", "S1,S2"]) == 0
    assert FakeSlide.calls == [PROBE_READ] and FORWARD_CALLS == []


def test_extract_single_tile_mismatch_publishes_nothing(world):
    w = world
    assert fingerprint(w) == 0
    FORWARD_CALLS.clear()
    alt = make_opener(alter=[tuple(coords_of("S1")[0]["wsi_loc"])])
    assert extract(w, ["--slides", "S1"], opener=alt) == 1
    assert FORWARD_CALLS == []                                    # no forward after mismatch
    assert not (w.out / "features" / ENC / "S1.pkl").exists()
    assert not (w.out / "features" / ENC / "S1.record.json").exists()
    (f,) = failed_records(w, "S1")
    assert f["stage"] == "fingerprint_verification" and f["published"] is False
    assert f["n_mismatched_tiles"] == 1 and f["n_verified_tiles"] == 4 and f["n_tiles"] == 5
    assert set(f["mismatch_counts"]) == set(S.SUITE_CONDITIONS)
    assert f["mismatch_counts"] == {c: (0 if c == "spatial_shift" else 1)
                                    for c in S.SUITE_CONDITIONS}
    assert f["mismatched_tiles"][0]["index"] == 0


def test_extract_late_and_multiple_mismatch_counts(world):
    w = world
    assert fingerprint(w) == 0
    FORWARD_CALLS.clear()
    last = make_opener(alter=[tuple(coords_of("S1")[4]["wsi_loc"])])
    assert extract(w, ["--slides", "S1"], opener=last) == 1
    assert FORWARD_CALLS != []                                    # earlier tiles were encoded ...
    assert not (w.out / "features" / ENC / "S1.pkl").exists()     # ... yet nothing published
    two = make_opener(alter=[tuple(coords_of("S2")[0]["wsi_loc"]),
                             tuple(coords_of("S2")[2]["wsi_loc"])])
    assert extract(w, ["--slides", "S2"], opener=two) == 1
    (f,) = failed_records(w, "S2")
    assert f["n_mismatched_tiles"] == 2 and f["mismatch_counts"]["identity"] == 2
    assert [m["index"] for m in f["mismatched_tiles"]] == [0, 2]
    assert not list((w.out / "features" / ENC).glob("S2.*"))


def test_extract_tile_error_wsi_change_and_missing_fingerprint(world):
    w = world
    assert fingerprint(w, slides="S1,S2") == 0
    bad = make_opener(fail_at=[tuple(coords_of("S1")[1]["wsi_loc"])])
    assert extract(w, ["--slides", "S1"], opener=bad) == 1
    assert failed_records(w, "S1")[0]["stage"] == "tile_render"

    def touching_opener(path):
        Path(path).write_bytes(b"changed during extraction!!")
        return FakeSlide(path)
    assert extract(w, ["--slides", "S2"], opener=touching_opener) == 1
    assert failed_records(w, "S2")[0]["stage"] == "wsi_changed"
    assert extract(w, ["--slides", "S3"]) == 1
    assert failed_records(w, "S3")[0]["stage"] == "fingerprint"
    assert not list((w.out / "features" / ENC).glob("*.pkl"))


def test_extract_conditions_subset_and_option_rules(world, monkeypatch):
    w = world
    assert fingerprint(w, slides="S3") == 0
    assert extract(w, ["--slides", "S3", "--conditions", "gaussian_blur_image,identity"]) == 0
    feats, _ = pickle.loads((w.out / "features" / ENC / "S3.pkl").read_bytes())
    assert list(feats) == ["identity", "gaussian_blur_image"]
    rec = json.loads((w.out / "features" / ENC / "S3.record.json").read_text())
    assert rec["fingerprint_verification"]["n_pictures_verified"] == 2 * 13
    # an existing output with a different condition set is never overwritten
    assert extract(w, ["--slides", "S3"]) == 1
    assert failed_records(w, "S3")[0]["stage"] == "existing_output"
    with pytest.raises(S.SuiteError):
        extract(w, ["--slides", "S3", "--conditions", "identity,nonsense"])
    with pytest.raises(S.SuiteError, match="exclusive"):
        extract(w, ["--slides", "S3", "--conditions", "identity", "--plan", str(w.v1)])
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "2")
    with pytest.raises(S.SuiteError, match="CPU request"):
        S.extract_main(input_args(w) + env_args(w) + ["--encoder", ENC, "--device", "cpu",
                                                      "--num-workers", "7"],
                       encoder_factory=fake_factory, **inject(w))
    with pytest.raises(SystemExit):
        S.extract_main(input_args(w) + env_args(w) + ["--encoder", "nope", "--device", "cpu"])


def test_extract_tampered_existing_output_is_failed(world):
    w = world
    assert fingerprint(w, slides="S3") == 0
    assert extract(w, ["--slides", "S3"]) == 0
    pkl = w.out / "features" / ENC / "S3.pkl"
    b = bytearray(pkl.read_bytes())
    b[-3] ^= 1
    pkl.write_bytes(bytes(b))
    snapshot = pkl.read_bytes()
    assert extract(w, ["--slides", "S3"]) == 1
    assert pkl.read_bytes() == snapshot
    f = failed_records(w, "S3")[0]
    assert f["stage"] == "existing_output" and "sha256" in f["reason"]


def test_r6d_extract_publish_conflict_record_carries_pickle_digest(world, monkeypatch):
    """A duplicate extract task refused at publish records the refused pickle's digest."""
    w = world
    assert fingerprint(w, slides="S3") == 0
    assert extract(w, ["--slides", "S3"]) == 0
    feat = w.out / "features" / ENC
    rec = json.loads((feat / "S3.record.json").read_text())
    before = ((feat / "S3.pkl").read_bytes(), (feat / "S3.record.json").read_bytes())
    with monkeypatch.context() as m:
        m.setattr(S, "existing_suite_output", lambda *a, **k: ("absent", ""))
        assert extract(w, ["--slides", "S3"]) == 1
    assert ((feat / "S3.pkl").read_bytes(), (feat / "S3.record.json").read_bytes()) == before
    [f] = failed_records(w, "S3")
    assert f["stage"] == "publish" and "refused to overwrite" in f["reason"]
    assert f["pickle_sha256"] == rec["pickle_sha256"] == _sha(before[0])


def test_extract_from_plan_reuses_first_run_identity(world):
    w = world
    fr = standard_first_run(w, w.tmp / "first_run")
    assert plan_ext(w, fr, w.out) == 0
    plan = w.out / "extension" / "PLAN.json"
    psha = _sha(plan.read_bytes())
    assert fingerprint(w) == 0
    with pytest.raises(S.SuiteError, match="plan sha256"):
        extract(w, ["--plan", str(plan), "--plan-sha256", "0" * 64])
    assert extract(w, ["--plan", str(plan), "--plan-sha256", psha, "--forward-batch", "256"]) == 0
    feats = {}
    for sid, conds in (("S1", ["identity", "gaussian_blur_image"]),
                       ("S2", ["identity", "reinhard", "gaussian_blur_image"]),
                       ("S3", list(S.SUITE_CONDITIONS))):
        f, _ = pickle.loads((w.out / "features" / ENC / f"{sid}.pkl").read_bytes())
        assert list(f) == conds
        feats[sid] = f
        rec = json.loads((w.out / "features" / ENC / f"{sid}.record.json").read_text())
        assert rec["plan_sha256"] == psha
        if sid == "S3":
            assert rec["first_run"]["status"] == "absent" and rec["first_run_record"] == "absent"
        else:
            assert rec["first_run"]["status"] == "present" and rec["first_run"]["identity_equal"]
    fr_s1, _ = pickle.loads((fr / "features" / ENC / "S1.pkl").read_bytes())
    assert np.array_equal(fr_s1["identity"], feats["S1"]["identity"])
    fr_s2, _ = pickle.loads((fr / "features" / ENC / "S2.pkl").read_bytes())
    assert not np.array_equal(fr_s2["reinhard"], feats["S2"]["reinhard"])   # re-encoded (pinned)


def test_extract_first_run_identity_mismatch_or_change_fails(world):
    w = world
    fr = standard_first_run(w, w.tmp / "first_run", slides=("S1",), s1_perturb=("identity",))
    out = w.tmp / "suite2"
    out.mkdir()
    assert plan_ext(w, fr, out, expected=0) == 0
    plan = out / "extension" / "PLAN.json"
    assert fingerprint(w, slides="S1", out=out) == 0
    assert extract(w, ["--plan", str(plan), "--plan-sha256", _sha(plan.read_bytes()),
                       "--forward-batch", "256", "--slides", "S1"], out=out) == 1
    f = failed_records(w, "S1", out)[0]
    assert f["stage"] == "first_run_identity" and "bit-exact" in f["reason"]
    assert not (out / "features" / ENC / "S1.pkl").exists()
    # the first-run record changes after planning -> FAILED, nothing published
    fr2 = standard_first_run(w, w.tmp / "first_run2", slides=("S1",))
    out = w.tmp / "suite3"
    out.mkdir()
    assert plan_ext(w, fr2, out, expected=0) == 0
    plan = out / "extension" / "PLAN.json"
    rewrite_first_run(fr2, "S1", lsf_job="999", threads=8, means=PINNED_MEANS)
    assert fingerprint(w, slides="S1", out=out) == 0
    assert extract(w, ["--plan", str(plan), "--plan-sha256", _sha(plan.read_bytes()),
                       "--forward-batch", "256", "--slides", "S1"], out=out) == 1
    f = failed_records(w, "S1", out)[0]
    assert f["stage"] == "first_run_identity" and "changed since the plan" in f["reason"]
    assert not (out / "features" / ENC / "S1.pkl").exists()


def test_canon_type_strict_and_numpy_aware():
    """[R2] _canon distinguishes 7 / 7.0 / True / np.int64(7) / np.int32(7) and never raises
    TypeError on numpy values."""
    forms = [7, 7.0, True, np.int64(7), np.int32(7), np.float64(7.0), [7], np.array([7])]
    canon = [S._canon(v) for v in forms]
    assert len(set(canon)) == len(forms)
    assert json.loads(S._canon(np.int64(7))) == ["np", "<i8", 7]
    assert json.loads(S._canon({"wsi_loc": [np.int64(1), 2]})) == {"wsi_loc": [["np", "<i8", 1], 2]}
    assert json.loads(S._canon(np.array([[1, 2]], np.int32))) == ["np", "<i4", [1, 2], [[1, 2]]]
    assert S._canon((1, 2)) == S._canon([1, 2])
    d = S._first_coord_difference([{"wsi_loc": [1, 2]}, {"wsi_loc": [3, 4]}],
                                  [{"wsi_loc": [1, 2]}, {"wsi_loc": [np.int64(3), 4]}])
    assert d == {"index": 1, "a": {"wsi_loc": [3, 4]}, "b": {"wsi_loc": [["np", "<i8", 3], 4]}}


def _numpy_coord_bank(w, sids, with_label=True):
    """Rewrite the coordinate banks of ``sids`` with numpy-typed values (as in a reader's
    P2b probe: np.int64 wsi_loc, optionally an np.int32 label) and refresh queries.csv."""
    rows = []
    for r in w.rows:
        sid, n = r["slide_id"], int(r["feature_rows"])
        if sid in sids:
            coords = []
            for k, c in enumerate(coords_of(sid)):
                d = {"wsi_loc": [np.int64(c["wsi_loc"][0]), np.int64(c["wsi_loc"][1])]}
                if with_label:
                    d["cluster_lbl"] = np.int32(k % 3)
                coords.append(d)
            data = pickle.dumps((np.zeros((n, 4), np.float32), coords), protocol=4)
            Path(r["resolved_path"]).write_bytes(data)
            r = dict(r, feature_size=len(data), feature_sha256=_sha(data))
        rows.append(r)
    with open(w.queries, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    w.rows, w.qsha = rows, _sha(w.queries.read_bytes())


@pytest.mark.parametrize("with_label", [True, False])
def test_extract_numpy_coords_on_both_sides_no_traceback(world, with_label):
    """[R2] P2/P2b: numpy-typed bank coordinates no longer crash the extract task;
    identical numpy coords on both sides compare equal and the slide is written."""
    w = world
    _numpy_coord_bank(w, {"S1", "S2", "S3"}, with_label)
    fr = standard_first_run(w, w.tmp / "first_run", slides=("S1",))
    assert plan_ext(w, fr, w.out, expected=0) == 0
    plan = w.out / "extension" / "PLAN.json"
    assert fingerprint(w, slides="S1") == 0
    assert extract(w, ["--plan", str(plan), "--plan-sha256", _sha(plan.read_bytes()),
                       "--forward-batch", "256", "--slides", "S1"]) == 0
    rec = json.loads((w.out / "features" / ENC / "S1.record.json").read_text())
    assert rec["first_run"]["identity_equal"]
    assert failed_records(w, "S1") == []
    assert list((w.out / "features" / "runs").glob("*.json"))


@pytest.mark.parametrize("with_label", [True, False])
def test_extract_numpy_vs_int_coords_fail_first_run_identity_and_continue(world, with_label):
    """[R2] first run with plain-int coordinates, bank now numpy-typed: a genuine type
    mismatch is reported in tagged form as a FAILED first_run_identity record; the task
    continues with the next slide and writes its run summary."""
    w = world
    fr = standard_first_run(w, w.tmp / "first_run")
    _numpy_coord_bank(w, {"S1"}, with_label)
    assert plan_ext(w, fr, w.out) == 0
    plan = w.out / "extension" / "PLAN.json"
    assert fingerprint(w, slides="S1,S2") == 0
    assert extract(w, ["--plan", str(plan), "--plan-sha256", _sha(plan.read_bytes()),
                       "--forward-batch", "256", "--slides", "S1,S2"]) == 1
    f = failed_records(w, "S1")
    assert len(f) == 1 and f[0]["stage"] == "first_run_identity"
    cmp_ = f[0]["comparison"]
    assert cmp_["coords_equal"] is False and cmp_["arms"]["identity"]["exact"] is True
    first = cmp_["coords_first_difference"]
    assert first["index"] == 0
    assert first["a"]["wsi_loc"] == [["np", "<i8", v] for v in coords_of("S1")[0]["wsi_loc"]]
    assert first["b"] == coords_of("S1")[0]
    if with_label:
        assert first["a"]["cluster_lbl"] == ["np", "<i4", 0]
    assert not (w.out / "features" / ENC / "S1.pkl").exists()
    assert (w.out / "features" / ENC / "S2.pkl").exists() and failed_records(w, "S2") == []
    summ = [json.loads(q.read_text()) for q in (w.out / "features" / "runs").glob("*.json")]
    assert len(summ) == 1


def test_extract_comparison_exception_is_failed_record(world, monkeypatch):
    """[R2] any exception in the first-run identity comparison -> FAILED (stage
    first_run_identity), never a bare traceback; the next slide still runs."""
    w = world
    fr = standard_first_run(w, w.tmp / "first_run")
    assert plan_ext(w, fr, w.out) == 0
    plan = w.out / "extension" / "PLAN.json"
    assert fingerprint(w, slides="S1,S2") == 0
    calls = []

    def boom(obj):
        if (isinstance(obj, list) and obj and isinstance(obj[0], dict) and "wsi_loc" in obj[0]
                and not calls):
            calls.append(1)
            raise TypeError("unserialisable coordinate")
        return real(obj)
    real = S._canon
    monkeypatch.setattr(S, "_canon", boom)
    assert extract(w, ["--plan", str(plan), "--plan-sha256", _sha(plan.read_bytes()),
                       "--forward-batch", "256", "--slides", "S1,S2"]) == 1
    f = failed_records(w, "S1")
    assert len(f) == 1 and f[0]["stage"] == "first_run_identity"
    assert "comparison error TypeError: unserialisable coordinate" in f[0]["reason"]
    assert calls == [1]
    assert (w.out / "features" / ENC / "S2.pkl").exists()
    assert list((w.out / "features" / "runs").glob("*.json"))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_main_dispatch_and_required_out_root(world, capsys):
    assert S.main([]) == 2
    assert S.main(["no-such-command"]) == 2
    for cmd in S.COMMANDS:
        with pytest.raises(SystemExit) as e:
            S.main([cmd])
        assert e.value.code == 2
        assert "--out-root" in capsys.readouterr().err
    assert S.main(["plan-tasks", "--out-root", str(world.out), "--queries", str(world.queries),
                   "--queries-sha256", ""]) == 1
    assert set(S.COMMANDS) == {
        "gen-manifest-v2", "capture-env", "pin-reference-libs", "env-check", "capture-host-tags",
        "plan-tasks", "fingerprint", "finalize", "reference-png", "select-pilot", "pilot-render",
        "pilot-process-slide", "plan-extension", "extract", "quarantine-orphans",
        "finalize-features"}                                # [C4-2] adapted (new subcommand)


def test_cli_module_entry_point(world, tmp_path):
    out = tmp_path / "cli_out"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-m", "prosper.evaluation.r2_1_perturb_suite_cli",
                        "gen-manifest-v2", "--out-root", str(out), "--v1", str(world.v1),
                        "--v1-sha256", world.v1sha], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert (out / "stimuli_v2.json").read_bytes() == world.v2.read_bytes()


def test_worker_tempdir_guard(world, monkeypatch):
    import tempfile
    S.check_worker_tempdir(0)
    monkeypatch.setattr(tempfile, "tempdir", "/" + "x" * 90)
    S.check_worker_tempdir(0)                                     # serial: no sockets needed
    with pytest.raises(S.SuiteError, match="TMPDIR"):
        S.check_worker_tempdir(1)
    with pytest.raises(S.SuiteError, match="TMPDIR"):
        pilot_b(world, world.tmp / "fr", world.tmp / "po")
    monkeypatch.setattr(tempfile, "tempdir", "/" + "x" * 74)
    S.check_worker_tempdir(7)


# --------------------------------------------------------------------------
# [C4-2] finalize-features (features/<E>/INDEX.json + COMPLETE over the extension plan)
# --------------------------------------------------------------------------

C4_2_EXTRA_REPORT_KEYS = ("report_schema", "rc", "written_at", "suite_code_sha256",
                          "manifest_v2_sha256", "queries_sha256", "accept_waivers")


def feat_plan(w, out=None):
    """Extension plan over S1 (direct: identity + gaussian_blur_image), S2 (pinned: identity,
    reinhard, gaussian_blur_image) and S3 (no first-run record: all 13)."""
    out = out or w.out
    fr = w.tmp / "first_run"
    if not fr.exists():
        standard_first_run(w, fr)
    assert plan_ext(w, fr, out) == 0
    plan = out / "extension" / "PLAN.json"
    return plan, _sha(plan.read_bytes())


def plan_extract(w, plan, psha, slides=None, extra=(), **over):
    a = ["--plan", str(plan), "--plan-sha256", psha, "--forward-batch", "256"]
    return extract(w, a + (["--slides", slides] if slides else []) + list(extra), **over)


def fp_complete(w, out=None):
    """Fingerprint every slide and finalize the fingerprint tree (repair 005, amendment 3:
    finalize-features requires fingerprints/COMPLETE)."""
    assert fingerprint(w, out=out) == 0
    assert S.finalize_main(input_args(w, out)) == 0


def feat_tree(w, slides=None):
    plan, psha = feat_plan(w)
    fp_complete(w)
    assert plan_extract(w, plan, psha, slides) == 0
    return plan, psha


def finalize_features_args(w, plan, psha, extra=(), out=None, encoder=ENC):
    return input_args(w, out) + ["--encoder", encoder, "--plan", str(plan),
                                 "--plan-sha256", psha, *extra]


def run_ff(w, capsys, plan, psha, extra=()):
    """(rc, compact summary, report file content) of one finalize-features run."""
    capsys.readouterr()
    rc = S.finalize_features_main(finalize_features_args(w, plan, psha, extra))
    s = _summary(capsys)
    rep = json.loads(Path(s["report"]["path"]).read_bytes())
    assert _digest(s["report"]["path"]) == s["report"]                 # printed sha == file
    return rc, s, rep


def fp_tree_bytes(w):
    """Bytes of fingerprints/INDEX.json and COMPLETE plus the finalize_reports listing."""
    fp = w.out / "fingerprints"
    return ((fp / "INDEX.json").read_bytes(), (fp / "COMPLETE").read_bytes(),
            sorted(x.name for x in (fp / "finalize_reports").iterdir()))


def feat_dir(w, out=None):
    return (out or w.out) / "features" / ENC


def feat_failed_dir(w, out=None):
    return (out or w.out) / "features" / "failed" / ENC


def rewrite_record(w, sid, **changes):
    p = feat_dir(w) / f"{sid}.record.json"
    original = p.read_bytes()
    rec = json.loads(original)
    for k, v in changes.items():
        if v is _DELETE:
            rec.pop(k, None)
        else:
            rec[k] = v
    p.write_bytes(S._dump_pretty(rec))
    return original


_DELETE = object()


def test_c4_2_finalize_features_complete(world, capsys):
    w = world
    plan, psha = feat_tree(w)
    d = feat_dir(w)
    fp_before = fp_tree_bytes(w)
    rc, s, rep = run_ff(w, capsys, plan, psha)
    # repair 006 (review 005 F6a): finalize-features leaves the fingerprint INDEX/COMPLETE
    # bytes (and its finalize_reports) unchanged
    assert fp_tree_bytes(w) == fp_before
    assert rc == 0
    assert set(s) == C4_SUMMARY_KEYS | {"index"}
    assert s["complete"] is True and s["rc"] == 0 and s["n_slides"] == 3
    assert s["n_blocking"] == 0 and s["blocking_failed_records"] == []
    assert s["index"] == _digest(d / "INDEX.json")
    r = Path(s["report"]["path"])
    assert r.parent == d / "finalize_reports" and C4_REPORT_NAME.fullmatch(r.name)
    assert rep["report_schema"] == S.FINALIZE_REPORT_SCHEMA and rep["rc"] == 0
    assert rep["index"] == "written" and rep["complete_status"] == "written"
    assert rep["index_published"] == s["index"]
    assert rep["complete_published"] == _digest(d / "COMPLETE")
    assert rep["encoder"] == ENC and rep["plan_sha256"] == psha
    assert rep["manifest_v2_sha256"] == w.v2sha and rep["queries_sha256"] == w.qsha
    assert "would_be_index_content" not in rep
    assert ((d / "INDEX.json").stat().st_mtime_ns <= (d / "COMPLETE").stat().st_mtime_ns
            <= r.stat().st_mtime_ns)
    idx = json.loads((d / "INDEX.json").read_bytes())
    assert idx["schema"] == "r2_1_perturb_suite_features_index_v1"
    assert idx["encoder"] == ENC and idx["plan_sha256"] == psha
    assert idx["manifest_v2_sha256"] == w.v2sha and idx["queries_sha256"] == w.qsha
    assert idx["n_slides"] == 3 and idx["n_tiles"] == sum(SLIDES.values())
    assert [e["slide_id"] for e in idx["slides"]] == ["S1", "S2", "S3"]
    plan_doc = json.loads(plan.read_bytes())
    planned = {e["slide_id"]: e["conditions"] for e in plan_doc["slides"]}
    for e in idx["slides"]:
        sid = e["slide_id"]
        raw = (d / f"{sid}.record.json").read_bytes()
        rec = json.loads(raw)
        assert set(e) == {"slide_id", "pickle_sha256", "pickle_bytes", "record_sha256",
                          "conditions_encoded", "n_tiles", "host", "gpu", "driver_tag",
                          "lsf_job", "invocation_tag", "plan_sha256", "reference_lib_waived"}
        assert e["plan_sha256"] == psha == rec["plan_sha256"]
        assert e["pickle_sha256"] == _sha((d / f"{sid}.pkl").read_bytes())
        assert e["pickle_bytes"] == (d / f"{sid}.pkl").stat().st_size
        assert e["record_sha256"] == _sha(raw)
        assert e["conditions_encoded"] == rec["conditions_encoded"] == planned[sid]
        assert e["n_tiles"] == SLIDES[sid]
        assert e["host"] == HOST == rec["environment"]["recorded"]["hostname"]
        assert e["gpu"] == rec["gpu"] and e["gpu"]["device"] == "cpu"
        assert e["driver_tag"] == "driver_595"
        assert e["lsf_job"] == rec["environment"]["recorded"]["lsf_job"]
        assert e["invocation_tag"] is None and "invocation_tag" not in rec   # frozen extract
        assert e["reference_lib_waived"] is False
    cov = idx["condition_coverage"]
    assert set(cov) == set(S.SUITE_CONDITIONS)
    assert cov["identity"] == {"planned": 3, "encoded": 3}
    assert cov["gaussian_blur_image"] == {"planned": 3, "encoded": 3}
    assert cov["reinhard"] == {"planned": 2, "encoded": 2}
    assert cov["spatial_shift"] == {"planned": 1, "encoded": 1}
    assert idx["hosts"] == [HOST] and idx["driver_tags"] == ["driver_595"]
    assert idx["n_pairs_other_plan"] == 0
    fp_comp = w.out / "fingerprints" / "COMPLETE"
    assert idx["fingerprints_complete_sha256"] == _sha(fp_comp.read_bytes())
    assert rep["fingerprints_complete_sha256"] == idx["fingerprints_complete_sha256"]
    fp_idx_sha = _sha((w.out / "fingerprints" / "INDEX.json").read_bytes())
    assert idx["fingerprints_index_sha256"] == fp_idx_sha == rep["fingerprints_index_sha256"]
    assert idx["n_reference_lib_waived"] == 0 and idx["reference_lib_waived"] == []
    assert idx["superseded_failures"] == [] and idx["extractor_code_sha256"] == X_SHA256
    assert "invocation_tag" in idx["field_sources"]
    comp = json.loads((d / "COMPLETE").read_bytes())
    assert comp == {"schema": "r2_1_perturb_suite_features_complete_v1", "encoder": ENC,
                    "plan_sha256": psha,
                    "fingerprints_complete_sha256": _sha(fp_comp.read_bytes()),
                    "fingerprints_index_sha256": fp_idx_sha, "index_file": "INDEX.json",
                    "index_sha256": _sha((d / "INDEX.json").read_bytes()), "n_slides": 3,
                    "superseded_failures": [],
                    "reference_lib_waivers": {"n_waived": 0, "slides": [],
                                              "accepted_reason": None}}
    # finalize-features does not finalize/modify the fingerprint tree (finalized by fp_complete)
    assert len(list((w.out / "fingerprints" / "finalize_reports").iterdir())) == 1
    # main() dispatch
    capsys.readouterr()
    assert S.main(["finalize-features"] + finalize_features_args(w, plan, psha)) == 0
    assert _summary(capsys)["index"] == s["index"]
    assert fp_tree_bytes(w) == fp_before


def test_c4_2_missing_slide_blocks(world, capsys):
    w = world
    plan, psha = feat_tree(w, slides="S1,S2")
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and s["complete"] is False and s["n_problems"] == 1
    assert set(s) == C4_SUMMARY_KEYS | {"would_be_index"} and s["would_be_index"] is None
    assert rep["problems"] == [{"slide_id": "S3", "status": "absent", "reason": ""}]
    assert rep["index"] is None and "index_published" not in rep
    d = feat_dir(w)
    assert not (d / "INDEX.json").exists() and not (d / "COMPLETE").exists()
    assert Path(s["report"]["path"]).parent == d / "finalize_reports"
    assert plan_extract(w, plan, psha, "S3") == 0
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0 and (d / "COMPLETE").exists()


def test_c4_2_bad_pickle_sha256_blocks(world, capsys):
    w = world
    plan, psha = feat_tree(w)
    d = feat_dir(w)
    original = rewrite_record(w, "S2", pickle_sha256="0" * 64)
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and rep["problems"] == [
        {"slide_id": "S2", "status": "invalid", "reason": "pickle sha256 != record"}]
    assert not (d / "INDEX.json").exists() and not (d / "COMPLETE").exists()
    (d / "S2.record.json").write_bytes(original)
    pkl = d / "S2.pkl"
    b = bytearray(pkl.read_bytes())
    b[-3] ^= 1                                               # same size, different content
    pkl.write_bytes(bytes(b))
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and rep["problems"][0]["reason"] == "pickle sha256 != record"
    pkl.write_bytes(bytes(b) + b"x")                         # size differs from pickle_bytes
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and rep["problems"][0]["reason"] == "record pickle_bytes mismatch"
    assert not (d / "INDEX.json").exists() and not (d / "COMPLETE").exists()


def test_c4_2_record_field_checks_block(world, capsys):
    """schema/slide_id/encoder/manifest/extractor sha/n_tiles mismatches each block."""
    w = world
    plan, psha = feat_tree(w)
    d = feat_dir(w)
    for key, value, why in (("schema", "other", "record schema mismatch"),
                            ("slide_id", "S1", "record slide_id mismatch"),
                            ("encoder", "uni2", "record encoder mismatch"),
                            ("manifest_v2_sha256", "0" * 64, "record manifest_v2_sha256 mismatch"),
                            ("extractor_code_sha256", "0" * 64,
                             "record extractor_code_sha256 mismatch"),
                            ("n_tiles", SLIDES["S2"] + 1, "record n_tiles mismatch"),
                            ("n_tiles", float(SLIDES["S2"]), "record n_tiles mismatch")):
        original = rewrite_record(w, "S2", **{key: value})
        rc, s, rep = run_ff(w, capsys, plan, psha)
        assert rc == 1 and rep["problems"] == [
            {"slide_id": "S2", "status": "invalid", "reason": why}], key
        (d / "S2.record.json").write_bytes(original)
    (d / "S2.record.json").write_text("[1, 2]")
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and "unreadable record" in rep["problems"][0]["reason"]
    assert not (d / "INDEX.json").exists() and not (d / "COMPLETE").exists()


def test_c4_2_fingerprint_verification_false_blocks(world, capsys):
    """[C4-2]/STOP-2: the frozen extract writes no ``ok`` key; a verification block that is
    absent/false, carries ``ok`` other than True, a non-matching status, mismatches != 0 or
    verified counts != n_tiles / 13 * n_tiles blocks."""
    w = world
    plan, psha = feat_tree(w)
    d = feat_dir(w)
    fv = json.loads((d / "S1.record.json").read_bytes())["fingerprint_verification"]
    assert "ok" not in fv and fv["status"] == S.FINGERPRINT_VERIFICATION_STATUS
    n = SLIDES["S1"]
    cases = [(False, "fingerprint_verification missing or not an object"),
             (_DELETE, "fingerprint_verification missing or not an object"),
             (dict(fv, ok=False), "fingerprint_verification.ok is not True"),
             (dict(fv, ok=1), "fingerprint_verification.ok is not True"),
             (dict(fv, status="partial"), "fingerprint_verification status mismatch"),
             (dict(fv, mismatches=1), "fingerprint_verification mismatches 1 != 0"),
             (dict(fv, mismatches=False), "fingerprint_verification mismatches False != 0"),
             (dict(fv, n_tiles_verified=n - 1),
              f"fingerprint_verification n_tiles_verified {n - 1} != {n}"),
             (dict(fv, n_pictures_verified=2 * n),
              f"fingerprint_verification n_pictures_verified {2 * n} != {13 * n}")]
    for value, why in cases:
        original = rewrite_record(w, "S1", fingerprint_verification=value)
        rc, s, rep = run_ff(w, capsys, plan, psha)
        assert rc == 1 and rep["problems"] == [
            {"slide_id": "S1", "status": "invalid", "reason": why}], value
        assert not (d / "INDEX.json").exists() and not (d / "COMPLETE").exists()
        (d / "S1.record.json").write_bytes(original)
    original = rewrite_record(w, "S1", fingerprint_verification=dict(fv, ok=True))
    rc, s, rep = run_ff(w, capsys, plan, psha)                 # an explicit ok=True passes
    assert rc == 0 and rep["problems"] == []


def test_c4_2_missing_condition_blocks_superset_accepted(world, capsys):
    w = world
    plan, psha = feat_tree(w)
    d = feat_dir(w)
    rec = json.loads((d / "S2.record.json").read_bytes())
    assert rec["conditions_encoded"] == ["identity", "reinhard", "gaussian_blur_image"]
    original = rewrite_record(w, "S2", conditions_encoded=["identity", "gaussian_blur_image"])
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and rep["problems"] == [{
        "slide_id": "S2", "status": "invalid",
        "reason": "record conditions_encoded lacks planned conditions ['reinhard']"}]
    rewrite_record(w, "S2", conditions_encoded="identity,reinhard,gaussian_blur_image")
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and "not a list" in rep["problems"][0]["reason"]
    (d / "S2.record.json").write_bytes(original)
    assert not (d / "INDEX.json").exists() and not (d / "COMPLETE").exists()


def test_c4_2_superset_pair_accepted_and_its_failed_record_superseded(world, capsys):
    """Re-pointed in repair 005 (finding 1): a pair produced WITHOUT --plan (all 13
    conditions, ``first_run`` not_applicable, ``plan_sha256`` null) is accepted only for a
    plan-"absent" slide (S3; a no-plan pair on a plan-"present" slide now blocks, see
    test_c5_no_plan_pair_on_present_slide_blocks); an earlier FAILED record for S3 is
    superseded by that pair; INDEX records that pair's null plan_sha256 and counts it in
    n_pairs_other_plan. (The superset case for a "present" slide is
    test_c5_other_plan_pair_with_passing_first_run_check_accepted.) The frozen extract SKIPS
    (does not refuse) an existing S3 pair whose conditions equal the plan's (all 13), so no
    existing_output record arises here; an earlier failed S3 attempt (tile_render FAILED
    record) is superseded instead."""
    w = world
    plan, psha = feat_plan(w)
    fp_complete(w)
    S.write_failed(w.out, feat_failed_dir(w), "S3", {"stage": "tile_render", "reason": "earlier",
                                                     "encoder": ENC})
    assert extract(w, ["--slides", "S3"]) == 0                       # no plan: all 13
    assert plan_extract(w, plan, psha) == 0                           # S3 skipped (existing)
    [f] = failed_records(w, "S3")
    assert f["stage"] == "tile_render"
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0 and s["n_superseded_failures"] == 1
    idx = json.loads((feat_dir(w) / "INDEX.json").read_bytes())
    s3 = idx["slides"][2]
    assert s3["slide_id"] == "S3" and s3["conditions_encoded"] == list(S.SUITE_CONDITIONS)
    assert s3["plan_sha256"] is None and idx["n_pairs_other_plan"] == 1
    assert [e["plan_sha256"] for e in idx["slides"][:2]] == [psha, psha]
    assert idx["condition_coverage"]["spatial_shift"] == {"planned": 1, "encoded": 1}
    [sup] = idx["superseded_failures"]
    assert sup["stage"] == "tile_render" and sup["slide_id"] == "S3"
    assert sup["record"].startswith(f"failed/{ENC}/S3.")
    assert sup["superseding_pair"] == {"pickle_sha256": s3["pickle_sha256"],
                                       "record_sha256": s3["record_sha256"]}


def test_c4_2_identical_duplicate_publish_superseded(world, capsys, monkeypatch):
    """[R6d] a duplicate extract refused at publish whose refused pickle digest equals the
    pair's is superseded (listed in INDEX and COMPLETE)."""
    w = world
    plan, psha = feat_tree(w)
    d = feat_dir(w)
    with monkeypatch.context() as m:
        m.setattr(S, "existing_suite_output", lambda *a, **k: ("absent", ""))
        assert plan_extract(w, plan, psha, "S3") == 1
    [f] = failed_records(w, "S3")
    s3_sha = _sha((d / "S3.pkl").read_bytes())
    assert f["stage"] == "publish" and f["pickle_sha256"] == s3_sha
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0 and s["n_superseded_failures"] == 1 and s["n_blocking"] == 0
    idx = json.loads((d / "INDEX.json").read_bytes())
    comp = json.loads((d / "COMPLETE").read_bytes())
    [frec] = list(feat_failed_dir(w).glob("S3.*.json"))
    for sup in (idx["superseded_failures"], comp["superseded_failures"]):
        assert sup == [{"record": f"failed/{ENC}/{frec.name}",
                        "record_sha256": _sha(frec.read_bytes()), "slide_id": "S3",
                        "stage": "publish", "failure_time": f["time"],
                        "superseding_pair": {
                            "pickle_sha256": s3_sha,
                            "record_sha256": _sha((d / "S3.record.json").read_bytes())}}]


def test_c4_2_differing_or_missing_publish_digest_blocks(world, capsys):
    """[R6d] digest key ``pickle_sha256``: a differing refused digest blocks with "duplicate
    produced different content"; a digest-less (or empty/non-string) publish record blocks
    with "publish conflict without digest"; no INDEX/COMPLETE either way."""
    w = world
    plan, psha = feat_tree(w)
    d, fdir = feat_dir(w), feat_failed_dir(w)
    s2_sha = _sha((d / "S2.pkl").read_bytes())
    S.write_failed(w.out, fdir, "S2", {"stage": "publish", "reason": "refused to overwrite",
                                       "encoder": ENC, "pickle_sha256": "f" * 64})
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and s["n_blocking"] == 1
    [b] = s["blocking_failed_records"]
    assert b["reason"] == "duplicate produced different content" and b["slide_id"] == "S2"
    assert b["refused_digest"] == {"pickle_sha256": "f" * 64}
    assert b["pair_digest"] == {"pickle_sha256": s2_sha}
    assert b["record"].startswith(f"failed/{ENC}/S2.") and b["stage"] == "publish"
    assert s["would_be_index"]["n_slides"] == 3
    assert rep["would_be_index"]["n_superseded_failures"] == 0
    assert not (d / "INDEX.json").exists() and not (d / "COMPLETE").exists()
    for f in fdir.glob("*.json"):
        f.unlink()
    for extra in ({}, {"pickle_sha256": ""}, {"pickle_sha256": None}, {"pickle_sha256": 7}):
        S.write_failed(w.out, fdir, "S1", {"stage": "publish", "reason": "refused", **extra})
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and s["n_blocking"] == 4
    assert {b["reason"] for b in s["blocking_failed_records"]} == {
        "publish conflict without digest"}
    assert not (d / "INDEX.json").exists() and not (d / "COMPLETE").exists()


def test_c4_2_failed_record_scope_is_features_failed_encoder_non_recursive(world, capsys):
    """FAILED records are read from ``features/failed/<E>/*.json`` only: not from other
    encoders, sub-directories (quarantine), ``features/<E>/failed/`` or the fingerprint tree.
    A foreign-slide or unreadable record there blocks."""
    w = world
    plan, psha = feat_tree(w)
    d, fdir = feat_dir(w), feat_failed_dir(w)
    junk = {"slide_id": "S9", "stage": "x"}
    for p in (w.out / "features" / "failed" / "uni2" / "S9.a.json", fdir / "sub" / "S9.b.json",
              d / "failed" / "S9.c.json", w.out / "features" / "failed" / "S9.d.json"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(junk))
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0 and rep["failed_records"] == []
    out2 = w.tmp / "scope2"
    shutil.copytree(w.out, out2, symlinks=True)
    for f in ("INDEX.json", "COMPLETE"):
        (feat_dir(w, out2) / f).unlink()
    S.write_failed(out2, feat_failed_dir(w, out2), "S9", {"stage": "extract"})
    (feat_failed_dir(w, out2) / "zz.json").write_text("{")
    capsys.readouterr()
    assert S.finalize_features_main(finalize_features_args(w, plan, psha, out=out2)) == 1
    s = _summary(capsys)
    reasons = sorted(b["reason"].split(":")[0] for b in s["blocking_failed_records"])
    assert reasons == ["slide_id not in the population", "unreadable"]
    assert not (feat_dir(w, out2) / "INDEX.json").exists()


def test_c4_2_orphan_quarantine_then_rerun_then_complete(world, capsys):
    """Orphan pair -> finalize-features blocks -> ``quarantine-orphans --tree features
    --encoder E`` moves the lone file and writes its FAILED record to features/failed/E,
    which finalize-features reads (blocking while S3 has no pair) -> extract rerun -> the
    quarantine record is superseded and COMPLETE is published."""
    w = world
    plan, psha = feat_tree(w)
    d = feat_dir(w)
    (d / "S3.record.json").unlink()                            # crash between the publishes
    lone = (d / "S3.pkl").read_bytes()
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and rep["problems"] == [{"slide_id": "S3", "status": "invalid",
                                            "reason": "orphan output (pickle=True, record=False)"}]
    capsys.readouterr()
    assert S.quarantine_orphans_main(["--out-root", str(w.out), "--tree", "features",
                                      "--encoder", ENC]) == 0
    assert not (d / "S3.pkl").exists()
    [moved] = list((w.out / "features" / "failed" / "quarantine").glob(f"*/{ENC}/S3.pkl"))
    assert moved.read_bytes() == lone
    [q] = list(feat_failed_dir(w).glob("S3.*.json"))
    assert json.loads(q.read_bytes())["stage"] == "orphan_quarantine"
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and rep["problems"][0]["status"] == "absent"
    [b] = s["blocking_failed_records"]
    assert b == {"record": f"failed/{ENC}/{q.name}", "slide_id": "S3",
                 "stage": "orphan_quarantine", "reason": "no validated pair for this slide"}
    assert plan_extract(w, plan, psha, "S3") == 0              # rerun now succeeds
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0 and s["n_blocking"] == 0 and s["n_superseded_failures"] == 1
    comp = json.loads((d / "COMPLETE").read_bytes())
    [sup] = comp["superseded_failures"]
    assert sup["record"] == f"failed/{ENC}/{q.name}" and sup["stage"] == "orphan_quarantine"
    assert moved.read_bytes() == lone                          # quarantined file kept
    # INDEX.json/COMPLETE/finalize_reports in features/E are never quarantined
    capsys.readouterr()
    assert S.quarantine_orphans_main(["--out-root", str(w.out), "--tree", "features",
                                      "--encoder", ENC]) == 0
    assert json.loads(capsys.readouterr().out)["n_orphans"] == 0
    assert (d / "INDEX.json").exists() and (d / "COMPLETE").exists()


def test_c4_2_index_only_with_complete_waivers(world, capsys):
    """[R5a]/[R6b] a waived pair blocks COMPLETE without --accept-waivers: no INDEX either; the
    would-be index is reported (summary digest, content only in the report file); with the
    reason INDEX (== the would-be bytes) and COMPLETE are published together."""
    w = world
    plan, psha = feat_tree(w, slides="S1,S2")
    assert plan_extract(w, plan, psha, "S3", ["--allow-missing-reference-libs"]) == 0
    d = feat_dir(w)
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and s["n_reference_lib_waived"] == 1 and "accept-waivers" in s["waiver_block"]
    assert not (d / "INDEX.json").exists() and not (d / "COMPLETE").exists()
    assert "would_be_index_content" not in s
    wbi = rep["would_be_index"]
    assert s["would_be_index"] == {k: wbi[k] for k in ("sha256", "bytes", "n_slides", "n_tiles")}
    content = rep["would_be_index_content"]
    assert content["reference_lib_waived"] == ["S3"] and content["n_reference_lib_waived"] == 1
    assert [e["reference_lib_waived"] for e in content["slides"]] == [False, False, True]
    with pytest.raises(S.SuiteError, match="reason"):
        S.finalize_features_main(finalize_features_args(w, plan, psha, ["--accept-waivers", " "]))
    reason = "lead accepted S3 waiver"
    rc, s, rep = run_ff(w, capsys, plan, psha, ["--accept-waivers", reason])
    assert rc == 0
    assert _sha((d / "INDEX.json").read_bytes()) == wbi["sha256"]
    comp = json.loads((d / "COMPLETE").read_bytes())
    assert comp["reference_lib_waivers"] == {"n_waived": 1, "slides": ["S3"],
                                             "accepted_reason": reason}
    assert rep["accept_waivers"] == reason


def test_c4_2_idempotent_rerun_and_late_failed_raises(world, capsys):
    w = world
    plan, psha = feat_tree(w)
    d = feat_dir(w)
    rc, s1, _ = run_ff(w, capsys, plan, psha)
    assert rc == 0
    idx, comp = (d / "INDEX.json").read_bytes(), (d / "COMPLETE").read_bytes()
    rc, s2, rep2 = run_ff(w, capsys, plan, psha)
    assert rc == 0 and rep2["index"] == "existing" and rep2["complete_status"] == "existing"
    assert s2["index"] == s1["index"] and s2["report"]["path"] != s1["report"]["path"]
    assert len(list((d / "finalize_reports").iterdir())) == 2
    assert (d / "INDEX.json").read_bytes() == idx and (d / "COMPLETE").read_bytes() == comp
    # a later (supersedable) FAILED record of a paired slide changes the index bytes: raises
    S.write_failed(w.out, feat_failed_dir(w), "S1", {"stage": "tile_render", "reason": "late"})
    capsys.readouterr()
    with pytest.raises(S.SuiteError, match="refusing to overwrite"):
        S.finalize_features_main(finalize_features_args(w, plan, psha))
    assert S.main(["finalize-features"] + finalize_features_args(w, plan, psha)) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert (d / "INDEX.json").read_bytes() == idx and (d / "COMPLETE").read_bytes() == comp
    assert len(list((d / "finalize_reports").iterdir())) == 2


def test_c4_2_plan_checks(world, capsys):
    """The plan is checked as ``extract --plan`` checks it (raw sha, schema, encoder, queries
    sha, manifest v1 sha) and must list unique query slides with valid conditions."""
    w = world
    plan, psha = feat_plan(w)
    with pytest.raises(S.SuiteError, match="plan sha256"):
        S.finalize_features_main(finalize_features_args(w, plan, "0" * 64))
    with pytest.raises(SystemExit):                            # --plan-sha256 is required
        S.finalize_features_main(input_args(w) + ["--encoder", ENC, "--plan", str(plan)])
    with pytest.raises(SystemExit):                            # encoder choices as extract
        S.finalize_features_main(finalize_features_args(w, plan, psha, encoder="nope"))
    with pytest.raises(S.SuiteError, match="mismatch"):        # plan made for ENC only
        S.finalize_features_main(finalize_features_args(w, plan, psha, encoder="uni2"))
    doc = json.loads(plan.read_bytes())

    def variant(name, mutate):
        dd = json.loads(json.dumps(doc))
        mutate(dd)
        p = w.tmp / f"plan_{name}.json"
        p.write_bytes(S._dump_pretty(dd))
        return p, _sha(p.read_bytes())
    for name, mutate, match in (
            ("schema", lambda dd: dd.update(schema="x"), "mismatch"),
            ("queries", lambda dd: dd.update(queries_sha256="0" * 64), "mismatch"),
            ("v1", lambda dd: dd.update(manifest_v1_sha256="0" * 64), "mismatch"),
            ("foreign", lambda dd: dd["slides"].append(dict(dd["slides"][0], slide_id="S9")),
             "not in the queries"),
            ("dup", lambda dd: dd["slides"].append(dict(dd["slides"][0])), "unique slide_id"),
            ("conds", lambda dd: dd["slides"][0].update(conditions=["nonsense"]),
             "bad conditions"),
            ("noslides", lambda dd: dd.update(slides=[]), "no slides")):
        p, sha = variant(name, mutate)
        with pytest.raises(S.SuiteError, match=match):
            S.finalize_features_main(finalize_features_args(w, p, sha))
    assert not (feat_dir(w) / "finalize_reports").exists()      # nothing written on refusal


def test_c4_2_compact_print_complete_with_60_superseded_and_60_blocking(world, capsys):
    w = world
    plan, psha = feat_tree(w)
    fdir = feat_failed_dir(w)
    pad = "x" * 300
    for i in range(60):
        S.write_failed(w.out, fdir, "S1", {"stage": "tile_render", "reason": f"{i}{pad}"})
    for i in range(30):
        S.write_failed(w.out, fdir, "S9", {"stage": "extract", "reason": f"{i}{pad}"})
    for i in range(30):
        S.write_failed(w.out, fdir, "S2", {"stage": "publish", "reason": f"refused {i}{pad}",
                                           "pickle_sha256": f"{i:064x}"})
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1
    assert len(json.dumps({k: v for k, v in rep.items() if k not in C4_2_EXTRA_REPORT_KEYS},
                          sort_keys=True, default=str)) > 20000
    assert s["n_blocking"] == 60 and s["blocking_failed_records"] == rep["blocking_failed_records"]
    assert s["n_superseded_failures"] == 60 == len(rep["superseded_failures"])
    assert sorted(b["reason"] for b in s["blocking_failed_records"]) == \
        ["duplicate produced different content"] * 30 + ["slide_id not in the population"] * 30
    assert {b["record"] for b in s["blocking_failed_records"]} == {
        f"failed/{ENC}/{f.name}" for f in fdir.glob("S[29].*.json")}
    assert _sha(S._dump_pretty(rep["would_be_index_content"])) == s["would_be_index"]["sha256"]
    assert not (feat_dir(w) / "INDEX.json").exists()


def test_c4_2_fingerprint_finalize_unaffected_by_features_tree(world, capsys):
    """The two finalizers read disjoint FAILED directories and publish disjoint files."""
    w = world
    plan, psha = feat_tree(w)
    S.write_failed(w.out, feat_failed_dir(w), "S9", {"stage": "extract"})
    capsys.readouterr()
    assert S.finalize_main(finalize_args(w)) == 0                      # fingerprints complete
    s = _summary(capsys)
    assert s["n_blocking"] == 0
    assert Path(s["report"]["path"]).parent == w.out / "fingerprints" / "finalize_reports"
    rc, s2, _ = run_ff(w, capsys, plan, psha)                          # features blocked
    assert rc == 1 and s2["n_blocking"] == 1
    assert not (feat_dir(w) / "INDEX.json").exists()
    assert (w.out / "fingerprints" / "COMPLETE").exists()


@pytest.mark.skipif(not os.environ.get("PS1_C003_SUITE_CLI"),
                    reason="set PS1_C003_SUITE_CLI to the candidate-003 module snapshot")
def test_c4_2_refactor_report_identical_running_candidate_003(world, monkeypatch, capsys):
    """Evidence run for the [C4-2] parameterisation of the fingerprint finalize: on a blocked
    tree (foreign, differing-digest and digest-less publish records, superseded records and a
    waiver) the candidate-003 printed report equals the candidate-004 report file minus its
    added keys; after accepting waivers INDEX/COMPLETE are byte-identical."""
    import importlib.util
    src = Path(os.environ["PS1_C003_SUITE_CLI"])
    assert _sha(src.read_bytes()) == C003_SUITE_SHA256
    name = "prosper.evaluation._ps1_candidate003_suite_cli_b"
    spec = importlib.util.spec_from_file_location(name, src)
    c003 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, c003)
    spec.loader.exec_module(c003)
    w = world
    _c4_shared_tree(w, monkeypatch, waived=True)
    fp = w.out / "fingerprints"
    _, s1 = load_fp(w, "S1")
    S.write_failed(w.out, fp / "failed", "S9", {"stage": "render"})
    S.write_failed(w.out, fp / "failed", "S2", {"stage": "publish", "npy_sha256": "a" * 64,
                                                 "content_sha256": "b" * 64})
    S.write_failed(w.out, fp / "failed", "S2", {"stage": "publish"})
    S.write_failed(w.out, fp / "failed", "S1", {"stage": "publish",
                                                 "npy_sha256": s1["array"]["npy_sha256"],
                                                 "content_sha256": "c" * 64})
    out_a, out_b = w.tmp / "c003b", w.tmp / "c004b"
    shutil.copytree(w.out, out_a, symlinks=True)
    shutil.copytree(w.out, out_b, symlinks=True)
    capsys.readouterr()
    assert c003.finalize_main(input_args(w, out_a)) == 1
    rep_a = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert S.finalize_main(input_args(w, out_b)) == 1
    s = _summary(capsys)
    rep_b = json.loads(Path(s["report"]["path"]).read_bytes())
    assert {k: v for k, v in rep_b.items() if k not in C4_2_EXTRA_REPORT_KEYS} == rep_a
    assert len(rep_a["blocking_failed_records"]) == 4 and rep_a["waiver_block"]
    for out in (out_a, out_b):                       # remove the blocking records, accept
        for f in (out / "fingerprints" / "failed").glob("*.json"):
            if json.loads(f.read_text()).get("slide_id") in ("S9", "S2") or \
                    json.loads(f.read_text()).get("content_sha256") == "c" * 64:
                f.unlink()
    extra = ["--accept-waivers", "lead accepted S3"]
    assert c003.finalize_main(input_args(w, out_a) + extra) == 0
    assert S.finalize_main(input_args(w, out_b) + extra) == 0
    capsys.readouterr()
    for f in ("INDEX.json", "COMPLETE"):
        a = (out_a / "fingerprints" / f).read_bytes()
        b = (out_b / "fingerprints" / f).read_bytes()
        assert a == b, f
        with capsys.disabled():
            print(f"[c4-2 refactor byte identity] {f} sha256={_sha(a)} bytes={len(a)}")


# --------------------------------------------------------------------------
# Repair 005 ([decision record] "Design decisions 14:40"): first-run identity provenance (finding 1),
# conditions_encoded names (amendment 2), fingerprint-tree binding (amendment 3), malformed
# FAILED records (finding 4), null host, finalize byte identity with candidate 004.
# --------------------------------------------------------------------------

C004_SUITE_SHA256 = "22a599cad36118def8fc1052871ec24fbf058c0834cc94b91f992572f63169e1"
NO_CHECK = ("plan entry first_run_record is 'present' but the record has no first-run "
            "identity check (first_run.status {!r})")


def plan_variant(w, plan, name, mutate):
    """A differently-hashed copy of ``plan`` with ``mutate(doc)`` applied."""
    doc = json.loads(plan.read_bytes())
    mutate(doc)
    p = w.tmp / f"plan_{name}" / "PLAN.json"
    p.parent.mkdir(parents=True)
    p.write_bytes(S._dump_pretty(doc))
    return p, _sha(p.read_bytes())


def plan_slide(plan, sid):
    [e] = [s for s in json.loads(plan.read_bytes())["slides"] if s["slide_id"] == sid]
    return e


def test_c5_no_plan_pair_on_present_slide_blocks(world, capsys):
    """Finding 1: a pair produced by ``extract`` WITHOUT --plan on a plan-"present" slide
    (S1) carries no first-run identity check and blocks; no INDEX/COMPLETE."""
    w = world
    plan, psha = feat_plan(w)
    assert plan_slide(plan, "S1")["first_run_record"] == "present"
    fp_complete(w)
    assert extract(w, ["--slides", "S1"]) == 0                       # no plan: all 13
    assert plan_extract(w, plan, psha, "S2,S3") == 0
    rec = json.loads((feat_dir(w) / "S1.record.json").read_bytes())
    assert rec["first_run"] == {"status": "not_applicable"} and rec["plan_sha256"] is None
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and s["complete"] is False and s["n_blocking"] == 0
    assert rep["problems"] == [{"slide_id": "S1", "status": "invalid",
                                "reason": NO_CHECK.format("not_applicable")}]
    assert s["would_be_index"] is None
    assert not (feat_dir(w) / "INDEX.json").exists() and not (feat_dir(w) / "COMPLETE").exists()


def test_c5_other_plan_pair_without_first_run_check_blocks(world, capsys):
    """Finding 1 / q7: a pair produced under ANOTHER plan in which S1 is "absent"
    (all 13, first_run.status "absent") blocks under the current plan, where S1 is
    "present"."""
    w = world
    plan, psha = feat_plan(w)

    def s1_absent(doc):
        for e in doc["slides"]:
            if e["slide_id"] == "S1":
                for k in ("first_run_class", "record_sha256", "pickle_sha256", "pickle_bytes"):
                    e.pop(k, None)
                e.update(first_run_record="absent", reason="other plan",
                         conditions=list(S.SUITE_CONDITIONS))
    p2, sha2 = plan_variant(w, plan, "other", s1_absent)
    assert sha2 != psha
    fp_complete(w)
    assert plan_extract(w, p2, sha2, "S1") == 0
    assert plan_extract(w, plan, psha, "S2,S3") == 0
    rec = json.loads((feat_dir(w) / "S1.record.json").read_bytes())
    assert rec["plan_sha256"] == sha2 and rec["first_run"]["status"] == "absent"
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and rep["problems"] == [{"slide_id": "S1", "status": "invalid",
                                            "reason": NO_CHECK.format("absent")}]
    assert not (feat_dir(w) / "INDEX.json").exists() and not (feat_dir(w) / "COMPLETE").exists()


def test_c5_other_plan_pair_with_passing_first_run_check_accepted(world, capsys):
    """A pair from another plan in which S1 is "present" against the SAME first-run record but
    with all 13 conditions (a superset) passes; INDEX records its plan_sha256 and
    n_pairs_other_plan == 1. A first-run record digest differing from the plan entry, a
    missing identity_equal, or a differing first-run class/pickle digest blocks."""
    w = world
    plan, psha = feat_plan(w)
    s1 = plan_slide(plan, "S1")

    def s1_all13(doc):
        for e in doc["slides"]:
            if e["slide_id"] == "S1":
                e["conditions"] = list(S.SUITE_CONDITIONS)
    p3, sha3 = plan_variant(w, plan, "superset", s1_all13)
    fp_complete(w)
    assert plan_extract(w, p3, sha3, "S1") == 0
    assert plan_extract(w, plan, psha, "S2,S3") == 0
    d = feat_dir(w)
    fr = json.loads((d / "S1.record.json").read_bytes())["first_run"]
    assert fr["status"] == "present" and fr["identity_equal"] is True
    assert fr["record_sha256"] == s1["record_sha256"]
    for change, why in (
            (dict(fr, record_sha256="0" * 64),
             "record first_run.record_sha256 != plan entry record_sha256"),
            ({k: v for k, v in fr.items() if k != "identity_equal"},
             "record first_run.identity_equal is not True"),
            (dict(fr, identity_equal=1), "record first_run.identity_equal is not True"),
            (dict(fr, **{"class": "other"}), "record first_run.class != plan entry first_run_class"),
            (dict(fr, pickle_sha256="0" * 64),
             "record first_run.pickle_sha256 != plan entry pickle_sha256"),
            ("present", NO_CHECK.format(None))):
        original = rewrite_record(w, "S1", first_run=change)
        rc, s, rep = run_ff(w, capsys, plan, psha)
        assert rc == 1 and rep["problems"] == [
            {"slide_id": "S1", "status": "invalid", "reason": why}], change
        (d / "S1.record.json").write_bytes(original)
    # repair 006 (review 005 F6b): ``extract --plan`` now refuses S1 (existing pair's
    # conditions are a superset, not equal) with an existing_output FAILED record, which the
    # passing other-plan pair supersedes
    assert plan_extract(w, plan, psha, "S1") == 1
    [f] = failed_records(w, "S1")
    assert f["stage"] == "existing_output"
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0 and s["n_superseded_failures"] == 1 and s["blocking_failed_records"] == []
    idx = json.loads((d / "INDEX.json").read_bytes())
    e1 = idx["slides"][0]
    [sup] = idx["superseded_failures"]
    assert sup["stage"] == "existing_output" and sup["slide_id"] == "S1"
    assert sup["record"].startswith(f"failed/{ENC}/S1.")
    assert sup["superseding_pair"] == {"pickle_sha256": e1["pickle_sha256"],
                                       "record_sha256": e1["record_sha256"]}
    assert json.loads((d / "COMPLETE").read_bytes())["superseded_failures"] == [sup]
    assert e1["slide_id"] == "S1" and e1["plan_sha256"] == sha3
    assert e1["conditions_encoded"] == list(S.SUITE_CONDITIONS)
    assert [e["plan_sha256"] for e in idx["slides"][1:]] == [psha, psha]
    assert idx["n_pairs_other_plan"] == 1 and idx["plan_sha256"] == psha
    assert idx["condition_coverage"]["spatial_shift"] == {"planned": 1, "encoded": 2}


def test_c5_detected_first_run_identity_failed_not_superseded_by_pair_lacking_check(
        world, capsys):
    """Finding 1 / q6: S1's identity arm differs from its first-run identity arm, so
    ``extract --plan`` writes a first_run_identity FAILED record; a later no-plan S1 pair
    lacks the check: the pair is invalid and the FAILED record blocks (not superseded)."""
    w = world
    standard_first_run(w, w.tmp / "first_run", s1_perturb=("identity",))
    plan, psha = feat_plan(w)
    fp_complete(w)
    assert plan_extract(w, plan, psha) == 1
    [f] = failed_records(w, "S1")
    assert f["stage"] == "first_run_identity"
    assert extract(w, ["--slides", "S1"]) == 0                       # rerun without --plan
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and rep["problems"] == [{"slide_id": "S1", "status": "invalid",
                                            "reason": NO_CHECK.format("not_applicable")}]
    [frec] = list(feat_failed_dir(w).glob("S1.*.json"))
    assert s["blocking_failed_records"] == [{
        "record": f"failed/{ENC}/{frec.name}", "slide_id": "S1", "stage": "first_run_identity",
        "reason": "no validated pair for this slide"}]
    assert s["n_superseded_failures"] == 0
    assert not (feat_dir(w) / "INDEX.json").exists() and not (feat_dir(w) / "COMPLETE").exists()


def test_c5_first_run_identity_failed_on_absent_slide_needs_passing_check(world, capsys):
    """A first_run_identity FAILED record can be superseded only by a pair that itself carries
    a passing first-run identity check: on the plan-"absent" slide S3 (valid pair, first_run
    "absent") it blocks with the explicit reason; other stages there are superseded."""
    w = world
    plan, psha = feat_tree(w)
    fdir = feat_failed_dir(w)
    S.write_failed(w.out, fdir, "S3", {"stage": "first_run_identity", "reason": "synthetic"})
    S.write_failed(w.out, fdir, "S3", {"stage": "tile_render", "reason": "synthetic"})
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and rep["problems"] == []
    [b] = s["blocking_failed_records"]
    assert b["slide_id"] == "S3" and b["stage"] == "first_run_identity"
    assert b["reason"] == S.FIRST_RUN_SUPERSEDE_REASON
    assert [x["stage"] for x in rep["superseded_failures"]] == ["tile_render"]
    assert not (feat_dir(w) / "INDEX.json").exists()


def test_c5_detected_first_run_identity_failed_superseded_by_passing_pair(world, capsys):
    """A first_run_identity FAILED record actually written by ``extract --plan`` (the plan's
    first-run record changed after planning) is superseded once the first-run record is
    restored and the S1 rerun passes the identity check."""
    w = world
    plan, psha = feat_plan(w)
    fp_complete(w)
    rp = Path(plan_slide(plan, "S1")["record_path"])
    original = rp.read_bytes()
    rp.write_bytes(original + b"\n")                                 # changed since the plan
    assert plan_extract(w, plan, psha, "S1") == 1
    [f] = failed_records(w, "S1")
    assert f["stage"] == "first_run_identity" and "changed since the plan" in f["reason"]
    rp.write_bytes(original)
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and s["blocking_failed_records"][0]["stage"] == "first_run_identity"
    assert plan_extract(w, plan, psha) == 0
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0 and s["n_blocking"] == 0
    comp = json.loads((feat_dir(w) / "COMPLETE").read_bytes())
    assert [(x["slide_id"], x["stage"]) for x in comp["superseded_failures"]] == [
        ("S1", "first_run_identity")]
    idx = json.loads((feat_dir(w) / "INDEX.json").read_bytes())
    assert idx["slides"][0]["plan_sha256"] == psha and idx["n_pairs_other_plan"] == 0


def test_c5_index_provenance_fields_and_null_host(world, capsys):
    """INDEX per-slide plan_sha256 (from the record) and n_pairs_other_plan (null counted); a
    missing host / driver tag is JSON null (not the string "None") per slide and in the
    aggregates (listed last)."""
    w = world
    plan, psha = feat_tree(w)
    d = feat_dir(w)
    rec1 = json.loads((d / "S1.record.json").read_bytes())
    env = json.loads(json.dumps(rec1["environment"]))
    del env["recorded"]["hostname"]
    env["recorded"]["gpu_driver_tags"] = None
    rewrite_record(w, "S1", environment=env)
    rewrite_record(w, "S3", plan_sha256=None)                         # absent slide: no check
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0
    raw = (d / "INDEX.json").read_bytes()
    idx = json.loads(raw)
    assert [e["plan_sha256"] for e in idx["slides"]] == [psha, psha, None]
    assert idx["n_pairs_other_plan"] == 1
    assert idx["slides"][0]["host"] is None and idx["slides"][0]["driver_tag"] is None
    assert idx["hosts"] == [HOST, None] and idx["driver_tags"] == ["driver_595", None]
    assert b'"None"' not in raw
    assert {"plan_sha256", "n_pairs_other_plan", "hosts"} <= set(idx["field_sources"])


def test_c5_conditions_encoded_duplicate_or_unknown_blocks_order_is_provenance(world, capsys):
    """Amendment 2: conditions_encoded must be a duplicate-free subset of the 13 suite names;
    a different order is recorded as provenance, not an error."""
    w = world
    plan, psha = feat_tree(w)
    d = feat_dir(w)
    for value, why in (
            (["identity", "reinhard", "reinhard", "gaussian_blur_image"],
             "record conditions_encoded has duplicates ['reinhard']"),
            (["identity", "reinhard", "gaussian_blur_image", "bogus"],
             "record conditions_encoded has unknown conditions ['bogus']")):
        original = rewrite_record(w, "S2", conditions_encoded=value)
        rc, s, rep = run_ff(w, capsys, plan, psha)
        assert rc == 1 and rep["problems"] == [
            {"slide_id": "S2", "status": "invalid", "reason": why}], value
        (d / "S2.record.json").write_bytes(original)
    rewrite_record(w, "S2", conditions_encoded=["gaussian_blur_image", "reinhard", "identity"])
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0
    idx = json.loads((d / "INDEX.json").read_bytes())
    assert idx["slides"][1]["conditions_encoded"] == ["gaussian_blur_image", "reinhard",
                                                      "identity"]


def test_c5_fingerprints_complete_required(world, capsys):
    """Amendment 3: without fingerprints/COMPLETE finalize-features fails closed before
    writing anything; after the fingerprint finalize it completes."""
    w = world
    plan, psha = feat_plan(w)
    assert fingerprint(w) == 0
    assert plan_extract(w, plan, psha) == 0
    with pytest.raises(S.SuiteError, match="fingerprints/COMPLETE"):
        S.finalize_features_main(finalize_features_args(w, plan, psha))
    capsys.readouterr()
    assert S.main(["finalize-features"] + finalize_features_args(w, plan, psha)) == 1
    assert "fingerprints/COMPLETE" in capsys.readouterr().err
    assert not (feat_dir(w) / "finalize_reports").exists()
    assert S.finalize_main(input_args(w)) == 0
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0


def test_c5_record_fingerprint_digests_must_equal_fingerprint_tree(world, capsys):
    """Amendment 3 / q8: the record's fp_json/fp_npy/fp_content digests must equal
    the fingerprint tree's <sid>.fp.json (its sha256 and array digests)."""
    w = world
    plan, psha = feat_tree(w)
    d = feat_dir(w)
    fv = json.loads((d / "S3.record.json").read_bytes())["fingerprint_verification"]
    fpj = w.out / "fingerprints" / "S3.fp.json"
    assert fv["fp_json_sha256"] == _sha(fpj.read_bytes())
    for key, why in (("fp_json_sha256", "record fp_json_sha256 != sha256 of fingerprint tree "
                                        "S3.fp.json"),
                     ("fp_npy_sha256", "record fp_npy_sha256 != fingerprint tree S3.fp.json "
                                       "array.npy_sha256"),
                     ("fp_content_sha256", "record fp_content_sha256 != fingerprint tree "
                                           "S3.fp.json array.content_sha256")):
        original = rewrite_record(w, "S3", fingerprint_verification=dict(fv, **{key: "0" * 64}))
        rc, s, rep = run_ff(w, capsys, plan, psha)
        assert rc == 1 and rep["problems"] == [
            {"slide_id": "S3", "status": "invalid", "reason": why}], key
        (d / "S3.record.json").write_bytes(original)
    fp_orig = fpj.read_bytes()
    fpj.write_bytes(fp_orig + b"\n")                                  # tree changed afterwards
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and rep["problems"][0]["reason"].startswith("record fp_json_sha256 != ")
    fpj.unlink()
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and "S3.fp.json unreadable" in rep["problems"][0]["reason"]
    fpj.write_bytes(fp_orig)
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0


def test_c5_plan_first_run_fields_checked(world, capsys):
    """The plan must state first_run_record present/absent; a "present" entry needs a
    non-empty record_sha256 (as plan-extension writes it)."""
    w = world
    plan, psha = feat_plan(w)
    fp_complete(w)

    def no_digest(doc):
        doc["slides"][0].pop("record_sha256")

    def bad_status(doc):
        doc["slides"][0]["first_run_record"] = "maybe"
    for name, mutate, match in (("nodigest", no_digest, "without record_sha256"),
                                ("status", bad_status, "is not 'present'/'absent'")):
        p, sha = plan_variant(w, plan, name, mutate)
        with pytest.raises(S.SuiteError, match=match):
            S.finalize_features_main(finalize_features_args(w, p, sha))
    assert not (feat_dir(w) / "finalize_reports").exists()


MALFORMED_FAILED = (("unparseable", "{not json", None),
                    ("list_json", "[1, 2]", "unreadable: FAILED record is not a JSON object"),
                    ("list_slide_id", json.dumps({"slide_id": ["S1"], "stage": "x"}),
                     "slide_id is not a string"),
                    ("dict_slide_id", json.dumps({"slide_id": {"a": 1}, "stage": "x"}),
                     "slide_id is not a string"),
                    ("int_slide_id", json.dumps({"slide_id": 7, "stage": "x"}),
                     "slide_id is not a string"),
                    ("no_slide_id", json.dumps({"stage": "x"}), "slide_id not in the population"))


def test_c5_malformed_failed_records_block_without_traceback_features(world, capsys):
    """Finding 4: malformed / ill-typed FAILED records in features/failed/E become blocking
    records with a reason (never a traceback); nothing is published."""
    w = world
    plan, psha = feat_tree(w)
    fdir = feat_failed_dir(w)
    fdir.mkdir(parents=True, exist_ok=True)
    for name, content, why in MALFORMED_FAILED:
        p = fdir / f"S1.probe_{name}.json"
        p.write_text(content)
        rc, s, rep = run_ff(w, capsys, plan, psha)
        assert rc == 1, name
        [b] = s["blocking_failed_records"]
        assert b["record"] == f"failed/{ENC}/{p.name}"
        if why is None:
            assert b["reason"].startswith("unreadable: ")
        else:
            assert b["reason"] == why, name
        assert not (feat_dir(w) / "INDEX.json").exists()
        p.unlink()
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0


def test_c5_malformed_failed_records_block_without_traceback_fingerprints(world, capsys):
    """Finding 4 in the fingerprint ``finalize`` (only the malformed path changes; the
    well-formed byte identity is covered by the 003/004 identity tests)."""
    w = world
    assert fingerprint(w) == 0
    fdir = w.out / "fingerprints" / "failed"
    fdir.mkdir(parents=True, exist_ok=True)
    for name, content, why in MALFORMED_FAILED:
        p = fdir / f"S1.probe_{name}.json"
        p.write_text(content)
        capsys.readouterr()
        assert S.main(["finalize"] + finalize_args(w)) == 1, name
        s = _summary(capsys)
        [b] = s["blocking_failed_records"]
        assert b["record"] == f"failed/{p.name}"
        if why is None:
            assert b["reason"].startswith("unreadable: ")
        else:
            assert b["reason"] == why, name
        assert not (w.out / "fingerprints" / "INDEX.json").exists()
        p.unlink()
    capsys.readouterr()
    assert S.finalize_main(finalize_args(w)) == 0


def _load_snapshot(monkeypatch, env, expected_sha, name):
    import importlib.util
    src = Path(os.environ[env])
    assert _sha(src.read_bytes()) == expected_sha
    spec = importlib.util.spec_from_file_location(name, src)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not os.environ.get("PS1_C004_SUITE_CLI"),
                    reason="set PS1_C004_SUITE_CLI to the candidate-004 module snapshot")
@pytest.mark.parametrize("waived", [False, True])
def test_c5_finalize_bytes_identical_running_candidate_004(world, monkeypatch, capsys, waived):
    """Repair 005 evidence run: the candidate-004 module itself (22a599ca...) and this module
    finalize copies of the same well-formed fixture trees. Blocked branch (foreign,
    differing-digest and digest-less publish records): report files equal after removing
    ``written_at``/``suite_code_sha256`` and normalising the out-root; complete branch: INDEX
    and COMPLETE byte-identical and report files equal under the same normalisation."""
    c004 = _load_snapshot(monkeypatch, "PS1_C004_SUITE_CLI", C004_SUITE_SHA256,
                          "prosper.evaluation._ps1_candidate004_suite_cli")
    _finalize_identity_vs(c004, "004", world, monkeypatch, capsys, waived)


def _finalize_identity_vs(c004, label, world, monkeypatch, capsys, waived):
    """Body of the finalize byte-identity regression against a frozen earlier candidate
    module (``c004``; factored out unchanged in repair 006 so it also runs against 005)."""
    w = world
    extra = _c4_shared_tree(w, monkeypatch, waived)
    fp = w.out / "fingerprints"
    _, s1 = load_fp(w, "S1")
    S.write_failed(w.out, fp / "failed", "S9", {"stage": "render"})
    S.write_failed(w.out, fp / "failed", "S2", {"stage": "publish", "npy_sha256": "a" * 64,
                                                 "content_sha256": "b" * 64})
    S.write_failed(w.out, fp / "failed", "S2", {"stage": "publish"})
    out_a, out_b = w.tmp / "c004r", w.tmp / "c005r"
    shutil.copytree(w.out, out_a, symlinks=True)
    shutil.copytree(w.out, out_b, symlinks=True)

    def run(mod, out, args):
        capsys.readouterr()
        rc = mod.finalize_main(input_args(w, out) + args)
        s = _summary(capsys)
        raw = Path(s["report"]["path"]).read_text()
        rep = json.loads(raw.replace(str(out_a), str(out_b)))
        for k in ("written_at", "suite_code_sha256"):
            rep.pop(k)
        s = json.loads(json.dumps(s).replace(str(out_a), str(out_b)))
        s.pop("report")
        return rc, s, rep
    ra, sa, rep_a = run(c004, out_a, [])
    rb, sb, rep_b = run(S, out_b, [])
    assert ra == rb == 1 and rep_a == rep_b and sa == sb
    assert len(rep_a["blocking_failed_records"]) == 3
    for out in (out_a, out_b):
        for f in (out / "fingerprints" / "failed").glob("*.json"):
            if json.loads(f.read_text()).get("slide_id") in ("S9", "S2"):
                f.unlink()
    ra, sa, rep_a = run(c004, out_a, extra)
    rb, sb, rep_b = run(S, out_b, extra)
    assert ra == rb == 0 and rep_a == rep_b and sa == sb
    for f in ("INDEX.json", "COMPLETE"):
        a = (out_a / "fingerprints" / f).read_bytes()
        b = (out_b / "fingerprints" / f).read_bytes()
        assert a == b, f
        with capsys.disabled():
            print(f"[c5 byte identity vs {label}] waived={waived} {f} sha256={_sha(a)} "
                  f"bytes={len(a)}")


C005_SUITE_SHA256 = "a9193ba3dde83510fb830939f2bd1c273a0af6226acfe26f0c57d7c523167da2"


@pytest.mark.skipif(not os.environ.get("PS1_C005_SUITE_CLI"),
                    reason="set PS1_C005_SUITE_CLI to the candidate-005 module snapshot")
@pytest.mark.parametrize("waived", [False, True])
def test_c6_finalize_bytes_identical_running_candidate_005(world, monkeypatch, capsys, waived):
    """Repair 006 evidence run: as the candidate-004 regression, against the candidate-005
    module (a9193ba3...): fingerprint ``finalize`` reports/summaries equal (normalised) and
    INDEX/COMPLETE byte-identical on blocked and complete well-formed trees."""
    c005 = _load_snapshot(monkeypatch, "PS1_C005_SUITE_CLI", C005_SUITE_SHA256,
                          "prosper.evaluation._ps1_candidate005_suite_cli")
    _finalize_identity_vs(c005, "005", world, monkeypatch, capsys, waived)


# --------------------------------------------------------------------------
# Repair 006 (review 005 F1): fingerprints/COMPLETE parsed and bound to fingerprints/INDEX.json;
# record fingerprint digests bound to the verified INDEX entry; fail closed (SuiteError,
# nothing written).
# --------------------------------------------------------------------------

def rebind_fp_index(w, mutate):
    """Rewrite fingerprints/INDEX.json as ``finalize`` would serialise it after ``mutate(doc)``
    and re-point fingerprints/COMPLETE (index_sha256 and n_slides) at the new bytes, so only
    the mutated content is inconsistent. Returns the original (INDEX, COMPLETE) bytes."""
    fp = w.out / "fingerprints"
    orig = ((fp / "INDEX.json").read_bytes(), (fp / "COMPLETE").read_bytes())
    doc = json.loads(orig[0])
    mutate(doc)
    raw = S._dump_pretty(doc)
    (fp / "INDEX.json").write_bytes(raw)
    comp = json.loads(orig[1])
    comp["index_sha256"] = _sha(raw)
    if isinstance(doc.get("slides"), list):
        comp["n_slides"] = len(doc["slides"])
    (fp / "COMPLETE").write_bytes(S._dump_pretty(comp))
    return orig


def restore_fp(w, orig):
    fp = w.out / "fingerprints"
    (fp / "INDEX.json").write_bytes(orig[0])
    (fp / "COMPLETE").write_bytes(orig[1])


def assert_ff_fails_closed(w, capsys, plan, psha, match):
    with pytest.raises(S.SuiteError, match=match):
        S.finalize_features_main(finalize_features_args(w, plan, psha))
    capsys.readouterr()
    assert S.main(["finalize-features"] + finalize_features_args(w, plan, psha)) == 1
    out, err = capsys.readouterr()
    assert "nothing written" in err and out == ""
    d = feat_dir(w)
    assert not (d / "finalize_reports").exists()
    assert not (d / "INDEX.json").exists() and not (d / "COMPLETE").exists()


def test_c6_fingerprints_complete_parsed_and_bound_to_index(world, capsys):
    """F1 (i)/(ii): garbage COMPLETE (with and without a fingerprint INDEX, review probe R3),
    a non-COMPLETE object, a COMPLETE whose index_sha256 does not match the INDEX bytes, an
    INDEX replaced by {} (probe R6c), a missing INDEX, an INDEX that does not re-serialise to
    its own bytes, a wrong INDEX schema / manifest, inconsistent n_slides and a duplicate slide
    entry each raise SuiteError before anything is written; the restored tree completes."""
    w = world
    plan, psha = feat_tree(w)
    fp = w.out / "fingerprints"
    comp_p, idx_p = fp / "COMPLETE", fp / "INDEX.json"
    comp0, idx0 = comp_p.read_bytes(), idx_p.read_bytes()

    comp_p.write_bytes(b"garbage, not a COMPLETE document")
    assert_ff_fails_closed(w, capsys, plan, psha, "COMPLETE does not parse as JSON")
    idx_p.unlink()
    assert_ff_fails_closed(w, capsys, plan, psha, "COMPLETE does not parse as JSON")
    idx_p.write_bytes(idx0)
    for doc, match in (([1, 2], "is not a r2_1_perturb_suite_fingerprint_complete_v2 object"),
                       (dict(json.loads(comp0), schema="other"),
                        "is not a r2_1_perturb_suite_fingerprint_complete_v2 object"),
                       (dict(json.loads(comp0), index_file="X.json"),
                        "index_file is not 'INDEX.json'"),
                       ({k: v for k, v in json.loads(comp0).items() if k != "index_sha256"},
                        "COMPLETE has no index_sha256"),
                       (dict(json.loads(comp0), index_sha256="0" * 64),
                        "COMPLETE index_sha256 0{64} != sha256 of fingerprints/INDEX.json")):
        comp_p.write_bytes(S._dump_pretty(doc))
        assert_ff_fails_closed(w, capsys, plan, psha, match)
    comp_p.write_bytes(comp0)
    idx_p.write_bytes(b"{}\n")                                       # probe R6c
    assert_ff_fails_closed(w, capsys, plan, psha, "COMPLETE index_sha256 [0-9a-f]{64} != sha256")
    idx_p.unlink()
    assert_ff_fails_closed(w, capsys, plan, psha, "cannot read fingerprints/INDEX.json")
    # consistent hash but not finalize's serialisation (compact bytes)
    compact = S._dump_compact(json.loads(idx0))
    idx_p.write_bytes(compact)
    comp_p.write_bytes(S._dump_pretty(dict(json.loads(comp0), index_sha256=_sha(compact))))
    assert_ff_fails_closed(w, capsys, plan, psha, "does not re-serialise to its own bytes")
    idx_p.write_bytes(b"[1, 2" + b"]")
    comp_p.write_bytes(S._dump_pretty(dict(json.loads(comp0),
                                           index_sha256=_sha(b"[1, 2]"))))
    assert_ff_fails_closed(w, capsys, plan, psha, "does not re-serialise")
    restore_fp(w, (idx0, comp0))
    def add_slide(extra):
        def mutate(doc):
            doc["slides"].append(extra(doc))
            doc["n_slides"] = len(doc["slides"])
        return mutate
    for mutate, match in (
            (lambda doc: doc.update(schema="other"),
             "is not a r2_1_perturb_suite_fingerprint_index_v2 object"),
            (lambda doc: doc.update(manifest_v2_sha256="0" * 64),
             "manifest_v2_sha256 != --manifest-v2"),
            (lambda doc: doc.update(n_slides=99), "n_slides inconsistent"),
            (add_slide(lambda doc: dict(doc["slides"][0])), "malformed or duplicate slide entry"),
            (add_slide(lambda doc: "S9"), "malformed or duplicate slide entry")):
        orig = rebind_fp_index(w, mutate)
        assert_ff_fails_closed(w, capsys, plan, psha, match)
        restore_fp(w, orig)
    assert (comp_p.read_bytes(), idx_p.read_bytes()) == (comp0, idx0)
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0
    idx = json.loads((feat_dir(w) / "INDEX.json").read_bytes())
    assert idx["fingerprints_complete_sha256"] == _sha(comp0)
    assert idx["fingerprints_index_sha256"] == _sha(idx0)


def test_c6_record_digests_must_equal_fingerprint_index_entry(world, capsys):
    """F1 (iii): with the .fp.json files untouched (so the amendment-3 .fp.json check passes),
    a fingerprint INDEX entry whose fp_json_sha256 / npy_sha256 / content_sha256 differs from
    the record's fingerprint_verification digests, or a slide absent from the INDEX, raises
    SuiteError (nothing written), even though COMPLETE certifies that INDEX."""
    w = world
    plan, psha = feat_tree(w)
    fpj = (w.out / "fingerprints" / "S3.fp.json").read_bytes()

    def set_s3(key):
        def mutate(doc):
            for e in doc["slides"]:
                if e["slide_id"] == "S3":
                    e[key] = "0" * 64
        return mutate
    for key, rkey in (("fp_json_sha256", "fp_json_sha256"), ("npy_sha256", "fp_npy_sha256"),
                      ("content_sha256", "fp_content_sha256")):
        orig = rebind_fp_index(w, set_s3(key))
        assert_ff_fails_closed(
            w, capsys, plan, psha,
            f"does not match the feature records: S3: record {rkey} != "
            f"fingerprints/INDEX.json entry {key}; nothing written")
        restore_fp(w, orig)

    def drop_s3(doc):
        doc["slides"] = [e for e in doc["slides"] if e["slide_id"] != "S3"]
        doc["n_slides"] = len(doc["slides"])
    orig = rebind_fp_index(w, drop_s3)
    assert_ff_fails_closed(w, capsys, plan, psha,
                           "S3: slide has no entry in the verified fingerprints/INDEX.json")
    restore_fp(w, orig)
    assert (w.out / "fingerprints" / "S3.fp.json").read_bytes() == fpj
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 0


def test_c6_fingerprint_index_check_applies_only_to_validated_pairs(world, capsys):
    """A slide whose pair is already invalid (here missing) stays a per-slide problem (rc 1,
    report written); the INDEX-entry check applies to validated pairs, so a blocked report is
    still produced when the fingerprint INDEX/COMPLETE themselves verify."""
    w = world
    plan, psha = feat_tree(w)
    (feat_dir(w) / "S2.pkl").unlink()
    rc, s, rep = run_ff(w, capsys, plan, psha)
    assert rc == 1 and [x["slide_id"] for x in rep["problems"]] == ["S2"]
    assert rep["fingerprints_index_sha256"] == _sha(
        (w.out / "fingerprints" / "INDEX.json").read_bytes())
