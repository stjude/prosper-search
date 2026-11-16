"""R2.1 P2: 13-arm H-Optimus-1 perturbation evaluation adapter ([decision record] P2 specification 001+002+003).

Inputs
  * first-run 12-arm tree ``<first-root>/features/h_optimus_1/<slide>.pkl|.record.json``
    (schema ``r2_1_perturb_record_v1``, frozen extractor X);
  * suite tree ``<suite-root>/features/h_optimus_1/<slide>.pkl|.record.json``
    (schema ``r2_1_perturb_suite_record_v1``) and ``<suite-root>/extension/PLAN.json``;
  * the population ``queries.csv`` (``--population-sha256`` required) whose ``resolved_path``
    rows are the accepted main-bank pickles.

Per slide (single pass, P2-A6): verify both sources (P2.3 + P2-A1/A2/A5/A13/A15/A17), the
identity == main-bank byte equality, route each of the 13 arms to its source (P2.3(4)), and
compute the SAMPLER of every arm from the same bytes (S.sampler, float64 percentiles, no fp16
cast; design decision A). Failures are collected for every slide; any failure writes only
``VERIFICATION_FAILED.json`` and exits 2 (P2-A6/P2-A12).

Evaluation re-implements the orchestration of the frozen evaluator E (``evaluate`` /
``run_encoder``) over the fixed 13-arm tuple and calls only E's condition-agnostic functions
(P2-A14 whitelist). The output layout is E's run layout with ``run_record.json`` schema
``r2_1_perturb_eval13_run_v1``; ``code_frozen/`` holds byte copies of the adapter, E, S and the
bootstrap kernel (P2-A10/P2-A12).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import r2_1_hoptimus_main as kernel
from . import r2_1_perturb_eval_cli as E
from . import r2_1_perturb_sampler as S

ENCODER = "h_optimus_1"
DIM = 1536
FIRST_RUN_CONDITIONS: Tuple[str, ...] = (
    "identity", "rotate_90", "rotate_180", "rotate_270", "flip_horizontal", "flip_vertical",
    "brightness", "saturation", "contrast", "spatial_shift", "reinhard", "gaussian_blur")
BLUR_IMAGE = "gaussian_blur_image"
ARMS13: Tuple[str, ...] = FIRST_RUN_CONDITIONS + (BLUR_IMAGE,)
IDENTITY = "identity"
ARM_LABELS = {
    "gaussian_blur": "post-preprocessing (encoder-specific), historical Fig. 4A definition",
    BLUR_IMAGE: "source-pixel blur k7 \u03c38/7 before preprocessing (stimuli_v2)",
}
SUITE_CONDITIONS = {"pinned": ["identity", BLUR_IMAGE],
                    "direct": ["identity", "reinhard", BLUR_IMAGE],
                    "absent": list(ARMS13)}
FIRST_RUN_SCHEMA = "r2_1_perturb_record_v1"
SUITE_SCHEMA = "r2_1_perturb_suite_record_v1"
RUN_SCHEMA = "r2_1_perturb_eval13_run_v1"
FAILED_SCHEMA = "r2_1_perturb_eval13_verification_failed_v1"
FINGERPRINT_STATUS = "all tiles x 13 pictures verified before transform"

DIRECT_MEANS = (67.08672332763672, 23.11690330505371, -15.080077171325684)
PINNED_MEANS = (67.08672332763672, 23.116905212402344, -15.08007526397705)
PINNED_STDS = (18.064807891845703, 10.140364646911621, 11.612739562988281)
DIRECT_MAIN_THREADS = 121

E_SHA256 = "07c0d27f36fc65f7c0e8c9cb8c7df7a611bda1a84931e612885e105028667377"
S_SHA256 = "e1880db79f8fcd40dee3e3db91f9664f298ef60250cb1bcd2feb7ea494c8e60c"
KERNEL_SHA256 = "dee7d68a8b9fcee61d2e845412afdaf20455388ae5c2e61844d4b21d73811faf"

V11_DISCLOSURE = (
    "the first-run 12-arm tree is lead-reconciled; its **independent content evidence** is (i) "
    "`SUITE/review/pilot_output_review_001/REVIEW.md` \u2014 ACCEPT WITH AMENDMENTS: 13 "
    "stratified slides (all 13 first-run groups, 3 CPU models, one slide per first-run host) "
    "\u00d7 12 arms re-encoded with frozen X on two A100 hosts, whole pickles byte-identical for "
    "12/13 slides and 11/12 arms for 10057 (only `reinhard`, = G3, expected); and (ii) features "
    "output review 001 \u2014 identity arm bit-exact 2,778/2,778 and direct/pinned class "
    "agreement 2,778/2,778. P2.3 adds per-slide integrity, record binding, routing and identity "
    "== main-bank checks; it does **not** re-encode or otherwise verify the content of the ten "
    "non-identity arms on the remaining 2,765 slides. The reuse chain remains \"established, not "
    "proved\" (item 8); the P2 result package must disclose this (V11).")
AUTHOR_DECISIONS = {
    "A": "aggregation = published perturbation-harness recipe: S.sampler float64 percentiles on "
         "the float32 features, no fp16 cast (differs from the fp16-cast main-grid SAMPLER; "
         "disclosed in Methods)",
    "B": "the zero-shift slides are kept and reported as null stimuli in the spatial_shift row",
    "C": "the Reinhard-only exclusion of the listed non-finite tiles is retained (indices from the "
         "record of the source actually read for reinhard) and the cells are reported",
    "D": "the historical top-k majority-vote macro-F1/accuracy is computed as a labelled "
         "secondary metric (HIST_LABEL) with the adverse-baseline disclosure",
}
DECISION_A_DISCLOSURE = (
    "Perturbation SAMPLER: np.percentile on the float32 features in float64 with no float16 "
    "cast (published perturbation-harness recipe, author decision A); the main-grid SAMPLER "
    "casts to float16 first. This difference is disclosed in Methods.")
HIST_BASELINE_DISCLOSURE = (
    "The published Fig. 4A \"Original\" baseline used self-only exclusion (2,743 slides, fp16 "
    "references) whereas P2 is same-protocol, so published sign patterns are not expected to "
    "reproduce.")
NO_CLAIM = "No equivalence or \"robust\" claim is produced by P2."

RUN_RECORD_KEYS: Tuple[str, ...] = (
    "schema", "created", "host", "lsf_job", "argv", "encoder", "feature_dim_used", "conditions",
    "arm_labels", "population", "inputs", "manifest_v2_sha256", "input_manifest", "code_sha256",
    "code_frozen", "versions", "policy", "routing", "per_arm_source", "null_stimuli",
    "reinhard_excluded_cells", "absent_branch_count", "identity_equals_main_bank",
    "first_run_provenance", "suite_gpu_provenance", "coverage_summary", "disclosures",
    "draw_tables", "diagnostics", "timings_seconds", "workers", "threads",
    "peak_rss_mb_main_process", "outputs")
MANIFEST_HEADER: Tuple[str, ...] = (
    "population_index", "slide_id", "class", "first_run_pickle", "first_run_pickle_sha256",
    "first_run_record", "first_run_record_sha256", "suite_pickle", "suite_pickle_sha256",
    "suite_record", "suite_record_sha256", "main_bank", "main_bank_sha256", "n_tiles",
    "reinhard_nonfinite_listed", "reinhard_nonfinite_source", "identity_equals_main_bank",
    "null_stimulus_spatial_shift", "first_run_host", "first_run_gpu", "first_run_lsf_job",
    "suite_gpu_model", "suite_gpu_driver", *[f"source_{c}" for c in ARMS13])


@dataclasses.dataclass(frozen=True)
class Pins:
    """Pinned production bindings (P2.2/P2.3/P2-A1/A2/A5/A7). Not exposed as CLI flags (no
    waiver flag, P2-A6); synthetic tests pass their own instance to :func:`run`."""
    queries_sha256: str = "5ff0ea3bd2999a4dbebb4c73e679d15870450d38f5bb6fc91be071ee72cbe471"
    plan_sha256: str = "e9aa968ea94990d525b6320f9fb46810940a37a4fc872f9e7235b8fabb9b7bc0"
    manifest_v1_sha256: str = "014513af5920ed621cfad2dc5cf527747936bb21c1db8336d282a187d669d6ce"
    manifest_v2_sha256: str = "6e78a883d408066a2bc909dcf323692b3eb18f8dc7105aaa95f0735b5348c262"
    extractor_code_sha256: str = (
        "14eb457ebfe2243a924bb7928483538985120b6987de51eb4a6cc6ccf3a8d0bb")
    suite_code_sha256: str = "cc023861719142aeb8993fb8ae4072a607bac2914386043f81c28ad443949879"
    expected_counts: Tuple[Tuple[str, int], ...] = (("direct", 655), ("pinned", 2123),
                                                    ("absent", 0))
    expected_direct_count: int = 655
    zero_shift_slides: Tuple[str, ...] = ("12116", "15703", "6176", "7896")
    e_sha256: str = E_SHA256
    s_sha256: str = S_SHA256
    kernel_sha256: str = KERNEL_SHA256


class VerificationFailed(E.EvalError):
    """Raised internally after VERIFICATION_FAILED.json has been written."""


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonable(v: Any, limit: int = 2000) -> Any:
    try:
        s = json.dumps(v, default=str)
    except Exception:                                          # noqa: BLE001
        s = json.dumps(repr(v))
    if len(s) > limit:
        return {"truncated_repr": s[:limit]}
    return json.loads(s)


def _fail(fails: List[dict], check: str, expected: Any, observed: Any, **extra) -> None:
    fails.append({"check": check, "expected": _jsonable(expected),
                  "observed": _jsonable(observed), **{k: _jsonable(v) for k, v in extra.items()}})


def _expect(fails: List[dict], check: str, expected: Any, observed: Any) -> bool:
    ok = type(observed) is type(expected) and observed == expected
    if not ok:
        _fail(fails, check, expected, observed)
    return ok


def _f32_bits_equal(values: Any, ref: Sequence[float]) -> bool:
    """Float32-byte equality (P2-A5); like the frozen suite CLI's ``_bits_equal`` a value that is
    not exactly a float32 number never matches."""
    try:
        arr = np.asarray([float(v) for v in values], dtype=np.float64)
        if not np.array_equal(arr.astype(np.float32).astype(np.float64), arr):
            return False
        return (np.asarray(arr, np.float32).tobytes()
                == np.asarray(ref, np.float32).tobytes())
    except Exception:                                          # noqa: BLE001
        return False


def direct_predicate(rec: dict) -> Dict[str, bool]:
    """The item-7 three criteria (P2-A5(3)); identical semantics to the frozen suite CLI."""
    rh = rec.get("reinhard")
    rh = rh if isinstance(rh, dict) else {}
    flags = rec.get("torch_backend_flags")
    flags = flags if isinstance(flags, dict) else {}
    threads = flags.get("main_process_num_threads")
    return {"lsf_job_null": rec.get("lsf_job", "missing") is None,
            "means_direct": _f32_bits_equal(rh.get("target_means", []), DIRECT_MEANS),
            "threads_121": type(threads) is int and threads == DIRECT_MAIN_THREADS}


def classify_record(rec: dict) -> Tuple[Optional[str], Dict[str, bool], str]:
    """Return (class or None, predicate, fit) under P2-A5(3)/P2-A13."""
    pred = direct_predicate(rec)
    rh = rec.get("reinhard")
    rh = rh if isinstance(rh, dict) else {}
    means, stds = rh.get("target_means", []), rh.get("target_stds", [])
    stds_pinned = _f32_bits_equal(stds, PINNED_STDS)
    if _f32_bits_equal(means, PINNED_MEANS) and stds_pinned:
        fit = "pinned_1thread"
    elif _f32_bits_equal(means, DIRECT_MEANS) and stds_pinned:
        fit = "direct_121thread"
    else:
        fit = "other"
    if all(pred.values()) and fit == "direct_121thread":
        return "direct", pred, fit
    if not any(pred.values()) and fit == "pinned_1thread":
        return "pinned", pred, fit
    return None, pred, fit


def arm_sources(cls: str) -> Dict[str, str]:
    """P2.3(4) routing."""
    if cls == "absent":
        return {c: "suite" for c in ARMS13}
    src = {c: "first_run" for c in FIRST_RUN_CONDITIONS}
    src[BLUR_IMAGE] = "suite"
    if cls == "direct":
        src["reinhard"] = "suite"
    return src


def _same_types(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_types(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_types(x, y) for x, y in zip(a, b))
    return True


def _load_pair(pkl: Path, rec: dict, n: int, keys: Sequence[str],
               fails: List[dict], tag: str):
    """sha/size binding, ``(dict, coords)`` layout, key set and float32 (n, DIM) arrays."""
    data = pkl.read_bytes()
    digest = _sha(data)
    ok = _expect(fails, f"{tag}.pickle_sha256", rec.get("pickle_sha256"), digest)
    ok &= _expect(fails, f"{tag}.pickle_bytes", rec.get("pickle_bytes"), len(data))
    if not ok:
        return None, None, digest
    obj = pickle.loads(data)
    del data
    if not (isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], dict)):
        _fail(fails, f"{tag}.pickle_layout", "(dict, coords)", type(obj).__name__)
        return None, None, digest
    feats, coords = obj
    if sorted(feats) != sorted(keys):
        _fail(fails, f"{tag}.pickle_keys", sorted(keys), sorted(map(str, feats)))
        return None, None, digest
    good = True
    for c in keys:
        x = feats[c]
        if not (isinstance(x, np.ndarray) and x.dtype == np.float32 and x.shape == (n, DIM)):
            _fail(fails, f"{tag}.array[{c}]", f"float32 ({n}, {DIM})",
                  f"{getattr(x, 'dtype', type(x).__name__)} {getattr(x, 'shape', None)}")
            good = False
    return (feats if good else None), coords, digest


def _spatial_shift_zero(rec: dict, fails: List[dict], tag: str) -> Optional[bool]:
    stim = rec.get("stimuli")
    els = [s for s in stim if isinstance(s, dict) and s.get("condition") == "spatial_shift"] \
        if isinstance(stim, list) else []
    if len(els) != 1:
        _fail(fails, f"{tag}.stimuli.spatial_shift_count", 1, len(els))
        return None
    p = els[0].get("parameters")
    p = p if isinstance(p, dict) else {}
    x, y = p.get("x_shift"), p.get("y_shift")
    return type(x) is int and x == 0 and type(y) is int and y == 0


def _listed(rec: dict, n: int, fails: List[dict], tag: str) -> Optional[List[int]]:
    if "reinhard_nonfinite_tile_indices" not in rec:
        _fail(fails, f"{tag}.reinhard_nonfinite_tile_indices", "present", "missing")
        return None
    v = rec["reinhard_nonfinite_tile_indices"]
    if not isinstance(v, list) or not all(type(i) is int and 0 <= i < n for i in v):
        _fail(fails, f"{tag}.reinhard_nonfinite_tile_indices", f"list of int in 0..{n - 1}", v)
        return None
    return [int(i) for i in v]


def _byte_diff_report(a: np.ndarray, b: np.ndarray, n: int) -> Dict[str, Any]:
    """A6-style report for two float32 (n, DIM) arrays whose contiguous bytes differ."""
    rows_a = np.ascontiguousarray(a).view(np.uint8).reshape(n, -1)
    rows_b = np.ascontiguousarray(b).view(np.uint8).reshape(n, -1)
    diff = int(np.count_nonzero((rows_a != rows_b).any(axis=1)))
    with np.errstate(invalid="ignore"):
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"differing_rows": diff,
            "max_abs_delta": float(np.nanmax(d)) if np.isfinite(d).any() else None,
            "dtype": [str(a.dtype), str(b.dtype)], "shape": [list(a.shape), list(b.shape)]}


# ---------------------------------------------------------------------------------------------
# Per-slide verification + SAMPLER (single pass, worker)
# ---------------------------------------------------------------------------------------------

def _verify_slide(task):
    """Worker: verify one slide, route its 13 arms and compute their SAMPLER from the same bytes.
    Never raises; returns (slide_id, failures, {arm: SamplerResult} or None, meta)."""
    sid = task[0]
    fails: List[dict] = []
    meta: Dict[str, Any] = {"slide_id": sid}
    try:
        out = _verify_and_sample(*task, fails=fails, meta=meta)
    except Exception as exc:                                   # noqa: BLE001
        _fail(fails, "exception", None, f"{type(exc).__name__}: {exc}",
              traceback=traceback.format_exc()[-3000:])
        out = None
    return sid, fails, (None if fails else out), meta


def _verify_and_sample(sid: str, row: Dict[str, str], plan_entry: Optional[dict],
                       first_dir: str, suite_dir: str, pins: Pins, *, fails: List[dict],
                       meta: Dict[str, Any]):
    first_dir, suite_dir = Path(first_dir), Path(suite_dir)
    n = int(row["feature_rows"])
    meta["n_tiles"] = n
    if plan_entry is None:
        _fail(fails, "plan.slide_entry", "present", "missing")
        return None
    branch = plan_entry.get("first_run_record")
    if branch not in ("present", "absent"):
        _fail(fails, "plan.first_run_record", "present|absent", branch)
        return None

    # (1) first-run pickle + record (P2.3(1), P2-A1)
    fr_feats = fr_coords = fr_rec = None
    fr_rec_path = first_dir / f"{sid}.record.json"
    fr_pkl_path = first_dir / f"{sid}.pkl"
    fr_rec_sha = fr_pkl_sha = None
    cls: Optional[str] = None
    if branch == "present":
        rb = fr_rec_path.read_bytes()
        fr_rec_sha = _sha(rb)
        fr_rec = json.loads(rb)
        meta.update(first_run_record=str(fr_rec_path), first_run_record_sha256=fr_rec_sha,
                    first_run_pickle=str(fr_pkl_path), first_run_host=fr_rec.get("host"),
                    first_run_gpu=fr_rec.get("gpu"), first_run_lsf_job=fr_rec.get("lsf_job"))
        t = "first_run"
        _expect(fails, f"{t}.schema", FIRST_RUN_SCHEMA, fr_rec.get("schema"))
        _expect(fails, f"{t}.slide_id", sid, fr_rec.get("slide_id"))
        _expect(fails, f"{t}.code_sha256", pins.extractor_code_sha256, fr_rec.get("code_sha256"))
        _expect(fails, f"{t}.manifest_sha256", pins.manifest_v1_sha256,
                fr_rec.get("manifest_sha256"))
        _expect(fails, f"{t}.queries_sha256", pins.queries_sha256, fr_rec.get("queries_sha256"))
        _expect(fails, f"{t}.encoder", ENCODER, fr_rec.get("encoder"))
        _expect(fails, f"{t}.dim", DIM, fr_rec.get("dim"))
        _expect(fails, f"{t}.n_tiles", n, fr_rec.get("n_tiles"))
        _expect(fails, f"{t}.conditions", list(FIRST_RUN_CONDITIONS), fr_rec.get("conditions"))
        qr = fr_rec.get("query_row")
        qr = qr if isinstance(qr, dict) else {}
        for k in ("resolved_path", "feature_sha256", "population_index"):
            _expect(fails, f"{t}.query_row.{k}", str(row[k]), qr.get(k))
        _expect(fails, f"{t}.coords_source", row["resolved_path"], fr_rec.get("coords_source"))
        _expect(fails, f"{t}.coords_source_sha256", row["feature_sha256"],
                fr_rec.get("coords_source_sha256"))
        fr_feats, fr_coords, fr_pkl_sha = _load_pair(fr_pkl_path, fr_rec, n,
                                                     FIRST_RUN_CONDITIONS, fails, t)
        meta["first_run_pickle_sha256"] = fr_pkl_sha
        # (3) class (P2-A5(3), P2-A13)
        cls, pred, fit = classify_record(fr_rec)
        meta.update(predicate=pred, fit=fit)
        if cls is None:
            _fail(fails, "class.predicate", "all three criteria agree and the fit is exactly "
                  "pinned (all false) or exactly direct (all true)", {"predicate": pred,
                                                                      "fit": fit})
        else:
            _expect(fails, "class.plan_first_run_class", cls, plan_entry.get("first_run_class"))
            _expect(fails, "class.plan_predicate", pred, plan_entry.get("predicate"))
    else:
        cls = "absent"
    meta["class"] = cls

    # (2) suite pickle + record (P2-A2)
    s_rec_path = suite_dir / f"{sid}.record.json"
    s_pkl_path = suite_dir / f"{sid}.pkl"
    rb = s_rec_path.read_bytes()
    s_rec_sha = _sha(rb)
    s_rec = json.loads(rb)
    meta.update(suite_record=str(s_rec_path), suite_record_sha256=s_rec_sha,
                suite_pickle=str(s_pkl_path))
    gpu = s_rec.get("gpu")
    gpu = gpu if isinstance(gpu, dict) else {}
    meta.update(suite_gpu_model=gpu.get("gpu_model"), suite_gpu_driver=gpu.get("driver"))
    t = "suite"
    _expect(fails, f"{t}.schema", SUITE_SCHEMA, s_rec.get("schema"))
    _expect(fails, f"{t}.slide_id", sid, s_rec.get("slide_id"))
    _expect(fails, f"{t}.encoder", ENCODER, s_rec.get("encoder"))
    _expect(fails, f"{t}.dim", DIM, s_rec.get("dim"))
    _expect(fails, f"{t}.n_tiles", n, s_rec.get("n_tiles"))
    _expect(fails, f"{t}.queries_sha256", pins.queries_sha256, s_rec.get("queries_sha256"))
    _expect(fails, f"{t}.plan_sha256", pins.plan_sha256, s_rec.get("plan_sha256"))
    _expect(fails, f"{t}.manifest_v2_sha256", pins.manifest_v2_sha256,
            s_rec.get("manifest_v2_sha256"))
    _expect(fails, f"{t}.manifest_v1_sha256", pins.manifest_v1_sha256,
            s_rec.get("manifest_v1_sha256"))
    _expect(fails, f"{t}.extractor_code_sha256", pins.extractor_code_sha256,
            s_rec.get("extractor_code_sha256"))
    _expect(fails, f"{t}.suite_code_sha256", pins.suite_code_sha256,
            s_rec.get("suite_code_sha256"))
    fv = s_rec.get("fingerprint_verification")
    fv = fv if isinstance(fv, dict) else {}
    _expect(fails, f"{t}.fingerprint_verification.mismatches", 0, fv.get("mismatches"))
    _expect(fails, f"{t}.fingerprint_verification.status", FINGERPRINT_STATUS, fv.get("status"))
    s_rh = s_rec.get("reinhard")
    s_rh = s_rh if isinstance(s_rh, dict) else {}
    if not (_f32_bits_equal(s_rh.get("target_means", []), PINNED_MEANS)
            and _f32_bits_equal(s_rh.get("target_stds", []), PINNED_STDS)):
        _fail(fails, f"{t}.reinhard_fit", {"target_means": PINNED_MEANS,
                                           "target_stds": PINNED_STDS},
              {"target_means": s_rh.get("target_means"), "target_stds": s_rh.get("target_stds")})
    s_fr = s_rec.get("first_run")
    s_fr = s_fr if isinstance(s_fr, dict) else {}
    status = "present" if branch == "present" else "absent"
    _expect(fails, f"{t}.first_run_record", status, s_rec.get("first_run_record"))
    _expect(fails, f"{t}.first_run.status", status, s_fr.get("status"))
    if branch == "present":
        rp = s_fr.get("record_path")
        if not (isinstance(rp, str)
                and os.path.realpath(rp) == os.path.realpath(str(fr_rec_path))):
            _fail(fails, f"{t}.first_run.record_path", os.path.realpath(str(fr_rec_path)), rp)
        _expect(fails, f"{t}.first_run.record_sha256", fr_rec_sha, s_fr.get("record_sha256"))
        if fr_pkl_sha is not None:
            _expect(fails, f"{t}.first_run.pickle_sha256", fr_pkl_sha, s_fr.get("pickle_sha256"))
        if cls is not None:
            _expect(fails, f"{t}.first_run.class", cls, s_fr.get("class"))
    s_keys = SUITE_CONDITIONS.get(cls) if cls is not None else None
    s_feats = s_coords = None
    if s_keys is not None:
        _expect(fails, f"{t}.conditions_encoded", list(s_keys), s_rec.get("conditions_encoded"))
        s_feats, s_coords, s_pkl_sha = _load_pair(s_pkl_path, s_rec, n, s_keys,
                                                  fails, t)
        meta["suite_pickle_sha256"] = s_pkl_sha

    # (6) identity == main bank (P2-A5(6), P2-A17)
    mb = Path(row["resolved_path"])
    meta["main_bank"] = str(mb)
    mb_size = mb.stat().st_size
    if not _expect(fails, "main_bank.st_size", int(row["feature_size"]), mb_size):
        return None
    data = mb.read_bytes()
    mb_sha = _sha(data)
    meta["main_bank_sha256"] = mb_sha
    if not _expect(fails, "main_bank.sha256", row["feature_sha256"], mb_sha):
        return None
    obj = pickle.loads(data)
    del data
    if not (isinstance(obj, tuple) and len(obj) == 2):
        _fail(fails, "main_bank.layout", "2-tuple (feats, coords)", type(obj).__name__)
        return None
    mb_feats, mb_coords = obj
    if not (isinstance(mb_feats, np.ndarray) and mb_feats.dtype == np.float32
            and mb_feats.shape == (n, DIM)):
        _fail(fails, "main_bank.array", f"float32 ndarray ({n}, {DIM})",
              f"{type(mb_feats).__name__} {getattr(mb_feats, 'dtype', None)} "
              f"{getattr(mb_feats, 'shape', None)}")
        return None
    if fr_feats is None and s_feats is None:
        return None
    sources = arm_sources(cls) if cls is not None else None
    if sources is None:
        return None
    arrays: Dict[str, np.ndarray] = {}
    for c in ARMS13:
        src = fr_feats if sources[c] == "first_run" else s_feats
        if src is None:
            return None
        arrays[c] = src[c]
    ident = arrays[IDENTITY]
    a_b = np.ascontiguousarray(ident).tobytes()
    m_b = np.ascontiguousarray(mb_feats).tobytes()
    eq_bytes = a_b == m_b
    meta["identity_array_equal_diagnostic"] = bool(np.array_equal(ident, mb_feats))
    meta["identity_equals_main_bank"] = eq_bytes
    if not eq_bytes:
        _fail(fails, "identity_equals_main_bank", "byte-identical", "differs",
              **_byte_diff_report(ident, mb_feats, n),
              array_equal=meta["identity_array_equal_diagnostic"])
    # present branch: the suite identity (paired partner of gaussian_blur_image's encoder run)
    # must also equal the main bank, hence the first-run identity (review 001 L4)
    if branch == "present" and s_feats is not None:
        s_ident = s_feats[IDENTITY]
        s_eq = np.ascontiguousarray(s_ident).tobytes() == m_b
        meta["suite_identity_equals_main_bank"] = s_eq
        if not s_eq:
            _fail(fails, "suite_identity_equals_main_bank", "byte-identical", "differs",
                  **_byte_diff_report(s_ident, mb_feats, n),
                  array_equal=bool(np.array_equal(s_ident, mb_feats)))
    # (5) coords (P2-A17): Python == and identical types, list[dict]
    coord_sets = [("main_bank", mb_coords)]
    if fr_coords is not None or branch == "present":
        coord_sets.append(("first_run", fr_coords))
    coord_sets.append(("suite", s_coords))
    if not (isinstance(mb_coords, list)
            and all(isinstance(e, dict) and "wsi_loc" in e for e in mb_coords)):
        _fail(fails, "coords.main_bank_type", "list[dict with key 'wsi_loc']",
              type(mb_coords).__name__)
    for name, cs in coord_sets[1:]:
        if not (cs == mb_coords and _same_types(cs, mb_coords)):
            _fail(fails, f"coords.{name}_equals_main_bank", "equal, identical types",
                  "differs" if cs != mb_coords else "type mismatch")

    # zero-shift null stimulus (P2-A7/P2-A15): from the record of the spatial_shift source
    ss_rec = fr_rec if sources["spatial_shift"] == "first_run" else s_rec
    zero = _spatial_shift_zero(ss_rec, fails, sources["spatial_shift"])
    meta["null_stimulus_spatial_shift"] = bool(zero)
    if zero:
        same = (np.ascontiguousarray(arrays["spatial_shift"]).tobytes()
                == np.ascontiguousarray(ident).tobytes())
        meta["spatial_shift_equals_identity"] = same
        if not same:
            _fail(fails, "zero_shift.spatial_shift_equals_identity", "byte-identical",
                  "differs (non-determinism signal)")

    # (7) non-finite Reinhard indices from the record of the source actually read
    r_src = sources["reinhard"]
    listed = _listed(fr_rec if r_src == "first_run" else s_rec, n, fails, r_src)
    meta["reinhard_nonfinite_source"] = r_src
    meta["reinhard_nonfinite_listed"] = listed
    meta["sources"] = sources
    if fails or listed is None:
        return None
    # descriptive: listed tiles whose identity feature rows are byte-identical (P2.4 15876 note)
    dup = {}
    for a in listed:
        same_rows = [b for b in listed if b != a and ident[a].tobytes() == ident[b].tobytes()]
        if same_rows:
            dup[str(a)] = same_rows
    meta["reinhard_listed_identity_duplicates"] = dup
    # SAMPLER (design decision A: float32 features, float64 percentiles, no cast)
    return {c: S.sampler(arrays[c], listed if c == "reinhard" else None) for c in ARMS13}


# ---------------------------------------------------------------------------------------------
# Evaluation over the 13 arms (thin re-implementation of E.evaluate's orchestration)
# ---------------------------------------------------------------------------------------------

def evaluate13(pop: pd.DataFrame, emb: Dict[str, np.ndarray], *, draw_count: int = 5000,
               seed: int = 123, threads: int = 1,
               hist_counts_from: Optional[Sequence[str]] = None) -> dict:
    conds = list(ARMS13)
    if set(emb) != set(conds):
        raise E.EvalError("embeddings must cover exactly the 13 arms")
    n = len(pop)
    qids = pop.slide_id.tolist()
    pids = pop.patient_id.tolist()
    labels = pop.diagnostic_group.tolist()
    groups = tuple(sorted(set(labels)))
    flat = {c: emb[c].reshape(n, -1) for c in conds}
    returned = {c: np.isfinite(flat[c]).all(axis=1) for c in conds}
    library = returned[IDENTITY]
    lib_idx = np.flatnonzero(library)
    draws = E.DrawCache(draw_count, seed)
    hist_keep = S.historical_keep(labels, counts_from=hist_counts_from)
    hist_labels = [S.historical_label(x) for x in labels]

    dist64, dist32, per, diags, timings = {}, {}, {}, {}, {}
    query_rows, rank_rows = [], []
    for c in conds:
        t0 = time.time()
        d = np.full((n, n), np.nan, dtype=np.float64)
        rows = np.flatnonzero(returned[c])
        if rows.size and lib_idx.size:
            d[np.ix_(rows, lib_idx)] = S.l1_distances(flat[c][rows], flat[IDENTITY][lib_idx],
                                                      threads=threads)
        t_l1 = time.time() - t0
        both = returned[c] & library
        diag_vals = d[np.flatnonzero(both), np.flatnonzero(both)]
        dd = {"diagonal_before_sentinel_max_abs": float(np.abs(diag_vals).max())
              if diag_vals.size else None,
              "diagonal_before_sentinel_n_zero": int(np.count_nonzero(diag_vals == 0))}
        p = np.asarray(pids, dtype=object)
        same = p[rows][:, None] == p[lib_idx][None, :]
        if rows.size and lib_idx.size:
            block = d[np.ix_(rows, lib_idx)]
            block[same] = S.SENTINEL
            d[np.ix_(rows, lib_idx)] = block
        dd["sentinel_cells"] = int(np.count_nonzero(same))
        d32 = d.astype(np.float32)
        fin = np.isfinite(d) & (d != S.SENTINEL)
        dd["float32_max_abs_rounding"] = float(np.abs(d32[fin].astype(np.float64) - d[fin]).max()) \
            if fin.any() else None
        st, rel, rk, ncand, ties, rdiag = E.rank_condition(pop, d, returned[c], library, d32)
        dd.update(rdiag)
        dd["boundary_ties"] = {str(k): int(sum(t[k] for t in ties)) for k in E.CUTOFFS}
        dd["returned"] = int(returned[c].sum())
        dd["intended"] = n
        diags[c] = dd
        vec = E.query_vectors(st, rel)
        per[c] = dict(statuses=st, vec=vec, returned=np.array([s == "returned" for s in st]))
        for i in range(n):
            for k in E.CUTOFFS:
                query_rows.append((c, i, qids[i], pids[i], labels[i], k, st[i],
                                   vec[("depth", k)][i],
                                   vec[("hit", k)][i] if st[i] == "returned" else None,
                                   vec[("precision", k)][i] if st[i] == "returned" else None,
                                   ncand[i], int(ties[i][k])))
            for r, (j, dv) in enumerate(rk[i], start=1):
                rank_rows.append((c, qids[i], r, qids[j], labels[j], int(labels[j] == labels[i]),
                                  dv))
        dist64[c] = d
        dist32[c] = d32
        timings[c] = {"l1_seconds": round(t_l1, 2), "total_seconds": round(time.time() - t0, 2)}

    est_rows, delta_rows = [], []
    for c in conds:
        ret = per[c]["returned"]
        d_ret = draws.get([pp for pp, r in zip(pids, ret) if r])
        for ep in E.ENDPOINTS:
            for k in E.CUTOFFS:
                s = kernel.summarize_metric(qids, pids, labels, groups, per[c]["vec"][(ep, k)],
                                            ret, d_ret)
                for g, e in zip((*groups, E.MACRO), (*s.group_estimates, s.macro)):
                    est_rows.append((c, g, ep, k, *E._est_cells(e), "returned",
                                     *E._status_cells(e)))
    for c in conds:
        if c == IDENTITY:
            continue
        both = per[c]["returned"] & per[IDENTITY]["returned"]
        d_both = draws.get([pp for pp, r in zip(pids, both) if r])
        for ep in E.ENDPOINTS:
            for k in E.CUTOFFS:
                ps = kernel.summarize_paired(qids, pids, labels, groups, per[c]["vec"][(ep, k)],
                                             per[IDENTITY]["vec"][(ep, k)], both, d_both)
                for g, e, a, b in zip((*groups, E.MACRO),
                                      (*ps.difference.group_estimates, ps.difference.macro),
                                      (*ps.arm_a.group_estimates, ps.arm_a.macro),
                                      (*ps.arm_b.group_estimates, ps.arm_b.macro)):
                    delta_rows.append((c, g, ep, k, e.point, e.ci_low, e.ci_high, e.n_queries,
                                       e.n_patients, a.point, b.point, "both_returned",
                                       *E._status_cells(e)))

    hist_rows = E.historical_table(pop, dist64, per, library, hist_keep, hist_labels, conds)
    return dict(groups=groups, conds=conds, dist32=dist32, per=per, returned=returned,
                library=library, estimates=est_rows, deltas=delta_rows, historical=hist_rows,
                query_rows=query_rows, rank_rows=rank_rows, diagnostics=diags, timings=timings,
                draws=draws, full_draws=draws.get(pids))


def _ordered_unique(values) -> List[str]:
    seen: List[str] = []
    for v in values:
        if not seen or seen[-1] != v:
            if v in seen:
                return seen + ["<non-contiguous:" + str(v) + ">"]
            seen.append(v)
    return seen


def assert_arms(tables: Dict[str, Sequence[str]], files: Sequence[str],
                record_conditions: Sequence[str]) -> None:
    """P2-A3: exactly the 13 names in the fixed order in every output (paired_deltas: the 12
    non-identity names, since a paired delta is defined against identity)."""
    want = list(ARMS13)
    problems = {}
    for name, col in tables.items():
        exp = want[1:] if name == "paired_deltas.csv" else want
        got = _ordered_unique(col)
        if got != exp:
            problems[name] = got
    for d in ("sampler", "distances"):
        got = [f[len(d) + 1:-4] for f in files if f.startswith(d + "/")]
        if got != want:
            problems[d + "/"] = got
    if list(record_conditions) != want:
        problems["run_record.conditions"] = list(record_conditions)
    if problems:
        raise E.EvalError(f"13-arm assertion failed before COMPLETE: {problems}")


# ---------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------

def _live_code() -> Dict[str, Path]:
    return {"adapter": Path(__file__).resolve(), "E": Path(E.__file__).resolve(),
            "S": Path(S.__file__).resolve(), "kernel": Path(kernel.__file__).resolve()}


def _write_failed(writer: E.Writer, global_fails: List[dict], slide_fails: Dict[str, List[dict]],
                  n_slides: int, argv: Sequence[str], code: Dict[str, str]) -> None:
    doc = {"schema": FAILED_SCHEMA,
           "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"),
           "argv": list(argv), "code_sha256": code,
           "n_slides_checked": n_slides, "n_slides_failed": len(slide_fails),
           "n_failures": len(global_fails) + sum(len(v) for v in slide_fails.values()),
           "global_failures": global_fails, "slide_failures": slide_fails,
           "note": "verification failed; no other output and no COMPLETE written (P2-A6/A12). "
                   "Any retry uses a new run_00N."}
    writer.put("VERIFICATION_FAILED.json", json.dumps(doc, indent=1, default=str).encode())


def run(*, population: Path, population_sha256: str, first_root: Path, suite_root: Path,
        plan: Path, out_dir: Path, adapter_sha256: str, workers: int = 1, threads: int = 1,
        draw_count: int = 5000, seed: int = 123, argv: Sequence[str] = (),
        pins: Pins = Pins()) -> int:
    """Returns 0 on COMPLETE, 2 after writing VERIFICATION_FAILED.json."""
    t_start = time.time()
    writer = E.Writer(Path(out_dir))           # P2-A12: before any input is read
    first_dir = Path(first_root) / "features" / ENCODER
    suite_dir = Path(suite_root) / "features" / ENCODER
    global_fails: List[dict] = []

    code_paths = _live_code()
    code_sha = {k: E.sha256_file(p) for k, p in code_paths.items()}
    pinned_code = {"adapter": adapter_sha256, "E": pins.e_sha256, "S": pins.s_sha256,
                   "kernel": pins.kernel_sha256}
    for k in code_paths:
        if code_sha[k] != pinned_code[k]:
            _fail(global_fails, f"code_sha256.{k}", pinned_code[k], code_sha[k],
                  path=str(code_paths[k]))
    live_code = {str(code_paths[k]): code_sha[k] for k in code_paths}

    # population (P2.2) and PLAN (P2-A5); an unreadable input is a verification failure too
    try:
        pop_sha = E.sha256_file(population)
        _expect(global_fails, "population.sha256_flag", population_sha256, pop_sha)
        _expect(global_fails, "population.sha256_pinned", pins.queries_sha256, pop_sha)
        pop = E.load_population(population)
        for col in ("feature_rows", "resolved_path", "feature_size", "feature_sha256"):
            if col not in pop.columns:
                _fail(global_fails, "population.column", col, "missing")
        plan_sha = E.sha256_file(plan)
        _expect(global_fails, "plan.sha256", pins.plan_sha256, plan_sha)
        plan_doc = json.loads(Path(plan).read_bytes())
        if not isinstance(plan_doc, dict):
            raise E.EvalError("PLAN.json is not a JSON object")
    except Exception as exc:                                   # noqa: BLE001
        _fail(global_fails, "inputs.readable", "population and PLAN readable",
              f"{type(exc).__name__}: {exc}")
        _write_failed(writer, global_fails, {}, 0, argv, live_code)
        return 2
    # review 001 L1: any exception while parsing PLAN/population/records after the Writer exists
    # is collected as a verification failure, so the root is never left empty
    n = len(pop)
    slide_fails: Dict[str, List[dict]] = {}
    try:
        plan_counts = plan_doc.get("counts") if isinstance(plan_doc.get("counts"), dict) else {}
        expected_counts = dict(pins.expected_counts)
        _expect(global_fails, "plan.counts", expected_counts,
                {k: plan_counts.get(k) for k in expected_counts})
        _expect(global_fails, "plan.expected_direct_count", pins.expected_direct_count,
                plan_doc.get("expected_direct_count"))
        plan_by_sid: Dict[str, dict] = {}
        for e in plan_doc.get("slides", []) if isinstance(plan_doc.get("slides"), list) else []:
            sid = e.get("slide_id") if isinstance(e, dict) else None
            if isinstance(e, dict) and type(sid) is not str:
                _fail(global_fails, "plan.slide_id", "str", repr(sid))
                continue
            if sid in plan_by_sid:
                _fail(global_fails, "plan.duplicate_slide", None, sid)
            plan_by_sid[sid] = e
        extra = sorted(map(str, set(plan_by_sid) - set(pop.slide_id)))
        if extra:
            _fail(global_fails, "plan.slides_not_in_population", [], extra[:50], n=len(extra))
        if global_fails and any(f["check"] == "population.column" for f in global_fails):
            _write_failed(writer, global_fails, {}, 0, argv, live_code)
            return 2

        cols = ("population_index", "slide_id", "feature_rows", "resolved_path", "feature_size",
                "feature_sha256")
        tasks = [(sid, {c: str(pop.at[i, c]) for c in cols}, plan_by_sid.get(sid), str(first_dir),
                  str(suite_dir), pins) for i, sid in enumerate(pop.slide_id)]
        pos = {s: i for i, s in enumerate(pop.slide_id)}
        emb = {c: np.full((n, len(S.PERCENTILES), DIM), np.nan, dtype=np.float64) for c in ARMS13}
        counts = {c: np.zeros((n, 4), dtype=np.int64) for c in ARMS13}
        metas: List[Optional[dict]] = [None] * n

        def absorb(res):
            sid, fails, out, meta = res
            i = pos[sid]
            metas[i] = meta
            if fails:
                slide_fails[sid] = fails
                return
            for c, r in out.items():
                counts[c][i] = (r.n_tiles, r.n_excluded_listed, r.n_excluded_nonfinite, r.n_used)
                if r.embedding is not None:
                    emb[c][i] = r.embedding

        t0 = time.time()
        pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None
        try:
            if pool is None:
                for t in tasks:
                    absorb(_verify_slide(t))
            else:
                for res in pool.imap_unordered(_verify_slide, tasks, chunksize=1):
                    absorb(res)
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
        t_sampler = time.time() - t0

        # global cross-checks (P2-A5 counts, P2-A7 zero-shift set)
        cls_counts = {k: sum(1 for m in metas if m and m.get("class") == k)
                      for k in expected_counts}
        _expect(global_fails, "class.recomputed_counts", expected_counts, cls_counts)
        zero_set = sorted(m["slide_id"] for m in metas
                          if m and m.get("null_stimulus_spatial_shift"))
        _expect(global_fails, "zero_shift.slide_set", sorted(pins.zero_shift_slides), zero_set)
    except Exception as exc:                                   # noqa: BLE001
        _fail(global_fails, "inputs.exception", None, f"{type(exc).__name__}: {exc}",
              traceback=traceback.format_exc()[-3000:])
    if global_fails or slide_fails:
        _write_failed(writer, global_fails, slide_fails, n, argv, live_code)
        return 2

    # P2-A12: code_frozen copies first, from bytes whose sha equals the asserted pin
    frozen = {}
    for k, p in code_paths.items():
        data = p.read_bytes()
        if _sha(data) != pinned_code[k]:
            raise E.EvalError(f"{p} changed after the code sha256 check")
        writer.put(f"code_frozen/{p.name}", data)
        frozen[f"code_frozen/{p.name}"] = _sha(data)

    idx_rows = []
    for c in ARMS13:
        writer.put(f"sampler/{c}.npy", E._npy_bytes(emb[c].astype(np.float32)))
        fin = np.isfinite(emb[c].reshape(n, -1)).all(axis=1)
        for i in range(n):
            t, li, nf, u = counts[c][i].tolist()
            idx_rows.append((c, i, int(pop.population_index[i]), pop.slide_id[i],
                             pop.patient_id[i], pop.diagnostic_group[i], t, li, nf, u,
                             int(fin[i])))
    writer.put("sampler_index.csv", E._csv_bytes(E.SAMPLER_INDEX_HEADER, idx_rows))
    t1 = time.time()
    res = evaluate13(pop, emb, draw_count=draw_count, seed=seed, threads=threads,
                     hist_counts_from=pop.diagnostic_group.tolist())
    t_eval = time.time() - t1
    for c in res["conds"]:
        writer.put(f"distances/{c}.npy", E._npy_bytes(res["dist32"][c]))
    cov = E.coverage_rows(pop, res["conds"], res["returned"], counts)
    writer.put("estimates.csv", E._csv_bytes(E.ESTIMATE_HEADER, res["estimates"]))
    writer.put("paired_deltas.csv", E._csv_bytes(E.DELTA_HEADER, res["deltas"]))
    writer.put("coverage.csv", E._csv_bytes(E.COVERAGE_HEADER, cov))
    writer.put("historical_topk_accuracy.csv", E._csv_bytes(E.HIST_HEADER, res["historical"]))
    writer.put("query_metrics.csv.gz", E._csv_bytes(E.QUERY_HEADER, res["query_rows"], True))
    writer.put("ranked_lists.csv.gz", E._csv_bytes(E.RANK_HEADER, res["rank_rows"], True))
    writer.put("patient_draws.npy", E._npy_bytes(
        np.asarray(res["full_draws"].multiplicities, dtype=np.int64)))

    def _s(v):
        return "" if v is None else v if isinstance(v, str) else json.dumps(v, sort_keys=True)

    man_rows = []
    for i, m in enumerate(metas):
        man_rows.append((
            int(pop.population_index[i]), m["slide_id"], m["class"],
            _s(m.get("first_run_pickle")), _s(m.get("first_run_pickle_sha256")),
            _s(m.get("first_run_record")), _s(m.get("first_run_record_sha256")),
            m["suite_pickle"], m["suite_pickle_sha256"], m["suite_record"],
            m["suite_record_sha256"], m["main_bank"], m["main_bank_sha256"], m["n_tiles"],
            json.dumps(m["reinhard_nonfinite_listed"]), m["reinhard_nonfinite_source"],
            int(m["identity_equals_main_bank"]), int(m["null_stimulus_spatial_shift"]),
            _s(m.get("first_run_host")), _s(m.get("first_run_gpu")),
            _s(m.get("first_run_lsf_job")), _s(m.get("suite_gpu_model")),
            _s(m.get("suite_gpu_driver")), *[m["sources"][c] for c in ARMS13]))
    writer.put("input_manifest.csv", E._csv_bytes(MANIFEST_HEADER, man_rows))

    # descriptive reporting items (P2.4, P2-A7..A9)
    per_arm = {c: {"first_run": sum(m["sources"][c] == "first_run" for m in metas),
                   "suite": sum(m["sources"][c] == "suite" for m in metas)} for c in ARMS13}
    null_sids = [m["slide_id"] for m in metas if m["null_stimulus_spatial_shift"]]
    excluded = []
    for m in metas:
        lst = m["reinhard_nonfinite_listed"]
        for ti in lst:
            excluded.append({"slide_id": m["slide_id"], "tile_index": ti,
                             "source": m["reinhard_nonfinite_source"],
                             "record": m["suite_record"] if m["reinhard_nonfinite_source"]
                             == "suite" else m["first_run_record"],
                             "identity_row_byte_equal_to_listed": m[
                                 "reinhard_listed_identity_duplicates"].get(str(ti), [])})
    gpu_prov: Dict[str, int] = {}
    for m in metas:
        key = f"{m.get('suite_gpu_model')} | driver {m.get('suite_gpu_driver')}"
        gpu_prov[key] = gpu_prov.get(key, 0) + 1
    fr_prov: Dict[str, int] = {}
    for m in metas:
        if m["class"] != "absent":
            key = f"{m.get('first_run_gpu')} | lsf_job {'null' if m.get('first_run_lsf_job') is None else 'set'}"
            fr_prov[key] = fr_prov.get(key, 0) + 1
    non_returns = {c: int(n - res["returned"][c].sum()) for c in ARMS13}
    record = {
        "schema": RUN_SCHEMA,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"),
        "argv": list(argv), "encoder": ENCODER, "feature_dim_used": DIM,
        "conditions": list(res["conds"]),
        "arm_labels": dict(ARM_LABELS),
        "population": {"path": str(population), "sha256": pop_sha, "intended_slides": n,
                       "evaluated_slides": n, "evaluated_patients": int(pop.patient_id.nunique()),
                       "groups": list(res["groups"])},
        "inputs": {"first_run_features": str(first_dir), "suite_features": str(suite_dir),
                   "plan": {"path": str(plan), "sha256": plan_sha},
                   "queries_sha256": pins.queries_sha256,
                   "manifest_v1_sha256": pins.manifest_v1_sha256,
                   "extractor_code_sha256": pins.extractor_code_sha256,
                   "suite_code_sha256": pins.suite_code_sha256},
        "manifest_v2_sha256": pins.manifest_v2_sha256,
        "input_manifest": "input_manifest.csv (first-run and suite pickle/record sha256 verified "
                          "against their records; main-bank sha256/size verified against the "
                          "population row; per-arm source and class per slide)",
        "code_sha256": live_code,
        "code_frozen": frozen,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": __import__("scipy").__version__,
                     "sklearn": __import__("sklearn").__version__,
                     "python": sys.version.split()[0]},
        "policy": {
            "sampler": "np.percentile(X, [5,15,...,95], axis=0) in float64 on kept rows; stored "
                       "float32; no float16 cast",
            "tile_exclusion": "reinhard: tiles in record reinhard_nonfinite_tile_indices; every "
                              "condition: rows with any non-finite value",
            "reinhard_index_source": "the record of the source actually read for the reinhard "
                                     "arm (first-run for pinned class, suite for direct/absent)",
            "non_return": "a slide/condition with zero usable tiles; empty_query status, kept in "
                          "coverage denominators",
            "references": "identity-arm SAMPLER of every evaluated slide with identity returned",
            "distance": "scipy cdist cityblock, float64, on flattened (10*D) vectors",
            "exclusion": "self and same population patient_id excluded by identity; cells set to "
                         "1e6 in the stored matrices",
            "tie_break": "stable ascending distance, population order",
            "precision_denominator": "k",
            "bootstrap": {"count": draw_count, "seed": seed, "generator": "PCG64",
                          "unit": "patient (one multiplicity across all of the patient's slides)",
                          "quantiles": [0.025, 0.975], "quantile_method": "linear",
                          "draws_per_population": "make_patient_draws(sorted included patients)"},
            "weighting": "slide-weighted group means (sorted groups); equal-weight macro",
            "paired": "condition minus identity on both-returned slides, one shared draw table",
            "historical_secondary": E.HIST_LABEL,
            "no_p_values": True,
            "arms": list(ARMS13),
            "routing": {"pinned": "gaussian_blur_image from suite; other 12 arms from first-run",
                        "direct": "reinhard and gaussian_blur_image from suite; other 11 arms "
                                  "from first-run",
                        "absent": "all 13 arms from suite"},
            "author_decisions": dict(AUTHOR_DECISIONS)},
        "routing": {"class_counts": cls_counts, "plan_counts": plan_counts,
                    "plan_expected_direct_count": plan_doc.get("expected_direct_count"),
                    "reinhard_pinned_fit_slides": n,
                    "statement": f"all {n} reinhard arms used are pinned-fit (first-run pinned "
                                 f"class {per_arm['reinhard']['first_run']}, suite "
                                 f"{per_arm['reinhard']['suite']})"},
        "per_arm_source": per_arm,
        "null_stimuli": {"arm": "spatial_shift", "slides": null_sids, "n": len(null_sids),
                         "tiles": int(sum(m["n_tiles"] for m in metas
                                          if m["null_stimulus_spatial_shift"])),
                         "spatial_shift_bit_equal_identity": all(
                             m.get("spatial_shift_equals_identity") for m in metas
                             if m["null_stimulus_spatial_shift"]),
                         "note": "x_shift == y_shift == 0: null stimuli (author decision B); "
                                 "paired delta 0 for these slides"},
        "reinhard_excluded_cells": excluded,
        "absent_branch_count": cls_counts.get("absent", 0),
        "identity_equals_main_bank": {
            "byte_equal": int(sum(m["identity_equals_main_bank"] for m in metas)), "of": n,
            "array_equal_diagnostic": int(sum(m["identity_array_equal_diagnostic"]
                                              for m in metas))},
        "first_run_provenance": {"per_slide": "input_manifest.csv first_run_host/first_run_gpu/"
                                              "first_run_lsf_job", "summary": fr_prov},
        "suite_gpu_provenance": gpu_prov,
        "coverage_summary": {"expected_returned_per_arm": n, "non_returns": non_returns,
                             "paired_results_readable": not any(non_returns.values()),
                             "note": "if any arm has a non-return no paired result is read until "
                                     "a hit-yield table over all intended queries is added by "
                                     "amendment (P2-A8)"},
        "disclosures": {
            "V11": V11_DISCLOSURE,
            "decision_A": DECISION_A_DISCLOSURE,
            "V1_retention": f"The {per_arm['reinhard']['suite'] - cls_counts.get('absent', 0)} "
                            "first-run direct-fit reinhard arrays (direct class) are retained "
                            "unchanged in the first-run tree and are not used by P2; those "
                            "slides' reinhard arm is read from the suite (pinned fit).",
            "historical_baseline_D": HIST_BASELINE_DISCLOSURE,
            "no_claim": NO_CLAIM},
        "draw_tables": res["draws"].describe(),
        "diagnostics": res["diagnostics"],
        "timings_seconds": {"verify_and_sampler_stage": round(t_sampler, 1),
                            "evaluate_stage": round(t_eval, 1),
                            "per_condition": res["timings"],
                            "total_before_record": round(time.time() - t_start, 1)},
        "workers": workers, "threads": threads,
        "peak_rss_mb_main_process": round(E.peak_rss_mb(), 1),
        "outputs": dict(writer.files),
    }
    if tuple(record) != RUN_RECORD_KEYS:
        raise E.EvalError("run_record key list differs from RUN_RECORD_KEYS")
    assert_arms({"estimates.csv": [r[0] for r in res["estimates"]],
                 "paired_deltas.csv": [r[0] for r in res["deltas"]],
                 "coverage.csv": [r[0] for r in cov],
                 "historical_topk_accuracy.csv": [r[1] for r in res["historical"]],
                 "sampler_index.csv": [r[0] for r in idx_rows],
                 "query_metrics.csv.gz": [r[0] for r in res["query_rows"]],
                 "ranked_lists.csv.gz": [r[0] for r in res["rank_rows"]]},
                list(writer.files), record["conditions"])
    writer.put("run_record.json", json.dumps(record, indent=1, default=str).encode())
    writer.put("COMPLETE", json.dumps({"files": writer.files}, indent=1, sort_keys=True).encode())
    return 0


def main(argv=None) -> int:
    out = E.default_out()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population", type=Path, default=(
        out / "r2_1_cpu_20261002" / "perturb_attempt_001" / "queries.csv") if out else None)
    ap.add_argument("--population-sha256", required=True)
    ap.add_argument("--first-root", type=Path,
                    default=out / "r2_1_perturb_20261003" if out else None)
    ap.add_argument("--suite-root", type=Path,
                    default=out / "r2_1_perturb_suite_20261005" if out else None)
    ap.add_argument("--plan", type=Path, default=None,
                    help="default <suite-root>/extension/PLAN.json")
    ap.add_argument("--out", type=Path, required=True, help="new empty run_00N directory")
    ap.add_argument("--adapter-sha256", required=True,
                    help="pinned sha256 of this adapter file (asserted against the live file)")
    ap.add_argument("--workers", type=int, default=1, help="processes for verify+SAMPLER")
    ap.add_argument("--threads", type=int, default=1, help="threads for the L1 stage")
    ap.add_argument("--draw-count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args(argv)
    if a.population is None or a.first_root is None or a.suite_root is None:
        ap.error("export PROSPER_OUT or pass --population/--first-root/--suite-root")
    plan = a.plan if a.plan is not None else a.suite_root / "extension" / "PLAN.json"
    argv_full = [sys.argv[0], *argv] if argv is not None else list(sys.argv)
    status = run(population=a.population, population_sha256=a.population_sha256,
                 first_root=a.first_root, suite_root=a.suite_root, plan=plan, out_dir=a.out,
                 adapter_sha256=a.adapter_sha256, workers=a.workers, threads=a.threads,
                 draw_count=a.draw_count, seed=a.seed, argv=argv_full)
    if status == 2:
        print(f"VERIFICATION FAILED: see {a.out / 'VERIFICATION_FAILED.json'}", file=sys.stderr)
    else:
        print(f"[perturb-eval13] COMPLETE -> {a.out}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
