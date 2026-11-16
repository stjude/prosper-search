"""R2.1 P3 comparator stage C2: same-stimulus 13-arm evaluation for UNI2 and Virchow2-fp32.

Specification: ``r2_1_comparators_20261008/declaration_004/DECLARATION.md`` (candidate 001 C2 as
amended by C2-06, C2-13, C2-15, C2-17, C3-02, C3-03, C3-08, C3-09, C3-10 and C4-01), and
``declaration_008`` C5-10 (G-01, G-02(ii), H-01, H-02): per-encoder suite CLI pin, per-slide
``environment.env_profile`` check, UNI2 weights/snapshot pins.

Every arm of every slide is read from the comparator suite tree
``<suite-root>/features/<encoder>/<slide>.pkl|.record.json`` (frozen ``extract``, all 13 arms;
there is no first-run tree and no PLAN). Per slide the record is verified, the raw suite arrays
are sliced ``[:, :dim_used]`` before SAMPLER (C2-06(a)), the identity arm must be byte-identical
to the bank-manifest pickle with type-strict equal coords (C2-03/C2-17), and the 13 SAMPLER
vectors are computed from the same bytes. Metrics, draws and writers are imported unchanged from
the frozen H-Optimus adapter (f19fe100...), E, S and the bootstrap kernel.

Hard-check results are the ``run_record.json`` key ``checks`` (C3-02); on any hard failure only
``FAILED.json`` is written (no metric file, no COMPLETE) and the exit status is 2. Neither file,
stdout nor stderr ever contains a metric magnitude.

The per-slide loader is the only routing surface. The production CLI uses :class:`SuiteLoader`;
:func:`run` accepts another loader object only through its Python API (test-only; used by the
REAL-gated H-Optimus replay, C2-06(b)/C3-10).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import functools
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import re
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import r2_1_hoptimus_main as kernel
from . import r2_1_perturb_eval13_cli as A
from . import r2_1_perturb_eval_cli as E
from . import r2_1_perturb_sampler as S

# C2-06(a): imported unchanged from the frozen adapter
from .r2_1_perturb_eval13_cli import (_byte_diff_report, _listed, _same_types,
                                      _spatial_shift_zero, assert_arms, evaluate13)

ARMS13 = A.ARMS13
IDENTITY = A.IDENTITY
RUN_SCHEMA = "r2_1_perturb_eval13_comparator_run_v1"
FAILED_SCHEMA = "r2_1_perturb_eval13_comparator_failed_v1"
FAILED_NAME = "FAILED.json"                     # C3-02
# A6 (impl review 001, I1-01c): run_001's single-threaded BLAS/hash environment
RUNTIME_ENV: Tuple[Tuple[str, str], ...] = (
    ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"), ("PYTHONHASHSEED", "0"))
ENV_EXIT = 4                                    # distinct from FAILED (2) and argparse (2)
PIPELINE_OF = {"uni2": "uni2_sampler", "virchow2": "virchow2_sampler",
               "h_optimus_1": "hoptimus_1"}     # h_optimus_1: test-only replay (C2-06(b))
DIM_USED = {"uni2": 1536, "virchow2": 1280, "h_optimus_1": 1536}
PRODUCTION_ENCODERS = ("uni2", "virchow2")
# C5-10: not 64-hex, so the uni2 per-slide check fails closed and run/main refuse --encoder uni2
# until the code_frozen_007 suite CLI sha is inserted by amendment
SUITE_CODE_PIN_SENTINEL = "__TO_BE_PINNED_007__"
BANK_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "population_index", "metadata_index", "slide_id", "patient_id", "diagnostic_group",
    "feature_rows", "feature_path", "resolved_path", "feature_size", "feature_sha256",
    "gpu_model")                                 # C2-17
POPULATION_SHARED_COLUMNS: Tuple[str, ...] = BANK_MANIFEST_COLUMNS[:6]
FIRST_RUN_MANIFEST_COLUMNS = ("first_run_pickle", "first_run_pickle_sha256", "first_run_record",
                              "first_run_record_sha256", "first_run_host", "first_run_gpu",
                              "first_run_lsf_job")
MANIFEST_HEADER: Tuple[str, ...] = tuple(c for c in A.MANIFEST_HEADER
                                         if c not in FIRST_RUN_MANIFEST_COLUMNS)
# C3-09: run_001's 34 keys minus first_run_provenance plus pipeline_id, main_bank, checks
RUN_RECORD_KEYS: Tuple[str, ...] = tuple(
    k for k in (*A.RUN_RECORD_KEYS[:5], "pipeline_id", *A.RUN_RECORD_KEYS[5:11], "main_bank",
                *A.RUN_RECORD_KEYS[11:33], "checks", A.RUN_RECORD_KEYS[33])
    if k != "first_run_provenance")
POLICY_DIFF_KEYS = ("routing", "reinhard_index_source")          # C3-09
SUITE_ROUTING = "all 13 arms from suite"
SUITE_REINHARD_INDEX_SOURCE = "suite"
# CSV outputs whose header line must be byte-equal to the reference run (C2 bullet 2)
HEADER_FILES = (("estimates.csv", E.ESTIMATE_HEADER), ("paired_deltas.csv", E.DELTA_HEADER),
                ("coverage.csv", E.COVERAGE_HEADER),
                ("historical_topk_accuracy.csv", E.HIST_HEADER),
                ("sampler_index.csv", E.SAMPLER_INDEX_HEADER),
                ("query_metrics.csv.gz", E.QUERY_HEADER),
                ("ranked_lists.csv.gz", E.RANK_HEADER))
NAMED_OUTPUTS = ("estimates.csv", "paired_deltas.csv", "coverage.csv",
                 "historical_topk_accuracy.csv", "sampler_index.csv", "query_metrics.csv.gz",
                 "ranked_lists.csv.gz", "patient_draws.npy", "input_manifest.csv",
                 "run_record.json")

C_N4_DISCLOSURE = (
    "run_001's H-Opt features come from mixed first-run and suite extractions (pixel-identical "
    "by fingerprint) whereas the comparators are all-suite; this asymmetry is disclosed wherever "
    "D is reported.")
C6_DISCLOSURE = (
    "UNI2 historical main-bank precision is inferred from the Stage0 inventory (float32) and ADR "
    "l.818 source review (no autocast in the UNI2 lib), not from a run record; the pilot byte "
    "test is the check. The Virchow2 comparator's identity arm is the adopted V1 fp32 bank, whose "
    "relationship to the published mixed fp16/fp32 bank is recorded in the V1 block and its "
    "technical note. No scientific statement about UNI2/Virchow2 robustness follows from this "
    "block.")


@dataclasses.dataclass(frozen=True)
class ComparatorPins:
    """Pinned bindings (no CLI waiver surface); synthetic tests pass their own instance."""
    queries_sha256: str = A.Pins.queries_sha256
    manifest_v1_sha256: str = A.Pins.manifest_v1_sha256
    manifest_v2_sha256: str = A.Pins.manifest_v2_sha256
    extractor_code_sha256: str = A.Pins.extractor_code_sha256
    suite_code_sha256: Tuple[Tuple[str, str], ...] = (                   # C5-10(b)
        ("virchow2", "cc023861719142aeb8993fb8ae4072a607bac2914386043f81c28ad443949879"),
        ("uni2", "39ddddd2d05bab3cf235595c31eb4c2e3289a5d45d4afca236b0c9fecf0d9ac1"))  # fill 2026-10-09 per release 008 (code_frozen_007 CLI sha)
    fingerprint_index_sha256: str = (
        "88d4a47a891744c77ceca0968656bcea81e673bbdba8c0a23364bce9ccf0b745")
    reference_run_record_sha256: str = (
        "9064605b80fb86a90d0bc821878aab809a6e61090410a6fdcf43fcceec621cb5")
    reference_complete_sha256: str = (
        "0b4352b9a9c383e80936a8fb46a81104dadf817cf25bde6c838dae24e1b44823")
    draw_table: Tuple[Tuple[str, Any], ...] = (
        ("n_patients", 1876),
        ("patients_sha256", "d203db7737201f78acbf0bf4764fcbf24d64f8d31a1f84546e9b780c5be46399"),
        ("multiplicities_sha256",
         "6aeaf004817321ec20c001812aa02fb9109c42c7e00fe8479b94aafeb685f6db"))
    zero_shift_slides: Tuple[str, ...] = ("12116", "15703", "6176", "7896")
    reinhard_cells: Tuple[Tuple[str, int], ...] = (("15821", 2328), ("15876", 2169),
                                                   ("15876", 2174))
    weights_sha256: Tuple[Tuple[str, str], ...] = (
        ("virchow2", "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"),
        ("uni2", "6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e"))  # G-02(ii)
    snapshot_revision: Tuple[Tuple[str, str], ...] = (                   # G-02(ii)
        ("uni2", "d517a8dd47902dd7c308b3c36f63bce47e7b9a43"),)
    adapter_sha256: str = "9591cbbc0cc1b9e556b7b0b9df9dbbd87185d756e91c3f2e3555078f3a9eb269"
    e_sha256: str = A.E_SHA256
    s_sha256: str = A.S_SHA256
    kernel_sha256: str = A.KERNEL_SHA256


class SuitePinRefusal(ValueError):
    """C5-10/H-02: suite CLI pin missing (production encoder) or still the sentinel; raised
    before any input is read."""


def suite_code_pin(pins: ComparatorPins, encoder: str) -> Optional[str]:
    """H-02: ``dict(pins.suite_code_sha256).get(encoder)``; ``None`` only for encoders reachable
    through the test loader API."""
    pin = dict(pins.suite_code_sha256).get(encoder)
    if encoder in PRODUCTION_ENCODERS and pin is None:
        raise SuitePinRefusal(f"no suite_code_sha256 pin for production encoder {encoder!r}")
    if pin == SUITE_CODE_PIN_SENTINEL:
        raise SuitePinRefusal(f"suite_code_sha256 pin for {encoder!r} is the unfilled sentinel "
                              f"{SUITE_CODE_PIN_SENTINEL!r} (inserted by amendment)")
    return pin


class Checks:
    """Hard-check accounting: per check id the number of evaluations and of failures; failure
    details hold only check ids, strings, ints and booleans (blinding; floats are dropped)."""

    def __init__(self) -> None:
        self.n: Dict[str, int] = {}
        self.fails: List[dict] = []
        self.slide_fails: List[dict] = []
        self.recorded: Dict[str, dict] = {}
        self.observed: Dict[str, str] = {}

    def touch(self, cid: str) -> None:
        self.n[cid] = self.n.get(cid, 0) + 1

    def fail(self, cid: str, expected: Any = None, observed: Any = None, **extra) -> None:
        self.fails.append({"check": cid, "expected": expected, "observed": observed, **extra})

    def ok(self, cid: str, cond: bool, expected: Any = None, observed: Any = None,
           **extra) -> bool:
        self.touch(cid)
        if not cond:
            self.fail(cid, expected, observed, **extra)
        return bool(cond)

    def expect(self, cid: str, expected: Any, observed: Any) -> bool:
        """Type-strict equality (as the frozen adapter's ``_expect``)."""
        return self.ok(cid, type(observed) is type(expected) and observed == expected,
                       expected, observed)

    def absorb(self, counts: Dict[str, int], fails: List[dict]) -> None:
        for k, v in counts.items():
            self.n[k] = self.n.get(k, 0) + v
        self.slide_fails.extend(fails)

    def record(self, cid: str, **counts) -> None:
        self.recorded[cid] = {"status": "recorded", **counts}

    def summary(self) -> Dict[str, dict]:
        failed: Dict[str, int] = {}
        for f in (*self.fails, *self.slide_fails):
            failed[f["check"]] = failed.get(f["check"], 0) + 1
        out = {}
        for cid in sorted(set(self.n) | set(failed)):
            nf = failed.get(cid, 0)
            out[cid] = {"status": "fail" if nf else "pass",
                        "n_checked": max(self.n.get(cid, 0), nf), "n_failed": nf}
            if cid in self.observed:
                out[cid]["observed"] = self.observed[cid]
        out.update(self.recorded)
        return out


_NUM = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+|nan|inf", re.I)


def _blind(v: Any, depth: int = 0) -> Any:
    """Drop every float (and decimal-looking text in free strings) from failure details."""
    if depth > 6:
        return "<nested>"
    if isinstance(v, bool) or v is None or isinstance(v, int):
        return v
    if isinstance(v, (float, np.floating)):
        return "<float omitted>"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, str):
        s = v if len(v) <= 2000 else v[:2000] + "...<truncated>"
        return s
    if isinstance(v, dict):
        return {str(k): _blind(x, depth + 1) for k, x in list(v.items())[:200]}
    if isinstance(v, (list, tuple, set)):
        return [_blind(x, depth + 1) for x in list(v)[:200]]
    return type(v).__name__


def _blind_failure(f: dict) -> dict:
    out = {}
    for k, v in f.items():
        if k == "max_abs_delta":               # magnitude from _byte_diff_report: never kept
            continue
        if k in ("traceback", "message"):
            out[k] = _NUM.sub("<num>", str(v))[-3000:]
            continue
        out[k] = _blind(v)
    return out


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(v: Any) -> bytes:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


# ---------------------------------------------------------------------------------------------
# Loader: the only routing surface (C2-06(a) "loader(routing) -> shared verify/sample/...")
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class SlideInputs:
    """What a loader hands to the shared per-slide verification for one slide."""
    arrays: Dict[str, np.ndarray]                 # the 13 raw arms (width >= dim_used)
    coord_sets: List[Tuple[str, Any]]             # coords that must equal the bank coords
    extra_identity: List[Tuple[str, np.ndarray]]  # further identity arrays bound to the bank
    spatial_shift_record: dict
    spatial_shift_source: str
    reinhard_record: dict
    reinhard_source: str
    reinhard_record_path: str
    sources: Dict[str, str]


def load_feature_pickle(path: Path, rec: dict, n: int, width: int, keys: Sequence[str],
                        ck: Checks, tag: str):
    """sha/size binding to the record, ``(dict, coords)`` layout, key set, float32 (n, width)."""
    data = Path(path).read_bytes()
    digest = _sha(data)
    ok = ck.expect(f"{tag}.pickle_sha256", rec.get("pickle_sha256"), digest)
    ok &= ck.expect(f"{tag}.pickle_bytes", rec.get("pickle_bytes"), len(data))
    if not ok:
        return None, None, digest
    obj = pickle.loads(data)
    del data
    if not ck.ok(f"{tag}.pickle_layout",
                 isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], dict),
                 "(dict, coords)", type(obj).__name__):
        return None, None, digest
    feats, coords = obj
    if not ck.ok(f"{tag}.pickle_keys", sorted(map(str, feats)) == sorted(keys), sorted(keys),
                 sorted(map(str, feats))):
        return None, coords, digest
    good = True
    for c in keys:
        x = feats[c]
        good &= ck.ok(f"{tag}.array[{c}]", isinstance(x, np.ndarray) and x.dtype == np.float32
                      and x.shape == (n, width), f"float32 ({n}, {width})",
                      f"{getattr(x, 'dtype', type(x).__name__)} {getattr(x, 'shape', None)}")
    return (feats if good else None), coords, digest


class SuiteLoader:
    """Production loader: all 13 arms of every slide from ``<suite-root>/features/<encoder>``."""
    policy_routing = SUITE_ROUTING                                  # C3-09
    policy_reinhard_index_source = SUITE_REINHARD_INDEX_SOURCE
    checks_weights = True
    input_manifest_text = (
        "input_manifest.csv (suite pickle/record sha256 verified against the record; record "
        "fingerprint shas verified against fingerprints/INDEX.json; bank pickle sha256/size "
        "verified against the bank manifest row; all 13 arms from the suite per slide)")

    def __init__(self, suite_root: Path, encoder: str, pins: ComparatorPins,
                 index: Dict[str, Tuple[Any, Any, Any]], index_path: str, index_sha256: str,
                 query_feature_sha256: Optional[Dict[str, str]] = None):
        self.query_feature_sha256 = dict(query_feature_sha256 or {})
        self.suite_root = str(suite_root)
        self.suite_dir = str(Path(suite_root) / "features" / encoder)
        self.encoder = encoder
        self.width = E.ENCODER_DIM[encoder]
        self.pins = pins
        self.index = index
        self.index_path = index_path
        self.index_sha256 = index_sha256

    @classmethod
    def open(cls, suite_root: Path, encoder: str, pins: ComparatorPins,
             slide_ids: Sequence[str], ck: Checks,
             query_feature_sha256: Optional[Dict[str, str]] = None) -> Optional["SuiteLoader"]:
        path = Path(suite_root) / "fingerprints" / "INDEX.json"
        data = path.read_bytes()
        digest = _sha(data)
        if not ck.expect("fingerprints.index_sha256", pins.fingerprint_index_sha256, digest):
            return None
        doc = json.loads(data)
        slides = doc.get("slides") if isinstance(doc, dict) else None
        if not ck.ok("fingerprints.index_layout", isinstance(slides, list), "slides list",
                     type(slides).__name__):
            return None
        index = {}
        for e in slides:
            if isinstance(e, dict) and isinstance(e.get("slide_id"), str):
                index[e["slide_id"]] = (e.get("npy_sha256"), e.get("content_sha256"),
                                        e.get("fp_json_sha256"))
        missing = [s for s in slide_ids if s not in index]
        ck.ok("fingerprints.index_covers_population", not missing, 0, len(missing))
        return cls(suite_root, encoder, pins, {s: index[s] for s in slide_ids if s in index},
                   str(path), digest, query_feature_sha256)

    def load(self, sid: str, row: Dict[str, str], n: int, ck: Checks,
             meta: Dict[str, Any]) -> Optional[SlideInputs]:
        d = Path(self.suite_dir)
        rec_path, pkl_path = d / f"{sid}.record.json", d / f"{sid}.pkl"
        rb = rec_path.read_bytes()
        rec = json.loads(rb)
        gpu = rec.get("gpu") if isinstance(rec.get("gpu"), dict) else {}
        meta.update({"class": "suite", "suite_record": str(rec_path),
                     "suite_record_sha256": _sha(rb), "suite_pickle": str(pkl_path),
                     "suite_gpu_model": gpu.get("gpu_model"), "suite_gpu_driver": gpu.get("driver"),
                     "first_run_record_field": rec.get("first_run_record"),
                     "plan_sha256_field": rec.get("plan_sha256")})
        t, p = "suite", self.pins
        ck.expect(f"{t}.schema", A.SUITE_SCHEMA, rec.get("schema"))
        ck.expect(f"{t}.slide_id", sid, rec.get("slide_id"))
        ck.expect(f"{t}.encoder", self.encoder, rec.get("encoder"))
        ck.expect(f"{t}.dim", self.width, rec.get("dim"))
        ck.expect(f"{t}.n_tiles", n, rec.get("n_tiles"))
        ck.expect(f"{t}.queries_sha256", p.queries_sha256, rec.get("queries_sha256"))
        # A7 (I1-10): the record's query_row is the queries.csv row the suite was built from
        # (frozen suite CLI copies it verbatim), so its feature_sha256 is the population row's
        # (H-Optimus inventory) sha, as the frozen adapter checks for first-run records.
        qr = rec.get("query_row") if isinstance(rec.get("query_row"), dict) else {}
        want_q = self.query_feature_sha256.get(sid)
        ck.ok(f"{t}.query_row_feature_sha256",
              isinstance(want_q, str) and len(want_q) == 64 and qr.get("feature_sha256") == want_q,
              want_q, qr.get("feature_sha256"))
        ck.expect(f"{t}.manifest_v2_sha256", p.manifest_v2_sha256, rec.get("manifest_v2_sha256"))
        ck.expect(f"{t}.manifest_v1_sha256", p.manifest_v1_sha256, rec.get("manifest_v1_sha256"))
        ck.expect(f"{t}.extractor_code_sha256", p.extractor_code_sha256,
                  rec.get("extractor_code_sha256"))
        ck.expect(f"{t}.suite_code_sha256", dict(p.suite_code_sha256)[self.encoder],
                  rec.get("suite_code_sha256"))
        if self.encoder == "uni2":                                      # C5-10(c)
            ck.expect(f"{t}.environment.env_profile", "uni2_historical",
                      rec["environment"].get("env_profile"))
        elif self.encoder == "virchow2":
            ck.ok(f"{t}.environment.env_profile", "env_profile" not in rec["environment"],
                  "absent", rec["environment"].get("env_profile"))
        # every slide must carry all 13 conditions (C2 bullet 1: refuse otherwise)
        ck.expect(f"{t}.conditions_encoded", list(ARMS13), rec.get("conditions_encoded"))
        fv = rec.get("fingerprint_verification")
        fv = fv if isinstance(fv, dict) else {}
        ck.expect(f"{t}.fingerprint_verification.status", A.FINGERPRINT_STATUS, fv.get("status"))
        ck.expect(f"{t}.fingerprint_verification.mismatches", 0, fv.get("mismatches"))
        entry = self.index.get(sid)
        if ck.ok("fingerprints.index_entry", entry is not None, "present", "missing"):
            # C3-04 mapping fp_npy->npy, fp_content->content, fp_json->fp_json
            for field, ref in zip(("fp_npy_sha256", "fp_content_sha256", "fp_json_sha256"),
                                  entry):
                ck.ok(f"{t}.fingerprint_verification.{field}",
                      isinstance(ref, str) and fv.get(field) == ref, ref, fv.get(field))
        rh = rec.get("reinhard") if isinstance(rec.get("reinhard"), dict) else {}
        ck.ok(f"{t}.reinhard_fit",
              A._f32_bits_equal(rh.get("target_means", []), A.PINNED_MEANS)
              and A._f32_bits_equal(rh.get("target_stds", []), A.PINNED_STDS),
              "pinned target means/stds (float32 bits)", "differs")
        em = rec.get("encoder_meta") if isinstance(rec.get("encoder_meta"), dict) else {}
        w = em.get("weights") if isinstance(em.get("weights"), dict) else {}
        ws = w.get("sha256")
        meta["weights_sha256"] = ws
        ck.ok(f"{t}.encoder_meta.weights.sha256", isinstance(ws, str) and len(ws) == 64,
              "64-hex sha256", type(ws).__name__)
        rev = dict(p.snapshot_revision).get(self.encoder)              # G-02(ii)
        if rev is not None:
            ck.expect(f"{t}.encoder_meta.weights.snapshot_revision", rev,
                      w.get("snapshot_revision"))
        feats, coords, pkl_sha = load_feature_pickle(pkl_path, rec, n, self.width, ARMS13, ck, t)
        meta["suite_pickle_sha256"] = pkl_sha
        if feats is None:
            return None
        return SlideInputs(arrays=feats, coord_sets=[("suite", coords)], extra_identity=[],
                           spatial_shift_record=rec, spatial_shift_source="suite",
                           reinhard_record=rec, reinhard_source="suite",
                           reinhard_record_path=str(rec_path),
                           sources={c: "suite" for c in ARMS13})

    def inputs(self) -> dict:
        return {"suite_features": self.suite_dir,
                "fingerprint_index": {"path": self.index_path, "sha256": self.index_sha256}}

    def routing_summary(self, metas: Sequence[dict], per_arm: dict, n: int) -> dict:
        return {"class_counts": {"suite": sum(m["class"] == "suite" for m in metas)},
                "reinhard_pinned_fit_slides": n,
                "statement": f"all 13 arms of all {n} slides from the comparator suite tree; all "
                             f"{n} reinhard arms are pinned-fit (suite {per_arm['reinhard']['suite']})"}

    def absent_branch_count(self, metas: Sequence[dict]) -> int:
        # slides whose 13 arms all come from the suite (run_001's "absent" branch meaning)
        return sum(all(v == "suite" for v in m["sources"].values()) for m in metas)

    def disclosures(self, metas: Sequence[dict], per_arm: dict) -> dict:
        return {"decision_A": A.DECISION_A_DISCLOSURE,
                "historical_baseline_D": A.HIST_BASELINE_DISCLOSURE,
                "no_claim": A.NO_CLAIM,
                "C_N4_asymmetry": C_N4_DISCLOSURE,
                "C6": C6_DISCLOSURE}


# ---------------------------------------------------------------------------------------------
# Shared per-slide verification + SAMPLER (worker)
# ---------------------------------------------------------------------------------------------

_WORKER: Dict[str, Any] = {}


def _init_worker(loader, dim_used: int) -> None:
    _WORKER["loader"] = loader
    _WORKER["dim_used"] = dim_used


def _slide_task(task):
    """Worker: never raises; returns (slide_id, check counts, failures, SAMPLER dict|None, meta)."""
    sid, row = task
    ck = Checks()
    meta: Dict[str, Any] = {"slide_id": sid}
    try:
        out = verify_and_sample(sid, row, _WORKER["loader"], _WORKER["dim_used"], ck, meta)
    except Exception as exc:                                   # noqa: BLE001
        ck.fail("exception", None, type(exc).__name__, message=str(exc),
                traceback=traceback.format_exc())
        out = None
    return sid, ck.n, ck.fails, (None if ck.fails else out), meta


def _helper_checks(ck: Checks, cid: str, fn, *args):
    """Run a frozen adapter helper that appends A-style failure dicts; account its check id."""
    fl: List[dict] = []
    res = fn(*args, fl, cid.split(".")[0])
    ck.touch(cid)
    ck.fails.extend(fl)
    return res


def verify_and_sample(sid: str, row: Dict[str, str], loader, dim_used: int, ck: Checks,
                      meta: Dict[str, Any]):
    n = int(row["feature_rows"])
    meta["n_tiles"] = n
    inp = loader.load(sid, row, n, ck, meta)

    # bank pickle bound to the bank-manifest row (C2-17); layout (feats, coords)
    mb = Path(row["resolved_path"])
    meta["main_bank"] = str(mb)
    if not ck.expect("main_bank.st_size", int(row["feature_size"]), mb.stat().st_size):
        return None
    data = mb.read_bytes()
    mb_sha = _sha(data)
    meta["main_bank_sha256"] = mb_sha
    if not ck.expect("main_bank.sha256", row["feature_sha256"], mb_sha):
        return None
    obj = pickle.loads(data)
    del data
    if not ck.ok("main_bank.layout", isinstance(obj, tuple) and len(obj) == 2,
                 "2-tuple (feats, coords)", type(obj).__name__):
        return None
    mb_feats, mb_coords = obj
    if not ck.ok("main_bank.array", isinstance(mb_feats, np.ndarray)
                 and mb_feats.dtype == np.float32 and mb_feats.shape == (n, dim_used),
                 f"float32 ndarray ({n}, {dim_used})",
                 f"{type(mb_feats).__name__} {getattr(mb_feats, 'dtype', None)} "
                 f"{getattr(mb_feats, 'shape', None)}"):
        return None
    ck.ok("coords.main_bank_type", isinstance(mb_coords, list)
          and all(isinstance(e, dict) and "wsi_loc" in e for e in mb_coords),
          "list[dict with key 'wsi_loc']", type(mb_coords).__name__)
    if inp is None:
        if not ck.fails:
            ck.fail("loader.inputs", "slide inputs", None)
        return None
    if not ck.ok("arrays.width_at_least_dim_used",
                 all(inp.arrays[c].ndim == 2 and inp.arrays[c].shape[1] >= dim_used
                     for c in ARMS13), f">= {dim_used}",
                 sorted({int(inp.arrays[c].shape[-1]) for c in ARMS13})):
        return None
    # C2-06(a): slice the raw suite array [:, :dim_used] before SAMPLER (and before identity)
    arrays = {c: np.ascontiguousarray(inp.arrays[c][:, :dim_used]) for c in ARMS13}
    inp.arrays = {}
    ident = arrays[IDENTITY]
    mb_bytes = np.ascontiguousarray(mb_feats, dtype=np.float32).tobytes()

    def bound_to_bank(name: str, arr: np.ndarray) -> bool:
        # C2-03 rule: np.ascontiguousarray(x[:, :dim_used], dtype=float32).tobytes() == bank
        eq = np.ascontiguousarray(arr, dtype=np.float32).tobytes() == mb_bytes
        meta[f"{name}_equals_main_bank"] = eq
        meta[f"{name}_array_equal_diagnostic"] = bool(np.array_equal(arr, mb_feats))
        if not ck.ok(f"{name}_equals_main_bank", eq, "byte-identical", "differs"):
            rep = _byte_diff_report(arr, mb_feats, n)
            ck.fails[-1].update(differing_rows=rep["differing_rows"], dtype=rep["dtype"],
                                shape=rep["shape"])
        return eq

    bound_to_bank("identity", ident)
    for name, arr in inp.extra_identity:
        bound_to_bank(f"{name}_identity", np.ascontiguousarray(arr[:, :dim_used]))
    for name, cs in inp.coord_sets:
        ck.ok(f"coords.{name}_equals_main_bank", cs == mb_coords and _same_types(cs, mb_coords),
              "equal, identical types", "differs" if cs != mb_coords else "type mismatch")

    # zero-shift null stimulus: spatial_shift bytes == identity bytes (C2-06(d))
    zero = _helper_checks(ck, f"{inp.spatial_shift_source}.stimuli.spatial_shift_count",
                          lambda fl, tag: _spatial_shift_zero(inp.spatial_shift_record, fl, tag))
    meta["null_stimulus_spatial_shift"] = bool(zero)
    if zero:
        same = arrays["spatial_shift"].tobytes() == ident.tobytes()
        meta["spatial_shift_equals_identity"] = same
        ck.ok("zero_shift.spatial_shift_equals_identity", same, "byte-identical",
              "differs (non-determinism signal)")

    # non-finite Reinhard indices from the record of the source actually read (C2-13)
    listed = _helper_checks(ck, f"{inp.reinhard_source}.reinhard_nonfinite_tile_indices",
                            lambda fl, tag: _listed(inp.reinhard_record, n, fl, tag))
    meta["reinhard_nonfinite_source"] = inp.reinhard_source
    meta["reinhard_nonfinite_listed"] = listed
    meta["reinhard_record"] = inp.reinhard_record_path
    meta["sources"] = dict(inp.sources)
    if ck.fails or listed is None:
        return None
    dup = {}
    for a in listed:
        same_rows = [b for b in listed if b != a and ident[a].tobytes() == ident[b].tobytes()]
        if same_rows:
            dup[str(a)] = same_rows
    meta["reinhard_listed_identity_duplicates"] = dup
    # SAMPLER (design decision A) on the sliced float32 arrays, same bytes as identity check
    return {c: S.sampler(arrays[c], listed if c == "reinhard" else None) for c in ARMS13}


# ---------------------------------------------------------------------------------------------
# Reference run, policy and global checks
# ---------------------------------------------------------------------------------------------

def _first_line(path: Path) -> bytes:
    import gzip
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as fh:
        return fh.readline()


def read_reference(ref_dir: Path, pins: ComparatorPins, ck: Checks) -> Optional[dict]:
    """Read run_001's COMPLETE, run_record.json and the header lines of its CSV outputs; every
    opened file's sha is verified against COMPLETE (and the two pinned shas)."""
    ref_dir = Path(ref_dir)
    cb = (ref_dir / "COMPLETE").read_bytes()
    if not ck.expect("reference.complete_sha256", pins.reference_complete_sha256, _sha(cb)):
        return None
    files = json.loads(cb).get("files")
    if not ck.ok("reference.complete_layout", isinstance(files, dict), "files dict",
                 type(files).__name__):
        return None
    rb = (ref_dir / "run_record.json").read_bytes()
    ck.expect("reference.run_record_sha256", pins.reference_run_record_sha256, _sha(rb))
    ck.ok("reference.file_sha256[run_record.json]",
          files.get("run_record.json", {}).get("sha256") == _sha(rb), "COMPLETE entry", "differs")
    rec = json.loads(rb)
    headers = {}
    for name, _ in HEADER_FILES:
        p = ref_dir / name
        ok = ck.ok(f"reference.file_sha256[{name}]", name in files
                   and E.sha256_file(p) == files[name].get("sha256"), "COMPLETE entry", "differs")
        headers[name] = _first_line(p) if ok else None
    return {"path": str(ref_dir), "complete_sha256": _sha(cb), "files": files,
            "run_record_sha256": _sha(rb), "record": rec, "headers": headers}


def build_policy(draw_count: int, seed: int, routing: Any, reinhard_index_source: Any) -> dict:
    """run_001's policy literal (frozen adapter l.915-945) with the two C3-09 keys replaced."""
    return {
        "sampler": "np.percentile(X, [5,15,...,95], axis=0) in float64 on kept rows; stored "
                   "float32; no float16 cast",
        "tile_exclusion": "reinhard: tiles in record reinhard_nonfinite_tile_indices; every "
                          "condition: rows with any non-finite value",
        "reinhard_index_source": reinhard_index_source,
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
        "routing": routing,
        "author_decisions": dict(A.AUTHOR_DECISIONS)}


def check_policy(policy: dict, ref_policy: Any, loader, ck: Checks) -> None:
    ref_policy = ref_policy if isinstance(ref_policy, dict) else {}
    ck.ok("policy.key_set", set(policy) == set(ref_policy), len(ref_policy), len(policy))
    for k in sorted(set(policy) | set(ref_policy)):
        if k in POLICY_DIFF_KEYS:
            continue
        ck.ok(f"policy.{k}", k in policy and k in ref_policy
              and _canon(policy[k]) == _canon(ref_policy[k]), "canonical JSON equal to reference",
              "differs")
    # C3-09: only these two keys may differ (routing may change type); values from the loader
    ck.expect("policy.routing", loader.policy_routing, policy.get("routing"))
    ck.expect("policy.reinhard_index_source", loader.policy_reinhard_index_source,
              policy.get("reinhard_index_source"))


def planned_output_names() -> set:
    return ({*NAMED_OUTPUTS} | {f"sampler/{c}.npy" for c in ARMS13}
            | {f"distances/{c}.npy" for c in ARMS13})


def read_bank_manifest(path: Path, pop: pd.DataFrame, ck: Checks) -> Optional[List[dict]]:
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not ck.ok("bank_manifest.columns", bool(rows) and tuple(rows[0]) == BANK_MANIFEST_COLUMNS,
                 list(BANK_MANIFEST_COLUMNS), rows[0] if rows else None):
        return None
    out = [dict(zip(BANK_MANIFEST_COLUMNS, r)) for r in rows[1:]]
    if not ck.ok("bank_manifest.row_count", len(out) == len(pop) and all(
            len(r) == len(BANK_MANIFEST_COLUMNS) for r in rows[1:]), len(pop), len(out)):
        return None
    for col in POPULATION_SHARED_COLUMNS:
        want = [str(v) for v in pop[col].tolist()]
        got = [r[col] for r in out]
        bad = sum(a != b for a, b in zip(want, got))
        ck.ok(f"bank_manifest.rows[{col}]", bad == 0, 0, bad)
    return out


# ---------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------

def _code_paths() -> Dict[str, Path]:
    return {"comparator": Path(__file__).resolve(), "adapter": Path(A.__file__).resolve(),
            "E": Path(E.__file__).resolve(), "S": Path(S.__file__).resolve(),
            "kernel": Path(kernel.__file__).resolve()}


def _write_failed(writer: E.Writer, ck: Checks, slide_fails: Dict[str, List[dict]],
                  n_slides: int, argv: Sequence[str], code: Dict[str, str]) -> None:
    doc = {"schema": FAILED_SCHEMA,
           "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"),
           "argv": list(argv), "code_sha256": code,
           "n_slides_checked": n_slides, "n_slides_failed": len(slide_fails),
           "n_failures": len(ck.fails) + sum(len(v) for v in slide_fails.values()),
           "checks": ck.summary(),
           "global_failures": [_blind_failure(f) for f in ck.fails],
           "slide_failures": {s: [_blind_failure(f) for f in fs]
                              for s, fs in sorted(slide_fails.items())},
           "note": "hard check failed; no metric output and no COMPLETE written (C2-06(d), "
                   "C3-02). Any retry uses a new run_00N. No magnitudes are recorded."}
    writer.put(FAILED_NAME, json.dumps(doc, indent=1, default=str).encode())


def _failed_ids(ck: Checks) -> List[str]:
    return sorted(k for k, v in ck.summary().items() if v["status"] == "fail")


def run(*, encoder: str, pipeline_id: str, dim_used: int, suite_root: Path, bank_manifest: Path,
        bank_manifest_sha256: str, population: Path, reference_run: Path, out_dir: Path,
        workers: int = 1, threads: int = 1, draw_count: int = 5000, seed: int = 123,
        argv: Sequence[str] = (), pins: ComparatorPins = ComparatorPins(),
        loader=None) -> int:
    """Returns 0 after COMPLETE, 2 after FAILED.json. ``loader`` is the test-only routing API
    (C2-06(b)); the production CLI never passes it."""
    if PIPELINE_OF.get(encoder) != pipeline_id:
        raise ValueError(f"pipeline id {pipeline_id!r} does not match encoder {encoder!r}")
    if DIM_USED.get(encoder) != dim_used:
        raise ValueError(f"--dim-used {dim_used} is not the contract width for {encoder!r}")
    if loader is None and encoder not in PRODUCTION_ENCODERS:
        raise ValueError(f"encoder {encoder!r} is only available through the test loader API")
    suite_code_pin(pins, encoder)                # H-02 / C5-10 sentinel: before any input
    t_start = time.time()
    writer = E.Writer(Path(out_dir))           # before any input is read (as run_001)
    ck = Checks()
    paths = _code_paths()
    code_sha = {k: E.sha256_file(p) for k, p in paths.items()}
    pinned = {"adapter": pins.adapter_sha256, "E": pins.e_sha256, "S": pins.s_sha256,
              "kernel": pins.kernel_sha256}
    for k, v in pinned.items():
        ck.expect(f"code_sha256.{k}", v, code_sha[k])
    for var, want in RUNTIME_ENV:              # A6: recorded as checks (run_record key set fixed)
        got = os.environ.get(var)
        ck.observed[f"env.{var}"] = "<unset>" if got is None else got
        ck.expect(f"env.{var}", want, got)
    live_code = {str(paths[k]): code_sha[k] for k in paths}
    if ck.fails:                     # hard fail at startup (frozen modules changed or A6 env)
        _write_failed(writer, ck, {}, 0, argv, live_code)
        return 2

    slide_fails: Dict[str, List[dict]] = {}
    n = 0
    try:
        pop_sha = E.sha256_file(population)
        ck.expect("population.sha256", pins.queries_sha256, pop_sha)
        pop = E.load_population(population)
        n = len(pop)
        bm_sha = E.sha256_file(bank_manifest)
        ck.expect("bank_manifest.sha256", bank_manifest_sha256, bm_sha)
        bank_rows = read_bank_manifest(bank_manifest, pop, ck)
        ref = read_reference(reference_run, pins, ck)
        if ref is None or bank_rows is None:
            raise _Stop()
        rr = ref["record"]
        groups = sorted(set(pop.diagnostic_group.tolist()))
        ck.ok("reference.conditions", rr.get("conditions") == list(ARMS13), list(ARMS13),
              rr.get("conditions"))
        rpop = rr.get("population") if isinstance(rr.get("population"), dict) else {}
        ck.expect("reference.population.sha256", pop_sha, rpop.get("sha256"))
        ck.ok("reference.population.groups", rpop.get("groups") == groups, len(groups),
              len(rpop.get("groups") or []))
        ref_draws = rr.get("draw_tables") if isinstance(rr.get("draw_tables"), list) else []
        ck.ok("reference.draw_table", ref_draws[:1] == [dict(pins.draw_table)],
              "pinned draw-table hashes", "differs")
        cache = E.DrawCache(draw_count, seed)
        cache.get(pop.patient_id.tolist())
        ck.ok("draws.full_population_precheck", cache.describe() == [dict(pins.draw_table)],
              "pinned draw-table hashes", "differs")
        policy = None
        if loader is None:
            qsha = (dict(zip(pop.slide_id, pop["feature_sha256"]))
                    if "feature_sha256" in pop.columns else {})
            loader = SuiteLoader.open(Path(suite_root), encoder, pins, pop.slide_id.tolist(), ck,
                                      qsha)
            if loader is None:
                raise _Stop()
        policy = build_policy(draw_count, seed, loader.policy_routing,
                              loader.policy_reinhard_index_source)
        check_policy(policy, rr.get("policy"), loader, ck)
        for name, header in HEADER_FILES:
            want = E._csv_bytes(header, [])
            ck.ok(f"header[{name}]", ref["headers"].get(name) == want, "reference header bytes",
                  "differs")
        ref_names = {k for k in ref["files"] if not k.startswith("code_frozen/")}
        ck.ok("complete.planned_name_set", planned_output_names() == ref_names,
              len(ref_names), len(planned_output_names()))
        if ck.fails:
            raise _Stop()

        tasks = [(r["slide_id"], r) for r in bank_rows]
        pos = {s: i for i, s in enumerate(pop.slide_id)}
        emb = {c: np.full((n, len(S.PERCENTILES), dim_used), np.nan, dtype=np.float64)
               for c in ARMS13}
        counts = {c: np.zeros((n, 4), dtype=np.int64) for c in ARMS13}
        metas: List[Optional[dict]] = [None] * n

        def absorb(res):
            sid, cnt, fails, out, meta = res
            i = pos[sid]
            metas[i] = meta
            ck.absorb(cnt, fails)
            if fails:
                slide_fails[sid] = fails
                return
            for c, r in out.items():
                counts[c][i] = (r.n_tiles, r.n_excluded_listed, r.n_excluded_nonfinite, r.n_used)
                if r.embedding is not None:
                    emb[c][i] = r.embedding

        t0 = time.time()
        _init_worker(loader, dim_used)
        pool = (mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                             initargs=(loader, dim_used))
                if workers > 1 else None)
        try:
            if pool is None:
                for t in tasks:
                    absorb(_slide_task(t))
            else:
                for res in pool.imap_unordered(_slide_task, tasks, chunksize=1):
                    absorb(res)
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
        t_sampler = time.time() - t0
        if slide_fails:
            raise _Stop()
        global_slide_checks(metas, ref, pins, encoder, loader, ck)
        if ck.fails:
            raise _Stop()
    except _Stop:
        pass
    except Exception as exc:                                   # noqa: BLE001
        ck.fail("inputs.exception", None, type(exc).__name__, message=str(exc),
                traceback=traceback.format_exc())
    if ck.fails or slide_fails:
        _write_failed(writer, ck, slide_fails, n, argv, live_code)
        return 2
    return _evaluate_and_write(
        writer=writer, ck=ck, pop=pop, pop_sha=pop_sha, population=population, emb=emb,
        counts=counts, metas=metas, ref=ref, loader=loader, policy=policy, encoder=encoder,
        pipeline_id=pipeline_id, dim_used=dim_used, bank_manifest=bank_manifest, bm_sha=bm_sha,
        pins=pins, paths=paths, code_sha=code_sha, live_code=live_code, draw_count=draw_count,
        seed=seed, threads=threads, workers=workers, argv=argv, t_start=t_start,
        t_sampler=t_sampler)


class _Stop(Exception):
    """Internal: stop before metrics; the failures are already in the Checks object."""


def global_slide_checks(metas: Sequence[dict], ref: dict, pins: ComparatorPins, encoder: str,
                        loader, ck: Checks) -> None:
    rr = ref["record"]
    zero = {m["slide_id"] for m in metas if m["null_stimulus_spatial_shift"]}
    ref_null = rr.get("null_stimuli") if isinstance(rr.get("null_stimuli"), dict) else {}
    ck.ok("zero_shift.slide_set_pinned", zero == set(pins.zero_shift_slides),
          len(pins.zero_shift_slides), len(zero))
    ck.ok("zero_shift.slide_set_reference", zero == set(ref_null.get("slides") or []),
          len(ref_null.get("slides") or []), len(zero))
    cells = {(m["slide_id"], int(t)) for m in metas for t in m["reinhard_nonfinite_listed"]}
    ref_cells = {(c.get("slide_id"), c.get("tile_index"))
                 for c in rr.get("reinhard_excluded_cells") or [] if isinstance(c, dict)}
    ck.ok("reinhard_cells.pinned", cells == set(pins.reinhard_cells),
          len(pins.reinhard_cells), len(cells))                 # C2-13
    ck.ok("reinhard_cells.reference", cells == ref_cells, len(ref_cells), len(cells))
    ck.record("reinhard_excluded_tile_count_B12", count=len(cells), expected=3)
    if loader.checks_weights:
        ws = sorted({str(m.get("weights_sha256")) for m in metas})
        ck.ok("weights_sha256.constant", len(ws) == 1, 1, len(ws))
        pin = dict(pins.weights_sha256).get(encoder)
        if pin is not None:
            ck.ok("weights_sha256.pinned", ws == [pin], pin, ws)
        else:
            ck.record("weights_sha256.value", values=ws)
    gpu: Dict[str, int] = {}
    for m in metas:
        key = f"{m.get('suite_gpu_model')} | driver {m.get('suite_gpu_driver')}"
        gpu[key] = gpu.get(key, 0) + 1
    ck.record("gpu_tuple_per_slide", counts=gpu)
    ck.record("identity_array_equal_diagnostic",
              count=sum(bool(m.get("identity_array_equal_diagnostic")) for m in metas),
              of=len(metas))
    ck.record("reinhard_listed_identity_duplicates",
              slides=sum(bool(m.get("reinhard_listed_identity_duplicates")) for m in metas))
    if "first_run_record_field" in (metas[0] if metas else {}):
        for f in ("first_run_record_field", "plan_sha256_field"):
            vals: Dict[str, int] = {}
            for m in metas:
                vals[str(m.get(f))] = vals.get(str(m.get(f)), 0) + 1
            ck.record(f"suite_record.{f[:-6]}", counts=vals)


# ---------------------------------------------------------------------------------------------
# Evaluation, post-evaluation checks and writing (all checks before the first output file)
# ---------------------------------------------------------------------------------------------

COVERAGE_NOTE = ("if any arm has a non-return no paired result is read until a hit-yield table "
                 "over all intended queries is added by amendment (P2-A8)")


def _evaluate_and_write(*, writer, ck, pop, pop_sha, population, emb, counts, metas, ref, loader,
                        policy, encoder, pipeline_id, dim_used, bank_manifest, bm_sha, pins,
                        paths, code_sha, live_code, draw_count, seed, threads, workers, argv,
                        t_start, t_sampler) -> int:
    n = len(pop)
    t1 = time.time()
    try:
        res = evaluate13(pop, emb, draw_count=draw_count, seed=seed, threads=threads,
                         hist_counts_from=pop.diagnostic_group.tolist())
        t_eval = time.time() - t1
        rpop = ref["record"].get("population") or {}
        ck.ok("evaluate.groups", list(res["groups"]) == rpop.get("groups"),
              len(rpop.get("groups") or []), len(res["groups"]))
        ck.ok("evaluate.conditions", list(res["conds"]) == list(ARMS13), list(ARMS13),
              list(res["conds"]))
        described = res["draws"].describe()
        ck.ok("draws.full_population_table", dict(pins.draw_table) in described,
              "pinned draw-table hashes present", "absent")
        full = np.ascontiguousarray(res["full_draws"].multiplicities)
        ck.ok("draws.full_draws_sha256",
              _sha(full.tobytes()) == dict(pins.draw_table)["multiplicities_sha256"],
              "pinned multiplicities sha256", "differs")
    except Exception as exc:                                   # noqa: BLE001
        ck.fail("evaluate.exception", None, type(exc).__name__, message=str(exc),
                traceback=traceback.format_exc())
    if ck.fails:
        _write_failed(writer, ck, {}, n, argv, live_code)
        return 2

    # C4-01: code_frozen/ holds the comparator CLI and every frozen module it imports, from
    # bytes whose sha equals the one checked at startup
    # Every output is queued as (name, producer) and written only after the 13-arm, name-set
    # and code_frozen checks pass, so a failure leaves FAILED.json alone
    pending: List[Tuple[str, Any]] = []
    frozen = {}
    for k, p in paths.items():
        data = p.read_bytes()
        frozen[f"code_frozen/{p.name}"] = _sha(data)
        pending.append((f"code_frozen/{p.name}", data))

    idx_rows = []
    for c in ARMS13:
        pending.append((f"sampler/{c}.npy",
                        functools.partial(lambda a: E._npy_bytes(a.astype(np.float32)), emb[c])))
        fin = np.isfinite(emb[c].reshape(n, -1)).all(axis=1)
        for i in range(n):
            t, li, nf, u = counts[c][i].tolist()
            idx_rows.append((c, i, int(pop.population_index[i]), pop.slide_id[i],
                             pop.patient_id[i], pop.diagnostic_group[i], t, li, nf, u,
                             int(fin[i])))
    pending.append(("sampler_index.csv",
                    functools.partial(E._csv_bytes, E.SAMPLER_INDEX_HEADER, idx_rows)))
    for c in res["conds"]:
        pending.append((f"distances/{c}.npy", functools.partial(E._npy_bytes, res["dist32"][c])))
    cov = E.coverage_rows(pop, res["conds"], res["returned"], counts)
    for name, header, rows, gz in (
            ("estimates.csv", E.ESTIMATE_HEADER, res["estimates"], False),
            ("paired_deltas.csv", E.DELTA_HEADER, res["deltas"], False),
            ("coverage.csv", E.COVERAGE_HEADER, cov, False),
            ("historical_topk_accuracy.csv", E.HIST_HEADER, res["historical"], False),
            ("query_metrics.csv.gz", E.QUERY_HEADER, res["query_rows"], True),
            ("ranked_lists.csv.gz", E.RANK_HEADER, res["rank_rows"], True)):
        pending.append((name, functools.partial(E._csv_bytes, header, rows, gz)))
    pending.append(("patient_draws.npy", functools.partial(
        lambda m: E._npy_bytes(np.asarray(m, dtype=np.int64)), res["full_draws"].multiplicities)))

    def _s(v):
        return "" if v is None else v if isinstance(v, str) else json.dumps(v, sort_keys=True)

    man_rows = []
    for i, m in enumerate(metas):
        man_rows.append((
            int(pop.population_index[i]), m["slide_id"], m["class"],
            m["suite_pickle"], m["suite_pickle_sha256"], m["suite_record"],
            m["suite_record_sha256"], m["main_bank"], m["main_bank_sha256"], m["n_tiles"],
            json.dumps(m["reinhard_nonfinite_listed"]), m["reinhard_nonfinite_source"],
            int(m["identity_equals_main_bank"]), int(m["null_stimulus_spatial_shift"]),
            _s(m.get("suite_gpu_model")), _s(m.get("suite_gpu_driver")),
            *[m["sources"][c] for c in ARMS13]))
    pending.append(("input_manifest.csv",
                    functools.partial(E._csv_bytes, MANIFEST_HEADER, man_rows)))
    planned = [name for name, _ in pending]

    per_arm = {c: {"first_run": sum(m["sources"][c] == "first_run" for m in metas),
                   "suite": sum(m["sources"][c] == "suite" for m in metas)} for c in ARMS13}
    null_sids = [m["slide_id"] for m in metas if m["null_stimulus_spatial_shift"]]
    excluded = []
    for m in metas:
        for ti in m["reinhard_nonfinite_listed"]:
            excluded.append({"slide_id": m["slide_id"], "tile_index": ti,
                             "source": m["reinhard_nonfinite_source"],
                             "record": m["reinhard_record"],
                             "identity_row_byte_equal_to_listed": m[
                                 "reinhard_listed_identity_duplicates"].get(str(ti), [])})
    gpu_prov: Dict[str, int] = {}
    for m in metas:
        key = f"{m.get('suite_gpu_model')} | driver {m.get('suite_gpu_driver')}"
        gpu_prov[key] = gpu_prov.get(key, 0) + 1
    non_returns = {c: int(n - res["returned"][c].sum()) for c in ARMS13}
    ck.record("nonreturn_cells", count=int(sum(non_returns.values())))   # C3-03(ii)
    rr = ref["record"]
    record = {
        "schema": RUN_SCHEMA,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"),
        "argv": list(argv), "pipeline_id": pipeline_id, "encoder": encoder,
        "feature_dim_used": dim_used,
        "conditions": list(res["conds"]),
        "arm_labels": dict(A.ARM_LABELS),
        "population": {"path": str(population), "sha256": pop_sha, "intended_slides": n,
                       "evaluated_slides": n, "evaluated_patients": int(pop.patient_id.nunique()),
                       "groups": list(res["groups"])},
        "inputs": {**loader.inputs(),
                   "queries_sha256": pins.queries_sha256,
                   "manifest_v1_sha256": pins.manifest_v1_sha256,
                   "extractor_code_sha256": pins.extractor_code_sha256,
                   "suite_code_sha256": suite_code_pin(pins, encoder),
                   "reference_run": {"path": ref["path"],
                                     "complete_sha256": ref["complete_sha256"],
                                     "run_record_sha256": ref["run_record_sha256"]}},
        "main_bank": {"manifest_path": str(bank_manifest), "manifest_sha256": bm_sha},
        "manifest_v2_sha256": pins.manifest_v2_sha256,
        "input_manifest": loader.input_manifest_text,
        "code_sha256": live_code,
        "code_frozen": frozen,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": __import__("scipy").__version__,
                     "sklearn": __import__("sklearn").__version__,
                     "python": sys.version.split()[0]},
        "policy": policy,
        "routing": loader.routing_summary(metas, per_arm, n),
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
        "absent_branch_count": loader.absent_branch_count(metas),
        "identity_equals_main_bank": {
            "byte_equal": int(sum(m["identity_equals_main_bank"] for m in metas)), "of": n,
            "array_equal_diagnostic": int(sum(m["identity_array_equal_diagnostic"]
                                              for m in metas))},
        "suite_gpu_provenance": gpu_prov,
        "coverage_summary": {"expected_returned_per_arm": n, "non_returns": non_returns,
                             "paired_results_readable": not any(non_returns.values()),
                             "note": COVERAGE_NOTE},
        "disclosures": loader.disclosures(metas, per_arm),
        "draw_tables": res["draws"].describe(),
        "diagnostics": res["diagnostics"],
        "timings_seconds": {"verify_and_sampler_stage": round(t_sampler, 1),
                            "evaluate_stage": round(t_eval, 1),
                            "per_condition": res["timings"],
                            "total_before_record": round(time.time() - t_start, 1)},
        "workers": workers, "threads": threads,
        "peak_rss_mb_main_process": round(E.peak_rss_mb(), 1),
        "checks": None,
        "outputs": None,
    }
    if tuple(record) != RUN_RECORD_KEYS:
        raise E.EvalError("run_record key list differs from RUN_RECORD_KEYS")
    ck.touch("assert_arms")
    try:
        assert_arms({"estimates.csv": [r[0] for r in res["estimates"]],
                     "paired_deltas.csv": [r[0] for r in res["deltas"]],
                     "coverage.csv": [r[0] for r in cov],
                     "historical_topk_accuracy.csv": [r[1] for r in res["historical"]],
                     "sampler_index.csv": [r[0] for r in idx_rows],
                     "query_metrics.csv.gz": [r[0] for r in res["query_rows"]],
                     "ranked_lists.csv.gz": [r[0] for r in res["rank_rows"]]},
                    planned, record["conditions"])
    except Exception as exc:                               # noqa: BLE001
        ck.fail("assert_arms", "13 arms in every table", type(exc).__name__)
    # C2-15/C3-08/C4-01: name set (minus code_frozen/) == run_001's 36 entries exactly
    names = {k for k in planned if not k.startswith("code_frozen/")} | {"run_record.json"}
    ref_names = {k for k in ref["files"] if not k.startswith("code_frozen/")}
    ck.ok("complete.name_set", names == ref_names and len(planned) == len(set(planned)),
          len(ref_names), len(names))
    # C4-01: the code_frozen bytes are the bytes whose sha was checked at startup
    ck.ok("complete.code_frozen", all(
        frozen.get(f"code_frozen/{paths[k].name}") == live_code[str(paths[k])] for k in paths)
        and len(frozen) == len(paths), len(paths), len(frozen))
    if ck.fails:
        _write_failed(writer, ck, {}, n, argv, live_code)
        return 2
    for name, produce in pending:
        writer.put(name, produce if isinstance(produce, bytes) else produce())
    record["checks"] = ck.summary()
    record["outputs"] = dict(writer.files)
    writer.put("run_record.json", json.dumps(record, indent=1, default=str).encode())
    writer.put("COMPLETE", json.dumps({"files": writer.files}, indent=1, sort_keys=True).encode())
    return 0


# ---------------------------------------------------------------------------------------------
# CLI (production surface: suite tree only; no first-root/plan/loader/waiver flag)
# ---------------------------------------------------------------------------------------------

def runtime_env_violations() -> List[str]:
    """A6: variables whose value differs from RUNTIME_ENV (stdlib only)."""
    return [f"{var}={os.environ.get(var, '<unset>')!r} (required {want!r})"
            for var, want in RUNTIME_ENV if os.environ.get(var) != want]


def main(argv=None) -> int:
    # A6: refuse before any computation. numpy/pandas are already imported by the prosper
    # package __init__ (not editable), so the guard is the first statement of main; BLAS reads
    # these variables from the inherited process environment, which nothing here modifies.
    bad = runtime_env_violations()
    if bad:
        print("REFUSED (runtime environment): " + "; ".join(bad), file=sys.stderr)
        return ENV_EXIT
    out = E.default_out()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, choices=PRODUCTION_ENCODERS)
    ap.add_argument("--pipeline-id", required=True,
                    choices=[PIPELINE_OF[e] for e in PRODUCTION_ENCODERS])
    ap.add_argument("--dim-used", type=int, required=True)
    ap.add_argument("--suite-root", type=Path, required=True,
                    help="comparator out-root holding features/<encoder>/ and fingerprints/")
    ap.add_argument("--bank-manifest", type=Path, required=True)
    ap.add_argument("--bank-manifest-sha256", required=True)
    ap.add_argument("--population", type=Path, default=(
        out / "r2_1_cpu_20261002" / "perturb_attempt_001" / "queries.csv") if out else None)
    ap.add_argument("--reference-run", type=Path, default=(
        out / "r2_1_p2_eval_20261006" / "run_001") if out else None)
    ap.add_argument("--out", type=Path, required=True, help="new empty run_00N directory")
    ap.add_argument("--workers", type=int, default=1, help="processes for verify+SAMPLER")
    ap.add_argument("--threads", type=int, default=1, help="threads for the L1 stage")
    ap.add_argument("--draw-count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args(argv)
    if a.population is None or a.reference_run is None:
        ap.error("export PROSPER_OUT or pass --population/--reference-run")
    if PIPELINE_OF[a.encoder] != a.pipeline_id:
        ap.error(f"--pipeline-id {a.pipeline_id} does not match --encoder {a.encoder}")
    if DIM_USED[a.encoder] != a.dim_used:
        ap.error(f"--dim-used must be {DIM_USED[a.encoder]} for {a.encoder}")
    argv_full = [sys.argv[0], *argv] if argv is not None else list(sys.argv)
    try:
        status = run(encoder=a.encoder, pipeline_id=a.pipeline_id, dim_used=a.dim_used,
                     suite_root=a.suite_root, bank_manifest=a.bank_manifest,
                     bank_manifest_sha256=a.bank_manifest_sha256, population=a.population,
                     reference_run=a.reference_run, out_dir=a.out, workers=a.workers,
                     threads=a.threads, draw_count=a.draw_count, seed=a.seed, argv=argv_full)
    except SuitePinRefusal as exc:               # C5-10: startup refusal, nothing created
        ap.error(f"REFUSED: {exc}")
    if status == 2:
        doc = json.loads((a.out / FAILED_NAME).read_bytes())
        failed = sorted(k for k, v in doc["checks"].items() if v["status"] == "fail")
        print(f"FAILED: {a.out / FAILED_NAME}; failed check ids: {', '.join(failed)}",
              file=sys.stderr)
    else:
        print(f"[perturb-eval13-comparator] COMPLETE -> {a.out} "
              f"({len(json.loads((a.out / 'COMPLETE').read_bytes())['files'])} files)", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
