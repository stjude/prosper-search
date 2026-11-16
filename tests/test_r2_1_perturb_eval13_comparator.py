"""Tests of the R2.1 P3 comparator stage C2 CLI (``r2_1_perturb_eval13_comparator_cli``;
specification r2_1_comparators_20261008 candidate 004: C2 as amended by C2-06, C2-13, C2-15, C2-17,
C3-02, C3-03, C3-08, C3-09, C3-10, C4-01; declaration_008 C5-10 with G-01, G-02(ii), H-01, H-02).

Synthetic fixtures only, except ``test_real_replay_run_001`` which runs only with
``R21_EVAL13C_REAL=1`` (executed by the analyst as an LSF CPU job, C2-06(b)). The synthetic
reference run is produced by the frozen H-Optimus adapter itself (``A.run`` through the frozen
adapter test fixture ``build``/``run13``).
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import functools
import gzip
import hashlib
import importlib.util
import io
import json
import os
import pickle
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prosper.evaluation import r2_1_hoptimus_main as K
from prosper.evaluation import r2_1_perturb_eval13_cli as A
from prosper.evaluation import r2_1_perturb_eval13_comparator_cli as C
from prosper.evaluation import r2_1_perturb_eval_cli as E
from prosper.evaluation import r2_1_perturb_sampler as S
from tests.test_r2_1_perturb_eval13 import DRAWS, SLIDES, build, run13

FROZEN = {"r2_1_perturb_eval13_cli.py":
          "9591cbbc0cc1b9e556b7b0b9df9dbbd87185d756e91c3f2e3555078f3a9eb269",
          "r2_1_perturb_eval_cli.py":
          "07c0d27f36fc65f7c0e8c9cb8c7df7a611bda1a84931e612885e105028667377",
          "r2_1_perturb_sampler.py":
          "e1880db79f8fcd40dee3e3db91f9664f298ef60250cb1bcd2feb7ea494c8e60c",
          "r2_1_hoptimus_main.py":
          "dee7d68a8b9fcee61d2e845412afdaf20455388ae5c2e61844d4b21d73811faf"}
WEIGHTS = dict(C.ComparatorPins().weights_sha256)          # G-02(ii): uni2 pinned too
SNAPSHOTS = {"uni2": "d517a8dd47902dd7c308b3c36f63bce47e7b9a43",
             "virchow2": "3158645804b69e3f3bc4439d4116edddf0840a72"}
# C5-10: stand-in for the code_frozen_007 CLI sha (the analyst inserts the real one by amendment);
# synthetic suite records carry the per-encoder value and the fixture pins replace the sentinel
UNI2_SUITE_SHA_STANDIN = "5" * 64
SUITE_SHA = {"virchow2": dict(C.ComparatorPins().suite_code_sha256)["virchow2"],
             "uni2": UNI2_SUITE_SHA_STANDIN}
TEST_SUITE_PINS = (("virchow2", SUITE_SHA["virchow2"]), ("uni2", UNI2_SUITE_SHA_STANDIN))
# F-1 (review 001): tests hold both before and after the analyst fills the uni2 default pin.
# Unpinned cases construct their own explicit sentinel pins instead of relying on the default.
UNI2_SUITE_SHA_C5_10 = "39ddddd2d05bab3cf235595c31eb4c2e3289a5d45d4afca236b0c9fecf0d9ac1"
HEX64 = re.compile(r"[0-9a-f]{64}")
SENTINEL_SUITE_PINS = (("virchow2", SUITE_SHA["virchow2"]), ("uni2", C.SUITE_CODE_PIN_SENTINEL))


def _default_uni2_suite_pin():
    return dict(C.ComparatorPins().suite_code_sha256)["uni2"]


FROZEN_9317 = "9317e27595ffc88e30e48642b7b1b7febebf3bc7a2b97f7a055c2e8d408bbd2e"
H01_EXCLUDED_KEYS = {"created", "host", "lsf_job", "argv", "code_sha256", "code_frozen",
                     "timings_seconds", "peak_rss_mb_main_process"}
# literal copies of the frozen adapter's two policy values that C3-09 allows to differ
A_POLICY_ROUTING = {"pinned": "gaussian_blur_image from suite; other 12 arms from first-run",
                    "direct": "reinhard and gaussian_blur_image from suite; other 11 arms "
                              "from first-run",
                    "absent": "all 13 arms from suite"}
A_REINHARD_INDEX_SOURCE = ("the record of the source actually read for the reinhard arm "
                           "(first-run for pinned class, suite for direct/absent)")
A_MANIFEST_TEXT = ("input_manifest.csv (first-run and suite pickle/record sha256 verified "
                   "against their records; main-bank sha256/size verified against the "
                   "population row; per-arm source and class per slide)")
# C3-10: run_record keys not compared in the replay (provenance; first_run_provenance is
# absent from the comparator by C3-09; outputs is compared on the 34 byte-identity files)
REPLAY_EXCLUDED_KEYS = {"schema", "created", "host", "lsf_job", "argv", "pipeline_id", "inputs",
                        "main_bank", "input_manifest", "code_sha256", "code_frozen", "versions",
                        "timings_seconds", "peak_rss_mb_main_process", "checks", "outputs",
                        "first_run_provenance"}
FLOAT_TEXT = re.compile(r"\d\.\d|\d[eE][-+]?\d|\bnan\b|\binf\b", re.I)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------------------------
# Test-only loader reproducing the frozen adapter's routing (C2-06(b)); module level so that
# spawn workers can unpickle it
# ---------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def runtime_env(request, monkeypatch):
    """A6: synthetic tests run with the required environment; the REAL replay must see the
    launcher's own environment, so it is left untouched there."""
    if request.node.name != "test_real_replay_run_001":
        for var, want in C.RUNTIME_ENV:
            monkeypatch.setenv(var, want)


class ReplayLoader:
    """First-run/suite routing of run_001 through the frozen adapter's own ``classify_record``,
    ``arm_sources`` and ``_load_pair`` (and its record checks), behind the comparator's loader
    interface. Never reachable from the production CLI."""
    policy_routing = A_POLICY_ROUTING
    policy_reinhard_index_source = A_REINHARD_INDEX_SOURCE
    checks_weights = False
    input_manifest_text = A_MANIFEST_TEXT

    def __init__(self, first_root: Path, suite_root: Path, plan: Path, pins: A.Pins):
        self.first_dir = str(Path(first_root) / "features" / A.ENCODER)
        self.suite_dir = str(Path(suite_root) / "features" / A.ENCODER)
        self.plan_path = str(plan)
        self.plan_sha = E.sha256_file(Path(plan))
        self.plan_doc = json.loads(Path(plan).read_bytes())
        self.plan_by_sid = {e["slide_id"]: e for e in self.plan_doc["slides"]}
        self.pins = pins

    def load(self, sid, row, n, ck, meta):
        fl: list = []
        try:
            return self._load(sid, row, n, fl, meta)
        finally:
            ck.touch("replay.load")
            for f in fl:
                ck.touch(f["check"])
            ck.fails.extend(fl)

    def _load(self, sid, row, n, fails, meta):
        pins, first_dir, suite_dir = self.pins, Path(self.first_dir), Path(self.suite_dir)
        entry = self.plan_by_sid.get(sid)
        if entry is None:
            A._fail(fails, "plan.slide_entry", "present", "missing")
            return None
        branch = entry.get("first_run_record")
        fr_feats = fr_coords = fr_rec = None
        fr_rec_path, fr_pkl_path = first_dir / f"{sid}.record.json", first_dir / f"{sid}.pkl"
        fr_rec_sha = fr_pkl_sha = None
        if branch == "present":
            rb = fr_rec_path.read_bytes()
            fr_rec_sha, fr_rec = _sha(rb), json.loads(rb)
            meta.update(first_run_record=str(fr_rec_path), first_run_record_sha256=fr_rec_sha,
                        first_run_pickle=str(fr_pkl_path), first_run_host=fr_rec.get("host"),
                        first_run_gpu=fr_rec.get("gpu"), first_run_lsf_job=fr_rec.get("lsf_job"))
            t = "first_run"
            A._expect(fails, f"{t}.schema", A.FIRST_RUN_SCHEMA, fr_rec.get("schema"))
            A._expect(fails, f"{t}.slide_id", sid, fr_rec.get("slide_id"))
            A._expect(fails, f"{t}.code_sha256", pins.extractor_code_sha256,
                      fr_rec.get("code_sha256"))
            A._expect(fails, f"{t}.queries_sha256", pins.queries_sha256,
                      fr_rec.get("queries_sha256"))
            A._expect(fails, f"{t}.n_tiles", n, fr_rec.get("n_tiles"))
            A._expect(fails, f"{t}.conditions", list(A.FIRST_RUN_CONDITIONS),
                      fr_rec.get("conditions"))
            fr_feats, fr_coords, fr_pkl_sha = A._load_pair(fr_pkl_path, fr_rec, n,
                                                           A.FIRST_RUN_CONDITIONS, fails, t)
            meta["first_run_pickle_sha256"] = fr_pkl_sha
            cls, pred, fit = A.classify_record(fr_rec)
            if cls is None:
                A._fail(fails, "class.predicate", "pinned|direct", fit)
            else:
                A._expect(fails, "class.plan_first_run_class", cls, entry.get("first_run_class"))
        else:
            cls = "absent"
        meta["class"] = cls
        s_rec_path, s_pkl_path = suite_dir / f"{sid}.record.json", suite_dir / f"{sid}.pkl"
        rb = s_rec_path.read_bytes()
        s_rec = json.loads(rb)
        gpu = s_rec.get("gpu") if isinstance(s_rec.get("gpu"), dict) else {}
        meta.update(suite_record=str(s_rec_path), suite_record_sha256=_sha(rb),
                    suite_pickle=str(s_pkl_path), suite_gpu_model=gpu.get("gpu_model"),
                    suite_gpu_driver=gpu.get("driver"))
        t = "suite"
        A._expect(fails, f"{t}.schema", A.SUITE_SCHEMA, s_rec.get("schema"))
        A._expect(fails, f"{t}.slide_id", sid, s_rec.get("slide_id"))
        A._expect(fails, f"{t}.plan_sha256", pins.plan_sha256, s_rec.get("plan_sha256"))
        A._expect(fails, f"{t}.queries_sha256", pins.queries_sha256, s_rec.get("queries_sha256"))
        A._expect(fails, f"{t}.suite_code_sha256", pins.suite_code_sha256,
                  s_rec.get("suite_code_sha256"))
        s_fr = s_rec.get("first_run") if isinstance(s_rec.get("first_run"), dict) else {}
        if branch == "present":
            A._expect(fails, f"{t}.first_run.record_sha256", fr_rec_sha, s_fr.get("record_sha256"))
            A._expect(fails, f"{t}.first_run.pickle_sha256", fr_pkl_sha, s_fr.get("pickle_sha256"))
        s_keys = A.SUITE_CONDITIONS.get(cls) if cls is not None else None
        if s_keys is None:
            return None
        A._expect(fails, f"{t}.conditions_encoded", list(s_keys), s_rec.get("conditions_encoded"))
        s_feats, s_coords, s_pkl_sha = A._load_pair(s_pkl_path, s_rec, n, s_keys, fails, t)
        meta["suite_pickle_sha256"] = s_pkl_sha
        if s_feats is None or (branch == "present" and fr_feats is None):
            return None
        sources = A.arm_sources(cls)
        arrays = {c: (fr_feats if sources[c] == "first_run" else s_feats)[c] for c in A.ARMS13}
        coord_sets = ([("first_run", fr_coords)] if branch == "present" else []) + [
            ("suite", s_coords)]
        extra = [("suite", s_feats[A.IDENTITY])] if branch == "present" else []
        ss_rec = fr_rec if sources["spatial_shift"] == "first_run" else s_rec
        r_src = sources["reinhard"]
        return C.SlideInputs(
            arrays=arrays, coord_sets=coord_sets, extra_identity=extra,
            spatial_shift_record=ss_rec, spatial_shift_source=sources["spatial_shift"],
            reinhard_record=fr_rec if r_src == "first_run" else s_rec, reinhard_source=r_src,
            reinhard_record_path=str(s_rec_path if r_src == "suite" else fr_rec_path),
            sources=sources)

    def inputs(self):
        return {"first_run_features": self.first_dir, "suite_features": self.suite_dir,
                "plan": {"path": self.plan_path, "sha256": self.plan_sha}}

    def routing_summary(self, metas, per_arm, n):
        cls_counts = {k: sum(1 for m in metas if m.get("class") == k)
                      for k in dict(self.pins.expected_counts)}
        counts = self.plan_doc.get("counts")
        return {"class_counts": cls_counts,
                "plan_counts": counts if isinstance(counts, dict) else {},
                "plan_expected_direct_count": self.plan_doc.get("expected_direct_count"),
                "reinhard_pinned_fit_slides": n,
                "statement": f"all {n} reinhard arms used are pinned-fit (first-run pinned "
                             f"class {per_arm['reinhard']['first_run']}, suite "
                             f"{per_arm['reinhard']['suite']})"}

    def absent_branch_count(self, metas):
        return sum(1 for m in metas if m.get("class") == "absent")

    def disclosures(self, metas, per_arm):
        absent = self.absent_branch_count(metas)
        return {"V11": A.V11_DISCLOSURE, "decision_A": A.DECISION_A_DISCLOSURE,
                "V1_retention": f"The {per_arm['reinhard']['suite'] - absent} "
                                "first-run direct-fit reinhard arrays (direct class) are retained "
                                "unchanged in the first-run tree and are not used by P2; those "
                                "slides' reinhard arm is read from the suite (pinned fit).",
                "historical_baseline_D": A.HIST_BASELINE_DISCLOSURE, "no_claim": A.NO_CLAIM}


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

def write_bank_manifest(path: Path, pop: pd.DataFrame, bank_files=None, gpu_model="x") -> str:
    """C2-17 columns; ``bank_files`` maps slide -> pickle path (default: the population row)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(C.BANK_MANIFEST_COLUMNS)
        for _, r in pop.iterrows():
            p = Path(bank_files[r.slide_id]) if bank_files else Path(r.resolved_path)
            data = p.read_bytes()
            w.writerow([r.population_index, r.metadata_index, r.slide_id, r.patient_id,
                        r.diagnostic_group, r.feature_rows, str(p), str(p), len(data),
                        _sha(data), gpu_model])
    return E.sha256_file(path)


def pins_for_reference(ref: Path, pop_sha: str, **kw) -> C.ComparatorPins:
    rr = json.loads((ref / "run_record.json").read_bytes())
    cells = tuple(sorted((c["slide_id"], c["tile_index"]) for c in rr["reinhard_excluded_cells"]))
    return dataclasses.replace(
        C.ComparatorPins(), queries_sha256=pop_sha,
        reference_run_record_sha256=E.sha256_file(ref / "run_record.json"),
        reference_complete_sha256=E.sha256_file(ref / "COMPLETE"),
        draw_table=tuple(rr["draw_tables"][0].items()),
        zero_shift_slides=tuple(sorted(rr["null_stimuli"]["slides"])), reinhard_cells=cells,
        **kw)


def rewrite_reference(ref: Path, dst: Path, edits) -> Path:
    """Copy a reference run, apply ``edits(dst)`` (which returns the changed names) and rewrite
    COMPLETE so that its entries stay consistent (the pins are taken from the copy)."""
    shutil.copytree(ref, dst)
    changed = edits(dst) or []
    doc = json.loads((dst / "COMPLETE").read_bytes())
    for name in changed:
        p = dst / name
        if p.exists():
            doc["files"][name] = {"sha256": E.sha256_file(p), "bytes": p.stat().st_size}
    (dst / "COMPLETE").write_bytes(json.dumps(doc, indent=1, sort_keys=True).encode())
    return dst


def read_csv_rows(path: Path):
    data = path.read_bytes()
    if path.name.endswith(".gz"):
        data = gzip.decompress(data)
    return list(csv.reader(io.StringIO(data.decode())))


def compare_replay(ref: Path, out: Path):
    """C3-10: byte identity of 34 files, input_manifest on the shared column projection,
    run_record key-wise minus provenance keys. Returns a list of differences (empty = pass)."""
    diffs = []
    ref_files = json.loads((ref / "COMPLETE").read_bytes())["files"]
    out_files = json.loads((out / "COMPLETE").read_bytes())["files"]
    byte_names = sorted(k for k in ref_files if not k.startswith("code_frozen/")
                        and k not in ("input_manifest.csv", "run_record.json"))
    if len(byte_names) != 34 and len(ref_files) == 40:
        diffs.append(f"byte-identity list has {len(byte_names)} entries")
    for name in byte_names:
        if (ref / name).read_bytes() != (out / name).read_bytes():
            diffs.append(f"bytes differ: {name}")
        if out_files.get(name) != ref_files.get(name):
            diffs.append(f"COMPLETE entry differs: {name}")
    if {k for k in out_files if not k.startswith("code_frozen/")} != {
            k for k in ref_files if not k.startswith("code_frozen/")}:
        diffs.append("COMPLETE non-code_frozen name sets differ")
    r_rows, o_rows = read_csv_rows(ref / "input_manifest.csv"), read_csv_rows(
        out / "input_manifest.csv")
    shared = [c for c in o_rows[0] if c in r_rows[0]]
    if shared != list(C.MANIFEST_HEADER):
        diffs.append("input_manifest shared projection is not the comparator header")
    ri = [r_rows[0].index(c) for c in shared]
    oi = [o_rows[0].index(c) for c in shared]
    if [[r[i] for i in ri] for r in r_rows] != [[r[i] for i in oi] for r in o_rows]:
        diffs.append("input_manifest shared projection differs")
    rr = json.loads((ref / "run_record.json").read_bytes())
    orr = json.loads((out / "run_record.json").read_bytes())
    if set(orr) != (set(rr) - {"first_run_provenance"}) | {"pipeline_id", "main_bank", "checks"}:
        diffs.append("run_record key set")
    for k in sorted(set(rr) - REPLAY_EXCLUDED_KEYS):
        if rr[k] != orr.get(k):
            diffs.append(f"run_record[{k}] differs")
    for name in byte_names:
        if rr["outputs"].get(name) != orr["outputs"].get(name):
            diffs.append(f"run_record.outputs[{name}] differs")
    return diffs


def no_floats(obj) -> bool:
    if isinstance(obj, float):
        return False
    if isinstance(obj, dict):
        return all(no_floats(v) for v in obj.values()) and "max_abs_delta" not in obj
    if isinstance(obj, list):
        return all(no_floats(v) for v in obj)
    return True


def failed_doc(out: Path):
    assert sorted(p.name for p in out.iterdir()) == ["FAILED.json"]
    doc = json.loads((out / "FAILED.json").read_bytes())
    assert doc["schema"] == C.FAILED_SCHEMA
    assert no_floats(doc)
    failed = {k for k, v in doc["checks"].items() if v["status"] == "fail"}
    return failed, doc


# ---------------------------------------------------------------------------------------------
# Production (suite-tree) fixture
# ---------------------------------------------------------------------------------------------

def build_prod(root: Path, ref_fx, ref: Path, encoder: str, *, n_tiles=6, listed=None,
               zero=("s03",), seed=1, rec_hook=None, arrays_hook=None, coords_hook=None,
               bank_hook=None, index_hook=None):
    """Comparator out-root ``root/suite`` (features/<encoder>, fingerprints/INDEX.json), bank
    pickles of width dim_used and the C2-17 bank manifest, for the population of ``ref_fx``."""
    rng = np.random.default_rng(seed)
    width, dim = E.ENCODER_DIM[encoder], C.DIM_USED[encoder]
    listed = {"s02": [1]} if listed is None else listed
    feat_dir = root / "suite" / "features" / encoder
    bank_dir = root / "bank"
    for d in (feat_dir, bank_dir, root / "suite" / "fingerprints"):
        d.mkdir(parents=True, exist_ok=True)
    pop = E.load_population(ref_fx.pop_path)
    pop_row = {r.slide_id: r for r in pop.itertuples()}
    base_pins = A.Pins()
    px = types.SimpleNamespace(root=root, encoder=encoder, width=width, dim=dim, arrs={},
                               bank={}, suite_root=root / "suite", pop_path=ref_fx.pop_path)
    index, bank_files = [], {}
    for sid in pop.slide_id:
        n = n_tiles
        bank = rng.normal(size=(n, dim)).astype(np.float32)
        ident = np.concatenate([bank, rng.normal(size=(n, width - dim)).astype(np.float32)],
                               axis=1) if width > dim else bank.copy()
        arrs = {"identity": ident}
        for c in A.ARMS13[1:]:
            arrs[c] = (ident + rng.normal(scale=0.3, size=(n, width))).astype(np.float32)
        if sid in zero:
            arrs["spatial_shift"] = ident.copy()
        for t in listed.get(sid, []):
            arrs["reinhard"][t] = np.nan
        if arrays_hook:
            arrays_hook(sid, arrs)
        coords = [{"wsi_loc": [224 * k, 17 * k]} for k in range(n)]
        bank_obj = (bank.copy(), copy.deepcopy(coords))
        if bank_hook:
            bank_obj = bank_hook(sid, bank_obj) or bank_obj
        bank_path = bank_dir / f"{sid}.pkl"
        bank_path.write_bytes(pickle.dumps(bank_obj, protocol=4))
        bank_files[sid] = bank_path
        crd = coords_hook(sid, copy.deepcopy(coords)) if coords_hook else coords
        data = pickle.dumps((arrs, crd), protocol=4)
        (feat_dir / f"{sid}.pkl").write_bytes(data)
        px.arrs[sid], px.bank[sid] = arrs, bank
        fp = {"npy_sha256": _sha(b"npy" + sid.encode()),
              "content_sha256": _sha(b"content" + sid.encode()),
              "fp_json_sha256": _sha(b"json" + sid.encode())}
        index.append({"slide_id": sid, **fp, "n_tiles": n,
                      "reinhard_nonfinite_tile_indices": list(listed.get(sid, []))})
        stim = []
        for c in A.ARMS13:
            p = {"level": 1}
            if c == "spatial_shift":
                p = {"x_shift": 0, "y_shift": 0} if sid in zero else {"x_shift": 5, "y_shift": -3}
            stim.append({"condition": c, "level": None, "parameters": p, "slide_id": sid})
        rec = {"schema": A.SUITE_SCHEMA, "slide_id": sid, "encoder": encoder, "dim": width,
               "n_tiles": n, "conditions_encoded": list(A.ARMS13),
               "pickle_sha256": _sha(data), "pickle_bytes": len(data), "plan_sha256": None,
               "first_run_record": "not_applicable",
               "manifest_v2_sha256": base_pins.manifest_v2_sha256,
               "manifest_v1_sha256": base_pins.manifest_v1_sha256,
               "queries_sha256": ref_fx.pop_sha,
               "extractor_code_sha256": base_pins.extractor_code_sha256,
               "suite_code_sha256": SUITE_SHA[encoder],
               "environment": {"suite_code_sha256": SUITE_SHA[encoder], "ok": True,
                               "deviations": [], "recorded": {"hostname": "synthetic"},
                               **({"env_profile": "uni2_historical"} if encoder == "uni2"
                                  else {})},
               "fingerprint_verification": {"mismatches": 0, "status": A.FINGERPRINT_STATUS,
                                            "fp_npy_sha256": fp["npy_sha256"],
                                            "fp_content_sha256": fp["content_sha256"],
                                            "fp_json_sha256": fp["fp_json_sha256"]},
               "reinhard": {"target_means": list(A.PINNED_MEANS),
                            "target_stds": list(A.PINNED_STDS), "info": {}},
               "reinhard_nonfinite_tile_indices": list(listed.get(sid, [])),
               "stimuli": stim,
               "query_row": {k: str(getattr(pop_row[sid], k)) for k in (
                   "population_index", "metadata_index", "slide_id", "feature_rows",
                   "resolved_path", "feature_sha256")},
               "encoder_meta": {"weights": {"sha256": WEIGHTS[encoder],
                                            "snapshot_revision": SNAPSHOTS[encoder]}},
               "gpu": {"gpu_model": "NVIDIA A100-SXM4-80GB", "driver": "595.58.03"}}
        if rec_hook:
            rec_hook(sid, rec)
        (feat_dir / f"{sid}.record.json").write_bytes(json.dumps(rec, indent=1).encode())
    idx_doc = {"schema": "synthetic", "slides": index}
    if index_hook:
        index_hook(idx_doc)
    idx_path = root / "suite" / "fingerprints" / "INDEX.json"
    idx_path.write_bytes(json.dumps(idx_doc, indent=1).encode())
    px.manifest = root / f"bank_manifest_{encoder}.csv"
    px.manifest_sha = write_bank_manifest(px.manifest, pop, bank_files,
                                          "unknown_historical" if encoder == "uni2" else "A100")
    px.pins = pins_for_reference(ref, ref_fx.pop_sha,
                                 fingerprint_index_sha256=E.sha256_file(idx_path),
                                 suite_code_sha256=TEST_SUITE_PINS)
    px.ref = ref
    return px


def run_c(px, out="run_001", *, workers=1, **kw):
    return C.run(encoder=kw.pop("encoder", px.encoder),
                 pipeline_id=kw.pop("pipeline_id", C.PIPELINE_OF[px.encoder]),
                 dim_used=kw.pop("dim_used", px.dim), suite_root=px.suite_root,
                 bank_manifest=px.manifest,
                 bank_manifest_sha256=kw.pop("bank_manifest_sha256", px.manifest_sha),
                 population=px.pop_path, reference_run=kw.pop("reference_run", px.ref),
                 out_dir=px.root / out, workers=workers, threads=1, draw_count=DRAWS, seed=123,
                 argv=["pytest"], pins=kw.pop("pins", px.pins), **kw)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Synthetic run_001 analogue produced by the frozen adapter (all pinned class)."""
    root = tmp_path_factory.mktemp("ref")
    fx = build(root)
    assert run13(fx, "run_001") == 0
    return fx, root / "run_001"


@pytest.fixture
def prod(tmp_path, reference):
    fx, ref = reference
    return functools.partial(build_prod, tmp_path, fx, ref)


# ---------------------------------------------------------------------------------------------
# Frozen imports and constants (C2-06(a), C3-09, C4-01)
# ---------------------------------------------------------------------------------------------

def test_frozen_modules_unchanged_and_imported_unchanged():
    for mod in (A, E, S, K):
        name = Path(mod.__file__).name
        assert E.sha256_file(Path(mod.__file__)) == FROZEN[name], name
    pins = C.ComparatorPins()
    assert (pins.adapter_sha256, pins.e_sha256, pins.s_sha256, pins.kernel_sha256) == (
        FROZEN["r2_1_perturb_eval13_cli.py"], FROZEN["r2_1_perturb_eval_cli.py"],
        FROZEN["r2_1_perturb_sampler.py"], FROZEN["r2_1_hoptimus_main.py"])
    for name in ("evaluate13", "assert_arms", "_byte_diff_report", "_same_types",
                 "_spatial_shift_zero", "_listed"):
        assert getattr(C, name) is getattr(A, name)
    assert C.S is S and C.E is E and C.kernel is K
    assert not hasattr(C, "ENCODER") and not hasattr(C, "DIM")


def test_run_record_keys_and_manifest_header():
    keys = set(A.RUN_RECORD_KEYS)
    assert len(A.RUN_RECORD_KEYS) == 34
    assert set(C.RUN_RECORD_KEYS) == (keys - {"first_run_provenance"}) | {
        "pipeline_id", "main_bank", "checks"}
    assert len(C.RUN_RECORD_KEYS) == len(set(C.RUN_RECORD_KEYS)) == 36
    assert not any(c.startswith("first_run") for c in C.MANIFEST_HEADER)
    assert [c for c in A.MANIFEST_HEADER if not c.startswith("first_run")] == list(
        C.MANIFEST_HEADER)
    assert len(C.planned_output_names()) == 36
    assert "historical_topk_accuracy.csv" in C.planned_output_names()


def test_policy_literal_matches_frozen_adapter(reference):
    _, ref = reference
    rr = json.loads((ref / "run_record.json").read_bytes())
    pol = C.build_policy(DRAWS, 123, A_POLICY_ROUTING, A_REINHARD_INDEX_SOURCE)
    assert pol == rr["policy"]
    assert list(pol) == list(rr["policy"]) and len(pol) == 17
    assert rr["input_manifest"] == A_MANIFEST_TEXT


# ---------------------------------------------------------------------------------------------
# Synthetic replay of the frozen adapter through the test-only loader API (C2-06(b), C3-10)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("workers", [1, 2])
def test_synthetic_replay_byte_identical(tmp_path, workers):
    fx = build(tmp_path, classes={"s04": "direct", "s06": "absent"},
               s_listed={"s04": [2]}, zero=("s03", "s06"))
    assert run13(fx, "ref", workers=workers) == 0
    ref = tmp_path / "ref"
    man = tmp_path / "bank_manifest.csv"
    man_sha = write_bank_manifest(man, E.load_population(fx.pop_path))
    loader = ReplayLoader(tmp_path / "first", tmp_path / "suite", fx.plan_path, fx.pins)
    rc = C.run(encoder="h_optimus_1", pipeline_id="hoptimus_1", dim_used=1536,
               suite_root=tmp_path / "suite", bank_manifest=man, bank_manifest_sha256=man_sha,
               population=fx.pop_path, reference_run=ref, out_dir=tmp_path / "out",
               workers=workers, threads=1, draw_count=DRAWS, seed=123, argv=["pytest"],
               pins=pins_for_reference(ref, fx.pop_sha), loader=loader)
    assert rc == 0
    assert compare_replay(ref, tmp_path / "out") == []
    rr = json.loads((tmp_path / "out" / "run_record.json").read_bytes())
    assert rr["absent_branch_count"] == 1
    assert rr["inputs"]["suite_code_sha256"] is None            # H-02: test-loader-only encoder
    assert {(c["slide_id"], c["tile_index"]) for c in rr["reinhard_excluded_cells"]} == {
        ("s02", 1), ("s04", 2)}
    assert all(v["status"] in ("pass", "recorded") for v in rr["checks"].values())


# ---------------------------------------------------------------------------------------------
# Production suite-tree runs
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("encoder", ["uni2", "virchow2"])
def test_production_complete(prod, encoder):
    px = prod(encoder)
    assert run_c(px) == 0
    out = px.root / "run_001"
    files = json.loads((out / "COMPLETE").read_bytes())["files"]
    ref_files = json.loads((px.ref / "COMPLETE").read_bytes())["files"]
    non_frozen = {k for k in files if not k.startswith("code_frozen/")}
    assert non_frozen == {k for k in ref_files if not k.startswith("code_frozen/")}
    assert len(non_frozen) == 36 and "checks.json" not in files and "FAILED.json" not in files
    for name, e in files.items():
        assert E.sha256_file(out / name) == e["sha256"]
    rr = json.loads((out / "run_record.json").read_bytes())
    rref = json.loads((px.ref / "run_record.json").read_bytes())
    assert tuple(rr) == C.RUN_RECORD_KEYS
    assert set(rr) == (set(rref) - {"first_run_provenance"}) | {"pipeline_id", "main_bank",
                                                                 "checks"}
    assert rr["pipeline_id"] == C.PIPELINE_OF[encoder] and rr["encoder"] == encoder
    assert rr["feature_dim_used"] == C.DIM_USED[encoder]
    assert rr["main_bank"] == {"manifest_path": str(px.manifest),
                               "manifest_sha256": px.manifest_sha}
    # C3-09 policy diff
    assert set(rr["policy"]) == set(rref["policy"])
    assert {k for k in rr["policy"] if rr["policy"][k] != rref["policy"][k]} == {
        "routing", "reinhard_index_source"}
    assert rr["policy"]["routing"] == "all 13 arms from suite"
    assert rr["policy"]["reinhard_index_source"] == "suite"
    # outputs carry every written file except run_record.json itself (as run_001)
    assert set(rr["outputs"]) == set(files) - {"run_record.json"}
    # C4-01 code_frozen: comparator CLI + every frozen module it imports
    frozen_names = {k[len("code_frozen/"):] for k in files if k.startswith("code_frozen/")}
    assert frozen_names == {"r2_1_perturb_eval13_comparator_cli.py", *FROZEN}
    assert {Path(p).name: s for p, s in rr["code_sha256"].items()} == {
        n: files[f"code_frozen/{n}"]["sha256"] for n in frozen_names}
    checks = rr["checks"]
    assert all(v["status"] in ("pass", "recorded") for v in checks.values())
    assert checks["reinhard_excluded_tile_count_B12"]["count"] == 1
    for cid in ("identity_equals_main_bank", "coords.suite_equals_main_bank",
                "zero_shift.spatial_shift_equals_identity", "policy.key_set",
                "complete.name_set", "draws.full_population_precheck",
                "suite.fingerprint_verification.fp_json_sha256", "header[estimates.csv]",
                "reinhard_cells.pinned", "weights_sha256.constant", "code_sha256.adapter"):
        assert checks[cid]["status"] == "pass", cid
    for var, want in C.RUNTIME_ENV:                                     # A6
        assert checks[f"env.{var}"] == {"status": "pass", "n_checked": 1, "n_failed": 0,
                                        "observed": want}
    assert checks["suite.query_row_feature_sha256"] == {"status": "pass", "n_checked": 7,
                                                        "n_failed": 0}            # A7
    # G-02(ii): uni2 weights now pinned through the existing weights_sha256.pinned path
    assert checks["weights_sha256.pinned"] == {"status": "pass", "n_checked": 1, "n_failed": 0}
    assert "weights_sha256.value" not in checks
    snap = checks.get("suite.encoder_meta.weights.snapshot_revision")
    assert snap == ({"status": "pass", "n_checked": 7, "n_failed": 0} if encoder == "uni2"
                    else None)
    # C5-10(b)/(c)
    assert checks["suite.suite_code_sha256"] == {"status": "pass", "n_checked": 7, "n_failed": 0}
    assert checks["suite.environment.env_profile"] == {"status": "pass", "n_checked": 7,
                                                       "n_failed": 0}
    assert rr["inputs"]["suite_code_sha256"] == SUITE_SHA[encoder]
    assert rr["routing"]["class_counts"] == {"suite": 7}
    assert rr["absent_branch_count"] == 7
    assert rr["null_stimuli"]["slides"] == ["s03"]
    assert "C_N4_asymmetry" in rr["disclosures"] and "C6" in rr["disclosures"]
    man = read_csv_rows(out / "input_manifest.csv")
    assert tuple(man[0]) == C.MANIFEST_HEADER
    assert {r[man[0].index("source_reinhard")] for r in man[1:]} == {"suite"}
    sam = np.load(out / "sampler" / "identity.npy")
    assert sam.shape == (7, 10, C.DIM_USED[encoder]) and sam.dtype == np.float32
    for i, sid in enumerate(E.load_population(px.pop_path).slide_id):
        want = S.sampler(px.bank[sid]).embedding.astype(np.float32)
        assert sam[i].tobytes() == want.tobytes()


def test_sampler_never_receives_width_2560(prod, monkeypatch):
    px = prod("virchow2")
    seen = []
    real = S.sampler

    def spy(features, exclude=None):
        seen.append(features.shape)
        return real(features, exclude)

    monkeypatch.setattr(S, "sampler", spy)
    assert run_c(px) == 0
    assert len(seen) == 7 * 13
    assert {s[1] for s in seen} == {1280}
    # the raw suite arrays really are 2560 wide; the CLS half [:, :1280] is what is sampled
    assert {a.shape[1] for a in px.arrs["s01"].values()} == {2560}
    sam = np.load(px.root / "run_001" / "sampler" / "rotate_90.npy")
    want = real(np.ascontiguousarray(px.arrs["s01"]["rotate_90"][:, :1280])).embedding
    assert sam[0].tobytes() == want.astype(np.float32).tobytes()


def test_identity_mismatch_fails_blinded(prod):
    def bank_hook(sid, obj):
        if sid == "s05":
            return (obj[0] + np.float32(0.5), obj[1])
    px = prod("uni2", bank_hook=bank_hook)
    assert run_c(px) == 2
    failed, doc = failed_doc(px.root / "run_001")
    assert failed == {"identity_equals_main_bank"}
    (f,) = doc["slide_failures"]["s05"]
    assert f["differing_rows"] == 6 and "max_abs_delta" not in f
    assert doc["n_slides_failed"] == 1


def test_virchow2_identity_uses_cls_half(prod):
    # bank = full 2560-wide identity is not the comparison: only [:, :1280] is bound to the bank
    def arrays_hook(sid, arrs):
        if sid == "s02":
            arrs["identity"][:, 1280:] = 0.0      # mean half is irrelevant to the identity rule
    px = prod("virchow2", arrays_hook=arrays_hook)
    assert run_c(px) == 0

    def arrays_hook2(sid, arrs):
        if sid == "s02":
            arrs["identity"][0, 1279] += np.float32(1.0)
    px2 = prod("virchow2", arrays_hook=arrays_hook2, seed=2)
    assert run_c(px2, "run_002") == 2
    failed, _ = failed_doc(px2.root / "run_002")
    assert "identity_equals_main_bank" in failed


def test_coords_type_strict(prod):
    def coords_hook(sid, crd):
        if sid == "s01":
            crd[0]["wsi_loc"] = tuple(crd[0]["wsi_loc"])
        return crd
    px = prod("uni2", coords_hook=coords_hook)
    assert run_c(px) == 2
    failed, _ = failed_doc(px.root / "run_001")
    assert failed == {"coords.suite_equals_main_bank"}


def test_missing_condition_refused(prod):
    def arrays_hook(sid, arrs):
        if sid == "s04":
            del arrs["gaussian_blur_image"]

    def rec_hook(sid, rec):
        if sid == "s04":
            rec["conditions_encoded"] = list(A.ARMS13[:-1])
    px = prod("uni2", arrays_hook=arrays_hook, rec_hook=rec_hook)
    assert run_c(px) == 2
    failed, _ = failed_doc(px.root / "run_001")
    assert {"suite.conditions_encoded", "suite.pickle_keys"} <= failed


@pytest.mark.parametrize("field,value,check", [
    ("manifest_v2_sha256", "0" * 64, "suite.manifest_v2_sha256"),
    ("suite_code_sha256", "0" * 64, "suite.suite_code_sha256"),
    ("extractor_code_sha256", "0" * 64, "suite.extractor_code_sha256"),
    ("encoder", "h_optimus_1", "suite.encoder"),
    ("dim", 1280, "suite.dim"),
    ("n_tiles", 5, "suite.n_tiles"),
    ("pickle_bytes", 1, "suite.pickle_bytes"),
])
def test_record_binding_failures(prod, field, value, check):
    def rec_hook(sid, rec):
        if sid == "s06":
            rec[field] = value
    px = prod("uni2", rec_hook=rec_hook)
    assert run_c(px) == 2
    failed, _ = failed_doc(px.root / "run_001")
    assert check in failed


def test_fingerprint_shas_against_index(prod):
    def rec_hook(sid, rec):
        if sid == "s01":
            rec["fingerprint_verification"]["fp_content_sha256"] = "f" * 64
        if sid == "s02":
            rec["fingerprint_verification"]["mismatches"] = 1
    px = prod("uni2", rec_hook=rec_hook)
    assert run_c(px) == 2
    failed, doc = failed_doc(px.root / "run_001")
    assert failed == {"suite.fingerprint_verification.fp_content_sha256",
                      "suite.fingerprint_verification.mismatches"}

    px2 = prod("uni2", seed=3, index_hook=lambda d: d["slides"].pop(0))
    assert run_c(px2, "run_002") == 2
    failed, _ = failed_doc(px2.root / "run_002")
    assert "fingerprints.index_covers_population" in failed


def test_reinhard_fit_pinned(prod):
    def rec_hook(sid, rec):
        if sid == "s03":
            rec["reinhard"]["target_means"] = list(A.DIRECT_MEANS)
    px = prod("uni2", rec_hook=rec_hook)
    assert run_c(px) == 2
    assert failed_doc(px.root / "run_001")[0] == {"suite.reinhard_fit"}


def test_reinhard_cells_must_equal_reference(prod):
    px = prod("uni2", listed={"s02": [1], "s05": [0]})
    assert run_c(px) == 2
    failed, doc = failed_doc(px.root / "run_001")
    assert failed == {"reinhard_cells.pinned", "reinhard_cells.reference"}
    assert doc["slide_failures"] == {}


def test_zero_shift_rules(prod):
    def arrays_hook(sid, arrs):
        if sid == "s03":
            arrs["spatial_shift"] = arrs["spatial_shift"] + np.float32(1.0)
    px = prod("uni2", arrays_hook=arrays_hook)
    assert run_c(px) == 2
    assert failed_doc(px.root / "run_001")[0] == {"zero_shift.spatial_shift_equals_identity"}

    px2 = prod("uni2", zero=("s03", "s07"), seed=4)
    assert run_c(px2, "run_002") == 2
    assert failed_doc(px2.root / "run_002")[0] == {"zero_shift.slide_set_pinned",
                                                   "zero_shift.slide_set_reference"}


def test_weights_pins(prod):
    def rec_hook(sid, rec):
        if sid == "s07":
            rec["encoder_meta"]["weights"]["sha256"] = "b" * 64
    px = prod("virchow2", rec_hook=rec_hook)
    assert run_c(px) == 2
    assert failed_doc(px.root / "run_001")[0] == {"weights_sha256.constant",
                                                  "weights_sha256.pinned"}
    px2 = prod("uni2", rec_hook=rec_hook, seed=5)
    assert run_c(px2, "run_002") == 2                      # G-02(ii): uni2 now pinned as well
    assert failed_doc(px2.root / "run_002")[0] == {"weights_sha256.constant",
                                                   "weights_sha256.pinned"}

    def all_hook(sid, rec):                                 # constant but not the pinned value
        rec["encoder_meta"]["weights"]["sha256"] = "b" * 64
    px3 = prod("uni2", rec_hook=all_hook, seed=6)
    assert run_c(px3, "run_003") == 2
    assert failed_doc(px3.root / "run_003")[0] == {"weights_sha256.pinned"}


def test_draw_table_mismatch(prod):
    px = prod("uni2")
    bad = dict(px.pins.draw_table)
    bad["multiplicities_sha256"] = "0" * 64
    assert run_c(px, pins=dataclasses.replace(px.pins, draw_table=tuple(bad.items()))) == 2
    assert failed_doc(px.root / "run_001")[0] == {"reference.draw_table",
                                                  "draws.full_population_precheck"}
    assert run_c(px, "run_002", pins=px.pins) == 0


def test_reference_header_policy_and_name_set(prod, tmp_path):
    px = prod("uni2")

    def header(dst):
        p = dst / "estimates.csv"
        lines = p.read_bytes().split(b"\n", 1)
        p.write_bytes(lines[0] + b",extra\n" + lines[1])
        return ["estimates.csv"]

    def policy(dst):
        p = dst / "run_record.json"
        rr = json.loads(p.read_bytes())
        rr["policy"]["bootstrap"]["quantile_method"] = "nearest"
        rr["policy"]["extra_key"] = 1
        p.write_bytes(json.dumps(rr, indent=1).encode())
        return ["run_record.json"]

    def names(dst):
        doc = json.loads((dst / "COMPLETE").read_bytes())
        del doc["files"]["historical_topk_accuracy.csv"]
        (dst / "COMPLETE").write_bytes(json.dumps(doc, indent=1, sort_keys=True).encode())
        return []

    expected = {"header": {"header[estimates.csv]"},
                "policy": {"policy.key_set", "policy.bootstrap", "policy.extra_key"},
                # a file missing from COMPLETE cannot be sha-verified, so its header is
                # not read either
                "names": {"complete.planned_name_set",
                          "reference.file_sha256[historical_topk_accuracy.csv]",
                          "header[historical_topk_accuracy.csv]"}}
    for i, (tag, edit) in enumerate((("header", header), ("policy", policy),
                                     ("names", names))):
        ref = rewrite_reference(px.ref, tmp_path / f"ref_{tag}", edit)
        pins = dataclasses.replace(
            px.pins, reference_complete_sha256=E.sha256_file(ref / "COMPLETE"),
            reference_run_record_sha256=E.sha256_file(ref / "run_record.json"))
        out = f"run_{i:03d}"
        assert run_c(px, out, reference_run=ref, pins=pins) == 2
        assert failed_doc(px.root / out)[0] == expected[tag], tag


def test_reference_and_input_shas_pinned(prod):
    px = prod("uni2")
    assert run_c(px, pins=dataclasses.replace(px.pins, reference_complete_sha256="0" * 64)) == 2
    assert failed_doc(px.root / "run_001")[0] == {"reference.complete_sha256"}
    assert run_c(px, "run_002", bank_manifest_sha256="0" * 64) == 2
    assert failed_doc(px.root / "run_002")[0] == {"bank_manifest.sha256"}
    assert run_c(px, "run_003",
                 pins=dataclasses.replace(px.pins, queries_sha256="0" * 64)) == 2
    assert "population.sha256" in failed_doc(px.root / "run_003")[0]
    assert run_c(px, "run_004",
                 pins=dataclasses.replace(px.pins, fingerprint_index_sha256="0" * 64)) == 2
    assert failed_doc(px.root / "run_004")[0] == {"fingerprints.index_sha256"}


def test_code_sha_mismatch_hard_fails_at_startup(prod):
    px = prod("uni2")
    assert run_c(px, pins=dataclasses.replace(px.pins, adapter_sha256="0" * 64)) == 2
    failed, doc = failed_doc(px.root / "run_001")
    assert failed == {"code_sha256.adapter"} and doc["n_slides_checked"] == 0


def test_bank_manifest_rows_bound_to_population(prod):
    px = prod("uni2")
    rows = read_csv_rows(px.manifest)
    rows[1], rows[2] = rows[2], rows[1]
    with open(px.manifest, "w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    sha = E.sha256_file(px.manifest)
    assert run_c(px, bank_manifest_sha256=sha) == 2
    failed, _ = failed_doc(px.root / "run_001")
    assert {"bank_manifest.rows[slide_id]", "bank_manifest.rows[population_index]"} <= failed


def test_nonreturn_is_empty_query_and_all_files_written(prod):
    # C3-03(ii): a zero-usable-tile slide/arm -> empty_query, kept in coverage denominators
    def arrays_hook(sid, arrs):
        if sid == "s05":
            arrs["brightness"][:] = np.nan
    px = prod("uni2", arrays_hook=arrays_hook)
    assert run_c(px) == 0
    out = px.root / "run_001"
    files = json.loads((out / "COMPLETE").read_bytes())["files"]
    assert len([k for k in files if not k.startswith("code_frozen/")]) == 36
    rr = json.loads((out / "run_record.json").read_bytes())
    assert rr["coverage_summary"]["non_returns"]["brightness"] == 1
    assert rr["coverage_summary"]["paired_results_readable"] is False
    assert rr["checks"]["nonreturn_cells"]["count"] == 1
    q = read_csv_rows(out / "query_metrics.csv.gz")
    h = q[0]
    pop = E.load_population(px.pop_path)
    s05 = str(int(pop.loc[pop.slide_id == "s05", "population_index"].iloc[0]))
    statuses = {(r[h.index("condition")], r[h.index("population_index")], r[h.index("status")])
                for r in q[1:]}
    assert ("brightness", s05, "empty_query") in statuses
    assert not any(st == "empty_query" for c, i, st in statuses if (c, i) != ("brightness", s05))
    cov = read_csv_rows(out / "coverage.csv")
    assert len(cov) > 1


def test_post_evaluation_failure_writes_nothing_but_failed(prod, monkeypatch):
    # checks after evaluate13 (13-arm assertion, name set) run before the first output file
    def boom(tables, files, conds):
        raise E.EvalError("synthetic 13-arm failure 0.5")
    px = prod("uni2")
    monkeypatch.setattr(C, "assert_arms", boom)
    assert run_c(px) == 2
    failed, doc = failed_doc(px.root / "run_001")
    assert failed == {"assert_arms"}


def test_planned_names_reach_assert_arms(prod, monkeypatch):
    seen = {}
    real = C.assert_arms

    def spy(tables, files, conds):
        seen["files"] = list(files)
        return real(tables, files, conds)
    px = prod("uni2")
    monkeypatch.setattr(C, "assert_arms", spy)
    assert run_c(px) == 0
    written = json.loads((px.root / "run_001" / "COMPLETE").read_bytes())["files"]
    assert set(seen["files"]) == set(written) - {"run_record.json"}


# ---------------------------------------------------------------------------------------------
# CLI surface: refusals and blinded stdout/stderr
# ---------------------------------------------------------------------------------------------

def test_run_refusals(prod):
    px = prod("uni2")
    with pytest.raises(ValueError):
        run_c(px, pipeline_id="virchow2_sampler")
    with pytest.raises(ValueError):
        run_c(px, dim_used=1280)
    with pytest.raises(ValueError):
        run_c(px, encoder="h_optimus_1", pipeline_id="hoptimus_1", dim_used=1536)
    assert not (px.root / "run_001").exists()


@pytest.mark.parametrize("extra", [
    ["--first-root", "x"], ["--plan", "x"], ["--encoder", "h_optimus_1"],
    ["--dim-used", "2560"], ["--pipeline-id", "uni2_sampler"]])
def test_main_refusals(prod, extra, capsys):
    px = prod("virchow2")
    argv = ["--encoder", "virchow2", "--pipeline-id", "virchow2_sampler", "--dim-used", "1280",
            "--suite-root", str(px.suite_root), "--bank-manifest", str(px.manifest),
            "--bank-manifest-sha256", px.manifest_sha, "--population", str(px.pop_path),
            "--reference-run", str(px.ref), "--out", str(px.root / "run_001")]
    for i in range(0, len(extra), 2):
        if extra[i] in argv:
            argv[argv.index(extra[i]) + 1] = extra[i + 1]
        else:
            argv += extra[i:i + 2]
    with pytest.raises(SystemExit) as ei:
        C.main(argv)
    assert ei.value.code == 2
    assert not (px.root / "run_001").exists()


def _main_argv(px, out):
    return ["--encoder", px.encoder, "--pipeline-id", C.PIPELINE_OF[px.encoder],
            "--dim-used", str(px.dim), "--suite-root", str(px.suite_root),
            "--bank-manifest", str(px.manifest), "--bank-manifest-sha256", px.manifest_sha,
            "--population", str(px.pop_path), "--reference-run", str(px.ref),
            "--out", str(px.root / out), "--draw-count", str(DRAWS)]


def test_main_outputs_are_blinded(prod, monkeypatch, capsys):
    px = prod("uni2")
    monkeypatch.setattr(C, "run", functools.partial(C.run, pins=px.pins))
    assert C.main(_main_argv(px, "run_001")) == 0
    cap = capsys.readouterr()
    assert "COMPLETE" in cap.out and cap.err == ""
    assert not FLOAT_TEXT.search(cap.out.replace(str(px.root), ""))

    def bank_hook(sid, obj):
        if sid == "s01":
            return (obj[0] * np.float32(1.5), obj[1])
    px2 = prod("uni2", bank_hook=bank_hook, seed=7)
    monkeypatch.setattr(C, "run", functools.partial(C.run.func, pins=px2.pins))
    assert C.main(_main_argv(px2, "run_002")) == 2
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "identity_equals_main_bank" in cap.err
    assert not FLOAT_TEXT.search(cap.err.replace(str(px2.root), ""))
    text = (px2.root / "run_002" / "FAILED.json").read_text()
    for key in ("max_abs_delta", "estimates", "hit"):
        assert f'"{key}"' not in text


def test_failed_json_scrubs_exception_text(prod):
    def rec_hook(sid, rec):
        if sid == "s01":
            rec["stimuli"] = "not a list 0.25"
    px = prod("uni2", rec_hook=rec_hook)
    assert run_c(px) == 2
    failed, doc = failed_doc(px.root / "run_001")
    assert "suite.stimuli.spatial_shift_count" in failed
    assert not FLOAT_TEXT.search(json.dumps(doc["slide_failures"]))


# ---------------------------------------------------------------------------------------------
# A6: runtime environment refusal; A7: query_row feature sha cross-check
# ---------------------------------------------------------------------------------------------

ENV_VARS = [v for v, _ in C.RUNTIME_ENV]


def _bad_value(var):
    return "1" if var == "PYTHONHASHSEED" else "8"


def test_runtime_env_constant():
    assert C.RUNTIME_ENV == (("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
                             ("OPENBLAS_NUM_THREADS", "1"), ("NUMEXPR_NUM_THREADS", "1"),
                             ("PYTHONHASHSEED", "0"))
    assert C.ENV_EXIT not in (0, 1, 2) and C.runtime_env_violations() == []


@pytest.mark.parametrize("var", ENV_VARS)
@pytest.mark.parametrize("unset", [False, True])
def test_main_refuses_runtime_env(prod, monkeypatch, capsys, var, unset):
    px = prod("uni2")
    if unset:
        monkeypatch.delenv(var)
    else:
        monkeypatch.setenv(var, _bad_value(var))
    called = []
    monkeypatch.setattr(C, "run", lambda **kw: called.append(kw) or 0)
    assert C.main(_main_argv(px, "run_001")) == C.ENV_EXIT
    cap = capsys.readouterr()
    assert cap.out == "" and var in cap.err and "REFUSED" in cap.err
    assert sum(v in cap.err for v in ENV_VARS) == 1
    assert called == [] and not (px.root / "run_001").exists()


@pytest.mark.parametrize("var", ENV_VARS)
def test_run_records_failing_env_check(prod, monkeypatch, var):
    px = prod("virchow2")
    monkeypatch.setenv(var, _bad_value(var))
    assert run_c(px) == 2
    failed, doc = failed_doc(px.root / "run_001")
    assert failed == {f"env.{var}"}
    assert doc["checks"][f"env.{var}"]["observed"] == _bad_value(var)


def test_run_env_unset_is_recorded(prod, monkeypatch):
    px = prod("uni2")
    monkeypatch.delenv("MKL_NUM_THREADS")
    assert run_c(px) == 2
    failed, doc = failed_doc(px.root / "run_001")
    assert failed == {"env.MKL_NUM_THREADS"}
    assert doc["checks"]["env.MKL_NUM_THREADS"]["observed"] == "<unset>"


@pytest.mark.parametrize("edit", ["bank_sha", "other_row", "missing_field", "no_query_row"])
def test_query_row_feature_sha256_cross_check(prod, edit):
    px = prod("uni2")
    rec_path = px.suite_root / "features" / "uni2" / "s05.record.json"
    rec = json.loads(rec_path.read_bytes())
    pop = E.load_population(px.pop_path).set_index("slide_id")
    assert rec["query_row"]["feature_sha256"] == pop.loc["s05", "feature_sha256"]
    if edit == "bank_sha":          # the literal I1-10 comparand: the bank-manifest row's sha
        bank = {r["slide_id"]: r for r in csv.DictReader(open(px.manifest, newline=""))}
        rec["query_row"]["feature_sha256"] = bank["s05"]["feature_sha256"]
        assert bank["s05"]["feature_sha256"] != pop.loc["s05", "feature_sha256"]
    elif edit == "other_row":
        rec["query_row"]["feature_sha256"] = pop.loc["s04", "feature_sha256"]
    elif edit == "missing_field":
        del rec["query_row"]["feature_sha256"]
    else:
        del rec["query_row"]
    rec_path.write_bytes(json.dumps(rec, indent=1).encode())
    assert run_c(px) == 2
    failed, doc = failed_doc(px.root / "run_001")
    assert failed == {"suite.query_row_feature_sha256"}
    assert set(doc["slide_failures"]) == {"s05"}


def test_query_row_check_fails_closed_without_population_sha(prod, monkeypatch):
    px = prod("uni2")
    orig = C.SuiteLoader.open.__func__
    monkeypatch.setattr(C.SuiteLoader, "open", classmethod(
        lambda cls, *a, **k: orig(cls, *a[:5], {})))
    assert run_c(px) == 2
    failed, _ = failed_doc(px.root / "run_001")
    assert failed == {"suite.query_row_feature_sha256"}


# ---------------------------------------------------------------------------------------------
# C5-10 (declaration_008): per-encoder suite pin (H-02, sentinel), env_profile, UNI2 weights/
# snapshot pins (G-02(ii)) and the H-01 regression against the frozen 9317e275... bytes
# ---------------------------------------------------------------------------------------------

def _frozen_9317_path() -> Path:
    env = os.environ.get("R21_EVAL13C_FROZEN_9317")
    if env:
        return Path(env)
    out = os.environ.get("PROSPER_OUT")
    if not out:
        pytest.skip("export PROSPER_OUT (or R21_EVAL13C_FROZEN_9317) for the frozen 9317e275 copy")
    return (Path(out) / "r2_1_comparators_20261008" / "eval_replay_hopt_003" / "run_001"
            / "code_frozen" / "r2_1_perturb_eval13_comparator_cli.py")


@pytest.fixture
def comparator_9317(tmp_path, monkeypatch):
    """The frozen 9317e275... comparator bytes (eval_replay_hopt_003/run_001/code_frozen), copied
    and imported as a sibling module of prosper.evaluation (its relative imports resolve to the
    same frozen A/E/S/kernel modules, sha-checked by test_frozen_modules_unchanged...)."""
    data = _frozen_9317_path().read_bytes()
    assert _sha(data) == FROZEN_9317
    d = tmp_path / "frozen_9317"
    d.mkdir()
    dst = d / "r2_1_perturb_eval13_comparator_cli.py"
    dst.write_bytes(data)
    name = "prosper.evaluation._r2_1_perturb_eval13_comparator_cli_9317e275"
    spec = importlib.util.spec_from_file_location(name, dst)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    assert mod.ComparatorPins().suite_code_sha256 == SUITE_SHA["virchow2"]
    return mod


def test_c5_10_pins():
    pins = C.ComparatorPins()
    assert A.Pins.suite_code_sha256 == (
        "cc023861719142aeb8993fb8ae4072a607bac2914386043f81c28ad443949879")
    assert pins.suite_code_sha256 == (("virchow2", A.Pins.suite_code_sha256),
                                      ("uni2", _default_uni2_suite_pin()))
    assert C.SUITE_CODE_PIN_SENTINEL == "__TO_BE_PINNED_007__"
    assert not re.fullmatch(r"[0-9a-f]{64}", C.SUITE_CODE_PIN_SENTINEL)
    # F-1: either still unfilled, or filled with the C5-10 amendment value (007 suite CLI sha)
    default = _default_uni2_suite_pin()
    assert default == C.SUITE_CODE_PIN_SENTINEL or (
        HEX64.fullmatch(default) and default == UNI2_SUITE_SHA_C5_10), default
    assert pins.weights_sha256 == (
        ("virchow2", "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"),
        ("uni2", "6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e"))
    assert pins.snapshot_revision == (("uni2", "d517a8dd47902dd7c308b3c36f63bce47e7b9a43"),)


def test_c5_10_other_pins_and_constants_unchanged(comparator_9317):
    old_mod = comparator_9317
    old, new = old_mod.ComparatorPins(), C.ComparatorPins()
    old_f = [f.name for f in dataclasses.fields(old)]
    new_f = [f.name for f in dataclasses.fields(new)]
    assert [f for f in new_f if f != "snapshot_revision"] == old_f
    for f in old_f:
        if f not in ("suite_code_sha256", "weights_sha256"):
            assert getattr(new, f) == getattr(old, f), f
    assert dict(new.suite_code_sha256)["virchow2"] == old.suite_code_sha256
    assert dict(new.weights_sha256)["virchow2"] == dict(old.weights_sha256)["virchow2"]
    for name in ("RUN_RECORD_KEYS", "MANIFEST_HEADER", "PIPELINE_OF", "DIM_USED",
                 "PRODUCTION_ENCODERS", "RUNTIME_ENV", "ENV_EXIT", "POLICY_DIFF_KEYS",
                 "BANK_MANIFEST_COLUMNS", "HEADER_FILES", "NAMED_OUTPUTS", "RUN_SCHEMA",
                 "FAILED_SCHEMA", "C_N4_DISCLOSURE", "C6_DISCLOSURE"):
        assert getattr(C, name) == getattr(old_mod, name), name     # C5-10(d): key set/order


def test_h02_suite_code_pin_lookup():
    pins = dataclasses.replace(C.ComparatorPins(), suite_code_sha256=TEST_SUITE_PINS)
    assert C.suite_code_pin(pins, "virchow2") == SUITE_SHA["virchow2"]
    assert C.suite_code_pin(pins, "uni2") == UNI2_SUITE_SHA_STANDIN
    assert C.suite_code_pin(pins, "h_optimus_1") is None
    for enc in C.PRODUCTION_ENCODERS:
        missing = dataclasses.replace(
            pins, suite_code_sha256=tuple(e for e in TEST_SUITE_PINS if e[0] != enc))
        with pytest.raises(C.SuitePinRefusal, match="no suite_code_sha256 pin"):
            C.suite_code_pin(missing, enc)
    assert C.suite_code_pin(C.ComparatorPins(), "virchow2") == A.Pins.suite_code_sha256
    assert C.suite_code_pin(C.ComparatorPins(), "h_optimus_1") is None
    sentinel = dataclasses.replace(pins, suite_code_sha256=SENTINEL_SUITE_PINS)
    with pytest.raises(C.SuitePinRefusal, match="sentinel"):
        C.suite_code_pin(sentinel, "uni2")
    assert C.suite_code_pin(sentinel, "virchow2") == SUITE_SHA["virchow2"]
    default = _default_uni2_suite_pin()
    if HEX64.fullmatch(default):                          # post-fill state
        assert C.suite_code_pin(C.ComparatorPins(), "uni2") == default == UNI2_SUITE_SHA_C5_10
    else:                                                 # pre-fill state
        assert default == C.SUITE_CODE_PIN_SENTINEL
        with pytest.raises(C.SuitePinRefusal, match="sentinel"):
            C.suite_code_pin(C.ComparatorPins(), "uni2")
    assert issubclass(C.SuitePinRefusal, ValueError)


@pytest.mark.parametrize("encoder", ["uni2", "virchow2"])
@pytest.mark.parametrize("case", ["missing_key", "sentinel"])
def test_pin_refusal_before_any_input_is_read(prod, monkeypatch, encoder, case):
    px = prod(encoder)
    if case == "missing_key":
        pin = tuple(e for e in TEST_SUITE_PINS if e[0] != encoder)
    else:
        pin = tuple((k, C.SUITE_CODE_PIN_SENTINEL if k == encoder else v)
                    for k, v in TEST_SUITE_PINS)
    touched = []
    for name in ("Writer", "load_population", "sha256_file"):
        real = getattr(E, name)
        monkeypatch.setattr(E, name, functools.partial(
            lambda real, name, *a, **k: touched.append(name) or real(*a, **k), real, name))
    with pytest.raises(C.SuitePinRefusal):
        run_c(px, pins=dataclasses.replace(px.pins, suite_code_sha256=pin))
    assert touched == [] and not (px.root / "run_001").exists()


def test_uni2_sentinel_refused_by_main(prod, capsys, monkeypatch):
    px = prod("uni2")
    sentinel = dataclasses.replace(px.pins, suite_code_sha256=SENTINEL_SUITE_PINS)
    monkeypatch.setattr(C, "run", functools.partial(C.run, pins=sentinel))
    with pytest.raises(SystemExit) as ei:
        C.main(_main_argv(px, "run_001"))
    assert ei.value.code == 2
    cap = capsys.readouterr()
    assert cap.out == "" and "REFUSED" in cap.err and C.SUITE_CODE_PIN_SENTINEL in cap.err
    assert not (px.root / "run_001").exists()


def test_uni2_sentinel_per_slide_check_fails_closed(prod):
    px = prod("uni2")
    pop = E.load_population(px.pop_path)
    sentinel = dataclasses.replace(px.pins, suite_code_sha256=SENTINEL_SUITE_PINS)
    ck = C.Checks()
    loader = C.SuiteLoader.open(px.suite_root, "uni2", sentinel, pop.slide_id.tolist(), ck,
                                dict(zip(pop.slide_id, pop["feature_sha256"])))
    assert ck.fails == []
    assert run_c(px, loader=loader) == 2       # run pins: stand-in; loader pins: sentinel
    failed, doc = failed_doc(px.root / "run_001")
    assert failed == {"suite.suite_code_sha256"}
    assert set(doc["slide_failures"]) == set(pop.slide_id)


@pytest.mark.parametrize("carried", ["pin", "other_hex"])
def test_uni2_filled_suite_pin_per_slide_check(prod, carried):
    """F-1 post-fill assertion, exercised in both states: the uni2 pin used is the production
    default once it is 64-hex (after the independent fill), otherwise the C5-10 amendment value."""
    default = _default_uni2_suite_pin()
    assert default == C.SUITE_CODE_PIN_SENTINEL or HEX64.fullmatch(default)
    if HEX64.fullmatch(default):
        suite_pins = C.ComparatorPins().suite_code_sha256
    else:
        suite_pins = (("virchow2", SUITE_SHA["virchow2"]), ("uni2", UNI2_SUITE_SHA_C5_10))
    pin = dict(suite_pins)["uni2"]
    assert pin == UNI2_SUITE_SHA_C5_10
    value = pin if carried == "pin" else "6" * 64

    def rec_hook(sid, rec):
        rec["suite_code_sha256"] = value
        rec["environment"]["suite_code_sha256"] = value
    px = prod("uni2", rec_hook=rec_hook)
    pins = dataclasses.replace(px.pins, suite_code_sha256=suite_pins)
    status = run_c(px, pins=pins)
    if carried == "pin":
        assert status == 0
        rr = json.loads((px.root / "run_001" / "run_record.json").read_bytes())
        assert rr["checks"]["suite.suite_code_sha256"]["status"] == "pass"
        assert rr["checks"]["suite.suite_code_sha256"]["n_failed"] == 0
        assert rr["inputs"]["suite_code_sha256"] == pin
    else:
        assert status == 2
        failed, doc = failed_doc(px.root / "run_001")
        assert failed == {"suite.suite_code_sha256"}


@pytest.mark.parametrize("encoder,edit,want", [
    ("uni2", "delete", {"suite.environment.env_profile"}),
    ("uni2", "pinned", {"suite.environment.env_profile"}),
    ("uni2", None, {"suite.environment.env_profile"}),
    ("uni2", "no_environment", {"exception"}),
    ("virchow2", "pinned", {"suite.environment.env_profile"}),
    ("virchow2", "uni2_historical", {"suite.environment.env_profile"}),
    ("virchow2", None, {"suite.environment.env_profile"}),
    ("virchow2", "no_environment", {"exception"}),
])
def test_env_profile_checks(prod, encoder, edit, want):
    def rec_hook(sid, rec):
        if sid != "s04":
            return
        if edit == "delete":
            del rec["environment"]["env_profile"]
        elif edit == "no_environment":
            del rec["environment"]
        else:
            rec["environment"]["env_profile"] = edit
    px = prod(encoder, rec_hook=rec_hook)
    assert run_c(px) == 2
    failed, doc = failed_doc(px.root / "run_001")
    assert failed == want
    assert set(doc["slide_failures"]) == {"s04"}
    if want == {"suite.environment.env_profile"}:
        assert doc["checks"]["suite.environment.env_profile"] == {
            "status": "fail", "n_checked": 7, "n_failed": 1}


@pytest.mark.parametrize("value", ["0" * 40, None, "delete"])
def test_uni2_snapshot_revision_pinned(prod, value):
    def rec_hook(sid, rec):
        if sid == "s02":
            w = rec["encoder_meta"]["weights"]
            if value == "delete":
                del w["snapshot_revision"]
            else:
                w["snapshot_revision"] = value
    px = prod("uni2", rec_hook=rec_hook)
    assert run_c(px) == 2
    assert failed_doc(px.root / "run_001")[0] == {"suite.encoder_meta.weights.snapshot_revision"}


def test_virchow2_snapshot_revision_not_checked(prod):
    def rec_hook(sid, rec):
        rec["encoder_meta"]["weights"]["snapshot_revision"] = "0" * 40
    px = prod("virchow2", rec_hook=rec_hook)
    assert run_c(px) == 0
    rr = json.loads((px.root / "run_001" / "run_record.json").read_bytes())
    assert "suite.encoder_meta.weights.snapshot_revision" not in rr["checks"]


def test_h01_virchow2_regression_against_frozen_9317(prod, comparator_9317):
    """C5-10(e)(2)/H-01: the same synthetic virchow2 fixture under the frozen 9317e275... bytes
    and under this version: byte-identical outputs; run_records equal except the H-01 keys, the
    code_frozen/ output entries and exactly one added pass check suite.environment.env_profile."""
    old_mod = comparator_9317
    px = prod("virchow2")
    old_pins = old_mod.ComparatorPins(**{
        f.name: getattr(px.pins, f.name) for f in dataclasses.fields(old_mod.ComparatorPins)
        if f.name not in ("suite_code_sha256", "weights_sha256")})     # old defaults for these
    rc_old = old_mod.run(encoder="virchow2", pipeline_id="virchow2_sampler", dim_used=1280,
                         suite_root=px.suite_root, bank_manifest=px.manifest,
                         bank_manifest_sha256=px.manifest_sha, population=px.pop_path,
                         reference_run=px.ref, out_dir=px.root / "old", workers=1, threads=1,
                         draw_count=DRAWS, seed=123, argv=["pytest"], pins=old_pins)
    assert rc_old == 0
    assert run_c(px, "new") == 0
    o, n = px.root / "old", px.root / "new"
    of = json.loads((o / "COMPLETE").read_bytes())["files"]
    nf = json.loads((n / "COMPLETE").read_bytes())["files"]
    names = sorted(k for k in of if not k.startswith("code_frozen/") and k != "run_record.json")
    assert names == sorted(k for k in nf if not k.startswith("code_frozen/")
                           and k != "run_record.json")
    assert len(names) == 35 and any(k.endswith(".csv.gz") for k in names)
    for k in names:
        assert (o / k).read_bytes() == (n / k).read_bytes(), k
        assert of[k] == nf[k], k
    cf = "code_frozen/r2_1_perturb_eval13_comparator_cli.py"
    assert of[cf]["sha256"] == FROZEN_9317
    assert nf[cf]["sha256"] == E.sha256_file(Path(C.__file__)) != FROZEN_9317
    assert {k: v for k, v in of.items() if k.startswith("code_frozen/") and k != cf} == {
        k: v for k, v in nf.items() if k.startswith("code_frozen/") and k != cf}
    ro = json.loads((o / "run_record.json").read_bytes())
    rn = json.loads((n / "run_record.json").read_bytes())
    assert list(ro) == list(rn) == list(C.RUN_RECORD_KEYS)               # C5-10(d)
    differing = {k for k in ro if ro[k] != rn[k]}
    assert differing - H01_EXCLUDED_KEYS <= {"outputs", "checks"}
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("code_frozen/")}  # noqa
    assert strip(ro["outputs"]) == strip(rn["outputs"])
    added = set(rn["checks"]) - set(ro["checks"])
    assert added == {"suite.environment.env_profile"} and set(ro["checks"]) <= set(rn["checks"])
    assert rn["checks"]["suite.environment.env_profile"] == {"status": "pass", "n_checked": 7,
                                                             "n_failed": 0}
    for cid, v in ro["checks"].items():
        assert rn["checks"][cid] == v, cid


# ---------------------------------------------------------------------------------------------
# REAL replay of run_001 (C2-06(b)/C3-10); executed by the analyst as an LSF CPU job
# ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("R21_EVAL13C_REAL") != "1",
                    reason="REAL replay runs only with R21_EVAL13C_REAL=1 (lead LSF CPU job)")
def test_real_replay_run_001():
    bad = C.runtime_env_violations()          # A6: refuse before reading or computing anything
    if bad:
        pytest.fail("REFUSED (runtime environment): " + "; ".join(bad))
    out_root = Path(os.environ["PROSPER_OUT"])
    ref = Path(os.environ.get("R21_EVAL13C_REFERENCE",
                              out_root / "r2_1_p2_eval_20261006" / "run_001"))
    replay_root = Path(os.environ.get("R21_EVAL13C_OUT",
                                      out_root / "r2_1_comparators_20261008" / "eval_replay_hopt"))
    rr = json.loads((ref / "run_record.json").read_bytes())
    pins = C.ComparatorPins()
    assert E.sha256_file(ref / "run_record.json") == pins.reference_run_record_sha256
    population = Path(rr["population"]["path"])
    first_root = Path(rr["inputs"]["first_run_features"]).parents[1]
    suite_root = Path(rr["inputs"]["suite_features"]).parents[1]
    plan = Path(rr["inputs"]["plan"]["path"])
    assert E.sha256_file(plan) == rr["inputs"]["plan"]["sha256"] == A.Pins().plan_sha256
    replay_root.mkdir(parents=True, exist_ok=True)
    manifest = replay_root / "bank_manifest_hoptimus_1.csv"
    if manifest.exists():
        man_sha = E.sha256_file(manifest)
    else:
        man_sha = write_bank_manifest(manifest, E.load_population(population),
                                      gpu_model="historical_main_bank")
    out = replay_root / os.environ.get("R21_EVAL13C_RUN", "run_001")
    loader = ReplayLoader(first_root, suite_root, plan, A.Pins())
    rc = C.run(encoder="h_optimus_1", pipeline_id="hoptimus_1", dim_used=1536,
               suite_root=suite_root, bank_manifest=manifest, bank_manifest_sha256=man_sha,
               population=population, reference_run=ref, out_dir=out,
               workers=rr["workers"], threads=rr["threads"],
               draw_count=rr["policy"]["bootstrap"]["count"],
               seed=rr["policy"]["bootstrap"]["seed"],
               argv=["pytest", "test_real_replay_run_001"], pins=pins, loader=loader)
    if rc != 0:
        doc = json.loads((out / "FAILED.json").read_bytes())
        pytest.fail("comparator FAILED: " + ", ".join(
            sorted(k for k, v in doc["checks"].items() if v["status"] == "fail")))
    diffs = compare_replay(ref, out)
    (replay_root / f"{out.name}_replay_report.json").write_text(json.dumps(
        {"reference": str(ref), "out": str(out), "differences": diffs,
         "byte_identity_files": 34, "excluded_run_record_keys": sorted(REPLAY_EXCLUDED_KEYS)},
        indent=1))
    assert diffs == []
