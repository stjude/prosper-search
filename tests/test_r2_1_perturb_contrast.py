"""Tests for the N3 cross-encoder perturbation contrast CLI (SPEC n3_spec_001 + amendments 001-004).

Synthetic fixtures only (no real comparator directory exists or is created).  The expected
values are produced by an independent plain-NumPy paired whole-patient bootstrap written in
this file (``ref_*``); it never calls the kernel or any N3 function.

Run (no /tmp):
    cd CODE && PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_r2_1_perturb_contrast.py -q \
        -p no:cacheprovider --basetemp=$PROSPER_OUT/r2_1_p2_eval_20261006/n3_impl_003/pytest_tmp
The run_001 replay/self-contrast/synthetic-shift test (§6.1/6.2/6.8) needs ``R21_N3_REAL=1``.
"""
from __future__ import annotations

import ast
import contextlib
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from prosper.evaluation import r2_1_perturb_contrast_cli as cli

CODE_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = CODE_ROOT / "prosper" / "evaluation" / "r2_1_perturb_contrast_cli.py"

CONDITIONS = ("identity", "rotate_90", "rotate_180", "rotate_270", "flip_horizontal",
              "flip_vertical", "brightness", "saturation", "contrast", "spatial_shift",
              "reinhard", "gaussian_blur", "gaussian_blur_image")
UNCHANGED_ARM = "rotate_180"
STRAT_ARMS = ("brightness", "saturation", "contrast", "spatial_shift")
KS = (1, 3, 5, 10)
QHEADER = ("condition", "population_index", "query", "patient_id", "diagnostic_group", "k",
           "status", "returned_depth", "conditional_hit", "conditional_precision",
           "n_candidates", "boundary_tie")
DELTA_HEADER = ("condition", "group", "metric", "k", "delta", "ci_low", "ci_high", "n_pairs",
                "n_patients", "estimate_condition", "estimate_identity", "population",
                "point_status", "point_reason", "ci_status", "ci_reason",
                "zero_support_replicates")
COUNT, SEED = 5000, 123
MANIFEST = "ab" * 32
TOL = 1e-12


# =============================================================================================
# Independent plain-NumPy reference (never calls the kernel or N3)
# =============================================================================================

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(Path(p).read_bytes())


def ref_draws(patients):
    """Whole-patient multiplicities: sorted unique patients, PCG64(seed), bincount per row."""
    order = tuple(sorted(set(patients)))
    p = len(order)
    gen = np.random.Generator(np.random.PCG64(SEED))
    w = np.empty((COUNT, p), dtype=np.int64)
    for b in range(COUNT):
        w[b] = np.bincount(gen.integers(p, size=p), minlength=p)
    return order, w


def ref_table_entry(order, w):
    return {"n_patients": len(order),
            "patients_sha256": sha_bytes("\n".join(order).encode()),
            "multiplicities_sha256": sha_bytes(np.ascontiguousarray(w, dtype=np.int64).tobytes())}


def ref_estimate(point, reason, nq, npat, reps, valid):
    """§4.5 / N3-07 estimate rule, written from the spec text."""
    invalid = int((~valid).sum())
    if point is None:
        ci_status, ci_reason, lo, hi = "unavailable", reason, None, None
    elif valid.all():
        lo, hi = (float(x) for x in np.quantile(reps, [0.025, 0.975], method="linear"))
        ci_status, ci_reason = "estimable", None
    else:
        ci_status, ci_reason, lo, hi = "unavailable", "zero_support_replicates", None, None
    return dict(point=point, point_status="estimable" if point is not None else "unavailable",
                point_reason=reason, ci_low=lo, ci_high=hi, ci_status=ci_status,
                ci_reason=ci_reason, nq=nq, npat=npat, zsr=invalid, reps=reps, valid=valid)


def ref_summary(values, mask, pids, labels, groups, draws):
    """Slide-weighted group means with whole-patient bootstrap replicates (einsum route)."""
    order, w = draws
    pindex = {p: i for i, p in enumerate(order)}
    gindex = {g: i for i, g in enumerate(groups)}
    idx = np.flatnonzero(mask)
    pi = np.array([pindex[pids[i]] for i in idx], dtype=np.int64)
    gi = np.array([gindex[labels[i]] for i in idx], dtype=np.int64)
    counts = np.zeros((len(order), len(groups)), dtype=np.int64)
    sums = np.zeros((len(order), len(groups)), dtype=np.float64)
    np.add.at(counts, (pi, gi), 1)
    np.add.at(sums, (pi, gi), np.asarray(values, dtype=np.float64)[idx])
    support = np.einsum("bp,pg->bg", w, counts)
    numer = np.einsum("bp,pg->bg", w.astype(np.float64), sums)
    valid = support > 0
    reps = np.full(support.shape, np.nan)
    reps[valid] = numer[valid] / support[valid]
    n_g = counts.sum(axis=0)
    s_g = sums.sum(axis=0)
    ests = []
    for g in range(len(groups)):
        n = int(n_g[g])
        ests.append(ref_estimate(float(s_g[g] / n) if n else None,
                                 None if n else "no_included_queries", n,
                                 int((counts[:, g] > 0).sum()), reps[:, g], valid[:, g]))
    mvalid = valid.all(axis=1)
    mreps = np.full(COUNT, np.nan)
    mreps[mvalid] = reps[mvalid].mean(axis=1)
    if (n_g > 0).all():
        mpoint, mreason = float(np.mean([e["point"] for e in ests])), None
    else:
        mpoint = None
        mreason = "no_included_queries" if len(idx) == 0 else "missing_group_support"
    macro = ref_estimate(mpoint, mreason, int(len(idx)), len(order), mreps, mvalid)
    return ests + [macro], support


def ref_paired(va, vb, mask, pids, labels, groups, draws):
    a, sa = ref_summary(va, mask, pids, labels, groups, draws)
    b, sb = ref_summary(vb, mask, pids, labels, groups, draws)
    assert np.array_equal(sa, sb)
    diff = []
    for x, y in zip(a, b):
        valid = x["valid"] & y["valid"]
        reps = np.full(COUNT, np.nan)
        reps[valid] = x["reps"][valid] - y["reps"][valid]
        point = (x["point"] - y["point"]
                 if x["point"] is not None and y["point"] is not None else None)
        diff.append(ref_estimate(point, x["point_reason"] or y["point_reason"], x["nq"],
                                 x["npat"], reps, valid))
    return dict(a=a, b=b, d=diff)


def ref_compose(eh, ec, identical_nonzero):
    """D = Delta_H - Delta_c (N3-07, N3-09) written from the spec text."""
    valid = eh["valid"] & ec["valid"]
    assert np.array_equal(valid, eh["valid"]) and np.array_equal(valid, ec["valid"])
    reps = np.full(COUNT, np.nan)
    reps[valid] = 0.0 if identical_nonzero else eh["reps"][valid] - ec["reps"][valid]
    if eh["point"] is None or ec["point"] is None:
        point = None
    else:
        point = 0.0 if identical_nonzero else eh["point"] - ec["point"]
    e = ref_estimate(point, eh["point_reason"] or ec["point_reason"], eh["nq"], eh["npat"],
                     reps, valid)
    e["excludes"] = (None if e["ci_status"] != "estimable"
                     else bool(e["ci_low"] > 0 or e["ci_high"] < 0))
    return e


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        return repr(v)
    return str(v)


def write_csv(path: Path, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([fmt(x) for x in r])
    Path(path).write_text(buf.getvalue(), encoding="utf-8")


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_gz_csv(path: Path, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue().encode("utf-8"))


def write_json(path: Path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, indent=1) + "\n", encoding="utf-8")


def write_complete(d: Path):
    files = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != "COMPLETE":
            rel = p.relative_to(d).as_posix()
            files[rel] = {"bytes": p.stat().st_size, "sha256": sha_file(p)}
    write_json(d / "COMPLETE", {"files": files})
    return files


# =============================================================================================
# World: population, per-pipeline value vectors, populations and the reference cache
# =============================================================================================

class World:
    """Holds one fixture universe (synthetic, or run_001 in the REAL test)."""

    def __init__(self, qids, pids, labels, groups, conditions):
        self.qids, self.pids, self.labels = list(qids), list(pids), list(labels)
        self.groups = tuple(groups)
        self.conditions = tuple(conditions)
        self.arms = tuple(c for c in conditions if c != "identity")
        self.n = len(qids)
        self.vals = {}          # key -> {(cond, metric, k): float64 vector}
        self.masks = {}         # population name -> bool mask
        self.strata = {}        # arm -> list of stratum labels
        self.strength = {}      # arm -> strengths
        self.null = {}          # arm -> sorted list of slide ids
        self._draws = {}
        self._cache = {}
        self.labels_arr = np.array(self.labels, dtype=object)

    def draws(self, pop):
        key = tuple(sorted({p for p, m in zip(self.pids, self.masks[pop]) if m}))
        if key not in self._draws:
            self._draws[key] = ref_draws(key)
        return self._draws[key]

    def add_populations(self):
        self.masks["full"] = np.ones(self.n, dtype=bool)
        for arm in STRAT_ARMS:
            lab = np.array(self.strata[arm], dtype=object)
            for t in ("T1", "T2", "T3"):
                self.masks[f"stratum:{arm}:{t}"] = lab == t
        for arm, pname in (("brightness", "brightness_null_excluded"),
                           ("spatial_shift", "shift_null_excluded")):
            s = set(self.null[arm])
            self.masks[pname] = np.array([q not in s for q in self.qids], dtype=bool)

    def pops_for(self, cond):
        out = ["full"]
        if cond in STRAT_ARMS:
            out += [f"stratum:{cond}:{t}" for t in ("T1", "T2", "T3")]
        if cond == "brightness":
            out.append("brightness_null_excluded")
        if cond == "spatial_shift":
            out.append("shift_null_excluded")
        return out

    def paired(self, vkey, pop, cond, metric, k):
        key = (vkey, pop, cond, metric, k)
        if key not in self._cache:
            v = self.vals[vkey]
            self._cache[key] = ref_paired(v[(cond, metric, k)], v[("identity", metric, k)],
                                          self.masks[pop], self.pids, self.labels,
                                          self.groups, self.draws(pop))
        return self._cache[key]

    def table_entry(self, pop):
        return ref_table_entry(*self.draws(pop))

    def delta_rows(self, vkey, pop, conds, extra=lambda cond: ()):
        rows = []
        for cond in conds:
            for metric in ("hit", "precision"):
                for k in KS:
                    pd_ = self.paired(vkey, pop(cond) if callable(pop) else pop, cond, metric, k)
                    for g, d, a, b in zip((*self.groups, "MACRO"), pd_["d"], pd_["a"], pd_["b"]):
                        rows.append([cond, g, metric, k, d["point"], d["ci_low"], d["ci_high"],
                                     d["nq"], d["npat"], a["point"], b["point"], "both_returned",
                                     d["point_status"], d["point_reason"], d["ci_status"],
                                     d["ci_reason"], d["zsr"], *extra(cond)])
        return rows


def values_from_relevance(rel):
    """rel: (n, 10) bool -> {(metric, k): vector}; precision@1 == hit@1 by construction."""
    out = {}
    for k in KS:
        top = rel[:, :k]
        out[("hit", k)] = top.any(axis=1).astype(np.float64)
        out[("precision", k)] = top.sum(axis=1).astype(np.float64) / np.float64(k)
    return out


def qm_rows(world, vkey, statuses=None):
    rows = []
    v = world.vals[vkey]
    for cond in world.conditions:
        for i in range(world.n):
            for k in KS:
                rows.append([cond, str(i), world.qids[i], world.pids[i], world.labels[i], str(k),
                             "returned", "10", repr(float(v[(cond, "hit", k)][i])),
                             repr(float(v[(cond, "precision", k)][i])), "200", "false"])
    return rows


def coverage_rows(world, nonreturns=0):
    rows = []
    for cond in world.conditions:
        for g in world.groups:
            rows.append([cond, g, 1, 1, 0, "", 1.0, 10, 0, 0])
    rows[0][4] = nonreturns
    return rows


COVERAGE_HEADER = ("condition", "group", "intended", "returned", "non_returns",
                   "non_return_slides", "coverage", "tiles_total", "tiles_excluded_listed",
                   "tiles_excluded_nonfinite")


def build_synthetic_world(root: Path, tiny_group: bool) -> World:
    """>= 4 groups of unequal size; optionally one single-patient group (zero support)."""
    rng = np.random.default_rng(20261007 + int(tiny_group))
    sizes = {"G0": 26, "G1": 31, "G2": 22, "G3": 28}
    q = {"G0": 0.2, "G1": 0.35, "G2": 0.5, "G3": 0.65, "G4": 0.4}
    entries = []
    for g, npat in sizes.items():
        for j in range(npat):
            for _ in range(int(rng.integers(1, 3))):
                entries.append((f"P{g}_{j:03d}", g))
    if tiny_group:
        entries += [("PG4_000", "G4")] * 3
    order = rng.permutation(len(entries))
    entries = [entries[i] for i in order]
    qids = [str(5000 + i) for i in range(len(entries))]
    pids = [e[0] for e in entries]
    labels = [e[1] for e in entries]
    groups = sorted(set(labels))
    w = World(qids, pids, labels, groups, CONDITIONS)
    qv = np.array([q[g] for g in labels])
    for pipe in ("H", "C"):
        base = rng.random((w.n, 10)) < qv[:, None]
        vals = {}
        for cond in CONDITIONS:
            rel = base.copy()
            if cond not in ("identity", UNCHANGED_ARM):
                u = rng.random(rel.shape)
                if pipe == "H" and cond == "gaussian_blur":
                    rel &= ~(u < 0.5)            # H degrades strongly -> D < 0
                elif pipe == "C" and cond == "gaussian_blur":
                    pass                          # C unchanged on this arm
                elif pipe == "H":
                    rel |= (u < 0.2)              # H improves
                else:
                    rel &= ~(u < 0.2)             # C degrades (opposite sign)
            for (metric, k), vec in values_from_relevance(rel).items():
                vals[(cond, metric, k)] = vec
        w.vals[pipe] = vals
    for arm in STRAT_ARMS:
        s = rng.random(w.n) * (20.0 if arm == "spatial_shift" else 0.15)
        c1, c2 = (float(x) for x in np.quantile(s, [1 / 3, 2 / 3]))
        w.strength[arm] = [float(x) for x in s]
        w.strata[arm] = ["T1" if x <= c1 else "T2" if x <= c2 else "T3" for x in s]
        w.strength[arm + ":cuts"] = [c1, c2]
    pick = rng.permutation(w.n)
    w.null["brightness"] = sorted(qids[i] for i in pick[:5])
    w.null["spatial_shift"] = sorted(qids[i] for i in pick[5:9])
    w.add_populations()
    write_reference_dirs(w, root)
    return w


def write_reference_dirs(w: World, root: Path) -> None:
    """Synthetic stand-ins for queries.csv, run_001, dose_summary_001, sensitivity_brightness_001."""
    root.mkdir(parents=True, exist_ok=True)
    w.root = root
    w.queries = root / "queries.csv"
    write_csv(w.queries, ("population_index", "slide_id", "patient_id", "diagnostic_group"),
              [[i, w.qids[i], w.pids[i], w.labels[i]] for i in range(w.n)])
    w.queries_sha = sha_file(w.queries)
    w.n_patients = len(set(w.pids))
    hdir = root / "run_001"
    hdir.mkdir()
    write_gz_csv(hdir / "query_metrics.csv.gz", QHEADER, qm_rows(w, "H"))
    order, mult = w.draws("full")
    np.save(hdir / "patient_draws.npy", mult, allow_pickle=False)
    write_csv(hdir / "paired_deltas.csv", DELTA_HEADER, w.delta_rows("H", "full", w.arms))
    w.run_record = {
        "conditions": list(w.conditions),
        "population": {"path": str(w.queries), "sha256": w.queries_sha,
                       "intended_slides": w.n, "evaluated_slides": w.n,
                       "evaluated_patients": w.n_patients, "groups": list(w.groups)},
        "draw_tables": [w.table_entry("full")],
        "manifest_v2_sha256": MANIFEST,
        "null_stimuli": {"arm": "spatial_shift", "slides": [int(s) for s in w.null["spatial_shift"]]},
        "policy": {"synthetic": True},
        # amendment 004 item 6: the run_record binds its own outputs (as run_001's does).
        "outputs": {f: {"sha256": sha_file(hdir / f), "bytes": (hdir / f).stat().st_size}
                    for f in ("query_metrics.csv.gz", "paired_deltas.csv",
                              "patient_draws.npy")},
    }
    write_json(hdir / "run_record.json", w.run_record)
    write_csv(hdir / "coverage.csv", COVERAGE_HEADER, coverage_rows(w))
    (hdir / "estimates.csv").write_text("condition,group\nidentity,G0\n")
    write_complete(hdir)
    w.hopt_dir = hdir
    ddir = root / "dose_summary_001"
    ddir.mkdir()
    rs = []
    for arm in STRAT_ARMS:
        for i in range(w.n):
            rs.append([w.qids[i], arm, 1.0 + w.strength[arm][i], w.strength[arm][i],
                       w.strata[arm][i]])
    write_csv(ddir / "realised_stimuli.csv", ("slide_id", "arm", "raw_parameter", "strength",
                                               "stratum"), rs)
    tables, seen = [], set()
    for pop in ["full"] + [f"stratum:{a}:{t}" for a in STRAT_ARMS for t in ("T1", "T2", "T3")]:
        e = w.table_entry(pop)
        if e["patients_sha256"] not in seen:
            seen.add(e["patients_sha256"])
            tables.append(e)
    write_json(ddir / "RECORD.json", {"tertile_cuts": {a: w.strength[a + ":cuts"]
                                                        for a in STRAT_ARMS},
                                       "draw_tables": tables})

    def stratum_extra(t):
        def f(cond):
            m = w.masks[f"stratum:{cond}:{t}"]
            s = np.array(w.strength[cond])[m]
            return (t, int(m.sum()), float(s.min()), float(s.max()))
        return f
    st_rows = []
    for t in ("T1", "T2", "T3"):
        st_rows += w.delta_rows("H", lambda c, t=t: f"stratum:{c}:{t}", STRAT_ARMS,
                                stratum_extra(t))
    write_csv(ddir / "strata_deltas.csv", DELTA_HEADER + ("stratum", "n_stratum_slides",
                                                          "strength_min", "strength_max"), st_rows)
    w.dose_dir = ddir
    sdir = root / "sensitivity_brightness_001"
    sdir.mkdir()
    ns = [[s, "brightness", "1.0001", "1<f<1+1/255"] for s in w.null["brightness"]]
    ns += [[s, "spatial_shift", "0", "zero_shift"] for s in w.null["spatial_shift"]]
    write_csv(sdir / "null_slides.csv", ("slide_id", "arm", "factor", "reason"), ns)
    write_json(sdir / "RECORD.json", {"draw_tables": [
        w.table_entry(p) for p in ("full", "brightness_null_excluded", "shift_null_excluded")]})
    se_rows = (w.delta_rows("H", "brightness_null_excluded", ("brightness",),
                            lambda c: (len(w.null["brightness"]), "exclude_null_stimuli"))
               + w.delta_rows("H", "shift_null_excluded", ("spatial_shift",),
                              lambda c: (len(w.null["spatial_shift"]), "exclude_null_stimuli")))
    write_csv(sdir / "sensitivity_deltas.csv", DELTA_HEADER + ("excluded_null_slides", "analysis"),
              se_rows)
    w.sens_dir = sdir


def make_comparator(w: World, base: Path, name: str, vkey: str = "C", *, pipeline_id="same",
                    qm_hook=None, rr_hook=None, pd_hook=None, nonreturns=0, acc_hook=None,
                    complete_hook=None):
    d = base / f"comp_{name}"
    d.mkdir(parents=True)
    rows = qm_rows(w, vkey)
    if qm_hook:
        rows = qm_hook(rows)
    header = QHEADER
    if isinstance(rows, tuple):
        header, rows = rows
    write_gz_csv(d / "query_metrics.csv.gz", header, rows)
    rec = json.loads(json.dumps(w.run_record))
    if pipeline_id == "same":
        rec["pipeline_id"] = name
    elif pipeline_id is not None:
        rec["pipeline_id"] = pipeline_id
    if rr_hook:
        rr_hook(rec)
    write_json(d / "run_record.json", rec)
    pdr = w.delta_rows(vkey, "full", w.arms)
    if pd_hook:
        pd_hook(pdr)
    write_csv(d / "paired_deltas.csv", DELTA_HEADER, pdr)
    write_csv(d / "coverage.csv", COVERAGE_HEADER, coverage_rows(w, nonreturns))
    (d / "estimates.csv").write_text("condition,group\nidentity,G0\n")
    (d / "input_manifest.csv").write_text("path,sha256\n")
    (d / "stimulus_manifest_v2.csv").write_text("slide_id,arm\n")
    (d / "logs").mkdir()
    (d / "logs" / "eval.log").write_text("ok\n")
    files = write_complete(d)
    if complete_hook:
        complete_hook(d)
        files = json.loads((d / "COMPLETE").read_text())["files"]
    acc = {"comparator": name, "disposition": "accepted", "pipeline_id": name,
           "recorded_at": "2026-10-07T18:00:00-05:00",
           "complete_sha256": sha_file(d / "COMPLETE"),
           "blinded_sha256": {r: e["sha256"] for r, e in files.items() if not cli.whitelisted(r)}}
    if acc_hook:
        acc_hook(acc)
    accp = base / f"acc_{name}.json"
    write_json(accp, acc)
    return dict(name=name, dir=d, acc=accp, pin=sha_file(accp), vkey=vkey)


def cli_argv(w: World, out: Path, comps=(), stage="replay", extra=(), dose_dir=None,
             pins=None, hopt=None, spec="auto"):
    """``spec="auto"``: non-replay stages get ``--spec-sha256`` of ``<P2ROOT>/n3_spec_001/
    SPEC.md`` when that file exists (amendment 004 item 1); ``None`` omits it; a string is
    passed verbatim."""
    dd = Path(dose_dir or w.dose_dir)
    hd = Path(hopt or w.hopt_dir)
    p = dict(
        qm=sha_file(hd / "query_metrics.csv.gz"),
        rr=sha_file(hd / "run_record.json"),
        pd=sha_file(hd / "paired_deltas.csv"),
        dr=sha_file(hd / "patient_draws.npy"),
        q=w.queries_sha,
        rs=sha_file(dd / "realised_stimuli.csv"), dr2=sha_file(dd / "RECORD.json"),
        st=sha_file(dd / "strata_deltas.csv"),
        ns=sha_file(w.sens_dir / "null_slides.csv"), sr=sha_file(w.sens_dir / "RECORD.json"),
        sd=sha_file(w.sens_dir / "sensitivity_deltas.csv"))
    p.update(pins or {})
    argv = ["--out", str(out), "--stage", stage, "--hopt-dir", str(hd),
            "--hopt-query-metrics-sha256", p["qm"], "--hopt-run-record-sha256", p["rr"],
            "--hopt-paired-deltas-sha256", p["pd"], "--hopt-patient-draws-sha256", p["dr"],
            "--queries", str(w.queries), "--queries-sha256", p["q"],
            "--dose-dir", str(dd), "--realised-stimuli-sha256", p["rs"],
            "--dose-record-sha256", p["dr2"], "--strata-deltas-sha256", p["st"],
            "--sensitivity-dir", str(w.sens_dir), "--null-slides-sha256", p["ns"],
            "--sensitivity-record-sha256", p["sr"], "--sensitivity-deltas-sha256", p["sd"]]
    for c in comps:
        argv += ["--comparator", f"{c['name']}={c['dir']}", "--acceptance", f"{c['name']}={c['acc']}",
                 "--acceptance-sha256", f"{c['name']}={c['pin']}"]
    spec_file = hd.resolve().parent / "n3_spec_001" / "SPEC.md"
    if spec == "auto":
        spec = sha_file(spec_file) if stage != "replay" and spec_file.is_file() else None
    if spec is not None:
        argv += ["--spec-sha256", spec]
    return argv + list(extra)


# =============================================================================================
# Expected outputs from the independent reference
# =============================================================================================

D_CUTOFFS = {"hit": (1, 3, 5, 10), "precision": (3, 5, 10)}
PATTERNS = ("both_exclude_same_sign_positive", "both_exclude_same_sign_negative", "mixed",
            "not_established", "unavailable")


def ref_cause(w: World, hkey, ckey, pop, cond, metric, k, group):
    m = w.masks[pop] if group == "MACRO" else w.masks[pop] & (w.labels_arr == group)
    dh = (w.vals[hkey][(cond, metric, k)] - w.vals[hkey][("identity", metric, k)])[m]
    dc = (w.vals[ckey][(cond, metric, k)] - w.vals[ckey][("identity", metric, k)])[m]
    if np.all(dh == 0) and np.all(dc == 0):
        return "no_change_either"
    if np.array_equal(dh, dc):
        return "identical_nonzero_changes"
    return None


def expected_rows(w: World, comps, hkey="H"):
    """comps: list of (name, vkey) in output order -> list of (key tuple, values dict, reps)."""
    out = []
    for name, ckey in comps:
        for cond in w.arms:
            for pop in w.pops_for(cond):
                ent = w.table_entry(pop)
                for metric in ("hit", "precision"):
                    for k in D_CUTOFFS[metric]:
                        H = w.paired(hkey, pop, cond, metric, k)
                        C = w.paired(ckey, pop, cond, metric, k)
                        for gi, g in enumerate((*w.groups, "MACRO")):
                            eh, ec = H["d"][gi], C["d"][gi]
                            cause = ref_cause(w, hkey, ckey, pop, cond, metric, k, g)
                            dd = ref_compose(eh, ec, cause == "identical_nonzero_changes")
                            primary = g == "MACRO" and pop == "full"
                            role = ("primary" if primary and k == 5 else
                                    "secondary" if primary else "descriptive")
                            zwb = (None if "unavailable" in (eh["ci_status"], ec["ci_status"])
                                   else (eh["ci_low"] == 0.0 and eh["ci_high"] == 0.0
                                         and ec["ci_low"] == 0.0 and ec["ci_high"] == 0.0))
                            ah, bh, ac, bc = H["a"][gi], H["b"][gi], C["a"][gi], C["b"][gi]
                            vals = {
                                "delta_hopt": eh["point"], "delta_hopt_ci_low": eh["ci_low"],
                                "delta_hopt_ci_high": eh["ci_high"],
                                "delta_c": ec["point"], "delta_c_ci_low": ec["ci_low"],
                                "delta_c_ci_high": ec["ci_high"],
                                "D": dd["point"], "D_ci_low": dd["ci_low"],
                                "D_ci_high": dd["ci_high"], "D_excludes_zero": dd["excludes"],
                                "identity_abs_hopt": bh["point"], "identity_abs_c": bc["point"],
                                "arm_abs_hopt": ah["point"], "arm_abs_c": ac["point"],
                                "identity_abs_hopt_ci_low": bh["ci_low"],
                                "identity_abs_hopt_ci_high": bh["ci_high"],
                                "identity_abs_c_ci_low": bc["ci_low"],
                                "identity_abs_c_ci_high": bc["ci_high"],
                                "arm_abs_hopt_ci_low": ah["ci_low"],
                                "arm_abs_hopt_ci_high": ah["ci_high"],
                                "arm_abs_c_ci_low": ac["ci_low"], "arm_abs_c_ci_high": ac["ci_high"],
                                "n_pairs": eh["nq"], "n_patients": eh["npat"],
                                "point_status": dd["point_status"],
                                "point_reason": dd["point_reason"], "ci_status": dd["ci_status"],
                                "ci_reason": dd["ci_reason"], "zero_support_replicates": dd["zsr"],
                                "zero_width_both": zwb, "zero_width_cause": cause,
                                "draws_multiplicities_sha256": ent["multiplicities_sha256"],
                                "draws_patients_sha256": ent["patients_sha256"]}
                            out.append(((name, cond, pop, g, metric, str(k), role), vals, dd))
    return out


FLOAT_COLS = {"delta_hopt", "delta_hopt_ci_low", "delta_hopt_ci_high", "delta_c",
              "delta_c_ci_low", "delta_c_ci_high", "D", "D_ci_low", "D_ci_high",
              "identity_abs_hopt", "identity_abs_c", "arm_abs_hopt", "arm_abs_c",
              "identity_abs_hopt_ci_low", "identity_abs_hopt_ci_high", "identity_abs_c_ci_low",
              "identity_abs_c_ci_high", "arm_abs_hopt_ci_low", "arm_abs_hopt_ci_high",
              "arm_abs_c_ci_low", "arm_abs_c_ci_high"}


def ref_pattern(no_d, h, p):
    if no_d:
        return "unavailable", no_d
    un = [lab for lab, e in (("hit5", h), ("prec5", p)) if e["ci_status"] == "unavailable"]
    if un:
        return "unavailable", "both_ci_unavailable" if len(un) == 2 else f"{un[0]}_ci_unavailable"
    if not (h["excludes"] and p["excludes"]):
        return "not_established", None
    if h["ci_low"] > 0 and p["ci_low"] > 0:
        return "both_exclude_same_sign_positive", None
    if h["ci_high"] < 0 and p["ci_high"] < 0:
        return "both_exclude_same_sign_negative", None
    return "mixed", None


def verify_against_reference(w: World, out: Path, comps, omitted=(), hkey="H"):
    """Assert every d_contrasts/patterns/counts/npy value equals the independent reference."""
    exp = expected_rows(w, comps, hkey)
    rows = read_csv(out / "d_contrasts.csv")
    assert tuple(rows[0].keys()) == cli.D_HEADER
    keys = [(r["comparator"], r["condition"], r["population"], r["group"], r["metric"], r["k"],
             r["endpoint_role"]) for r in rows]
    assert keys == [e[0] for e in exp], "d_contrasts row set/order differs from reference"
    bad = []
    for r, (key, vals, _) in zip(rows, exp):
        for col, v in vals.items():
            s = r[col]
            if col in FLOAT_COLS:
                if (s == "") != (v is None) or (v is not None and not abs(float(s) - v) <= TOL):
                    bad.append((key, col, s, v))
            elif s != fmt(v):
                bad.append((key, col, s, v))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:3]}"
    # replicates (N3-17): every MACRO full row
    arr = np.load(out / "d_replicates.npy", allow_pickle=False)
    idx = read_csv(out / "d_replicates_index.csv")
    want = [(key, dd) for key, _, dd in exp if key[3] == "MACRO" and key[2] == "full"]
    assert arr.dtype == np.float64 and arr.shape == (len(want), COUNT) and arr.flags.c_contiguous
    assert [(r["comparator"], r["condition"], r["population"], r["group"], r["metric"], r["k"],
             r["endpoint_role"]) for r in idx] == [k for k, _ in want]
    assert [int(r["row"]) for r in idx] == list(range(len(want)))
    for i, (_, dd) in enumerate(want):
        assert np.array_equal(np.isnan(arr[i]), ~dd["valid"])
        assert np.all(np.abs(arr[i][dd["valid"]] - dd["reps"][dd["valid"]]) <= TOL)
    # patterns + counts
    prim = {(key[0], key[1], key[4]): dd for key, _, dd in exp if key[6] == "primary"}
    prow = read_csv(out / "patterns.csv")
    counts = json.loads((out / "counts.json").read_text())
    names = cli.pipeline_order([n for n, _ in comps] + [n for n, _ in omitted])
    want_p = []
    for name in names:
        tally = {"primary_intervals_nominal": 24,
                 "primary_intervals_computed": {"hit@5": 0, "precision@5": 0, "total": 0},
                 "primary_intervals_excluding_zero": {"hit@5": 0, "precision@5": 0, "total": 0},
                 "patterns": {p: 0 for p in PATTERNS}}
        no_d = dict(omitted).get(name)
        for cond in w.arms:
            h = p = None
            if not no_d:
                h, p = prim[(name, cond, "hit")], prim[(name, cond, "precision")]
                for ep, e in (("hit@5", h), ("precision@5", p)):
                    if e["ci_status"] == "estimable":
                        tally["primary_intervals_computed"][ep] += 1
                        tally["primary_intervals_computed"]["total"] += 1
                        if e["excludes"]:
                            tally["primary_intervals_excluding_zero"][ep] += 1
                            tally["primary_intervals_excluding_zero"]["total"] += 1
            pat, reason = ref_pattern(no_d, h, p)
            tally["patterns"][pat] += 1
            want_p.append([name, name, cond] + ([fmt(h["point"]), fmt(h["ci_low"]), fmt(h["ci_high"]),
                                                 fmt(h["excludes"]), fmt(p["point"]),
                                                 fmt(p["ci_low"]), fmt(p["ci_high"]),
                                                 fmt(p["excludes"])] if h else [""] * 8)
                          + [pat, fmt(reason), fmt(cond == "reinhard")])
        got = counts["per_comparator"][name]
        assert got["disposition"] == (no_d or "contrasted")
        for f in tally:
            assert got[f] == tally[f], (name, f)
    got_p = [[r[c] for c in cli.PATTERN_HEADER] for r in prow]
    for g, e in zip(got_p, want_p):
        for i, col in enumerate(cli.PATTERN_HEADER):
            if col in ("hit5_D", "hit5_ci_low", "hit5_ci_high", "prec5_D", "prec5_ci_low",
                       "prec5_ci_high") and e[i] != "":
                assert abs(float(g[i]) - float(e[i])) <= TOL, (e[:3], col)
            else:
                assert g[i] == e[i], (e[:3], col, g[i], e[i])
    assert len(got_p) == len(want_p)
    assert counts["pipeline_ids"] == names
    assert counts["stage_totals"]["primary_intervals_nominal"] == 24 * len(names)
    assert counts["stage_totals"]["required_set_size"] == len(names)
    return exp


def check_complete(out: Path):
    comp = json.loads((out / "COMPLETE").read_text())["files"]
    present = sorted(p.name for p in out.iterdir() if p.name != "COMPLETE")
    assert sorted(comp) == present
    for f, e in comp.items():
        assert e["sha256"] == sha_file(out / f) and e["bytes"] == (out / f).stat().st_size
    assert not Path(str(out) + ".partial").exists()
    return comp


SHA64 = re.compile(r"\b[0-9a-f]{64}\b")
# A float token is a standalone numeral with a decimal point or exponent (identifiers such as
# "check2.incomplete_grid" or "n3_16" are not numerals), or nan/inf.
FLOAT_TOKEN = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?!\w)"
                         r"|(?<![\w.])[-+]?\d+[eE][-+]?\d+(?!\w)"
                         r"|\b(?:nan|inf|infinity)\b", re.IGNORECASE)


def float_tokens(text: str, allow=()):
    """Strict leak rule (N3-03): no float token after removing allow-listed paths and shas."""
    for a in sorted({str(x) for x in allow if str(x)}, key=len, reverse=True):
        text = text.replace(a, " ")
    text = SHA64.sub(" ", text)
    return FLOAT_TOKEN.findall(text)


def json_has_float(obj) -> bool:
    if isinstance(obj, float):
        return True
    if isinstance(obj, dict):
        return any(json_has_float(v) for v in obj.values())
    if isinstance(obj, list):
        return any(json_has_float(v) for v in obj)
    return False


# =============================================================================================
# Fixtures and runners
# =============================================================================================

@pytest.fixture(scope="module")
def world_ok(tmp_path_factory):
    return build_synthetic_world(tmp_path_factory.mktemp("world_ok"), tiny_group=False)


@pytest.fixture(scope="module")
def world_tiny(tmp_path_factory):
    return build_synthetic_world(tmp_path_factory.mktemp("world_tiny"), tiny_group=True)


@pytest.fixture
def patch_pop(monkeypatch):
    def apply(w: World):
        monkeypatch.setattr(cli, "POPULATION_SHA256", w.queries_sha)
        monkeypatch.setattr(cli, "POPULATION_N_SLIDES", w.n)
        monkeypatch.setattr(cli, "POPULATION_N_PATIENTS", w.n_patients)
    return apply


def run_main(argv, capsys=None):
    rc = cli.main(argv)
    return rc


def expect_failed(argv, check, capsys):
    out = Path(argv[argv.index("--out") + 1])
    rc = cli.main(argv)
    cap = capsys.readouterr()
    assert rc == 1, cap.err
    assert sorted(p.name for p in out.iterdir()) == ["FAILED.json"]
    assert not Path(str(out) + ".partial").exists()
    failed = json.loads((out / "FAILED.json").read_text())
    assert failed["check"] == check, failed["check"]
    assert failed["status"] == "FAILED"
    assert not json_has_float(failed)
    allow = [out, *argv]
    assert float_tokens((out / "FAILED.json").read_text(), allow) == []
    assert float_tokens(cap.err, allow) == [] and float_tokens(cap.out, allow) == []
    assert cap.err.strip() == f"FAILED: {out} check={check}"
    return failed


def expect_refused(argv, check, capsys):
    """Amendment 004 items 1, 4, 5: refusal before any output -- rc 2, nothing written."""
    out = Path(argv[argv.index("--out") + 1])
    before = sorted(p.name for p in out.parent.iterdir())
    rc = cli.main(argv)
    cap = capsys.readouterr()
    assert rc == 2, cap.err
    assert cap.err.strip() == f"REFUSED: {out} check={check}" and cap.out == ""
    assert sorted(p.name for p in out.parent.iterdir()) == before
    assert not os.path.lexists(out) and not os.path.lexists(str(out) + ".partial")


def replace_arg(argv, flag, value):
    a = list(argv)
    a[a.index(flag) + 1] = str(value)
    return a


def make_acceptance(base: Path, name: str, d: Path, **over):
    files = json.loads((d / "COMPLETE").read_text())["files"]
    acc = {"comparator": name, "disposition": "accepted", "pipeline_id": name,
           "recorded_at": "2026-10-07T18:00:00-05:00",
           "complete_sha256": sha_file(d / "COMPLETE"),
           "blinded_sha256": {r: e["sha256"] for r, e in files.items() if not cli.whitelisted(r)}}
    acc.update(over)
    p = base / f"acc_{name}.json"
    write_json(p, acc)
    return dict(name=name, dir=d, acc=p, pin=sha_file(p))


def run_ok(argv, capsys):
    out = Path(argv[argv.index("--out") + 1])
    rc = cli.main(argv)
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    check_complete(out)
    allow = [out, *argv]
    assert float_tokens(cap.out, allow) == [] and float_tokens(cap.err, allow) == []
    assert cap.err == ""
    return out, cap


OUTPUT_FILES = ("d_contrasts.csv", "patterns.csv", "counts.json", "d_replicates.npy",
                "d_replicates_index.csv", "replay_check.json")


# =============================================================================================
# §6.4/6.5/6.8-style known-D tests against the independent reference
# =============================================================================================

def test_reference_draws_match_kernel_rule():
    """Sanity: in-test draws reproduce the kernel table (pins the shared RNG convention only)."""
    pats = [f"P{i:03d}" for i in range(17)]
    order, w = ref_draws(pats)
    kd = cli.kernel.make_patient_draws(pats, count=COUNT, seed=SEED)
    assert tuple(kd.patient_ids) == order and np.array_equal(kd.multiplicities, w)


def test_known_d_matches_independent_bootstrap(world_ok, patch_pop, tmp_path, capsys):
    w = world_ok
    patch_pop(w)
    c = make_comparator(w, tmp_path, "synthetic_c", "C")
    out, cap = run_ok(cli_argv(w, tmp_path / "out", [c]), capsys)
    assert "N3 comparators processed: 1 omitted: 0" in cap.out
    exp = verify_against_reference(w, out, [("synthetic_c", "C")])
    prim = {(k[1], k[4]): dd for k, _, dd in exp if k[6] == "primary"}
    # designed signs: H improves / C degrades -> D > 0; gaussian_blur: H degrades -> D < 0
    assert prim[("rotate_90", "hit")]["ci_low"] > 0 and prim[("rotate_90", "precision")]["ci_low"] > 0
    assert prim[("gaussian_blur", "hit")]["ci_high"] < 0
    assert prim[(UNCHANGED_ARM, "hit")]["point"] == 0.0
    rows = read_csv(out / "d_contrasts.csv")
    causes = {r["zero_width_cause"] for r in rows if r["condition"] == UNCHANGED_ARM}
    assert causes == {"no_change_either"}
    pats = {r["condition"]: r["pattern"] for r in read_csv(out / "patterns.csv")}
    assert pats["rotate_90"] == "both_exclude_same_sign_positive"
    assert pats[UNCHANGED_ARM] == "not_established"
    rec = json.loads((out / "RECORD.json").read_text())
    assert rec["stage"] == "replay" and rec["attempt"] == 1
    assert rec["comparators"]["synthetic_c"]["check6"]["passed"] is True
    assert rec["code"]["kernel_sha256"] == cli.KERNEL_SHA256
    rc = json.loads((out / "replay_check.json").read_text())
    assert set(rc["targets"]) == {"paired_deltas.csv", "strata_deltas.csv",
                                  "sensitivity_deltas.csv"}


def test_known_d_with_zero_support_group(world_tiny, patch_pop, tmp_path, capsys):
    w = world_tiny
    patch_pop(w)
    c = make_comparator(w, tmp_path, "synthetic_c", "C")
    out, _ = run_ok(cli_argv(w, tmp_path / "out", [c]), capsys)
    exp = verify_against_reference(w, out, [("synthetic_c", "C")])
    prim = [dd for k, _, dd in exp if k[6] == "primary"]
    assert all(dd["point"] is not None for dd in prim)
    assert all(dd["ci_status"] == "unavailable" and dd["ci_reason"] == "zero_support_replicates"
               and dd["zsr"] > 0 for dd in prim)
    pr = read_csv(out / "patterns.csv")
    assert {(r["pattern"], r["pattern_reason"]) for r in pr} == {("unavailable",
                                                                  "both_ci_unavailable")}
    counts = json.loads((out / "counts.json").read_text())
    assert counts["per_comparator"]["synthetic_c"]["primary_intervals_computed"]["total"] == 0


def test_sign_flip_mutation_detected(world_ok, patch_pop, tmp_path, capsys, monkeypatch):
    w = world_ok
    patch_pop(w)
    orig = cli.compose

    def flipped(e_h, e_c, identical_nonzero=False):
        d = orig(e_h, e_c, identical_nonzero)
        if d["point"] is not None:
            d["point"] = -d["point"]
        if d["ci_low"] is not None:
            d["ci_low"], d["ci_high"] = -d["ci_high"], -d["ci_low"]
        d["replicates"] = -d["replicates"]
        return d
    monkeypatch.setattr(cli, "compose", flipped)
    c = make_comparator(w, tmp_path, "synthetic_c", "C")
    out, _ = run_ok(cli_argv(w, tmp_path / "out", [c]), capsys)
    with pytest.raises(AssertionError):
        verify_against_reference(w, out, [("synthetic_c", "C")])


def test_swapped_argument_mutation_detected(world_ok, patch_pop, tmp_path, capsys, monkeypatch):
    w = world_ok
    patch_pop(w)
    orig = cli.compose
    monkeypatch.setattr(cli, "compose", lambda e_h, e_c, identical_nonzero=False:
                        orig(e_c, e_h, identical_nonzero))
    c = make_comparator(w, tmp_path, "synthetic_c", "C")
    out, _ = run_ok(cli_argv(w, tmp_path / "out", [c]), capsys)
    with pytest.raises(AssertionError):
        verify_against_reference(w, out, [("synthetic_c", "C")])


def test_kernel_arm_swap_fails_replay(world_ok, patch_pop, tmp_path, capsys, monkeypatch):
    w = world_ok
    patch_pop(w)
    orig = cli.kernel.summarize_paired

    def swapped(q, p, g, groups, va, vb, inc, draws):
        return orig(q, p, g, groups, vb, va, inc, draws)
    monkeypatch.setattr(cli.kernel, "summarize_paired", swapped)
    c = make_comparator(w, tmp_path, "synthetic_c", "C")
    expect_failed(cli_argv(w, tmp_path / "out", [c]), "replay.paired_deltas.mismatch", capsys)


def test_self_contrast_identical_nonzero(world_ok, patch_pop, tmp_path, capsys):
    """§6.2 synthetic analogue: comparator == H values -> D = 0, CI [0, 0] (N3-09)."""
    w = world_ok
    patch_pop(w)
    c = make_comparator(w, tmp_path, "synthetic_self", "H")
    out, _ = run_ok(cli_argv(w, tmp_path / "out", [c]), capsys)
    verify_against_reference(w, out, [("synthetic_self", "H")])
    rows = read_csv(out / "d_contrasts.csv")
    assert {r["zero_width_cause"] for r in rows} == {"no_change_either",
                                                      "identical_nonzero_changes"}
    for r in rows:
        assert r["D"] in ("", "0.0")
        if r["ci_status"] == "estimable":
            assert (r["D_ci_low"], r["D_ci_high"], r["D_excludes_zero"]) == ("0.0", "0.0", "false")
    arr = np.load(out / "d_replicates.npy", allow_pickle=False)
    assert np.all((arr == 0.0) | np.isnan(arr))
    assert {r["pattern"] for r in read_csv(out / "patterns.csv")} == {"not_established"}


def _e(lo, hi, status="estimable"):
    return {"ci_status": status, "ci_low": lo, "ci_high": hi,
            "excludes_zero": None if status != "estimable" else (lo > 0 or hi < 0)}


def test_pattern_of_precedence():
    assert cli.pattern_of("dropped_n4", None, None) == ("unavailable", "dropped_n4")
    un = _e(None, None, "unavailable")
    assert cli.pattern_of(None, un, un) == ("unavailable", "both_ci_unavailable")
    assert cli.pattern_of(None, un, _e(0.1, 0.2)) == ("unavailable", "hit5_ci_unavailable")
    assert cli.pattern_of(None, _e(0.1, 0.2), un) == ("unavailable", "prec5_ci_unavailable")
    assert cli.pattern_of(None, _e(-0.1, 0.2), _e(0.1, 0.2)) == ("not_established", None)
    assert cli.pattern_of(None, _e(0.1, 0.2), _e(0.05, 0.3)) == (
        "both_exclude_same_sign_positive", None)
    assert cli.pattern_of(None, _e(-0.3, -0.1), _e(-0.2, -0.01)) == (
        "both_exclude_same_sign_negative", None)
    assert cli.pattern_of(None, _e(0.1, 0.2), _e(-0.2, -0.1)) == ("mixed", None)


def test_compose_identical_nonzero_and_unavailable():
    from types import SimpleNamespace as NS
    valid = np.ones(COUNT, dtype=bool)
    reps = np.linspace(-0.1, 0.1, COUNT)
    e = NS(point=0.05, point_reason=None, replicates=reps, valid=valid, n_queries=10,
           n_patients=7)
    d = cli.compose(e, e, identical_nonzero=True)
    assert d["point"] == 0.0 and (d["ci_low"], d["ci_high"]) == (0.0, 0.0)
    assert d["excludes_zero"] is False and np.all(d["replicates"] == 0.0)
    e2 = NS(point=None, point_reason="missing_group_support", replicates=reps, valid=valid,
            n_queries=10, n_patients=7)
    d = cli.compose(e, e2)
    assert d["point"] is None and d["ci_status"] == "unavailable"
    assert d["ci_reason"] == "missing_group_support" and d["excludes_zero"] is None
    inv = valid.copy()
    inv[3] = False
    e3 = NS(point=0.01, point_reason=None, replicates=reps, valid=inv, n_queries=10,
            n_patients=7)
    with pytest.raises(cli.N3Error):
        cli.compose(e, e3)
    d = cli.compose(e3, e3)
    assert d["ci_reason"] == "zero_support_replicates" and d["zero_support_replicates"] == 1


# =============================================================================================
# Admission / input-check refusals (FAILED.json only, no partial outputs)
# =============================================================================================

def _first_row(rows, cond, k, pred=lambda r: True):
    for r in rows:
        if r[0] == cond and r[5] == str(k) and pred(r):
            return r
    raise AssertionError("fixture row not found")


def test_nonreturn_comparator_reported_not_contrasted(world_ok, patch_pop, tmp_path, capsys):
    w = world_ok
    patch_pop(w)

    def nr(rows):
        r = _first_row(rows, "rotate_90", 5)
        r[6], r[8], r[9] = "empty_query", "", ""
        return rows
    ok = make_comparator(w, tmp_path, "synthetic_ok", "C")
    bad = make_comparator(w, tmp_path, "synthetic_nr", "C", qm_hook=nr, nonreturns=1)
    out, cap = run_ok(cli_argv(w, tmp_path / "out", [bad, ok]), capsys)
    assert "processed: 1 omitted: 1" in cap.out
    verify_against_reference(w, out, [("synthetic_ok", "C")],
                             omitted=[("synthetic_nr", "nonreturn")])
    rec = json.loads((out / "RECORD.json").read_text())
    assert rec["comparators"]["synthetic_nr"]["disposition"] == "nonreturn"
    assert rec["comparators"]["synthetic_nr"]["query_metrics_nonreturn_rows"] == 1


@pytest.mark.parametrize("case,check", [
    ("join", "check2.joined_key_identity_mismatch"),
    ("incomplete", "check2.incomplete_grid"),
    ("duplicate", "check2.duplicate_key"),
    ("header", "check1.query_metrics_header"),
    ("range", "check3.value_outside_unit_interval"),
    ("prec1", "n3_16.precision1_differs_from_hit1"),
])
def test_query_metrics_refusals(world_ok, patch_pop, tmp_path, capsys, case, check):
    w = world_ok
    patch_pop(w)

    def hook(rows):
        if case == "join":
            _first_row(rows, "brightness", 3)[3] = "PX_999"
        elif case == "incomplete":
            rows.pop(len(rows) // 2)
        elif case == "duplicate":
            rows[len(rows) // 2] = list(rows[len(rows) // 2 + 4])
        elif case == "header":
            return (("condition", "population_index", "query", "patient", *QHEADER[4:]), rows)
        elif case == "range":
            _first_row(rows, "contrast", 3)[8] = "1.5"
        elif case == "prec1":
            r = _first_row(rows, "saturation", 1, lambda r: r[8] == "1.0")
            r[9] = "0.5"
        return rows
    c = make_comparator(w, tmp_path, "synthetic_c", "C", qm_hook=hook)
    expect_failed(cli_argv(w, tmp_path / "out", [c]), check, capsys)


@pytest.mark.parametrize("case,check", [
    ("cond12", "check1.conditions_not_13_identity_first"),
    ("cond14", "check1.conditions_not_13_identity_first"),
    ("pid_missing", "n3_01.pipeline_id_missing"),
    ("pid_other", "n3_01.pipeline_id_differs"),
    ("manifest", "c2.manifest_v2_sha256_differs"),
    ("no_tables", "c2.full_draw_table_missing"),
    ("wrong_table", "c2.full_draw_table_missing"),
    ("population", "n3_10.population_invariant"),
])
def test_run_record_refusals(world_ok, patch_pop, tmp_path, capsys, case, check):
    w = world_ok
    patch_pop(w)
    pid = "same"
    if case == "pid_missing":
        pid = None
    elif case == "pid_other":
        pid = "synthetic_other"

    def hook(rec):
        if case == "cond12":
            rec["conditions"] = rec["conditions"][:12]
        elif case == "cond14":
            rec["conditions"] = rec["conditions"] + ["extra_arm"]
        elif case == "manifest":
            rec["manifest_v2_sha256"] = "cd" * 32
        elif case == "no_tables":
            rec["draw_tables"] = []
        elif case == "wrong_table":
            rec["draw_tables"] = [dict(rec["draw_tables"][0], multiplicities_sha256="0" * 64)]
        elif case == "population":
            rec["population"]["evaluated_patients"] += 1
    c = make_comparator(w, tmp_path, "synthetic_c", "C", pipeline_id=pid, rr_hook=hook)
    expect_failed(cli_argv(w, tmp_path / "out", [c]), check, capsys)


def test_check6_mismatch_fails_whole_run(world_ok, patch_pop, tmp_path, capsys):
    w = world_ok
    patch_pop(w)

    def hook(rows):
        for r in rows:
            if r[0] == "flip_vertical" and r[1] == "MACRO" and r[2] == "precision" and r[3] == 1:
                r[4] = r[4] + 1e-6
    ok = make_comparator(w, tmp_path, "synthetic_a", "C")
    bad = make_comparator(w, tmp_path, "synthetic_b", "C", pd_hook=hook)
    failed = expect_failed(cli_argv(w, tmp_path / "out", [ok, bad]), "check6.mismatch", capsys)
    assert failed["context"]["pipeline_id"] == "synthetic_b"
    assert failed["context"]["columns"] == ["delta"]


@pytest.mark.parametrize("case,check", [
    ("no_acceptance", "acceptance.record_missing"),
    ("pin", "acceptance.record_sha256_differs_from_pin"),
    ("blinded_sha", "acceptance.sha_differs_from_complete"),
    ("blinded_set", "acceptance.blinded_set_differs_from_complete_complement"),
    ("blinded_extra", "acceptance.blinded_set_differs_from_complete_complement"),
    ("complete_sha", "acceptance.complete_sha256_differs"),
    ("disposition", "acceptance.disposition_not_accepted"),
    ("recorded_at", "acceptance.recorded_at_missing"),
    ("acc_pid", "acceptance.pipeline_id_differs_from_name"),
    ("read_differs", "complete_chain.read_file_differs"),
    ("blinded_missing", "acceptance.blinded_file_missing"),
    ("blinded_tamper", "acceptance.blinded_file_differs"),
])
def test_acceptance_refusals(world_ok, patch_pop, tmp_path, capsys, case, check):
    w = world_ok
    patch_pop(w)

    def acc_hook(acc):
        if case == "blinded_sha":
            acc["blinded_sha256"]["estimates.csv"] = "0" * 64
        elif case == "blinded_set":
            acc["blinded_sha256"].pop("estimates.csv")
        elif case == "blinded_extra":
            acc["blinded_sha256"]["coverage.csv"] = "1" * 64
        elif case == "complete_sha":
            acc["complete_sha256"] = "2" * 64
        elif case == "disposition":
            acc["disposition"] = "pending"
        elif case == "recorded_at":
            acc["recorded_at"] = "yesterday"
        elif case == "acc_pid":
            acc["pipeline_id"] = "synthetic_other"
    c = make_comparator(w, tmp_path, "synthetic_c", "C", acc_hook=acc_hook)
    argv = cli_argv(w, tmp_path / "out", [c])
    if case == "no_acceptance":
        i = argv.index("--acceptance")
        argv = argv[:i] + argv[i + 2:]
    elif case == "pin":
        argv = replace_arg(argv, "--acceptance-sha256", "synthetic_c=" + "3" * 64)
    elif case == "read_differs":
        (c["dir"] / "coverage.csv").write_text("tampered\n")
    elif case == "blinded_missing":
        (c["dir"] / "estimates.csv").unlink()
    elif case == "blinded_tamper":
        # N3I-01: COMPLETE and the acceptance record still agree; only the file bytes changed.
        (c["dir"] / "estimates.csv").write_text("condition,group\nidentity,G1\n")
    expect_failed(argv, check, capsys)


def test_log_suffix_whitelist_narrowed(world_ok, patch_pop, tmp_path, capsys):
    """Amendment 003 item 11 (A4): *.log escapes the blinded set only under logs/, lsf_logs/."""
    assert cli.whitelisted("logs/eval.log") and cli.whitelisted("lsf_logs/job.log")
    assert not cli.whitelisted("run.log") and not cli.whitelisted("sub/x.log")
    w = world_ok
    patch_pop(w)

    def add_log(d):
        (d / "run.log").write_text("top-level log\n")
        write_complete(d)
    c = make_comparator(w, tmp_path / "a", "synthetic_c", "C", complete_hook=add_log)
    assert "run.log" in json.loads(c["acc"].read_text())["blinded_sha256"]
    out, _ = run_ok(cli_argv(w, tmp_path / "o1", [c]), capsys)
    rec = json.loads((out / "RECORD.json").read_text())
    assert "run.log" in rec["comparators"]["synthetic_c"]["blinded_sha256"]
    assert "logs/eval.log" not in rec["comparators"]["synthetic_c"]["blinded_sha256"]
    (c["dir"] / "run.log").write_text("tampered\n")
    f = expect_failed(cli_argv(w, tmp_path / "o2", [c]), "acceptance.blinded_file_differs",
                      capsys)
    assert f["context"]["file"] == "run.log"


def test_input_pin_and_population_refusals(world_ok, patch_pop, tmp_path, capsys):
    w = world_ok
    c = make_comparator(w, tmp_path, "synthetic_c", "C")
    # population constant not patched -> real 5ff0ea3b... pin differs from the synthetic queries
    expect_failed(cli_argv(w, tmp_path / "o1", [c]), "n3_10.queries_sha256_differs", capsys)
    patch_pop(w)
    expect_failed(cli_argv(w, tmp_path / "o2", [c], pins={"st": "4" * 64}),
                  "pins.sha256_differs", capsys)
    expect_failed(cli_argv(w, tmp_path / "o3", [c], pins={"st": "xyz"}), "pins.malformed",
                  capsys)


def test_tampered_dose_record_refused(world_ok, patch_pop, tmp_path, capsys):
    w = world_ok
    patch_pop(w)
    c = make_comparator(w, tmp_path, "synthetic_c", "C")
    d1 = tmp_path / "dose_mult"
    shutil.copytree(w.dose_dir, d1)
    rec = json.loads((d1 / "RECORD.json").read_text())
    rec["draw_tables"][1]["multiplicities_sha256"] = "5" * 64
    write_json(d1 / "RECORD.json", rec)
    expect_failed(cli_argv(w, tmp_path / "o1", [c], dose_dir=d1),
                  "check4.multiplicities_sha256_differs", capsys)
    d2 = tmp_path / "dose_dup"
    shutil.copytree(w.dose_dir, d2)
    rec = json.loads((d2 / "RECORD.json").read_text())
    rec["draw_tables"].append(dict(rec["draw_tables"][1]))
    write_json(d2 / "RECORD.json", rec)
    expect_failed(cli_argv(w, tmp_path / "o2", [c], dose_dir=d2),
                  "check4.patients_sha256_match_count", capsys)
    d3 = tmp_path / "dose_stratum"
    shutil.copytree(w.dose_dir, d3)
    rows = read_csv(d3 / "realised_stimuli.csv")
    rows[0]["stratum"] = {"T1": "T3", "T2": "T1", "T3": "T1"}[rows[0]["stratum"]]
    write_csv(d3 / "realised_stimuli.csv", tuple(rows[0]), [list(r.values()) for r in rows])
    expect_failed(cli_argv(w, tmp_path / "o3", [c], dose_dir=d3),
                  "population.stratum_inconsistent_with_cuts", capsys)


def test_shift_null_rows_must_equal_run_record(world_ok, patch_pop, tmp_path, capsys):
    w = world_ok
    patch_pop(w)
    c = make_comparator(w, tmp_path, "synthetic_c", "C")
    s = tmp_path / "sens"
    shutil.copytree(w.sens_dir, s)
    rows = read_csv(s / "null_slides.csv")
    other = next(q for q in w.qids if q not in w.null["spatial_shift"]
                 and q not in w.null["brightness"])
    next(r for r in rows if r["arm"] == "spatial_shift")["slide_id"] = other
    write_csv(s / "null_slides.csv", ("slide_id", "arm", "factor", "reason"),
              [list(r.values()) for r in rows])
    argv = cli_argv(w, tmp_path / "out", [c])
    argv = replace_arg(argv, "--sensitivity-dir", s)
    argv = replace_arg(argv, "--null-slides-sha256", sha_file(s / "null_slides.csv"))
    expect_failed(argv, "n3_13.shift_null_rows_differ_from_run_record", capsys)


def test_kernel_pin_refusals(world_ok, patch_pop, tmp_path, capsys, monkeypatch):
    w = world_ok
    patch_pop(w)
    c = make_comparator(w, tmp_path, "synthetic_c", "C")
    with monkeypatch.context() as m:
        m.setattr(cli, "KERNEL_SHA256", "6" * 64)
        expect_failed(cli_argv(w, tmp_path / "o1", [c]), "n3_11.kernel_sha256_differs", capsys)
    with monkeypatch.context() as m:
        m.setattr(cli, "kernel", None)
        expect_failed(cli_argv(w, tmp_path / "o2", [c]),
                      "n3_11.kernel_sha256_differs_before_import", capsys)
    assert cli.KERNEL_PRE_IMPORT_SHA256 == cli.KERNEL_SHA256
    assert sha_file(Path(cli.kernel.__file__)) == cli.KERNEL_SHA256


def test_existing_out_or_partial_refused(world_ok, patch_pop, tmp_path, capsys):
    w = world_ok
    patch_pop(w)
    out = tmp_path / "out"
    out.mkdir()
    assert cli.main(cli_argv(w, out)) == 2
    assert list(out.iterdir()) == []
    out2 = tmp_path / "out2"
    Path(str(out2) + ".partial").mkdir()
    assert cli.main(cli_argv(w, out2)) == 2
    assert not out2.exists()
    capsys.readouterr()


def test_unexpected_exception_is_blinded(world_ok, patch_pop, tmp_path, capsys, monkeypatch):
    w = world_ok
    patch_pop(w)

    def boom(*a, **k):
        raise ValueError("leak 0.123456 D=-0.5")
    monkeypatch.setattr(cli, "compose", boom)
    c = make_comparator(w, tmp_path, "synthetic_c", "C")
    failed = expect_failed(cli_argv(w, tmp_path / "out", [c]), "unexpected", capsys)
    assert failed["exception_class"] == "ValueError"
    assert "0.123456" not in (tmp_path / "out" / "FAILED.json").read_text()


# =============================================================================================
# N3-01 run_001 recognition by run_record sha
# =============================================================================================

def test_noncanonical_hoptimus_1_refused_in_replay(world_ok, patch_pop, tmp_path, capsys,
                                                   monkeypatch):
    """N3I-08: a directory self-declaring pipeline_id hoptimus_1 is refused in every stage."""
    w = world_ok
    patch_pop(w)
    monkeypatch.setattr(cli, "HOPT_RUN_RECORD_SHA256", sha_file(w.hopt_dir / "run_record.json"))
    h = make_comparator(w, tmp_path, "hoptimus_1", "H")
    assert json.loads((h["dir"] / "run_record.json").read_text())["pipeline_id"] == "hoptimus_1"
    expect_failed(cli_argv(w, tmp_path / "o1", [h]),
                  "n3_01.hoptimus_1_requires_run001_run_record", capsys)


def test_hopt_run001_recognised_only_by_run_record_sha(world_ok, patch_pop, tmp_path, capsys,
                                                       monkeypatch):
    w = world_ok
    patch_pop(w)
    assert "pipeline_id" not in json.loads((w.hopt_dir / "run_record.json").read_text())
    h = make_acceptance(tmp_path, "hoptimus_1", w.hopt_dir)
    # without the constant matching, the key-less run_record is refused
    expect_failed(cli_argv(w, tmp_path / "o1", [h]), "n3_01.pipeline_id_missing", capsys)
    monkeypatch.setattr(cli, "HOPT_RUN_RECORD_SHA256", sha_file(w.hopt_dir / "run_record.json"))
    out, _ = run_ok(cli_argv(w, tmp_path / "o2", [h]), capsys)
    verify_against_reference(w, out, [("hoptimus_1", "H")])
    rec = json.loads((out / "RECORD.json").read_text())
    assert rec["comparators"]["hoptimus_1"]["pipeline_id_source"].startswith("constant")
    # the recognised directory cannot be passed under another name
    x = make_acceptance(tmp_path, "synthetic_x", w.hopt_dir)
    expect_failed(cli_argv(w, tmp_path / "o3", [x]), "n3_01.pipeline_id_differs", capsys)
    # nothing was written into the reference directory
    assert sorted(p.name for p in w.hopt_dir.iterdir()) == sorted(
        ["query_metrics.csv.gz", "patient_draws.npy", "run_record.json", "paired_deltas.csv",
         "coverage.csv", "estimates.csv", "COMPLETE"])


# =============================================================================================
# N3-02 stages, N3-14 attempts
# =============================================================================================

def _dropped(base: Path, name: str, disp: str):
    p = base / f"dropped_{name}.json"
    write_json(p, {"pipeline_id": name, "disposition": disp, "reason": "synthetic fixture",
                   "timestamp": "2026-10-07T18:00:00-05:00"})
    return p


def dropped_args(base: Path, name: str, disp: str, pin=None):
    """--dropped plus its mandatory --dropped-sha256 pin (amendment 003 item 4)."""
    p = _dropped(base, name, disp)
    return ["--dropped", f"{name}={p}", "--dropped-sha256", f"{name}={pin or sha_file(p)}"]


def make_p2root(w: World, base: Path, spec=True) -> Path:
    """Per-test stand-in for P2ROOT: a byte-identical copy of the synthetic run_001 (so the
    parent of --hopt-dir is a fresh directory) and optionally a fake n3_spec_001/SPEC.md."""
    p2 = base / "p2root"
    p2.mkdir(parents=True)
    shutil.copytree(w.hopt_dir, p2 / "run_001")
    if spec:
        (p2 / "n3_spec_001").mkdir()
        (p2 / "n3_spec_001" / "SPEC.md").write_text("synthetic spec fixture\n")
    return p2


@pytest.fixture
def real_stage(patch_pop, monkeypatch, tmp_path):
    """Non-replay stages: population pins, the run_001 run_record constant (amendment 003
    item 7) bound to the synthetic run_record, and a fresh per-test P2ROOT."""
    def apply(w: World, spec=True):
        patch_pop(w)
        monkeypatch.setattr(cli, "HOPT_RUN_RECORD_SHA256",
                            sha_file(w.hopt_dir / "run_record.json"))
        p2 = make_p2root(w, tmp_path, spec)
        # amendment 004 items 1 and 4: canonical P2ROOT and single-thread environment.
        monkeypatch.setattr(cli, "CANONICAL_P2ROOT", str(p2))
        for k in cli.THREAD_ENV:
            monkeypatch.setenv(k, "1")
        return p2
    return apply


def b4_gate(base: Path, name="b4_gate.json", **over):
    """Gate JSON naming two existing documents and their shas (amendment 003 item 3)."""
    a = base / "b4_adapter_contract_review.md"
    c = base / "b4_conch_inclusion_rule.md"
    a.write_text("synthetic adapter contract review\n")
    c.write_text("synthetic conch inclusion rule\n")
    g = {"adapter_contract_review_path": str(a), "adapter_contract_review_sha256": sha_file(a),
         "conch_inclusion_rule_path": str(c), "conch_inclusion_rule_sha256": sha_file(c)}
    g.update(over)
    g = {k: v for k, v in g.items() if v is not None}
    p = base / name
    write_json(p, g)
    return p


def s1_extra(base: Path, gate: Path, more=()):
    return [*dropped_args(base, "virchow2_sampler", "dropped_fidelity"), "--b4-gate", str(gate),
            *more]


def s2_extra(base: Path, record: Path, more=()):
    return [*dropped_args(base, "conch_sampler", "dropped_n4"),
            *dropped_args(base, "conch_titan", "nonreturn"), "--stage1-record", str(record),
            *more]


def single_extra(base: Path, more=()):
    return [*dropped_args(base, "virchow2_sampler", "dropped_fidelity"),
            *dropped_args(base, "conch_sampler", "dropped_n4"),
            *dropped_args(base, "conch_titan", "nonreturn"), *more]


@pytest.mark.parametrize("stage,names,extra,check", [
    ("replay", ["uni2_sampler"], [], "n3_02.replay_refuses_pipeline"),
    ("replay", ["hoptimus_x"], [], "n3_02.replay_refuses_pipeline"),
    ("stage1_512px", ["uni2_sampler", "synthetic_a"], ["b4"], "n3_02.pipeline_not_in_enum_for_stage"),
    ("stage1_512px", ["uni2_sampler", "hoptimus_1"], ["b4"], "n3_02.pipeline_not_in_enum_for_stage"),
    ("stage1_512px", ["uni2_sampler"], ["b4"], "n3_02.required_member_omitted"),
    ("stage1_512px", ["uni2_sampler", "virchow2_sampler", "conch_sampler"], ["b4"],
     "n3_02.member_outside_required_set"),
    ("stage1_512px", ["uni2_sampler", "virchow2_sampler"], [], "n3_02.b4_gate_required"),
    ("stage1_512px", ["uni2_sampler", "virchow2_sampler"], ["b4", "s1"],
     "n3_02.stage1_record_only_for_stage2"),
    ("stage2_conch", ["conch_sampler", "conch_titan"], [], "n3_02.stage1_record_required"),
    ("stage2_conch", ["conch_sampler", "conch_titan"], ["b4", "s1"], "n3_02.b4_gate_only_for_stage1"),
    ("stage2_conch", ["conch_sampler"], ["s1"], "n3_02.required_member_omitted"),
    ("single", ["uni2_sampler", "virchow2_sampler"], [], "n3_02.required_member_omitted"),
    ("replay", ["synthetic_a"], ["b4"], "n3_02.b4_gate_only_for_stage1"),
])
def test_stage_refusals(world_ok, real_stage, tmp_path, capsys, stage, names, extra, check):
    w = world_ok
    p2 = real_stage(w)
    a = []
    for n in names:
        a += dropped_args(tmp_path, n, "dropped_n4")
    if "b4" in extra:
        a += ["--b4-gate", str(b4_gate(tmp_path))]
    if "s1" in extra:
        a += ["--stage1-record", str(p2 / "n3_run_s1" / "RECORD.json")]
    out = tmp_path / "n3_replay_x" if stage == "replay" else p2 / "n3_run_x"
    expect_failed(cli_argv(w, out, stage=stage, extra=a, hopt=p2 / "run_001"), check, capsys)


def test_stage1_then_stage2_cumulative_counts(world_ok, real_stage, tmp_path, capsys):
    w = world_ok
    p2 = real_stage(w)
    hopt = p2 / "run_001"
    gate = b4_gate(tmp_path)
    uni = make_comparator(w, tmp_path / "c", "uni2_sampler", "C")
    out1, cap = run_ok(cli_argv(w, p2 / "n3_run_stage1", [uni], stage="stage1_512px",
                                hopt=hopt, extra=s1_extra(tmp_path, gate)), capsys)
    assert "processed: 1 omitted: 1" in cap.out
    verify_against_reference(w, out1, [("uni2_sampler", "C")],
                             omitted=[("virchow2_sampler", "dropped_fidelity")])
    c1 = json.loads((out1 / "counts.json").read_text())
    assert c1["pipeline_ids"] == ["uni2_sampler", "virchow2_sampler"]
    assert c1["stage_totals"]["primary_intervals_nominal"] == 48
    assert "cumulative_totals" not in c1
    rec1 = json.loads((out1 / "RECORD.json").read_text())
    g = json.loads(gate.read_text())
    assert rec1["b4_gate"]["sha256"] == sha_file(gate)
    for k in ("adapter_contract_review", "conch_inclusion_rule"):
        assert rec1["b4_gate"][k + "_sha256"] == g[k + "_sha256"]
        assert rec1["b4_gate"][k + "_path"] == str(Path(g[k + "_path"]).resolve())
    # amendment 003 items 4 and 8
    assert rec1["comparators"]["virchow2_sampler"]["dropped_record_sha256"] == sha_file(
        tmp_path / "dropped_virchow2_sampler.json")
    assert rec1["spec_sha256"] == sha_file(p2 / "n3_spec_001" / "SPEC.md")
    assert rec1["spec_path"] == str((p2 / "n3_spec_001" / "SPEC.md").resolve())
    assert sorted(rec1["thread_env"]) == sorted(cli.THREAD_ENV)
    assert all(rec1["thread_env"][k] == os.environ.get(k) for k in cli.THREAD_ENV)
    assert isinstance(rec1["threadpool_info"], list)
    assert rec1["attempt"] == 1 and rec1["prior_attempts"] == []
    # stage 2: both CONCH members dropped/non-returning; stage 1 is not a conflicting prior
    out2, _ = run_ok(cli_argv(w, p2 / "n3_run_stage2", stage="stage2_conch", hopt=hopt,
                              extra=s2_extra(tmp_path, out1 / "RECORD.json")), capsys)
    rec2 = json.loads((out2 / "RECORD.json").read_text())
    assert rec2["attempt"] == 2 and rec2["supersedes"] is None
    assert rec2["prior_attempts"] == [{"path": str(out1.resolve()), "stage": "stage1_512px"}]
    assert rec2["stage1_record"]["complete_sha256"] == sha_file(out1 / "COMPLETE")
    assert rec2["stage1_record"]["record_sha256"] == sha_file(out1 / "RECORD.json")
    c2 = json.loads((out2 / "counts.json").read_text())
    assert c2["stage_totals"]["primary_intervals_nominal"] == 48
    cum = c2["cumulative_totals"]
    assert cum["primary_intervals_nominal"] == 96 and cum["stages"] == ["stage1_512px",
                                                                         "stage2_conch"]
    assert cum["primary_intervals_computed"] == c1["stage_totals"]["primary_intervals_computed"]
    assert cum["patterns"]["unavailable"] == c1["stage_totals"]["patterns"]["unavailable"] + 24
    pr = read_csv(out2 / "patterns.csv")
    assert {(r["comparator"], r["pattern"], r["pattern_reason"]) for r in pr} == {
        ("conch_sampler", "unavailable", "dropped_n4"), ("conch_titan", "unavailable", "nonreturn")}
    assert read_csv(out2 / "d_contrasts.csv") == []
    assert np.load(out2 / "d_replicates.npy").shape == (0, COUNT)
    # a stage-1 record whose counts no longer match its COMPLETE is refused
    saved = (out1 / "counts.json").read_bytes()
    (out1 / "counts.json").write_text("{}\n")
    try:
        expect_failed(cli_argv(w, p2 / "n3_run_stage2b", stage="stage2_conch", hopt=hopt,
                               extra=s2_extra(tmp_path, out1 / "RECORD.json",
                                              ["--supersedes", str(out2), "--defect", "x"])),
                      "n3_02.stage1_complete_chain", capsys)
    finally:
        (out1 / "counts.json").write_bytes(saved)


def test_stage1_record_location_and_latest(world_ok, real_stage, tmp_path, capsys):
    """N3I-05: P2ROOT/n3_run_* only, never .partial, latest non-superseded stage-1 attempt."""
    w = world_ok
    p2 = real_stage(w)
    hopt = p2 / "run_001"
    gate = b4_gate(tmp_path)
    uni = make_comparator(w, tmp_path / "c", "uni2_sampler", "C")
    s1a, _ = run_ok(cli_argv(w, p2 / "n3_run_s1a", [uni], stage="stage1_512px", hopt=hopt,
                             extra=s1_extra(tmp_path, gate)), capsys)
    prev = []

    def s2(name, record):
        argv = cli_argv(w, p2 / name, stage="stage2_conch", hopt=hopt,
                        extra=s2_extra(tmp_path, record, prev))
        prev[:] = ["--supersedes", str(p2 / name), "--defect", "fixture refusal"]
        return argv
    ext = tmp_path / "elsewhere" / "n3_run_s1a"
    shutil.copytree(s1a, ext)
    expect_failed(s2("n3_run_s2a", ext / "RECORD.json"), "n3_02.stage1_record_location", capsys)
    misc = p2 / "misc_s1a"
    shutil.copytree(s1a, misc)
    expect_failed(s2("n3_run_s2b", misc / "RECORD.json"), "n3_02.stage1_record_location",
                  capsys)
    # a superseding stage-1 attempt makes s1a stale
    s1b, _ = run_ok(cli_argv(w, p2 / "n3_run_s1b", [uni], stage="stage1_512px", hopt=hopt,
                             extra=s1_extra(tmp_path, gate, ["--supersedes", str(s1a),
                                                             "--defect", "fixture rerun"])),
                    capsys)
    f = expect_failed(s2("n3_run_s2c", s1a / "RECORD.json"), "n3_02.stage1_record_not_latest",
                      capsys)
    s1b_attempt = json.loads((s1b / "RECORD.json").read_text())["attempt"]
    assert f["context"] == {"file": str(s1b.resolve()), "count": s1b_attempt,
                            "state": "complete"}
    ok, _ = run_ok(s2("n3_run_s2d", s1b / "RECORD.json"), capsys)
    rec = json.loads((ok / "RECORD.json").read_text())
    assert rec["stage1_record"]["path"] == str((s1b / "RECORD.json").resolve())
    assert rec["stage1_record"]["complete_sha256"] == sha_file(s1b / "COMPLETE")
    # a fully written COMPLETE inside a crashed .partial is never a stage-1 record
    crashed = p2 / "n3_run_s1c.partial"
    shutil.copytree(s1b, crashed)
    expect_failed(s2("n3_run_s2e", crashed / "RECORD.json"), "n3_02.stage1_record_location",
                  capsys)


def test_cross_stage_reopen_requires_supersedes(world_ok, real_stage, tmp_path, capsys):
    """N3I-02(b), probe 1: single after stage1 (and stage1/stage2 after single)."""
    w = world_ok
    p2 = real_stage(w)
    hopt = p2 / "run_001"
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    s1, _ = run_ok(cli_argv(w, p2 / "n3_run_s1", [uni], stage="stage1_512px", hopt=hopt,
                            extra=s1_extra(tmp_path, b4_gate(tmp_path))), capsys)
    uni_b = make_comparator(w, tmp_path / "b", "uni2_sampler", "H")     # different values
    f = expect_failed(cli_argv(w, p2 / "n3_run_single", [uni_b], stage="single", hopt=hopt,
                               extra=single_extra(tmp_path)), "n3_14.supersedes_required",
                      capsys)
    assert f["stage"] == "single" and f["context"]["count"] == 1
    expect_failed(cli_argv(w, p2 / "n3_run_single2", [uni_b], stage="single", hopt=hopt,
                           extra=single_extra(tmp_path, ["--supersedes", str(s1)])),
                  "n3_14.defect_required", capsys)
    sg, _ = run_ok(cli_argv(w, p2 / "n3_run_single3", [uni_b], stage="single", hopt=hopt,
                            extra=single_extra(tmp_path, ["--supersedes", str(s1),
                                                          "--defect", "fixture reopen"])),
                   capsys)
    rec = json.loads((sg / "RECORD.json").read_text())
    assert rec["attempt"] == 4 and rec["supersedes"] == [str(s1)]
    assert sorted(e["stage"] for e in rec["prior_attempts"]) == ["single", "single",
                                                                  "stage1_512px"]
    # reverse direction: a single attempt now exists
    expect_failed(cli_argv(w, p2 / "n3_run_s1x", [uni], stage="stage1_512px", hopt=hopt,
                           extra=s1_extra(tmp_path, b4_gate(tmp_path))),
                  "n3_14.supersedes_required", capsys)
    f = expect_failed(cli_argv(w, p2 / "n3_run_s2x", stage="stage2_conch", hopt=hopt,
                               extra=s2_extra(tmp_path, s1 / "RECORD.json")),
                      "n3_14.supersedes_required", capsys)
    assert f["context"]["count"] == 3      # three single attempts; stage-1 entries do not conflict


def test_real_stage_out_outside_p2root_refused(world_ok, real_stage, tmp_path, capsys):
    """N3I-02(a), probe 2: only --stage replay may write outside P2ROOT."""
    w = world_ok
    p2 = real_stage(w)
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    other = tmp_path / "other"
    other.mkdir()
    expect_failed(cli_argv(w, other / "n3_run_s1", [uni], stage="stage1_512px",
                           hopt=p2 / "run_001", extra=s1_extra(tmp_path, b4_gate(tmp_path))),
                  "n3_14.out_parent_not_p2root", capsys)
    assert not list(p2.glob("n3_run_*"))
    expect_failed(cli_argv(w, p2 / "plain", [uni], stage="stage1_512px",
                           hopt=p2 / "run_001", extra=s1_extra(tmp_path, b4_gate(tmp_path))),
                  "n3_14.real_stage_out_must_be_n3_run", capsys)


def test_partial_and_unparsable_entries_count_as_attempts(world_ok, real_stage, tmp_path,
                                                         capsys):
    """N3I-02(c)/N3I-06, probe 3: .partial, unparsable, stage-less and non-directory
    n3_run_* entries are attempts of unknown stage and conflict with every stage."""
    w = world_ok
    p2 = real_stage(w)
    hopt = p2 / "run_001"
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    gate = b4_gate(tmp_path)
    crashed = p2 / "n3_run_crash.partial"
    crashed.mkdir()
    f = expect_failed(cli_argv(w, p2 / "n3_run_a", [uni], stage="stage1_512px", hopt=hopt,
                               extra=s1_extra(tmp_path, gate)), "n3_14.supersedes_required",
                      capsys)
    assert f["context"]["count"] == 1
    # amendment 004 item 2(iii): every un-superseded stage-1/unknown entry must be named
    f = expect_failed(cli_argv(w, p2 / "n3_run_a2", [uni], stage="stage1_512px", hopt=hopt,
                               extra=s1_extra(tmp_path, gate, ["--supersedes", str(crashed),
                                                               "--defect", "crashed attempt"])),
                      "n3_14.supersedes_incomplete", capsys)
    assert f["context"] == {"entries": [str((p2 / "n3_run_a").resolve())], "count": 1}
    assert f["supersedes_resolved"] is None and f["supersedes_ignored"] == [str(crashed)]
    b, _ = run_ok(cli_argv(w, p2 / "n3_run_b", [uni], stage="stage1_512px", hopt=hopt,
                           extra=s1_extra(tmp_path, gate, [
                               "--supersedes", str(crashed), "--supersedes", str(p2 / "n3_run_a"),
                               "--supersedes", str(p2 / "n3_run_a2"),
                               "--defect", "crashed attempt"])),
                  capsys)
    rec = json.loads((b / "RECORD.json").read_text())
    assert rec["attempt"] == 4
    assert {(Path(e["path"]).name, e["stage"]) for e in rec["prior_attempts"]} == {
        ("n3_run_a", "stage1_512px"), ("n3_run_a2", "stage1_512px"),
        ("n3_run_crash.partial", "unknown")}
    assert rec["supersedes_resolved"] == sorted(
        str(x.resolve()) for x in (crashed, p2 / "n3_run_a", p2 / "n3_run_a2"))
    (p2 / "n3_run_junk").mkdir()
    (p2 / "n3_run_junk" / "RECORD.json").write_text("{not json")
    (p2 / "n3_run_nostage").mkdir()
    write_json(p2 / "n3_run_nostage" / "FAILED.json", {"status": "FAILED"})
    (p2 / "n3_run_file").write_text("not a directory\n")
    (p2 / "n3_run_empty").mkdir()
    # stage 2 after stage 1 would not conflict; the five unknown entries do
    f = expect_failed(cli_argv(w, p2 / "n3_run_s2", stage="stage2_conch", hopt=hopt,
                               extra=s2_extra(tmp_path, b / "RECORD.json")),
                      "n3_14.supersedes_required", capsys)
    assert f["context"]["count"] == 5


def test_failure_handler_fallback_and_interrupt(world_ok, real_stage, tmp_path, capsys,
                                                monkeypatch):
    """N3I-06: FAILED.json is attempted even when the partial cannot be removed (the leftover
    partial then counts as an attempt) and on KeyboardInterrupt (rc 130)."""
    w = world_ok
    p2 = real_stage(w)
    hopt = p2 / "run_001"
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    gate = b4_gate(tmp_path)

    def boom(*a, **k):
        raise ValueError("leak 0.123456")
    out = p2 / "n3_run_a"
    with monkeypatch.context() as m:
        m.setattr(cli, "compose", boom)
        m.setattr(cli.shutil, "rmtree", lambda *a, **k: None)
        rc = cli.main(cli_argv(w, out, [uni], stage="stage1_512px", hopt=hopt,
                               extra=s1_extra(tmp_path, gate)))
    cap = capsys.readouterr()
    assert rc == 1 and cap.err.strip() == f"FAILED: {out} check=unexpected"
    assert sorted(p.name for p in out.iterdir()) == ["FAILED.json"]
    failed = json.loads((out / "FAILED.json").read_text())
    assert failed["check"] == "unexpected" and failed["stage"] == "stage1_512px"
    assert failed["attempt"] == 1 and failed["partial_left"] is True       # amendment 004 item 7
    assert "0.123456" not in (out / "FAILED.json").read_text()
    assert Path(str(out) + ".partial").is_dir()
    f = expect_failed(cli_argv(w, p2 / "n3_run_b", [uni], stage="stage1_512px", hopt=hopt,
                               extra=s1_extra(tmp_path, gate)), "n3_14.supersedes_required",
                      capsys)
    assert f["context"]["count"] == 2

    def interrupt(*a, **k):
        raise KeyboardInterrupt
    syn = make_comparator(w, tmp_path / "s", "synthetic_c", "C")
    for target, name in (("compose", "o_mid"), ("run", "o_run")):
        o = tmp_path / name
        with monkeypatch.context() as m:
            m.setattr(cli, target, interrupt)
            rc = cli.main(cli_argv(w, o, [syn], hopt=hopt))
        cap = capsys.readouterr()
        assert rc == 130 and cap.err.strip() == f"FAILED: {o} check=interrupted"
        assert sorted(p.name for p in o.iterdir()) == ["FAILED.json"]
        assert not Path(str(o) + ".partial").exists()
        failed = json.loads((o / "FAILED.json").read_text())
        assert failed["check"] == "interrupted" and failed["partial_left"] is False
        assert failed["exception_class"] == "KeyboardInterrupt"


@pytest.mark.parametrize("case,check", [
    ("malformed", "n3_02.b4_gate_malformed"),
    ("doc_missing", "n3_02.b4_gate_document_missing"),
    ("doc_sha", "n3_02.b4_gate_document_sha_differs"),
])
def test_b4_gate_documents_verified(world_ok, real_stage, tmp_path, capsys, case, check):
    """N3I-03: the gate binds two existing documents by sha."""
    w = world_ok
    p2 = real_stage(w)
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    if case == "malformed":
        gate = b4_gate(tmp_path, conch_inclusion_rule_path=None)
    elif case == "doc_missing":
        gate = b4_gate(tmp_path, adapter_contract_review_path=str(tmp_path / "absent.md"))
    else:
        gate = b4_gate(tmp_path, conch_inclusion_rule_sha256="c" * 64)
    expect_failed(cli_argv(w, p2 / "n3_run_a", [uni], stage="stage1_512px",
                           hopt=p2 / "run_001", extra=s1_extra(tmp_path, gate)), check, capsys)


def test_dropped_record_pins(world_ok, patch_pop, tmp_path, capsys):
    """N3I-04: every --dropped record needs a matching --dropped-sha256 pin."""
    w = world_ok
    patch_pop(w)
    p = _dropped(tmp_path, "synthetic_a", "dropped_n4")
    expect_failed(cli_argv(w, tmp_path / "o1", extra=["--dropped", f"synthetic_a={p}"]),
                  "dropped.record_pin_missing", capsys)
    expect_failed(cli_argv(w, tmp_path / "o2",
                           extra=dropped_args(tmp_path, "synthetic_a", "dropped_n4", "9" * 64)),
                  "dropped.record_sha256_differs_from_pin", capsys)
    expect_failed(cli_argv(w, tmp_path / "o3", extra=[
        *dropped_args(tmp_path, "synthetic_a", "dropped_n4"),
        "--dropped-sha256", f"synthetic_b={sha_file(p)}"]),
        "arguments.dropped_pin_for_undeclared_pipeline", capsys)
    out, _ = run_ok(cli_argv(w, tmp_path / "o4",
                             extra=dropped_args(tmp_path, "synthetic_a", "dropped_n4")), capsys)
    rec = json.loads((out / "RECORD.json").read_text())
    assert rec["comparators"]["synthetic_a"]["dropped_record_sha256"] == sha_file(p)


def test_real_stage_requires_run001_reference_and_spec(world_ok, patch_pop, real_stage,
                                                       tmp_path, capsys, monkeypatch):
    """N3I-07: non-replay stages require --hopt-run-record-sha256 == the run_001 constant;
    N3I-09 / amendment 004 item 1: non-replay stages require the SPEC file and its sha."""
    w = world_ok
    p2 = real_stage(w, spec=False)
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    gate = b4_gate(tmp_path)
    argv = cli_argv(w, p2 / "n3_run_a", [uni], stage="stage1_512px", hopt=p2 / "run_001",
                    extra=s1_extra(tmp_path, gate), spec="a" * 64)
    expect_refused(argv, "n3_09.spec_missing", capsys)
    monkeypatch.setattr(cli, "HOPT_RUN_RECORD_SHA256", "e" * 64)
    (p2 / "n3_spec_001").mkdir()
    (p2 / "n3_spec_001" / "SPEC.md").write_text("synthetic spec fixture\n")
    argv = cli_argv(w, p2 / "n3_run_b", [uni], stage="stage1_512px", hopt=p2 / "run_001",
                    extra=s1_extra(tmp_path, gate))
    expect_failed(argv, "n3_07.hopt_reference_not_run001", capsys)


def test_attempt_numbering_and_supersedes(world_ok, patch_pop, tmp_path, capsys):
    w = world_ok
    patch_pop(w)
    base = tmp_path / "attempts"
    base.mkdir()
    expect_failed(cli_argv(w, base / "n3_run_bad"), "n3_14.replay_must_not_write_n3_run", capsys)
    # amendment 004 item 1: a non-replay stage off the canonical P2ROOT is refused first
    expect_refused(cli_argv(w, base / "plain", stage="stage1_512px"),
                   "n3_14.p2root_not_canonical", capsys)
    expect_failed(cli_argv(w, base / "n3_replay_a", extra=["--supersedes", str(base / "x"),
                                                           "--defect", "d"]),
                  "n3_14.no_prior_attempt_to_supersede", capsys)
    shutil.rmtree(base / "n3_replay_a")
    a, _ = run_ok(cli_argv(w, base / "n3_replay_a"), capsys)
    assert json.loads((a / "RECORD.json").read_text())["attempt"] == 1
    fb = expect_failed(cli_argv(w, base / "n3_replay_b"), "n3_14.supersedes_required", capsys)
    assert fb["stage"] == "replay"
    expect_failed(cli_argv(w, base / "n3_replay_c", extra=["--supersedes", str(a)]),
                  "n3_14.defect_required", capsys)
    d, _ = run_ok(cli_argv(w, base / "n3_replay_d", extra=["--supersedes", str(a),
                                                           "--defect", "fixture defect"]), capsys)
    rec = json.loads((d / "RECORD.json").read_text())
    assert rec["attempt"] == 4 and len(rec["prior_attempts"]) == 3
    assert rec["supersedes"] == [str(a)] and rec["defect"] == "fixture defect"
    assert rec["supersedes_resolved"] == [str(a.resolve())]


# =============================================================================================
# Amendment 004 regressions (review 002: N3II-01..07; probes gap_b1/b4/b5/e/g adapted)
# =============================================================================================

def _rec(d: Path) -> dict:
    return json.loads((d / "RECORD.json").read_text())


def _s1_argv(w, p2, name, comp, base, more=()):
    return cli_argv(w, p2 / name, [comp], stage="stage1_512px", hopt=p2 / "run_001",
                    extra=s1_extra(base, b4_gate(base), list(more)))


def _s2_argv(w, p2, name, record, base, more=()):
    return cli_argv(w, p2 / name, stage="stage2_conch", hopt=p2 / "run_001",
                    extra=s2_extra(base, record, list(more)))


def test_noncanonical_p2root_refused(world_ok, real_stage, tmp_path, capsys):
    """Item 1 (N3II-01, probe gap_b1): a byte-identical copy of run_001 + SPEC elsewhere is not
    P2ROOT; every non-replay stage is refused before any output; replay stays unrestricted."""
    w = world_ok
    p2 = real_stage(w)
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    run_ok(_s1_argv(w, p2, "n3_run_s1", uni, tmp_path), capsys)
    fake = tmp_path / "fake_p2root"
    fake.mkdir()
    shutil.copytree(p2 / "run_001", fake / "run_001")
    shutil.copytree(p2 / "n3_spec_001", fake / "n3_spec_001")
    uni_b = make_comparator(w, tmp_path / "b", "uni2_sampler", "H")
    expect_refused(cli_argv(w, fake / "n3_run_single", [uni_b], stage="single",
                            hopt=fake / "run_001", extra=single_extra(tmp_path)),
                   "n3_14.p2root_not_canonical", capsys)
    expect_refused(_s1_argv(w, fake, "n3_run_s1", uni_b, tmp_path),
                   "n3_14.p2root_not_canonical", capsys)
    expect_refused(_s2_argv(w, fake, "n3_run_s2", p2 / "n3_run_s1" / "RECORD.json", tmp_path),
                   "n3_14.p2root_not_canonical", capsys)
    # --out inside the canonical root does not help when --hopt-dir is the copy
    expect_refused(cli_argv(w, p2 / "n3_run_single", [uni_b], stage="single",
                            hopt=fake / "run_001", extra=single_extra(tmp_path)),
                   "n3_14.p2root_not_canonical", capsys)
    assert sorted(x.name for x in p2.glob("n3_run_*")) == ["n3_run_s1"]
    out, _ = run_ok(cli_argv(w, tmp_path / "n3_replay_fake", hopt=fake / "run_001"), capsys)
    assert _rec(out)["stage"] == "replay"


@pytest.mark.parametrize("spec", [None, "0" * 64, "not-a-sha"])
def test_spec_sha256_required_in_nonreplay(world_ok, real_stage, tmp_path, capsys, spec):
    """Item 1: non-replay stages need --spec-sha256 equal to P2ROOT/n3_spec_001/SPEC.md."""
    w = world_ok
    p2 = real_stage(w)
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    expect_refused(cli_argv(w, p2 / "n3_run_a", [uni], stage="stage1_512px",
                            hopt=p2 / "run_001", extra=s1_extra(tmp_path, b4_gate(tmp_path)),
                            spec=spec), "n3_14.spec_sha_mismatch", capsys)


def test_spec_sha256_in_replay_is_verified_when_given(world_ok, patch_pop, tmp_path, capsys):
    """Ambiguity C3 (literal reading): replay does not require --spec-sha256, but a given value
    is verified like in the non-replay stages."""
    w = world_ok
    patch_pop(w)
    expect_refused(cli_argv(w, tmp_path / "n3_replay_a", spec="0" * 64),
                   "n3_09.spec_missing", capsys)
    p2 = make_p2root(w, tmp_path / "root")
    expect_refused(cli_argv(w, tmp_path / "n3_replay_b", hopt=p2 / "run_001", spec="0" * 64),
                   "n3_14.spec_sha_mismatch", capsys)
    good = sha_file(p2 / "n3_spec_001" / "SPEC.md")
    out, _ = run_ok(cli_argv(w, tmp_path / "n3_replay_c", hopt=p2 / "run_001", spec=good),
                    capsys)
    assert _rec(out)["spec_sha256"] == good


@pytest.mark.parametrize("env", [{"OMP_NUM_THREADS": "2"}, {"MKL_NUM_THREADS": None},
                                 {"OPENBLAS_NUM_THREADS": "01"}, {"OMP_NUM_THREADS": "1 "}])
def test_thread_env_must_be_one_in_nonreplay(world_ok, real_stage, tmp_path, capsys,
                                             monkeypatch, env):
    """Item 4 (N3II-04, probe gap_g): exactly "1" for all three variables, before any output."""
    w = world_ok
    p2 = real_stage(w)
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    expect_refused(_s1_argv(w, p2, "n3_run_a", uni, tmp_path), "n3_09.thread_env_not_1",
                   capsys)
    # replay is unrestricted and records the environment
    out, _ = run_ok(cli_argv(w, tmp_path / "n3_replay_t", hopt=p2 / "run_001"), capsys)
    assert _rec(out)["thread_env"] == {k: os.environ.get(k) for k in cli.THREAD_ENV}


@pytest.mark.parametrize("stage", ["replay", "stage1_512px"])
@pytest.mark.parametrize("defect", ["rerun 2", "delta was 0.1", "attempt two\u0663"])
def test_defect_with_digit_refused(world_ok, real_stage, tmp_path, capsys, stage, defect):
    """Item 5 (N3II-05/A17): any digit in --defect is refused before any output."""
    w = world_ok
    p2 = real_stage(w)
    if stage == "replay":
        argv = cli_argv(w, tmp_path / "n3_replay_a", hopt=p2 / "run_001",
                        extra=["--defect", defect])
    else:
        uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
        argv = _s1_argv(w, p2, "n3_run_a", uni, tmp_path, ["--defect", defect])
    expect_refused(argv, "n3_14.defect_contains_number", capsys)


def test_stage1_supersedes_all_and_refused_runs_cannot_choose(world_ok, real_stage, tmp_path,
                                                              capsys):
    """Item 2 (N3II-02, probe gap_b4): (iii) a new stage-1 run must name every un-superseded
    stage-1/unknown entry; (i) a run refused before derive_attempt supersedes nothing; (ii) the
    latest stage-1 record is the highest valid attempt."""
    w = world_ok
    p2 = real_stage(w)
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    uni_b = make_comparator(w, tmp_path / "b", "uni2_sampler", "H")
    a, _ = run_ok(_s1_argv(w, p2, "n3_run_A", uni, tmp_path), capsys)
    junk = p2 / "n3_run_junk"
    junk.mkdir()
    f = expect_failed(_s1_argv(w, p2, "n3_run_B", uni_b, tmp_path,
                               ["--supersedes", str(junk), "--defect", "rerun"]),
                      "n3_14.supersedes_incomplete", capsys)
    assert f["context"] == {"entries": [str(a.resolve())], "count": 1}
    assert f["attempt"] is None and f["supersedes"] is None
    assert f["supersedes_resolved"] is None and f["supersedes_ignored"] == [str(junk)]
    # naming A and junk is still incomplete: the refused B is itself an un-superseded entry
    f = expect_failed(_s1_argv(w, p2, "n3_run_B2", uni_b, tmp_path,
                               ["--supersedes", str(junk), "--supersedes", str(a),
                                "--defect", "rerun"]),
                      "n3_14.supersedes_incomplete", capsys)
    assert f["context"]["entries"] == [str((p2 / "n3_run_B").resolve())]
    names = [junk, a, p2 / "n3_run_B", p2 / "n3_run_B2"]
    b3, _ = run_ok(_s1_argv(w, p2, "n3_run_B3", uni_b, tmp_path,
                            [x for n in names for x in ("--supersedes", str(n))]
                            + ["--defect", "rerun"]), capsys)
    rb = _rec(b3)
    assert rb["attempt"] == 5 and rb["supersedes_resolved"] == sorted(
        str(n.resolve()) for n in names)
    # the older A is no longer the latest stage-1 record
    f = expect_failed(_s2_argv(w, p2, "n3_run_s2a", a / "RECORD.json", tmp_path,
                               ["--supersedes", str(junk), "--defect", "x"]),
                      "n3_02.stage1_record_not_latest", capsys)
    assert f["context"] == {"file": str(b3.resolve()), "count": 5, "state": "complete"}
    # a refused stage-2 run naming B3 (not a conflicting prior) supersedes nothing
    g = expect_failed(cli_argv(w, p2 / "n3_run_G", stage="stage2_conch", hopt=p2 / "run_001",
                               extra=["--supersedes", str(b3), "--defect", "x"]),
                      "n3_14.supersedes_required", capsys)
    assert g["attempt"] is None and g["supersedes_resolved"] is None
    assert g["supersedes_ignored"] == [str(b3)] and g["partial_left"] is False
    f = expect_failed(_s2_argv(w, p2, "n3_run_s2b", a / "RECORD.json", tmp_path,
                               ["--supersedes", str(p2 / "n3_run_G"), "--defect", "x"]),
                      "n3_02.stage1_record_not_latest", capsys)
    assert f["context"]["file"] == str(b3.resolve())
    s2, _ = run_ok(_s2_argv(w, p2, "n3_run_s2", b3 / "RECORD.json", tmp_path,
                            ["--supersedes", str(p2 / "n3_run_G"), "--defect", "x"]), capsys)
    assert _rec(s2)["stage1_record"]["path"] == str((b3 / "RECORD.json").resolve())


def test_refused_replay_entry_is_unknown_and_inert(world_ok, real_stage, tmp_path, capsys):
    """Items 2(i)/3 (N3II-02/03, probe gap_b5): a refused replay written under P2ROOT/n3_run_*
    neither supersedes the sole stage-1 attempt nor is excluded; it is an unknown entry."""
    w = world_ok
    p2 = real_stage(w)
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    a, _ = run_ok(_s1_argv(w, p2, "n3_run_A", uni, tmp_path), capsys)
    x = expect_failed(cli_argv(w, p2 / "n3_run_x", hopt=p2 / "run_001",
                               extra=["--supersedes", str(a), "--defect", "x"]),
                      "n3_14.replay_must_not_write_n3_run", capsys)
    assert x["stage"] == "replay" and x["supersedes_resolved"] is None
    assert x["supersedes_ignored"] == [str(a)]
    # a COMPLETE replay copied under P2ROOT (valid attempt number, forged supersession) is
    # also only an unknown entry
    rp, _ = run_ok(cli_argv(w, tmp_path / "n3_replay_rp", hopt=p2 / "run_001"), capsys)
    shutil.copytree(rp, p2 / "n3_run_rp")
    r = _rec(p2 / "n3_run_rp")
    r["supersedes_resolved"] = [str(a.resolve())]
    write_json(p2 / "n3_run_rp" / "RECORD.json", r)
    f = expect_failed(_s2_argv(w, p2, "n3_run_s2a", a / "RECORD.json", tmp_path),
                      "n3_14.supersedes_required", capsys)
    assert f["context"]["count"] == 2
    # a new stage-1 run must name both unknown entries too
    f = expect_failed(_s1_argv(w, p2, "n3_run_s1b", uni, tmp_path,
                               ["--supersedes", str(a), "--defect", "x"]),
                      "n3_14.supersedes_incomplete", capsys)
    assert f["context"]["entries"] == sorted(str((p2 / n).resolve())
                                             for n in ("n3_run_rp", "n3_run_x"))
    s2, _ = run_ok(_s2_argv(w, p2, "n3_run_s2", a / "RECORD.json", tmp_path,
                            ["--supersedes", str(p2 / "n3_run_x"), "--defect", "x"]), capsys)
    rec = _rec(s2)
    assert rec["stage1_record"]["path"] == str((a / "RECORD.json").resolve())
    stages = {Path(e["path"]).name: e["stage"] for e in rec["prior_attempts"]}
    assert stages["n3_run_x"] == "unknown" and stages["n3_run_rp"] == "unknown"
    assert stages["n3_run_A"] == "stage1_512px"


def test_stage1_latest_state_failed_superseded_ambiguous(world_ok, real_stage, tmp_path,
                                                         capsys, monkeypatch):
    """Item 2(ii): the highest valid stage-1 attempt must be COMPLETE, not superseded by a
    valid entry and unique; otherwise every stage-1 record is refused with its state."""
    w = world_ok
    p2 = real_stage(w)
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    a, _ = run_ok(_s1_argv(w, p2, "n3_run_A", uni, tmp_path), capsys)

    def boom(*args, **kw):
        raise ValueError("synthetic failure")
    with monkeypatch.context() as m:
        m.setattr(cli, "compose", boom)
        assert cli.main(_s1_argv(w, p2, "n3_run_B", uni, tmp_path,
                                 ["--supersedes", str(a), "--defect", "rerun"])) == 1
    capsys.readouterr()
    fb = json.loads((p2 / "n3_run_B" / "FAILED.json").read_text())
    assert fb["attempt"] == 2 and fb["supersedes_resolved"] == [str(a.resolve())]
    f = expect_failed(_s2_argv(w, p2, "n3_run_s2a", a / "RECORD.json", tmp_path),
                      "n3_02.stage1_record_not_latest", capsys)
    assert f["context"] == {"file": str((p2 / "n3_run_B").resolve()), "count": 2,
                            "state": "failed"}
    c, _ = run_ok(_s1_argv(w, p2, "n3_run_C", uni, tmp_path,
                           ["--supersedes", str(p2 / "n3_run_B"), "--defect", "rerun"]), capsys)
    assert _rec(c)["attempt"] == 4
    # an exact copy of C carries the same attempt number: the latest is ambiguous
    shutil.copytree(c, p2 / "n3_run_Cdup")
    f = expect_failed(_s2_argv(w, p2, "n3_run_s2b", c / "RECORD.json", tmp_path,
                               ["--supersedes", str(p2 / "n3_run_s2a"), "--defect", "x"]),
                      "n3_02.stage1_record_not_latest", capsys)
    assert f["context"] == {"file": str(c.resolve()), "count": 4, "state": "ambiguous"}
    shutil.rmtree(p2 / "n3_run_Cdup")
    # a valid single run superseding C leaves no usable stage-1 record
    uni_s = make_comparator(w, tmp_path / "s", "uni2_sampler", "H")
    sg, _ = run_ok(cli_argv(w, p2 / "n3_run_single", [uni_s], stage="single",
                            hopt=p2 / "run_001",
                            extra=single_extra(tmp_path, ["--supersedes", str(c),
                                                          "--defect", "reopen"])), capsys)
    f = expect_failed(_s2_argv(w, p2, "n3_run_s2c", c / "RECORD.json", tmp_path,
                               ["--supersedes", str(sg), "--defect", "x"]),
                      "n3_02.stage1_record_not_latest", capsys)
    assert f["context"] == {"file": str(c.resolve()), "count": 4, "state": "superseded"}


@pytest.mark.parametrize("variant", ["qm_sha", "pd_sha", "no_outputs"])
def test_hopt_output_sha_must_be_in_run_record(world_ok, real_stage, tmp_path, capsys,
                                               monkeypatch, variant):
    """Item 6 (N3II-06): in non-replay stages the pinned query_metrics/paired_deltas must be
    the outputs recorded in the pinned run_record; replay reference is not subject to it."""
    w = world_ok
    p2 = real_stage(w)
    rr = p2 / "run_001" / "run_record.json"
    rec = json.loads(rr.read_text())
    fname = "paired_deltas.csv" if variant == "pd_sha" else "query_metrics.csv.gz"
    if variant == "no_outputs":
        del rec["outputs"]
    else:
        rec["outputs"][fname]["sha256"] = "d" * 64
    write_json(rr, rec)
    monkeypatch.setattr(cli, "HOPT_RUN_RECORD_SHA256", sha_file(rr))
    uni = make_comparator(w, tmp_path / "a", "uni2_sampler", "C")
    f = expect_failed(_s1_argv(w, p2, "n3_run_a", uni, tmp_path),
                      "n3_07.hopt_output_sha_not_in_run_record", capsys)
    assert f["context"]["file"] == fname
    run_ok(cli_argv(w, tmp_path / "n3_replay_a", hopt=p2 / "run_001"), capsys)


@pytest.mark.parametrize("variant", ["query_metrics", "paired_deltas"])
def test_constant_hoptimus_1_requires_recorded_outputs(world_ok, patch_pop, tmp_path, capsys,
                                                       monkeypatch, variant):
    """Item 6 (N3II-07, probe gap_e): in replay a directory whose run_record is byte-identical
    to the pinned one is hoptimus_1 only if the files read are that run_record's outputs."""
    w = world_ok
    patch_pop(w)
    monkeypatch.setattr(cli, "HOPT_RUN_RECORD_SHA256", sha_file(w.hopt_dir / "run_record.json"))
    if variant == "query_metrics":
        h = make_comparator(w, tmp_path, "hoptimus_1", "C", pipeline_id=None)
        shutil.copyfile(w.hopt_dir / "run_record.json", h["dir"] / "run_record.json")
        d = h["dir"]
    else:
        d = tmp_path / "comp_hoptimus_1"
        shutil.copytree(w.hopt_dir, d)
        lines = (d / "paired_deltas.csv").read_text().splitlines(keepends=True)
        (d / "paired_deltas.csv").write_text("".join([lines[0], *reversed(lines[1:])]))
    write_complete(d)
    h = make_acceptance(tmp_path, "hoptimus_1", d)
    f = expect_failed(cli_argv(w, tmp_path / "o", [h]), "n3_07.hopt_output_sha_not_in_run_record",
                      capsys)
    want = "query_metrics.csv.gz" if variant == "query_metrics" else "paired_deltas.csv"
    assert f["context"] == {"pipeline_id": "hoptimus_1", "file": want}


# =============================================================================================
# Determinism, permutation invariance, kernel boundary (§6.5/6.7) and blinding (§6.6)
# =============================================================================================

def test_outputs_invariant_to_row_and_argument_order(world_ok, patch_pop, tmp_path, capsys):
    w = world_ok
    patch_pop(w)
    a = make_comparator(w, tmp_path / "x", "synthetic_a", "C")
    b = make_comparator(w, tmp_path / "x", "synthetic_b", "H")
    rng = np.random.default_rng(5)
    a2 = make_comparator(w, tmp_path / "y", "synthetic_a", "C",
                         qm_hook=lambda rows: [rows[i] for i in rng.permutation(len(rows))])
    b2 = make_comparator(w, tmp_path / "y", "synthetic_b", "H")
    o1, _ = run_ok(cli_argv(w, tmp_path / "o1", [a, b]), capsys)
    o2, _ = run_ok(cli_argv(w, tmp_path / "o2", [b2, a2]), capsys)
    for f in OUTPUT_FILES:
        assert (o1 / f).read_bytes() == (o2 / f).read_bytes(), f


def test_kernel_boundary(world_ok, patch_pop, tmp_path, capsys, monkeypatch):
    w = world_ok
    patch_pop(w)
    k = cli.kernel
    orig = k.summarize_paired
    calls = []

    def guarded(q, p, g, groups, va, vb, inc, draws):
        assert len(q) == len(p) == len(g) == len(va) == len(vb) == len(inc) == w.n
        assert tuple(q) == tuple(w.qids) and tuple(groups) == w.groups
        for v in (va, vb):
            v = np.asarray(v)
            assert v.dtype == np.float64 and np.all((v >= 0) & (v <= 1))
        assert isinstance(inc, np.ndarray) and inc.dtype == bool and not inc.flags.writeable
        calls.append((id(inc), id(draws), int(inc.sum())))
        return orig(q, p, g, groups, va, vb, inc, draws)
    monkeypatch.setattr(k, "summarize_paired", guarded)
    snapshot = dict(k.__dict__)
    c = make_comparator(w, tmp_path, "synthetic_c", "C")
    out, _ = run_ok(cli_argv(w, tmp_path / "out", [c]), capsys)
    assert dict(k.__dict__) == snapshot and all(k.__dict__[n] is v for n, v in snapshot.items())
    inc_to_draws = {}
    for inc, dr, _ in calls:
        inc_to_draws.setdefault(inc, set()).add(dr)
    assert all(len(v) == 1 for v in inc_to_draws.values())
    assert len(inc_to_draws) == 1 + 12 + 2          # full + strata + two null-excluded
    verify_against_reference(w, out, [("synthetic_c", "C")])


def test_cli_uses_only_public_kernel_api():
    tree = ast.parse(CLI_PATH.read_text())
    used = {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "kernel"}
    assert used <= {"summarize_paired", "make_patient_draws", "ValidationError",
                    "ComputationError", "__file__"}, used
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and "r2_1_hoptimus_main" in n.module:
            raise AssertionError("kernel names must be accessed through the module object")
    src = CLI_PATH.read_text()
    assert "tempfile" not in src and "/tmp" not in src


def _subprocess_main(tmp_path: Path, argv, patch):
    cfg = tmp_path / "subprocess_cfg.json"
    cfg.write_text(json.dumps({"argv": [str(a) for a in argv], "patch": patch}))
    code = ("import json, sys\n"
            "from prosper.evaluation import r2_1_perturb_contrast_cli as cli\n"
            "cfg = json.load(open(sys.argv[1]))\n"
            "[setattr(cli, k, v) for k, v in cfg['patch'].items()]\n"
            "sys.exit(cli.main(cfg['argv']))\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", code, str(cfg)], cwd=CODE_ROOT, env=env,
                          capture_output=True, text=True, timeout=900)


def test_blinding_stdout_stderr_failed_json(world_ok, tmp_path):
    """§6.6 + N3-03: real process streams never carry a float token (D/Δ/CI/excess)."""
    w = world_ok
    patch = {"POPULATION_SHA256": w.queries_sha, "POPULATION_N_SLIDES": w.n,
             "POPULATION_N_PATIENTS": w.n_patients}
    c = make_comparator(w, tmp_path / "ok", "synthetic_c", "C")
    argv = cli_argv(w, tmp_path / "out_ok", [c])
    p = _subprocess_main(tmp_path, argv, patch)
    assert p.returncode == 0, p.stderr
    allow = [tmp_path / "out_ok", *argv]
    assert p.stderr == "" and float_tokens(p.stdout, allow) == []

    def hook(rows):
        for r in rows:
            if r[0] == "rotate_90" and r[1] == "MACRO" and r[2] == "hit" and r[3] == 5:
                r[4] = r[4] + 0.0123
    bad = make_comparator(w, tmp_path / "bad", "synthetic_c", "C", pd_hook=hook)
    argv = cli_argv(w, tmp_path / "out_bad", [bad])
    p = _subprocess_main(tmp_path, argv, patch)
    assert p.returncode == 1
    allow = [tmp_path / "out_bad", *argv]
    text = (tmp_path / "out_bad" / "FAILED.json").read_text()
    for s in (p.stdout, p.stderr, text):
        assert float_tokens(s, allow) == [], s
    assert not json_has_float(json.loads(text))
    assert sorted(x.name for x in (tmp_path / "out_bad").iterdir()) == ["FAILED.json"]
    assert "Traceback" not in p.stderr


# =============================================================================================
# REAL-gated (§6.1 replay, §6.2 self-contrast via N3-01 recognition, §6.8/N3-15 synthetic shift)
# =============================================================================================

REAL = os.environ.get("R21_N3_REAL") == "1"
P2ROOT_ENV = "r2_1_p2_eval_20261006"


def _real_paths():
    out = Path(os.environ["PROSPER_OUT"])
    p2 = out / P2ROOT_ENV
    return dict(queries=out / "r2_1_cpu_20261002" / "perturb_attempt_001" / "queries.csv",
                hopt=p2 / "run_001", dose=p2 / "dose_summary_001",
                sens=p2 / "sensitivity_brightness_001")


def build_real_world(paths) -> World:
    """World from run_001 metadata only (queries ID columns, query_metrics, strata, nulls)."""
    with open(paths["queries"], newline="", encoding="utf-8") as fh:
        q = sorted(({"i": int(r["population_index"]), "s": r["slide_id"], "p": r["patient_id"],
                     "g": r["diagnostic_group"]} for r in csv.DictReader(fh)),
                   key=lambda r: r["i"])
    assert [r["i"] for r in q] == list(range(len(q)))
    rec = json.loads((paths["hopt"] / "run_record.json").read_text())
    w = World([r["s"] for r in q], [r["p"] for r in q], [r["g"] for r in q],
              rec["population"]["groups"], rec["conditions"])
    vals = {(c, m, k): np.full(w.n, np.nan) for c in w.conditions for m in ("hit", "precision")
            for k in KS}
    with gzip.open(paths["hopt"] / "query_metrics.csv.gz", "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            assert r["status"] == "returned"
            i, k = int(r["population_index"]), int(r["k"])
            vals[(r["condition"], "hit", k)][i] = float(r["conditional_hit"])
            vals[(r["condition"], "precision", k)][i] = float(r["conditional_precision"])
    assert all(np.isfinite(v).all() for v in vals.values())
    w.vals["H"] = vals
    pos = {s: i for i, s in enumerate(w.qids)}
    for arm in STRAT_ARMS:
        w.strata[arm] = [None] * w.n
    for r in read_csv(paths["dose"] / "realised_stimuli.csv"):
        w.strata[r["arm"]][pos[r["slide_id"]]] = r["stratum"]
    ns = read_csv(paths["sens"] / "null_slides.csv")
    w.null = {a: sorted(r["slide_id"] for r in ns if r["arm"] == a)
              for a in ("brightness", "spatial_shift")}
    w.add_populations()
    w.run_record = rec
    w.hopt_dir, w.dose_dir, w.sens_dir = paths["hopt"], paths["dose"], paths["sens"]
    w.queries = paths["queries"]
    w.queries_sha = sha_file(paths["queries"])
    w.n_patients = len(set(w.pids))
    return w


def add_synthetic_shift(w: World):
    """N3-15 recipe: known per-slide changes at k = 5 (hit and precision) in a disjoint subset
    of rotate_90 rows (S1) and identity rows (S2); every other value equals run_001."""
    s = {key: v.copy() for key, v in w.vals["H"].items()}
    idx = np.arange(w.n)
    h90, hid = s[("rotate_90", "hit", 5)], s[("identity", "hit", 5)]
    s1 = idx[(idx % 50 == 0) & (h90 == 1.0)]
    s2 = idx[(idx % 50 == 25) & (hid == 0.0)]
    assert len(s1) and len(s2) and not set(s1) & set(s2)
    s[("rotate_90", "hit", 5)][s1] = 0.0
    s[("rotate_90", "precision", 5)][s1] = 0.0
    s[("identity", "hit", 5)][s2] = 1.0
    s[("identity", "precision", 5)][s2] = 0.2
    w.vals["S"] = s
    return {"rule_S1": "rotate_90, k=5, population_index % 50 == 0 and run_001 hit@5 == 1: "
                       "hit@5 -> 0.0, precision@5 -> 0.0",
            "rule_S2": "identity, k=5, population_index % 50 == 25 and run_001 hit@5 == 0: "
                       "hit@5 -> 1.0, precision@5 -> 0.2",
            "S1_population_index": [int(i) for i in s1],
            "S2_population_index": [int(i) for i in s2],
            "S1_slide_id": [w.qids[i] for i in s1], "S2_slide_id": [w.qids[i] for i in s2]}


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    if not REAL:
        pytest.skip("set R21_N3_REAL=1 (and PROSPER_OUT) to run the run_001 replay")
    paths = _real_paths()
    w = build_real_world(paths)
    recipe = add_synthetic_shift(w)
    base = tmp_path_factory.mktemp("real_inputs")
    h = make_acceptance(base, "hoptimus_1", w.hopt_dir)
    s = make_comparator(w, base, "synthetic_shift", "S")
    recipe["synthetic_query_metrics_sha256"] = sha_file(s["dir"] / "query_metrics.csv.gz")
    recipe["synthetic_paired_deltas_sha256"] = sha_file(s["dir"] / "paired_deltas.csv")
    recipe["hoptimus_1_acceptance_path"] = str(h["acc"])
    recipe["hoptimus_1_acceptance_sha256"] = h["pin"]
    write_json(base / "synthetic_shift_recipe.json", recipe)
    out = tmp_path_factory.getbasetemp() / "n3_replay_real"
    argv = cli_argv(w, out, [h, s])
    so, se = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
        rc = cli.main(argv)
    return dict(w=w, out=out, rc=rc, argv=argv, recipe=recipe, base=base,
                stdout=so.getvalue(), stderr=se.getvalue())


def test_real_replay_self_contrast_and_synthetic_shift(real_run):
    w, out = real_run["w"], real_run["out"]
    assert real_run["rc"] == 0, (out / "FAILED.json").read_text() if (out / "FAILED.json").exists() else ""
    check_complete(out)
    rc = json.loads((out / "replay_check.json").read_text())["targets"]
    assert {k: v["rows_checked"] for k, v in rc.items()} == {
        "paired_deltas.csv": 1536, "strata_deltas.csv": 1536, "sensitivity_deltas.csv": 256}
    assert all(v["passed"] for v in rc.values())
    rec = json.loads((out / "RECORD.json").read_text())
    assert rec["comparators"]["hoptimus_1"]["pipeline_id_source"].startswith("constant")
    assert rec["comparators"]["hoptimus_1"]["check6"]["rows_checked"] == 1536
    assert rec["comparators"]["synthetic_shift"]["check6"]["rows_checked"] == 1536
    assert rec["population"]["n_slides"] == 2778 and rec["population"]["n_patients"] == 1876
    exp = verify_against_reference(w, out, [("hoptimus_1", "H"), ("synthetic_shift", "S")])
    rows = read_csv(out / "d_contrasts.csv")
    for r in rows:
        if r["comparator"] == "hoptimus_1" or r["k"] != "5":
            assert r["D"] in ("", "0.0") and r["zero_width_cause"] in (
                "no_change_either", "identical_nonzero_changes"), r
    macro = {(r["condition"], r["metric"]): r for r in rows
             if r["comparator"] == "synthetic_shift" and r["k"] == "5"
             and r["group"] == "MACRO" and r["population"] == "full"}
    assert all(macro[(c, m)]["D"] not in ("", "0.0") and macro[(c, m)]["zero_width_cause"] == ""
               for c in w.arms for m in ("hit", "precision"))
    # N3I-11: analytic sign/size anchor for all 24 primary rows.  S differs from run_001 only
    # at k = 5 on S1 (rotate_90) and S2 (identity) slides, so with every query returned
    # D(a) = Delta_H(a) - Delta_S(a) = mean_g[sum_{S2 in g} (id_S - id_H) / n_g]
    #        + [a == rotate_90] * mean_g[sum_{S1 in g} (rot90_H - rot90_S) / n_g].
    rec = real_run["recipe"]
    n_g = {g: sum(1 for x in w.labels if x == g) for g in w.groups}

    def macro_term(idx, change):
        return float(np.mean([sum(float(change[i]) for i in idx if w.labels[i] == g) / n_g[g]
                              for g in w.groups]))
    H = w.vals["H"]
    s1, s2 = rec["S1_population_index"], rec["S2_population_index"]
    exp_s2 = {"hit": macro_term(s2, np.ones(w.n)), "precision": macro_term(s2, np.full(w.n, 0.2))}
    exp_s1 = {"hit": macro_term(s1, H[("rotate_90", "hit", 5)]),
              "precision": macro_term(s1, H[("rotate_90", "precision", 5)])}
    assert exp_s2["hit"] > 0 and exp_s1["hit"] > 0
    assert len(macro) == 24
    for c in w.arms:
        for m in ("hit", "precision"):
            d = float(macro[(c, m)]["D"])
            assert d > 0, (c, m)
            expected = exp_s2[m] + (exp_s1[m] if c == "rotate_90" else 0.0)
            assert abs(d - expected) <= TOL, (c, m)
    rec_out = json.loads((out / "RECORD.json").read_text())
    assert rec_out["spec_sha256"] is not None and len(rec_out["spec_sha256"]) == 64
    assert sorted(rec_out["thread_env"]) == sorted(cli.THREAD_ENV)
    assert len(exp) == len(rows)
    allow = [out, *real_run["argv"]]
    assert real_run["stderr"] == "" and "COMPLETE: " in real_run["stdout"]
    assert float_tokens(real_run["stdout"], allow) == []
