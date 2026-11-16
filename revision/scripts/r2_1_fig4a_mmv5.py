#!/usr/bin/env python3
"""Paired majority-vote accuracy (mMV@5) deltas for the perturbation analysis (R2.1 / R2.7).

Authorized on 2026-10-21 (Fig. 4 session, ask_user "mmv_three"): a bounded CPU
derivation of delta mMV@5 from the retained ranked lists of the accepted perturbation runs,
so the perturbation figure can show hit@5, mMV@5 and precision@5 side by side. No embedding,
distance or ranking is recomputed; only a per-query 0/1 vote score is added and pushed through
the frozen paired whole-patient bootstrap kernel that produced the audited hit@5/precision@5
deltas.

Inputs (read-only, SHA-256 bound in input_bindings.json)
  H-Optimus-1 + SAMPLER  $PROSPER_OUT/r2_1_p2_eval_20261006/run_001/
  UNI2 + SAMPLER         $PROSPER_OUT/r2_1_comparators_20261008/eval_uni2/run_001/
  Virchow2 + SAMPLER     $PROSPER_OUT/r2_1_comparators_20261008/eval_virchow2/run_001/
    ranked_lists.csv.gz  condition, query, rank (1..10), neighbour, neighbour_group, relevant
    query_metrics.csv.gz query -> patient_id, diagnostic_group, status (k = 5 rows)
    patient_draws.npy    5,000 x P whole-patient multiplicities (PCG64 seed 123)
    paired_deltas.csv    audited hit/precision deltas (replication gate)
    code_frozen/r2_1_hoptimus_main.py  the bootstrap kernel (loaded from the frozen copy)

Vote rule: plurality of the five nearest other-patient slides' groups, ties broken toward the
closest-ranked tied label (the RR-5 plurality rule, `scripts/r2_3_internal_failure_tables.py::
plurality`); scored per query as 1 if the plurality label equals the query's group. Macro =
unweighted mean over the 15 groups (= macro recall = mMV@5 as defined for Fig. 2).

Gates (all must pass before COMPLETE is written)
  G1 coverage: every condition returned all 2,778 queries in all three runs.
  G2 draws: patient_draws.npy equals kernel.make_patient_draws(sorted patients) and matches the
     run record's SHA-256.
  G3 replication: hit@5 and precision@5 rebuilt from ranked_lists.csv.gz and pushed through the
     frozen kernel reproduce paired_deltas.csv (all groups + MACRO, 12 arms, point and CI) to
     1e-12 in all three runs.
  G4 rule identity: the plurality rule applied to identity lists gives the same per-query
     correctness as an independent Counter-based re-implementation.

Outputs  $PROSPER_OUT/r2_1_fig4a_mmv5_20261021/run_001/
  deltas.csv          pipeline, condition, group, metric (hit/precision/mmv), k=5, delta, ci,
                      identity/arm points, n_pairs, n_patients, status columns
  identity.csv        identity absolute values per pipeline/metric/group with CI
  fig_source.csv      MACRO rows in the fig4a_source_data layout (endpoint, panel, arm, point, ci)
  input_bindings.json, gates.json, SHA256SUMS.txt, run.log, COMPLETE
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

OUT_ROOT = Path(os.environ.get("PROSPER_OUT") or Path.cwd() / "prosper_revision_output")
RUNS = {
    "hopt": OUT_ROOT / "r2_1_p2_eval_20261006/run_001",
    "uni2_sampler": OUT_ROOT / "r2_1_comparators_20261008/eval_uni2/run_001",
    "virchow2_sampler": OUT_ROOT / "r2_1_comparators_20261008/eval_virchow2/run_001",
}
ARMS = ["rotate_90", "rotate_180", "rotate_270", "flip_horizontal", "flip_vertical",
        "brightness", "saturation", "contrast", "spatial_shift", "gaussian_blur",
        "gaussian_blur_image", "reinhard"]
K = 5
TOL = 1e-12
OUT = OUT_ROOT / "r2_1_fig4a_mmv5_20261021/run_001"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plurality(labels, k):
    """Most frequent label in labels[:k]; tie -> closest-ranked label among the tied ones."""
    c = Counter(labels[:k])
    mx = max(c.values())
    return next(l for l in labels[:k] if c[l] == mx)


def plurality_independent(labels, k):
    """Independent re-implementation for G4: count, then first rank reaching the max count."""
    top = list(labels[:k])
    counts = {l: top.count(l) for l in set(top)}
    best = max(counts.values())
    for l in top:
        if counts[l] == best:
            return l
    raise AssertionError


def load_kernel(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass decorators resolve the module via sys.modules
    spec.loader.exec_module(mod)
    return mod


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "w")

    def __call__(self, msg):
        line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "COMPLETE").exists():
        sys.exit(f"{OUT} already COMPLETE; refusing to overwrite an accepted run")
    log = Log(OUT / "run.log")
    log(f"start; PROSPER_OUT={OUT_ROOT}; python={sys.version.split()[0]}; numpy={np.__version__}")

    bindings, gates = {}, {"G1_coverage": {}, "G2_draws": {}, "G3_replication": {}, "G4_rule": {}}
    kernel_sha = None
    kernel = None
    delta_rows, ident_rows = [], []

    for pipe, run in RUNS.items():
        files = {n: run / n for n in ("ranked_lists.csv.gz", "query_metrics.csv.gz",
                                      "patient_draws.npy", "paired_deltas.csv", "coverage.csv",
                                      "run_record.json", "code_frozen/r2_1_hoptimus_main.py")}
        bindings[pipe] = {n: {"path": str(p), "sha256": sha256(p)} for n, p in files.items()}
        ksha = bindings[pipe]["code_frozen/r2_1_hoptimus_main.py"]["sha256"]
        if kernel is None:
            kernel = load_kernel(files["code_frozen/r2_1_hoptimus_main.py"])
            kernel_sha = ksha
        elif ksha != kernel_sha:
            sys.exit(f"frozen kernel differs between runs ({pipe})")
        log(f"{pipe}: inputs bound; kernel {ksha[:12]}")

        # population from the k = 5 identity rows of query_metrics
        qm = pd.read_csv(files["query_metrics.csv.gz"])
        pop = (qm[(qm.condition == "identity") & (qm.k == K)]
               .sort_values("population_index")[["query", "patient_id", "diagnostic_group"]]
               .reset_index(drop=True))
        pop["query"] = pop["query"].astype(str)
        assert pop["query"].is_unique and len(pop) == 2778
        qids = pop["query"].tolist()
        pids = pop["patient_id"].tolist()
        labels = pop["diagnostic_group"].tolist()
        groups = tuple(sorted(set(labels)))
        assert len(groups) == 15
        qpos = {q: i for i, q in enumerate(qids)}

        # G1 coverage
        cov = pd.read_csv(files["coverage.csv"])
        cov_all = cov[cov.group == "ALL"]
        conds = ["identity"] + ARMS
        cov_ok = (set(cov.condition) == set(conds)
                  and bool((cov.returned == cov.intended).all())
                  and len(cov_all) == len(conds) and int(cov_all.returned.min()) == 2778
                  and bool((qm[qm.k == K].status == "returned").all()))
        gates["G1_coverage"][pipe] = {"pass": cov_ok}
        if not cov_ok:
            sys.exit(f"G1 failed for {pipe}")

        # G2 draws
        draws_arr = np.load(files["patient_draws.npy"])
        rec = json.load(open(files["run_record.json"]))
        draws = kernel.make_patient_draws(sorted(set(pids)), count=5000, seed=123)
        same = np.array_equal(np.asarray(draws.multiplicities), draws_arr)
        rec_sha = json.dumps(rec).count(bindings[pipe]["patient_draws.npy"]["sha256"]) > 0
        gates["G2_draws"][pipe] = {"pass": bool(same), "shape": list(draws_arr.shape),
                                   "sha256_in_run_record": bool(rec_sha)}
        if not same:
            sys.exit(f"G2 failed for {pipe}")
        log(f"{pipe}: G1/G2 pass; draws {draws_arr.shape}")

        # per-query vectors from ranked lists
        rl = pd.read_csv(files["ranked_lists.csv.gz"], dtype={"query": str, "neighbour": str})
        rl = rl.sort_values(["condition", "query", "rank"], kind="stable")
        vec = {}
        n = len(qids)
        g4_mismatch = 0
        for c in conds:
            sub = rl[rl.condition == c]
            hit = np.zeros(n)
            prec = np.zeros(n)
            mmv = np.zeros(n)
            seen = np.zeros(n, dtype=bool)
            for q, grp in sub.groupby("query", sort=False):
                i = qpos[q]
                seen[i] = True
                ranks = grp["rank"].to_numpy()
                assert np.array_equal(ranks, np.arange(1, len(ranks) + 1)) and len(ranks) >= K
                nb = grp["neighbour_group"].tolist()
                rel = (np.asarray(nb[:K]) == labels[i])
                assert np.array_equal(rel.astype(int), grp["relevant"].to_numpy()[:K])
                hit[i] = float(rel.any())
                prec[i] = rel.sum() / K
                pl = plurality(nb, K)
                if c == "identity" and plurality_independent(nb, K) != pl:
                    g4_mismatch += 1
                mmv[i] = float(pl == labels[i])
            assert seen.all()
            vec[c] = {"hit": hit, "precision": prec, "mmv": mmv}
        gates["G4_rule"][pipe] = {"pass": g4_mismatch == 0, "mismatches": g4_mismatch}
        log(f"{pipe}: per-query vectors built for {len(conds)} conditions; G4 mismatches {g4_mismatch}")

        incl = np.ones(n, dtype=bool)
        # identity absolute values
        for m in ("hit", "precision", "mmv"):
            s = kernel.summarize_metric(qids, pids, labels, groups, vec["identity"][m], incl, draws)
            for g, e in zip((*groups, "MACRO"), (*s.group_estimates, s.macro)):
                ident_rows.append(dict(pipeline=pipe, condition="identity", group=g, metric=m, k=K,
                                       estimate=e.point, ci_low=e.ci_low, ci_high=e.ci_high,
                                       n_slides=e.n_queries, n_patients=e.n_patients,
                                       point_status=e.point_status, ci_status=e.ci_status))

        # paired deltas and G3 replication
        ref = pd.read_csv(files["paired_deltas.csv"])
        ref = ref[ref.k == K].set_index(["condition", "group", "metric"])
        g3 = {"n_checked": 0, "max_abs_point": 0.0, "max_abs_ci": 0.0, "fail": []}
        for c in ARMS:
            for m in ("hit", "precision", "mmv"):
                ps = kernel.summarize_paired(qids, pids, labels, groups, vec[c][m],
                                             vec["identity"][m], incl, draws)
                for g, e, a, b in zip((*groups, "MACRO"),
                                      (*ps.difference.group_estimates, ps.difference.macro),
                                      (*ps.arm_a.group_estimates, ps.arm_a.macro),
                                      (*ps.arm_b.group_estimates, ps.arm_b.macro)):
                    delta_rows.append(dict(pipeline=pipe, condition=c, group=g, metric=m, k=K,
                                           delta=e.point, ci_low=e.ci_low, ci_high=e.ci_high,
                                           n_pairs=e.n_queries, n_patients=e.n_patients,
                                           estimate_condition=a.point, estimate_identity=b.point,
                                           point_status=e.point_status, ci_status=e.ci_status,
                                           zero_support_replicates=e.zero_support_replicates,
                                           excludes_zero=(e.ci_low > 0 or e.ci_high < 0)
                                           if e.ci_status == "estimable" else None))
                    if m in ("hit", "precision"):
                        r = ref.loc[(c, g, m)]
                        dp = abs(r.delta - e.point)
                        dc = max(abs(r.ci_low - e.ci_low), abs(r.ci_high - e.ci_high))
                        g3["n_checked"] += 1
                        g3["max_abs_point"] = max(g3["max_abs_point"], dp)
                        g3["max_abs_ci"] = max(g3["max_abs_ci"], dc)
                        if dp > TOL or dc > TOL or int(r.n_pairs) != e.n_queries:
                            g3["fail"].append([c, g, m, float(dp), float(dc)])
        g3["pass"] = not g3["fail"] and g3["n_checked"] == 12 * 16 * 2
        gates["G3_replication"][pipe] = g3
        log(f"{pipe}: G3 checked {g3['n_checked']} cells; max |dpoint| {g3['max_abs_point']:.2e}, "
            f"max |dCI| {g3['max_abs_ci']:.2e}; pass={g3['pass']}")
        if not g3["pass"]:
            sys.exit(f"G3 failed for {pipe}")

    deltas = pd.DataFrame(delta_rows)
    ident = pd.DataFrame(ident_rows)
    deltas.to_csv(OUT / "deltas.csv", index=False)
    ident.to_csv(OUT / "identity.csv", index=False)

    # figure source rows: MACRO deltas in the v002 fig4a_source_data layout
    ep = {"hit": "hit@5", "precision": "precision@5", "mmv": "mMV@5"}
    src = deltas[deltas.group == "MACRO"].copy()
    src["endpoint"] = src.metric.map(ep)
    src["panel"] = "delta:" + src.pipeline
    src = src.rename(columns={"condition": "arm", "delta": "point"})
    src = src[["endpoint", "panel", "arm", "point", "ci_low", "ci_high", "excludes_zero",
               "estimate_identity", "estimate_condition", "n_pairs", "n_patients"]]
    src.to_csv(OUT / "fig_source.csv", index=False)

    all_pass = all(v["pass"] for gate in gates.values() for v in gate.values())
    json.dump({"kernel_sha256": kernel_sha, "runs": bindings,
               "vote_rule": "plurality of top-5 other-patient neighbour groups; ties to closest rank",
               "bootstrap": "frozen kernel summarize_paired; 5,000 whole-patient draws, PCG64 seed 123; "
                            "95 % linear percentile intervals"},
              open(OUT / "input_bindings.json", "w"), indent=1)
    json.dump({"all_pass": all_pass, "tolerance": TOL, **gates}, open(OUT / "gates.json", "w"),
              indent=1)
    with open(OUT / "SHA256SUMS.txt", "w") as f:
        for name in ("deltas.csv", "identity.csv", "fig_source.csv", "input_bindings.json",
                     "gates.json"):
            f.write(f"{sha256(OUT / name)}  {name}\n")
        f.write(f"{sha256(Path(__file__))}  script:{Path(__file__).name}\n")
    macro = src[src.endpoint == "mMV@5"]
    for _, r in macro.iterrows():
        log(f"mMV@5 {r.panel:24s} {r.arm:20s} {r.point:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
            f"{' *' if r.excludes_zero else ''}")
    if all_pass:
        (OUT / "COMPLETE").write_text(dt.datetime.now().isoformat() + "\n")
        log("all gates pass; COMPLETE written")
    else:
        log("GATE FAILURE; COMPLETE not written")
        sys.exit(1)


if __name__ == "__main__":
    main()
