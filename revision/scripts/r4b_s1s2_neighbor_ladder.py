"""Figs. S1/S2 re-derivation (D1/D5, 2026-10-21): per-group hit@K ladder and top-K
neighbour-label diversity for the main configuration (UNI2, OT cosine), same-patient slides
excluded from the library, 2,778 all-tissue queries from 1,876 patients, 15 groups.

Replaces the submitted Virchow2/SAMPLER "shortlist F1" analysis (Figs. S1, S2; same-patient
slides in the library) whose citation was dropped (R4.b-T3) or became inconsistent (l.59).

Design
- Ranking: the frozen unmasked UNI2/OT distance matrix is masked per query (self and every
  slide of the query's patient -> NaN) and ranked with the stable mergesort used by the as-run
  slide_eval ranking, to depth 21 (K up to 20 plus one tie-census rank).
- Gates (any failure -> FAILED.json, no COMPLETE):
  G1 population 2,778 x 2,778, 15 labels; G2 the derived depth-10 neighbour ids and labels
  equal the frozen c2_metric_suite masked lists for every query and rank (distinct masking
  route); G3 per-query hit@{1,3,5,10} equal the accepted r4a_vii c1e_masked frame; G4 the
  whole-population hit@K intervals (same patient draws, B = 5,000, seed 123) equal the
  accepted r4a_vii summary rows to 1e-9; G5 ties at the rank-20/21 boundary <= 1 % of queries.
- hit@K per group = share of the group's queries whose K nearest neighbours contain >= 1
  slide of the query's group (Methods l.299). Intervals: whole-patient bootstrap, nominal
  95 % percentile, unadjusted.
- Diversity (S2): per query, number of distinct group labels among the K nearest neighbours;
  per group, the mean over its queries. Association with hit@K across the 15 groups: OLS,
  R^2 = squared Pearson r, two-sided p for nonzero slope, plus Spearman rho. Descriptive;
  n = 15 points.

Outputs (`$PROSPER_OUT/figS1S2_neighbor_ladder_20261021/run_NNN/`): per_query_ladder.csv.gz,
s1_hitk_per_group.csv, s2_diversity_per_group.csv, s2_fit.csv, tie_census.json, gates.json,
input_bindings.json, run_resources.json, SHA256SUMS.txt, COMPLETE.json | FAILED.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
OUT = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
spec = importlib.util.spec_from_file_location("ladder", os.path.join(HERE, "r2_2_tile_ladder_whole_patient.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
spec4 = importlib.util.spec_from_file_location("r4a", os.path.join(HERE, "r4a_vii_whole_patient.py"))
R4 = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(R4)

TAG = "figS1S2_neighbor_ladder_20261021"
ID, LAB, PAT = R4.ID, R4.LAB, R4.PAT
KS = [1, 3, 5, 10, 15, 20]
KS_S2 = [5, 10, 15, 20]
DEPTH = 21
B, SEED = 5000, 123
TOL = 1e-9
ACCEPTED = "r4a_vii_whole_patient_20261015/production"
INPUTS = {
    "matrix": (R4.MATRIX_UNMASKED, R4.EXPECTED_SHA256[R4.MATRIX_UNMASKED], "frozen unmasked UNI2/OT distance matrix"),
    "meta": (R4.META, R4.EXPECTED_SHA256[R4.META], "slide metadata with UID.Subject"),
    "masked_ranks": (os.path.join(OUT, R4.MASKED_RANKS), R4.EXPECTED_SHA256[R4.MASKED_RANKS],
                     "frozen c2_metric_suite masked ranked lists, depth 10 (gate G2)"),
    "accepted_frame": (os.path.join(OUT, ACCEPTED, "per_query_frame.csv.gz"), None,
                       "accepted r4a_vii per-query frame, c1e_masked arm (gate G3)"),
    "accepted_summary": (os.path.join(OUT, ACCEPTED, "summary.csv"), None,
                         "accepted r4a_vii whole-patient summary (gate G4)"),
}


def sha(path):
    return L.sha256_bytes(L.read_bytes(path))


def next_run_dir():
    base = os.path.join(OUT, TAG)
    os.makedirs(base, exist_ok=True)
    n = 1 + max([int(d.split("_")[1]) for d in os.listdir(base) if d.startswith("run_")] or [0])
    d = os.path.join(base, f"run_{n:03d}")
    os.makedirs(d)
    return d


def mask_and_rank(values, patients, depth):
    n = values.shape[0]
    pats = np.asarray(patients)
    idx = np.full((n, depth), -1, dtype=np.int64)
    dist = np.full((n, depth), np.nan)
    for i in range(n):
        s = values[i, :].astype(float).copy()
        s[pats == pats[i]] = np.nan  # self and every same-patient slide
        valid = np.where(~np.isnan(s))[0]
        order = valid[np.argsort(s[valid], kind="mergesort")][:depth]
        idx[i, :len(order)] = order
        dist[i, :len(order)] = s[order]
    return idx, dist


def main():
    t0 = time.time()
    run = next_run_dir()
    gates, bindings = [], {}

    def gate(name, ok, detail, **extra):
        gates.append(dict(gate=name, passed=bool(ok), detail=detail, **extra))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return bool(ok)

    ok_all = True
    for key, (path, expected, role) in INPUTS.items():
        h = sha(path)
        bindings[key] = dict(path=path, role=role, sha256=h, expected_sha256=expected,
                             matches_expected=(expected is None or h == expected), bytes=os.path.getsize(path))
        ok_all &= gate(f"binding_{key}", expected is None or h == expected, f"{path} sha256 {h[:16]}…")

    slide_eval, _, _ = R4.prosper_modules()
    sim = slide_eval.load_similarity_matrix(INPUTS["matrix"][0])
    meta = slide_eval.load_metadata(INPUTS["meta"][0], label_col=LAB, id_col=ID, patient_col=PAT, drop_labels=["MISC"])
    sim_f, meta_f = slide_eval.filter_similarity_matrix(sim, meta, ID)
    ids = sim_f.index.tolist()
    lab = dict(zip(meta_f[ID], meta_f[LAB]))
    pat = dict(zip(meta_f[ID], meta_f[PAT]))
    classes = sorted(set(lab.values()))
    ok_all &= gate("G1_population", sim_f.shape == (R4.N_QUERIES, R4.N_QUERIES) and classes == sorted(L.CLASSES)
                   and len(set(pat.values())) == R4.N_PATIENTS,
                   f"matrix {sim_f.shape}, {len(classes)} labels, {len(set(pat.values()))} patients")

    idx, dist = mask_and_rank(sim_f.values, [pat[q] for q in ids], DEPTH)
    nb_ids = [[ids[j] for j in row] for row in idx]
    nb_lab = [[lab[x] for x in row] for row in nb_ids]

    # G2: distinct masking route vs frozen c2 masked lists (depth 10)
    fr = pd.read_csv(INPUTS["masked_ranks"][0], dtype={"query": str, "neighbour": str})
    fr = fr.sort_values(["query", "rank"])
    frozen = {q: (g["neighbour"].tolist(), g["neighbour_label"].tolist()) for q, g in fr.groupby("query", sort=False)}
    bad_q = [q for i, q in enumerate(ids) if frozen.get(str(q), ([], []))[0] != [str(x) for x in nb_ids[i][:10]]
             or frozen.get(str(q), ([], []))[1] != nb_lab[i][:10]]
    ok_all &= gate("G2_depth10_equals_frozen_masked_lists", not bad_q and len(frozen) == len(ids),
                   f"{len(bad_q)} of {len(ids)} queries differ from the frozen lists (ids or labels, ranks 1-10); "
                   f"frozen queries {len(frozen)}")

    # per-query ladder
    rows = []
    for i, q in enumerate(ids):
        t = lab[q]
        r = dict(query=q, true_label=t, patient=pat[q], top20="|".join(nb_lab[i][:20]))
        for k in KS:
            r[f"hit{k}"] = float(t in nb_lab[i][:k])
        for k in KS_S2:
            r[f"distinct{k}"] = len(set(nb_lab[i][:k]))
        rows.append(r)
    pq = pd.DataFrame(rows)

    # G3: per-query hit@k vs accepted c1e_masked frame
    acc = pd.read_csv(INPUTS["accepted_frame"][0], dtype={"query": str})
    acc = acc[acc.config == "c1e_masked"].set_index("query")
    mism = 0
    for k in (1, 3, 5, 10):
        mism += int((pq.set_index(pq["query"].astype(str))[f"hit{k}"] != acc.loc[pq["query"].astype(str), f"hit{k}"]).sum())
    ok_all &= gate("G3_per_query_hitk_equals_accepted_frame", mism == 0 and len(acc) == len(pq),
                   f"{mism} per-query hit@k disagreements over k in 1,3,5,10; accepted rows {len(acc)}")

    # G5: tie census at the depth boundary
    ties20 = int((dist[:, 19] == dist[:, 20]).sum())
    tc = dict(n_queries=len(ids), tie_rank20_rank21=ties20, tie_rank10_rank11=int((dist[:, 9] == dist[:, 10]).sum()),
              tie_rank5_rank6=int((dist[:, 4] == dist[:, 5]).sum()), tie_rank15_rank16=int((dist[:, 14] == dist[:, 15]).sum()),
              rule="stable mergesort on matrix row order after masking self and same-patient slides")
    ok_all &= gate("G5_tie_census_depth20", ties20 / len(ids) <= R4.TIE_STOP_FRACTION,
                   f"{ties20} of {len(ids)} queries tie at the rank-20/21 boundary", **tc)

    # whole-patient bootstrap
    patients, mult, W = L.weight_matrix(pq["patient"].tolist(), B, SEED)
    G = np.stack([(pq["true_label"] == c).values.astype(float) for c in classes])  # (15, n)
    out = []
    for k in KS:
        v = pq[f"hit{k}"].values.astype(float)
        for ci_, c in enumerate(classes):
            m = G[ci_] == 1
            reps = L.boot_means(W, v, m)
            lo, hi, bad = L.ci(reps)
            out.append(dict(group=c, k=k, n_slides=int(m.sum()), n_patients=pq.loc[m, "patient"].nunique(),
                            hit=float(v[m].mean()), ci_lo=lo, ci_hi=hi, n_nonfinite=bad, flag=L.boundary_flag(lo, hi),
                            average="group"))
        reps = L.boot_means(W, v)
        lo, hi, bad = L.ci(reps)
        out.append(dict(group="ALL", k=k, n_slides=len(v), n_patients=len(patients), hit=float(v.mean()), ci_lo=lo,
                        ci_hi=hi, n_nonfinite=bad, flag=L.boundary_flag(lo, hi), average="micro"))
        reps = L.macro_reps(W, G, v)
        lo, hi, bad = L.ci(reps)
        out.append(dict(group="ALL", k=k, n_slides=len(v), n_patients=len(patients),
                        hit=float(np.mean([v[G[j] == 1].mean() for j in range(len(classes))])), ci_lo=lo, ci_hi=hi,
                        n_nonfinite=bad, flag=L.boundary_flag(lo, hi), average="macro"))
    s1 = pd.DataFrame(out)

    # G4: whole-population intervals equal the accepted r4a_vii masked rows (same draws)
    summ = pd.read_csv(INPUTS["accepted_summary"][0])
    summ = summ[(summ.arm == "c1e") & (summ.config == "masked")]
    diffs = []
    for k in (1, 3, 5, 10):
        for avg in ("micro", "macro"):
            a = summ[(summ.metric == f"hit{k}") & (summ.average == avg)].iloc[0]
            b = s1[(s1.group == "ALL") & (s1.k == k) & (s1.average == avg)].iloc[0]
            d = max(abs(a.point - b.hit), abs(a.ci_lo - b.ci_lo), abs(a.ci_hi - b.ci_hi))
            diffs.append((k, avg, d))
    worst = max(d for _, _, d in diffs)
    ok_all &= gate("G4_population_intervals_equal_accepted", worst <= TOL,
                   f"max |point/ci difference| vs accepted r4a_vii masked hit@k rows = {worst:.2e} (8 rows)",
                   mult_sha256=R4.mult_sha(mult))

    # S2 diversity
    s2_rows, fits = [], []
    for k in KS_S2:
        for c in classes:
            m = pq["true_label"] == c
            s2_rows.append(dict(group=c, k=k, n_slides=int(m.sum()), mean_distinct=float(pq.loc[m, f"distinct{k}"].mean()),
                                hit=float(pq.loc[m, f"hit{k}"].mean())))
        d = pd.DataFrame([r for r in s2_rows if r["k"] == k])
        lr = stats.linregress(d["mean_distinct"], d["hit"])
        rho = stats.spearmanr(d["mean_distinct"], d["hit"])
        fits.append(dict(k=k, n_groups=len(d), slope=lr.slope, intercept=lr.intercept, r2=lr.rvalue ** 2,
                         pearson_r=lr.rvalue, p_slope_two_sided=lr.pvalue, spearman_rho=rho.statistic,
                         spearman_p=rho.pvalue, x_min=d["mean_distinct"].min(), x_max=d["mean_distinct"].max(),
                         note="descriptive OLS across 15 tumour-group points; not a test of a mechanism"))
    s2 = pd.DataFrame(s2_rows)
    fit = pd.DataFrame(fits)

    # write
    pq.to_csv(os.path.join(run, "per_query_ladder.csv.gz"), index=False, compression="gzip")
    s1.to_csv(os.path.join(run, "s1_hitk_per_group.csv"), index=False)
    s2.to_csv(os.path.join(run, "s2_diversity_per_group.csv"), index=False)
    fit.to_csv(os.path.join(run, "s2_fit.csv"), index=False)
    json.dump(tc, open(os.path.join(run, "tie_census.json"), "w"), indent=2)
    json.dump(gates, open(os.path.join(run, "gates.json"), "w"), indent=2)
    json.dump(dict(inputs=bindings, kernel=dict(path=os.path.join(HERE, "r2_2_tile_ladder_whole_patient.py"),
                                                  sha256=sha(os.path.join(HERE, "r2_2_tile_ladder_whole_patient.py"))),
                   r4a_vii=dict(path=os.path.join(HERE, "r4a_vii_whole_patient.py"),
                                sha256=sha(os.path.join(HERE, "r4a_vii_whole_patient.py"))),
                   prosper_search=R4.prosper_provenance(), script_sha256=sha(os.path.abspath(__file__)),
                   design=dict(B=B, seed=SEED, depth=DEPTH, ks=KS, ks_s2=KS_S2, masking="self + same UID.Subject",
                               ranking="stable mergesort ascending distance", interval="whole-patient percentile 95%")),
              open(os.path.join(run, "input_bindings.json"), "w"), indent=2)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=R, stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"unavailable: {e}"
    json.dump(dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, host=platform.node(),
                   wall_seconds=round(time.time() - t0, 1),
                   max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                   workspace_head=head, argv=sys.argv, lsf_jobid=os.environ.get("LSB_JOBID")),
              open(os.path.join(run, "run_resources.json"), "w"), indent=2)
    with open(os.path.join(run, "SHA256SUMS.txt"), "w") as f:
        for name in sorted(os.listdir(run)):
            if name != "SHA256SUMS.txt":
                f.write(f"{sha(os.path.join(run, name))}  {name}\n")
    final = "COMPLETE.json" if ok_all else "FAILED.json"
    json.dump(dict(passed=ok_all, gates=[g["gate"] for g in gates if not g["passed"]] if not ok_all else "all",
                   run=run), open(os.path.join(run, final), "w"), indent=2)
    print(f"{final} -> {run}")
    print(fit.to_string())
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
