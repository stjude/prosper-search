#!/usr/bin/env python3
"""R2.2-ALL Stage E: matched evaluation of the all-eligible (100 %) bank against the accepted mosaic 15 % results,
for SAMPLER, Yottixel and production median pooling (the last added by amendment 2026-11-13
~21:15 CST; exact OT at 100 % was halted on 2026-11-13 20:57 CST; the measured
evidence is in stage_c/c3/HALTED.md and no OT arm is evaluated here).

Population: the 2,778 non-MISC slides (1,876 patients, 15 groups incl. NORMAL); queries = library; self and
same-patient references excluded. This is the accepted main population of c2_metric_suite / R3.1.

Arms (6):
  sampler_mosaic    accepted ranked lists c2_metric_suite/ranked_lists_uni2_sampler.csv.gz (15 % mosaic, published
                    matrix); reconciled by re-ranking the published no-same-patient matrix with the production rule
  yottixel_mosaic   same for ranked_lists_uni2_yottixel.csv.gz
  sampler_all       stage_d/sampler/sampler_matrix.npz ranked with the production rule
  yottixel_all      stage_d/yottixel/matrix/yottixel_matrix.npz ranked with the production rule
  median_pooling_mosaic / median_pooling_all   the same for ranked_lists_uni2_median_pooling.csv.gz and
                    stage_d/median_pooling/median_pooling_matrix.npz

Ranking rule (identical for every arm): prosper-search ``metrics_cli.ranked_neighbours`` (imported, not copied):
stable mergesort of the query row in metadata order (published ``filter_similarity_matrix`` order), self and
same-patient cells dropped, depth 10. All-eligible excluded cells carry the production 1e6 sentinel (as the
published no-same-patient matrices do) and are dropped by the same filter; a real distance >= 1e6 is a failure.

Estimands and bootstrap: the accepted functions of scripts/r2_2_tile_ladder_whole_patient.py (imported; per-query
hit@1/3/5/10, precision@5/10, plurality@5 with the A-3 tie rule, depth-10 tie bounds) plus the production
per-query vectors of every metric the mosaic evaluation reported (``metrics_cli.per_query_metric_vectors``:
hit/precision/recall/AP/nDCG at 1,3,5,10 and MRR). Whole-patient bootstrap ([decision record]): patients drawn with replacement,
B = 5,000, PCG64 seed 123, identical draws for all arms; paired difference = all-eligible minus mosaic within each
method and aggregation. Primary: macro and micro hit@5 and precision@5. Intervals are nominal 95 %, unadjusted; an
interval including 0 is not evidence of equivalence.

Tie sensitivity: (i) the ladder's depth-10 arrangement bounds; (ii) exact full-row bounds for hit@5 and precision@5
from the whole matrix row (every reference tied with the 5th-nearest distance), so ties beyond depth 10 are seen.

Checks (reconciliation.json): mosaic re-ranking reproduces the accepted ranked lists exactly (neighbours and
distances); mosaic metric points equal the saved c2_metric_suite points; the Yottixel and median-pooling mosaic
whole-patient points and intervals equal the accepted R3.1 comparators summary (configs yottixel, median); an independent numpy recount (lexsort, no
ranked_neighbours) of the top-10 for every arm and query; ladder and production per-query vectors agree; a
brute-force rebuild of 25 bootstrap replicates; population, exclusion and coverage checks.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import io
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = Path("/PROSPER_WORKSPACE/prosper_revision")
H = Path("/PROSPER_WORKSPACE")
D_REPO = H / "prosper-search"
OUT = Path(os.environ.get("PROSPER_OUT", str(R / "prosper_revision_output")))
RUN = OUT / "r2_2_uni2_all_eligible_20261113" / "run_001"
LADDER = R / "scripts" / "r2_2_tile_ladder_whole_patient.py"
PUBLISHED = H / "comet-feature-gen/comet-uni2/output/similarity_matrix"
C2 = OUT / "c2_metric_suite"
R31 = OUT / "r3_1_comparators_whole_patient_20261008" / "summary.csv"
SENTINEL = 1.0e6
DEPTH = 10
KS = (1, 3, 5, 10)
# median_pooling added by amendment 2026-11-13 ~21:15 CST (mirrors SAMPLER).
METHODS = ("sampler", "yottixel", "median_pooling")
ARMS = [("sampler_mosaic", "sampler", "mosaic15"), ("sampler_all", "sampler", "all_eligible"),
        ("yottixel_mosaic", "yottixel", "mosaic15"), ("yottixel_all", "yottixel", "all_eligible"),
        ("median_pooling_mosaic", "median_pooling", "mosaic15"), ("median_pooling_all", "median_pooling", "all_eligible")]
CONTRASTS = [("sampler_all", "sampler_mosaic"), ("yottixel_all", "yottixel_mosaic"),
             ("median_pooling_all", "median_pooling_mosaic")]
R31_CONFIG = {"yottixel_mosaic": "yottixel", "median_pooling_mosaic": "median"}
LADDER_METRICS = ["hit1", "hit3", "hit5", "hit10", "precision5", "precision10", "plurality5"]
N_SPOT = 25


def load_ladder():
    spec = importlib.util.spec_from_file_location("ladder", LADDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def production_eval():
    sys.path.insert(0, str(D_REPO))
    from prosper.evaluation import metrics_cli, slide_eval  # noqa: E402
    from prosper.evaluation.cli import PUBLISHED as PUB  # noqa: E402
    return metrics_cli, slide_eval, PUB


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def all_eligible_frame(npz_path, order):
    with np.load(npz_path) as z:
        ids = [str(s) for s in z["ids"]]
        d = z["d"].astype(float).copy()
        excl = z["excluded"]
    if not np.isfinite(d).all() or (d[~excl] >= SENTINEL).any():
        raise ValueError(f"{npz_path}: non-finite distance or a real distance >= sentinel")
    d[excl] = SENTINEL
    df = pd.DataFrame(d, index=ids, columns=ids)
    return df.loc[order, order], int(excl.sum())


def ranked_to_frame(nbr_ids, nbr_labels, nbr_dists, query_ids, true_labels):
    n = len(query_ids)
    return pd.DataFrame({"query": np.repeat(query_ids, DEPTH), "true_label": np.repeat(true_labels, DEPTH),
                         "rank": np.tile(np.arange(1, DEPTH + 1), n), "neighbour": nbr_ids.ravel(),
                         "neighbour_label": nbr_labels.ravel(), "distance": nbr_dists.ravel()})


def write_ranked(df, path):
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format=None)
    data = gzip.compress(buf.getvalue().encode(), mtime=0)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def independent_top10(arr, ids, patients):
    """Distinct route: lexsort by (distance, metadata position) after masking self/same-patient with +inf."""
    pat = np.asarray(patients, dtype=object)
    out = np.empty((len(ids), DEPTH), dtype=object)
    pos = np.arange(len(ids))
    for i in range(len(ids)):
        row = arr[i].astype(float).copy()
        row[pat == pat[i]] = np.inf
        o = np.lexsort((pos, row))[:DEPTH]
        out[i] = [ids[j] for j in o]
    return out


def full_row_bounds(arr, labels, patients, k=5):
    """Exact hit@k / precision@k bounds over every reordering of references tied at the k-th distance."""
    lab = np.asarray(labels, dtype=object)
    pat = np.asarray(patients, dtype=object)
    rows = []
    for i in range(len(lab)):
        keep = pat != pat[i]
        d, l = arr[i][keep].astype(float), lab[keep]
        dk = np.partition(d, k - 1)[k - 1]
        s, t = d < dk, d == dk
        need = k - int(s.sum())
        hs, ht = int((l[s] == lab[i]).sum()), int((l[t] == lab[i]).sum())
        nt = int(t.sum())
        rows.append(dict(hit_lo=int(hs > 0 or (nt - ht) < need), hit_hi=int(hs > 0 or ht > 0),
                         prec_lo=(hs + max(0, need - (nt - ht))) / k, prec_hi=(hs + min(need, ht)) / k,
                         tie_block=nt, straddles=int(nt > need)))
    return pd.DataFrame(rows)


def macro_mean(v, groups, classes):
    return float(np.mean([v[groups == c].mean() for c in classes]))


def arm_costs(a, base):
    """Library/query tile counts per arm (both sides use the same bank) and the Stage D compute record."""
    bp = pd.read_csv(a.bank_per_slide, dtype={"slide": str}).set_index("slide").loc[base.index.astype(str)]
    out = dict(population_slides=int(len(bp)),
               tiles_all_eligible=int(bp["n_rows"].sum()), tiles_mosaic15=int(bp["n_code0"].sum()),
               tiles_ratio_all_over_mosaic=float(bp["n_rows"].sum() / bp["n_code0"].sum()),
               per_slide_all_eligible_median=float(bp["n_rows"].median()),
               per_slide_mosaic15_median=float(bp["n_code0"].median()),
               bank_per_slide=str(a.bank_per_slide), bank_per_slide_sha256=sha_file(a.bank_per_slide))
    cp = Path(a.stage_d_costs)
    out["stage_d"] = json.loads(cp.read_text()) if cp.exists() else "not supplied"
    if cp.exists():
        out["stage_d_costs_sha256"] = sha_file(cp)
    return out


def write_readme(out, recon, s, pdiff, tb):
    prim = s[s["metric"].isin(["hit5", "precision5"])] if "metric" in s.columns else s
    lines = ["# R2.2 Stage E: matched evaluation, all-eligible 100 % vs mosaic 15 % (SAMPLER, Yottixel, median pooling)", "",
             f"Status: **{recon['status']}** (failed checks: {recon['failed_checks'] or 'none'}). "
             f"Generated {recon['config']['generated']} by `{recon['config']['script']}` "
             f"(sha256 {recon['config']['script_sha256']}).", "",
             "Population: 2,778 non-MISC slides / 1,876 patients; self and same-patient references excluded "
             "exactly as in the accepted mosaic evaluation (prosper-search `metrics_cli.ranked_neighbours`, stable "
             "mergesort in metadata order). Whole-patient bootstrap B=%d, seed %d; paired differences are "
             "all-eligible minus mosaic on the same patient draws." % (recon["config"]["B"], recon["config"]["seed"]), "",
             "OT arm: " + recon["config"]["ot_arm"] + ".", "",
             "Retrieval metrics describe slide-level label agreement of nearest neighbours; they are not diagnostic "
             "accuracy or clinical usefulness.", "",
             "## Primary metrics (hit@5, precision@5)", "", prim.to_markdown(index=False), "",
             "## Paired differences (all-eligible minus mosaic)", "", pdiff.to_markdown(index=False), "",
             "## Tie sensitivity (exact full-row bounds)", "", tb.T.to_markdown(), "",
             "## Costs", "", "```json", json.dumps(recon["costs"], indent=1, default=str), "```", "",
             "## Files", "", "- `summary.csv`: all arms, metrics, micro/macro, point and 95 % percentile CI.",
             "- `paired_differences.csv`, `per_group.csv`, `tie_bounds.csv`, `per_query_frame.csv.gz`, "
             "`ranked_*.csv.gz` (all-eligible ranked lists, depth 10).",
             "- `reconciliation.json`: every check, input SHA256s, configuration, production point estimates.",
             "- `sha256.txt`: checksums of every file in this directory; `COMPLETE` written only on PASS.", ""]
    (out / "README.md").write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(RUN / "stage_e"))
    ap.add_argument("--sampler-all", default=str(RUN / "stage_d/sampler/sampler_matrix.npz"))
    ap.add_argument("--yottixel-all", default=str(RUN / "stage_d/yottixel/matrix/yottixel_matrix.npz"))
    ap.add_argument("--median-pooling-all", default=str(RUN / "stage_d/median_pooling/median_pooling_matrix.npz"))
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--bank-per-slide", default=str(RUN / "stage_b/bank_per_slide.csv"))
    ap.add_argument("--stage-d-costs", default=str(RUN / "stage_d/costs.json"))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    Lw = load_ladder()
    mcli, se, PUB = production_eval()
    idc, labc, patc = PUB["id_col"], PUB["label_col"], PUB["patient_col"]
    meta_idx, meta_sha = Lw.load_meta(Lw.META)
    meta_all = se.load_metadata(Lw.META, label_col=labc, id_col=idc, patient_col=patc, drop_labels=["MISC"],
                                min_class_count=None)
    bank = sorted((RUN / "stage_b/gap_d1/target_slides.txt").read_text().split(), key=int)
    checks, hashes, info = {}, {"metadata": meta_sha}, {}
    frames, ranked, per_q_prod, arrays, prod_points = {}, {}, {}, {}, {}
    meta_f_ref = None
    for tag, method, source in ARMS:
        pub_path = PUBLISHED / f"combined_similarity_matrix_uni2_{method}_no_same_patient.pkl"
        hashes[pub_path.name] = sha_file(pub_path)
        with open(pub_path, "rb") as f:
            pub = pickle.load(f)
        pub_f, meta_f = se.filter_similarity_matrix(pub, meta_all, id_col=idc)
        ids = meta_f[idc].astype(str).tolist()
        if meta_f_ref is None:
            meta_f_ref = ids
            checks["population_equals_bank_slides"] = sorted(ids, key=int) == bank
            checks["population_n"] = len(ids)
            checks["population_patients"] = int(meta_f[patc].nunique())
        elif ids != meta_f_ref:
            raise ValueError(f"{tag}: metadata order differs between arms")
        if source == "mosaic15":
            sim = pub_f
            n_excl = int((pub_f.loc[ids, ids].to_numpy() >= SENTINEL).sum())
        else:
            p = Path({"sampler": a.sampler_all, "yottixel": a.yottixel_all,
                      "median_pooling": a.median_pooling_all}[method])
            hashes[f"{tag}:{p.name}"] = sha_file(p)
            sim, n_excl = all_eligible_frame(p, list(pub_f.index))
        nbr_ids, nbr_labels, nbr_dists, n_rel, diag = mcli.ranked_neighbours(sim, meta_f, DEPTH, idc, labc, patc)
        info[tag] = dict(method=method, source=source, sentinel_or_excluded_cells=n_excl, ranking_diag=diag)
        qids = meta_f[idc].astype(str).tolist()
        tl = meta_f[labc].astype(str).tolist()
        rk = ranked_to_frame(nbr_ids, nbr_labels, nbr_dists, qids, tl)
        if source == "mosaic15":
            acc = C2 / f"ranked_lists_uni2_{method}.csv.gz"
            accepted, h = Lw.load_ranks(str(acc), meta_idx)
            hashes[acc.name] = h
            mine = rk.sort_values(["query", "rank"], kind="mergesort").reset_index(drop=True)
            checks[f"{tag}_rerank_equals_accepted_neighbours"] = bool((mine["neighbour"].values == accepted["neighbour"].values).all())
            checks[f"{tag}_rerank_equals_accepted_distances"] = bool((mine["distance"].values.astype(float) == accepted["distance"].values).all())
            rpath = acc
        else:
            rpath = out / f"ranked_lists_{tag}.csv.gz"
            hashes[rpath.name] = write_ranked(rk, rpath)
        r, _ = Lw.load_ranks(str(rpath), meta_idx)
        ranked[tag] = r
        frames[tag] = Lw.per_query(r, meta_idx)
        vec = mcli.per_query_metric_vectors(nbr_labels, tl, n_rel, list(KS))
        pq = pd.DataFrame(vec, index=qids).sort_index()
        per_q_prod[tag] = pq
        pts = mcli.metrics.compute_all(nbr_labels, tl, k_values=list(KS), n_relevant=n_rel)
        prod_points[tag] = pts
        arr = sim.loc[ids, ids].to_numpy(float)
        arrays[tag] = arr
        # distinct route for the top-10
        ind = independent_top10(arr, ids, meta_f[patc].astype(str).tolist())
        checks[f"{tag}_independent_top10_equal"] = bool((ind == nbr_ids).all())
        h5 = pq[f"hit_at_5"].reindex(frames[tag].index).to_numpy()
        checks[f"{tag}_ladder_vs_production_hit5_equal"] = bool((h5 == frames[tag]["hit5"].to_numpy()).all())
        p5 = pq[f"precision_at_5"].reindex(frames[tag].index).to_numpy()
        checks[f"{tag}_ladder_vs_production_precision5_equal"] = bool(np.allclose(p5, frames[tag]["precision5"], atol=1e-12))
        if source == "mosaic15":
            saved = json.loads((C2 / f"metrics_uni2_{method}.json").read_text())["point_estimates"]
            hashes[f"metrics_uni2_{method}.json"] = sha_file(C2 / f"metrics_uni2_{method}.json")
            diffs = {k: abs(pts[k] - saved[k]) for k in saved if k in pts and isinstance(saved[k], float)}
            checks[f"{tag}_points_equal_saved_c2_metric_suite_max_abs"] = max(diffs.values())
    # common index / patient checks
    base = frames[ARMS[0][0]]
    for tag in frames:
        if not frames[tag].index.equals(base.index) or not (frames[tag]["patient"] == base["patient"]).all():
            raise ValueError(f"{tag}: query index/patient differs")
    # merge production per-query vectors into frames
    prod_cols = sorted(per_q_prod[ARMS[0][0]].columns)
    for tag in frames:
        for c in prod_cols:
            frames[tag][c] = per_q_prod[tag][c].reindex(frames[tag].index).to_numpy(float)
    classes = list(Lw.CLASSES)
    grp = base["true_label"].to_numpy()
    patients, mult, W = Lw.weight_matrix(base["patient"], a.B, a.seed)
    G = np.array([(grp == c) for c in classes]).astype(float)
    metrics_all = LADDER_METRICS + [c for c in prod_cols if c not in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
                                                                      "precision_at_5", "precision_at_10")]
    reps, points, summary, paired, per_group = {}, {}, [], [], []
    n, npat = len(base), len(patients)
    for tag, method, source in ARMS:
        fr = frames[tag]
        for m in metrics_all:
            v = fr[m].to_numpy(float)
            points[(tag, m, "micro")] = float(v.mean())
            points[(tag, m, "macro")] = macro_mean(v, grp, classes)
            reps[(tag, m, "micro")] = Lw.boot_means(W, v)
            reps[(tag, m, "macro")] = Lw.macro_reps(W, G, v)
            for pop in ("micro", "macro"):
                lo, hi, bad = Lw.ci(reps[(tag, m, pop)])
                summary.append(dict(arm=tag, method=method, source=source, metric=m, average=pop,
                                    point=points[(tag, m, pop)], ci_lo=lo, ci_hi=hi, n_queries=n, n_patients=npat,
                                    n_groups=len(classes), n_replicates=a.B, n_nonfinite=bad,
                                    flag=Lw.boundary_flag(lo, hi),
                                    tier=("primary" if m in ("hit5", "precision5") else
                                          "secondary" if m in LADDER_METRICS else "reported_by_mosaic_eval")))
        for m in ("hit5", "precision5", "hit10", "plurality5"):
            v = fr[m].to_numpy(float)
            for i, c in enumerate(classes):
                mask = G[i] > 0
                lo, hi, bad = Lw.ci(Lw.boot_means(W, v, mask))
                per_group.append(dict(arm=tag, method=method, source=source, group=c, metric=m,
                                      n_queries=int(mask.sum()), n_patients=int(fr.loc[mask, "patient"].nunique()),
                                      point=float(v[mask].mean()), ci_lo=lo, ci_hi=hi, n_replicates=a.B,
                                      n_nonfinite=bad, flag=Lw.boundary_flag(lo, hi)))
    for x, y in CONTRASTS:
        for m in metrics_all:
            for pop in ("micro", "macro"):
                dd = reps[(x, m, pop)] - reps[(y, m, pop)]
                lo, hi, bad = Lw.ci(dd)
                paired.append(dict(minuend=x, subtrahend=y, metric=m, average=pop,
                                   point=points[(x, m, pop)] - points[(y, m, pop)], ci_lo=lo, ci_hi=hi,
                                   n_queries=n, n_patients=npat, n_replicates=a.B, n_nonfinite=bad,
                                   ci_includes_zero_nominal=bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi),
                                   tier=("primary" if m in ("hit5", "precision5") else
                                         "secondary" if m in LADDER_METRICS else "reported_by_mosaic_eval"),
                                   definition="all-eligible minus mosaic 15 % on the same queries and the same "
                                              "patient draws; nominal 95%, unadjusted; including 0 is not evidence "
                                              "of equivalence"))
    # brute-force replicate rebuild
    rows_of = {p: np.flatnonzero((base["patient"] == p).to_numpy()) for p in patients}
    worst = 0.0
    for b in np.linspace(0, a.B - 1, N_SPOT).astype(int):
        idx = np.concatenate([np.repeat(rows_of[p], mult[b, i]) for i, p in enumerate(patients)])
        g = grp[idx]
        complete = all((g == c).any() for c in classes)
        for tag, _, _ in ARMS:
            for m in ("hit5", "precision5"):
                v = frames[tag][m].to_numpy(float)[idx]
                worst = max(worst, abs(v.mean() - reps[(tag, m, "micro")][b]))
                if complete:
                    worst = max(worst, abs(np.mean([v[g == c].mean() for c in classes]) - reps[(tag, m, "macro")][b]))
    checks["bootstrap_bruteforce_max_abs"] = worst
    # accepted R3.1 comparators (yottixel and median-pooling mosaic, same population and draws)
    r31 = pd.read_csv(R31)
    r31 = r31[r31.population == "all_tissue_2778"]
    hashes["r3_1_summary.csv"] = sha_file(R31)
    s = pd.DataFrame(summary)
    for arm, cfg in R31_CONFIG.items():
        mx = 0.0
        for m in LADDER_METRICS:
            for pop in ("micro", "macro"):
                acc = r31[(r31.config == cfg) & (r31.metric == m) & (r31.average == pop)].iloc[0]
                mine = s[(s.arm == arm) & (s.metric == m) & (s.average == pop)].iloc[0]
                mx = max(mx, abs(acc.point - mine.point), abs(acc.ci_lo - mine.ci_lo), abs(acc.ci_hi - mine.ci_hi))
        checks[f"{arm}_equals_accepted_r3_1_max_abs"] = mx
    # tie sensitivity
    tie_rows = []
    for tag, method, source in ARMS:
        fr = frames[tag]
        fb = full_row_bounds(arrays[tag], meta_all.set_index(idc).loc[meta_f_ref, labc].astype(str).to_numpy(),
                             meta_all.set_index(idc).loc[meta_f_ref, patc].astype(str).to_numpy())
        fb.index = meta_f_ref
        fb = fb.reindex(fr.index)
        for c in ("hit_lo", "hit_hi", "prec_lo", "prec_hi", "straddles"):
            fr[f"fullrow_{c}"] = fb[c].to_numpy()
        row = dict(arm=tag, method=method, source=source,
                   queries_any_tie_ranks1to6=int(fr["any_tie_ranks1to6"].sum()),
                   queries_rank56_tie=int(fr["rank56_tie"].sum()),
                   queries_tie_block_reaches_depth=int(fr["tie_block_reaches_depth"].sum()),
                   queries_fullrow_kth_tie_straddles=int(fb["straddles"].sum()),
                   queries_hit5_tie_dependent_depth10=int((fr["hit5_tie_lower"] != fr["hit5_tie_upper"]).sum()),
                   queries_hit5_tie_dependent_fullrow=int((fb["hit_lo"] != fb["hit_hi"]).sum()),
                   queries_precision5_tie_dependent_fullrow=int((fb["prec_lo"] != fb["prec_hi"]).sum()))
        for m, lo_c, hi_c in (("hit5", "fullrow_hit_lo", "fullrow_hit_hi"), ("precision5", "fullrow_prec_lo", "fullrow_prec_hi")):
            for pop in ("micro", "macro"):
                f = (lambda v: float(v.mean())) if pop == "micro" else (lambda v: macro_mean(v, grp, classes))
                row[f"{pop}_{m}_fullrow_lower"] = f(fr[lo_c].to_numpy(float))
                row[f"{pop}_{m}_saved_order"] = f(fr[m].to_numpy(float))
                row[f"{pop}_{m}_fullrow_upper"] = f(fr[hi_c].to_numpy(float))
        within = bool(((fr["fullrow_hit_lo"] <= fr["hit5"]) & (fr["hit5"] <= fr["fullrow_hit_hi"])).all()
                      and ((fr["fullrow_prec_lo"] <= fr["precision5"] + 1e-12) & (fr["precision5"] <= fr["fullrow_prec_hi"] + 1e-12)).all())
        checks[f"{tag}_saved_order_within_fullrow_bounds"] = within
        tie_rows.append(row)
    tb = pd.DataFrame(tie_rows)
    # coverage / exclusion
    checks["every_query_has_depth_neighbours"] = all(info[t]["ranking_diag"]["queries_with_fewer_than_depth_neighbours"] == 0 for t in info)
    checks["no_sentinel_before_depth"] = all(info[t]["ranking_diag"]["sentinels_encountered_before_depth"] == 0 for t in info)
    checks["no_misc_in_population"] = bool((base["true_label"] != "MISC").all())
    bool_keys = [k for k, v in checks.items() if isinstance(v, bool)]
    tol_keys = {"bootstrap_bruteforce_max_abs": 1e-12,
                **{f"{arm}_equals_accepted_r3_1_max_abs": 1e-12 for arm in R31_CONFIG},
                **{f"{mm}_mosaic_points_equal_saved_c2_metric_suite_max_abs": 1e-12 for mm in METHODS}}
    failed = [k for k in bool_keys if not checks[k]] + [k for k, t in tol_keys.items() if not checks.get(k, 1) <= t]
    status = "PASS" if not failed else "FAIL"
    s.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_differences.csv", index=False)
    pd.DataFrame(per_group).to_csv(out / "per_group.csv", index=False)
    tb.to_csv(out / "tie_bounds.csv", index=False)
    pqf = pd.concat({t: frames[t] for t, _, _ in ARMS}, names=["arm", "query"]).reset_index()
    pqf.to_csv(out / "per_query_frame.csv.gz", index=False, compression="gzip")
    recon = dict(status=status, failed_checks=failed, checks=checks, arms=info, inputs_sha256=hashes,
                 config=dict(B=a.B, seed=a.seed, depth=DEPTH, k_values=list(KS), classes=classes,
                             contrasts=CONTRASTS, draws=Lw.DRAWS_NOTE,
                             ranking="prosper-search metrics_cli.ranked_neighbours (stable mergesort in metadata order; "
                                     "self/same-patient dropped)",
                             ot_arm="not evaluated: exact OT at 100 % halted by the author 2026-11-13 20:57 CST; "
                                    "measured evidence in stage_c/c3/HALTED.md",
                             script=str(Path(__file__).resolve()), script_sha256=sha_file(__file__),
                             ladder_script=str(LADDER), ladder_sha256=sha_file(LADDER),
                             metrics_cli_sha256=sha_file(D_REPO / "prosper/evaluation/metrics_cli.py"),
                             metrics_sha256=sha_file(D_REPO / "prosper/evaluation/metrics.py"),
                             slide_eval_sha256=sha_file(D_REPO / "prosper/evaluation/slide_eval.py"),
                             generated=dt.datetime.now().isoformat()),
                 production_points={t: {k: v for k, v in p.items() if isinstance(v, (int, float))} for t, p in prod_points.items()})
    recon["costs"] = arm_costs(a, base)
    (out / "reconciliation.json").write_text(json.dumps(recon, indent=2, default=str))
    write_readme(out, recon, s, pd.DataFrame(paired), tb)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("sha256.txt", "COMPLETE"))
    (out / "sha256.txt").write_text("".join(f"{sha_file(p)}  {p.relative_to(out)}\n" for p in files))
    if status == "PASS":
        (out / "COMPLETE").write_text(dt.datetime.now().isoformat() + "\n")
    print(json.dumps(dict(status=status, failed=failed), indent=1), flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
