#!/usr/bin/env python3
"""R2.2: whole-patient bootstrap 95% CIs for the tile-retention ladder from the SAVED depth-10 ranked lists.

A review comment asked whether the 15 % tile-retention setting is arbitrary and what the trade-off is. The ten
ladder rungs (median pooling at 5/15/30/50/100 %, Yottixel at 5/15/30 %, OT cosine at 15 % and — added
2026-10-07 under the adopted OT-30 specification, [internal record] — OT cosine
at 30 %) were searched; their depth-10 ranked lists live in $PROSPER_OUT/r22_metrics/. The intervals saved beside them
use the retired pps_cluster_one_slide sampler, so this script recomputes every estimand under the accepted
[decision record] whole-patient bootstrap (patients drawn with replacement, B = 5,000, PCG64 seed 123, every query
weighted by its patient's draw multiplicity; identical draws for all rungs, so paired differences are exact).
No retrieval is re-run; no feature, model or package code is touched.

Population: the ladder's own 2,750-query fully-tissue-tile universe (UNI2, 15 groups incl. NORMAL, MISC
dropped, same-patient masked; library = the queries). It is NOT the 2,778-query main population of
c2_metric_suite (28 non-MISC slides / 19 patients absent; counts written to reconciliation.json),
and "100 %" means every tile the mosaic admits, not every tissue tile. Points are not comparable with the
main-grid numbers.

Estimands per rung (per query, then micro = mean over queries, macro = mean of the 15 group means):
  hit@k (k = 1,3,5,10)      truth appears among the top-k neighbours (vote-rule independent)
  precision@k (k = 5,10)    fraction of top-k neighbours with the truth label
  plurality@5               most frequent label among the top-5 equals the truth; ties -> closest-ranked
                            tied label (current A-3 rule; never uses the truth)
Paired differences (same queries, same patient draws): each median rung minus median 15 %, each Yottixel
rung minus Yottixel 15 %, OT 15 % minus median 15 %, OT 15 % minus Yottixel 15 %, OT 30 % minus OT 15 %
(the OT-30 specification's primary paired question; tiles nested as prefixes of one per-slide permutation),
OT 30 % minus median 30 %, OT 30 % minus Yottixel 30 %.
Per-group hit@5 and plurality@5 for every rung.
Tie bounds: for hit@5, precision@5 and plurality@5, the lower/upper macro and micro values obtainable by
reordering neighbours within exact-distance tie blocks that intersect the top-5 (exhaustive enumeration of
distinct label arrangements; visible top-10 only, so a tie block reaching rank 10 is flagged as possibly
truncated and ties at ranks 10/11 are unobservable). Points only; they show how much each saved Yottixel
value rests on the arbitrary saved order within ties. Median and OT rungs have no ties.

Population accounting: the ladder library is exactly its 2,750 queries. Relative to the 2,778-query main
population, 28 non-MISC slides (19 patients) are absent; their per-group counts are written to
reconciliation.json and README so the ladder can never be read as the main grid.

Intervals are nominal 95 %, unadjusted for multiplicity. An interval that includes 0 is not evidence of
equivalence or of "no gain". Primary outcomes under [decision record] are hit@5 and precision@5.

Outputs (written to --out-dir, default $PROSPER_OUT/r2_2_tile_ladder_whole_patient_<YYYYMMDD>/):
  ladder_summary.csv, paired_differences.csv, per_group.csv, tie_bounds.csv, per_query_frame.csv.gz,
  reconciliation.json, README.md
"""
import argparse
import datetime as dt
import gzip
import hashlib
import io
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
META = ("/PROSPER_WORKSPACE/comet-feature-gen/comet-analysis/"
        "data/final_scan_id_disease_map_with_sample_id_patient_id.csv")
CLASSES = ["ACT/ACC", "DSRCT", "EWS", "LIVER", "MEL", "MPNST", "MRT", "NB", "NF", "NORMAL",
           "OS", "RB", "RMS", "SS", "WT"]
DEPTH = 10
EXPECTED_QUERIES = 2750
EXPECTED_PATIENTS = 1857
MAIN_POPULATION_QUERIES = 2778
# sha256 of the nine frozen ranked lists as read on 2026-10-06 (analyst), plus the OT 30 % list produced 2026-10-07
# (combine 329952808 -> metrics_cli 329952818; see [analysis plan] OT-30 record); a mismatch is a FAIL, not a silent rebind
ACCEPTED_INPUT_SHA256 = {
    "ranked_lists_uni2apf5_median_pooling.csv.gz": "7ca1eb55de92a9451939bfdfecc84fc1d670895dda6aa03f672f3d670392c20f",
    "ranked_lists_uni2apf15_median_pooling.csv.gz": "fac6394a69d2136843a881e418704013cc9ab418ee328adc4390641f36a394b8",
    "ranked_lists_uni2apf30_median_pooling.csv.gz": "6dba61274c66ab98dcaea0d2248e37695d0e99a4e4a841c08cffd1d3db1bbdb3",
    "ranked_lists_uni2apf50_median_pooling.csv.gz": "7c84387aa842d5dfb7bbf1519f961745737ca68247d28d9038f48956ac4fe462",
    "ranked_lists_uni2apf100_median_pooling.csv.gz": "00f1d2e48e8b48173d31c8ca6019bb1f4a99e38455c16eca66413971a688bb7e",
    "ranked_lists_uni2apf5_yottixel.csv.gz": "c001c15f8ebbe96cdb9ca4bbd5e331b36117205fe3aa6bdc6f1bba6749c46c14",
    "ranked_lists_uni2apf15_yottixel.csv.gz": "f0c22c03cb250605bb72e0645baa037bd0b5ea2ae437d3db6b4ded3b971d706f",
    "ranked_lists_uni2apf30_yottixel.csv.gz": "48e24d605831d129296a2029ef54d017db8bbb8174af052fbaffc7b3c9d2f955",
    "ranked_lists_uni2apf15_ot_cosine.csv.gz": "7b6c3e6df67a4873e60ebe308fa041a59e01abfd282881ab65be43261e91f22d",
    "ranked_lists_uni2apf30_ot_cosine.csv.gz": "e8ca2067b72708678cbc99f8aa5f38704b5f6f5973596151fe0475ea5df1aef2",
}
MAX_TIE_ARRANGEMENTS = 200000
RUNGS = [  # tag, family, retention %
    ("uni2apf5_median_pooling", "median_pooling", 5),
    ("uni2apf15_median_pooling", "median_pooling", 15),
    ("uni2apf30_median_pooling", "median_pooling", 30),
    ("uni2apf50_median_pooling", "median_pooling", 50),
    ("uni2apf100_median_pooling", "median_pooling", 100),
    ("uni2apf5_yottixel", "yottixel", 5),
    ("uni2apf15_yottixel", "yottixel", 15),
    ("uni2apf30_yottixel", "yottixel", 30),
    ("uni2apf15_ot_cosine", "ot_cosine", 15),
    ("uni2apf30_ot_cosine", "ot_cosine", 30),
]
CONTRASTS = [
    ("uni2apf5_median_pooling", "uni2apf15_median_pooling"),
    ("uni2apf30_median_pooling", "uni2apf15_median_pooling"),
    ("uni2apf50_median_pooling", "uni2apf15_median_pooling"),
    ("uni2apf100_median_pooling", "uni2apf15_median_pooling"),
    ("uni2apf5_yottixel", "uni2apf15_yottixel"),
    ("uni2apf30_yottixel", "uni2apf15_yottixel"),
    ("uni2apf15_ot_cosine", "uni2apf15_median_pooling"),
    ("uni2apf15_ot_cosine", "uni2apf15_yottixel"),
    ("uni2apf30_ot_cosine", "uni2apf15_ot_cosine"),
    ("uni2apf30_ot_cosine", "uni2apf30_median_pooling"),
    ("uni2apf30_ot_cosine", "uni2apf30_yottixel"),
]
METRICS = ["hit1", "hit3", "hit5", "hit10", "precision5", "precision10", "plurality5"]
N_SPOT = 25
TOL = 1e-12
DRAWS_NOTE = ("make_patient_draws copied verbatim from scripts/r2_3_internal_failure_cis.py (accepted, frozen), "
              "itself copied from prosper-search prosper/evaluation/r2_1_hoptimus_main.py; identical sampling")


def make_patient_draws(patient_ids, count=5000, seed=123):
    count, seed = int(count), int(seed)
    if any(not isinstance(v, str) or not v.strip() for v in patient_ids):
        raise ValueError("patient_ids must contain nonempty string IDs")
    patients = tuple(sorted(set(patient_ids)))
    p = len(patients)
    multiplicities = np.zeros((count, p), dtype=np.int64)
    if p:
        rng = np.random.Generator(np.random.PCG64(seed))
        for b in range(count):
            multiplicities[b] = np.bincount(rng.integers(p, size=p), minlength=p)
    return patients, multiplicities


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def plurality(labels):
    c = Counter(labels)
    mx = max(c.values())
    return next(lab for lab in labels if c[lab] == mx)


def load_meta(path):
    data = read_bytes(path)
    meta = pd.read_csv(io.BytesIO(data), dtype=str)
    for col in ("Slide.Scan.File", "UID.Subject", "Tumor.group"):
        if col not in meta.columns:
            raise ValueError(f"metadata lacks column {col}")
    if meta["Slide.Scan.File"].duplicated().any():
        raise ValueError("metadata: a Slide.Scan.File maps to more than one row")
    if meta["UID.Subject"].isna().any() or meta["UID.Subject"].str.strip().eq("").any():
        raise ValueError("metadata: missing UID.Subject")
    return meta.set_index("Slide.Scan.File"), sha256_bytes(data)


def load_ranks(path, meta):
    data = read_bytes(path)
    with gzip.open(io.BytesIO(data), "rt") as f:
        r = pd.read_csv(f, dtype={"query": str, "neighbour": str, "true_label": str, "neighbour_label": str},
                        keep_default_na=False, float_precision="round_trip")
    need = ["query", "true_label", "rank", "neighbour", "neighbour_label", "distance"]
    if list(r.columns) != need:
        raise ValueError(f"{path}: columns {list(r.columns)} != {need}")
    r = r.sort_values(["query", "rank"], kind="mergesort").reset_index(drop=True)
    sizes = r.groupby("query").size()
    if not (sizes == DEPTH).all():
        raise ValueError(f"{path}: not every query has exactly {DEPTH} neighbours")
    if not (r.groupby("query")["rank"].apply(lambda s: list(s) == list(range(1, DEPTH + 1)))).all():
        raise ValueError(f"{path}: ranks are not 1..{DEPTH} per query")
    if not r["distance"].map(np.isfinite).all():
        raise ValueError(f"{path}: non-finite distance")
    # checks ported from the accepted r2_3_internal_failure_tables.run_checks
    m = meta
    ids = set(r["query"]) | set(r["neighbour"])
    missing = ids - set(m.index)
    if missing:
        raise ValueError(f"{path}: {len(missing)} query/neighbour IDs absent from metadata, e.g. {sorted(missing)[:3]}")
    qset = set(r["query"])
    outside = set(r["neighbour"]) - qset
    if outside:
        raise ValueError(f"{path}: {len(outside)} neighbour slides are not in the query/library set, e.g. {sorted(outside)[:3]}")
    if (r["true_label"] != r["query"].map(m["Tumor.group"])).any():
        raise ValueError(f"{path}: true_label disagrees with metadata Tumor.group for some query")
    if (r["neighbour_label"] != r["neighbour"].map(m["Tumor.group"])).any():
        raise ValueError(f"{path}: neighbour_label disagrees with metadata Tumor.group for some neighbour")
    labs = set(r["true_label"]) | set(r["neighbour_label"])
    if labs != set(CLASSES):
        raise ValueError(f"{path}: label set {sorted(labs)} != expected {sorted(CLASSES)}")
    if r.duplicated(["query", "neighbour"]).any():
        raise ValueError(f"{path}: duplicated (query, neighbour) pair")
    if (r["query"] == r["neighbour"]).any():
        raise ValueError(f"{path}: a query is its own neighbour")
    if (r.groupby("query")["true_label"].nunique() != 1).any():
        raise ValueError(f"{path}: true_label varies within a query")
    same = r["query"].map(m["UID.Subject"]) == r["neighbour"].map(m["UID.Subject"])
    if same.any():
        raise ValueError(f"{path}: {int(same.sum())} neighbours share UID.Subject with their query")
    return r, sha256_bytes(data)


def distinct_perms(seq):
    """Distinct permutations of a label multiset (order of first appearance matters for the tie-break)."""
    def rec(counter, n):
        if n == 0:
            yield ()
            return
        for lab in sorted(counter):
            if counter[lab]:
                counter[lab] -= 1
                for rest in rec(counter, n - 1):
                    yield (lab,) + rest
                counter[lab] += 1
    return rec(Counter(seq), len(seq))


def top5_arrangements(labs, d):
    """Yield every distinct top-5 label sequence obtainable by reordering neighbours inside exact-distance tie
    blocks that intersect ranks 1-5 (positions of blocks fully outside the top-5 are irrelevant). Returns
    None when the arrangement count would exceed MAX_TIE_ARRANGEMENTS."""
    blocks = []
    i = 0
    while i < 5:
        j = i
        while j < len(d) and d[j] == d[i]:
            j += 1
        blocks.append((i, j))  # [i, j) share one distance; the last block may extend beyond rank 5
        i = j
    if all(j - i == 1 for i, j in blocks):
        return [tuple(labs[:5])]
    per_block = []
    total = 1
    for i, j in blocks:
        members = labs[i:j]
        slots = min(5, j) - i
        opts = set()
        for combo in itertools.combinations(range(len(members)), slots):
            chosen = [members[c] for c in combo]
            for perm in distinct_perms(chosen):
                opts.add(perm)
        per_block.append(sorted(opts))
        total *= len(opts)
        if total > MAX_TIE_ARRANGEMENTS:
            return None
    return [sum(parts, ()) for parts in itertools.product(*per_block)]


def per_query(r, meta):
    """One row per query with the estimands and tie-bound columns."""
    pat = meta["UID.Subject"]
    rows = []
    for q, g in r.groupby("query", sort=True):
        labs = g["neighbour_label"].tolist()
        d = g["distance"].to_numpy()
        t = g["true_label"].iloc[0]
        if not (np.diff(d) >= 0).all():
            raise ValueError(f"query {q}: distances not non-decreasing with rank")
        row = dict(query=q, true_label=t, patient=pat[q],
                   hit1=int(t in labs[:1]), hit3=int(t in labs[:3]), hit5=int(t in labs[:5]), hit10=int(t in labs[:10]),
                   precision5=sum(l == t for l in labs[:5]) / 5.0, precision10=sum(l == t for l in labs[:10]) / 10.0,
                   plurality5=int(plurality(labs[:5]) == t), plurality5_pred=plurality(labs[:5]))
        d5 = d[4]
        straddles = bool(d[5] == d5)
        any_tie_top6 = bool((np.diff(d[:6]) == 0).any())
        arr = top5_arrangements(labs, d)
        if arr is None:
            lo = {m: np.nan for m in ("hit5", "precision5", "plurality5", "hit1", "hit3")}
            hi = dict(lo)
            enumerated = 0
        else:
            vals = {"hit5": [], "precision5": [], "plurality5": [], "hit1": [], "hit3": []}
            for top in arr:
                vals["hit1"].append(int(t in top[:1]))
                vals["hit3"].append(int(t in top[:3]))
                vals["hit5"].append(int(t in top))
                vals["precision5"].append(sum(l == t for l in top) / 5.0)
                vals["plurality5"].append(int(plurality(list(top)) == t))
            lo = {m: min(v) for m, v in vals.items()}
            hi = {m: max(v) for m, v in vals.items()}
            enumerated = len(arr)
        row.update(hit1_tie_lower=lo["hit1"], hit1_tie_upper=hi["hit1"], hit3_tie_lower=lo["hit3"], hit3_tie_upper=hi["hit3"],
                   hit5_tie_lower=lo["hit5"], hit5_tie_upper=hi["hit5"],
                   precision5_tie_lower=lo["precision5"], precision5_tie_upper=hi["precision5"],
                   plurality5_tie_lower=lo["plurality5"], plurality5_tie_upper=hi["plurality5"],
                   rank56_tie=int(straddles), any_tie_ranks1to6=int(any_tie_top6),
                   tie_ranks9_10=int(d[9] == d[8]), tie_block_reaches_depth=int(straddles and d[DEPTH - 1] == d5),
                   n_tie_arrangements=enumerated)
        rows.append(row)
    return pd.DataFrame(rows).set_index("query")


def weight_matrix(patient_col, B, seed):
    patients, mult = make_patient_draws(list(patient_col), count=B, seed=seed)
    idx = pd.Index(patients).get_indexer(list(patient_col))
    return patients, mult, mult[:, idx].astype(float)


def boot_means(W, v, mask=None):
    if mask is not None:
        W, v = W[:, mask], v[mask]
    den = W.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ v) / np.where(den > 0, den, np.nan)


def macro_reps(W, G, v):
    """(B,) macro mean over groups; NaN replicate if any group has zero weight."""
    num = W @ (G * v).T
    den = W @ G.T
    with np.errstate(invalid="ignore", divide="ignore"):
        per = num / np.where(den > 0, den, np.nan)
    return per.mean(1)


def ci(v):
    v = np.asarray(v, float)
    bad = int((~np.isfinite(v)).sum())
    if bad or not v.size:
        return np.nan, np.nan, bad
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), 0


def boundary_flag(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "ci_unavailable"
    f = [n for n, c in (("lower_eq_0", lo == 0), ("upper_eq_1", hi == 1), ("zero_width", lo == hi)) if c]
    return "|".join(f)


def compute(frames, B, seed):
    base = frames[RUNGS[1][0]]
    patients, mult, W = weight_matrix(base["patient"], B, seed)
    groups = CLASSES
    G = np.array([(base["true_label"] == g).to_numpy() for g in groups]).astype(float)
    if (G.sum(1) == 0).any():
        raise ValueError("a class has no queries")
    n, npat = len(base), len(patients)
    reps = {}   # (tag, metric, pop) -> (B,)
    points = {}
    summary, paired, per_group = [], [], []
    for tag, fam, ret in RUNGS:
        fr = frames[tag]
        for m in METRICS:
            v = fr[m].to_numpy(float)
            micro_pt = float(v.mean())
            macro_pt = float(np.mean([v[G[i] > 0].mean() for i in range(len(groups))]))
            reps[(tag, m, "micro")] = boot_means(W, v)
            reps[(tag, m, "macro")] = macro_reps(W, G, v)
            points[(tag, m, "micro")], points[(tag, m, "macro")] = micro_pt, macro_pt
            for pop in ("micro", "macro"):
                lo, hi, bad = ci(reps[(tag, m, pop)])
                summary.append(dict(rung=tag, family=fam, retention_pct=ret, metric=m, population=pop,
                                    point=points[(tag, m, pop)], ci_lo=lo, ci_hi=hi, n_queries=n, n_patients=npat,
                                    n_replicates=B, n_nonfinite=bad, flag=boundary_flag(lo, hi)))
        for m in ("hit5", "precision5", "hit10", "plurality5"):
            v = fr[m].to_numpy(float)
            for i, g in enumerate(groups):
                mask = G[i] > 0
                r = boot_means(W, v, mask)
                lo, hi, bad = ci(r)
                per_group.append(dict(rung=tag, family=fam, retention_pct=ret, group=g, metric=m,
                                      n_queries=int(mask.sum()), n_patients=int(fr.loc[mask, "patient"].nunique()),
                                      point=float(v[mask].mean()), ci_lo=lo, ci_hi=hi, n_replicates=B,
                                      n_nonfinite=bad, flag=boundary_flag(lo, hi)))
    meta_of = {t: (f, r) for t, f, r in RUNGS}
    for a, b in CONTRASTS:
        for m in METRICS:
            for pop in ("micro", "macro"):
                d = reps[(a, m, pop)] - reps[(b, m, pop)]
                lo, hi, bad = ci(d)
                paired.append(dict(minuend=a, minuend_family=meta_of[a][0], minuend_retention_pct=meta_of[a][1],
                                   subtrahend=b, subtrahend_family=meta_of[b][0], subtrahend_retention_pct=meta_of[b][1],
                                   metric=m, population=pop,
                                   point=points[(a, m, pop)] - points[(b, m, pop)], ci_lo=lo, ci_hi=hi,
                                   n_queries=n, n_patients=npat, n_replicates=B, n_nonfinite=bad,
                                   ci_includes_zero_nominal=bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi),
                                   definition="minuend - subtrahend on the same queries and the same patient draws; "
                                              "nominal 95%, unadjusted; including 0 is not evidence of equivalence"))
    return dict(patients=patients, mult=mult, W=W, G=G, groups=groups, reps=reps, points=points,
                summary=pd.DataFrame(summary), paired=pd.DataFrame(paired), per_group=pd.DataFrame(per_group))


def tie_bounds(frames):
    rows = []
    for tag, fam, ret in RUNGS:
        fr = frames[tag]
        G = [(fr["true_label"] == g).to_numpy() for g in CLASSES]

        def macro(col):
            return float(np.mean([fr.loc[g, col].mean() for g in G]))
        row = dict(rung=tag, family=fam, retention_pct=ret,
                   queries_any_tie_ranks1to6=int(fr["any_tie_ranks1to6"].sum()),
                   queries_with_rank56_tie=int(fr["rank56_tie"].sum()),
                   queries_tie_ranks9_10=int(fr["tie_ranks9_10"].sum()),
                   queries_tie_block_reaches_depth=int(fr["tie_block_reaches_depth"].sum()),
                   queries_enumeration_capped=int((fr["n_tie_arrangements"] == 0).sum()),
                   max_tie_arrangements=int(fr["n_tie_arrangements"].max()))
        for m in ("hit1", "hit3"):
            row[f"queries_{m}_tie_dependent"] = int((fr[f"{m}_tie_lower"] != fr[f"{m}_tie_upper"]).sum())
        for m in ("hit5", "precision5", "plurality5"):
            row[f"queries_{m}_tie_dependent"] = int((fr[f"{m}_tie_lower"] != fr[f"{m}_tie_upper"]).sum())
            row.update({f"macro_{m}_lower": macro(f"{m}_tie_lower"), f"macro_{m}_saved": macro(m),
                        f"macro_{m}_upper": macro(f"{m}_tie_upper"),
                        f"micro_{m}_lower": float(fr[f"{m}_tie_lower"].mean()), f"micro_{m}_saved": float(fr[m].mean()),
                        f"micro_{m}_upper": float(fr[f"{m}_tie_upper"].mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def spot_check(frames, res, B):
    """Materialise N_SPOT replicates from the explicit patient -> row map and recompute, by brute force, micro/macro
    hit@5, precision@10 and plurality@5 on three rungs plus the paired hit@5 difference 100% - 15% (median), then
    compare with the matrix-weighted replicate values."""
    base = frames[RUNGS[1][0]]
    rows_of = {p: np.flatnonzero((base["patient"] == p).to_numpy()) for p in res["patients"]}
    grp = base["true_label"].to_numpy()
    picks = np.linspace(0, B - 1, N_SPOT).astype(int)
    worst = 0.0

    def cmp(val, ref):
        nonlocal worst
        if np.isfinite(val) and np.isfinite(ref):
            worst = max(worst, abs(val - ref))
        elif np.isfinite(val) != np.isfinite(ref):
            worst = np.inf

    for b in picks:
        idx = np.concatenate([np.repeat(rows_of[p], res["mult"][b, i]) for i, p in enumerate(res["patients"])])
        g = grp[idx]
        complete = all((g == c).any() for c in CLASSES)
        brute = {}
        for tag in ("uni2apf15_median_pooling", "uni2apf100_median_pooling", "uni2apf15_yottixel"):
            for m in ("hit5", "precision10", "plurality5"):
                v = frames[tag][m].to_numpy(float)[idx]
                micro = v.mean()
                macro = np.mean([v[g == c].mean() for c in CLASSES]) if complete else np.nan
                brute[(tag, m, "micro")], brute[(tag, m, "macro")] = micro, macro
                for pop in ("micro", "macro"):
                    cmp(brute[(tag, m, pop)], res["reps"][(tag, m, pop)][b])
        for pop in ("micro", "macro"):
            d = brute[("uni2apf100_median_pooling", "hit5", pop)] - brute[("uni2apf15_median_pooling", "hit5", pop)]
            ref = res["reps"][("uni2apf100_median_pooling", "hit5", pop)][b] - res["reps"][("uni2apf15_median_pooling", "hit5", pop)][b]
            cmp(d, ref)
    return worst


def population_accounting(frames, meta):
    """Non-MISC metadata slides that are not ladder queries, per group, plus the MISC count in the metadata universe."""
    base = frames[RUNGS[1][0]]
    qset = set(base.index)
    non_misc = meta[meta["Tumor.group"] != "MISC"]
    absent = non_misc[~non_misc.index.isin(qset)]
    return dict(
        ladder_queries=int(len(base)), ladder_library_equals_queries=True, ladder_patients=int(base["patient"].nunique()),
        metadata_rows=int(len(meta)), metadata_misc_slides=int((meta["Tumor.group"] == "MISC").sum()),
        metadata_non_misc_slides=int(len(non_misc)), metadata_non_misc_patients=int(non_misc["UID.Subject"].nunique()),
        non_misc_slides_absent_from_ladder=int(len(absent)),
        non_misc_patients_absent_from_ladder=int(len(set(non_misc["UID.Subject"]) - set(base["patient"]))),
        absent_by_group={g: int(c) for g, c in absent["Tumor.group"].value_counts().sort_index().items()},
        note="The ladder universe (fully-tissue tiles) is NOT the 2,778-query main population; the 2,887-slide count "
             "includes MISC slides that are neither queries nor library members. '100%' retains 100% of the ladder's "
             "fully-tissue tiles, not every tissue tile of the slide.")


def reconcile(frames, res, hashes, meta_sha, saved_points, meta):
    checks = []

    def add(name, ok, **detail):
        checks.append(dict(name=name, status="PASS" if ok else "FAIL", **detail))

    base = frames[RUNGS[1][0]]
    add("n_queries_equals_2750", len(base) == EXPECTED_QUERIES, observed=len(base))
    add("n_patients_equals_1857", len(res["patients"]) == EXPECTED_PATIENTS, observed=int(len(res["patients"])))
    bad_hash = {k: (hashes.get(k), v) for k, v in ACCEPTED_INPUT_SHA256.items() if hashes.get(k) != v}
    add("input_sha256_match_accepted_constants", not bad_hash, mismatches=bad_hash)
    add("all_rungs_same_queries_and_truth",
        all(frames[t].index.equals(base.index) and (frames[t]["true_label"] == base["true_label"]).all()
            and (frames[t]["patient"] == base["patient"]).all() for t, _, _ in RUNGS))
    add("fifteen_groups_present", sorted(base["true_label"].unique()) == sorted(CLASSES),
        observed=sorted(base["true_label"].unique()))
    add("each_replicate_draws_n_patients", bool((res["mult"].sum(1) == len(res["patients"])).all()))
    add("query_weights_equal_patient_multiplicity",
        bool(np.array_equal(res["W"].sum(1), (res["mult"] * base.groupby("patient").size()
                                              .reindex(res["patients"]).to_numpy()).sum(1))))
    worst = spot_check(frames, res, int(res["W"].shape[0]))
    add("materialised_resample_spot_check", bool(worst <= TOL), worst_abs_diff=float(worst), tolerance=TOL,
        n_replicates_checked=N_SPOT)
    # saved point estimates (frozen prosper JSONs, pps intervals ignored) must equal the recomputed points;
    # a missing JSON, missing key or n_queries mismatch is a mismatch, not a skip
    mism = []
    for tag, _, _ in RUNGS:
        sp = saved_points.get(tag)
        if not sp:
            mism.append((tag, "no saved json / no point_estimates"))
            continue
        if int(sp.get("n_queries", -1)) != len(base):
            mism.append((tag, "n_queries", sp.get("n_queries"), len(base)))
        for m, key in (("hit1", "hit_at_1"), ("hit3", "hit_at_3"), ("hit5", "hit_at_5"), ("hit10", "hit_at_10"),
                       ("precision5", "precision_at_5"), ("precision10", "precision_at_10")):
            for pop in ("micro", "macro"):
                k = f"{pop}_{key}"
                if k not in sp:
                    mism.append((tag, k, "missing key"))
                elif abs(float(sp[k]) - res["points"][(tag, m, pop)]) > 1e-9:
                    mism.append((tag, k, float(sp[k]), res["points"][(tag, m, pop)]))
    add("saved_json_points_reproduced", not mism, mismatches=mism[:20], keys_compared=12 * len(RUNGS))
    # paired point = difference of points
    pdf = res["paired"]
    ok = True
    for _, r in pdf.iterrows():
        ok &= abs(r["point"] - (res["points"][(r["minuend"], r["metric"], r["population"])]
                                - res["points"][(r["subtrahend"], r["metric"], r["population"])])) <= TOL
    add("paired_points_equal_point_differences", bool(ok))
    add("per_query_in_unit_interval", all(((frames[t][m] >= 0) & (frames[t][m] <= 1)).all()
                                          for t, _, _ in RUNGS for m in METRICS))
    add("tie_bounds_bracket_saved", all(((frames[t][f"{m}_tie_lower"] <= frames[t][m])
                                         & (frames[t][m] <= frames[t][f"{m}_tie_upper"])).all()
                                        for t, _, _ in RUNGS for m in ("hit1", "hit3", "hit5", "precision5", "plurality5")))
    add("tie_enumeration_not_capped", all((frames[t]["n_tie_arrangements"] > 0).all() for t, _, _ in RUNGS),
        cap=MAX_TIE_ARRANGEMENTS)
    acct = population_accounting(frames, meta)
    add("neighbours_subset_of_queries", True, detail="enforced in load_ranks; equality of library and query set is "
                                                   "not observable from depth-10 lists")
    draw_id = dict(n_patients=int(len(res["patients"])),
                   patient_list_sha256=hashlib.sha256("\n".join(res["patients"]).encode()).hexdigest(),
                   multiplicity_sha256=hashlib.sha256(np.ascontiguousarray(res["mult"]).tobytes()).hexdigest(),
                   multiplicity_dtype=str(res["mult"].dtype), multiplicity_shape=list(res["mult"].shape))
    return dict(status="PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL", checks=checks,
                input_hashes=hashes, accepted_input_sha256=ACCEPTED_INPUT_SHA256, metadata_sha256=meta_sha,
                metadata_path=META, draw_identity=draw_id, population_accounting=acct)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks-dir", default=os.path.join(OUT, "r22_metrics"))
    ap.add_argument("--meta", default=META)
    ap.add_argument("--out-dir", default=os.path.join(OUT, f"r2_2_tile_ladder_whole_patient_{dt.date.today():%Y%m%d}"))
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args(argv)
    if os.path.exists(a.out_dir) and os.listdir(a.out_dir):
        sys.exit(f"refusing to write into non-empty {a.out_dir}")
    os.makedirs(a.out_dir, exist_ok=True)

    meta, meta_sha = load_meta(a.meta)
    frames, hashes, saved_points = {}, {}, {}
    for tag, _, _ in RUNGS:
        path = os.path.join(a.ranks_dir, f"ranked_lists_{tag}.csv.gz")
        r, h = load_ranks(path, meta)
        frames[tag] = per_query(r, meta)
        hashes[os.path.basename(path)] = h
        jp = os.path.join(a.ranks_dir, f"metrics_{tag}.json")
        if os.path.exists(jp):
            jb = read_bytes(jp)
            hashes[os.path.basename(jp)] = sha256_bytes(jb)
            saved_points[tag] = json.loads(jb).get("point_estimates", {})
    res = compute(frames, a.B, a.seed)
    tb = tie_bounds(frames)
    rec = reconcile(frames, res, hashes, meta_sha, saved_points, meta)
    rec.update(config=dict(B=a.B, seed=a.seed, draws=DRAWS_NOTE, depth=DEPTH, classes=CLASSES,
                           population="ladder universe: 2,750 queries = library, fully-tissue tiles, UNI2; NOT the "
                                      f"{MAIN_POPULATION_QUERIES}-query main population (see population_accounting)",
                           rungs=[dict(tag=t, family=f, retention_pct=p) for t, f, p in RUNGS],
                           contrasts=CONTRASTS, ci="percentile 2.5/97.5 over all replicates; nominal 95%, unadjusted for "
                                                   "multiplicity; any non-finite replicate -> interval unavailable",
                           primary_outcomes_adr0021=["hit5", "precision5"], script=os.path.relpath(__file__, R),
                           script_sha256=sha256_bytes(read_bytes(__file__)), generated=dt.datetime.now().isoformat()))

    res["summary"].to_csv(os.path.join(a.out_dir, "ladder_summary.csv"), index=False)
    res["paired"].to_csv(os.path.join(a.out_dir, "paired_differences.csv"), index=False)
    res["per_group"].to_csv(os.path.join(a.out_dir, "per_group.csv"), index=False)
    tb.to_csv(os.path.join(a.out_dir, "tie_bounds.csv"), index=False)
    pq = pd.concat({t: frames[t] for t, _, _ in RUNGS}, names=["rung", "query"]).reset_index()
    pq.to_csv(os.path.join(a.out_dir, "per_query_frame.csv.gz"), index=False, compression="gzip")
    with open(os.path.join(a.out_dir, "reconciliation.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)
    s = res["summary"]
    acct = rec["population_accounting"]
    absent = ", ".join(f"{g} {c}" for g, c in acct["absent_by_group"].items())
    lines = ["# R2.2 tile-retention ladder: whole-patient bootstrap (ADR0021)", "",
             f"Status: {rec['status']}. B={a.B}, seed={a.seed}, {len(res['patients'])} patients, {len(frames[RUNGS[1][0]])} queries, "
             "15 groups incl. NORMAL.", "",
             "## Population", "",
             f"Ladder universe (UNI2, fully-tissue tiles): the library is exactly the {acct['ladder_queries']} queries "
             f"({acct['ladder_patients']} patients). This is NOT the {MAIN_POPULATION_QUERIES}-query main population: "
             f"{acct['non_misc_slides_absent_from_ladder']} non-MISC slides ({acct['non_misc_patients_absent_from_ladder']} "
             f"patients) of the metadata universe are absent from the ladder ({absent}). The metadata also holds "
             f"{acct['metadata_misc_slides']} MISC slides, which are neither queries nor library members; any slide count "
             "larger than the query count is therefore not a library size. '100%' retains 100% of the ladder's "
             "fully-tissue tiles, not every tissue tile.", "",
             "## Macro hit@5 [nominal 95% CI] by rung", ""]
    for _, r in s[(s.metric == "hit5") & (s.population == "macro")].iterrows():
        lines.append(f"- {r.rung}: {r.point:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]")
    tie_note = []
    for _, r in tb.iterrows():
        if r["queries_any_tie_ranks1to6"] > 0:
            tie_note.append(f"- {r.rung}: {int(r.queries_any_tie_ranks1to6)} queries tie somewhere in ranks 1-6, "
                            f"{int(r.queries_tie_ranks9_10)} tie at ranks 9/10; tie-dependent queries hit@1 "
                            f"{int(r.queries_hit1_tie_dependent)}, hit@3 {int(r.queries_hit3_tie_dependent)}, hit@5 "
                            f"{int(r.queries_hit5_tie_dependent)}, plurality@5 {int(r.queries_plurality5_tie_dependent)}; "
                            f"macro plurality@5 "
                            f"{r.macro_plurality5_lower:.4f}-{r.macro_plurality5_upper:.4f} (saved {r.macro_plurality5_saved:.4f}), "
                            f"macro precision@5 {r.macro_precision5_lower:.4f}-{r.macro_precision5_upper:.4f}, "
                            f"macro hit@5 {r.macro_hit5_lower:.4f}-{r.macro_hit5_upper:.4f}")
    lines += ["", "## Reading rules", "",
              "- Intervals are percentile whole-patient bootstrap, nominal 95%, unadjusted for multiplicity "
              f"({len(res['paired'])} paired intervals are reported). An interval that includes 0 is NOT evidence of "
              "equivalence or of 'no gain'. Primary outcomes under ADR0021: hit@5 and precision@5.",
              "- Points are vote-rule independent except plurality@5 (closest-ranked tie-break).",
              "- Tie-order sensitivity (tie_bounds.csv) is exact within-block enumeration over the visible top-10, points "
              "only; ties at ranks 10/11 are unobservable, so hit@10/precision@10 may also depend on tie order. Yottixel "
              "points (hit@1/3/5, precision@5, plurality@5), and every contrast that uses a Yottixel rung, rest on the "
              "arbitrary saved order within ties."]
    no_tie = [r.rung for _, r in tb.iterrows() if r["queries_any_tie_ranks1to6"] == 0 and r["queries_tie_ranks9_10"] == 0]
    lines.append(f"- Rungs with no distance tie anywhere in the visible top 10: {', '.join(no_tie) if no_tie else 'none'}.")
    if tie_note:
        lines += ["", "Rungs with ties:"] + tie_note
    lines += ["", "Files: ladder_summary.csv, paired_differences.csv, per_group.csv, tie_bounds.csv, "
              "per_query_frame.csv.gz, reconciliation.json.",
              "These whole-patient intervals replace the pps_cluster_one_slide intervals saved in "
              "r22_metrics/metrics_*.json (same points, checked in reconciliation.json). Other directories are not "
              "inspected by this script; their status is recorded in the ledger."]
    with open(os.path.join(a.out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(rec["status"], a.out_dir)
    for c in rec["checks"]:
        print(f"  {c['status']:4s} {c['name']}")
    return 0 if rec["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
