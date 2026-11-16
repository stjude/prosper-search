#!/usr/bin/env python3
"""R3.1 (E4): whole-patient bootstrap 95% CIs for the comparator / ablation configurations from SAVED ranked lists.

A review comment asked for direct comparison with existing WSI retrieval systems, simpler
baselines and the incremental contribution of each module. Every such comparison already exists as a depth-10
ranked list on the main all-tissue population (2,778 queries, 15 groups incl. NORMAL, MISC dropped, same-patient
masked), but the intervals saved beside them (c2_metric_suite/metrics_*.json, c14_sish/, c5_resnet50/,
c5_ladder/) use the retired `pps_cluster_one_slide` sampler (B = 2,000 or 5,000) and are not quotable under
[decision record]. This script recomputes points and intervals for the following configurations under the accepted
whole-patient bootstrap (patients drawn with replacement, B = 5,000, PCG64 seed 123, every query weighted by its
patient's draw multiplicity; identical draws for all configurations, so paired differences are exact), reusing the
accepted functions of scripts/r2_2_tile_ladder_whole_patient.py (imported, not copied). No retrieval is re-run;
no feature, model or package code is touched.

Main population (2,778 queries = library):
  primary      uni2 + exact OT (cosine)           c2_metric_suite/ranked_lists_uni2_ot_cosine.csv.gz
  median       uni2 + median pooling              c2_metric_suite/ranked_lists_uni2_median_pooling.csv.gz   (ladder rung 3-proxy)
  yottixel     uni2 + Yottixel-style aggregation  c2_metric_suite/ranked_lists_uni2_yottixel.csv.gz
  sish         SISH full system (C14, [decision record])   c2_metric_suite/ranked_lists_sish_system.csv.gz
  resnet50     ImageNet ResNet-50 + median pooling (C5 floor)  c5_resnet50/ranked_lists_resnet50_median_pooling.csv.gz
Tumour-only sub-population (2,604 queries, 14 groups; ladder rung 2 -> 3):
  primary_tumour  uni2 + OT on the tumour-only library         c2_metric_suite/ranked_lists_uni2_ot_cosine_tumor_only.csv.gz
  primary_sub     the all-tissue primary ranked lists restricted to the same 2,604 queries (library unchanged:
                  all 2,778 slides incl. NORMAL) -- this is the saved rung-2 side of c5_ladder/rung_2to3.json.
Not recomputed (no ranked-list basis): ladder rung 4 -> 5 (Bayesian scoring) needs per-query kNN posteriors that
were never saved; its saved point estimates are copied into reconciliation.json for reference only and are labelled
pps-legacy. Rung 1 -> 2 (mosaic selection) was never runnable (c5_module_ladder.md). TITAN (R2.6) already has
accepted whole-patient intervals from the 10-04 thin reset and is not duplicated here.

SISH padding rule (disclosed): the SISH ranked list has 35 queries with fewer than 10 neighbours (1 empty, 34 short;
206 rows padded with neighbour "__NONE__", empty distance). Padded positions are counted as NON-hits for hit@k and
precision@k and are excluded from the plurality@5 vote (vote among the available labels; a query with no neighbours
is a plurality miss). This is the most favourable reading consistent with "nothing was retrieved"; the per-query
frame carries `n_padded` so a reader can re-score. The tumour-only primary list has one query (16738, RB) whose ten rows carry neighbour IDs but EMPTY distances
(no valid tumour-only similarity); those rows are treated the same way (non-hit, excluded from the vote) and the
query is kept in the population. Its listed neighbours are all non-RB, so hit@k/precision@k are unchanged by this
rule; plurality@5 becomes a miss either way. The other configurations have no padding.

Estimands per configuration (per query, then micro = mean over queries, macro = mean of group means):
  hit@k (k = 1,3,5,10), precision@k (k = 5,10), plurality@5 (A-3 rule; ties -> closest-ranked tied label).
Paired differences (same queries, same patient draws), minuend - subtrahend:
  sish - primary; resnet50 - primary; median - primary (= -(rung 3 -> 4 OT increment)); yottixel - primary;
  resnet50 - median (encoder increment at fixed median aggregation);
  primary_tumour - primary_sub (rung 2 -> 3 tumour-only filtering; 2,604 queries, 14-group macro).
Per-group hit@5 / plurality@5 for every configuration.
Intervals are nominal 95%, unadjusted. An interval including 0 is not evidence of equivalence.

Outputs (--out-dir, default $PROSPER_OUT/r3_1_comparators_whole_patient_<YYYYMMDD>/):
  summary.csv, paired_differences.csv, per_group.csv, per_query_frame.csv.gz, reconciliation.json, README.md,
  sha256.txt, COMPLETE
"""
import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
OUT = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
LADDER = os.path.join(HERE, "r2_2_tile_ladder_whole_patient.py")
LADDER_SHA_ACCEPTED = None  # recorded at run time; the accepted hash lives in [analysis plan]

spec = importlib.util.spec_from_file_location("ladder", LADDER)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

CLASSES15 = list(L.CLASSES)
CLASSES14 = [c for c in CLASSES15 if c != "NORMAL"]
DEPTH = 10
PAD = "__NONE__"

MAIN = [
    ("primary", "uni2 + exact OT (cosine)", "c2_metric_suite/ranked_lists_uni2_ot_cosine.csv.gz", "c2_metric_suite/metrics_uni2_ot_cosine.json"),
    ("median", "uni2 + median pooling", "c2_metric_suite/ranked_lists_uni2_median_pooling.csv.gz", "c2_metric_suite/metrics_uni2_median_pooling.json"),
    ("yottixel", "uni2 + Yottixel-style", "c2_metric_suite/ranked_lists_uni2_yottixel.csv.gz", "c2_metric_suite/metrics_uni2_yottixel.json"),
    ("sish", "SISH full system (C14)", "c2_metric_suite/ranked_lists_sish_system.csv.gz", "c2_metric_suite/metrics_sish_system.json"),
    ("resnet50", "ImageNet ResNet-50 + median pooling (C5 floor)", "c5_resnet50/ranked_lists_resnet50_median_pooling.csv.gz", "c5_resnet50/metrics_resnet50_median_pooling.json"),
]
TUMOUR = ("primary_tumour", "uni2 + exact OT, tumour-only library (2,604)", "c2_metric_suite/ranked_lists_uni2_ot_cosine_tumor_only.csv.gz", "c2_metric_suite/metrics_uni2_ot_cosine_tumor_only.json")
CONTRASTS_MAIN = [("sish", "primary"), ("resnet50", "primary"), ("median", "primary"), ("yottixel", "primary"), ("resnet50", "median")]
METRICS = ["hit1", "hit3", "hit5", "hit10", "precision5", "precision10", "plurality5"]
LEGACY_RUNG = "c5_ladder/rung_4to5.json"


def load_ranks_padded(path, meta, classes):
    """Accepted loader for unpadded lists; a disclosed relaxation for the SISH '__NONE__' padding."""
    data = L.read_bytes(path)
    with gzip.open(io.BytesIO(data), "rt") as f:
        r = pd.read_csv(f, dtype={"query": str, "neighbour": str, "true_label": str, "neighbour_label": str},
                        keep_default_na=False, float_precision="round_trip")
    padded = r["neighbour_label"].eq(PAD) | r["neighbour"].eq("") | r["distance"].astype(str).eq("")
    if not padded.any():
        rr, sha = L.load_ranks(path, meta)
        rr["padded"] = 0
        return rr, sha, 0
    # SISH: validate the unpadded rows with the accepted checks on a reconstructed frame, then re-attach padding
    r = r.sort_values(["query", "rank"], kind="mergesort").reset_index(drop=True)
    padded = r["neighbour_label"].eq(PAD) | r["neighbour"].eq("") | r["distance"].astype(str).eq("")   # recomputed after the sort
    r["padded"] = padded.astype(int)
    real = r[~padded].copy()
    real["distance"] = pd.to_numeric(real["distance"], errors="raise")
    if not real["distance"].map(np.isfinite).all():
        raise ValueError(f"{path}: non-finite distance among unpadded rows")
    m = meta
    ids = set(real["query"]) | set(real["neighbour"]) | set(r["query"])
    missing = ids - set(m.index)
    if missing:
        raise ValueError(f"{path}: {len(missing)} IDs absent from metadata")
    if (set(real["neighbour"]) - set(r["query"])):
        raise ValueError(f"{path}: neighbours outside the query/library set")
    if (r["true_label"] != r["query"].map(m["Tumor.group"])).any():
        raise ValueError(f"{path}: true_label disagrees with metadata")
    if (real["neighbour_label"] != real["neighbour"].map(m["Tumor.group"])).any():
        raise ValueError(f"{path}: neighbour_label disagrees with metadata")
    if real.duplicated(["query", "neighbour"]).any() or (real["query"] == real["neighbour"]).any():
        raise ValueError(f"{path}: duplicate pair or self-neighbour")
    same = real["query"].map(m["UID.Subject"]) == real["neighbour"].map(m["UID.Subject"])
    if same.any():
        raise ValueError(f"{path}: {int(same.sum())} same-patient neighbours")
    sizes = r.groupby("query").size()
    if not (sizes == DEPTH).all():
        raise ValueError(f"{path}: not every query has exactly {DEPTH} rows")
    # padding must be a suffix of each query's list
    bad = r.groupby("query")["padded"].apply(lambda s: list(s) != sorted(s)).sum()
    if bad:
        raise ValueError(f"{path}: {bad} queries have padding before a real neighbour")
    r["distance"] = pd.to_numeric(r["distance"], errors="coerce")
    r.loc[padded, "distance"] = np.inf
    labs = set(r["true_label"]) | set(real["neighbour_label"])
    if labs != set(classes):
        raise ValueError(f"{path}: label set {sorted(labs)} != {sorted(classes)}")
    return r, L.sha256_bytes(data), int(r.groupby("query")["padded"].sum().gt(0).sum())


def per_query_simple(r, meta):
    pat = meta["UID.Subject"]
    rows = []
    for q, g in r.groupby("query", sort=True):
        labs_all = g["neighbour_label"].tolist()
        padded = g["padded"].to_numpy()
        labs = [l if not p else None for l, p in zip(labs_all, padded)]
        t = g["true_label"].iloc[0]
        d = g["distance"].to_numpy(float)
        if not (np.diff(d[np.isfinite(d)]) >= 0).all():
            raise ValueError(f"query {q}: distances not non-decreasing with rank")
        avail5 = [l for l in labs[:5] if l is not None]
        rows.append(dict(query=q, true_label=t, patient=pat[q], n_padded=int(padded.sum()),
                         hit1=int(t in labs[:1]), hit3=int(t in labs[:3]), hit5=int(t in labs[:5]), hit10=int(t in labs[:10]),
                         precision5=sum(l == t for l in labs[:5]) / 5.0, precision10=sum(l == t for l in labs[:10]) / 10.0,
                         plurality5=int(bool(avail5) and L.plurality(avail5) == t),
                         plurality5_pred=(L.plurality(avail5) if avail5 else "")))
    return pd.DataFrame(rows).set_index("query")


def boot_block(frames, order, classes, B, seed, contrasts, pop_tag):
    base = frames[order[0]]
    for tag in order[1:]:
        if not frames[tag].index.equals(base.index):
            raise ValueError(f"{tag}: query index differs from {order[0]}")
        if not (frames[tag]["patient"] == base["patient"]).all():
            raise ValueError(f"{tag}: patient column differs")
    patients, mult, W = L.weight_matrix(base["patient"], B, seed)
    G = np.array([(base["true_label"] == g).to_numpy() for g in classes]).astype(float)
    if (G.sum(1) == 0).any():
        raise ValueError("a class has no queries")
    n, npat = len(base), len(patients)
    reps, points, summary, paired, per_group = {}, {}, [], [], []
    for tag in order:
        fr = frames[tag]
        for m in METRICS:
            v = fr[m].to_numpy(float)
            points[(tag, m, "micro")] = float(v.mean())
            points[(tag, m, "macro")] = float(np.mean([v[G[i] > 0].mean() for i in range(len(classes))]))
            reps[(tag, m, "micro")] = L.boot_means(W, v)
            reps[(tag, m, "macro")] = L.macro_reps(W, G, v)
            for pop in ("micro", "macro"):
                lo, hi, bad = L.ci(reps[(tag, m, pop)])
                summary.append(dict(population=pop_tag, config=tag, metric=m, average=pop, point=points[(tag, m, pop)],
                                    ci_lo=lo, ci_hi=hi, n_queries=n, n_patients=npat, n_groups=len(classes),
                                    n_replicates=B, n_nonfinite=bad, flag=L.boundary_flag(lo, hi)))
        for m in ("hit5", "precision5", "hit10", "plurality5"):
            v = fr[m].to_numpy(float)
            for i, g in enumerate(classes):
                mask = G[i] > 0
                lo, hi, bad = L.ci(L.boot_means(W, v, mask))
                per_group.append(dict(population=pop_tag, config=tag, group=g, metric=m, n_queries=int(mask.sum()),
                                      n_patients=int(fr.loc[mask, "patient"].nunique()), point=float(v[mask].mean()),
                                      ci_lo=lo, ci_hi=hi, n_replicates=B, n_nonfinite=bad, flag=L.boundary_flag(lo, hi)))
    for a, b in contrasts:
        for m in METRICS:
            for pop in ("micro", "macro"):
                dd = reps[(a, m, pop)] - reps[(b, m, pop)]
                lo, hi, bad = L.ci(dd)
                paired.append(dict(population=pop_tag, minuend=a, subtrahend=b, metric=m, average=pop,
                                   point=points[(a, m, pop)] - points[(b, m, pop)], ci_lo=lo, ci_hi=hi,
                                   n_queries=n, n_patients=npat, n_replicates=B, n_nonfinite=bad,
                                   ci_includes_zero_nominal=bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi),
                                   definition="minuend - subtrahend on the same queries and the same patient draws; "
                                              "nominal 95%, unadjusted; including 0 is not evidence of equivalence"))
    return dict(summary=pd.DataFrame(summary), paired=pd.DataFrame(paired), per_group=pd.DataFrame(per_group),
                n_queries=n, n_patients=npat, points=points)


def saved_points(path):
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        return None
    pe = d.get("point_estimates", {})
    return {k: pe.get(k) for k in ("micro_hit_at_5", "macro_hit_at_5", "micro_hit_at_10", "macro_hit_at_10",
                                    "micro_precision_at_5", "macro_precision_at_5")}, d.get("config", {}).get("bootstrap_mode")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=L.META)
    ap.add_argument("--out-dir", default=os.path.join(OUT, f"r3_1_comparators_whole_patient_{dt.date.today():%Y%m%d}"))
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=False)
    meta, meta_sha = L.load_meta(a.meta)
    ladder_sha = L.sha256_bytes(L.read_bytes(LADDER))
    hashes, frames, padded_n, saved = {}, {}, {}, {}
    for tag, desc, rel, mj in MAIN:
        r, sha, npad = load_ranks_padded(os.path.join(OUT, rel), meta, CLASSES15)
        hashes[rel] = sha
        padded_n[tag] = npad
        frames[tag] = per_query_simple(r, meta)
        if len(frames[tag]) != L.MAIN_POPULATION_QUERIES:
            raise ValueError(f"{tag}: {len(frames[tag])} queries != {L.MAIN_POPULATION_QUERIES}")
        saved[tag] = saved_points(os.path.join(OUT, mj))
    tag, desc, rel, mj = TUMOUR
    r, sha, npad = load_ranks_padded(os.path.join(OUT, rel), meta, CLASSES14)
    hashes[rel] = sha
    padded_n[tag] = npad
    ft = per_query_simple(r, meta)
    saved[tag] = saved_points(os.path.join(OUT, mj))
    if set(ft["true_label"]) != set(CLASSES14):
        raise ValueError("tumour-only label set unexpected")
    sub = frames["primary"].loc[ft.index]
    if not (sub["true_label"] == ft["true_label"]).all() or not (sub["patient"] == ft["patient"]).all():
        raise ValueError("tumour-only queries do not align with the primary frame")
    main_block = boot_block(frames, [t for t, *_ in MAIN], CLASSES15, a.B, a.seed, CONTRASTS_MAIN, "all_tissue_2778")
    tum_block = boot_block({"primary_tumour": ft, "primary_sub": sub}, ["primary_tumour", "primary_sub"], CLASSES14, a.B, a.seed,
                           [("primary_tumour", "primary_sub")], "tumour_only_2604")
    summary = pd.concat([main_block["summary"], tum_block["summary"]], ignore_index=True)
    paired = pd.concat([main_block["paired"], tum_block["paired"]], ignore_index=True)
    per_group = pd.concat([main_block["per_group"], tum_block["per_group"]], ignore_index=True)
    frame = pd.concat({t: f for t, f in list(frames.items()) + [("primary_tumour", ft), ("primary_sub", sub)]}, names=["config", "query"])
    # reconciliation: points vs saved legacy JSON points (micro/macro hit@5/10, precision@5), legacy rung 4->5 copy
    rec_points = []
    for t in list(frames) + ["primary_tumour"]:
        sp = saved.get(t)
        if not sp or sp[0] is None:
            continue
        pts, mode = sp
        for key, val in pts.items():
            if val is None:
                continue
            avg, metric = key.split("_", 1)
            metric = metric.replace("_at_", "")
            mine = main_block["points"].get((t, metric, avg)) if t != "primary_tumour" else tum_block["points"].get((t, metric, avg))
            rec_points.append(dict(config=t, metric=metric, average=avg, saved_point=val, recomputed_point=mine,
                                   abs_diff=(None if mine is None else abs(mine - val)), saved_bootstrap_mode=mode))
    legacy = None
    try:
        lj = json.load(open(os.path.join(OUT, LEGACY_RUNG)))
        legacy = dict(file=LEGACY_RUNG, sha256=L.sha256_bytes(L.read_bytes(os.path.join(OUT, LEGACY_RUNG))),
                      config=lj.get("config"), point_estimates=lj.get("point_estimates"),
                      note="rung 4->5 (Bayesian scoring) has no ranked-list basis; saved points copied for reference, "
                           "intervals in that file are pps_cluster_one_slide legacy and NOT quotable")
    except FileNotFoundError:
        pass
    recon = dict(
        script=os.path.relpath(__file__, R), script_sha256=L.sha256_bytes(L.read_bytes(__file__)),
        imported_ladder_script=os.path.relpath(LADDER, R), imported_ladder_sha256=ladder_sha,
        metadata=a.meta, metadata_sha256=meta_sha, inputs_sha256=hashes, B=a.B, seed=a.seed,
        bootstrap="ADR0021 whole-patient: patients with replacement, identical draws for all configurations in a block",
        populations=dict(all_tissue_2778=dict(n_queries=main_block["n_queries"], n_patients=main_block["n_patients"], n_groups=15),
                         tumour_only_2604=dict(n_queries=tum_block["n_queries"], n_patients=tum_block["n_patients"], n_groups=14,
                                               note="primary_sub = all-tissue primary lists restricted to the 2,604 tumour queries; "
                                                    "library differs between the two sides (2,778 incl. NORMAL vs 2,604 tumour-only)")),
        sish_padding=dict(rule="padded positions = non-hit for hit@k/precision@k; excluded from plurality@5 vote",
                          queries_with_padding={t: int(v) for t, v in padded_n.items()},
                          sish_rows_padded=int(frames["sish"]["n_padded"].sum()),
                          sish_queries_zero_neighbours=int((frames["sish"]["n_padded"] == DEPTH).sum())),
        saved_point_reconciliation=rec_points,
        max_abs_point_diff=max([x["abs_diff"] for x in rec_points if x["abs_diff"] is not None] or [None]),
        legacy_rung_4to5=legacy,
        not_recomputed=["rung 1->2 mosaic selection (never runnable)", "rung 4->5 Bayesian scoring (no saved posteriors)",
                        "TITAN (accepted whole-patient intervals exist from the 10-04 thin reset)"],
        generated=dt.datetime.now().isoformat(timespec="seconds"),
    )
    o = a.out_dir
    summary.to_csv(os.path.join(o, "summary.csv"), index=False)
    paired.to_csv(os.path.join(o, "paired_differences.csv"), index=False)
    per_group.to_csv(os.path.join(o, "per_group.csv"), index=False)
    frame.to_csv(os.path.join(o, "per_query_frame.csv.gz"), compression="gzip")
    json.dump(recon, open(os.path.join(o, "reconciliation.json"), "w"), indent=2, default=str)
    readme = [__doc__, "\n## Run\n", f"B = {a.B}, seed = {a.seed}; {recon['generated']}",
              f"max |saved point - recomputed point| over hit@5/10, precision@5: {recon['max_abs_point_diff']}",
              f"SISH queries with padding: {padded_n['sish']} (rows {recon['sish_padding']['sish_rows_padded']}; "
              f"zero-neighbour queries {recon['sish_padding']['sish_queries_zero_neighbours']})",
              "\n## Headline paired differences (macro, hit@5 / precision@5)\n"]
    for _, row in paired[(paired["average"] == "macro") & (paired["metric"].isin(["hit5", "precision5"]))].iterrows():
        readme.append(f"- {row['minuend']} - {row['subtrahend']} [{row['population']}] {row['metric']}: "
                      f"{row['point']:+.3f} [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]")
    open(os.path.join(o, "README.md"), "w").write("\n".join(readme) + "\n")
    with open(os.path.join(o, "sha256.txt"), "w") as f:
        for fn in sorted(os.listdir(o)):
            if fn in ("sha256.txt", "COMPLETE"):
                continue
            f.write(f"{L.sha256_bytes(L.read_bytes(os.path.join(o, fn)))}  {fn}\n")
    open(os.path.join(o, "COMPLETE"), "w").write(recon["generated"] + "\n")
    print(open(os.path.join(o, "README.md")).read().split("## Run")[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
