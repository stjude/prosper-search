#!/usr/bin/env python3
"""R3.7 fine-label re-evaluation of the saved S6 depth-10 rank lists.

Conditional fine-label retrieval: each query is ranked against the known-fine-label
sub-library of its diagnostic family (query patient excluded). This script does
NOT re-rank; it re-scores the saved rank lists uniformly with non-oracle rules and
the agreed [decision record] whole-patient bootstrap, and reconciles with the legacy S6 JSONs.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
INV = os.path.join(O, "s6_0_label_inventory")
FROZEN_META = ("/PROSPER_WORKSPACE/comet-feature-gen/"
               "comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv")
KIDRARE_MAN = os.path.join(O, "external_kidrare", "manifest_inscope.csv")
KIDRARE_MAP = {"Poorly differentiated neuroblastoma": "poorly_differentiated",
               "Differentiating neuroblastoma": "differentiating",
               "Ganglioneuroblastoma, intermixed": "gnb_intermixed"}
DEPTH, ABSTAIN = 10, "ABSTAIN"
DRAWS_SOURCE = ("make_patient_draws from prosper-search prosper/evaluation/r2_1_hoptimus_main.py, an UNTRACKED "
                "working-tree file (repository HEAD 0edc479 does not contain it); bound only by its sha256 "
                "dee7d68a8b9fcee61d2e845412afdaf20455388ae5c2e61844d4b21d73811faf")
KIDRARE_UNIT = "slide (patient linkage unknown; slide-resampling CI, may be too narrow)"
KIDRARE_BLOCK = "s6_5_kidrare_nb/block_kidrare_nb_uni2_ot_cosine.pkl (135x303)"

# kind: internal (COMET query+reference, manifest labels), fusion (COMET, assay labels from
# saved per_query), ccdi (CCDI queries -> COMET reference), kidrare (within-KidRare, slide units)
ARMS = {
    "s61_int_2c": dict(d="s6_1_rms", kind="internal", man="s61_rms_comet", labels=["ERMS", "ARMS"], pos="ARMS"),
    "s61_int_3c": dict(d="s6_1_rms", kind="internal", man="s61_rms_comet", labels=["ERMS", "ARMS", "Spindle"]),
    "s61_int_2c_xeno": dict(d="s6_1_rms", kind="internal", man="s61_rms_comet", labels=["ERMS", "ARMS"],
                            pos="ARMS", xeno=True),
    "s61_fusion": dict(d="s6_1_rms", kind="fusion", man="s61_rms_comet",
                       labels=["fusion_positive", "fusion_negative"], pos="fusion_positive"),
    "s61_ext_2c": dict(d="s6_1_rms", kind="ccdi", man="s61_rms_comet", labels=["ERMS", "ARMS"], pos="ARMS"),
    "s61_ext_3c": dict(d="s6_1_rms", kind="ccdi", man="s61_rms_comet", labels=["ERMS", "ARMS", "Spindle"]),
    "s62": dict(d="s6_2_mycn", kind="internal", man="s62_mycn_comet", labels=["amplified", "not_amplified"],
                pos="amplified", drop_patients=["SJHBB3FZN"]),
    "s62_xeno": dict(d="s6_2_mycn", kind="internal", man="s62_mycn_comet", labels=["amplified", "not_amplified"],
                     pos="amplified", xeno=True),
    "s64": dict(d="s6_4_hbhcc", kind="internal", man="s64_hbhcc_comet", labels=["HB", "HCC"], pos="HCC"),
    "s65_inpc": dict(d="s6_5_nbdiff", kind="internal", man="s65_inpc_comet", labels=["favorable", "unfavorable"],
                     pos="unfavorable"),
    "s65b_gnb": dict(d="s6_5_nbdiff", kind="internal", man="s65b_gnb_comet", labels=["NB", "GNB"], pos="GNB",
                     relabel={"Ganglioneuroblastoma": "GNB"}),
    "s65_ext_3c": dict(d="s6_5_kidrare_nb", kind="kidrare",
                       labels=["poorly_differentiated", "differentiating", "gnb_intermixed"]),
    "s65_ext_binary": dict(d="s6_5_kidrare_nb", kind="kidrare", labels=["poorly_differentiated", "differentiating"],
                           pos="poorly_differentiated"),
}
BROAD = {"s61_rms_comet": "RMS", "s62_mycn_comet": "NB", "s64_hbhcc_comet": "LIVER",
         "s65_inpc_comet": "NB", "s65b_gnb_comet": "NB"}
METRICS = ["hit1", "hit5", "precision5", "plurality5_recall", "strict5_recall", "strict5_abstain_rate",
           "first_rank_absent_rate"]


def make_patient_draws(patient_ids, count=5000, seed=123):
    """Copied from DRAWS_SOURCE (package import pulls heavy deps); identical sampling.
    Only the input validators (_integer/_ids) are reduced to int()/str checks."""
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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plurality(labs):
    """Most frequent label; tie -> the tied label at the closest rank."""
    c = Counter(labs)
    top = max(c.values())
    return next(lab for lab in labs if c[lab] == top)


def strict_majority(labs):
    lab, n = Counter(labs).most_common(1)[0]
    return lab if n > len(labs) // 2 else ABSTAIN


def check_rank_list(rl, q_patient, ref_label, ref_patient, labels):
    """Raises on structural/eligibility violations; returns #query-neighbour pairs checked."""
    ranks = rl.groupby("query")["rank"].apply(lambda s: sorted(s.tolist()))
    bad = [q for q, r in ranks.items() if r != list(range(1, DEPTH + 1))]
    if bad:
        raise ValueError(f"{len(bad)} queries without exactly ranks 1..{DEPTH}: {bad[:3]}")
    if not set(rl["neighbour_label"]) <= set(labels):
        raise ValueError(f"rank-list labels {set(rl['neighbour_label']) - set(labels)} not in task labels")
    missing = set(rl["query"]) - set(q_patient)
    if missing:
        raise ValueError(f"{len(missing)} queries lack patient/truth: {sorted(missing)[:3]}")
    if not set(rl["neighbour"]) <= set(ref_label):
        raise ValueError("neighbours outside the reference sub-library")
    if (rl["neighbour"].map(ref_label) != rl["neighbour_label"]).any():
        raise ValueError("neighbour_label disagrees with the reference label")
    if (rl["neighbour"] == rl["query"]).any():
        raise ValueError("query retrieved itself")
    same = rl["query"].map(q_patient) == rl["neighbour"].map(ref_patient)
    if same.any():
        raise ValueError(f"{int(same.sum())} same-patient query-neighbour pairs")
    return int(len(rl))


def per_query_table(rl, truth, q_patient, ref, pos=None):
    """ref: DataFrame slide_id,label,patient (the evaluated reference sub-library)."""
    rl = rl.sort_values(["query", "rank"])
    rows = []
    for q, g in rl.groupby("query", sort=True):
        labs, t = g["neighbour_label"].tolist(), truth[q]
        rel = [lab == t for lab in labs]
        support = int(((ref["label"] == t) & (ref["patient"] != q_patient[q]) & (ref["slide_id"] != q)).sum())
        r = {"query": q, "patient": q_patient[q], "truth": t, "top1": labs[0],
             "plurality5": plurality(labs[:5]), "strict5": strict_majority(labs[:5]),
             "hit1": float(rel[0]), "hit3": float(any(rel[:3])), "hit5": float(any(rel[:5])),
             "precision5": sum(rel[:5]) / 5.0,
             "first_relevant_rank": (rel.index(True) + 1) if any(rel) else np.nan,
             "library_support": support}
        if pos is not None:
            r["pos_share5"] = sum(lab == pos for lab in labs[:5]) / 5.0
        rows.append(r)
    df = pd.DataFrame(rows)
    df["plurality5_recall"] = (df["plurality5"] == df["truth"]).astype(float)
    df["strict5_recall"] = (df["strict5"] == df["truth"]).astype(float)
    df["strict5_abstain_rate"] = (df["strict5"] == ABSTAIN).astype(float)
    df["first_rank_absent_rate"] = df["first_relevant_rank"].isna().astype(float)
    return df


def ci(v):
    """Percentile 2.5/97.5 over ALL replicates; any non-finite replicate (e.g. zero class support or a
    one-class AUROC draw) makes the interval unavailable (NaN) rather than being dropped ([decision record]).
    Returns (lo, hi, n_nonfinite)."""
    v = np.asarray(v, float)
    bad = int((~np.isfinite(v)).sum())
    if bad or not v.size:
        return np.nan, np.nan, bad
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), 0


def boot_means(W, v, mask):
    den = W[:, mask].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W[:, mask] @ v[mask]) / np.where(den > 0, den, np.nan)


def boot_auroc(W, y, s):
    out = np.full(W.shape[0], np.nan)
    for b in range(W.shape[0]):
        w = W[b]
        k = w > 0
        if 0 < y[k].sum() < k.sum():
            out[b] = roc_auc_score(y[k], s[k], sample_weight=w[k])
    return out


def load_populations(arm, cfg):
    """Return (rank list, legacy per_query, query population, reference sub-library, coverage dict);
    populations are DataFrames with slide_id, label, patient."""
    d = os.path.join(O, cfg["d"])
    rl = pd.read_csv(os.path.join(d, f"ranked_lists_{arm}.csv.gz"), dtype={"query": str, "neighbour": str})
    pq = pd.read_csv(os.path.join(d, f"per_query_{arm}.csv"), dtype={"query": str})
    if pq["true_label"].isna().any():
        raise ValueError(f"{arm}: per_query rows without true_label")
    cov = {"arm": arm}
    if cfg["kind"] == "kidrare":
        km = pd.read_csv(KIDRARE_MAN, dtype={"slide_id": str})
        nb = km[km["tumor_group"] == "NB"]
        lab = nb.dropna(subset=["keep_subtype"]).assign(label=lambda x: x["keep_subtype"].map(KIDRARE_MAP))
        cov.update(query_manifest="external_kidrare/manifest_inscope.csv (tumor_group NB, keep_subtype)",
                   manifest_rows=len(lab), dropped_label_filter=int((~lab["label"].isin(cfg["labels"])).sum()))
        lab = lab[lab["label"].isin(cfg["labels"])]
        ref = pd.DataFrame({"slide_id": lab["slide_id"], "label": lab["label"], "patient": lab["slide_id"]})
        missing = sorted(set(ref["slide_id"]) - set(pq["query"]))
        ref = ref[ref["slide_id"].isin(set(pq["query"]))]  # saved OT block covers evaluated slides only
        cov.update(broad_group="KidRare NB", broad_group_slides=len(nb), broad_source=KIDRARE_MAN,
                   fine_labelled_slides=int(nb["keep_subtype"].notna().sum()),
                   no_subtype=len(nb) - int(nb["keep_subtype"].notna().sum()), xenograft_excluded=0,
                   no_fusion_status=0,
                   note=(f"labelled slide(s) {', '.join(missing)} absent from the saved OT block {KIDRARE_BLOCK}; "
                         "not evaluated" if missing else ""))
        return rl, pq, ref, ref, cov
    man = pd.read_csv(os.path.join(INV, f"manifest_{cfg['man']}.csv"), dtype={"slide_id": str, "patient_id": str})
    man["label"] = man["label"].replace(cfg.get("relabel", {}))
    man["patient"] = man["patient_id"]
    full = man
    n0 = len(man)
    if not cfg.get("xeno"):
        man = man[man["sample_type"] != "Xenograft"]
    n1 = len(man)
    man = man[~man["patient"].isin(cfg.get("drop_patients", []))]
    n2 = len(man)
    if cfg["kind"] == "fusion":  # fusion status is assay-derived; saved per_query carries it
        fl = dict(zip(pq["query"], pq["true_label"]))
        if not set(fl) <= set(man["slide_id"]):
            raise ValueError("fusion query outside non-xenograft RMS manifest")
        man = man[man["slide_id"].isin(fl)].assign(label=lambda x: x["slide_id"].map(fl))
    n3 = len(man[man["label"].isin(cfg["labels"])])
    ref = man[man["label"].isin(cfg["labels"])][["slide_id", "label", "patient"]]
    broad = pd.read_csv(FROZEN_META)
    broad_n = int((broad["Tumor.group"] == BROAD[cfg["man"]]).sum())
    if cfg["kind"] == "ccdi":
        cm = pd.read_csv(os.path.join(INV, "manifest_s61_rms_ccdi.csv"), dtype={"slide_id": str, "patient_id": str})
        if set(rl["neighbour"]) & set(cm["slide_id"]) or set(cm["patient_id"]) & set(ref["patient"]):
            raise ValueError("CCDI query/COMET reference overlap")
        summ = json.load(open(os.path.join(INV, "summary.json")))["tasks"]["s61_rms_ccdi"]
        cq = cm[cm["label"].isin(cfg["labels"])]
        cov.update(query_manifest="manifest_s61_rms_ccdi.csv", manifest_rows=len(cm),
                   dropped_label_filter=len(cm) - len(cq), broad_group="CCDI RMS after SW1 exclusion",
                   broad_group_slides=summ["n_rms_slides_after_sw1_exclusion"],
                   broad_source="s6_0_label_inventory/summary.json tasks.s61_rms_ccdi",
                   fine_labelled_slides=summ["n_slides_labelled"],
                   reference_manifest=f"manifest_{cfg['man']}.csv", reference_manifest_rows=n0,
                   reference_dropped_xenograft=n0 - n1, reference_dropped_label_filter=n1 - n3,
                   no_subtype=summ["n_rms_slides_after_sw1_exclusion"] - summ["n_slides_labelled"],
                   xenograft_excluded=0, no_fusion_status=0,
                   note="query-side counts; reference-side xenograft exclusion in reference_dropped_xenograft")
        qpop = pd.DataFrame({"slide_id": cq["slide_id"], "label": cq["label"], "patient": cq["patient_id"]})
    else:
        bad = set(pq["query"]) - set(ref["slide_id"])
        if bad:
            raise ValueError(f"{len(bad)} queries not eligible in manifest (xenograft/label): {sorted(bad)[:3]}")
        cov.update(query_manifest=f"manifest_{cfg['man']}.csv", manifest_rows=n0, dropped_xenograft=n0 - n1,
                   dropped_discordant_patient=n1 - n2,
                   dropped_no_fusion_status=(n2 - len(man)) if cfg["kind"] == "fusion" else 0,
                   dropped_label_filter=len(man) - n3, broad_group=f"COMET {BROAD[cfg['man']]} (Tumor.group)",
                   broad_group_slides=broad_n, broad_source=FROZEN_META,
                   fine_labelled_slides=len(full) if cfg["kind"] != "fusion" else n3,
                   no_subtype=broad_n - len(full), xenograft_excluded=n0 - n1,
                   no_fusion_status=(n2 - len(man)) if cfg["kind"] == "fusion" else 0,
                   note=(f"discordant patient(s) {', '.join(cfg['drop_patients'])} excluded"
                         if cfg.get("drop_patients") else ""))
        qpop = ref
    return rl, pq, qpop, ref, cov


def evaluate_arm(arm, cfg, B, seed, out):
    rl, pq, qpop, ref, cov = load_populations(arm, cfg)
    truth = dict(zip(pq["query"], pq["true_label"]))
    if set(rl["query"]) != set(truth):
        raise ValueError(f"{arm}: rank-list and per_query query sets differ")
    qpop = qpop[qpop["slide_id"].isin(truth)]
    if set(qpop["slide_id"]) != set(truth):
        raise ValueError(f"{arm}: per_query queries absent from the eligible query population")
    if (qpop["slide_id"].map(truth) != qpop["label"]).any():
        raise ValueError(f"{arm}: per_query truth disagrees with the manifest label")
    q_patient = dict(zip(qpop["slide_id"], qpop["patient"]))
    pairs = check_rank_list(rl, q_patient, dict(zip(ref["slide_id"], ref["label"])),
                            dict(zip(ref["slide_id"], ref["patient"])), cfg["labels"])
    pos = cfg.get("pos")
    df = per_query_table(rl, truth, q_patient, ref, pos)
    unit = "slide (patient linkage unknown)" if cfg["kind"] == "kidrare" else "patient"
    df.insert(2, "unit", unit)
    df.to_csv(os.path.join(out, f"per_query_{arm}.csv"), index=False)
    labs = cfg["labels"]
    for rule, col, extra in (("plurality", "plurality5", []), ("strict", "strict5", [ABSTAIN])):
        pd.crosstab(df["truth"], df[col]).reindex(index=labs, columns=labs + extra, fill_value=0) \
            .to_csv(os.path.join(out, f"confusion_k5_{rule}_{arm}.csv"))
    cov.update(n_evaluated_queries=len(df), n_reference_slides=len(ref))
    if cfg["kind"] != "ccdi":
        cov["dropped_not_evaluated_residual"] = cov["manifest_rows"] - sum(
            v for k, v in cov.items() if k.startswith("dropped_")) - len(df)

    patients, mult = make_patient_draws(df["patient"].tolist(), count=B, seed=seed)
    W = mult[:, pd.Index(patients).get_indexer(df["patient"])].astype(float)
    legacy = json.load(open(os.path.join(O, cfg["d"], f"result_{arm}.json")))
    pe, pch, lci, vf = (legacy["point_estimates"], legacy["per_class_hit"],
                        legacy["confidence_intervals"], legacy["voting_f1"])
    sunit = KIDRARE_UNIT if cfg["kind"] == "kidrare" else unit
    rows, class_boot, nonfinite, zero_support = [], {}, {}, {}

    def flags(row, cis, n_zero):
        n_bad = max(b for _, _, b in cis.values())
        row.update(n_nonfinite_draws=n_bad, n_zero_support_draws=n_zero, ci_available=n_bad == 0,
                   ci_degenerate=any(lo == hi for lo, hi, _ in cis.values()),
                   ci_degenerate_metrics="|".join(k for k, (lo, hi, _) in cis.items() if lo == hi),
                   ci_patient_level=cfg["kind"] != "kidrare")
        nonfinite[row["class"]], zero_support[row["class"]] = n_bad, n_zero

    for cls in ["ALL"] + labs:
        m = np.ones(len(df), bool) if cls == "ALL" else (df["truth"] == cls).to_numpy()
        rsub = ref if cls == "ALL" else ref[ref["label"] == cls]
        row = {"arm": arm, "class": cls, "unit": sunit, "n_queries": int(m.sum()),
               "n_patients": "unknown" if cfg["kind"] == "kidrare" else int(df.loc[m, "patient"].nunique()),
               "library_support_median": float(df.loc[m, "library_support"].median()) if m.any() else np.nan,
               "n_reference_slides": len(rsub),
               "n_reference_patients": "unknown" if cfg["kind"] == "kidrare" else int(rsub["patient"].nunique())}
        cis = {}
        for k in METRICS:
            v = df[k].to_numpy(float)
            bm = boot_means(W, v, m)
            class_boot[(cls, k)] = bm
            row[k] = float(v[m].mean()) if m.any() else np.nan
            cis[k] = ci(bm)
            row[f"{k}_lo"], row[f"{k}_hi"] = cis[k][:2]
        if cls == "ALL":
            row.update(legacy_hit1=pe["micro_hit_at_1"], legacy_hit5=pe["micro_hit_at_5"],
                       legacy_precision5=pe["micro_precision_at_5"],
                       legacy_plurality5_recall=vf["closest_k5"]["accuracy"],
                       legacy_strict5_recall=vf["absolute_k5"]["accuracy"])
            if pos is not None:
                y, s = (df["truth"] == pos).to_numpy(int), df["pos_share5"].to_numpy(float)
                row["auroc_top5_pos_share"] = float(roc_auc_score(y, s))
                cis["auroc"] = ci(boot_auroc(W, y, s))
                row["auroc_lo"], row["auroc_hi"] = cis["auroc"][:2]
                la = legacy["auroc"]
                row.update(legacy_auroc_count_share=la["count_vote_share_top5"],
                           legacy_auroc_weighted_share=la["weighted_vote_share_top5"],
                           legacy_auroc_weighted_lo=la["weighted_ci"][0], legacy_auroc_weighted_hi=la["weighted_ci"][1])
        else:
            row.update(legacy_hit1=pch["hit_at_1"].get(cls), legacy_hit5=pch["hit_at_5"].get(cls))
        flags(row, cis, int((W[:, m].sum(1) == 0).sum()))
        rows.append(row)
    macro = {"arm": arm, "class": "MACRO", "unit": sunit, "n_queries": len(df)}
    cis = {}
    for k in METRICS:
        macro[k] = float(np.mean([r[k] for r in rows[1:]]))
        cis[k] = ci(np.mean([class_boot[(c, k)] for c in labs], axis=0))
        macro[f"{k}_lo"], macro[f"{k}_hi"] = cis[k][:2]
    macro.update(legacy_hit1=pe["macro_hit_at_1"], legacy_hit5=pe["macro_hit_at_5"],
                 legacy_hit1_lo=lci["macro_hit_at_1"]["lo"], legacy_hit1_hi=lci["macro_hit_at_1"]["hi"],
                 legacy_hit5_lo=lci["macro_hit_at_5"]["lo"], legacy_hit5_hi=lci["macro_hit_at_5"]["hi"],
                 legacy_precision5=pe["macro_precision_at_5"],
                 legacy_plurality5_recall=vf["closest_k5"]["macro_recall"],
                 legacy_strict5_recall=vf["absolute_k5"]["macro_recall"])
    any_zero = np.zeros(W.shape[0], bool)
    for c in labs:
        any_zero |= W[:, (df["truth"] == c).to_numpy()].sum(1) == 0
    flags(macro, cis, int(any_zero.sum()))
    rows.append(macro)

    lq = pq.set_index("query").loc[df["query"]]
    rec = {"n_queries": [int(legacy["config"]["n_queries"]), len(df)],
           "n_reference": [int(legacy["config"]["n_library"]), len(ref)],
           "query_neighbour_pairs_checked": pairs,
           "plurality_vs_pred_closest_k5_agreement": float((lq["pred_closest_k5"].to_numpy() == df["plurality5"]).mean()),
           "strict_vs_pred_absolute_k5_agreement": float(
               (lq["pred_absolute_k5"].replace("Ambiguous", ABSTAIN).to_numpy() == df["strict5"]).mean()),
           "per_query_hit5_agreement": float((lq["hit_k5"].to_numpy() == df["hit5"]).mean()),
           "hit5_micro": [pe["micro_hit_at_5"], rows[0]["hit5"]],
           "hit5_micro_abs_diff": abs(pe["micro_hit_at_5"] - rows[0]["hit5"]),
           "hit5_per_class_max_abs_diff": max(abs(pch["hit_at_5"][r["class"]] - r["hit5"]) for r in rows[1:-1]),
           "ci_nonfinite_draws_by_row": nonfinite, "ci_zero_support_draws_by_row": zero_support,
           "ci_patient_level": cfg["kind"] != "kidrare"}
    if pos is not None:
        rec["auroc_count_share"] = [legacy["auroc"]["count_vote_share_top5"], rows[0]["auroc_top5_pos_share"]]
        rec["auroc_abs_diff"] = abs(rec["auroc_count_share"][0] - rec["auroc_count_share"][1])
        rec["auroc_positive_class"] = [legacy["auroc"]["positive_class"], pos]
    rec["reconciles"] = bool(rec["plurality_vs_pred_closest_k5_agreement"] == 1.0
                             and rec["strict_vs_pred_absolute_k5_agreement"] == 1.0
                             and rec["per_query_hit5_agreement"] == 1.0 and rec["hit5_micro_abs_diff"] == 0.0
                             and rec["n_queries"][0] == rec["n_queries"][1] and rec["n_reference"][0] == rec["n_reference"][1]
                             and (pos is None or rec["auroc_abs_diff"] <= 1e-9))
    inputs = [os.path.join(O, cfg["d"], f"{p}_{arm}.{e}") for p, e in
              (("ranked_lists", "csv.gz"), ("per_query", "csv"), ("result", "json"))]
    inputs.append(KIDRARE_MAN if cfg["kind"] == "kidrare" else os.path.join(INV, f"manifest_{cfg['man']}.csv"))
    rec["inputs"] = {p: {"sha256": sha256(p), "rows": len(rl) if "ranked" in p else None} for p in inputs}
    return rows, cov, rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(O, f"r3_7_fine_label_{datetime.date.today():%Y%m%d}"))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    summary, coverage, recon = [], [], {}
    for arm in a.arms:
        rows, cov, rec = evaluate_arm(arm, ARMS[arm], a.B, a.seed, a.out_dir)
        summary += rows
        coverage.append(cov)
        recon[arm] = rec
        print(f"[{arm}] n={rec['n_queries'][1]} reconciles={rec['reconciles']}", flush=True)
    pd.DataFrame(summary).to_csv(os.path.join(a.out_dir, "summary_all_arms.csv"), index=False)
    pd.DataFrame(coverage).to_csv(os.path.join(a.out_dir, "coverage.csv"), index=False)
    meta = {"script": os.path.abspath(__file__), "script_sha256": sha256(__file__),
            "command": " ".join(sys.argv), "B": a.B, "seed": a.seed, "draws": DRAWS_SOURCE,
            "rules": {
                "top1": "label of rank-1 neighbour",
                "plurality5": "most frequent label in top-5; tie -> tied label at the closest rank",
                "strict5": "label with >=3 of top-5, else ABSTAIN (counted as not correct in strict5_recall)",
                "hit_k": "truth among top-k", "precision5": "#truth in top-5 / 5",
                "first_relevant_rank": "1..10, absent if truth not in top-10",
                "library_support": "reference slides with the query's label, different patient, not the query",
                "pos_share5": "fraction of top-5 with the designated positive label (unweighted count share)",
                "bootstrap": "shared whole-patient multiplicity draws per arm (slide units for KidRare); "
                             "each slide weighted by its unit multiplicity; percentile 2.5/97.5; "
                             "MACRO CI = per-draw mean of class estimates; any non-finite replicate -> "
                             "NaN bounds (counted, not dropped); intervals conditional on the fixed reference library",
                "reference_population": "internal: manifest minus xenografts (unless _xeno), minus SJHBB3FZN "
                                        "for s62, restricted to task labels; fusion: non-xenograft RMS slides with "
                                        "saved fusion status (per_query true_label); ccdi: COMET non-xenograft RMS "
                                        "with task labels; KidRare: keep_subtype-labelled NB slides in the saved block"},
            "arms": {k: ARMS[k] for k in a.arms}, "reconciliation": recon}
    with open(os.path.join(a.out_dir, "reconciliation.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    with open(os.path.join(a.out_dir, "README.md"), "w") as fh:
        fh.write(README.format(cmd="export PROSPER_OUT=\"$PWD/prosper_revision_output\"; "
                               "/HOME/miniconda3/bin/python3 " + " ".join(sys.argv),
                               kidrare_unit=KIDRARE_UNIT,
                               kidrare_note=" ".join(sorted({c["note"] for c in coverage
                                                            if c["arm"].startswith("s65_ext") and c["note"]}))))
    return 0


README = """# R3.7 fine-label re-evaluation of saved S6 rank lists

Conditional fine-label retrieval: a query from a diagnostic family is ranked against the
known-fine-label sub-library of that family, with the query patient excluded. The saved S6
depth-10 rank lists are re-scored (no re-ranking) with non-oracle rules and the ADR0021
whole-patient bootstrap. Rules and population definitions: `reconciliation.json`.

Command (from the repository root): `{cmd}`

## Files
- `per_query_<arm>.csv`: per query top1, plurality5, strict5 (ABSTAIN), hit1/3/5, precision5,
  first_relevant_rank (blank = absent from top-10), library_support, pos_share5 (binary arms).
- `confusion_k5_plurality_<arm>.csv`, `confusion_k5_strict_<arm>.csv`: truth rows x predicted columns.
- `summary_all_arms.csv`: arm x class rows plus ALL (slide-pooled) and MACRO (mean of classes);
  estimates with bootstrap `_lo/_hi`; `legacy_*` columns are the original S6 values; legacy CIs
  (MACRO hit rows, weighted-share AUROC) used the superseded pps_cluster_one_slide bootstrap.
  AUROC here uses the unweighted top-5 positive share (legacy `count_vote_share_top5`).
  Interval flags: `n_nonfinite_draws`/`n_zero_support_draws` count replicates without a defined
  estimate; any non-finite replicate makes that row's affected bounds NaN (`ci_available` False),
  never silently dropped. `ci_degenerate` is True when some interval has lo == hi (e.g. [0,0] or
  [1,1]; metrics listed in `ci_degenerate_metrics`): a degenerate interval reflects resampling a
  small/homogeneous set, not certainty. `ci_patient_level` is False for KidRare (slide resampling).
- `coverage.csv`: manifest rows, eligibility drops (`dropped_*`), evaluated queries, and the
  broad-group slides lacking evaluated labels kept as SEPARATE counts, never summed:
  `no_subtype` (broad-group slides absent from the task manifest, i.e. no fine/task label),
  `xenograft_excluded`, `no_fusion_status` (s61_fusion only); `note` names special exclusions.
- `reconciliation.json`: legacy-vs-recomputed checks (plurality, strict majority, hit@5, AUROC),
  per-row non-finite/zero-support draw counts, input/script SHA256, rules, draw provenance.

## Positive classes (binary arms; AUROC of the top-5 positive share)
s61_int_2c, s61_int_2c_xeno, s61_ext_2c: ARMS; s61_fusion: fusion_positive; s62, s62_xeno:
amplified; s64: HCC; s65_inpc: unfavorable; s65b_gnb: GNB; s65_ext_binary: poorly_differentiated
(versus differentiating; the 33 gnb_intermixed slides are excluded from this arm).

## Interpretation and caveats
1. Fusion status (s61_fusion) and MYCN amplification (s62*) are assay-derived labels: results are
   associations between morphology retrieval and molecular status, not morphology-visible truth.
2. Spindle (3-class RMS arms) has very few slides/patients; its estimates are sparse and unstable.
3. KidRare (s65_ext_*) has no patient linkage: units are slides and intervals come from slide
   resampling (`unit` = "{kidrare_unit}"); they may be too narrow if slides share patients, and no
   patient-level or same-patient-exclusion claim is possible. {kidrare_note}
4. Slides from the same patient are pooled and are not independent: all point estimates,
   including per-class rates, are slide-level; only the intervals account for patient clustering.
5. Intervals are conditional on the fixed reference library (the saved sub-library is not
   resampled); they do not cover variation from a different library composition.
6. s62_xeno differs from s62 by 30 slides: 28 xenografts plus the 2 slides of discordant-label
   patient SJHBB3FZN, both excluded from s62 and included in s62_xeno.
"""


if __name__ == "__main__":
    raise SystemExit(main())
