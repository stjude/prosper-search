"""S1b: retrieval performance stratified by specimen type and by scanner instrument.

Answers two things the audit found unowned (plan v5 S1b; [decision record]/0012):

* **Specimen type** (R3.2 / R4.a): primary tumour vs xenograft vs normal, plus
  metastasis / relapse flags, from ``final_scan_id_disease_map_rich_data.csv``.
  The brainstorm hypothesised COMET's NB set is xenograft-heavy; the C1c query
  refuted that (NB 7%; RB 27.5%). This reports what retrieval does per stratum
  so the composition can be disclosed with numbers.
* **Instrument** (R2.7): the two Aperio AT2 ``ScanScope ID``s from S1a are
  temporally disjoint (D180280 to 2022-07-18, D220721 from 2022-07-27), so a
  split by instrument is a before/after-replacement comparison confounded with
  date and case mix. It is reported **with that confound named in the JSON**,
  never as cross-scanner robustness.

Inputs are the persisted ranked lists from C2 (``c2_metric_suite/ranked_lists_
<model>_<method>.parquet``, depth 10, self and same-patient already excluded),
so no matrix is re-read. Per-query hit@k / precision@k / generous-correct@k /
strict-correct@k are recomputed from the neighbour labels; strata are joined by
slide id. CIs: patient-clustered PPS bootstrap within stratum (the frozen
protocol's design), macro over the classes present in the stratum.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import paths, resample
from .cli import PUBLISHED, PUBLISHED_DROP_LABELS, _git_commit
from .slide_eval import load_metadata

RICH = Path("/PROSPER_WORKSPACE/"
            "comet-new-scans/data/final_slide_list/final_scan_id_disease_map_rich_data.csv")


def _load_ranked(path: Path, depth: int):
    df = (pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path))
    df = df.sort_values(["query", "rank"], kind="mergesort")
    q = df["query"].astype(str).drop_duplicates().tolist()
    truth = df.drop_duplicates("query").set_index(df.drop_duplicates("query")["query"].astype(str))["true_label"].astype(str)
    labs = df["neighbour_label"].astype(str).to_numpy().reshape(len(q), -1)[:, :depth]
    return q, [truth[x] for x in q], labs


def per_query(labs: np.ndarray, truth: List[str], k: int) -> Dict[str, np.ndarray]:
    rel = labs[:, :k] == np.asarray(truth, dtype=object).reshape(-1, 1)
    hit = rel.any(axis=1).astype(float)
    prec = rel.sum(axis=1) / float(k)
    gen = np.zeros(len(truth)); strict = np.zeros(len(truth))
    need = k // 2 + 1
    for i in range(len(truth)):
        row = [l for l in labs[i, :k] if l != "__NONE__"]
        if not row:
            continue
        c = Counter(row); top = max(c.values())
        tied = {l for l, n in c.items() if n == top}
        gen[i] = float(truth[i] in tied)
        reached = [l for l, n in c.items() if n >= need]
        strict[i] = float(len(reached) == 1 and reached[0] == truth[i])
    return {f"hit_at_{k}": hit, f"precision_at_{k}": prec,
            f"generous_correct_at_{k}": gen, f"strict_correct_at_{k}": strict}


def _macro(vals, rows, tl, classes):
    per = [float(np.mean(vals[rows][tl[rows] == c])) for c in classes if (tl[rows] == c).any()]
    return float(np.mean(per)) if per else float("nan")


def stratum_block(qs, truth, vecs, meta, B, seed, mode) -> Dict:
    tl = np.asarray(truth, dtype=object)
    classes = sorted(set(truth))
    rows_all = np.arange(len(qs))
    out = {"n_queries": int(len(qs)), "n_classes": len(classes),
           "n_patients": int(meta[PUBLISHED["patient_col"]].nunique()),
           "class_counts": dict(Counter(truth)), "point": {}, "ci": {}}
    for name, v in vecs.items():
        out["point"][f"macro_{name}"] = _macro(v, rows_all, tl, classes)
        out["point"][f"micro_{name}"] = float(np.mean(v))
    if B and len(qs) > 1:
        index = resample.build_patient_index(meta, PUBLISHED["id_col"], PUBLISHED["label_col"],
                                             PUBLISHED["patient_col"])
        idx_of = {q: i for i, q in enumerate(qs)}
        rng = np.random.default_rng(seed)
        draws = {n: np.empty(B) for n in vecs}
        for b in range(B):
            s = resample.sample_queries(index, classes, rng, mode=mode)
            rows = np.array([idx_of[x] for x in s if x in idx_of], dtype=int)
            for n, v in vecs.items():
                draws[n][b] = _macro(v, rows, tl, classes) if rows.size else np.nan
        for n, d in draws.items():
            d = d[~np.isnan(d)]
            out["ci"][f"macro_{n}"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d.size else None
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="uni2")
    p.add_argument("--method", default="ot_cosine")
    p.add_argument("--ranked-dir", default=None)
    p.add_argument("--headers", default=None, help="S1a slide_headers.csv")
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--iter", type=int, default=2000, dest="B")
    p.add_argument("--seed", type=int, default=PUBLISHED["seed"])
    p.add_argument("--task", default="s1b_stratified")
    a = p.parse_args(argv)

    out_root = paths.output_root()
    ranked_dir = Path(a.ranked_dir) if a.ranked_dir else out_root / "c2_metric_suite"
    rl = ranked_dir / f"ranked_lists_{a.model}_{a.method}.parquet"
    if not rl.exists():
        rl = rl.with_suffix(".csv.gz")
    paths.require(rl, "ranked lists")
    qs, truth, labs = _load_ranked(rl, a.depth)

    meta = load_metadata(str(paths.slide_metadata()), label_col=PUBLISHED["label_col"],
                         id_col=PUBLISHED["id_col"], patient_col=PUBLISHED["patient_col"],
                         drop_labels=list(PUBLISHED_DROP_LABELS[False]), min_class_count=None)
    meta = meta[meta[PUBLISHED["id_col"]].isin(set(qs))].set_index(PUBLISHED["id_col"], drop=False)
    rich = pd.read_csv(RICH, dtype=str)
    rich["slide_id"] = rich["slide_id"].astype(str)
    rich = rich.drop_duplicates("slide_id").set_index("slide_id")
    hdr = None
    hp = Path(a.headers) if a.headers else out_root / "s1a_slide_headers" / "slide_headers.csv"
    if hp.exists():
        hdr = pd.read_csv(hp, dtype={"slide_id": str}).drop_duplicates("slide_id").set_index("slide_id")

    vecs = {}
    for k in a.k_values:
        vecs.update(per_query(labs, truth, k))

    strata = {
        "specimen_type": [str(rich["Sample.Type"].get(q, "unjoined")) for q in qs],
        "metastasis": [str(rich["Met"].get(q, "unjoined")) for q in qs],
        "evaluation": [str(rich["Evaluation"].get(q, "unjoined")) for q in qs],
    }
    if hdr is not None:
        strata["instrument"] = [str(hdr["scanscope_id"].get(q, "unknown")) for q in qs]
        yrs = [str(hdr["scan_date"].get(q, ""))[:4] or "unknown" for q in qs]
        strata["scan_year"] = yrs

    results = {"config": {"model": a.model, "method": a.method, "ranked_lists": str(rl),
                          "depth": a.depth, "k_values": a.k_values, "bootstrap_B": a.B,
                          "seed": a.seed, "rich_metadata": str(RICH),
                          "headers": str(hp) if hdr is not None else None,
                          "prosper_commit": _git_commit(), "lsf_job": os.environ.get("LSB_JOBID")},
               "n_queries": len(qs), "strata": {}}
    for sname, svals in strata.items():
        svals = np.asarray(svals, dtype=object)
        block = {}
        for level in sorted(set(svals.tolist())):
            rows = np.where(svals == level)[0]
            if rows.size < 5:
                block[level] = {"n_queries": int(rows.size), "note": "too few queries"}
                continue
            sub_q = [qs[i] for i in rows]
            sub_t = [truth[i] for i in rows]
            sub_v = {n: v[rows] for n, v in vecs.items()}
            block[level] = stratum_block(sub_q, sub_t, sub_v, meta.loc[sub_q], a.B, a.seed,
                                         PUBLISHED["bootstrap_mode"])
        results["strata"][sname] = block
    results["notes"] = {
        "instrument": "The two ScanScope IDs are temporally disjoint (D180280 to 2022-07-18, "
                      "D220721 from 2022-07-27): an instrument replacement, confounded with "
                      "date, case mix and staining lots. NOT a cross-scanner test.",
        "specimen_type": "Per C1c (2026-08-29): 269 xenograft slides / 198 patients, all "
                         "masked under UID.Subject; NB 7% xenograft, RB 27.5%.",
    }
    out = paths.result_json(a.task, f"stratified_{a.model}_{a.method}")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[s1b] wrote {out}")
    for sname, block in results["strata"].items():
        for level, b in block.items():
            if "point" in b:
                print(f"[s1b] {sname:<14} {level:<16} n={b['n_queries']:<5} "
                      f"hit@5={b['point'].get('macro_hit_at_5', float('nan')):.4f} "
                      f"prec@5={b['point'].get('macro_precision_at_5', float('nan')):.4f} "
                      f"gen@5={b['point'].get('macro_generous_correct_at_5', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
