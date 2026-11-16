"""R2.1 / R4.b / R3.1 — retrieval metrics for the revised Fig. 2 and Supplementary
Table S-R2.1-E (design decision 2026-10-21 ≈13:1x CDT: Fig. 2 on macro hit@k with
mMV@5; mAP@5/@10 for every cell and comparator in the Supplement).

Derived from SAVED rankings only; no retrieval, feature or model code is run.

Inputs (all accepted; SHA-256 bound in input_bindings.json)
  * population  r2_1_main_grid_comparators_20261011/main_uni2_attempt_001/populations.csv
                (2,778 queries = library, 15 groups, MISC excluded; reference_index order)
  * draws       .../patient_draws.npy  (5,000 x 1,876 patient multiplicities, PCG64 seed 123;
                the draws of RR-5, R3.1 and R4.a-vii)
  * 37 grid cells: the RR-5 replacement bindings (32 published four-method matrices with the
                Virchow2 cells replaced by the fp32 matrices, 4 H-Optimus-1 matrices,
                Conchv1.5/TITAN repaired matrix); ranking per RR-5 item 8 (ascending distance,
                exact ties by ascending reference_index, same-patient entries ineligible)
  * comparators from depth-10 ranked lists: SISH, PRISM (native tiling), ImageNet ResNet-50
                floor, RetCCL x 4 aggregations (R3.1); SISH padding rule as in R3.1 (missing
                positions are non-relevant)

Per query: relevance of ranks 1..10 (same group, other patient) -> hit@k (k = 1,3,5,10),
precision@k (k = 1,3,5,10), AP@5, AP@10 (normaliser min(k, library_support)); micro
(mean over queries) and macro (mean of 15 group means) with 95 % percentile intervals over
the pinned draws; per-group hit@5, precision@5, AP@5 with intervals.

Gates (any failure -> FAILED, no COMPLETE):
  G1 every grid cell x k: hit/precision micro+macro point and interval == RR-5 hit_precision.csv (1e-6)
  G2 every comparator: hit1/3/5/10, precision5/10 micro+macro point and interval == R3.1 summaries (1e-9)
  G3 UNI2 + OT: mAP@5/@10 micro+macro point and interval == Table S25 source summary_ext.csv (1e-9)
  G4 draws: 5,000 rows each summing to 1,876; patients sorted as in make_patient_draws
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
K_VALUES = (1, 3, 5, 10)
DEPTH = 10
QUANTILES = (0.025, 0.975)

RR5 = OUT_ROOT / "r2_1_virchow2_fp32_main_20261013/run_003/rr5_replacement_001"
RR5_BINDINGS = RR5 / "raw/input_bindings.json"
R31_CMP = OUT_ROOT / "r3_1_comparators_whole_patient_20261008/summary.csv"
R31_PRISM = OUT_ROOT / "r3_1_prism_20261018/eval_native_20261019/summary.csv"
R31_RETCCL = OUT_ROOT / "r3_1_retccl_whole_patient_20261016/summary.csv"
S25_EXT = OUT_ROOT / "r2_3_internal_failure_cis_ext_prod_20261005/summary_ext.csv"
COMPARATORS = {
    # config name -> (ranked list, R3.1 summary file, R3.1 config id, display system, aggregation, kind)
    "sish": (OUT_ROOT / "c2_metric_suite/ranked_lists_sish_system.csv.gz", R31_CMP, "sish", "SISH", "end-to-end", "e2e"),
    "prism_native": (OUT_ROOT / "r3_1_prism_20261018/eval_native_20261019/ranked_lists_prism_native.csv.gz", R31_PRISM, "prism_native", "PRISM (native tiling)", "end-to-end", "e2e"),
    "resnet50": (OUT_ROOT / "c5_resnet50/ranked_lists_resnet50_median_pooling.csv.gz", R31_CMP, "resnet50", "ImageNet ResNet-50", "Median Pooling", "floor"),
    "retccl_median": (OUT_ROOT / "c2_metric_suite/ranked_lists_retccl_median_pooling.csv.gz", R31_RETCCL, "retccl_median", "RetCCL", "Median Pooling", "grid"),
    "retccl_ot": (OUT_ROOT / "c2_metric_suite/ranked_lists_retccl_ot_cosine.csv.gz", R31_RETCCL, "retccl_ot", "RetCCL", "Optimal Transport", "grid"),
    "retccl_yottixel": (OUT_ROOT / "c2_metric_suite/ranked_lists_retccl_yottixel.csv.gz", R31_RETCCL, "retccl_yottixel", "RetCCL", "Yottixel", "grid"),
    "retccl_sampler": (OUT_ROOT / "c2_metric_suite/ranked_lists_retccl_sampler.csv.gz", R31_RETCCL, "retccl_sampler", "RetCCL", "SAMPLER", "grid"),
}
AGG_NAME = {"median_pooling": "Median Pooling", "ot_cosine": "Optimal Transport", "sampler": "SAMPLER", "yottixel": "Yottixel", "titan_repaired": "TITAN"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- population / draws
def load_population(path):
    pop = pd.read_csv(path)
    pop = pop.sort_values("reference_index").reset_index(drop=True)
    assert (pop.reference_index.values == np.arange(len(pop))).all()
    assert len(pop) == 2778 and pop.diagnostic_group.nunique() == 15 and "MISC" not in set(pop.diagnostic_group)
    pop["slide_id"] = pop.slide_id.astype(int).astype(str)
    return pop


def query_weights(draws, pop):
    patients = tuple(sorted(set(pop.patient_id)))
    assert draws.shape == (5000, len(patients)) and (draws.sum(1) == len(patients)).all()
    pidx = {p: i for i, p in enumerate(patients)}
    cols = np.array([pidx[p] for p in pop.patient_id])
    return draws[:, cols].astype(np.float64)  # (B, Q) multiplicity of each query's patient


# ----------------------------------------------------------------------------- relevance
def relevance_from_matrix(path, pop):
    M = pickle.load(open(path, "rb"))
    assert isinstance(M, pd.DataFrame) and M.values.dtype == np.float64
    M.index = M.index.astype(str); M.columns = M.columns.astype(str)
    ids = pop.slide_id.tolist()
    X = M.loc[ids, ids].to_numpy(copy=True)
    pat = pop.patient_id.to_numpy()
    same = pat[:, None] == pat[None, :]
    assert np.all(X[same] == 1e6), "same-patient sentinel missing"
    X[same] = np.inf
    assert np.isfinite(X[~same]).all() and (X[~same] >= 0).all()
    order = np.argsort(X, axis=1, kind="stable")[:, :DEPTH]  # stable: ties by ascending reference_index
    grp = pop.diagnostic_group.to_numpy()
    rel = grp[order] == grp[:, None]
    assert not same[np.arange(len(pop))[:, None], order].any()
    return rel.astype(np.int8), order


def relevance_from_ranked_list(path, pop):
    fr = pd.read_csv(path)
    fr["query"] = fr["query"].astype(int).astype(str)
    fr = fr[fr["query"].isin(set(pop.slide_id))]
    fr = fr[(fr["rank"] >= 1) & (fr["rank"] <= DEPTH)]
    qidx = {s: i for i, s in enumerate(pop.slide_id)}
    grp = pop.diagnostic_group.to_numpy()
    rel = np.zeros((len(pop), DEPTH), dtype=np.int8)
    present = np.zeros((len(pop), DEPTH), dtype=bool)
    lab = fr["neighbour_label"].astype(object)
    nb = fr["neighbour"]
    # R3.1 padding rule: SISH short lists carry neighbour_label "__NONE__" / empty neighbour & distance
    padded = lab.isna() | lab.eq("__NONE__") | nb.isna() | fr["distance"].isna()
    for q, r, l, pd_ in zip(fr["query"], fr["rank"], lab, padded):
        i, j = qidx[q], int(r) - 1
        present[i, j] = not pd_
        rel[i, j] = int((not pd_) and l == grp[i])
    n_padded = int((~present).sum())
    return rel, n_padded


# ----------------------------------------------------------------------------- metrics
def per_query_metrics(rel, support):
    out = {}
    c = np.cumsum(rel.astype(np.float64), axis=1)  # (Q, 10)
    for k in K_VALUES:
        out[f"hit{k}"] = (c[:, k - 1] > 0).astype(np.float64)
        out[f"precision{k}"] = c[:, k - 1] / k
    prec_i = c / np.arange(1, DEPTH + 1, dtype=np.float64)[None, :]
    for k in (5, 10):
        out[f"ap{k}"] = (prec_i[:, :k] * rel[:, :k]).sum(1) / np.minimum(float(k), support)
    return out


def boot(W, v, groups, G):
    """micro and macro point + percentile interval of a per-query vector v under weights W (B, Q)."""
    res = {}
    tot = W.sum(1)
    res["micro"] = (float(v.mean()), *np.quantile((W @ v) / tot, QUANTILES))
    gm = np.zeros((W.shape[0], len(G)))
    pts = np.zeros(len(G))
    for gi, g in enumerate(G):
        m = groups == g
        gm[:, gi] = (W[:, m] @ v[m]) / W[:, m].sum(1)
        pts[gi] = v[m].mean()
    res["macro"] = (float(pts.mean()), *np.quantile(gm.mean(1), QUANTILES))
    res["per_group"] = {g: (float(pts[gi]), *np.quantile(gm[:, gi], QUANTILES)) for gi, g in enumerate(G)}
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_ROOT / "r2_1_fig2_retrieval_20261021" / "run_001"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    bind = json.load(open(RR5_BINDINGS))["bindings"]
    pop_path, draws_path = Path(bind["population"]["path"]), Path(bind["draws"]["path"])
    inputs = {"population": pop_path, "draws": draws_path, "rr5_hit_precision": RR5 / "hit_precision.csv",
              "r3_1_comparators_summary": R31_CMP, "r3_1_prism_summary": R31_PRISM, "r3_1_retccl_summary": R31_RETCCL,
              "s25_summary_ext": S25_EXT}
    # canonical RR-5 cell set: the fp32 Virchow2 cells replace the pre-fp32 `virchow2/*` cells; TITAN published excluded
    rr5_cells = set(pd.read_csv(RR5 / "hit_precision.csv").cell_id) - {"titan/titan_published"}
    cells = {k.split(":", 1)[1]: Path(v["path"]) for k, v in bind.items() if k.startswith("matrix:") and k.split(":", 1)[1] in rr5_cells}
    assert len(cells) == 37 and set(cells) == rr5_cells, len(cells)
    for c, p in cells.items():
        inputs[f"matrix:{c}"] = p
    for name, (p, *_rest) in COMPARATORS.items():
        inputs[f"ranked_list:{name}"] = p
    log("hashing inputs")
    bindings = {k: {"path": str(p), "sha256": sha256(p), "size": p.stat().st_size} for k, p in inputs.items()}
    for c in cells:  # the matrix hashes must equal the RR-5 bound hashes
        assert bindings[f"matrix:{c}"]["sha256"] == bind[f"matrix:{c}"]["sha256"], c
    assert bindings["draws"]["sha256"] == bind["draws"]["sha256"] and bindings["population"]["sha256"] == bind["population"]["sha256"]
    json.dump({"created": datetime.now().isoformat(timespec="seconds"), "script_sha256": sha256(Path(__file__)), "bindings": bindings},
              open(out / "input_bindings.json", "w"), indent=1)

    pop = load_population(pop_path)
    draws = np.load(draws_path)
    W = query_weights(draws, pop)
    groups = pop.diagnostic_group.to_numpy()
    G = sorted(set(groups))
    # library_support: same group, other patient
    pat = pop.patient_id.to_numpy()
    support = np.array([((groups == groups[i]) & (pat != pat[i])).sum() for i in range(len(pop))], dtype=np.float64)
    assert support.min() >= DEPTH

    rr5 = pd.read_csv(RR5 / "hit_precision.csv"); rr5 = rr5[rr5.tie_order == "canonical"]
    disp = {r.cell_id: r.display_encoder for r in rr5.drop_duplicates("cell_id").itertuples()}
    gates = {"G1_grid_vs_rr5": {"n": 0, "max_abs": 0.0, "fail": []}, "G2_comparators_vs_r3_1": {"n": 0, "max_abs": 0.0, "fail": []},
             "G3_main_map_vs_s25": {"n": 0, "max_abs": 0.0, "fail": []}, "G4_draws": {"rows": int(draws.shape[0]), "patients": int(draws.shape[1])}}
    summary, per_group, pq_rows = [], [], []

    def emit(system, aggregation, kind, cell_id, lineage, rel, n_padded=0):
        m = per_query_metrics(rel, support)
        for name, v in m.items():
            k = int("".join(ch for ch in name if ch.isdigit()))
            metric = name.rstrip("0123456789")
            b = boot(W, v, groups, G)
            for avg in ("micro", "macro"):
                pt, lo, hi = b[avg]
                summary.append(dict(system=system, aggregation=aggregation, kind=kind, cell_id=cell_id, lineage=lineage, k=k,
                                    metric={"hit": "hit_at_k", "precision": "precision_at_k", "ap": "map_at_k"}[metric], average=avg,
                                    point=pt, ci_lo=lo, ci_hi=hi, n_queries=len(pop), n_patients=draws.shape[1], n_replicates=W.shape[0], n_padded=n_padded))
            if k == 5:
                for g, (pt, lo, hi) in b["per_group"].items():
                    per_group.append(dict(system=system, aggregation=aggregation, kind=kind, cell_id=cell_id, k=5, group=g,
                                          metric={"hit": "hit_at_k", "precision": "precision_at_k", "ap": "map_at_k"}[metric],
                                          n_queries=int((groups == g).sum()), point=pt, ci_lo=lo, ci_hi=hi))
        pq_rows.append(pd.DataFrame({"cell_id": cell_id, "slide_id": pop.slide_id, "group": groups,
                                     "rel_top10": ["".join(map(str, r)) for r in rel], **{k: v for k, v in m.items()}}))
        return m

    # ---- grid cells
    for i, (cell, path) in enumerate(sorted(cells.items())):
        log(f"cell {i + 1}/37 {cell}")
        rel, _ = relevance_from_matrix(path, pop)
        enc, agg = cell.split("/")
        aggregation = AGG_NAME[agg]
        kind = "e2e" if aggregation == "TITAN" else "grid"
        system = "TITAN" if kind == "e2e" else disp[cell]
        emit(system, aggregation, kind, cell, "rr5_replacement_001", rel)
        # G1
        for row in [s for s in summary if s["cell_id"] == cell and s["metric"] in ("hit_at_k", "precision_at_k")]:
            ref = rr5[(rr5.cell_id == cell) & (rr5.k == row["k"]) & (rr5.metric == f"{'hit' if row['metric'] == 'hit_at_k' else 'precision'}_{row['average']}")]
            assert len(ref) == 1, (cell, row["k"], row["metric"])
            ref = ref.iloc[0]
            d = max(abs(ref.value - row["point"]), abs(ref.ci_low - row["ci_lo"]), abs(ref.ci_high - row["ci_hi"]))
            gates["G1_grid_vs_rr5"]["n"] += 1; gates["G1_grid_vs_rr5"]["max_abs"] = max(gates["G1_grid_vs_rr5"]["max_abs"], float(d))
            if d > 1e-6:
                gates["G1_grid_vs_rr5"]["fail"].append([cell, row["k"], row["metric"], row["average"], float(d)])
    # ---- comparators
    for name, (path, summ, cfg, system, aggregation, kind) in COMPARATORS.items():
        log(f"comparator {name}")
        rel, n_padded = relevance_from_ranked_list(path, pop)
        emit(system, aggregation, kind, name, "r3_1_saved_ranked_list", rel, n_padded)
        ref = pd.read_csv(summ); ref = ref[(ref.population == "all_tissue_2778") & (ref.config == cfg)]
        for row in [s for s in summary if s["cell_id"] == name and s["metric"] in ("hit_at_k", "precision_at_k")]:
            mname = f"{'hit' if row['metric'] == 'hit_at_k' else 'precision'}{row['k']}"
            r = ref[(ref.metric == mname) & (ref.average == row["average"])]
            if len(r) == 0:
                continue  # R3.1 did not report precision@1/@3
            r = r.iloc[0]
            d = max(abs(r.point - row["point"]), abs(r.ci_lo - row["ci_lo"]), abs(r.ci_hi - row["ci_hi"]))
            gates["G2_comparators_vs_r3_1"]["n"] += 1; gates["G2_comparators_vs_r3_1"]["max_abs"] = max(gates["G2_comparators_vs_r3_1"]["max_abs"], float(d))
            if d > 1e-9:
                gates["G2_comparators_vs_r3_1"]["fail"].append([name, mname, row["average"], float(d)])
    # ---- G3: main configuration mAP vs Table S25 source
    s25 = pd.read_csv(S25_EXT).set_index("metric")
    for k in (5, 10):
        for avg in ("micro", "macro"):
            row = [s for s in summary if s["cell_id"] == "uni2/ot_cosine" and s["metric"] == "map_at_k" and s["k"] == k and s["average"] == avg][0]
            r = s25.loc[f"map{k}_{avg}"]
            d = max(abs(r.point - row["point"]), abs(r.lo - row["ci_lo"]), abs(r.hi - row["ci_hi"]))
            gates["G3_main_map_vs_s25"]["n"] += 1; gates["G3_main_map_vs_s25"]["max_abs"] = max(gates["G3_main_map_vs_s25"]["max_abs"], float(d))
            if d > 1e-9:
                gates["G3_main_map_vs_s25"]["fail"].append([k, avg, float(d)])

    passed = all(not v.get("fail") for v in gates.values() if isinstance(v, dict) and "fail" in v)
    gates["passed"] = passed
    gates["runtime_s"] = round(time.time() - t0, 1)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(per_group).to_csv(out / "per_group.csv", index=False)
    pd.concat(pq_rows).to_csv(out / "per_query_top10.csv.gz", index=False)
    json.dump(gates, open(out / "gates.json", "w"), indent=1)
    (out / "README.md").write_text(__doc__ + f"\n\n## Run\n\n{datetime.now().isoformat(timespec='seconds')}; B = {W.shape[0]}; "
                                   f"gates passed = {passed}; runtime {gates['runtime_s']} s; python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}.\n")
    lines = [f"{sha256(p)}  {p.name}" for p in sorted(out.iterdir()) if p.name not in ("SHA256SUMS.txt", "COMPLETE", "FAILED")]
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    (out / ("COMPLETE" if passed else "FAILED")).write_text(json.dumps(gates, indent=1))
    log(f"{'COMPLETE' if passed else 'FAILED'}: {json.dumps({k: (v['n'], v['max_abs'], len(v['fail'])) for k, v in gates.items() if isinstance(v, dict) and 'fail' in v})}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
