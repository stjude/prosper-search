#!/usr/bin/env python3
"""R3.5-ROI6 evaluation ([analysis plan], pre-registered 2026-11-11).

Patient-excluded raw-hit plurality voting of ROI hits (C1b), computed by the unchanged
``prosper/evaluation/roi_eval.py`` per slide, with whole-patient bootstrap intervals from the
retained R4a-vii code: the executed copy ``r4a_vii_whole_patient.executed_56d1194f.py`` is
executed as-is (its relative kernel import resolves to ``scripts/``), and its draws, F1
replicates and interval rows are reused. Gate G4: the submitted (float16) hit directory must
reproduce the R4a-vii C1b rows exactly, with the same draw multiplicity hash, before any new
configuration is scored; it is a gate only. Paired differences are per replicate against the
float32 Virchow2 L1 configuration (``--paired-ref``), the revised counterpart of the submitted one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
R4_EXEC = OUT / "r4a_vii_whole_patient_20261015/r4a_vii_whole_patient.executed_56d1194f.py"
R4_SHA = "0667923b04b529b38cdfcbc3d418cef58b49af581090a95c5ebcd1da278b826d"
LADDER_SHA = "c635ad4a65d96fea39e308cf568cbe3133e939b21c7ecc5de5df0ae9b5227138"
ROI_EVAL_SHA = "050081c5e0fa6587300408c96fc01223e4530ce9fde7f2297c0d35710fdbc689"
R4_SUMMARY = OUT / "r4a_vii_whole_patient_20261015/production/summary.csv"
C1B_MULT_SHA = "6aeaf004817321ec20c001812aa02fb9109c42c7e00fe8479b94aafeb685f6db"
B, SEED, KS = 5000, 123, (5, 10)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_r4():
    src = R4_EXEC.read_bytes()
    if hashlib.sha256(src).hexdigest() != R4_SHA:
        raise SystemExit("R4a-vii executed copy changed")
    if sha(REPO / "scripts/r2_2_tile_ladder_whole_patient.py") != LADDER_SHA:
        raise SystemExit("bootstrap kernel changed since R4a-vii")
    m = types.ModuleType("r4a_vii_executed")
    m.__file__ = str(REPO / "scripts/r4a_vii_whole_patient.py")
    exec(compile(src, str(R4_EXEC), "exec"), m.__dict__)
    return m


R4 = load_r4()
L = R4.L
_W = {}


def _worker(sid):
    _, roi_eval, _ = R4.prosper_modules()
    c = _W
    path = os.path.join(c["dir"], f"top_10_hit_slides_slide_{sid}.pkl")
    df = pd.read_pickle(path)
    out = dict(query=sid, rows=len(df), tiles=int(df["query_patch_id"].nunique()),
               ten_per_tile=bool((df.groupby("query_patch_id", sort=False).size() == 10).all()),
               ascending=bool((np.diff(df["similarity"].to_numpy(np.float64).reshape(-1, 10), axis=1) >= 0).all()))
    for k in KS:
        r = roi_eval.predict_slide_from_roi_neighbors_patient_excluded(sid, c["dir"], c["s2l"], c["s2p"], k, "raw_hits")
        out[f"k{k}"] = None if r is None else {x: r.get(x) for x in (
            "true_label", "pred_label", "n_rois", "skipped_rois", "total_hits_counted", "missing_label_hits",
            "same_patient_hits_dropped", "same_patient_hits_surviving", "hit_at_5")}
    return out


def predict(name, d, ids, s2l, s2p, n_workers):
    _W.update(dir=str(d), s2l=s2l, s2p=s2p)
    with mp.get_context("fork").Pool(n_workers) as pool:
        res = list(pool.imap(_worker, ids, chunksize=4))
    rows = []
    for r in res:
        for k in KS:
            x = dict(r[f"k{k}"] or {}, config=name, k=k, query=r["query"], patient=s2p[r["query"]],
                     rows=r["rows"], tiles=r["tiles"], ten_per_tile=r["ten_per_tile"], ascending=r["ascending"],
                     predicted=r[f"k{k}"] is not None)
            rows.append(x)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=OUT / "roi_search_6cfg_20261111")
    ap.add_argument("--configs", default="v2_l1,v2_cos,uni2_l1,uni2_cos,hopt1_l1,hopt1_cos")
    ap.add_argument("--primary", default="uni2_cos")
    ap.add_argument("--paired-ref", default="v2_l1",
                    help="float32 counterpart of the submitted configuration; paired differences are against it")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--gate-only", action="store_true", help="run G4 on the submitted hits and stop")
    a = ap.parse_args(argv)
    out = a.out or a.root / "eval" / time.strftime("run_%Y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    _, roi_eval, roi_cli = R4.prosper_modules()
    gates = {"roi_eval_sha256": sha(Path(roi_eval.__file__)) == ROI_EVAL_SHA,
             "meta_sha256": sha(Path(R4.META)) == R4.EXPECTED_SHA256[R4.META]}
    meta = roi_eval.load_metadata(R4.META, R4.ID, R4.LAB, drop_labels=["MISC"], patient_col=R4.PAT)
    s2l, s2p = dict(zip(meta[R4.ID], meta[R4.LAB])), dict(zip(meta[R4.ID], meta[R4.PAT]))
    dirs = {"submitted": Path(R4.ROI_DIR)}
    if not a.gate_only:
        dirs.update({c: a.root / "hits" / c for c in a.configs.split(",")})
    ids = sorted(set(meta[R4.ID]) & set(roi_eval.discover_slides_with_files(str(dirs["submitted"]))))
    gates["population_2778"] = len(ids) == R4.N_QUERIES
    bank_rows = {}
    for b in ("v2", "uni2", "hopt1"):
        s = pd.read_csv(a.root / "banks" / b / "slides.csv", dtype={"slide_id": str})
        bank_rows[b] = dict(zip(s["slide_id"], s["n_rows"]))
    frames = []
    for name, d in dirs.items():
        files = set(roi_eval.discover_slides_with_files(str(d)))
        gates[f"{name}:all_queries_have_files"] = set(ids) <= files
        fb = roi_cli.count_glob_fallbacks(ids, str(d))
        gates[f"{name}:glob_fallbacks_zero"] = fb["n_would_use_glob_fallback"] == 0 and fb["n_no_file_at_all"] == 0
        f = predict(name, d, ids, s2l, s2p, a.n_workers)
        gates[f"{name}:every_slide_predicted"] = bool(f["predicted"].all())
        gates[f"{name}:same_patient_hits_surviving_zero"] = int(f["same_patient_hits_surviving"].sum()) == 0
        gates[f"{name}:ten_rows_per_tile_ascending"] = bool(f["ten_per_tile"].all() and f["ascending"].all())
        if name != "submitted":
            nr = bank_rows["v2" if name.startswith("v2") else name.split("_")[0]]
            gates[f"{name}:tiles_equal_bank_rows"] = bool((f["tiles"] == f["query"].map(nr)).all())
        frames.append(f)
        print(f"{name}: predicted {len(f) // 2} slides, {time.time() - t0:.0f}s", flush=True)
    fr = pd.concat(frames, ignore_index=True)
    fr["correct"] = (fr["true_label"] == fr["pred_label"]).astype(int)
    fr.to_csv(out / "per_query.csv.gz", index=False)

    base = fr[(fr["config"] == "submitted") & (fr["k"] == 5)].set_index("query").sort_index()
    classes = sorted(base["true_label"].unique().tolist())
    patients, mult, W = L.weight_matrix(base["patient"], B, SEED)
    gates["draws_identical_to_r4a_vii_c1b"] = R4.mult_sha(mult) == C1B_MULT_SHA
    G = R4.group_matrix(base["true_label"], classes)
    one = np.ones((1, len(base)))
    pop = "c1b_roi_2778"
    reps, summ, paired = {}, [], []
    for name in dirs:
        for k in KS:
            f = fr[(fr["config"] == name) & (fr["k"] == k)].set_index("query").sort_index()
            if not (f.index.equals(base.index) and (f["true_label"] == base["true_label"]).all()):
                gates[f"{name}:k{k}:aligned"] = False
                continue
            m = roi_eval.compute_metrics_from_predictions(f[["true_label", "pred_label"]].reset_index(drop=True),
                                                         labels_order=classes)
            mu, iu = R4.f1_reps(one, f["true_label"], f["pred_label"], classes, micro="accuracy")
            gates[f"{name}:k{k}:unit_weight_replicate"] = bool(abs(mu[0] - m["macro_f1"]) <= 1e-12
                                                               and abs(iu[0] - m["micro_f1"]) <= 1e-12)
            mac, mic = R4.f1_reps(W, f["true_label"], f["pred_label"], classes, micro="accuracy")
            h = f["hit_at_5"].to_numpy(float)
            vals = {("f1", "macro"): (m["macro_f1"], mac), ("f1", "micro"): (m["micro_f1"], mic),
                    ("hit_at_5", "micro"): (float(h.mean()), L.boot_means(W, h)),
                    ("hit_at_5", "macro"): (float(np.mean([h[G[i] > 0].mean() for i in range(len(classes))])),
                                            L.macro_reps(W, G, h))}
            cfg = f"{name}_k{k}"
            for (met, avg), (p, rr) in vals.items():
                reps[(cfg, met, avg)] = (float(p), rr)
                role = ("primary" if name == a.primary and k == 5 else
                        "gate only (submitted float16 hits, superseded)" if name == "submitted" else
                        "paired reference (float32 Virchow2 L1)" if name == a.paired_ref else "sensitivity")
                summ.append(R4.summary_row("r3_5_roi6", pop, cfg, met, avg, float(p), rr, len(base), len(patients),
                                           len(classes), B, role=role,
                                           note="micro F1 = accuracy (roi_eval)" if (met, avg) == ("f1", "micro") else ""))
    for k in KS:
        if a.gate_only:
            break
        for name in dirs:
            if name in ("submitted", a.paired_ref):
                continue
            for met, avg in (("f1", "macro"), ("f1", "micro"), ("hit_at_5", "micro"), ("hit_at_5", "macro")):
                (pa, ra), (pb, rb) = reps[(f"{name}_k{k}", met, avg)], reps[(f"{a.paired_ref}_k{k}", met, avg)]
                paired.append(R4.paired_row("r3_5_roi6", pop, f"{name}_k{k}", f"{a.paired_ref}_k{k}", met, avg,
                                            pa - pb, ra - rb, len(base), len(patients), B,
                                            role="primary" if name == a.primary and k == 5 else "sensitivity"))
    S = pd.DataFrame(summ)
    r4 = pd.read_csv(R4_SUMMARY)
    r4 = r4[(r4["arm"] == "c1b") & r4["config"].isin(["patient_excluded_k5", "patient_excluded_k10"])]
    g4 = []
    for _, r in r4.iterrows():
        mine = S[(S["config"] == r["config"].replace("patient_excluded", "submitted")) & (S["metric"] == r["metric"])
                 & (S["average"] == r["average"])].iloc[0]
        g4.append(max(abs(mine[c] - r[c]) for c in ("point", "ci_lo", "ci_hi")))
    gates["G4_submitted_reproduces_r4a_vii_c1b"] = bool(len(g4) == 8 and max(g4) <= 1e-12)
    gates["G4_max_abs_diff"] = float(max(g4))
    S.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_differences.csv", index=False)
    rec = dict(gates=gates, passed=all(v for k, v in gates.items() if isinstance(v, bool)),
               configs=list(dirs), primary=a.primary, B=B, seed=SEED, n_queries=len(base), n_patients=len(patients),
               r4a_vii_executed=str(R4_EXEC), r4a_vii_sha256=R4_SHA, roi_eval=roi_eval.__file__,
               hits_dirs={k: str(v) for k, v in dirs.items()}, seconds=round(time.time() - t0, 1),
               script_sha256=sha(Path(__file__)), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "gates.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["gates"], indent=1), flush=True)
    show = S[(S["metric"] == "f1") & (S["average"] == "macro")][["config", "point", "ci_lo", "ci_hi", "role"]]
    print(show.to_string(index=False), flush=True)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
