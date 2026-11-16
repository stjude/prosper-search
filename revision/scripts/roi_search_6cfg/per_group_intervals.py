#!/usr/bin/env python3
"""R3.5-ROI6 display support: per-group F1 with whole-patient intervals for the six
configurations of the revised Fig. 6A (2026-11-12: "all three in both l1 and cosine
on figure 6a. so 6 config").

Reads the per-slide predictions of ``eval/run_001`` and applies the C6 per-class kernel
(``r4a_c6_roi_per_group_whole_patient.per_class_f1_reps``) on the R4a-vii draws (B = 5,000,
seed 123). Paired per-group differences are against ``v2_l1``, per replicate. No hit is
re-read and no prediction recomputed.

Gates: G1 unit-weight per-class F1 equals scikit-learn on the same predictions; G2 macro and
micro intervals equal the ``run_001`` summary rows (same draws); G3 the submitted hits'
per-group rows equal the accepted C6 ``patient_excluded_k5`` / ``_k10`` rows (the current
Fig. 6A source).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
ROOT = OUT / "roi_search_6cfg_20261111"
EVAL = ROOT / "eval" / "run_001"
C6 = OUT / "c6_roi_per_group_whole_patient_20261021" / "run_001" / "per_group_f1.csv"
B, SEED, TOL, REF = 5000, 123, 1e-9, "v2_l1"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    t0 = time.time()
    C6M = load("c6", "r4a_c6_roi_per_group_whole_patient.py")
    L = C6M.L
    out = ROOT / "eval" / "per_group_001"
    out.mkdir(parents=True, exist_ok=False)
    fr = pd.read_csv(EVAL / "per_query.csv.gz", dtype={"query": str, "patient": str})
    summ = pd.read_csv(EVAL / "summary.csv")
    c6 = pd.read_csv(C6)
    classes = sorted(fr["true_label"].unique())
    gates, rows, pairs, reps = {}, [], [], {}
    W, order = None, None
    for (cfg, k), f in fr.groupby(["config", "k"], sort=False):
        f = f.sort_values("query").reset_index(drop=True)
        if W is None:
            _, mult, W = L.weight_matrix(f["patient"].tolist(), B, SEED)
            order = f["patient"].tolist()
        assert f["patient"].tolist() == order
        truth, pred = f["true_label"].tolist(), f["pred_label"].tolist()
        unit, _ = C6M.per_class_f1_reps(np.ones((1, len(f))), truth, pred, classes)
        sk = f1_score(truth, pred, labels=classes, average=None, zero_division=0)
        gates[f"G1_{cfg}_k{k}"] = float(np.abs(unit[0] - sk).max()) <= TOL
        f1c, Tc = C6M.per_class_f1_reps(W, truth, pred, classes)
        macro = f1c.mean(1)
        macro[(Tc == 0).any(1)] = np.nan
        micro = L.boot_means(W, (np.asarray(truth, dtype=object) == np.asarray(pred, dtype=object)).astype(float))
        worst = 0.0
        for avg, r in (("macro", macro), ("micro", micro)):
            s = summ[(summ["config"] == f"{cfg}_k{k}") & (summ["metric"] == "f1") & (summ["average"] == avg)].iloc[0]
            lo, hi, _ = L.ci(r)
            worst = max(worst, abs(s["ci_lo"] - lo), abs(s["ci_hi"] - hi))
        gates[f"G2_{cfg}_k{k}"] = worst <= TOL
        reps[(cfg, k)] = f1c
        for i, c in enumerate(classes):
            lo, hi, bad = L.ci(f1c[:, i])
            m = f["true_label"] == c
            rows.append(dict(config=cfg, k=k, group=c, n_slides=int(m.sum()), n_patients=int(f.loc[m, "patient"].nunique()),
                             f1=float(unit[0, i]), ci_lo=lo, ci_hi=hi, n_replicates=B, n_nonfinite=bad,
                             interval="whole-patient percentile 95%, B=5000, seed 123, R4a-vii draws"))
    df = pd.DataFrame(rows)
    for k in (5, 10):
        a = df[(df.config == "submitted") & (df.k == k)].set_index("group")
        b = c6[c6.arm == f"patient_excluded_k{k}"].set_index("group")
        gates[f"G3_submitted_k{k}_equals_c6"] = bool(
            a.index.sort_values().equals(b.index.sort_values())
            and max((a.loc[b.index, c] - b[c]).abs().max() for c in ("f1", "ci_lo", "ci_hi")) <= TOL)
        for cfg in df.config.unique():
            if cfg in ("submitted", REF):
                continue
            d = reps[(cfg, k)] - reps[(REF, k)]
            for i, c in enumerate(classes):
                lo, hi, bad = L.ci(d[:, i])
                pa = df[(df.config == cfg) & (df.k == k) & (df.group == c)].f1.iloc[0]
                pb = df[(df.config == REF) & (df.k == k) & (df.group == c)].f1.iloc[0]
                pairs.append(dict(k=k, group=c, minuend=cfg, subtrahend=REF, point=pa - pb, ci_lo=lo, ci_hi=hi,
                                  n_nonfinite=bad, note="per replicate on the same draws; nominal 95% percentile, unadjusted"))
    df.to_csv(out / "per_group_f1.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "paired_per_group_vs_v2_l1.csv", index=False)
    gates["draws_identical_to_r4a_vii_c1b"] = C6M.R4.mult_sha(mult) == "6aeaf004817321ec20c001812aa02fb9109c42c7e00fe8479b94aafeb685f6db"
    rec = dict(gates=gates, passed=all(gates.values()), mult_sha256=C6M.R4.mult_sha(mult),
               inputs={str(p): sha(p) for p in (EVAL / "per_query.csv.gz", EVAL / "summary.csv", C6)},
               script_sha256=sha(Path(__file__)), seconds=round(time.time() - t0, 1),
               created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "gates.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(gates, indent=1))
    print(df[df.k == 5].pivot(index="group", columns="config", values="f1").round(3).to_string())
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
