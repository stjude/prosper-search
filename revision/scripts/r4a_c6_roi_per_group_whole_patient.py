"""C6 (D3, 2026-10-21): per-group whole-patient F1 intervals for the ROI-aggregated arms of
Fig. 6A (published = query-slide ROIs excluded; patient-excluded = every slide of the query's patient
excluded), k = 5 and k = 10 hits per ROI.

Closes the Q-Y gap: every other per-group panel (Figs. 2C, 3C, 4, S1) carries whole-patient intervals;
Fig. 6A showed points only because this derivation was not authorized.

Inputs (accepted, read-only):
  $PROSPER_OUT/r4a_vii_whole_patient_20261015/production/per_query_frame.csv.gz
      arms published_k5 / patient_excluded_k5 / published_k10 / patient_excluded_k10: per-slide
      true_label, pred_label, patient (2,778 slides, 1,876 patients)
  $PROSPER_OUT/r4a_vii_whole_patient_20261015/production/summary.csv   (gate G2)
  $PROSPER_OUT/c1b_roi_patient_excluded/roi_agg_k{5,10}_raw_hits[_patient_excluded].json  (gate G1)

Design: same kernel and draws as R4.a-vii (`make_patient_draws`, B = 5,000, PCG64 seed 123 over the
sorted 1,876 patient IDs; each slide weighted by its patient's multiplicity). Per class c and replicate:
F1_c = 2 TP_c / (T_c + P_c) with weighted counts (0 if T_c + P_c = 0; "Ambiguous" predictions count as a
false negative only). Paired per-group differences published - patient_excluded are computed per
replicate on the same draws. Intervals: nominal 95 % percentile, unadjusted.

Gates: G1 unit-weight per-class F1 equals the saved JSON `per_class_f1` (1e-9); G2 the macro / micro
replicate intervals equal the accepted r4a_vii c1b summary rows (1e-9; proves identical draws);
G3 population 2,778 / 1,876 / 15 per arm.

Outputs ($PROSPER_OUT/c6_roi_per_group_whole_patient_20261021/run_NNN/): per_group_f1.csv,
paired_differences.csv, gates.json, input_bindings.json, run_resources.json, SHA256SUMS.txt,
COMPLETE.json | FAILED.json.
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

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
OUT = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
spec = importlib.util.spec_from_file_location("ladder", os.path.join(HERE, "r2_2_tile_ladder_whole_patient.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
spec4 = importlib.util.spec_from_file_location("r4a", os.path.join(HERE, "r4a_vii_whole_patient.py"))
R4 = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(R4)

TAG = "c6_roi_per_group_whole_patient_20261021"
ACCEPTED = os.path.join(OUT, "r4a_vii_whole_patient_20261015", "production")
B, SEED, TOL = 5000, 123, 1e-9
ARMS = {"published_k5": (5, "published"), "patient_excluded_k5": (5, "patient_excluded"),
        "published_k10": (10, "published"), "patient_excluded_k10": (10, "patient_excluded")}


def sha(p):
    return L.sha256_bytes(L.read_bytes(p))


def next_run_dir():
    base = os.path.join(OUT, TAG)
    os.makedirs(base, exist_ok=True)
    n = 1 + max([int(d.split("_")[1]) for d in os.listdir(base) if d.startswith("run_")] or [0])
    d = os.path.join(base, f"run_{n:03d}")
    os.makedirs(d)
    return d


def per_class_f1_reps(W, truth, pred, classes):
    T, P = R4.onehot(truth, classes), R4.onehot(pred, classes)
    TP, Tc, Pc = W @ (T * P), W @ T, W @ P
    den = Tc + Pc
    with np.errstate(invalid="ignore", divide="ignore"):
        f1c = np.where(den > 0, 2 * TP / np.where(den > 0, den, 1.0), 0.0)
    return f1c, Tc  # (B, C), (B, C)


def main():
    t0 = time.time()
    run = next_run_dir()
    gates, bindings, ok_all = [], {}, True

    def gate(name, ok, detail, **extra):
        nonlocal ok_all
        gates.append(dict(gate=name, passed=bool(ok), detail=detail, **extra))
        ok_all &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    frame_p, summ_p = os.path.join(ACCEPTED, "per_query_frame.csv.gz"), os.path.join(ACCEPTED, "summary.csv")
    for key, p in (("accepted_frame", frame_p), ("accepted_summary", summ_p)):
        bindings[key] = dict(path=p, sha256=sha(p), bytes=os.path.getsize(p))
    jsons = {}
    for arm, (k, kind) in ARMS.items():
        p = os.path.join(OUT, R4.C1B_JSON[(k, kind)])
        h = sha(p)
        exp = R4.EXPECTED_SHA256.get(R4.C1B_JSON[(k, kind)])
        bindings[f"json_{arm}"] = dict(path=p, sha256=h, expected_sha256=exp, matches_expected=(exp is None or exp == h))
        gate(f"binding_json_{arm}", exp is None or exp == h, f"{os.path.basename(p)} {h[:16]}…")
        jsons[arm] = json.load(open(p))

    frame = pd.read_csv(frame_p, dtype={"query": str})
    summ = pd.read_csv(summ_p)
    summ = summ[summ.arm == "c1b"]
    classes = sorted(L.CLASSES)

    rows, pairs = [], []
    reps_by_arm = {}
    W = None
    for arm, (k, kind) in ARMS.items():
        f = frame[frame.config == arm].sort_values("query").reset_index(drop=True)
        gate(f"G3_population_{arm}", len(f) == R4.N_QUERIES and f.patient.nunique() == R4.N_PATIENTS
             and sorted(f.true_label.unique()) == classes, f"{len(f)} slides, {f.patient.nunique()} patients")
        if W is None:
            patients, mult, W = L.weight_matrix(f.patient.tolist(), B, SEED)
            pat_order = f.patient.tolist()
        else:
            assert f.patient.tolist() == pat_order  # same slide order -> same W
        truth, pred = f.true_label.tolist(), f.pred_label.tolist()
        f1_unit, _ = per_class_f1_reps(np.ones((1, len(f))), truth, pred, classes)
        saved = jsons[arm]["per_class_f1"]
        worst = max(abs(f1_unit[0, i] - saved[c]) for i, c in enumerate(classes))
        gate(f"G1_unit_per_class_f1_equals_json_{arm}", worst <= TOL, f"max |diff| vs saved per_class_f1 = {worst:.2e}")
        f1c, Tc = per_class_f1_reps(W, truth, pred, classes)
        reps_by_arm[arm] = f1c
        macro = f1c.mean(1)
        macro[(Tc == 0).any(1)] = np.nan
        correct = (np.asarray(truth, dtype=object) == np.asarray(pred, dtype=object)).astype(float)
        micro = L.boot_means(W, correct)
        worst2 = 0.0
        for avg, reps in (("macro", macro), ("micro", micro)):
            s = summ[(summ.config == arm) & (summ.metric == "f1") & (summ.average == avg)].iloc[0]
            lo, hi, _ = L.ci(reps)
            worst2 = max(worst2, abs(s.ci_lo - lo), abs(s.ci_hi - hi))
        gate(f"G2_macro_micro_intervals_equal_accepted_{arm}", worst2 <= TOL,
             f"max |ci difference| vs accepted r4a_vii c1b rows = {worst2:.2e}")
        for i, c in enumerate(classes):
            lo, hi, bad = L.ci(f1c[:, i])
            m = f.true_label == c
            rows.append(dict(arm=arm, k=k, aggregation=kind, group=c, n_slides=int(m.sum()), n_patients=f.loc[m, "patient"].nunique(),
                             f1=float(f1_unit[0, i]), ci_lo=lo, ci_hi=hi, n_replicates=B, n_nonfinite=bad,
                             flag=L.boundary_flag(lo, hi), interval="whole-patient percentile 95%, B=5000, seed 123, shared with R4.a-vii"))
    for k in (5, 10):
        a, b = reps_by_arm[f"published_k{k}"], reps_by_arm[f"patient_excluded_k{k}"]
        pa = [r for r in rows if r["arm"] == f"published_k{k}"]
        pb = [r for r in rows if r["arm"] == f"patient_excluded_k{k}"]
        for i, c in enumerate(classes):
            d = a[:, i] - b[:, i]
            lo, hi, bad = L.ci(d)
            pairs.append(dict(k=k, group=c, minuend="published", subtrahend="patient_excluded", point=pa[i]["f1"] - pb[i]["f1"],
                              ci_lo=lo, ci_hi=hi, n_nonfinite=bad, excludes_zero=bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                              note=R4.PAIRED_DEF if isinstance(R4.PAIRED_DEF, str) else " ".join(R4.PAIRED_DEF)))

    pd.DataFrame(rows).to_csv(os.path.join(run, "per_group_f1.csv"), index=False)
    pd.DataFrame(pairs).to_csv(os.path.join(run, "paired_differences.csv"), index=False)
    json.dump(gates, open(os.path.join(run, "gates.json"), "w"), indent=2)
    json.dump(dict(inputs=bindings, design=dict(B=B, seed=SEED, draws="make_patient_draws over sorted UID.Subject",
                                                  mult_sha256=R4.mult_sha(mult)),
                   kernel_sha256=sha(os.path.join(HERE, "r2_2_tile_ladder_whole_patient.py")),
                   r4a_vii_sha256=sha(os.path.join(HERE, "r4a_vii_whole_patient.py")), script_sha256=sha(os.path.abspath(__file__))),
              open(os.path.join(run, "input_bindings.json"), "w"), indent=2)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=R, stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"unavailable: {e}"
    json.dump(dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, host=platform.node(),
                   wall_seconds=round(time.time() - t0, 1), max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                   workspace_head=head, argv=sys.argv, lsf_jobid=os.environ.get("LSB_JOBID")),
              open(os.path.join(run, "run_resources.json"), "w"), indent=2)
    with open(os.path.join(run, "SHA256SUMS.txt"), "w") as fh:
        for name in sorted(os.listdir(run)):
            if name != "SHA256SUMS.txt":
                fh.write(f"{sha(os.path.join(run, name))}  {name}\n")
    final = "COMPLETE.json" if ok_all else "FAILED.json"
    json.dump(dict(passed=ok_all, failed=[g["gate"] for g in gates if not g["passed"]], run=run),
              open(os.path.join(run, final), "w"), indent=2)
    print(f"{final} -> {run}")
    df = pd.DataFrame(rows)
    print(df[df.k == 5].pivot(index="group", columns="aggregation", values=["f1", "ci_lo", "ci_hi"]).round(3).to_string())
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
