#!/usr/bin/env python3
"""R3.5-v / Table S16: whole-patient bootstrap intervals for the slide-grouped out-of-fold tile classifier.

Inputs (read-only, hashes recorded): the pooled 5-fold slide-GroupKFold OOF probabilities written by
scripts/s16_grouped_tile_classifier.py (job 323489975) and the annotation split arrays it consumed, plus the
Slide.Scan.File -> UID.Subject map used by every accepted whole-patient script.

Bootstrap: [decision record] whole-patient convention, reusing the accepted ladder helpers verbatim through importlib
(scripts/r2_2_tile_ladder_whole_patient.py: make_patient_draws / ci / load_meta / sha256). Resampling unit =
patient; P1 keeps all 237 annotated slides with each unmapped slide as its own singleton unit; P2 keeps the
202 mapped slides / 192 patients (the R2.1 Stage1a convention). Tiles are weighted by their unit's multiplicity.

Statistics per replicate: tie-correct weighted one-vs-rest AUROC per class (unique-score grouping), tumour
sensitivity/specificity at argmax, 5-class weighted tile accuracy. Percentile 2.5/97.5, B=5000, PCG64(123),
unadjusted. Classes whose positive tiles sit in fewer than MIN_POS_UNITS resampling units get a point only.

Limitation (carried into README): folds are slide-grouped; 7 patients (16 slides) span folds. The interval
covers patient-resampling uncertainty given those fold models, not the between-fold patient leakage.
"""
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
LADDER = os.path.join(HERE, "r2_2_tile_ladder_whole_patient.py")
LADDER_SHA_ACCEPTED = "38a452c2"  # prefix of the accepted ladder script hash ([analysis plan], R2.2); enforced below

A = "/PROSPER_WORKSPACE/comet-annotation"
SPLIT = f"{A}/output/pathologist_annotated_train_test_split_uni2"
NAMES = {1: "Tumor", 2: "Immune cells", 3: "Necrosis", 4: "Non-tumor", 5: "Artifacts"}
EXPECTED_TUMOR_OOF_AUROC = 0.9457737203096972
MIN_POS_UNITS = 5


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ladder():
    sha = sha256_file(LADDER)
    if not sha.startswith(LADDER_SHA_ACCEPTED):
        raise RuntimeError(f"ladder script hash {sha} does not match accepted prefix {LADDER_SHA_ACCEPTED}")
    spec = importlib.util.spec_from_file_location("ladder", LADDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha


def weighted_auroc_codes(code, is_pos, w, n_unique):
    """Tie-correct weighted Mann-Whitney AUROC. code = rank code of each tile's score among unique sorted scores."""
    posw = np.bincount(code, weights=w * is_pos, minlength=n_unique)
    negw = np.bincount(code, weights=w * (~is_pos), minlength=n_unique)
    wp, wn = posw.sum(), negw.sum()
    if wp <= 0 or wn <= 0:
        return np.nan
    neg_below = np.cumsum(negw) - negw
    return float(np.sum(posw * (neg_below + 0.5 * negw)) / (wp * wn))


def main():
    t0 = time.time()
    out_root = os.environ.get("PROSPER_OUT")
    if not out_root:
        raise RuntimeError("PROSPER_OUT must be exported")
    B = int(os.environ.get("S16_B", "5000"))
    OUT = os.path.join(out_root, "r3_5_s16_whole_patient_ci_20261008" + ("" if B == 5000 else f"_smoke{B}"))
    os.makedirs(OUT, exist_ok=False)
    L, ladder_sha = load_ladder()

    inputs = {
        "oof_probs": f"{out_root}/s16_grouped/oof_probs.npy",
        "s16_reeval": f"{out_root}/s16_grouped/s16_reeval.json",
        "per_slide_oof": f"{out_root}/s16_grouped/per_slide_oof.csv",
        "y_train_rand": f"{SPLIT}/y_train_rand.npy",
        "y_test_rand": f"{SPLIT}/y_test_rand.npy",
        "idx_train_rand": f"{SPLIT}/idx_train_rand.npy",
        "idx_test_rand": f"{SPLIT}/idx_test_rand.npy",
        "slide_info": f"{SPLIT}/slide_info.csv",
        "meta": L.META,
    }
    hashes = {k: sha256_file(v) for k, v in inputs.items()}

    oof = np.load(inputs["oof_probs"])
    y = np.concatenate([np.load(inputs["y_train_rand"]), np.load(inputs["y_test_rand"])])
    with open(inputs["slide_info"]) as f:
        sid_all = np.array([r["slide_id"] for r in csv.DictReader(f)])
    idx = np.concatenate([np.load(inputs["idx_train_rand"]), np.load(inputs["idx_test_rand"])])
    sid = sid_all[idx]
    classes = np.unique(y)
    if oof.shape != (len(y), len(classes)):
        raise RuntimeError(f"oof shape {oof.shape} vs labels {len(y)} x {len(classes)}")
    if list(classes) != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"unexpected class codes {classes}")

    # --- point reproduction against the frozen S16 outputs ---
    reeval = json.load(open(inputs["s16_reeval"]))
    pt_auroc = {NAMES[c]: float(roc_auc_score((y == c).astype(int), oof[:, i])) for i, c in enumerate(classes)}
    if abs(pt_auroc["Tumor"] - EXPECTED_TUMOR_OOF_AUROC) > 1e-9:
        raise RuntimeError(f"tumour OOF AUROC {pt_auroc['Tumor']} != expected {EXPECTED_TUMOR_OOF_AUROC}")
    for k, v in reeval["slide_groupkfold_5"]["auroc_ovr"].items():
        if abs(pt_auroc[k] - v) > 1e-9:
            raise RuntimeError(f"{k} OOF AUROC {pt_auroc[k]} != s16_reeval {v}")
    pred = classes[oof.argmax(1)]
    t, tp = y == 1, pred == 1
    pt_sens = float((tp & t).sum() / t.sum())
    pt_spec = float((~tp & ~t).sum() / (~t).sum())
    pt_acc = float((pred == y).mean())
    tv = reeval["slide_groupkfold_5"]["tumour_vs_rest"]
    if abs(pt_sens - tv["sensitivity"]) > 1e-9 or abs(pt_spec - tv["specificity"]) > 1e-9:
        raise RuntimeError("tumour sensitivity/specificity do not reproduce s16_reeval")
    ps = pd.read_csv(inputs["per_slide_oof"], dtype={"slide": str})
    max_dev = 0.0
    for s in ps.itertuples():
        m = sid == s.slide
        yy, pp = y[m], pred[m]
        tt, tpp = yy == 1, pp == 1
        max_dev = max(max_dev, abs(round(float((tpp == tt).mean()), 4) - s.acc), abs(int(m.sum()) - s.n_annot))
        if tt.any():
            max_dev = max(max_dev, abs(round(float((tpp & tt).sum() / tt.sum()), 4) - s.sens))
    if max_dev > 1e-12:
        raise RuntimeError(f"per_slide_oof.csv not reproduced (max dev {max_dev})")

    # --- patient map ---
    meta, meta_sha = L.load_meta(inputs["meta"])
    if meta_sha != hashes["meta"]:
        raise RuntimeError("meta hash mismatch between loaders")
    slides = sorted(set(sid.tolist()))
    pat_of_slide = {s: (meta["UID.Subject"][s] if s in meta.index else None) for s in slides}
    mapped = [s for s in slides if pat_of_slide[s] is not None]
    unmapped = [s for s in slides if pat_of_slide[s] is None]

    populations = {
        "P1_all237_unmapped_singletons": {
            "slides": slides,
            "unit_of_slide": {s: (pat_of_slide[s] if pat_of_slide[s] else f"SLIDE:{s}") for s in slides},
        },
        "P2_mapped202_192patients": {
            "slides": mapped,
            "unit_of_slide": {s: pat_of_slide[s] for s in mapped},
        },
    }

    summary_rows, unit_rows, pop_info = [], [], {}
    for pname, pop in populations.items():
        keep = np.isin(sid, pop["slides"])
        y_p, sid_p, pred_p = y[keep], sid[keep], pred[keep]
        unit_of_tile = np.array([pop["unit_of_slide"][s] for s in sid_p])
        units, mult = L.make_patient_draws(unit_of_tile.tolist(), count=B, seed=123)
        unit_index = {u: k for k, u in enumerate(units)}
        tile_unit = np.array([unit_index[u] for u in unit_of_tile], dtype=np.int64)
        n_pat = len([u for u in units if not u.startswith("SLIDE:")])
        n_single = len(units) - n_pat
        pop_info[pname] = dict(n_slides=len(pop["slides"]), n_tiles=int(keep.sum()), n_units=len(units),
                               n_patient_units=n_pat, n_singleton_slide_units=n_single)
        for u in units:
            sl = sorted({s for s in pop["slides"] if pop["unit_of_slide"][s] == u})
            unit_rows.append(dict(population=pname, unit=u, unit_type="singleton_slide" if u.startswith("SLIDE:") else "patient",
                                  n_slides=len(sl), slides=";".join(sl), n_tiles=int(np.isin(sid_p, sl).sum())))

        # unit-weight check: weighted AUROC with all weights 1 must equal sklearn; codes recomputed per population
        w1 = np.ones(len(y_p))
        pop_codes = []
        for i in range(len(classes)):
            u, inv = np.unique(oof[keep, i], return_inverse=True)
            pop_codes.append((inv.astype(np.int64), len(u)))
        pt_pop = {}
        for i, c in enumerate(classes):
            is_pos = y_p == c
            a = weighted_auroc_codes(pop_codes[i][0], is_pos, w1, pop_codes[i][1])
            ref = float(roc_auc_score(is_pos.astype(int), oof[keep, i]))
            if abs(a - ref) > 1e-10:
                raise RuntimeError(f"{pname} {NAMES[c]}: weighted AUROC {a} != sklearn {ref}")
            pt_pop[NAMES[c]] = a
        t_p, tp_p = y_p == 1, pred_p == 1
        pt_pop["tumour_sensitivity"] = float((tp_p & t_p).sum() / t_p.sum())
        pt_pop["tumour_specificity"] = float((~tp_p & ~t_p).sum() / (~t_p).sum())
        pt_pop["tile_accuracy_5class"] = float((pred_p == y_p).mean())

        pos_units = {NAMES[c]: int(len(set(tile_unit[y_p == c].tolist()))) for c in classes}
        pos_slides = {NAMES[c]: int(len(set(sid_p[y_p == c].tolist()))) for c in classes}
        est_classes = [c for c in classes if pos_units[NAMES[c]] >= MIN_POS_UNITS]

        reps = {NAMES[c]: np.full(B, np.nan) for c in est_classes}
        reps.update(tumour_sensitivity=np.full(B, np.nan), tumour_specificity=np.full(B, np.nan),
                    tile_accuracy_5class=np.full(B, np.nan))
        correct = (pred_p == y_p).astype(float)
        for b in range(B):
            w = mult[b][tile_unit].astype(float)
            W = w.sum()
            for c in est_classes:
                i = int(np.where(classes == c)[0][0])
                reps[NAMES[c]][b] = weighted_auroc_codes(pop_codes[i][0], y_p == c, w, pop_codes[i][1])
            wt, wtp = (w * t_p).sum(), (w * (tp_p & t_p)).sum()
            wn, wtn = (w * (~t_p)).sum(), (w * (~tp_p & ~t_p)).sum()
            reps["tumour_sensitivity"][b] = wtp / wt if wt > 0 else np.nan
            reps["tumour_specificity"][b] = wtn / wn if wn > 0 else np.nan
            reps["tile_accuracy_5class"][b] = (w * correct).sum() / W
            if b % 1000 == 0:
                print(f"{pname} rep {b} {time.time()-t0:.0f}s", flush=True)

        for c in classes:
            nm = NAMES[c]
            row = dict(population=pname, statistic=f"auroc_ovr_{nm}", n_tiles=int(len(y_p)), n_pos_tiles=int((y_p == c).sum()),
                       n_slides_with_class=pos_slides[nm], n_units_with_class=pos_units[nm], point=pt_pop[nm])
            if c in est_classes:
                lo, hi, bad = L.ci(reps[nm])
                row.update(ci_lo=lo, ci_hi=hi, n_reps_finite=int(np.isfinite(reps[nm]).sum()), n_reps_nonfinite=bad,
                           interval="whole-patient B=%d" % B)
            else:
                row.update(ci_lo="", ci_hi="", n_reps_finite="", n_reps_nonfinite="",
                           interval=f"not estimable: positives in {pos_units[nm]} units (<{MIN_POS_UNITS})")
            summary_rows.append(row)
        for k in ("tumour_sensitivity", "tumour_specificity", "tile_accuracy_5class"):
            lo, hi, bad = L.ci(reps[k])
            summary_rows.append(dict(population=pname, statistic=k, n_tiles=int(len(y_p)), n_pos_tiles=int(t_p.sum()),
                                     n_slides_with_class=pos_slides["Tumor"], n_units_with_class=pos_units["Tumor"],
                                     point=pt_pop[k], ci_lo=lo, ci_hi=hi, n_reps_finite=int(np.isfinite(reps[k]).sum()),
                                     n_reps_nonfinite=bad,
                                     interval="whole-patient B=%d" % B))

    pd.DataFrame(summary_rows).to_csv(f"{OUT}/summary.csv", index=False)
    pd.DataFrame(unit_rows).to_csv(f"{OUT}/units.csv", index=False)

    # fold-crossing patients (disclosure)
    fold_of_slide = dict(zip(ps["slide"], ps["fold"]))
    cross = {}
    for s in mapped:
        cross.setdefault(pat_of_slide[s], set()).add(int(fold_of_slide[s]))
    crossing = {p: sorted(f) for p, f in cross.items() if len(f) > 1}
    rec = dict(
        script=os.path.basename(__file__), script_sha256=sha256_file(os.path.abspath(__file__)),
        ladder_script=LADDER, ladder_sha256=ladder_sha, input_sha256=hashes, B=B, seed=123, generator="PCG64",
        convention="ADR0021 whole-patient: sorted units, rng.integers(p,size=p) bincount multiplicities, tiles weighted by unit multiplicity, percentile 2.5/97.5, unadjusted",
        point_reproduction=dict(auroc_ovr=pt_auroc, tumour_sensitivity=pt_sens, tumour_specificity=pt_spec, tile_accuracy=pt_acc,
                                expected_tumor_auroc=EXPECTED_TUMOR_OOF_AUROC, per_slide_oof_max_dev=max_dev),
        populations=pop_info, n_unmapped_slides=len(unmapped), unmapped_slides=unmapped,
        fold_crossing_patients=dict(n_patients=len(crossing), n_slides=sum(len([s for s in mapped if pat_of_slide[s] == p]) for p in crossing),
                                    detail=crossing),
        min_pos_units_for_interval=MIN_POS_UNITS, runtime_s=round(time.time() - t0, 1),
    )
    json.dump(rec, open(f"{OUT}/reconciliation.json", "w"), indent=1)
    with open(f"{OUT}/README.md", "w") as f:
        f.write("# R3.5-v Table S16 whole-patient intervals (slide-grouped OOF tile classifier)\n\n")
        f.write(__doc__ + "\n")
        f.write(f"Run: B={B}, seed 123, runtime {rec['runtime_s']} s. Populations: {json.dumps(pop_info)}\n\n")
        f.write(f"Fold-crossing patients: {len(crossing)} ({rec['fold_crossing_patients']['n_slides']} slides) span more than one fold.\n")
        f.write("Immune cells / Artifacts intervals are reported as not estimable (positives in too few units).\n")
    with open(f"{OUT}/sha256.txt", "w") as f:
        for n in sorted(os.listdir(OUT)):
            if n not in ("sha256.txt", "COMPLETE"):
                f.write(f"{sha256_file(os.path.join(OUT, n))}  {n}\n")
    open(f"{OUT}/COMPLETE", "w").write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print("wrote", OUT, f"{time.time()-t0:.0f}s")
    print(pd.DataFrame(summary_rows)[["population", "statistic", "point", "ci_lo", "ci_hi", "interval"]].to_string())


if __name__ == "__main__":
    sys.exit(main())
