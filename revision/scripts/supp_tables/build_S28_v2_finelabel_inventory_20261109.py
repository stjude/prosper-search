#!/usr/bin/env python3
"""Workspace Table S28 v2 (final S9 on 2026-11-09): inventory of the evaluated fine-label tasks with a
representation column and both representations for the two main-text internal arms.

Assembly only. v1 (supp_tables_20261022/A_ladder_failure_finelabel/R3.7-T5_S28_S-D2*) is read, not modified.
Sources (accepted packages; read-only):
  v1 rows / frozen inventory: r3_7_task_inventory_20261004b/task_inventory.csv (= v1 full csv; asserted)
  frozen fine-label summary:  r3_7_fine_label_20261003/summary_all_arms.csv (traces every v1 point and interval)
  representation per arm:      s6_*/result_<arm>.json config.matrix
  all-tissue rows:             r3_7_all_tissue_ci_20261103/{results.json, s26_rows_all_tissue.csv}
                               (= the Supplementary Table S8 / workspace S26 v2 values; asserted)
  all-tissue points, other internal arms (audit INFO only): r3_7_audit_20261020/results_seed20261021_B5000.json
Output: .../B_external_perturbation_finelabel/S28_S9_v2/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _supp_common_20261109 import OUT_ROOT, REPO, Build, sha  # noqa: E402

V1 = OUT_ROOT / "supp_tables_20261022/A_ladder_failure_finelabel"
INV = OUT_ROOT / "r3_7_task_inventory_20261004b"
FINE = OUT_ROOT / "r3_7_fine_label_20261003"
AT = OUT_ROOT / "r3_7_all_tissue_ci_20261103"
AUD = OUT_ROOT / "r3_7_audit_20261020"
S26V2 = OUT_ROOT / "supp_tables_20261103/A_ladder_failure_finelabel"
RESULT = {"s61_int_2c": "s6_1_rms", "s61_int_3c": "s6_1_rms", "s61_int_2c_xeno": "s6_1_rms", "s61_fusion": "s6_1_rms",
          "s61_ext_2c": "s6_1_rms", "s61_ext_3c": "s6_1_rms", "s62": "s6_2_mycn", "s62_xeno": "s6_2_mycn", "s64": "s6_4_hbhcc",
          "s65_inpc": "s6_5_nbdiff", "s65b_gnb": "s6_5_nbdiff", "s65_ext_3c": "s6_5_kidrare_nb", "s65_ext_binary": "s6_5_kidrare_nb"}
REP_TUMOUR = "tumour-filtered (precomputed tumour-only tile selection; tile classifier not held out by patient)"
REP_ALL_CCDI = "all-tissue (CCDI-MCI x COMET mosaics)"
REP_ALL_KIDRARE = "all-tissue (KidRare mosaics)"
REP_ALL_COMET = "all-tissue (COMET mosaics)"
MAIN_VALUE = {"s61_int_2c": 0.847, "s64": 0.914}
CI_TXT = "whole-patient 95% CI"
DASH = "\u2013"


def r3(x) -> str:
    return f"{float(x):.3f}"


def main() -> None:
    b = Build("S28", "S28_S9_v2", "scripts/supp_tables/build_S28_v2_finelabel_inventory_20261109.py", "### R3.7-T5",
              "R3.7-T5 (v1 table, 2026-10-22 build, C1 = R3.7 F1 option (a) legend insertion) plus the R3.7-T4 version-2 note "
              "(2026-11-03, C574: all-tissue reported rows; tumour-only values to the S28 notes) and the S9 legend in "
              "docs/prosper_revised/prosper_supplementary_legends.md (\u2018a `representation` column states which\u2019).")

    # ---------------- v1 (read only) ----------------
    v1_full = b.csv(V1 / "R3.7-T5_S28_S-D2_task_inventory_full.csv", "v1 S28 full csv (13 x 35, incl. num_*)", dtype=str, keep_default_na=False)
    v1_tab = b.csv(V1 / "R3.7-T5_S28_S-D2_table_verbatim.csv", "v1 S28 table (13 x 25)", dtype=str, keep_default_na=False)
    b.bind(V1 / "R3.7-T5_S28_S-D2.xlsx", "v1 S28 workbook (retained unchanged)")
    v1_sums = (V1 / "SHA256SUMS.txt").read_text(encoding="utf-8")
    b.bind(V1 / "SHA256SUMS.txt", "v1 package SHA256SUMS")
    for name in ("R3.7-T5_S28_S-D2.xlsx", "R3.7-T5_S28_S-D2_task_inventory_full.csv", "R3.7-T5_S28_S-D2_table_verbatim.csv"):
        b.check(f"v1 untouched: {name} sha256 equals the v1 package SHA256SUMS", f"{sha(V1 / name)}  {name}" in v1_sums)
    frozen = b.csv(INV / "task_inventory.csv", "frozen task inventory (r3_7_task_inventory_20261004b)", dtype=str, keep_default_na=False)
    b.bind(INV / "task_inventory.md", "frozen task inventory (markdown)")
    b.bind(INV / "reconciliation.json", "frozen inventory reconciliation")
    b.check("v1 full csv equals the frozen task_inventory.csv (13 x 35, all cells)", v1_full.equals(frozen) and v1_full.shape == (13, 35))
    b.check("v1 table = first 25 columns of the full csv", v1_tab.equals(v1_full[v1_tab.columns]) and list(v1_tab.columns) == list(v1_full.columns[:25]))

    # ---------------- trace every v1 point / interval to the frozen summary ----------------
    summ = b.csv(FINE / "summary_all_arms.csv", "frozen fine-label summary (points and whole-patient intervals)")
    b.bind(FINE / "reconciliation.json", "frozen fine-label reconciliation")
    trace = []
    for r in v1_full.itertuples():
        s = summ[summ.arm == r.arm].set_index("class")
        allr, mac = s.loc["ALL"], s.loc["MACRO"]
        n = int(r.n_queries)
        k = int(r.num_plurality5_correct)
        b.check(f"{r.arm}: plurality@5 {k}/{n} = frozen ALL plurality5_recall", abs(k / n - allr.plurality5_recall) < 1e-9 and r.plurality5_accuracy == f"{k}/{n} ({r3(k / n)})")
        trace.append((r.arm, "tumour-filtered" if r.arm.startswith(("s61_int", "s61_fusion", "s62", "s64", "s65_inpc", "s65b")) else "all-tissue",
                      "plurality5_accuracy", r.plurality5_accuracy, f"r3_7_fine_label_20261003/summary_all_arms.csv [{r.arm}, ALL].plurality5_recall = {allr.plurality5_recall:.6f}"))
        b.check(f"{r.arm}: mean class recall = frozen MACRO plurality5_recall", r.mean_class_recall == r3(mac.plurality5_recall) and abs(float(r.num_mean_class_recall) - mac.plurality5_recall) < 1e-6)
        trace.append((r.arm, trace[-1][1], "mean_class_recall", r.mean_class_recall, f"summary_all_arms.csv [{r.arm}, MACRO].plurality5_recall = {mac.plurality5_recall:.6f}"))
        if bool(allr.ci_patient_level):
            for col, row, lo, hi in (("plurality5_accuracy_ci", allr, "plurality5_recall_lo", "plurality5_recall_hi"),
                                     ("mean_class_recall_ci", mac, "plurality5_recall_lo", "plurality5_recall_hi")):
                want = f"{r3(row[lo])}{DASH}{r3(row[hi])} ({CI_TXT})"
                b.check(f"{r.arm}: {col} '{getattr(r, col)}' = frozen interval", getattr(r, col) == want, want)
                trace.append((r.arm, trace[-1][1], col, getattr(r, col), f"summary_all_arms.csv [{r.arm}, {row.name}].{lo}/{hi} = {row[lo]:.6f}/{row[hi]:.6f}"))
            if not r.strict5_accuracy.startswith("="):
                want = f"{r3(allr.strict5_recall_lo)}{DASH}{r3(allr.strict5_recall_hi)} ({CI_TXT})"
                b.check(f"{r.arm}: strict5_accuracy_ci = frozen interval", r.strict5_accuracy_ci == want, want)
                trace.append((r.arm, trace[-1][1], "strict5_accuracy_ci", r.strict5_accuracy_ci, f"summary_all_arms.csv [{r.arm}, ALL].strict5_recall_lo/hi"))
            if ";" not in r.num_minority_n:
                mc = r.minority_class_recall.split()[0]
                m = s.loc[mc]
                want = f"{r3(m.plurality5_recall_lo)}{DASH}{r3(m.plurality5_recall_hi)} ({CI_TXT})"
                b.check(f"{r.arm}: minority ({mc}) recall {r.num_minority_correct}/{r.num_minority_n} and interval = frozen", r.minority_class_recall_ci == want
                        and abs(int(r.num_minority_correct) / int(r.num_minority_n) - m.plurality5_recall) < 1e-9, want)
                trace.append((r.arm, trace[-1][1], "minority_class_recall(_ci)", f"{r.minority_class_recall}; {r.minority_class_recall_ci}", f"summary_all_arms.csv [{r.arm}, {mc}].plurality5_recall(_lo/_hi)"))
        else:
            b.check(f"{r.arm}: frozen interval not patient-level and v1 prints no interval", "no whole-patient CI" in r.plurality5_accuracy_ci and r.unit == "slide")
            trace.append((r.arm, "all-tissue", "plurality5_accuracy_ci", r.plurality5_accuracy_ci, f"summary_all_arms.csv [{r.arm}].ci_patient_level = False"))

    # ---------------- representation per arm, from the executed configuration ----------------
    rep, rep_rows = {}, []
    for arm, d in RESULT.items():
        rj = b.json(OUT_ROOT / d / f"result_{arm}.json", f"executed configuration for {arm} (config.matrix)")
        mat = rj["config"]["matrix"]
        if "tumor_only" in Path(mat).name:
            rep[arm] = REP_TUMOUR
        elif "ccdi_uni2_ot_cosine" in Path(mat).name:
            rep[arm] = REP_ALL_CCDI
        elif "block_kidrare_nb" in Path(mat).name:
            rep[arm] = REP_ALL_KIDRARE
        else:
            raise AssertionError(f"unrecognised matrix for {arm}: {mat}")
        rep_rows.append(dict(arm=arm, representation=rep[arm], matrix=mat, source=f"$PROSPER_OUT/{d}/result_{arm}.json config.matrix"))
    b.bind(REPO / "scripts/s6_5_build_kidrare_nb_block.py", "KidRare block builder (FEATS = features/kidrare_uni2, no tile filter)")
    kb = (REPO / "scripts/s6_5_build_kidrare_nb_block.py").read_text(encoding="utf-8")
    b.check("KidRare block built from features/kidrare_uni2 without a tumour-tile filter", 'FEATS = f"{OUT}/features/kidrare_uni2"' in kb and "tumor_only" not in kb and "tumour_only" not in kb)
    internal = [a for a in RESULT if not a.startswith(("s61_ext", "s65_ext"))]
    b.check("all 9 internal COMET arms ranked on the tumour-only matrix; 2 CCDI-MCI and 2 KidRare arms on all-tissue",
            all(rep[a] == REP_TUMOUR for a in internal) and len(internal) == 9 and
            [rep[a] for a in ("s61_ext_2c", "s61_ext_3c")] == [REP_ALL_CCDI] * 2 and [rep[a] for a in ("s65_ext_3c", "s65_ext_binary")] == [REP_ALL_KIDRARE] * 2)
    aud = b.json(AUD / "results_seed20261021_B5000.json", "2026-10-20 audit trace (all-tissue matrix points for internal arms, INFO)")
    b.check("audit trace re-ranked internal arms on the same tumour-only matrix path", "tumor_only" in aud["inputs"]["tumour_only_matrix"]["path"])

    # ---------------- all-tissue rows (accepted 11-03 run) ----------------
    res = b.json(AT / "results.json", "all-tissue marginal CI run 2026-11-03 (seed 123, B 5000)")
    atr = b.csv(AT / "s26_rows_all_tissue.csv", "all-tissue / tumour-only rows (11-03)")
    b.bind(AT / "all_tissue_marginal_ci.py", "11-03 run script")
    b.bind(AT / "run.log", "11-03 run log")
    b.bind(AT / "lead_check.log", "11-03 independent lead check log")
    b.check("11-03 run: 36 reproduction checks, all PASS", len(res["checks"]) == 36 and all(c["status"] == "PASS" for c in res["checks"]))
    b.check("11-03 run: seed 123, B 5000", res["seed"] == 123 and res["B"] == 5000)
    b.check("11-03 run used the same tumour-only matrix as the frozen arms and the audit all-tissue matrix",
            res["inputs"]["tumour_only_matrix"]["sha256"] == aud["inputs"]["tumour_only_matrix"]["sha256"]
            and res["inputs"]["all_tissue_matrix"]["sha256"] == aud["inputs"]["all_tissue_matrix"]["sha256"])
    s26 = b.csv(S26V2 / "R3.7-T4_S26_S-D1_v2_body_raw.csv", "workspace S26 v2 (final S8) body raw")
    ms = (REPO / "docs/prosper_revised/prosper_manuscript_revised.md")
    b.bind(ms, "revised manuscript (Results subtype sentence)")
    ms_txt = ms.read_text(encoding="utf-8")
    b.check("manuscript Results: 0.847 (0.807\u20130.885; 373 slides) and 0.914 (0.855\u20130.963; 139 slides)",
            "0.847 of RMS queries (95% CI 0.807\u20130.885; 373 slides) and 0.914 of liver tumor queries (0.855\u20130.963; 139 slides)" in ms_txt)
    leg = REPO / "docs/prosper_revised/prosper_supplementary_legends.md"
    b.bind(leg, "supplementary legends (S8/S9 captions; lead's source of truth)")
    leg_txt = leg.read_text(encoding="utf-8")

    new_rows, raw_rows, at_trace = [], [], []
    cls_pairs = {"s61_int_2c": ("ERMS", "ARMS"), "s64": ("HB", "HCC")}
    for arm, (maj, mino) in cls_pairs.items():
        a = res["arms"][arm]["all_tissue"]
        t = res["arms"][arm]["tumour_only"]
        v = v1_full[v1_full.arm == arm].iloc[0]
        rows = atr[(atr.arm == arm) & (atr.representation == "all_tissue")].set_index("class")
        srows = s26[(s26.arm == arm) & (s26.representation == "all_tissue")].set_index("class")
        n = int(v.n_queries)
        b.check(f"{arm}: all-tissue population = v1 population ({n} slides / {v.n_patients} patients / {v.n_reference_slides} reference)",
                a["n_queries"] == n and a["n_patients"] == int(v.n_patients) and a["n_reference_slides"] == int(v.n_reference_slides))
        b.close(f"{arm}: 11-03 tumour-only point = v1 frozen point", t["plurality"], int(v.num_plurality5_correct) / n, 1e-12)
        b.close(f"{arm}: 11-03 all-tissue point = audit all-tissue point (distinct route)", a["plurality"], aud["results"][arm]["all_tissue_matrix"]["plurality"], 1e-12)
        for c in ("ALL", "MACRO", maj, mino):
            for col in ("plurality5_recall", "plurality5_recall_lo", "plurality5_recall_hi"):
                b.close(f"{arm} {c} {col}: s26_rows_all_tissue.csv = S26 v2 body raw (final S8)", rows.loc[c, col], srows.loc[c, col], 0)
        b.close(f"{arm}: results.json all-tissue point = s26_rows ALL", a["plurality"], rows.loc["ALL", "plurality5_recall"], 1e-12)
        b.check(f"{arm}: results.json all-tissue interval = s26_rows ALL", abs(a["plurality_ci"][0] - rows.loc["ALL", "plurality5_recall_lo"]) < 1e-12 and abs(a["plurality_ci"][1] - rows.loc["ALL", "plurality5_recall_hi"]) < 1e-12)
        conf = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r"(\w+) (\d+)/(\d+)", rows.loc["ALL", "confusion"])}
        k_maj = conf[maj][0]
        k_min = conf[mino][1]
        n_maj, n_min = int(rows.loc[maj, "n_queries"]), int(rows.loc[mino, "n_queries"])
        k = k_maj + k_min
        b.check(f"{arm}: all-tissue correct {k}/{n} from the confusion ({maj} {k_maj}/{n_maj}, {mino} {k_min}/{n_min}) = point estimates",
                abs(k / n - a["plurality"]) < 1e-12 and abs(k_maj / n_maj - rows.loc[maj, "plurality5_recall"]) < 1e-12 and abs(k_min / n_min - rows.loc[mino, "plurality5_recall"]) < 1e-12
                and n_maj + n_min == n)
        b.check(f"{arm}: all-tissue plurality@5 rounds to the main-text value {MAIN_VALUE[arm]}", r3(k / n) == f"{MAIN_VALUE[arm]:.3f}")
        mac = rows.loc["MACRO"]
        mn = rows.loc[mino]
        newr = {c: v[c] for c in v1_tab.columns}
        newr.update({
            "plurality5_accuracy": f"{k}/{n} ({r3(k / n)})",
            "plurality5_accuracy_ci": f"{r3(a['plurality_ci'][0])}{DASH}{r3(a['plurality_ci'][1])} ({CI_TXT})",
            "mean_class_recall": r3(mac.plurality5_recall),
            "mean_class_recall_ci": f"{r3(mac.plurality5_recall_lo)}{DASH}{r3(mac.plurality5_recall_hi)} ({CI_TXT})",
            "per_class_plurality5_recall": f"{maj} {k_maj}/{n_maj} ({r3(k_maj / n_maj)}); {mino} {k_min}/{n_min} ({r3(k_min / n_min)})",
            "minority_class_recall": f"{mino} {k_min}/{n_min} ({r3(k_min / n_min)})",
            "minority_class_recall_ci": f"{r3(mn.plurality5_recall_lo)}{DASH}{r3(mn.plurality5_recall_hi)} ({CI_TXT})",
            "uncertainty": "whole-patient bootstrap 95% percentile CI (B=5000, seed 123; all-tissue run 2026-11-03, the intervals of Supplementary Table S8); conditional on the fixed reference library",
            "caveats": v.caveats + "; all-tissue mosaics, the representation used in the main text and Supplementary Table S8",
        })
        b.check(f"{arm}: majority-class rate unchanged ({v.majority_class_rate})", v.majority_class_rate == f"{maj} {n_maj}/{n} ({r3(n_maj / n)})")
        new_rows.append((arm, newr))
        raw_rows.append(dict(arm=arm, representation=REP_ALL_COMET, plurality5_correct=k, n_queries=n, plurality5=a["plurality"], plurality5_lo=a["plurality_ci"][0], plurality5_hi=a["plurality_ci"][1],
                             mean_class_recall=mac.plurality5_recall, mean_class_recall_lo=mac.plurality5_recall_lo, mean_class_recall_hi=mac.plurality5_recall_hi,
                             minority_class=mino, minority_correct=k_min, minority_n=n_min, minority_recall=mn.plurality5_recall, minority_lo=mn.plurality5_recall_lo, minority_hi=mn.plurality5_recall_hi,
                             majority_n=n_maj, coverage_evaluated=int(v.num_coverage_evaluated), coverage_denominator=int(v.num_coverage_denominator), fine_labelled_slides=int(v.num_fine_labelled_slides),
                             value_source="r3_7_all_tissue_ci_20261103/results.json + s26_rows_all_tissue.csv"))
        for fld, val, src in (("plurality5_accuracy", newr["plurality5_accuracy"], f"results.json arms.{arm}.all_tissue.plurality; count from s26_rows_all_tissue.csv confusion"),
                              ("plurality5_accuracy_ci", newr["plurality5_accuracy_ci"], f"results.json arms.{arm}.all_tissue.plurality_ci"),
                              ("mean_class_recall(_ci)", f"{newr['mean_class_recall']}; {newr['mean_class_recall_ci']}", "s26_rows_all_tissue.csv [all_tissue, MACRO]"),
                              ("per_class / minority (_ci)", f"{newr['per_class_plurality5_recall']}; {newr['minority_class_recall_ci']}", f"s26_rows_all_tissue.csv [all_tissue, {maj}/{mino}]"),
                              ("descriptive columns", "copied from the tumour-filtered row", "same population (asserted)")):
            at_trace.append((arm, "all-tissue", fld, val, src))
        # tumour-only rerun intervals (11-03) vs the frozen v1 intervals: shown, not substituted
        s_t = summ[(summ.arm == arm) & (summ["class"] == "ALL")].iloc[0]
        raw_rows.append(dict(arm=arm, representation=REP_TUMOUR, plurality5_correct=int(v.num_plurality5_correct), n_queries=n, plurality5=s_t.plurality5_recall,
                             plurality5_lo=s_t.plurality5_recall_lo, plurality5_hi=s_t.plurality5_recall_hi, rerun_20261103_plurality5_lo=t["plurality_ci"][0], rerun_20261103_plurality5_hi=t["plurality_ci"][1],
                             value_source="r3_7_fine_label_20261003/summary_all_arms.csv (frozen; v1); rerun_20261103_* from r3_7_all_tissue_ci_20261103/results.json tumour_only"))
        dlt = res["arms"][arm]["delta_plurality_all_tissue_minus_tumour_only"]
        raw_rows[-2].update(delta_all_minus_tumour=dlt["point"], delta_lo=dlt["ci"][0], delta_hi=dlt["ci"][1])
    d1, d2 = res["arms"]["s61_int_2c"]["delta_plurality_all_tissue_minus_tumour_only"], res["arms"]["s64"]["delta_plurality_all_tissue_minus_tumour_only"]
    b.check("paired differences (all-tissue \u2212 tumour-only) \u22120.040 [\u22120.065, \u22120.017] and \u22120.022 [\u22120.061, +0.015] (S8 legend)",
            [round(x, 3) for x in (d1["point"], *d1["ci"], d2["point"], *d2["ci"])] == [-0.040, -0.065, -0.017, -0.022, -0.061, 0.015])
    b.check("legend S9 text promises a representation column", "a `representation` column states which" in leg_txt)
    rr = {a: res["arms"][a]["tumour_only"]["plurality_ci"] for a in cls_pairs}
    fz = {a: summ[(summ.arm == a) & (summ["class"] == "ALL")].iloc[0] for a in cls_pairs}
    gap = max(abs(rr[a][i] - fz[a][c]) for a in cls_pairs for i, c in ((0, "plurality5_recall_lo"), (1, "plurality5_recall_hi")))
    b.check("tumour-only intervals: frozen (v1) and 11-03 rerun agree within the audit tolerance 0.006 (they are different bootstrap runs; both retained)", gap <= 0.006, f"max |diff| {gap:.4f}")

    # ---------------- assemble v2 table ----------------
    ROLE = {"s61_int_2c": "sensitivity (tumour-filtered); the main text reports the all-tissue row",
            "s64": "sensitivity (tumour-filtered); the main text reports the all-tissue row",
            "s61_ext_2c": "main-text value (all-tissue)"}
    out = []
    for r in v1_tab.to_dict("records"):
        r2 = {"arm": r["arm"], "representation": rep[r["arm"]],
              "value_in_main_text": ROLE.get(r["arm"], "spindle-cell row: main-text footnote" if r["arm"] in ("s61_int_3c", "s61_ext_3c") else "not in main text")}
        r2.update({c: r[c] for c in v1_tab.columns if c != "arm"})
        r2["value_source"] = "v1 (frozen task inventory 2026-10-04; summary_all_arms.csv)"
        out.append(r2)
        for arm, newr in new_rows:
            if arm == r["arm"]:
                r3_ = {"arm": arm, "representation": REP_ALL_COMET, "value_in_main_text": f"main-text value (all-tissue; {MAIN_VALUE[arm]:.3f})"}
                r3_.update({c: newr[c] for c in v1_tab.columns if c != "arm"})
                r3_["value_source"] = "all-tissue run 2026-11-03 (r3_7_all_tissue_ci_20261103; = Supplementary Table S8)"
                out.append(r3_)
    T = pd.DataFrame(out)
    b.check("v2 inventory: 15 rows (13 v1 arms + all-tissue rows for s61_int_2c and s64)", len(T) == 15 and T.arm.nunique() == 13)
    vv = T[T.value_source.str.startswith("v1")].drop(columns=["representation", "value_in_main_text", "value_source"]).reset_index(drop=True)
    b.check("v2 rows for the 13 v1 arms reproduce v1 cell for cell", vv.equals(v1_tab))
    b.check("v2 main-text values: all-tissue s61_int_2c 0.847, s64 0.914; CCDI s61_ext_2c 0.880",
            set(T[T.value_in_main_text.str.startswith("main-text")].plurality5_accuracy) == {"316/373 (0.847)", "127/139 (0.914)", "221/251 (0.880)"})

    num_cols = [c for c in v1_full.columns if c.startswith("num_")]
    raw = v1_full[["arm"] + num_cols].copy()
    raw.insert(1, "representation", raw.arm.map(rep))
    raw = raw.merge(pd.DataFrame([x for x in raw_rows if x["representation"] == REP_TUMOUR]).drop(columns=["representation", "plurality5_correct", "n_queries"]), on="arm", how="left")
    atraw = pd.DataFrame([x for x in raw_rows if x["representation"] == REP_ALL_COMET])
    prov = pd.DataFrame(trace + at_trace, columns=["arm", "representation", "field", "value", "source"])
    prov["representation_source"] = prov.arm.map({x["arm"]: x["source"] for x in rep_rows})

    # ---------------- audit INFO: all-tissue points for the other internal arms ----------------
    pts = []
    for arm in internal:
        at = aud["results"].get(arm, {}).get("all_tissue_matrix")
        v = v1_full[v1_full.arm == arm].iloc[0]
        n = int(v.n_queries)
        if at is None:
            pts.append(dict(arm=arm, n_queries=n, tumour_filtered_plurality5=int(v.num_plurality5_correct) / n, all_tissue_plurality5=None,
                            status="no all-tissue value in any retained output"))
            continue
        kk = at["plurality"] * n
        b.check(f"{arm}: audit all-tissue plurality@5 is an integer count over {n} queries", abs(kk - round(kk)) < 1e-9)
        pts.append(dict(arm=arm, n_queries=n, tumour_filtered_plurality5=int(v.num_plurality5_correct) / n, all_tissue_plurality5=at["plurality"],
                        all_tissue_correct=int(round(kk)), delta_point=at["delta_plurality"],
                        delta_ci_audit_seed20261021=f"[{at['delta_plurality_ci'][0]:.4f}, {at['delta_plurality_ci'][1]:.4f}]",
                        per_class_all_tissue="; ".join(f"{c} {x:.3f}" for c, x in at["per_class_plurality"].items()),
                        status=("reported: see the all-tissue row (accepted 11-03 intervals)" if arm in cls_pairs else
                                "audit-trace INFO point only (no marginal interval; difference interval from audit seed 20261021, not the accepted seed-123 route); not a reported result")))
    P = pd.DataFrame(pts)
    b.check("audit all-tissue points exist for 8 of 9 internal arms (none for s61_fusion)", P.all_tissue_plurality5.notna().sum() == 8 and P[P.all_tissue_plurality5.isna()].arm.tolist() == ["s61_fusion"])
    P_disp = pd.DataFrame({"Arm": P.arm, "Query slides": P.n_queries,
                           "Plurality@5, tumour-filtered (inventory row)": [f"{x:.3f}" for x in P.tumour_filtered_plurality5],
                           "Plurality@5, all-tissue (point)": ["\u2014" if pd.isna(x) else f"{int(c)}/{n} ({x:.3f})" for x, c, n in zip(P.all_tissue_plurality5, P.get("all_tissue_correct", 0).fillna(0), P.n_queries)],
                           "Difference all-tissue \u2212 tumour-filtered (point)": ["\u2014" if pd.isna(x) else f"{x:+.3f}".replace("-", "\u2212") for x in P.delta_point],
                           "Per-class plurality@5, all-tissue": P.per_class_all_tissue.fillna("\u2014"), "Status": P.status})

    # ---------------- caption ----------------
    title = "Inventory of the evaluated fine-label tasks (v2: representation column)"
    caption = ("All thirteen arms evaluated in the frozen fine-label re-evaluation are listed so that the selection reported in the main text is visible. "
               "A `representation` column states whether each arm was ranked on all-tissue mosaics or on the precomputed tumour-only tile selection "
               "(tile classifier not held out by patient). Internal COMET arms were ranked on the tumour-only tile selection and external CCDI-MCI and KidRare arms on all-tissue mosaics; "
               "for the two internal arms reported in the main text (RMS embryonal versus alveolar; liver HB versus HCC) both representations are shown, and the all-tissue rows "
               "(plurality@5 0.847 and 0.914) carry the values reported in the main text and in Supplementary Table S8.")
    foot = [
        "Conditional retrieval: each query slide has a known broad diagnosis and is ranked only against the known-fine-label sub-library of the same diagnostic family (`n_reference_slides`), not against the whole library; results do not describe whole-library search. Saved depth-10 rank lists were re-scored without re-ranking.",
        "The query patient's slides are excluded from the reference for internal COMET arms; external CCDI-MCI queries have no patient overlap with the COMET reference; for KidRare only the query slide itself is excluded because patient linkage is unknown. KidRare point estimates may be optimistic, and its slide-resampling intervals are not patient-level and are not reported.",
        "k = 5. Plurality@5: most frequent label in the top 5, ties to the tied label at the closest rank. Strict@5: label with at least 3 of the top 5, otherwise abstain (counted as not correct); in two-class arms identical to plurality@5 by construction. Mean class recall: unweighted mean of per-class plurality@5 recalls. Minority class: the class with fewest evaluated query slides.",
        "Intervals: whole-patient percentile bootstrap (5,000 resamples, seed 123), conditional on the fixed reference library. Tumour-filtered rows carry the frozen intervals of the 2026-10-04 inventory; the all-tissue rows carry the intervals of the 2026-11-03 all-tissue run, which are those of Supplementary Table S8 (paired differences in plurality@5, all-tissue minus tumour-filtered, \u22120.040 [\u22120.065, \u22120.017] for RMS and \u22120.022 [\u22120.061, +0.015] for liver tumours). Estimates are slide-level; slides of one patient are not independent.",
        "The tumour-only tile selection used for the internal arms is the precomputed selection of the submitted analysis, whose tile classifier was not held out by patient; it is not the patient-held-out filtering of Fig. 3 (Supplementary Tables S13 and S14).",
        "Fusion, MYCN, INPC, NB-versus-GNB, xenograft-retained and KidRare arms are reported here only; the inventory records where each arm is reported and its caveats and gives no reason for non-reporting. All-tissue point values for the other internal arms exist only as audit-trace information without intervals (sheet 'all_tissue_points_audit') and are not reported results.",
    ]
    sources = ["$PROSPER_OUT/r3_7_task_inventory_20261004b/task_inventory.csv (= v1 supp_tables_20261022/.../R3.7-T5_S28_S-D2_task_inventory_full.csv)",
               "$PROSPER_OUT/r3_7_fine_label_20261003/summary_all_arms.csv (v1 points and intervals)",
               "$PROSPER_OUT/s6_*/result_<arm>.json config.matrix (representation)",
               "$PROSPER_OUT/r3_7_all_tissue_ci_20261103/{results.json, s26_rows_all_tissue.csv} (all-tissue rows; = supp_tables_20261103 S26 v2)",
               "$PROSPER_OUT/r3_7_audit_20261020/results_seed20261021_B5000.json (audit-trace all-tissue points, INFO)"]
    cap = b.caption_frame(title, caption, foot, sources, "v2 (2026-11-09): representation column; all-tissue rows for s61_int_2c and s64; v1 (supp_tables_20261022) retained unchanged")

    disp_cols = ["arm", "representation", "value_in_main_text"] + [c for c in v1_tab.columns if c != "arm"] + ["value_source"]
    sheets = [("inventory", T[disp_cols]), ("inventory_raw_num", raw), ("all_tissue_rows_raw", atraw),
              ("representation_trace", pd.DataFrame(rep_rows)), ("provenance", prov), ("all_tissue_points_audit", P_disp), ("all_tissue_points_audit_raw", P)]
    b.write_workbook("Table_S28_finelabel_inventory_v2", cap, sheets)

    b.open_items += [
        "Tumour-filtered intervals for s61_int_2c and s64: v2 keeps the frozen v1 intervals (0.854\u20130.919; 0.886\u20130.976), whereas the S26 v2 note (edit sheet line \u22482876) and legends line 413 quote the 11-03 rerun intervals 0.855\u20130.920 and 0.884\u20130.977 for the S9 notes. Both are seed-123 B-5000 whole-patient bootstraps that agree within the audit tolerance; the lead should choose one so the S9 rows and the S9/S8 note text match (the rerun values are in sheet inventory_raw_num, rerun_20261103_*).",
        "Legends line 413 states that S9's internal RMS and liver rows carry only the tumour-filtered values; v2 now shows both representations, so that sentence needs updating if v2 is adopted (no edit made under docs/).",
        "All-tissue values for the other internal arms (s61_int_3c, s61_int_2c_xeno, s62, s62_xeno, s65_inpc, s65b_gnb) exist only as audit-trace points without marginal intervals; they are shown in a separate INFO sheet, not as inventory rows. s61_fusion has no all-tissue value in any retained output. Producing intervals would be new computation (not authorized here).",
        "The 2026-10-20 audit trace recorded one FAIL (s62 amplified-class plurality@5 lower bound, frozen 0.224 vs audit 0.231, outside the 0.012 small-group tolerance); the v1 s62 minority-class interval is carried unchanged.",
        "The task specified the representation wording 'all-tissue / tumor-filtered, classifier not held out by patient'; British 'tumour' spelling follows the v1 table; the final SI style may need 'tumor'.",
    ]
    b.finish(title, [
        "15 rows: the 13 v1 arms reproduced cell for cell, each with a representation traced to its result_<arm>.json matrix, plus all-tissue rows for s61_int_2c (316/373, 0.847 [0.807, 0.885]) and s64 (127/139, 0.914 [0.855, 0.963]) flagged as the main-text values.",
        "All-tissue rows equal the 11-03 run, the S26 v2 (final S8) body and the manuscript Results sentence; tumour-only points equal v1; every v1 point/interval traced to summary_all_arms.csv.",
        "v1 files untouched (sha256 equal to the v1 SHA256SUMS).",
    ], cap)


if __name__ == "__main__":
    main()
