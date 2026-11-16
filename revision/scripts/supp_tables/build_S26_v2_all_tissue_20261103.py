#!/usr/bin/env python3
"""Supplementary Table S26 (S-D1) version 2 — all-tissue COMET arms as the reported rows, tumour-only rows as a
sensitivity block (2026-11-03; design decision: a manuscript comment on the Results subtype paragraph and
"proceed with what you recommended on all items").

Assembly only, no statistic computed here. Sources:
  * all-tissue and tumour-only COMET rows (s61_int_2c, s64): $PROSPER_OUT/r3_7_all_tissue_ci_20261103/s26_rows_all_tissue.csv
    (seed 123, B 5000, whole-patient; analyst distinct-route check PASS, SHA256SUMS.txt there);
  * CCDI-MCI arm (s61_ext_2c) and the spindle footnote: unchanged from version 1
    ($PROSPER_OUT/supp_tables_20261022/A_ladder_failure_finelabel/R3.7-T4_S26_S-D1_body_raw.csv, _footnote_spindle.csv).
Checks: the tumour-only rows of the new run reproduce version 1 (points to 1e-9 after rounding to 3 d.p.; interval bounds
within 0.006, or 0.015 for the liver arm whose HCC group has 25 patients — different bootstrap route, same seed/B); the all-tissue points equal the 10-20 audit INFO values.
Writes $PROSPER_OUT/supp_tables_20261103/A_ladder_failure_finelabel/R3.7-T4_S26_S-D1_v2{.xlsx,_body_raw.csv,_body_formatted.csv,_footnote_spindle.csv}
and BUILD_REPORT.md. Version 1 is left untouched."""
from __future__ import annotations
import datetime as dt, hashlib, json, math, os
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(os.environ.get("PROSPER_OUT") or REPO / "prosper_revision_output")
V1 = OUT_ROOT / "supp_tables_20261022/A_ladder_failure_finelabel"
NEW = OUT_ROOT / "r3_7_all_tissue_ci_20261103"
OUT = OUT_ROOT / "supp_tables_20261103/A_ladder_failure_finelabel"; OUT.mkdir(parents=True, exist_ok=True)
MINUS = "−"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f3(x): return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.3f}"
def ci3(p, lo, hi): return f"{f3(p)} [{f3(lo)}, {f3(hi)}]"
checks = []
def check(name, ok, detail=""): checks.append((name, bool(ok), detail)); assert ok, (name, detail)

new = pd.read_csv(NEW / "s26_rows_all_tissue.csv"); v1 = pd.read_csv(V1 / "R3.7-T4_S26_S-D1_body_raw.csv"); fn = pd.read_csv(V1 / "R3.7-T4_S26_S-D1_footnote_spindle.csv")
res = json.load(open(NEW / "results.json"))
aud = json.load(open(OUT_ROOT / "r3_7_audit_20261020/results_seed20261021_B5000.json"))["results"]
check("new run: seed 123, B 5000", res["seed"] == 123 and res["B"] == 5000)
check("new run: all 36 internal checks PASS", all(c["status"] == "PASS" for c in res["checks"]), str(sum(c["status"] != "PASS" for c in res["checks"])))
cols = ["plurality5_recall", "plurality5_recall_lo", "plurality5_recall_hi", "hit1", "hit1_lo", "hit1_hi", "precision5", "precision5_lo", "precision5_hi", "auroc_top5_pos_share", "auroc_lo", "auroc_hi"]
for arm in ("s61_int_2c", "s64"):
    for cl in new[new.arm == arm]["class"].unique():
        a = new[(new.arm == arm) & (new.representation == "tumour_only") & (new["class"] == cl)].iloc[0]
        b = v1[(v1.arm == arm) & (v1["class"] == cl)].iloc[0]
        for c in cols:
            if pd.isna(b[c]) and pd.isna(a[c]): continue
            tol = 1e-9 if not c.endswith(("_lo", "_hi")) else (0.015 if arm == "s64" else 0.006)  # audit tolerances: 0.006; 0.015 where a group has < 25 patients (HCC) and in the macro/ALL rows that include it
            check(f"{arm}/{cl}/{c}: tumour-only reproduces version 1 (tol {tol})", abs(float(a[c]) - float(b[c])) <= tol, f"{a[c]} vs {b[c]}")
    at = new[(new.arm == arm) & (new.representation == "all_tissue") & (new["class"] == "ALL")].iloc[0]
    check(f"{arm}: all-tissue plurality point = 10-20 audit INFO", abs(at.plurality5_recall - aud[arm]["all_tissue_matrix"]["plurality"]) < 1e-9)
    check(f"{arm}: all-tissue hit@1 point = 10-20 audit INFO", abs(at.hit1 - aud[arm]["all_tissue_matrix"]["hit1"]) < 1e-9)

labels = {"s61_int_2c": ("ERMS vs ARMS, COMET", ["ERMS", "ARMS"]), "s61_ext_2c": ("ERMS vs ARMS, CCDI-MCI → COMET", ["ERMS", "ARMS"]), "s64": ("HB vs HCC, COMET", ["HB", "HCC"])}
def fmt_rows(df, arm, label, cls, block):
    rows = []
    for j, (cl, rl) in enumerate([("ALL", "all"), ("MACRO", "macro")] + [(c, c) for c in cls]):
        r = df[df["class"] == cl].iloc[0]
        conf = r.get("confusion", r.get("confusion_k5_plurality", "")) if cl == "ALL" else ""
        rows.append({"Block": block if j == 0 else "", "Comparison": label if j == 0 else "", "Row": rl,
                     "n slides / patients": "—" if cl == "MACRO" else f"{int(r.n_queries)} / {int(r.n_patients)}",
                     "Plurality@5 [95% CI]": ci3(r.plurality5_recall, r.plurality5_recall_lo, r.plurality5_recall_hi),
                     "hit@1 [95% CI]": ci3(r.hit1, r.hit1_lo, r.hit1_hi), "P@5 [95% CI]": ci3(r.precision5, r.precision5_lo, r.precision5_hi),
                     "AUROC [95% CI]": ci3(r.auroc_top5_pos_share, r.auroc_lo, r.auroc_hi) if cl == "ALL" else "—",
                     "Confusion (rows = truth)": str(conf).replace(";", "; ") if cl == "ALL" else ""})
    return rows
body, raw = [], []
# reported rows: all-tissue COMET arms + CCDI arm (unchanged)
for arm, blk in (("s61_int_2c", "All-tissue mosaics (reported)"), ("s61_ext_2c", "All-tissue mosaics (reported)"), ("s64", "All-tissue mosaics (reported)")):
    lab, cls = labels[arm]
    if arm == "s61_ext_2c":
        df = v1[v1.arm == arm].copy(); df["representation"] = "all_tissue"; df["confusion"] = df["confusion_k5_plurality"]
    else:
        df = new[(new.arm == arm) & (new.representation == "all_tissue")]
    body += fmt_rows(df, arm, lab, cls, blk); raw.append(df.assign(block="reported"))
for arm in ("s61_int_2c", "s64"):
    lab, cls = labels[arm]; df = new[(new.arm == arm) & (new.representation == "tumour_only")]
    body += fmt_rows(df, arm, lab + " (tumour-only tile selection, Supplementary Table S5)", cls, "Sensitivity: tumour-only representations"); raw.append(df.assign(block="sensitivity"))
X = pd.DataFrame(body)
R = pd.concat(raw, ignore_index=True)[["block", "arm", "representation", "class", "unit", "n_queries", "n_patients", "n_reference_slides", "n_reference_patients"] + cols + ["confusion"]]
dR = {arm: res["arms"][arm]["delta_plurality_all_tissue_minus_tumour_only"] for arm in ("s61_int_2c", "s64")}
def sgn(x): return (MINUS if x < 0 else "+") + f"{abs(x):.3f}"
legend = ("Supplementary Table S26. Subtype-level retrieval within a known tumor family (UNI2, Optimal Transport, all-tissue mosaics). "
          "Each query was searched only against subtype-labelled slides of its family, excluding slides from the query patient. External CCDI-MCI queries were searched against the COMET RMS library (373 slides, 306 patients). "
          "Plurality@5: the most frequent subtype among the five nearest slides matches the recorded subtype. hit@1: the nearest slide has the query's subtype. P@5: share of the five nearest slides with the query's subtype. "
          "AUROC: discrimination of the alveolar RMS or hepatocellular carcinoma share among the five nearest slides. Macro: unweighted mean of the two subtypes. Point estimates are over slides; 95% percentile intervals resample whole patients (5,000 resamples, seed 123) with the library fixed. "
          "Majority-class shares: 0.606 (COMET RMS), 0.677 (CCDI-MCI RMS), 0.683 (liver). "
          f"Sensitivity block: the COMET arms repeated on the precomputed tumour-only tile selection underlying Supplementary Table S5 (not the patient-held-out classifiers of Fig. 3); paired differences in plurality@5 (all-tissue minus tumour-only, same bootstrap draws) {sgn(dR['s61_int_2c']['point'])} [{sgn(dR['s61_int_2c']['ci'][0])}, {sgn(dR['s61_int_2c']['ci'][1])}] for RMS and {sgn(dR['s64']['point'])} [{sgn(dR['s64']['ci'][0])}, {sgn(dR['s64']['ci'][1])}] for liver tumors. "
          "Percentile intervals for small groups (e.g., HCC, 25 patients) are approximate and may under-cover. "
          "Footnote: spindle-cell RMS (COMET 18 slides/14 patients; CCDI-MCI 20 slides) was evaluated only in a three-subtype search on the tumour-only representations (ties among three subtypes were resolved to the closest-ranked tied subtype; none occurred for the spindle-cell queries). "
          "All 18 COMET and 17 of 20 CCDI-MCI spindle-cell queries were retrieved as embryonal RMS (plurality@5 0 and 0.05). These sparse estimates are not interpreted further.")
cap = pd.DataFrame([("field", "text"), ("table_id (edit-sheet block)", "R3.7-T4"), ("version", "S-D1 v2 (2026-11-03): all-tissue COMET rows reported, tumour-only rows as sensitivity; replaces the v1 option-(a) legend sentence"),
                    ("author decision", "Haoran 2026-11-03, comment C574 on the Results subtype paragraph; 'proceed with what you recommended on all items'"),
                    ("legend (APPLIED, v2)", legend),
                    ("sources", f"{NEW.name}/s26_rows_all_tissue.csv ({sha(NEW / 's26_rows_all_tissue.csv')[:16]}…); v1 body_raw ({sha(V1 / 'R3.7-T4_S26_S-D1_body_raw.csv')[:16]}…; CCDI arm, footnote)"),
                    ("checks", f"{len(checks)} PASS, 0 FAIL (see BUILD_REPORT.md)")], columns=["field", "text"]).iloc[1:]
base = OUT / "R3.7-T4_S26_S-D1_v2"
X.to_csv(f"{base}_body_formatted.csv", index=False); R.to_csv(f"{base}_body_raw.csv", index=False); fn.to_csv(f"{base}_footnote_spindle.csv", index=False)
with pd.ExcelWriter(f"{base}.xlsx", engine="openpyxl") as xw:
    for name, df in (("caption", cap), ("S26 body", X), ("footnote spindle", fn)):
        df.to_excel(xw, sheet_name=name, index=False)
    for ws in xw.book.worksheets:
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 80 if ws.title != "caption" else 160)
rep = [f"# BUILD_REPORT — Supplementary Table S26 (S-D1) v2 — {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", "", "Script: `scripts/supp_tables/build_S26_v2_all_tissue_20261103.py`", "",
       "## Checks", ""] + [f"- {'PASS' if ok else 'FAIL'} {n}" + (f" ({d})" if d and not ok else "") for n, ok, d in checks] + ["", "## Outputs", ""] + \
      [f"- `{p.name}` sha256 {sha(p)}" for p in sorted(OUT.glob('R3.7-T4_S26_S-D1_v2*'))] + ["", "## Legend (v2)", "", legend, ""]
(OUT / "BUILD_REPORT.md").write_text("\n".join(rep), encoding="utf-8")
print(X.to_string()); print(f"\nchecks PASS {sum(ok for _, ok, _ in checks)} / {len(checks)}"); print(legend)
