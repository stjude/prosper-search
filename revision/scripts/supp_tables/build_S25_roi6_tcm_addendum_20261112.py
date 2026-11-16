#!/usr/bin/env python3
"""Supplementary Table S25 addendum (revision draft, 2026-11-12): cosine heat maps, six ROI-search
configurations and the tumour-correspondence map.

decision 2026-11-11/12: cosine for ROI search and heat maps; Fig. 6A with six configurations; supplementary
tumour-correspondence map; "draft them for my reading". The S25 workbook built by build_S-R3.5-L1_20261109.py
(blocks A-D) is unchanged; assemble_final_tables_20261110.py (handler s25) appends these sheets after D4:

  C3  cosine heat maps on the 44-reference annotation design (with the L1 rows of C1 on the same draws and
      the L1 / cosine tie census)
  C4  per-reference AUROC / pointing / AP, L1 and cosine (internal slide index and tumour group only)
  D5  six search configurations (+ submitted float16 gate row): macro / micro F1 and hit@5, k = 5 and 10
  D6  paired differences against Virchow2 L1 (float32)
  D7  per-group F1, seven configurations x 15 groups x k = 5, 10
  D8  per-group paired differences against Virchow2 L1
  E1  tumour-correspondence map and comparison maps: AUROC by pair set, query arm and reference half
  E2  pre-registered paired contrasts

Assembly / re-tabulation only: every value is read from an accepted, gated result package.
Output: $PROSPER_OUT/supp_tables_20261112_roi6_tcm/.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c_localization_common_20261109 import OUT_ROOT, REPO, ci, deterministic_xlsx, rel, sha256  # noqa: E402

OUT = OUT_ROOT / "supp_tables_20261112_roi6_tcm"
COS = OUT_ROOT / "r3_5_cosine_heatmap_c4_44_20261112"
EVAL = OUT_ROOT / "roi_search_6cfg_20261111" / "eval"
TCM = OUT_ROOT / "r3_5_tumour_correspondence_map_20261111"
FIG6 = OUT_ROOT / "figures_20261021" / "fig6_roi_search" / "source_data_panelA.csv"
TIE_L1 = OUT_ROOT / "localization_tie_readback_20261110" / "results.json"
TIE_COS = OUT_ROOT / "localization_tie_readback_cosine_20261112" / "results.json"
GROUPED = "threshold-grouped AP / tie-fraction pointing (replacement conventions)"
STEM = "Table_S25_addendum_roi6_tcm"

CFG = {  # key -> (encoder, distance, role)
    "submitted": ("Virchow2, first 1,280 dimensions (submitted float16 hits)", "L1", "gate only (superseded)"),
    "v2_l1": ("Virchow2, first 1,280 dimensions", "L1", "paired reference"),
    "v2_cos": ("Virchow2, first 1,280 dimensions", "cosine", "sensitivity"),
    "uni2_l1": ("UNI2", "L1", "sensitivity"),
    "uni2_cos": ("UNI2", "cosine", "primary (pre-specified)"),
    "hopt1_l1": ("H-Optimus-1", "L1", "sensitivity"),
    "hopt1_cos": ("H-Optimus-1", "cosine", "sensitivity"),
}
MAPS = {"tcm": "tumour-correspondence map (reference-held-out tumour classifier on the query)",
        "tcm_own": "tumour-correspondence map, query's own classifier head (sensitivity; not leakage-safe)",
        "roi_cos": "cosine ROI heat map (query feature = mean of up to eight tumour-labelled tiles)",
        "nn1": "nearest query tile, all query tiles (unconstrained)",
        "ot": "transport matched-cost map (exact Optimal Transport, cosine cost)"}
SETS = {"P2_manuscript44": "constructed annotated pairs, the 44 references of block C (other-group arm also reference 13748)",
        "P2_all46": "constructed annotated pairs, all 46 references", "P1_all": "natural top-ten retrieval pairs"}
checks, inputs = [], {}


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    if not ok:
        raise SystemExit(f"CHECK FAILED: {name} {detail}")


def read(p, what, **kw):
    inputs[rel(p)] = {"sha256": sha256(p), "what": what}
    return pd.read_csv(p, **kw) if str(p).endswith((".csv", ".csv.gz")) else json.loads(Path(p).read_text())


# ------------------------------------------------------------------ inputs and gates
cres = read(COS / "result.json", "cosine 44-reference packaging record (gates G1-G4)")
check("C3: cosine packaging gates G1-G4 passed", all(cres["gates"].values()), json.dumps(cres["gates"]))
for f in ("summary.csv", "per_reference.csv", "result.json"):
    rec = dict(l.split()[::-1] for l in (COS / "SHA256SUMS.txt").read_text().splitlines())
    check(f"C3: {f} matches its SHA256SUMS", rec[f] == sha256(COS / f))
csum = read(COS / "summary.csv", "cosine 44-reference estimands").set_index("estimand")
tl1 = read(TIE_L1, "L1 heat-map tie readback (pass 029)")
tcos = read(TIE_COS, "cosine heat-map tie readback (pass 036)")
check("C3: L1 tie readback fidelity passed", tl1["S25"]["fidelity"]["pass"])
check("C3: cosine tie readback gates F1-F3 passed", tcos["passed"] and all(tcos["gates"].values()),
      json.dumps(tcos["gates"]))
check("C3: L1 tied-at-maximum count equal in both readbacks (703)",
      tl1["S25"]["tie_census"]["pairs_with_tie_at_max"] == tcos["tie_census"]["l1"]["pairs_with_tie_at_max"] == 703)
cper = read(COS / "per_reference.csv", "cosine 44-reference per-reference values", dtype={"reference": str})
g1 = read(EVAL / "run_001" / "gates.json", "ROI6 evaluation gates")
check("D: ROI6 run_001 gates all true (G4 re-passed, identical draws)", g1["passed"] and all(g1["gates"].values())
      and g1["primary"] == "uni2_cos", str({k: v for k, v in g1["gates"].items() if not v}))
g2 = read(EVAL / "per_group_001" / "gates.json", "ROI6 per-group gates")
check("D: ROI6 per_group_001 gates passed (G1-G3, identical draws)", g2["passed"])
summ = read(EVAL / "run_001" / "summary.csv", "ROI6 overall summary")
pdif = read(EVAL / "run_001" / "paired_differences.csv", "ROI6 overall paired differences")
pg = read(EVAL / "per_group_001" / "per_group_f1.csv", "ROI6 per-group F1")
pgd = read(EVAL / "per_group_001" / "paired_per_group_vs_v2_l1.csv", "ROI6 per-group paired differences")
tj = read(TCM / "run.json", "TCM run record")
check("E: TCM fidelity pass (cosine ROI map equals the accepted per_pair_A, 1,130 pairs)", tj["fidelity"]["pass"]
      and tj["fidelity"]["pairs_compared"] == 1130)
ts = read(TCM / "summary.csv", "TCM summary")
td = read(TCM / "paired_differences.csv", "TCM paired differences")
f6 = read(FIG6, "Fig. 6A source data (consistency)")

# ------------------------------------------------------------------ block C3 / C4
ROWS = [("AUROC, same-group queries", "auroc_same"), ("AUROC, other-group queries", "auroc_other"),
        ("AUROC, same − other", "auroc_diff"), ("pointing: tumour at highest-scoring annotated location, same-group", "pointing_same"),
        ("pointing, other-group", "pointing_other"), ("pointing, same − other", "pointing_diff"),
        ("average precision, same-group", "ap_same"), ("average precision, other-group", "ap_other"),
        ("average precision, same − other", "ap_diff")]
c3 = []
for lab, k in ROWS:
    for dist in ("l1", "cosine", "cosine_minus_l1"):
        r = csum.loc[f"{dist}_{k}"]
        c3.append({"endpoint": lab, "map": {"l1": "L1 (UNI2)", "cosine": "cosine (UNI2)",
                                            "cosine_minus_l1": "cosine − L1 (paired)"}[dist],
                   "estimate [95% CI]": ci(r.point, r.ci95_lo, r.ci95_hi, plus=dist == "cosine_minus_l1" or k.endswith("diff")),
                   "n references": int(r.n_references)})
r = csum.loc["tumour_share"]
c3.append({"endpoint": "annotated tumour share (chance level for pointing)", "map": "—",
           "estimate [95% CI]": ci(r.point, r.ci95_lo, r.ci95_hi), "n references": 44})
r = csum.loc["cosine_pointing_same_minus_share"]
c3.append({"endpoint": "pointing (same-group) minus tumour share", "map": "cosine (UNI2)",
           "estimate [95% CI]": ci(r.point, r.ci95_lo, r.ci95_hi, plus=True), "n references": 44})
for lab, key in (("references with cosine > L1 same-group AUROC (descriptive count)", "n_refs_cosine_gt_l1_same"),
                 ("references with cosine same-group > other-group AUROC (descriptive count)", "n_refs_cosine_same_gt_other")):
    c3.append({"endpoint": lab, "map": "cosine", "estimate [95% CI]": f"{cres['counts'][key]}/44", "n references": 44})
census = {"L1 (UNI2)": (tl1["S25"]["tie_census"], tl1["S25"]["differences_vs_stored"][GROUPED]),
          "cosine (UNI2)": (tcos["tie_census"]["cosine"], tcos["differences_vs_stored_cosine"][GROUPED])}
for lab, f in (("pairs with the maximum score tied (tie census)", lambda c, _: f"{c['pairs_with_tie_at_max']}/880"),
               ("pairs with tumour and non-tumour tied at the maximum", lambda c, _: f"{c['pairs_with_mixed_label_tie_at_max']}/880"),
               ("largest change of an AP aggregate when tied scores are grouped",
                lambda _, d: f"{max(abs(d[k]) for k in ('ap_same', 'ap_other', 'ap_diff')):.4f}")):
    for m, (c, d) in census.items():
        c3.append({"endpoint": lab, "map": m, "estimate [95% CI]": f(c, d), "n references": 44})
check("C3: no tumour / non-tumour tie at the maximum; grouped AP within the 0.005 rule (both maps)",
      all(c["pairs_with_mixed_label_tie_at_max"] == 0 and max(abs(d[k]) for k in ("ap_same", "ap_other", "ap_diff")) < 0.005
          for c, d in census.values()))
C3 = pd.DataFrame(c3)
check("C3: L1 same-group AUROC = block C1 (0.691 [0.614, 0.764])",
      ci(*csum.loc["l1_auroc_same", ["point", "ci95_lo", "ci95_hi"]]) == "0.691 [0.614, 0.764]")
C4 = cper.rename(columns={"reference": "reference slide (internal index)", "ref_group": "reference tumour group"}).round(3)
check("C4: 88 rows = 44 references × 2 maps", len(cper) == 88 and cper.reference.nunique() == 44)

# ------------------------------------------------------------------ block D5-D8
s = summ[summ.metric.isin(["f1", "hit_at_5"])].copy()
s["cfg"], s["k"] = s.config.str.rsplit("_k", n=1).str[0], s.config.str.rsplit("_k", n=1).str[1].astype(int)
check("D5: 7 configurations × k 5/10 × {F1, hit@5} × {macro, micro}; 2,778 slides, 1,876 patients, B 5,000, 0 non-finite",
      len(s) == 56 and set(s.cfg) == set(CFG) and (s.n_queries == 2778).all() and (s.n_patients == 1876).all()
      and (s.n_replicates == 5000).all() and (s.n_nonfinite == 0).all())
order = {c: i for i, c in enumerate(CFG)}
s = s.sort_values(["k", "metric", "average"], key=lambda c: c.map(order) if c.name == "cfg" else c)
s = s.assign(_o=s.cfg.map(order)).sort_values(["k", "_o", "metric", "average"])
D5 = pd.DataFrame([{"encoder": CFG[r.cfg][0], "tile distance": CFG[r.cfg][1], "role": CFG[r.cfg][2], "hits per ROI (k)": r.k,
                    "metric": {"f1": "F1", "hit_at_5": "hit@5"}[r.metric], "average": r.average,
                    "estimate [95% CI]": ci(r.point, r.ci_lo, r.ci_hi), "slides": int(r.n_queries), "patients": int(r.n_patients),
                    "configuration key": r.cfg} for r in s.itertuples()])
d = pdif[pdif.metric.isin(["f1", "hit_at_5"])].copy()
d["cfg"], d["k"] = d.minuend.str.rsplit("_k", n=1).str[0], d.minuend.str.rsplit("_k", n=1).str[1].astype(int)
check("D6: 5 configurations × k 5/10 × {F1, hit@5} × {macro, micro} against v2_l1 at the same k",
      len(d) == 40 and set(d.subtrahend.str.rsplit("_k", n=1).str[0]) == {"v2_l1"})
d = d.assign(_o=d.cfg.map(order)).sort_values(["k", "_o", "metric", "average"])
D6 = pd.DataFrame([{"contrast": f"{CFG[r.cfg][0]}, {CFG[r.cfg][1]} − Virchow2 first 1,280 dimensions, L1", "role": CFG[r.cfg][2],
                    "hits per ROI (k)": r.k, "metric": {"f1": "F1", "hit_at_5": "hit@5"}[r.metric], "average": r.average,
                    "paired difference [95% CI]": ci(r.point, r.ci_lo, r.ci_hi, plus=True),
                    "interval includes 0 (nominal)": bool(r.ci_includes_zero_nominal)} for r in d.itertuples()])
check("D7: 7 configurations × 15 groups × k 5/10; B 5,000; 0 non-finite", len(pg) == 210 and pg.group.nunique() == 15
      and (pg.n_replicates == 5000).all() and (pg.n_nonfinite == 0).all())
p_ = pg.assign(_o=pg.config.map(order)).sort_values(["k", "_o", "group"])
D7 = pd.DataFrame([{"encoder": CFG[r.config][0], "tile distance": CFG[r.config][1], "role": CFG[r.config][2],
                    "hits per ROI (k)": int(r.k), "tumour group": r.group, "slides": int(r.n_slides),
                    "patients": int(r.n_patients), "F1 [95% CI]": ci(r.f1, r.ci_lo, r.ci_hi)} for r in p_.itertuples()])
check("D8: 5 contrasts × 15 groups × k 5/10 against v2_l1", len(pgd) == 150 and set(pgd.subtrahend) == {"v2_l1"}
      and "submitted" not in set(pgd.minuend))
q_ = pgd.assign(_o=pgd.minuend.map(order)).sort_values(["k", "_o", "group"])
D8 = pd.DataFrame([{"contrast": f"{CFG[r.minuend][0]}, {CFG[r.minuend][1]} − Virchow2 first 1,280 dimensions, L1",
                    "role": CFG[r.minuend][2], "hits per ROI (k)": int(r.k), "tumour group": r.group,
                    "paired difference [95% CI]": ci(r.point, r.ci_lo, r.ci_hi, plus=True)} for r in q_.itertuples()])
dev = 0.0
for r in f6[f6.drawn].itertuples():
    src = (pg[(pg.config == r.config) & (pg.k == 5) & (pg.group == r.category)] if r.kind == "group" else
           summ[(summ.config == f"{r.config}_k5") & (summ.metric == "f1") & (summ.average == r.category.split()[0].lower())]
           .rename(columns={"point": "f1"}))
    dev = max(dev, *(abs(float(src.iloc[0][c]) - getattr(r, c)) for c in ("f1", "ci_lo", "ci_hi")))
check("D5/D7: every drawn Fig. 6A value equals these sources", dev < 1e-12, f"max |Δ| {dev:.1e}")

# ------------------------------------------------------------------ block E
check("E1: 3 sets × 2 arms × 3 halves × 5 maps minus P1 transport map (not computed for natural pairs)",
      len(ts) == 3 * 2 * 3 * 5 - 6 and set(ts["map"]) == set(MAPS) and (ts.B == 5000).all() and (ts.seed == 123).all())
HALF = {"development": "development half", "report": "report half", "pooled": "pooled"}
e1 = ts.assign(_s=ts["set"].map(list(SETS).index), _m=ts["map"].map(list(MAPS).index), _h=ts.half.map(list(HALF).index))
E1 = pd.DataFrame([{"pair set": SETS[r.set], "query arm": r.arm.replace("_", "-"), "reference half": HALF[r.half],
                    "map": MAPS[r.map], "AUROC [95% CI]": ci(r.point, r.ci95_lo, r.ci95_hi), "pairs": int(r.n_pairs),
                    "references": int(r.n_references), "reference patients": int(r.n_patients), "set key": r.set, "map key": r.map}
                   for r in e1.sort_values(["_s", "arm", "_h", "_m"], ascending=[True, False, True, True]).itertuples()])
e2 = td.assign(_s=td["set"].map(list(SETS).index), _h=td.half.map(list(HALF).index))
E2 = pd.DataFrame([{"pair set": SETS[r.set], "query arm": r.arm.replace("_", "-"), "reference half": HALF[r.half],
                    "contrast": f"{r.minuend} − {r.subtrahend}", "paired AUROC difference [95% CI]": ci(r.point, r.ci95_lo, r.ci95_hi, plus=True),
                    "pairs": int(r.n_pairs), "references": int(r.n_references), "set key": r.set}
                   for r in e2.sort_values(["_s", "arm", "_h"], ascending=[True, False, True]).itertuples()])
ref = ts[(ts["set"] == "P2_manuscript44") & (ts.arm == "same_group") & (ts.half == "pooled")].set_index("map")
dE = max(abs(ref.loc["roi_cos", a] - csum.loc["cosine_auroc_same", b])
         for a, b in (("point", "point"), ("ci95_lo", "ci95_lo"), ("ci95_hi", "ci95_hi")))
check("E1: same-group cosine ROI map on the manuscript references equals C3, point and interval (separate code paths)",
      dE < 1e-9, f"max |Δ| {dE:.1e}")
check("E1: TCM same-group manuscript references 0.752 [0.704, 0.797]",
      ci(*ref.loc["tcm", ["point", "ci95_lo", "ci95_hi"]]) == "0.752 [0.704, 0.797]")
note_cs = ("C3 (block-C draws, scripts/r3_5_cosine_heatmap_c4_44.py) and E1 (TCM run, scripts/r3_5_tumour_correspondence_map.py) "
           "compute the same-group cosine ROI map on the manuscript references by separate code; both give the same point and "
           "interval (checked above). The other-group arm of E1 includes reference 13748, which has no same-group arm and is "
           "therefore outside block C.")

# ------------------------------------------------------------------ write
OUT.mkdir(parents=True, exist_ok=True)
sheets = [("C3_cosine_annot44_endpoints", C3, csum.reset_index()), ("C4_cosine_annot44_per_ref", C4, cper),
          ("D5_roi6_overall", D5, s.drop(columns=["_o"])), ("D6_roi6_overall_paired", D6, d.drop(columns=["_o"])),
          ("D7_roi6_per_group", D7, pg), ("D8_roi6_per_group_paired", D8, pgd),
          ("E1_tcm_maps", E1, ts), ("E2_tcm_paired", E2, td)]
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    for name, disp, _ in sheets:
        assert len(name) <= 31, name
        disp.to_excel(xw, sheet_name=name, index=False)
    for ws in xw.book.worksheets:
        for col in ws.columns:
            w = max(len(str(c.value)) if c.value is not None else 0 for c in col[:300])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 60)
        ws.freeze_panes = "A2"
xlsx = OUT / f"{STEM}.xlsx"
xlsx.write_bytes(deterministic_xlsx(buf.getvalue()))
for name, disp, raw in sheets:
    disp.to_csv(OUT / f"{STEM}__{name}.csv", index=False)
    raw.to_csv(OUT / f"{STEM}__{name}_raw.csv", index=False)
(OUT / "input_bindings.json").write_text(json.dumps({"builder": rel(Path(__file__)), "builder_sha256": sha256(Path(__file__)),
                                                      "inputs": inputs}, indent=1, ensure_ascii=False) + "\n")
rep = ["# Supplementary Table S25 addendum (pass 036 draft, 2026-11-12)", "",
       f"Builder `{rel(Path(__file__))}`; workbook `{xlsx.name}` (sheets appended to S25 after D4 by "
       "`assemble_final_tables_20261110.py`, handler `s25`). Draft for Haoran's reading; not approved.", "",
       "## Checks", ""] + [f"- {'PASS' if ok else 'FAIL'} {n}{(' — ' + d_) if d_ else ''}" for n, ok, d_ in checks] + [
       "", "## Inputs", ""] + [f"- `{k}` {v['sha256'][:16]}… — {v['what']}" for k, v in inputs.items()] + [
       "", "## Sheets", ""] + [f"- `{n}`: {disp.shape[0]} rows × {disp.shape[1]} columns (raw `{STEM}__{n}_raw.csv`)"
                               for n, disp, _ in sheets] + ["", "## Note", "", note_cs, ""]
(OUT / "BUILD_REPORT.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
with open(OUT / "SHA256SUMS.txt", "w") as fh:
    for p in sorted(OUT.iterdir()):
        if p.name != "SHA256SUMS.txt":
            fh.write(f"{sha256(p)}  {p.name}\n")
print(f"{len(checks)} checks pass; wrote {xlsx}")
