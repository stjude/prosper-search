#!/usr/bin/env python3
"""Table S-R3.5-L2 (current final number Supplementary Table S27): Optimal Transport distance
decomposition by pathologist annotation class.

Spec: docs/manuscript_supplement_edits.md, '**R3.5-T7 — Supplementary Table S-R3.5-L2 (new):**' (rows =
the endpoints in [internal record] section 3 for pooled, development half and report half with
estimate, 95% CI and n references; plus per-class retention and the probe confusion matrix; sources
l2/run_003/report_estimates.json, matrices_D1.csv, confusion_D3a.csv), with the secondary analyses of the
current S27 notes (composition-adjusted alignment, bounded variant with headroom >= 0.02; embedding probe;
forced cross-class transport).

Sheets:
  A   endpoints ([internal record] section 3) by population
  B1  class-normalised-cost AUROC by reference class      B2  class cost shares and class-normalised costs
  C1  composition-adjusted alignment (all variants)       C2  pair counts (scored, no headroom, NA > 1)
  D   per-class retention (observed, chance, ceiling; tile-count and restricted-mass marginals)
  E1  class-transport matrices over pairs                 E2  class-transport matrices, per-reference means
  F1  embedding probe readings and controls               F2  probe confusion matrices
  G1  forced cross-class transport and nearest-neighbour matching   G2  forced-mismatch distribution over pairs

Assembly/re-tabulation only. Writes $PROSPER_OUT/supp_tables_20261109/C_localization/S-R3.5-L2/.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c_localization_common_20261109 import (  # noqa: E402
    BUILD_DATE, BUILT_BY, ES, GROUP_DIR, LG, MS, OUT_ROOT, REPO, Build, ci, fmt, numbering, rel,
    report_header, report_tail, sha256)

WID = "S-R3.5-L2"
TITLE = "Optimal Transport distance decomposition by pathologist annotation class"
LOC = OUT_ROOT / "r3_5_localisation_maps_20261017"
L2 = LOC / "l2/run_003"
CON = LOC / "contract_004"
SCRIPT_SHA = "32fc331de61a0ea0933111df29633de90a03ca57347d016edeb2b806d0e3b5ff"
POPS = ["pooled", "development", "report"]
POPLAB = {"pooled": "pooled (46 reference patients)", "development": "development half (consistency check)",
          "report": "report half (consistency check)"}
RAWF = ["point", "ci95_lo", "ci95_hi", "n_references", "n_patients", "n_replicates_finite"]
CL = {"c0": "unannotated (label 0)", "c1": "tumour (1)", "c2": "immune (2)", "c3": "necrosis (3)",
      "c4": "non-tumour (4)", "c5": "artefact (5)", "nt": "pooled non-tumour (2–5)", "D": "full distance (all rows)"}
ARM = {"same_group": "same-group", "other_group": "other-group", "all": "all pairs"}
GRAN = {"bin": "binary (tumour / pooled non-tumour)", "5": "5-class"}
PLAN = {"full": "full exact OT plan", "mt1": "mutual top-1 sparsified plan (sensitivity)"}
SUBSET = {"all": "all pairs", "share_ge_0.20": "annotated-mass share ≥ 0.20 (sensitivity)"}

b = Build(WID, GROUP_DIR / WID, __file__)
final = numbering(b, WID)
b.check("numbering map: S-R3.5-L2 → S27", final == "S27", final)

# ------------------------------------------------------------------ inputs
comp = b.json(L2 / "COMPLETE.json", "L2 run_003 completion record (output hashes; its script field names the L1 helper, see PROVENANCE_NOTE)")
rep = b.json(L2 / "report_estimates.json", "L2 estimates: pooled, development and report populations; probe readings")
chk = b.json(L2 / "checks.json", "L2 fidelity checks F1-F6")
bind = b.json(L2 / "bindings.json", "L2 input bindings (script, plans, slides)")
probe = b.json(L2 / "probe_D3a.json", "embedding probe readings (D3a)")
sd3 = b.json(L2 / "summary_D3.json", "forced mismatch distribution, NN matching, switch readings")
conf = b.csv(L2 / "confusion_D3a.csv", "probe confusion counts", dtype={"granularity": str})
mat = b.csv(L2 / "matrices_D1.csv", "class-transport matrices averaged over pairs", dtype={"granularity": str})
matr = b.csv(L2 / "matrices_D1_per_reference.csv", "class-transport matrices averaged within then across references",
             dtype={"granularity": str})
mixed = b.csv(L2 / "probe_D3a_mixed_slides.csv", "probe per-slide readings (counted only; patient codes not copied)",
              dtype={"granularity": str})
for f in ("report_estimates.json", "checks.json", "bindings.json", "probe_D3a.json", "summary_D3.json", "confusion_D3a.csv",
          "matrices_D1.csv", "matrices_D1_per_reference.csv", "probe_D3a_mixed_slides.csv"):
    b.producer_hash(L2 / f, comp["outputs"][f], "l2/run_003/COMPLETE.json")
impl = b.text(L2 / "IMPLEMENTATION_CHOICES.md", "implementer choices (headroom rule, marginals, probe)")
b.producer_hash(L2 / "IMPLEMENTATION_CHOICES.md", comp["outputs"]["IMPLEMENTATION_CHOICES.md"], "l2/run_003/COMPLETE.json")
prov = b.text(L2 / "PROVENANCE_NOTE.md", "provenance note (true producing-script sha)")
b.inp(L2 / "IMPLEMENTER_REPORT.md", "L2 implementer report")
contract = b.text(CON / "CONTRACT.md", "L2 analysis contract 004 (endpoint definitions, headroom rule)")
amend = b.text(CON / "AMENDMENT_001.md", "contract amendment 001 (endpoint readings)")
lr2 = b.text(LOC / "LEAD_REVIEW_002.md", "lead review 002 (verdict, distinct-route recount, section 3 endpoints)")
snap = b.inp(OUT_ROOT / "script_snapshots/r3_5_l2_alignment_v3_28b781b6.py", "producing-script snapshot (v3)")
b.inp(REPO / "scripts/r3_5_l2_alignment.py", "producing script in the repository (read, not run)")

# ------------------------------------------------------------------ provenance checks
shas = {"report_estimates.json": rep["script_sha256"], "checks.json": chk["script_sha256"],
        "bindings.json": bind["script_sha256"], "probe_D3a.json": probe["script_sha256"],
        "summary_D3.json": sd3["script_sha256"]}
b.check("producing-script sha 28b781b6… recorded in every L2 result JSON", all(v == SCRIPT_SHA for v in shas.values()),
        ", ".join(f"{k} {v[:8]}" for k, v in shas.items()))
b.check("script snapshot and repository script hash to 28b781b6…",
        sha256(snap) == SCRIPT_SHA == sha256(REPO / "scripts/r3_5_l2_alignment.py"))
b.check("PROVENANCE_NOTE records the L1-helper script field in COMPLETE.json and the true L2 sha",
        SCRIPT_SHA in prov and comp["script_sha256"][:8] in prov, f"COMPLETE.script_sha256 {comp['script_sha256'][:8]}…")
b.check("COMPLETE.json contract_sha256 equals CONTRACT.md", comp["contract_sha256"] == sha256(CON / "CONTRACT.md"))
b.check("AMENDMENT_001.md hash equals the final text named in LEAD_REVIEW_002 §1.1 (b122c805…)",
        sha256(CON / "AMENDMENT_001.md").startswith("b122c805") and "b122c805" in lr2)
b.check("checks.json all_pass and COMPLETE.checks_pass", chk["all_pass"] is True and comp["checks_pass"] is True,
        ", ".join(f"{k} {chk[k]['pass']}" for k in ("F1", "F2", "F3", "F4", "F5", "F6")))
b.check("LEAD_REVIEW_002 verdict is ACCEPT for run_003", "**Verdict: ACC" in lr2)
b.check("probe_D3a.json readings identical to report_estimates.json probe_D3a",
        all(probe[k] == rep["probe_D3a"][k] for k in rep["probe_D3a"]), ", ".join(rep["probe_D3a"]))

P, DV, RP = (rep[p] for p in POPS)


def g(pop, k):
    return rep[pop].get(k)


def cell(v, plus=False, dp=3):
    if not isinstance(v, dict) or v.get("point") is None:
        return "n.e.", None
    return ci(v["point"], v.get("ci95_lo"), v.get("ci95_hi"), dp=dp, plus=plus), v.get("n_references")


def raw_long(keys):
    rows = []
    for pop in POPS:
        for k in keys:
            v = g(pop, k)
            rows.append(dict(population=pop, key=k, **{f: (v.get(f) if isinstance(v, dict) else None) for f in RAWF}))
    return pd.DataFrame(rows)


def wide(rows, plus_fn=lambda k: False):
    """rows: list of (dict of descriptive columns, key). One display row per key across the three populations."""
    out = []
    for desc, k in rows:
        r = dict(desc)
        for pop in POPS:
            e, n = cell(g(pop, k), plus_fn(k))
            r[f"{pop}: estimate [95% CI]"] = e
            r[f"{pop}: n references"] = n
        r["source key"] = k
        out.append(r)
    return intn(pd.DataFrame(out))


def intn(df):
    for c in df.columns:
        if c.endswith("n references"):
            df[c] = df[c].astype("Int64")
    return df


is_contrast = lambda k: "minus" in k  # noqa: E731

# ------------------------------------------------------------------ A endpoints ([internal record] section 3)
ENDP = [
    ("D2|disc_auroc_A", "primary", "Discriminative contribution map (mean other-group − mean same-group matched cost): AUROC, annotated tumour vs non-tumour"),
    ("D2|disc_ap_A", "primary (component)", "Discriminative map: average precision"),
    ("D2|disc_pointing_A", "primary (component)", "Discriminative map: pointing (highest-scoring annotated location is tumour)"),
    ("D2|disc_tumour_fraction_A", "primary (component)", "Annotated tumour share (pointing baseline)"),
    ("D2|yard_auroc_A", "comparator", "Same-group-only map (−mean same-group matched cost): AUROC"),
    ("D2|disc_minus_yard_auroc_A", "paired contrast", "Discriminative − same-group-only map AUROC"),
    ("D2|auroc_tumour_minus_nt", "primary", "Same- vs other-group separation by class-normalised cost: tumour − pooled non-tumour AUROC (paired; references with both classes)"),
    ("D2|auroc_c1", "primary (component)", "Separation AUROC, tumour-tile class-normalised cost"),
    ("D2|auroc_nt", "primary (component)", "Separation AUROC, pooled non-tumour class-normalised cost"),
    ("D2|auroc_D", "context", "Separation AUROC, full OT distance"),
    ("D1|full|bin|all|same_group|NA_massmarg", "secondary (reported)", "Composition-adjusted alignment, same-group, binary, full plan: (S − S0)/(S* − S0), restricted-mass marginals (bounded by 1), headroom ≥ 0.02"),
    ("D1|full|bin|all|same_group|kappa", "secondary", "Chance-corrected agreement kappa (S − S0)/(1 − S0), same-group, binary, full plan"),
    ("D1|full|bin|all|same_group|NA", "pre-named reading, not quoted", "Composition-adjusted alignment with tile-count S0/S* (can exceed 1; LEAD_REVIEW_002 §1.2)"),
    ("D1|full|bin|all|other_group|NA", "reported beside", "Same, other-group (tile-count S0/S*)"),
    ("D1|full|bin|all|same_group|annotated_mass_share", "context", "Share of same-group transported mass on links annotated at both endpoints"),
    ("D3|same_group|diff_bin_NA_NN_minus_OT", "no claim", "Nearest-neighbour − OT alignment (same-group, binary, tile-count scale)"),
    ("D3|same_group|full_bin_forced_mismatch", "reported as limit", "Forced cross-class transport 1 − S* (same-group, binary, tile-count marginals)"),
]
PROBE = rep["probe_D3a"]["observed|bin"]["mixed_slide_balanced_accuracy"]
A = wide([({"endpoint": lab, "role": role}, k) for k, role, lab in ENDP], is_contrast)
prow = {"endpoint": "Embedding probe: binary per-slide balanced accuracy on mixed-class slides (leave-one-patient-out, "
                    "ten-neighbour cosine); chance 0.5", "role": "diagnostic",
        "pooled: estimate [95% CI]": cell(PROBE)[0], "pooled: n references": PROBE["n_slides"],
        "development: estimate [95% CI]": "not computed (pooled only)", "development: n references": None,
        "report: estimate [95% CI]": "not computed (pooled only)", "report: n references": None,
        "source key": "probe_D3a['observed|bin'].mixed_slide_balanced_accuracy"}
recs = A.to_dict("records")
A = intn(pd.DataFrame(recs[:15] + [prow] + recs[15:]))
Ar = pd.concat([raw_long([k for k, _, _ in ENDP]),
                pd.DataFrame([dict(population="pooled", key=prow["source key"], **{f: PROBE.get(f) for f in RAWF},
                                   n_slides=PROBE["n_slides"], chance=PROBE["chance_per_slide"])])], ignore_index=True)
for pop in ("development", "report"):
    b.check(f"{pop}: every section-3 endpoint present", all(k in rep[pop] for k, _, _ in ENDP))
nd, nr, npool = (g(p, "D2|disc_auroc_A")["n_references"] for p in ("development", "report", "pooled"))
b.check("discriminative-map references: development + report = pooled", nd + nr == npool, f"{nd} + {nr} = {npool}")

# ------------------------------------------------------------------ B class costs
B1 = wide([({"reference class": CL[c], "quantity": "same- vs other-group separation AUROC of class-normalised cost"
             if c != "D" else "same- vs other-group separation AUROC of the full distance"}, f"D2|auroc_{c}")
           for c in ("c1", "c2", "c3", "c4", "c5", "nt", "c0", "D")]
          + [({"reference class": "tumour − pooled non-tumour", "quantity": "paired AUROC difference"},
              "D2|auroc_tumour_minus_nt")], is_contrast)
B1r = raw_long([f"D2|auroc_{c}" for c in ("c1", "c2", "c3", "c4", "c5", "nt", "c0", "D")] + ["D2|auroc_tumour_minus_nt"])
b2keys = []
for arm in ("same_group", "other_group"):
    for side, sl in (("ref", "reference tiles"), ("qry", "query tiles")):
        for stat, stl in (("s", "share of transport cost D_c / D"), ("k", "class-normalised cost D_c / mass_c")):
            for c in ("c1", "c2", "c3", "c4", "c5", "nt", "c0"):
                b2keys.append(({"arm": ARM[arm], "side": sl, "quantity": stl, "class": CL[c]}, f"D2|{arm}|{side}_{stat}_{c}"))
    b2keys.append(({"arm": ARM[arm], "side": "pair", "quantity": "mean full OT distance D", "class": CL["D"]}, f"D2|{arm}|D"))
B2 = wide(b2keys)
B2r = raw_long([k for _, k in b2keys])
b.check("all D2 per-arm class keys tabulated", len(b2keys) == sum(1 for k in P if re.match(r"D2\|(same|other)_group\|", k)),
        f"{len(b2keys)} keys")

# ------------------------------------------------------------------ C alignment
STATS = [("S", "same-class mass S (restricted to links annotated at both ends; normalised to 1)"),
         ("S0", "independence agreement S0, tile-count marginals"),
         ("Sstar", "maximum agreement S*, tile-count marginals"),
         ("headroom", "headroom S* − S0, tile-count marginals"),
         ("NA", "alignment (S − S0)/(S* − S0), tile-count marginals — can exceed 1; not quoted"),
         ("scored", "share of pairs scored (headroom ≥ 0.02), tile-count marginals"),
         ("S0_massmarg", "independence agreement S0, restricted-mass marginals"),
         ("Sstar_massmarg", "maximum agreement S*, restricted-mass marginals"),
         ("headroom_massmarg", "headroom S* − S0, restricted-mass marginals"),
         ("NA_massmarg", "alignment (S − S0)/(S* − S0), restricted-mass marginals — bounded; reported"),
         ("scored_massmarg", "share of pairs scored (headroom ≥ 0.02), restricted-mass marginals"),
         ("kappa", "kappa (S − S0)/(1 − S0), tile-count marginals"),
         ("kappa_massmarg", "kappa, restricted-mass marginals"),
         ("kappa_L1conv", "kappa, L1 chance convention"),
         ("chance_count", "chance agreement from annotated tile-count shares (L1 convention)"),
         ("annotated_mass_share", "share of transported mass on links annotated at both endpoints")]
ckeys, cntrows = [], []
for plan in PLAN:
    for gr in GRAN:
        for sub in SUBSET:
            for arm in ARM:
                base = f"D1|{plan}|{gr}|{sub}|{arm}"
                for st, stl in STATS:
                    if f"{base}|{st}" not in P:
                        continue
                    ckeys.append(({"plan": PLAN[plan], "granularity": GRAN[gr], "pairs": SUBSET[sub], "arm": ARM[arm],
                                   "statistic": stl}, f"{base}|{st}"))
                for pop in POPS:
                    cntrows.append(dict(population=pop, plan=PLAN[plan], granularity=GRAN[gr], pairs=SUBSET[sub],
                                        arm=ARM[arm], **g(pop, f"{base}|counts"), source_key=f"{base}|counts"))
C1 = wide(ckeys)
C1r = raw_long([k for _, k in ckeys])
C2 = pd.DataFrame(cntrows)
d1_nonR = {k for k in P if k.startswith("D1|") and not k.split("|")[-1].startswith("R") and not k.endswith("|counts")}
b.check("C1 covers every non-retention D1 statistic", d1_nonR == {k for _, k in ckeys}, f"{len(d1_nonR)} keys")
cnt = {a: P[f"D1|full|bin|all|{a}|counts"] for a in ARM}
b.check("pair counts: all = same-group + other-group = 910", cnt["all"]["n_pairs"] == 910 ==
        cnt["same_group"]["n_pairs"] + cnt["other_group"]["n_pairs"],
        f"{cnt['same_group']['n_pairs']} + {cnt['other_group']['n_pairs']}")
b.check("headroom rule ≥ 0.02 stated in contract and implementation choices",
        "S* − S0 ≥ 0.02" in contract and "S* - S0 >= 0.02" in impl)
b.check("pooled population is all 46 references (46 patients)",
        rep["populations"]["pooled"] == "all 46 P2 references (46 patients)"
        and max(v["n_references"] for v in P.values() if isinstance(v, dict) and "n_references" in v) == 46)

# ------------------------------------------------------------------ D retention
dkeys = []
for plan in PLAN:
    for gr in GRAN:
        for sub in SUBSET:
            for arm in ARM:
                base = f"D1|{plan}|{gr}|{sub}|{arm}"
                for k in sorted(k for k in P if k.startswith(base + "|R")):
                    m = re.match(r"^R(chance|ceiling)?(_massmarg)?_(.+)$", k.split("|")[-1])
                    dkeys.append(({"plan": PLAN[plan], "granularity": GRAN[gr], "pairs": SUBSET[sub], "arm": ARM[arm],
                                   "class": m.group(3),
                                   "marginals": "restricted-mass" if m.group(2) else "tile-count",
                                   "quantity": {None: "observed retention (class mass staying in class)",
                                                "chance": "chance (independence)", "ceiling": "ceiling (composition)"}[m.group(1)]},
                                  k))
dkeys.sort(key=lambda x: (list(PLAN).index(x[1].split("|")[1]), list(GRAN).index(x[1].split("|")[2]),
                          list(SUBSET).index(x[1].split("|")[3]), list(ARM).index(x[1].split("|")[4]),
                          x[0]["class"], x[0]["marginals"], x[0]["quantity"]))
D = wide(dkeys)
Dr = raw_long([k for _, k in dkeys])
b.check("retention rows: every D1 R* statistic tabulated",
        len(dkeys) == sum(1 for k in P if k.startswith("D1|") and k.split("|")[-1].startswith("R")), f"{len(dkeys)} rows")


# ------------------------------------------------------------------ E matrices
def piv(m, ncol):
    m = m.copy()
    m["value"] = m["value"].map(lambda x: fmt(x, 3))
    w = m.pivot_table(index=["granularity", "arm", "kind", "ref_class"], columns="query_class", values="value",
                      aggfunc="first", sort=False).reset_index()
    n = m.groupby(["granularity", "arm", "kind", "ref_class"], sort=False)[ncol].first().reset_index()
    w = w.merge(n, on=["granularity", "arm", "kind", "ref_class"])
    w.columns.name = None
    return w.rename(columns={"ref_class": "reference class (row)"})


E1 = piv(mat, "n_pairs")
E2 = piv(matr, "n_references")
b.check("matrices: each (granularity, arm, kind, reference class) row appears once",
        not mat.duplicated(["granularity", "arm", "kind", "ref_class", "query_class"]).any()
        and not matr.duplicated(["granularity", "arm", "kind", "ref_class", "query_class"]).any())

# ------------------------------------------------------------------ F probe
prows = []
for lk, v in rep["probe_D3a"].items():
    lab, gr = lk.split("|")
    for rd in ("mixed_slide_balanced_accuracy", "balanced_accuracy"):
        x = v[rd]
        prows.append(dict(labels=lab, granularity=GRAN[gr], reading=rd.replace("_", " "),
                          estimate=cell(x)[0], n_slides=x.get("n_slides"),
                          chance=x.get("chance_per_slide", v.get("chance")), n_true=None,
                          note=x.get("reading", "per-slide over mixed-class slides, averaged per patient"),
                          point=x["point"], ci95_lo=x.get("ci95_lo"), ci95_hi=x.get("ci95_hi")))
    for c, x in v["per_class_recall"].items():
        prows.append(dict(labels=lab, granularity=GRAN[gr], reading=f"recall: {c}", estimate=cell(x)[0], n_slides=None,
                          chance=None, n_true=x.get("n_true"), note="pooled rows", point=x["point"],
                          ci95_lo=x.get("ci95_lo"), ci95_hi=x.get("ci95_hi")))
Fr = pd.DataFrame(prows)
F1 = Fr.drop(columns=["point", "ci95_lo", "ci95_hi"])
mo = mixed[(mixed["labels"] == "observed") & (mixed["granularity"] == "bin")]
b.check("probe mixed-slide reading: 36 slides from 36 patients (counted in probe_D3a_mixed_slides.csv)",
        len(mo) == PROBE["n_slides"] == 36 and mo["patient"].nunique() == PROBE["n_patients"] == 36,
        f"{len(mo)} slides, {mo['patient'].nunique()} patients")
parts = []
for gr in ("5", "bin"):
    w = conf[conf["granularity"] == gr].pivot_table(index=["labels", "granularity", "true"], columns="pred", values="count",
                                                     aggfunc="sum", fill_value=0, sort=False).reset_index()
    w.columns.name = None
    pc = [c for c in w.columns if c not in ("labels", "granularity", "true")]
    w["row total"] = w[pc].sum(axis=1)
    w["recall (re-tabulated)"] = [fmt(r[r["true"]] / r["row total"], 3) if r["true"] in pc and r["row total"] else ""
                                  for _, r in w.iterrows()]
    parts.append(w)
F2 = pd.concat(parts, ignore_index=True)
predcols = [c for c in F2.columns if c not in ("labels", "granularity", "true", "row total", "recall (re-tabulated)")]
F2[predcols] = F2[predcols].astype("Int64")
F2 = F2[["labels", "granularity", "true"] + predcols + ["row total", "recall (re-tabulated)"]]
b.check("confusion matrices: 3 label sets × 2 granularities, row totals equal across label sets",
        F2.groupby(["granularity", "true"])["row total"].nunique().eq(1).all()
        and F2[["labels", "granularity"]].drop_duplicates().shape[0] == 6)
bad = []
for (lab, gr), v in [(k.split("|"), x) for k, x in rep["probe_D3a"].items()]:
    for c, x in v["per_class_recall"].items():
        row = F2[(F2["labels"] == lab) & (F2["granularity"] == gr) & (F2["true"] == c)]
        if len(row) != 1 or int(row["row total"].iloc[0]) != x["n_true"] or \
                abs(row[c].iloc[0] / row["row total"].iloc[0] - x["point"]) > 1e-12:
            bad.append(f"{lab}|{gr}|{c}")
b.check("confusion counts reproduce every per-class recall and n_true in probe_D3a (re-tabulation)", not bad, ", ".join(bad))
b.check("summary_D3 switch reading equals the probe primary reading",
        sd3["switch"]["D3_switch|diff_bin|same_group (primary)"]["probe_bin_mixed_slide_balanced_accuracy"] == PROBE)

# ------------------------------------------------------------------ G forced mismatch / NN
METH = {"nn": "cosine nearest-neighbour matching", "diff": "nearest-neighbour − OT", "full": "full exact OT plan"}
gkeys = []
for arm in ARM:
    for k in [k for k in P if k.startswith(f"D3|{arm}|")]:
        m = re.match(r"^(nn|diff|full)_(bin|5)_(.+)$", k.split("|")[-1])
        gkeys.append(({"arm": ARM[arm], "method": METH[m.group(1)], "granularity": GRAN[m.group(2)],
                       "statistic": m.group(3).replace("_", " ")}, k))
G1 = wide(gkeys, is_contrast)
G1r = raw_long([k for _, k in gkeys])
grows = []
for k, v in sd3.items():
    if not k.startswith("D3c|"):
        continue
    _, arm, gr = k.split("|")
    for var, q in (("tile-count marginals", "forced_mismatch_quantiles"), ("restricted-mass marginals",
                                                                         "forced_mismatch_massmarg_quantiles")):
        grows.append(dict(arm=ARM[arm], granularity=GRAN[gr], marginals=var, n_pairs=v["n_pairs"],
                          **{f"q{float(p):g}": fmt(x, 3) for p, x in v[q].items()},
                          mean_over_pairs=fmt(v["forced_mismatch_mean_pairs"], 3) if var.startswith("tile") else "",
                          per_reference_estimate=cell(v["per_reference_estimate_pooled"])[0] if var.startswith("tile") else "",
                          source_key=f"summary_D3['{k}'].{q}"))
G2 = pd.DataFrame(grows)
G2r = pd.json_normalize([dict(key=k, **v) for k, v in sd3.items() if k.startswith("D3")], sep=".")
b.check("summary_D3 per-reference forced mismatch equals report_estimates (same-group binary)",
        sd3["D3c|same_group|bin"]["per_reference_estimate_pooled"] == P["D3|same_group|full_bin_forced_mismatch"])
b.check("summary_D3 NN − OT difference equals report_estimates (same-group binary)",
        sd3["D3b|same_group|bin"]["diff_NA_NN_minus_OT"] == P["D3|same_group|diff_bin_NA_NN_minus_OT"])

# ------------------------------------------------------------------ value assertions
RE = L2 / "report_estimates.json"
LR = LOC / "LEAD_REVIEW_002.md"
v = b.value
c1, cnt_s = P["D1|full|bin|all|all|counts"], P["D1|full|bin|all|same_group|counts"]
v("constructed slide pairs", "910", RE, "pooled['D1|full|bin|all|all|counts'].n_pairs", c1["n_pairs"], exact=True,
  doc=MS, needle="over 910 constructed slide pairs")
v("constructed pairs (S27 notes)", "910", RE, "pooled['D1|full|bin|all|all|counts'].n_pairs", c1["n_pairs"], exact=True,
  doc=LG, needle="(910 pairs)")
v("reference slides / patients", "46", RE, "populations.pooled; max n_references",
  max(x["n_references"] for x in P.values() if isinstance(x, dict) and "n_references" in x), exact=True,
  doc=LG, needle="46 reference slides from 46 patients")
k = "D2|disc_auroc_A"
v("discriminative (difference) map AUROC", "0.75", RE, f"pooled['{k}'].point", P[k]["point"], 2, MS, "AUROC 0.75 (95% CI 0.67–0.81; 45 references")
v("difference map lower", "0.67", RE, f"pooled['{k}'].ci95_lo", P[k]["ci95_lo"], 2, MS, "AUROC 0.75 (95% CI 0.67–0.81; 45 references")
v("difference map upper", "0.81", RE, f"pooled['{k}'].ci95_hi", P[k]["ci95_hi"], 2, MS, "AUROC 0.75 (95% CI 0.67–0.81; 45 references")
v("map-scored references", "45", RE, f"pooled['{k}'].n_references", P[k]["n_references"], exact=True, doc=MS,
  needle="45 references; Supplementary Table S27")
v("map-scored references (notes: 'Forty-five')", "45", RE, f"pooled['{k}'].n_references", P[k]["n_references"], exact=True,
  doc=LG, needle="Forty-five references supported map scoring and 35")
v("difference map, development half", "0.753", RE, f"development['{k}'].point", DV[k]["point"], 3, LR, "dev 0.753, rep 0.738")
v("difference map, report half", "0.738", RE, f"report['{k}'].point", RP[k]["point"], 3, LR, "dev 0.753, rep 0.738")
k = "D2|disc_minus_yard_auroc_A"
v("difference map − same-group-only map", "−0.03", RE, f"pooled['{k}'].point", P[k]["point"], 2, LG, "paired difference −0.03, −0.09 to 0.02")
v("… lower", "−0.09", RE, f"pooled['{k}'].ci95_lo", P[k]["ci95_lo"], 2, LG, "paired difference −0.03, −0.09 to 0.02")
v("… upper", "0.02", RE, f"pooled['{k}'].ci95_hi", P[k]["ci95_hi"], 2, LG, "paired difference −0.03, −0.09 to 0.02")
k = "D2|yard_auroc_A"
v("same-group-only (yardstick) map AUROC", "0.779", RE, f"pooled['{k}'].point", P[k]["point"], 3, LR, "yardstick 0.779 [0.717, 0.834]")
k = "D2|auroc_tumour_minus_nt"
for lab, pr, f in (("tumour − non-tumour cost AUROC", "0.19", "point"), ("… lower", "0.13", "ci95_lo"), ("… upper", "0.26", "ci95_hi")):
    v(lab, pr, RE, f"pooled['{k}'].{f}", P[k][f], 2, LG, "paired AUROC difference 0.19, 95% CI 0.13–0.26; 35 references")
v("references with both classes", "35", RE, f"pooled['{k}'].n_references", P[k]["n_references"], exact=True, doc=LG,
  needle="35 references with both classes")
for lab, pr, f in (("tumour-tile cost AUROC", "0.82", "point"), ("… lower", "0.77", "ci95_lo"), ("… upper", "0.87", "ci95_hi")):
    v(lab, pr, RE, f"pooled['D2|auroc_c1'].{f}", P["D2|auroc_c1"][f], 2, ES, "AUROC 0.82 (95 % CI 0.77–0.87)")
for lab, pr, f in (("non-tumour-tile cost AUROC", "0.62", "point"), ("… lower", "0.56", "ci95_lo"), ("… upper", "0.69", "ci95_hi")):
    v(lab, pr, RE, f"pooled['D2|auroc_nt'].{f}", P["D2|auroc_nt"][f], 2, ES, "non-tumour tiles reached 0.62 (0.56–0.69)")
v("slides with tumour class", "41", RE, "pooled['D2|auroc_c1'].n_references", P["D2|auroc_c1"]["n_references"], exact=True,
  doc=ES, needle="means over the 41 and 36 slides")
v("slides with non-tumour class", "36", RE, "pooled['D2|auroc_nt'].n_references", P["D2|auroc_nt"]["n_references"], exact=True,
  doc=ES, needle="means over the 41 and 36 slides")
v("full-distance separation AUROC", "0.783", RE, "pooled['D2|auroc_D'].point", P["D2|auroc_D"]["point"], 3, LR, "full distance 0.783")
k = "D1|full|bin|all|same_group|NA_massmarg"
for lab, pr, f in (("composition-adjusted alignment (bounded)", "0.49", "point"), ("… lower", "0.37", "ci95_lo"),
                   ("… upper", "0.61", "ci95_hi")):
    v(lab, pr, RE, f"pooled['{k}'].{f}", P[k][f], 2, ES, "normalised alignment 0.49, 0.37–0.61")
k = "D1|full|bin|all|same_group|kappa"
for lab, pr, f in (("kappa", "0.116", "point"), ("kappa lower", "0.035", "ci95_lo"), ("kappa upper", "0.196", "ci95_hi")):
    v(lab, pr, RE, f"pooled['{k}'].{f}", P[k][f], 3, LR, "kappa 0.116 [0.035, 0.196]")
v("tile-count NA > 1 (pairs)", "48", RE, "pooled['D1|full|bin|all|same_group|counts'].n_pairs_NA_gt_1",
  cnt_s["n_pairs_NA_gt_1"], exact=True, doc=LR, needle="it exceeds 1 on 48 of the 114")
v("scored same-group binary pairs", "114", RE, "pooled['D1|full|bin|all|same_group|counts'].n_pairs_scored",
  cnt_s["n_pairs_scored"], exact=True, doc=LR, needle="it exceeds 1 on 48 of the 114")
k = "D1|full|bin|all|same_group|annotated_mass_share"
v("same-group mass on doubly annotated links ('about 11%')", "11", RE, f"pooled['{k}'].point ×100", 100 * P[k]["point"], 0,
  LG, "about 11% of same-group transported mass")
k = "D3|same_group|full_bin_forced_mismatch"
v("forced cross-class transport ('about 30 %')", "30", RE, f"pooled['{k}'].point ×100", 100 * P[k]["point"], 0, ES,
  "about 30 % of mass is necessarily exchanged")
for lab, pr, f in (("forced mismatch 1 − S*", "0.299", "point"), ("… lower", "0.239", "ci95_lo"), ("… upper", "0.358", "ci95_hi")):
    v(lab, pr, RE, f"pooled['{k}'].{f}", P[k][f], 3, LR, "0.299 [0.239, 0.358]")
for lab, pr, f in (("probe balanced accuracy", "0.71", "point"), ("… lower", "0.66", "ci95_lo"), ("… upper", "0.76", "ci95_hi")):
    v(lab, pr, RE, f"probe_D3a['observed|bin'].mixed_slide_balanced_accuracy.{f}", PROBE[f], 2, ES,
      "balanced accuracy 0.71, 0.66–0.76")
v("probe slides/patients", "36", RE, "probe_D3a['observed|bin'].mixed_slide_balanced_accuracy.n_slides", PROBE["n_slides"],
  exact=True, doc=LR, needle="0.709 [0.656, 0.764] (36 slides/patients)")
v("probe non-tumour recall", "0.579", RE, "probe_D3a['observed|bin'].per_class_recall.nontumour_pooled.point",
  rep["probe_D3a"]["observed|bin"]["per_class_recall"]["nontumour_pooled"]["point"], 3, LR, "non-tumour recall 0.579")
p5 = rep["probe_D3a"]["observed|5"]["mixed_slide_balanced_accuracy"]
v("probe 5-class balanced accuracy", "0.675", RE, "probe_D3a['observed|5'].mixed_slide_balanced_accuracy.point", p5["point"], 3,
  LR, "5-class 0.675 vs chance 0.487")
v("probe 5-class chance", "0.487", RE, "probe_D3a['observed|5'].mixed_slide_balanced_accuracy.chance_per_slide",
  p5["chance_per_slide"], 3, LR, "5-class 0.675 vs chance 0.487")
k = "D3|same_group|diff_bin_NA_NN_minus_OT"
for lab, pr, f in (("NN − OT alignment", "+0.72", "point"), ("… lower", "−0.18", "ci95_lo"), ("… upper", "+1.66", "ci95_hi")):
    v(lab, pr, RE, f"pooled['{k}'].{f}", P[k][f], 2, LR, "+0.72 [−0.18, +1.66]")
sg = "D1|full|bin|all|same_group|"
v("tumour retention (tile-count marginals)", "0.956", RE, f"pooled['{sg}R_tumour'].point", P[sg + "R_tumour"]["point"], 3, LR,
  "tumour mass stays tumour 0.956 (chance 0.915)")
v("tumour retention chance (tile-count marginals)", "0.915", RE, f"pooled['{sg}Rchance_tumour'].point",
  P[sg + "Rchance_tumour"]["point"], 3, LR, "tumour mass stays tumour 0.956 (chance 0.915)")
v("non-tumour retention (restricted-mass marginals)", "0.120", RE, f"pooled['{sg}R_massmarg_nontumour_pooled'].point",
  P[sg + "R_massmarg_nontumour_pooled"]["point"], 3, LR, "non-tumour stays non-tumour 0.120 (chance 0.066)")
v("non-tumour retention chance (restricted-mass marginals)", "0.066", RE,
  f"pooled['{sg}Rchance_massmarg_nontumour_pooled'].point", P[sg + "Rchance_massmarg_nontumour_pooled"]["point"], 3, LR,
  "non-tumour stays non-tumour 0.120 (chance 0.066)")
v("headroom threshold", "0.02", CON / "CONTRACT.md", "§ normalised alignment definition", "0.02", exact=True, doc=LG,
  needle="required headroom of at least 0.02")

# ------------------------------------------------------------------ caption and workbook
mm = P[sg + "R_massmarg_tumour"]
cap_text = (
    "Class decomposition of the saved exact Optimal Transport plans between UNI2 mosaics for 46 annotated reference slides "
    "(46 patients), each compared with ten other-group and up to ten same-group annotated query slides (910 constructed pairs; "
    "not exclusively retrieved neighbours). Sheet A lists the pre-registered endpoints of the analysis for all references pooled "
    "and, as consistency checks, for the development and report halves of the earlier parameter study (Supplementary Table S26). "
    "Sheets B1–B2 give same- versus other-group separation by the class-normalised transport cost of each reference tissue class; "
    "C1–C2 composition-adjusted class alignment, with pair counts; D per-class retention against chance and composition "
    "ceilings; E1–E2 class-to-class transport matrices; F1–F2 the tile-embedding probe and its confusion matrices; G1–G2 forced "
    "cross-class transport and nearest-neighbour matching. Estimates are means over references (pair statistics averaged within "
    "reference first) with nominal percentile 95% intervals from 5,000 whole-reference-patient resamples (PCG64 seed 123), "
    "paired for contrasts, unadjusted for multiplicity and conditional on the fixed query draws, features and annotations. "
    "These evaluations measure overlap with annotated tissue, not diagnostic accuracy or pathologist-assessed usefulness.")
footnotes = "\n".join([
    "(a) Annotation classes (pathologist tile labels): 1 tumour, 2 immune, 3 necrosis, 4 non-tumour, 5 artefact, 0 unannotated. "
    "Binary analyses pool classes 2–5 as non-tumour; unannotated rows remain in the full distance and are reported as class 0, "
    "not scored as tissue. A class was scored for a reference when at least 20 of its mosaic rows carried that class; immune "
    "had no such reference. n.e. = not estimable (no reference met the class rule in that population or variant); blank "
    "matrix cells are undefined for that kind (e.g. ceilings off the diagonal).",
    "(b) Discriminative map: mean other-group minus mean same-group matched cost at each reference tile, projected to patch "
    "locations; the same-group-only map (negative mean same-group cost) is the paired comparison. Forty-five references "
    "supported map scoring; 35 had both tumour and non-tumour classes for the paired class contrast (41 tumour, 36 non-tumour).",
    "(c) Alignment restricts transport to links annotated at both endpoints, with that mass normalised to one. S is same-class "
    "mass, S0 the independence agreement and S* the maximum agreement allowed by the marginals; (S − S0)/(S* − S0) is defined "
    "only when S* − S0 ≥ 0.02 (pairs below are counted as 'no headroom', sheet C2). The reported bounded variant uses "
    "restricted-mass marginals throughout. The pre-named tile-count variant (S0, S* from annotated tile counts) is retained "
    f"as computed but not quoted: it exceeded 1 on {cnt_s['n_pairs_NA_gt_1']} of {cnt_s['n_pairs_scored']} scored same-group "
    "binary pairs (LEAD_REVIEW_002 §1.2).",
    "(d) Retention is the share of a class's restricted transported mass that stays within that class, shown with the "
    "independence (chance) value and the composition ceiling, for both marginal conventions.",
    f"(e) Probe: leave-one-patient-out ten-neighbour cosine classification of annotated tiles ({probe['n_rows']:,} rows from "
    f"{probe['n_slides']} slides, {probe['n_patients']} patients); the primary reading is per-slide balanced accuracy over "
    "mixed-class slides, averaged per patient (chance 0.5 binary; per-slide 1/K for 5-class). Shuffle and global-permutation "
    "rows are negative controls. Confusion counts are pooled over rows. The probe was computed on the pooled references only.",
    "(f) Forced cross-class transport 1 − S*: because exact transport with uniform marginals moves all mass, at least this "
    "share must cross classes when compositions differ. NN rows replace the OT plan by cosine nearest-neighbour matching.",
    "(g) Development/report halves are those of the earlier reference-patient split (seed 123); they are consistency checks, "
    "not a new held-out test.",
])
sources = "; ".join(f"{k} (sha256 {v_['sha256'][:16]}…)" for k, v_ in sorted(b.inputs.items()) if not k.startswith("docs/"))
caption = [
    ("workspace_id", WID),
    ("final_number", f"Supplementary Table {final} (scripts/figures/supp_table_numbering.json; applied at assembly)"),
    ("title", TITLE),
    ("caption", cap_text),
    ("footnotes", footnotes),
    ("sheets", "A endpoints; B1 class-cost AUROC; B2 class cost shares and class-normalised costs; C1 composition-adjusted "
               "alignment; C2 pair counts; D per-class retention; E1 matrices over pairs; E2 matrices per reference; F1 probe "
               "readings; F2 probe confusion; G1 forced mismatch and NN matching; G2 forced-mismatch distribution. Unrounded "
               "values: <stem>__<sheet>_raw.csv."),
    ("caption_status", "DRAFT by the builder from the edit-sheet spec (R3.5-T7) and the current S27 notes; the lead's "
                       "Supplementary Information Markdown is the source of truth for caption wording."),
    ("sources", sources),
    ("built_by/date", f"{BUILT_BY}; {BUILD_DATE}; builder {rel(b.script)}"),
]
sheets = [("A_endpoints", A, Ar), ("B1_class_cost_auroc", B1, B1r), ("B2_class_shares_costs", B2, B2r),
          ("C1_alignment", C1, C1r), ("C2_alignment_pair_counts", C2, None), ("D_retention", D, Dr),
          ("E1_matrices_over_pairs", E1, mat), ("E2_matrices_per_reference", E2, matr), ("F1_probe", F1, Fr),
          ("F2_probe_confusion", F2, conf), ("G1_forced_mismatch_NN", G1, G1r), ("G2_forced_mismatch_dist", G2, G2r)]
xlsx = b.write_workbook(f"Table_{WID}", caption, sheets)
b.check("workbook written with caption first and 12 data sheets", xlsx.exists())

# ------------------------------------------------------------------ notes, open items, report
b.note("Report-estimate keys are copied unchanged into each sheet's 'source key' column; all estimates come from "
       "l2/run_003/report_estimates.json except the forced-mismatch distribution (summary_D3.json), matrices and confusion counts.")
b.note("The quoted alignment changed from the pre-named tile-count NA (1.16 [0.70, 1.77]) to the bounded restricted-mass "
       "variant (0.49 [0.37, 0.61]) after results were seen; LEAD_REVIEW_002 §1.2 records this as a presentation choice with "
       "the same wording class. Both appear in sheets A and C1 with their roles.")
b.note("probe_D3a_mixed_slides.csv is bound and counted (36 slides, 36 patients) but not copied: it carries patient codes.")
b.note("Re-tabulations by the builder (counting only): confusion-matrix row totals and recalls (checked against probe_D3a), "
       "mixed-slide counts.")
b.open_item("Caption and footnotes are builder drafts; the legends file reads 'Supplementary Table S27 | Optimal Transport "
            "distance decomposition by pathologist annotation class. [PENDING: caption]' and was being edited during the build "
            f"(bound version sha256 {sha256(LG)[:16]}…). Final wording comes from the lead's Supplementary Information Markdown "
            "and Haoran's reading approval.")
b.open_item("LEAD_REVIEW_002 §3 labels the retention line 'same-group, mass marginals', but its tumour figures 0.956 "
            f"(chance 0.915) are the tile-count R_tumour/Rchance_tumour; the restricted-mass tumour retention is "
            f"{fmt(mm['point'])} (chance {fmt(P[sg + 'Rchance_massmarg_tumour']['point'])}). Its non-tumour 0.120 (chance 0.066) "
            "is the restricted-mass value. Sheet D shows both conventions; if any text quotes retention, the lead should pick one "
            "convention.")
b.open_item("The embedding probe was computed on pooled references only; development/report cells say 'not computed'.")
b.open_item("Supplementary Fig. S-R3.5-L2 (overlays 4288/4298, binary same-group matrix) is a figure product and is not in this "
            "workbook; the edit sheet's legend note on the clipped colour-bar label of panel C applies to the figure.")
b.open_item("E1/E2 and F2 are aggregate counts/means with no patient identifiers; per-pair and per-tile files "
            "(per_pair_D*.csv.gz, per_tile_cost_D2.csv.gz) were not copied. Whether aggregate restricted-annotation matrices are "
            "published needs the author's access decision.")
b.write_bindings()
vc = b.write_value_checks()
spec = [f"`docs/manuscript_supplement_edits.md` (sha256 {sha256(ES)[:16]}…): '**R3.5-T7 — Supplementary Table S-R3.5-L2 (new):**' "
        f"({b.locate(ES, '**R3.5-T7 — Supplementary Table S-R3.5-L2 (new):**')}) — rows = LEAD_REVIEW_002 §3 endpoints (pooled, "
        "development, report; estimate, 95% CI, n references), per-class retention, probe confusion matrix.",
        f"Candidate main-text block of the 2026-10-18 R3.5-L2 amendment ({b.locate(ES, 'normalised alignment 0.49, 0.37–0.61')}) "
        "for the quoted values.",
        f"S27 notes in `docs/prosper_revised/prosper_supplementary_legends.md` (sha256 {sha256(LG)[:16]}…; "
        f"{b.locate(LG, 'Notes to Supplementary Table S27.')}): secondary sheets for composition-adjusted alignment (bounded "
        "variant, headroom ≥ 0.02), embedding probe and forced cross-class transport.",
        f"Manuscript citation: `docs/prosper_revised/prosper_manuscript_revised.md` (sha256 {sha256(MS)[:16]}…; "
        f"{b.locate(MS, 'Supplementary Table S27')})."]
lines = report_header(b, TITLE, final, spec) + [
    "", "## Workbook", "", f"`{xlsx.name}` — sheet `caption` (field | text) then 12 data sheets; each data sheet is also written as "
    f"`Table_{WID}__<sheet>.csv`, with unrounded source values in `Table_{WID}__<sheet>_raw.csv` (C2 is integer counts and has "
    "no raw counterpart)."] + report_tail(b)
(b.out / "BUILD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
b.write_sums()
n_fail_c = sum(not ok for _, ok, _ in b.checks)
print(f"{WID}: checks {len(b.checks) - n_fail_c}/{len(b.checks)} pass; values {int((vc.result == 'pass').sum())}/{len(vc)} pass")
sys.exit(1 if n_fail_c or (vc.result != "pass").any() else 0)
