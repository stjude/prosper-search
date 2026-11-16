"""Fidelity readback: compare analyze_reader80.py --mode round2-readback against round 2's frozen join outputs.

Reference (read-only): $PROSPER_OUT/s5c_pilot_round2_20261110/analysis/join/{endpoints.json, analysis_work.json,
flag_classification.csv, rating_join_r1.csv} and the rater-1 / automatic rows transcribed in
analysis/ENDPOINTS_REPORT_R2.md. Writes the comparison to --out (Markdown) and exits 1 on any mismatch.
Rater-2, agreement and round-2-example-rule rows are out of scope (single rater; plan 001 §4 S5).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
ap = argparse.ArgumentParser()
ap.add_argument("--ref", type=Path, required=True, help="round-2 analysis/join directory")
ap.add_argument("--report", type=Path, required=True, help="round-2 ENDPOINTS_REPORT_R2.md")
ap.add_argument("--port", type=Path, required=True, help="readback output directory of analyze_reader80.py")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ref_e = json.load(open(a.ref / "endpoints.json"))["endpoints"]
port_e = json.load(open(a.port / "endpoints.json"))["endpoints"]
ref_w = json.load(open(a.ref / "analysis_work.json"))
port_w = json.load(open(a.port / "analysis_work.json"))
rows, bad = [], 0


def cmp(label, r, p):
    global bad
    ok = json.dumps(r, sort_keys=True, default=str) == json.dumps(p, sort_keys=True, default=str)
    bad += not ok
    short = lambda x: (json.dumps(x, default=str) if not isinstance(x, str) else x)[:110]
    rows.append((label, "match" if ok else "**DIFF**", short(r), short(p)))


# 1. endpoint rows: auto / analyst / R1 rows of the round-2 table, matched by (rater, name)
keep = lambda e: e["rater"] in ("R1",) or e["rater"].startswith("auto") or e["rater"] == "analyst adjudication"
ref_rows = [e for e in ref_e if keep(e)]
port_by = {(e["rater"], e["name"]): e for e in port_e}
for e in ref_rows:
    p = port_by.get((e["rater"], e["name"]))
    for fld in ("numerator", "denominator", "value", "notes"):
        cmp(f"endpoint [{e['rater']}] {e['name'][:60]} — {fld}", e.get(fld), p.get(fld) if p else "<missing>")
extra = [k for k in port_by if k not in {(e["rater"], e["name"]) for e in ref_rows}]
cmp("port endpoint rows not in reference", [], extra)

# 2. analysis_work components
for k_ref, k_port in [(("dist", "R1"), ("dist",)), (("dist", "R1_first_draft"), ("dist_first_draft",)),
                      (("rated", "R1|as_rated"), ("rated", "as_rated")),
                      (("rated", "R1|strict_discrepant"), ("rated", "strict_discrepant")),
                      (("rated", "R1|alias_accepted_posthoc"), ("rated", "alias_accepted_posthoc")),
                      (("rated", "R1|alternative_wording_strict"), ("rated", "alternative_wording_strict")),
                      (("auto_end",), ("auto_end",)), (("tg_abs",), ("tg_abs",)), (("zero",), ("zero",)),
                      (("zero_lists",), ("zero_lists",)), (("flagged_after",), ("flagged_after",)),
                      (("prob_acc",), ("prob_acc",)), (("prob_detail",), ("prob_detail",)),
                      (("probtag", "R1"), ("probtag",)), (("harness_reproduction_diffs",), ("harness_reproduction_diffs",)),
                      (("indep_bad",), ("indep_bad",))]:
    r, p = ref_w, port_w
    for k in k_ref:
        r = r[k]
    for k in k_port:
        p = p[k]
    if k_ref == ("auto_end",):
        p = {k: v for k, v in p.items() if k in r}
    cmp("analysis_work " + "/".join(k_ref), r, p)

# spec_rows: reference carries R1 and R2 columns; compare the R1-coded rows' identity and group lists
ref_spec = [{k: v for k, v in s.items() if k != "R2"} for s in ref_w["spec_rows"] if s.get("R1") in
            ("unsupported", "speculative-diagnostic")]
cmp("analysis_work spec_rows (R1-coded U/SD)", ref_spec, port_w["spec_rows"])

# 3. flag classification and rating join, byte-level after identical CSV serialisation
for name in ("flag_classification.csv", "rating_join_r1.csv"):
    r = pd.read_csv(a.ref / name, dtype=str, keep_default_na=False)
    p = pd.read_csv(a.port / name, dtype=str, keep_default_na=False)
    cmp(f"{name} (rows x columns, all cells)", r.to_dict("records"), p[list(r.columns)].to_dict("records")
        if set(r.columns) <= set(p.columns) else list(p.columns))

# 4. the automatic and rater-1 cells of ENDPOINTS_REPORT_R2.md §2-§3, re-rendered from the port's outputs
rep = a.report.read_text().splitlines()


def cell(label_prefix, col=1):
    hit = [l for l in rep if l.startswith("| " + label_prefix)]
    assert len(hit) == 1, (label_prefix, len(hit))
    return [c.strip() for c in hit[0].strip().strip("|").split("|")][col]


ae, rt, rs = port_w["auto_end"], port_w["rated"]["as_rated"], port_w["rated"]["strict_discrepant"]
al, aw = port_w["rated"]["abbreviation_expansion_accepted"], port_w["rated"]["alternative_wording_strict"]
z, pa, tg = port_w["zero"], port_w["prob_acc"], port_w["tg_abs"]
n, d, f = ae["n"], port_w["dist"], port_w["dist_first_draft"]
R1_PREFIX = lambda c: c.split(" / ")[0]
for label, rendered, ref_cell in [
    ("rejected / too long", f"{ae['non_ok_state']}/{n}", cell("rejected / too long")),
    ("style revisions applied", f"{ae['revision_applied']} ({', '.join(ae['revised_queries'])})", cell("style revisions")),
    ("≥ 1 invalid or fabricated citation", f"{max(ae['bad_citation_harness'], ae['bad_citation_independent'], ae['fabricated_ids_any_source'])}/{n}",
     cell("≥ 1 invalid or fabricated")),
    ("query's own native ID or patient stem", f"{ae['own_native_or_stem_any_source']}/{n}", cell("query's own native")),
    ("true group named when absent (lexical / adjudicated)",
     f"{len(tg['lexical_hits'])}/{tg['eligible']} / {len(tg['after_adjudication'])}/{tg['eligible']}", cell("true group named")),
    ("probability statement present; all correct; every class quoted",
     f"{pa.get('with_statement', 0)}/{pa['n_nominal']}; {pa.get('all_correct', 0)}; {pa.get('all_classes_quoted', 0)}",
     cell("probability statement present")),
    ("automatic claim contradicting the cited record", f"{ae['claim_contradicting_record']}/{n} ({ae['n_claims_returned']})",
     cell("automatic claim contradicting")),
    ("zero automatic flags raw / after α–γ", f"{z['raw']}/{n} / {z['post_alpha_gamma']}/{n}", cell("zero automatic flags")),
    ("similarity or distance mentioned", f"{len(ae['rule12_similarity_mention'])}/{n}", cell("similarity or distance")),
    ("≥ 1 contradicted sentence, R1", f"{len(rt['contradicted_summaries'])}/{n} ({rt['contradicted_sentences']})",
     R1_PREFIX(cell("≥ 1 contradicted sentence"))),
    ("strict 'discrepant', R1", f"{len(rs['contradicted_summaries'])}/{n}", R1_PREFIX(cell("… strict"))),
    ("≥ 1 unsupported or spec.-diag., R1",
     f"{len(rt['unsup_or_specdiag_summaries'])}/{n} ({rt['unsup_or_specdiag_sentences']})",
     R1_PREFIX(cell("≥ 1 unsupported or spec.-diag."))),
    ("uncertain sentences, R1", f"{rt['uncertain_sentences']} (in {len(rt['uncertain_summaries'])} summaries)",
     R1_PREFIX(cell("uncertain sentences"))),
    ("every sentence supported or derivable, R1", f"{len(rt['all_sentences_supported_or_derivable'])}/{n}",
     R1_PREFIX(cell("every sentence supported"))),
    ("code distribution R1 (delivered)",
     " | ".join(str(d["codes"][c]) for c in ["supported", "derivable", "speculative-diagnostic", "unsupported",
                                             "contradicted", "uncertain"]) + f" | {d['prob_tag_Y']}",
     " | ".join(cell("R1 |", k) for k in range(1, 8))),
    ("code distribution R1 first drafts",
     " | ".join(str(f["codes"][c]) for c in ["supported", "derivable", "speculative-diagnostic", "unsupported",
                                             "contradicted", "uncertain"]),
     " | ".join(cell("R1 first drafts", k) for k in range(1, 7))),
    ("delivered rows / first-draft rows", f"{d['n_sentences']} / {f['n_sentences']}", "134 / 14"),
    ("alias sensitivity: uncertain 13 → 0; all-OK 7/20 → 20/20",
     f"uncertain {rt['uncertain_sentences']} → {al['uncertain_sentences']}; "
     f"{len(rt['all_sentences_supported_or_derivable'])}/{n} → {len(al['all_sentences_supported_or_derivable'])}/{n}",
     "uncertain 13 → 0; 7/20 → 20/20"),
    ("alternative_wording_strict: 2 sentences in 2/20; all-OK 7/20 → 6/20",
     f"{aw['unsup_or_specdiag_sentences']} sentences in {len(aw['unsup_or_specdiag_summaries'])}/{n}; "
     f"{len(rt['all_sentences_supported_or_derivable'])}/{n} → {len(aw['all_sentences_supported_or_derivable'])}/{n}",
     "2 sentences in 2/20; 7/20 → 6/20"),
]:
    cmp(f"report §2-§3: {label}", ref_cell, rendered)
rep_text = "\n".join(rep)
for s_ in ("uncertain 13 → 0", "7/20 → 20/20", "2 sentences in 2/20 summaries", "7/20 → 6/20", "134; first-draft rows of 5674 and 5681, 14"):
    cmp(f"report sensitivity text present: '{s_}'", True, s_ in rep_text)

# 5. the mechanical strict-"discrepant" rows reproduce round 2's v2 list (0 strict; 3 ambiguous)
sm = port_w["strict_discrepant_mechanical"]
cmp("mechanical strict-discrepant rows (v2: none)", [], sm["strict_rows"])
cmp("mechanical ambiguous strict rows (v2 note: R-7R2V s5, R-SARZ s2, R-2TMG s2)",
    [["R-2TMG", "2"], ["R-7R2V", "5"], ["R-SARZ", "2"]],
    sorted([x["rating_id"], x["sentence_no"]] for x in sm["ambiguous"] if x["delivered"]))
# 6. the mechanical abbreviation-expansion rule reproduces v2 alias_accepted_posthoc (14 rows)
adj = json.load(open(a.ref.parent / "lead_adjudication_r2_v2.json"))
v2 = next(s for s in adj["recode_sensitivities"] if s["name"] == "alias_accepted_posthoc")["rows"]
cmp("mechanical abbreviation-expansion rows = v2 alias_accepted_posthoc", sorted(v2),
    sorted(port_w["abbreviation_expansion_rows"]))
cmp("abbreviation-expansion sensitivity endpoints = R1|alias_accepted_posthoc", ref_w["rated"]["R1|alias_accepted_posthoc"],
    port_w["rated"]["abbreviation_expansion_accepted"])

lines = ["# Fidelity readback of analyze_reader80.py on round 2 (A0, rater 1)", "",
         f"Port output: `{a.port}`  ", f"Reference: `{a.ref}` and `{a.report}`  ",
         f"analyze_reader80.py sha256 `{json.load(open(a.port / 'endpoints.json'))['script_sha256']}`  ",
         f"compare script sha256 `{sha(__file__)}`  ",
         f"reference endpoints.json sha256 `{sha(a.ref / 'endpoints.json')}`; analysis_work.json "
         f"`{sha(a.ref / 'analysis_work.json')}`; ENDPOINTS_REPORT_R2.md `{sha(a.report)}`", "",
         f"**Result: {'ALL MATCH' if not bad else f'{bad} MISMATCH(ES)'}** ({len(rows)} comparisons). "
         "Out of scope: rater-2 rows, R1-vs-R2 agreement, the round-2 §7 example rule (single rater here).", "",
         "| check | result | reference | port |", "|---|---|---|---|"]
lines += [f"| {l} | {r} | `{x}` | `{y}` |".replace("\n", " ") for l, r, x, y in rows]
a.out.write_text("\n".join(lines) + "\n")
print("readback:", "ALL MATCH" if not bad else f"{bad} mismatches", len(rows), "comparisons")
sys.exit(1 if bad else 0)
