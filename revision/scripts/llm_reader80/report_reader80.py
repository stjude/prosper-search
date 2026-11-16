"""S5d report writer: assembles analysis/ENDPOINTS_REPORT_READER80.md (KEY-SIDE) from the join outputs.

Inputs (read only): analysis/join/{endpoints.json, analysis_work.json, rating_join_r1.csv, trace_readback.csv,
leading_consideration_sentences.csv, leading_consideration_by_summary_KEYSIDE.csv}, forms_filled/ (hash check),
lead_adjudication_reader80_v1.json (hash), round-2 ENDPOINTS_REPORT_R2.md §2 (orientation rows, quoted).
No verdict is written. Truth group names are never printed; key-side rows give case IDs and categories only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reader80_common as C  # noqa: E402

CODES = ["supported", "derivable", "unsupported", "speculative-diagnostic", "contradicted", "uncertain"]
US = ("unsupported", "speculative-diagnostic")
# Rows whose rater note gives a second reason besides the "primary" qualifier (null-site neighbours); excluded from
# the post hoc "primary qualifier set aside" sensitivity, which recodes only rows where that qualifier is the stated
# sole defect.
PRIMARY_SECOND_REASON = {("R-3WM3", "5"), ("R-VJ9M", "5")}
EXPECTED_V1_SHA = "e40b41f92b3947849dcb77cfefee8b7858c77aacbf0e03b071e5e33d06d77f04"
ORIENT_ROWS = ["rejected / too long", "style revisions", "invalid or fabricated citation", "true group named",
               "probability statement present", "automatic claim contradicting", "zero automatic flags",
               "similarity or distance", "≥ 1 contradicted sentence", "strict \"discrepant\"",
               "≥ 1 unsupported or spec.-diag.", "uncertain sentences", "every sentence supported or derivable"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=C.OUT_ROOT)
    a = ap.parse_args()
    R, J = a.root, a.root / "analysis" / "join"
    E = json.load(open(J / "endpoints.json"))
    W = json.load(open(J / "analysis_work.json"))
    rows = list(csv.DictReader(open(J / "rating_join_r1.csv")))
    tr = list(csv.DictReader(open(J / "trace_readback.csv")))
    lcs = list(csv.DictReader(open(J / "leading_consideration_sentences.csv")))
    lcq = {r["query"]: r for r in csv.DictReader(open(J / "leading_consideration_by_summary_KEYSIDE.csv"))}
    adj = R / "lead_adjudication_reader80_v1.json"
    assert sha(adj) == EXPECTED_V1_SHA
    frozen = {ln.split()[1]: ln.split()[0] for ln in open(R / "forms_filled" / "sha256_forms_frozen.txt") if ln.strip()}
    for f, h in frozen.items():
        assert sha(R / "forms_filled" / f) == h, f
    deliv = [r for r in rows if r["sentence_no"].isdigit()]
    first = [r for r in rows if not r["sentence_no"].isdigit()]
    assert len(deliv) == 480 and len(first) == 13
    coh = {r["query"]: r["cohort"] for r in rows}
    queries = sorted(coh)
    L = []
    w = L.append

    # ------------------------------------------------------------------ header
    w("# S5d — grounding rating of the 80 reader-study summaries: key-side endpoint report")
    w("")
    w("**Key-side document** (case IDs, run IDs, rater traces and summary sentences; no reference-group names are "
      "printed). Plan `docs/analysis/s5d_reader_summaries_rating_plan_candidate_001_20261113.md` as amended by "
      "`..._candidate_002_20261115.md` (binding). Descriptive only: **no met / not-met verdict** (001 §5). Model-rated "
      "containment check, not expert assessment; one rater configuration (Claude Opus 5.5, high reasoning) in eight "
      "fresh contexts, one per packet of ten; no replicate rater, so no agreement figure (author decision); the only "
      "calibration of this configuration is rounds 1–2 (R1–R2 same code 105/134). Held internally until Haoran decides.")
    w("")
    w(md_table(["input", "sha256"], [
        ["`lead_adjudication_reader80_v1.json` (frozen by the lead before any form was filled)", sha(adj)[:16] + "…"],
        *[[f"`forms_filled/{f}`", h[:16] + "…"] for f, h in sorted(frozen.items())],
        ["`analysis/join/endpoints.json` (written by `analyze_reader80.py` " + E["script_sha256"][:8] + "…)",
         sha(J / "endpoints.json")[:16] + "…"],
        ["`analysis/join/rating_join_r1.csv`", sha(J / "rating_join_r1.csv")[:16] + "…"],
        ["`analysis/join/trace_readback.csv`", sha(J / "trace_readback.csv")[:16] + "…"]]))
    w("")

    # ------------------------------------------------------------------ 0 validation
    fc = W["form_checks"]
    w("## 0. Form validation (before the join)")
    w("")
    w(f"- Eight forms re-hashed against `sha256_forms_frozen.txt`: 8/8 equal. Each form's rows, order, "
      f"`rating_id`, `sentence_no` and `sentence_text` equal its packet's empty form; columns equal; every code in the "
      f"six allowed codes; 0 missing codes; 493 rows (480 delivered sentences + 13 first-draft sentences of 2 items).")
    w("- Probability tag: the instructions said only \"leave empty if not a probability statement\" and fixed no value. "
      "Contexts wrote `probability-statement` (B–H), `probability-statement (qualitative; no value stated)` (H, 4 rows) "
      "or the substantive code (A, 11 rows). Format-only normalisation: any non-empty tag = tagged; raw values kept in "
      "`probability_statement_tag_raw`. Contexts also differed in what they tagged (B, D and F: only sentences stating a "
      "value; C and G: also sentences invoking or ranking the probability line; H: qualitative ones marked as such), so the tag count is a per-context convention, not a measured quantity.")
    w("- Rater notes (read in full; key-side): every context reports `sha256sum -c` OK and rating in packet order; A, B, C "
      "and F state explicitly that no earlier rating was revised; all but E state that a script only wrote their "
      "manually decided codes to CSV (and checked completeness/order). Context F "
      "discloses that its \"primary\" policy was formed while reading item 3 and applied to item 1 before item 1 was "
      "recorded. Self-reported times 2.5–12 min per packet.")
    w("- Adjudication v1 differs from the implementer's DRAFT only by the two lead rulings (case053 \"24\" = β; case040 "
      "leak flag = not a leak, class raw, listed in `true_group_not_a_leak`).")
    w("")

    # ------------------------------------------------------------------ 1 headline endpoints
    by = defaultdict(dict)
    order = []
    for e in E["endpoints"]:
        if e["name"] not in order:
            order.append(e["name"])
        by[e["name"]][e["subset"]] = e
    w("## 1. Endpoints of plan 001 §5 (delivered texts; overall and by cohort)")
    w("")
    w("Denominators: summaries unless stated; sentence rows count delivered sentences only (480: CCDI-MCI 243, STV 237). "
      "Rater = the coded forms; auto = harness/parser; SD rule = the mechanical SD-1/SD-2 rule fixed before rating.")
    w("")
    trows = []
    for name in order:
        cells = []
        for s in ("all", "CCDI-MCI", "STV"):
            e = by[name].get(s)
            cells.append("" if e is None else (e["value"] if e["value"] is not None else ""))
        trows.append([name, *cells, by[name]["all"]["rater"]])
    w(md_table(["endpoint", "all 80", "CCDI-MCI 40", "STV 40", "source"], trows))
    w("")
    w("Notes on the overall rows (verbatim from `endpoints.json`):")
    w("")
    for name in order:
        n = by[name]["all"]["notes"]
        if n:
            w(f"- *{name}*: {n}")
    w("")

    # ------------------------------------------------------------------ 2 code distributions
    w("## 2. Code distribution — delivered rows and first-draft rows separately")
    w("")
    def dist(rs):
        c = Counter(r["code"] for r in rs)
        return [c.get(k, 0) for k in CODES] + [sum(r["probability_statement_tag"] == "Y" for r in rs), len(rs)]
    w(md_table(["rows", *CODES, "prob-tag", "n"], [
        ["delivered, all", *dist(deliv)],
        ["delivered, CCDI-MCI", *dist([r for r in deliv if r["cohort"] == "CCDI-MCI"])],
        ["delivered, STV", *dist([r for r in deliv if r["cohort"] == "STV"])],
        ["first draft (case026, case028; STV)", *dist(first)]]))
    w("")
    w("First-draft rows are reported only here; they enter no endpoint. case073's style revision left the text "
      "unchanged, so it has no first-draft rows. The 13 first-draft rows contain no unsupported, speculative-diagnostic, "
      "contradicted or uncertain code.")
    w("")

    # ------------------------------------------------------------------ 3 SD rule
    sd = W["sd_rows"]
    w("## 3. SD-1 / SD-2 (plan 001 §4 S4; mechanical, fixed before rating)")
    w("")
    w(f"Rule: {W['sd_rule']}")
    w("")
    sdc = Counter((r["cohort"], r["sd_class"]) for r in sd)
    w(md_table(["", "SD-1 sentences", "SD-2 sentences", "summaries with SD-1", "summaries with SD-2"], [
        [c, sdc[(c, "SD-1")], sdc[(c, "SD-2")], len({r["query"] for r in sd if r["cohort"] == c and r["sd_class"] == "SD-1"}),
         len({r["query"] for r in sd if r["cohort"] == c and r["sd_class"] == "SD-2"})] for c in ("CCDI-MCI", "STV")]))
    w("")
    n_stated = sum(r["form"] == "stated" for r in lcs)
    lc_codes = Counter(r["code"] for r in lcs if r["form"] == "stated")
    w(f"- The one SD-1 sentence (R-33DX s7, case045) is SD-1 by form, but the rater's stated reason is the \"primary\" "
      f"qualifier (\"testicular primary\"), not the leading-consideration conclusion. No sentence was coded "
      f"unsupported or speculative-diagnostic *because of* the instruction-invited conclusion.")
    w(f"- The stated SD-1 form occurs in {n_stated} delivered sentences in {len({r['query'] for r in lcs if r['form'] == 'stated'})} "
      f"summaries (one more uses \"X is the leading consideration\"). The raters coded them: "
      f"{', '.join(f'{k} {v}' for k, v in lc_codes.most_common())}. Plan 001 expected SD-1 \"in most summaries\"; under "
      f"this rater configuration the hedged conclusion was read as supported or derivable, so SD-1 is near-empty and the "
      f"SD-1-set-aside row equals the as-coded row. The pre-registered split therefore does not separate the two "
      f"populations it was built for; §5 shows that contexts split between *supported* and *derivable* for this sentence.")
    w("- \"Any sentence naming a diagnosis absent from the input\" counts groups the harness alias table recognises "
      "(plus probability keys); a diagnosis outside that vocabulary would not be detected. Sentences with "
      "non-group content (\"small round blue cell tumor differential\") are listed in §6.")
    w("")

    # ------------------------------------------------------------------ 4 sensitivities
    w("## 4. Sensitivities")
    w("")
    sm = W["strict_discrepant_mechanical"]
    jx0 = {(r["rating_id"], r["sentence_no"]): r for r in rows}
    rated = W["rated"]
    n_disc_d = sum(bool(re.search("discrepan", r["sentence_text"], re.I)) for r in deliv)
    strict_codes = Counter(jx0[tuple(k)]["code"] for k in sm["strict_rows"])
    w(f"**Strict \"discrepant\" (pre-specified).** {sm['sentences_with_discrepant']} rated sentences ({n_disc_d} delivered) "
      f"contain \"discrepan\"; the mechanical round-2 rule finds {len(sm['strict_rows'])} strict rows (a neighbour called "
      f"discrepant whose flag is N) and {sum(x['delivered'] for x in sm['ambiguous'])} delivered ambiguous rows "
      f"(+{sum(not x['delivered'] for x in sm['ambiguous'])} first-draft) that the rule cannot resolve; the "
      f"adjudication JSON lists no strict rows. Strict reading: ≥ 1 contradicted sentence "
      f"{len(rated['strict_discrepant']['contradicted_summaries'])}/80 ({rated['strict_discrepant']['contradicted_sentences']} "
      f"sentences) vs lenient {len(rated['as_rated']['contradicted_summaries'])}/80 "
      f"({rated['as_rated']['contradicted_sentences']}). The raters had coded the 15 strict rows "
      f"{', '.join(f'{k} {v}' for k, v in strict_codes.most_common())}: the lenient reading (\"discrepant\" = differs "
      f"from the majority) was the one every context applied, and one context (B, R-WD2G s4) coded contradicted on the "
      f"flag reading. Ambiguous rows for the lead (not recoded):")
    w("")
    jx = {(r["rating_id"], r["sentence_no"]): r for r in rows}
    w(md_table(["rating", "s", "case", "packet", "rule buckets", "rater code", "sentence"],
               [[x["rating_id"], x["sentence_no"], x["query"], jx[(x["rating_id"], x["sentence_no"])]["packet"],
                 x["buckets"], jx[(x["rating_id"], x["sentence_no"])]["code"],
                 jx[(x["rating_id"], x["sentence_no"])]["sentence_text"]] for x in sm["ambiguous"]]))
    w("")
    w("Strict rows recoded contradicted (mechanical): " + ", ".join(f"{a} s{b}" for a, b in sm["strict_rows"]) + ".")
    w("")
    w(f"**Abbreviation expansion for \"uncertain\" (pre-specified).** {len(W['abbreviation_expansion_rows'])} rows "
      f"qualify: neither uncertain row concerns an expansion (both contexts that used *uncertain* accepted spelled-out "
      f"abbreviations as aliases). Uncertain stays 2/480 (R-8UPL s6 \"priors\"; R-HP5Z s6 tissue-category site).")
    w("")
    # post hoc primary
    prim = {(t["rating_id"], t["sentence_no"]) for t in tr if t["reason_type"] == "primary qualifier"
            and t["code"] in US} - PRIMARY_SECOND_REASON
    def us_summaries(rs, drop=frozenset()):
        return {r["query"] for r in rs if r["code"] in US and (r["rating_id"], r["sentence_no"]) not in drop}
    def ok_summaries(rs, drop=frozenset()):
        bad = {r["query"] for r in rs if r["code"] not in ("supported", "derivable")
               and (r["rating_id"], r["sentence_no"]) not in drop}
        return set(queries) - bad
    w(f"**Post hoc, not pre-specified — the \"primary\" qualifier.** "
      f"{sum(1 for t in tr if t['reason_type'] == 'primary qualifier' and t['code'] in US)} of the "
      f"{sum(r['code'] in US for r in deliv)} unsupported/speculative-diagnostic sentences were coded unsupported because "
      f"the summary calls the query site \"primary\" while the query's `met` field is null (CCDI-MCI payloads only; "
      f"STV payloads carry no query site). Every context that met a query-site \"primary\" coded it unsupported (or "
      f"contradicted for another reason); packet D's summaries contain none, so the reading was applied uniformly and "
      f"its per-packet counts reflect which summaries use the word (§5). Setting aside the {len(prim)} rows where the rater named the qualifier as the "
      f"sole defect (all other content supported in the rater's own trace):")
    w("")
    w(md_table(["", "all 80", "CCDI-MCI 40", "STV 40"], [
        ["≥ 1 unsupported/spec.-diag. sentence — as coded",
         *[f"{len(us_summaries([r for r in deliv if c in (None, r['cohort'])]))}/{n}" for c, n in ((None, 80), ("CCDI-MCI", 40), ("STV", 40))]],
        ["… \"primary\"-only rows set aside (post hoc)",
         *[f"{len(us_summaries([r for r in deliv if c in (None, r['cohort'])], prim))}/{n}" for c, n in ((None, 80), ("CCDI-MCI", 40), ("STV", 40))]],
        ["U/SD sentences — as coded / set aside",
         *[f"{sum(r['code'] in US for r in deliv if c in (None, r['cohort']))} / "
           f"{sum(r['code'] in US and (r['rating_id'], r['sentence_no']) not in prim for r in deliv if c in (None, r['cohort']))}"
           for c in (None, "CCDI-MCI", "STV")]],
        ["every sentence supported or derivable — as coded",
         *[f"{len([q for q in ok_summaries(deliv) if c in (None, coh[q])])}/{n}" for c, n in ((None, 80), ("CCDI-MCI", 40), ("STV", 40))]],
        ["… \"primary\"-only rows set aside (post hoc)",
         *[f"{len([q for q in ok_summaries(deliv, prim) if c in (None, coh[q])])}/{n}" for c, n in ((None, 80), ("CCDI-MCI", 40), ("STV", 40))]]]))
    w("")
    w("This sensitivity is descriptive and was defined after reading the rater notes; it is not a recoding of the "
      "primary result. The cohort difference in U/SD (CCDI-MCI 24/40 vs STV 3/40) is largely this qualifier, which "
      "only CCDI-MCI payloads (query site present, `met` null) can trigger.")
    w("")
    w("**Probability statements (automatic, analyst-checked).** The parser marks one value \"below 2 significant "
      "figures\": case040 \"RMS 0.06\" (R-XTGD s5), where the payload value is exactly 0.06 (= 6 %, a whole percent); "
      "the rater coded it supported. Analyst check: all values correct in 80/80 (parser 79/80).")
    w("")

    # ------------------------------------------------------------------ 5 per-packet variability
    w("## 5. Between-context variability (one context per packet)")
    w("")
    pk = sorted({r["packet"] for r in rows})
    def pc(rs, p):
        c = Counter(r["code"] for r in rs if r["packet"] == p)
        return [c.get(k, 0) for k in CODES]
    w(md_table(["packet", *CODES, "delivered n", "first-draft n", "prob-tag (raw convention)"],
               [[p, *pc(deliv, p), sum(r["packet"] == p for r in deliv), sum(r["packet"] == p for r in first),
                 sum(r["probability_statement_tag"] == "Y" for r in deliv if r["packet"] == p)] for p in pk]))
    w("")
    w("Each packet holds 5 CCDI-MCI + 5 STV items, so packet differences mix context and item effects; with one "
      "context per packet they cannot be separated. Three recurring sentence types, coded by packet:")
    w("")
    def by_packet(pred):
        out = []
        for p in pk:
            c = Counter(r["code"] for r in deliv if r["packet"] == p and pred(r))
            out.append([p, sum(c.values()), *[c.get(k, 0) for k in CODES]])
        return out
    lc_keys = {(r["rating_id"], r["sentence_no"]) for r in lcs}
    w("*Leading-consideration sentences* (stated SD-1 form or \"X is the leading consideration\"; 72):")
    w("")
    w(md_table(["packet", "n", *CODES], by_packet(lambda r: (r["rating_id"], r["sentence_no"]) in lc_keys)))
    w("")
    w("*CCDI-MCI sentence 1 containing \"primary\"* (the opening description of the query):")
    w("")
    w(md_table(["packet", "n", *CODES], by_packet(lambda r: r["cohort"] == "CCDI-MCI" and r["sentence_no"] == "1"
                                                  and re.search(r"\bprimary\b", r["sentence_text"], re.I))))
    w("")
    w("*Sentences containing \"discrepan\"*:")
    w("")
    w(md_table(["packet", "n", *CODES], by_packet(lambda r: re.search(r"discrepan", r["sentence_text"], re.I))))
    w("")
    w("Packets A, C and H coded the leading-consideration sentence derivable (inference from votes/probabilities); "
      "B, D, E, F and most of G coded it supported. Both are non-flagging codes, so endpoints do not move, but the "
      "supported/derivable split is a context convention. The \"primary\" openers were coded unsupported in every "
      "packet that has one (D has none; the other uses of \"primary\" describe neighbours, whose sites carry (P)/(M), "
      "and were coded supported/derivable). Contradicted codes come from four contexts (B 1, D 1, E 3, G 1).")
    w("")

    # ------------------------------------------------------------------ 6 listing
    w("## 6. Every sentence coded contradicted, unsupported, speculative-diagnostic or uncertain (46)")
    w("")
    w("Original codes kept. *Readback* = the implementer's check, against the payload as sent, of the payload facts the "
      "rater cited (`scripts/llm_reader80/trace_readback_reader80.py`; each check prints the actual value). It verifies "
      "the stated reason, not the code; it is not a re-rating. Interpretive steps are named in the remark.")
    w("")
    trk = {(t["rating_id"], t["sentence_no"]): t for t in tr}
    order_codes = {"contradicted": 0, "speculative-diagnostic": 1, "unsupported": 2, "uncertain": 3}
    lst = sorted([r for r in rows if r["code"] in order_codes],
                 key=lambda r: (order_codes[r["code"]], r["cohort"], r["query"], int(r["sentence_no"])))
    for r in lst:
        t = trk[(r["rating_id"], r["sentence_no"])]
        w(f"- **{r['rating_id']} s{r['sentence_no']}** — {r['query']} ({r['cohort']}), packet {r['packet']}, "
          f"**{r['code']}**, SD class: {t['sd_class'] or 'n/a (not U/SD)'}  ")
        w(f"  Sentence: \"{r['sentence_text']}\"  ")
        w(f"  Rater trace: {r['support_trace']}  ")
        if r["note"]:
            w(f"  Rater note: {r['note']}  ")
        w(f"  Readback: cited facts verified **{t['cited_facts_verified']}** ({t['checks_passed']}): {t['evidence']}.  ")
        w(f"  Remark: {t['remark']}")
    w("")
    w(f"Readback result: {sum(t['cited_facts_verified'] == 'yes' for t in tr)}/{len(tr)} rows — every payload fact "
      f"a rater cited for these codes is literally present (or absent, where the rater said absent). "
      f"Contradicted: {sum(t['code'] == 'contradicted' for t in tr)}; SD-2: {sum(t['sd_class'] == 'SD-2' for t in tr)}; "
      f"SD-1: {sum(t['sd_class'] == 'SD-1' for t in tr)}; uncertain: {sum(t['code'] == 'uncertain' for t in tr)}. "
      f"Where the code depends on a reading (\"Chest\" = thorax; \"two treated\" scoped to RMS; \"discrepant\" as flag "
      f"vs majority; \"primary\" as claim vs wording) the remark says so.")
    w("")

    # ------------------------------------------------------------------ 7 §6 can row
    w("## 7. Key-side: the instruction-invited leading-consideration statement vs the reference group (001 §6 \"can\")")
    w("")
    w("Per summary, the group(s) named as the leading consideration (X of the stated SD-1 form, else the clause before "
      "\"is the leading consideration\"; groups by harness aliases + probability keys) compared with the reference "
      "group from `key/`. Independent of the rater's code. Descriptive; this is a property of retrieval + instruction, "
      "not diagnostic accuracy.")
    w("")
    cats = ["reference group only", "reference group and other group(s)", "other group(s) only", "no group recognised",
            "no leading-consideration statement"]
    def catrow(c):
        out = []
        for s, n in ((None, 80), ("CCDI-MCI", 40), ("STV", 40)):
            out.append(f"{sum(1 for q, v in lcq.items() if v['category'] == c and s in (None, v['cohort']))}/{n}")
        return [c, *out]
    w(md_table(["category", "all", "CCDI-MCI", "STV"], [catrow(c) for c in cats]))
    w("")
    other = [v for v in lcq.values() if v["category"] == "other group(s) only"]
    absent = [v for v in other if v["reference_in_input"] == "False"]
    present = [v for v in other if v["reference_in_input"] == "True"]
    w(f"- Of the {len(other)} \"other group(s) only\": {len(absent)} are the truth-absent cases (reference group in no "
      f"neighbour label or probability key: {', '.join(sorted(v['query'] for v in absent))}), where naming it was "
      f"impossible from the input; {len(present)} had the reference group in the input but named another "
      f"({', '.join(sorted(v['query'] for v in present))}).")
    nb_top = sum(v["named_equals_top_probability"] == "True" for v in lcq.values())
    nb_mode = sum(v["named_equals_neighbour_mode"] == "True" for v in lcq.values())
    has = sum(v["basis"] != "none" for v in lcq.values())
    w(f"- The named group(s) equal the top-probability class in {nb_top}/{has} and the neighbour tumor_group mode in "
      f"{nb_mode}/{has} summaries with a statement: the statement restates the input's leading class; whether it "
      f"matches the reference is a property of retrieval.")
    w(f"- Summaries without a leading-consideration statement: "
      f"{', '.join(sorted((q for q, v in lcq.items() if v['basis'] == 'none'), key=lambda x: x))}.")
    w("")

    # ------------------------------------------------------------------ 8 orientation
    rep = C.ROUND2 / "analysis" / "ENDPOINTS_REPORT_R2.md"
    txt = rep.read_text().split("## 2.")[1].split("## 3.")[0]
    orient = [ln for ln in txt.splitlines() if ln.startswith("| ") and any(k in ln for k in ORIENT_ROWS)]
    w("## 8. Orientation only: rounds 1–2 A0 (20 queries)")
    w("")
    w(f"Quoted verbatim from `s5c_pilot_round2_20261110/analysis/ENDPOINTS_REPORT_R2.md` §2 (sha256 {sha(rep)[:16]}…; "
      f"round-2 rater-1 rows reproduced by this port, `analysis/readback_round2.md`). Different queries, k (5 vs 10), "
      f"instruction, date and rater contexts: not a comparison of like with like; never pooled.")
    w("")
    w("| endpoint | round 2 A0 | round 1 A0 (unchanged) |")
    w("|---|---|---|")
    for ln in orient:
        w(ln)
    w("")

    # ------------------------------------------------------------------ 9 disclosures
    w("## 9. What qualifies these numbers")
    w("")
    for s in [
        "Model rater, single configuration, eight contexts, no replicate: no agreement estimate; §5 shows convention "
        "differences between contexts (supported vs derivable; \"primary\"; probability tag) that a replicate would "
        "have measured.",
        "The dominant unsupported reason (32/38: the query site called \"primary\" with `met` null, CCDI-MCI only) is a "
        "strict but uniformly applied reading; the raters' traces call it the only (or a minor) defect, and the cohort contrast follows the payload "
        "format. Set aside post hoc: summaries with U/SD 8/80, every sentence supported/derivable 64/80.",
        "The pre-registered SD-1/SD-2 split assumed the leading-consideration sentence would be flagged; it was not, so "
        "SD-2 here is essentially \"everything flagged\", and the SD-1 row is 0 by summary.",
        "Strict \"discrepant\" (17/80) vs lenient (6/80) is wide; 7 delivered ambiguous rows await a lead ruling.",
        "The 6 contradicted sentences are site-list, treatment-count and \"discrepant\" characterisations of neighbours; none concerns a "
        "probability, a citation or the leading consideration; cited facts verified 6/6, four depend on a reading "
        "(R-WD2G flag vs majority, R-SW9V chest = thorax, R-33DX scope, R-W7QY \"Nasal\" label).",
        "Automatic layer: 0 invalid/fabricated citations; 0/5 reference group named when absent; 0 automatic claim "
        "contradictions; zero flags after α–γ 74/80 (remaining: 5 `grounding_other` warnings, 1 not-a-leak flag).",
        "Case set = the reader-study design set (selected, not prevalence-weighted; 5 truth-absent cases); no typical-use "
        "rate; no link to reader outcomes (001 §6).",
        "The rubric paragraph is the round-1 text, written for that pilot; the sentence split, IDs and payloads are "
        "as described in EXECUTION_RECORD.md. The probability-tag column is a context convention (§0).",
    ]:
        w(f"- {s}")
    w("")
    (R / "analysis" / "ENDPOINTS_REPORT_READER80.md").write_text("\n".join(L) + "\n")
    print("wrote", R / "analysis" / "ENDPOINTS_REPORT_READER80.md", len(L), "lines")


if __name__ == "__main__":
    main()
