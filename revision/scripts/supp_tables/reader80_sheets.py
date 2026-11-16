"""The two new S48 publication sheets for the sentence rating of the 80 study-case
summaries shown to readers.

reader80_endpoints   endpoints overall and per collection with denominators; code distribution of the 480 delivered
                     sentences; the pre-specified strict-"discrepant" sensitivity; the post hoc "primary" sensitivity
reader80_by_summary  one row per study case (opaque case number used in S31) and collection: delivered sentences,
                     count per code, automatic flags and their classes, leading-consideration statement yes/no

Built only from the retained analysis outputs of $PROSPER_OUT/s5d_reader80_rating_20261115/analysis/join/
(endpoints.json, rating_join_r1.csv delivered rows, flag_classification.csv, leading_consideration_by_summary_KEYSIDE.csv
[basis column only], leading_consideration_sentences.csv [form and code only]); the post hoc "primary" rows cannot be
recomputed from structured fields and are transcribed from ENDPOINTS_REPORT_READER80.md section 4 (marked so).
No sentence text, input content, reference label, rating identifier, rater label or slide identifier is written.
Every published total is asserted against the sources.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output")
AN = OUT / "s5d_reader80_rating_20261115/analysis"
J = AN / "join"
SOURCES = {
    "endpoints": J / "endpoints.json",
    "rating_join": J / "rating_join_r1.csv",
    "flags": J / "flag_classification.csv",
    "leading_by_summary": J / "leading_consideration_by_summary_KEYSIDE.csv",
    "leading_sentences": J / "leading_consideration_sentences.csv",
    "report": AN / "ENDPOINTS_REPORT_READER80.md",
}
CODES = ["supported", "derivable", "unsupported", "speculative-diagnostic", "contradicted", "uncertain"]
COLL = ["CCDI-MCI", "STV"]
FLAG_LABEL = {
    ("grounding_unverified_number", "beta"): "number equal to a value in the input (checker artifact)",
    ("grounding_other", "raw"): "other grounding warning",
    ("true_group_leak_candidate", "raw"): ("group abbreviation matched as a substring (adjudicated: not a naming of the "
                                           "reference group)"),
}
# Post hoc "primary" sensitivity: ENDPOINTS_REPORT_READER80.md section 4 (all / CCDI-MCI / STV), transcribed.
POST_HOC = [
    ("summaries with >= 1 unsupported or speculative-diagnostic sentence, 'primary'-only sentences set aside",
     (8, 80), (5, 40), (3, 40), "summaries"),
    ("unsupported or speculative-diagnostic sentences, 'primary'-only sentences set aside",
     (8, 480), (5, 243), (3, 237), "sentences"),
    ("summaries with every sentence supported or derivable, 'primary'-only sentences set aside",
     (64, 80), (27, 40), (37, 40), "summaries"),
]
POST_HOC_REPORT_LINES = ["| … \"primary\"-only rows set aside (post hoc) | 8/80 | 5/40 | 3/40 |",
                         "| U/SD sentences — as coded / set aside | 38 / 8 | 35 / 5 | 3 / 3 |",
                         "| … \"primary\"-only rows set aside (post hoc) | 64/80 | 27/40 | 37/40 |"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    ep = json.loads(SOURCES["endpoints"].read_text())
    E = {(e["name"], e["subset"]): e for e in ep["endpoints"]}
    rows = list(csv.DictReader(open(SOURCES["rating_join"], encoding="utf-8")))
    deliv = [r for r in rows if not r["sentence_no"].startswith("F")]
    flags = list(csv.DictReader(open(SOURCES["flags"], encoding="utf-8")))
    lead = {r["query"]: r for r in csv.DictReader(open(SOURCES["leading_by_summary"], encoding="utf-8"))}
    lsent = list(csv.DictReader(open(SOURCES["leading_sentences"], encoding="utf-8")))
    return E, rows, deliv, flags, lead, lsent


def build():
    E, rows, deliv, flags, lead, lsent = load()
    # ---------------------------------------------------------------- totals asserted
    assert len(rows) == 493 and len(deliv) == 480, (len(rows), len(deliv))
    cc = Counter(r["code"] for r in deliv)
    assert [cc[c] for c in CODES] == [330, 104, 36, 2, 6, 2], cc
    coll_of = {}
    for r in rows:
        assert coll_of.setdefault(r["query"], r["cohort"]) == r["cohort"]
    assert sorted(coll_of) == [f"case{n:03d}" for n in range(1, 81)]
    n_coll = Counter(coll_of.values())
    assert n_coll == {"CCDI-MCI": 40, "STV": 40}, n_coll
    per = defaultdict(Counter)
    for r in deliv:
        per[r["query"]][r["code"]] += 1
    nsent = {q: sum(per[q].values()) for q in coll_of}
    assert sum(nsent[q] for q in coll_of if coll_of[q] == "CCDI-MCI") == 243
    all_sd = {q for q in coll_of if per[q]["supported"] + per[q]["derivable"] == nsent[q]}
    contra = {q for q in coll_of if per[q]["contradicted"]}
    usd = {q for q in coll_of if per[q]["unsupported"] + per[q]["speculative-diagnostic"]}
    assert (len(all_sd), len(contra), len(usd)) == (48, 6, 27)
    assert sorted(contra) == sorted(eval(E[(">= 1 contradicted sentence — lenient 'discrepant' (primary)", "all")]["notes"]
                                         .split("; ", 1)[1]))
    assert len(flags) == 42 and all(f["source"] == "final" for f in flags)
    fl = defaultdict(Counter)
    for f in flags:
        key = (f["flag_type"], f["class"])
        assert key in FLAG_LABEL, key
        assert (f["counts_after_alpha_gamma"] == "True") == (f["class"] == "raw")
        fl[f["query"]][key] += 1
    assert Counter(k for q in fl for k in fl[q].elements()) == {("grounding_unverified_number", "beta"): 36,
                                                                  ("grounding_other", "raw"): 5,
                                                                  ("true_group_leak_candidate", "raw"): 1}
    assert len(fl) == 41 and 80 - len(fl) == 39
    remaining = {q for q in fl if any(k[1] == "raw" for k in fl[q])}
    assert 80 - len(remaining) == 74
    lc = {q: lead[q]["basis"] for q in coll_of}
    assert Counter(lc.values()) == {"stated": 71, "other phrasing": 1, "none": 8}
    assert {s["query"] for s in lsent} == {q for q in lc if lc[q] != "none"} and len(lsent) == 72

    def ep(name, sub):
        e = E[(name, sub)]
        return e["numerator"], e["denominator"]

    def frac(name):
        return [ep(name, s) for s in ("all", "CCDI-MCI", "STV")]

    def by_coll(pred, unit="summaries"):
        out = [(sum(1 for q in coll_of if pred(q)), 80)]
        for c in COLL:
            out.append((sum(1 for q in coll_of if coll_of[q] == c and pred(q)), 40))
        return out

    def sent_coll(pred):
        out = [(sum(1 for r in deliv if pred(r)), 480)]
        for c in COLL:
            sub = [r for r in deliv if r["cohort"] == c]
            out.append((sum(1 for r in sub if pred(r)), len(sub)))
        return out

    # ---------------------------------------------------------------- endpoint rows; cross-checked with endpoints.json
    hdr = ["block", "endpoint", "all: n", "all: N", "CCDI-MCI: n", "CCDI-MCI: N", "STV: n", "STV: N", "unit",
           "evaluated by", "note"]
    R = []

    def add(block, name, vals, unit, by, note=""):
        R.append([block, name, vals[0][0], vals[0][1], vals[1][0], vals[1][1], vals[2][0], vals[2][1], unit, by, note])

    B1 = "Summaries and automatic checks"
    add(B1, "rejected, too long, empty or error", frac("rejected / too long / empty / error"), "summaries", "automatic")
    add(B1, "style revision applied", frac("style revisions applied"), "summaries", "automatic")
    add(B1, ">= 1 invalid or fabricated citation (cited identifier not in the input)",
        frac(">= 1 invalid or fabricated citation (every cited id in payload)"), "summaries", "automatic")
    add(B1, "reference group named when absent from the ten neighbors and the probability line: word search",
        frac("reference group named when absent from neighbours and scores — lexical"), "summaries", "automatic")
    add(B1, "reference group named when absent from the ten neighbors and the probability line: after adjudication",
        frac("reference group named when absent from neighbours and scores — adjudicated"), "summaries",
        "analyst adjudication")
    pr = [E[("probability statement present; all values correct; every class quoted", s)]["value"]
          for s in ("all", "CCDI-MCI", "STV")]
    assert pr[0].startswith("present 80/80; all correct 79; incorrect 1; every class quoted 80")
    assert pr[1].startswith("present 40/40; all correct 40;") and pr[2].startswith("present 40/40; all correct 39;")
    add(B1, "probability statement present", [(80, 80), (40, 40), (40, 40)], "summaries", "automatic (parser)")
    add(B1, "all probability values correct: parser", [(79, 80), (40, 40), (39, 40)], "summaries", "automatic (parser)",
        "the one value the parser marked (below two significant figures) equals the input value exactly")
    add(B1, "all probability values correct: analyst check", [(80, 80), (40, 40), (40, 40)], "summaries", "analyst check")
    add(B1, "every probability class quoted", [(80, 80), (40, 40), (40, 40)], "summaries", "automatic (parser)")
    claims = [int(E[("automatic claim contradicting the cited record", s)]["notes"].split()[0]) for s in
              ("all", "CCDI-MCI", "STV")]
    assert claims == [396, 192, 204]
    add(B1, "structured claims contradicting the cited record", [(0, claims[0]), (0, claims[1]), (0, claims[2])],
        "claims returned", "automatic")
    add(B1, "no automatic flag: raw", frac("zero automatic flags — raw"), "summaries", "automatic")
    add(B1, "no automatic flag: after the pre-defined checker-artifact classes",
        frac("zero automatic flags — after alpha-gamma"), "summaries", "automatic and analyst classes",
        "flags and classes per summary in reader80_by_summary")
    add(B1, "similarity or distance mentioned", frac("similarity or distance mentioned"), "summaries", "automatic")
    add(B1, "query's own identifier in the text", frac(
        "query's own identifier appears in the text (given in the payload; descriptive, not a leak channel)"), "summaries",
        "automatic", "the identifier is given in the input; descriptive, not a leak channel")

    B2 = "Sentence rating (one model rater)"
    v = frac(">= 1 contradicted sentence — lenient 'discrepant' (primary)")
    assert v == by_coll(lambda q: q in contra)
    add(B2, ">= 1 contradicted sentence (\"discrepant\" read as differing from the majority diagnosis; primary)", v,
        "summaries", "model rater", "6 sentences")
    v = frac(">= 1 contradicted sentence — strict 'discrepant' (sensitivity)")
    assert v == [(17, 80), (11, 40), (6, 40)]
    add(B2, ">= 1 contradicted sentence (\"discrepant\" read strictly as the quality-control flag; pre-specified "
        "sensitivity)", v, "summaries", "model rater and a mechanical rule fixed before rating",
        "20 sentences (CCDI-MCI 14, STV 6)")
    v = frac(">= 1 unsupported or speculative-diagnostic sentence — total")
    assert v == by_coll(lambda q: q in usd)
    add(B2, ">= 1 unsupported or speculative-diagnostic sentence", v, "summaries", "model rater")
    add(B2, "unsupported or speculative-diagnostic sentences",
        sent_coll(lambda r: r["code"] in ("unsupported", "speculative-diagnostic")), "sentences", "model rater")
    add(B2, "any sentence naming a diagnosis absent from the input",
        frac("any sentence naming a diagnosis absent from the input (subset of SD-2)"), "summaries",
        "model rater and the checker's group aliases", "groups recognized by the checker's alias table only")
    v = frac("uncertain sentences — as coded")
    assert v == sent_coll(lambda r: r["code"] == "uncertain")
    add(B2, "uncertain sentences", v, "sentences", "model rater")
    assert frac("uncertain sentences — abbreviation-expansion accepted (sensitivity)") == v
    add(B2, "uncertain sentences, spelled-out abbreviations accepted (pre-specified sensitivity)", v, "sentences",
        "model rater and a mechanical rule fixed before rating", "no uncertain sentence concerned an abbreviation")
    v = frac("every sentence supported or derivable — as coded")
    assert v == by_coll(lambda q: q in all_sd)
    add(B2, "every sentence supported or derivable", v, "summaries", "model rater")

    B3 = "Code distribution of the delivered sentences"
    for c in CODES:
        add(B3, c, sent_coll(lambda r, c=c: r["code"] == c), "sentences", "model rater",
            "13 sentences of two first-draft texts that were revised for style are excluded (11 supported, 2 derivable)"
            if c == "supported" else "")
    first = Counter(r["code"] for r in rows if r["sentence_no"].startswith("F"))
    assert first == {"supported": 11, "derivable": 2}

    B4 = "Leading-consideration statement"
    add(B4, "statement present (any phrasing)", by_coll(lambda q: lc[q] != "none"), "summaries", "automatic")
    v = frac("leading-consideration statement (stated SD-1 form) present — key-side, any code")
    assert v == by_coll(lambda q: lc[q] == "stated")
    add(B4, "statement in the form permitted by the instruction", v, "summaries", "automatic",
        "one further summary used other phrasing")
    lcode = Counter(s["code"] for s in lsent)
    assert lcode == {"supported": 43, "derivable": 27, "unsupported": 1, "uncertain": 1}
    for c in ("supported", "derivable", "unsupported", "uncertain"):
        vals = [(lcode[c], 72)] + [(sum(1 for s in lsent if s["cohort"] == k and s["code"] == c),
                                    sum(1 for s in lsent if s["cohort"] == k)) for k in COLL]
        add(B4, f"statement coded {c}", vals, "statements", "model rater",
            "of the 71 statements in the permitted form: 43 supported, 26 derivable, 1 unsupported, 1 uncertain; the "
            "statement in other phrasing was coded derivable" if c == "supported" else "")
    cat = {q: lead[q]["category"] for q in coll_of}
    stated = lambda q: lc[q] != "none"  # noqa: E731
    for label, key in (("names the reference group only", "reference group only"),
                       ("names other group(s) only", "other group(s) only")):
        vals = [(sum(1 for q in coll_of if cat[q] == key), 72)] + [
            (sum(1 for q in coll_of if coll_of[q] == k and cat[q] == key), sum(1 for q in coll_of if coll_of[q] == k and stated(q)))
            for k in COLL]
        add(B4, f"statement {label}", vals, "statements", "automatic (group aliases) and reference label",
            "in 5 of the 18 the reference group was absent from the input" if key.startswith("other") else "")
    assert [r[2] for r in R[-2:]] == [54, 18]
    absent = sum(1 for q in coll_of if cat[q] == "other group(s) only" and lead[q]["reference_in_input"] == "False")
    assert absent == 5
    top = by_coll(lambda q: stated(q) and lead[q]["named_equals_top_probability"] == "True")
    add(B4, "statement names the highest-probability group",
        [(top[0][0], 72)] + [(top[i][0], sum(1 for q in coll_of if coll_of[q] == k and stated(q))) for i, k in
                             ((1, "CCDI-MCI"), (2, "STV"))], "statements", "automatic")
    assert R[-1][2] == 68

    B5 = "Post hoc sensitivity (defined after the ratings were read): the word \"primary\""
    rep = SOURCES["report"].read_text(encoding="utf-8")
    for line in POST_HOC_REPORT_LINES:
        assert line in rep, line
    for name, a, c1, c2, unit in POST_HOC:
        add(B5, name, [a, c1, c2], unit, "model rater; post hoc",
            "post hoc; transcribed from the endpoint report (the sentences whose only stated defect was the word "
            "\"primary\" for the query site are identified in the rater's notes, not in a structured field)")

    # ---------------------------------------------------------------- per-summary rows
    fl_keys = list(FLAG_LABEL)
    hdr2 = (["case", "collection", "delivered sentences"] + [f"{c} (n)" for c in CODES] +
            ["every sentence supported or derivable", "automatic flags, raw (n)"] +
            [f"flags: {FLAG_LABEL[k]} (n)" for k in fl_keys] +
            ["automatic flag remaining after the checker-artifact classes", "leading-consideration statement present"])
    R2 = []
    for q in sorted(coll_of):
        R2.append([q, coll_of[q], nsent[q]] + [per[q][c] for c in CODES] +
                  ["yes" if q in all_sd else "no", sum(fl[q].values())] + [fl[q][k] for k in fl_keys] +
                  ["yes" if q in remaining else "no", "yes" if lc[q] != "none" else "no"])
    assert sum(r[2] for r in R2) == 480 and sum(r[10] for r in R2) == 42
    bindings = {k: {"path": str(p.relative_to(OUT)), "sha256": sha(p)} for k, p in SOURCES.items()}
    return (hdr, R), (hdr2, R2), bindings


if __name__ == "__main__":
    (h, r), (h2, r2), b = build()
    for x in [h] + r:
        print(x)
    print(h2)
    for x in r2[:3]:
        print(x)
    print(json.dumps(b, indent=1))
