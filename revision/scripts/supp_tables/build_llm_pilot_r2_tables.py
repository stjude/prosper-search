#!/usr/bin/env python3
"""Supplementary Table S48 (workspace id S-LLM-PILOT): LLM grounding pilot endpoints, sentence codes and the automatic
checks recorded with the summaries shown to readers (revision, 2026-11-10).

Inputs (read only): the grounding-pilot analysis files under `$PROSPER_OUT/s5c_pilot_round2_20261110/analysis/join/`
(`endpoints.json`, `analysis_work.json`, the primary-rater join `rating_join_r1.csv`, `flag_classification.csv`) and the
`audit` fields of `runs.jsonl` in `$PROSPER_OUT/st1_llm_summaries_v5_ccdi/` and `..._v5_stv/` (the summaries shown to
readers, k = 10) plus the practice flags of `reader_study_suite/data/cases.json` (read only, never modified).
Design decisions (revision): only the corrected-instruction pilot run and only the pre-registered primary rating are
reported. Cell text carries no run or rater label: sources are named by role with a sha256 prefix. Query ids are the
opaque codes sent to the model (CASE_<n>); sentence text is not copied (it carries reference-slide identifiers).

Output: `$PROSPER_OUT/supp_tables_20261110_llm_pilot/Table_S-LLM-PILOT.xlsx` (+ BUILD_RECORD.json).
Usage: /HOME/miniconda3/bin/python3 scripts/supp_tables/build_llm_pilot_r2_tables.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path

import openpyxl
from openpyxl.styles import Font

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(os.environ.get("PROSPER_OUT") or REPO / "prosper_revision_output")
PILOT = OUT_ROOT / "s5c_pilot_round2_20261110"
JOIN = PILOT / "analysis/join"
V5 = {"CCDI-MCI": OUT_ROOT / "st1_llm_summaries_v5_ccdi/runs.jsonl", "St. Jude validation": OUT_ROOT / "st1_llm_summaries_v5_stv/runs.jsonl"}
CASES = OUT_ROOT / "reader_study_suite/data/cases.json"
DST = OUT_ROOT / "supp_tables_20261110_llm_pilot"
BANNED = re.compile(r"(?i)\bR[12]\b|round ?[12]|round2|\br1\b|s5[bc]|a0_masked")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    files = {"endpoints": JOIN / "endpoints.json", "work": JOIN / "analysis_work.json", "join": JOIN / "rating_join_r1.csv",
             "flags": JOIN / "flag_classification.csv", "report": PILOT / "analysis/ENDPOINTS_REPORT_R2.md"}
    h = {k: sha(p)[:8] + "…" for k, p in files.items()}
    role = {"endpoints": f"pilot endpoints file {h['endpoints']}", "work": f"pilot analysis file {h['work']}",
            "join": f"primary-rater join {h['join']}", "flags": f"automatic-flag classification {h['flags']}",
            "report": f"pilot report {h['report']}"}
    ep = json.load(open(files["endpoints"]))
    E = ep["endpoints"]
    work = json.load(open(files["work"]))
    rows = list(csv.DictReader(open(files["join"], encoding="utf-8")))
    flags = list(csv.DictReader(open(files["flags"], encoding="utf-8")))

    def find(name: str, rater: str) -> tuple[int, dict]:
        hits = [(i, e) for i, e in enumerate(E) if e["name"] == name and e["rater"] == rater]
        assert len(hits) == 1, (name, rater, len(hits))
        return hits[0]

    delivered = [r for r in rows if not r["sentence_no"].startswith("F")]
    first = [r for r in rows if r["sentence_no"].startswith("F")]
    assert len(delivered) == 134 and len(first) == 14, (len(delivered), len(first))
    codes: dict[str, int] = {}
    for r in delivered:
        codes[r["code"]] = codes.get(r["code"], 0) + 1
    rated = work["rated"]
    as_rated = rated["R1|as_rated"]["all_sentences_supported_or_derivable"]
    alias = rated["R1|alias_accepted_posthoc"]["all_sentences_supported_or_derivable"]
    cls: dict[str, int] = {}
    for f in flags:
        cls[f["class"]] = cls.get(f["class"], 0) + 1
    raw_after = {f["query"] for f in flags if f["source"] == "final" and f["counts_after_alpha_gamma"] == "True"}

    out = []  # endpoint, numerator, denominator, display, unit, evaluated by, source

    def add(label, num, den, unit, by, src, display=None):
        out.append([label, num, den, display if display is not None else f"{num}/{den}", unit, by, src])

    def epsrc(i, extra=""):
        return f"{role['endpoints']}, entry {i}" + (f" ({extra})" if extra else "")

    i, e = find("rejection / too_long / empty / error output", "auto")
    add("Rejected, too long, empty or error output", e["numerator"], e["denominator"], "summaries", "automatic", epsrc(i))
    add("Style revision applied (delivered text from the revision call)", 2, 20, "summaries", "automatic", epsrc(i, "notes: revisions applied 2"))
    i, e = find("genuine invalid or fabricated citation (summaries with >=1)", "auto")
    add("At least one invalid or fabricated citation", e["numerator"], e["denominator"], "summaries", "automatic", epsrc(i))
    i, e = find("query native ID or patient stem in any output (text, text_first, raw)", "auto")
    add("Query's own identifier or patient code in any output", e["numerator"], e["denominator"], "summaries", "automatic", epsrc(i))
    i, e = find("true group named when absent from neighbours ∪ posterior — after §5 adjudication", "analyst adjudication")
    add("True group named when absent from the neighbors and the probabilities (after adjudication)", e["numerator"], e["denominator"],
        "summaries whose true group was absent", "automatic, adjudicated", epsrc(i))
    i, e = find("probability statements: outputs with >=1 numeric statement / all correct / >=1 incorrect / absent / no evaluable text",
                "auto (parsed + analyst-checked)")
    add("Probability statement present and every stated value correct", 20, 20, "summaries", "automatic, checked", epsrc(i, "with 20/20; all correct 20"))
    i, e = find("automatic claim contradicting record (summaries)", "auto")
    add("Automatic claim check: claim contradicting the cited record", e["numerator"], e["denominator"], "summaries", "automatic", epsrc(i))
    add("Structured claims returned", 103, None, "claims", "automatic", epsrc(i, "notes: 103 claims"), display="103")
    i, e = find("zero automatic flags — raw (E7-A3(2))", "auto")
    add("No automatic flag, raw", e["numerator"], e["denominator"], "summaries", "automatic", epsrc(i))
    i, e = find("zero automatic flags — post alpha/beta/gamma (E7-A3(2))", "auto + listed discounts")
    add("No automatic flag after the pre-defined checker-artifact classes", e["numerator"], e["denominator"], "summaries", "automatic", epsrc(i))
    add("Automatic flag rows (final and first-draft texts)", len(flags), None, "flag rows", "automatic", f"{role['flags']}, rows", display=str(len(flags)))
    for k, lab in (("alpha", "correctly copied probability rejected by the number check"),
                   ("beta", "number literally present in the payload"),
                   ("raw", "not in a pre-defined artifact class")):
        add(f"Flag rows: {lab}", cls.get(k, 0), len(flags), "flag rows", "automatic", f"{role['flags']}, class {k}")
    add("Summaries with a flag after the artifact classes (final text)", len(raw_after), 20, "summaries", "automatic",
        f"{role['flags']}, final-text rows counted after the classes")
    i, e = find("rule-12 descriptive: summaries mentioning similarity/distance", "auto")
    add("Similarity or distance mentioned", e["numerator"], e["denominator"], "summaries", "automatic", epsrc(i))
    add("Delivered sentences rated", len(delivered), None, "sentences", "model rater", f"{role['join']}, delivered rows", display=str(len(delivered)))
    add("First-draft sentences rated (two summaries replaced by a style revision)", len(first), None, "sentences", "model rater",
        f"{role['join']}, first-draft rows", display=str(len(first)))
    for c in ("supported", "derivable", "unsupported", "speculative-diagnostic", "contradicted", "uncertain"):
        add(f"Delivered sentences coded {c}", codes.get(c, 0), len(delivered), "sentences", "model rater", f"{role['join']}, code {c}")
    i, e = find("manual contradicted sentence: summaries with >=1", "R1")
    add("At least one contradicted sentence", e["numerator"], e["denominator"], "summaries", "model rater", epsrc(i))
    i, e = find("manual unsupported or speculative-diagnostic sentences: summaries with >=1", "R1")
    add("At least one unsupported or speculative-diagnostic sentence", e["numerator"], e["denominator"], "summaries", "model rater", epsrc(i))
    i, e = find("manual uncertain / not resolvable sentences (never counted as supported)", "R1")
    unc_q = len(e["notes"].split(",")) if e["numerator"] else 0
    add("Sentences coded uncertain", e["numerator"], e["denominator"], "sentences", "model rater", epsrc(i))
    add("Summaries with an uncertain sentence", unc_q, 20, "summaries", "model rater", epsrc(i, "notes: summaries listed"))
    add("Every delivered sentence supported or derivable, as rated", len(as_rated), 20, "summaries", "model rater",
        f"{role['work']}, all-sentences-supported list as rated")
    add("Every delivered sentence supported or derivable, spelled-out abbreviations read as their labels (post hoc)", len(alias), 20,
        "summaries", "model rater", f"{role['work']}, all-sentences-supported list with abbreviations accepted")
    i, e = find("rated probability-tagged sentences by code", "R1")
    n_prob = sum(json.loads(e["value"]).values())
    add("Sentences stating a probability", n_prob, None, "sentences", "model rater", epsrc(i), display=str(n_prob))

    # summaries shown to readers: automatic checks recorded with each summary (audit fields only)
    practice = {c["slide_id"] for c in json.load(open(CASES)) if c.get("pilot")}
    rs = []
    tot = {k: 0 for k in ("n", "practice", "accepted", "n_citations", "n_bad_citations", "n_numbers", "n_unsupported_numbers",
                          "n_claims", "n_claims_contradicting_record", "leaks_true_group")}
    per = []
    for coh, p in V5.items():
        runs = [json.loads(l) for l in open(p)]
        assert all(r.get("audit") for r in runs) and all(r["k"] == 10 for r in runs), p
        c = {k: 0 for k in tot}
        c["n"] = len(runs)
        c["practice"] = sum(r["query"] in practice for r in runs)
        c["accepted"] = sum(bool(r["accepted"]) for r in runs)
        for k in ("n_citations", "n_bad_citations", "n_numbers", "n_unsupported_numbers", "n_claims", "n_claims_contradicting_record"):
            c[k] = sum(r["audit"][k] for r in runs)
        c["leaks_true_group"] = sum(bool(r["audit"]["leaks_true_group"]) for r in runs)
        per.append((coh, p, c))
        for k in tot:
            tot[k] += c[k]
    labels = [("n", "Summaries (ten neighbors each)"), ("practice", "of which for the two practice cases"), ("accepted", "Accepted by the module"),
              ("n_citations", "Case citations checked"), ("n_bad_citations", "Invalid case citations"), ("n_numbers", "Numbers checked"),
              ("n_unsupported_numbers", "Numbers not found in the input"), ("n_claims", "Structured claims checked"),
              ("n_claims_contradicting_record", "Claims contradicting the cited record"), ("leaks_true_group", "Summaries flagged for naming the query's true group")]
    src_v5 = {coh: f"reader-study summaries file ({coh}) {sha(p)[:8]}…" for coh, p, _ in per}
    for k, lab in labels:
        field = {"n": "rows", "practice": "rows whose case is a practice case in the case list", "accepted": "accepted"}.get(k, f"audit.{k} summed")
        rs.append([lab, per[0][2][k], per[1][2][k], tot[k], f"{src_v5['CCDI-MCI']} and {src_v5['St. Jude validation']}, {field}"])

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "endpoints"
    ws.append(["endpoint", "numerator", "denominator", "display", "unit", "evaluated by", "source"])
    for r in out:
        ws.append(r)
    ws2 = wb.create_sheet("sentence_codes")
    ws2.append(["query (code sent to the model)", "sentence", "text version", "code", "probability statement", "support field (rater)"])
    for r in rows:
        trace = r["support_trace"] or ""
        ws2.append([f"CASE_{r['query']}", r["sentence_no"], "first draft" if r["sentence_no"].startswith("F") else "delivered",
                    r["code"], r["probability_statement_tag"] or "",
                    re.sub(r"SJ[A-Z0-9_]+", "<reference id>", trace.split("=")[0].split(";")[0].strip())])
    ws3 = wb.create_sheet("reader_study_summaries_checks")
    ws3.append(["check", "CCDI-MCI", "St. Jude validation", "total", "source"])
    for r in rs:
        ws3.append(r)
    ws4 = wb.create_sheet("source")
    ws4.append(["file (role)", "sha256"])
    for k, p in files.items():
        ws4.append([role[k].rsplit(" ", 1)[0], sha(p)])
    for coh, p, _ in per:
        ws4.append([f"reader-study summaries file ({coh})", sha(p)])
    ws4.append(["reader-study case list (practice flags)", sha(CASES)])
    bad = []
    for w in wb.worksheets:
        if BANNED.search(w.title):
            bad.append(w.title)
        for c in w[1]:
            c.font = Font(bold=True)
        w.freeze_panes = "A2"
        for row in w.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and BANNED.search(cell.value):
                    bad.append((w.title, cell.coordinate, cell.value[:60]))
    assert not bad, f"run or rater labels in the workbook: {bad[:5]}"
    DST.mkdir(parents=True, exist_ok=True)
    dst = DST / "Table_S-LLM-PILOT.xlsx"
    wb.save(dst)
    rec = {"output": str(dst.relative_to(OUT_ROOT)), "output_sha256": sha(dst), "script": "scripts/supp_tables/build_llm_pilot_r2_tables.py",
           "script_sha256": sha(Path(__file__)), "n_endpoint_rows": len(out), "n_sentence_rows": len(rows),
           "reader_study_summaries": {"per_cohort": {coh: c for coh, _, c in per}, "total": tot}}
    (DST / "BUILD_RECORD.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(rec, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
