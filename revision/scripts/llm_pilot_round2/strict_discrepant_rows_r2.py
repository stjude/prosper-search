"""S5c strict-"discrepant" rows (lead_adjudication_r2_v1.json, `strict_discrepant_rule`), computed mechanically.

Rule (pre-specified by the analyst): any delivered sentence that calls a specific neighbour "discrepant" whose record
has discrepant != "Y" is recoded contradicted under the strict reading (as round-1 strict12).

Mechanics (no judgement; every sentence containing "discrepan" is listed in exactly one bucket):
1. Rows: the packet's `rating_form.csv` (rater-facing, unrated copy); delivered rows are those whose sentence_no
   is a number (first-draft rows F1.. are reported separately, never in the output list).
2. Each sentence is cut into clauses at ';' and at ', while'/', whereas'/', but'. Only clauses containing
   "discrepan" are read.
3. A clause is **negated** if it contains no/not/none/neither/nor/without before "discrepan", or "concordant", or
   states a flag value N ("discrepant=N", "discrepant flag N", "flag of N"). Negated clauses call no neighbour
   discrepant (bucket `negated`).
4. Neighbours referenced by the clause: the case_ids written in it (bracketed or bare); if none, the neighbours
   whose diagnosis or tumor_group (or a harness alias of the group) occurs in the clause as a word.
5. Bucket: no neighbour resolved -> `ambiguous`; every resolved neighbour has discrepant == "Y" -> `flag_Y`
   (correct use, not listed); every resolved neighbour has discrepant != "Y" -> `strict` (listed); mixed ->
   `ambiguous`. A sentence is a strict row if any of its clauses is `strict` and none is `ambiguous`; a sentence
   with an ambiguous clause goes to the ambiguous list for the analyst whatever its other clauses say.
Outputs (--out-dir): strict_rows.json ([[rating_id, sentence_no], ...]), strict_rows.csv (one row per resolved
neighbour of every clause, with bucket, matched neighbour, its flag and the match basis), ambiguous.csv, summary.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.machinery
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_aliases(harness):
    ld = importlib.machinery.SourceFileLoader("s5c_harness_ro", str(harness))
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(ld.name, ld))
    ld.exec_module(m)
    return m.ALIASES


CLAUSE_SPLIT = re.compile(r";|,\s+(?:while|whereas|but)\b", re.I)
NEG = re.compile(r"\b(no|not|none|neither|nor|without)\b[^;]*?discrepan|concordant|"
                 r"discrepant\s*(?:=|==|:)\s*['\"]?N\b|discrepant\s+flag\s+(?:of\s+)?['\"]?N\b|flag\s+(?:of\s+)?['\"]?N\b",
                 re.I)
DISC = re.compile(r"discrepan", re.I)


def word_re(term):
    flags = 0 if re.fullmatch(r"[A-Z0-9/]+", term) else re.I
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?:s|es)?(?![A-Za-z0-9])", flags)


def resolve(clause, neighbors, aliases):
    by_id = [n for n in neighbors if str(n.get("case_id")) and str(n["case_id"]) in clause]
    if by_id:
        return by_id, "case_id"
    hit = []
    for n in neighbors:
        terms = {str(n.get("diagnosis") or ""), str(n.get("tumor_group") or "")} - {""}
        g = n.get("tumor_group")
        terms |= set(aliases.get(g, [])) if g else set()
        if any(word_re(t).search(clause) for t in terms):
            hit.append(n)
    return hit, "label"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--form", type=Path, required=True, help="packet rating_form.csv (unrated)")
    ap.add_argument("--rating-map", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--harness", type=Path, default=HERE / "llm_grounding_cli.executed_fc88a2e7.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    aliases = load_aliases(a.harness)
    rmap = {r["rating_id"]: r for r in csv.DictReader(open(a.rating_map))}
    runs = {}
    for l in open(a.run_dir / "runs.jsonl"):
        r = json.loads(l)
        if r["selected"]:
            runs[r["run_uid"]] = r
    out_rows, detail, ambiguous, first_draft_strict = [], [], [], []
    n_disc = 0
    for row in csv.DictReader(open(a.form, encoding="utf-8")):
        s = row["sentence_text"]
        if not DISC.search(s):
            continue
        n_disc += 1
        uid = rmap[row["rating_id"]]["run_uid"]
        nb = json.load(open(a.run_dir / runs[uid]["payload_file"]))["neighbors"]
        buckets = []
        for clause in CLAUSE_SPLIT.split(s):
            if not DISC.search(clause):
                continue
            if NEG.search(clause):
                buckets.append("negated")
                detail.append({**row, "run_uid": uid, "clause": clause.strip(), "bucket": "negated",
                               "neighbour": "", "flag": "", "basis": ""})
                continue
            hit, basis = resolve(clause, nb, aliases)
            flags = {str(n.get("discrepant")).upper() == "Y" for n in hit}
            b = "ambiguous" if not hit or len(flags) > 1 else ("flag_Y" if flags == {True} else "strict")
            buckets.append(b)
            for n in hit or [{}]:
                detail.append({**row, "run_uid": uid, "clause": clause.strip(), "bucket": b,
                               "neighbour": n.get("case_id", ""), "flag": n.get("discrepant", ""), "basis": basis if hit else ""})
        delivered = row["sentence_no"].isdigit()
        if "ambiguous" in buckets:
            ambiguous.append({"rating_id": row["rating_id"], "sentence_no": row["sentence_no"], "run_uid": uid,
                              "delivered": delivered, "buckets": "/".join(buckets)})
        elif "strict" in buckets:
            (out_rows if delivered else first_draft_strict).append([row["rating_id"], row["sentence_no"]])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "strict_rows.json").write_text(json.dumps(out_rows) + "\n")
    cols = ["rating_id", "sentence_no", "run_uid", "bucket", "neighbour", "flag", "basis", "clause", "sentence_text"]
    with open(a.out_dir / "strict_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(detail)
    with open(a.out_dir / "ambiguous.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rating_id", "sentence_no", "run_uid", "delivered", "buckets"])
        w.writeheader()
        w.writerows(ambiguous)
    summary = {"script_sha256": sha(__file__), "form_sha256": sha(a.form), "rating_map_sha256": sha(a.rating_map),
               "runs_sha256": sha(a.run_dir / "runs.jsonl"), "sentences_with_discrepant": n_disc,
               "strict_delivered_rows": len(out_rows), "strict_first_draft_rows_not_listed": first_draft_strict,
               "ambiguous_sentences": len(ambiguous)}
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
