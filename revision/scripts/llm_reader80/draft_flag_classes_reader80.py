"""S1 draft: mechanical classification of the reader-80 automatic flags into the E7-A3(2) classes.

KEY-SIDE. Input: auto_flags/auto_flags_reader80.csv (analyze_reader80.py --flags-only) and the frozen payloads in
inputs/payloads/. No summary prose is read: the only inputs per flag are the flag type, the flagged value and the
payload (round-2 specification §5, "Automatic-flag classification": "from the auditor records (flag type, number,
payload) and not from summary prose").

Rule text applied (s5c_pilot_round2_contract_20261110.md §5, E7-A3(2)): "a flag counts only if it survives the
pre-defined checker-artifact classes (α) correctly copied probability rejected by the number check, (β) number
literally in the payload, (γ) "diagnosis 'X (k of 5)' not found" matcher artifact"; everything else is raw.

Mechanical clauses (each flag records the clause it used):
  beta-literal   grounding_unverified_number whose value occurs as a whole number token in a payload string or
                 equals str() of a payload numeric value (round-2 precedent: "22" vs "22 mo", "21" vs "21 yr").
  alpha          grounding_unverified_number whose value equals a payload probability rounded to the value's own
                 decimals (fraction or percent) - "correctly copied probability".
  gamma          grounding_claim_not_found of the form "diagnosis 'X (k of N)' not found" (the round-2 class was
                 written for k of 5; k of N with N = neighbour count is the same matcher artefact - needs_lead to
                 confirm the N = 10 reading).
  raw-default    any other flag type, incl. grounding_other claim warnings without the "(k of N)" suffix and the
                 true-group leak candidate.
needs_lead is set when the class is not mechanically determined: a number that matches a payload value only in a
non-literal form (e.g. "24" vs payload "24.0 yr"), a number matching both alpha and beta or neither, a gamma at
N != 5, and the true-group leak candidate (its "not a leak" ruling is the independent, round-2 true_group_rule).
The output is a DRAFT for the analyst to review and freeze; it is not named v1 and is not an input to any join.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reader80_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--flags", type=Path, default=C.OUT_ROOT / "auto_flags" / "auto_flags_reader80.csv")
ap.add_argument("--out", type=Path, default=C.OUT_ROOT / "lead_adjudication_reader80_DRAFT.json")
a = ap.parse_args()
if a.out.exists():
    sys.exit(f"refusing to overwrite {a.out}")

ID_FIELDS = {"slide_id", "comp_bio_id", "case_id"}  # identifiers are not numbers in the payload
NUM_TOKEN = r"(?<![\d.]){}(?![\d]|\.\d)"


def walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, x


def nb_path(path, pl):
    m = re.match(r"neighbors\[(\d+)\]\.(.+)", path)
    if m:
        return f"neighbors[case_id={pl['neighbors'][int(m.group(1))]['case_id']}].{m.group(2)}"
    return path


def classify_number(val, pl):
    lit, nonlit, alpha = [], [], []
    for path, v in walk({k: pl[k] for k in ("current_case", "neighbors")}):
        if v is None or isinstance(v, bool) or path.rsplit(".", 1)[-1] in ID_FIELDS:
            continue
        s = str(v)
        if isinstance(v, (int, float)):
            if s == val:
                lit.append(f'{nb_path(path, pl)}={s}')
            elif re.fullmatch(r"\d+(\.\d+)?", val) and abs(float(v) - float(val)) < 1e-9:
                nonlit.append(f'{nb_path(path, pl)}={s}')
        else:
            if re.search(NUM_TOKEN.format(re.escape(val)), s):
                lit.append(f'{nb_path(path, pl)}="{s}"')
            else:
                for m in re.findall(r"\d+(?:\.\d+)?", s):
                    if re.fullmatch(r"\d+(\.\d+)?", val) and abs(float(m) - float(val)) < 1e-9:
                        nonlit.append(f'{nb_path(path, pl)}="{s}"')
    pct = val.endswith("%")
    v0 = val.rstrip("%")
    if re.fullmatch(r"\d+(\.\d+)?", v0):
        dec = len(v0.split(".")[1]) if "." in v0 else 0
        for g, p in (pl.get("probabilities") or {}).items():
            for scale, tag in ((1, "fraction"), (100, "percent")):
                if pct and scale == 1:
                    continue
                if abs(round(p * scale, dec) - float(v0)) < 1e-9 and (dec > 0 or scale == 100):
                    alpha.append(f"probabilities.{g}={p} ({tag}, rounded to {dec} dp)")
    return lit, nonlit, alpha


rows = list(csv.DictReader(open(a.flags)))
out, counts = [], {}
for f in rows:
    pl = json.loads((C.OUT_ROOT / "inputs" / "payloads" / f"{f['slide_id']}_packet.json").read_text())
    rec = {k: f[k] for k in ("run_uid", "source", "flag_type", "value")}
    rec["occurrence"] = int(f["occurrence"])
    rec["query"] = f["query"]
    ft, val = f["flag_type"], f["value"]
    if ft == "grounding_unverified_number":
        lit, nonlit, alpha = classify_number(val, pl)
        if lit and not alpha:
            cls, clause, need, ev = "beta", "beta-literal", False, lit
            note = "number literally present in payload (E7-A3(2) beta)"
        elif alpha and not lit:
            cls, clause, need, ev = "alpha", "alpha", False, alpha
            note = "correctly copied probability rejected by the number check (E7-A3(2) alpha)"
        elif alpha and lit:
            cls, clause, need, ev = "beta", "beta-literal (also alpha-matching)", True, lit + alpha
            note = "matches a literal payload number and a probability; class immaterial to counting (both discount)"
        elif nonlit:
            cls, clause, need, ev = "beta", "beta? non-literal numeric equality", True, nonlit
            note = (f"value '{val}' equals a payload number only numerically (e.g. integer form of a one-decimal "
                    "value); the round-2 beta precedents were literal integers ('22 mo', '21 yr'). Lead to rule "
                    "beta or raw.")
        else:
            cls, clause, need, ev = "raw", "raw-default (no payload match)", True, []
            note = "number not found in payload by literal, numeric or probability match"
        rec.update({"class": cls, "payload_field": "; ".join(ev)})
    elif ft == "grounding_claim_not_found":
        m = re.match(r"(.+) \((\d+) of (\d+)\)$", val)
        n_nb = len(pl["neighbors"])
        need = m is None or int(m.group(3)) != 5
        cls, clause = "gamma", "gamma (diagnosis 'X (k of N)' not found)"
        ev = [f"neighbours k={n_nb}"]
        note = "matcher artefact class gamma" + ("; N != 5 - lead to confirm the k-of-10 reading" if need else "")
        rec.update({"class": cls, "payload_field": "; ".join(ev)})
    elif ft == "grounding_other":
        cls, clause, need = "raw", "raw-default (not alpha/beta/gamma: no '(k of N)' suffix)", False
        lab = sorted(set(C_ for C_ in [str(n.get("tumor_group")) for n in pl["neighbors"]] if C_ != "None"))
        ev = [f"neighbour tumor_group values {lab}", f"probabilities keys {sorted(pl.get('probabilities') or {})}"]
        note = ("raw under E7-A3(2): the warning has no '(k of N)' suffix, so it is outside class gamma (round-2 "
                "precedent: 5674 'raw: no (k of 5) suffix'). The quoted label is a composite/negated form that is "
                "not a payload label value; probable matcher artefact, but the rule keeps it raw.")
        rec.update({"class": cls, "payload_field": "; ".join(ev)})
    elif ft == "true_group_leak_candidate":
        cls, clause, need = "raw", "raw-default (true-group leak candidate is not an alpha-gamma class)", True
        ev = ["key/leaks_true_group_detail.json: reference group absent from neighbour labels and probability keys; "
              "harness mention_test hits 0 (mentioned_any False, leak_candidate False); stored audit "
              "leaks_true_group True from the auditor's plain substring test"]
        note = ("lead to rule under the round-2 true_group_rule whether this is 'not a leak' (substring artefact; "
                "mention_test finds no word-bounded mention). Class raw either way; a not-a-leak ruling goes in "
                "true_group_not_a_leak and the lead may also drop or keep the flag from the zero-flag count.")
        rec.update({"class": cls, "payload_field": "; ".join(ev)})
    else:
        cls, clause, need, ev, note = "raw", "raw-default", True, [], "unexpected flag type"
        rec.update({"class": cls, "payload_field": ""})
    rec.update({"rule_clause": clause, "payload_evidence": ev, "needs_lead": need, "note": note})
    counts[(cls, need)] = counts.get((cls, need), 0) + 1
    out.append(rec)

doc = {
    "schema": "analyze_r2 adjudication v1 (DRAFT; for lead review and freeze)",
    "status": "DRAFT - not frozen; not v1; written by the implementer before any rating form exists",
    "written_by": "implementer (mechanical; never a rater)",
    "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "basis": "E7-A3(2) fixed classes alpha/beta/gamma; everything else raw. Classified from auto_flags_reader80.csv "
             "(flag type + value) and inputs/payloads only; no summary prose read.",
    "inputs_sha256": {str(a.flags): C.fsha(a.flags), "script": C.fsha(Path(__file__))},
    "counts_by_class_needs_lead": {f"{k[0]}|needs_lead={k[1]}": v for k, v in sorted(counts.items())},
    "flag_classes": out,
    "true_group_not_a_leak": [],
    "true_group_not_a_leak_pending_lead": [r["run_uid"] for r in out if r["flag_type"] == "true_group_leak_candidate"],
    "true_group_rule": "As round 2: a lexical true-group mention for an eligible (truth-absent) query is adjudicated "
                       "by the lead from payload labels/probabilities (payload-supported mention is not a leak).",
    "dispositions": [],
    "strict_discrepant_rows": [],
    "strict_discrepant_rule": "Computed mechanically after the forms are hashed by analyze_reader80.py (round-2 rule of "
                              "strict_discrepant_rows_r2.py); ambiguous sentences are listed for the lead.",
    "recode_sensitivities": [],
}
a.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
print(json.dumps(doc["counts_by_class_needs_lead"]))
