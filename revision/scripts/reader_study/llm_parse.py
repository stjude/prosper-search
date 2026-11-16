"""Tolerant parser for the Step 3 LLM summaries (plan 002 §3.8 as amended by 003 §B(4)).

The summaries state the neighbour-label scores in many phrasings ("Bayesian disease probabilities
are G p, ...", "favor G (p)", "G at p", "P(G)=p", "assigns p probability to G", "probability is p
for G", percentages, full disease names). Every parse is hand-checked against the text
(handoff step 3); this module only has to make that check easy and the rules explicit.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Group mentions. Codes are case-sensitive; disease names are case-insensitive. Longer forms come
# first so "ACT/ACC" is not read as "ACT", "Ewing sarcoma" not as a bare word, etc.
_CODE_PATTERNS: List[Tuple[str, str]] = [
    (r"ACT/ACC", "ACT/ACC"), (r"ACT", "ACT/ACC"), (r"ACC", "ACT/ACC"),
    (r"DSRCT", "DSRCT"), (r"EWS", "EWS"), (r"LIVER", "LIVER"), (r"HB", "LIVER"), (r"HCC", "LIVER"),
    (r"MEL", "MEL"), (r"MPNST", "MPNST"), (r"MRT", "MRT"), (r"NB", "NB"), (r"NF", "NF"),
    (r"OS", "OS"), (r"RB", "RB"), (r"RMS", "RMS"), (r"SS", "SS"), (r"WT", "WT"),
]
_NAME_PATTERNS: List[Tuple[str, str]] = [
    (r"adrenocortical (?:carcinoma|tumou?r|neoplasm)s?", "ACT/ACC"),
    (r"desmoplastic small round cell tumou?rs?", "DSRCT"),
    (r"ewing(?:'s)? sarcomas?", "EWS"),
    (r"hepatoblastomas?", "LIVER"), (r"hepatocellular carcinomas?", "LIVER"), (r"liver", "LIVER"),
    (r"melanomas?", "MEL"),
    (r"malignant peripheral nerve sheath tumou?rs?", "MPNST"),
    (r"(?:malignant )?rhabdoid tumou?rs?", "MRT"),
    (r"ganglioneuroblastomas?", "NB"), (r"neuroblastomas?", "NB"),
    (r"neurofibromas?", "NF"),
    (r"osteosarcomas?", "OS"),
    (r"retinoblastomas?", "RB"),
    (r"rhabdomyosarcomas?", "RMS"),
    (r"synovial sarcomas?", "SS"),
    (r"wilms(?:'s?)? tumou?rs?", "WT"),
]
_GROUP_RE = re.compile(
    "|".join([rf"(?P<c{i}>(?<![A-Za-z]){p}(?![A-Za-z]))" for i, (p, _) in enumerate(_CODE_PATTERNS)]
             + [rf"(?P<n{i}>(?i:(?<![A-Za-z]){p}(?![A-Za-z])))" for i, (p, _) in enumerate(_NAME_PATTERNS)])
)

# A score: a decimal (0.427, 1.0, 1.000) or a percentage (30.2%). Integers (vote counts) and
# hyphenated quantities ("0.2-year-old") are not scores.
_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?%|\d*\.\d+)(?![\w%-]|\.\d)")
_PROB_SENT_RE = re.compile(r"probabilit|Bayesian|\bP\([^)]*\)\s*=", re.IGNORECASE)
_WORDING_PROB_RE = re.compile(r"probabilit|\bP\([^)]+\)\s*=", re.IGNORECASE)
_RIGHT_CONNECTOR_RE = re.compile(r"^\s*(?:probability\s+)?(?:to|for)\s+(?:the\s+)?$", re.IGNORECASE)
# Literal 002 §3.8 terms.
DISCREPANCY_TERMS_RE = re.compile(r"\b(discrepant|split|uncertain)\b", re.IGNORECASE)
_LEADING_RE = re.compile(r"leading considerations?", re.IGNORECASE)
_LEAD_VERB_RE = re.compile(r"\b(?:support(?:s|ed|ing)?|favou?r(?:s|ed|ing)?|consider(?:s|ing)?)\b",
                           re.IGNORECASE)
_SPAN_STOP_RE = re.compile(r"\bwhile\b|,\s*with\b|\bwith\b|;|,\s*but\b|\bbut\b", re.IGNORECASE)


def _group_of(m: re.Match) -> str:
    name = m.lastgroup
    idx = int(name[1:])
    return (_CODE_PATTERNS if name[0] == "c" else _NAME_PATTERNS)[idx][1]


def group_mentions(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), _group_of(m)) for m in _GROUP_RE.finditer(text)]


def groups_in(text: str) -> List[str]:
    out: List[str] = []
    for _, _, g in group_mentions(text):
        if g not in out:
            out.append(g)
    return out


def split_sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])", text.strip()) if s]


def _value(tok: str) -> Optional[float]:
    if tok.endswith("%"):
        v = round(float(tok[:-1]) / 100.0, 6)
    else:
        v = float(tok)
    return v if 0.0 <= v <= 1.0 else None


def parse_scores(text: str) -> Tuple[Dict[str, float], List[str]]:
    """(group -> score) from every sentence with probability/Bayesian wording; the first value
    stated for a group is kept. A number is attached to the group that follows it when joined by
    "(probability) to/for (the)", else to the nearest preceding group in the sentence, else to the
    next group in the sentence. Returns the scores and a list of numbers that could not be placed."""
    scores: Dict[str, float] = {}
    unplaced: List[str] = []
    for sent in split_sentences(text):
        if not _PROB_SENT_RE.search(sent):
            continue
        mentions = group_mentions(sent)
        for m in _NUM_RE.finditer(sent):
            v = _value(m.group(1))
            if v is None:
                continue
            after = [g for g in mentions if g[0] >= m.end()]
            before = [g for g in mentions if g[1] <= m.start()]
            group = None
            if after and _RIGHT_CONNECTOR_RE.match(sent[m.end():after[0][0]]):
                group = after[0][2]
            elif before:
                group = before[-1][2]
            elif after:
                group = after[0][2]
            if group is None:
                unplaced.append(m.group(1))
            elif group not in scores:
                scores[group] = v
    return scores, unplaced


def leading_groups(text: str) -> Tuple[List[str], str, str]:
    """Groups the summary names as its leading consideration(s): (groups, source, span).

    1. The last sentence containing "leading consideration(s)": the groups named between the last
       support/favour/consider verb before the phrase and "as (a|the)"; if there is no such verb
       ("X is the leading consideration"), the groups in the clause before "is/are".
    2. Otherwise the last sentence beginning "Overall": the groups named after its last
       support/favour/consider verb, up to "while"/"with"/"but"/";".
    3. Otherwise none (source "none")."""
    sents = split_sentences(text)
    for s in reversed(sents):
        hits = list(_LEADING_RE.finditer(s))
        if not hits:
            continue
        clause = s[:hits[-1].start()].split(";")[-1]
        verbs = list(_LEAD_VERB_RE.finditer(clause))
        if verbs:
            span = clause[verbs[-1].end():]
            span = re.sub(r"\bas\s+(?:a\s+|the\s+)?$", "", span.rstrip() + " ").strip()
            return groups_in(span), "leading_sentence", span
        m = re.search(r"([^,]*?)\b(?:is|are)\s+(?:a\s+|the\s+)?$", clause)
        span = m.group(1).strip() if m else clause.strip()
        return groups_in(span), "leading_sentence", span
    for s in reversed(sents):
        if not re.match(r"overall\b", s, re.IGNORECASE):
            continue
        verbs = list(_LEAD_VERB_RE.finditer(s))
        if not verbs:
            break
        rest = s[verbs[-1].end():]
        stop = _SPAN_STOP_RE.search(rest)
        span = (rest[:stop.start()] if stop else rest).strip()
        return groups_in(span), "overall_sentence", span
    return [], "none", ""


def parse_summary(text: str) -> Dict[str, object]:
    scores, unplaced = parse_scores(text)
    top: Optional[str] = None
    tie = False
    if scores:
        best = max(scores.values())
        leaders = sorted(g for g, v in scores.items() if v == best)
        tie = len(leaders) > 1
        top = "|".join(leaders)
    lead, lead_source, lead_span = leading_groups(text)
    disc = [m.group(1).lower() for m in DISCREPANCY_TERMS_RE.finditer(text)]
    return {
        "scores": scores,
        "unplaced_numbers": unplaced,
        "score_sum": round(sum(scores.values()), 4) if scores else None,
        "llm_top": top,
        "llm_top_tie": tie,
        "leading": lead,
        "leading_source": lead_source,
        "leading_span": lead_span,
        "discrepancy_terms": disc,
        "llm_wording_probability": bool(_WORDING_PROB_RE.search(text)),
    }


def llm_case_vars(parsed: Dict[str, object], truth: str) -> Dict[str, object]:
    """Case-level LLM variables. llm_truth_prob is 0.0 when the truth group is not listed (the
    listed scores sum to ~1, so an unlisted group was effectively given none)."""
    scores: Dict[str, float] = parsed["scores"]  # type: ignore[assignment]
    top = parsed["llm_top"]
    return {
        "llm_top": top,
        "llm_top_correct": bool(top == truth and not parsed["llm_top_tie"]),
        "llm_truth_prob": scores.get(truth, 0.0) if scores else None,
        "llm_names_truth_leading": truth in parsed["leading"],  # type: ignore[operator]
        "llm_flags_discrepancy": bool(parsed["discrepancy_terms"]),
        "llm_wording_probability": parsed["llm_wording_probability"],
    }
