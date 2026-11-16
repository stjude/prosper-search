"""Grounding, provenance verification, and safety guardrails.

Provides:
- Medical disclaimer text
- Structured citation prompt for LLM
- Post-hoc claim verification against source data
- Provenance guard ensuring no fabricated values
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Medical disclaimer – appended to every output (deterministic or LLM)
# ---------------------------------------------------------------------------

MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This AI-generated summary is intended to support, not "
    "replace, clinical judgment. All findings should be independently "
    "verified by a qualified pathologist before any clinical decision-making."
)


# ---------------------------------------------------------------------------
# Grounded system prompt – requests structured JSON with inline citations
# ---------------------------------------------------------------------------

GROUNDED_SYSTEM_PROMPT_NEIGHBORS = (
    "You are given a pathology neighbor-set payload containing: current_case "
    "demographics (age, sex, site, tumor_group, diagnosis, met, treatment, "
    "discrepant), and neighbors each with similarity score (lower is better) "
    "or probability (higher is better), plus the same metadata fields.\n\n"
    "Produce a JSON response with this EXACT schema:\n"
    "{\n"
    '  "summary": "<clinician-friendly narrative, ≤200 words, with inline '
    'case-id citations>",\n'
    '  "claims": [\n'
    '    {\n'
    '      "statement": "<one factual claim>",\n'
    '      "source_case_ids": ["<case_id>", ...],\n'
    '      "field": "<data field referenced, e.g. diagnosis, similarity, '
    'tumor_group>",\n'
    '      "value": "<the value being claimed>"\n'
    '    }\n'
    '  ]\n'
    "}\n\n"
    "Rules:\n"
    "1. The FIRST sentence of the summary describes the current_case using "
    "the fields that are present: age, sex, site, diagnosis, tumor_group, met "
    "status, treatment status (treatment=Y means treated, N means untreated), "
    "and discrepant flag (discrepant=Y means the case is discrepant, N means "
    "concordant). Omit fields that are null instead of listing them as 'not "
    "reported'. NEVER put the current case's own id in square brackets.\n"
    "2. A citation is a square-bracketed, comma-separated list of neighbor "
    "case_ids, e.g. [SJEXAMPLE001_D, SJEXAMPLE002_D]. Place each citation IMMEDIATELY "
    "after the phrase it supports, never at the end of a sentence that "
    "contains facts about different neighbors. When you state a count, the "
    "bracket lists exactly those neighbors: 'four are RMS [A, B, C, E] and one "
    "is NB [D]' -- NOT 'four RMS and one NB [A, B, C, D, E]'.\n"
    "3. Do NOT cite set-level values: mean, range, minimum-maximum, vote "
    "totals, discrepant count, or statements true of every neighbor ('no "
    "discrepant neighbors', 'all five are pediatric'). Cite one neighbor when a "
    "value belongs to it: 'the closest match is 0.685 [A]'.\n"
    "4. Cite a group once. After a group has been introduced with its ids, "
    "refer to it by its label ('the four RMS neighbors') without brackets. "
    "Never repeat the same bracket list.\n"
    "5. Interpretive or concluding sentences ('Overall ...') carry no "
    "citations.\n"
    "6. One syntax: comma-separated ids inside ONE pair of brackets per claim, "
    "ids in neighbor rank order. Never write ][ chains, never put ids in "
    "parentheses, never write a neighbor id outside brackets.\n"
    "7. Enumerations cite per item: 'femur [A, C], humerus [D], lung "
    "metastasis [B]'.\n"
    "8. Every citation must also appear in the 'claims' array with the same "
    "source_case_ids; source_case_ids MUST only contain case_id values present "
    "in the provided neighbors.\n"
    "9. Numeric values (scores, means, ranges) MUST match the provided data or "
    "derived_stats; write scores with 3 decimals (e.g. 0.705).\n"
    "10. Do NOT invent data, case_ids, diagnoses, or scores beyond the "
    "payload.\n"
    "11. Use cautious verbs ('supports', 'suggests', 'is consistent with').\n"
    "12. If probability is provided, do NOT mention similarity.\n"
    "13. A neighbor whose diagnosis or tumor_group differs from the most "
    "frequent one among the neighbors is a 'differing' neighbor; note "
    "differing neighbors with their case_ids. Use the word 'discrepant' only "
    "for a neighbor whose record carries discrepant=='Y' (note those with "
    "their case_ids too); never call a neighbor discrepant because its "
    "diagnosis or tumor_group differs.\n"
    "14. State the dominant diagnosis with vote counts, score mean/range, and "
    "highest probability / closest similarity.\n"
    "15. If a 'probabilities' key is present at the top level of the "
    "payload, mention the Bayesian tumor-group probabilities in the summary.\n"
    "16. Every neighbor appears in exactly one bracket of the diagnosis "
    "partition. A neighbor whose diagnosis is null is grouped by its "
    "tumor_group ('one is OS by tumor group with no recorded diagnosis [E]'); "
    "never silently drop a neighbor.\n"
    "17. If the current case's site or age is present and matches none of "
    "the neighbors, say so in one uncited sentence (e.g. 'The adrenal primary "
    "site matches none of the neighbors, which are bone and lung.'); this "
    "caveat is part of the summary, not optional.\n"
    "18. Describe the neighbors and their probabilities; do not state or "
    "imply what the current case's diagnosis is (no 'leading consideration', "
    "'supports X as the diagnosis' or similar conclusions about the current "
    "case).\n\n"
    "Example of the required citation style (five neighbors with ids A-E in "
    "rank order): 'Four of the five neighbors are RMS [A, B, C, E] and one is "
    "NB [D]. Similarity is tight (0.705-0.743, mean 0.732); the closest match "
    "is an RMS lymph-node metastasis [A]. All five neighbors are pediatric "
    "males. RMS sites: lymph node [A], testis [B], hand [C], nasopharynx [E]; "
    "the NB neighbor is a retroperitoneal primary [D]. The NB case is the only "
    "differing neighbor. Overall the neighbor evidence supports an "
    "RMS-predominant pattern.'\n"
    "Wrong (do not do this): 'The dominant diagnosis is RMS (4 votes) with one "
    "NB case [A, B, C, D, E].'"
)

# Instruction carried by the single style-revision call (see llm.py)
STYLE_REVISION_INSTRUCTION = (
    "Your previous response ('previous_response') violates the citation-"
    "placement rules listed under 'style_issues'. Return the corrected JSON "
    "with the SAME schema. Fix every listed issue. Keep every fact, case_id "
    "and number unchanged; only move, merge or remove citations and rephrase "
    "as needed."
)


# ---------------------------------------------------------------------------
# Reference value extraction
# ---------------------------------------------------------------------------

def extract_reference_values(
    payload: Dict[str, Any],
    derived_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract all valid reference values from *payload* for verification.

    Returns a dict with sets of valid case_ids, diagnoses, tumor_groups,
    and numeric values that are allowed to appear in LLM output.
    """
    neighbors: List[Dict[str, Any]] = payload.get("neighbors", [])

    valid_case_ids: Set[str] = {
        str(n["case_id"]) for n in neighbors if n.get("case_id") is not None
    }
    valid_diagnoses: Set[str] = {
        n["diagnosis"] for n in neighbors if n.get("diagnosis")
    }
    valid_tumor_groups: Set[str] = {
        n["tumor_group"] for n in neighbors if n.get("tumor_group")
    }

    # Collect every numeric value in the neighbour records
    numeric_values: Set[float] = set()
    for n in neighbors:
        for key in ("similarity", "probability"):
            v = n.get(key)
            if v is not None:
                numeric_values.add(float(v))

    # Current-case values
    current = payload.get("current_case", {})
    if current.get("diagnosis"):
        valid_diagnoses.add(current["diagnosis"])
    if current.get("tumor_group"):
        valid_tumor_groups.add(current["tumor_group"])
    if current.get("slide_id"):
        valid_case_ids.add(str(current["slide_id"]))

    # Top-level Bayesian disease probabilities (rule 15 asks the model to
    # report them): the class names and the values, as fractions and as the
    # percentage form the model may write ("OS 0.81" or "81%").
    probs = payload.get("probabilities")
    if isinstance(probs, dict):
        for cls, p in probs.items():
            if cls:
                valid_tumor_groups.add(str(cls))
            try:
                pf = float(p)
            except (TypeError, ValueError):
                continue
            numeric_values.add(pf)
            numeric_values.add(100.0 * pf)

    # Derived stats (means, ranges, top scores, …)
    if derived_stats:
        for key in (
            "mean_score", "min_score", "max_score", "top_score",
            "mean_similarity", "min_similarity", "max_similarity",
            "mean_probability", "min_probability", "max_probability",
        ):
            v = derived_stats.get(key)
            if v is not None:
                numeric_values.add(float(v))

    return {
        "case_ids": valid_case_ids,
        "diagnoses": valid_diagnoses,
        "tumor_groups": valid_tumor_groups,
        "numeric_values": numeric_values,
    }


# ---------------------------------------------------------------------------
# Provenance guard – numeric-value verification
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _build_allowed_numerics(reference: Dict[str, Any]) -> Set[str]:
    """Return every plausible string representation of allowed numbers."""
    allowed: Set[str] = set()

    for v in reference.get("numeric_values", set()):
        # Fixed-decimal representations (0-4 dp; 3 dp is the requested style)
        for fmt in ("{:.0f}", "{:.1f}", "{:.2f}", "{:.3f}", "{:.4f}"):
            allowed.add(fmt.format(v))
        # Full-precision representations — covers arbitrary dp like 1729.9012888908446
        allowed.add(str(v))      # e.g. "1729.9012888908446"
        allowed.add(repr(v))     # e.g. "1729.9012888908446"
        # Also allow any sub-decimal truncation the LLM might produce
        s = str(v)
        if "." in s:
            integer_part, frac = s.split(".", 1)
            for length in range(5, len(frac) + 1):
                allowed.add(f"{integer_part}.{frac[:length]}")
        if v == int(v):
            allowed.add(str(int(v)))

    # Case-IDs are valid numbers when they look numeric
    for cid in reference.get("case_ids", set()):
        allowed.add(str(cid))

    # Small integers for vote counts, neighbor counts, etc.
    for i in range(21):
        allowed.add(str(i))

    # Common constants that may appear in boilerplate (e.g. "200 words")
    allowed.update({"200", "25"})

    allowed.discard("")
    return allowed


def provenance_guard(
    text: str,
    reference: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Verify that every number in *text* comes from *reference* data.

    Returns ``(text, violations)`` where *violations* is a list of
    human-readable strings for each unrecognised numeric token.
    The text is returned unchanged; the caller decides how to react.
    """
    allowed = _build_allowed_numerics(reference)
    found = set(_NUM_RE.findall(text))
    violations = sorted(
        f"Unverified number in output: {num}"
        for num in found
        if num not in allowed
    )
    return text, violations


# ---------------------------------------------------------------------------
# Claim-level verification
# ---------------------------------------------------------------------------

def verify_claims(
    claims: List[Dict[str, Any]],
    payload: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Check each claim's ``source_case_ids`` and ``value`` against *payload*.

    Returns ``(verified_claims, warnings)``.
    """
    ref = extract_reference_values(payload)
    valid_case_ids = ref["case_ids"]
    valid_diagnoses = ref["diagnoses"]
    valid_tumor_groups = ref["tumor_groups"]

    # Build lookup:  case_id  →  neighbour record
    neighbor_lookup: Dict[str, Dict[str, Any]] = {}
    for n in payload.get("neighbors", []):
        cid = str(n.get("case_id", ""))
        if cid:
            neighbor_lookup[cid] = n

    verified: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for i, claim in enumerate(claims):
        ok = True
        source_ids = [str(s) for s in claim.get("source_case_ids", [])]

        # 1. Every cited case_id must exist in the payload
        for cid in source_ids:
            if cid not in valid_case_ids:
                warnings.append(f"Claim {i}: fabricated case_id '{cid}'")
                ok = False

        # 2. Field-value consistency
        field = claim.get("field", "")
        value = claim.get("value", "")

        if field == "diagnosis" and isinstance(value, str) and value:
            if value not in valid_diagnoses:
                warnings.append(
                    f"Claim {i}: diagnosis '{value}' not found in payload"
                )
                ok = False
            else:
                # Cross-check: every cited case should carry that diagnosis
                for cid in source_ids:
                    n = neighbor_lookup.get(cid)
                    if n and n.get("diagnosis") and n["diagnosis"] != value:
                        warnings.append(
                            f"Claim {i}: case_id '{cid}' has diagnosis "
                            f"'{n['diagnosis']}', not '{value}'"
                        )
                        ok = False

        if field == "tumor_group" and isinstance(value, str) and value:
            if value not in valid_tumor_groups:
                warnings.append(
                    f"Claim {i}: tumor_group '{value}' not found in payload"
                )
                ok = False

        if ok:
            verified.append(claim)

    return verified, warnings


# ---------------------------------------------------------------------------
# Bracket-cited case-id extraction & validation in free text
# ---------------------------------------------------------------------------

_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_CASE_ID_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def verify_case_ids_in_text(
    text: str,
    valid_case_ids: Set[str],
) -> List[str]:
    """Return fabricated case_ids found in bracket citations like ``[15696, 15151]``."""
    fabricated: List[str] = []
    for match in _BRACKET_RE.finditer(text):
        tokens = [t.strip() for t in match.group(1).split(",")]
        for tok in tokens:
            if tok and _CASE_ID_TOKEN_RE.match(tok) and tok not in valid_case_ids:
                fabricated.append(tok)
    return fabricated


# ---------------------------------------------------------------------------
# Top-level: parse & verify a structured LLM response
# ---------------------------------------------------------------------------

def parse_grounded_response(
    raw: str,
    payload: Dict[str, Any],
    derived_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]], List[str], bool]:
    """Parse a grounded LLM response and run all verification checks.

    Tries the structured ``{summary, claims}`` format first, then falls back
    to the legacy ``{summary}`` format, then to raw text.

    Returns ``(summary_text, verified_claims, all_warnings, is_grounded)``.

    * *summary_text*: the narrative (may contain inline citations), or None.
    * *verified_claims*: claims that passed verification.
    * *all_warnings*: every issue found during verification.
    * *is_grounded*: True when structured claims were present **and**
      at least one passed verification.
    """
    all_warnings: List[str] = []

    # -- 1. Parse JSON -------------------------------------------------------
    data = loads_lenient(raw)
    if data is None:
        text = raw.strip() if raw else None
        return text, [], ["Response was not valid JSON"], False

    if not isinstance(data, dict):
        return str(data).strip(), [], ["Response was not a JSON object"], False

    # -- 2. Extract summary ---------------------------------------------------
    summary: Optional[str] = None
    if isinstance(data.get("summary"), str):
        summary = data["summary"].strip()
    if not summary:
        return None, [], ["No 'summary' field in response"], False

    # -- 3. Extract & verify claims -------------------------------------------
    claims_raw = data.get("claims", [])
    if not isinstance(claims_raw, list):
        claims_raw = []
        all_warnings.append("'claims' field was not a list")

    is_grounded = len(claims_raw) > 0

    if claims_raw:
        verified_claims, claim_warnings = verify_claims(claims_raw, payload)
        all_warnings.extend(claim_warnings)
    else:
        verified_claims = []
        all_warnings.append("No claims provided in response")

    # -- 4. Verify bracket-cited IDs in the summary text ----------------------
    ref = extract_reference_values(payload, derived_stats)
    fabricated = verify_case_ids_in_text(summary, ref["case_ids"])
    if fabricated:
        all_warnings.append(f"Fabricated case_ids in summary text: {fabricated}")

    # -- 5. Provenance guard on numeric values --------------------------------
    _, num_violations = provenance_guard(summary, ref)
    all_warnings.extend(num_violations)

    return summary, verified_claims, all_warnings, is_grounded


# ---------------------------------------------------------------------------
# Lenient JSON loading (reasoning models cannot be forced into JSON mode and
# occasionally wrap the object in ``` fences or prose)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def loads_lenient(raw: Any) -> Optional[Any]:
    """Parse *raw* as JSON, tolerating code fences and surrounding prose.

    Returns the parsed value or None (never raises).
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except (json.JSONDecodeError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Citation style: normaliser (syntax, deterministic) + verifier (placement)
# ---------------------------------------------------------------------------
#
# Style rules (revision workspace, [internal record]
# section 2): attach citations to the phrase they support, cite exact
# partitions, never cite the query, do not cite set-level values or
# conclusions, cite a group once, one bracket syntax in rank order, only
# neighbour ids. The normaliser fixes what is purely syntactic; the verifier
# reports what only the model can fix, and its output feeds the single
# revision call. Style warnings are SEPARATE from grounding warnings and
# never trigger the deterministic fallback.

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_AGG_RE = re.compile(
    r"\b(mean|average|averag\w+|range[sd]?|rang\w+|span(?:s|ned)?|minimum|maximum)\b", re.I)
_SETFACT_RE = re.compile(
    r"\b(no discrepant|discrepant[_ ]count|none (?:of (?:the )?neighbou?rs )?(?:is|are|were) "
    r"(?:flagged )?discrepant|no neighbou?rs? (?:is|are|were) (?:flagged )?discrepant)\b", re.I)
_CONCL_RE = re.compile(r"^\s*(overall|taken together|in summary|in conclusion|altogether)\b", re.I)
_NUMWORD_RE = re.compile(
    r"(?<![\w./])(one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})"
    r"(?![\w.]|\s*(?:yr|yrs|year|mo|month|percent|%|/|of\b|-|votes?\)))", re.I)
_WORD2N = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _query_ids(payload: Dict[str, Any]) -> Set[str]:
    cur = payload.get("current_case", {}) or {}
    return {str(cur[k]) for k in ("comp_bio_id", "slide_id", "case_id") if cur.get(k)}


def _neighbor_ids(payload: Dict[str, Any]) -> List[str]:
    return [str(n.get("case_id")) for n in payload.get("neighbors", []) if n.get("case_id") is not None]


def _split_group(group: str) -> List[str]:
    return [t for t in re.split(r"[,;\s]+", group.strip()) if t]


def normalize_citations(text: str, payload: Dict[str, Any]) -> str:
    """Deterministic syntax pass: merge ``][`` chains, comma+space separators,
    ids in neighbour rank order, the query's own id removed, duplicates
    collapsed, empty brackets dropped. Never moves or deletes an informative
    citation."""
    if not text:
        return text
    nb = _neighbor_ids(payload)
    rank = {cid: i for i, cid in enumerate(nb)}
    qids = _query_ids(payload)
    out = re.sub(r"\]\s*\[", ", ", text)

    def _fix(m: "re.Match[str]") -> str:
        seen: Set[str] = set()
        toks: List[str] = []
        for t in _split_group(m.group(1)):
            if t in qids or t in seen:
                continue
            seen.add(t)
            toks.append(t)
        toks.sort(key=lambda t: rank.get(t, len(rank)))
        return "[" + ", ".join(toks) + "]" if toks else ""

    out = _BRACKET_RE.sub(_fix, out)
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def verify_citation_style(text: str, payload: Dict[str, Any]) -> List[str]:
    """Return human-readable style warnings for *text* (placement rules).

    Checks: query self-citation; second full-set citation; repeated
    multi-id group; numeric / non-neighbour token in a bracket; ``][`` chains
    and ``;`` separators; neighbour id outside brackets; a count followed by a
    bracket of a different size; a bracket on a set-level statistic sentence;
    a bracket on a 'no discrepant' sentence; a bracket on a concluding
    sentence.
    """
    if not text:
        return []
    warnings: List[str] = []
    nb = _neighbor_ids(payload)
    full = set(nb)
    qids = _query_ids(payload)
    groups = [_split_group(g) for g in _BRACKET_RE.findall(text)]

    if any(any(t in qids for t in g) for g in groups):
        warnings.append("Query self-citation: the current case's own id appears inside brackets; "
                        "name it once in the opening sentence without brackets")
    if len(nb) >= 2:
        n_full = sum(1 for g in groups if set(g) == full)
        if n_full > 1:
            warnings.append(f"The full neighbor set is cited {n_full} times; cite it at most once "
                            "(when introducing the set) and refer to it by label afterwards")
    seen: Dict[tuple, int] = {}
    for g in groups:
        if len(g) >= 2:
            key = tuple(sorted(g))
            seen[key] = seen.get(key, 0) + 1
    for key, cnt in seen.items():
        if cnt > 1 and set(key) != full:
            warnings.append(f"The group [{', '.join(key)}] is cited {cnt} times; cite a group once "
                            "and refer to it by its label afterwards")
    bad = sorted({t for g in groups for t in g if t not in full and t not in qids})
    if bad:
        warnings.append(f"Bracket tokens that are not neighbor ids: {bad}")
    cited = {t for g in groups for t in g}
    never = [cid for cid in nb if cid not in cited]
    if never and len(nb) >= 2:
        warnings.append(f"Neighbors never cited: {never}; every neighbor belongs in exactly one "
                        "bracket of the diagnosis partition (group a neighbor with no diagnosis by "
                        "its tumor_group)")
    numeric = sorted({t for g in groups for t in g if t.isdigit()})
    if numeric:
        warnings.append(f"Numeric scan ids cited {numeric}; cite sample ids (Comp.Bio.ID) only")
    if re.search(r"\]\s*\[", text):
        warnings.append("Adjacent bracket chains ][ ...][ ...]; merge into one comma-separated bracket")
    if any(";" in g for g in _BRACKET_RE.findall(text)):
        warnings.append("Semicolons inside brackets; separate ids with ', '")
    stripped = _BRACKET_RE.sub(" ", text)
    bare = sorted({cid for cid in nb if len(cid) >= 3 and not cid.isdigit()
                   and re.search(r"(?<![\w-])" + re.escape(cid) + r"(?![\w-])", stripped)})
    if bare:
        warnings.append(f"Neighbor ids written outside brackets {bare}; ids belong in brackets only")
    # Count check: the number word closest to a bracket (within the same clause,
    # i.e. since the previous bracket / sentence or clause boundary) must equal
    # the bracket size. Denominators ('four of the five'), ages ('16 yr'),
    # percentages and hyphenated ages are excluded by the regex.
    for m in _BRACKET_RE.finditer(text):
        pre = text[:m.start()]
        cut = max(pre.rfind("]"), pre.rfind(". "), pre.rfind("; "), pre.rfind(": "))
        window = pre[cut + 1:][-90:]
        nums = [x for x in _NUMWORD_RE.finditer(window)
                if not re.search(r"\bof\s+(?:the\s+)?$", window[:x.start()], re.I)]
        if not nums:
            continue
        last = nums[-1]
        n = _WORD2N.get(last.group(1).lower()) or (int(last.group(1)) if last.group(1).isdigit() else None)
        ids = _split_group(m.group(1))
        if n is None or n > max(len(nb), 1) or n == 0:
            continue
        if len(ids) != n:
            snippet = (window[last.start():] + m.group(0))[:90]
            warnings.append(f"Count/citation mismatch: '{snippet}' states {n} but cites "
                            f"{len(ids)} ids; the bracket must list exactly the {n} neighbors meant")
    for sent in _SENT_SPLIT_RE.split(text):
        sg = [_split_group(g) for g in _BRACKET_RE.findall(sent)]
        if not sg:
            continue
        if _AGG_RE.search(sent) and any(len(g) >= 2 for g in sg):
            warnings.append(f"Set-level statistic cited: '{sent[:70]}...' -- means/ranges describe the "
                            "whole set and carry no citation (cite one id only for a single neighbor's value)")
        if _SETFACT_RE.search(sent):
            warnings.append(f"Set-level fact cited: '{sent[:70]}...' -- 'no discrepant neighbors' needs no citation")
        if _CONCL_RE.match(sent):
            warnings.append(f"Concluding sentence cited: '{sent[:70]}...' -- interpretation carries no citation")
    return warnings
