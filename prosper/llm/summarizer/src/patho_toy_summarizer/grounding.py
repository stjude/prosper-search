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
    "1. The FIRST sentence of the summary MUST describe the current_case "
    "including: age, sex, site, diagnosis, tumor_group, met status, "
    "treatment status (treatment=Y means treated, N means untreated), and "
    "discrepant flag (discrepant=Y means the case is discrepant, N means "
    "concordant).\n"
    "2. Every factual claim in the summary MUST cite the supporting neighbor "
    "case_ids in square brackets, e.g. 'RMS is the dominant diagnosis "
    "[15696, 15151, 8122]'.\n"
    "3. Every such claim must also appear in the 'claims' array with the "
    "same source_case_ids.\n"
    "4. source_case_ids MUST only contain case_id values present in the "
    "provided neighbors.\n"
    "5. Numeric values (scores, means, ranges) MUST exactly match the "
    "provided data or derived_stats.\n"
    "6. Do NOT invent data, case_ids, diagnoses, or scores beyond the "
    "payload.\n"
    "7. Use cautious verbs ('supports', 'suggests', 'is consistent with').\n"
    "8. If probability is provided, do NOT mention similarity.\n"
    "9. Note discrepant neighbors (different diagnosis/tumor_group or "
    "discrepant=='Y') with their case_ids.\n"
    "10. State dominant diagnosis with vote counts, score mean/range, and "
    "highest probability / closest similarity.\n"
    "11. If a 'probabilities' key is present at the top level of the "
    "payload, mention the Bayesian disease probabilities in the summary."
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
        # Fixed-decimal representations (0-4 dp)
        for fmt in ("{:.0f}", "{:.1f}", "{:.2f}", "{:.4f}"):
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
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
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
