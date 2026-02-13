"""Tests for grounding, provenance guard, claim verification, and citation parsing."""
import json

from patho_toy_summarizer.grounding import (
    MEDICAL_DISCLAIMER,
    extract_reference_values,
    parse_grounded_response,
    provenance_guard,
    verify_case_ids_in_text,
    verify_claims,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PAYLOAD = {
    "current_case": {
        "slide_id": "14212",
        "age": "11 yr",
        "sex": "M",
        "site": "Mandible (P)",
        "tumor_group": "RMS",
        "diagnosis": "RMS",
    },
    "neighbors": [
        {"case_id": "15696", "similarity": 1729.90, "diagnosis": "RMS", "tumor_group": "RMS"},
        {"case_id": "15151", "similarity": 1926.46, "diagnosis": "RMS", "tumor_group": "RMS"},
        {"case_id": "7860",  "similarity": 1930.34, "diagnosis": "EWS", "tumor_group": "EWS"},
        {"case_id": "7855",  "similarity": 1972.96, "diagnosis": "EWS", "tumor_group": "EWS"},
        {"case_id": "8122",  "similarity": 1977.38, "diagnosis": "RMS", "tumor_group": "RMS"},
    ],
}


DERIVED_STATS = {
    "mean_score": 1907.41,
    "min_score": 1729.90,
    "max_score": 1977.38,
}


# ---------------------------------------------------------------------------
# MEDICAL_DISCLAIMER
# ---------------------------------------------------------------------------

def test_disclaimer_is_string():
    assert isinstance(MEDICAL_DISCLAIMER, str)
    assert "clinical judgment" in MEDICAL_DISCLAIMER.lower()


# ---------------------------------------------------------------------------
# extract_reference_values
# ---------------------------------------------------------------------------

def test_extract_reference_values_basic():
    ref = extract_reference_values(PAYLOAD)
    assert "15696" in ref["case_ids"]
    assert "7860" in ref["case_ids"]
    assert "14212" in ref["case_ids"]  # current_case slide_id
    assert "RMS" in ref["diagnoses"]
    assert "EWS" in ref["diagnoses"]
    assert "RMS" in ref["tumor_groups"]
    assert "EWS" in ref["tumor_groups"]
    assert 1729.90 in ref["numeric_values"]
    assert 1977.38 in ref["numeric_values"]


def test_extract_reference_values_with_derived():
    ref = extract_reference_values(PAYLOAD, DERIVED_STATS)
    assert 1907.41 in ref["numeric_values"]
    assert 1729.90 in ref["numeric_values"]


# ---------------------------------------------------------------------------
# provenance_guard
# ---------------------------------------------------------------------------

def test_provenance_guard_no_violations():
    ref = extract_reference_values(PAYLOAD, DERIVED_STATS)
    text = "RMS is dominant (3 of 5 neighbors) [15696, 15151, 8122]. Mean similarity 1907.41."
    _, violations = provenance_guard(text, ref)
    assert violations == []


def test_provenance_guard_detects_fabricated_number():
    ref = extract_reference_values(PAYLOAD, DERIVED_STATS)
    text = "The similarity mean is 9999.99 which is very high."
    _, violations = provenance_guard(text, ref)
    assert any("9999" in v for v in violations)


def test_provenance_guard_allows_small_ints():
    ref = extract_reference_values(PAYLOAD)
    text = "3 of 5 neighbors support RMS."
    _, violations = provenance_guard(text, ref)
    assert violations == []


def test_provenance_guard_allows_case_ids():
    ref = extract_reference_values(PAYLOAD)
    text = "Neighbor 15696 has RMS diagnosis."
    _, violations = provenance_guard(text, ref)
    assert violations == []


# ---------------------------------------------------------------------------
# verify_claims
# ---------------------------------------------------------------------------

def test_verify_claims_all_valid():
    claims = [
        {
            "statement": "RMS is the dominant diagnosis",
            "source_case_ids": ["15696", "15151", "8122"],
            "field": "diagnosis",
            "value": "RMS",
        },
        {
            "statement": "EWS appears in 2 neighbors",
            "source_case_ids": ["7860", "7855"],
            "field": "diagnosis",
            "value": "EWS",
        },
    ]
    verified, warnings = verify_claims(claims, PAYLOAD)
    assert len(verified) == 2
    assert warnings == []


def test_verify_claims_fabricated_case_id():
    claims = [
        {
            "statement": "Fabricated case",
            "source_case_ids": ["99999"],
            "field": "diagnosis",
            "value": "RMS",
        },
    ]
    verified, warnings = verify_claims(claims, PAYLOAD)
    assert len(verified) == 0
    assert any("99999" in w for w in warnings)


def test_verify_claims_wrong_diagnosis_for_case():
    claims = [
        {
            "statement": "7860 has RMS",
            "source_case_ids": ["7860"],
            "field": "diagnosis",
            "value": "RMS",
        },
    ]
    verified, warnings = verify_claims(claims, PAYLOAD)
    # 7860 actually has EWS, not RMS
    assert len(verified) == 0
    assert any("7860" in w and "EWS" in w for w in warnings)


def test_verify_claims_nonexistent_diagnosis():
    claims = [
        {
            "statement": "ALL appears",
            "source_case_ids": ["15696"],
            "field": "diagnosis",
            "value": "ALL",
        },
    ]
    verified, warnings = verify_claims(claims, PAYLOAD)
    assert len(verified) == 0
    assert any("ALL" in w for w in warnings)


def test_verify_claims_tumor_group_check():
    claims = [
        {
            "statement": "RMS tumor group",
            "source_case_ids": ["15696"],
            "field": "tumor_group",
            "value": "RMS",
        },
    ]
    verified, warnings = verify_claims(claims, PAYLOAD)
    assert len(verified) == 1
    assert warnings == []


def test_verify_claims_bad_tumor_group():
    claims = [
        {
            "statement": "FAKE tumor group",
            "source_case_ids": ["15696"],
            "field": "tumor_group",
            "value": "FAKE",
        },
    ]
    verified, warnings = verify_claims(claims, PAYLOAD)
    assert len(verified) == 0
    assert any("FAKE" in w for w in warnings)


# ---------------------------------------------------------------------------
# verify_case_ids_in_text
# ---------------------------------------------------------------------------

def test_verify_case_ids_in_text_all_valid():
    valid = {"15696", "15151", "8122"}
    text = "Dominant diagnosis is RMS [15696, 15151, 8122]."
    fabricated = verify_case_ids_in_text(text, valid)
    assert fabricated == []


def test_verify_case_ids_in_text_detects_fabricated():
    valid = {"15696", "15151"}
    text = "See cases [15696, 99999]."
    fabricated = verify_case_ids_in_text(text, valid)
    assert "99999" in fabricated


def test_verify_case_ids_ignores_non_id_brackets():
    valid = {"15696"}
    # Content like "higher is better" in brackets should not be flagged
    text = "Score [higher is better]. Case [15696]."
    fabricated = verify_case_ids_in_text(text, valid)
    # "higher", "is", "better" don't match the alphanumeric-only pattern
    # when split by comma, so they should not be flagged
    assert fabricated == []


# ---------------------------------------------------------------------------
# parse_grounded_response  (full integration)
# ---------------------------------------------------------------------------

def test_parse_grounded_response_structured():
    raw = json.dumps({
        "summary": "RMS is dominant [15696, 15151, 8122]. EWS in 2 [7860, 7855]. Mean similarity 1907.41.",
        "claims": [
            {
                "statement": "RMS is the dominant diagnosis",
                "source_case_ids": ["15696", "15151", "8122"],
                "field": "diagnosis",
                "value": "RMS",
            },
            {
                "statement": "EWS appears in 2 neighbors",
                "source_case_ids": ["7860", "7855"],
                "field": "diagnosis",
                "value": "EWS",
            },
        ],
    })
    summary, claims, warnings, is_grounded = parse_grounded_response(
        raw, PAYLOAD, DERIVED_STATS,
    )
    assert summary is not None
    assert "RMS" in summary
    assert is_grounded is True
    assert len(claims) == 2
    # Should have no serious warnings (valid ids, valid diagnoses, valid numbers)
    assert not any("fabricated" in w.lower() for w in warnings)


def test_parse_grounded_response_fabricated_id_in_summary():
    raw = json.dumps({
        "summary": "RMS is dominant [15696, 99999].",
        "claims": [
            {
                "statement": "RMS is dominant",
                "source_case_ids": ["15696"],
                "field": "diagnosis",
                "value": "RMS",
            },
        ],
    })
    summary, claims, warnings, is_grounded = parse_grounded_response(
        raw, PAYLOAD, DERIVED_STATS,
    )
    assert any("99999" in w for w in warnings)


def test_parse_grounded_response_not_json():
    raw = "This is plain text, not JSON."
    summary, claims, warnings, is_grounded = parse_grounded_response(
        raw, PAYLOAD,
    )
    assert summary == raw.strip()
    assert is_grounded is False
    assert any("not valid JSON" in w for w in warnings)


def test_parse_grounded_response_legacy_format():
    """Legacy format with only 'summary' key (no claims)."""
    raw = json.dumps({"summary": "RMS is dominant."})
    summary, claims, warnings, is_grounded = parse_grounded_response(
        raw, PAYLOAD,
    )
    assert summary == "RMS is dominant."
    assert claims == []
    assert is_grounded is False  # no claims present
    assert any("No claims" in w for w in warnings)


def test_parse_grounded_response_fabricated_number():
    raw = json.dumps({
        "summary": "Mean similarity is 42424242.00.",
        "claims": [],
    })
    summary, claims, warnings, is_grounded = parse_grounded_response(
        raw, PAYLOAD, DERIVED_STATS,
    )
    assert any("42424242" in w for w in warnings)
