import json
from patho_toy_summarizer.llm import attempt_llm_summary_from_payload

ENRICHED_PAYLOAD = {
    "schema_version": 1,
    "current_case": {
        "slide_id": "CASE123",
        "age": "10 yr",
        "sex": "F",
        "site": "Brain (P)",
        "tumor_group": "MED",
        "evaluation": "Diagnostic",
        "sample_type": "Patient tumor",
        "treatment": "N",
        "discrepant": "N",
        "diagnosis": "Medulloblastoma",
        "met": "Primary",
        "ped_adult": "Pediatric"
    },
    "neighbors": [
        {"case_id": "N1", "probability": 0.12, "diagnosis": "Medulloblastoma", "tumor_group": "MED", "age": "11 yr", "sex": "F", "site": "Brain (P)", "evaluation": "Diagnostic", "sample_type": "Patient tumor", "treatment": "N", "discrepant": "N", "met": "Primary", "ped_adult": "Pediatric"},
        {"case_id": "N2", "probability": 0.09, "diagnosis": "Medulloblastoma", "tumor_group": "MED", "age": "9 yr", "sex": "M", "site": "Brain (P)", "evaluation": "Diagnostic", "sample_type": "Patient tumor", "treatment": "N", "discrepant": "N", "met": "Primary", "ped_adult": "Pediatric"},
        {"case_id": "N3", "probability": 0.32, "diagnosis": "Medulloblastoma", "tumor_group": "MED", "age": "10 yr", "sex": "F", "site": "Brain (P)", "evaluation": "Diagnostic", "sample_type": "Patient tumor", "treatment": "N", "discrepant": "N", "met": "Primary", "ped_adult": "Pediatric"},
        {"case_id": "N4", "probability": 0.15, "diagnosis": "Medulloblastoma", "tumor_group": "MED", "age": "10 yr", "sex": "F", "site": "Brain (P)", "evaluation": "Diagnostic", "sample_type": "Patient tumor", "treatment": "N", "discrepant": "N", "met": "Primary", "ped_adult": "Pediatric"},
        {"case_id": "N5", "probability": 0.55, "diagnosis": "Ependymoma", "tumor_group": "EPN", "age": "8 yr", "sex": "F", "site": "Spine (P)", "evaluation": "Diagnostic", "sample_type": "Patient tumor", "treatment": "N", "discrepant": "Y", "met": "Primary", "ped_adult": "Pediatric"}
    ]
}


def test_llm_enriched_fake_mode(monkeypatch):
    monkeypatch.setenv("PATHO_LLM_FAKE", "1")
    # OPENAI_API_KEY optional for fake, set to avoid early return
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    summary, accepted = attempt_llm_summary_from_payload(ENRICHED_PAYLOAD)
    assert summary is not None
    assert "Medulloblastoma" in summary
    # Should mention discrepant neighbor N5
    assert "N5" in summary or "discrepant" in summary.lower()
    assert "Highest probability" in summary
    assert len(summary.split()) <= 201
    assert accepted in (True, False)


def test_llm_enriched_fake_mode_has_citations(monkeypatch):
    """Fake grounded output should contain inline [case_id] citations."""
    monkeypatch.setenv("PATHO_LLM_FAKE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    summary, _ = attempt_llm_summary_from_payload(ENRICHED_PAYLOAD)
    assert summary is not None
    # Should have bracket citations with real case IDs
    assert "[" in summary and "]" in summary
    # At least one valid case ID should appear inside brackets
    assert any(cid in summary for cid in ("N1", "N2", "N3", "N4", "N5"))


def test_llm_enriched_grounded_response(monkeypatch):
    """Real LLM call returns grounded structured JSON with claims."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    grounded_response = json.dumps({
        "summary": (
            "Neighbors suggest Medulloblastoma as the dominant diagnosis "
            "(4 of 5) [N1, N2, N3, N4]. Probability mean 0.25 range "
            "0.09-0.55 (higher is better) [N1, N2, N3, N4, N5]. "
            "Highest probability 0.55 for Ependymoma [N5]. "
            "Discrepant neighbor N5 (Ependymoma, EPN) diverges from "
            "the dominant pattern [N5]."
        ),
        "claims": [
            {
                "statement": "Medulloblastoma is the dominant diagnosis (4 of 5)",
                "source_case_ids": ["N1", "N2", "N3", "N4"],
                "field": "diagnosis",
                "value": "Medulloblastoma",
            },
            {
                "statement": "Ependymoma appears in 1 neighbor",
                "source_case_ids": ["N5"],
                "field": "diagnosis",
                "value": "Ependymoma",
            },
        ],
    })

    def fake_safe_import_openai():
        class DummyClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        class C:
                            class message:
                                content = grounded_response
                        class R:
                            choices = [C()]
                        return R()
        return DummyClient

    monkeypatch.setattr("patho_toy_summarizer.llm._safe_import_openai", fake_safe_import_openai)
    summary, accepted = attempt_llm_summary_from_payload(ENRICHED_PAYLOAD)
    assert summary is not None
    assert "Medulloblastoma" in summary
    assert "[N1" in summary  # inline citation
    assert accepted in (True, False)


def test_llm_enriched_rejects_fabricated_ids(monkeypatch):
    """Response with fabricated case_ids should be rejected (returns None)."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    bad_response = json.dumps({
        "summary": (
            "RMS is dominant [FAKE1, FAKE2, FAKE3]. "
            "Fabricated score 9999.99. "
            "Another fabricated ID [FAKE4]."
        ),
        "claims": [
            {
                "statement": "RMS is dominant",
                "source_case_ids": ["FAKE1", "FAKE2", "FAKE3"],
                "field": "diagnosis",
                "value": "RMS",
            },
        ],
    })

    def fake_safe_import_openai():
        class DummyClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        class C:
                            class message:
                                content = bad_response
                        class R:
                            choices = [C()]
                        return R()
        return DummyClient

    monkeypatch.setattr("patho_toy_summarizer.llm._safe_import_openai", fake_safe_import_openai)
    summary, accepted = attempt_llm_summary_from_payload(ENRICHED_PAYLOAD)
    # Should be rejected because of multiple fabricated IDs + numbers
    assert summary is None
    assert accepted is False
