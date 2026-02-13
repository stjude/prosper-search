import json
from patho_toy_summarizer.llm import attempt_llm_summary


def test_llm_monkeypatch_success(monkeypatch):
    # Prepare fake stats
    stats = {
        "mode_diagnosis": "Invasive ductal carcinoma",
        "diagnosis_counts": {"Invasive ductal carcinoma": 4, "Lobular features": 1},
        "mean_probability": 0.79,
        "min_probability": 0.72,
        "max_probability": 0.86,
        "score_name": "probability",
        "score_direction": "higher",
        "outlier_count": 1,
        "total_neighbors": 5,
    }

    # Monkeypatch environment and network call
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    def fake_safe_import_openai():  # returns dummy client class
        class DummyClient:
            class chat:  # noqa: N801 - mimic structure
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs):  # noqa: D401
                        class C:
                            class message:  # noqa: N801
                                content = json.dumps({
                                    "summary": "Neighbors support invasive ductal carcinoma; high similarity, 1 outlier."
                                })

                        class R:
                            choices = [C()]

                        return R()

        return DummyClient

    monkeypatch.setattr("patho_toy_summarizer.llm._safe_import_openai", fake_safe_import_openai)
    summary, accepted = attempt_llm_summary(stats)
    assert summary is not None
    assert accepted in (True, False)
    assert "invasive ductal carcinoma" in summary.lower()
    # With 200 word limit, ensure not absurdly long (>400) and truncated if needed
    assert len(summary.split()) <= 201


def test_llm_invalid_json(monkeypatch):
    stats = {
        "mode_diagnosis": "Adenocarcinoma",
        "diagnosis_counts": {"Adenocarcinoma": 5},
        "mean_similarity": 0.80,
        "min_similarity": 0.75,
        "max_similarity": 0.82,
        "score_name": "similarity",
        "score_direction": "lower",
        "outlier_count": 0,
        "total_neighbors": 5,
    }
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    def fake_safe_import_openai():
        class DummyClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs):
                        class C:
                            class message:  # noqa: N801
                                content = "NOT JSON"

                        class R:
                            choices = [C()]
                        return R()
        return DummyClient

    monkeypatch.setattr("patho_toy_summarizer.llm._safe_import_openai", fake_safe_import_openai)
    summary, accepted = attempt_llm_summary(stats)
    assert summary == "NOT JSON"  # raw text accepted
    assert accepted is True  # "NOT JSON" under 200 words


def test_llm_probability_postprocess(monkeypatch):
    """Post-processing replaces 'similarity' with 'probability' and appends top score."""
    stats = {
        "mode_diagnosis": "Adenocarcinoma",
        "diagnosis_counts": {"Adenocarcinoma": 5},
        "mean_probability": 0.80,
        "min_probability": 0.75,
        "max_probability": 0.82,
        "score_name": "probability",
        "score_direction": "higher",
        "top_score": 0.82,
        "top_tumor_type": "ADC",
        "outlier_count": 0,
        "total_neighbors": 5,
    }
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    def fake_safe_import_openai():
        class DummyClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs):
                        class C:
                            class message:  # noqa: N801
                                # Use 0.80 (a real stat value) so provenance guard passes
                                content = "Similarity mean 0.80 (lower is better)."

                        class R:
                            choices = [C()]
                        return R()
        return DummyClient

    monkeypatch.setattr("patho_toy_summarizer.llm._safe_import_openai", fake_safe_import_openai)
    summary, accepted = attempt_llm_summary(stats)
    assert summary is not None
    assert "probability" in summary.lower()
    assert "highest probability" in summary.lower()


def test_llm_fake_mode(monkeypatch):
    stats = {
        "mode_diagnosis": "Adenocarcinoma",
        "diagnosis_counts": {"Adenocarcinoma": 4, "Squamous": 1},
        "mean_probability": 0.78,
        "min_probability": 0.72,
        "max_probability": 0.86,
        "score_name": "probability",
        "score_direction": "higher",
        "outlier_count": 1,
        "total_neighbors": 5,
    }
    monkeypatch.setenv("PATHO_LLM_FAKE", "1")
    # OPENAI_API_KEY not needed in fake mode but set anyway for completeness
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    summary, accepted = attempt_llm_summary(stats)
    assert "Adenocarcinoma" in summary
    assert accepted in (True, False)
    assert len(summary.split()) <= 201  # allow ellipsis under 200 limit


def test_llm_provenance_guard_rejects_fabricated_numbers(monkeypatch):
    """Provenance guard rejects LLM output containing numbers not in stats."""
    stats = {
        "mode_diagnosis": "Adenocarcinoma",
        "diagnosis_counts": {"Adenocarcinoma": 5},
        "mean_similarity": 0.80,
        "min_similarity": 0.75,
        "max_similarity": 0.82,
        "score_name": "similarity",
        "score_direction": "lower",
        "outlier_count": 0,
        "total_neighbors": 5,
    }
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    def fake_safe_import_openai():
        class DummyClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs):
                        class C:
                            class message:  # noqa: N801
                                # Contains a fabricated number (99.99)
                                content = json.dumps({
                                    "summary": "Similarity mean 99.99, clearly fabricated."
                                })

                        class R:
                            choices = [C()]
                        return R()
        return DummyClient

    monkeypatch.setattr("patho_toy_summarizer.llm._safe_import_openai", fake_safe_import_openai)
    summary, accepted = attempt_llm_summary(stats)
    # Provenance guard should reject → returns None to trigger deterministic fallback
    assert summary is None
    assert accepted is False
