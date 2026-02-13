"""
LLM Summarizer for PROSPER.

Provides LLM-powered summarization with grounded citations
and provenance verification.

Quick usage (fake mode, no API key needed):

    PATHO_LLM_FAKE=1 OPENAI_API_KEY=dummy \
      python -m patho_toy_summarizer.cli <neighbors.json> --llm --neighbors-json

See run_summarizer_dev.sh or run_summarizer_user.sh for full examples.
"""

from .src.patho_toy_summarizer import (
    attempt_llm_summary,
    attempt_llm_summary_from_payload,
    MEDICAL_DISCLAIMER,
    provenance_guard,
    verify_claims,
    parse_grounded_response,
    summarize_neighbors,
    compute_stats,
)

__all__ = [
    "attempt_llm_summary",
    "attempt_llm_summary_from_payload",
    "MEDICAL_DISCLAIMER",
    "provenance_guard",
    "verify_claims",
    "parse_grounded_response",
    "summarize_neighbors",
    "compute_stats",
]
