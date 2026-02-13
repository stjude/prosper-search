"""
LLM integration module for PROSPER.

Submodules:
    - search:      Generate LLM-ready JSON from slide search results
    - summarizer:  LLM-powered summarization with grounded citations

Usage (library):
    from prosper.llm import (
        load_results_new, load_metadata, build_json_for_slide,
        attempt_llm_summary_from_payload, MEDICAL_DISCLAIMER,
    )

Usage (CLI):
    # 1. Generate neighbor JSON
    python -m prosper.llm.search.slide_to_json \\
        --sim data/slide_embeddings/virchow2_sampler.pkl \\
        --metadata data/metadata/final_scan_id_disease_map_rich_data.csv \\
        --slide 4261 --out output/llm/slide_4261.json

    # 2. Summarize with LLM (fake mode for testing)
    PATHO_LLM_FAKE=1 OPENAI_API_KEY=dummy \\
      python -m patho_toy_summarizer.cli output/llm/slide_4261.json \\
        --llm --neighbors-json
"""

from .search import (
    load_results_new,
    load_metadata,
    build_json_for_slide,
    build_json_from_search_results,
    top_k_neighbors,
    slide_to_json_main,
)

from .summarizer import (
    attempt_llm_summary,
    attempt_llm_summary_from_payload,
    MEDICAL_DISCLAIMER,
    provenance_guard,
    verify_claims,
    parse_grounded_response,
)

__all__ = [
    # Search / payload building
    "load_results_new",
    "load_metadata",
    "build_json_for_slide",
    "build_json_from_search_results",
    "top_k_neighbors",
    "slide_to_json_main",
    # Summarizer
    "attempt_llm_summary",
    "attempt_llm_summary_from_payload",
    "MEDICAL_DISCLAIMER",
    "provenance_guard",
    "verify_claims",
    "parse_grounded_response",
]
