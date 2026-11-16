from __future__ import annotations
import json
import os
import sys
import argparse
from .logic import summarize_neighbors, compute_stats
from .llm import attempt_llm_summary, attempt_llm_summary_from_payload
from .grounding import MEDICAL_DISCLAIMER


def _emit(text: str, *, disclaimer: bool = True) -> None:
    """Print *text* and, unless suppressed, append the medical disclaimer."""
    print(text)
    if disclaimer:
        print()
        print(MEDICAL_DISCLAIMER)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pathology neighbor cases")
    parser.add_argument("json_path", help="Input JSON file (original 5-neighbor stats JSON or enriched neighbors.json)")
    parser.add_argument("--llm", action="store_true", help="Attempt LLM rewrite if OPENAI_API_KEY set")
    parser.add_argument("--neighbors-json", action="store_true", help="Treat input as enriched neighbors.json (includes demographics) for richer LLM narrative")
    parser.add_argument("--no-disclaimer", action="store_true", help="Suppress the medical disclaimer footer")
    args = parser.parse_args()
    show_disclaimer = not args.no_disclaimer
    path = args.json_path
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        sys.exit(3)

    if args.neighbors_json:
        # For enriched payload we still produce a deterministic short line using existing logic if possible
        deterministic = None
        try:
            deterministic = summarize_neighbors(data)
        except Exception:
            # If legacy compute_stats fails due to schema mismatch, skip deterministic summary
            pass
        if args.llm:
            llm_text, accepted = attempt_llm_summary_from_payload(data)
            if llm_text and not llm_text.startswith("ERROR:"):
                label = "LLM" if accepted else "LLM (truncated)"
                _emit(f"{label}: {llm_text}", disclaimer=show_disclaimer)
                return
            if llm_text and llm_text.startswith("ERROR:"):
                print(llm_text)
        if deterministic:
            _emit(deterministic, disclaimer=show_disclaimer)
        return
    # Original path (5-neighbor minimal schema)
    try:
        deterministic = summarize_neighbors(data)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(4)
    if args.llm:
        llm_text, accepted = attempt_llm_summary(compute_stats(data))
        if llm_text and not llm_text.startswith("ERROR:"):
            label = "LLM" if accepted else "LLM (truncated)"
            _emit(f"{label}: {llm_text}", disclaimer=show_disclaimer)
            return
        if llm_text and llm_text.startswith("ERROR:"):
            print(llm_text)
    _emit(deterministic, disclaimer=show_disclaimer)


if __name__ == "__main__":  # pragma: no cover
    main()
