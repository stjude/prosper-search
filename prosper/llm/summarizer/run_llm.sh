#!/usr/bin/env bash
# Helper script to run LLM summarization on an enriched neighbors.json payload.
# Usage examples:
#   ./run_llm.sh output/neighbors.json
#   PATHO_LLM_FAKE=1 ./run_llm.sh output/neighbors.json
#   PATHO_LLM_MODEL=gpt-4o-mini ./run_llm.sh output/neighbors.json
#
# Supports two backends:
#   1) OpenAI        – set OPENAI_API_KEY
#   2) Azure Foundry – set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
#                      (optionally AZURE_OPENAI_DEPLOYMENT; default: o3-mini)
#
# Optional: OPENAI_ORG_ID or PATHO_OPENAI_ORG for organization scoping (OpenAI only).
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <neighbors.json path> [extra args]" >&2
  exit 2
fi

JSON_PATH="$1"
shift || true

if [ ! -f "$JSON_PATH" ]; then
  echo "File not found: $JSON_PATH" >&2
  exit 3
fi

# Warn if no key and not in fake mode
if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${AZURE_OPENAI_API_KEY:-}" ] && [ -z "${PATHO_LLM_FAKE:-}" ]; then
  echo "WARNING: No API key set (OPENAI_API_KEY or AZURE_OPENAI_API_KEY) and PATHO_LLM_FAKE not enabled; LLM call will be skipped." >&2
fi

# Run summarizer with enriched flag
python -m patho_toy_summarizer.cli "$JSON_PATH" --llm --neighbors-json "$@"
