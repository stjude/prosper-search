#!/usr/bin/env bash
###############################################################################
# run_summarizer_user.sh — Pathologist-facing summarizer
#
# Generates a concise, clinician-friendly summary for the query case based on
# its nearest-neighbor pathology slides.  All safety guardrails (grounding
# verification, provenance checks, medical disclaimer) run silently in the
# background.
#
# Usage:
#   bash run_summarizer_user.sh                          # default input
#   bash run_summarizer_user.sh /path/to/neighbors.json  # custom input
###############################################################################
set -euo pipefail

# ─── Python interpreter ─────────────────────────────────────────────────────
# Use the conda Python that has openai installed.  Falls back to `python` if
# the conda path doesn't exist.
PYTHON="/mnt/storage1/Haoran/anaconda3/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi

# ─── Resolve paths ──────────────────────────────────────────────────────────
CANONICAL_ROOT="REDACTED_INTERNAL_PATH/ImageAnalysis/Core/Haoran/comet-llm/patho-toy-summarizer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "$SCRIPT_DIR/src/patho_toy_summarizer" ]; then
  SRC_DIR="$SCRIPT_DIR/src"
else
  SRC_DIR="$CANONICAL_ROOT/src"
fi

DEFAULT_INPUT="REDACTED_INTERNAL_PATH/ImageAnalysis/Core/Haoran/comet-llm/output/neighbors.json"
INPUT="${1:-$DEFAULT_INPUT}"

if [ ! -f "$INPUT" ]; then
  echo "Error: Input file not found — $INPUT" >&2
  exit 1
fi

# Ensure our local src is first on PYTHONPATH
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

# ─── Azure AI Foundry credentials (silent) ──────────────────────────────────
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://REDACTED_INTERNAL_ENDPOINT}"
export AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-5}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-REDACTED_REMOVED_SECRET}"

# ─── Run summarizer ────────────────────────────────────────────────────────
"$PYTHON" -m patho_toy_summarizer.cli "$INPUT" --llm --neighbors-json 2>/dev/null
