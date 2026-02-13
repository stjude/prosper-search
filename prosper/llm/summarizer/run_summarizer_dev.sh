#!/usr/bin/env bash
###############################################################################
# run_summarizer.sh — Master test wrapper for the grounded pathology summarizer
#
# Runs all test stages sequentially:
#   1. Unit tests  (pytest, no network)
#   2. Fake LLM    (local synthetic output, no network)
#   3. Real LLM    (Azure AI Foundry, requires API key)
#
# Usage:
#   ./run_summarizer.sh                         # run all stages
#   ./run_summarizer.sh --skip-real              # skip the real LLM call
#   ./run_summarizer.sh --only-real              # only the real LLM call
#   AZURE_OPENAI_API_KEY=xxx ./run_summarizer.sh # override key inline
#
# Prerequisites:
#   pip install -e .[llm]       # installs openai dependency
###############################################################################
set -euo pipefail

# ─── Python interpreter ─────────────────────────────────────────────────────
PYTHON="/mnt/storage1/Haoran/anaconda3/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi

# ─── Resolve paths ──────────────────────────────────────────────────────────
CANONICAL_ROOT="REDACTED_INTERNAL_PATH/ImageAnalysis/Core/Haoran/comet-llm/patho-toy-summarizer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use canonical root if the local src dir doesn't exist (worktree case)
if [ -d "$SCRIPT_DIR/src/patho_toy_summarizer" ]; then
  SRC_DIR="$SCRIPT_DIR/src"
  SAMPLE_JSON="$SCRIPT_DIR/sample.json"
  NEIGHBORS_JSON="${SCRIPT_DIR}/../output/neighbors.json"
else
  SRC_DIR="$CANONICAL_ROOT/src"
  SAMPLE_JSON="$CANONICAL_ROOT/sample.json"
  NEIGHBORS_JSON="REDACTED_INTERNAL_PATH/ImageAnalysis/Core/Haoran/comet-llm/output/neighbors.json"
fi

# Ensure our local src is first on the path (avoids patho-search-llm-agent shadow)
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

# ─── Azure AI Foundry defaults ─────────────────────────────────────────────
#   Fill in your API key here or pass it as an environment variable.
#   These are stored in shell variables (not exported) so they don't leak
#   into Stage 1/2.  Only Stage 3 exports them.
_AZ_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://REDACTED_INTERNAL_ENDPOINT}"
_AZ_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-5}"
# >>> PUT YOUR KEY HERE (or export AZURE_OPENAI_API_KEY before running) <<<
_AZ_API_KEY="${AZURE_OPENAI_API_KEY:-REDACTED_REMOVED_SECRET}"

# Unset Azure vars so they don't pollute Stages 1 & 2
unset AZURE_OPENAI_ENDPOINT AZURE_OPENAI_API_KEY AZURE_OPENAI_DEPLOYMENT 2>/dev/null || true

# ─── Parse flags ────────────────────────────────────────────────────────────
SKIP_REAL=false
ONLY_REAL=false
for arg in "$@"; do
  case "$arg" in
    --skip-real) SKIP_REAL=true ;;
    --only-real) ONLY_REAL=true ;;
    -h|--help)
      echo "Usage: $0 [--skip-real | --only-real]"
      exit 0
      ;;
  esac
done

# ─── Helpers ────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
banner() { printf '\n\033[1;36m══════════════════════════════════════════════════════════════\033[0m\n'; printf '\033[1;36m  %s\033[0m\n' "$1"; printf '\033[1;36m══════════════════════════════════════════════════════════════\033[0m\n'; }
ok()     { PASS=$((PASS+1)); printf '  \033[1;32m✔ %s\033[0m\n' "$1"; }
fail()   { FAIL=$((FAIL+1)); printf '  \033[1;31m✘ %s\033[0m\n' "$1"; }

# ─── STAGE 1: Unit tests ───────────────────────────────────────────────────
if [ "$ONLY_REAL" = false ]; then

banner "STAGE 1 / 3 — Unit tests (pytest)"
TESTS_DIR="$SCRIPT_DIR/tests"
[ ! -d "$TESTS_DIR" ] && TESTS_DIR="$CANONICAL_ROOT/tests"
if "$PYTHON" -m pytest "$TESTS_DIR/" -v --tb=short 2>&1; then
  ok "All unit tests passed"
else
  fail "Some unit tests failed (see above)"
fi

# ─── STAGE 2: Fake LLM mode (no network) ─────────────────────────────────

banner "STAGE 2 / 3 — Fake LLM mode (no network)"

echo ""
echo "--- 2a. Enriched neighbors.json (similarity-based) ---"
OUTPUT_2A=$(PATHO_LLM_FAKE=1 OPENAI_API_KEY=dummy \
  "$PYTHON" -m patho_toy_summarizer.cli "$NEIGHBORS_JSON" --llm --neighbors-json 2>&1) || true
echo "$OUTPUT_2A"
echo ""

# Check: citations present
if echo "$OUTPUT_2A" | grep -q '\['; then
  ok "Inline [case_id] citations present"
else
  fail "Missing inline citations"
fi

# Check: disclaimer present
if echo "$OUTPUT_2A" | grep -q 'DISCLAIMER'; then
  ok "Medical disclaimer present"
else
  fail "Medical disclaimer missing"
fi

# Check: all case_ids from payload appear
for CID in 15696 15151 7860 7855 8122; do
  if echo "$OUTPUT_2A" | grep -q "$CID"; then
    ok "Case ID $CID cited"
  else
    fail "Case ID $CID not cited"
  fi
done

echo ""
echo "--- 2b. sample.json (probability-based) ---"
OUTPUT_2B=$(PATHO_LLM_FAKE=1 OPENAI_API_KEY=dummy \
  "$PYTHON" -m patho_toy_summarizer.cli "$SAMPLE_JSON" --llm 2>&1) || true
echo "$OUTPUT_2B"
echo ""

if echo "$OUTPUT_2B" | grep -qi 'probability'; then
  ok "Probability mode detected"
else
  fail "Probability mode not detected"
fi
if echo "$OUTPUT_2B" | grep -q 'DISCLAIMER'; then
  ok "Medical disclaimer present"
else
  fail "Medical disclaimer missing"
fi

echo ""
echo "--- 2c. Deterministic only (no --llm) ---"
OUTPUT_2C=$("$PYTHON" -m patho_toy_summarizer.cli "$NEIGHBORS_JSON" --neighbors-json 2>&1) || true
echo "$OUTPUT_2C"
echo ""

if echo "$OUTPUT_2C" | grep -q 'DISCLAIMER'; then
  ok "Disclaimer on deterministic output"
else
  fail "Disclaimer missing on deterministic output"
fi

echo ""
echo "--- 2d. --no-disclaimer flag ---"
OUTPUT_2D=$("$PYTHON" -m patho_toy_summarizer.cli "$SAMPLE_JSON" --no-disclaimer 2>&1) || true
echo "$OUTPUT_2D"
echo ""

if echo "$OUTPUT_2D" | grep -q 'DISCLAIMER'; then
  fail "--no-disclaimer flag did not suppress disclaimer"
else
  ok "--no-disclaimer correctly suppressed"
fi

fi  # end ONLY_REAL guard

# ─── STAGE 3: Real LLM call (Azure AI Foundry) ────────────────────────────

if [ "$SKIP_REAL" = true ]; then
  banner "STAGE 3 / 3 — SKIPPED (--skip-real)"
else

banner "STAGE 3 / 3 — Real LLM call (Azure AI Foundry)"

# Export Azure vars only for this stage
export AZURE_OPENAI_ENDPOINT="$_AZ_ENDPOINT"
export AZURE_OPENAI_DEPLOYMENT="$_AZ_DEPLOYMENT"
export AZURE_OPENAI_API_KEY="$_AZ_API_KEY"

if [ "$_AZ_API_KEY" = "REPLACE_ME_WITH_YOUR_KEY" ]; then
  echo ""
  echo "  ⚠  AZURE_OPENAI_API_KEY is not set."
  echo "     Either:"
  echo "       1. Edit this script and replace REPLACE_ME_WITH_YOUR_KEY"
  echo "       2. Export it before running:  export AZURE_OPENAI_API_KEY=<key>"
  echo "       3. Pass it inline:  AZURE_OPENAI_API_KEY=<key> $0"
  echo ""
  fail "Real LLM test skipped — no API key"
else
  echo "  Endpoint:   $AZURE_OPENAI_ENDPOINT"
  echo "  Deployment: $AZURE_OPENAI_DEPLOYMENT"
  echo ""

  echo "--- 3a. Enriched neighbors.json (real LLM, grounded prompt) ---"
  OUTPUT_3A=$("$PYTHON" -m patho_toy_summarizer.cli "$NEIGHBORS_JSON" \
    --llm --neighbors-json 2>&1) || true
  echo "$OUTPUT_3A"
  echo ""

  if echo "$OUTPUT_3A" | grep -q 'ERROR:'; then
    fail "LLM returned an error (see above)"
  elif echo "$OUTPUT_3A" | grep -q 'DISCLAIMER'; then
    ok "Real LLM call succeeded"
    # Check for grounded citations
    if echo "$OUTPUT_3A" | grep -q '\['; then
      ok "Real LLM output contains inline citations"
    else
      echo "  ℹ  No bracket citations found — model may not have followed structured format."
      echo "     Deterministic fallback may have been used (post-hoc verification rejected output)."
    fi
  else
    fail "Unexpected output (no DISCLAIMER, no ERROR)"
  fi

  echo ""
  echo "--- 3b. sample.json (real LLM, probability mode) ---"
  OUTPUT_3B=$("$PYTHON" -m patho_toy_summarizer.cli "$SAMPLE_JSON" --llm 2>&1) || true
  echo "$OUTPUT_3B"
  echo ""

  if echo "$OUTPUT_3B" | grep -q 'ERROR:'; then
    fail "LLM returned an error (see above)"
  elif echo "$OUTPUT_3B" | grep -q 'DISCLAIMER'; then
    ok "Real LLM call succeeded (sample.json)"
  else
    fail "Unexpected output"
  fi
fi

fi  # end SKIP_REAL guard

# ─── Summary ────────────────────────────────────────────────────────────────
banner "RESULTS"
TOTAL=$((PASS + FAIL))
printf '  \033[1;32m%d passed\033[0m  /  ' "$PASS"
if [ "$FAIL" -gt 0 ]; then
  printf '\033[1;31m%d failed\033[0m  /  %d total\n' "$FAIL" "$TOTAL"
else
  printf '0 failed  /  %d total\n' "$TOTAL"
fi
echo ""

exit "$FAIL"
