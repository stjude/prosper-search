#!/usr/bin/env bash
# ============================================================
# LLM API credentials for PROSPER demo notebook
# ============================================================
# Copy this file and fill in your real keys:
#   cp llm_env.sh llm_env.local.sh   # <-- gitignored
#   nano llm_env.local.sh
#   source llm_env.local.sh
#
# Then run the demo notebook. The LLM cells will use these.
# ============================================================

# --- Azure OpenAI (preferred) ---
export AZURE_OPENAI_ENDPOINT=""        # e.g. https://your-resource.openai.azure.com
export AZURE_OPENAI_API_KEY=""
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_DEPLOYMENT=""      # e.g. o3-mini

# --- Standard OpenAI (fallback) ---
# export OPENAI_API_KEY=""
