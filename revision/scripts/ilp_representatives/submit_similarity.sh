#!/bin/bash
# Submit the within-cohort UNI2+OT similarity arrays for [analysis plan].
# Usage: PROSPER_OUT=... scripts/ilp_representatives/submit_similarity.sh stv|ccdi
# Requires the params file written by stage_population.py.
set -euo pipefail
cohort="${1:?cohort (stv|ccdi)}"
: "${PROSPER_OUT:?export PROSPER_OUT first (see CLAUDE.md)}"
PROSPER_ROOT="${PROSPER_ROOT:-/PROSPER_WORKSPACE/prosper-search}"

params="$PROSPER_OUT/ilp_representatives/$cohort/params_similarity.txt"
n=$(wc -l < "$params")
task="similarity_${cohort}_inlib_uni2_ot"
log="$PROSPER_OUT/logs/$task"
mkdir -p "$log"

# 8 threads for OT's per-query library loop; ~6 GB library + queries, ask 16 GB total.
# Canary (2026-09-25, local): 688-slide library loads in 61 s, one row in 32 s on ~10 cores.
bsub -q priority -n 8 -R "span[hosts=1] rusage[mem=2000]" -W 180 \
     -J "prosper_${task}[1-${n}]" \
     -oo "$log/prosper_${task}.%J_%I.out" -eo "$log/prosper_${task}.%J_%I.err" \
     env PROSPER_OUT="$PROSPER_OUT" PROSPER_PARAMS="$params" \
         PROSPER_MODULE=prosper.embeddings.similarity_cli \
         "$PROSPER_ROOT/scripts/lsf/prosper_job.sh"
