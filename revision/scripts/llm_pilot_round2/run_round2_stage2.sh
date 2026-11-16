#!/bin/bash
# S5c round 2, STAGE 2 (real Azure calls; 20 A0 summaries). Run ONLY after the independent release recorded in
# [internal record] "S5c — LLM pilot round 2 (revision)". Submit through the lsf-max-throughput skill
# (round 1 ran as LSF job 330406465 on heavy_io, 4 GB, egress verified by a probe job first).
# Specification [internal record]; harness fc88a2e7 (byte copy), wrapper 61d75726d874cf34.
set -u
WS=/PROSPER_WORKSPACE/prosper_revision
export PROSPER_OUT=$WS/prosper_revision_output
cd $WS || exit 97
P=scripts/llm_pilot_round2
IN=$PROSPER_OUT/r3_5_s5b_posterior_20q_20261009_v2
M=/PROSPER_WORKSPACE/comet-feature-gen/comet-uni2/output/similarity_matrix/combined_similarity_matrix_uni2_ot_cosine_no_same_patient.pkl
OUT=$PROSPER_OUT/s5c_pilot_round2_20261110/run
H=$(sha256sum $P/llm_grounding_cli.executed_fc88a2e7.py | cut -c1-16)
WR=$(sha256sum $P/run_round2.py | cut -c1-16)
echo "host=$(hostname) start=$(date -u +%FT%TZ) harness=$H wrapper=$WR"
[ "$H" = "fc88a2e702ad23fc" ] || { echo "HARNESS HASH MISMATCH $H"; exit 98; }
[ "$WR" = "61d75726d874cf34" ] || { echo "WRAPPER HASH MISMATCH $WR"; exit 98; }
[ -e "$OUT/COMPLETE" ] && { echo "$OUT already COMPLETE"; exit 99; }
unset PATHO_LLM_FAKE
# only the --env-file key and the harness defaults may apply (review 2026-11-10, fix B)
unset AZURE_OPENAI_API_VERSION AZURE_OPENAI_ENDPOINT AZURE_OPENAI_DEPLOYMENT AZURE_OPENAI_API_KEY OPENAI_API_KEY
/HOME/miniconda3/bin/python3 -B $P/run_round2.py \
  --matrix $M --k 5 --summarizer manuscript --arms a0_masked_posterior \
  --queries-file $IN/harness_query_ids.txt \
  --probabilities-file $IN/posterior_fragments.json --weights-file $IN/neighbor_weights.csv \
  --swap-table $PROSPER_OUT/s5b_pilot_20261009/key/a2_swap_table.csv \
  --env-file /PROSPER_WORKSPACE/comet-llm/.env \
  --out-dir $OUT \
  2> $PROSPER_OUT/s5c_pilot_round2_20261110/run.stderr.keyside.log
rc=$?
echo "end=$(date -u +%FT%TZ) rc=$rc"
exit $rc
