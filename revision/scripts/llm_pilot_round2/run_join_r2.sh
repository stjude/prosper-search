#!/bin/bash
# S5c join (key-side). Run ONLY after the analyst confirms both rating forms are delivered and hashed.
# Usage: run_join_r2.sh <sha256 of rating_rater1/rating_form_filled.csv> <sha256 of rating_rater2/rating_form_filled.csv>
# Steps: (1) assert the two form hashes and the independent v1 adjudication hash; (2) compute the strict-"discrepant"
# rows mechanically from the packet form (strict_discrepant_rows_r2.py) and write lead_adjudication_r2_v2.json =
# v1 + those rows (nothing else changed); (3) join with v1 (analysis/join_v1/) and with v2 (analysis/join_v2/).
# Never writes into run/, the packets, the raters' directories or round 1.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <r1_form_sha256> <r2_form_sha256>"; exit 1; }
WS=/PROSPER_WORKSPACE/prosper_revision
export PROSPER_OUT=$WS/prosper_revision_output
R=$PROSPER_OUT/s5c_pilot_round2_20261110
P=$WS/scripts/llm_pilot_round2
PY=/HOME/miniconda3/bin/python3
F1=$R/rating_rater1/rating_form_filled.csv
F2=$R/rating_rater2/rating_form_filled.csv
V1=$R/analysis/lead_adjudication_r2_v1.json
V2=$R/analysis/lead_adjudication_r2_v2.json
V1_SHA=a2b95bd976934e7c57f3999c44c9f135869ed420cf7999a8e6444fa12be5c40e
check() { local got; got=$(sha256sum "$1" | cut -d' ' -f1); [ "$got" = "$2" ] || { echo "HASH MISMATCH $1: $got != $2"; exit 98; }; }
check "$F1" "$1"; check "$F2" "$2"; check "$V1" "$V1_SHA"
check $P/analyze_r2.py 37d8ce302e680405d7c040c2f1751762008a7761b5bcfec13957b91ed55f56b4
check $P/strict_discrepant_rows_r2.py 518d78622453d6d1e572258de4e293297da713059b224d012fc3f1e00c42deae
[ -e "$R/analysis/join_v1" ] && { echo "join_v1 exists; refusing to overwrite"; exit 99; }
[ -e "$V2" ] && { echo "$V2 exists; refusing to overwrite"; exit 99; }
COMMON="--run-dir $R/run --harness $P/llm_grounding_cli.executed_fc88a2e7.py --arm a0_masked_posterior \
 --ratings-r1 $F1 --ratings-r2 $F2 --rating-map $R/rating_packet_key/rating_id_map.csv \
 --scoring-key $PROSPER_OUT/r3_5_s5b_posterior_20q_20261009_v2/scoring_key_DO_NOT_SEND.csv"
# (2) strict rows from the unrated packet form (identical in both packets) -> v2
$PY -B $P/strict_discrepant_rows_r2.py --form $R/rating_packet_1/rating_form.csv \
  --rating-map $R/rating_packet_key/rating_id_map.csv --run-dir $R/run --out-dir $R/analysis/strict_discrepant_r2
$PY - "$V1" "$V2" "$R/analysis/strict_discrepant_r2/strict_rows.json" <<'PYEOF'
import json, sys, hashlib
v1, v2, rows = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(v1)); strict = json.load(open(rows))
d["strict_discrepant_rows"] = strict
d["v2_note"] = ("v1 (sha256 " + hashlib.sha256(open(v1, "rb").read()).hexdigest() + ") plus strict_discrepant_rows "
                "computed by strict_discrepant_rows_r2.py (strict_rows.json sha256 "
                + hashlib.sha256(open(rows, "rb").read()).hexdigest() + "); no other field changed")
open(v2, "w").write(json.dumps(d, indent=1) + "\n")
PYEOF
sha256sum $V1 $V2 $R/analysis/strict_discrepant_r2/* | tee $R/analysis/join_inputs_sha256.txt
echo "ambiguous strict sentences for the lead: $(($(wc -l < $R/analysis/strict_discrepant_r2/ambiguous.csv) - 1))"
# (3) joins
$PY -B $P/analyze_r2.py $COMMON --adjudication $V1 --out-dir $R/analysis/join_v1
$PY -B $P/analyze_r2.py $COMMON --adjudication $V2 --out-dir $R/analysis/join_v2
sha256sum $R/analysis/join_v1/* $R/analysis/join_v2/* | tee -a $R/analysis/join_inputs_sha256.txt
