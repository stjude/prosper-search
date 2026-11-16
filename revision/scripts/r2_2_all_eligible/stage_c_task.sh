#!/bin/bash
# R2.2-ALL Stage C task wrapper. Usage: stage_c_task.sh <threads> <gpu:0|1> <script.py> [args...]
# Pins BLAS/OpenMP threads, never writes bytecode into read-only trees, records the
# environment, applies the foreign-GPU-occupancy guard (exit 75) and the A100-80G/RHEL8
# admission rule (exit 3) for GPU tasks, and prints "[task] rc=N" for the watcher.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
THREADS="$1"; GPU_TASK="$2"; SCRIPT="$3"; shift 3
export PYTHONDONTWRITEBYTECODE=1
export PROSPER_OUT=/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
PY=/HOME/miniconda3/bin/python3
# Per-attempt log kept beside lsfq's (which -oo overwrites on retry): originals are retained.
JOBLOGS="$PROSPER_OUT/r2_2_uni2_all_eligible_20261113/run_001/stage_c/joblogs"
mkdir -p "$JOBLOGS"
exec > >(tee -a "$JOBLOGS/${LSB_JOBNAME:-local}.${LSB_JOBID:-$$}.log") 2>&1
OSREL=$(grep -o 'release [0-9.]*' /etc/redhat-release 2>/dev/null)
echo "[task] host=$(hostname) job=${LSB_JOBID:-local} queue=${LSB_QUEUE:-} os='$OSREL' threads=$THREADS gpu_task=$GPU_TASK start=$(date +%FT%T%z)"
if [ "$GPU_TASK" = "1" ]; then
  MIN_FREE_MB=16384
  DEV="${CUDA_VISIBLE_DEVICES:-0}"; DEV="${DEV%%,*}"
  MEM=$(nvidia-smi -i "$DEV" --query-gpu=index,name,uuid,memory.used,memory.total --format=csv,noheader,nounits 2>&1)
  echo "[c-guard] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} gpu(index,name,uuid,used,total MiB)=$MEM"
  NAME=$(echo "$MEM" | awk -F', ' 'NR==1{print $2}')
  USED=$(echo "$MEM" | awk -F', ' 'NR==1{print $4}'); TOTAL=$(echo "$MEM" | awk -F', ' 'NR==1{print $5}')
  if [ "$NAME" != "NVIDIA A100-SXM4-80GB" ] || [[ "$OSREL" != release\ 8* ]]; then
    echo "[c-guard] GPU '$NAME' / '$OSREL' is not admitted (A100-SXM4-80GB, RHEL8 only); refusing"
    echo "[task] rc=3"; exit 3
  fi
  if [[ "$USED" =~ ^[0-9]+$ && "$TOTAL" =~ ^[0-9]+$ ]] && (( TOTAL - USED < MIN_FREE_MB )); then
    echo "[c-guard] FOREIGN_GPU_OCCUPANCY free=$((TOTAL-USED))MiB on $(hostname) gpu $DEV"
    echo "[task] rc=75"; exit 75
  fi
fi
"$PY" "$HERE/$SCRIPT" "$@"
rc=$?
echo "[task] rc=$rc end=$(date +%FT%T%z)"
exit $rc
