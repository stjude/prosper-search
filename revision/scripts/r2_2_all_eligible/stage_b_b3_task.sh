#!/bin/bash
# R2.2-ALL B3 task wrapper (design decision 2026-11-13 16:40 CST): production architecture is
# A100-SXM4-80GB on RHEL8 only. Refuse to run on any other GPU, then hand over to the frozen
# launcher (stage_b_task.sh, unchanged). Usage: stage_b_b3_task.sh <stage_b_task.sh args...>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# Free-memory guard (2026-11-13 16:55 CST): LSF cannot see non-LSF GPU processes, so a
# GPU it allocates may already be full. Exit 75 within seconds if the assigned GPU has
# < 8 GB free; the watcher then excludes the host and requeues the task without charge.
MIN_FREE_MB=8192
DEV="${CUDA_VISIBLE_DEVICES:-0}"; DEV="${DEV%%,*}"
MEM=$(nvidia-smi -i "$DEV" --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits 2>&1)
echo "[b3-guard] host=$(hostname) job=${LSB_JOBID:-local} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} gpu(index,uuid,used_MiB,total_MiB)=$MEM"
USED=$(echo "$MEM" | awk -F', ' 'NR==1{print $3}'); TOTAL=$(echo "$MEM" | awk -F', ' 'NR==1{print $4}')
if [[ "$USED" =~ ^[0-9]+$ && "$TOTAL" =~ ^[0-9]+$ ]]; then
  if (( TOTAL - USED < MIN_FREE_MB )); then
    UUID=$(echo "$MEM" | awk -F', ' 'NR==1{print $2}')
    echo "[b3-guard] FOREIGN_GPU_OCCUPANCY free=$((TOTAL-USED))MiB < ${MIN_FREE_MB}MiB on $(hostname) gpu $DEV; processes:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>&1 \
      | grep -F "$UUID" | while IFS=, read -r _ pid name mem; do
          echo "[b3-guard]   pid=$pid user=$(ps -o user= -p $pid 2>/dev/null || echo '?') name=$name mem=$mem"
        done
    exit 75
  fi
else
  echo "[b3-guard] could not read GPU memory ($MEM); continuing to the model/OOM checks"
fi
REQUIRED="NVIDIA A100-SXM4-80GB"
GPU=$(env -u PYTHONPATH PYTHONNOUSERSITE=1 /HOME/miniconda3/envs/comet-uni2/bin/python3 -c \
  'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)
echo "[b3] host=$(hostname) job=${LSB_JOBID:-local} queue=${LSB_QUEUE:-} gpu='$GPU'"
if [ "$GPU" != "$REQUIRED" ]; then
  echo "[b3] GPU '$GPU' is not admitted (required '$REQUIRED'); refusing" >&2
  exit 3
fi
if [ ! -e /lib64/libopenslide.so.0 ]; then
  echo "[b3] not a RHEL8 image (no /lib64/libopenslide.so.0); refusing" >&2
  exit 3
fi
exec bash "$HERE/stage_b_task.sh" "$@"
