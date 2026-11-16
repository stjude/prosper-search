#!/bin/bash
# R2.2-ALL Stage B launcher: env hardening (R2.1 route-b profile) + comet-uni2 python + extract.
# Usage: stage_b_task.sh <extract args...>   (all args passed to stage_b_extract.py)
set -uo pipefail
export LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
unset NVIDIA_TF32_OVERRIDE CUBLAS_WORKSPACE_CONFIG HF_HOME HF_HUB_CACHE PYTHONPATH PYTHONHOME
PY=/HOME/miniconda3/envs/comet-uni2/bin/python3
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "[task] host=$(hostname) job=${LSB_JOBID:-local} queue=${LSB_QUEUE:-} start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader 2>&1 | head -2
EXTRA=()
if [ ! -e /lib64/libopenslide.so.0 ]; then
  # RHEL9 image: no system libopenslide; use the virchow2 env's libopenslide 4.0.0 (Stage A rhel9 probe 336556876)
  export LD_LIBRARY_PATH=/HOME/miniconda3/envs/virchow2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  EXTRA=(--accept-libopenslide 4.0.0)
  echo "[task] no /lib64/libopenslide.so.0 -> LD_LIBRARY_PATH=$LD_LIBRARY_PATH ${EXTRA[*]}"
fi
"$PY" "$HERE/stage_b_extract.py" "$@" "${EXTRA[@]}"
RC=$?
echo "[task] rc=$RC end=$(date -Is)"
exit $RC
