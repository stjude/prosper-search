#!/bin/bash
# LSF worker for prosper.evaluation.* entry points.
#
# Deliberate design points, each of which is a defect in the scripts this
# replaces (see [internal record] and execution spec rule 3):
#
#  * NO internal `exec > "$LOG" 2> "$ERR"` redirect. The old drivers did that
#    before doing any work, which swallowed LSF's "Resource usage summary"
#    epilogue -- the reason no timing or memory data exists anywhere in the
#    original logs. Log files are owned by `#BSUB -oo/-eo` instead.
#  * The interpreter is pinned by absolute path rather than by
#    `source ~/.bashrc; conda activate`, which on this cluster emits a
#    conda-libmamba-solver / libarchive.so.19 error into every job log.
#  * Paths come from the environment (COMET_ROOT, PROSPER_OUT), never from a
#    hardcoded mount. comet-* trees are read-only inputs.
#
# Two invocation modes:
#   direct : prosper_job.sh --model uni2 --method ot_cosine ...
#   array  : PROSPER_PARAMS=<file> prosper_job.sh    # reads line $LSB_JOBINDEX
set -euo pipefail

PROSPER_ROOT="${PROSPER_ROOT:-/PROSPER_WORKSPACE/prosper-search}"
export COMET_ROOT="${COMET_ROOT:-/PROSPER_WORKSPACE/comet-feature-gen}"
export PROSPER_OUT="${PROSPER_OUT:-/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output}"  # in-repo root since 2026-09-02; the sibling ../prosper_revision_output is the old root
export PYTHONUNBUFFERED=1
# Defence in depth for [decision record]. CPython randomises the hash of str per process,
# so `list(<set of str>)` iterates in a run-dependent order. The evaluator built
# its reference-library order that way and `_rank_neighbors` breaks distance ties
# by array position, so ~4% of Yottixel queries got an arbitrary top-1 that moved
# between runs. That construction is fixed, but pinning the seed means a future
# set-ordered construction cannot silently reintroduce it.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
# tqdm writes one \r-delimited line, so a progress bar becomes ~28 KB of
# unreadable log per job -- 7 MB across a 256-job sweep, and it makes the
# resource epilogue hard to find. tqdm reads TQDM_DISABLE natively.
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
PYTHON="${PROSPER_PYTHON:-/HOME/miniconda3/bin/python3}"
MODULE="${PROSPER_MODULE:-prosper.evaluation.cli}"

if [[ -n "${PROSPER_PARAMS:-}" ]]; then
    idx="${LSB_JOBINDEX:?PROSPER_PARAMS set but LSB_JOBINDEX is not}"
    line="$(sed -n "${idx}p" "$PROSPER_PARAMS")"
    [[ -n "$line" ]] || { echo "FATAL: no line $idx in $PROSPER_PARAMS" >&2; exit 3; }
    # shellcheck disable=SC2206
    set -- $line
fi

echo "[worker] host=$(hostname) job=${LSB_JOBID:-none} idx=${LSB_JOBINDEX:-none}"
echo "[worker] slots=${LSB_DJOB_NUMPROC:-?} python=$PYTHON"
echo "[worker] module=$MODULE args=$*"
cd "$PROSPER_ROOT"
exec "$PYTHON" -m "$MODULE" "$@"
