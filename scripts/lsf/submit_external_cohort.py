#!/usr/bin/env python3
"""C12: process an EXTERNAL cohort end to end against the COMET library.

First target: ``solid_tumor_validation`` ("stv") -- 1,247 generic-TIFF WSIs from
a scanner outside COMET (parent tree holds ``example_philips_slides``; vendor
tag is ``generic-tiff`` with 2x pyramid steps and no MPP in the header). It
answers R1.3 (institution/scanner generalisation) and R2.7 (real, not
simulated, scanner variability) -- once a label manifest exists. Until then
every step below is label-free and the evaluation reports retrieval behaviour
(predicted-label distribution, vote share, vote entropy, distance shift vs
COMET's own queries), not accuracy.

Encoder / aggregation: the frozen primary (UNI2 + exact OT, cosine) plus the
three cheap aggregations on the same features (median pooling, SAMPLER,
Yottixel/BoB), all against the PUBLISHED UNI2 mosaic features of the 2,921
COMET slides so the library is byte-identical to Table S3's.

Assumptions stated once, to be confirmed by the cohort owner:
  * level 0 is ~0.25 um/px (40x-equivalent), so 512 px tiles at level 0 are
    the same physical footprint as COMET's;
  * one TIFF = one slide; the filename stem is the slide id.

Stages (each a canary-then-fan-out or a chained single job):
  headers-canary / headers-fanout / headers-merge     scanner facts (S1a-style)
  mosaic-canary / mosaic-fanout / mosaic-combine       15% mosaic, TIFF-faithful
  extract-canary / extract-fanout                      UNI2 tile features (GPU)
  aggregate                                            median + SAMPLER vs library
  sim-canary / sim-fanout / sim-combine  --method ot|bob
  evaluate [--query-metadata CSV]                      external_cli x 4 matrices
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prosper.hpc import submit as S             # noqa: E402
from prosper.evaluation import paths            # noqa: E402

WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
OUT = paths.output_root()

LIB_FEATURES = Path("/PROSPER_WORKSPACE/"
                    "comet-feature-gen/comet-uni2/data/tmp_final")
ENCODER, DIM = "uni2", 1536
GPU_SLOTS, GPU_WORKERS = 16, 14

# Cohorts. Every stage enumerates ``<slides_dir>/*<ext>`` (flat, sorted), so a
# cohort that lives in sub-folders (KidRare: one folder per entity) is staged as
# a flat directory of symlinks first (``scripts/stage_kidrare.py``); the folder
# label goes into the manifest, not the pipeline.
COHORTS = {
    "stv": dict(slides_dir=Path("/PROSPER_GROUP_DATA/comet/common/solid_tumor_validation"),
                ext=".tiff"),
    # C13, 2026-08-30. 15-Gpx slides whose 16x coarsest level is ~60 Mpx: read the
    # thumbnail directly (one region read) instead of tile-wise (58k NFS reads;
    # the first canary sat at 7% CPU for 100 min on its first slide).
    # Their share costs ~0.35 s per random 64 KB read (sequential 59 MB/s), so
    # every slide is copied to node-local /scratch_local (242 GB) first.
    "kidrare": dict(slides_dir=OUT / "slides_kidrare", ext=".tiff",
                    mosaic_args=" --thumb-direct-max-px 120000000 --stage-local",
                    extract_args=" --stage-local",
                    env="PROSPER_STAGE_DIR=/scratch_local "),
    # C3 (2026-08-30): CCDI-MCI. UNI2 mosaic-tile features for 4,160 slides already
    # exist in the per-slide format the aggregation/similarity stages read, so the
    # pipeline starts at `aggregate`; headers/mosaic/extract are refused.
    "ccdi": dict(slides_dir=None, ext=".svs",
                 features_dir=Path("/PROSPER_WORKSPACE/"
                                   "comet-testsets/output/COG_CCDI_MCI_ALL/patch_embeddings/uni2/tmp")),
}


def _cohort_from_argv() -> str:
    for i, tok in enumerate(sys.argv):
        if tok == "--cohort" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if tok.startswith("--cohort="):
            return tok.split("=", 1)[1]
    return "stv"


COHORT = _cohort_from_argv()
if COHORT not in COHORTS:
    raise SystemExit(f"unknown --cohort {COHORT}; known: {sorted(COHORTS)}")
SLIDES_DIR = COHORTS[COHORT]["slides_dir"]
EXT = COHORTS[COHORT]["ext"]
MOSAIC_TASK = f"mosaic/{COHORT}"
MOSAIC_PKL = OUT / MOSAIC_TASK / "combined_mosaics.pkl"
FEAT_TASK = f"features/{COHORT}_{ENCODER}"
FEATURES = COHORTS[COHORT].get("features_dir") or (OUT / FEAT_TASK)
MOSAIC_ARGS = COHORTS[COHORT].get("mosaic_args", "")
EXTRACT_ARGS = COHORTS[COHORT].get("extract_args", "")
MATRIX_MODEL = f"{COHORT}_{ENCODER}"
ENV = f"{COHORTS[COHORT].get('env', '')}COMET_SLIDES_ROOT={SLIDES_DIR} COMET_SLIDES_EXT={EXT} "
SLIDE_STAGES = ("headers-canary", "headers-fanout", "headers-merge", "mosaic-canary",
                "mosaic-fanout", "mosaic-combine", "extract-canary", "extract-fanout")


def _n_slides() -> int:
    return len(glob.glob(str(SLIDES_DIR / f"*{EXT}")))


def _dep(ids, kind="ended"):
    ids = [i for i in ids if i]
    return " && ".join(f"{kind}({i})" for i in ids) or None


def _opt(**kw):
    """Only the CLI overrides that were given, so stage defaults stay in force."""
    return {k: v for k, v in kw.items() if v is not None}


# ---------------------------------------------------------------- headers
def headers_canary(go):
    res = S.Resources(slots=1, mem_mb_per_slot=2048, wall_minutes=20)
    return S.canary(f"PROSPER_MODULE=prosper.evaluation.slide_headers bash {WORKER} extract "
                    f"--slides-dir {SLIDES_DIR} --ext {EXT} --start 1 --end 25 "
                    f"--task headers_{COHORT}",
                    job_name=f"prosper_headers_{COHORT}", res=res, kind="cpu",
                    log_dir=OUT / "logs" / f"headers_{COHORT}", dry_run=not go)


def headers_fanout(epilogue, go, batch=25):
    epi = S.parse_epilogue(epilogue.read_text())
    n = _n_slides()
    ranges = [(s + 1, min(s + batch, n)) for s in range(batch, n, batch)]  # canary did 1..25
    params = OUT / f"headers_{COHORT}" / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("\n".join(f"extract --slides-dir {SLIDES_DIR} --ext {EXT} --start {lo} "
                                f"--end {hi} --task headers_{COHORT}" for lo, hi in ranges) + "\n")
    res = S.Resources(slots=1, mem_mb_per_slot=2048, wall_minutes=20)
    return S.fan_out(f"PROSPER_MODULE=prosper.evaluation.slide_headers PROSPER_PARAMS={params} "
                     f"bash {WORKER}", n=len(ranges), job_name=f"prosper_headers_{COHORT}",
                     res=res, kind="cpu", log_dir=OUT / "logs" / f"headers_{COHORT}",
                     n_queues=2, canary_epilogue=epi, dry_run=not go)


def headers_merge(after, go):
    res = S.Resources(slots=1, mem_mb_per_slot=4096, wall_minutes=20)
    return S.one(f"PROSPER_MODULE=prosper.evaluation.slide_headers bash {WORKER} merge "
                 f"--slides-dir {SLIDES_DIR} --ext {EXT} --task headers_{COHORT}",
                 job_name=f"prosper_headers_{COHORT}_merge", res=res, kind="cpu",
                 log_dir=OUT / "logs" / f"headers_{COHORT}", dry_run=not go,
                 dependency=_dep(after))


def headers_retry(go, wall=60, batch=25):
    """Re-run the header ranges the merge reported missing (TERM_RUNLIMIT on
    KidRare: 25 multi-GB files per element did not fit a 20-min wall)."""
    import json
    summ = OUT / f"headers_{COHORT}" / "merge_summary.json"
    missing = set(json.loads(paths.require(summ, "merge summary").read_text())["missing_slide_ids"])
    ids = sorted(Path(p).name[:-len(EXT)] for p in glob.glob(str(SLIDES_DIR / f"*{EXT}")))
    idx = sorted(i + 1 for i, s in enumerate(ids) if s in missing)
    if not idx:
        raise SystemExit("nothing missing")
    ranges, lo = [], idx[0]
    for a_, b_ in zip(idx, idx[1:] + [None]):
        if b_ is None or b_ != a_ + 1 or a_ - lo + 1 >= batch:
            ranges.append((lo, a_))
            lo = b_
    params = OUT / f"headers_{COHORT}" / "params_retry.txt"
    params.write_text("\n".join(f"extract --slides-dir {SLIDES_DIR} --ext {EXT} --start {lo} "
                                f"--end {hi} --task headers_{COHORT}" for lo, hi in ranges) + "\n")
    print(f"[headers-retry] {len(idx)} missing slides -> {len(ranges)} ranges, wall {wall} min")
    res = S.Resources(slots=1, mem_mb_per_slot=2048, wall_minutes=wall)
    return S.fan_out(f"PROSPER_MODULE=prosper.evaluation.slide_headers PROSPER_PARAMS={params} "
                     f"bash {WORKER}", n=len(ranges), job_name=f"prosper_headers_{COHORT}_retry",
                     res=res, kind="cpu", log_dir=OUT / "logs" / f"headers_{COHORT}",
                     n_queues=2, canary_epilogue={"wall_s": 1200, "max_mem_mb": 420, "effective_cores": 1.0},
                     dry_run=not go)


# ---------------------------------------------------------------- mosaic
def mosaic_canary(go, n=2, wall=120):
    res = S.Resources(slots=1, mem_mb_per_slot=8192, wall_minutes=wall)
    return S.canary(f"{ENV}PROSPER_MODULE=prosper.preprocessing.mosaic_cli bash {WORKER} extract "
                    f"--slides-dir {SLIDES_DIR} --ext {EXT} --lo 1 --hi {n} --task {MOSAIC_TASK}{MOSAIC_ARGS}",
                    job_name=f"prosper_mosaic_{COHORT}", res=res, kind="cpu",
                    log_dir=OUT / "logs" / f"mosaic_{COHORT}", dry_run=not go)


def mosaic_fanout(epilogue, go, batch=2, wall=120):
    epi = S.parse_epilogue(epilogue.read_text())
    if not epi:
        raise SystemExit(f"no epilogue in {epilogue}")
    n = _n_slides()
    n_elem = math.ceil(n / batch)
    params = OUT / MOSAIC_TASK / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("\n".join(
        f"extract --slides-dir {SLIDES_DIR} --ext {EXT} --lo {i*batch+1} "
        f"--hi {min(n,(i+1)*batch)} --task {MOSAIC_TASK}{MOSAIC_ARGS}" for i in range(n_elem)) + "\n")
    res = S.Resources(slots=1, mem_mb_per_slot=8192, wall_minutes=wall)
    return S.fan_out(f"{ENV}PROSPER_MODULE=prosper.preprocessing.mosaic_cli PROSPER_PARAMS={params} "
                     f"bash {WORKER}", n=n_elem, job_name=f"prosper_mosaic_{COHORT}", res=res,
                     kind="cpu", log_dir=OUT / "logs" / f"mosaic_{COHORT}", n_queues=2,
                     canary_epilogue=epi, dry_run=not go)


def mosaic_combine(after, go, min_slides):
    res = S.Resources(slots=1, mem_mb_per_slot=8192, wall_minutes=30)
    return S.one(f"PROSPER_MODULE=prosper.preprocessing.mosaic_cli bash {WORKER} combine "
                 f"--task {MOSAIC_TASK} --ext {EXT} --min-slides {min_slides}",
                 job_name=f"prosper_mosaic_{COHORT}_combine", res=res, kind="cpu",
                 log_dir=OUT / "logs" / f"mosaic_{COHORT}", dry_run=not go,
                 dependency=_dep(after))


# ---------------------------------------------------------------- features
def extract_canary(go, n=4):
    paths.require(MOSAIC_PKL, "combined mosaics")
    res = S.Resources(slots=GPU_SLOTS, mem_mb_per_slot=4096, wall_minutes=28, gpus=1)
    return S.canary(f"{ENV}PROSPER_MODULE=prosper.embeddings.extract_cli bash {WORKER} "
                    f"--encoder {ENCODER} --mosaic {MOSAIC_PKL} --lo 1 --hi {n} "
                    f"--task {FEAT_TASK} --num-workers {GPU_WORKERS} --batch-size 256{EXTRACT_ARGS}",
                    job_name=f"prosper_extract_{COHORT}", res=res, kind="gpu",
                    log_dir=OUT / "logs" / f"extract_{COHORT}", dry_run=not go)


def extract_fanout(epilogue, go, batch=30, wall=600, n_queues=2):
    import pickle
    epi = S.parse_epilogue(epilogue.read_text())
    if not epi:
        raise SystemExit(f"no epilogue in {epilogue}")
    with open(MOSAIC_PKL, "rb") as fh:
        n = len(pickle.load(fh))
    n_elem = math.ceil(n / batch)
    params = OUT / f"features_{COHORT}_{ENCODER}" / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("\n".join(
        f"--encoder {ENCODER} --mosaic {MOSAIC_PKL} --lo {i*batch+1} --hi {min(n,(i+1)*batch)} "
        f"--task {FEAT_TASK} --num-workers {GPU_WORKERS} --batch-size 256{EXTRACT_ARGS}"
        for i in range(n_elem)) + "\n")
    peak = epi.get("max_mem_mb") or 4096
    mem = max(2048, int(peak * 2.0 / GPU_SLOTS) + 1)
    res = S.Resources(slots=GPU_SLOTS, mem_mb_per_slot=mem, wall_minutes=wall, gpus=1)
    epi = dict(epi)
    epi["effective_cores"] = float(GPU_SLOTS)      # I/O-bound workers; see submit_extract.py
    return S.fan_out(f"{ENV}PROSPER_MODULE=prosper.embeddings.extract_cli PROSPER_PARAMS={params} "
                     f"bash {WORKER}", n=n_elem, job_name=f"prosper_extract_{COHORT}", res=res,
                     kind="gpu", log_dir=OUT / "logs" / f"extract_{COHORT}", n_queues=n_queues,
                     canary_epilogue=epi, dry_run=not go)


# ---------------------------------------------------------------- aggregation + similarity
def aggregate(after, go, min_slides):
    res = S.Resources(slots=8, mem_mb_per_slot=4096, wall_minutes=120)
    jids = []
    for m in ("median_pooling", "sampler"):
        jids.append(S.one(f"PROSPER_MODULE=prosper.embeddings.aggregate_cli bash {WORKER} "
                          f"--encoder {ENCODER} --features {FEATURES} --library-features {LIB_FEATURES} "
                          f"--matrix-model {MATRIX_MODEL} --method {m} --metric cityblock "
                          f"--min-slides {min_slides}",
                          job_name=f"prosper_agg_{COHORT}_{m}", res=res, kind="cpu",
                          log_dir=OUT / "logs" / "aggregate", dry_run=not go,
                          dependency=_dep(after)))
    return jids


def _sim_task(method):
    return f"similarity/{MATRIX_MODEL}_{method}"


def sim_canary(method, go, n=3):
    res = S.Resources(slots=8 if method == "ot" else 1,
                      mem_mb_per_slot=(40960 // 8) if method == "ot" else 40960,
                      wall_minutes=600 if method == "ot" else 240)
    workers = " --num-workers 8" if method == "ot" else ""
    return S.canary(f"PROSPER_MODULE=prosper.embeddings.similarity_cli bash {WORKER} "
                    f"--encoder {ENCODER} --method {method} --features {FEATURES} "
                    f"--library-features {LIB_FEATURES} --feature-dim {DIM} --lo 1 --hi {n} "
                    f"--task {_sim_task(method)}{workers}",
                    job_name=f"prosper_sim_{MATRIX_MODEL}_{method}", res=res, kind="cpu",
                    log_dir=OUT / "logs" / f"similarity_{MATRIX_MODEL}_{method}", dry_run=not go)


def sim_fanout(method, epilogue, go, batch=None, wall=None, n_queues=3):
    epi = S.parse_epilogue(epilogue.read_text())
    if not epi:
        raise SystemExit(f"no epilogue in {epilogue}")
    n = len(glob.glob(str(FEATURES / "*.pkl")))
    ot = method == "ot"
    batch = batch or (1 if ot else 20)
    wall = wall or (240 if ot else 720)
    n_elem = math.ceil(n / batch)
    # OT: 8 threads per row (POT releases the GIL; canary CPU/wall = 7.0), one
    # row per element so no large slide can blow a shared wall limit.
    slots = 8 if ot else 1
    workers = " --num-workers 8" if ot else ""
    params = OUT / f"similarity_{MATRIX_MODEL}_{method}" / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("\n".join(
        f"--encoder {ENCODER} --method {method} --features {FEATURES} "
        f"--library-features {LIB_FEATURES} --feature-dim {DIM} --lo {i*batch+1} "
        f"--hi {min(n,(i+1)*batch)} --task {_sim_task(method)}{workers}" for i in range(n_elem)) + "\n")
    res = S.Resources(slots=slots, mem_mb_per_slot=40960 // slots, wall_minutes=wall)
    epi = dict(epi)
    if ot:
        epi["effective_cores"] = 8.0        # pin: --num-workers is baked into params
    return S.fan_out(f"PROSPER_MODULE=prosper.embeddings.similarity_cli PROSPER_PARAMS={params} "
                     f"bash {WORKER}", n=n_elem, job_name=f"prosper_sim_{MATRIX_MODEL}_{method}",
                     res=res, kind="cpu", log_dir=OUT / "logs" / f"similarity_{MATRIX_MODEL}_{method}",
                     n_queues=n_queues, canary_epilogue=epi, dry_run=not go)


def sim_combine(method, after, go, min_slides):
    res = S.Resources(slots=4, mem_mb_per_slot=8192, wall_minutes=60)
    return S.one(f"PROSPER_MODULE=prosper.embeddings.combine_similarity_cli bash {WORKER} "
                 f"--encoder {ENCODER} --method {method} --similarity-dir {OUT}/{_sim_task(method)} "
                 f"--matrix-model {MATRIX_MODEL} --external --min-slides {min_slides}",
                 job_name=f"prosper_combine_{MATRIX_MODEL}_{method}", res=res, kind="cpu",
                 log_dir=OUT / "logs" / "combine_similarity", dry_run=not go,
                 dependency=_dep(after))


# ---------------------------------------------------------------- evaluation
def evaluate(after, go, query_metadata=None, id_col="slide_id", label_col="tumor_group",
             patient_col=None, methods=("ot_cosine", "yottixel", "median_pooling", "sampler"),
             task_suffix=""):
    res = S.Resources(slots=4, mem_mb_per_slot=8192, wall_minutes=240)
    jids = []
    task = f"external_{COHORT}{task_suffix}"
    for m in methods:
        matrix = OUT / "similarity_matrix" / f"combined_similarity_matrix_{MATRIX_MODEL}_{m}.pkl"
        ref = paths.similarity_matrix(ENCODER, m, no_same_patient=True)
        qm = (f" --query-metadata {query_metadata} --query-id-col {id_col} "
              f"--query-label-col {label_col}" if query_metadata else "")
        if query_metadata and patient_col:
            qm += f" --query-patient-col {patient_col}"
        jids.append(S.one(f"PROSPER_MODULE=prosper.evaluation.external_cli bash {WORKER} "
                          f"--matrix {matrix} --cohort {COHORT} --model {ENCODER} --method {m} "
                          f"--internal-reference-matrix {ref} --task {task}{qm}",
                          job_name=f"prosper_{task}_{m}", res=res, kind="cpu",
                          log_dir=OUT / "logs" / f"external_{COHORT}", dry_run=not go,
                          dependency=_dep(after)))
    return jids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=[
        "headers-canary", "headers-fanout", "headers-merge", "headers-retry",
        "mosaic-canary", "mosaic-fanout", "mosaic-combine",
        "extract-canary", "extract-fanout", "aggregate",
        "sim-canary", "sim-fanout", "sim-combine", "evaluate"])
    ap.add_argument("--cohort", choices=sorted(COHORTS), default="stv",
                    help="which external cohort (read at import; every task/matrix name carries it)")
    ap.add_argument("--method", choices=["ot", "bob"], default="ot")
    ap.add_argument("--epilogue", type=Path)
    ap.add_argument("--after", default="")
    ap.add_argument("--batch", type=int, default=None, help="fan-out: slides per element (stage default if omitted)")
    ap.add_argument("--wall", type=int, default=None, help="fan-out: wall minutes per element (stage default if omitted)")
    ap.add_argument("--task-suffix", default="", help="evaluate: e.g. _inscope so two labelled runs do not collide")
    ap.add_argument("--methods", default="ot_cosine,yottixel,median_pooling,sampler",
                    help="evaluate: comma-separated subset of the four aggregation blocks")
    ap.add_argument("--min-slides", type=int, default=1200)
    ap.add_argument("--query-metadata", default=None)
    ap.add_argument("--query-id-col", default="slide_id")
    ap.add_argument("--query-label-col", default="tumor_group")
    ap.add_argument("--query-patient-col", default=None)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    after = [x for x in a.after.split(",") if x.strip()]
    act = a.action
    need_epi = act.endswith("fanout")
    if act in SLIDE_STAGES and SLIDES_DIR is None:
        raise SystemExit(f"cohort {COHORT} has no slide directory: its features exist already "
                         f"({FEATURES}); start at `aggregate` / `sim-canary`")
    if need_epi and not a.epilogue:
        raise SystemExit(f"{act} needs --epilogue <canary .out>")
    out = {
        "headers-canary": lambda: headers_canary(a.go),
        "headers-fanout": lambda: headers_fanout(a.epilogue, a.go),
        "headers-merge": lambda: headers_merge(after, a.go),
        "headers-retry": lambda: headers_retry(a.go, **_opt(wall=a.wall)),
        "mosaic-canary": lambda: mosaic_canary(a.go, **_opt(wall=a.wall)),
        "mosaic-fanout": lambda: mosaic_fanout(a.epilogue, a.go, **_opt(batch=a.batch, wall=a.wall)),
        "mosaic-combine": lambda: mosaic_combine(after, a.go, a.min_slides),
        "extract-canary": lambda: extract_canary(a.go),
        "extract-fanout": lambda: extract_fanout(a.epilogue, a.go, **_opt(batch=a.batch, wall=a.wall)),
        "aggregate": lambda: aggregate(after, a.go, a.min_slides),
        "sim-canary": lambda: sim_canary(a.method, a.go),
        "sim-fanout": lambda: sim_fanout(a.method, a.epilogue, a.go, **_opt(batch=a.batch, wall=a.wall)),
        "sim-combine": lambda: sim_combine(a.method, after, a.go, a.min_slides),
        "evaluate": lambda: evaluate(after, a.go, a.query_metadata, a.query_id_col,
                                     a.query_label_col, a.query_patient_col,
                                     tuple(x for x in a.methods.split(",") if x),
                                     task_suffix=a.task_suffix),
    }[act]()
    print(out)
    if not a.go:
        print("\n[dry run] nothing submitted. Re-run with --go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
