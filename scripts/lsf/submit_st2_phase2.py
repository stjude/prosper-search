#!/usr/bin/env python3
"""ST2.0 phase 2 (spec 0002 record, [analysis plan]; user go 2026-09-16): tumour-only retrieval
cohort-wide on the three external cohorts -- CCDI in-scope (765), STV labelled (771; 691
in-scope + 80 OOS), KidRare in-scope (1,227) -- against the COMET UNI2 library, exact-OT
cosine (the frozen primary).

Two arms, both with the query restricted to predicted-Tumor mosaic tiles (five-class head,
published proba-argmax rule, as phase 1):
  tumor_only       query tumour-only vs the FULL 2,921-slide library   (the registered ST2.0 design; phase-1 arm)
  tumor_only_both  query tumour-only vs the tumour-only library         (the Table S5 / internal `COMET_TO` configuration;
                                                                        evaluated with NORMAL dropped, like Table S5)
Stages (each chained on ended() of the previous):
  stage    symlink the query pickles per cohort under $PROSPER_OUT/st2p2/<cohort>/features_src (login node)
  score    tumor_tiles_cli score, arrays of --chunk slides (queries + the 2,921 library pickles)
  filter   tumor_tiles_cli filter --arms tumor_only [--skip-empty for the library]
  sim      similarity_cli OT, one query row per array element, spread over the quiet queues
  combine  combine_similarity_cli --external  -> similarity_matrix/combined_similarity_matrix_st2p2_<cohort>_<arm>_ot_cosine.pkl
  evaluate external_cli with the same manifests / id / patient columns as the whole-slide runs
           -> $PROSPER_OUT/st2p2_external_<cohort>_<arm>/{per_query,ranked_lists}_<cohort>_uni2_ot_cosine.*
  gapfill  (2026-09-22) submit ONLY the similarity rows that are missing on disk for each (cohort, arm) --
           e.g. after a manifest grows (STV v2: +5 slides) or after killed array elements -- then chain
           combine -> evaluate for the (cohort, arm) pairs that had gaps. Avoids re-launching every element
           of `sim` just to have it skip existing rows (each element pays the ~100 s library load).
Usage:  submit_st2_phase2.py all --go        (or a single stage name; without --go prints bsub lines)
        submit_st2_phase2.py stage|score|filter --sets stv     (restrict the login-node / scoring stages to one set)
        submit_st2_phase2.py gapfill --go [--arms tumor_only]
        submit_st2_phase2.py ce --sets kidrare --arms tumor_only --go   (combine -> evaluate for a pair whose rows are complete)
"""
from __future__ import annotations
import argparse, glob, json, math, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from prosper.hpc import submit as S              # noqa: E402
from prosper.evaluation import paths             # noqa: E402

OUT = paths.output_root()
WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
ROOT = OUT / "st2p2"
LIB_FULL = Path("/PROSPER_WORKSPACE/comet-feature-gen/comet-uni2/data/tmp_final")
CLF = Path("/PROSPER_WORKSPACE/comet-annotation/model/"
           "pathologist_annotated_slides_uni2/tumor_detector_sgd_test_fraction_0.95_training.pkl")
CCDI_FEATS = Path("/PROSPER_WORKSPACE/comet-testsets/output/"
                  "COG_CCDI_MCI_ALL/patch_embeddings/uni2/tmp")
COHORTS = {
    "ccdi":    dict(src=CCDI_FEATS,                 manifest=OUT / "external_ccdi/manifest_inscope.csv",   patient_col="slide_id"),
    "stv":     dict(src=OUT / "features/stv_uni2",  manifest=OUT / "external_stv/manifest_labelled.csv",   patient_col="patient_id"),
    "kidrare": dict(src=OUT / "features/kidrare_uni2", manifest=OUT / "external_kidrare/manifest_inscope.csv", patient_col="slide_id"),
}
ARMS = {"tumor_only": dict(lib=LIB_FULL, drop=None, ref_tumor_only=False),
        "tumor_only_both": dict(lib=ROOT / "library/features_in/tumor_only", drop=["MISC", "NORMAL"], ref_tumor_only=True)}
SETS = list(COHORTS) + ["library"]
# Queues measured 2026-09-16 15:00 (bqueues / bhosts): dgx 640 cores / 201 used, large_core 512 / 187,
# rhel88_gpu 384 / 10, heavy_io shares rhel8_cpu (PEND 2,265). priority PEND 6,391 -> not used. Probe jobs
# (hostname + `import ot`) ran on dgx, rhel88_gpu, large_core_count, heavy_io: all RHEL 8.10.
# dgx REMOVED 2026-09-16 15:10: a 1-slot probe without -R ran, but every array element with rusage[mem] pended on
# "resource reservation not satisfied (Resource: ngpus_physical)" -- the queue needs a GPU reservation, and this
# work is CPU-only (POT EMD). User: "run it on any available cpu nodes".
# 2026-09-22 (gap-fill): heavy_io PEND 661 and it placed ~3 elements on 09-16; large_core 0 PEND, rhel88_gpu 0 PEND,
# priority 116 PEND -> heavy_io dropped, priority added for the small gap-fill arrays.
SIM_QUEUES = {"large_core_count": 0.5, "rhel88_gpu": 0.3, "priority": 0.2}
SCORE_QUEUES = ["large_core_count"]
# Phase-1 OT epilogue (job 319335986, 22 rows): Max Memory 27,711 MB, CPU/Run 0.99 x 8 threads, 976 s/row.
SIM_EPI = {"max_mem_mb": 27711.0, "effective_cores": 8.0}

def ids_of(cohort: str) -> list:
    import pandas as pd
    return pd.read_csv(COHORTS[cohort]["manifest"]).slide_id.astype(str).tolist()

def src_dir(s: str) -> Path:
    return LIB_FULL if s == "library" else ROOT / s / "features_src"

def n_pkl(d: Path) -> int:
    return len(glob.glob(str(d / "*.pkl")))

def _dep(ids, kind="ended"):
    ids = [i for i in ids if i]
    return " && ".join(f"{kind}({i})" for i in ids) or None

# ---------------------------------------------------------------- stage
def stage(sets=tuple(SETS)) -> dict:
    rec = {}
    for c, cfg in COHORTS.items():
        if c not in sets: continue
        d = ROOT / c / "features_src"; d.mkdir(parents=True, exist_ok=True)
        ids = ids_of(c); missing = []
        for i in ids:
            s = cfg["src"] / f"{i}.pkl"; t = d / f"{i}.pkl"
            if not s.exists(): missing.append(i); continue
            if not t.exists(): t.symlink_to(s)
        rec[c] = {"n_manifest": len(ids), "n_linked": n_pkl(d), "missing": missing}
        print(f"[stage] {c}: {len(ids)} manifest ids, {n_pkl(d)} linked, {len(missing)} missing")
    rec["library"] = {"n": n_pkl(LIB_FULL), "src": str(LIB_FULL)}
    (ROOT / "_stage_record.json").write_text(json.dumps(rec, indent=1))
    return rec

# ---------------------------------------------------------------- score
def score(go: bool, chunk: int = 100, sets=tuple(SETS)) -> dict:
    jids = {}
    for s in sets:
        src = src_dir(s); n = n_pkl(src); n_elem = math.ceil(n / chunk)
        params = ROOT / s / "score_params.txt"; params.parent.mkdir(parents=True, exist_ok=True)
        params.write_text("\n".join(
            f"score --features-dir {src} --classifier {CLF} --out-dir {ROOT / s / 'scores'} "
            f"--lo {i*chunk+1} --hi {min(n, (i+1)*chunk)}" for i in range(n_elem)) + "\n")
        res = S.Resources(slots=1, mem_mb_per_slot=6144, wall_minutes=120)
        jids[s] = S.fan_out(f"PROSPER_MODULE=prosper.preprocessing.tumor_tiles_cli PROSPER_PARAMS={params} bash {WORKER}",
                            n=n_elem, job_name=f"st2p2_score_{s}", res=res, log_dir=OUT / "logs" / f"st2p2_score_{s}",
                            queues=SCORE_QUEUES, canary_epilogue={"max_mem_mb": 2500.0, "effective_cores": 1.0}, dry_run=not go)
    return jids

# ---------------------------------------------------------------- filter
def filt(go: bool, after: dict, sets=tuple(SETS)) -> dict:
    jids = {}
    for s in sets:
        src = src_dir(s)
        skip = " --skip-empty"       # a query with no predicted-Tumor tile leaves the arm (recorded), it does not abort the cohort
        res = S.Resources(slots=1, mem_mb_per_slot=12288, wall_minutes=240)
        jids[s] = S.one(f"PROSPER_MODULE=prosper.preprocessing.tumor_tiles_cli bash {WORKER} filter "
                        f"--features-dir {src} --scores-dir {ROOT / s / 'scores'} --out-root {ROOT / s / 'features_in'} "
                        f"--arms tumor_only --min-tiles 8{skip}",
                        job_name=f"st2p2_filter_{s}", res=res, queue="large_core_count",
                        log_dir=OUT / "logs" / "st2p2_filter", dry_run=not go, dependency=_dep(after.get(s, [])))
    return jids

# ---------------------------------------------------------------- similarity (one query row per element)
def _weighted_arrays(cmd: str, n: int, job_name: str, res: S.Resources, log_dir: Path, dry_run: bool, dependency):
    """Split 1..n across SIM_QUEUES by weight (fan_out splits evenly; heavy_io must get less)."""
    res = S.right_size(res, SIM_EPI)
    print(f"[sim] {job_name}: {n} rows, -n {res.slots} -M {res.mem_mb_per_slot} ({res.total_reserved_gb:.1f} GB), "
          f"queues {SIM_QUEUES}")
    start, jids = 1, []
    items = list(SIM_QUEUES.items())
    for i, (q, w) in enumerate(items):
        end = n if i == len(items) - 1 else min(n, start + max(1, round(n * w)) - 1)
        if start > end: break
        for lo, hi in S.chunk_array(end - start + 1, S.MAX_JOB_ARRAY_SIZE):
            jids.append(S._bsub(cmd, job_name=job_name, queue=q, res=res, log_dir=log_dir,
                                array=f"{start+lo-1}-{start+hi-1}", dry_run=dry_run, dependency=dependency))
        start = end + 1
    return jids

def n_queries(c: str) -> int:
    """Rows to compute = tumour-only query pickles once `filter` has run (a query with 0 tumour tiles is skipped)."""
    fi = ROOT / c / "features_in" / "tumor_only"
    return n_pkl(fi) if fi.exists() and n_pkl(fi) else n_pkl(src_dir(c))

def sim(go: bool, after: dict, arms=tuple(ARMS), batch: int = 2) -> dict:
    jids = {}
    for arm in arms:
        for c in COHORTS:
            feats = ROOT / c / "features_in" / "tumor_only"
            n = n_queries(c); n_elem = math.ceil(n / batch)
            task = f"similarity_st2p2_{c}_{arm}_ot"
            params = ROOT / c / f"sim_params_{arm}.txt"
            # `batch` rows per element: the 27.7 GB library load (~100 s) is paid once per element; rows are
            # resumable (an existing similarity_<qid>.pkl is skipped), so a re-submission only fills gaps.
            params.write_text("\n".join(
                f"--encoder uni2 --method ot --features {feats} --library-features {ARMS[arm]['lib']} --feature-dim 1536 "
                f"--lo {i*batch+1} --hi {min(n, (i+1)*batch)} --task {task} --num-workers 8" for i in range(n_elem)) + "\n")
            res = S.Resources(slots=8, mem_mb_per_slot=40960 // 8, wall_minutes=600)
            deps = [after.get(c)] + ([after.get("library")] if arm == "tumor_only_both" else [])
            jids[(c, arm)] = _weighted_arrays(
                f"PROSPER_MODULE=prosper.embeddings.similarity_cli PROSPER_PARAMS={params} bash {WORKER}", n_elem,
                f"st2p2_sim_{c}_{arm}", res, OUT / "logs" / f"st2p2_sim_{c}_{arm}", not go, _dep(deps))
    return jids

# ---------------------------------------------------------------- gap-fill (missing rows only)
def missing_rows(c: str, arm: str) -> tuple[list, list]:
    """(sorted query ids in the tumour-only features dir, 1-based indices of those with no similarity row)."""
    feats = ROOT / c / "features_in" / "tumor_only"
    ids = [Path(f).stem for f in sorted(glob.glob(str(feats / "*.pkl")))]
    have = {Path(f).stem[len("similarity_"):] for f in glob.glob(str(OUT / f"similarity_st2p2_{c}_{arm}_ot" / "similarity_*.pkl"))}
    return ids, [i + 1 for i, q in enumerate(ids) if q not in have]

def gapfill(go: bool, arms=tuple(ARMS), batch: int = 2) -> dict:
    jids = {}
    for arm in arms:
        for c in COHORTS:
            ids, miss = missing_rows(c, arm)
            print(f"[gapfill] {c}/{arm}: {len(ids)} tumour-only queries, {len(miss)} rows missing")
            if not miss:
                continue
            # consecutive indices share an element (one library load), at most `batch` rows per element
            runs, run = [], [miss[0]]
            for i in miss[1:]:
                if i == run[-1] + 1 and len(run) < batch: run.append(i)
                else: runs.append(run); run = [i]
            runs.append(run)
            feats = ROOT / c / "features_in" / "tumor_only"
            task = f"similarity_st2p2_{c}_{arm}_ot"
            params = ROOT / c / f"sim_params_{arm}_gapfill.txt"
            params.write_text("\n".join(
                f"--encoder uni2 --method ot --features {feats} --library-features {ARMS[arm]['lib']} --feature-dim 1536 "
                f"--lo {r[0]} --hi {r[-1]} --task {task} --num-workers 8" for r in runs) + "\n")
            res = S.Resources(slots=8, mem_mb_per_slot=40960 // 8, wall_minutes=600)
            jids[(c, arm)] = _weighted_arrays(
                f"PROSPER_MODULE=prosper.embeddings.similarity_cli PROSPER_PARAMS={params} bash {WORKER}", len(runs),
                f"st2p2_gap_{c}_{arm}", res, OUT / "logs" / f"st2p2_sim_{c}_{arm}", not go, None)
    return jids

# ---------------------------------------------------------------- combine + evaluate
def combine(go: bool, after: dict, arms=tuple(ARMS), pairs=None) -> dict:
    jids = {}
    for arm in arms:
        for c in COHORTS:
            if pairs is not None and (c, arm) not in pairs: continue
            n = n_queries(c); task = f"similarity_st2p2_{c}_{arm}_ot"
            res = S.Resources(slots=4, mem_mb_per_slot=8192, wall_minutes=120)
            jids[(c, arm)] = S.one(f"PROSPER_MODULE=prosper.embeddings.combine_similarity_cli bash {WORKER} "
                                   f"--encoder uni2 --method ot --similarity-dir {OUT / task} "
                                   f"--matrix-model st2p2_{c}_{arm} --external --min-slides {n}",
                                   job_name=f"st2p2_combine_{c}_{arm}", res=res, queue="large_core_count",
                                   log_dir=OUT / "logs" / "combine_similarity", dry_run=not go,
                                   dependency=_dep(after.get((c, arm), [])))
    return jids

def evaluate(go: bool, after: dict, arms=tuple(ARMS), pairs=None) -> dict:
    jids = {}
    for arm in arms:
        for c, cfg in COHORTS.items():
            if pairs is not None and (c, arm) not in pairs: continue
            matrix = OUT / "similarity_matrix" / f"combined_similarity_matrix_st2p2_{c}_{arm}_ot_cosine.pkl"
            ref = paths.similarity_matrix("uni2", "ot_cosine", tumor_only=ARMS[arm]["ref_tumor_only"], no_same_patient=True)
            drop = f" --library-drop-labels {' '.join(ARMS[arm]['drop'])}" if ARMS[arm]["drop"] else ""
            res = S.Resources(slots=4, mem_mb_per_slot=8192, wall_minutes=240)
            jids[(c, arm)] = S.one(f"PROSPER_MODULE=prosper.evaluation.external_cli bash {WORKER} "
                                   f"--matrix {matrix} --cohort {c} --model uni2 --method ot_cosine "
                                   f"--internal-reference-matrix {ref} --task st2p2_external_{c}_{arm} "
                                   f"--query-metadata {cfg['manifest']} --query-id-col slide_id --query-label-col tumor_group "
                                   f"--query-patient-col {cfg['patient_col']}{drop}",
                                   job_name=f"st2p2_eval_{c}_{arm}", res=res, queue="large_core_count",
                                   log_dir=OUT / "logs" / f"st2p2_external_{c}", dry_run=not go,
                                   dependency=_dep([after.get((c, arm))]))
    return jids

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage", "score", "filter", "sim", "combine", "evaluate", "ce", "rest", "all", "gapfill"])
    ap.add_argument("--go", action="store_true"); ap.add_argument("--after", nargs="*", default=[])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--sets", default=",".join(SETS), help="restrict stage/score/filter to these sets (comma list)")
    a = ap.parse_args(); go = a.go; arms = tuple(a.arms.split(",")); sets = tuple(a.sets.split(","))
    ROOT.mkdir(parents=True, exist_ok=True)
    rec = {"go": go, "arms": arms, "sets": sets}
    if a.stage in ("stage", "all"): rec["stage"] = stage(sets)
    if a.stage == "score": rec["score"] = score(go, sets=sets)
    if a.stage == "filter": rec["filter"] = filt(go, {s: a.after for s in SETS}, sets=sets)
    if a.stage == "gapfill":         # missing rows only, then combine -> evaluate for the pairs that had gaps
        gf = gapfill(go, arms); rec["gapfill"] = {f"{k[0]}_{k[1]}": v for k, v in gf.items()}
        pairs = list(gf)
        co = combine(go, gf, arms, pairs=pairs); rec["combine"] = {f"{k[0]}_{k[1]}": v for k, v in co.items()}
        ev = evaluate(go, co, arms, pairs=pairs); rec["evaluate"] = {f"{k[0]}_{k[1]}": v for k, v in ev.items()}
    if a.stage == "sim": rec["sim"] = {f"{k[0]}_{k[1]}": v for k, v in sim(go, {s: _dep(a.after) for s in SETS}, arms).items()}
    pairs = [(c, arm) for c in sets if c in COHORTS for arm in arms]
    if a.stage == "combine": rec["combine"] = {f"{k[0]}_{k[1]}": v for k, v in combine(go, {}, arms, pairs=pairs).items()}
    if a.stage == "evaluate": rec["evaluate"] = {f"{k[0]}_{k[1]}": v for k, v in evaluate(go, {}, arms, pairs=pairs).items()}
    if a.stage == "ce":              # combine -> evaluate chained, for --sets x --arms (rows already complete on disk)
        co = combine(go, {}, arms, pairs=pairs); rec["combine"] = {f"{k[0]}_{k[1]}": v for k, v in co.items()}
        ev = evaluate(go, co, arms, pairs=pairs); rec["evaluate"] = {f"{k[0]}_{k[1]}": v for k, v in ev.items()}
    if a.stage == "rest":            # after `filter` has finished: sim -> combine -> evaluate, chained
        si = sim(go, {}, arms); rec["sim"] = {f"{k[0]}_{k[1]}": v for k, v in si.items()}
        co = combine(go, si, arms); rec["combine"] = {f"{k[0]}_{k[1]}": v for k, v in co.items()}
        ev = evaluate(go, co, arms); rec["evaluate"] = {f"{k[0]}_{k[1]}": v for k, v in ev.items()}
    if a.stage == "all":
        sc = score(go); rec["score"] = sc
        fl = filt(go, sc); rec["filter"] = fl
        si = sim(go, fl, arms); rec["sim"] = {f"{k[0]}_{k[1]}": v for k, v in si.items()}
        co = combine(go, si, arms); rec["combine"] = {f"{k[0]}_{k[1]}": v for k, v in co.items()}
        ev = evaluate(go, co, arms); rec["evaluate"] = {f"{k[0]}_{k[1]}": v for k, v in ev.items()}
    if go:
        (ROOT / f"_submission_{a.stage}.json").write_text(json.dumps(rec, indent=1, default=str))
    return 0

if __name__ == "__main__":
    sys.exit(main())
