#!/usr/bin/env python3
"""Follow-up compute from the 2026-08-29 audit (plan v5 register):

  c1e            R4.a  slide-level masked-vs-UNMASKED delta on the primary config
                       (the "report both protocols" transparency that was requested)
  c7b            R2.2  the 5% rung of the tile ladder (nested prefix)
  c7c-bob-canary / c7c-bob-fanout   R2.2  Yottixel/BoB at a tile fraction, all-patches universe
  c7c-ot-canary  R2.2  ONE exact-OT query row at a tile fraction: measures the scaling wall
  combine-ladder R2.2  combine a fraction's OT/BoB rows into a matrix and evaluate it
  x2             R2.3  methylation 10-NN neighbourhood of every library sample vs the
                       H&E failure set (methylation_knn_cli; one CPU job)
  s2             R2.6  tumour-only TITAN: score all-patch UNI2 tiles -> subset CONCH h5 ->
                       TITAN slide encoding (GPU, trident env) -> cosine matrix -> tumour-only
                       voting cells + IR metrics, all chained with -w done()
  s5             R3.6  LLM faithfulness audit (llm_grounding_cli in the comet-llm venv; one CPU
                       job with network egress; --fake for the offline dry run)
  c4             R3.5  ROI-heatmap-vs-annotation localisation (localization_cli; one CPU job)
  paired-ladder  R2.2  paired deltas of every r22_metrics ranked list vs a chosen reference
                       (v3: reference uni2apf15_yottixel -- the set-based rungs and OT-15)

Every action submits through prosper.hpc.submit (canary epilogue required for
a fan-out; -oo/-eo; live-ranked queues; never `standard`).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prosper.hpc import submit as S             # noqa: E402
from prosper.evaluation import paths            # noqa: E402

WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
OUT = paths.output_root()
AP_FEATURES = OUT / "features" / "uni2_all_patches"
EXPECT_AP = 2887
KS = (1, 3, 5, 10)
SLOTS = 8


def _c2_epilogue() -> dict:
    for c in sorted(glob.glob(str(OUT / "logs" / "c2_vote_rules*" / "*canary*.out"))):
        epi = S.parse_epilogue(Path(c).read_text())
        if epi.get("max_mem_mb"):
            epi = dict(epi)
            epi["effective_cores"] = float(SLOTS)
            return epi
    raise SystemExit("no C2 canary epilogue found")


def _dep(ids, kind="ended"):
    ids = [i for i in ids if i]
    return " && ".join(f"{kind}({i})" for i in ids) or None


def c1e(go: bool) -> list:
    lines = [f"--model uni2 --method ot_cosine --topk {k} --tie-break {r} "
             f"--allow-same-patient --iter 5000 --n-jobs {SLOTS} --task c1e_slide_unmasked"
             for r in ("generous_majority", "absolute") for k in KS]
    params = OUT / "c1e_slide_unmasked" / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("\n".join(lines) + "\n")
    res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=30)
    return S.fan_out(f"PROSPER_PARAMS={params} bash {WORKER}", n=len(lines),
                     job_name="prosper_c1e", res=res,
                     log_dir=OUT / "logs" / "c1e_slide_unmasked", n_queues=2,
                     canary_epilogue=_c2_epilogue(), dry_run=not go)


def c7b(go: bool) -> dict:
    """5% rung: aggregate -> 4 voting evals + IR metrics -> paired vs 15% (v2)."""
    model = "uni2apf5"
    agg_res = S.Resources(slots=8, mem_mb_per_slot=4096, wall_minutes=180)
    agg = S.one(f"PROSPER_MODULE=prosper.embeddings.aggregate_cli bash {WORKER} "
                f"--encoder uni2 --features {AP_FEATURES} --matrix-model {model} "
                f"--method median_pooling --metric cityblock --tile-fraction 0.05 "
                f"--min-slides {EXPECT_AP}",
                job_name="prosper_r22_agg_f5", res=agg_res,
                log_dir=OUT / "logs" / "r22_ladder", kind="cpu", dry_run=not go)
    dep = _dep([agg], "done")          # no matrix -> no evaluation, by design
    ev_res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=60)
    evs = []
    for k in KS:
        evs.append(S.one(f"bash {WORKER} --model {model} --method median_pooling --topk {k} "
                         f"--tie-break generous_majority --iter 5000 --n-jobs {SLOTS} "
                         f"--task r22_ladder",
                         job_name=f"prosper_r22_eval_f5_k{k}", res=ev_res,
                         log_dir=OUT / "logs" / "r22_ladder", kind="cpu",
                         dry_run=not go, dependency=dep))
    met = S.one(f"PROSPER_MODULE=prosper.evaluation.metrics_cli bash {WORKER} "
                f"--model {model} --method median_pooling --depth 10 --iter 5000 "
                f"--task r22_metrics",
                job_name="prosper_r22_met_f5", res=ev_res,
                log_dir=OUT / "logs" / "r22_paired", kind="cpu", dry_run=not go,
                dependency=dep)
    pr = []
    for k in KS:
        pr.append(S.one(f"PROSPER_MODULE=prosper.evaluation.paired bash {WORKER} "
                        f"--ranked-dir {OUT}/r22_metrics --reference uni2apf15_median_pooling "
                        f"--k {k} --depth 10 --iter 5000 --task r22_paired_v2",
                        job_name=f"prosper_r22_paired_v2_k{k}", res=ev_res,
                        log_dir=OUT / "logs" / "r22_paired", kind="cpu",
                        dry_run=not go, dependency=_dep([met], "done")))
    return {"aggregate": agg, "evals": evs, "metrics": met, "paired": pr}


def _sim_task(pct: int, method: str) -> str:
    return f"similarity/uni2apf{pct}_{method}"


def c7c_canary(pct: int, method: str, go: bool) -> str:
    frac = pct / 100.0
    n = 3 if method == "bob" else 1
    mem = 40960 if pct <= 15 else 65536
    res = S.Resources(slots=SLOTS if method == "ot" else 1, mem_mb_per_slot=mem // (SLOTS if method == "ot" else 1),
                      wall_minutes=1440 if method == "ot" else 240)
    workers = f" --num-workers {SLOTS}" if method == "ot" else ""
    cmd = (f"PROSPER_MODULE=prosper.embeddings.similarity_cli bash {WORKER} "
           f"--encoder uni2 --method {method} --features {AP_FEATURES} --feature-dim 1536 "
           f"--tile-fraction {frac} --lo 1 --hi {n} --task {_sim_task(pct, method)}{workers}")
    return S.canary(cmd, job_name=f"prosper_sim_uni2apf{pct}_{method}", res=res, kind="cpu",
                    log_dir=OUT / "logs" / f"similarity_uni2apf{pct}_{method}", dry_run=not go)


def c7c_fanout(pct: int, method: str, epilogue: Path, go: bool, batch: int = None) -> list:
    """BoB: 1 slot, 20 rows/element. OT: 8 threads (POT releases the GIL --
    canary at 15%: 472 s/row, CPU/wall = 6.8), ONE row per element so a large
    slide cannot blow a shared wall limit (spec C5 design note)."""
    epi = S.parse_epilogue(epilogue.read_text())
    if not epi:
        raise SystemExit(f"no epilogue in {epilogue}")
    frac = pct / 100.0
    n_slides = len(glob.glob(str(AP_FEATURES / "*.pkl")))
    import math
    ot = method == "ot"
    batch = batch or (1 if ot else 20)
    n_elem = math.ceil(n_slides / batch)
    workers = f" --num-workers {SLOTS}" if ot else ""
    params = OUT / f"similarity_uni2apf{pct}_{method}" / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("\n".join(
        f"--encoder uni2 --method {method} --features {AP_FEATURES} --feature-dim 1536 "
        f"--tile-fraction {frac} --lo {i*batch+1} --hi {min(n_slides,(i+1)*batch)} "
        f"--task {_sim_task(pct, method)}{workers}" for i in range(n_elem)) + "\n")
    slots = SLOTS if ot else 1
    res = S.Resources(slots=slots, mem_mb_per_slot=(40960 if pct <= 15 else 65536) // slots,
                      wall_minutes=240 if ot else 720)
    epi = dict(epi)
    if ot:
        epi["effective_cores"] = float(SLOTS)     # pin: --num-workers is baked into params
    return S.fan_out(f"PROSPER_MODULE=prosper.embeddings.similarity_cli "
                     f"PROSPER_PARAMS={params} bash {WORKER}", n=n_elem,
                     job_name=f"prosper_sim_uni2apf{pct}_{method}", res=res, kind="cpu",
                     log_dir=OUT / "logs" / f"similarity_uni2apf{pct}_{method}",
                     n_queues=2, canary_epilogue=epi, dry_run=not go)


def combine_ladder(pct: int, method: str, after: list, go: bool) -> dict:
    tag = {"ot": "ot_cosine", "bob": "yottixel"}[method]
    model = f"uni2apf{pct}"
    res = S.Resources(slots=4, mem_mb_per_slot=8192, wall_minutes=60)
    cb = S.one(f"PROSPER_MODULE=prosper.embeddings.combine_similarity_cli bash {WORKER} "
               f"--encoder uni2 --method {method} --similarity-dir {OUT}/{_sim_task(pct, method)} "
               f"--matrix-model {model} --min-slides {EXPECT_AP - 10}",
               job_name=f"prosper_combine_{model}_{method}", res=res,
               log_dir=OUT / "logs" / "combine_similarity", kind="cpu",
               dry_run=not go, dependency=_dep(after))
    ev_res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=60)
    evs = [S.one(f"bash {WORKER} --model {model} --method {tag} --topk {k} "
                 f"--tie-break generous_majority --iter 5000 --n-jobs {SLOTS} --task r22_ladder",
                 job_name=f"prosper_r22_eval_f{pct}_{tag}_k{k}", res=ev_res,
                 log_dir=OUT / "logs" / "r22_ladder", kind="cpu", dry_run=not go,
                 dependency=_dep([cb], "done")) for k in KS]
    met = S.one(f"PROSPER_MODULE=prosper.evaluation.metrics_cli bash {WORKER} "
                f"--model {model} --method {tag} --depth 10 --iter 5000 --task r22_metrics",
                job_name=f"prosper_r22_met_f{pct}_{tag}", res=ev_res,
                log_dir=OUT / "logs" / "r22_paired", kind="cpu", dry_run=not go,
                dependency=_dep([cb], "done"))
    return {"combine": cb, "evals": evs, "metrics": met}


BETA = Path("/PROSPER_GROUP_DATA/comet/common/ali/COMET_top20K_4735samples_swan.csv")


def x2(go: bool, metric: str = "correlation") -> str:
    res = S.Resources(slots=4, mem_mb_per_slot=8192, wall_minutes=180)
    task = "x2_methylation" if metric == "correlation" else f"x2_methylation_{metric}"
    return S.one(f"PROSPER_MODULE=prosper.evaluation.methylation_knn_cli bash {WORKER} "
                 f"--beta {BETA} --coverage {OUT}/x2_methylation_coverage.csv "
                 f"--ranked-lists {OUT}/c2_metric_suite/ranked_lists_uni2_ot_cosine.csv.gz "
                 f"--k 10 --he-k 5 --depth 10 --metric {metric} --iter 5000 --seed 123 --task {task}",
                 job_name=f"prosper_{task}", res=res, log_dir=OUT / "logs" / "x2_methylation",
                 kind="cpu", dry_run=not go)


TILE_CLF = Path("/PROSPER_WORKSPACE/comet-annotation/model/"
                "pathologist_annotated_slides_uni2/tumor_detector_sgd_test_fraction_0.95_training.pkl")
TRIDENT = Path("/PROSPER_WORKSPACE/comet-feature-gen/comet-titan/trident")
TRIDENT_PY = "/HOME/miniconda3/envs/trident/bin/python"
CONCH_DIR = TRIDENT / "trident_processed" / "40x_1024px_0px_overlap" / "features_conch_v15"
WSI_DIR = Path("/PROSPER_WORKSPACE/data/comet_final_HPC")


def s2(go: bool, stage: str = "all") -> dict:
    """Tumour-only TITAN chain. ``stage`` limits to one step (score|subset|titan|matrix|eval)."""
    job_dir = OUT / "s2_titan_tumor_only"
    log = OUT / "logs" / "s2_titan_tumor_only"
    mod = "PROSPER_MODULE=prosper.preprocessing.tumor_tiles_cli"
    ids = {}
    if stage in ("all", "score"):
        ids["score"] = S.one(f"{mod} bash {WORKER} score --classifier {TILE_CLF} "
                             f"--features-dir {AP_FEATURES} --out-dir {OUT}/s2_tumor_tiles/scores",
                             job_name="prosper_s2_score", res=S.Resources(slots=8, mem_mb_per_slot=4096, wall_minutes=240),
                             log_dir=log, kind="cpu", dry_run=not go)
    if stage in ("all", "subset"):
        ids["subset"] = S.one(f"{mod} bash {WORKER} subset --scores-dir {OUT}/s2_tumor_tiles/scores "
                              f"--conch-dir {CONCH_DIR} --out-job-dir {job_dir} --min-tumour-tiles 2 --min-patches 8",
                              job_name="prosper_s2_subset", res=S.Resources(slots=4, mem_mb_per_slot=4096, wall_minutes=240),
                              log_dir=log, kind="cpu", dry_run=not go, dependency=_dep([ids.get("score")], "done"))
    if stage in ("all", "titan"):
        cmd = (f"cd {TRIDENT} && {TRIDENT_PY} run_batch_of_slides.py --task feat --wsi_dir {WSI_DIR} "
               f"--job_dir {job_dir} --patch_encoder conch_v15 --slide_encoder titan --mag 40 "
               f"--patch_size 1024 --coords_dir 40x_1024px_0px_overlap --skip_errors")
        ids["titan"] = S.one(cmd, job_name="prosper_s2_titan",
                             res=S.Resources(slots=4, mem_mb_per_slot=8192, wall_minutes=720, gpus=1),
                             log_dir=log, kind="gpu", dry_run=not go, dependency=_dep([ids.get("subset")], "done"))
    if stage in ("all", "matrix"):
        ids["matrix"] = S.one(f"{mod} bash {WORKER} matrix --slide-features-dir "
                              f"{job_dir}/40x_1024px_0px_overlap/slide_features_titan --model titan --method titan "
                              f"--tumor-only --min-slides 2500",
                              job_name="prosper_s2_matrix", res=S.Resources(slots=4, mem_mb_per_slot=4096, wall_minutes=60),
                              log_dir=log, kind="cpu", dry_run=not go, dependency=_dep([ids.get("titan")], "done"))
    if stage == "paired":
        # paired deltas on the common queries: tumour-only TITAN (reference) vs all-tissue TITAN
        # paired.py requires identical query sets: restrict both lists to the common queries
        # (the tumour-only matrix lacks the 40 slides excluded for < 8 tumour patches / no scores)
        import pandas as pd
        pin = OUT / "s2_paired_in"
        pin.mkdir(parents=True, exist_ok=True)
        srcs = (OUT / "s2_tumor_only" / "ranked_lists_titan_titan_tumor_only.csv.gz",
                OUT / "s2_tumor_only_reference" / "ranked_lists_titan_titan.csv.gz")
        frames = [pd.read_csv(x, dtype={"query": str, "neighbour": str}) for x in srcs]
        common = set(frames[0]["query"]).intersection(frames[1]["query"])
        for src, df in zip(srcs, frames):
            dst = pin / src.name
            if dst.is_symlink():
                dst.unlink()
            df[df["query"].isin(common)].to_csv(dst, index=False, compression="gzip")
        (pin / "_common_queries.txt").write_text("\n".join(sorted(common)) + "\n")
        print(f"[s2 paired] common queries: {len(common)}")
        ev_res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=60)
        ids["paired"] = [S.one(f"PROSPER_MODULE=prosper.evaluation.paired bash {WORKER} --ranked-dir {pin} "
                               f"--reference titan_titan_tumor_only --tumor_only --k {k} --depth 10 --iter 5000 "
                               f"--task s2_paired",
                               job_name=f"prosper_s2_paired_k{k}", res=ev_res, log_dir=log, kind="cpu", dry_run=not go)
                        for k in KS]
        return ids
    if stage == "ref":
        # comparator: the PUBLISHED all-tissue TITAN matrix evaluated on the tumour-only
        # population (drop MISC + NORMAL), so the S2 delta isolates the tumour-only embedding
        ev_res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=60)
        ids["ref_evals"] = [S.one(f"bash {WORKER} --model titan --method titan --drop-labels MISC NORMAL "
                                  f"--topk {k} --tie-break {r} --iter 5000 --n-jobs {SLOTS} --task s2_tumor_only_reference",
                                  job_name=f"prosper_s2ref_eval_k{k}_{r[:3]}", res=ev_res, log_dir=log, kind="cpu",
                                  dry_run=not go)
                           for r in ("generous_majority", "absolute") for k in KS]
        ids["ref_metrics"] = S.one(f"PROSPER_MODULE=prosper.evaluation.metrics_cli bash {WORKER} "
                                   f"--model titan --method titan --drop-labels MISC NORMAL --depth 10 --iter 5000 "
                                   f"--task s2_tumor_only_reference",
                                   job_name="prosper_s2ref_metrics", res=ev_res, log_dir=log, kind="cpu", dry_run=not go)
        return ids
    if stage in ("all", "eval"):
        dep = _dep([ids.get("matrix")], "done")
        ev_res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=60)
        ids["evals"] = [S.one(f"bash {WORKER} --model titan --method titan --tumor_only --topk {k} "
                              f"--tie-break {r} --iter 5000 --n-jobs {SLOTS} --task s2_tumor_only",
                              job_name=f"prosper_s2_eval_k{k}_{r[:3]}", res=ev_res, log_dir=log, kind="cpu",
                              dry_run=not go, dependency=dep)
                       for r in ("generous_majority", "absolute") for k in KS]
        ids["metrics"] = S.one(f"PROSPER_MODULE=prosper.evaluation.metrics_cli bash {WORKER} "
                               f"--model titan --method titan --tumor_only --depth 10 --iter 5000 --task s2_tumor_only",
                               job_name="prosper_s2_metrics", res=ev_res, log_dir=log, kind="cpu",
                               dry_run=not go, dependency=dep)
    return ids


LLM_VENV_PY = OUT / "envs" / "comet-llm" / "bin" / "python"
LLM_ENV_FILE = Path("/PROSPER_WORKSPACE/comet-llm/.env")
PRIMARY_MATRIX = Path("/PROSPER_WORKSPACE/comet-feature-gen/"
                      "comet-uni2/output/similarity_matrix/combined_similarity_matrix_uni2_ot_cosine_no_same_patient.pkl")


def s5(go: bool, n_queries: int = 100, fake: bool = False, k: int = 5,
       summarizer: str = "manuscript", suffix: str = "") -> str:
    """S5 grounding audit. `summarizer` selects the copy under test (see
    llm_grounding_cli --summarizer); `suffix` names the output dir, e.g. "_v2"
    for the 2026-09-02 re-run on the manuscript configuration (the original
    2026-08-30 run, no suffix, was the agentic branch)."""
    task = "s5_llm_grounding" + ("_fake" if fake else "") + (f"_k{k}" if k != 5 else "") + suffix
    cmd = (f"cd {REPO} && PROSPER_OUT={OUT} {LLM_VENV_PY} prosper/evaluation/llm_grounding_cli.py "
           f"--matrix {PRIMARY_MATRIX} --k {k} --n-queries {n_queries} --seed 123 --repeat-original 20 "
           f"--summarizer {summarizer} "
           f"--env-file {LLM_ENV_FILE} --out-dir {OUT}/{task}" + (" --fake" if fake else ""))
    return S.one(cmd, job_name=f"prosper_{task}", res=S.Resources(slots=2, mem_mb_per_slot=8192, wall_minutes=720),
                 log_dir=OUT / "logs" / "s5_llm_grounding", kind="cpu", dry_run=not go)


def c4(go: bool, n_queries: int = 10, roi_tiles: int = 1) -> str:
    res = S.Resources(slots=4, mem_mb_per_slot=8192, wall_minutes=240)
    task = "c4_localization" if roi_tiles == 1 else f"c4_localization_roi{roi_tiles}"
    return S.one(f"PROSPER_MODULE=prosper.evaluation.localization_cli bash {WORKER} "
                 f"--n-queries {n_queries} --roi-tiles {roi_tiles} --iter 2000 --seed 123 --task {task}",
                 job_name=f"prosper_{task}", res=res, log_dir=OUT / "logs" / "c4_localization",
                 kind="cpu", dry_run=not go)


def paired_ladder(reference: str, task: str, go: bool) -> list:
    ev_res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=60)
    return [S.one(f"PROSPER_MODULE=prosper.evaluation.paired bash {WORKER} "
                  f"--ranked-dir {OUT}/r22_metrics --reference {reference} "
                  f"--k {k} --depth 10 --iter 5000 --task {task}",
                  job_name=f"prosper_{task}_k{k}", res=ev_res,
                  log_dir=OUT / "logs" / "r22_paired", kind="cpu", dry_run=not go)
            for k in KS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["c1e", "c7b", "c7c-bob-canary", "c7c-bob-fanout",
                                       "c7c-ot-canary", "c7c-ot-fanout", "combine-ladder",
                                       "paired-ladder", "x2", "s2", "c4", "s5"])
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--roi-tiles", type=int, default=1)
    ap.add_argument("--metric", default="correlation")
    ap.add_argument("--reference", default="uni2apf15_yottixel")
    ap.add_argument("--task", default="r22_paired_v3")
    ap.add_argument("--pct", type=int, default=15)
    ap.add_argument("--method", choices=["ot", "bob"], default="bob")
    ap.add_argument("--epilogue", type=Path)
    ap.add_argument("--after", default="")
    ap.add_argument("--summarizer", choices=["manuscript", "agentic"], default="manuscript",
                    help="s5 only: which summariser copy to audit")
    ap.add_argument("--suffix", default="", help="s5 only: output-dir suffix, e.g. _v2")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    after = [x for x in a.after.split(",") if x.strip()]
    if a.action == "c1e":
        print(c1e(a.go))
    elif a.action == "c7b":
        print(c7b(a.go))
    elif a.action == "c7c-bob-canary":
        print(c7c_canary(a.pct, "bob", a.go))
    elif a.action == "c7c-ot-canary":
        print(c7c_canary(a.pct, "ot", a.go))
    elif a.action in ("c7c-bob-fanout", "c7c-ot-fanout"):
        if not a.epilogue:
            raise SystemExit("needs --epilogue")
        print(c7c_fanout(a.pct, "ot" if "ot" in a.action else "bob", a.epilogue, a.go))
    elif a.action == "s5":
        print(s5(a.go, a.n_queries, a.fake, a.k, a.summarizer, a.suffix))
    elif a.action == "c4":
        print(c4(a.go, roi_tiles=a.roi_tiles))
    elif a.action == "s2":
        print(s2(a.go, a.stage))
    elif a.action == "x2":
        print(x2(a.go, a.metric))
    elif a.action == "paired-ladder":
        print(paired_ladder(a.reference, a.task, a.go))
    else:
        print(combine_ladder(a.pct, a.method, after, a.go))
    if not a.go:
        print("\n[dry run] nothing submitted. Re-run with --go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
