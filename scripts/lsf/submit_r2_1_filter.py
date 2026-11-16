#!/usr/bin/env python3
"""R2.1 F1/F2: H-Optimus internal tumour-filtered retrieval (both-filtered arms).

Stages (without --go every stage only PRINTS the exact bsub lines; with --go it submits):
  features  r2_1_filter_features_cli: verified tumour-filtered pickles for budgets 5 and 100
            -> $PROSPER_OUT/r2_1_filter_20261003/features/budget{5,100}/<slide>.pkl
  sim       similarity_cli in SELF-LIBRARY mode (no --library-features): the filtered directory is both
            the query set and the reference library, exactly like the main H-Optimus OT/BoB matrices
            (self pair -> 1e6 sentinel). OT: --sim-batch-ot queries per element (default 1);
            BoB: --sim-batch-bob (default 50). Rows: r2_1_filter_20261003/similarity/budget{b}_{ot,bob}/
  combine   combine_similarity_cli (non-external): square matrix + `_no_same_patient` twin
            -> similarity_matrix/combined_similarity_matrix_r2_1_hopt_filter_budget{b}_{ot_cosine,yottixel}[_no_same_patient].pkl
            chained on ended(sim) and gated by --min-slides = number of written filtered pickles
  evaluate  r2_1_filter_eval_cli --standard, chained on ended(combine)
  pilot     bounded pilot jobs (logged to r2_1_filter_20261003/pilot_jobs.jsonl)

The query index space of similarity_cli is the sorted glob order of the filtered directory; the
params files are generated from that directory, so `sim` requires `features` to have finished.
Rows are resumable: an existing similarity_<id>.pkl is skipped, so `sim --missing-only` re-submits
only absent rows.

Usage: submit_r2_1_filter.py sim [--budgets 5,100] [--methods ot,bob] [--go]
       submit_r2_1_filter.py combine --after <jobid ...> [--go]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from prosper.hpc import submit as S              # noqa: E402
from prosper.evaluation import paths             # noqa: E402

OUT = paths.output_root()
WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
ROOT = OUT / "r2_1_filter_20261003"
FEAT = ROOT / "features"
LOGS = ROOT / "logs"
PARAMS = ROOT / "params"
TAG = {"ot": "ot_cosine", "bob": "yottixel"}
QUEUE = "large_core_count"


def task(b: int, m: str, pilot: bool = False) -> str:
    return f"r2_1_filter_20261003/{'pilot/' if pilot else ''}similarity/budget{b}_{m}"


def filtered_ids(b: int) -> list:
    return [Path(f).stem for f in sorted(glob.glob(str(FEAT / f"budget{b}" / "*.pkl")))]


def _dep(ids, kind="ended"):
    ids = [i for i in ids if i]
    return " && ".join(f"{kind}({i})" for i in ids) or None


def _write_params(path: Path, lines: list) -> Path:
    data = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != data:
            raise SystemExit(f"{path} exists with different content; refusing to overwrite "
                             f"(use a new --params-tag)")
    else:
        path.write_text(data)
    return path


def sim_line(b: int, m: str, lo: int, hi: int, pilot: bool = False) -> str:
    return (f"--encoder h_optimus_1 --method {m} --features {FEAT / f'budget{b}'} --feature-dim 1536 "
            f"--lo {lo} --hi {hi} --task {task(b, m, pilot)} --num-workers 1")


def _runs(indices: list, batch: int) -> list:
    """Group 1-based indices into runs of consecutive indices, at most `batch` per run."""
    runs = []
    for i in indices:
        if runs and i == runs[-1][-1] + 1 and len(runs[-1]) < batch:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def sim(go: bool, budgets, methods, res_ot: S.Resources, res_bob: S.Resources, batch_ot: int,
        batch_bob: int, missing_only: bool, params_tag: str, after=None) -> dict:
    jids = {}
    for b in budgets:
        ids = filtered_ids(b)
        if not ids:
            raise SystemExit(f"no filtered pickles for budget {b}: run `features` first")
        for m in methods:
            have = {Path(f).stem[len("similarity_"):]
                    for f in glob.glob(str(OUT / task(b, m) / "similarity_*.pkl"))}
            idx = [i + 1 for i, q in enumerate(ids) if not (missing_only and q in have)]
            if not idx:
                print(f"[sim] budget{b}/{m}: all {len(ids)} rows present")
                continue
            runs = _runs(idx, batch_ot if m == "ot" else batch_bob)
            params = _write_params(PARAMS / f"sim_budget{b}_{m}{params_tag}.txt",
                                   [sim_line(b, m, r[0], r[-1]) for r in runs])
            res = res_ot if m == "ot" else res_bob
            print(f"[sim] budget{b}/{m}: {len(ids)} queries, {len(idx)} to compute, {len(runs)} elements, "
                  f"-n {res.slots} -M {res.mem_mb_per_slot} -W {res.wall_minutes} on {QUEUE}")
            out = []
            for lo, hi in S.chunk_array(len(runs), S.MAX_JOB_ARRAY_SIZE):
                out.append(S._bsub(
                    f"PROSPER_MODULE=prosper.embeddings.similarity_cli PROSPER_PARAMS={params} bash {WORKER}",
                    job_name=f"r21f_sim_b{b}_{m}", queue=QUEUE, res=res, log_dir=LOGS / f"sim_budget{b}_{m}",
                    array=f"{lo}-{hi}", dry_run=not go, dependency=_dep(after or [])))
            jids[f"budget{b}_{m}"] = out
    return jids


def combine(go: bool, budgets, methods, after: dict) -> dict:
    jids = {}
    res = S.Resources(slots=1, mem_mb_per_slot=16384, wall_minutes=120)
    for b in budgets:
        n = len(filtered_ids(b))
        for m in methods:
            jids[f"budget{b}_{m}"] = S.one(
                f"PROSPER_MODULE=prosper.embeddings.combine_similarity_cli bash {WORKER} "
                f"--encoder h_optimus_1 --method {m} --similarity-dir {OUT / task(b, m)} "
                f"--matrix-model r2_1_hopt_filter_budget{b} --min-slides {n}",
                job_name=f"r21f_combine_b{b}_{m}", res=res, queue=QUEUE, log_dir=LOGS / "combine",
                dry_run=not go, dependency=_dep(after.get(f"budget{b}_{m}", [])))
    return jids


def evaluate(go: bool, after: list) -> str:
    res = S.Resources(slots=1, mem_mb_per_slot=16384, wall_minutes=120)
    return S.one(f"PROSPER_MODULE=prosper.evaluation.r2_1_filter_eval_cli bash {WORKER} --standard",
                 job_name="r21f_evaluate", res=res, queue=QUEUE, log_dir=LOGS / "evaluate",
                 dry_run=not go, dependency=_dep(after))


def features(go: bool) -> str:
    res = S.Resources(slots=1, mem_mb_per_slot=8192, wall_minutes=120)
    return S.one(f"PROSPER_MODULE=prosper.evaluation.r2_1_filter_features_cli bash {WORKER} --budgets 5 100",
                 job_name="r21f_features", res=res, queue=QUEUE, log_dir=LOGS / "features", dry_run=not go)


# ---------------------------------------------------------------------------------------------
# Query-fold (specification) design, [decision record] section B: head f filters fold f's queries AND the whole
# reference bank. Additive stages (qf-*); the stages above are unchanged.
#   qf-features  one job per head fold: r2_1_filter_features_cli --head-fold f --budgets ...
#                -> features_qf/budget{b}/fold{f}/<slide>.pkl (+ _filter_summary.csv/_filter_record.json)
#   qf-views     r2_1_filter_features_cli --make-views: symlink query sets (own-fold returned slides of
#                bank (b,f)) and the unfiltered 2,604-slide population library view
#   qf-sim       similarity_cli EXTERNAL mode (--library-features), one row per returned fold-f query:
#                arm both  -> library features_qf/budget{b}/fold{f} (all slides head f returns)
#                arm query -> library similarity_inputs_qf/library_unfiltered_pop2604
#                rows: r2_1_filter_20261003/similarity_qf/budget{b}/fold{f}/{both,query}_{ot,bob}/
#   qf-combine   r2_1_filter_eval_cli combine-qf per budget x arm x method, ended() on its 5 fold arrays
#   qf-evaluate  r2_1_filter_eval_cli evaluate-qf --standard, ended() on all combines
#   qf-rest      qf-sim -> qf-combine -> qf-evaluate (views must exist: qf-sim reads the query index space)
#   qf-cost      print the CPU-hour estimate only (cost_qf/old_production_rowtimes.csv)
# Without --go nothing is submitted; dependencies print as ended(<job-name>) placeholders.
# ---------------------------------------------------------------------------------------------
FEATQF = ROOT / "features_qf"
VIEWS = ROOT / "similarity_inputs_qf"
LOGSQF = ROOT / "logs_qf"
PARAMSQF = ROOT / "params_qf"
QF_ARM_NAME = {"both": "both_filtered", "query": "query_only"}
UNFILTERED_VIEW = "library_unfiltered_pop2604"


def qf_task(b: int, f: int, arm: str, m: str) -> str:
    return f"r2_1_filter_20261003/similarity_qf/budget{b}/fold{f}/{arm}_{m}"


def qf_query_ids(b: int, f: int) -> list:
    d = VIEWS / "queries" / f"budget{b}" / f"fold{f}"
    return [Path(x).stem for x in sorted(glob.glob(str(d / "*.pkl")))]


def qf_library(b: int, f: int, arm: str) -> Path:
    return FEATQF / f"budget{b}" / f"fold{f}" if arm == "both" else VIEWS / UNFILTERED_VIEW


def _jid(ret: str, go: bool, name: str) -> str:
    return ret if go else (ret or f"<{name}>")


class _Queues:
    def __init__(self, names):
        self.names, self.i = list(names), 0

    def next(self) -> str:
        q = self.names[self.i % len(self.names)]
        self.i += 1
        return q


def qf_features(go: bool, budgets, folds, queues: _Queues) -> dict:
    res = S.Resources(slots=1, mem_mb_per_slot=8192, wall_minutes=120)
    bl = " ".join(str(b) for b in budgets)
    return {f"fold{f}": _jid(S.one(
        f"PROSPER_MODULE=prosper.evaluation.r2_1_filter_features_cli bash {WORKER} --head-fold {f} --budgets {bl}",
        job_name=f"r21qf_features_f{f}", res=res, queue=queues.next(), log_dir=LOGSQF / "features",
        dry_run=not go), go, f"r21qf_features_f{f}") for f in folds}


def qf_views(go: bool, budgets, folds, queues: _Queues, after=None) -> str:
    res = S.Resources(slots=1, mem_mb_per_slot=4096, wall_minutes=120)
    bl = " ".join(str(b) for b in budgets)
    fl = " ".join(str(f) for f in folds)
    return _jid(S.one(
        f"PROSPER_MODULE=prosper.evaluation.r2_1_filter_features_cli bash {WORKER} --make-views "
        f"--budgets {bl} --head-folds {fl}", job_name="r21qf_views", res=res, queue=queues.next(),
        log_dir=LOGSQF / "views", dry_run=not go, dependency=_dep(after or [])), go, "r21qf_views")


def qf_sim_line(b: int, f: int, arm: str, m: str, lo: int, hi: int) -> str:
    return (f"--encoder h_optimus_1 --method {m} --features {VIEWS / 'queries' / f'budget{b}' / f'fold{f}'} "
            f"--library-features {qf_library(b, f, arm)} --feature-dim 1536 "
            f"--lo {lo} --hi {hi} --task {qf_task(b, f, arm, m)} --num-workers 1")


def qf_sim(go: bool, budgets, folds, arms, methods, res: dict, batch_ot: int, batch_bob: int,
           missing_only: bool, params_tag: str, queues: _Queues, after=None) -> dict:
    """res[(arm, m)] -> Resources. Returns {"budget{b}_{arm}_{m}": [array ids over folds]}."""
    jids = {}
    for b in budgets:
        for arm in arms:
            for m in methods:
                key = f"budget{b}_{arm}_{m}"
                jids[key] = []
                for f in folds:
                    ids = qf_query_ids(b, f)
                    if not ids:
                        raise SystemExit(f"no query view for budget {b} fold {f}: run qf-features and qf-views first")
                    lib = qf_library(b, f, arm)
                    if not lib.is_dir():
                        raise SystemExit(f"library {lib} missing")
                    have = {Path(x).stem[len("similarity_"):]
                            for x in glob.glob(str(OUT / qf_task(b, f, arm, m) / "similarity_*.pkl"))}
                    idx = [i + 1 for i, q in enumerate(ids) if not (missing_only and q in have)]
                    if not idx:
                        print(f"[qf-sim] budget{b}/fold{f}/{arm}_{m}: all {len(ids)} rows present")
                        continue
                    runs = _runs(idx, batch_ot if m == "ot" else batch_bob)
                    params = _write_params(PARAMSQF / f"sim_budget{b}_fold{f}_{arm}_{m}{params_tag}.txt",
                                           [qf_sim_line(b, f, arm, m, r[0], r[-1]) for r in runs])
                    r = res[(arm, m)]
                    q = queues.next()
                    name = f"r21qf_sim_b{b}_f{f}_{arm}_{m}"
                    print(f"[qf-sim] budget{b}/fold{f}/{arm}_{m}: {len(ids)} queries, {len(idx)} to compute, "
                          f"{len(runs)} elements, -n {r.slots} -M {r.mem_mb_per_slot} -W {r.wall_minutes} on {q}")
                    for lo, hi in S.chunk_array(len(runs), S.MAX_JOB_ARRAY_SIZE):
                        jids[key].append(_jid(S._bsub(
                            f"PROSPER_MODULE=prosper.embeddings.similarity_cli PROSPER_PARAMS={params} bash {WORKER}",
                            job_name=name, queue=q, res=r, log_dir=LOGSQF / f"sim_budget{b}_fold{f}_{arm}_{m}",
                            array=f"{lo}-{hi}", dry_run=not go, dependency=_dep(after or [])), go, name))
    return jids


def qf_combine(go: bool, budgets, arms, methods, after: dict, queues: _Queues) -> dict:
    res = S.Resources(slots=1, mem_mb_per_slot=8192, wall_minutes=60)
    jids = {}
    for b in budgets:
        for arm in arms:
            for m in methods:
                key = f"budget{b}_{arm}_{m}"
                name = f"r21qf_combine_b{b}_{arm}_{m}"
                jids[key] = _jid(S.one(
                    f"PROSPER_MODULE=prosper.evaluation.r2_1_filter_eval_cli bash {WORKER} combine-qf "
                    f"--arm {QF_ARM_NAME[arm]} --method {m} --budget {b}",
                    job_name=name, res=res, queue=queues.next(), log_dir=LOGSQF / "combine",
                    dry_run=not go, dependency=_dep(after.get(key, []))), go, name)
    return jids


def qf_evaluate(go: bool, after: list, queues: _Queues) -> str:
    res = S.Resources(slots=1, mem_mb_per_slot=16384, wall_minutes=120)
    return _jid(S.one(f"PROSPER_MODULE=prosper.evaluation.r2_1_filter_eval_cli bash {WORKER} evaluate-qf --standard",
                      job_name="r21qf_evaluate", res=res, queue=queues.next(), log_dir=LOGSQF / "evaluate",
                      dry_run=not go, dependency=_dep(after)), go, "r21qf_evaluate")


def qf_cost(budgets, folds, arms, methods, query_ratio: dict, batch_ot: int, batch_bob: int) -> dict:
    """CPU-hour estimate from the MEASURED previous production rows (cost_qf/old_production_rowtimes.csv).

    both arm: the query pickle of an own-fold slide is byte-identical to its previous own-fold bank pickle
    and the library is a filtered bank of similar size, so the estimate is the measured previous row time
    times (new library tiles / previous library tiles)^alpha. query arm: additionally scaled to the
    unfiltered population library and by the pilot-calibrated query_ratio[m] (1.0 = pure size scaling).
    Queries returned now but absent from the previous production rows use the budget/method median."""
    import pandas as pd
    alpha = {"ot": 1.03, "bob": 0.83}
    old = pd.read_csv(ROOT / "cost_qf" / "old_production_rowtimes.csv", dtype={"slide_id": str})
    pop = pd.read_csv(OUT / "r2_1_mosaic_cpu_20261003" / "attempt_001" / "slide_offsets.csv",
                      dtype={"slide_id": str})
    lib_unf = int(pop.feature_rows.sum())
    est = {}
    for b in budgets:
        oldsum = pd.read_csv(FEAT / f"budget{b}" / "_filter_summary.csv", dtype={"slide_id": str})
        lib_old = int(oldsum.loc[oldsum.written == 1, "retained_tiles"].sum())
        for f in folds:
            bank = pd.read_csv(FEATQF / f"budget{b}" / f"fold{f}" / "_filter_summary.csv", dtype={"slide_id": str})
            lib_new = int(bank.loc[bank.written == 1, "retained_tiles"].sum())
            qids = qf_query_ids(b, f)
            for arm in arms:
                L = lib_new if arm == "both" else lib_unf
                for m in methods:
                    t = old[(old.budget == b) & (old.method == m)].set_index("slide_id").row_seconds
                    scale = (L / lib_old) ** alpha[m] * (query_ratio.get(m, 1.0) if arm == "query" else 1.0)
                    per = [float(t.get(q, t.median())) * scale for q in qids]
                    batch = batch_ot if m == "ot" else batch_bob
                    elems = [sum(per[i:i + batch]) for i in range(0, len(per), batch)]
                    est[f"budget{b}_fold{f}_{arm}_{m}"] = {
                        "queries": len(qids), "missing_old_rows": sum(1 for q in qids if q not in t.index),
                        "library_tiles": L, "previous_library_tiles": lib_old, "scale": scale,
                        "cpu_h": sum(per) / 3600, "element_s_max": max(elems) if elems else 0.0,
                        "element_s_median": float(pd.Series(elems).median()) if elems else 0.0}
    tot = {}
    for k, v in est.items():
        arm_m = "_".join(k.split("_")[2:])
        tot[arm_m] = tot.get(arm_m, 0.0) + v["cpu_h"]
    return {"per_array": est, "cpu_h_by_arm_method": tot, "total_cpu_h": sum(tot.values()),
            "query_ratio": query_ratio, "alpha": alpha}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["features", "sim", "combine", "evaluate", "rest", "qf-features",
                                      "qf-views", "qf-sim", "qf-combine", "qf-evaluate", "qf-rest", "qf-cost"])
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--budgets", default="5,100")
    ap.add_argument("--methods", default="ot,bob")
    ap.add_argument("--after", nargs="*", default=[], help="job ids to chain on with ended()")
    ap.add_argument("--sim-batch-ot", type=int, default=1)
    ap.add_argument("--sim-batch-bob", type=int, default=50)
    ap.add_argument("--ot-mem-mb", type=int, default=28672)
    ap.add_argument("--ot-wall-min", type=int, default=600)
    ap.add_argument("--bob-mem-mb", type=int, default=10240)
    ap.add_argument("--bob-wall-min", type=int, default=600)
    ap.add_argument("--missing-only", action="store_true", help="only rows absent on disk")
    ap.add_argument("--params-tag", default="", help="suffix for a new params file (gap-fill)")
    ap.add_argument("--folds", default="0,1,2,3,4", help="qf-*: query/head folds")
    ap.add_argument("--arms", default="both,query", help="qf-*: both (contract) and/or query (query-only)")
    ap.add_argument("--queues", default=QUEUE, help="qf-*: comma list, used round-robin per job/array")
    ap.add_argument("--qf-query-ot-mem-mb", type=int, default=32768, help="qf-sim query arm OT -M")
    ap.add_argument("--qf-query-ot-wall-min", type=int, default=900, help="qf-sim query arm OT -W")
    ap.add_argument("--qf-query-ratio-ot", type=float, default=1.0, help="qf-cost pilot calibration")
    ap.add_argument("--qf-query-ratio-bob", type=float, default=1.0, help="qf-cost pilot calibration")
    a = ap.parse_args()
    if os.environ.get("PROSPER_OUT") is None:
        raise SystemExit("export PROSPER_OUT first")
    budgets = [int(x) for x in a.budgets.split(",")]
    methods = a.methods.split(",")
    res_ot = S.Resources(slots=1, mem_mb_per_slot=a.ot_mem_mb, wall_minutes=a.ot_wall_min)
    res_bob = S.Resources(slots=1, mem_mb_per_slot=a.bob_mem_mb, wall_minutes=a.bob_wall_min)
    rec = {"stage": a.stage, "go": a.go, "argv": sys.argv}
    if a.stage == "features":
        rec["features"] = features(a.go)
    elif a.stage == "sim":
        rec["sim"] = sim(a.go, budgets, methods, res_ot, res_bob, a.sim_batch_ot, a.sim_batch_bob,
                         a.missing_only, a.params_tag, after=a.after)
    elif a.stage == "combine":
        rec["combine"] = combine(a.go, budgets, methods, {f"budget{b}_{m}": a.after
                                                          for b in budgets for m in methods})
    elif a.stage == "evaluate":
        rec["evaluate"] = evaluate(a.go, a.after)
    elif a.stage == "rest":       # sim -> combine -> evaluate, chained on ended()
        si = sim(a.go, budgets, methods, res_ot, res_bob, a.sim_batch_ot, a.sim_batch_bob,
                 a.missing_only, a.params_tag)
        co = combine(a.go, budgets, methods, si)
        rec.update(sim=si, combine=co, evaluate=evaluate(a.go, list(co.values())))
    if a.stage.startswith("qf-"):
        folds = [int(x) for x in a.folds.split(",")]
        arms = a.arms.split(",")
        if not set(arms) <= {"both", "query"}:
            raise SystemExit("--arms must be a subset of both,query")
        qs = _Queues(a.queues.split(","))
        res_qf = {("both", "ot"): res_ot, ("both", "bob"): res_bob, ("query", "bob"): res_bob,
                  ("query", "ot"): S.Resources(slots=1, mem_mb_per_slot=a.qf_query_ot_mem_mb,
                                               wall_minutes=a.qf_query_ot_wall_min)}
        simargs = (budgets, folds, arms, methods, res_qf, a.sim_batch_ot, a.sim_batch_bob,
                   a.missing_only, a.params_tag, qs)
        if a.stage == "qf-features":
            rec["qf_features"] = qf_features(a.go, budgets, folds, qs)
        elif a.stage == "qf-views":
            rec["qf_views"] = qf_views(a.go, budgets, folds, qs, after=a.after)
        elif a.stage == "qf-sim":
            rec["qf_sim"] = qf_sim(a.go, *simargs, after=a.after)
        elif a.stage == "qf-combine":
            rec["qf_combine"] = qf_combine(a.go, budgets, arms, methods,
                                           {f"budget{b}_{arm}_{m}": a.after for b in budgets
                                            for arm in arms for m in methods}, qs)
        elif a.stage == "qf-evaluate":
            rec["qf_evaluate"] = qf_evaluate(a.go, a.after, qs)
        elif a.stage == "qf-rest":
            si = qf_sim(a.go, *simargs, after=a.after)
            co = qf_combine(a.go, budgets, arms, methods, si, qs)
            rec.update(qf_sim=si, qf_combine=co, qf_evaluate=qf_evaluate(a.go, list(co.values()), qs))
        if a.stage in ("qf-sim", "qf-rest", "qf-cost"):
            rec["qf_cost"] = qf_cost(budgets, folds, arms, methods,
                                     {"ot": a.qf_query_ratio_ot, "bob": a.qf_query_ratio_bob},
                                     a.sim_batch_ot, a.sim_batch_bob)
            print(json.dumps({k: v for k, v in rec["qf_cost"].items() if k != "per_array"}, indent=1))
    if a.go:
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        (ROOT / f"_submission_{a.stage}_{stamp}.json").write_text(json.dumps(rec, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
