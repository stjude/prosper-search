#!/usr/bin/env python3
"""[decision record] pilot-only KidRare saved-feature tumour-filter extension.

Stages: score (preflight and score354), filter (validate and freeze selection),
ot (one already-selected query/arm), verify (all eight rows and bindings).
The script does not submit jobs or run the full extension.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROSPER = (ROOT / "../prosper-search").resolve()
COMET = (ROOT / "../comet-feature-gen").resolve()
sys.path.insert(0, str(PROSPER))
from prosper.evaluation import external_cli  # noqa: E402

DEST = "r1_3_kidrare_filter_extension_20260930"
CONTROL = "NULL-20241227-133138"
PYTHON = Path("/HOME/miniconda3/bin/python3")
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1", "GOTO_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1", "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE",
    "OMP_NESTED": "FALSE", "OMP_MAX_ACTIVE_LEVELS": "1", "CUDA_VISIBLE_DEVICES": "",
}
ARMS = ("query_only", "both_sided")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, obj: dict) -> None:
    require(not path.exists(), f"Refusing to overwrite {path}")
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def json_read(path: Path) -> dict:
    require(path.is_file(), f"Missing stage receipt: {path}")
    return json.loads(path.read_text())


def ids_in(directory: Path) -> list[str]:
    require(directory.is_dir(), f"Missing feature bank {directory}")
    names = sorted(directory.iterdir())
    require(all(p.is_file() and p.suffix == ".pkl" for p in names),
            f"Unexpected file/non-pickle in bank {directory}")
    ids = [p.stem for p in names]
    require(len(ids) == len(set(ids)) and
            [p.name for p in names] == [f"{sid}.pkl" for sid in ids],
            f"Duplicate or malformed feature ID in {directory}")
    return ids


def out_root() -> Path:
    raw = os.environ.get("PROSPER_OUT")
    expected = (ROOT / "prosper_revision_output").resolve()
    require(bool(raw) and Path(raw).resolve() == expected and expected.is_dir(),
            f"PROSPER_OUT must explicitly resolve to repository-local {expected}")
    return expected


@dataclass(frozen=True)
class Spec:
    n_new: int = 354
    raw_all: int = 2921
    raw_both: int = 2909
    eligible_all: int = 2778
    eligible_both: int = 2603
    control_tiles: int = 259
    manifest_sha: str = "31e4d95bdb91349f7ae4fed41ca988eb6026be7e4ee9905239e244cddde4d085"
    ids_sha: str = "b06270426145b98493a95f17c734f17617af3050fe7145c4a6fabe10b6acf56f"
    classifier_sha: str = "4a61c4b58bd6bcd1c65aca5eca24dc6196919ec1675841b925e791dbf1957771"
    metadata_sha: str = "45f1466c158f41ceea11abe06138356597a65ce1e97d2b072b780951864582b9"
    source_commit: str = "0edc479a0aa20f992a9445a34ce2ea799495e284"


CONTRACT = Spec()


@dataclass(frozen=True)
class Inputs:
    source_features: Path
    raw_all: Path
    raw_both: Path
    classifier: Path
    changed_manifest: Path
    metadata: Path
    old_filter_summary: Path
    old_control_feature: Path
    old_query_row: Path
    old_both_row: Path
    old_query_rank: Path
    old_both_rank: Path
    worker_root: Path


def production_inputs(out: Path) -> Inputs:
    score_dir = out / "st2p2/kidrare/features_in"
    return Inputs(
        source_features=out / "features/kidrare_uni2",
        raw_all=COMET / "comet-uni2/data/tmp_final",
        raw_both=out / "st2p2/library/features_in/tumor_only",
        classifier=(ROOT / "../comet-annotation/model/pathologist_annotated_slides_uni2/"
                    "tumor_detector_sgd_test_fraction_0.95_training.pkl").resolve(),
        changed_manifest=(out / "r1_3_external_correction_20260930/"
                          "kidrare_pathpt_changed_357_input_354_processed.csv"),
        metadata=(COMET / "comet-analysis/data/"
                  "final_scan_id_disease_map_with_sample_id_patient_id.csv"),
        old_filter_summary=out / "st2p2/library/features_in/_filter_summary.csv",
        old_control_feature=score_dir / f"tumor_only/{CONTROL}.pkl",
        old_query_row=(out / "similarity_st2p2_kidrare_tumor_only_ot" /
                       f"similarity_{CONTROL}.pkl"),
        old_both_row=(out / "similarity_st2p2_kidrare_tumor_only_both_ot" /
                      f"similarity_{CONTROL}.pkl"),
        old_query_rank=(out / "st2p2_external_kidrare_tumor_only/"
                        "ranked_lists_kidrare_uni2_ot_cosine.csv.gz"),
        old_both_rank=(out / "st2p2_external_kidrare_tumor_only_both/"
                       "ranked_lists_kidrare_uni2_ot_cosine.csv.gz"),
        worker_root=PROSPER)


def worker_files(root: Path) -> dict[str, Path]:
    return {name: root / name for name in (
        "scripts/lsf/prosper_job.sh",
        "prosper/preprocessing/tumor_tiles_cli.py",
        "prosper/embeddings/similarity_cli.py",
        "prosper/embeddings/slide_embedding/ot_embedding.py",
        "prosper/embeddings/slide_embedding/base.py",
        "prosper/embeddings/combine_similarity_cli.py",
        "prosper/evaluation/external_cli.py")}


def verify_worker_head(root: Path) -> None:
    relevant = worker_files(root)
    require(all(path.is_file() for path in relevant.values()),
            "Audited stock worker/helper file missing")
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *relevant],
        capture_output=True, text=True, check=False)
    require(result.returncode == 0,
            f"Audited stock worker/helper differs from pinned HEAD "
            f"(git diff exit {result.returncode}): {result.stderr}")


def runtime_info(stage: str) -> dict:
    require(stage in {"score", "filter", "ot", "verify"}, f"Unknown stage: {stage}")
    require(all(os.environ.get(k) == value for k, value in THREAD_ENV.items()),
            "Native thread/CUDA limits must be exported before Python starts")
    require(os.environ.get("PROSPER_PARAMS") in (None, ""),
            "Unset PROSPER_PARAMS; stock worker array mode is forbidden")
    lsf_index = os.environ.get("LSB_JOBINDEX")
    require(lsf_index in (None, "", "0"),
            f"One-query pilot tasks require a scalar LSF job index, got {lsf_index!r}")
    require(bool(os.environ.get("LSB_JOBID")),
            "Pilot stages must run in lead-submitted LSF jobs")
    import ot
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    require(all(p["num_threads"] == 1 for p in pools),
            f"Native runtime pool exceeds one thread: {pools}")
    require(ot.__version__ == "0.9.6.post1" and
            inspect.signature(ot.emd2).parameters["numThreads"].default == 1,
            "POT exact OT numThreads default changed")
    requested = 8 if stage == "ot" else 1
    slots = os.environ.get("LSB_DJOB_NUMPROC")
    require(slots is not None and int(slots) == requested,
            f"{stage} needs {requested} LSF slots, got {slots}")
    return {"thread_env": dict(THREAD_ENV), "thread_pools": pools,
            "pot_version": ot.__version__, "pot_emd2_numThreads": 1,
            "joblib_workers": 8 if stage == "ot" else 0,
            "affinity_cpus": sorted(os.sched_getaffinity(0)),
            "lsf_job": os.environ.get("LSB_JOBID"),
            "lsf_index": lsf_index,
            "lsf_slots": slots, "lsf_hosts": os.environ.get("LSB_HOSTS"),
            "lsf_queue": os.environ.get("LSB_QUEUE"),
            "lsf_cpu_hosts": os.environ.get("LSB_MCPU_HOSTS"),
            "lsf_resource_request": os.environ.get("LSB_RESREQ"),
            "lsf_memory_limit": os.environ.get("LSB_JOB_MEMLIMIT")}


def score_rows(path: Path, coords: list[dict], sid: str) -> pd.DataFrame:
    require(path.is_file(), f"{sid}: missing score file; not an empty slide")
    s = pd.read_csv(path)
    require(list(s.columns) == ["wsi_loc_x", "wsi_loc_y", "pred", "p_tumor"] and
            len(s) == len(coords), f"{sid}: score schema/row count changed")
    xy = np.asarray([x["wsi_loc"] for x in coords])
    require(xy.shape == (len(s), 2) and
            np.array_equal(xy, s[["wsi_loc_x", "wsi_loc_y"]].to_numpy()) and
            np.isfinite(s[["wsi_loc_x", "wsi_loc_y", "pred", "p_tumor"]]
                        .to_numpy(dtype=float)).all() and
            s.pred.isin([1, 2, 3, 4, 5]).all() and
            s.p_tumor.between(0, 1).all(),
            f"{sid}: score/feature coordinate, class or probability conflict")
    return s


def feature(path: Path, sid: str, nonempty: bool = True) -> tuple[np.ndarray, list[dict]]:
    require(path.is_file(), f"Missing {sid} feature file: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)
    require(isinstance(obj, tuple) and len(obj) == 2,
            f"{sid}: feature pickle is not (features, coordinates)")
    feats, coords = obj
    feats = np.asarray(feats)
    require(feats.ndim == 2 and feats.shape[1] == 1536 and
            len(feats) == len(coords) and (not nonempty or len(feats) > 0) and
            np.isfinite(feats).all(), f"{sid}: invalid feature bank")
    xy = np.asarray([c["wsi_loc"] for c in coords])
    require(xy.shape == (len(coords), 2) and np.issubdtype(xy.dtype, np.integer)
            and (xy >= 0).all() and len(set(map(tuple, xy))) == len(xy),
            f"{sid}: invalid/duplicate feature coordinates")
    return feats, coords


def row(path: Path, qid: str, reference_ids: list[str]) -> np.ndarray:
    require(path.is_file(), f"{qid}: missing similarity row: {path}")
    with path.open("rb") as f:
        entries = pickle.load(f)
    require(isinstance(entries, list) and len(entries) == len(reference_ids) and
            all(isinstance(x, tuple) and len(x) == 2 for x in entries) and
            [str(x[0]) for x in entries] == reference_ids,
            f"{qid}: incomplete/reordered/duplicate reference IDs")
    distances = np.asarray([x[1] for x in entries], dtype=float)
    require(np.isfinite(distances).all() and
            (distances >= 0).all() and (distances < 1e6).all(),
            f"{qid}: nonfinite/negative/sentinel reference distance")
    return distances


def compatibility(new: np.ndarray, original: np.ndarray) -> dict:
    require(new.shape == original.shape and len(new) > 0 and
            np.isfinite(new).all() and np.isfinite(original).all() and
            (new < 1e6).all() and (original < 1e6).all(),
            "Legacy reference distances incomplete or nonfinite")
    delta = np.abs(new - original)
    max_abs = float(delta.max())
    max_rel = float((delta / np.maximum(np.abs(original), 1e-12)).max())
    require(max_rel < 1e-6, f"Legacy OT relative error {max_rel} is not <1e-6")
    return {"max_absolute_discrepancy": max_abs, "max_relative_discrepancy": max_rel}


def top10(reference_ids: list[str], distances: np.ndarray, eligible: set[str],
          qid: str) -> list[str]:
    ids = [r for r in reference_ids if r in eligible]
    require(len(ids) >= 10 and qid not in ids, f"{qid}: insufficient/excluded references")
    lookup = dict(zip(reference_ids, distances))
    selected = np.array([lookup[r] for r in ids], dtype=float)
    require(np.isfinite(selected).all() and (selected < 1e6).all(),
            f"{qid}: invalid eligible top10 candidates")
    return [ids[i] for i in np.argsort(selected, kind="mergesort")[:10]]


def choose_three(nonempty: dict[str, int]) -> dict:
    require(len(nonempty) >= 3 and all(n > 0 for n in nonempty.values()),
            f"Need three distinct algorithmically nonempty new queries; have {len(nonempty)}")
    counts = np.array(list(nonempty.values()), dtype=float)
    target = {"median": float(np.percentile(counts, 50)),
              "p95": float(np.percentile(counts, 95)),
              "maximum": float(np.max(counts))}
    maximum = sorted(sid for sid, n in nonempty.items() if n == target["maximum"])
    selected = {"maximum": maximum[0]}
    ties = {"maximum": maximum}
    for label in ("median", "p95"):
        value = target[label]
        candidates = sorted((abs(n - value), sid) for sid, n in nonempty.items()
                            if sid not in selected.values())
        best = candidates[0][0]
        ties[label] = [sid for dist, sid in candidates if dist == best]
        selected[label] = candidates[0][1]
    require(len(set(selected.values())) == 3, "Pilot selection not unique")
    roles = ("median", "p95", "maximum")
    return {"selection": {label: selected[label] for label in roles},
            "targets": target,
            "distance_ties": {label: ties[label] for label in roles},
            "n_nonempty": len(nonempty), "retained_tiles": {
                label: nonempty[selected[label]] for label in roles},
            "target_distances": {
                label: abs(nonempty[selected[label]] - target[label])
                for label in roles}}


class Pilot:
    def __init__(self, out: Path, inputs: Inputs | None = None,
                 spec: Spec = CONTRACT) -> None:
        self.out = out.resolve()
        self.inputs = inputs or production_inputs(out)
        self.spec = spec
        self.root = out / DEST

    def file_inputs(self, new_ids: list[str]) -> dict[str, Path]:
        p = self.inputs
        files = {
            "manifest": p.changed_manifest, "classifier": p.classifier,
            "comet_metadata": p.metadata, "old_filter_summary": p.old_filter_summary,
            "control_feature": p.old_control_feature,
            "old_query_row": p.old_query_row, "old_both_row": p.old_both_row,
            "old_query_rank": p.old_query_rank, "old_both_rank": p.old_both_rank,
            "implementation": Path(__file__).resolve(),
            "implementation_tests": ROOT / "scripts/test_r1_3_kidrare_filter_extension.py",
        }
        files.update({f"code_{name}": path for name, path in
                      worker_files(p.worker_root).items()})
        files.update({f"source_{sid}": p.source_features / f"{sid}.pkl"
                      for sid in new_ids})
        files.update({f"raw_all_{sid}": p.raw_all / f"{sid}.pkl"
                      for sid in ids_in(p.raw_all)})
        files.update({f"raw_both_{sid}": p.raw_both / f"{sid}.pkl"
                      for sid in ids_in(p.raw_both)})
        require(all(path.is_file() for path in files.values()),
                f"Missing pilot input: {[k for k, v in files.items() if not v.is_file()][:5]}")
        return files

    def new_ids(self) -> list[str]:
        p = self.inputs.changed_manifest
        require(p.is_file() and digest(p) == self.spec.manifest_sha,
                "ADR0019 changed-label manifest hash changed")
        d = pd.read_csv(p, dtype=str, keep_default_na=False)
        require(len(d) == 357 if self.spec == CONTRACT else len(d) == self.spec.n_new,
                "Changed-label manifest row count differs")
        require(d.slide_id.is_unique and
                (d.pathpt_only_new_label == "True").all() and
                (d.corrected_tumor_group == "WT").all() and
                (d.original_tumor_group == "WT_UNVERIFIED").all() and
                (d.legacy_eligible == "False").all() and
                d.amended_eligible.eq(d.processed).all() and
                (d.query_only_eligible == "False").all() and
                (d.both_sided_eligible == "False").all(),
                "Changed-label eligibility/source annotations changed")
        selected = sorted(d.loc[d.processed == "True", "slide_id"])
        require(len(selected) == self.spec.n_new and
                hashlib.sha256(("".join(s + "\n" for s in selected)).encode()).hexdigest()
                == self.spec.ids_sha, "Exact sorted new-query set/hash changed")
        return selected

    def reference_ids(self) -> tuple[list[str], list[str], dict[str, set[str]]]:
        p = self.inputs
        require(p.metadata.is_file() and digest(p.metadata) == self.spec.metadata_sha,
                "Final COMET metadata hash changed from accepted fixed-reference binding")
        all_ids, both_ids = ids_in(p.raw_all), ids_in(p.raw_both)
        require(len(all_ids) == self.spec.raw_all and len(both_ids) == self.spec.raw_both
                and set(both_ids).issubset(set(all_ids)),
                "Raw reference-bank counts/membership changed")
        metadata = pd.read_csv(p.metadata, dtype=str, keep_default_na=False)
        require(metadata["Slide.Scan.File"].is_unique and
                len(metadata) == self.spec.raw_all and
                set(metadata["Slide.Scan.File"]) == set(all_ids),
                "Final COMET reference metadata mismatch")
        labels = dict(zip(metadata["Slide.Scan.File"], metadata["Tumor.group"]))
        eligible = {"query_only": {sid for sid in all_ids if labels[sid] != "MISC"},
                    "both_sided": {sid for sid in both_ids
                                   if labels[sid] not in ("MISC", "NORMAL")}}
        require(len(eligible["query_only"]) == self.spec.eligible_all and
                len(eligible["both_sided"]) == self.spec.eligible_both,
                "Eligible reference library counts changed")
        summary = pd.read_csv(p.old_filter_summary, dtype=str, keep_default_na=False)
        require(len(summary) == len(all_ids) and summary.slide.is_unique and
                set(summary.slide) == set(all_ids) and
                set(summary.loc[summary.kept_tumor_only.astype(int).gt(0), "slide"])
                == set(both_ids), "Prior library tumour filter bank/summary disagrees")
        for arm, source in (("query_only", p.old_query_row),
                            ("both_sided", p.old_both_row)):
            refs = all_ids if arm == "query_only" else both_ids
            dists = row(source, CONTROL, refs)
            ranked = pd.read_csv(p.old_query_rank if arm == "query_only"
                                 else p.old_both_rank, dtype={"query": str,
                                                              "neighbour": str})
            saved = ranked[ranked["query"] == CONTROL].sort_values("rank")
            require(len(saved) == 10 and saved["rank"].tolist() == list(range(1, 11))
                    and saved.neighbour.tolist() ==
                    top10(refs, dists, eligible[arm], CONTROL),
                    f"{arm}: saved control rank is not the frozen eligible top10")
        return all_ids, both_ids, eligible

    def bindings(self, new_ids: list[str]) -> dict:
        files = self.file_inputs(new_ids)
        return {k: {"path": str(v.resolve()), "sha256": digest(v)}
                for k, v in files.items()}

    def check_bindings(self, preflight: dict, new_ids: list[str],
                       full: bool = False) -> None:
        files = self.file_inputs(new_ids)
        original = preflight["input_bindings"]
        require(set(files) == set(original) and
                all(str(p.resolve()) == original[k]["path"] for k, p in files.items()),
                "Pilot input path/membership changed")
        check = files if full else {k: v for k, v in files.items()
                                    if not k.startswith(("source_", "raw_all_", "raw_both_"))}
        require(all(digest(v) == original[k]["sha256"] for k, v in check.items()),
                "Pilot source input changed")
        commit = subprocess.check_output(
            ["git", "-C", str(self.inputs.worker_root), "rev-parse", "HEAD"],
            text=True).strip()
        require(commit == preflight["source_worker_commit"] == self.spec.source_commit,
                "Stock worker commit changed during pilot")
        verify_worker_head(self.inputs.worker_root)

    def preflight(self) -> tuple[list[str], dict]:
        require(not self.root.exists(), f"Refusing pre-existing pilot root: {self.root}")
        runtime = runtime_info("score")
        require(self.inputs.classifier.is_file() and
                digest(self.inputs.classifier) == self.spec.classifier_sha,
                "Fixed five-class classifier hash changed")
        commit = subprocess.check_output(
            ["git", "-C", str(self.inputs.worker_root), "rev-parse", "HEAD"],
            text=True).strip()
        require(commit == self.spec.source_commit, "Stock worker commit changed")
        verify_worker_head(self.inputs.worker_root)
        ids = self.new_ids()
        all_ids, both_ids, eligible = self.reference_ids()
        for sid in ids:
            feature(self.inputs.source_features / f"{sid}.pkl", sid)
        control, _ = feature(self.inputs.old_control_feature, CONTROL)
        require(len(control) == self.spec.control_tiles,
                "Legacy control retained-tile count changed")
        bound = self.bindings(ids)
        return ids, {"schema_version": 1, "stage": "score_preflight",
                     "new_ids": ids, "reference_counts": {
                         "query_only_raw": len(all_ids), "both_sided_raw": len(both_ids),
                         "query_only_eligible": len(eligible["query_only"]),
                         "both_sided_eligible": len(eligible["both_sided"])},
                     "input_bindings": bound, "source_worker_commit": commit,
                     "runtime": runtime}

    def run_worker(self, module: str, args: list[str], stage: str) -> None:
        env = os.environ.copy()
        env.update(THREAD_ENV)
        env.update({"PROSPER_ROOT": str(self.inputs.worker_root.resolve()),
                    "COMET_ROOT": str(COMET), "PROSPER_OUT": str(self.out),
                    "PROSPER_PYTHON": str(PYTHON), "PROSPER_MODULE": module})
        env.pop("PROSPER_PARAMS", None)
        env.pop("LSB_JOBINDEX", None)
        require(PYTHON.is_file() and self.inputs.worker_root.is_dir() and
                all(env[k] == value for k, value in THREAD_ENV.items()),
                "Worker executable/root/native thread binding invalid")
        subprocess.run(["bash", str(self.inputs.worker_root /
                                    "scripts/lsf/prosper_job.sh"), *args],
                       cwd=self.inputs.worker_root, env=env, check=True)

    def score(self) -> None:
        ids, preflight = self.preflight()
        subset = self.root / "source_subset"
        subset.mkdir(parents=True, exist_ok=False)
        for sid in ids:
            (subset / f"{sid}.pkl").symlink_to(
                (self.inputs.source_features / f"{sid}.pkl").resolve())
        require(ids_in(subset) == ids and
                all((subset / f"{sid}.pkl").resolve() ==
                    (self.inputs.source_features / f"{sid}.pkl").resolve()
                    for sid in ids), "Subset symlinks differ from frozen354")
        json_write(self.root / "preflight.json", preflight)
        self.run_worker("prosper.preprocessing.tumor_tiles_cli",
                        ["score", "--features-dir", str(subset),
                         "--classifier", str(self.inputs.classifier),
                         "--out-dir", str(self.root / "scores"),
                         "--lo", "1", "--hi", str(len(ids))], "score")
        scores = self.verify_scores(ids)
        self.check_bindings(preflight, ids)
        json_write(self.root / "score_receipt.json",
                   {"stage": "score", "n_scored": len(scores),
                    "preflight_sha256": digest(self.root / "preflight.json"),
                    "score_sha256": {sid: digest(self.root / "scores" / f"{sid}.csv.gz")
                                    for sid in ids}, "runtime": runtime_info("score")})

    def preflight_receipt(self) -> tuple[list[str], dict]:
        pre = json_read(self.root / "preflight.json")
        ids = self.new_ids()
        require(pre["new_ids"] == ids and
                ids_in(self.root / "source_subset") == ids and
                all((self.root / "source_subset" / f"{sid}.pkl").is_symlink()
                    and (self.root / "source_subset" / f"{sid}.pkl").resolve() ==
                    (self.inputs.source_features / f"{sid}.pkl").resolve()
                    for sid in ids),
                "Pilot preflight selection/symlinks changed")
        self.check_bindings(pre, ids)
        return ids, pre

    def verify_scores(self, ids: list[str]) -> dict[str, pd.DataFrame]:
        directory = self.root / "scores"
        require(directory.is_dir() and
                {p.name for p in directory.iterdir()} ==
                {f"{sid}.csv.gz" for sid in ids} | {f"_score_1_{len(ids)}.json"},
                "Score output set incomplete/extra (stock CLI can skip existing files)")
        record = json_read(directory / f"_score_1_{len(ids)}.json")
        require(record["action"] == "score" and record["n_written"] == len(ids) and
                record["classes"] == [1, 2, 3, 4, 5] and
                record["classifier_sha256_16"] == self.spec.classifier_sha[:16] and
                Path(record["features_dir"]).resolve() ==
                (self.root / "source_subset").resolve() and
                Path(record["classifier"]).resolve() == self.inputs.classifier.resolve()
                and record["lo"] == 1 and record["hi"] == len(ids),
                "Incomplete/misaligned scoring record")
        result = {}
        for sid in ids:
            _, coords = feature(self.root / "source_subset" / f"{sid}.pkl", sid)
            result[sid] = score_rows(directory / f"{sid}.csv.gz", coords, sid)
        return result

    def filter(self) -> None:
        runtime = runtime_info("filter")
        ids, pre = self.preflight_receipt()
        require(not (self.root / "filter_receipt.json").exists() and
                not (self.root / "features_in").exists() and
                not (self.root / "selection.json").exists() and
                not (self.root / "pilot_queries").exists(),
                "Refusing existing filter/selection/staging outputs")
        score_rec = json_read(self.root / "score_receipt.json")
        require(score_rec["preflight_sha256"] == digest(self.root / "preflight.json"),
                "Score preflight binding changed")
        scores = self.verify_scores(ids)
        require(score_rec["score_sha256"] == {
            sid: digest(self.root / "scores" / f"{sid}.csv.gz") for sid in ids},
            "Scoring files changed since receipt")
        self.run_worker("prosper.preprocessing.tumor_tiles_cli",
                        ["filter", "--features-dir", str(self.root / "source_subset"),
                         "--scores-dir", str(self.root / "scores"),
                         "--out-root", str(self.root / "features_in"),
                         "--arms", "tumor_only", "--min-tiles", "8", "--skip-empty"],
                        "filter")
        nonempty = self.verify_filter(ids, scores)
        if len(nonempty) < 3:
            json_write(self.root / "selection_diagnostics.json",
                       {"stage": "filter", "n_nonempty": len(nonempty),
                        "required": 3, "no_replacement_selected": True})
            raise ValueError(f"Only {len(nonempty)} nonempty additions; no three-query pilot")
        selected = choose_three(nonempty)
        planned = self.tasks(selected)
        require(len(planned) == 8 and len({task for _, _, task in planned}) == 8
                and len({task.replace("/", "_") for _, _, task in planned}) == 8,
                "Pilot task/log namespace collision")
        for _, _, task in planned:
            require(not (self.out / task).exists() and
                    not (self.out / "logs" / task.replace("/", "_")).exists(),
                    f"Pilot task or normalized log already exists: {task}")
        json_write(self.root / "selection.json", {
            **selected, "new_query_ids": ids,
            "pilot_queries": sorted({*selected["selection"].values(), CONTROL}),
            "tasks": [{"arm": arm, "query": sid, "task": task}
                      for arm, sid, task in planned],
            "score_receipt_sha256": digest(self.root / "score_receipt.json"),
            "filter_summary_sha256": digest(self.root / "features_in/_filter_summary.csv"),
        })
        for arm, sid, _ in planned:
            dst = self.root / "pilot_queries" / arm / sid
            dst.mkdir(parents=True, exist_ok=False)
            src = (self.inputs.old_control_feature if sid == CONTROL else
                   self.root / "features_in/tumor_only" / f"{sid}.pkl")
            (dst / f"{sid}.pkl").symlink_to(src.resolve())
        json_write(self.root / "filter_receipt.json", {
            "stage": "filter", "n_new": len(ids), "n_nonempty": len(nonempty),
            "n_algorithmically_empty": len(ids) - len(nonempty),
            "selection_sha256": digest(self.root / "selection.json"),
            "filter_sha256": {sid: digest(self.root / "features_in/tumor_only" /
                                          f"{sid}.pkl") for sid in nonempty},
            "runtime": runtime, "preflight_sha256": digest(self.root / "preflight.json")})
        self.check_bindings(pre, ids)

    def verify_filter(self, ids: list[str], scores: dict[str, pd.DataFrame]
                      ) -> dict[str, int]:
        directory = self.root / "features_in"
        record = json_read(directory / "_filter_record.json")
        summary = pd.read_csv(directory / "_filter_summary.csv", dtype=str,
                              keep_default_na=False)
        required = {"slide", "n_tiles", "n_tumor", "n_immune", "n_necrosis",
                    "n_nontumor", "n_artifacts", "kept_tumor_only"}
        optional = {"skipped_tumor_only", "below_min_tumor_only"}
        columns = set(summary.columns)
        require(required <= columns <= required | optional,
                "Filter summary has missing required or foreign columns")
        has_skipped = "skipped_tumor_only" in columns
        has_below_min = "below_min_tumor_only" in columns
        require(record["action"] == "filter" and record["arms"] == ["tumor_only"] and
                record["n_slides"] == len(ids) and record["min_tiles"] == 8 and
                Path(record["features_dir"]).resolve() ==
                (self.root / "source_subset").resolve() and
                Path(record["scores_dir"]).resolve() ==
                (self.root / "scores").resolve() and
                Path(record["out_root"]).resolve() == directory.resolve() and
                len(summary) == len(ids) and summary.slide.is_unique and
                set(summary.slide) == set(ids) and
                {x.name for x in directory.iterdir()} ==
                {"tumor_only", "_filter_record.json", "_filter_summary.csv"},
                "Incomplete/foreign filter record or summary")
        kept = {}
        for entry in summary.itertuples(index=False):
            sid = entry.slide
            raw, coords = feature(self.root / "source_subset" / f"{sid}.pkl", sid)
            s = scores[sid]
            n = int(entry.kept_tumor_only)
            mask = s.pred.to_numpy() == 1
            require(int(entry.n_tiles) == len(raw) and int(entry.n_tumor) == n ==
                    int(mask.sum()) and
                    sum(int(getattr(entry, name)) for name in (
                        "n_tumor", "n_immune", "n_necrosis", "n_nontumor", "n_artifacts"))
                    == len(raw), f"{sid}: summary/mask count mismatch")
            dest = directory / "tumor_only" / f"{sid}.pkl"
            if n == 0:
                require(not dest.exists() and has_skipped and
                        entry.skipped_tumor_only == "True" and
                        (not has_below_min or entry.below_min_tumor_only == ""),
                        f"{sid}: zero retained tiles not explicitly skipped")
            else:
                below_min_valid = (has_below_min and entry.below_min_tumor_only == "True"
                                   if n < 8 else
                                   not has_below_min or entry.below_min_tumor_only == "")
                require(dest.is_file() and
                        (not has_skipped or entry.skipped_tumor_only == "") and
                        below_min_valid,
                        f"{sid}: missing nonempty feature or below-eight warning")
                filtered, filtered_coords = feature(dest, sid)
                require(np.array_equal(filtered, raw[mask]) and
                        filtered_coords == [coords[i] for i in np.flatnonzero(mask)],
                        f"{sid}: filtered features/coords do not match argmax class1")
                kept[sid] = n
        require({p.stem for p in (directory / "tumor_only").iterdir()} == set(kept),
                "Filtered feature output has missing or extra IDs")
        return kept

    def tasks(self, selection: dict) -> list[tuple[str, str, str]]:
        qids = [*selection["selection"].values(), CONTROL]
        require(len(set(qids)) == 4, "Pilot selection/control query collision")
        return [(arm, sid, f"{DEST}/pilot_ot/{arm}/{sid}")
                for arm in ARMS for sid in qids]

    def selection_receipt(self) -> tuple[list[str], dict, dict]:
        ids, pre = self.preflight_receipt()
        rec = json_read(self.root / "filter_receipt.json")
        sel = json_read(self.root / "selection.json")
        score_rec = json_read(self.root / "score_receipt.json")
        require(rec["selection_sha256"] == digest(self.root / "selection.json") and
                rec["preflight_sha256"] == digest(self.root / "preflight.json") and
                score_rec["preflight_sha256"] ==
                digest(self.root / "preflight.json") and
                score_rec["n_scored"] == len(ids) and
                score_rec["score_sha256"] == {
                    sid: digest(self.root / "scores" / f"{sid}.csv.gz") for sid in ids}
                and rec["n_new"] == len(ids) and
                sel["new_query_ids"] == ids and
                sel["score_receipt_sha256"] ==
                digest(self.root / "score_receipt.json") and
                sel["filter_summary_sha256"] ==
                digest(self.root / "features_in/_filter_summary.csv") and
                len(sel["tasks"]) == 8 and
                sel["tasks"] == [{"arm": a, "query": q, "task": task}
                                 for a, q, task in self.tasks(sel)],
                "Frozen selection/filter receipt changed")
        scores = self.verify_scores(ids)
        nonempty = self.verify_filter(ids, scores)
        require(choose_three(nonempty) == {k: sel[k] for k in (
            "selection", "targets", "distance_ties", "n_nonempty",
            "retained_tiles", "target_distances")}
                and rec["n_nonempty"] == len(nonempty) and
                rec["n_algorithmically_empty"] == len(ids) - len(nonempty)
                and rec["filter_sha256"] == {sid: digest(
                    self.root / "features_in/tumor_only" / f"{sid}.pkl")
                    for sid in nonempty}, "Retained counts/selection changed")
        for arm, sid, _ in self.tasks(sel):
            link = self.root / "pilot_queries" / arm / sid / f"{sid}.pkl"
            source = (self.inputs.old_control_feature if sid == CONTROL else
                      self.root / "features_in/tumor_only" / f"{sid}.pkl")
            require(link.is_symlink() and link.resolve() == source.resolve() and
                    ids_in(link.parent) == [sid],
                    f"{arm}/{sid}: pilot query symlink changed")
        return ids, pre, sel

    def check_row(self, arm: str, sid: str, task: str,
                  eligible: set[str], refs: list[str]) -> dict:
        row_path = self.out / task / f"similarity_{sid}.pkl"
        got = row(row_path, sid, refs)
        observed = top10(refs, got, eligible, sid)
        result = {"row_sha256": digest(row_path), "n_raw_references": len(refs),
                  "n_eligible_references": len(eligible), "top10_ids": observed}
        if sid == CONTROL:
            old = row(self.inputs.old_query_row if arm == "query_only" else
                      self.inputs.old_both_row, sid, refs)
            result.update(compatibility(got, old))
            rank = pd.read_csv(self.inputs.old_query_rank if arm == "query_only"
                               else self.inputs.old_both_rank,
                               dtype={"query": str, "neighbour": str})
            saved = rank[rank["query"] == CONTROL].sort_values("rank")
            require(saved["rank"].tolist() == list(range(1, 11)) and
                    observed == saved.neighbour.tolist(),
                    f"{arm}: control eligible top10 rank/ID differs from historical")
        log_path = self.out / "logs" / task.replace("/", "_")
        require(log_path.is_dir() and
                {p.name for p in log_path.iterdir()} == {"provenance_1_1.json"},
                f"{task}: missing/foreign normalized provenance sidecar")
        prov_path = log_path / "provenance_1_1.json"
        prov = json_read(prov_path)
        query_dir = self.root / "pilot_queries" / arm / sid
        lib_dir = self.inputs.raw_all if arm == "query_only" else self.inputs.raw_both
        require(prov["task"] == task and prov["encoder"] == "uni2" and
                prov["method"] == "ot" and prov["feature_dim"] == 1536 and
                prov["metric"] == "cosine" and prov["reg"] == 0 and
                prov["external"] is True and prov["num_workers"] == 8 and
                prov["n_library"] == len(refs) and prov["n_queries_total"] == 1 and
                prov["query_range"] == [1, 1] and
                prov["n_queries_this_element"] == 1 and prov["n_done"] == 1 and
                prov["tile_fraction"] is None and
                prov["git_commit"] == self.spec.source_commit and
                bool(prov["lsf_job"]) and prov.get("lsf_index") is None and
                Path(prov["query_features"]).resolve() == query_dir.resolve() and
                Path(prov["library_features"]).resolve() == lib_dir.resolve(),
                f"{task}: worker sidecar does not bind executed task")
        return {**result, "provenance_sha256": digest(prov_path),
                "source_job": prov["lsf_job"]}

    def ot(self, arm: str, sid: str) -> None:
        require(arm in ARMS, f"Invalid pilot arm: {arm}")
        runtime = runtime_info("ot")
        ids, pre, sel = self.selection_receipt()
        require(sid in sel["selection"].values() or sid == CONTROL,
                f"{sid}: query not in frozen pilot")
        task = next(x["task"] for x in sel["tasks"] if x["arm"] == arm and
                    x["query"] == sid)
        row_dir = self.out / task
        logs = self.out / "logs" / task.replace("/", "_")
        receipt = self.root / f"receipt_{arm}_{sid}.json"
        require(not row_dir.exists() and not logs.exists() and not receipt.exists(),
                f"Refusing pre-existing task/normalized logs/receipt: {task}")
        all_ids, both_ids, eligible = self.reference_ids()
        lib = self.inputs.raw_all if arm == "query_only" else self.inputs.raw_both
        refs = all_ids if arm == "query_only" else both_ids
        query_dir = self.root / "pilot_queries" / arm / sid
        self.run_worker("prosper.embeddings.similarity_cli",
                        ["--encoder", "uni2", "--method", "ot",
                         "--feature-dim", "1536", "--metric", "cosine",
                         "--reg", "0", "--num-workers", "8",
                         "--features", str(query_dir),
                         "--library-features", str(lib),
                         "--lo", "1", "--hi", "1", "--task", task], "ot")
        require(row_dir.is_dir() and
                {p.name for p in row_dir.iterdir()} == {f"similarity_{sid}.pkl"},
                f"{task}: row output set incomplete/extra")
        result = self.check_row(arm, sid, task, eligible[arm], refs)
        require(result["source_job"] == runtime["lsf_job"],
                f"{task}: worker provenance job differs from active pilot job")
        json_write(receipt, {
            "stage": "ot", "arm": arm, "query": sid, "task": task,
            "selection_sha256": digest(self.root / "selection.json"),
            "preflight_sha256": digest(self.root / "preflight.json"),
            "result": result, "runtime": runtime})
        self.check_bindings(pre, ids)

    def verify(self) -> None:
        runtime = runtime_info("verify")
        ids, pre, sel = self.selection_receipt()
        all_ids, both_ids, eligible = self.reference_ids()
        records = []
        for entry in sel["tasks"]:
            arm, sid, task = (entry[k] for k in ("arm", "query", "task"))
            receipt = json_read(self.root / f"receipt_{arm}_{sid}.json")
            refs = all_ids if arm == "query_only" else both_ids
            result = self.check_row(arm, sid, task, eligible[arm], refs)
            require(receipt["stage"] == "ot" and
                    (receipt["arm"], receipt["query"], receipt["task"]) ==
                    (arm, sid, task) and
                    receipt["selection_sha256"] == digest(self.root / "selection.json")
                    and receipt["preflight_sha256"] ==
                    digest(self.root / "preflight.json") and
                    receipt["result"] == result and
                    receipt["runtime"]["lsf_job"] == result["source_job"],
                    f"{arm}/{sid}: OT receipt/source changed")
            records.append({"arm": arm, "query": sid, "task": task,
                            "receipt_sha256": digest(
                                self.root / f"receipt_{arm}_{sid}.json"),
                            "result": result})
        self.check_bindings(pre, ids, full=True)
        marker = self.root / "PILOT_COMPLETE.json"
        require(not marker.exists(), "Refusing pre-existing pilot verification marker")
        json_write(marker, {
            "status": "technical_pilot_complete_for_independent_review",
            "not_scientific_or_author_gate_c_approval": True,
            "n_new_scored": len(ids), "n_ot_tasks": len(records),
            "preflight_sha256": digest(self.root / "preflight.json"),
            "selection_sha256": digest(self.root / "selection.json"),
            "score_receipt_sha256": digest(self.root / "score_receipt.json"),
            "filter_receipt_sha256": digest(self.root / "filter_receipt.json"),
            "tasks": records, "runtime": runtime})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("score", "filter", "ot", "verify"),
                        required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--query")
    args = parser.parse_args(argv)
    if args.stage == "ot":
        require(args.arm is not None and args.query is not None,
                "OT stage requires both --arm and --query")
    else:
        require(args.arm is None and args.query is None,
                "Non-OT stages refuse --arm and --query")
    pilot = Pilot(out_root())
    if args.stage == "score":
        pilot.score()
    elif args.stage == "filter":
        pilot.filter()
    elif args.stage == "ot":
        pilot.ot(args.arm, args.query)
    else:
        pilot.verify()


if __name__ == "__main__":
    main()
