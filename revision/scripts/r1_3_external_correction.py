#!/usr/bin/env python3
"""[decision record]: correct external-query eligibility and uncertainty from saved ranks.

Requires PROSPER_OUT to explicitly name this repository's output directory.
Never reads similarity matrices, extracts features, or modifies source artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "../prosper-search").resolve()))
from prosper.evaluation import external_cli, metrics  # noqa: E402

DEST = "r1_3_external_correction_20260930"
K = (1, 3, 5, 10)
B = 5000
SEED = 123
METHODS = {"median_pooling": "median_pooling", "whole_slide_ot": "ot_cosine",
           "sampler": "sampler", "yottixel": "yottixel"}
EXPECTED_PAIRS = {
    ("stv", "main_696", method) for method in ("median_pooling", "sampler", "yottixel")
} | {
    ("kidrare", population, method)
    for population in ("amended_1581", "legacy_1227")
    for method in ("median_pooling", "sampler", "yottixel")
} | {
    ("stv", "query_only_696", "query_only"),
    ("stv", "both_sided_688", "both_sided"),
    ("kidrare", "query_only_1226", "query_only"),
    ("kidrare", "both_sided_1176", "both_sided"),
}
PRIMARY = ("micro_hit_at_5", "macro_hit_at_5", "micro_precision_at_5",
           "macro_precision_at_5")
ZERO_LIVER = "NULL-20250103-133602"
R1_2_SHA = "6b62801c9fb944eb210082a16662c993036a6dd800c758eb535021f7a002e8c4"
STV_SOURCE_SHA = "899862746d3d503f9989e3f2724bb028201da8804821a61bcfe669bf64ff11be"
PATHPT_SHA = "de46ce4d04c8aca66d1881260d875b6dbb956f8469873c00d55a95c8cc3e9d2e"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def unique(df: pd.DataFrame, col: str, name: str) -> None:
    require(col in df and df[col].notna().all() and df[col].astype(str).str.len().gt(0).all(),
            f"{name}: missing/blank {col}")
    require(df[col].is_unique, f"{name}: duplicate {col}")


def out_root() -> Path:
    raw = os.environ.get("PROSPER_OUT")
    require(bool(raw), "PROSPER_OUT must be explicitly set")
    target = (ROOT / "prosper_revision_output").resolve()
    require(Path(raw).resolve() == target and target.is_dir(),
            f"PROSPER_OUT must be the existing repository-local {target}")
    return target


def refuse_nonempty(dest: Path) -> None:
    require(not dest.exists() or (dest.is_dir() and not any(dest.iterdir())),
            f"Refusing nonempty/non-directory destination: {dest}")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def sources(out: Path) -> dict[str, Path]:
    p = {
        "stv_source": ROOT / "data/COMET_validation_set_deid_20260902.csv",
        "stv_staged_copy": out / "external_stv/COMET_validation_set_deid_20260902.csv",
        "stv_staging": out / "external_stv/_staging_record.json",
        "stv_all_manifest": out / "external_stv/manifest_all.csv",
        "stv_labelled_manifest": out / "external_stv/manifest_labelled.csv",
        "stv_inscope_manifest": out / "external_stv/manifest_inscope.csv",
        "stv_qc": out / "qc_stv/zero_tile_scan.csv",
        "kid_manifest": out / "external_kidrare/manifest.csv",
        "kid_all_manifest": out / "external_kidrare/manifest_all.csv",
        "kid_inscope_manifest": out / "external_kidrare/manifest_inscope.csv",
        "kid_staging": out / "external_kidrare/_staging_record.json",
        "pathpt": out / "external_kidrare/keep_labels/pathpt_division.json",
        "keep_wt": out / "external_kidrare/keep_labels/Nephroblastoma_labels.json",
        "library_filter_summary": out / "st2p2/library/features_in/_filter_summary.csv",
        "library_filter_record": out / "st2p2/library/features_in/_filter_record.json",
        "comet_metadata": (ROOT / "../comet-feature-gen/comet-analysis/data/"
                           "final_scan_id_disease_map_with_sample_id_patient_id.csv").resolve(),
        "metrics_helper": Path(metrics.__file__).resolve(),
        "external_helper": Path(external_cli.__file__).resolve(),
        "implementation": Path(__file__).resolve(),
        "implementation_tests": ROOT / "scripts/test_r1_3_external_correction.py",
        "ccdi_complete": out / "r1_2_scope_correction_20260930/COMPLETE.json",
        "ccdi_provenance": out / "r1_2_scope_correction_20260930/provenance.json",
    }
    for cohort in ("stv", "kidrare"):
        for name, method in METHODS.items():
            for suffix, prefix, ext in (("ranked", "ranked_lists_", ".csv.gz"),
                                        ("per_query", "per_query_", ".csv"),
                                        ("json", "external_", ".json")):
                folder = "external_stv_labelled" if cohort == "stv" else "external_kidrare_all"
                p[f"{cohort}_{name}_{suffix}"] = out / folder / (
                    f"{prefix}{cohort}_uni2_{method}{ext}")
                if cohort == "kidrare":
                    p[f"kidrare_legacy_{name}_{suffix}"] = (
                        out / "external_kidrare_inscope" /
                        f"{prefix}kidrare_uni2_{method}{ext}")
        for arm, folder in (("query_only", "tumor_only"),
                            ("both_sided", "tumor_only_both")):
            for suffix, prefix, ext in (("ranked", "ranked_lists_", ".csv.gz"),
                                        ("per_query", "per_query_", ".csv"),
                                        ("json", "external_", ".json")):
                p[f"{cohort}_{arm}_{suffix}"] = (
                    out / f"st2p2_external_{cohort}_{folder}" /
                    f"{prefix}{cohort}_uni2_ot_cosine{ext}")
    require(all(path.is_file() for path in p.values()),
            f"Missing source(s): {[k for k, v in p.items() if not v.is_file()]}")
    return p


def pathpt_kidney(data: dict, manifest: pd.DataFrame, keep: dict) -> dict[str, str]:
    sub = data["Nephroblastoma"]
    labels = {str(v): str(k) for k, v in sub["name2label"].items()}
    require(len(labels) == 5 and len(set(labels.values())) == 5,
            "PathPT kidney subtype dictionary changed")
    mapped = {}
    for split in ("train_IDs", "test_IDs"):
        require(set(sub[split]) == set(labels), "PathPT subtype coverage changed")
        for category, ids in sub[split].items():
            for raw in ids:
                require(raw.endswith(".ibl"), f"PathPT unexpected ID: {raw}")
                slide = raw[:-4]
                require(slide not in mapped, f"PathPT duplicate/cross-subtype: {slide}")
                mapped[slide] = labels[category]
    unique(manifest, "slide_id", "KidRare manifest")
    require(len(mapped) == 416 and set(mapped).issubset(set(manifest.slide_id)),
            "PathPT kidney source coverage changed")
    wt = manifest[manifest.kidrare_class.eq("Nephroblastoma")].set_index("slide_id")
    require(set(mapped).issubset(set(wt.index)), "PathPT labels include non-kidney slides")
    require(set(keep).issubset({x + ".ibl" for x in wt.index}), "KEEP kidney ID absent")
    for raw, label in keep.items():
        require(label in {"Tumor", "Normal"} and
                wt.loc[raw[:-4], "keep_subtype"] == label,
                "KEEP kidney label disagrees with manifest")
    tumor = set(wt.index[wt.keep_subtype.eq("Tumor")])
    normal = set(wt.index[wt.keep_subtype.eq("Normal")])
    require(len(tumor) == 59 and len(normal) == 51 and tumor.issubset(mapped) and
            not normal.intersection(mapped), "KEEP/PathPT Tumor or Normal conflict")
    changed = set(mapped) - tumor
    require(len(changed) == 357 and
            set(wt.loc[list(changed), "tumor_group"]) == {"WT_UNVERIFIED"},
            "Approved 357 PathPT-only WT labels changed")
    return mapped


def kid_manifest(paths: dict[str, Path], processed: set[str]) -> pd.DataFrame:
    original = csv(paths["kid_manifest"])
    require(original.equals(csv(paths["kid_all_manifest"])), "KidRare source/all mismatch")
    unique(original, "slide_id", "KidRare source")
    legacy = csv(paths["kid_inscope_manifest"])
    unique(legacy, "slide_id", "KidRare original inscope")
    require(len(original) == 2331 and len(legacy) == 1233 and
            set(legacy.slide_id) == set(original.loc[
                original.tumor_group.isin({"LIVER", "NB", "WT", "NORMAL"}), "slide_id"]),
            "KidRare staged membership changed")
    for col in original.columns:
        require(legacy[col].equals(original.set_index("slide_id").loc[
            legacy.slide_id, col].reset_index(drop=True)),
            f"KidRare legacy manifest disagrees: {col}") if col != "slide_id" else None
    require(len(processed) == 2322 and processed.issubset(set(original.slide_id)),
            "KidRare processed 2322 set changed")
    with paths["pathpt"].open() as f:
        pt = json.load(f)
    with paths["keep_wt"].open() as f:
        keep = json.load(f)
    mapped = pathpt_kidney(pt, original, keep)
    d = original.copy()
    d["processed"] = d.slide_id.isin(processed)
    d["pathpt_subtype"] = d.slide_id.map(mapped).fillna("")
    d["original_tumor_group"] = d.tumor_group
    d["corrected_tumor_group"] = d.tumor_group
    changed = d.kidrare_class.eq("Nephroblastoma") & d.tumor_group.eq("WT_UNVERIFIED") & d.pathpt_subtype.ne("")
    d.loc[changed, "corrected_tumor_group"] = "WT"
    d["corrected_label_source"] = np.where(changed, "PathPT archived subtype",
                                             d.label_source)
    d["pathpt_only_new_label"] = changed
    d["legacy_eligible"] = d.processed & d.original_tumor_group.isin(
        {"LIVER", "NB", "WT", "NORMAL"})
    d["amended_eligible"] = d.legacy_eligible | (d.processed & changed)
    unassigned = d.processed & d.kidrare_class.eq("Nephroblastoma") & d.pathpt_subtype.eq("") & d.original_tumor_group.eq("WT_UNVERIFIED")
    require(changed.sum() == 357 and (changed & d.processed).sum() == 354 and
            unassigned.sum() == 259 and
            (d.processed & d.kidrare_class.eq("Medulloblastoma")).sum() == 482 and
            d.legacy_eligible.sum() == 1227 and d.amended_eligible.sum() == 1581 and
            (d.amended_eligible & d.corrected_tumor_group.eq("WT")).sum() == 413,
            "KidRare approved membership margins changed")
    require(d.loc[changed & d.processed, "pathpt_subtype"].value_counts().to_dict() == {
        "Mixed blastemal and stromal nephroblastoma": 32,
        "Nephroblastoma, blastemal type": 4,
        "Mixed blastemal, epithelial and stromal nephroblastoma": 175,
        "Mixed blastemal and epithelial nephroblastoma": 55,
        "Nephroblastoma, stromal type": 88},
        "KidRare PathPT subtype margins changed")
    require(d.loc[~changed, "original_tumor_group"].equals(
        d.loc[~changed, "corrected_tumor_group"]), "Other diagnoses changed")
    return d


def stv_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    authoritative = csv(paths["stv_source"])
    require(authoritative.equals(csv(paths["stv_staged_copy"])),
            "STV authoritative and staged source differ")
    staging = json.loads(paths["stv_staging"].read_text())
    require(staging["sha256"] == STV_SOURCE_SHA and
            staging["n_labelled"] == 782 and
            staging["n_patients_labelled"] == 394 and
            staging["labelled_qc"] == {"ok": 776, "truncated": 6},
            "STV staging provenance changed")
    unique(authoritative, "file_name", "STV782 source")
    require(len(authoritative) == 782 and authoritative["de-id_uid"].nunique() == 394,
            "STV source counts changed")
    staged = csv(paths["stv_all_manifest"])
    unique(staged, "slide_id", "STV staged")
    unique(staged, "file_name", "STV staged")
    require(len(staged) == 1247 and set(authoritative.file_name).issubset(set(staged.file_name)),
            "STV source/staged membership changed")
    labelled = staged.set_index("file_name").loc[authoritative.file_name].reset_index()
    require((labelled.patient_id.to_numpy() == authoritative["de-id_uid"].to_numpy()).all()
            and (labelled.stv_dx.to_numpy() == authoritative.dx.to_numpy())
            .all() and (labelled.stv_subclass.to_numpy() == authoritative.subclass.to_numpy()).all()
            and labelled.patient_id.ne("").all() and
            labelled.groupby("patient_id").tumor_group.nunique().eq(1).all(),
            "STV staging/source annotations changed")
    qc = csv(paths["stv_qc"])
    unique(qc, "slide_id", "STV QC")
    require(set(staged.slide_id) == set(qc.slide_id), "STV QC membership changed")
    require(len(labelled) == 782 and labelled.qc.value_counts().to_dict() ==
            {"ok": 776, "truncated": 6}, "STV QC annotation changed")
    saved = csv(paths["stv_labelled_manifest"])
    in_scope = csv(paths["stv_inscope_manifest"])
    unique(saved, "slide_id", "STV saved labelled")
    unique(in_scope, "slide_id", "STV saved in-scope")
    processed = labelled[labelled.qc.eq("ok")]
    require(set(processed.slide_id) == set(saved.slide_id) and
            processed.set_index("slide_id").sort_index().equals(
                saved.set_index("slide_id").sort_index()) and
            len(saved) == 776 and saved.patient_id.nunique() == 393 and
            saved.groupby("patient_id").tumor_group.nunique().eq(1).all() and
            set(in_scope.slide_id) == set(saved.loc[saved.tumor_group.ne("MISC") &
                saved.tumor_group.ne("OOS"), "slide_id"]) and
            len(in_scope) == 696 and in_scope.patient_id.nunique() == 350,
            "STV labelled/in-scope manifest changed")
    for col in saved.columns:
        require(in_scope[col].equals(saved.set_index("slide_id").loc[
            in_scope.slide_id, col].reset_index(drop=True)),
            f"STV in-scope value differs: {col}") if col != "slide_id" else None
    d = labelled.copy()
    d["processed"] = d.qc.eq("ok")
    d["main_eligible"] = d.processed & d.tumor_group.isin(
        set(in_scope.tumor_group))
    d["both_eligible"] = d.main_eligible & d.tumor_group.ne("NORMAL")
    d["overlap_with_COMET"] = "unconfirmed; author-reported none; Mia confirmation pending"
    d["RET_to_RB"] = "provisional; Mia source confirmation pending"
    d["collection"] = "later COMET/St. Jude extension; cutoff after 2019 author-reported"
    require(d.main_eligible.sum() == 696 and d.both_eligible.sum() == 688 and
            d.loc[d.both_eligible, "patient_id"].nunique() == 346 and
            int(d.stv_dx.eq("RET").sum()) == 32 and
            d.loc[d.stv_dx.eq("RET"), "tumor_group"].eq("RB").all() and
            int((d.main_eligible & d.tumor_group.eq("NORMAL")).sum()) == 8 and
            (d.patient_id.str.startswith("ST_")).all(),
            "STV main/both eligibility changed")
    return d


def cluster_weights(pop: pd.DataFrame, patient_col: str, b: int = B,
                    seed: int = SEED) -> np.ndarray:
    unique(pop, "slide_id", "bootstrap slides")
    require(len(pop) > 0 and pop.tumor_group.notna().all() and
            pop.tumor_group.astype(str).str.len().gt(0).all(),
            "Missing/blank bootstrap truth")
    require(patient_col in pop and pop[patient_col].notna().all() and
            pop[patient_col].astype(str).str.len().gt(0).all(),
            "Missing/blank patient cluster")
    require(pop.groupby(patient_col).tumor_group.nunique().eq(1).all(),
            "Cross-class patient cluster; stratified bootstrap invalid")
    rng = np.random.default_rng(seed)
    w = np.zeros((b, len(pop)), dtype=np.int16)
    truth = pop.tumor_group.to_numpy()
    patients = pop[patient_col].to_numpy()
    for label in sorted(set(truth)):
        ix = np.flatnonzero(truth == label)
        ids, inverse = np.unique(patients[ix], return_inverse=True)
        require(len(ids) > 0, f"Missing patient stratum: {label}")
        for first in range(0, b, 128):
            last = min(first + 128, b)
            draws = rng.integers(0, len(ids), size=(last - first, len(ids)))
            counts = np.zeros((last - first, len(ids)), dtype=np.int16)
            np.add.at(counts, (np.arange(last - first)[:, None], draws), 1)
            w[first:last, ix] = counts[:, inverse]
        require((w[:, ix].sum(axis=1) > 0).all(), f"Lost stratum: {label}")
    require(np.all(w >= 0) and np.all(w <= len(pop)),
            "Invalid whole-cluster multiplicities")
    return w


def interval(values: np.ndarray) -> dict:
    require(len(values) > 0 and np.isfinite(values).all(),
            "Nonfinite or absent bootstrap outcome")
    return {"lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5)), "B_effective": len(values)}


def bootstrap_description(cohort: str, weights: np.ndarray,
                          observed_classes: list[str]) -> dict:
    require(cohort in {"stv", "kidrare"}, f"Unknown bootstrap cohort: {cohort}")
    return {
        "B": B, "seed": SEED, "percentiles": [2.5, 97.5],
        "resampling_unit": "supplied ST_ patient ID" if cohort == "stv" else "slide (nominal)",
        "design": ("Class-stratified whole-patient-cluster resampling with replacement; "
                   "all eligible slides per drawn patient with the same multiplicity"
                   if cohort == "stv" else
                   "Class-stratified slide resampling with replacement; no verified patient "
                   "IDs and no patient-level uncertainty inference"),
        "min_resampled_slides": int(weights.sum(axis=1).min()),
        "max_resampled_slides": int(weights.sum(axis=1).max()),
        "fixed_observed_classes": sorted(observed_classes),
        "fixed_reference_library": True,
    }


def validate_ranks(r: pd.DataFrame, q: pd.DataFrame, allowed: set[str],
                   lib: pd.Series, name: str) -> None:
    require(len(r) == len(q) * 10 and r["query"].notna().all() and
            set(r["query"]) == set(q["query"]) and
            r.groupby("query")["rank"].agg(
                lambda s: set(s) == set(range(1, 11)) and len(s) == 10).all() and
            not r.duplicated(["query", "neighbour"]).any() and
            r["neighbour"].notna().all() and
            set(r["neighbour"]).issubset(allowed) and
            (r.neighbour_label.to_numpy() ==
             r.neighbour.map(lib).to_numpy()).all() and
            np.isfinite(r.distance.to_numpy(dtype=float)).all(),
            f"{name}: rank depth/uniqueness/COMET reference labels invalid")


def validate_library_filter(paths: dict[str, Path], metadata: pd.DataFrame) -> None:
    summary = csv(paths["library_filter_summary"])
    record = json.loads(paths["library_filter_record"].read_text())
    unique(summary, "slide", "saved library tumour filter")
    lib = metadata.set_index("Slide.Scan.File")["Tumor.group"]
    zero = summary.loc[summary.n_tumor.eq("0"), "slide"]
    kept = summary.loc[summary.kept_tumor_only.astype(int).gt(0), "slide"]
    require(record["n_slides"] == 2921 and record["arms"] == ["tumor_only"] and
            record["min_tiles"] == 8 and len(summary) == 2921 and
            set(summary.slide) == set(lib.index) and
            len(kept) == 2909 and len(zero) == 12 and
            set(zero) == set(summary.slide) - set(kept) and
            zero.eq("16738").sum() == 1 and
            lib.loc[zero].value_counts().to_dict() == {"NORMAL": 11, "RB": 1} and
            set(kept[~lib.loc[kept].isin({"MISC", "NORMAL"}).to_numpy()]) ==
            set(lib.index[~lib.isin({"MISC", "NORMAL"})]) - {"16738"},
            "Raw 2909 / eligible 2603 filtered library membership changed")


def compare_per_query_frames(expected: pd.DataFrame, actual: pd.DataFrame,
                             name: str) -> None:
    require(list(expected.columns) == list(actual.columns) and
            len(expected) == len(actual), f"{name}: per-query schema/count changed")
    for col in expected:
        if pd.api.types.is_numeric_dtype(actual[col]):
            a = pd.to_numeric(expected[col], errors="coerce").to_numpy(dtype=float)
            v = pd.to_numeric(actual[col], errors="coerce").to_numpy(dtype=float)
            atol = 1e-10 if col in ("top1_distance", "mean_top5_distance") else 1e-12
            require(np.allclose(a, v, atol=atol, rtol=0, equal_nan=True),
                    f"{name}: per-query numeric outcome differs: {col}")
        else:
            require(expected[col].fillna("").astype(str).equals(
                actual[col].fillna("").astype(str)),
                f"{name}: per-query categorical outcome differs: {col}")


def replay(name: str, cohort: str, paths: dict[str, Path], metadata: pd.DataFrame,
           labels: dict[str, str], expected: set[str], expected_source_n: int
           ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stem = f"{cohort}_{name}"
    source = json.loads(paths[f"{stem}_json"].read_text())
    cfg = source["config"]
    both = name == "both_sided"
    matrix_name = (f"combined_similarity_matrix_st2p2_{cohort}_"
                   f"{'tumor_only_both' if both else 'tumor_only'}_ot_cosine.pkl"
                   if name in ("query_only", "both_sided") else
                   f"combined_similarity_matrix_{cohort}_uni2_"
                   f"{METHODS[name.removeprefix('legacy_')]}.pkl")
    require(cfg["cohort"] == cohort and cfg["model"] == "uni2" and
            cfg["method"] == (METHODS.get(name.removeprefix("legacy_")) or "ot_cosine") and
            Path(cfg["matrix"]).name == matrix_name and
            Path(cfg["matrix"]).parent.name == "similarity_matrix" and
            cfg["query_id_col"] == "slide_id" and
            cfg["query_label_col"] == "tumor_group" and
            cfg["query_patient_col"] == ("patient_id" if cohort == "stv" else "slide_id") and
            cfg["depth"] == 10 and cfg["k_values"] == list(K) and
            cfg["self_check_mode"] is False and
            cfg["library_drop_labels"] == (["MISC", "NORMAL"] if both else ["MISC"]) and
            cfg["n_library"] == (2603 if both else 2778) and
            source["n_queries"] == expected_source_n and
            source["queries_with_fewer_than_depth_neighbours"] == 0,
            f"{stem}: frozen configuration changed")
    unique(metadata, "Slide.Scan.File", "COMET reference")
    lib = metadata.set_index("Slide.Scan.File")["Tumor.group"]
    require(len(lib) == 2921 and lib.notna().all() and
            len(lib[~lib.isin({"MISC"})]) == 2778 and
            len(lib[~lib.isin({"MISC", "NORMAL"})]) == 2604,
            "COMET metadata/library class margins changed")
    require(lib.get("16738") == "RB", "Both-sided zero-tumour RB16738 reference changed")
    allowed = lib.index[~lib.isin(cfg["library_drop_labels"])]
    if both:
        allowed = allowed.drop("16738")
    require(len(allowed) == cfg["n_library"] and
            Counter(lib.loc[allowed]) == cfg["library_class_counts"] and
            sorted(set(lib.loc[allowed])) == cfg["library_classes"],
            f"{stem}: saved reference library disagrees with metadata")
    q = pd.read_csv(paths[f"{stem}_per_query"], dtype={"query": str, "true_label": str})
    r = pd.read_csv(paths[f"{stem}_ranked"], dtype={"query": str, "neighbour": str})
    unique(q, "query", stem)
    require(len(q) == expected_source_n and set(q["query"]) == set(labels) and
            set(expected).issubset(set(labels)), f"{stem}: query membership changed")
    require(q.set_index("query").loc[list(labels), "true_label"].to_dict() == labels,
            f"{stem}: source query truth differs")
    validate_ranks(r, q, set(allowed), lib, stem)
    r = r.sort_values(["query", "rank"]).set_index(["query", "rank"], drop=False)
    ordered = r.loc[pd.MultiIndex.from_product([q["query"], range(1, 11)])]
    labs = ordered.neighbour_label.to_numpy().reshape(len(q), 10)
    ids = ordered.neighbour.to_numpy().reshape(len(q), 10)
    distances = ordered.distance.to_numpy(dtype=float).reshape(len(q), 10)
    regenerated = external_cli.per_query_frame(
        q["query"].tolist(), q.true_label.tolist(), ids, labs, distances, K, 10)
    compare_per_query_frames(q, regenerated, f"{stem} original replay")
    scored = q["query"].isin(expected).to_numpy()
    truth = q.loc[scored, "true_label"].tolist()
    point = metrics.compute_all(labs[scored], truth, K,
                                n_relevant=np.array([
                                    cfg["library_class_counts"][t] for t in truth]))
    require(len(point) == 44 and set(point) == set(source["point_estimates"]) and
            all(np.isclose(v, source["point_estimates"][key], atol=1e-12, rtol=0)
                for key, v in point.items()) and
            source["query_class_counts"] == dict(Counter(truth)) and
            source["n_in_library_queries"] == len(truth),
            f"{stem}: saved summary does not reproduce from ranked neighbours")
    for k in K:
        for rule in ("generous", "absolute", "closest"):
            actual = external_cli._f1_block(
                truth, q.loc[scored, f"pred_{rule}_k{k}"].tolist(), cfg["library_classes"])
            require(all(np.isclose(actual[x], source["voting_f1"][f"{rule}_k{k}"][x],
                                   rtol=0, atol=1e-12)
                        for x in ("macro_f1", "micro_f1", "accuracy",
                                  "macro_precision", "macro_recall")),
                    f"{stem}: source voting summary differs: {rule}/{k}")
    return q.set_index("query", drop=False), r, source


def derive(pop: pd.DataFrame, q: pd.DataFrame, r: pd.DataFrame,
           config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = pop.slide_id.tolist()
    require(len(set(ids)) == len(ids) and set(ids).issubset(set(q.index)),
            "Derived query membership missing/duplicate")
    ranks = r.loc[pd.MultiIndex.from_product([ids, range(1, 11)])].copy()
    original_q = q.loc[ids].copy()
    labs = ranks.neighbour_label.to_numpy().reshape(len(ids), 10)
    nbr_ids = ranks.neighbour.to_numpy().reshape(len(ids), 10)
    distances = ranks.distance.to_numpy(dtype=float).reshape(len(ids), 10)
    truth = pop.tumor_group.tolist()
    require(set(truth).issubset(set(config["library_classes"])),
            "Unsupported truth class in derived population")
    revised = external_cli.per_query_frame(ids, truth, nbr_ids, labs, distances, K, 10)
    revised.attrs["ranked_labels"] = labs
    revised.attrs["source_query"] = original_q
    return revised, ranks.reset_index(drop=True)


def score(pop: pd.DataFrame, q: pd.DataFrame, cfg: dict, weights: np.ndarray
          ) -> tuple[dict, dict[str, np.ndarray]]:
    truth = pop.tumor_group.to_numpy()
    require(q["query"].tolist() == pop.slide_id.tolist() and
            np.array_equal(q.true_label.to_numpy(), truth) and
            weights.shape[1] == len(pop), "Scoring population/weights misaligned")
    classes = sorted(set(truth))
    lab = q.attrs["ranked_labels"]
    point = metrics.compute_all(lab, truth, K, n_relevant=np.array([
        cfg["library_class_counts"][c] for c in truth]))
    rel = lab == truth[:, None]
    vals = {kind: (metrics.hit_at_k(rel, 5) if kind == "hit" else
                    metrics.precision_at_k(rel, 5)) for kind in ("hit", "precision")}
    vals["generous_vote"] = q["generous_correct_k5"].to_numpy(dtype=float)
    require(np.isfinite(np.column_stack(list(vals.values()))).all(),
            "Nonfinite query outcome")
    w = weights.astype(float, copy=False)
    denominator = w.sum(axis=1)
    require((denominator > 0).all(), "Bootstrap zero total slides")
    ci = {}
    draws = {}
    per_class = []
    for kind, v in vals.items():
        class_draws = []
        for label in classes:
            ix = truth == label
            den = w[:, ix].sum(axis=1)
            require((den > 0).all(), f"Missing class draw: {label}")
            cd = (w[:, ix] @ v[ix]) / den
            class_draws.append(cd)
            ci[f"per_class_{label}_{kind}_at_5"] = interval(cd)
            if kind == "hit":
                per_class.append({"tumor_group": label, "n_slides": int(ix.sum()),
                                  "n_resampling_units": int(pop.loc[ix, "cluster_id"].nunique()),
                                  "hit_at_5": float(v[ix].mean()),
                                  "hit_at_5_ci": interval(cd)})
            elif kind == "precision":
                next(row for row in per_class if row["tumor_group"] == label).update(
                    precision_at_5=float(v[ix].mean()),
                    precision_at_5_ci=interval(cd))
        prefix = "generous_accuracy_k5" if kind == "generous_vote" else f"{kind}_at_5"
        draws[f"micro_{prefix}"] = (w @ v) / denominator
        draws[f"macro_{prefix}"] = np.mean(class_draws, axis=0)
        ci[f"micro_{prefix}"] = interval(draws[f"micro_{prefix}"])
        ci[f"macro_{prefix}"] = interval(draws[f"macro_{prefix}"])
    require(all(np.isclose(point[key], float(np.mean(vals[kind] if key.startswith("micro")
                              else [vals[kind][truth == label].mean() for label in classes])),
                           atol=1e-12, rtol=0)
                for kind in ("hit", "precision")
                for key in (f"micro_{kind}_at_5", f"macro_{kind}_at_5")),
            "Primary fixed-data point estimate inconsistency")
    votes = {rule: external_cli._f1_block(
        truth.tolist(), q[f"pred_{rule}_k5"].tolist(), cfg["library_classes"])
        for rule in ("generous", "absolute", "closest")}
    return {"n": len(pop), "n_resampling_units": pop.cluster_id.nunique(),
            "query_class_counts": dict(Counter(truth)),
            "observed_classes": classes, "point_estimates": point,
            "per_class_primary": per_class, "voting_k5": votes,
            "macro_generous_accuracy_k5": float(np.mean([
                vals["generous_vote"][truth == c].mean() for c in classes])),
            "confidence_intervals_95": ci,
            "secondary_CI": "Not computed for k1/3/10, recall, AP, nDCG, MRR or voting F1",
            "voting_macro_f1_denominator": "Configuration library class set; not comparable "
            "across different library class sets"}, draws


def paired(pop: pd.DataFrame, a: dict, baseline: dict, ar: pd.DataFrame,
           br: pd.DataFrame, cohort: str, population: str,
           alternative_configuration: str, baseline_configuration: str) -> dict:
    require(alternative_configuration != baseline_configuration,
            "A paired alternative must differ from its baseline")
    comparison_id = (f"{cohort}_{population}_{alternative_configuration}"
                     f"_vs_{baseline_configuration}")
    keys = (*PRIMARY, "micro_generous_accuracy_k5", "macro_generous_accuracy_k5")
    differences = {key: {"difference": float(a["report"]["point_estimates"][key] -
                                              baseline["report"]["point_estimates"][key])
                         if key in PRIMARY else
                         float((a["report"]["voting_k5"]["generous"]["accuracy"] -
                                baseline["report"]["voting_k5"]["generous"]["accuracy"])
                               if key.startswith("micro") else
                               a["report"]["macro_generous_accuracy_k5"] -
                               baseline["report"]["macro_generous_accuracy_k5"]),
                         "ci_95": interval(a["draws"][key] - baseline["draws"][key])}
                   for key in keys}
    require(ar["query"].tolist() == br["query"].tolist() == pop.slide_id.tolist() and
            (ar.true_label.to_numpy() == br.true_label.to_numpy()).all(),
            f"{comparison_id}: paired truth/query mismatch")
    changes = {}
    for endpoint, field in (("hit_at_5", "hit_k5"),
                            ("generous_vote_accuracy_k5", "generous_correct_k5")):
        old = br[field].to_numpy(dtype=float) > .5
        new = ar[field].to_numpy(dtype=float) > .5
        changes[endpoint] = {"fixed": int((~old & new).sum()),
                             "broken": int((old & ~new).sum()),
                             "per_class": {c: {"n": int((pop.tumor_group == c).sum()),
                                                "fixed": int((~old & new &
                                                              pop.tumor_group.eq(c).to_numpy()).sum()),
                                                "broken": int((old & ~new &
                                                               pop.tumor_group.eq(c).to_numpy()).sum())}
                                           for c in sorted(pop.tumor_group.unique())}}
    return {"comparison_id": comparison_id, "cohort": cohort, "population": population,
            "alternative_configuration": alternative_configuration,
            "baseline_configuration": baseline_configuration, "n_paired": len(pop),
            "n_resampling_units": pop.cluster_id.nunique(),
            "baseline": "matched all-tissue whole_slide_ot",
            "primary_and_vote_differences": differences,
            "hit_and_vote_fixed_broken": changes,
            "library_label_sets_equal": a["config"]["library_classes"] ==
            baseline["config"]["library_classes"],
            "note": "Composite configuration; no across-library macro-F1 inference"}


def run() -> None:
    out = out_root()
    dest = out / DEST
    refuse_nonempty(dest)
    paths = sources(out)
    initial = {key: sha(path) for key, path in paths.items()}
    require(initial["stv_source"] == STV_SOURCE_SHA and
            initial["pathpt"] == PATHPT_SHA,
            "Approved STV or PathPT source version changed")
    ccdi = json.loads(paths["ccdi_complete"].read_text())
    require(ccdi["status"] == "complete_after_all_guards" and
            ccdi["provenance_sha256"] == R1_2_SHA and
            sha(paths["ccdi_provenance"]) == R1_2_SHA,
            "Completed CCDI provenance binding changed")
    original_binding = json.loads(paths["ccdi_provenance"].read_text())["input_sha256"]
    for here, there in (("comet_metadata", "comet_metadata"),
                        ("metrics_helper", "shared_metrics"),
                        ("external_helper", "shared_external_cli")):
        require(initial[here] == original_binding[there]["sha256"] and
                paths[here].resolve() == Path(original_binding[there]["path"]).resolve(),
                f"Frozen COMET/shared helper binding changed: {here}")
    lib = csv(paths["comet_metadata"])
    unique(lib, "Slide.Scan.File", "COMET metadata")
    validate_library_filter(paths, lib)
    stv = stv_manifest(paths)
    labels_stv = dict(zip(stv.loc[stv.processed, "slide_id"],
                          stv.loc[stv.processed, "tumor_group"]))
    kid_ranks = pd.read_csv(paths["kidrare_whole_slide_ot_ranked"],
                            usecols=["query"], dtype=str)
    kid = kid_manifest(paths, set(kid_ranks["query"]))
    labels_kid = dict(zip(kid.loc[kid.processed, "slide_id"],
                          kid.loc[kid.processed, "original_tumor_group"]))
    manifests = {"stv": stv, "kidrare": kid}
    lookup = {}
    replay_log = {}
    for cohort in ("stv", "kidrare"):
        d = manifests[cohort]
        labels = labels_stv if cohort == "stv" else labels_kid
        scored = set(d.loc[d.main_eligible if cohort == "stv" else d.legacy_eligible,
                           "slide_id"])
        for method in METHODS:
            q, r, source = replay(method, cohort, paths, lib, labels, scored, len(labels))
            lookup[cohort, method] = (q, r, source)
            replay_log[f"{cohort}_{method}"] = {
                "source_job": source["config"].get("lsf_job"),
                "source_prosper_commit": source["config"].get("prosper_commit"),
                "source_query_count": source["n_queries"],
                "independent_rank_to_query_and_point_replay": True}
            if cohort == "kidrare":
                lq, lr, ls = replay("legacy_" + method, cohort, paths, lib,
                                    {x: labels[x] for x in labels if x in scored},
                                    scored, 1227)
                require(lr.sort_index().equals(r.loc[pd.MultiIndex.from_product(
                    [lq["query"], range(1, 11)])].sort_index()) and
                    lq.equals(q.loc[lq["query"]]),
                    f"KidRare {method}: legacy and all-query records differ")
                require(source["point_estimates"] == ls["point_estimates"],
                        f"KidRare {method}: old all/legacy points differ")
                replay_log[f"kidrare_legacy_{method}"] = {
                    "source_job": ls["config"].get("lsf_job"),
                    "source_prosper_commit": ls["config"].get("prosper_commit"),
                    "source_query_count": 1227,
                    "old_1227_rows_identical_to_all_query": True}
        for arm in ("query_only", "both_sided"):
            if cohort == "stv":
                arm_labels = labels
                arm_scored = set(d.loc[d.main_eligible if arm == "query_only"
                                      else d.both_eligible, "slide_id"])
                expected_n = 776
            else:
                eligible = set(d.loc[d.legacy_eligible, "slide_id"])
                arm_scored = eligible - {ZERO_LIVER}
                if arm == "both_sided":
                    arm_scored -= set(d.loc[d.legacy_eligible &
                                             d.tumor_group.eq("NORMAL"), "slide_id"])
                arm_labels = {x: labels[x] for x in labels if x in eligible and x != ZERO_LIVER}
                expected_n = 1226
            require(len(arm_scored) == (696 if cohort == "stv" and arm == "query_only"
                                         else 688 if cohort == "stv" else
                                         1226 if arm == "query_only" else 1176),
                    f"{cohort}/{arm}: paired membership count changed")
            q, r, source = replay(arm, cohort, paths, lib, arm_labels,
                                  arm_scored, expected_n)
            require(set(q.index) == (set(labels) if cohort == "stv" else
                                     set(d.loc[d.legacy_eligible, "slide_id"]) - {ZERO_LIVER})
                    and not (set(q.index) & set(d.loc[d.pathpt_only_new_label, "slide_id"])
                             if cohort == "kidrare" else set()),
                    f"{cohort}/{arm}: unexpected filtering membership")
            lookup[cohort, arm] = (q, r, source)
            replay_log[f"{cohort}_{arm}"] = {
                "source_job": source["config"].get("lsf_job"),
                "source_prosper_commit": source["config"].get("prosper_commit"),
                "source_query_count": source["n_queries"],
                "independent_rank_to_query_and_point_replay": True}
    require(set(kid.loc[kid.legacy_eligible, "slide_id"]) -
            set(lookup["kidrare", "query_only"][0].index) == {ZERO_LIVER} and
            kid.set_index("slide_id").loc[ZERO_LIVER, "tumor_group"] == "LIVER",
            "KidRare zero-tumour LIVER exclusion changed")
    require(set(stv.loc[stv.both_eligible, "slide_id"]) ==
            set(stv.loc[stv.main_eligible, "slide_id"]) -
            set(stv.loc[stv.main_eligible & stv.tumor_group.eq("NORMAL"), "slide_id"]),
            "STV both-sided NORMAL exclusion changed")
    populations = {
        "stv": [("main_696", "main_eligible"), ("query_only_696", "main_eligible"),
                ("both_sided_688", "both_eligible")],
        "kidrare": [("amended_1581", "amended_eligible"),
                    ("legacy_1227", "legacy_eligible"),
                    ("query_only_1226", "query_only_eligible"),
                    ("both_sided_1176", "both_sided_eligible")]}
    kid["query_only_eligible"] = kid.legacy_eligible & kid.slide_id.ne(ZERO_LIVER)
    kid["both_sided_eligible"] = kid.query_only_eligible & kid.tumor_group.ne("NORMAL")
    require(kid.query_only_eligible.sum() == 1226 and
            kid.both_sided_eligible.sum() == 1176 and
            not kid.loc[kid.pathpt_only_new_label, "query_only_eligible"].any() and
            not kid.loc[kid.pathpt_only_new_label, "both_sided_eligible"].any(),
            "KidRare filtering eligibility expanded unexpectedly")
    dest.mkdir(exist_ok=True)
    stv.to_csv(dest / "stv_source_782_eligibility.csv", index=False)
    kid.to_csv(dest / "kidrare_source_2331_eligibility.csv", index=False)
    kid.loc[kid.pathpt_only_new_label].to_csv(
        dest / "kidrare_pathpt_changed_357_input_354_processed.csv", index=False)
    summaries = []
    perclass = []
    comparisons = []
    for cohort in ("stv", "kidrare"):
        d = manifests[cohort]
        for name, flag in populations[cohort]:
            methods = (("whole_slide_ot", "query_only") if name.startswith("query_only")
                       else ("whole_slide_ot", "both_sided") if name.startswith(
                           "both_sided") else tuple(METHODS))
            pop = d.loc[d[flag]].copy()
            if cohort == "kidrare":
                pop["tumor_group"] = pop["corrected_tumor_group"] if name == "amended_1581" else pop["original_tumor_group"]
                pop["cluster_id"] = pop.slide_id
            else:
                pop["cluster_id"] = pop.patient_id
            unique(pop, "slide_id", f"{cohort}/{name}")
            require(pop.groupby("cluster_id").tumor_group.nunique().eq(1).all(),
                    f"{cohort}/{name}: cross-class patient")
            if cohort == "stv":
                require(pop.cluster_id.str.startswith("ST_").all() and
                        pop.cluster_id.nunique() == (350 if name != "both_sided_688" else 346),
                        f"{name}: STV patient ID count changed")
            weights = cluster_weights(pop, "cluster_id", b=B, seed=SEED)
            entries = {}
            for method in methods:
                q, r, source = lookup[cohort, method]
                revised, ranks = derive(pop, q, r, source["config"])
                if cohort == "kidrare" and name == "legacy_1227":
                    prior = q.loc[pop.slide_id].reset_index(drop=True)
                    compare_per_query_frames(
                        prior, revised, f"KidRare legacy {method} derived")
                report, draws = score(pop, revised, source["config"], weights)
                report.update({"cohort": cohort, "population": name, "configuration": method,
                               "uncertainty_unit": "ST_ biological patient (author-confirmed)"
                               if cohort == "stv" else "nominal slide; patient IDs unavailable",
                               "bootstrap": bootstrap_description(
                                   cohort, weights, list(pop.tumor_group.unique())),
                               "source_job": source["config"].get("lsf_job"),
                               "source_config": source["config"],
                               "limitations": "Conditional fixed-library intervals; tiny patient "
                               "strata and unverified STV source overlap/RET mapping remain limited"
                               if cohort == "stv" else "Nominal slide-level uncertainty does not "
                               "establish patient independence"})
                prefix = f"{cohort}_{name}_{method}"
                (dest / f"{prefix}.json").write_text(json.dumps(report, indent=2) + "\n")
                revised.to_csv(dest / f"{prefix}_per_query.csv", index=False)
                ranks.to_csv(dest / f"{prefix}_ranked.csv.gz", index=False,
                             compression="gzip")
                summaries.append({"cohort": cohort, "population": name,
                                  "configuration": method, "n": len(pop),
                                  "n_resampling_units": pop.cluster_id.nunique(),
                                  **report["point_estimates"],
                                  "generous_vote_accuracy_k5": report["voting_k5"]["generous"]["accuracy"],
                                  "macro_generous_accuracy_k5": report["macro_generous_accuracy_k5"],
                                  **{k + "_ci_lo": report["confidence_intervals_95"][k]["lo"]
                                     for k in PRIMARY},
                                  **{k + "_ci_hi": report["confidence_intervals_95"][k]["hi"]
                                     for k in PRIMARY}})
                perclass += [{"cohort": cohort, "population": name,
                              "configuration": method, **row}
                             for row in report["per_class_primary"]]
                entries[method] = {"report": report, "draws": draws,
                                   "config": source["config"], "per_query": revised}
            for method in methods:
                if method == "whole_slide_ot":
                    continue
                comparisons.append(paired(pop, entries[method],
                                          entries["whole_slide_ot"],
                                          entries[method]["per_query"],
                                          entries["whole_slide_ot"]["per_query"],
                                          cohort, name, method, "whole_slide_ot"))
    identities = [(row["cohort"], row["population"],
                   row["alternative_configuration"]) for row in comparisons]
    require(set(identities) == EXPECTED_PAIRS and len(identities) == len(EXPECTED_PAIRS)
            and len({row["comparison_id"] for row in comparisons}) == len(identities)
            and all(row["baseline_configuration"] == "whole_slide_ot"
                    for row in comparisons),
            "Paired output identities differ from approved configurations")
    pd.DataFrame(summaries).to_csv(dest / "points_all_44_and_primary_intervals.csv", index=False)
    pd.DataFrame(perclass).to_csv(dest / "per_class_primary.csv", index=False)
    (dest / "paired_differences.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    provenance = {
        "contract": "ADR0019; post-result saved-ranking correction, no retrieval rerun",
        "uncertainty": "Within each class draw P_c unique patient clusters with replacement, "
        "include all slides of each selected patient with multiplicity; slide-weighted "
        "class ratios, variable-total micro ratio and fixed-class macro. KidRare is "
        "class-stratified nominal slide resampling, not patient-independent uncertainty.",
        "PROSPER_OUT": str(out.resolve()), "destination": str(dest.resolve()),
        "execution_job_id": os.environ.get("LSB_JOBID"),
        "execution_host": os.uname().nodename, "python": sys.version,
        "execution_environment": {k: os.environ.get(k) for k in (
            "LSB_JOBID", "LSB_QUEUE", "LSB_HOSTS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "script_git_HEAD": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "bootstrap": {"B": B, "seed": SEED, "percentiles": [2.5, 97.5]},
        "reference_libraries": {"all_tissue_and_query_only": 2778,
                                "both_sided_raw_positive_tumour": 2909,
                                "both_sided_eligible": 2603,
                                "both_sided_exclusions": "11 NORMAL + COMET slide 16738 "
                                "(RB) zero tumour; MISC and remaining NORMAL dropped"},
        "inputs": {k: {"path": str(path.resolve()), "sha256": initial[k]}
                   for k, path in paths.items()},
        "source_configurations": replay_log,
        "ccdi_reference_only": {"COMPLETE": str(paths["ccdi_complete"].resolve()),
                                "provenance_sha256": R1_2_SHA,
                                "no_recomputation": True},
        "limitations": ["STV later COMET/St. Jude extension, not institutional external",
                        "STV patient non-overlap and RET->RB provisional pending Mia",
                        "KidRare patient identifiers unknown; nominal slide uncertainty",
                        "Filtered KidRare arms do not contain newly eligible 354 queries",
                        "No cross-library voting macro-F1 comparison or causal inference"],
        "historical_fig4b": "unreconciled; not reused as revised quantitative evidence"}
    (dest / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    require(all(sha(path) == initial[k] for k, path in paths.items()),
            "One or more source inputs changed during execution; refusing COMPLETE")
    manifest = {"schema_version": 1, "artifacts": {
        p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(dest.iterdir()) if p.is_file()}}
    (dest / "output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dest / "COMPLETE.json").write_text(json.dumps({
        "status": "complete_after_all_guards", "n_reports": len(summaries),
        "n_paired": len(comparisons), "bootstrap_B": B,
        "provenance_sha256": sha(dest / "provenance.json"),
        "output_manifest_sha256": sha(dest / "output_manifest.json")}, indent=2) + "\n")
    print(f"Completed {dest}")


if __name__ == "__main__":
    run()
