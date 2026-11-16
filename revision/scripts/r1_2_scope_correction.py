#!/usr/bin/env python3
"""Re-score existing CCDI ranks under the approved R1.2 metadata restriction.

Run only with PROSPER_OUT set to this repository's prosper_revision_output.
This is a post-result, query-only resampling analysis, not a retrieval job.
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
from prosper.evaluation import external_cli, metrics, resample  # noqa: E402

POLICY = ROOT / "data/ccdi_scope_review_20260930.csv"
DEST_NAME = "r1_2_scope_correction_20260930"
K = (1, 3, 5, 10)
B = 5000
SEED = 123
CONFIGS = {
    "median_pooling": ("external_ccdi_inscope", "median_pooling"),
    "whole_slide_ot": ("external_ccdi_inscope", "ot_cosine"),
    "sampler": ("external_ccdi_inscope", "sampler"),
    "yottixel": ("external_ccdi_inscope", "yottixel"),
    "query_tumor_ot": ("st2p2_external_ccdi_tumor_only", "ot_cosine"),
    "both_tumor_ot": ("st2p2_external_ccdi_tumor_only_both", "ot_cosine"),
}
EXPECTED_FOXR2 = set("PBCHZU PBCLJU PBCLZN PBCNWG PBCSLE PBCVWB".split())
EXPECTED_COMPOUND = set("PBCIHB PBCISE PBCJIG PBCKIM PBCKYY PBCTCP PBCXXP".split())
EXPECTED_IDS = set(("PBBKSP PBBNWE PBBSUL PBBUXS PBCELV PBCESN PBCFGV "
                    "PBCHBP PBCHZU PBCIHB PBCISE PBCJIG PBCKIM PBCKYY PBCLJU "
                    "PBCLZN PBCMCW PBCNWG PBCPZG PBCRHB PBCRNW PBCRSB PBCSLE "
                    "PBCTCP PBCVWB PBCXXP PBCYBL PBCZHG PBDABS PBDAVL").split())
META_FIELDS = ["Diagnosis", "Diagnosis Anatomic Site", "Diagnosis Category",
               "Age at Diagnosis (days)", "Sex at Birth", "Race",
               "Treatment Type", "Last Known Survival Status"]
SOURCES = {
    "manifest_all": "external_ccdi/manifest_all.csv",
    "manifest_inscope": "external_ccdi/manifest_inscope.csv",
    "slide_preparation": "external_ccdi/slide_preparation.csv",
    "series_preparation": "external_ccdi/series_preparation_dicom.csv",
}
EXTERNAL = {
    "ccdi_metadata": ROOT / "../comet-testsets/data/input/CCDI_metadata_all.csv",
    "mapping_v02": ROOT / "../comet-testsets/data/input/ccdi_comet_mapping_v02.csv",
    "mapping_v03": ROOT / "../comet-testsets/data/input/ccdi_comet_mapping_v03.csv",
    "original_prediction": ROOT / "../comet-testsets/output/COG_CCDI_MCI_ALL/search_results/uni2/predicted_majority_disease_top_10.csv",
    "comet_metadata": ROOT / "../comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv",
    "original_nb_helper": ROOT / "../comet-testsets/src/vertical_pipeline/search/prediction_analysis_reasoning.py",
    "original_supplement_helper": ROOT / "../comet-testsets/src/vertical_pipeline/search/create_supplementary.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique(df: pd.DataFrame, col: str, name: str) -> None:
    require(df[col].notna().all() and df[col].ne("").all(), f"{name}: null/empty {col}")
    require(df[col].is_unique, f"{name}: duplicate {col}")


def output_root() -> Path:
    raw = os.environ.get("PROSPER_OUT")
    require(bool(raw), "PROSPER_OUT must be explicitly exported")
    target = (ROOT / "prosper_revision_output").resolve()
    require(Path(raw).resolve() == target, f"PROSPER_OUT must resolve to {target}")
    require(target.is_dir(), f"Missing output root: {target}")
    return target


def refuse_nonempty(destination: Path) -> None:
    require(not destination.exists() or
            (destination.is_dir() and not any(destination.iterdir())),
            f"Refusing nonempty or non-directory destination: {destination}")


def read_csv(path: Path) -> pd.DataFrame:
    require(path.is_file(), f"Missing input: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def checked_merge(left: pd.DataFrame, right: pd.DataFrame, left_key: str,
                  right_key: str, name: str, how: str = "left") -> pd.DataFrame:
    unique(left, left_key, f"{name} left")
    unique(right, right_key, f"{name} right")
    require(set(left[left_key]).issubset(set(right[right_key])),
            f"{name}: missing join keys: {sorted(set(left[left_key])-set(right[right_key]))[:8]}")
    out = left.merge(right, left_on=left_key, right_on=right_key,
                     how=how, validate="one_to_one", sort=False)
    require(len(out) == len(left), f"{name}: join changed row count")
    return out


def build_decisions(source: dict[str, pd.DataFrame], policy: pd.DataFrame) -> pd.DataFrame:
    m, all_m, prep, meta, series = (source[x] for x in
                                    ("manifest_inscope", "manifest_all", "slide_preparation",
                                     "ccdi_metadata", "series_preparation"))
    for name, df, col in (("inscope", m, "slide_id"), ("all", all_m, "slide_id"),
                          ("preparation", prep, "slide_id"), ("metadata", meta, "Participant ID"),
                          ("policy", policy, "slide_id"), ("series", series, "series")):
        unique(df, col, name)
    require(len(m) == 765 and m.ccdi_diagnosis.nunique() == 37,
            "Original 765/37 raw diagnosis strings changed")
    require(set(m.slide_id).issubset(set(all_m.slide_id)), "inscope IDs absent in all manifest")
    require(all_m.set_index("slide_id").loc[m.slide_id, m.columns.drop("slide_id")].reset_index(drop=True).equals(
        m[m.columns.drop("slide_id")].reset_index(drop=True)), "all/inscope manifest values differ")
    d = checked_merge(m, prep.drop(columns=[x for x in m.columns if x in prep and x != "slide_id"]),
                      "slide_id", "slide_id", "preparation")
    d = checked_merge(d, meta[["Participant ID", *META_FIELDS]], "participant",
                      "Participant ID", "source metadata")
    require((d.ccdi_diagnosis == d.Diagnosis).all() and
            (d.diagnosis_category == d["Diagnosis Category"]).all() and
            (d.slide_id == d.participant).all() and (d.suffixed == "False").all(),
            "Manifest / metadata / participant values disagree")
    require(set(d.prep_dicom) == {"FFPE", "FROZEN_OCT", "FFPE;FROZEN_OCT", "NONE"} and
            d.prep_dicom.value_counts().to_dict() ==
            {"FFPE": 546, "FROZEN_OCT": 209, "FFPE;FROZEN_OCT": 6, "NONE": 4},
            "Preparation margins differ from approved source")
    ffpe = d[d.prep_dicom == "FFPE"]
    s = series[series.participant.isin(ffpe.participant)]
    require(len(s) == 560 and s.participant.nunique() == 546 and
            s.prep.eq("FFPE").all() and s.fixative.eq("Formalin").all() and
            s.embedding.eq("Paraffin wax").all(), "FFPE series evidence changed")
    none = d[d.prep_dicom == "NONE"]
    require(none.prep_manifest.eq("Cytospin Slide").sum() == 3,
            "NONE/cytospin support changed")
    require(len(policy) == 30 and set(policy.slide_id) == EXPECTED_IDS and
            set(policy.slide_id).issubset(set(m.slide_id)), "Reviewed policy ID set changed")
    require(set(policy.loc[policy.reason == "known_foxr2", "slide_id"]) == EXPECTED_FOXR2 and
            set(policy.loc[policy.reason == "compound_support", "slide_id"]) == EXPECTED_COMPOUND and
            (policy.reason == "site_only").sum() == 17, "Policy disposition group changed")
    require(((policy.reason == "known_foxr2") ==
             (policy.disposition == "EXCLUDE_KNOWN_FOXR2")).all() and
            (policy.loc[policy.reason != "known_foxr2", "disposition"] == "UNRESOLVED").all(),
            "Policy disposition/reason conflict")
    # Flags audit completeness of an explicitly reviewed set; they never decide eligibility.
    flagged = d["Diagnosis Anatomic Site"].str.contains(
        r"^C7[012](?:[. :]|$)|\b(?:brain|central nervous system|spinal cord)\b",
        case=False, regex=True) | d.Diagnosis.str.contains("CNS|Olfactory neuroblastoma",
                                                           case=False, regex=True)
    require(set(d.loc[flagged, "slide_id"]) == set(policy.slide_id),
            "Potentially reviewable source set changed; manual policy review required")
    lookup = policy.set_index("slide_id")
    for expected, actual in (("expected_group", "tumor_group"),
                             ("expected_diagnosis", "Diagnosis"),
                             ("expected_site", "Diagnosis Anatomic Site")):
        require((lookup[expected] == d.set_index("slide_id").loc[lookup.index, actual]).all(),
                f"Policy {expected} disagrees with source")
    d = d.merge(policy, on="slide_id", how="left", validate="one_to_one")
    d["scope_status"] = d.disposition.where(d.disposition.notna() & d.disposition.ne(""), "CLEAR")
    d["scope_reason"] = d.reason.where(d.reason.notna() & d.reason.ne(""), "not_review_flagged")
    d["preparation_status"] = d.prep_dicom.map({
        "FFPE": "VERIFIED_FFPE", "FROZEN_OCT": "FROZEN_OCT",
        "FFPE;FROZEN_OCT": "MIXED", "NONE": "NONE_UNKNOWN"})
    require(d.preparation_status.notna().all(), "Unrecognized preparation")
    clear = d.scope_status.eq("CLEAR")
    not_fox = ~d.scope_status.eq("EXCLUDE_KNOWN_FOXR2")
    is_ffpe = d.prep_dicom.eq("FFPE")
    for name, mask in {
        "legacy_765": np.ones(len(d), dtype=bool),
        "original_ffpe_546": is_ffpe,
        "ffpe_scope_clear": is_ffpe & clear,
        "ffpe_add_site_only": is_ffpe & not_fox & ~d.scope_reason.eq("compound_support"),
        "ffpe_add_all_unresolved": is_ffpe & not_fox,
        "allprep_scope_clear": clear,
    }.items():
        d[name] = mask
    for name, expected in (("legacy_765", 765), ("original_ffpe_546", 546),
                           ("ffpe_scope_clear", 524), ("ffpe_add_site_only", 536),
                           ("ffpe_add_all_unresolved", 542), ("allprep_scope_clear", 735)):
        require(int(d[name].sum()) == expected, f"{name} count changed")
    main = d[d.ffpe_scope_clear]
    require(main.tumor_group.nunique() == 12 and main.tumor_group.eq("NB").sum() == 89
            and not main.tumor_group.eq("NF").any() and main.participant.is_unique,
            "Main population group/NB/NF/participant guard failed")
    for name in ("ffpe_scope_clear", "ffpe_add_site_only", "ffpe_add_all_unresolved",
                 "allprep_scope_clear"):
        require(not d.loc[d[name], "slide_id"].isin(EXPECTED_FOXR2).any(),
                f"{name} contains known FOXR2")
        require(d.loc[d[name], "participant"].is_unique, f"{name}: repeated participant")
    require(d.loc[d.Diagnosis.eq("Ewing sarcoma") &
                  d["Diagnosis Category"].str.contains("CNS", case=False), "scope_status"].eq("CLEAR").all(),
            "Category-only Ewing must remain clear")
    return d.drop(columns=["expected_group", "expected_diagnosis", "expected_site", "disposition", "reason"])


def build_all_manifest(source: dict[str, pd.DataFrame], d: pd.DataFrame) -> pd.DataFrame:
    original = source["manifest_all"]
    prep = source["slide_preparation"]
    unique(original, "slide_id", "all-query manifest")
    unique(prep, "slide_id", "all-query preparation")
    require(len(original) == 4160 and set(prep.slide_id) == set(original.slide_id),
            "All-query/preparation coverage changed")
    for col in original.columns.drop("slide_id"):
        require((original.set_index("slide_id")[col] ==
                 prep.set_index("slide_id").loc[original.slide_id, col].to_numpy()).all(),
                f"All-query/preparation source mismatch: {col}")
    population_cols = ("legacy_765", "original_ffpe_546", "ffpe_scope_clear",
                       "ffpe_add_site_only", "ffpe_add_all_unresolved", "allprep_scope_clear")
    require(len(d) == 765 and d.slide_id.is_unique and
            set(d.slide_id).issubset(set(original.slide_id)), "Decision join coverage changed")
    pcols = ["slide_id", "participant", "suffixed", "prep_manifest", "n_series",
             "prep_dicom", "population"]
    full = original.merge(prep[pcols], on="slide_id", how="left", validate="one_to_one")
    full = full.merge(d[["slide_id", "scope_status", "scope_reason",
                         "preparation_status", *population_cols]],
                      on="slide_id", how="left", validate="one_to_one", indicator=True)
    require(len(full) == 4160 and full.participant.notna().all() and
            int(full["_merge"].eq("both").sum()) == 765, "All-query annotation lost rows")
    full["in_original_inscope"] = full["_merge"].eq("both")
    outside = ~full.in_original_inscope
    full.loc[outside, "scope_status"] = "NOT_ASSESSED_OUTSIDE_ORIGINAL_INSCOPE"
    full.loc[outside, "scope_reason"] = "outside_original_inscope"
    full.loc[outside, "preparation_status"] = "NOT_ASSESSED_OUTSIDE_ORIGINAL_INSCOPE"
    for col in population_cols:
        full[col] = full[col].eq(True)
        require(not full.loc[outside, col].any(), f"Outside query included: {col}")
    require(full.loc[full.in_original_inscope, population_cols].reset_index(drop=True).equals(
        d[list(population_cols)].reset_index(drop=True)), "All-query eligibility differs")
    return full.drop(columns="_merge")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def matrix_path(serialized: str, out: Path) -> Path:
    marker = "/similarity_matrix/"
    require(serialized.count(marker) == 1, f"Unexpected matrix path: {serialized}")
    suffix = serialized.split(marker, 1)[1]
    require(suffix and Path(suffix).name == suffix, "Unsafe matrix suffix")
    actual = out / "similarity_matrix" / suffix
    require(actual.is_file(), f"Relocated matrix missing: {actual}")
    return actual


def input_paths(out: Path) -> dict[str, Path]:
    paths = {**{name: out / rel for name, rel in SOURCES.items()}, **EXTERNAL,
             "policy": POLICY, "script": Path(__file__).resolve(),
             "shared_metrics": Path(metrics.__file__).resolve(),
             "shared_external_cli": Path(external_cli.__file__).resolve(),
             "shared_resample": Path(resample.__file__).resolve()}
    for conf, (folder, method) in CONFIGS.items():
        base = out / folder
        stem = f"ccdi_uni2_{method}"
        files = {"per_query": base / f"per_query_{stem}.csv",
                 "ranked": base / f"ranked_lists_{stem}.csv.gz",
                 "json": base / f"external_{stem}.json"}
        for key, path in files.items():
            require(path.is_file(), f"Missing configuration source: {path}")
            paths[f"{conf}_{key}"] = path
        serialized = json.loads(files["json"].read_text())["config"]["matrix"]
        paths[f"{conf}_matrix"] = matrix_path(serialized, out)
    return paths


def verify_input_hashes(paths: dict[str, Path], initial: dict[str, str]) -> None:
    require(set(paths) == set(initial), "Input inventory changed during execution")
    for key, path in paths.items():
        require(sha256(path) == initial[key],
                f"Input {key} changed during execution; refusing completion")


def compare_frames(actual: pd.DataFrame, expected: pd.DataFrame, name: str) -> None:
    require(list(actual.columns) == list(expected.columns) and len(actual) == len(expected),
            f"{name}: schema/count disagreement")
    for col in actual:
        if pd.api.types.is_numeric_dtype(actual[col]) and pd.api.types.is_numeric_dtype(expected[col]):
            require(np.allclose(actual[col].to_numpy(dtype=float),
                                expected[col].to_numpy(dtype=float), rtol=0,
                                atol=1e-12, equal_nan=True), f"{name}: {col} values differ")
        else:
            require(actual[col].fillna("").astype(str).equals(
                expected[col].fillna("").astype(str)), f"{name}: {col} values differ")


def replay(config: str, out: Path, d: pd.DataFrame, lib: pd.DataFrame) -> tuple:
    folder, method = CONFIGS[config]
    base = out / folder
    stem = f"ccdi_uni2_{method}"
    files = {k: base / f"{prefix}{stem}{suffix}" for k, prefix, suffix in
             (("per_query", "per_query_", ".csv"),
              ("ranked", "ranked_lists_", ".csv.gz"),
              ("json", "external_", ".json"))}
    source = json.loads(files["json"].read_text())
    cfg = source["config"]
    require(source["n_queries"] == 765 and source["queries_with_fewer_than_depth_neighbours"] == 0
            and cfg["depth"] == 10 and cfg["k_values"] == list(K) and
            cfg["n_library"] == (2603 if config == "both_tumor_ot" else 2778) and
            cfg["library_drop_labels"] == (["MISC", "NORMAL"] if config == "both_tumor_ot" else ["MISC"]),
            f"{config}: source configuration changed")
    path = matrix_path(cfg["matrix"], out)
    matrix = pd.read_pickle(path)
    require(isinstance(matrix, pd.DataFrame), f"{config}: non-DataFrame matrix")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    unique(pd.DataFrame({"id": matrix.index}), "id", f"{config} matrix rows")
    unique(pd.DataFrame({"id": matrix.columns}), "id", f"{config} matrix columns")
    unique(lib, "Slide.Scan.File", "COMET metadata")
    labels = dict(zip(lib["Slide.Scan.File"], lib["Tumor.group"]))
    patients = dict(zip(lib["Slide.Scan.File"], lib["UID.Subject"]))
    ids = [i for i in matrix.columns if i in labels and labels[i] not in cfg["library_drop_labels"]]
    require(len(ids) == cfg["n_library"] and Counter(labels[i] for i in ids) ==
            cfg["library_class_counts"] and sorted(set(labels[i] for i in ids)) ==
            cfg["library_classes"], f"{config}: COMET library changed")
    require(set(d.slide_id).issubset(set(matrix.index)), f"{config}: missing query rows")
    source_q = pd.read_csv(files["per_query"], dtype={"query": str, "true_label": str})
    source_r = pd.read_csv(files["ranked"], dtype={"query": str, "neighbour": str})
    unique(source_q, "query", f"{config} per-query")
    require(set(source_q["query"]) == set(d.slide_id) and len(source_r) == 7650 and
            source_r.groupby("query")["rank"].agg(["count", "nunique", "min", "max"]).eq(
                [10, 10, 1, 10]).all().all() and set(source_r["query"]) == set(d.slide_id),
            f"{config}: incomplete per-query/rank coverage")
    qids = [q for q in matrix.index if q in set(d.slide_id)]
    require(len(qids) == 765 and list(source_q["query"]) == qids,
            f"{config}: source query order/coverage disagreement")
    truth = d.set_index("slide_id").loc[qids, "tumor_group"].tolist()
    nids, nlabs, ndists, short = external_cli.rank_block(
        matrix, ids, labels, patients, qids, None, 10)
    require(short == 0, f"{config}: insufficient ranked neighbours")
    actual_q = external_cli.per_query_frame(qids, truth, nids, nlabs, ndists, K, 10)
    actual_r = pd.DataFrame({"query": np.repeat(qids, 10), "rank": np.tile(np.arange(1, 11), 765),
                             "neighbour": nids.ravel(), "neighbour_label": nlabs.ravel(),
                             "distance": ndists.ravel()})
    compare_frames(actual_q, source_q, f"{config} per-query replay")
    compare_frames(actual_r, source_r, f"{config} ranked replay")
    nrel = np.array([cfg["library_class_counts"].get(t, 0) for t in truth])
    point = metrics.compute_all(nlabs, truth, k_values=K, n_relevant=nrel)
    require(set(point) == set(source["point_estimates"]) and len(point) == 44 and all(
        np.isclose(v, source["point_estimates"][k], rtol=0, atol=1e-12)
        for k, v in point.items()), f"{config}: 44 source point estimates changed")
    for k in K:
        for rule in ("generous", "absolute", "closest"):
            f = external_cli._f1_block(truth, actual_q[f"pred_{rule}_k{k}"].tolist(),
                                       cfg["library_classes"])
            require(all(np.isclose(f[key], source["voting_f1"][f"{rule}_k{k}"][key],
                                   rtol=0, atol=1e-12)
                        for key in ("macro_f1", "micro_f1", "macro_precision",
                                    "macro_recall", "accuracy")),
                    f"{config}: voting summary differs")
    return actual_q, actual_r, source, files, path


def draw_weights(pop: pd.DataFrame, b: int = B, seed: int = SEED) -> np.ndarray:
    unique(pop, "slide_id", "bootstrap queries")
    unique(pop, "participant", "bootstrap participants")
    index = resample.build_patient_index(pop, "slide_id", "tumor_group", "participant")
    order = sorted(pop.tumor_group.unique())
    rng = np.random.default_rng(seed)
    id_to_i = {q: i for i, q in enumerate(pop.slide_id)}
    counts = np.empty((b, len(pop)), dtype=np.int16)
    for i in range(b):
        ids = resample.sample_queries(index, order, rng, mode="pps_cluster_one_slide")
        require(len(ids) == len(pop), "Bootstrap lost queries")
        counts[i] = np.bincount([id_to_i[q] for q in ids], minlength=len(pop))
    return counts


def interval(draw: np.ndarray) -> dict:
    require(np.isfinite(draw).all(), "Nonfinite bootstrap estimate")
    return {"lo": float(np.percentile(draw, 2.5)), "hi": float(np.percentile(draw, 97.5)),
            "B_effective": len(draw)}


def score(pop: pd.DataFrame, q: pd.DataFrame, cfg: dict, weights: np.ndarray | None) -> tuple:
    require(list(q["query"]) == list(pop["slide_id"]) and
            (q.true_label.to_numpy() == pop.tumor_group.to_numpy()).all(),
            "Scoring query order/labels differ")
    true = pop.tumor_group.to_numpy()
    classes = sorted(set(true))
    ids = cfg["library_classes"]
    lib_counts = cfg["library_class_counts"]
    ranked = q.attrs["ranked_labels"]
    point = metrics.compute_all(ranked, true, K,
                                n_relevant=np.array([lib_counts[t] for t in true]))
    rel = (ranked == true[:, None])
    vals = {f"{kind}_at_{k}": (metrics.hit_at_k(rel, k) if kind == "hit" else
                              metrics.precision_at_k(rel, k))
            for k in K for kind in ("hit", "precision")}
    per_class = []
    for c in classes:
        ix = true == c
        row = {"tumor_group": c, "n": int(ix.sum())}
        for metric, v in vals.items():
            row[metric] = float(v[ix].mean())
        per_class.append(row)
    votes = {rule: external_cli._f1_block(true.tolist(),
              q[f"pred_{rule}_k5"].tolist(), ids)
             for rule in ("generous", "absolute", "closest")}
    ci = {}
    draw_metrics = {}
    if weights is not None:
        w = weights.astype(float, copy=False)
        denom = w.sum(axis=1)
        require((denom == len(pop)).all(), "Invalid bootstrap draw sizes")
        for kind in ("hit_at_5", "precision_at_5"):
            v = vals[kind]
            draw_metrics[f"micro_{kind}"] = w @ v / denom
            class_draws = []
            for c in classes:
                ix = true == c
                class_n = w[:, ix].sum(axis=1)
                require((class_n > 0).all(), "Stratified draw lost an observed class")
                cd = (w[:, ix] @ v[ix]) / class_n
                class_draws.append(cd)
                ci[f"per_class_{c}_{kind}"] = interval(cd)
            draw_metrics[f"macro_{kind}"] = np.mean(class_draws, axis=0)
        for rule in votes:
            pred = q[f"pred_{rule}_k5"].to_numpy()
            draw_metrics[f"{rule}_accuracy_k5"] = w @ (true == pred).astype(float) / denom
            if rule == "generous":
                correct = q["generous_correct_k5"].to_numpy(dtype=float) if (
                    "generous_correct_k5" in q) else (true == pred).astype(float)
                draw_metrics["macro_generous_accuracy_k5"] = np.mean([
                    (w[:, true == c] @ correct[true == c]) / w[:, true == c].sum(axis=1)
                    for c in classes], axis=0)
            actual_n = np.column_stack([w @ (true == c).astype(float) for c in ids])
            predicted_n = np.column_stack([w @ (pred == c).astype(float) for c in ids])
            tp = np.column_stack([w @ ((true == c) & (pred == c)).astype(float) for c in ids])
            f1 = np.divide(2 * tp, actual_n + predicted_n,
                           out=np.zeros_like(tp), where=(actual_n + predicted_n) != 0)
            draw_metrics[f"{rule}_macro_f1_k5"] = f1.mean(axis=1)
            draw_metrics[f"{rule}_micro_f1_k5"] = np.divide(
                2 * tp.sum(axis=1), (actual_n + predicted_n).sum(axis=1),
                out=np.zeros(len(w)), where=(actual_n + predicted_n).sum(axis=1) != 0)
        ci.update({key: interval(v) for key, v in draw_metrics.items()})
    return {"query_class_counts": dict(Counter(true)), "point_estimates": point,
            "per_class_hit_precision": per_class, "voting_k5": votes,
            "macro_generous_accuracy_k5": float(np.mean([
                np.mean(q.loc[true == c, "generous_correct_k5"].to_numpy(dtype=float))
                if "generous_correct_k5" in q else np.mean(
                    q.loc[true == c, "pred_generous_k5"].to_numpy() == c)
                for c in classes])),
            "confidence_intervals_95": ci,
            "secondary_endpoint_CI": "not computed; no historical intervals reused"}, draw_metrics


def paired(a: dict, baseline: dict) -> dict:
    keys = ("micro_hit_at_5", "macro_hit_at_5", "micro_precision_at_5",
            "macro_precision_at_5", "generous_accuracy_k5",
            "macro_generous_accuracy_k5")
    return {key: {"difference": float(a["point"][key] - baseline["point"][key]),
                  "ci_95": interval(a["draws"][key] - baseline["draws"][key])}
            for key in keys}


def original_nb(source: dict, dest: Path, d: pd.DataFrame) -> None:
    sys.path.insert(0, str((ROOT / "../comet-testsets/src/vertical_pipeline/search").resolve()))
    from prediction_analysis_reasoning import (age_bucket, classify_nb_subtype,
                                               extract_anatomic_region)
    original = source["original_prediction"].copy()
    unique(original, "Scan ID", "original k10 predictions")
    unique(source["ccdi_metadata"], "Participant ID", "original CCDI metadata")
    original["participant"] = original["Scan ID"].str.split("_").str[0]
    source_count = len(original)
    original = original.merge(source["ccdi_metadata"][["Participant ID", *META_FIELDS]],
                              left_on="participant", right_on="Participant ID",
                              how="left", validate="many_to_one", indicator=True)
    require(len(original) == source_count and original["_merge"].eq("left_only").sum() == 19 and
            original["Scan ID"].is_unique,
            "Original k10 prediction-to-metadata join lost/duplicated records or changed "
            "its 19 missing participant records")
    original = original.drop(columns="_merge")
    memberships = {}
    for version in ("v02", "v03"):
        mp = source[f"mapping_{version}"]
        unique(mp, "ccdi_diagnosis", version)
        mapped = original.Diagnosis.map(dict(zip(mp.ccdi_diagnosis, mp.comet_diagnosis)))
        nb = original[mapped == "NB"].copy()
        memberships[version] = set(nb["Scan ID"])
        require(len(nb) == 182 and nb.participant.nunique() == 176 and
                (nb["Majority Disease"] == "NB").sum() == 64 and
                nb.Diagnosis.str.contains("FOXR2", regex=False).sum() == 8 and
                nb.loc[nb.Diagnosis.str.contains("FOXR2", regex=False),
                       "participant"].nunique() == 7 and
                set(nb.loc[nb.Diagnosis.str.contains("FOXR2", regex=False),
                           "Scan ID"]) == EXPECTED_FOXR2 | {"PBCZBC_0", "PBCZBC_1"},
                f"Original NB182/64/FOXR2 reconciliation failed: {version}")
        nb["legacy_correct_fine_label_k10"] = nb["Majority Disease"].eq("NB")
        nb["legacy_subtype_label_not_validated"] = nb.Diagnosis.map(classify_nb_subtype)
        nb["legacy_region_source_heuristic"] = nb["Diagnosis Anatomic Site"].map(
            extract_anatomic_region)
        nb["age_bucket"] = nb["Age at Diagnosis (days)"].map(age_bucket)
        if version == "v02":
            nb.to_csv(dest / "original_nb182_membership_and_source.csv", index=False)
            for field in ("Diagnosis", "Diagnosis Anatomic Site", "legacy_subtype_label_not_validated",
                          "legacy_region_source_heuristic", "age_bucket", "Sex at Birth",
                          "Race", "Diagnosis Category", "Treatment Type",
                          "Last Known Survival Status"):
                raw_field = {"age_bucket": "Age at Diagnosis (days)",
                             "legacy_region_source_heuristic": "Diagnosis Anatomic Site",
                             "legacy_subtype_label_not_validated": "Diagnosis"}.get(field, field)
                nb["source_value_missing"] = nb[raw_field].isna() | nb[raw_field].eq("")
                tab = nb.groupby([field, "source_value_missing"],
                                 dropna=False).legacy_correct_fine_label_k10.agg(
                    n="size", legacy_correct_fine_label_k10_count="sum").reset_index()
                tab.insert(0, "factor", field)
                tab = tab.rename(columns={field: "source_value"})
                if field == "Diagnosis":
                    tables = [tab]
                else:
                    tables.append(tab)
            pd.concat(tables, ignore_index=True).to_csv(
                dest / "original_nb_subgroup_region_marginals.csv", index=False)
    require(memberships["v02"] == memberships["v03"], "Old NB membership differs between mappings")
    current = set(d.loc[d.tumor_group == "NB", "slide_id"])
    require(len(current) == 178, "Current NB count changed")
    membership = sorted(memberships["v02"] | current)
    pd.DataFrame({"scan_id": membership,
                  "old_nb182": [x in memberships["v02"] for x in membership],
                  "current_nb178": [x in current for x in membership],
                  "main_nb89": [x in set(d.loc[d.ffpe_scope_clear & d.tumor_group.eq("NB"), "slide_id"])
                                for x in membership]}).to_csv(dest / "nb_membership_reconciliation.csv", index=False)
    require(len(memberships["v02"] - current) == 9 and len(current - memberships["v02"]) == 5,
            "Old/current NB membership arithmetic changed")


def verify_ffpe546_ot(d: pd.DataFrame, q: pd.DataFrame) -> dict:
    f = q.set_index("query").loc[d.loc[d.original_ffpe_546, "slide_id"]]
    counts = {col: int(f[col].sum()) for col in (
        "hit_k1", "hit_k5", "hit_k10", "generous_correct_k5")}
    macro = float(f.groupby("true_label").hit_k5.mean().mean())
    require(len(f) == 546 and counts == {
        "hit_k1": 429, "hit_k5": 507, "hit_k10": 523,
        "generous_correct_k5": 460} and
        np.isclose(macro, 0.7091165157508199, rtol=0, atol=1e-12),
        "Archived C3 FFPE546 OT counts/macro hit@5 not reproduced")
    return {"n": len(f), "observed_classes": f.true_label.nunique(),
            "hit_counts": counts, "macro_hit_at_5": macro,
            "source": "archived C3 preparation stratification, historical only"}


def nb_descriptives(dest: Path, d: pd.DataFrame, q: pd.DataFrame) -> None:
    sys.path.insert(0, str((ROOT / "../comet-testsets/src/vertical_pipeline/search").resolve()))
    from prediction_analysis_reasoning import age_bucket, extract_anatomic_region
    nb = d[d.ffpe_scope_clear & d.tumor_group.eq("NB")].merge(
        q[["query", "hit_k5", "generous_correct_k5", "pred_closest_k5"]],
        left_on="slide_id", right_on="query", validate="one_to_one")
    require(len(nb) == 89, "Main NB descriptive joins changed")
    nb["age_bucket"] = nb["Age at Diagnosis (days)"].map(age_bucket)
    nb["source_site_region_heuristic"] = nb["Diagnosis Anatomic Site"].map(extract_anatomic_region)
    tables = []
    for col in ("Diagnosis", "Diagnosis Anatomic Site", "source_site_region_heuristic",
                "age_bucket", "Sex at Birth", "Race", "Diagnosis Category",
                "Treatment Type", "Last Known Survival Status", "prep_dicom"):
        source_col = {"age_bucket": "Age at Diagnosis (days)",
                      "source_site_region_heuristic": "Diagnosis Anatomic Site"}.get(col, col)
        nb["source_value_missing"] = nb[source_col].isna() | nb[source_col].eq("")
        g = nb.groupby([col, "source_value_missing"], dropna=False).agg(
            n=("slide_id", "size"), hit5_count=("hit_k5", "sum"),
            generous_vote5_count=("generous_correct_k5", "sum")).reset_index()
        g.insert(0, "factor", col)
        tables.append(g.rename(columns={col: "source_value"}))
    result = pd.concat(tables, ignore_index=True)
    result["hit5_rate"] = result.hit5_count / result.n
    result["generous_vote5_accuracy"] = result.generous_vote5_count / result.n
    result.to_csv(dest / "main_nb_descriptive_factors.csv", index=False)
    nb["source_diagnosis_missing"] = nb.Diagnosis.isna() | nb.Diagnosis.eq("")
    nb["pred_closest_k5_missing"] = nb.pred_closest_k5.isna() | nb.pred_closest_k5.eq("")
    nb.groupby(["Diagnosis", "source_diagnosis_missing", "pred_closest_k5",
                "pred_closest_k5_missing"], dropna=False).agg(
        n=("slide_id", "size"), hit5_count=("hit_k5", "sum"),
        generous_vote5_count=("generous_correct_k5", "sum")).reset_index().to_csv(
        dest / "main_nb_raw_diagnosis_closest_vote5_confusion.csv", index=False)


def common_class_context(d: pd.DataFrame, base_r: dict, configs: dict,
                         populations: tuple[str, ...]) -> pd.DataFrame:
    common = set(d.loc[d.ffpe_scope_clear, "tumor_group"])
    require(len(common) == 12, "Main common-class support changed")
    rows = []
    for name in populations:
        pop = d[d[name] & d.tumor_group.isin(common)]
        for conf, ranks in base_r.items():
            selected = ranks.loc[
                pd.MultiIndex.from_product([pop.slide_id, range(1, 11)])]
            point = metrics.compute_all(
                selected.neighbour_label.to_numpy().reshape(len(pop), 10),
                pop.tumor_group.tolist(), K,
                n_relevant=np.array([configs[conf]["library_class_counts"][label]
                                     for label in pop.tumor_group]))
            require(len(point) == 44, f"{name}/{conf}: incomplete common-class metrics")
            rows.append({"configuration": conf, "population": name,
                         "n_common_class": len(pop),
                         "common_classes": "|".join(sorted(common)),
                         "n_excluded_noncommon": int(d[name].sum() - len(pop)),
                         **point, "descriptive_only": True})
    return pd.DataFrame(rows)


def main() -> None:
    out = output_root()
    dest = out / DEST_NAME
    refuse_nonempty(dest)
    paths = input_paths(out)
    initial_hashes = {key: sha256(path) for key, path in paths.items()}
    source = {name: read_csv(path) for name, path in paths.items() if
              name in (*SOURCES, "ccdi_metadata", "mapping_v02", "mapping_v03",
                       "original_prediction", "comet_metadata", "policy")}
    d = build_decisions(source, source["policy"])
    all_manifest = build_all_manifest(source, d)
    lib = source["comet_metadata"]
    # Refuse at entry, then create one new versioned destination. A failed guard
    # leaves no completion marker and can only be retried after manual removal.
    dest.mkdir(exist_ok=True)
    all_manifest.to_csv(dest / "manifest_original_all_4160_scope_review_20260930.csv", index=False)
    d.to_csv(dest / "decision_table_765.csv", index=False)
    for name in ("legacy_765", "original_ffpe_546", "ffpe_scope_clear",
                 "ffpe_add_site_only", "ffpe_add_all_unresolved", "allprep_scope_clear"):
        d.loc[d[name]].to_csv(dest / f"manifest_{name}.csv", index=False)
    original_nb(source, dest, d)
    replay_prov = {}
    base_q = {}
    base_r = {}
    configs = {}
    ffpe_fidelity = None
    for conf in CONFIGS:
        q, ranked, prior, files, matrix = replay(conf, out, d, lib)
        base_q[conf] = q.set_index("query")
        base_r[conf] = ranked.set_index(["query", "rank"])
        configs[conf] = prior["config"]
        if conf == "whole_slide_ot":
            ffpe_fidelity = verify_ffpe546_ot(d, q)
        replay_prov[conf] = {
            "source_files": {key: str(p.resolve()) for key, p in files.items()},
            "original_serialized_matrix": prior["config"]["matrix"],
            "relocated_matrix": str(matrix.resolve()),
            "source_job": prior["config"].get("lsf_job"),
            "source_prosper_commit": prior["config"].get("prosper_commit"),
            "library_count": prior["config"]["n_library"],
            "library_labels": prior["config"]["library_classes"],
            "query_count": 765}
        require(all(paths[f"{conf}_{key}"].resolve() == p.resolve()
                    for key, p in files.items()) and
                paths[f"{conf}_matrix"].resolve() == matrix.resolve(),
                f"{conf}: input paths changed between preflight and replay")
        # Source ranks are replayed in full; emit filtered copies only after guards.
        for name in ("ffpe_scope_clear", "ffpe_add_site_only", "ffpe_add_all_unresolved",
                     "allprep_scope_clear"):
            ids = set(d.loc[d[name], "slide_id"])
            prefix = dest / f"{conf}_{name}"
            q.loc[q["query"].isin(ids)].to_csv(f"{prefix}_per_query.csv", index=False)
            ranked.loc[ranked["query"].isin(ids)].to_csv(
                f"{prefix}_ranked.csv.gz", index=False, compression="gzip")
    populations = ("legacy_765", "original_ffpe_546", "ffpe_scope_clear",
                   "ffpe_add_site_only", "ffpe_add_all_unresolved", "allprep_scope_clear")
    summaries = []
    perclass = []
    comparisons = []
    for name in populations:
        pop = d[d[name]].copy()
        historical_only = name in ("legacy_765", "original_ffpe_546")
        weights = None if historical_only else draw_weights(pop)
        entries = {}
        for conf in CONFIGS:
            ordered = base_q[conf].loc[pop.slide_id].reset_index()
            # Replayed rank IDs/labels have already been checked, including all
            # distances. Select by explicit query order, never inner-join.
            ranks = base_r[conf].loc[
                pd.MultiIndex.from_product([pop.slide_id, range(1, 11)])]
            ordered.attrs["ranked_labels"] = ranks.neighbour_label.to_numpy().reshape(len(pop), 10)
            report, draws = score(pop, ordered, configs[conf], weights)
            report.update({"population": name, "configuration": conf,
                           "n_participants": pop.participant.nunique(),
                           "observed_classes": sorted(pop.tumor_group.unique()),
                           "historical_context_only": historical_only,
                           "unresolved_labels_provisional": name in (
                               "ffpe_add_site_only", "ffpe_add_all_unresolved"),
                           "NF": {"observed_n": int(pop.tumor_group.eq("NF").sum()),
                                  "estimate": None if not pop.tumor_group.eq("NF").any()
                                  else next((row for row in report["per_class_hit_precision"]
                                             if row["tumor_group"] == "NF"), None)},
                           "CI_design": None if historical_only else
                           "B=5000 seed=123 class-stratified PPS patient cluster, "
                           "one slide per drawn patient; query-only, fixed reference library",
                           "voting_macro_f1_denominator": "All classes in this configuration's "
                           "library, including absent query classes with zero F1 "
                           "(sklearn zero_division=0); NF retrieval is NA if n=0"})
            (dest / f"{conf}_{name}.json").write_text(json.dumps(report, indent=2) + "\n")
            summary = {"configuration": conf, "population": name, "n": len(pop),
                       "n_classes": len(report["observed_classes"]), **report["point_estimates"]}
            for rule, v in report["voting_k5"].items():
                summary.update({f"{rule}_{k}_k5": v[k] for k in
                                ("accuracy", "macro_f1", "micro_f1")})
            summary["macro_generous_accuracy_k5"] = report["macro_generous_accuracy_k5"]
            for key, v in report["confidence_intervals_95"].items():
                if not key.startswith("per_class_"):
                    summary[key + "_ci_lo"] = v["lo"]
                    summary[key + "_ci_hi"] = v["hi"]
            summaries.append(summary)
            for row in report["per_class_hit_precision"]:
                perclass.append({"configuration": conf, "population": name, **row,
                                 **{k + "_ci_lo": report["confidence_intervals_95"].get(
                                     f"per_class_{row['tumor_group']}_{k}", {}).get("lo")
                                     for k in ("hit_at_5", "precision_at_5")},
                                 **{k + "_ci_hi": report["confidence_intervals_95"].get(
                                     f"per_class_{row['tumor_group']}_{k}", {}).get("hi")
                                     for k in ("hit_at_5", "precision_at_5")}})
            if not historical_only:
                entries[conf] = {"draws": draws, "point": {
                    **{k: report["point_estimates"][k] for k in
                       ("micro_hit_at_5", "macro_hit_at_5", "micro_precision_at_5",
                        "macro_precision_at_5")},
                    "generous_accuracy_k5": report["voting_k5"]["generous"]["accuracy"],
                    "macro_generous_accuracy_k5": report["macro_generous_accuracy_k5"]}}
            if name == "ffpe_scope_clear" and conf == "whole_slide_ot":
                nb_descriptives(dest, d, ordered)
        if not historical_only:
            baseline = entries["whole_slide_ot"]
            for conf in CONFIGS:
                if conf != "whole_slide_ot":
                    comparisons.append({"population": name, "configuration": conf,
                                        "baseline": "whole_slide_ot", "n_paired": len(pop),
                                        "difference": paired(entries[conf], baseline),
                                        "observed_truth_classes": sorted(pop.tumor_group.unique()),
                                        "library_label_sets_equal": replay_prov[conf]["library_labels"] ==
                                        replay_prov["whole_slide_ot"]["library_labels"],
                                        "note": "No macro F1 compared across distinct library label sets"})
    pd.DataFrame(summaries).to_csv(dest / "point_estimates_all_44_and_voting.csv", index=False)
    pd.DataFrame(perclass).to_csv(dest / "per_class_primary.csv", index=False)
    (dest / "paired_differences.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    common_context = common_class_context(d, base_r, configs, populations)
    common_context.to_csv(dest / "common_12_class_context.csv", index=False)
    common_context.loc[common_context.population.isin(
        ("legacy_765", "original_ffpe_546"))].rename(
            columns={"population": "historical_population"}).to_csv(
                dest / "historical_common_12_class_context.csv", index=False)
    prep_rows = []
    clear = d[d.allprep_scope_clear]
    for prep_name, group in clear.groupby("preparation_status"):
        for conf in CONFIGS:
            q = base_q[conf].loc[group.slide_id]
            prep_rows.append({"configuration": conf, "preparation": prep_name, "n": len(group),
                              "n_classes": group.tumor_group.nunique(),
                              "hit_at_5": float(q.hit_k5.mean()),
                              "generous_vote_accuracy_k5": float(q.generous_correct_k5.mean()),
                              "descriptive_only": True})
    pd.DataFrame(prep_rows).to_csv(dest / "allprep_clear_preparation_descriptive.csv", index=False)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     text=True).strip()
    provenance = {
        "restriction": "Post-result metadata restriction; not prospective independence, "
                       "pathology adjudication, preparation causality or extrapolation",
        "uncertainty": "2.5/97.5 percentiles; B=5000 seed=123 pps_cluster_one_slide "
                       "class-stratified patient resampling; fixed library, queries only",
        "paired_design": "All methods share each population's draws and query set. "
                         "Historical ST2 paired resampling lacked class stratification; "
                         "its old confidence intervals are not reused.",
        "common_class_context": "The 12 groups observed in the main FFPE scope-clear "
                                "population are held constant across all six populations; "
                                "this table is descriptive only, without between-population "
                                "paired intervals or significance claims.",
        "script_git_HEAD": commit, "execution_job_id": os.environ.get("LSB_JOBID"),
        "execution_host": os.uname().nodename, "python": sys.version,
        "execution_environment": {key: os.environ.get(key) for key in (
            "LSB_JOBID", "LSB_QUEUE", "LSB_HOSTS", "LSB_MCPU_HOSTS",
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "PROSPER_OUT": str(out.resolve()), "destination": str(dest.resolve()),
        "input_sha256": {key: {"path": str(p.resolve()), "sha256": initial_hashes[key]}
                         for key, p in paths.items()},
        "configurations": replay_prov, "populations": {
            name: {"n": int(d[name].sum()), "n_participants": d.loc[d[name], "participant"].nunique(),
                   "observed_groups": d.loc[d[name], "tumor_group"].value_counts().to_dict()}
            for name in populations},
        "archived_ffpe546_ot_fidelity": ffpe_fidelity,
        "metric_definition": "prosper.evaluation.metrics.compute_all; k=1,3,5,10, "
                             "MRR depth10; vote external_cli._f1_block; "
                             "macro retrieval averages observed query classes; voting "
                             "macro F1 averages the configuration's library classes, "
                             "including absent query labels with zero F1; absent NF retrieval "
                             "rates remain not estimable, not zero",
        "source_unchanged": "Inputs read-only; original Word/supplement and published results untouched"}
    (dest / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    verify_input_hashes(paths, initial_hashes)
    (dest / "COMPLETE.json").write_text(json.dumps({
        "status": "complete_after_all_guards", "n_configurations": len(CONFIGS),
        "n_populations": len(populations), "main_queries": 524, "bootstrap_B": B,
        "provenance_sha256": sha256(dest / "provenance.json")}, indent=2) + "\n")
    print(f"Completed {dest}")


if __name__ == "__main__":
    main()
