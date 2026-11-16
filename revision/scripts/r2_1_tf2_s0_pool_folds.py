#!/usr/bin/env python3
"""R2.1-TF2 stage S0: input bindings, pool table, fold extension, label validation,
1024-px labels (F5), Virchow2 annotation-bank identity (F8) and the planned
mask-assignment table with F7 checkpoint (i).

Binding specification: r2_1_filter_tf2_20261016/contract_candidate_003/SPECIFICATION.md
(3a444fc714c87d70...) + contract_candidate_004/CONTRACT_AMENDMENTS.md (ed3d8487e79deb50...);
release briefs implementation_release_001/BRIEF.md and implementation_release_004/BRIEF.md
(amendment P1, contract_candidate_006/CONTRACT_AMENDMENT_POOL_40X.md: optional slide exclusion
with `--exclude-slides FILE --exclusion-reason TEXT`; excluded slides stay in pool.csv with
in_pool = 0, their provenance class and `excluded_reason`; everything downstream of the pool is
computed from the remaining slides). Sections implemented: §2 (bindings),
§3.1 (pool, annotated patient), §3.2 (folds, Immune rule, class-support gate), §3.4 / R1 (F7 (i)),
§4 S0, §7 F1, F5, F8 (R9: rows with |delta| > 1e-4 fail).

Reads inputs only. Writes only into --out, which must be $PROSPER_OUT/r2_1_filter_tf2_20261016/s0
or a directory below it, and must be absent or empty. Never submits jobs, never runs a model.

Subcommands
  run                 the S0 stage (default outputs listed in OUTPUTS)
  labels-1024-supplement
                      the same 1024-px label function for slides whose trident grid only exists
                      after S1c (the 35 non-library slides); writes into a new directory.

Reused by import (not copied): prosper.evaluation.r2_1_hoptimus_filter (tile identity,
training hash, patient hash) and scripts/r2_6_titan_patch_labels_1024.py (nn_source_index,
label_windows, child_counts: the R2.6 majority-pixel rule).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import hashlib
import importlib.util
import io
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

HAORAN = os.environ.get(
    "TF2_HAORAN_ROOT",
    "/PROSPER_WORKSPACE")
PROSPER_SEARCH = os.environ.get("TF2_PROSPER_SEARCH", HAORAN + "/prosper-search")
if PROSPER_SEARCH not in sys.path:
    sys.path.insert(0, PROSPER_SEARCH)

from prosper.evaluation import r2_1_hoptimus_filter as kernel  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
RUN_NAME = "r2_1_filter_tf2_20261016"
STAGE = "s0"
CONTRACT_SHA = {
    "contract_candidate_003/CONTRACT.md":
        "3a444fc714c87d708ebc83cb0c600c3ce6db278bed0088d780079c78ec21a881",
    "contract_candidate_004/CONTRACT_AMENDMENTS.md":
        "ed3d8487e79deb50ef36e4503e83356b1da07189fa275b7c91837c44142fad1e",
    "contract_candidate_005/CONTRACT_AMENDMENT_R3c.md":
        "caf64f139d25860c743e075dc93e0b3458e726f135c40f4f6902bb901966aa41",
    "contract_candidate_006/CONTRACT_AMENDMENT_POOL_40X.md":
        "1c1193ee8944275e99eb9aafc195234330fbc4a321107181322d84d48750daeb",
}
ENCODERS = ("uni2", "virchow2", "h_optimus_1", "conch_v15")
BUDGETS = (100, 5)
FOLDS = (0, 1, 2, 3, 4)
SLOTS = ("fold0", "fold1", "fold2", "fold3", "fold4", "full")
CLASSES = (1, 2, 3, 4, 5)
IMMUNE = 2
NONLIBRARY_GROUP = "NOT_IN_LIBRARY"  # placeholder group for the 35 non-library slides (not a diagnosis)
F8_LIMIT = 1e-4
LEGACY_PATCH = 512
PATCH_1024 = 1024
PROVENANCE_CLASSES = ("library", "qc_removed", "consolidation_dropped", "listed_unlabelled",
                      "no_features")
POOL_COLUMNS = ["slide_id", "labelled_tiles", "in_pool", "in_final_library", "provenance_class",
                "UID.Subject", "patient_source", "qc_bad_slide", "diagnostic_group", "excluded_reason"]
TILE_COLUMNS = ["tile_id", "slide_id", "patient_id", "diagnostic_group", "level", "x", "y",
                "patch_size", "tissue_label", "label_csv_row", "held_out_fold", "qc_removed_slide",
                "in_final_library", "training_hash"]
LABEL_1024_COLUMNS = ["slide_id", "patient_id", "diagnostic_group", "held_out_fold", "x", "y",
                      "label_1024", "n_raster_pixels", "frac_label_pixels", "n512_available",
                      "n512_tumor", "feature_row"]
OUTPUTS = ("pool.csv", "annotation_tiles.csv.gz", "patient_folds_tf2.csv", "class_support.csv",
           "labels_1024.csv.gz", "labels_1024_report.json", "validation_512_disagreements.csv.gz",
           "virchow2_identity_check.json", "virchow2_identity_per_slide.csv",
           "planned_head_patients.csv", "mask_assignment.csv", "query_assignment.csv",
           "f7_checkpoint_i.json", "input_bindings.json")


@dataclass(frozen=True)
class Expected:
    """Specification §0 counts (gated) and record-only expectations (§3.1/§3.2 'expected ~').

    labelled / library / outside / listed_unlabelled are the pool counts AFTER any exclusion
    (P1 run: 229 / 202 / 27 / 3); `excluded` is the number of excluded slides; labelled_tiles
    is the labelled-tile total of all labelled slides with features BEFORE exclusion (an input
    property, 546,910); the post-exclusion pool tile total is recorded, not gated."""
    listed: int = 241
    labelled: int = 237
    library: int = 202
    outside: int = 35
    listed_unlabelled: int = 3
    excluded: int = 0
    no_features: int = 1
    qc_removed: int = 14
    consolidation_dropped: int = 21
    labelled_tiles: int = 546910
    union_patients: int = 1826
    queries: int = 2604
    groups: int = 14
    # record-only (specification says "expected ~ / 19-20"; not a gate)
    annotated_patients_expected: int = 223
    appended_patients_expected: tuple = (19, 20)
    share_patient_with_library_expected: int = 18


def default_inputs(haoran: str = HAORAN, prosper_out: str | None = None) -> dict:
    """§2 input roles (read-only). Paths are bound with sha256 by `bind_inputs`."""
    core = str(Path(haoran).parent)
    out = prosper_out or os.environ["PROSPER_OUT"]
    ca = haoran + "/comet-annotation"
    trident = haoran + "/comet-feature-gen/comet-titan/trident/trident_processed/40x_1024px_0px_overlap"
    return {
        # files
        "slide_list": ca + "/output/pathologist_annotated_slide_ids/slide_ids.txt",
        "provenance_ilp_845": haoran + "/comet-ilp/results/selected_slides_ilp_topk_5_best_None_slides.txt",
        "provenance_further_238": ca + "/src/create_tile_coordinates_for_Alex/further_selected_slides_ilp_topk_5_best_None_slides.txt",
        "library_map": haoran + "/comet-new-scans/data/final_slide_list/final_scan_id_disease_map_with_sample_id_patient_id.csv",
        "rcode_map": ca + "/data/COMET_02162024_final_R_code_output_includes_SurgIDs_without_PHI_rerun.csv",
        "qc_list": haoran + "/comet-new-scans/data/final_slide_list/COMET_slides_proposed_action_20250508_Ali.csv",
        "accepted_folds": out + "/r2_1_cpu_20261002/filter_attempt_001/patient_folds.csv",
        "retrieval_population": out + "/r2_1_cpu_20261002/filter_attempt_001/retrieval_population.csv",
        "routeb_population": out + "/r2_1_route_b_uni2_20261013/mosaic/attempt_001/retrieval_population.csv",
        # directories (per-slide files)
        "label_dir": ca + "/output/pathologist_annotated_labels",
        "patch_dir": ca + "/output/all_patches",
        "raster_dir": core + "/COMET/QuPathAnnotation_Tumor_NonTumor_processed_final/export",
        "ann_uni2_dir": ca + "/output/pathologist_annotated_uni2_features",
        "ann_virchow2_dir": ca + "/output/pathologist_annotated_virchow2_features",
        "ann_h_optimus_1_dir": out + "/features/h_optimus_1_all_patches",
        "lib_uni2_dir": haoran + "/comet-feature-gen/comet-uni2/data/tmp_final",
        "lib_virchow2_fp32_dir": out + "/r2_1_virchow2_fp32_20261006/run_001/features/virchow2_fp32",
        "lib_h_optimus_1_dir": out + "/features/h_optimus_1",
        "trident_root": trident,
        "wsi_library_dir": haoran + "/data/comet_final_HPC",
        "wsi_nonlibrary_dir": haoran + "/data/comet_all_HPC",
        # baseline matrices (§2 [F13], R6)
        "baseline:uni2/ot_cosine": haoran + "/comet-feature-gen/comet-uni2/output/similarity_matrix/combined_similarity_matrix_uni2_ot_cosine_no_same_patient.pkl",
        "baseline:uni2/yottixel": haoran + "/comet-feature-gen/comet-uni2/output/similarity_matrix/combined_similarity_matrix_uni2_yottixel_no_same_patient.pkl",
        "baseline:virchow2_fp32/ot_cosine": out + "/r2_1_virchow2_fp32_main_20261013/run_003/similarity_matrix/combined_similarity_matrix_virchow2_fp32_ot_cosine_no_same_patient.pkl",
        "baseline:virchow2_fp32/yottixel": out + "/r2_1_virchow2_fp32_main_20261013/run_003/similarity_matrix/combined_similarity_matrix_virchow2_fp32_yottixel_no_same_patient.pkl",
        "baseline:h_optimus_1/ot_cosine": out + "/similarity_matrix/combined_similarity_matrix_h_optimus_1_ot_cosine_no_same_patient.pkl",
        "baseline:h_optimus_1/yottixel": out + "/similarity_matrix/combined_similarity_matrix_h_optimus_1_yottixel_no_same_patient.pkl",
        "baseline:titan/titan": out + "/r2_6_titan_arm1_regen_20261004/combined_similarity_matrix_titan_titan_repaired_no_same_patient.pkl",
        "virchow2_run003_complete": out + "/r2_1_virchow2_fp32_main_20261013/run_003/COMPLETE.json",
        "h_optimus_1_main_grid_bindings": out + "/r2_1_main_grid_plurality_20261012/run_001/input_bindings.json",
        # code pins (recipe, evaluator, TITAN pins)
        "code:fit": PROSPER_SEARCH + "/prosper/evaluation/r2_1_hoptimus_fit.py",
        "code:filter_kernel": PROSPER_SEARCH + "/prosper/evaluation/r2_1_hoptimus_filter.py",
        "code:filter_eval_cli": PROSPER_SEARCH + "/prosper/evaluation/r2_1_filter_eval_cli.py",
        "code:filter_rer": str(REPO / "scripts/r2_1_filter_rer.py"),
        "code:titan_pins": str(REPO / "scripts/r2_6_titan_arm1_regen.py"),
        "code:labels_1024_rule": str(REPO / "scripts/r2_6_titan_patch_labels_1024.py"),
        # regression references (accepted outputs; read-only)
        "ref:stage1a_eligible": out + "/r2_1_annotation_audit_20261001/stage1a_attempt_001/eligible_tiles.csv",
        "ref:stage1a_quarantine": out + "/r2_1_annotation_audit_20261001/stage1a_attempt_001/quarantine_tiles.csv",
        "ref:stage1a_slide_inventory": out + "/r2_1_annotation_audit_20261001/stage1a_attempt_001/slide_inventory.csv",
        "ref:r2_6_labels_1024": out + "/r2_6_titan_direct1024_20261004/labels_1024/labels_1024.csv.gz",
    }


DIRECTORY_ROLES = ("label_dir", "patch_dir", "raster_dir", "ann_uni2_dir", "ann_virchow2_dir",
                   "ann_h_optimus_1_dir", "lib_uni2_dir", "lib_virchow2_fp32_dir",
                   "lib_h_optimus_1_dir", "trident_root", "wsi_library_dir", "wsi_nonlibrary_dir")


class GateError(RuntimeError):
    """A specification gate or input conflict: STOP."""


def require(cond: bool, message: str) -> None:
    if not cond:
        raise GateError(message)


# ---------------------------------------------------------------- small utilities
def sha256_file(path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_sha(value) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                   separators=(",", ":")).encode())


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def load_module(name: str, path: Path):
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(path):
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses with postponed annotations need the module registered
    spec.loader.exec_module(mod)
    return mod


def labels_rule_module(inputs: dict):
    return load_module("tf2_r2_6_labels_rule", Path(inputs["code:labels_1024_rule"]))


def slide_sort_key(s: str):
    return (len(s), s)


def stage_root(prosper_out: str) -> Path:
    return Path(prosper_out).resolve() / RUN_NAME


def check_out_dir(out: Path, prosper_out: str, stage: str) -> Path:
    """--out must lie under $PROSPER_OUT/<run>/<stage> and be absent or empty."""
    root = (stage_root(prosper_out) / stage).resolve()
    target = Path(out).absolute()
    resolved = target.resolve()
    require(resolved == root or root in resolved.parents,
            f"--out {target} is not under {root}")
    if target.exists():
        require(target.is_dir() and not any(target.iterdir()),
                f"refusing non-empty destination {target}")
    return target


def write_csv(df, path: Path) -> None:
    if str(path).endswith(".gz"):
        df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0},
                  float_format="%.9g", lineterminator="\n")
    else:
        df.to_csv(path, index=False, float_format="%.9g", lineterminator="\n")


def write_json(value, path: Path) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=False, default=str) + "\n")


def file_record(role: str, path: str, *, hash_it: bool = True) -> dict:
    p = Path(path)
    st = p.stat()
    rec = {"role": role, "path": str(p), "resolved_path": str(p.resolve()),
           "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if hash_it:
        rec["sha256"] = sha256_file(p)
    return rec


# ---------------------------------------------------------------- inputs: tables
def read_slide_list(path) -> list[str]:
    raw = Path(path).read_text().splitlines()
    ids = [line.strip() for line in raw if line.strip()]
    require(len(ids) == len(set(ids)), "slide list has duplicate ids")
    return ids


def read_library_map(path):
    import pandas as pd
    lib = pd.read_csv(path, dtype=str)
    for col in ("Tumor.group", "Slide.Scan.File", "UID.Subject"):
        require(col in lib.columns, f"library map lacks {col}")
    require(not lib["Slide.Scan.File"].duplicated().any(), "library map: duplicate slide")
    require((lib["Slide.Scan.File"] == lib["Slide.Scan.File"].str.strip()).all(),
            "library map: untrimmed slide ids")
    return lib


def read_rcode_map(path):
    import pandas as pd
    r = pd.read_csv(path, dtype=str, low_memory=False)
    for col in ("Slide.Scan.File", "UID.Subject.x"):
        require(col in r.columns, f"R-code output lacks {col}")
    return r


def rcode_patients(r, slides) -> dict:
    """slide -> sorted list of distinct non-null UID.Subject.x values in the R-code output."""
    sub = r[r["Slide.Scan.File"].isin(set(slides))]
    out = {s: [] for s in slides}
    for s, g in sub.groupby("Slide.Scan.File"):
        out[s] = sorted(set(g["UID.Subject.x"].dropna().tolist()))
    return out


def read_qc_bad(path) -> set:
    import pandas as pd
    qc = pd.read_csv(path, dtype=str)
    for col in ("slides", "comet_action"):
        require(col in qc.columns, f"QC list lacks {col}")
    bad = qc.loc[qc["comet_action"] == "STAGING BAD SLIDE", "slides"].dropna().astype(str)
    out = set()
    for s in bad:
        s = s.strip()
        out.add(s[:-4] if s.endswith(".svs") else s)
    return out


def read_label_pair(slide: str, label_csv: Path, patch_csv: Path):
    """Stage1a-style validation of one (label CSV, all_patches CSV) pair.

    Rules: exact headers; constant slide_id equal to the file's slide; integral nonnegative
    wsi_loc_x/y (never loc_x/y); unique coordinates; label codes 0..5; identical coordinate
    sequence in the two files. Returns (all_tiles_frame, record)."""
    import pandas as pd
    lab = pd.read_csv(label_csv)
    pat = pd.read_csv(patch_csv)
    require(list(lab.columns) == ["tissue_coverage", "slide_id", "wsi_loc_x", "wsi_loc_y",
                                  "loc_x", "loc_y", "patch_label"],
            f"{slide}: unexpected label CSV header {list(lab.columns)}")
    require(list(pat.columns) == ["tissue_coverage", "slide_id", "wsi_loc_x", "wsi_loc_y",
                                  "loc_x", "loc_y"],
            f"{slide}: unexpected all_patches header {list(pat.columns)}")
    for name, df in (("label", lab), ("patch", pat)):
        require(df["slide_id"].astype(str).eq(slide).all(), f"{slide}: {name} slide_id mismatch")
        for c in ("wsi_loc_x", "wsi_loc_y"):
            require(df[c].dtype.kind in "iu", f"{slide}: {name} {c} not integral")
            require(bool((df[c] >= 0).all()), f"{slide}: {name} {c} negative")
        require(not df.duplicated(["wsi_loc_x", "wsi_loc_y"]).any(),
                f"{slide}: {name} duplicate coordinates")
    require(lab["patch_label"].dtype.kind in "iu", f"{slide}: non-integral labels")
    require(bool(lab["patch_label"].between(0, 5).all()), f"{slide}: label outside 0..5")
    same = (len(lab) == len(pat) and
            bool((lab[["wsi_loc_x", "wsi_loc_y"]].to_numpy() ==
                  pat[["wsi_loc_x", "wsi_loc_y"]].to_numpy()).all()))
    require(same, f"{slide}: label/all_patches coordinate sequences differ")
    tiles = pd.DataFrame({"x": lab["wsi_loc_x"].to_numpy(np.int64),
                          "y": lab["wsi_loc_y"].to_numpy(np.int64),
                          "label": lab["patch_label"].to_numpy(np.int64),
                          "label_csv_row": np.arange(len(lab), dtype=np.int64)})
    counts = {str(k): int(v) for k, v in sorted(collections.Counter(tiles["label"]).items())}
    rec = {"slide_id": slide, "rows": int(len(lab)), "labelled_tiles": int((tiles["label"] > 0).sum()),
           "label_counts": counts,
           "coords_multiple_of_512": bool(((tiles["x"] % 512 == 0) & (tiles["y"] % 512 == 0)).all())}
    return tiles, rec


# ---------------------------------------------------------------- pool (§3.1, F1)
def build_pool(slide_ids, label_recs, library, rcode, qc_bad, ann_feature_slides, excluded=None):
    """241-row pool table + F1 evidence. `ann_feature_slides`: slides with both historical
    annotation-bank pickles (UNI2 and Virchow2). `excluded`: slide -> reason (amendment P1); an
    excluded slide keeps its row and provenance class, gets in_pool = 0 and `excluded_reason`.
    Every excluded slide must be a listed, labelled slide with features (else STOP)."""
    import pandas as pd
    excluded = dict(excluded or {})
    unknown = sorted(set(excluded) - set(slide_ids), key=slide_sort_key)
    require(not unknown, f"excluded slides not in the annotated slide list: {unknown}")
    not_pool = sorted((s for s in excluded if not (label_recs[s]["labelled_tiles"] > 0
                                                   and s in ann_feature_slides)), key=slide_sort_key)
    require(not not_pool, f"excluded slides that would not be in the pool anyway: {not_pool}")
    L = library.set_index("Slide.Scan.File")
    rp = rcode_patients(rcode, slide_ids)
    rows, conflicts = [], []
    for s in slide_ids:
        n = label_recs[s]["labelled_tiles"]
        in_lib = s in L.index
        has_feat = s in ann_feature_slides
        if not has_feat:
            cls = "no_features"
        elif n == 0:
            cls = "listed_unlabelled"
        elif in_lib:
            cls = "library"
        elif s in qc_bad:
            cls = "qc_removed"
        else:
            cls = "consolidation_dropped"
        if in_lib:
            patient, source, group = L.loc[s, "UID.Subject"], "library_map", L.loc[s, "Tumor.group"]
        elif len(rp[s]) == 1:
            patient, source, group = rp[s][0], "rcode_output", ""
        else:
            patient, source, group = "", "", ""
            if n > 0:
                conflicts.append({"slide_id": s, "rcode_uids": rp[s]})
        rows.append({"slide_id": s, "labelled_tiles": int(n),
                     "in_pool": int(n > 0 and has_feat and s not in excluded),
                     "in_final_library": int(in_lib), "provenance_class": cls,
                     "UID.Subject": patient, "patient_source": source,
                     "qc_bad_slide": int(s in qc_bad), "diagnostic_group": group,
                     "excluded_reason": excluded.get(s, "")})
    pool = pd.DataFrame(rows, columns=POOL_COLUMNS)
    pooled = pool[pool["in_pool"] == 1]
    lib_pool = pooled[pooled["in_final_library"] == 1]["slide_id"].tolist()
    agree = [s for s in lib_pool if rp[s] == [L.loc[s, "UID.Subject"]]]
    disagree = [{"slide_id": s, "library": L.loc[s, "UID.Subject"], "rcode": rp[s]}
                for s in lib_pool if s not in set(agree)]
    nonlib_pool = pooled[pooled["in_final_library"] == 0]["slide_id"].tolist()
    single = [s for s in nonlib_pool if len(rp[s]) == 1]
    lib_patients = set(library["UID.Subject"])
    share = [s for s in nonlib_pool if len(rp[s]) == 1 and rp[s][0] in lib_patients]
    qc_in_lib_pool = sorted(set(lib_pool) & qc_bad)
    f1 = {
        "library_vs_rcode_agreement": {"agree": len(agree), "of": len(lib_pool),
                                       "disagreements": disagree},
        "recovered_slides_single_uid": {"single": len(single), "of": len(nonlib_pool),
                                        "conflicts": conflicts},
        "nonlibrary_sharing_library_patient": len(share),
        "qc_bad_slides_in_library_pool": qc_in_lib_pool,
    }
    return pool, f1


def annotation_tiles(pool, tiles_by_slide):
    """Labelled (1..5) 512-px tiles of the pool slides with kernel tile ids and training hashes."""
    import pandas as pd
    frames = []
    for row in pool[pool["in_pool"] == 1].rename(columns={"UID.Subject": "patient_id"}).itertuples(index=False):
        t = tiles_by_slide[row.slide_id]
        t = t[t["label"] > 0]
        slide, patient = row.slide_id, row.patient_id
        group = row.diagnostic_group if row.in_final_library else NONLIBRARY_GROUP
        ids, hashes = [], []
        for x, y, lab_, r in zip(t["x"].tolist(), t["y"].tolist(), t["label"].tolist(),
                                 t["label_csv_row"].tolist()):
            rec = kernel.AnnotationRecord(
                tile_id=kernel.physical_tile_id(slide, x, y), slide_id=slide, patient_id=patient,
                diagnostic_group=group, level=0, x=x, y=y, patch_size=512, tissue_label=lab_,
                h_optimus_1_feature_row=r)
            ids.append(rec.tile_id)
            hashes.append(kernel.training_tile_hash(rec))
        frames.append(pd.DataFrame({
            "tile_id": ids, "slide_id": slide, "patient_id": patient, "diagnostic_group": group,
            "level": 0, "x": t["x"].to_numpy(), "y": t["y"].to_numpy(), "patch_size": 512,
            "tissue_label": t["label"].to_numpy(), "label_csv_row": t["label_csv_row"].to_numpy(),
            "held_out_fold": -1, "qc_removed_slide": int(row.provenance_class == "qc_removed"),
            "in_final_library": int(row.in_final_library), "training_hash": hashes}))
    tiles = pd.concat(frames, ignore_index=True)
    tiles = tiles.sort_values(["slide_id", "x", "y"], kind="mergesort").reset_index(drop=True)
    require(not tiles["tile_id"].duplicated().any(), "duplicate tile ids")
    return tiles


def read_exclusion_file(path) -> list[str]:
    """One slide id per line (blank lines ignored); duplicates or whitespace inside ids STOP."""
    ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    require(ids, f"exclusion file {path} lists no slides")
    require(len(ids) == len(set(ids)), f"exclusion file {path} has duplicate ids")
    require(all(len(s.split()) == 1 for s in ids), f"exclusion file {path}: malformed id")
    return ids


def compare_support(before, after):
    """(held_out_fold, tissue_label) rows whose training support differs before/after exclusion."""
    key = ["held_out_fold", "tissue_label"]
    a = before.assign(held_out_fold=before["held_out_fold"].astype(str)).set_index(key)
    b = after.assign(held_out_fold=after["held_out_fold"].astype(str)).set_index(key)
    rows = []
    for k in a.index:
        ra, rb = a.loc[k], b.loc[k]
        if any(int(ra[c]) != int(rb[c]) for c in ("training_patients", "training_slides", "training_tiles")):
            rows.append({"held_out_fold": k[0], "tissue_label": int(k[1]),
                         **{f"{c}_before": int(ra[c]) for c in ("training_patients", "training_slides", "training_tiles")},
                         **{f"{c}_after": int(rb[c]) for c in ("training_patients", "training_slides", "training_tiles")},
                         "patients_changed": int(ra["training_patients"]) != int(rb["training_patients"])})
    return rows


# ---------------------------------------------------------------- folds (§3.2)
def read_accepted_folds(path):
    import pandas as pd
    pf = pd.read_csv(path, dtype=str)
    require(list(pf.columns) == ["patient_id", "fold", "kernel_order", "patient_hash",
                                 "has_annotations", "has_retrieval"], "accepted folds header")
    require(not pf["patient_id"].duplicated().any(), "accepted folds: duplicate patient")
    require(bool(pf["fold"].isin([str(f) for f in FOLDS]).all()), "accepted folds: invalid fold")
    bad_hash = int((pf["patient_id"].map(kernel.patient_fold_hash) != pf["patient_hash"]).sum())
    require(bad_hash == 0, f"accepted folds: {bad_hash} patient hashes do not reproduce")
    return {p: int(f) for p, f in zip(pf["patient_id"], pf["fold"])}


def extend_folds(accepted: dict, annotated: set, immune: set):
    """§3.2: keep the accepted assignment; append annotated patients outside the union in
    UID.Subject order; each goes to the fold with the fewest annotated patients at that moment,
    except an Immune-carrying patient goes to the fold with the fewest Immune-carrying annotated
    patients; ties -> lowest fold index. Counts include appended patients as they are placed."""
    fold = dict(accepted)
    ann_count = collections.Counter({f: 0 for f in FOLDS})
    imm_count = collections.Counter({f: 0 for f in FOLDS})
    for p in annotated:
        if p in accepted:
            ann_count[accepted[p]] += 1
            if p in immune:
                imm_count[accepted[p]] += 1
    start = {"annotated_per_fold": dict(sorted(ann_count.items())),
             "immune_per_fold": dict(sorted(imm_count.items()))}
    log = []
    for position, p in enumerate(sorted(x for x in annotated if x not in accepted)):
        if p in immune:
            f = min(FOLDS, key=lambda k: (imm_count[k], k))
            rule = "immune_fewest_immune_annotated"
            imm_count[f] += 1
        else:
            f = min(FOLDS, key=lambda k: (ann_count[k], k))
            rule = "fewest_annotated"
        ann_count[f] += 1
        fold[p] = f
        log.append({"append_position": position, "patient_id": p, "fold": f, "rule": rule})
    end = {"annotated_per_fold": dict(sorted(ann_count.items())),
           "immune_per_fold": dict(sorted(imm_count.items()))}
    return fold, log, start, end


def folds_table(fold, accepted, annotated, immune, log):
    import pandas as pd
    pos = {r["patient_id"]: r for r in log}
    rows = []
    for p in sorted(fold):
        rows.append({"patient_id": p, "fold": fold[p],
                     "fold_source": "accepted_r2_1_cpu_20261002" if p in accepted else "appended_tf2_s3_2",
                     "accepted_fold": accepted.get(p, ""),
                     "annotated_tf2": int(p in annotated), "immune_tf2": int(p in immune),
                     "append_position": pos[p]["append_position"] if p in pos else "",
                     "append_rule": pos[p]["rule"] if p in pos else ""})
    return pd.DataFrame(rows)


def class_support(tiles, fold):
    """Training-set support per held-out fold (complement) and for the full pool."""
    import pandas as pd
    t = tiles.assign(fold=tiles["patient_id"].map(fold))
    rows = []
    for held in list(FOLDS) + ["full"]:
        train = t if held == "full" else t[t["fold"] != held]
        for c in CLASSES:
            sub = train[train["tissue_label"] == c]
            rows.append({"held_out_fold": held, "tissue_label": c,
                         "training_patients": int(sub["patient_id"].nunique()),
                         "training_slides": int(sub["slide_id"].nunique()),
                         "training_tiles": int(len(sub))})
    df = pd.DataFrame(rows)
    gate = bool((df["training_patients"] >= 1).all())
    return df, gate


# ---------------------------------------------------------------- 1024-px labels / F5
def label_slide_task(task):
    """One slide: F5 validation at P = 512 on every legacy tile coordinate, and 1024-px labels on
    the trident grid when that grid exists. Uses the R2.6 rule functions by import."""
    import pandas as pd
    import tifffile
    (slide, rule_path, raster_path, patches_h5, conch_h5, wsi_path, label_csv) = task
    rule = load_module("tf2_r2_6_labels_rule_worker", Path(rule_path))
    t0 = time.time()
    rec = {"slide_id": slide}
    raster = tifffile.memmap(raster_path, mode="r")
    require(raster.ndim == 2 and raster.dtype == np.uint8,
            f"{slide}: raster shape/dtype {raster.shape} {raster.dtype}")
    pix = np.zeros(256, np.int64)
    for r0 in range(0, raster.shape[0], 1024):
        pix += np.bincount(np.asarray(raster[r0:r0 + 1024]).ravel(), minlength=256)
    vals = np.flatnonzero(pix)
    require(int(vals.max()) <= 5, f"{slide}: raster values outside 0..5")
    rec["raster_values"] = [int(v) for v in vals]
    rec["raster_shape_hw"] = [int(raster.shape[0]), int(raster.shape[1])]
    import openslide
    with openslide.OpenSlide(wsi_path) as osl:
        os_dims = tuple(int(v) for v in osl.dimensions)
    rec["openslide_dims"] = list(os_dims)
    out = None
    if patches_h5 is not None:
        import h5py
        with h5py.File(patches_h5, "r") as h:
            coords = h["coords"][:]
            attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in h["coords"].attrs.items()}
        with h5py.File(conch_h5, "r") as h:
            fcoords = h["coords"][:]
            fshape = list(h["features"].shape)
        require(np.array_equal(coords, fcoords), f"{slide}: patch/feature coords differ")
        require(fshape == [coords.shape[0], 768], f"{slide}: CONCH feature shape {fshape}")
        require(int(attrs["patch_size"]) == PATCH_1024 and int(attrs["patch_size_level0"]) == PATCH_1024
                and int(attrs["overlap"]) == 0, f"{slide}: unexpected patch attrs {attrs}")
        W, H = int(attrs["level0_width"]), int(attrs["level0_height"])
        require((W, H) == os_dims, f"{slide}: openslide dims {os_dims} != trident {(W, H)}")
        rec["dims_source"] = "trident_attrs_eq_openslide"
    else:
        W, H = os_dims
        rec["dims_source"] = "openslide"
    rec["level0_width"], rec["level0_height"] = W, H
    leg = pd.read_csv(label_csv)
    lc = leg[["wsi_loc_x", "wsi_loc_y"]].to_numpy(np.int64)
    llab, _, lfrac, _, lclip = rule.label_windows(raster, W, H, lc, LEGACY_PATCH)
    legacy_label = leg["patch_label"].to_numpy(np.int64)
    agree = llab == legacy_label
    rec.update(n_legacy=int(len(leg)), n_legacy_agree=int(agree.sum()),
               n_legacy_clipped=int(lclip.sum()))
    dis = pd.DataFrame({"slide_id": slide, "x": lc[~agree, 0], "y": lc[~agree, 1],
                        "legacy_label": legacy_label[~agree], "recomputed_label": llab[~agree],
                        "recomputed_frac": lfrac[~agree]})
    if patches_h5 is not None:
        lab, npx, frac, _, clip = rule.label_windows(raster, W, H, coords, PATCH_1024)
        require(not (lab < 0).any(), f"{slide}: empty 1024 window")
        legacy_map = {(int(a), int(b)): int(v) for (a, b), v in zip(lc.tolist(), legacy_label)}
        n_av, n_tu = rule.child_counts(coords, legacy_map, PATCH_1024)
        rec["n_patches_1024"] = int(coords.shape[0])
        rec["n_clipped_1024"] = int(clip.sum())
        out = {"x": coords[:, 0].astype(np.int64), "y": coords[:, 1].astype(np.int64),
               "label_1024": lab, "n_raster_pixels": npx, "frac_label_pixels": frac,
               "n512_available": n_av, "n512_tumor": n_tu,
               "feature_row": np.arange(coords.shape[0], dtype=np.int64)}
    rec["seconds"] = round(time.time() - t0, 2)
    return rec, out, dis


def run_labels(slides, meta, inputs, trident_root, wsi_dirs, workers, log=print):
    """Return (labels_1024 frame, per-slide records, disagreements frame)."""
    import pandas as pd
    tasks = []
    roots = [trident_root] if isinstance(trident_root, (str, Path)) else list(trident_root)
    for s in slides:
        hits = [(Path(r) / "patches" / f"{s}_patches.h5", Path(r) / "features_conch_v15" / f"{s}.h5")
                for r in roots]
        hits = [(p, f) for p, f in hits if p.exists() and f.exists()]
        require(len(hits) <= 1, f"{s}: trident grid present in more than one root")
        has = bool(hits)
        p, f = hits[0] if hits else (None, None)
        wsi = Path(wsi_dirs[meta[s]["wsi_dir_role"]]) / f"{s}.svs"
        tasks.append((s, inputs["code:labels_1024_rule"], str(Path(inputs["raster_dir"]) / f"{s}.tif"),
                      str(p) if has else None, str(f) if has else None, str(wsi),
                      str(Path(inputs["label_dir"]) / f"{s}.csv")))
    recs, frames, dis = [], [], []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for k, (rec, out, d) in enumerate(ex.map(label_slide_task, tasks)):
            s = rec["slide_id"]
            recs.append(rec)
            if len(d):
                dis.append(d)
            if out is not None:
                df = pd.DataFrame(out)
                df.insert(0, "slide_id", s)
                df.insert(1, "patient_id", meta[s]["patient_id"])
                df.insert(2, "diagnostic_group", meta[s]["diagnostic_group"])
                df.insert(3, "held_out_fold", int(meta[s]["fold"]))
                frames.append(df[LABEL_1024_COLUMNS])
            if (k + 1) % 20 == 0:
                log(f"[labels] {k + 1}/{len(tasks)} slides")
    labels = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=LABEL_1024_COLUMNS))
    disdf = (pd.concat(dis, ignore_index=True) if dis else
             pd.DataFrame(columns=["slide_id", "x", "y", "legacy_label", "recomputed_label",
                                   "recomputed_frac"]))
    return labels, recs, disdf


def f5_report(recs, slides_expected):
    n = sum(r["n_legacy"] for r in recs)
    a = sum(r["n_legacy_agree"] for r in recs)
    covered = sorted({r["slide_id"] for r in recs}, key=slide_sort_key)
    return {"n_slides": len(recs), "n_tiles": n, "n_agree": a,
            "agreement": (a / n) if n else None,
            "slides_with_disagreement": sorted([r["slide_id"] for r in recs
                                                if r["n_legacy_agree"] != r["n_legacy"]],
                                               key=slide_sort_key),
            "all_expected_slides_validated": covered == sorted(slides_expected, key=slide_sort_key),
            "pass": bool(n and a == n and covered == sorted(slides_expected, key=slide_sort_key))}


# ---------------------------------------------------------------- F8 (Virchow2 identity)
def virchow2_slide_task(task):
    import pickle
    slide, ann_path, lib_path, limit = task
    with open(ann_path, "rb") as fh:
        A, at = pickle.load(fh)
    with open(lib_path, "rb") as fh:
        V, vt = pickle.load(fh)
    require(isinstance(A, np.ndarray) and A.ndim == 2 and A.shape[1] >= 1280,
            f"{slide}: annotation Virchow2 matrix")
    require(isinstance(V, np.ndarray) and V.ndim == 2 and V.shape[1] == 1280 and
            V.dtype == np.float32, f"{slide}: fp32 library matrix")
    idx = {}
    for i, d in enumerate(at):
        key = tuple(int(v) for v in d["wsi_loc"])
        require(key not in idx, f"{slide}: duplicate annotation coordinate")
        idx[key] = i
    pairs = [(i, idx[tuple(int(v) for v in d["wsi_loc"])]) for i, d in enumerate(vt)
             if tuple(int(v) for v in d["wsi_loc"]) in idx]
    rec = {"slide_id": slide, "library_rows": int(V.shape[0]), "annotation_rows": int(A.shape[0]),
           "annotation_dtype": str(A.dtype), "coincident": len(pairs)}
    exceptions = []
    if pairs:
        li = np.array([p[0] for p in pairs]); ai = np.array([p[1] for p in pairs])
        X = V[li].astype(np.float64)
        Y = A[ai, :1280].astype(np.float64)
        d = np.abs(X - Y)
        rowmax = d.max(axis=1)
        bitwise = (V[li] == A[ai, :1280].astype(np.float32)).all(axis=1) & (A[ai, :1280].dtype == np.float32)
        rel = rowmax / np.maximum(np.abs(X).max(axis=1), np.finfo(np.float64).tiny)
        rec.update(bitwise_rows=int(bitwise.sum()), max_abs=float(rowmax.max()),
                   max_rel=float(rel.max()), rows_over_limit=int((rowmax > limit).sum()))
        for k in np.flatnonzero(~bitwise):
            x, y = vt[int(li[k])]["wsi_loc"]
            exceptions.append({"slide_id": slide, "x": int(x), "y": int(y),
                               "max_abs": float(rowmax[k]), "max_rel": float(rel[k]),
                               "over_limit": bool(rowmax[k] > limit)})
    else:
        rec.update(bitwise_rows=0, max_abs=None, max_rel=None, rows_over_limit=0)
    return rec, exceptions


def run_f8(slides, inputs, workers, limit=F8_LIMIT):
    tasks = []
    absent = []
    for s in slides:
        a = Path(inputs["ann_virchow2_dir"]) / f"{s}.pkl"
        v = Path(inputs["lib_virchow2_fp32_dir"]) / f"{s}.pkl"
        if a.exists() and v.exists():
            tasks.append((s, str(a), str(v), limit))
        else:
            absent.append({"slide_id": s, "annotation_bank": a.exists(), "fp32_library_bank": v.exists()})
    recs, exc = [], []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for rec, e in ex.map(virchow2_slide_task, tasks):
            recs.append(rec)
            exc.extend(e)
    coinc = sum(r["coincident"] for r in recs)
    bitwise = sum(r["bitwise_rows"] for r in recs)
    over = sum(r["rows_over_limit"] for r in recs)
    maxabs = max([r["max_abs"] for r in recs if r["max_abs"] is not None], default=None)
    summary = {
        "rule": "first 1,280 columns of the historical annotation bank vs the fp32 library bank on "
                "all coincident tiles (same wsi_loc); a row with max |delta| > 1e-4 FAILS (R9)",
        "limit_abs": limit, "slides_compared": len(recs), "slides_not_compared": absent,
        "coincident_rows": coinc, "bitwise_rows": bitwise, "non_bitwise_rows": coinc - bitwise,
        "rows_over_limit": over, "max_abs": maxabs,
        "pass": bool(recs) and over == 0,
        "exceptions": exc,
    }
    return summary, recs


# ---------------------------------------------------------------- mask assignment / F7 (i)
def head_id(encoder: str, budget: int, slot: str) -> str:
    return f"{encoder}/b{budget}/{slot}"


def planned_head_patients(annotated: set, fold: dict) -> dict:
    """Planned training-patient set per head slot: fold f -> annotated patients outside fold f;
    full -> all annotated patients. Identical for every encoder and budget (upper bound: a
    budget/resolution can only drop patients, which F7 (ii) re-checks on the saved sets)."""
    plan = {f"fold{f}": sorted(p for p in annotated if fold[p] != f) for f in FOLDS}
    plan["full"] = sorted(annotated)
    return plan


def mask_tables(population, annotated: set, fold: dict, planned: dict):
    """Per-query assignment (Option C) and the stratum/version -> head table, with F7 (i)."""
    import pandas as pd
    q = population.copy()
    q["patient_annotated"] = q["patient_id"].isin(annotated).astype(int)
    q["patient_fold"] = q["patient_id"].map(fold)
    require(not q["patient_fold"].isna().any(), "query patient without fold")
    q["patient_fold"] = q["patient_fold"].astype(int)
    q["stratum"] = np.where(q["patient_annotated"] == 1, "fold" + q["patient_fold"].astype(str),
                            "unannotated")
    q["head_slot"] = np.where(q["patient_annotated"] == 1, "fold" + q["patient_fold"].astype(str),
                              "full")
    q["library_version_both_sided"] = "v_" + q["head_slot"]
    q["library_version_query_only"] = "unfiltered"
    planned_sets = {k: set(v) for k, v in planned.items()}
    q["f7_i_query_patient_in_planned_training"] = [
        int(p in planned_sets[h]) for p, h in zip(q["patient_id"], q["head_slot"])]
    qcols = ["query_index", "slide_id", "patient_id", "diagnostic_group", "patient_annotated",
             "patient_fold", "stratum", "head_slot", "library_version_both_sided",
             "library_version_query_only", "f7_i_query_patient_in_planned_training"]
    q = q[qcols]
    rows = []
    for enc in ENCODERS:
        for b in BUDGETS:
            for slot in SLOTS:
                stratum = "unannotated" if slot == "full" else slot
                members = q[q["stratum"] == stratum]
                leak = int(members["f7_i_query_patient_in_planned_training"].sum())
                pats = planned[slot]
                rows.append({"encoder": enc, "budget_percent": b, "entry_kind": "query_stratum",
                             "entry_id": stratum, "head_slot": slot, "head_id": head_id(enc, b, slot),
                             "both_sided_library_version": "v_" + slot,
                             "n_queries": int(len(members)),
                             "planned_training_patients": len(pats),
                             "planned_training_patients_sha256": json_sha(pats),
                             "f7_i": "pass" if leak == 0 else "fail",
                             "f7_ii": "pending_s2"})
                rows.append({"encoder": enc, "budget_percent": b, "entry_kind": "library_version",
                             "entry_id": "v_" + slot, "head_slot": slot,
                             "head_id": head_id(enc, b, slot),
                             "both_sided_library_version": "v_" + slot,
                             "n_queries": int(len(members)),
                             "planned_training_patients": len(pats),
                             "planned_training_patients_sha256": json_sha(pats),
                             "f7_i": "pass", "f7_ii": "pending_s2"})
    table = pd.DataFrame(rows)
    return q, table


def f7_checkpoint_i(q, table, planned, fold):
    """R1 (i): every (query stratum -> head) names a head whose planned training-patient set
    excludes that stratum's patients; every library version's head equals its query fold's head."""
    violations = []
    leak = q[q["f7_i_query_patient_in_planned_training"] == 1]
    for r in leak.itertuples(index=False):
        violations.append({"kind": "query_patient_in_planned_training", "query_index": int(r.query_index),
                           "slide_id": r.slide_id, "patient_id": r.patient_id, "head_slot": r.head_slot})
    for slot in SLOTS:
        if slot == "full":
            continue
        f = int(slot[4:])
        bad = [p for p in planned[slot] if fold[p] == f]
        if bad:
            violations.append({"kind": "planned_fold_head_contains_fold_patients", "head_slot": slot,
                               "patients": bad[:20]})
    for (enc, b), g in table.groupby(["encoder", "budget_percent"]):
        qs = g[g["entry_kind"] == "query_stratum"].set_index("head_slot")
        vs = g[g["entry_kind"] == "library_version"].set_index("entry_id")
        for slot in SLOTS:
            v = "v_" + slot
            if qs.loc[slot, "both_sided_library_version"] != v or vs.loc[v, "head_id"] != qs.loc[slot, "head_id"]:
                violations.append({"kind": "library_version_head_mismatch", "encoder": enc,
                                   "budget_percent": int(b), "slot": slot})
    mism = q[q["library_version_both_sided"] != "v_" + q["head_slot"]]
    for r in mism.itertuples(index=False):
        violations.append({"kind": "query_version_mismatch", "query_index": int(r.query_index)})
    return {"checkpoint": "F7 (i) at S0 on the planned mask-assignment table (R1)",
            "queries_checked": int(len(q)), "strata": q["stratum"].value_counts().sort_index().to_dict(),
            "head_slots": list(SLOTS), "violations": violations, "pass": not violations}


# ---------------------------------------------------------------- bindings
def bind_inputs(inputs: dict, per_slide: dict, workers: int, log=print):
    """sha256 of every bound file. `per_slide`: role -> list of files actually relevant under a
    directory role. WSIs are bound by path/size only (header read for level-0 dimensions)."""
    file_roles = [r for r in inputs if r not in DIRECTORY_ROLES]
    jobs = [(r, inputs[r]) for r in file_roles]
    for role, paths in per_slide.items():
        if role.startswith("wsi_"):
            continue
        jobs.extend((role, str(p)) for p in paths)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        files = list(ex.map(lambda j: file_record(j[0], j[1]), jobs))
    log(f"[bind] hashed {len(files)} files in {time.time() - t0:.1f}s")
    wsis = []
    for role, paths in per_slide.items():
        if role.startswith("wsi_"):
            for p in paths:
                rec = file_record(role, str(p), hash_it=False)
                rec["reason"] = "WSI read for level-0 dimensions only (openslide header); not hashed"
                wsis.append(rec)
    dirs = []
    for role in DIRECTORY_ROLES:
        sub = sorted((f for f in files if f["role"] == role), key=lambda f: f["path"])
        dirs.append({"role": role, "path": inputs[role], "n_files_bound": len(sub),
                     "aggregate_sha256": json_sha([[Path(f["path"]).name, f["sha256"]] for f in sub])})
    return {"files": files, "directories": dirs, "unhashed_wsi": wsis}


def recheck_bindings(bindings) -> list:
    """Cheap end-of-run recheck (size + mtime) of every bound file; full rehash is S7."""
    changed = []
    for f in bindings["files"] + bindings["unhashed_wsi"]:
        st = Path(f["path"]).stat()
        if st.st_size != f["size"] or st.st_mtime_ns != f["mtime_ns"]:
            changed.append(f["path"])
    return changed


# ---------------------------------------------------------------- the S0 stage
def code_record():
    paths = {"s0_script": Path(__file__).resolve(), "filter_kernel": Path(kernel.__file__).resolve()}
    return {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in paths.items()}


def environment():
    import importlib.metadata as md
    vers = {}
    for n in ("numpy", "pandas", "scipy", "scikit-learn", "h5py", "tifffile", "openslide-python"):
        try:
            vers[n] = md.version(n)
        except md.PackageNotFoundError:
            vers[n] = None
    return {"python": sys.version, "platform": platform.platform(), "host": platform.node(),
            "versions": vers, "lsf_job": os.environ.get("LSB_JOBID")}


def run_s0(out: Path, inputs: dict, expected: Expected, *, workers: int = 6,
           hash_workers: int = 8, exclusion: dict | None = None, log=print) -> dict:
    """`exclusion`: None or {"path": file, "reason": text} (amendment P1)."""
    import pandas as pd
    t_start = time.time()
    inputs = dict(inputs)
    out.mkdir(parents=True, exist_ok=True)
    gates, records = {}, {}

    def gate(name, ok, detail):
        gates[name] = {"pass": bool(ok), "detail": detail}
        log(f"[gate] {name}: {'PASS' if ok else 'FAIL'}")

    try:
        # ---- slide list and labels
        slide_ids = read_slide_list(inputs["slide_list"])
        label_recs, tiles_by_slide = {}, {}
        for s in slide_ids:
            t, rec = read_label_pair(s, Path(inputs["label_dir"]) / f"{s}.csv",
                                     Path(inputs["patch_dir"]) / f"{s}.csv")
            label_recs[s], tiles_by_slide[s] = rec, t
        log(f"[s0] {len(slide_ids)} listed slides validated")
        library = read_library_map(inputs["library_map"])
        rcode = read_rcode_map(inputs["rcode_map"])
        qc_bad = read_qc_bad(inputs["qc_list"])
        ann = {s for s in slide_ids
               if (Path(inputs["ann_uni2_dir"]) / f"{s}.pkl").exists()
               and (Path(inputs["ann_virchow2_dir"]) / f"{s}.pkl").exists()}
        excluded = {}
        if exclusion is not None:
            require(str(exclusion.get("reason") or "").strip(), "exclusion requires a reason")
            ex_ids = read_exclusion_file(exclusion["path"])
            excluded = {s: str(exclusion["reason"]).strip() for s in ex_ids}
            inputs["exclusion_list"] = str(exclusion["path"])
        pool, f1 = build_pool(slide_ids, label_recs, library, rcode, qc_bad, ann, excluded)
        pooled = pool[pool["in_pool"] == 1]
        pool_pre = pool.copy()  # the pool before exclusion (input-integrity gates, change report)
        pool_pre.loc[pool_pre["excluded_reason"] != "", "in_pool"] = 1
        cls = pool["provenance_class"].value_counts().to_dict()
        counts = {
            "listed": int(len(pool)), "labelled": int(len(pooled)),
            "library": int((pooled["in_final_library"] == 1).sum()),
            "outside": int((pooled["in_final_library"] == 0).sum()),
            "listed_unlabelled": int(cls.get("listed_unlabelled", 0)),
            "excluded": int((pool["excluded_reason"] != "").sum()),
            "no_features": int(cls.get("no_features", 0)),
            "qc_removed": int(cls.get("qc_removed", 0)),
            "consolidation_dropped": int(cls.get("consolidation_dropped", 0)),
            "labelled_tiles": int(pool_pre.loc[pool_pre["in_pool"] == 1, "labelled_tiles"].sum()),
        }
        want = {k: getattr(expected, k) for k in counts}
        gate(f"counts_{expected.labelled}_{expected.library}_{expected.outside}_{expected.listed_unlabelled}",
             counts == want,
             {"observed": counts, "expected": want,
              "note": "labelled/library/outside/listed_unlabelled are post-exclusion pool counts; "
                      "labelled_tiles is the pre-exclusion total over all labelled slides with features",
              "pool_labelled_tiles_after_exclusion": int(pooled["labelled_tiles"].sum())})
        ex_rows = pool[pool["excluded_reason"] != ""]
        records["exclusion"] = {
            "applied": bool(excluded),
            "file": (file_record("exclusion_list", exclusion["path"]) if exclusion is not None else None),
            "reason": (str(exclusion["reason"]).strip() if exclusion is not None else None),
            "slides": [{"slide_id": r.slide_id, "provenance_class": r.provenance_class,
                        "in_final_library": int(r.in_final_library), "patient_id": r._5,
                        "labelled_tiles": int(r.labelled_tiles), "qc_bad_slide": int(r.qc_bad_slide)}
                       for r in ex_rows.itertuples(index=False)],
            "excluded_labelled_tiles": int(ex_rows["labelled_tiles"].sum()),
            "pool_labelled_tiles_after_exclusion": int(pooled["labelled_tiles"].sum()),
        }
        f1_ok = (f1["library_vs_rcode_agreement"]["agree"] == f1["library_vs_rcode_agreement"]["of"]
                 == expected.library
                 and f1["recovered_slides_single_uid"]["single"] == f1["recovered_slides_single_uid"]["of"]
                 == expected.outside
                 and not f1["qc_bad_slides_in_library_pool"])
        # ---- tiles and annotated patients (pool after exclusion); tiles_all = before exclusion
        tiles_all = annotation_tiles(pool_pre, tiles_by_slide)
        tiles = tiles_all[~tiles_all["slide_id"].isin(set(excluded))].reset_index(drop=True)
        annotated = set(tiles["patient_id"])
        immune = set(tiles.loc[tiles["tissue_label"] == IMMUNE, "patient_id"])
        annotated_pre = set(tiles_all["patient_id"])
        immune_pre = set(tiles_all.loc[tiles_all["tissue_label"] == IMMUNE, "patient_id"])
        records["annotated_patients"] = len(annotated)
        records["immune_patients"] = sorted(immune)
        leaving = sorted(annotated_pre - annotated)
        records["exclusion"]["annotated_patients_before_exclusion"] = len(annotated_pre)
        records["exclusion"]["patients_leaving_annotated_set"] = [
            {"patient_id": p,
             "excluded_slides": sorted(set(ex_rows.loc[ex_rows["UID.Subject"] == p, "slide_id"]), key=slide_sort_key),
             "carried_immune": p in immune_pre}
            for p in leaving]
        records["exclusion"]["patients_with_excluded_slides_remaining_annotated"] = sorted(
            set(ex_rows["UID.Subject"]) & annotated)
        records["exclusion"]["immune_patients_lost"] = sorted(immune_pre - immune)
        # ---- Stage1a regression (202 eligible + 35 quarantined), on the pre-exclusion tile set
        el = pd.read_csv(inputs["ref:stage1a_eligible"], dtype={"slide_id": str, "patient_id": str},
                         usecols=["tile_id", "slide_id", "patient_id", "tissue_label", "h_optimus_1_feature_row"])
        qu = pd.read_csv(inputs["ref:stage1a_quarantine"], dtype={"slide_id": str},
                         usecols=["tile_id", "slide_id", "tissue_label"])
        lib_t = tiles_all[tiles_all["in_final_library"] == 1]
        non_t = tiles_all[tiles_all["in_final_library"] == 0]
        m1 = lib_t.merge(el, on="tile_id", how="outer", suffixes=("", "_ref"), indicator=True)
        m2 = non_t.merge(qu, on="tile_id", how="outer", suffixes=("", "_ref"), indicator=True)
        reg = {
            "eligible_rows": int(len(el)), "library_pool_rows": int(len(lib_t)),
            "eligible_tile_set_equal": bool((m1["_merge"] == "both").all()),
            "eligible_label_patient_row_equal": bool(
                (m1["_merge"] == "both").all() and (m1["tissue_label"] == m1["tissue_label_ref"]).all()
                and (m1["patient_id"] == m1["patient_id_ref"]).all()
                and (m1["label_csv_row"] == m1["h_optimus_1_feature_row"]).all()),
            "quarantine_rows": int(len(qu)), "nonlibrary_pool_rows": int(len(non_t)),
            "quarantine_tile_set_equal": bool((m2["_merge"] == "both").all()),
            "quarantine_label_equal": bool((m2["_merge"] == "both").all()
                                           and (m2["tissue_label"] == m2["tissue_label_ref"]).all()),
        }
        reg["evaluated_on"] = "all labelled slides with features before exclusion (input integrity)"
        reg_ok = all(v for k, v in reg.items() if isinstance(v, bool))
        gate("stage1a_regression", reg_ok, reg)
        # ---- folds (§3.2)
        accepted = read_accepted_folds(inputs["accepted_folds"])
        fold, applog, start, end = extend_folds(accepted, annotated, immune)
        unchanged = all(fold[p] == f for p, f in accepted.items())
        f1["fold_extension_changes_no_existing_patient"] = unchanged
        f1["accepted_union_patients"] = len(accepted)
        f1_ok = f1_ok and unchanged and len(accepted) == expected.union_patients
        gate("F1_pool_patient_map", f1_ok, f1)
        records["appended_patients"] = applog
        records["fold_balance_start"], records["fold_balance_end"] = start, end
        records["appended_immune"] = [r for r in applog if r["rule"].startswith("immune")]
        records["record_only_expectations"] = {
            "annotated_patients": {"observed": len(annotated), "contract_expected": expected.annotated_patients_expected},
            "appended_patients": {"observed": len(applog), "contract_expected": list(expected.appended_patients_expected)},
            "nonlibrary_sharing_library_patient": {"observed": f1["nonlibrary_sharing_library_patient"],
                                                   "contract_expected": expected.share_patient_with_library_expected},
        }
        tiles["held_out_fold"] = tiles["patient_id"].map(fold).astype(int)
        support, support_ok = class_support(tiles, fold)
        gate("class_support", support_ok,
             {"min_training_patients_per_class": support.groupby("tissue_label")["training_patients"].min().to_dict()})
        # change report vs the pool before exclusion (same fold rule applied to the larger set)
        fold_pre, applog_pre, _, end_pre = extend_folds(accepted, annotated_pre, immune_pre)
        support_pre, _ = class_support(tiles_all, fold_pre)
        imm = lambda fo, ann_, imm_: {f: int(sum(1 for p in imm_ if fo[p] == f)) for f in FOLDS}
        records["exclusion"]["fold_changes_vs_before_exclusion"] = [
            {"patient_id": p, "fold_before": int(fold_pre[p]), "fold_after": int(fold[p])}
            for p in sorted(annotated) if fold_pre.get(p) != fold.get(p)]
        records["exclusion"]["appended_before_exclusion"] = [r["patient_id"] for r in applog_pre]
        records["exclusion"]["appended_after_exclusion"] = [r["patient_id"] for r in applog]
        records["exclusion"]["annotated_per_fold"] = {"before": end_pre["annotated_per_fold"],
                                                     "after": end["annotated_per_fold"]}
        records["exclusion"]["immune_carriers_per_fold"] = {"before": imm(fold_pre, annotated_pre, immune_pre),
                                                           "after": imm(fold, annotated, immune)}
        records["exclusion"]["class_support_changes"] = compare_support(support_pre, support)
        records["exclusion"]["class_support_patient_count_changes"] = [
            r for r in records["exclusion"]["class_support_changes"] if r["patients_changed"]]
        # ---- retrieval population
        pop_bytes = Path(inputs["retrieval_population"]).read_bytes()
        rb_bytes = Path(inputs["routeb_population"]).read_bytes()
        population = pd.read_csv(io.BytesIO(pop_bytes), dtype={"slide_id": str, "patient_id": str})
        pop_ok = (pop_bytes == rb_bytes and len(population) == expected.queries
                  and population["diagnostic_group"].nunique() == expected.groups
                  and bool((population["patient_id"].map(fold) == population["fold"]).all()))
        gate("retrieval_population_identity", pop_ok,
             {"sha256": sha256_bytes(pop_bytes), "routeb_sha256": sha256_bytes(rb_bytes),
              "rows": int(len(population)), "groups": int(population["diagnostic_group"].nunique()),
              "fold_column_equals_tf2_fold": bool((population["patient_id"].map(fold) == population["fold"]).all())})
        # ---- mask assignment, F7 (i)
        planned = planned_head_patients(annotated, fold)
        q, table = mask_tables(population, annotated, fold, planned)
        f7 = f7_checkpoint_i(q, table, planned, fold)
        gate("F7_i", f7["pass"], {"violations": len(f7["violations"]), "strata": f7["strata"]})
        # ---- write the pool / fold / tile tables now (cheap)
        write_csv(pool, out / "pool.csv")
        write_csv(tiles[TILE_COLUMNS], out / "annotation_tiles.csv.gz")
        write_csv(folds_table(fold, accepted, annotated, immune, applog), out / "patient_folds_tf2.csv")
        write_csv(support, out / "class_support.csv")
        write_csv(pd.DataFrame([{"head_slot": k, "patient_id": p} for k, v in planned.items() for p in v]),
                  out / "planned_head_patients.csv")
        write_csv(table, out / "mask_assignment.csv")
        write_csv(q, out / "query_assignment.csv")
        write_json(f7, out / "f7_checkpoint_i.json")
        # ---- labels 1024 / F5 on the pool slides (after exclusion)
        meta = {}
        for r in pooled.rename(columns={"UID.Subject": "patient_id"}).itertuples(index=False):
            meta[r.slide_id] = {"patient_id": r.patient_id,
                                "diagnostic_group": r.diagnostic_group if r.in_final_library else NONLIBRARY_GROUP,
                                "fold": fold[r.patient_id],
                                "wsi_dir_role": "wsi_library_dir" if r.in_final_library else "wsi_nonlibrary_dir"}
        pool_slides = sorted(meta, key=slide_sort_key)
        labels, lrecs, disdf = run_labels(pool_slides, meta, inputs, inputs["trident_root"],
                                          {k: inputs[k] for k in ("wsi_library_dir", "wsi_nonlibrary_dir")},
                                          workers, log=log)
        f5 = f5_report(lrecs, pool_slides)
        gate("F5_rule_reproduces_legacy_512", f5["pass"], {k: v for k, v in f5.items() if k != "pass"})
        with_grid = sorted(labels["slide_id"].unique().tolist(), key=slide_sort_key)
        pending = [s for s in pool_slides if s not in set(with_grid)]
        ref = pd.read_csv(inputs["ref:r2_6_labels_1024"], dtype={"slide_id": str, "patient_id": str})
        mine = labels[labels["slide_id"].isin(set(ref["slide_id"]))].reset_index(drop=True)
        ref = ref[ref["slide_id"].isin(set(mine["slide_id"]))]
        key = ["slide_id", "x", "y"]
        a = mine.sort_values(key, kind="mergesort").reset_index(drop=True)
        b = ref.sort_values(key, kind="mergesort").reset_index(drop=True)
        same_cols = [c for c in LABEL_1024_COLUMNS if c != "frac_label_pixels"]
        r26 = {"slides_compared": int(a["slide_id"].nunique()), "rows": int(len(a)), "ref_rows": int(len(b)),
               "identical_except_float_format": bool(len(a) == len(b) and
                                                     (a[same_cols].astype(str).values == b[same_cols].astype(str).values).all()),
               "frac_label_pixels_max_abs_diff": float(np.abs(a["frac_label_pixels"].to_numpy(float) -
                                                              b["frac_label_pixels"].to_numpy(float)).max()) if len(a) == len(b) and len(a) else None}
        lab15 = labels[labels["label_1024"].between(1, 5)]
        lreport = {
            "rule": "R2.6 majority-pixel rule (scripts/r2_6_titan_patch_labels_1024.py, imported)",
            "validation_512": f5, "per_slide": lrecs,
            "slides_with_trident_grid": len(with_grid),
            "slides_pending_s1c_grid": pending,
            "coverage_status": ("complete" if not pending else
                                "partial: the trident 1024-px grid for these pool slides does not exist "
                                "until S1c; their labels come from `labels-1024-supplement` on the S1c "
                                "trident output (same function)"),
            "counts_overall": {str(k): int(v) for k, v in labels["label_1024"].value_counts().sort_index().items()},
            "labelled_1to5_patches": int(len(lab15)),
            "patients_per_class": {str(k): int(g["patient_id"].nunique()) for k, g in lab15.groupby("label_1024")},
            "regression_vs_r2_6_accepted_labels": r26,
        }
        write_csv(labels, out / "labels_1024.csv.gz")
        write_json(lreport, out / "labels_1024_report.json")
        write_csv(disdf, out / "validation_512_disagreements.csv.gz")
        gate("r2_6_label_regression", r26["identical_except_float_format"]
             and (r26["frac_label_pixels_max_abs_diff"] or 0) <= 1e-6, r26)
        records["labels_1024_coverage"] = {"with_grid": len(with_grid), "pending_s1c": pending}
        # ---- F8
        f8_slides = [s for s in slide_ids if s in ann and s not in excluded]
        f8, f8recs = run_f8(f8_slides, inputs, workers)
        write_json(f8, out / "virchow2_identity_check.json")
        write_csv(pd.DataFrame(f8recs), out / "virchow2_identity_per_slide.csv")
        gate("F8_virchow2_identity", f8["pass"],
             {k: f8[k] for k in ("slides_compared", "coincident_rows", "bitwise_rows", "rows_over_limit", "max_abs")})
        # ---- bindings (+ baseline matrices)
        per_slide = {
            "label_dir": [Path(inputs["label_dir"]) / f"{s}.csv" for s in slide_ids],
            "patch_dir": [Path(inputs["patch_dir"]) / f"{s}.csv" for s in slide_ids],
            "raster_dir": [Path(inputs["raster_dir"]) / f"{s}.tif" for s in slide_ids
                           if (Path(inputs["raster_dir"]) / f"{s}.tif").exists()],
            "ann_uni2_dir": [Path(inputs["ann_uni2_dir"]) / f"{s}.pkl" for s in sorted(ann)],
            "ann_virchow2_dir": [Path(inputs["ann_virchow2_dir"]) / f"{s}.pkl" for s in sorted(ann)],
            "ann_h_optimus_1_dir": [p for p in (Path(inputs["ann_h_optimus_1_dir"]) / f"{s}.pkl" for s in slide_ids) if p.exists()],
            "lib_uni2_dir": [Path(inputs["lib_uni2_dir"]) / f"{s}.pkl" for s in library["Slide.Scan.File"]],
            "lib_virchow2_fp32_dir": [Path(inputs["lib_virchow2_fp32_dir"]) / f"{s}.pkl" for s in library["Slide.Scan.File"]],
            "lib_h_optimus_1_dir": [Path(inputs["lib_h_optimus_1_dir"]) / f"{s}.pkl" for s in library["Slide.Scan.File"]],
            "trident_root": ([Path(inputs["trident_root"]) / "features_conch_v15" / f"{s}.h5" for s in library["Slide.Scan.File"]]
                             + [Path(inputs["trident_root"]) / "patches" / f"{s}_patches.h5" for s in library["Slide.Scan.File"]]),
            "wsi_library_dir": [Path(inputs["wsi_library_dir"]) / f"{s}.svs" for s in pool_slides if meta[s]["wsi_dir_role"] == "wsi_library_dir"],
            "wsi_nonlibrary_dir": [Path(inputs["wsi_nonlibrary_dir"]) / f"{s}.svs" for s in pool_slides if meta[s]["wsi_dir_role"] == "wsi_nonlibrary_dir"],
        }
        missing = {r: [str(p) for p in ps if not Path(p).exists()] for r, ps in per_slide.items()}
        missing = {r: v for r, v in missing.items() if v}
        gate("bound_files_present", not missing, {"missing": {r: v[:10] for r, v in missing.items()},
                                                  "missing_counts": {r: len(v) for r, v in missing.items()}})
        per_slide = {r: [p for p in ps if Path(p).exists()] for r, ps in per_slide.items()}
        bindings = bind_inputs(inputs, per_slide, hash_workers, log=log)
        complete_v2 = json.loads(Path(inputs["virchow2_run003_complete"]).read_text())
        mg = json.loads(Path(inputs["h_optimus_1_main_grid_bindings"]).read_text())["bindings"]
        bound = {f["role"]: f for f in bindings["files"]}
        ho_ok = all(bound[f"baseline:h_optimus_1/{m}"]["sha256"] == mg[f"matrix:h_optimus_1/{m}"]["sha256"]
                    and Path(mg[f"matrix:h_optimus_1/{m}"]["path"]).resolve() ==
                    Path(inputs[f"baseline:h_optimus_1/{m}"]).resolve() for m in ("ot_cosine", "yottixel"))
        v2_failed = (Path(inputs["virchow2_run003_complete"]).parent / "FAILED").exists() or \
            (Path(inputs["virchow2_run003_complete"]).parent / "FAILED.json").exists()
        uni2_ok = all(bound[f"baseline:uni2/{m}"]["sha256"] == mg[f"matrix:uni2/{m}"]["sha256"]
                      for m in ("ot_cosine", "yottixel"))
        v2_ok = (complete_v2.get("status") == "COMPLETE" and all(
            bound[f"baseline:virchow2_fp32/{m}"]["sha256"] ==
            complete_v2["matrices"].get(f"combined_similarity_matrix_virchow2_fp32_{m}_no_same_patient.pkl")
            for m in ("ot_cosine", "yottixel")))
        gate("baseline_matrix_bindings", ho_ok and uni2_ok and v2_ok and not v2_failed,
             {"h_optimus_1_matches_main_grid_bindings": ho_ok,
              "uni2_matches_main_grid_bindings": uni2_ok,
              "virchow2_matches_run_003_complete_record": v2_ok,
              "virchow2_run_003_has_failed_marker": v2_failed,
              "virchow2_run_003_complete_keys": sorted(complete_v2)[:20],
              "matrices": {r: bound[r]["sha256"] for r in bound if r.startswith("baseline:")}})
        prov_list = Path(inputs["provenance_further_238"]).read_text().split()
        records["provenance_record_only"] = {
            "ilp_845_lines": len(Path(inputs["provenance_ilp_845"]).read_text().split()),
            "further_238_lines": len(prov_list),
            "listed_minus_further_238": sorted(set(slide_ids) - {x.split(".")[0] for x in prov_list}, key=slide_sort_key),
        }
        changed = recheck_bindings(bindings)
        gate("inputs_unchanged_during_run", not changed, {"changed": changed[:20]})
        write_json({"schema_version": 1, "run": f"{RUN_NAME}/{STAGE}", "created": now(),
                    "contract": CONTRACT_SHA,
                    "exclusion": {k: records["exclusion"][k] for k in ("applied", "file", "reason")},
                    **bindings}, out / "input_bindings.json")
    except Exception as exc:  # STOP: record the failure with all gates so far
        marker = {"schema_version": 1, "run": f"{RUN_NAME}/{STAGE}", "status": "FAILED",
                  "created": now(), "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc(), "gates": gates, "records": records,
                  "argv": sys.argv, "code": code_record(), "environment": environment(),
                  "seconds": round(time.time() - t_start, 1)}
        write_json(marker, out / "FAILED.json")
        raise
    produced = {p.name: {"sha256": sha256_file(p), "size": p.stat().st_size}
                for p in sorted(out.iterdir()) if p.is_file()}
    ok = all(g["pass"] for g in gates.values())
    marker = {"schema_version": 1, "run": f"{RUN_NAME}/{STAGE}",
              "status": "COMPLETE" if ok else "FAILED", "created": now(),
              "contract": CONTRACT_SHA, "gates": gates, "records": records, "outputs": produced,
              "argv": sys.argv, "code": code_record(), "environment": environment(),
              "seconds": round(time.time() - t_start, 1),
              "limitations": [
                  "labels_1024.csv.gz covers pool slides with an existing trident grid; slides listed in "
                  "records.labels_1024_coverage.pending_s1c need the S1c grid (labels-1024-supplement)",
                  "WSIs are bound by path/size only (header read); full sha256 of every other bound file",
                  "end-of-run input recheck is size+mtime; full re-verification is S7",
                  "pickle inputs are trusted local files",
                  "excluded slides (records.exclusion) stay in pool.csv with in_pool = 0; the Stage1a "
                  "regression and the labelled_tiles count gate are evaluated before exclusion; every "
                  "other output is computed from the pool after exclusion"]}
    write_json(marker, out / ("COMPLETE.json" if ok else "FAILED.json"))
    return marker


# ---------------------------------------------------------------- supplement (after S1c)
def run_supplement(s0_dir: Path, trident_root, out: Path, inputs: dict, slides=None,
                   workers: int = 6, log=print) -> dict:
    """1024-px labels with the S0 function for pool slides whose grid exists only after S1c."""
    import pandas as pd
    t0 = time.time()
    complete = json.loads((s0_dir / "COMPLETE.json").read_text())
    require(complete["status"] == "COMPLETE", "S0 is not COMPLETE")
    for name, entry in complete["outputs"].items():
        require(sha256_file(s0_dir / name) == entry["sha256"], f"S0 output {name} changed")
    pool = pd.read_csv(s0_dir / "pool.csv", dtype={"slide_id": str, "UID.Subject": str})
    folds = pd.read_csv(s0_dir / "patient_folds_tf2.csv", dtype={"patient_id": str})
    fold = dict(zip(folds["patient_id"], folds["fold"].astype(int)))
    pending = complete["records"]["labels_1024_coverage"]["pending_s1c"]
    pooled = pool[pool["in_pool"] == 1].set_index("slide_id")
    nonlib = sorted(pooled.index[pooled["in_final_library"] == 0].tolist(), key=slide_sort_key)
    require(sorted(pending, key=slide_sort_key) == nonlib,
            "S0 pending slides differ from the S0 pool's non-library labelled slides")
    slides = list(slides or nonlib)
    require(set(slides) <= set(nonlib), "supplement slides must be S0 pool non-library labelled slides")
    meta = {s: {"patient_id": pooled.loc[s, "UID.Subject"],
                "diagnostic_group": (pooled.loc[s, "diagnostic_group"] if pooled.loc[s, "in_final_library"]
                                     else NONLIBRARY_GROUP),
                "fold": fold[pooled.loc[s, "UID.Subject"]],
                "wsi_dir_role": "wsi_library_dir" if pooled.loc[s, "in_final_library"] else "wsi_nonlibrary_dir"}
            for s in slides}
    out.mkdir(parents=True, exist_ok=True)
    roots = [str(trident_root)] if isinstance(trident_root, (str, Path)) else [str(r) for r in trident_root]
    labels, recs, dis = run_labels(slides, meta, inputs, roots,
                                   {k: inputs[k] for k in ("wsi_library_dir", "wsi_nonlibrary_dir")},
                                   workers, log=log)
    covered = set(labels["slide_id"])
    f5 = f5_report(recs, slides)
    write_csv(labels, out / "labels_1024_supplement.csv.gz")
    write_csv(dis, out / "validation_512_disagreements.csv.gz")
    per_slide_counts = []
    for s in sorted(slides, key=slide_sort_key):
        g = labels[labels["slide_id"] == s]
        per_slide_counts.append({"slide_id": s, "tiles_1024": int(len(g)),
                                 "labelled_1to5": int(g["label_1024"].between(1, 5).sum()),
                                 **{f"label_{c}": int((g["label_1024"] == c).sum()) for c in (0,) + CLASSES}})
    report = {"trident_roots": roots, "slides": slides, "validation_512": f5,
              "s0_excluded_slides_not_read": [r["slide_id"] for r in
                                              complete["records"].get("exclusion", {}).get("slides", [])],
              "per_slide_counts": per_slide_counts,
              "totals": {"slides": len(slides), "tiles_1024": int(len(labels)),
                         "labelled_1to5": int(labels["label_1024"].between(1, 5).sum())},
              "per_slide": recs, "slides_without_grid": [s for s in slides if s not in covered]}
    write_json(report, out / "labels_1024_supplement_report.json")
    ok = f5["pass"] and not report["slides_without_grid"]
    marker = {"status": "COMPLETE" if ok else "FAILED", "created": now(), "s0": str(s0_dir),
              "outputs": {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()},
              "code": code_record(), "seconds": round(time.time() - t0, 1)}
    write_json(marker, out / ("COMPLETE.json" if ok else "FAILED.json"))
    return marker


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=6, help="processes for labels/F8 (<= 6 on a shared node)")
    r.add_argument("--hash-workers", type=int, default=8)
    r.add_argument("--exclude-slides", help="file with one slide id per line to exclude from the pool (P1)")
    r.add_argument("--exclusion-reason", help="reason written to pool.csv excluded_reason (required with --exclude-slides)")
    r.add_argument("--expected-counts", default="237,202,35,3",
                   help="gated post-exclusion pool counts: labelled,library,nonlibrary,listed_unlabelled")
    s = sub.add_parser("labels-1024-supplement")
    s.add_argument("--s0", required=True)
    s.add_argument("--trident-root", required=True, nargs="+",
                   help="S1c .../40x_1024px_0px_overlap directories (one per chunk; each slide in exactly one)")
    s.add_argument("--out", required=True)
    s.add_argument("--stage", default="s1", help="stage directory the output must lie under")
    s.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    prosper_out = os.environ.get("PROSPER_OUT")
    if not prosper_out:
        sys.exit("PROSPER_OUT must be exported")
    if a.workers > 6:
        sys.exit("--workers must be <= 6")
    inputs = default_inputs(HAORAN, prosper_out)
    if a.cmd == "run":
        if bool(a.exclude_slides) != bool(a.exclusion_reason and a.exclusion_reason.strip()):
            sys.exit("--exclude-slides and --exclusion-reason must be given together")
        try:
            lab_, lib_, non_, unl_ = (int(v) for v in a.expected_counts.split(","))
        except ValueError:
            sys.exit("--expected-counts must be four integers: labelled,library,nonlibrary,listed_unlabelled")
        require(lab_ == lib_ + non_, "--expected-counts: labelled != library + nonlibrary")
        exclusion = None
        n_ex = 0
        if a.exclude_slides:
            exclusion = {"path": str(Path(a.exclude_slides).absolute()), "reason": a.exclusion_reason}
            n_ex = len(read_exclusion_file(exclusion["path"]))
        expected = Expected(labelled=lab_, library=lib_, outside=non_, listed_unlabelled=unl_, excluded=n_ex)
        out = check_out_dir(Path(a.out), prosper_out, STAGE)
        marker = run_s0(out, inputs, expected, workers=a.workers, hash_workers=a.hash_workers,
                        exclusion=exclusion)
    else:
        out = check_out_dir(Path(a.out), prosper_out, a.stage)
        marker = run_supplement(Path(a.s0), [Path(r) for r in a.trident_root], out, inputs, workers=a.workers)
    print(json.dumps({"status": marker["status"],
                      "gates": {k: v["pass"] for k, v in marker.get("gates", {}).items()}}))
    return 0 if marker["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
