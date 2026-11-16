"""Full original-mosaic masks only; no fitting, retrieval or new canonical claim.

Route-(b) encoder adapter (A-2) of the frozen H-Optimus-1 mosaic driver. The diff
is restricted to: the --bank selection of Stage0 rows (with the in-job M-1
binding of every decoded pickle), a head source that is a HEAD_REGISTRY.json
written by r2_1_encoder_head_registry (replacing the H-Optimus-1-specific
head-reader acceptance chain), and the feature width read from input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import importlib
import io
import os
from pathlib import Path
import socket
import sys

import numpy as np
import prosper
import prosper.evaluation

from . import feature_bank_audit as legacy
from . import r2_1_hoptimus_asset_io as assets
from . import r2_1_hoptimus_filter as kernel
from . import r2_1_hoptimus_filter_cli as manifests
from . import r2_1_hoptimus_fit as fitter
from . import r2_1_hoptimus_head_cli as producer
from . import r2_1_hoptimus_io as cpu
from . import r2_1_encoder_head_registry as registry


TASK = "hoptimus_mosaic_predictions"
SCOPE = "matched_original_mosaic_masks_only"
MIB = 1024 * 1024
ARRAY_LIMIT = 2 * 1024**3
OUTPUT_RECORD_LIMIT = 8192 + 256
TUPLES = tuple((fold, budget) for fold in range(5) for budget in (5, 100))
BANKS = ("h_optimus_1", "uni2")
# The registry (HEAD_REGISTRY.json) replaces head_acceptance/pilot_acceptance; UNI2 mode
# additionally binds the independently written G-5 accepted-heads JSON.
SOURCES = {
    "stage0_review", "stage0_complete", "stage0_inventory", "stage0_summary",
    "cpu_review", "cpu_complete", "cpu_input_bindings", "retrieval_population",
    "patient_folds", "cpu_summary", "head_plan", "head_registry",
}
BANK_SOURCES = {"h_optimus_1": frozenset(), "uni2": frozenset({"accepted_heads"})}
# Gate M-1: the UNI2 mosaic pickles are exactly CFG/comet-uni2/data/tmp_final/<slide>.pkl.
M1_FEATURE_DIRS = {
    "uni2": "/PROSPER_WORKSPACE/"
            "comet-feature-gen/comet-uni2/data/tmp_final",
}
HEAD_ASSETS = ("complete", "spec", "input_bindings", "summary", "diagnostics", "model")
REGISTRY_NAMES = ("head.pkl", "fit_diagnostics.json", "input_bindings.json", "summary.json")


def sources_for(bank):
    require(bank in BANKS, "unsupported bank")
    return SOURCES | BANK_SOURCES[bank]
EXPECTED = {
    "slides", "patients", "diagnostic_groups", "union_patients", "feature_files",
    "original_tiles", "heads", "slide_head_cells", "tile_head_predictions",
}
INVENTORY_HEADER = (
    "bank", "slide_id", "canonical_tiles", "feature_rows", "dtype",
    "feature_path", "resolved_path", "size", "sha256",
)
OFFSET_HEADER = (
    "slide_index", "slide_id", "patient_id", "diagnostic_group", "slide_fold",
    "tile_start", "tile_stop", "feature_rows", "feature_sha256",
)
COUNT_HEADER = (
    "held_out_fold", "budget_percent", "slide_index", "slide_id", "patient_id",
    "slide_fold", "query_fold_eligible", "original_tiles",
    "predicted_class_1", "predicted_class_2", "predicted_class_3",
    "predicted_class_4", "predicted_class_5", "retained_tiles", "discarded_tiles", "status",
)
LIMITATIONS = (
    "accepted_stage0_ordered_coordinate_verification_reused_with_unchanged_feature_bytes",
    "no_fresh_canonical_comparison_or_physical_scale_proof",
    "historical_encoder_identity_not_established",
    "annotation_to_mosaic_transfer_not_assessed",
    "predicted_class_counts_are_not_accuracy",
    "not_every_slide_head_cell_is_patient_held_out",
    "future_query_fold_head_filters_both_query_and_references",
    "no_retrieval_or_scientific_acceptance",
    "trusted_local_pickle_not_untrusted_model_ingestion",
    "cooperative_inputs_and_directory_ownership_not_atomic_snapshot",
)

# These are the frozen first-party imports of the two accepted initializers.
# A spec cannot expand this graph into newly authorized computational helpers.
INHERITED = frozenset({
    "prosper.config", "prosper.pipeline", "prosper.embeddings",
    "prosper.embeddings.patch_embedding", "prosper.embeddings.patch_embedding.base",
    "prosper.embeddings.patch_embedding.gigapath_embedding",
    "prosper.embeddings.patch_embedding.uni2_embedding",
    "prosper.embeddings.patch_embedding.virchow2_embeddings",
    "prosper.embeddings.slide_embedding", "prosper.embeddings.slide_embedding.base",
    "prosper.embeddings.slide_embedding.bob_embedding",
    "prosper.embeddings.slide_embedding.ot_embedding",
    "prosper.embeddings.slide_embedding.percentile_embedding",
    "prosper.evaluation.metrics", "prosper.evaluation.paths", "prosper.evaluation.slide_headers",
    "prosper.preprocessing", "prosper.preprocessing.mosaic",
    "prosper.search", "prosper.search.roi_search", "prosper.search.slide_search",
})


class MosaicError(ValueError):
    """Invalid declaration, custody, coverage, prediction or written result."""


def require(condition, message):
    if not condition:
        raise MosaicError(message)


def _reader():
    # Fixed normal-package dependency, never a spec-selected import or fallback.
    return importlib.import_module("prosper.evaluation.r2_1_hoptimus_head_reader")


def code_paths(reader):
    direct = {
        "runner": Path(__file__), "head_reader": Path(reader.__file__),
        "head_registry": Path(registry.__file__),
        "fit": Path(fitter.__file__), "asset_io": Path(assets.__file__),
        "head_cli": Path(producer.__file__), "filter_kernel": Path(kernel.__file__),
        "filter_cli": Path(manifests.__file__), "cpu_io": Path(cpu.__file__),
        "legacy_io": Path(legacy.__file__), "package_init": Path(prosper.__file__),
        "evaluation_init": Path(prosper.evaluation.__file__),
    }
    for name in sorted(INHERITED):
        module = sys.modules.get(name)
        require(module is not None and getattr(module, "__file__", None) is not None,
                f"frozen inherited startup module not imported: {name}")
        direct["inherited:" + name] = Path(module.__file__)
    return direct


def source_limit(role):
    if role in ("head_registry", "accepted_heads"):
        return 2 * MIB
    if role == "head_plan":
        return 4 * MIB
    return 16 * MIB


def _file(entry, context, limit):
    producer._binding(entry, context, limit)
    require(Path(entry["path"]).is_absolute(), f"{context}: absolute explicit path required")


def _key(head):
    fold = producer._int(head["held_out_fold"], "head fold")
    budget = producer._int(head["budget_percent"], "head budget")
    require(fold < 5 and budget in (5, 100), "invalid head tuple")
    return fold, budget


def _name(key):
    return f"fold{key[0]}_budget{key[1]}"


def packed_bytes(tiles):
    producer._int(tiles, "original_tiles", 1)
    return tiles * (2 * 8 + 10 * (5 * 4 + 8 + 1))


def _schema(spec, paths, bank):
    cpu.keys(spec, {"schema_version", "run_id", "task", "sources", "code_files",
                    "features", "heads", "parameters", "expected"}, "mosaic spec")
    producer._version(spec, "mosaic spec")
    producer._text(spec["run_id"], "run ID")
    require(spec["task"] == TASK, "wrong mosaic task")
    cpu.keys(spec["sources"], sources_for(bank), "mosaic source roles")
    for role, entry in spec["sources"].items():
        _file(entry, role, source_limit(role))
    cpu.keys(spec["code_files"], paths, "direct/inherited executing code roles")
    for role, entry in spec["code_files"].items():
        _file(entry, role, 2 * MIB)
        require(Path(entry["path"]).resolve(strict=True) == paths[role].resolve(strict=True),
                f"{role}: wrong actual imported source location")
    cpu.keys(spec["parameters"], {"prediction_batch_rows"}, "mosaic parameters")
    require(producer._int(spec["parameters"]["prediction_batch_rows"], "batch rows") == 4096,
            "prediction batch rows must be 4096")
    cpu.keys(spec["expected"], EXPECTED, "mosaic expected")
    for name, value in spec["expected"].items():
        producer._int(value, name, 1)
    expected = spec["expected"]
    require(expected["heads"] == 10 and expected["feature_files"] == expected["slides"] and
            expected["slide_head_cells"] == expected["slides"] * 10 and
            expected["tile_head_predictions"] == expected["original_tiles"] * 10 and
            expected["patients"] <= expected["slides"] and
            expected["diagnostic_groups"] <= expected["slides"] and
            expected["patients"] <= expected["union_patients"],
            "expected head/cell/tile totals disagree")
    require(packed_bytes(expected["original_tiles"]) <= ARRAY_LIMIT,
            "packed array operational limit exceeds 2GiB")
    require(type(spec["features"]) is list and len(spec["features"]) == expected["feature_files"],
            "feature declaration count")
    seen, physical, widths = set(), set(), set()
    for feature in spec["features"]:
        cpu.keys(feature, {"slide_id", "path", "size", "sha256", "rows", "dtype", "columns"},
                 "mosaic feature")
        slide = producer._text(feature["slide_id"], "feature slide")
        require(slide not in seen, "duplicate feature slide")
        seen.add(slide)
        _file({k: feature[k] for k in ("path", "size", "sha256")}, slide, 64 * MIB)
        producer._int(feature["rows"], "feature rows", 1)
        require(feature["dtype"] == "float32", "mosaic feature profile must be float32")
        widths.add(producer._int(feature["columns"], "feature columns", 1))
        observed = Path(feature["path"]).stat()
        identity = observed.st_dev, observed.st_ino
        require(identity not in physical, "duplicate physical feature declaration")
        physical.add(identity)
    require(len(widths) == 1, "one feature width across the mosaic declaration")
    require(type(spec["heads"]) is list and len(spec["heads"]) == 10, "ten heads required")
    tuples = set()
    for head in spec["heads"]:
        cpu.keys(head, {"held_out_fold", "budget_percent", "assets"}, "saved head")
        key = _key(head)
        require(key not in tuples, "duplicate head tuple")
        tuples.add(key)
        cpu.keys(head["assets"], HEAD_ASSETS, "head asset roles")
        for role, entry in head["assets"].items():
            _file(entry, f"{_name(key)} {role}", 2 * MIB)
    require(tuples == set(TUPLES), "missing head tuple")


def declarations(spec, spec_path, spec_data, digest):
    result = [assets.AssetSpec("spec", "spec", spec_path, digest, len(spec_data))]
    policies = {
        "spec": assets.AssetPolicy("bytes", 8 * MIB),
        "code": assets.AssetPolicy("bytes", 2 * MIB),
        "feature": assets.AssetPolicy("bytes", 64 * MIB),
        "head_control": assets.AssetPolicy("bytes", 2 * MIB),
        "head_model": assets.AssetPolicy("bytes", 2 * MIB),
    }
    for role, entry in spec["sources"].items():
        policies[role] = assets.AssetPolicy("bytes", source_limit(role))
        result.append(assets.AssetSpec("source:" + role, role, **entry))
    for role, entry in spec["code_files"].items():
        result.append(assets.AssetSpec("code:" + role, "code", **entry))
    for feature in spec["features"]:
        result.append(assets.AssetSpec("feature:" + feature["slide_id"], "feature",
                                     **{k: feature[k] for k in ("path", "size", "sha256")}))
    for head in spec["heads"]:
        for role, entry in head["assets"].items():
            result.append(assets.AssetSpec(
                f"head:{_name(_key(head))}:{role}",
                "head_model" if role == "model" else "head_control", **entry))
    return result, policies


def _binding_map(summary):
    require(type(summary.get("input_bindings")) is list, "Stage0 binding list required")
    result = {}
    for entry in summary["input_bindings"]:
        require(type(entry) is dict, "Stage0 binding must be an object")
        key = producer._text(entry.get("kind"), "binding kind"), producer._text(entry.get("name"), "binding name")
        require(key not in result, "duplicate Stage0 input binding")
        producer._digest(entry.get("sha256"), "Stage0 binding hash")
        producer._int(entry.get("size"), "Stage0 binding size")
        for name in ("path", "resolved_path"):
            require(Path(producer._text(entry.get(name), "Stage0 " + name)).is_absolute(),
                    "Stage0 recorded input path must be absolute")
        result[key] = entry
    return result


def _content(entry, data, context):
    require(type(entry) is dict, f"{context}: binding required")
    producer._same(entry.get("size"), len(data), context + " size")
    producer._same(entry.get("sha256"), producer._sha(data), context + " hash")


def authenticate(spec, sources, bank):
    review = producer._json(sources["stage0_review"], "Stage0 review")
    complete = producer._json(sources["stage0_complete"], "Stage0 COMPLETE")
    summary = producer._json(sources["stage0_summary"], "Stage0 summary")
    cpu._review(review, "Stage0 feature-bank binding only", "Stage0")
    producer._version(complete, "Stage0 COMPLETE")
    producer._version(summary, "Stage0 summary")
    require(complete.get("scope") == "feature-bank binding only" and
            type(summary.get("scope")) is str and summary["scope"].startswith("feature-bank binding only") and
            producer._text(complete.get("run_id"), "Stage0 run") == summary.get("run_id") ==
            summary.get("declarations", {}).get("run_id"), "Stage0 run/scope mismatch")
    cpu.keys(complete.get("outputs"), {"inventory.csv", "summary.json"}, "Stage0 outputs")
    links = cpu._review_links(review, spec["sources"]["stage0_review"]["path"])
    for role, filename in (("stage0_complete", "COMPLETE.json"),
                           ("stage0_inventory", "inventory.csv"), ("stage0_summary", "summary.json")):
        path = Path(spec["sources"][role]["path"]).resolve(strict=True)
        require(path.name == filename and links.get(str(path)) == producer._sha(sources[role]),
                f"Stage0 selected {filename} review link")
        if role != "stage0_complete":
            require(path.parent == Path(spec["sources"]["stage0_complete"]["path"]).resolve().parent,
                    "Stage0 consumed output location")
            _content(complete["outputs"][filename], sources[role], filename)
    old = _binding_map(summary)
    for role, field in (("metadata", "metadata"), ("mosaic", "canonical_mosaic")):
        key = (role, "metadata" if role == "metadata" else "canonical")
        require(key in old and review.get("source_sha256", {}).get(field) == old[key]["sha256"],
                f"Stage0 {role} lineage mismatch")
    require(("spec", "spec") in old and links.get(str(Path(old["spec", "spec"]["resolved_path"]).resolve())) ==
            old["spec", "spec"]["sha256"], "Stage0 spec lineage not accepted")
    original_spec = old["spec", "spec"]
    require(Path(original_spec["path"]).resolve() == Path(original_spec["resolved_path"]).resolve(),
            "Stage0 original spec lexical/resolved identity")
    original_base = Path(original_spec["path"]).parent
    metadata = summary["declarations"].get("metadata", {})
    require(all(metadata.get(name) == cpu.METADATA_COLUMNS[name]
                for name in ("id_col", "patient_col", "label_col")) and
            (original_base / producer._text(metadata.get("path"), "metadata declaration")).resolve() ==
            Path(old["metadata", "metadata"]["resolved_path"]).resolve(),
            "Stage0 metadata selectors/original-spec-relative identity")
    mosaic = summary["declarations"].get("mosaic", {})
    require(type(mosaic.get("level")) is int and mosaic["level"] == 0 and
            type(mosaic.get("patch_size")) is int and mosaic["patch_size"] == 512,
            "Stage0 geometry is not level0/512")
    require((original_base / producer._text(mosaic.get("path"), "canonical declaration")).resolve() ==
            Path(old["mosaic", "canonical"]["resolved_path"]).resolve(),
            "Stage0 canonical declaration/binding path mismatch")
    banks = summary["declarations"].get("banks")
    require(type(banks) is list, "Stage0 bank declarations required")
    selected = [entry for entry in banks if entry.get("name") == bank]
    require(len(selected) == 1 and type(selected[0].get("feature_dim")) is int and
            selected[0]["feature_dim"] == fitter._WIDTH and
            {feature["columns"] for feature in spec["features"]} == {selected[0]["feature_dim"]},
            f"authenticated Stage0 {bank} width must equal the declared input width and fit _WIDTH")
    require(Path(producer._text(selected[0].get("path"), "Stage0 bank path")).is_absolute(),
            "Stage0 bank path must be absolute")
    require({bank.get("name") for bank in banks} == {"uni2", "virchow2", "h_optimus_1"} and
            len(banks) == 3, "Stage0 exact three bank declarations")
    consumed = {
        "input_bindings.json": sources["cpu_input_bindings"],
        "retrieval_population.csv": sources["retrieval_population"],
        "patient_folds.csv": sources["patient_folds"], "summary.json": sources["cpu_summary"],
        "head_plan.json": sources["head_plan"],
    }
    assets.verify_cpu_review_link(
        sources["cpu_review"], sources["cpu_complete"], branch="filter",
        receipt_key="filter_attempt_001/COMPLETE.json", consumed=consumed)
    bindings = producer._json(sources["cpu_input_bindings"], "CPU bindings")
    current = {(entry["kind"], entry["name"]): entry for entry in bindings["bindings"]}
    for role in ("stage0_review", "stage0_complete", "stage0_inventory", "stage0_summary"):
        _content(current.get(("source", role)), sources[role], "CPU " + role)
    prior_metadata = current.get(("source", "metadata"))
    require(prior_metadata is not None, "CPU metadata lineage missing")
    for field in ("size", "sha256"):
        producer._same(prior_metadata.get(field), old["metadata", "metadata"][field],
                       "CPU/Stage0 metadata " + field)
    require(Path(producer._text(prior_metadata.get("resolved_path"), "CPU metadata path")).resolve() ==
            Path(old["metadata", "metadata"]["resolved_path"]).resolve(), "CPU/Stage0 metadata identity")
    for direct, historical in (
        ("filter_kernel", "kernel"), ("filter_cli", "runner"), ("cpu_io", "io"),
        ("legacy_io", "legacy_io"), ("package_init", "package_init"),
        ("evaluation_init", "evaluation_init"),
    ):
        prior = current.get(("code", historical))
        require(prior is not None, f"missing CPU code/{historical}")
        producer._same(prior["sha256"], spec["code_files"][direct]["sha256"], direct + " code hash")
        producer._same(prior["size"], spec["code_files"][direct]["size"], direct + " code size")
    producer._same(review["source_sha256"]["auditor_module"],
                   spec["code_files"]["legacy_io"]["sha256"], "Stage0 accepted decoder")
    accepted_decoder = old.get(("code", "running_auditor"))
    require(accepted_decoder is not None, "Stage0 decoder binding missing")
    for field in ("size", "sha256"):
        producer._same(accepted_decoder.get(field), spec["code_files"]["legacy_io"][field],
                       "Stage0 decoder " + field)
    cpu_summary = producer._json(sources["cpu_summary"], "CPU summary")
    producer._version(cpu_summary, "CPU summary")
    require(cpu_summary.get("task") == manifests.TASK and cpu_summary.get("run_id") == bindings["run_id"],
            "CPU summary family mismatch")
    producer._same(cpu_summary["policy"], manifests.POLICY, "CPU filtering policy")
    for name in ("canonical_slides", "canonical_tiles", "inventory_rows"):
        producer._int(summary.get(name), "Stage0 " + name, 1)
    producer._same(cpu_summary["counts"]["raw_slides"], summary["canonical_slides"],
                   "CPU/Stage0 raw slide population")
    for key, historical in (("slides", "target_slides"), ("patients", "patients"),
                            ("diagnostic_groups", "groups")):
        producer._same(spec["expected"][key], cpu_summary["counts"][historical], "CPU " + key)
    plans = producer._plans(sources["head_plan"], bindings["run_id"])
    return summary, plans, consumed, selected[0]


def reconcile(spec, sources, stage_summary, bank, bank_declaration):
    expected = spec["expected"]
    mapping, retrieval_flags, annotation_flags, order, presented = {}, {}, {}, {}, []
    for row in producer.csv_rows(sources["patient_folds"], manifests.FOLD_HEADER, "patient folds",
                                 expanded_limit=16 * MIB, row_limit=expected["union_patients"]):
        patient = producer._text(row["patient_id"], "fold patient")
        fold = producer._decimal(row["fold"], "patient fold")
        index = producer._decimal(row["kernel_order"], "patient kernel order")
        require(patient not in mapping and index not in order and fold < 5 and
                row["patient_hash"] == kernel.patient_fold_hash(patient), "patient fold/hash/order")
        require(row["has_annotations"] in ("0", "1") and row["has_retrieval"] in ("0", "1") and
                (row["has_annotations"], row["has_retrieval"]) != ("0", "0"),
                "patient union membership flags")
        mapping[patient], order[index] = fold, patient
        retrieval_flags[patient] = row["has_retrieval"] == "1"
        annotation_flags[patient] = row["has_annotations"] == "1"
        presented.append(patient)
    require(len(mapping) == expected["union_patients"] and presented == sorted(mapping) and
            set(order) == set(range(len(mapping))) and
            [order[i] for i in range(len(mapping))] == sorted(
                mapping, key=lambda patient: (kernel.patient_fold_hash(patient), patient)),
            "complete patient fold presentation/kernel order")
    cpu_summary = producer._json(sources["cpu_summary"], "CPU summary")
    producer._same(sum(annotation_flags.values()), cpu_summary["counts"]["annotation_patients"],
                   "CPU annotation patient membership")
    population = []
    slides, groups, patients, metadata_indices = set(), set(), set(), set()
    for index, row in enumerate(producer.csv_rows(
        sources["retrieval_population"], manifests.RETRIEVAL_HEADER, "mosaic population",
        expanded_limit=16 * MIB, row_limit=expected["slides"],
    )):
        for name in ("query_index", "reference_index", "metadata_index", "fold"):
            row[name] = producer._decimal(row[name], name)
        require(row["query_index"] == index == row["reference_index"], "population index/order")
        require(row["metadata_index"] not in metadata_indices, "duplicate population metadata index")
        metadata_indices.add(row["metadata_index"])
        for name in ("slide_id", "patient_id", "diagnostic_group"):
            producer._text(row[name], name)
        require(row["slide_id"] not in slides and row["diagnostic_group"] not in ("MISC", "NORMAL"),
                "duplicate/excluded population slide")
        require(row["patient_id"] in mapping and mapping[row["patient_id"]] == row["fold"],
                "population patient fold mismatch")
        slides.add(row["slide_id"]); patients.add(row["patient_id"]); groups.add(row["diagnostic_group"])
        population.append(row)
    require(len(population) == expected["slides"] and len(patients) == expected["patients"] and
            len(groups) == expected["diagnostic_groups"], "population scope counts")
    require(patients == {patient for patient, eligible in retrieval_flags.items() if eligible},
            "complete retrieval patient fold membership")
    require(sum(annotation_flags.values()) > 0, "annotation union unexpectedly empty")
    require([feature["slide_id"] for feature in spec["features"]] ==
            [row["slide_id"] for row in population], "feature declaration/population order mismatch")
    main, bank_slides = {}, {}
    seen, bank_totals, dtype_counts = set(), Counter(), {}
    bank_bindings = _binding_map(stage_summary)
    for row in producer.csv_rows(sources["stage0_inventory"], INVENTORY_HEADER, "Stage0 inventory",
                                 expanded_limit=16 * MIB, row_limit=16 * MIB):
        key = row["bank"], row["slide_id"]
        require(row["bank"] in ("uni2", "virchow2", "h_optimus_1") and key not in seen,
                "Stage0 unknown/duplicate bank/slide")
        seen.add(key)
        for name in ("canonical_tiles", "feature_rows", "size"):
            row[name] = producer._decimal(row[name], "inventory " + name)
        producer._digest(row["sha256"], "inventory hash")
        require(row["canonical_tiles"] == row["feature_rows"] > 0 and
                row["dtype"] in ("float16", "float32"), "canonical/feature row count/dtype mismatch")
        prior = bank_bindings.get(("feature", row["bank"] + "/" + row["slide_id"]))
        require(prior is not None, "inventory feature absent from Stage0 bindings")
        for field, bound in (("feature_path", "path"), ("resolved_path", "resolved_path"),
                             ("size", "size"), ("sha256", "sha256")):
            producer._same(row[field], prior.get(bound), "inventory/Stage0 binding " + field)
        bank_totals[row["bank"]] += row["feature_rows"]
        bank_slides.setdefault(row["bank"], set()).add(row["slide_id"])
        dtype_counts.setdefault(row["bank"], Counter())[row["dtype"]] += 1
        if row["bank"] == bank:
            main[row["slide_id"]] = row
    require(len(seen) == producer._int(stage_summary["inventory_rows"], "Stage0 inventory rows") and
            len(main) == stage_summary["canonical_slides"] and
            all(ids == set(main) for ids in bank_slides.values()) and
            set(bank_slides) == {"uni2", "virchow2", "h_optimus_1"} and
            all(total == stage_summary["canonical_tiles"] for total in bank_totals.values()) and
            dict(bank_totals) == stage_summary["bank_tile_totals"] and
            {name: dict(counts) for name, counts in dtype_counts.items()} == stage_summary["bank_dtype_counts"],
            "Stage0 inventory/summary totals")
    producer._same(dict(bank_totals), stage_summary["bank_tile_totals"], "strict Stage0 bank totals")
    producer._same({name: dict(counts) for name, counts in dtype_counts.items()},
                   stage_summary["bank_dtype_counts"], "strict Stage0 dtype counts")
    require({name for kind, name in bank_bindings if kind == "feature"} ==
            {bank + "/" + slide for bank, slide in seen}, "Stage0 feature-binding/catalog coverage")
    producer._same(stage_summary["declarations"]["mosaic"]["expected_slides"],
                   stage_summary["canonical_slides"], "Stage0 declared slide count")
    producer._same(stage_summary["declarations"]["mosaic"]["expected_tiles"],
                   stage_summary["canonical_tiles"], "Stage0 declared tile count")
    require(stage_summary["declarations"]["mosaic"]["expected_slides"] ==
            stage_summary["canonical_slides"] and stage_summary["declarations"]["mosaic"]["expected_tiles"] ==
            stage_summary["canonical_tiles"], "Stage0 canonical declaration totals")
    require(slides <= set(main), f"mosaic population absent from selected {bank} bank")
    bank_dir = Path(bank_declaration["path"]).resolve(strict=True)
    if bank in M1_FEATURE_DIRS:
        require(bank_dir == Path(M1_FEATURE_DIRS[bank]).resolve(strict=True),
                f"M-1: Stage0 {bank} bank directory is not {M1_FEATURE_DIRS[bank]}")
    offsets, start = [], 0
    for index, (row, feature) in enumerate(zip(population, spec["features"])):
        original = main[row["slide_id"]]
        for field, historical in (("path", "feature_path"), ("size", "size"),
                                  ("sha256", "sha256"), ("rows", "feature_rows"), ("dtype", "dtype")):
            producer._same(feature[field], original[historical], "main feature " + field)
        expected_path = bank_dir / (row["slide_id"] + ".pkl")
        require(Path(feature["path"]).resolve(strict=True) == expected_path and
                Path(original["resolved_path"]).resolve(strict=True) == expected_path,
                f"M-1: {row['slide_id']} feature_path does not resolve to {expected_path}")
        stop = start + feature["rows"]
        offsets.append({
            "slide_index": index, "slide_id": row["slide_id"], "patient_id": row["patient_id"],
            "diagnostic_group": row["diagnostic_group"], "slide_fold": row["fold"],
            "tile_start": start, "tile_stop": stop, "feature_rows": feature["rows"],
            "feature_sha256": feature["sha256"],
        })
        start = stop
    require(start == expected["original_tiles"], "original mosaic tile total")
    validate_offsets(offsets, expected)
    return population, offsets


def validate_offsets(offsets, expected):
    require(len(offsets) == expected["slides"], "offset slide count")
    end, slides = 0, set()
    for index, row in enumerate(offsets):
        require(row["slide_index"] == index and row["slide_id"] not in slides and
                row["tile_start"] == end and row["tile_stop"] - row["tile_start"] ==
                row["feature_rows"] > 0, "offset overlap/gap/duplicate")
        slides.add(row["slide_id"])
        end = row["tile_stop"]
    require(end == expected["original_tiles"], "offset coverage incomplete")


def registered_heads(sources, declared_heads, bank):
    """Parse HEAD_REGISTRY.json; its artifacts must be exactly the spec-declared head assets."""
    entries = registry.parse_registry(
        sources["head_registry"], bank=bank, head_plan_data=sources["head_plan"],
        accepted_data=sources.get("accepted_heads"))
    for key in TUPLES:
        artifacts = entries[_name(key)]["artifacts"]
        for role in HEAD_ASSETS:
            declared = declared_heads[key]["assets"][role]
            require({name: declared[name] for name in ("path", "size", "sha256")} == artifacts[role],
                    f"{_name(key)} {role}: spec head asset differs from HEAD_REGISTRY.json")
    return entries


def load_registered_head(guard, sources, entry, key, bank, width):
    data = {role: guard.read_verified(f"head:{_name(key)}:{role}") for role in HEAD_ASSETS}
    data["head_plan"] = sources["head_plan"]
    return registry.verify_head(
        data, held_out_fold=key[0], budget_percent=key[1], bank=bank,
        accepted_sha256=entry["accepted_head_sha256"], width=width,
        producer_spec_path=entry["producer_spec_path"])


def producer_links(guard, declared_heads, entries):
    records = []
    roles = dict(zip(REGISTRY_NAMES, ("model", "diagnostics", "input_bindings", "summary")))
    for key in TUPLES:
        declared, entry = declared_heads[key], entries[_name(key)]
        complete = producer._json(guard.read_verified("head:" + _name(key) + ":complete"),
                                  "already-authenticated producer COMPLETE")
        records.append({
            "held_out_fold": key[0], "budget_percent": key[1],
            "registry_directory": entry["directory"],
            "complete_current_path": declared["assets"]["complete"]["path"],
            "producer_spec_current_path": declared["assets"]["spec"]["path"],
            "producer_spec_historical_path": entry["producer_spec_path"],
            "accepted_head_sha256": entry["accepted_head_sha256"],
            "consumed_artifacts": {
                name: {"asset_id": f"head:{_name(key)}:{role}",
                       "current_path": declared["assets"][role]["path"],
                       **complete["outputs"][name]} for name, role in roles.items()},
            "referenced_not_reread": {
                name: info for name, info in complete["outputs"].items() if name not in roles},
        })
    return records


def allocate(tiles, slides):
    require(packed_bytes(tiles) <= ARRAY_LIMIT, "packed array operational limit exceeds 2GiB")
    coordinates = np.full((tiles, 2), -1, np.int64)
    arrays = {key: (
        np.full((tiles, 5), np.nan, np.float32),
        np.full(tiles, -1, np.int64),
        np.zeros(tiles, bool),
    ) for key in TUPLES}
    return coordinates, arrays, np.zeros((10, slides), bool), np.zeros((10, slides), np.int64)


def decode_feature(data, feature):
    payload = legacy._pickle(data, "main mosaic " + feature["slide_id"])
    require(type(payload) is tuple and len(payload) == 2, "main feature tuple required")
    raw, tiles = payload
    require(type(raw) is np.ndarray and raw.dtype == np.dtype(np.float32) and
            raw.shape == (feature["rows"], feature["columns"]) and bool(np.isfinite(raw).all()),
            "main feature base FP32 shape/rows/width/finiteness")
    coordinates = legacy._coordinates(tiles, "main mosaic " + feature["slide_id"])
    require(len(coordinates) == feature["rows"] and
            all(0 <= value <= np.iinfo(np.int64).max for xy in coordinates for value in xy),
            "main coordinates require nonnegative unique int64 XY")
    return raw, np.array(coordinates, dtype=np.int64)


def score_slide(heads, raw, xy, row, coordinates, arrays, cells, rows):
    index, start, stop = row["slide_index"], row["tile_start"], row["tile_stop"]
    require(stop - start == len(raw) == len(xy), "slide feature/offset row mismatch")
    require(not cells[:, index].any() and not rows[:, index].any(), "slide/head slice already written")
    coordinates[start:stop] = xy
    for number, key in enumerate(TUPLES):
        probabilities, labels, mask = arrays[key]
        written = 0
        for offset in range(0, len(raw), 4096):
            feature_batch = np.ascontiguousarray(raw[offset:offset + 4096])
            batch = fitter.predict_head(heads[key], feature_batch)
            length = min(4096, len(raw) - offset)
            require(len(batch.labels) == length and batch.probabilities.shape == (length, 5),
                    "prediction batch row count")
            require(offset == written, "noncontiguous or duplicated prediction batch")
            target = slice(start + offset, start + offset + length)
            probabilities[target] = batch.probabilities
            labels[target] = batch.labels
            mask[target] = batch.labels == 1
            written += length
            del feature_batch, batch
        require(written == stop - start, "incomplete slide/head predictions")
        rows[number, index] = written
        cells[number, index] = True


def validate_array_slice(probabilities, labels, mask):
    require(type(probabilities) is np.ndarray and probabilities.dtype.str == "<f4" and
            probabilities.ndim == 2 and probabilities.shape[1] == 5 and
            type(labels) is np.ndarray and labels.dtype.str == "<i8" and
            labels.shape == (len(probabilities),) and type(mask) is np.ndarray and
            mask.dtype.str == "|b1" and mask.shape == labels.shape, "packed array dtype/shape")
    for start in range(0, len(labels), 4096):
        p, y, keep = probabilities[start:start + 4096], labels[start:start + 4096], mask[start:start + 4096]
        require(bool(np.isfinite(p).all()) and bool(((p >= 0) & (p <= 1)).all()) and
                bool((np.abs(p.sum(axis=1, dtype=np.float64) - 1) <=
                      8 * np.finfo(np.float32).eps).all()), "packed probability validity")
        require(np.array_equal(y, p.argmax(axis=1) + 1) and np.array_equal(keep, y == 1),
                "packed argmax/Boolean mask mismatch")


def validate_coverage(cells, processed, offsets, expected):
    shape = (10, expected["slides"])
    require(type(cells) is np.ndarray and cells.dtype.str == "|b1" and cells.shape == shape and
            type(processed) is np.ndarray and processed.dtype.str == "<i8" and processed.shape == shape and
            bool(cells.all()) and all(processed[number, index] == row["feature_rows"]
                                     for number in range(10) for index, row in enumerate(offsets)) and
            int(processed.sum()) == expected["tile_head_predictions"],
            "explicit row/cell prediction coverage incomplete")


def counts_rows(offsets, arrays):
    for key in TUPLES:
        _, labels, mask = arrays[key]
        for row in offsets:
            start, stop = row["tile_start"], row["tile_stop"]
            counts = np.bincount(labels[start:stop], minlength=6)[1:6]
            kept = int(mask[start:stop].sum())
            yield (
                *key, row["slide_index"], row["slide_id"], row["patient_id"], row["slide_fold"],
                int(row["slide_fold"] == key[0]), row["feature_rows"],
                *(int(value) for value in counts), kept, row["feature_rows"] - kept,
                "nonempty" if kept else "zero_predicted_tumor",
            )


def _bool_output(output, name, array, bindings):
    require(type(array) is np.ndarray and array.dtype.str == "|b1" and
            array.ndim == 1 and array.flags.c_contiguous, "Boolean NPY mask required")
    output.write(name, lambda handle: np.lib.format.write_array(handle, array, version=(1, 0),
                                                               allow_pickle=False))

    def check(handle):
        require(np.lib.format.read_magic(handle) == (1, 0), "Boolean NPY version")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        require(shape == array.shape and not fortran and dtype.str == "|b1",
                "Boolean mask header mismatch")
        require(os.fstat(handle.fileno()).st_size == handle.tell() + array.nbytes,
                "Boolean mask trailing/truncated bytes")
        handle.seek(0)
        restored = np.load(handle, allow_pickle=False)
        producer._array_same(restored, array, name)
        require(handle.read(1) == b"", "Boolean mask extra array")

    binding, _ = output.readback(name, array.nbytes + 65545, check)
    bindings[name] = binding


def _csv_output(output, name, header, row_factory, row_count, bindings, *, compressed=False):
    def write(handle):
        raw = gzip.GzipFile(fileobj=handle, mode="wb", filename="", mtime=0) if compressed else handle
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        try:
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(row_factory())
            text.flush()
        finally:
            text.detach()
            if compressed:
                raw.close()

    output.write(name, write)
    cap = (row_count + 1) * OUTPUT_RECORD_LIMIT + 4096

    def check(handle):
        if compressed:
            require(handle.read(8) == b"\x1f\x8b\x08\x00" + b"\x00" * 4, "count gzip header/mtime")
            handle.seek(0)
        count = 0
        expected = iter(row_factory())
        for row in producer.csv_rows(handle, header, name, compressed=compressed,
                                     expanded_limit=cap, row_limit=row_count,
                                     record_limit=OUTPUT_RECORD_LIMIT):
            wanted = next(expected, None)
            require(wanted is not None and tuple(row.values()) == tuple(str(x) for x in wanted),
                    f"{name}: serialized row/count mismatch")
            count += 1
        require(count == row_count and next(expected, None) is None, f"{name}: incomplete rows")

    binding, _ = output.readback(name, cap, check)
    bindings[name] = binding


def artifact_names():
    return {"coordinates.npy", "retrieval_population.csv", "slide_offsets.csv",
            "mask_counts.csv.gz", "summary.json", "input_bindings.json"} | {
        f"{_name(key)}_{kind}.npy" for key in TUPLES for kind in ("probabilities", "predictions", "mask")
    }


def _written_recount(output, offsets, expected, zero_cells, bindings):
    """Read one written tuple at a time; no full thirty-array readback cache."""
    def array(name, shape, dtype):
        def check(handle):
            require(np.lib.format.read_magic(handle) == (1, 0), "written NPY version")
            observed_shape, fortran, observed_dtype = np.lib.format.read_array_header_1_0(handle)
            require(observed_shape == shape and not fortran and observed_dtype.str == dtype and
                    os.fstat(handle.fileno()).st_size == handle.tell() +
                    int(np.prod(shape)) * observed_dtype.itemsize, "written NPY shape/dtype/EOF")
            handle.seek(0)
            value = np.load(handle, allow_pickle=False)
            require(handle.read(1) == b"", "written NPY trailing bytes")
            return value
        return output.readback(name, bindings[name]["size"], check, expected=bindings[name])[1]

    def offset_readback(handle):
        written = []
        for row in producer.csv_rows(handle, OFFSET_HEADER, "written offsets",
                                     expanded_limit=bindings["slide_offsets.csv"]["size"],
                                     row_limit=expected["slides"], record_limit=OUTPUT_RECORD_LIMIT):
            for name in ("slide_index", "slide_fold", "tile_start", "tile_stop", "feature_rows"):
                row[name] = producer._decimal(row[name], "written " + name)
            written.append(row)
        validate_offsets(written, expected)
        producer._same(written, offsets, "independently written offsets")
        return written
    _, written_offsets = output.readback("slide_offsets.csv", bindings["slide_offsets.csv"]["size"],
                                        offset_readback, expected=bindings["slide_offsets.csv"])
    coordinates = array("coordinates.npy", (expected["original_tiles"], 2), "<i8")
    require(bool((coordinates >= 0).all()), "written coordinate recount")
    for row in written_offsets:
        xy = coordinates[row["tile_start"]:row["tile_stop"]]
        require(len({tuple(pair) for pair in xy}) == len(xy), "written duplicate coordinate")
    del coordinates

    def recount(handle):
        count_rows = iter(producer.csv_rows(
            handle, COUNT_HEADER, "written counts", compressed=True,
            expanded_limit=(expected["slide_head_cells"] + 1) * OUTPUT_RECORD_LIMIT + 4096,
            row_limit=expected["slide_head_cells"], record_limit=OUTPUT_RECORD_LIMIT))
        recounted_zero, cells, predictions = 0, 0, 0
        for key in TUPLES:
            p = array(f"{_name(key)}_probabilities.npy", (expected["original_tiles"], 5), "<f4")
            y = array(f"{_name(key)}_predictions.npy", (expected["original_tiles"],), "<i8")
            mask = array(f"{_name(key)}_mask.npy", (expected["original_tiles"],), "|b1")
            validate_array_slice(p, y, mask)
            for row in written_offsets:
                start, stop = row["tile_start"], row["tile_stop"]
                counts = np.bincount(y[start:stop], minlength=6)[1:6]
                kept = int(mask[start:stop].sum())
                wanted = (*key, row["slide_index"], row["slide_id"], row["patient_id"], row["slide_fold"],
                          int(row["slide_fold"] == key[0]), stop - start,
                          *(int(value) for value in counts), kept, stop - start - kept,
                          "nonempty" if kept else "zero_predicted_tumor")
                actual = next(count_rows, None)
                require(actual is not None and tuple(actual.values()) == tuple(str(x) for x in wanted),
                        "written array/CSV independent class/mask/count mismatch")
                recounted_zero += int(kept == 0)
                cells += 1
                predictions += stop - start
            del p, y, mask
        require(next(count_rows, None) is None and cells == expected["slide_head_cells"] and
                predictions == expected["tile_head_predictions"] and recounted_zero == zero_cells,
                "written independent cell/tile/zero-mask coverage recount")
    output.readback("mask_counts.csv.gz", bindings["mask_counts.csv.gz"]["size"], recount,
                    expected=bindings["mask_counts.csv.gz"])


def run(spec_path, spec_sha256, out, out_root, *, bank):
    require(bank in BANKS, "unsupported bank")
    reader = _reader()
    started = producer._time()
    producer._digest(spec_sha256, "mosaic spec hash")
    spec_path = Path(spec_path).absolute()
    data, initial = cpu._scan_file(spec_path, "spec", "spec", 8 * MIB)
    require(initial["sha256"] == spec_sha256, "mosaic spec hash mismatch")
    spec = producer._json(data, "mosaic spec")
    paths = code_paths(reader)
    _schema(spec, paths, bank)
    width = spec["features"][0]["columns"]
    declared, policies = declarations(spec, spec_path, data, spec_sha256)
    source_paths = [Path(entry.path) for entry in declared]
    feature_roots = [Path(feature["path"]).parent for feature in spec["features"]]
    output = producer._Output(out, out_root, source_paths, feature_roots)
    primary = None
    try:
        with assets.AssetGuard(declared, base_dir=spec_path.parent, policies=policies) as guard:
            require(guard.read_verified("spec") == data, "mosaic spec changed on acquisition")
            observed = spec_path.stat()
            require((observed.st_dev, observed.st_ino, observed.st_size) ==
                    (initial["device"], initial["inode"], initial["size"]), "mosaic spec identity changed")
            for role in spec["code_files"]:
                guard.read_verified("code:" + role)
            sources = {role: guard.read_verified("source:" + role) for role in sources_for(bank)}
            stage, _, consumed, bank_declaration = authenticate(spec, sources, bank)
            population, offsets = reconcile(spec, sources, stage, bank, bank_declaration)
            declarations_by_key = {_key(head): head for head in spec["heads"]}
            entries = registered_heads(sources, declarations_by_key, bank)
            output.claim()
            heads, provenance = {}, {}
            for key in TUPLES:
                saved, evidence = load_registered_head(guard, sources, entries[_name(key)], key, bank, width)
                require(type(saved) is fitter.FittedHead, "registry loader returned wrong wrapper")
                require(type(evidence) is dict and
                        evidence["held_out_fold"] == key[0] and evidence["budget_percent"] == key[1],
                        "registry loader returned wrong tuple/provenance")
                heads[key], provenance[key] = saved, evidence
            head_links = producer_links(guard, declarations_by_key, entries)
            m1_checked = 0
            coordinates, arrays, cells, processed = allocate(
                spec["expected"]["original_tiles"], spec["expected"]["slides"])
            decoded, encoded_bytes = 0, 0
            for feature, row in zip(spec["features"], offsets):
                encoded = guard.read_verified("feature:" + feature["slide_id"])
                # M-1 (addendum A4c): the offsets hash is that of the bytes actually decoded.
                require(producer._sha(encoded) == row["feature_sha256"] == feature["sha256"],
                        f"M-1: decoded {feature['slide_id']} bytes != slide_offsets/Stage0 sha256")
                m1_checked += 1
                raw, xy = decode_feature(encoded, feature)
                encoded_bytes += len(encoded)
                del encoded
                score_slide(heads, raw, xy, row, coordinates, arrays, cells, processed)
                decoded += 1
                del raw, xy
            require(decoded == spec["expected"]["feature_files"] == m1_checked, "decoded feature-file coverage")
            validate_coverage(cells, processed, offsets, spec["expected"])
            for key in TUPLES:
                validate_array_slice(*arrays[key])
            require(bool((coordinates >= 0).all()), "coordinate slices not completely written")
            bindings = {}
            producer._array_output(output, "coordinates.npy", coordinates, bindings)
            del coordinates
            offset_factory = lambda: (tuple(row[name] for name in OFFSET_HEADER) for row in offsets)
            _csv_output(output, "slide_offsets.csv", OFFSET_HEADER, offset_factory,
                        len(offsets), bindings)
            count_factory = lambda: counts_rows(offsets, arrays)
            _csv_output(output, "mask_counts.csv.gz", COUNT_HEADER, count_factory,
                        spec["expected"]["slide_head_cells"], bindings, compressed=True)
            population_bytes = sources["retrieval_population"]
            output.write("retrieval_population.csv", lambda handle: handle.write(population_bytes))
            binding, _ = output.readback(
                "retrieval_population.csv", 16 * MIB,
                lambda handle: require(handle.read() == population_bytes, "population copy byte mismatch"))
            bindings["retrieval_population.csv"] = binding
            zero_cells = sum(row[-1] == "zero_predicted_tumor" for row in count_factory())
            for key in TUPLES:
                p, y, mask = arrays[key]
                producer._array_output(output, f"{_name(key)}_probabilities.npy", p, bindings)
                producer._array_output(output, f"{_name(key)}_predictions.npy", y, bindings)
                _bool_output(output, f"{_name(key)}_mask.npy", mask, bindings)
                del arrays[key]
                del p, y, mask
            _written_recount(output, offsets, spec["expected"], zero_cells, bindings)
            summary = {
                "schema_version": 1, "run_id": spec["run_id"], "task": TASK,
                "encoder": bank, "bank": bank,
                "policy": {
                    "scope": SCOPE, "budgets": [5, 100], "folds": [0, 1, 2, 3, 4],
                    "classes": [1, 2, 3, 4, 5], "decision": "probability_argmax_class1",
                    "prediction_batch_rows": 4096, "population_order": "accepted_cpu_filter_order",
                    "coordinates": "accepted_stage0_order_reuse_unchanged_feature_bytes",
                    "no_new_canonical_comparison": True,
                },
                "counts": {**spec["expected"], "zero_mask_cells": zero_cells,
                           "nonzero_mask_cells": spec["expected"]["slide_head_cells"] - zero_cells},
                "details": {
                    "head_provenance": [provenance[key] for key in TUPLES],
                    "tuple_order": [list(key) for key in TUPLES], "feature_columns": width,
                    "coordinate_columns": 2,
                    "packed_array_data_bytes": packed_bytes(spec["expected"]["original_tiles"]),
                    "array_count": 31, "decoded_feature_files": decoded,
                    "consumed_feature_bytes": encoded_bytes,
                    "minimum_feature_bytes_with_final_sweep": encoded_bytes * 2,
                    "maximum_encoded_feature_bytes": max(feature["size"] for feature in spec["features"]),
                    "maximum_dense_feature_bytes": max(feature["rows"] * width * 4
                                                       for feature in spec["features"]),
                    "maximum_contiguous_feature_batch_bytes": min(
                        4096, max(feature["rows"] for feature in spec["features"])) * width * 4,
                    "coverage_accounting_bytes": cells.nbytes + processed.nbytes,
                    "written_slide_head_cells": int(cells.sum()),
                    "written_tile_head_predictions": int(processed.sum()),
                    "consumed_source_roles": sorted(sources_for(bank)),
                    "semantic_readback": "all_arrays_offsets_counts_masks_and_coverage_recounted",
                },
                "limitations": list(LIMITATIONS),
            }
            producer._json_output(output, "summary.json", summary, bindings)
            output.check(bindings)
            final = guard.final_bindings()
            input_record = {
                "schema_version": 1, "run_id": spec["run_id"], "task": TASK,
                "encoder": bank, "bank": bank,
                "spec_sha256": spec_sha256, "bindings": list(final),
                "executing_code_paths": {name: str(path.resolve(strict=True)) for name, path in paths.items()},
                "consumed_cpu_artifacts": {
                    name: {"size": len(value), "sha256": producer._sha(value)}
                    for name, value in consumed.items()},
                "head_provenance": [provenance[key] for key in TUPLES],
                "producer_artifact_links": head_links,
                "m1_stage0_binding": {
                    "bank": bank, "bank_directory": str(Path(bank_declaration["path"]).resolve()),
                    "required_directory": M1_FEATURE_DIRS.get(bank),
                    "slides": len(offsets), "decoded_bytes_sha256_equal_offsets_and_stage0": m1_checked,
                    "feature_path_is_bank_directory_slide_pkl": len(offsets)},
                "feature_consumption": [{"slide_id": feature["slide_id"], "status": "decoded_once",
                                         "asset_id": "feature:" + feature["slide_id"]}
                                        for feature in spec["features"]],
                "current_environment": {
                    "python": sys.version, "hostname": socket.gethostname(),
                    "native_environment": {name: os.environ.get(name) for name in producer.NATIVE_VARIABLES},
                    "head_consumer_runtime": [provenance[key]["consumer_runtime_json"] for key in TUPLES]},
                "started_at": started, "captured_at": producer._time(),
                "limitations": list(LIMITATIONS),
            }
            producer._json_output(output, "input_bindings.json", input_record, bindings)
            require(set(bindings) == artifact_names() and len(bindings) == 36, "packed artifact allowlist")
            output.check(bindings)
            marker = {
                "schema_version": 1, "run_id": spec["run_id"], "task": TASK, "scope": SCOPE,
                "encoder": bank, "bank": bank,
                "spec_sha256": spec_sha256, "outputs": dict(bindings),
                "verified_at": producer._time(), "limitations": list(LIMITATIONS),
            }
            producer._json_output(output, "COMPLETE.json", marker, bindings)
            output.check(bindings, marker=True)
        output.close()
        return marker
    except BaseException as exc:
        primary = exc
        try:
            output.fail(exc, [])
        except BaseException as additional:
            producer._note(exc, additional, "mosaic failure record/marker cleanup")
        raise
    finally:
        output.close(primary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--bank", required=True, choices=BANKS)
    args = parser.parse_args(argv)
    print(cpu.json_bytes(run(args.spec, args.spec_sha256, args.out, args.out_root,
                             bank=args.bank)).decode().rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
