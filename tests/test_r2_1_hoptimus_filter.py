"""Synthetic-only checks of the released R2.1 filtering manifest kernels."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib
import json
import random

import numpy as np
import pytest

from prosper.evaluation import r2_1_hoptimus_filter as module
from prosper.evaluation.r2_1_hoptimus_filter import (
    AnnotationRecord,
    ManifestValidationError,
    PatientFold,
    RetrievalRecord,
    build_patient_folds,
    build_training_manifests,
    five_percent_count,
    patient_fold_hash,
    physical_tile_id,
    training_tile_hash,
    validate_manifests,
)


def annotation(slide="slide", patient="patient", group="WT", label=1, index=0):
    x, y = index * 512, 1024
    return AnnotationRecord(
        physical_tile_id(slide, x, y), slide, patient, group,
        0, x, y, 512, label, index,
    )


def population():
    annotations = []
    for patient_index in range(8):
        patient = f"p{patient_index}"
        for label, count in ((1, 13), (3, 9), (4, 5), (5, 3)):
            slide = f"s{patient_index}_{label}"
            group = "WT" if label in (1, 3) else "NB"
            annotations.extend(annotation(slide, patient, group, label, i) for i in range(count))
    for patient, count in (("immune_a", 47), ("immune_b", 3)):
        annotations.extend(annotation(patient, patient, "NORMAL", 2, i) for i in range(count))
    annotations.append(annotation("misc_training", "p0", "MISC", 1))
    retrieval = [
        RetrievalRecord("s0_1", "p0", "WT"),
        RetrievalRecord("s0_4", "p0", "NB"),
        RetrievalRecord("query_only_a", "retrieval_only", "EWS"),
        RetrievalRecord("query_only_b", "retrieval_only", "OS"),
        RetrievalRecord("other_query", "other_patient", "RB"),
    ]
    return annotations, retrieval


def independent_sha(values):
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_hash_definitions_exact_unicode_and_no_encoder_or_budget():
    record = annotation("slide_é", "患者", "WT", 4, 7)
    assert record.tile_id == independent_sha(["slide_é", 0, 3584, 1024, 512])
    assert patient_fold_hash("患者") == independent_sha(["r2_1_patient_fold_v1", 1234, "患者"])
    assert training_tile_hash(record) == independent_sha([
        "r2_1_training_tile_v1", 1234, "患者", "slide_é", 0, 3584, 1024, 512, 4,
    ])
    assert physical_tile_id("001", 0, 0) != physical_tile_id("1", 0, 0)
    assert patient_fold_hash(" patient ") != patient_fold_hash("patient")


@pytest.mark.parametrize("field", [
    "level", "x", "y", "patch_size", "tissue_label", "h_optimus_1_feature_row",
])
@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1", None])
def test_annotation_integer_fields_reject_invalid_values(field, value):
    with pytest.raises(ManifestValidationError):
        replace(annotation(), **{field: value})


@pytest.mark.parametrize("field", ["tile_id", "slide_id", "patient_id", "diagnostic_group"])
@pytest.mark.parametrize("value", ["", "  ", None, 1, True])
def test_annotation_identifier_fields_are_exact_nonempty_strings(field, value):
    with pytest.raises(ManifestValidationError):
        replace(annotation(), **{field: value})


@pytest.mark.parametrize("field,value", [
    ("level", 1), ("patch_size", 256), ("tissue_label", 0), ("tissue_label", 6),
    ("tile_id", "0" * 64), ("tile_id", annotation().tile_id.upper()), ("x", 512),
])
def test_wrong_geometry_labels_and_physical_identity_rejected(field, value):
    with pytest.raises(ManifestValidationError):
        replace(annotation(), **{field: value})


@pytest.mark.parametrize("bad", [True, -1, 0.0, "0"])
def test_physical_tile_id_validates_coordinate_types(bad):
    with pytest.raises(ManifestValidationError):
        physical_tile_id("s", bad, 0)
    with pytest.raises(ManifestValidationError):
        physical_tile_id("s", 0, bad)


@pytest.mark.parametrize("field", ["slide_id", "patient_id", "diagnostic_group"])
@pytest.mark.parametrize("value", ["", " ", None, 7, False])
def test_retrieval_identifier_validation(field, value):
    with pytest.raises(ManifestValidationError):
        replace(RetrievalRecord("s", "p", "WT"), **{field: value})


@pytest.mark.parametrize("group", ["MISC", "NORMAL"])
def test_retrieval_exclusion_does_not_exclude_annotation_training(group):
    record = annotation(group=group)
    assert validate_manifests([record], [])[0] == (record,)
    with pytest.raises(ManifestValidationError, match="retrieval excludes"):
        RetrievalRecord("s", "p", group)


def test_duplicate_tiles_source_indices_and_retrieval_slides_refused():
    first = annotation(index=0)
    with pytest.raises(ManifestValidationError, match="duplicate annotation"):
        validate_manifests([first, first], [])
    other = replace(annotation(index=1), h_optimus_1_feature_row=0)
    with pytest.raises(ManifestValidationError, match="source index"):
        validate_manifests([first, other], [])
    query = RetrievalRecord("q", "p", "WT")
    with pytest.raises(ManifestValidationError, match="duplicate retrieval"):
        validate_manifests([], [query, query])
    assert len(validate_manifests([first, annotation(slide="another")], [])[0]) == 2


@pytest.mark.parametrize("change", [{"patient_id": "other"}, {"diagnostic_group": "NB"}])
def test_slide_metadata_consistency_within_annotation_and_across_populations(change):
    with pytest.raises(ManifestValidationError, match="metadata"):
        validate_manifests([annotation(), replace(annotation(index=1), **change)], [])
    query = replace(RetrievalRecord("slide", "patient", "WT"), **change)
    with pytest.raises(ManifestValidationError, match="metadata"):
        validate_manifests([annotation()], [query])


def test_nonrecord_inputs_rejected():
    with pytest.raises(ManifestValidationError, match="AnnotationRecord"):
        validate_manifests([{}], [])
    with pytest.raises(ManifestValidationError, match="RetrievalRecord"):
        validate_manifests([], [{}])
    with pytest.raises(ManifestValidationError, match="AnnotationRecord"):
        training_tile_hash({})


@pytest.mark.parametrize("annotations,retrieval", [(None, []), ([], None), (3, [])])
def test_noniterable_populations_have_specific_validation_error(annotations, retrieval):
    with pytest.raises(ManifestValidationError, match="must be iterable"):
        validate_manifests(annotations, retrieval)


def test_fold_assignment_matches_independent_hash_balancing():
    annotations, retrieval = population()
    folds = build_patient_folds(annotations, retrieval)
    patients = {row.patient_id for row in annotations + retrieval}
    order = sorted(patients, key=lambda patient: (
        independent_sha(["r2_1_patient_fold_v1", 1234, patient]), patient,
    ))
    immune = [patient for patient in order if patient in {"immune_a", "immune_b"}]
    expected = dict(zip(immune, (0, 1)))
    counts = [1, 1, 0, 0, 0]
    for patient in order:
        if patient not in expected:
            fold = counts.index(min(counts))
            expected[patient] = fold
            counts[fold] += 1
    assert [row.patient_id for row in folds.patients] == order
    assert {row.patient_id: row.fold for row in folds.patients} == expected
    assert [row.patient_count for row in folds.support] == counts
    assert sorted(expected[patient] for patient in immune) == [0, 1]
    assert max(counts) - min(counts) <= 1


def test_fold_support_counts_and_cross_label_whole_patient_cohesion():
    annotations, retrieval = population()
    folds = build_patient_folds(annotations, retrieval)
    mapping = {row.patient_id: row.fold for row in folds.patients}
    assert len(folds.patients) == len(set(mapping)) == 12
    assert sum(row.annotation_tile_count for row in folds.support) == len(annotations)
    assert sum(row.retrieval_slide_count for row in folds.support) == len(retrieval)
    for support in folds.support:
        rows = [row for row in annotations if mapping[row.patient_id] == support.fold]
        assert support.annotation_patient_count == len({row.patient_id for row in rows})
        assert support.annotation_slide_count == len({row.slide_id for row in rows})
        assert tuple(row.count for row in support.class_counts) == tuple(
            sum(row.tissue_label == label for row in rows) for label in range(1, 6)
        )
    assert len({mapping[row.patient_id] for row in annotations if row.patient_id == "p0"}) == 1
    assert {row.diagnostic_group for row in annotations if row.patient_id == "p0"} == {
        "WT", "NB", "MISC",
    }


@pytest.mark.parametrize("immune_count", [0, 1, 3])
def test_exact_two_immune_patients_required(immune_count):
    rows = [annotation(f"i{i}", f"i{i}", label=2) for i in range(immune_count)]
    with pytest.raises(ManifestValidationError, match="exactly two"):
        build_patient_folds(rows, [])


def test_immune_tiles_across_multiple_slides_still_count_patients():
    annotations, retrieval = population()
    annotations.append(annotation("second_immune_slide", "immune_a", label=2))
    folds = build_patient_folds(annotations, retrieval)
    assert {row.fold for row in folds.patients if row.patient_id.startswith("immune")} == {0, 1}


@pytest.mark.parametrize("count,expected", [
    (1, 1), (9, 1), (10, 1), (19, 1), (20, 1), (30, 2), (47, 2),
    (50, 2), (70, 4), (90, 4), (3421, 171), (3468, 173),
])
def test_five_percent_rounding_and_minimum(count, expected):
    assert five_percent_count(count) == expected


@pytest.mark.parametrize("count", [0, -1, True, False, 47.0, "47", None])
def test_five_percent_count_rejects_absent_or_invalid_support(count):
    with pytest.raises(ManifestValidationError):
        five_percent_count(count)


def test_input_order_invariance_generators_and_no_input_mutation():
    annotations, retrieval = population()
    originals = list(annotations), list(retrieval)
    folds = build_patient_folds(annotations, retrieval)
    training = build_training_manifests(annotations, retrieval, folds)
    shuffled_ann, shuffled_ret = list(annotations), list(retrieval)
    random.Random(19).shuffle(shuffled_ann)
    random.Random(23).shuffle(shuffled_ret)
    assert build_patient_folds(iter(shuffled_ann), iter(shuffled_ret)) == folds
    assert build_training_manifests(iter(shuffled_ann), iter(shuffled_ret), folds) == training
    assert (annotations, retrieval) == originals
    with pytest.raises(FrozenInstanceError):
        annotations[0].x = 999


def test_training_nesting_global_order_full_pool_and_no_heldout_leakage():
    annotations, retrieval = population()
    folds = build_patient_folds(annotations, retrieval)
    mapping = {row.patient_id: row.fold for row in folds.patients}
    manifests = build_training_manifests(annotations, retrieval, folds)
    assert [(row.held_out_fold, row.budget_percent) for row in manifests] == [
        (fold, budget) for fold in range(5) for budget in (5, 100)
    ]
    for primary, full in zip(manifests[::2], manifests[1::2]):
        expected = sorted(
            (row for row in annotations if mapping[row.patient_id] != full.held_out_fold),
            key=lambda row: (training_tile_hash(row), row.physical_key),
        )
        assert full.tiles == tuple(expected)
        primary_ids = {row.tile_id for row in primary.tiles}
        assert primary.tiles == tuple(row for row in full.tiles if row.tile_id in primary_ids)
        assert all(mapping[row.patient_id] != full.held_out_fold for row in primary.tiles)
        assert {row.tissue_label for row in primary.tiles} == {1, 2, 3, 4, 5}
        for stat in primary.class_stats:
            candidates = [row for row in full.tiles if row.tissue_label == stat.tissue_label]
            selected = [row for row in primary.tiles if row.tissue_label == stat.tissue_label]
            assert selected == candidates[:five_percent_count(len(candidates))]
    fold = mapping["immune_b"]
    primary = next(row for row in manifests if (row.held_out_fold, row.budget_percent) == (fold, 5))
    immune = next(stat for stat in primary.class_stats if stat.tissue_label == 2)
    assert (immune.pool_count, immune.selected_count) == (47, 2)
    for full in manifests[1::2]:
        assert any(row.diagnostic_group == "MISC" for row in full.tiles) == (
            mapping["p0"] != full.held_out_fold
        )
        assert any(row.diagnostic_group == "NORMAL" for row in full.tiles)


def test_balanced_weights_and_fractions_use_selected_training_labels_only():
    annotations, retrieval = population()
    folds = build_patient_folds(annotations, retrieval)
    manifests = build_training_manifests(annotations, retrieval, folds)
    weights_differ = False
    for primary, full in zip(manifests[::2], manifests[1::2]):
        for manifest in (primary, full):
            assert manifest.selected_count == len(manifest.tiles)
            assert manifest.selected_fraction == len(manifest.tiles) / manifest.pool_count
            for stat in manifest.class_stats:
                selected = sum(row.tissue_label == stat.tissue_label for row in manifest.tiles)
                assert stat.selected_count == selected
                assert stat.fraction == selected / stat.pool_count
                assert stat.balanced_weight == len(manifest.tiles) / (5 * selected)
        weights_differ |= any(a.balanced_weight != b.balanced_weight
                              for a, b in zip(primary.class_stats, full.class_stats))
        assert all(stat.fraction == 1.0 for stat in full.class_stats)
    assert weights_differ


def test_heldout_nonimmune_labels_do_not_enter_selection_or_weights():
    annotations, retrieval = population()
    folds = build_patient_folds(annotations, retrieval)
    held_out = next(row.fold for row in folds.patients if row.patient_id == "p0")
    original = build_training_manifests(annotations, retrieval, folds)
    changed = [
        replace(row, tissue_label=3) if row.patient_id == "p0" and row.tissue_label == 1 else row
        for row in annotations
    ]
    changed_folds = build_patient_folds(changed, retrieval)
    assert changed_folds.patients == folds.patients
    revised = build_training_manifests(changed, retrieval, changed_folds)
    assert [row for row in original if row.held_out_fold == held_out] == [
        row for row in revised if row.held_out_fold == held_out
    ]


def test_training_absent_class_in_a_heldout_fold_is_an_error():
    annotations, retrieval = population()
    annotations = [row for row in annotations if row.tissue_label != 5 or row.patient_id == "p0"]
    folds = build_patient_folds(annotations, retrieval)
    with pytest.raises(ManifestValidationError, match="training pool lacks tissue classes"):
        build_training_manifests(annotations, retrieval, folds)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "wrong_fold", "wrong_support", "reordered"]
)
def test_training_rejects_altered_or_incomplete_common_fold_manifest(mutation):
    annotations, retrieval = population()
    folds = build_patient_folds(annotations, retrieval)
    if mutation == "missing":
        altered = replace(folds, patients=folds.patients[:-1])
    elif mutation == "extra":
        altered = replace(folds, patients=folds.patients + (PatientFold("extra", 0),))
    elif mutation == "wrong_fold":
        first = folds.patients[0]
        altered = replace(
            folds, patients=(replace(first, fold=(first.fold + 1) % 5),) + folds.patients[1:]
        )
    elif mutation == "wrong_support":
        altered = replace(
            folds, support=(replace(folds.support[0], patient_count=99),) + folds.support[1:]
        )
    else:
        altered = replace(folds, patients=tuple(reversed(folds.patients)))
    with pytest.raises(ManifestValidationError, match="prescribed common patient map"):
        build_training_manifests(annotations, retrieval, altered)


def test_hash_collision_ties_follow_exact_physical_key(monkeypatch):
    annotations, retrieval = population()
    folds = build_patient_folds(annotations, retrieval)
    monkeypatch.setattr(module, "training_tile_hash", lambda row: "0" * 64)
    manifests = build_training_manifests(annotations, retrieval, folds)
    for manifest in manifests:
        assert list(manifest.tiles) == sorted(manifest.tiles, key=lambda row: row.physical_key)
    monkeypatch.setattr(module, "patient_fold_hash", lambda patient: "0" * 64)
    tied = build_patient_folds(annotations, retrieval)
    assert [row.patient_id for row in tied.patients] == sorted(
        row.patient_id for row in tied.patients
    )
    assert next(row.fold for row in tied.patients if row.patient_id == "immune_a") == 0


def test_import_and_functions_preserve_global_rng_and_do_not_open_files(monkeypatch):
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    annotations, retrieval = population()

    def forbidden(*args, **kwargs):
        raise AssertionError("pure manifest module must not perform file IO")

    monkeypatch.setattr("builtins.open", forbidden)
    original_namespace = dict(vars(module))
    try:
        importlib.reload(module)
        rows = [module.AnnotationRecord(**vars(row)) for row in annotations]
        queries = [module.RetrievalRecord(**vars(row)) for row in retrieval]
        folds = module.build_patient_folds(rows, queries)
        module.build_training_manifests(rows, queries, folds)
    finally:
        vars(module).update(original_namespace)
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    assert np.array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]
    assert random.getstate() == python_state
