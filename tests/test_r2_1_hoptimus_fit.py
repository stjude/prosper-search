"""Tiny synthetic fixtures for the exact array-only fitting release."""
from dataclasses import FrozenInstanceError, replace
from functools import wraps
import hashlib
import importlib
import json
import random
import weakref
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier

from prosper.evaluation import r2_1_hoptimus_fit as module
from prosper.evaluation.r2_1_hoptimus_filter import (
    AnnotationRecord, ClassSelection, TrainingManifest, physical_tile_id,
    training_tile_hash,
)


EXPECTED = {
    "loss": "log_loss", "penalty": "l2", "alpha": 0.001,
    "max_iter": 1000, "tol": 0.001, "random_state": 1234,
    "fit_intercept": True, "shuffle": True, "learning_rate": "optimal",
    "early_stopping": False, "n_iter_no_change": 5,
    "warm_start": False, "average": False, "n_jobs": 1, "verbose": 1,
    "l1_ratio": 0.15, "epsilon": 0.1, "eta0": 0.0, "power_t": 0.5,
    "validation_fraction": 0.1,
}


def record(slide="s", patient="p", label=1, row=0, *, x=None, y=512, group="WT"):
    x = row * 512 if x is None else x
    return AnnotationRecord(
        physical_tile_id(slide, x, y), slide, patient, group,
        0, x, y, 512, label, row,
    )


def manifest(budget=100, counts=(2, 2, 2, 2, 2), pools=None):
    pools = counts if pools is None else pools
    rows = tuple(sorted(
        (record(f"s{label}", "shared_patient", label, i,
                group="MISC" if label == 5 else "WT")
         for label, count in enumerate(counts, 1) for i in range(count)),
        key=lambda row: (training_tile_hash(row), row.physical_key),
    ))
    n = sum(counts)
    stats = tuple(ClassSelection(label, pool, selected, selected / pool,
                                 n / (5 * selected))
                  for label, pool, selected in zip(range(1, 6), pools, counts))
    return TrainingManifest(0, budget, rows, stats, sum(pools), n, n / sum(pools))


def features_for(man):
    output = np.zeros((len(man.tiles), 1536), dtype=np.float32)
    for position, row in enumerate(man.tiles):
        output[position, row.tissue_label - 1] = np.float32(0.1)
        output[position, 15] = np.float32((position + 1) / 10000)
    return output


def raw_block(slide="s", rows=3, dtype=np.float32, coord_dtype=np.int64):
    features = np.repeat(np.arange(rows, dtype=dtype)[:, None], 1536, axis=1)
    coordinates = np.array([(i * 512, 512) for i in range(rows)], dtype=coord_dtype)
    return slide, features, coordinates


def assert_rng_equal(before, after):
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.fixture
def head():
    man = manifest()
    return module.fit_head(features_for(man), man, {"shared_patient": 1})


def test_import_does_not_fit_create_outputs_or_change_rng(monkeypatch, tmp_path):
    monkeypatch.setattr(SGDClassifier, "fit",
                        lambda *a, **k: pytest.fail("fit during import"))
    before_files = tuple(tmp_path.iterdir())
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    importlib.reload(module)
    assert tuple(tmp_path.iterdir()) == before_files
    assert random.getstate() == python_rng
    assert_rng_equal(numpy_rng, np.random.get_state())


def test_assembly_reorders_blocks_and_rows_without_changing_source_or_record_order():
    rows = (record("b", row=0), record("a", row=2), record("a", row=0, label=3))
    a, b = raw_block("a"), raw_block("b")
    snapshots = [(block[1].copy(), block[2].copy()) for block in (a, b)]
    output, labels = module.assemble_selected_features(rows, iter((a, b)))
    assert output.shape == (3, 1536) and output.dtype == np.float32
    assert output.flags.c_contiguous and output.flags.owndata
    assert np.array_equal(output[:, 0], [0, 2, 0])
    assert labels.dtype == np.int64 and labels.tolist() == [1, 1, 3]
    for block, (old_features, old_coords) in zip((a, b), snapshots):
        assert np.array_equal(block[1], old_features)
        assert np.array_equal(block[2], old_coords)
        assert not np.shares_memory(output, block[1])


def test_sequential_assembly_releases_previous_raw_block_before_next_acquisition():
    rows = (record("a"), record("b"))
    previous = []

    class Blocks:
        def __iter__(self):
            return self

        def __next__(self):
            if len(previous) == 2:
                assert previous[-1]() is None
                raise StopIteration
            if previous:
                assert previous[-1]() is None
            block = raw_block("a" if not previous else "b", rows=1)
            previous.append(weakref.ref(block[1]))
            return block

    module.assemble_selected_features(rows, Blocks())
    assert all(reference() is None for reference in previous)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint16, np.uint64])
def test_signed_and_unsigned_coordinates(dtype):
    block = ("s", np.zeros((1, 1536), np.float32), np.array([[0, 0]], dtype=dtype))
    output, labels = module.assemble_selected_features((record(y=0),), (block,))
    assert output.shape == (1, 1536) and labels.tolist() == [1]


def test_large_uint64_coordinate_joins_exactly_without_float_coercion():
    x = 2**63 + 1
    coordinates = np.array([[x, 512]], dtype=np.uint64)
    block = ("s", np.zeros((1, 1536), dtype=np.float32), coordinates)
    module.assemble_selected_features((record(x=x),), (block,))
    with pytest.raises(module.HeadValidationError, match="coordinate mismatch"):
        module.assemble_selected_features((record(x=x - 1),), (block,))


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize("bad", [
    None, [[0, 512]], np.array([[0, 512]], dtype=np.float32),
    np.array([[False, True]]), np.array([[0, 512]], dtype=object),
    np.ma.array([[0, 512]], mask=False),
    np.array([[0, 512]]).view(ArraySubclass),
    np.array([0, 512]), np.array([[0, 512, 0]]),
    np.array([[-1, 512]]), np.array([[0, 512], [0, 512]]),
])
def test_coordinates_rejected_before_coercion(bad):
    raw = np.zeros((1, 1536), dtype=np.float32)
    with pytest.raises(module.HeadValidationError):
        module.assemble_selected_features((record(),), (("s", raw, bad),))


@pytest.mark.parametrize("bad", [
    None, [[0] * 1536], np.zeros((1, 1536), dtype=np.float64),
    np.zeros((1, 1536), dtype=object), np.zeros((1, 1536), dtype=bool),
    np.ma.zeros((1, 1536), dtype=np.float32),
    np.zeros((1, 1536), dtype=np.float32).view(ArraySubclass),
    np.zeros(1536, dtype=np.float32), np.zeros((1, 1535), dtype=np.float32),
    np.full((1, 1536), np.nan, dtype=np.float32),
    np.full((1, 1536), np.inf, dtype=np.float32),
])
def test_assembly_feature_validation(bad):
    with pytest.raises(module.HeadValidationError):
        module.assemble_selected_features((record(),),
                                          (("s", bad, np.array([[0, 512]])),))


@pytest.mark.parametrize("blocks", [
    (), (raw_block("unknown"),), (raw_block(), raw_block()),
    (("s", np.zeros((0, 1536), np.float32), np.empty((0, 2), np.int64)),),
    (("s", np.zeros((1, 1536), np.float32), np.array([[512, 512]])),),
    (raw_block()[1:],),
])
def test_unknown_duplicate_missing_out_of_bounds_and_misjoined_blocks(blocks):
    with pytest.raises(module.HeadValidationError):
        module.assemble_selected_features((record(),), blocks)


@pytest.mark.parametrize("rows", [
    (), [record()], (record(), record()),
    (record(), record(x=512)),
    (record(), record(row=1, patient="other")),
    (record(), record(row=1, group="NB")),
])
def test_record_shape_duplicates_and_joint_slide_metadata(rows):
    with pytest.raises(module.HeadValidationError):
        module.assemble_selected_features(rows, (raw_block(),))


def test_raw_noncontiguous_blocks_are_scattered_into_contiguous_output():
    features = np.zeros((3, 3072), dtype=np.float32)[:, ::2]
    assert not features.flags.c_contiguous
    output, _ = module.assemble_selected_features(
        (record(),), (("s", features, raw_block()[2]),))
    assert output.flags.c_contiguous


@pytest.mark.parametrize("budget,counts,pools", [
    (100, (2, 2, 2, 2, 2), None),
    (5, (1, 2, 1, 1, 1), (1, 47, 1, 1, 1)),
])
def test_real_tiny_fit_preserves_rng_inputs_order_weights_and_support(budget, counts, pools):
    man = manifest(budget, counts, pools)
    features = features_for(man)
    original = features.copy()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    fitted = module.fit_head(features, man, {"shared_patient": 1})
    result = module.predict_head(fitted, features)
    assert np.array_equal(features, original)
    assert random.getstate() == python_rng
    assert_rng_equal(numpy_rng, np.random.get_state())
    assert result.probabilities.dtype == np.float32 and result.labels.dtype == np.int64
    assert result.probabilities.shape == (len(man.tiles), 5)
    assert np.isfinite(result.probabilities).all()
    assert result.labels.tolist() == fitted.estimator.classes_[
        result.probabilities.argmax(axis=1)].tolist()
    report = fitted.diagnostics
    assert report["selected_patients"] == 1 and report["selected_slides"] == 5
    assert report["ordered_tile_ids_sha256"] == hashlib.sha256(
        json.dumps([row.tile_id for row in man.tiles], ensure_ascii=True,
                   separators=(",", ":")).encode()).hexdigest()
    assert report["feature_sha256"] == hashlib.sha256(features).hexdigest()
    assert report["n_iter"] == fitted.estimator.n_iter_
    assert 1 <= report["n_iter"] <= 1000
    assert report["n_iter_semantics"] == "maximum_across_five_one_vs_rest_fits"
    assert report["t"] == fitted.estimator.t_
    assert report["classes"] == [1, 2, 3, 4, 5]
    assert report["finite_state"] and report["coefficient_shape"] == [5, 1536]
    assert report["intercept_shape"] == [5]
    assert report["runtime"]["distribution_versions"] == {
        "scikit-learn": "1.7.0", "numpy": "2.2.4", "scipy": "1.16.0"}
    assert all(item["num_threads"] == 1 for item in report["runtime"]["native_pools"])
    assert len(report["fitted_state_sha256"]) == 64
    json.dumps(report, allow_nan=False)
    assert report["class_support"][1]["selected_tiles"] == counts[1]
    assert report["class_support"][1]["selected_patients"] == 1
    assert report["class_support"][1]["balanced_weight"] == sum(counts) / (5 * counts[1])


def test_constructor_and_single_fit_call_use_class_weight_not_sample_weight(monkeypatch):
    original_init, original_fit = SGDClassifier.__init__, SGDClassifier.fit
    initializers, fit_calls = [], []

    @wraps(original_init)
    def init(self, *args, **kwargs):
        initializers.append(kwargs.copy())
        original_init(self, *args, **kwargs)

    def fit(self, features, labels, **kwargs):
        fit_calls.append((self, features, labels.copy(), kwargs.copy()))
        return original_fit(self, features, labels, **kwargs)

    monkeypatch.setattr(SGDClassifier, "__init__", init)
    monkeypatch.setattr(SGDClassifier, "fit", fit)
    man = manifest(5, (1, 2, 1, 1, 1), (1, 47, 1, 1, 1))
    features = features_for(man)
    first = module.fit_head(features, man, {"shared_patient": 1})
    second = module.fit_head(features.copy(), man, {"shared_patient": 1})
    weights = {stat.tissue_label: stat.balanced_weight for stat in man.class_stats}
    assert initializers == [{**EXPECTED, "class_weight": weights}] * 2
    assert len(fit_calls) == 2 and first.estimator is not second.estimator
    for estimator, matrix, labels, kwargs in fit_calls:
        assert kwargs == {"sample_weight": None}
        assert labels.dtype == np.int64
        assert labels.tolist() == [row.tissue_label for row in man.tiles]
        assert np.array_equal(matrix, features)
        assert estimator.get_params() == {**EXPECTED, "class_weight": weights}


@pytest.mark.parametrize("field,bad", [
    ("held_out_fold", True), ("held_out_fold", 0.0), ("held_out_fold", -1),
    ("held_out_fold", 5), ("budget_percent", True), ("budget_percent", 5.0),
    ("budget_percent", 20), ("pool_count", True), ("pool_count", 10.0),
    ("pool_count", 0), ("selected_count", True), ("selected_count", 10.0),
    ("selected_count", 9), ("selected_fraction", True),
    ("selected_fraction", "1.0"), ("selected_fraction", 1),
    ("selected_fraction", np.nan), ("selected_fraction", 0.5),
])
def test_manifest_scalar_validation_before_fit(field, bad, monkeypatch):
    man = replace(manifest(), **{field: bad})
    monkeypatch.setattr(SGDClassifier, "fit",
                        lambda *a, **k: pytest.fail("invalid manifest fitted"))
    with pytest.raises(module.HeadValidationError):
        module.fit_head(features_for(man), man, {"shared_patient": 1})


@pytest.mark.parametrize("field,bad", [
    ("tissue_label", True), ("tissue_label", 1.0), ("tissue_label", 2),
    ("pool_count", True), ("pool_count", 2.0), ("pool_count", 0),
    ("selected_count", True), ("selected_count", 2.0), ("selected_count", 0),
    ("selected_count", 1), ("fraction", True), ("fraction", 1),
    ("fraction", "1.0"), ("fraction", np.inf), ("fraction", 0.5),
    ("balanced_weight", True), ("balanced_weight", 1),
    ("balanced_weight", "1.0"), ("balanced_weight", np.nan),
    ("balanced_weight", 0.5),
])
def test_class_count_fraction_and_weight_inputs_are_not_coerced(field, bad, monkeypatch):
    man = manifest()
    man = replace(man, class_stats=(replace(man.class_stats[0], **{field: bad}),
                                   *man.class_stats[1:]))
    monkeypatch.setattr(SGDClassifier, "fit", lambda *a, **k: pytest.fail("fit"))
    with pytest.raises(module.HeadValidationError):
        module.fit_head(features_for(man), man, {"shared_patient": 1})


@pytest.mark.parametrize("mapping", [
    {}, None, {"shared_patient": True}, {"shared_patient": 1.0},
    {"shared_patient": "1"}, {"shared_patient": -1}, {"shared_patient": 5},
    {"shared_patient": 0}, {"wrong_patient": 1}, {"shared_patient": 1, " ": 2},
])
def test_patient_mapping_and_whole_patient_leakage(mapping, monkeypatch):
    man = manifest()
    monkeypatch.setattr(SGDClassifier, "fit", lambda *a, **k: pytest.fail("fit"))
    with pytest.raises(module.HeadValidationError):
        module.fit_head(features_for(man), man, mapping)


def test_fit_duplicate_metadata_absent_class_and_order_rejection(monkeypatch):
    man = manifest()
    duplicate = replace(man, tiles=(man.tiles[0], *man.tiles[:-1]))
    bad_patient_rows = list(man.tiles)
    target = next(i for i, row in enumerate(man.tiles)
                  if row.slide_id == man.tiles[0].slide_id and i != 0)
    bad_patient_rows[target] = replace(man.tiles[target], patient_id="other")
    bad_group_rows = list(man.tiles)
    bad_group_rows[target] = replace(man.tiles[target], diagnostic_group="NB")
    no_class = replace(man, tiles=tuple(row for row in man.tiles if row.tissue_label != 5))
    reordered = replace(man, tiles=man.tiles[::-1])
    bad_stats = replace(man, class_stats=man.class_stats[::-1])
    incomplete_stats = replace(man, class_stats=man.class_stats[:-1])
    monkeypatch.setattr(SGDClassifier, "fit", lambda *a, **k: pytest.fail("fit"))
    for bad in (duplicate, replace(man, tiles=tuple(bad_patient_rows)),
                replace(man, tiles=tuple(bad_group_rows)), no_class, reordered,
                bad_stats, incomplete_stats):
        with pytest.raises(module.HeadValidationError):
            module.fit_head(features_for(bad), bad, {"shared_patient": 1, "other": 2})


def test_fit_source_indices_are_per_slide_and_same_slide_duplicates_fail(monkeypatch):
    man = manifest()
    assert sum(row.h_optimus_1_feature_row == 0 for row in man.tiles) == 5
    changed = tuple(replace(row, h_optimus_1_feature_row=0) if row.tissue_label == 1
                    else row for row in man.tiles)
    monkeypatch.setattr(SGDClassifier, "fit", lambda *a, **k: pytest.fail("fit"))
    with pytest.raises(module.HeadValidationError, match="source index"):
        module.fit_head(features_for(man), replace(man, tiles=changed),
                        {"shared_patient": 1})


def test_rounding_and_pool_total_validation(monkeypatch):
    man = manifest(5, (1, 2, 1, 1, 1), (1, 47, 1, 1, 1))
    bad_stat = replace(man.class_stats[1], pool_count=10, fraction=0.2)
    bad = replace(man, class_stats=(man.class_stats[0], bad_stat, *man.class_stats[2:]),
                  pool_count=14, selected_fraction=6 / 14)
    other = replace(man, pool_count=man.pool_count + 1,
                    selected_fraction=man.selected_count / (man.pool_count + 1))
    monkeypatch.setattr(SGDClassifier, "fit", lambda *a, **k: pytest.fail("fit"))
    for invalid in (bad, other):
        with pytest.raises(module.HeadValidationError):
            module.fit_head(features_for(invalid), invalid, {"shared_patient": 1})


@pytest.mark.parametrize("pool,selected", [(1, 1), (10, 1), (20, 1), (30, 2),
                                          (47, 2), (50, 2), (70, 4)])
def test_accepted_minimum_ties_to_even_and_selected_weights(pool, selected):
    man = manifest(5, (1, selected, 1, 1, 1), (1, pool, 1, 1, 1))
    weights = module._validate_manifest(man, {"shared_patient": 1})
    assert weights[2] == (selected + 4) / (5 * selected)


def test_unseen_pool_membership_and_bare_matrix_identity_limits_are_explicit(head):
    # Both unseen pools yield exactly the same selected rows and weights.
    first = manifest(5, (1, 1, 1, 1, 1), (10, 10, 10, 10, 10))
    second = replace(first, class_stats=tuple(
        replace(stat, pool_count=20, fraction=1 / 20) for stat in first.class_stats),
        pool_count=100, selected_fraction=5 / 100)
    assert module._validate_manifest(first, {"shared_patient": 1}) == \
        module._validate_manifest(second, {"shared_patient": 1})
    assert "selected_manifest_cannot_prove_unseen_pool_membership" in \
        head.diagnostics["limits"]
    assert "bare_matrix_cannot_prove_row_identity_or_file_provenance" in \
        head.diagnostics["limits"]


@pytest.mark.parametrize("bad", [
    None, np.zeros((10, 1536), dtype=np.float64),
    np.zeros((10, 1535), dtype=np.float32),
    np.zeros((9, 1536), dtype=np.float32),
    np.zeros((10, 3072), dtype=np.float32)[:, ::2],
    np.ma.zeros((10, 1536), dtype=np.float32),
    np.zeros((10, 1536), dtype=np.float32).view(ArraySubclass),
    np.full((10, 1536), np.nan, dtype=np.float32),
])
def test_fit_matrix_validation(bad, monkeypatch):
    monkeypatch.setattr(SGDClassifier, "fit", lambda *a, **k: pytest.fail("fit"))
    with pytest.raises(module.HeadValidationError):
        module.fit_head(bad, manifest(), {"shared_patient": 1})


def test_version_profile_and_native_pools_fail_without_fit(monkeypatch):
    man = manifest()
    original_version = module.importlib.metadata.version
    monkeypatch.setattr(SGDClassifier, "fit", lambda *a, **k: pytest.fail("fit"))
    monkeypatch.setattr(module.importlib.metadata, "version",
                        lambda name: "0.0.0" if name == "scikit-learn"
                        else original_version(name))
    with pytest.raises(module.HeadValidationError, match="versions"):
        module.fit_head(features_for(man), man, {"shared_patient": 1})
    monkeypatch.setattr(module.importlib.metadata, "version", original_version)
    monkeypatch.setattr(module, "threadpool_info", lambda: [{"num_threads": 2}])
    with pytest.raises(module.HeadValidationError, match="native pools"):
        module.fit_head(features_for(man), man, {"shared_patient": 1})


def test_mock_convergence_warning_retains_honest_public_max_ovr_head(monkeypatch):
    original_fit = SGDClassifier.fit
    calls = []

    def fit(self, *args, **kwargs):
        calls.append(self)
        original_fit(self, *args, **kwargs)
        warnings.warn("synthetic iteration-limit warning", ConvergenceWarning)
        return self

    monkeypatch.setattr(SGDClassifier, "fit", fit)
    man = manifest()
    fitted = module.fit_head(features_for(man), man, {"shared_patient": 1})
    report = fitted.diagnostics
    assert len(calls) == 1 and report["iteration_limit_warning"]
    assert any(item["message"] == "synthetic iteration-limit warning"
               for item in report["warnings"])
    assert report["n_iter"] == fitted.estimator.n_iter_
    assert report["convergence_status"] == "iteration_limit_warning_retained"
    assert module.predict_head(fitted, features_for(man)).labels.shape == (10,)


@pytest.mark.parametrize("category", [UserWarning, RuntimeWarning, DeprecationWarning])
def test_unexpected_fit_warnings_fail_without_retry(category, monkeypatch):
    original_fit = SGDClassifier.fit
    calls = []

    def fit(self, *args, **kwargs):
        calls.append(self)
        original_fit(self, *args, **kwargs)
        warnings.warn("synthetic unexpected warning", category)
        return self

    monkeypatch.setattr(SGDClassifier, "fit", fit)
    man = manifest()
    with pytest.raises(module.HeadValidationError, match="unexpected warnings"):
        module.fit_head(features_for(man), man, {"shared_patient": 1})
    assert len(calls) == 1


def test_failed_fit_retains_context_all_warnings_and_original_error_without_retry(monkeypatch):
    error = ValueError("synthetic solver failure")
    calls = []

    def fit(self, *args, **kwargs):
        calls.append(self)
        warnings.warn("first convergence warning", ConvergenceWarning)
        warnings.warn("second unexpected warning", RuntimeWarning)
        raise error

    monkeypatch.setattr(SGDClassifier, "fit", fit)
    man = manifest()
    with pytest.raises(ValueError) as caught:
        module.fit_head(features_for(man), man, {"shared_patient": 1})
    assert caught.value is error and len(calls) == 1
    notes = "\n".join(error.__notes__)
    assert "fold=0, budget=100" in notes and "no retry" in notes
    assert "first convergence warning" in notes and "second unexpected warning" in notes


def test_invalid_fitted_state_fails_at_fit_return(monkeypatch):
    original_fit = SGDClassifier.fit

    def fit(self, *args, **kwargs):
        original_fit(self, *args, **kwargs)
        self.coef_[0, 0] = np.nan
        warnings.warn("synthetic convergence then invalid state", ConvergenceWarning)
        return self

    monkeypatch.setattr(SGDClassifier, "fit", fit)
    man = manifest()
    with pytest.raises(module.HeadValidationError, match="coef_") as caught:
        module.fit_head(features_for(man), man, {"shared_patient": 1})
    assert "synthetic convergence then invalid state" in "\n".join(caught.value.__notes__)
    assert "fold=0, budget=100" in "\n".join(caught.value.__notes__)


def test_diagnostics_are_detached_immutable_snapshot(head):
    report = head.diagnostics
    duplicate = module.FittedHead(head.estimator, report)
    report["parameters"]["class_weight"]["1"] = 999.0
    report["class_support"][0]["selected_patients"] = 999
    report["fitted_state_sha256"] = "0" * 64
    assert duplicate.diagnostics == head.diagnostics
    assert duplicate.diagnostics["class_support"][0]["selected_patients"] == 1
    with pytest.raises(FrozenInstanceError):
        head.estimator = SGDClassifier()
    assert module.predict_head(duplicate, np.empty((0, 1536), np.float32)).labels.size == 0


@pytest.mark.parametrize("name,value", [
    ("loss", "modified_huber"), ("alpha", 0.1), ("average", 0),
    ("n_jobs", True), ("random_state", None), ("tol", 0.0001),
])
@pytest.mark.parametrize("rows", [0, 1])
def test_changed_full_parameters_fail_before_even_empty_prediction(head, name, value,
                                                                  rows, monkeypatch):
    setattr(head.estimator, name, value)
    monkeypatch.setattr(head.estimator, "predict_proba",
                        lambda *a: pytest.fail("prediction on changed state"))
    with pytest.raises(module.HeadValidationError):
        module.predict_head(head, np.zeros((rows, 1536), np.float32))


@pytest.mark.parametrize("mutation", [
    lambda e: e.coef_.__setitem__((0, 0), e.coef_[0, 0] + np.float32(1)),
    lambda e: e.intercept_.__setitem__(0, e.intercept_[0] + np.float32(1)),
    lambda e: setattr(e, "n_iter_", e.n_iter_ + 1),
    lambda e: setattr(e, "t_", e.t_ + 0.5),
    lambda e: setattr(e, "n_features_in_", 1536.0),
    lambda e: setattr(e, "classes_", e.classes_[::-1].copy()),
    lambda e: setattr(e, "coef_", e.coef_.astype(np.float64)),
    lambda e: setattr(e, "intercept_", np.zeros(4, np.float32)),
    lambda e: e.coef_.__setitem__((0, 0), np.inf),
    lambda e: e.class_weight.__setitem__(1, 2.0),
    lambda e: setattr(e, "class_weight", {True: e.class_weight[1],
                                        **{i: e.class_weight[i] for i in range(2, 6)}}),
])
@pytest.mark.parametrize("rows", [0, 1])
def test_mutable_state_fingerprint_and_validation_before_predictions(head, mutation,
                                                                   rows, monkeypatch):
    mutation(head.estimator)
    monkeypatch.setattr(head.estimator, "predict_proba",
                        lambda *a: pytest.fail("prediction on changed state"))
    with pytest.raises(module.HeadValidationError):
        module.predict_head(head, np.zeros((rows, 1536), np.float32))


def test_wrong_estimator_type_and_stale_diagnostics_fail_for_empty(head):
    class WrongSGD(SGDClassifier):
        pass

    report = head.diagnostics
    with pytest.raises(module.HeadValidationError, match="SGDClassifier"):
        module.predict_head(module.FittedHead(WrongSGD(), report),
                            np.empty((0, 1536), np.float32))
    report["fitted_state_sha256"] = "0" * 64
    with pytest.raises(module.HeadValidationError, match="fingerprint"):
        module.predict_head(module.FittedHead(head.estimator, report),
                            np.empty((0, 1536), np.float32))


@pytest.mark.parametrize("field,value", [("n_iter", 0), ("t", 0.0),
                                        ("n_features_in", 1536.0), ("classes", [1] * 5)])
def test_inconsistent_diagnostic_snapshots_are_rejected_before_empty(head, field, value):
    report = head.diagnostics
    report[field] = value
    with pytest.raises(module.HeadValidationError, match="recorded"):
        module.predict_head(module.FittedHead(head.estimator, report),
                            np.empty((0, 1536), np.float32))


def test_refitted_estimator_cannot_reuse_old_fit_diagnostics(head):
    man = manifest()
    changed = features_for(man)
    changed[:, 16] = 0.2
    head.estimator.fit(changed, np.array([row.tissue_label for row in man.tiles],
                                       np.int64), sample_weight=None)
    with pytest.raises(module.HeadValidationError):
        module.predict_head(head, np.empty((0, 1536), np.float32))


def test_class_array_dtype_changes_are_fingerprinted_even_when_values_match(head):
    head.estimator.classes_ = head.estimator.classes_.astype(np.uint64)
    with pytest.raises(module.HeadValidationError, match="fingerprint"):
        module.predict_head(head, np.empty((0, 1536), np.float32))


def test_empty_prediction_checks_state_but_does_not_call_estimator(head, monkeypatch):
    monkeypatch.setattr(head.estimator, "decision_function",
                        lambda *a: pytest.fail("empty decision call"))
    monkeypatch.setattr(head.estimator, "predict_proba",
                        lambda *a: pytest.fail("empty proba call"))
    result = module.predict_head(head, np.empty((0, 1536), dtype=np.float32))
    assert result.probabilities.shape == (0, 5) and result.probabilities.dtype == np.float32
    assert result.labels.shape == (0,) and result.labels.dtype == np.int64


@pytest.mark.parametrize("bad", [
    None, np.empty((0, 1535), np.float32), np.empty((0, 1536), np.float64),
    np.zeros((1, 3072), np.float32)[:, ::2],
    np.ma.zeros((1, 1536), dtype=np.float32),
    np.full((1, 1536), np.inf, np.float32),
])
def test_prediction_inputs_fail_before_method_call(head, bad, monkeypatch):
    monkeypatch.setattr(head.estimator, "predict_proba",
                        lambda *a: pytest.fail("invalid prediction input"))
    with pytest.raises(module.HeadValidationError):
        module.predict_head(head, bad)


def test_probability_saturation_uses_proba_argmax_not_predict(head, monkeypatch):
    # These finite scores saturate expit, so class 1 wins the probability tie
    # while class 2 has the largest decision score.
    monkeypatch.setattr(head.estimator, "decision_function",
                        lambda x: np.repeat(
                            np.array([[100, 101, 99, 98, 97]], np.float32),
                            len(x), axis=0))
    inputs = np.zeros((1, 1536), np.float32)
    assert head.estimator.predict(inputs).tolist() == [2]
    result = module.predict_head(head, inputs)
    assert result.probabilities.tolist() == [[np.float32(0.2)] * 5]
    assert result.labels.tolist() == [1]


@pytest.mark.parametrize("probabilities", [
    np.zeros((1, 5), np.float32), np.ones((1, 5), np.float32),
    np.array([[np.nan, 0, 0, 0, 0]], np.float32),
    np.array([[np.inf, 0, 0, 0, 0]], np.float32),
    np.array([[-0.1, 0.3, 0.3, 0.3, 0.2]], np.float32),
    np.array([[1.1, 0, 0, 0, 0]], np.float32),
    np.full((1, 5), 0.2, np.float64), np.full((1, 4), 0.25, np.float32),
    np.ma.array([[0.2] * 5], dtype=np.float32),
    np.array([[0.2] * 5], np.float32).view(ArraySubclass),
])
def test_invalid_native_probabilities_are_never_cast_cleaned_or_substituted(
        head, probabilities, monkeypatch):
    original = probabilities.copy()
    monkeypatch.setattr(head.estimator, "predict_proba", lambda x: probabilities)
    with pytest.raises(module.HeadValidationError):
        module.predict_head(head, np.zeros((1, 1536), np.float32))
    assert np.array_equal(probabilities, original, equal_nan=True)


def test_probability_roundoff_uses_fp64_sum_and_exact_eight_epsilon_bound(head, monkeypatch):
    eps = np.finfo(np.float32).eps
    probabilities = np.array([[1, 0, 0, 0, 0]], np.float32)
    probabilities[0, 1] = 8 * eps
    monkeypatch.setattr(head.estimator, "predict_proba", lambda x: probabilities)
    accepted = module.predict_head(head, np.zeros((1, 1536), np.float32))
    assert accepted.probabilities is probabilities
    probabilities[0, 1] = np.nextafter(np.float32(8 * eps), np.float32(np.inf))
    with pytest.raises(module.HeadValidationError, match="normalization"):
        module.predict_head(head, np.zeros((1, 1536), np.float32))


def test_nonfinite_decisions_and_prediction_warning_fail(head, monkeypatch):
    monkeypatch.setattr(head.estimator, "decision_function",
                        lambda x: np.full((len(x), 5), np.nan, np.float32))
    with pytest.raises(module.HeadValidationError, match="decision scores"):
        module.predict_head(head, np.zeros((1, 1536), np.float32))
    monkeypatch.setattr(head.estimator, "decision_function",
                        lambda x: np.zeros((len(x), 5), np.float32))

    def warn(x):
        warnings.warn("synthetic prediction warning", RuntimeWarning)
        return np.full((len(x), 5), 0.2, np.float32)

    monkeypatch.setattr(head.estimator, "predict_proba", warn)
    with pytest.raises(module.HeadValidationError, match="prediction: unexpected warnings"):
        module.predict_head(head, np.zeros((1, 1536), np.float32))


def test_pooled_tumour_counts_all_rest_classes_and_not_fold_average():
    truth = np.array([1, 1, 1, 2, 3, 4, 5, 5], np.int64)
    predicted = np.array([1, 1, 2, 1, 3, 4, 1, 5], np.uint64)
    result = module.tumour_counts(truth, predicted)
    assert {key: result[key] for key in ("tp", "fn", "fp", "tn", "total")} == {
        "tp": 2, "fn": 1, "fp": 2, "tn": 3, "total": 8}
    assert result["sensitivity"] == 2 / 3 and result["specificity"] == 3 / 5
    assert result["sensitivity_status"] == result["specificity_status"] == "estimable"
    assert result["sensitivity_reason"] is result["specificity_reason"] is None
    first = module.tumour_counts(truth[:1], predicted[:1])
    second = module.tumour_counts(truth[1:], predicted[1:])
    assert result["sensitivity"] != (first["sensitivity"] + second["sensitivity"]) / 2


@pytest.mark.parametrize("truth,predicted,unavailable", [
    ([], [], ("sensitivity", "specificity")),
    ([2, 3], [1, 3], ("sensitivity",)),
    ([1, 1], [1, 2], ("specificity",)),
])
def test_zero_denominators_have_explicit_unavailable_status(truth, predicted, unavailable):
    result = module.tumour_counts(np.array(truth, np.int64), np.array(predicted, np.int64))
    assert result["total"] == len(truth)
    for name in unavailable:
        assert result[name] is None and result[name + "_status"] == "unavailable"
        assert result[name + "_reason"] is not None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [
    None, [1], np.array([True]), np.array([1.0]), np.array([1], dtype=object),
    np.ma.array([1]), np.array([1]).view(ArraySubclass), np.array([[1]]),
    np.array([0]), np.array([6]), np.array([-1]),
    np.array([2**64 - 1], np.uint64),
])
@pytest.mark.parametrize("side", ["truth", "predictions"])
def test_count_inputs_rejected_before_coercion(bad, side):
    inputs = {"truth": np.array([1], np.int64), "predictions": np.array([1], np.int64)}
    inputs[side] = bad
    with pytest.raises(module.HeadValidationError):
        module.tumour_counts(**inputs)


def test_count_length_mismatch_fails():
    with pytest.raises(module.HeadValidationError, match="shape mismatch"):
        module.tumour_counts(np.array([1]), np.array([1, 2]))
