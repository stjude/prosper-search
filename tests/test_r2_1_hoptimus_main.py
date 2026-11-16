"""Small synthetic fixtures for the released R2.1 pure inference specification."""

import numpy as np
import pytest

from prosper.evaluation.r2_1_hoptimus_main import (
    PatientDraws,
    ValidationError,
    make_patient_draws,
    query_metrics,
    summarize_metric,
    summarize_paired,
)


def fixture_inputs():
    return {
        "query_ids": ["q1", "q2", "q3", "q4", "q5"],
        "patient_ids": ["p1", "p1", "p1", "p2", "p2"],
        "group_ids": ["A", "A", "B", "A", "B"],
        "groups": ["A", "B"],
        "values": np.array([1.0, 1.0, 0.0, 0.0, 1.0]),
        "inclusion": np.ones(5, dtype=bool),
        "draws": PatientDraws(
            ("p1", "p2"), np.array([[2, 0], [1, 1], [0, 2]], dtype=np.int64)),
    }


@pytest.mark.parametrize(
    "relevance,k,hit,precision,depth,short",
    [
        ([True, False], 5, 1, 0.2, 2, True),
        ([False, False], 1, 0, 0.0, 1, False),
        ([False, True, True], 2, 1, 0.5, 2, False),
        ([False, False, True], 2, 0, 0.0, 2, False),
        ([True] * 5, 5, 1, 1.0, 5, False),
    ],
)
def test_query_metrics_fixed_cutoff(relevance, k, hit, precision, depth, short):
    outcome = query_metrics(relevance, k, "returned")
    assert outcome.conditional_hit == hit
    assert outcome.conditional_precision == precision
    assert outcome.conditional_reason is None
    assert outcome.coverage == 1
    assert outcome.hit_yield == hit
    assert outcome.returned_depth == depth
    assert outcome.short_list is short


@pytest.mark.parametrize("status", ["empty_query", "empty_reference_library"])
def test_nonreturn_is_explicit(status):
    outcome = query_metrics([], 5, status)
    assert outcome.conditional_hit is None
    assert outcome.conditional_precision is None
    assert outcome.conditional_reason == status
    assert outcome.coverage == outcome.hit_yield == outcome.returned_depth == 0
    assert outcome.short_list is False


@pytest.mark.parametrize(
    "relevance,k,status",
    [
        ([], 5, "returned"), ([True], 5, "empty_query"),
        ([False], 5, "empty_reference_library"), ([], 5, "solver_error"),
        ([True], 0, "returned"), ([True], -1, "returned"),
        ([True], True, "returned"), ([True], 1.0, "returned"),
        ([1], 1, "returned"), ([[True]], 1, "returned"),
        ([[True], []], 1, "returned"), ([True], 1, ["returned"]),
    ],
)
def test_query_metrics_rejects_invalid_inputs(relevance, k, status):
    with pytest.raises(ValidationError):
        query_metrics(relevance, k, status)


def test_draws_exact_local_pcg64_and_production_defaults():
    draws = make_patient_draws(["z", "a", "a"], count=7)
    assert draws.patient_ids == ("a", "z")
    rng = np.random.Generator(np.random.PCG64(123))
    expected = np.array([np.bincount(rng.integers(2, size=2), minlength=2)
                         for _ in range(7)])
    np.testing.assert_array_equal(draws.multiplicities, expected)
    np.testing.assert_array_equal(draws.multiplicities.sum(axis=1), 2)
    assert draws.multiplicities.dtype == np.int64
    assert not draws.multiplicities.flags.writeable
    defaults = make_patient_draws(["a", "z"])
    assert defaults.multiplicities.shape == (5000, 2)
    np.testing.assert_array_equal(defaults.multiplicities[:7], expected)


@pytest.mark.parametrize("patients", [[""], [" "], ["p", 1], "patient", None, 1])
def test_draws_rejects_invalid_ids(patients):
    with pytest.raises(ValidationError):
        make_patient_draws(patients, count=2)


@pytest.mark.parametrize("kwargs", [
    {"count": 0}, {"count": True}, {"count": 1.5},
    {"seed": -1}, {"seed": False}, {"seed": 2.5},
])
def test_draws_rejects_invalid_policy_arguments(kwargs):
    with pytest.raises(ValidationError):
        make_patient_draws(["p"], **kwargs)


def test_empty_draw_population_does_not_sample(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("empty populations must not initialize a sampler")
    monkeypatch.setattr(np.random, "Generator", forbidden)
    draws = make_patient_draws([], count=3)
    assert draws.patient_ids == ()
    assert draws.multiplicities.shape == (3, 0)


def test_cross_label_whole_patient_weights_and_slide_weighted_points():
    summary = summarize_metric(**fixture_inputs())
    assert summary.groups == ("A", "B")
    a, b = summary.group_estimates
    assert (a.n_queries, a.n_patients) == (3, 2)
    assert (b.n_queries, b.n_patients) == (2, 2)
    assert a.point == pytest.approx(2 / 3)
    assert a.point != 0.5  # The equally patient-weighted answer differs.
    assert b.point == 0.5
    assert summary.macro.point == pytest.approx(7 / 12)
    np.testing.assert_array_equal(summary.replicate_group_support, [[4, 2], [3, 2], [2, 2]])
    np.testing.assert_allclose(a.replicates, [1, 2 / 3, 0])
    np.testing.assert_allclose(b.replicates, [0, 0.5, 1])
    np.testing.assert_allclose(summary.macro.replicates, [0.5, 7 / 12, 0.5])
    assert summary.macro.ci_status == "estimable"
    np.testing.assert_allclose(
        [summary.macro.ci_low, summary.macro.ci_high],
        np.quantile(summary.macro.replicates, [0.025, 0.975], method="linear"))


def test_fixed_order_and_equal_group_macro_not_micro():
    args = fixture_inputs()
    args["groups"] = ["B", "A"]
    summary = summarize_metric(**args)
    assert summary.groups == ("B", "A")
    assert summary.group_estimates[0].point == 0.5
    assert summary.macro.point == pytest.approx(7 / 12)
    assert summary.macro.point != pytest.approx(args["values"].mean())


def test_absent_group_point_and_ci_unavailable_other_group_remains_estimable():
    args = fixture_inputs()
    args["groups"] = ["A", "B", "C"]
    summary = summarize_metric(**args)
    absent = summary.group_estimates[2]
    assert absent.point is absent.ci_low is absent.ci_high is None
    assert absent.point_status == absent.ci_status == "unavailable"
    assert absent.point_reason == absent.ci_reason == "no_included_queries"
    assert absent.zero_support_replicates == 3
    assert not absent.valid.any()
    assert np.isnan(absent.replicates).all()
    assert summary.macro.point is None
    assert summary.macro.point_reason == "missing_group_support"
    assert summary.group_estimates[0].ci_status == "estimable"


def test_invalid_replicate_is_not_dropped_or_redrawn():
    summary = summarize_metric(
        ["q1", "q2", "q3"], ["p1", "p2", "p2"], ["A", "B", "B"],
        ["A", "B"], [1, 0, 1], [True] * 3,
        PatientDraws(("p1", "p2"), np.array([[0, 2], [1, 1]])))
    a, b = summary.group_estimates
    assert a.point == 1
    assert a.ci_status == "unavailable"
    assert a.ci_reason == "zero_support_replicates"
    assert a.ci_low is a.ci_high is None
    assert a.zero_support_replicates == 1
    np.testing.assert_array_equal(a.valid, [False, True])
    assert np.isnan(a.replicates[0])
    assert b.ci_status == "estimable"
    assert b.ci_low == b.ci_high == 0.5
    assert summary.macro.zero_support_replicates == 1
    assert summary.macro.ci_status == "unavailable"
    assert summary.macro.point == 0.75


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_constant_values_have_explicit_degenerate_intervals(value):
    args = fixture_inputs()
    args["values"][:] = value
    summary = summarize_metric(**args)
    for result in (*summary.group_estimates, summary.macro):
        assert result.point == result.ci_low == result.ci_high == value
        assert result.ci_status == "estimable"
        assert result.zero_support_replicates == 0


def test_excluded_only_patients_are_validated_but_not_drawn():
    summary = summarize_metric(
        ["q1", "q2", "excluded"], ["p1", "p1", "excluded-only"],
        ["A", "B", "A"], ["A", "B"], [1.0, 0.5, 0.7],
        [True, True, False], make_patient_draws(["p1"], count=4))
    assert summary.patient_ids == ("p1",)
    assert summary.included_query_ids == ("q1", "q2")
    assert summary.macro.point == 0.75
    assert summary.macro.n_queries == 2
    assert summary.macro.n_patients == 1
    assert summary.macro.ci_low == summary.macro.ci_high == 0.75


def test_empty_conditional_intersection_is_explicitly_nonestimable():
    result = summarize_paired(
        ["q1", "q2"], ["p1", "p2"], ["A", "B"], ["A", "B"],
        [1, 0], [0, 1], [False, False], make_patient_draws([], count=4))
    for summary in (result.arm_a, result.arm_b, result.difference):
        assert summary.patient_ids == summary.included_query_ids == ()
        assert summary.macro.n_queries == summary.macro.n_patients == 0
        assert summary.macro.point is None
        assert summary.macro.point_reason == "no_included_queries"
        assert summary.macro.ci_status == "unavailable"
        assert summary.macro.zero_support_replicates == 4
        assert not summary.macro.valid.any()


def test_empty_query_declaration_is_supported():
    summary = summarize_metric([], [], [], ["A"], [], [], make_patient_draws([], count=2))
    assert summary.macro.point is None
    assert summary.macro.n_queries == 0


def test_exact_string_ids_are_not_normalized_or_coerced():
    draws = make_patient_draws(["001", "1", " 1"], count=3)
    assert draws.patient_ids == (" 1", "001", "1")
    summary = summarize_metric(
        ["001", "1", " 1"], ["001", "1", " 1"], ["A"] * 3,
        ["A"], [1, 0, 0.5], [True] * 3, draws)
    assert summary.included_query_ids == ("001", "1", " 1")
    assert summary.macro.n_patients == 3


@pytest.mark.parametrize("change", [
    {"query_ids": ["q1", "q1", "q3", "q4", "q5"]},
    {"query_ids": ["q1", "", "q3", "q4", "q5"]},
    {"patient_ids": ["p1", "", "p1", "p2", "p2"]},
    {"patient_ids": ["p1", "p1", "p1", "p2"]},
    {"group_ids": ["A", "A", "outside", "A", "B"]},
    {"groups": ["A", "A", "B"]}, {"groups": []},
    {"inclusion": [1] * 5}, {"inclusion": [[True] * 5]},
    {"inclusion": [True] * 4},
    {"values": [0, 0, 0, 0]}, {"values": [[0] * 5]},
    {"values": [0, 0, np.nan, 0, 0]},
    {"values": [0, 0, np.inf, 0, 0]},
    {"values": [0, 0, -0.01, 0, 0]},
    {"values": [0, 0, 1.01, 0, 0]},
    {"values": ["0"] * 5},
    {"values": [0j] * 5},
    {"values": [[0], []]},
    {"patient_ids": None},
    {"inclusion": [[True], []]},
])
def test_invalid_population_and_values(change):
    args = fixture_inputs()
    args.update(change)
    with pytest.raises(ValidationError):
        summarize_metric(**args)


@pytest.mark.parametrize("field,value", [
    ("query_ids", ["q1", "q2", "q3", "q4", ""]),
    ("patient_ids", ["p1", "p1", "p1", "p2", ""]),
    ("group_ids", ["A", "A", "B", "A", "outside"]),
    ("values", [1, 1, 0, 0, np.nan]),
    ("values", [1, 1, 0, 0, np.inf]),
    ("values", [1, 1, 0, 0, -0.1]),
])
def test_excluded_rows_still_require_valid_identity_and_finite_values(field, value):
    args = fixture_inputs()
    args["inclusion"][-1] = False
    args[field] = value
    with pytest.raises(ValidationError):
        summarize_metric(**args)


@pytest.mark.parametrize("draws", [
    PatientDraws(("p2", "p1"), np.array([[1, 1]])),
    PatientDraws(("p1", "p1"), np.array([[1, 1]])),
    PatientDraws(("p1",), np.array([[1]])),
    PatientDraws(("p1", "p2", "extra"), np.array([[1, 1, 1]])),
    PatientDraws(("p1", "p2"), np.array([1, 1])),
    PatientDraws(("p1", "p2"), np.empty((0, 2), dtype=int)),
    PatientDraws(("p1", "p2"), np.array([[1]])),
    PatientDraws(("p1", "p2"), np.array([[1.0, 1.0]])),
    PatientDraws(("p1", "p2"), np.array([[True, True]])),
    PatientDraws(("p1", "p2"), np.array([[-1, 3]])),
    PatientDraws(("p1", "p2"), np.array([[0, 1]])),
    PatientDraws(("p1", "p2"), np.array([[2**64 - 1, 3]], dtype=np.uint64)),
    PatientDraws(("p1", "p2"), [[1], []]),
])
def test_invalid_draw_tables(draws):
    args = fixture_inputs()
    args["draws"] = draws
    with pytest.raises(ValidationError):
        summarize_metric(**args)


def test_excluded_only_and_other_extra_draw_patients_are_rejected():
    for extra in ("excluded-only", "unrelated"):
        with pytest.raises(ValidationError, match="exactly"):
            summarize_metric(
                ["q1", "excluded"], ["p1", "excluded-only"], ["A", "A"],
                ["A"], [1, 0], [True, False],
                make_patient_draws(["p1", extra], count=2))


def paired_inputs():
    args = fixture_inputs()
    args["values_a"] = args.pop("values")
    args["values_b"] = 1 - args["values_a"]
    return args


def test_pairing_uses_same_denominators_and_exact_a_minus_b_draws():
    result = summarize_paired(**paired_inputs())
    assert result.difference.macro.point == pytest.approx(1 / 6)
    np.testing.assert_array_equal(
        result.arm_a.replicate_group_support, result.arm_b.replicate_group_support)
    for a, b, diff in zip(
        (*result.arm_a.group_estimates, result.arm_a.macro),
        (*result.arm_b.group_estimates, result.arm_b.macro),
        (*result.difference.group_estimates, result.difference.macro),
    ):
        assert diff.point == a.point - b.point
        np.testing.assert_allclose(diff.replicates, a.replicates - b.replicates)
        np.testing.assert_array_equal(diff.valid, a.valid & b.valid)


def test_identical_arms_have_zero_difference_without_certainty_claim():
    args = paired_inputs()
    args["values_b"] = args["values_a"].copy()
    result = summarize_paired(**args)
    assert result.difference.macro.point == 0
    assert result.difference.macro.ci_low == result.difference.macro.ci_high == 0
    assert result.difference.macro.ci_status == "estimable"


def test_pair_with_invalid_group_support_does_not_salvage_interval():
    args = paired_inputs()
    args["group_ids"] = ["A", "A", "A", "B", "B"]
    result = summarize_paired(**args)
    assert result.difference.macro.point is not None
    assert result.difference.macro.zero_support_replicates == 2
    assert result.difference.macro.ci_low is result.difference.macro.ci_high is None
    assert result.difference.macro.ci_reason == "zero_support_replicates"


def test_pair_cannot_use_separately_subsetted_or_nan_coded_arms():
    args = paired_inputs()
    for invalid in ([0, 1], [0, 0, np.nan, 0, 0]):
        args["values_b"] = invalid
        with pytest.raises(ValidationError):
            summarize_paired(**args)
    args = paired_inputs()
    with pytest.raises(TypeError):
        summarize_paired(**args, inclusion_b=[False] * 5)


def test_conditional_pair_uses_intersection_patient_population():
    args = paired_inputs()
    args["inclusion"] = [True, True, True, False, False]
    with pytest.raises(ValidationError, match="exactly"):
        summarize_paired(**args)
    args["draws"] = make_patient_draws(["p1"], count=3)
    result = summarize_paired(**args)
    assert result.difference.patient_ids == ("p1",)
    assert result.difference.macro.n_queries == 3
    assert result.difference.macro.point == 0


def test_inputs_and_global_rng_are_unchanged_and_outputs_are_readonly():
    args = fixture_inputs()
    original = {k: v.copy() if isinstance(v, (list, np.ndarray)) else v
                for k, v in args.items()}
    draw_values = args["draws"].multiplicities.copy()
    rng_before = np.random.get_state()
    query_metrics(np.array([True, False]), 5, "returned")
    make_patient_draws(["p2", "p1"], count=3)
    result = summarize_metric(**args)
    summarize_paired(**paired_inputs())
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
    for k, value in original.items():
        if isinstance(value, (list, np.ndarray)):
            np.testing.assert_array_equal(args[k], value)
    np.testing.assert_array_equal(args["draws"].multiplicities, draw_values)
    for estimate in (*result.group_estimates, result.macro):
        assert not estimate.replicates.flags.writeable
        assert not estimate.valid.flags.writeable
    assert not result.replicate_group_support.flags.writeable


def test_patient_and_query_order_arm_order_and_replicate_chunks_are_invariant():
    first = make_patient_draws(["p2", "p1"], count=6)
    reordered = make_patient_draws(["p1", "p2"], count=6)
    np.testing.assert_array_equal(first.multiplicities, reordered.multiplicities)
    np.testing.assert_array_equal(
        first.multiplicities[:3], make_patient_draws(["p1", "p2"], count=3).multiplicities)
    args = paired_inputs()
    args["draws"] = first
    whole = summarize_paired(**args)
    for selection in (slice(0, 3), slice(3, 6)):
        args["draws"] = PatientDraws(first.patient_ids, first.multiplicities[selection])
        chunk = summarize_paired(**args)
        np.testing.assert_allclose(
            chunk.difference.macro.replicates,
            whole.difference.macro.replicates[selection], equal_nan=True)
    args["draws"] = first
    args["values_a"], args["values_b"] = args["values_b"], args["values_a"]
    reversed_arms = summarize_paired(**args)
    np.testing.assert_allclose(
        reversed_arms.difference.macro.replicates,
        -whole.difference.macro.replicates, equal_nan=True)
    indices = np.array([4, 2, 0, 3, 1])
    for key in ("query_ids", "patient_ids", "group_ids", "inclusion", "values_a", "values_b"):
        args[key] = np.asarray(args[key])[indices]
    reordered_queries = summarize_paired(**args)
    np.testing.assert_allclose(
        reordered_queries.difference.macro.replicates,
        reversed_arms.difference.macro.replicates, equal_nan=True)


MASKED_VARIANTS = (
    "array", "nomask_array", "scalar", "constant", "list", "tuple",
    "object_array", "nomask_object_array", "nested_array",
)


def masked_variant(values, variant):
    ordinary = np.asarray(values)
    mask = np.zeros(ordinary.shape, dtype=bool)
    mask.flat[0] = True
    if variant == "array":
        return np.ma.array(ordinary, mask=mask)
    if variant == "nomask_array":
        return np.ma.array(ordinary, mask=np.ma.nomask)
    if variant == "scalar":
        return np.ma.array(ordinary.flat[0], mask=True)
    if variant == "constant":
        return np.ma.masked
    if variant == "nested_array":
        return [np.ma.array(ordinary, mask=np.ma.nomask)]
    nested = ordinary.astype(object)
    nested.flat[0] = np.ma.array(
        ordinary.flat[0], mask=variant != "nomask_object_array")
    if variant == "list":
        return nested.tolist()
    if variant == "tuple":
        return tuple(nested.tolist())
    return nested


@pytest.mark.parametrize("variant", MASKED_VARIANTS)
def test_relevance_rejects_masked_representations_before_coercion(variant):
    with pytest.raises(ValidationError, match="relevance.*masked representations"):
        query_metrics(masked_variant([True, False], variant), 5, "returned")


@pytest.mark.parametrize("variant", MASKED_VARIANTS)
@pytest.mark.parametrize("field", ["values", "inclusion"])
def test_single_metric_rejects_masked_representations_before_coercion(field, variant):
    args = fixture_inputs()
    args[field] = masked_variant(args[field], variant)
    with pytest.raises(ValidationError, match=f"{field}.*masked representations"):
        summarize_metric(**args)


@pytest.mark.parametrize("variant", MASKED_VARIANTS)
@pytest.mark.parametrize("field", ["values_a", "values_b", "inclusion"])
def test_paired_metric_rejects_masked_representations_before_coercion(field, variant):
    args = paired_inputs()
    args[field] = masked_variant(args[field], variant)
    with pytest.raises(ValidationError, match=f"{field}.*masked representations"):
        summarize_paired(**args)


@pytest.mark.parametrize("variant", MASKED_VARIANTS)
def test_patient_draws_construction_rejects_masked_multiplicities(variant):
    with pytest.raises(ValidationError, match="multiplicities.*masked representations"):
        PatientDraws(("p1", "p2"), masked_variant([[1, 1]], variant))


@pytest.mark.parametrize("variant", MASKED_VARIANTS)
@pytest.mark.parametrize("paired", [False, True])
def test_summary_rechecks_masked_multiplicities_even_after_construction(variant, paired):
    args = paired_inputs() if paired else fixture_inputs()
    object.__setattr__(
        args["draws"], "multiplicities",
        masked_variant(args["draws"].multiplicities, variant))
    with pytest.raises(ValidationError, match="multiplicities.*masked representations"):
        (summarize_paired if paired else summarize_metric)(**args)


@pytest.mark.parametrize("variant", MASKED_VARIANTS)
def test_draw_population_rejects_masked_ids_before_coercion(variant):
    with pytest.raises(ValidationError, match="patient_ids.*masked representations"):
        make_patient_draws(masked_variant(["p1", "p2"], variant), count=2)


def test_nonreturn_and_empty_population_still_reject_nomask_arrays():
    with pytest.raises(ValidationError, match="masked representations"):
        query_metrics(np.ma.array([], dtype=bool), 5, "empty_query")
    with pytest.raises(ValidationError, match="masked representations"):
        PatientDraws((), np.ma.array(np.empty((2, 0), dtype=np.int64)))


def test_explicit_boolean_inclusion_masks_remain_valid_and_distinct_from_missingness():
    args = fixture_inputs()
    args["inclusion"] = np.array([True, True, True, False, False])
    args["draws"] = make_patient_draws(["p1"], count=3)
    summary = summarize_metric(**args)
    assert summary.included_query_ids == ("q1", "q2", "q3")
    assert summary.macro.point == summary.macro.ci_low == summary.macro.ci_high == 0.5


def test_excluded_masked_value_is_not_inferred_as_exclusion_or_filled():
    args = fixture_inputs()
    args["inclusion"][-1] = False
    args["values"] = np.ma.array(args["values"], mask=[False] * 4 + [True])
    with pytest.raises(ValidationError, match="values.*masked representations"):
        summarize_metric(**args)
