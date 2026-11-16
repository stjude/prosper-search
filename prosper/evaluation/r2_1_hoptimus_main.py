"""Pure query-outcome and whole-patient inference kernels for R2.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


QueryStatus = Literal["returned", "empty_query", "empty_reference_library"]
EstimateStatus = Literal["estimable", "unavailable"]


class ValidationError(ValueError):
    """An input violates the declared metric or patient-population specification."""


class ComputationError(ArithmeticError):
    """A defined numerical output could not be computed as a finite value."""


@dataclass(frozen=True)
class QueryMetrics:
    status: QueryStatus
    conditional_hit: float | None
    conditional_precision: float | None
    conditional_reason: str | None
    coverage: int
    hit_yield: int
    returned_depth: int
    short_list: bool


@dataclass(frozen=True)
class PatientDraws:
    patient_ids: tuple[str, ...]
    multiplicities: NDArray[np.int64]

    def __post_init__(self) -> None:
        _reject_masked(self.patient_ids, "draw patient_ids")
        _reject_masked(self.multiplicities, "multiplicities")


@dataclass(frozen=True)
class Estimate:
    point: float | None
    point_status: EstimateStatus
    point_reason: str | None
    ci_low: float | None
    ci_high: float | None
    ci_status: EstimateStatus
    ci_reason: str | None
    n_queries: int
    n_patients: int
    zero_support_replicates: int
    replicates: NDArray[np.float64]
    valid: NDArray[np.bool_]


@dataclass(frozen=True)
class MetricSummary:
    groups: tuple[str, ...]
    group_estimates: tuple[Estimate, ...]
    macro: Estimate
    included_query_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    replicate_group_support: NDArray[np.int64]


@dataclass(frozen=True)
class PairedSummary:
    arm_a: MetricSummary
    arm_b: MetricSummary
    difference: MetricSummary


def _reject_masked(values: object, name: str) -> None:
    # Array coercion discards masked-array metadata, including nomask arrays.
    pending = [values]
    seen = set()
    while pending:
        value = pending.pop()
        if np.ma.isMaskedArray(value):
            raise ValidationError(f"{name} must not contain masked representations")
        if isinstance(value, (list, tuple)):
            children = value
        elif isinstance(value, np.ndarray) and value.dtype.hasobject:
            children = value.flat
        else:
            continue
        if id(value) not in seen:
            seen.add(id(value))
            pending.extend(children)


def _integer(value: int, name: str, minimum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValidationError(f"{name} must be an integer >= {minimum}, not bool")
    return int(value)


def _ids(values: Sequence[str], name: str, *, unique: bool = False) -> tuple[str, ...]:
    _reject_masked(values, name)
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{name} must be a sequence of exact string IDs")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValidationError(f"{name} must be a sequence of exact string IDs") from exc
    if any(not isinstance(v, str) or not v.strip() for v in result):
        raise ValidationError(f"{name} must contain nonempty string IDs")
    if unique and len(set(result)) != len(result):
        raise ValidationError(f"{name} must be unique")
    return result


def _booleans(values: ArrayLike, name: str) -> NDArray[np.bool_]:
    _reject_masked(values, name)
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a one-dimensional Boolean sequence") from exc
    if array.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional")
    if array.dtype.kind != "b":
        if array.size == 0 and isinstance(values, (list, tuple)):
            return array.astype(bool)
        raise ValidationError(f"{name} must contain Booleans, not numeric masks")
    return array


def _readonly(values: NDArray) -> NDArray:
    result = values.copy()
    result.setflags(write=False)
    return result


def query_metrics(
    relevance: Sequence[bool] | NDArray[np.bool_],
    cutoff: int,
    status: QueryStatus,
) -> QueryMetrics:
    """Score an explicit return/nonreturn; precision always divides by cutoff."""
    cutoff = _integer(cutoff, "cutoff", 1)
    rel = _booleans(relevance, "relevance")
    if not isinstance(status, str) or status not in (
            "returned", "empty_query", "empty_reference_library"):
        raise ValidationError(f"unsupported query status: {status!r}")
    if status != "returned":
        if rel.size:
            raise ValidationError("a nonreturn must have no neighbours")
        return QueryMetrics(status, None, None, status, 0, 0, 0, False)
    if not rel.size:
        raise ValidationError("returned status requires a nonempty ranking")
    depth = min(cutoff, rel.size)
    count = int(np.count_nonzero(rel[:depth]))
    hit = int(count > 0)
    return QueryMetrics(
        status, float(hit), count / cutoff, None, 1, hit, depth, depth < cutoff)


def make_patient_draws(
    patient_ids: Sequence[str], *, count: int = 5000, seed: int = 123,
) -> PatientDraws:
    """Return lexicographically ordered, shared whole-patient multiplicities.

    Count/seed overrides are for small synthetic fixtures only. An empty
    population returns zero-column draws without sampling from zero patients.
    """
    count = _integer(count, "count", 1)
    seed = _integer(seed, "seed", 0)
    patients = tuple(sorted(set(_ids(patient_ids, "patient_ids"))))
    p = len(patients)
    multiplicities = np.zeros((count, p), dtype=np.int64)
    if p:
        rng = np.random.Generator(np.random.PCG64(seed))
        for b in range(count):
            multiplicities[b] = np.bincount(
                rng.integers(p, size=p), minlength=p)
    return PatientDraws(patients, _readonly(multiplicities))


def _values(values: ArrayLike, n: int, name: str) -> NDArray[np.float64]:
    _reject_masked(values, name)
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must have one real numeric value per query") from exc
    if array.ndim != 1 or array.shape[0] != n:
        raise ValidationError(f"{name} must have one value per declared query")
    if array.dtype.kind not in ("b", "i", "u", "f"):
        raise ValidationError(f"{name} must be real numeric values")
    if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
        raise ValidationError(f"{name} must be finite and in [0,1], including excluded rows")
    return array.astype(np.float64, copy=True)


def _population(
    query_ids: Sequence[str],
    patient_ids: Sequence[str],
    group_ids: Sequence[str],
    groups: Sequence[str],
    inclusion: ArrayLike,
    draws: PatientDraws,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...],
           NDArray[np.bool_], NDArray[np.int64], tuple[str, ...]]:
    queries = _ids(query_ids, "query_ids", unique=True)
    patients = _ids(patient_ids, "patient_ids")
    labels = _ids(group_ids, "group_ids")
    fixed = _ids(groups, "groups", unique=True)
    if not fixed:
        raise ValidationError("groups must declare at least one fixed group")
    mask = _booleans(inclusion, "inclusion")
    if len(patients) != len(queries) or len(labels) != len(queries) or len(mask) != len(queries):
        raise ValidationError("query/patient/group/inclusion lengths must match")
    if not set(labels).issubset(fixed):
        raise ValidationError("every group ID, including excluded rows, must belong to groups")
    if not isinstance(draws, PatientDraws):
        raise ValidationError("draws must be PatientDraws")
    order = _ids(draws.patient_ids, "draw patient_ids", unique=True)
    if order != tuple(sorted(order)):
        raise ValidationError("draw patient_ids must be lexicographically ordered")
    included_patients = tuple(sorted({p for p, include in zip(patients, mask) if include}))
    if order != included_patients:
        raise ValidationError("draw patients must equal exactly patients among included queries")
    _reject_masked(draws.multiplicities, "multiplicities")
    try:
        weights = np.asarray(draws.multiplicities)
    except (TypeError, ValueError) as exc:
        raise ValidationError("multiplicities must be a rectangular integer array") from exc
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] != len(order):
        raise ValidationError("multiplicities must have positive replicate count and P columns")
    if weights.dtype.kind not in ("i", "u"):
        raise ValidationError("multiplicities must be integers, not bool or float")
    p = len(order)
    if np.any(weights < 0) or np.any(weights > p):
        raise ValidationError("multiplicities must be nonnegative and no larger than P")
    weights = weights.astype(np.int64, copy=True)
    if np.any(weights.sum(axis=1) != p):
        raise ValidationError("every multiplicity row must sum to P")
    return queries, patients, labels, fixed, mask, weights, order


def _estimate(
    point: float | None,
    point_reason: str | None,
    n_queries: int,
    n_patients: int,
    replicates: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> Estimate:
    if ((point is not None and not np.isfinite(point))
            or not np.all(np.isfinite(replicates[valid]))):
        raise ComputationError("defined point/replicate values must be finite")
    invalid = int(np.count_nonzero(~valid))
    reason = point_reason if point is None else (
        "zero_support_replicates" if invalid else None)
    low = high = None
    if reason is None:
        low, high = map(float, np.quantile(
            replicates, [0.025, 0.975], method="linear"))
    return Estimate(
        point=point,
        point_status="unavailable" if point is None else "estimable",
        point_reason=point_reason,
        ci_low=low,
        ci_high=high,
        ci_status="unavailable" if reason else "estimable",
        ci_reason=reason,
        n_queries=n_queries,
        n_patients=n_patients,
        zero_support_replicates=invalid,
        replicates=_readonly(replicates),
        valid=_readonly(valid),
    )


def _summarize(
    values: NDArray[np.float64],
    queries: tuple[str, ...],
    patients: tuple[str, ...],
    labels: tuple[str, ...],
    groups: tuple[str, ...],
    mask: NDArray[np.bool_],
    weights: NDArray[np.int64],
    patient_order: tuple[str, ...],
) -> MetricSummary:
    patient_index = {p: i for i, p in enumerate(patient_order)}
    group_index = {g: i for i, g in enumerate(groups)}
    counts = np.zeros((len(patient_order), len(groups)), dtype=np.int64)
    sums = np.zeros(counts.shape, dtype=np.float64)
    for i in np.flatnonzero(mask):
        p, g = patient_index[patients[i]], group_index[labels[i]]
        counts[p, g] += 1
        sums[p, g] += values[i]
    support = weights @ counts
    valid = support > 0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            numerators = weights @ sums
            replicates = np.full(support.shape, np.nan, dtype=np.float64)
            np.divide(numerators, support, out=replicates, where=valid)
        except FloatingPointError as exc:
            raise ComputationError("nonfinite patient-weighted metric arithmetic") from exc
    observed_counts = counts.sum(axis=0)
    observed_sums = sums.sum(axis=0)
    estimates = []
    for g in range(len(groups)):
        n = int(observed_counts[g])
        estimates.append(_estimate(
            float(observed_sums[g] / n) if n else None,
            None if n else "no_included_queries",
            n, int(np.count_nonzero(counts[:, g])),
            replicates[:, g], valid[:, g]))
    macro_valid = valid.all(axis=1)
    macro_replicates = np.full(weights.shape[0], np.nan, dtype=np.float64)
    macro_replicates[macro_valid] = replicates[macro_valid].mean(axis=1)
    n_included = int(np.count_nonzero(mask))
    macro_point = (float(np.mean([e.point for e in estimates]))
                   if np.all(observed_counts > 0) else None)
    macro_reason = (None if macro_point is not None else
                    "no_included_queries" if not n_included else "missing_group_support")
    macro = _estimate(
        macro_point, macro_reason, n_included, len(patient_order),
        macro_replicates, macro_valid)
    return MetricSummary(
        groups, tuple(estimates), macro,
        tuple(q for q, include in zip(queries, mask) if include),
        patient_order, _readonly(support))


def summarize_metric(
    query_ids: Sequence[str],
    patient_ids: Sequence[str],
    group_ids: Sequence[str],
    groups: Sequence[str],
    values: ArrayLike,
    inclusion: ArrayLike,
    draws: PatientDraws,
) -> MetricSummary:
    """Summarize one explicitly included, slide-weighted metric population."""
    population = _population(query_ids, patient_ids, group_ids, groups, inclusion, draws)
    vector = _values(values, len(population[0]), "values")
    return _summarize(vector, *population)


def summarize_paired(
    query_ids: Sequence[str],
    patient_ids: Sequence[str],
    group_ids: Sequence[str],
    groups: Sequence[str],
    values_a: ArrayLike,
    values_b: ArrayLike,
    inclusion: ArrayLike,
    draws: PatientDraws,
) -> PairedSummary:
    """A-minus-B on one common membership declaration and one shared draw table.

    Conditional callers must explicitly supply the both-returned intersection
    and draws for exactly its patients. There are no arm-specific masks.
    """
    population = _population(query_ids, patient_ids, group_ids, groups, inclusion, draws)
    n = len(population[0])
    a = _summarize(_values(values_a, n, "values_a"), *population)
    b = _summarize(_values(values_b, n, "values_b"), *population)
    if (a.included_query_ids != b.included_query_ids
            or not np.array_equal(a.replicate_group_support, b.replicate_group_support)):
        raise ValidationError("paired arms must have identical membership and denominator tensors")

    def difference(x: Estimate, y: Estimate) -> Estimate:
        valid = x.valid & y.valid
        replicates = np.full(x.replicates.shape, np.nan, dtype=np.float64)
        replicates[valid] = x.replicates[valid] - y.replicates[valid]
        point = x.point - y.point if x.point is not None and y.point is not None else None
        return _estimate(
            point, x.point_reason or y.point_reason, x.n_queries, x.n_patients,
            replicates, valid)

    diff = MetricSummary(
        a.groups,
        tuple(difference(x, y) for x, y in zip(a.group_estimates, b.group_estimates)),
        difference(a.macro, b.macro),
        a.included_query_ids, a.patient_ids, a.replicate_group_support)
    return PairedSummary(a, b, diff)
