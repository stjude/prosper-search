"""Pure statistical helpers for the reader-study endpoints (plan 002 §4–§6 as amended by 003).

Descriptive only: proportions with Wilson intervals, whole-case bootstrap percentile intervals,
class and Likert tabulations. No hypothesis tests.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SEED = 20261027
N_DRAWS = 5000
Z95 = 1.959963984540054

CLASS_ORDER = ("mislead", "correct", "expand", "confirm", "unhelpful", "other change")
CONFIRM_SUBS = ("confirm_favoured", "confirm_differential_only")
LIKERT_LEVELS = (1, 2, 3, 4, 5)


def wilson(k: int, n: int, z: float = Z95) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Point estimate and Wilson score interval for k successes in n trials."""
    if n == 0:
        return None, None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def case_weights(n_cases: int, n_draws: int = N_DRAWS, seed: int = SEED) -> np.ndarray:
    """Whole-case bootstrap: (n_draws × n_cases) matrix of how often each case is drawn when
    n_cases cases are resampled with replacement. Generated from one seeded generator."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_cases, size=(n_draws, n_cases))
    w = np.zeros((n_draws, n_cases), dtype=np.int64)
    rows = np.repeat(np.arange(n_draws), n_cases)
    np.add.at(w, (rows, idx.ravel()), 1)
    return w


def bootstrap_ratio(num_by_case: Sequence[float], den_by_case: Sequence[float],
                    weights: np.ndarray) -> np.ndarray:
    """Statistic sum(num)/sum(den) recomputed in every draw. All readings of a drawn case (every
    reader's) enter together because num/den are per-case sums over readings."""
    num = np.asarray(num_by_case, dtype=float)
    den = np.asarray(den_by_case, dtype=float)
    return (weights @ num) / (weights @ den)


def percentile_ci(stats: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(stats, [a, 100 - a])
    return float(lo), float(hi)


def per_case(case_ids: Sequence[str], values: Sequence[float],
             case_order: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-case sum of values and count of readings, aligned to case_order."""
    pos = {c: i for i, c in enumerate(case_order)}
    num = np.zeros(len(case_order))
    den = np.zeros(len(case_order))
    for c, v in zip(case_ids, values):
        num[pos[c]] += float(v)
        den[pos[c]] += 1
    return num, den


def proportion_endpoint(case_ids: Sequence[str], flags: Sequence[bool], case_order: Sequence[str],
                        weights: np.ndarray) -> Dict[str, object]:
    flags = [bool(f) for f in flags]
    k, n = sum(flags), len(flags)
    num, den = per_case(case_ids, flags, case_order)
    lo, hi = percentile_ci(bootstrap_ratio(num, den, weights))
    p, wlo, whi = wilson(k, n)
    return {"k": k, "n": n, "estimate": p, "boot95_lo": lo, "boot95_hi": hi,
            "wilson95_lo": wlo, "wilson95_hi": whi}


def paired_difference(case_ids: Sequence[str], flags_x: Sequence[bool], flags_y: Sequence[bool],
                      case_order: Sequence[str], weights: np.ndarray) -> Dict[str, object]:
    """Paired difference Y − X in proportions; the difference is formed within each draw.
    Discordant pairs are returned as counts (no test)."""
    x = [int(bool(v)) for v in flags_x]
    y = [int(bool(v)) for v in flags_y]
    d = [b - a for a, b in zip(x, y)]
    num, den = per_case(case_ids, d, case_order)
    lo, hi = percentile_ci(bootstrap_ratio(num, den, weights))
    n = len(d)
    return {"n": n, "estimate": sum(d) / n if n else None, "boot95_lo": lo, "boot95_hi": hi,
            "x_wrong_y_right": sum(1 for a, b in zip(x, y) if a == 0 and b == 1),
            "x_right_y_wrong": sum(1 for a, b in zip(x, y) if a == 1 and b == 0)}


def class_table(classes: Iterable[str], confirm_subs: Iterable[str]) -> List[Dict[str, object]]:
    """Counts and proportions in CLASS_ORDER, with the confirm sub-counts (003 §B1) after confirm."""
    classes = list(classes)
    subs = list(confirm_subs)
    n = len(classes)
    unknown = set(classes) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(f"unknown outcome classes {unknown}")
    rows = []
    for c in CLASS_ORDER:
        k = classes.count(c)
        rows.append({"class": c, "count": k, "proportion": k / n if n else None})
        if c == "confirm":
            for s in CONFIRM_SUBS:
                ks = sum(1 for cl, sb in zip(classes, subs) if cl == "confirm" and sb == s)
                rows.append({"class": f"  {s}", "count": ks, "proportion": ks / n if n else None})
    return rows


def likert_summary(values: Iterable[Optional[float]], levels: Sequence[int] = LIKERT_LEVELS) -> Dict[str, object]:
    """Counts per level (zeros kept), n, median and quartiles; missing values are counted apart."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    out: Dict[str, object] = {f"n_{lv}": sum(1 for v in vals if v == lv) for lv in levels}
    other = [v for v in vals if v not in levels]
    if other:
        raise ValueError(f"values outside the Likert levels: {sorted(set(other))}")
    out["n"] = len(vals)
    if vals:
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        out.update({"median": float(med), "q1": float(q1), "q3": float(q3)})
    else:
        out.update({"median": None, "q1": None, "q3": None})
    return out


def median_iqr(values: Iterable[Optional[float]]) -> Dict[str, object]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    return {"n": len(vals), "median": float(med), "q1": float(q1), "q3": float(q3)}


def group_tertiles(group_counts: Dict[str, int]) -> Dict[str, str]:
    """Tertile (low/mid/high) of each library group by its support count, with cut points at the
    1/3 and 2/3 quantiles (linear interpolation) of the per-group counts; the group is the unit."""
    vals = np.array(list(group_counts.values()), dtype=float)
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    return {g: ("low" if v <= q1 else "mid" if v <= q2 else "high") for g, v in group_counts.items()}


def chrono_quartile(rank: int, n: int) -> int:
    """Quartile 1–4 of a 1-based chronological rank among n readings."""
    return int(math.ceil(rank * 4 / n))


_TERM_RE = re.compile(r"\b(discrepant|split|uncertain)\b", re.IGNORECASE)
_NEG_RE = re.compile(r"\b(no|none|not|without)\b", re.IGNORECASE)


def discrepancy_negation_only(text: str) -> bool:
    """True when the summary matches the literal llm_flags_discrepancy terms and every match is
    negated: a negation word (no/none/not/without) earlier in the same clause (clauses split at
    ';' and sentence ends). Reporting aid for 003 Amendment 001 item 6."""
    found = False
    for sent in re.split(r"(?<=[.!?])\s+", text):
        for clause in sent.split(";"):
            for m in _TERM_RE.finditer(clause):
                found = True
                if not _NEG_RE.search(clause[:m.start()]):
                    return False
    return found
