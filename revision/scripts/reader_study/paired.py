"""Between-reader analysis on the paired set (plan 002 §5.7; 003 Amendment 004 item 4). Descriptive only."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import pandas as pd

import derive as d

KAPPA_ONE_LABEL = "only one label used by both readers; kappa undefined"
KAPPA_EMPTY = "empty paired set"


def paired_case_ids(cases_r1: Sequence[str], cases_r2: Sequence[str]) -> List[str]:
    """002 §2 paired set: cases with A and B submitted by both readers (one readings row per such case)."""
    return sorted(set(cases_r1) & set(cases_r2))


def cohen_kappa(labels1: Sequence[str], labels2: Sequence[str],
                categories: Sequence[str] = d.TUMOUR_GROUPS) -> Tuple[float, str]:
    """Unweighted Cohen's kappa over the closed-list categories. Returns (kappa, note)."""
    if len(labels1) != len(labels2):
        raise ValueError("label sequences differ in length")
    n = len(labels1)
    if n == 0:
        return math.nan, KAPPA_EMPTY
    # Free-text (off-list) entries are folded into MISC (other) for kappa only (003 Amendment 006 item 4);
    # the strict rule never matches them, and the raw strings stay in the per-case table.
    cats = set(categories)
    l1 = [x if x in cats else d.MISC for x in labels1]
    l2 = [x if x in cats else d.MISC for x in labels2]
    n_off = sum(x not in cats for x in list(labels1) + list(labels2))
    note = f"{n_off} off-list label(s) folded into {d.MISC}" if n_off else ""
    if len(set(l1) | set(l2)) == 1:
        return math.nan, KAPPA_ONE_LABEL if not note else f"{KAPPA_ONE_LABEL}; {note}"
    p_o = sum(a == b for a, b in zip(l1, l2)) / n
    p_e = sum((l1.count(c) / n) * (l2.count(c) / n) for c in categories)
    return (p_o - p_e) / (1 - p_e), note


def agreement_summary(fav1: Sequence[str], fav2: Sequence[str],
                      correct1: Sequence[bool], correct2: Sequence[bool]) -> Dict[str, object]:
    """Raw agreement of the favoured diagnosis, kappa, and the four-way favoured-correct counts."""
    if not (len(fav1) == len(fav2) == len(correct1) == len(correct2)):
        raise ValueError("inputs differ in length")
    n = len(fav1)
    agree = sum(a == b for a, b in zip(fav1, fav2))
    kappa, note = cohen_kappa(fav1, fav2)
    c1 = [bool(x) for x in correct1]
    c2 = [bool(x) for x in correct2]
    return {"n": n, "n_agree": agree, "raw_agreement": agree / n if n else None,
            "cohen_kappa": kappa, "kappa_note": note,
            "both_correct": sum(a and b for a, b in zip(c1, c2)),
            "only_reader1_correct": sum(a and not b for a, b in zip(c1, c2)),
            "only_reader2_correct": sum(b and not a for a, b in zip(c1, c2)),
            "neither_correct": sum(not a and not b for a, b in zip(c1, c2))}


def _present(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and v != ""


def between_reader_tables(r: pd.DataFrame, readers: Sequence[str],
                          tiers: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per reader pair (in --readers order): per-step summary and per-case table on the paired set.
    A step enters a case only when both readers have an effective answer at that step."""
    srows, crows = [], []
    for r1, r2 in combinations(readers, 2):
        a = r[r.reader_id == r1].set_index("case_id")
        b = r[r.reader_id == r2].set_index("case_id")
        pc = paired_case_ids(a.index, b.index)
        for p in d.PHASES:
            ok = [c for c in pc if _present(a.at[c, f"favoured_{p}"]) and _present(b.at[c, f"favoured_{p}"])]
            s = agreement_summary([a.at[c, f"favoured_{p}"] for c in ok], [b.at[c, f"favoured_{p}"] for c in ok],
                                  [a.at[c, f"fav_correct_{p}"] for c in ok], [b.at[c, f"fav_correct_{p}"] for c in ok])
            srows.append({"reader1_id": r1, "reader1_tier": tiers[r1], "reader2_id": r2, "reader2_tier": tiers[r2],
                          "paired_set_n": len(pc), "step": p, **s})
        for c in pc:
            row = {"reader1_id": r1, "reader2_id": r2, "case_id": c, "cohort": a.at[c, "cohort"],
                   "reference": a.at[c, "reference"], "hit5": a.at[c, "hit5"]}
            for tag, x in (("r1", a), ("r2", b)):
                for p in d.PHASES:
                    row[f"{tag}_favoured_{p}"] = x.at[c, f"favoured_{p}"]
                for p in d.PHASES:
                    row[f"{tag}_fav_correct_{p}"] = x.at[c, f"fav_correct_{p}"]
                row[f"{tag}_class_AB"] = x.at[c, "class_AB"]
            crows.append(row)
    scols = ["reader1_id", "reader1_tier", "reader2_id", "reader2_tier", "paired_set_n", "step", "n", "n_agree",
             "raw_agreement", "cohen_kappa", "kappa_note", "both_correct", "only_reader1_correct",
             "only_reader2_correct", "neither_correct"]
    ccols = ["reader1_id", "reader2_id", "case_id", "cohort", "reference", "hit5"]
    for t in ("r1", "r2"):
        ccols += [f"{t}_favoured_{p}" for p in d.PHASES] + [f"{t}_fav_correct_{p}" for p in d.PHASES]
        ccols.append(f"{t}_class_AB")
    return pd.DataFrame(srows, columns=scols), pd.DataFrame(crows, columns=ccols)
