"""Derived variables of the PROSPER reader study (plan candidate 002 §3 as amended by 003 §B).

Every function here is pure: it takes already-loaded values and returns derived values, so the
rules can be unit-tested on constructed examples before any real reading is touched.
"""
from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TUMOUR_GROUPS = (
    "ACT/ACC", "DSRCT", "EWS", "LIVER", "MEL", "MPNST", "MRT", "NB", "NF", "NORMAL",
    "OS", "RB", "RMS", "SS", "WT", "MISC (other)",
)
MISC = "MISC (other)"
NEVER_MATCH = frozenset({MISC, "NORMAL"})
PHASES = ("A", "B", "C")

from reader_config import load_reader_config  # noqa: E402

# Reader logins live in a local, git-ignored config (see reader_config.py).
READER_CONFIG = load_reader_config()

# Expertise tier per reader (003 §A, §8.6 roster). Readers missing here cannot be analysed.
EXPERTISE_TIER = dict(READER_CONFIG["expertise_tier"])
# Excluded by name regardless of `role` (003 §A; Amendment 001 item 3).
EXCLUDED_READERS = frozenset(READER_CONFIG["excluded_readers"])
EXCLUDED_PREFIXES = ("demo-",)
PILOT_CASES = frozenset({"case01", "case02"})

INTERRUPT_S = 1800.0
SESSION_GAP_S = 1800.0
# 2026-10-26 16:28:33 CDT: restart that exposed STV age/site in Step 1 (002 §3.7).
META_VISIBLE_T = 1793050113.0

RETRIEVAL_KINDS = ("roi_selected", "hit_opened", "neighbour_opened", "search_mode_changed")


def is_excluded_reader(reader_id: str) -> bool:
    return reader_id in EXCLUDED_READERS or reader_id.startswith(EXCLUDED_PREFIXES)


# ---------------------------------------------------------------- §3.1 effective answer

@dataclass(frozen=True)
class Answer:
    dx: Tuple[str, ...]
    favoured: str
    source_phase: str


def make_answer(dx: Sequence[str], favoured: str, source_phase: str) -> Answer:
    """Compact the differential as the instrument does and check the favoured entry is in it."""
    entries = tuple(d.strip() for d in dx if d and d.strip())
    fav = (favoured or "").strip()
    if not entries:
        raise ValueError(f"phase {source_phase}: empty differential")
    if fav not in entries:
        raise ValueError(f"phase {source_phase}: favoured {fav!r} not in differential")
    if len({d.lower() for d in entries}) != len(entries):
        raise ValueError(f"phase {source_phase}: duplicate differential entries")
    return Answer(entries, fav, source_phase)


def effective_answers(ans_a: Answer, changed_b: str, ans_b: Optional[Answer],
                      changed_c: Optional[str], ans_c: Optional[Answer]) -> Dict[str, Optional[Answer]]:
    """eff_A = Step 1; eff_B = Step 2 if changed else eff_A; eff_C = Step 3 if changed else eff_B.
    `changed_c` None means Step 3 was not submitted (eff_C is then None)."""
    def step(prev: Answer, changed: str, ans: Optional[Answer], phase: str) -> Answer:
        if changed == "yes":
            if ans is None:
                raise ValueError(f"phase {phase}: changed=yes without an answer")
            return ans
        if changed == "no":
            if ans is not None:
                raise ValueError(f"phase {phase}: changed=no but an answer is stored")
            return prev
        raise ValueError(f"phase {phase}: changed_assessment must be 'yes' or 'no', got {changed!r}")

    eff_b = step(ans_a, changed_b, ans_b, "B")
    eff_c = step(eff_b, changed_c, ans_c, "C") if changed_c is not None else None
    return {"A": ans_a, "B": eff_b, "C": eff_c}


# ---------------------------------------------------------------- §3.2 matching

def score_answer(ans: Answer, reference: str) -> Dict[str, object]:
    """Strict closed-list matching: exact string equality; MISC/NORMAL never match."""
    if reference in NEVER_MATCH or reference not in TUMOUR_GROUPS:
        raise ValueError(f"reference {reference!r} is not a matchable library group")
    rank = next((i + 1 for i, d in enumerate(ans.dx) if d == reference), None)
    return {
        "fav_correct": ans.favoured == reference,
        "in_diff": rank is not None,
        "truth_rank": rank,
        "n_dx": len(ans.dx),
    }


@dataclass(frozen=True)
class EntityMapRow:
    entity_as_written: str
    normalised_entity: str
    mapped_group: str
    patterns: Tuple[str, ...]


def _phrase_regex(phrase: str) -> str:
    words = [re.escape(w) for w in phrase.strip().split()]
    return r"(?<![A-Za-z])" + r"\s+".join(words) + r"(?![A-Za-z])"


def entity_patterns(entity_as_written: str, normalised_entity: str) -> Tuple[str, ...]:
    """Text forms a comment must contain for a map row to apply: the written entity, its
    ' / '-separated alternatives, and the normalised form (case-insensitive, whitespace-tolerant)."""
    forms = [entity_as_written.strip()]
    forms += [p.strip() for p in re.split(r"\s+/\s+", entity_as_written) if p.strip()]
    forms.append(normalised_entity.strip())
    seen: List[str] = []
    for f in forms:
        if f and f.lower() not in (s.lower() for s in seen):
            seen.append(f)
    return tuple(_phrase_regex(f) for f in seen)


def load_entity_map(path: Path) -> List[EntityMapRow]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            group = r["mapped_group"].strip()
            if group != "outside_vocabulary" and (group not in TUMOUR_GROUPS or group in NEVER_MATCH):
                raise ValueError(f"entity map row {r['entity_as_written']!r}: bad group {group!r}")
            rows.append(EntityMapRow(r["entity_as_written"], r["normalised_entity"], group,
                                     entity_patterns(r["entity_as_written"], r["normalised_entity"])))
    return rows


def comment_entities(comment: str, entity_map: Sequence[EntityMapRow]) -> List[EntityMapRow]:
    text = comment or ""
    return [row for row in entity_map
            if any(re.search(p, text, flags=re.IGNORECASE) for p in row.patterns)]


def apply_entity_map(ans: Answer, comment: str,
                     entity_map: Sequence[EntityMapRow]) -> Tuple[Answer, str]:
    """003 §D sensitivity: the MISC entry is replaced by the mapped group only when the comment of
    the phase the entry was written in names mapped entities that all point to one library group.
    Returns (answer, note). Comments naming entities with different outcomes leave it unmatched."""
    if MISC not in ans.dx:
        return ans, ""
    hits = comment_entities(comment, entity_map)
    if not hits:
        return ans, "misc_no_named_entity"
    outcomes = {h.mapped_group for h in hits}
    names = ";".join(h.normalised_entity for h in hits)
    if len(outcomes) > 1:
        return ans, f"misc_ambiguous:{names}"
    group = outcomes.pop()
    if group == "outside_vocabulary":
        return ans, f"misc_outside_vocabulary:{names}"
    dx = tuple(group if d == MISC else d for d in ans.dx)
    fav = group if ans.favoured == MISC else ans.favoured
    return Answer(dx, fav, ans.source_phase), f"misc_mapped:{names}->{group}"


# ---------------------------------------------------------------- §3.3 outcome class

def outcome_class(fav_x: bool, in_diff_x: bool, fav_y: bool, in_diff_y: bool,
                  changed: bool) -> Tuple[str, str]:
    """Class of a transition X→Y (rules evaluated in order) and the 003 §B(1) confirm sub-count."""
    if (fav_x and not fav_y) or (in_diff_x and not in_diff_y):
        return "mislead", ""
    if not fav_x and fav_y:
        return "correct", ""
    if not in_diff_x and in_diff_y and not fav_y:
        return "expand", ""
    if not changed and in_diff_x:
        return "confirm", ("confirm_favoured" if fav_x else "confirm_differential_only")
    if not changed and not in_diff_x:
        return "unhelpful", ""
    return "other change", ""


# ---------------------------------------------------------------- §3.4 confidence/usefulness

def confidence_vars(conf_a: int, conf_b: int, conf_c: Optional[int], fav_correct_b: bool) -> Dict[str, object]:
    return {
        "dconf_BA": conf_b - conf_a,
        "dconf_CB": (conf_c - conf_b) if conf_c is not None else None,
        "harmful_conf_B": bool(conf_b > conf_a and not fav_correct_b),
    }


# ---------------------------------------------------------------- §3.5 reading time

def reading_time_vars(times: Dict[str, Tuple[float, float]]) -> Dict[str, object]:
    """times[phase] = (t_open, t_submit). sec_X = t_submit - t_open; > 1800 s flags an interrupted
    phase; 003 §B(5): a > 1800 s gap between consecutive phases flags interrupted_between_phases."""
    out: Dict[str, object] = {}
    for p in PHASES:
        if p in times:
            t_open, t_submit = times[p]
            sec = t_submit - t_open
            out[f"sec_{p}"] = sec
            out[f"interrupted_{p}"] = bool(sec > INTERRUPT_S)
        else:
            out[f"sec_{p}"] = None
            out[f"interrupted_{p}"] = None
    gap_ab = times["B"][0] - times["A"][1] if "A" in times and "B" in times else None
    gap_bc = times["C"][0] - times["B"][1] if "B" in times and "C" in times else None
    out["gap_AB_s"] = gap_ab
    out["gap_BC_s"] = gap_bc
    out["interrupted_between_phases"] = bool((gap_ab is not None and gap_ab > INTERRUPT_S)
                                             or (gap_bc is not None and gap_bc > INTERRUPT_S))
    return out


# ---------------------------------------------------------------- §3.6 engagement

def is_auto(payload: dict) -> bool:
    return payload.get("auto") is True


def event_windows(ts: float, t_submit_a: float, t_open_b: float, t_submit_b: float,
                  t_open_c: Optional[float], t_submit_c: Optional[float]) -> Dict[str, bool]:
    """Step 2 window: t_open_B <= ts <= t_submit_B, plus the A→B gap (t_submit_A < ts < t_open_B),
    which 003 §B(2) attributes to B. Step 3 window: t_open_C <= ts <= t_submit_C. Windows are
    evaluated independently, exactly as written."""
    in_b = (t_open_b <= ts <= t_submit_b) or (t_submit_a < ts < t_open_b)
    in_c = t_open_c is not None and t_submit_c is not None and t_open_c <= ts <= t_submit_c
    return {"in_B_window": bool(in_b), "in_C_window": bool(in_c)}


def phase_label(ts: float, t_submit_a: float, t_open_b: float, t_submit_b: float,
                t_open_c: Optional[float], t_submit_c: Optional[float]) -> str:
    """Descriptive position of an event on the reading's timeline (for the event table only)."""
    if ts <= t_submit_a:
        return "A"
    if ts <= t_submit_b:
        return "B"
    if t_open_c is None or t_submit_c is None:
        return "after_B"
    if ts < t_open_c:
        return "gap_BC"
    if ts <= t_submit_c:
        return "C"
    return "after_C"


def engagement_vars(events: Iterable[dict], t_submit_a: float, t_open_b: float, t_submit_b: float,
                    t_open_c: Optional[float], t_submit_c: Optional[float]) -> Dict[str, object]:
    """events: dicts with `kind`, `ts`, `payload` (dict) for one reader × case."""
    counts = Counter()
    ranks: List[int] = []
    n_c_retrieval = 0
    for e in events:
        if is_auto(e["payload"]):
            continue
        w = event_windows(e["ts"], t_submit_a, t_open_b, t_submit_b, t_open_c, t_submit_c)
        kind = e["kind"]
        if w["in_B_window"]:
            counts[kind] += 1
            if kind in ("neighbour_opened", "hit_opened") and e["payload"].get("rank") is not None:
                ranks.append(int(e["payload"]["rank"]))
        if w["in_C_window"] and kind in RETRIEVAL_KINDS:
            n_c_retrieval += 1
    out = {
        "n_roi_selected": counts["roi_selected"],
        "n_hit_opened": counts["hit_opened"],
        "n_neighbour_opened": counts["neighbour_opened"],
        "n_heatmap_toggled": counts["heatmap_toggled"],
        "n_search_mode_changed": counts["search_mode_changed"],
        "n_pecan_opened": counts["pecan_opened"],
    }
    out["engaged_B"] = bool(out["n_hit_opened"] + out["n_neighbour_opened"] + out["n_roi_selected"] > 0)
    out["deepest_rank_opened"] = max(ranks) if ranks else None
    out["n_retrieval_events_C"] = n_c_retrieval
    out["engaged_C_retrieval"] = bool(n_c_retrieval > 0)
    return out


# ---------------------------------------------------------------- §3.7 case covariates

def neighbour_retrieval(labels_by_rank: Sequence[str], truth: str) -> Dict[str, object]:
    """labels_by_rank: slide-level neighbour labels in rank order (rank 1 first). Plurality rule of
    003 §B(3): the single most frequent of the ten labels equals the truth; a tie for most frequent
    is false."""
    top10 = list(labels_by_rank[:10])
    counts = Counter(top10)
    plurality = ""
    if counts:
        best = max(counts.values())
        leaders = [g for g, n in counts.items() if n == best]
        plurality = leaders[0] if len(leaders) == 1 else ""
    return {
        "top1_correct": bool(top10[:1] == [truth]),
        "hit5": truth in top10[:5],
        "hit10": truth in top10,
        "n_truth_top10": counts.get(truth, 0),
        "plurality10_label": plurality,
        "majority10_correct": plurality == truth,
        "n_neighbours": len(top10),
    }


def roi_success(rank1_labels: Sequence[Optional[str]], truth: str) -> Dict[str, object]:
    """Fraction of precomputed ROIs whose rank-1 hit label equals the truth. ROIs without hits are
    counted in the denominator as not successful and reported separately."""
    n = len(rank1_labels)
    n_without = sum(1 for lab in rank1_labels if lab is None)
    ok = sum(1 for lab in rank1_labels if lab == truth)
    return {"n_rois": n, "n_rois_without_hits": n_without,
            "roi_success_frac": (ok / n) if n else None}


def library_support(labels: Iterable[str]) -> Dict[str, int]:
    return dict(Counter(labels))


def meta_visible_a(t_open_a: float, cohort: str) -> bool:
    return bool(cohort == "CCDI" or t_open_a > META_VISIBLE_T)


def chronological_rank(t_open_a: Sequence[float]) -> List[int]:
    order = sorted(range(len(t_open_a)), key=lambda i: t_open_a[i])
    ranks = [0] * len(t_open_a)
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    return ranks


def session_ids(t_submits: Sequence[float], gap: float = SESSION_GAP_S) -> List[int]:
    """Session number (1-based) of each submission: a gap > `gap` seconds between consecutive
    submissions (sorted by time) starts a new session. Returned in the input order."""
    order = sorted(range(len(t_submits)), key=lambda i: t_submits[i])
    ids = [0] * len(t_submits)
    session, prev = 0, None
    for i in order:
        if prev is None or t_submits[i] - prev > gap:
            session += 1
        ids[i] = session
        prev = t_submits[i]
    return ids


def nan_to_none(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v
