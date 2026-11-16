from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Tuple

SchemaCase = Dict[str, Any]


def _extract_score_values(neighbors: List[SchemaCase]) -> Tuple[str, List[float], str]:
    """Return (score_name, values, direction).

    If any neighbor provides `probability`, prefer it and treat higher as better.
    Otherwise fall back to `similarity` and treat lower as better.
    """
    if any(n.get("probability") is not None for n in neighbors):
        values = [float(n.get("probability", 0.0)) for n in neighbors]
        return "probability", values, "higher"
    values = [float(n.get("similarity", 0.0)) for n in neighbors]
    return "similarity", values, "lower"


def _top_score_case(neighbors: List[SchemaCase], score_name: str, direction: str) -> Dict[str, Any]:
    if not neighbors:
        return {}
    key = score_name
    # Choose max for probability, min for similarity
    best = None
    for n in neighbors:
        if n.get(key) is None:
            continue
        value = float(n.get(key, 0.0))
        if best is None:
            best = (value, n)
        else:
            if direction == "higher" and value > best[0]:
                best = (value, n)
            if direction == "lower" and value < best[0]:
                best = (value, n)
    if best is None:
        return {}
    value, n = best
    return {
        "top_score": value,
        "top_case_id": n.get("case_id"),
        "top_diagnosis": n.get("diagnosis"),
        "top_tumor_type": n.get("tumor_type") or n.get("tumor_group"),
    }


def _mode_or_none(values: List[str | None]) -> str | None:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    counts = Counter(filtered)
    # If tie, return first encountered in sorted order for determinism
    max_count = max(counts.values())
    top = [v for v, c in counts.items() if c == max_count]
    return sorted(top)[0]


def compute_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compute reusable stats from the payload for summarization and optional LLM rewrite."""
    neighbors: List[SchemaCase] = payload.get("neighbors", [])
    if len(neighbors) != 5:
        raise ValueError("Expected exactly 5 neighbors")
    diagnoses = [n.get("diagnosis") for n in neighbors]
    mode_dx = _mode_or_none(diagnoses)
    score_name, scores, direction = _extract_score_values(neighbors)
    mean_score = sum(scores) / len(scores)
    min_score, max_score = min(scores), max(scores)
    outlier_details: List[Dict[str, Any]] = []
    for i, (n, score) in enumerate(zip(neighbors, scores)):
        dx = n.get("diagnosis")
        if mode_dx is not None and dx != mode_dx:
            outlier_details.append({
                "index": i,
                "case_id": n.get("case_id"),
                "diagnosis": dx,
                "score": score,
                "diagnosis_differs": True
            })
    counts = Counter(d for d in diagnoses if d is not None)
    top_info = _top_score_case(neighbors, score_name, direction)
    return {
        "mode_diagnosis": mode_dx,
        "diagnosis_counts": dict(counts),
        "score_name": score_name,
        "score_direction": direction,
        "mean_score": mean_score,
        "min_score": min_score,
        "max_score": max_score,
        "mean_similarity": mean_score if score_name == "similarity" else None,
        "min_similarity": min_score if score_name == "similarity" else None,
        "max_similarity": max_score if score_name == "similarity" else None,
        "mean_probability": mean_score if score_name == "probability" else None,
        "min_probability": min_score if score_name == "probability" else None,
        "max_probability": max_score if score_name == "probability" else None,
        "top_score": top_info.get("top_score"),
        "top_case_id": top_info.get("top_case_id"),
        "top_diagnosis": top_info.get("top_diagnosis"),
        "top_tumor_type": top_info.get("top_tumor_type"),
        "outlier_count": len(outlier_details),
        "outliers": outlier_details,
        "total_neighbors": len(neighbors),
        "current_case": payload.get("current_case", {}),
    }


def summarize_neighbors(payload: Dict[str, Any]) -> str:
    """Produce a <=25 word cautious summary (deterministic)."""
    stats = compute_stats(payload)
    mode_dx = stats["mode_diagnosis"]
    score_name = stats.get("score_name", "similarity")
    direction = stats.get("score_direction", "lower")
    mean_sim = stats.get("mean_score", stats.get("mean_similarity", 0.0))
    min_sim = stats.get("min_score", stats.get("min_similarity", 0.0))
    max_sim = stats.get("max_score", stats.get("max_similarity", 0.0))
    outlier_count = stats["outlier_count"]
    if score_name == "probability":
        strength_word = "supports" if mean_sim >= 0.5 else "suggests"
    else:
        strength_word = "supports" if mean_sim > 0.75 else "suggests"
    parts: List[str] = []
    if mode_dx:
        parts.append(f"Neighbors {strength_word} {mode_dx}")
    else:
        parts.append("Neighbors show no dominant diagnosis")
    better_phrase = "higher is better" if direction == "higher" else "lower is better"
    parts.append(f"{score_name} mean {mean_sim:.2f} (range {min_sim:.2f}-{max_sim:.2f}, {better_phrase})")
    if outlier_count:
        parts.append(f"{outlier_count} outlier(s)")
    else:
        parts.append("no outliers")
    sentence = ", ".join(parts)
    words = sentence.split()
    if len(words) > 25:
        trimmed = ", ".join(parts[:-1])
        if len(trimmed.split()) <= 25:
            sentence = trimmed
        if len(sentence.split()) > 25:
            sentence = " ".join(sentence.split()[:25]) + "…"
    return sentence
