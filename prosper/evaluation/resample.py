"""Patient-clustered query resampling, extracted for reuse.

Why this module exists rather than a refactor of ``slide_eval.py``: that file is
a byte-identical copy of the script that produced the submitted manuscript's
numbers, and that identity is the whole reason gate G2's reproduction means
anything. Editing it to expose the sampler would trade a verified fidelity
property for a small convenience. So the sampler is extracted **verbatim** here
instead, and ``slide_eval.py`` is left alone.

The design, which is the one thing R4.a's "patient-level independent evaluation"
actually turns on: within each class, ``S_c`` **patients** are drawn with
replacement with probability proportional to their slide count, and then one
random slide is taken from each. That preserves the per-class slide count while
making the *patient* the resampling unit -- so a patient contributing 13 slides
cannot inflate the apparent sample size.

Scope limit, to be stated wherever these intervals are reported: this resamples
**queries only**. The reference library is held fixed, so library-side
variability is not in the interval.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def build_patient_index(meta: pd.DataFrame, id_col: str, label_col: str,
                        patient_col: str) -> Dict[str, dict]:
    """Per-class patient -> slides index, plus the S_c and P_c counts."""
    out: Dict[str, dict] = {}
    for c, g in meta.groupby(label_col, sort=True):
        patient_to_slides: Dict[str, List[str]] = {}
        for p, gg in g.groupby(patient_col, sort=True):
            patient_to_slides[str(p)] = gg[id_col].astype(str).tolist()
        out[str(c)] = {
            "patient_to_slides": patient_to_slides,
            "weights": {p: len(s) for p, s in patient_to_slides.items()},
            "S": int(len(g)),                    # slides in class
            "P": int(len(patient_to_slides)),    # patients in class
        }
    return out


def sample_queries(index: Dict[str, dict], labels_order: Sequence[str],
                   rng: np.random.Generator,
                   mode: str = "pps_cluster_one_slide") -> List[str]:
    """One replicate's worth of sampled query slide IDs.

    Extracted verbatim from ``slide_eval._bootstrap_worker``'s inner
    ``sample_queries_for_class``; the ``w.sum() == 0`` uniform fallback is kept
    even though it cannot trigger for real slide counts, because dropping a
    branch is a behaviour change.
    """
    sample_ids: List[str] = []
    for c in labels_order:
        entry = index.get(str(c))
        if entry is None:
            continue
        patients = list(entry["patient_to_slides"].keys())
        if not patients:
            continue
        if mode == "pps_cluster_one_slide":
            n_draws = entry["S"]
            w = np.array([entry["weights"][p] for p in patients], dtype=float)
            w = np.ones_like(w) / len(w) if w.sum() == 0 else (w / w.sum())
            chosen = rng.choice(len(patients), size=n_draws, replace=True, p=w)
        elif mode == "one_per_patient":
            n_draws = entry["P"]
            if n_draws == 0:
                continue
            chosen = rng.choice(len(patients), size=n_draws, replace=True)
        else:
            raise ValueError(
                "mode must be 'pps_cluster_one_slide' or 'one_per_patient'")
        for idx in chosen:
            slides = entry["patient_to_slides"][patients[idx]]
            sample_ids.append(slides[rng.integers(len(slides))])
    return sample_ids
