"""Tests for C1b: patient-excluded ROI-level evaluation.

These are synthetic, small-n tests -- there is no cluster data available at
test time -- and check three things that a fidelity check cannot: (1) the
patient filter actually removes same-patient hits from the *counted* pool,
not just from some intermediate list; (2) the published voting/tie-break
functions are the literal objects reused, not reimplementations that happen
to agree on these examples; (3) with an empty/absent patient map the new
function's behaviour degenerates to nothing being filtered (the shape the
"regression" argument in roi_cli.py depends on, even though the regression
check itself runs the *original*, untouched ``predict_slide_from_roi_neighbors``
rather than this function with an empty map).
"""

import os
import pickle
import tempfile

import pandas as pd
import pytest

from prosper.evaluation import roi_eval as R


def _write_pickle(d, slide_id, df):
    path = os.path.join(d, f"top_10_hit_slides_slide_{slide_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return path


def test_reuses_the_published_voting_and_tie_break_functions():
    """Not a behavioural check -- an identity check. If someone reimplements
    the tie-break inside the patient-excluded path, this catches it even if
    the reimplementation happens to agree numerically."""
    import inspect

    src = inspect.getsource(R.predict_slide_from_roi_neighbors_patient_excluded)
    assert "_tie_break_by_counts_lex(" in src
    assert "_dedup_preserve_order(" in src
    # And they must be the *same* functions the published path calls.
    pub_src = inspect.getsource(R.predict_slide_from_roi_neighbors)
    assert "_tie_break_by_counts_lex(" in pub_src
    assert "_dedup_preserve_order(" in pub_src


def test_same_patient_hits_are_removed_from_the_counted_pool():
    with tempfile.TemporaryDirectory() as d:
        # Query 100 (patient P1) has a same-patient sibling 101 (patient P1)
        # ranked ahead of a different-patient hit 200 in every ROI. Without
        # filtering, 101 dominates the vote (2 classes tied only by luck); with
        # filtering, only 200's label can be counted.
        df = pd.DataFrame({
            "roi_id": [0, 0, 1, 1],
            "hit_slide_id": [101, 200, 101, 200],
            "query_slide_id": [100, 100, 100, 100],
        })
        _write_pickle(d, 100, df)

        slide_to_label = {"100": "A", "101": "B", "200": "C"}
        slide_to_patient = {"100": "P1", "101": "P1", "200": "P2"}

        rec = R.predict_slide_from_roi_neighbors_patient_excluded(
            "100", d, slide_to_label, slide_to_patient,
            top_k_per_roi=5, vote_mode="raw_hits")

        assert rec["same_patient_hits_dropped"] == 2
        assert rec["same_patient_hits_surviving"] == 0
        assert rec["total_hits_counted"] == 2          # only the two "200" hits
        assert rec["pred_label"] == "C"                # sibling can no longer win


def test_hit_at_5_membership_is_over_final_vote_counts():
    with tempfile.TemporaryDirectory() as d:
        # Five distinct-patient classes each get exactly 1 vote; a 6th class
        # gets 0. True label "F" never appears -> hit@5 should be 0 (only 5
        # slots, F isn't one of the 5 non-empty classes... construct so true
        # label is the 6th-ranked class by leaving it with the lowest count).
        df = pd.DataFrame({
            "roi_id": [0] * 6,
            "hit_slide_id": [10, 20, 30, 40, 50, 60],
            "query_slide_id": [999] * 6,
        })
        _write_pickle(d, 999, df)
        slide_to_label = {"999": "F", "10": "A", "20": "B", "30": "C",
                          "40": "D", "50": "E", "60": "G"}
        slide_to_patient = {str(k): f"P{k}" for k in
                            (999, 10, 20, 30, 40, 50, 60)}  # all distinct

        rec = R.predict_slide_from_roi_neighbors_patient_excluded(
            "999", d, slide_to_label, slide_to_patient,
            top_k_per_roi=6, vote_mode="raw_hits")
        assert rec["hit_at_5"] == 0   # true label F never among the 6 hit classes

        # Now put the true label's class among the hits -> hit@5 must flip.
        slide_to_label["10"] = "F"
        rec2 = R.predict_slide_from_roi_neighbors_patient_excluded(
            "999", d, slide_to_label, slide_to_patient,
            top_k_per_roi=6, vote_mode="raw_hits")
        assert rec2["hit_at_5"] == 1


def test_missing_patient_lookup_does_not_filter():
    """If a slide isn't in slide_to_patient (shouldn't happen with the real,
    complete metadata -- [decision record] found 0 NaNs over 2,921 rows), the function
    must not crash and must not filter a hit it cannot classify -- silently
    dropping unmatched hits would understate the leak, not remove it."""
    with tempfile.TemporaryDirectory() as d:
        df = pd.DataFrame({
            "roi_id": [0],
            "hit_slide_id": [200],
            "query_slide_id": [100],
        })
        _write_pickle(d, 100, df)
        slide_to_label = {"100": "A", "200": "C"}
        rec = R.predict_slide_from_roi_neighbors_patient_excluded(
            "100", d, slide_to_label, slide_to_patient={},  # no patient info at all
            top_k_per_roi=5, vote_mode="raw_hits")
        assert rec["same_patient_hits_dropped"] == 0
        assert rec["total_hits_counted"] == 1
        assert rec["pred_label"] == "C"


def test_load_metadata_with_patient_col_is_additive():
    """paths.roi_metadata_legacy()-style calls (patient_col=None, the G3
    default) must be byte-for-byte unaffected by the new parameter."""
    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, "meta.csv")
        pd.DataFrame({
            "Slide.Scan.File": [100, 101, 200],
            "Tumor.group": ["A", "A", "C"],
            "UID.Subject": ["P1", "P1", "P2"],
        }).to_csv(csv, index=False)

        legacy = R.load_metadata(csv, drop_labels=["MISC"])
        assert list(legacy.columns) == ["Slide.Scan.File", "Tumor.group"]

        with_patient = R.load_metadata(csv, drop_labels=["MISC"],
                                       patient_col="UID.Subject")
        assert list(with_patient.columns) == \
            ["Slide.Scan.File", "Tumor.group", "UID.Subject"]
        assert len(with_patient) == len(legacy) == 3
