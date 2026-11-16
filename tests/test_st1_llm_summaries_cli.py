"""Pure checks for st1_llm_summaries_cli helpers (no LLM, no comet-llm import)."""
import numpy as np
import pandas as pd
import pytest

from prosper.evaluation.st1_llm_summaries_cli import bayesian_probabilities, exclude_library_groups, resumable_done


def _block():
    # 2 queries x 6 library slides; distances chosen so that the nearest slide is NORMAL.
    cols = ["n1", "m1", "a1", "a2", "b1", "b2"]
    sim = pd.DataFrame([[0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
                        [0.15, 0.60, 0.50, 0.40, 0.30, 0.20]], index=["q1", "q2"], columns=cols)
    true_of = {"n1": "NORMAL", "m1": "MISC", "a1": "RMS", "a2": "RMS", "b1": "ews", "b2": "EWS",
               "q1": "RMS", "q2": "EWS"}
    return sim, true_of


def test_exclude_drops_only_named_groups_case_insensitively():
    sim, true_of = _block()
    out, info = exclude_library_groups(sim, true_of, ["normal", "MISC"])
    assert list(out.columns) == ["a1", "a2", "b1", "b2"]
    assert info == {"groups": ["MISC", "NORMAL"], "columns_total": 6, "columns_kept": 4,
                    "dropped": {"NORMAL": 1, "MISC": 1}}
    assert list(out.index) == ["q1", "q2"]            # query rows untouched
    assert out.loc["q1", "a1"] == 0.30


def test_exclude_noop_when_no_groups():
    sim, true_of = _block()
    out, info = exclude_library_groups(sim, true_of, [])
    assert out is sim and info["columns_kept"] == 6 and info["groups"] == []


def test_exclude_refuses_unlabelled_columns_and_unknown_groups():
    sim, true_of = _block()
    with pytest.raises(SystemExit, match="no tumor_group label"):
        exclude_library_groups(sim, {k: v for k, v in true_of.items() if k != "a2"}, ["NORMAL"])
    with pytest.raises(SystemExit, match="match no library column"):
        exclude_library_groups(sim, true_of, ["NORMAL", "LIVER"])


def test_bayesian_uses_only_remaining_columns(tmp_path):
    sim, true_of = _block()
    (tmp_path / "full").mkdir()
    (tmp_path / "filt").mkdir()
    full = bayesian_probabilities(sim, ["q1", "q2"], true_of, n=2, T=5.0, out=tmp_path / "full")
    filt_sim, _ = exclude_library_groups(sim, true_of, ["NORMAL", "MISC"])
    filt = bayesian_probabilities(filt_sim, ["q1", "q2"], true_of, n=2, T=5.0, out=tmp_path / "filt")
    assert "NORMAL" in full["q1"] and max(full["q1"], key=full["q1"].get) == "NORMAL"
    assert "NORMAL" not in filt["q1"] and "MISC" not in filt["q1"]
    assert max(filt["q1"], key=filt["q1"].get) == "RMS"
    assert max(filt["q2"], key=filt["q2"].get) == "EWS"
    for d in filt.values():
        assert abs(sum(d.values()) - 1.0) < 2e-3 and all(0 < v <= 1 for v in d.values())
        assert all(v == round(v, 3) for v in d.values())
    assert np.isclose(pd.read_csv(tmp_path / "filt" / "disease_probabilities.csv", index_col=0).sum(axis=1), 1).all()


def test_resumable_done_refuses_other_settings_and_legacy_records(tmp_path):
    import json
    p = tmp_path / "runs.jsonl"
    s = {"fake": False, "k": 10, "bayesian_n": 10, "bayesian_t": 5.0,
         "exclude_groups": ["MISC", "NORMAL"], "columns_used": 2604, "summarizer": "manuscript"}
    assert resumable_done(p, s) == set()                                   # no file yet
    p.write_text(json.dumps({"query": "q1", "arm": "packet", "settings": s}) + "\n")
    assert resumable_done(p, s) == {("q1", "packet")}                      # same settings: skip
    with pytest.raises(SystemExit, match="use a fresh --out-dir"):         # a --fake pass in the same dir
        resumable_done(p, {**s, "fake": True})
    with pytest.raises(SystemExit, match="use a fresh --out-dir"):         # different T
        resumable_done(p, {**s, "bayesian_t": 20.0})
    p.write_text(json.dumps({"query": "q1", "arm": "packet"}) + "\n")      # legacy record, no settings
    with pytest.raises(SystemExit, match="use a fresh --out-dir"):
        resumable_done(p, s)


def test_load_label_map_keeps_first_duplicate_row_and_sample_id_fallback(tmp_path):
    """One scanned slide may be listed under several samples in the rich file; the first row's
    Comp.Bio.ID must win (the panel manifests use the same rule), and label-only backfilled slides
    without a co-author Comp.Bio.ID fall back to the library map's Sample.ID instead of the slide id."""
    from prosper.evaluation.st1_llm_summaries_cli import load_label_map
    from prosper.llm.search import analysis_lib

    rich = tmp_path / "rich.csv"
    pd.DataFrame({"slide_id": [1, 1, 2], "tumor_group": ["OS", "OS", "RMS"],
                  "Comp.Bio.ID": ["SJOS1_D1", "SJOS1_D5", "SJRHB2_D1"], "Site": ["a", "b", "c"]}).to_csv(rich, index=False)
    qrows = tmp_path / "q.csv"
    pd.DataFrame({"slide_id": ["Q"], "tumor_group": ["EWS"]}).to_csv(qrows, index=False)
    lab = tmp_path / "lab.csv"
    pd.DataFrame({"Slide.Scan.File": [1, 2, 3, 4], "Tumor.group": ["OS", "RMS", "NB", "RB"],
                  "Sample.ID": ["OS_1", "RMS_2", "NB_908", "RB_185"]}).to_csv(lab, index=False)
    ali = tmp_path / "ali.csv"
    pd.DataFrame({"Slide.Scan.File": ["3", "3"], "Comp.Bio.ID": ["SJNBL3_D1", "SJNBL3_D2"]}).to_csv(ali, index=False)

    meta, true_of = load_label_map(analysis_lib, rich, qrows, lab, ali)
    by = meta.drop_duplicates("slide_id").set_index(meta["slide_id"].astype(str))
    assert meta["slide_id"].astype(str).is_unique
    assert by.loc["1", "Comp.Bio.ID"] == "SJOS1_D1" and by.loc["1", "Site"] == "a"
    assert by.loc["3", "Comp.Bio.ID"] == "SJNBL3_D1"          # a co-author first row
    assert by.loc["4", "Comp.Bio.ID"] == "RB_185"             # Sample.ID fallback, not "4"
    assert "Sample.ID" not in meta.columns
    assert true_of == {"1": "OS", "2": "RMS", "Q": "EWS", "3": "NB", "4": "RB"}

    # without --ali-map the Sample.ID fallback must still apply (review finding 2026-10-09)
    meta2, _ = load_label_map(analysis_lib, rich, qrows, lab, None)
    by2 = meta2.set_index(meta2["slide_id"].astype(str))
    assert by2.loc["3", "Comp.Bio.ID"] == "NB_908" and by2.loc["4", "Comp.Bio.ID"] == "RB_185"
    assert "Sample.ID" not in meta2.columns
