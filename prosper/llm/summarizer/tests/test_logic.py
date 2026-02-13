import re
from patho_toy_summarizer.logic import summarize_neighbors


def build_payload(dxs, sims, use_probability=False):
    return {
        "current_case": {"site": None, "age": None, "sex": None},
        "neighbors": [
            {"case_id": f"C{i}", ("probability" if use_probability else "similarity"): sim, "tumor_type": None, "diagnosis": dx, "site": None, "age": None, "sex": None}
            for i, (dx, sim) in enumerate(zip(dxs, sims), start=1)
        ],
    }


def test_basic_mode_and_similarity():
    payload = build_payload([
        "A", "A", "A", "B", "A"
    ], [0.8, 0.82, 0.81, 0.79, 0.83])
    sentence = summarize_neighbors(payload)
    assert "supports A" in sentence
    assert re.search(r"similarity mean 0\.81", sentence)  # mean rounded to 0.81
    assert "1 outlier" in sentence or "1 outlier(s)" in sentence


def test_no_outliers_when_all_match_and_high_similarity():
    payload = build_payload([
        "A", "A", "A", "A", "A"
    ], [0.9, 0.91, 0.89, 0.92, 0.9])
    sentence = summarize_neighbors(payload)
    assert "supports A" in sentence
    assert "no outliers" in sentence
    assert len(sentence.split()) <= 25


def test_probability_mode_summary():
    payload = build_payload([
        "A", "A", "A", "B", "A"
    ], [0.8, 0.82, 0.81, 0.79, 0.83], use_probability=True)
    sentence = summarize_neighbors(payload)
    assert "probability mean" in sentence
    assert "higher is better" in sentence


def test_mode_none_and_truncation():
    payload = build_payload([
        None, None, None, None, None
    ], [0.5, 0.55, 0.52, 0.51, 0.53])
    sentence = summarize_neighbors(payload)
    assert "no dominant diagnosis" in sentence
    assert "similarity mean" in sentence
    assert len(sentence.split()) <= 25


def test_invalid_neighbor_count():
    bad_payload = {"current_case": {}, "neighbors": []}
    try:
        summarize_neighbors(bad_payload)
    except ValueError as e:
        assert "5 neighbors" in str(e)
    else:
        assert False, "Expected ValueError"
