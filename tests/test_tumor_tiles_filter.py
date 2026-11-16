"""The `filter` action of tumor_tiles_cli (ST2.0 phase 1): class-filtered copies
of (features, coords) pickles, arm dedup, and the coords-alignment refusal."""
import gzip
import pickle

import numpy as np
import pandas as pd
import pytest

from prosper.preprocessing.tumor_tiles_cli import build_parser, filter_features


def _write_slide(features_dir, scores_dir, sid, preds):
    n = len(preds)
    feats = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    coords = [{"wsi_loc": [512 * i, 1024], "loc": [i, 2]} for i in range(n)]
    with open(features_dir / f"{sid}.pkl", "wb") as fh:
        pickle.dump((feats, coords), fh)
    xy = np.array([c["wsi_loc"] for c in coords])
    pd.DataFrame({"wsi_loc_x": xy[:, 0], "wsi_loc_y": xy[:, 1],
                  "pred": preds, "p_tumor": [0.5] * n}
                 ).to_csv(scores_dir / f"{sid}.csv.gz", index=False, compression="gzip")
    return feats, coords


@pytest.fixture()
def dirs(tmp_path):
    features = tmp_path / "src"
    scores = tmp_path / "scores"
    out = tmp_path / "out"
    features.mkdir()
    scores.mkdir()
    return features, scores, out


def _run(features, scores, out, extra=()):
    a = build_parser().parse_args(
        ["filter", "--features-dir", str(features), "--scores-dir", str(scores),
         "--out-root", str(out), "--min-tiles", "2", *extra])
    return filter_features(a)


def test_arms_masks_and_dedup(dirs):
    features, scores, out = dirs
    fa, ca = _write_slide(features, scores, "AAA", [1, 1, 1, 1, 1, 1, 5, 5, 4, 4])
    _write_slide(features, scores, "BBB", [1, 1, 1, 5, 5])

    rec = _run(features, scores, out)
    assert rec["n_slides"] == 2
    assert rec["n_minus_artifacts_written"] == 1  # BBB deduped to tumor_only

    with open(out / "passthrough" / "AAA.pkl", "rb") as fh:
        f, c = pickle.load(fh)
    assert np.array_equal(f, fa) and c == ca

    with open(out / "tumor_only" / "AAA.pkl", "rb") as fh:
        f, c = pickle.load(fh)
    assert np.array_equal(f, fa[:6]) and [d["wsi_loc"] for d in c] == [d["wsi_loc"] for d in ca[:6]]

    with open(out / "minus_artifacts" / "AAA.pkl", "rb") as fh:
        f, _ = pickle.load(fh)
    assert np.array_equal(f, fa[[0, 1, 2, 3, 4, 5, 8, 9]])

    assert not (out / "minus_artifacts" / "BBB.pkl").exists()
    s = pd.read_csv(out / "_filter_summary.csv").set_index("slide")
    assert bool(s.loc["BBB", "minus_artifacts_equiv_tumor_only"]) is True
    assert int(s.loc["AAA", "kept_tumor_only"]) == 6
    assert int(s.loc["AAA", "n_artifacts"]) == 2


def test_misaligned_scores_refused(dirs):
    features, scores, out = dirs
    _write_slide(features, scores, "AAA", [1, 1, 1])
    # corrupt the order of the score rows
    df = pd.read_csv(scores / "AAA.csv.gz").iloc[::-1]
    df.to_csv(scores / "AAA.csv.gz", index=False, compression="gzip")
    with pytest.raises(SystemExit, match="not aligned"):
        _run(features, scores, out)


def test_zero_kept_refused(dirs):
    features, scores, out = dirs
    _write_slide(features, scores, "AAA", [5, 5, 5])
    with pytest.raises(SystemExit, match="0 tiles"):
        _run(features, scores, out, extra=["--arms", "tumor_only"])


def test_zero_kept_skipped_with_flag(dirs):
    features, scores, out = dirs
    _write_slide(features, scores, "AAA", [5, 5, 5])
    _write_slide(features, scores, "BBB", [1, 1, 5])
    _run(features, scores, out, extra=["--arms", "tumor_only", "--skip-empty"])
    assert not (out / "tumor_only" / "AAA.pkl").exists()
    assert (out / "tumor_only" / "BBB.pkl").exists()
    summary = pd.read_csv(out / "_filter_summary.csv").set_index("slide")
    assert bool(summary.loc["AAA", "skipped_tumor_only"]) is True
