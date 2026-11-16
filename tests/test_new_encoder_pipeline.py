"""Regression tests for the H-Optimus-1 / RetCCL rung: the registry additions,
SAMPLER pooling, and the OT/BoB combine path -- everything that does not
require a GPU or a real LSF submission.

Model loading itself (``_build_h_optimus_1``, ``_build_retccl``) is verified
separately by a one-off CPU smoke run recorded in [internal record] (both load
and produce the declared output dim on a synthetic patch); it is not repeated
here since it downloads/loads real weights and is not something a normal test
run should do on every invocation.
"""
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prosper.embeddings import tile_features as T
from prosper.embeddings.aggregate_cli import _pool, SENTINEL
from prosper.embeddings.combine_similarity_cli import (
    OUTPUT_METHOD_TAG, _load_rows, build,
)


def test_new_encoders_registered_with_correct_dims():
    assert T.REGISTRY["h_optimus_1"].dim == 1536
    assert T.REGISTRY["retccl"].dim == 2048
    assert T.REGISTRY["h_optimus_1"].builder in T._BUILDERS
    assert T.REGISTRY["retccl"].builder in T._BUILDERS


def test_sampler_pooling_shape_and_values():
    """SAMPLER = np.percentile(feats, [5,15,...,95], axis=0), flattened --
    per comet-uni2/src/sampler_gen.py. A (N, D) input yields a (10*D,) vector,
    and each D-sized block is one percentile level. The float16-before-
    percentile cast matches
    prosper.embeddings.slide_embedding.percentile_embedding's existing
    generator and was confirmed load-bearing by a fidelity check against the
    published uni2/sampler matrix (FAILED without it: max rel diff 3.4e-5)."""
    rng = np.random.default_rng(0)
    feats = rng.random((37, 6)).astype(np.float32)
    out = _pool(feats, "sampler")
    assert out.shape == (60,)
    expected = np.percentile(feats.astype(np.float16),
                             [5, 15, 25, 35, 45, 55, 65, 75, 85, 95], axis=0)
    np.testing.assert_allclose(out, expected.ravel())


def test_median_pooling_still_float16_cast():
    """Unaffected by the SAMPLER addition: the float16-before-median cast is
    part of the published definition, not a storage detail (see
    aggregate_cli.py's module docstring)."""
    feats = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    out = _pool(feats, "median_pooling")
    expected = np.median(feats.astype(np.float16), axis=0)
    np.testing.assert_array_equal(out, expected)


def test_output_method_tag_matches_published_convention():
    """Published filenames use ot_cosine / yottixel, not the generator
    registry keys ot / bob."""
    assert OUTPUT_METHOD_TAG == {"ot": "ot_cosine", "bob": "yottixel"}


def _write_row(dir_: Path, qid: str, row: list) -> None:
    with open(dir_ / f"similarity_{qid}.pkl", "wb") as fh:
        pickle.dump(row, fh)


def test_combine_builds_square_matrix_with_self_sentinel():
    ids = ["s1", "s2", "s3"]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # A tiny, hand-built, symmetric distance table.
        dist = {("s1", "s2"): 0.2, ("s1", "s3"): 0.5, ("s2", "s3"): 0.7}

        def d_of(a, b):
            if a == b:
                return SENTINEL
            return dist.get((a, b), dist.get((b, a)))

        for qid in ids:
            row = [(oid, d_of(qid, oid)) for oid in ids]
            _write_row(d, qid, row)

        prov = build(d, "testenc", "ot_cosine", d / "out")
        assert prov["n_query_slides"] == 3
        assert prov["n_reference_slides"] == 3
        assert prov["square"] is True

        with open(d / "out" / "combined_similarity_matrix_testenc_ot_cosine.pkl", "rb") as fh:
            df = pickle.load(fh)
        assert df.loc["s1", "s2"] == 0.2
        assert df.loc["s2", "s3"] == 0.7
        assert df.loc["s1", "s1"] == SENTINEL


def test_combine_refuses_mismatched_reference_library():
    """Every row must be computed against the same reference library, or the
    matrix silently is not square/comparable -- fail loudly instead."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_row(d, "s1", [("s2", 0.1), ("s3", 0.2)])
        _write_row(d, "s2", [("s1", 0.1)])  # missing s3 -> different library
        try:
            _load_rows(d)
            assert False, "expected a mismatch to raise"
        except SystemExit as exc:
            assert "mismatch" in str(exc)
