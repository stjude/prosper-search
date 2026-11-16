"""
Regression tests for the Bag of Barcodes slide representation.

These pin two things that were wrong before:

1. **Barcode construction.** The reference implementation binarises patch
   features by the sign of adjacent-dimension differences,
   ``(np.diff(features, axis=1) < 0) * 1``, giving ``D - 1`` bits. An earlier
   version of this module used 64-bit random-hyperplane LSH instead -- a
   different algorithm, approximating cosine similarity rather than encoding the
   feature-gradient pattern, and not the one behind the published Yottixel
   results.

2. **Routing.** The generator declared ``PRODUCES_SIMILARITY = False`` while
   returning a variable-length set of per-patch barcodes, so the pipeline tried
   to ``np.stack`` ragged arrays into a search matrix and the Hamming code was
   unreachable.
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prosper.embeddings.slide_embedding import bob_embedding as be
from prosper.embeddings.slide_embedding.bob_embedding import (
    BoB,
    BoBSlideEmbeddingGenerator,
    bob_from_int_array,
    features_to_barcodes,
    hamming_min_median,
)

RNG = np.random.default_rng(0)


def _naive_min_median(a, b):
    """Explicit double loop, the definition we are vectorising."""
    minima = []
    for row in a:
        minima.append(min(int(np.sum(row != other)) for other in b))
    return float(np.median(minima))


def test_barcode_matches_reference_formula():
    """Must equal the reference expression exactly, bit for bit."""
    features = RNG.normal(size=(37, 128))
    reference = (np.diff(np.array(features), axis=1) < 0) * 1  # compute_bob_indices.py
    assert np.array_equal(features_to_barcodes(features), reference.astype(np.uint8))


def test_barcode_length_is_input_width_minus_one():
    gen = BoBSlideEmbeddingGenerator(feature_dim=1280)
    assert gen.barcode_length_for(1280) == 1279

    out = gen.process_single_slide(RNG.normal(size=(11, 1280)), "slide_x")
    assert out['barcodes'].shape == (11, 1279)
    assert out['barcode_length'] == 1279
    assert out['n_patches'] == 11
    assert out['barcodes'].dtype == np.uint8


def test_features_are_not_truncated_to_feature_dim():
    """The reference implementation binarises the full stored width.

    UNI2 features are stored 1536-d, so its barcodes are 1535 bits. Truncating to
    feature_dim=1280 would give 1279 and would not reproduce the published
    UNI2 + Yottixel numbers.
    """
    gen = BoBSlideEmbeddingGenerator(feature_dim=1280)
    out = gen.process_single_slide(RNG.normal(size=(8, 1536)), "uni2_slide")
    assert out['barcodes'].shape == (8, 1535), (
        "BoB must not truncate to feature_dim -- see module docstring"
    )


def test_barcode_invariant_to_positive_rescaling():
    """diff(c*x) = c*diff(x) preserves sign for c > 0, so L2 norm is a no-op here."""
    features = RNG.normal(size=(23, 64))
    scaled = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    assert np.array_equal(features_to_barcodes(features), features_to_barcodes(scaled))

    per_row = features * RNG.uniform(0.1, 10.0, size=(23, 1))
    assert np.array_equal(features_to_barcodes(features), features_to_barcodes(per_row))


def test_hamming_min_median_matches_naive_loop():
    a = (RNG.random((40, 129)) < 0.5).astype(np.uint8)
    b = (RNG.random((55, 129)) < 0.5).astype(np.uint8)
    assert np.isclose(hamming_min_median(a, b), _naive_min_median(a, b))


def test_hamming_is_exact_at_realistic_barcode_length():
    """The float32 matmul must be *exactly* right at 1279 bits, not merely close.

    Dot products are bounded by the barcode length, far below float32's 2**24
    exact-integer limit, so `sa + sb - 2 * a.b` is exact. Asserted with `==`
    against the naive definition rather than `isclose`.

    An odd row count is used so the median is itself one of the per-row minima
    and therefore integral -- with an even count the median is the mean of the
    two middle minima and can legitimately be a half-integer.
    """
    a = (RNG.random((17, 1279)) < 0.5).astype(np.uint8)
    b = (RNG.random((23, 1279)) < 0.5).astype(np.uint8)

    got = hamming_min_median(a, b)
    assert got == _naive_min_median(a, b), "vectorised Hamming disagrees with the definition"
    assert got == float(int(got)), f"median of an odd row count should be integral, got {got}"


def test_hamming_exact_for_even_row_count_too():
    """Even row counts give half-integer medians; still exact against the definition."""
    a = (RNG.random((16, 1279)) < 0.5).astype(np.uint8)
    b = (RNG.random((16, 1279)) < 0.5).astype(np.uint8)
    assert hamming_min_median(a, b) == _naive_min_median(a, b)


def test_chunking_matches_unchunked():
    """Blocking over query rows must not change the result."""
    a = (RNG.random((64, 97)) < 0.5).astype(np.uint8)
    b = (RNG.random((48, 97)) < 0.5).astype(np.uint8)
    unchunked = hamming_min_median(a, b)

    original = be._MAX_BLOCK
    try:
        be._MAX_BLOCK = 100  # forces several small blocks
        assert np.isclose(hamming_min_median(a, b), unchunked)
    finally:
        be._MAX_BLOCK = original


def test_self_distance_is_zero():
    barcodes = (RNG.random((30, 65)) < 0.5).astype(np.uint8)
    assert hamming_min_median(barcodes, barcodes) == 0.0


def test_asymmetric_by_construction():
    """Median-of-minimums is directional; the API must not pretend otherwise."""
    a = (RNG.random((5, 32)) < 0.5).astype(np.uint8)
    b = (RNG.random((60, 32)) < 0.5).astype(np.uint8)
    # not asserting inequality (it can coincide), only that both directions run
    # and are finite -- the point is that callers must fix an orientation.
    assert np.isfinite(hamming_min_median(a, b))
    assert np.isfinite(hamming_min_median(b, a))


def test_produces_similarity_is_declared():
    """Guards the routing regression: BoB has no fixed-length slide vector."""
    assert BoBSlideEmbeddingGenerator.PRODUCES_SIMILARITY is True
    assert hasattr(BoBSlideEmbeddingGenerator, 'compute_pairwise_similarities')


def test_compute_distance_with_indices_locates_the_median():
    gen = BoBSlideEmbeddingGenerator(feature_dim=64)
    a = gen.process_single_slide(RNG.normal(size=(9, 64)), "a")
    b = gen.process_single_slide(RNG.normal(size=(12, 64)), "b")

    dist, ref_idx, query_idx = gen.compute_distance_with_indices(a, b)
    assert np.isclose(dist, gen.compute_distance(a, b))
    # odd row count -> single representative pair
    assert isinstance(ref_idx, int) and isinstance(query_idx, int)
    assert 0 <= ref_idx < 9 and 0 <= query_idx < 12


def test_int_array_roundtrip():
    barcodes = (RNG.random((7, 130)) < 0.5).astype(np.uint8)
    bob = BoB(barcodes)
    restored = bob_from_int_array(bob.to_int_array(), barcode_dim=130)
    assert np.array_equal(restored.barcodes, barcodes)


def test_pairwise_similarities_shape_and_self_placeholder(tmp_path=None):
    import tempfile

    gen = BoBSlideEmbeddingGenerator(feature_dim=32, verbose=False)
    data = {
        sid: gen.process_single_slide(RNG.normal(size=(6, 32)), sid)
        for sid in ("s1", "s2", "s3")
    }
    with tempfile.TemporaryDirectory() as d:
        sims = gen.compute_pairwise_similarities(slide_data=data, output_dir=d)

    assert set(sims) == {"s1", "s2", "s3"}
    for qid, row in sims.items():
        assert len(row) == 3
        self_dist = dict(row)[qid]
        assert self_dist == be.SELF_MATCH_DISTANCE, "self-match placeholder missing"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall BoB regression tests passed")
