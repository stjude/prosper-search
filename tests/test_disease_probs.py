"""
Regression tests for the Bayesian disease-probability KDE weighting.

These guard a numerical-stability property of `_disease_probs_from_neighbors`:
kernel weights must be computed on distances measured *relative to the nearest
neighbor*. The module documentation specifies the intended scale -- "tau is set
so that the weight scale goes from 1 at the nearest neighbor to 1/T at the
(N+1)-th distance" -- which only holds for exp(-tau * (d - d_min)).

Computing exp(-tau * d) on absolute distances is mathematically equivalent under
the normalization, but underflows at the scales this pipeline actually operates
at: optimal-transport cosine distances are ~0.5 while tau ~ 1e2, so every weight
becomes ~1e-21, the sum trips the `total < 1e-15` guard, and the posterior is
silently replaced by a uniform one. Measured on the UNI2 / OT-cosine matrix that
backs Supplementary Tables S11 and S13, that collapsed ~33% of queries to uniform
and dropped argmax accuracy from 0.86 to 0.62 at k=5.
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prosper.bayesian.disease_probs import _disease_probs_from_neighbors

T = 20.0
N_CLASSES = 3


def _probs(distances, labels, n_neighbors=5, t=T, n_classes=N_CLASSES):
    return _disease_probs_from_neighbors(
        np.asarray(distances, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        n_neighbors,
        t,
        n_classes,
    )


def test_no_underflow_at_realistic_distance_scale():
    """The regression. Absolute distances ~0.5 with a tight gap must not collapse.

    gap = 0.01 gives tau = log(20)/0.01 ~= 300, so exp(-tau * 0.5) ~= 1e-65 on
    absolute distances -- every weight underflows and the posterior goes uniform.
    """
    distances = [0.500, 0.502, 0.504, 0.506, 0.508, 0.510]
    labels = [0, 1, 1, 2, 2, 0]

    probs = _probs(distances, labels)

    uniform = np.full(N_CLASSES, 1.0 / N_CLASSES)
    assert not np.allclose(probs, uniform), (
        "posterior collapsed to uniform at a realistic distance scale -- the "
        "d_min shift is missing from the kernel exponent"
    )
    assert probs.argmax() == 0, "the nearest neighbor's class should dominate"
    assert np.isclose(probs.sum(), 1.0)


def test_invariant_to_constant_offset():
    """Shifting every distance by a constant must not change the posterior.

    gap, tau and (d - d_min) are all offset-invariant, so this is exact. It fails
    only once the un-shifted exponent underflows.
    """
    base = np.array([0.001, 0.003, 0.005, 0.007, 0.009, 0.011])
    labels = [0, 1, 1, 2, 2, 0]

    reference = _probs(base, labels)
    for offset in (0.5, 5.0, 500.0):
        shifted = _probs(base + offset, labels)
        assert np.allclose(reference, shifted, atol=1e-12), (
            f"posterior changed under a constant offset of {offset}"
        )


def test_nearest_neighbor_anchors_the_weight_scale():
    """Documented invariant: the nearest neighbor's weight is 1, so total >= 1.

    Probed through the public return value: with every neighbor in one class the
    posterior is degenerate, and it must stay so at any distance scale rather
    than falling through to the uniform guard.
    """
    for scale in (0.0, 0.5, 50.0):
        distances = np.array([0.0, 0.002, 0.004, 0.006, 0.008, 0.010]) + scale
        probs = _probs(distances, [1, 1, 1, 1, 1, 1])
        assert np.isclose(probs[1], 1.0), f"degenerate case broke at scale {scale}"
        assert np.isclose(probs.sum(), 1.0)


def test_equidistant_neighbors_reduce_to_vote_counting():
    """gap < 1e-10 gives tau = 0, so every weight is 1.

    That makes the posterior the class *frequencies* among the k nearest -- not
    uniform over classes. With n_neighbors=5 the first five labels are
    [0, 1, 2, 0, 1], i.e. counts 2/2/1, so the posterior is 0.4/0.4/0.2. This is
    the correct degenerate behaviour and the fix must not disturb it.
    """
    probs = _probs([0.4] * 6, [0, 1, 2, 0, 1, 2])
    assert np.allclose(probs, [0.4, 0.4, 0.2]), probs
    assert np.isclose(probs.sum(), 1.0)


def test_single_near_neighbor_outweighs_two_distant_ones():
    """The paper's DSRCT case: 1 nearest neighbor dominates a 2-vote majority."""
    # class 0 nearest; class 1 holds the plurality but sits further out
    distances = [0.10, 0.30, 0.32, 0.50, 0.52, 0.34]
    labels = [0, 1, 1, 2, 2, 0]

    probs = _probs(distances, labels)
    assert probs[0] > probs[1], (
        "the nearest neighbor should dominate the posterior even against a "
        "larger raw vote count"
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall disease_probs regression tests passed")
