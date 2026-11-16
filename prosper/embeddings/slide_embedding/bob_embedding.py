"""
Bag of Barcodes (BoB) slide embedding generation.

A slide is represented as a *set* of per-patch binary barcodes, and the distance
between two slides is the median, over the query's barcodes, of the minimum
Hamming distance to any barcode of the reference. This is the Yottixel
formulation used for the published results.

The distance is **asymmetric** -- the median is taken over the first argument's
barcodes. The published distance matrices follow row = query, column = reference,
i.e. ``M[q, r] == hamming_min_median(barcodes[q], barcodes[r])``. Verified
against ``combined_similarity_matrix_virchow2_yottixel.pkl``: 15 of 15 probed
slide pairs match exactly in both orientations, half-integer medians included.

Barcode construction
--------------------
Each patch feature vector is binarised by the sign of its adjacent-dimension
differences::

    barcode = (np.diff(features, axis=1) < 0).astype(np.uint8)

so a ``D``-dimensional feature vector yields a ``D - 1`` bit barcode. The barcode
length is *derived from the input width*, not configured.

``feature_dim`` is deliberately **not** applied here. The reference
implementation binarises the full stored feature width, so barcodes are 1279 bits
for Virchow2 (stored 1280-d, already truncated at storage time) and **1535 bits
for UNI2** (stored 1536-d). Truncating to ``feature_dim`` here would silently
produce 1279-bit UNI2 barcodes and would not reproduce the published UNI2 +
Yottixel numbers. Other aggregations (e.g. percentile) *do* truncate, which is
why the parameter exists on the base class.

Note this construction is invariant to any positive per-patch rescaling
(including L2 normalisation), since ``diff(c * x) = c * diff(x)`` preserves sign
for ``c > 0``. No normalisation is applied.

History
-------
An earlier version of this module binarised features with 64-bit
random-hyperplane LSH instead. That is a different algorithm -- it approximates
cosine similarity rather than encoding the feature-gradient pattern -- and it did
not match the reference implementation behind the published Yottixel results. It
was also unreachable: the generator declared ``PRODUCES_SIMILARITY = False``, so
the pipeline tried to stack per-patch integer arrays of differing lengths into a
fixed-size matrix. Both are corrected here.
"""

import os
import pickle
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

try:
    import bitarray
    import bitarray.util as butil
    HAS_BITARRAY = True
except ImportError:
    HAS_BITARRAY = False

from .base import BaseSlideEmbeddingGenerator, register_slide_embedding

# Cap on the (n_a x n_b) float32 distance block held at once, in elements.
# 16M elements ~= 64 MB, which bounds memory for the largest slides
# (~5,000 patches) while keeping the matmul in one BLAS call for typical ones.
_MAX_BLOCK = 16_000_000

SELF_MATCH_DISTANCE = 1e6


def features_to_barcodes(features: np.ndarray) -> np.ndarray:
    """
    Binarise patch features by the sign of adjacent-dimension differences.

    Args:
        features: Array of shape (N, D).

    Returns:
        Binary array of shape (N, D - 1), dtype uint8.
    """
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"expected a 2-D (N, D) array, got shape {features.shape}")
    if features.shape[1] < 2:
        raise ValueError("need at least 2 feature dimensions to take differences")
    return (np.diff(features, axis=1) < 0).astype(np.uint8)


def hamming_min_median(barcodes_a: np.ndarray, barcodes_b: np.ndarray) -> float:
    """
    Median over rows of `barcodes_a` of the minimum Hamming distance to `barcodes_b`.

    Uses the identity ``hamming(a, b) = |a| + |b| - 2 * a . b`` for binary
    vectors, which turns the whole pairwise computation into one matmul. Exact in
    float32: dot products are bounded by the barcode length (~1279), far below
    the 2**24 limit of the float32 mantissa.
    """
    A = np.ascontiguousarray(barcodes_a, dtype=np.float32)
    B = np.ascontiguousarray(barcodes_b, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(
            f"barcode shape mismatch: {barcodes_a.shape} vs {barcodes_b.shape}"
        )

    sb = B.sum(axis=1)
    n_b = B.shape[0]
    rows_per_block = max(1, _MAX_BLOCK // max(n_b, 1))

    minima = np.empty(A.shape[0], dtype=np.float32)
    for start in range(0, A.shape[0], rows_per_block):
        block = A[start:start + rows_per_block]
        d = block.sum(axis=1)[:, None] + sb[None, :] - 2.0 * (block @ B.T)
        minima[start:start + block.shape[0]] = d.min(axis=1)

    return float(np.median(minima))


class BoB:
    """Bag of Barcodes representation for a single slide."""

    def __init__(self, barcodes):
        """
        Args:
            barcodes: (N, L) binary array, or a sequence of binary arrays /
                bitarrays.
        """
        if isinstance(barcodes, np.ndarray):
            self.barcodes = barcodes.astype(np.uint8, copy=False)
        elif len(barcodes) == 0:
            self.barcodes = np.zeros((0, 0), dtype=np.uint8)
        elif HAS_BITARRAY and isinstance(barcodes[0], bitarray.bitarray):
            self.barcodes = np.array([list(b) for b in barcodes], dtype=np.uint8)
        else:
            self.barcodes = np.asarray(barcodes, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.barcodes)

    @property
    def barcode_length(self) -> int:
        return self.barcodes.shape[1] if self.barcodes.ndim == 2 else 0

    def select_subset(self, n: int = 3, rng: Optional[np.random.Generator] = None) -> "BoB":
        """Select a random subset of n barcodes. Pass `rng` for determinism."""
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.permutation(len(self.barcodes))[:n]
        return BoB(self.barcodes[idx])

    def distance(self, other: "BoB") -> float:
        """Median of minimum Hamming distances from each barcode in self to other."""
        return hamming_min_median(self.barcodes, other.barcodes)

    def to_int_array(self) -> np.ndarray:
        """
        Pack each barcode into one Python integer, for compact storage.

        Barcodes are longer than 64 bits (1279 for 1280-d features), so this
        returns an object-dtype array of Python ints. It is lossy about length --
        leading zeros are not recoverable -- so pass the length to
        `bob_from_int_array` when round-tripping.
        """
        if HAS_BITARRAY:
            return np.array(
                [butil.ba2int(bitarray.bitarray(b.tolist())) for b in self.barcodes],
                dtype=object,
            )
        return np.array(
            [int("".join(map(str, b.tolist())) or "0", 2) for b in self.barcodes],
            dtype=object,
        )


def bob_from_int_array(int_array: np.ndarray, barcode_dim: int) -> BoB:
    """
    Reconstruct a BoB from packed integers.

    Args:
        int_array: Packed barcodes, as produced by `BoB.to_int_array`.
        barcode_dim: Barcode length in bits. Required -- packing loses leading
            zeros, so the length cannot be inferred.
    """
    rows = [
        np.array(
            [int(c) for c in format(int(x), "b").zfill(barcode_dim)[-barcode_dim:]],
            dtype=np.uint8,
        )
        for x in int_array
    ]
    return BoB(np.array(rows, dtype=np.uint8) if rows else np.zeros((0, barcode_dim), np.uint8))


@register_slide_embedding("bob")
class BoBSlideEmbeddingGenerator(BaseSlideEmbeddingGenerator):
    """
    Slide representation as a bag of per-patch binary barcodes.

    Produces pairwise distances rather than a fixed-length slide vector, so
    `PRODUCES_SIMILARITY` is True and the pipeline routes this through its
    pairwise-similarity path.
    """

    METHOD_NAME = "bob"
    PRODUCES_SIMILARITY = True

    def __init__(
        self,
        feature_dim: int = 1280,
        n_workers: int = 32,
        verbose: bool = True,
    ):
        super().__init__(feature_dim, n_workers, verbose)

    @staticmethod
    def barcode_length_for(feature_width: int) -> int:
        """Barcode length in bits for a given stored feature width."""
        return feature_width - 1

    def features_to_barcodes(self, features: np.ndarray) -> np.ndarray:
        """Binarise features by adjacent-dimension difference sign."""
        return features_to_barcodes(features)

    def process_single_slide(
        self,
        patch_embeddings: np.ndarray,
        slide_id: str,
        mosaic: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate the BoB representation for one slide.

        Args:
            patch_embeddings: Array of patch embeddings (N, D).
            slide_id: Identifier for the slide.
            mosaic: Unused for BoB.

        Returns:
            Dict with the barcode array and metadata.
        """
        # No truncation: see the module docstring. The reference implementation
        # binarises the full stored width, which differs per encoder.
        barcodes = self.features_to_barcodes(patch_embeddings)

        return {
            'slide_id': slide_id,
            'barcodes': barcodes,
            'n_patches': len(barcodes),
            'barcode_length': barcodes.shape[1] if barcodes.ndim == 2 else 0,
        }

    def compute_distance(
        self,
        bob_a: Dict[str, Any],
        bob_b: Dict[str, Any]
    ) -> float:
        """Median of minimum Hamming distances, from bob_a's barcodes to bob_b's."""
        return hamming_min_median(bob_a['barcodes'], bob_b['barcodes'])

    def compute_distance_with_indices(
        self,
        bob_a: Dict[str, Any],
        bob_b: Dict[str, Any]
    ) -> Tuple[float, Any, Any]:
        """
        As `compute_distance`, plus the patch indices realising the median.

        Returns:
            (distance, ref_patch_index(es), query_patch_index(es)). For an even
            number of barcodes the median falls between two rows, so both are
            returned.
        """
        A = np.ascontiguousarray(bob_a['barcodes'], dtype=np.float32)
        B = np.ascontiguousarray(bob_b['barcodes'], dtype=np.float32)
        d = A.sum(1)[:, None] + B.sum(1)[None, :] - 2.0 * (A @ B.T)

        min_dists = d.min(axis=1)
        argmins = d.argmin(axis=1)
        median_value = float(np.median(min_dists))

        n = len(min_dists)
        order = np.argsort(min_dists)
        if n % 2 == 1:
            ref_idx = int(order[n // 2])
            return median_value, ref_idx, int(argmins[ref_idx])
        r1, r2 = int(order[n // 2 - 1]), int(order[n // 2])
        return median_value, [r1, r2], [int(argmins[r1]), int(argmins[r2])]

    def compute_pairwise_similarities(
        self,
        slide_data: Dict[str, Dict],
        output_dir: str,
        slide_id: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute pairwise BoB distances between slides.

        Mirrors the OT generator's interface, including the per-query pickle
        cache and the self-match placeholder, so the pipeline can dispatch both
        through one path.

        Args:
            slide_data: Slide ID -> dict as returned by `process_single_slide`.
            output_dir: Directory for per-query caches.
            slide_id: If given, compute only this query's row.

        Returns:
            Slide ID -> list of (other_id, distance) tuples.
        """
        from tqdm import tqdm

        os.makedirs(output_dir, exist_ok=True)
        slide_ids = sorted(slide_data.keys())
        query_ids = [slide_id] if slide_id is not None else slide_ids

        results = {}
        for qid in tqdm(
            query_ids, desc="Computing BoB distances", disable=not self.verbose
        ):
            output_path = os.path.join(output_dir, f'similarity_{qid}.pkl')
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    results[qid] = pickle.load(f)
                continue

            query_barcodes = slide_data[qid]['barcodes']
            similarities = []
            for oid in slide_ids:
                if oid == qid:
                    similarities.append((oid, SELF_MATCH_DISTANCE))
                    continue
                try:
                    dist = hamming_min_median(
                        query_barcodes, slide_data[oid]['barcodes']
                    )
                except Exception as exc:  # noqa: BLE001 - mirror OT's tolerance
                    if self.verbose:
                        print(f"BoB distance failed for {qid} vs {oid}: {exc}")
                    dist = SELF_MATCH_DISTANCE
                similarities.append((oid, dist))

            with open(output_path, 'wb') as f:
                pickle.dump(similarities, f)
            results[qid] = similarities

        return results
