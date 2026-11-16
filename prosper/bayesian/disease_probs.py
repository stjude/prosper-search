"""
Compute disease probabilities from a distance matrix and reference labels.
Rows of the distance matrix are test slides; columns correspond to reference
samples whose labels are given by class_labels (list of strings).
"""

import numpy as np
import pandas as pd


def _disease_probs_from_neighbors(
    distances, class_labels_int, n_neighbors, T, n_classes
):
    """
    Core logic: compute class probabilities from a single distance vector
    and integer class labels. Used internally by compute_disease_probs.
    """
    distances = np.asarray(distances, dtype=np.float64)
    class_labels_int = np.asarray(class_labels_int, dtype=np.int64)

    if distances.size != class_labels_int.size:
        raise ValueError("distances and class_labels must have the same length")

    n = min(n_neighbors, len(distances))
    if n == 0:
        return np.zeros(n_classes)

    nn_inds = np.argsort(distances)
    nn_dists_sorted = distances[nn_inds]
    nn_labs = class_labels_int[nn_inds[:n]]

    d_min = nn_dists_sorted[0]
    if len(distances) > n_neighbors:
        d_scale = nn_dists_sorted[n_neighbors]
    else:
        d_scale = nn_dists_sorted[n - 1]
    gap = d_scale - d_min
    if gap < 1e-10:
        tau = 0.0
    else:
        tau = np.log(T) / gap

    nn_dists = nn_dists_sorted[:n]
    probs = np.zeros((n_classes,), dtype=np.float64)
    for j in range(n):
        clab = nn_labs[j]
        # Weight relative to the nearest neighbour, so the weight scale runs from
        # 1 at d_min to 1/T at d_scale. The offset cancels under the
        # normalisation below, but computing exp(-tau * d) on absolute distances
        # underflows at realistic scales: OT-cosine distances are ~0.5 with
        # tau ~ 1e2, giving exp(-50) ~ 1e-21 for every neighbour. The sum then
        # trips the `total < 1e-15` guard and the posterior is silently replaced
        # by a uniform one.
        cdist = nn_dists[j] - d_min
        probs[clab] += np.exp(-tau * cdist)

    total = np.sum(probs)
    if total < 1e-15:
        # Defensive only, and unreachable: the nearest neighbour contributes
        # exp(0) == 1, so total >= 1 whenever n >= 1.
        probs = np.ones(n_classes) / n_classes
    else:
        probs = probs / total

    return probs


def compute_disease_probs(
    distance_matrix,
    class_labels,
    n_neighbors,
    T,
    test_slide_ids=None,
):
    """
    Compute disease probabilities for each test slide from a distance matrix
    and reference labels.

    Rows of the distance matrix are test cases (test slides); columns are
    reference samples. class_labels[j] is the disease label (string) of the
    j-th column (j-th reference sample).

    Parameters
    ----------
    distance_matrix : array-like, shape (n_test_slides, n_reference_samples)
        Distance from each test slide to each reference sample.
    class_labels : list of str, length n_reference_samples
        class_labels[j] is the disease label of the j-th column (reference sample).
    n_neighbors : int
        Number of nearest neighbors to use.
    T : float
        Temperature parameter for exponential weighting.
    test_slide_ids : list, optional, length n_test_slides
        Identifiers for each test slide (for DataFrame index). If None, uses
        "slide_0", "slide_1", ...

    Returns
    -------
    list_of_dicts : list of dict
        list_of_dicts[i] is a dictionary {disease_name: probability} for the
        i-th test slide (i-th row of the distance matrix).
    df : pandas.DataFrame
        Rows = test slides (index = test_slide_ids if provided).
        Columns = disease names sorted alphabetically. Entry (i, d) is the
        probability of disease d for the i-th test slide.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n_test, n_ref = distance_matrix.shape

    if len(class_labels) != n_ref:
        raise ValueError(
            "class_labels length must equal number of columns of distance_matrix "
            f"(got {len(class_labels)} vs {n_ref})"
        )

    # Diseases in alphabetical order (column order for DataFrame)
    diseases = sorted(set(class_labels))
    n_classes = len(diseases)
    label_to_idx = {d: i for i, d in enumerate(diseases)}

    # Integer labels for each reference column (for internal computation)
    class_labels_int = np.array([label_to_idx[lbl] for lbl in class_labels], dtype=np.int64)

    list_of_dicts = []
    rows_for_df = []

    for i in range(n_test):
        probs = _disease_probs_from_neighbors(
            distance_matrix[i, :],
            class_labels_int,
            n_neighbors,
            T,
            n_classes,
        )
        d = {diseases[j]: float(probs[j]) for j in range(n_classes)}
        list_of_dicts.append(d)
        rows_for_df.append(probs)

    if test_slide_ids is not None:
        if len(test_slide_ids) != n_test:
            raise ValueError(
                "test_slide_ids length must equal number of rows of distance_matrix"
            )
        index = test_slide_ids
    else:
        index = [f"slide_{i}" for i in range(n_test)]

    df = pd.DataFrame(
        np.array(rows_for_df),
        index=index,
        columns=diseases,
    )

    return list_of_dicts, df
