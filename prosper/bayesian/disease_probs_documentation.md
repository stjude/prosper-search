# disease_probs — Documentation

This module computes **disease (class) probabilities** from a distance matrix and reference labels. Each test slide (row) gets a probability distribution over diseases by weighting its nearest neighbors with an exponential kernel controlled by a temperature parameter `T`.

---

## Overview

- **Rows** of the distance matrix = test slides (one distance vector per test case).
- **Columns** = reference samples; each column has a disease label (string).
- For each test slide, the code finds the `n_neighbors` nearest reference samples (by distance), assigns weights `exp(-τ × distance)` with τ derived from `T`, and normalizes the weighted counts per disease to get probabilities.

---

## Functions

### 1. `compute_disease_probs` (main API)

Compute disease probabilities for **every** test slide from a full distance matrix and string labels. Use this for batch evaluation.

#### Signature

```python
def compute_disease_probs(
    distance_matrix,
    class_labels,
    n_neighbors,
    T,
    test_slide_ids=None,
):
```

#### Parameters

| Parameter         | Type   | Description |
|------------------|--------|-------------|
| `distance_matrix`| array-like | Shape `(n_test_slides, n_reference_samples)`. Row `i` = distances from test slide `i` to each reference sample. |
| `class_labels`   | list of str | Length = `n_reference_samples`. `class_labels[j]` = disease name for the j-th column. |
| `n_neighbors`    | int    | Number of nearest neighbors (N) to use. |
| `T`              | float  | Temperature for exponential weighting; larger T gives softer decay. |
| `test_slide_ids` | list, optional | Length = `n_test_slides`. IDs for the DataFrame index. If `None`, uses `"slide_0"`, `"slide_1"`, … |

#### Returns

- **`list_of_dicts`**: List of length `n_test_slides`. `list_of_dicts[i]` is `{disease_name: probability}` for the i-th test slide.
- **`df`**: `pandas.DataFrame`. Rows = test slides (index = `test_slide_ids` or default); columns = disease names **sorted alphabetically**; values = probabilities.

#### Example 1: Basic usage

```python
import numpy as np
from disease_probs import compute_disease_probs

# 3 test slides, 6 reference samples
distance_matrix = np.array([
    [0.1, 0.5, 0.3, 0.8, 0.2, 0.4],   # test slide 0
    [0.2, 0.4, 0.6, 0.1, 0.9, 0.3],   # test slide 1
    [0.7, 0.2, 0.5, 0.3, 0.1, 0.6],   # test slide 2
])
# One label per column (reference sample)
class_labels = ["AML", "ALL", "AML", "OTHER", "ALL", "AML"]

list_of_dicts, df = compute_disease_probs(
    distance_matrix,
    class_labels,
    n_neighbors=3,
    T=20,
)

# Per-slide dicts: disease name -> probability
print(list_of_dicts[0])
# e.g. {'ALL': 0.12, 'AML': 0.78, 'OTHER': 0.10}

# DataFrame: rows = slides, columns = diseases (alphabetical)
print(df)
#        ALL   AML  OTHER
# slide_0  ...   ...   ...
# slide_1  ...   ...   ...
# slide_2  ...   ...   ...
```

#### Example 2: With custom slide IDs

```python
list_of_dicts, df = compute_disease_probs(
    distance_matrix,
    class_labels,
    n_neighbors=3,
    T=20,
    test_slide_ids=["case_001", "case_002", "case_003"],
)
print(df.index.tolist())  # ['case_001', 'case_002', 'case_003']
print(df.columns.tolist())  # ['ALL', 'AML', 'OTHER'] — alphabetical
```

#### Example 3: Get predicted disease per slide

```python
list_of_dicts, df = compute_disease_probs(
    distance_matrix, class_labels, n_neighbors=3, T=20
)
for i, d in enumerate(list_of_dicts):
    pred = max(d, key=d.get)
    print(f"Slide {i}: predicted disease = {pred} (prob = {d[pred]:.3f})")
```

---

### 2. `_disease_probs_from_neighbors` (internal / low-level)

Compute class probabilities for a **single** distance vector using **integer** class labels. Used inside `compute_disease_probs`; exposed for callers who already have integer labels and want one query at a time.

#### Signature

```python
def _disease_probs_from_neighbors(
    distances,
    class_labels_int,
    n_neighbors,
    T,
    n_classes,
):
```

#### Parameters

| Parameter          | Type   | Description |
|--------------------|--------|-------------|
| `distances`        | array-like | Length = number of reference samples. One distance per reference. |
| `class_labels_int` | array-like of int | Same length as `distances`. `class_labels_int[j]` = class index (0 to `n_classes - 1`) for the j-th reference. |
| `n_neighbors`      | int    | Number of nearest neighbors (N). |
| `T`                | float  | Temperature for exponential weighting. |
| `n_classes`        | int    | Total number of classes. |

#### Returns

- **`probs`**: 1D `np.ndarray` of length `n_classes`. Probability per class (indices 0 … `n_classes - 1`), summing to 1.

#### Example: Single query with integer labels

```python
import numpy as np
from disease_probs import _disease_probs_from_neighbors

# One test slide: distances to 5 reference samples
distances = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
# Integer labels: 0=ALL, 1=AML, 2=OTHER (example mapping)
class_labels_int = np.array([0, 1, 0, 2, 1])
n_classes = 3

probs = _disease_probs_from_neighbors(
    distances,
    class_labels_int,
    n_neighbors=3,
    T=20,
    n_classes=n_classes,
)
print(probs)  # e.g. [0.15 0.78 0.07]
print(probs.sum())  # 1.0
```

#### Example: Replicate one row of `compute_disease_probs`

```python
# Same as first row of distance_matrix and class_labels in Example 1
distances = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.4])
diseases = ["ALL", "AML", "OTHER"]  # alphabetical
label_to_idx = {d: i for i, d in enumerate(diseases)}
class_labels = ["AML", "ALL", "AML", "OTHER", "ALL", "AML"]
class_labels_int = np.array([label_to_idx[l] for l in class_labels])

probs = _disease_probs_from_neighbors(
    distances, class_labels_int, n_neighbors=3, T=20, n_classes=3
)
d = {diseases[j]: probs[j] for j in range(3)}
print(d)  # same as list_of_dicts[0] from compute_disease_probs
```

---

## Notes

- **Distance convention**: Smaller distance = more similar. The N **nearest** (smallest) distances are used.
- **T**: τ is set so that the weight scale goes from 1 at the nearest neighbor to 1/T at the (N+1)-th distance (when available). Larger T → slower decay → farther neighbors matter more.
- **Edge cases**: If the distance range is negligible, weights are effectively uniform. If total weight is zero, probabilities are set to uniform over classes.
- **DataFrame columns**: Always sorted alphabetically by disease name.
