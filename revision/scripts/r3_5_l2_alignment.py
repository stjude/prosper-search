#!/usr/bin/env python3
"""R3.5-L2 -- do the OT slide-to-slide maps overlap pathologist-annotated diagnostic regions? (specification 004)

Binding specification: $PROSPER_OUT/r3_5_localisation_maps_20261017/contract_004/SPECIFICATION.md (sha256 679ee37f...).
Post-processing of the saved R3.5-L1 exact-OT plans (no new OT solve) plus one cross-patient k-NN probe:

  design 1  class-conditional transport (secondary): S, S0, S*, normalised alignment NA, kappa, retention
  design 2  class-wise decomposition of the OT distance and the discriminative contribution map (primary)
  design 3  ceiling diagnostics: 3a embedding probe, 3b unconstrained NN matching, 3c forced mismatch

One stage, `run`, writes $PROSPER_OUT/r3_5_localisation_maps_20261017/l2/run_NNN/ (never overwritten; a failed
run keeps FAILED.json + checks.json).  Fidelity checks F1-F6 gate the result files: on any failure no result
file is written.  L1 code is imported read-only from scripts/r3_5_l1_maps.py (MapSlide, class_of, project,
score_continuous, b_tile_values, sparsify, cross_slide_agreement, bootstrap helpers); prosper-search is
imported read-only through it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
ROOT = OUT / "r3_5_localisation_maps_20261017"
STAGE = ROOT / "l2"
CONTRACT = ROOT / "contract_004" / "CONTRACT.md"
CONTRACT_SHA = "679ee37f9c3f2fb8123efdbe14e7b88db6fd58e5ac2596a9d3a7e77de3297698"
AMENDMENT = ROOT / "contract_004" / "AMENDMENT_001.md"
AMENDMENT_SHA_QUOTED = "e6e4ae6618dc1e89"   # prefix quoted in the independent message; the file read hashes to b122c805...
L1_SCRIPT = REPO / "scripts" / "r3_5_l1_maps.py"
L1_SHA = "b316497077dc03dd344918aeb8f47b441b85efc2ed45b71da65981d49650f47b"
TESTS = REPO / "tests" / "test_r3_5_l2_alignment.py"
S0 = ROOT / "s0_pairs" / "run_001"
BIND = ROOT / "s0_bind" / "run_003"
S1 = ROOT / "s1" / "run_001"
S2 = ROOT / "s2" / "run_003"
TMP = Path("/HOME/.claude/jobs/355449d9/tmp/r35l2")
PY = "/HOME/miniconda3/bin/python3"

L1_SET_C = "C|none|None|connected_components|None"
L1_SET_B = "B|meancost|none|None"
HEADROOM = 0.02          # NA defined only when S* - S0 >= 0.02 (specification 3)
MIN_ROWS = 20            # per-class threshold on annotated reference rows (specification 3, 4)
SHARE_MIN = 0.20         # annotated-mass sensitivity (specification 3)
K_NN = 10                # design 3a
CAP_PER_SLIDE, CAP_TOTAL = 400, 200_000
F1_TOL = 1e-9
F1_COST_TOL = 1e-5       # float32 ot.dist cosine vs float64 recomputation (side verification)
F2_TOL = 1e-9
F3_TOL = 1e-9
F4_TOL = 0.03
# F2 gate.  "literal": specification 7 as written (design-1 kappa (S - S0)/(1 - S0) with the specification-3 restricted-mass
# S0 equals L1 excess_bin / (1 - chance_bin)) -- fails by definition (l2/run_001/DIAGNOSIS_F2.md). "components" (v2):
# S = agreement_bin, tile-count chance = chance_bin, L1-convention kappa = L1 kappa, annotated mass, all to 1e-9.
F2_GATE = "literal"
SMOKE_REFS = ("4288", "4298", "13748")
CLASSES5 = (1, 2, 3, 4, 5)
CLASSES2 = (1, 2)
CNAME5 = {1: "tumour", 2: "immune", 3: "necrosis", 4: "nontumour", 5: "artefact"}
CNAME2 = {1: "tumour", 2: "nontumour_pooled"}
# design-2 class groups on one side: unannotated reported (shares sum to 1), five classes, pooled non-tumour
D2_GROUPS = {"c0": (0,), "c1": (1,), "c2": (2,), "c3": (3,), "c4": (4,), "c5": (5,), "nt": (2, 3, 4, 5)}
D2_GROUP_NAME = {"c0": "unannotated", "c1": "tumour", "c2": "immune", "c3": "necrosis", "c4": "nontumour",
                 "c5": "artefact", "nt": "nontumour_pooled"}


def _load_l1():
    spec = importlib.util.spec_from_file_location("r3_5_l1_maps", L1_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L1 = _load_l1()
sha256_file = L1.sha256_file
now = L1.now


# ====================================================================== pure kernels (unit-tested, F5)
def restricted_matrix(i, j, w, cq, cr, classes):
    """Design-1 matrix.  Plan links (i = query row, j = reference row, w = mass); cq/cr = class per query/reference
    row (0 = unannotated).  Keep links with both ends annotated, renormalise their mass to 1 and return
    M[a, b] = mass from reference class classes[a] to query class classes[b], plus the annotated mass share."""
    i, j, w = np.asarray(i), np.asarray(j), np.asarray(w, float)
    a_cls, b_cls = np.asarray(cr)[j], np.asarray(cq)[i]
    ann = (a_cls > 0) & (b_cls > 0)
    mass = float(w[ann].sum())
    K = len(classes)
    Mx = np.zeros((K, K))
    if mass <= 0:
        return np.full((K, K), np.nan), 0.0
    pos = {c: k for k, c in enumerate(classes)}
    ai = np.array([pos[c] for c in a_cls[ann]], dtype=np.int64)
    bi = np.array([pos[c] for c in b_cls[ann]], dtype=np.int64)
    np.add.at(Mx, (ai, bi), w[ann] / mass)
    return Mx, mass


def alignment_from(S, p, q, headroom_min=HEADROOM) -> dict:
    """Same-class mass S with reference marginal p and query marginal q -> S0, S*, NA, kappa."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    S0 = float(p @ q)
    Ss = float(np.minimum(p, q).sum())
    h = Ss - S0
    no_head = not (h >= headroom_min)
    return {"S": float(S), "S0": S0, "Sstar": Ss, "headroom": h, "no_headroom": bool(no_head),
            "NA": float("nan") if no_head else float((S - S0) / h),
            "kappa": float((S - S0) / (1.0 - S0)) if (1.0 - S0) > 1e-12 else float("nan")}


def alignment(Mx, headroom_min=HEADROOM) -> dict:
    if not np.all(np.isfinite(Mx)):
        out = {k: float("nan") for k in ("S", "S0", "Sstar", "headroom", "NA", "kappa")}
        out["no_headroom"] = True
        out["p"] = np.full(Mx.shape[0], np.nan)
        out["q"] = np.full(Mx.shape[0], np.nan)
        return out
    p, q = Mx.sum(1), Mx.sum(0)
    out = alignment_from(float(np.trace(Mx)), p, q, headroom_min)
    out["p"], out["q"] = p, q
    return out


def count_marginals(cls_rows, classes) -> np.ndarray:
    """Annotated tile-count shares of one slide (amendment 001 A1; L1 chance_bin convention)."""
    c = np.asarray(cls_rows)
    c = c[c > 0]
    if not len(c):
        return np.full(len(classes), np.nan)
    return np.array([(c == k).mean() for k in classes])


def alignment_counts(Mx, p_cnt, q_cnt, headroom_min=HEADROOM) -> dict:
    """Design 1 (amendment 001 A1): S from the restricted, renormalised plan mass; S0 and S* from the annotated
    tile-count shares p_cnt (reference) and q_cnt (query)."""
    if not (np.all(np.isfinite(Mx)) and np.all(np.isfinite(p_cnt)) and np.all(np.isfinite(q_cnt))):
        out = {k: float("nan") for k in ("S", "S0", "Sstar", "headroom", "NA", "kappa")}
        out["no_headroom"] = True
        return out
    return alignment_from(float(np.trace(Mx)), p_cnt, q_cnt, headroom_min)


def retention(Mx, p, q, n_ref_rows, min_rows=MIN_ROWS) -> list:
    """Per-class retention R_c = M[c,c]/p_c, chance q_c, ceiling min(p_c,q_c)/p_c for classes with >= min_rows
    annotated reference rows (NaN when p_c = 0)."""
    out = []
    for k in range(len(p)):
        if n_ref_rows[k] < min_rows:
            out.append(None)
            continue
        if not (p[k] > 0):
            out.append({"R": float("nan"), "chance": float("nan"), "ceiling": float("nan")})
            continue
        out.append({"R": float(Mx[k, k] / p[k]), "chance": float(q[k]), "ceiling": float(min(p[k], q[k]) / p[k])})
    return out


def class_decomposition(cls_link, w, m, groups=None) -> dict:
    """Design 2: D = sum t*m; per class group g: D_g, mass_g, share s_g = D_g/D, class-normalised cost k_g."""
    groups = groups or D2_GROUPS
    w, m = np.asarray(w, float), np.asarray(m, float)
    cls_link = np.asarray(cls_link)
    D = float((w * m).sum())
    out = {"D": D}
    for g, cs in groups.items():
        sel = np.isin(cls_link, cs)
        Dg = float((w[sel] * m[sel]).sum())
        mg = float(w[sel].sum())
        out[f"D_{g}"] = Dg
        out[f"mass_{g}"] = mg
        out[f"s_{g}"] = Dg / D if D > 0 else float("nan")
        out[f"k_{g}"] = Dg / mg if mg > 0 else float("nan")
    return out


def discriminative_values(cost_same, cost_other) -> np.ndarray:
    """Delta_i = mean over other-group pair rows of cost_i - mean over same-group pair rows of cost_i."""
    return np.mean(np.asarray(cost_other, float), axis=0) - np.mean(np.asarray(cost_same, float), axis=0)


def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def auroc_same_vs_other(k_same, k_other, LC) -> float:
    """AUROC of the class-normalised cost for separating same-group (positive) from other-group pair rows; lower
    cost = same group, so the score is -k."""
    ks, ko = np.asarray(k_same, float), np.asarray(k_other, float)
    ks, ko = ks[np.isfinite(ks)], ko[np.isfinite(ko)]
    if not len(ks) or not len(ko):
        return float("nan")
    score = -np.concatenate([ks, ko])
    y = np.r_[np.ones(len(ks), int), np.zeros(len(ko), int)]
    return LC._auroc(score, y)


def l2n(X, dtype=np.float64):
    X = np.asarray(X, dtype=dtype)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def nn_match(Fr, Fq) -> np.ndarray:
    """Each reference row -> index of its cosine-nearest query row (ties -> lowest query index)."""
    return np.argmax(l2n(Fr) @ l2n(Fq).T, axis=1)


def count_alignment(c_ref, c_q_matched, c_query_all, classes, headroom_min=HEADROOM) -> dict:
    """Design 3b: same-class rate of the unconstrained matching (each reference row weight 1/n) with the
    row-count marginals of the annotated rows of both slides."""
    c_ref, c_q_matched, c_query_all = map(np.asarray, (c_ref, c_q_matched, c_query_all))
    S = float((c_ref == c_q_matched).mean()) if len(c_ref) else float("nan")
    p = np.array([(c_ref == c).mean() for c in classes]) if len(c_ref) else np.full(len(classes), np.nan)
    q = np.array([(c_query_all == c).mean() for c in classes]) if len(c_query_all) else np.full(len(classes), np.nan)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        return {"S": float("nan"), "S0": float("nan"), "Sstar": float("nan"), "headroom": float("nan"),
                "no_headroom": True, "NA": float("nan"), "kappa": float("nan")}
    return alignment_from(S, p, q, headroom_min)


def knn_lopo(X, groups, k=K_NN, chunk=1024) -> np.ndarray:
    """Cosine k-NN of every row among rows of other groups (patients).  Returns (n, k) neighbour indices."""
    Xn = l2n(X, np.float32)
    groups = np.asarray(groups)
    n = len(Xn)
    out = np.empty((n, k), dtype=np.int64)
    for g in pd.unique(groups):
        rows = np.where(groups == g)[0]
        for s in range(0, len(rows), chunk):
            rr = rows[s:s + chunk]
            sim = Xn[rr] @ Xn.T
            sim[:, rows] = -np.inf
            out[rr] = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    return out


def majority_vote(neigh_labels, classes):
    """Majority vote over neighbour labels; ties -> smallest class id (as sklearn KNeighborsClassifier)."""
    classes = np.asarray(classes)
    counts = np.stack([(neigh_labels == c).sum(1) for c in classes], axis=1)
    pred = classes[np.argmax(counts, axis=1)]
    tied = (counts == counts.max(1, keepdims=True)).sum(1) > 1
    return pred, tied


def confusion(y, pred, classes) -> np.ndarray:
    pos = {c: k for k, c in enumerate(classes)}
    C = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(C, (np.array([pos[v] for v in y], dtype=np.int64), np.array([pos[v] for v in pred], dtype=np.int64)), 1)
    return C


def balanced_accuracy(C) -> float:
    C = np.asarray(C, float)
    rs = C.sum(1)
    ok = rs > 0
    return float((np.diag(C)[ok] / rs[ok]).mean()) if ok.any() else float("nan")


def ba_bootstrap(Cp, mult) -> tuple:
    """Whole-patient bootstrap of the pooled confusion: Cp (P, K, K) per patient, mult (B, P)."""
    Cb = np.einsum("bp,pij->bij", mult.astype(float), Cp.astype(float))
    rs = Cb.sum(2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(rs > 0, np.diagonal(Cb, axis1=1, axis2=2) / np.where(rs > 0, rs, 1), np.nan)
    return np.nanmean(rec, axis=1), rec


def mixed_slide_ba(yt, pred, slides, patients, classes, min_rows=MIN_ROWS) -> pd.DataFrame:
    """Amendment 001 A2: per-slide balanced accuracy over the slide's classes with >= min_rows rows; a slide is
    included if it has >= 2 such classes (binary: >= min_rows of each class)."""
    yt, pred, slides, patients = map(np.asarray, (yt, pred, slides, patients))
    out = []
    for s in pd.unique(slides):
        m = slides == s
        ys, ps_ = yt[m], pred[m]
        incl = [c for c in classes if (ys == c).sum() >= min_rows]
        if len(incl) < 2:
            continue
        rec = [float((ps_[ys == c] == c).mean()) for c in incl]
        out.append({"slide_id": s, "patient": patients[m][0], "k_s": len(incl), "balanced_accuracy": float(np.mean(rec)),
                    "n_rows": int(m.sum())})
    return pd.DataFrame(out, columns=["slide_id", "patient", "k_s", "balanced_accuracy", "n_rows"])


def mixed_slide_estimate(ms: pd.DataFrame, count=None, seed=None) -> dict:
    """Mean per patient, then over patients; whole-patient percentile bootstrap (L1 draws)."""
    if not len(ms):
        return {"point": None, "ci95_lo": None, "ci95_hi": None, "n_slides": 0, "n_patients": 0, "chance_per_slide": None}
    pt = ms.assign(ch=1.0 / ms.k_s).groupby("patient")[["balanced_accuracy", "ch"]].mean().reset_index()
    pats, mult = L1.make_patient_draws(list(pt.patient), count or L1.N_BOOT, L1.SEED if seed is None else seed)
    reps = L1.replicate_means(pt.balanced_accuracy, pt.patient, pats, mult)
    out = L1.summarise_reps(float(pt.balanced_accuracy.mean()), reps, len(pt))
    out.update({"n_slides": int(len(ms)), "n_patients": int(len(pt)), "chance_per_slide": float(pt.ch.mean())})
    return out


def wording_map(lo) -> str:
    if lo is not None and lo > 0.70:
        return "localises annotated tumour"
    if lo is not None and lo > 0.50:
        return "modestly localises"
    return "does not localise"


def wording_class(lo, hi) -> str:
    if lo is not None and lo > 0:
        return "the same-diagnosis discrimination is carried by tumour tiles"
    if hi is not None and hi < 0:
        return "is carried by non-tumour tiles (adverse)"
    return "is not specific to tumour tiles"


def wording_d1(lo) -> str:
    return ("matched regions share tissue class above chance after accounting for composition"
            if lo is not None and lo > 0 else "matched regions do not share tissue class above chance after "
                                               "accounting for composition")


def wording_d3(ba_lo, ba_hi, diff_lo) -> str:
    if ba_lo is not None and ba_lo >= 0.80 and diff_lo is not None and diff_lo > 0:
        return ("transport constraint: the maps match tissue composition and must be read with the correspondence "
                "pattern, not as class labels")
    if ba_hi is not None and ba_hi < 0.65:
        return "embedding: the embedding does not separate the classes across patients"
    return "in between: report both numbers without a mechanism claim"


# ====================================================================== run helpers
IMPLEMENTATION_CHOICES = """# R3.5-L2 implementation choices (implementer; written into the run directory before any result exists)

Every item resolves a mechanical point contract 004 + amendment 001 (lead rulings A1-A5, 2026-10-18) do not
fix.

1. Plan side assignment. In every saved plan `.npz`, `i` indexes the QUERY slide's feature rows and `j` the
   REFERENCE slide's feature rows; `shape = (n_query_rows, n_reference_rows)`. Source: L1
   `ot_plan(gen, Fq, ref.F)` ("query rows x reference columns") called in `stage_s1`, saved as
   `i, j = np.nonzero(T)`. Verified by F1 on all 988 plans: shape[0] = query feature-pickle row count and
   shape[1] = reference row count; plans.csv n_query/n_ref agree; row sums = 1/n_query and column sums =
   1/n_ref; and, decisive also for plans with n_query = n_ref, the stored cost `m` of every link equals the
   cosine distance between query row i and reference row j recomputed from the features (tolerance 1e-5;
   float32 ot.dist).
2. Plan location. There is no `s1/run_001b/` directory: the ten re-run references (pool r35l1_s1_run001b)
   wrote into `s1/run_001/refs/<ref>/` after the killed attempts were renamed `<ref>.attempt01/`; those
   attempts are not read. Each plan's sha256 is read back against its reference's DONE.json.
3. Labels. Reference rows: `MapSlide.row_label` (L1). Query rows: `load_labels(q)` at the query feature
   coordinates, label 0 where absent (exactly the L1 C stage). Binary classes by `class_of(binary=True)`.
4. Pair rows are drawn with replacement (C4 draw), so a (reference, query) plan can appear in several P2 rows.
   Per-pair statistics are computed once per plan and copied to every pair row; reference means are over pair
   rows, so duplicates count with their multiplicity (L1 convention).
5. Design 1 under amendment 001 A1: S = same-class share of the restricted, renormalised plan mass (links
   with both tiles annotated); p, q = annotated tile-count shares of the reference and query (L1 chance_bin
   convention); S0 = sum p q, S* = sum min(p, q), NA = (S - S0)/(S* - S0) when S* - S0 >= 0.02 (else "no
   headroom", NaN), kappa = (S - S0)/(1 - S0) (NaN when 1 - S0 <= 1e-12), forced mismatch = 1 - S*. Because S
   comes from the plan's restricted mass while S* comes from tile counts, S can exceed S*, so NA can exceed 1;
   the number of such pairs is reported. Sensitivity columns `*_massmarg` repeat S0, S*, NA, kappa and the
   forced mismatch with the restricted-mass marginals (contract-3 original). `kappa_L1conv` (L1's own
   `cross_slide_agreement`) is kept for the F2 regression. Mutual top-1 uses the same tile-count p, q.
6. Design-1 per-class retention: the >= 20-row threshold counts the reference slide's OT rows of class c
   (whole slide). Primary (A1): R_c = M[c,c]/p_c with tile-count p_c, chance q_c, ceiling min(p_c,q_c)/p_c
   (R_c can exceed 1 for the reason in item 5). Sensitivity `R_massmarg`: M[c,c]/p_mass_c (a true conditional
   share), chance q_mass_c, ceiling min(p_mass_c,q_mass_c)/p_mass_c. NaN when the denominator is 0.
7. Design-1 matrices: per pair, the observed row c = M[c,:]/p_mass_c (the conditional distribution of the
   class-c reference mass over query classes); independence row c = tile-count q; ceiling = diagonal
   min(p_c,q_c)/p_c with tile-count p, q (off-diagonals not fixed, left blank). Rows need p_mass_c > 0 and
   p_c > 0. `matrices_D1.csv` + PNGs = mean over pairs per arm (contract 3, ratified A4);
   `matrices_D1_per_reference.csv` = mean per reference, then over references (A4). Full plan only. Mutual
   top-1 sparsification: `L1.sparsify(T, "mutual_topk", 1, OV)` (the imported `sparsify_transport_matrix`)
   under the NPY_DISABLE_CPU_FEATURES pin (the run refuses AVX-512 dispatch).
8. Design-2 per-tile cost: `L1.b_tile_values(T, Mc, "meancost")` on the full dense plan (identical code path
   to L1 family B). Discriminative map: Delta over the reference's OT rows, restricted to `MapSlide.rows`,
   projected by `L1.project(.., "mean")`, then min-max scaled WITHOUT inversion (positive = better matched by
   same-group queries; scaling does not change AUROC/AP/pointing) and scored by
   `L1.score_continuous(.., min_neg=1)`. Yardstick: `L1.normalise_invert(L1.project(mean same-group cost))`
   (= -mean same-group cost). Overlays show the raw projected Delta on a zero-centred diverging scale.
9. Class discrimination AUROC_c: `localization_cli._auroc` with score -k_c and positives = same-group pair
   rows (10 vs 10, duplicates kept); computed for classes with >= 20 reference rows, for pooled non-tumour
   (labels 2-5, >= 20 rows), for unannotated rows (descriptive, >= 20 rows) and for the full distance D (no
   threshold). The query-side mirror (class of the query tile) is in per_pair_D2 only; no query-side AUROC.
10. Reference 13748 has no same-group pairs: it has no discriminative map, yardstick or AUROC_c; it enters
    design-1/3 other-group and all-pair summaries.
11. Bootstrap: one PCG64(123) draw matrix (B = 5,000) per population from `L1.make_patient_draws` over that
    population's reference patients (pooled 46, development 24, report 22); endpoints with missing
    references use `L1.replicate_means` (replicates with no finite reference dropped; count reported);
    percentile 2.5/97.5. Paired contrasts are per-reference differences.
12. Design 3a: rows = `MapSlide.F` rows with `row_label > 0` of the 202 annotated library slides (pool.csv
    in_pool = in_final_library = 1); 67,051 rows < 200,000, so no cap. Float32 L2-normalised vectors, cosine
    similarity, rows of the same `UID.Subject` excluded, top-10 by `np.argpartition` (pinned dispatch),
    majority vote with ties to the smallest class id (sklearn-equivalent; tied count reported). Primary
    reading (A2): per-slide balanced accuracy over the slide's classes with >= 20 annotated rows, for slides
    with >= 2 such classes (binary: >= 20 tumour and >= 20 pooled non-tumour); rows of other classes in that
    slide do not enter it, while their predictions may fall in any class. Mean per patient, then over patients;
    whole-patient bootstrap (L1 draws over the patients that have such a slide). Its chance is 1/K_s averaged
    the same way: 0.5 for binary, the mean of 1/K_s for 5-class. Secondary: pooled all-slide balanced accuracy
    (mean recall over classes with >= 1 true row) and per-class recall with the whole-patient bootstrap of the
    pooled confusion. Controls (F4, A2): (i) global permutation of all labels (one PCG64(123) permutation);
    pooled balanced accuracy within 0.03 of 1/K. (ii) Within-slide shuffle (per slide in numeric order, one
    PCG64(123) stream); mixed-slide per-slide balanced accuracy within 0.03 of its chance (binary 0.5; 5-class
    mean 1/K_s; the deviation from 1/5 is also reported). Both controls are checked at both granularities.
13. Design 3b: cosine (float64) argmax of each annotated reference row over the annotated query rows (ties to
    the lowest index); marginals = annotated row-count shares (the same p, q as design 1 under A1); NA, kappa
    and headroom as in item 5; NA_NN - NA_OT only where both are defined. `_massmarg` = against design-1
    NA_massmarg (sensitivity).
14. Design 3c: forced mismatch 1 - S* with tile-count marginals (primary under A1) and `_massmarg`.
15. Wording rules (contract 6) applied mechanically with the inequalities as written. Primary readings (A3):
    design-1 NA = same-group, binary, full plan, all pairs; switch NA_NN - NA_OT = same-group, binary; probe
    = binary mixed-slide per-slide balanced accuracy (A2). The other readings are reported without a wording
    rule.
16. Overlay references: the Fig. 5 reference slides (15155 and 16867 for RMS, 7330 for RB) are not P2
    references, so the two P2 references of lowest numeric id (4288, 4298) are used (ratified A4).
17. F3 is evaluated on every row of L1 per_pair_B at B|meancost|none|None (P1 and P2 rows), a superset of
    "where the pair exists in P1"; counts by design are reported.
"""


def snapshot_script() -> dict:
    sha = sha256_file(Path(__file__).resolve())
    snap = OUT / "script_snapshots"
    snap.mkdir(parents=True, exist_ok=True)
    existing = sorted(snap.glob(f"r3_5_l2_alignment_v*_{sha[:8]}.py"))
    if existing:
        return {"snapshot": str(existing[0]), "sha256": sha}
    n = len(list(snap.glob("r3_5_l2_alignment_v*_*.py"))) + 1
    p = snap / f"r3_5_l2_alignment_v{n}_{sha[:8]}.py"
    p.write_bytes(Path(__file__).resolve().read_bytes())
    return {"snapshot": str(p), "sha256": sha}


class Stop(Exception):
    pass


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def cpu_s() -> float:
    return time.process_time()


# ====================================================================== stage: bind (F6)
def stage_bind(d: Path) -> dict:
    files = []
    if sha256_file(CONTRACT) != CONTRACT_SHA:
        raise Stop(f"contract sha256 differs from {CONTRACT_SHA}")
    files.append(L1.binding(CONTRACT, "contract_004"))
    am = L1.binding(AMENDMENT, "contract_004_amendment_001")
    am["sha256_prefix_quoted_by_lead_message"] = AMENDMENT_SHA_QUOTED
    am["matches_quoted_prefix"] = am["sha256"].startswith(AMENDMENT_SHA_QUOTED)
    files.append(am)
    if sha256_file(L1_SCRIPT) != L1_SHA:
        raise Stop("L1 script sha256 differs from the L1 final version")
    files.append(L1.binding(L1_SCRIPT, "l1_script"))
    files.append(L1.binding(TESTS, "l2_tests"))
    s0rec = L1.verify_complete(S0)
    for name in ("pairs_P2.csv", "pairs_P1.csv", "split.csv", "references.csv", "COMPLETE.json"):
        b = L1.binding(S0 / name, f"s0_pairs:{name}")
        if name != "COMPLETE.json":
            b["readback_vs_l1"] = b["sha256"] == s0rec["outputs"][name]
        files.append(b)
    bindrec = L1.verify_complete(BIND)
    for name in ("meta_tiles.pkl", "input_bindings.json", "per_slide_bindings.csv"):
        b = L1.binding(BIND / name, f"s0_bind:{name}")
        b["readback_vs_l1"] = b["sha256"] == bindrec["outputs"][name]
        files.append(b)
    s2rec = L1.verify_complete(S2)
    for name in ("per_pair_C.csv.gz", "per_pair_B.csv.gz", "COMPLETE.json"):
        b = L1.binding(S2 / name, f"s2:{name}")
        if name != "COMPLETE.json":
            b["readback_vs_l1"] = b["sha256"] == s2rec["outputs"][name]
        files.append(b)
    pool = L1.bound_path(BIND, "pool")
    files.append({**L1.binding(pool, "pool"), "readback_vs_l1": True})
    for role in [f"code:{k}" for k in L1.CODE_FILES]:
        files.append({**L1.binding(L1.bound_path(BIND, role), role), "readback_vs_l1": True})
    # plan manifest: every plan of every reference, sha256 read back against the reference's S1 DONE.json
    p2 = pd.read_csv(S0 / "pairs_P2.csv", dtype=str)
    refs = L1.numeric_sorted(set(p2.reference))
    man = []
    for r in refs:
        done = json.loads((S1 / "refs" / r / "DONE.json").read_text())
        pl = pd.read_csv(S1 / "refs" / r / "plans.csv", dtype={"reference": str, "query": str})
        for q in pl["query"]:
            name = f"plans/{q}.npz"
            h = sha256_file(S1 / "refs" / r / name)
            man.append({"reference": r, "query": q, "path": str(S1 / "refs" / r / name), "sha256": h,
                        "readback_vs_l1": h == done["outputs"][name], "s1_done_sha256": sha256_file(S1 / "refs" / r / "DONE.json")})
    man = pd.DataFrame(man)
    man.to_csv(d / "plan_manifest.csv", index=False)
    # per-slide features/labels: read back against s0_bind per_slide_bindings where L1 recorded them
    psb = pd.read_csv(BIND / "per_slide_bindings.csv", dtype=str).set_index("slide_id")
    p1 = pd.read_csv(S0 / "pairs_P1.csv", dtype=str)
    poold = pd.read_csv(pool, dtype=str)
    lib = poold[(poold.in_pool == "1") & (poold.in_final_library == "1")].slide_id
    slides = L1.numeric_sorted(set(p2.reference) | set(p2["query"]) | set(p1["query"]) | set(lib))
    ps = []
    for s in slides:
        fp, lp = L1.FEATURES / f"{s}.pkl", L1.LABELS / f"{s}.csv"
        rec = {"slide_id": s, "features_sha256": sha256_file(fp),
               "labels_sha256": sha256_file(lp) if lp.exists() else None}
        if s in psb.index:
            rec["features_readback_vs_l1"] = rec["features_sha256"] == psb.at[s, "features_sha256"]
            l1l = psb.at[s, "labels_sha256"]
            rec["labels_readback_vs_l1"] = rec["labels_sha256"] == (None if pd.isna(l1l) else l1l)
        else:
            rec["features_readback_vs_l1"] = None
            rec["labels_readback_vs_l1"] = None
        ps.append(rec)
    ps = pd.DataFrame(ps)
    ps.to_csv(d / "slide_bindings.csv", index=False)
    script_sha = sha256_file(Path(__file__).resolve())
    bindings = {"created": now(), "script": str(Path(__file__).resolve()), "script_sha256": script_sha,
                "files": files, "plan_manifest": {"file": "plan_manifest.csv", "sha256": sha256_file(d / "plan_manifest.csv"),
                                                  "n_plans": int(len(man)), "n_references": len(refs)},
                "slide_bindings": {"file": "slide_bindings.csv", "sha256": sha256_file(d / "slide_bindings.csv"),
                                   "n_slides": int(len(ps))}}
    jdump(d / "bindings.json", bindings)
    # readback: re-read the file just written and compare every recorded hash with the file on disk
    back = json.loads((d / "bindings.json").read_text())
    rb_files = all(sha256_file(f["path"]) == f["sha256"] for f in back["files"])
    rb_l1 = all(f.get("readback_vs_l1", True) for f in back["files"])
    rb_man = (sha256_file(d / "plan_manifest.csv") == back["plan_manifest"]["sha256"]) and bool(man.readback_vs_l1.all())
    known = ps[ps.features_readback_vs_l1.notna()]
    rb_sl = bool(known.features_readback_vs_l1.astype(bool).all() and
                 known.labels_readback_vs_l1.astype(bool).all())
    f6 = {"pass": bool(rb_files and rb_l1 and rb_man and rb_sl), "files_rehashed_equal": rb_files,
          "files_equal_l1_records": rb_l1, "n_plans": int(len(man)), "plans_equal_s1_done": bool(man.readback_vs_l1.all()),
          "plan_manifest_sha256": back["plan_manifest"]["sha256"], "n_slides": int(len(ps)),
          "n_slides_with_l1_record": int(len(known)), "slides_equal_l1_records": rb_sl,
          "contract_sha256": CONTRACT_SHA, "l1_script_sha256": L1_SHA, "script_sha256": script_sha}
    return f6


# ====================================================================== stage: per-reference loop (F1, F2, F3, D1, D2, D3b)
def run_pairs(d: Path, LC, OV, meta, only_refs=None) -> dict:
    p2 = pd.read_csv(S0 / "pairs_P2.csv", dtype=str)
    refs = L1.numeric_sorted(set(p2.reference))
    if only_refs:                                   # smoke test only (never a study run)
        refs = [r for r in refs if r in set(only_refs)]
    Lc = pd.read_csv(S2 / "per_pair_C.csv.gz", dtype={"reference": str, "query": str}, float_precision="round_trip")
    Lc = Lc[Lc.setting == L1_SET_C].set_index("pair_id")
    Lb = pd.read_csv(S2 / "per_pair_B.csv.gz", dtype={"reference": str, "query": str}, float_precision="round_trip")
    Lb = Lb[Lb.setting == L1_SET_B]
    f1rows, d1rows, d2rows, d3rows, f3rows, refrows, costrows = [], [], [], [], [], [], []
    mats = []      # per-pair row-normalised matrices
    for r in refs:
        ref = L1.MapSlide(r, meta[r], LC=LC)
        cr5 = L1.class_of(ref.row_label, False)
        cr2 = L1.class_of(ref.row_label, True)
        nr = len(ref.coords)
        pl = pd.read_csv(S1 / "refs" / r / "plans.csv", dtype={"reference": str, "query": str},
                         float_precision="round_trip")
        rows_r = p2[p2.reference == r]
        p2q = set(rows_r["query"])
        lb_r = Lb[Lb.reference == r]
        cost_by_q = {}
        Fr64 = ref.F.astype(np.float64)
        nrm_r = np.linalg.norm(Fr64, axis=1)
        for _, prow in pl.iterrows():
            q = prow["query"]
            path = S1 / "refs" / r / "plans" / f"{q}.npz"
            z = np.load(path)
            ii, jj, tt = z["i"].astype(np.int64), z["j"].astype(np.int64), z["t"].astype(np.float64)
            mm = z["m"]
            shape = tuple(int(v) for v in z["shape"])
            Fq, qc = L1.load_features(q)
            # ---- F1
            dist_sum = float((tt * mm.astype(np.float64)).sum())
            Fq64 = Fq.astype(np.float64)
            cos = 1.0 - (Fq64[ii] * Fr64[jj]).sum(1) / (np.linalg.norm(Fq64[ii], axis=1) * nrm_r[jj])
            rs = np.bincount(ii, weights=tt, minlength=shape[0])
            cs = np.bincount(jj, weights=tt, minlength=shape[1])
            f1rows.append({"reference": r, "query": q, "n_query_rows": len(qc), "n_ref_rows": nr,
                           "shape0": shape[0], "shape1": shape[1], "plans_csv_n_query": int(prow.n_query),
                           "plans_csv_n_ref": int(prow.n_ref), "distance_saved": float(z["distance"][0]),
                           "sum_tm": dist_sum, "abs_diff_distance": abs(dist_sum - float(z["distance"][0])),
                           "max_i_ok": bool(ii.max() < shape[0]), "max_j_ok": bool(jj.max() < shape[1]),
                           "square": shape[0] == shape[1],
                           "max_rowsum_dev": float(np.abs(rs - 1.0 / shape[0]).max()),
                           "max_colsum_dev": float(np.abs(cs - 1.0 / shape[1]).max()),
                           "max_cost_dev": float(np.abs(cos - mm.astype(np.float64)).max())})
            # ---- dense plan (L1 C-stage reconstruction) and per-tile mean cost (F3, design 2)
            T, Mc = L1.load_plan_with_costs(path)
            val, linked = L1.b_tile_values(T, Mc, "meancost")
            cost_by_q[q] = val
            hB = L1.normalise_invert(L1.project(val[ref.rows], ref.owners, "mean"))
            sB = L1.score_continuous(hB, ref.patch_label, LC, min_neg=1)
            for _, lrow in lb_r[lb_r["query"] == q].iterrows():
                f3rows.append({"pair_id": lrow.pair_id, "design": lrow.design, "reference": r, "query": q,
                               "auroc_l1": float(lrow.auroc_A), "auroc_l2": sB.get("auroc_A", float("nan")),
                               "abs_diff": abs(float(lrow.auroc_A) - sB.get("auroc_A", float("nan")))})
            if q not in p2q:
                del T, Mc
                continue
            labq = L1.load_labels(q)
            lab_q = np.array([labq.get(c, 0) for c in qc], dtype=np.int8)
            cq5, cq2 = L1.class_of(lab_q, False), L1.class_of(lab_q, True)
            base = {"reference": r, "query": q}
            # ---- design 1 (full plan and mutual top-1) + F2 inputs
            rec1 = dict(base)
            for plan_name, Tp in (("full", T), ("mt1", L1.sparsify(T, "mutual_topk", 1, OV))):
                if plan_name == "full":
                    pi, pj, pw = ii, jj, tt
                else:
                    pi, pj = np.nonzero(Tp)
                    pw = Tp[pi, pj]
                rec1[f"{plan_name}_edges"] = int(len(pw))
                for gran, classes, cq, cr, names in (("bin", CLASSES2, cq2, cr2, CNAME2), ("5", CLASSES5, cq5, cr5, CNAME5)):
                    Mx, amass = restricted_matrix(pi, pj, pw, cq, cr, classes)
                    pc, qc_ = count_marginals(cr, classes), count_marginals(cq, classes)
                    al = alignment_counts(Mx, pc, qc_)          # amendment 001 A1: tile-count marginals (primary)
                    am = alignment(Mx)                          # restricted-mass marginals (sensitivity)
                    pre = f"{plan_name}_{gran}"
                    rec1[f"{pre}_annotated_mass"] = amass
                    rec1[f"{pre}_annotated_mass_share"] = amass / float(pw.sum()) if pw.sum() > 0 else float("nan")
                    for k_ in ("S", "S0", "Sstar", "headroom", "no_headroom", "NA", "kappa"):
                        rec1[f"{pre}_{k_}"] = al[k_]
                    for k_ in ("S0", "Sstar", "headroom", "no_headroom", "NA", "kappa"):
                        rec1[f"{pre}_{k_}_massmarg"] = am[k_]
                    rec1[f"{pre}_forced_mismatch"] = 1.0 - al["Sstar"]
                    rec1[f"{pre}_forced_mismatch_massmarg"] = 1.0 - am["Sstar"]
                    nref = [int((cr == c).sum()) for c in classes]
                    ret = retention(Mx, pc, qc_, nref)
                    retm = retention(Mx, am["p"], am["q"], nref)
                    for kk, c in enumerate(classes):
                        nm = names[c]
                        rec1[f"{pre}_p_cnt_{nm}"] = float(pc[kk])
                        rec1[f"{pre}_q_cnt_{nm}"] = float(qc_[kk])
                        rec1[f"{pre}_p_mass_{nm}"] = float(am["p"][kk])
                        rec1[f"{pre}_q_mass_{nm}"] = float(am["q"][kk])
                        rec1[f"{pre}_nrefrows_{nm}"] = nref[kk]
                        for suf, rr_ in (("", ret), ("_massmarg", retm)):
                            x = rr_[kk]
                            rec1[f"{pre}_R{suf}_{nm}"] = x["R"] if x else float("nan")
                            rec1[f"{pre}_Rchance{suf}_{nm}"] = x["chance"] if x else float("nan")
                            rec1[f"{pre}_Rceiling{suf}_{nm}"] = x["ceiling"] if x else float("nan")
                    if plan_name == "full":
                        ag = L1.cross_slide_agreement(T, lab_q, ref.row_label, binary=(gran == "bin"))
                        rec1[f"{pre}_chance_count"] = ag["chance"]
                        rec1[f"{pre}_agreement_l1fn"] = ag["agreement"]
                        rec1[f"{pre}_kappa_L1conv"] = ((ag["agreement"] - ag["chance"]) / (1 - ag["chance"])
                                                      if (1 - ag["chance"]) > 1e-12 else float("nan"))
                        # matrices: observed = conditional distribution of the row's restricted mass; independence
                        # row = tile-count q; ceiling diagonal = min(p_cnt, q_cnt)/p_cnt (amendment 001 A1, A4)
                        for kk, c in enumerate(classes):
                            if not (am["p"][kk] > 0 and pc[kk] > 0):
                                continue
                            for kind, rowv in (("observed", Mx[kk] / am["p"][kk]), ("independence", qc_),
                                               ("ceiling", np.where(np.arange(len(classes)) == kk,
                                                                    min(pc[kk], qc_[kk]) / pc[kk], np.nan))):
                                for ll, c2 in enumerate(classes):
                                    mats.append((r, q, gran, kind, names[c], names[c2], float(rowv[ll])))
            d1rows.append(rec1)
            # ---- design 2 (reference side primary, query side mirror)
            rec2 = dict(base)
            for side, cls in (("ref", cr5[jj]), ("qry", cq5[ii])):
                dec = class_decomposition(cls, tt, mm)
                if side == "ref":
                    rec2["D"] = dec["D"]
                for k_, v in dec.items():
                    if k_ != "D":
                        rec2[f"{side}_{k_}"] = v
                lab_side = cr5 if side == "ref" else cq5
                for g, cs_ in D2_GROUPS.items():
                    rec2[f"{side}_nrows_{g}"] = int(np.isin(lab_side, cs_).sum())
            d2rows.append(rec2)
            # ---- design 3b (unconstrained NN matching)
            rec3 = dict(base)
            ra, qa = np.where(cr5 > 0)[0], np.where(cq5 > 0)[0]
            nn = nn_match(ref.F[ra], Fq[qa]) if len(ra) and len(qa) else np.zeros(0, dtype=np.int64)
            for gran, classes, cq, cr in (("bin", CLASSES2, cq2, cr2), ("5", CLASSES5, cq5, cr5)):
                if len(ra) and len(qa):
                    al = count_alignment(cr[ra], cq[qa][nn], cq[qa], classes)
                else:
                    al = {k_: float("nan") for k_ in ("S", "S0", "Sstar", "headroom", "NA", "kappa")}
                    al["no_headroom"] = True
                for k_ in ("S", "S0", "Sstar", "headroom", "no_headroom", "NA", "kappa"):
                    rec3[f"nn_{gran}_{k_}"] = al[k_]
                rec3[f"ot_{gran}_NA"] = rec1[f"full_{gran}_NA"]
                rec3[f"ot_{gran}_kappa"] = rec1[f"full_{gran}_kappa"]
                both = np.isfinite(al["NA"]) and np.isfinite(rec1[f"full_{gran}_NA"])
                rec3[f"diff_{gran}_NA_NN_minus_OT"] = al["NA"] - rec1[f"full_{gran}_NA"] if both else float("nan")
                bm = np.isfinite(al["NA"]) and np.isfinite(rec1[f"full_{gran}_NA_massmarg"])
                rec3[f"diff_{gran}_NA_NN_minus_OT_massmarg"] = (al["NA"] - rec1[f"full_{gran}_NA_massmarg"]
                                                               if bm else float("nan"))
            rec3["n_ref_annotated_rows"] = int(len(ra))
            rec3["n_query_annotated_rows"] = int(len(qa))
            d3rows.append(rec3)
            del T, Mc
        # ---- per-reference design-2 maps and class discrimination
        same = rows_r[rows_r.arm == "same_group"]
        other = rows_r[rows_r.arm == "other_group"]
        rr = {"reference": r, "n_same_rows": int(len(same)), "n_other_rows": int(len(other)),
              "n_ref_ot_rows": nr, "n_mapped_rows": int(len(ref.rows))}
        if len(same) and len(other):
            cs_same = [cost_by_q[q] for q in same["query"]]
            cs_other = [cost_by_q[q] for q in other["query"]]
            delta = discriminative_values(cs_same, cs_other)
            h_disc = minmax(L1.project(delta[ref.rows], ref.owners, "mean"))
            sd = L1.score_continuous(h_disc, ref.patch_label, LC, min_neg=1)
            h_yard = L1.normalise_invert(L1.project(np.mean(cs_same, axis=0)[ref.rows], ref.owners, "mean"))
            sy = L1.score_continuous(h_yard, ref.patch_label, LC, min_neg=1)
            for pre, s_ in (("disc", sd), ("yard", sy)):
                for k_ in ("auroc_A", "ap_A", "pointing_A", "tumour_fraction_A", "n_annotated"):
                    v = s_.get(k_, float("nan"))
                    rr[f"{pre}_{k_}"] = float(v) if not isinstance(v, (bool, np.bool_)) else float(bool(v))
            rr["disc_minus_yard_auroc_A"] = rr["disc_auroc_A"] - rr["yard_auroc_A"]
            # keep delta for overlays
            refrows.append(("delta", r, delta))
        d2r = pd.DataFrame([x for x in d2rows if x["reference"] == r]).set_index("query")
        if len(same) and len(other):
            for g in list(D2_GROUPS) + ["D"]:
                if g == "D":
                    ks = d2r.loc[same["query"], "D"].values
                    ko = d2r.loc[other["query"], "D"].values
                    rr["auroc_D"] = auroc_same_vs_other(ks, ko, LC)
                    continue
                nrow = int(np.isin(cr5, D2_GROUPS[g]).sum())
                rr[f"nrows_{g}"] = nrow
                if nrow >= MIN_ROWS:
                    rr[f"auroc_{g}"] = auroc_same_vs_other(d2r.loc[same["query"], f"ref_k_{g}"].values,
                                                           d2r.loc[other["query"], f"ref_k_{g}"].values, LC)
                else:
                    rr[f"auroc_{g}"] = float("nan")
            rr["auroc_tumour_minus_nt"] = rr["auroc_c1"] - rr["auroc_nt"]
        refrows.append(("ref", r, rr))
        for q, v in cost_by_q.items():
            if q in p2q:
                costrows.append(("cost", r, q, v, ref.row_label))
        print(f"[l2] reference {r}: {len(pl)} plans", flush=True)
    return {"f1": pd.DataFrame(f1rows), "f3": pd.DataFrame(f3rows), "d1": pd.DataFrame(d1rows),
            "d2": pd.DataFrame(d2rows), "d3": pd.DataFrame(d3rows),
            "refs": pd.DataFrame([x[2] for x in refrows if x[0] == "ref"]),
            "deltas": {x[1]: x[2] for x in refrows if x[0] == "delta"},
            "costs": [x for x in costrows if x[0] == "cost"], "mats": mats, "Lc": Lc}


# ====================================================================== stage: probe (design 3a, F4)
def run_probe(LC, meta, bind_pool, limit_slides=None) -> dict:
    pool = pd.read_csv(bind_pool, dtype=str)
    ann = pool[(pool.in_pool == "1") & (pool.in_final_library == "1")]
    ids = L1.numeric_sorted(ann.slide_id)
    if limit_slides:                                # smoke test only (never a study run)
        ids = ids[:limit_slides]
    pat = dict(zip(ann.slide_id, ann["UID.Subject"]))
    grp = dict(zip(ann.slide_id, ann.diagnostic_group))
    X, y, sl, pp = [], [], [], []
    counts = []
    for s in ids:
        ms = L1.MapSlide(s, meta[s], LC=LC)
        a = np.where(ms.row_label > 0)[0]
        counts.append({"slide_id": s, "patient": pat[s], "group": grp[s], "n_rows": len(ms.row_label),
                       "n_annotated_rows": int(len(a))})
        X.append(ms.F[a])
        y.append(ms.row_label[a].astype(np.int64))
        sl += [s] * len(a)
        pp += [pat[s]] * len(a)
    total = int(sum(c["n_annotated_rows"] for c in counts))
    capped = False
    if total > CAP_TOTAL:                      # specification 5: cap 400 per slide by seeded draw
        rng = np.random.Generator(np.random.PCG64(L1.SEED))
        keep = []
        off = 0
        for c in counts:
            n = c["n_annotated_rows"]
            idx = np.arange(off, off + n)
            keep.append(np.sort(rng.choice(idx, size=min(n, CAP_PER_SLIDE), replace=False)))
            off += n
        keep = np.concatenate(keep)
        capped = True
    X = np.concatenate(X)
    y = np.concatenate(y)
    sl = np.array(sl)
    pp = np.array(pp)
    if capped:
        X, y, sl, pp = X[keep], y[keep], sl[keep], pp[keep]
    t0 = time.time()
    nb = knn_lopo(X, pp, K_NN)
    secs = time.time() - t0
    # shuffle control: per slide (numeric order), PCG64(123), permutation of the slide's labels
    rng = np.random.Generator(np.random.PCG64(L1.SEED))
    ysh = y.copy()
    for s in ids:
        idx = np.where(sl == s)[0]
        if len(idx):
            ysh[idx] = y[idx][rng.permutation(len(idx))]
    res = {"n_slides": len(ids), "n_patients": int(len(set(pp))), "n_rows": int(len(y)), "total_annotated_rows": total,
           "capped": capped, "class_counts_5": {CNAME5[c]: int((y == c).sum()) for c in CLASSES5},
           "knn_seconds": round(secs, 1), "k": K_NN}
    pats_all = sorted(set(pp))
    pats, mult = L1.make_patient_draws(pats_all, L1.N_BOOT, L1.SEED)
    pidx = pd.Index(pats).get_indexer(pp)
    yglob = y[np.random.Generator(np.random.PCG64(L1.SEED)).permutation(len(y))]   # amendment 001 A2 control (i)
    conf_out, slide_out = [], []
    for label_kind, yy in (("observed", y), ("shuffle_control", ysh), ("global_permutation", yglob)):
        for gran, classes, names in (("5", CLASSES5, CNAME5), ("bin", CLASSES2, CNAME2)):
            yt = L1.class_of(yy, gran == "bin")
            nl = yt[nb]
            pred, tied = majority_vote(nl, classes)
            C = confusion(yt, pred, classes)
            Cp = np.zeros((len(pats), len(classes), len(classes)), dtype=np.int64)
            pos = {c: k for k, c in enumerate(classes)}
            np.add.at(Cp, (pidx, np.array([pos[v] for v in yt]), np.array([pos[v] for v in pred])), 1)
            assert np.array_equal(Cp.sum(0), C)
            ba = balanced_accuracy(C)
            reps, rec_reps = ba_bootstrap(Cp, mult)
            rec = np.diag(C) / np.maximum(C.sum(1), 1)
            key = f"{label_kind}|{gran}"
            ms = mixed_slide_ba(yt, pred, sl, pp, classes)
            ms["labels"], ms["granularity"] = label_kind, gran
            slide_out.append(ms)
            res[key] = {"mixed_slide_balanced_accuracy": mixed_slide_estimate(ms),
                        "balanced_accuracy": {"point": ba, "ci95_lo": float(np.nanpercentile(reps, 2.5)),
                                              "ci95_hi": float(np.nanpercentile(reps, 97.5)),
                                              "n_replicates_finite": int(np.isfinite(reps).sum()),
                                              "reading": "all slides, pooled rows (secondary, amendment 001 A2)"},
                        "chance": 1.0 / len(classes), "n_tied_votes": int(tied.sum()),
                        "per_class_recall": {names[c]: {"point": float(rec[k]) if C[k].sum() else float("nan"),
                                                        "ci95_lo": float(np.nanpercentile(rec_reps[:, k], 2.5)) if np.isfinite(rec_reps[:, k]).any() else None,
                                                        "ci95_hi": float(np.nanpercentile(rec_reps[:, k], 97.5)) if np.isfinite(rec_reps[:, k]).any() else None,
                                                        "n_true": int(C[k].sum())} for k, c in enumerate(classes)}}
            for a_, c in enumerate(classes):
                for b_, c2 in enumerate(classes):
                    conf_out.append({"labels": label_kind, "granularity": gran, "true": names[c], "pred": names[c2],
                                     "count": int(C[a_, b_])})
    f4 = {}
    for gran, K in (("5", 5), ("bin", 2)):
        g = res[f"global_permutation|{gran}"]["balanced_accuracy"]["point"]
        f4[f"i_global_permutation_pooled|{gran}"] = {"balanced_accuracy": g, "chance": 1.0 / K, "abs_dev": abs(g - 1.0 / K),
                                                     "pass": bool(abs(g - 1.0 / K) <= F4_TOL)}
        m = res[f"shuffle_control|{gran}"]["mixed_slide_balanced_accuracy"]
        f4[f"ii_within_slide_shuffle_mixed_slide|{gran}"] = {
            "balanced_accuracy": m["point"], "chance": m["chance_per_slide"], "abs_dev": abs(m["point"] - m["chance_per_slide"]),
            "abs_dev_from_1_over_5": abs(m["point"] - 0.2) if gran == "5" else None,
            "n_slides": m["n_slides"], "n_patients": m["n_patients"],
            "pass": bool(abs(m["point"] - m["chance_per_slide"]) <= F4_TOL)}
    f4["pass"] = bool(all(v["pass"] for v in f4.values() if isinstance(v, dict)))
    res["mixed_slide_rule"] = ("per-slide balanced accuracy over the slide's classes with >= 20 annotated rows; slide "
                               "included if >= 2 such classes (binary: >= 20 tumour and >= 20 pooled non-tumour rows); "
                               "mean per patient, then over patients; whole-patient bootstrap B = 5,000 seed 123; "
                               "chance = 1/K_s averaged the same way (amendment 001 A2)")
    return {"probe": res, "confusion": pd.DataFrame(conf_out), "counts": pd.DataFrame(counts), "f4": f4,
            "slides": pd.concat(slide_out, ignore_index=True)}


# ====================================================================== estimates (specification 6)
def make_populations(refinfo: pd.DataFrame) -> dict:
    pops = {}
    for name, sel in (("pooled", np.ones(len(refinfo), bool)), ("development", (refinfo.half == "development").values),
                      ("report", (refinfo.half == "report").values)):
        pats = sorted(set(refinfo.ref_patient[sel]))
        p, mult = L1.make_patient_draws(pats, L1.N_BOOT, L1.SEED)
        pops[name] = {"refs": set(refinfo.reference[sel]), "pats": p, "mult": mult}
    return pops


def est(df_ref: pd.DataFrame, col: str, pop: dict) -> dict:
    x = df_ref[df_ref.reference.isin(pop["refs"])]
    x = x[np.isfinite(x[col].astype(float))]
    if not len(x):
        return {"point": None, "ci95_lo": None, "ci95_hi": None, "n_references": 0, "n_patients": 0}
    reps = L1.replicate_means(x[col].astype(float), x.ref_patient, pop["pats"], pop["mult"])
    out = L1.summarise_reps(float(x[col].astype(float).mean()), reps, len(x))
    out["n_patients"] = int(x.ref_patient.nunique())
    return out


def per_ref_mean(pp: pd.DataFrame, cols, arm=None, mask=None) -> pd.DataFrame:
    x = pp if arm in (None, "all") else pp[pp.arm == arm]
    if mask is not None:
        x = x[mask.loc[x.index]]
    return x.groupby(["reference", "ref_patient", "half"])[cols].mean().reset_index()


def build_estimates(pairs: pd.DataFrame, D1: pd.DataFrame, D2: pd.DataFrame, D3: pd.DataFrame, R2: pd.DataFrame,
                    refinfo: pd.DataFrame, probe: dict) -> dict:
    pops = make_populations(refinfo)
    out = {p: {} for p in pops}
    R2 = R2.merge(refinfo[["reference", "ref_patient", "half"]], on="reference")
    for pname, pop in pops.items():
        E = out[pname]
        # ---- design 2 (primary)
        for col in ("disc_auroc_A", "disc_ap_A", "disc_pointing_A", "disc_tumour_fraction_A", "yard_auroc_A",
                    "yard_ap_A", "yard_pointing_A", "disc_minus_yard_auroc_A", "auroc_D",
                    "auroc_c0", "auroc_c1", "auroc_c2", "auroc_c3", "auroc_c4", "auroc_c5", "auroc_nt",
                    "auroc_tumour_minus_nt"):
            if col in R2:
                E[f"D2|{col}"] = est(R2, col, pop)
        sh_cols = [c for c in D2.columns if c.startswith(("ref_s_", "ref_k_", "qry_s_", "qry_k_"))] + ["D"]
        for arm in ("same_group", "other_group"):
            pr = per_ref_mean(D2, sh_cols, arm)
            for col in sh_cols:
                E[f"D2|{arm}|{col}"] = est(pr, col, pop)
        # ---- design 1 (secondary)
        for plan in ("full", "mt1"):
            for gran in ("bin", "5"):
                pre = f"{plan}_{gran}"
                cols = [f"{pre}_{k}" for k in ("NA", "kappa", "S", "S0", "Sstar", "headroom", "annotated_mass_share",
                                                "NA_massmarg", "kappa_massmarg", "S0_massmarg", "Sstar_massmarg",
                                                "headroom_massmarg")]
                if plan == "full":
                    cols += [f"{pre}_kappa_L1conv", f"{pre}_chance_count"]
                    cols += [c for c in D1.columns if c.startswith(f"{pre}_R")]
                D1x = D1.copy()
                D1x[f"{pre}_scored"] = (~D1x[f"{pre}_no_headroom"].astype(bool)).astype(float)
                D1x[f"{pre}_scored_massmarg"] = (~D1x[f"{pre}_no_headroom_massmarg"].astype(bool)).astype(float)
                cols2 = cols + [f"{pre}_scored", f"{pre}_scored_massmarg"]
                for subset in ("all", "share_ge_0.20"):
                    mask = None if subset == "all" else (D1x[f"full_{gran}_annotated_mass_share"] >= SHARE_MIN)
                    for arm in ("same_group", "other_group", "all"):
                        pr = per_ref_mean(D1x, cols2, arm, mask)
                        for col in cols2:
                            E[f"D1|{plan}|{gran}|{subset}|{arm}|{col[len(pre) + 1:]}"] = est(pr, col, pop)
                        sub = D1x if arm == "all" else D1x[D1x.arm == arm]
                        if mask is not None:
                            sub = sub[mask.loc[sub.index]]
                        sub = sub[sub.reference.isin(pop["refs"])]
                        E[f"D1|{plan}|{gran}|{subset}|{arm}|counts"] = {
                            "n_pairs": int(len(sub)), "n_pairs_no_headroom": int(sub[f"{pre}_no_headroom"].astype(bool).sum()),
                            "n_pairs_scored": int((~sub[f"{pre}_no_headroom"].astype(bool)).sum()),
                            "n_pairs_NA_gt_1": int((sub[f"{pre}_NA"].astype(float) > 1).sum()),
                            "n_pairs_scored_massmarg": int((~sub[f"{pre}_no_headroom_massmarg"].astype(bool)).sum())}
        # ---- design 3b / 3c
        d3cols = [c for c in D3.columns if c.startswith(("nn_", "diff_"))]
        d3cols = [c for c in d3cols if not c.endswith("no_headroom")]
        D1c = D1[["pair_id", "full_bin_forced_mismatch", "full_5_forced_mismatch", "full_bin_forced_mismatch_massmarg",
                  "full_5_forced_mismatch_massmarg"]]
        D3x = D3.merge(D1c, on="pair_id")
        cols3 = d3cols + ["full_bin_forced_mismatch", "full_5_forced_mismatch", "full_bin_forced_mismatch_massmarg",
                          "full_5_forced_mismatch_massmarg"]
        for arm in ("same_group", "other_group", "all"):
            pr = per_ref_mean(D3x, cols3, arm)
            for col in cols3:
                E[f"D3|{arm}|{col}"] = est(pr, col, pop)
    # ---- design 3a (library probe; not split by reference halves)
    out["probe_D3a"] = {k: v for k, v in probe.items() if "|" in k}
    # ---- mechanical wording (specification 6 rules; primary readings fixed by amendment 001 A3)
    NOW = "no wording rule (amendment 001 A3: reported beside the primary reading)"
    W = {}
    pb = out["probe_D3a"]["observed|bin"]["mixed_slide_balanced_accuracy"]
    for pname in pops:
        E = out[pname]
        w = {}
        e = E["D2|disc_auroc_A"]
        w["D2_map (primary)"] = {"endpoint": "D2|disc_auroc_A", **e, "wording": wording_map(e["ci95_lo"])}
        e = E["D2|auroc_tumour_minus_nt"]
        w["D2_class (primary)"] = {"endpoint": "D2|auroc_tumour_minus_nt", **e,
                                   "wording": wording_class(e["ci95_lo"], e["ci95_hi"])}
        for gran in ("bin", "5"):
            for arm in ("same_group", "other_group", "all"):
                for plan in ("full", "mt1"):
                    for na in ("NA", "NA_massmarg"):
                        k = f"D1|{plan}|{gran}|all|{arm}|{na}"
                        prim = (gran, arm, plan, na) == ("bin", "same_group", "full", "NA")
                        w[f"D1|{plan}|{gran}|{arm}|{na}" + (" (primary)" if prim else "")] = {
                            "endpoint": k, **E[k], "wording": wording_d1(E[k]["ci95_lo"]) if prim else NOW}
                k = f"D1|full|{gran}|share_ge_0.20|{arm}|NA"
                w[f"D1|full|{gran}|{arm}|NA|share_ge_0.20"] = {"endpoint": k, **E[k], "wording": NOW}
        for gran in ("bin", "5"):
            for arm in ("same_group", "other_group", "all"):
                for suf in ("", "_massmarg"):
                    k = f"D3|{arm}|diff_{gran}_NA_NN_minus_OT{suf}"
                    prim = (gran, arm, suf) == ("bin", "same_group", "")
                    w[f"D3_switch|diff_{gran}{suf}|{arm}" + (" (primary)" if prim else "")] = {
                        "probe_bin_mixed_slide_balanced_accuracy": pb, "diff": {"endpoint": k, **E[k]},
                        "wording": wording_d3(pb["ci95_lo"], pb["ci95_hi"], E[k]["ci95_lo"]) if prim else NOW}
        W[pname] = w
    out["wording"] = W
    return out


# ====================================================================== figures
CLASS_COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4"}
SEQ = ["#f7f9fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def fig_matrices(mat_df: pd.DataFrame, d: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
    cmap.set_bad("#ffffff")
    names_out = []
    for gran, names in (("5", CNAME5), ("bin", CNAME2)):
        labs = [names[c] for c in sorted(names)]
        for arm in ("same_group", "other_group"):
            x = mat_df[(mat_df.granularity == gran) & (mat_df.arm == arm)]
            fig, axes = plt.subplots(1, 3, figsize=(3.6 * 3 + 1, 3.9 if gran == "5" else 3.0), constrained_layout=True)
            for ax, kind in zip(axes, ("observed", "independence", "ceiling")):
                xx = x[x.kind == kind]
                A = np.full((len(labs), len(labs)), np.nan)
                N = np.zeros(len(labs), int)
                for a_, ra in enumerate(labs):
                    for b_, cb in enumerate(labs):
                        v = xx[(xx.ref_class == ra) & (xx.query_class == cb)]
                        if len(v):
                            A[a_, b_] = v.value.iloc[0]
                            N[a_] = int(v.n_pairs.iloc[0])
                im = ax.imshow(np.ma.masked_invalid(A), vmin=0, vmax=1, cmap=cmap)
                for a_ in range(len(labs)):
                    for b_ in range(len(labs)):
                        if np.isfinite(A[a_, b_]):
                            ax.text(b_, a_, f"{A[a_, b_]:.2f}", ha="center", va="center", fontsize=7,
                                    color="#ffffff" if A[a_, b_] > 0.55 else "#0b0b0b")
                ax.set_xticks(range(len(labs)))
                ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(labs)))
                ax.set_yticklabels([f"{l} (n={n})" for l, n in zip(labs, N)], fontsize=7)
                ax.set_xlabel("query-tile class", fontsize=8)
                ax.set_title(kind, fontsize=9)
            axes[0].set_ylabel("reference-tile class", fontsize=8)
            fig.colorbar(im, ax=axes, shrink=0.8, label="share of the row's transported mass")
            fig.suptitle(f"R3.5-L2 design 1, {'5-class' if gran == '5' else 'binary'}, P2 {arm.replace('_', '-')} pairs: "
                         "row-normalised class-to-class transport (full exact plan, links with both tiles annotated), "
                         "mean over pairs; n = pairs with that reference class present. Ceiling = min(p,q)/p on the "
                         "diagonal (off-diagonal not fixed by the marginals).", fontsize=7, wrap=True)
            nm = f"matrix_D1_{gran}_{arm}.png"
            fig.savefig(d / nm, dpi=150)
            plt.close(fig)
            names_out.append(nm)
    return names_out


def fig_overlay(r: str, delta: np.ndarray, rr: dict, LC, meta, d: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch, Rectangle
    import openslide
    ref = L1.MapSlide(r, meta[r], LC=LC)
    pv = L1.project(delta[ref.rows], ref.owners, "mean")
    ws = openslide.OpenSlide(str(L1.WSI).format(r))
    W0, H0 = ws.dimensions
    thumb = ws.get_thumbnail((1400, 1400))
    sc = thumb.size[0] / W0
    locs = np.array(ref.slide.patches, dtype=float)
    vmax = float(np.abs(pv).max()) or 1.0
    div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ec9a8a", "#b8322a"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), constrained_layout=True)
    for ax in axes:
        ax.imshow(thumb, alpha=1.0 if ax is axes[0] else 0.35)
        ax.set_axis_off()
    axes[0].set_title(f"reference {r} ({rr.get('group', '')})", fontsize=9)
    rects = [Rectangle((x * sc, y * sc), 512 * sc, 512 * sc) for x, y in locs]
    pc = PatchCollection(rects, cmap=div, norm=Normalize(-vmax, vmax), linewidth=0)
    pc.set_array(pv)
    axes[1].add_collection(pc)
    fig.colorbar(pc, ax=axes[1], shrink=0.7, label="Delta = mean other-group cost - mean same-group cost")
    axes[1].set_title(f"discriminative map: tumour AUROC {rr.get('disc_auroc_A', float('nan')):.3f}, "
                      f"AP {rr.get('disc_ap_A', float('nan')):.3f}", fontsize=9)
    lab = ref.patch_label
    for c, col in CLASS_COLOURS.items():
        sel = lab == c
        if sel.any():
            axes[2].add_collection(PatchCollection([Rectangle((x * sc, y * sc), 512 * sc, 512 * sc) for x, y in locs[sel]],
                                                   facecolor=col, edgecolor="none", alpha=0.85))
    axes[2].legend(handles=[Patch(color=CLASS_COLOURS[c], label=f"{CNAME5[c]} (n={int((lab == c).sum())})") for c in CNAME5],
                   loc="lower left", fontsize=7, frameon=True)
    axes[2].set_title("pathologist annotation classes (unannotated patches not drawn)", fontsize=9)
    fig.suptitle(f"R3.5-L2 design 2, reference {r}. Rule: per reference tile, cost_i = transport-weighted mean cosine cost "
                 f"of its exact-OT links (full plan); Delta_i = mean of cost_i over the reference's {rr['n_other_rows']} "
                 f"other-group P2 queries minus mean over its {rr['n_same_rows']} same-group P2 queries (positive, red = matched "
                 "better by same-diagnosis slides); patch value = mean of its owning mosaic tiles. Reference choice: the "
                 "Fig. 5 reference slides are not P2 references, so the two P2 references of lowest numeric id are shown. "
                 "No manual choice.", fontsize=8, wrap=True)
    nm = f"overlay_D2_{r}.png"
    fig.savefig(d / nm, dpi=120)
    plt.close(fig)
    return nm


# ====================================================================== stage: run
def run_tests(d: Path) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    bt = TMP / f"pytest_{d.name}"
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(TESTS), "--basetemp", str(bt)]
    pr = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, env={**os.environ})
    (d / "pytest_output.txt").write_text(pr.stdout + pr.stderr)
    last = [l for l in pr.stdout.strip().splitlines() if l.strip()][-1] if pr.stdout.strip() else ""
    return {"pass": pr.returncode == 0, "returncode": pr.returncode, "summary": last,
            "tests_sha256": sha256_file(TESTS)}


def stage_run(a) -> int:
    smoke = bool(getattr(a, "smoke", False))
    TMP.mkdir(parents=True, exist_ok=True)
    d = L1.new_run_dir("smoke", TMP) if smoke else L1.new_run_dir("l2", ROOT)
    t_wall, t_cpu = time.time(), cpu_s()
    checks, timing = {"script_sha256": sha256_file(Path(__file__).resolve()), "F2_gate": F2_GATE}, {}
    try:
        checks["dispatch"] = L1.assert_pinned_dispatch()
        snap = snapshot_script() if not smoke else {"smoke": True}
        checks["snapshot"] = snap
        (d / "IMPLEMENTATION_CHOICES.md").write_text(IMPLEMENTATION_CHOICES)
        # ---- F6
        c0 = cpu_s()
        checks["F6"] = stage_bind(d) if not smoke else {"pass": True, "n_plans": 988, "smoke": True}
        timing["bind_cpu_s"] = cpu_s() - c0
        if not checks["F6"]["pass"]:
            raise Stop("F6 input hash readback failed")
        # ---- F5
        checks["F5"] = run_tests(d) if not smoke else {"pass": True, "smoke": True}
        if not checks["F5"]["pass"]:
            raise Stop("F5 unit tests failed")
        LC, OV = L1.import_prosper()
        meta = L1.meta_cache_load(BIND / "meta_tiles.pkl")
        c0 = cpu_s()
        R = run_pairs(d, LC, OV, meta, only_refs=SMOKE_REFS if smoke else None)
        timing["pairs_cpu_s"] = cpu_s() - c0
        # ---- F1
        f1 = R["f1"]
        f1.to_csv(d / "f1_plans.csv.gz", index=False)
        ok_shape = bool(((f1.shape0 == f1.n_query_rows) & (f1.shape1 == f1.n_ref_rows) &
                         (f1.plans_csv_n_query == f1.shape0) & (f1.plans_csv_n_ref == f1.shape1) &
                         f1.max_i_ok & f1.max_j_ok).all())
        checks["F1"] = {"n_plans": int(len(f1)), "max_abs_diff_distance": float(f1.abs_diff_distance.max()),
                        "distance_pass": bool((f1.abs_diff_distance <= F1_TOL).all()),
                        "side_shape_rowcount_pass": ok_shape, "n_square_plans": int(f1.square.sum()),
                        "max_rowsum_dev": float(f1.max_rowsum_dev.max()), "max_colsum_dev": float(f1.max_colsum_dev.max()),
                        "max_cost_dev_query_i_reference_j": float(f1.max_cost_dev.max()),
                        "cost_side_pass": bool((f1.max_cost_dev <= F1_COST_TOL).all())}
        checks["F1"]["n_plans_expected"] = 988
        checks["F1"]["pass"] = bool(checks["F1"]["distance_pass"] and ok_shape and checks["F1"]["cost_side_pass"]
                                    and len(f1) == checks["F6"]["n_plans"] == 988)
        if not checks["F1"]["pass"] and not smoke:
            raise Stop("F1 failed")
        # ---- assemble per-pair tables with pair metadata
        p2 = pd.read_csv(S0 / "pairs_P2.csv", dtype=str)
        meta_cols = ["pair_id", "reference", "query", "arm", "ref_patient", "ref_group", "query_patient",
                     "query_group", "half", "ref_strict"]
        pairs = p2[meta_cols]
        D1 = pairs.merge(R["d1"], on=["reference", "query"], how="left")
        D2 = pairs.merge(R["d2"], on=["reference", "query"], how="left")
        D3 = pairs.merge(R["d3"], on=["reference", "query"], how="left")
        if smoke:
            p2 = p2[p2.reference.isin(SMOKE_REFS)]
            pairs = pairs[pairs.reference.isin(SMOKE_REFS)]
            D1, D2, D3 = (t[t.reference.isin(SMOKE_REFS)].reset_index(drop=True) for t in (D1, D2, D3))
        if not (len(D1) == len(D2) == len(D3) == len(p2) == (910 if not smoke else len(p2)) and D1.full_bin_S.notna().all()):
            raise Stop("pair table assembly mismatch")
        # ---- F2
        Lc = R["Lc"]
        x = D1.set_index("pair_id")
        Lx = Lc.loc[x.index]
        l1k = Lx.excess_bin / (1 - Lx.chance_bin)
        d_lit = (x.full_bin_kappa - l1k).abs()
        d_S = (x.full_bin_S - Lx.agreement_bin).abs()
        d_ch = (x.full_bin_chance_count - Lx.chance_bin).abs()
        d_conv = (x.full_bin_kappa_L1conv - l1k).abs()
        d_mass = (x.full_bin_annotated_mass - Lx.annotated_mass_bin).abs()
        def mx(s):
            s = s.astype(float)
            return float(s.max()) if s.notna().any() else None
        lit_nan_mismatch = int((x.full_bin_kappa.isna() != l1k.isna()).sum())
        checks["F2"] = {
            "n_pairs": int(len(x)),
            "literal": {"definition": "design-1 kappa (S - S0)/(1 - S0), S0 from annotated tile-count marginals "
                                      "(amendment 001 A1) vs L1 excess_bin/(1 - chance_bin)", "max_abs_diff": mx(d_lit),
                        "n_pairs_over_tol": int((d_lit > F2_TOL).sum()), "nan_pattern_mismatch": lit_nan_mismatch,
                        "pass": bool(lit_nan_mismatch == 0 and (d_lit.dropna() <= F2_TOL).all())},
            "components": {"S_vs_agreement_bin_max_abs_diff": mx(d_S),
                           "count_chance_vs_chance_bin_max_abs_diff": mx(d_ch),
                           "kappa_L1conv_vs_L1_max_abs_diff": mx(d_conv),
                           "annotated_mass_vs_annotated_mass_bin_max_abs_diff": mx(d_mass),
                           "mean_abs_S0_massmarg_minus_chance_count": float((x.full_bin_S0_massmarg - x.full_bin_chance_count).abs().mean()),
                           "kappa_massmarg_vs_L1_max_abs_diff (sensitivity, not gated)": mx((x.full_bin_kappa_massmarg - l1k).abs()),
                           "pass": bool((d_S <= F2_TOL).all() and (d_ch <= F2_TOL).all() and (d_conv.fillna(0) <= F2_TOL).all()
                                        and (d_mass <= F2_TOL).all() and (d_conv.isna() == l1k.isna()).all())}}
        checks["F2"]["pass"] = checks["F2"]["literal" if F2_GATE == "literal" else "components"]["pass"]
        if not checks["F2"]["pass"] and not smoke:
            raise Stop(f"F2 ({F2_GATE}) failed")
        # ---- F3
        f3 = R["f3"]
        checks["F3"] = {"n_rows": int(len(f3)), "n_P1_rows": int((f3.design == "P1").sum()),
                        "n_P2_rows": int((f3.design == "P2").sum()), "max_abs_diff": float(f3.abs_diff.max()),
                        "n_nan": int(f3.abs_diff.isna().sum()), "n_P2_pairs_also_P1": int((p2.also_P1 == "True").sum()),
                        "pass": bool(len(f3) > 0 and f3.abs_diff.notna().all() and (f3.abs_diff <= F3_TOL).all())}
        if not checks["F3"]["pass"]:
            raise Stop("F3 failed")
        # ---- probe (F4)
        c0 = cpu_s()
        P = run_probe(LC, meta, L1.bound_path(BIND, "pool"), limit_slides=25 if smoke else None)
        timing["probe_cpu_s"] = cpu_s() - c0
        checks["F4"] = P["f4"]
        if not checks["F4"]["pass"] and not smoke:
            raise Stop("F4 label-shuffle control failed")
        checks["all_pass"] = True
        jdump(d / "checks.json", checks)
        # ---- result files
        D1.to_csv(d / "per_pair_D1.csv.gz", index=False)
        D2.to_csv(d / "per_pair_D2.csv.gz", index=False)
        D3.to_csv(d / "per_pair_D3b.csv.gz", index=False)
        refinfo = pd.read_csv(S0 / "references.csv", dtype=str).rename(columns={"slide_id": "reference", "patient": "ref_patient"})
        if smoke:
            refinfo = refinfo[refinfo.reference.isin(SMOKE_REFS)]
        R2 = R["refs"].merge(refinfo[["reference", "ref_patient", "group", "half", "strict"]], on="reference", how="left")
        R2.to_csv(d / "per_reference_D2.csv.gz", index=False)
        crow = []
        for _, r_, q_, v_, lab_ in R["costs"]:
            crow.append(pd.DataFrame({"reference": r_, "query": q_, "ref_row": np.arange(len(v_)), "cost": v_,
                                      "ref_label": lab_}))
        pd.concat(crow, ignore_index=True).to_csv(d / "per_tile_cost_D2.csv.gz", index=False)
        # matrices (mean over pairs, per arm)
        mt = pd.DataFrame(R["mats"], columns=["reference", "query", "granularity", "kind", "ref_class", "query_class", "value"])
        mt = pairs[["reference", "query", "arm", "pair_id"]].merge(mt, on=["reference", "query"])
        mat = (mt.groupby(["granularity", "arm", "kind", "ref_class", "query_class"])
               .agg(value=("value", lambda s: float(np.nanmean(s)) if np.isfinite(s).any() else float("nan")),
                    n_pairs=("pair_id", "nunique")).reset_index())
        mat.to_csv(d / "matrices_D1.csv", index=False)
        # amendment 001 A4: per reference (mean over its pairs), then over references
        mr = (mt.groupby(["granularity", "arm", "kind", "ref_class", "query_class", "reference"])
              .value.agg(lambda s: float(np.nanmean(s)) if np.isfinite(s).any() else float("nan")).reset_index())
        mr = (mr.groupby(["granularity", "arm", "kind", "ref_class", "query_class"])
              .agg(value=("value", lambda s: float(np.nanmean(s)) if np.isfinite(s).any() else float("nan")),
                   n_references=("reference", "nunique")).reset_index())
        mr.to_csv(d / "matrices_D1_per_reference.csv", index=False)
        figs = fig_matrices(mat, d)
        # probe outputs
        jdump(d / "probe_D3a.json", {"script_sha256": checks["script_sha256"], **P["probe"],
                                     "F4": P["f4"], "slide_counts": P["counts"].to_dict(orient="records")})
        P["confusion"].to_csv(d / "confusion_D3a.csv", index=False)
        P["slides"].to_csv(d / "probe_D3a_mixed_slides.csv", index=False)
        # estimates
        refinfo2 = refinfo[["reference", "ref_patient", "half"]]
        E = build_estimates(pairs, D1, D2, D3, R["refs"], refinfo2, P["probe"])
        summ = {"script_sha256": checks["script_sha256"]}
        for arm in ("same_group", "other_group", "all"):
            sub = D1 if arm == "all" else D1[D1.arm == arm]
            for gran in ("bin", "5"):
                v = sub[f"full_{gran}_forced_mismatch"].astype(float)
                vc = sub[f"full_{gran}_forced_mismatch_massmarg"].astype(float)
                summ[f"D3c|{arm}|{gran}"] = {"n_pairs": int(len(sub)),
                                             "forced_mismatch_quantiles": {str(qq): float(np.nanquantile(v, qq)) for qq in (0, .1, .25, .5, .75, .9, 1)},
                                             "forced_mismatch_mean_pairs": float(np.nanmean(v)),
                                             "forced_mismatch_massmarg_quantiles": {str(qq): float(np.nanquantile(vc, qq)) for qq in (0, .1, .25, .5, .75, .9, 1)},
                                             "per_reference_estimate_pooled": E["pooled"][f"D3|{arm}|full_{gran}_forced_mismatch"]}
            s3 = D3 if arm == "all" else D3[D3.arm == arm]
            for gran in ("bin", "5"):
                summ[f"D3b|{arm}|{gran}"] = {"n_pairs": int(len(s3)),
                                             "n_nn_no_headroom": int(s3[f"nn_{gran}_no_headroom"].astype(bool).sum()),
                                             "n_diff_defined": int(s3[f"diff_{gran}_NA_NN_minus_OT"].notna().sum()),
                                             **{k: E["pooled"][f"D3|{arm}|{c}"] for k, c in
                                                (("NA_NN", f"nn_{gran}_NA"), ("kappa_NN", f"nn_{gran}_kappa"),
                                                 ("S_NN", f"nn_{gran}_S"), ("diff_NA_NN_minus_OT", f"diff_{gran}_NA_NN_minus_OT"))}}
        summ["D3a"] = {k: v for k, v in P["probe"].items() if "|" in k}
        summ["switch"] = {k: v for k, v in E["wording"]["pooled"].items() if k.startswith("D3_switch")}
        jdump(d / "summary_D3.json", summ)
        jdump(d / "report_estimates.json", {"created": now(), "script_sha256": checks["script_sha256"],
                                            "B": L1.N_BOOT, "seed": L1.SEED, "generator": "numpy PCG64 (L1.make_patient_draws)",
                                            "unit": "pair statistics averaged per reference, then over references; "
                                                    "whole-patient bootstrap over reference patients, percentile 95 %",
                                            "populations": {"pooled": "all 46 P2 references (46 patients)",
                                                            "development": "L1 development half (consistency check)",
                                                            "report": "L1 report half (consistency check)"},
                                            **E})
        # overlays
        refs_sorted = L1.numeric_sorted(R["deltas"])
        ov = []
        rr_map = {row.reference: row._asdict() for row in R2.itertuples(index=False)}
        for r in refs_sorted[:2]:
            ov.append(fig_overlay(r, R["deltas"][r], rr_map[r], LC, meta, d))
        timing["total_cpu_s"] = cpu_s() - t_cpu
        timing["wall_s"] = time.time() - t_wall
        L1.finish(d, "l2", {"contract_sha256": CONTRACT_SHA, "checks_pass": True, "timing": timing,
                            "figures": figs + ov, "F2_gate": F2_GATE})
        print(f"[l2] COMPLETE {d}", flush=True)
        return 0
    except Stop as e:
        timing["total_cpu_s"] = cpu_s() - t_cpu
        checks["all_pass"] = False
        checks["stop_reason"] = str(e)
        jdump(d / "checks.json", checks)
        L1.fail(d, "l2", f"STOP: {e}", {"timing": timing, "checks": checks})
        return 3
    except Exception as e:
        timing["total_cpu_s"] = cpu_s() - t_cpu
        jdump(d / "checks.json", checks)
        L1.fail(d, "l2", repr(e), {"traceback": traceback.format_exc(), "timing": timing})
        return 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=["run"])
    p.add_argument("--smoke", action="store_true", help="3 references + 25 probe slides into the job tmp dir (code check only)")
    a = p.parse_args(argv)
    return {"run": stage_run}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
