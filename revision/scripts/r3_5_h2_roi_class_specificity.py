#!/usr/bin/env python3
"""R3.5-H2: does the cosine ROI heat map follow the tissue class of the ROI (non-tumour ROI vs tumour ROI, same slide).

Binding specification: $PROSPER_OUT/r3_5_localisation_maps_20261017/contract_008/SPECIFICATION.md
(sha256 0be53f656c952587c3540cc2470dea4161062b5e6f7232607a4ca50639201296).  Status: post hoc.

Slides: the 46 P2 reference slides (one per patient).  For a source slide s and class c in {3 necrosis, 4 nonTumor}
an eight-tile ROI of class c (and a tumour ROI, class 1, from the same s) is mapped onto every other patient's slide
r with L1.roi_map(r, q, "cosine", "mean") and scored by AUROC_c = LC._auroc over r's annotated patches (labels 1-5),
y = (patch_label == c).  M = AUROC_c(class-c ROI), C = AUROC_c(tumour ROI), D = M - C; mirror TM = AUROC_1(tumour
ROI), TC = AUROC_1(class-c ROI), TD = TM - TC.  Aggregated per target (mean over sources, bootstrap over target
patients) and per source (mean over targets, bootstrap over source patients); L1.make_patient_draws B 5000 seed 123.

Gate order (specification 6): bind -> G5 (hashes vs specification table) -> G4 (pytest) -> G6 (patient distinctness) -> G2
(label counts) -> G1 (positive control) -> per-pair pass -> G3 (determinism, second pass) -> estimates -> G5 final
readback of every bound file.  Any FAIL writes checks.json + FAILED.json and stops before results.json.

Implementation choices (also in [internal record]):
 1. Per-pair AUROCs are stored whenever both classes have >= 1 annotated patch on the target; target eligibility
    (>= thr patches of the class and >= thr other annotated patches; thr 20, sensitivities 10 / 40) is applied at
    aggregation.  M, C, D use eligibility for class c; the mirror TM, TC, TD uses eligibility for class 1 (tumour)
    on the same (source, target) pairs of class c.
 2. ROI sensitivities (all tiles; draws with seeds 124 / 125 / 126; L1 distance) are applied to both ROIs of a pair
    (class c and tumour), so D keeps its paired meaning; they are computed on targets eligible for c at thr 20.
 3. Class matrix: ROI class a in {1, 3, 4}, target class b in {1, 3, 4}; sources of a = source slides of a (a = 3 / 4)
    or of either non-tumour class (a = 1); targets eligible for b at thr 20; target != source.
 4. Rule sentence filled with the target-patient interval (the primary aggregation) and n target patients; the
    source-patient interval is recorded beside it.  The case "D both lo > 0, M not both lo > 0.5 and no M interval
    spans 0.5 (an M interval wholly below 0.5)" is not covered by the specification text and is reported as not_covered.
 5. Same-slide reference value: own class-c ROI on the source slide itself, ROI tiles' own patches left in;
    aggregated over sources that are themselves eligible targets for c (thr 20), bootstrap over source patients.
 6. G3 second run = an in-process second pass that recomputes every ROI draw and every base (cosine, seed-123)
    map score from scratch (no cache shared) and every estimate from the second-pass table; sensitivity map scores
    are not recomputed a second time (time budget).  Seed 124 for the bootstrap: same points, different intervals.
"""
import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_SHA = "0be53f656c952587c3540cc2470dea4161062b5e6f7232607a4ca50639201296"
STUDY = "r3_5_localisation_maps_20261017"
L1_SCRIPT_SHA = "b316497077dc03dd344918aeb8f47b441b85efc2ed45b71da65981d49650f47b"
TIE_SCRIPT_SHA = "e68d5a29508983fdcfd36d7047904ba076af6ba25a21a92d33e2a9c39c1a8c81"
INPUTS = {  # role -> (path relative to $PROSPER_OUT, specification sha256 or None = bind at run start)
    "pairs_P2": (f"{STUDY}/s0_pairs/run_001/pairs_P2.csv",
                 "9b90ad629ee9e9b7bf04f43e913be5efa0000e9cc73180215260953ae187ffbe"),
    "s0_pairs_complete": (f"{STUDY}/s0_pairs/run_001/COMPLETE.json", None),
    "design_counts": (f"{STUDY}/contract_008/design_label_only_mosaic_tile_counts.csv",
                      "3db1a6a52af7f4a73f35e525f867b5dc18ac1734e863bf1a34b954da50fab186"),
    "per_pair_A": (f"{STUDY}/s2/run_003/per_pair_A.csv.gz",
                   "442b871ba2e4ec564aa4f0ec889ae7017b92f048efdff066561181f2ae4dcc61"),
    "s2_complete": (f"{STUDY}/s2/run_003/COMPLETE.json", None),
    "cosine_c4_44_result": ("r3_5_cosine_heatmap_c4_44_20261112/result.json",
                            "76f18291de37cfef41b0576365582e238c77fbf0b1ff1a0ce4240188ca0c2d16"),
    "c4_44_references": ("r2_5_c4_whole_patient_ci_20261008/per_reference_variant_A.csv", None),
}
TUMOUR, NECROSIS, NONTUMOR = 1, 3, 4
CLASSES = (NECROSIS, NONTUMOR)
CLASS_NAME = {1: "tumour", 3: "necrosis", 4: "nonTumor"}
ANNOTATED = (1, 2, 3, 4, 5)
ROI_TILES = 8
ROI_SEED = 123
ALT_SEEDS = (124, 125, 126)
THR = 20
THR_SENS = (10, 40)
MIN_N_RULE = 12
TOL = 1e-9
STEP_LIMIT_S = 30 * 60
METRICS = ("M", "C", "D", "TM", "TC", "TD")
MIRROR = ("TM", "TC", "TD")
LEAD_PREDICTIONS = {"necrosis": {"M": "0.75-0.92", "C": "0.25-0.45"},
                    "nonTumor": {"M": "0.55-0.75", "C": "0.35-0.50"},
                    "mirror": {"TM": "0.60-0.72", "TC": "0.25-0.45"},
                    "note": "stated in contract_008 section 4 before any non-tumour-ROI heat map was computed"}
CLASS_TEXT = {3: ("necrosis", "necrosis"), 4: ("non-tumor", "non-tumor tissue")}
RULE_TEXT = {
    "branch_1_follows": ("A [cls] ROI ranked annotated [tissue] on other patients' slides above other annotated tissue "
                         "(AUROC x, 95% CI lo–hi; n reference patients), whereas a tumor ROI from the same slide did "
                         "not (y; paired difference d, 95% CI lo–hi)."),
    "branch_2_shifted": ("shifted towards the selected tissue (paired difference d, CI) without ranking it above other "
                         "annotated tissue (AUROC x, CI)"),
    "branch_3_not_distinguished": "the heat map did not distinguish a [cls] ROI from a tumor ROI",
    "descriptive_only": "fewer than 12 eligible targets or 12 sources: reported descriptively only, no rule sentence",
    "not_covered": ("case not covered by the contract 4 rule text (D lower bounds > 0 in both directions, M not above "
                    "0.5 in both and no M interval spans 0.5); no rule sentence"),
}
RULE_CLAUSE = {
    "branch_1_follows": "The heat map may then be described as following the selected tissue class for that class.",
    "branch_2_shifted": "No \"follows the selected tissue\" description for that class.",
    "branch_3_not_distinguished": ("Reported verbatim in the supplement and the letter; the manuscript must not describe "
                                   "the heat map as following the selected tissue class."),
    "descriptive_only": "",
    "not_covered": "",
}


# ---------------------------------------------------------------- utilities
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def binding(path, role, expected=None) -> dict:
    p = Path(path)
    st = p.stat()
    rec = {"role": role, "path": str(p), "resolved_path": str(p.resolve()), "size": st.st_size,
           "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(p)}
    if expected is not None:
        rec["contract_sha256"] = expected
        rec["matches_contract"] = rec["sha256"] == expected
    return rec


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None
    return o


def jdump(path, obj):
    Path(path).write_text(json.dumps(_clean(obj), indent=1, default=_json_default))


class Stop(Exception):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def out_root() -> Path:
    out = os.environ.get("PROSPER_OUT")
    if not out:
        raise Stop("PROSPER_OUT is not exported; run `export PROSPER_OUT=\"$PWD/prosper_revision_output\"`")
    return Path(out)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_l1():
    path = repo_root() / "scripts" / "r3_5_l1_maps.py"
    return load_module(path, "r3_5_l1_maps"), path


def numeric_sorted(ids):
    return sorted((str(s) for s in ids), key=lambda s: (len(s), s))


def snapshot_script(root: Path) -> dict:
    me = Path(__file__).resolve()
    sha = sha256_file(me)
    snap = root / "script_snapshots"
    snap.mkdir(parents=True, exist_ok=True)
    existing = sorted(snap.glob(f"r3_5_h2_roi_class_specificity_v*_{sha[:8]}.py"))
    if existing:
        return {"snapshot": str(existing[0]), "sha256": sha, "created_now": False}
    n = len(list(snap.glob("r3_5_h2_roi_class_specificity_v*_*.py"))) + 1
    p = snap / f"r3_5_h2_roi_class_specificity_v{n}_{sha[:8]}.py"
    shutil.copyfile(me, p)
    return {"snapshot": str(p), "sha256": sha, "created_now": True}


# ---------------------------------------------------------------- pure kernels (unit-tested, G4)
def roi_rows(tile_label, c) -> np.ndarray:
    """Ascending Slide.X row indices with tile_label == c."""
    return np.flatnonzero(np.asarray(tile_label) == c).astype(np.int64)


def draw_roi(tile_label, c: int, sid, seed: int = ROI_SEED, n: int = ROI_TILES):
    """Specification 3: np.sort(default_rng([seed, c, int(sid)]).choice(rows_c, size=8, replace=False)); None if < 8."""
    rows_c = roi_rows(tile_label, c)
    if len(rows_c) < n:
        return None
    return np.sort(np.random.default_rng([int(seed), int(c), int(sid)]).choice(rows_c, size=n, replace=False))


def roi_vector(X, idx) -> np.ndarray:
    return np.asarray(X)[np.asarray(idx, dtype=np.int64)].mean(axis=0)


def is_source(tile_label, c: int, n: int = ROI_TILES) -> bool:
    """Source for class c: >= 8 tiles of class c and >= 8 tumour tiles."""
    tl = np.asarray(tile_label)
    return bool((tl == c).sum() >= n and (tl == TUMOUR).sum() >= n)


def class_counts(patch_label, c: int) -> tuple:
    """(n patches of class c, n other annotated patches (labels 1-5, not c))."""
    pl = np.asarray(patch_label)
    ann = np.isin(pl, ANNOTATED)
    n_c = int((pl == c).sum())
    return n_c, int(ann.sum()) - n_c


def target_eligible(patch_label, c: int, thr: int = THR) -> bool:
    n_c, n_o = class_counts(patch_label, c)
    return bool(n_c >= thr and n_o >= thr)


def class_labels(patch_label, c: int) -> tuple:
    """(annotated mask over the patch grid, y = (patch_label == c) on annotated patches as int)."""
    pl = np.asarray(patch_label)
    ann = np.isin(pl, ANNOTATED)
    return ann, (pl[ann] == c).astype(int)


def auroc_class(h, patch_label, c: int, LC, min_pos: int = 1, min_neg: int = 1) -> float:
    """AUROC_c = LC._auroc(h[annotated], y[annotated]); NaN if < min_pos class-c or < min_neg other annotated."""
    ann, y = class_labels(patch_label, c)
    if y.sum() < min_pos or (len(y) - y.sum()) < min_neg:
        return float("nan")
    return LC._auroc(np.asarray(h)[ann], y)


def check_patients_distinct(slide_patient: dict, pairs=None) -> dict:
    """G6 / specification 3: every slide a different patient and no (source, target) pair sharing a patient."""
    pats = list(slide_patient.values())
    dup = sorted({p for p in pats if pats.count(p) > 1})
    shared = []
    if pairs is not None:
        shared = [(s, r) for s, r in pairs if slide_patient[s] == slide_patient[r]]
    rec = {"n_slides": len(slide_patient), "n_patients": len(set(pats)), "duplicated_patients": dup,
           "n_pairs_checked": 0 if pairs is None else len(pairs), "n_pairs_sharing_patient": len(shared),
           "pairs_sharing_patient": shared[:20]}
    rec["pass"] = bool(not dup and not shared)
    return rec


def assert_patients_distinct(slide_patient: dict, pairs=None):
    rec = check_patients_distinct(slide_patient, pairs)
    if not rec["pass"]:
        raise AssertionError(f"patient distinctness violated: {rec}")
    return rec


def aggregate(pp: pd.DataFrame, col: str, mask, by: str) -> pd.DataFrame:
    """Per-unit mean of `col` over rows in `mask` with a finite value; by = 'target' or 'source'.
    Returns unit, patient, value, n_rows."""
    x = pp[np.asarray(mask, bool) & np.isfinite(pp[col].to_numpy(float))]
    g = x.groupby(by)
    out = pd.DataFrame({"value": g[col].mean(), "n_rows": g.size(), "patient": g[f"{by}_patient"].first()})
    out = out.reset_index().rename(columns={by: "unit"})
    order = sorted(range(len(out)), key=lambda i: (len(str(out.unit.iat[i])), str(out.unit.iat[i])))
    return out.iloc[order].reset_index(drop=True)


def boot(L1, values, patients, n_rows, B: int, seed: int) -> dict:
    v = np.asarray(values, float)
    pats_in = [str(p) for p in patients]
    pats, mult = L1.make_patient_draws(pats_in, B, seed)
    reps = L1.replicate_means(v, pats_in, pats, mult)
    out = L1.summarise_reps(float(v.mean()) if len(v) else float("nan"), reps, len(v))
    out.update({"n_patients": int(len(set(pats_in))), "n_pairs": int(np.sum(n_rows)), "B": int(B), "seed": int(seed)})
    return out


def estimate_both(L1, pp, col, mask, B, seed) -> dict:
    """Both aggregation directions of one per-pair quantity on one row set."""
    res = {}
    for by, key in (("target", "by_target"), ("source", "by_source")):
        a = aggregate(pp, col, mask, by)
        res[key] = boot(L1, a.value, a.patient, a.n_rows, B, seed) if len(a) else \
            {"point": float("nan"), "ci95_lo": None, "ci95_hi": None, "n_references": 0, "n_patients": 0,
             "n_pairs": 0, "B": B, "seed": seed}
    m = np.asarray(mask, bool) & np.isfinite(pp[col].to_numpy(float))
    res["n_targets"] = int(pp.loc[m, "target"].nunique())
    res["n_sources"] = int(pp.loc[m, "source"].nunique())
    res["n_pairs"] = int(m.sum())
    return res


def _lo(e):
    v = e.get("ci95_lo")
    return float("nan") if v is None else float(v)


def _hi(e):
    v = e.get("ci95_hi")
    return float("nan") if v is None else float(v)


def rule_branch(M: dict, D: dict, n_targets: int, n_sources: int, min_n: int = MIN_N_RULE) -> str:
    """Specification 4 rule on both intervals (by_target, by_source) of M and D; strict inequalities."""
    if n_targets < min_n or n_sources < min_n:
        return "descriptive_only"
    dirs = ("by_target", "by_source")
    d_ok = all(_lo(D[k]) > 0 for k in dirs)
    m_ok = all(_lo(M[k]) > 0.5 for k in dirs)
    if not d_ok:
        return "branch_3_not_distinguished"
    if m_ok:
        return "branch_1_follows"
    if any(_lo(M[k]) <= 0.5 <= _hi(M[k]) for k in dirs):
        return "branch_2_shifted"
    return "not_covered"


def fill_rule(branch: str, c: int, M: dict, C: dict, D: dict) -> str:
    cls, tissue = CLASS_TEXT[c]
    m, cc, d = M["by_target"], C["by_target"], D["by_target"]
    t = RULE_TEXT[branch].replace("[cls]", cls).replace("[tissue]", tissue)
    if branch == "branch_1_follows":
        t = (t.replace("AUROC x, 95% CI lo–hi; n reference patients",
                       f"AUROC {m['point']:.2f}, 95% CI {_lo(m):.2f}–{_hi(m):.2f}; {m['n_patients']} reference patients")
             .replace("(y; paired difference d, 95% CI lo–hi)",
                      f"({cc['point']:.2f}; paired difference {d['point']:.2f}, 95% CI {_lo(d):.2f}–{_hi(d):.2f})"))
    elif branch == "branch_2_shifted":
        t = (t.replace("paired difference d, CI", f"paired difference {d['point']:.2f}, 95% CI {_lo(d):.2f}–{_hi(d):.2f}")
             .replace("AUROC x, CI", f"AUROC {m['point']:.2f}, 95% CI {_lo(m):.2f}–{_hi(m):.2f}"))
    return t


def general_statement(branches: dict) -> dict:
    passed = [CLASS_NAME[c] for c in CLASSES if branches[c] == "branch_1_follows"]
    if len(passed) == len(CLASSES):
        return {"general_non_tumor_statement_licensed": True, "classes_passing_branch_1": passed}
    return {"general_non_tumor_statement_licensed": False, "classes_passing_branch_1": passed,
            "note": "the statement names only the class(es) that passed branch 1" if passed else "no class passed branch 1"}


# ---------------------------------------------------------------- slide layer
class SlideSet:
    def __init__(self, L1, LC, meta, bind_files: dict):
        self.L1, self.LC, self.meta, self.bind = L1, LC, meta, bind_files
        self.cache = {}

    def _bind(self, s):
        if s not in self.bind:
            f = self.L1.FEATURES / f"{s}.pkl"
            lab = self.L1.LABELS / f"{s}.csv"
            self.bind[s] = {"features": binding(f, f"features:{s}"), "labels": binding(lab, f"labels:{s}")}

    def get(self, s):
        if s not in self.cache:
            self._bind(s)
            self.cache[s] = self.LC.Slide(s, self.L1.FEATURES, self.meta[s], self.L1.LABELS)
        return self.cache[s]

    def feature_row_counts(self, s) -> dict:
        self._bind(s)
        _, keys = self.L1.load_features(s)
        lab = self.L1.load_labels(s)
        tl = np.array([lab.get(k, 0) for k in keys])
        return {"n_rows": len(keys), **{f"t{c}": int((tl == c).sum()) for c in (1, 2, 3, 4, 5)}}


def draw_all(slides: dict, sids, seeds=(ROI_SEED,) + ALT_SEEDS) -> dict:
    """{(seed, class, slide): sorted Slide.X row indices or None}."""
    out = {}
    for s in sids:
        tl = slides[s].tile_label
        for seed in seeds:
            for c in (TUMOUR, NECROSIS, NONTUMOR):
                out[(seed, c, s)] = draw_roi(tl, c, s, seed)
    return out


def roi_q(slide, draws, variant: str, c: int, s: str):
    if variant == "all":
        rows = roi_rows(slide.tile_label, c)
        return slide.X[rows].mean(axis=0)
    seed = ROI_SEED if variant == "base" else int(variant.replace("seed", ""))
    return roi_vector(slide.X, draws[(seed, c, s)])


def score_map(L1, LC, target, q, distance) -> tuple:
    h = L1.roi_map(target, q, distance, "mean")
    return tuple(auroc_class(h, target.patch_label, b, LC) for b in (TUMOUR, NECROSIS, NONTUMOR))


B_INDEX = {TUMOUR: 0, NECROSIS: 1, NONTUMOR: 2}


def base_pass(L1, LC, slides, refs, src, draws, deadline) -> tuple:
    """Every base (seed-123 eight-tile, cosine) ROI of every union source on every other reference slide, scored for
    target classes 1, 3, 4; plus the same-slide value.  Returns ({(roi_class, s, r): (a1, a3, a4)}, same_slide)."""
    union = [s for s in refs if s in src[NECROSIS] or s in src[NONTUMOR]]
    scores = {}
    for r in refs:
        if time.time() > deadline:
            raise Stop("base map pass exceeded the 30-minute step limit")
        tgt = slides[r]
        for s in union:
            if s == r:
                continue
            for a in (TUMOUR, NECROSIS, NONTUMOR):
                if a != TUMOUR and s not in src[a]:
                    continue
                q = roi_q(slides[s], draws, "base", a, s)
                scores[(a, s, r)] = score_map(L1, LC, tgt, q, "cosine")
    same = {}
    for c in CLASSES:
        for s in src[c]:
            q = roi_q(slides[s], draws, "base", c, s)
            h = L1.roi_map(slides[s], q, "cosine", "mean")
            same[(c, s)] = auroc_class(h, slides[s].patch_label, c, LC)
    return scores, same


def sensitivity_pass(L1, LC, slides, refs, src, tgt_ok, draws, deadline) -> dict:
    """{(variant, distance, roi_class, s, r): (a1, a3, a4)} for variants all / seed124-126 (cosine) and base (l1),
    both ROIs of each class-c pair, targets eligible for c at thr 20."""
    variants = [("all", "cosine")] + [(f"seed{k}", "cosine") for k in ALT_SEEDS] + [("base", "l1")]
    out = {}
    for var, dist in variants:
        for c in CLASSES:
            for s in src[c]:
                if time.time() > deadline:
                    raise Stop("sensitivity map pass exceeded the 30-minute step limit")
                for r in refs:
                    if r == s or not tgt_ok[(c, THR)][r]:
                        continue
                    for a in (c, TUMOUR):
                        key = (var, dist, a, s, r)
                        if key not in out:
                            out[key] = score_map(L1, LC, slides[r], roi_q(slides[s], draws, var, a, s), dist)
    return out


SENS_VARIANTS = {"all_tiles": ("all", "cosine"), "seed124": ("seed124", "cosine"), "seed125": ("seed125", "cosine"),
                 "seed126": ("seed126", "cosine"), "l1_distance": ("base", "l1")}


def per_pair_table(refs, info, src, counts, scores, sens) -> pd.DataFrame:
    rows = []
    for c in CLASSES:
        bi = B_INDEX[c]
        for s in src[c]:
            for r in refs:
                if r == s:
                    continue
                kc, k1 = scores[(c, s, r)], scores[(TUMOUR, s, r)]
                rec = {"source": s, "target": r, "class": c, "class_name": CLASS_NAME[c],
                       "source_patient": info[s]["patient"], "target_patient": info[r]["patient"],
                       "source_group": info[s]["group"], "target_group": info[r]["group"],
                       "same_group": info[s]["group"] == info[r]["group"],
                       "M": kc[bi], "C": k1[bi], "TM": k1[0], "TC": kc[0],
                       "target_n_class": counts[(r, c)][0], "target_n_other_annotated": counts[(r, c)][1],
                       "target_n_tumour": counts[(r, TUMOUR)][0], "target_n_nontumour_annotated": counts[(r, TUMOUR)][1]}
                for thr in (THR,) + THR_SENS:
                    rec[f"target_eligible_c_thr{thr}"] = counts[(r, c)][0] >= thr and counts[(r, c)][1] >= thr
                rec["target_eligible_tumour_thr20"] = counts[(r, TUMOUR)][0] >= THR and counts[(r, TUMOUR)][1] >= THR
                for name, (var, dist) in SENS_VARIANTS.items():
                    a, b = sens.get((var, dist, c, s, r)), sens.get((var, dist, TUMOUR, s, r))
                    rec[f"M_{name}"] = a[bi] if a is not None else float("nan")
                    rec[f"C_{name}"] = b[bi] if b is not None else float("nan")
                rows.append(rec)
    pp = pd.DataFrame(rows)
    pp["D"] = pp.M - pp.C
    pp["TD"] = pp.TM - pp.TC
    for name in SENS_VARIANTS:
        pp[f"D_{name}"] = pp[f"M_{name}"] - pp[f"C_{name}"]
    return pp


def metric_mask(pp, c, metric, thr=THR):
    base = (pp["class"] == c).to_numpy()
    if metric in MIRROR:
        return base & pp["target_eligible_tumour_thr20"].to_numpy(bool)
    return base & pp[f"target_eligible_c_thr{thr}"].to_numpy(bool)


def class_matrix_long(refs, info, src, counts, scores) -> pd.DataFrame:
    rows = []
    for a in (TUMOUR, NECROSIS, NONTUMOR):
        sources = [s for s in refs if (s in src[NECROSIS] or s in src[NONTUMOR])] if a == TUMOUR else src[a]
        for b in (TUMOUR, NECROSIS, NONTUMOR):
            for s in sources:
                for r in refs:
                    if r == s or not (counts[(r, b)][0] >= THR and counts[(r, b)][1] >= THR):
                        continue
                    rows.append({"roi_class": a, "target_class": b, "source": s, "target": r,
                                 "source_patient": info[s]["patient"], "target_patient": info[r]["patient"],
                                 "value": scores[(a, s, r)][B_INDEX[b]]})
    return pd.DataFrame(rows)


def build_estimates(L1, pp, cm_long, same, refs, info, src, counts, B, seed) -> dict:
    res = {"primary": {}, "secondary": {}}
    branches = {}
    for c in CLASSES:
        e = {m: estimate_both(L1, pp, m, metric_mask(pp, c, m), B, seed) for m in METRICS}
        nt, ns, npairs = e["M"]["n_targets"], e["M"]["n_sources"], e["M"]["n_pairs"]
        br = rule_branch(e["M"], e["D"], nt, ns)
        branches[c] = br
        res["primary"][CLASS_NAME[c]] = {
            "class": c, "n_targets": nt, "n_sources": ns, "n_pairs": npairs,
            "n_targets_mirror": e["TM"]["n_targets"], "n_sources_mirror": e["TM"]["n_sources"],
            "n_pairs_mirror": e["TM"]["n_pairs"],
            "estimates": e, "rule_branch": br, "rule_outcome": RULE_TEXT[br],
            "rule_outcome_filled": fill_rule(br, c, e["M"], e["C"], e["D"]),
            "rule_clause": RULE_CLAUSE[br],
            "rule_inputs": {k: {d: {"point": e[k][d]["point"], "ci95_lo": e[k][d]["ci95_lo"], "ci95_hi": e[k][d]["ci95_hi"]}
                                for d in ("by_target", "by_source")} for k in ("M", "D")},
            "rule": ("n_targets < 12 or n_sources < 12 -> descriptive_only; any D lo <= 0 -> branch_3_not_distinguished; "
                     "both M lo > 0.5 and both D lo > 0 -> branch_1_follows; both D lo > 0 and an M interval spans 0.5 "
                     "-> branch_2_shifted; else not_covered (both = by_target and by_source intervals)"),
            "filled_sentence_interval": "by_target (n = target patients); by_source interval in estimates",
        }
    res["primary"]["general_statement"] = general_statement(branches)
    res["primary"]["lead_predictions"] = LEAD_PREDICTIONS
    # secondary 1: mirror (also inside primary estimates)
    res["secondary"]["1_tumour_target_mirror"] = {
        CLASS_NAME[c]: {m: res["primary"][CLASS_NAME[c]]["estimates"][m] for m in MIRROR} for c in CLASSES}
    # secondary 2: same-group vs other-group
    sg = {}
    for c in CLASSES:
        sg[CLASS_NAME[c]] = {}
        for arm, flag in (("same_group", True), ("other_group", False)):
            sub = pp["same_group"].to_numpy(bool) == flag
            sg[CLASS_NAME[c]][arm] = {m: estimate_both(L1, pp, m, metric_mask(pp, c, m) & sub, B, seed)
                                      for m in ("M", "C", "D")}
    res["secondary"]["2_same_vs_other_group"] = sg
    # secondary 3: class matrix
    cmat = {}
    for (a, b), g in cm_long.groupby(["roi_class", "target_class"]):
        g = g.reset_index(drop=True)
        cmat[f"roi_{CLASS_NAME[a]}__target_{CLASS_NAME[b]}"] = {
            "roi_class": int(a), "target_class": int(b), **estimate_both(L1, g, "value", np.ones(len(g), bool), B, seed)}
    res["secondary"]["3_class_matrix"] = cmat
    # secondary 4: sensitivities of M and D
    sens = {}
    for c in CLASSES:
        s = {}
        for name in SENS_VARIANTS:
            s[name] = {m: estimate_both(L1, pp, f"{m}_{name}", metric_mask(pp, c, "M"), B, seed) for m in ("M", "D")}
        for thr in THR_SENS:
            s[f"target_eligibility_{thr}"] = {m: estimate_both(L1, pp, m, metric_mask(pp, c, m, thr), B, seed)
                                              for m in ("M", "D")}
        sens[CLASS_NAME[c]] = s
    res["secondary"]["4_sensitivities"] = sens
    # secondary 5: same-slide reference value
    ss = {}
    for c in CLASSES:
        rows = [{"source": s, "source_patient": info[s]["patient"], "value": same[(c, s)],
                 "eligible_as_target": counts[(s, c)][0] >= THR and counts[(s, c)][1] >= THR} for s in src[c]]
        df = pd.DataFrame(rows)
        e = df[df.eligible_as_target & np.isfinite(df.value.astype(float))]
        ss[CLASS_NAME[c]] = {"label": "same-slide (ROI tiles' own patches left in); descriptive, not an endpoint",
                             "by_source": boot(L1, e.value, e.source_patient, np.ones(len(e)), B, seed),
                             "n_sources_with_value": int(np.isfinite(df.value.astype(float)).sum()),
                             "n_sources_eligible_as_target": int(len(e)),
                             "mean_all_sources_with_value": float(np.nanmean(df.value.astype(float)))
                             if np.isfinite(df.value.astype(float)).any() else float("nan")}
    res["secondary"]["5_same_slide_reference"] = ss
    return res


# ---------------------------------------------------------------- fidelity checks
def check_G4(repo: Path, out_dir: Path) -> dict:
    test = repo / "tests" / "test_r3_5_h2_roi_class_specificity.py"
    cmd = [sys.executable, "-m", "pytest", str(test.relative_to(repo)), "-q"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    (out_dir / "pytest_output.txt").write_text(f"$ {' '.join(cmd)}\n(cwd {repo}, {now()})\n\n{out}")
    summary = [ln for ln in proc.stdout.strip().splitlines() if "passed" in ln or "failed" in ln or "error" in ln]
    return {"command": " ".join(cmd), "cwd": str(repo), "returncode": proc.returncode, "wall_s": time.time() - t0,
            "summary": summary[-1] if summary else proc.stdout.strip()[-200:], "test_file_sha256": sha256_file(test),
            "pass": proc.returncode == 0}


def check_G1(L1, LC, slides: SlideSet, p2_all, stored, cres, c4refs, B, seed) -> dict:
    refs = list(c4refs.reference)
    p2 = p2_all[p2_all.reference.isin(set(refs))].reset_index(drop=True)
    st = stored[(stored.design == "P2") & (stored.setting == "A|cosine|mean") & stored.reference.isin(set(refs))]
    st = st.set_index("pair_id")
    rec = {"n_references": len(refs), "n_pairs": int(len(p2)), "n_stored": int(len(st)),
           "stored_pair_ids_unique": bool(st.index.is_unique)}
    vals = []
    for r in numeric_sorted(set(p2.reference)):
        ref = slides.get(r)
        for row in p2[p2.reference == r].itertuples(index=False):
            q = slides.get(row.query).X[L1.parse_idx(row.query_tiles)].mean(axis=0)
            h = L1.roi_map(ref, q, "cosine", "mean")
            vals.append({"pair_id": row.pair_id, "reference": r, "arm": row.arm, "ref_patient": row.ref_patient,
                         "auroc_1": auroc_class(h, ref.patch_label, TUMOUR, LC, 1, 1)})
    df = pd.DataFrame(vals)
    stored_v = st.loc[df.pair_id, "auroc_A"].to_numpy(float)
    diff = np.abs(df.auroc_1.to_numpy(float) - stored_v)
    rec["max_abs_diff_per_pair"] = float(np.max(diff))
    rec["n_pairs_over_tol"] = int((~(diff <= TOL)).sum())
    rec["n_nan_rebuilt"] = int(np.isnan(df.auroc_1).sum())
    g = df.groupby(["arm", "reference"]).auroc_1.mean()
    same, other = g.loc["same_group"].reindex(refs).to_numpy(), g.loc["other_group"].reindex(refs).to_numpy()
    agg = {"same": float(np.mean(same)), "other": float(np.mean(other)), "diff": float(np.mean(same - other))}
    rec["aggregates_rebuilt"] = agg
    rec["aggregates_stored"] = {k: cres[f"cosine_auroc_{k}"]["point"] for k in agg}
    rec["aggregate_abs_diff"] = {k: abs(agg[k] - rec["aggregates_stored"][k]) for k in agg}
    # information only (not gated): intervals through L1.make_patient_draws on the 44 reference patients
    pats = list(c4refs.ref_patient.astype(str))
    info = {}
    for k, v in (("same", same), ("other", other), ("diff", same - other)):
        b = boot(L1, v, pats, np.ones(len(v)), B, seed)
        info[k] = {"ci95_lo": b["ci95_lo"], "ci95_hi": b["ci95_hi"],
                   "stored_ci95_lo": cres[f"cosine_auroc_{k}"]["ci95_lo"],
                   "stored_ci95_hi": cres[f"cosine_auroc_{k}"]["ci95_hi"]}
    rec["intervals_info_not_gated"] = info
    rec["eligibility"] = "stored eligibility: >= 1 tumour and >= 1 other annotated patch (min_pos = min_neg = 1)"
    rec["c4_44_all_eligible_thr20"] = bool(all(target_eligible(slides.get(r).patch_label, TUMOUR, THR) for r in refs))
    rec["tolerance"] = TOL
    rec["pass"] = bool(len(p2) == 880 and len(df) == 880 and rec["stored_pair_ids_unique"]
                       and rec["n_pairs_over_tol"] == 0 and rec["n_nan_rebuilt"] == 0
                       and all(v <= TOL for v in rec["aggregate_abs_diff"].values()))
    return rec


def check_G2(slides: SlideSet, design: pd.DataFrame, refs) -> tuple:
    rows = []
    for d in design.itertuples(index=False):
        s = str(d.slide)
        fr = slides.feature_row_counts(s)
        sl = slides.get(s)
        x = {f"X_t{c}": int((sl.tile_label == c).sum()) for c in (1, 3, 4, 5)}
        rec = {"slide": s, "is_ref": s in set(refs), "design_n_rows": int(d.n_rows), "featrow_n_rows": fr["n_rows"],
               "X_n_rows": int(sl.X.shape[0])}
        for c in (1, 3, 4, 5):
            rec[f"design_t{c}"] = int(getattr(d, f"t{c}"))
            rec[f"featrow_t{c}"] = fr[f"t{c}"]
            rec[f"X_t{c}"] = x[f"X_t{c}"]
        rec["featrow_equals_design"] = all(rec[f"featrow_t{c}"] == rec[f"design_t{c}"] for c in (1, 3, 4, 5)) \
            and rec["featrow_n_rows"] == rec["design_n_rows"]
        rec["X_differs_from_featrow"] = any(rec[f"X_t{c}"] != rec[f"featrow_t{c}"] for c in (1, 3, 4, 5))
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = {"n_slides_design_csv": int(len(design)), "n_slides_checked": int(len(df)),
           "n_featrow_mismatch": int((~df.featrow_equals_design).sum()),
           "featrow_mismatch_slides": df.loc[~df.featrow_equals_design, "slide"].tolist(),
           "n_slides_X_differs": int(df.X_differs_from_featrow.sum()),
           "slides_X_differs": df.loc[df.X_differs_from_featrow, "slide"].tolist(),
           "all_46_refs_checked": bool(set(refs) <= set(df.slide))}
    out["pass"] = bool(out["n_featrow_mismatch"] == 0 and out["all_46_refs_checked"])
    return out, df


def eq_nan(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(a.shape == b.shape and np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def flatten(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{prefix}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, o


def same_value(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="run directory, e.g. $PROSPER_OUT/r3_5_localisation_maps_20261017/h2/run_001")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args(argv)

    t_wall0, t_cpu0 = time.time(), time.process_time()
    timing = {}
    root = out_root()
    study = root / STUDY
    repo = repo_root()
    d = Path(args.out)
    if d.exists():
        raise Stop(f"{d} exists; refusing to overwrite (a failed run is renamed run_failed_00N)")
    d.mkdir(parents=True)
    me = Path(__file__).resolve()
    script_sha = sha256_file(me)
    snap = snapshot_script(root)
    checks = {"created": now(), "script": str(me), "script_sha256": script_sha, "contract_sha256": CONTRACT_SHA}
    slide_bind = {}

    def fail(stage, msg):
        checks["all_pass"] = False
        checks["failed_stage"] = stage
        checks["failure"] = msg
        jdump(d / "checks.json", checks)
        jdump(d / "FAILED.json", {"created": now(), "stage": stage, "message": msg, "script": str(me),
                                  "script_sha256": script_sha, "checks": "checks.json"})
        print(f"FAIL at {stage}: {msg}", file=sys.stderr)
        return 2

    # ---- bind (specification table)
    t0 = time.time()
    L1, l1_path = load_l1()
    LC, _ = L1.import_prosper()
    files = [binding(study / "contract_008" / "CONTRACT.md", "contract_008", CONTRACT_SHA),
             binding(l1_path, "l1_helpers", L1_SCRIPT_SHA),
             binding(repo / "scripts" / "r3_5_cosine_tie_readback.py", "g1_model_tie_readback", TIE_SCRIPT_SHA),
             binding(Path(LC.__file__), "code:localization_cli")]
    for role, (rel, sha) in INPUTS.items():
        files.append(binding(root / rel, role, sha))
    s0c = json.loads((root / INPUTS["s0_pairs_complete"][0]).read_text())
    meta_pkl = Path(s0c["s0_bind"]) / "meta_tiles.pkl"
    files.append(binding(meta_pkl, "meta_tiles (s0_bind via s0_pairs COMPLETE.json)"))
    bad = [f["role"] for f in files if f.get("matches_contract") is False]
    checks["G5_hash_readback_start"] = {"items": [{k: f[k] for k in ("role", "path", "sha256")} |
                                                  ({"contract_sha256": f["contract_sha256"],
                                                    "matches_contract": f["matches_contract"]} if "contract_sha256" in f else {})
                                                  for f in files],
                                        "pass": not bad}
    if bad:
        return fail("G5", f"sha256 mismatch vs contract table: {bad}")
    s2c = json.loads((root / INPUTS["s2_complete"][0]).read_text())
    if s2c["outputs"]["per_pair_A.csv.gz"] != INPUTS["per_pair_A"][1]:
        return fail("G5", "s2 COMPLETE.json hash of per_pair_A.csv.gz differs from the contract")
    s0_out = s0c.get("outputs", {}).get("pairs_P2.csv")
    checks["G5_hash_readback_start"]["s0_complete_pairs_P2_sha256_matches"] = s0_out == INPUTS["pairs_P2"][1]
    if s0_out != INPUTS["pairs_P2"][1]:
        return fail("G5", "s0_pairs COMPLETE.json hash of pairs_P2.csv differs from the contract")
    timing["bind_s"] = time.time() - t0

    # ---- G4 unit tests
    checks["G4_unit_tests"] = check_G4(repo, d)
    if not checks["G4_unit_tests"]["pass"]:
        return fail("G4", f"pytest returncode {checks['G4_unit_tests']['returncode']}")

    # ---- reference slides, G6
    P = lambda role: root / INPUTS[role][0]
    p2 = pd.read_csv(P("pairs_P2"), dtype=str)
    rinfo = p2[["reference", "ref_patient", "ref_group", "half"]].drop_duplicates("reference")
    refs = numeric_sorted(rinfo.reference)
    info = {r.reference: {"patient": r.ref_patient, "group": r.ref_group, "half": r.half} for r in rinfo.itertuples()}
    if len(refs) != 46:
        return fail("G6", f"{len(refs)} reference slides, contract expects 46")
    pairs_all = [(s, r) for s in refs for r in refs if s != r]
    checks["G6_patient_distinct"] = check_patients_distinct({r: info[r]["patient"] for r in refs}, pairs_all)
    if not checks["G6_patient_distinct"]["pass"]:
        return fail("G6", "reference slides / pairs share a patient")

    t0 = time.time()
    meta = L1.meta_cache_load(meta_pkl)
    slides = SlideSet(L1, LC, meta, slide_bind)
    for r in refs:
        slides.get(r)
    timing["load_refs_s"] = time.time() - t0

    # ---- G2 label counts
    t0 = time.time()
    design = pd.read_csv(P("design_counts"), dtype={"slide": str})
    checks["G2_label_counts"], g2df = check_G2(slides, design, refs)
    timing["G2_s"] = time.time() - t0
    if not checks["G2_label_counts"]["pass"]:
        return fail("G2", "feature-row tile counts differ from the design-stage counts")

    # ---- G1 positive control
    t0 = time.time()
    stored = pd.read_csv(P("per_pair_A"), dtype={"reference": str, "query": str, "ref_patient": str},
                         float_precision="round_trip")
    cres = json.loads(P("cosine_c4_44_result").read_text())["results"]
    c4refs = pd.read_csv(P("c4_44_references"), dtype={"reference": str, "ref_patient": str})
    checks["G1_positive_control"] = check_G1(L1, LC, slides, p2, stored, cres, c4refs, args.B, args.seed)
    del stored
    timing["G1_s"] = time.time() - t0
    if not checks["G1_positive_control"]["pass"]:
        return fail("G1", "tumour-ROI cosine route does not reproduce s2 per_pair_A / the C4-44 aggregates")
    jdump(d / "checks.json", checks)

    # ---- eligibility, ROI draws
    S = {r: slides.get(r) for r in refs}
    counts = {(r, c): class_counts(S[r].patch_label, c) for r in refs for c in (TUMOUR, NECROSIS, NONTUMOR)}
    src = {c: [r for r in refs if is_source(S[r].tile_label, c)] for c in CLASSES}
    tgt_ok = {(c, thr): {r: counts[(r, c)][0] >= thr and counts[(r, c)][1] >= thr for r in refs}
              for c in (TUMOUR, NECROSIS, NONTUMOR) for thr in (THR,) + THR_SENS}
    g2i = g2df.set_index("slide")
    el_rows = []
    for r in refs:
        for c in (TUMOUR, NECROSIS, NONTUMOR):
            rec = {"slide": r, "patient": info[r]["patient"], "group": info[r]["group"], "class": c,
                   "class_name": CLASS_NAME[c], "X_tiles_class": int((S[r].tile_label == c).sum()),
                   "X_tiles_tumour": int((S[r].tile_label == TUMOUR).sum()),
                   "featrow_tiles_class": int(g2i.at[r, f"featrow_t{c}"]),
                   "featrow_tiles_tumour": int(g2i.at[r, "featrow_t1"]),
                   "patches_class": counts[(r, c)][0], "patches_other_annotated": counts[(r, c)][1],
                   "source_eligible": (r in src[c]) if c in CLASSES else None,
                   "source_eligible_featrow": (int(g2i.at[r, f"featrow_t{c}"]) >= ROI_TILES
                                               and int(g2i.at[r, "featrow_t1"]) >= ROI_TILES) if c in CLASSES else None}
            for thr in (THR,) + THR_SENS:
                rec[f"target_eligible_thr{thr}"] = tgt_ok[(c, thr)][r]
            el_rows.append(rec)
    elig = pd.DataFrame(el_rows)
    draws = draw_all(S, refs)
    dr_rows = [{"slide": s, "class": c, "class_name": CLASS_NAME[c], "seed": seed, "primary": seed == ROI_SEED,
                "rows": ";".join(map(str, v.tolist()))} for (seed, c, s), v in draws.items() if v is not None]
    roi_df = pd.DataFrame(dr_rows)  # order: slide (numeric), seed, class

    # ---- per-pair pass
    t0 = time.time()
    scores, same = base_pass(L1, LC, S, refs, src, draws, t0 + STEP_LIMIT_S)
    timing["base_pass_s"] = time.time() - t0
    t0 = time.time()
    sens = sensitivity_pass(L1, LC, S, refs, src, tgt_ok, draws, t0 + STEP_LIMIT_S)
    timing["sensitivity_pass_s"] = time.time() - t0
    timing["n_base_maps"] = len(scores)
    timing["n_sensitivity_maps"] = len(sens)
    pp = per_pair_table(refs, info, src, counts, scores, sens)
    cm_long = class_matrix_long(refs, info, src, counts, scores)
    G6b = check_patients_distinct({r: info[r]["patient"] for r in refs}, list(zip(pp.source, pp.target)))
    checks["G6_patient_distinct"]["per_pair_table"] = G6b
    if not G6b["pass"]:
        return fail("G6", "a per-pair row joins two slides of one patient")

    # ---- estimates (pass 1)
    t0 = time.time()
    est = build_estimates(L1, pp, cm_long, same, refs, info, src, counts, args.B, args.seed)
    timing["estimates_s"] = time.time() - t0

    # ---- G3 determinism (second pass)
    t0 = time.time()
    S2 = {r: LC.Slide(r, L1.FEATURES, meta[r], L1.LABELS) for r in refs}
    draws2 = draw_all(S2, refs)
    draws_equal = all((draws[k] is None and draws2[k] is None) or
                      (draws[k] is not None and draws2[k] is not None and np.array_equal(draws[k], draws2[k]))
                      for k in draws) and set(draws) == set(draws2)
    src2 = {c: [r for r in refs if is_source(S2[r].tile_label, c)] for c in CLASSES}
    scores2, same2 = base_pass(L1, LC, S2, refs, src2, draws2, t0 + STEP_LIMIT_S)
    scores_equal = set(scores) == set(scores2) and all(eq_nan(scores[k], scores2[k]) for k in scores)
    same_equal = set(same) == set(same2) and all(eq_nan(same[k], same2[k]) for k in same)
    pp2 = per_pair_table(refs, info, src2, counts, scores2, sens)
    cm2 = class_matrix_long(refs, info, src2, counts, scores2)
    est2 = build_estimates(L1, pp2, cm2, same2, refs, info, src2, counts, args.B, args.seed)
    f1, f2 = dict(flatten(est)), dict(flatten(est2))
    est_diff = [k for k in f1 if k not in f2 or not same_value(f1[k], f2[k])]
    alt = {}
    for c in CLASSES:
        for m in ("M", "D"):
            a = estimate_both(L1, pp, m, metric_mask(pp, c, m), args.B, args.seed)
            b = estimate_both(L1, pp, m, metric_mask(pp, c, m), args.B, args.seed + 1)
            for k in ("by_target", "by_source"):
                alt[f"{CLASS_NAME[c]}_{m}_{k}"] = {
                    "same_point": a[k]["point"] == b[k]["point"],
                    "different_interval": (a[k]["ci95_lo"] != b[k]["ci95_lo"]) or (a[k]["ci95_hi"] != b[k]["ci95_hi"])}
    timing["G3_s"] = time.time() - t0
    checks["G3_determinism"] = {
        "roi_draws_identical_second_pass": bool(draws_equal), "n_roi_draws": len(draws),
        "base_map_scores_identical_second_pass": bool(scores_equal), "n_base_map_scores": len(scores),
        "same_slide_identical_second_pass": bool(same_equal),
        "per_pair_primary_identical": bool(all(eq_nan(pp[c], pp2[c]) for c in ("M", "C", "D", "TM", "TC", "TD"))),
        "every_estimate_identical_second_pass": not est_diff, "n_estimate_leaves": len(f1),
        "estimate_leaves_differing": est_diff[:20],
        "alt_bootstrap_seed": args.seed + 1, "alt_seed": alt,
        "alt_seed_same_points_all": all(v["same_point"] for v in alt.values()),
        "alt_seed_different_intervals_all": all(v["different_interval"] for v in alt.values()),
        "scope_note": ("second pass recomputes slides, ROI draws, all base (cosine seed-123) map scores, same-slide "
                       "values and every estimate; sensitivity map scores are reused from pass 1"),
        "note": "estimate values are not recorded in this check"}
    g3 = checks["G3_determinism"]
    g3["pass"] = bool(g3["roi_draws_identical_second_pass"] and g3["base_map_scores_identical_second_pass"]
                      and g3["same_slide_identical_second_pass"] and g3["per_pair_primary_identical"]
                      and g3["every_estimate_identical_second_pass"] and g3["alt_seed_same_points_all"]
                      and g3["alt_seed_different_intervals_all"])
    if not g3["pass"]:
        return fail("G3", "determinism / alternative-seed check failed")
    del S2, scores2, pp2

    # ---- G5 final readback
    t0 = time.time()
    items, ok = [], True
    for f in files:
        h = sha256_file(f["path"])
        items.append({"role": f["role"], "match": h == f["sha256"]})
        ok &= h == f["sha256"]
    n_sl, sl_bad = 0, []
    for s, b in slide_bind.items():
        for k in ("features", "labels"):
            n_sl += 1
            if sha256_file(b[k]["path"]) != b[k]["sha256"]:
                sl_bad.append(f"{k}:{s}")
    checks["G5_hash_readback_end"] = {"n_table_files": len(files), "table_files_all_match": bool(ok),
                                      "n_slide_files": n_sl, "n_slides": len(slide_bind), "slide_files_mismatch": sl_bad,
                                      "wall_s": time.time() - t0}
    checks["G5_hash_readback_end"]["pass"] = bool(ok and not sl_bad)
    if not checks["G5_hash_readback_end"]["pass"]:
        return fail("G5", "input files changed during the run")
    checks["G2_source_counts"] = {
        CLASS_NAME[c]: {"sources_X": len(src[c]),
                        "sources_featrow": int(elig[(elig["class"] == c)].source_eligible_featrow.sum()),
                        "design_contract": {3: 17, 4: 22}[c],
                        "slides_differing": elig[(elig["class"] == c) & (elig.source_eligible != elig.source_eligible_featrow)
                                                 ].slide.tolist()} for c in CLASSES}
    checks["all_pass"] = all(checks[k]["pass"] for k in ("G1_positive_control", "G2_label_counts", "G3_determinism",
                                                         "G4_unit_tests", "G5_hash_readback_start",
                                                         "G5_hash_readback_end", "G6_patient_distinct"))
    jdump(d / "checks.json", checks)
    if not checks["all_pass"]:
        return fail("checks", "not all checks passed")

    # ---- outputs
    bindings = {"created": now(), "script": str(me), "script_sha256": script_sha, "snapshot": snap,
                "contract_sha256": CONTRACT_SHA, "B": args.B, "seed": args.seed, "files": files,
                "slides": slide_bind, "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}
    jdump(d / "bindings.json", bindings)
    roi_df.to_csv(d / "roi_draws.csv", index=False)
    elig.to_csv(d / "eligibility.csv", index=False)
    pp.to_csv(d / "per_pair_H2.csv.gz", index=False, float_format="%.17g")
    for by, name in (("target", "per_target_H2.csv"), ("source", "per_source_H2.csv")):
        tabs = []
        for c in CLASSES:
            t = None
            for m in METRICS:
                a = aggregate(pp, m, metric_mask(pp, c, m), by)[["unit", "patient", "value", "n_rows"]]
                a = a.rename(columns={"value": m, "n_rows": f"n_{m}"})
                t = a if t is None else t.merge(a, on=["unit", "patient"], how="outer")
            t.insert(0, "class_name", CLASS_NAME[c])
            t.insert(0, "class", c)
            tabs.append(t.rename(columns={"unit": by, "patient": f"{by}_patient"}))
        pd.concat(tabs, ignore_index=True).to_csv(d / name, index=False, float_format="%.17g")
    cm_rows = []
    for k, v in est["secondary"]["3_class_matrix"].items():
        cm_rows.append({"roi_class": v["roi_class"], "roi_class_name": CLASS_NAME[v["roi_class"]],
                        "target_class": v["target_class"], "target_class_name": CLASS_NAME[v["target_class"]],
                        "point": v["by_target"]["point"],
                        "by_target_ci95_lo": v["by_target"]["ci95_lo"], "by_target_ci95_hi": v["by_target"]["ci95_hi"],
                        "by_source_point": v["by_source"]["point"],
                        "by_source_ci95_lo": v["by_source"]["ci95_lo"], "by_source_ci95_hi": v["by_source"]["ci95_hi"],
                        "n_targets": v["n_targets"], "n_sources": v["n_sources"], "n_pairs": v["n_pairs"]})
    pd.DataFrame(cm_rows).sort_values(["roi_class", "target_class"]).to_csv(d / "class_matrix.csv", index=False,
                                                                           float_format="%.17g")
    timing["total_wall_s"] = time.time() - t_wall0
    timing["total_cpu_s"] = time.process_time() - t_cpu0
    results = {"created": now(), "stage": "R3.5-H2", "status": "post hoc", "script": str(me),
               "script_sha256": script_sha, "snapshot": snap["snapshot"], "contract_sha256": CONTRACT_SHA,
               "B": args.B, "seed": args.seed,
               "generator": "L1.make_patient_draws (numpy Generator(PCG64(seed))) over the sorted patients of each "
                            "estimate's unit population; L1.replicate_means / L1.summarise_reps; 95 % percentile",
               "unit": {"by_target": "per target: mean over its sources -> mean over targets; bootstrap over target patients",
                        "by_source": "per source: mean over its targets -> mean over sources; bootstrap over source patients"},
               "heat_map": "L1.roi_map(target, q, 'cosine', 'mean'); q = mean of 8 Slide.X rows (default_rng([123, c, sid]))",
               "n_sources": {CLASS_NAME[c]: len(src[c]) for c in CLASSES},
               "n_target_eligible_thr20": {CLASS_NAME[c]: int(sum(tgt_ok[(c, THR)].values())) for c in (1, 3, 4)},
               **est, "timing": timing}
    jdump(d / "results.json", results)
    names = ("bindings.json", "checks.json", "results.json", "roi_draws.csv", "eligibility.csv", "per_pair_H2.csv.gz",
             "per_target_H2.csv", "per_source_H2.csv", "class_matrix.csv", "pytest_output.txt")
    jdump(d / "COMPLETE.json", {"created": now(), "stage": "R3.5-H2", "script": str(me), "script_sha256": script_sha,
                                "snapshot": snap, "contract_sha256": CONTRACT_SHA, "B": args.B, "seed": args.seed,
                                "outputs": {n: sha256_file(d / n) for n in names}, "timing": timing,
                                "all_checks_pass": True, "argv": sys.argv[1:], "cwd": os.getcwd(), "python": sys.version})
    print(json.dumps({"out": str(d), "all_pass": True, "timing": timing}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        sys.exit(3)
