#!/usr/bin/env python3
"""R2.6 TITAN all-tissue vs tumour-enriched: legacy reconciliation, common-pool
paired re-rank with whole-patient bootstrap, and connectivity of retained patches.

Run from the repository root with PROSPER_OUT exported. Reads only the `coords`
dataset of the tumour-enriched CONCH h5 files (never `features`).
Optional: `--pool-matrices` restricts queries/library to the common queries present in every
listed matrix (intersection pool); `--variant-dir stage2/<V>` takes the connectivity inputs
from a stage-2 variant and allows its `_titan_infeasible_slides.csv` slides to lack embeddings.
"""
import argparse, datetime, glob, hashlib, json, os, sys, time

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

DROP = {"MISC", "NORMAL"}
SENTINEL_MIN = 1.0e5          # matrices store masked cells (self, same patient) as 1e6
DEPTH, KS, B, SEED, GRID = 10, (1, 3, 5, 10), 5000, 123, 1024
PKG = "/PROSPER_WORKSPACE/prosper-search"
DRAWS_SRC = PKG + "/prosper/evaluation/r2_1_hoptimus_main.py"
RL_COLS = ["query", "true_label", "rank", "neighbour", "neighbour_label", "distance"]


class CheckError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# Copied from DRAWS_SRC::make_patient_draws (sampling body verbatim; the package's
# input-validation helpers are reduced to sorted(set(str ids))).
def make_patient_draws(patient_ids, *, count=5000, seed=123):
    patients = tuple(sorted(set(str(p) for p in patient_ids)))
    p = len(patients)
    multiplicities = np.zeros((count, p), dtype=np.int64)
    if p:
        rng = np.random.Generator(np.random.PCG64(seed))
        for b in range(count):
            multiplicities[b] = np.bincount(
                rng.integers(p, size=p), minlength=p)
    return patients, multiplicities


def load_meta(path):
    m = pd.read_csv(path, dtype=str)
    if m["Slide.Scan.File"].duplicated().any():
        raise CheckError("duplicate Slide.Scan.File in metadata")
    return m.set_index("Slide.Scan.File")[["Tumor.group", "UID.Subject", "Sample.ID"]].rename(
        columns={"Tumor.group": "label", "UID.Subject": "patient", "Sample.ID": "sample"})


def masked(M, queries, library, meta):
    """Query x library distances with self, sentinel and same-patient cells set to inf."""
    D = M.loc[queries, library].to_numpy(dtype=float, copy=True)
    qp = meta.loc[queries, "patient"].to_numpy()
    lp = meta.loc[library, "patient"].to_numpy()
    same_pat = qp[:, None] == lp[None, :]
    selfm = np.asarray(queries)[:, None] == np.asarray(library)[None, :]
    sent = D >= SENTINEL_MIN
    D[sent | same_pat | selfm] = np.inf
    diag = {"same_patient_unmasked_in_matrix": int((same_pat & ~sent & ~selfm).sum()),
            "sentinel_cells_not_same_patient": int((sent & ~same_pat & ~selfm).sum())}
    return D, diag


def rank(M, queries, library, meta, depth=DEPTH):
    """Depth-ranked neighbours: ascending distance, stable order of `library`;
    self, sentinel cells and same-patient pairs are excluded before sorting."""
    D, diag = masked(M, queries, library, meta)
    order = np.argsort(D, axis=1, kind="stable")[:, :depth]
    dist = np.take_along_axis(D, order, axis=1)
    if not np.isfinite(dist).all():
        raise CheckError("a query has fewer than depth eligible neighbours")
    lib = np.asarray(library)
    nb = lib[order]
    df = pd.DataFrame({"query": np.repeat(queries, depth),
                       "true_label": np.repeat(meta.loc[queries, "label"].to_numpy(), depth),
                       "rank": np.tile(np.arange(1, depth + 1), len(queries)),
                       "neighbour": nb.ravel(),
                       "neighbour_label": meta.loc[nb.ravel(), "label"].to_numpy(),
                       "distance": dist.ravel()})
    return df, diag


def tie_diagnostics(M, queries, library, meta, depth=DEPTH, k=5):
    """Exact-distance ties affecting the top-`depth` list, and the min/max hit@k
    attainable over all valid tie orders (per query, in `queries` order)."""
    D, _ = masked(M, queries, library, meta)
    Ds = np.sort(D, axis=1)
    top = Ds[:, :depth + 1]
    tied = (np.diff(top, axis=1) == 0).any(axis=1)
    rel = meta.loc[queries, "label"].to_numpy()[:, None] == meta.loc[library, "label"].to_numpy()[None, :]
    dk = Ds[:, k - 1:k]
    below, at = D < dk, D == dk
    slots = k - below.sum(axis=1)
    sure = (below & rel).any(axis=1)
    r_at, n_at = (at & rel).sum(axis=1), at.sum(axis=1)
    hmax = (sure | (r_at > 0)).astype(float)
    hmin = (sure | (slots > n_at - r_at)).astype(float)
    return {"queries_with_ties_in_top_or_at_cutoff": int(tied.sum()),
            "queries_hit_at_5_undecided": int((hmin != hmax).sum())}, hmin, hmax


def identical_clique(M):
    """Slides in any zero-distance (non-sentinel, off-diagonal) pair of matrix M."""
    a = M.to_numpy()
    z = (a == 0) & ~np.eye(len(a), dtype=bool)
    members = [s for s, f in zip(M.index, z.any(axis=1)) if f]
    idx = M.index.get_indexer(members)
    rest = np.setdiff1d(np.arange(len(a)), idx)
    sub = a[np.ix_(idx, rest)]
    real = ~(sub >= SENTINEL_MIN).any(axis=0)       # skip columns masked (same patient) for any member
    return members, {"n_slides": len(members), "n_zero_pairs": int(z.sum() // 2),
                     "n_other_columns_compared": int(real.sum()), "n_other_columns_with_sentinel": int((~real).sum()),
                     "rows_identical_outside_clique": bool((sub[:, real] == sub[:1, real]).all())}


def same_patient_guard(df, meta):
    qp = meta.loc[df["query"].astype(str), "patient"].to_numpy()
    npat = meta.loc[df["neighbour"].astype(str), "patient"].to_numpy()
    bad = int((qp == npat).sum())
    if bad:
        raise CheckError(f"{bad} emitted query-neighbour pairs share a patient")
    return len(df)


def load_ranked(path):
    df = pd.read_csv(path, dtype={"query": str, "neighbour": str})
    return df.sort_values(["query", "rank"], kind="stable").reset_index(drop=True)


def compare(new, old, M, library):
    """Per-query comparison of ranked neighbours; tie-permutations allowed only
    among exactly equal matrix distances, and every saved neighbour must be a
    distinct library member (so a final tied neighbour is an eligible one)."""
    new = new.sort_values(["query", "rank"], kind="stable").reset_index(drop=True)
    old = old.sort_values(["query", "rank"], kind="stable").reset_index(drop=True)
    out = {"n_queries_new": int(new["query"].nunique()), "n_queries_old": int(old["query"].nunique())}
    if set(new["query"]) != set(old["query"]) or len(new) != len(old):
        out.update(passed=False, reason="query sets or row counts differ")
        return out
    ri, ci = M.index.get_indexer(old["query"]), M.columns.get_indexer(old["neighbour"])
    old_mat = M.to_numpy()[ri, ci]
    out["max_abs_saved_vs_matrix_distance"] = float(np.max(np.abs(old_mat - old["distance"].to_numpy())))
    nq = out["n_queries_new"]
    nb_n, nb_o = new["neighbour"].to_numpy().reshape(nq, -1), old["neighbour"].to_numpy().reshape(nq, -1)
    d_n, d_o = new["distance"].to_numpy().reshape(nq, -1), old_mat.reshape(nq, -1)
    exact = (nb_n == nb_o).all(axis=1)
    lib = set(library)
    tie_ok, bad = np.zeros(nq, bool), []
    for i in np.where(~exact)[0]:
        ok = (np.array_equal(d_n[i], d_o[i]) and len(set(nb_o[i])) == len(nb_o[i])
              and set(nb_o[i]) <= lib and all(set(nb_n[i][d_n[i] == v]) == set(nb_o[i][d_o[i] == v])
                                              for v in np.unique(d_n[i])[:-1]))
        tie_ok[i] = ok
        if not ok and len(bad) < 10:
            bad.append({"query": str(new["query"].iloc[i * nb_n.shape[1]]), "new": nb_n[i].tolist(),
                        "old": nb_o[i].tolist(), "new_d": d_n[i].tolist(), "old_d_matrix": d_o[i].tolist()})
    out.update(exact_match_queries=int(exact.sum()), tie_permutation_queries=int(tie_ok.sum()),
               mismatch_queries=int((~exact & ~tie_ok).sum()), mismatch_examples=bad,
               passed=bool((exact | tie_ok).all() and out["max_abs_saved_vs_matrix_distance"] <= 1e-9))
    return out


def plurality(labels):
    """Most frequent label; ties -> the tied label occurring at the closest rank."""
    labels = list(labels)
    counts = {l: labels.count(l) for l in labels}
    top = max(counts.values())
    return next(l for l in labels if counts[l] == top)


def per_query(df):
    df = df.sort_values(["query", "rank"], kind="stable")
    q = df["query"].drop_duplicates().to_numpy()
    L = df["neighbour_label"].to_numpy().reshape(len(q), -1)
    t = df.groupby("query", sort=False)["true_label"].first().loc[q].to_numpy()
    rel = L == t[:, None]
    out = pd.DataFrame({"query": q, "true_label": t})
    for k in KS:
        out[f"hit@{k}"] = rel[:, :k].any(axis=1).astype(float)
    out["precision@5"] = rel[:, :5].mean(axis=1)
    out["plurality5_label"] = [plurality(r[:5]) for r in L]
    out["plurality@5"] = (out["plurality5_label"].to_numpy() == t).astype(float)
    out["top1_label"] = L[:, 0]
    return out


def micro(v, W):
    return (W @ v) / W.sum(axis=1)


def class_rate(v, W, sel):
    den = W[:, sel].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (W[:, sel] @ v[sel]) / den, np.nan)


def macro(v, W, labels, classes):
    """Unweighted class mean; a draw with any zero-support class is NaN ([decision record]),
    never silently averaged over the remaining classes."""
    return np.column_stack([class_rate(v, W, labels == c) for c in classes]).mean(axis=1)


def ci(point, draws):
    """Percentile interval; unavailable (NaN) if any draw is undefined (zero support)."""
    n_bad = int(np.isnan(draws).sum())
    lo = hi = np.nan
    if n_bad == 0:
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return dict(estimate=float(point), lo=lo, hi=hi, B_effective=int(draws.size - n_bad),
                n_zero_support_draws=n_bad, ci_available=n_bad == 0, ci_degenerate=bool(n_bad == 0 and lo == hi))


def paired_summary(pq1, pq2, meta, population, draws):
    """pq1/pq2: per-query tables with identical query order. Δ = arm2 − arm1."""
    q = pq1["query"].to_numpy()
    if not np.array_equal(q, pq2["query"].to_numpy()):
        raise CheckError("arms have different query order")
    pts, mult = draws
    pat = meta.loc[q, "patient"].to_numpy()
    idx = {p: i for i, p in enumerate(pts)}
    W = mult[:, [idx[p] for p in pat]].astype(float)
    one = np.ones((1, len(q)))
    labels = pq1["true_label"].to_numpy()
    classes = sorted(set(labels))
    stats = {f"micro_hit@{k}": (lambda v, W: micro(v, W), f"hit@{k}") for k in KS}
    stats.update({f"macro_hit@{k}": (lambda v, W: macro(v, W, labels, classes), f"hit@{k}") for k in KS})
    stats["micro_plurality@5"] = (lambda v, W: micro(v, W), "plurality@5")
    rows, cls_rows = [], []
    base = dict(population=population, n_queries=len(q), n_patients=len(set(pat)))
    for name, (fn, col) in stats.items():
        v1, v2 = pq1[col].to_numpy(), pq2[col].to_numpy()
        p1, p2, b1, b2 = fn(v1, one)[0], fn(v2, one)[0], fn(v1, W), fn(v2, W)
        for arm, p, b in (("arm1", p1, b1), ("arm2", p2, b2), ("delta", p2 - p1, b2 - b1)):
            rows.append({**base, "metric": name, "arm": arm, **ci(p, b)})
    for c in classes:
        sel = labels == c
        for col in ("hit@5", "plurality@5"):
            v1, v2 = pq1[col].to_numpy(), pq2[col].to_numpy()
            p1, p2 = v1[sel].mean(), v2[sel].mean()
            b1, b2 = class_rate(v1, W, sel), class_rate(v2, W, sel)
            for arm, p, b in (("arm1", p1, b1), ("arm2", p2, b2), ("delta", p2 - p1, b2 - b1)):
                cls_rows.append({**base, "class": c, "metric": col + ("" if col == "hit@5" else "_recall"),
                                 "arm": arm, "n_class_queries": int(sel.sum()),
                                 "n_class_patients": int(len(set(pat[sel]))), **ci(p, b)})
    return pd.DataFrame(rows), pd.DataFrame(cls_rows)


def components(coords, grid=GRID):
    """4-connected components of patches on the `grid` lattice."""
    g = coords // grid
    g = g - g.min(axis=0)
    img = np.zeros(tuple(g.max(axis=0)[::-1] + 1), dtype=bool)
    img[g[:, 1], g[:, 0]] = True
    lab, n = ndimage.label(img)          # default structure = 4-connectivity
    sizes = np.bincount(lab[img])[1:] if n else np.array([], int)
    return dict(n_components=int(n), largest_component=int(sizes.max()) if n else 0,
                n_singletons=int((sizes == 1).sum()), n_off_grid=int((coords % grid != 0).any(axis=1).sum()),
                n_duplicate_coords=int(len(coords) - int(img.sum())))


def connectivity(h5dir, summary, excluded, meta):
    rows, coord_hash = [], hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(h5dir, "*.h5"))):
        s = os.path.basename(f)[:-3]
        with h5py.File(f, "r") as h:
            c = h["coords"][()].reshape(-1, 2)      # trident squeezes one-patch coords to (2,)
        coord_hash.update(s.encode() + c.astype("<i8").tobytes())
        rows.append({"slide": s, "n_kept_h5": len(c), **components(c)})
    df = pd.DataFrame(rows)
    df["label"] = meta.loc[df["slide"], "label"].to_numpy()
    df["patient"] = meta.loc[df["slide"], "patient"].to_numpy()
    df = df.merge(summary, on="slide", how="left")
    if "mean_tiles_scored_per_patch" not in df:     # variant patch summaries lack this column
        df["mean_tiles_scored_per_patch"] = np.nan
    df["largest_component_frac"] = df["largest_component"] / df["n_kept_h5"]
    df["unscored_tile_fraction_all_patches"] = 1 - df["mean_tiles_scored_per_patch"] / 4
    q = lambda p: (lambda x: x.quantile(p))
    agg = {}
    for col in ("n_kept_h5", "n_components", "largest_component_frac", "frac_kept", "unscored_tile_fraction_all_patches"):
        agg.update({f"{col}_median": (col, "median"), f"{col}_q1": (col, q(.25)), f"{col}_q3": (col, q(.75))})
    for t in (8, 32, 128):
        agg[f"n_largest_ge_{t}"] = ("largest_component", lambda x, t=t: int((x >= t).sum()))
    g = pd.concat([df, df.assign(label="ALL")]).groupby("label").agg(
        n_slides=("slide", "size"), **agg).reset_index()
    g.insert(0, "section", "kept")
    ex = excluded.copy()
    ex["label"] = meta.loc[ex["slide"], "label"].to_numpy()
    ex["reason_category"] = np.where(ex["reason"].str.contains("tumour patches <"),
                                     "fewer_than_8_tumour_patches", ex["reason"].str.replace(" ", "_"))
    e = ex.groupby(["label", "reason_category"]).size().rename("n_slides").reset_index()
    e.insert(0, "section", "excluded")
    return df, pd.concat([g, e], ignore_index=True), coord_hash.hexdigest()


SKIP_REASON = {"arm1_matrix": "arm-1 matrix overridden (metric/selection differs from legacy run)",
               "arm2_matrix": "arm-2 matrix overridden (metric/selection differs from legacy run)"}
LEGACY_POP = "legacy_population_lib1_2604_lib2_2564"
LEGACY_NOTE = "legacy ranked lists; arm 2 cosine; not an L1 result"
LEGACY_ARMS_NOTE = ("legacy arms: arm1 = published all-tissue TITAN (cityblock), arm2 = S2 tumour-enriched TITAN "
                    "(cosine); not this run's arms")
S2_WORDS = "S2 tumour-enriched TITAN (UNI2 tile-random selection)"
ARM1_L1_CHECK = "b_arm1_cityblock_recompute_equals_published_full_matrix_info"


def resolve_outdir(arg, default):
    """Without --outdir, refuse to write into a default outdir that already holds results."""
    if arg is not None:
        return arg
    if os.path.exists(os.path.join(default, "summary_common_pool.csv")):
        raise SystemExit(f"REFUSING: {default} already contains summary_common_pool.csv (frozen run); "
                         "pass --outdir to write elsewhere")
    return default


def legacy_skips(overrides):
    """Legacy reconciliation checks to skip (name -> reason) given the overridden matrices."""
    s = {}
    if "arm2_matrix" in overrides:
        s["reconcile_a_arm2_legacy"] = SKIP_REASON["arm2_matrix"]
    if "arm1_matrix" in overrides:
        s["reconcile_b_arm1_legacy"] = s["reconcile_b_arm1_reference_2604"] = SKIP_REASON["arm1_matrix"]
    reasons = [SKIP_REASON[a] for a in ("arm1_matrix", "arm2_matrix") if a in overrides]
    if reasons:
        s.update({f"c_legacy_macro_delta_hit@{k}": "; ".join(reasons) for k in KS})
    return s


def arm1_metric_evidence(arm2_record, arm1_sha, recorded):
    """Arm-1 metric: cityblock only if the reconciliation.json next to the arm-2 record shows a
    passing bitwise cityblock recompute of this exact arm-1 matrix; else the recorded value."""
    fallback = "unrecorded" if str(recorded).startswith("no record") else recorded
    p = os.path.join(os.path.dirname(arm2_record), "reconciliation.json")
    try:
        r = json.load(open(p))
    except (OSError, ValueError):
        return fallback, None
    c = r.get("checks", {}).get(ARM1_L1_CHECK, {})
    if (c.get("result") == "PASS" and c.get("bitwise_equal_full_matrix") is True
            and r.get("inputs", {}).get("arm1_matrix", {}).get("sha256") == arm1_sha):
        return "cityblock", f"`{p}` check `{ARM1_L1_CHECK}` (arm-1 matrix sha256 matches this run)"
    return fallback, None


def annotate_metrics(summary, m1, m2, legacy_note=LEGACY_NOTE):
    s = summary.copy()
    leg = (s["population"] == LEGACY_POP).to_numpy()
    s["arm1_metric"] = np.where(leg, "cityblock", m1)
    s["arm2_metric"] = np.where(leg, "cosine", m2)
    s["note"] = np.where(leg, legacy_note, "")
    return s


def same_path(a, b):
    """Same file (any mount alias) if both exist, else the same absolute path."""
    if os.path.exists(a) and os.path.exists(b):
        return os.path.samefile(a, b)
    return os.path.abspath(a) == os.path.abspath(b)


def arm_words(record, overridden, default_words, matrix_path, variant_dir=None):
    """Plain-language description of an arm from its variant dir or its matrix record."""
    if variant_dir:
        return f"direct-1024 variant {os.path.basename(os.path.normpath(variant_dir))}"
    if not overridden:
        return default_words
    if not record:
        return f"unidentified matrix `{os.path.basename(matrix_path)}` (no matching record)"
    sfd, name = str(record.get("slide_features_dir") or ""), os.path.basename(matrix_path)
    if record.get("tumor_only") is False:
        return ("repaired all-tissue TITAN" if "repaired" in name or "regenerated" in sfd
                else f"all-tissue TITAN (`{name}`)")
    if record.get("tumor_only") is True:
        if "s2_titan_tumor_only" in sfd:
            return S2_WORDS
        parts = os.path.normpath(sfd).split(os.sep)
        if "stage2" in parts and parts.index("stage2") + 1 < len(parts):
            return f"direct-1024 variant {parts[parts.index('stage2') + 1]}"
        return f"tumour-enriched TITAN (`{name}`)"
    return f"unidentified matrix `{name}` (record lacks `tumor_only`)"


VARIANT_SUB = os.path.join("job_dir", "40x_1024px_0px_overlap")
INFEASIBLE_CSV = "_titan_infeasible_slides.csv"
RECORD_GLOB = "/_combined_similarity_matrix_titan_titan*_record.json"


def intersection_pool(qc, indexes):
    """Queries of `qc` (order kept) present in every (name, index) of `indexes`; also the
    dropped queries with the names of the indexes they are absent from."""
    sets = [(name, set(ix)) for name, ix in indexes]
    pool, dropped = [], []
    for s in qc:
        absent = [name for name, ix in sets if s not in ix]
        if absent:
            dropped.append((s, absent))
        else:
            pool.append(s)
    return pool, dropped


def pool_dropped_table(dropped, meta):
    return pd.DataFrame([{"slide": s, "label": meta.at[s, "label"], "patient": meta.at[s, "patient"],
                          "reason": "absent_from:" + ";".join(a)} for s, a in dropped],
                        columns=["slide", "label", "patient", "reason"])


def variant_paths(variant_dir):
    """Connectivity inputs of a stage-2 variant directory (`stage2/<V>`); the TITAN-infeasible
    list is included only if present."""
    d = os.path.join(variant_dir, VARIANT_SUB)
    p = {"patch_summary": os.path.join(d, "_tumour_patch_summary.csv"),
         "excluded": os.path.join(d, "_excluded_slides.csv"),
         "subset_record": os.path.join(d, "_subset_record.json")}
    if os.path.exists(os.path.join(d, INFEASIBLE_CSV)):
        p["titan_infeasible"] = os.path.join(d, INFEASIBLE_CSV)
    return p, os.path.join(d, "features_conch_v15"), os.path.join(d, "slide_features_titan")


def read_infeasible(path):
    return [] if path is None else pd.read_csv(path, dtype={"slide": str})["slide"].tolist()


def h5_vs_matrix(h5_slides, matrix_index, infeasible):
    """Variant check: h5 slides minus the TITAN-infeasible ones equal the arm-2 matrix slides."""
    h5, inf, mx = set(h5_slides), set(infeasible), set(matrix_index)
    return h5 - inf == mx and inf <= h5, dict(n_h5=len(h5), n_infeasible=len(inf), n_infeasible_in_h5=len(inf & h5),
                                                n_infeasible_in_arm2_matrix=len(inf & mx))


def matrix_records(matrix_path):
    """Record files next to a matrix whose `out` is this matrix, their common metric and the other
    record files. Metric None if there is no record file at all, "unverified" if record files exist
    but none names this matrix (never borrowed); CheckError if matching records disagree."""
    recs = sorted(glob.glob(os.path.dirname(matrix_path) + RECORD_GLOB))
    loaded = [(r, json.load(open(r))) for r in recs]
    same = [(r, j) for r, j in loaded if j.get("out") and os.path.exists(j["out"])
            and os.path.exists(matrix_path) and os.path.samefile(j["out"], matrix_path)]
    others = [r for r, j in loaded if (r, j) not in same]
    metrics = sorted({str(j.get("metric")) for _, j in same})
    if len(metrics) > 1:
        raise CheckError(f"record files next to {matrix_path} disagree on metric: {metrics}")
    return [r for r, _ in same], (metrics[0] if same else "unverified" if others else None), others


def main():
    t0 = time.time()
    R = os.getcwd()
    out_root = os.environ.get("PROSPER_OUT")
    if not out_root:
        sys.exit("export PROSPER_OUT first")
    P = os.path.dirname(R)
    paths = {
        "arm1_matrix": P + "/comet-feature-gen/comet-titan/output/similarity_matrix/combined_similarity_matrix_titan_titan_no_same_patient.pkl",
        "arm2_matrix": out_root + "/similarity_matrix/combined_similarity_matrix_titan_titan_tumor_only_no_same_patient.pkl",
        "arm2_record": out_root + "/similarity_matrix/_combined_similarity_matrix_titan_titan_tumor_only_record.json",
        "metadata": P + "/comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv",
        "legacy_arm1": out_root + "/s2_paired_in/ranked_lists_titan_titan.csv.gz",
        "legacy_arm2": out_root + "/s2_paired_in/ranked_lists_titan_titan_tumor_only.csv.gz",
        "common_queries": out_root + "/s2_paired_in/_common_queries.txt",
        "arm1_reference": out_root + "/s2_tumor_only_reference/ranked_lists_titan_titan.csv.gz",
        "patch_summary": out_root + "/s2_titan_tumor_only/40x_1024px_0px_overlap/_tumour_patch_summary.csv",
        "excluded": out_root + "/s2_titan_tumor_only/40x_1024px_0px_overlap/_excluded_slides.csv",
        "subset_record": out_root + "/s2_titan_tumor_only/40x_1024px_0px_overlap/_subset_record.json",
        **{f"legacy_paired_k{k}": out_root + f"/s2_paired/paired_vs_titan_titan_tumor_only_k{k}.json" for k in KS}}
    h5dir = out_root + "/s2_titan_tumor_only/40x_1024px_0px_overlap/features_conch_v15"
    outdir = os.path.join(out_root, "r2_6_titan_common_pool_20261003")
    ap = argparse.ArgumentParser()
    for a in ("arm1_matrix", "arm2_matrix", "arm2_record", "outdir"):
        ap.add_argument("--" + a.replace("_", "-"), default=None if a == "outdir" else paths[a])
    ap.add_argument("--pool-matrices", nargs="+", default=None,
                    help="restrict queries/library to common queries present in every listed matrix (and arms 1/2)")
    ap.add_argument("--variant-dir", default=None,
                    help="stage2/<V>: take connectivity inputs from <V>/job_dir/40x_1024px_0px_overlap/")
    ap.add_argument("--expect-pool", default=None,
                    help="pool_queries.txt of another run: fail unless this run's pool has the same ids in the same order")
    args = ap.parse_args()
    overrides = {a: getattr(args, a) for a in ("arm1_matrix", "arm2_matrix", "arm2_record")
                 if not same_path(getattr(args, a), paths[a])}
    paths.update(overrides)
    variant_sfd = None
    if args.variant_dir:
        vp, h5dir, variant_sfd = variant_paths(args.variant_dir)
        paths.update(vp)
    if args.expect_pool:
        paths["expect_pool"] = args.expect_pool
    custom = bool(overrides or args.pool_matrices or args.variant_dir or args.expect_pool)
    outdir = resolve_outdir(args.outdir, outdir)
    skips = legacy_skips(overrides)
    os.makedirs(outdir, exist_ok=True)
    rec = {"timestamp": datetime.datetime.now().astimezone().isoformat(), "command": " ".join([sys.executable] + sys.argv),
           "script": {"path": os.path.abspath(__file__), "sha256": sha256(__file__)},
           "make_patient_draws_source": {"path": DRAWS_SRC, "sha256": sha256(DRAWS_SRC)},
           "B": B, "seed": SEED, "depth": DEPTH, "drop_labels": sorted(DROP),
           "inputs": {k: {"path": v, "sha256": sha256(v)} for k, v in paths.items()}, "checks": {}}
    if overrides:
        rec["overrides"] = overrides
        rec["checks_skipped"] = {}
        rec["note_legacy_population"] = ("population legacy_population_lib1_2604_lib2_2564 is computed from the "
                                         "saved legacy ranked lists, not from the overridden matrices")
    if args.variant_dir:
        rec["variant"] = {"dir": args.variant_dir, "h5dir": h5dir, "slide_features_dir": variant_sfd,
                          "infeasible_csv": paths.get("titan_infeasible")}
    checks = rec["checks"]

    def skipped(name, reason):
        rec["checks_skipped"][name] = {"skipped": True, "reason": reason}
        print(f"[SKIP] {name} {reason}")

    def check(name, ok, **info):
        checks[name] = {"passed": bool(ok), **info}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info, default=str)[:300]}")

    def finish(code):
        rec["runtime_s"] = round(time.time() - t0, 1)
        with open(os.path.join(outdir, "reconciliation.json"), "w") as f:
            json.dump(rec, f, indent=1, default=str)
        return code

    meta = load_meta(paths["metadata"])
    M1, M2 = pd.read_pickle(paths["arm1_matrix"]), pd.read_pickle(paths["arm2_matrix"])
    rec2 = json.load(open(paths["arm2_record"]))
    rec1, rec1_metric, rec1_other = matrix_records(paths["arm1_matrix"])
    for name, M in (("arm1", M1), ("arm2", M2)):
        a = M.to_numpy()
        real = a[a < SENTINEL_MIN]
        off = ~np.eye(len(a), dtype=bool)
        info = {"shape": list(M.shape), "index_equals_columns": bool((M.index == M.columns).all()),
                "symmetric": bool(np.array_equal(a, a.T)), "sentinel_value": sorted(set(a[a >= SENTINEL_MIN].tolist())),
                "sentinel_offdiag_pairs": int((a[off] >= SENTINEL_MIN).sum() // 2),
                "max_real_distance": float(real.max()), "min_offdiag_distance": float(a[off].min()),
                "zero_offdiag_pairs": int((a[off] == 0).sum() // 2), "nan_or_inf": int((~np.isfinite(a)).sum())}
        rec["inputs"][f"{name}_matrix"].update(info)
        check(f"{name}_matrix_well_formed", info["index_equals_columns"] and info["nan_or_inf"] == 0
              and set(M.index) <= set(meta.index), **info)
    rec["inputs"]["arm1_matrix"]["metric"] = (
        rec1_metric if rec1_metric is not None else
        "no record file; inferred non-cosine distance (max real value "
        f"{rec['inputs']['arm1_matrix']['max_real_distance']:.1f} exceeds the cosine-distance bound 2)")
    if rec1:
        rec["inputs"]["arm1_matrix"]["metric_record"] = rec1
    if rec1_other:
        rec["inputs"]["arm1_matrix"]["records_not_naming_this_matrix"] = rec1_other
    rec["inputs"]["arm2_matrix"].update(metric=rec2["metric"], record_max_real_distance=rec2["max_real_distance"],
                                        record_same_patient_pairs_masked=rec2["same_patient_pairs_masked"])
    check("arm2_record_consistent", abs(rec2["max_real_distance"] - rec["inputs"]["arm2_matrix"]["max_real_distance"]) < 1e-12
          and rec2["same_patient_pairs_masked"] == rec["inputs"]["arm2_matrix"]["sentinel_offdiag_pairs"])
    if "arm2_matrix" in overrides or "arm2_record" in overrides:
        o2 = rec2.get("out")
        check("arm2_record_out_is_arm2_matrix", bool(o2) and os.path.exists(o2)
              and os.path.samefile(o2, paths["arm2_matrix"]), arm2_record_out=o2, arm2_matrix=paths["arm2_matrix"])
    if args.variant_dir:
        sfd = rec2.get("slide_features_dir")
        check("variant_arm2_record_features_dir", bool(sfd) and os.path.isdir(sfd) and os.path.isdir(variant_sfd)
              and os.path.samefile(sfd, variant_sfd), arm2_record_slide_features_dir=sfd, expected=variant_sfd)

    common = [l.strip() for l in open(paths["common_queries"]) if l.strip()]
    order = list(meta.index)
    lib1 = [s for s in order if s in set(M1.index) and meta.at[s, "label"] not in DROP]
    lib2 = [s for s in order if s in set(M2.index) and meta.at[s, "label"] not in DROP]
    qc = [s for s in order if s in set(common)]
    full_pop = "common_pool_2564"
    if args.pool_matrices:
        named, seen = [], []
        for role, p, M in [("arm1_matrix", paths["arm1_matrix"], M1), ("arm2_matrix", paths["arm2_matrix"], M2)] + [
                ("pool_matrix", p, None) for p in args.pool_matrices]:
            if any(same_path(p, q) for q in seen):
                continue
            seen.append(p)
            ix = M.index if M is not None else pd.read_pickle(p).index
            named.append((role, p, os.path.basename(p), list(map(str, ix))))
        qp, dropped = intersection_pool(qc, [(b, ix) for _, _, b, ix in named])
        dropped_df = pool_dropped_table(dropped, meta)
        full_pop = f"common_pool_{len(qp)}_intersection"
        rec["pool"] = {
            "label": full_pop, "n_common": len(common), "n_pool": len(qp), "n_dropped": len(dropped_df),
            "n_lib1_descriptive": len(lib1), "n_lib2_descriptive": len(lib2),
            "matrices": [{"role": r, "path": p, "sha256": rec["inputs"][r]["sha256"] if r != "pool_matrix" else sha256(p),
                          "n_index": len(ix)} for r, p, _, ix in named],
            "dropped": dropped_df.to_dict("records"), "dropped_slides": dropped_df["slide"].tolist()}
        check("population_sizes_pool", set(qp) <= set(common) and set(dropped_df["slide"]) <= set(common)
              and len(qp) == len(common) - len(dropped_df) and len(qp) > 0, n_common_queries=len(common),
              n_pool=len(qp), n_dropped=len(dropped_df), n_lib1=len(lib1), n_lib2=len(lib2),
              dropped_slides=dropped_df["slide"].tolist())
    else:
        pool = [s for s in lib2 if s in set(M1.index)]
        check("population_sizes", len(common) == 2564 and len(lib1) == 2604 and len(lib2) == 2564
              and set(pool) == set(common), n_common_queries=len(common), n_lib1=len(lib1), n_lib2=len(lib2),
              n_pool=len(pool), lib1_minus_lib2=len(set(lib1) - set(lib2)))
        qp = qc
    if args.expect_pool:
        exp = [l.strip() for l in open(args.expect_pool) if l.strip()]
        check("pool_matches_expected", qp == exp, expect_pool=args.expect_pool, n_expected=len(exp), n_pool=len(qp),
              same_set=set(qp) == set(exp), same_order=qp == exp, only_in_pool=sorted(set(qp) - set(exp))[:50],
              only_in_expected=sorted(set(exp) - set(qp))[:50])

    old = {k: load_ranked(paths[k]) for k in ("legacy_arm1", "legacy_arm2", "arm1_reference")}
    for k, df in old.items():
        rec["inputs"][k]["rows"] = len(df)
    pairs = 0
    for name, M, qs, lib, key in (("a_arm2_legacy", M2, qc, lib2, "legacy_arm2"),
                                  ("b_arm1_legacy", M1, qc, lib1, "legacy_arm1"),
                                  ("b_arm1_reference_2604", M1, lib1, lib1, "arm1_reference")):
        if f"reconcile_{name}" in skips:
            skipped(f"reconcile_{name}", skips[f"reconcile_{name}"])
            continue
        new, diag = rank(M, qs, lib, meta)
        res = compare(new, old[key], M, lib)
        pairs += same_patient_guard(new, meta) + same_patient_guard(old[key], meta)
        check(f"reconcile_{name}", res.pop("passed"), **res, **diag)
    if not all(checks[c]["passed"] for c in checks):
        print("reconciliation failed; see reconciliation.json", file=sys.stderr)
        return finish(1)

    # c. legacy point estimates from the saved lists (legacy Δ sign: arm1 − arm2)
    lq1, lq2 = per_query(old["legacy_arm1"]), per_query(old["legacy_arm2"])
    cls = sorted(set(lq1["true_label"]))
    for k in KS:
        if f"c_legacy_macro_delta_hit@{k}" in skips:
            skipped(f"c_legacy_macro_delta_hit@{k}", skips[f"c_legacy_macro_delta_hit@{k}"])
            continue
        leg = json.load(open(paths[f"legacy_paired_k{k}"]))["comparisons"]["titan_titan"][f"macro_hit_at_{k}"]["delta"]
        mac = [lq.groupby("true_label")[f"hit@{k}"].mean().loc[cls].mean() for lq in (lq1, lq2)]
        mic = [lq[f"hit@{k}"].mean() for lq in (lq1, lq2)]
        check(f"c_legacy_macro_delta_hit@{k}", abs((mac[0] - mac[1]) - leg) <= 1e-9, legacy_json_arm1_minus_arm2=leg,
              recomputed_macro_arm1_minus_arm2=mac[0] - mac[1], micro_arm1=mic[0], micro_arm2=mic[1],
              micro_arm2_minus_arm1=mic[1] - mic[0], n_classes=len(cls))

    # e-g. common pool re-rank
    rl1, d1 = rank(M1, qp, qp, meta)
    rl2, d2 = rank(M2, qp, qp, meta)
    pairs += same_patient_guard(rl1, meta) + same_patient_guard(rl2, meta)
    check("d_same_patient_guard", True, pairs_checked=pairs)
    rec["common_pool_rank_diagnostics"] = {"arm1": d1, "arm2": d2}
    gz = {"method": "gzip", "mtime": 0}
    rl1.to_csv(os.path.join(outdir, "ranked_lists_common_pool_arm1.csv.gz"), index=False, compression=gz)
    rl2.to_csv(os.path.join(outdir, "ranked_lists_common_pool_arm2.csv.gz"), index=False, compression=gz)
    pq1, pq2 = per_query(rl1), per_query(rl2)
    pq = pq1[["query", "true_label"]].copy()
    pq["patient"] = meta.loc[pq["query"], "patient"].to_numpy()
    for arm, t in (("arm1", pq1), ("arm2", pq2)):
        for c in t.columns[2:]:
            pq[f"{arm}_{c}"] = t[c].to_numpy()
    pq.to_csv(os.path.join(outdir, "per_query_common_pool.csv"), index=False)
    draws = make_patient_draws(meta.loc[qp, "patient"], count=B, seed=SEED)
    s_cp, c_cp = paired_summary(pq1, pq2, meta, full_pop, draws)
    lq1 = lq1.set_index("query").loc[pq1["query"]].reset_index()
    lq2 = lq2.set_index("query").loc[pq1["query"]].reset_index()
    s_lg, _ = paired_summary(lq1, lq2, meta, "legacy_population_lib1_2604_lib2_2564", draws)

    # 1. arm-1 identical-embedding clique: tie diagnostics and exclusion sensitivity
    clique, cinfo = identical_clique(M1)
    if "arm1_matrix" in overrides:
        check("arm1_override_no_zero_distance_slides", cinfo["n_slides"] == 0, n_slides=cinfo["n_slides"],
              n_zero_pairs=cinfo["n_zero_pairs"], reason="the clique handling describes the published arm-1 failure")
        if cinfo["n_slides"]:
            return finish(1)
    in_pool = set(qp)
    ties = {}
    for arm, M in (("arm1", M1), ("arm2", M2)):
        d, hmin, hmax = tie_diagnostics(M, qp, qp, meta)
        ties[arm], ties[arm + "_h"] = d, (hmin, hmax)
    dmin = ties["arm2_h"][0].mean() - ties["arm1_h"][1].mean()
    dmax = ties["arm2_h"][1].mean() - ties["arm1_h"][0].mean()
    rec["tie_diagnostics_common_pool"] = {
        "arm1": ties["arm1"], "arm2": ties["arm2"], "tie_order_used": "metadata row order (matches legacy)",
        "micro_hit@5_delta_min_over_tie_orders": float(dmin), "micro_hit@5_delta_max_over_tie_orders": float(dmax)}
    check("arm1_identical_embedding_clique", cinfo["rows_identical_outside_clique"],
          **cinfo, n_in_pool=sum(s in in_pool for s in clique), n_patients=int(meta.loc[clique, "patient"].nunique()))
    cl = meta.loc[clique].rename(columns={"label": "Tumor.group", "patient": "UID.Subject", "sample": "Sample.ID"})
    cl.index.name = "slide"
    cl["in_pool"] = [s in in_pool for s in clique]
    for arm, t in (("arm1", pq1), ("arm2", pq2)):
        t2 = t.set_index("query")
        for k in (1, 5):
            cl[f"{arm}_common_pool_hit@{k}"] = [t2.at[s, f"hit@{k}"] if s in in_pool else np.nan for s in clique]
    cl.reset_index().to_csv(os.path.join(outdir, "arm1_identical_embedding_clique.csv"), index=False)
    qx = [s for s in qp if s not in set(clique)]
    px1, px2 = per_query(rank(M1, qx, qx, meta)[0]), per_query(rank(M2, qx, qx, meta)[0])
    draws_x = make_patient_draws(meta.loc[qx, "patient"], count=B, seed=SEED)
    s_x, c_x = paired_summary(px1, px2, meta, "common_pool_excl_clique", draws_x)
    rec["n_queries_common_pool_excl_clique"], rec["n_patients_common_pool_excl_clique"] = len(qx), len(draws_x[0])
    summary = pd.concat([s_cp, s_lg, s_x], ignore_index=True)
    if custom:
        m1, evidence1 = arm1_metric_evidence(paths["arm2_record"], rec["inputs"]["arm1_matrix"]["sha256"],
                                             rec["inputs"]["arm1_matrix"]["metric"])
        rec["arm_metrics"] = {"arm1": m1, "arm1_evidence": evidence1, "arm2": rec2["metric"]}
        r1 = json.load(open(rec1[0])) if rec1 else None
        legacy_note = LEGACY_ARMS_NOTE if args.variant_dir or "arm1_matrix" in overrides else LEGACY_NOTE
        rec["arm_metrics"].update(
            arm1_matrix=paths["arm1_matrix"], arm1_record=rec1 or None,
            arm1_tumor_only=r1.get("tumor_only") if r1 else None,
            arm2_matrix=paths["arm2_matrix"], arm2_slide_features_dir=rec2.get("slide_features_dir"),
            arm1_words=arm_words(r1, "arm1_matrix" in overrides, "published all-tissue TITAN", paths["arm1_matrix"]),
            arm2_words=arm_words(rec2, "arm2_matrix" in overrides, S2_WORDS, paths["arm2_matrix"], args.variant_dir),
            legacy_note=legacy_note)
        summary = annotate_metrics(summary, m1, rec2["metric"], legacy_note)
    summary.to_csv(os.path.join(outdir, "summary_common_pool.csv"), index=False)
    pd.concat([c_cp, c_x], ignore_index=True).to_csv(os.path.join(outdir, "per_class_common_pool.csv"), index=False)
    rec["n_patients_common_pool"] = len(draws[0])
    if args.pool_matrices:
        with open(os.path.join(outdir, "pool_queries.txt"), "w") as f:
            f.write("".join(s + "\n" for s in qp))
        dropped_df.to_csv(os.path.join(outdir, "pool_dropped.csv"), index=False)

    # i. connectivity
    summ = pd.read_csv(paths["patch_summary"], dtype={"slide": str})
    excl = pd.read_csv(paths["excluded"], dtype={"slide": str})
    conn, grp, ch = connectivity(h5dir, summ, excl, meta)
    rec["inputs"]["h5_coords"] = {"dir": h5dir, "n_files": len(conn), "sha256_of_slide_and_coords_bytes": ch}
    rec["inputs"]["patch_summary"]["rows"], rec["inputs"]["excluded"]["rows"] = len(summ), len(excl)
    if args.variant_dir:
        infeasible = read_infeasible(paths.get("titan_infeasible"))
        rec["variant"].update(n_infeasible=len(infeasible), infeasible_slides=infeasible,
                              mean_tiles_scored_available="mean_tiles_scored_per_patch" in summ)
        conn["titan_infeasible"] = conn["slide"].isin(infeasible)
        ok, info = h5_vs_matrix(conn["slide"], M2.index, infeasible)
        check("i_h5_slides_equal_arm2_matrix", ok, **info)
    else:
        check("i_h5_slides_equal_arm2_matrix", set(conn["slide"]) == set(M2.index), n_h5=len(conn))
    check("i_n_kept_matches_summary", (conn["n_kept_h5"] == conn["n_kept"]).all(),
          n_mismatch=int((conn["n_kept_h5"] != conn["n_kept"]).sum()), n_missing_summary=int(conn["n_kept"].isna().sum()))
    check("i_coords_on_1024_grid", conn["n_off_grid"].sum() == 0 and conn["n_duplicate_coords"].sum() == 0,
          off_grid_patches=int(conn["n_off_grid"].sum()), duplicate_coords=int(conn["n_duplicate_coords"].sum()))
    check("i_kept_plus_excluded_cover_metadata", set(conn["slide"]).isdisjoint(excl["slide"])
          and set(conn["slide"]) | set(excl["slide"]) == set(meta.index), n_excluded=len(excl))
    conn.to_csv(os.path.join(outdir, "connectivity_per_slide.csv"), index=False)
    grp.to_csv(os.path.join(outdir, "connectivity_per_group.csv"), index=False)
    with open(os.path.join(outdir, "README.md"), "w") as f:
        fmt = dict(
            n_pat=len(draws[0]), B=B, seed=SEED, n_cl=cinfo["n_slides"], n_zero=cinfo["n_zero_pairs"],
            n_cl_pool=int(cl["in_pool"].sum()), n_x=len(qx), n_pat_x=len(draws_x[0]),
            n_tie=ties["arm1"]["queries_with_ties_in_top_or_at_cutoff"],
            n_und=ties["arm1"]["queries_hit_at_5_undecided"], dmin=dmin, dmax=dmax,
            max1=rec["inputs"]["arm1_matrix"]["max_real_distance"])
        if not custom:
            f.write(README.format(**fmt))
        else:
            f.write(override_readme(fmt, summary, rec, overrides))
    ok = all(c["passed"] for c in checks.values())
    print(summary.to_string())
    return finish(0 if ok else 1)


README = """# R2.6 TITAN common-pool comparison and retained-patch connectivity (2026-10-03)

Produced by `scripts/r2_6_titan_common_pool.py`; provenance, input hashes, checks and
counts are in `reconciliation.json`. Descriptive outputs only.

- `ranked_lists_common_pool_arm1.csv.gz` / `_arm2.csv.gz`: depth-10 neighbours for each of the
  2,564 common-pool queries against the same common pool minus the query (self and same-patient
  pairs excluded). Arm 1 = all-tissue TITAN matrix; arm 2 = tumour-enriched TITAN matrix.
- `per_query_common_pool.csv`: per query hit@1/3/5/10, precision@5, plurality@5 (most frequent
  label among the top 5; ties go to the tied label at the closest rank), plurality label and
  top-1 label, both arms side by side.
- `summary_common_pool.csv`: micro and macro (unweighted mean over classes) hit@k and micro
  plurality@5 per arm and paired delta (arm2 - arm1). `population=common_pool_2564` is the
  harmonised-library result; `population=legacy_population_lib1_2604_lib2_2564` recomputes the
  earlier saved lists (arm-1 library 2,604 slides, arm-2 library 2,564) with the same bootstrap;
  `population=common_pool_excl_clique` is the common pool with the arm-1 identical-embedding
  slides removed as both queries and library and re-ranked ({n_x} queries, {n_pat_x} patients,
  its own patient draws with the same B/seed). Columns `n_zero_support_draws`, `ci_available`
  and `ci_degenerate` (lo == hi) qualify each interval.
- `per_class_common_pool.csv`: per-class hit@5 and plurality@5 recall per arm and delta, for
  `common_pool_2564` and `common_pool_excl_clique`.
- `arm1_identical_embedding_clique.csv`: the {n_cl} slides whose arm-1 TITAN embeddings are
  identical (distance 0 between every pair; {n_zero} unmasked pairs plus one same-patient masked
  pair), with Tumor.group, UID.Subject, Sample.ID, common-pool membership and per-arm common-pool
  hit@1/hit@5 ({n_cl_pool} are in the pool).
- `connectivity_per_slide.csv`: per tumour-enriched slide, retained 1024-px patches, 4-connected
  components on the 1024-px lattice, largest component, singletons, and summary fields.
  `mean_tiles_scored_per_patch` is computed over ALL patches of the slide before tumour selection
  (not over retained patches); `unscored_tile_fraction_all_patches` = 1 - mean_tiles_scored_per_patch/4
  is therefore the fraction of tile positions (4 per 1024-px patch) without a tile score across
  all of the slide's patches before selection.
- `connectivity_per_group.csv`: section `kept` = per Tumor.group medians/IQR and counts of slides
  with largest component >= 8/32/128 patches (plus ALL); section `excluded` = the excluded slides by
  reason category and Tumor.group.

Notes:
- The unit of analysis is the slide query: 2,564 queries from {n_pat} patients; queries from the
  same patient are not independent.
- Intervals are whole-patient percentile bootstrap intervals (B={B}, seed={seed}, PCG64); the same
  patient draws are used for both arms and the delta is computed within each draw (common random
  numbers). No draw is dropped: if any draw has a class with zero support (macro or per-class
  statistic undefined), the interval is reported as unavailable (lo/hi NaN, `ci_available` False)
  and the number of such draws is given in `n_zero_support_draws`.
- Delta sign: this script reports arm2 - arm1 (tumour-enriched minus all-tissue). The legacy S2
  paired JSONs (`s2_paired/paired_vs_titan_titan_tumor_only_k*.json`) report arm1 - arm2.
- Arm-1 metric: the arm-1 matrix has no record file; its metric is unrecorded and not cosine
  (maximum real distance {max1:.1f}, above the cosine-distance bound of 2). Masked cells hold 1e6.
- Arm-1 identical embeddings: {n_cl} slides have identical arm-1 embeddings ({n_zero} zero-distance
  pairs). In the common pool, {n_tie} arm-1 queries have exact-distance ties inside the top 10 or at
  the cut-off; their rankings depend on tie order, which here is metadata row order (matching the
  legacy lists). Across all valid tie orders, hit@5 is undecided for {n_und} arm-1 queries and the
  common-pool micro hit@5 delta ranges from {dmin:+.4f} to {dmax:+.4f}. The
  `common_pool_excl_clique` population is provided as a sensitivity population.
- "Tumour-enriched", not "tumour-only": retained patches can contain mixed tissue.
- Patch selection used the historical tile-random UNI2 tumour classifier, whose training was not
  patient-independent.
- The 1024-px lattice spacing is specific to this cohort's 40x patching.
"""


def override_readme(fmt, summary, rec, overrides):
    """README for runs with overridden matrices, an intersection pool or a variant (the default-run
    README above stays unchanged)."""
    am = rec.get("arm_metrics", {})
    m1, m2 = am.get("arm1", "unrecorded"), am.get("arm2", "unrecorded")
    pool, var = rec.get("pool"), rec.get("variant")
    pops = list(dict.fromkeys(summary["population"]))
    fl = next(p for p in pops if p.startswith("common_pool_") and p != "common_pool_excl_clique")
    d = summary[(summary["metric"] == "micro_hit@5") & (summary["arm"] == "delta")].set_index("population")
    full, excl = d.loc[fl], d.loc["common_pool_excl_clique"]
    ci = lambda r: f"{r['estimate']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
    ev = am.get("arm1_evidence")
    s2_arm2 = var is None and "s2_titan_tumor_only" in str(am.get("arm2_slide_features_dir") or "s2_titan_tumor_only")
    if m1 == m2 == "cityblock" and "arm2_matrix" in overrides and ev and s2_arm2:
        metric = ("Both arms use L1 (cityblock) distance. Arm 1 (published all-tissue TITAN matrix) is confirmed "
                  f"cityblock by a bitwise recompute from the published slide embeddings: {ev}. "
                  "Arm 2 (tumour-enriched) was recomputed from the saved S2 slide embeddings with cityblock "
                  f"(record `{overrides.get('arm2_record', '')}`); it differs from the 2026-10-03 run only by metric "
                  "(that run's arm 2 was cosine).")
    elif "arm1_matrix" in am or pool or var:
        recs = am.get("arm1_record") or []
        src1 = (f"bitwise cityblock recompute: {ev}" if ev else
                "per record " + ", ".join(f"`{r}`" for r in recs) if recs else "no matching record; unverified")
        sfd = am.get("arm2_slide_features_dir")
        metric = (f"Arm 1 = `{am.get('arm1_matrix', overrides.get('arm1_matrix', 'default arm-1 matrix'))}`, metric "
                  f"{m1} ({src1}). Arm 2 = `{am.get('arm2_matrix', overrides.get('arm2_matrix', 'default arm-2 matrix'))}`"
                  f", metric {m2} per its record `{overrides.get('arm2_record', 'default arm-2 record')}`"
                  + (f" (slide embeddings `{sfd}`)" if sfd else "") + ".")
    else:
        metric = f"Arm-1 metric: {m1}. Arm-2 metric (per its record): {m2}."
    shown = dict(overrides)
    if pool:
        shown["pool_matrices"] = " ".join(m["path"] for m in pool["matrices"] if m["role"] == "pool_matrix")
    if var:
        shown["variant_dir"] = var["dir"]
    if fmt["n_cl"] and "arm1_matrix" in overrides:
        raise CheckError(f"overridden arm-1 matrix has {fmt['n_cl']} zero-distance slides; the clique README text "
                         "describes the published arm-1 failure and does not apply - inspect before reporting")
    if fmt["n_cl"]:
        headline = f"""**Headline population: `common_pool_excl_clique`** ({fmt['n_x']} queries, {fmt['n_pat_x']} patients), until the
{fmt['n_cl']} failed arm-1 embeddings are regenerated: micro hit@5 delta (arm2 - arm1) {ci(excl)}.
The full common pool (`{fl}`) is **artefact-affected
(arm-1 zero vectors; ~{full['estimate'] - excl['estimate']:+.4f} of the {full['estimate']:+.4f} comes from them)**: micro hit@5 delta {ci(full)}; its point estimate and interval also
depend on the arm-1 tie order ({fmt['n_tie']} arm-1 queries with exact ties; delta range over valid tie orders
{fmt['dmin']:+.4f} to {fmt['dmax']:+.4f}). The ~ figure is the difference between the full-pool and
clique-excluded estimates (the populations differ in both queries and library)."""
        failed = f"""

**Failed arm-1 embeddings.** The {fmt['n_cl']} arm-1 slides listed in `arm1_identical_embedding_clique.csv` have
all-zero (failed) published TITAN slide embeddings, not genuinely identical embeddings ({fmt['n_zero']} zero-distance
unmasked pairs; {fmt['n_cl_pool']} in the pool). The file keeps its historical name (frozen); read "identical" as
"all-zero / indistinguishable"."""
        summ_pops = f"`common_pool_excl_clique` (headline), `{fl}` (artefact-affected) and `{LEGACY_POP}`"
        clique_file = "the failed all-zero arm-1 slides (see above)"
    else:
        headline = f"""**Headline population: `{fl}`** ({full.get('n_queries', fmt['n_x'])} queries, {fmt['n_pat']} patients): micro hit@5 delta
(arm2 - arm1) {ci(full)}. Arm 1 has no zero-distance off-diagonal pairs, so `common_pool_excl_clique` is the same
population (reported for completeness). Exact-distance ties: {fmt['n_tie']} arm-1 queries have ties inside the top 10
or at the cut-off; over all valid tie orders in both arms the micro hit@5 delta ranges from {fmt['dmin']:+.4f} to
{fmt['dmax']:+.4f}."""
        failed = ""
        summ_pops = f"`{fl}` (headline), `common_pool_excl_clique` (identical here) and `{LEGACY_POP}`"
        clique_file = "empty (no zero-distance arm-1 pairs); kept for schema continuity"
    extra = ""
    if pool:
        mats = ", ".join(f"`{os.path.basename(m['path'])}`" for m in pool["matrices"])
        drop = "; ".join(f"{r['slide']} ({r['label']})" for r in pool["dropped"]) or "none"
        extra += f"""

**Pool.** Intersection pool `{fl}`: the {pool['n_common']} common queries restricted to slides present in every
listed matrix ({mats}); {pool['n_pool']} queries remain and serve as both queries and library in both arms (same pool
for every run given the same matrix list). Dropped ({pool['n_dropped']}): {drop}; see `pool_dropped.csv` and
`pool_queries.txt`. The `{LEGACY_POP}` rows use the saved legacy ranked lists for the pool queries only (their
neighbours come from the legacy libraries, which may include dropped slides)."""
    if var:
        miss = ("" if var.get("mean_tiles_scored_available", True) else
                " The variant patch summary has no `mean_tiles_scored_per_patch`, so"
                " `unscored_tile_fraction_all_patches` is empty.")
        extra += f"""

**Variant.** Arm 2 is the stage-2 variant `{var['dir']}` (check `variant_arm2_record_features_dir`: the arm-2 record's
slide embeddings are `{var['slide_features_dir']}`). Connectivity uses that variant's CONCH-subset coords
(`{var['h5dir']}`), `_tumour_patch_summary.csv` and `_excluded_slides.csv`. {var.get('n_infeasible', 0)} slide(s) in
`_titan_infeasible_slides.csv` have retained patches but no TITAN embedding; they appear in the connectivity files
(`titan_infeasible` True) but not in the arm-2 matrix.{miss}"""
    conn_files = ("from the variant (see above); column definitions as in the 2026-10-03 README" if var else
                  "unchanged from the 2026-10-03 run (do not depend on\n  the matrices); see that run's README for column definitions")
    pool_files = ("\n- `pool_queries.txt` / `pool_dropped.csv`: the intersection-pool query ids (metadata order) and the"
                  "\n  dropped common queries with label, patient and the matrices they are absent from." if pool else "")
    w1 = am.get("arm1_words", "published all-tissue TITAN")
    w2 = am.get("arm2_words", "S2 tumour-enriched TITAN")
    sign = f"arm2 - arm1 ({w2} minus {w1})"
    lnote = am.get("legacy_note", LEGACY_NOTE)
    legacy_par = (f"""

**Legacy rows.** The `{LEGACY_POP}` rows are {lnote}.""" if lnote == LEGACY_ARMS_NOTE else "")
    sel = ("patch selection used the historical tile-random UNI2 tumour classifier,\n  whose training was not "
           "patient-independent." if not var else "arm-2 patch selection is the variant's (see its `_subset_record.json`);"
           "\n  the S2 tumour-only arm used the historical tile-random UNI2 tumour classifier (training not patient-independent).")
    sk = ", ".join(f"`{k}`" for k in rec.get("checks_skipped", {})) or "none"
    return f"""# R2.6 TITAN common-pool comparison, override run ({rec['timestamp'][:10]})

Produced by `scripts/r2_6_titan_common_pool.py` with overridden inputs
({"; ".join(f"`--{k.replace('_', '-')}` = `{v}`" for k, v in shown.items())}); provenance, input hashes,
checks and counts are in `reconciliation.json`. Descriptive outputs only.

**Arms.** Arm 1 = {w1}; arm 2 = {w2}. Deltas are arm 2 minus arm 1 ({w2} minus {w1}).

**Metric.** {metric}

{headline}

**Multiplicity.** {summary['metric'].nunique()} metrics x {summary['population'].nunique()} populations are reported,
each with a 95% interval and no multiplicity adjustment; an interval that excludes 0 is not a significance claim.{failed}{extra}{legacy_par}

Files:
- `ranked_lists_common_pool_arm1.csv.gz` / `_arm2.csv.gz`: depth-10 neighbours for each common-pool query against the
  same common pool minus the query (self and same-patient pairs excluded).
- `per_query_common_pool.csv`: per query hit@1/3/5/10, precision@5, plurality@5, plurality and top-1 labels, both arms.
- `summary_common_pool.csv`: micro and macro hit@k and micro plurality@5 per arm and paired delta (arm2 - arm1) for
  {summ_pops}.
  Columns `arm1_metric`/`arm2_metric` give each row's metrics; the `{LEGACY_POP}` rows are recomputed from the saved
  legacy ranked lists (`note`: "{lnote}"). `n_zero_support_draws`, `ci_available` and `ci_degenerate`
  qualify each interval.
- `per_class_common_pool.csv`: per-class hit@5 and plurality@5 recall per arm and delta.
- `arm1_identical_embedding_clique.csv`: {clique_file}, with group, patient, sample,
  pool membership and per-arm hit@1/hit@5.
- `connectivity_per_slide.csv` / `connectivity_per_group.csv`: {conn_files}.{pool_files}

Notes:
- Unit of analysis: the slide query; queries from the same patient are not independent ({fmt['n_pat']} patients in
  the full pool). Intervals: whole-patient percentile bootstrap (B={fmt['B']}, seed={fmt['seed']}, PCG64), common
  random numbers across arms; zero-support draws make an interval unavailable, never dropped.
- Delta sign: {sign}; legacy S2 JSONs report arm1 - arm2.
- Skipped legacy reconciliations (overridden matrix): {sk}.
- "Tumour-enriched", not "tumour-only"; {sel}
"""


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckError as e:
        sys.exit(f"CHECK FAILED: {e}")
