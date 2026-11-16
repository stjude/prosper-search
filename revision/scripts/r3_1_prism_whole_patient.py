#!/usr/bin/env python3
"""
R3.1 ([decision record]): PRISM end-to-end comparator -- depth-10 ranked lists from per-slide PRISM embeddings (with
integrity gates G3/G4), a TITAN-repaired ranked list in the same schema, and the accepted whole-patient bootstrap
block adding PRISM (and TITAN-repaired) to the frozen R3.1 reference configurations.

This script does not modify the frozen r3_1_comparators_whole_patient_20261008/ bundle. It imports the accepted
functions of scripts/r3_1_comparators_whole_patient.py (loader, per-query scoring, bootstrap block), which in turn
imports scripts/r2_2_tile_ladder_whole_patient.py, unchanged.

Steps (any combination in one invocation; all outputs go to --out-dir; no step overwrites an existing output):
  --build-ranked-list        PRISM slide embeddings (<prism-dir>/<slide_id>.h5: `features` (1280,), `coords`
                             (n_tiles, 2)) -> L2-normalised -> cosine distance (1 - dot) -> depth-10 ranked list.
                             Query population = library = the 2,778 non-MISC slides (MISC are neither queries nor
                             neighbours); self and same-UID.Subject neighbours excluded; ties broken by ascending
                             integer slide id. Gates: G3 degenerate rows (non-finite / all-zero / byte-identical
                             vectors; any -> hard fail), G4 coverage (n_tiles from coords; 0 or < 50 flagged).
  --build-titan-ranked-list  accepted repaired TITAN L1 matrix (R2.6-iv; sha256 asserted; sentinel 1e6 on self and
                             same-patient entries) -> same population, mask, depth, tie-break and schema.
  --block                    whole-patient bootstrap ([decision record]; B = 5,000, PCG64 seed 123) over the five frozen
                             reference configurations (primary, median, yottixel, sish, resnet50) plus prism_<tag>
                             and titan_repaired; reference points/intervals must reproduce the frozen bundle to 1e-12.
Paired differences (minuend - subtrahend; same queries, same draws):
  prism_<tag> - {primary, median, yottixel, resnet50, sish, titan_repaired}; titan_repaired - primary (descriptive).
Optional extra PRISM arms (--block only; --extra-prism TAG=PATH, repeatable): each already-built PRISM ranked list
  is added as configuration prism_<TAG> after the configurations above (same 2,778 queries, same draws; its query
  set must equal the main arm's and the frozen references', 2,778 x depth 10, else hard fail), with the further
  pairs prism_<tag> - prism_<TAG> and prism_<TAG> - primary (descriptive). Without extra arms the block is unchanged.
Intervals are nominal 95 %, unadjusted. An interval including 0 is not evidence of equivalence. Retrieval
performance is not diagnostic accuracy.
"""
import argparse
import datetime as dt
import importlib.util
import io
import json
import os
import platform
import socket
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
OUT = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
BASE = os.path.join(HERE, "r3_1_comparators_whole_patient.py")
FROZEN = os.path.join(OUT, "r3_1_comparators_whole_patient_20261008")
PRISM_ROOT = os.path.join(OUT, "r3_1_prism_20261018")
PRISM_DIR_DEFAULT = os.path.join(PRISM_ROOT, "trident_native", "20x_224px_0px_overlap", "slide_features_prism")
TITAN_MATRIX_DEFAULT = os.path.join(OUT, "r2_6_titan_arm1_regen_20261004",
                                    "combined_similarity_matrix_titan_titan_repaired_no_same_patient.pkl")
TITAN_MATRIX_SHA256 = "f645d32f6414f03c99fa863a73c10d6ba6c767344268ee0120d843545b0010a0"
TITAN_SENTINEL = 1e6

spec = importlib.util.spec_from_file_location("r31base", BASE)
B_ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B_)
L = B_.L

DEPTH = B_.DEPTH
HEADER = ["query", "true_label", "rank", "neighbour", "neighbour_label", "distance"]
PRISM_DIM = 1280
G3_ZERO_NORM = 1e-8
G4_MIN_TILES = 50
FROZEN_B, FROZEN_SEED = 5000, 123
REFERENCE = list(B_.MAIN)  # primary, median, yottixel, sish, resnet50 (SISH kept: [decision record] pre-registers it)
CHECK_METRICS = ["hit1", "hit3", "hit5", "hit10", "precision5", "precision10", "plurality5"]
BOOTSTRAP = "ADR0021 whole-patient: patients with replacement, identical draws for all configurations in the block"
SAME_PATIENT_MASK = ("self and every neighbour sharing UID.Subject with the query are excluded before ranking; "
                     "MISC slides are neither queries nor library members (library = query population = 2,778)")
TIE_BREAK = "distance ascending, then ascending integer slide id (stable sort over id-ordered library columns)"
PRISM_DESC = {"native": "PRISM (Virchow v1 → PRISM, native 20×/224 dense tiling)",
              "mosaic": "PRISM applied to the published 512-px mosaic (Virchow v1 → PRISM)"}
TITAN_DESC = "Conchv1.5 + TITAN (repaired matrix, R2.6-iv)"


class GateFailure(Exception):
    pass


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def sha_file(path):
    return L.sha256_bytes(L.read_bytes(path))


def rel(path):
    p = os.path.abspath(path)
    for root in (R, OUT):
        if p.startswith(os.path.abspath(root) + os.sep):
            return os.path.relpath(p, root)
    return p


def id_key(ids):
    ids = list(ids)
    if all(s.isdigit() for s in ids):
        return sorted(ids, key=int)
    raise ValueError("slide ids are not all integer strings; ascending-id tie-break is undefined")


def library(meta):
    lib = meta[meta["Tumor.group"].ne("MISC")]
    ids = id_key(lib.index)
    if len(ids) != L.MAIN_POPULATION_QUERIES:
        raise ValueError(f"non-MISC library has {len(ids)} slides != {L.MAIN_POPULATION_QUERIES}")
    if set(meta.loc[ids, "Tumor.group"]) != set(B_.CLASSES15):
        raise ValueError("non-MISC library label set differs from the 15 accepted classes")
    return ids


def eligibility_mask(ids, meta):
    pat = meta.loc[ids, "UID.Subject"].to_numpy()
    return pat[:, None] == pat[None, :]  # includes the diagonal (self)


def rank_matrix(D, ids, meta):
    """D: (n, n) float64 distances over `ids` (integer-id order), masked entries = +inf. Returns ranked frame + tie stats."""
    n = len(ids)
    order = np.argsort(D, axis=1, kind="stable")  # columns are in ascending-id order -> ascending-id tie-break
    rows = np.arange(n)[:, None]
    top = order[:, :DEPTH]
    dtop = D[rows, top]
    if not np.isfinite(dtop).all():
        bad = [ids[i] for i in np.where(~np.isfinite(dtop).all(1))[0]]
        raise GateFailure(f"{len(bad)} queries have fewer than {DEPTH} eligible neighbours, e.g. {bad[:5]}")
    d10 = dtop[:, -1]
    rest = D[rows, order[:, DEPTH:]]
    tied_beyond = (rest == d10[:, None]).sum(1)
    within = (np.diff(dtop, axis=1) == 0).any(1)
    ties = dict(
        boundary_tie_queries=int((tied_beyond > 0).sum()),
        boundary_tie_extra_candidates=int(tied_beyond.sum()),
        boundary_tie_query_ids=[ids[i] for i in np.where(tied_beyond > 0)[0]][:200],
        within_top10_tie_queries=int(within.sum()),
        within_top10_tie_query_ids=[ids[i] for i in np.where(within)[0]][:200],
        definition=("boundary tie: an eligible candidate beyond rank 10 has exactly the rank-10 distance (membership "
                    "of the top 10 then depends on the tie-break); within-top-10 tie: two adjacent ranks 1..10 have "
                    "exactly equal distances (order depends on the tie-break)"),
    )
    lab = meta.loc[ids, "Tumor.group"].to_numpy()
    ida = np.asarray(ids, dtype=object)
    frame = pd.DataFrame({
        "query": np.repeat(ida, DEPTH), "true_label": np.repeat(lab, DEPTH),
        "rank": np.tile(np.arange(1, DEPTH + 1), n), "neighbour": ida[top.ravel()],
        "neighbour_label": lab[top.ravel()], "distance": dtop.ravel().astype(np.float64),
    })[HEADER]
    return frame, ties


def write_ranked(frame, path, meta):
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    # validate with the accepted loader (schema, depth, ranks 1..10, metadata labels, no self / same-patient,
    # neighbours inside the query set, 15-class label set, non-decreasing distances via per_query_simple)
    r, sha, npad = B_.load_ranks_padded(path, meta, B_.CLASSES15)
    if npad:
        raise GateFailure(f"{path}: unexpected padding")
    if r["query"].nunique() != L.MAIN_POPULATION_QUERIES or (r["true_label"] == "MISC").any():
        raise GateFailure(f"{path}: population is not the 2,778 non-MISC queries")
    f = B_.per_query_simple(r, meta)
    if len(f) != L.MAIN_POPULATION_QUERIES:
        raise GateFailure(f"{path}: {len(f)} scored queries")
    return sha


def summ(v):
    v = np.asarray(v, float)
    return dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()), n=int(v.size))


# ----------------------------------------------------------------------------------------------- Part A: PRISM
def prism_outputs(out_dir, tag):
    return {k: os.path.join(out_dir, v.format(tag=tag)) for k, v in dict(
        ranked="ranked_lists_prism_{tag}.csv.gz", gates="gates_prism_{tag}.json",
        tiles="tile_counts_prism_{tag}.csv", manifest="prism_{tag}_h5_manifest.csv",
        emb="prism_{tag}_embeddings_l2.npy", ids="prism_{tag}_ids.txt").items()}


def build_prism(a, meta, meta_sha):
    import h5py  # lazy: only this step needs it

    o = prism_outputs(a.out_dir, a.tag)
    lib_ids = library(meta)
    files = sorted(os.listdir(a.prism_dir))
    h5 = {fn[:-3]: os.path.join(a.prism_dir, fn) for fn in files if fn.endswith(".h5")}
    other = [fn for fn in files if not fn.endswith(".h5")]
    extras = sorted(set(h5) - set(meta.index))
    missing = id_key(set(meta.index) - set(h5))
    missing_lib = sorted(set(missing) & set(lib_ids), key=int)
    gates = dict(step="build_prism_ranked_list", tag=a.tag, generated=now(), prism_dir=a.prism_dir,
                 metadata=a.meta, metadata_sha256=meta_sha, n_h5=len(h5), n_metadata=len(meta),
                 non_h5_files_ignored=other, extra_ids_not_in_metadata=extras, missing_ids=missing,
                 missing_library_ids=missing_lib, allow_missing=bool(a.allow_missing), library_n=len(lib_ids))

    def fail(msg, code_key):
        gates["status"] = "FAIL"
        gates["failure"] = dict(gate=code_key, message=msg)
        json.dump(gates, open(o["gates"], "w"), indent=2, default=str)
        raise GateFailure(f"{code_key}: {msg} (see {o['gates']})")

    if extras:
        fail(f"{len(extras)} h5 files for ids not in the metadata, e.g. {extras[:10]}", "completeness_extras")
    if missing and not a.allow_missing:
        fail(f"{len(missing)} of {len(meta)} metadata slides lack an h5 (non-MISC: {len(missing_lib)}), e.g. "
             f"{missing[:10]}; pass --allow-missing only for MISC-only gaps", "completeness_missing")
    if missing_lib:
        fail(f"{len(missing_lib)} non-MISC library slides lack an h5, e.g. {missing_lib[:10]}; the population "
             "cannot change silently", "completeness_missing_library")

    ids = id_key(h5)
    X = np.empty((len(ids), a.dim), dtype=np.float32)
    man, tiles, shapes = [], [], set()
    for i, sid in enumerate(ids):
        data = L.read_bytes(h5[sid])
        with h5py.File(io.BytesIO(data), "r") as h:
            feat = np.asarray(h["features"][()])
            shapes.add(tuple(feat.shape))
            if feat.size != a.dim or feat.squeeze().ndim != 1:
                fail(f"{sid}: features shape {feat.shape} is not ({a.dim},)", "schema")
            X[i] = feat.reshape(-1).astype(np.float32)
            if "coords" not in h:
                fail(f"{sid}: no `coords` dataset (needed for G4)", "schema")
            c = h["coords"]
            nt = int(c.shape[0]) if len(c.shape) == 2 and c.shape[1] == 2 else (0 if c.size == 0 else -1)
            if nt < 0:
                fail(f"{sid}: coords shape {c.shape} is not (n, 2)", "schema")
            feat_dtype = str(h["features"].dtype)
        man.append(dict(slide_id=sid, file=os.path.basename(h5[sid]), size_bytes=len(data),
                        sha256=L.sha256_bytes(data), features_dtype=feat_dtype))
        tiles.append(dict(slide_id=sid, n_tiles=nt, tumor_group=meta.at[sid, "Tumor.group"],
                          patient=meta.at[sid, "UID.Subject"]))
    pd.DataFrame(man).to_csv(o["manifest"], index=False)
    tc = pd.DataFrame(tiles)
    tc.to_csv(o["tiles"], index=False)
    gates["features_shapes_seen"] = sorted(shapes)
    gates["h5_manifest"] = dict(path=o["manifest"], sha256=sha_file(o["manifest"]))
    gates["tile_counts"] = dict(path=o["tiles"], sha256=sha_file(o["tiles"]))

    # G3 degenerate rows (over every loaded slide, MISC included)
    nonfinite = [ids[i] for i in np.where(~np.isfinite(X).all(1))[0]]
    norms = np.linalg.norm(np.where(np.isfinite(X), X, 0).astype(np.float64), axis=1)
    zero = [ids[i] for i in np.where(norms < G3_ZERO_NORM)[0]]
    groups = {}
    for i, sid in enumerate(ids):
        groups.setdefault(X[i].tobytes(), []).append(sid)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    g3 = dict(n_nonfinite=len(nonfinite), nonfinite_ids=nonfinite, n_all_zero=len(zero), all_zero_ids=zero,
              zero_norm_threshold=G3_ZERO_NORM, n_duplicate_slides=sum(len(g) for g in dup_groups),
              duplicate_groups=dup_groups, rule="any count > 0 -> hard fail; slides are never dropped silently")
    g3["pass"] = not (nonfinite or zero or dup_groups)
    gates["G3"] = g3

    # G4 coverage
    lo = tc[tc["n_tiles"] < G4_MIN_TILES]
    gates["G4"] = dict(min_tiles=G4_MIN_TILES, n_zero=int((tc["n_tiles"] == 0).sum()),
                       zero_ids=tc.loc[tc["n_tiles"] == 0, "slide_id"].tolist(), n_below_min=len(lo),
                       below_min=lo[["slide_id", "n_tiles", "tumor_group"]].to_dict("records"),
                       n_tiles_summary_all=summ(tc["n_tiles"]),
                       n_tiles_summary_library=summ(tc.set_index("slide_id").loc[lib_ids, "n_tiles"]),
                       rule="flag list, not a hard fail")
    pos = {s: i for i, s in enumerate(ids)}
    li = np.array([pos[s] for s in lib_ids])
    gates["norm_before_l2"] = dict(all=summ(norms[np.isfinite(norms)]), library=summ(norms[li]))
    if not g3["pass"]:
        fail(f"G3 degenerate PRISM embeddings: non-finite {len(nonfinite)} {nonfinite[:10]}, all-zero {len(zero)} "
             f"{zero[:10]}, duplicate groups {len(dup_groups)} {dup_groups[:5]}", "G3")

    E = X[li].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    off = S[~np.eye(len(lib_ids), dtype=bool)]
    gates["pairwise_cosine_library_offdiag"] = summ(off)
    D = 1.0 - S
    D[eligibility_mask(lib_ids, meta)] = np.inf
    frame, ties = rank_matrix(D, lib_ids, meta)
    np.save(o["emb"], E)
    open(o["ids"], "w").write("\n".join(lib_ids) + "\n")
    sha = write_ranked(frame, o["ranked"], meta)
    gates.update(status="PASS", ties=ties, tie_break=TIE_BREAK, same_patient_mask=SAME_PATIENT_MASK,
                 similarity="L2-normalised (float64) PRISM vectors; distance = 1 - dot (cosine distance)",
                 ranked_list=dict(path=o["ranked"], sha256=sha, rows=len(frame)),
                 embeddings_l2=dict(path=o["emb"], sha256=sha_file(o["emb"]), dtype="float64",
                                    shape=list(E.shape), ids=o["ids"], ids_sha256=sha_file(o["ids"])),
                 script=rel(__file__), script_sha256=sha_file(__file__),
                 h5py_version=h5py.__version__, numpy_version=np.__version__, pandas_version=pd.__version__)
    json.dump(gates, open(o["gates"], "w"), indent=2, default=str)
    print(f"[prism:{a.tag}] {len(frame)} rows -> {o['ranked']} (sha256 {sha[:12]}); G3 PASS; "
          f"G4 flags {len(lo)} (<{G4_MIN_TILES} tiles); boundary ties {ties['boundary_tie_queries']}, "
          f"within-top-10 ties {ties['within_top10_tie_queries']}")
    return gates


# ----------------------------------------------------------------------------------------------- Part A': TITAN
def titan_outputs(out_dir):
    return dict(ranked=os.path.join(out_dir, "ranked_lists_titan_repaired.csv.gz"),
                record=os.path.join(out_dir, "ranked_list_titan_repaired_record.json"))


def build_titan(a, meta, meta_sha):
    o = titan_outputs(a.out_dir)
    data = L.read_bytes(a.titan_matrix)
    sha_m = L.sha256_bytes(data)
    if sha_m != TITAN_MATRIX_SHA256:
        raise GateFailure(f"TITAN matrix sha256 {sha_m} != accepted {TITAN_MATRIX_SHA256}")
    M = pd.read_pickle(io.BytesIO(data))
    if not isinstance(M, pd.DataFrame) or M.shape != (len(meta), len(meta)):
        raise GateFailure(f"TITAN matrix is not a {len(meta)}x{len(meta)} DataFrame")
    if not M.index.equals(M.columns) or set(M.index) != set(meta.index):
        raise GateFailure("TITAN matrix index/columns differ from each other or from the metadata ids")
    full_ids = list(M.index)
    A = M.to_numpy(dtype=np.float64)
    if not np.isfinite(A).all():
        raise GateFailure("TITAN matrix has non-finite entries")
    pat = meta.loc[full_ids, "UID.Subject"].to_numpy()
    masked = pat[:, None] == pat[None, :]
    sent = A == TITAN_SENTINEL
    sp_pairs = int((masked.sum() - len(full_ids)) // 2)
    if not sent[masked].all():
        raise GateFailure(f"{int((~sent[masked]).sum())} self/same-patient entries are not the sentinel")
    if sent[~masked].any():
        raise GateFailure(f"{int(sent[~masked].sum())} eligible entries equal the sentinel")
    elig = A[~masked]
    if (elig < 0).any():
        raise GateFailure("negative eligible L1 distance")
    lib_ids = library(meta)
    D = M.loc[lib_ids, lib_ids].to_numpy(dtype=np.float64)
    lm = eligibility_mask(lib_ids, meta)
    if not np.array_equal(D == TITAN_SENTINEL, lm):
        raise GateFailure("library sentinel pattern differs from the self/same-patient mask")
    D[lm] = np.inf
    frame, ties = rank_matrix(D, lib_ids, meta)
    sha = write_ranked(frame, o["ranked"], meta)
    rec = dict(step="build_titan_ranked_list", generated=now(), matrix=a.titan_matrix, matrix_sha256=sha_m,
               matrix_sha256_expected=TITAN_MATRIX_SHA256, metadata=a.meta, metadata_sha256=meta_sha,
               metric="L1 (cityblock) on TITAN slide embeddings, as stored in the accepted repaired matrix",
               sentinel=TITAN_SENTINEL, n_sentinel_entries=int(sent.sum()), same_patient_pairs_unordered=sp_pairs,
               sentinel_check="sentinel exactly on self + same-patient entries (full 2,921 matrix); no eligible entry equals it",
               max_abs_asymmetry=float(np.abs(A - A.T).max()), eligible_min=float(elig.min()),
               eligible_max=float(elig.max()), library_n=len(lib_ids), same_patient_mask=SAME_PATIENT_MASK,
               tie_break=TIE_BREAK, ties=ties, ranked_list=dict(path=o["ranked"], sha256=sha, rows=len(frame)),
               script=rel(__file__), script_sha256=sha_file(__file__))
    json.dump(rec, open(o["record"], "w"), indent=2, default=str)
    print(f"[titan_repaired] {len(frame)} rows -> {o['ranked']} (sha256 {sha[:12]}); same-patient pairs {sp_pairs}; "
          f"boundary ties {ties['boundary_tie_queries']} (extra candidates {ties['boundary_tie_extra_candidates']}), "
          f"within-top-10 ties {ties['within_top10_tie_queries']}")
    return rec


# ----------------------------------------------------------------------------------------------- Part B: block
BLOCK_FILES = ["summary.csv", "paired.csv", "per_group.csv", "per_query.csv.gz", "record.json", "README.md",
               "sha256.txt", "COMPLETE"]


def build_record(path, kind):
    """Locate the build record written next to a ranked list (if any)."""
    d = os.path.dirname(os.path.abspath(path))
    if kind == "titan":
        p = os.path.join(d, "ranked_list_titan_repaired_record.json")
    else:
        p = os.path.join(d, os.path.basename(path).replace("ranked_lists_prism_", "gates_prism_").replace(".csv.gz", ".json"))
    return (p, json.load(open(p))) if os.path.exists(p) else (p, None)


def block(a, meta, meta_sha):
    ptag = f"prism_{a.tag}"
    prism_rl = a.prism_ranked_list or prism_outputs(a.out_dir, a.tag)["ranked"]
    new = [(ptag, a.prism_desc or PRISM_DESC.get(a.tag, f"PRISM ({a.tag})"), prism_rl, None)]
    contrasts = [(ptag, t) for t in ("primary", "median", "yottixel", "resnet50", "sish")]
    if not a.no_titan:
        titan_rl = a.titan_ranked_list or titan_outputs(a.out_dir)["ranked"]
        new.append(("titan_repaired", TITAN_DESC, titan_rl, None))
        contrasts += [(ptag, "titan_repaired"), ("titan_repaired", "primary")]
    # extra PRISM arms go after every existing configuration/contrast, so the default block is unchanged
    extra_tags = []
    for etag, epath in a.extra_prism:
        xtag = f"prism_{etag}"
        new.append((xtag, PRISM_DESC.get(etag, f"PRISM ({etag})"), epath, None))
        contrasts += [(ptag, xtag), (xtag, "primary")]
        extra_tags.append(xtag)
    configs = REFERENCE + new
    hashes, frames, padded_n, saved, builds, pop, extra_checks = {}, {}, {}, {}, {}, {}, []
    n_exp, rows_exp = L.MAIN_POPULATION_QUERIES, L.MAIN_POPULATION_QUERIES * DEPTH
    for tag, desc, path, mj in configs:
        full = os.path.join(OUT, path)
        r, sha, npad = B_.load_ranks_padded(full, meta, B_.CLASSES15)
        hashes[rel(full)] = sha
        padded_n[tag] = npad
        if extra_tags and (tag == ptag or tag in extra_tags):
            # population gate (only with extra arms): main and extra PRISM arms, before scoring; the references and
            # the main arm precede the extra arms in `configs`, so their query sets are already available here
            sizes = r.groupby("query").size()
            p = pop[tag] = dict(ranked_list=rel(full), sha256=sha, rows=int(len(r)), n_queries=int(r["query"].nunique()),
                                depth_min=int(sizes.min()), depth_max=int(sizes.max()), queries=frozenset(r["query"]))
            if (p["rows"], p["n_queries"], p["depth_min"], p["depth_max"]) != (rows_exp, n_exp, DEPTH, DEPTH):
                raise GateFailure(f"{tag}: {p['rows']} rows / {p['n_queries']} queries / depth {p['depth_min']}..{p['depth_max']}"
                                  f" != {rows_exp} / {n_exp} / {DEPTH} ({p['ranked_list']})")
            if tag in extra_tags:
                eq_main = p["queries"] == pop[ptag]["queries"]
                eq_ref = {t: p["queries"] == frozenset(frames[t].index) for t, *_ in REFERENCE}
                if not eq_main or not all(eq_ref.values()):
                    raise GateFailure(f"{tag}: query set differs from {ptag} (equal={eq_main}) or from the frozen reference "
                                      f"lists ({eq_ref}); the population cannot change silently")
                extra_checks.append(dict(config=tag, tag=tag[len("prism_"):], description=desc, ranked_list=p["ranked_list"],
                                         sha256=sha, rows=p["rows"], rows_expected=rows_exp, n_queries=p["n_queries"],
                                         n_queries_expected=n_exp, depth_min=p["depth_min"], depth_max=p["depth_max"],
                                         depth_expected=DEPTH, query_set_equals_main_arm=eq_main,
                                         query_set_equals_references=eq_ref))
        frames[tag] = B_.per_query_simple(r, meta)
        if len(frames[tag]) != L.MAIN_POPULATION_QUERIES:
            raise ValueError(f"{tag}: {len(frames[tag])} queries != {L.MAIN_POPULATION_QUERIES}")
        saved[tag] = B_.saved_points(os.path.join(OUT, mj)) if mj else None
        if (tag, desc, path, mj) in new:
            bp, br = build_record(full, "titan" if tag == "titan_repaired" else "prism")
            if br is not None and br.get("ranked_list", {}).get("sha256") != sha:
                raise GateFailure(f"{tag}: ranked list sha256 {sha} differs from its build record {bp}")
            if br is not None and tag != "titan_repaired" and br.get("status") != "PASS":
                raise GateFailure(f"{tag}: build record {bp} status is not PASS")
            builds[tag] = dict(record=bp, record_sha256=(sha_file(bp) if br else None), found=br is not None,
                               ties=(br or {}).get("ties"), G3=(br or {}).get("G3"),
                               G4=({k: v for k, v in br["G4"].items() if k != "below_min"} if br and "G4" in br else None),
                               G4_below_min=(br or {}).get("G4", {}).get("below_min") if br else None,
                               h5_manifest=(br or {}).get("h5_manifest"), embeddings_l2=(br or {}).get("embeddings_l2"),
                               norm_before_l2=(br or {}).get("norm_before_l2"),
                               pairwise_cosine_library_offdiag=(br or {}).get("pairwise_cosine_library_offdiag"),
                               missing_ids=(br or {}).get("missing_ids"), matrix_sha256=(br or {}).get("matrix_sha256"))
    for x in extra_checks:
        x.update(build_record=builds[x["config"]]["record"], build_record_found=builds[x["config"]]["found"])
    ref = frames["primary"]
    for tag, f in frames.items():
        if not f.index.equals(ref.index) or not (f["true_label"] == ref["true_label"]).all() or not (f["patient"] == ref["patient"]).all():
            raise ValueError(f"{tag}: query/label/patient frame does not align with primary")
    unexpected = {t: v for t, v in padded_n.items() if v and t != "sish"}
    if unexpected or padded_n.get("sish") != 35:
        raise ValueError(f"unexpected padding: {padded_n} (only SISH's disclosed 35 padded queries are expected)")
    block = B_.boot_block(frames, [t for t, *_ in configs], B_.CLASSES15, a.B, a.seed, contrasts, "all_tissue_2778")
    summary, paired, per_group = block["summary"], block["paired"], block["per_group"]

    # 1) reference configurations must reproduce the frozen bundle exactly (same sampler, B, seed, draws)
    frozen_sum_path = os.path.join(a.frozen, "summary.csv")
    ref_diff = None
    if a.skip_frozen_reconciliation:
        print("WARNING " + "!" * 70 + "\nWARNING: --skip-frozen-reconciliation: reference configurations were NOT "
              "checked against the frozen bundle.\nWARNING: smoke-test only; this output is not a production result.\n"
              "WARNING " + "!" * 70, file=sys.stderr)
    else:
        frozen_summary = pd.read_csv(frozen_sum_path)
        frozen_summary = frozen_summary[frozen_summary["population"].eq("all_tissue_2778")]
        ref_tags = [t for t, *_ in REFERENCE]
        key = ["config", "metric", "average"]
        m = summary[summary["config"].isin(ref_tags)].merge(frozen_summary, on=key, suffixes=("", "_frozen"))
        if len(m) != len(ref_tags) * len(CHECK_METRICS) * 2:
            raise ValueError(f"frozen reconciliation join incomplete: {len(m)} rows != {len(ref_tags) * len(CHECK_METRICS) * 2}")
        ref_diff = {c: float((m[c] - m[f"{c}_frozen"]).abs().max()) for c in ("point", "ci_lo", "ci_hi")}
        if max(ref_diff.values()) > 1e-12:
            raise ValueError(f"reference configurations do not reproduce the frozen bundle: {ref_diff}")

    # 2) saved-point reconciliation for the new configurations (none has a saved metrics JSON -> recorded as skipped)
    rec_points, skipped = [], []
    for t, _, _, mj in new:
        sp = saved.get(t)
        if not sp or sp[0] is None:
            skipped.append(t)
            continue
        pts, mode = sp
        for k, val in pts.items():
            if val is None:
                continue
            avg, metric = k.split("_", 1)
            metric = metric.replace("_at_", "")
            mine = block["points"].get((t, metric, avg))
            rec_points.append(dict(config=t, metric=metric, average=avg, saved_point=val, recomputed_point=mine,
                                   abs_diff=(None if mine is None else abs(mine - val)), saved_bootstrap_mode=mode))

    frame = pd.concat(frames, names=["config", "query"])
    record = dict(
        step="block", generated=now(), hostname=socket.gethostname(), python=platform.python_version(),
        numpy_version=np.__version__, pandas_version=pd.__version__, argv=sys.argv,
        script=rel(__file__), script_sha256=sha_file(__file__),
        imported_base_script=rel(BASE), imported_base_sha256=sha_file(BASE),
        imported_ladder_script=rel(B_.LADDER), imported_ladder_sha256=sha_file(B_.LADDER),
        metadata=a.meta, metadata_sha256=meta_sha, inputs_sha256=hashes,
        configs=[dict(config=t, description=d, ranked_list=rel(os.path.join(OUT, p)), saved_metrics=mj) for t, d, p, mj in configs],
        contrasts=[dict(minuend=x, subtrahend=y) for x, y in contrasts],
        B=a.B, seed=a.seed, bootstrap=BOOTSTRAP,
        population=dict(all_tissue_2778=dict(n_queries=block["n_queries"], n_patients=block["n_patients"], n_groups=15,
                                             library_n=L.MAIN_POPULATION_QUERIES)),
        library_n=L.MAIN_POPULATION_QUERIES, same_patient_mask=SAME_PATIENT_MASK, tie_break=TIE_BREAK,
        frozen_bundle=a.frozen, frozen_summary_sha256=sha_file(frozen_sum_path),
        reconciliation_skipped=bool(a.skip_frozen_reconciliation),
        reference_vs_frozen_max_abs_diff=ref_diff,
        saved_point_reconciliation=rec_points,
        saved_point_reconciliation_skipped=dict(configs=skipped, reason="no saved metrics JSON exists for these rows"),
        builds=builds,
        extra_prism_arms=extra_checks,
        main_prism_population_check=({k: v for k, v in pop[ptag].items() if k != "queries"} if extra_tags else None),
        padding={t: int(v) for t, v in padded_n.items()},
        sish_padding_rule="padded positions = non-hit for hit@k/precision@k; excluded from plurality@5 vote (base module)",
        intervals="nominal 95 %, unadjusted; an interval including 0 is not evidence of equivalence",
    )
    o = a.out_dir
    summary.to_csv(os.path.join(o, "summary.csv"), index=False)
    paired.to_csv(os.path.join(o, "paired.csv"), index=False)
    per_group.to_csv(os.path.join(o, "per_group.csv"), index=False)
    frame.to_csv(os.path.join(o, "per_query.csv.gz"), compression={"method": "gzip", "mtime": 0})
    json.dump(record, open(os.path.join(o, "record.json"), "w"), indent=2, default=str)

    lines = ["## Run", "", f"B = {a.B}, seed = {a.seed}; {record['generated']}"]
    if a.skip_frozen_reconciliation:
        lines.append("**FROZEN RECONCILIATION SKIPPED (smoke test; not a production result)**")
    else:
        lines.append(f"reference configurations vs frozen bundle max |diff| (point/ci_lo/ci_hi): {ref_diff}")
    if extra_checks:
        lines += ["", f"## Extra PRISM arms (main arm {ptag}; same queries, same draws)", ""]
        for x in extra_checks:
            lines.append(f"- {x['config']}: {x['description']}; {x['ranked_list']} (sha256 {x['sha256']}); "
                         f"{x['rows']} rows = {x['n_queries']} queries x depth {x['depth_min']}; query set equals "
                         f"{ptag} and all frozen references: {x['query_set_equals_main_arm'] and all(x['query_set_equals_references'].values())}; "
                         f"extra pairs: {ptag} - {x['config']}, {x['config']} - primary (descriptive)")
    lines += ["", "## hit@5 (point [95% CI], flag)", ""]
    for t, *_ in configs:
        parts = []
        for avg in ("macro", "micro"):
            h = summary[(summary.config == t) & (summary.metric == "hit5") & (summary.average == avg)].iloc[0]
            parts.append(f"{avg} {h.point:.4f} [{h.ci_lo:.4f}, {h.ci_hi:.4f}]{(' ' + h.flag) if isinstance(h.flag, str) and h.flag else ''}")
        lines.append(f"- {t}: " + "; ".join(parts))
    lines += ["", "## Paired differences hit@5 (minuend - subtrahend; CI includes 0?)", ""]
    for x, y in contrasts:
        parts = []
        for avg in ("macro", "micro"):
            row = paired[(paired.minuend == x) & (paired.subtrahend == y) & (paired.metric == "hit5") & (paired.average == avg)].iloc[0]
            parts.append(f"{avg} {row.point:+.4f} [{row.ci_lo:+.4f}, {row.ci_hi:+.4f}] incl0={row.ci_includes_zero_nominal}")
        lines.append(f"- {x} - {y}: " + "; ".join(parts))
    open(os.path.join(o, "README.md"), "w").write(__doc__.strip() + "\n\n" + "\n".join(lines) + "\n")
    with open(os.path.join(o, "sha256.txt"), "w") as f:
        for fn in sorted(os.listdir(o)):
            if fn in ("sha256.txt", "COMPLETE") or os.path.isdir(os.path.join(o, fn)):
                continue
            f.write(f"{sha_file(os.path.join(o, fn))}  {fn}\n")
    open(os.path.join(o, "COMPLETE"), "w").write(record["generated"] + "\n")
    print("\n".join(lines))
    return record


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-ranked-list", action="store_true", help="Part A: PRISM embeddings -> ranked list + G3/G4")
    ap.add_argument("--build-titan-ranked-list", action="store_true", help="TITAN repaired matrix -> ranked list")
    ap.add_argument("--block", action="store_true", help="Part B: whole-patient bootstrap block")
    ap.add_argument("--meta", default=L.META)
    ap.add_argument("--out-dir", default=os.path.join(PRISM_ROOT, f"eval_{dt.date.today():%Y%m%d}"),
                    help="created if absent; if present, no output of the requested steps may already exist")
    ap.add_argument("--prism-dir", default=PRISM_DIR_DEFAULT, help="directory of <slide_id>.h5 PRISM slide embeddings")
    ap.add_argument("--tag", default="native", help="PRISM arm tag (config = prism_<tag>; e.g. native, mosaic)")
    ap.add_argument("--prism-desc", default=None, help="description for the PRISM config (default by tag)")
    ap.add_argument("--dim", type=int, default=PRISM_DIM)
    ap.add_argument("--allow-missing", action="store_true",
                    help="tolerate missing h5 for MISC slides only (listed in the gates record)")
    ap.add_argument("--prism-ranked-list", default=None, help="block input (default <out-dir>/ranked_lists_prism_<tag>.csv.gz)")
    ap.add_argument("--titan-matrix", default=TITAN_MATRIX_DEFAULT)
    ap.add_argument("--titan-ranked-list", default=None, help="block input (default <out-dir>/ranked_lists_titan_repaired.csv.gz)")
    ap.add_argument("--no-titan", action="store_true", help="omit titan_repaired from the block")
    ap.add_argument("--extra-prism", action="append", default=[], metavar="TAG=PATH",
                    help="block only, repeatable: add an already-built PRISM ranked list as configuration prism_<TAG> "
                         "(after all other configurations; same queries and draws) with pairs prism_<tag> - prism_<TAG> "
                         "and prism_<TAG> - primary; PATH resolves like --prism-ranked-list (absolute, or relative to "
                         "$PROSPER_OUT); 2,778 queries x depth 10 and a query set identical to the main arm and the "
                         "frozen references are asserted (hard fail otherwise)")
    ap.add_argument("--B", type=int, default=FROZEN_B)
    ap.add_argument("--seed", type=int, default=FROZEN_SEED)
    ap.add_argument("--frozen", default=FROZEN)
    ap.add_argument("--skip-frozen-reconciliation", action="store_true",
                    help="SMOKE TESTS ONLY (e.g. small --B); recorded as reconciliation_skipped=true")
    a = ap.parse_args(argv)
    if not (a.build_ranked_list or a.build_titan_ranked_list or a.block):
        ap.error("nothing to do: pass --build-ranked-list, --build-titan-ranked-list and/or --block")
    if a.block and not a.skip_frozen_reconciliation and (a.B, a.seed) != (FROZEN_B, FROZEN_SEED):
        ap.error(f"frozen reconciliation requires --B {FROZEN_B} --seed {FROZEN_SEED}; "
                 "use --skip-frozen-reconciliation for smoke tests only")
    extras, taken = [], {t for t, *_ in REFERENCE} | {"titan_repaired", f"prism_{a.tag}"}
    for spec_ in a.extra_prism:
        etag, sep, epath = spec_.partition("=")
        if not a.block:
            ap.error("--extra-prism is only used by --block")
        if not sep or not etag or not epath:
            ap.error(f"--extra-prism expects TAG=PATH, got {spec_!r}")
        if not all(c.isalnum() or c == "_" for c in etag):
            ap.error(f"--extra-prism tag {etag!r} must be alphanumeric/underscore")
        if f"prism_{etag}" in taken:
            ap.error(f"--extra-prism tag {etag!r} gives configuration prism_{etag}, which is already in the block")
        if not os.path.isfile(os.path.join(OUT, epath)):
            ap.error(f"--extra-prism {etag}: ranked list not found: {os.path.join(OUT, epath)}")
        taken.add(f"prism_{etag}")
        extras.append((etag, epath))
    a.extra_prism = extras
    a.out_dir = os.path.abspath(a.out_dir)
    planned = []
    if a.build_ranked_list:
        planned += list(prism_outputs(a.out_dir, a.tag).values())
    if a.build_titan_ranked_list:
        planned += list(titan_outputs(a.out_dir).values())
    if a.block:
        planned += [os.path.join(a.out_dir, f) for f in BLOCK_FILES]
    if os.path.isdir(a.out_dir):
        clash = [p for p in planned if os.path.exists(p)]
        if clash:
            print(f"refusing to overwrite existing outputs in {a.out_dir}: {[os.path.basename(p) for p in clash]}", file=sys.stderr)
            return 2
    else:
        os.makedirs(a.out_dir, exist_ok=False)
    meta, meta_sha = L.load_meta(a.meta)
    try:
        if a.build_ranked_list:
            build_prism(a, meta, meta_sha)
        if a.build_titan_ranked_list:
            build_titan(a, meta, meta_sha)
        if a.block:
            block(a, meta, meta_sha)
    except GateFailure as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
