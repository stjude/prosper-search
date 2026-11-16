#!/usr/bin/env python3
"""R2.6 TITAN arm 2 (tumour-enriched) slide-embedding distance matrix recomputed
with L1 (cityblock), the manuscript metric, plus validation of the construction.

The saved arm-2 matrix was built by `prosper.preprocessing.tumor_tiles_cli matrix`
with its default `--metric cosine`; arm 1 (published all-tissue TITAN) is
`pdist(metric='cityblock')`. This script replicates `tumor_tiles_cli.matrix`
(L172-202) with metric cityblock and writes only to
`$PROSPER_OUT/r2_6_titan_arm2_l1_20261004/` (never `similarity_matrix/`).

Default invocation (no options) reproduces the frozen 10-04 S2 run; it refuses if the output directory already
exists. Optional `--slide-features-dir DIR --expected-n N [--outdir OUT] [--label LABEL]` builds the same L1
(cityblock, same-patient mask, sentinel 1e6) matrix + record from another slide_features_titan directory into
OUT/combined_similarity_matrix_titan_titan_<LABEL>_l1_no_same_patient.pkl; the comparisons to the saved S2
cosine matrix/record are then skipped with reasons (recorded under `skipped_checks`). `--outdir` alone (S2 input)
is a replay of the default run into another directory.

Run from the repository root with PROSPER_OUT exported. Does not import `prosper`;
the needed functions are copied below (sources and sha256 recorded in
reconciliation.json).
"""
import argparse, datetime, hashlib, json, os, pickle, sys, time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

PKG = "/PROSPER_WORKSPACE/prosper-search"
AGG_SRC = PKG + "/prosper/embeddings/aggregate_cli.py"
MATRIX_SRC = PKG + "/prosper/preprocessing/tumor_tiles_cli.py"
PATHS_SRC = PKG + "/prosper/evaluation/paths.py"
OUT_NAME = "r2_6_titan_arm2_l1_20261004"
BASE = "combined_similarity_matrix_titan_titan_tumor_only_l1"
SAMPLE_SEED, SAMPLE_N = 20261004, 256
SENTINEL_MIN = 1.0e5

# ---- copied verbatim from AGG_SRC (L44, L123-148) except where marked ----
SENTINEL = 1e6


def _distance_frame(ids, mat, metric):
    d = squareform(pdist(mat, metric=metric))
    np.fill_diagonal(d, SENTINEL)
    return pd.DataFrame(d, index=ids, columns=ids)


def _mask_same_patient(df, metadata_csv, patient_col="UID.Subject"):
    """Demote same-patient off-diagonal cells to the 1e6 sentinel.

    Modification: the metadata path is a parameter instead of
    `paths.slide_metadata()` (PATHS_SRC L131-140 resolves to
    COMET_ROOT/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv)."""
    meta = pd.read_csv(metadata_csv)
    id_col = "Slide.Scan.File"
    meta[id_col] = meta[id_col].astype(str)
    lookup = (meta.drop_duplicates(id_col)
                  .set_index(id_col)[patient_col].to_dict())
    ids = [str(i) for i in df.index]
    pat = np.array([str(lookup.get(i, f"__unknown_{i}")) for i in ids])
    same = pat[:, None] == pat[None, :]
    np.fill_diagonal(same, False)
    out = df.copy()
    vals = out.to_numpy()
    n_masked = int(same.sum())
    vals[same] = SENTINEL
    out.iloc[:, :] = vals
    return out, n_masked // 2
# ---- end of copy ----


def load_slide_embeddings(src):
    """As MATRIX_SRC::matrix L176-182: sorted(*.h5), stem as id, float64 flattened."""
    ids, vecs = [], []
    for h5 in sorted(Path(src).glob("*.h5")):
        with h5py.File(h5, "r") as h:
            v = np.asarray(h["features"][:], dtype=np.float64).reshape(-1)
        ids.append(h5.stem)
        vecs.append(v)
    return ids, np.stack(vecs)


def build(ids, mat, metric, metadata_csv):
    """As MATRIX_SRC::matrix L185-187 (raw + same-patient-masked)."""
    raw = _distance_frame(ids, mat, metric)
    masked, n_pairs = _mask_same_patient(raw, metadata_csv)
    return raw, masked, n_pairs


def all_zero_ids(ids, mat):
    return [s for s, z in zip(ids, ~mat.any(axis=1)) if z]


def masked_offdiag(a):
    off = ~np.eye(len(a), dtype=bool)
    return (a >= SENTINEL_MIN) & off


def summarise(M):
    a = M.to_numpy()
    off = ~np.eye(len(a), dtype=bool)
    real = a[a < SENTINEL_MIN]
    return {"n": int(len(a)), "shape": list(a.shape), "index_equals_columns": bool((M.index == M.columns).all()),
            "index_sorted": list(M.index) == sorted(M.index), "symmetric": bool(np.array_equal(a, a.T)),
            "diagonal_all_sentinel": bool((np.diag(a) == SENTINEL).all()),
            "sentinel_values": sorted(set(a[a >= SENTINEL_MIN].tolist())),
            "sentinel_offdiag_pairs": int(masked_offdiag(a).sum() // 2),
            "max_real_distance": float(real.max()), "min_offdiag_distance": float(a[off].min()),
            "zero_offdiag_pairs": int((a[off] == 0).sum() // 2), "nan_or_inf": int((~np.isfinite(a)).sum())}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_hash(src):
    """sha256 over (file name, sha256 of bytes) of the sorted *.h5 files."""
    h, n = hashlib.sha256(), 0
    for f in sorted(Path(src).glob("*.h5")):
        h.update(f.name.encode() + b"\0" + sha256(f).encode() + b"\n")
        n += 1
    return {"dir": str(src), "n_files": n, "sha256_of_name_and_file_sha256": h.hexdigest()}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slide-features-dir", default=None,
                    help="slide_features_titan dir to use instead of the S2 arm-2 dir (skips S2-only comparisons)")
    ap.add_argument("--expected-n", type=int, default=None,
                    help="required with --slide-features-dir: expected number of slide embeddings")
    ap.add_argument("--outdir", default=None, help=f"default $PROSPER_OUT/{OUT_NAME}; must not exist")
    ap.add_argument("--label", default="tumor_only",
                    help="output file stem combined_similarity_matrix_titan_titan_<label>_l1 (default tumor_only)")
    a = ap.parse_args(argv)
    if (a.slide_features_dir is None) != (a.expected_n is None):
        ap.error("--slide-features-dir and --expected-n go together")
    if not a.label or "/" in a.label:
        ap.error("--label must be a non-empty file-name fragment")
    return a


def main(argv=None):
    t0 = time.time()
    args = parse_args(argv)
    override = args.slide_features_dir is not None
    R = os.getcwd()
    out_root = os.environ.get("PROSPER_OUT")
    if not out_root:
        sys.exit("export PROSPER_OUT first")
    P = os.path.dirname(R)
    inp = {
        "arm2_h5_dir": out_root + "/s2_titan_tumor_only/40x_1024px_0px_overlap/slide_features_titan",
        "arm1_h5_dir": P + "/comet-feature-gen/comet-titan/trident/trident_processed/40x_1024px_0px_overlap/slide_features_titan",
        "metadata": P + "/comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv",
        "arm2_cosine_matrix": out_root + "/similarity_matrix/combined_similarity_matrix_titan_titan_tumor_only_no_same_patient.pkl",
        "arm2_cosine_record": out_root + "/similarity_matrix/_combined_similarity_matrix_titan_titan_tumor_only_record.json",
        "arm1_matrix": P + "/comet-feature-gen/comet-titan/output/similarity_matrix/combined_similarity_matrix_titan_titan_no_same_patient.pkl",
        "arm1_clique_csv": out_root + "/r2_6_titan_common_pool_20261003/arm1_identical_embedding_clique.csv",
    }
    n_expected = args.expected_n if override else 2804
    if override:
        inp["arm2_h5_dir"] = os.path.abspath(args.slide_features_dir)
        for k in ("arm2_cosine_matrix", "arm2_cosine_record"):
            del inp[k]
        if not Path(inp["arm2_h5_dir"]).is_dir():
            sys.exit(f"REFUSING: {inp['arm2_h5_dir']} is not a directory")
    base = BASE if args.label == "tumor_only" else f"combined_similarity_matrix_titan_titan_{args.label}_l1"
    outdir = Path(args.outdir) if args.outdir else Path(out_root) / OUT_NAME
    if outdir.exists():
        sys.exit(f"REFUSING: {outdir} exists; this script never writes into an existing output directory")
    out_pkl = outdir / f"{base}_no_same_patient.pkl"
    out_rec = outdir / f"_{base}_record.json"
    out_recon = outdir / "reconciliation.json"
    for f in (out_pkl, out_rec, out_recon):
        if f.exists():
            sys.exit(f"REFUSING: {f} exists; this script never overwrites outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    rec = {"timestamp": datetime.datetime.now().astimezone().isoformat(),
           "command": " ".join([sys.executable] + sys.argv),
           "script": {"path": os.path.abspath(__file__), "sha256": sha256(__file__)},
           "copied_code_sources": {s: {"path": s, "sha256": sha256(s)} for s in (AGG_SRC, MATRIX_SRC, PATHS_SRC)},
           "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": __import__("scipy").__version__, "h5py": h5py.__version__},
           "inputs": {}, "checks": {}, "outputs": {}}
    for k, v in inp.items():
        rec["inputs"][k] = dir_hash(v) if k.endswith("_h5_dir") else {"path": v, "sha256": sha256(v)}
    checks = rec["checks"]
    if override:
        rec["override"] = {"slide_features_dir": inp["arm2_h5_dir"], "expected_n": n_expected,
                           "outdir": str(outdir), "label": args.label}
        rec["skipped_checks"] = {}

    def skip(name, reason):
        rec["skipped_checks"][name] = {"result": "SKIPPED", "reason": reason}
        print(f"[SKIP] {name} {reason}", flush=True)

    def check(name, ok, **info):
        checks[name] = {"result": "PASS" if ok else "FAIL", **info}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info, default=str)[:400]}", flush=True)

    # ---- arm 2: embeddings, new L1 matrix ----
    ids2, X2 = load_slide_embeddings(inp["arm2_h5_dir"])
    check("arm2_embeddings_loaded", len(ids2) == n_expected and X2.shape == (n_expected, 768),
          n_slides=len(ids2), shape=list(X2.shape))
    raw_l1, l1, n_l1 = build(ids2, X2, "cityblock", inp["metadata"])
    with open(out_pkl, "wb") as fh:
        pickle.dump(l1, fh)
    finite = raw_l1.to_numpy()[raw_l1.to_numpy() < SENTINEL]
    record = {"action": "matrix", "slide_features_dir": inp["arm2_h5_dir"], "n_slides": len(ids2),
              "feature_dim": int(X2.shape[1]), "metric": "cityblock", "model": "titan",
              "method": "titan", "tumor_only": True, "same_patient_pairs_masked": n_l1,
              "max_real_distance": float(finite.max()) if finite.size else None,
              "out": str(out_pkl)}
    if override:
        skip("record_keys_equal_existing", "compares with the saved S2 cosine record; not applicable to "
             "--slide-features-dir input (record is built by the same code)")
    else:
        old_rec = json.load(open(inp["arm2_cosine_record"]))
        check("record_keys_equal_existing", list(record) == list(old_rec), keys=list(record))
    with open(out_rec, "w") as fh:
        json.dump(record, fh, indent=1)

    # ---- (a) cosine recompute reproduces the saved arm-2 matrix ----
    if override:
        reason = "S2-only: the saved cosine matrix/record were built from the S2 slide_features_titan dir"
        skip("a_cosine_recompute_equals_saved_arm2", reason)
        skip("a_l1_mask_equals_cosine_mask", reason)
    else:
        M2 = pd.read_pickle(inp["arm2_cosine_matrix"])
        _, cos, n_cos = build(ids2, X2, "cosine", inp["metadata"])
        a_new, a_old = cos.to_numpy(), M2.to_numpy()
        same_layout = list(cos.index) == list(M2.index) and list(cos.columns) == list(M2.columns)
        m_new, m_old = masked_offdiag(a_new), masked_offdiag(a_old)
        unm = ~m_old & ~np.eye(len(a_old), dtype=bool)
        mad = float(np.abs(a_new[unm] - a_old[unm]).max()) if same_layout else None
        check("a_cosine_recompute_equals_saved_arm2",
              same_layout and np.array_equal(m_new, m_old) and mad is not None and mad <= 1e-12
              and n_cos == old_rec["same_patient_pairs_masked"] == int(m_old.sum() // 2) == 1765,
              same_index_and_column_order=same_layout, max_abs_diff_unmasked_offdiag=mad,
              bitwise_equal_full_matrix=bool(same_layout and np.array_equal(a_new, a_old)),
              masked_cell_sets_identical=bool(np.array_equal(m_new, m_old)),
              masked_pairs_recomputed=n_cos, masked_pairs_saved_matrix=int(m_old.sum() // 2),
              masked_pairs_saved_record=old_rec["same_patient_pairs_masked"])
        check("a_l1_mask_equals_cosine_mask", np.array_equal(masked_offdiag(l1.to_numpy()), m_old) and n_l1 == 1765,
              masked_pairs_l1=n_l1)
        del cos, a_new

    # ---- (b) arm-1 cityblock recompute reproduces published arm-1 cells ----
    M1 = pd.read_pickle(inp["arm1_matrix"])
    ids1, X1 = load_slide_embeddings(inp["arm1_h5_dir"])
    _, l1_arm1, n1 = build(ids1, X1, "cityblock", inp["metadata"])
    b_new, b_old = l1_arm1.to_numpy(), M1.to_numpy()
    lay1 = list(l1_arm1.index) == list(M1.index) and list(l1_arm1.columns) == list(M1.columns)
    rng = np.random.default_rng(SAMPLE_SEED)
    samp = np.sort(rng.choice(len(ids1), size=SAMPLE_N, replace=False))
    unm1 = ~masked_offdiag(b_old) & ~np.eye(len(b_old), dtype=bool)
    s_unm = unm1[samp]
    s_diff = float(np.abs(b_new[samp][s_unm] - b_old[samp][s_unm]).max()) if lay1 else None
    full_diff = float(np.abs(b_new[unm1] - b_old[unm1]).max()) if lay1 else None
    s_mask_eq = bool(lay1 and np.array_equal(masked_offdiag(b_new)[samp], masked_offdiag(b_old)[samp]))
    check("b_arm1_cityblock_recompute_equals_published_sample",
          lay1 and s_diff is not None and s_diff <= 1e-9 and s_mask_eq,
          sample_rows=SAMPLE_N, sample_seed=SAMPLE_SEED, sample_columns=len(ids1),
          sample_cells_compared=int(s_unm.sum()), max_abs_diff_sample=s_diff,
          sample_masked_cells_identical=s_mask_eq, same_index_and_column_order=lay1, n_arm1_h5=len(ids1))
    check("b_arm1_cityblock_recompute_equals_published_full_matrix_info",
          lay1 and full_diff is not None and full_diff <= 1e-9
          and np.array_equal(masked_offdiag(b_new), masked_offdiag(b_old)),
          max_abs_diff_all_unmasked_offdiag=full_diff,
          bitwise_equal_full_matrix=bool(lay1 and np.array_equal(b_new, b_old)),
          masked_pairs_recomputed=n1, masked_pairs_published=int(masked_offdiag(b_old).sum() // 2))
    del l1_arm1, b_new

    # ---- (c) all-zero embeddings ----
    z1, z2 = all_zero_ids(ids1, X1), all_zero_ids(ids2, X2)
    clique = sorted(pd.read_csv(inp["arm1_clique_csv"], dtype={"slide": str})["slide"])
    check("c_arm1_all_zero_embeddings", len(z1) == 21 and sorted(z1) == clique,
          n_all_zero=len(z1), ids=z1, equals_clique_csv=sorted(z1) == clique,
          n_clique_csv=len(clique))
    check("c_arm2_all_zero_embeddings", len(z2) == 0, n_all_zero=len(z2), ids=z2,
          arm2_present_for_arm1_zero_ids=[s for s in z1 if s in set(ids2)])

    # ---- (d) summaries ----
    s_l1 = summarise(l1)
    rec["summary_arm2_l1_matrix"] = s_l1
    if override:
        skip("summary_arm2_cosine_matrix_saved", "S2-only saved cosine matrix not read with --slide-features-dir")
    else:
        rec["summary_arm2_cosine_matrix_saved"] = summarise(M2)
    check("d_l1_matrix_well_formed", s_l1["n"] == n_expected and s_l1["symmetric"] and s_l1["diagonal_all_sentinel"]
          and s_l1["index_equals_columns"] and s_l1["nan_or_inf"] == 0
          and s_l1["sentinel_values"] == [SENTINEL]
          and s_l1["sentinel_offdiag_pairs"] == (n_l1 if override else 1765), **s_l1)
    rec["record_max_real_distance_note"] = (
        "record max_real_distance follows the package: max over the UNMASKED matrix (same-patient cells "
        "included); summary max_real_distance is over the same-patient-masked matrix")
    check("d_record_max_real_distance_vs_masked_matrix",
          abs(record["max_real_distance"] - s_l1["max_real_distance"]) < 1e-12,
          record_max_real_distance=record["max_real_distance"],
          masked_matrix_max_real_distance=s_l1["max_real_distance"])

    for f in (out_pkl, out_rec):
        rec["outputs"][f.name] = {"path": str(f), "sha256": sha256(f)}
    rec["all_passed"] = all(c["result"] == "PASS" for c in checks.values())
    rec["runtime_s"] = round(time.time() - t0, 1)
    with open(out_recon, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(json.dumps({"all_passed": rec["all_passed"], "runtime_s": rec["runtime_s"]}))
    return 0 if rec["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
