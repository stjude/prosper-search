import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import r2_6_titan_arm2_l1_matrix as T  # noqa: E402


def meta_csv(tmp_path, rows):
    p = tmp_path / "meta.csv"
    pd.DataFrame(rows, columns=["Tumor.group", "Slide.Scan.File", "Sample.ID", "UID.Subject"]).to_csv(p, index=False)
    return p


def test_distance_frame_cityblock_and_diagonal():
    X = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]])
    df = T._distance_frame(["1", "2", "3"], X, "cityblock")
    a = df.to_numpy()
    assert list(df.index) == list(df.columns) == ["1", "2", "3"]
    assert (np.diag(a) == T.SENTINEL).all()
    assert a[0, 1] == 3.0 and a[0, 2] == 1.5 and a[1, 2] == 3.5
    assert np.array_equal(a, a.T)
    c = T._distance_frame(["1", "2"], np.array([[1.0, 0.0], [0.0, 2.0]]), "cosine").to_numpy()
    assert c[0, 1] == 1.0


def test_mask_same_patient_int_ids_and_unknown(tmp_path):
    p = meta_csv(tmp_path, [["A", 11, "s1", "P1"], ["A", 12, "s2", "P1"], ["B", 13, "s3", "P2"],
                            ["B", 13, "dup", "P9"]])        # duplicate id: first row wins
    ids = ["11", "12", "13", "99"]                          # 99 absent -> its own unknown patient
    raw = T._distance_frame(ids, np.arange(8, dtype=float).reshape(4, 2) ** 2, "cityblock")
    out, n = T._mask_same_patient(raw, p)
    a, r = out.to_numpy(), raw.to_numpy()
    assert n == 1
    assert a[0, 1] == a[1, 0] == T.SENTINEL
    assert r[0, 1] < T.SENTINEL                             # input frame not modified
    m = T.masked_offdiag(a)
    assert m.sum() == 2 and m[0, 1] and m[1, 0]
    assert np.array_equal(a[~m & ~np.eye(4, dtype=bool)], r[~m & ~np.eye(4, dtype=bool)])


def test_all_zero_detection():
    X = np.array([[0.0, 0.0, 0.0], [0.0, 1e-30, 0.0], [-0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    assert T.all_zero_ids(["a", "b", "c", "d"], X) == ["a", "c"]


def test_zero_vectors_form_zero_distance_clique_under_l1():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, -2.0], [0.0, 0.0]])
    a = T._distance_frame(list("abcd"), X, "cityblock").to_numpy()
    assert a[0, 1] == a[0, 3] == a[1, 3] == 0.0
    assert a[0, 2] == a[1, 2] == 3.0                        # distance = L1 norm of the other slide
    s = T.summarise(pd.DataFrame(a, index=list("abcd"), columns=list("abcd")))
    assert s["zero_offdiag_pairs"] == 3 and s["symmetric"] and s["diagonal_all_sentinel"]
    assert s["sentinel_offdiag_pairs"] == 0 and s["max_real_distance"] == 3.0


def test_load_slide_embeddings_sorted_and_float64(tmp_path):
    import h5py
    for sid, v in (("20", [1, 2]), ("100", [3, 4]), ("3", [5, 6])):
        with h5py.File(tmp_path / f"{sid}.h5", "w") as h:
            h["features"] = np.asarray(v, dtype=np.float32)
    ids, X = T.load_slide_embeddings(tmp_path)
    assert ids == ["100", "20", "3"]                        # lexicographic path order, as the package
    assert X.dtype == np.float64 and X.tolist() == [[3, 4], [1, 2], [5, 6]]


# ---- --slide-features-dir/--outdir/--label override and output-directory guard ----
import json, pickle  # noqa: E402

import h5py  # noqa: E402
import pytest  # noqa: E402

FEAT = "comet-feature-gen/comet-titan/trident/trident_processed/40x_1024px_0px_overlap/slide_features_titan"


def _write_vecs(d, vecs):
    d.mkdir(parents=True)
    for sid, v in vecs.items():
        with h5py.File(d / f"{sid}.h5", "w") as h:
            h["features"] = np.asarray(v, dtype=np.float32).reshape(1, 768)


def synthetic_env(tmp_path, monkeypatch):
    """Repo/out layout with a 25-slide arm 1 (21 all-zero = clique) and a 6-slide new arm-2 dir. No S2 cosine
    matrix/record exists, so the override path must not read them."""
    P, repo, out = tmp_path / "P", tmp_path / "P" / "repo", tmp_path / "out"
    repo.mkdir(parents=True)
    out.mkdir()
    rng = np.random.default_rng(1)
    arm1 = {str(1000 + i): (np.zeros(768) if i < 21 else rng.standard_normal(768)) for i in range(25)}
    _write_vecs(P / FEAT, arm1)
    meta = P / "comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv"
    meta.parent.mkdir(parents=True)
    rows = [["A", int(s), f"x{s}", f"P{s}"] for s in arm1] + [["A", 2000 + i, f"y{i}", "Q" if i < 2 else f"R{i}"]
                                                                for i in range(6)]
    pd.DataFrame(rows, columns=["Tumor.group", "Slide.Scan.File", "Sample.ID", "UID.Subject"]).to_csv(meta, index=False)
    ids1, X1 = T.load_slide_embeddings(P / FEAT)
    _, m1, _ = T.build(ids1, X1, "cityblock", meta)
    mp = P / "comet-feature-gen/comet-titan/output/similarity_matrix/combined_similarity_matrix_titan_titan_no_same_patient.pkl"
    mp.parent.mkdir(parents=True)
    m1.to_pickle(mp)
    cq = out / "r2_6_titan_common_pool_20261003" / "arm1_identical_embedding_clique.csv"
    cq.parent.mkdir()
    pd.DataFrame({"slide": [str(1000 + i) for i in range(21)]}).to_csv(cq, index=False)
    new = tmp_path / "new" / "slide_features_titan"
    _write_vecs(new, {str(2000 + i): rng.standard_normal(768) for i in range(6)})
    monkeypatch.chdir(repo)
    monkeypatch.setenv("PROSPER_OUT", str(out))
    monkeypatch.setattr(T, "SAMPLE_N", 5)
    return {"out": out, "new": new, "meta": meta}


def test_default_outdir_refuses_if_exists(tmp_path, monkeypatch):
    env = synthetic_env(tmp_path, monkeypatch)
    frozen = env["out"] / T.OUT_NAME
    frozen.mkdir()
    (frozen / "keep.txt").write_text("frozen")
    with pytest.raises(SystemExit, match="REFUSING.*existing output directory"):
        T.main([])
    assert os.listdir(frozen) == ["keep.txt"] and (frozen / "keep.txt").read_text() == "frozen"


def test_explicit_outdir_refuses_if_exists(tmp_path, monkeypatch):
    env = synthetic_env(tmp_path, monkeypatch)
    od = tmp_path / "exists"
    od.mkdir()
    with pytest.raises(SystemExit, match="REFUSING"):
        T.main(["--slide-features-dir", str(env["new"]), "--expected-n", "6", "--outdir", str(od)])
    assert os.listdir(od) == []


@pytest.mark.parametrize("argv", [["--slide-features-dir", "x"], ["--expected-n", "3"], ["--label", "a/b"]])
def test_override_argument_validation(tmp_path, monkeypatch, argv):
    synthetic_env(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        T.parse_args(argv)
    assert e.value.code == 2


def test_override_builds_l1_matrix_and_skips_s2_checks(tmp_path, monkeypatch):
    env = synthetic_env(tmp_path, monkeypatch)
    od = tmp_path / "variant_out"
    rc = T.main(["--slide-features-dir", str(env["new"]), "--expected-n", "6", "--outdir", str(od),
                 "--label", "tumor_only_b5_argmax"])
    base = "combined_similarity_matrix_titan_titan_tumor_only_b5_argmax_l1"
    assert sorted(os.listdir(od)) == sorted([f"{base}_no_same_patient.pkl", f"_{base}_record.json",
                                             "reconciliation.json"])
    recon = json.load(open(od / "reconciliation.json"))
    assert rc == 0 and recon["all_passed"]
    assert set(recon["skipped_checks"]) == {"record_keys_equal_existing", "a_cosine_recompute_equals_saved_arm2",
                                            "a_l1_mask_equals_cosine_mask", "summary_arm2_cosine_matrix_saved"}
    assert all(v["result"] == "SKIPPED" and v["reason"] for v in recon["skipped_checks"].values())
    assert "arm2_cosine_matrix" not in recon["inputs"] and "summary_arm2_cosine_matrix_saved" not in recon
    assert recon["inputs"]["arm2_h5_dir"]["dir"] == str(env["new"]) and recon["inputs"]["arm2_h5_dir"]["n_files"] == 6
    assert recon["override"]["expected_n"] == 6 and recon["checks"]["d_l1_matrix_well_formed"]["sentinel_offdiag_pairs"] == 1
    rec = json.load(open(od / f"_{base}_record.json"))
    assert list(rec) == ["action", "slide_features_dir", "n_slides", "feature_dim", "metric", "model", "method",
                         "tumor_only", "same_patient_pairs_masked", "max_real_distance", "out"]
    assert rec["slide_features_dir"] == str(env["new"]) and rec["n_slides"] == 6 and rec["metric"] == "cityblock"
    assert rec["same_patient_pairs_masked"] == 1                       # 2000/2001 share patient Q
    M = pickle.load(open(od / f"{base}_no_same_patient.pkl", "rb"))
    ids, X = T.load_slide_embeddings(env["new"])
    from scipy.spatial.distance import cdist
    D = cdist(X, X, "cityblock")
    np.fill_diagonal(D, T.SENTINEL)
    D[0, 1] = D[1, 0] = T.SENTINEL
    assert list(M.index) == list(M.columns) == ids and np.allclose(M.to_numpy(), D, rtol=0, atol=1e-9)


def test_override_wrong_expected_n_fails(tmp_path, monkeypatch):
    env = synthetic_env(tmp_path, monkeypatch)
    od = tmp_path / "o"
    assert T.main(["--slide-features-dir", str(env["new"]), "--expected-n", "7", "--outdir", str(od)]) == 1
    recon = json.load(open(od / "reconciliation.json"))
    assert recon["checks"]["arm2_embeddings_loaded"]["result"] == "FAIL" and not recon["all_passed"]
