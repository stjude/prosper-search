"""Synthetic fixtures for the R2.1 F1a tumour-filtered feature writer."""
from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from prosper.evaluation import r2_1_filter_features_cli as F

WIDTH = 6


def _bundle(tmp_path, n_tiles=(12, 10, 9), folds=(0, 1, 0)):
    rng = np.random.default_rng(0)
    mosaic, feats_dir = tmp_path / "mosaic", tmp_path / "features"
    mosaic.mkdir()
    feats_dir.mkdir()
    total = sum(n_tiles)
    coords_all = rng.integers(0, 10_000, size=(total, 2)).astype(np.int64)
    np.save(mosaic / "coordinates.npy", coords_all)
    rows, start = [], 0
    for i, (n, f) in enumerate(zip(n_tiles, folds)):
        sid = f"{100 + i}"
        feats = rng.normal(size=(n, WIDTH)).astype(np.float32)
        coords = [{"wsi_loc": [int(x), int(y)]} for x, y in coords_all[start:start + n]]
        data = pickle.dumps((feats, coords), protocol=4)
        (feats_dir / f"{sid}.pkl").write_bytes(data)
        rows.append(dict(slide_index=i, slide_id=sid, patient_id=f"P{i}", diagnostic_group="G",
                         slide_fold=f, tile_start=start, tile_stop=start + n, feature_rows=n,
                         feature_sha256=hashlib.sha256(data).hexdigest()))
        start += n
    pd.DataFrame(rows).to_csv(mosaic / "slide_offsets.csv", index=False)
    pd.DataFrame(rows)[["slide_id"]].to_csv(mosaic / "retrieval_population.csv", index=False)
    (mosaic / "COMPLETE.json").write_text("{}")
    # hand-built masks: fold 0 keeps tiles with even index; fold 1 keeps first 3
    count_rows = []
    for b in (5, 100):
        for f in (0, 1):
            pred = np.full(total, 2, dtype=np.int64)
            for r in rows:
                a, z = r["tile_start"], r["tile_stop"]
                if f == 0:
                    pred[a:z][::2] = 1
                else:
                    pred[a:a + 3] = 1
                if b == 100 and r["slide_id"] == "100" and f == 0:
                    pred[a:z] = 2          # budget 100, fold 0: slide 100 retains nothing
                keep = pred[a:z] == 1
                count_rows.append(dict(held_out_fold=f, budget_percent=b, slide_index=r["slide_index"],
                                       slide_id=r["slide_id"], slide_fold=r["slide_fold"],
                                       query_fold_eligible=int(f == r["slide_fold"]),
                                       retained_tiles=int(keep.sum())))
            np.save(mosaic / f"fold{f}_budget{b}_mask.npy", pred == 1)
            np.save(mosaic / f"fold{f}_budget{b}_predictions.npy", pred)
    pd.DataFrame(count_rows).to_csv(mosaic / "mask_counts.csv.gz", index=False)
    return mosaic, feats_dir, coords_all, rows


def _run(mosaic, feats_dir, out, **kw):
    args = dict(mosaic_dir=mosaic, features_dir=feats_dir, out_root=out, budgets=[5, 100],
                min_tiles=4, lo=1, hi=0, passthrough=[], width=WIDTH)
    args.update(kw)
    return F.run(**args)


def test_filter_writes_head_of_own_fold_and_skips_small(tmp_path):
    mosaic, feats_dir, coords_all, rows = _bundle(tmp_path)
    out = tmp_path / "out"
    rec = _run(mosaic, feats_dir, out, passthrough=["101"])
    # slide 100 (fold 0, 12 tiles): even tiles -> 6 kept; slide 101 (fold 1): first 3 -> skipped (<4)
    # slide 102 (fold 0, 9 tiles): 5 kept
    feats, coords = pickle.loads((out / "budget5" / "100.pkl").read_bytes())
    src_f, src_c = pickle.loads((feats_dir / "100.pkl").read_bytes())
    assert np.array_equal(feats, src_f[::2]) and coords == src_c[::2]
    assert not (out / "budget5" / "101.pkl").exists()
    assert (out / "budget5" / "102.pkl").exists()
    assert not (out / "budget100" / "100.pkl").exists()       # zero retained at budget 100
    s = pd.read_csv(out / "budget5" / "_filter_summary.csv", dtype={"slide_id": str})
    assert s.set_index("slide_id").retained_tiles.to_dict() == {"100": 6, "101": 3, "102": 5}
    assert s.written.tolist() == [1, 0, 1]
    assert rec["budgets"]["5"]["skipped"] == 1 and rec["budgets"]["100"]["skipped"] == 2
    assert s.loc[0, "sha256"] == hashlib.sha256((out / "budget5" / "100.pkl").read_bytes()).hexdigest()
    # passthrough is the all-True-mask path through the same writer
    pt = rec["passthrough"][0]
    assert pt["byte_identical"] is True
    assert (out / "passthrough" / "101.pkl").read_bytes() == (feats_dir / "101.pkl").read_bytes()


def test_rerun_is_idempotent_and_never_overwrites(tmp_path):
    mosaic, feats_dir, *_ = _bundle(tmp_path)
    out = tmp_path / "out"
    _run(mosaic, feats_dir, out, budgets=[5], lo=1, hi=2)
    before = (out / "budget5" / "100.pkl").read_bytes()
    _run(mosaic, feats_dir, out, budgets=[5], lo=1, hi=2)      # identical bytes -> accepted
    assert (out / "budget5" / "100.pkl").read_bytes() == before
    assert (out / "budget5" / "_filter_record_1_2.json").exists()
    assert (out / "budget5" / "_filter_record_1_2.run2.json").exists()
    (out / "budget5" / "102.pkl").write_bytes(b"tampered")
    with pytest.raises(F.FilterError, match="refusing to overwrite"):
        _run(mosaic, feats_dir, out, budgets=[5], lo=3, hi=3)


def test_sha256_mismatch_rejected(tmp_path):
    mosaic, feats_dir, *_ = _bundle(tmp_path)
    f, c = pickle.loads((feats_dir / "101.pkl").read_bytes())
    f[0, 0] += 1.0
    (feats_dir / "101.pkl").write_bytes(pickle.dumps((f, c), protocol=4))
    with pytest.raises(F.FilterError, match="sha256"):
        _run(mosaic, feats_dir, tmp_path / "out")


def test_coordinate_mismatch_rejected_even_with_matching_sha(tmp_path):
    mosaic, feats_dir, *_ = _bundle(tmp_path)
    f, c = pickle.loads((feats_dir / "102.pkl").read_bytes())
    c[3] = {"wsi_loc": [c[3]["wsi_loc"][0] + 1, c[3]["wsi_loc"][1]]}
    data = pickle.dumps((f, c), protocol=4)
    (feats_dir / "102.pkl").write_bytes(data)
    off = pd.read_csv(mosaic / "slide_offsets.csv", dtype={"slide_id": str})
    off.loc[off.slide_id == "102", "feature_sha256"] = hashlib.sha256(data).hexdigest()
    off.to_csv(mosaic / "slide_offsets.csv", index=False)
    with pytest.raises(F.FilterError, match="first bad row 3"):
        _run(mosaic, feats_dir, tmp_path / "out")


def test_mask_counts_disagreement_rejected(tmp_path):
    mosaic, feats_dir, *_ = _bundle(tmp_path)
    mc = pd.read_csv(mosaic / "mask_counts.csv.gz", dtype={"slide_id": str})
    mc.loc[(mc.slide_id == "100") & (mc.held_out_fold == 0) & (mc.budget_percent == 5),
           "retained_tiles"] = 7
    mc.to_csv(mosaic / "mask_counts.csv.gz", index=False)
    with pytest.raises(F.FilterError, match="mask_counts"):
        _run(mosaic, feats_dir, tmp_path / "out", budgets=[5])


def test_output_is_similarity_cli_loadable(tmp_path):
    from prosper.embeddings import similarity_cli
    from prosper.embeddings.slide_embedding import get_slide_embedding_generator
    mosaic, feats_dir, *_ = _bundle(tmp_path)
    out = tmp_path / "out"
    _run(mosaic, feats_dir, out, budgets=[5])
    gen = get_slide_embedding_generator("bob", feature_dim=WIDTH, verbose=False)
    ids, data = similarity_cli._load_slide_data(out / "budget5", gen, None)
    assert ids == ["100", "102"] and data["100"]["barcodes"].shape == (6, WIDTH - 1)
    rec = json.loads((out / "budget5" / "_filter_record.json").read_text())
    assert rec["budgets"]["5"]["written"] == 2


# ---------------------------------------------------------------- query-fold design (--head-fold)
def _all_true_head(mosaic, fold):
    """Rewrite head `fold` (both budgets) to keep every tile, with consistent mask_counts."""
    mc = pd.read_csv(mosaic / "mask_counts.csv.gz", dtype={"slide_id": str})
    off = pd.read_csv(mosaic / "slide_offsets.csv", dtype={"slide_id": str})
    total = int(off.tile_stop.iloc[-1])
    for b in (5, 100):
        np.save(mosaic / f"fold{fold}_budget{b}_mask.npy", np.ones(total, dtype=bool))
        np.save(mosaic / f"fold{fold}_budget{b}_predictions.npy", np.ones(total, dtype=np.int64))
    n = off.set_index("slide_id").feature_rows
    sel = mc.held_out_fold == fold
    mc.loc[sel, "retained_tiles"] = mc.loc[sel, "slide_id"].map(n).to_numpy()
    mc.to_csv(mosaic / "mask_counts.csv.gz", index=False)


def test_head_fold_filters_every_slide_with_that_head(tmp_path):
    mosaic, feats_dir, *_ = _bundle(tmp_path)
    out = tmp_path / "qf"
    rec = _run(mosaic, feats_dir, out, head_fold=1)
    # head 1 keeps the first 3 tiles of EVERY slide (also fold-0 slides 100 and 102) -> all < 4
    for b in (5, 100):
        d = out / f"budget{b}" / "fold1"
        s = pd.read_csv(d / "_filter_summary.csv", dtype={"slide_id": str})
        assert tuple(s.columns) == F.SUMMARY_HEADER_QF
        assert (s.head_fold == 1).all() and s.slide_fold.tolist() == [0, 1, 0]
        assert s.own_fold_query.tolist() == [0, 1, 0] and s.retained_tiles.tolist() == [3, 3, 3]
        assert s.written.sum() == 0 and not list(d.glob("*.pkl"))
        assert rec["budgets"][str(b)]["own_fold_queries_intended"] == 1
        assert rec["budgets"][str(b)]["library_size"] == 0
    assert rec["head_fold"] == 1
    assert not (out / "budget5" / "100.pkl").exists()          # nothing in the own-fold layout
    out0 = tmp_path / "qf0"
    rec0 = _run(mosaic, feats_dir, out0, head_fold=0, budgets=[5])
    # head 0 keeps even tiles of every slide: 100 -> 6, 101 (fold 1!) -> 5, 102 -> 5
    s = pd.read_csv(out0 / "budget5" / "fold0" / "_filter_summary.csv", dtype={"slide_id": str})
    assert s.retained_tiles.tolist() == [6, 5, 5] and s.written.tolist() == [1, 1, 1]
    f101, c101 = pickle.loads((out0 / "budget5" / "fold0" / "101.pkl").read_bytes())
    src_f, src_c = pickle.loads((feats_dir / "101.pkl").read_bytes())
    assert np.array_equal(f101, src_f[::2]) and c101 == src_c[::2] and f101.dtype == np.float32
    assert rec0["budgets"]["5"]["own_fold_queries_returned"] == 2
    assert rec0["budgets"]["5"]["library_excluded_slides"] == []


def test_head_fold_all_true_masks_are_byte_identical_passthrough(tmp_path):
    mosaic, feats_dir, *_ = _bundle(tmp_path)
    _all_true_head(mosaic, 1)
    out = tmp_path / "qf"
    _run(mosaic, feats_dir, out, head_fold=1, min_tiles=1)
    for b in (5, 100):
        for sid in ("100", "101", "102"):
            assert (out / f"budget{b}" / "fold1" / f"{sid}.pkl").read_bytes() == \
                (feats_dir / f"{sid}.pkl").read_bytes()


def test_head_fold_bank_equals_own_fold_bank_for_own_fold_slides(tmp_path):
    mosaic, feats_dir, *_ = _bundle(tmp_path)
    own, qf = tmp_path / "own", tmp_path / "qf"
    _run(mosaic, feats_dir, own, budgets=[5])
    _run(mosaic, feats_dir, qf, budgets=[5], head_fold=0)
    a = pd.read_csv(own / "budget5" / "_filter_summary.csv", dtype={"slide_id": str}).set_index("slide_id")
    b = pd.read_csv(qf / "budget5" / "fold0" / "_filter_summary.csv", dtype={"slide_id": str}).set_index("slide_id")
    for sid in ("100", "102"):                                   # own fold 0: same head
        assert (own / "budget5" / f"{sid}.pkl").read_bytes() == (qf / "budget5" / "fold0" / f"{sid}.pkl").read_bytes()
        assert a.sha256[sid] == b.sha256[sid]
    # 101 (own fold 1): head 1 keeps 3 tiles (skipped in own-fold bank); head 0 keeps 5 (written)
    assert a.written["101"] == 0 and b.written["101"] == 1


def test_head_fold_mask_counts_disagreement_rejected(tmp_path):
    mosaic, feats_dir, *_ = _bundle(tmp_path)
    mc = pd.read_csv(mosaic / "mask_counts.csv.gz", dtype={"slide_id": str})
    sel = (mc.slide_id == "101") & (mc.held_out_fold == 0) & (mc.budget_percent == 5)
    mc.loc[sel, "retained_tiles"] = 4                          # true count under head 0 is 5
    mc.to_csv(mosaic / "mask_counts.csv.gz", index=False)
    with pytest.raises(F.FilterError, match="head fold 0"):
        _run(mosaic, feats_dir, tmp_path / "qf", budgets=[5], head_fold=0)
    with pytest.raises(F.FilterError, match="not a slide fold"):
        _run(mosaic, feats_dir, tmp_path / "qf2", budgets=[5], head_fold=3)


def test_default_mode_bytes_equal_pre_edit_module(tmp_path):
    """Default (own-fold) outputs are byte-identical to the archived pre-edit module's."""
    import importlib.util
    import os
    root = os.environ.get("PROSPER_OUT")
    old_p = (None if not root else
             __import__("pathlib").Path(root) / "r2_1_filter_20261003" / "qf_code_baseline_before_edit"
             / "r2_1_filter_features_cli.py")
    if old_p is None or not old_p.exists():
        pytest.skip("archived pre-edit module not available")
    spec = importlib.util.spec_from_file_location("old_features_cli", old_p)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    mosaic, feats_dir, *_ = _bundle(tmp_path)
    kw = dict(mosaic_dir=mosaic, features_dir=feats_dir, budgets=[5, 100], min_tiles=4, lo=1, hi=0,
              passthrough=["101"], width=WIDTH)
    old.run(out_root=tmp_path / "old", **kw)
    F.run(out_root=tmp_path / "new", **kw)
    names = sorted(str(p.relative_to(tmp_path / "old")) for p in (tmp_path / "old").rglob("*")
                   if p.is_file() and "_record" not in p.name)
    assert names == sorted(str(p.relative_to(tmp_path / "new")) for p in (tmp_path / "new").rglob("*")
                           if p.is_file() and "_record" not in p.name)
    assert len(names) >= 6
    for n in names:
        assert (tmp_path / "old" / n).read_bytes() == (tmp_path / "new" / n).read_bytes(), n


def test_make_views_links_fold_queries_and_unfiltered_library(tmp_path):
    mosaic, feats_dir, *_ = _bundle(tmp_path)
    banks, views = tmp_path / "qf", tmp_path / "views"
    _run(mosaic, feats_dir, banks, budgets=[5], head_fold=0)
    _run(mosaic, feats_dir, banks, budgets=[5], head_fold=1)
    rec = F.make_views(mosaic_dir=mosaic, features_dir=feats_dir, banks_root=banks, views_root=views,
                       budgets=[5], head_folds=[0, 1])
    q0 = sorted(p.name for p in (views / "queries" / "budget5" / "fold0").glob("*.pkl"))
    assert q0 == ["100.pkl", "102.pkl"]                              # fold-0 queries only, not 101
    assert (views / "queries" / "budget5" / "fold0" / "100.pkl").read_bytes() == \
        (banks / "budget5" / "fold0" / "100.pkl").read_bytes()
    assert not list((views / "queries" / "budget5" / "fold1").glob("*.pkl"))   # 101 not returned
    v1 = rec["views"]["queries/budget5/fold1"]
    assert v1["queries_intended"] == 1 and v1["queries_returned"] == 0
    assert v1["non_returned_queries"] == ["101"]
    lib = views / F.UNFILTERED_LIBRARY_VIEW
    assert sorted(p.name for p in lib.glob("*.pkl")) == ["100.pkl", "101.pkl", "102.pkl"]
    assert (lib / "101.pkl").read_bytes() == (feats_dir / "101.pkl").read_bytes()
    F.make_views(mosaic_dir=mosaic, features_dir=feats_dir, banks_root=banks, views_root=views,
                 budgets=[5], head_folds=[0, 1])                    # idempotent
    assert (views / "_views_record.run2.json").exists()
    # a tampered bank pickle is caught by the sha check
    (banks / "budget5" / "fold0" / "102.pkl").write_bytes(b"x")
    with pytest.raises(F.FilterError, match="sha256"):
        F.make_views(mosaic_dir=mosaic, features_dir=feats_dir, banks_root=banks,
                     views_root=tmp_path / "views2", budgets=[5], head_folds=[0],
                     unfiltered_library=False)
    # an existing link pointing elsewhere is never replaced
    os_link = views / "queries" / "budget5" / "fold0" / "100.pkl"
    os_link.unlink()
    os_link.symlink_to(feats_dir / "100.pkl")
    with pytest.raises(F.FilterError, match="refusing to replace"):
        F.make_views(mosaic_dir=mosaic, features_dir=feats_dir, banks_root=banks, views_root=views,
                     budgets=[5], head_folds=[0], verify_sha=False, unfiltered_library=False)
