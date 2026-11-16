"""Synthetic fixtures for the R2.1 route-(a) legacy tumour-only filter adapter.

Every fixture is generated in ``tmp_path`` (at most 12 slides); no real matrix,
population, bank or index dict is read. The frozen evaluator is only imported read-only
to confirm that the adapter outputs are accepted by ``returned_mask``/``load_matrix``.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prosper.evaluation import r2_1_legacy_filter_adapter_cli as A

SENT = 1e6
# 10 population slides, 7 patients (P1 x2, P3 x2, P6 x2), 3 groups; numeric-looking ids.
POP = pd.DataFrame({
    "query_index": range(10), "reference_index": range(10), "metadata_index": range(1, 11),
    "slide_id": ["7824", "12159", "12167", "0420", "15397", "9939", "9940", "16738",
                 "14601", "15678"],
    "patient_id": ["P1", "P1", "P2", "P3", "P3", "P4", "P5", "P6", "P6", "P7"],
    "diagnostic_group": ["LIVER", "LIVER", "RB", "OS", "OS", "RB", "LIVER", "RB", "RB", "OS"],
    "fold": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
})
EXTRA = ["99999", "88888"]  # axis slides outside the population (like the 317 extras)
IDS = POP.slide_id.tolist()
PAT = dict(zip(POP.slide_id, POP.patient_id))


def _matrix(nonret=(), diag_style="1e6", seed=0, extra=True):
    ids = IDS + (EXTRA if extra else [])
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.1, 0.9, size=(len(ids), len(ids)))  # asymmetric on purpose
    pat = np.array([PAT.get(s, f"X{s}") for s in ids], dtype=object)
    same = (pat[:, None] == pat[None, :]) & ~np.eye(len(ids), dtype=bool)
    d[same] = SENT
    np.fill_diagonal(d, SENT)
    for s in nonret:
        i = ids.index(s)
        cross = pat != pat[i]
        d[i, cross] = np.nan
        d[cross, i] = np.nan
        d[i, i] = np.nan if diag_style == "nan" else SENT
    if extra:  # extras may hold anything; they are outside the population and unchecked
        d[ids.index("99999"), :] = np.nan
    return pd.DataFrame(d, index=list(ids), columns=list(ids))


def _index_dict(nonret=(), extra=True):
    out = {}
    for k, s in enumerate(IDS + (EXTRA if extra else [])):
        n = 0 if s in nonret else (5 if s == "9940" else 10 + k)
        out[s] = np.arange(n, dtype=np.int64)
    return out


def _write(tmp_path, df=None, pop=None, idx=None, nonret=(), diag_style="1e6"):
    df = _matrix(nonret, diag_style) if df is None else df
    pop = POP if pop is None else pop
    idx = _index_dict(nonret) if idx is None else idx
    mp, pp, ip = tmp_path / "m.pkl", tmp_path / "pop.csv", tmp_path / "idx.pkl"
    with open(mp, "wb") as fh:
        pickle.dump(df, fh)
    pop.to_csv(pp, index=False)
    with open(ip, "wb") as fh:
        pickle.dump(idx, fh)
    return mp, pp, ip


def _args(mp, pp, ip, out, *extra):
    return ["--matrix", str(mp), "--population", str(pp), "--index-dict", str(ip),
            "--out-dir", str(out), *extra]


def _ok(tmp_path, **kw):
    mp, pp, ip = _write(tmp_path, **kw)
    out = tmp_path / "adapter"
    assert A.main(_args(mp, pp, ip, out)) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [A.SUMMARY_NAME, A.MATRIX_NAME, A.NONRETURN_NAME, A.META_NAME])
    summ = pd.read_csv(out / A.SUMMARY_NAME, dtype=str, keep_default_na=False)
    with open(out / A.MATRIX_NAME, "rb") as fh:
        mat = pickle.load(fh)
    meta = json.loads((out / A.META_NAME).read_text())
    nonret = json.loads((out / A.NONRETURN_NAME).read_text())
    return out, summ, mat, meta, nonret, mp


def _reject(tmp_path, capsys, needle=None, extra=(), **kw):
    mp, pp, ip = _write(tmp_path, **kw)
    out = tmp_path / "adapter"
    rc = A.main(_args(mp, pp, ip, out, *extra))
    err = capsys.readouterr().err
    assert rc != 0
    assert not out.exists() and not out.is_symlink()
    assert "ERROR" in err
    if needle:
        assert needle in err, err
    return err


def _assert_identity(src: pd.DataFrame, mat: pd.DataFrame, written):
    s = src.copy()
    s.index = [str(v) for v in s.index]
    s.columns = [str(v) for v in s.columns]
    assert list(mat.index) == written and list(mat.columns) == written
    assert all(t == np.float64 for t in mat.dtypes)
    expect = s.loc[written, written].to_numpy()
    assert np.array_equal(mat.to_numpy(), expect, equal_nan=True)
    for a in written:  # every retained ordered pair, individually
        for b in written:
            assert np.array_equal(np.array(mat.loc[a, b]), np.array(s.loc[a, b]), equal_nan=True)


# ---------------------------------------------------------------- accepted cases

def test_clean_case(tmp_path):
    out, summ, mat, meta, nonret, mp = _ok(tmp_path)
    assert list(summ.columns) == list(A.SUMMARY_HEADER)
    assert summ.slide_id.tolist() == IDS
    assert summ.slide_index.tolist() == [str(i) for i in range(10)]
    assert set(summ.written) == {"1"} and set(summ.reason) == {""}
    assert nonret == []
    _assert_identity(_matrix(), mat, IDS)
    assert meta["written_count"] == 10 and meta["nonreturned_count"] == 0
    assert meta["kept_tiles"]["min"] == 5
    assert meta["kept_tiles"]["ids_below_threshold"] == ["9940"]
    assert meta["kept_tiles"]["median"] == float(np.median(
        [len(_index_dict()[s]) for s in IDS]))
    for name in (A.SUMMARY_NAME, A.MATRIX_NAME, A.NONRETURN_NAME):
        assert meta["outputs"][name]["sha256"] == hashlib.sha256(
            (out / name).read_bytes()).hexdigest()
    assert meta["inputs"]["matrix"]["sha256"] == hashlib.sha256(mp.read_bytes()).hexdigest()
    assert meta["inputs"]["matrix"]["size_bytes"] == mp.stat().st_size
    assert set(meta["inputs"]) == {"matrix", "population", "index_dict"}
    assert meta["code"]["sha256"] == hashlib.sha256(Path(A.__file__).read_bytes()).hexdigest()
    for k in ("host", "argv", "created_utc", "versions"):
        assert meta[k]
    assert set(meta["versions"]) == {"numpy", "pandas", "python"}
    assert meta["matrix_extra_axis_ids_outside_population"] == 2
    assert len(meta["per_slide"]) == 10


@pytest.mark.parametrize("diag_style", ["1e6", "nan"], ids=["yottixel_diag_1e6", "ot_diag_nan"])
def test_nonreturn_styles(tmp_path, diag_style):
    out, summ, mat, meta, nonret, _ = _ok(tmp_path, nonret=("16738",), diag_style=diag_style)
    s = summ.set_index("slide_id")
    assert s.loc["16738", "written"] == "0"
    assert s.loc["16738", "reason"] == "legacy_zero_kept_tiles"
    assert (s.drop("16738").written == "1").all() and (s.drop("16738").reason == "").all()
    assert nonret == ["16738"]
    written = [x for x in IDS if x != "16738"]
    _assert_identity(_matrix(("16738",), diag_style), mat, written)
    ps = {r["slide_id"]: r for r in meta["per_slide"]}
    assert ps["16738"]["diagonal"] == ("NaN" if diag_style == "nan" else SENT)
    assert ps["16738"]["kept_tiles"] == 0
    assert meta["nonreturned_ids"] == ["16738"]


def test_partial_nan_from_nonreturned_slide_passes(tmp_path):
    src = _matrix(("14601", "15678"))
    # every other population row/column has exactly the NaNs towards the two N slides
    assert np.isnan(src.loc["7824", "15678"]) and np.isnan(src.loc["15678", "7824"])
    out, summ, mat, meta, nonret, _ = _ok(tmp_path, nonret=("14601", "15678"))
    assert nonret == ["14601", "15678"]
    written = [x for x in IDS if x not in nonret]
    assert summ.set_index("slide_id").written.reindex(IDS).tolist() == [
        "0" if x in nonret else "1" for x in IDS]
    _assert_identity(src, mat, written)
    pat = POP.set_index("slide_id").patient_id.reindex(written).to_numpy()
    cross = pat[:, None] != pat[None, :]
    assert not np.isnan(mat.to_numpy()[cross]).any()


def test_asymmetric_preserved_no_transposition(tmp_path):
    src = _matrix()
    assert src.loc["7824", "12167"] != src.loc["12167", "7824"]
    _, _, mat, _, _, _ = _ok(tmp_path)
    assert mat.loc["7824", "12167"] == src.loc["7824", "12167"]
    assert mat.loc["12167", "7824"] == src.loc["12167", "7824"]
    assert not np.array_equal(mat.to_numpy(), mat.to_numpy().T)


def test_int_axis_labels_and_string_slide_ids(tmp_path):
    pop = POP.copy()
    pop["slide_id"] = ["7824", "12159", "12167", "420", "15397", "9939", "9940", "16738",
                       "14601", "15678"]
    ids = pop.slide_id.tolist()
    pat = dict(zip(pop.slide_id, pop.patient_id))
    rng = np.random.default_rng(3)
    d = rng.uniform(0.1, 0.9, size=(10, 10))
    p = np.array([pat[s] for s in ids], dtype=object)
    d[(p[:, None] == p[None, :])] = SENT
    labels = [int(s) for s in ids]
    df = pd.DataFrame(d, index=labels, columns=labels)
    assert df.index.dtype.kind == "i"
    idx = {int(s): np.arange(9, dtype=np.int32) for s in ids}
    out, summ, mat, meta, nonret, _ = _ok(tmp_path, df=df, pop=pop, idx=idx)
    text = (out / A.SUMMARY_NAME).read_text()
    assert ",7824,P1," in text and "7824.0" not in text and ".0," not in text
    assert summ.slide_id.tolist() == ids
    assert list(mat.index) == ids and list(mat.columns) == ids
    assert all(isinstance(x, str) for x in mat.index)
    assert np.array_equal(mat.to_numpy(), d, equal_nan=True)


def test_summary_metadata_equals_population(tmp_path):
    _, summ, _, _, _, _ = _ok(tmp_path, nonret=("16738",))
    assert summ.slide_id.tolist() == POP.slide_id.tolist()
    assert summ.patient_id.tolist() == POP.patient_id.tolist()
    assert summ.diagnostic_group.tolist() == POP.diagnostic_group.tolist()
    assert summ.slide_index.tolist() == [str(v) for v in POP.query_index]


def test_matching_expected_sha_passes(tmp_path):
    mp, pp, ip = _write(tmp_path)
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    out = tmp_path / "adapter"
    assert A.main(_args(mp, pp, ip, out, "--expected-matrix-sha256", h(mp),
                        "--expected-population-sha256", h(pp),
                        "--expected-index-dict-sha256", h(ip).upper())) == 0


def test_frozen_evaluator_accepts_outputs(tmp_path):
    from prosper.evaluation import r2_1_filter_eval_cli as E
    nonret = ("14601", "15678")
    out, _, _, _, _, _ = _ok(tmp_path, nonret=nonret)
    pop = E.load_population(tmp_path / "pop.csv")
    arm = E.Arm("legacy_tumour_only", "ot_cosine", "legacy", out / A.MATRIX_NAME,
                out / A.SUMMARY_NAME)
    ret = E.returned_mask(arm, pop)
    assert ret.tolist() == [s not in nonret for s in pop.slide_id]
    dist, diag = E.load_matrix(arm, pop, ret)
    assert diag["extra_axis_ids_outside_population"] == 0
    src = _matrix(nonret)
    w = [s for s in IDS if s not in nonret]
    sel = np.flatnonzero(ret)
    assert np.array_equal(dist[np.ix_(sel, sel)], src.loc[w, w].to_numpy(), equal_nan=True)
    pat = pop.patient_id.to_numpy()
    cross = (pat[:, None] != pat[None, :]) & ret[:, None] & ret[None, :]
    assert np.isfinite(dist[cross]).all() and not (dist[cross] == SENT).any()


# ---------------------------------------------------------------- fail-closed cases

def test_written_by_written_nan_rejected(tmp_path, capsys):
    df = _matrix()
    df.loc["7824", "9939"] = np.nan
    _reject(tmp_path, capsys, "partial-NaN", df=df)


@pytest.mark.parametrize("value", [0.5, 0.0, np.inf, -1e6])
def test_bad_diagonal_rejected(tmp_path, capsys, value):
    df = _matrix(("16738",))
    df.loc["9939", "9939"] = value
    _reject(tmp_path, capsys, "diagonal", df=df, nonret=("16738",))


@pytest.mark.parametrize("value", [0.3, np.nan, np.inf])
def test_same_patient_not_sentinel_rejected(tmp_path, capsys, value):
    df = _matrix()
    df.loc["12159", "7824"] = value
    _reject(tmp_path, capsys, "same-patient", df=df)


def test_cross_patient_sentinel_rejected(tmp_path, capsys):
    df = _matrix()
    df.loc["7824", "12167"] = SENT
    _reject(tmp_path, capsys, "== 1e6", df=df)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_cross_patient_inf_rejected(tmp_path, capsys, value):
    df = _matrix()
    df.loc["12167", "15397"] = value
    _reject(tmp_path, capsys, "inf", df=df)


def test_population_slide_absent_from_matrix_rejected(tmp_path, capsys):
    df = _matrix().drop(index="9939", columns="9939")
    _reject(tmp_path, capsys, "missing from the axes", df=df)


def test_slide_absent_from_one_axis_rejected(tmp_path, capsys):
    df = _matrix().drop(columns="9939")
    _reject(tmp_path, capsys, "index != columns", df=df)


def test_index_columns_order_mismatch_rejected(tmp_path, capsys):
    df = _matrix()
    cols = list(df.columns)
    cols[0], cols[1] = cols[1], cols[0]
    df = df[cols]
    _reject(tmp_path, capsys, "index != columns", df=df)


def test_duplicate_axis_id_rejected(tmp_path, capsys):
    df = _matrix()
    labels = list(df.index)
    labels[-1] = "7824"
    df.index = labels
    df.columns = labels
    _reject(tmp_path, capsys, "duplicated", df=df)


def test_duplicate_axis_id_after_str_cast_rejected(tmp_path, capsys):
    df = _matrix()
    labels = list(df.index)
    labels[-1] = 7824  # int 7824 and str "7824" collide after str-cast
    df.index = labels
    df.columns = labels
    _reject(tmp_path, capsys, "duplicated", df=df)


@pytest.mark.parametrize("axis", ["row", "column"])
def test_one_axis_all_nan_other_finite_rejected(tmp_path, capsys, axis):
    df = _matrix()
    cross = [s for s in IDS if PAT[s] != PAT["9939"]]
    if axis == "row":
        df.loc["9939", cross] = np.nan
    else:
        df.loc[cross, "9939"] = np.nan
    idx = _index_dict()
    _reject(tmp_path, capsys, "partial-NaN", df=df, idx=idx)
    # also rejected when the index dict claims zero tiles for it
    idx["9939"] = np.arange(0, dtype=np.int64)
    _reject(tmp_path, capsys, None, df=df, idx=idx)


def test_index_dict_count_positive_for_nonreturned_rejected(tmp_path, capsys):
    idx = _index_dict(("16738",))
    idx["16738"] = np.arange(3, dtype=np.int64)
    _reject(tmp_path, capsys, "index-dict disagreement", nonret=("16738",), idx=idx)


def test_index_dict_zero_for_written_rejected(tmp_path, capsys):
    idx = _index_dict()
    idx["9939"] = np.arange(0, dtype=np.int64)
    _reject(tmp_path, capsys, "index-dict disagreement", idx=idx)


def test_population_slide_missing_from_index_dict_rejected(tmp_path, capsys):
    idx = _index_dict()
    del idx["15397"]
    _reject(tmp_path, capsys, "index-dict: population slides missing", idx=idx)


def test_index_dict_duplicate_key_after_cast_rejected(tmp_path, capsys):
    idx = _index_dict()
    idx[7824] = np.arange(4, dtype=np.int64)
    _reject(tmp_path, capsys, "duplicate key", idx=idx)


@pytest.mark.parametrize("value", [
    [1, 2, 3], np.arange(3, dtype=np.float64), np.zeros((2, 2), dtype=np.int64),
    np.array([True, False]), np.array([], dtype=np.float64)])
def test_index_dict_value_not_1d_integer_array_rejected(tmp_path, capsys, value):
    idx = _index_dict()
    idx["12167"] = value
    _reject(tmp_path, capsys, "1-D integer array", idx=idx)


def test_index_dict_not_dict_rejected(tmp_path, capsys):
    _reject(tmp_path, capsys, "not a dict", idx=[("7824", np.arange(3))])


def test_matrix_not_float64_rejected(tmp_path, capsys):
    df = _matrix().astype(np.float32)
    _reject(tmp_path, capsys, "float64", df=df)


def test_population_bad_query_index_rejected(tmp_path, capsys):
    pop = POP.copy()
    pop["query_index"] = [1, 0] + list(range(2, 10))
    _reject(tmp_path, capsys, "query_index", pop=pop)


@pytest.mark.parametrize("which", ["matrix", "population", "index-dict"])
def test_expected_sha_mismatch_rejected(tmp_path, capsys, which):
    _reject(tmp_path, capsys, "!= expected", extra=(f"--expected-{which}-sha256", "0" * 64))


@pytest.mark.parametrize("prefill", [False, True], ids=["empty_dir", "nonempty_dir"])
def test_output_dir_present_rejected_unchanged(tmp_path, capsys, prefill):
    mp, pp, ip = _write(tmp_path)
    out = tmp_path / "adapter"
    out.mkdir()
    if prefill:
        (out / "keep.txt").write_text("sentinel")
    before = sorted((p.name, p.read_bytes()) for p in out.iterdir())
    rc = A.main(_args(mp, pp, ip, out))
    assert rc != 0
    assert "already exists" in capsys.readouterr().err
    assert sorted((p.name, p.read_bytes()) for p in out.iterdir()) == before


def test_missing_output_parent_rejected(tmp_path, capsys):
    mp, pp, ip = _write(tmp_path)
    out = tmp_path / "absent_parent" / "adapter"
    assert A.main(_args(mp, pp, ip, out)) != 0
    assert not out.parent.exists()
