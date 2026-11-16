"""Tests for the external-cohort path added 2026-08-29: mosaic combine,
query-vs-library similarity rows, rectangular aggregation, and the external
evaluator's ranking / voting rules. No LSF, no GPU, no real slides."""
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prosper.embeddings import similarity_cli as SC
from prosper.embeddings.aggregate_cli import SENTINEL, build_external
from prosper.embeddings.combine_similarity_cli import build as combine_build
from prosper.evaluation import external_cli as E
from prosper.preprocessing import mosaic_cli as MC


def test_vote_plurality_tie_and_entropy():
    closest, tied, share, ent, n = E.vote(["A", "B", "B", "A", "C"])
    assert set(tied) == {"A", "B"} and closest == "A"      # earliest tied label in rank order
    assert abs(share - 0.4) < 1e-12 and n == 3 and 0 < ent <= 1
    assert E.vote(["A"] * 5)[3] == 0.0                    # no uncertainty
    assert E.vote(["__NONE__"] * 5)[0] == E.AMBIGUOUS


def test_strict_majority_rule():
    assert E.strict_majority(["A", "A", "A", "B", "C"], 5) == "A"
    assert E.strict_majority(["A", "A", "B", "B", "C"], 5) == E.AMBIGUOUS
    assert E.strict_majority(["A"], 1) == "A"


def test_rank_block_skips_sentinel_self_and_same_patient():
    lib = ["l1", "l2", "l3", "q1"]
    M = pd.DataFrame([[0.5, 0.1, SENTINEL, 0.0],      # q1: own id at 0.0 must be skipped
                      [0.3, np.nan, 0.2, 0.9]],
                     index=["q1", "q2"], columns=lib)
    labels = {"l1": "X", "l2": "Y", "l3": "Z", "q1": "X"}
    patients = {"l1": "pA", "l2": "pB", "l3": "pC", "q1": "pQ"}
    ids, labs, dists, short = E.rank_block(M, lib, labels, patients, ["q1", "q2"],
                                           {"q1": "pB", "q2": None}, 2)
    # q1: skip self (q1), skip l2 (patient pB), skip sentinel l3 -> only l1
    assert list(ids[0]) == ["l1", None] and labs[0][1] == "__NONE__" and short == 1
    # q2: nan skipped -> l3 (0.2) then l1 (0.3)
    assert list(ids[1]) == ["l3", "l1"] and list(labs[1]) == ["Z", "X"]


def test_row_vs_library_bob_and_self_sentinel():
    rng = np.random.default_rng(0)
    gen = SC.get_slide_embedding_generator("bob", feature_dim=8, verbose=False)
    lib = {sid: gen.process_single_slide(rng.random((5, 8)).astype(np.float32), sid)
           for sid in ["a", "b", "c"]}
    q = gen.process_single_slide(rng.random((4, 8)).astype(np.float32), "q")
    row = SC._row_vs_library(gen, "bob", "q", q, lib, ["a", "b", "c"], 1)
    assert [r[0] for r in row] == ["a", "b", "c"] and all(0 <= r[1] <= 7 for r in row)
    row2 = SC._row_vs_library(gen, "bob", "b", lib["b"], lib, ["a", "b", "c"], 1)
    assert row2[1] == ("b", SC._SELF_SENTINEL)


def _write_feats(d: Path, ids, n, dim, rng):
    for sid in ids:
        with open(d / f"{sid}.pkl", "wb") as fh:
            pickle.dump((rng.random((n, dim)).astype(np.float32), None), fh)


def test_build_external_rectangular_with_overlap_sentinel():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "q").mkdir(); (d / "lib").mkdir()
        _write_feats(d / "q", ["q1", "shared"], 6, 4, rng)
        _write_feats(d / "lib", ["l1", "l2", "shared"], 6, 4, rng)
        prov = build_external(d / "q", d / "lib", "toy", "median_pooling", "cityblock",
                              d / "out", None)
        assert prov["n_query_slides"] == 2 and prov["n_library_slides"] == 3
        assert prov["n_query_ids_also_in_library"] == 1
        with open(d / "out" / "combined_similarity_matrix_toy_median_pooling.pkl", "rb") as fh:
            df = pickle.load(fh)
        assert df.shape == (2, 3) and df.loc["shared", "shared"] == SENTINEL
        assert not (d / "out" / "combined_similarity_matrix_toy_median_pooling_no_same_patient.pkl").exists()


def test_combine_external_builds_rectangular_block():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for q in ("q1", "q2"):
            with open(d / f"similarity_{q}.pkl", "wb") as fh:
                pickle.dump([("l1", 0.1), ("l2", 0.2), ("l3", 0.3)], fh)
        prov = combine_build(d, "toy", "ot_cosine", d / "out", external=True)
        assert prov["square"] is False and prov["same_patient_pairs_masked"] is None
        with open(d / "out" / "combined_similarity_matrix_toy_ot_cosine.pkl", "rb") as fh:
            df = pickle.load(fh)
        assert df.shape == (2, 3)


def test_mosaic_combine_output_contract():
    with tempfile.TemporaryDirectory() as d:
        os.environ["PROSPER_OUT"] = d
        tmp = Path(d) / "mosaic" / "t" / "tmp"
        tmp.mkdir(parents=True)
        for sid, n in (("s2", 3), ("s1", 2)):
            with open(tmp / f"{sid}.pkl", "wb") as fh:
                pickle.dump([{"wsi_loc": [i * 512, 0]} for i in range(n)], fh)
        with open(tmp / "s3.pkl", "wb") as fh:
            pickle.dump([], fh)                       # empty -> reported, not included
        class A:  # noqa: D401 - minimal argparse stand-in
            task, ext, min_slides = "mosaic/t", ".tiff", 0
        assert MC.cmd_combine(A) == 0
        with open(Path(d) / "mosaic" / "t" / "combined_mosaics.pkl", "rb") as fh:
            entries = pickle.load(fh)
        assert [e[0] for e in entries] == ["s1.tiff", "s2.tiff"]   # sorted, ext restored
        assert entries[1][1][0]["wsi_loc"] == [0, 0]
        del os.environ["PROSPER_OUT"]
