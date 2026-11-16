"""Tests for the new retrieval-metric suite (R4.b) and path/queue plumbing.

The metric definitions are checked against hand-computed values rather than
against another implementation, since there is no reference implementation
anywhere in the project -- that absence is why this module was written.
"""

import numpy as np
import pytest

from prosper.evaluation import metrics, paths
from prosper.hpc import queues, submit


# ---------------------------------------------------------------- metrics ----

# Three queries, all of true class "A". Retrieval depth 3.
#   q0: A B A -> rel 1 0 1
#   q1: B B B -> rel 0 0 0
#   q2: C A A -> rel 0 1 1
NL = np.array([["A", "B", "A"],
               ["B", "B", "B"],
               ["C", "A", "A"]], dtype=object)
TL = ["A", "A", "A"]
COUNTS = {"A": 10, "B": 5, "C": 2}


def _rel():
    return metrics._relevance(NL, np.asarray(TL, dtype=object))


def test_hit_at_k_is_monotone_in_k():
    rel = _rel()
    h1 = metrics.hit_at_k(rel, 1).mean()
    h3 = metrics.hit_at_k(rel, 3).mean()
    assert h1 == pytest.approx(1 / 3)
    assert h3 == pytest.approx(2 / 3)
    assert h1 <= h3


def test_precision_at_k_hand_computed():
    rel = _rel()
    assert metrics.precision_at_k(rel, 1).tolist() == [1.0, 0.0, 0.0]
    np.testing.assert_allclose(metrics.precision_at_k(rel, 3),
                               [2 / 3, 0.0, 2 / 3])


def test_textbook_recall_uses_class_size_denominator():
    """recall@k here is bounded by k/|class|, which is the point of the warning
    in the module docstring: it must never be presented as hit@k."""
    rel = _rel()
    n_rel = np.array([10.0, 10.0, 10.0])
    np.testing.assert_allclose(metrics.recall_at_k(rel, 3, n_rel),
                               [0.2, 0.0, 0.2])
    # And it is a genuinely different number from hit@k.
    assert metrics.recall_at_k(rel, 3, n_rel).mean() != \
        pytest.approx(metrics.hit_at_k(rel, 3).mean())


def test_reciprocal_rank_is_zero_when_no_relevant_in_list():
    rr = metrics.reciprocal_rank(_rel())
    np.testing.assert_allclose(rr, [1.0, 0.0, 0.5])


def test_average_precision_normaliser_is_min_k_R():
    """AP@k must be attainable: with R >> k, dividing by R would cap the best
    possible score at k/R."""
    perfect = np.array([["A", "A", "A"]], dtype=object)
    ap = metrics.average_precision_at_k(
        metrics._relevance(perfect, np.array(["A"], dtype=object)), 3,
        np.array([500.0]))
    assert ap[0] == pytest.approx(1.0)


def test_average_precision_hand_computed():
    rel = _rel()
    ap = metrics.average_precision_at_k(rel, 3, np.array([10.0, 10.0, 10.0]))
    # q0: (1/1 + 2/3)/3 ; q1: 0 ; q2: (1/2 + 2/3)/3
    np.testing.assert_allclose(ap, [(1 + 2 / 3) / 3, 0.0, (0.5 + 2 / 3) / 3])


def test_ndcg_hand_computed():
    rel = _rel()
    nd = metrics.ndcg_at_k(rel, 3, np.array([10.0, 10.0, 10.0]))
    disc = 1 / np.log2(np.arange(2, 5))
    idcg = disc.sum()
    np.testing.assert_allclose(
        nd, [(disc[0] + disc[2]) / idcg, 0.0, (disc[1] + disc[2]) / idcg])


def test_ndcg_is_one_for_a_perfect_ranking():
    perfect = np.array([["A", "A", "A"]], dtype=object)
    nd = metrics.ndcg_at_k(
        metrics._relevance(perfect, np.array(["A"], dtype=object)), 3,
        np.array([3.0]))
    assert nd[0] == pytest.approx(1.0)


def test_macro_and_micro_differ_under_class_imbalance():
    # Two classes, wildly different query counts, different per-class accuracy.
    nl = np.array([["X"], ["X"], ["X"], ["X"], ["Y"]], dtype=object)
    tl = ["X", "X", "X", "X", "Y"]
    out = metrics.compute_all(nl, tl, k_values=(1,))
    assert out["micro_hit_at_1"] == pytest.approx(1.0)
    assert out["macro_hit_at_1"] == pytest.approx(1.0)
    # Now make the rare class wrong: micro barely moves, macro halves.
    nl2 = np.array([["X"], ["X"], ["X"], ["X"], ["X"]], dtype=object)
    out2 = metrics.compute_all(nl2, tl, k_values=(1,))
    assert out2["micro_hit_at_1"] == pytest.approx(0.8)
    assert out2["macro_hit_at_1"] == pytest.approx(0.5)


def test_compute_all_skips_denominator_metrics_without_counts():
    out = metrics.compute_all(NL, TL, k_values=(1, 3))
    assert "micro_hit_at_3" in out
    assert "micro_recall_at_3" not in out, \
        "recall must be skipped rather than computed against a wrong denominator"


def test_compute_all_skips_k_beyond_retrieval_depth():
    out = metrics.compute_all(NL, TL, k_values=(1, 3, 10), library_class_counts=COUNTS)
    assert "micro_hit_at_3" in out
    assert "micro_hit_at_10" not in out
    assert out["retrieval_depth"] == 3


def test_relevance_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="n_queries mismatch"):
        metrics._relevance(NL, np.array(["A", "A"], dtype=object))


# ------------------------------------------------------------------ paths ----

def test_paths_are_env_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("COMET_ROOT", str(tmp_path / "fg"))
    assert paths.comet_root() == tmp_path / "fg"


def test_similarity_matrix_name_encodes_masking(monkeypatch, tmp_path):
    monkeypatch.setenv("COMET_ROOT", str(tmp_path))
    masked = paths.similarity_matrix("uni2", "ot_cosine", tumor_only=True)
    unmasked = paths.similarity_matrix("uni2", "ot_cosine", tumor_only=True,
                                       no_same_patient=False)
    assert masked.name.endswith("_tumor_only_no_same_patient.pkl")
    assert unmasked.name.endswith("_tumor_only.pkl")
    assert masked != unmasked


def test_require_raises_with_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="COMET_ROOT"):
        paths.require(tmp_path / "nope.pkl", "test artifact")


# ------------------------------------------------------------------- hpc -----

def test_array_chunking_respects_cluster_cap():
    assert queues.chunk_array(2921) == [(1, 2921)]
    assert queues.chunk_array(4307) == [(1, 4000), (4001, 4307)]
    assert sum(hi - lo + 1 for lo, hi in queues.chunk_array(4307)) == 4307


def test_pressure_ranks_a_stalled_queue_worst():
    empty = queues.Queue("q_empty", prio=20, pend=0, run=100)
    busy = queues.Queue("q_busy", prio=100, pend=26456, run=1027)
    stalled = queues.Queue("q_stalled", prio=100, pend=500, run=0)
    assert empty.pressure < busy.pressure < stalled.pressure
    assert stalled.pressure == float("inf")


def test_an_idle_queue_with_a_trivial_backlog_is_quiet_not_stalled():
    """The distinction that decides where a whole sweep lands.

    Reading ``run == 0`` as "stalled" regardless of the backlog ranked ``short``
    -- reliably the emptiest CPU queue on this cluster -- dead last whenever it
    was observed at PEND 2 / RUN 0, which is the *best* state a queue can be in.
    Measured live on 2026-08-27: short PEND 2 / RUN 0 against standard PEND
    34,175 / RUN 9,552. Sending the sweep to standard on that reading would have
    been precisely backwards.
    """
    quiet = queues.Queue("short", prio=100, pend=2, run=0)
    backed_up = queues.Queue("standard", prio=50, pend=34175, run=9552)
    assert quiet.pressure == 0.0
    assert quiet.pressure < backed_up.pressure
    # The boundary is explicit so a future edit cannot drift it silently.
    assert queues.Queue("q", pend=queues.STALL_PEND, run=0).pressure == 0.0
    assert queues.Queue("q", pend=queues.STALL_PEND + 1, run=0).pressure == float("inf")


def test_a_short_job_can_reach_the_empty_short_queue():
    """rank() must not offer large_core_count to a job too small for it."""
    small = submit.Resources(slots=4, mem_mb_per_slot=4096, wall_minutes=20)
    names = [q.name for q in queues.rank(kind="cpu",
                                        wall_minutes=small.wall_minutes,
                                        slots=small.slots,
                                        total_mem_gb=small.total_reserved_gb)]
    assert "large_core_count" not in names
    assert "large_mem" not in names


def test_large_mem_needs_a_real_memory_requirement():
    """A 32-slot job with modest memory must not be able to dodge into large_mem."""
    modest = submit.Resources(slots=32, mem_mb_per_slot=2048, wall_minutes=120)
    assert modest.total_reserved_gb == pytest.approx(64.0)
    names = [q.name for q in queues.rank(kind="cpu", wall_minutes=120, slots=32,
                                         total_mem_gb=modest.total_reserved_gb)]
    assert "large_mem" not in names


def test_any_hit_and_generous_majority_agree_at_k1():
    """C2 fidelity check (iii): a known-answer test on the relabelling.

    A single neighbour cannot tie, so the two published tie-break rules must
    coincide exactly at k=1 -- and the archived JSONs from the two separate CI
    directories do both report 0.769514542.
    """
    gm = cli.KNOWN_MACRO_F1[("generous_majority", False)]
    ah = cli.KNOWN_MACRO_F1[("any_hit", False)]
    assert gm[1] == ah[1] == 0.769515


def test_any_hit_is_monotone_in_k_by_construction():
    """hit@k cannot decrease as k grows; if it does, it is not hit@k."""
    ah = cli.KNOWN_MACRO_F1[("any_hit", False)]
    vals = [ah[k] for k in (1, 3, 5, 10)]
    assert vals == sorted(vals)
    # generous_majority is NOT monotone -- it peaks at k=3 and falls. Reporting
    # the two side by side without saying so is what R4.b objects to.
    gm = cli.KNOWN_MACRO_F1[("generous_majority", False)]
    assert gm[3] > gm[5] > gm[10]


def test_memory_is_per_slot_in_reserved_total():
    """The over-reservation this module exists to prevent."""
    r = submit.Resources(slots=64, mem_mb_per_slot=8192)
    assert r.total_reserved_gb == pytest.approx(512.0)


def test_right_size_cuts_the_measured_over_reservation():
    r = submit.Resources(slots=64, mem_mb_per_slot=8192)
    ep = submit.parse_epilogue(
        "Run time : 269 sec.\n CPU time : 899.00 sec.\n"
        " Max Memory : 3827 MB\n Total Requested Memory : 524288 MB\n"
        " Max Processes : 11\n Max Threads : 594\n")
    assert ep["max_mem_mb"] == 3827
    assert ep["effective_cores"] == pytest.approx(899 / 269)
    sized = submit.right_size(r, ep)
    assert sized.total_reserved_gb < r.total_reserved_gb / 8
    assert sized.slots < r.slots


def test_right_size_never_raises_slot_count():
    r = submit.Resources(slots=4, mem_mb_per_slot=4096)
    ep = {"max_mem_mb": 1000.0, "effective_cores": 32.0}
    assert submit.right_size(r, ep).slots <= 4


def test_parse_epilogue_normalises_gb_to_mb():
    ep = submit.parse_epilogue("Max Memory : 4 GB\n")
    assert ep["max_mem_mb"] == 4096


def test_fan_out_refuses_to_submit_without_a_canary():
    with pytest.raises(RuntimeError, match="requires canary_epilogue"):
        submit.fan_out("cmd", n=10, job_name="x",
                       res=submit.Resources(), log_dir=None, dry_run=False)


# ------------------------------------------------- published-config coupling ---
# These lock in a real error caught by audit on 2026-08-27, before G2 ran. The
# spec had attached Table S3's macro F1 values to the tumour-only matrix and to
# n_slides=2766. All three were wrong together: Table S3 is the all-tissue arm
# (n=2778, dropped_labels=['MISC'], NORMAL retained as one of 15 classes), the
# tumour-only arm is n=2604 with ['MISC','NORMAL'] and 14 classes, and 2766 is
# the row count of the tumour-only distance matrix, not of any CI evaluation.

from prosper.evaluation import cli  # noqa: E402


def test_two_published_populations_are_distinct():
    assert cli.PUBLISHED_N_SLIDES[False] == 2778   # all-tissue, Table S3
    assert cli.PUBLISHED_N_SLIDES[True] == 2604    # tumour-only
    # 2604 = 2778 - 174 NORMAL. The populations differ by exactly the NORMAL class.
    assert cli.PUBLISHED_N_SLIDES[False] - cli.PUBLISHED_N_SLIDES[True] == 174


def test_drop_labels_are_coupled_to_population():
    """The published runs couple drop_labels to the population; a free-standing
    default silently evaluates the wrong cohort."""
    assert cli.PUBLISHED_DROP_LABELS[False] == ["MISC"]
    assert cli.PUBLISHED_DROP_LABELS[True] == ["MISC", "NORMAL"]
    assert "NORMAL" not in cli.PUBLISHED_DROP_LABELS[False], \
        "Table S3 retains NORMAL: its per-class support is 174 in all 132 rows"


def test_drop_labels_default_is_not_hardcoded():
    """--drop-labels must default to None so run() can resolve it per population."""
    a = cli.build_parser().parse_args(["--model", "uni2", "--method", "ot_cosine"])
    assert a.drop_labels is None


def test_table_s3_targets_belong_to_the_all_tissue_arm():
    assert cli.REGRESSION_MACRO_F1[False] == {
        1: 0.769515, 3: 0.821829, 5: 0.791156, 10: 0.765022}
    assert cli.TABLE_S3_UNI2_OT_MACRO_F1 is cli.REGRESSION_MACRO_F1[False]
    # The tumour-only arm has different values and must not be confused with them.
    for k in (1, 3, 5, 10):
        assert cli.REGRESSION_MACRO_F1[True][k] != cli.REGRESSION_MACRO_F1[False][k]


def test_2766_is_not_a_ci_population():
    """2,766 is the tumour-only distance-matrix row count (Tables S10-S13),
    not a CI n_slides. Asserting it in a gate is a false target."""
    assert 2766 not in cli.PUBLISHED_N_SLIDES.values()


def test_published_config_no_longer_carries_a_drop_labels_key():
    assert "drop_labels" not in cli.PUBLISHED, \
        "drop_labels is population-dependent; keep it in PUBLISHED_DROP_LABELS"
