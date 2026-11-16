import os, sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import r2_3_internal_failure_tables as T  # noqa: E402

# 12 slides, 3 labels (A, B, C), 4 patients; plus one MISC slide outside the library
META = pd.DataFrame({
    'Slide.Scan.File': list(range(1, 13)) + [99],
    'Tumor.group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'A', 'B', 'C', 'MISC'],
    'Sample.ID': [f's{i}' for i in range(13)],
    'UID.Subject': ['P1', 'P1', 'P2', 'P2', 'P3', 'P3', 'P4', 'P4', 'P1', 'P3', 'P4', 'P2', 'P2'],
})
LAB = META.set_index('Slide.Scan.File')['Tumor.group']
PAT = META.set_index('Slide.Scan.File')['UID.Subject']
LABELS = ['A', 'B', 'C']


def make_ranks(neigh):
    """neigh: {query: [10 neighbour IDs in rank order]}."""
    rows = [dict(query=q, true_label=LAB[q], rank=r + 1, neighbour=n, neighbour_label=LAB[n], distance=0.1 * (r + 1))
            for q, ns in neigh.items() for r, n in enumerate(ns)]
    return pd.DataFrame(rows)


# q1 (A,P1): top5 B B A A C -> A/B tie, closest B: plurality wrong, generous correct, strict ABSTAIN
# q4 (B,P2): top5 B B B A C -> strict B
# q7 (C,P4): no C in top10 -> absent@10; top5 A A A B B -> generous wrong, pred A
# q3 (A,P2): top5 C A C A B -> C/A tie, closest C; generous correct
# q5 (B,P3): top5 B A A A B -> top1 B, plurality A
# q10 (A,P3): top5 A A A A B
NEIGH = {
    1: [4, 5, 3, 10, 7, 6, 11, 12, 4, 5],
    4: [5, 6, 11, 1, 7, 2, 10, 8, 9, 1],
    7: [1, 2, 3, 4, 5, 6, 10, 1, 2, 3],
    3: [7, 1, 8, 10, 5, 2, 5, 6, 9, 11],
    5: [4, 1, 2, 3, 11, 12, 7, 8, 9, 2],
    10: [1, 2, 3, 1, 4, 2, 3, 7, 8, 9],
    # remaining slides are also queries, so the non-MISC library equals the query set
    2: [3, 10, 4, 5, 6, 7, 8, 11, 12, 3],
    6: [4, 11, 1, 2, 3, 7, 8, 9, 12, 4],
    8: [9, 12, 1, 2, 3, 4, 5, 6, 10, 9],
    9: [7, 8, 12, 3, 4, 5, 6, 10, 11, 7],
    11: [4, 5, 6, 1, 2, 3, 9, 10, 12, 4],
    12: [7, 8, 9, 1, 2, 5, 6, 10, 11, 7],
}
N = len(NEIGH)


@pytest.fixture
def frame():
    r = make_ranks(NEIGH)
    s = T.run_checks(r, META, expected_n=N, expected_labels=LABELS)
    assert s['same_patient_pairs_checked'] == 10 * N and s['n_library_slides'] == 12
    return T.build_frame(r, META).set_index('query')


def test_plurality_tiebreak_vs_generous(frame):
    q = frame.loc[1]
    assert q.plurality_k5 == 'B' and q.plurality_k3 == 'B'
    assert q.legacy_generous_k5_correct == 1 and q.legacy_generous_k5_pred == 'A'
    assert q.strict_majority_k5 == 'ABSTAIN'
    q3 = frame.loc[3]
    assert q3.plurality_k5 == 'C' and q3.legacy_generous_k5_correct == 1
    assert frame.loc[5].top1 == 'B' and frame.loc[5].plurality_k5 == 'A'


def test_strict_and_generous_wrong(frame):
    assert frame.loc[4].strict_majority_k5 == 'B'
    assert frame.loc[10].strict_majority_k5 == 'A' and frame.loc[10].precision5 == 0.8
    q = frame.loc[7]
    assert q.legacy_generous_k5_correct == 0 and q.legacy_generous_k5_pred == 'A'


def test_first_relevant_and_hits(frame):
    q = frame.loc[7]
    assert q.first_relevant_rank == 'absent@10' and q.hit10 == 0 and q.precision10 == 0
    q1 = frame.loc[1]
    assert q1.first_relevant_rank == '3' and q1.hit1 == 0 and q1.hit3 == 1
    assert T.ladder(frame.reset_index()).tolist().count('absent') == 1


def test_library_support_and_patients(frame):
    # A: 1(P1) 2(P1) 3(P2) 10(P3); MISC slide 99 (P2) is excluded
    assert frame.loc[1].library_support == 2
    assert frame.loc[3].library_support == 3
    # C: 7(P4) 8(P4) 9(P1) 12(P2)
    assert frame.loc[7].library_support == 2
    assert frame.loc[1].n_patients_top5 == 3


def test_same_patient_raises():
    bad = dict(NEIGH); bad[1] = [2] + NEIGH[1][1:]    # slide 2 belongs to P1, as does query 1
    with pytest.raises(ValueError, match='same-patient'):
        T.run_checks(make_ranks(bad), META, expected_n=N, expected_labels=LABELS)


def test_structural_checks():
    r = make_ranks(NEIGH)
    with pytest.raises(ValueError, match='ranks 1..10'):
        T.run_checks(r[~((r['query'] == 4) & (r['rank'] == 7))], META, expected_n=N, expected_labels=None)
    r2 = r.copy(); r2.loc[0, 'neighbour_label'] = 'C'
    with pytest.raises(ValueError, match='neighbour_label'):
        T.run_checks(r2, META, expected_n=N, expected_labels=None)
    with pytest.raises(ValueError, match='distinct queries'):
        T.run_checks(r, META, expected_n=N + 1, expected_labels=None)


def test_library_equals_query_set():
    r = make_ranks(NEIGH)
    with pytest.raises(ValueError, match='non-MISC metadata slides != query slides'):
        T.run_checks(r[r['query'] != 12], META, expected_n=N - 1, expected_labels=None)
    r2 = r.copy(); r2.loc[0, ['neighbour', 'neighbour_label']] = [99, 'MISC']   # MISC slide 99 (P2) as a neighbour
    with pytest.raises(ValueError, match='not in the query/library set'):
        T.run_checks(r2, META, expected_n=N, expected_labels=None)


def test_plurality_tiebreak_closest_not_alphabetical():
    assert T.plurality(['NB', 'RMS', 'EWS', 'RMS', 'EWS'], 5) == 'RMS'
    assert T.plurality(['Z', 'A', 'A', 'Z', 'B'], 5) == 'Z'


def test_summary_all_row_fields(frame):
    s = T.summarise(frame.reset_index())
    assert s['n_plurality5_wrong'] == int((frame.plurality_k5 != frame.true_label).sum())


def test_destinations_keeps_ties():
    assert T.destinations(['R'] * 15 + ['E'] * 4 + ['W'] * 2 + ['N'] * 2 + ['X']) == 'R 15; E 4; N 2; W 2'
    assert T.destinations([]) == ''
