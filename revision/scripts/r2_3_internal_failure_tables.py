"""R2.3 complete all-query, non-oracle failure inventory for the retained primary configuration.

Configuration: UNI2 encoder / Optimal Transport cosine / unfiltered / 2,778 COMET slides /
15 groups incl. NORMAL (MISC dropped) / same-patient masked. Reads the existing depth-10 ranked
lists (no new retrieval) and writes per-query, confusion and per-group tables, plus a reconciliation
of the legacy generous rule against earlier first-pass summaries (exit 1 on any mismatch).

Decision rules (see RULES, also written to reconciliation.json):
  top1                 label of the rank-1 neighbour.
  plurality@k          most frequent label among the top-k (k=3, 5 primary, 10); ties broken by the
                       closest-ranked neighbour among the tied labels. Never uses the truth.
  strict_majority@5    label with count > k/2 (>= 3 of 5) among the top-5, else 'ABSTAIN'.
  legacy_generous@5    oracle rule of scripts/failure_modes_first_pass.py (vote_k): correct if the truth is
                       among the tied-max labels of the top-5; otherwise pred = closest-ranked neighbour whose
                       label is among the tied-max labels. Reconciliation only.
  hit@k                any top-k neighbour label == truth (k=1,3,5,10).
  precision@k          fraction of the top-k with label == truth (k=5,10).
  first_relevant_rank  rank (1..10) of the first neighbour with label == truth, else 'absent@10'.
  library_support      slides in the metadata with the query's Tumor.group and a different UID.Subject
                       (MISC excluded from the library).
  n_patients_top5      distinct UID.Subject among the top-5 neighbours.
"""
import argparse, datetime, hashlib, json, os, sys
from collections import Counter

import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('PROSPER_OUT', os.path.join(R, 'prosper_revision_output'))
META = ('/PROSPER_WORKSPACE/comet-feature-gen/comet-analysis/'
        'data/final_scan_id_disease_map_with_sample_id_patient_id.csv')
CLASSES = ['ACT/ACC', 'DSRCT', 'EWS', 'LIVER', 'MEL', 'MPNST', 'MRT', 'NB', 'NF', 'NORMAL',
           'OS', 'RB', 'RMS', 'SS', 'WT']
DEPTH = 10
RULES = {
    'top1': 'label of the rank-1 neighbour',
    'plurality_k': 'most frequent label among top-k (k=3,5,10; k=5 primary); ties broken by the closest-ranked '
                   'neighbour among the tied labels; never uses the truth',
    'strict_majority_k5': "label with count > k/2 (>=3 of 5) among top-5, else 'ABSTAIN'",
    'legacy_generous_k5': 'ORACLE, reconciliation only: correct if truth is among the tied-max labels of the top-5; '
                          'otherwise pred = closest-ranked neighbour whose label is among the tied-max labels '
                          '(scripts/failure_modes_first_pass.py vote_k)',
    'hit_k': 'any top-k neighbour label == truth (k=1,3,5,10)',
    'precision_k': 'fraction of top-k neighbours with label == truth (k=5,10)',
    'first_relevant_rank': "rank 1..10 of first neighbour with label == truth, else 'absent@10'",
    'library_support': 'metadata slides with the same Tumor.group and a different UID.Subject than the query; '
                       'MISC excluded from the library',
    'n_patients_top5': 'distinct UID.Subject among the top-5 neighbours',
}


def plurality(labels, k):
    """Most frequent label in labels[:k]; tie -> closest-ranked label among the tied ones."""
    c = Counter(labels[:k]); mx = max(c.values())
    return next(l for l in labels[:k] if c[l] == mx)


def strict_majority(labels, k=5):
    c = Counter(labels[:k]); lab, n = c.most_common(1)[0]
    return lab if n > k / 2 else 'ABSTAIN'


def legacy_generous(labels, truth, k=5):
    """Legacy oracle vote_k: (correct, pred)."""
    c = Counter(labels[:k]); mx = max(c.values()); tied = {l for l, n in c.items() if n == mx}
    if truth in tied:
        return True, truth
    return False, next(l for l in labels[:k] if l in tied)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run_checks(ranks, meta, expected_n=2778, expected_labels=CLASSES):
    """Raise ValueError on any structural violation; return a dict of the checks that passed."""
    if meta['Slide.Scan.File'].duplicated().any():
        raise ValueError('metadata: a Slide.Scan.File maps to more than one row/UID.Subject')
    if meta['UID.Subject'].isna().any():
        raise ValueError('metadata: missing UID.Subject')
    nq = ranks['query'].nunique()
    if nq != expected_n:
        raise ValueError(f'expected {expected_n} distinct queries, found {nq}')
    rk = ranks.groupby('query')['rank'].apply(lambda s: sorted(s.tolist()))
    bad = rk[rk.apply(lambda s: s != list(range(1, DEPTH + 1)))]
    if len(bad):
        raise ValueError(f'{len(bad)} queries lack ranks 1..{DEPTH} exactly once, e.g. {bad.index[0]}')
    m = meta.set_index('Slide.Scan.File')
    ids = set(ranks['query']) | set(ranks['neighbour'])
    missing = ids - set(m.index)
    if missing:
        raise ValueError(f'{len(missing)} query/neighbour IDs absent from metadata, e.g. {sorted(missing)[:3]}')
    # library_support counts non-MISC metadata slides, so the library must be exactly the query set
    qset = set(ranks['query'])
    libset = set(meta.loc[meta['Tumor.group'] != 'MISC', 'Slide.Scan.File'])
    if libset != qset:
        raise ValueError(f'non-MISC metadata slides != query slides ({len(libset - qset)} metadata-only, '
                         f'{len(qset - libset)} query-only)')
    outside = set(ranks['neighbour']) - qset
    if outside:
        raise ValueError(f'{len(outside)} neighbour slides are not in the query/library set, e.g. {sorted(outside)[:3]}')
    if (ranks['true_label'] != ranks['query'].map(m['Tumor.group'])).any():
        raise ValueError('true_label disagrees with metadata Tumor.group for some query')
    if (ranks['neighbour_label'] != ranks['neighbour'].map(m['Tumor.group'])).any():
        raise ValueError('neighbour_label disagrees with metadata Tumor.group for some neighbour')
    labs = set(ranks['true_label']) | set(ranks['neighbour_label'])
    if 'MISC' in labs:
        raise ValueError('MISC present in ranked lists')
    if expected_labels is not None and labs != set(expected_labels):
        raise ValueError(f'label set {sorted(labs)} != expected {sorted(expected_labels)}')
    same = ranks['query'].map(m['UID.Subject']) == ranks['neighbour'].map(m['UID.Subject'])
    if same.any():
        raise ValueError(f'{int(same.sum())} neighbours share UID.Subject with their query (same-patient masking violated)')
    return dict(same_patient_pairs_checked=int(len(ranks)), same_patient_pairs_found=0,
                non_misc_metadata_slides_equal_query_slides=True, n_library_slides=len(libset),
                all_neighbours_in_query_slides=True, n_distinct_neighbours=ranks['neighbour'].nunique())


def build_frame(ranks, meta):
    """One row per query with all decision-rule outputs."""
    m = meta.set_index('Slide.Scan.File')
    pat = m['UID.Subject']
    lib = meta[meta['Tumor.group'] != 'MISC']
    rows = []
    for q, g in ranks.sort_values(['query', 'rank']).groupby('query', sort=True):
        truth = g['true_label'].iat[0]; labs = g['neighbour_label'].tolist(); nb = g['neighbour'].tolist()
        p = pat[q]
        rel = [l == truth for l in labs]
        first = next((i + 1 for i, r in enumerate(rel) if r), None)
        gen_ok, gen_pred = legacy_generous(labs, truth)
        rows.append(dict(
            query=q, true_label=truth, patient=p,
            library_support=int(((lib['Tumor.group'] == truth) & (lib['UID.Subject'] != p)).sum()),
            top1=labs[0], plurality_k3=plurality(labs, 3), plurality_k5=plurality(labs, 5),
            plurality_k10=plurality(labs, 10), strict_majority_k5=strict_majority(labs),
            legacy_generous_k5_correct=int(gen_ok), legacy_generous_k5_pred=gen_pred,
            hit1=int(any(rel[:1])), hit3=int(any(rel[:3])), hit5=int(any(rel[:5])), hit10=int(any(rel[:10])),
            precision5=sum(rel[:5]) / 5, precision10=sum(rel[:10]) / 10,
            first_relevant_rank=str(first) if first else 'absent@10',
            n_patients_top5=int(pat.reindex(nb[:5]).nunique()),
            top5_labels='|'.join(labs[:5]), top5_neighbours='|'.join(map(str, nb[:5])),
            top1_distance=g['distance'].iat[0]))
    return pd.DataFrame(rows)


def ladder(fr):
    r = fr['first_relevant_rank'].map(lambda x: 99 if x == 'absent@10' else int(x))
    return pd.cut(r, [0, 1, 5, 10, 100], labels=['rank1', 'rank2_5', 'rank6_10', 'absent'])


def confusion(fr, col, classes, extra=()):
    return pd.crosstab(pd.Categorical(fr['true_label'], classes),
                       pd.Categorical(fr[col], list(classes) + list(extra)),
                       rownames=['true_label'], colnames=['predicted'], dropna=False)


def destinations(preds, n=3):
    """Top-n wrong destinations; all labels tied with the n-th count are kept."""
    c = pd.Series(preds).value_counts()
    if not len(c):
        return ''
    c = c.reset_index(); c.columns = ['lab', 'n']; c = c.sort_values(['n', 'lab'], ascending=[False, True])
    cut = c['n'].iloc[min(n, len(c)) - 1]
    return '; '.join(f'{a} {b}' for a, b in zip(c['lab'], c['n']) if b >= cut)


def summarise(g):
    lad = ladder(g).value_counts()
    wrong = g.loc[g['plurality_k5'] != g['true_label'], 'plurality_k5']
    return dict(
        n_queries=len(g), n_patients=g['patient'].nunique(), median_library_support=float(g['library_support'].median()),
        hit1=g['hit1'].mean(), hit3=g['hit3'].mean(), hit5=g['hit5'].mean(), hit10=g['hit10'].mean(),
        precision5=g['precision5'].mean(), precision10=g['precision10'].mean(),
        plurality5_recall=(g['plurality_k5'] == g['true_label']).mean(),
        n_plurality5_wrong=len(wrong),
        strict5_recall=(g['strict_majority_k5'] == g['true_label']).mean(),
        strict5_abstain_rate=(g['strict_majority_k5'] == 'ABSTAIN').mean(),
        legacy_generous5_recall=g['legacy_generous_k5_correct'].mean(),
        ladder_rank1=int(lad['rank1']), ladder_rank2_5=int(lad['rank2_5']),
        ladder_rank6_10=int(lad['rank6_10']), ladder_absent=int(lad['absent']),
        top3_plurality5_destinations=destinations(wrong))


def reconcile(fr, ladder_csv, pertype_csv, c6_csv):
    """Compare the legacy generous rule with three earlier summaries; returns list of check dicts."""
    out = []
    def add(target, key, exp, got, n=None, note=None):
        d = dict(target=target, key=key, expected=exp, computed=got)
        if n is None:
            d['match'] = bool(exp == got)
        else:
            # proportions: match on the implied integer count (legacy CSV floats can differ by 1 ULP)
            d.update(expected_count=int(round(exp * n)), computed_count=int(round(got * n)), abs_diff=abs(exp - got))
            d['match'] = bool(d['expected_count'] == d['computed_count'] and d['abs_diff'] < 1e-12)
        if note:
            d['note'] = note
        out.append(d)
    # legacy order: queries sorted as strings, then grouped by truth
    leg = fr.assign(_q=fr['query'].astype(str)).sort_values('_q')
    lad = ladder(leg).value_counts()
    L = pd.read_csv(ladder_csv).set_index('cohort').loc['COMET']
    got = {'n': len(leg), 'rank_1': int(lad['rank1']), 'rank_2-5': int(lad['rank2_5']),
           'rank_6-10': int(lad['rank6_10']), 'rank_absent': int(lad['absent']),
           'wrong_at_k5': int((leg['legacy_generous_k5_correct'] == 0).sum()),
           'hit1': leg['hit1'].mean(), 'hit5': leg['hit5'].mean(), 'hit10': leg['hit10'].mean()}
    for k, v in got.items():
        if k.startswith('hit'):
            add('severity_ladder', k, float(L[k]), float(v), n=len(leg))
        else:
            add('severity_ladder', k, int(L[k]), v)
    P = pd.read_csv(pertype_csv); P = P[P['cohort'] == 'COMET'].set_index('tumour_type')
    for t, g in leg.groupby('true_label'):
        wr = g[g['legacy_generous_k5_correct'] == 0]; top = wr['legacy_generous_k5_pred'].value_counts()
        e = P.loc[t]
        add('per_tumour_type', f'{t}:n', int(e['n']), len(g))
        add('per_tumour_type', f'{t}:wrong', int(e['wrong']), len(wr))
        tied = sorted(top[top == top.iloc[0]].index) if len(top) else []
        note = None
        if len(tied) > 1:
            note = (f"legacy value was a tie ({'/'.join(tied)} {int(top.iloc[0])} each); this match reflects reproduced "
                    f"row order, not a meaningful agreement; top_wrong_n is the real match")
        add('per_tumour_type', f'{t}:top_wrong_call', str(e['top_wrong_call']), top.index[0] if len(top) else '', note=note)
        add('per_tumour_type', f'{t}:top_wrong_n', int(e['top_wrong_n']), int(top.iloc[0]) if len(top) else 0)
    C = pd.read_csv(c6_csv); C = C[C['k'] == 5].set_index('class')
    for t, g in leg.groupby('true_label'):
        add('c6_confusion_pairs_k5', f'{t}:n', int(C.loc[t, 'n']), len(g))
        add('c6_confusion_pairs_k5', f'{t}:n_wrong', int(C.loc[t, 'n_wrong']),
            int((g['legacy_generous_k5_correct'] == 0).sum()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', default=os.path.join(OUT, 'r2_3_internal_failure_' + datetime.date.today().strftime('%Y%m%d')))
    ap.add_argument('--ranks', default=os.path.join(OUT, 'c2_metric_suite/ranked_lists_uni2_ot_cosine.csv.gz'))
    ap.add_argument('--metadata', default=META)
    ap.add_argument('--metrics-json', default=os.path.join(OUT, 'c2_metric_suite/metrics_uni2_ot_cosine.json'))
    ap.add_argument('--ladder-csv', default=os.path.join(R, 'docs/results/failure_modes_first_pass/severity_ladder.csv'))
    ap.add_argument('--pertype-csv', default=os.path.join(R, 'docs/results/failure_modes_first_pass/per_tumour_type_error_rates.csv'))
    ap.add_argument('--c6-csv', default=os.path.join(OUT, 'c6_confusion_pairs_primary.csv'))
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    ranks = pd.read_csv(a.ranks)
    meta = pd.read_csv(a.metadata)
    struct = run_checks(ranks, meta)
    fr = build_frame(ranks, meta)
    fr.to_csv(os.path.join(a.out_dir, 'all_query_frame.csv'), index=False)
    confusion(fr, 'plurality_k5', CLASSES).to_csv(os.path.join(a.out_dir, 'confusion_k5_plurality.csv'))
    confusion(fr, 'strict_majority_k5', CLASSES, ['ABSTAIN']).to_csv(os.path.join(a.out_dir, 'confusion_k5_strict.csv'))
    confusion(fr, 'top1', CLASSES).to_csv(os.path.join(a.out_dir, 'confusion_k1_top1.csv'))
    # median library support is per-class only; a pooled median mixes classes
    pg = pd.DataFrame([dict(true_label=t, **summarise(g)) for t, g in fr.groupby('true_label')]
                      + [{'true_label': 'ALL', **summarise(fr), 'median_library_support': float('nan')}])
    pg.to_csv(os.path.join(a.out_dir, 'per_group_summary.csv'), index=False)

    checks = reconcile(fr, a.ladder_csv, a.pertype_csv, a.c6_csv)
    tab = pd.DataFrame(checks)
    with pd.option_context('display.max_rows', None, 'display.width', 200, 'display.precision', 17):
        print(tab.to_string(index=False))
    cfg = json.load(open(a.metrics_json))
    inputs = {k: dict(path=os.path.abspath(p), sha256=sha256(p)) for k, p in
              [('ranks', a.ranks), ('metadata', a.metadata), ('metrics_json', a.metrics_json),
               ('ladder_csv', a.ladder_csv), ('pertype_csv', a.pertype_csv), ('c6_csv', a.c6_csv)]}
    inputs['ranks']['rows'] = len(ranks); inputs['metadata']['rows'] = len(meta)
    rep = dict(
        timestamp=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        script=dict(path=os.path.abspath(__file__), sha256=sha256(os.path.abspath(__file__))),
        command=' '.join([sys.executable] + sys.argv),
        inputs=inputs, metrics_json_config=cfg.get('config'),
        metrics_json_ranked_lists_note=(f"metrics_json 'ranked_lists' points to {cfg.get('ranked_lists')} (old sibling root); "
                                        f"the bound ranks file above was used instead"),
        structural_checks=dict(n_queries=int(fr.shape[0]), depth=DEPTH, labels=sorted(fr['true_label'].unique()), **struct),
        rules=RULES, reconciliation=[{k: (v.item() if hasattr(v, 'item') else v) for k, v in c.items()} for c in checks],
        all_match=bool(tab['match'].all()))
    with open(os.path.join(a.out_dir, 'reconciliation.json'), 'w') as f:
        json.dump(rep, f, indent=2)
    with open(os.path.join(a.out_dir, 'README.md'), 'w') as f:
        f.write(f"""# R2.3 internal failure tables (UNI2 / OT cosine / unfiltered / 2,778 COMET / 15 groups / same-patient masked)
- all_query_frame.csv: one row per query with top1, plurality@3/5/10, strict-majority@5, legacy generous@5, hit@k, precision@k, first relevant rank, library support.
- confusion_k5_plurality.csv: truth x plurality@5 prediction counts (15x15; row sums = class n).
- confusion_k5_strict.csv: truth x strict-majority@5 prediction counts plus ABSTAIN column.
- confusion_k1_top1.csv: truth x top-1 label counts (15x15).
- per_group_summary.csv: per true label plus micro 'ALL' row; point estimates only, no confidence intervals. median_library_support is per-class only (blank in ALL: a pooled median mixes classes). n_plurality5_wrong = wrong plurality@5 calls (plurality never abstains), the base for top3_plurality5_destinations.
- reconciliation.json: legacy generous@5 vs earlier summaries, input SHA256s, metrics config, rule definitions, structural checks.
- Decision rules are defined in the script docstring and reconciliation.json 'rules'.
- Plurality and strict-majority rules never use the truth; legacy_generous is an oracle rule kept only for reconciliation.
- Unit of analysis is the slide query: {fr.shape[0]} queries from {fr['patient'].nunique()} patients. Slides of one patient are not independent, so per-class rates are slide-level, not patient-level.
- The metrics JSON 'ranked_lists' path points to the old sibling output root, where the file no longer exists; no byte-identity check against it was possible. That the bound ranks file is the same ranking rests on the 99 legacy reconciliations in reconciliation.json.

Command: `{rep['command']}` (PROSPER_OUT={OUT})
""")
    if not rep['all_match']:
        print(f"RECONCILIATION FAILED: {int((~tab['match']).sum())} mismatches", file=sys.stderr)
        return 1
    print(f"all {len(tab)} reconciliation checks match; outputs in {a.out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
