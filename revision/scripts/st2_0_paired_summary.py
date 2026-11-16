"""ST2.0 closure — paired whole-slide vs tumour-only reading on the three external cohorts.

For each cohort (ccdi, stv, kidrare) and arm:
  tumor_only       query tumour-only vs the full library (MISC dropped)            -> row <COHORT>_TOQ
  tumor_only_both  query + library tumour-only (MISC + NORMAL dropped)             -> row <COHORT>_TO
the whole-slide frozen-primary per_query table (external_{ccdi_inscope,stv_labelled,kidrare_inscope}) is joined
to the tumour-only per_query table on the query id, restricted to the truth classes the evaluator scored
(STV's per_query also lists 70 MISC + 10 OOS truth rows, which the JSON excludes). NORMAL-truth queries are dropped from the _TO rows (the
tumour-only library has no NORMAL class). Metrics on the common queries: micro hit@1/5/10, generous-vote
accuracy at k=5 (micro and macro over truth classes), macro hit@5; fixed / broken at generous k=5; paired
bootstrap (B=5000, seed 123) of every difference, resampling patients with one slide per draw
(pps_cluster_one_slide, the frozen protocol) — patient = patient_id for STV, slide for CCDI and KidRare.

Inputs ($PROSPER_OUT): external_*/per_query_*_uni2_ot_cosine.csv, st2p2_external_<c>_<arm>/per_query_*.csv,
external_<c>/manifest_*.csv (patient column), the evaluate JSONs (job ids, n).
Outputs: $PROSPER_OUT/st2p2_paired/paired_summary.csv, fixed_broken_by_class.csv, and a Markdown copy in
[internal record]. Arms whose evaluate directory is absent are skipped and listed.
"""
import os, sys, json, csv
import numpy as np

OUT = os.environ.get('PROSPER_OUT', 'prosper_revision_output')
DST = os.path.join(OUT, 'st2p2_paired'); os.makedirs(DST, exist_ok=True)
DOC = 'docs/results/st2_0'; os.makedirs(DOC, exist_ok=True)
B, SEED = 5000, 123

COHORTS = {
    'ccdi':    dict(ws='external_ccdi_inscope',   manifest='external_ccdi/manifest_inscope.csv',   pcol=None),
    'stv':     dict(ws='external_stv_labelled',   manifest='external_stv/manifest_labelled.csv',   pcol='patient_id'),
    'kidrare': dict(ws='external_kidrare_inscope', manifest='external_kidrare/manifest_inscope.csv', pcol=None),
}
ARMS = {'tumor_only': 'TOQ', 'tumor_only_both': 'TO'}
METRICS = ['hit_k1', 'hit_k5', 'hit_k10', 'generous_correct_k5']


def read_pq(path):
    with open(path) as f:
        rows = {r['query']: r for r in csv.DictReader(f)}
    return rows


def read_json(d, c):
    p = os.path.join(OUT, d, f'external_{c}_uni2_ot_cosine.json')
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    return dict(n=j['n_queries'], job=j['config'].get('lsf_job'), commit=j['config'].get('prosper_commit'),
                truth_classes=set(j['query_class_counts']),
                micro5=j['point_estimates']['micro_hit_at_5'], macro5=j['point_estimates']['macro_hit_at_5'],
                macro5_ci=(j['confidence_intervals']['macro_hit_at_5']['lo'], j['confidence_intervals']['macro_hit_at_5']['hi']),
                acc5=j['voting_f1']['generous_k5']['accuracy'], f1_5=j['voting_f1']['generous_k5']['macro_f1'],
                f1_5_ci=(j['confidence_intervals']['generous_macro_f1_k5']['lo'], j['confidence_intervals']['generous_macro_f1_k5']['hi']))


def macro_acc(truth, correct, classes):
    vals = [correct[truth == c].mean() for c in classes if (truth == c).any()]
    return float(np.mean(vals))


def stats(truth, ws, to, classes):
    out = {}
    for m in METRICS:
        out[f'ws_{m}'] = ws[m].mean(); out[f'to_{m}'] = to[m].mean()
    out['ws_macro_acc5'] = macro_acc(truth, ws['generous_correct_k5'], classes)
    out['to_macro_acc5'] = macro_acc(truth, to['generous_correct_k5'], classes)
    out['ws_macro_hit5'] = macro_acc(truth, ws['hit_k5'], classes)
    out['to_macro_hit5'] = macro_acc(truth, to['hit_k5'], classes)
    return out


def main():
    summary, fb_rows, skipped = [], [], []
    for c, spec in COHORTS.items():
        ws_pq = read_pq(os.path.join(OUT, spec['ws'], f'per_query_{c}_uni2_ot_cosine.csv'))
        ws_js = read_json(spec['ws'], c)
        pat = {}
        if spec['pcol']:
            with open(os.path.join(OUT, spec['manifest'])) as f:
                for r in csv.DictReader(f):
                    pat[r['slide_id']] = r[spec['pcol']]
        for arm, tag in ARMS.items():
            d = f'st2p2_external_{c}_{arm}'
            p = os.path.join(OUT, d, f'per_query_{c}_uni2_ot_cosine.csv')
            if not os.path.exists(p):
                skipped.append(f'{c}/{arm}'); continue
            to_pq = read_pq(p); to_js = read_json(d, c)
            # the evaluator scores only queries whose truth is an evaluated class (STV's per_query also lists MISC/OOS truth)
            common = [q for q in ws_pq if q in to_pq and ws_pq[q]['true_label'] in ws_js['truth_classes']]
            if arm == 'tumor_only_both':
                common = [q for q in common if ws_pq[q]['true_label'] != 'NORMAL']
            n_ws_only = sum(1 for q in ws_pq if q not in to_pq and ws_pq[q]['true_label'] in ws_js['truth_classes']
                            and not (arm == 'tumor_only_both' and ws_pq[q]['true_label'] == 'NORMAL'))
            truth = np.array([ws_pq[q]['true_label'] for q in common])
            classes = sorted(set(truth))
            ws = {m: np.array([float(ws_pq[q][m]) for q in common]) for m in METRICS}
            to = {m: np.array([float(to_pq[q][m]) for q in common]) for m in METRICS}
            pt = stats(truth, ws, to, classes)
            wc, tc = ws['generous_correct_k5'] > 0.5, to['generous_correct_k5'] > 0.5
            fixed, broken = int((~wc & tc).sum()), int((wc & ~tc).sum())
            # per-class fixed / broken (what the tumour-only arm changed, by truth class)
            for k in classes:
                m = truth == k
                fb_rows.append(dict(cohort=c.upper(), arm=tag, truth=k, n=int(m.sum()), ws_wrong=int((~wc & m).sum()),
                                    to_wrong=int((~tc & m).sum()), fixed=int((~wc & tc & m).sum()), broken=int((wc & ~tc & m).sum())))
            # paired bootstrap, patients resampled, one slide per drawn patient
            pid = np.array([pat.get(q, q) for q in common])
            groups = {}
            for i, g in enumerate(pid):
                groups.setdefault(g, []).append(i)
            glist = list(groups.values()); G = len(glist)
            rng = np.random.default_rng(SEED)  # per arm, so adding an arm never shifts another arm's CI
            diffs = {k: [] for k in ['hit_k1', 'hit_k5', 'hit_k10', 'generous_correct_k5', 'macro_acc5', 'macro_hit5']}
            for _ in range(B):
                gi = rng.integers(0, G, G)
                idx = np.array([glist[g][rng.integers(0, len(glist[g]))] for g in gi])
                t = truth[idx]
                for m in METRICS:
                    diffs[m].append(to[m][idx].mean() - ws[m][idx].mean())
                diffs['macro_acc5'].append(macro_acc(t, to['generous_correct_k5'][idx], classes) - macro_acc(t, ws['generous_correct_k5'][idx], classes))
                diffs['macro_hit5'].append(macro_acc(t, to['hit_k5'][idx], classes) - macro_acc(t, ws['hit_k5'][idx], classes))
            row = dict(cohort=c.upper(), arm=tag, arm_dir=arm, n_paired=len(common), n_patients=G, n_ws_not_in_to=n_ws_only,
                       ws_job=ws_js['job'], to_job=to_js['job'] if to_js else None, to_commit=to_js['commit'] if to_js else None,
                       to_n_eval=to_js['n'] if to_js else None, fixed=fixed, broken=broken)
            for m, lab in [('hit_k1', 'hit1'), ('hit_k5', 'hit5'), ('hit_k10', 'hit10'), ('generous_correct_k5', 'acc5')]:
                row[f'ws_{lab}'] = round(pt[f'ws_{m}'], 4); row[f'to_{lab}'] = round(pt[f'to_{m}'], 4)
                lo, hi = np.percentile(diffs[m], [2.5, 97.5])
                row[f'd_{lab}'] = round(pt[f'to_{m}'] - pt[f'ws_{m}'], 4); row[f'd_{lab}_lo'] = round(lo, 4); row[f'd_{lab}_hi'] = round(hi, 4)
            for m in ['macro_acc5', 'macro_hit5']:
                row[f'ws_{m}'] = round(pt[f'ws_{m}'], 4); row[f'to_{m}'] = round(pt[f'to_{m}'], 4)
                lo, hi = np.percentile(diffs[m], [2.5, 97.5])
                row[f'd_{m}'] = round(pt[f'to_{m}'] - pt[f'ws_{m}'], 4); row[f'd_{m}_lo'] = round(lo, 4); row[f'd_{m}_hi'] = round(hi, 4)
            if to_js:
                row['to_json_macro_hit5'] = round(to_js['macro5'], 4); row['to_json_macro_hit5_ci'] = f"[{to_js['macro5_ci'][0]:.3f}, {to_js['macro5_ci'][1]:.3f}]"
                row['to_json_acc5'] = round(to_js['acc5'], 4); row['to_json_f1_5'] = round(to_js['f1_5'], 4)
                row['to_json_f1_5_ci'] = f"[{to_js['f1_5_ci'][0]:.3f}, {to_js['f1_5_ci'][1]:.3f}]"
            row['ws_json_acc5'] = round(ws_js['acc5'], 4); row['ws_json_macro_hit5'] = round(ws_js['macro5'], 4)
            summary.append(row)
            print(f"{c.upper()}_{tag}: n={len(common)} acc5 {pt['ws_generous_correct_k5']:.3f}->{pt['to_generous_correct_k5']:.3f} "
                  f"(d {row['d_acc5']:+.4f} [{row['d_acc5_lo']:+.4f}, {row['d_acc5_hi']:+.4f}]) fixed/broken {fixed}/{broken} "
                  f"hit10 {pt['ws_hit_k10']:.3f}->{pt['to_hit_k10']:.3f}", flush=True)

    keys = list(summary[0].keys()) if summary else []
    with open(os.path.join(DST, 'paired_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
    with open(os.path.join(DST, 'fixed_broken_by_class.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fb_rows[0].keys())); w.writeheader(); w.writerows(fb_rows)

    # Markdown copy
    L = ['# ST2.0 — paired whole-slide vs tumour-only, external cohorts', '',
         f'Generated by `scripts/st2_0_paired_summary.py` (B={B}, seed {SEED}, patients resampled with one slide per draw). '
         'Whole-slide = frozen primary (UNI2 + exact-OT cosine). `_TOQ` = query tumour-only vs full library; '
         '`_TO` = query + library tumour-only, NORMAL-truth queries dropped. Differences are tumour-only minus whole-slide on the paired queries.', '']
    if skipped:
        L += [f'Skipped (evaluate output absent): {", ".join(skipped)}', '']
    L += ['| row | n (patients) | WS job → TO job | acc@5 WS → TO | Δ acc@5 [95 % CI] | macro acc@5 WS → TO | Δ macro acc@5 [CI] | hit@1 WS → TO | hit@5 WS → TO | hit@10 WS → TO | Δ hit@10 [CI] | fixed / broken |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in summary:
        L.append(f"| {r['cohort']}_{r['arm']} | {r['n_paired']} ({r['n_patients']}) | {r['ws_job']} → {r['to_job']} | "
                 f"{r['ws_acc5']:.3f} → {r['to_acc5']:.3f} | {r['d_acc5']:+.4f} [{r['d_acc5_lo']:+.4f}, {r['d_acc5_hi']:+.4f}] | "
                 f"{r['ws_macro_acc5']:.3f} → {r['to_macro_acc5']:.3f} | {r['d_macro_acc5']:+.4f} [{r['d_macro_acc5_lo']:+.4f}, {r['d_macro_acc5_hi']:+.4f}] | "
                 f"{r['ws_hit1']:.3f} → {r['to_hit1']:.3f} | {r['ws_hit5']:.3f} → {r['to_hit5']:.3f} | {r['ws_hit10']:.3f} → {r['to_hit10']:.3f} | "
                 f"{r['d_hit10']:+.4f} [{r['d_hit10_lo']:+.4f}, {r['d_hit10_hi']:+.4f}] | {r['fixed']} / {r['broken']} |")
    L += ['', 'Evaluate-JSON values of the tumour-only arm (own bootstrap, full evaluated population):', '',
          '| row | n eval | commit | acc@5 | macro hit@5 [CI] | generous macro-F1@5 [CI] |', '|---|---|---|---|---|---|']
    for r in summary:
        if r.get('to_job'):
            L.append(f"| {r['cohort']}_{r['arm']} | {r['to_n_eval']} | {r['to_commit']} | {r['to_json_acc5']:.3f} | {r['to_json_macro_hit5']:.3f} {r['to_json_macro_hit5_ci']} | {r['to_json_f1_5']:.3f} {r['to_json_f1_5_ci']} |")
    L += ['', '## Fixed / broken at generous k=5 by truth class (classes with any change)', '',
          '| row | truth | n | wrong WS → TO | fixed | broken |', '|---|---|---|---|---|---|']
    for r in fb_rows:
        if r['fixed'] or r['broken']:
            L.append(f"| {r['cohort']}_{r['arm']} | {r['truth']} | {r['n']} | {r['ws_wrong']} → {r['to_wrong']} | {r['fixed']} | {r['broken']} |")
    open(os.path.join(DOC, 'paired_summary.md'), 'w').write('\n'.join(L) + '\n')
    print('wrote', os.path.join(DOC, 'paired_summary.md'))


if __name__ == '__main__':
    main()
