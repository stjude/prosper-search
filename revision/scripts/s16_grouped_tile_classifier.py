"""Table S16 re-evaluation (R3.5 / R2.4): slide-held-out tile classifier.

Finding (2026-09-22): Table S16's per-class tile counts (Tumor 414,331 / Immune 3,483 / Necrosis 28,994 /
Non-tumor 70,718 / Artifacts 2,039 = 519,565) equal `y_test_rand` of
`comet-annotation/output/pathologist_annotated_train_test_split_uni2` exactly — the published 0.998 AUROC is on a
tile-random 5/95 split (`train_test_split(shuffle=True)`) whose 27,345 training tiles come from 231 of the 237
annotated slides. The same directory holds an unused slide-grouped split (`GroupShuffleSplit`, 11 train slides /
226 test slides, 15,580 / 531,330 tiles).

This script (a) reproduces the published figure from the deployed model on the tile-random test set, (b) trains
the identical SGD recipe (log_loss, l2, alpha 1e-3, balanced class weights, seed 1234) on the grouped split and
reports one-vs-rest AUROC per class, macro AUROC, tile accuracy, tumour-vs-rest sensitivity/specificity and a
5-class confusion matrix on the 226 held-out slides, with slide-clustered bootstrap CIs (first run 09-22: the 11
training slides carry no Immune/Necrosis tiles and the model calls 92 % of non-tumour tiles tumour — the saved
grouped split is too small to train on, so it is kept only as the like-for-like 5 % row), (c) runs 5-fold
GroupKFold over all 237 annotated slides (same recipe; ~190 train / ~47 test slides per fold), pools the
out-of-fold probabilities and reports the same metrics — the slide-held-out figure for the revised Table S16 —
and (d) writes per-slide out-of-fold tumour-vs-rest accuracy so a genuinely held-out slide can be chosen for Fig. 3A.

Label codes (from the S16 order / counts): 1 Tumor, 2 Immune cells, 3 Necrosis, 4 Non-tumor, 5 Artifacts.
Outputs: $PROSPER_OUT/s16_grouped/{s16_reeval.json, per_slide_oof.csv, per_slide_heldout_5pct.csv, model_grouped_uni2.pkl, oof_probs.npy}.
"""
import os, json, csv, time
import numpy as np
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

A = '/PROSPER_WORKSPACE/comet-annotation'
SPLIT = f'{A}/output/pathologist_annotated_train_test_split_uni2'
DEPLOYED = f'{A}/model/pathologist_annotated_slides_uni2/tumor_detector_sgd_test_fraction_0.95_training.pkl'
OUT = os.path.join(os.environ.get('PROSPER_OUT', 'prosper_revision_output'), 's16_grouped')
os.makedirs(OUT, exist_ok=True)
NAMES = {1: 'Tumor', 2: 'Immune cells', 3: 'Necrosis', 4: 'Non-tumor', 5: 'Artifacts'}
B, SEED = 2000, 123


def slide_ids(idx_file):
    with open(f'{SPLIT}/slide_info.csv') as f:
        sid = np.array([r['slide_id'] for r in csv.DictReader(f)])
    return sid[np.load(f'{SPLIT}/{idx_file}')]


def evaluate(y, probs, classes, sid, tag):
    pred = classes[probs.argmax(1)]
    res = dict(tag=tag, n_tiles=int(len(y)), n_slides=int(len(set(sid.tolist()))),
               class_counts={NAMES[c]: int((y == c).sum()) for c in classes})
    res['auroc_ovr'] = {NAMES[c]: float(roc_auc_score((y == c).astype(int), probs[:, i])) for i, c in enumerate(classes)}
    res['auroc_macro'] = float(np.mean(list(res['auroc_ovr'].values())))
    res['tile_accuracy'] = float((pred == y).mean())
    t, tp = y == 1, pred == 1
    res['tumour_vs_rest'] = dict(sensitivity=float((tp & t).sum() / t.sum()), specificity=float((~tp & ~t).sum() / (~t).sum()),
                                 auroc=res['auroc_ovr']['Tumor'])
    res['confusion_rows_true_cols_pred'] = confusion_matrix(y, pred, labels=classes).tolist()
    # slide-clustered bootstrap of the headline numbers
    rng = np.random.default_rng(SEED)
    groups = {}
    for i, s in enumerate(sid):
        groups.setdefault(s, []).append(i)
    glist = [np.array(v) for v in groups.values()]
    boots = {'auroc_tumor': [], 'tile_accuracy': [], 'sensitivity': [], 'specificity': []}
    for _ in range(B):
        idx = np.concatenate([glist[g] for g in rng.integers(0, len(glist), len(glist))])
        yy, pp, pr = y[idx], probs[idx], pred[idx]
        tt, tpp = yy == 1, pr == 1
        if tt.all() or (~tt).all():
            continue
        boots['auroc_tumor'].append(roc_auc_score(tt.astype(int), pp[:, 0]))
        boots['tile_accuracy'].append((pr == yy).mean())
        boots['sensitivity'].append((tpp & tt).sum() / tt.sum())
        boots['specificity'].append((~tpp & ~tt).sum() / (~tt).sum())
    res['ci95_slide_clustered'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)] for k, v in boots.items()}
    return res, pred

def per_slide(sid, y, pred, path, fold=None):
    rows = []
    for s in sorted(set(sid.tolist())):
        m = sid == s
        yy, p = y[m], pred[m]
        t, tp = yy == 1, p == 1
        r = dict(slide=s, n_annot=int(m.sum()), tumour=int(t.sum()), nontum=int((~t).sum()), necrosis=int((yy == 3).sum()),
                 acc=round(float((tp == t).mean()), 4), sens=round(float((tp & t).sum() / t.sum()), 4) if t.any() else '',
                 spec=round(float((~tp & ~t).sum() / (~t).sum()), 4) if (~t).any() else '')
        if fold is not None:
            r['fold'] = int(fold[m][0])
        rows.append(r)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    t0 = time.time()
    out = dict(split_dir=SPLIT, deployed_model=DEPLOYED, label_codes=NAMES, bootstrap=dict(B=B, seed=SEED, cluster='slide'))
    # (a) deployed model on the tile-random test set (reproduces Table S16)
    clf0 = joblib.load(DEPLOYED)
    Xr, yr = np.load(f'{SPLIT}/X_test_rand.npy'), np.load(f'{SPLIT}/y_test_rand.npy')
    sr = slide_ids('idx_test_rand.npy')
    tr_slides = set(slide_ids('idx_train_rand.npy').tolist())
    out['tile_random'], _ = evaluate(yr, clf0.predict_proba(Xr), clf0.classes_, sr, 'deployed model, tile-random 95 % test split')
    out['tile_random']['n_train_tiles'] = int(len(np.load(f'{SPLIT}/y_train_rand.npy')))
    out['tile_random']['n_train_slides'] = len(tr_slides)
    out['tile_random']['test_slides_also_in_train'] = int(sum(1 for s in set(sr.tolist()) if s in tr_slides))
    print('tile-random reproduced:', out['tile_random']['auroc_macro'], out['tile_random']['auroc_ovr'], f'{time.time()-t0:.0f}s', flush=True)
    del Xr

    # (b) same recipe on the slide-grouped split
    Xtr, ytr = np.load(f'{SPLIT}/X_train_grouped.npy'), np.load(f'{SPLIT}/y_train_grouped.npy')
    Xte, yte = np.load(f'{SPLIT}/X_test_grouped.npy'), np.load(f'{SPLIT}/y_test_grouped.npy')
    ste = slide_ids('idx_test_grouped.npy')
    train_slides = sorted(set(np.load(f'{SPLIT}/train_slide_ids.npy', allow_pickle=True).tolist()))
    assert not (set(train_slides) & set(ste.tolist())), 'grouped split leaks slides'
    classes = np.unique(np.concatenate((ytr, yte)))
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=np.concatenate((ytr, yte)))
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.001, max_iter=1000, tol=1e-3,
                        class_weight={c: w for c, w in zip(classes, cw)}, random_state=1234, n_jobs=4)
    clf.fit(Xtr, ytr)
    joblib.dump(clf, f'{OUT}/model_grouped_uni2.pkl')
    probs = clf.predict_proba(Xte)
    out['slide_grouped'], pred = evaluate(yte, probs, clf.classes_, ste, 'same recipe, slide-grouped split (GroupShuffleSplit)')
    out['slide_grouped'].update(n_train_tiles=int(len(ytr)), train_slides=train_slides,
                                train_class_counts={NAMES[c]: int((ytr == c).sum()) for c in classes})
    print('slide-grouped:', out['slide_grouped']['auroc_macro'], out['slide_grouped']['auroc_ovr'],
          out['slide_grouped']['tumour_vs_rest'], f'{time.time()-t0:.0f}s', flush=True)

    per_slide(ste, yte, pred, f'{OUT}/per_slide_heldout_5pct.csv')
    del Xtr, Xte

    # (c) 5-fold GroupKFold over all annotated slides, same recipe; out-of-fold predictions
    X = np.concatenate([np.load(f'{SPLIT}/X_train_rand.npy'), np.load(f'{SPLIT}/X_test_rand.npy')])
    y = np.concatenate([np.load(f'{SPLIT}/y_train_rand.npy'), yr])
    sid = np.concatenate([slide_ids('idx_train_rand.npy'), sr])
    classes = np.unique(y)
    oof = np.zeros((len(y), len(classes)), dtype=np.float32)
    fold_of = np.zeros(len(y), dtype=int)
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=sid)):
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=y[tr])
        m = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.001, max_iter=1000, tol=1e-3,
                          class_weight={c: w for c, w in zip(classes, cw)}, random_state=1234, n_jobs=4)
        m.fit(X[tr], y[tr])
        pr = m.predict_proba(X[te])
        # align columns in case a fold's training set lacks a class
        for j, c in enumerate(m.classes_):
            oof[te, np.where(classes == c)[0][0]] = pr[:, j]
        fold_of[te] = k
        folds.append(dict(fold=k, n_train_slides=int(len(set(sid[tr].tolist()))), n_test_slides=int(len(set(sid[te].tolist()))),
                          n_train_tiles=int(len(tr)), n_test_tiles=int(len(te)),
                          train_class_counts={NAMES[c]: int((y[tr] == c).sum()) for c in classes}))
        print(f'fold {k} done {time.time()-t0:.0f}s', flush=True)
    np.save(f'{OUT}/oof_probs.npy', oof)
    out['slide_groupkfold_5'], pred_oof = evaluate(y, oof, classes, sid, 'same recipe, 5-fold GroupKFold over 237 slides, pooled out-of-fold')
    out['slide_groupkfold_5']['folds'] = folds
    print('GroupKFold:', out['slide_groupkfold_5']['auroc_macro'], out['slide_groupkfold_5']['auroc_ovr'],
          out['slide_groupkfold_5']['tumour_vs_rest'], f'{time.time()-t0:.0f}s', flush=True)
    per_slide(sid, y, pred_oof, f'{OUT}/per_slide_oof.csv', fold_of)
    out['runtime_s'] = round(time.time() - t0, 1)
    json.dump(out, open(f'{OUT}/s16_reeval.json', 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
