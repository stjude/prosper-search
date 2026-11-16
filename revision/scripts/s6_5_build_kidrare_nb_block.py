#!/usr/bin/env python3
"""S6.5 external: build the within-KidRare NB OT block (labelled 136 queries x
303 NB library) declared in [internal record].

Subcommands:
  stage     symlink the query/library feature dirs under $PROSPER_OUT/s6_5_kidrare_nb/
  assemble  collect similarity_<qid>.pkl rows into one block DataFrame pickle

The rows themselves are computed by the frozen similarity CLI (external mode,
same settings as the existing KidRare x COMET run: uni2 features, feature_dim
1536, cosine ground cost, reg=0 exact EMD):

  PYTHONPATH=<prosper-search> python3 -m prosper.embeddings.similarity_cli \
      --encoder kidrare_nb_uni2 --method ot --feature-dim 1536 \
      --features $PROSPER_OUT/s6_5_kidrare_nb/features_queries \
      --library-features $PROSPER_OUT/s6_5_kidrare_nb/features_library \
      --task s6_5_kidrare_nb/similarity_rows --num-workers 8 --lo L --hi H
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

CORE = "/PROSPER_WORKSPACE"
OUT = os.environ.get("PROSPER_OUT", f"{CORE}/prosper_revision_output")
BASE = f"{OUT}/s6_5_kidrare_nb"
FEATS = f"{OUT}/features/kidrare_uni2"


def stage():
    km = pd.read_csv(f"{OUT}/external_kidrare/manifest_inscope.csv")
    km["slide_id"] = km["slide_id"].astype(str)
    nb = km[km.tumor_group == "NB"]
    have = {os.path.basename(f).split(".")[0]
            for f in glob.glob(f"{FEATS}/*.pkl")}
    lib = sorted(set(nb["slide_id"]) & have)
    labelled = nb[nb["keep_subtype"].notna()]
    q = sorted(set(labelled["slide_id"]) & have)
    missing_q = sorted(set(labelled["slide_id"]) - have)
    for sub, ids in (("features_queries", q), ("features_library", lib)):
        d = f"{BASE}/{sub}"
        os.makedirs(d, exist_ok=True)
        for sid in ids:
            dst = f"{d}/{sid}.pkl"
            if not os.path.islink(dst):
                os.symlink(f"{FEATS}/{sid}.pkl", dst)
    rec = {"n_queries_staged": len(q), "n_library_staged": len(lib),
           "labelled_without_features": missing_q,
           "source_features": FEATS}
    with open(f"{BASE}/staging_record.json", "w") as fh:
        json.dump(rec, fh, indent=2)
    print(json.dumps(rec, indent=2))


def assemble():
    rows_dir = f"{BASE}/similarity_rows"
    q_ids = sorted(os.path.basename(f).split(".")[0]
                   for f in glob.glob(f"{BASE}/features_queries/*.pkl"))
    lib_ids = sorted(os.path.basename(f).split(".")[0]
                     for f in glob.glob(f"{BASE}/features_library/*.pkl"))
    missing = [q for q in q_ids
               if not os.path.exists(f"{rows_dir}/similarity_{q}.pkl")]
    if missing:
        raise SystemExit(f"{len(missing)} query rows missing, e.g. {missing[:5]}")
    block = pd.DataFrame(np.full((len(q_ids), len(lib_ids)), np.nan),
                         index=q_ids, columns=lib_ids)
    for qid in q_ids:
        with open(f"{rows_dir}/similarity_{qid}.pkl", "rb") as fh:
            row = pickle.load(fh)
        d = dict(row)
        block.loc[qid] = [d.get(c, np.nan) for c in lib_ids]
    n_nan = int(block.isna().sum().sum())
    out_path = f"{BASE}/block_kidrare_nb_uni2_ot_cosine.pkl"
    block.to_pickle(out_path)
    finite = block.values[np.isfinite(block.values) & (block.values < 1e6)]
    print(json.dumps({"block": out_path, "shape": list(block.shape),
                      "n_nan": n_nan,
                      "n_self_sentinel": int((block.values >= 1e6).sum()),
                      "dist_min": float(finite.min()),
                      "dist_max": float(finite.max())}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["stage", "assemble"])
    a = p.parse_args()
    stage() if a.cmd == "stage" else assemble()
