"""Stage the FFPE-only CCDI in-library cohort (`ccdi_ffpe`) for the ILP pipeline.

Why: 209/765 in-library CCDI slides are frozen/OCT ([analysis plan] amendment 3).
Pool v2 included 8/40 OCT slides, whereas a co-author's 125 are all FFPE.
The pathologist review population is now restricted to confirmed FFPE.

Inputs (all existing; no similarity recompute needed):
  $PROSPER_OUT/ilp_representatives/ccdi/population.csv                 765 in-library slides
  $PROSPER_OUT/external_ccdi/slide_preparation.csv                     scripts/c3_ccdi_specimen_preparation.py
  $PROSPER_OUT/similarity_matrix/combined_similarity_matrix_ccdi_inlib_uni2_ot_cosine.pkl

Outputs
  $PROSPER_OUT/ilp_representatives/ccdi_ffpe/population.csv           prep_dicom == "FFPE" only
  $PROSPER_OUT/ilp_representatives/ccdi_ffpe/tile_counts.csv          copied from ccdi
  $PROSPER_OUT/similarity_matrix/combined_similarity_matrix_ccdi_ffpe_inlib_uni2_ot_cosine.pkl

Then:
  POOL_VER=v3 POOL_TAG=<date> python3 scripts/ilp_representatives/build_review_pool.py --cohort ccdi
No preliminary representative ILP is needed for v3.
"""
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

OUT = Path(os.environ.get("PROSPER_OUT") or sys.exit("export PROSPER_OUT first"))
SRC = OUT / "ilp_representatives/ccdi"
DST = OUT / "ilp_representatives/ccdi_ffpe"
MAT_IN = OUT / "similarity_matrix/combined_similarity_matrix_ccdi_inlib_uni2_ot_cosine.pkl"
MAT_OUT = OUT / "similarity_matrix/combined_similarity_matrix_ccdi_ffpe_inlib_uni2_ot_cosine.pkl"


def main() -> int:
    pop = pd.read_csv(SRC / "population.csv", dtype=str)
    prep = pd.read_csv(OUT / "external_ccdi/slide_preparation.csv", dtype=str)
    m = pop.merge(prep[["slide_id", "prep_dicom", "prep_manifest"]], on="slide_id", how="left")
    assert m.prep_dicom.notna().all(), "slides missing from slide_preparation.csv"
    print("in-library by DICOM prep:\n", m.prep_dicom.value_counts().to_string())

    keep = m[m.prep_dicom == "FFPE"].copy()
    dropped = m[m.prep_dicom != "FFPE"]
    DST.mkdir(parents=True, exist_ok=True)
    keep.to_csv(DST / "population.csv", index=False)
    dropped.to_csv(DST / "excluded_non_ffpe.csv", index=False)
    shutil.copy(SRC / "tile_counts.csv", DST / "tile_counts.csv")

    sm = pd.read_pickle(MAT_IN)
    sm.index = sm.index.astype(str); sm.columns = sm.columns.astype(str)
    ids = list(keep.slide_id)
    sub = sm.loc[ids, ids]
    assert sub.shape == (len(ids), len(ids))
    sub.to_pickle(MAT_OUT)
    print(f"\nkept {len(keep)} FFPE / dropped {len(dropped)}; matrix {sub.shape} -> {MAT_OUT}")
    print(keep.tumor_group.value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
