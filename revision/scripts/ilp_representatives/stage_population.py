"""Stage the in-library, NORMAL-excluded STV / CCDI populations for the
within-cohort UNI2+OT similarity job ([analysis plan]).

Writes, under $PROSPER_OUT:
  ilp_representatives/<cohort>/population.csv        the slides in scope
  features/<cohort>_inlib_uni2/<slide_id>.pkl         symlinks to the UNI2 patch features
  ilp_representatives/<cohort>/params_similarity.txt  one similarity_cli line per LSF element

Both --features and --library-features point at the same staged directory, so
similarity_cli's EXTERNAL mode yields a square cohort x cohort block with the
1e6 self sentinel on the diagonal (same convention as the COMET matrices).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

OUT = Path(os.environ["PROSPER_OUT"])
CCDI_FEATURES = Path(
    "/PROSPER_WORKSPACE/"
    "comet-testsets/output/COG_CCDI_MCI_ALL/patch_embeddings/uni2/tmp")

COHORTS = {
    "stv": dict(manifest=OUT / "external_stv" / "manifest_inscope.csv",
                features=OUT / "features" / "stv_uni2"),
    "ccdi": dict(manifest=OUT / "external_ccdi" / "manifest_inscope.csv",
                 features=CCDI_FEATURES),
}


def stage(cohort: str, per_element: int) -> None:
    cfg = COHORTS[cohort]
    m = pd.read_csv(cfg["manifest"])
    n0 = len(m)
    if "in_matrix" in m.columns:
        m = m[m["in_matrix"] == 1]
    m = m[m["tumor_group"] != "NORMAL"].copy()
    feats = {p.stem: p for p in cfg["features"].glob("*.pkl")}
    missing = [s for s in m["slide_id"] if s not in feats]
    if missing:
        raise SystemExit(f"{cohort}: {len(missing)} slides without features, e.g. {missing[:3]}")
    m = m.sort_values("slide_id").reset_index(drop=True)

    task_dir = OUT / "ilp_representatives" / cohort
    task_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(task_dir / "population.csv", index=False)

    stage_dir = OUT / "features" / f"{cohort}_inlib_uni2"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for sid in m["slide_id"]:
        link = stage_dir / f"{sid}.pkl"
        if not link.exists():
            link.symlink_to(feats[sid].resolve())
    n_links = len(list(stage_dir.glob("*.pkl")))
    if n_links != len(m):
        raise SystemExit(f"{cohort}: {n_links} links but {len(m)} slides in population")

    task = f"similarity/{cohort}_inlib_uni2_ot"
    lines = []
    for lo in range(1, len(m) + 1, per_element):
        hi = min(lo + per_element - 1, len(m))
        lines.append(
            f"--encoder uni2 --method ot --features {stage_dir} "
            f"--library-features {stage_dir} --feature-dim 1536 "
            f"--lo {lo} --hi {hi} --task {task} --num-workers 8")
    (task_dir / "params_similarity.txt").write_text("\n".join(lines) + "\n")
    print(f"{cohort}: manifest {n0} -> population {len(m)} "
          f"(NORMAL dropped: {n0 - len(m)}); {n_links} feature links; "
          f"{len(lines)} LSF elements x {per_element} queries -> {task}")
    print(m["tumor_group"].value_counts().to_dict())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=list(COHORTS), action="append")
    p.add_argument("--per-element", type=int, default=10)
    a = p.parse_args()
    for c in a.cohort or list(COHORTS):
        stage(c, a.per_element)
