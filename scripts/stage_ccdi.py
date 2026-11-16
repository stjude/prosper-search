#!/usr/bin/env python3
"""C3: query manifests for the CCDI-MCI cohort (re-analysis + out-of-scope arm).

The published external evaluation (Fig. 4C / Table S15) used SAMPLER slide
embeddings of 4,154 CCDI slides against the 2,921-slide COMET library
(``comet-testsets/src/vertical_pipeline/search/slide_search.py``) and scored
only the slides whose CCDI diagnosis maps to an in-library COMET group; every
diagnosis flagged ``Normal`` in Table S15 is *out of scope* (G5), not normal
tissue. UNI2 mosaic-tile features exist for 4,160 CCDI slides in exactly the
per-slide format the external pipeline consumes
(``comet-testsets/output/COG_CCDI_MCI_ALL/patch_embeddings/uni2/tmp/<id>.pkl``),
so C3 runs the C12/C13 pipeline from the aggregation stage onward with those
features in place (read-only) and the manifests written here:

* ``manifest_all.csv``     every slide with features: ``slide_id`` (= CCDI
  participant id; one slide per participant), ``tumor_group`` = COMET group for
  in-library diagnoses, ``MISC``, ``OOS`` (Table S15 ``Normal`` flag),
  ``UNMAPPED`` (diagnosis text absent from the mapping) or ``NO_METADATA``;
  plus ``ccdi_diagnosis``, ``diagnosis_category``, ``comet_diagnosis``.
* ``manifest_inscope.csv`` the in-library rows only (accuracy, per-entity
  recall, chance baselines).
* ``manifest_oos.csv``     the OOS rows only (abstention arm, per CNS/heme
  entity tables -- D6).

Reads three CSVs; touches no feature file.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from prosper.evaluation import paths  # noqa: E402

TESTSETS = Path("/PROSPER_WORKSPACE/comet-testsets")
FEATURES = TESTSETS / "output/COG_CCDI_MCI_ALL/patch_embeddings/uni2/tmp"
META = TESTSETS / "data/input/CCDI_metadata_all.csv"
MAPPING = TESTSETS / "data/input/ccdi_comet_mapping_v03.csv"
LIBRARY_GROUPS = {"RMS", "NB", "WT", "OS", "RB", "EWS", "NORMAL", "ACT/ACC", "LIVER", "SS",
                  "MEL", "MPNST", "NF", "MRT", "DSRCT"}          # 14 tumour groups + NORMAL


def main() -> int:
    out = paths.output_root() / "external_ccdi"
    out.mkdir(parents=True, exist_ok=True)
    stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(FEATURES / "*.pkl")))
    with open(MAPPING, encoding="utf-8-sig") as fh:
        mp = {r["ccdi_diagnosis"]: (r["comet_diagnosis"], r["comet_tumor_group"]) for r in csv.DictReader(fh)}
    meta = {}
    with open(META, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            pid = r["Participant ID"]
            if pid in meta and meta[pid]["Diagnosis"] != r["Diagnosis"]:
                raise SystemExit(f"{pid}: two diagnoses in CCDI_metadata_all.csv")
            meta[pid] = r
    rows = []
    for sid in stems:
        r = meta.get(sid)
        if r is None:
            rows.append({"slide_id": sid, "tumor_group": "NO_METADATA", "ccdi_diagnosis": "",
                         "diagnosis_category": "", "comet_diagnosis": ""})
            continue
        dx = r["Diagnosis"]
        cdx, grp = mp.get(dx, ("", "UNMAPPED"))
        if grp.upper() == "NORMAL":              # Table S15 flag (v03 spells it NORMAL): out of scope (G5), not normal tissue
            grp = "OOS"
        elif grp in ("Unmatched", "", "UNMAPPED"):   # the v03 file itself uses 'UNMAPPED' for compound strings
            grp = "UNMAPPED"
        elif grp not in LIBRARY_GROUPS and grp != "MISC":
            raise SystemExit(f"{sid}: unexpected comet_tumor_group {grp!r} for {dx!r}")
        rows.append({"slide_id": sid, "tumor_group": grp, "ccdi_diagnosis": dx,
                     "diagnosis_category": r.get("Diagnosis Category", ""), "comet_diagnosis": cdx})
    cols = ["slide_id", "tumor_group", "ccdi_diagnosis", "diagnosis_category", "comet_diagnosis"]
    sel = {"manifest_all.csv": rows,
           "manifest_inscope.csv": [x for x in rows if x["tumor_group"] in LIBRARY_GROUPS],
           "manifest_oos.csv": [x for x in rows if x["tumor_group"] == "OOS"]}
    for name, rs in sel.items():
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rs)
    counts = {}
    for x in rows:
        counts[x["tumor_group"]] = counts.get(x["tumor_group"], 0) + 1
    oos_cat = {}
    for x in sel["manifest_oos.csv"]:
        oos_cat[x["diagnosis_category"]] = oos_cat.get(x["diagnosis_category"], 0) + 1
    rec = {"features_dir": str(FEATURES), "n_features": len(stems), "n_with_metadata": sum(1 for x in rows if x["tumor_group"] != "NO_METADATA"),
           "per_tumor_group": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
           "n_inscope": len(sel["manifest_inscope.csv"]), "n_oos": len(sel["manifest_oos.csv"]),
           "oos_by_diagnosis_category": dict(sorted(oos_cat.items(), key=lambda kv: -kv[1])),
           "mapping": str(MAPPING), "metadata": str(META)}
    with open(out / "_staging_record.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
