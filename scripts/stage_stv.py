"""Stage the ``solid_tumor_validation`` (stv) label manifest for the C12 labelled evaluation.

Input: the cohort owner's label manifest (columns ``file_name, de-id_uid, dx,
subclass``), ``--source-csv``. Default = the authoritative
``prosper_revision/data/COMET_validation_set_deid_20260902.csv`` (782 rows, 395 cases;
in the repo since 2026-09-22). The first delivery (777 rows, received 2026-09-02, now
``data/COMET_validation_set_deid_v1_777rows.csv``) is what the 2026-09-02 C12 numbers
used; pass it explicitly to reproduce them. The file is copied verbatim next to the
outputs and its sha256 recorded.

Writes to ``$PROSPER_OUT/external_stv/``:

* ``manifest_all.csv``      every one of the 1,247 TIFFs (from the QC scan):
  ``slide_id`` (file stem = matrix row id), ``file_name``, ``tumor_group`` (COMET group
  for in-library diagnoses; ``MISC`` for entities COMET files under MISC; ``OOS`` for
  entities absent from COMET; ``NO_METADATA`` for the 470 unlabelled files),
  ``stv_dx``, ``stv_subclass``, ``patient_id`` (= ``de-id_uid``; 1:1 with the 16-hex
  filename hash), ``qc`` (``ok`` / ``truncated`` / ``no_tissue``), ``in_matrix``.
* ``manifest_labelled.csv`` labelled AND qc == ok AND in_matrix -- the evaluation
  input (``external_cli`` buckets MISC/OOS rows as out-of-library itself).
* ``manifest_inscope.csv``  the in-library subset of the above.
* ``_staging_record.json``  counts + the mapping, for the spec record.

The dx -> COMET mapping follows the library's own composition (LabKey ``Disease`` x
``Tumor.group`` over the 2,921 library slides) and the CCDI v03 precedent
(``ccdi_comet_mapping_v03.csv``): HB/HCC -> LIVER, ACT -> ACT/ACC, ganglioneuroblastoma
-> NB; CIC-rearranged sarcoma, CCSK, desmoid, schwannoma, paraganglioma are MISC in
COMET (dropped from the library, so unretrievable). ``RET`` is read as retinoblastoma
(RB) -- the only abbreviation not in COMET's vocabulary; flagged for the owner.
QC: the 11 single-level exports with zero-byte level-0 tiles are excluded as QC
failures (user decision 2026-08-29, [internal record] section 0a); the two no-tissue mosaic
exclusions are whatever is absent from the matrix index.

Reads three CSVs and one matrix index; touches no feature file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from prosper.evaluation import paths  # noqa: E402

DEFAULT_SOURCE_CSV = Path("/PROSPER_WORKSPACE/"
                          "prosper_revision/data/COMET_validation_set_deid_20260902.csv")
LIBRARY_GROUPS = {"RMS", "NB", "WT", "OS", "RB", "EWS", "NORMAL", "ACT/ACC", "LIVER", "SS",
                  "MEL", "MPNST", "NF", "MRT", "DSRCT"}          # 14 tumour groups + NORMAL

# stv dx -> (comet_diagnosis, comet_tumor_group)
STV_TO_COMET = {
    "OS": ("OS", "OS"), "NB": ("NB", "NB"), "WT": ("WT", "WT"), "EWS": ("EWS", "EWS"),
    "RMS": ("RMS", "RMS"), "SS": ("SS", "SS"), "MEL": ("MEL", "MEL"), "MRT": ("MRT", "MRT"),
    "MPNST": ("MPNST", "MPNST"), "NF": ("NF", "NF"), "DSRCT": ("DSRCT", "DSRCT"),
    "hepatoblastoma": ("HB", "LIVER"), "HCC": ("HCC", "LIVER"),
    "ACT": ("ACT", "ACT/ACC"),
    "RET": ("RB", "RB"),                       # retinoblastoma -- assumption, see docstring
    "Normal_LN": ("Normal", "NORMAL"), "Normal_thyroid": ("Normal", "NORMAL"),
    "Normal Liver": ("Normal", "NORMAL"),
    "desmoid": ("Desmoid tumor", "MISC"), "CIC sarcoma": ("CIC-rearranged sarcoma", "MISC"),
    "CCSK": ("CCSK", "MISC"), "schwannoma": ("Schwannoma", "MISC"),
    "PHEO": ("Paraganglioma", "MISC"),         # pheochromocytoma; COMET files paraganglioma under MISC
    "DFSP": ("DFSP", "OOS"),                   # dermatofibrosarcoma protuberans: not in COMET at all
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV,
                    help="owner's label manifest (default: the authoritative 782-row file)")
    SOURCE_CSV = ap.parse_args().source_csv
    if not SOURCE_CSV.exists():
        raise SystemExit(f"source manifest not found: {SOURCE_CSV}")
    out = paths.output_root() / "external_stv"
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / SOURCE_CSV.name
    if not frozen.exists():
        shutil.copy2(SOURCE_CSV, frozen)
    elif hashlib.sha256(frozen.read_bytes()).hexdigest() != hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest():
        raise SystemExit(f"{frozen} exists with different content than {SOURCE_CSV}; rename one")
    sha = hashlib.sha256(frozen.read_bytes()).hexdigest()

    with open(frozen, encoding="utf-8-sig") as fh:
        src = list(csv.DictReader(fh))
    lab = {}
    for r in src:
        sid = r["file_name"][:-5] if r["file_name"].endswith(".tiff") else r["file_name"]
        if sid in lab:
            raise SystemExit(f"{sid}: duplicated in {SOURCE_CSV.name}")
        if r["dx"] not in STV_TO_COMET:
            raise SystemExit(f"{sid}: unmapped dx {r['dx']!r}")
        lab[sid] = r

    qc_path = paths.output_root() / "qc_stv" / "zero_tile_scan.csv"
    with open(qc_path) as fh:
        qc = {r["slide_id"]: r for r in csv.DictReader(fh)}
    matrix = paths.output_root() / "similarity_matrix" / "combined_similarity_matrix_stv_uni2_ot_cosine.pkl"
    with open(matrix, "rb") as fh:
        in_matrix = set(str(i) for i in pickle.load(fh).index)

    rows = []
    for sid in sorted(qc):
        r = lab.get(sid)
        truncated = float(qc[sid]["zero_fraction"]) > 0
        q = "truncated" if truncated else ("ok" if sid in in_matrix else "no_tissue")
        if r is None:
            rows.append({"slide_id": sid, "file_name": sid + ".tiff", "tumor_group": "NO_METADATA",
                         "comet_diagnosis": "", "stv_dx": "", "stv_subclass": "", "patient_id": "",
                         "qc": q, "in_matrix": int(sid in in_matrix)})
            continue
        cdx, grp = STV_TO_COMET[r["dx"]]
        if r["file_name"].split("_")[0] != r["file_name"].split("_")[0]:
            raise AssertionError
        rows.append({"slide_id": sid, "file_name": r["file_name"], "tumor_group": grp,
                     "comet_diagnosis": cdx, "stv_dx": r["dx"], "stv_subclass": r.get("subclass", ""),
                     "patient_id": r["de-id_uid"], "qc": q, "in_matrix": int(sid in in_matrix)})
    missing = set(lab) - set(qc)
    if missing:
        raise SystemExit(f"{len(missing)} labelled files not on disk: {sorted(missing)[:5]}")

    cols = ["slide_id", "file_name", "tumor_group", "comet_diagnosis", "stv_dx", "stv_subclass",
            "patient_id", "qc", "in_matrix"]
    labelled = [x for x in rows if x["tumor_group"] != "NO_METADATA" and x["qc"] == "ok" and x["in_matrix"]]
    sel = {"manifest_all.csv": rows,
           "manifest_labelled.csv": labelled,
           "manifest_inscope.csv": [x for x in labelled if x["tumor_group"] in LIBRARY_GROUPS]}
    for name, rs in sel.items():
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rs)

    def count(rs, key):
        c = {}
        for x in rs:
            c[x[key]] = c.get(x[key], 0) + 1
        return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))

    lab_rows = [x for x in rows if x["tumor_group"] != "NO_METADATA"]
    rec = {"source_csv": str(SOURCE_CSV), "frozen_copy": str(frozen), "sha256": sha,
           "n_files_on_disk": len(rows), "n_labelled": len(lab_rows),
           "n_patients_labelled": len(set(x["patient_id"] for x in lab_rows)),
           "labelled_qc": count(lab_rows, "qc"),
           "labelled_qc_excluded": [{k: x[k] for k in ("slide_id", "stv_dx", "patient_id", "qc")}
                                    for x in lab_rows if x["qc"] != "ok"],
           "n_evaluated": len(labelled),
           "n_inscope": len(sel["manifest_inscope.csv"]),
           "n_out_of_library": len(labelled) - len(sel["manifest_inscope.csv"]),
           "evaluated_per_tumor_group": count(labelled, "tumor_group"),
           "evaluated_per_stv_dx": count(labelled, "stv_dx"),
           "subclass_counts": count([x for x in labelled if x["stv_subclass"]], "stv_subclass"),
           "mapping": {k: {"comet_diagnosis": v[0], "tumor_group": v[1]} for k, v in STV_TO_COMET.items()},
           "qc_scan": str(qc_path), "matrix_index": str(matrix)}
    with open(out / "_staging_record.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
