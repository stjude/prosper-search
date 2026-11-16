#!/usr/bin/env python3
"""C13: stage the public KidRare cohort as a flat directory + manifest.

KidRare (Hugging Face ``Firehdx233/KidRare``; 2,331 WSIs) is organised as one
folder per entity -- ``Hepatoblastoma/``, ``Medulloblastoma/``,
``Nephroblastoma/``, ``Neuroblastoma/`` -- of Aperio-converted ``*.ibl.tiff``
files. Every stage of ``submit_external_cohort.py`` enumerates a flat
``<slides_dir>/*<ext>``, so this script writes

* ``$PROSPER_OUT/slides_kidrare/<id>.tiff`` -- a symlink per slide, id = file
  stem without ``.ibl`` (checked unique across folders);
* ``$PROSPER_OUT/external_kidrare/manifest.csv`` -- ``slide_id``,
  ``kidrare_class``, ``tumor_group`` (COMET group: WT / NB / LIVER / MB-out-of-
  library), ``source_path``, ``file_bytes``;
* ``manifest_inscope.csv`` -- the WT / NB / LIVER rows only (accuracy run);
* ``manifest_all.csv`` -- every row (confusion + out-of-library arm).

Idempotent; re-running refreshes links and manifests. Reads nothing but
directory listings and ``stat`` -- the 6 TB of pixels are untouched.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from prosper.evaluation import paths  # noqa: E402

SRC = Path("/PROSPER_GROUP_DATA/comet/common/ali/KidRare")
CLASS_TO_GROUP = {          # COMET tumour-group codes (final_scan_id_disease_map_rich_data.csv)
    "Nephroblastoma": "WT",
    "Neuroblastoma": "NB",
    "Hepatoblastoma": "LIVER",     # COMET LIVER = HB 98 + HCC 45 + embryonal sarcoma 1
    "Medulloblastoma": "MB",       # CNS -- not in the library; out-of-library arm
}
IN_SCOPE = {"WT", "NB", "LIVER", "NORMAL"}      # groups the COMET library contains


def slide_id(name: str) -> str:
    stem = name[:-len(".tiff")] if name.endswith(".tiff") else name
    return stem[:-len(".ibl")] if stem.endswith(".ibl") else stem


def main() -> int:
    out = paths.output_root()
    links = out / "slides_kidrare"
    meta = out / "external_kidrare"
    links.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    # KEEP's per-slide labels (WSI_evaluation/kidrare_label/<Class>_labels.json,
    # archived with provenance under external_kidrare/keep_labels/). They cover a
    # subset of each folder and, for Nephroblastoma, distinguish Tumor from
    # Normal (51 of the 110 labelled kidney slides are Normal) -- so the folder
    # name is NOT a slide-level diagnosis there. Rules, stated once:
    #   Hepatoblastoma / Neuroblastoma / Medulloblastoma: folder = entity
    #     (KEEP's task on them is subtyping, every labelled slide is tumour);
    #   Nephroblastoma: KEEP 'Tumor' -> WT, KEEP 'Normal' -> NORMAL (COMET has a
    #     NORMAL class), unlabelled -> WT_UNVERIFIED (label-free block only).
    keep = {}
    for cls in CLASS_TO_GROUP:
        p = meta / "keep_labels" / f"{cls}_labels.json"
        if p.exists():
            for k, v in json.load(open(p)).items():
                keep[slide_id(k if k.endswith(".tiff") else k + ".tiff")] = (cls, v)
    rows, seen = [], {}
    for cls, grp in CLASS_TO_GROUP.items():
        for f in sorted((SRC / cls).iterdir()):
            if not f.name.lower().endswith(".tiff"):
                continue
            sid = slide_id(f.name)
            if sid in seen:
                raise SystemExit(f"duplicate slide id {sid}: {seen[sid]} and {f}")
            seen[sid] = f
            link = links / f"{sid}.tiff"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(f)
            kcls, sub = keep.get(sid, (None, None))
            if kcls is not None and kcls != cls:
                raise SystemExit(f"{sid}: KEEP file {kcls} but folder {cls}")
            group, src = grp, "folder"
            if cls == "Nephroblastoma":
                if sub == "Tumor":
                    group, src = "WT", "keep"
                elif sub == "Normal":
                    group, src = "NORMAL", "keep"
                else:
                    group, src = "WT_UNVERIFIED", "none"
            rows.append({"slide_id": sid, "kidrare_class": cls, "tumor_group": group,
                         "label_source": src, "keep_subtype": sub or "",
                         "source_path": str(f), "file_bytes": f.stat().st_size})
    cols = ["slide_id", "kidrare_class", "tumor_group", "label_source", "keep_subtype",
            "source_path", "file_bytes"]
    for name, sel in (("manifest.csv", rows), ("manifest_all.csv", rows),
                      ("manifest_inscope.csv", [r for r in rows if r["tumor_group"] in IN_SCOPE])):
        with open(meta / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(sel)
    by_cls = {c: sum(1 for r in rows if r["kidrare_class"] == c) for c in CLASS_TO_GROUP}
    gb = {c: round(sum(r["file_bytes"] for r in rows if r["kidrare_class"] == c) / 1e9, 1) for c in CLASS_TO_GROUP}
    by_grp = {}
    for r in rows:
        by_grp[r["tumor_group"]] = by_grp.get(r["tumor_group"], 0) + 1
    rec = {"source": str(SRC), "links_dir": str(links), "n_slides": len(rows), "per_class": by_cls,
           "gb_per_class": gb, "total_gb": round(sum(gb.values()), 1),
           "class_to_group": CLASS_TO_GROUP, "per_tumor_group": by_grp,
           "n_keep_labelled": sum(1 for r in rows if r["keep_subtype"]),
           "n_inscope": sum(1 for r in rows if r["tumor_group"] in IN_SCOPE)}
    with open(meta / "_staging_record.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
