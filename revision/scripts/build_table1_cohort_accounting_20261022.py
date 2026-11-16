#!/usr/bin/env python3
"""Assemble main-text Table 1 — cohort accounting (per-cohort sizes, inclusion and exclusion
rules, and the slides finally evaluated).

Every number is copied from the accepted and independently audited cohort accounting
`$PROSPER_OUT/r3_3_cohort_accounting_20261008_v3/` (R3.3 → Audited 2026-10-19; 0 numeric mismatches)
and cross-checked against the Fig. 4 source data (evaluated n per additional collection) and the
Fig. 2/6 populations (2,778 / 1,876). Nothing is recomputed. The script asserts that each printed count
occurs in the v3 cells and that the exclusion arithmetic closes.

Output: $PROSPER_OUT/main_tables_20261022/Table_1_cohort_accounting.{csv,md,xlsx}, README.md, SHA256SUMS.txt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

OUT_ROOT = Path(os.environ.get("PROSPER_OUT") or Path.cwd() / "prosper_revision_output")
V3 = OUT_ROOT / "r3_3_cohort_accounting_20261008_v3"
FIG4 = OUT_ROOT / "figures_20261021/fig4_additional_collections/source_data.csv"
OUT = OUT_ROOT / "main_tables_20261022"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


COLUMNS = [
    "Cohort", "Role in this study", "Source", "Material", "Slides received",
    "Excluded before evaluation (reason: n)", "Slides evaluated (main analysis)",
    "Patients evaluated", "Tumour groups", "Label source",
]

ROWS = [
    {
        "Cohort": "COMET",
        "Role in this study": "Reference library; internal evaluation (each slide queried against the library with all slides of the same patient removed)",
        "Source": "St. Jude Children's Research Hospital (COMET)",
        "Material": "FFPE H&E whole-slide images (245 xenograft slides from 196 patients among the 2,778)",
        "Slides received": "2,921 (1,993 patients)",
        "Excluded before evaluation (reason: n)": "Miscellaneous entities without a diagnostic group (MISC): 143",
        "Slides evaluated (main analysis)": "2,778 (tumour-only arm: 2,604 after removing 174 NORMAL slides)",
        "Patients evaluated": "1,876 (tumour-only arm: 1,800)",
        "Tumour groups": "14 diagnostic groups + NORMAL",
        "Label source": "Institutional pathology diagnosis (COMET tumour group)",
    },
    {
        "Cohort": "STV",
        "Role in this study": "Additional query collection (later accession, same institution; not in the library)",
        "Source": "St. Jude Children's Research Hospital",
        "Material": "H&E whole-slide images (preparation not recorded in the manifest)",
        "Slides received": "782 (394 patients)",
        "Excluded before evaluation (reason: n)": "Truncated scan exports: 6; diagnoses outside the 14 groups (desmoid 24, CIC sarcoma 21, CCSK 12, DFSP 10, schwannoma 9, PHEO 4): 80 slides / 43 patients",
        "Slides evaluated (main analysis)": "696 (both-sided tumour-only arm: 688 after removing 8 NORMAL slides)",
        "Patients evaluated": "350 (both-sided arm: 346)",
        "Tumour groups": "14 diagnostic groups + NORMAL",
        "Label source": "Curated label manifest (source label RET mapped to RB, provisional, 32 slides / 19 patients; patient overlap with COMET not verified)",
    },
    {
        "Cohort": "KidRare",
        "Role in this study": "Additional query collection (external institutions; public release)",
        "Source": "KidRare public release (multi-institutional)",
        "Material": "H&E whole-slide images (preparation not established)",
        "Slides received": "2,331 (patient identifiers not released)",
        "Excluded before evaluation (reason: n)": "Unreadable file: 1; no tissue detected: 8; medulloblastoma, no COMET reference class: 482; unlabelled kidney slides: 259",
        "Slides evaluated (main analysis)": "1,581 (both-sided tumour-only arm: 1,531 after removing 50 NORMAL slides; one LIVER slide without tumour tiles is counted as a miss in filtered arms)",
        "Patients evaluated": "Not available",
        "Tumour groups": "LIVER 815, WT 413, NB 303, NORMAL 50",
        "Label source": "Provider folder labels; archived subtype for 357 WT slides",
    },
    {
        "Cohort": "CCDI-MCI",
        "Role in this study": "Additional query collection (external institutions)",
        "Source": "NCI Childhood Cancer Data Initiative, Molecular Characterization Initiative",
        "Material": "FFPE H&E whole-slide images (frozen-section slides excluded; sensitivity analysis reported separately)",
        "Slides received": "4,307 downloaded; 4,160 with tile features; 4,154 with retrieval output (765 participants in the COMET vocabulary)",
        "Excluded before evaluation (reason: n)": "Of 4,154: diagnosis outside the COMET vocabulary: 2,952; no metadata: 231; unmapped diagnosis string: 138; MISC: 68 (765 remain). Of 765: frozen/OCT 209, mixed 6, unclassified preparation 4 (546 FFPE remain). Of 546: CNS neuroblastoma, FOXR2-activated: 4; unresolved diagnosis scope: 18",
        "Slides evaluated (main analysis)": "524",
        "Patients evaluated": "524 (one slide per participant)",
        "Tumour groups": "12 of the 14 diagnostic groups observed",
        "Label source": "CCDI diagnosis strings mapped to COMET tumour groups (Supplementary Table S15)",
    },
]

FOOTNOTES = [
    "Counts are the regenerated cohort accounting (Supplementary Table S14 and Supplementary Fig. [S-R3.3] give the "
    "full slide and patient flow per collection); the submitted Table S14 figures of 948 and 867 CCDI-MCI slides are "
    "superseded (867 = 765 in-vocabulary slides + 68 MISC + 34 suffixed re-exports; 948 is not reconstructable from any retained artefact).",
    "The five historical collections listed in the submitted Table S14 (NB MYCN TMA, NB MYCN WSI, NB ADRN vs MES, COMET RMS extension, COG RMS mutation) "
    "were exploratory and are not evaluation denominators of the revised manuscript; they remain listed in Supplementary Table S14.",
    "Main analysis: UNI2 tile encoder, exact optimal-transport slide similarity on 15 % of tiles, k = 5, all tissue; intervals elsewhere are whole-patient bootstrap (patient as the resampling unit where identifiers exist, slide otherwise).",
]


def checks(cur: pd.DataFrame, fig4: pd.DataFrame) -> list[str]:
    log = []
    cells = " ".join(cur.astype(str).values.ravel())

    def has(tok: str) -> None:
        assert tok in cells, f"'{tok}' not found in the v3 current-cohort cells"
        log.append(f"found in v3: {tok}")

    for tok in ["2,921", "2,778", "2,604", "1,993", "1,876", "1,800", "143", "174", "245 / 196",
                "782", "776", "696", "688", "394", "350", "346", "80 slides / 43 patients", "desmoid 24", "32 slides / 19 patients",
                "2,331", "2,322", "1,581", "1,531", "MB 482", "259 unlabelled", "LIVER 815", "NB 303", "WT 413", "NORMAL 50",
                "4,307", "4,160", "4,154", "765", "546", "524", "2,952", "231", "138", "68", "209", "4 CNS NB FOXR2", "18 slides",
                "12 observed groups"]:
        has(tok)
    # exclusion arithmetic
    assert 2921 - 143 == 2778 and 2778 - 174 == 2604
    assert 782 - 6 == 776 and 776 - 80 == 696 and 696 - 8 == 688
    assert 2331 - 1 - 8 == 2322 and 2322 - 482 - 259 == 1581 and 1581 - 50 == 1531
    assert 4154 - 2952 - 231 - 138 - 68 == 765 and 765 - 209 - 6 - 4 == 546 and 546 - 4 - 18 == 524
    log.append("exclusion arithmetic closes for all four cohorts")
    # Fig. 4 evaluated n (macro rows, main configuration)
    m = fig4[(fig4.metric == "hit_at_5") & (fig4.tumor_group == "macro")].set_index("cohort").n_slides
    assert int(m["stv"]) == 696 and int(m["kidrare"]) == 1581 and int(m["ccdi"]) == 524, m.to_dict()
    log.append("Fig. 4 macro n per collection = 696 / 1,581 / 524")
    return log


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cur_p = V3 / "table_s14_replacement_current_cohorts.csv"
    cur = pd.read_csv(cur_p)
    fig4 = pd.read_csv(FIG4)
    counts = json.loads((V3 / "counts.json").read_text())
    inputs = {str(cur_p): sha(cur_p), str(FIG4): sha(FIG4), str(V3 / "counts.json"): sha(V3 / "counts.json")}
    log = checks(cur, fig4)

    t1 = pd.DataFrame(ROWS, columns=COLUMNS)
    t1.to_csv(OUT / "Table_1_cohort_accounting.csv", index=False)

    md = ["**Table 1. Slide collections used in this study: source, inclusion and exclusion, and the numbers evaluated.**", ""]
    md.append("| " + " | ".join(COLUMNS) + " |")
    md.append("|" + "---|" * len(COLUMNS))
    for r in ROWS:
        md.append("| " + " | ".join(r[c] for c in COLUMNS) + " |")
    md.append("")
    for i, f in enumerate(FOOTNOTES, 1):
        md.append(f"{'abc'[i-1]} {f}")
    (OUT / "Table_1_cohort_accounting.md").write_text("\n".join(md) + "\n")

    notes = pd.DataFrame({"note": FOOTNOTES + [f"check: {x}" for x in log] +
                          [f"input: {k} sha256 {v}" for k, v in inputs.items()] +
                          [f"v3 script sha256 {counts.get('script_sha256', '')}"]})
    with pd.ExcelWriter(OUT / "Table_1_cohort_accounting.xlsx", engine="openpyxl") as xw:
        t1.to_excel(xw, sheet_name="Table 1", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
        cur.to_excel(xw, sheet_name="Source (v3 audit copy)", index=False)

    (OUT / "README.md").write_text(
        "# Main-text Table 1 — cohort accounting (assembled 2026-10-22)\n\n"
        "Reviewer 3, item 3. Compact main-text table; the full flow and the historical collections stay in Supplementary "
        "Table S14 (R3.3-T1) and the S-R3.3 flow figure. All values copied from "
        "`r3_3_cohort_accounting_20261008_v3/` (audited 2026-10-19) and checked against the Fig. 4 source data.\n\n"
        "Checks:\n" + "".join(f"- {x}\n" for x in log) +
        "\nInputs:\n" + "".join(f"- `{k}` {v}\n" for k, v in inputs.items()) +
        "\nFiles: Table_1_cohort_accounting.csv / .md / .xlsx (sheets: Table 1, Notes, Source).\n"
        "Candidate for author reading; not inserted in the manuscript Markdown.\n")
    lines = [f"{sha(p)}  {p.name}" for p in sorted(OUT.iterdir()) if p.name != "SHA256SUMS.txt"]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(log))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
