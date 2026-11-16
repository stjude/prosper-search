#!/usr/bin/env python3
"""A_packaging (2026-11-09): rerun the four table builders and the S26 v2 audit, then write the top-level
manifest.json ({workspace_id: {final_number, path, sha256}}), README.md and SHA256SUMS.txt.

Run from the repository root:
    export PROSPER_OUT="$PWD/prosper_revision_output"
    /HOME/miniconda3/bin/python3 scripts/supp_tables/finalize_A_packaging_20261109.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packaging_common_20261109 import BUILD_DATE, NUMBERING, PKG, REPO, SI_MD, numbering, rel, sha256, write_sha256sums  # noqa: E402

HERE = Path(__file__).resolve().parent
STEPS = [("S-R2.1-E", "build_S-R2.1-E_merged_20261109.py", "S-R2.1-E", "Table_S-R2.1-E.xlsx",
          "Encoder × aggregation grid and per-group results (submitted sheets 1–4 and 5–6 merged)"),
         ("S16", "build_S16_classifier_20261109.py", "S16", "Table_S16.xlsx",
          "Tumor-tile classifier: submitted within-slide AUROC, slide-held-out values and TF-2 cross-fitted heads"),
         ("S14", "build_S14_cohort_accounting_20261109.py", "S14", "Table_S14.xlsx",
          "Cohort accounting with material and label sources"),
         ("S15", "build_S15_ccdi_mapping_20261109.py", "S15", "Table_S15.xlsx",
          "CCDI-MCI diagnosis-string mapping to COMET tumor groups"),
         ("S26", "audit_S26_v2_20261109.py", "S26_audit", None,
          "Independent audit of Table S26 v2 (subtype retrieval within a known family); workbook not rebuilt")]


def main() -> int:
    py = sys.executable
    runs = {}
    for wid, script, d, _, _ in STEPS:
        p = subprocess.run([py, str(HERE / script)], cwd=REPO, capture_output=True, text=True)
        runs[wid] = (p.returncode, (p.stdout.strip().splitlines() or [""])[-1], p.stderr.strip()[-2000:])
        print(f"{wid}: rc {p.returncode} {runs[wid][1]}")
        if p.returncode:
            print(p.stderr[-3000:], file=sys.stderr)
            return 1

    manifest, rows = {}, []
    for wid, script, d, xlsx, desc in STEPS:
        mr = json.loads((PKG / d / "manifest_row.json").read_text())[wid]
        path = Path(mr["path"].replace("$PROSPER_OUT", str(PKG.parents[1])))
        num = numbering(wid)
        assert num["current"] == mr["final_number"], (wid, num["current"], mr["final_number"])
        e = {"final_number": num["current"], "path": mr["path"], "sha256": sha256(path)}
        if wid == "S26":
            e.update(role=mr["role"], audit=mr["audit"], audit_sha256=sha256(PKG / d / "AUDIT_S26_v2.md"))
        manifest[wid] = e
        rep = (PKG / d / ("AUDIT_S26_v2.md" if wid == "S26" else "BUILD_REPORT.md")).read_text(encoding="utf-8")
        m = re.search(r"^## Checks \((.*)\)\s*$", rep, re.M)
        if m:
            checks = m.group(1)
        else:
            v = re.search(r"\*\*(\d+) of (\d+) printed values reproduce\*\*.*?checks (\d+) PASS, (\d+) FAIL", rep, re.S)
            checks = f"{v.group(1)}/{v.group(2)} printed values reproduce; {v.group(3)} PASS, {v.group(4)} FAIL"
        n_open = len(re.findall(r"^- ", rep.split("## Open items", 1)[1], re.M)) if "## Open items" in rep else 0
        rows.append((wid, num["current"], num["history"], desc, f"`{d}/`", checks, n_open))
    (PKG / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    L = ["# A_packaging — supplementary tables S-R2.1-E, S16, S14, S15 and the S26 v2 audit", "",
         f"Built {BUILD_DATE} by `scripts/supp_tables/finalize_A_packaging_20261109.py`, which reruns the per-table "
         "builders below and writes `manifest.json` and `SHA256SUMS.txt`. Assembly and audit only: no new scientific "
         "computation, no LSF jobs, no reader-study data; every input is pinned by SHA-256 in the table's "
         "`input_bindings.json`; values are copied from accepted sources (raw unrounded values in CSVs beside each "
         "workbook or in `raw/` / `audit/`).", "",
         f"Final numbers are the current field of `{rel(NUMBERING)}` at build time (sha256 `{sha256(NUMBERING)[:12]}…`). "
         "The brief's numbers (S3 / S12 / S16 / S25 / S8) are the earlier pass; earlier numbers are listed below. "
         f"Caption wording: the lead's SI Markdown `{rel(SI_MD)}` (sha256 at build `{sha256(SI_MD)[:12]}…`) is the "
         "source of truth; each workbook's `caption` sheet is a draft mirror and the BUILD_REPORT lists where the SI "
         "caption and the workbook disagree.", "",
         "| Workspace id | Final number | Earlier numbers | Content | Directory | Checks | Open items |",
         "|---|---|---|---|---|---|---|"]
    L += [f"| {w} | {n} | {h} | {c} | {d} | {k} | {o} |" for w, n, h, c, d, k, o in rows]
    L += ["", "Each table directory holds the workbook (first sheet `caption`), per-sheet CSVs, `input_bindings.json`, "
          "`BUILD_REPORT.md` (checks, SI caption consistency, open items), `manifest_row.json` and `SHA256SUMS.txt`. "
          "`S26_audit/` holds `AUDIT_S26_v2.md`, `audit_values.csv` (each printed value against the re-derived and raw "
          "values) and its bindings; the audited workbook stays in `supp_tables_20261103/`.", "",
          "## Access", "",
          "- Restricted: `S16/Sheet1_units` and its CSV carry institutional (SJ) patient identifiers; `S26_audit` names "
          "three CCDI-MCI participant ids. `S14/audit/` and `S15/audit/` hold internal audit copies. Keep the whole "
          "directory under the workspace's controlled storage; packaging is not permission to publish.",
          "- Builders read `comet-*` trees and the similarity matrices read-only; nothing here was committed.", "",
          "## Rebuild", "",
          "```bash", 'export PROSPER_OUT="$PWD/prosper_revision_output"',
          "/HOME/miniconda3/bin/python3 scripts/supp_tables/finalize_A_packaging_20261109.py", "```", "",
          "Workbook bytes are deterministic (fixed document properties and zip times); BUILD_REPORT and audit run "
          "timestamps change between runs. Because the SI Markdown and numbering file are live, rerun before use; the "
          "bound hashes record which versions a build saw.", ""]
    (PKG / "README.md").write_text("\n".join(L), encoding="utf-8")
    write_sha256sums(PKG)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
