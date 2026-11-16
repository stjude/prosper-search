#!/usr/bin/env python3
"""Package index for supp_tables_20261109/B_external_perturbation_finelabel: README.md, manifest.json and a
recursive SHA256SUMS.txt. Run after the five table builders; reads their outputs, computes nothing."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _supp_common_20261109 import ACCESS, BUILD_DATE, OUT_ROOT, PKG, REPO, sha  # noqa: E402

TABLES = [("S6", "S6_S19", "build_S6_nb_factors_20261109.py"),
          ("S7", "S7_S20", "build_S7_nb_confusion_20261109.py"),
          ("S8", "S8_S21", "build_S8_ccdi_eligibility_sensitivity_20261109.py"),
          ("S-R2.1-C", "S-R2.1-C_S15", "build_SR21C_perturbation_suite_20261109.py"),
          ("S28", "S28_S9_v2", "build_S28_v2_finelabel_inventory_20261109.py")]


def main() -> None:
    entries, lines = [], []
    for wid, sub, script in TABLES:
        d = PKG / sub
        xl = sorted(d.glob("*.xlsx"))
        assert len(xl) == 1, (d, xl)
        wb = openpyxl.load_workbook(xl[0], read_only=True)
        cap = {r[0]: r[1] for r in wb["caption"].iter_rows(min_row=2, values_only=True)}
        sheets = wb.sheetnames
        wb.close()
        rep = (d / "BUILD_REPORT.md").read_text(encoding="utf-8")
        m = re.search(r"\*\*Checks: (\d+) total, (\d+) PASS, (\d+) FAIL\.\*\*", rep)
        assert m and m.group(3) == "0", (wid, m)
        n_open = rep.split("## Open items", 1)[1].split("\n## ", 1)[0].count("\n- ")
        sums = (d / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        for ln in sums:
            h, name = ln.split("  ", 1)
            assert sha(d / name) == h, (wid, name)
        bind = json.loads((d / "input_bindings.json").read_text(encoding="utf-8"))
        assert bind["script_sha256"] == sha(REPO / "scripts/supp_tables" / script), wid
        files = sorted(p.name for p in d.iterdir() if p.is_file())
        entries.append({"workspace_id": wid, "final_number": cap["final_number"], "title": cap["title"], "directory": sub,
                        "workbook": xl[0].name, "workbook_sha256": sha(xl[0]), "sheets": sheets, "files": files,
                        "builder": f"scripts/supp_tables/{script}", "builder_sha256": bind["script_sha256"],
                        "checks_total": int(m.group(1)), "checks_pass": int(m.group(2)), "open_items": n_open,
                        "n_inputs_pinned": len(bind["inputs"]), "caption_status": cap["caption_status"]})
        lines.append(f"| {wid} | {cap['final_number']} | {cap['title']} | `{sub}/{xl[0].name}` | {m.group(2)}/{m.group(1)} | {n_open} |")

    manifest = {"package": str(PKG.relative_to(OUT_ROOT)), "build_date": BUILD_DATE,
                "numbering_source": "scripts/figures/supp_table_numbering.json ('new' field)",
                "common_module": "scripts/supp_tables/_supp_common_20261109.py",
                "common_module_sha256": sha(REPO / "scripts/supp_tables/_supp_common_20261109.py"),
                "index_builder_sha256": sha(Path(__file__).resolve()), "access": ACCESS, "tables": entries}
    (PKG / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    readme = [
        "# Supplementary-table workbooks \u2014 package B (external NB cohort, perturbation suite, fine-label inventory)", "",
        f"Built {BUILD_DATE} from accepted, already-computed outputs. No new statistic was computed: values are copied, rounded "
        "for display or re-tabulated (counts) from retained per-query outputs; no LSF job; no reader-study data. Files are named by "
        "**workspace** id; the final number (`scripts/figures/supp_table_numbering.json`, field `new`) is recorded in each caption sheet and in `manifest.json`.", "",
        "| Workspace id | Final number | Title | Workbook | Checks PASS | Open items |", "|---|---|---|---|---|---|", *lines, "",
        "Each table directory holds the workbook (first sheet `caption`: workspace_id, final_number, title, caption, footnotes, sources, "
        "built_by/build_date; then data sheets; unrounded values in `_raw` sheets/CSVs), one CSV per sheet, `input_bindings.json` "
        "(every input pinned by SHA-256, plus script/common-module hashes and Python/pandas versions), `BUILD_REPORT.md` (summary, open items, "
        "sources, caption mirror, every check) and `SHA256SUMS.txt`.", "",
        "Caption text is a draft mirror assembled from the edit-sheet specification and the current legends; the lead's Supplementary "
        "Information Markdown is the source of truth for wording.", "",
        "Rebuild (deterministic; xlsx zip timestamps and core properties are fixed):", "",
        "```bash",
        "export PROSPER_OUT=\"$PWD/prosper_revision_output\"",
        *[f"/HOME/miniconda3/bin/python3 scripts/supp_tables/{s}" for _, _, s in TABLES],
        "/HOME/miniconda3/bin/python3 scripts/supp_tables/build_B_index_20261109.py",
        "```", "",
        f"Access: {ACCESS}", "",
    ]
    (PKG / "README.md").write_text("\n".join(readme), encoding="utf-8")

    allf = sorted(p for p in PKG.rglob("*") if p.is_file() and p != PKG / "SHA256SUMS.txt")
    (PKG / "SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.relative_to(PKG)}\n" for p in allf), encoding="utf-8")
    print(f"index: {len(entries)} tables; {len(allf)} files hashed in {PKG}")


if __name__ == "__main__":
    main()
