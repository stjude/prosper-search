#!/usr/bin/env python3
"""Top-level index for $PROSPER_OUT/supp_tables_20261109/C_localization/: README.md, manifest.json and
SHA256SUMS.txt over both table directories. Run after build_S-R3.5-L1_20261109.py and
build_S-R3.5-L2_20261109.py; reads only their outputs.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c_localization_common_20261109 import BUILD_DATE, BUILT_BY, GROUP_DIR, rel, sha256  # noqa: E402

TABLES = [("S-R3.5-L1", "build_S-R3.5-L1_20261109.py"), ("S-R3.5-L2", "build_S-R3.5-L2_20261109.py")]
HERE = Path(__file__).resolve().parent

manifest = {"group": "C_localization", "build_date": BUILD_DATE, "built_by": BUILT_BY,
            "index_builder": rel(Path(__file__)), "index_builder_sha256": sha256(Path(__file__)),
            "common_module": rel(HERE / "c_localization_common_20261109.py"),
            "common_module_sha256": sha256(HERE / "c_localization_common_20261109.py"), "tables": []}
readme = ["# Supplementary tables — group C_localization (assembled 2026-11-09; candidates, not approval)", "",
          "Localization-study tables assembled from accepted outputs; nothing re-estimated. Each workbook's first sheet is "
          "`caption` (field | text: workspace_id, final_number, title, caption, footnotes, sheets, caption_status, sources, "
          "built_by/date), followed by the data sheets. Every data sheet is also a CSV beside the workbook; unrounded source "
          "values are in the `_raw` CSVs. Files are named by workspace id; the final number comes from "
          "`scripts/figures/supp_table_numbering.json` and is applied at assembly. Captions are builder drafts; the lead's "
          "Supplementary Information Markdown is the source of truth for caption wording.", "",
          "Per table: `input_bindings.json` (every input with sha256), `value_checks.csv` (each printed manuscript value against "
          "its source field), `BUILD_REPORT.md` (sources, spec, value mapping, checks, open items) and `SHA256SUMS.txt`.", ""]
ok_all = True
for wid, script in TABLES:
    d = GROUP_DIR / wid
    bind = json.loads((d / "input_bindings.json").read_text())
    vc = pd.read_csv(d / "value_checks.csv")
    sums = dict(l.split()[::-1] for l in (d / "SHA256SUMS.txt").read_text().splitlines())
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    stale = [f for f in files if f != "SHA256SUMS.txt" and sums.get(f) != sha256(d / f)]
    bsha = sha256(HERE / script)
    fresh = bind["builder_sha256"] == bsha
    ok_all &= not stale and fresh and (vc.result == "pass").all()
    cap = pd.read_csv(d / f"Table_{wid}__caption.csv").set_index("field")["text"]
    manifest["tables"].append({
        "workspace_id": wid, "final_number": cap["final_number"].split(" (")[0], "title": cap["title"],
        "directory": rel(d), "workbook": f"Table_{wid}.xlsx", "builder": rel(HERE / script), "builder_sha256": bsha,
        "builder_matches_bindings": fresh, "n_inputs": len(bind["inputs"]),
        "value_checks": {"pass": int((vc.result == "pass").sum()), "total": len(vc)},
        "files": {f: sha256(d / f) for f in files}})
    readme += [f"## {wid} — {cap['final_number'].split(' (')[0]}: {cap['title']}", "",
               f"Directory `{wid}/`; workbook `Table_{wid}.xlsx`; builder `scripts/supp_tables/{script}`; "
               f"{len(bind['inputs'])} bound inputs; value checks {int((vc.result == 'pass').sum())}/{len(vc)} pass.", "",
               "Sheet → CSV:", ""]
    readme += [f"- `{s}` → `{c}`" for s, c in
               [(n[len(f"Table_{wid}__"):-4], n) for n in files if n.startswith(f"Table_{wid}__") and n.endswith(".csv")]]
    readme.append("")
readme += ["## Reproduce", "", "```bash", "export PROSPER_OUT=\"$PWD/prosper_revision_output\"",
           *[f"/HOME/miniconda3/bin/python3 scripts/supp_tables/{s}" for _, s in TABLES],
           f"/HOME/miniconda3/bin/python3 scripts/supp_tables/{Path(__file__).name}", "```", "",
           "Outputs are byte-deterministic (fixed workbook timestamps; no wall-clock values). Restricted data: per-reference "
           "rows carry internal slide indices and tumour group only (patient codes dropped); access conditions are an open "
           "item in each BUILD_REPORT. Not for upload or commit without authorization.", ""]
(GROUP_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")
(GROUP_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
lines = [f"{sha256(p)}  {p.relative_to(GROUP_DIR)}" for p in sorted(GROUP_DIR.rglob("*"))
         if p.is_file() and p != GROUP_DIR / "SHA256SUMS.txt"]
(GROUP_DIR / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"index: {len(lines)} files hashed; tables consistent: {ok_all}")
sys.exit(0 if ok_all else 1)
