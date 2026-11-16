#!/usr/bin/env python3
"""Source-data bundle for the supplementary figures (revision).

For every Supplementary Fig. S1–S16 of the revision numbering (scripts/figures/supp_figure_numbering.json, field
`pass026`, key `render`), copies the CSV files that sit beside the render (top level of the render directory, not
superseded/candidate subdirectories) and the directory's PROVENANCE file into

    $PROSPER_OUT/supp_package_20261110/source_data/Supplementary_Fig_<S…>/

Figures that share one render directory (S1/S2, S9/S10) share one folder. Image-only figures (tile montages,
screenshots, the submitted count figure) are listed in the index with no files. Writes SOURCE_DATA_INDEX.md,
source_data_manifest.json and refreshes the package SHA256SUMS.txt. Read-only on every input.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packaging_common_20261109 as pc  # noqa: E402

REPO, OUT = pc.REPO, pc.OUT_ROOT
FIG_MAP = REPO / "scripts/figures/supp_figure_numbering.json"
PKG = OUT / "supp_package_20261110"
DEST = PKG / "source_data"
IMAGE_ONLY_NOTE = {
    "S11": "representative tiles (images) of the Fig. 5 pair correspondence clusters",
    "S12": "difference maps (images); the matched-cost decomposition by annotation class is Supplementary Table S24",
    "S13": "linked tile pairs (images); sampled from the Optimal Transport plans of the Fig. 5 pairs",
    "S15": "submitted figure (tumor-group counts); the counts are those of Supplementary Table S3",
    "S16": "instrument screenshots",
}


def main() -> int:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    fmap = json.load(open(FIG_MAP))["map"]
    by_fig: dict[str, Path] = {}
    for wid, e in fmap.items():
        n = e.get("pass026")
        if isinstance(n, str) and re.fullmatch(r"S\d+", n) and e.get("render"):
            r = e["render"]
            by_fig[n] = (REPO / r) if r.startswith("docs/") else (OUT / r)
    order = sorted(by_fig, key=lambda s: int(s[1:]))
    groups: dict[Path, list[str]] = {}
    for n in order:
        groups.setdefault(by_fig[n].parent, []).append(n)
    index = ["# Source data for the supplementary figures", "",
             "One folder per render directory; figures sharing a directory share a folder. Files are copied unchanged "
             "from beside the figure render (`PROVENANCE.md` where the figure script wrote one).", "",
             "| Figure(s) | Folder | Files | Note |", "|---|---|---|---|"]
    manifest = {}
    for d, figs in groups.items():
        files = sorted(p for p in d.glob("*.csv") if p.is_file())
        # a directory shared with a main figure (fig8 + S14): keep only the files tagged with the supplementary number
        tagged = [p for p in files if any(f"_{f}_" in p.name for f in figs)]
        if tagged:
            files = tagged
        prov = sorted(p for p in d.iterdir() if p.is_file() and p.name.upper().startswith("PROVENANCE"))
        label = "Supplementary_Fig_" + "_".join(figs)
        note = "; ".join(IMAGE_ONLY_NOTE[f] for f in figs if f in IMAGE_ONLY_NOTE)
        if not files:
            index.append(f"| {', '.join(figs)} | — | — | image-only: {note or 'no tabular source data'} |")
            manifest[label] = {"figures": figs, "render_dir": pc.rel(d), "files": [], "note": note}
            continue
        sub = DEST / label
        sub.mkdir()
        copied = []
        for p in files + prov:
            shutil.copyfile(p, sub / p.name)
            copied.append({"file": p.name, "sha256": pc.sha256(p), "source": pc.rel(p)})
        index.append(f"| {', '.join(figs)} | `{label}/` | {', '.join(c['file'] for c in copied)} | {note} |")
        manifest[label] = {"figures": figs, "render_dir": pc.rel(d), "files": copied, "note": note}
    (DEST / "SOURCE_DATA_INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    (DEST / "source_data_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    if (PKG / "SHA256SUMS.txt").exists():
        pc.write_sha256sums(PKG)
    n_files = sum(len(v["files"]) for v in manifest.values())
    print(f"wrote {DEST}: {len(groups)} folders/entries, {n_files} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
