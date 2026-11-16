"""Write $PROSPER_OUT/figures_20261021/scan_to_slide_id.json (decision D1, 2026-11-04).

Lists every internal numeric scan key that was on a figure face, with the de-identified slide ID
(Comp.Bio.ID) now shown instead and the source (file + columns). Each entry of
_common.SCAN_TO_SLIDE_ID is re-verified against data/ALIdf3Ali.csv (Slide.Scan.File -> Comp.Bio.ID;
all rows of that scan must carry the same ID); the run fails on any disagreement. Display text only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FIGURES_ROOT_NAME, REPO, SCAN_TO_SLIDE_ID, prosper_out, sha256  # noqa: E402

META = REPO / "data" / "ALIdf3Ali.csv"
USED_ON = {
    "Fig. S-R2.4-N (S4) panels A/B/C": ["10113", "17284", "15182", "7899", "10108", "10111"],
    "Fig. S-R3.5-L1 (S12) panel titles": ["16875", "15155", "16867", "7333", "7330", "12151", "7443",
                                          "16039", "16676", "7292", "7347"],
    "Fig. S-R3.5-L2 (S14) panel and sub-panel titles": ["4288", "4298"],
    "Fig. S-R2.4-T3 (i) H&E title": ["7968"],
    "Fig. S-R2.4-T3 (ii) legend": ["12151", "7968"],
    "prose only (letter), not on a face": ["7965"],
}


def main():
    d = pd.read_csv(META, low_memory=False)
    d["scan"] = pd.to_numeric(d["Slide.Scan.File"], errors="coerce")
    out = {}
    for scan, sid in sorted(SCAN_TO_SLIDE_ID.items(), key=lambda kv: int(kv[0])):
        ids = set(d.loc[d["scan"] == int(scan), "Comp.Bio.ID"].astype(str))
        assert ids == {sid}, (scan, ids, sid)
        out[scan] = dict(slide_id=sid, tumor_group=str(d.loc[d["scan"] == int(scan), "Tumor.group"].iloc[0]),
                         used_on=[k for k, v in USED_ON.items() if scan in v])
    assert set(sum(USED_ON.values(), [])) == set(SCAN_TO_SLIDE_ID)
    doc = dict(decision="D1, Haoran 2026-11-04 (docs/analysis/supplementary_figure_audit_20261104.md section 4): "
                        "figure faces show the de-identified slide ID, not the internal numeric scan key",
               source=dict(path="data/ALIdf3Ali.csv", sha256=sha256(META), key_column="Slide.Scan.File",
                           value_column="Comp.Bio.ID",
                           note=f"all 21 keys resolve here; 16039 has two rows, both {SCAN_TO_SLIDE_ID['16039']}; "
                                f"16676 is recorded as {SCAN_TO_SLIDE_ID['16676']} (shorter form, as in the file)"),
               scans=out)
    p = prosper_out() / FIGURES_ROOT_NAME / "scan_to_slide_id.json"
    p.write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote", p, len(out), "scans")


if __name__ == "__main__":
    main()
