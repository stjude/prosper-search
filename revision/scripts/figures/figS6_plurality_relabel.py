"""Supplementary Fig. S6 (simulated query perturbations): header relabel only (revision preparation).

The adopted S6 image ($PROSPER_OUT/figures_20261021/figS_perturbation/Figure_S_perturbation.png,
sha256 e6b2f635...) was rendered on 2026-11-10 by `figS_perturbation_candidates.render_forest(...,
band=False)` (revision) and saved with `_common.save` under the name Figure_S_perturbation. Its
middle column header reads "Δ majority-vote accuracy"; the metric is plurality vote (SI close audit
si-figs F2). This driver re-runs exactly that render path from the same retained source CSVs and
changes nothing but the middle header text:

  --variant original   unchanged header (reproduction check against the adopted PNG)
  --variant plurality  header "Δ plurality-vote accuracy (mMV@5)"

`figS_perturbation_candidates.py` and `figS_perturbation.py` are not edited; the header text is
passed by overriding the module's METRICS label for this run only.

Outputs  $PROSPER_OUT/figures_20261021/figS6_plurality_relabel_20261116/<variant>/
         Figure_S_perturbation.{png,pdf}, source_data.csv, legibility_check.json, SHA256SUMS.txt,
         PROVENANCE.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import out_dir, save, sha256, style  # noqa: E402
import figS_perturbation as base  # noqa: E402
import figS_perturbation_candidates as cand  # noqa: E402

ROOT = "figS6_plurality_relabel_20261116"
LABELS = {"original": "majority-vote accuracy", "plurality": "plurality-vote accuracy (mMV@5)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(LABELS), required=True)
    args = ap.parse_args()
    assert [m[2] for m in cand.METRICS] == ["hit@5", "majority-vote accuracy", "precision@5"], cand.METRICS
    cand.METRICS = [(ep, key, LABELS[args.variant] if key == "mmv" else title) for ep, key, title in cand.METRICS]
    style()
    d, ident, _groups, _strata, stim = cand.load_all()
    dose = cand.dose_labels(stim)
    fig, src, desc = cand.render_forest(d, ident, dose, band=False)
    key = f"{ROOT}/{args.variant}"
    out = out_dir(key)
    leg = base.legibility(fig, out)
    note = (f"Supplementary Fig. S6, variant `{args.variant}`: middle header 'Δ {LABELS[args.variant]}'. "
            f"Render path of the adopted image (figS_perturbation_candidates.render_forest, band=False; {desc}); "
            f"values copied from the retained sources, display arithmetic only. Scripts: "
            f"figS6_plurality_relabel.py {sha256(Path(__file__))}, figS_perturbation_candidates.py "
            f"{sha256(Path(cand.__file__))}, figS_perturbation.py {sha256(Path(base.__file__))}. "
            f"Legibility pass = {leg['pass']} (overlaps: {leg['overlaps']}).")
    inputs = [base.V002, base.MMV, base.MMV_IDENT, cand.SUPP, cand.STIM]
    save(fig, key, "Figure_S_perturbation", inputs, note, {"source_data": src})
    with open(out / "SHA256SUMS.txt", "a") as fh:
        fh.write(f"{sha256(out / 'legibility_check.json')}  legibility_check.json\n")
    plt.close(fig)
    print(args.variant, "legibility", leg["pass"], leg["overlaps"][:6])


if __name__ == "__main__":
    main()
