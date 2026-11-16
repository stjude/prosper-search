"""Shared helpers for the revision figure scripts.

Every figure script in this directory:
  * reads only accepted source tables under $PROSPER_OUT (or the submitted
    package under docs/prosper_submission/), never recomputes a result;
  * writes PNG (300 dpi) + PDF + its exact source rows (source_data.csv) +
    SHA256SUMS.txt + a short PROVENANCE.md into
    $PROSPER_OUT/figures_20261021/<figure>/ ;
  * is deterministic (no randomness) so re-running reproduces the hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIGURES_ROOT_NAME = "figures_20261021"

# One palette for the whole revision (colour-blind safe where it matters).
AGG_COLORS = {
    "Optimal Transport": "#1f77b4",  # blue (submitted)
    "Yottixel": "#2ca02c",  # green (submitted)
    "SAMPLER": "#9467bd",  # purple
    "Median Pooling": "#8c564b",  # brown
    "TITAN": "#ff7f0e",  # orange (submitted)
}
HIT_COLOR = "#1f77b4"  # blue circles  = hit@5
PREC_COLOR = "#e07b00"  # orange diamonds = precision@5
GREY = "#8c8c8c"

GROUP_ORDER_ALPHA = [
    "ACT/ACC", "DSRCT", "EWS", "LIVER", "MEL", "MPNST", "MRT", "NB", "NF",
    "NORMAL", "OS", "RB", "RMS", "SS", "WT",
]


# Decision D1 (2026-11-04): figure faces show the de-identified slide ID (Comp.Bio.ID), never the
# internal numeric scan key. The scan key -> slide ID table is not part of the public release: it is read
# from a local, git-ignored file, revision/config/figure_ids.local.json (or the file named by
# PROSPER_FIGURE_IDS), with the format of revision/config/figure_ids.example.json (synthetic slide IDs,
# used when no local file exists). scripts/figures/scan_to_slide_id.py re-verifies the table against
# data/ALIdf3Ali.csv (Slide.Scan.File -> Comp.Bio.ID) and writes
# $PROSPER_OUT/figures_20261021/scan_to_slide_id.json.
FIGURE_IDS_DIR = Path(__file__).resolve().parents[2] / "config"


def load_figure_ids() -> dict:
    env = os.environ.get("PROSPER_FIGURE_IDS")
    path = Path(env) if env else FIGURE_IDS_DIR / "figure_ids.local.json"
    if not env and not path.exists():
        path = FIGURE_IDS_DIR / "figure_ids.example.json"
        print("_common: figure_ids.local.json not found; using the example (synthetic slide IDs)", file=sys.stderr)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


FIGURE_IDS = load_figure_ids()
SCAN_TO_SLIDE_ID = dict(FIGURE_IDS["scan_to_slide_id"])


def slide_id(scan) -> str:
    """De-identified slide ID shown on a figure face for an internal numeric scan key."""
    return SCAN_TO_SLIDE_ID[str(scan)]


def prosper_out() -> Path:
    root = os.environ.get("PROSPER_OUT")
    if not root:
        root = str(REPO / "prosper_revision_output")
        os.environ["PROSPER_OUT"] = root
    return Path(root)


def out_dir(figure: str) -> Path:
    d = prosper_out() / FIGURES_ROOT_NAME / figure
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def style():
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 100,
        }
    )


def save(fig, figure: str, stem: str, inputs: list[Path], note: str,
         source_frames: dict[str, "object"] | None = None) -> Path:
    """Save PNG/PDF, source data, SHA256SUMS and PROVENANCE; return the dir."""
    d = out_dir(figure)
    png = d / f"{stem}.png"
    pdf = d / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", metadata={"CreationDate": None})
    written = [png, pdf]
    if source_frames:
        for name, frame in source_frames.items():
            p = d / f"{name}.csv"
            frame.to_csv(p, index=False)
            written.append(p)
    write_provenance(d, written, inputs, note, Path(sys.argv[0]), title=f"{figure} / {stem}")
    print(f"wrote {png}")
    return d


def write_provenance(d: Path, written: list[Path], inputs: list[Path], note: str,
                     script: Path, title: str | None = None) -> None:
    """Write SHA256SUMS.txt (all outputs in `written`) and PROVENANCE.md."""
    lines = []
    for p in sorted(set(written)):
        lines.append(f"{sha256(p)}  {p.name}")
    (d / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    prov = [
        f"# {title or d.name}",
        "",
        f"Generated {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')} "
        f"by `{Path(script).name}` (repo HEAD {git_head()}).",
        "",
        "## Inputs (read-only, SHA-256)",
        "",
    ]
    for p in inputs:
        p = Path(p)
        rel = str(p)
        try:
            rel = str(p.relative_to(prosper_out()))
            rel = "$PROSPER_OUT/" + rel
        except ValueError:
            try:
                rel = str(p.relative_to(REPO))
            except ValueError:
                pass
        prov.append(f"- `{rel}` — {sha256(p)}")
    prov += ["", "## Note", "", note, ""]
    (d / "PROVENANCE.md").write_text("\n".join(prov))
