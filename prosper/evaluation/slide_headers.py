"""S1a -- WSI header/scanner metadata extraction (execution spec 0002, S1 split).

Reads vendor, scanner model, MPP, objective power, scan date and level-0
dimensions directly from the SVS file headers of all 2,921 COMET slides via
``openslide``. This is extraction + a consistency check only -- it does **not**
stratify retrieval accuracy by scanner. That is S1b, gated on G2/G6, and lives
elsewhere.

Path specification (discovered 2026-08-27, not previously documented anywhere)
---------------------------------------------------------------------------
``slide_metadata()``'s id column, ``Slide.Scan.File``, is a bare numeric scan
id (e.g. ``13760``), not a filename or path. The WSI files live under a
sibling tree to ``COMET_ROOT``/``COMET_SCANS_ROOT``, resolved by
``paths.slides_root()`` (default
``/PROSPER_WORKSPACE/data/comet_final_HPC``): 2,921 symlinks named
``<Slide.Scan.File>.svs``, each pointing into
``Core/COMET/SVS_RAW_SLIDES/<scan-date>/<id>.svs``. Verified exact set
equality against ``slide_metadata()``'s 2,921 ids (no missing, no extra) on
2026-08-27. Every file in this cohort is ``.svs`` (Aperio) -- there is no
``.ndpi`` or other vendor format to handle here.

Fidelity check (F), per the spec
---------------------------------------------------------------------------
Header-derived MPP must be consistent with the paper's 40x / ~0.25
um-per-pixel tiling assumption for *every* slide. ``mpp_consistent_with_40x``
is computed per row (``objective_power == 40`` AND ``|mpp_x - 0.25| <=
MPP_TOLERANCE``) so violations can be counted and named rather than smoothed
over in an aggregate.

Two-stage LSF flow, matching ``prosper.hpc.submit``'s canary-then-fan-out
discipline
---------------------------------------------------------------------------
``extract`` processes one contiguous, 1-indexed range of a *sorted* slide-id
list (sorted, not directory order, so the LSF array index -> slide id mapping
is reproducible) and writes one chunk CSV -- avoiding concurrent writers on a
shared file. ``merge`` (cheap, run once after the sweep, no LSF needed)
concatenates every chunk into ``slide_headers.csv`` and cross-checks coverage
against ``slide_metadata()``.
"""

from __future__ import annotations

import os
import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import paths

try:
    import openslide
except ImportError:  # pragma: no cover -- base env is expected to have it
    openslide = None

# The paper's tiling assumption: 40x, ~0.25 um/px.
EXPECTED_OBJECTIVE_POWER = 40.0
EXPECTED_MPP = 0.25
# Wide enough to not flag Aperio's normal per-scanner calibration spread
# (observed ~0.2455-0.2530 um/px at 40x) while still catching a 20x slide
# (~0.50 um/px) with margin.
MPP_TOLERANCE = 0.05

FIELDNAMES = [
    "slide_id", "file_path", "resolved_path", "status", "error",
    "vendor", "scanner_model", "scanscope_id", "dsr_id",
    "mpp_x", "mpp_y", "objective_power", "scan_date", "scan_time",
    "width", "height", "level_count",
    "mpp_consistent_with_40x",
    "raw_properties_json",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _parse_aperio_date(props: dict) -> tuple:
    """``aperio.Date`` is ``MM/DD/YY``. Keep the raw string on a parse failure
    rather than silently dropping it -- an unparsable date is itself a finding.
    """
    date_s = props.get("aperio.Date")
    time_s = props.get("aperio.Time")
    iso_date = None
    if date_s:
        try:
            iso_date = datetime.strptime(date_s, "%m/%d/%y").date().isoformat()
        except ValueError:
            iso_date = date_s
    return iso_date, time_s


def _scanner_model(vendor: Optional[str], props: dict) -> Optional[str]:
    """Best-effort scanner-model proxy, vendor-specific and deliberately
    explicit about being a proxy.

    Aperio SVS carries no ``model`` property. The ICC profile name (e.g.
    ``AT2``) is the closest available proxy for the scanner family; the
    ``ScanScope ID`` / ``DSR ID`` properties identify the physical
    instrument / capture workstation and are kept as their own columns so
    this proxy choice is auditable rather than folded silently into one
    field. Branches for other vendors are included for completeness even
    though this cohort has none.
    """
    if vendor == "aperio":
        return props.get("aperio.ICC Profile")
    if vendor == "hamamatsu":
        return props.get("hamamatsu.Model")
    if vendor == "leica":
        return props.get("leica.DeviceModel") or props.get("leica.DeviceVersion")
    if vendor == "philips":
        return props.get("philips.DICOM_MANUFACTURER_MODEL_NAME")
    return None


def read_header(slide_id, wsi_path: Path) -> dict:
    """Read one slide's header. Never raises -- failures are recorded in the
    row (``status``/``error``) so a bad slide does not lose the whole chunk."""
    row = {k: None for k in FIELDNAMES}
    row["slide_id"] = str(slide_id)
    row["file_path"] = str(wsi_path)

    try:
        row["resolved_path"] = str(wsi_path.resolve())
    except OSError:
        row["resolved_path"] = str(wsi_path)

    if openslide is None:
        row["status"] = "error"
        row["error"] = "openslide not importable in this environment"
        return row

    if not wsi_path.exists():
        row["status"] = "missing"
        row["error"] = "file does not exist (broken symlink or moved input)"
        return row

    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        return row

    try:
        props = dict(slide.properties)
        vendor = props.get("openslide.vendor")
        mpp_x_raw = props.get("openslide.mpp-x")
        mpp_y_raw = props.get("openslide.mpp-y")
        obj_power_raw = props.get("openslide.objective-power")
        width, height = slide.dimensions
        level_count = slide.level_count

        mpp_x = float(mpp_x_raw) if mpp_x_raw is not None else None
        mpp_y = float(mpp_y_raw) if mpp_y_raw is not None else None
        obj_power = float(obj_power_raw) if obj_power_raw is not None else None

        iso_date, scan_time = (None, None)
        if vendor == "aperio":
            iso_date, scan_time = _parse_aperio_date(props)

        consistent = None
        if mpp_x is not None and obj_power is not None:
            consistent = (
                abs(obj_power - EXPECTED_OBJECTIVE_POWER) < 1e-6
                and abs(mpp_x - EXPECTED_MPP) <= MPP_TOLERANCE
            )

        row.update({
            "status": "ok",
            "error": None,
            "vendor": vendor,
            "scanner_model": _scanner_model(vendor, props),
            "scanscope_id": props.get("aperio.ScanScope ID"),
            "dsr_id": props.get("aperio.DSR ID"),
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "objective_power": obj_power,
            "scan_date": iso_date,
            "scan_time": scan_time,
            "width": width,
            "height": height,
            "level_count": level_count,
            "mpp_consistent_with_40x": consistent,
            "raw_properties_json": json.dumps(props, sort_keys=True),
        })
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"header read failed after open: {type(e).__name__}: {e}"
    finally:
        try:
            slide.close()
        except Exception:
            pass
    return row


def _sorted_slide_ids(slides_dir: Optional[str] = None, ext: str = ".svs") -> list:
    """Deterministic id order for the whole S1a sweep: ``slide_metadata()``'s
    id column, sorted. Sorted rather than directory-listing order so the LSF
    array index -> slide id mapping is reproducible independent of filesystem
    iteration order.

    With ``slides_dir`` (external cohorts, 2026-08-29) the ids are the sorted
    ``*<ext>`` basenames of that directory instead."""
    import pandas as pd

    if slides_dir:
        import glob
        names = sorted(os.path.basename(p) for p in glob.glob(
            os.path.join(slides_dir, f"*{ext}")))
        return [n[:-len(ext)] for n in names]
    meta = pd.read_csv(paths.require(paths.slide_metadata(), "slide metadata"))
    ids = meta["Slide.Scan.File"].tolist()
    return sorted(ids, key=str)


def extract_range(start: int, end: int, task: str, out: Optional[Path] = None,
                  slides_dir: Optional[str] = None, ext: str = ".svs") -> Path:
    """Process slides ``[start, end]`` (1-indexed, inclusive) of the sorted id
    list and write one chunk CSV. Disjoint ranges across array elements avoid
    any concurrent-writer hazard on a shared output file."""
    ids = _sorted_slide_ids(slides_dir, ext)
    n = len(ids)
    if not (1 <= start <= end <= n):
        raise ValueError(f"range [{start},{end}] out of bounds for {n} slides")

    chunk_dir = paths.output_root() / task / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out_path = out or (chunk_dir / f"chunk_{start:05d}_{end:05d}.csv")

    rows = []
    for i in range(start, end + 1):
        sid = ids[i - 1]
        path = (Path(slides_dir) / f"{sid}{ext}") if slides_dir else paths.slide_wsi_path(sid)
        rows.append(read_header(sid, path))

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    return out_path


def _to_bool(v) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return None
    except TypeError:
        pass
    s = str(v).strip()
    if s in ("True", "true", "1"):
        return True
    if s in ("False", "false", "0"):
        return False
    return None


def merge(task: str, out: Optional[Path] = None,
          slides_dir: Optional[str] = None, ext: str = ".svs") -> dict:
    """Concatenate every ``chunks/chunk_*.csv`` into ``slide_headers.csv`` and
    cross-check coverage against ``slide_metadata()``. Returns a summary dict
    (also written to ``merge_summary.json``) with the counts the findings
    note needs: status breakdown, vendor/model/scanner counts, and the MPP
    fidelity-check verdict."""
    import pandas as pd

    task_dir = paths.output_root() / task
    chunk_dir = task_dir / "chunks"
    chunks = sorted(chunk_dir.glob("chunk_*.csv"))
    if not chunks:
        raise SystemExit(f"no chunk CSVs found under {chunk_dir}")

    frames = [pd.read_csv(c, dtype={"slide_id": str}) for c in chunks]
    df = pd.concat(frames, ignore_index=True)
    df["mpp_consistent_with_40x"] = df["mpp_consistent_with_40x"].map(_to_bool)

    expected_ids = set(str(x) for x in _sorted_slide_ids(slides_dir, ext))
    got_ids = set(df["slide_id"])
    missing = expected_ids - got_ids
    dup = int(df["slide_id"].duplicated().sum())

    out_path = out or (task_dir / "slide_headers.csv")
    df.to_csv(out_path, index=False)

    n_true = int((df["mpp_consistent_with_40x"] == True).sum())    # noqa: E712
    n_false = int((df["mpp_consistent_with_40x"] == False).sum())  # noqa: E712
    n_unknown = len(df) - n_true - n_false

    summary = {
        "n_chunks": len(chunks),
        "n_rows": len(df),
        "n_expected": len(expected_ids),
        "n_missing_from_output": len(missing),
        "missing_slide_ids": sorted(missing),
        "n_duplicate_rows": dup,
        "status_counts": df["status"].value_counts(dropna=False).to_dict(),
        "n_mpp_consistent": n_true,
        "n_mpp_inconsistent": n_false,
        "n_mpp_unknown": n_unknown,
        "inconsistent_slide_ids": sorted(
            df.loc[df["mpp_consistent_with_40x"] == False, "slide_id"].tolist()  # noqa: E712
        ),
        "vendor_counts": df["vendor"].value_counts(dropna=False).to_dict(),
        "scanner_model_counts": df["scanner_model"].value_counts(dropna=False).to_dict(),
        "scanscope_id_counts": df["scanscope_id"].value_counts(dropna=False).to_dict(),
        "objective_power_counts": df["objective_power"].value_counts(dropna=False).to_dict(),
        "scan_date_min": df["scan_date"].dropna().min() if df["scan_date"].notna().any() else None,
        "scan_date_max": df["scan_date"].dropna().max() if df["scan_date"].notna().any() else None,
        "prosper_commit": _git_commit(),
        "out_path": str(out_path),
    }
    (task_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="S1a: WSI header/scanner-metadata extraction + MPP consistency check"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="process one contiguous range of slides")
    ex.add_argument("--start", type=int, required=True, help="1-indexed inclusive")
    ex.add_argument("--end", type=int, required=True, help="1-indexed inclusive")
    ex.add_argument("--task", default="s1a_slide_headers")
    ex.add_argument("--out", default=None)

    mg = sub.add_parser("merge", help="combine all chunk CSVs into slide_headers.csv")
    mg.add_argument("--task", default="s1a_slide_headers")
    mg.add_argument("--out", default=None)

    ct = sub.add_parser("count", help="print the total slide count (for array sizing)")

    # External cohorts (2026-08-29): ids from a directory listing instead of
    # slide_metadata(); files are <slides-dir>/<id><ext>.
    for sp in (ex, mg, ct):
        sp.add_argument("--slides-dir", default=None)
        sp.add_argument("--ext", default=".svs")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "count":
        print(len(_sorted_slide_ids(args.slides_dir, args.ext)))
        return 0
    if args.cmd == "extract":
        out_path = extract_range(
            args.start, args.end, args.task,
            Path(args.out) if args.out else None,
            args.slides_dir, args.ext)
        print(f"[slide_headers] wrote {out_path}")
        return 0
    if args.cmd == "merge":
        summary = merge(args.task, Path(args.out) if args.out else None,
                        args.slides_dir, args.ext)
        print(json.dumps(summary, indent=2, default=str))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
