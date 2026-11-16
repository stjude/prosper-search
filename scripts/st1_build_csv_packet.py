"""ST1.5: flat CSV packet for the pathologist pilot (replaces the HTML packet).

Five files + an images folder, no hyperlinks, nothing depends on anything else
being open in a browser:

  01_cases.csv          one row per case, reading order: id, age, site,
                        presentation, query image filename, full-slide path
                        (written Windows-style via --windows-map, user ruling
                        2026-09-14: the pathologist pastes it on Windows).
                        No diagnosis, no difficulty, no category (blinded).
  02_search_results.csv one row per (case, rank 1..10): retrieved diagnosis,
                        SJ sample id + Pecan portal link, distance, neighbour
                        metadata, thumbnail filename, full-slide path.
  03_response_sheet.csv the answer template (copied, same reading order).
  04_instructions.md    copied from the workspace docs (reader-facing).
  LLM summaries/        one <case_id>.txt per case: the blinded production
                        summary ("LLM", not "AI", per user terminology ruling
                        2026-09-08; sourced from st1_llm_summaries_v2 — the
                        single-shot manuscript summarizer, ST1.7b).
  images/               query previews (--no-grid, the default packet since
                        2026-09-03, omits the ROI reference grid and reshapes
                        the response sheet to the packet standard: grid q8/q9
                        dropped, summary-accuracy q5 dropped 2026-09-08,
                        renumbered to q1-q6 + q7_comments; without the flag
                        the grid is burned into the PNGs and the template's
                        q1-q10 pass through) +
                        neighbours/<Comp.Bio.ID>.jpg (scan id appended only
                        on a sample-id collision).
  qupath_project/       with --qupath-slides-dir: a QuPath 0.5 project whose
                        22 entries (case01_<id> .. case22_<id>, reading order)
                        point at the pyramidal viewing copies made by
                        st1_convert_query_slides.py, as Windows UNC file URIs;
                        pixel calibration comes from the conversion log (the
                        DICOM-recovered mpp), so QuPath's scale bar is right
                        even where the OpenSlide build ignores generic-tiff
                        resolution tags.

Pecan links: scan id -> Comp.Bio.ID from the rich table, falling back to the co-author metadata CSV (--ali-map); the portal serves HTTP 200 for any path, so the
mapping, not the status code, is the correctness guarantee.

Reading order comes from the already-frozen response_template.csv so the
sheet, the cases file, and the dossier reader index can never disagree.
CSVs are written utf-8-sig so Excel renders them without an import dialog.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BLIND_TOKENS = ("FAILURE", "PARTIAL", "SUCCESS", "difficulty", "Difficulty",
                "tumor_group", "discrepan")


def load_font(size: int = 22):
    for cand in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"):
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


PECAN = "https://pecan.stjude.cloud/histology/sample/{}"

# default Linux-mount -> Windows-UNC mapping for reader-facing slide paths.
# \\FILESERVER\share is a placeholder for the SMB export of the cluster
# file system — override with --windows-map for the local share.
DEFAULT_WINDOWS_MAP = ["/PROSPER_STORAGE=\\\\FILESERVER\\share"]


def to_windows(path: str, maps: list[tuple[str, str]]) -> str:
    for pre, unc in maps:
        if path.startswith(pre):
            return (unc + path[len(pre):]).replace("/", "\\")
    return path


def unc_uri(win_path: str) -> str:
    """\\\\server\\share\\a -> file:////server/share/a (Java File.toURI form,
    which QuPath's Paths.get round-trips to the UNC path on Windows)."""
    assert win_path.startswith("\\\\"), win_path
    assert " " not in win_path, f"URI would need escaping: {win_path}"
    return "file:" + "//" + win_path.replace("\\", "/")


def write_qupath_project(out: Path, order, slides_dir: Path, wmaps) -> None:
    """One QuPath project over the pyramidal query-slide copies, so the
    pathologist opens all 22 full-resolution slides from a single file."""
    import time
    import uuid

    import openslide

    log = {}
    with (slides_dir / "conversion_log.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            log[rec["case"]] = rec
    images = []
    for r in order.itertuples():
        pid = str(r.case_id)
        svs = slides_dir / f"{pid}.svs"
        s = openslide.OpenSlide(str(svs))
        levels = [{"downsample": round(d, 6), "width": w, "height": h}
                  for d, (w, h) in zip(s.level_downsamples, s.level_dimensions)]
        width, height = s.dimensions
        s.close()
        mpp = log.get(pid, {}).get("mpp")
        if mpp:
            cal = {"pixelWidth": {"value": mpp[0], "unit": "µm"},
                   "pixelHeight": {"value": mpp[1], "unit": "µm"}}
        else:
            cal = {"pixelWidth": {"value": 1.0, "unit": "px"},
                   "pixelHeight": {"value": 1.0, "unit": "px"}}
        win = to_windows(str(svs), wmaps)
        assert win.startswith("\\\\"), f"{pid}: no --windows-map prefix covers {svs}"
        images.append({
            "serverBuilder": {
                "builderType": "uri",
                "providerClassName":
                    "qupath.lib.images.servers.openslide.OpenslideServerBuilder",
                "uri": unc_uri(win),
                "args": [],
                "metadata": {
                    "name": f"{pid}.svs",
                    "width": width, "height": height, "sizeZ": 1, "sizeT": 1,
                    "channelType": "DEFAULT", "isRGB": True,
                    "pixelType": "UINT8", "levels": levels,
                    "channels": [{"name": "Red", "color": -65536},
                                 {"name": "Green", "color": -16711936},
                                 {"name": "Blue", "color": -16776961}],
                    "pixelCalibration": {
                        **cal,
                        "zSpacing": {"value": 1.0, "unit": "z-slice"},
                        "timeUnit": "SECONDS", "timepoints": []},
                    "preferredTileWidth": 512, "preferredTileHeight": 512,
                },
            },
            "entryID": int(r.case_order),
            "randomizedName": str(uuid.uuid4()),
            "imageName": f"case{r.case_order:02d}_{pid}",
            "metadata": {},
        })
    proj_dir = out / "qupath_project"
    proj_dir.mkdir(exist_ok=True)
    now = int(time.time() * 1000)
    proj = {"version": "0.5.1", "createTimestamp": now, "modifyTimestamp": now,
            "uri": unc_uri(to_windows(str(proj_dir / "project.qpproj"), wmaps)),
            "lastID": len(images) + 1, "images": images}
    (proj_dir / "project.qpproj").write_text(json.dumps(proj, indent=2))


def burn_grid(src: Path, dst: Path, cols: int, rows: int) -> None:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    font = load_font()
    for i in range(1, cols):
        x = i * w / cols
        d.line([(x, 0), (x, h)], fill=(255, 255, 255), width=3)
        d.line([(x, 0), (x, h)], fill=(60, 60, 60), width=1)
    for j in range(1, rows):
        y = j * h / rows
        d.line([(0, y), (w, y)], fill=(255, 255, 255), width=3)
        d.line([(0, y), (w, y)], fill=(60, 60, 60), width=1)
    for j in range(rows):
        for i in range(cols):
            d.text((i * w / cols + 6, j * h / rows + 4), f"{chr(65 + i)}{j + 1}",
                   font=font, fill=(40, 40, 40),
                   stroke_width=2, stroke_fill=(255, 255, 255))
    img.save(dst)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dossiers-dir", required=True,
                   help="st1_dossiers (response_template, grid_geometry, images)")
    p.add_argument("--summaries-dir", required=True, help="st1_llm_summaries")
    p.add_argument("--ranked-lists", required=True)
    p.add_argument("--rich-meta", required=True)
    p.add_argument("--labels", required=True, help="canonical label CSV (backfill)")
    p.add_argument("--ali-map", required=True,
                   help="Ali metadata CSV (Slide.Scan.File -> Comp.Bio.ID fallback)")
    p.add_argument("--query-btiff-dir", required=True)
    p.add_argument("--neighbour-svs-dir", required=True)
    p.add_argument("--instructions", required=True, help="reader-facing .md")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-grid", action="store_true",
                   help="copy query previews without the ROI reference grid")
    p.add_argument("--windows-map", action="append", default=None,
                   metavar="LINUX=UNC",
                   help="Linux prefix -> Windows UNC prefix for reader-facing "
                        f"slide paths (repeatable; default {DEFAULT_WINDOWS_MAP})")
    p.add_argument("--qupath-slides-dir", default=None,
                   help="pyramidal <case>.svs copies + conversion_log.jsonl "
                        "(st1_convert_query_slides.py); emits qupath_project/")
    a = p.parse_args(argv)
    wmaps = [tuple(m.split("=", 1))
             for m in (a.windows_map or DEFAULT_WINDOWS_MAP)]

    dos = Path(a.dossiers_dir)
    out = Path(a.out_dir)
    (out / "images" / "neighbours").mkdir(parents=True, exist_ok=True)
    (out / "LLM summaries").mkdir(exist_ok=True)

    order = pd.read_csv(dos / "response_template.csv")[["case_order", "case_id"]]
    geo = json.loads((dos / "grid_geometry.json").read_text())["cases"]
    qrows = pd.read_csv(Path(a.summaries_dir) / "query_metadata_rows.csv",
                        dtype={"slide_id": str}).set_index("slide_id")

    rich = pd.read_csv(a.rich_meta, dtype=str)
    rich_of = (rich.drop_duplicates("slide_id", keep="first")
               .set_index("slide_id").to_dict("index"))
    sj_of = (rich.dropna(subset=["Comp.Bio.ID"])
             .drop_duplicates("slide_id", keep="first")
             .set_index("slide_id")["Comp.Bio.ID"].to_dict())
    ali = pd.read_csv(a.ali_map, dtype=str, low_memory=False)
    sj_of = {**(ali.dropna(subset=["Comp.Bio.ID", "Slide.Scan.File"])
                .drop_duplicates("Slide.Scan.File", keep="first")
                .set_index("Slide.Scan.File")["Comp.Bio.ID"].to_dict()),
             **sj_of}  # rich wins where both know the slide
    lab = pd.read_csv(a.labels).rename(
        columns={"Slide.Scan.File": "slide_id", "Tumor.group": "tumor_group"})
    label_of = dict(zip(lab.slide_id.astype(str), lab.tumor_group))
    ranked = pd.read_csv(a.ranked_lists)
    ranked["query"] = ranked["query"].astype(str)

    cases, results, summaries = [], [], []
    thumb_owner: dict[str, str] = {}
    for r in order.itertuples():
        pid = str(r.case_id)
        qr = qrows.loc[pid]
        g = geo[pid]
        qimg = f"images/case{r.case_order:02d}_{pid}_query.png"
        if a.no_grid:
            Image.open(dos / "preview" / f"{pid}.jpg").convert("RGB").save(out / qimg)
        else:
            burn_grid(dos / "preview" / f"{pid}.jpg", out / qimg,
                      g["cols"], g["rows"])
        cases.append({
            "case_order": r.case_order, "case_id": pid,
            "age_at_diagnosis": qr.get("Age.at.Collection"),
            "site": qr.get("Site"), "presentation": qr.get("Met"),
            "query_image": qimg,
            "full_slide_path": to_windows(
                str(Path(a.query_btiff_dir) / f"{pid}.tiff"), wmaps),
        })
        for n in ranked[(ranked["query"] == pid) & (ranked["rank"] <= 10)].itertuples():
            nid = str(n.neighbour)
            rm = rich_of.get(nid, {})
            sj = sj_of.get(nid)
            thumb = dos / "neighbor_thumbs" / f"{nid}.jpg"
            nimg = ""
            if thumb.exists():
                stem = sj or nid
                if stem in thumb_owner and thumb_owner[stem] != nid:
                    stem = f"{stem}_{nid}"
                thumb_owner[stem] = nid
                nimg = f"images/neighbours/{stem}.jpg"
                if not (out / nimg).exists():
                    shutil.copy2(thumb, out / nimg)
            results.append({
                "case_order": r.case_order, "case_id": pid, "rank": int(n.rank),
                "retrieved_diagnosis": n.neighbour_label or label_of.get(nid),
                "sample_id": sj,
                "pecan_link": PECAN.format(sj) if sj else None,
                "distance": round(float(n.distance), 4),
                "site": rm.get("Site"), "age": rm.get("Age.at.Collection"),
                "presentation": rm.get("Met"),
                "neighbour_image": nimg,
                "full_slide_path": str(Path(a.neighbour_svs_dir) / f"{nid}.svs"),
            })
        summary = (Path(a.summaries_dir) / "packet" / f"{pid}.md").read_text().strip()
        (out / "LLM summaries" / f"{pid}.txt").write_text(summary + "\n")
        summaries.append(summary)

    frames = {"01_cases.csv": pd.DataFrame(cases),
              "02_search_results.csv": pd.DataFrame(results)}
    for name, df in frames.items():
        df.to_csv(out / name, index=False, encoding="utf-8-sig")
    sheet = pd.read_csv(dos / "response_template.csv")
    if a.no_grid:
        # packet-standard sheet (2026-09-08): grid questions dropped (user,
        # 09-03) and the summary-accuracy question dropped (user, 09-08 —
        # faithfulness is the S5 audit's job; reader-facing harm is caught by
        # the confidence question, specifics by comments). 6 scored + comments.
        sheet = (sheet.drop(columns=["q5_summary_accuracy",
                                     "q8_roi_cells", "q9_roi_confidence"])
                 .rename(columns={
                     "q6_summary_usefulness_1to5": "q5_summary_usefulness_1to5",
                     "q7_summary_confidence_effect": "q6_summary_confidence_effect",
                     "q10_comments": "q7_comments"}))
    sheet.to_csv(out / "03_response_sheet.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(a.instructions, out / "04_instructions.md")
    if a.qupath_slides_dir:
        write_qupath_project(out, order, Path(a.qupath_slides_dir), wmaps)
        qp = (out / "qupath_project" / "project.qpproj").read_text()
        for tok in BLIND_TOKENS:
            assert tok not in qp, f"qupath project leaks {tok!r}"

    # blinding: the reader-facing tables must never carry the verdict or the
    # query's truth; retrieved_diagnosis in 02 is deployment-visible by design.
    for name in ("01_cases.csv", "03_response_sheet.csv"):
        text = (out / name).read_text(encoding="utf-8-sig")
        for tok in BLIND_TOKENS:
            assert tok not in text, f"{name} leaks {tok!r}"
    # Summaries come from the packet (blind_query) arm; the query-truth
    # zero-leak property is the S5 auditor's (audit_summary.json, leak 0.0) —
    # here we only guard against the verdict vocabulary ever entering them.
    for summary in summaries:
        for tok in ("FAILURE", "PARTIAL", "SUCCESS", "difficulty"):
            assert tok not in summary, f"AI summary leaks {tok!r}"
    assert "true_label" not in frames["02_search_results.csv"].columns
    n_sj = frames["02_search_results.csv"].sample_id.notna().sum()
    assert n_sj == len(results), f"pecan mapping incomplete: {n_sj}/{len(results)}"

    print(json.dumps({"cases": len(cases), "result_rows": len(results),
                      "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
