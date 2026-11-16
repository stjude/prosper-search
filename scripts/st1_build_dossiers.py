"""ST1.3 (spec 0002, [analysis plan]): per-case dossiers for the CCDI shortlist.

Two variants per case:

  blinded/   what the deployed system would show a reader: query age/site/met
             (never entity/diagnosis), the slide, the top-10 retrieval strip
             (neighbour thumbnails + entity + distance + metadata), and the
             packet LLM summary. No category chips, no attribution overlays --
             both encode whether retrieval matched the ground truth.
  unblinded/ the analyst/debrief page: everything above plus true entity and
             CCDI diagnosis, a co-author's difficulty, the SUCCESS/PARTIAL/FAILURE
             category per aggregation (recomputed here from the C3 per-query
             CSVs, not copied from the note -- the F-clause), five-class
             composition, the ST2.3 overlay + montages, and the QC summary.

Numbers are joined from the C3 per-query CSVs at render time; the rubric is
[decision record]'s (SUCCESS = hit_k5 & generous_correct_k5; PARTIAL = ~S & hit_k10;
FAILURE = ~hit_k10). Runs under the sk161 env (tifffile/PIL/pandas only).
"""
from __future__ import annotations

import argparse
import html
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

METHODS = ["ot_cosine", "yottixel", "median_pooling", "sampler"]
METHOD_LABEL = {"ot_cosine": "exact OT (primary)", "yottixel": "Yottixel/BoB",
                "median_pooling": "median pooling", "sampler": "SAMPLER"}
CLASS_NAMES = {1: "Tumor", 2: "Immune", 3: "Necrosis", 4: "nonTumor", 5: "Artifacts"}
CAT_COLOR = {"SUCCESS": "#1a7f37", "PARTIAL": "#b58900", "FAILURE": "#c0392b"}

CSS = """
body{font-family:system-ui,sans-serif;margin:24px auto;max-width:1200px;color:#222}
h1{font-size:22px} h2{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd}
table{border-collapse:collapse;font-size:13px} td,th{border:1px solid #ccc;padding:4px 8px}
.chip{display:inline-block;padding:2px 10px;border-radius:10px;color:#fff;font-weight:600}
.strip{display:flex;gap:8px;overflow-x:auto} .strip figure{margin:0;font-size:11px;width:150px}
.strip img{width:150px;border:1px solid #999}
img.wide{max-width:100%;border:1px solid #999}
.summary{background:#f6f8fa;border:1px solid #ddd;padding:12px;font-size:14px;white-space:pre-wrap}
.notes{border:1px dashed #999;min-height:90px;padding:8px;color:#666;font-size:13px}
.meta td:first-child{font-weight:600;background:#fafafa}
.bar{display:flex;height:18px;width:420px;border:1px solid #999}
.footer{margin-top:36px;font-size:11px;color:#777}
"""


def category(row) -> str:
    if bool(row.hit_k5) and bool(row.generous_correct_k5):
        return "SUCCESS"
    return "PARTIAL" if bool(row.hit_k10) else "FAILURE"


def neighbour_thumb(svs: Path, dst: Path, max_dim: int = 1400) -> bool:
    if dst.exists():
        return True
    try:
        tf = tifffile.TiffFile(str(svs))
        lv = min((l for l in tf.series[0].levels if max(l.shape[:2]) <= max_dim * 3),
                 key=lambda l: -max(l.shape[:2]), default=tf.series[0].levels[-1])
        img = lv.asarray()
        pil = Image.fromarray(img)
        pil.thumbnail((max_dim, max_dim), Image.LANCZOS)
        pil.save(dst, quality=85)
        return True
    except Exception as e:                                   # noqa: BLE001
        print(f"[dossier] thumb FAILED {svs.name}: {e}", flush=True)
        return False


def query_preview(thumb_png: Path, dst: Path, width: int = 1400) -> None:
    if dst.exists():
        return
    img = Image.open(thumb_png)
    h = int(img.height * width / img.width)
    img.resize((width, h), Image.LANCZOS).save(dst, quality=88)


GRID_COLS = 12  # ROI reference grid: columns A..L, square-ish cells


def grid_dims(w: int, h: int) -> tuple[int, int]:
    return GRID_COLS, max(1, round(h / (w / GRID_COLS)))


def grid_svg(w: int, h: int) -> str:
    cols, rows = grid_dims(w, h)
    cw, rh = w / cols, h / rows
    seg = "".join(f'<line x1="{i * cw:.1f}" y1="0" x2="{i * cw:.1f}" y2="{h}"/>'
                  for i in range(1, cols))
    seg += "".join(f'<line x1="0" y1="{j * rh:.1f}" x2="{w}" y2="{j * rh:.1f}"/>'
                   for j in range(1, rows))
    labels = "".join(
        f'<text x="{i * cw + 4:.1f}" y="{j * rh + 14:.1f}">{chr(65 + i)}{j + 1}</text>'
        for j in range(rows) for i in range(cols))
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
        'style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none">'
        f'<g stroke="#fff" stroke-width="2.4" opacity="0.5">{seg}</g>'
        f'<g stroke="#333" stroke-width="0.8" opacity="0.8">{seg}</g>'
        '<g style="font:600 12px system-ui,sans-serif;paint-order:stroke" '
        f'stroke="#fff" stroke-width="2.5" fill="#333">{labels}</g></svg>')


def comp_bar(counts: dict) -> str:
    total = sum(counts.values()) or 1
    palette = {1: "#5b8def", 2: "#9b59b6", 3: "#2c3e50", 4: "#95a5a6", 5: "#e67e22"}
    cells = "".join(
        f'<div style="width:{100 * v / total:.2f}%;background:{palette[c]}" '
        f'title="{CLASS_NAMES[c]}: {v}"></div>'
        for c, v in counts.items() if v)
    legend = " · ".join(f"{CLASS_NAMES[c]} {v}" for c, v in counts.items() if v)
    return f'<div class="bar">{cells}</div><div style="font-size:12px">{legend}</div>'


def esc(v) -> str:
    return html.escape("—" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))


def build_case(pid: str, ctx: dict, blinded: bool) -> str:
    q = ctx["meta"]
    parts = [f"<style>{CSS}</style>"]
    title = f"Case {pid}" + ("" if blinded else f" — {esc(q['tumor_group'])}")
    parts.append(f"<h1>{title}</h1>")

    rows = [("Case id", pid), ("Age", esc(q.get("age"))), ("Site", esc(q.get("site"))),
            ("Presentation", esc(q.get("met")))]
    if not blinded:
        rows += [("True entity (COMET group)", esc(q["tumor_group"])),
                 ("CCDI diagnosis", esc(q.get("diagnosis"))),
                 ("Ali difficulty", esc(q.get("difficulty")))]
    parts.append('<table class="meta">' + "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>")

    if not blinded:
        chips = []
        for m in METHODS:
            r = ctx["per_method"].get(m)
            if r is None:
                continue
            cat = category(r)
            chips.append(
                f'<span class="chip" style="background:{CAT_COLOR[cat]}" '
                f'title="pred@5 {r.pred_generous_k5}, share {r.vote_share_k5:.2f}, '
                f'entropy@10 {r.vote_entropy_k10:.2f}">{METHOD_LABEL[m]}: {cat}</span>')
        parts.append("<h2>Search outcome (frozen rubric, per aggregation)</h2>" +
                     " ".join(chips))
        r = ctx["per_method"]["ot_cosine"]
        parts.append(f'<p style="font-size:13px">OT generous@5 prediction: '
                     f'<b>{esc(r.pred_generous_k5)}</b> (vote share {r.vote_share_k5:.2f}, '
                     f'vote entropy@10 {r.vote_entropy_k10:.2f})</p>')
        parts.append("<h2>Mosaic composition (five-class head)</h2>" +
                     comp_bar(ctx["composition"]))

    parts.append("<h2>Query slide</h2>")
    pw, ph = ctx["preview_size"]
    parts.append(
        '<div style="position:relative;display:inline-block;max-width:100%">'
        f'<a href="../thumbs/{pid}.png"><img class="wide" '
        f'src="../preview/{pid}.jpg" alt="query slide"></a>'
        f"{grid_svg(pw, ph)}</div>"
        '<p style="font-size:12px;color:#555">Reference grid for the response '
        "form: cite cells (e.g. “C4”) or ranges (“C4–E6”) "
        "when asked for a region. Click the image for the full-resolution "
        "thumbnail (grid-free).</p>")

    parts.append("<h2>Top-10 retrieved neighbours (exact OT, primary)</h2>"
                 '<div class="strip">')
    for n in ctx["neighbours"]:
        cap = (f"<b>#{n['rank']} {esc(n['label'])}</b><br>d={n['distance']:.3f}<br>"
               f"{esc(n.get('site'))}<br>{esc(n.get('age'))} {esc(n.get('met'))}")
        img = (f'<img src="../neighbor_thumbs/{n["id"]}.jpg" alt="{n["id"]}">'
               if n["has_thumb"] else "<div style='width:150px;height:100px;"
               "background:#eee'>no image</div>")
        parts.append(f"<figure>{img}<figcaption>{cap}<br>"
                     f"<span style='color:#999'>{n['id']}</span></figcaption></figure>")
    parts.append("</div>")

    arm = "packet" if blinded else "qc"
    text = ctx["summaries"].get(arm)
    parts.append(f"<h2>System case summary ({'deployed view' if blinded else 'unblinded QC'})</h2>")
    parts.append(f'<div class="summary">{html.escape(text)}</div>' if text
                 else "<p>summary unavailable</p>")

    if not blinded and ctx["has_bundle"]:
        parts.append("<h2>OT transport-mass attribution (ST2.3)</h2>")
        parts.append(f'<img class="wide" src="../attr/overlay_{pid}.png">')
        parts.append(f'<p><img class="wide" src="../attr/montageA_{pid}.png"></p>')
        parts.append(f'<p><img class="wide" src="../attr/montageB_{pid}.png"></p>')

    if blinded:
        parts.append('<h2>Reader notes</h2><div class="notes">(assessment form '
                     "follows the H10-ratified protocol)</div>")

    parts.append(f'<div class="footer">PROSPER revision ST1.3 dossier · variant '
                 f'{"blinded" if blinded else "unblinded"} · generated from C3 '
                 f'per-query CSVs + ST2.3 bundles + ST1.7 summaries · '
                 f'{esc(ctx["provenance"])}</div>')
    return "\n".join(parts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--shortlist", required=True, help="docs/selected_cases.csv")
    p.add_argument("--per-query-dir", required=True, help="external_ccdi_all")
    p.add_argument("--ranked-lists", required=True)
    p.add_argument("--rich-meta", required=True)
    p.add_argument("--labels", required=True, help="canonical label CSV (backfill)")
    p.add_argument("--filter-summary", required=True)
    p.add_argument("--thumbs-dir", required=True)
    p.add_argument("--bundles-dir", required=True)
    p.add_argument("--summaries-dir", required=True)
    p.add_argument("--query-rows", required=True)
    p.add_argument("--svs-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--provenance", default="")
    a = p.parse_args(argv)

    out = Path(a.out_dir)
    for sub in ("blinded", "unblinded", "preview", "neighbor_thumbs", "attr"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(a.manifest)
    sc = pd.read_csv(a.shortlist).rename(columns={"patient_id": "slide_id"})
    qrows = pd.read_csv(a.query_rows).set_index("slide_id")
    per = {m: pd.read_csv(Path(a.per_query_dir) / f"per_query_ccdi_uni2_{m}.csv")
           .set_index("query") for m in METHODS}
    ranked = pd.read_csv(a.ranked_lists)
    rich = pd.read_csv(a.rich_meta)
    rich["slide_id"] = rich["slide_id"].astype(str)
    lab = pd.read_csv(a.labels).rename(
        columns={"Slide.Scan.File": "slide_id", "Tumor.group": "tumor_group"})
    lab["slide_id"] = lab["slide_id"].astype(str)
    label_of = dict(zip(lab.slide_id, lab.tumor_group))
    rich_of = (rich.drop_duplicates("slide_id", keep="first")
               .set_index("slide_id").to_dict("index"))
    fsum = pd.read_csv(a.filter_summary).set_index("slide")
    sdir = Path(a.summaries_dir)

    index_rows = []
    grid_geo = {}
    for row in man.itertuples():
        pid = row.slide_id
        qr = qrows.loc[pid]
        scr = sc[sc.slide_id == pid].iloc[0]
        f = fsum.loc[pid]
        comp = {c: int(f[f"n_{CLASS_NAMES[c].split()[0].lower()}"]) for c in CLASS_NAMES}

        nbrs = []
        for n in ranked[(ranked["query"] == pid) & (ranked["rank"] <= 10)].itertuples():
            nid = str(n.neighbour)
            rm = rich_of.get(nid, {})
            svs = Path(a.svs_dir) / f"{nid}.svs"
            has = neighbour_thumb(svs, out / "neighbor_thumbs" / f"{nid}.jpg") \
                if svs.exists() else False
            nbrs.append({"rank": int(n.rank), "id": nid, "label": n.neighbour_label,
                         "distance": float(n.distance), "site": rm.get("Site"),
                         "age": rm.get("Age.at.Collection"), "met": rm.get("Met"),
                         "has_thumb": has})

        query_preview(Path(a.thumbs_dir) / f"{pid}.png", out / "preview" / f"{pid}.jpg")
        with Image.open(out / "preview" / f"{pid}.jpg") as pim:
            preview_size = pim.size
        with Image.open(Path(a.thumbs_dir) / f"{pid}.png") as tim:
            thumb_size = tim.size
        cols, rows_n = grid_dims(*preview_size)
        grid_geo[pid] = {"preview_size": list(preview_size),
                         "thumb_size": list(thumb_size),
                         "cols": cols, "rows": rows_n}
        for stem in (f"overlay_{pid}", f"montageA_{pid}", f"montageB_{pid}"):
            src = Path(a.bundles_dir) / f"{stem}.png"
            if src.exists() and not (out / "attr" / f"{stem}.png").exists():
                shutil.copy(src, out / "attr" / f"{stem}.png")

        summaries = {}
        for arm in ("packet", "qc"):
            fpath = sdir / arm / f"{pid}.md"
            if fpath.exists():
                summaries[arm] = fpath.read_text()

        ctx = {"meta": {"tumor_group": row.tumor_group,
                        "diagnosis": getattr(row, "ccdi_diagnosis", None),
                        "age": qr["Age.at.Collection"], "site": qr["Site"],
                        "met": qr["Met"], "difficulty": scr.get("difficulty_score")},
               "per_method": {m: per[m].loc[pid] for m in METHODS if pid in per[m].index},
               "composition": comp, "neighbours": nbrs, "summaries": summaries,
               "preview_size": preview_size,
               "has_bundle": (out / "attr" / f"overlay_{pid}.png").exists(),
               "provenance": a.provenance}
        for variant, blind in (("blinded", True), ("unblinded", False)):
            (out / variant / f"{pid}.html").write_text(build_case(pid, ctx, blind))
        cat = category(ctx["per_method"]["ot_cosine"])
        index_rows.append((pid, row.tumor_group, cat))

    order = {"FAILURE": 0, "PARTIAL": 1, "SUCCESS": 2}
    index_rows.sort(key=lambda r: (order[r[2]], r[1]))
    li_u = "".join(
        f'<li><a href="unblinded/{p}.html">{p}</a> — {html.escape(g)} '
        f'<span class="chip" style="background:{CAT_COLOR[c]}">{c}</span></li>'
        for p, g, c in index_rows)
    (out / "index.html").write_text(
        f"<style>{CSS}</style><h1>ST1.3 dossiers — CCDI shortlist (22)</h1>"
        f'<p>Reader-facing packet: <a href="reader_index.html">reader_index.html</a> '
        f"(hand out that page only — this one labels the verdicts).</p>"
        f"<h2>Unblinded analyst pages</h2><ol>{li_u}</ol>")

    # Reader-facing index: category-free and order-shuffled (a category-sorted
    # list would let position stand in for the verdict). Seeded so the reading
    # order is reproducible across rebuilds.
    reader_pids = [p for p, *_ in index_rows]
    random.Random(20260901).shuffle(reader_pids)
    li_b = "".join(f'<li><a href="blinded/{p}.html">Case {i:02d} — {p}</a></li>'
                   for i, p in enumerate(reader_pids, 1))
    (out / "reader_index.html").write_text(
        f"<style>{CSS}</style><h1>PROSPER reading packet — 22 cases</h1>"
        f"<p>Please read the cases in the order listed.</p><ol>{li_b}</ol>")

    # ROI grid geometry: cell (col letter i=0.., row number j=1..) covers
    # level-0 x in [i, i+1] * thumb_w * DS / cols, y in [j-1, j] * thumb_h * DS
    # / rows, DS = the thumbnail downsample recorded by st2_render_bundles (16).
    (out / "grid_geometry.json").write_text(json.dumps(
        {"thumb_downsample_see_bundles_record": 16, "cases": grid_geo}, indent=1))

    # Per-case response sheet skeleton, prefilled in reader order; question
    # definitions live in docs/st1_5_response_form (workspace repo).
    cols = ["case_order", "case_id", "q1_initial_impression",
            "q2_top5_plausible_matches_0to5", "q3_retrieval_usefulness_1to5",
            "q4_next_step_effect", "q5_summary_accuracy",
            "q6_summary_usefulness_1to5", "q7_summary_confidence_effect",
            "q8_roi_cells", "q9_roi_confidence", "q10_comments", "minutes_spent"]
    tmpl = pd.DataFrame({"case_order": range(1, len(reader_pids) + 1),
                         "case_id": reader_pids}).reindex(columns=cols)
    tmpl.to_csv(out / "response_template.csv", index=False)
    print(json.dumps({"cases": len(index_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
