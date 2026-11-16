"""ST2.3 phase 1, pixel stage: montages + heatmap overlays from attribution output.

Deliberately imports nothing from ``prosper`` -- the CCDI BTIFFs need zarr-chunked
reads (no pyramid), and the compute env (miniconda, POT) and the pixel env
(``$PROSPER_OUT/envs/sk161``: tifffile+zarr+imagecodecs+PIL) are different
interpreters. Run under sk161.

Per query, from ``attr_<pid>.csv`` (attribution_cli):

  montage A ("toward-wrong"): top --top-n tiles by delta_os (when OS is in the
      top-10; else by ascending delta_true -- the tiles most pulled AWAY from
      the true entity), 512-px crops from the BTIFF.
  montage B ("toward-true"):  top tiles by delta_true (when the true entity is
      in the top-10; else by ascending delta_os -- least OS-pulled).
  overlay: the ds-16 thumbnail (banded decode, cached to --thumbs-dir, shared
      with the ST1.3 dossiers) with each mosaic tile painted by the case's
      primary delta, diverging blue (toward-true) .. red (toward-wrong),
      symmetric around 0.

Tile captions carry the five-class letter (T/I/N/nT/A), the delta, and (x, y).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from PIL import Image, ImageDraw

DS = 16          # thumbnail downsample
BAND = 1792      # thumbnail decode band height (multiple of 224 tiles and DS)
CLASS_LETTER = {1: "T", 2: "I", 3: "N", 4: "nT", 5: "A"}


def thumbnail(btiff: Path, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.asarray(Image.open(cache))
    z = zarr.open(tifffile.imread(str(btiff), aszarr=True), mode="r")
    H, W = z.shape[:2]
    Ht, Wt = H // DS, W // DS
    thumb = np.zeros((Ht, Wt, 3), np.uint8)
    for y0 in range(0, Ht * DS, BAND):
        h = min(BAND, Ht * DS - y0)
        band = np.asarray(z[y0:y0 + h, :Wt * DS])
        thumb[y0 // DS:(y0 + h) // DS] = (
            band.reshape(h // DS, DS, Wt, DS, 3).mean((1, 3)).astype(np.uint8))
    cache.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(thumb).save(cache)
    return thumb


def _crop(z, x: int, y: int, px: int) -> np.ndarray:
    H, W = z.shape[:2]
    tile = np.full((px, px, 3), 255, np.uint8)
    y1, x1 = min(y + px, H), min(x + px, W)
    if y < H and x < W:
        tile[:y1 - y, :x1 - x] = np.asarray(z[y:y1, x:x1])
    return tile


def montage(z, rows: pd.DataFrame, value_col: str, title: str, out: Path,
            tile_px: int, cell: int = 256, grid: int = 4) -> None:
    cap, head = 18, 26
    W = grid * cell
    H = head + grid * (cell + cap)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), title, fill="black")
    for i, r in enumerate(rows.itertuples()):
        if i >= grid * grid:
            break
        gx, gy = (i % grid) * cell, head + (i // grid) * (cell + cap)
        crop = _crop(z, int(r.x), int(r.y), tile_px)
        canvas.paste(Image.fromarray(crop).resize((cell, cell), Image.LANCZOS), (gx, gy))
        v = getattr(r, value_col)
        d.text((gx + 4, gy + cell + 3),
               f"{CLASS_LETTER.get(int(r.pred), '?')}  {value_col}={v:+.3f}  ({int(r.x)},{int(r.y)})",
               fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def overlay(thumb: np.ndarray, df: pd.DataFrame, value_col: str, title: str,
            out: Path, tile_px: int) -> None:
    img = thumb.astype(np.float32)
    v = df[value_col].to_numpy()
    vmax = np.nanmax(np.abs(v)) or 1.0
    side = tile_px // DS
    for x, y, val in zip(df.x, df.y, v):
        if not np.isfinite(val):
            continue
        t = float(np.clip(val / vmax, -1, 1))
        color = (np.array([255.0, 40, 40]) if t > 0 else np.array([40.0, 80, 255]))
        alpha = 0.55 * abs(t) + 0.08
        xs, ys = int(x) // DS, int(y) // DS
        patch = img[ys:ys + side, xs:xs + side]
        img[ys:ys + side, xs:xs + side] = (1 - alpha) * patch + alpha * color
    pil = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, pil.width, 22], fill="white")
    d.text((6, 4), f"{title}   red = {value_col} > 0 (toward wrong), "
                   f"blue = toward true; |max| = {vmax:.3f}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out)


def one_query(pid: str, a) -> dict:
    t0 = time.time()
    df = pd.read_csv(Path(a.attr_dir) / f"attr_{pid}.csv")
    has_os = df.cost_os.notna().any()
    has_true = df.cost_true.notna().any()
    bt = Path(a.btiff_dir) / f"{pid}.tiff"
    thumb = thumbnail(bt, Path(a.thumbs_dir) / f"{pid}.png")
    z = zarr.open(tifffile.imread(str(bt), aszarr=True), mode="r")
    bundles = Path(a.out_dir)

    if has_os:
        rows_a, col_a, ttl_a = df.nlargest(a.top_n, "delta_os"), "delta_os", "strongest OS-pull tiles"
    else:
        rows_a, col_a, ttl_a = df.nsmallest(a.top_n, "delta_true"), "delta_true", "tiles pulled away from the true entity"
    montage(z, rows_a, col_a, f"{pid} A: {ttl_a}", bundles / f"montageA_{pid}.png", a.tile_px)

    if has_true:
        rows_b, col_b, ttl_b = df.nlargest(a.top_n, "delta_true"), "delta_true", "strongest true-entity-pull tiles"
    else:
        rows_b, col_b, ttl_b = df.nsmallest(a.top_n, "delta_os"), "delta_os", "least OS-pulled tiles"
    montage(z, rows_b, col_b, f"{pid} B: {ttl_b}", bundles / f"montageB_{pid}.png", a.tile_px)

    ov_col = "delta_os" if has_os else "delta_true"
    if has_os:
        ov = df.assign(delta_os=df.delta_os)
    else:
        ov = df.assign(delta_true=-df.delta_true)  # red = away from true
    overlay(thumb, ov, ov_col, pid, bundles / f"overlay_{pid}.png", a.tile_px)
    return {"query": pid, "primary": ov_col, "wall_s": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attr-dir", required=True)
    p.add_argument("--btiff-dir", required=True)
    p.add_argument("--thumbs-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--queries", required=True, help="CSV with slide_id")
    p.add_argument("--top-n", type=int, default=16)
    p.add_argument("--tile-px", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    a = p.parse_args(argv)
    pids = pd.read_csv(a.queries).slide_id.tolist()
    from joblib import Parallel, delayed
    recs = Parallel(n_jobs=a.num_workers)(delayed(one_query)(pid, a) for pid in pids)
    (Path(a.out_dir) / "_render_record.json").write_text(json.dumps(recs, indent=1))
    print(json.dumps({"n": len(recs),
                      "total_wall_s": sum(r["wall_s"] for r in recs)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
