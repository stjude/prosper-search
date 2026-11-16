"""R2.2-ALL Stage A per-slide worker: eligibility enumeration, containment checks,
reuse inventory and cross-bank row comparison.

Usage (one shard):
  python3 stage_a_slides.py --slides-file target_slides.txt --lo 0 --hi 50 --out OUT/stage_a

Writes, atomically per slide, under OUT/slides/:
  <sid>.manifest.csv.gz  every grid cell with coverage > 0 (plus every mosaic /
                         CSV / cluster_locs coordinate even if coverage 0)
  <sid>.json             dims, thumbnail route, counts, checks, timings
  <sid>.bankcmp.npz      per-tile comparison of tiles present in both UNI2 banks
Inputs are read only. A slide whose JSON already exists with the same code
SHA-256 and eligibility mode is skipped (binding-checked resume).

--eligibility generator-list (HOLD option (i) 2026-11-13) defines eligibility as
the frozen generator's cluster_locs union; the manifest then gains a trailing
`recon_eligible` column and the reconstruction-vs-record differences move from `checks`
to `recon_verification`. The default (recon-threshold) output is unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import pickle
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eligibility_lib as L  # noqa: E402

H = Path("/PROSPER_WORKSPACE")
DEFAULTS = dict(
    wsi_root=H / "data/comet_final_HPC",
    simplified=H / "comet-new-scans/output/mosaic/mosaic512_final_simplified.pkl",
    mosaic_dicts=H / "comet-new-scans/data/mosaic_final_final",
    csv_root=H / "comet-annotation/output/all_patches",
    mosaic_bank=H / "comet-feature-gen/comet-uni2/data/tmp_final",
)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


CODE_SHA = {n: sha256_file(HERE / n) for n in ("stage_a_slides.py", "eligibility_lib.py")}


def atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def wsi_xy(entries):
    return np.asarray([e["wsi_loc"] for e in entries], dtype=np.int64).reshape(-1, 2)


ELIGIBILITY_MODES = ("recon-threshold", "generator-list")


def eligibility_flags(gkeys, cov, ckeys, mode):
    """Return (eligible, recon_eligible, full) per grid cell.

    recon-threshold (default): eligible = reconstructed coverage >= 0.7, full = coverage == 1.0.
    generator-list (HOLD option (i) 2026-11-13): eligible = membership of the frozen
    generator's cluster_locs union; the reconstruction is verification only, and full is
    restricted to eligible cells (eligible & reconstructed coverage == 1.0).
    """
    recon = cov >= L.ELIGIBLE_THRESHOLD
    if mode == "recon-threshold":
        return recon, recon, cov == 1.0
    if mode == "generator-list":
        elig = np.isin(gkeys, ckeys)
        return elig, recon, elig & (cov == 1.0)
    raise ValueError(f"unknown eligibility mode {mode!r}")


def process(sid, mosaic_xy, a, outdir: Path):
    import openslide

    t0 = time.time()
    rec = {"slide": sid, "code_sha256": CODE_SHA, "host": socket.gethostname(),
           "lsf_jobid": os.environ.get("LSB_JOBID"), "lsf_index": os.environ.get("LSB_JOBINDEX"),
           "openslide_python": openslide.__version__, "libopenslide": openslide.__library_version__}
    wsi = Path(a.wsi_root) / f"{sid}.svs"
    rec["wsi_path"] = str(wsi)
    rec["wsi_realpath"] = os.path.realpath(wsi)
    st = os.stat(wsi)
    rec["wsi_size"] = st.st_size
    rec["wsi_mtime"] = st.st_mtime
    slide = openslide.open_slide(str(wsi))
    w, h = slide.dimensions
    rec.update(width=w, height=h, level_count=slide.level_count,
               level_dimensions=[list(d) for d in slide.level_dimensions],
               objective_power=slide.properties.get("openslide.objective-power"),
               mpp_x=slide.properties.get("openslide.mpp-x"))
    thumb, route = L.get_thumbnail(slide, wsi)
    rec["thumbnail_route"] = route
    rec["thumbnail_size"] = list(thumb.size)
    mask = L.tissue_mask_from_thumbnail(thumb)
    rec["mask_shape"] = list(mask.shape)
    rec["mask_hratio"] = (mask.shape[0] / h) * L.TILE
    rec["mask_wratio"] = (mask.shape[1] / w) * L.TILE
    gi, gj, gx, gy, cov = L.grid_coverage(mask, w, h)
    rec["grid_rows"] = int(gi.max()) + 1
    rec["grid_cols"] = int(gj.max()) + 1
    rec["n_cells"] = int(gi.size)
    t_mask = time.time()

    gkeys = L.coord_keys(np.c_[gx, gy])
    mode = getattr(a, "eligibility", "recon-threshold")
    gen = mode == "generator-list"
    # original per-slide mosaic dicts: stored coverage + cluster_locs (pre-clustering list)
    dpath = Path(a.mosaic_dicts) / f"{sid}.pkl"
    dicts = None
    ckeys = np.zeros(0, np.int64)
    if dpath.exists():
        with open(dpath, "rb") as f:
            dicts = pickle.load(f)
        cl = [c for e in dicts for c in e.get("cluster_locs", [])]
        ckeys = L.coord_keys(np.asarray(cl, np.int64).reshape(-1, 2))
    elif gen:
        raise FileNotFoundError(f"generator-list eligibility needs {dpath}")
    elig, recon_elig, full = eligibility_flags(gkeys, cov, ckeys, mode)
    ekeys = gkeys[elig]
    fkeys = gkeys[full]
    rkeys = gkeys[recon_elig]
    rec["n_eligible"] = int(elig.sum())
    rec["n_full"] = int(full.sum())
    rec["n_partial_eligible"] = int((elig & ~full).sum())
    chk = {}
    rv = {}  # generator-list mode: reconstruction-vs-frozen-record verification (informational)
    if gen:
        rec["eligibility_mode"] = mode
        rec["eligibility_definition"] = ("eligible = union of cluster_locs in the frozen generator dicts "
                                         "(HOLD option (i), Haoran 2026-11-13); coverage/recon_eligible are "
                                         "reconstruction (verification only); full = eligible & coverage == 1.0")
        rec["n_recon_eligible"] = int(recon_elig.sum())
        rec["n_eligible_recon_cov_below_thr"] = int((elig & ~recon_elig).sum())
        rec["n_recon_eligible_not_in_list"] = int((recon_elig & ~elig).sum())
        rec["n_recon_full_not_in_list"] = int(((cov == 1.0) & ~elig).sum())
        chk["a_cluster_locs_off_grid"] = int((L.containment(ckeys, gkeys) != 1).sum())

    # ---- (d) duplicates
    chk["d_dup_grid"] = L.n_duplicates(gkeys)
    mkeys = L.coord_keys(mosaic_xy)
    chk["d_dup_mosaic"] = L.n_duplicates(mkeys)

    # ---- (a) mosaic containment in reconstructed >=0.7 set
    occ = L.containment(mkeys, ekeys)
    rec["n_mosaic"] = int(mkeys.size)
    chk["a_mosaic_not_exactly_once"] = int((occ != 1).sum())
    if gen:
        rv["a_mosaic_not_in_recon"] = int((L.containment(mkeys, rkeys) != 1).sum())
    pos = {int(k): n for n, k in enumerate(gkeys)}
    mcov = np.array([cov[pos[int(k)]] if int(k) in pos else np.nan for k in mkeys])
    chk["a_mosaic_off_grid"] = int(np.isnan(mcov).sum())
    rec["mosaic_recon_cov_min"] = float(np.nanmin(mcov)) if mkeys.size else None
    rec["mosaic_recon_cov_n_full"] = int((mcov == 1.0).sum())

    rec["mosaic_dicts_present"] = dpath.exists()
    if dicts is not None:
        dxy = wsi_xy(dicts)
        chk["a_dicts_order_equals_simplified"] = bool(dxy.shape == mosaic_xy.shape and np.array_equal(dxy, mosaic_xy))
        stored = np.array([float(e["tissue_coverage"]) for e in dicts])
        (rv if gen else chk)["a_stored_cov_mismatch"] = int((stored != mcov).sum()) if stored.shape == mcov.shape else -1
        locs = np.asarray([e["loc"] for e in dicts], dtype=np.int64)
        chk["c_dicts_loc_mismatch"] = int((locs[:, 0] * L.TILE != dxy[:, 1]).sum() + (locs[:, 1] * L.TILE != dxy[:, 0]).sum())
        chk["d_dup_cluster_locs"] = L.n_duplicates(ckeys)
        only_c, only_e = L.set_diff_counts(ckeys, rkeys)
        (rv if gen else chk)["a_cluster_locs_not_in_recon"] = only_c
        (rv if gen else chk)["a_recon_not_in_cluster_locs"] = only_e
        rec["n_cluster_locs"] = int(ckeys.size)

    # ---- (b) coverage==1.0 set vs all_patches CSV
    cpath = Path(a.csv_root) / f"{sid}.csv"
    rec["csv_present"] = cpath.exists()
    csv_keys = np.zeros(0, np.int64)
    csv_xy = np.zeros((0, 2), np.int64)
    if cpath.exists():
        df = pd.read_csv(cpath)
        csv_xy = df[["wsi_loc_x", "wsi_loc_y"]].to_numpy(np.int64)
        csv_keys = L.coord_keys(csv_xy)
        rec["n_csv"] = int(len(df))
        chk["d_dup_csv"] = L.n_duplicates(csv_keys)
        only_r, only_csv = L.set_diff_counts(fkeys, csv_keys)
        chk["b_full_not_in_csv"] = only_r
        chk["b_csv_not_in_full"] = only_csv
        chk["b_csv_cov_not_1"] = int((df["tissue_coverage"].to_numpy() != 1.0).sum())
        chk["b_csv_slide_id_mismatch"] = int((df["slide_id"].astype(str) != sid).sum())
        idx = np.array([pos.get(int(k), -1) for k in csv_keys])
        ok = idx >= 0
        chk["c_csv_off_grid"] = int((~ok).sum())
        chk["c_csv_ij_mismatch"] = int((df["loc_x"].to_numpy()[ok] != gi[idx[ok]]).sum()
                                       + (df["loc_y"].to_numpy()[ok] != gj[idx[ok]]).sum())

    # ---- (c) dims/grid agreement of mosaic coords
    chk["c_mosaic_not_on_grid"] = int(((mosaic_xy % L.TILE) != 0).any(1).sum())
    chk["c_mosaic_outside_dims"] = int(((mosaic_xy[:, 0] >= w) | (mosaic_xy[:, 1] >= h)).sum())
    if csv_xy.size:
        chk["c_csv_outside_dims"] = int(((csv_xy[:, 0] >= w) | (csv_xy[:, 1] >= h)).sum())
    chk["c_grid_shape"] = bool(rec["grid_rows"] == -(-h // L.TILE) and rec["grid_cols"] == -(-w // L.TILE))

    # ---- A5: bank binding + inventory + cross-bank comparison
    bank = {}
    mb = Path(a.mosaic_bank) / f"{sid}.pkl"
    rec["mosaic_bank_present"] = mb.exists()
    with open(mb, "rb") as f:
        mfeat, mlist = pickle.load(f)
    mbxy = wsi_xy(mlist)
    bank["mosaic_bank_dtype"] = str(mfeat.dtype)
    bank["mosaic_bank_shape"] = list(mfeat.shape)
    bank["mosaic_bank_coords_equal_simplified"] = bool(mbxy.shape == mosaic_xy.shape and np.array_equal(mbxy, mosaic_xy))
    bank["mosaic_bank_rows_equal_coords"] = bool(mfeat.shape[0] == len(mlist))
    bank["mosaic_bank_nonfinite"] = int((~np.isfinite(mfeat)).sum())
    fb = Path(a.full_bank) / f"{sid}.pkl"
    rec["full_bank_present"] = fb.exists()
    fbkeys = np.zeros(0, np.int64)
    if fb.exists():
        with open(fb, "rb") as f:
            ffeat, flist = pickle.load(f)
        fbxy = wsi_xy(flist)
        fbkeys = L.coord_keys(fbxy)
        bank["full_bank_dtype"] = str(ffeat.dtype)
        bank["full_bank_shape"] = list(ffeat.shape)
        bank["full_bank_rows_equal_coords"] = bool(ffeat.shape[0] == len(flist))
        bank["full_bank_coords_equal_csv_order"] = bool(fbxy.shape == csv_xy.shape and np.array_equal(fbxy, csv_xy))
        bank["full_bank_dup"] = L.n_duplicates(fbkeys)
        bank["full_bank_nonfinite"] = int((~np.isfinite(ffeat)).sum())
        # tiles in both banks
        fpos = {int(k): n for n, k in enumerate(fbkeys)}
        both = [(n, fpos[int(k)]) for n, k in enumerate(mkeys) if int(k) in fpos]
        rec["n_both_banks"] = len(both)
        if both:
            mi = np.array([b[0] for b in both]); fi = np.array([b[1] for b in both])
            cs, mad, agr, ex = L.row_compare(mfeat[mi], ffeat[fi])
            buf = io.BytesIO()
            np.savez_compressed(buf, wsi_xy=mosaic_xy[mi], cosine=cs, max_abs=mad, barcode_agree=agr, exact=ex)
            atomic_write_bytes(outdir / f"{sid}.bankcmp.npz", buf.getvalue())
            bank["cmp_min_cosine"] = float(cs.min())
            bank["cmp_max_abs"] = float(mad.max())
            bank["cmp_min_barcode_agree"] = float(agr.min())
            bank["cmp_mean_barcode_agree"] = float(agr.mean())
            bank["cmp_n_exact"] = int(ex.sum())
    rec["bank"] = bank

    in_m = np.isin(gkeys, mkeys)
    in_fb = np.isin(gkeys, fbkeys)
    source = np.where(in_m, "mosaic_bank", np.where(in_fb, "full_bank", np.where(elig, "missing", "")))
    rec["n_src_mosaic_bank"] = int((elig & in_m).sum())
    rec["n_src_full_bank_only"] = int((elig & ~in_m & in_fb).sum())
    rec["n_src_missing"] = int((elig & ~in_m & ~in_fb).sum())
    rec["n_missing_full"] = int((elig & ~in_m & ~in_fb & full).sum())
    rec["n_missing_partial"] = int((elig & ~in_m & ~in_fb & ~full).sum())
    rec["n_full_bank_not_eligible"] = int(np.setdiff1d(fbkeys, ekeys).size)

    keep = (cov > 0) | in_m | np.isin(gkeys, csv_keys) | np.isin(gkeys, ckeys)
    man = pd.DataFrame({
        "wsi_loc_x": gx[keep], "wsi_loc_y": gy[keep], "grid_i": gi[keep], "grid_j": gj[keep],
        "coverage": cov[keep], "eligible": elig[keep].astype(np.int8), "full": full[keep].astype(np.int8),
        "in_mosaic": in_m[keep].astype(np.int8), "in_csv": np.isin(gkeys[keep], csv_keys).astype(np.int8),
        "in_cluster_locs": np.isin(gkeys[keep], ckeys).astype(np.int8), "in_full_bank": in_fb[keep].astype(np.int8),
        "source": source[keep]})
    if gen:
        man["recon_eligible"] = recon_elig[keep].astype(np.int8)
        rec["recon_verification"] = rv
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(man.to_csv(index=False, float_format="%.17g").encode())
    atomic_write_bytes(outdir / f"{sid}.manifest.csv.gz", raw.getvalue())
    rec["manifest_rows"] = int(len(man))

    rec["checks"] = chk
    fails = []
    for k, v in chk.items():
        if isinstance(v, bool):
            if not v:
                fails.append(k)
        elif v != 0:
            fails.append(k)
    bank_fail = [k for k in ("mosaic_bank_coords_equal_simplified", "mosaic_bank_rows_equal_coords",
                             "full_bank_rows_equal_coords", "full_bank_coords_equal_csv_order")
                 if k in bank and not bank[k]]
    bank_fail += [k for k in ("mosaic_bank_nonfinite", "full_bank_dup", "full_bank_nonfinite")
                  if bank.get(k, 0) != 0]
    rec["check_failures"] = fails
    rec["bank_binding_failures"] = bank_fail
    rec["t_mask_s"] = round(t_mask - t0, 3)
    rec["t_total_s"] = round(time.time() - t0, 3)
    atomic_write_bytes(outdir / f"{sid}.json", json.dumps(rec, indent=1, default=str).encode())
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slides-file", required=True)
    p.add_argument("--lo", type=int, required=True)
    p.add_argument("--hi", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--full-bank", default=os.path.join(os.environ["PROSPER_OUT"], "features/uni2_all_patches"))
    p.add_argument("--eligibility", choices=ELIGIBILITY_MODES, default="recon-threshold")
    p.add_argument("--slides-dirname", default="slides", help="per-slide output subdirectory under --out")
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), default=str(v))
    a = p.parse_args()
    outdir = Path(a.out) / a.slides_dirname
    outdir.mkdir(parents=True, exist_ok=True)
    sids = [s.strip() for s in open(a.slides_file) if s.strip()][a.lo:a.hi]
    with open(a.simplified, "rb") as f:
        simp = pickle.load(f)
    mos = {os.path.splitext(n)[0]: wsi_xy(t) for n, t in simp}
    del simp
    print(f"[shard] lo={a.lo} hi={a.hi} n={len(sids)} host={socket.gethostname()} code={CODE_SHA}", flush=True)
    nfail = 0
    for sid in sids:
        jp = outdir / f"{sid}.json"
        if jp.exists():
            try:
                old = json.loads(jp.read_text())
                if (old.get("code_sha256") == CODE_SHA
                        and old.get("eligibility_mode", "recon-threshold") == a.eligibility):
                    print(f"[skip] {sid} (done, same code and eligibility mode)", flush=True)
                    continue
            except Exception:
                pass
        try:
            r = process(sid, mos[sid], a, outdir)
            print(f"[ok] {sid} elig={r['n_eligible']} full={r['n_full']} mosaic={r["n_mosaic"]} "
                  f"missing={r['n_src_missing']} fails={r['check_failures']} bankfail={r['bank_binding_failures']} "
                  f"route={r['thumbnail_route']} t={r['t_total_s']}s", flush=True)
        except Exception:
            nfail += 1
            err = traceback.format_exc()
            print(f"[FAIL] {sid}\n{err}", flush=True)
            atomic_write_bytes(outdir / f"{sid}.FAILED.txt", err.encode())
    print(f"[shard-done] failed={nfail}", flush=True)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
