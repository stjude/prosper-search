"""SISH (Chen et al., Nat. Biomed. Eng. 2022) as a full-system baseline. C14 / R3.1.

R3.1 asked for direct comparisons with "existing WSI retrieval systems". C5
answered the encoder half (RetCCL, ResNet50 floor) and the aggregation half
(Yottixel/BoB); the plan originally CLOSED SISH ("no reference implementation
compatible with our index"). The user reopened it 2026-08-30, so this module
supplies the compatibility layer and otherwise runs the *reference
implementation verbatim*: the mahmoodlab/SISH clone under ``$SISH_ROOT`` is
imported (``models.vqvae``, ``database.HistoDatabase``, ``veb.VEB``,
``eval.Uncertainty_Cal/Clean/Filtered_BY_Prediction``, ``build_index
.min_max_binarized``), never copied, so their algorithmic core cannot drift
from the published one. That code is GPLv3; importing it here for a private,
never-distributed analysis is deliberate, and none of it may be pushed to the
public prosper-search repo.

Design decisions, fixed before any result ([decision record] in the revision workspace):

* **Same tile index, SISH's own patch scale.** Patch positions come from the
  published 15% mosaic (the index every other row of the comparison table
  uses), but each position is read at SISH's spec -- 2048 px @ 40x centred on
  the 512 px mosaic tile, resized to 1024 -- because its VQ-VAE and DenseNet
  checkpoints were built for that field of view, and feeding them 4x-finer
  crops would sandbag the baseline.
* **Mosaic capped at <=50 patches/slide** (evenly spaced over mosaic order,
  deterministic, no RNG), applied to BOTH database and queries: SISH's
  published operating point is a compact mosaic (tens of patches), its
  guided search cost scales with both index and query size, and its ranking
  pipeline (Clean/Filtered) was tuned at that scale.
* **Leave-one-patient-out == the pipeline's same-patient masking**, applied
  inside the search (``meta_clean`` filtered on ``UID.Subject``), replacing
  the reference code's TCGA-barcode parsing.
* **Scoring**: SISH's own ranked slide list (uncertainty-then-Hamming order,
  after Clean and Filtered_BY_Prediction) is converted to a distance matrix
  (distance = rank, unretrieved = the 1e6 sentinel) and fed to the standard
  audited evaluator -- same metrics, masking convention, bootstrap and tie
  handling as every other configuration. SISH's native mMV@5 verdict is also
  recorded for completeness.

One deliberate deviation from the reference ``eval.py``: a query whose bags
all come back empty, or empty after Clean, yields an empty ranked list (the
reference crashes on ``final_vote_candidates[0]``); occurrences are counted
and reported.

Subcommands: ``extract`` (GPU, per-slide latents + binarized DenseNet
features), ``build`` (veb tree + meta), ``search`` (per-query ranked lists),
``matrix`` (assemble the distance matrix + native-metric summary).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..evaluation import paths

MOSAIC_CAP = 50            # patches per slide, database AND queries ([decision record])
PATCH_READ = 2048          # px @ 40x level 0, centred on the 512 mosaic tile
PATCH_NET = 1024           # px fed to VQ-VAE / DenseNet (SISH 40x convention)
SENTINEL = 1e6             # same "excluded" value as the published matrices
MODEL_NAME = "sish"        # matrix filename tokens: model=sish, method=system
METHOD_NAME = "system"


def sish_root() -> Path:
    return Path(os.environ.get(
        "SISH_ROOT",
        str(paths.output_root() / "third_party" / "SISH")))


_SISH_NS: Optional[dict] = None


def _sish() -> dict:
    """Import the reference implementation once, from the clone."""
    global _SISH_NS
    if _SISH_NS is None:
        root = paths.require(sish_root(), "SISH clone (mahmoodlab/SISH)")
        sys.path.insert(0, str(root))
        import importlib
        _SISH_NS = {
            "vqvae": importlib.import_module("models.vqvae"),
            "database": importlib.import_module("database"),
            "veb": importlib.import_module("veb"),
            "eval": importlib.import_module("eval"),
            "build_index": importlib.import_module("build_index"),
            "root": root,
        }
    return _SISH_NS


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _out_dir(sub: str) -> Path:
    d = paths.output_root() / "sish" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slide_ids() -> List[str]:
    from ..embeddings import tile_features as T
    mosaic_path = (Path(os.environ.get(
        "COMET_SCANS_ROOT",
        "/PROSPER_WORKSPACE/comet-new-scans"))
        / "output" / "mosaic" / "mosaic512_final_simplified.pkl")
    entries = T.load_mosaic(paths.require(mosaic_path, "mosaic pkl"))
    return [T.slide_id_of(e) for e in entries], entries


def _cap_indices(n: int, cap: int = MOSAIC_CAP) -> np.ndarray:
    """<= cap evenly spaced positions over mosaic order. Deterministic."""
    if n <= cap:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, cap)).astype(int))


def _metadata() -> pd.DataFrame:
    meta = pd.read_csv(paths.slide_metadata())
    meta["sid"] = meta["Slide.Scan.File"].astype(str)
    return meta.set_index("sid")


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------

class _DualPatchSet(Dataset):
    """One slide's capped mosaic positions, read at SISH scale, returning BOTH
    network inputs (VQ-VAE in [-1,1]; DenseNet ImageNet-normalised)."""

    _IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, slide_path: Path, locs: List[Tuple[int, int]]):
        self.slide_path = str(slide_path)
        self.locs = locs
        self._slide = None
        self._dims = None

    def __len__(self):
        return len(self.locs)

    def _handle(self):
        if self._slide is None:
            import openslide
            self._slide = openslide.OpenSlide(self.slide_path)
            self._dims = self._slide.dimensions
        return self._slide

    def __getitem__(self, i: int):
        x, y = self.locs[i]
        h = self._handle()
        w_dim, h_dim = self._dims
        # centre the 2048 read on the 512 tile; clamp inside the slide
        rx = min(max(int(x) - (PATCH_READ - 512) // 2, 0), max(w_dim - PATCH_READ, 0))
        ry = min(max(int(y) - (PATCH_READ - 512) // 2, 0), max(h_dim - PATCH_READ, 0))
        img = h.read_region((rx, ry), 0, (PATCH_READ, PATCH_READ)).convert("RGB")
        img = img.resize((PATCH_NET, PATCH_NET))
        t = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        vq = 2.0 * t - 1.0
        dn = (t - self._IN_MEAN) / self._IN_STD
        return vq, dn, i


def _load_vqvae(device: str):
    ns = _sish()
    ck = ns["root"] / "checkpoints" / "model_9.pt"
    sd = torch.load(paths.require(ck, "SISH VQ-VAE checkpoint"),
                    map_location="cpu", weights_only=True)
    vq = ns["vqvae"].LargeVectorQuantizedVAE_Encode(256, 128)
    enc = OrderedDict()
    for k, v in sd["model"].items():
        if "encoder" in k or "codebook" in k:
            enc[k.replace("module.", "", 1)] = v
    vq.load_state_dict(enc, strict=True)      # checkpoint was DataParallel-saved
    return vq.to(device).eval()


def _load_densenet(device: str):
    from torchvision.models import densenet121, DenseNet121_Weights
    dn = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    # the reference construction: features trunk + 32x32 avg pool, no ReLU
    dn = torch.nn.Sequential(*list(dn.children())[:-1],
                             torch.nn.AvgPool2d(kernel_size=(32, 32)))
    return dn.to(device).eval()


def cmd_extract(a) -> int:
    ns = _sish()
    min_max_binarized = ns["build_index"].min_max_binarized
    ids, entries = _slide_ids()
    lo = max(1, a.lo)
    hi = a.hi or len(entries)
    sel = list(zip(ids, entries))[lo - 1:hi]
    out = _out_dir("latent")
    device = a.device if torch.cuda.is_available() else "cpu"
    if device != a.device:
        print(f"[sish-extract] WARNING cuda unavailable, using {device}", flush=True)
    vq = _load_vqvae(device)
    dn = _load_densenet(device)
    print(f"[sish-extract] slides {lo}..{hi} ({len(sel)}), device={device}", flush=True)

    done = skipped = failed = 0
    t0 = time.time()
    for sid, entry in sel:
        dest = out / f"{sid}.pkl"
        if dest.exists():
            try:
                with open(dest, "rb") as fh:
                    pickle.load(fh)
                skipped += 1
                continue
            except Exception:
                print(f"[sish-extract] {sid}: unreadable output, recomputing", flush=True)
        mosaic = entry[1]
        keep = _cap_indices(len(mosaic), a.cap)
        locs = [tuple(mosaic[i]["wsi_loc"]) for i in keep]
        slide_path = paths.slide_wsi_path(sid)
        if not slide_path.exists():
            print(f"[sish-extract] {sid}: MISSING SLIDE {slide_path}", flush=True)
            failed += 1
            continue
        ts = time.time()
        try:
            ds = _DualPatchSet(slide_path, locs)
            dl = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers,
                            shuffle=False, pin_memory=True)
            lat_chunks, dense_chunks, idx_chunks = [], [], []
            with torch.no_grad():
                for vq_in, dn_in, idx in dl:
                    vq_in = vq_in.to(device, non_blocking=True)
                    dn_in = dn_in.to(device, non_blocking=True)
                    lat_chunks.append(vq(vq_in).cpu().numpy())
                    dense_chunks.append(dn(dn_in).squeeze(-1).squeeze(-1).cpu().numpy())
                    idx_chunks.append(idx.numpy())
            order = np.argsort(np.concatenate(idx_chunks))
            latent = np.concatenate(lat_chunks)[order].astype(np.uint8)
            dense = np.concatenate(dense_chunks)[order]
            dense_bin = [min_max_binarized(f) for f in dense]
        except Exception as exc:                      # noqa: BLE001
            print(f"[sish-extract] {sid}: FAILED {type(exc).__name__}: {exc}", flush=True)
            failed += 1
            continue
        rec = {"latent": latent, "dense_binarized": dense_bin,
               "locs": locs, "mosaic_indices": keep.tolist(),
               "n_mosaic_total": len(mosaic)}
        tmp = dest.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(rec, fh, protocol=4)
        tmp.replace(dest)
        done += 1
        print(f"[sish-extract] {sid}: {latent.shape[0]} patches in "
              f"{time.time()-ts:.1f}s", flush=True)

    wall = time.time() - t0
    prov = {"task": "sish/latent", "slide_range": [lo, hi], "cap": a.cap,
            "patch_read": PATCH_READ, "patch_net": PATCH_NET,
            "n_written": done, "n_skipped": skipped, "n_failed": failed,
            "wall_seconds": round(wall, 1), "device": device,
            "git_commit": _git_commit(),
            "lsf_job": os.environ.get("LSB_JOBID"),
            "lsf_index": os.environ.get("LSB_JOBINDEX")}
    pdir = paths.output_root() / "logs" / "sish_extract"
    pdir.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("LSB_JOBINDEX") or f"{lo}_{hi}"
    (pdir / f"provenance_{tag}.json").write_text(json.dumps(prov, indent=2))
    print(f"[sish-extract] done={done} skipped={skipped} failed={failed} "
          f"wall={wall:.1f}s", flush=True)
    return 1 if failed and not done else 0


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _semantic_lut(codebook_semantic: dict) -> np.ndarray:
    lut = np.zeros(128, dtype=np.int64)
    for k, v in codebook_semantic.items():
        lut[int(k)] = int(v)
    return lut


def _indices_reference(latent: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Faithful vectorised copy of build_index.slide_to_index: level 0 = raw
    sum (unused, power 0), level k = sum after k avg-pools, index = level1 +
    level2*1e6 + level3*1e11.

    CRITICAL: the reference pools a **Long tensor** (``to_latent_semantic``
    keeps int64), so AvgPool2d does *truncating integer* averaging at every
    level. Computing this in float64 gives different (larger) indices -- the
    build-time fidelity assert below caught exactly that. Keep int64."""
    sem = torch.from_numpy(lut[latent])                          # int64 (n,64,64)
    pools = torch.nn.AvgPool2d(kernel_size=(2, 2))
    feat = sem
    level_sum = {0: torch.sum(feat, (1, 2)).numpy().astype(float)}
    for level in (1, 2, 3):
        feat = pools(feat)
        level_sum[level] = torch.sum(feat, (1, 2)).numpy().astype(float)
    return (level_sum[1] + level_sum[2] * 1e6 + level_sum[3] * 1e11)


def cmd_build(a) -> int:
    ns = _sish()
    cs = torch.load(ns["root"] / "checkpoints" / "codebook_semantic.pt",
                    map_location="cpu", weights_only=False)
    lut = _semantic_lut(cs)
    meta_df = _metadata()
    ids, _ = _slide_ids()
    lat_dir = _out_dir("latent")
    db_dir = _out_dir("db")

    database: Dict[int, list] = {}
    key_list: List[int] = []
    n_slides = 0
    missing = []
    fidelity_checked = 0
    for sid in ids:
        f = lat_dir / f"{sid}.pkl"
        if not f.exists():
            missing.append(sid)
            continue
        with open(f, "rb") as fh:
            rec = pickle.load(fh)
        idxs = _indices_reference(rec["latent"], lut)
        # fidelity guard: vectorised path == reference double loop
        if fidelity_checked < 3 and len(idxs):
            db_probe = ns["database"].HistoDatabase.__new__(ns["database"].HistoDatabase)
            db_probe.codebook_semantic = cs
            db_probe.pool_layers = [torch.nn.AvgPool2d(kernel_size=(2, 2))] * 3
            db_probe.is_patch = False
            ref = db_probe._slide_to_index(rec["latent"][0].astype(np.int64))
            assert int(idxs[0]) == int(ref), (sid, int(idxs[0]), int(ref))
            fidelity_checked += 1
        row = meta_df.loc[sid]
        for j, key in enumerate(idxs):
            key = int(key)
            entry = {"slide_name": sid,
                     "dense_binarized": rec["dense_binarized"][j],
                     "x": rec["locs"][j][0], "y": rec["locs"][j][1],
                     "slide_ext": ".svs",
                     "diagnosis": str(row["Tumor.group"]),
                     "site": "comet",
                     "patient": str(row["UID.Subject"])}
            database.setdefault(key, []).append(entry)
            key_list.append(key)
        n_slides += 1
    if missing:
        print(f"[sish-build] MISSING {len(missing)} latents, e.g. {missing[:5]}", flush=True)
        if not a.allow_missing:
            raise SystemExit("refusing to build a partial database; "
                             "pass --allow-missing to override")
    universe = max(key_list)
    print(f"[sish-build] {n_slides} slides, {len(key_list)} entries, "
          f"{len(database)} distinct keys, universe {universe:.3e}", flush=True)
    veb = ns["veb"].VEB(universe)
    for k in key_list:
        veb.insert(int(k))
    sys.setrecursionlimit(100000)
    with open(db_dir / "veb.pkl", "wb") as fh:
        pickle.dump(veb, fh, protocol=4)
    with open(db_dir / "meta.pkl", "wb") as fh:
        pickle.dump(database, fh, protocol=4)
    (db_dir / "build_provenance.json").write_text(json.dumps({
        "n_slides": n_slides, "n_entries": len(key_list),
        "n_distinct_keys": len(database), "universe": universe,
        "missing_slides": missing, "cap": MOSAIC_CAP,
        "fidelity_checked_vs_reference": fidelity_checked,
        "git_commit": _git_commit()}, indent=2))
    print("[sish-build] wrote veb.pkl + meta.pkl", flush=True)
    return 0


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def _weights(meta_df: pd.DataFrame) -> Dict[str, float]:
    """The reference eval.py weighting: inverse label counts over the query
    cohort, normalised so the weights sum to 10."""
    counts = meta_df["Tumor.group"].astype(str).value_counts().to_dict()
    sum_inv = sum(1.0 / v for v in counts.values())
    norm = 10.0 / sum_inv
    return {k: norm / v for k, v in counts.items()}


def _rank_one(per_patch_results: List[list], ev) -> Tuple[list, dict]:
    """Port of the ranked-list assembly in the reference eval.py __main__
    (Uncertainty_Cal -> Clean -> Filtered_BY_Prediction -> ordered dedup).
    Returns (ranked [(slide, hamming, diagnosis, uncertainty, bag)], info)."""
    info = {"n_bags": len(per_patch_results), "empty": False}
    if sum(len(t) for t in per_patch_results) == 0:
        info["empty"] = True
        return [], info
    bag_summary, label_count_summary, len_info = [], {}, []
    for idx, bag in enumerate(per_patch_results):
        ent, label_cnt, dist = ev.Uncertainty_Cal(bag)
        if ent is not None:
            label_count_summary[idx] = label_cnt
            bag_summary.append((idx, ent, dist, len(bag)))
            len_info.append(len(bag))
    if not bag_summary:
        info["empty"] = True
        return [], info
    bag_summary, hamming_thrsh = ev.Clean(len_info, bag_summary)
    if not bag_summary:                      # guard; reference would crash
        info["empty"] = True
        return [], info
    bag_removed = ev.Filtered_BY_Prediction(bag_summary, label_count_summary)
    ret_final, visited = [], {}
    for b in bag_summary:
        bag_index, uncertainty = b[0], b[1]
        for r in per_patch_results[bag_index]:
            if uncertainty == 0:
                if r["slide_name"] not in visited:
                    ret_final.append((r["slide_name"], r["hamming_dist"],
                                      r["diagnosis"], uncertainty, bag_index))
                    visited[r["slide_name"]] = 1
            else:
                if (r["hamming_dist"] <= hamming_thrsh) and \
                        (r["slide_name"] not in visited):
                    ret_final.append((r["slide_name"], r["hamming_dist"],
                                      r["diagnosis"], uncertainty, bag_index))
                    visited[r["slide_name"]] = 1
    ranked = [e for e in sorted(ret_final, key=lambda x: (x[3], x[1]))
              if e[-1] not in bag_removed]
    info["hamming_thrsh"] = float(hamming_thrsh)
    info["n_bags_kept"] = len(bag_summary)
    info["n_bags_removed_by_prediction"] = len(bag_removed)
    return ranked, info


def cmd_search(a) -> int:
    ns = _sish()
    ev = ns["eval"]
    db = ns["database"].HistoDatabase(
        database_index_path=str(_out_dir("db") / "veb.pkl"),
        index_meta_path=str(_out_dir("db") / "meta.pkl"),
        codebook_semantic=str(ns["root"] / "checkpoints" / "codebook_semantic.pt"))
    meta_df = _metadata()
    ev.weight = _weights(meta_df)            # module-global the reference expects
    ids, _ = _slide_ids()
    lo = max(1, a.lo)
    hi = a.hi or len(ids)
    sel = ids[lo - 1:hi]
    lat_dir = _out_dir("latent")
    res_dir = _out_dir("results")
    print(f"[sish-search] queries {lo}..{hi} ({len(sel)})", flush=True)
    done = skipped = failed = 0
    t0 = time.time()
    for sid in sel:
        dest = res_dir / f"{sid}.pkl"
        if dest.exists():
            try:
                with open(dest, "rb") as fh:
                    pickle.load(fh)
                skipped += 1
                continue
            except Exception:
                pass
        f = lat_dir / f"{sid}.pkl"
        if not f.exists():
            print(f"[sish-search] {sid}: NO LATENT", flush=True)
            failed += 1
            continue
        with open(f, "rb") as fh:
            rec = pickle.load(fh)
        patient = str(meta_df.loc[sid]["UID.Subject"])
        # leave-one-patient-out on UID.Subject (replaces the TCGA parsing)
        db.meta_clean = {k: [e for e in v if e["patient"] != patient]
                         for k, v in db.meta.items()}
        ts = time.time()
        per_patch = []
        for j in range(rec["latent"].shape[0]):
            res = db.query(rec["latent"][j].astype(np.int64),
                           rec["dense_binarized"][j])
            per_patch.append(res)
        ranked, info = _rank_one(per_patch, ev)
        info["seconds"] = round(time.time() - ts, 2)
        info["patient"] = patient
        out = {"slide_id": sid,
               "ranked": [(r[0], int(r[1]), r[2], float(r[3])) for r in ranked],
               "info": info}
        tmp = dest.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(out, fh, protocol=4)
        tmp.replace(dest)
        done += 1
        print(f"[sish-search] {sid}: {len(ranked)} ranked in "
              f"{info['seconds']}s (empty={info['empty']})", flush=True)
    print(f"[sish-search] done={done} skipped={skipped} failed={failed} "
          f"wall={time.time()-t0:.1f}s", flush=True)
    return 1 if failed and not done else 0


# --------------------------------------------------------------------------
# matrix
# --------------------------------------------------------------------------

def cmd_matrix(a) -> int:
    ids, _ = _slide_ids()
    meta_df = _metadata()
    res_dir = _out_dir("results")
    n = len(ids)
    pos = {s: i for i, s in enumerate(ids)}
    mat = np.full((n, n), SENTINEL, dtype=np.float64)
    missing, empty, short10 = [], [], []
    ranked_lens = []
    native_hits = Counter()
    native_total = Counter()
    import math
    for qi, sid in enumerate(ids):
        f = res_dir / f"{sid}.pkl"
        if not f.exists():
            missing.append(sid)
            continue
        with open(f, "rb") as fh:
            rec = pickle.load(fh)
        ranked = rec["ranked"]
        ranked_lens.append(len(ranked))
        if rec["info"].get("empty") or not ranked:
            empty.append(sid)
            continue
        if len(ranked) < 10:
            short10.append(sid)
        for rank, (rsid, _h, _d, _u) in enumerate(ranked, start=1):
            if rsid in pos:
                mat[qi, pos[rsid]] = float(rank)
        # SISH-native mMV@5 (reference eval.py rule)
        qlabel = str(meta_df.loc[sid]["Tumor.group"])
        top = [r[2] for r in ranked[:5]]
        hit_label, hit_n = Counter(top).most_common(1)[0]
        ok = (hit_label == qlabel and
              (len(top) == 5 or hit_n >= math.ceil(5 / 2)))
        native_total[qlabel] += 1
        native_hits[qlabel] += int(ok)
    # belt and braces: same-patient masking on the matrix, like the published
    # *_no_same_patient artifacts (leave-one-patient already excluded these
    # during search, so this must be a no-op on retrieved cells)
    pat = meta_df.loc[ids]["UID.Subject"].astype(str).values
    same = pat[:, None] == pat[None, :]
    n_overwritten = int((mat[same] < SENTINEL).sum())
    mat[same] = SENTINEL
    df = pd.DataFrame(mat, index=ids, columns=ids)
    out = paths.output_root() / "similarity_matrix"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / (f"combined_similarity_matrix_{MODEL_NAME}_{METHOD_NAME}"
                  f"_no_same_patient.pkl")
    with open(dest, "wb") as fh:
        pickle.dump(df, fh, protocol=4)
    native = {k: {"acc": native_hits[k] / native_total[k], "n": native_total[k]}
              for k in sorted(native_total)}
    macro = float(np.mean([v["acc"] for v in native.values()]))
    summary = {
        "n_queries": n, "n_missing_results": len(missing),
        "n_empty_ranked": len(empty), "n_ranked_lt10": len(short10),
        "ranked_len_mean": float(np.mean(ranked_lens)) if ranked_lens else None,
        "ranked_len_median": float(np.median(ranked_lens)) if ranked_lens else None,
        "same_patient_cells_overwritten_nonsentinel": n_overwritten,
        "sish_native_mMV5_per_class": native,
        "sish_native_mMV5_macro": macro,
        "matrix": str(dest), "git_commit": _git_commit(),
        "missing": missing[:20], "empty": empty[:50]}
    (paths.output_root() / "sish" / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("sish_native_mMV5_per_class", "missing", "empty")},
                     indent=2), flush=True)
    if missing:
        print(f"[sish-matrix] REFUSING completeness: {len(missing)} queries "
              f"missing", flush=True)
        return 1
    return 0


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--lo", type=int, default=1)
    e.add_argument("--hi", type=int, default=0)
    e.add_argument("--cap", type=int, default=MOSAIC_CAP)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--num-workers", type=int, default=6)
    e.add_argument("--device", default="cuda")
    e.set_defaults(fn=cmd_extract)

    b = sub.add_parser("build")
    b.add_argument("--allow-missing", action="store_true")
    b.set_defaults(fn=cmd_build)

    s = sub.add_parser("search")
    s.add_argument("--lo", type=int, default=1)
    s.add_argument("--hi", type=int, default=0)
    s.set_defaults(fn=cmd_search)

    m = sub.add_parser("matrix")
    m.set_defaults(fn=cmd_matrix)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
