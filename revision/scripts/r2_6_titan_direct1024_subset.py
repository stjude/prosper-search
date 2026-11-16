#!/usr/bin/env python3
"""R2.6 TITAN arm 2, direct-1024 stage 2: build tumour-only CONCH v1.5 patch-feature
subsets for TITAN slide encoding, one per stage-1 selection variant.

  subset  for each variant V in VARIANTS, write under $PROSPER_OUT/r2_6_titan_direct1024_20261004/stage2/<V>/:
            job_dir/40x_1024px_0px_overlap/features_conch_v15/<slide>.h5   kept rows of the published CONCH
                h5 (original row order; coords/features datasets with their attrs copied), one per included slide
            job_dir/40x_1024px_0px_overlap/{_excluded_slides.csv,_tumour_patch_summary.csv,_subset_record.json}
                (same location/format as the August S2 subset, prosper tumor_tiles_cli.py::subset L116-171)
            wsi_dir/<slide>.svs   symlinks to the WSI, included slides only
            subset_files.csv      per written slide: rows, file bytes, sha256 of the written coords/features bytes
            command.txt, command_record.json   trident command (NOT run here)
          then stage2/stage2_record.json. Rule (design decision 10-04 12:30, the submission's rule): a slide is
          included iff it has >= 1 kept patch; slides with 1-7 kept patches are reported, not excluded (the
          August `--min-patches 8` is NOT applied). All inputs are validated before anything is written; each
          variant is built in stage2/.<V>.partial and renamed to stage2/<V> only when complete. Refuses if any
          stage2/<V>, stage2/.<V>.partial or stage2/stage2_record.json exists.
  verify  after the GPU run of variant V: refuse unless job_dir/40x_1024px_0px_overlap/slide_features_titan holds
          exactly one <slide>.h5 per included slide, no *.lock under job_dir, every embedding is 768 finite
          values with population std >= 0.1 (hence non-zero), output coords (and patch-size attrs) equal the
          subset coords, every trident _config*.json has skip_errors false (the slide-encoder config must exist
          and say so explicitly), and job_dir/gpu_info.txt is present and non-empty. Writes <V>/verify_record.json
          only if everything passes; violations are written to <V>/verify_failed_<timestamp>.json (exit 1).
          If job_dir/40x_1024px_0px_overlap/_titan_infeasible_slides.csv exists (written by prune-infeasible),
          it is first authenticated (_prune_record.json present and its csv/summary sha256 match the files; the
          listed set equals the summary's n_kept == 1 set; wsi_dir holds exactly the written-minus-infeasible
          symlinks and wsi_dir_infeasible exactly the infeasible ones); then the expected set is the written
          slides minus those listed, and no h5 or .lock may exist for them.
  prune-infeasible  the pinned TITAN implementation (dac6773d) fails on a 1x1 patch grid, here exactly
          n_kept == 1 (see INFEASIBLE_REASON). For variant V, every slide with
          n_kept == 1 in _tumour_patch_summary.csv has its wsi_dir/<slide>.svs symlink moved (never deleted) to
          <V>/wsi_dir_infeasible/ and any stale <slide>.*.lock under job_dir moved to
          job_dir/_stale_locks/<lock name>.<timestamp>; writes _titan_infeasible_slides.csv and _prune_record.json
          next to _subset_record.json. command.txt and job_dir are not changed. Refuses if any infeasible slide
          has an h5 in slide_features_titan (finished or partial), if a lock exists for any other slide (a run
          may be active), or if it has already been run. A variant with no one-patch slide: reports zero, writes
          and moves nothing (exit 0).

Run from the repository root with PROSPER_OUT exported. Does not import `prosper`. WSI resolution, the trident
command template, parse_gpu_info, check_trident_config, check_coords, lock_files and sha256 are copied verbatim
from scripts/r2_6_titan_arm1_regen.py, except that check_coords here also accepts trident's squeezed (2,) coords
for a one-patch slide (staged (1, 2)); the h5 subset write is copied from tumor_tiles_cli.py::subset (source
paths and sha256 recorded in every _subset_record.json).
"""
import argparse, datetime, hashlib, json, os, re, sys, time
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

OUT_NAME = "r2_6_titan_direct1024_20261004"
COORDS = "40x_1024px_0px_overlap"
VARIANTS = ("b100_argmax", "b5_argmax", "b100_strict")
N_SLIDES = 2921
FEAT_DIM = 768
PKG = "/PROSPER_WORKSPACE/prosper-search"
SUBSET_SRC = PKG + "/prosper/preprocessing/tumor_tiles_cli.py"
REGEN_SRC_NAME = "scripts/r2_6_titan_arm1_regen.py"
REGEN_OUT_NAME = "r2_6_titan_arm1_regen_20261004"
RULE = ("include a slide iff it has >= 1 kept patch in the variant's selection manifest (submission rule, author "
        "decision 2026-10-04 12:30); zero-kept slides excluded; slides with 1-7 kept patches included and reported")
# The pinned TITAN implementation (MahmoodLab/TITAN dac6773d) fails on a 1x1 patch grid (here exactly n_kept == 1):
# LSF 327638230 (b100_strict) crashed at slide 12663 (n_kept=1)
# in TITAN vision_transformer.py:542 get_alibi, `x = x[bg_mask.cpu().squeeze(0)]` -> IndexError: index 1 is out of
# bounds for axis 0 with size 1 (lsf/titan_b100_strict.327638230.err); slides with 2-7 patches encoded normally.
INFEASIBLE_N_KEPT = 1
INFEASIBLE_REASON = "titan_single_patch_alibi_indexerror"
INFEASIBLE_LSF_JOB = 327638230
INFEASIBLE_CSV = "_titan_infeasible_slides.csv"
PRUNE_RECORD = "_prune_record.json"

# ---- copied verbatim from scripts/r2_6_titan_arm1_regen.py (sha256 recorded; one marked deviation in check_coords) ----
TRIDENT = "/PROSPER_WORKSPACE/comet-feature-gen/comet-titan/trident"
TRIDENT_PY = "/HOME/miniconda3/envs/trident/bin/python"
WSI_SRC = "/PROSPER_WORKSPACE/data/comet_final_HPC"
SLIDE_CONFIG = "_config_slide_features_titan.json"
PATCH_ATTRS = ("patch_size", "patch_size_level0")
MIN_STD = 0.1                 # regenerated vector population std must be >= this (published min 0.234)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def trident_command(wsi_dir, job_dir):
    return (f"cd {TRIDENT} && HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 {TRIDENT_PY} run_batch_of_slides.py "
            f"--task feat --wsi_dir {wsi_dir} --job_dir {job_dir} --patch_encoder conch_v15 --slide_encoder titan "
            f"--mag 40 --patch_size 1024 --coords_dir {COORDS} --gpu 0")


GPU_L_RE = re.compile(r"^GPU\s+\d+:\s*(.+?)\s*\(UUID:")


def parse_gpu_info(path):
    """Parse the independently written gpu_info.txt; raise SystemExit if missing or malformed.

    Rule (blank lines ignored, each line stripped): line 1 = hostname (must not look like an nvidia-smi line);
    then >=1 `nvidia-smi -L` lines `GPU <i>: <model> (UUID: ...)` (`MIG ...` sub-device lines tolerated and
    recorded); then >=1 `--query-gpu=name,driver_version --format=csv,noheader` lines `<model>, <driver>` (split at
    the last comma). The query names must equal the -L models in order and all listed GPUs must be one model, so
    the device used by `--gpu 0` is unambiguous."""
    f = Path(path)
    if not f.is_file():
        raise SystemExit(f"REFUSING: {f} missing (lead must write hostname + nvidia-smi output)")
    lines = [ln.strip() for ln in f.read_text(errors="replace").splitlines() if ln.strip()]
    bad = lambda why: SystemExit(f"REFUSING: {f} malformed: {why}")
    if not lines or GPU_L_RE.match(lines[0]) or "," in lines[0]:
        raise bad("line 1 must be the hostname")
    l_models, mig, rows = [], [], []
    for ln in lines[1:]:
        m = GPU_L_RE.match(ln)
        if m or ln.startswith("MIG "):
            if rows:
                raise bad(f"nvidia-smi -L line after query lines: {ln!r}")
            if m:
                l_models.append(m.group(1))
            else:
                mig.append(ln)
        elif "," in ln:
            name, drv = (x.strip() for x in ln.rsplit(",", 1))
            if not name or not drv:
                raise bad(f"query line {ln!r}")
            rows.append([name, drv])
        else:
            raise bad(f"unrecognised line {ln!r}")
    if not l_models or not rows:
        raise bad("needs >=1 `nvidia-smi -L` line and >=1 name,driver_version line")
    if [r[0] for r in rows] != l_models:
        raise bad(f"-L models {l_models} != query names {[r[0] for r in rows]}")
    if len(set(l_models)) != 1:
        raise bad(f"heterogeneous GPU models {sorted(set(l_models))}")
    return {"file_sha256": sha256(f), "hostname": lines[0], "model": l_models[0], "models": l_models,
            "driver_versions": sorted({r[1] for r in rows}), "query_rows": rows, "mig_lines": mig,
            "raw": "\n".join(lines)[:4000]}


def lock_files(job_dir):
    return sorted(str(f) for f in Path(job_dir).rglob("*.lock"))


def check_trident_config(job_dir):
    """Raise SystemExit unless the slide-encoder _config exists and no _config* json used skip_errors."""
    f = Path(job_dir) / COORDS / SLIDE_CONFIG
    if not f.exists():
        raise SystemExit(f"REFUSING: trident config {f} missing")
    if json.load(open(f)).get("skip_errors") is not False:
        raise SystemExit(f"REFUSING: {f} does not record skip_errors=false")
    for c in sorted(Path(job_dir).rglob("_config*.json")):
        v = json.load(open(c)).get("skip_errors", False)
        if v is not False:
            raise SystemExit(f"REFUSING: {c} has skip_errors={v!r}")


def check_coords(regen_dir, staged_dir, ids):
    """Raise SystemExit unless each regenerated h5's coords (and patch-size attrs present in the staged copy)
    equal the staged CONCH copy."""
    bad = []
    for s in ids:
        with h5py.File(Path(regen_dir) / f"{s}.h5", "r") as r, h5py.File(Path(staged_dir) / f"{s}.h5", "r") as c:
            if "coords" not in r:
                bad.append(f"{s}: no coords")
                continue
            # Deviation from the verbatim copy: trident saves coords with .squeeze() (trident/wsi_objects/WSI.py:950),
            # so a one-patch slide's (1, 2) coords are stored as (2,); accept only that exact case.
            rc, cc = r["coords"][:], c["coords"][:]
            if rc.shape == (2,) and cc.shape == (1, 2):
                rc = rc.reshape(1, 2)
            if rc.shape != cc.shape or not np.array_equal(rc, cc):
                bad.append(f"{s}: coords differ")
            for k in PATCH_ATTRS:
                if k in c["coords"].attrs and (k not in r["coords"].attrs
                                               or r["coords"].attrs[k] != c["coords"].attrs[k]):
                    bad.append(f"{s}: coords attr {k} differs")
    if bad:
        raise SystemExit("REFUSING: regenerated outputs not tied to staged inputs: " + "; ".join(bad))
# ---- end of copy ----


def default_paths(repo, out_root):
    P = os.path.dirname(repo)
    base = Path(out_root) / OUT_NAME
    return {
        "conch_dir": P + f"/comet-feature-gen/comet-titan/trident/trident_processed/{COORDS}/features_conch_v15",
        "wsi_src": WSI_SRC,
        "selection_dir": str(base / "conch_heads"),
        "stage2": str(base / "stage2"),
        "regen_command_record": os.path.join(out_root, REGEN_OUT_NAME, "command_record.json"),
        "regen_script": os.path.join(repo, REGEN_SRC_NAME),
        "subset_src": SUBSET_SRC,
    }


def variant_dirs(stage2, v, partial=False):
    root = Path(stage2) / (f".{v}.partial" if partial else v)
    job = root / "job_dir"
    return {"root": root, "job": job, "coords": job / COORDS, "feat": job / COORDS / "features_conch_v15",
            "wsi": root / "wsi_dir", "slide_out": job / COORDS / "slide_features_titan"}


class Refuse(Exception):
    """Input/consistency refusal; a plain Exception so it propagates out of Pool workers (SystemExit would not)."""


def _keys(xy):
    xy = np.asarray(xy, dtype=np.int64).reshape(-1, 2)
    if len(xy) and (xy.min() < 0 or xy.max() >= 1 << 31):
        raise Refuse("REFUSING: coordinate outside [0, 2^31)")
    return (xy[:, 0] << 32) | xy[:, 1]


def match_keep(coords, sel_xy, slide=""):
    """Boolean mask over `coords` rows selected by `sel_xy`; each selected (x, y) must occur exactly once in coords
    and at most once in the selection."""
    kc, ks = _keys(coords), _keys(sel_xy)
    if len(np.unique(ks)) != len(ks):
        raise Refuse(f"REFUSING: {slide}: duplicate (x, y) in selection manifest")
    srt = np.sort(kc)
    cnt = np.searchsorted(srt, ks, side="right") - np.searchsorted(srt, ks, side="left")
    if (cnt != 1).any():
        i = int(np.flatnonzero(cnt != 1)[0])
        raise Refuse(f"REFUSING: {slide}: selected (x, y) {tuple(np.asarray(sel_xy)[i].tolist())} occurs "
                         f"{int(cnt[i])} times in the CONCH coords (must be exactly once)")
    keep = np.isin(kc, ks)
    if int(keep.sum()) != len(ks):
        raise Refuse(f"REFUSING: {slide}: kept {int(keep.sum())} != selected {len(ks)}")
    return keep


def load_selection(p, variants):
    """Read selection_summary.csv and the per-variant manifests; return per-variant {slide: (n_patches, n_kept)}
    and {slide: (k, 2) int64 xy}. Refuses on any inconsistency between manifest and summary."""
    sel = Path(p["selection_dir"])
    summ = pd.read_csv(sel / "selection_summary.csv", dtype={"slide_id": str})
    out, hashes = {}, {"selection_summary.csv": {"path": str(sel / "selection_summary.csv"),
                                                  "sha256": sha256(sel / "selection_summary.csv")}}
    for v in variants:
        s = summ[summ.variant == v]
        if s.slide_id.duplicated().any() or len(s) == 0:
            raise Refuse(f"REFUSING: selection_summary rows for {v} empty or duplicated")
        f = sel / f"selection_{v}.csv.gz"
        hashes[f.name] = {"path": str(f), "sha256": sha256(f)}
        m = pd.read_csv(f, dtype={"slide_id": str, "x": np.int64, "y": np.int64})
        if list(m.columns) != ["slide_id", "x", "y"]:
            raise Refuse(f"REFUSING: {f} columns {list(m.columns)}")
        xy = {k: g[["x", "y"]].to_numpy(np.int64) for k, g in m.groupby("slide_id", sort=False)}
        info = {r.slide_id: (int(r.n_patches), int(r.n_kept)) for r in s.itertuples()}
        extra = sorted(set(xy) - set(info))
        if extra:
            raise Refuse(f"REFUSING: {v}: manifest slides absent from summary: {extra[:10]}")
        bad = [k for k, (_, nk) in info.items() if len(xy.get(k, ())) != nk]
        if bad:
            raise Refuse(f"REFUSING: {v}: manifest count != summary n_kept for {bad[:10]}")
        out[v] = {"info": info, "xy": xy, "n_manifest_rows": int(len(m))}
    sets = {v: set(out[v]["info"]) for v in variants}
    if len({frozenset(x) for x in sets.values()}) != 1:
        raise Refuse("REFUSING: variants cover different slide sets in selection_summary")
    return out, hashes


def _validate_slide(args):
    sid, h5path, per_v = args
    with h5py.File(h5path, "r") as h:
        coords = h["coords"][:]
        fshape = h["features"].shape
    if coords.ndim != 2 or coords.shape[1] != 2 or fshape != (len(coords), FEAT_DIM):
        raise Refuse(f"REFUSING: {sid}: coords {coords.shape} / features {fshape}")
    res = {}
    for v, (n_patches, n_kept, xy) in per_v.items():
        if len(coords) != n_patches:
            raise Refuse(f"REFUSING: {sid}: CONCH rows {len(coords)} != summary n_patches {n_patches} ({v})")
        keep = match_keep(coords, xy, f"{v}/{sid}")
        if int(keep.sum()) != n_kept:
            raise Refuse(f"REFUSING: {sid}: kept {int(keep.sum())} != summary n_kept {n_kept} ({v})")
        res[v] = int(keep.sum())
    return sid, int(len(coords)), res


def _write_slide(args):
    """As tumor_tiles_cli.py::subset L135-161 (read whole datasets + attrs, write coords[keep]/features[keep] with
    attrs copied), selection by matched (x, y) instead of tile votes; one read serves all variants."""
    sid, h5path, per_v, dst = args
    with h5py.File(h5path, "r") as h:
        coords = h["coords"][:]
        feats = h["features"][:]
        c_attrs = dict(h["coords"].attrs)
        f_attrs = dict(h["features"].attrs)
    out = {}
    for v, (n_patches, n_kept, xy) in per_v.items():
        keep = match_keep(coords, xy, f"{v}/{sid}")
        if len(coords) != n_patches or int(keep.sum()) != n_kept or n_kept < 1:
            raise Refuse(f"REFUSING: {sid}: inconsistent at write time ({v})")
        c, f = coords[keep], feats[keep]
        path = Path(dst[v]) / f"{sid}.h5"
        with h5py.File(path, "w") as h:
            d = h.create_dataset("coords", data=c)
            d.attrs.update(c_attrs)
            d = h.create_dataset("features", data=f)
            d.attrs.update(f_attrs)
        with h5py.File(path, "r") as h:
            if h["coords"].shape != c.shape or h["features"].shape != f.shape:
                raise Refuse(f"REFUSING: {path}: readback shape mismatch")
        out[v] = {"slide": sid, "n_kept": int(len(c)), "file_bytes": os.path.getsize(path),
                  "coords_sha256": hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest(),
                  "features_sha256": hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()}
    return out


def command_from_regen(regen_record, wsi_dir, job_dir):
    """Substitute --wsi_dir/--job_dir in the arm-1 regen main_command; must equal trident_command(wsi_dir, job_dir)."""
    main = json.load(open(regen_record))["main_command"]
    cmd = re.sub(r"--wsi_dir \S+", f"--wsi_dir {wsi_dir}", main, count=1)
    cmd = re.sub(r"--job_dir \S+", f"--job_dir {job_dir}", cmd, count=1)
    if cmd != trident_command(wsi_dir, job_dir) or "--skip_errors" in cmd:
        raise SystemExit("REFUSING: regen main_command pattern differs from the copied trident_command template")
    return cmd, main


def run_subset(p, variants=VARIANTS, workers=4, n_expected=N_SLIDES):
    t0 = time.time()
    stage2 = Path(p["stage2"])
    for v in variants:
        for part in (False, True):
            d = variant_dirs(stage2, v, part)["root"]
            if d.exists() or d.is_symlink():
                sys.exit(f"REFUSING: {d} exists; subset never overwrites")
    if (stage2 / "stage2_record.json").exists():
        sys.exit(f"REFUSING: {stage2 / 'stage2_record.json'} exists")

    # ---- validate everything before writing ----
    sel, sel_hashes = load_selection(p, variants)
    conch = Path(p["conch_dir"])
    h5s = {f.stem: f for f in sorted(conch.glob("*.h5"))}
    slides = sorted(sel[variants[0]]["info"])
    if len(h5s) != n_expected or sorted(h5s) != slides:
        raise Refuse(f"REFUSING: CONCH h5 set ({len(h5s)}) != selection_summary slide set ({len(slides)}) "
                         f"or != expected {n_expected}")
    included = {v: [s for s in slides if sel[v]["info"][s][1] >= 1] for v in variants}
    wsi = {}
    for s in sorted(set().union(*included.values())):
        w = Path(p["wsi_src"]) / f"{s}.svs"
        if not w.exists():
            raise Refuse(f"REFUSING: WSI {w} missing")
        wsi[s] = {"wsi": str(w), "wsi_resolved": os.path.realpath(w)}
    empty = np.zeros((0, 2), np.int64)
    per_slide = {s: {v: (sel[v]["info"][s][0], sel[v]["info"][s][1], sel[v]["xy"].get(s, empty))
                     for v in variants} for s in slides}
    with get_context("spawn").Pool(workers) as pool:
        val = list(pool.imap_unordered(_validate_slide, [(s, str(h5s[s]), per_slide[s]) for s in slides],
                                       chunksize=8))
    if len(val) != len(slides):
        raise Refuse("REFUSING: validation incomplete")
    print(f"[validate] {len(val)} slides: every selected (x, y) found exactly once; counts match summary "
          f"({time.time() - t0:.0f}s)", flush=True)
    copied = {"subset_src": {"path": p["subset_src"], "sha256": sha256(p["subset_src"]), "lines": "L116-171"},
              "regen_script": {"path": p["regen_script"], "sha256": sha256(p["regen_script"])}}
    cmds = {}
    for v in variants:
        D = variant_dirs(stage2, v)
        cmds[v] = command_from_regen(p["regen_command_record"], str(D["wsi"]), str(D["job"]))

    # ---- write (partial dirs) ----
    stage2.mkdir(parents=True, exist_ok=True)
    PD = {v: variant_dirs(stage2, v, partial=True) for v in variants}
    for v in variants:
        PD[v]["feat"].mkdir(parents=True)
        PD[v]["wsi"].mkdir()
    tasks = []
    for s in slides:
        pv = {v: per_slide[s][v] for v in variants if per_slide[s][v][1] >= 1}
        if pv:
            tasks.append((s, str(h5s[s]), pv, {v: str(PD[v]["feat"]) for v in pv}))
    files = {v: [] for v in variants}
    with get_context("spawn").Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_write_slide, tasks, chunksize=4), 1):
            for v, r in res.items():
                files[v].append(r)
            if i % 250 == 0:
                print(f"[write] {i}/{len(tasks)} slides ({time.time() - t0:.0f}s)", flush=True)

    summary = {}
    for v in variants:
        Dp, D = PD[v], variant_dirs(stage2, v)
        fl = pd.DataFrame(sorted(files[v], key=lambda r: r["slide"]),
                          columns=["slide", "n_kept", "file_bytes", "coords_sha256", "features_sha256"])
        if list(fl.slide) != included[v] or sorted(f.stem for f in Dp["feat"].glob("*.h5")) != included[v]:
            raise Refuse(f"REFUSING: {v}: written h5 set != included slides")
        for s in included[v]:
            os.symlink(wsi[s]["wsi"], Dp["wsi"] / f"{s}.svs")
        info = sel[v]["info"]
        rows = [{"slide": s, "n_patches": info[s][0], "n_kept": info[s][1],
                 "frac_kept": info[s][1] / info[s][0] if info[s][0] else 0.0} for s in slides]
        summ = pd.DataFrame(rows)
        summ.to_csv(Dp["coords"] / "_tumour_patch_summary.csv", index=False)
        excl = [{"slide": s, "reason": f"0 kept patches in selection_{v}"} for s in slides if info[s][1] == 0]
        pd.DataFrame(excl, columns=["slide", "reason"]).to_csv(Dp["coords"] / "_excluded_slides.csv", index=False)
        fl.to_csv(Dp["root"] / "subset_files.csv", index=False)
        few = [s for s in included[v] if info[s][1] <= 7]
        cmd, regen_main = cmds[v]
        rec = {"action": "subset", "variant": v, "rule": RULE, "min_patches": 1,
               "selection_manifest": sel_hashes[f"selection_{v}.csv.gz"],
               "selection_summary": sel_hashes["selection_summary.csv"],
               "conch_dir": str(conch), "out": str(D["feat"]), "wsi_dir": str(D["wsi"]), "wsi_src": p["wsi_src"],
               "n_slides_in": len(slides), "n_written": len(included[v]),
               "n_excluded": len(excl), "excluded_ids": [e["slide"] for e in excl],
               "n_with_1_to_7": len(few), "ids_with_1_to_7": few,
               "n_kept_of_ids_with_1_to_7": {s: info[s][1] for s in few},
               "n_rows_written": int(fl.n_kept.sum()), "n_manifest_rows": sel[v]["n_manifest_rows"],
               "n_patches_total_included": int(sum(info[s][0] for s in included[v])),
               "median_frac_kept": float(summ.frac_kept.median()),
               "median_frac_kept_note": "over all n_slides_in (zero-kept slides included), as the August subset",
               "bytes_h5_written": int(fl.file_bytes.sum()), "subset_files_csv_sha256": sha256(Dp["root"] / "subset_files.csv"),
               "wsi_symlinks": {s: wsi[s] for s in included[v]},
               "script": {"path": os.path.abspath(__file__), "sha256": sha256(__file__)},
               "copied_code_sources": copied,
               "timestamp": datetime.datetime.now().astimezone().isoformat(),
               "command": " ".join([sys.executable] + sys.argv)}
        if rec["n_rows_written"] != rec["n_manifest_rows"]:
            raise Refuse(f"REFUSING: {v}: rows written {rec['n_rows_written']} != manifest rows")
        with open(Dp["coords"] / "_subset_record.json", "w") as fh:
            json.dump(rec, fh, indent=1)
        (Dp["root"] / "command.txt").write_text(cmd + "\n")
        crec = {"trident_command": cmd, "executed_by_this_script": False, "variant": v,
                "job_dir": str(D["job"]), "wsi_dir": str(D["wsi"]),
                "expected_output_dir": str(D["slide_out"]), "n_slides_expected": len(included[v]),
                "pattern_source": {"path": p["regen_command_record"], "sha256": sha256(p["regen_command_record"]),
                                   "main_command": regen_main,
                                   "derivation": "--wsi_dir/--job_dir substituted; equals copied trident_command()"},
                "notes": ["no --skip_errors: every included slide has >= 1 patch; a failure must stop the run",
                          "lead's LSF job must write gpu_info.txt (hostname, nvidia-smi -L, name,driver_version) "
                          "into job_dir; `verify` requires it"],
                "subset_record_sha256": sha256(Dp["coords"] / "_subset_record.json"),
                "timestamp": datetime.datetime.now().astimezone().isoformat()}
        with open(Dp["root"] / "command_record.json", "w") as fh:
            json.dump(crec, fh, indent=1)
        os.rename(Dp["root"], D["root"])
        summary[v] = {k: rec[k] for k in ("n_slides_in", "n_written", "n_excluded", "excluded_ids", "n_with_1_to_7",
                                          "ids_with_1_to_7", "n_rows_written", "median_frac_kept", "bytes_h5_written")}
        summary[v].update(trident_command=cmd, dir=str(D["root"]),
                          subset_record_sha256=sha256(D["coords"] / "_subset_record.json"),
                          command_record_sha256=sha256(D["root"] / "command_record.json"))
        print(f"[{v}] n_written={rec['n_written']} n_excluded={rec['n_excluded']} n_1_to_7={rec['n_with_1_to_7']} "
              f"rows={rec['n_rows_written']} bytes={rec['bytes_h5_written']}", flush=True)
    srec = {"action": "stage2_subset", "rule": RULE, "variants": summary,
            "inputs": {**sel_hashes, "conch_dir": str(conch), "wsi_src": p["wsi_src"]},
            "script": {"path": os.path.abspath(__file__), "sha256": sha256(__file__)},
            "workers": workers, "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "command": " ".join([sys.executable] + sys.argv), "runtime_s": round(time.time() - t0, 1)}
    with open(stage2 / "stage2_record.json", "w") as fh:
        json.dump(srec, fh, indent=1)
    return srec


def run_verify(p, v):
    D = variant_dirs(p["stage2"], v)
    if not (D["coords"] / "_subset_record.json").exists():
        sys.exit(f"REFUSING: {D['coords'] / '_subset_record.json'} missing")
    out = D["root"] / "verify_record.json"
    if out.exists():
        sys.exit(f"REFUSING: {out} exists; verify never overwrites")
    srec = json.load(open(D["coords"] / "_subset_record.json"))
    ids = sorted(f.stem for f in D["feat"].glob("*.h5"))
    viol, stats = [], {}
    if len(ids) != srec["n_written"]:
        viol.append(f"subset h5 count {len(ids)} != _subset_record n_written {srec['n_written']}")
    infeasible = read_infeasible(D)
    infeasible_checks = None
    if infeasible is not None:
        infeasible_checks = authenticate_infeasible(D, ids, infeasible)
        viol.extend(f"infeasible authentication: {k}: {c['detail']}"
                    for k, c in infeasible_checks.items() if not c["ok"])
        if not set(infeasible) <= set(ids):
            viol.append(f"infeasible slides not in the written subset: {sorted(set(infeasible) - set(ids))}")
        for s in infeasible:
            if (D["slide_out"] / f"{s}.h5").exists():
                viol.append(f"{s}: h5 output exists for an infeasible slide")
            if any(D["job"].rglob(f"{s}.*lock")):
                viol.append(f"{s}: lock exists for an infeasible slide")
        ids = sorted(set(ids) - set(infeasible))
    got = sorted(f.stem for f in D["slide_out"].glob("*.h5")) if D["slide_out"].is_dir() else []
    missing, extra = sorted(set(ids) - set(got)), sorted(set(got) - set(ids))
    if missing:
        viol.append(f"{len(missing)} missing slide embeddings: {missing[:20]}")
    if extra:
        viol.append(f"{len(extra)} extra slide embeddings: {extra[:20]}")
    locks = lock_files(D["job"])
    if locks:
        viol.append(f"lock files present: {locks[:20]}")
    stds, raw_shapes = [], set()
    for s in sorted(set(ids) & set(got)):
        with h5py.File(D["slide_out"] / f"{s}.h5", "r") as h:
            raw = np.asarray(h["features"][:])
        raw_shapes.add(tuple(raw.shape))
        x = raw.astype(np.float64).reshape(-1)
        if x.shape != (FEAT_DIM,):
            viol.append(f"{s}: embedding shape {raw.shape}")
        elif not np.isfinite(x).all():
            viol.append(f"{s}: non-finite")
        elif not x.any():
            viol.append(f"{s}: all-zero")
        elif x.std() < MIN_STD:
            viol.append(f"{s}: std {x.std():.4g} < {MIN_STD}")
        else:
            stds.append(float(x.std()))
    for name, fn in (("coords", lambda: check_coords(D["slide_out"], D["feat"], sorted(set(ids) & set(got)))),
                     ("trident_config", lambda: check_trident_config(D["job"]))):
        try:
            fn()
        except SystemExit as e:
            viol.append(f"{name}: {e}")
    g = D["job"] / "gpu_info.txt"
    if not g.is_file() or not g.read_text(errors="replace").strip():
        viol.append(f"{g} missing or empty")
        gpu = None
    else:
        try:
            gpu = parse_gpu_info(g)
        except SystemExit as e:
            gpu = {"parse_error_not_a_violation": str(e), "file_sha256": sha256(g)}
    rec = {"action": "verify", "variant": v, "n_expected": len(ids), "n_found": len(got),
           "missing": missing, "extra": extra, "lock_files": locks,
           "embedding_raw_shapes": sorted(list(x) for x in raw_shapes),
           "embedding_std_min_median_max": ([min(stds), float(np.median(stds)), max(stds)] if stds else None),
           "gpu_info": gpu, "subset_record_sha256": sha256(D["coords"] / "_subset_record.json"),
           "trident_configs": {str(c.relative_to(D["job"])): sha256(c) for c in sorted(D["job"].rglob("_config*.json"))},
           "violations": viol, "all_passed": not viol,
           "infeasible_slides": infeasible or [], "n_infeasible": len(infeasible or []),
           "infeasible_csv_sha256": (sha256(D["coords"] / INFEASIBLE_CSV) if infeasible is not None else None),
           "infeasible_checks": infeasible_checks,
           "script": {"path": os.path.abspath(__file__), "sha256": sha256(__file__)},
           "timestamp": datetime.datetime.now().astimezone().isoformat(),
           "command": " ".join([sys.executable] + sys.argv)}
    if viol:
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
        with open(D["root"] / f"verify_failed_{stamp}.json", "w") as fh:
            json.dump(rec, fh, indent=1)
        print("\n".join(["VERIFY FAILED:"] + viol[:50]), file=sys.stderr)
        return 1
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({"variant": v, "all_passed": True, "n": len(ids)}))
    return 0


def read_infeasible(D):
    """Slides listed in _titan_infeasible_slides.csv, or None when the file does not exist."""
    f = D["coords"] / INFEASIBLE_CSV
    if not f.exists():
        return None
    return sorted(pd.read_csv(f, dtype={"slide": str})["slide"].tolist())


def authenticate_infeasible(D, written, infeasible):
    """Checks that _titan_infeasible_slides.csv is the unaltered output of prune-infeasible and that the TITAN
    input set (wsi_dir) is exactly written - infeasible. Returns {check: {"ok": bool, "detail": ...}}."""
    chk = {}
    csv_f, summ_f, rec_f = D["coords"] / INFEASIBLE_CSV, D["coords"] / "_tumour_patch_summary.csv", D["coords"] / PRUNE_RECORD
    if not rec_f.is_file():
        chk["prune_record_present"] = {"ok": False, "detail": f"{rec_f} missing"}
    else:
        rec = json.load(open(rec_f))
        chk["prune_record_present"] = {"ok": True, "detail": sha256(rec_f)}
        for key, f in (("infeasible_csv_sha256", csv_f), ("tumour_patch_summary_sha256", summ_f)):
            got = sha256(f) if f.is_file() else None
            chk[f"prune_record_{key}_matches"] = {"ok": rec.get(key) == got and got is not None,
                                                 "detail": {"record": rec.get(key), "on_disk": got}}
    csv = pd.read_csv(csv_f, dtype={"slide": str})
    summ = pd.read_csv(summ_f, dtype={"slide": str})
    want = sorted(summ.loc[summ["n_kept"] == INFEASIBLE_N_KEPT, "slide"])
    extra, lack = sorted(set(infeasible) - set(want)), sorted(set(want) - set(infeasible))
    chk["csv_set_equals_summary_n_kept_1"] = {"ok": not extra and not lack and len(infeasible) == len(set(infeasible)),
                                              "detail": {"not_n_kept_1": extra, "n_kept_1_not_listed": lack}}
    badn = sorted(csv.loc[csv["n_kept"] != INFEASIBLE_N_KEPT, "slide"])
    chk["csv_n_kept_all_1"] = {"ok": not badn, "detail": badn}
    for key, d, exp in (("wsi_dir_equals_expected", D["wsi"], sorted(set(written) - set(infeasible))),
                        ("wsi_dir_infeasible_equals_infeasible", D["root"] / "wsi_dir_infeasible", sorted(infeasible))):
        have = sorted(f.name[:-4] for f in d.iterdir() if f.name.endswith(".svs")) if d.is_dir() else []
        other = sorted(f.name for f in d.iterdir() if not f.name.endswith(".svs")) if d.is_dir() else []
        miss, more = sorted(set(exp) - set(have)), sorted(set(have) - set(exp))
        chk[key] = {"ok": d.is_dir() and not miss and not more and not other,
                    "detail": {"n": len(have), "missing": miss[:20], "extra": more[:20], "non_svs": other[:20]}}
    return chk


def run_prune(p, v):
    """Move n_kept == 1 slides (1x1 patch grid; pinned TITAN dac6773d fails, see INFEASIBLE_REASON) out of the
    TITAN input set. Returns the record, or None if
    the variant has no such slide (nothing written or moved). Raises SystemExit on any refusal, before moving."""
    D = variant_dirs(p["stage2"], v)
    summ_f = D["coords"] / "_tumour_patch_summary.csv"
    if not summ_f.exists() or not (D["coords"] / "_subset_record.json").exists():
        sys.exit(f"REFUSING: {D['coords']} has no subset records")
    done = [x for x in (D["coords"] / INFEASIBLE_CSV, D["coords"] / PRUNE_RECORD, D["root"] / "wsi_dir_infeasible",
                        D["job"] / "_stale_locks") if os.path.lexists(x)]
    if done:
        sys.exit(f"REFUSING: prune-infeasible already run (or partially run) for {v}: {[str(x) for x in done]}")
    summ = pd.read_csv(summ_f, dtype={"slide": str})
    bad = summ[summ["n_kept"] == INFEASIBLE_N_KEPT].sort_values("slide")
    names = bad["slide"].tolist()
    if not names:
        print(json.dumps({"variant": v, "n_infeasible": 0, "action": "nothing to prune; nothing written"}))
        return None
    problems = []
    for s in names:
        if not (D["feat"] / f"{s}.h5").is_file():
            problems.append(f"{s}: no subset CONCH h5")
        if not (D["wsi"] / f"{s}.svs").is_symlink():
            problems.append(f"{s}: wsi_dir/{s}.svs symlink missing")
        if os.path.lexists(D["slide_out"] / f"{s}.h5"):
            locked = (D["slide_out"] / f"{s}.h5.lock").exists()
            problems.append(f"{s}: slide_features_titan/{s}.h5 exists ({'locked partial' if locked else 'finished'})"
                            " - not pruning; inspect it")
    locks = {s: sorted(D["job"].rglob(f"{s}.*lock")) for s in names}
    own = {str(x) for ls in locks.values() for x in ls}
    other = [x for x in lock_files(D["job"]) if x not in own]
    if other:
        problems.append(f"locks for other slides exist (a trident run may be active): {other[:20]}")
    if problems:
        sys.exit("REFUSING prune-infeasible:\n" + "\n".join(problems))
    summ_sha, cmd_sha = sha256(summ_f), sha256(D["root"] / "command.txt")
    now = datetime.datetime.now().astimezone()
    stamp = now.strftime("%Y%m%dT%H%M%S")
    moved_wsi, moved_locks = [], []
    (D["root"] / "wsi_dir_infeasible").mkdir()
    for s in names:
        src, dst = D["wsi"] / f"{s}.svs", D["root"] / "wsi_dir_infeasible" / f"{s}.svs"
        target = os.readlink(src)
        os.rename(src, dst)
        moved_wsi.append({"slide": s, "from": str(src), "to": str(dst), "symlink_target": target})
    if own:
        (D["job"] / "_stale_locks").mkdir()
    for s in names:
        for lk in locks[s]:
            dst = D["job"] / "_stale_locks" / f"{lk.name}.{stamp}"
            st = lk.stat()
            os.rename(lk, dst)
            moved_locks.append({"slide": s, "from": str(lk), "to": str(dst), "bytes": st.st_size,
                                "mtime": datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat()})
    out = bad[["slide", "n_kept"]].assign(reason=INFEASIBLE_REASON, lsf_job=INFEASIBLE_LSF_JOB)
    out.to_csv(D["coords"] / INFEASIBLE_CSV, index=False)
    if sha256(D["root"] / "command.txt") != cmd_sha:
        sys.exit("REFUSING: command.txt changed during prune (should never happen)")
    rec = {"action": "prune-infeasible", "variant": v,
           "rule": f"n_kept == {INFEASIBLE_N_KEPT} in _tumour_patch_summary.csv",
           "reason": INFEASIBLE_REASON, "lsf_job": INFEASIBLE_LSF_JOB,
           "evidence": "TITAN vision_transformer.py:542 get_alibi IndexError at slide 12663 (n_kept=1), "
                       f"lsf/titan_b100_strict.{INFEASIBLE_LSF_JOB}.err",
           "n_written": int(len(list(D["feat"].glob("*.h5")))), "n_infeasible": len(names),
           "n_titan_expected": int(len(list(D["feat"].glob("*.h5")))) - len(names),
           "infeasible_slides": names, "moved_wsi_symlinks": moved_wsi, "moved_locks": moved_locks,
           "n_moved_wsi_symlinks": len(moved_wsi), "n_moved_locks": len(moved_locks),
           "tumour_patch_summary_sha256": summ_sha,
           "infeasible_csv_sha256": sha256(D["coords"] / INFEASIBLE_CSV),
           "command_txt_sha256_unchanged": cmd_sha,
           "script": {"path": os.path.abspath(__file__), "sha256": sha256(__file__)},
           "timestamp": now.isoformat(), "command": " ".join([sys.executable] + sys.argv)}
    with open(D["coords"] / PRUNE_RECORD, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in ("variant", "n_infeasible", "infeasible_slides", "n_moved_wsi_symlinks",
                                          "n_moved_locks", "n_titan_expected")}))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--workers", type=int, default=4)
    vv = sub.add_parser("verify")
    vv.add_argument("--variant", required=True, choices=VARIANTS)
    pr = sub.add_parser("prune-infeasible")
    pr.add_argument("--variant", required=True, choices=VARIANTS)
    a = ap.parse_args(argv)
    out_root = os.environ.get("PROSPER_OUT")
    if not out_root:
        sys.exit("export PROSPER_OUT first")
    p = default_paths(os.getcwd(), out_root)
    if a.cmd == "subset":
        if not 1 <= a.workers <= 4:
            sys.exit("--workers must be 1..4 (shared login node)")
        try:
            rec = run_subset(p, workers=a.workers)
        except Refuse as e:
            sys.exit(f"{e} (any stage2/.<variant>.partial left behind is incomplete and must not be used)")
        print(json.dumps({v: {k: r[k] for k in ("n_written", "n_excluded", "n_with_1_to_7", "excluded_ids")}
                          for v, r in rec["variants"].items()}))
        return 0
    if a.cmd == "prune-infeasible":
        run_prune(p, a.variant)
        return 0
    return run_verify(p, a.variant)


if __name__ == "__main__":
    sys.exit(main())
