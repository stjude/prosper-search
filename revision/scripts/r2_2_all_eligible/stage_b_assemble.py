"""R2.2-ALL Stage B (B3 verification + B4 assembly): per-slide all-eligible UNI2 bank (D1/D4).

``shard`` (LSF array task): for each slide, verify the B3 gap output against the gap
manifest (D1), then assemble the eligible rows in Stage A grid order:
  code 0 = accepted mosaic bank ``tmp_final`` (fp32), code 1 = fully-tissue bank (fp16-autocast,
  as-is), code 2 = new B3 gap rows (fp32). Writes ``bank/<slide>.npz`` (feats float32 n x 1536,
  xy int64, coverage float64, source uint8) and ``bank/<slide>.json`` (counts, input/output
  SHA-256, verification). A fallback slide whose B3 output is not yet present gets
  ``bank/<slide>.WAITING.json`` (not a failure); it is assembled on a later run.
``merge`` (local, small): combine the per-slide jsons, check totals, write SHA256SUMS.
No input is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b_lib as L  # noqa: E402

REQUIRED_GPU = "NVIDIA A100-SXM4-80GB"
CODE_MOSAIC, CODE_FULL, CODE_GAP = 0, 1, 2


def wsi_xy(entries):
    return np.asarray([e["wsi_loc"] for e in entries], dtype=np.int64).reshape(-1, 2)


def verify_gap_output(slide, out_dir, gap_xy, gap_cov, frozen):
    """Return (feats, xy, record) for a B3 output; raise on any violation."""
    js = json.loads((out_dir / f"{slide}.json").read_text())
    npz_path = out_dir / f"{slide}.npz"
    sha = L.sha256_file(npz_path)
    if sha != js["npz_sha256"]:
        raise ValueError("npz sha != json npz_sha256")
    if js.get("gpu_model") != REQUIRED_GPU:
        raise ValueError(f"gpu_model {js.get('gpu_model')!r} not admitted")
    if js.get("config_sha256") != frozen["config_sha256"]:
        raise ValueError("config_sha256 differs from config_frozen.json")
    for k in ("stage_b_extract.py", "stage_b_lib.py"):
        if js.get("code_sha256", {}).get(k) != frozen["code_sha256"][k]:
            raise ValueError(f"code sha {k} differs from config_frozen.json")
    if js.get("env_mismatch"):
        raise ValueError(f"env_mismatch {js['env_mismatch']}")
    with np.load(npz_path) as z:
        feats, xy, cov = z["feats"], z["xy"], z["coverage"]
    if feats.dtype != np.float32 or feats.shape != (len(gap_xy), L.DIM):
        raise ValueError(f"feats {feats.dtype} {feats.shape} != float32 ({len(gap_xy)}, {L.DIM})")
    if not np.isfinite(feats).all():
        raise ValueError("non-finite features")
    idx = L.match_rows(xy, gap_xy)  # raises on duplicates / absent
    if len(xy) != len(gap_xy):
        raise ValueError("extra coordinates in B3 output")
    if not np.array_equal(cov[idx], gap_cov):
        raise ValueError("coverage mismatch")
    rec = {"npz": str(npz_path), "npz_sha256": sha, "gpu_model": js["gpu_model"],
           "host": js["env"].get("host"), "lsf_job": js["env"].get("lsf_job"),
           "lsf_queue": js["env"].get("lsf_queue"), "driver": js.get("driver"),
           "tiles_per_s": js["timing"].get("tiles_per_s"), "wall_s": js["timing"].get("wall_s"),
           "same_order_as_manifest": bool(np.array_equal(idx, np.arange(len(idx))))}
    return feats[idx], rec


def assemble(slide, a, frozen):
    t0 = time.perf_counter()
    run = Path(a.run)
    fb = slide in L.FALLBACK_SLIDES
    mdir = run / "stage_a" / ("slides_option_i" if fb else "slides")
    mpath = mdir / f"{slide}.manifest.csv.gz"
    df = L.read_manifest(mpath)
    g1 = json.loads((Path(a.gap_d1) / "slides" / f"{slide}.json").read_text())
    elig = df["eligible"].to_numpy() == 1
    mos = df["in_mosaic"].to_numpy() == 1
    fbk = df["in_full_bank"].to_numpy() == 1
    xy = df[["wsi_loc_x", "wsi_loc_y"]].to_numpy(np.int64)[elig]
    cov = df["coverage"].to_numpy(np.float64)[elig]
    src = np.full(len(xy), 255, np.uint8)
    src[mos[elig]] = CODE_MOSAIC
    src[(fbk & ~mos)[elig]] = CODE_FULL
    src[(~fbk & ~mos)[elig]] = CODE_GAP
    counts = {c: int((src == c).sum()) for c in (CODE_MOSAIC, CODE_FULL, CODE_GAP)}
    exp = {CODE_MOSAIC: g1["n_code0_mosaic"], CODE_FULL: g1["n_code1_full_bank"],
           CODE_GAP: g1["n_code2_gap"]}
    if counts != exp or len(xy) != g1["n_eligible"]:
        raise ValueError(f"code counts {counts} != gap_d1 {exp}")
    if len(np.unique(L.coord_key(xy))) != len(xy):
        raise ValueError("duplicate eligible coordinates")

    out_dir = Path(a.fallback_out if fb else a.b3_out)
    gap_xy, gap_cov = xy[src == CODE_GAP], cov[src == CODE_GAP]
    if not fb and counts[CODE_GAP]:
        cxy, ccov = L.read_coords(Path(a.gap_d1) / "coords" / f"{slide}.coords.csv.gz")
        if not (np.array_equal(cxy, gap_xy) and np.array_equal(ccov, gap_cov)):
            raise ValueError("gap_d1 coords file != manifest gap rows")
    feats = np.full((len(xy), L.DIM), np.nan, np.float32)
    inputs = {"manifest": {"path": str(mpath), "sha256": L.sha256_file(mpath)}}
    b3rec = None
    if counts[CODE_GAP]:
        if not (out_dir / f"{slide}.json").exists():
            if fb:
                return {"slide": slide, "status": "WAITING", "reason": f"no {out_dir}/{slide}.json"}
            raise FileNotFoundError(f"missing B3 output {out_dir}/{slide}.json")
        gfeat, b3rec = verify_gap_output(slide, out_dir, gap_xy, gap_cov, frozen)
        feats[src == CODE_GAP] = gfeat
        inputs["gap"] = {"path": b3rec["npz"], "sha256": b3rec["npz_sha256"]}

    mpkl = L.MOSAIC_BANK / f"{slide}.pkl"
    with open(mpkl, "rb") as fh:
        mfeat, mlist = pickle.load(fh)
    inputs["mosaic_bank"] = {"path": str(mpkl), "sha256": L.sha256_file(mpkl)}
    mfeat = np.asarray(mfeat)
    if mfeat.dtype != np.float32:
        raise ValueError(f"mosaic bank dtype {mfeat.dtype}")
    mxy = wsi_xy(mlist)
    if len(mxy) != counts[CODE_MOSAIC]:
        raise ValueError(f"mosaic bank rows {len(mxy)} != code-0 count {counts[CODE_MOSAIC]}")
    feats[src == CODE_MOSAIC] = mfeat[L.match_rows(mxy, xy[src == CODE_MOSAIC])]

    if counts[CODE_FULL]:
        fpkl = Path(a.full_bank) / f"{slide}.pkl"
        with open(fpkl, "rb") as fh:
            ffeat, flist = pickle.load(fh)
        inputs["full_bank"] = {"path": str(fpkl), "sha256": L.sha256_file(fpkl)}
        ffeat = np.asarray(ffeat)
        if ffeat.dtype != np.float32:
            raise ValueError(f"full bank dtype {ffeat.dtype}")
        feats[src == CODE_FULL] = ffeat[L.match_rows(wsi_xy(flist), xy[src == CODE_FULL])]

    if not np.isfinite(feats).all():
        raise ValueError("non-finite rows after assembly (unfilled or bad input)")
    bank = Path(a.bank)
    npz = bank / f"{slide}.npz"
    tmp = bank / f".{slide}.npz.tmp{os.getpid()}.npz"
    np.savez(tmp, feats=feats, xy=xy, coverage=cov, source=src)
    os.replace(tmp, npz)
    return {"slide": slide, "status": "OK", "fallback": fb, "n_rows": int(len(xy)),
            "n_code0_mosaic": counts[CODE_MOSAIC], "n_code1_full_bank": counts[CODE_FULL],
            "n_code2_gap": counts[CODE_GAP], "source_codes": {"0": "mosaic bank tmp_final fp32",
            "1": "fully-tissue bank fp16-autocast", "2": "B3 gap fp32"},
            "row_order": "Stage A manifest grid order (eligible rows)",
            "npz": str(npz), "npz_sha256": L.sha256_file(npz), "inputs": inputs, "b3": b3rec,
            "lsf_job": os.environ.get("LSB_JOBID"), "host": os.uname().nodename,
            "seconds": round(time.perf_counter() - t0, 2)}


def cmd_shard(a):
    frozen = json.loads(Path(a.config_frozen).read_text())
    bank = Path(a.bank)
    bank.mkdir(parents=True, exist_ok=True)
    slides = [s for s in Path(a.slides_file).read_text().split() if s]
    nfail = 0
    for s in slides[a.index::a.n]:
        if (bank / f"{s}.json").exists():
            continue  # resume: assembled and recorded
        try:
            rec = assemble(s, a, frozen)
        except Exception as exc:
            nfail += 1
            L.dump_json(bank / f"{s}.FAILED.{os.environ.get('LSB_JOBID', 'local')}.json",
                        {"slide": s, "error": repr(exc)})
            print(f"[FAIL] {s} {exc!r}", flush=True)
            continue
        name = f"{s}.json" if rec["status"] == "OK" else f"{s}.WAITING.json"
        L.dump_json(bank / name, rec)
        print(f"[{rec['status']}] {s}", flush=True)
    return 1 if nfail else 0


def cmd_merge(a):
    import pandas as pd
    bank = Path(a.bank)
    slides = [s for s in Path(a.slides_file).read_text().split() if s]
    recs, waiting, missing = [], [], []
    for s in slides:
        p = bank / f"{s}.json"
        if p.exists():
            recs.append(json.loads(p.read_text()))
        elif (bank / f"{s}.WAITING.json").exists():
            waiting.append(s)
        else:
            missing.append(s)
    g1 = pd.read_csv(Path(a.gap_d1) / "per_slide.csv", dtype={"slide": str}).set_index("slide")
    rows = []
    for r in recs:
        e = g1.loc[r["slide"]]
        ok = (r["n_rows"] == e.n_eligible and r["n_code0_mosaic"] == e.n_code0_mosaic
              and r["n_code1_full_bank"] == e.n_code1_full_bank and r["n_code2_gap"] == e.n_code2_gap)
        b3 = r.get("b3") or {}
        rows.append({"slide": r["slide"], "fallback": r["fallback"], "n_rows": r["n_rows"],
                     "n_code0": r["n_code0_mosaic"], "n_code1": r["n_code1_full_bank"],
                     "n_code2": r["n_code2_gap"], "matches_gap_d1": ok,
                     "b3_gpu": b3.get("gpu_model"), "b3_queue": b3.get("lsf_queue"),
                     "b3_host": b3.get("host"), "b3_job": b3.get("lsf_job"),
                     "npz_sha256": r["npz_sha256"]})
    df = pd.DataFrame(rows)
    df.to_csv(bank.parent / "bank_per_slide.csv", index=False)
    lines = [f"{r['npz_sha256']}  {Path(r['npz']).name}" for r in recs]
    (bank.parent / "bank_SHA256SUMS.txt").write_text("\n".join(sorted(lines, key=lambda x: x[66:])) + "\n")
    summ = {"n_slides_listed": len(slides), "n_assembled": len(recs), "waiting": waiting,
            "missing": missing, "all_match_gap_d1": bool(df.matches_gap_d1.all()) if len(df) else None,
            "rows": int(df.n_rows.sum()), "code0": int(df.n_code0.sum()),
            "code1": int(df.n_code1.sum()), "code2": int(df.n_code2.sum()),
            "b3_gpu_models": df.b3_gpu.dropna().value_counts().to_dict(),
            "b3_slides_by_queue": df.b3_queue.dropna().value_counts().to_dict()}
    L.dump_json(bank.parent / "bank_summary.json", summ)
    print(json.dumps(summ, indent=1))
    return 0 if not missing and summ["all_match_gap_d1"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard")
    for k in ("--run", "--gap-d1", "--b3-out", "--fallback-out", "--full-bank", "--bank",
              "--config-frozen", "--slides-file"):
        s.add_argument(k, required=True)
    s.add_argument("--index", type=int, required=True)
    s.add_argument("--n", type=int, required=True)
    m = sub.add_parser("merge")
    for k in ("--bank", "--gap-d1", "--slides-file"):
        m.add_argument(k, required=True)
    a = ap.parse_args(argv)
    return cmd_shard(a) if a.cmd == "shard" else cmd_merge(a)


if __name__ == "__main__":
    sys.exit(main())
