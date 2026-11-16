#!/usr/bin/env python3
"""R2.2-ALL B2: compare probe outputs with the stored fp32 mosaic-bank rows; apply the
criteria pre-declared in [analysis plan] (R2.2-ALL Stage B amendment)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_b_lib as L  # noqa: E402

REF_GPU = "NVIDIA A100-SXM4-80GB"
CRIT = {
    "ref": {"min_cos": 0.99999, "max_abs": 0.01, "mean_barcode": 0.9995, "min_barcode": 0.99,
            "replay_identical": True},
    "cross": {"min_cos": 0.9999, "max_abs": 0.03, "mean_barcode": 0.999, "min_barcode": 0.99,
              "replay_identical": False},
}


def q(a):
    a = np.asarray(a, dtype=np.float64)
    return {"min": float(a.min()), "p001": float(np.quantile(a, 0.001)), "p01": float(np.quantile(a, 0.01)),
            "median": float(np.median(a)), "mean": float(a.mean()), "max": float(a.max())}


def compare_dir(seg_dir, probe_set_dir, seg):
    slides = [s for s in (Path(probe_set_dir) / f"{seg}_slides.txt").read_text().split() if s]
    rows = {k: [] for k in ("cos", "maxabs", "barcode", "identical", "full")}
    per_slide = {}
    g0 = []
    meta = None
    for s in slides:
        js = Path(seg_dir) / f"{s}.json"
        if not js.exists():
            g0.append(f"{s}: missing output")
            continue
        rec = json.loads(js.read_text())
        meta = meta or rec
        z = np.load(Path(seg_dir) / f"{s}.npz")
        f, xy, cov = z["feats"], z["xy"], z["coverage"]
        cxy, ccov = L.read_coords(Path(probe_set_dir) / seg / f"{s}.coords.csv.gz")
        if f.dtype != np.float32 or f.shape != (len(cxy), L.DIM) or not np.isfinite(f).all():
            g0.append(f"{s}: bad array {f.dtype} {f.shape}")
            continue
        if not np.array_equal(xy, cxy) or not np.array_equal(cov, ccov):
            g0.append(f"{s}: coords differ from probe manifest")
            continue
        if L.sha256_file(Path(seg_dir) / f"{s}.npz") != rec["npz_sha256"]:
            g0.append(f"{s}: npz sha mismatch")
        bf, bxy = L.load_mosaic_bank(s)
        ref = bf[L.match_rows(bxy, xy)]
        c = L.compare_rows(f, ref)
        for k in ("cos", "maxabs", "barcode", "identical"):
            rows[k].append(c[k])
        rows["full"].append(cov == 1.0)
        per_slide[s] = {"n": int(len(xy)), "min_cos": float(c["cos"].min()), "max_abs": float(c["maxabs"].max()),
                        "min_barcode": float(c["barcode"].min()), "identical_frac": float(c["identical"].mean()),
                        "array_identical": bool(c["identical"].all()), "gpu": rec.get("gpu_model")}
    cat = {k: np.concatenate(v) if v else np.array([]) for k, v in rows.items()}
    out = {"segment": seg, "n_slides": len(per_slide), "n_tiles": int(len(cat["cos"])), "g0_failures": g0,
           "per_slide": per_slide}
    if len(cat["cos"]):
        out.update(cos=q(cat["cos"]), maxabs=q(cat["maxabs"]), barcode=q(cat["barcode"]),
                   identical_frac=float(cat["identical"].mean()))
        for name, m in (("full", cat["full"]), ("partial", ~cat["full"])):
            if m.any():
                out[f"stratum_{name}"] = {"n": int(m.sum()), "min_cos": float(cat["cos"][m].min()),
                                          "max_abs": float(cat["maxabs"][m].max()),
                                          "mean_barcode": float(cat["barcode"][m].mean()),
                                          "min_barcode": float(cat["barcode"][m].min()),
                                          "identical_frac": float(cat["identical"][m].mean())}
    return out, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-out", required=True, help="dir with p1/ p2/ (p3/) outputs of ONE probe job")
    ap.add_argument("--probe-set", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    po = Path(args.probe_out)
    p1, meta = compare_dir(po / "p1", args.probe_set, "p1")
    p2, meta2 = compare_dir(po / "p2", args.probe_set, "p2")
    meta = meta or meta2 or {}
    gpu = meta.get("gpu_model")
    arch = "ref" if gpu == REF_GPU else "cross"
    cr = CRIT[arch]
    checks = {
        "G0_p1": not p1["g0_failures"] and p1["n_tiles"] > 0,
        "G0_p2": not p2["g0_failures"] and p2["n_tiles"] > 0,
        "min_cos": p1.get("cos", {}).get("min", -1) >= cr["min_cos"],
        "max_abs": p1.get("maxabs", {}).get("max", 9) <= cr["max_abs"],
        "mean_barcode": p1.get("barcode", {}).get("mean", 0) >= cr["mean_barcode"],
        "min_barcode": p1.get("barcode", {}).get("min", 0) >= cr["min_barcode"],
    }
    replay_ident = bool(p2["per_slide"]) and all(v["array_identical"] for v in p2["per_slide"].values())
    if cr["replay_identical"]:
        checks["replay_identical"] = replay_ident
    p3 = {}
    for js in sorted((po / "p3").glob("*.json")):
        r = json.loads(js.read_text())
        p3 = {"slide": r["slide"], "n": r["n_tiles"], **r["timing"], "model_load_s": r["model_load_s"],
              "num_workers": r["num_workers"]}
    env = meta.get("env", {})
    res = {"probe_out": str(po), "gpu_model": gpu, "driver": meta.get("driver"),
           "host": env.get("host"), "queue": env.get("lsf_queue"), "job": env.get("lsf_job"),
           "torch": env.get("torch"), "cudnn": env.get("cudnn"), "env_mismatch": meta.get("env_mismatch"),
           "arch_class": arch, "criteria": cr, "checks": checks, "replay_identical": replay_ident,
           "PASS": all(checks.values()), "p1": p1, "p2": p2, "p3_throughput": p3}
    L.dump_json(args.out, res)
    print(json.dumps({k: res[k] for k in ("gpu_model", "host", "queue", "arch_class", "checks", "PASS")}))


if __name__ == "__main__":
    main()
