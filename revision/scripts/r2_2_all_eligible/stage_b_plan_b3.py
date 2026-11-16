"""R2.2-ALL Stage B (B3 planning): architecture-agnostic extraction tasks over the D1 gap.

Reads ``gap_d1/per_slide.csv`` (D1 re-aggregation) and packs the non-fallback slides with a
gap into whole-slide tasks by estimated GPU cost
``n_tiles / tiles_per_s + per_slide_overhead_s`` (measured on the B2 probes), so many
small-gap slides share one model load and large slides form their own task. Writes
``tasks.json`` (``{"tasks": [{"task", "slides", "tiles", "est_s"}]}``, the format read by
``stage_b_extract.py --tasks-json/--task-id``) and ``cmds.txt`` (one lsfq command per task).
Tasks carry no queue/architecture; the GPU model is recorded in every output json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b_lib as L  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-slide", required=True)
    ap.add_argument("--coords-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="B3 directory (tasks.json, cmds.txt)")
    ap.add_argument("--code-dir", required=True, help="frozen code dir holding stage_b_task.sh")
    ap.add_argument("--tiles-per-s", type=float, required=True)
    ap.add_argument("--overhead-s", type=float, required=True)
    ap.add_argument("--target-s", type=float, required=True)
    args = ap.parse_args(argv)
    import pandas as pd
    df = pd.read_csv(args.per_slide, dtype={"slide": str})
    df = df[(~df.fallback) & (df.n_code2_gap > 0)]
    tiles = dict(zip(df.slide, df.n_code2_gap.astype(int)))
    eq = args.overhead_s * args.tiles_per_s  # overhead expressed in tile-equivalents
    cost = {s: int(round(n + eq)) for s, n in tiles.items()}
    packed = L.pack_tasks(cost, int(round(args.target_s * args.tiles_per_s)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    tasks = []
    for i, ss in enumerate(packed):
        n = sum(tiles[s] for s in ss)
        tasks.append({"task": i, "slides": ss, "tiles": n,
                      "est_s": round(n / args.tiles_per_s + len(ss) * args.overhead_s, 1)})
    assert sorted(s for t in tasks for s in t["slides"]) == sorted(tiles)
    assert sum(t["tiles"] for t in tasks) == sum(tiles.values())
    L.dump_json(out / "tasks.json", {"params": vars(args), "n_slides": len(tiles),
                                     "n_tiles": sum(tiles.values()), "tasks": tasks})
    task_sh = Path(args.code_dir) / "stage_b_task.sh"
    lines = [f"bash {task_sh} --coords-dir {args.coords_dir} --out {out / 'out'} "
             f"--tasks-json {out / 'tasks.json'} --task-id {t['task']} "
             f"> {out / 'logs'}/task_{t['task']:04d}.$LSB_JOBID.log 2>&1" for t in tasks]
    (out / "cmds.txt").write_text("\n".join(lines) + "\n")
    est = [t["est_s"] for t in tasks]
    summ = {"n_tasks": len(tasks), "n_slides": len(tiles), "n_tiles": sum(tiles.values()),
            "est_gpu_h": round(sum(est) / 3600, 2), "est_task_s_max": max(est),
            "est_task_s_median": sorted(est)[len(est) // 2],
            "slides_per_task_max": max(len(t["slides"]) for t in tasks)}
    L.dump_json(out / "tasks_summary.json", summ)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    sys.exit(main())
