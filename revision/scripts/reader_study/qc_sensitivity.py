"""QC-pass sensitivity of the reader-study primary trajectory (collaborator-draft decision 2, 2026-11-02).

Restricts `readings.csv` of an analysis directory to the cases whose QC verdict in the frozen export is
`pass` (role = qc, one row per case) and recomputes fav_correct / in_diff per step and the paired
differences with the plan's whole-case bootstrap (endpoints.py; seed 20261027, 5,000 draws). Descriptive;
per reader only; no pooled row. Writes to a sibling directory `<analysis>_qc<n_pass>/` and never under
the export or analysis directories.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import endpoints as ep  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PHASES = ("A", "B", "C")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--readers", required=True)
    args = ap.parse_args(argv)
    readers = [x.strip() for x in args.readers.split(",") if x.strip()]
    analysis, export = args.analysis.resolve(), args.export.resolve()

    run = json.loads((analysis / "analyze_run.json").read_text())
    if sha256_file(analysis / "readings.csv") != run["readings_sha256"]:
        raise SystemExit("readings.csv differs from the analysed file")
    r = pd.read_csv(analysis / "readings.csv")
    r = r[r.reader_id.isin(readers)].copy()
    for p in PHASES:
        for col in (f"fav_correct_{p}", f"in_diff_{p}"):
            r[col] = r[col].map(lambda v: str(v).lower() == "true")

    sub = pd.read_csv(export / "submissions_export.csv")
    qc = sub[sub.role == "qc"]
    if qc.case_id.duplicated().any():
        raise SystemExit("QC export has more than one row per case")
    verdict = dict(zip(qc.case_id, qc.a_verdict))
    cases_all = sorted(r.case_id.unique())
    missing = [c for c in cases_all if c not in verdict]
    if missing:
        raise SystemExit(f"cases without a QC verdict: {missing}")
    qc_pass = sorted(c for c in cases_all if verdict[c] == "pass")
    qc_fail = sorted(c for c in cases_all if verdict[c] != "pass")
    rq = r[r.case_id.isin(qc_pass)].copy()

    out = analysis.parent / f"{analysis.name}_qc{len(qc_pass)}"
    for forbidden in (export, analysis):
        if out == forbidden or forbidden in out.parents:
            raise SystemExit("refusing to write under a read-only input")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for rid in readers:
        x = rq[rq.reader_id == rid]
        tier = r[r.reader_id == rid].expertise_tier.iloc[0]
        n_all = int((r.reader_id == rid).sum())
        # As in analyze.py: the bootstrap resamples the reader's own (QC-pass) cases.
        cases = sorted(x.case_id.unique())
        weights = ep.case_weights(len(cases))
        for meas in ("fav_correct", "in_diff"):
            for p in PHASES:
                e = ep.proportion_endpoint(x.case_id, x[f"{meas}_{p}"], cases, weights)
                rows.append({"reader_id": rid, "expertise_tier": tier, "n_readings_all": n_all,
                             "measure": f"{meas}_{p}", **e})
            for a, b in (("A", "B"), ("B", "C"), ("A", "C")):
                e = ep.paired_difference(x.case_id, x[f"{meas}_{a}"], x[f"{meas}_{b}"], cases, weights)
                rows.append({"reader_id": rid, "expertise_tier": tier, "n_readings_all": n_all,
                             "measure": f"{meas} paired difference {b} − {a}", "k": None, **e})
    df = pd.DataFrame(rows)
    df.to_csv(out / "QC_sensitivity_trajectory.csv", index=False)
    md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, row in df.iterrows():
        md.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
                                    for v in row) + " |")
    (out / "QC_sensitivity_trajectory.md").write_text("\n".join(md) + "\n")

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    rec = [
        "# RUN_RECORD — QC-pass sensitivity (qc_sensitivity.py)", "",
        f"- run: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- analysis: `{analysis}` (readings.csv {run['readings_sha256']}; analyze commit {run['commit']})",
        f"- export: `{export}`; submissions_export.csv {sha256_file(export / 'submissions_export.csv')}",
        f"- QC verdicts: {len(qc_pass)} pass / {len(qc_fail)} fail; failed cases {qc_fail}",
        f"- readers: {readers}; readings on QC-pass cases: "
        + ", ".join(f"{rid} {int((rq.reader_id == rid).sum())}/{int((r.reader_id == rid).sum())}" for rid in readers),
        f"- bootstrap: whole-case, seed {ep.SEED}, {ep.N_DRAWS} draws over each reader's own QC-pass cases (as analyze.py)",
        f"- script commit (HEAD): {git('rev-parse', 'HEAD')}; scripts/reader_study clean: "
        + ("yes" if not git('status', '--porcelain', '--', 'scripts/reader_study') else "NO"),
        f"- output sha256: QC_sensitivity_trajectory.csv {sha256_file(out / 'QC_sensitivity_trajectory.csv')}",
    ]
    (out / "RUN_RECORD.generated.md").write_text("\n".join(rec) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
