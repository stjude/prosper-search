"""Readback for analyze_r2.py (specification S5c §5): run the port on round 1's A0 and compare with round 1's A0 rows.

Step 1 transcribes round 1's per-response rulings for A0 into the external adjudication file the port reads
(sources: round-1 analyze.py outputs `analysis/flag_classification.csv` (alpha/beta/gamma classes, produced by
that script's rules), `analysis/adjudication.csv` (its hard-coded dispositions) and `analysis/work/
analysis_work.json` (`strict12`, `split23`, the rows its strict-"discrepant" and 23-split sensitivities recode)).
Step 2 runs analyze_r2.py on the round-1 run directory (read only). Step 3 compares the port's A0 numbers with
round-1 `analysis/endpoints.json` (the source of ENDPOINTS_REPORT.md) and `work/analysis_work.json`, and writes
READBACK_A0.md. Nothing in round 1 is written.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ["PROSPER_OUT"])
R1 = OUT_ROOT / "s5b_pilot_20261009"
DEST = OUT_ROOT / "s5c_pilot_round2_20261110" / "analysis"
RB = DEST / "readback_round1_A0"
ARM = "a0_masked_posterior"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def transcribe():
    fc = pd.read_csv(R1 / "analysis/flag_classification.csv", dtype=str, keep_default_na=False)
    fc = fc[fc.arm == "A0"]
    occ = Counter()
    classes = []
    for _, f in fc.iterrows():
        k = (f.run_uid, f.source, f.flag_type, f.value)
        occ[k] += 1
        classes.append({"run_uid": f.run_uid, "source": f.source, "flag_type": f.flag_type, "value": f.value,
                        "occurrence": occ[k], "class": f["class"], "payload_field": f.payload_field, "note": f.note})
    adj = pd.read_csv(R1 / "analysis/adjudication.csv", dtype=str, keep_default_na=False)
    adj = adj[adj.arm == "A0"]
    disp = [{k: r[k] for k in ("item", "run_uid", "original_automatic_code", "analyst_disposition", "reason",
                               "evidence")} for _, r in adj.iterrows()]
    w = json.load(open(R1 / "analysis/work/analysis_work.json"))
    strict = [[x["rating_id"], str(x["sentence_no"])] for x in w["strict12"] if x["arm_s"] == "A0"]
    split = [[x["rating_id"], str(x["sentence_no"])] for x in w["split23"] if x["arm_s"] == "A0"]
    rec = {"schema": "s5c_adjudication_v1",
           "provenance": {"purpose": "readback of analyze_r2.py on round 1 A0 (transcription of round-1 rulings)",
                          "sources": {str(R1 / p): sha(R1 / p) for p in ("analysis/flag_classification.csv",
                                                                          "analysis/adjudication.csv",
                                                                          "analysis/work/analysis_work.json",
                                                                          "analysis/scripts/analyze.py")}},
           "flag_classes": classes,
           "true_group_not_a_leak": [],   # round 1 removed only 10007 A2 (not A0) from the lexical list
           "dispositions": disp,
           "strict_discrepant_rows": strict,
           "recode_sensitivities": [{"name": "R1_with_23_split_rows_as_derivable", "rater": "R1", "rows": split,
                                     "to_code": "derivable"}]}
    RB.mkdir(parents=True, exist_ok=True)
    p = RB / "round1_A0_adjudication.json"
    p.write_text(json.dumps(rec, indent=1) + "\n")
    return p


def run_port(adj):
    cmd = [sys.executable, "-B", str(HERE / "analyze_r2.py"), "--run-dir", str(R1 / "run"), "--out-dir", str(RB / "port_output"),
           "--harness", str(HERE / "llm_grounding_cli.executed_fc88a2e7.py"), "--arm", ARM, "--adjudication", str(adj),
           "--ratings-r1", str(R1 / "rating_r1/rating_form_filled.csv"),
           "--ratings-r2", str(R1 / "rating_r2/rating_form_filled.csv"),
           "--rating-map", str(R1 / "rating_packet_key/rating_id_map.csv"),
           "--scoring-key", str(OUT_ROOT / "r3_5_s5b_posterior_20q_20261009_v2/scoring_key_DO_NOT_SEND.csv")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (RB / "port_stdout.txt").write_text(r.stdout + r.stderr)
    assert r.returncode == 0, r.stderr[-2000:]
    return cmd


def compare(cmd, adj):
    E1 = [e for e in json.load(open(R1 / "analysis/endpoints.json"))["endpoints"] if e["arm"] == "A0"]
    E2 = json.load(open(RB / "port_output/endpoints.json"))["endpoints"]
    w1 = json.load(open(R1 / "analysis/work/analysis_work.json"))
    w2 = json.load(open(RB / "port_output/analysis_work.json"))
    rows = []

    def row(item, v1, v2):
        rows.append((item, v1, v2, v1 == v2))

    k2 = {(e["name"], e["rater"]): e for e in E2}
    for e in E1:
        f = k2.get((e["name"], e["rater"]))
        row(f"endpoints.json: {e['name']} [{e['rater']}]", (e["numerator"], e["denominator"], e["value"]),
            None if f is None else (f["numerator"], f["denominator"], f["value"]))
    for rk in ("R1", "R2"):
        row(f"table 2.1 code distribution {rk} A0 (delivered)", w1["dist"][rk]["A0"],
            {k: w2["dist"][rk][k] for k in w1["dist"][rk]["A0"]})
        row(f"table 2.1 code distribution {rk} A0 first draft", w1["dist"][rk]["A0_first_draft_5237"]["codes"],
            w2["dist"][rk + "_first_draft"]["codes"])
        for sens in ("as_rated", "strict_discrepant"):
            row(f"table 2.2 rated endpoints {rk} {sens}", w1["rated"][f"{rk}|A0|{sens}"], w2["rated"][f"{rk}|{sens}"])
    row("table 2.2 R1 with 23 split rows as derivable", w1["rated"]["R1|A0|R1_with_23_split_rows_as_derivable"],
        w2["rated"]["R1|R1_with_23_split_rows_as_derivable"])
    ae1 = {k: v for k, v in w1["auto_end"]["A0"].items() if k != "true_group_named_text"}
    row("table 2.3 automatic endpoints A0", ae1, {k: w2["auto_end"][k] for k in ae1})
    row("table 2.3 true group named when absent A0", w1["tg_abs"]["A0"], w2["tg_abs"])
    row("table 2.3 probability statements A0", w1["prob_acc"]["A0"], w2["prob_acc"])
    for rk in ("R1", "R2"):
        row(f"probability-tagged sentences {rk} A0", w1["probtag"][f"{rk}|A0"], w2["probtag"][rk])
    for m in ("raw", "post_alpha_gamma"):
        row(f"table 2.4 zero flags {m} A0 (queries)", w1["zero_lists"][m]["A0"], w2["zero_lists"][m])
    row("table 2.4 flagged after alpha-gamma A0", w1["flagged_after"]["A0"], w2["flagged_after"])
    fc1 = pd.read_csv(R1 / "analysis/flag_classification.csv", dtype=str, keep_default_na=False)
    fc1 = fc1[fc1.arm == "A0"][["run_uid", "source", "flag_type", "value", "class"]].sort_values(list(fc1.columns[:1])).values.tolist()
    fc2 = pd.read_csv(RB / "port_output/flag_classification.csv", dtype=str, keep_default_na=False)
    fc2 = fc2[["run_uid", "source", "flag_type", "value", "class"]].sort_values("run_uid").values.tolist()
    row("flag inventory A0 (run_uid, source, type, value, class) as a multiset", sorted(map(tuple, fc1)), sorted(map(tuple, fc2)))
    row("generated flags left unclassified", 0, sum(r[4] == "unclassified" for r in fc2))
    for k, v in w1["l133"].items():
        rk, fm, rd = k.split("|")
        row(f"§3 example rule ({rk}, {fm}, {rd})", v[0], w2["round1_rule"][k])
    row("rater agreement (all rows, A0) — recomputed from round-1 join files",
        int(((lambda a, b: (a.code == b.code) & (a.arm == ARM))(
            pd.read_csv(R1 / "analysis/rating_join_r1.csv", dtype=str, keep_default_na=False),
            pd.read_csv(R1 / "analysis/rating_join_r2.csv", dtype=str, keep_default_na=False))).sum()),
        w2["n_agree"])
    ok = all(r[3] for r in rows)
    L = ["# S5c analysis port — readback on round 1 A0", "",
         f"Port `scripts/llm_pilot_round2/analyze_r2.py` sha256 `{sha(HERE / 'analyze_r2.py')}`; this readback script "
         f"`{sha(__file__)}`. Run on round 1 (`{R1}/run`, read only) with round 1's A0 rulings transcribed into the "
         f"external adjudication file `{adj.name}` sha256 `{sha(adj)}`. Reference: round-1 `analysis/endpoints.json` "
         f"`{sha(R1 / 'analysis/endpoints.json')}` (source of `ENDPOINTS_REPORT.md` "
         f"`{sha(R1 / 'analysis/ENDPOINTS_REPORT.md')}`) and `analysis/work/analysis_work.json` "
         f"`{sha(R1 / 'analysis/work/analysis_work.json')}`.", "",
         f"**Result: {'PASS' if ok else 'FAIL'}** — {sum(r[3] for r in rows)}/{len(rows)} items equal.", "",
         "Command: `" + " ".join(cmd) + "`", "",
         "| item | round 1 | port | equal |", "|---|---|---|---|"]
    for item, v1, v2, eq in rows:
        s1, s2 = json.dumps(v1, default=str), json.dumps(v2, default=str)
        if eq and len(s1) > 160:
            s1 = s2 = f"(identical; sha256 {hashlib.sha256(s1.encode()).hexdigest()[:12]})"
        L.append(f"| {item} | {s1.replace('|', '/')} | {s2.replace('|', '/')} | {'yes' if eq else 'NO'} |")
    L += ["", "Not ported (not A0, or not an endpoint): A1/A2/R rows, A2 (a)-(c), conflicting cases, repeats, the "
          "review-comparison columns of table 2.4, the 'paired A0/A1' table and report prose."]
    (DEST / "READBACK_A0.md").write_text("\n".join(L) + "\n")
    return ok, len(rows)


if __name__ == "__main__":
    adj = transcribe()
    cmd = run_port(adj)
    ok, n = compare(cmd, adj)
    print("READBACK", "PASS" if ok else "FAIL", n)
    sys.exit(0 if ok else 1)
