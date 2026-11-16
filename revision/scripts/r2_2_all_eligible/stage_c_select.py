#!/usr/bin/env python3
"""R2.2-ALL Stage C: deterministic, size-based canary selection (declared before any Stage C result).

Selection uses only tile counts, the no-list/fallback flags and patient ids. It uses no labels,
no distances and no retrieval outcome.

Inputs: stage_b/bank_per_slide.csv (per-slide eligible/code counts), stage_b/gap_d1/per_slide.csv
(n_gap_partial), the labels/patient CSV (patient ids, used only to avoid same-patient probe pairs).

Rules (fixed in the [analysis plan] Stage C pre-registration):
  Q_size     12 slides at size-rank quantiles {0,.05,.10,.25,.40,.50,.60,.75,.90,.95,.99,1}
             of the 2,778 slides sorted by (n_eligible, slide id); index = round(q*(N-1)).
  Q_partial  4 slides with the highest n_gap_partial / n_eligible among slides that have a
             fully-tissue bank file (n_code1 > 0) and n_eligible >= the 25th percentile.
  Q_nolist   4 of the 25 non-fallback slides without a fully-tissue bank file, at size-rank
             quantiles {0, 1/3, 2/3, 1} of that subset.
  Q_fallback the 3 fallback slides 9934, 11682, 14191.
  When a rule picks an already-selected slide, the next slide in that rule's order is used.
  OT rows    the Q_size slides at q = .10, .50, .90, .99.
  OT pairs   for each q in {0, .10, .50, .90, .99}: (Q_size slide, nearest-in-size slide with a
             different patient); plus the largest slide paired with the 2nd and with the 3rd largest
             slides of different patients. Both orientations are solved.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

QS = [0, .05, .10, .25, .40, .50, .60, .75, .90, .95, .99, 1.0]
FALLBACK = ["9934", "11682", "14191"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = Path(a.stage_b)
    bank = pd.read_csv(B / "bank_per_slide.csv", dtype={"slide": str})
    gap = pd.read_csv(B / "gap_d1/per_slide.csv", dtype={"slide": str})
    lab = pd.read_csv(a.labels, dtype={"Slide.Scan.File": str})
    pat = dict(zip(lab["Slide.Scan.File"], lab["UID.Subject"].astype(str)))
    d = bank.merge(gap[["slide", "n_gap_partial"]], on="slide", validate="1:1")
    assert len(d) == 2778, len(d)
    d["n"] = d["n_rows"].astype(int)
    d = d.sort_values(["n", "slide"], key=lambda s: s if s.name == "n" else s.astype(int)).reset_index(drop=True)
    N = len(d)
    chosen, why = [], {}

    def take(order, label):
        for s in order:
            if s not in chosen:
                chosen.append(s)
                why[s] = label
                return s
        raise RuntimeError(label)

    qsize = {}
    for q in QS:
        i = int(round(q * (N - 1)))
        order = list(d.slide.iloc[i:]) + list(d.slide.iloc[:i][::-1])
        qsize[q] = take(order, f"Q_size q={q}")
    p25 = np.percentile(d.n, 25)
    c = d[(d.n_code1 > 0) & (d.n >= p25)].copy()
    c["pf"] = c.n_gap_partial / c.n
    c = c.sort_values(["pf", "slide"], ascending=[False, True])
    for k in range(4):
        take(list(c.slide), f"Q_partial #{k+1}")
    nl = d[(d.n_code1 == 0) & (~d.slide.isin(FALLBACK))].reset_index(drop=True)
    assert len(nl) == 25, len(nl)
    for q in [0, 1 / 3, 2 / 3, 1]:
        i = int(round(q * (len(nl) - 1)))
        take(list(nl.slide.iloc[i:]) + list(nl.slide.iloc[:i][::-1]), f"Q_nolist q={q:.3f}")
    for s in FALLBACK:
        take([s], "Q_fallback")

    ot_rows = [qsize[q] for q in (.10, .50, .90, .99)]
    pos = {s: i for i, s in enumerate(d.slide)}
    pairs = []

    def partner(s, exclude=()):
        i = pos[s]
        for k in range(1, N):
            for j in (i + k, i - k):
                if 0 <= j < N:
                    t = d.slide.iloc[j]
                    if t != s and t not in exclude and pat.get(t) != pat.get(s):
                        return t
        raise RuntimeError(s)

    for q in (0, .10, .50, .90, .99):
        s = qsize[q]
        pairs.append({"label": f"q={q}", "a": s, "b": partner(s)})
    top = list(d.slide.iloc[::-1])
    big = top[0]
    others = [t for t in top[1:] if pat.get(t) != pat.get(big)]
    pairs.append({"label": "largest x 2nd", "a": big, "b": others[0]})
    pairs.append({"label": "largest x 3rd", "a": big, "b": others[1]})
    n = dict(zip(d.slide, d.n))
    for p in pairs:
        p["n_a"], p["n_b"] = int(n[p["a"]]), int(n[p["b"]])
        p["cells"] = p["n_a"] * p["n_b"]
        p["same_patient"] = pat.get(p["a"]) == pat.get(p["b"])
    out = {
        "rule": __doc__,
        "inputs_sha256": {str(B / "bank_per_slide.csv"): sha(B / "bank_per_slide.csv"),
                          str(B / "gap_d1/per_slide.csv"): sha(B / "gap_d1/per_slide.csv"),
                          a.labels: sha(a.labels)},
        "queries": [{"slide": s, "why": why[s], "n_eligible": int(n[s]),
                     "n_code0": int(d.loc[pos[s], "n_code0"]), "n_code1": int(d.loc[pos[s], "n_code1"]),
                     "n_code2": int(d.loc[pos[s], "n_code2"]),
                     "n_gap_partial": int(d.loc[pos[s], "n_gap_partial"]),
                     "patient": pat.get(s)} for s in chosen],
        "ot_rows": ot_rows,
        "ot_pairs": pairs,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for qq in out["queries"]:
        print(qq["slide"], qq["why"], qq["n_eligible"], qq["n_code2"], qq["n_gap_partial"])
    print("ot_rows", ot_rows)
    for p in pairs:
        print(p)


if __name__ == "__main__":
    main()
