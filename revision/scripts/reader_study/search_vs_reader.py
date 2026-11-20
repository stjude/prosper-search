"""Exploratory, post hoc: slide search alone versus each reader on the same reader-study cases.

Not part of the locked plan (002/003). Reads `readings.csv` of a finished analysis directory, the frozen
manifests (ten nearest reference slides per case, as shown to the readers) and the QC verdicts of the
frozen export; writes to a sibling directory `<analysis>_search_alone_vs_readers/`. Nothing is written
under the export, the study material or the analysis directory, and no stored reader answer is changed.

"Search alone" is a rule applied to the displayed neighbour labels, with no human input:
  top1          label of the nearest reference slide
  plurality5    most frequent label among the 5 nearest; count ties go to the nearest-ranked tied label
                (the manuscript's vote rule; headline rule, fixed before the paired comparison was run)
  plurality10   same rule over the 10 displayed slides
It is compared with the reader's favoured diagnosis (strict match, as in the locked analysis) at Step 1
(slide and clinical metadata only), Step 2 (after the search results) and Step 3 (after the summary).
Whole-case bootstrap as in endpoints.py (seed 20261027, 5,000 draws); differences are search − reader.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import endpoints as ep  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PHASES = ("A", "B", "C")
STEP = {"A": "Step 1 (slide + metadata)", "B": "Step 2 (after search)", "C": "Step 3 (after summary)"}
RULES = ("plurality5", "top1", "plurality10")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def plurality(labels: Sequence[str], k: int) -> str:
    """Most frequent label among the first k; count ties go to the tied label ranked nearest."""
    top = list(labels[:k])
    counts = Counter(top)
    best = max(counts.values())
    return next(lab for lab in top if counts[lab] == best)


def distinct_in_rank_order(labels: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(labels))


def consensus(favoured: Sequence[str]) -> str:
    """Single most frequent favoured diagnosis among the readers; a tie gives no consensus ('')."""
    counts = Counter(favoured)
    best = max(counts.values())
    leaders = [g for g, n in counts.items() if n == best]
    return leaders[0] if len(leaders) == 1 else ""


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p for b versus c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(case_ids: Sequence[str], reader_ok: Sequence[bool], search_ok: Sequence[bool]) -> Dict[str, object]:
    cases = sorted(set(case_ids))
    w = ep.case_weights(len(cases))
    e = ep.paired_difference(case_ids, reader_ok, search_ok, cases, w)
    r = [bool(v) for v in reader_ok]
    s = [bool(v) for v in search_ok]
    only_s, only_r = e["x_wrong_y_right"], e["x_right_y_wrong"]
    return {"n": e["n"], "reader_correct": sum(r), "search_correct": sum(s),
            "diff_search_minus_reader": e["estimate"], "boot95_lo": e["boot95_lo"], "boot95_hi": e["boot95_hi"],
            "both_right": sum(a and b for a, b in zip(r, s)), "only_search_right": only_s,
            "only_reader_right": only_r, "neither_right": sum(not a and not b for a, b in zip(r, s)),
            "mcnemar_exact_p_unadjusted": mcnemar_exact(only_s, only_r)}


def to_md(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.3f}"
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--readers", required=True, help="comma-separated, in manuscript order (Reader 1..n)")
    args = ap.parse_args(argv)
    readers = [x.strip() for x in args.readers.split(",") if x.strip()]
    analysis, export, data = args.analysis.resolve(), args.export.resolve(), args.data.resolve()
    out = analysis.parent / f"{analysis.name}_search_alone_vs_readers"
    for forbidden in (export, data, analysis):
        if out == forbidden or forbidden in out.parents:
            raise SystemExit("refusing to write under a read-only input")

    run = json.loads((analysis / "analyze_run.json").read_text())
    if sha256_file(analysis / "readings.csv") != run["readings_sha256"]:
        raise SystemExit("readings.csv differs from the analysed file")
    r = pd.read_csv(analysis / "readings.csv")
    r = r[r.reader_id.isin(readers)].copy()
    name = {rid: f"Reader {i}" for i, rid in enumerate(readers, start=1)}
    r["reader"] = r.reader_id.map(name)
    bool_cols = [f"{m}_{p}" for m in ("fav_correct", "in_diff") for p in PHASES] + [
        "top1_correct", "hit5", "hit10", "majority10_correct", "llm_top_correct"]
    for col in bool_cols:
        r[col] = r[col].map(lambda v: str(v).lower() == "true")

    # ---- search-alone answers per case, from the frozen manifests
    case = r.drop_duplicates("case_id").set_index("case_id").sort_index()
    if not (r.groupby("case_id").size() == len(readers)).all():
        raise SystemExit("not every case was read by every reader")
    top10: Dict[str, List[str]] = {}
    for cid, sid in case.slide_id.items():
        m = json.loads((data / "manifests" / f"{sid}.json").read_text())
        labs = [n["label"] for n in sorted(m["slide_neighbours"], key=lambda n: n["rank"])[:10]]
        if len(labs) != 10:
            raise SystemExit(f"{cid}: {len(labs)} neighbours")
        top10[cid] = labs
    ans = pd.DataFrame({"top1": {c: v[0] for c, v in top10.items()},
                        "plurality5": {c: plurality(v, 5) for c, v in top10.items()},
                        "plurality10": {c: plurality(v, 10) for c, v in top10.items()}})
    ok = ans.eq(case.reference, axis=0)
    # cross-check against the locked pipeline's case covariates
    if not (ok.top1 == case.top1_correct).all():
        raise SystemExit("top-1 recount differs from readings.csv")
    if not ((case.hit5 == [case.reference[c] in top10[c][:5] for c in case.index]).all()
            and (case.hit10 == [case.reference[c] in top10[c] for c in case.index]).all()):
        raise SystemExit("hit@5/hit@10 recount differs from readings.csv")
    n_tie10 = int((ok.plurality10 & ~case.majority10_correct).sum())
    if (case.majority10_correct & ~ok.plurality10).any():
        raise SystemExit("plurality-of-10 (tie = false) correct where the tie-broken rule is not")

    sub = pd.read_csv(export / "submissions_export.csv", usecols=["role", "case_id", "a_verdict"])
    qc = sub[sub.role == "qc"]
    verdict = dict(zip(qc.case_id, qc.a_verdict))
    qc_pass = sorted(c for c in case.index if verdict.get(c) == "pass")
    sets = {"all 80": list(case.index), "CCDI-MCI 40": list(case.index[case.cohort == "CCDI"]),
            "STV 40": list(case.index[case.cohort == "STV"]), f"QC-pass {len(qc_pass)}": qc_pass}
    out.mkdir(parents=True, exist_ok=True)
    T: Dict[str, pd.DataFrame] = {}

    # ---- V1 search alone, each rule and shortlist criterion
    rows = []
    crit = {**{k: ok[k] for k in RULES}, "plurality10 (tie = incorrect; locked covariate)": case.majority10_correct,
            "summary's stated leading group (parsed; needs the LLM step)": case.llm_top_correct,
            "hit@5 (shortlist)": case.hit5, "hit@10 (shortlist)": case.hit10}
    for sname, cs in sets.items():
        w = ep.case_weights(len(cs))
        for cname, flags in crit.items():
            e = ep.proportion_endpoint(cs, flags.loc[cs], cs, w)
            rows.append({"case_set": sname, "search_criterion": cname, **e})
    T["V1_search_alone"] = pd.DataFrame(rows)

    # ---- V2 search rule versus reader favoured, per reader x step x case set
    rows = []
    for sname, cs in sets.items():
        rs = r[r.case_id.isin(cs)]
        for rule in RULES:
            s_ok = rs.case_id.map(ok[rule])
            for p in PHASES:
                for rid in readers:
                    x = rs[rs.reader_id == rid]
                    rows.append({"case_set": sname, "search_rule": rule, "step": STEP[p], "reader": name[rid],
                                 "tier": x.expertise_tier.iloc[0],
                                 **compare(x.case_id, x[f"fav_correct_{p}"], s_ok.loc[x.index])})
                # all readers together: every reading of a drawn case enters the draw
                e = compare(rs.case_id, rs[f"fav_correct_{p}"], s_ok)
                e["mcnemar_exact_p_unadjusted"] = float("nan")   # readings of a case are not independent
                rows.append({"case_set": sname, "search_rule": rule, "step": STEP[p],
                             "reader": f"all {len(readers)} readers (readings pooled)", "tier": "", **e})
                fav = rs.pivot(index="case_id", columns="reader_id", values=f"favoured_{p}").loc[cs]
                cons_ok = [consensus(list(fav.loc[c])) == case.reference[c] for c in cs]
                rows.append({"case_set": sname, "search_rule": rule, "step": STEP[p],
                             "reader": f"consensus of {len(readers)} (most frequent favoured; tie = incorrect)",
                             "tier": "", **compare(cs, cons_ok, ok[rule].loc[cs])})
    T["V2_search_vs_reader_favoured"] = pd.DataFrame(rows)

    # ---- V3 shortlists: reference label in the reader's Step 1 differential versus in the retrieved list
    rows = []
    distinct = {c: distinct_in_rank_order(v) for c, v in top10.items()}
    for rid in readers:
        x = r[r.reader_id == rid]
        matched = pd.Series([case.reference[c] in distinct[c][:int(n)] for c, n in zip(x.case_id, x.n_dx_A)],
                            index=x.index)
        for cname, flags in (("hit@5 (5 slides)", x.case_id.map(case.hit5)),
                             ("hit@10 (10 slides)", x.case_id.map(case.hit10)),
                             ("first n distinct retrieved groups, n = reader's Step 1 list length", matched)):
            e = compare(x.case_id, x.in_diff_A, flags)
            rows.append({"reader": name[rid], "tier": x.expertise_tier.iloc[0],
                         "reader_list_len_mean": float(x.n_dx_A.mean()), "search_shortlist": cname,
                         "search_distinct_groups_mean": float(np.mean(
                             [len(distinct[c][:5]) if cname.startswith("hit@5") else len(distinct[c])
                              for c in x.case_id])) if cname.startswith("hit@") else float("nan"),
                         **{k.replace("reader_correct", "reader_in_differential").replace(
                             "search_correct", "search_in_shortlist"): v for k, v in e.items()}})
    T["V3_shortlist_vs_differential"] = pd.DataFrame(rows)

    # ---- V4 complementarity under the headline rule
    rows = []
    for p in PHASES:
        n_right = r.groupby("case_id")[f"fav_correct_{p}"].sum().loc[case.index]
        for s_right in (True, False):
            cs = case.index[ok.plurality5 == s_right]
            dist = Counter(int(v) for v in n_right.loc[cs])
            rows.append({"step": STEP[p], "search_plurality5": "correct" if s_right else "incorrect",
                         "n_cases": len(cs), **{f"{k}_of_{len(readers)}_readers_right": dist.get(k, 0)
                                                for k in range(len(readers) + 1)},
                         "reader_readings_right": int(n_right.loc[cs].sum()),
                         "reader_readings": len(cs) * len(readers)})
    T["V4_complementarity"] = pd.DataFrame(rows)

    # ---- V5 by reference group
    g = pd.DataFrame({"n_cases": case.groupby("reference").size()})
    for rule in RULES:
        g[f"search_{rule}_correct"] = ok[rule].groupby(case.reference).sum()
    for p in ("A", "B"):
        per = r.groupby(["reference", "reader"])[f"fav_correct_{p}"].sum().unstack()
        for col in per.columns:
            g[f"{col}_step{'1' if p == 'A' else '2'}_correct"] = per[col]
    T["V5_by_reference_group"] = g.sort_values("n_cases", ascending=False).reset_index()

    # ---- V6 per case
    pc = pd.DataFrame({"cohort": case.cohort, "reference": case.reference, "qc_verdict": pd.Series(verdict),
                       "difficulty_tier": case.difficulty_tier,
                       "retrieved_top10": {c: " ".join(v) for c, v in top10.items()}}).loc[case.index]
    for rule in RULES:
        pc[f"search_{rule}"] = ans[rule]
        pc[f"search_{rule}_correct"] = ok[rule]
    for p in ("A", "B"):
        fav = r.pivot(index="case_id", columns="reader", values=f"favoured_{p}")
        for col in fav.columns:
            pc[f"{col}_favoured_step{'1' if p == 'A' else '2'}"] = fav[col]
        pc[f"n_readers_right_step{'1' if p == 'A' else '2'}"] = r.groupby("case_id")[f"fav_correct_{p}"].sum()
    T["V6_per_case"] = pc.reset_index().rename(columns={"index": "case_id"})

    # ---- V7 per reader, by whether the headline rule matched: cases right at each step, and whether an
    #      answer that was right at Step 1 was still right at the later step
    rows = []
    for rid in readers:
        x = r[r.reader_id == rid].set_index("case_id")
        for s_right in (False, True):
            cs = case.index[ok.plurality5 == s_right]
            a = x.fav_correct_A.loc[cs]
            row = {"reader": name[rid], "tier": x.expertise_tier.iloc[0],
                   "search_plurality5": "correct" if s_right else "incorrect", "n_cases": len(cs)}
            for p, lab in zip(PHASES, ("step1", "step2", "step3")):
                row[f"reader_right_{lab}"] = int(x[f"fav_correct_{p}"].loc[cs].sum())
            row["right_step1_still_right_step2"] = int((a & x.fav_correct_B.loc[cs]).sum())
            row["right_step1_still_right_step3"] = int((a & x.fav_correct_C.loc[cs]).sum())
            rows.append(row)
    T["V7_by_vote_outcome_per_reader"] = pd.DataFrame(rows)

    # ---- V8 who corrected whom: per reader, by search outcome (three strata) and by whether the reader's
    #      Step 1 favored diagnosis was right: right after retrieval, and favored label equal to the vote label
    stratum = pd.Series("vote of 5 right", index=case.index)
    stratum[~ok.plurality5 & case.hit5] = "vote wrong, reference group among the 5 nearest"
    stratum[~ok.plurality5 & ~case.hit5] = "reference group not among the 5 nearest"
    rows = []
    for rid in readers:
        x = r[r.reader_id == rid].set_index("case_id").loc[case.index]
        same_a, same_b = x.favoured_A == ans["plurality5"], x.favoured_B == ans["plurality5"]
        for st in stratum.unique():
            for a_right in (True, False):
                m = (stratum == st) & (x.fav_correct_A == a_right)
                rows.append({"reader": name[rid], "tier": x.expertise_tier.iloc[0], "search_outcome": st,
                             "reader_step1": "right" if a_right else "wrong", "n_cases": int(m.sum()),
                             "right_step2": int((m & x.fav_correct_B).sum()),
                             "favored_equals_vote_label_step1": int((m & same_a).sum()),
                             "favored_equals_vote_label_step2": int((m & same_b).sum()),
                             "favored_changed_step2": int((m & (x.favoured_A != x.favoured_B)).sum()),
                             "changed_to_vote_label": int((m & (x.favoured_A != x.favoured_B) & same_b).sum())})
    T["V8_who_corrected_whom"] = pd.DataFrame(rows)

    for k, df in T.items():
        df.to_csv(out / f"{k}.csv", index=False)
        (out / f"{k}.md").write_text(to_md(df))

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    rec = [
        "# RUN_RECORD — search alone versus readers (search_vs_reader.py; exploratory, post hoc)", "",
        f"- run: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- analysis: `{analysis}` (readings.csv {run['readings_sha256']}; analyze commit {run['commit']})",
        f"- export: `{export}`; submissions_export.csv {sha256_file(export / 'submissions_export.csv')} "
        "(QC verdicts only; opened read-only)",
        f"- manifests: `{data / 'manifests'}` ({len(top10)} cases, ten neighbours each)",
        f"- readers: {', '.join(f'{name[x]} = {x}' for x in readers)}",
        f"- QC verdicts: {len(qc_pass)} pass of {len(case)}",
        f"- recount checks: top-1, hit@5 and hit@10 equal the readings.csv covariates; plurality of 10 with the "
        f"nearest-rank tie-break is correct on {n_tie10} case(s) that the tie = incorrect covariate counts as wrong",
        f"- bootstrap: whole-case, seed {ep.SEED}, {ep.N_DRAWS} draws over the cases of each case set",
        f"- script commit (HEAD): {git('rev-parse', 'HEAD')}; scripts/reader_study clean: "
        + ("yes" if not git("status", "--porcelain", "--", "scripts/reader_study") else "NO (this script uncommitted)"),
        "- output sha256:",
    ] + [f"  - {p.name}: {sha256_file(p)}" for p in sorted(out.glob("V*.csv"))]
    (out / "RUN_RECORD.generated.md").write_text("\n".join(rec) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
