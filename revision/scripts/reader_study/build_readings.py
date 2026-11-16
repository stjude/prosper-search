"""PROSPER reader study, handoff steps 1–3: integrity checks, derived variables, LLM parse review.

    python3 scripts/reader_study/build_readings.py --readers <reader-login>

Writes into --out (never under the export): hash_check.txt, population_check.txt, readings.csv,
events_engagement.csv, llm_parse_review.md, RUN_RECORD.md. Exits non-zero, before building any
table, if a hash, population or material check fails. Steps 4–6 (endpoints, bootstrap, tables,
figures, results note) are not implemented here.

The sqlite snapshot is opened read-only and its `readers` table (sign-in tokens) is never queried.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive as d  # noqa: E402
import llm_parse as lp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXPORT_STAMP = "prod_export_20261027-101111"
DEFAULT_OUT_ROOT = os.environ.get("PROSPER_OUT", str(REPO / "prosper_revision_output"))
CDT = dt.timezone(dt.timedelta(hours=-5), "CDT")

# Handoff step 1 expectations for this export only.
_EXP = d.READER_CONFIG.get("export_checks", {}).get(EXPORT_STAMP, {})
EXPECTED = {
    EXPORT_STAMP: {
        "counts": {k: tuple(v) for k, v in _EXP.get("counts", {}).items()},
        "material_matches": dict(_EXP.get("material_matches", {})),
    }
}


class CheckFailed(Exception):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sha_list(p: Path, basename: bool = False) -> Dict[str, str]:
    """Parse `sha256sum` output. `basename=True` keys by file name only, so prod_hashes listings
    made as `sha256sum manifests/*.json` (exports from 2026-11-11 on) match those made inside the
    directory (exports up to 2026-11-08); digests are compared unchanged."""
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            name = name.strip().lstrip("*")
            out[Path(name).name if basename else name] = digest
    return out


# ---------------------------------------------------------------- step 1: hashes

def hash_check(export: Path, data: Path) -> List[str]:
    lines = [f"Hash check, {dt.datetime.now(CDT).isoformat(timespec='seconds')}", f"export: {export}", ""]
    ok = True
    sums = read_sha_list(export / "SHA256SUMS")
    lines.append(f"SHA256SUMS ({len(sums)} entries), sha256 of SHA256SUMS = {sha256_file(export / 'SHA256SUMS')}")
    for name, digest in sums.items():
        actual = sha256_file(export / name)
        good = actual == digest
        ok &= good
        lines.append(f"  {name}: {'OK' if good else 'FAILED'}")
    lines.append("")
    for sha_name, sub, ext in (("user_manifests.sha", "manifests", ".json"),
                               ("user_llm.sha", "llm", ".txt")):
        prod = read_sha_list(export / "prod_hashes" / sha_name, basename=True)
        files = sorted((data / sub).glob(f"*{ext}"))
        names = {f.name for f in files}
        match = sum(1 for f in files if prod.get(f.name) == sha256_file(f))
        missing = sorted(set(prod) - names)
        extra = sorted(names - set(prod))
        good = match == len(prod) == len(files) and not missing and not extra
        ok &= good
        lines.append(f"analysis copies {data / sub}: {match}/{len(prod)} identical to prod_hashes/{sha_name}"
                     f" ({len(files)} files; missing {missing or 'none'}; extra {extra or 'none'}): "
                     f"{'OK' if good else 'FAILED'}")
    lines += ["", f"RESULT: {'PASS' if ok else 'FAIL'}"]
    if not ok:
        raise CheckFailed("\n".join(lines))
    return lines


# ---------------------------------------------------------------- loading

def load_export(export: Path) -> Dict[str, object]:
    subs = pd.read_csv(export / "submissions_export.csv", dtype=str, keep_default_na=False)
    for col in ("t_open", "t_submit", "seconds"):
        subs[col] = pd.to_numeric(subs[col].replace("", None))
    cases = json.loads((export / "study_config" / "cases.json").read_text())
    readers = [json.loads(l) for l in (export / "readers_list.jsonl").read_text().splitlines() if l.strip()]
    if any("token" in r for r in readers):
        raise CheckFailed("readers_list.jsonl unexpectedly carries tokens; refusing to continue")
    slides_index = pd.read_csv(export / "study_config" / "slides_index.csv", dtype=str, keep_default_na=False)
    con = sqlite3.connect(f"file:{export / 'db' / 'study.sqlite'}?mode=ro", uri=True)
    try:
        db_subs = pd.read_sql_query(
            "SELECT reader_id, case_id, phase, payload, t_open, t_submit FROM submissions", con)
        events = pd.read_sql_query("SELECT id, reader_id, case_id, kind, payload, ts FROM events", con)
        assignments = pd.read_sql_query("SELECT reader_id, case_id, order_idx FROM assignments", con)
    finally:
        con.close()
    events["payload"] = events["payload"].map(lambda s: json.loads(s) if s else {})
    return {"subs": subs, "cases": cases, "readers": readers, "slides_index": slides_index,
            "db_subs": db_subs, "events": events, "assignments": assignments}


def dx_columns(df: pd.DataFrame) -> List[str]:
    return sorted((c for c in df.columns if re.fullmatch(r"a_dx\d+", c)), key=lambda c: int(c[4:]))


def row_answer(row: pd.Series, dx_cols: List[str], phase: str) -> Optional[d.Answer]:
    dx = [row[c] for c in dx_cols if row[c].strip()]
    if not dx and not row["a_favoured"].strip():
        return None
    ans = d.make_answer(dx, row["a_favoured"], phase)
    if int(row["a_n_dx"]) != len(ans.dx):
        raise CheckFailed(f"{row['reader_id']}/{row['case_id']}/{phase}: a_n_dx disagrees with dx fields")
    slot = row["a_favoured_slot"]
    if slot != f"dx{ans.dx.index(ans.favoured) + 1}":
        raise CheckFailed(f"{row['reader_id']}/{row['case_id']}/{phase}: favoured_slot inconsistent")
    return ans


# ---------------------------------------------------------------- step 1: population

def population_check(ex: Dict[str, object], export: Path, analysed: List[str]) -> List[str]:
    subs: pd.DataFrame = ex["subs"]  # type: ignore[assignment]
    db_subs: pd.DataFrame = ex["db_subs"]  # type: ignore[assignment]
    readers = ex["readers"]
    roles = {r["reader_id"]: r.get("role") for r in readers}
    expected = EXPECTED.get(export.name)
    ok = True
    L = [f"Population check, {dt.datetime.now(CDT).isoformat(timespec='seconds')}", f"export: {export}", ""]

    L.append("Submissions per reader x phase in submissions_export.csv (counts only):")
    counts = subs.groupby(["reader_id", "phase"]).size().unstack(fill_value=0)
    for rid, r in counts.iterrows():
        status = ("excluded by name" if d.is_excluded_reader(rid)
                  else "ANALYSED" if rid in analysed else "HELD (counts only)" if rid in d.EXPERTISE_TIER
                  else "not in expertise map")
        pilots = sorted(set(subs[(subs.reader_id == rid) & subs.case_id.isin(d.PILOT_CASES)].case_id))
        L.append(f"  {rid:20s} role={roles.get(rid)!s:12s} A/B/C = {r.get('A', 0)}/{r.get('B', 0)}/{r.get('C', 0)}"
                 f"  pilots={pilots or 'none'}  [{status}]")
    L.append("")
    no_subs = sorted(set(roles) - set(counts.index))
    L.append(f"Accounts in readers_list.jsonl without submissions: {no_subs}")
    unlisted = sorted(r for r in roles if not d.is_excluded_reader(r) and r not in d.EXPERTISE_TIER)
    L.append(f"Accounts neither excluded by name nor in the expertise map: {unlisted}")
    L.append("")

    dup = subs.groupby(["reader_id", "case_id", "phase"]).size()
    n_dup = int((dup > 1).sum())
    ok &= n_dup == 0
    L.append(f"Duplicate reader x case x phase rows: {n_dup}")

    in_scope = [r for r in counts.index if not d.is_excluded_reader(r)]
    for rid in in_scope:
        rs = subs[subs.reader_id == rid]
        phases = rs.groupby("case_id").phase.apply(set)
        a_wo_b = sorted(c for c, ps in phases.items() if "A" in ps and "B" not in ps)
        b_wo_a = sorted(c for c, ps in phases.items() if "B" in ps and "A" not in ps)
        c_wo_b = sorted(c for c, ps in phases.items() if "C" in ps and "B" not in ps)
        # A Step 1 without Step 2 is tolerated only for the single case a reader had open at the snapshot:
        # exactly one such case, and its Step 1 is that reader's latest submission. It is not a reading
        # (build_readings requires A and B); it is reported here and never edited or dropped from the data.
        in_prog = ""
        if len(a_wo_b) == 1 and rs.t_submit.notna().all():
            last = rs.loc[rs.t_submit.idxmax()]
            if last.case_id == a_wo_b[0] and last.phase == "A":
                in_prog = a_wo_b[0]
        ok &= (not a_wo_b or bool(in_prog)) and not b_wo_a and not c_wo_b
        a_txt = (f"{a_wo_b} (in progress at snapshot: latest submission is its Step 1; not a reading)"
                 if in_prog else (a_wo_b or 'none'))
        L.append(f"{rid}: A without B {a_txt}; B without A {b_wo_a or 'none'}; C without B {c_wo_b or 'none'}")
        if expected and rid in expected["counts"]:
            got = tuple(int(counts.loc[rid].get(p, 0)) for p in d.PHASES)
            good = got == expected["counts"][rid]
            ok &= good
            L.append(f"{rid}: A/B/C {got} vs handoff {expected['counts'][rid]}: {'OK' if good else 'MISMATCH'}")
    L.append("")

    # export CSV vs read-only sqlite snapshot (non-demo rows)
    demo = {r["reader_id"] for r in readers if r.get("role") == "demo"}
    db_nd = db_subs[~db_subs.reader_id.isin(demo)]
    key = ["reader_id", "case_id", "phase"]
    m = subs.merge(db_nd, on=key, how="outer", suffixes=("", "_db"), indicator=True)
    only = Counter(m["_merge"])
    both = m[m["_merge"] == "both"]
    t_bad = int(((both.t_open - both.t_open_db).abs() > 1e-6).sum() + ((both.t_submit - both.t_submit_db).abs() > 1e-6).sum())
    pay_bad = 0
    dx_cols = dx_columns(subs)
    for _, r in both.iterrows():
        p = json.loads(r["payload"])
        for k, v in p.items():
            col = f"a_{k}"
            if col in subs.columns and str(v) != str(r[col]):
                pay_bad += 1
        for c in dx_cols:
            if r[c] and c[2:] not in p:
                pay_bad += 1
    # db-only rows are acceptable only if every one is a pilot case: since the 2026-11-08 export the deliverable
    # CSV drops case01/case02 as a pure row filter (amendment 008 item 2), while the sqlite backup is unfiltered.
    db_only = m[m["_merge"] == "right_only"]
    db_only_pilot = db_only.case_id.isin(d.PILOT_CASES)
    n_db_only_nonpilot = int((~db_only_pilot).sum())
    if len(db_only) and n_db_only_nonpilot == 0:
        db_txt = f"{len(db_only)} (all pilot cases: {'/'.join(sorted(set(db_only.case_id)))})"
    elif len(db_only):
        db_txt = f"{len(db_only)} (non-pilot: {n_db_only_nonpilot})"
    else:
        db_txt = "0"
    good = only.get("left_only", 0) == 0 and n_db_only_nonpilot == 0 and t_bad == 0 and pay_bad == 0
    ok &= good
    L.append(f"Export CSV vs sqlite submissions (non-demo): both={only.get('both', 0)} csv_only={only.get('left_only', 0)}"
             f" db_only={db_txt} time_mismatches={t_bad} payload_mismatches={pay_bad}: "
             f"{'OK' if good else 'MISMATCH'}")
    L.append("")

    # material hashes: first 16 hex of the prod sha256 of the manifest (B) / LLM text (C)
    prod = {"B": read_sha_list(export / "prod_hashes" / "user_manifests.sha", basename=True),
            "C": read_sha_list(export / "prod_hashes" / "user_llm.sha", basename=True)}
    ext = {"B": ".json", "C": ".txt"}
    L.append("Material hash re-verification (B: manifests, C: LLM text; non-pilot cases):")
    for rid in in_scope:
        rs = subs[(subs.reader_id == rid) & subs.phase.isin(["B", "C"])]
        np_rows = rs[~rs.case_id.isin(d.PILOT_CASES)]
        match = sum(1 for _, r in np_rows.iterrows()
                    if r["a_material_sha256"] and r["a_material_sha256"] == prod[r["phase"]].get(r["slide_id"] + ext[r["phase"]], "")[:16])
        pil = rs[rs.case_id.isin(d.PILOT_CASES)]
        pil_missing = int((pil["a_material_sha256"] == "").sum())
        good = match == len(np_rows)
        if expected and rid in expected["material_matches"]:
            good &= match == expected["material_matches"][rid]
        ok &= good
        L.append(f"  {rid}: {match}/{len(np_rows)} non-pilot B/C match"
                 + (f" (handoff expects {expected['material_matches'][rid]})" if expected and rid in expected["material_matches"] else "")
                 + f"; pilot B/C rows {len(pil)}, of which without material_sha256 {pil_missing}: {'OK' if good else 'FAIL'}")
    L += ["", f"RESULT: {'PASS' if ok else 'FAIL'}"]
    if not ok:
        raise CheckFailed("\n".join(L))
    return L


# ---------------------------------------------------------------- case covariates

def case_table(cases: List[dict], data: Path, slides_index: pd.DataFrame) -> pd.DataFrame:
    lib = d.library_support(slides_index.loc[slides_index.kind == "library", "label"])
    rows = []
    for c in cases:
        t = c["truth"]
        m = json.loads((data / "manifests" / f"{c['slide_id']}.json").read_text())
        nb = sorted(m["slide_neighbours"], key=lambda n: n["rank"])
        ret = d.neighbour_retrieval([n["label"] for n in nb], t["tumor_group"])
        roi = d.roi_success([(r["hits"][0]["label"] if r.get("hits") else None) for r in m["rois"]], t["tumor_group"])
        rows.append({
            "case_id": c["case_id"], "slide_id": c["slide_id"], "pilot": bool(c.get("pilot")),
            "cohort": t["cohort"], "tumor_group": t["tumor_group"],
            "stv_subclass": t.get("stv_subclass", ""), "stv_dx": t.get("stv_dx", ""),
            "ccdi_diagnosis": t.get("ccdi_diagnosis", ""),
            "difficulty_tier": t.get("selection", ""),
            **ret, **roi,
            "library_support_n": lib.get(t["tumor_group"], 0),
            "manifest_library_n_slides": m.get("method", {}).get("library", {}).get("n_slides"),
        })
    return pd.DataFrame(rows)


def llm_table(case_df: pd.DataFrame, data: Path) -> pd.DataFrame:
    rows = []
    for _, c in case_df.iterrows():
        text = (data / "llm" / f"{c.slide_id}.txt").read_text()
        parsed = lp.parse_summary(text)
        rows.append({"case_id": c.case_id, **lp.llm_case_vars(parsed, c.tumor_group), "_parsed": parsed, "_text": text})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- readings

def build_readings(ex: Dict[str, object], analysed: List[str], case_df: pd.DataFrame,
                   llm_df: pd.DataFrame, entity_map) -> (pd.DataFrame, pd.DataFrame):
    subs: pd.DataFrame = ex["subs"]  # type: ignore[assignment]
    events: pd.DataFrame = ex["events"]  # type: ignore[assignment]
    dx_cols = dx_columns(subs)
    cases = case_df.set_index("case_id")
    llm = llm_df.set_index("case_id")
    out_rows, ev_rows = [], []
    for rid in analysed:
        if d.is_excluded_reader(rid) or rid not in d.EXPERTISE_TIER:
            raise CheckFailed(f"{rid} is excluded or has no expertise tier")
        rs = subs[subs.reader_id == rid]
        all_submits = list(rs.t_submit)
        sess = dict(zip(zip(rs.case_id, rs.phase), d.session_ids(all_submits)))
        by_case = {cid: {r.phase: r for _, r in g.iterrows()} for cid, g in rs.groupby("case_id")}
        rcases = sorted(cid for cid, ph in by_case.items()
                        if cid not in d.PILOT_CASES and "A" in ph and "B" in ph)
        chrono = dict(zip(rcases, d.chronological_rank([by_case[c]["A"].t_open for c in rcases])))
        rev = events[(events.reader_id == rid)]
        for cid in rcases:
            ph = by_case[cid]
            cv = cases.loc[cid]
            ref = cv.tumor_group
            a, b, c = ph["A"], ph["B"], ph.get("C")
            ans = {"A": row_answer(a, dx_cols, "A"), "B": row_answer(b, dx_cols, "B"),
                   "C": row_answer(c, dx_cols, "C") if c is not None else None}
            eff = d.effective_answers(ans["A"], b["a_changed_assessment"], ans["B"],
                                      c["a_changed_assessment"] if c is not None else None, ans["C"])
            comments = {"A": a["a_comments"], "B": b["a_comments"], "C": c["a_comments"] if c is not None else ""}
            row = {"reader_id": rid, "expertise_tier": d.EXPERTISE_TIER[rid], "case_id": cid,
                   "slide_id": cv.slide_id, "cohort": cv.cohort, "reference": ref,
                   "stv_subclass": cv.stv_subclass, "stv_dx": cv.stv_dx, "ccdi_diagnosis": cv.ccdi_diagnosis,
                   "has_C": c is not None,
                   "changed_B": b["a_changed_assessment"],
                   "changed_C": c["a_changed_assessment"] if c is not None else None}
            for p in d.PHASES:
                e = eff[p]
                if e is None:
                    row.update({f"dx_{p}": None, f"favoured_{p}": None, f"eff_source_{p}": None,
                                f"n_dx_{p}": None, f"fav_correct_{p}": None, f"in_diff_{p}": None,
                                f"truth_rank_{p}": None, f"misc_in_{p}": None, f"normal_in_{p}": None,
                                f"cmap_note_{p}": None, f"fav_correct_{p}_cmap": None,
                                f"in_diff_{p}_cmap": None, f"truth_rank_{p}_cmap": None})
                    continue
                s = d.score_answer(e, ref)
                mapped, note = d.apply_entity_map(e, comments[e.source_phase], entity_map)
                sm = d.score_answer(mapped, ref)
                row.update({f"dx_{p}": " | ".join(e.dx), f"favoured_{p}": e.favoured,
                            f"eff_source_{p}": e.source_phase, f"n_dx_{p}": s["n_dx"],
                            f"fav_correct_{p}": s["fav_correct"], f"in_diff_{p}": s["in_diff"],
                            f"truth_rank_{p}": s["truth_rank"], f"misc_in_{p}": d.MISC in e.dx,
                            f"normal_in_{p}": "NORMAL" in e.dx, f"cmap_note_{p}": note,
                            f"fav_correct_{p}_cmap": sm["fav_correct"], f"in_diff_{p}_cmap": sm["in_diff"],
                            f"truth_rank_{p}_cmap": sm["truth_rank"]})
            for x, y, changed in (("A", "B", b["a_changed_assessment"] == "yes"),
                                  ("B", "C", c is not None and c["a_changed_assessment"] == "yes"),
                                  ("A", "C", b["a_changed_assessment"] == "yes"
                                   or (c is not None and c["a_changed_assessment"] == "yes"))):
                if row[f"fav_correct_{y}"] is None:
                    row[f"class_{x}{y}"] = None
                    row[f"confirm_sub_{x}{y}"] = None
                    continue
                cls, sub = d.outcome_class(row[f"fav_correct_{x}"], row[f"in_diff_{x}"],
                                           row[f"fav_correct_{y}"], row[f"in_diff_{y}"], changed)
                row[f"class_{x}{y}"] = cls
                row[f"confirm_sub_{x}{y}"] = sub
            conf = {"A": int(a["a_confidence"]), "B": int(b["a_confidence"]),
                    "C": int(c["a_confidence"]) if c is not None else None}
            row.update({"conf_A": conf["A"], "conf_B": conf["B"], "conf_C": conf["C"],
                        "use_B": int(b["a_search_usefulness"]),
                        "use_C": int(c["a_summary_usefulness"]) if c is not None else None})
            row.update(d.confidence_vars(conf["A"], conf["B"], conf["C"], row["fav_correct_B"]))
            times = {p: (ph[p].t_open, ph[p].t_submit) for p in d.PHASES if p in ph}
            for p in d.PHASES:
                row[f"t_open_{p}"] = times[p][0] if p in times else None
                row[f"t_submit_{p}"] = times[p][1] if p in times else None
            rt = d.reading_time_vars(times)
            for p in d.PHASES:
                if p in ph and abs(rt[f"sec_{p}"] - ph[p].seconds) > 1e-6:
                    raise CheckFailed(f"{rid}/{cid}/{p}: seconds column disagrees with t_submit - t_open")
            row.update(rt)
            t_open_c = c.t_open if c is not None else None
            t_submit_c = c.t_submit if c is not None else None
            cev = rev[rev.case_id == cid]
            evs = [{"kind": e.kind, "ts": e.ts, "payload": e.payload} for e in cev.itertuples()]
            row.update(d.engagement_vars(evs, a.t_submit, b.t_open, b.t_submit, t_open_c, t_submit_c))
            for e in cev.itertuples():
                w = d.event_windows(e.ts, a.t_submit, b.t_open, b.t_submit, t_open_c, t_submit_c)
                auto = d.is_auto(e.payload)
                ev_rows.append({
                    "reader_id": rid, "expertise_tier": d.EXPERTISE_TIER[rid], "case_id": cid,
                    "event_id": e.id, "kind": e.kind, "ts": e.ts, "auto": auto,
                    "rank": e.payload.get("rank"), "via": e.payload.get("via"), "mode": e.payload.get("mode"),
                    "phase_shown": e.payload.get("phase") if e.kind == "phase_shown" else None,
                    "timeline_position": d.phase_label(e.ts, a.t_submit, b.t_open, b.t_submit, t_open_c, t_submit_c),
                    "in_B_window": w["in_B_window"], "in_C_window": w["in_C_window"],
                    "counted_B": bool(w["in_B_window"] and not auto),
                    "counted_C_retrieval": bool(w["in_C_window"] and not auto and e.kind in d.RETRIEVAL_KINDS),
                })
            row.update({k: cv[k] for k in ("top1_correct", "hit5", "hit10", "n_truth_top10", "plurality10_label",
                                           "majority10_correct", "n_neighbours", "n_rois", "n_rois_without_hits",
                                           "roi_success_frac", "library_support_n", "difficulty_tier")})
            row["meta_visible_A"] = d.meta_visible_a(a.t_open, cv.cohort)
            row["chrono_rank"] = chrono[cid]
            row["session_id"] = sess[(cid, "A")]
            lv = llm.loc[cid]
            row.update({k: lv[k] for k in ("llm_top", "llm_top_correct", "llm_truth_prob",
                                           "llm_names_truth_leading", "llm_flags_discrepancy",
                                           "llm_wording_probability")})
            row["material_sha256_B"] = b["a_material_sha256"]
            row["material_sha256_C"] = c["a_material_sha256"] if c is not None else None
            row["comments_A"], row["comments_B"], row["comments_C"] = comments["A"], comments["B"], comments["C"]
            out_rows.append(row)
    return pd.DataFrame(out_rows), pd.DataFrame(ev_rows)


# ---------------------------------------------------------------- step 3 review file

def llm_review(case_df: pd.DataFrame, llm_df: pd.DataFrame) -> List[str]:
    L = ["# LLM summary parse review (80 non-pilot summaries)", "",
         "Generated by `build_readings.py`; each row is to be read against the quoted text.", ""]
    merged = case_df.merge(llm_df, on="case_id")
    merged = merged[~merged.pilot]
    for _, r in merged.iterrows():
        p = r["_parsed"]
        sents = lp.split_sentences(r["_text"])
        score_sents = [s for s in sents if lp._PROB_SENT_RE.search(s)]
        L += [f"## {r.case_id} ({r.slide_id}; truth {r.tumor_group})", "",
              f"- llm_top: **{r.llm_top}** (tie {p['llm_top_tie']}); scores {p['scores']}; sum {p['score_sum']};"
              f" unplaced numbers {p['unplaced_numbers'] or 'none'}",
              f"- llm_truth_prob: **{r.llm_truth_prob}**",
              f"- llm_names_truth_leading: **{r.llm_names_truth_leading}** (leading {p['leading']}, source"
              f" {p['leading_source']}, span \"{p['leading_span']}\")",
              f"- llm_flags_discrepancy: **{r.llm_flags_discrepancy}** (terms {p['discrepancy_terms'] or 'none'})",
              f"- llm_wording_probability: **{r.llm_wording_probability}**",
              "- score sentence(s): " + " || ".join(f"\"{s}\"" for s in score_sents),
              "- last two sentences: " + " || ".join(f"\"{s}\"" for s in sents[-2:]), ""]
    return L


# ---------------------------------------------------------------- main

def git_state() -> Dict[str, str]:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--", "scripts/reader_study")
    return {"commit": run("rev-parse", "HEAD"), "scripts_reader_study_clean": "yes" if not dirty else f"NO:\n{dirty}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path,
                    default=Path(DEFAULT_OUT_ROOT) / "reader_study_suite" / "exports" / EXPORT_STAMP)
    ap.add_argument("--data", type=Path, default=Path(DEFAULT_OUT_ROOT) / "reader_study_suite" / "data")
    ap.add_argument("--entity-map", type=Path,
                    default=REPO / "docs" / "pathologist_study" / "reader_study_misc_entity_map_20261027.csv")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--readers", required=True, help="comma-separated reader ids to analyse")
    args = ap.parse_args(argv)

    export = args.export.resolve()
    out = (args.out or Path(DEFAULT_OUT_ROOT) / "reader_study_suite" / "analysis"
           / f"20261027_{export.name}").resolve()
    for forbidden in (export, args.data.resolve()):
        if out == forbidden or forbidden in out.parents:
            raise SystemExit(f"refusing to write under the read-only input {forbidden}")
    analysed = [r.strip() for r in args.readers.split(",") if r.strip()]
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(CDT)
    git = git_state()

    try:
        hc = hash_check(export, args.data)
    except CheckFailed as e:
        (out / "hash_check.txt").write_text(str(e) + "\n")
        print("HASH CHECK FAILED; see hash_check.txt", file=sys.stderr)
        return 2
    (out / "hash_check.txt").write_text("\n".join(hc) + "\n")

    ex = load_export(export)
    try:
        pc = population_check(ex, export, analysed)
    except CheckFailed as e:
        (out / "population_check.txt").write_text(str(e) + "\n")
        print("POPULATION/MATERIAL CHECK FAILED; see population_check.txt", file=sys.stderr)
        return 3

    entity_map = d.load_entity_map(args.entity_map)
    case_df = case_table(ex["cases"], args.data, ex["slides_index"])
    llm_df = llm_table(case_df, args.data)
    pc += ["", "Slide-level retrieval recount from the frozen manifests (all 82 cases incl. pilots; "
               "plurality = single most frequent of the ten labels, tie = false):"]
    for label, sub in (("all 82", case_df), ("80 non-pilot", case_df[~case_df.pilot])):
        pc.append(f"  {label}: hit@1 {int(sub.top1_correct.sum())}, hit@5 {int(sub.hit5.sum())}, "
                  f"hit@10 {int(sub.hit10.sum())}, majority10 {int(sub.majority10_correct.sum())}, "
                  f"plurality ties {int((sub.plurality10_label == '').sum())}, "
                  f"cases with <10 neighbours {int((sub.n_neighbours < 10).sum())}")
    pc.append(f"  library slides in slides_index.csv: {int((ex['slides_index'].kind == 'library').sum())}; "
              f"manifest method.library.n_slides values: {sorted(set(case_df.manifest_library_n_slides))}")
    (out / "population_check.txt").write_text("\n".join(pc) + "\n")

    readings, ev = build_readings(ex, analysed, case_df, llm_df, entity_map)
    readings.to_csv(out / "readings.csv", index=False)
    ev.to_csv(out / "events_engagement.csv", index=False)
    (out / "llm_parse_review.md").write_text("\n".join(llm_review(case_df, llm_df)) + "\n")

    finished = dt.datetime.now(CDT)
    outputs = {p.name: sha256_file(p) for p in sorted(out.iterdir())
               if p.name in ("hash_check.txt", "population_check.txt", "readings.csv",
                             "events_engagement.csv", "llm_parse_review.md")}
    rr = ["# RUN_RECORD (generated section)", "",
          f"- started: {started.isoformat(timespec='seconds')}; finished: {finished.isoformat(timespec='seconds')}",
          f"- export: `{export}`",
          f"- sha256(SHA256SUMS): {sha256_file(export / 'SHA256SUMS')}",
          f"- sha256(db/study.sqlite): {sha256_file(export / 'db' / 'study.sqlite')}",
          f"- material root: `{args.data}`",
          f"- entity map: `{args.entity_map}` sha256 {sha256_file(args.entity_map)}",
          f"- readers analysed: {analysed} (tiers {[d.EXPERTISE_TIER[r] for r in analysed]})",
          f"- script commit (HEAD): {git['commit']}; scripts/reader_study clean: {git['scripts_reader_study_clean']}",
          f"- python: {platform.python_version()} ({sys.executable}); pandas {pd.__version__}",
          f"- command: `{' '.join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:]))}`",
          f"- readings.csv rows: {len(readings)}; events_engagement.csv rows: {len(ev)}",
          "- output sha256:"] + [f"  - {k}: {v}" for k, v in outputs.items()]
    (out / "RUN_RECORD.generated.md").write_text("\n".join(rr) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
