"""S5d stage S0 — freeze the 80 reader-study summaries and their inputs. KEY-SIDE SCRIPT.

Plan 001 §3/§4 S0 and plan 002 §A1/§B3. Reads, read-only: the 14 November export's cases.json, production LLM
hash list and submissions_export.csv (columns role/phase/case_id/slide_id/a_material_sha256/reader_id only; the
reader_id is used only to count readings per reader and is never printed or stored; comment columns are not
read); the 8 November export's cases.json; the served summaries reader_study_suite/data/llm/*.txt; the
generating runs st1_llm_summaries_v5_{ccdi,stv}/ (runs.jsonl, payloads/, packet/, queries.csv).
Writes only <OUT_ROOT>/inputs/ (truth-free) and <OUT_ROOT>/key/ (truth). Refuses to overwrite. Aborts (non-zero
exit, nothing written) if any population assertion fails.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

import reader80_common as C  # noqa: E402

TRUTH_ABSENT_PLAN = ["case040", "case053", "case057", "case067", "case073"]


def main() -> int:
    inp, key = C.OUT_ROOT / "inputs", C.OUT_ROOT / "key"
    for d in (inp, key):
        if d.exists():
            raise SystemExit(f"{d} exists; refusing to overwrite")
    checks, fails = {}, []

    def check(name, ok, detail):
        checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            fails.append(name)

    # ------------------------------------------------------------ cases.json (14 Nov vs 8 Nov)
    cj = C.EXPORT / "study_config" / "cases.json"
    cj8 = C.EXPORT_1108 / "study_config" / "cases.json"
    check("cases_json_8nov_equals_14nov", cj.read_bytes() == cj8.read_bytes(),
          {"sha256_14nov": C.fsha(cj), "sha256_8nov": C.fsha(cj8)})
    cases = json.loads(cj.read_text())
    practice = [c for c in cases if c.get("pilot")]
    study = [c for c in cases if not c.get("pilot")]
    check("cases_82_with_2_practice", len(cases) == 82 and sorted(c["case_id"] for c in practice) == list(C.PRACTICE_CASES),
          {"n_entries": len(cases), "practice": sorted(c["case_id"] for c in practice)})
    coh = Counter(c["truth"]["cohort"] for c in study)
    check("study_80_case001_case080_40_40", len(study) == 80 and
          sorted(c["case_id"] for c in study) == [f"case{i:03d}" for i in range(1, 81)] and
          coh == Counter({"CCDI": 40, "STV": 40}) and len({c["slide_id"] for c in study}) == 80,
          {"n_study": len(study), "cohorts": dict(coh)})
    all_slides = {c["slide_id"] for c in cases}

    # ------------------------------------------------------------ served files and production hash list
    served = {p.stem: p for p in C.SERVED_DIR.glob("*.txt")}
    check("served_files_82_equal_case_slides", set(served) == all_slides, {"n_served": len(served)})
    prod = {}
    for line in (C.EXPORT / "prod_hashes" / "user_llm.sha").read_text().splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            prod[Path(name.strip()).stem] = h
    n_prod_eq = sum(prod.get(s) == C.fsha(p) for s, p in served.items())
    check("prod_hash_list_82_of_82", n_prod_eq == 82 and len(prod) == 82,
          {"n_hash_list": len(prod), "n_equal": n_prod_eq, "hash_list_sha256": C.fsha(C.EXPORT / "prod_hashes" / "user_llm.sha")})

    # ------------------------------------------------------------ Step-3 material hashes (read only, ids not kept)
    sub = pd.read_csv(C.EXPORT / "submissions_export.csv", dtype=str, keep_default_na=False,
                      usecols=["reader_id", "role", "case_id", "slide_id", "phase", "a_material_sha256"])
    study_ids = {c["case_id"]: c["slide_id"] for c in study}
    s3 = sub[(sub.phase == "C") & (sub.role == "pathologist") & sub.case_id.isin(study_ids)]
    match = sum(m == C.fsha(served[s])[:16] and study_ids[cid] == s
                for cid, s, m in zip(s3.case_id, s3.slide_id, s3.a_material_sha256) if s in served)
    missing = int((~s3.slide_id.isin(list(served))).sum())
    per_reader = sorted(s3.groupby("reader_id").size().tolist(), reverse=True)
    check("step3_material_319_of_319", len(s3) == 319 and match == 319 and missing == 0 and
          per_reader == [80, 80, 80, 73, 6],
          {"n_rows": len(s3), "n_match": int(match), "n_missing": missing,
           "per_reader_counts_sorted_desc": per_reader,
           "submissions_export_sha256": C.fsha(C.EXPORT / "submissions_export.csv"),
           "note": "reader_id used only to count; not stored; comment columns not read"})
    del sub, s3

    # ------------------------------------------------------------ generating runs
    runs, run_src = {}, {}
    queries_csv = {}
    for coh_key, d in C.GEN_RUNS.items():
        for i, line in enumerate(open(d / "runs.jsonl"), 1):
            r = json.loads(line)
            if r["arm"] != "packet":
                continue
            assert r["query"] not in runs, ("duplicate packet row", r["query"])
            runs[r["query"]] = r
            run_src[r["query"]] = (coh_key, d / "runs.jsonl", i)
        for row in csv.DictReader(open(d / "queries.csv")):
            queries_csv[row["slide_id"]] = row["tumor_group"]
    check("generating_runs_one_packet_row_per_served_slide", set(runs) == all_slides,
          {"n_packet_rows": len(runs), "runs_sha256": {k: C.fsha(d / "runs.jsonl") for k, d in C.GEN_RUNS.items()}})
    n_md = sum((C.GEN_RUNS[run_src[s][0]] / "packet" / f"{s}.md").read_bytes() == served[s].read_bytes()
               for s in all_slides)
    n_txt = sum(runs[s]["text"].encode("utf-8") == served[s].read_bytes() for s in all_slides)
    check("generating_packet_md_82_byte_identical", n_md == 82, {"n_equal": n_md})
    check("runs_text_field_82_byte_identical", n_txt == 82, {"n_equal": n_txt})
    payloads, n_psha = {}, 0
    for s in all_slides:
        p = C.GEN_RUNS[run_src[s][0]] / "payloads" / f"{s}_packet.json"
        payloads[s] = p
        n_psha += C.payload_sha12(json.loads(p.read_text())) == runs[s]["payload_sha"]
    check("payload_file_equals_run_payload_sha_82", n_psha == 82, {"n_equal": n_psha,
          "definition": "sha256(json.dumps(payload, sort_keys=True))[:12] == runs.jsonl payload_sha"})
    study80 = sorted(study, key=lambda c: c["case_id"])
    n_ok = sum(bool(runs[c["slide_id"]]["accepted"]) and not runs[c["slide_id"]]["error"] and bool(runs[c["slide_id"]]["text"])
               for c in study80)
    check("study_80_accepted_no_error_text_present", n_ok == 80, {"n_ok": n_ok})
    check("study_80_texts_equal_served_and_hash_list",
          all(runs[c["slide_id"]]["text"].encode() == served[c["slide_id"]].read_bytes() and
              prod[c["slide_id"]] == C.fsha(served[c["slide_id"]]) for c in study80), {"n": len(study80)})

    # ------------------------------------------------------------ truth (key side) and truth-absent set
    H = C.load_harness()
    tr_rows, tr_fail, absent_plain, absent_alias = [], [], [], []
    for c in study80:
        s, tg = c["slide_id"], c["truth"]["tumor_group"]
        pl = json.loads(payloads[s].read_text())
        if not (tg == runs[s]["true_group"] == queries_csv.get(s)):
            tr_fail.append(c["case_id"])
        in_plain = tg in ({n.get("tumor_group") for n in pl["neighbors"]} | set(pl.get("probabilities") or {}))
        in_alias = H.group_in_labels(tg, pl)
        if not in_plain:
            absent_plain.append(c["case_id"])
        if not in_alias:
            absent_alias.append(c["case_id"])
        tr_rows.append({"case_id": c["case_id"], "slide_id": s, "cohort": C.COHORT_NAME[c["truth"]["cohort"]],
                        "true_group": tg, "comet_diagnosis": c["truth"].get("comet_diagnosis", ""),
                        "runs_true_group": runs[s]["true_group"], "queries_csv_tumor_group": queries_csv.get(s, ""),
                        "true_in_neighbour_groups_or_prob_keys": in_plain, "true_in_payload_labels_alias": in_alias,
                        "leaks_true_group_stored": runs[s]["audit"]["leaks_true_group"]})
    check("truth_cases_json_eq_runs_eq_queries_csv", not tr_fail, {"mismatch": tr_fail})
    check("truth_absent_set_equals_plan_001", absent_plain == TRUTH_ABSENT_PLAN and absent_alias == TRUTH_ABSENT_PLAN,
          {"plain_neighbour_groups_or_prob_keys": absent_plain, "alias_aware_label_fields": absent_alias,
           "plan_001": TRUTH_ABSENT_PLAN})

    if fails:
        print(json.dumps(checks, indent=1, default=str))
        print("ABORT: failed assertions:", fails)
        return 2

    # ------------------------------------------------------------ informational: auditor identity, derived stats
    audit_rep, n_audit_eq = [], 0
    for c in study80:
        r = runs[c["slide_id"]]
        pl = json.loads(payloads[c["slide_id"]].read_text())
        au = H.audit(r["text"], r["telemetry"].get("claims") or [], pl, c["truth"]["tumor_group"])
        diff = sorted(k for k in set(au) | set(r["audit"]) if au.get(k) != r["audit"].get(k))
        n_audit_eq += not diff
        audit_rep.append({"case_id": c["case_id"], "slide_id": c["slide_id"], "fields_differing": diff,
                          **({"stored": {k: r["audit"].get(k) for k in diff}, "recomputed": {k: au.get(k) for k in diff}}
                             if diff else {})})
    derive, derive_src_sha = C.round1_derive_stats_fn()
    derived = {}
    for c in study80:
        d = derive(json.loads(payloads[c["slide_id"]].read_text()))[0]
        derived[c["slide_id"]] = d

    # ------------------------------------------------------------ write inputs/ (truth-free) and key/
    inp.mkdir(parents=True)
    key.mkdir(parents=True)
    (inp / "texts").mkdir()
    (inp / "payloads").mkdir()
    man = []
    with open(inp / "runs_reader80_truthstripped.jsonl", "w") as fh:
        for c in study80:
            s = c["slide_id"]
            r = json.loads(json.dumps(runs[s]))
            r.pop("true_group")
            r["audit"].pop("leaks_true_group")
            fh.write(json.dumps({"case_id": c["case_id"], "slide_id": s, "run": r}, ensure_ascii=False) + "\n")
            shutil.copyfile(served[s], inp / "texts" / f"{s}.txt")
            shutil.copyfile(payloads[s], inp / "payloads" / f"{s}_packet.json")
            assert C.fsha(inp / "texts" / f"{s}.txt") == C.fsha(served[s])
            assert C.fsha(inp / "payloads" / f"{s}_packet.json") == C.fsha(payloads[s])
            t = runs[s]["telemetry"]
            coh_key, rpath, ln = run_src[s]
            man.append({"case_id": c["case_id"], "slide_id": s, "cohort": C.COHORT_NAME[c["truth"]["cohort"]],
                        "text_sha256": C.sha(runs[s]["text"].encode()), "served_file_sha256": C.fsha(served[s]),
                        "prod_hash_list_sha256": prod[s],
                        "payload_file": str(payloads[s]), "payload_file_sha256": C.fsha(payloads[s]),
                        "payload_sha12_runs": runs[s]["payload_sha"],
                        "generating_run": f"{rpath}:{ln}", "generating_runs_sha256": C.fsha(rpath),
                        "revision_applied": bool(t.get("revision_applied")),
                        "text_first_differs": bool(t.get("text_first")) and t.get("text_first") != runs[s]["text"],
                        "word_count": len(runs[s]["text"].split()),
                        "n_sentences_round1_splitter": len(C.split_sentences(runs[s]["text"]))})
    pd.DataFrame(man).to_csv(inp / "manifest_reader80.csv", index=False)
    (inp / "derived_stats_round1_recomputed.json").write_text(json.dumps(
        {"method": "frozen llm.py.round1 `_derive_stats_from_payload` source extracted with ast and executed on each "
                   "payload file (no request record exists for these runs; recomputation, not the record)",
         "llm_py_round1_sha256": C.ROUND1_LLM_SHA, "function_source_sha256": derive_src_sha,
         "derived_stats": derived}, indent=1, default=str) + "\n")
    population = {
        "plan": "plan 001 §1 (8 November export) and plan 002 §A1 (14 November export), re-run here",
        "export_14nov": str(C.EXPORT), "export_8nov_cases_json": str(cj8),
        "plan_001_8nov_checks": {"prod_hash_list": "82/82 sha256 equal", "step3_material": "226/226 (80/80/45/15/6)",
                                 "packet_md_identity": "82/82", "truth_absent": TRUTH_ABSENT_PLAN},
        "plan_002_14nov_checks_reported": {"prod_hash_list": "82/82", "cases_json_8nov_vs_14nov": "identical",
                                           "step3_material": "319/319 (80/80/73/6/80)",
                                           "packet_md_identity": "82/82", "study_slides_with_summary": "80/80"},
        "assertions_rerun_here": checks,
        "informational_auditor_identity": {
            "harness": str(C.HARNESS), "harness_sha256": C.fsha(C.HARNESS),
            "n_audit_equal_80": n_audit_eq,
            "note": "H.audit(text, claims, payload, truth) recomputed against the stored audit; the detailed "
                    "per-item comparison is key-side (key/audit_reproduction.json) because leaks_true_group needs truth"},
    }
    (inp / "population_checks.json").write_text(json.dumps(population, indent=1, default=str) + "\n")
    pd.DataFrame(tr_rows).to_csv(key / "scoring_key_DO_NOT_SEND.csv", index=False)
    (key / "audit_reproduction.json").write_text(json.dumps(
        {"harness_sha256": C.fsha(C.HARNESS), "n_equal": n_audit_eq, "items": audit_rep}, indent=1, default=str) + "\n")
    leak = [{"case_id": c["case_id"], "slide_id": c["slide_id"], "true_group": c["truth"]["tumor_group"],
             "neighbour_groups": sorted({n.get("tumor_group") for n in json.loads(payloads[c["slide_id"]].read_text())["neighbors"]} - {None}),
             "prob_keys": sorted(json.loads(payloads[c["slide_id"]].read_text()).get("probabilities") or {}),
             "true_group_mention": H.mention_test(c["truth"]["tumor_group"],
                                                  {"text": runs[c["slide_id"]]["text"],
                                                   "text_first": runs[c["slide_id"]]["telemetry"].get("text_first")},
                                                  json.loads(payloads[c["slide_id"]].read_text()))}
            for c in study80 if runs[c["slide_id"]]["audit"]["leaks_true_group"]]
    (key / "leaks_true_group_detail.json").write_text(json.dumps(leak, indent=1, default=str) + "\n")
    n_inp = C.write_sha_list(inp)
    n_key = C.write_sha_list(key)
    print(json.dumps({"assertions": {k: v["pass"] for k, v in checks.items()}, "n_audit_equal": n_audit_eq,
                      "inputs_files": n_inp, "key_files": n_key, "leak_candidates": [x["case_id"] for x in leak]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
