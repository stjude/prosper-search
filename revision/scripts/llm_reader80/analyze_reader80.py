"""S5d key-side analysis: arm-less, single-rater port of `scripts/llm_pilot_round2/analyze_r2.py` (823de3fa...).

Plan: [internal record] §4 S4/S5, §5, as amended by
..._candidate_002_20261115.md. KEY-SIDE SCRIPT. Read-only on every input; writes only to --out-dir.

Two modes, one code path for everything after loading:
  --mode round2-readback   loads round 2's A0 run exactly as analyze_r2 did (runs.jsonl selected rows, payload and
                           response files, key/endpoints_auto.jsonl, harness reproduction incl. audit_first and the
                           raw-call channel) and joins rater 1's form only. Used for the fidelity readback against
                           s5c_pilot_round2_20261110/analysis/join/ (see compare_readback_round2.py).
  --mode reader80          loads the S0 freeze (inputs/runs_reader80_truthstripped.jsonl, inputs/payloads/,
                           key/scoring_key_DO_NOT_SEND.csv) and joins the eight packet forms (one rater).

Ported unchanged from analyze_r2: harness reproduction of audit(); independent citation check; flag generation (types,
values, sources, the "first(=final)" rule, occurrence numbering); external flag classes (analyst's JSON; the script
never classifies); summary_flags(); zero-flag counts raw and after alpha/beta/gamma; probability-statement parser
and checker; rated_endpoints(); strict-"discrepant" recode from the adjudication JSON; recode_sensitivities.

Differences (plan 001 §4-§5; all arm-less, k = 10, single rater):
* one rater; no agreement rows; the round-2 example rules are computed only in readback mode (R1 round-1 rule).
* reader80 has no raw-response channel, no audit_first and no run-record fabricated-ID / query-ID fields: sources
  are text and text_first; fabricated IDs are computed with the harness function at analysis time; the query's own
  ID is given unmasked in the payload, so query-ID appearance is reported descriptively, never as a flag.
* reader80 true-group leak candidate = stored audit `leaks_true_group` (substring auditor; key-side column) OR the
  harness mention_test leak candidate; the flag value never names the group.
* SD-1/SD-2 (plan 001 §4 S4), mechanical, computed after the join on rows coded unsupported/speculative-diagnostic.
* strict "discrepant" rows are also computed mechanically (round-2 rule of strict_discrepant_rows_r2.py) and unioned
  with any rows listed in the adjudication JSON; ambiguous sentences are listed for the analyst, not recoded.
* abbreviation-expansion sensitivity for "uncertain" (round 2's alias_accepted_posthoc, made mechanical here).
* plan 001 §5 endpoint table overall and by cohort; no met / not-met verdict.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
from scipy.stats import beta  # noqa: E402

import reader80_common as C  # noqa: E402

OK_CODES = {"supported", "derivable"}
CODES = ["supported", "derivable", "speculative-diagnostic", "unsupported", "contradicted", "uncertain"]
US_CODES = ["unsupported", "speculative-diagnostic"]
CP_LABEL = "nominal model-based interval under an independent common-probability assumption (AF1); per-arm only"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return [round(float(100 * lo), 1), round(float(100 * hi), 1)]


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["round2-readback", "reader80"], required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--harness", type=Path, default=C.HARNESS)
    ap.add_argument("--adjudication", type=Path, default=None, help="external rulings JSON ({} allowed)")
    ap.add_argument("--flags-only", action="store_true")
    # round2-readback
    ap.add_argument("--run-dir", type=Path, default=C.ROUND2 / "run")
    ap.add_argument("--arm", default="a0_masked_posterior")
    ap.add_argument("--ratings-r1", type=Path, default=None)
    ap.add_argument("--rating-map", type=Path, default=None)
    ap.add_argument("--scoring-key", type=Path, default=None)
    # reader80
    ap.add_argument("--root", type=Path, default=C.OUT_ROOT, help="S5d output root (inputs/, key/, packets)")
    ap.add_argument("--forms", nargs="+", action="extend", default=[], help="LETTER=path/to/filled rating_form.csv, one per packet")
    a = ap.parse_args()
    if not a.flags_only:
        if a.mode == "round2-readback" and not (a.ratings_r1 and a.rating_map):
            ap.error("--ratings-r1 and --rating-map are required in round2-readback mode unless --flags-only")
        if a.mode == "reader80" and len(a.forms) != len(C.PACKET_LETTERS):
            ap.error("reader80 join needs one --forms LETTER=path per packet")
    return a


a = parse()
OUT = a.out_dir
OUT.mkdir(parents=True, exist_ok=True)
ADJ = json.loads(a.adjudication.read_text()) if a.adjudication else {}
H = C.load_harness()
READBACK = a.mode == "round2-readback"
input_hashes = {str(a.harness): sha(a.harness)}
if a.adjudication:
    input_hashes[str(a.adjudication)] = sha(a.adjudication)

# ================================================================ loaders -> sel, ep, payload, sources
if READBACK:
    RUN, ARM = a.run_dir, a.arm
    S_ARM = {"a0_masked_posterior": "A0", "a1_opaque": "A1", "a2_cf_plurality": "A2", "repeat": "R"}[ARM]
    for p in [RUN / "runs.jsonl", RUN / "key/endpoints_auto.jsonl", RUN / "meta.json", RUN / "audit_summary.json"]:
        input_hashes[str(p)] = sha(p)
    rows = [json.loads(l) for l in open(RUN / "runs.jsonl")]
    sel = {r["run_uid"]: r for r in rows if r["selected"] and r["arm"] == ARM and r["repeat"] == 0}
    calls_by_uid = defaultdict(list)
    for r in rows:
        if r["run_uid"] in sel:
            calls_by_uid[r["run_uid"]].append(r)
    ep = {}
    for e in map(json.loads, open(RUN / "key/endpoints_auto.jsonl")):
        u = f"{e['query']}_{e['arm']}_{e['repeat']}"
        if u in sel:
            ep[u] = e
    assert set(ep) == set(sel), "endpoint records != selected summaries"
    payload = {u: json.load(open(RUN / sel[u]["payload_file"])) for u in sel}
    resp = {u: json.load(open(RUN / sel[u]["response_file"])) for u in sel}
    COHORT = {u: "all" for u in sel}
    if a.scoring_key:
        input_hashes[str(a.scoring_key)] = sha(a.scoring_key)
        sk = pd.read_csv(a.scoring_key, dtype=str).set_index("query")
        assert all(sk.loc[sel[u]["query"], "true_group"] == ep[u]["true_group"] for u in sel)

    def sources(u):
        r = sel[u]
        s = {"text": r["text"], "text_first": r["text_first"]}
        for c in resp[u]["calls"]:
            s[f"raw_call_{c['call_ordinal']}"] = c.get("raw_content")
        return s
else:
    ROOT = a.root
    S_ARM = "packet"
    runs_path = ROOT / "inputs" / "runs_reader80_truthstripped.jsonl"
    key_path = ROOT / "key" / "scoring_key_DO_NOT_SEND.csv"
    for p in [runs_path, key_path, ROOT / "inputs" / "manifest_reader80.csv"]:
        input_hashes[str(p)] = sha(p)
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False).set_index("case_id")
    sel, payload, ep, COHORT = {}, {}, {}, {}
    for line in open(runs_path):
        j = json.loads(line)
        r, cid, sid = j["run"], j["case_id"], j["slide_id"]
        u = f"{sid}_packet_0"
        t = r["telemetry"] or {}
        pl_path = ROOT / "inputs" / "payloads" / f"{sid}_packet.json"
        pl = json.loads(pl_path.read_text())
        assert C.payload_sha12(pl) == r["payload_sha"], cid
        text_first = t.get("text_first") if t.get("revision_applied") else None
        state = "ok" if (r["accepted"] and not r["error"] and r["text"]) else ("error" if r["error"] else "rejected")
        sel[u] = {"run_uid": u, "query": cid, "slide_id": sid, "text": r["text"], "text_first": text_first,
                  "telemetry": t, "audit": r["audit"], "audit_first": None, "output_state": state,
                  "n_calls": t.get("calls")}
        payload[u] = pl
        COHORT[u] = key.loc[cid, "cohort"]
        assert key.loc[cid, "slide_id"] == sid
        tg = key.loc[cid, "true_group"]
        srcs = {"text": r["text"], "text_first": text_first}
        ep[u] = {"true_group": tg, "native_id": sid, "patient_stem": H.patient_stem(sid),
                 "true_in_payload_labels": H.group_in_labels(tg, pl),
                 "true_group_mention": H.mention_test(tg, srcs, pl),
                 "leaks_true_group_stored": key.loc[cid, "leaks_true_group_stored"] == "True"}
        fab = H.fabricated_ids(srcs, pl, sid)
        sel[u].update({k: fab[k] for k in ("fabricated_ids", "n_fabricated_ids", "query_id_leak",
                                            "query_id_substring_sources")})
    assert len(sel) == C.N_ITEMS
    calls_by_uid = {}

    def sources(u):
        return {"text": sel[u]["text"], "text_first": sel[u]["text_first"]}


def qkey(q):
    return (0, int(q), "") if str(q).isdigit() else (1, 0, str(q))


QUERIES = sorted({r["query"] for r in sel.values()}, key=qkey)
UID_OF = {sel[u]["query"]: u for u in sel}


def uid(q):
    return UID_OF[q]


def kq(u):
    e = ep[u]
    return {"true_group": e["true_group"], "native_id": e["native_id"], "patient_stem": e["patient_stem"]}


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def sentences(t):
    return [s for s in _SENT.split(t or "") if s.strip()]


# ================================================================ automatic layer: harness reproduction
recomp, diffs = {}, []
for u, r in sel.items():
    pl = payload[u]
    au = H.audit(r["text"], r["telemetry"].get("claims") or [], pl, kq(u)["true_group"])
    leak_re = au.pop("leaks_true_group")
    plur = (au.pop("plurality"), au.pop("plurality_tied"))
    stored = dict(r["audit"])
    for k in ("plurality", "plurality_tied"):
        stored.pop(k, None)
    stored.pop("leaks_true_group", None)
    if au != stored:
        diffs.append({"uid": u, "field": [k for k in au if au.get(k) != stored.get(k)]})
    if not READBACK and leak_re != ep[u]["leaks_true_group_stored"]:
        diffs.append({"uid": u, "field": "leaks_true_group"})
    if r["text_first"] and r["audit_first"] is not None:
        af = H.audit(r["text_first"], [], pl, kq(u)["true_group"])
        af.pop("leaks_true_group"); af.pop("plurality"); af.pop("plurality_tied")
        sf = dict(r["audit_first"]); sf.pop("plurality", None); sf.pop("plurality_tied", None)
        if af != sf:
            diffs.append({"uid": u, "source": "first", "field": [k for k in af if af.get(k) != sf.get(k)]})
    if READBACK:
        fab = H.fabricated_ids(sources(u), pl, kq(u)["native_id"])
        for k in ("fabricated_ids", "n_fabricated_ids", "query_id_leak", "query_id_substring_sources"):
            if fab[k] != r[k]:
                diffs.append({"uid": u, "field": k})
        mt = H.mention_test(kq(u)["true_group"], sources(u), pl)
        if mt != ep[u]["true_group_mention"]:
            diffs.append({"uid": u, "field": "true_group_mention"})
    recomp[u] = {"plurality": plur}
assert not diffs, diffs

_CIT = re.compile(r"\[([^\[\]]+)\]")
indep = {}
for u, r in sel.items():
    ids = {n["case_id"] for n in payload[u]["neighbors"]} | {payload[u]["current_case"]["comp_bio_id"]}
    bad, ncit = [], 0
    for src in ("text", "text_first"):
        for grp in _CIT.findall(r[src] or ""):
            for tok in re.split(r"[,;\s]+", grp.strip()):
                if tok:
                    ncit += src == "text"
                    if tok not in ids:
                        bad.append((src, tok))
    nat, stem = kq(u)["native_id"], kq(u)["patient_stem"]
    own = [s for s, t in sources(u).items() if t and (nat in t or stem in t)]
    selfcite = [g for g in _CIT.findall(r["text"] or "") if payload[u]["current_case"]["comp_bio_id"] in g]
    indep[u] = {"bad_citation_tokens": bad, "n_citation_tokens_final": ncit, "own_native_or_stem": own,
                "self_citations": selfcite}

# ================================================================ flags (generation only; classes are external)
_m2 = re.compile(r"Claim \d+: diagnosis '(.+) \((\d+) of (\d+)\)' not found in payload")
flag_rows = []
_occ = Counter()


def addflag(u, src, ftype, value, note):
    k = (u, src, ftype, str(value))
    _occ[k] += 1
    flag_rows.append({"run_uid": u, "arm": S_ARM, "query": sel[u]["query"], "source": src, "flag_type": ftype,
                      "value": str(value), "occurrence": _occ[k], "note": note})


for u, r in sel.items():
    for src, au in (("final", r["audit"]), ("first", r["audit_first"])):
        if au is None:
            continue
        if src == "first" and r["text_first"] == r["text"] and au == r["audit"]:
            src = "first(=final)"
        for x in au["unsupported_numbers"]:
            addflag(u, src, "unsupported_numbers", x, "")
        assert au["n_unsupported_numbers"] == len(au["unsupported_numbers"])
        if au["n_bad_citations"]:
            addflag(u, src, "bad_citation", ",".join(au["bad_citations"]), "")
        if au["dominant_statement_consistent"] is False:
            addflag(u, src, "dominant_statement_inconsistent", "plurality=" + str(recomp[u]["plurality"][0])
                    + " tied=" + "/".join(recomp[u]["plurality"][1]), "")
        if au["n_claims_contradicting_record"]:
            addflag(u, src, "claim_contradicting_record", str(au["n_claims_contradicting_record"]), "")
    for w in r["telemetry"].get("grounding_warnings") or []:
        m1 = re.match(r"Unverified number in output: (\S+)", w)
        m2 = _m2.match(w)
        if m1:
            addflag(u, "final", "grounding_unverified_number", m1.group(1), w)
        elif m2:
            addflag(u, "final", "grounding_claim_not_found", f"{m2.group(1)} ({m2.group(2)} of {m2.group(3)})", w)
        else:
            addflag(u, "final", "grounding_other", w, w)
    if r["n_fabricated_ids"]:
        addflag(u, "any", "fabricated_id", ",".join(r["fabricated_ids"]), "" if READBACK else
                "computed with the harness function at analysis time (no run-record field)")
    if READBACK and (r["query_id_leak"] or r["query_id_substring_sources"]):
        addflag(u, "any", "query_id_leak_or_substring", str(r["query_id_substring_sources"]), "")
    if READBACK:
        if ep[u]["true_group_mention"]["leak_candidate"]:
            hits = ep[u]["true_group_mention"]["hits"]
            addflag(u, ",".join(sorted({h["source"] for h in hits})), "true_group_leak_candidate",
                    ep[u]["true_group"] + ": " + "/".join(sorted({h["match"] for h in hits})), "")
    elif ep[u]["leaks_true_group_stored"] or ep[u]["true_group_mention"]["leak_candidate"]:
        addflag(u, "final", "true_group_leak_candidate",
                f"audit.leaks_true_group={ep[u]['leaks_true_group_stored']}; "
                f"mention_test.leak_candidate={ep[u]['true_group_mention']['leak_candidate']}",
                "group name withheld here (key/leaks_true_group_detail.json)")
    if r["output_state"] != "ok":
        addflag(u, "final", "output_state", r["output_state"], "")

cls_map = {(c["run_uid"], c["source"], c["flag_type"], str(c["value"]), int(c.get("occurrence", 1))): c
           for c in ADJ.get("flag_classes", [])}
assert all(c["class"] in ("alpha", "beta", "gamma", "raw") for c in cls_map.values())
used = set()
for f in flag_rows:
    k = (f["run_uid"], f["source"], f["flag_type"], f["value"], f["occurrence"])
    c = cls_map.get(k)
    used.add(k) if c else None
    f["class"] = c["class"] if c else "unclassified"
    f["payload_field"] = c.get("payload_field", "") if c else ""
    if c and c.get("note") is not None:
        f["note"] = c["note"]
    f["counts_after_alpha_gamma"] = f["class"] in ("raw", "unclassified")
unmatched_classes = sorted(set(cls_map) - used)
assert not unmatched_classes, f"external classes that match no generated flag: {unmatched_classes}"

per_summary = []
for u, r in sorted(sel.items(), key=lambda kv: qkey(kv[1]["query"])):
    t = r["telemetry"] or {}
    per_summary.append({
        "run_uid": u, "query": r["query"], "cohort": COHORT[u], "output_state": r["output_state"],
        "n_calls": r["n_calls"],
        "revision_applied": bool(t.get("revision_applied")), "revision_rejected": t.get("revision_rejected"),
        "n_grounding_warnings_final": len(t.get("grounding_warnings") or []),
        "n_style_warnings_first": len(t.get("style_warnings_first") or []),
        "n_style_warnings_final": len(t.get("style_warnings_final") or []),
        "n_fabricated_ids": r["n_fabricated_ids"], "query_id_leak": bool(r["query_id_leak"]),
        "query_id_substring_sources": len(r["query_id_substring_sources"] or []),
        "true_group_leak_candidate": any(f["run_uid"] == u and f["flag_type"] == "true_group_leak_candidate"
                                         for f in flag_rows),
        "n_flag_rows": sum(f["run_uid"] == u for f in flag_rows)})

FLAG_COLS = ["run_uid", "arm", "query", "source", "flag_type", "value", "occurrence", "class", "payload_field",
             "counts_after_alpha_gamma", "note"]
pd.DataFrame(flag_rows, columns=FLAG_COLS).to_csv(OUT / "flag_classification.csv", index=False)
if a.flags_only:
    out_cols = [c for c in FLAG_COLS if c not in ("class", "payload_field", "counts_after_alpha_gamma")]
    df = pd.DataFrame(flag_rows, columns=FLAG_COLS)[out_cols]
    df.insert(3, "slide_id", [sel[u].get("slide_id", "") for u in df.run_uid])
    df.insert(len(df.columns), "class_lead", "")
    name = "auto_flags_reader80.csv" if not READBACK else "auto_flags.csv"
    df.to_csv(OUT / name, index=False)
    pd.DataFrame(per_summary).to_csv(OUT / name.replace(".csv", "_per_summary.csv"), index=False)
    json.dump({"input_hashes": input_hashes, "script_sha256": sha(__file__), "n_flag_rows": len(flag_rows),
               "flag_types": dict(Counter(f["flag_type"] for f in flag_rows)),
               "harness_reproduction_diffs": diffs,
               "sources_available": "text, text_first" if not READBACK else "text, text_first, raw calls",
               "not_available_reader80": ["audit_first (first-draft auditor record)", "raw responses",
                                          "run-record fabricated-ID / query-ID fields"] if not READBACK else []},
              open(OUT / "flags_only_meta.json", "w"), indent=1)
    print("flags-only:", len(flag_rows), "flag rows;", len(per_summary), "summaries;", dict(Counter(f["flag_type"] for f in flag_rows)))
    sys.exit(0)


def summary_flags(u, discount):
    fl = [f for f in flag_rows if f["run_uid"] == u and f["source"] in ("final", "first(=final)", "any")
          or (f["run_uid"] == u and f["flag_type"] in ("true_group_leak_candidate", "a2_original_plurality_mention"))]
    return [f for f in fl if (not discount) or f["counts_after_alpha_gamma"]]


# ================================================================ probability statements
_alias_pat = []
for g, als in H.ALIASES.items():
    for al in als + ([g.title()] if g in ("NORMAL", "MISC") else []):
        _alias_pat.append((al, g))
_alias_pat.sort(key=lambda x: -len(x[0]))


def class_spans(s, extra_keys=()):
    """Harness aliases (as analyze_r2) plus, when given, payload probability keys that the alias table lacks
    (e.g. DSRCT), matched case-sensitively as whole words (reader80 addition; no effect on round-2 readback)."""
    spans = []
    known = set(H.ALIASES)
    pats = _alias_pat + sorted([(k, k) for k in extra_keys if k not in known], key=lambda x: -len(x[0]))
    for al, g in pats:
        flags = 0 if re.fullmatch(r"[A-Z0-9/]+", al) else re.I
        for m in re.finditer(r"(?<![A-Za-z])" + re.escape(al) + r"(?:s|es)?(?![A-Za-z])", s, flags):
            if any(m.start() < e and st < m.end() for st, e, _ in spans):
                continue
            spans.append((m.start(), m.end(), g))
    return sorted(spans)


def sigfigs(s):
    d = s.replace(".", "").lstrip("0")
    return len(d) if d else 1


_PROB_SENT = re.compile(r"probabilit|bayesian|classifier|model assigns|model probabilit", re.I)
_EXCL = re.compile(r"similar|closest", re.I)


def prob_statements(text, keys=()):
    out = []
    for s in sentences(text):
        if not _PROB_SENT.search(s) or _EXCL.search(s):
            continue
        spans = class_spans(_CIT.sub(" ", s), keys)
        s2 = _CIT.sub(" ", s)
        # analyze_r2's alternation tries the plain decimal first, so "30.2%" was read as 30.2 (fraction); in reader80
        # mode the percent form is tried first (round-2 readback keeps the original order).
        num_re = r"(?<![\w.])(\d+\.\d+|\d+(?:\.\d+)?%)" if not keys else r"(?<![\w.])(\d+(?:\.\d+)?%|\d+\.\d+)"
        for m in re.finditer(num_re, s2):
            x = m.group(1)
            after = s2[m.end():m.end() + 12]
            fm = re.match(r"\s+for\s+", after) or (re.match(r"\s+(?:probability\s+)?to\s+", s2[m.end():m.end() + 20])
                                                   if keys else None)
            g = None
            if fm:
                nxt = [sp for sp in spans if sp[0] >= m.end()]
                g = nxt[0][2] if nxt else None
            if g is None and (not fm or keys):
                prev = [sp for sp in spans if sp[1] <= m.start()]
                g = prev[-1][2] if prev else None
                if g is None:
                    nxt = [sp for sp in spans if sp[0] >= m.end()]
                    g = nxt[0][2] if nxt else None
            out.append({"sentence": s, "value": x, "group": g})
    return out


def check_stmt(st, probs):
    x = st["value"]
    pct = x.endswith("%")
    xv = x.rstrip("%")
    dec = len(xv.split(".")[1]) if "." in xv else 0
    if st["group"] is None or st["group"] not in probs:
        return "incorrect_class"
    p = probs[st["group"]] * (100 if pct else 1)
    if abs(round(p, dec) - float(xv)) > 1e-9:
        return "incorrect_value"
    if sigfigs(xv) < 2:
        return "below_2sf"
    return "correct"


prob_detail_all = {}
for u in sel:
    r = sel[u]
    if r["output_state"] not in ("ok", "truncated") or not r["text"]:
        continue
    sts = prob_statements(r["text"], () if READBACK else tuple(payload[u].get("probabilities") or {}))
    prob_detail_all[u] = [{**s, "result": check_stmt(s, payload[u].get("probabilities") or {})} for s in sts]


def prob_counts(us):
    cnt = Counter()
    for u in us:
        r = sel[u]
        probs = payload[u].get("probabilities") or {}
        if u not in prob_detail_all:
            cnt["no_evaluable_text"] += 1
            continue
        det = prob_detail_all[u]
        qual = bool(re.search(r"probabilit|bayesian|classifier", r["text"], re.I))
        if not det:
            cnt["statement_absent"] += 1
            cnt["of_which_qualitative_only"] += qual
        else:
            cnt["with_statement"] += 1
            cnt["all_correct" if all(d["result"] == "correct" for d in det) else "with_incorrect"] += 1
            cnt["all_classes_quoted"] += set(probs) <= {d["group"] for d in det}
    cnt["n_nominal"] = len(us)
    return dict(cnt)


# ================================================================ automatic endpoints (any subset of summaries)
def auto_endpoints(us):
    return {
        "n": len(us),
        "bad_citation_harness": sum(sel[u]["audit"]["n_bad_citations"] > 0 or
                                    bool(sel[u]["audit_first"] and sel[u]["audit_first"]["n_bad_citations"]) for u in us),
        "bad_citation_independent": sum(bool(indep[u]["bad_citation_tokens"]) for u in us),
        "fabricated_ids_any_source": sum(sel[u]["n_fabricated_ids"] > 0 for u in us),
        "own_native_or_stem_any_source": sum(bool(indep[u]["own_native_or_stem"]) for u in us),
        "query_id_leak_harness": sum(bool(sel[u]["query_id_leak"] or sel[u]["query_id_substring_sources"]) for u in us),
        "self_citation_of_query": sum(bool(indep[u]["self_citations"]) for u in us),
        "non_ok_state": sum(sel[u]["output_state"] != "ok" for u in us),
        "states": dict(Counter(sel[u]["output_state"] for u in us)),
        "revision_applied": sum(bool(sel[u]["telemetry"].get("revision_applied")) for u in us),
        "revised_run_uids": [u for u in us if sel[u]["telemetry"].get("revision_applied")],
        "revised_queries": [sel[u]["query"] for u in us if sel[u]["telemetry"].get("revision_applied")],
        "first_draft_differs": [sel[u]["query"] for u in us if sel[u]["text_first"] and sel[u]["text_first"] != sel[u]["text"]],
        "claim_contradicting_record": sum(sel[u]["audit"]["n_claims_contradicting_record"] > 0 for u in us),
        "n_claims_returned": sum(sel[u]["audit"]["n_claims"] for u in us),
        "dominant_false_raw": sum(sel[u]["audit"]["dominant_statement_consistent"] is False for u in us),
        "rule12_similarity_mention": [sel[u]["query"] for u in us if re.search(
            r"similarit|distance", re.sub(r"similar to the current case", "", sel[u]["text"]), re.I)],
    }


def zero_flags(us):
    z = {mode: [sel[u]["query"] for u in us if not summary_flags(u, disc)]
         for mode, disc in (("raw", False), ("post_alpha_gamma", True))}
    after = {sel[u]["query"]: sorted({f["flag_type"] for f in summary_flags(u, True)}) for u in us if summary_flags(u, True)}
    return {k: len(v) for k, v in z.items()}, z, after


not_leak = set(ADJ.get("true_group_not_a_leak", []))


def tg_absent(us):
    elig = [u for u in us if not ep[u]["true_in_payload_labels"]]
    lex = [sel[u]["query"] for u in elig if ep[u]["true_group_mention"]["mentioned_any"]]
    return {"eligible": len(elig), "eligible_queries": [sel[u]["query"] for u in elig], "lexical_hits": lex,
            "after_adjudication": [q for q in lex if uid(q) not in not_leak]}


US_ALL = [uid(q) for q in QUERIES]
auto_end = auto_endpoints(US_ALL)
zero, zero_lists, flagged_after = zero_flags(US_ALL)
prob_acc = prob_counts(US_ALL)
tg_abs = tg_absent(US_ALL)

# ================================================================ rating join (one rater)
if READBACK:
    for p in (a.ratings_r1, a.rating_map):
        input_hashes[str(p)] = sha(p)
    idmap = pd.read_csv(a.rating_map, dtype=str, keep_default_na=False)
    idmap = idmap[idmap.arm == ARM]
    imap = idmap.set_index("rating_id")
    orig = pd.read_csv(a.ratings_r1, dtype=str, keep_default_na=False)
    orig = orig[orig.rating_id.isin(imap.index)].reset_index(drop=True)
    form_checks = {}
else:
    rmap_path = ROOT / "rating_packet_key" / "rating_id_map.csv"
    input_hashes[str(rmap_path)] = sha(rmap_path)
    imap = pd.read_csv(rmap_path, dtype=str, keep_default_na=False).set_index("rating_id")
    parts, form_checks = [], {}
    for spec in a.forms:
        letter, path = spec.split("=", 1)
        path = Path(path)
        input_hashes[str(path)] = sha(path)
        blank = pd.read_csv(ROOT / f"rating_packet_{letter}" / "rating_form.csv", dtype=str, keep_default_na=False)
        f = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(f.columns) == list(blank.columns), (letter, list(f.columns))
        assert f[["rating_id", "sentence_no", "sentence_text"]].equals(blank[["rating_id", "sentence_no", "sentence_text"]]), \
            f"packet {letter}: rows/sentences differ from the packet form"
        assert set(imap.loc[f.rating_id.unique(), "packet"]) == {letter}, letter
        bad_codes = sorted(set(f.code) - set(CODES))
        assert not bad_codes, (letter, bad_codes)
        assert (f.code != "").all(), (letter, "missing code")
        # The instructions said only "leave empty if not a probability statement" and fixed no tag value; contexts
        # wrote 'probability-statement', 'probability-statement (qualitative; no value stated)' or the substantive
        # code. Format-only normalisation: any non-empty tag other than 'N' = tagged ('Y'); raw values kept in
        # probability_statement_tag_raw and counted in form_checks.
        raw_tags = dict(Counter(f.probability_statement_tag))
        f["probability_statement_tag_raw"] = f.probability_statement_tag
        f["probability_statement_tag"] = ["" if t.strip() in ("", "N") else "Y" for t in f.probability_statement_tag]
        form_checks[letter] = {"sha256": sha(path), "rows": len(f), "bad_codes": bad_codes, "raw_tag_values": raw_tags,
                               "rows_equal_blank_form": True, "missing_codes": 0}
        parts.append(f)
    orig = pd.concat(parts, ignore_index=True)
    assert set(orig.rating_id) == set(imap.index) and len(orig.rating_id.unique()) == C.N_ITEMS


def join_form(orig_df, outname):
    out = orig_df.copy()
    out["arm"] = [imap.loc[x, "arm"] if "arm" in imap.columns else S_ARM for x in orig_df.rating_id]
    out["query"] = [imap.loc[x, "query"] if READBACK else imap.loc[x, "case_id"] for x in orig_df.rating_id]
    out["run_uid"] = [imap.loc[x, "run_uid"] for x in orig_df.rating_id]
    if not READBACK:
        out["packet"] = [imap.loc[x, "packet"] for x in orig_df.rating_id]
        out["cohort"] = [COHORT[x] for x in out.run_uid]
    out.to_csv(OUT / outname, index=False)
    chk = pd.read_csv(OUT / outname, dtype=str, keep_default_na=False)
    assert chk[list(orig_df.columns)].equals(orig_df)
    return chk


J = join_form(orig, "rating_join_r1.csv")
J["first_draft_row"] = J["sentence_no"].str.startswith("F")
assert set(J.run_uid) == set(sel)
for rid, g in J[~J.first_draft_row].groupby("rating_id"):
    u = imap.loc[rid, "run_uid"]
    assert re.sub(r"\s+", " ", " ".join(g.sentence_text)) == re.sub(r"\s+", " ", sel[u]["text"]).strip(), rid


def code_dist(df):
    d = df[~df.first_draft_row]
    out = {"n_sentences": len(d), "n_summaries": d.rating_id.nunique(),
           "codes": {c: int((d.code == c).sum()) for c in CODES},
           "prob_tag_Y": int((d.probability_statement_tag == "Y").sum())}
    d = df[df.first_draft_row]
    out_f = {"n_sentences": len(d), "items": sorted(d["query"].unique().tolist(), key=qkey),
             "codes": {c: int((d.code == c).sum()) for c in CODES}}
    return out, out_f


def rated_endpoints(df, recode=None):
    d = df[~df.first_draft_row].copy()
    if recode is not None:
        d["code"] = [recode.get((x, y), c) for x, y, c in zip(d.rating_id, d.sentence_no, d.code)]
    per = d.groupby("query").code.apply(list)
    us_mask = d.code.isin(US_CODES)
    us_sum = d[us_mask].groupby("query").size()
    return {"n_summaries": int(d.rating_id.nunique()), "n_sentences": len(d),
            "contradicted_sentences": int((d.code == "contradicted").sum()),
            "contradicted_summaries": sorted([q for q, cs in per.items() if "contradicted" in cs], key=qkey),
            "unsup_or_specdiag_sentences": int(us_mask.sum()),
            "unsup_or_specdiag_summaries": sorted(us_sum.index.tolist(), key=qkey),
            "unsup_or_specdiag_per_summary": {q: int(v) for q, v in us_sum.items()},
            "specdiag_sentences": int((d.code == "speculative-diagnostic").sum()),
            "unsupported_sentences": int((d.code == "unsupported").sum()),
            "uncertain_sentences": int((d.code == "uncertain").sum()),
            "uncertain_summaries": sorted(d[d.code == "uncertain"]["query"].unique().tolist(), key=qkey),
            "all_sentences_supported_or_derivable": sorted([q for q, cs in per.items()
                                                            if all(c in OK_CODES for c in cs)], key=qkey)}


# ---------------------------------------------------------------- strict "discrepant" (round-2 mechanical rule)
CLAUSE_SPLIT = re.compile(r";|,\s+(?:while|whereas|but)\b", re.I)
NEG = re.compile(r"\b(no|not|none|neither|nor|without)\b[^;]*?discrepan|concordant|"
                 r"discrepant\s*(?:=|==|:)\s*['\"]?N\b|discrepant\s+flag\s+(?:of\s+)?['\"]?N\b|flag\s+(?:of\s+)?['\"]?N\b",
                 re.I)
DISC = re.compile(r"discrepan", re.I)


def _word_re(term):
    flags = 0 if re.fullmatch(r"[A-Z0-9/]+", term) else re.I
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?:s|es)?(?![A-Za-z0-9])", flags)


def _resolve(clause, neighbors):
    by_id = [n for n in neighbors if str(n.get("case_id")) and str(n["case_id"]) in clause]
    if by_id:
        return by_id, "case_id"
    hit = []
    for n in neighbors:
        terms = {str(n.get("diagnosis") or ""), str(n.get("tumor_group") or "")} - {""}
        g = n.get("tumor_group")
        terms |= set(H.ALIASES.get(g, [])) if g else set()
        if any(_word_re(t).search(clause) for t in terms):
            hit.append(n)
    return hit, "label"


def strict_discrepant(df):
    out_rows, ambiguous, first_draft_strict, n_disc = [], [], [], 0
    for row in df.itertuples():
        s = row.sentence_text
        if not DISC.search(s):
            continue
        n_disc += 1
        nb = payload[row.run_uid]["neighbors"]
        buckets = []
        for clause in CLAUSE_SPLIT.split(s):
            if not DISC.search(clause):
                continue
            if NEG.search(clause):
                buckets.append("negated")
                continue
            hit, _ = _resolve(clause, nb)
            flags = {str(n.get("discrepant")).upper() == "Y" for n in hit}
            buckets.append("ambiguous" if not hit or len(flags) > 1 else ("flag_Y" if flags == {True} else "strict"))
        delivered = row.sentence_no.isdigit()
        if "ambiguous" in buckets:
            ambiguous.append({"rating_id": row.rating_id, "sentence_no": row.sentence_no, "query": row.query,
                              "delivered": delivered, "buckets": "/".join(buckets)})
        elif "strict" in buckets:
            (out_rows if delivered else first_draft_strict).append([row.rating_id, row.sentence_no])
    return {"sentences_with_discrepant": n_disc, "strict_rows": out_rows,
            "strict_first_draft_rows_not_listed": first_draft_strict, "ambiguous": ambiguous}


strict_mech = strict_discrepant(J)
strict_rows_used = sorted({(x, str(y)) for x, y in ADJ.get("strict_discrepant_rows", [])} |
                          (set() if READBACK else {(x, y) for x, y in strict_mech["strict_rows"]}))
strict_recode = {k: "contradicted" for k in strict_rows_used}

# ---------------------------------------------------------------- abbreviation-expansion sensitivity (mechanical)
EXPANSION_NOTE = re.compile(r"expan|spell|abbreviat|alias|full name|written out", re.I)
_SPELLED = [(al, g) for g, als in H.ALIASES.items() for al in als if not re.fullmatch(r"[A-Z0-9/]+", al)]


def expansion_rows(df):
    rows = []
    for row in df.itertuples():
        if row.code != "uncertain":
            continue
        why = f"{row.note} {row.support_trace}"
        named = sorted({g for al, g in _SPELLED if re.search(r"(?<![A-Za-z])" + re.escape(al) + r"(?:s|es)?(?![A-Za-z])",
                                                                 row.sentence_text, re.I)
                        and H.group_in_labels(g, payload[row.run_uid])})
        if EXPANSION_NOTE.search(why) and named:
            rows.append([row.rating_id, row.sentence_no])
    return rows


abbr_rows = expansion_rows(J)
abbr_recode = {(x, y): "derivable" for x, y in abbr_rows}

# ---------------------------------------------------------------- SD-1 / SD-2 (plan 001 §4 S4; mechanical)
SD1_RE = re.compile(
    r"\b(?:support(?:s|ed|ing)?|favou?r(?:s|ed|ing)?)\s+(?P<X>[^.;]{1,120}?)\s+as\s+(?:the\s+|a\s+)?"
    r"(?:leading|primary)\s+(?:considerations?|diagnos[ie]s)\b", re.I)
SD1_RULE = ("Fixed before rating (plan 001 §4 S4). A delivered sentence the rater coded unsupported or "
            "speculative-diagnostic is SD-1 iff (a) it matches /\\b(support(s|ed|ing)?|favou?r(s|ed|ing)?)\\s+X\\s+as\\s+"
            "(the |a )?(leading|primary) (consideration(s)|diagnosis|diagnoses)\\b/i with X a span of <=120 characters "
            "without '.' or ';', (b) X names >= 1 tumor group (harness aliases) or exact label value present in the "
            "payload label fields (neighbour tumor_group/diagnosis, probability keys; harness group_in_labels), and "
            "(c) no tumor group named anywhere in the sentence (harness aliases) is absent from those label fields. "
            "Everything else is SD-2. Plural 'as leading considerations' and the article-less form are accepted as the "
            "same form; 'X is the leading consideration' is not of the stated form and is SD-2.")


def _labels_in(x, pl):
    x2 = _CIT.sub(" ", x)
    gs = {g for _, _, g in class_spans(x2)}
    present = {g for g in gs if H.group_in_labels(g, pl)}
    absent = gs - present
    vals = {v for v in H.label_fields(pl) if v and re.search(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", x2, re.I)}
    return present | vals, absent


def sd_class(sentence, pl):
    m = SD1_RE.search(sentence)
    _, absent_all = _labels_in(sentence, pl)
    if not m:
        return "SD-2", None, sorted(absent_all)
    X = m.group("X")
    present, absent = _labels_in(X, pl)
    if present and not absent and not absent_all:
        return "SD-1", m.group(0), []
    return "SD-2", m.group(0), sorted(absent_all | absent)


def sd_rows(df):
    d = df[~df.first_draft_row & df.code.isin(US_CODES)]
    out = []
    for row in d.itertuples():
        c, span, absent = sd_class(row.sentence_text, payload[row.run_uid])
        out.append({"rating_id": row.rating_id, "sentence_no": row.sentence_no, "query": row.query,
                    "cohort": COHORT[row.run_uid], "code": row.code, "sd_class": c, "matched_span": span,
                    "groups_absent_from_input": absent})
    return out


SD = sd_rows(J)

# ---------------------------------------------------------------- leading-consideration statements (plan 001 §6 "can")
# Key-side, descriptive, independent of the rater's code: every delivered sentence of the stated SD-1 form (SD1_RE),
# and, secondarily, any other sentence with "leading/primary consideration(s)/diagnosis" phrasing.
LC_BROAD = re.compile(r"\b(?:leading|primary)\s+(?:considerations?|diagnos[ie]s)\b", re.I)
LC, LC_SUM = [], {}
if not READBACK:
    for row in J[~J.first_draft_row].itertuples():
        m = SD1_RE.search(row.sentence_text)
        mb = LC_BROAD.search(row.sentence_text)
        if not (m or mb):
            continue
        pl = payload[row.run_uid]
        keys = tuple(pl.get("probabilities") or {})
        # other phrasing ("X is the leading consideration"): X = last clause before the phrase
        X = m.group("X") if m else re.split(r"[,;:]", row.sentence_text[:mb.start()])[-1]
        gx = sorted({g for _, _, g in class_spans(_CIT.sub(" ", X), keys)})
        gs = sorted({g for _, _, g in class_spans(_CIT.sub(" ", row.sentence_text), keys)})
        LC.append({"rating_id": row.rating_id, "sentence_no": row.sentence_no, "packet": row.packet, "query": row.query,
                   "cohort": row.cohort, "code": row.code, "form": "stated" if m else "other phrasing",
                   "X": X.strip(), "groups_in_X": gx, "groups_in_sentence": gs})
    for u in US_ALL:
        q, tg, pl = sel[u]["query"], ep[u]["true_group"], payload[u]
        probs = pl.get("probabilities") or {}
        top = sorted(k for k, v in probs.items() if v == max(probs.values())) if probs else []
        votes = Counter(n.get("tumor_group") for n in pl["neighbors"] if n.get("tumor_group"))
        mode = sorted(k for k, v in votes.items() if v == max(votes.values())) if votes else []
        rows = [r for r in LC if r["query"] == q]
        st = [r for r in rows if r["form"] == "stated"]
        basis = "stated" if st else ("other phrasing" if rows else "none")
        named = sorted({g for r in (st or rows) for g in r["groups_in_X"]})
        if basis == "none":
            cat = "no leading-consideration statement"
        elif not named:
            cat = "no group recognised"
        elif tg in named:
            cat = "reference group only" if named == [tg] else "reference group and other group(s)"
        else:
            cat = "other group(s) only"
        LC_SUM[q] = {"query": q, "cohort": COHORT[u], "basis": basis, "named": named, "category": cat,
                     "reference_in_input": ep[u]["true_in_payload_labels"], "top_probability": top,
                     "neighbour_mode": mode, "named_equals_top_probability": bool(named) and named == top,
                     "named_equals_neighbour_mode": bool(named) and named == mode}
    pd.DataFrame(LC).to_csv(OUT / "leading_consideration_sentences.csv", index=False)
    pd.DataFrame(LC_SUM.values()).to_csv(OUT / "leading_consideration_by_summary_KEYSIDE.csv", index=False)

# ---------------------------------------------------------------- rated endpoints, overall and by subset
rated = {"as_rated": rated_endpoints(J), "strict_discrepant": rated_endpoints(J, strict_recode),
         "abbreviation_expansion_accepted": rated_endpoints(J, abbr_recode)}
for s in ADJ.get("recode_sensitivities", []):
    if s["rater"] != "R1":
        continue
    rec = {(x, str(y)): s["to_code"] for x, y in s["rows"]}
    rated[s["name"]] = rated_endpoints(J, rec)
dist, dist_first = code_dist(J)
spec_rows = []
for x in J[J.code.isin(US_CODES)].itertuples():
    gs = {g for _, _, g in class_spans(_CIT.sub(" ", x.sentence_text))}
    spec_rows.append({"rating_id": x.rating_id, "sentence_no": x.sentence_no, "query": x.query, "R1": x.code,
                      "groups_named": sorted(gs),
                      "groups_absent_from_input": sorted(g for g in gs if not H.group_in_labels(g, payload[x.run_uid]))})
dd = J[~J.first_draft_row & (J.probability_statement_tag == "Y")]
probtag = {"n_tagged_sentences": len(dd), "n_summaries_with_tag": int(dd.rating_id.nunique()), "codes": dict(Counter(dd.code))}

# ================================================================ endpoints
E = []


def add(name, clause, num, den, rater, value=None, notes="", subset="all"):
    if value is None and isinstance(num, (int, float)) and isinstance(den, (int, float)) and den:
        value = f"{num}/{den}"
    E.append({"name": name, "contract_clause": clause, "arm": S_ARM, "subset": subset, "numerator": num,
              "denominator": den, "rater": rater, "value": value, "notes": notes})


if READBACK:
    # analyze_r2 endpoint rows restricted to the automatic layer and rater 1 (same names, same order)
    ae = auto_end
    add("genuine invalid or fabricated citation (summaries with >=1)", "§6 row 1", max(
        ae["bad_citation_harness"], ae["bad_citation_independent"], ae["fabricated_ids_any_source"]), ae["n"], "auto",
        notes=f"CP95 {cp(ae['bad_citation_independent'], ae['n'])} — {CP_LABEL}")
    x, xs = rated["as_rated"], rated["strict_discrepant"]
    add("manual contradicted sentence: summaries with >=1", "§6 row 2 (target 0/20)",
        len(x["contradicted_summaries"]), x["n_summaries"], "R1",
        notes=f"queries {x['contradicted_summaries']}; {x['contradicted_sentences']} sentences / {x['n_sentences']}; "
              f"strict-discrepant: {len(xs['contradicted_summaries'])}/{xs['n_summaries']} ({xs['contradicted_summaries']}), "
              f"{xs['contradicted_sentences']} sentences")
    add("automatic claim contradicting record (summaries)", "§6 row 2; §5 automatic", ae["claim_contradicting_record"],
        ae["n"], "auto", notes=f"{ae['n_claims_returned']} atomic claims returned")
    add("manual unsupported or speculative-diagnostic sentences: summaries with >=1",
        "§6 row 3 (target <=1 sentence in <=1 summary)", len(x["unsup_or_specdiag_summaries"]), x["n_summaries"], "R1",
        notes=f"{x['unsup_or_specdiag_sentences']} sentences ({x['specdiag_sentences']} spec-diag, "
              f"{x['unsupported_sentences']} unsupported); per summary {x['unsup_or_specdiag_per_summary']}")
    add("manual uncertain / not resolvable sentences (never counted as supported)", "§5", x["uncertain_sentences"],
        x["n_sentences"], "R1", notes=f"summaries {x['uncertain_summaries']}")
    for layer, lst in (("lexical (automatic)", tg_abs["lexical_hits"]), ("after §5 adjudication", tg_abs["after_adjudication"])):
        add(f"true group named when absent from neighbours ∪ posterior — {layer}", "§6 row 4 (target 0/11)", len(lst),
            tg_abs["eligible"], "auto" if layer.startswith("lex") else "analyst adjudication", notes=f"queries {lst}")
    add("query native ID or patient stem in any output (text, text_first, raw)", "§6 row 8",
        ae["own_native_or_stem_any_source"], ae["n"], "auto",
        notes=f"harness query_id_leak/substring {ae['query_id_leak_harness']}; self-citation {ae['self_citation_of_query']}")
    pa = prob_acc
    add("probability statements: outputs with >=1 numeric statement / all correct / >=1 incorrect / absent / no evaluable text",
        "§6 row 12", pa.get("with_statement", 0), pa["n_nominal"], "auto (parsed + analyst-checked)",
        value=f"with {pa.get('with_statement', 0)}/{pa['n_nominal']}; all correct {pa.get('all_correct', 0)}; "
              f"incorrect {pa.get('with_incorrect', 0)}; absent {pa.get('statement_absent', 0)}; "
              f"no evaluable text {pa.get('no_evaluable_text', 0)}",
        notes=f"every positive class quoted in {pa.get('all_classes_quoted', 0)}")
    add("rated probability-tagged sentences by code", "§5 tag", probtag["n_tagged_sentences"], None, "R1",
        value=json.dumps(probtag["codes"]), notes=f"{probtag['n_summaries_with_tag']} summaries carry >=1 tagged sentence")
    add("rejection / too_long / empty / error output", "§6 rejection row", ae["non_ok_state"], ae["n"], "auto",
        notes=f"states {ae['states']}; revisions applied {ae['revision_applied']} {ae['revised_run_uids']}")
    add("rule-12 descriptive: summaries mentioning similarity/distance", "§6 rule-12 row",
        len(ae["rule12_similarity_mention"]), ae["n"], "auto", notes=f"queries {ae['rule12_similarity_mention']}")
    add("zero automatic flags — raw (E7-A3(2))", "E7-A3(2)", zero["raw"], ae["n"], "auto")
    add("zero automatic flags — post alpha/beta/gamma (E7-A3(2))", "E7-A3(2)", zero["post_alpha_gamma"], ae["n"],
        "auto + listed discounts", notes=f"remaining flagged {flagged_after}; unclassified flags count as raw: "
                                         f"{sum(f['class'] == 'unclassified' for f in flag_rows)}")

    def example(flag_mode, recode=None):
        ok_rated = set(rated_endpoints(J, recode)["all_sentences_supported_or_derivable"])
        zf = set(zero_lists[flag_mode])
        for q in QUERIES:
            if q in zf and q in ok_rated:
                return q
        return None

    for fm in ("post_alpha_gamma", "raw"):
        for rd in ("as_rated", "strict_discrepant"):
            add(f"MS L133 example: lowest qualifying A0 query ({fm} flags; {rd})", "round-1 §8 + E7-A3(2)", None, None,
                "R1", value=example(fm, strict_recode if rd == "strict_discrepant" else None))
else:
    SUBSETS = {"all": US_ALL, "CCDI-MCI": [u for u in US_ALL if COHORT[u] == "CCDI-MCI"],
               "STV": [u for u in US_ALL if COHORT[u] == "STV"]}
    for sname, us in SUBSETS.items():
        qs = {sel[u]["query"] for u in us}
        Jsub = J[J["query"].isin(qs)]
        ae, (zc, zl, fa), pa, tga = auto_endpoints(us), zero_flags(us), prob_counts(us), tg_absent(us)
        x, xs = rated_endpoints(Jsub), rated_endpoints(Jsub, strict_recode)
        xa = rated_endpoints(Jsub, abbr_recode)
        sd = [r for r in SD if r["query"] in qs]
        sd1_q = sorted({r["query"] for r in sd if r["sd_class"] == "SD-1"}, key=qkey)
        sd2_q = sorted({r["query"] for r in sd if r["sd_class"] == "SD-2"}, key=qkey)
        absent_q = sorted({r["query"] for r in sd if r["groups_absent_from_input"]}, key=qkey)
        sd1_keys = {(r["rating_id"], r["sentence_no"]) for r in sd if r["sd_class"] == "SD-1"}
        x_sd1 = rated_endpoints(Jsub, {k: "derivable" for k in sd1_keys})
        n = len(us)
        add("rejected / too long / empty / error", "001 §5", ae["non_ok_state"], n, "auto", subset=sname,
            notes=f"states {ae['states']}")
        add("style revisions applied", "001 §5", ae["revision_applied"], n, "auto", subset=sname,
            notes=f"items {ae['revised_queries']}; first draft differs from delivered text in {ae['first_draft_differs']}")
        add(">= 1 invalid or fabricated citation (every cited id in payload)", "001 §5",
            max(ae["bad_citation_harness"], ae["bad_citation_independent"], ae["fabricated_ids_any_source"]), n, "auto",
            subset=sname, notes=f"harness {ae['bad_citation_harness']}; independent {ae['bad_citation_independent']}; "
                                f"fabricated SJ-type ids {ae['fabricated_ids_any_source']}; CP95 "
                                f"{cp(ae['bad_citation_independent'], n)} — {CP_LABEL}")
        add("reference group named when absent from neighbours and scores — lexical", "001 §5",
            len(tga["lexical_hits"]), tga["eligible"], "auto", subset=sname,
            notes=f"eligible {tga['eligible_queries']}; hits {tga['lexical_hits']}")
        add("reference group named when absent from neighbours and scores — adjudicated", "001 §5",
            len(tga["after_adjudication"]), tga["eligible"], "lead adjudication", subset=sname,
            notes=f"after true_group_not_a_leak: {tga['after_adjudication']}")
        add("probability statement present; all values correct; every class quoted", "001 §5",
            pa.get("with_statement", 0), n, "auto (parsed)", subset=sname,
            value=f"present {pa.get('with_statement', 0)}/{n}; all correct {pa.get('all_correct', 0)}; "
                  f"incorrect {pa.get('with_incorrect', 0)}; every class quoted {pa.get('all_classes_quoted', 0)}; "
                  f"absent {pa.get('statement_absent', 0)}")
        add("automatic claim contradicting the cited record", "001 §5", ae["claim_contradicting_record"], n, "auto",
            subset=sname, notes=f"{ae['n_claims_returned']} claims returned")
        add("zero automatic flags — raw", "001 §5 / E7-A3(2)", zc["raw"], n, "auto", subset=sname)
        add("zero automatic flags — after alpha-gamma", "001 §5 / E7-A3(2)", zc["post_alpha_gamma"], n,
            "auto + lead classes", subset=sname,
            notes=f"remaining flagged {fa}; unclassified flags count as raw: "
                  f"{sum(f['class'] == 'unclassified' and f['run_uid'] in us for f in flag_rows)}")
        add(">= 1 contradicted sentence — lenient 'discrepant' (primary)", "001 §5", len(x["contradicted_summaries"]),
            n, "R1", subset=sname, notes=f"{x['contradicted_sentences']} sentences / {x['n_sentences']}; "
                                         f"{x['contradicted_summaries']}")
        add(">= 1 contradicted sentence — strict 'discrepant' (sensitivity)", "001 §5", len(xs["contradicted_summaries"]),
            n, "R1 + mechanical strict rows", subset=sname,
            notes=f"{xs['contradicted_sentences']} sentences; {xs['contradicted_summaries']}")
        add(">= 1 unsupported or speculative-diagnostic sentence — total", "001 §5",
            len(x["unsup_or_specdiag_summaries"]), n, "R1", subset=sname,
            notes=f"{x['unsup_or_specdiag_sentences']} sentences ({x['specdiag_sentences']} spec-diag, "
                  f"{x['unsupported_sentences']} unsupported)")
        add(">= 1 unsupported or speculative-diagnostic sentence — SD-1 only", "001 §5 / §4 S4",
            len(set(sd1_q) - set(sd2_q)), n, "R1 + SD rule", subset=sname,
            notes=f"summaries with any SD-1 {len(sd1_q)}; SD-1 sentences {sum(r['sd_class'] == 'SD-1' for r in sd)}")
        add(">= 1 unsupported or speculative-diagnostic sentence — SD-2 (any)", "001 §5 / §4 S4", len(sd2_q), n,
            "R1 + SD rule", subset=sname,
            notes=f"SD-2 sentences {sum(r['sd_class'] == 'SD-2' for r in sd)}; summaries {sd2_q}")
        add("any sentence naming a diagnosis absent from the input (subset of SD-2)", "001 §5", len(absent_q), n,
            "R1 + harness aliases", subset=sname, notes=f"summaries {absent_q} (groups recognised by harness aliases only)")
        add("uncertain sentences — as coded", "001 §5", x["uncertain_sentences"], x["n_sentences"], "R1", subset=sname,
            notes=f"in {len(x['uncertain_summaries'])} summaries")
        add("uncertain sentences — abbreviation-expansion accepted (sensitivity)", "001 §5", xa["uncertain_sentences"],
            xa["n_sentences"], "R1 + mechanical recode", subset=sname,
            notes=f"in {len(xa['uncertain_summaries'])} summaries; every sentence supported/derivable "
                  f"{len(xa['all_sentences_supported_or_derivable'])}/{n}")
        add("every sentence supported or derivable — as coded", "001 §5",
            len(x["all_sentences_supported_or_derivable"]), n, "R1", subset=sname)
        add("every sentence supported or derivable — SD-1 set aside", "001 §5",
            len(x_sd1["all_sentences_supported_or_derivable"]), n, "R1 + SD rule", subset=sname)
        add("similarity or distance mentioned", "001 §5", len(ae["rule12_similarity_mention"]), n, "auto", subset=sname)
        add("query's own identifier appears in the text (given in the payload; descriptive, not a leak channel)",
            "001 §2", ae["own_native_or_stem_any_source"], n, "auto", subset=sname)
        lcs = [LC_SUM[q] for q in sorted(qs, key=qkey)]
        cats = Counter(r["category"] for r in lcs)
        add("leading-consideration statement (stated SD-1 form) present — key-side, any code", "001 §6 can", sum(
            r["basis"] == "stated" for r in lcs), n, "auto (SD-1 regex)", subset=sname,
            notes=f"other phrasing only {sum(r['basis'] == 'other phrasing' for r in lcs)}; none "
                  f"{sum(r['basis'] == 'none' for r in lcs)}")
        add("leading-consideration statement names the reference group vs another group — key-side, descriptive",
            "001 §6 can", cats.get("reference group only", 0) + cats.get("reference group and other group(s)", 0), n,
            "auto (aliases) + key", subset=sname, value=json.dumps(dict(sorted(cats.items()))),
            notes=f"named group(s) = top-probability class in {sum(r['named_equals_top_probability'] for r in lcs)}; "
                  f"= neighbour tumor_group mode in {sum(r['named_equals_neighbour_mode'] for r in lcs)}; "
                  f"reference group absent from input in {sum(not r['reference_in_input'] for r in lcs)}")
        add("round-2 target rows, descriptive only (no verdict): 0 contradicted summaries / <=1 U-or-SD sentence in "
            "<=1 summary", "001 §5", None, n, "R1", subset=sname,
            value=f"contradicted {len(x['contradicted_summaries'])}/{n}; U-or-SD {x['unsup_or_specdiag_sentences']} "
                  f"sentences in {len(x['unsup_or_specdiag_summaries'])}/{n} (SD-1 expected by instruction design)")

json.dump({"script_sha256": sha(__file__), "mode": a.mode, "input_hashes": input_hashes, "endpoints": E},
          open(OUT / "endpoints.json", "w"), indent=1, default=str)
work = {"dist": dist, "dist_first_draft": dist_first, "rated": rated, "auto_end": auto_end, "tg_abs": tg_abs,
        "zero": zero, "zero_lists": zero_lists, "flagged_after": flagged_after, "prob_acc": prob_acc,
        "prob_detail": {u: prob_detail_all[u] for u in US_ALL if u in prob_detail_all}, "probtag": probtag,
        "spec_rows": spec_rows, "sd_rows": SD, "sd_rule": SD1_RULE, "strict_discrepant_mechanical": strict_mech,
        "strict_rows_used": strict_rows_used, "abbreviation_expansion_rows": abbr_rows,
        "abbreviation_rule": "uncertain rows whose rater note/support_trace matches /expan|spell|abbreviat|alias|full "
                             "name|written out/i and whose sentence contains a spelled-out harness alias of a group "
                             "present in the payload label fields; recoded derivable",
        "form_checks": form_checks, "leading_consideration_sentences": LC, "leading_consideration_by_summary": LC_SUM, "harness_reproduction_diffs": diffs,
        "indep_bad": {u: v for u, v in indep.items() if v["bad_citation_tokens"] or (READBACK and v["own_native_or_stem"])}}
json.dump(work, open(OUT / "analysis_work.json", "w"), indent=1, default=str)
print("OK", a.mode, len(E), "endpoints;", len(flag_rows), "flags; zero", zero, "; SD", Counter(r["sd_class"] for r in SD))
