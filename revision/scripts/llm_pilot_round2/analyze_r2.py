"""S5c key-side endpoint analysis: A0-only port of round-1 `s5b_pilot_20261009/analysis/scripts/analyze.py`.

Specification: [internal record] §5-§7 (round-1 specification §5/§6 + E7-A3(2)/E7-A4).
KEY-SIDE SCRIPT. Read-only on every input; writes only to --out-dir.

What is ported unchanged (same code, same definitions): harness reproduction of audit()/fabricated_ids()/
mention_test(); independent citation and own-ID checks; flag generation (types, values, sources, the
"first(=final)" rule); summary_flags(); zero-flag counts raw and after the alpha/beta/gamma discount;
probability-statement parser and checker; rating join; rated_endpoints(); rated probability tags; rejection,
rule-12 and true-group-absent rows; the round-1 example rule (readback only).

What differs from round 1 (port requirements):
* paths, arm, ratings, rating map and harness are parameters; A0 (or any one arm) only; no A1/A2/R code.
* NO adjudication is hard-coded. Everything that was a per-response ruling in round 1 is read from one external
  JSON file (--adjudication, sha256 recorded; `{}` allowed):
    flag_classes                      [{run_uid, source, flag_type, value, occurrence, class, payload_field, note}]
                                      class in alpha|beta|gamma|raw; an unclassified flag counts as raw
    true_group_not_a_leak             [run_uid, ...]  lexical true-group hits adjudicated as not a leak
    dispositions                      [{item, run_uid, original_automatic_code, analyst_disposition, reason, evidence}]
    strict_discrepant_rows            [[rating_id, sentence_no], ...]  recoded contradicted (strict reading)
    recode_sensitivities              [{name, rater, rows: [[rating_id, sentence_no]], to_code}]
  The script does not classify flags itself: classification is the independent, before unblinding.
* --flags-only: no ratings, no truth-dependent endpoint beyond the automatic leak candidates; writes the
  automatic flag list (auto_flags.csv) for the independent classification and stops.
* the round-2 example rule (specification §7) is computed after the join, beside the round-1 rule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.machinery
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
import pandas as pd  # noqa: E402
from scipy.stats import beta  # noqa: E402

SHORT = {"a0_masked_posterior": "A0", "a1_opaque": "A1", "a2_cf_plurality": "A2", "repeat": "R"}
OK_CODES = {"supported", "derivable"}
CODES = ["supported", "derivable", "speculative-diagnostic", "unsupported", "contradicted", "uncertain"]
CP_LABEL = "nominal model-based interval under an independent common-probability assumption (AF1); per-arm only"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return [round(float(100 * lo), 1), round(float(100 * hi), 1)]


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--harness", type=Path, required=True)
    ap.add_argument("--arm", default="a0_masked_posterior")
    ap.add_argument("--adjudication", type=Path, default=None, help="external rulings JSON ({} allowed)")
    ap.add_argument("--ratings-r1", type=Path, default=None)
    ap.add_argument("--ratings-r2", type=Path, default=None)
    ap.add_argument("--rating-map", type=Path, default=None, help="rating_packet_key/rating_id_map.csv")
    ap.add_argument("--scoring-key", type=Path, default=None, help="optional cross-check of the key-side truth")
    ap.add_argument("--flags-only", action="store_true")
    a = ap.parse_args()
    if not a.flags_only and not (a.ratings_r1 and a.ratings_r2 and a.rating_map):
        ap.error("--ratings-r1/--ratings-r2/--rating-map are required unless --flags-only")
    return a


a = parse()
RUN, OUT, ARM = a.run_dir, a.out_dir, a.arm
S_ARM = SHORT[ARM]
OUT.mkdir(parents=True, exist_ok=True)
ADJ = json.loads(a.adjudication.read_text()) if a.adjudication else {}
input_hashes = {str(p): sha(p) for p in [a.harness, RUN / "runs.jsonl", RUN / "key/endpoints_auto.jsonl",
                                         RUN / "meta.json", RUN / "audit_summary.json"]
                + ([a.adjudication] if a.adjudication else [])
                + ([a.ratings_r1, a.ratings_r2, a.rating_map] if not a.flags_only else [])
                + ([a.scoring_key] if a.scoring_key else [])}

# ---------------------------------------------------------------- harness (read-only import, by path)
_ld = importlib.machinery.SourceFileLoader("s5c_harness_readonly", str(a.harness))
H = importlib.util.module_from_spec(importlib.util.spec_from_loader(_ld.name, _ld))
_ld.exec_module(H)

# ---------------------------------------------------------------- load run (selected rows of ARM only)
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
QUERIES = sorted({r["query"] for r in sel.values()}, key=int)
if a.scoring_key:
    sk = pd.read_csv(a.scoring_key, dtype=str).set_index("query")
    assert all(sk.loc[q, "true_group"] == ep[f"{q}_{ARM}_0"]["true_group"] for q in QUERIES)


def uid(q):
    return f"{q}_{ARM}_0"


def kq(u):
    e = ep[u]
    return {"true_group": e["true_group"], "native_id": e["native_id"], "patient_stem": e["patient_stem"]}


def sources(u):
    r = sel[u]
    s = {"text": r["text"], "text_first": r["text_first"]}
    for c in resp[u]["calls"]:
        s[f"raw_call_{c['call_ordinal']}"] = c.get("raw_content")
    return s


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def sentences(t):
    return [s for s in _SENT.split(t or "") if s.strip()]


# ---------------------------------------------------------------- automatic layer: harness reproduction
recomp, diffs = {}, []
for u, r in sel.items():
    pl = payload[u]
    au = H.audit(r["text"], r["telemetry"].get("claims") or [], pl, kq(u)["true_group"])
    au.pop("leaks_true_group")
    plur = (au.pop("plurality"), au.pop("plurality_tied"))
    stored = dict(r["audit"])
    for k in ("plurality", "plurality_tied"):
        stored.pop(k, None)
    if au != stored:
        diffs.append({"uid": u, "field": [k for k in au if au.get(k) != stored.get(k)]})
    af = None
    if r["text_first"]:
        af = H.audit(r["text_first"], [], pl, kq(u)["true_group"])
        af.pop("leaks_true_group"); af.pop("plurality"); af.pop("plurality_tied")
        sf = dict(r["audit_first"]); sf.pop("plurality", None); sf.pop("plurality_tied", None)
        if af != sf:
            diffs.append({"uid": u, "source": "first", "field": [k for k in af if af.get(k) != sf.get(k)]})
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

# ---------------------------------------------------------------- flags (generation only; classes are external)
_m2 = re.compile(r"Claim \d+: diagnosis '(.+) \((\d+) of (\d+)\)' not found in payload")
flag_rows = []
_occ = Counter()


def addflag(u, src, ftype, value, note):
    key = (u, src, ftype, str(value))
    _occ[key] += 1
    flag_rows.append({"run_uid": u, "arm": S_ARM, "query": sel[u]["query"], "source": src, "flag_type": ftype,
                      "value": str(value), "occurrence": _occ[key], "note": note})


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
        addflag(u, "any", "fabricated_id", ",".join(r["fabricated_ids"]), "")
    if r["query_id_leak"] or r["query_id_substring_sources"]:
        addflag(u, "any", "query_id_leak_or_substring", str(r["query_id_substring_sources"]), "")
    if ep[u]["true_group_mention"]["leak_candidate"]:
        hits = ep[u]["true_group_mention"]["hits"]
        addflag(u, ",".join(sorted({h["source"] for h in hits})), "true_group_leak_candidate",
                ep[u]["true_group"] + ": " + "/".join(sorted({h["match"] for h in hits})), "")
    if r["output_state"] != "ok":
        addflag(u, "final", "output_state", r["output_state"], "")

# external classes
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
for u, r in sorted(sel.items(), key=lambda kv: int(kv[1]["query"])):
    t = r["telemetry"] or {}
    per_summary.append({
        "run_uid": u, "query": r["query"], "output_state": r["output_state"], "n_calls": r["n_calls"],
        "revision_applied": bool(t.get("revision_applied")), "revision_rejected": t.get("revision_rejected"),
        "n_grounding_warnings_final": len(t.get("grounding_warnings") or []),
        "n_style_warnings_first": len(t.get("style_warnings_first") or []),
        "n_style_warnings_final": len(t.get("style_warnings_final") or []),
        "n_fabricated_ids": r["n_fabricated_ids"], "query_id_leak": bool(r["query_id_leak"]),
        "query_id_substring_sources": len(r["query_id_substring_sources"] or []),
        "true_group_leak_candidate": bool(ep[u]["true_group_mention"]["leak_candidate"]),
        "n_flag_rows": sum(f["run_uid"] == u for f in flag_rows)})

FLAG_COLS = ["run_uid", "arm", "query", "source", "flag_type", "value", "occurrence", "class", "payload_field",
             "counts_after_alpha_gamma", "note"]
pd.DataFrame(flag_rows, columns=FLAG_COLS).to_csv(OUT / "flag_classification.csv", index=False)
if a.flags_only:
    out_cols = [c for c in FLAG_COLS if c not in ("class", "payload_field", "counts_after_alpha_gamma")]
    df = pd.DataFrame(flag_rows, columns=FLAG_COLS)[out_cols]
    df.insert(len(out_cols), "class_lead", "")
    df.to_csv(OUT / "auto_flags.csv", index=False)
    pd.DataFrame(per_summary).to_csv(OUT / "auto_flags_per_summary.csv", index=False)
    json.dump({"input_hashes": input_hashes, "script_sha256": sha(__file__), "n_flag_rows": len(flag_rows),
               "harness_reproduction_diffs": diffs}, open(OUT / "flags_only_meta.json", "w"), indent=1)
    print("flags-only:", len(flag_rows), "flag rows;", len(per_summary), "summaries")
    sys.exit(0)


def summary_flags(u, discount):
    fl = [f for f in flag_rows if f["run_uid"] == u and f["source"] in ("final", "first(=final)", "any")
          or (f["run_uid"] == u and f["flag_type"] in ("true_group_leak_candidate", "a2_original_plurality_mention"))]
    return [f for f in fl if (not discount) or f["counts_after_alpha_gamma"]]


zero, zero_lists, flagged_after = {}, {}, {}
us_all = [uid(q) for q in QUERIES]
for mode, disc in (("raw", False), ("post_alpha_gamma", True)):
    z = [sel[u]["query"] for u in us_all if not summary_flags(u, disc)]
    zero[mode], zero_lists[mode] = len(z), z
flagged_after = {sel[u]["query"]: sorted({f["flag_type"] for f in summary_flags(u, True)})
                 for u in us_all if summary_flags(u, True)}

# ---------------------------------------------------------------- probability statements
_alias_pat = []
for g, als in H.ALIASES.items():
    for al in als + ([g.title()] if g in ("NORMAL", "MISC") else []):
        _alias_pat.append((al, g))
_alias_pat.sort(key=lambda x: -len(x[0]))


def class_spans(s):
    spans = []
    for al, g in _alias_pat:
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


def prob_statements(text):
    out = []
    for s in sentences(text):
        if not _PROB_SENT.search(s) or _EXCL.search(s):
            continue
        spans = class_spans(_CIT.sub(" ", s))
        s2 = _CIT.sub(" ", s)
        for m in re.finditer(r"(?<![\w.])(\d+\.\d+|\d+(?:\.\d+)?%)", s2):
            x = m.group(1)
            after = s2[m.end():m.end() + 12]
            fm = re.match(r"\s+for\s+", after)
            g = None
            if fm:
                nxt = [sp for sp in spans if sp[0] >= m.end()]
                g = nxt[0][2] if nxt else None
            else:
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


cnt, prob_detail = Counter(), {}
for u in us_all:
    r = sel[u]
    probs = payload[u].get("probabilities") or {}
    if r["output_state"] not in ("ok", "truncated") or not r["text"]:
        cnt["no_evaluable_text"] += 1
        continue
    sts = prob_statements(r["text"])
    res = [check_stmt(s, probs) for s in sts]
    prob_detail[u] = [{**s, "result": c} for s, c in zip(sts, res)]
    qual = bool(re.search(r"probabilit|bayesian|classifier", r["text"], re.I))
    if not sts:
        cnt["statement_absent"] += 1
        cnt["of_which_qualitative_only"] += qual
    else:
        cnt["with_statement"] += 1
        cnt["all_correct" if all(c == "correct" for c in res) else "with_incorrect"] += 1
        cnt["all_classes_quoted"] += set(probs) <= {s["group"] for s in sts}
cnt["n_nominal"] = len(us_all)
prob_acc = dict(cnt)

# ---------------------------------------------------------------- automatic endpoints
us = us_all
auto_end = {
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
    "claim_contradicting_record": sum(sel[u]["audit"]["n_claims_contradicting_record"] > 0 for u in us),
    "n_claims_returned": sum(sel[u]["audit"]["n_claims"] for u in us),
    "dominant_false_raw": sum(sel[u]["audit"]["dominant_statement_consistent"] is False for u in us),
    "rule12_similarity_mention": [sel[u]["query"] for u in us if re.search(
        r"similarit|distance", re.sub(r"similar to the current case", "", sel[u]["text"]), re.I)],
}
not_leak = set(ADJ.get("true_group_not_a_leak", []))
elig = [u for u in us if not ep[u]["true_in_payload_labels"]]
lex = [sel[u]["query"] for u in elig if ep[u]["true_group_mention"]["mentioned_any"]]
tg_abs = {"eligible": len(elig), "eligible_queries": [sel[u]["query"] for u in elig], "lexical_hits": lex,
          "after_adjudication": [q for q in lex if uid(q) not in not_leak]}

# ---------------------------------------------------------------- rating join
idmap = pd.read_csv(a.rating_map, dtype=str, keep_default_na=False)
idmap = idmap[idmap.arm == ARM]
imap = idmap.set_index("rating_id")


def join_form(path, outname):
    orig = pd.read_csv(path, dtype=str, keep_default_na=False)
    orig = orig[orig.rating_id.isin(imap.index)].reset_index(drop=True)
    out = orig.copy()
    out["arm"] = [imap.loc[x, "arm"] for x in orig.rating_id]
    out["query"] = [imap.loc[x, "query"] for x in orig.rating_id]
    out["run_uid"] = [imap.loc[x, "run_uid"] for x in orig.rating_id]
    out.to_csv(OUT / outname, index=False)
    chk = pd.read_csv(OUT / outname, dtype=str, keep_default_na=False)
    assert chk[list(orig.columns)].equals(orig)
    return chk


J = {"R1": join_form(a.ratings_r1, "rating_join_r1.csv"), "R2": join_form(a.ratings_r2, "rating_join_r2.csv")}
for k in J:
    J[k]["first_draft_row"] = J[k]["sentence_no"].str.startswith("F")
    J[k]["arm_s"] = J[k]["arm"].map(SHORT)
assert (J["R1"][["rating_id", "sentence_no", "sentence_text"]].values ==
        J["R2"][["rating_id", "sentence_no", "sentence_text"]].values).all()
assert set(J["R1"].run_uid) == set(sel)
for rid, g in J["R1"][~J["R1"].first_draft_row].groupby("rating_id"):
    u = imap.loc[rid, "run_uid"]
    assert re.sub(r"\s+", " ", " ".join(g.sentence_text)) == re.sub(r"\s+", " ", sel[u]["text"]).strip(), rid

dist = {}
for rk, df in J.items():
    d = df[~df.first_draft_row]
    dist[rk] = {"n_sentences": len(d), "n_summaries": d.rating_id.nunique(),
                "codes": {c: int((d.code == c).sum()) for c in CODES},
                "prob_tag_Y": int((d.probability_statement_tag == "Y").sum())}
    d = df[df.first_draft_row]
    dist[rk + "_first_draft"] = {"n_sentences": len(d), "items": sorted(d["query"].unique().tolist(), key=int),
                                 "codes": {c: int((d.code == c).sum()) for c in CODES}}
both = J["R1"][["rating_id", "sentence_no", "sentence_text", "arm_s", "query", "run_uid", "first_draft_row"]].copy()
both["c1"], both["c2"] = J["R1"].code, J["R2"].code
both["p1"], both["p2"] = J["R1"].probability_statement_tag, J["R2"].probability_statement_tag
n_agree = int((both.c1 == both.c2).sum())
n_agree_delivered = int(((both.c1 == both.c2) & ~both.first_draft_row).sum())


def rated_endpoints(df, recode=None):
    d = df[~df.first_draft_row].copy()
    if recode is not None:
        d["code"] = [recode.get((x, y), c) for x, y, c in zip(d.rating_id, d.sentence_no, d.code)]
    per = d.groupby("query").code.apply(list)
    us_mask = d.code.isin(["unsupported", "speculative-diagnostic"])
    us_sum = d[us_mask].groupby("query").size()
    return {"n_summaries": int(d.rating_id.nunique()), "n_sentences": len(d),
            "contradicted_sentences": int((d.code == "contradicted").sum()),
            "contradicted_summaries": sorted([q for q, cs in per.items() if "contradicted" in cs], key=int),
            "unsup_or_specdiag_sentences": int(us_mask.sum()),
            "unsup_or_specdiag_summaries": sorted(us_sum.index.tolist(), key=int),
            "unsup_or_specdiag_per_summary": {q: int(v) for q, v in us_sum.items()},
            "specdiag_sentences": int((d.code == "speculative-diagnostic").sum()),
            "unsupported_sentences": int((d.code == "unsupported").sum()),
            "uncertain_sentences": int((d.code == "uncertain").sum()),
            "uncertain_summaries": sorted(d[d.code == "uncertain"]["query"].unique().tolist(), key=int),
            "all_sentences_supported_or_derivable": sorted([q for q, cs in per.items()
                                                            if all(c in OK_CODES for c in cs)], key=int)}


strict_recode = {(x, str(y)): "contradicted" for x, y in ADJ.get("strict_discrepant_rows", [])}
rated = {}
for rk in ("R1", "R2"):
    rated[(rk, "as_rated")] = rated_endpoints(J[rk])
    rated[(rk, "strict_discrepant")] = rated_endpoints(J[rk], strict_recode)
for s in ADJ.get("recode_sensitivities", []):
    rec = {(x, str(y)): s["to_code"] for x, y in s["rows"]}
    rated[(s["rater"], s["name"])] = rated_endpoints(J[s["rater"]], rec)

spec_rows = []
for i, x in both[both.c1.isin(["unsupported", "speculative-diagnostic"]) |
                 both.c2.isin(["unsupported", "speculative-diagnostic"])].iterrows():
    gs = {g for _, _, g in class_spans(_CIT.sub(" ", x.sentence_text))}
    spec_rows.append({"rating_id": x.rating_id, "sentence_no": x.sentence_no, "query": x["query"], "R1": x.c1,
                      "R2": x.c2, "groups_named": sorted(gs),
                      "groups_absent_from_input": sorted(g for g in gs if not H.group_in_labels(g, payload[x.run_uid]))})
probtag = {}
for rk, cc, pp in (("R1", "c1", "p1"), ("R2", "c2", "p2")):
    d = both[~both.first_draft_row & (both[pp] == "Y")]
    probtag[rk] = {"n_tagged_sentences": len(d), "n_summaries_with_tag": int(d.rating_id.nunique()),
                   "codes": dict(Counter(d[cc]))}

# ---------------------------------------------------------------- example rules (after the join)


def example(rk, flag_mode, recode=None, need_hit=False):
    ok_rated = set(rated_endpoints(J[rk], recode)["all_sentences_supported_or_derivable"])
    zf = set(zero_lists[flag_mode])
    for q in QUERIES:
        u = uid(q)
        pl = payload[u]
        hit = ep[u]["true_group"] in ({n.get("tumor_group") for n in pl["neighbors"]} | set(pl.get("probabilities") or {}))
        if q in zf and q in ok_rated and (hit or not need_hit):
            return q
    return None


round1_rule = {(rk, fm, rd): example(rk, fm, strict_recode if rd == "strict_discrepant" else None)
               for rk in ("R1", "R2") for fm in ("post_alpha_gamma", "raw") for rd in ("as_rated", "strict_discrepant")}
both_ok = set(rated_endpoints(J["R1"])["all_sentences_supported_or_derivable"]) & \
    set(rated_endpoints(J["R2"])["all_sentences_supported_or_derivable"])
cand = [q for q in QUERIES if q in both_ok and q in set(zero_lists["post_alpha_gamma"])]
hits = [q for q in cand if ep[uid(q)]["true_group"] in
        ({n.get("tumor_group") for n in payload[uid(q)]["neighbors"]} | set(payload[uid(q)].get("probabilities") or {}))]
round2_rule = {"qualifying_clean_both_raters_post_alpha_gamma": cand, "of_which_true_group_in_input": hits,
               "selected": hits[0] if hits else (cand[0] if cand else None),
               "basis": "hit" if hits else ("fallback (no hit)" if cand else "none qualifies")}
if round2_rule["selected"]:
    u = uid(round2_rule["selected"])
    c = next(c for c in calls_by_uid[u] if c["selected"])
    round2_rule.update({"run_uid": u, "response_id": c["call"]["response_id"],
                        "rating_id": idmap[idmap.run_uid == u].rating_id.iloc[0]})

# ---------------------------------------------------------------- endpoints.json (A0 rows, round-1 names)
E = []


def add(name, clause, num, den, rater, value=None, notes=""):
    if value is None and isinstance(num, (int, float)) and isinstance(den, (int, float)) and den:
        value = f"{num}/{den}"
    E.append({"name": name, "contract_clause": clause, "arm": S_ARM, "numerator": num, "denominator": den,
              "rater": rater, "value": value, "notes": notes})


ae = auto_end
add("genuine invalid or fabricated citation (summaries with >=1)", "§6 row 1", max(
    ae["bad_citation_harness"], ae["bad_citation_independent"], ae["fabricated_ids_any_source"]), ae["n"], "auto",
    notes=f"CP95 {cp(ae['bad_citation_independent'], ae['n'])} — {CP_LABEL}")
for rk in ("R1", "R2"):
    x, xs = rated[(rk, "as_rated")], rated[(rk, "strict_discrepant")]
    add("manual contradicted sentence: summaries with >=1", "§6 row 2 (target 0/20)",
        len(x["contradicted_summaries"]), x["n_summaries"], rk,
        notes=f"queries {x['contradicted_summaries']}; {x['contradicted_sentences']} sentences / {x['n_sentences']}; "
              f"strict-discrepant: {len(xs['contradicted_summaries'])}/{xs['n_summaries']} ({xs['contradicted_summaries']}), "
              f"{xs['contradicted_sentences']} sentences")
add("automatic claim contradicting record (summaries)", "§6 row 2; §5 automatic", ae["claim_contradicting_record"],
    ae["n"], "auto", notes=f"{ae['n_claims_returned']} atomic claims returned")
for rk in ("R1", "R2"):
    x = rated[(rk, "as_rated")]
    add("manual unsupported or speculative-diagnostic sentences: summaries with >=1",
        "§6 row 3 (target <=1 sentence in <=1 summary)", len(x["unsup_or_specdiag_summaries"]), x["n_summaries"], rk,
        notes=f"{x['unsup_or_specdiag_sentences']} sentences ({x['specdiag_sentences']} spec-diag, "
              f"{x['unsupported_sentences']} unsupported); per summary {x['unsup_or_specdiag_per_summary']}")
    add("manual uncertain / not resolvable sentences (never counted as supported)", "§5", x["uncertain_sentences"],
        x["n_sentences"], rk, notes=f"summaries {x['uncertain_summaries']}")
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
for rk in ("R1", "R2"):
    pt = probtag[rk]
    add("rated probability-tagged sentences by code", "§5 tag", pt["n_tagged_sentences"], None, rk,
        value=json.dumps(pt["codes"]), notes=f"{pt['n_summaries_with_tag']} summaries carry >=1 tagged sentence")
add("rejection / too_long / empty / error output", "§6 rejection row", ae["non_ok_state"], ae["n"], "auto",
    notes=f"states {ae['states']}; revisions applied {ae['revision_applied']} {ae['revised_run_uids']}")
add("rule-12 descriptive: summaries mentioning similarity/distance", "§6 rule-12 row",
    len(ae["rule12_similarity_mention"]), ae["n"], "auto", notes=f"queries {ae['rule12_similarity_mention']}")
add("zero automatic flags — raw (E7-A3(2))", "E7-A3(2)", zero["raw"], ae["n"], "auto")
add("zero automatic flags — post alpha/beta/gamma (E7-A3(2))", "E7-A3(2)", zero["post_alpha_gamma"], ae["n"],
    "auto + listed discounts", notes=f"remaining flagged {flagged_after}; unclassified flags count as raw: "
                                     f"{sum(f['class'] == 'unclassified' for f in flag_rows)}")
for (rk, fm, rd), q in round1_rule.items():
    add(f"MS L133 example: lowest qualifying A0 query ({fm} flags; {rd})", "round-1 §8 + E7-A3(2)", None, None, rk,
        value=q)
add("round-2 example rule (contract §7)", "S5c §7", None, None, "R1 and R2", value=round2_rule["selected"],
    notes=json.dumps(round2_rule))
add("R1 vs R2 sentence-level raw agreement (all rows incl. first drafts)", "E7-A4", n_agree, len(both), "R1 vs R2")
add("R1 vs R2 sentence-level raw agreement (delivered rows)", "E7-A4", n_agree_delivered,
    int((~both.first_draft_row).sum()), "R1 vs R2")

json.dump({"script_sha256": sha(__file__), "input_hashes": input_hashes, "arm": ARM, "endpoints": E},
          open(OUT / "endpoints.json", "w"), indent=1, default=str)
adj_rows = [{**d, "arm": S_ARM, "query": sel[d["run_uid"]]["query"], "true_group": ep[d["run_uid"]]["true_group"]}
            for d in ADJ.get("dispositions", []) if d["run_uid"] in sel]
pd.DataFrame(adj_rows).to_csv(OUT / "adjudication.csv", index=False)
work = {"dist": dist, "rated": {"|".join(k): v for k, v in rated.items()}, "auto_end": auto_end, "tg_abs": tg_abs,
        "zero": zero, "zero_lists": zero_lists, "flagged_after": flagged_after, "prob_acc": prob_acc,
        "prob_detail": prob_detail, "probtag": probtag, "spec_rows": spec_rows, "n_agree": n_agree,
        "n_agree_delivered": n_agree_delivered, "n_rows": len(both),
        "round1_rule": {"|".join(k): v for k, v in round1_rule.items()}, "round2_rule": round2_rule,
        "harness_reproduction_diffs": diffs, "indep_bad": {u: v for u, v in indep.items()
                                                           if v["bad_citation_tokens"] or v["own_native_or_stem"]}}
json.dump(work, open(OUT / "analysis_work.json", "w"), indent=1, default=str)
print("OK", len(E), "endpoints;", len(flag_rows), "flags;", len(adj_rows), "dispositions; zero", zero)
