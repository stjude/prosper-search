"""E5-ext (specification amendment E7-A5): S5b EXTENSION query list and published
Bayesian posterior (N = 5, T = 20) for each extension query.

Derived from the frozen E5 v2 script (r3_5_s5b_posterior_20q.py 2f66ee82...);
only the query selection (E7-A5(1): the 90 eligible queries minus the 20 pilot
queries minus pilot patients, one query per patient keeping the lowest numeric
id; deterministic, no RNG), the absence of a repeat list, the output directory
and the selection checks differ. Posterior/fragment/weights/scoring-key
rendering is unchanged.

Inputs are the saved S5 artefacts and the same all-tissue UNI2 OT-cosine
same-patient-masked matrix / rich metadata the S5 harness used, loaded through
the deployed ``comet-llm`` payload builder so the neighbours the posterior is
computed from are the neighbours the LLM will be shown; plus the pilot list
``r3_5_s5b_posterior_20q_20261009_v2/harness_query_ids.txt``.

Outputs (``$PROSPER_OUT/r3_5_s5b_posterior_ext_20261012/``):
  harness_query_ids.txt             extension ids only, numeric order (harness-facing)
  ext_queries.csv                   every S5 query with selected/excluded status and reason
                                    (KEY MATERIAL: never a harness input)
  posterior_fragments.json          {query: {"probabilities": {...}}} payload fragment,
                                    deployed rounding (6 dp, > 1e-8, descending); no truth
  scoring_key_DO_NOT_SEND.csv       truth, plurality, posterior argmax and margin per query
  posteriors_full.csv               full posterior row per query (all classes)
  neighbor_weights.csv              per shown neighbour: rank, distance, normalised weight w_j
                                    (posterior = sum of w_j by neighbour label; lets a
                                    counterfactual arm re-aggregate without the matrix)
  checks.json, sha256.txt, COMPLETE
Read-only on every input. Local run, CPU seconds.
"""
from __future__ import annotations

import collections
import re
import contextlib
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
OUT_ROOT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
S = (REPO.parent / "prosper-search").resolve()
COMET = Path("/PROSPER_WORKSPACE")
LLM_SRC = COMET / "comet-llm" / "src"
MATRIX = (COMET / "comet-feature-gen" / "comet-uni2" / "output" / "similarity_matrix"
          / "combined_similarity_matrix_uni2_ot_cosine_no_same_patient.pkl")
RICH_META = COMET / "comet-llm" / "data" / "final_scan_id_disease_map_rich_data.csv"
S5 = OUT_ROOT / "s5_llm_grounding"
OUT = OUT_ROOT / "r3_5_s5b_posterior_ext_20261012"
PILOT_IDS = OUT_ROOT / "r3_5_s5b_posterior_20q_20261009_v2" / "harness_query_ids.txt"

N_PILOT = 20
N_EXT_MAX = 70                                 # E7-A5(1): <= 90 eligible - 20 pilot
K = 5
N_NEIGHBORS, T = 5, 20.0                      # published rule ([analysis plan] c5_ladder rung 4->5)
PATIENT_RE = re.compile(r"_[A-Za-z]\d*$")       # Comp.Bio.ID minus the sample suffix = patient
FORBIDDEN_IN_FRAGMENT = ("tumor_group", "true_group", "true_absent", "discrepant", "diagnosis",
                         "evaluation", "plurality", "argmax")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_deployed():
    sys.path.insert(0, str(LLM_SRC))
    sys.path.insert(0, str(S))
    with contextlib.redirect_stdout(io.StringIO()):
        from search import analysis_lib                       # deployed payload builder
        from prosper.bayesian import compute_disease_probs    # published rule
        sim = analysis_lib.load_results_new(str(MATRIX))
        meta = analysis_lib.load_metadata(str(RICH_META))
    assert Path(analysis_lib.__file__).resolve().is_relative_to(LLM_SRC.resolve())
    import prosper.bayesian.disease_probs as dp
    assert Path(dp.__file__).resolve().is_relative_to(S)
    return analysis_lib, compute_disease_probs, sim, meta, Path(dp.__file__)


def independent_posterior(d: np.ndarray, labels: np.ndarray, n: int, T: float) -> dict:
    """Independent restatement of the published rule, written from the manuscript
    description rather than the package code: weight exp(-tau * (d_j - d_1)),
    tau = ln(T) / (d_{n+1} - d_1), summed per class over the n nearest references."""
    order = np.argsort(d, kind="stable")
    d1, dscale = d[order[0]], d[order[n]]
    gap = dscale - d1
    tau = 0.0 if gap < 1e-10 else np.log(T) / gap
    acc: dict[str, float] = collections.defaultdict(float)
    for j in order[:n]:
        acc[labels[j]] += float(np.exp(-tau * (d[j] - d1)))
    z = sum(acc.values())
    return {c: v / z for c, v in acc.items()}


def s5_original_rows() -> dict[str, dict]:
    runs = [json.loads(l) for l in (S5 / "runs.jsonl").read_text().splitlines()]
    orig, n = {}, 0
    for r in runs:
        if r["arm"] == "original" and int(r.get("repeat", 0)) == 0:
            n += 1
            if r["query"] not in orig:      # first record wins (one duplicate: 16518, same payload_sha)
                orig[r["query"]] = r
    return orig, n


def s5_payload(q: str) -> dict:
    cands = sorted((S5 / "payloads").glob(f"{q}_original*.json")) or sorted((S5 / "payloads").glob(f"{q}_*.json"))
    assert cands, f"no S5 payload for {q}"
    return json.load(open(cands[0]))


def read_pilot_ids() -> list[str]:
    return [l.strip() for l in PILOT_IDS.read_text().splitlines() if l.strip()]


def select_queries(rows: list[dict], pilot_ids: list[str]) -> tuple[list[dict], dict]:
    """E7-A5(1), deterministic, no RNG: from the eligible S5 queries remove the pilot
    queries, then every query whose patient is a pilot patient, then keep one query per
    patient (lowest numeric id). Returns the selection and an accounting record whose
    ``rule_assertions`` are (name, condition) pairs checked by main()."""
    by_q = {x["query"]: x for x in rows}
    pilot_patients = {by_q[q]["patient"] for q in pilot_ids}
    ledger, sel, kept_for = [], [], {}
    for x in sorted(rows, key=lambda x: int(x["query"])):
        if x["query"] in pilot_ids:
            ledger.append(dict(query=x["query"], status="excluded", reason="pilot_query", kept_query=""))
        elif x["patient"] in pilot_patients:
            ledger.append(dict(query=x["query"], status="excluded", reason="pilot_patient", kept_query=""))
        elif x["patient"] in kept_for:
            ledger.append(dict(query=x["query"], status="excluded", reason="duplicate_patient_lowest_id_kept",
                               kept_query=kept_for[x["patient"]]))
        else:
            kept_for[x["patient"]] = x["query"]
            sel.append({**x, "reason": "extension_remainder_lowest_id_per_patient"})
            ledger.append(dict(query=x["query"], status="selected", reason=sel[-1]["reason"], kept_query=""))
    by_reason = collections.Counter(r["reason"] for r in ledger)
    sel_ids = {x["query"] for x in sel}
    sel_patients = [x["patient"] for x in sel]
    acct = dict(
        ledger=ledger, pilot_query_ids=list(pilot_ids), pilot_patients=sorted(pilot_patients),
        n_eligible=len(rows), n_selected=len(sel),
        n_removed_pilot_query=by_reason["pilot_query"],
        n_removed_pilot_patient=by_reason["pilot_patient"],
        removed_pilot_patient=[r["query"] for r in ledger if r["reason"] == "pilot_patient"],
        n_removed_duplicate_patient=by_reason["duplicate_patient_lowest_id_kept"],
        removed_duplicate_patient=[dict(query=r["query"], kept_query=r["kept_query"]) for r in ledger
                                   if r["reason"] == "duplicate_patient_lowest_id_kept"],
        rule_assertions=[
            ("pilot list has 20 distinct ids", len(pilot_ids) == N_PILOT == len(set(pilot_ids))),
            ("every pilot id is an eligible S5 query", set(pilot_ids) <= set(by_q)),
            ("pilot patients are distinct", len(pilot_patients) == N_PILOT),
            ("n selected <= 70", len(sel) <= N_EXT_MAX),
            ("no pilot query selected", not (sel_ids & set(pilot_ids))),
            ("selected patients distinct", len(set(sel_patients)) == len(sel_patients)),
            ("selected patients disjoint from pilot patients", not (set(sel_patients) & pilot_patients)),
            ("partition of the eligible pool", len(ledger) == len(rows)
             and len(sel) + sum(by_reason[r] for r in ("pilot_query", "pilot_patient",
                                                       "duplicate_patient_lowest_id_kept")) == len(rows)),
            ("kept id is the lowest numeric id of its patient",
             all(int(kept_for[by_q[r["query"]]["patient"]]) < int(r["query"]) for r in ledger
                 if r["reason"] == "duplicate_patient_lowest_id_kept")),
        ])
    return sel, acct


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    analysis_lib, compute_disease_probs, sim, meta, dp_file = load_deployed()
    checks: dict = {"n_assertions": 0}

    def ok(cond, msg):
        assert cond, msg
        checks["n_assertions"] += 1

    ok(sim.shape == (2921, 2921), "matrix shape")
    ok(not np.isnan(sim.values).any(), "matrix has NaN")
    ok(float(np.diag(sim.values).min()) == 1e6, "diagonal sentinel")
    n_sentinel = int((sim.values >= 1e6).sum())

    # one label per reference column; duplicated metadata rows must agree
    m = meta.assign(slide_id=meta["slide_id"].astype(str))
    conflicts = m.groupby("slide_id")["tumor_group"].nunique().gt(1).sum()
    ok(conflicts == 0, "conflicting tumor_group within duplicated slide ids")
    label_of = m.drop_duplicates("slide_id").set_index("slide_id")["tumor_group"].to_dict()
    cols = [str(c) for c in sim.columns]
    col_labels = np.array([label_of.get(c, "UNKNOWN") for c in cols], dtype=object)
    n_unknown_cols = int((col_labels == "UNKNOWN").sum())
    raw = pd.read_csv(RICH_META, dtype=str)
    raw_ids = set(raw["slide_id"].astype(str))
    absent = [c for c in cols if c not in raw_ids]
    dropped = [c for c in cols if c in raw_ids and c not in label_of]
    raw_label = raw.drop_duplicates("slide_id").set_index("slide_id")["tumor_group"].to_dict()
    checks["reference_columns_without_label"] = dict(
        absent_from_metadata_csv=len(absent),
        present_in_csv_but_dropped_by_deployed_loader=len(dropped),
        dropped_groups=dict(collections.Counter(raw_label[c] for c in dropped)),
        note="deployed analysis_lib.load_metadata applies MIN_SITE_COUNT; pipeline.py reads the raw CSV")
    ok(len(absent) + len(dropped) == n_unknown_cols, "unlabelled-column decomposition")
    patient_of = {str(k): PATIENT_RE.sub("", v) for k, v in
                  raw.drop_duplicates("slide_id").set_index("slide_id")["Comp.Bio.ID"].items() if isinstance(v, str)}
    checks.update(n_sentinel_entries=n_sentinel, n_reference_columns_without_metadata=n_unknown_cols,
                  reference_class_counts=dict(collections.Counter(col_labels.tolist())))

    # candidate rows from the S5 sample (same construction as H section D.2)
    orig, n_rep0 = s5_original_rows()
    checks["s5_original_rep0_records"] = n_rep0
    rows = []
    for q, r in orig.items():
        p = s5_payload(q)
        groups = [n.get("tumor_group") for n in p["neighbors"]]
        c = collections.Counter(groups)
        top = max(c.values())
        plur = sorted(str(g) for g, v in c.items() if v == top)
        rows.append(dict(query=q, patient=patient_of[q], true_group=r["true_group"], true_absent=r["true_group"] not in groups,
                         plurality="|".join(plur), plurality_tie=len(plur) > 1,
                         top1_correct=groups[0] == r["true_group"], s5_accepted=bool(r["accepted"]),
                         has_unlabelled_neighbour=any(g is None for g in groups),
                         s5_neighbor_ids=[str(n["slide_id"]) for n in p["neighbors"]]))
    ok(len(rows) == 97, f"expected 97 distinct S5 queries, got {len(rows)}")
    checks["s5_distinct_queries"] = len(rows)
    checks["s5_true_absent_total"] = int(sum(x["true_absent"] for x in rows))

    # eligibility: every shown neighbour must carry a tumour group, otherwise the posterior
    # would expose an "UNKNOWN" class that the deployed payload renders as null
    excluded = [x["query"] for x in rows if x["has_unlabelled_neighbour"]]
    checks["excluded_unlabelled_neighbour_queries"] = excluded
    eligible = [x for x in rows if not x["has_unlabelled_neighbour"]]
    checks["excluded_queries_detail"] = [dict(query=x["query"], true_group=x["true_group"], true_absent=x["true_absent"],
                                              unlabelled_neighbours=[s for s, g in zip(x["s5_neighbor_ids"], [n.get("tumor_group") for n in s5_payload(x["query"])["neighbors"]]) if g is None])
                                         for x in rows if x["has_unlabelled_neighbour"]]
    checks["eligible_true_absent_total"] = int(sum(x["true_absent"] for x in eligible))
    ok(len(eligible) + len(excluded) == 97, "eligibility partition")
    pilot_ids = read_pilot_ids()
    sel, acct = select_queries(eligible, pilot_ids)
    for name, cond in acct["rule_assertions"]:
        ok(cond, name)

    frag, key_rows, full_rows, w_rows = {}, [], [], []
    max_dev = 0.0
    for x in sel:
        q = x["query"]
        payload = analysis_lib.build_json_for_slide(sim, meta, slide_id=q, k=K)
        shown = [str(n["slide_id"]) for n in payload["neighbors"]]
        ok(shown == x["s5_neighbor_ids"], f"{q}: neighbours differ from the saved S5 payload")
        ok(all(label_of.get(s) is not None for s in shown), f"{q}: a shown neighbour lacks metadata")
        row = sim.loc[q].drop(labels=[q])
        d = row.values.astype(np.float64)
        labs = np.array([label_of.get(str(c), "UNKNOWN") for c in row.index], dtype=object)
        order = np.argsort(d, kind="stable")
        ok([str(row.index[i]) for i in order[:K]] == shown, f"{q}: posterior neighbours != shown neighbours")
        ok(d[order[N_NEIGHBORS]] < 1e6, f"{q}: scale neighbour is a masked sentinel")
        _, df = compute_disease_probs(d[None, :], labs.tolist(), n_neighbors=N_NEIGHBORS, T=T, test_slide_ids=[q])
        probs = {c: float(df.loc[q, c]) for c in df.columns}
        ok(abs(sum(probs.values()) - 1.0) < 1e-12, f"{q}: posterior does not sum to 1")
        indep = independent_posterior(d, labs, N_NEIGHBORS, T)
        dev = max(abs(probs.get(c, 0.0) - indep.get(c, 0.0)) for c in set(probs) | set(indep))
        max_dev = max(max_dev, dev)
        ok(dev < 1e-12, f"{q}: independent restatement deviates by {dev}")
        # per-neighbour weights (depend on distances only); posterior == weights aggregated by label
        d_sorted = d[order]
        gap = d_sorted[N_NEIGHBORS] - d_sorted[0]
        tau = 0.0 if gap < 1e-10 else np.log(T) / gap
        raw = np.exp(-tau * (d_sorted[:K] - d_sorted[0])); w = raw / raw.sum()
        agg = collections.defaultdict(float)
        for j in range(K):
            agg[labs[order[j]]] += float(w[j])
        ok(max(abs(agg.get(c, 0.0) - probs.get(c, 0.0)) for c in set(agg) | set(probs)) < 1e-12, f"{q}: weights do not aggregate to posterior")
        for j in range(K):
            w_rows.append(dict(query=q, rank=j + 1, neighbor_slide_id=shown[j], distance=float(d_sorted[j]),
                               weight=float(w[j]), scale_distance_6th=float(d_sorted[N_NEIGHBORS])))
        nonzero = {c: round(p, 6) for c, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p > 1e-8}
        frag[q] = {"probabilities": nonzero}
        amax = max(probs, key=probs.get)
        srt = sorted(probs.values(), reverse=True)
        full_rows.append({"query": q, **{c: probs[c] for c in sorted(probs)}})
        key_rows.append(dict(query=q, patient=x["patient"], true_group=x["true_group"], true_absent=x["true_absent"],
                             posterior_argmax_in_plurality_set=amax in x["plurality"].split("|"),
                             plurality=x["plurality"], plurality_tie=x["plurality_tie"],
                             top1_correct=x["top1_correct"], s5_accepted=x["s5_accepted"],
                             posterior_argmax=amax, posterior_max=round(srt[0], 6),
                             posterior_margin=round(srt[0] - srt[1], 6),
                             posterior_argmax_correct=amax == x["true_group"],
                             n_classes_nonzero=len(nonzero), reason=x["reason"]))

    frag_text = json.dumps({q: frag[q] for q in sorted(frag, key=int)}, indent=2)  # keep descending class order
    for w in FORBIDDEN_IN_FRAGMENT:
        ok(w not in frag_text, f"forbidden key {w!r} in payload fragment")
    (OUT / "posterior_fragments.json").write_text(frag_text + "\n")
    # key material: every S5 query with its status (the 7 ineligible ones included for traceability)
    with open(OUT / "ext_queries.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query", "status", "reason", "kept_query"]); w.writeheader()
        led = acct["ledger"] + [dict(query=q, status="excluded", reason="unlabelled_neighbour_ineligible", kept_query="")
                                for q in excluded]
        for r in sorted(led, key=lambda r: int(r["query"])):
            w.writerow(r)
    pd.DataFrame(key_rows).to_csv(OUT / "scoring_key_DO_NOT_SEND.csv", index=False)
    # harness-facing list: ids only, sorted numerically; no repeat list (E7-A5(2): --repeats-file omitted)
    ids_sorted = sorted((x["query"] for x in sel), key=int)
    (OUT / "harness_query_ids.txt").write_text("".join(f"{q}\n" for q in ids_sorted))
    pd.DataFrame(full_rows).to_csv(OUT / "posteriors_full.csv", index=False)
    pd.DataFrame(w_rows).to_csv(OUT / "neighbor_weights.csv", index=False)

    kr = pd.DataFrame(key_rows)
    checks.update(
        rule=dict(n_neighbors=N_NEIGHBORS, T=T, k_shown=K, rounding="round 6 dp, keep > 1e-8, descending (pipeline.py)"),
        selection_rule="E7-A5(1): eligible - pilot queries - pilot patients; one per patient, lowest numeric id; no RNG",
        n_eligible=acct["n_eligible"], n_selected=len(sel),
        n_removed_pilot_query=acct["n_removed_pilot_query"],
        n_removed_pilot_patient=acct["n_removed_pilot_patient"],
        removed_pilot_patient_queries=acct["removed_pilot_patient"],
        n_removed_duplicate_patient=acct["n_removed_duplicate_patient"],
        removed_duplicate_patient=acct["removed_duplicate_patient"],
        pilot_query_ids=acct["pilot_query_ids"], pilot_ids_file=str(PILOT_IDS), pilot_ids_sha256=sha(PILOT_IDS),
        selection_reasons=dict(collections.Counter(x["reason"] for x in sel)),
        groups=dict(sorted(collections.Counter(x["true_group"] for x in sel).items())),
        n_true_absent=int(kr["true_absent"].sum()), frac_true_absent=float(kr["true_absent"].mean()),
        n_plurality_tie=int(kr["plurality_tie"].sum()),
        plurality_tie_queries=kr.loc[kr["plurality_tie"], "query"].tolist(),
        n_top1_correct=int(kr["top1_correct"].sum()), n_posterior_argmax_correct=int(kr["posterior_argmax_correct"].sum()),
        n_s5_accepted=int(kr["s5_accepted"].sum()),
        n_patients=int(kr["patient"].nunique()),
        n_selected_with_unlabelled_neighbour=int(sum(x["has_unlabelled_neighbour"] for x in sel)),
        posterior_argmax_equals_plurality_string=int((kr["posterior_argmax"] == kr["plurality"]).sum()),
        posterior_argmax_in_plurality_set=int(kr["posterior_argmax_in_plurality_set"].sum()),
        repeat_list="none (E7-A5(2): no repeat arm; harness --repeats-file is optional and omitted)",
        max_abs_dev_independent_restatement=max_dev,
        inputs={str(p): sha(p) for p in (MATRIX, RICH_META, S5 / "runs.jsonl",
                                         LLM_SRC / "search" / "analysis_lib.py", dp_file)},
        script_sha256=sha(HERE),
    )
    ok(kr["patient"].nunique() == len(sel), "one query per patient")
    ok(checks["n_selected_with_unlabelled_neighbour"] == 0, "selected query with a null-label neighbour")
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    outs = sorted(p for p in OUT.iterdir() if p.name not in ("sha256.txt", "COMPLETE"))
    (OUT / "sha256.txt").write_text("".join(f"{sha(p)}  {p.name}\n" for p in outs))
    (OUT / "COMPLETE").write_text("ok\n")
    print(json.dumps({k: v for k, v in checks.items() if k not in ("inputs",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
