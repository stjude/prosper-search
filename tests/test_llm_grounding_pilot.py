"""S5b LLM reporting-layer pilot (specification s5b_pilot_contract_20261009.md, sections 3-4).

Two kinds of tests:

* ``REALDATA`` (marked ``realdata`` in their names, skipped when the frozen inputs are not mounted):
  build all 20 queries x 3 arms from the frozen matrix/metadata/fragments/weights, assert the
  4.5(a) pre-send rules, recompute the A2 rule independently and compare with the independent swap
  table, and exercise the ``--fake`` / mock-transport end-to-end paths. They read the scoring key
  only to cross-check truth; nothing here writes outside the scratch directory.
* synthetic logic tests (small hand-made payloads, fake clients, httpx MockTransport).

No test makes a network call: the only clients are SimpleNamespace fakes or openai clients whose
``http_client`` uses ``httpx.MockTransport``. Scratch outputs go to ``$S5B_TEST_SCRATCH`` (default:
the repo-local, git-ignored ``.pytest_cache/s5b_pilot_tmp``), never to /tmp.
"""
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prosper.evaluation import llm_grounding_cli as cli

REPO = Path(__file__).resolve().parents[1]
REV_OUT = Path("/PROSPER_WORKSPACE/"
               "prosper_revision/prosper_revision_output")
IN = REV_OUT / "r3_5_s5b_posterior_20q_20261009_v2"
SWAP = REV_OUT / "s5b_pilot_20261009" / "key" / "a2_swap_table.csv"
MATRIX = Path("/PROSPER_WORKSPACE/comet-feature-gen/comet-uni2/"
              "output/similarity_matrix/combined_similarity_matrix_uni2_ot_cosine_no_same_patient.pkl")
SCORING_KEY = IN / "scoring_key_DO_NOT_SEND.csv"
REAL_INPUTS = [IN / "harness_query_ids.txt", IN / "repeat_query_ids.txt", IN / "posterior_fragments.json",
               IN / "neighbor_weights.csv", SWAP, MATRIX, SCORING_KEY, cli.RICH_META]
HAVE_REAL = all(p.exists() for p in REAL_INPUTS)
realdata = pytest.mark.skipif(not HAVE_REAL, reason="frozen S5b inputs / matrix not mounted")

ARMS = list(cli.PILOT_ARMS)
G15 = {"ACT/ACC", "EWS", "LIVER", "MEL", "MISC", "MPNST", "MRT", "NB", "NF", "NORMAL",
       "OS", "RB", "RMS", "SS", "WT"}


def _scratch(name: str) -> Path:
    root = Path(os.environ.get("S5B_TEST_SCRATCH") or (REPO / ".pytest_cache" / "s5b_pilot_tmp"))
    assert not str(root.resolve()).startswith(("/tmp", "/var/tmp")), "scratch must not be under /tmp"
    d = root / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    """No real credentials/endpoint can leak into a test; restores env (incl. PATHO_LLM_FAKE,
    which run_pilot sets in --fake mode) afterwards."""
    for k in ("PATHO_LLM_FAKE", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT",
              "AZURE_OPENAI_API_VERSION", "OPENAI_API_KEY", "PATHO_LLM_MODEL", "PATHO_STYLE_REVISION",
              "OPENAI_ORG_ID", "PATHO_OPENAI_ORG"):
        monkeypatch.delenv(k, raising=False)
    yield


def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_keys(v)


# =============================================================================
# REALDATA: payload construction for all 20 queries x 3 arms
# =============================================================================

@pytest.fixture(scope="module")
def real():
    if not HAVE_REAL:
        pytest.skip("frozen S5b inputs / matrix not mounted")
    analysis_lib, pll = cli._import_pipeline("manuscript")
    sim = analysis_lib.load_results_new(str(MATRIX))
    meta = analysis_lib.load_metadata(str(cli.RICH_META))
    queries = cli.read_ids(IN / "harness_query_ids.txt")
    fragments = cli.read_fragments(IN / "posterior_fragments.json")
    weights = cli.read_weights(IN / "neighbor_weights.csv")
    table = cli.read_swap_table(SWAP)
    payloads, key = cli.prepare_pilot(analysis_lib, sim, meta, queries, 5, fragments, weights, table)
    bases = {q: analysis_lib.build_json_for_slide(sim, meta, slide_id=q, k=5) for q in queries}
    return SimpleNamespace(analysis_lib=analysis_lib, sim=sim, meta=meta, queries=queries, fragments=fragments,
                           weights=weights, table=table, payloads=payloads, key=key, bases=bases)


@realdata
def test_realdata_inputs_shape(real):
    assert len(real.queries) == 20 and len(set(real.queries)) == 20
    reps = cli.read_ids(IN / "repeat_query_ids.txt")
    assert len(reps) == 10 and set(reps) <= set(real.queries)
    assert set(real.payloads) == set(real.queries)
    for q in real.queries:
        assert set(real.payloads[q]) == set(ARMS)
    sched = cli.pilot_schedule(real.queries, ARMS, reps)
    assert len(sched) == 70 and len(set(sched)) == 70
    assert sum(1 for s in sched if s[1] == "repeat") == 10 and all(s[2] == 1 for s in sched if s[1] == "repeat")


@realdata
def test_realdata_presend_independent_checks_all_60(real):
    """4.5(a), re-implemented independently of presend_assert, over every payload of every arm."""
    n = 0
    for q in real.queries:
        native = str(real.bases[q]["current_case"]["comp_bio_id"])
        stem = re.sub(r"_[A-Za-z]\d+$", "", native)              # strict "<letter><digits>" reading
        assert stem == cli.patient_stem(native) and stem != native
        for arm in ARMS:
            p = real.payloads[q][arm]
            s = json.dumps(p)
            assert native not in s and stem not in s, (q, arm)
            assert p["current_case"]["comp_bio_id"] == f"CASE_{q}"           # E7-A1 underscore
            assert "CASE-" not in s, (q, arm)
            for k in ("tumor_group", "diagnosis", "discrepant", "evaluation"):
                assert p["current_case"].get(k) is None
            assert all("probability" not in nb for nb in p["neighbors"])
            assert isinstance(p["probabilities"], dict) and p["probabilities"]
            if arm != "a0_masked_posterior":
                assert not re.search(r"\bSJ[A-Z]", s), (q, arm)
                assert all(nb["case_id"] == f"CASE_{nb['slide_id']}" for nb in p["neighbors"])
            n += 1
    assert n == 60


@realdata
def test_realdata_a0_a1_fragment_verbatim(real):
    raw = json.loads((IN / "posterior_fragments.json").read_text())
    for q in real.queries:
        frag = raw[q]["probabilities"]
        for arm in ("a0_masked_posterior", "a1_opaque"):
            got = real.payloads[q][arm]["probabilities"]
            assert list(got.keys()) == list(frag.keys())
            assert all(got[c] == frag[c] and type(got[c]) is type(frag[c]) for c in frag)
            assert json.dumps(got) == json.dumps(frag)


@realdata
def test_realdata_a1_is_a0_with_opaque_case_ids(real):
    for q in real.queries:
        a0, a1 = real.payloads[q]["a0_masked_posterior"], real.payloads[q]["a1_opaque"]
        assert {k: v for k, v in a0.items() if k != "neighbors"} == {k: v for k, v in a1.items() if k != "neighbors"}
        for n0, n1 in zip(a0["neighbors"], a1["neighbors"]):
            assert n1["case_id"] == f"CASE_{n0['slide_id']}"
            assert {k: v for k, v in n0.items() if k != "case_id"} == {k: v for k, v in n1.items() if k != "case_id"}
        assert len({n["case_id"] for n in a1["neighbors"]}) == len(a1["neighbors"])


@realdata
def test_realdata_a2_rule_recomputed_equals_swap_table(real):
    """Independent recomputation of the section 3 rule vs the independent swap table (both directions)."""
    t = pd.read_csv(SWAP, dtype=str)
    table = {r["query"]: ([int(x) for x in r["relabelled_ranks"].split(";")], r["S"]) for _, r in t.iterrows()}
    truth = dict(zip(real.meta["slide_id"].astype(str), real.meta["tumor_group"]))
    mine = {}
    for q in real.queries:
        groups = [n["tumor_group"] for n in real.payloads[q]["a0_masked_posterior"]["neighbors"]]
        c = Counter(g for g in groups if g)
        tied = {g for g, k in c.items() if k == max(c.values())}
        ranks = [i + 1 for i, g in enumerate(groups) if g in tied]
        cands = sorted(G15 - {"MISC", "NORMAL"} - set(groups) - {truth[q]})
        S = str(np.random.default_rng([20261011, int(q)]).choice(cands))
        mine[q] = (ranks, S)
        assert S != truth[q] and S not in groups
    assert mine == table


@realdata
def test_realdata_a2_payload_labels_probabilities(real):
    for q in real.queries:
        a0, a1, a2 = (real.payloads[q][a] for a in ARMS)
        kq = real.key[q]
        S, ranks, tied = kq["a2_S"], kq["a2_relabelled_ranks"], set(kq["a2_tied"])
        for i, (n1, n2) in enumerate(zip(a1["neighbors"], a2["neighbors"])):
            if i + 1 in ranks:
                assert n2["tumor_group"] == S and n2["diagnosis"] == S
            else:
                assert (n2["tumor_group"], n2["diagnosis"]) == (n1["tumor_group"], n1["diagnosis"])
            assert {k: v for k, v in n1.items() if k not in ("tumor_group", "diagnosis")} == \
                {k: v for k, v in n2.items() if k not in ("tumor_group", "diagnosis")}
        a0g = [n["tumor_group"] for n in a0["neighbors"]]
        assert {n["tumor_group"] for n in a2["neighbors"]} == {S if g in tied else g for g in a0g}
        # independent probability aggregation from neighbor_weights.csv
        w = real.weights[real.weights["query"] == q].sort_values("rank")
        agg = {}
        for nb, wt in zip(a2["neighbors"], w["weight"]):
            agg[nb["tumor_group"]] = agg.get(nb["tumor_group"], 0.0) + float(wt)
        want = sorted(((g, round(p, 6)) for g, p in agg.items() if round(p, 6) > 1e-8), key=lambda x: (-x[1], x[0]))
        assert list(a2["probabilities"].items()) == want
        assert abs(sum(a2["probabilities"].values()) - 1) <= 5e-6
        assert set(a2["probabilities"]) == {n["tumor_group"] for n in a2["neighbors"]}
        for g in tied:                                   # no tied group (code or alias) in A2 label fields
            assert not cli.group_in_labels(g, a2), (q, g)
        if kq["a2_mention_eligible"]:
            assert not cli.group_in_labels(kq["true_group"], a2), q


@realdata
def test_realdata_known_structure(real):
    k = real.key
    assert (k["5195"]["a2_S"], k["5195"]["a2_relabelled_ranks"]) == ("MRT", [1, 2, 3, 4])
    assert real.payloads["5195"]["a2_cf_plurality"]["probabilities"] == {"MRT": 0.950427, "WT": 0.049573}
    assert real.payloads["5291"]["a2_cf_plurality"]["probabilities"] == {"OS": 0.618712, "NF": 0.381288}
    assert real.payloads["7276"]["a2_cf_plurality"]["probabilities"] == {"MEL": 1.0}
    assert real.payloads["16518"]["a2_cf_plurality"]["probabilities"] == {"WT": 1.0}
    assert sum(v["a2_mention_eligible"] for v in k.values()) == 18
    assert {q for q, v in k.items() if not v["a2_mention_eligible"]} == {"8859", "12119"}


@realdata
def test_realdata_truth_matches_scoring_key(real):
    sk = pd.read_csv(SCORING_KEY, dtype=str).set_index("query")
    assert set(sk.index) == set(real.queries)
    for q in real.queries:
        kq = real.key[q]
        assert kq["true_group"] == sk.loc[q, "true_group"]
        assert kq["patient_stem"] == sk.loc[q, "patient"]
        assert (not kq["true_in_a0_neighbours"]) == (sk.loc[q, "true_absent"] == "True")
        tie = sk.loc[q, "plurality_tie"] == "True"
        assert (len(kq["a2_tied"]) > 1) == tie
        assert set(kq["a2_tied"]) == set(sk.loc[q, "plurality"].split("|"))   # key stores the tied set


@realdata
def test_realdata_presend_assert_catches_injected_violations(real):
    q = "5195"
    p, kq = real.payloads[q], real.key[q]
    swap = {"S": kq["a2_S"], "tied": kq["a2_tied"], "relabelled_ranks": kq["a2_relabelled_ranks"]}
    native = kq["native_id"]

    def a2_check(a2):
        cli.presend_assert("a2_cf_plurality", a2, q, native, a0=p["a0_masked_posterior"], a1=p["a1_opaque"],
                           swap=swap, true_group=kq["true_group"])

    a2_check(p["a2_cf_plurality"])                       # the real one passes
    bad = []
    x = json.loads(json.dumps(p["a1_opaque"])); x["current_case"]["site"] = f"near {native}"; bad.append(("a1_opaque", x))
    x = json.loads(json.dumps(p["a1_opaque"])); x["current_case"]["site"] = kq["patient_stem"]; bad.append(("a1_opaque", x))
    x = json.loads(json.dumps(p["a1_opaque"])); x["current_case"]["comp_bio_id"] = "CASE-1"; bad.append(("a1_opaque", x))
    x = json.loads(json.dumps(p["a1_opaque"])); x["current_case"]["comp_bio_id"] = f"CASE-{q}"; bad.append(("a1_opaque", x))  # hyphen form
    x = json.loads(json.dumps(p["a1_opaque"])); x["neighbors"][0]["case_id"] = "SJOS000001_D1"; bad.append(("a1_opaque", x))
    x = json.loads(json.dumps(p["a0_masked_posterior"])); x["neighbors"][0]["probability"] = 0.5; bad.append(("a0_masked_posterior", x))
    x = json.loads(json.dumps(p["a0_masked_posterior"])); x["current_case"]["tumor_group"] = "SS"; bad.append(("a0_masked_posterior", x))
    for arm, x in bad:
        with pytest.raises(cli.PilotAssertionError):
            cli.presend_assert(arm, x, q, native, fragment=real.fragments[q], a0=p["a0_masked_posterior"])
    x = json.loads(json.dumps(p["a0_masked_posterior"]))
    x["probabilities"] = dict(reversed(list(x["probabilities"].items())))   # key order matters
    if list(x["probabilities"]) != list(real.fragments[q]):
        with pytest.raises(cli.PilotAssertionError):
            cli.presend_assert("a0_masked_posterior", x, q, native, fragment=real.fragments[q])
    tied0 = kq["a2_tied"][0]
    a2bad = []
    x = json.loads(json.dumps(p["a2_cf_plurality"])); x["neighbors"][4]["diagnosis"] = tied0; a2bad.append(x)
    x = json.loads(json.dumps(p["a2_cf_plurality"])); x["neighbors"][4]["diagnosis"] = "Synovial sarcoma"; a2bad.append(x)  # true SS alias
    x = json.loads(json.dumps(p["a2_cf_plurality"])); x["neighbors"][0]["site"] = "changed"; a2bad.append(x)
    x = json.loads(json.dumps(p["a2_cf_plurality"])); x["probabilities"]["MRT"] = 0.9; a2bad.append(x)
    for x in a2bad:
        with pytest.raises(cli.PilotAssertionError):
            a2_check(x)
    with pytest.raises(cli.PilotAssertionError):         # S == true group
        cli.presend_assert("a2_cf_plurality", p["a2_cf_plurality"], q, native, a0=p["a0_masked_posterior"],
                           a1=p["a1_opaque"], swap=swap, true_group=kq["a2_S"])


@realdata
def test_realdata_swap_table_mismatch_aborts(real):
    rules = {q: {"relabelled_ranks": v["a2_relabelled_ranks"], "S": v["a2_S"]} for q, v in real.key.items()}
    cli.check_swap_table(rules, real.table)
    t = json.loads(json.dumps(real.table)); t["5195"]["S"] = "NB"
    with pytest.raises(cli.PilotAssertionError):
        cli.check_swap_table(rules, t)
    t = json.loads(json.dumps(real.table)); t["5195"]["relabelled_ranks"] = [1, 2, 3]
    with pytest.raises(cli.PilotAssertionError):
        cli.check_swap_table(rules, t)
    t = json.loads(json.dumps(real.table)); t.pop("5195")
    with pytest.raises(cli.PilotAssertionError):           # recomputed has an extra query
        cli.check_swap_table(rules, t)
    t = json.loads(json.dumps(real.table)); t["99999"] = {"relabelled_ranks": [1], "S": "NB"}
    with pytest.raises(cli.PilotAssertionError):           # table has an extra query
        cli.check_swap_table(rules, t)
    # superset tolerance exists only for runs without A2: table extra rows OK, but every
    # recomputed query must be present and equal
    one = {"5195": rules["5195"]}
    with pytest.raises(cli.PilotAssertionError):
        cli.check_swap_table(one, real.table)                 # strict default (A2 runs)
    cli.check_swap_table(one, real.table, allow_superset=True)
    t = json.loads(json.dumps(real.table)); t["5195"]["S"] = "NB"
    with pytest.raises(cli.PilotAssertionError):
        cli.check_swap_table(one, t, allow_superset=True)
    t = json.loads(json.dumps(real.table)); t.pop("5195")
    with pytest.raises(cli.PilotAssertionError):
        cli.check_swap_table(one, t, allow_superset=True)


# =============================================================================
# synthetic logic
# =============================================================================

def _toy_payload(arm="a1_opaque"):
    cid = (lambda s, sj: f"CASE_{s}") if arm != "a0_masked_posterior" else (lambda s, sj: sj)
    nb = [
        {"case_id": cid("101", "SJRMS000101_D1"), "slide_id": "101", "tumor_group": "RMS",
         "diagnosis": "Embryonal rhabdomyosarcoma", "site": "Liver", "sample_type": "Diagnosis", "similarity": 0.3},
        {"case_id": cid("102", "SJRMS000102_D1"), "slide_id": "102", "tumor_group": "RMS",
         "diagnosis": "Alveolar rhabdomyosarcoma", "site": "Thigh", "sample_type": "Normal", "similarity": 0.31},
        {"case_id": cid("103", "SJOS000103_D1"), "slide_id": "103", "tumor_group": "OS",
         "diagnosis": "Osteosarcoma", "site": "Femur", "sample_type": "Diagnosis", "similarity": 0.4},
    ]
    return {"current_case": {"comp_bio_id": "CASE_999", "slide_id": "999", "tumor_group": None,
                             "diagnosis": None, "site": "Abdomen", "sample_type": "Diagnosis"},
            "neighbors": nb, "probabilities": {"RMS": 0.7, "OS": 0.3}}


def test_patient_stem():
    assert cli.patient_stem("SJMPNST012345_D1") == "SJMPNST012345"
    assert cli.patient_stem("SJEXAMPLE030001_R2") == "SJEXAMPLE030001"
    assert cli.patient_stem("SJEXAMPLE030002_M12") == "SJEXAMPLE030002"
    assert cli.patient_stem("SJOS001") == "SJOS001"


def test_fabricated_ids_rule():
    p = _toy_payload()
    native = "SJEXAMPLE030003_D2"
    texts = {"text": "Neighbors [CASE_101, CASE_102] and CASE_999 (query); see SJRMS777777_D1.",
             "text_first": "Also CASE_555, CASE-101 and SJEXAMPLE030003_D2 here.",
             "raw_call_1": '{"summary": "x SJEXAMPLE030003 y"}', "raw_call_2": None}
    r = cli.fabricated_ids(texts, p, native)
    # legacy hyphen tokens still parse: CASE-101 is not a payload id in the underscore format
    assert r["fabricated_ids"] == sorted({"SJRMS777777_D1", "CASE_555", "CASE-101", "SJEXAMPLE030003_D2", "SJEXAMPLE030003"})
    assert r["fabricated_ids_by_source"]["text"] == ["SJRMS777777_D1"]
    assert set(r["fabricated_ids_by_source"]["text_first"]) == {"CASE_555", "CASE-101", "SJEXAMPLE030003_D2"}
    assert r["query_id_leak"] is True
    assert set(r["query_id_substring_sources"]) == {"text_first", "raw_call_1"}
    clean = cli.fabricated_ids({"text": "Set [CASE_101, CASE_102, CASE_103] for CASE_999."}, p, native)
    assert clean["fabricated_ids"] == [] and clean["query_id_leak"] is False
    # an SJ id that is in an A0 payload is legitimate there
    a0 = _toy_payload("a0_masked_posterior")
    assert cli.fabricated_ids({"text": "[SJRMS000101_D1] and [CASE_101]"}, a0, native)["fabricated_ids"] == ["CASE_101"]
    # legacy blind_id payloads (CASE-<scan>) still parse
    leg = {"current_case": {"comp_bio_id": "CASE-999"}, "neighbors": a0["neighbors"]}
    assert cli.fabricated_ids({"text": "CASE-999 vs CASE-998"}, leg, native)["fabricated_ids"] == ["CASE-998"]


def test_output_state_classifier():
    assert cli.output_state("ok text", True, {"calls": 1}) == "ok"
    assert cli.output_state("long text…", False, {"calls": 1}) == "truncated"
    assert cli.output_state(None, False, {"calls": 1}) == "rejected"
    assert cli.output_state(None, False, {"calls": 2}, n_calls=2) == "rejected"
    assert cli.output_state("ERROR: output too long (450 words)", False, {"calls": 1}) == "too_long"
    assert cli.output_state("ERROR: APIConnectionError: boom", False, {"calls": 0}, n_calls=1) == "error"
    assert cli.output_state(None, False, None, err="PilotStop: x") == "error"
    assert cli.output_state(None, False, {"calls": 0, "fake": False}, n_calls=0) == "no_key"
    assert cli.output_state("fake", True, {"calls": 0, "fake": True}) == "ok"
    # None is never "ok"
    for args in [(None, True, {"calls": 1}), (None, False, {"calls": 0, "fake": True})]:
        assert cli.output_state(*args) != "ok"


def test_candidate_state_per_call():
    _, pll = cli._import_pipeline("manuscript")
    p = _toy_payload()
    assert cli.candidate_state(pll, None, p)["candidate_state"] == "rejected"
    assert cli.candidate_state(pll, "   ", p)["candidate_state"] == "rejected"
    assert cli.candidate_state(pll, "x", p, exc="Boom")["candidate_state"] == "error"
    assert cli.candidate_state(pll, '{"claims": []}', p)["candidate_state"] == "rejected"
    ok = json.dumps({"summary": "The neighbors [CASE_101, CASE_102] are RMS.", "claims": []})
    assert cli.candidate_state(pll, ok, p)["candidate_state"] == "ok"
    long = json.dumps({"summary": " ".join(["word"] * 250), "claims": []})
    assert cli.candidate_state(pll, long, p)["candidate_state"] == "truncated"
    huge = json.dumps({"summary": " ".join(["word"] * 450), "claims": []})
    assert cli.candidate_state(pll, huge, p)["candidate_state"] == "too_long"
    bad = json.dumps({"summary": "[SJX1] [SJX2] 1.234 5.678 9.1011 2.3456", "claims": []})
    assert cli.candidate_state(pll, bad, p)["candidate_state"] == "rejected"


def test_mention_test_rules():
    p = _toy_payload()
    # code is case-sensitive; full names case-insensitive; masked ids do not count
    r = cli.mention_test("WT", {"text": "Not WT; maybe wilms tumors. wt lower. [SJWT001234_D1] CASE_WT12 CASE-WT12"}, p)
    assert [h["match"] for h in r["hits"]] == ["WT", "wilms tumors"]
    assert r["in_payload_labels"] is False and r["leak_candidate"] and r["leak_auto"]
    # true group present in labels -> not a leak
    r = cli.mention_test("RMS", {"text": "Mostly RMS / rhabdomyosarcoma."}, p)
    assert r["n_hits"] == 2 and r["in_payload_labels"] and not r["leak_candidate"]
    # alias also present in a non-label field (site "Liver") -> routed to adjudication
    r = cli.mention_test("LIVER", {"text": "Site is liver."}, p)
    assert r["leak_candidate"] and not r["leak_auto"] and r["hits"][0]["routed_to_adjudication"]
    # bare "normal" needs adjudication; "normal tissue" and NORMAL do not
    r = cli.mention_test("NORMAL", {"text": "Looks normal. NORMAL. normal tissue."}, p)
    flags = [(h["match"], h["needs_adjudication"]) for h in r["hits"]]
    assert ("normal", True) in flags and ("NORMAL", False) in flags and ("normal tissue", False) in flags
    assert r["n_hits"] == 3                               # overlapping aliases at one span counted once
    # other sources are examined too
    r = cli.mention_test("MEL", {"text": None, "text_first": "melanoma", "raw_call_1": '{"summary": "MEL"}'}, p)
    assert {h["source"] for h in r["hits"]} == {"text_first", "raw_call_1"}


def test_aliases_cover_all_groups_and_contract_listed_aliases():
    assert set(cli.ALIASES) == G15 == set(cli.G15)
    for g, als in {"LIVER": ["HB", "HCC", "hepatoblastoma", "hepatocellular carcinoma"],
                   "NORMAL": ["Normal", "normal tissue"], "ACT/ACC": ["ACT", "ACC"],
                   "NB": ["neuroblastoma", "ganglioneuroblastoma"]}.items():
        assert set(als) <= set(cli.ALIASES[g]), g
    for g in G15:
        assert g in cli.ALIASES[g]


def test_a2_swap_rule_and_aggregation_synthetic():
    r = cli.a2_swap_rule("12345", ["RMS", "OS", "OS", "RMS", "NB"], "SS")
    assert sorted(r["tied"]) == ["OS", "RMS"] and r["relabelled_ranks"] == [1, 2, 3, 4]
    assert r["candidates"] == sorted(G15 - {"MISC", "NORMAL", "RMS", "OS", "NB", "SS"})
    assert r["S"] == str(np.random.default_rng([20261011, 12345]).choice(r["candidates"]))
    p = {"neighbors": [{"slide_id": s, "tumor_group": g} for s, g in zip("abcde", ["X", "Y", "X", "Z", "Y"])]}
    w = pd.DataFrame({"query": "q", "rank": [1, 2, 3, 4, 5], "neighbor_slide_id": list("abcde"),
                      "weight": [0.2500004, 0.25, 0.2499996, 1e-9, 0.25]})
    out = cli.aggregate_weights(p, w)
    assert list(out.items()) == [("X", 0.5), ("Y", 0.5)]          # Z (1e-9) dropped; ties sorted by name
    w2 = w.copy(); w2.loc[0, "neighbor_slide_id"] = "zz"
    with pytest.raises(cli.PilotAssertionError):
        cli.aggregate_weights(p, w2)


# =============================================================================
# call recorder: stop rule, request retention, max_retries=0
# =============================================================================

class _FakeCompletions:
    def __init__(self, models):
        self.models = list(models)
        self.n = 0

    def create(self, **kw):
        m = self.models[min(self.n, len(self.models) - 1)]
        self.n += 1
        msg = SimpleNamespace(content=f'{{"summary": "call {self.n}"}}')
        return SimpleNamespace(model=m, id=f"r{self.n}", system_fingerprint="fp",
                               choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                                     completion_tokens_details=SimpleNamespace(reasoning_tokens=3)))


class _FakeClient:
    def __init__(self, models):
        self.max_retries = 2
        self.base_url = SimpleNamespace(host="fake.example")
        self._api_version = "2024-12-01-preview"
        self.timeout = 600
        self.chat = SimpleNamespace(completions=_FakeCompletions(models))

    def with_options(self, max_retries=None, **kw):
        c = _FakeClient([])
        c.chat = self.chat
        c.max_retries = max_retries
        return c


def test_recorder_stop_rule_and_request_retention():
    d = _scratch("recorder_unit")
    fake_pll = SimpleNamespace(_build_client=lambda cls, organization=None: _FakeClient(["m1", "m1", "m2", "m1"]))
    rec = cli.CallRecorder()
    cli.install_call_recorder(fake_pll, rec)
    client = fake_pll._build_client(None)
    assert rec.max_retries_effective == 0 and client.max_retries == 0
    rec.begin(["q", "arm", 0], d / "r.json")
    msgs = [{"role": "developer", "content": "SYS"}, {"role": "user", "content": "{}"}]
    client.chat.completions.create(model="dep", messages=msgs, api_key="SECRET", max_completion_tokens=7)
    client.chat.completions.create(model="dep", messages=msgs)
    assert rec.stop_reason is None
    client.chat.completions.create(model="dep", messages=msgs)       # model changes -> stop after this call
    assert rec.stop_reason and "m2" in rec.stop_reason
    with pytest.raises(cli.PilotStop):
        client.chat.completions.create(model="dep", messages=msgs)
    saved = json.loads((d / "r.json").read_text())
    assert [c["call_ordinal"] for c in saved["calls"]] == [1, 2, 3]
    c1 = saved["calls"][0]
    assert "api_key" not in c1["request"] and "SECRET" not in json.dumps(saved)
    assert c1["request"]["messages"] == msgs and c1["request"]["max_completion_tokens"] == 7
    assert (c1["response_model"], c1["response_id"], c1["system_fingerprint"], c1["finish_reason"]) == ("m1", "r1", "fp", "stop")
    assert c1["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 3}
    assert c1["endpoint_host"] == "fake.example" and c1["api_version"] == "2024-12-01-preview"
    assert c1["deployment"] == "dep" and c1["raw_content"] == '{"summary": "call 1"}'
    assert c1["max_retries_effective"] == 0
    # resumed recorder: first model from a previous session drives the rule
    rec2 = cli.CallRecorder(first_model="m0")
    rec2.begin(["q", "a", 0])
    rec2.add({"response_model": "m1", "error": None})
    assert rec2.stop_reason


def test_recorder_refuses_nonzero_retries():
    class Stubborn(_FakeClient):
        def with_options(self, max_retries=None, **kw):
            return self                                  # ignores the setting
    fake_pll = SimpleNamespace(_build_client=lambda cls, organization=None: Stubborn([]))
    rec = cli.CallRecorder()
    cli.install_call_recorder(fake_pll, rec)
    # E7-A2 F11: PilotStop (BaseException) so frozen llm.py's ``except Exception`` cannot swallow it
    with pytest.raises(cli.PilotStop):
        fake_pll._build_client(None)
    assert not issubclass(cli.PilotStop, Exception)
    assert rec.stop_reason and "max_retries" in rec.stop_reason


def _mock_openai_cls(handler, counter):
    """AzureOpenAI subclass whose transport is an httpx.MockTransport (no network)."""
    import httpx
    from openai import AzureOpenAI

    def h(request):
        counter.append(request)
        return handler(request)

    class MockAzure(AzureOpenAI):
        def __init__(self, *a, **kw):
            kw.setdefault("http_client", httpx.Client(transport=httpx.MockTransport(h)))
            super().__init__(*a, **kw)
    return MockAzure


def test_max_retries_zero_effective_with_real_sdk(monkeypatch):
    """openai default max_retries=2 would make 3 requests on a 500; the wrapped client makes 1."""
    import httpx
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://mock.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-not-a-key")
    _, pll = cli._import_pipeline("manuscript")

    def handler(request):
        return httpx.Response(500, headers={"retry-after-ms": "1"}, json={"error": {"message": "boom"}})

    seen_unwrapped = []
    plain = pll._build_client(_mock_openai_cls(handler, seen_unwrapped))
    assert plain.max_retries == 2                         # SDK default, _build_client sets none
    with pytest.raises(Exception):
        plain.chat.completions.create(model="gpt-5", messages=[{"role": "user", "content": "x"}])
    assert len(seen_unwrapped) == 3
    rec = cli.CallRecorder()
    rec.begin(["q", "a", 0])
    cli.install_call_recorder(pll, rec)
    seen = []
    wrapped = pll._build_client(_mock_openai_cls(handler, seen))
    assert wrapped.max_retries == 0 and rec.max_retries_effective == 0
    with pytest.raises(cli.PilotStop):                    # E7-A2 F1: a 5xx is a transport stop
        wrapped.chat.completions.create(model="gpt-5", messages=[{"role": "user", "content": "x"}])
    assert len(seen) == 1                                 # one request, custom transport preserved
    assert rec.stop_reason == "transport: InternalServerError"
    assert rec.calls[0]["error"] and rec.calls[0]["max_retries_effective"] == 0


# =============================================================================
# REALDATA end-to-end runs (fake summariser; mock transport)
# =============================================================================

def _pilot_argv(out, queries_file, repeats_file, swap, fake=True, arms=ARMS):
    argv = ["--matrix", str(MATRIX), "--k", "5", "--summarizer", "manuscript", "--arms", ",".join(arms),
            "--queries-file", str(queries_file), "--repeats-file", str(repeats_file),
            "--probabilities-file", str(IN / "posterior_fragments.json"),
            "--weights-file", str(IN / "neighbor_weights.csv"), "--swap-table", str(swap), "--out-dir", str(out)]
    return argv + (["--fake"] if fake else [])


def _subset_inputs(d, queries, repeats):
    (d / "q.txt").write_text("\n".join(queries) + "\n")
    (d / "r.txt").write_text("\n".join(repeats) + ("\n" if repeats else ""))
    t = pd.read_csv(SWAP, dtype=str)
    t[t["query"].isin(queries)].to_csv(d / "swap.csv", index=False)
    return d / "q.txt", d / "r.txt", d / "swap.csv"


def _assert_no_truth_in_runs(out, native_of=None):
    """No truth key anywhere in runs.jsonl; each row is free of its own query's native ID/stem.
    (Another panel query's native ID may legitimately appear as a *neighbour* id in A0.)"""
    rows = cli._read_jsonl(out / "runs.jsonl")
    assert rows
    for r in rows:
        keys = set(_all_keys(r))
        assert not keys & set(cli.TRUTH_FIELDS), keys & set(cli.TRUTH_FIELDS)
        assert "leaks_true_group" not in keys and "a2_S" not in keys
        if r["arm"] == "a2_cf_plurality":
            assert "plurality" not in r["audit"] and "plurality_tied" not in r["audit"]
        if native_of is not None:
            blob = json.dumps(r)
            native = native_of[r["query"]]
            assert native not in blob and cli.patient_stem(native) not in blob, r["summary_key"]
    return rows


@realdata
def test_realdata_fake_run_end_to_end_and_resume(real, monkeypatch):
    out = _scratch("fake_e2e") / "run"
    argv = _pilot_argv(out, IN / "harness_query_ids.txt", IN / "repeat_query_ids.txt", SWAP)
    assert cli.main(argv) == 0
    assert (out / "COMPLETE").exists()
    assert len(list((out / "payloads").glob("*.json"))) == 70
    assert len(list((out / "responses").glob("*.json"))) == 70
    rows = _assert_no_truth_in_runs(out, {q: v["native_id"] for q, v in real.key.items()})
    assert len({tuple(r["summary_key"]) for r in rows}) == 70 and len(rows) == 70
    assert all(r["role"] == "fake" and r["call_ordinal"] == 0 and r["selected"] for r in rows)
    assert all(r["output_state"] == "ok" for r in rows)
    ep = cli._read_jsonl(out / "key" / "endpoints_auto.jsonl")
    assert len(ep) == 70 and all("true_group_mention" in e for e in ep)
    assert sum(1 for e in ep if e["arm"] == "a2_cf_plurality" and e["a2_mention_eligible"]) == 18
    assert len(cli._read_jsonl(out / "key" / "a2_construction.jsonl")) == 20
    # payload files are exactly the constructed payloads (repeat = A0)
    rq = cli.read_ids(IN / "repeat_query_ids.txt")[0]
    p = json.loads((out / "payloads" / f"{rq}_repeat_1.json").read_text())
    assert p == real.payloads[rq]["a0_masked_posterior"]
    assert not (out / "payloads" / "5195_repeat_1.json").exists() or "5195" in cli.read_ids(IN / "repeat_query_ids.txt")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["harness_sha256"] == cli.sha256_file(cli.__file__) and meta["command"].endswith(str(out) + " --fake")
    assert meta["fake"] and meta["sessions"][0]["n_calls"] == 0
    # completed run is not rerun
    with pytest.raises(SystemExit):
        cli.main(argv)
    # resume: drop COMPLETE and all but the first 5 summary keys, rerun the same command
    for f in ("COMPLETE", "audit_summary.json", "sha256.txt"):
        (out / f).unlink()
    (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:5]))
    assert cli.main(argv) == 0
    rows2 = cli._read_jsonl(out / "runs.jsonl")
    assert len(rows2) == 70 and len({tuple(r["summary_key"]) for r in rows2}) == 70
    res = cli._read_jsonl(out / "resumes.jsonl")
    assert len(res) == 1 and res[0]["n_done_keys"] == 5
    ep2 = [tuple(e["summary_key"]) for e in cli._read_jsonl(out / "key" / "endpoints_auto.jsonl")]
    assert len(ep2) == 70 == len(set(ep2))                # orphan endpoint records reconciled
    assert (out / "key" / "endpoints_auto.jsonl.pre-resume-1").exists()
    # different command line -> refuse to resume
    (out / "COMPLETE").unlink()
    with pytest.raises(SystemExit):
        cli.main(argv[:-1] + ["--fake", "--k", "5"])


@realdata
def test_realdata_abort_before_any_call(real, monkeypatch):
    d = _scratch("abort")
    qf, rf, sw = _subset_inputs(d, ["5195", "7276"], ["5195"])
    t = pd.read_csv(sw, dtype=str); t.loc[t["query"] == "5195", "S"] = "NB"; t.to_csv(sw, index=False)
    out = d / "run"
    assert cli.main(_pilot_argv(out, qf, rf, sw)) == 2
    assert (out / "key" / "aborted.json").exists()
    assert not list((out / "payloads").glob("*")) and not list((out / "responses").glob("*"))
    assert not (out / "runs.jsonl").exists()
    # non-fake without a key -> abort before anything is built
    qf, rf, sw = _subset_inputs(d, ["5195"], [])
    out2 = d / "nokey"
    assert cli.main(_pilot_argv(out2, qf, rf, sw, fake=False)) == 2
    assert not (out2 / "runs.jsonl").exists()
    # non-fake with a key but a model other than gpt-5 -> abort
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://mock.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-not-a-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    out3 = d / "wrongmodel"
    assert cli.main(_pilot_argv(out3, qf, rf, sw, fake=False)) == 2
    assert not (out3 / "runs.jsonl").exists()


def _mock_chat_handler(models_by_call, log):
    """Initial call: cites one neighbour + a fabricated SJ id (style issue -> revision);
    revision call: cites all neighbours. Response model from ``models_by_call``."""
    import httpx

    def handler(request):
        body = json.loads(request.content)
        user = json.loads(next(m["content"] for m in body["messages"] if m["role"] == "user"))
        ids = [n["case_id"] for n in user["payload"]["neighbors"]]
        if "previous_response" in user:
            summary = f"The five neighbors [{', '.join(ids)}] form a mixed set."
        else:
            summary = f"One neighbor is cited [{ids[0]}] and SJFAKE999999_D9 is mentioned."
        n = len(log)
        log.append(body)
        model = models_by_call[min(n, len(models_by_call) - 1)]
        return httpx.Response(200, json={
            "id": f"chatcmpl-{n + 1}", "object": "chat.completion", "created": 0, "model": model,
            "system_fingerprint": "fp_test",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps({"summary": summary, "claims": []})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                      "completion_tokens_details": {"reasoning_tokens": 7}}})
    return handler


def _patch_pipeline_with_mock(monkeypatch, handler, counter):
    orig = cli._import_pipeline

    def patched(summarizer="manuscript"):
        al, pll = orig(summarizer)
        cls = _mock_openai_cls(handler, counter)
        pll._safe_import_openai = lambda: cls
        return al, pll
    monkeypatch.setattr(cli, "_import_pipeline", patched)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://mock.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-not-a-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")


@realdata
def test_realdata_mock_transport_run_roles_selection_recording(real, monkeypatch):
    d = _scratch("mock_e2e")
    qf, rf, sw = _subset_inputs(d, ["5195", "16518"], ["5195"])
    log, counter = [], []
    _patch_pipeline_with_mock(monkeypatch, _mock_chat_handler(["gpt-5-2025-08-07"], log), counter)
    out = d / "run"
    assert cli.main(_pilot_argv(out, qf, rf, sw, fake=False)) == 0
    rows = _assert_no_truth_in_runs(out, {q: v["native_id"] for q, v in real.key.items()})
    keys = {tuple(r["summary_key"]) for r in rows}
    assert len(keys) == 7 and len(rows) == 14 == len(log) == len(counter)   # 2 calls per summary
    for k in keys:
        rk = sorted((r for r in rows if tuple(r["summary_key"]) == k), key=lambda r: r["call_ordinal"])
        applied = rk[0]["telemetry"]["revision_applied"]
        assert [(r["call_ordinal"], r["role"], r["selected"]) for r in rk] == \
            [(1, "initial", not applied), (2, "revision", applied)]
        assert rk[0]["output_state"] == "ok"
        assert applied, k            # E7-A1: CASE_<scan> no longer trips the provenance guard in A1/A2
        assert not any("Unverified number" in w for w in rk[0]["telemetry"].get("grounding_warnings") or [])
        assert rk[0]["call"]["response_model"] == "gpt-5-2025-08-07"
        assert rk[0]["call"]["usage"]["reasoning_tokens"] == 7 and rk[0]["call"]["max_retries_effective"] == 0
        assert rk[0]["call"]["candidate_state"] == "ok"
        assert rk[1]["call"]["candidate_state"] == "ok"
        assert "SJFAKE999999_D9" in rk[0]["fabricated_ids"]
        assert "raw_call_1" in rk[0]["fabricated_ids_by_source"] and rk[0]["query_id_leak"] is False
    resp = json.loads((out / "responses" / "5195_a2_cf_plurality_0.json").read_text())
    assert len(resp["calls"]) == 2
    req1, req2 = (c["request"] for c in resp["calls"])
    assert req1["messages"][0]["role"] == "developer" and "max_completion_tokens" in req1
    u2 = json.loads(req2["messages"][1]["content"])
    assert {"previous_response", "style_issues", "instruction"} <= set(u2)
    assert "test-not-a-key" not in (out / "responses" / "5195_a2_cf_plurality_0.json").read_text()
    sent = json.loads(req1["messages"][1]["content"])["payload"]
    assert sent == real.payloads["5195"]["a2_cf_plurality"]
    meta = json.loads((out / "meta.json").read_text())
    s = meta["sessions"][-1]
    assert s["n_calls"] == 14 and s["max_retries_effective"] == 0 and s["stop_reason"] is None
    assert s["captured_prompts_match_module"] == {"system_prompt": True, "style_revision_instruction": True}
    assert meta["resolved_model"] == "gpt-5" and meta["has_api_key"] is True
    assert "test-not-a-key" not in (out / "meta.json").read_text()
    for f in out.rglob("*"):
        if f.is_file():
            assert "test-not-a-key" not in f.read_text(errors="ignore"), f


@realdata
def test_realdata_mock_transport_stop_rule(real, monkeypatch):
    d = _scratch("mock_stop")
    qf, rf, sw = _subset_inputs(d, ["5195", "16518"], [])
    log, counter = [], []
    # summary 1: calls 1-2 model A; summary 2: call 3 returns model B -> stop after that call
    _patch_pipeline_with_mock(monkeypatch, _mock_chat_handler(["A", "A", "B"], log), counter)
    out = d / "run"
    argv = _pilot_argv(out, qf, rf, sw, fake=False)
    assert cli.main(argv) == cli.EXIT_STOPPED
    assert len(log) == 3                                  # no call after the differing one
    assert (out / "STOPPED").exists() and not (out / "COMPLETE").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert "'B'" in meta["stop_reason"]
    rows = cli._read_jsonl(out / "runs.jsonl")
    assert [tuple(r["summary_key"]) for r in rows][-1] == ("5195", "a1_opaque", 0)
    assert rows[-1]["output_state"] == "error" and "PilotStop" in rows[-1]["error"]
    with pytest.raises(SystemExit):                       # continuation needs a new design decision
        cli.main(argv)


@realdata
def test_realdata_E7A1_underscore_ids_pass_frozen_provenance_guard(real):
    """Regression for amendment E7-A1 (author Q-P). The frozen grounding.provenance_guard extracts
    ``\\b\\d+`` tokens. With the original hyphen format each cited ``CASE-<neighbour scan>`` was an
    "Unverified number" (only the query slide_id is allowed), so citing all 5 neighbours exceeded
    _MAX_GROUNDING_WARNINGS = 3 and rejected A1/A2 but not A0 (negative control below). With
    ``CASE_<scan>`` (underscore is a word character) the same text is accepted in all arms."""
    _, pll = cli._import_pipeline("manuscript")

    def cite_all(p):
        ids = [n["case_id"] for n in p["neighbors"]]
        raw = json.dumps({"summary": f"The five neighbors [{', '.join(ids)}] form a mixed set.",
                          "claims": [{"text": "x", "source_case_ids": [ids[0]], "field": "tumor_group",
                                      "value": p["neighbors"][0]["tumor_group"]}]})
        derived = pll._derive_stats_from_payload(p)[0]
        _t, _c, warnings, _g = pll.parse_grounded_response(raw, p, derived)
        return [w for w in warnings if "Unverified number" in w], cli.candidate_state(pll, raw, p)["candidate_state"]

    for q in real.queries:
        for arm in ARMS:
            unverified, st = cite_all(real.payloads[q][arm])
            assert unverified == [] and st == "ok", (q, arm, unverified)
    # negative control: the pre-amendment hyphen format, same text, is rejected in A1/A2
    for arm in ("a1_opaque", "a2_cf_plurality"):
        p = json.loads(json.dumps(real.payloads["5195"][arm]))
        p["current_case"]["comp_bio_id"] = p["current_case"]["comp_bio_id"].replace("CASE_", "CASE-")
        for n in p["neighbors"]:
            n["case_id"] = n["case_id"].replace("CASE_", "CASE-")
        unverified, st = cite_all(p)
        assert len(unverified) == 5 and st == "rejected", (arm, unverified)


@realdata
def test_realdata_E7A1_no_extractable_number_and_no_hyphen_ids(real):
    """(b) the frozen _NUM_RE extracts nothing from any A1/A2 case_id (or the query CASE_ id);
    (c) no A1/A2 payload contains ``CASE-``."""
    _, pll = cli._import_pipeline("manuscript")
    grounding = sys.modules[pll.__name__.rsplit(".", 1)[0] + ".grounding"]
    assert cli.MANUSCRIPT_SRC in Path(grounding.__file__).parents           # the frozen deployed copy
    num_re = grounding._NUM_RE
    assert num_re.pattern == r"\b\d+(?:\.\d+)?\b"
    assert num_re.findall("CASE-12590") == ["12590"] and num_re.findall("CASE_12590") == []   # the reason
    n = 0
    for q in real.queries:
        for arm in ("a1_opaque", "a2_cf_plurality"):
            p = real.payloads[q][arm]
            ids = [nb["case_id"] for nb in p["neighbors"]] + [p["current_case"]["comp_bio_id"]]
            assert num_re.findall(json.dumps(ids)) == [], (q, arm)
            for nb in p["neighbors"]:
                assert num_re.findall(nb["case_id"]) == []
                n += 1
            assert "CASE-" not in json.dumps(p), (q, arm)
    assert n == 20 * 2 * 5


@realdata
def test_realdata_probe_shape_a0_only_full_swap_table(real, monkeypatch):
    """The planned live probe: --queries-file {5195}, --arms a0_masked_posterior, no --repeats-file,
    the full analyst swap table (superset accepted only because A2 is excluded). Run here in --fake."""
    d = _scratch("probe_shape")
    (d / "q.txt").write_text("5195\n")
    argv = ["--matrix", str(MATRIX), "--k", "5", "--summarizer", "manuscript", "--arms", "a0_masked_posterior",
            "--queries-file", str(d / "q.txt"),
            "--probabilities-file", str(IN / "posterior_fragments.json"),
            "--weights-file", str(IN / "neighbor_weights.csv"), "--swap-table", str(SWAP)]
    out = d / "run"
    assert cli.main(argv + ["--out-dir", str(out), "--fake"]) == 0
    assert sorted(f.name for f in (out / "payloads").iterdir()) == ["5195_a0_masked_posterior_0.json"]
    p = json.loads((out / "payloads" / "5195_a0_masked_posterior_0.json").read_text())
    assert p == real.payloads["5195"]["a0_masked_posterior"] and p["current_case"]["comp_bio_id"] == "CASE_5195"
    rows = _assert_no_truth_in_runs(out, {q: v["native_id"] for q, v in real.key.items()})
    assert [tuple(r["summary_key"]) for r in rows] == [("5195", "a0_masked_posterior", 0)]
    # the same full table with A2 selected is refused (both-directions check unchanged)
    out2 = d / "run_a2"
    assert cli.main(argv + ["--arms", "a0_masked_posterior,a2_cf_plurality", "--out-dir", str(out2), "--fake"]) == 2
    assert "swap table queries differ" in json.loads((out2 / "key" / "aborted.json").read_text())["error"]
    assert not (out2 / "runs.jsonl").exists() and not any((out2 / "payloads").iterdir())


# =============================================================================
# E7-A2 repairs (independent code review F1-F6, F11)
# =============================================================================

def _status_handler(code):
    import httpx

    def handler(request):
        return httpx.Response(code, headers={"retry-after-ms": "1"}, json={"error": {"message": f"mock {code}"}})
    return handler


def _connect_error_handler(request):
    import httpx
    raise httpx.ConnectError("mock connection refused", request=request)


@realdata
@pytest.mark.parametrize("name,handler,cls", [
    ("503", _status_handler(503), "InternalServerError"),
    ("connect", _connect_error_handler, "APIConnectionError"),
    ("429", _status_handler(429), "RateLimitError"),
    ("401", _status_handler(401), "AuthenticationError"),
])
def test_realdata_E7A2_F1_transport_error_stops_on_first(real, monkeypatch, name, handler, cls):
    """F1: the FIRST transport-class failure is recorded (summary state ``error``) and stops the run:
    exactly one request, exit 3, STOPPED, no COMPLETE, the other summaries are never sent."""
    d = _scratch(f"f1_{name}")
    qf, rf, sw = _subset_inputs(d, ["5195", "16518"], [])
    counter = []
    _patch_pipeline_with_mock(monkeypatch, handler, counter)
    out = d / "run"
    assert cli.main(_pilot_argv(out, qf, rf, sw, fake=False)) == cli.EXIT_STOPPED
    assert len(counter) == 1
    assert (out / "STOPPED").exists() and not (out / "COMPLETE").exists()
    assert (out / "STOPPED").read_text().strip() == f"transport: {cls}"
    meta = json.loads((out / "meta.json").read_text())
    assert meta["stop_reason"] == f"transport: {cls}" and meta["sessions"][-1]["n_calls"] == 1
    rows = cli._read_jsonl(out / "runs.jsonl")
    assert [tuple(r["summary_key"]) for r in rows] == [("5195", "a0_masked_posterior", 0)]
    assert rows[0]["output_state"] == "error" and "PilotStop" in rows[0]["error"]
    assert rows[0]["call"]["error"].startswith(cls)
    assert sorted(f.name for f in (out / "responses").iterdir()) == ["5195_a0_masked_posterior_0.json"]
    saved = json.loads((out / "responses" / "5195_a0_masked_posterior_0.json").read_text())
    assert "in_flight" not in saved and len(saved["calls"]) == 1          # in-flight marker replaced
    assert saved["calls"][0]["state"] == "error" and len(saved["calls"][0]["request_messages_sha256"]) == 64
    eps = cli._read_jsonl(out / "key" / "endpoints_auto.jsonl")
    assert [tuple(e["summary_key"]) for e in eps] == [("5195", "a0_masked_posterior", 0)]
    with pytest.raises(SystemExit):                                       # continuation = design decision
        cli.main(_pilot_argv(out, qf, rf, sw, fake=False))
    assert len(counter) == 1


class _Kill(BaseException):
    """Simulates the process dying while a request is in flight (escapes every ``except Exception``)."""


@realdata
def test_realdata_E7A2_F2_in_flight_record_and_resume(real, monkeypatch):
    d = _scratch("f2_inflight")
    qf, rf, sw = _subset_inputs(d, ["5195", "16518"], [])
    log, counter = [], []
    ok = _mock_chat_handler(["gpt-5-2025-08-07"], log)
    state = {"kill_at": 3}                                 # summary 1 = calls 1-2; die on call 3

    def handler(request):
        if len(counter) == state["kill_at"]:
            raise _Kill("killed while in flight")
        return ok(request)
    _patch_pipeline_with_mock(monkeypatch, handler, counter)
    out = d / "run"
    argv = _pilot_argv(out, qf, rf, sw, fake=False)
    with pytest.raises(_Kill):
        cli.main(argv)
    k2 = "5195_a1_opaque_0"
    saved = json.loads((out / "responses" / f"{k2}.json").read_text())
    assert saved["state"] == "in_flight" and saved["calls"] == []
    inf = saved["in_flight"]
    assert inf["state"] == "in_flight" and inf["sent_utc"] and len(inf["request_messages_sha256"]) == 64
    sent_body = json.loads(counter[2].content)
    assert inf["request_messages_sha256"] == cli._sha(json.dumps(sent_body["messages"], sort_keys=True))
    assert {tuple(r["summary_key"]) for r in cli._read_jsonl(out / "runs.jsonl")} == {("5195", "a0_masked_posterior", 0)}
    # resume with the same command: only the in-flight key + unsent keys are sent
    state["kill_at"] = -1
    n_before = len(counter)
    assert cli.main(argv) == 0
    sent_payload_ids = [json.loads(next(m["content"] for m in json.loads(r.content)["messages"]
                                        if m["role"] == "user"))["payload"]["current_case"]["comp_bio_id"]
                        for r in counter[n_before:]]
    assert len(counter) - n_before == 2 * 5                # 5 summaries x (initial + revision)
    rows = cli._read_jsonl(out / "runs.jsonl")
    keys = [tuple(r["summary_key"]) for r in rows]
    assert len(set(keys)) == 6 and len(keys) == 12 and keys.count(("5195", "a0_masked_posterior", 0)) == 2
    assert sent_payload_ids.count("CASE_5195") == 4 and sent_payload_ids.count("CASE_16518") == 6
    res = cli._read_jsonl(out / "resumes.jsonl")
    assert res[0]["event"] == "resume" and res[0]["possibly_delivered_unrecorded_keys"] == [["5195", "a1_opaque", 0]]
    assert res[1] == {**res[1], "event": "in_flight_without_response", "summary_key": ["5195", "a1_opaque", 0],
                      "possibly_delivered_unrecorded": True}
    assert all(r.get("possibly_delivered_unrecorded_prior_attempt") for r in rows
               if tuple(r["summary_key"]) == ("5195", "a1_opaque", 0))
    assert not any(r.get("possibly_delivered_unrecorded_prior_attempt") for r in rows
                   if tuple(r["summary_key"]) != ("5195", "a1_opaque", 0))
    assert json.loads((out / "responses" / f"{k2}.attempt-1.json").read_text())["state"] == "in_flight"
    assert "state" not in json.loads((out / "responses" / f"{k2}.json").read_text())
    assert (out / "COMPLETE").exists()
    _assert_no_truth_in_runs(out, {q: v["native_id"] for q, v in real.key.items()})


@realdata
def test_realdata_E7A2_F3_second_resume_refused_without_flag(real):
    d = _scratch("f3_resume")
    qf, rf, sw = _subset_inputs(d, ["5195"], [])
    out = d / "run"
    argv = _pilot_argv(out, qf, rf, sw, arms=["a0_masked_posterior", "a1_opaque"])

    def crash_back_to_one_key():
        for f in ("COMPLETE", "audit_summary.json", "sha256.txt"):
            (out / f).unlink()
        rows = cli._read_jsonl(out / "runs.jsonl")
        (out / "runs.jsonl").write_text(json.dumps(rows[0]) + "\n")
    assert cli.main(argv) == 0
    crash_back_to_one_key()
    assert cli.main(argv) == 0                             # first resume: allowed
    crash_back_to_one_key()
    with pytest.raises(SystemExit) as e:                   # second resume: refused (exit status 1)
        cli.main(argv)
    assert isinstance(e.value.code, str) and "--allow-additional-resume" in e.value.code
    assert len(cli._read_jsonl(out / "resumes.jsonl")) == 1
    assert cli.main(argv + ["--allow-additional-resume"]) == 0
    res = cli._read_jsonl(out / "resumes.jsonl")
    assert [(r["resume_number"], r["allow_additional_resume"]) for r in res] == [(1, False), (2, True)]
    assert "--allow-additional-resume" in res[1]["argv"]
    assert "--allow-additional-resume" not in (out / "command.txt").read_text()


@realdata
def test_realdata_E7A2_F4_post_call_integrity_failure_rc4(real, monkeypatch, capsys):
    d = _scratch("f4_integrity")
    qf, rf, sw = _subset_inputs(d, ["5195"], [])
    out = d / "run"
    orig = cli._append_jsonl

    def dup_endpoints(path, *recs):                        # corrupt: two endpoint records per key
        orig(path, *recs)
        if Path(path).name == "endpoints_auto.jsonl":
            orig(path, *recs)
    monkeypatch.setattr(cli, "_append_jsonl", dup_endpoints)
    assert cli.main(_pilot_argv(out, qf, rf, sw, arms=["a0_masked_posterior"])) == cli.EXIT_INTEGRITY == 4
    err = capsys.readouterr().err
    assert "post-call integrity failure" in err and "ABORTED" not in err
    assert not (out / "COMPLETE").exists()


@realdata
def test_realdata_E7A2_F5_swap_mismatch_message_hashes_S(real, capsys):
    rules = {q: {"relabelled_ranks": v["a2_relabelled_ranks"], "S": v["a2_S"]} for q, v in real.key.items()}
    t = json.loads(json.dumps(real.table))
    wrong = sorted(G15 - {t["5195"]["S"]})[0]
    t["5195"]["S"] = wrong
    with pytest.raises(cli.PilotAssertionError) as e:
        cli.check_swap_table(rules, t)
    msg = str(e.value)
    assert cli._sha(rules["5195"]["S"]) in msg and cli._sha(wrong) in msg
    # end-to-end: the stderr mismatch text names no G15 group
    d = _scratch("f5_hash")
    qf, rf, sw = _subset_inputs(d, ["5195", "7276"], ["5195"])
    tt = pd.read_csv(sw, dtype=str); tt.loc[tt["query"] == "5195", "S"] = wrong; tt.to_csv(sw, index=False)
    capsys.readouterr()
    assert cli.main(_pilot_argv(d / "run", qf, rf, sw)) == 2
    lines = [ln for ln in capsys.readouterr().err.splitlines() if "swap table" in ln]
    assert lines
    for ln in lines:
        for g in G15:
            assert not re.search(r"(?<![A-Za-z0-9])" + re.escape(g) + r"(?![A-Za-z0-9])", ln), (g, ln)
    for ln in [json.loads((d / "run" / "key" / "aborted.json").read_text())["error"]]:
        assert not any(re.search(r"(?<![A-Za-z0-9])" + re.escape(g) + r"(?![A-Za-z0-9])", ln) for g in G15)


def test_E7A2_F6_new_aliases_detected():
    p = _toy_payload()                                     # labels: RMS, OS (+ their diagnoses)
    new = {"MRT": ["rhabdoid tumour", "rhabdoid tumor", "rhabdoid"],
           "OS": ["osteogenic sarcoma"],
           "MPNST": ["nerve sheath tumour", "nerve sheath tumor", "neurofibrosarcoma", "malignant triton tumour",
                     "malignant triton tumor", "malignant schwannoma"],
           "EWS": ["Ewing"], "ACT/ACC": ["adrenocortical"]}
    for g, als in new.items():
        assert set(als) <= set(cli.ALIASES[g]), g
        for al in als:
            r = cli.mention_test(g, {"text": f"Features suggest {al.upper()} here."}, p)
            assert r["n_hits"] == 1 and r["hits"][0]["alias"] == al, (g, al, r["hits"])
            if g != "OS":
                assert r["leak_candidate"] and r["leak_auto"], (g, al)
    # one span, several aliases (full name contains "nerve sheath tumour") -> one hit
    r = cli.mention_test("MPNST", {"text": "a malignant peripheral nerve sheath tumour"}, p)
    assert r["n_hits"] == 1 and r["hits"][0]["alias"] == "malignant peripheral nerve sheath tumour"
    assert cli.mention_test("MRT", {"text": "embryonal rhabdomyosarcoma"}, p)["n_hits"] == 0
    assert cli.mention_test("NF", {"text": "neurofibrosarcoma"}, p)["n_hits"] == 0


def test_E7A2_F6_codes_with_digits_and_nf_adjudication_route():
    p = _toy_payload()
    # codes stay case-sensitive with alphanumeric boundaries
    assert cli.group_hits("NF", "NF1 loss") == [] and cli.group_hits("SS", "SS18-SSX fusion") == []
    assert cli.mention_test("SS", {"text": "SS18-SSX fusion"}, p)["n_hits"] == 0
    assert cli.group_hits("NF", "neurofibromatosis type 1") == []
    for txt, al in (("NF1 loss", "NF1"), ("history of neurofibromatosis", "neurofibromatosis")):
        r = cli.mention_test("NF", {"text": txt}, p)
        assert r["n_hits"] == 1 and r["hits"][0]["alias"] == al and r["hits"][0]["adjudicate"] == ["NF"]
        assert r["hits"][0]["needs_adjudication"] and not r["leak_auto"] and r["leak_candidate"]
        assert r["needs_adjudication"] and r["adjudicate_routes"] == ["NF"]
    assert cli.mention_test("NF", {"text": "nf1 lower case"}, p)["n_hits"] == 0      # code rule


def test_E7A2_F6_atrt_mrt_family_and_misc_entry_kept():
    p = _toy_payload()
    for txt in ("consistent with ATRT", "an AT/RT", "an AT-RT", "atypical teratoid tumour"):
        r = cli.mention_test("MRT", {"text": txt}, p)
        assert r["n_hits"] == 1 and r["hits"][0]["adjudicate"] == ["MRT", "MISC"], txt
        assert r["mrt_family_hit"] and r["leak_candidate"] and not r["leak_auto"] and r["needs_adjudication"], txt
        assert r["adjudicate_routes"] == ["MISC", "MRT"]
    # existing MISC subtype entry is kept (automatic for MISC) and carries the same route
    r = cli.mention_test("MISC", {"text": "consistent with ATRT"}, p)
    assert r["n_hits"] == 1 and r["leak_auto"] and r["hits"][0]["adjudicate"] == ["MISC", "MRT"]
    # routes never make a group "present in labels" (would suppress the MRT-family leak flag)
    q = json.loads(json.dumps(p))
    q["neighbors"][0].update({"tumor_group": "MISC", "diagnosis": "ATRT"})
    assert not cli.group_in_labels("MRT", q) and cli.group_in_labels("MISC", q)
    r = cli.mention_test("MRT", {"text": "ATRT"}, q)
    assert r["leak_candidate"] and r["mrt_family_hit"]
