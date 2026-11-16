"""Citation style v2: normaliser, verifier, lenient JSON, single revision call."""
import json

import pytest

from patho_toy_summarizer.grounding import (
    _build_allowed_numerics,
    loads_lenient,
    normalize_citations,
    verify_citation_style,
)
from patho_toy_summarizer.llm import attempt_llm_summary_from_payload_ex

PAYLOAD = {
    "schema_version": 1,
    "current_case": {"slide_id": "PBBRBS", "comp_bio_id": "PBBRBS", "age": "14.13 yr",
                     "site": "Pelvis, NOS", "met": "Primary"},
    "neighbors": [
        {"case_id": "A", "similarity": 0.7046, "diagnosis": "RMS", "tumor_group": "RMS",
         "site": "Lymph Node (M)", "met": "Metastatic", "discrepant": "N"},
        {"case_id": "B", "similarity": 0.7356, "diagnosis": "RMS", "tumor_group": "RMS",
         "site": "Testicle (P)", "met": "Primary", "discrepant": "N"},
        {"case_id": "C", "similarity": 0.7377, "diagnosis": "RMS", "tumor_group": "RMS",
         "site": "Hand (P)", "met": "Primary", "discrepant": "N"},
        {"case_id": "D", "similarity": 0.7387, "diagnosis": "NB", "tumor_group": "NB",
         "site": "Retroperitoneum (P)", "met": "Primary", "discrepant": "N"},
        {"case_id": "E", "similarity": 0.7430, "diagnosis": "RMS", "tumor_group": "RMS",
         "site": "Nasopharyngeal (P)", "met": "Primary", "discrepant": "N"},
    ],
}

BAD = (
    "Case PBBRBS [PBBRBS]: pelvis primary. The dominant diagnosis is RMS (4 votes) with one NB "
    "case [A, B, C, D, E]. Similarity scores span 0.705 to 0.743 (mean 0.732) [A, E]. "
    "All neighbors are pediatric [A;B;C;D;E]. Sites include lymph node [A][B][C]. "
    "The closest match is A at 0.705 [A]. No discrepant neighbors were flagged [A, B, C]. "
    "Overall, the evidence supports RMS [A, B, C, E]."
)

GOOD = (
    "Current case PBBRBS: 14.1-year-old, pelvis, primary. Four of the five neighbors are RMS "
    "[A, B, C, E] and one is NB [D]. Similarity is tight (0.705-0.743, mean 0.732); the "
    "closest match is an RMS lymph-node metastasis [A]. RMS sites: lymph node [A], testis "
    "[B], hand [C], nasopharynx [E]; the NB neighbor is a retroperitoneal primary [D]. The NB "
    "case is the only discrepant neighbor. Overall the neighbor evidence supports an "
    "RMS-predominant pattern."
)


def test_good_text_has_no_style_warnings():
    assert verify_citation_style(GOOD, PAYLOAD) == []


def test_bad_text_flags_each_rule():
    w = "\n".join(verify_citation_style(BAD, PAYLOAD))
    assert "self-citation" in w
    assert "full neighbor set is cited" in w
    assert "Count/citation mismatch" in w          # 'one NB case [A, B, C, D, E]'
    assert "Set-level statistic cited" in w        # mean/span sentence with [A, E]
    assert "Adjacent bracket chains" in w
    assert "Semicolons inside brackets" in w
    assert "Set-level fact cited" in w             # no discrepant ... [A, B, C]
    assert "Concluding sentence cited" in w


def test_bare_id_outside_brackets_is_flagged_for_real_ids():
    payload = json.loads(json.dumps(PAYLOAD))
    payload["neighbors"][0]["case_id"] = "SJEXAMPLE001_D"
    txt = "The closest match is SJEXAMPLE001_D at 0.705 [SJEXAMPLE001_D]."
    assert any("outside brackets" in x for x in verify_citation_style(txt, payload))
    # single-letter test ids are deliberately exempt (they would match prose)
    assert not any("outside brackets" in x for x in verify_citation_style("A match [A].", PAYLOAD))


def test_uncited_neighbor_is_flagged():
    txt = "Four of the five neighbors are RMS [A, B, C, E]. Overall RMS is favoured."
    w = verify_citation_style(txt, PAYLOAD)
    assert any("never cited: ['D']" in x for x in w)
    assert not any("never cited" in x for x in verify_citation_style(GOOD, PAYLOAD))


def test_count_check_ignores_of_the_five_and_ages():
    txt = "Four of the five neighbors are RMS [A, B, C, E]. Ages 16 yr [A] and 5 yr [C]."
    assert not [x for x in verify_citation_style(txt, PAYLOAD) if "mismatch" in x]


def test_count_check_uses_the_number_nearest_the_bracket():
    # the canary false positive of 2026-09-02: 'five' is the set size, 'four' is the count
    txt = ("Among the five nearest neighbors, four are rhabdomyosarcoma (RMS) [A, B, C, E] and one "
           "is neuroblastoma (NB) [D]. There are five neighbors. Four carry an OS diagnosis [A, B, C, E]. "
           "A 2-year-old ocular primary [D]. Sites: femur (ages 16 and 10 yr) [A, C], humerus [B].")
    assert not [x for x in verify_citation_style(txt, PAYLOAD) if "mismatch" in x]
    bad = "The dominant diagnosis is RMS (4 votes) with one NB case [A, B, C, D, E]."
    assert [x for x in verify_citation_style(bad, PAYLOAD) if "mismatch" in x]
    # the ST1.7b false positive (PBCCEU): '2/5 votes' is a fraction, not a count
    frac = "RMS is the dominant diagnosis (2/5 votes) and includes the closest match at 0.770 [A]."
    assert not [x for x in verify_citation_style(frac, PAYLOAD) if "mismatch" in x]


def test_normaliser_merges_chains_sorts_by_rank_and_strips_query():
    txt = "Sites [C][A] [B] and the case [PBBRBS] and dup [E;E, D] ."
    out = normalize_citations(txt, PAYLOAD)
    assert out == "Sites [A, B, C] and the case and dup [D, E]."


def test_normaliser_keeps_good_text():
    assert normalize_citations(GOOD, PAYLOAD) == GOOD


def test_loads_lenient_handles_fences_and_prose():
    obj = {"summary": "x [A]", "claims": []}
    assert loads_lenient("```json\n" + json.dumps(obj) + "\n```") == obj
    assert loads_lenient("Here you go:\n" + json.dumps(obj) + "\nThanks") == obj
    assert loads_lenient("not json at all") is None
    assert loads_lenient(None) is None


def test_three_decimals_allowed_by_provenance_guard():
    allowed = _build_allowed_numerics({"numeric_values": {0.7046322}, "case_ids": set()})
    assert "0.705" in allowed


class _Dummy:
    """Client whose responses come from a queue; records the number of calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def factory(self):
        outer = self

        class Client:
            def __init__(self, *a, **k):
                pass

            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        outer.calls.append(kwargs)
                        content = outer.responses.pop(0)

                        class C:
                            class message:
                                pass
                        C.message.content = content

                        class R:
                            choices = [C()]
                        return R()
        return lambda: Client


def _resp(summary):
    return json.dumps({"summary": summary, "claims": [
        {"statement": "RMS dominant", "source_case_ids": ["A", "B", "C", "E"],
         "field": "diagnosis", "value": "RMS"}]})


def test_revision_call_fires_once_and_is_adopted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.delenv("PATHO_LLM_FAKE", raising=False)
    monkeypatch.delenv("PATHO_STYLE_REVISION", raising=False)
    d = _Dummy([_resp(BAD), _resp(GOOD)])
    monkeypatch.setattr("patho_toy_summarizer.llm._safe_import_openai", d.factory())
    text, accepted, tele = attempt_llm_summary_from_payload_ex(PAYLOAD)
    assert text is not None and accepted
    assert tele["calls"] == 2 and tele["revision_applied"] is True
    assert tele["style_warnings_first"] and tele["style_warnings_final"] == []
    assert "previous_response" in d.calls[1]["messages"][-1]["content"]
    assert [c["source_case_ids"] for c in tele["claims"]] == [["A", "B", "C", "E"]]


def test_revision_disabled_keeps_first_answer_but_normalises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.delenv("PATHO_LLM_FAKE", raising=False)
    monkeypatch.setenv("PATHO_STYLE_REVISION", "0")
    d = _Dummy([_resp(BAD)])
    monkeypatch.setattr("patho_toy_summarizer.llm._safe_import_openai", d.factory())
    text, accepted, tele = attempt_llm_summary_from_payload_ex(PAYLOAD)
    assert tele["calls"] == 1 and tele["revision_applied"] is False
    assert "[PBBRBS]" not in text and "][" not in text and ";" not in text.split("[", 1)[1].split("]")[0]
    assert tele["style_warnings_final"]                      # placement issues remain


def test_revision_rejected_when_it_breaks_grounding(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.delenv("PATHO_LLM_FAKE", raising=False)
    monkeypatch.delenv("PATHO_STYLE_REVISION", raising=False)
    broken = json.dumps({"summary": "RMS [X1, X2, X3, X4] with 9.99 and 8.88", "claims": [
        {"statement": "s", "source_case_ids": ["X1", "X2"], "field": "diagnosis", "value": "RMS"}]})
    d = _Dummy([_resp(BAD), broken])
    monkeypatch.setattr("patho_toy_summarizer.llm._safe_import_openai", d.factory())
    text, accepted, tele = attempt_llm_summary_from_payload_ex(PAYLOAD)
    assert text is not None
    assert tele["revision_applied"] is False and tele["revision_rejected"]
    assert "RMS (4 votes)" in text                             # first answer kept


def test_fake_mode_is_style_clean(monkeypatch):
    monkeypatch.setenv("PATHO_LLM_FAKE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    text, accepted, tele = attempt_llm_summary_from_payload_ex(PAYLOAD)
    assert tele["fake"] and tele["style_warnings_final"] == []
    assert "[A, B, C, E]" in text and "[D]" in text
