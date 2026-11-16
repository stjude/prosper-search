"""S5 (R3.6 / R2.9) -- faithfulness audit of the LLM reporting layer.

Runs the summariser selected by ``--summarizer`` -- ``manuscript`` (default since
2026-09-02): the single-shot copy in ``prosper/llm/summarizer`` that the
manuscript's Methods describe, with citation style v2 and its one optional
revision call; ``agentic``: the LangGraph branch in ``comet-llm/src`` that the
2026-08-30 run (S5 as first reported) and ST1.7 of 2026-09-01 used -- on the
neighbour payloads the deployed builder (always ``comet-llm/src/search``) makes for a stratified sample
of library queries, plus two counterfactual payloads per query, and audits every
summary against the payload it was given -- independently of the pipeline's own
verifier. Design: ``[internal record]`` (revision
workspace). No accuracy or usefulness is measured; only whether what the
summary says is supported by what it was shown, and whether that changes when
the input changes.

Environment: the summariser's dependencies live in ``$PROSPER_OUT/envs/comet-llm``;
credentials are read from ``--env-file`` as KEY=VALUE lines (never printed).
``PATHO_LLM_FAKE=1`` (or ``--fake``) exercises the whole harness offline with
the pipeline's deterministic fake summary.

S5b pilot (``--arms a0_masked_posterior,a1_opaque,a2_cf_plurality`` with
``--queries-file/--repeats-file/--probabilities-file/--weights-file/--swap-table``):
implements specification ``[internal record]`` sections 3-4
(revision workspace) -- every payload is built and asserted before any call;
each call is recorded through a wrapped client (transport retries 0); truth is
written only under ``<out>/key/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

COMET = Path("/PROSPER_WORKSPACE")
LLM_SRC = COMET / "comet-llm" / "src"                       # deployed payload builder + agentic copy
MANUSCRIPT_SRC = Path(__file__).resolve().parents[2] / "prosper" / "llm" / "summarizer" / "src"
RICH_META = COMET / "comet-llm" / "data" / "final_scan_id_disease_map_rich_data.csv"
SUMMARIZERS = ("manuscript", "agentic")


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except Exception:                                   # pragma: no cover
        return "unknown"


AZURE_DEFAULTS = {   # the deployed summariser's own defaults (comet-llm README, run_summarizer_*.sh)
    "AZURE_OPENAI_ENDPOINT": "",  # set AZURE_OPENAI_ENDPOINT in the environment
    "AZURE_OPENAI_DEPLOYMENT": "gpt-5",
}


def load_env_file(path: Path) -> list:
    """Export KEY=VALUE lines into os.environ; return the key names only.

    A non-comment line without ``=`` that looks like a secret (no whitespace,
    >= 32 chars) is taken as ``AZURE_OPENAI_API_KEY`` -- the file created on
    2026-08-29 holds the bare key. Endpoint/deployment fall back to the
    deployed system's defaults. Values are never printed or logged.
    """
    keys = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            if len(line) >= 32 and not re.search(r"\s", line) \
                    and not os.environ.get("AZURE_OPENAI_API_KEY"):
                os.environ["AZURE_OPENAI_API_KEY"] = line
                keys.append("AZURE_OPENAI_API_KEY (bare line)")
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            os.environ[k] = v
            keys.append(k)
    for k, v in AZURE_DEFAULTS.items():
        if not os.environ.get(k):
            os.environ[k] = v
            keys.append(f"{k} (default)")
    return keys


def probe(a) -> int:
    """One-word live call; prints success/failure, endpoint host and deployment only."""
    from urllib.parse import urlparse
    keys = load_env_file(a.env_file) if a.env_file else []
    _, pll = _import_pipeline(a.summarizer)
    host = urlparse(os.environ.get("AZURE_OPENAI_ENDPOINT", "")).netloc
    dep = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    t0 = time.time()
    try:
        client = pll._build_client(pll._safe_import_openai())
        r = client.chat.completions.create(
            model=pll._resolve_model(),
            messages=[{"role": "user", "content": "Reply with the single word: ready"}])
        txt = (r.choices[0].message.content or "").strip()
        print(json.dumps({"probe": "OK", "env_keys": keys, "endpoint_host": host, "deployment": dep,
                          "model_reported": getattr(r, "model", None), "reply": txt[:40],
                          "seconds": round(time.time() - t0, 1), **summarizer_info(pll, a.summarizer)}))
        return 0
    except Exception as e:                                                  # noqa: BLE001
        msg = re.sub(r"[A-Za-z0-9]{32,}", "<redacted>", str(e))[:300]
        print(json.dumps({"probe": "FAIL", "env_keys": keys, "endpoint_host": host,
                          "deployment": dep, "error": msg}))
        return 1


# ----------------------------------------------------------------------------
# payload construction (the deployed system's own builder)
# ----------------------------------------------------------------------------

def _import_pipeline(summarizer: str = "manuscript"):
    """Payload builder always from ``comet-llm/src`` (the deployed builder); the
    summariser module from the selected copy. Fails loudly if the import
    resolved elsewhere (both copies share the package name)."""
    if summarizer not in SUMMARIZERS:
        raise ValueError(f"--summarizer must be one of {SUMMARIZERS}, got {summarizer!r}")
    sys.path.insert(0, str(LLM_SRC))
    from search import analysis_lib                          # noqa: WPS433
    for m in [k for k in list(sys.modules) if k.split(".")[0] == "patho_toy_summarizer"]:
        del sys.modules[m]
    if summarizer == "manuscript":
        sys.path.insert(0, str(MANUSCRIPT_SRC))
    from patho_toy_summarizer import llm as pll              # noqa: WPS433
    src = Path(pll.__file__)
    expect = MANUSCRIPT_SRC if summarizer == "manuscript" else LLM_SRC
    if expect not in src.parents:
        raise RuntimeError(f"summariser import resolved to {src}, expected under {expect}")
    return analysis_lib, pll


def summarizer_info(pll, summarizer: str) -> dict:
    """Provenance of the summariser actually imported (module path + commit)."""
    mod = Path(pll.__file__).resolve().parent
    if summarizer == "agentic":
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                             cwd=mod, text=True).strip()
        except Exception:                                   # pragma: no cover
            commit = "unknown"
    else:
        commit = _commit()
    return {"summarizer": summarizer, "summarizer_module": str(mod), "summarizer_commit": commit,
            "style_revision": os.getenv("PATHO_STYLE_REVISION", "1")}


def summarize_payload(pll, payload: dict) -> tuple:
    """Call the selected summariser; normalise the return shape to
    ``(text, accepted, telemetry_or_None)`` across the single-shot copy
    (``_ex`` variant with telemetry), the legacy 2-tuple, and the agentic
    3-tuple."""
    fn = getattr(pll, "attempt_llm_summary_from_payload_ex", None) \
        or pll.attempt_llm_summary_from_payload
    res = fn(payload)
    if len(res) == 2:
        return res[0], res[1], None
    return res


def stratified_queries(meta: pd.DataFrame, sim_ids, n_total: int, seed: int, drop=("MISC",)) -> list:
    m = meta[meta["slide_id"].isin(sim_ids) & ~meta["tumor_group"].isin(drop)]
    groups = sorted(m["tumor_group"].unique())
    per = max(1, n_total // len(groups))
    rng = np.random.default_rng(seed)
    out = []
    for g in groups:
        ids = m.loc[m["tumor_group"] == g, "slide_id"].tolist()
        take = min(per, len(ids))
        out += [str(x) for x in rng.choice(ids, size=take, replace=False)]
    return out


def blind_query(payload: dict) -> dict:
    """Remove the query's own diagnosis fields: the deployed UI does not know them."""
    p = json.loads(json.dumps(payload))
    for k in ("tumor_group", "diagnosis", "discrepant", "evaluation"):
        if k in p.get("current_case", {}):
            p["current_case"][k] = None
    return p


def counterfactual_label(payload: dict, rng: np.random.Generator, classes: list) -> tuple:
    p = json.loads(json.dumps(payload))
    nb = p["neighbors"]
    i = int(rng.integers(len(nb)))
    old = nb[i].get("tumor_group")
    new = str(rng.choice([c for c in classes if c != old]))
    nb[i]["tumor_group"] = new
    nb[i]["diagnosis"] = new
    return p, {"index": i, "case_id": nb[i]["case_id"], "old": old, "new": new}


def blind_id(payload: dict, scan_id: str, prefix: str = "CASE-") -> dict:
    """Replace the query's Comp.Bio.ID (whose prefix encodes the diagnosis, e.g. SJMPNST...)
    by the opaque scan id: measures how much of the summary's 'diagnosis' comes from the id.
    The legacy ``blind_id`` arm keeps ``CASE-``; the S5b pilot arms pass ``CASE_`` (E7-A1)."""
    p = json.loads(json.dumps(payload))
    if "current_case" in p:
        p["current_case"]["comp_bio_id"] = f"{prefix}{scan_id}"
    return p


def counterfactual_order(payload: dict) -> dict:
    p = json.loads(json.dumps(payload))
    p["neighbors"] = list(reversed(p["neighbors"]))
    return p


# ----------------------------------------------------------------------------
# audit
# ----------------------------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
_CIT = re.compile(r"\[([^\[\]]+)\]")
_CASEID = re.compile(r"\b(?:SJ|CASE[-_])[A-Z0-9]*\d+[A-Za-z0-9_]*\b")   # CASE-: legacy; CASE_: S5b (E7-A1)


def plurality(groups: list) -> tuple:
    c = Counter(g for g in groups if g)
    if not c:
        return None, []
    top = max(c.values())
    tied = [g for g, n in c.items() if n == top]
    first = next(g for g in groups if g in tied)          # closest-tie rule
    return first, tied


def allowed_numbers(payload: dict) -> set:
    vals = set()
    nb = payload["neighbors"]
    k = len(nb)
    vals.update({str(k), str(k - 1)})
    for field in ("tumor_group", "diagnosis", "site", "sex", "met", "treatment", "discrepant", "sample_type"):
        counts = Counter(str(n.get(field)) for n in nb if n.get(field) is not None)
        for n in counts.values():
            vals.add(str(n))
            vals.add(f"{100 * n / k:.0f}")
            vals.add(f"{100 * n / k:.1f}")
    for rec in nb + [payload.get("current_case", {})]:
        for key, v in rec.items():
            if v is None:
                continue
            if isinstance(v, (int, float)):
                vals.update({str(v), f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", str(int(round(v)))})
            else:
                for m in _NUM.findall(str(v)):
                    vals.add(m)
                    if "." in m:
                        vals.add(str(int(round(float(m)))))
    sims = [n.get("similarity") for n in nb if isinstance(n.get("similarity"), (int, float))]
    if sims:
        for v in (np.mean(sims), min(sims), max(sims)):
            vals.update({f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", str(int(round(v)))})
    # top-level Bayesian disease probabilities, as fractions and percentages
    for p in (payload.get("probabilities") or {}).values():
        if isinstance(p, (int, float)):
            for v in (float(p), 100.0 * float(p)):
                vals.update({f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", str(int(round(v)))})
    return vals


def audit(text: str, claims: list, payload: dict, true_group: str) -> dict:
    nb = payload["neighbors"]
    ids = {str(n["case_id"]) for n in nb} | {str(payload.get("current_case", {}).get("comp_bio_id"))}
    cites = [x.strip() for grp in _CIT.findall(text or "") for x in re.split(r"[,;\s]+", grp)]
    cites = [c for c in cites if c and not c.isdigit()]
    bad_cites = [c for c in cites if c not in ids]
    nums = _NUM.findall(text or "")
    allowed = allowed_numbers(payload)
    bad_nums = [x for x in nums if x not in allowed and x.rstrip("0").rstrip(".") not in allowed]
    groups = [n.get("tumor_group") for n in nb]
    plur, tied = plurality(groups)
    # mask citations and case ids (SJRHB..., SJMPNST...) whose prefixes encode a class
    masked = _CIT.sub(" ", text or "")
    masked = _CASEID.sub(" ", masked)
    tl = masked.upper().replace("_", " ")
    # a class is "mentioned" if its tumour-group name or any diagnosis string recorded for
    # that class among the neighbours appears in the text
    aliases = {}
    for n in nb:
        g = n.get("tumor_group")
        if g:
            aliases.setdefault(g, set()).add(str(g).upper().replace("_", " "))
            if n.get("diagnosis"):
                aliases[g].add(str(n["diagnosis"]).upper().replace("_", " "))
    def _hit(g, blob):
        return any(re.search(r"(?<![A-Z])" + re.escape(al) + r"(?![A-Z])", blob) for al in aliases.get(g, ()))
    mentions = {g for g in aliases if _hit(g, tl)}
    # dominant-diagnosis statement: the class named alongside majority/support/dominant words
    dom_ok = None
    if plur:
        window = re.findall(r"(?:MAJORITY|MOST|DOMINANT|PREDOMINANT|CONSENSUS|SUPPORT\w*|CONSISTENT WITH|FAVOU?R\w*|PRIMAR\w*)[^.]{0,100}", tl)
        named = [g for g in aliases if any(_hit(g, w) for w in window)]
        if named:
            dom_ok = plur in named or any(t in named for t in tied)
    # metadata consistency of claims (cited case field/value)
    claim_bad = 0
    by_id = {str(n["case_id"]): n for n in nb}
    for c in claims or []:
        try:
            field, value, srcs = c.get("field"), str(c.get("value")), c.get("source_case_ids") or []
        except AttributeError:
            continue
        for s in srcs:
            rec = by_id.get(str(s))
            if rec is None or field not in rec or rec[field] is None:
                continue
            rv = rec[field]
            if isinstance(rv, (int, float)) or re.fullmatch(r"-?\d+(?:\.\d+)?", str(rv)):
                # numeric field: the summary writes 3 dp (style rule 9) -- compare as numbers
                m = re.search(r"-?\d+(?:\.\d+)?", value)
                if m and abs(float(m.group(0)) - float(rv)) <= 5e-4 + 1e-9:
                    continue
            if str(rv).upper() != value.upper() and value.upper() not in str(rv).upper() \
                    and str(rv).upper() not in value.upper():
                claim_bad += 1
    # a class that reaches the text only through the payload's Bayesian probabilities is
    # payload-supported, not a leak of the query's own label
    prob_groups = {str(g) for g in (payload.get("probabilities") or {})}
    leak = bool(true_group) and (true_group.replace("_", " ") in tl) \
        and (true_group not in set(groups)) and (true_group not in prob_groups)
    return {"n_citations": len(cites), "n_bad_citations": len(bad_cites), "bad_citations": bad_cites[:5],
            "n_numbers": len(nums), "n_unsupported_numbers": len(bad_nums), "unsupported_numbers": bad_nums[:5],
            "plurality": plur, "plurality_tied": tied, "groups_mentioned": sorted(mentions),
            "dominant_statement_consistent": dom_ok, "n_claims": len(claims or []),
            "n_claims_contradicting_record": claim_bad, "leaks_true_group": leak,
            "word_count": len((text or "").split()),
            **{f"s_{k}": v for k, v in citation_style(text, payload).items()}}


# Citation-style counters (independent of the pipeline's own style verifier; the
# same definitions as [internal record] section 1).
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_AGG = re.compile(r"\b(mean|average|averag\w+|range[sd]?|rang\w+|span(?:s|ned)?|minimum|maximum)\b", re.I)
_SETFACT = re.compile(r"\b(no discrepant|discrepant[_ ]count)\b", re.I)
_CONCL = re.compile(r"^\s*(overall|taken together|in summary|in conclusion|altogether)\b", re.I)


def citation_style(text: str, payload: dict) -> dict:
    """Count style defects in one summary: id mentions, bracket groups, full-set
    groups, repeated multi-id groups, query self-citations, ``][`` chains,
    semicolon separators, numeric ids, neighbour ids outside brackets, cited
    statistic / set-fact / concluding sentences."""
    text = text or ""
    nb = [str(n.get("case_id")) for n in payload.get("neighbors", [])]
    full = set(nb)
    cur = payload.get("current_case", {}) or {}
    qids = {str(cur[k]) for k in ("comp_bio_id", "slide_id", "case_id") if cur.get(k)}
    raw_groups = _CIT.findall(text)
    groups = [[t for t in re.split(r"[,;\s]+", g.strip()) if t] for g in raw_groups]
    multi = Counter(tuple(sorted(g)) for g in groups if len(g) >= 2)
    stripped = _CIT.sub(" ", text)
    bare = sum(1 for cid in nb if len(cid) >= 3 and not cid.isdigit()
               and re.search(r"(?<![\w-])" + re.escape(cid) + r"(?![\w-])", stripped))
    agg = setfact = concl = 0
    for s in _SENT.split(text):
        sg = [[t for t in re.split(r"[,;\s]+", g.strip()) if t] for g in _CIT.findall(s)]
        if not sg:
            continue
        if _AGG.search(s) and any(len(g) >= 2 for g in sg):
            agg += 1
        if _SETFACT.search(s):
            setfact += 1
        if _CONCL.match(s):
            concl += 1
    return {"id_mentions": sum(len(g) for g in groups), "groups": len(groups),
            "full_set": sum(1 for g in groups if len(nb) >= 2 and set(g) == full),
            "repeat_multi": sum(v - 1 for v in multi.values() if v > 1),
            "self_cite": sum(1 for g in groups if any(t in qids for t in g)),
            "chain": len(re.findall(r"\]\s*\[", text)),
            "semicolon": sum(1 for g in raw_groups if ";" in g),
            "numeric_id": sum(1 for g in groups for t in g if t.isdigit()),
            "bare_id": bare, "agg_cited": agg, "setfact_cited": setfact, "conclusion_cited": concl}


# ----------------------------------------------------------------------------
# S5b reporting-layer pilot (specification s5b_pilot_contract_20261009.md v4, sections 3-4)
# ----------------------------------------------------------------------------

PILOT_ARMS = ("a0_masked_posterior", "a1_opaque", "a2_cf_plurality")
REPEAT_ARM = "repeat"                     # A0 payload re-run once (repeat index 1)
LEGACY_ARMS = ("original", "blind_id", "cf_label", "cf_order")
# literal loader group list (specification section 3)
G15 = ("ACT/ACC", "EWS", "LIVER", "MEL", "MISC", "MPNST", "MRT", "NB", "NF", "NORMAL",
       "OS", "RB", "RMS", "SS", "WT")
A2_SWAP_SEED = 20261011
A2_EXCLUDED = ("MISC", "NORMAL")
EXPECTED_MODEL = "gpt-5"
PROB_SUM_TOL = 5e-6
PROB_KEEP = 1e-8
EXIT_STOPPED = 3
EXIT_INTEGRITY = 4                                       # E7-A2 F4: post-call integrity failure
# Amendment E7-A1 (author Q-P, 2026-10-10): pilot opaque IDs use an underscore. A hyphen is a word
# boundary, so the frozen grounding.provenance_guard (r"\b\d+") read each cited CASE-<scan> as an
# unverified number; ``CASE_<scan>`` yields no extractable number.
PILOT_CASE = "CASE_"

# ALIASES: from specification §5 (i) "lexical detection map" -- used only for possible-mention
# detection (4.5(a) A2 label assertions and 4.5(c)). Specification-listed: LIVER <- HB, HCC,
# hepatoblastoma, hepatocellular carcinoma; NORMAL <- Normal, normal tissue; ACT/ACC <- ACT,
# ACC, adrenocortical tumour/carcinoma; NB <- NB, neuroblastoma, ganglioneuroblastoma; code <->
# full name for every group; subtypes map upward. The full names, spelling variants and the
# loader `diagnosis` subtype strings (mapped upward) below are provisional and
# listed in [internal record] for review. Codes (all-caps tokens) are matched
# case-sensitively, everything else case-insensitively (with an optional plural s/es).
ALIASES = {
    "ACT/ACC": ["ACT/ACC", "ACT", "ACC", "adrenocortical tumour", "adrenocortical tumor",
                "adrenocortical carcinoma", "adrenocortical"],
    "EWS": ["EWS", "Ewing sarcoma", "Ewing's sarcoma", "Ewing"],
    "LIVER": ["LIVER", "HB", "HCC", "hepatoblastoma", "hepatocellular carcinoma", "liver",
              "Embryonal sarcoma"],
    "MEL": ["MEL", "melanoma"],
    "MISC": ["MISC", "miscellaneous", "CIC-rearranged sarcoma", "Schwannoma", "CCSK", "CCS",
             "Desmoid tumor", "Desmoid tumour", "Sarcoma with BCOR genetic alteration",
             "Paraganglioma", "Desmoid fibromatosis", "ATRT", "Osteoblastoma"],
    "MPNST": ["MPNST", "malignant peripheral nerve sheath tumour",
              "malignant peripheral nerve sheath tumor", "nerve sheath tumour", "nerve sheath tumor",
              "neurofibrosarcoma", "malignant triton tumour", "malignant triton tumor",
              "malignant schwannoma"],
    "MRT": ["MRT", "malignant rhabdoid tumour", "malignant rhabdoid tumor", "rhabdoid tumour",
            "rhabdoid tumor", "rhabdoid"],
    "NB": ["NB", "neuroblastoma", "ganglioneuroblastoma"],
    "NF": ["NF", "neurofibroma"],
    "NORMAL": ["NORMAL", "Normal", "normal tissue"],
    "OS": ["OS", "osteosarcoma", "osteogenic sarcoma"],
    "RB": ["RB", "retinoblastoma"],
    "RMS": ["RMS", "rhabdomyosarcoma"],
    "SS": ["SS", "synovial sarcoma"],
    "WT": ["WT", "Wilms tumour", "Wilms tumor", "Wilms' tumour", "Wilms' tumor", "Wilms",
           "nephroblastoma", "Nephrogenic rest", "Nephroblastic process"],
}
# E7-A2 (specification §5(i) additions, F6): MRT <- rhabdoid tumour/tumor, rhabdoid; OS <- osteogenic
# sarcoma; MPNST <- nerve sheath tumour/tumor, neurofibrosarcoma, malignant triton tumour/tumor,
# malignant schwannoma; EWS <- Ewing (already present); ACT/ACC <- adrenocortical (added above).
# ADJUDICATION ROUTES (E7-A2): detected for 4.5(c) but never automatic. Each alias lists the groups
# the rater adjudicates it against; ``MRT_FAMILY`` aliases count as MRT-family for the leak detector
# (leak_candidate for true group MRT), the inclusive unsafe-direction-protecting choice. The MISC
# subtype entry "ATRT" above is kept (automatic for MISC) and does not suppress the MRT-family route.
# Routes are NOT used for the 4.5(a) label assertions or ``group_in_labels`` (payloads unchanged).
ADJUDICATION_ROUTES = {
    "ATRT": ["MRT", "MISC"], "AT/RT": ["MRT", "MISC"], "AT-RT": ["MRT", "MISC"],
    "atypical teratoid": ["MRT", "MISC"],
    "neurofibromatosis": ["NF"], "NF1": ["NF"],
}
MRT_FAMILY = ("ATRT", "AT/RT", "AT-RT", "atypical teratoid")
_CODE_ALIAS = re.compile(r"^[A-Z0-9/]+$")
_ID_TOKEN = re.compile(r"(?<![A-Za-z0-9_-])(?:SJ[A-Za-z0-9_]+|CASE[-_][A-Za-z0-9_]+)")
_SJ_TOKEN = re.compile(r"\bSJ[A-Z]")
_STEM = re.compile(r"_[A-Za-z]\d*$")


class PilotAssertionError(RuntimeError):
    """A specification pre-send / construction assertion failed: abort before any API call."""


class PilotIntegrityError(RuntimeError):
    """E7-A2 F4: a post-call completeness/integrity check failed (calls were made): exit 4."""


class PilotStop(BaseException):
    """Raised inside the wrapped client after the model-change stop rule fired. BaseException so
    that llm.py's ``except Exception`` cannot convert it into an ``ERROR:`` summary."""


def patient_stem(comp_bio_id: str) -> str:
    """Comp.Bio.ID minus the trailing ``_<letter><digits>`` sample suffix."""
    return _STEM.sub("", str(comp_bio_id))


def _alias_regex(alias: str) -> re.Pattern:
    if _CODE_ALIAS.match(alias):
        return re.compile(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])")
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?:e?s)?(?![A-Za-z0-9])", re.I)


_ALIAS_RX = {g: [(al, _alias_regex(al)) for al in als] for g, als in ALIASES.items()}
_ROUTE_RX = {g: [(al, _alias_regex(al)) for al, gs in ADJUDICATION_ROUTES.items() if g in gs]
             for g in ALIASES}


def route_hits(group: str, text: str) -> list:
    """(alias, matched_text, start) hits of the E7-A2 adjudication-route aliases routed to ``group``."""
    out = []
    for al, rx in _ROUTE_RX.get(group, []):
        for m in rx.finditer(text or ""):
            out.append((al, m.group(0), m.start()))
    return out


def group_hits(group: str, text: str) -> list:
    """All (alias, matched_text, start) hits of ``group``'s code/aliases in ``text``."""
    out = []
    for al, rx in _ALIAS_RX.get(group, [(group, _alias_regex(group))]):
        for m in rx.finditer(text or ""):
            out.append((al, m.group(0), m.start()))
    return out


def _is_bare_normal(group: str, alias: str, matched: str, text: str, start: int) -> bool:
    """Bare "normal" (not the code NORMAL, not "normal tissue"): a NORMAL mention only when
    used as a class/diagnosis -- not decidable automatically, so flagged for adjudication."""
    if group != "NORMAL" or matched == "NORMAL":
        return False
    return not re.match(r"normal\s+tissue", text[start:], re.I)


def label_fields(payload: dict) -> list:
    """Label fields (specification 4.5(a)): neighbour tumor_group, diagnosis and probabilities keys."""
    vals = []
    for n in payload.get("neighbors", []):
        for k in ("tumor_group", "diagnosis"):
            if n.get(k) is not None:
                vals.append(str(n[k]))
    vals += [str(k) for k in (payload.get("probabilities") or {})]
    return vals


def group_in_labels(group: str, payload: dict) -> bool:
    """True iff ``group`` (code or alias) occurs in any label field of ``payload``."""
    return any(group_hits(group, v) for v in label_fields(payload))


def read_ids(path) -> list:
    """Ids one per line, order as listed (blank lines ignored, duplicates rejected)."""
    ids = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if len(set(ids)) != len(ids):
        raise PilotAssertionError(f"duplicate ids in {path}")
    return ids


def read_fragments(path) -> dict:
    """posterior_fragments.json -> {query: OrderedDict(class: p)} with key order and float
    values exactly as stored (json preserves order; floats parse round-trip)."""
    raw = json.loads(Path(path).read_text())
    return {str(q): v["probabilities"] for q, v in raw.items()}


def read_weights(path) -> pd.DataFrame:
    w = pd.read_csv(path, dtype={"query": str, "neighbor_slide_id": str}, float_precision="round_trip")
    need = {"query", "rank", "neighbor_slide_id", "weight"}
    if not need <= set(w.columns):
        raise PilotAssertionError(f"{path} lacks columns {sorted(need - set(w.columns))}")
    return w


def read_swap_table(path) -> dict:
    t = pd.read_csv(path, dtype=str)
    out = {}
    for _, r in t.iterrows():
        q = str(r["query"]).strip()
        if q in out:
            raise PilotAssertionError(f"duplicate query {q} in swap table")
        out[q] = {"relabelled_ranks": [int(x) for x in str(r["relabelled_ranks"]).split(";") if x.strip()],
                  "S": str(r["S"]).strip()}
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def true_groups_KEY_ONLY(meta: pd.DataFrame, queries: list) -> dict:
    """TRUTH. The query's own metadata ``tumor_group`` (the loader's analysis group). The only
    place truth enters the pilot run path: it feeds the A2 swap rule (S must differ from the
    true group), the A2 pre-send label assertions and the key-side endpoint writer. Its values
    are written only under ``key/``, never to runs.jsonl, payloads or responses."""
    tg = dict(zip(meta["slide_id"].astype(str), meta["tumor_group"]))
    missing = [q for q in queries if q not in tg]
    if missing:
        raise PilotAssertionError(f"queries absent from metadata: {missing}")
    return {q: tg[q] for q in queries}


def a2_swap_rule(query: str, a0_groups: list, true_group: str) -> dict:
    """Specification section 3: tied set by the harness ``plurality()`` rule; candidates =
    sorted(G15 - {MISC, NORMAL} - A0 neighbour groups - {true}); S drawn with
    default_rng([20261011, int(query)]). Returns key material (contains truth-derived S)."""
    plur, tied = plurality(list(a0_groups))
    if not tied:
        raise PilotAssertionError(f"{query}: no neighbour groups, plurality undefined")
    ranks = [i + 1 for i, g in enumerate(a0_groups) if g in tied]
    cands = sorted(set(G15) - set(A2_EXCLUDED) - {g for g in a0_groups if g} - {true_group})
    if not cands:
        raise PilotAssertionError(f"{query}: empty A2 candidate list")
    s = str(np.random.default_rng([A2_SWAP_SEED, int(query)]).choice(cands))
    return {"plurality": plur, "tied": sorted(tied), "relabelled_ranks": ranks,
            "candidates": cands, "S": s}


def check_swap_table(rules: dict, table: dict, allow_superset: bool = False) -> None:
    """Recomputed ranks/S must equal the independent swap table, in both directions. Only for runs whose
    ``--arms`` exclude A2 (e.g. the one-query A0 probe) may the table be a superset of the run's
    queries (``allow_superset``); every recomputed query must still be present and equal."""
    if allow_superset:
        if not set(rules) <= set(table):
            raise PilotAssertionError(f"swap table lacks recomputed queries {sorted(set(rules) - set(table))}")
    elif set(rules) != set(table):
        raise PilotAssertionError(
            f"swap table queries differ: only recomputed={sorted(set(rules) - set(table))} "
            f"only table={sorted(set(table) - set(rules))}")
    for q, r in rules.items():
        t = table[q]
        if r["relabelled_ranks"] != t["relabelled_ranks"] or r["S"] != t["S"]:
            # E7-A2 F5: S is truth-derived key material -> only its sha256 is printed
            raise PilotAssertionError(
                f"{q}: recomputed ranks {r['relabelled_ranks']} / S sha256 {_sha(str(r['S']))} "
                f"!= swap table ranks {t['relabelled_ranks']} / S sha256 {_sha(str(t['S']))}")


def _jcopy(x):
    return json.loads(json.dumps(x))


def aggregate_weights(payload: dict, weights_q: pd.DataFrame) -> dict:
    """Sum ``neighbor_weights.csv`` weights by the payload's neighbour ``tumor_group`` (rank
    order), round to 6 dp, keep > 1e-8, sort by descending p then class name."""
    nb = payload["neighbors"]
    w = weights_q.sort_values("rank")
    if list(w["rank"]) != list(range(1, len(nb) + 1)):
        raise PilotAssertionError(f"weights ranks {list(w['rank'])} do not match {len(nb)} neighbours")
    if [str(x) for x in w["neighbor_slide_id"]] != [str(n["slide_id"]) for n in nb]:
        raise PilotAssertionError("weights neighbour slide ids differ from payload neighbours")
    acc = {}
    for n, wt in zip(nb, w["weight"]):
        g = n.get("tumor_group")
        acc[g] = acc.get(g, 0.0) + float(wt)
    rounded = {g: round(p, 6) for g, p in acc.items()}
    kept = [(g, p) for g, p in rounded.items() if p > PROB_KEEP]
    kept.sort(key=lambda gp: (-gp[1], gp[0]))
    return dict(kept)


def build_pilot_payloads(base: dict, query: str, fragment: dict, weights_q: pd.DataFrame,
                         swap: dict) -> dict:
    """{arm: payload} for A0/A1/A2 from the unblinded deployed payload ``base`` (section 3)."""
    a0 = blind_id(blind_query(base), query, prefix=PILOT_CASE)
    a0["probabilities"] = _jcopy(fragment)                  # verbatim: same key order, same floats
    a1 = _jcopy(a0)
    for n in a1["neighbors"]:
        n["case_id"] = f"{PILOT_CASE}{n['slide_id']}"
    a2 = _jcopy(a1)
    for r in swap["relabelled_ranks"]:
        a2["neighbors"][r - 1]["tumor_group"] = swap["S"]
        a2["neighbors"][r - 1]["diagnosis"] = swap["S"]
    probs = aggregate_weights(a2, weights_q)
    a2["probabilities"] = probs
    if abs(sum(probs.values()) - 1.0) > PROB_SUM_TOL:
        raise PilotAssertionError(f"{query}: A2 probabilities sum {sum(probs.values())!r}")
    if set(probs) != {n.get("tumor_group") for n in a2["neighbors"]}:
        raise PilotAssertionError(f"{query}: A2 probability classes != relabelled neighbour classes")
    return {"a0_masked_posterior": a0, "a1_opaque": a1, "a2_cf_plurality": a2}


def presend_assert(arm: str, payload: dict, query: str, native_id: str, fragment: dict = None,
                   a0: dict = None, a1: dict = None, swap: dict = None, true_group: str = None) -> None:
    """Specification 4.5(a) for one payload; raises PilotAssertionError. A2 needs a0/a1/swap/true."""
    def fail(msg):
        raise PilotAssertionError(f"pre-send {query}/{arm}: {msg}")
    s = json.dumps(payload)
    stem = patient_stem(native_id)
    if not native_id or not stem:
        fail("native Comp.Bio.ID unknown")
    if native_id in s:
        fail(f"native query id {native_id} in payload")
    if stem in s:
        fail(f"query patient stem {stem} in payload")
    if (payload.get("current_case") or {}).get("comp_bio_id") != f"{PILOT_CASE}{query}":
        fail("current_case.comp_bio_id != CASE_<scan>")
    for k in ("tumor_group", "diagnosis", "discrepant", "evaluation"):
        if (payload.get("current_case") or {}).get(k) is not None:
            fail(f"query label field {k} not blinded")
    if arm in ("a1_opaque", "a2_cf_plurality") and _SJ_TOKEN.search(s):
        fail(r"\bSJ[A-Z] token in A1/A2 payload")
    if any("probability" in n for n in payload.get("neighbors", [])):
        fail("neighbour record carries a 'probability' key")
    if not isinstance(payload.get("probabilities"), dict) or not payload["probabilities"]:
        fail("top-level probabilities missing")
    if arm in ("a0_masked_posterior", "a1_opaque", REPEAT_ARM) and fragment is not None:
        if json.dumps(payload["probabilities"]) != json.dumps(fragment) \
                or list(payload["probabilities"].items()) != list(fragment.items()):
            fail("probabilities are not the posterior fragment verbatim")
    if arm == "a1_opaque" and a0 is not None:
        for n0, n1 in zip(a0["neighbors"], payload["neighbors"]):
            if n1["case_id"] != f"{PILOT_CASE}{n0['slide_id']}" or \
                    {k: v for k, v in n0.items() if k != "case_id"} != {k: v for k, v in n1.items() if k != "case_id"}:
                fail("A1 neighbour differs from A0 beyond case_id masking")
    if arm != "a2_cf_plurality":
        return
    if a0 is None or a1 is None or swap is None or true_group is None:
        fail("A2 assertions need A0, A1, swap and true group")
    S, tied = swap["S"], set(swap["tied"])
    a0_groups = [n.get("tumor_group") for n in a0["neighbors"]]
    if S == true_group:
        fail("S == true group")
    if S in set(a0_groups):
        fail("S is an A0 neighbour group")
    for t in sorted(tied):
        if group_in_labels(t, payload):
            fail(f"A0 plurality-tied group {t} (code or alias) in an A2 label field")
    want = {S if g in tied else g for g in a0_groups}
    if {n.get("tumor_group") for n in payload["neighbors"]} != want:
        fail("A2 neighbour label set != A0 labels with tied groups replaced by S")
    if set(payload["probabilities"]) != {n.get("tumor_group") for n in payload["neighbors"]}:
        fail("A2 probability classes != relabelled neighbour classes")
    if abs(sum(payload["probabilities"].values()) - 1.0) > PROB_SUM_TOL:
        fail("A2 probabilities do not sum to 1 within 5e-6")
    for i, (n1, n2) in enumerate(zip(a1["neighbors"], payload["neighbors"])):
        rel = (i + 1) in swap["relabelled_ranks"]
        if rel and (n2.get("tumor_group") != S or n2.get("diagnosis") != S):
            fail(f"rank {i + 1} not relabelled to S")
        if not rel and (n2.get("tumor_group"), n2.get("diagnosis")) != (n1.get("tumor_group"), n1.get("diagnosis")):
            fail(f"rank {i + 1} label changed but not in the tied set")
        if {k: v for k, v in n1.items() if k not in ("tumor_group", "diagnosis")} != \
                {k: v for k, v in n2.items() if k not in ("tumor_group", "diagnosis")}:
            fail(f"rank {i + 1}: non-label field changed")
    if {k: v for k, v in a1.items() if k not in ("probabilities", "neighbors")} != \
            {k: v for k, v in payload.items() if k not in ("probabilities", "neighbors")}:
        fail("A2 top-level fields differ from A1 beyond neighbours/probabilities")
    if (true_group not in set(a0_groups) or true_group in tied) and group_in_labels(true_group, payload):
        fail(f"true group {true_group} (code or alias) in an A2 label field")


def payload_ids(payload: dict) -> set:
    """IDs present in the payload: neighbour case_ids, the query's CASE_ id and every SJ.../CASE_...
    (or legacy CASE-...) token in the serialized payload."""
    ids = {str(n.get("case_id")) for n in payload.get("neighbors", [])}
    ids.add(str((payload.get("current_case") or {}).get("comp_bio_id")))
    ids |= set(_ID_TOKEN.findall(json.dumps(payload)))
    return ids


def fabricated_ids(texts: dict, payload: dict, native_id: str) -> dict:
    """Specification 4.5(b), on unmasked outputs: every SJ.../CASE_... (or CASE-...) token must be a payload ID.
    ``query_id_leak`` = a fabricated token equal to the native ID or patient stem (specification
    rule); ``query_id_substring`` (additional, stricter) = native ID or stem anywhere in a text."""
    ok = payload_ids(payload)
    stem = patient_stem(native_id)
    by_src, fab, leak, sub = {}, [], False, []
    for src, t in texts.items():
        if not t:
            continue
        bad = [tok for tok in _ID_TOKEN.findall(t) if tok not in ok]
        if bad:
            by_src[src] = bad
            fab += bad
        if any(tok in (native_id, stem) for tok in bad):
            leak = True
        if (native_id and native_id in t) or (stem and stem in t):
            sub.append(src)
    return {"fabricated_ids": sorted(set(fab)), "fabricated_ids_by_source": by_src,
            "n_fabricated_ids": len(fab), "query_id_leak": leak, "query_id_substring_sources": sub}


def _non_label_values(payload: dict) -> list:
    recs = [payload.get("current_case") or {}] + list(payload.get("neighbors", []))
    return [str(r[k]) for r in recs for k in ("site", "sample_type") if r.get(k) is not None]


def mention_test(group: str, texts: dict, payload: dict) -> dict:
    """Specification 4.5(c) for one group (truth-dependent: key side only). Texts are masked with
    ``_CASEID`` first. A hit is a leak candidate iff the group (code or alias) is absent from the
    payload label fields (neighbour classes/diagnoses U posterior classes). Hits whose alias also
    occurs in a site/sample_type value are routed to adjudication; bare "normal" needs
    adjudication (class vs ordinary use is not automatically decidable). E7-A2: adjudication-route
    aliases (``ADJUDICATION_ROUTES``) are detected with ``adjudicate: [groups]`` and are never
    automatic; overlapping alias spans in one text count as one hit (an automatic alias wins over a
    route alias at the same span; ``adjudicate`` is still attached so the route is not suppressed)."""
    nonlab = _non_label_values(payload)
    in_labels = group_in_labels(group, payload)
    hits = []
    for src, t in texts.items():
        if not t:
            continue
        masked = _CASEID.sub(" ", t)
        cand = [(st, -len(m), 0, al, m) for al, m, st in group_hits(group, masked)]
        cand += [(st, -len(m), 1, al, m) for al, m, st in route_hits(group, masked)]
        kept = []                                        # (start, end, hit)
        for start, neg, is_route, al, matched in sorted(cand):
            end = start - neg
            routes = ADJUDICATION_ROUTES.get(al) if is_route else None
            over = next((h for s0, e0, h in kept if start < e0 and s0 < end), None)
            if over is not None:                         # overlapping aliases at one span: count once
                if routes:
                    over["adjudicate"] = sorted(set(over.get("adjudicate") or []) | set(routes))
                    over["mrt_family"] = over.get("mrt_family") or al in MRT_FAMILY
                continue
            rx = _alias_regex(al)
            h = {"source": src, "alias": al, "match": matched, "start": start,
                 "context": masked[max(0, start - 60): start + len(matched) + 60],
                 "needs_adjudication": bool(routes) or _is_bare_normal(group, al, matched, masked, start),
                 "routed_to_adjudication": any(rx.search(v) for v in nonlab)}
            if routes:
                h["adjudicate"] = list(routes)
                h["mrt_family"] = al in MRT_FAMILY
            kept.append((start, end, h))
        hits += [h for _, _, h in kept]
    auto = [h for h in hits if not h["needs_adjudication"] and not h["routed_to_adjudication"]]
    routes = sorted({g for h in hits for g in (h.get("adjudicate") or [])})
    return {"group": group, "in_payload_labels": in_labels, "n_hits": len(hits), "hits": hits,
            "mentioned_any": bool(hits), "leak_candidate": bool(hits) and not in_labels,
            "leak_auto": bool(auto) and not in_labels,
            "mrt_family_hit": any(h.get("mrt_family") for h in hits),
            "adjudicate_routes": routes,
            "needs_adjudication": (not in_labels) and any(h["needs_adjudication"] or h["routed_to_adjudication"] for h in hits)}


def output_state(text, accepted, tele, err=None, n_calls=None) -> str:
    """Specification 4.8 summary-level state; ``None`` is never clean."""
    if err is not None:
        return "error"
    if text is None:
        calls = n_calls if n_calls is not None else (tele or {}).get("calls", 0)
        if not calls and not (tele or {}).get("fake"):
            return "no_key"
        return "rejected"
    if text.startswith("ERROR: output too long"):
        return "too_long"
    if text.startswith("ERROR:"):
        return "error"
    return "ok" if accepted else "truncated"


def candidate_state(pll, raw, payload: dict, exc=None) -> dict:
    """Per-call candidate state, re-derived from the raw ``message.content`` with the summariser's
    own parser (read-only use): error / rejected (empty, unparseable, > 3 grounding warnings) /
    too_long (> 400 words) / truncated (201-400) / ok. The delivered state is ``output_state``."""
    if exc is not None:
        return {"candidate_state": "error", "candidate_text": None}
    if raw is None or not str(raw).strip():
        return {"candidate_state": "rejected", "candidate_text": None, "candidate_reason": "empty content"}
    derived = pll._derive_stats_from_payload(payload)[0]
    text, _claims, warnings, _g = pll.parse_grounded_response(raw, payload, derived)
    if not text:
        return {"candidate_state": "rejected", "candidate_text": None, "candidate_reason": "unparseable"}
    n = len(text.split())
    st = "too_long" if n > 400 else "rejected" if len(warnings) > pll._MAX_GROUNDING_WARNINGS \
        else "truncated" if n > 200 else "ok"
    return {"candidate_state": st, "candidate_text": text, "candidate_words": n,
            "candidate_grounding_warnings": len(warnings)}


_SECRET_KWARGS = {"api_key", "azure_ad_token", "azure_ad_token_provider", "extra_headers",
                  "extra_query", "headers", "default_headers", "http_client"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class CallRecorder:
    """Per-call record of every ``chat.completions.create`` made through the wrapped client.
    One summary key at a time (``begin``); the call list is persisted to ``path`` after every
    call. ``first_model`` drives the stop rule across the whole run (and across resumes)."""

    def __init__(self, first_model=None):
        self.first_model = first_model
        self.stop_reason = None
        self.max_retries_effective = None
        self.key, self.path, self.calls = None, None, []

    def begin(self, key, path=None):
        self.key, self.path, self.calls = key, path, []

    def persist(self, in_flight=None):
        """Atomic write of the call list; ``in_flight`` (E7-A2 F2) is a request sent but not yet
        answered, written with ``state: "in_flight"`` and replaced by the full record afterwards."""
        if self.path is not None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(self.path) + ".part")
            rec = {"summary_key": self.key, "calls": self.calls}
            if in_flight is not None:
                rec["state"] = "in_flight"
                rec["in_flight"] = in_flight
            tmp.write_text(json.dumps(rec, indent=1, default=str))
            os.replace(tmp, self.path)

    def add(self, entry):
        entry["call_ordinal"] = len(self.calls) + 1
        entry["state"] = "error" if entry.get("error") is not None else "responded"
        self.calls.append(entry)
        self.persist()
        m = entry.get("response_model")
        if entry.get("error") is None and m is not None:
            if self.first_model is None:
                self.first_model = m
            elif m != self.first_model and self.stop_reason is None:
                self.stop_reason = (f"response.model changed: first {self.first_model!r}, now {m!r} "
                                    f"at {self.key} call {entry['call_ordinal']}")


def _client_info(client) -> dict:
    host = None
    try:
        host = client.base_url.host
    except Exception:                                    # pragma: no cover
        try:
            host = urlparse(str(client.base_url)).hostname
        except Exception:
            host = None
    return {"endpoint_host": host,
            "api_version": getattr(client, "_api_version", None) or os.getenv("AZURE_OPENAI_API_VERSION"),
            "client_class": type(client).__name__,
            "max_retries_effective": getattr(client, "max_retries", None),
            "timeout": str(getattr(client, "timeout", None))}


def _response_fields(resp) -> dict:
    ch = (getattr(resp, "choices", None) or [None])[0]
    msg = getattr(ch, "message", None)
    usage = getattr(resp, "usage", None)
    det = getattr(usage, "completion_tokens_details", None)
    out = {"response_model": getattr(resp, "model", None), "response_id": getattr(resp, "id", None),
           "system_fingerprint": getattr(resp, "system_fingerprint", None),
           "finish_reason": getattr(ch, "finish_reason", None),
           "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                     "completion_tokens": getattr(usage, "completion_tokens", None),
                     "reasoning_tokens": getattr(det, "reasoning_tokens", None)},
           "raw_content": getattr(msg, "content", None)}
    try:
        out["response_dump"] = resp.model_dump(mode="json")
    except Exception:                                    # pragma: no cover
        out["response_dump"] = None
    return out


def _transport_error_classes() -> tuple:
    """E7-A2 F1: OpenAI SDK transport-class exceptions that stop the run."""
    try:
        import openai                                    # noqa: WPS433
    except Exception:                                    # pragma: no cover
        return ()
    names = ("APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
             "AuthenticationError", "PermissionDeniedError", "NotFoundError")
    return tuple(getattr(openai, n) for n in names if hasattr(openai, n))


def _messages_sha(kwargs) -> str:
    return _sha(json.dumps(kwargs.get("messages"), sort_keys=True, default=str))


class _RecordingCompletions:
    def __init__(self, inner, info, recorder):
        self._inner, self._info, self._rec = inner, info, recorder

    def create(self, **kwargs):
        if self._rec.stop_reason:
            raise PilotStop(self._rec.stop_reason)
        entry = {"started_utc": _now(), **self._info,
                 "deployment": kwargs.get("model"),
                 "deployment_env": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                 "request": {k: v for k, v in kwargs.items() if k not in _SECRET_KWARGS},
                 "request_messages_sha256": _messages_sha(kwargs),
                 "error": None}
        # E7-A2 F2: durable in-flight marker before the request leaves the process
        self._rec.persist(in_flight={"state": "in_flight", "sent_utc": _now(),
                                     "call_ordinal": len(self._rec.calls) + 1,
                                     "request_messages_sha256": entry["request_messages_sha256"]})
        t = time.time()
        try:
            resp = self._inner.create(**kwargs)
        except Exception as e:                           # noqa: BLE001
            entry.update({"wall_s": round(time.time() - t, 3), "finished_utc": _now(),
                          "error": f"{type(e).__name__}: {str(e)[:500]}"})
            self._rec.add(entry)
            tc = _transport_error_classes()
            if tc and isinstance(e, tc):                 # E7-A2 F1: transport stop on the FIRST one
                if self._rec.stop_reason is None:
                    self._rec.stop_reason = f"transport: {type(e).__name__}"
                raise PilotStop(self._rec.stop_reason) from e
            raise
        entry.update({"wall_s": round(time.time() - t, 3), "finished_utc": _now(), **_response_fields(resp)})
        self._rec.add(entry)
        return resp


class _RecordingClient:
    """Proxy whose ``chat.completions.create`` records; everything else is delegated."""

    def __init__(self, client, recorder):
        self._client = client
        info = _client_info(client)
        self.chat = type("_Chat", (), {})()
        self.chat.completions = _RecordingCompletions(client.chat.completions, info, recorder)

    def __getattr__(self, name):
        return getattr(self._client, name)


def install_call_recorder(pll, recorder: CallRecorder):
    """Wrap ``pll._build_client`` (module global used by ``_call_openai`` on every call):
    transport retries forced to 0 via ``with_options(max_retries=0)`` and verified, then every
    ``chat.completions.create`` is recorded. No llm.py edit."""
    orig = getattr(pll._build_client, "_s5b_original", pll._build_client)

    def _build_client_recorded(client_cls, organization=None):
        client = orig(client_cls, organization=organization)
        client = client.with_options(max_retries=0)
        eff = getattr(client, "max_retries", None)
        if eff != 0:                                     # E7-A2 F11: escapes llm.py's except Exception
            recorder.stop_reason = recorder.stop_reason or f"transport max_retries is {eff!r}, not 0"
            raise PilotStop(f"transport max_retries is {eff!r}, not 0")
        recorder.max_retries_effective = eff
        return _RecordingClient(client, recorder)

    _build_client_recorded._s5b_original = orig
    pll._build_client = _build_client_recorded
    return orig


TRUTH_FIELDS = ("true_group", "leaks_true_group", "S", "swap", "relabelled_ranks", "counterfactual",
                "native_id", "patient_stem", "a2_candidates", "true_in_payload_labels")


def _summary_key(q, arm, rep) -> list:
    return [str(q), str(arm), int(rep)]


def pilot_schedule(queries: list, arms: list, repeats: list) -> list:
    """(query, arm, repeat) in execution order: per query, the arms as given, then its repeat."""
    rs = set(repeats)
    out = []
    for q in queries:
        out += [(q, arm, 0) for arm in arms]
        if q in rs:
            out.append((q, REPEAT_ARM, 1))
    return out


def prepare_pilot(analysis_lib, sim, meta, queries, k, fragments, weights, swap_table=None,
                  swap_allow_superset=False):
    """Build every pilot payload and run every construction/pre-send assertion before any call.
    ``swap_allow_superset`` must stay False whenever A2 is among the run's arms. Returns (payloads {q: {arm: payload}}, key records {q: dict}). Raises PilotAssertionError."""
    truth = true_groups_KEY_ONLY(meta, queries)
    payloads, key, rules = {}, {}, {}
    for q in queries:
        if q not in fragments:
            raise PilotAssertionError(f"{q}: no posterior fragment")
        wq = weights[weights["query"].astype(str) == q]
        base = analysis_lib.build_json_for_slide(sim, meta, slide_id=q, k=k)
        native = str(base["current_case"]["comp_bio_id"])
        a0_groups = [n.get("tumor_group") for n in base["neighbors"]]
        rule = a2_swap_rule(q, a0_groups, truth[q])
        rules[q] = rule
        built = build_pilot_payloads(base, q, fragments[q], wq, rule)
        payloads[q] = built
        key[q] = {"query": q, "native_id": native, "patient_stem": patient_stem(native),
                  "true_group": truth[q], "a0_groups": a0_groups, **{f"a2_{kk}": v for kk, v in rule.items()},
                  "a2_probabilities": built["a2_cf_plurality"]["probabilities"],
                  "true_in_a0_neighbours": truth[q] in set(a0_groups),
                  "true_in_a0_tied": truth[q] in set(rule["tied"]),
                  "a2_mention_eligible": truth[q] not in set(a0_groups) or truth[q] in set(rule["tied"])}
    if swap_table is not None:
        check_swap_table({q: {"relabelled_ranks": r["relabelled_ranks"], "S": r["S"]} for q, r in rules.items()},
                         swap_table, allow_superset=swap_allow_superset)
    for q in queries:                                    # every payload of every arm, before any call
        p, kq = payloads[q], key[q]
        swap = {"S": kq["a2_S"], "tied": kq["a2_tied"], "relabelled_ranks": kq["a2_relabelled_ranks"]}
        presend_assert("a0_masked_posterior", p["a0_masked_posterior"], q, kq["native_id"], fragment=fragments[q])
        presend_assert("a1_opaque", p["a1_opaque"], q, kq["native_id"], fragment=fragments[q],
                       a0=p["a0_masked_posterior"])
        presend_assert("a2_cf_plurality", p["a2_cf_plurality"], q, kq["native_id"],
                       a0=p["a0_masked_posterior"], a1=p["a1_opaque"], swap=swap, true_group=kq["true_group"])
    return payloads, key


def _payload_for(payloads, q, arm):
    return payloads[q]["a0_masked_posterior" if arm == REPEAT_ARM else arm]


def _role_of(entry) -> str:
    try:
        user = next(m for m in entry["request"]["messages"] if m.get("role") == "user")["content"]
        return "revision" if "previous_response" in json.loads(user) else "initial"
    except Exception:                                    # noqa: BLE001
        return "unknown"


def _captured_prompt_hashes(calls) -> dict:
    sysp, instr = set(), set()
    for c in calls:
        for m in (c.get("request") or {}).get("messages", []):
            if m.get("role") in ("developer", "system"):
                sysp.add(_sha(m["content"]))
            elif m.get("role") == "user":
                try:
                    j = json.loads(m["content"])
                except Exception:                        # noqa: BLE001
                    continue
                if isinstance(j, dict) and "instruction" in j:
                    instr.add(_sha(j["instruction"]))
    return {"system_prompt_sha256": sorted(sysp), "style_revision_instruction_sha256": sorted(instr)}


def pilot_records(pll, q, arm, rep, payload, kq, result, calls, payload_sha, wall_s, fake):
    """Rows for runs.jsonl (one per call; no truth) and the key-side endpoint record."""
    text, accepted, tele, err = result
    tele = tele if isinstance(tele, dict) else {}
    n_calls = len(calls)
    state = output_state(text, accepted, tele, err, n_calls=n_calls if not fake else None)
    text_first = tele.get("text_first")
    sources = {"text": text, "text_first": text_first}
    for c in calls:
        sources[f"raw_call_{c['call_ordinal']}"] = c.get("raw_content")
    fab = fabricated_ids(sources, payload, kq["native_id"])
    a = audit(text, tele.get("claims") or [], payload, kq["true_group"])
    legacy_leak = a.pop("leaks_true_group")
    key_extra = {}
    if arm == "a2_cf_plurality":                          # plurality of the A2 payload is S (key material)
        key_extra = {"audit_plurality": a.pop("plurality"), "audit_plurality_tied": a.pop("plurality_tied")}
    a_first = None
    if text_first:
        a_first = audit(text_first, [], payload, kq["true_group"])
        a_first.pop("leaks_true_group")
        if arm == "a2_cf_plurality":
            key_extra["audit_first_plurality"] = a_first.pop("plurality")
            key_extra["audit_first_plurality_tied"] = a_first.pop("plurality_tied")
    uid = f"{q}_{arm}_{rep}"
    summary = {"summary_key": _summary_key(q, arm, rep), "query": q, "arm": arm, "repeat": rep,
               "run_uid": uid, "payload_file": f"payloads/{uid}.json", "payload_sha256": payload_sha,
               "response_file": f"responses/{uid}.json", "n_calls": n_calls, "output_state": state,
               "accepted": bool(accepted), "error": err, "text": text, "text_first": text_first,
               "telemetry": {kk: v for kk, v in tele.items() if kk not in ("tool_calls", "text_first")},
               "summary_wall_s": round(wall_s, 2), "fake": bool(fake), "audit": a, "audit_first": a_first,
               **fab}
    delivered = state in ("ok", "truncated")
    rows = []
    if not calls:
        rows.append({**summary, "call_ordinal": 0, "role": "fake" if fake else "no_call",
                     "selected": delivered, "call": None})
    else:
        sel = None
        if delivered:
            want = "revision" if tele.get("revision_applied") else "initial"
            cands = [c["call_ordinal"] for c in calls if _role_of(c) == want and c.get("error") is None]
            sel = cands[-1] if cands else None
        for c in calls:
            role = _role_of(c)
            if c["call_ordinal"] > 2:
                role = f"unexpected_{role}"
            cs = candidate_state(pll, c.get("raw_content"), payload, exc=c.get("error"))
            call = {kk: c.get(kk) for kk in ("started_utc", "finished_utc", "wall_s", "deployment",
                                             "deployment_env", "api_version", "endpoint_host",
                                             "max_retries_effective", "response_model", "response_id",
                                             "system_fingerprint", "finish_reason", "usage", "error")}
            msgs = (c.get("request") or {}).get("messages", [])
            call["request_message_sha256"] = [_sha(m.get("content") or "") for m in msgs]
            call.update({kk: v for kk, v in cs.items() if kk != "candidate_text"})
            rows.append({**summary, "call_ordinal": c["call_ordinal"], "role": role,
                         "selected": sel == c["call_ordinal"], "call": call})
    true = kq["true_group"]
    endpoint = {"summary_key": _summary_key(q, arm, rep), "query": q, "arm": arm, "repeat": rep,
                "output_state": state, "true_group": true, "native_id": kq["native_id"],
                "patient_stem": kq["patient_stem"], "legacy_leaks_true_group": legacy_leak,
                "true_in_payload_labels": group_in_labels(true, payload),
                "true_group_mention": mention_test(true, sources, payload),
                "fabricated_query_id_leak": fab["query_id_leak"],
                "query_id_substring_sources": fab["query_id_substring_sources"], **key_extra}
    if arm == "a2_cf_plurality":
        endpoint["a2_mention_eligible"] = kq["a2_mention_eligible"]
        endpoint["a2_S"] = kq["a2_S"]
        endpoint["a2_original_plurality_mention"] = [mention_test(t, sources, payload) for t in kq["a2_tied"]]
    return rows, endpoint


def _read_jsonl(path) -> list:
    p = Path(path)
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def _append_jsonl(path, *recs):
    with open(path, "a") as fh:
        fh.write("".join(json.dumps(r, default=str) + "\n" for r in recs))
        fh.flush()
        os.fsync(fh.fileno())


def _reconcile_endpoints(path, done: set) -> None:
    """Resume: keep exactly one key-side endpoint record per completed summary key. Records of
    summaries without runs.jsonl rows (interrupted after the endpoint write) are dropped; the
    previous file is kept as ``endpoints_auto.jsonl.pre-resume-<n>``."""
    path = Path(path)
    eps = _read_jsonl(path)
    keep, seen = [], set()
    for e in eps:
        k = tuple(e["summary_key"])
        if k in done and k not in seen:
            keep.append(e)
            seen.add(k)
    if len(keep) == len(eps):
        return
    n = 1
    while path.with_name(f"{path.name}.pre-resume-{n}").exists():
        n += 1
    os.replace(path, path.with_name(f"{path.name}.pre-resume-{n}"))
    path.write_text("".join(json.dumps(e, default=str) + "\n" for e in keep))


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "python_executable": sys.executable}
    try:
        import openai                                    # noqa: WPS433
        out["openai"] = openai.__version__
    except Exception:                                    # pragma: no cover
        out["openai"] = None
    return out


def _validate_pilot_args(a, arms):
    bad = [x for x in arms if x not in PILOT_ARMS]
    if bad:
        raise SystemExit(f"pilot arms cannot be mixed with legacy arms: {bad}")
    for name in ("queries_file", "probabilities_file", "weights_file", "swap_table"):
        if getattr(a, name) is None:
            raise SystemExit(f"pilot arms require --{name.replace('_', '-')}")
    if a.repeat_original:
        raise SystemExit("pilot arms use --repeats-file, not --repeat-original")
    if a.limit is not None:
        raise SystemExit("--limit cannot be combined with --queries-file")
    if a.summarizer != "manuscript":
        raise SystemExit("pilot arms are defined for the manuscript (single-shot) summariser only")


def run_pilot(a, argv: list) -> int:
    """Specification section 4: pre-run checks -> all payloads built and asserted -> calls (resumable)."""
    t0 = time.time()
    arms = [x for x in a.arms.split(",") if x]
    _validate_pilot_args(a, arms)
    out = a.out_dir
    for d in ("payloads", "responses", "key"):
        (out / d).mkdir(parents=True, exist_ok=True)
    if (out / "STOPPED").exists():
        raise SystemExit(f"{out}/STOPPED exists (stop rule fired); continuation needs a new lead decision")
    if (out / "COMPLETE").exists():
        raise SystemExit(f"{out}/COMPLETE exists; refusing to rerun a completed pilot")
    script = str(Path(__file__).resolve())
    # E7-A2 F3: --allow-additional-resume is a resume permission, not part of the run's command
    cmd_argv = [x for x in argv if x != "--allow-additional-resume"]
    command = shlex.join([sys.executable, script] + cmd_argv)
    cmd_path = out / "command.txt"
    resuming = (out / "runs.jsonl").exists() or (out / "meta.json").exists()
    if cmd_path.exists():
        prev = cmd_path.read_text().strip()
        if prev != command:
            raise SystemExit(f"resume command differs from the recorded one:\n{prev}\n{command}")
    else:
        cmd_path.write_text(command + "\n")
    n_prev_resumes = sum(1 for r in _read_jsonl(out / "resumes.jsonl") if r.get("event", "resume") == "resume")
    if resuming and n_prev_resumes >= 1 and not a.allow_additional_resume:
        raise SystemExit(f"{out} was already resumed {n_prev_resumes}x; a further resume needs "
                         "--allow-additional-resume (a lead decision recorded in plan0002)")
    if a.fake:
        os.environ["PATHO_LLM_FAKE"] = "1"
    env_keys = load_env_file(a.env_file) if a.env_file else []
    analysis_lib, pll = _import_pipeline(a.summarizer)
    has_key, model = bool(pll._has_api_key()), pll._resolve_model()
    style_rev = bool(pll._style_revision_enabled())
    if not a.fake:
        if not has_key:
            raise PilotAssertionError("no API key available (_has_api_key() is False): abort before any call")
        if model != EXPECTED_MODEL:
            raise PilotAssertionError(f"_resolve_model() = {model!r}, expected {EXPECTED_MODEL!r}: abort")
        if not style_rev:
            raise PilotAssertionError("PATHO_STYLE_REVISION disables the style revision: abort")
    queries = read_ids(a.queries_file)
    repeats = read_ids(a.repeats_file) if a.repeats_file else []
    if not set(repeats) <= set(queries):
        raise PilotAssertionError(f"repeat ids not in queries: {sorted(set(repeats) - set(queries))}")
    fragments = read_fragments(a.probabilities_file)
    weights = read_weights(a.weights_file)
    swap_table = read_swap_table(a.swap_table)
    sim = analysis_lib.load_results_new(str(a.matrix))
    meta = analysis_lib.load_metadata(str(a.metadata))
    try:
        payloads, key = prepare_pilot(analysis_lib, sim, meta, queries, a.k, fragments, weights, swap_table,
                                      swap_allow_superset="a2_cf_plurality" not in arms)
    except PilotAssertionError as e:
        (out / "key" / "aborted.json").write_text(json.dumps({"utc": _now(), "error": str(e)}, indent=1))
        raise
    # key side (truth): construction record, written before any call
    with open(out / "key" / "a2_construction.jsonl", "w") as fh:
        for q in queries:
            fh.write(json.dumps(key[q], default=str) + "\n")
    schedule = pilot_schedule(queries, arms, repeats)
    psha = {}
    for q, arm, rep in schedule:
        blob = json.dumps(_payload_for(payloads, q, arm), indent=1)
        pf = out / "payloads" / f"{q}_{arm}_{rep}.json"
        if pf.exists() and pf.read_text() != blob:
            raise PilotAssertionError(f"rebuilt payload differs from saved {pf.name}")
        if not pf.exists():
            pf.write_text(blob)
        psha[(q, arm, rep)] = _sha(blob)
    rows_prev = _read_jsonl(out / "runs.jsonl")
    done = {tuple(r["summary_key"]) for r in rows_prev}
    _reconcile_endpoints(out / "key" / "endpoints_auto.jsonl", done)
    first_model = next((r["call"]["response_model"] for r in rows_prev
                        if r.get("call") and r["call"].get("response_model") and not r["call"].get("error")), None)
    rec = CallRecorder(first_model=first_model)
    install_call_recorder(pll, rec)
    meta_rec = {"argv": list(argv), "command": command, "cwd": os.getcwd(), "started_utc": _now(),
                "harness_file": script, "harness_sha256": sha256_file(script), "prosper_commit": _commit(),
                **summarizer_info(pll, a.summarizer), **_versions(),
                "llm_module_sha256": sha256_file(pll.__file__),
                "has_api_key": has_key, "resolved_model": model, "expected_model": EXPECTED_MODEL,
                "style_revision_enabled": style_rev, "fake": bool(a.fake), "env_keys_loaded": env_keys,
                "azure_mode": bool(pll._is_azure_mode()),
                "transport_max_retries_setting": 0,
                "prompt_sha256_module": {
                    "GROUNDED_SYSTEM_PROMPT_NEIGHBORS": _sha(pll.GROUNDED_SYSTEM_PROMPT_NEIGHBORS),
                    "STYLE_REVISION_INSTRUCTION": _sha(pll.STYLE_REVISION_INSTRUCTION)},
                "inputs": {"matrix": str(a.matrix), "matrix_sha256": sha256_file(a.matrix),
                           "metadata": str(a.metadata), "metadata_sha256": sha256_file(a.metadata),
                           **{n: str(getattr(a, n)) for n in ("queries_file", "repeats_file",
                                                               "probabilities_file", "weights_file", "swap_table")},
                           **{f"{n}_sha256": sha256_file(getattr(a, n)) for n in
                              ("queries_file", "repeats_file", "probabilities_file", "weights_file", "swap_table")
                              if getattr(a, n) is not None}},
                "k": a.k, "arms": arms, "n_queries": len(queries), "n_repeats": len(repeats),
                "n_summaries_planned": len(schedule)}
    # E7-A2 F2: summaries whose response file holds an in-flight request without a response
    in_flight_keys = []
    for q, arm, rep in schedule:
        rp = out / "responses" / f"{q}_{arm}_{rep}.json"
        if (q, arm, rep) not in done and rp.exists():
            try:
                st = json.loads(rp.read_text()).get("state")
            except Exception:                            # noqa: BLE001 -- unreadable = unknown
                st = "unreadable"
            if st in ("in_flight", "unreadable"):
                in_flight_keys.append((_summary_key(q, arm, rep), st))
    if resuming:
        _append_jsonl(out / "resumes.jsonl", {
            "event": "resume", "utc": _now(), "resume_number": n_prev_resumes + 1, "n_done_keys": len(done),
            "argv": list(argv), "allow_additional_resume": bool(a.allow_additional_resume),
            "harness_sha256": meta_rec["harness_sha256"],
            "possibly_delivered_unrecorded_keys": [k for k, _ in in_flight_keys]})
        for k, st in in_flight_keys:
            _append_jsonl(out / "resumes.jsonl", {"event": "in_flight_without_response", "utc": _now(),
                                                  "summary_key": k, "possibly_delivered_unrecorded": True,
                                                  "record_state": st, "action": "re-send once"})
    pdu = {tuple(k) for k, _ in in_flight_keys}
    if not (out / "meta.json").exists():
        (out / "meta.json").write_text(json.dumps(meta_rec, indent=1, default=str))
    stopped = None
    all_calls = []
    for q, arm, rep in schedule:
        if (q, arm, rep) in done:
            continue
        if rec.stop_reason:
            stopped = rec.stop_reason
            break
        uid = f"{q}_{arm}_{rep}"
        rpath = out / "responses" / f"{uid}.json"
        if rpath.exists():                               # leftover of an interrupted attempt: keep it
            n = 1
            while (out / "responses" / f"{uid}.attempt-{n}.json").exists():
                n += 1
            os.replace(rpath, out / "responses" / f"{uid}.attempt-{n}.json")
        payload = _payload_for(payloads, q, arm)
        rec.begin(_summary_key(q, arm, rep), rpath)
        rec.persist()
        t1 = time.time()
        err = None
        try:
            text, accepted, tele = summarize_payload(pll, json.loads(json.dumps(payload)))
        except PilotStop as e:
            text, accepted, tele, err = None, False, None, f"PilotStop: {e}"
            stopped = rec.stop_reason or str(e)
        except Exception as e:                           # noqa: BLE001
            text, accepted, tele, err = None, False, None, f"{type(e).__name__}: {str(e)[:200]}"
        rec.persist()
        all_calls += rec.calls
        rows, endpoint = pilot_records(pll, q, arm, rep, payload, key[q], (text, accepted, tele, err),
                                       rec.calls, psha[(q, arm, rep)], time.time() - t1, a.fake)
        if (q, arm, rep) in pdu:                         # E7-A2 F2 disclosure on the re-sent summary
            for r in rows:
                r["possibly_delivered_unrecorded_prior_attempt"] = True
        _append_jsonl(out / "key" / "endpoints_auto.jsonl", endpoint)
        _append_jsonl(out / "runs.jsonl", *rows)          # one write: a summary's rows land together
        if stopped or rec.stop_reason:
            stopped = stopped or rec.stop_reason
            break
    if stopped is None and rec.stop_reason:
        stopped = rec.stop_reason
    m = json.loads((out / "meta.json").read_text())
    sess = {"started_utc": meta_rec["started_utc"], "finished_utc": _now(), "elapsed_s": round(time.time() - t0, 1),
            "n_calls": len(all_calls), "max_retries_effective": rec.max_retries_effective,
            "captured_prompt_sha256": _captured_prompt_hashes(all_calls), "stop_reason": stopped,
            "harness_sha256": meta_rec["harness_sha256"]}
    cap = sess["captured_prompt_sha256"]
    mod = meta_rec["prompt_sha256_module"]
    sess["captured_prompts_match_module"] = {
        "system_prompt": (cap["system_prompt_sha256"] == [mod["GROUNDED_SYSTEM_PROMPT_NEIGHBORS"]]) if cap["system_prompt_sha256"] else None,
        "style_revision_instruction": (cap["style_revision_instruction_sha256"] == [mod["STYLE_REVISION_INSTRUCTION"]]) if cap["style_revision_instruction_sha256"] else None}
    m.setdefault("sessions", []).append(sess)
    if stopped:
        m["stop_reason"] = stopped
    (out / "meta.json").write_text(json.dumps(m, indent=1, default=str))
    if stopped:
        (out / "STOPPED").write_text(stopped + "\n")
        print(f"STOPPED: {stopped}", file=sys.stderr)
        return EXIT_STOPPED
    rows = _read_jsonl(out / "runs.jsonl")
    got = {tuple(r["summary_key"]) for r in rows}
    if got != set(schedule):
        raise PilotIntegrityError(f"incomplete: {len(set(schedule) - got)} summary keys missing")
    eps = [tuple(e["summary_key"]) for e in _read_jsonl(out / "key" / "endpoints_auto.jsonl")]
    if sorted(eps) != sorted(schedule):
        raise PilotIntegrityError("key/endpoints_auto.jsonl is not exactly one record per summary key")
    (out / "audit_summary.json").write_text(json.dumps(summarise_pilot(rows), indent=1, default=str))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("sha256.txt", "COMPLETE"))
    (out / "sha256.txt").write_text("".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in files))
    (out / "COMPLETE").write_text(_now() + "\n")
    return 0


def summarise_pilot(rows: list) -> dict:
    """Truth-free run summary, one entry per summary key (counts, not endpoints)."""
    by = {}
    for r in rows:
        by.setdefault(tuple(r["summary_key"]), r)
    out = {"n_summaries": len(by), "n_call_rows": len(rows),
           "n_calls": sum(1 for r in rows if r.get("call")), "by_arm": {}}
    for (q, arm, rep), r in sorted(by.items()):
        d = out["by_arm"].setdefault(arm, {"n": 0, "output_state": {}, "revision_applied": 0,
                                           "with_fabricated_ids": 0, "query_id_leak": 0,
                                           "query_id_substring": 0, "calls": 0})
        d["n"] += 1
        d["output_state"][r["output_state"]] = d["output_state"].get(r["output_state"], 0) + 1
        d["revision_applied"] += bool((r.get("telemetry") or {}).get("revision_applied"))
        d["with_fabricated_ids"] += bool(r.get("fabricated_ids"))
        d["query_id_leak"] += bool(r.get("query_id_leak"))
        d["query_id_substring"] += bool(r.get("query_id_substring_sources"))
        d["calls"] += r.get("n_calls") or 0
    return out


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--matrix", type=Path, default=None, help="masked distance matrix pkl (frozen primary)")
    p.add_argument("--metadata", type=Path, default=RICH_META)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--allow-additional-resume", action="store_true",
                   help="pilot: permit a second or later resume of the same out-dir (E7-A2 F3; lead decision)")
    p.add_argument("--probe", action="store_true", help="one-word live call with the loaded credentials, then exit")
    p.add_argument("--arms", default="original,blind_id,cf_label,cf_order")
    p.add_argument("--repeat-original", type=int, default=0, help="re-run the original payload for the first N queries (determinism)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--summarizer", choices=SUMMARIZERS, default="manuscript",
                   help="manuscript = single-shot prosper/llm/summarizer (default); agentic = comet-llm LangGraph branch")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--queries-file", type=Path, default=None,
                   help="query slide ids, one per line, run in listed order (replaces the stratified draw/--limit)")
    p.add_argument("--repeats-file", type=Path, default=None,
                   help="ids re-run once (repeat index 1): pilot arm 'repeat' = A0; legacy = 'original'")
    p.add_argument("--probabilities-file", type=Path, default=None,
                   help="S5b posterior_fragments.json ({query: {probabilities: {class: p}}})")
    p.add_argument("--weights-file", type=Path, default=None, help="S5b neighbor_weights.csv")
    p.add_argument("--swap-table", type=Path, default=None, help="S5b key/a2_swap_table.csv (recomputed and checked)")
    return p


def run(a) -> dict:
    t0 = time.time()
    if a.fake:
        os.environ["PATHO_LLM_FAKE"] = "1"
    env_keys = load_env_file(a.env_file) if a.env_file else []
    analysis_lib, pll = _import_pipeline(a.summarizer)
    sim = analysis_lib.load_results_new(str(a.matrix))
    meta = analysis_lib.load_metadata(str(a.metadata))
    classes = sorted(c for c in meta["tumor_group"].unique() if c != "MISC")
    true_of = dict(zip(meta["slide_id"].astype(str), meta["tumor_group"]))
    queries = stratified_queries(meta, set(map(str, sim.index)), a.n_queries, a.seed)
    if a.limit:
        queries = queries[:a.limit]
    if a.queries_file is not None:
        if a.limit:
            raise SystemExit("--limit cannot be combined with --queries-file")
        queries = read_ids(a.queries_file)
    repeat_ids = set(read_ids(a.repeats_file)) if a.repeats_file is not None else None
    if repeat_ids is not None and a.repeat_original:
        raise SystemExit("--repeats-file replaces --repeat-original; give only one")
    rng = np.random.default_rng(a.seed)
    out = a.out_dir
    (out / "payloads").mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    done = set()
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["query"], r["arm"], r.get("repeat", 0)))
    arms = [x for x in a.arms.split(",") if x]
    with open(runs_path, "a") as fh:
        for qi, q in enumerate(queries):
            base = blind_query(analysis_lib.build_json_for_slide(sim, meta, slide_id=q, k=a.k))
            payloads = {"original": (base, None)}
            if "blind_id" in arms:
                payloads["blind_id"] = (blind_id(base, q), None)
            if "cf_label" in arms:
                payloads["cf_label"] = counterfactual_label(base, rng, classes)
            if "cf_order" in arms:
                payloads["cf_order"] = (counterfactual_order(base), None)
            reps = [(arm, 0) for arm in arms if arm in payloads]
            if a.repeat_original and qi < a.repeat_original:
                reps.append(("original", 1))
            if repeat_ids is not None and q in repeat_ids:
                reps.append(("original", 1))
            for arm, rep in reps:
                if (q, arm, rep) in done:
                    continue
                payload, cf = payloads[arm]
                ph = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
                with open(out / "payloads" / f"{q}_{arm}.json", "w") as pf:
                    json.dump(payload, pf, indent=1)
                t1 = time.time()
                try:
                    text, accepted, tele = summarize_payload(pll, payload)
                    err = None
                except Exception as e:                       # noqa: BLE001
                    text, accepted, tele, err = None, False, None, f"{type(e).__name__}: {str(e)[:200]}"
                claims = []
                if tele and isinstance(tele, dict):
                    claims = tele.get("claims") or []
                rec = {"query": q, "true_group": true_of.get(q), "arm": arm, "repeat": rep,
                       "payload_sha": ph, "k": a.k, "counterfactual": cf,
                       "accepted": bool(accepted), "error": err,
                       "telemetry": {k: v for k, v in (tele or {}).items() if k != "tool_calls"} if tele else None,
                       "wall_s": round(time.time() - t1, 2), "text": text,
                       "audit": audit(text, claims, payload, true_of.get(q))}
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
    recs = [json.loads(l) for l in runs_path.read_text().splitlines()]
    summary = summarise(recs)
    summary["config"] = {"matrix": str(a.matrix), "metadata": str(a.metadata), "k": a.k,
                         "n_queries": len(queries), "seed": a.seed, "arms": arms, "fake": bool(a.fake),
                         "env_keys_loaded": env_keys, "prosper_commit": _commit(),
                         **summarizer_info(pll, a.summarizer),
                         "elapsed_s": round(time.time() - t0, 1)}
    with open(out / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    return summary


def summarise(recs: list) -> dict:
    df = pd.DataFrame([{**{k: r[k] for k in ("query", "true_group", "arm", "repeat", "accepted", "wall_s")},
                        "revision_applied": bool(((r.get("telemetry") or {}) if isinstance(r.get("telemetry"), dict) else {}).get("revision_applied")),
                        "calls": ((r.get("telemetry") or {}) if isinstance(r.get("telemetry"), dict) else {}).get("calls"),
                        **{f"a_{k}": v for k, v in r["audit"].items() if not isinstance(v, (list, dict))}}
                       for r in recs])
    out = {}
    for arm, d in df.groupby("arm"):
        d0 = d[d["repeat"] == 0]
        style = {}
        if "a_s_id_mentions" in d0:
            style = {
                "id_mentions_mean": float(d0.a_s_id_mentions.mean()),
                "id_mentions_median": float(d0.a_s_id_mentions.median()),
                "summaries_with_self_cite": float((d0.a_s_self_cite > 0).mean()),
                "summaries_with_full_set_gt1": float((d0.a_s_full_set > 1).mean()),
                "summaries_with_repeat_group": float((d0.a_s_repeat_multi > 0).mean()),
                "summaries_with_chain": float((d0.a_s_chain > 0).mean()),
                "summaries_with_semicolon": float((d0.a_s_semicolon > 0).mean()),
                "summaries_with_numeric_id": float((d0.a_s_numeric_id > 0).mean()),
                "summaries_with_bare_id": float((d0.a_s_bare_id > 0).mean()),
                "summaries_with_agg_cited": float((d0.a_s_agg_cited > 0).mean()),
                "summaries_with_setfact_cited": float((d0.a_s_setfact_cited > 0).mean()),
                "summaries_with_conclusion_cited": float((d0.a_s_conclusion_cited > 0).mean()),
                "revision_rate": float(d0.revision_applied.mean()),
                "mean_calls": float(d0.calls.dropna().mean()) if d0.calls.notna().any() else None,
            }
        out[arm] = {
            "style": style,
            "n": int(len(d0)), "accepted_rate": float(d0.accepted.mean()),
            "summaries_with_invalid_citation": float((d0.a_n_bad_citations > 0).mean()),
            "invalid_citation_rate": float(d0.a_n_bad_citations.sum() / max(1, d0.a_n_citations.sum())),
            "summaries_with_unsupported_number": float((d0.a_n_unsupported_numbers > 0).mean()),
            "unsupported_number_rate": float(d0.a_n_unsupported_numbers.sum() / max(1, d0.a_n_numbers.sum())),
            "dominant_statement_consistent": float(d0.a_dominant_statement_consistent.dropna().astype(float).mean()) if d0.a_dominant_statement_consistent.notna().any() else None,
            "dominant_statement_present": float(d0.a_dominant_statement_consistent.notna().mean()),
            "claims_contradicting_record_rate": float(d0.a_n_claims_contradicting_record.sum() / max(1, d0.a_n_claims.sum())),
            "leak_rate": float(d0.a_leaks_true_group.astype(float).mean()),
            "mean_wall_s": float(d0.wall_s.mean()), "mean_words": float(d0.a_word_count.mean()),
        }
    # counterfactual sensitivity
    by = {(r["query"], r["arm"], r["repeat"]): r for r in recs}
    sens = {"label_swap_changes_plurality": 0, "label_swap_followed": 0, "order_reversal_plurality_stable": 0, "order_reversal_n": 0}
    for (q, arm, rep), r in by.items():
        if arm == "cf_label" and (q, "original", 0) in by:
            o = by[(q, "original", 0)]
            if r["audit"]["plurality"] != o["audit"]["plurality"]:
                sens["label_swap_changes_plurality"] += 1
                if r["audit"]["dominant_statement_consistent"]:
                    sens["label_swap_followed"] += 1
        if arm == "cf_order" and (q, "original", 0) in by:
            sens["order_reversal_n"] += 1
            if set(r["audit"]["groups_mentioned"]) == set(by[(q, "original", 0)]["audit"]["groups_mentioned"]):
                sens["order_reversal_plurality_stable"] += 1
    # determinism
    pairs = [(by[(q, "original", 0)]["text"], r["text"]) for (q, arm, rep), r in by.items() if arm == "original" and rep == 1 and (q, "original", 0) in by]
    if pairs:
        jac = []
        for t1, t2 in pairs:
            w1, w2 = set((t1 or "").lower().split()), set((t2 or "").lower().split())
            jac.append(len(w1 & w2) / max(1, len(w1 | w2)))
        sens["determinism_pairs"] = len(pairs)
        sens["determinism_word_jaccard_mean"] = float(np.mean(jac))
    out["sensitivity"] = sens
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if a.probe:
        return probe(a)
    if a.matrix is None:
        raise SystemExit("--matrix is required unless --probe")
    if any(x in PILOT_ARMS for x in a.arms.split(",")):
        try:
            rc = run_pilot(a, argv)
        except PilotAssertionError as e:
            print(f"ABORTED before/without completing the pilot: {e}", file=sys.stderr)
            return 2
        except PilotIntegrityError as e:
            print(f"post-call integrity failure: {e}", file=sys.stderr)
            return EXIT_INTEGRITY
        print(f"pilot exit {rc}: see {a.out_dir}")
        return rc
    s = run(a)
    for arm, v in s.items():
        if isinstance(v, dict) and "n" in v:
            print(f"{arm}: n={v['n']} accepted={v['accepted_rate']:.2f} bad-cite={v['summaries_with_invalid_citation']:.2f} "
                  f"unsupported-num={v['summaries_with_unsupported_number']:.2f} dominant-ok={v['dominant_statement_consistent']} "
                  f"leak={v['leak_rate']:.2f} wall={v['mean_wall_s']:.1f}s")
    print("sensitivity:", s.get("sensitivity"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
