"""Optional LLM rewrite support with grounded citations.

Supports both standard OpenAI and Azure AI Foundry backends.
Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY to use Azure AI Foundry;
otherwise falls back to standard OpenAI with OPENAI_API_KEY.

Separated so tests can monkeypatch _call_openai without performing network calls.
"""
from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional

from .grounding import (
    GROUNDED_SYSTEM_PROMPT_NEIGHBORS,
    MEDICAL_DISCLAIMER,
    STYLE_REVISION_INSTRUCTION,
    extract_reference_values,
    loads_lenient,
    normalize_citations,
    parse_grounded_response,
    provenance_guard,
    verify_citation_style,
)


def _is_azure_mode() -> bool:
    """Return True when Azure OpenAI environment variables are configured."""
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT"))


def _has_api_key() -> bool:
    """Return True when any usable API key is present (Azure or OpenAI)."""
    if _is_azure_mode():
        return bool(os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = (
    "Using provided stats, write a clinician-friendly narrative (≤200 words) summarizing: dominant diagnosis with vote counts, score/probability mean and range (use higher-is-better if probability, lower-is-better if similarity), highest probability (if available) with tumor_type/diagnosis, outlier cases (specify their case_id and why: divergent diagnosis), and current_case metadata (age, sex, site). If probability is provided, do NOT mention similarity. Use cautious language ('supports'/'suggests'). Prefer JSON {\"summary\": \"...\"}."
)

SYSTEM_PROMPT_NEIGHBORS = (
    "You are given a pathology neighbor set payload containing: current_case demographics (age, sex, site, tumor_group, diagnosis, met, treatment, discrepant), and neighbors each with similarity score (lower is better) or probability (higher is better), plus the same metadata fields. Produce a concise clinician-friendly summary (≤200 words) that: \n"
    "1. Opens with a sentence describing the current case including ALL of: age, sex, site, diagnosis, tumor_group, met status, treatment status (Y=treated, N=untreated), and discrepant flag (Y=discrepant, N=concordant). \n"
    "2. States the apparent dominant diagnosis and tumor_group support (vote counts). \n"
    "3. Mentions average and range of similarity/probability (use the correct direction). \n"
    "4. States the highest probability (if provided) and the associated tumor_type or diagnosis. \n"
    "5. Notes notable discrepant neighbors (different diagnosis or tumor_group, or discrepant=='Y'). \n"
    "6. Highlights any contrasting neighbor features (e.g., metastatic vs primary, ped_adult differences). \n"
    "7. Uses cautious verbs ('supports', 'suggests'). \n"
    "8. If a 'probabilities' key is present at the top level, mention the Bayesian disease probabilities. \n"
    "If probability is provided, do NOT mention similarity. Return JSON of shape {\"summary\": \"...\"}. If impossible, return a single plain text paragraph. Do NOT invent data beyond payload."
)


def _postprocess_probability_summary(stats: Dict[str, Any], text: str) -> str:
    """Ensure probability phrasing is present when probability is the score.

    - If probability is present and output mentions similarity, replace that wording.
    - If probability is present and output doesn't mention probability, append a top-probability sentence.
    """
    score_name = stats.get("score_name") or ("probability" if stats.get("mean_probability") is not None else "similarity")
    if score_name != "probability":
        return text
    lower = text.lower()
    if "similarity" in lower:
        text = text.replace("similarity", "probability").replace("Similarity", "Probability")
    if "lower is better" in text:
        text = text.replace("lower is better", "higher is better")
    if "highest probability" not in text.lower():
        top_score = stats.get("top_score")
        tumor_label = stats.get("top_tumor_type") or stats.get("top_diagnosis") or "unknown"
        if top_score is not None:
            cite = f" [{stats['top_case_id']}]" if stats.get("top_case_id") else ""
            text = f"{text} Highest probability {float(top_score):.3f} for {tumor_label}{cite}."
    return text


def _score_from_stats(stats: Dict[str, Any]) -> tuple[str, float, float, float, str]:
    """Return (score_name, mean, min, max, direction) with fallbacks."""
    score_name = stats.get("score_name") or ("probability" if stats.get("mean_probability") is not None else "similarity")
    mean_score = stats.get("mean_score")
    min_score = stats.get("min_score")
    max_score = stats.get("max_score")
    if mean_score is None:
        mean_score = stats.get("mean_probability") if score_name == "probability" else stats.get("mean_similarity")
    if min_score is None:
        min_score = stats.get("min_probability") if score_name == "probability" else stats.get("min_similarity")
    if max_score is None:
        max_score = stats.get("max_probability") if score_name == "probability" else stats.get("max_similarity")
    direction = stats.get("score_direction") or ("higher" if score_name == "probability" else "lower")
    return score_name, float(mean_score or 0.0), float(min_score or 0.0), float(max_score or 0.0), direction


def _safe_import_openai():  # pragma: no cover - trivial
    """Import and return the appropriate OpenAI client class.

    For Azure AI Foundry we use AzureOpenAI; otherwise standard OpenAI.
    """
    try:
        if _is_azure_mode():
            from openai import AzureOpenAI  # type: ignore
            return AzureOpenAI
        from openai import OpenAI  # type: ignore
        return OpenAI
    except Exception:  # noqa: BLE001
        return None


def _build_client(client_cls, organization: str | None = None):
    """Construct the OpenAI client from environment variables.

    For Azure AI Foundry we use AzureOpenAI with the azure_endpoint,
    api_key, and api_version.  For standard OpenAI we use the regular client.
    """
    if _is_azure_mode():
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        return client_cls(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    # Standard OpenAI
    if organization:
        try:
            return client_cls(organization=organization)
        except TypeError:  # fallback if test stub or SDK variant rejects arg
            return client_cls()
    return client_cls()


def _resolve_model() -> str:
    """Return the model / deployment name to use."""
    if _is_azure_mode():
        return os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("PATHO_LLM_MODEL", "o3-mini")
    return os.getenv("PATHO_LLM_MODEL", "gpt-4o-mini")


def _call_openai(client_cls, system_prompt: str, user_content: str, model: str, organization: str | None = None) -> str:
    """Instantiate client and perform request returning raw content string.

    Supports both standard OpenAI and Azure AI Foundry backends.
    Isolated for monkeypatching in tests.
    """
    client = _build_client(client_cls, organization=organization)
    # Reasoning models (o1, o3, o4, ...) have limited parameter support
    _reasoning_prefixes = ('o1', 'o3', 'o4', 'gpt-5')
    is_reasoning = any(model.startswith(p) for p in _reasoning_prefixes)

    # For reasoning models: use developer message instead of system, no
    # response_format, no temperature, and use max_completion_tokens
    if is_reasoning:
        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=16000,
        )
    else:
        kwargs = dict(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _reference_from_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Build a reference-values dict from a legacy *stats* dict.

    This allows the provenance guard to work on the simpler
    ``attempt_llm_summary`` path that only receives computed statistics.
    """
    numeric_values: set[float] = set()
    for key in (
        "mean_score", "min_score", "max_score", "top_score",
        "mean_similarity", "min_similarity", "max_similarity",
        "mean_probability", "min_probability", "max_probability",
    ):
        v = stats.get(key)
        if v is not None:
            numeric_values.add(float(v))

    case_ids: set[str] = set()
    for o in stats.get("outliers", []):
        cid = o.get("case_id")
        if cid is not None:
            case_ids.add(str(cid))
    top_cid = stats.get("top_case_id")
    if top_cid is not None:
        case_ids.add(str(top_cid))

    diagnoses: set[str] = set()
    for dx in stats.get("diagnosis_counts", {}).keys():
        diagnoses.add(dx)
    top_dx = stats.get("top_diagnosis")
    if top_dx:
        diagnoses.add(top_dx)

    return {
        "case_ids": case_ids,
        "diagnoses": diagnoses,
        "tumor_groups": set(),
        "numeric_values": numeric_values,
    }


def attempt_llm_summary(stats: Dict[str, Any]) -> tuple[Optional[str], bool]:
    """Attempt LLM rewrite.

    Returns (text, accepted_flag):
    - text: raw or truncated LLM output if call succeeded, else an error string or None.
    - accepted_flag: True if original output <=200 words; False if truncated or error.

    Behavior:
    - Accept JSON {"summary": "..."} or raw text.
    - If >200 words but <400, truncate to 200 + ellipsis (returned, accepted_flag False).
    - If >=400 words treat as error.
    - Fake mode returns a synthetic sentence reflecting stats, respecting 200-word limit.
    - Errors surface as ERROR: <Type>: <message>.
    """
    if not _has_api_key():
        return (None, False)
    # Fake mode to bypass network for testing/demo
    if os.getenv("PATHO_LLM_FAKE"):
        # Build richer fake narrative using stats keys
        dx = stats.get("mode_diagnosis") or "uncertain diagnosis"
        counts = stats.get("diagnosis_counts", {})
        score_name, mean_sim, min_sim, max_sim, direction = _score_from_stats(stats)
        outliers = stats.get("outliers", [])
        current = stats.get("current_case", {})
        better_phrase = "higher is better" if direction == "higher" else "lower is better"
        top_prob = stats.get("top_score") if score_name == "probability" else None
        top_dx = stats.get("top_diagnosis")
        top_tumor = stats.get("top_tumor_type")
        parts = [
            f"Neighbors support {dx} (votes: {counts}).",
            f"{score_name.title()} mean {mean_sim:.2f} range {min_sim:.2f}-{max_sim:.2f} ({better_phrase}).",
            f"Current case metadata: site={current.get('site')}, age={current.get('age')}, sex={current.get('sex')}.",
        ]
        if top_prob is not None:
            tumor_label = top_tumor or top_dx or "unknown"
            parts.append(f"Highest probability {top_prob:.2f} for {tumor_label}.")
        if outliers:
            detail = "; ".join(
                f"{o.get('case_id')}({('low' if o.get('low_similarity') else 'diff')}: {o.get('diagnosis')}, {score_name} {o.get('score', 0.0):0.2f})" for o in outliers
            )
            parts.append(f"Outliers: {detail}.")
        else:
            parts.append("No discrepant neighbors detected.")
        fake = " ".join(parts)
        words = fake.split()
        accepted = len(words) <= 200
        if not accepted:
            fake = " ".join(words[:200]) + "…"
        return (fake, accepted)

    ClientCls = _safe_import_openai()
    if ClientCls is None:  # library not installed
        return ("ERROR: openai library missing", False)
    user_content = json.dumps({"stats": stats, "constraints": {"target_max_words": 200}})
    model = _resolve_model()
    org_id = os.getenv("OPENAI_ORG_ID") or os.getenv("PATHO_OPENAI_ORG")
    try:
        raw = _call_openai(ClientCls, SYSTEM_PROMPT, user_content, model=model, organization=org_id)
    except Exception as e:  # noqa: BLE001
        # Surface error string so user sees quota / auth issues
        return (f"ERROR: {getattr(e, '__class__', type(e)).__name__}: {getattr(e, 'message', str(e))[:120]}", False)

    text: Optional[str] = None
    original_word_count = 0
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("summary"), str):
            text = data["summary"].strip()
    except json.JSONDecodeError:
        text = raw.strip()

    if not text:
        return (None, False)
    words = text.split()
    original_word_count = len(words)
    if original_word_count > 400:  # hard cap for safety
        return (f"ERROR: output too long ({original_word_count} words)", False)
    accepted = original_word_count <= 200
    if not accepted:
        # Truncate display to 200 words for rendering
        text = " ".join(words[:200]) + "…"
    text = _postprocess_probability_summary(stats, text)

    # -- Provenance guard: reject if LLM fabricated numbers ----------------
    ref = _reference_from_stats(stats)
    _, violations = provenance_guard(text, ref)
    if violations:
        # Too risky — fall back so deterministic path is used
        return (None, False)

    return (text, accepted)


def _derive_stats_from_payload(
    payload: Dict[str, Any],
) -> tuple[Dict[str, Any], str, str, float, float, float, Optional[Dict], List[Dict], Optional[str]]:
    """Derive lightweight statistics from a full neighbours payload.

    Returns ``(derived_stats_dict, score_name, direction, mean, min, max,
    top_case, discrepants, mode_dx)``.
    """
    from collections import Counter as _C

    neighbors = payload.get("neighbors", [])
    diagnoses = [n.get("diagnosis") for n in neighbors if n.get("diagnosis")]
    counts = _C(diagnoses)
    mode_dx: Optional[str] = None
    if counts:
        maxc = max(counts.values())
        mode_dx = sorted([d for d, c in counts.items() if c == maxc])[0]

    top_case: Optional[Dict[str, Any]] = None
    if any(n.get("probability") is not None for n in neighbors):
        score_name = "probability"
        direction = "higher"
        scores = [float(n.get("probability", 0.0)) for n in neighbors if n.get("probability") is not None]
        top_case = max(
            (n for n in neighbors if n.get("probability") is not None),
            key=lambda n: float(n.get("probability", 0.0)),
            default=None,
        )
    else:
        score_name = "similarity"
        direction = "lower"
        scores = [float(n.get("similarity", 0.0)) for n in neighbors if n.get("similarity") is not None]

    if scores:
        mean_score = sum(scores) / len(scores)
        min_score, max_score = min(scores), max(scores)
    else:
        mean_score = min_score = max_score = 0.0

    discrepants = [
        n for n in neighbors
        if (mode_dx and n.get("diagnosis") and n.get("diagnosis") != mode_dx)
        or str(n.get("discrepant", "")).upper() == "Y"
    ]

    derived: Dict[str, Any] = {
        "mode_diagnosis": mode_dx,
        "diagnosis_counts": dict(counts),
        "score_name": score_name,
        "score_direction": direction,
        "mean_score": mean_score,
        "min_score": min_score,
        "max_score": max_score,
        "top_score": float(top_case.get("probability", 0.0)) if (score_name == "probability" and top_case) else None,
        "top_case_id": top_case.get("case_id") if top_case else None,
        "top_tumor_type": (top_case.get("tumor_type") or top_case.get("tumor_group")) if top_case else None,
        "top_diagnosis": top_case.get("diagnosis") if top_case else None,
        "discrepant_count": len(discrepants),
    }
    return derived, score_name, direction, mean_score, min_score, max_score, top_case, discrepants, mode_dx


def _build_fake_grounded(
    payload: Dict[str, Any],
    mode_dx: Optional[str],
    counts: Dict[str, int],
    score_name: str,
    direction: str,
    mean_score: float,
    min_score: float,
    max_score: float,
    top_case: Optional[Dict[str, Any]],
    discrepants: List[Dict[str, Any]],
) -> tuple[str, bool]:
    """Produce a fake grounded summary with inline citations (no network)."""
    neighbors = payload.get("neighbors", [])
    current = payload.get("current_case", {})
    better_phrase = "higher is better" if direction == "higher" else "lower is better"

    # Citation style v2: the diagnosis partition cites each group exactly
    # once, set-level statistics are uncited, single-neighbour values cite
    # one id, the query is never cited (see grounding.verify_citation_style).
    ids_by_dx: Dict[str, List[str]] = {}
    for n in neighbors:
        dx = n.get("diagnosis")
        cid = str(n.get("case_id", "?"))
        if dx:
            ids_by_dx.setdefault(dx, []).append(cid)          # rank order kept
    ordered = sorted(ids_by_dx.items(), key=lambda kv: (-len(kv[1]), neighbors.index(
        next(n for n in neighbors if str(n.get("case_id", "?")) == kv[1][0]))))
    k = len(neighbors)

    cur_bits = [f"{key}={current.get(key)}" for key in
                ("age", "sex", "site", "tumor_group", "diagnosis", "met", "treatment", "discrepant")
                if current.get(key) not in (None, "")]
    parts = ["Current case: " + ", ".join(cur_bits) + "." if cur_bits
             else "Current case: no metadata provided."]
    partition = "; ".join(
        f"{len(ids)} {'are' if len(ids) > 1 else 'is'} {dx} [{', '.join(ids)}]"
        for dx, ids in ordered)
    parts.append(f"Of the {k} neighbors, {partition}." if partition
                 else f"The {k} neighbors carry no diagnosis labels.")
    parts.append(
        f"{score_name.title()} mean {mean_score:.3f}, range "
        f"{min_score:.3f}-{max_score:.3f} ({better_phrase})."
    )
    if score_name == "probability" and top_case is not None:
        tumor_label = (
            top_case.get("tumor_type")
            or top_case.get("tumor_group")
            or top_case.get("diagnosis")
            or "unknown"
        )
        parts.append(
            f"Highest probability {float(top_case.get('probability', 0.0)):.3f} "
            f"for {tumor_label} [{top_case.get('case_id')}]."
        )
    if discrepants:
        key = "probability" if score_name == "probability" else "similarity"
        desc = "; ".join(
            f"{d.get('diagnosis')} ({score_name} {float(d.get(key, 0.0)):.3f}) "
            f"[{d.get('case_id')}]"
            for d in discrepants
        )
        parts.append(f"Discrepant neighbors: {desc}.")
    else:
        parts.append("No discrepant neighbors detected.")

    fake = " ".join(parts)
    words = fake.split()
    accepted = len(words) <= 200
    if not accepted:
        fake = " ".join(words[:200]) + "…"
    return fake, accepted


# Max number of provenance / verification warnings before we reject LLM output
_MAX_GROUNDING_WARNINGS = 3


def attempt_llm_summary_from_payload(payload: Dict[str, Any]) -> tuple[Optional[str], bool]:
    """LLM rewrite with **grounded citations** from full neighbors.json payload.

    Uses a structured prompt that asks the LLM to produce inline ``[case_id]``
    citations and a ``claims`` array.  The response is then verified:

    1. **Structured parse** — extracts ``summary`` and ``claims``.
    2. **Claim verification** — every ``source_case_id`` must exist in the
       payload and field/value must match the source record.
    3. **Provenance guard** — every number in the summary must be traceable
       to the payload or derived statistics.
    4. **Fabricated-ID check** — bracket-cited IDs in the text are validated.

    If verification produces more than ``_MAX_GROUNDING_WARNINGS`` issues the
    LLM output is rejected (returns ``None``) so the deterministic fallback
    is used instead.

    Returns ``(text, accepted_flag)`` — same specification as ``attempt_llm_summary``.
    """
    text, accepted, _tele = attempt_llm_summary_from_payload_ex(payload)
    return (text, accepted)


def _style_revision_enabled() -> bool:
    return os.getenv("PATHO_STYLE_REVISION", "1").strip().lower() not in ("0", "false", "no", "off")


def attempt_llm_summary_from_payload_ex(
    payload: Dict[str, Any],
) -> tuple[Optional[str], bool, Optional[Dict[str, Any]]]:
    """Grounded single-shot summary with telemetry.

    Same specification as :func:`attempt_llm_summary_from_payload` plus a third
    element, a telemetry dict::

        framework, model, calls, revision_applied, revision_rejected,
        grounding_warnings, style_warnings_first, style_warnings_final,
        claims (verified), word_count, fake

    Pipeline (citation style v2):

    1. one schema-guided call -> ``{summary, claims}``;
    2. grounding verification (claims, fabricated ids, provenance guard);
    3. citation-style verification (placement rules). If there are style
       issues and ``PATHO_STYLE_REVISION`` is not ``0``, ONE revision call
       returns the corrected JSON; it is adopted only if it parses and does
       not exceed the grounding-warning threshold;
    4. deterministic citation normalisation (syntax only);
    5. rejection on grounding warnings > ``_MAX_GROUNDING_WARNINGS`` exactly
       as before -- style warnings never cause rejection.
    """
    tele: Dict[str, Any] = {
        "framework": "single_shot", "model": None, "calls": 0,
        "revision_applied": False, "revision_rejected": None,
        "grounding_warnings": [], "style_warnings_first": None,
        "style_warnings_final": None, "claims": [], "word_count": 0, "fake": False,
    }
    if not _has_api_key() and not os.getenv("PATHO_LLM_FAKE"):
        return (None, False, tele)

    # -- Derive stats --------------------------------------------------------
    (derived, score_name, direction, mean_score, min_score, max_score,
     top_case, discrepants, mode_dx) = _derive_stats_from_payload(payload)
    counts = derived["diagnosis_counts"]

    # -- Fake mode (no network) ----------------------------------------------
    if os.getenv("PATHO_LLM_FAKE"):
        text, accepted = _build_fake_grounded(
            payload, mode_dx, counts, score_name, direction,
            mean_score, min_score, max_score, top_case, discrepants,
        )
        text = normalize_citations(text, payload)
        tele.update({"fake": True, "model": "fake", "calls": 0,
                     "style_warnings_final": verify_citation_style(text, payload),
                     "word_count": len(text.split())})
        return (text, accepted, tele)

    # -- Real LLM call -------------------------------------------------------
    ClientCls = _safe_import_openai()
    if ClientCls is None:
        return ("ERROR: openai library missing", False, tele)

    model = _resolve_model()
    tele["model"] = model
    org_id = os.getenv("OPENAI_ORG_ID") or os.getenv("PATHO_OPENAI_ORG")

    user_payload = {
        "payload": payload,
        "derived_stats": derived,
        "constraints": {"target_max_words": 200},
    }

    try:
        raw = _call_openai(
            ClientCls,
            GROUNDED_SYSTEM_PROMPT_NEIGHBORS,
            json.dumps(user_payload),
            model=model,
            organization=org_id,
        )
        tele["calls"] = 1
    except Exception as e:  # noqa: BLE001
        return (
            f"ERROR: {getattr(e, '__class__', type(e)).__name__}: "
            f"{getattr(e, 'message', str(e))[:120]}",
            False, tele,
        )

    # -- Parse & verify structured response -----------------------------------
    text, verified_claims, warnings, is_grounded = parse_grounded_response(
        raw, payload, derived,
    )
    if not text:
        tele["grounding_warnings"] = warnings
        return (None, False, tele)

    # -- Citation style: verify, then at most ONE revision call ---------------
    style = verify_citation_style(text, payload)
    tele["style_warnings_first"] = style
    tele["text_first"] = text
    if style and _style_revision_enabled():
        prev = loads_lenient(raw)
        if not isinstance(prev, dict):
            prev = {"summary": text}
        revision_payload = {
            **user_payload,
            "previous_response": prev,
            "style_issues": style,
            "instruction": STYLE_REVISION_INSTRUCTION,
        }
        raw2: Optional[str] = None
        try:
            raw2 = _call_openai(
                ClientCls,
                GROUNDED_SYSTEM_PROMPT_NEIGHBORS,
                json.dumps(revision_payload),
                model=model,
                organization=org_id,
            )
            tele["calls"] = 2
        except Exception as e:  # noqa: BLE001
            tele["revision_rejected"] = f"call failed: {type(e).__name__}"
        if raw2:
            text2, claims2, warnings2, grounded2 = parse_grounded_response(
                raw2, payload, derived,
            )
            if (text2 and len(warnings2) <= _MAX_GROUNDING_WARNINGS
                    and len(text2.split()) <= 400):
                text, verified_claims, warnings, is_grounded = text2, claims2, warnings2, grounded2
                tele["revision_applied"] = True
            else:
                tele["revision_rejected"] = (
                    "unparseable" if not text2 else
                    f"{len(warnings2)} grounding warnings / {len(text2.split())} words"
                )

    # Word-count limits
    words = text.split()
    if len(words) > 400:
        tele["grounding_warnings"] = warnings
        return (f"ERROR: output too long ({len(words)} words)", False, tele)
    accepted = len(words) <= 200
    if not accepted:
        text = " ".join(words[:200]) + "…"

    # Probability post-processing, then deterministic citation syntax
    text = _postprocess_probability_summary(derived, text)
    text = normalize_citations(text, payload)

    tele.update({
        "grounding_warnings": warnings,
        "style_warnings_final": verify_citation_style(text, payload),
        "claims": verified_claims,
        "word_count": len(text.split()),
    })

    # -- Reject if too many grounding issues ----------------------------------
    if len(warnings) > _MAX_GROUNDING_WARNINGS:
        return (None, False, tele)

    return (text, accepted, tele)
