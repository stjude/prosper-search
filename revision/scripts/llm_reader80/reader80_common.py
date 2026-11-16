"""S5d (reader-80 grounding rating) shared constants and helpers. KEY-SIDE MODULE: not for the raters.

Plan: [internal record] as amended by
..._candidate_002_20261115.md. Everything here is read-only on its inputs; writers live in the stage scripts.

The sentence splitter is copied verbatim from scripts/llm_pilot_round2/build_rating_packets_r2.py (itself the
round-1 rule); the round-1 derived-statistics function is executed from the frozen module copy
scripts/llm_pilot_round2/summarizer_r2/round1_reference/llm.py.round1 (b6527495...) by extracting its source
with `ast` (the module itself has a package-relative import and cannot be imported standalone).
"""
from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROSPER_OUT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
OUT_ROOT = PROSPER_OUT / "s5d_reader80_rating_20261115"
RSS = PROSPER_OUT / "reader_study_suite"
EXPORT = RSS / "exports" / "prod_export_20261114-095616"
EXPORT_1108 = RSS / "exports" / "prod_export_20261108-114240"
SERVED_DIR = RSS / "data" / "llm"
GEN_RUNS = {"CCDI": PROSPER_OUT / "st1_llm_summaries_v5_ccdi", "STV": PROSPER_OUT / "st1_llm_summaries_v5_stv"}
ROUND1 = PROSPER_OUT / "s5b_pilot_20261009"
ROUND2 = PROSPER_OUT / "s5c_pilot_round2_20261110"
R2_SCRIPTS = REPO / "scripts" / "llm_pilot_round2"
HARNESS = R2_SCRIPTS / "llm_grounding_cli.executed_fc88a2e7.py"
ROUND1_LLM = R2_SCRIPTS / "summarizer_r2" / "round1_reference" / "llm.py.round1"
ROUND1_LLM_SHA = "efabdf797c6fd6f04b7d6c39ebd1fce5e0f096cde8b941d3cc0e119e1c14196d"
ROUND1_GROUNDING = R2_SCRIPTS / "summarizer_r2" / "round1_reference" / "grounding.py.round1"
ROUND1_GROUNDING_SHA = "b8326d2ea4e2d4f64bc93546c6ce703877c8beed9625e259c3345fe73822dd06"
ROUND1_CONTRACT = REPO / "docs" / "analysis" / "s5b_pilot_contract_20261009.md"
ROUND1_PARAGRAPH_SHA = "2a27ada7deedd58b54b4a9a31491df377d047c50388a1f1b816f612cfae031ff"
PRACTICE_CASES = ("case01", "case02")
N_ITEMS = 80
COHORT_NAME = {"CCDI": "CCDI-MCI", "STV": "STV"}
PACKET_LETTERS = tuple("ABCDEFGH")
ITEMS_PER_PACKET = 10
PER_COHORT_PER_PACKET = 5


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fsha(p) -> str:
    return sha(Path(p).read_bytes())


def payload_sha12(payload: dict) -> str:
    """The generating CLI's `payload_sha` (first 12 hex of sha256 of the sort_keys JSON dump)."""
    return sha(json.dumps(payload, sort_keys=True).encode())[:12]


def load_harness(name="s5d_harness_readonly"):
    ld = importlib.machinery.SourceFileLoader(name, str(HARNESS))
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(ld.name, ld))
    ld.exec_module(m)
    return m


# ---------------------------------------------------------------- sentence splitter (round-1 rule, verbatim)
ABBR_RE = re.compile(r'(?:^|[\s(\[])(?:vs|e\.g|i\.e|etc|approx|cf|dr|no|fig|al)$', re.IGNORECASE)
BOUNDARY_RE = re.compile(r'([.!?])([)\]"\'”’]*)(\s+)(?=CASE_|[A-Z0-9])')
SPLITTER_RULE = (
    "Split after '.', '?' or '!' (plus any immediately following closing ')', ']', quote or apostrophe) "
    "when followed by whitespace and then 'CASE_', an uppercase ASCII letter or a digit; do not split when "
    "the token ending at that punctuation is one of vs / e.g / i.e / etc / approx / cf / Dr / No / Fig / al "
    "(case-insensitive). Decimals such as 0.95 are never split because no whitespace follows their point. "
    "Sentences are stripped of surrounding whitespace; empty pieces are dropped; text is otherwise unchanged."
)


def split_sentences(text):
    out, start = [], 0
    for m in BOUNDARY_RE.finditer(text):
        if ABBR_RE.search(text[start:m.start(1)]):
            continue
        end = m.end(2)
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start = m.end(3)
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    assert ' '.join(out).replace(' ', '') == text.replace(' ', '').replace('\n', '').replace('\t', ''), 'splitter lost text'
    return out


# ---------------------------------------------------------------- round-1 derived statistics (frozen code path)
def round1_derive_stats_fn():
    """Return `_derive_stats_from_payload` executed from the frozen llm.py.round1 source (function only)."""
    src_bytes = ROUND1_LLM.read_bytes()
    assert sha(src_bytes) == ROUND1_LLM_SHA, "llm.py.round1 hash changed"
    src = src_bytes.decode("utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_derive_stats_from_payload")
    fsrc = ast.get_source_segment(src, fn)
    ns = {}
    exec("from __future__ import annotations\nfrom typing import Any, Dict, List, Optional\n" + fsrc, ns)
    return ns["_derive_stats_from_payload"], sha(fsrc.encode())


def rubric_paragraph():
    clines = ROUND1_CONTRACT.read_text(encoding="utf-8").split("\n")
    i0 = next(i for i, l in enumerate(clines) if l.startswith("Manual containment audit (A0 20 + A1 20;"))
    i1 = next(i for i in range(i0, len(clines)) if clines[i].endswith("no κ and no agreement/independent-validation claim."))
    para_lines = clines[i0:i1 + 1]
    assert sha("\n".join(para_lines).encode()) == ROUND1_PARAGRAPH_SHA, "rubric paragraph differs from round 1's quote"
    return para_lines, (i0 + 1, i1 + 1)


def write_sha_list(directory: Path, name="sha256.txt"):
    files = sorted(p for p in Path(directory).rglob("*") if p.is_file() and p.name != name)
    with open(Path(directory) / name, "w") as fh:
        for f in files:
            fh.write(f"{fsha(f)}  {f.relative_to(directory)}\n")
    return len(files)
