"""S5c -- LLM-summary pilot, round 2 (revision): A0 only, two instruction fixes.

Specification: [internal record]; [analysis plan] record
"S5c -- LLM pilot round 2 (revision)".

This wrapper runs the round-1 harness UNCHANGED (``llm_grounding_cli.executed_fc88a2e7.py``
in this directory, a byte copy of the executed file, sha256 asserted) with three
run-time bindings and nothing else:

1. ``MANUSCRIPT_SRC`` -> ``summarizer_r2/src`` (vendored copy of the executed
   prosper-search summariser, llm.py b6527495... / grounding.py 6d2974ae..., with
   the round-2 instruction fixes; hashes pinned below). The harness's own import
   check then requires the summariser to resolve under that directory.
2. ``install_call_recorder`` is wrapped so that, after the harness has written
   every payload and before the first call is possible, the 20 payload files are
   compared byte-for-byte (sha256) with round 1's ``run/payloads/`` A0 files and
   FIDELITY.md is written; any difference aborts (exit 2) before any call.
3. For real calls the recorder's model-change stop rule is anchored on round 1's
   API-reported model (``gpt-5-2025-08-07``): a different model stops the run
   after its first call (STOPPED, exit 3).

The argument set is fixed by the specification and checked here: arms
``a0_masked_posterior`` only, no repeats, manuscript summariser, k = 5, and the
round-1 inputs (matrix, metadata, query list, fragments, weights, swap table)
by sha256. Round 1's directory is read only. ``--fake`` / PATHO_LLM_FAKE=1 makes
no network call; without it the run calls the Azure deployment, which needs a
analyst release (stage 2).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "llm_grounding_cli.executed_fc88a2e7.py"
HARNESS_SHA = "034b6dd8a467c800e87f5bce2d4e93f36a682053ecc5d2aa642b651e74fb0980"
VENDORED_SRC = HERE / "summarizer_r2" / "src"
PKG = VENDORED_SRC / "patho_toy_summarizer"

# vendored package: round-1 executed hashes (prosper-search fd50373 copy) and round-2 pins
ROUND1_PKG_SHA = {
    "__init__.py": "90ec0011eebf10b555542ef9e1726344f54c8f4428cd43e8a59c95785d4806a8",
    "cli.py": "ef07150b22437a559ff3d5e91bd073cd1f568137d893e49a9bc3f180d2527e4d",
    "logic.py": "cd9229df36d5315bfcab8893d0bd26bc492e433bbe4148866c7ebcf5e6d8162d",
    "llm.py": "efabdf797c6fd6f04b7d6c39ebd1fce5e0f096cde8b941d3cc0e119e1c14196d",
    "grounding.py": "b8326d2ea4e2d4f64bc93546c6ce703877c8beed9625e259c3345fe73822dd06",
}
ROUND2_PKG_SHA = {   # pinned at stage 1 (2026-11-10); unchanged files keep the round-1 hash
    "__init__.py": ROUND1_PKG_SHA["__init__.py"],
    "cli.py": ROUND1_PKG_SHA["cli.py"],
    "logic.py": ROUND1_PKG_SHA["logic.py"],
    "llm.py": "4bc46d00621ad45a71cd1a66e077d838bec15f90e4b578e08510d5c7570b8a0a",
    "grounding.py": "3aa505183313a60307121f1aa3bf03478e3695561e36873bbf003dd8a493e0f6",
}

PROSPER_OUT = Path(os.environ.get(
    "PROSPER_OUT",
    "/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output"))
ROUND1_RUN = PROSPER_OUT / "s5b_pilot_20261009" / "run"
ROUND2_ROOT = PROSPER_OUT / "s5c_pilot_round2_20261110"
ROUND1_MODEL = "gpt-5-2025-08-07"      # API-reported model of all 72 round-1 calls (Supplementary Note 2)
ARM = "a0_masked_posterior"
N_QUERIES = 20

# round-1 inputs (run/meta.json "inputs"), asserted by sha256
ROUND1_INPUT_SHA = {
    "matrix": "a288d8926c8715c86ae3063ff68d9f7b0c6b365eaa77549f45aa02142d2e1714",
    "metadata": "a611115d091f8729bc85899ee93e8e1d02ef3a5abfa1aba37e8d23f06cfdd0f7",
    "queries_file": "571dbb076498dfb8ea50f8f6bc7265fe3010d225a7b7893269ed8c8b33bc0741",
    "probabilities_file": "11f3ff09b1a6640eeb892f1da05e010608251154d225d7c3505f51d98d59eff9",
    "weights_file": "4b9133ea2152c5d3d77dae1264319b9e13195a4adb8453b0cab576c2cba3926c",
    "swap_table": "14d3ad2a0a49493340f46108ea8f47b7b106c3d4bf68317fc18e956a41eb8098",
}
ANALYSIS_LIB_SHA = "c3bf659246aaaa2af7d72003dd060e3d79d5ac5af27ca784679d03dec70e5346"   # specification §2 (F13)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Round2Refusal(SystemExit):
    pass


def load_harness():
    got = sha256_file(HARNESS)
    if got != HARNESS_SHA:
        raise Round2Refusal(f"harness copy sha256 {got} != executed round-1 harness {HARNESS_SHA}")
    spec = importlib.util.spec_from_file_location("llm_grounding_cli_fc88a2e7", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_vendored() -> dict:
    out = {}
    for name, pin in ROUND2_PKG_SHA.items():
        got = sha256_file(PKG / name)
        if got != pin:
            raise Round2Refusal(f"vendored {name} sha256 {got} != round-2 pin {pin}")
        out[name] = got
    return out


def check_args(H, a, argv) -> None:
    """The specification fixes every argument except --fake/--env-file/--out-dir."""
    if a.probe:
        raise Round2Refusal("--probe is not part of the round-2 procedure")
    if a.allow_additional_resume:
        raise Round2Refusal("--allow-additional-resume is refused in round 2 (a further resume needs a new decision)")
    if a.arms != ARM:
        raise Round2Refusal(f"round 2 runs arm {ARM} only (got --arms {a.arms})")
    if a.repeats_file is not None or a.repeat_original:
        raise Round2Refusal("round 2 has no repeats")
    if a.summarizer != "manuscript" or a.k != 5 or a.limit is not None:
        raise Round2Refusal("round 2 uses --summarizer manuscript, --k 5 and no --limit")
    for name, pin in ROUND1_INPUT_SHA.items():
        p = getattr(a, name)
        if p is None:
            raise Round2Refusal(f"--{name.replace('_', '-')} is required")
        got = sha256_file(p)
        if got != pin:
            raise Round2Refusal(f"--{name.replace('_', '-')} sha256 {got} != round-1 input {pin}")
    lib = H.LLM_SRC / "search" / "analysis_lib.py"
    if sha256_file(lib) != ANALYSIS_LIB_SHA:
        raise Round2Refusal(f"payload builder {lib} changed since round 1")
    out = a.out_dir.resolve()
    if ROUND2_ROOT.resolve() not in out.parents:
        raise Round2Refusal(f"--out-dir must be under {ROUND2_ROOT}")
    if (PROSPER_OUT / "s5b_pilot_20261009").resolve() in out.parents:
        raise Round2Refusal("round 1 directory is read-only")
    fake = bool(a.fake or os.environ.get("PATHO_LLM_FAKE"))
    if fake and out.name == "run":
        raise Round2Refusal("<round2>/run is reserved for the released real-call run; use dryrun_fake/")
    if not fake and out != (ROUND2_ROOT / "run").resolve():
        raise Round2Refusal(f"real calls write only to {ROUND2_ROOT / 'run'}")
    if not fake and a.env_file is None:
        raise Round2Refusal("real calls need --env-file (the round-1 file named in run/command.txt)")


def round1_payloads() -> dict:
    """{query: (sha256, path)} of round 1's A0 payload files, cross-checked against runs.jsonl."""
    files = sorted(ROUND1_RUN.glob(f"payloads/*_{ARM}_0.json"))
    ref = {p.name.split("_")[0]: (sha256_file(p), p) for p in files}
    runs = {}
    with open(ROUND1_RUN / "runs.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r["arm"] == ARM and r["repeat"] == 0:
                runs[r["query"]] = r["payload_sha256"]
    for q, (s, p) in ref.items():
        # runs.jsonl records sha256 of the serialised text (identical bytes, utf-8)
        if runs.get(q) != s:
            raise Round2Refusal(f"round-1 payload {p.name} does not match its runs.jsonl record")
    return ref


def fidelity(out: Path, queries: list) -> dict:
    ref = round1_payloads()
    rows, ok = [], True
    if sorted(ref) != sorted(queries) or len(queries) != N_QUERIES:
        ok = False
    for q in queries:
        mine = out / "payloads" / f"{q}_{ARM}_0.json"
        s2 = sha256_file(mine) if mine.exists() else None
        s1 = ref.get(q, (None, None))[0]
        same = s1 is not None and s1 == s2
        ok &= same
        rows.append({"query": q, "round1_sha256": s1, "round2_sha256": s2, "identical": same})
    extra = sorted(p.name for p in (out / "payloads").glob("*.json")
                   if p.name not in {f"{q}_{ARM}_0.json" for q in queries})
    ok &= not extra
    return {"ok": bool(ok), "rows": rows, "extra_payload_files": extra, "round1_dir": str(ROUND1_RUN / "payloads")}


def write_fidelity(out: Path, res: dict, mode: str) -> None:
    L = ["# S5c round 2 -- payload fidelity (A0, round 2 vs round 1)", "",
         f"Checked {_now()} by `scripts/llm_pilot_round2/run_round2.py` ({mode}), after the harness wrote the "
         "payloads and before any summariser call. Rule: every round-2 payload file is byte-identical (sha256) to "
         f"round 1's `{res['round1_dir']}/<query>_{ARM}_0.json`; one file per query, no other payload file.", "",
         f"**Result: {'PASS' if res['ok'] else 'FAIL'}** "
         f"({sum(r['identical'] for r in res['rows'])}/{len(res['rows'])} identical; "
         f"extra payload files: {len(res['extra_payload_files'])}).", "",
         "| query | round 1 sha256 | round 2 sha256 | identical |", "|---|---|---|---|"]
    L += [f"| {r['query']} | `{r['round1_sha256']}` | `{r['round2_sha256']}` | {'yes' if r['identical'] else 'NO'} |"
          for r in res["rows"]]
    if res["extra_payload_files"]:
        L += ["", "Extra payload files: " + ", ".join(res["extra_payload_files"])]
    (out / "FIDELITY.md").write_text("\n".join(L) + "\n")
    (out / "fidelity.json").write_text(json.dumps(res, indent=1) + "\n")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    H = load_harness()
    a = H.build_parser().parse_args(argv)
    if a.matrix is None:
        raise Round2Refusal("--matrix is required")
    check_args(H, a, argv)
    vend = check_vendored()
    fake = bool(a.fake or os.environ.get("PATHO_LLM_FAKE"))
    H.MANUSCRIPT_SRC = VENDORED_SRC                       # binding 1
    queries = H.read_ids(a.queries_file)
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    orig_install = H.install_call_recorder

    def install_after_fidelity(pll, recorder):           # binding 2 (+3)
        res = fidelity(out, queries)
        write_fidelity(out, res, "fake dry run" if fake else "real-call run")
        if not res["ok"]:
            raise H.PilotAssertionError("round-2 payloads are not byte-identical to round 1 (see FIDELITY.md)")
        mod = Path(pll.__file__).resolve()
        if mod != (PKG / "llm.py").resolve() or sha256_file(mod) != ROUND2_PKG_SHA["llm.py"]:
            raise H.PilotAssertionError(f"summariser resolved to {mod}, not the pinned round-2 copy")
        if not fake and recorder.first_model is None:
            recorder.first_model = ROUND1_MODEL
        return orig_install(pll, recorder)

    H.install_call_recorder = install_after_fidelity
    rec = {"utc": _now(), "wrapper": str(Path(__file__).resolve()), "wrapper_sha256": sha256_file(__file__),
           "harness": str(HARNESS), "harness_sha256": HARNESS_SHA, "vendored_src": str(VENDORED_SRC),
           "vendored_sha256": vend, "round1_vendored_sha256": ROUND1_PKG_SHA, "fake": fake,
           "pinned_response_model": None if fake else ROUND1_MODEL, "argv": argv,
           "python": sys.executable}
    with open(out / "round2_wrapper.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return H.main(argv)


if __name__ == "__main__":
    sys.exit(main())
