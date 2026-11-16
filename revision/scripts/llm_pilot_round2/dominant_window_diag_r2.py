"""S5c diagnostic: why the harness's dominant-statement check returns False (key-side; reads stored texts).

For every summary/source whose stored `audit` (final) or `audit_first` (first) has dominant_statement_consistent
False, this reproduces the harness `audit()` window logic (executed harness fc88a2e7, l.288-311: citation/case-id
masking, upper-casing, class aliases from neighbour tumor_group + diagnosis, the window regex, the named-class
test) on the same stored text and reports each window, the trigger word, the classes it names and the plurality.
A window is counted as a rule-13 window when its trigger is MOST followed by FREQUENT (the round-2 instruction's
"differs from the most frequent one") and it names a class outside the plurality/tied set. Reproduction is
asserted: the recomputed dom_ok must equal the stored value for every row. Writes one Markdown file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
WINDOW = r"(?:MAJORITY|MOST|DOMINANT|PREDOMINANT|CONSENSUS|SUPPORT\w*|CONSISTENT WITH|FAVOU?R\w*|PRIMAR\w*)[^.]{0,100}"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--harness", type=Path, default=HERE / "llm_grounding_cli.executed_fc88a2e7.py")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ld = importlib.machinery.SourceFileLoader("s5c_harness_ro", str(a.harness))
    H = importlib.util.module_from_spec(importlib.util.spec_from_loader(ld.name, ld))
    ld.exec_module(H)
    rows = [json.loads(l) for l in open(a.run_dir / "runs.jsonl")]
    sel = [r for r in rows if r["selected"]]
    res = []
    for r in sorted(sel, key=lambda r: int(r["query"])):
        pl = json.load(open(a.run_dir / r["payload_file"]))
        for src, text, au in (("final", r["text"], r["audit"]), ("first", r["text_first"], r["audit_first"])):
            if au is None or au.get("dominant_statement_consistent") is not False:
                continue
            nb = pl["neighbors"]
            plur, tied = H.plurality([n.get("tumor_group") for n in nb])
            masked = H._CASEID.sub(" ", H._CIT.sub(" ", text or ""))
            tl = masked.upper().replace("_", " ")
            aliases = {}
            for n in nb:
                g = n.get("tumor_group")
                if g:
                    aliases.setdefault(g, set()).add(str(g).upper().replace("_", " "))
                    if n.get("diagnosis"):
                        aliases[g].add(str(n["diagnosis"]).upper().replace("_", " "))

            def hit(g, blob):
                return any(re.search(r"(?<![A-Z])" + re.escape(al) + r"(?![A-Z])", blob) for al in aliases.get(g, ()))

            wins = []
            for m in re.finditer(WINDOW, tl):
                w = m.group(0)
                named = sorted(g for g in aliases if hit(g, w))
                trig = re.match(r"CONSISTENT WITH|\w+", w).group(0)
                wins.append({"trigger": trig, "most_frequent": bool(re.match(r"MOST\s+FREQUENT", w)),
                             "named": named, "window": w})
            named_all = [g for g in aliases if any(hit(g, w["window"]) for w in wins)]
            dom_ok = (plur in named_all or any(t in named_all for t in tied)) if named_all else None
            assert dom_ok is False, (r["run_uid"], src, dom_ok)
            ok_set = {plur, *tied}
            res.append({"run_uid": r["run_uid"], "query": r["query"], "source": src, "plurality": plur,
                        "tied": list(tied), "windows": wins,
                        "rule13_window_naming_non_plurality": any(w["most_frequent"] and set(w["named"]) - ok_set
                                                                  for w in wins),
                        "windows_naming_any_class": [w["trigger"] for w in wins if w["named"]]})
    n_final = [x for x in res if x["source"] == "final"]
    L = ["# S5c dominant-statement window diagnostic (key-side)", "",
         f"Script `{Path(__file__).name}` sha256 `{sha(__file__)}`; harness `{sha(a.harness)}`; runs.jsonl "
         f"`{sha(a.run_dir / 'runs.jsonl')}`. Recomputed dom_ok equals the stored False for all {len(res)} rows.", "",
         "Note: the harness scores dom_ok False when the classes named inside any trigger window include neither the "
         "tumor-group plurality nor a tied group. A window naming only the plurality would make it True, so False "
         "means no window named the plurality group under the harness alias rule.", "",
         f"**Summary:** {len(n_final)} final-text flags (plus {len(res) - len(n_final)} first-text rows). "
         f"Final-text flags with a MOST FREQUENT window naming a non-plurality class: "
         f"{sum(x['rule13_window_naming_non_plurality'] for x in n_final)}/{len(n_final)}.", "",
         "| run_uid | source | plurality (tied) | rule-13 window naming non-plurality | windows (trigger: classes named) |",
         "|---|---|---|---|---|"]
    for x in res:
        ws = "; ".join(f"{w['trigger']}: {','.join(w['named']) or '-'}" for w in x["windows"])
        L.append(f"| {x['run_uid']} | {x['source']} | {x['plurality']} ({'/'.join(x['tied'])}) | "
                 f"{'yes' if x['rule13_window_naming_non_plurality'] else 'no'} | {ws} |")
    L += ["", "## Window text (upper-cased, citations and case ids masked, as the harness sees it)", ""]
    for x in res:
        L.append(f"- **{x['run_uid']} / {x['source']}**")
        L += [f"  - `{w['window'].strip()}` → {w['named']}" for w in x["windows"]]
    a.out.write_text("\n".join(L) + "\n")
    print(json.dumps({"rows": len(res), "final": len(n_final),
                      "final_rule13_non_plurality": sum(x["rule13_window_naming_non_plurality"] for x in n_final),
                      "per_final": {x["query"]: [x["plurality"], x["rule13_window_naming_non_plurality"],
                                                 x["windows_naming_any_class"]] for x in n_final}}))


if __name__ == "__main__":
    main()
