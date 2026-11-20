"""revision (unchanged copy of si_entries_043.py). revision: parse the table entries of the relabelled Supplementary Information (FIXLIST_043 B2).

Same output as assemble_final_tables_20261110.si_entries / caption_rows (revision use), adapted to
  heading     "## Supplementary Data 1–6: titles, legends and previews" with "### Supplementary Data k. <title>" sub-headings
  captions    "Supplementary Data k, S# | Title. Legend"
  shared      "Notes to Supplementary Data k, Sa and Sb." (two or more tables; single-table notes stay ordinary paragraphs,
              as in the revision parser, where only the plural "Supplementary Tables" form was shared)
caption_rows drops the caption_source/source_workbook/assembled rows (as in revision) and labels the table
"Supplementary Data k, S#".
"""
import re
import sys
from pathlib import Path

REPO = Path("/PROSPER_WORKSPACE/prosper_revision")
sys.path.insert(0, str(REPO / "scripts/supp_tables"))
import packaging_common_20261109 as pc  # noqa: E402

HEADING = "## Supplementary Data 1–6: titles, legends and previews"
HEAD_RE = re.compile(r"^Supplementary Data (\d), (S\d+) \| (.+)$")
SHARED_RE = re.compile(r"^(?:Notes|Footnotes) to Supplementary Data \d, ((?:S\d+(?:, | and )?)+)\.?$")
NOTES_HEADING_RE = re.compile(r"^(?:Notes|Footnotes?) to Supplementary Data \d, ")


def si_entries(si_md: Path = None) -> dict:
    lines = (si_md or pc.SI_MD).read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == HEADING)
    ent: dict[str, dict] = {}
    cur = None
    shared = None
    for i in range(start + 1, len(lines)):
        l = lines[i].rstrip()
        if not l or l.startswith("### "):
            continue
        if l.startswith("## "):
            break
        m = HEAD_RE.match(l)
        if m:
            if shared:
                for t in shared[0]:
                    ent.setdefault(t, {"line": None, "title": None, "caption": None, "paragraphs": []})
                    ent[t]["paragraphs"].extend(shared[1])
                shared = None
            k, num, rest = m.group(1), m.group(2), m.group(3)
            title, _, cap = rest.partition(". ")
            e = ent.setdefault(num, {"line": None, "title": None, "caption": None, "paragraphs": []})
            e.update(line=i + 1, file=int(k), title=title.rstrip("."), caption=cap.strip())
            cur = num
            continue
        ms = SHARED_RE.match(l)
        if ms and len(re.findall(r"S\d+", ms.group(1))) > 1:
            shared = (re.findall(r"S\d+", ms.group(1)), [l])
            continue
        if shared is not None:
            shared[1].append(l)
        elif cur:
            if l.startswith("Workbook: Supplementary Data "):
                continue
            ent[cur]["paragraphs"].append(l)
    if shared:
        for t in shared[0]:
            ent.setdefault(t, {"line": None, "title": None, "caption": None, "paragraphs": []})
            ent[t]["paragraphs"].extend(shared[1])
    return ent


def caption_rows(final: str, e: dict) -> list[tuple[str, str]]:
    rows = [("table", f"Supplementary Data {e['file']}, {final}"), ("title", e["title"]), ("caption", e["caption"])]
    k = 0
    for p in e["paragraphs"]:
        if NOTES_HEADING_RE.match(p) and p.rstrip(".").endswith(tuple(re.findall(r"S\d+", p))):
            rows.append(("notes_heading", p))
            continue
        if p.startswith("Footnote"):
            rows.append(("footnotes", p))
            continue
        k += 1
        rows.append((f"note_{k}", p))
    return rows
