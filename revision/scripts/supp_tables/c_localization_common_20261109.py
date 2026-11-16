"""Shared helpers for the C_localization supplementary-table builders (2026-11-09).

Assembly only: inputs are read, hashed and copied/formatted; no statistic is
recomputed. Used by build_S-R3.5-L1_20261109.py, build_S-R3.5-L2_20261109.py
and build_C_localization_index_20261109.py.

Determinism: no wall-clock time is written into any output. Workbooks are
re-zipped with fixed member timestamps and fixed docProps/core.xml dates, so
re-running a builder on unchanged inputs reproduces every output byte.
"""
import hashlib
import io
import json
import math
import os
import re
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
GROUP_DIR = OUT_ROOT / "supp_tables_20261109" / "C_localization"
BUILD_DATE = "2026-11-09"
BUILT_BY = "claude-opus-5.5 (implementer, lead-authorized packaging task 2026-11-09)"
MINUS = "\u2212"
FIXED_ZIP_TIME = (2026, 11, 9, 0, 0, 0)
FIXED_ISO = "2026-11-09T00:00:00Z"

MS = REPO / "docs/prosper_revised/prosper_manuscript_revised.md"
LG = REPO / "docs/prosper_revised/prosper_supplementary_legends.md"
ES = REPO / "docs/manuscript_supplement_edits.md"
NUMBERING = REPO / "scripts/figures/supp_table_numbering.json"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    p = Path(p)
    for root, tag in ((OUT_ROOT, "$PROSPER_OUT"), (REPO, "")):
        try:
            r = p.resolve().relative_to(root.resolve())
            return f"{tag}/{r}" if tag else str(r)
        except ValueError:
            continue
    return str(p)


class Build:
    """Collects input bindings, checks, value checks and open items for one table."""

    def __init__(self, workspace_id, out_dir, script):
        self.wid = workspace_id
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.script = Path(script)
        self.inputs = {}
        self.checks = []
        self.values = []
        self.open_items = []
        self.notes = []
        self.sheets = []
        self._lines = {}

    # ---------------------------------------------------------- inputs
    def inp(self, p, role=""):
        p = Path(p)
        assert p.exists(), f"missing input {p}"
        k = rel(p)
        if k not in self.inputs:
            self.inputs[k] = {"sha256": sha256(p), "bytes": p.stat().st_size, "role": role}
        elif role and not self.inputs[k]["role"]:
            self.inputs[k]["role"] = role
        return p

    def csv(self, p, role="", **kw):
        return pd.read_csv(self.inp(p, role), **kw)

    def json(self, p, role=""):
        with open(self.inp(p, role)) as f:
            return json.load(f)

    def text(self, p, role=""):
        return self.inp(p, role).read_text(encoding="utf-8")

    # ---------------------------------------------------------- checks
    def check(self, name, ok, detail=""):
        self.checks.append((name, bool(ok), str(detail)))
        if not ok:
            print(f"CHECK FAIL: {name} {detail}")

    def producer_hash(self, path, recorded, producer):
        actual = self.inputs[rel(self.inp(path))]["sha256"]
        self.check(f"{rel(path)} sha256 equals producer record ({producer})", actual == recorded,
                   f"{actual[:16]}… vs {recorded[:16]}…")

    def locate(self, doc, needle):
        """Return 'docname:line[,line…]' of every line containing needle, or None."""
        doc = Path(doc)
        if doc not in self._lines:
            self._lines[doc] = self.text(doc, "manuscript/spec text located for value checks").splitlines()
        hits = [str(i + 1) for i, s in enumerate(self._lines[doc]) if needle in s]
        return f"{rel(doc)}:{','.join(hits)}" if hits else None

    def value(self, quoted, printed, src, field, val, dp=None, doc=None, needle=None, exact=False):
        """Assert a manuscript-printed number against a source value at the printed rounding."""
        loc = self.locate(doc, needle) if doc is not None else None
        if doc is not None and loc is None:
            loc = f"(text not located in {rel(doc)}: {needle!r})"
        if exact:
            got = str(val)
            ok = got == str(printed)
        else:
            got = fmt(val, dp, plus=False)
            ok = norm(got) == norm(printed)
        self.values.append(dict(table=self.wid, quoted=quoted, location=loc or "", printed=str(printed),
                                source=rel(src) if src else "", field=field,
                                source_value=repr(val) if not isinstance(val, str) else val,
                                source_rounded=got, result="pass" if ok else "FAIL"))
        if not ok:
            print(f"VALUE FAIL: {quoted} printed {printed} vs {got} ({field})")
        return ok

    def open_item(self, text):
        self.open_items.append(text)

    def note(self, text):
        self.notes.append(text)

    # ---------------------------------------------------------- outputs
    def write_workbook(self, stem, caption_rows, sheets):
        """caption_rows: list of (field, text); sheets: list of (name, display_df, raw_df or None)."""
        xlsx = self.out / f"{stem}.xlsx"
        cap = pd.DataFrame(caption_rows, columns=["field", "text"])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            cap.to_excel(xw, sheet_name="caption", index=False)
            for name, disp, _raw in sheets:
                assert len(name) <= 31, name
                disp.to_excel(xw, sheet_name=name, index=False)
            for ws in xw.book.worksheets:
                for col in ws.columns:
                    width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:300])
                    cap_w = 150 if ws.title == "caption" else 60
                    ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), cap_w)
                if ws.title == "caption":
                    from openpyxl.styles import Alignment
                    for row in ws.iter_rows(min_row=2):
                        for c in row:
                            c.alignment = Alignment(wrap_text=True, vertical="top")
                ws.freeze_panes = "A2"
        xlsx.write_bytes(deterministic_xlsx(buf.getvalue()))
        cap.to_csv(self.out / f"{stem}__caption.csv", index=False)
        for name, disp, raw in sheets:
            disp.to_csv(self.out / f"{stem}__{name}.csv", index=False)
            self.sheets.append((name, f"{stem}__{name}.csv", disp.shape))
            if raw is not None:
                raw.to_csv(self.out / f"{stem}__{name}_raw.csv", index=False)
                self.sheets.append((name + "_raw", f"{stem}__{name}_raw.csv", raw.shape))
        return xlsx

    def write_bindings(self):
        d = {"workspace_id": self.wid, "builder": rel(self.script), "builder_sha256": sha256(self.script),
             "build_date": BUILD_DATE, "inputs": dict(sorted(self.inputs.items()))}
        (self.out / "input_bindings.json").write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")

    def write_value_checks(self):
        df = pd.DataFrame(self.values)
        df.to_csv(self.out / "value_checks.csv", index=False)
        return df

    def write_sums(self):
        lines = []
        for p in sorted(self.out.iterdir()):
            if p.is_file() and p.name != "SHA256SUMS.txt":
                lines.append(f"{sha256(p)}  {p.name}")
        (self.out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def deterministic_xlsx(data):
    zin = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            b = zin.read(info.filename)
            if info.filename == "docProps/core.xml":
                s = b.decode("utf-8")
                s = re.sub(r"(<dcterms:created[^>]*>)[^<]*(</dcterms:created>)", rf"\g<1>{FIXED_ISO}\g<2>", s)
                s = re.sub(r"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)", rf"\g<1>{FIXED_ISO}\g<2>", s)
                b = s.encode("utf-8")
            zi = zipfile.ZipInfo(info.filename, date_time=FIXED_ZIP_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            zout.writestr(zi, b)
    return out.getvalue()


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt(x, dp=3, plus=False):
    if isnan(x):
        return ""
    s = f"{float(x):.{dp}f}"
    if s.startswith("-"):
        s = MINUS + s[1:]
        if float(s.replace(MINUS, "")) == 0:
            s = s.replace(MINUS, "")
    elif plus and float(x) > 0:
        s = "+" + s
    return s


def ci(p, lo, hi, dp=3, plus=False):
    if isnan(p):
        return "n.e."
    if isnan(lo) or isnan(hi):
        return f"{fmt(p, dp, plus)} [n.e.]"
    return f"{fmt(p, dp, plus)} [{fmt(lo, dp, plus)}, {fmt(hi, dp, plus)}]"


def norm(s):
    return str(s).replace(MINUS, "-").replace("+", "").strip()


def numbering(build, wid):
    m = build.json(NUMBERING, "final supplementary-table numbering map")["map"][wid]
    return m["new"]


def report_header(build, title, final_number, spec_lines):
    return [f"# BUILD_REPORT — Table {build.wid} (current final number: Supplementary Table {final_number})",
            "", f"{title}", "",
            f"Builder: `{rel(build.script)}` (sha256 `{sha256(build.script)}`); built {BUILD_DATE} by {BUILT_BY}.",
            f"Output directory: `{rel(build.out)}`. Assembly/re-tabulation of accepted outputs only — no "
            "statistic re-estimated, no bootstrap re-run, no LSF job, no reader-study data.",
            "", "## Spec implemented", ""] + [f"- {s}" for s in spec_lines]


def report_tail(build):
    vc = pd.DataFrame(build.values)
    n_pass = int((vc.result == "pass").sum()) if len(vc) else 0
    lines = ["", "## Sources (sha256)", "", "| input | role | sha256 |", "|---|---|---|"]
    for k, v in sorted(build.inputs.items()):
        lines.append(f"| `{k}` | {v['role']} | `{v['sha256']}` |")
    lines += ["", f"## Manuscript value → source mapping ({n_pass}/{len(vc)} pass)", "",
              "Each printed value is compared with the source field at the printed rounding "
              "(`value_checks.csv` carries the same rows).", "",
              "| # | quoted value | where printed | printed | source | field | source value | rounded | result |",
              "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(build.values, 1):
        sv = r["source_value"]
        sv = sv if len(sv) <= 40 else sv[:37] + "…"
        lines.append(f"| {i} | {r['quoted']} | {r['location']} | {r['printed']} | `{r['source']}` | "
                     f"`{r['field']}` | {sv} | {r['source_rounded']} | {r['result']} |")
    n_ok = sum(ok for _, ok, _ in build.checks)
    lines += ["", f"## Build checks ({n_ok}/{len(build.checks)} pass)", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'} — {n}" + (f" ({d})" if d else "") for n, ok, d in build.checks]
    lines += ["", "## Outputs (sheet → CSV, rows × cols)", ""]
    lines += [f"- `{n}` → `{c}` ({s[0]} × {s[1]})" for n, c, s in build.sheets]
    lines += ["", "## Notes", ""] + [f"- {n}" for n in build.notes]
    lines += ["", "## Open items (recorded, not fabricated)", ""] + [f"{i}. {t}" for i, t in
                                                                     enumerate(build.open_items, 1)]
    return lines
