"""Shared helpers for the 2026-11-09 supplementary-table builders
(group B_external_perturbation_finelabel; S6/S7/S8, S-R2.1-C, S28 v2).

Assembly only: every builder copies or re-tabulates accepted outputs, pins each
input by SHA-256 in input_bindings.json, writes SHA256SUMS.txt and BUILD_REPORT.md
in its own table directory, and fails (AssertionError, no partial report) on the
first failed check. Workbooks are normalised after writing (fixed zip entry
dates and document-property timestamps) so that a re-run on unchanged inputs is
byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(os.environ.get("PROSPER_OUT") or REPO / "prosper_revision_output")
PKG = OUT_ROOT / "supp_tables_20261109" / "B_external_perturbation_finelabel"
NUMBERING = REPO / "scripts/figures/supp_table_numbering.json"
EDIT_SHEET = REPO / "docs/manuscript_supplement_edits.md"
BUILD_DATE = "2026-11-09"
BUILT_BY = ("claude-opus-5.5 implementer, lead-authorized bounded packaging task (2026-11-09); "
            "assembly from accepted outputs, no new statistic")
CAPTION_STATUS = ("Draft mirror of the edit-sheet specification. The lead's Supplementary Information Markdown "
                  "is the source of truth for caption wording; numbering follows scripts/figures/supp_table_numbering.json.")
ACCESS = ("Sources include patient-level restricted COMET / CCDI-MCI derivatives. These aggregate tables carry no "
          "case identifiers, paths or hashes in their data sheets, but are not authorised for public release "
          "until the author approves the supplement.")
MINUS = "\u2212"
_FIXED_ZIP_DT = (2026, 11, 9, 0, 0, 0)
_FIXED_ISO = "2026-11-09T00:00:00Z"


def sha(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def f3(x, nd: int = 3) -> str:
    return "" if isnan(x) else f"{float(x):.{nd}f}".replace("-", MINUS)


def sgn(x, nd: int = 4) -> str:
    if isnan(x):
        return ""
    v = float(x)
    s = f"{abs(v):.{nd}f}"
    if v == 0:
        return f"{0:.{nd}f}"
    return (MINUS if v < 0 else "+") + s


def ci(p, lo, hi, nd: int = 3, signed: bool = False, unavailable: str = "not available") -> str:
    fmt = (lambda v: sgn(v, nd)) if signed else (lambda v: f3(v, nd))
    if isnan(p):
        return ""
    if isnan(lo) or isnan(hi):
        return f"{fmt(p)} [{unavailable}]"
    return f"{fmt(p)} [{fmt(lo)}, {fmt(hi)}]"


def frac(k, n, nd: int = 3) -> str:
    if isnan(k) or isnan(n) or int(n) == 0:
        return "\u2014"
    return f"{int(round(k))}/{int(n)} ({float(k) / float(n):.{nd}f})"


def edit_sheet_line(heading_prefix: str) -> tuple[int, str]:
    for i, line in enumerate(EDIT_SHEET.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith(heading_prefix):
            return i, line.strip()
    raise AssertionError(f"edit-sheet heading not found: {heading_prefix}")


def final_number(workspace_id: str) -> str:
    m = json.loads(NUMBERING.read_text(encoding="utf-8"))["map"]
    return m[workspace_id]["new"]


def normalize_xlsx(p: Path) -> None:
    with zipfile.ZipFile(p) as src:
        items = [(i.filename, src.read(i.filename)) for i in src.infolist()]
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in items:
            if name == "docProps/core.xml":
                s = data.decode("utf-8")
                for tag in ("created", "modified"):
                    s = re.sub(rf"(<dcterms:{tag}[^>]*>)[^<]*(</dcterms:{tag}>)", rf"\g<1>{_FIXED_ISO}\g<2>", s)
                data = s.encode("utf-8")
            zi = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_DT)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            z.writestr(zi, data)


class Build:
    """One table build: bindings, checks, outputs, report."""

    def __init__(self, workspace_id: str, subdir: str, script: str, spec_prefix: str, spec_note: str):
        self.wid = workspace_id
        self.final = final_number(workspace_id)
        self.dir = PKG / subdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.script = REPO / script
        self.spec_line, self.spec_heading = edit_sheet_line(spec_prefix)
        self.spec_note = spec_note
        self.inputs: dict[str, dict] = {}
        self.checks: list[tuple[str, bool, str]] = []
        self.outputs: list[Path] = []
        self.notes: list[str] = []
        self.open_items: list[str] = []
        self.bind(NUMBERING, "final supplementary-table numbering ('new' field)")
        self.bind(EDIT_SHEET, "edit sheet (spec blocks)")

    # inputs -----------------------------------------------------------------
    def bind(self, p: Path | str, role: str) -> Path:
        p = Path(p)
        assert p.exists(), f"missing input {p}"
        key = str(p)
        if key not in self.inputs:
            self.inputs[key] = {"role": role, "sha256": sha(p), "bytes": p.stat().st_size}
        return p

    def csv(self, p: Path | str, role: str, **kw) -> pd.DataFrame:
        return pd.read_csv(self.bind(p, role), **kw)

    def json(self, p: Path | str, role: str):
        return json.loads(self.bind(p, role).read_text(encoding="utf-8"))

    # checks -----------------------------------------------------------------
    def check(self, name: str, ok, detail: str = "") -> None:
        ok = bool(ok)
        self.checks.append((name, ok, detail))
        assert ok, f"CHECK FAILED: {name} :: {detail}"

    def close(self, name: str, a, b, tol: float = 1e-12) -> None:
        d = abs(float(a) - float(b))
        self.check(name, d <= tol, f"{a!r} vs {b!r} (|diff| {d:.3g}, tol {tol:g})")

    # outputs ----------------------------------------------------------------
    def caption_frame(self, title: str, caption: str, footnotes: list[str], sources: list[str], version: str) -> pd.DataFrame:
        rows = [
            ("workspace_id", self.wid),
            ("final_number", f"Supplementary Table {self.final}"),
            ("numbering_source", "scripts/figures/supp_table_numbering.json, map['%s']['new'] (sha256 %s\u2026)"
             % (self.wid, self.inputs[str(NUMBERING)]["sha256"][:16])),
            ("title", title),
            ("caption", caption),
            ("footnotes", "\n".join(f"{i}. {t}" for i, t in enumerate(footnotes, 1))),
            ("sources", "\n".join(sources)),
            ("version", version),
            ("spec_block", f"docs/manuscript_supplement_edits.md line {self.spec_line}: {self.spec_heading}"),
            ("spec_version_implemented", self.spec_note),
            ("caption_status", CAPTION_STATUS),
            ("access", ACCESS),
            ("built_by", f"{BUILT_BY}; script {self.script.relative_to(REPO)}"),
            ("build_date", BUILD_DATE),
        ]
        return pd.DataFrame(rows, columns=["field", "text"])

    def write_workbook(self, stem: str, caption: pd.DataFrame, sheets: list[tuple[str, pd.DataFrame]],
                       csv_sheets: list[str] | None = None) -> Path:
        """Write <stem>.xlsx (caption first) and one CSV per sheet named in csv_sheets (default: all)."""
        xlsx = self.dir / f"{stem}.xlsx"
        names = [n for n, _ in sheets]
        assert len(set(names)) == len(names) and all(len(n) <= 31 for n in names), names
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
            caption.to_excel(xw, sheet_name="caption", index=False)
            for name, df in sheets:
                df.to_excel(xw, sheet_name=name, index=False)
            for ws in xw.book.worksheets:
                for col in ws.columns:
                    width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:300])
                    ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 150 if ws.title == "caption" else 60)
                ws.freeze_panes = "A2"
        normalize_xlsx(xlsx)
        self.outputs.append(xlsx)
        for name, df in sheets:
            if csv_sheets is None or name in csv_sheets:
                p = self.dir / f"{stem}__{name}.csv"
                df.to_csv(p, index=False, lineterminator="\n")
                self.outputs.append(p)
        return xlsx

    def finish(self, title: str, summary: list[str], caption: pd.DataFrame, extra_sections: list[tuple[str, list[str]]] = ()) -> None:
        common = Path(__file__).resolve()
        bindings = {
            "workspace_id": self.wid,
            "final_number": self.final,
            "script": str(self.script.relative_to(REPO)),
            "script_sha256": sha(self.script),
            "common_module": str(common.relative_to(REPO)),
            "common_module_sha256": sha(common),
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "platform": platform.system(),
            "build_date": BUILD_DATE,
            "PROSPER_OUT": str(OUT_ROOT),
            "inputs": dict(sorted(self.inputs.items())),
        }
        (self.dir / "input_bindings.json").write_text(json.dumps(bindings, indent=1) + "\n", encoding="utf-8")
        npass = sum(ok for _, ok, _ in self.checks)
        lines = [f"# BUILD_REPORT \u2014 workspace {self.wid} (final Supplementary Table {self.final}): {title}", "",
                 f"Build date {BUILD_DATE}; script `{self.script.relative_to(REPO)}` (sha256 {bindings['script_sha256'][:12]}\u2026); "
                 f"common module `{common.relative_to(REPO)}` (sha256 {bindings['common_module_sha256'][:12]}\u2026); "
                 f"output `$PROSPER_OUT/{self.dir.relative_to(OUT_ROOT)}/`.", "",
                 f"**Checks: {len(self.checks)} total, {npass} PASS, {len(self.checks) - npass} FAIL.** "
                 "No statistic was computed; values are copied, rounded for display or re-tabulated (counts) from retained per-query outputs.", "",
                 "## Specification implemented", "",
                 f"- Edit sheet line {self.spec_line}: `{self.spec_heading}`",
                 f"- {self.spec_note}", "",
                 "## Summary", ""] + [f"- {s}" for s in summary]
        for head, body in extra_sections:
            lines += ["", f"## {head}", ""] + body
        lines += ["", "## Open items", ""] + ([f"- {o}" for o in self.open_items] or ["- none"])
        lines += ["", "## Sources (sha256)", ""] + [
            f"- `{k.replace(str(OUT_ROOT), '$PROSPER_OUT').replace(str(REPO) + '/', '')}` \u2014 {v['role']} \u2014 {v['sha256']}"
            for k, v in sorted(self.inputs.items())]
        lines += ["", "## Caption sheet (draft mirror)", ""] + [f"- **{r.field}**: {str(r.text).replace(chr(10), ' / ')}" for r in caption.itertuples()]
        lines += ["", "## Checks", ""] + [f"- {'PASS' if ok else 'FAIL'} {n}" + (f" \u2014 {d}" if d else "") for n, ok, d in self.checks]
        (self.dir / "BUILD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        sums = [f"{sha(p)}  {p.name}" for p in sorted(self.dir.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"]
        (self.dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
        print(f"[{self.wid} -> {self.final}] {npass}/{len(self.checks)} checks PASS; {len(self.outputs)} data outputs in {self.dir}")
