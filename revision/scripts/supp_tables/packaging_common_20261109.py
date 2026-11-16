"""Shared helpers for the 2026-11-09 supplementary-table packaging pass (A_packaging).

Assembly only: the builders that import this module copy values from accepted outputs, bind every input by
SHA-256 (input_bindings.json), write SHA256SUMS.txt and a BUILD_REPORT.md per table directory. Nothing here
computes a statistic.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(os.environ.get("PROSPER_OUT") or REPO / "prosper_revision_output")
PKG = OUT_ROOT / "supp_tables_20261109" / "A_packaging"
NUMBERING = REPO / "scripts/figures/supp_table_numbering.json"
EDIT_SHEET = REPO / "docs/manuscript_supplement_edits.md"
SI_MD = REPO / "docs/prosper_revised/prosper_supplementary_legends.md"
BUILD_DATE = "2026-11-09"


def sha256(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path | str) -> str:
    """Display path: $PROSPER_OUT/... or repository-relative where possible."""
    p = Path(p).resolve()
    for base, tag in ((OUT_ROOT.resolve(), "$PROSPER_OUT"), (REPO.resolve(), "")):
        try:
            r = p.relative_to(base)
            return f"{tag}/{r}" if tag else str(r)
        except ValueError:
            pass
    return str(p)


class Bindings:
    """Records every input read: path -> sha256, size, mtime (UTC ISO)."""

    def __init__(self):
        self.d: dict[str, dict] = {}

    def bind(self, p: Path | str, role: str = "") -> Path:
        p = Path(p)
        st = p.stat()
        self.d[rel(p)] = {
            "sha256": sha256(p), "bytes": st.st_size,
            "mtime_utc": _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).isoformat(timespec="seconds"),
            "role": role,
        }
        return p

    def sha(self, p: Path | str) -> str:
        return self.d[rel(p)]["sha256"]

    def write(self, out_dir: Path, extra: dict | None = None) -> None:
        payload = {"inputs": self.d}
        if extra:
            payload.update(extra)
        (out_dir / "input_bindings.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")


def numbering(workspace_id: str) -> dict:
    """Map entry plus `current`: the latest `passNNN` number (revision note: 'revision' is the current final number,
    'revision'/'new' the number before) or `new` when no later pass exists; `history` lists the earlier numbers."""
    e = dict(json.load(open(NUMBERING))["map"][workspace_id])
    passes = {int(k[4:]): v for k, v in e.items() if k.startswith("pass") and k[4:].isdigit()}
    later = {p: v for p, v in passes.items() if p > 16}
    e["current"] = later[max(later)] if later else e["new"]
    e["current_field"] = f"pass{max(later):03d}" if later else "new"
    e["history"] = ", ".join(f"pass{p:03d} {v}" for p, v in sorted(passes.items())
                             if f"pass{p:03d}" != e["current_field"]) + f" (new {e['new']})"
    return e


def edit_sheet_lines(start: int, end: int) -> list[str]:
    """1-based inclusive line range of the [internal record] (spec text is quoted, never rewritten)."""
    lines = EDIT_SHEET.read_text(encoding="utf-8").splitlines()
    return lines[start - 1:end]


def find_line(path: Path, needle: str, start: int = 1) -> int:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if i >= start and needle in line:
            return i
    raise KeyError(f"{needle!r} not found in {path}")


def md_table_after(path: Path, needle: str, start: int = 1) -> tuple[int, list[str], list[list[str]]]:
    """First GitHub-style table after the line containing `needle`: (line of header, header, rows), cells stripped."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = find_line(path, needle, start) - 1
    while not lines[i].lstrip().startswith("|"):
        i += 1
    hdr_line = i + 1
    block = []
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        block.append(lines[i])
        i += 1
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block]
    return hdr_line, cells[0], [r for r in cells[2:]]


def write_rows_sheet(wb, name: str, header: list, rows: list[list]) -> None:
    ws = wb.create_sheet(name)
    ws.append(list(header))
    for r in rows:
        ws.append(list(r))


def si_entry(final_number: str) -> tuple[int, str, str | None]:
    """The SI Markdown entry `Supplementary Table <N> | <title>. <caption or [PENDING: caption]>` (revision layout):
    returns (line number, title, caption text or None while the independent caption is pending)."""
    head = f"Supplementary Table {final_number} | "
    for i, line in enumerate(SI_MD.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith(head):
            title, _, rest = line[len(head):].partition(". ")
            rest = rest.strip()
            return i, title.rstrip("."), (None if not rest or rest.startswith("[PENDING") else rest)
    raise KeyError(f"no SI entry {head!r}")


def caption_fields(final_number: str, spec_caption: str, sha_si: str) -> tuple[list[tuple[str, str]], int, str, str | None]:
    """Caption-sheet rows: the independent SI caption verbatim when written (source of truth), the spec draft beside it."""
    line, title, si_cap = si_entry(final_number)
    rows = [("title", title),
            ("caption", si_cap if si_cap else "[SI caption PENDING at build; spec draft shown] " + spec_caption),
            ("caption_spec_draft", spec_caption),
            ("si_markdown", f"docs/prosper_revised/prosper_supplementary_legends.md l.{line} (sha256 {sha_si[:16]}… at "
                            f"build); caption {'mirrored verbatim' if si_cap else 'pending there'}. The SI Markdown is "
                            f"the source of truth for caption wording; this sheet is a draft mirror.")]
    return rows, line, title, si_cap


def save_xlsx(wb, path: Path) -> None:
    """Byte-deterministic save: fixed document times and fixed zip-entry times (openpyxl stamps both with now)."""
    import datetime as dt
    import io
    import zipfile
    fixed = dt.datetime.fromisoformat(BUILD_DATE)
    wb.properties.created = fixed
    wb.properties.modified = fixed
    wb.properties.lastModifiedBy = wb.properties.creator = "scripts/supp_tables (A_packaging)"
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    out = io.BytesIO()
    with zipfile.ZipFile(buf) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            zi = zipfile.ZipInfo(info.filename, date_time=fixed.timetuple()[:6])
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            data = zin.read(info.filename)
            if info.filename == "docProps/core.xml":
                import re
                data = re.sub(rb'(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)',
                              rb"\g<1>" + fixed.strftime("%Y-%m-%dT%H:%M:%SZ").encode() + rb"\g<2>", data)
            zout.writestr(zi, data)
    path.write_bytes(out.getvalue())


def write_sheet_csv(ws, path: Path) -> None:
    """CSV straight from openpyxl cell values (no pandas type inference): floats by repr, booleans as True/False,
    empty cells empty."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for row in ws.iter_rows(values_only=True):
            w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in row])


def write_caption_sheet(wb, rows: list[tuple[str, str]]) -> None:
    """First sheet `caption`: field | text."""
    ws = wb.create_sheet("caption", 0)
    ws.append(["field", "text"])
    for f, t in rows:
        ws.append([f, t])
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 160


def write_sha256sums(out_dir: Path) -> None:
    lines = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            lines.append(f"{sha256(p)}  {p.relative_to(out_dir)}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class Report:
    """Collects checks (PASS/FAIL with numbers), notes and open items for BUILD_REPORT.md."""

    def __init__(self):
        self.checks: list[tuple[str, str]] = []
        self.notes: list[str] = []
        self.open_items: list[str] = []
        self.si: list[tuple[str, str]] = []

    def check(self, ok: bool, text: str) -> bool:
        self.checks.append(("PASS" if ok else "FAIL", text))
        return ok

    def check_open(self, ok: bool, text: str) -> bool:
        """A spec-conformance check whose failure lies in an accepted source value that this pass may not edit:
        reported as FAIL (open item), listed under open items, not counted as a build failure."""
        self.checks.append(("PASS" if ok else "FAIL (open item)", text))
        return ok

    def si_check(self, ok: bool, text: str) -> bool:
        """A claim of the independent SI caption tested against the workbook (findings for the analyst; not build failures)."""
        self.si.append(("CONSISTENT" if ok else "MISMATCH", text))
        return ok

    def md_si(self, si_cap: str | None, line: int) -> str:
        if not si_cap:
            return f"The SI caption (l.{line}) was pending at build; nothing to compare.\n"
        n_mis = sum(s == "MISMATCH" for s, _ in self.si)
        head = (f"SI caption at l.{line} (mirrored verbatim in the caption sheet) tested against the workbook: "
                f"{len(self.si) - n_mis} consistent, {n_mis} mismatch.\n\n")
        return head + "\n".join(f"- **{s}** {t}" for s, t in self.si) + "\n"

    def n_fail(self) -> int:
        return sum(s == "FAIL" for s, _ in self.checks)

    def summary(self) -> str:
        n_open = sum(s == "FAIL (open item)" for s, _ in self.checks)
        n_pass = sum(s == "PASS" for s, _ in self.checks)
        return f"{n_pass} PASS, {self.n_fail()} FAIL" + (f", {n_open} FAIL (open item)" if n_open else "")

    def md_checks(self) -> str:
        return "\n".join(f"- **{s}** {t}" for s, t in self.checks)
