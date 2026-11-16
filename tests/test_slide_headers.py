"""Tests for S1a's WSI header extraction (``prosper.evaluation.slide_headers``).

No real SVS files are touched: ``openslide.OpenSlide`` is monkeypatched with a
fake that returns the same ``.properties``/``.dimensions``/``.level_count``
shape as a real Aperio slide, so the parsing and consistency-check logic is
exercised without depending on the read-only comet-* filesystem.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from prosper.evaluation import paths, slide_headers as SH


class _FakeSlide:
    def __init__(self, properties: dict, dimensions=(100, 200), level_count=4):
        self.properties = properties
        self.dimensions = dimensions
        self.level_count = level_count

    def close(self):
        pass


APERIO_40X_PROPS = {
    "openslide.vendor": "aperio",
    "openslide.mpp-x": "0.2516",
    "openslide.mpp-y": "0.2516",
    "openslide.objective-power": "40",
    "aperio.ScanScope ID": "D180280",
    "aperio.DSR ID": "SPWPESLIDE01",
    "aperio.ICC Profile": "AT2",
    "aperio.Date": "10/26/21",
    "aperio.Time": "04:04:29",
}

APERIO_20X_PROPS = {
    **APERIO_40X_PROPS,
    "openslide.mpp-x": "0.5032",
    "openslide.mpp-y": "0.5032",
    "openslide.objective-power": "20",
}


def _patch_open(monkeypatch, props, dims=(91631, 70411), levels=4):
    def _fake_open(path):
        return _FakeSlide(props, dimensions=dims, level_count=levels)

    class _FakeOpenslideModule:
        OpenSlide = staticmethod(_fake_open)

    monkeypatch.setattr(SH, "openslide", _FakeOpenslideModule)


# --------------------------------------------------------------- read_header --

def test_read_header_ok_40x_is_consistent(monkeypatch, tmp_path):
    _patch_open(monkeypatch, APERIO_40X_PROPS)
    wsi = tmp_path / "10002.svs"
    wsi.write_bytes(b"not a real slide, just needs to exist")

    row = SH.read_header("10002", wsi)

    assert row["status"] == "ok"
    assert row["error"] is None
    assert row["vendor"] == "aperio"
    assert row["scanner_model"] == "AT2"
    assert row["scanscope_id"] == "D180280"
    assert row["mpp_x"] == pytest.approx(0.2516)
    assert row["objective_power"] == pytest.approx(40.0)
    assert row["scan_date"] == "2021-10-26"
    assert row["width"] == 91631 and row["height"] == 70411
    assert row["mpp_consistent_with_40x"] is True
    # The raw blob round-trips and carries every property, not a curated subset.
    assert json.loads(row["raw_properties_json"]) == APERIO_40X_PROPS


def test_read_header_flags_20x_slide_as_inconsistent(monkeypatch, tmp_path):
    _patch_open(monkeypatch, APERIO_20X_PROPS)
    wsi = tmp_path / "99999.svs"
    wsi.write_bytes(b"stub")

    row = SH.read_header("99999", wsi)

    assert row["status"] == "ok"
    assert row["objective_power"] == pytest.approx(20.0)
    assert row["mpp_consistent_with_40x"] is False


def test_read_header_missing_file_is_reported_not_raised(tmp_path):
    row = SH.read_header("absent", tmp_path / "absent.svs")
    assert row["status"] == "missing"
    assert row["error"]
    assert row["mpp_consistent_with_40x"] is None


def test_read_header_open_failure_is_caught(monkeypatch, tmp_path):
    class _Boom:
        @staticmethod
        def OpenSlide(path):
            raise OSError("corrupt tiff")

    monkeypatch.setattr(SH, "openslide", _Boom)
    wsi = tmp_path / "corrupt.svs"
    wsi.write_bytes(b"junk")

    row = SH.read_header("corrupt", wsi)
    assert row["status"] == "error"
    assert "corrupt tiff" in row["error"]


# ------------------------------------------------------------------ _to_bool --

@pytest.mark.parametrize("raw,expected", [
    (True, True), (False, False),
    ("True", True), ("False", False),
    ("true", True), ("false", False),
    (None, None), (float("nan"), None), ("", None),
])
def test_to_bool(raw, expected):
    assert SH._to_bool(raw) is expected


# ------------------------------------------------------ extract_range/merge --

@pytest.fixture()
def fake_cohort(monkeypatch, tmp_path):
    """Three slides: two consistent 40x, one 20x violation, wired through the
    same path-resolution functions the CLI uses (not hardcoded)."""
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    ids = ["111", "222", "333"]
    for sid in ids:
        (slides_dir / f"{sid}.svs").write_bytes(b"stub")

    meta = pd.DataFrame({
        "Tumor.group": ["LIVER", "LIVER", "MISC"],
        "Slide.Scan.File": [int(i) for i in ids],
        "Sample.ID": ["s1", "s2", "s3"],
        "UID.Subject": ["p1", "p2", "p3"],
    })
    meta_path = tmp_path / "meta.csv"
    meta.to_csv(meta_path, index=False)

    out_root = tmp_path / "out"
    monkeypatch.setattr(paths, "slides_root", lambda: slides_dir)
    monkeypatch.setattr(paths, "slide_metadata", lambda: meta_path)
    monkeypatch.setattr(paths, "output_root", lambda: out_root)

    def _fake_open(path):
        p = str(path)
        props = APERIO_20X_PROPS if "333" in p else APERIO_40X_PROPS
        return _FakeSlide(props)

    class _FakeOpenslideModule:
        OpenSlide = staticmethod(_fake_open)

    monkeypatch.setattr(SH, "openslide", _FakeOpenslideModule)
    return out_root


def test_extract_range_writes_one_row_per_slide(fake_cohort):
    out_path = SH.extract_range(1, 3, task="t1")
    with open(out_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {r["slide_id"] for r in rows} == {"111", "222", "333"}


def test_extract_range_rejects_out_of_bounds(fake_cohort):
    with pytest.raises(ValueError):
        SH.extract_range(1, 4, task="t1")


def test_merge_reports_full_coverage_and_the_one_inconsistent_slide(fake_cohort):
    SH.extract_range(1, 2, task="t1")
    SH.extract_range(3, 3, task="t1")

    summary = SH.merge("t1")

    assert summary["n_rows"] == 3
    assert summary["n_missing_from_output"] == 0
    assert summary["missing_slide_ids"] == []
    assert summary["n_mpp_inconsistent"] == 1
    assert summary["inconsistent_slide_ids"] == ["333"]
    assert summary["n_mpp_consistent"] == 2
    assert summary["status_counts"] == {"ok": 3}

    merged = pd.read_csv(fake_cohort / "t1" / "slide_headers.csv")
    assert len(merged) == 3


def test_merge_fails_loudly_with_no_chunks(fake_cohort):
    with pytest.raises(SystemExit):
        SH.merge("nonexistent_task")
