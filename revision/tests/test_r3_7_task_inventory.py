"""Checks for scripts/r3_7_task_inventory.py (read-only on the frozen Stream D outputs)."""
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "r37inv", os.path.join(HERE, "..", "scripts", "r3_7_task_inventory.py"))
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)

FROZEN = inv.DEFAULT_IN
pytestmark = pytest.mark.skipif(not os.path.isdir(FROZEN), reason="frozen Stream D outputs absent")
BINARY = {a for a in inv.ARM_META if a not in ("s61_int_3c", "s61_ext_3c", "s65_ext_3c")}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def read_rows(out):
    with open(os.path.join(out, "task_inventory.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_md(out):
    lines = open(os.path.join(out, "task_inventory.md"), encoding="utf-8").read().splitlines()
    tab = [ln for ln in lines if ln.startswith("|")]
    split = lambda ln: [c.strip() for c in ln.strip().strip("|").split(" | ")]
    header = split(tab[0])
    assert re.fullmatch(r"\|(---\|)+", tab[1])
    return header, [dict(zip(header, split(ln))) for ln in tab[2:]]


def copy_frozen(tmp_path, name="in"):
    dst = tmp_path / name
    shutil.copytree(FROZEN, dst)
    return str(dst)


def edit_csv(path, match, col, value):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    hits = 0
    for r in rows[1:]:
        if all(r[hdr.index(k)] == v for k, v in match.items()):
            r[hdr.index(col)] = value
            hits += 1
    assert hits == 1
    with open(path, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def repin(monkeypatch, src):
    """Re-pin to the tampered copy so the deeper consistency checks (not the hash pin) are exercised."""
    monkeypatch.setattr(inv, "PINNED_SHA256", {f: sha(os.path.join(src, f)) for f in inv.PINNED_SHA256})


def assert_refused(capsys, src, out, reason):
    assert inv.main(["--in-dir", src, "--out-dir", out]) == 1
    assert not os.path.exists(out)
    err = capsys.readouterr().err
    assert "REFUSED (no output written)" in err and reason in err, err


@pytest.fixture(scope="module")
def real_out(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("real") / "out")
    before = {f: sha(os.path.join(FROZEN, f)) for f in os.listdir(FROZEN)}
    assert inv.main(["--out-dir", out]) == 0
    after = {f: sha(os.path.join(FROZEN, f)) for f in os.listdir(FROZEN)}
    assert before == after, "frozen inputs changed"
    return out


def by_arm(out):
    return {r["arm"]: r for r in read_rows(out)}


def test_all_arms_and_spot_values(real_out):
    rows = by_arm(real_out)
    assert set(rows) == set(inv.ARM_META) and len(rows) == 13
    assert rows["s61_int_2c"]["plurality5_accuracy"].startswith("331/373 ")
    assert rows["s61_ext_2c"]["plurality5_accuracy"].startswith("221/251 ")
    assert rows["s64"]["plurality5_accuracy"].startswith("130/139 ")
    assert rows["s62"]["minority_class_recall"].startswith("amplified 23/66 ")
    assert rows["s65b_gnb"]["minority_class_recall"].startswith("GNB 14/38 ")
    complete = json.load(open(os.path.join(real_out, "COMPLETE")))
    for name, h in complete["outputs_sha256"].items():
        assert sha(os.path.join(real_out, name)) == h
    rec = json.load(open(os.path.join(real_out, "record.json")))
    assert rec["frozen_d_script_sha256"] == inv.FROZEN_D_SCRIPT_SHA256
    assert rec["new_bootstrap_computed"] is False
    assert rec["input_sha256"] == inv.PINNED_SHA256 and len(rec["input_sha256"]) == 4 + 3 * 13
    assert rec["interpreter"] == sys.executable and rec["command"].startswith(sys.executable + " ")


def test_md_csv_consistency(real_out):
    header, md_rows = parse_md(real_out)
    assert header == inv.DISPLAY_COLUMNS
    csv_rows = read_rows(real_out)
    assert [r["arm"] for r in md_rows] == [r["arm"] for r in csv_rows]
    for m, c in zip(md_rows, csv_rows):
        for k in inv.DISPLAY_COLUMNS:
            assert m[k] == c[k], (c["arm"], k)
    md = open(os.path.join(real_out, "task_inventory.md"), encoding="utf-8").read()
    for term in ("strict@5 is identical to plurality@5", "*SW1*", "_N`-suffixed", "Conditional retrieval",
                 "*Majority-class rate*", "*Mean class recall*", "same-patient slides cannot"):
        assert term in md, term


def test_numeric_columns_match_display(real_out):
    for r in read_rows(real_out):
        n = int(r["n_queries"])
        assert r["plurality5_accuracy"].startswith(f"{r['num_plurality5_correct']}/{n} ")
        if r["arm"] in BINARY:
            assert r["strict5_accuracy"] == r["strict5_accuracy_ci"] == inv.BINARY_STRICT
            assert r["num_strict5_correct"] == r["num_plurality5_correct"] and r["num_strict5_abstain"] == "0"
        else:
            assert r["strict5_accuracy"].startswith(f"{r['num_strict5_correct']}/{n} ")
            assert r["strict5_accuracy"].endswith(f"abstained {r['num_strict5_abstain']}")
        assert f" {r['num_majority_n']}/{n} " in r["majority_class_rate"]
        assert r["mean_class_recall"] == f"{float(r['num_mean_class_recall']):.3f}"


def test_majority_and_balanced_columns(real_out):
    rows = by_arm(real_out)
    g = rows["s65b_gnb"]
    assert g["majority_class_rate"] == "NB 316/354 (0.893)"
    assert g["mean_class_recall"] == "0.656" and g["mean_class_recall_ci"].startswith("0.580–0.727 ")
    assert "does not exceed the majority-class rate (316/354)" in g["caveats"]
    assert rows["s62"]["majority_class_rate"] == "not_amplified 244/310 (0.787)"
    assert rows["s62"]["mean_class_recall_ci"].startswith("0.586–0.715 ")
    assert sum("majority-class rate" in r["caveats"] for r in rows.values()) == 1


def test_main_text_column(real_out):
    rows = by_arm(real_out)
    main = {a for a, r in rows.items() if r["main_text"] == "main text"}
    assert main == {"s61_int_2c", "s61_ext_2c", "s64"}
    for a in ("s61_int_3c", "s61_ext_3c"):
        assert "Spindle" in rows[a]["main_text"]
    assert all(r["main_text"] == inv.SUPPLEMENT_ONLY for a, r in rows.items()
               if a not in main | {"s61_int_3c", "s61_ext_3c"})


def test_denominator_named(real_out):
    pat = re.compile(r"^(\d+)/(\d+) slides evaluated \(denominator: (.+?)\); "
                     r"(\d+)/(\d+) carry a fine label in the family manifest$")
    for r in read_rows(real_out):
        name, count = inv.EXPECTED_BROAD[r["arm"]]
        assert r["num_coverage_denominator"] == str(count)
        if r["arm"] == "s61_fusion":
            assert r["label_coverage"] == (
                "190/471 slides evaluated (denominator: COMET RMS (Tumor.group)); 190 non-xenograft slides with "
                "saved S6 fusion status; status not recorded for 201 others; xenografts not queried (47); "
                "33 without an RMS subtype in the family manifest")
            continue
        m = pat.match(r["label_coverage"])
        assert m, r["label_coverage"]
        assert m.group(1) == r["n_queries"] and m.group(2) == m.group(5) == str(count)
        assert m.group(3) == name and m.group(4) == r["num_fine_labelled_slides"]
    ext = by_arm(real_out)["s61_ext_2c"]
    assert ext["label_coverage"].startswith("251/439 slides evaluated (denominator: CCDI RMS after SW1 exclusion)")
    assert ("of 439 CCDI RMS slides (after SW1 exclusion), 271 carry a CCDI histologic subtype label and 251 "
            "are ERMS/ARMS; not all-RMS performance") in ext["caveats"]
    assert "no patient overlap between CCDI queries and the COMET reference (checked)" in ext["exclusions"]


def test_label_sources(real_out):
    rows = by_arm(real_out)
    assert rows["s61_fusion"]["label_source"].startswith("saved S6 s6_1_rms/per_query_s61_fusion.csv true_label")
    for arm, (mode, base, field) in inv.LABEL_FIELD.items():
        assert field in rows[arm]["label_source"]
        if mode in ("manifest", "ccdi"):
            assert rows[arm]["label_source"] == f"{field} (label_source of {base})"
    rec = json.load(open(os.path.join(real_out, "record.json")))
    assert rec["label_source_files"]["s61_ext_2c"]["sha256"] == inv.CCDI_MANIFEST_SHA256
    assert "not listed" in rec["label_source_files"]["s61_ext_2c"]["binding"]


def test_kidrare_bias_caveat(real_out):
    for arm in ("s65_ext_3c", "s65_ext_binary"):
        r = by_arm(real_out)[arm]
        assert "point estimates may be optimistic" in r["caveats"]
        assert "not comparable with the patient-excluded" in r["caveats"]
        assert "query slide itself is excluded" in r["caveats"]
        assert r["n_patients"].startswith("unknown")


def test_no_invented_ci_for_kidrare(real_out):
    num_ci = re.compile(r"\d\.\d+–\d\.\d+")
    keys = ("plurality5_accuracy_ci", "strict5_accuracy_ci", "minority_class_recall_ci", "mean_class_recall_ci")
    for r in read_rows(real_out):
        cells = [r[k] for k in keys if r[k] != inv.BINARY_STRICT]
        if r["arm"].startswith("s65_ext"):
            assert r["unit"] == "slide"
            assert all(not num_ci.search(c) and "whole-patient 95% CI" not in c for c in cells)
            assert r["uncertainty"] == inv.NOT_PATIENT_CI
        else:
            assert all("(whole-patient 95% CI)" in c for c in cells), r["arm"]


def test_absent_ci_is_not_invented(tmp_path, monkeypatch):
    src = copy_frozen(tmp_path)
    s = os.path.join(src, "summary_all_arms.csv")
    for col, v in (("plurality5_recall_lo", ""), ("plurality5_recall_hi", ""), ("ci_available", "False")):
        edit_csv(s, {"arm": "s64", "class": "ALL"}, col, v)
    repin(monkeypatch, src)
    out = str(tmp_path / "out")
    assert inv.main(["--in-dir", src, "--out-dir", out]) == 0
    r = by_arm(out)["s64"]
    assert r["plurality5_accuracy_ci"] == inv.NO_CI and r["uncertainty"] == inv.NO_CI
    assert "whole-patient 95% CI" in r["minority_class_recall_ci"]


def test_hash_pin_refusal(tmp_path, capsys):
    src = copy_frozen(tmp_path)
    with open(os.path.join(src, "README.md"), "a") as f:
        f.write("\nKidRare intervals are patient-level.\n")
    assert_refused(capsys, src, str(tmp_path / "out"), "pinned frozen input README.md")


SUMMARY = "summary_all_arms.csv"
COVERAGE = "coverage.csv"


@pytest.mark.parametrize("target,match,col,value,reason", [
    # earlier cross-source mismatches
    (SUMMARY, {"arm": "s64", "class": "HCC"}, "plurality5_recall", "0.8409", "plurality5 recall"),
    (SUMMARY, {"arm": "s61_ext_2c", "class": "ALL"}, "n_queries", "252", "n_queries"),
    (SUMMARY, {"arm": "s62", "class": "ALL"}, "n_patients", "230", "n_patients"),
    ("confusion_k5_plurality_s62.csv", {"truth": "amplified"}, "amplified", "24", "cell truth=amplified"),
    ("confusion_k5_strict_s65_ext_3c.csv", {"truth": "differentiating"}, "ABSTAIN", "7", "ABSTAIN"),
    (COVERAGE, {"arm": "s61_ext_2c"}, "n_evaluated_queries", "250", "n_evaluated_queries"),
    (COVERAGE, {"arm": "s64"}, "dropped_xenograft", "3.0", "dropped_*"),
    (COVERAGE, {"arm": "s61_int_2c"}, "n_reference_slides", "372", "n_reference_slides"),
    # CI tamper (impossible intervals; degenerate flag)
    (SUMMARY, {"arm": "s64", "class": "ALL"}, "plurality5_recall_lo", "0.99", "impossible for estimate"),
    (SUMMARY, {"arm": "s64", "class": "ALL"}, "plurality5_recall_hi", "0.90", "impossible for estimate"),
    (SUMMARY, {"arm": "s64", "class": "MACRO"}, "plurality5_recall_lo", "0.97", "impossible for estimate"),
    (SUMMARY, {"arm": "s62", "class": "amplified"}, "plurality5_recall_hi", "0.30", "impossible for estimate"),
    (SUMMARY, {"arm": "s61_int_3c", "class": "Spindle"}, "ci_degenerate_metrics", "", "ci_degenerate_metrics"),
    (SUMMARY, {"arm": "s64", "class": "ALL"}, "plurality5_recall_hi", "", "ci_available is True"),
    # coverage tamper
    (COVERAGE, {"arm": "s61_ext_2c"}, "broad_group_slides", "440", "broad_group_slides"),
    (COVERAGE, {"arm": "s61_int_2c"}, "broad_group_slides", "500", "broad_group_slides"),
    (COVERAGE, {"arm": "s61_int_2c"}, "no_subtype", "30", "broad = no_subtype + fine_labelled"),
    (COVERAGE, {"arm": "s61_int_2c"}, "fine_labelled_slides", "400", "broad = no_subtype + fine_labelled"),
    (COVERAGE, {"arm": "s61_fusion"}, "no_fusion_status", "200", "no_fusion_status + fine_labelled"),
    (COVERAGE, {"arm": "s61_ext_2c"}, "broad_group", "CCDI RMS (all)", "broad_group name"),
    (COVERAGE, {"arm": "s64"}, "broad_group", "COMET LIVER", "broad_group name"),
    # explicit boolean flags
    (SUMMARY, {"arm": "s64", "class": "HCC"}, "ci_patient_level", "", "ci_patient_level is ''"),
    (SUMMARY, {"arm": "s64", "class": "HCC"}, "ci_available", "", "ci_available is ''"),
    (SUMMARY, {"arm": "s64", "class": "ALL"}, "ci_available", "true", "ci_available is 'true'"),
    (SUMMARY, {"arm": "s64", "class": "HCC"}, "ci_patient_level", "False", "consistent with unit"),
    (SUMMARY, {"arm": "s65_ext_binary", "class": "ALL"}, "ci_patient_level", "True", "ci_patient_level"),
])
def test_tamper_refused_by_pin_and_by_content(tmp_path, capsys, monkeypatch, target, match, col, value, reason):
    src = copy_frozen(tmp_path)
    edit_csv(os.path.join(src, target), match, col, value)
    assert_refused(capsys, src, str(tmp_path / "out_pin"), f"pinned frozen input {target}")
    repin(monkeypatch, src)
    assert_refused(capsys, src, str(tmp_path / "out_content"), reason)


def test_compensating_coverage_tamper_refused(tmp_path, capsys, monkeypatch):
    src = copy_frozen(tmp_path)
    c = os.path.join(src, COVERAGE)
    edit_csv(c, {"arm": "s61_int_2c"}, "dropped_xenograft", "46.0")
    edit_csv(c, {"arm": "s61_int_2c"}, "dropped_label_filter", "19")
    repin(monkeypatch, src)
    assert_refused(capsys, src, str(tmp_path / "out"), "xenograft_excluded equals dropped_xenograft")


def test_per_query_flip_refused(tmp_path, capsys, monkeypatch):
    src = copy_frozen(tmp_path)
    path = os.path.join(src, "per_query_s65b_gnb.csv")
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    h = rows[0]
    rows[1][h.index("plurality5")] = "GNB" if rows[1][h.index("plurality5")] == "NB" else "NB"
    with open(path, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    repin(monkeypatch, src)
    assert_refused(capsys, src, str(tmp_path / "out"), "per-row plurality5_recall")


def test_reconciliation_flag_must_be_json_bool(tmp_path, capsys, monkeypatch):
    src = copy_frozen(tmp_path)
    rp = os.path.join(src, "reconciliation.json")
    d = json.load(open(rp))
    d["reconciliation"]["s64"]["ci_patient_level"] = "True"
    json.dump(d, open(rp, "w"))
    repin(monkeypatch, src)
    assert_refused(capsys, src, str(tmp_path / "out"), "not a JSON boolean")


def test_label_source_verification_refuses(tmp_path, capsys, monkeypatch):
    fields = dict(inv.LABEL_FIELD)
    fields["s64"] = ("manifest", "manifest_s64_hbhcc_comet.csv", "ALIdf3Ali.Subtype")
    monkeypatch.setattr(inv, "LABEL_FIELD", fields)
    assert_refused(capsys, FROZEN, str(tmp_path / "out1"), "label_source column")
    monkeypatch.setattr(inv, "LABEL_FIELD", dict(inv.LABEL_FIELD, s64=("manifest", "manifest_s64_hbhcc_comet.csv",
                                                                       "rich.Diagnosis")))
    monkeypatch.setattr(inv, "CCDI_MANIFEST_SHA256", "0" * 64)
    assert_refused(capsys, FROZEN, str(tmp_path / "out2"), "pinned CCDI manifest hash")


def test_unknown_arm_and_frozen_script_hash_refused(tmp_path, capsys, monkeypatch):
    src = copy_frozen(tmp_path)
    rp = os.path.join(src, "reconciliation.json")
    d = json.load(open(rp))
    d["script_sha256"] = "0" * 64
    json.dump(d, open(rp, "w"))
    repin(monkeypatch, src)
    out = str(tmp_path / "out")
    assert_refused(capsys, src, out, "script_sha256")
    src2 = copy_frozen(tmp_path, "in2")
    with open(os.path.join(src2, COVERAGE), "a") as f:
        f.write("s99_new" + "," * 21 + "\n")
    repin(monkeypatch, src2)
    assert_refused(capsys, src2, out, "arm set")


def test_refuses_nonempty_out_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale").write_text("x")
    assert inv.main(["--out-dir", str(out)]) == 1
    assert sorted(os.listdir(out)) == ["stale"]


def test_reconciliation_binds_every_display_number(real_out):
    rec = json.load(open(os.path.join(real_out, "reconciliation.json")))
    assert rec["n_checks"] > 0
    for r in read_rows(real_out):
        b = rec["arms"][r["arm"]]["bindings"]
        assert b["plurality5_correct"]["value"] == int(r["num_plurality5_correct"])
        assert b["strict5_correct"]["value"] == int(r["num_strict5_correct"])
        assert b["majority_class"]["n"] == int(r["num_majority_n"])
        assert repr(b["mean_class_recall"]["value"]) == r["num_mean_class_recall"]
        assert b["label_coverage"]["denominator"] == int(r["num_coverage_denominator"])
        assert b["n_reference_slides"]["value"] == int(r["n_reference_slides"])
        assert b["ci"]["mean_class_recall"]["text"] == r["mean_class_recall_ci"]
        for cls, v in b["per_class_plurality5"].items():
            assert f"{cls} {v['correct']}/{v['n']} " in r["per_class_plurality5_recall"]
            assert v["source"].startswith(f"confusion_k5_plurality_{r['arm']}.csv")
        for c in rec["arms"][r["arm"]]["checks"]:
            if "abs_diff" in c:
                assert c["abs_diff"] <= inv.TOL
            elif "recomputed" in c:
                assert c["recomputed"] == c["frozen"]
            else:
                assert c["lo"] <= c["estimate"] + inv.TOL and c["estimate"] <= c["hi"] + inv.TOL
