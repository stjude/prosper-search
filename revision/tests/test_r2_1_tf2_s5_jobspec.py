"""Synthetic-fixture tests for scripts/r2_1_tf2_s5_jobspec.py (R2.1-TF2 S5, load-time masking).

Reuses the S3/S4 fixture world. Covers: pilot seed reproducibility, element specs (Option C library
per query: v_<head_slot> for both-sided, unfiltered bank for query-only; OT one query per element,
BoB chunks), a real end-to-end run of the generated element commands (load-time masking) on the
mocked banks, row identity with stock similarity_cli run on written filtered pickles, F7 (iii) receipt fields and violation detection, the Option-C
combine (own version per row, same-patient masking), projection / journal arithmetic from mocked
lsfq pool epilogues, and the control comparison rule. Run with
    python3 -m pytest -p no:cacheprovider --basetemp=<tmp> tests/test_r2_1_tf2_s5_jobspec.py
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_r2_1_tf2_s3_masks as T3  # noqa: E402

S3 = T3.S3
S4 = T3._load("r2_1_tf2_s4_subsets", T3.REPO / "scripts" / "r2_1_tf2_s4_subsets.py")
S5 = T3._load("r2_1_tf2_s5_jobspec", T3.REPO / "scripts" / "r2_1_tf2_s5_jobspec.py")
S0 = S3.S0
QUIET = dict(log=lambda *_: None)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = T3.build_world(tmp_path_factory.mktemp("s5world"))
    S3.AGREEMENT_SAMPLE = 3
    T3.run_s3(w, "uni2", 100)
    w["s3_root"] = w["run"] / "s3"
    w["s3_unit"] = S3.unit_dir(w["s3_root"], "uni2", 100)
    w["s4_unit"] = w["run"] / "s4" / "index" / "uni2" / "b100"
    S4.run_unit("uni2", 100, w["s3_root"], w["s0"], w["s4_unit"], w["prosper_out"], inputs=w["inputs"],
                projection={"status": "PASS", "projected_total_gb": 0.0}, **QUIET)
    S5.PILOT_N = 3
    S5.run_pilot(w["s0"], w["run"] / "s5" / "pilot", w["prosper_out"], inputs=w["inputs"])
    w["pilot"] = w["run"] / "s5" / "pilot" / "pilot_queries.csv"
    w["spec_dir"] = w["run"] / "s5" / "jobspec" / "uni2" / "b100"
    S5.build_specs("uni2", 100, w["s3_unit"], w["s4_unit"], w["s0"], w["pilot"], w["spec_dir"], w["prosper_out"],
                   inputs=w["inputs"], **QUIET)
    w["jobspec"] = w["spec_dir"] / "jobspec.json"
    return w


def _elements(w):
    return S5._jobspec_elements(w["jobspec"])[1]


def test_pilot_seed_reproducible():
    ids = [str(i) for i in range(2604)]
    a = S5.pilot_queries(ids, n=200)
    assert a == S5.pilot_queries(ids, n=200) and len(set(a)) == 200
    assert [int(x) for x in a] == sorted(int(x) for x in a)
    assert a != S5.pilot_queries(ids, n=200, seed=124)
    rng = np.random.Generator(np.random.PCG64(123))
    assert a == [ids[i] for i in np.sort(rng.choice(2604, 200, replace=False))]
    assert S5.PILOT_BUDGETS == (100, 5)


def test_specs_option_c_population_libraries(world):
    el = _elements(world)
    comp4 = json.loads((world["s4_unit"] / "COMPLETE.json").read_text())
    q = pd.read_csv(world["s4_unit"] / "_summary_queries.csv", dtype={"slide_id": str}, keep_default_na=False)
    returned = q[q["written"] == 1].set_index("slide_id")
    ot = el[el["method"] == "ot"]
    assert (ot["n_queries"] == 1).all() and len(ot) == 2 * len(returned)
    for r in ot.itertuples():
        slot = returned.loc[r.query_ids, "head_slot"]
        assert r.head_slot == slot and r.query_head_id == f"uni2/b100/{slot}"
        ids = S5.read_ids(Path(r.library_ids_file), r.library_ids_sha256)
        if r.arm == "both_sided":
            v = comp4["sets"][f"v_{slot}"]
            assert r.library_version == f"v_{slot}" and r.library_index_file.endswith(v["npz"])
            assert ids == S5.lib_order(S4.load_index(world["s4_unit"] / v["npz"]).keys())
        else:
            assert r.library_version == "unfiltered" and r.library_index_file == "" and ids == S5.lib_order(T3.POP)
        assert set(ids) <= set(T3.POP)
        assert r.query_index_file.endswith("queries.npz") and r.library_bank_dir == world["inputs"]["lib_uni2_dir"]
    bob = el[el["method"] == "bob"]
    assert (bob["n_queries"] <= S5.BOB_CHUNK).all() and bob["n_queries"].sum() == 2 * len(returned)
    pilot = set(pd.read_csv(world["pilot"], dtype={"slide_id": str})["slide_id"])
    assert set(el.loc[el["pilot"] == 1, "query_ids"]) == pilot & set(returned.index)
    for arm in S5.ARMS:
        prod = (world["spec_dir"] / arm / "cmds_production.txt").read_text().splitlines()
        pil = (world["spec_dir"] / arm / "cmds_pilot.txt").read_text().splitlines()
        pe = set(el.loc[(el["pilot"] == 1) & (el["arm"] == arm), "element_id"])
        assert not any(c.split("--element ")[1] in pe for c in prod)
        assert len(prod) + len(pil) == len(returned)
        for c in pil + prod:
            assert " element --jobspec " in c and "PYTHONHASHSEED=0" in c and "OMP_NUM_THREADS=1" in c
    spec = json.loads(world["jobspec"].read_text())
    assert spec["shapes"]["ot"] == {"cpu": 1, "mem_gib": 32, "walltime_h": 6, "lsfq": "-n 1 -m 32 -W 6h"}
    assert spec["shapes"]["bob"]["mem_gib"] == 16 and spec["shapes"]["bob"]["walltime_h"] == 8


def _run_cmd(cmd):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LSB_")}
    return subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True, timeout=900)


def test_end_to_end_elements_receipts_and_combine(world):
    el = _elements(world)
    sub = el[(el["arm"] == "both_sided")]
    cmds = {}
    for name in ("cmds_pilot.txt", "cmds_production.txt", "cmds_production_bob.txt"):
        for c in (world["spec_dir"] / "both_sided" / name).read_text().splitlines():
            cmds[c.split("--element ")[1]] = c
    for e in sub["element_id"]:
        r = _run_cmd(cmds[e])
        assert r.returncode == 0, r.stderr[-3000:] + r.stdout[-2000:]
    for e in sub.itertuples():
        rec = json.loads(Path(e.receipt_file).read_text())
        # F7 (iii): receipt names the S0 library version and head id of the query's fold
        assert rec["library_version"] == e.library_version and rec["query_head_id"] == e.query_head_id
        assert rec["f7_iii"]["pass"] and rec["queries"] == e.query_ids.split()
        assert rec["library_index_sha256"] == e.library_index_sha256 and Path(rec["provenance"]["path"]).exists()
    for m in ("ot", "bob"):
        assert S5.f7_iii(world["jobspec"], world["prosper_out"], arms=("both_sided",), methods=(m,))["pass"]
    out = world["run"] / "s5" / "similarity_matrix" / "uni2" / "b100"
    S5.combine(world["jobspec"], "both_sided", "ot", out, world["prosper_out"], inputs=world["inputs"], **QUIET)
    base = S5.matrix_name("uni2", 100, "both_sided", "ot", False)[:-4]
    raw = pickle.loads((out / f"{base}.pkl").read_bytes())
    msk = pickle.loads((out / S5.matrix_name("uni2", 100, "both_sided", "ot", True)).read_bytes())
    side = json.loads((out / f"{base}_rows.json").read_text())
    qa = pd.read_csv(world["s0"] / "query_assignment.csv", dtype={"slide_id": str}).set_index("slide_id")
    assert list(raw.columns) == T3.POP and [r["query"] for r in side["rows"]] == list(raw.index)
    comp4 = json.loads((world["s4_unit"] / "COMPLETE.json").read_text())
    for r in side["rows"]:
        assert r["library_version"] == qa.loc[r["query"], "library_version_both_sided"]
        assert r["head_id"] == f"uni2/b100/{qa.loc[r['query'], 'head_slot']}"
        lib = set(S4.load_index(world["s4_unit"] / comp4["sets"][r["library_version"]]["npz"]))
        for s in T3.POP:
            assert np.isnan(raw.loc[r["query"], s]) == (s not in lib)
        assert raw.loc[r["query"], r["query"]] == S5.SENTINEL
    assert len({r["library_version"] for r in side["rows"]}) >= 2
    if {"101", "102"} <= set(raw.index) | set(raw.columns) and "101" in raw.index and not np.isnan(raw.loc["101", "102"]):
        assert msk.loc["101", "102"] == S5.SENTINEL and raw.loc["101", "102"] != S5.SENTINEL


def test_load_time_rows_equal_stock_similarity_cli_on_written_subsets(world, tmp_path):
    """The load-time element and stock similarity_cli on written filtered pickles give identical rows."""
    el = _elements(world)
    ff = S4.ff
    for method in ("ot", "bob"):
        e = el[(el["arm"] == "both_sided") & (el["method"] == method) & (el["head_slot"] == "full")].iloc[0]
        lidx = S4.load_index(Path(e["library_index_file"]))
        qidx = S4.load_index(Path(e["query_index_file"]))
        lib, qd = tmp_path / f"lib_{method}", tmp_path / f"q_{method}"
        lib.mkdir(); qd.mkdir()
        bank = Path(e["library_bank_dir"])
        for sid, p in lidx.items():
            f, c, _ = ff.load_feature_pickle(bank / f"{sid}.pkl")
            keep = np.zeros(len(c), bool); keep[p] = True
            (lib / f"{sid}.pkl").write_bytes(ff.filtered_bytes(f, c, keep))
        qids = e["query_ids"].split()
        for q in qids:
            f, c, _ = ff.load_feature_pickle(bank / f"{q}.pkl")
            keep = np.zeros(len(c), bool); keep[qidx[q]] = True
            (qd / f"{q}.pkl").write_bytes(ff.filtered_bytes(f, c, keep))
        cmd = (f"cd {S5.PROSPER_SEARCH} && PROSPER_OUT={tmp_path / 'po'} PYTHONHASHSEED=0 OMP_NUM_THREADS=1 "
               f"{S5.PY} -m prosper.embeddings.similarity_cli --encoder uni2 --method {method} --features {qd} "
               f"--library-features {lib} --feature-dim 1536 --lo 1 --hi {len(qids)} --task stock_{method} --num-workers 1")
        r = _run_cmd(cmd)
        assert r.returncode == 0, r.stderr[-3000:]
        for q in qids:
            stock = pickle.loads((tmp_path / "po" / f"stock_{method}" / f"similarity_{q}.pkl").read_bytes())
            mine = pickle.loads((Path(e["rows_dir"]) / f"similarity_{q}.pkl").read_bytes())
            assert stock == mine


def test_receipt_detects_version_violation(world, tmp_path):
    el = _elements(world)
    e = el[(el["arm"] == "both_sided") & (el["method"] == "ot") & (el["head_slot"] == "full")].iloc[0].to_dict()
    s0b = tmp_path / "s0b"
    shutil.copytree(world["s0"], s0b)
    qa = pd.read_csv(s0b / "query_assignment.csv", dtype={"slide_id": str})
    qa.loc[qa["slide_id"] == e["query_ids"], ["head_slot", "library_version_both_sided"]] = ["fold3", "v_fold3"]
    qa.to_csv(s0b / "query_assignment.csv", index=False)
    spec = {"s0": {"dir": str(s0b), "query_assignment_sha256": S0.sha256_file(s0b / "query_assignment.csv")}}
    e["receipt_file"] = str(tmp_path / "r" / "x.json")
    with pytest.raises(S5.S5Error, match="F7 \\(iii\\) violation"):
        S5.receipt(spec, e, [], str(tmp_path / "po"), **QUIET)
    bad = json.loads(Path(e["receipt_file"]).read_text())
    assert not bad["f7_iii"]["pass"] and bad["f7_iii"]["violations"][0]["s0_version"] == "v_fold3"


def _pool(dirp, name, items):
    """items: list of (element_id, cpu_seconds, rc)."""
    d = dirp / name
    for x in ("logs", "markers"):
        (d / x).mkdir(parents=True)
    tasks = {}
    for i, (eid, cpu, rc) in enumerate(items, 1):
        tid = str(i).zfill(5)
        tasks[tid] = {"cmd": f"export X=1; bash job.sh && python3 s5 receipt --jobspec j --element {eid}",
                      "state": "done", "attempts": 1}
        (d / "logs" / f"{tid}.out").write_text(f"stuff\nResource usage summary:\n\n    CPU time :   {cpu:.2f} sec.\n")
        (d / "markers" / f"{tid}.rc").write_text(f"{rc}\n")
    (d / "state.json").write_text(json.dumps({"name": name, "tasks": tasks}))
    return d


def test_projection_and_journal_arithmetic(world, tmp_path):
    el = _elements(world)
    pil = el[el["pilot"] == 1]
    items = [(e, 3600.0 * (1.0 if arm == "both_sided" else 2.0), 0) for e, arm in zip(pil["element_id"], pil["arm"])]
    pool = _pool(tmp_path, "pilot", items)
    jroot = world["run"] / "s5" / "jobspec"
    res = S5.project_cpu([pool], jroot, s4_root=world["run"] / "s4")
    assert res["pilot_arm_means"] == {"both_sided": 1.0, "query_only": 2.0}
    ot = el[el["method"] == "ot"]
    done = set(pil["element_id"])
    rem_spec = sum(ot.loc[(ot["arm"] == a) & ~ot["element_id"].isin(done), "n_queries"].sum() * m
                   for a, m in (("both_sided", 1.0), ("query_only", 2.0)))
    rem_other = (3 * 2 - 1) * 2604 * (1.0 + 2.0)       # encoder x budget units without a jobspec
    consumed = sum(c for _, c, _ in items) / 3600.0
    expect = consumed + rem_spec + rem_other + S5.BOB_FALLBACK_CPU_H_PER_ENCODER * 3 + S5.FIXED_OTHER_CPU_H
    assert res["projected_total_cpu_h"] == pytest.approx(expect, rel=1e-9)
    assert res["status"] == "STOP" and res["tile_check"]["checked"]
    src = {(d["encoder"], d["budget_percent"], d["arm"]): d["per_row_source"] for d in res["detail"]}
    assert src[("uni2", 100, "both_sided")] == "own pilot" and src[("uni2", 5, "both_sided")] == "arm mean"
    small = _pool(tmp_path, "pilot_small", [(e, 3.6, 0) for e in pil["element_id"]])
    assert S5.project_cpu([small], jroot)["status"] == "PASS"
    j = S5.journal([pool, small], jroot)
    assert j["consumed_cpu_h"] == pytest.approx(consumed + 3.6 * len(pil) / 3600.0, rel=1e-9) and j["status"] == "CONTINUE"
    big = _pool(tmp_path, "big", [(pil["element_id"].iat[0], 16000 * 3600.0, 0)])
    assert S5.journal([big], jroot)["status"] == "STOP"
    assert S5.parse_cpu_seconds("CPU time : 1.5 sec.\n...\nCPU time :   90.97 sec.") == 90.97
    assert S5.parse_cpu_seconds("no epilogue") is None


def test_compare_row_rules():
    order = [str(i) for i in range(15)]
    ref = {s: 0.1 + 0.01 * i for i, s in enumerate(order)}
    ref["0"] = 1e6                                     # self sentinel is not an eligible cell
    same = dict(ref)
    assert S5.compare_row(same, ref, order, bitwise=True)["pass"]
    near = {k: v * (1 + 5e-7) for k, v in ref.items()}
    r = S5.compare_row(near, ref, order)
    assert r["pass"] and not S5.compare_row(near, ref, order, bitwise=True)["pass"]
    far = {k: v * (1 + 2e-6) for k, v in ref.items()}
    assert not S5.compare_row(far, ref, order)["pass"]
    swap = dict(ref)
    swap["3"], swap["4"] = ref["4"], ref["3"]          # rank change inside the top 10
    assert not S5.compare_row(swap, ref, order)["top10_equal"]
    assert not S5.compare_row({k: v for k, v in ref.items() if k != "5"}, ref, order)["pass"]
