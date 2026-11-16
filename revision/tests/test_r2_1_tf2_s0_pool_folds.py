"""Synthetic-fixture tests for scripts/r2_1_tf2_s0_pool_folds.py (R2.1-TF2 S0).

No real data are read except the (read-only) code files bound as pins. openslide is replaced by a
fake module (worker processes are forked, so the fake is inherited). Run with
    python3 -m pytest -p no:cacheprovider --basetemp=<tmp> tests/test_r2_1_tf2_s0_pool_folds.py
"""
import collections
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "r2_1_tf2_s0_pool_folds.py"


def _load():
    name = "r2_1_tf2_s0_pool_folds"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S0 = _load()
kernel = S0.kernel


# ---------------------------------------------------------------- unit tests
def test_extend_folds_immune_rule_and_ties():
    accepted = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 0, "U": 4}
    # annotated (existing): A,B,C,D,E,F -> per fold {0:2,1:1,2:1,3:1,4:1}; immune existing: A (fold 0)
    annotated = {"A", "B", "C", "D", "E", "F", "N1", "N2", "N3", "Z9"}
    immune = {"A", "N2"}
    fold, log, start, end = S0.extend_folds(accepted, annotated, immune)
    assert start["annotated_per_fold"] == {0: 2, 1: 1, 2: 1, 3: 1, 4: 1}
    assert start["immune_per_fold"] == {0: 1, 1: 0, 2: 0, 3: 0, 4: 0}
    order = [r["patient_id"] for r in log]
    assert order == ["N1", "N2", "N3", "Z9"]  # UID.Subject order
    by = {r["patient_id"]: r for r in log}
    assert by["N1"]["fold"] == 1 and by["N1"]["rule"] == "fewest_annotated"        # tie 1..4 -> 1
    assert by["N2"]["fold"] == 1 and by["N2"]["rule"].startswith("immune")       # fewest immune: 1..4 -> 1
    assert by["N3"]["fold"] == 2                                                # now {0:2,1:3,2:1,3:1,4:1}
    assert by["Z9"]["fold"] == 3
    assert all(fold[p] == f for p, f in accepted.items())  # existing folds unchanged
    assert fold["U"] == 4 and "U" not in by                 # unannotated union patient untouched
    assert end["annotated_per_fold"] == {0: 2, 1: 3, 2: 2, 3: 2, 4: 1}


def _tiles(rows):
    return pd.DataFrame(rows, columns=["slide_id", "patient_id", "tissue_label"])


def test_class_support_gate():
    fold = {"p0": 0, "p1": 1, "p2": 2, "p3": 3, "p4": 4}
    rows = [("s%d" % i, "p%d" % i, c) for i in range(5) for c in (1, 3, 4, 5)]
    rows += [("s0", "p0", 2), ("s1", "p1", 2)]
    df, ok = S0.class_support(_tiles(rows), fold)
    assert ok
    assert int(df[(df.held_out_fold == 0) & (df.tissue_label == 2)].training_patients.iloc[0]) == 1
    rows2 = [r for r in rows if not (r[2] == 2 and r[1] == "p1")]  # Immune only in fold 0
    df2, ok2 = S0.class_support(_tiles(rows2), fold)
    assert not ok2
    assert int(df2[(df2.held_out_fold == 0) & (df2.tissue_label == 2)].training_patients.iloc[0]) == 0


def _write_pair(d, slide, coords, labels, *, patch_order=None, slide_col=None):
    lab = pd.DataFrame({"tissue_coverage": 1.0, "slide_id": slide_col or slide,
                        "wsi_loc_x": [c[0] for c in coords], "wsi_loc_y": [c[1] for c in coords],
                        "loc_x": 0, "loc_y": 0, "patch_label": labels})
    pat = lab.drop(columns=["patch_label"])
    if patch_order is not None:
        pat = pat.iloc[patch_order]
    lab.to_csv(d / f"{slide}_l.csv", index=False)
    pat.to_csv(d / f"{slide}_p.csv", index=False)
    return d / f"{slide}_l.csv", d / f"{slide}_p.csv"


def test_read_label_pair_rules(tmp_path):
    coords = [(0, 0), (512, 0), (0, 512)]
    l, p = _write_pair(tmp_path, "7", coords, [0, 1, 2])
    t, rec = S0.read_label_pair("7", l, p)
    assert rec["labelled_tiles"] == 2 and list(t["label_csv_row"]) == [0, 1, 2]
    l, p = _write_pair(tmp_path, "8", coords, [0, 1, 2], patch_order=[1, 0, 2])
    with pytest.raises(S0.GateError, match="sequences differ"):
        S0.read_label_pair("8", l, p)
    l, p = _write_pair(tmp_path, "9", coords, [0, 6, 2])
    with pytest.raises(S0.GateError, match="outside 0..5"):
        S0.read_label_pair("9", l, p)
    l, p = _write_pair(tmp_path, "10", [(0, 0), (0, 0)], [1, 1])
    with pytest.raises(S0.GateError, match="duplicate"):
        S0.read_label_pair("10", l, p)
    l, p = _write_pair(tmp_path, "11", coords, [0, 1, 2], slide_col="99")
    with pytest.raises(S0.GateError, match="slide_id mismatch"):
        S0.read_label_pair("11", l, p)


def test_build_pool_classes_and_f1():
    ids = ["1", "2", "3", "4", "5"]
    recs = {"1": {"labelled_tiles": 4}, "2": {"labelled_tiles": 2}, "3": {"labelled_tiles": 3},
            "4": {"labelled_tiles": 0}, "5": {"labelled_tiles": 0}}
    lib = pd.DataFrame({"Tumor.group": ["WT", "NB"], "Slide.Scan.File": ["1", "4"],
                        "UID.Subject": ["PA", "PD"], "Sample.ID": ["x", "y"]})
    rc = pd.DataFrame({"Slide.Scan.File": ["1", "2", "2", "3", "3", "4"],
                       "UID.Subject.x": ["PA", "PB", "PB", "PC", "PC2", "PD"]})
    pool, f1 = S0.build_pool(ids, recs, lib, rc, {"2"}, ann_feature_slides={"1", "2", "3", "4"})
    cls = dict(zip(pool.slide_id, pool.provenance_class))
    assert cls == {"1": "library", "2": "qc_removed", "3": "consolidation_dropped",
                   "4": "listed_unlabelled", "5": "no_features"}
    assert dict(zip(pool.slide_id, pool.patient_source))["2"] == "rcode_output"
    assert f1["library_vs_rcode_agreement"] == {"agree": 1, "of": 1, "disagreements": []}
    # slide 3 has two UIDs in the R-code output -> conflict, no patient
    assert f1["recovered_slides_single_uid"]["single"] == 1
    assert f1["recovered_slides_single_uid"]["conflicts"][0]["slide_id"] == "3"


def _population(rows):
    return pd.DataFrame(rows, columns=["query_index", "reference_index", "metadata_index", "slide_id",
                                       "patient_id", "diagnostic_group", "fold"])


def test_mask_tables_and_f7_checkpoint_i():
    fold = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "X": 0, "Y": 3}
    annotated = {"A", "B", "C", "D", "E"}
    pop = _population([(0, 0, 0, "s1", "A", "WT", 0), (1, 1, 1, "s2", "X", "WT", 0),
                       (2, 2, 2, "s3", "C", "NB", 2), (3, 3, 3, "s4", "Y", "NB", 3)])
    planned = S0.planned_head_patients(annotated, fold)
    assert planned["fold0"] == ["B", "C", "D", "E"] and planned["full"] == ["A", "B", "C", "D", "E"]
    q, table = S0.mask_tables(pop, annotated, fold, planned)
    assert list(q["head_slot"]) == ["fold0", "full", "fold2", "full"]
    assert list(q["library_version_both_sided"]) == ["v_fold0", "v_full", "v_fold2", "v_full"]
    assert len(table) == 4 * 2 * 6 * 2
    res = S0.f7_checkpoint_i(q, table, planned, fold)
    assert res["pass"] and res["strata"] == {"fold0": 1, "fold2": 1, "unannotated": 2}
    # leak: planned fold0 head containing a fold-0 patient
    bad = dict(planned, fold0=sorted(planned["fold0"] + ["A"]))
    q2, t2 = S0.mask_tables(pop, annotated, fold, bad)
    res2 = S0.f7_checkpoint_i(q2, t2, bad, fold)
    kinds = {v["kind"] for v in res2["violations"]}
    assert not res2["pass"] and {"query_patient_in_planned_training",
                                 "planned_fold_head_contains_fold_patients"} <= kinds
    # version/head mismatch
    t3 = table.copy()
    i = t3.index[(t3.entry_kind == "library_version") & (t3.entry_id == "v_fold2")][0]
    t3.loc[i, "head_id"] = "uni2/b100/full"
    res3 = S0.f7_checkpoint_i(q, t3, planned, fold)
    assert not res3["pass"] and any(v["kind"] == "library_version_head_mismatch" for v in res3["violations"])


def test_virchow2_identity_rule(tmp_path):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 2560)).astype(np.float32)
    tiles = [{"wsi_loc": [512 * i, 0]} for i in range(6)]
    V = A[[1, 3, 4], :1280].copy()
    V[1, 7] += np.float32(5e-5)   # listed exception, passes
    pickle.dump((A, tiles), open(tmp_path / "a.pkl", "wb"))
    pickle.dump((V, [tiles[1], tiles[3], tiles[4]]), open(tmp_path / "v.pkl", "wb"))
    rec, exc = S0.virchow2_slide_task(("s", str(tmp_path / "a.pkl"), str(tmp_path / "v.pkl"), 1e-4))
    assert rec["coincident"] == 3 and rec["bitwise_rows"] == 2 and rec["rows_over_limit"] == 0
    assert len(exc) == 1 and exc[0]["x"] == 1536 and not exc[0]["over_limit"]
    V[2, 0] += np.float32(3e-4)   # > 1e-4 FAILS (R9)
    pickle.dump((V, [tiles[1], tiles[3], tiles[4]]), open(tmp_path / "v.pkl", "wb"))
    rec, exc = S0.virchow2_slide_task(("s", str(tmp_path / "a.pkl"), str(tmp_path / "v.pkl"), 1e-4))
    assert rec["rows_over_limit"] == 1 and sum(e["over_limit"] for e in exc) == 1


def test_check_out_dir(tmp_path):
    po = tmp_path / "po"
    root = po / S0.RUN_NAME / "s0"
    root.mkdir(parents=True)
    assert S0.check_out_dir(root, str(po), "s0") == root
    (root / "x").write_text("1")
    with pytest.raises(S0.GateError, match="non-empty"):
        S0.check_out_dir(root, str(po), "s0")
    with pytest.raises(S0.GateError, match="not under"):
        S0.check_out_dir(tmp_path / "elsewhere", str(po), "s0")
    assert S0.check_out_dir(root / "attempt_002", str(po), "s0") == root / "attempt_002"


# ---------------------------------------------------------------- end-to-end fixture
class _FakeSlide:
    dims = {}

    def __init__(self, path):
        self.dimensions = _FakeSlide.dims[Path(path).stem]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def fake_openslide(monkeypatch):
    mod = types.ModuleType("openslide")
    mod.OpenSlide = _FakeSlide
    monkeypatch.setitem(sys.modules, "openslide", mod)
    return _FakeSlide


def _raster_and_labels(rng, nx, ny, label_cells):
    """Raster with 128x128 blocks (= 512 level-0 px at downsample 4); label per block."""
    raster = np.zeros((ny * 128, nx * 128), np.uint8)
    for (i, j), lab in label_cells.items():
        raster[j * 128:(j + 1) * 128, i * 128:(i + 1) * 128] = lab
    return raster


def informative(labels, width, rng):
    """Label-informative synthetic features (class signal + small noise) so that the fixed recipe
    yields finite, normalisable probabilities on held-out patients."""
    X = (0.05 * rng.standard_normal((len(labels), width))).astype(np.float32)
    for i, lab in enumerate(labels):
        if lab > 0:
            X[i, lab] += np.float32(3.0)
    return X


def window_labels_1024(cells, grid):
    """Majority label (ties -> lowest, 0 included) of each 1024 window's four 512 blocks."""
    out = []
    for x, y in grid:
        bx, by = x // 512, y // 512
        vals = [cells.get((bx + dx, by + dy), 0) for dx in (0, 1) for dy in (0, 1)]
        counts = np.bincount(vals, minlength=6)
        out.append(int(np.argmax(counts)))
    return out


def build_fixture(root: Path, fake):
    import h5py
    import tifffile
    rng = np.random.default_rng(1)
    H = root / "H"
    out = root / "PO"
    (out / S0.RUN_NAME / "s0").mkdir(parents=True)
    d = {k: H / k for k in ("labels", "patches", "raster", "ann_uni2", "ann_v2", "ann_ho", "lib_uni2",
                            "lib_v2", "lib_ho", "trident/patches", "trident/features_conch_v15",
                            "wsi_lib", "wsi_non", "misc")}
    for p in d.values():
        p.mkdir(parents=True)
    # slides: L1 (patient P1, Tumor+Immune), L2 (P2), L3 (P3), L4 (P4 tumour+necrosis+nontumour+artifact),
    # L5 (P5), N1 non-library QC (patient P1), N2 non-library dropped (new patient P9 with Immune),
    # U1 listed unlabelled (library, patient P6), Z no features
    def win(spec, extra=None):
        """1024-px windows (wx, wy) -> class, expanded to their four 512-px blocks."""
        cells = {(2 * wx + dx, 2 * wy + dy): c for (wx, wy), c in spec.items() for dx in (0, 1) for dy in (0, 1)}
        cells.update(extra or {})
        return cells

    slides = {
        "L1": ("P1", True, win({(0, 0): 1, (1, 0): 2, (0, 1): 4})),
        "L2": ("P2", True, win({(0, 0): 1, (1, 0): 3, (1, 1): 4})),
        "L3": ("P3", True, win({(0, 0): 1, (1, 0): 5, (0, 1): 4, (1, 1): 3})),
        "L4": ("P4", True, win({(0, 0): 1, (1, 0): 3, (0, 1): 4, (1, 1): 5})),
        "L5": ("P5", True, win({(0, 0): 1, (1, 1): 4, (1, 0): 2})),
        "N1": ("P1", False, win({(0, 0): 1, (1, 0): 4})),
        # N2: top-left window is a four-way tie {1,2,3,5} (lowest label wins -> 1)
        "N2": ("P9", False, win({(1, 0): 2, (0, 1): 5, (1, 1): 3},
                                {(0, 0): 1, (1, 0): 2, (0, 1): 5, (1, 1): 3})),
        "U1": ("P6", True, {}),
        "Z": ("P7", False, {}),
    }
    groups = {"L1": "WT", "L2": "NB", "L3": "WT", "L4": "NB", "L5": "OS", "U1": "NORMAL"}
    (H / "slide_ids.txt").write_text("\n".join(slides) + "\n")
    nx, ny = 4, 4
    for s, (pat, lib, cells) in slides.items():
        coords = [(512 * i, 512 * j) for j in range(ny) for i in range(nx)]
        labels = [cells.get((c[0] // 512, c[1] // 512), 0) for c in coords]
        lab = pd.DataFrame({"tissue_coverage": 1.0, "slide_id": s, "wsi_loc_x": [c[0] for c in coords],
                            "wsi_loc_y": [c[1] for c in coords], "loc_x": 0, "loc_y": 0, "patch_label": labels})
        lab.to_csv(d["labels"] / f"{s}.csv", index=False)
        lab.drop(columns=["patch_label"]).to_csv(d["patches"] / f"{s}.csv", index=False)
        tifffile.imwrite(d["raster"] / f"{s}.tif", _raster_and_labels(rng, nx, ny, cells))
        fake.dims[s] = (nx * 512, ny * 512)
        tiles = [{"wsi_loc": [int(x), int(y)]} for x, y in coords]
        if s != "Z":
            pickle.dump((informative(labels, 1536, rng), tiles),
                        open(d["ann_uni2"] / f"{s}.pkl", "wb"))
            v2 = informative(labels, 2560, rng)
            pickle.dump((v2, tiles), open(d["ann_v2"] / f"{s}.pkl", "wb"))
        if lib:
            pickle.dump((informative(labels, 1536, rng), tiles),
                        open(d["ann_ho"] / f"{s}.pkl", "wb"))
            sub = [0, 3, 5]
            pickle.dump((v2[sub, :1280].copy(), [tiles[i] for i in sub]), open(d["lib_v2"] / f"{s}.pkl", "wb"))
            for k in ("lib_uni2", "lib_ho"):
                pickle.dump((np.zeros((3, 1536), np.float32), tiles[:3]), open(d[k] / f"{s}.pkl", "wb"))
            g = np.array([(1024 * i, 1024 * j) for j in range(2) for i in range(2)], np.int64)
            with h5py.File(d["trident/patches"] / f"{s}_patches.h5", "w") as h:
                ds = h.create_dataset("coords", data=g)
                for k, v in dict(level0_width=nx * 512, level0_height=ny * 512, patch_size=1024,
                                 patch_size_level0=1024, overlap=0).items():
                    ds.attrs[k] = v
            with h5py.File(d["trident/features_conch_v15"] / f"{s}.h5", "w") as h:
                h.create_dataset("coords", data=g)
                h.create_dataset("features", data=informative(window_labels_1024(cells, g.tolist()), 768, rng))
            (d["wsi_lib"] / f"{s}.svs").write_bytes(b"wsi")
        else:
            (d["wsi_non"] / f"{s}.svs").write_bytes(b"wsi")
    lib_slides = [s for s, v in slides.items() if v[1]] + ["R1", "R2", "R3"]
    lib_pat = {**{s: slides[s][0] for s in slides if slides[s][1]}, "R1": "Q1", "R2": "Q2", "R3": "P3"}
    lib_grp = {**groups, "R1": "WT", "R2": "NB", "R3": "WT"}
    pd.DataFrame({"Tumor.group": [lib_grp[s] for s in lib_slides], "Slide.Scan.File": lib_slides,
                  "Sample.ID": "x", "UID.Subject": [lib_pat[s] for s in lib_slides]}
                 ).to_csv(H / "library.csv", index=False)
    for s in ("R1", "R2", "R3"):
        tiles = [{"wsi_loc": [0, 0]}]
        for k in ("lib_uni2", "lib_ho"):
            pickle.dump((np.zeros((1, 1536), np.float32), tiles), open(d[k] / f"{s}.pkl", "wb"))
        pickle.dump((np.zeros((1, 1280), np.float32), tiles), open(d["lib_v2"] / f"{s}.pkl", "wb"))
        with h5py.File(d["trident/patches"] / f"{s}_patches.h5", "w") as h:
            h.create_dataset("coords", data=np.zeros((1, 2), np.int64))
        with h5py.File(d["trident/features_conch_v15"] / f"{s}.h5", "w") as h:
            h.create_dataset("coords", data=np.zeros((1, 2), np.int64))
    rc_rows = [(s, slides[s][0]) for s in slides] + [("N1", "P1"), ("R1", "Q1")]
    pd.DataFrame(rc_rows, columns=["Slide.Scan.File", "UID.Subject.x"]).to_csv(H / "rcode.csv", index=False)
    pd.DataFrame({"compbioid": ["a", "b"], "slides": ["N1.svs", "R9.svs"],
                  "comet_action": ["STAGING BAD SLIDE", "keep"]}).to_csv(H / "qc.csv", index=False)
    # accepted folds (union of retrieval + old annotated): P1..P6, Q1, Q2; P9 new
    acc = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4, "P6": 1, "Q1": 2, "Q2": 3}
    pf = pd.DataFrame({"patient_id": sorted(acc), "fold": [acc[p] for p in sorted(acc)],
                       "kernel_order": range(len(acc)),
                       "patient_hash": [kernel.patient_fold_hash(p) for p in sorted(acc)],
                       "has_annotations": 0, "has_retrieval": 1})
    pf.to_csv(H / "patient_folds.csv", index=False)
    pop_slides = ["L1", "L2", "L3", "L4", "L5", "R1", "R2", "R3"]
    pop = pd.DataFrame({"query_index": range(8), "reference_index": range(8), "metadata_index": range(8),
                        "slide_id": pop_slides, "patient_id": [lib_pat[s] for s in pop_slides],
                        "diagnostic_group": [lib_grp[s] for s in pop_slides],
                        "fold": [acc[lib_pat[s]] for s in pop_slides]})
    pop.to_csv(H / "population.csv", index=False)
    (H / "population_routeb.csv").write_bytes((H / "population.csv").read_bytes())
    # matrices and their references
    mats = {}
    for k in ("uni2/ot_cosine", "uni2/yottixel", "virchow2_fp32/ot_cosine", "virchow2_fp32/yottixel",
              "h_optimus_1/ot_cosine", "h_optimus_1/yottixel", "titan/titan"):
        p = d["misc"] / (k.replace("/", "_") + ".pkl")
        p.write_bytes(k.encode())
        mats[k] = p
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    mg = {"bindings": {f"matrix:{k}": {"path": str(mats[k]), "sha256": sha(mats[k])}
                       for k in ("uni2/ot_cosine", "uni2/yottixel", "h_optimus_1/ot_cosine", "h_optimus_1/yottixel")}}
    (d["misc"] / "mg.json").write_text(json.dumps(mg))
    v2c = {"status": "COMPLETE", "matrices": {
        f"combined_similarity_matrix_virchow2_fp32_{m}_no_same_patient.pkl": sha(mats[f"virchow2_fp32/{m}"])
        for m in ("ot_cosine", "yottixel")}}
    (d["misc"] / "v2run003").mkdir()
    (d["misc"] / "v2run003" / "COMPLETE.json").write_text(json.dumps(v2c))
    # Stage1a references built from the fixture labels
    el, qu = [], []
    for s, (pat, lib, cells) in slides.items():
        lab = pd.read_csv(d["labels"] / f"{s}.csv")
        for r, row in lab.iterrows():
            if row.patch_label > 0:
                tid = kernel.physical_tile_id(s, int(row.wsi_loc_x), int(row.wsi_loc_y))
                if lib:
                    el.append((tid, s, pat, int(row.patch_label), r))
                else:
                    qu.append((tid, s, int(row.patch_label)))
    pd.DataFrame(el, columns=["tile_id", "slide_id", "patient_id", "tissue_label", "h_optimus_1_feature_row"]
                 ).to_csv(d["misc"] / "eligible.csv", index=False)
    pd.DataFrame(qu, columns=["tile_id", "slide_id", "tissue_label"]).to_csv(d["misc"] / "quarantine.csv", index=False)
    pd.DataFrame(columns=S0.LABEL_1024_COLUMNS).to_csv(d["misc"] / "r26.csv.gz", index=False)
    (d["misc"] / "inventory.csv").write_text("slide_id\n")
    (d["misc"] / "ilp.txt").write_text("L1\n")
    (d["misc"] / "further.txt").write_text("L1\nL2\n")
    real = S0.default_inputs(str(S0.HAORAN), str(out))
    inputs = {
        "slide_list": str(H / "slide_ids.txt"), "provenance_ilp_845": str(d["misc"] / "ilp.txt"),
        "provenance_further_238": str(d["misc"] / "further.txt"), "library_map": str(H / "library.csv"),
        "rcode_map": str(H / "rcode.csv"), "qc_list": str(H / "qc.csv"),
        "accepted_folds": str(H / "patient_folds.csv"), "retrieval_population": str(H / "population.csv"),
        "routeb_population": str(H / "population_routeb.csv"),
        "label_dir": str(d["labels"]), "patch_dir": str(d["patches"]), "raster_dir": str(d["raster"]),
        "ann_uni2_dir": str(d["ann_uni2"]), "ann_virchow2_dir": str(d["ann_v2"]),
        "ann_h_optimus_1_dir": str(d["ann_ho"]), "lib_uni2_dir": str(d["lib_uni2"]),
        "lib_virchow2_fp32_dir": str(d["lib_v2"]), "lib_h_optimus_1_dir": str(d["lib_ho"]),
        "trident_root": str(H / "trident"), "wsi_library_dir": str(d["wsi_lib"]),
        "wsi_nonlibrary_dir": str(d["wsi_non"]),
        **{f"baseline:{k}": str(v) for k, v in mats.items()},
        "virchow2_run003_complete": str(d["misc"] / "v2run003" / "COMPLETE.json"),
        "h_optimus_1_main_grid_bindings": str(d["misc"] / "mg.json"),
        **{k: real[k] for k in real if k.startswith("code:")},
        "ref:stage1a_eligible": str(d["misc"] / "eligible.csv"),
        "ref:stage1a_quarantine": str(d["misc"] / "quarantine.csv"),
        "ref:stage1a_slide_inventory": str(d["misc"] / "inventory.csv"),
        "ref:r2_6_labels_1024": str(d["misc"] / "r26.csv.gz"),
    }
    assert set(inputs) == set(real)
    n_lab = sum(len(v[2]) for v in slides.values())
    expected = S0.Expected(listed=9, labelled=7, library=5, outside=2, listed_unlabelled=1, no_features=1,
                           qc_removed=1, consolidation_dropped=1, labelled_tiles=n_lab, union_patients=8,
                           queries=8, groups=3, annotated_patients_expected=6, appended_patients_expected=(1, 1),
                           share_patient_with_library_expected=1)
    return out, inputs, expected


def test_end_to_end_fixture(tmp_path, fake_openslide):
    po, inputs, expected = build_fixture(tmp_path, fake_openslide)
    out = S0.check_out_dir(po / S0.RUN_NAME / "s0", str(po), "s0")
    m = S0.run_s0(out, inputs, expected, workers=1, hash_workers=2, log=lambda *a: None)
    assert m["status"] == "COMPLETE", {k: v for k, v in m["gates"].items() if not v["pass"]}
    pool = pd.read_csv(out / "pool.csv", dtype={"slide_id": str})
    assert list(pool.columns[:8]) == ["slide_id", "labelled_tiles", "in_pool", "in_final_library",
                                      "provenance_class", "UID.Subject", "patient_source", "qc_bad_slide"]
    assert pool.provenance_class.value_counts().to_dict() == {
        "library": 5, "qc_removed": 1, "consolidation_dropped": 1, "listed_unlabelled": 1, "no_features": 1}
    folds = pd.read_csv(out / "patient_folds_tf2.csv")
    app = folds[folds.fold_source == "appended_tf2_s3_2"]
    # P9 is the only new annotated patient and carries Immune: fewest Immune-carrying annotated
    # patients among folds (P1 fold 0, P5 fold 4 carry Immune) -> fold 1 (lowest tie)
    assert app.patient_id.tolist() == ["P9"] and app.fold.tolist() == [1]
    assert app.append_rule.tolist() == ["immune_fewest_immune_annotated"]
    q = pd.read_csv(out / "query_assignment.csv", dtype={"slide_id": str})
    assert q.set_index("slide_id").loc["R1", "head_slot"] == "full"
    assert q.set_index("slide_id").loc["L3", "head_slot"] == "fold2"
    # R3 belongs to annotated patient P3 (fold 2) -> fold head even though R3 itself is unlabelled
    assert q.set_index("slide_id").loc["R3", "head_slot"] == "fold2"
    f7 = json.loads((out / "f7_checkpoint_i.json").read_text())
    assert f7["pass"]
    lab = pd.read_csv(out / "labels_1024.csv.gz", dtype={"slide_id": str})
    assert set(lab.slide_id) == {"L1", "L2", "L3", "L4", "L5"}
    assert m["records"]["labels_1024_coverage"]["pending_s1c"] == ["N1", "N2"]
    assert m["gates"]["F5_rule_reproduces_legacy_512"]["detail"]["n_slides"] == 7
    f8 = json.loads((out / "virchow2_identity_check.json").read_text())
    assert f8["pass"] and f8["coincident_rows"] == 3 * 6 and f8["bitwise_rows"] == 18
    b = json.loads((out / "input_bindings.json").read_text())
    roles = collections.Counter(f["role"] for f in b["files"])
    assert roles["raster_dir"] == 9 and roles["lib_uni2_dir"] == 9 and roles["trident_root"] == 18
    assert all("sha256" in f for f in b["files"]) and len(b["unhashed_wsi"]) == 7
    produced = set(m["outputs"])
    assert set(S0.OUTPUTS) <= produced
    # re-running into the same (now non-empty) destination is refused
    with pytest.raises(S0.GateError, match="non-empty"):
        S0.check_out_dir(out, str(po), "s0")


def test_end_to_end_counts_gate_fails(tmp_path, fake_openslide):
    po, inputs, expected = build_fixture(tmp_path, fake_openslide)
    out = S0.check_out_dir(po / S0.RUN_NAME / "s0", str(po), "s0")
    wrong = S0.Expected(**{**expected.__dict__, "labelled": 8})
    m = S0.run_s0(out, inputs, wrong, workers=1, hash_workers=2, log=lambda *a: None)
    assert m["status"] == "FAILED" and not m["gates"]["counts_8_5_2_1"]["pass"]
    assert (out / "FAILED.json").exists() and not (out / "COMPLETE.json").exists()


def test_end_to_end_f8_failure(tmp_path, fake_openslide):
    po, inputs, expected = build_fixture(tmp_path, fake_openslide)
    p = Path(inputs["lib_virchow2_fp32_dir"]) / "L2.pkl"
    V, t = pickle.load(open(p, "rb"))
    V[0, 0] += np.float32(1e-3)
    pickle.dump((V, t), open(p, "wb"))
    out = S0.check_out_dir(po / S0.RUN_NAME / "s0", str(po), "s0")
    m = S0.run_s0(out, inputs, expected, workers=1, hash_workers=2, log=lambda *a: None)
    assert m["status"] == "FAILED" and not m["gates"]["F8_virchow2_identity"]["pass"]


def test_supplement_labels_for_pending_slides(tmp_path, fake_openslide):
    import h5py
    po, inputs, expected = build_fixture(tmp_path, fake_openslide)
    out = S0.check_out_dir(po / S0.RUN_NAME / "s0", str(po), "s0")
    assert S0.run_s0(out, inputs, expected, workers=1, hash_workers=2, log=lambda *a: None)["status"] == "COMPLETE"
    tr = tmp_path / "s1c" / "40x_1024px_0px_overlap"
    (tr / "patches").mkdir(parents=True)
    (tr / "features_conch_v15").mkdir()
    g = np.array([(0, 0), (1024, 0), (0, 1024), (1024, 1024)], np.int64)
    for s in ("N1", "N2"):
        with h5py.File(tr / "patches" / f"{s}_patches.h5", "w") as h:
            ds = h.create_dataset("coords", data=g)
            for k, v in dict(level0_width=2048, level0_height=2048, patch_size=1024,
                             patch_size_level0=1024, overlap=0).items():
                ds.attrs[k] = v
        with h5py.File(tr / "features_conch_v15" / f"{s}.h5", "w") as h:
            h.create_dataset("coords", data=g)
            h.create_dataset("features", data=np.zeros((4, 768), np.float32))
    sup = S0.check_out_dir(po / S0.RUN_NAME / "s1" / "labels_1024_nonlibrary", str(po), "s1")
    m = S0.run_supplement(out, tr, sup, inputs, workers=1, log=lambda *a: None)
    assert m["status"] == "COMPLETE"
    lab = pd.read_csv(sup / "labels_1024_supplement.csv.gz", dtype={"slide_id": str})
    assert set(lab.slide_id) == {"N1", "N2"} and set(lab.held_out_fold[lab.slide_id == "N2"]) == {1}
    # N2's top-left 1024 window holds labels {1,2,3,5} -> tie, lowest label wins (1)
    assert int(lab[(lab.slide_id == "N2") & (lab.x == 0) & (lab.y == 0)].label_1024.iloc[0]) == 1


def test_supplement_multiple_chunk_roots(tmp_path, fake_openslide):
    import h5py
    po, inputs, expected = build_fixture(tmp_path, fake_openslide)
    out = S0.check_out_dir(po / S0.RUN_NAME / "s0", str(po), "s0")
    assert S0.run_s0(out, inputs, expected, workers=1, hash_workers=2, log=lambda *a: None)["status"] == "COMPLETE"
    g = np.array([(0, 0), (1024, 0), (0, 1024), (1024, 1024)], np.int64)
    roots = []
    for k, slides in enumerate((["N1"], ["N2"], ["N2"]), 1):
        tr = tmp_path / f"chunk{k}" / "40x_1024px_0px_overlap"
        (tr / "patches").mkdir(parents=True)
        (tr / "features_conch_v15").mkdir()
        for s in slides:
            with h5py.File(tr / "patches" / f"{s}_patches.h5", "w") as h:
                ds = h.create_dataset("coords", data=g)
                for key, v in dict(level0_width=2048, level0_height=2048, patch_size=1024,
                                   patch_size_level0=1024, overlap=0).items():
                    ds.attrs[key] = v
            with h5py.File(tr / "features_conch_v15" / f"{s}.h5", "w") as h:
                h.create_dataset("coords", data=g)
                h.create_dataset("features", data=np.zeros((4, 768), np.float32))
        roots.append(tr)
    sup = S0.check_out_dir(po / S0.RUN_NAME / "s1" / "sup_a", str(po), "s1")
    m = S0.run_supplement(out, roots[:2], sup, inputs, workers=1, log=lambda *a: None)
    assert m["status"] == "COMPLETE"
    lab = pd.read_csv(sup / "labels_1024_supplement.csv.gz", dtype={"slide_id": str})
    assert set(lab.slide_id) == {"N1", "N2"}
    sup2 = S0.check_out_dir(po / S0.RUN_NAME / "s1" / "sup_b", str(po), "s1")
    with pytest.raises(S0.GateError, match="more than one root"):
        S0.run_supplement(out, roots, sup2, inputs, workers=1, log=lambda *a: None)


# ---------------------------------------------------------------- amendment P1: slide exclusion
def exclusion_expected(expected, excluded_nonlibrary):
    """Post-exclusion gated counts for excluding `excluded_nonlibrary` non-library pool slides."""
    n = len(excluded_nonlibrary)
    return S0.Expected(**{**expected.__dict__, "labelled": expected.labelled - n,
                          "outside": expected.outside - n, "excluded": n})


def write_exclusion(tmp_path, slides):
    p = tmp_path / "excluded.txt"
    p.write_text("".join(f"{s}\n" for s in slides))
    return p


def test_end_to_end_exclusion(tmp_path, fake_openslide):
    po, inputs, expected = build_fixture(tmp_path, fake_openslide)
    ex = write_exclusion(tmp_path, ["N2"])
    out = S0.check_out_dir(po / S0.RUN_NAME / "s0" / "run_002", str(po), "s0")
    m = S0.run_s0(out, inputs, exclusion_expected(expected, ["N2"]), workers=1, hash_workers=2,
                  exclusion={"path": str(ex), "reason": "scan_magnification_20x"}, log=lambda *a: None)
    assert m["status"] == "COMPLETE", {k: v for k, v in m["gates"].items() if not v["pass"]}
    g = m["gates"]["counts_6_5_1_1"]["detail"]
    assert g["observed"]["excluded"] == 1 and g["observed"]["labelled_tiles"] == expected.labelled_tiles
    assert g["pool_labelled_tiles_after_exclusion"] < expected.labelled_tiles
    # the excluded slide stays in pool.csv: in_pool 0, original provenance class, reason column
    pool = pd.read_csv(out / "pool.csv", dtype={"slide_id": str}, keep_default_na=False)
    assert list(pool.columns) == S0.POOL_COLUMNS and len(pool) == 9
    r = pool.set_index("slide_id").loc["N2"]
    assert int(r.in_pool) == 0 and r.provenance_class == "consolidation_dropped"
    assert r.excluded_reason == "scan_magnification_20x"
    assert set(pool.loc[pool.slide_id != "N2", "excluded_reason"]) == {""}
    # downstream recomputed from the remaining slides
    tiles = pd.read_csv(out / "annotation_tiles.csv.gz", dtype={"slide_id": str})
    assert "N2" not in set(tiles.slide_id) and "N1" in set(tiles.slide_id)
    folds = pd.read_csv(out / "patient_folds_tf2.csv")
    assert "P9" not in set(folds.patient_id)               # P9 was annotated only through N2
    assert (folds.fold_source == "appended_tf2_s3_2").sum() == 0
    ex_rec = m["records"]["exclusion"]
    assert [p["patient_id"] for p in ex_rec["patients_leaving_annotated_set"]] == ["P9"]
    assert ex_rec["patients_leaving_annotated_set"][0]["carried_immune"]
    assert ex_rec["immune_patients_lost"] == ["P9"]
    assert ex_rec["annotated_patients_before_exclusion"] == 6 and m["records"]["annotated_patients"] == 5
    assert ex_rec["immune_carriers_per_fold"] == {"before": {0: 1, 1: 1, 2: 0, 3: 0, 4: 1},
                                                  "after": {0: 1, 1: 0, 2: 0, 3: 0, 4: 1}}
    changed = {(c["held_out_fold"], c["tissue_label"]) for c in ex_rec["class_support_patient_count_changes"]}
    assert ("full", 2) in changed and ("0", 2) in changed and ("1", 2) not in changed
    support = pd.read_csv(out / "class_support.csv")
    full_imm = support[(support.held_out_fold == "full") & (support.tissue_label == 2)]
    assert int(full_imm.training_patients.iloc[0]) == 2
    planned = pd.read_csv(out / "planned_head_patients.csv")
    assert "P9" not in set(planned.patient_id)
    lab = pd.read_csv(out / "labels_1024.csv.gz", dtype={"slide_id": str})
    assert set(lab.slide_id) == {"L1", "L2", "L3", "L4", "L5"}
    assert m["records"]["labels_1024_coverage"]["pending_s1c"] == ["N1"]
    assert m["gates"]["F5_rule_reproduces_legacy_512"]["detail"]["n_slides"] == 6
    assert m["gates"]["stage1a_regression"]["pass"]       # evaluated before exclusion
    assert json.loads((out / "f7_checkpoint_i.json").read_text())["pass"]
    b = json.loads((out / "input_bindings.json").read_text())
    rec = [f for f in b["files"] if f["role"] == "exclusion_list"]
    assert len(rec) == 1 and rec[0]["sha256"] == hashlib.sha256(ex.read_bytes()).hexdigest()
    assert b["exclusion"]["reason"] == "scan_magnification_20x"
    assert len(b["unhashed_wsi"]) == 6                      # WSI of the excluded slide not read


def test_exclusion_keeps_patient_with_other_slides(tmp_path, fake_openslide):
    po, inputs, expected = build_fixture(tmp_path, fake_openslide)
    ex = write_exclusion(tmp_path, ["N1"])                  # N1's patient P1 also owns library slide L1
    out = S0.check_out_dir(po / S0.RUN_NAME / "s0" / "run_x", str(po), "s0")
    m = S0.run_s0(out, inputs, exclusion_expected(expected, ["N1"]), workers=1, hash_workers=2,
                  exclusion={"path": str(ex), "reason": "r"}, log=lambda *a: None)
    assert m["status"] == "COMPLETE"
    ex_rec = m["records"]["exclusion"]
    assert ex_rec["patients_leaving_annotated_set"] == []
    assert ex_rec["patients_with_excluded_slides_remaining_annotated"] == ["P1"]
    pool = pd.read_csv(out / "pool.csv", dtype={"slide_id": str}, keep_default_na=False)
    assert pool.set_index("slide_id").loc["N1", "provenance_class"] == "qc_removed"


def test_exclusion_refuses_non_pool_slide_and_wrong_counts(tmp_path, fake_openslide):
    po, inputs, expected = build_fixture(tmp_path, fake_openslide)
    ex = write_exclusion(tmp_path, ["U1"])                  # listed but unlabelled: not a pool slide
    out = S0.check_out_dir(po / S0.RUN_NAME / "s0" / "run_a", str(po), "s0")
    with pytest.raises(S0.GateError, match="would not be in the pool"):
        S0.run_s0(out, inputs, expected, workers=1, hash_workers=2,
                  exclusion={"path": str(ex), "reason": "r"}, log=lambda *a: None)
    assert (out / "FAILED.json").exists()
    ex2 = write_exclusion(tmp_path, ["N2"])
    out2 = S0.check_out_dir(po / S0.RUN_NAME / "s0" / "run_b", str(po), "s0")
    m = S0.run_s0(out2, inputs, expected, workers=1, hash_workers=2,   # default counts: gate must fail
                  exclusion={"path": str(ex2), "reason": "r"}, log=lambda *a: None)
    assert m["status"] == "FAILED" and not m["gates"]["counts_7_5_2_1"]["pass"]


def test_cli_exclusion_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("PROSPER_OUT", str(tmp_path))
    with pytest.raises(SystemExit, match="together"):
        S0.main(["run", "--out", str(tmp_path / "x"), "--exclude-slides", str(tmp_path / "e.txt")])
    with pytest.raises(SystemExit, match="four integers"):
        S0.main(["run", "--out", str(tmp_path / "x"), "--expected-counts", "229,202,27"])


def _trident_root(base, specs):
    """specs: slide -> patch_size_level0."""
    import h5py
    tr = base / "40x_1024px_0px_overlap"
    (tr / "patches").mkdir(parents=True)
    (tr / "features_conch_v15").mkdir()
    for s, psl0 in specs.items():
        step = psl0
        g = np.array([(0, 0), (step, 0), (0, step), (step, step)], np.int64)
        with h5py.File(tr / "patches" / f"{s}_patches.h5", "w") as h:
            ds = h.create_dataset("coords", data=g)
            for k, v in dict(level0_width=2048, level0_height=2048, patch_size=1024,
                             patch_size_level0=psl0, overlap=0).items():
                ds.attrs[k] = v
        with h5py.File(tr / "features_conch_v15" / f"{s}.h5", "w") as h:
            h.create_dataset("coords", data=g)
            h.create_dataset("features", data=np.zeros((4, 768), np.float32))
    return tr


def test_supplement_restricted_to_pool_nonlibrary_slides(tmp_path, fake_openslide):
    po, inputs, expected = build_fixture(tmp_path, fake_openslide)
    ex = write_exclusion(tmp_path, ["N2"])
    out = S0.check_out_dir(po / S0.RUN_NAME / "s0" / "run_002", str(po), "s0")
    assert S0.run_s0(out, inputs, exclusion_expected(expected, ["N2"]), workers=1, hash_workers=2,
                     exclusion={"path": str(ex), "reason": "scan_magnification_20x"},
                     log=lambda *a: None)["status"] == "COMPLETE"
    # the chunk root also holds the excluded slide at 20x (patch_size_level0 512): it is not read
    tr = _trident_root(tmp_path / "chunk1", {"N1": 1024, "N2": 512})
    sup = S0.check_out_dir(po / S0.RUN_NAME / "s1" / "labels_1024_nonlibrary_002", str(po), "s1")
    m = S0.run_supplement(out, [tr], sup, inputs, workers=1, log=lambda *a: None)
    assert m["status"] == "COMPLETE"
    rep = json.loads((sup / "labels_1024_supplement_report.json").read_text())
    assert rep["slides"] == ["N1"] and rep["s0_excluded_slides_not_read"] == ["N2"]
    assert rep["per_slide_counts"][0]["tiles_1024"] == 4 and rep["totals"]["tiles_1024"] == 4
    assert rep["per_slide_counts"][0]["labelled_1to5"] == sum(
        rep["per_slide_counts"][0][f"label_{c}"] for c in (1, 2, 3, 4, 5))
    lab = pd.read_csv(sup / "labels_1024_supplement.csv.gz", dtype={"slide_id": str})
    assert set(lab.slide_id) == {"N1"}
    # explicitly requesting a slide outside the pool's non-library slides is refused
    sup2 = S0.check_out_dir(po / S0.RUN_NAME / "s1" / "sup_c", str(po), "s1")
    with pytest.raises(S0.GateError, match="non-library"):
        S0.run_supplement(out, [tr], sup2, inputs, slides=["N2"], workers=1, log=lambda *a: None)


def test_supplement_refuses_non_1024_level0_patch(tmp_path, fake_openslide):
    po, inputs, expected = build_fixture(tmp_path, fake_openslide)
    out = S0.check_out_dir(po / S0.RUN_NAME / "s0", str(po), "s0")
    assert S0.run_s0(out, inputs, expected, workers=1, hash_workers=2, log=lambda *a: None)["status"] == "COMPLETE"
    tr = _trident_root(tmp_path / "chunk1", {"N1": 1024, "N2": 512})
    sup = S0.check_out_dir(po / S0.RUN_NAME / "s1" / "sup_d", str(po), "s1")
    with pytest.raises(S0.GateError, match="unexpected patch attrs"):
        S0.run_supplement(out, [tr], sup, inputs, workers=1, log=lambda *a: None)
