"""Synthetic fixtures for the R2.1-TF2 S6 evaluator (r2_1_tf2_eval_cli)."""
from __future__ import annotations

import gzip
import io
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prosper.evaluation import r2_1_filter_eval_cli as fe
from prosper.evaluation import r2_1_hoptimus_main as kernel
from prosper.evaluation import r2_1_tf2_eval_cli as T

FROZEN_RER = Path(fe.__file__).resolve().parents[3] / "prosper_revision" / "scripts" / "r2_1_filter_rer.py"
NDRAW = 200


# ------------------------------------------------------------------------------------------------
# Fixture world: 24 slides, 14 patients, 3 groups, 3 folds; two cross-slide patients per group.
# ------------------------------------------------------------------------------------------------
def make_pop(n=24, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.array(["A", "B", "C"] * (n // 3))
    patients = [f"P{i:02d}" for i in range(14)]
    pid = [patients[i % 14] for i in range(n)]
    pid_fold = {p: i % 3 for i, p in enumerate(patients)}
    pop = pd.DataFrame({"query_index": range(n), "reference_index": range(n), "metadata_index": range(n),
                        "slide_id": [f"{100 + i}" for i in range(n)], "patient_id": pid,
                        "diagnostic_group": groups, "fold": [pid_fold[p] for p in pid]})
    return pop


def sentinel_same_patient(d, pop, rows=None):
    rows = pop.slide_id.tolist() if rows is None else rows
    pat = pop.set_index("slide_id").patient_id
    rp = pat.reindex(rows).to_numpy()
    cp = pop.patient_id.to_numpy()
    same = rp[:, None] == cp[None, :]
    d = d.copy()
    d[same & ~np.isnan(d)] = 1e6
    return d


def base_matrix(pop, seed=1):
    rng = np.random.default_rng(seed)
    n = len(pop)
    d = rng.uniform(0.1, 0.9, size=(n, n))
    d = sentinel_same_patient(d, pop)
    ids = pop.slide_id.tolist()
    return pd.DataFrame(d, index=ids, columns=ids)


def filtered_matrix(pop, returned, lib, seed):
    rng = np.random.default_rng(seed)
    n = len(pop)
    d = rng.uniform(0.1, 0.9, size=(n, n))
    d[~lib] = np.nan
    ids = pop.slide_id.tolist()
    rows = [ids[i] for i in np.flatnonzero(returned)]
    d = sentinel_same_patient(d[returned], pop, rows)
    return pd.DataFrame(d, index=rows, columns=ids)


def elig_rows(pop, encoder, budget, arm, versions, members, returned, heads=None):
    """versions: per-query version name; members: version -> list of ids."""
    rows = []
    for i, s in enumerate(pop.slide_id):
        v = versions[i]
        ids = sorted(members[v])
        rows.append({"encoder": encoder, "budget_percent": budget, "query": s, "query_index": i, "arm": arm,
                     "head_slot": heads[i] if heads else f"h{v}", "head_id": f"{encoder}/b{budget}/{v}",
                     "library_version": v, "query_returned": int(returned[i]), "n_eligible": len(ids),
                     "eligible_ids_sha256": T.ids_sha(ids), "eligible_reference_ids": ";".join(ids)})
    return rows


def pickle_to(path, obj):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


class World:
    """Writes population, draws, baseline/filtered matrices, eligibility tables and a config."""

    def __init__(self, tmp: Path, pop=None, methods=("ot_cosine",), encoder="uni2", seed=0,
                 lib_drop=None, nonreturn=None, budgets=("5", "100")):
        self.tmp, self.pop = tmp, make_pop() if pop is None else pop
        self.pop_path = tmp / "pop.csv"
        self.pop.to_csv(self.pop_path, index=False)
        self.draws_dir = tmp / "draws"
        T.make_shared_draws(fe.load_population(self.pop_path), self.draws_dir, population_path=self.pop_path,
                            count=NDRAW)
        n = len(self.pop)
        ids = self.pop.slide_id.tolist()
        fold = self.pop.fold.to_numpy()
        lib_drop = lib_drop or {0: ["101"], 1: ["102", "110"], 2: []}
        nonreturn = nonreturn or {"5": ["105"], "100": []}
        self.encoder, self.arms, self.elig = encoder, [], {}
        for mi, m in enumerate(methods):
            p = pickle_to(tmp / f"base_{m}.pkl", base_matrix(self.pop, seed + 10 + mi))
            self.arms.append({"arm": "all_tissue", "method": m, "budget": "", "matrix": str(p),
                              "matrix_sha256": fe.sha256_file(p)})
        self.lib = {}
        for b in budgets:
            returned = np.array([s not in nonreturn.get(b, []) for s in ids])
            members = {"unfiltered": ids}
            for f, drop in lib_drop.items():
                members[f"v_fold{f}"] = [s for s in ids if s not in drop]
            vb = [f"v_fold{f}" for f in fold]
            rows = elig_rows(self.pop, encoder, b, "query_only", ["unfiltered"] * n, members, returned,
                             heads=[f"fold{f}" for f in fold])
            rows += elig_rows(self.pop, encoder, b, "both_sided", vb, members, returned,
                              heads=[f"fold{f}" for f in fold])
            info = T.write_eligibility_table(pd.DataFrame(rows), tmp / "elig" / encoder / f"b{b}.csv.gz")
            self.elig[b] = {"path": info["path"], "sha256": info["sha256"]}
            col = {s: j for j, s in enumerate(ids)}
            for arm, vers in (("query_only", ["unfiltered"] * n), ("both_filtered", vb)):
                lib = np.zeros((n, n), dtype=bool)
                for i, v in enumerate(vers):
                    lib[i, [col[s] for s in members[v]]] = True
                self.lib[(arm, b)] = (returned, lib)
                for mi, m in enumerate(methods):
                    mat = filtered_matrix(self.pop, returned, lib, seed + 100 + 7 * mi + len(self.arms))
                    p = pickle_to(tmp / f"{arm}_{m}_{b}.pkl", mat)
                    self.arms.append({"arm": arm, "method": m, "budget": b, "matrix": str(p),
                                      "matrix_sha256": fe.sha256_file(p)})

    def config(self, name="config.json", qa=None, arms=None, eligibility=None):
        cfg = {"schema": T.CONFIG_SCHEMA,
               "population": {"path": str(self.pop_path), "sha256": fe.sha256_file(self.pop_path)},
               "query_assignment": qa, "draws_dir": str(self.draws_dir),
               "encoders": [{"encoder": self.encoder, "eligibility": eligibility or self.elig,
                             "arms": self.arms if arms is None else arms}]}
        p = self.tmp / name
        p.write_text(json.dumps(cfg))
        return p


def est_frame(rows):
    return pd.DataFrame(rows, columns=T.EST_HEADER)


def ranked(out_dir):
    return pd.read_csv(Path(out_dir) / "ranked_lists.csv.gz", dtype=str, keep_default_na=False)


# ------------------------------------------------------------------------------------------------
# R2: per-query reference eligibility
# ------------------------------------------------------------------------------------------------
def test_two_queries_in_different_folds_rank_against_their_own_eligible_sets(tmp_path):
    pop = make_pop()
    ids = pop.slide_id.tolist()
    n = len(ids)
    # q_a = "100" (P00, fold 0), q_b = "101"? choose queries of folds 0 and 1 with different patients.
    qa_i = int(np.flatnonzero((pop.fold == 0).to_numpy())[0])
    qb_i = int(np.flatnonzero((pop.fold == 1).to_numpy())[0])
    pa, pb = pop.patient_id[qa_i], pop.patient_id[qb_i]
    x = next(j for j in range(n) if pop.patient_id[j] not in (pa, pb))          # shared nearest reference
    y = next(j for j in range(n) if j != x and pop.patient_id[j] not in (pa, pb, pop.patient_id[x]))
    w = World(tmp_path, pop=pop, lib_drop={0: [ids[x]], 1: [], 2: []}, nonreturn={"5": [], "100": []})
    # Overwrite the both-sided budget-5 matrix so that x is nearest and y second for both queries.
    returned, lib = w.lib[("both_filtered", "5")]
    d = np.full((n, n), 0.5)
    d[lib == False] = np.nan  # noqa: E712
    for q in (qa_i, qb_i):
        if lib[q, x]:
            d[q, x] = 0.1
        d[q, y] = 0.2
    mat = pd.DataFrame(sentinel_same_patient(d, pop), index=ids, columns=ids)
    arm = next(a for a in w.arms if a["arm"] == "both_filtered" and a["budget"] == "5")
    pickle_to(Path(arm["matrix"]), mat)
    arm["matrix_sha256"] = fe.sha256_file(Path(arm["matrix"]))
    cfg = T.load_config(w.config())
    res = T.evaluate(cfg, out_dir=tmp_path / "out")
    r = ranked(tmp_path / "out")
    r = r[(r.arm == "both_filtered") & (r.budget == "5") & (r["rank"] == "1")].set_index("query")
    assert r.loc[ids[qa_i], "neighbour"] == ids[y]      # x is not eligible for the fold-0 query
    assert r.loc[ids[qb_i], "neighbour"] == ids[x]      # x is eligible for the fold-1 query
    q = pd.read_csv(tmp_path / "out" / "query_metrics.csv.gz", dtype=str, keep_default_na=False)
    q = q[(q.arm == "both_filtered") & (q.budget == "5") & (q.k == "1")].set_index("query")
    assert q.loc[ids[qa_i], "library_version"] == "v_fold0"
    assert q.loc[ids[qb_i], "library_version"] == "v_fold1"
    same_a = int((pop.patient_id == pa).sum()) - 1
    assert int(q.loc[ids[qa_i], "n_candidates"]) == n - 1 - 1 - same_a      # minus self, x, same patient
    assert res["summary"]["matrix_diagnostics"]["uni2|both_filtered|ot_cosine|5"]["used_1e6_entries"] == 0


def test_cells_outside_the_eligible_set_must_be_nan_and_eligible_cells_present(tmp_path):
    w = World(tmp_path)
    arm = next(a for a in w.arms if a["arm"] == "both_filtered" and a["budget"] == "100")
    df = pickle.load(open(arm["matrix"], "rb"))
    q = df.index[0]
    fold = int(w.pop.set_index("slide_id").fold[q])
    outside = {0: "101", 1: "102", 2: None}[fold]
    if outside is None:
        q = next(s for s in df.index if int(w.pop.set_index("slide_id").fold[s]) == 0)
        outside = "101"
    df.loc[q, outside] = 0.3
    pickle_to(Path(arm["matrix"]), df)
    arm["matrix_sha256"] = fe.sha256_file(Path(arm["matrix"]))
    with pytest.raises(fe.EvalError, match="outside the rows' eligible sets"):
        T.evaluate(T.load_config(w.config()), out_dir=tmp_path / "o1")
    df.loc[q, outside] = np.nan
    inside = next(c for c in df.columns if not np.isnan(df.loc[q, c]) and df.loc[q, c] != 1e6)
    df.loc[q, inside] = np.nan
    pickle_to(Path(arm["matrix"]), df)
    arm["matrix_sha256"] = fe.sha256_file(Path(arm["matrix"]))
    with pytest.raises(fe.EvalError, match="eligible cells are NaN"):
        T.evaluate(T.load_config(w.config("c2.json")), out_dir=tmp_path / "o2")


def test_eligibility_reader_rejects_inconsistent_tables(tmp_path):
    w = World(tmp_path)
    pop = fe.load_population(w.pop_path)
    df = pd.read_csv(w.elig["5"]["path"], dtype=str, keep_default_na=False)

    def bad(mut, match):
        d = df.copy()
        mut(d)
        p = tmp_path / f"bad_{abs(hash(match)) % 10**6}.csv.gz"
        d.to_csv(p, index=False, compression="gzip")
        with pytest.raises(fe.EvalError, match=match):
            T.read_eligibility(p, pop, "uni2", "5")

    def two_sets(d):
        i = d.index[(d.arm == "both_sided") & (d.library_version == "v_fold0")][1]
        ids = d.at[i, "eligible_reference_ids"].split(";")[1:]
        d.at[i, "eligible_reference_ids"] = ";".join(ids)
        d.at[i, "n_eligible"] = str(len(ids))
        d.at[i, "eligible_ids_sha256"] = T.ids_sha(ids)
    bad(two_sets, "different eligible sets")

    def ret_differs(d):
        i = d.index[(d.arm == "both_sided")][0]
        d.at[i, "query_returned"] = "0" if d.at[i, "query_returned"] == "1" else "1"
    bad(ret_differs, "query_returned differs between arms")

    def outside(d):
        i = d.index[(d.arm == "query_only")][0]
        ids = d.at[i, "eligible_reference_ids"].split(";") + ["999"]
        for j in d.index[d.arm == "query_only"]:
            d.at[j, "eligible_reference_ids"] = ";".join(ids)
            d.at[j, "n_eligible"] = str(len(ids))
            d.at[j, "eligible_ids_sha256"] = T.ids_sha(ids)
    bad(outside, "outside the population")

    def missing_query(d):
        d.drop(index=d.index[(d.arm == "query_only")][0], inplace=True)
    bad(missing_query, "one row per population slide")

    def sha(d):
        i = d.index[(d.arm == "query_only")][0]
        d.at[i, "eligible_ids_sha256"] = "0" * 64
    bad(sha, "sha256")


def test_eligibility_cross_checked_against_s0_query_assignment(tmp_path):
    w = World(tmp_path)
    pop = fe.load_population(w.pop_path)
    qa = pop[["slide_id", "patient_id", "diagnostic_group"]].copy()
    qa["patient_annotated"] = (pop.fold == 0).astype(int).astype(str)
    qa["stratum"] = np.where(pop.fold == 0, "fold0", "unannotated")
    qa["head_slot"] = [f"fold{f}" for f in pop.fold]
    qa["library_version_both_sided"] = [f"v_fold{f}" for f in pop.fold]
    qa["library_version_query_only"] = "unfiltered"
    qp = tmp_path / "qa.csv"
    qa.to_csv(qp, index=False)
    q = T.load_query_assignment(qp, pop)
    T.read_eligibility(w.elig["5"]["path"], pop, "uni2", "5", q)          # agrees
    qa.loc[3, "library_version_both_sided"] = "v_full"
    qa.to_csv(qp, index=False)
    with pytest.raises(fe.EvalError, match="disagrees with S0 query_assignment"):
        T.read_eligibility(w.elig["5"]["path"], pop, "uni2", "5", T.load_query_assignment(qp, pop))


# ------------------------------------------------------------------------------------------------
# F10: one external draw matrix
# ------------------------------------------------------------------------------------------------
def test_shared_policy_equals_kernel_on_full_population_and_uses_columns_on_subsets(tmp_path):
    pop = make_pop()
    groups = tuple(sorted(pop.diagnostic_group.unique()))
    d = kernel.make_patient_draws(pop.patient_id.tolist(), count=NDRAW, seed=123)
    pol = T.SharedDrawPolicy(pop, d, groups)
    rng = np.random.default_rng(3)
    va, vb = rng.uniform(size=len(pop)), rng.uniform(size=len(pop))
    full = np.ones(len(pop), dtype=bool)
    q, p, g = pop.slide_id.tolist(), pop.patient_id.tolist(), pop.diagnostic_group.tolist()
    ref = kernel.summarize_metric(q, p, g, groups, va, full, d)
    got = pol.summarize(va, full)
    assert np.array_equal(ref.macro.replicates, got.macro.replicates, equal_nan=True)
    refp = kernel.summarize_paired(q, p, g, groups, va, vb, full, d).difference
    gotp = pol.paired(va, vb, full)
    for x, y in zip((*refp.group_estimates, refp.macro), (*gotp.group_estimates, gotp.macro)):
        assert x.point == y.point and x.ci_low == y.ci_low and x.ci_high == y.ci_high
        assert np.array_equal(x.replicates, y.replicates, equal_nan=True)
    # Subset: drop every slide of one patient; replicate b uses that replicate's shared multiplicities.
    drop = pop.patient_id == "P03"
    inc = ~drop.to_numpy()
    s = pol.summarize(va, inc)
    W = d.multiplicities
    pc = {pp: i for i, pp in enumerate(d.patient_ids)}
    for b in (0, 7, 50):
        means = []
        for grp in groups:
            sel = inc & (pop.diagnostic_group == grp).to_numpy()
            wq = np.array([W[b, pc[pp]] for pp in pop.patient_id[sel]])
            means.append((wq * va[sel]).sum() / wq.sum() if wq.sum() else np.nan)
        assert s.macro.replicates[b] == pytest.approx(np.mean(means), abs=1e-14)
    assert s.macro.n_patients == len(set(pop.patient_id[inc]))
    # Within-replicate pairing on the subset: difference replicate = A replicate - B replicate.
    dp = pol.paired(va, vb, inc)
    sa, sb = pol.summarize(va, inc), pol.summarize(vb, inc)
    assert np.array_equal(dp.macro.replicates, sa.macro.replicates - sb.macro.replicates, equal_nan=True)
    # The per-population (route-b) rule regenerates draws for the subset and therefore differs.
    pp_ = T.PerPopulationDrawPolicy(pop, d, groups)
    assert not np.array_equal(pp_.summarize(va, inc).macro.replicates, s.macro.replicates)
    assert np.array_equal(pp_.summarize(va, full).macro.replicates, got.macro.replicates)


def test_one_draw_file_bound_for_every_arm_and_regeneration_verified(tmp_path):
    w = World(tmp_path, methods=("ot_cosine", "yottixel"))
    res = T.evaluate(T.load_config(w.config()), out_dir=tmp_path / "out")
    b = json.loads((tmp_path / "out" / "input_bindings.json").read_text())
    assert b["draws"]["regeneration_verified"] is True
    assert b["draws"]["patient_draws_sha256"] == fe.sha256_file(w.draws_dir / "patient_draws.npy")
    assert res["summary"]["draw_policy"]["mode"] == "shared"
    # Contrast replicates are within-replicate differences of the arm replicates on the same matrix.
    reps = np.load(tmp_path / "out" / "replicates.npy")
    est = pd.read_csv(tmp_path / "out" / "estimates.csv", dtype=str, keep_default_na=False)
    con = pd.read_csv(tmp_path / "out" / "contrasts.csv", dtype=str, keep_default_na=False)
    sel = lambda df, **kw: df[np.logical_and.reduce([df[k] == v for k, v in kw.items()])]  # noqa: E731
    c = sel(con, arm_a="query_only|ot_cosine|100", arm_b="all_tissue|ot_cosine|", endpoint="hit_yield", k="5",
            level="macro")
    a = sel(est, arm="query_only", method="ot_cosine", budget="100", endpoint="hit_yield", k="5", level="macro")
    z = sel(est, arm="all_tissue", method="ot_cosine", budget="", endpoint="hit_yield", k="5", level="macro")
    ca, aa, za = (int(x.column.iat[0]) for x in (c, a, z))
    assert np.array_equal(reps[ca], reps[aa] - reps[za], equal_nan=True)
    # Tampering with the draw matrix (sha re-pinned) is caught by the regeneration check.
    m = np.load(w.draws_dir / "patient_draws.npy")
    m[[0, 1]] = m[[1, 0]]
    tam = tmp_path / "tam"
    tam.mkdir()
    np.save(tam / "patient_draws.npy", m)
    (tam / "patient_draws_patients.txt").write_bytes((w.draws_dir / "patient_draws_patients.txt").read_bytes())
    rec = json.loads((w.draws_dir / "draws.json").read_text())
    rec["outputs"]["patient_draws.npy"]["sha256"] = fe.sha256_file(tam / "patient_draws.npy")
    (tam / "draws.json").write_text(json.dumps(rec))
    with pytest.raises(fe.EvalError, match="differs from make_patient_draws"):
        T.load_shared_draws(tam, fe.load_population(w.pop_path))


def test_npy_sink_bytes_equal_np_save(tmp_path):
    rows = [np.arange(5, dtype=float) + i for i in range(4)]
    rows[2][1] = np.nan
    files = {}
    s = T.NpyRowSink(tmp_path, "r.npy")
    for r in rows:
        s.add(r)
    s.close(files)
    buf = io.BytesIO()
    np.save(buf, np.vstack(rows).astype(np.float64))
    assert (tmp_path / "r.npy").read_bytes() == buf.getvalue()
    assert files["r.npy"]["sha256"] == T.sha256_bytes(buf.getvalue())
    assert not (tmp_path / "r.npy.partial").exists()


# ------------------------------------------------------------------------------------------------
# Equivalence with the accepted evaluate-qf (mini F4) and contrast families
# ------------------------------------------------------------------------------------------------
def write_route_b_banks(tmp, pop, written):
    """written: (budget, fold) -> bool array of slides written by head f."""
    root = tmp / "banks"
    for (b, f), w in written.items():
        d = root / f"budget{b}" / f"fold{f}"
        d.mkdir(parents=True)
        pd.DataFrame({"slide_id": pop.slide_id, "written": w.astype(int), "head_fold": f,
                      "slide_fold": pop.fold, "own_fold_query": (pop.fold == f).astype(int)}).to_csv(
            d / "_filter_summary.csv", index=False)
    return root


def test_per_population_mode_reproduces_evaluate_qf_on_a_route_b_fixture(tmp_path):
    pop = make_pop()
    n = len(pop)
    rng = np.random.default_rng(5)
    written = {(b, f): rng.uniform(size=n) > 0.12 for b in ("5", "100") for f in (0, 1, 2)}
    banks = write_route_b_banks(tmp_path, pop, written)
    pop_path = tmp_path / "pop.csv"
    pop.to_csv(pop_path, index=False)
    P = fe.load_population(pop_path)
    base = pickle_to(tmp_path / "base.pkl", base_matrix(pop, 2))
    qf_arms, cfg_arms = [fe.QfArm("all_tissue", "ot_cosine", "", base, None)], [
        {"arm": "all_tissue", "method": "ot_cosine", "budget": "", "matrix": str(base),
         "matrix_sha256": fe.sha256_file(base)}]
    elig = {}
    for b in ("5", "100"):
        et = T.eligibility_from_route_b(P, banks, b)
        info = T.write_eligibility_table(et, tmp_path / "elig" / f"b{b}.csv.gz")
        elig[b] = {"path": info["path"], "sha256": info["sha256"]}
        for arm in ("query_only", "both_filtered"):
            des = fe.qf_design(P, banks, b, arm)
            lib = np.array([des["libraries"][int(f)]["mask"] for f in des["folds"]])
            mat = filtered_matrix(pop, des["returned"], lib, seed=hash((b, arm)) % 1000)
            mp = pickle_to(tmp_path / f"{arm}_{b}.pkl", mat)
            side = pd.DataFrame([{"query": s, "query_index": i, "patient_id": pop.patient_id[i],
                                  "diagnostic_group": pop.diagnostic_group[i], "query_fold": int(des["folds"][i]),
                                  "library_id": des["libraries"][int(des["folds"][i])]["library_id"],
                                  "library_size": des["libraries"][int(des["folds"][i])]["size"],
                                  "library_ids_sha256": des["libraries"][int(des["folds"][i])]["ids_sha256"]}
                                 for i, s in enumerate(pop.slide_id) if des["returned"][i]])
            sp = tmp_path / f"{arm}_{b}_rows.csv"
            side.to_csv(sp, index=False)
            qf_arms.append(fe.QfArm(arm, "ot_cosine", b, mp, sp))
            cfg_arms.append({"arm": arm, "method": "ot_cosine", "budget": b, "matrix": str(mp),
                             "matrix_sha256": fe.sha256_file(mp), "rows": str(sp), "rows_sha256": fe.sha256_file(sp)})
    ref = fe.evaluate_qf(P, qf_arms, banks_root=banks, draw_count=NDRAW, seed=123)
    T.make_shared_draws(P, tmp_path / "draws", count=NDRAW)
    cfg = {"schema": T.CONFIG_SCHEMA, "population": {"path": str(pop_path), "sha256": fe.sha256_file(pop_path)},
           "query_assignment": None, "draws_dir": str(tmp_path / "draws"),
           "encoders": [{"encoder": "uni2", "eligibility": elig, "arms": cfg_arms}]}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    got = T.evaluate(T.load_config(tmp_path / "cfg.json"), draw_mode="per_population", out_dir=tmp_path / "pp")

    def project(rows):
        return [tuple(r[:2]) + tuple(r[5:]) for r in rows]
    assert project(got["_est_rows"]) == list(ref["estimates"])
    assert project(got["_con_rows"]) == list(ref["contrasts"])
    assert [r[0] for r in got["_con_rows"]][:3] == [r[0] for r in ref["contrasts"]][:3]
    fam = Counter(r[2] for r in got["_con_rows"])
    assert set(fam) == {"a_filtered_minus_all_tissue", "b_budget5_minus_budget100", "c_query_only_minus_both_sided"}
    # Shared mode: identical points everywhere; identical rows wherever all patients are included.
    sh = T.evaluate(T.load_config(tmp_path / "cfg.json"), draw_mode="shared", out_dir=tmp_path / "sh")
    H = {c: i for i, c in enumerate(T.EST_HEADER)}
    nfull = P.patient_id.nunique()
    for a, b in zip(sh["_est_rows"] + sh["_con_rows"], got["_est_rows"] + got["_con_rows"]):
        assert a[H["point"]] == b[H["point"]]
    full_rows = [(a, b) for a, b in zip(sh["_est_rows"], got["_est_rows"])
                 if a[H["population"]] == "intended" or a[H["arm"]] == "all_tissue"]
    assert full_rows and all(a == b for a, b in full_rows)
    assert nfull == 14


def test_contrast_specs_order_generalises_qf_contrast_specs():
    arms = [T.ArmSpec("uni2", a, m, b, Path("x"), "s") for m in ("ot_cosine", "yottixel")
            for a, b in [("all_tissue", "")] + [(x, y) for y in ("5", "100") for x in ("query_only", "both_filtered")]]
    got = [(T._legacy(a), T._legacy(b)) for _, a, b in T.contrast_specs(arms)]
    assert got == fe.qf_contrast_specs([a.legacy_key for a in arms])
    tit = [T.ArmSpec("titan", a, "titan", b, Path("x"), "s") for a, b in [("all_tissue", ""), ("query_only", "5"),
                                                                            ("query_only", "100"),
                                                                            ("both_filtered", "5"),
                                                                            ("both_filtered", "100")]]
    fams = Counter(f for f, _, _ in T.contrast_specs(arms + tit))
    assert fams == {"a_filtered_minus_all_tissue": 12, "c_query_only_minus_both_sided": 6,
                    "b_budget5_minus_budget100": 6}
    assert all(a.split("|")[0] == b.split("|")[0] for _, a, b in T.contrast_specs(arms + tit))  # no cross-encoder


# ------------------------------------------------------------------------------------------------
# S6-c (R4), sentinel rule, TITAN
# ------------------------------------------------------------------------------------------------
def test_s6c_removes_only_reference_columns_in_every_arm(tmp_path):
    w = World(tmp_path)
    remove = ["104", "117"]
    rp = tmp_path / "remove.txt"
    rp.write_text("\n".join(remove) + "\n")
    cfg = T.load_config(w.config())
    T.evaluate(cfg, out_dir=tmp_path / "prim")
    T.evaluate(cfg, removed_path=rp, out_dir=tmp_path / "s6c")
    r = ranked(tmp_path / "s6c")
    assert not r.neighbour.isin(remove).any()
    assert ranked(tmp_path / "prim").neighbour.isin(remove).any()
    q0 = pd.read_csv(tmp_path / "prim" / "query_metrics.csv.gz", dtype=str, keep_default_na=False)
    q1 = pd.read_csv(tmp_path / "s6c" / "query_metrics.csv.gz", dtype=str, keep_default_na=False)
    assert len(q0) == len(q1) and (q0.status == q1.status).all()          # queries unchanged
    for s in remove:                                                      # removed slides still queried
        assert (q1[(q1["query"] == s) & (q1.arm == "all_tissue")].status == "returned").all()
    pat = w.pop.set_index("slide_id").patient_id
    k1 = (q0.k == "1") & (q0.status == "returned")
    dn = q0[k1].n_candidates.astype(int).to_numpy() - q1[k1].n_candidates.astype(int).to_numpy()
    rem = q1[k1].removed_eligible.astype(int).to_numpy()
    same = np.array([sum(pat[s] == pat[x] for s in remove) for x in q1[k1]["query"]])
    assert np.array_equal(dn, rem - same)                                 # only eligible, other-patient refs
    lib = pd.read_csv(tmp_path / "s6c" / "libraries.csv", dtype=str, keep_default_na=False)
    assert (lib[lib.arm == "all_tissue"].removed_eligible == "2").all()
    b = json.loads((tmp_path / "s6c" / "input_bindings.json").read_text())
    assert b["removed_reference_slides"]["n_slides"] == 2


def test_make_s6c_lists_counts_and_stop(tmp_path):
    pop = make_pop()
    pop.to_csv(tmp_path / "pop.csv", index=False)
    pool = pd.DataFrame({"slide_id": ["100", "103", "999", "500"], "in_pool": ["1", "1", "1", "0"],
                         "in_final_library": ["1", "1", "0", "1"],
                         "UID.Subject": [pop.patient_id[0], pop.patient_id[3], "PX", pop.patient_id[5]]})
    pool.to_csv(tmp_path / "pool.csv", index=False)
    ann = set(pool[pool.in_pool == "1"]["UID.Subject"])
    qa = pop[["slide_id", "patient_id", "diagnostic_group"]].copy()
    qa["patient_annotated"] = qa.patient_id.isin(ann).astype(int).astype(str)
    for c, v in (("stratum", "x"), ("head_slot", "full"), ("library_version_both_sided", "v_full"),
                 ("library_version_query_only", "unfiltered")):
        qa[c] = v
    qa.to_csv(tmp_path / "qa.csv", index=False)
    rec = T.make_s6c_lists(tmp_path / "pool.csv", tmp_path / "qa.csv", tmp_path / "pop.csv", tmp_path / "o1")
    assert rec["status"] == "STOP" and not (tmp_path / "o1").exists()
    n_sec = int(pop.patient_id.isin(ann).sum())
    rec = T.make_s6c_lists(tmp_path / "pool.csv", tmp_path / "qa.csv", tmp_path / "pop.csv", tmp_path / "o2",
                           expect={"primary": 2, "secondary": n_sec})
    assert rec["status"] == "OK" and rec["counts"] == {"primary": 2, "secondary": n_sec}
    assert (tmp_path / "o2" / "s6c_primary_annotated_population_slides.txt").read_text().split() == ["100", "103"]
    rc = T.main(["make-s6c-lists", "--pool", str(tmp_path / "pool.csv"), "--query-assignment",
                 str(tmp_path / "qa.csv"), "--population", str(tmp_path / "pop.csv"), "--out-dir", str(tmp_path / "o3")])
    assert rc == 1


def test_used_sentinel_is_an_error_unless_allowed_and_counted(tmp_path):
    w = World(tmp_path)
    arm = next(a for a in w.arms if a["arm"] == "query_only" and a["budget"] == "100")
    df = pickle.load(open(arm["matrix"], "rb"))
    q = df.index[0]
    other = next(c for c in df.columns if w.pop.set_index("slide_id").patient_id[c] !=
                 w.pop.set_index("slide_id").patient_id[q])
    df.loc[q, other] = 1e6
    pickle_to(Path(arm["matrix"]), df)
    arm["matrix_sha256"] = fe.sha256_file(Path(arm["matrix"]))
    with pytest.raises(fe.EvalError, match="1e6 sentinel"):
        T.evaluate(T.load_config(w.config()), out_dir=tmp_path / "o1")
    res = T.evaluate(T.load_config(w.config()), allow_used_sentinel=True, out_dir=tmp_path / "o2")
    assert res["summary"]["matrix_diagnostics"]["uni2|query_only|ot_cosine|100"]["used_1e6_entries"] == 1


def test_titan_l1_matrix_is_consumed_and_ranked_ascending(tmp_path):
    pop = make_pop()
    ids = pop.slide_id.tolist()
    n = len(ids)
    rng = np.random.default_rng(11)
    emb = rng.normal(size=(n, 8))
    l1 = np.abs(emb[:, None, :] - emb[None, :, :]).sum(-1)
    w = World(tmp_path, pop=pop, methods=("titan",), encoder="titan")
    base = pd.DataFrame(sentinel_same_patient(l1, pop), index=ids, columns=ids)
    arm = w.arms[0]
    pickle_to(Path(arm["matrix"]), base)
    arm["matrix_sha256"] = fe.sha256_file(Path(arm["matrix"]))
    res = T.evaluate(T.load_config(w.config()), out_dir=tmp_path / "o")
    r = ranked(tmp_path / "o")
    r = r[(r.arm == "all_tissue") & (r.method == "titan")]
    pat = pop.patient_id.to_numpy()
    for i in (0, 5, 13):
        cand = np.flatnonzero(pat != pat[i])
        want = [ids[j] for j in cand[np.argsort(l1[i, cand], kind="mergesort")[:10]]]
        got = r[r["query"] == ids[i]].sort_values("rank", key=lambda s: s.astype(int)).neighbour.tolist()
        assert got == want
    assert any(x["family"] == "a_filtered_minus_all_tissue" and x["arm_a"].endswith("|titan|100")
               for x in res["summary"]["contrast_pairs"])


def test_baseline_bound_by_role_from_s0_bindings_and_sha_enforced(tmp_path):
    w = World(tmp_path)
    m = w.arms[0]
    bind = {"files": [{"role": "baseline:uni2/ot_cosine", "path": m["matrix"], "resolved_path": m["matrix"],
                       "sha256": m["matrix_sha256"]}]}
    bp = tmp_path / "input_bindings.json"
    bp.write_text(json.dumps(bind))
    arms = [{"arm": "all_tissue", "method": "ot_cosine", "binding_role": "baseline:uni2/ot_cosine"}] + w.arms[1:]
    cfgp = w.config("role.json", arms=arms)
    raw = json.loads(cfgp.read_text())
    raw["baseline_bindings"] = {"path": str(bp), "sha256": fe.sha256_file(bp)}
    cfgp.write_text(json.dumps(raw))
    cfg = T.load_config(cfgp)
    assert cfg.arms[0].binding_role == "baseline:uni2/ot_cosine" and str(cfg.arms[0].matrix) == m["matrix"]
    bind["files"][0]["sha256"] = "0" * 64
    bp.write_text(json.dumps(bind))
    raw["baseline_bindings"]["sha256"] = fe.sha256_file(bp)
    cfgp.write_text(json.dumps(raw))
    with pytest.raises(fe.EvalError, match="sha256"):
        T.evaluate(T.load_config(cfgp), out_dir=tmp_path / "o")
    with pytest.raises(fe.EvalError, match="incomplete design"):
        T.load_config(w.config("partial.json", arms=w.arms[:2]))


# ------------------------------------------------------------------------------------------------
# Wording classes (section 6, R5)
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("hit,prec,cls", [
    ((0.01, 0.03), (0.02, 0.04), "i"), ((-0.01, 0.03), (0.02, 0.04), "ii"), ((0.01, 0.03), (-0.01, 0.04), "iii"),
    ((-0.01, 0.03), (-0.02, 0.04), "iv"), ((-0.03, -0.01), (-0.02, 0.04), "v"), ((-0.03, -0.01), (-0.04, -0.02), "v"),
    ((-0.03, -0.01), (0.02, 0.04), "v_mixed"), ((0.01, 0.03), (-0.04, -0.02), "v_mixed"),
    ((0.0, 0.03), (0.02, 0.04), "ii"), ((-0.03, 0.0), (0.02, 0.04), "ii")])
def test_wording_class_on_constructed_intervals(hit, prec, cls):
    hd = T.interval_direction(*hit, "estimable")
    pd_ = T.interval_direction(*prec, "estimable")
    assert T.wording_class(hd, pd_)[0] == cls


def test_wording_texts_and_not_estimable():
    assert T.wording_class("favour", "favour")[1] == "increased hit@5 and precision@5"
    assert T.wording_class("includes_0", "favour")[1] == "increased precision@5; a change in hit@5 was not established"
    assert T.wording_class("favour", "includes_0")[1] == "increased hit@5; a change in precision@5 was not established"
    assert T.wording_class("includes_0", "includes_0")[1] == "no change established"
    assert T.wording_class("against", "favour")[1] == "increased precision@5; decreased hit@5"
    assert T.interval_direction(None, None, "unavailable") == "not_estimable"
    assert T.wording_class("not_estimable", "favour")[0] == "not_estimable"


def _fake_contrast(enc, m, arm, b, ep, lo, hi):
    r = {c: "" for c in T.EST_HEADER}
    r.update(column=0, row_kind="contrast", family="a_filtered_minus_all_tissue", encoder=enc, stratum="all",
             arm=arm, method=m, budget=b, arm_a=f"{arm}|{m}|{b}", arm_b=f"all_tissue|{m}|", endpoint=ep, k=5,
             level="macro", group="", population="both_returned", point=(lo + hi) / 2, point_status="estimable",
             ci_low=lo, ci_high=hi, ci_status="estimable", n_queries=10, n_patients=5, zero_support_replicates=0)
    return tuple(r[c] for c in T.EST_HEADER)


def test_claim_sentences_one_per_encoder_no_comparison_and_rer_headline_only(tmp_path):
    encs = ("uni2", "virchow2_fp32")
    arms = [T.ArmSpec(e, a, m, b, Path("x"), "s") for e in encs for m in ("ot_cosine", "yottixel")
            for a, b in [("all_tissue", "")] + [(x, y) for y in ("5", "100") for x in ("query_only", "both_filtered")]]
    cfg = T.EvalConfig(Path("c"), "s", {}, Path("p"), "s", None, None, Path("d"),
                       [{"encoder": e, "display_name": T.DEFAULT_DISPLAY["encoder"][e]} for e in encs], arms, {}, None)
    rows = []
    for e in encs:
        for m in ("ot_cosine", "yottixel"):
            for arm in ("query_only", "both_filtered"):
                for b in ("5", "100"):
                    rows.append(_fake_contrast(e, m, arm, b, "hit", -0.01, 0.02))
                    rows.append(_fake_contrast(e, m, arm, b, "precision", 0.01, 0.03))
    R = {c: i for i, c in enumerate(T.RER_HEADER)}
    rer = []
    for cell, role in ((("uni2", "ot_cosine", "both_filtered", "100", "miss"), "headline_prenamed_cell"),
                       (("uni2", "yottixel", "both_filtered", "100", "miss"), "macro_table")):
        r = [""] * len(T.RER_HEADER)
        r[R["encoder"]], r[R["method"]], r[R["arm"]], r[R["budget"]], r[R["handling"]] = cell
        r[R["class"]], r[R["cell_role"]] = "MACRO", role
        r[R["rer"]], r[R["rer_ci_low"]], r[R["rer_ci_high"]], r[R["rer_ci_status"]] = 0.25, 0.1, 0.4, "reported"
        rer.append(tuple(r))
    wrows, sent = T.wording_tables(rows, cfg, "primary", rer)
    assert len(wrows) == 2 * 2 * 2 * 2 and {r[-2] for r in wrows} == {"ii"}
    body = [s for s in sent if not s.startswith("#")]
    enc_sent = [s for s in body if not s.startswith("RER")]
    assert len(enc_sent) == 4                                              # 2 encoders x (primary, sensitivity)
    for s in enc_sent:
        names = [T.DEFAULT_DISPLAY["encoder"][e] for e in encs if T.DEFAULT_DISPLAY["encoder"][e] in s]
        assert len(names) == 1                                             # never names two encoders
        for bad in ("best", "better", "worse", "similar", "robust", "consistent", "up to"):
            assert bad not in s.lower()
    assert sum(s.startswith("Sensitivity (5 %") for s in enc_sent) == 2
    rer_lines = [s for s in body if s.startswith("RER")]
    assert len(rer_lines) == 1 and "+0.2500" in rer_lines[0] and "Yottixel" not in rer_lines[0]


# ------------------------------------------------------------------------------------------------
# RER through the frozen core, checked by an independent computation; strata; CLI end to end
# ------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not FROZEN_RER.exists(), reason="frozen RER core not available")
def test_cli_end_to_end_with_rer_strata_and_wording(tmp_path):
    w = World(tmp_path, nonreturn={"5": ["105"], "100": ["111"]})
    pop = fe.load_population(w.pop_path)
    qa = pop[["slide_id", "patient_id", "diagnostic_group"]].copy()
    qa["patient_annotated"] = (pop.fold == 0).astype(int).astype(str)
    qa["stratum"] = np.where(pop.fold == 0, "fold0", "unannotated")
    qa["head_slot"] = [f"fold{f}" for f in pop.fold]
    qa["library_version_both_sided"] = [f"v_fold{f}" for f in pop.fold]
    qa["library_version_query_only"] = "unfiltered"
    qp = tmp_path / "qa.csv"
    qa.to_csv(qp, index=False)
    cfgp = w.config(qa={"path": str(qp), "sha256": fe.sha256_file(qp)})
    rc = T.main(["evaluate", "--config", str(cfgp), "--out-dir", str(tmp_path / "out"),
                 "--frozen-rer", str(FROZEN_RER)])
    assert rc == 0
    out = tmp_path / "out"
    for f in ("estimates.csv", "contrasts.csv", "strata_estimates.csv", "strata_contrasts.csv", "rer_contrasts.csv",
              "wording_classes.csv", "claim_sentences.txt", "COMPLETE.json", "SHA256SUMS.txt",
              "strata_replicates.npy", "tie_counts.csv", "coverage.csv"):
        assert (out / f).exists(), f
    st = pd.read_csv(out / "strata_estimates.csv", dtype=str, keep_default_na=False)
    assert set(st.stratum) == {T.STRATUM_DEPLOYMENT, T.STRATUM_ANNOTATED}
    dep = st[(st.stratum == T.STRATUM_DEPLOYMENT) & (st.arm == "all_tissue") & (st.endpoint == "coverage")
             & (st.level == "macro")]
    assert int(dep.n_queries.iat[0]) == int((pop.fold != 0).sum())
    # Independent RER for one cell: plurality top-5 vote from the ranked lists, macro F1, RER.
    rl = ranked(out)
    q = pd.read_csv(out / "query_metrics.csv.gz", dtype=str, keep_default_na=False)
    truth = dict(zip(pop.slide_id, pop.diagnostic_group))

    def preds(arm, b):
        r = rl[(rl.arm == arm) & (rl.budget == b)]
        st_ = q[(q.arm == arm) & (q.budget == b) & (q.k == "5")].set_index("query").status
        out_ = {}
        for s in pop.slide_id:
            if st_[s] != "returned":
                out_[s] = "NONE"
                continue
            labs = r[r["query"] == s].sort_values("rank", key=lambda x: x.astype(int)).neighbour_group.tolist()[:5]
            c = Counter(labs)
            mx = max(c.values())
            out_[s] = next(lbl for lbl in labs if c[lbl] == mx)
        return out_

    def macro_f1(p):
        f = []
        for g in sorted(set(truth.values())):
            tp = sum(1 for s in p if truth[s] == g and p[s] == g)
            fp = sum(1 for s in p if truth[s] != g and p[s] == g)
            fn = sum(1 for s in p if truth[s] == g and p[s] != g)
            f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        return float(np.mean(f))
    fa, ff = macro_f1(preds("all_tissue", "")), macro_f1(preds("both_filtered", "100"))
    rer = pd.read_csv(out / "rer_contrasts.csv", dtype=str, keep_default_na=False)
    head = rer[rer.cell_role == "headline_prenamed_cell"]
    assert len(head) == 1 and head["class"].iat[0] == "MACRO"
    assert float(head.f1_all.iat[0]) == pytest.approx(fa, abs=1e-12)
    assert float(head.f1_filt.iat[0]) == pytest.approx(ff, abs=1e-12)
    assert float(head.rer.iat[0]) == pytest.approx((ff - fa) / (1 - fa), abs=1e-12)
    wc = pd.read_csv(out / "wording_classes.csv", dtype=str, keep_default_na=False)
    assert set(wc.budget_role) == {"primary", "sensitivity"} and len(wc) == 4
    sent = (out / "claim_sentences.txt").read_text().splitlines()
    assert sum(1 for s in sent if s.startswith("UNI2")) == 1
    assert sum(1 for s in sent if s.startswith("RER headline")) == 1
    sums = dict(reversed(l.split("  ")) for l in (out / "SHA256SUMS.txt").read_text().splitlines())
    assert all(fe.sha256_file(out / n) == h for n, h in sums.items())
    with pytest.raises(fe.EvalError, match="not empty"):
        T.evaluate(T.load_config(cfgp), out_dir=out)


# ------------------------------------------------------------------------------------------------
# Addendum 001 (design decisions Q2, Q7, Q8): JSON sidecar, F4b control, S0 provenance of S6-c lists
# ------------------------------------------------------------------------------------------------
def test_json_rows_sidecar_checked_against_eligibility(tmp_path):
    w = World(tmp_path)
    arm = next(a for a in w.arms if a["arm"] == "both_filtered" and a["budget"] == "5")
    df = pickle.load(open(arm["matrix"], "rb"))
    el = pd.read_csv(w.elig["5"]["path"], dtype=str, keep_default_na=False)
    el = el[el.arm == "both_sided"].set_index("query")
    rows = [{"query": q, "library_version": el.library_version[q], "head_id": el.head_id[q],
             "library_ids_sha256": el.eligible_ids_sha256[q], "library_size": int(el.n_eligible[q])}
            for q in df.index]
    sp = tmp_path / "rows.json"
    sp.write_text(json.dumps({"rows": rows}))
    arm.update(rows=str(sp), rows_sha256=fe.sha256_file(sp))
    res = T.evaluate(T.load_config(w.config()), out_dir=tmp_path / "o1")
    got = res["summary"]["matrix_diagnostics"]["uni2|both_filtered|ot_cosine|5"]["rows_sidecar"]
    assert set(got["fields_checked"]) == {"library_version", "library_ids_sha256", "library_size", "head_id"}
    rows[0]["head_id"] = "conch_v15/b5/wrong"
    sp.write_text(json.dumps(rows))                       # bare list form
    arm["rows_sha256"] = fe.sha256_file(sp)
    with pytest.raises(fe.EvalError, match="head_id differs"):
        T.evaluate(T.load_config(w.config("c2.json")), out_dir=tmp_path / "o2")


def _fake_r26(tmp, pop, l1):
    """A miniature R2.6 common-pool directory computed independently of the evaluator."""
    d = tmp / "r26"
    d.mkdir()
    ids = pop.slide_id.tolist()
    mp = pickle_to(tmp / "titan.pkl", pd.DataFrame(sentinel_same_patient(l1, pop), index=ids, columns=ids))
    (d / "reconciliation.json").write_text(json.dumps(
        {"inputs": {"arm1_matrix": {"path": str(mp), "sha256": fe.sha256_file(mp)}}}))
    pat, lab = pop.patient_id.to_numpy(), pop.diagnostic_group.to_numpy()
    rl, pq = [], []
    for i, q in enumerate(ids):
        cand = np.flatnonzero(pat != pat[i])
        top = cand[np.argsort(l1[i, cand], kind="stable")[:10]]
        rel = lab[top] == lab[i]
        for r, j in enumerate(top, 1):
            rl.append({"query": q, "true_label": lab[i], "rank": r, "neighbour": ids[j], "neighbour_label": lab[j],
                       "distance": l1[i, j]})
        pq.append({"query": q, "true_label": lab[i], "patient": pat[i],
                   **{f"arm1_hit@{k}": float(rel[:k].any()) for k in (1, 3, 5, 10)},
                   "arm1_precision@5": rel[:5].mean()})
    pd.DataFrame(rl).to_csv(d / "ranked_lists_common_pool_arm1.csv.gz", index=False, compression="gzip")
    pq = pd.DataFrame(pq)
    pq.to_csv(d / "per_query_common_pool.csv", index=False)
    pats, mult = (lambda x: (x.patient_ids, x.multiplicities))(
        kernel.make_patient_draws(pat.tolist(), count=5000, seed=123))
    W = mult[:, [pats.index(p) for p in pat]].astype(float)
    classes = sorted(set(lab))
    srows, crows = [], []
    for k in (1, 3, 5, 10):
        v = pq[f"arm1_hit@{k}"].to_numpy()
        cr = np.column_stack([(W[:, lab == c] @ v[lab == c]) / W[:, lab == c].sum(1) for c in classes])
        b = cr.mean(1)
        pt = np.mean([v[lab == c].mean() for c in classes])
        srows.append({"population": "common_pool_2564", "metric": f"macro_hit@{k}", "arm": "arm1", "estimate": pt,
                      "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5), "ci_available": True})
        srows.append({"population": "common_pool_2564", "metric": f"micro_hit@{k}", "arm": "arm1",
                      "estimate": v.mean(), "lo": np.nan, "hi": np.nan, "ci_available": True})
        if k == 5:
            for j, c in enumerate(classes):
                crows.append({"population": "common_pool_2564", "class": c, "metric": "hit@5", "arm": "arm1",
                              "n_class_queries": int((lab == c).sum()),
                              "n_class_patients": len(set(pat[lab == c])), "estimate": v[lab == c].mean(),
                              "lo": np.percentile(cr[:, j], 2.5), "hi": np.percentile(cr[:, j], 97.5),
                              "ci_available": True})
    pd.DataFrame(srows).to_csv(d / "summary_common_pool.csv", index=False)
    pd.DataFrame(crows).to_csv(d / "per_class_common_pool.csv", index=False)
    return d, mp


def test_f4b_titan_reproduces_an_independent_r26_style_computation(tmp_path):
    pop = make_pop(n=36, seed=2)
    ids = pop.slide_id.tolist()
    rng = np.random.default_rng(4)
    emb = rng.normal(size=(len(ids), 6))
    l1 = np.abs(emb[:, None, :] - emb[None, :, :]).sum(-1)
    d, mp = _fake_r26(tmp_path, pop, l1)
    res = T.f4b_titan(d, tmp_path / "out", tol=1e-12)
    assert res["status"] == "PASS" and res["gating"] is False
    assert res["comparisons"]["ranked_lists"]["rows_new"] == 10 * len(ids)
    # A perturbed reference value is detected.
    sm = pd.read_csv(d / "summary_common_pool.csv")
    sm.loc[sm.metric == "macro_hit@5", "hi"] += 1e-6
    sm.to_csv(d / "summary_common_pool.csv", index=False)
    assert T.f4b_titan(d, tmp_path / "out2")["status"] == "FAIL"


def test_make_s6c_lists_binds_inputs_to_s0(tmp_path):
    pop = make_pop()
    pop.to_csv(tmp_path / "pop.csv", index=False)
    pool = pd.DataFrame({"slide_id": ["100", "103"], "in_pool": ["1", "1"], "in_final_library": ["1", "1"],
                         "UID.Subject": [pop.patient_id[0], pop.patient_id[3]]})
    pool.to_csv(tmp_path / "pool.csv", index=False)
    qa = pop[["slide_id", "patient_id", "diagnostic_group"]].copy()
    qa["patient_annotated"] = qa.patient_id.isin(set(pool["UID.Subject"])).astype(int).astype(str)
    for c, v in (("stratum", "x"), ("head_slot", "full"), ("library_version_both_sided", "v_full"),
                 ("library_version_query_only", "unfiltered")):
        qa[c] = v
    qa.to_csv(tmp_path / "qa.csv", index=False)
    s0 = tmp_path / "s0"
    s0.mkdir()
    (s0 / "input_bindings.json").write_text(json.dumps(
        {"files": [{"role": "retrieval_population", "sha256": fe.sha256_file(tmp_path / "pop.csv")}]}))
    comp = {"status": "COMPLETE", "outputs": {n: {"sha256": fe.sha256_file(p)} for n, p in (
        ("pool.csv", tmp_path / "pool.csv"), ("query_assignment.csv", tmp_path / "qa.csv"),
        ("input_bindings.json", s0 / "input_bindings.json"))}}
    (s0 / "COMPLETE.json").write_text(json.dumps(comp))
    n_sec = int(qa.patient_annotated.eq("1").sum())
    rec = T.make_s6c_lists(tmp_path / "pool.csv", tmp_path / "qa.csv", tmp_path / "pop.csv", tmp_path / "o",
                           expect={"primary": 2, "secondary": n_sec}, s0_dir=s0)
    assert rec["status"] == "OK" and rec["s0_provenance"]["pool.csv_sha256"] == fe.sha256_file(tmp_path / "pool.csv")
    pool.loc[1, "in_pool"] = "0"
    pool.to_csv(tmp_path / "pool.csv", index=False)
    with pytest.raises(fe.EvalError, match="S0 pool.csv"):
        T.make_s6c_lists(tmp_path / "pool.csv", tmp_path / "qa.csv", tmp_path / "pop.csv", tmp_path / "o2",
                         expect={"primary": 2, "secondary": n_sec}, s0_dir=s0)
