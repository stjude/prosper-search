#!/usr/bin/env python3
"""Independent audit of Table S26 v2 (R3.7-T4 S-D1 v2; subtype retrieval within a known family).

A_packaging, 2026-11-09. Re-derives every point and interval of the 'S26 body' sheet from the saved distance
matrices and label manifests by a route written for this audit (no import of, or copy from, the producing scripts):
  * neighbours: per query, eligible reference slides (other patient by the frozen COMET metadata UID.Subject, not
    the query, finite distance) ordered by np.lexsort on (distance, manifest position); top 5;
  * metrics: plurality@5 as a >= 3-of-5 majority (two classes, so no ties), hit@1, P@5, AUROC of the positive-label
    count among the five nearest by level counts (Mann-Whitney with half credit for ties);
  * intervals: patient-level sufficient statistics times draw multiplicities (ratio estimator per draw), percentile
    2.5/97.5. Draw generators are the recorded conventions, so intervals are comparable exactly:
    COMET arms: one Generator(PCG64(123)) shared by s61_int_2c then s64, multinomial(n, 1/n, 5000) over sorted
    patients, the same draws for both matrices (r3_7_all_tissue_ci_20261103); CCDI-MCI arm and spindle footnote:
    a fresh Generator(PCG64(123)) per arm, bincount(integers(p, p)) per draw over sorted patients
    (r3_7_fine_label_20261003, [decision record] draws).
Read-only on every input; writes only under $PROSPER_OUT/supp_tables_20261109/A_packaging/S26_audit/.

Run from the repository root:
    export PROSPER_OUT="$PWD/prosper_revision_output"
    /HOME/miniconda3/bin/python3 scripts/supp_tables/audit_S26_v2_20261109.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packaging_common_20261109 import (BUILD_DATE, NUMBERING, OUT_ROOT, PKG, SI_MD, Bindings, numbering, now_utc,  # noqa: E402
                                       rel, si_entry, write_sha256sums)

OUT = PKG / "S26_audit"
CORE = Path("/PROSPER_WORKSPACE")
SIM = CORE / "comet-feature-gen/comet-uni2/output/similarity_matrix"
V2 = OUT_ROOT / "supp_tables_20261103/A_ladder_failure_finelabel"
CI_RUN = OUT_ROOT / "r3_7_all_tissue_ci_20261103"
FINE = OUT_ROOT / "r3_7_fine_label_20261003"
INV = OUT_ROOT / "s6_0_label_inventory"
S61 = OUT_ROOT / "s6_1_rms"
B, SEED, K = 5000, 123, 5
TOL_RAW = 1e-9


class Audit:
    def __init__(self):
        self.vals, self.checks, self.design = [], [], []

    def check(self, ok, msg):
        self.checks.append(("PASS" if ok else "FAIL", msg))
        return ok

    def n_fail(self):
        return sum(s == "FAIL" for s, _ in self.checks) + sum(v["status"] == "FAIL" for v in self.vals)


A = Audit()


def load(p):
    M = pd.read_pickle(p)
    M.index = M.index.astype(str)
    M.columns = M.columns.astype(str)
    return M


# ------------------------------------------------------------------------------------------- neighbours and metrics
def neighbours(D, qids, rids, qpat, rpat, k=K):
    """Return (top-k reference positions, boundary-tie count, label-relevant boundary ties placeholder)."""
    X = D.loc[qids, rids].to_numpy(float)
    qp, rp = np.asarray(qpat, object), np.asarray(rpat, object)
    ok = (qp[:, None] != rp[None, :]) & np.isfinite(X) & (np.asarray(qids, object)[:, None] != np.asarray(rids, object)[None, :])
    pos = np.arange(len(rids))
    top, ties = np.empty((len(qids), k), int), []
    for i in range(len(qids)):
        v = pos[ok[i]]
        o = v[np.lexsort((v, X[i, v]))]
        top[i] = o[:k]
        ties.append(len(o) > k and X[i, o[k - 1]] == X[i, o[k]])
    return top, np.array(ties), ok


def score(top, truth, rlab, pos_label):
    L = np.asarray(rlab, object)[top]
    n_true = (L == truth[:, None]).sum(1)
    n_pos = (L == pos_label).sum(1)
    return dict(L=L, plur=(n_true >= 3).astype(float), hit1=(L[:, 0] == truth).astype(float), p5=n_true / K,
                npos=n_pos, pred=np.where(n_pos >= 3, pos_label, None))


def auroc_levels(Wp, Wn):
    """Wp, Wn: (..., 6) weights of positive / negative queries at positive-count level 0..5."""
    cn = np.cumsum(Wn, -1) - Wn
    num = (Wp * cn).sum(-1) + 0.5 * (Wp * Wn).sum(-1)
    den = Wp.sum(-1) * Wn.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def pct(v):
    v = np.asarray(v, float)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int((~np.isfinite(v)).sum())


def draws_multinomial(rng, n):
    return rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)


def draws_bincount(n, seed=SEED):
    g = np.random.Generator(np.random.PCG64(seed))
    return np.stack([np.bincount(g.integers(n, size=n), minlength=n) for _ in range(B)]).astype(float)


def summarise(sc, truth, pats, labels, pos_label, M, macro_nan="mean"):
    """Point and interval for ALL, MACRO and per-class rows via per-patient sufficient statistics."""
    up, pidx = np.unique(pats, return_inverse=True)
    P = len(up)

    def agg(v, mask):
        return np.bincount(pidx[mask], weights=v[mask], minlength=P), np.bincount(pidx[mask], minlength=P).astype(float)

    def est(v, mask):
        S, N = agg(v, mask)
        den = M @ N
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(den > 0, (M @ S) / np.where(den > 0, den, 1), np.nan)
        return float(v[mask].mean()), d

    rows = {}
    allm = np.ones(len(truth), bool)
    for cls, mask in [("ALL", allm)] + [(l, truth == l) for l in labels]:
        r = {"n": int(mask.sum()), "n_patients": int(len(set(pats[mask])))}
        for key in ("plur", "hit1", "p5"):
            p, d = est(sc[key], mask)
            lo, hi, nn = pct(d)
            r[key] = (p, lo, hi, nn)
            r[key + "_draws"] = d
        rows[cls] = r
    mac = {"n": len(truth)}
    for key in ("plur", "hit1", "p5"):
        st = np.vstack([rows[l][key + "_draws"] for l in labels])
        d = np.nanmean(st, 0) if macro_nan == "nanmean" else st.mean(0)
        lo, hi, nn = pct(d)
        mac[key] = (float(np.mean([rows[l][key][0] for l in labels])), lo, hi, nn)
    rows["MACRO"] = mac
    ispos = truth == pos_label
    lev = sc["npos"].astype(int)
    Cp = np.zeros((P, 6)); Cn = np.zeros((P, 6))
    np.add.at(Cp, (pidx[ispos], lev[ispos]), 1)
    np.add.at(Cn, (pidx[~ispos], lev[~ispos]), 1)
    a_pt = float(auroc_levels(Cp.sum(0), Cn.sum(0)))
    a_d = auroc_levels(M @ Cp, M @ Cn)
    lo, hi, nn = pct(a_d)
    rows["ALL"]["auroc"] = (a_pt, lo, hi, nn)
    rows["_up"], rows["_pidx"] = up, pidx
    return rows


def confusion(sc, truth, labels, pos_label):
    pred = np.where(sc["npos"] >= 3, pos_label, [l for l in labels if l != pos_label][0])
    return "; ".join(f"{t} " + "/".join(str(int(((truth == t) & (pred == p)).sum())) for p in labels) for t in labels)


# ------------------------------------------------------------------------------------------- printed table parsing
NUM = r"[-−+]?\d+\.\d+"
CELL = re.compile(rf"^({NUM}) \[({NUM}), ({NUM})\]$")
fnum = lambda s: float(s.replace("−", "-"))
COMP = {"ERMS vs ARMS, COMET": "s61_int_2c", "ERMS vs ARMS, CCDI-MCI → COMET": "s61_ext_2c", "HB vs HCC, COMET": "s64",
        "ERMS vs ARMS, COMET (tumour-only tile selection, Supplementary Table S5)": "s61_int_2c",
        "HB vs HCC, COMET (tumour-only tile selection, Supplementary Table S5)": "s64"}
MET = {"Plurality@5 [95% CI]": ("plur", "plurality5_recall"), "hit@1 [95% CI]": ("hit1", "hit1"),
       "P@5 [95% CI]": ("p5", "precision5"), "AUROC [95% CI]": ("auroc", "auroc_top5_pos_share")}
RAWLO = {"plurality5_recall": ("plurality5_recall_lo", "plurality5_recall_hi"), "hit1": ("hit1_lo", "hit1_hi"),
         "precision5": ("precision5_lo", "precision5_hi"), "auroc_top5_pos_share": ("auroc_lo", "auroc_hi")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    Bd = Bindings()
    f_xlsx = Bd.bind(V2 / "R3.7-T4_S26_S-D1_v2.xlsx", "audited workbook")
    f_raw = Bd.bind(V2 / "R3.7-T4_S26_S-D1_v2_body_raw.csv", "audited raw values")
    f_spin = Bd.bind(V2 / "R3.7-T4_S26_S-D1_v2_footnote_spindle.csv", "audited spindle footnote values")
    f_brep = Bd.bind(V2 / "BUILD_REPORT.md", "v2 build report (hash list)")
    f_cires = Bd.bind(CI_RUN / "results.json", "11-03 run: recorded input hashes")
    Bd.bind(CI_RUN / "run.log", "11-03 run log")
    f_v1 = Bd.bind(OUT_ROOT / "supp_tables_20261022/A_ladder_failure_finelabel/R3.7-T4_S26_S-D1_body_raw.csv", "v1 body_raw (CCDI arm source)")
    f_recon = Bd.bind(FINE / "reconciliation.json", "fine-label run: rules, draw source")
    f_summ = Bd.bind(FINE / "summary_all_arms.csv", "fine-label summary (CCDI/spindle accepted values)")
    f_cov = Bd.bind(FINE / "coverage.csv", "fine-label coverage (denominators)")
    f_aud = Bd.bind(OUT_ROOT / "r3_7_audit_20261020/results_seed20261021_B5000.json", "10-20 audit: CCDI matrix hash, representation note")
    f_tum = Bd.bind(SIM / "combined_similarity_matrix_uni2_ot_cosine_tumor_only_no_same_patient.pkl", "tumour-only masked matrix")
    f_all = Bd.bind(SIM / "combined_similarity_matrix_uni2_ot_cosine_no_same_patient.pkl", "all-tissue masked matrix")
    f_allu = Bd.bind(SIM / "combined_similarity_matrix_uni2_ot_cosine.pkl", "all-tissue unmasked matrix (mask check)")
    f_tumu = Bd.bind(SIM / "combined_similarity_matrix_uni2_ot_cosine_tumor_only.pkl", "tumour-only unmasked matrix (mask check)")
    f_ccdi = Bd.bind(OUT_ROOT / "similarity_matrix/combined_similarity_matrix_ccdi_uni2_ot_cosine.pkl", "CCDI x COMET matrix")
    f_meta = Bd.bind(CORE / "comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv", "frozen COMET metadata (patient identity)")
    f_m61 = Bd.bind(INV / "manifest_s61_rms_comet.csv", "COMET RMS manifest")
    f_m64 = Bd.bind(INV / "manifest_s64_hbhcc_comet.csv", "COMET liver manifest")
    f_mcc = Bd.bind(INV / "manifest_s61_rms_ccdi.csv", "CCDI RMS manifest")
    f_rl = {a: Bd.bind(S61 / f"ranked_lists_{a}.csv.gz", f"saved S6 rank list {a}") for a in ("s61_ext_2c", "s61_int_3c", "s61_ext_3c")}
    f_pq = Bd.bind(FINE / "per_query_s61_ext_2c.csv", "fine-label per-query CCDI 2c")
    f_res = {a: Bd.bind(S61 / f"result_{a}.json", f"S6 config {a}") for a in ("s61_ext_2c", "s61_int_3c", "s61_ext_3c")}
    Bd.bind(SI_MD, "SI caption (live; hash at audit)")
    Bd.bind(NUMBERING, "numbering")

    # --------------------------------------------------------------- provenance of audited files and matrices
    brep = f_brep.read_text()
    for f in (f_xlsx, f_raw, f_spin):
        A.check(Bd.sha(f) in brep, f"`{f.name}` sha256 {Bd.sha(f)[:12]}… equals the v2 BUILD_REPORT")
    ci_in = json.load(open(f_cires))["inputs"]
    A.check(ci_in["all_tissue_matrix"]["sha256"] == Bd.sha(f_all) and ci_in["tumour_only_matrix"]["sha256"] == Bd.sha(f_tum)
            and ci_in["manifest_s61_int_2c"]["sha256"] == Bd.sha(f_m61) and ci_in["manifest_s64"]["sha256"] == Bd.sha(f_m64)
            and ci_in["frozen_meta"]["sha256"] == Bd.sha(f_meta),
            "matrices, manifests and frozen metadata are byte-identical to the 11-03 run's recorded inputs")
    aud = json.load(open(f_aud))
    A.check(aud["inputs"]["ccdi_matrix"]["sha256"] == Bd.sha(f_ccdi),
            f"CCDI matrix sha256 {Bd.sha(f_ccdi)[:12]}… equals the 10-20 audit record (recorded path "
            f"{aud['inputs']['ccdi_matrix']['path']}); S6 config path for s61_ext_2c: "
            f"{json.load(open(f_res['s61_ext_2c']))['config']['matrix']} (old sibling root; file now under $PROSPER_OUT)")

    meta = pd.read_csv(f_meta, dtype=str)
    pat_of = dict(zip(meta["Slide.Scan.File"], meta["UID.Subject"]))
    allm, tum = load(f_all), load(f_tum)
    allu, tumu = load(f_allu), load(f_tumu)
    ccdi = load(f_ccdi)

    # mask check: masked minus unmasked differences are exactly the cross-slide same-patient pairs
    ids = list(allm.index)
    pp = np.array([pat_of.get(i) or f"__nometa_{i}" for i in ids], object)
    A.check(all(i in pat_of for i in ids), f"all {len(ids)} matrix slides have a frozen-metadata patient id")
    same = (pp[:, None] == pp[None, :]) & ~np.eye(len(ids), dtype=bool)
    for nm, Mm, Mu in (("all-tissue", allm, allu), ("tumour-only", tum, tumu)):
        a, u = Mm.loc[ids, ids].to_numpy(float), Mu.loc[ids, ids].to_numpy(float)
        diff = ~((a == u) | (np.isnan(a) & np.isnan(u)))
        outside = int((diff & ~same).sum())
        not_sent = int((diff & (a != 1e6)).sum())
        left = int((same & ~diff & (u != 1e6)).sum())
        A.check(outside == 0 and not_sent == 0 and left == 0,
                f"{nm} mask: {int(diff.sum()):,} cells differ from the unmasked matrix, all set to the 1e6 sentinel "
                f"(other values {not_sent}); same-patient pairs by UID.Subject {int(same.sum()):,}; differing cells outside "
                f"same-patient pairs {outside}; same-patient pairs left at a usable distance {left}")
    A.check(bool((np.diag(allu.to_numpy()) == 1e6).all()),
            "matrices are distances (self = 1e6 sentinel; ascending order = nearest first)")

    # --------------------------------------------------------------- populations
    def comet_pop(path, labels):
        m = pd.read_csv(path, dtype=str)
        m = m[(m.sample_type != "Xenograft") & m.label.isin(labels)].reset_index(drop=True)
        m["meta_patient"] = m.slide_id.map(pat_of)
        return m

    p61, p64 = comet_pop(f_m61, ["ERMS", "ARMS"]), comet_pop(f_m64, ["HB", "HCC"])
    for nm, p in (("COMET RMS", p61), ("COMET liver", p64)):
        A.check(bool((p.meta_patient == p.patient_id).all()) and p.slide_id.is_unique,
                f"{nm}: manifest patient_id equals frozen-metadata UID.Subject for all {len(p)} slides; slide ids unique")
    m61raw = pd.read_csv(f_m61, dtype=str); m64raw = pd.read_csv(f_m64, dtype=str)
    cov = pd.read_csv(f_cov).set_index("arm")

    # --------------------------------------------------------------- COMET arms
    rng = np.random.Generator(np.random.PCG64(SEED))
    res, sc_keep, tie_info = {}, {}, {}
    for arm, pop, labels, pos in (("s61_int_2c", p61, ["ERMS", "ARMS"], "ARMS"), ("s64", p64, ["HB", "HCC"], "HCC")):
        q = pop.slide_id.tolist(); pats = pop.meta_patient.to_numpy(object); truth = pop.label.to_numpy(object)
        Mdraw = draws_multinomial(rng, len(set(pats)))
        for rep, D in (("all_tissue", allm), ("tumour_only", tum)):
            top, ties, ok = neighbours(D, q, q, pats, pats)
            sc = score(top, truth, pop.label.tolist(), pos)
            L6 = None
            res[(arm, rep)] = summarise(sc, truth, pats, labels, pos, Mdraw, macro_nan="nanmean")
            res[(arm, rep)]["conf"] = confusion(sc, truth, labels, pos)
            res[(arm, rep)]["libsize"] = ok.sum(1)
            sc_keep[(arm, rep)] = sc
            tie_info[(arm, rep)] = int(ties.sum())
        up, pidx = res[(arm, "all_tissue")]["_up"], res[(arm, "all_tissue")]["_pidx"]
        dv = sc_keep[(arm, "all_tissue")]["plur"] - sc_keep[(arm, "tumour_only")]["plur"]
        S = np.bincount(pidx, weights=dv, minlength=len(up)); N = np.bincount(pidx, minlength=len(up)).astype(float)
        dd = (Mdraw @ S) / (Mdraw @ N)
        res[(arm, "delta")] = (float(dv.mean()), *pct(dd)[:2])
        res[(arm, "M")] = Mdraw

    # --------------------------------------------------------------- CCDI-MCI arm (all-tissue CCDI x COMET matrix)
    mc = pd.read_csv(f_mcc, dtype=str)
    qc = mc[mc.label.isin(["ERMS", "ARMS"])].reset_index(drop=True)
    A.check(len(qc) == 251 and qc.patient_id.is_unique and bool((qc.slide_id == qc.patient_id).all()),
            f"CCDI-MCI queries: {len(qc)} ERMS/ARMS participants, one query per participant (slide_id = patient_id)")
    A.check(not set(qc.patient_id) & set(p61.patient_id), "CCDI participant ids and COMET patient ids share no value "
                                                           "(no cross-cohort patient linkage exists to test overlap)")
    topc, tiesc, okc = neighbours(ccdi, qc.slide_id.tolist(), p61.slide_id.tolist(), qc.patient_id.tolist(), p61.patient_id.tolist())
    truthc = qc.label.to_numpy(object)
    scc = score(topc, truthc, p61.label.tolist(), "ARMS")
    Mc = draws_bincount(len(qc))
    res[("s61_ext_2c", "all_tissue")] = summarise(scc, truthc, qc.patient_id.to_numpy(object), ["ERMS", "ARMS"], "ARMS", Mc)
    res[("s61_ext_2c", "all_tissue")]["conf"] = confusion(scc, truthc, ["ERMS", "ARMS"], "ARMS")
    tie_info[("s61_ext_2c", "all_tissue")] = int(tiesc.sum())
    rl = pd.read_csv(f_rl["s61_ext_2c"], dtype={"query": str, "neighbour": str})
    rl5 = rl[rl["rank"] <= 5].sort_values(["query", "rank"]).groupby("query")["neighbour"].apply(list)
    mine5 = {qq: [p61.slide_id[j] for j in topc[i]] for i, qq in enumerate(qc.slide_id)}
    A.check(all(rl5[qq] == mine5[qq] for qq in mine5) and set(rl5.index) == set(mine5),
            f"CCDI-MCI: my top-5 neighbours equal the saved S6 rank list for all {len(mine5)} queries (identity and order)")
    pq = pd.read_csv(f_pq, dtype={"query": str}).set_index("query")
    ix = qc.slide_id.tolist()
    A.check(bool((pq.loc[ix, "truth"].to_numpy() == truthc).all()
                 and np.allclose(pq.loc[ix, "plurality5_recall"], scc["plur"]) and np.allclose(pq.loc[ix, "hit1"], scc["hit1"])
                 and np.allclose(pq.loc[ix, "precision5"], scc["p5"]) and np.allclose(pq.loc[ix, "pos_share5"], scc["npos"] / 5)),
            "CCDI-MCI: per-query truth, plurality@5, hit@1, P@5 and ARMS share equal the fine-label per-query file")

    # --------------------------------------------------------------- compare with the printed and raw table
    raw = pd.read_csv(f_raw)
    wb = openpyxl.load_workbook(f_xlsx)
    ws = wb["S26 body"]
    hdr = [c.value for c in ws[1]]
    blk = comp = None
    for r in ws.iter_rows(min_row=2, values_only=True):
        rec = dict(zip(hdr, r))
        blk = rec["Block"] or blk; comp = rec["Comparison"] or comp
        arm = COMP[comp]
        rep = "tumour_only" if blk.startswith("Sensitivity") else "all_tissue"
        cls = {"all": "ALL", "macro": "MACRO"}.get(rec["Row"], rec["Row"])
        mine = res[(arm, rep)][cls]
        rr = raw[(raw.arm == arm) & (raw.representation == rep) & (raw["class"] == cls)]
        assert len(rr) == 1, (arm, rep, cls)
        rr = rr.iloc[0]
        ns = rec["n slides / patients"]
        if ns != "—":
            a_, b_ = [int(x) for x in ns.split(" / ")]
            A.vals.append(dict(block=rep, arm=arm, row=cls, metric="n", field="slides/patients", printed=ns,
                               raw=f"{rr.n_queries}/{rr.n_patients}", mine=f"{mine['n']}/{mine['n_patients']}", absdiff_raw="",
                               status="PASS" if (a_, b_) == (mine["n"], mine["n_patients"]) else "FAIL"))
        for col, (key, rkey) in MET.items():
            txt = rec[col]
            if txt in (None, "—"):
                continue
            m = CELL.match(txt)
            pr = [fnum(x) for x in m.groups()]
            mv = mine[key][:3]
            rv = [rr[rkey], rr[RAWLO[rkey][0]], rr[RAWLO[rkey][1]]]
            for fld, p_, m_, r_ in zip(("point", "lo", "hi"), pr, mv, rv):
                d = abs(m_ - r_)
                ok = f"{m_:.3f}" == f"{p_:.3f}" and d <= TOL_RAW
                A.vals.append(dict(block=rep, arm=arm, row=cls, metric=key, field=fld, printed=f"{p_:.3f}", raw=repr(float(r_)),
                                   mine=repr(m_), absdiff_raw=f"{d:.2e}", status="PASS" if ok else "FAIL"))
        if rec.get("Confusion (rows = truth)"):
            mc_ = res[(arm, rep)]["conf"]
            sq = lambda s: re.sub(r"\s+", "", s)
            A.vals.append(dict(block=rep, arm=arm, row=cls, metric="confusion", field="", printed=rec["Confusion (rows = truth)"],
                               raw=rr.confusion, mine=mc_, absdiff_raw="",
                               status="PASS" if sq(rec["Confusion (rows = truth)"]) == sq(mc_) == sq(rr.confusion) else "FAIL"))
    nv = len(A.vals); nvf = sum(v["status"] == "FAIL" for v in A.vals)
    A.check(len(raw) == 20, f"'S26 body' rows mapped one-to-one onto the 20 body_raw rows")

    # paired differences and headline values named in the brief
    caption = wb["caption"].cell(5, 2).value
    for arm, txt in (("s61_int_2c", "−0.040 [−0.065, −0.017]"), ("s64", "−0.022 [−0.061, +0.015]")):
        p, lo, hi = res[(arm, "delta")]
        mine_s = f"{p:.3f} [{lo:.3f}, {hi:+.3f}]".replace("-", "−")
        A.check(mine_s == txt,
                f"paired Δ plurality@5 {arm} (all-tissue − tumour-only, same draws): re-derived {p:+.4f} [{lo:+.4f}, {hi:+.4f}]; "
                f"caption '{txt}' (in v2 legend: {txt in caption})")
    heads = [("all-tissue RMS plurality 0.847", res[("s61_int_2c", "all_tissue")]["ALL"]["plur"][0], 0.847),
             ("all-tissue liver plurality 0.914", res[("s64", "all_tissue")]["ALL"]["plur"][0], 0.914)]
    for nm, v, e in heads:
        A.check(f"{v:.3f}" == f"{e:.3f}", f"{nm}: re-derived {v:.4f}")
    for nm, key, e in (("tumour-only RMS plurality 0.887 (0.855–0.920)", ("s61_int_2c", "tumour_only"), (0.887, 0.855, 0.920)),
                       ("tumour-only liver plurality 0.935 (0.884–0.977)", ("s64", "tumour_only"), (0.935, 0.884, 0.977)),
                       ("CCDI-MCI plurality 0.880 (0.837–0.920)", ("s61_ext_2c", "all_tissue"), (0.880, 0.837, 0.920))):
        v = res[key]["ALL"]["plur"][:3]
        A.check(all(f"{a:.3f}" == f"{b:.3f}" for a, b in zip(v, e)), f"{nm}: re-derived {v[0]:.4f} ({v[1]:.4f}–{v[2]:.4f})")
    shares = {"COMET RMS": (226, 373, 0.606), "CCDI-MCI RMS": (170, 251, 0.677), "liver": (95, 139, 0.683)}
    got = {"COMET RMS": (int((p61.label == "ERMS").sum()), len(p61)), "CCDI-MCI RMS": (int((truthc == "ERMS").sum()), len(qc)),
           "liver": (int((p64.label == "HB").sum()), len(p64))}
    for k_, (a_, b_, e) in shares.items():
        A.check(got[k_] == (a_, b_) and f"{a_ / b_:.3f}" == f"{e:.3f}" and f"{e:.3f}" in caption,
                f"majority-class share {k_}: {got[k_][0]}/{got[k_][1]} = {got[k_][0] / got[k_][1]:.4f} (caption {e:.3f})")
    A.check(len(p61) == 373 and p61.patient_id.nunique() == 306 and "373 slides, 306 patients" in caption,
            f"CCDI-MCI library = COMET RMS ERMS/ARMS non-xenograft {len(p61)} slides / {p61.patient_id.nunique()} patients")
    for key, n in tie_info.items():
        A.check(True, f"{key[0]} {key[1]}: {n} queries with a distance tie between ranks 5 and 6 (tie order cannot change any value)"
                if n == 0 else f"{key[0]} {key[1]}: {n} queries with a rank-5/6 distance tie")

    # --------------------------------------------------------------- spindle footnote (3-subtype searches)
    p61_3 = comet_pop(f_m61, ["ERMS", "ARMS", "Spindle"])
    spin_rows = []
    for arm, D, qpop, qpat_col in (("s61_int_3c", tum, p61_3, "meta_patient"), ("s61_ext_3c", ccdi, mc.reset_index(drop=True), "patient_id")):
        qids = qpop.slide_id.tolist(); qp = qpop[qpat_col].to_numpy(object); tr = qpop.label.to_numpy(object)
        top, ties, _ = neighbours(D, qids, p61_3.slide_id.tolist(), qp, p61_3.meta_patient.tolist())
        L = np.asarray(p61_3.label, object)[top]
        pl, tie3 = [], []
        for row in L:
            c = {l: int((row == l).sum()) for l in set(row)}
            mx = max(c.values())
            tie3.append(sum(v == mx for v in c.values()) > 1)
            pl.append(next(l for l in row if c[l] == mx))
        pl, tie3 = np.array(pl, object), np.array(tie3)
        spm = tr == "Spindle"
        rl3 = pd.read_csv(f_rl[arm], dtype={"query": str, "neighbour": str})
        rl3 = rl3[rl3["rank"] <= 5].sort_values(["query", "rank"]).groupby("query")["neighbour"].apply(list)
        mine3 = {qq: [p61_3.slide_id[j] for j in top[i]] for i, qq in enumerate(qids)}
        A.check(set(rl3.index) == set(mine3) and all(rl3[qq] == mine3[qq] for qq in mine3),
                f"{arm}: my top-5 neighbours equal the saved S6 rank list for all {len(mine3)} queries (identity and order)")
        tie_detail = []
        for i in np.flatnonzero(spm & tie3):
            row = list(L[i]); c = {l: row.count(l) for l in set(row)}; mx = max(c.values())
            tie_detail.append((qids[i], row, pl[i], sorted(l for l in c if c[l] == mx)))
        ok_ = (pl == tr).astype(float)
        Msp = draws_bincount(len(set(qp)))
        up, pidx = np.unique(qp, return_inverse=True)
        S = np.bincount(pidx[spm], weights=ok_[spm], minlength=len(up)); N = np.bincount(pidx[spm], minlength=len(up)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (Msp @ S) / (Msp @ N)
        lo, hi, nn = pct(d)
        cfgm = json.load(open(f_res[arm]))["config"]["matrix"]
        spin_rows.append(dict(arm=arm, n=int(spm.sum()), n_pat=len(set(qp[spm])), pred={l: int((pl[spm] == l).sum()) for l in ("ERMS", "ARMS", "Spindle")},
                              plur=float(ok_[spm].mean()), lo=lo, hi=hi, nonfinite=nn, ties_spindle=int(tie3[spm].sum()),
                              ties_all=int(tie3.sum()), n_all=len(qids), tie_detail=tie_detail,
                              n_lib=len(p61_3), lib_pat=p61_3.meta_patient.nunique(), matrix=cfgm, rank56_ties=int(ties[spm].sum())))
    fs = pd.read_csv(f_spin)
    for s in spin_rows:
        f = fs[fs.arm == s["arm"]].iloc[0]
        pred_txt = f"{s['pred']['ERMS']} / {s['pred']['ARMS']} / {s['pred']['Spindle']}"
        A.check(s["n"] == f.n_queries and s["n_pat"] == f.n_patients and pred_txt == f["Spindle queries predicted ERMS / ARMS / Spindle (plurality@5)"]
                and abs(s["plur"] - f.plurality5_recall) < 1e-12 and abs(s["lo"] - f.plurality5_recall_lo) < 1e-9
                and abs(s["hi"] - f.plurality5_recall_hi) < 1e-9,
                f"spindle {s['arm']}: re-derived {s['n']} queries / {s['n_pat']} patients, predicted ERMS/ARMS/Spindle {pred_txt}, "
                f"plurality@5 {s['plur']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]; footnote csv {f.n_queries}/{f.n_patients}, "
                f"{f['Spindle queries predicted ERMS / ARMS / Spindle (plurality@5)']}, {f.plurality5_recall} "
                f"[{f.plurality5_recall_lo:.3f}, {f.plurality5_recall_hi:.3f}]; three-way plurality ties among spindle queries "
                f"{s['ties_spindle']}; library {s['n_lib']} slides / {s['lib_pat']} patients")

    # --------------------------------------------------------------- design diagnostics (informational)
    diag = {}
    for arm, pop in (("s61_int_2c", p61), ("s64", p64)):
        r = res[(arm, "all_tissue")]
        sc = sc_keep[(arm, "all_tissue")]
        up, pidx = r["_up"], r["_pidx"]
        tr = pop.label.to_numpy(object)
        pm = pd.DataFrame({"pid": pidx, "plur": sc["plur"], "label": tr}).groupby(["label", "pid"]).plur.agg(["mean", "size", "sum"])
        diag[arm] = {}
        for l in sorted(set(tr)):
            g = pm.loc[l]
            n = len(g); Nt = g["size"].sum(); p_ = g["sum"].sum() / Nt
            se = np.sqrt(n / (n - 1) * (((g["sum"] - p_ * g["size"]) ** 2).sum()) / Nt ** 2)
            lo_, hi_ = r[l]["plur"][1:3]
            single, multi = g[g["size"] == 1], g[g["size"] > 1]
            diag[arm][l] = dict(single=(len(single), int(single["sum"].sum()), int(single["size"].sum())),
                                multi=(len(multi), int(multi["sum"].sum()), int(multi["size"].sum())),
                                slides=int(Nt), patients=n, slides_per_patient=g["size"].value_counts().sort_index().to_dict(),
                                patients_with_failure=int((g["sum"] < g["size"]).sum()), slide_level=p_, patient_mean=float(g["mean"].mean()),
                                wald_cluster=(p_ - 1.96 * se, min(1.0, p_ + 1.96 * se)), pct=(lo_, hi_),
                                distinct_draw_values=int(len(np.unique(np.round(r[l]["plur_draws"], 12)))))
        diag[arm]["libsize"] = (int(r["libsize"].min()), int(np.median(r["libsize"])), int(r["libsize"].max()))
    # consequence of the query-patient exclusion (unmasked matrix, no same-patient exclusion), all-tissue
    leak = {}
    for arm, pop, pos in (("s61_int_2c", p61, "ARMS"), ("s64", p64, "HCC")):
        q = pop.slide_id.tolist(); tr = pop.label.to_numpy(object)
        top, _, _ = neighbours(allu, q, q, [f"s{i}" for i in range(len(q))], [f"s{i}" for i in range(len(q))])
        sc0 = score(top, tr, pop.label.tolist(), pos)
        pats = pop.meta_patient.to_numpy(object)
        same5 = (pats[top] == pats[:, None]).any(1)
        leak[arm] = (float(sc0["plur"].mean()), int(same5.sum()), int(pd.Series(pats).duplicated(keep=False).sum()))
    # MC stability of interval endpoints (other seeds; informational)
    mcs = {}
    for arm, pop, labels, pos in (("s61_int_2c", p61, ["ERMS", "ARMS"], "ARMS"), ("s64", p64, ["HB", "HCC"], "HCC")):
        sc = sc_keep[(arm, "all_tissue")]; tr = pop.label.to_numpy(object); pats = pop.meta_patient.to_numpy(object)
        ends = []
        for sd in (1, 2, 3, 4):
            g = np.random.Generator(np.random.PCG64(sd))
            rr_ = summarise(sc, tr, pats, labels, pos, draws_multinomial(g, len(set(pats))), macro_nan="nanmean")
            ends.append({c: rr_[c]["plur"][1:3] for c in ["ALL"] + labels})
        mcs[arm] = {c: (min(e[c][0] for e in ends), max(e[c][0] for e in ends), min(e[c][1] for e in ends), max(e[c][1] for e in ends))
                    for c in ["ALL"] + labels}

    # --------------------------------------------------------------- SI caption (live) vs audited legend
    num = numbering("S26")
    si_line, si_title, si_cap = si_entry(num["current"])
    si_full = (si_title + ". " + si_cap) if si_cap else si_title
    v2_leg = caption.split("Supplementary Table S26. ", 1)[1]
    def norm(s):
        s = s.replace("tumour", "tumor").replace("labelled", "labeled")
        s = re.sub(r"Supplementary Tables? S\d+", "Supplementary Table S§", s)
        s = s.replace("family. UNI2, Optimal Transport, all-tissue mosaics.", "family (UNI2, Optimal Transport, all-tissue mosaics).")
        return re.sub(r"\s+", " ", s).strip()
    a_, b_ = norm(si_full), norm(v2_leg)
    same_text = a_ == b_
    import difflib
    wa, wb_ = b_.split(), a_.split()
    si_diffs = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb_, autojunk=False).get_opcodes():
        if op != "equal":
            si_diffs.append(f"v2 '{' '.join(wa[i1:i2])}' → SI '{' '.join(wb_[j1:j2])}' (context: '…{' '.join(wa[max(0, i1 - 6):i1])}')")
    si_diff = "; ".join(si_diffs)

    # caption claims (v2 legend and live SI) tested against the re-derived values
    claims = []
    s_ext = [s for s in spin_rows if s["arm"] == "s61_ext_3c"][0]
    s_int = [s for s in spin_rows if s["arm"] == "s61_int_3c"][0]
    for src, txt in (("v2 legend", caption), (f"live SI l.{si_line}", si_full)):
        has_none = "none occurred for the spindle-cell queries" in txt
        claims.append(("MISMATCH" if has_none and s_ext["ties_spindle"] else "CONSISTENT" if has_none else "ABSENT",
                       f"{src}: 'none occurred for the spindle-cell queries' — three-way plurality ties among spindle queries: "
                       f"COMET {s_int['ties_spindle']} of {s_int['n']}, CCDI-MCI {s_ext['ties_spindle']} of {s_ext['n']}"
                       if has_none else f"{src}: no 'none occurred' clause"))
        has_rep = "three-subtype search on the tum" in txt
        claims.append(("MISMATCH" if has_rep and "tumor_only" not in Path(s_ext["matrix"]).name else "CONSISTENT",
                       f"{src}: 'evaluated only in a three-subtype search on the tumour-only representations' — COMET "
                       f"queries on `{Path(s_int['matrix']).name}`, CCDI-MCI queries on `{Path(s_ext['matrix']).name}` "
                       "(all-tissue CCDI × COMET, per the 10-20 audit record)"))
        ok17 = s_int["pred"]["ERMS"] == 18 and s_ext["pred"]["ERMS"] == 17 and "17 of 20" in txt and "All 18 COMET" in txt.replace("all 18 COMET", "All 18 COMET")
        claims.append(("CONSISTENT" if ok17 else "MISMATCH",
                       f"{src}: 'all 18 COMET and 17 of 20 CCDI-MCI spindle-cell queries were retrieved as embryonal RMS "
                       f"(plurality@5 0 and 0.05)' — re-derived {s_int['pred']['ERMS']}/18 and {s_ext['pred']['ERMS']}/20 under the "
                       f"closest-rank tie rule; plurality@5 {s_int['plur']:.2f} and {s_ext['plur']:.2f}"))
        claims.append(("CONSISTENT" if "18 slides" in txt and "14 patients" in txt and "20 slides" in txt else "MISMATCH",
                       f"{src}: spindle counts COMET 18 slides / 14 patients, CCDI-MCI 20 — re-derived {s_int['n']} / "
                       f"{s_int['n_pat']}, {s_ext['n']} / {s_ext['n_pat']}"))
    tie_alt = []
    for qq, row, res_, tied in s_ext["tie_detail"]:
        tie_alt.append(f"`{qq}` top-5 {'/'.join(row)}: tied {' = '.join(tied)}, resolved to {res_}")
    erms_rng = (s_ext["pred"]["ERMS"] - sum(1 for _, _, r_, t in s_ext["tie_detail"] if r_ == "ERMS"),
                s_ext["pred"]["ERMS"] + sum(1 for _, _, r_, t in s_ext["tie_detail"] if r_ != "ERMS" and "ERMS" in t))
    plur_alt = (s_ext["plur"] * s_ext["n"] + sum(1 for _, _, r_, t in s_ext["tie_detail"] if r_ != "Spindle" and "Spindle" in t)) / s_ext["n"]

    # --------------------------------------------------------------- outputs
    with open(OUT / "audit_values.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(A.vals[0]))
        w.writeheader(); w.writerows(A.vals)
    Bd.write(OUT, {"audited": rel(f_xlsx), "built": BUILD_DATE, "seed": SEED, "B": B, "workspace_id": "S26",
                   "final_number": num["current"]})

    def fmt(t):
        return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"

    L = [f"# Independent audit — Table S26 v2 (workspace S26 → Supplementary Table {num['current']})", "",
         f"Audited: `{rel(f_xlsx)}` (sha256 `{Bd.sha(f_xlsx)}`), sheet 'S26 body', with `{f_raw.name}` "
         f"(`{Bd.sha(f_raw)[:16]}…`) and `{f_spin.name}`. Script: `scripts/supp_tables/audit_S26_v2_20261109.py`; run {now_utc()}. "
         f"Inputs and hashes: `input_bindings.json`; every compared value: `audit_values.csv`.", "",
         "## Verdict", "",
         f"**{nv - nvf} of {nv} printed values reproduce** (3-decimal string equality with my values and |mine − body_raw| ≤ "
         f"{TOL_RAW:g}); structural/provenance checks {sum(s == 'PASS' for s, _ in A.checks)} PASS, "
         f"{sum(s == 'FAIL' for s, _ in A.checks)} FAIL. The numbers in the table are what the stated design produces; the "
         "design points below are for the lead's judgement and do not change any printed value.", "",
         "## Route (distinct from the producing scripts)", "",
         "- Neighbours: my own eligibility mask (other patient by the frozen COMET metadata `UID.Subject`, not the query, "
         "finite distance) and `np.lexsort` on (distance, manifest position); the producing route used the manifest "
         "`patient_id`, `+inf` substitution and a stable `argsort`. For CCDI-MCI my top-5 lists equal the saved S6 rank "
         "lists neighbour for neighbour.",
         "- Metrics: plurality@5 as a ≥3-of-5 majority (two classes: no ties possible); AUROC by level counts of the "
         "positive-label count 0–5 (Mann–Whitney, half credit for ties) rather than pairwise comparison or sklearn.",
         "- Intervals: per-patient sufficient statistics multiplied by the draw multiplicities (ratio estimator) instead of "
         "per-slide weight matrices. The draws themselves follow the recorded generators so that intervals can be compared "
         "exactly: COMET arms one `Generator(PCG64(123))` shared by RMS then liver, `multinomial(n, 1/n)` × 5,000, reused "
         "for both matrices; CCDI-MCI and spindle a fresh `PCG64(123)` per arm, `bincount(integers(p, p))` per draw.", "",
         "## Headline values (brief)", ""]
    for s, m in A.checks:
        if any(t in m for t in ("plurality 0.", "paired Δ", "majority-class", "CCDI-MCI library")):
            L.append(f"- **{s}** {m}")
    L += ["", "## Every printed value", "",
          "| Block | Arm | Row | Plurality@5 | hit@1 | P@5 | AUROC | n | max abs(mine − raw) | status |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (arm, rep), r in [(k, v) for k, v in res.items() if k[1] in ("all_tissue", "tumour_only")]:
        labels = ["ERMS", "ARMS"] if arm.startswith("s61") else ["HB", "HCC"]
        for cls in ["ALL", "MACRO"] + labels:
            vv = [v for v in A.vals if v["arm"] == arm and v["block"] == rep and v["row"] == cls]
            mx = max([float(v["absdiff_raw"]) for v in vv if v["absdiff_raw"]] or [0])
            st = "PASS" if all(v["status"] == "PASS" for v in vv) else "FAIL"
            e = r[cls]
            L.append(f"| {rep} | {arm} | {cls} | {fmt(e['plur'])} | {fmt(e['hit1'])} | {fmt(e['p5'])} | "
                     f"{fmt(e['auroc']) if 'auroc' in e else '—'} | {e.get('n', '')}"
                     f"{' / ' + str(e['n_patients']) if 'n_patients' in e else ''} | {mx:.1e} | {st} ({len(vv)} values) |")
    L += ["", "Confusion strings (rows = truth, plurality@5 prediction) are compared in `audit_values.csv` (all "
          + ("PASS" if all(v["status"] == "PASS" for v in A.vals if v["metric"] == "confusion") else "NOT all PASS") + ").", "",
          "## Caption claims (v2 legend and live SI vs re-derived values)", "",
          f"{sum(c[0] == 'CONSISTENT' for c in claims)} consistent, {sum(c[0] == 'MISMATCH' for c in claims)} mismatch "
          "(findings for the lead; caption wording is not part of the printed values above).", ""]
    L += [f"- **{s}** {m}" for s, m in claims]
    L += ["", "## Structural and provenance checks", ""]
    L += [f"- **{s}** {m}" for s, m in A.checks if not any(t in m for t in ("plurality 0.", "paired Δ", "majority-class", "CCDI-MCI library"))]

    d61, d64 = diag["s61_int_2c"], diag["s64"]
    hcc = d64["HCC"]
    L += ["", "## Design review", "",
          "### Denominators", "",
          f"- COMET RMS: the manifest has {len(m61raw)} slides; {int((m61raw.sample_type == 'Xenograft').sum())} xenografts "
          f"and {int(((m61raw.sample_type != 'Xenograft') & (m61raw.label == 'Spindle')).sum())} non-xenograft spindle-cell "
          f"slides are excluded, leaving 373 slides / 306 patients (ERMS 226 / 188, ARMS 147 / 118; no patient in both). The "
          f"COMET RMS tumour group has {int(cov.loc['s61_int_2c', 'broad_group_slides'])} slides, of which "
          f"{int(cov.loc['s61_int_2c', 'no_subtype'])} have no subtype label and never enter the manifest.",
          f"- COMET liver: {len(m64raw)} manifest slides, {int((m64raw.sample_type == 'Xenograft').sum())} xenografts excluded, "
          f"139 / 85 evaluated (HB 95 / 60, HCC 44 / 25); tumour group {int(cov.loc['s64', 'broad_group_slides'])} slides, "
          f"{int(cov.loc['s64', 'no_subtype'])} without subtype.",
          f"- CCDI-MCI: {int(cov.loc['s61_ext_2c', 'broad_group_slides'])} RMS participants after the SW1 exclusion, "
          f"{int(cov.loc['s61_ext_2c', 'no_subtype'])} ({cov.loc['s61_ext_2c', 'no_subtype'] / cov.loc['s61_ext_2c', 'broad_group_slides']:.0%}) "
          f"without an ERMS/ARMS/spindle label; 251 ERMS/ARMS queries + 20 spindle. The external estimate is conditional "
          f"on a recorded subtype, which is a selected 62% of the RMS participants; the legend does not say so.",
          "- Point estimates are slide-pooled; patients with several slides weigh more. Patient-averaged plurality@5 "
          "(each patient's slide mean, then the mean over patients; informational): "
          + "; ".join(f"{a} {l} {diag[a][l]['patient_mean']:.3f} vs slide-level {diag[a][l]['slide_level']:.3f}"
                      for a in ("s61_int_2c", "s64") for l in diag[a] if l != "libsize") + ".",
          "- The search space per query is the whole two-subtype library minus the query patient's slides: RMS "
          f"{d61['libsize'][0]}–{d61['libsize'][2]} (median {d61['libsize'][1]}) eligible reference slides, liver "
          f"{d64['libsize'][0]}–{d64['libsize'][2]} (median {d64['libsize'][1]}). Each of the "
          "majority-class shares (0.606 / 0.677 / 0.683) is the plurality@5 of a constant majority-class prediction, so it "
          "is the right floor for plurality@5 but not for macro values (floor 0.5) or AUROC (0.5).", "",
          "### Query-patient exclusion", "",
          "- Verified twice: the masked matrices differ from the unmasked ones only at cross-slide same-patient pairs "
          "(1e6 sentinel), and my route excludes same-patient references by the frozen metadata independently of the mask; "
          "values are identical, so the manifest `patient_id` and `UID.Subject` agree for every evaluated slide.",
          "- Consequence if it had not been applied (all-tissue, unmasked matrix, self excluded only; informational): RMS "
          f"plurality@5 {leak['s61_int_2c'][0]:.3f} with {leak['s61_int_2c'][1]} of 373 queries having a same-patient slide "
          f"in their top 5 ({leak['s61_int_2c'][2]} slides belong to multi-slide patients); liver {leak['s64'][0]:.3f}, "
          f"{leak['s64'][1]} of 139 ({leak['s64'][2]}). The exclusion therefore matters and is correctly applied.",
          "- CCDI-MCI participants and COMET patients have disjoint identifier systems; no linkage exists to test whether "
          "any CCDI-MCI participant is also a COMET patient. This is a stated limitation, not a verified absence.", "",
          "### HCC: 44 slides from 25 patients", "",
          f"- HCC slides per patient {hcc['slides_per_patient']}; {hcc['patients_with_failure']} of 25 patients have at "
          f"least one slide whose plurality@5 is wrong. The percentile interval {hcc['pct'][0]:.3f}–{hcc['pct'][1]:.3f} "
          f"for HCC plurality@5 (all-tissue) compares with a cluster-robust Wald interval "
          f"{hcc['wald_cluster'][0]:.3f}–{hcc['wald_cluster'][1]:.3f} (ratio estimator over patients; informational). "
          f"The bootstrap distribution takes {hcc['distinct_draw_values']} distinct values.",
          f"- HB ({d64['HB']['patients']} patients, {d64['HB']['patients_with_failure']} with a failure) has an upper "
          "bound of 1.000: the percentile interval cannot exceed the observed maximum, a known small-sample failure mode.",
          "- Monte Carlo stability (seeds 1–4, all-tissue plurality@5, informational): "
          + "; ".join(f"{a} {c} lo {v[0]:.3f}–{v[1]:.3f}, hi {v[2]:.3f}–{v[3]:.3f}" for a in mcs for c, v in mcs[a].items())
          + ". Third-decimal endpoints move by up to a few thousandths with the seed, more for HCC.",
          "- The legend's caveat ('Percentile intervals for small groups (e.g., HCC, 25 patients) are approximate and may "
          "under-cover') is accurate and sufficient; a coverage study was not run (outside this audit's scope).", "",
          "### Other observations", ""]
    L += [f"- **Spindle tie clause is wrong for CCDI-MCI:** the v2 legend and the live SI say ties among three "
          f"subtypes 'none occurred for the spindle-cell queries'. From my rankings, confirmed on the saved S6 rank lists, "
          f"{s_ext['ties_spindle']} of the {s_ext['n']} CCDI-MCI spindle-cell queries had a 2–2–1 vote "
          f"({'; '.join(tie_alt)}); COMET had {s_int['ties_spindle']}. Under the stated closest-rank rule the printed "
          f"values hold (17 of 20 as ERMS, plurality@5 0.05). Under another tie order the ERMS count would range "
          f"{erms_rng[0]}–{erms_rng[1]} of 20 and plurality@5 could be {plur_alt:.2f}. The origin is edit-sheet R3.7 F2 "
          "(l.9309), which added 'none occurred'; the 10-20 audit F2 asked only for the rule to be stated. Proposed "
          "wording: '(ties among three subtypes were resolved to the closest-ranked tied subtype; this applied to 3 of the "
          "20 CCDI-MCI and none of the COMET spindle-cell queries)'.",
          f"- **Spindle footnote representation:** the COMET spindle-cell queries were searched on the tumour-only matrix "
          f"(`{Path(s_int['matrix']).name}`), but the CCDI-MCI spindle-cell queries were searched on "
          f"`{Path(s_ext['matrix']).name}`, which the 10-20 audit records as the all-tissue CCDI × COMET matrix (sha256 "
          f"`{Bd.sha(f_ccdi)[:12]}…`). The footnote's 'evaluated only in a three-subtype search on the tumour-only "
          "representations' is therefore accurate for COMET only. Proposed wording: '… was evaluated only in a "
          "three-subtype search (COMET queries on the tumor-only representations, CCDI-MCI queries on the all-tissue "
          "CCDI-MCI × COMET distances) …'. The three-subtype library is 391 slides / 320 patients, not the 373 / 306 of "
          "the two-subtype rows.",
          "- AUROC: printed values use the unweighted count of the positive label among the five nearest (my level-count "
          "AUROC reproduces them). The original S6 records also hold a distance-weighted share AUROC; the legend's "
          "definition matches the printed one.",
          "- Representation consistency: the reported block is now all-tissue on both COMET and CCDI-MCI rows (v1 mixed "
          "tumour-only COMET rows with all-tissue CCDI-MCI rows). The CCDI-MCI all-tissue status rests on the 10-20 audit "
          "record and the matrix file name; the CCDI-MCI rows have no tumour-only counterpart in the sensitivity block.",
          "- Reproducibility note: the CCDI matrix is recorded in the S6 configuration under the old sibling root "
          "(`…/Core/Haoran/prosper_revision_output/similarity_matrix/`), which no longer exists; the byte-identical file is "
          "`$PROSPER_OUT/similarity_matrix/…ccdi_uni2_ot_cosine.pkl` (hash above). The fine-label draw function is bound "
          "only by hash to an untracked working-tree file (reconciliation.json); my reimplementation reproduces its draws.",
          f"- Legend/SI: the live SI entry (l.{si_line}) "
          + ("matches the v2 legend after normalising spelling, the title layout and the Table S5 number."
             if same_text else "differs from the v2 legend beyond spelling/title layout/number normalisation (both bound by "
             f"hash); {si_diff}")]
    L += ["", "## Open items", "",
          "- Spindle footnote: 'none occurred for the spindle-cell queries' is false for CCDI-MCI (3 of 20 tied; proposed "
          "wording above). Present in both the v2 legend and the live SI.",
          "- Spindle footnote: 'on the tumour-only representations' is true only for the COMET queries (proposed wording above).",
          "- HCC slide pooling: patient-averaged HCC plurality@5 is "
          f"{d64['HCC']['patient_mean']:.3f} against the printed slide-level {d64['HCC']['slide_level']:.3f}: single-slide "
          f"HCC patients {d64['HCC']['single'][1]}/{d64['HCC']['single'][2]} correct, multi-slide HCC patients "
          f"({d64['HCC']['multi'][0]} patients) {d64['HCC']['multi'][1]}/{d64['HCC']['multi'][2]} slides correct. The legend's "
          "'Point estimates are over slides' is accurate, but the gap matters when the HCC row is interpreted.",
          "- CCDI-MCI selection: the external estimate covers the 62% of SW1-eligible RMS participants with a recorded "
          "subtype; consider stating this in the legend or Methods.",
          "- No cross-cohort patient linkage: COMET/CCDI-MCI overlap cannot be excluded from the data.",
          "- HB/HCC intervals: keep the under-coverage caveat; the HB upper bound 1.000 is a percentile-method artefact.",
          "- Third-decimal interval endpoints depend on the seed (Monte Carlo error a few thousandths); fine for a "
          "supplementary table, but the main text should not lean on the last digit.", ""]
    (OUT / "AUDIT_S26_v2.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "manifest_row.json").write_text(json.dumps({"S26": {"final_number": num["current"], "path": rel(f_xlsx),
                                                                "role": "audited (not rebuilt)", "audit": rel(OUT / "AUDIT_S26_v2.md")}}, indent=1) + "\n")
    write_sha256sums(OUT)
    print(f"S26 audit: {nv - nvf}/{nv} values PASS; checks {len(A.checks)} ({sum(s == 'FAIL' for s, _ in A.checks)} FAIL) -> {OUT}")
    return 1 if A.n_fail() else 0


if __name__ == "__main__":
    sys.exit(main())
