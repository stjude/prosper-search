#!/usr/bin/env python3
"""R3.3 / R4.a cohort accounting (batch 3, E4): authoritative cohort table (Table S14 replacement),
CONSORT-style cohort flow figure, and a regenerated Table S15 (CCDI diagnosis-string mapping).

Every count is recomputed from a named source file whose sha256 is recorded, or is carried as a tagged
constant from an accepted record (T7 = R1.3-T7 [internal record]; T1 = R1.2-T1 [internal record]; MS = manuscript line).
Every recomputed count is asserted against the value previously recorded in the A-stream crosswalk
(`a_counts.json`) / T7 / T1; any mismatch aborts before writing.

Sources
  PM  COMET patient map            final_scan_id_disease_map_with_sample_id_patient_id.csv
  RM  COMET rich metadata          final_scan_id_disease_map_rich_data.csv (Sample.Type, Discrepant)
  SM  STV label manifest           data/COMET_validation_set_deid_20260902.csv (782 rows)
  ZT  STV zero-tile export scan    $PROSPER_OUT/qc_stv/zero_tile_scan.csv
  C12 STV accepted per-query       $PROSPER_OUT/r1_3_external_correction_20260930/stv_main_696_whole_slide_ot_per_query.csv
  KE  KidRare eligibility          $PROSPER_OUT/r1_3_external_correction_20260930/kidrare_source_2331_eligibility.csv
  SY  R1.3 synthesis coverage      $PROSPER_OUT/r1_3_external_synthesis_20260930/{coverage_and_hit_yield,separate_coverage_losses}.csv
  CM  CCDI 4,160-slide manifest    $PROSPER_OUT/r1_2_scope_correction_20260930/manifest_original_all_4160_scope_review_20260930.csv
  CP  CCDI historical predictions  comet-testsets/output/COG_CCDI_MCI_ALL/search_results/uni2/predicted_majority_disease_top_10.csv
  V03 CCDI->COMET mapping v03      comet-testsets/data/input/ccdi_comet_mapping_v03.csv (Fig. 4C notebook input)
  CMD CCDI participant metadata    comet-testsets/data/input/CCDI_metadata_all.csv (Fig. 4C notebook input; 867 reproduction only)
  S15 submitted Table S15          docs/prosper_submission/supplementary/supplementary_tables/Table_S15_CCDI_MCI_mapping.csv
  HX  historical UNI2 outputs      comet-testsets/output/<collection>/search_results/uni2/predicted_majority_disease_top_5.csv

No model, feature, distance matrix or image is read. CPU-seconds.
"""
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/PROSPER_WORKSPACE"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSETS = f"{ROOT}/comet-testsets"
FSL = f"{ROOT}/comet-new-scans/data/final_slide_list"

# Tagged constants not recomputable from files read here (source in tag).
T7 = dict(kidrare_unreadable=1, kidrare_no_tissue=8, stv_ret_to_rb_slides=32, stv_ret_to_rb_patients=19)
MS = dict(ccdi_downloaded=4307, comet_final_slides_line="L155")
# One slide per participant for the 765 in-vocabulary CCDI set (R1.2-T1); the rule itself is stated from T1.
T1 = dict(one_per_participant_rule="one slide per participant retained in the 765 in-vocabulary set (R1.2-T1)")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def expect(name, got, want, log):
    log[name] = dict(got=got, expected=want)
    if got != want:
        raise RuntimeError(f"ASSERTION {name}: got {got!r}, expected {want!r}")


def comet(paths, chk):
    a = pd.read_csv(paths["PM"], dtype=str)
    a["sid"] = a["Slide.Scan.File"].astype(str)
    r = pd.read_csv(paths["RM"], dtype=str, low_memory=False)
    dup = r[r.slide_id.duplicated(keep=False)]
    disc_conflict = sorted(dup.groupby("slide_id").Discrepant.nunique().loc[lambda x: x > 1].index.astype(str))
    expect("comet_rm_discrepant_conflicting_duplicates", len(disc_conflict), 5, chk)
    r1 = r.drop_duplicates("slide_id")[["slide_id", "Sample.Type", "Discrepant"]]
    m = a.merge(r1, left_on="sid", right_on="slide_id", how="left")
    groups14 = sorted(set(a["Tumor.group"]) - {"MISC", "NORMAL"})
    expect("comet_n_groups14", len(groups14), 14, chk)

    def pop(d):
        return dict(slides=int(len(d)), patients=int(d["UID.Subject"].nunique()))

    def pairs(d):
        n = d.groupby("UID.Subject").size().to_numpy()
        return int((n * (n - 1) // 2).sum())

    def xeno(d):
        x = d[d["Sample.Type"].str.contains("eno", case=False, na=False)]
        return dict(slides=int(len(x)), patients=int(x["UID.Subject"].nunique()))

    full = pop(a); expect("comet_2921", full, dict(slides=2921, patients=1993), chk)
    misc = a[a["Tumor.group"] == "MISC"]; expect("comet_misc", len(misc), 143, chk)
    at = m[m["Tumor.group"] != "MISC"]; expect("comet_2778", pop(at), dict(slides=2778, patients=1876), chk)
    expect("comet_within_patient_pairs_2778", pairs(at), 1898, chk)
    normal = at[at["Tumor.group"] == "NORMAL"]; expect("comet_normal_174", len(normal), 174, chk)
    tum = at[at["Tumor.group"] != "NORMAL"]; expect("comet_2604", pop(tum), dict(slides=2604, patients=1800), chk)
    expect("comet_2603_refs", int((tum["sid"] != "16738").sum()), 2603, chk)
    # 2,766 = tumour-only distance-matrix row count ([analysis plan]: 2,778 minus 12 slides — 11 NORMAL + RB 16738 — with no finite
    # tumour-only distance). Constant from [analysis plan], not re-derived here (matrix not read); never a CI population.
    expect("comet_xeno_2921", xeno(m), dict(slides=250, patients=198), chk)
    expect("comet_xeno_2778", xeno(at), dict(slides=245, patients=196), chk)
    expect("comet_unjoined_2778", int(at["slide_id"].isna().sum()), 33, chk)
    expect("comet_unjoined_2921", int(m["slide_id"].isna().sum()), 40, chk)
    st_2778 = at["Sample.Type"].fillna("not joined").value_counts().to_dict()
    # Discrepant flag: newly computed, NOT previously accepted — reported as a descriptive observation only.
    disc_2778 = at["Discrepant"].fillna("not joined").value_counts().to_dict()
    per_group = at.groupby("Tumor.group").agg(slides=("sid", "size"), patients=("UID.Subject", "nunique")).reset_index()
    return dict(full=full, misc=int(len(misc)), all_tissue=pop(at), within_patient_pairs=pairs(at), normal=int(len(normal)),
                tumour_only=pop(tum), refs_nonempty=2603, zero_tumour_tile_slide="16738 (RB)", matrix_rows_2766="2,766 tumour-only matrix rows [constant; plan0002] = 2,778 − 12 slides (11 NORMAL + RB 16738) without finite tumour-only distance",
                xenograft_library=xeno(m), xenograft_2778=xeno(at), sample_type_2778=st_2778, discrepant_2778_descriptive=dict(counts=disc_2778, note="descriptive only; NOT the pre-library discrepant-exclusion count; 5 duplicated slide_ids carry conflicting N/Y so the split is order-dependent (±5)", conflicting_duplicate_slide_ids=disc_conflict),
                groups14=groups14, per_group_2778=per_group.to_dict("records"))


def stv(paths, chk):
    s = pd.read_csv(paths["SM"], dtype=str)
    s["stem"] = s.file_name.str.replace(r"\.[^.]+$", "", regex=True)
    expect("stv_782", dict(slides=len(s), patients=int(s["de-id_uid"].nunique())), dict(slides=782, patients=394), chk)
    z = pd.read_csv(paths["ZT"], dtype=str); z["n_zero_tiles"] = z.n_zero_tiles.astype(int)
    trunc = set(z[z.n_zero_tiles > 0].slide_id)
    t = s[s.stem.isin(trunc)]; expect("stv_truncated_6", len(t), 6, chk)
    rest = s[~s.stem.isin(trunc)]
    expect("stv_776", dict(slides=len(rest), patients=int(rest["de-id_uid"].nunique())), dict(slides=776, patients=393), chk)
    q = pd.read_csv(paths["C12"], dtype=str)
    expect("stv_696_rows", len(q), 696, chk)
    expect("stv_696_all_in_manifest", int(q["query"].isin(rest.stem).sum()), 696, chk)
    main = rest[rest.stem.isin(q["query"])]
    expect("stv_696", dict(slides=len(main), patients=int(main["de-id_uid"].nunique())), dict(slides=696, patients=350), chk)
    un = rest[~rest.stem.isin(q["query"])]
    expect("stv_unsupported_80_43", dict(slides=len(un), patients=int(un["de-id_uid"].nunique())), dict(slides=80, patients=43), chk)
    lab = q.true_label.value_counts().to_dict()
    expect("stv_normal_8", int(lab.get("NORMAL", 0)), 8, chk)
    both = main[~main.stem.isin(q[q.true_label == "NORMAL"]["query"])]
    expect("stv_both_sided_688", len(both), 688, chk)
    return dict(labelled=dict(slides=782, patients=394), truncated_exports=dict(slides=6, dx=sorted(t.dx.tolist())),
                after_export_exclusion=dict(slides=776, patients=393),
                unsupported_dx=dict(slides=80, patients=43, dx=un.dx.value_counts().to_dict()),
                main=dict(slides=696, patients=350, labels=lab), both_sided=dict(slides=688, patients=int(both["de-id_uid"].nunique()),
                excluded="8 NORMAL queries (no tumour tiles by definition)"))


def kidrare(paths, chk):
    e = pd.read_csv(paths["KE"], dtype=str)
    expect("kidrare_2331", len(e), 2331, chk)
    expect("kidrare_processed_2322", int((e.processed == "True").sum()), 2322, chk)
    am = e[e.amended_eligible == "True"]; expect("kidrare_1581", len(am), 1581, chk)
    grp = am.corrected_tumor_group.value_counts().to_dict()
    expect("kidrare_groups", grp, {"LIVER": 815, "NB": 303, "WT": 413, "NORMAL": 50}, chk)
    notel = e[(e.processed == "True") & (e.amended_eligible != "True")]
    excl = notel.corrected_tumor_group.value_counts().to_dict()
    expect("kidrare_excluded_741", excl, {"MB": 482, "WT_UNVERIFIED": 259}, chk)
    expect("kidrare_pathpt_357", int((e.pathpt_only_new_label == "True").sum()), 357, chk)
    expect("kidrare_legacy_1227", int((e.legacy_eligible == "True").sum()), 1227, chk)
    c = pd.read_csv(paths["SY_cov"])
    k = c[(c.cohort == "kidrare") & (c.metric == "coverage") & (c.tumor_group == "micro")].set_index("population")
    expect("kidrare_query_only_returned", dict(n=int(k.loc["expanded_query_only_1581", "n"]), returned=int(k.loc["expanded_query_only_1581", "n_returned"])), dict(n=1581, returned=1580), chk)
    expect("kidrare_both_sided_returned", dict(n=int(k.loc["expanded_both_sided_1531", "n"]), returned=int(k.loc["expanded_both_sided_1531", "n_returned"])), dict(n=1531, returned=1530), chk)
    loss = pd.read_csv(paths["SY_loss"])
    lk = loss[(loss.cohort == "kidrare") & (loss.population == "expanded_both_sided_1531")]
    expect("kidrare_coverage_loss_query", lk["query"].tolist(), ["NULL-20250103-133602"], chk)
    return dict(source=2331, unprocessed=dict(n=9, unreadable_T7=T7["kidrare_unreadable"], no_tissue_T7=T7["kidrare_no_tissue"]), processed=2322,
                excluded_unsupported=excl, amended=dict(slides=1581, groups=grp, pathpt_integrated_labels=357, legacy_subset=1227),
                query_only=dict(target=1581, returned=1580), both_sided=dict(target=1531, returned=1530, excluded_normal=50,
                lost_query=dict(id="NULL-20250103-133602", group=str(lk.tumor_group.iloc[0]), reason="no tumour tiles retained after filtering (confirmed empty)")),
                patients="not available (provider release carries no patient identifier)")


def ccdi(paths, chk):
    m = pd.read_csv(paths["CM"], dtype=str)
    expect("ccdi_4160", len(m), 4160, chk)
    p = pd.read_csv(paths["CP"], dtype=str)
    expect("ccdi_4154_pred_rows", len(p), 4154, chk)
    expect("ccdi_pred_subset", int(p["Scan ID"].isin(m.slide_id).sum()), 4154, chk)
    m["evaluated"] = m.slide_id.isin(p["Scan ID"])
    miss = m[~m.evaluated].population.value_counts().to_dict()
    expect("ccdi_6_missing", miss, {"NO_METADATA": 5, "OOS": 1}, chk)
    pop4160 = m.population.value_counts().to_dict()
    expect("ccdi_pop_4160", pop4160, {"OOS": 2953, "INSCOPE": 765, "NO_METADATA": 236, "UNMAPPED": 138, "MISC": 68}, chk)
    pop4154 = m[m.evaluated].population.value_counts().to_dict()
    expect("ccdi_pop_4154", pop4154, {"OOS": 2952, "INSCOPE": 765, "NO_METADATA": 231, "UNMAPPED": 138, "MISC": 68}, chk)
    expect("ccdi_linked_3924", int((m.population != "NO_METADATA").sum()), 3924, chk)
    ins = m[m.population == "INSCOPE"]
    expect("ccdi_765_participants", int(ins.participant.nunique()), 765, chk)
    prep = ins.preparation_status.value_counts().to_dict()
    expect("ccdi_prep", prep, {"VERIFIED_FFPE": 546, "FROZEN_OCT": 209, "MIXED": 6, "NONE_UNKNOWN": 4}, chk)
    ffpe = ins[ins.original_ffpe_546 == "True"]; expect("ccdi_546", len(ffpe), 546, chk)
    sc = ffpe.scope_status.value_counts().to_dict()
    expect("ccdi_546_scope", sc, {"CLEAR": 524, "UNRESOLVED": 18, "EXCLUDE_KNOWN_FOXR2": 4}, chk)
    unres = ffpe[ffpe.scope_status == "UNRESOLVED"]
    unres_reasons = unres.scope_reason.value_counts().to_dict(); expect("ccdi_unresolved_18_slides", int(unres_reasons and sum(unres_reasons.values())), 18, chk)
    unres_strings = int(unres.ccdi_diagnosis.nunique()); expect("ccdi_unresolved_11_strings", unres_strings, 11, chk)
    nomd = m[m.population == "NO_METADATA"]
    reexport = nomd[nomd.participant.isin(ins.participant)]
    expect("ccdi_reexports_26_from_15", dict(slides=len(reexport), participants=int(reexport.participant.nunique())), dict(slides=26, participants=15), chk)
    fin = m[m.ffpe_scope_clear == "True"]; expect("ccdi_524", len(fin), 524, chk)
    expect("ccdi_524_participants", int(fin.participant.nunique()), 524, chk)
    groups = fin.tumor_group.value_counts().to_dict()
    expect("ccdi_524_groups_n", len(groups), 12, chk)
    scope765 = ins.scope_status.value_counts().to_dict()
    # published Fig. 4C denominator 867: notebook route (participant-level CCDI_metadata_all join,
    # v03 ccdi_diagnosis -> comet_diagnosis, keep every label except Normal/Unmatched), decomposed on CM
    md = pd.read_csv(paths["CMD"], dtype=str)
    expect("ccdi_md_participants_unique", int(md["Participant ID"].duplicated().sum()), 0, chk)
    v3 = pd.read_csv(paths["V03"], dtype=str, encoding="utf-8-sig")
    q = p[["Scan ID"]].copy(); q["participant"] = q["Scan ID"].str.split("_").str[0]
    q = q.merge(md[["Participant ID", "Diagnosis"]], left_on="participant", right_on="Participant ID", how="left")
    q["comet_dx"] = q.Diagnosis.map(dict(zip(v3.ccdi_diagnosis, v3.comet_diagnosis)))
    f = q[q.comet_dx.notna() & ~q.comet_dx.isin(["Normal", "Unmatched"])]
    expect("ccdi_867_reproduced", int(len(f)), 867, chk); expect("ccdi_867_unique_scans", int(f["Scan ID"].duplicated().sum()), 0, chk)
    expect("ccdi_867_entities", int(f.comet_dx.nunique()), 20, chk)
    fj = f.merge(m[["slide_id", "population", "suffixed"]], left_on="Scan ID", right_on="slide_id", how="left")
    dec = fj.population.value_counts().to_dict()
    expect("ccdi_867_decomposition", dec, {"INSCOPE": 765, "MISC": 68, "NO_METADATA": 34}, chk)
    suf = fj[fj.population == "NO_METADATA"]
    expect("ccdi_867_reexports_all_suffixed", int((suf.suffixed == "True").sum()), 34, chk)
    n_suf_parents = int(suf.participant.nunique()); expect("ccdi_867_reexport_parents", n_suf_parents, 21, chk)
    fig4c_867 = dict(n=867, entities=20, route="Fig. 4C notebook: v03 mapping; participant-level CCDI_metadata_all join; all non-Normal/Unmatched comet_diagnosis labels",
                     decomposition_on_CM=dec, suffixed_reexport_parent_participants=n_suf_parents, per_entity=f.comet_dx.value_counts().to_dict(),
                     note="867 -> 765 = v03 comet_tumor_group rule dropping 68 MISC-group entity slides + slide-level manifest join removing 34 suffixed re-exports (21 participants); the published 867 counted more than one slide for some participants")
    return dict(fig4c_867=fig4c_867, downloaded_MS=MS["ccdi_downloaded"], staged_with_features=4160, evaluated_nonempty=4154, missing_6=miss,
                populations_4160=pop4160, populations_4154=pop4154, linked_to_participant=3924,
                in_vocabulary=dict(slides=765, participants=765, note="765 slides = 765 participants is an observed outcome in CM; 26 suffixed re-exports from 15 of these participants lacked metadata and sit in NO_METADATA", T1_rule=T1["one_per_participant_rule"]),
                unresolved_546_reasons=unres_reasons, unresolved_546_n_strings=unres_strings, reexports_no_metadata=dict(slides=26, participants=15), preparation_765=prep,
                scope_765=scope765, ffpe_546=546, scope_546=sc, final=dict(slides=524, participants=524, groups=groups), m=m)


def historical(paths, chk):
    rows = []
    sub = {"NB_MycN_TMA": ("NB MYCN TMA", 102, "TMA cores, H&E + MYCN IHC", "NB", "Not applicable (NB only)", "St. Jude", "cores (TMA), not WSIs"),
           "NB_MycN_WSI": ("NB MYCN WSI validation set", 6, "WSI, H&E + FISH", "NB", "Not applicable (NB only)", "St. Jude", ""),
           "NB_ADRN_vs_MES": ("NB ADRN vs MES", 58, "WSI, H&E (plus molecular evidence)", "NB", "Not applicable (NB only)", "St. Jude", "T7: remove '14 Brain/CNS-region slides' wording"),
           "COMET_RMS_arispe": ("COMET RMS arispe", 167, "WSI, H&E", "RMS", "Native", "St. Jude", "same institution as library"),
           "COG_RMS_mutation": ("COG RMS mutation", 515, "WSI, H&E", "RMS", "Not applicable (RMS only)", "CCDI/MCI", "possible participant overlap with CCDI-MCI collection: not established")}
    exp = {"NB_MycN_TMA": 99, "NB_MycN_WSI": 6, "NB_ADRN_vs_MES": 55, "COMET_RMS_arispe": 167, "COG_RMS_mutation": 515}
    for k, (name, n_sub, mat, lab, align, src, note) in sub.items():
        d = pd.read_csv(paths[f"HX_{k}"], dtype=str)
        expect(f"hist_{k}_rows_eq_unique", int(len(d)), int(d["Scan ID"].nunique()), chk)
        n = int(d["Scan ID"].nunique()); expect(f"hist_{k}", n, exp[k], chk)
        unit = "cores" if k == "NB_MycN_TMA" else "slides"
        printed = f"{n}" + (f" (submitted {n_sub}; {n_sub - n} {unit} without retained output)" if n != n_sub else "")
        v2note = {"NB_MycN_TMA": "102 (A map); not checked here", "NB_ADRN_vs_MES": "58 (A map); not checked here"}.get(k, "[U] not checked here")
        rows.append(dict(collection=name, n_slides_printed=printed, submitted_count_S14=n_sub, retained_uni2_output_rows=n,
                         retained_virchow2_output_rows=v2note, material_stain_S14=mat, labels_used_S14=lab,
                         alignment_to_COMET_S14=align, source_S14=src, patients="not established",
                         current_comparison="no (historical exploratory; not a reconciled evaluation denominator)", note=note))
    return rows


def s15(paths, chk, m, groups14):
    v3 = pd.read_csv(paths["V03"], dtype=str, encoding="utf-8-sig")
    expect("v03_467", len(v3), 467, chk); expect("v03_unique_keys", int(v3.ccdi_diagnosis.duplicated().sum()), 0, chk)
    s = pd.read_csv(paths["S15"], dtype=str, encoding="utf-8-sig")
    expect("s15_471", len(s), 471, chk)
    j = s.merge(v3, on="ccdi_diagnosis", how="left", suffixes=("_s15", "_v03"))
    shared = j[j.comet_diagnosis_v03.notna()]
    expect("s15_v03_target_disagreements", int((shared.comet_diagnosis_s15 != shared.comet_diagnosis_v03).sum()), 0, chk)
    s15_only = j[j.comet_diagnosis_v03.isna()]
    expect("s15_only_rows", s15_only.comet_diagnosis_s15.tolist(), ["ATRT"] * 4, chk)
    mm = m.dropna(subset=["ccdi_diagnosis"])
    jm = mm.merge(v3[["ccdi_diagnosis", "comet_diagnosis"]].rename(columns={"comet_diagnosis": "v03"}), on="ccdi_diagnosis", how="left")
    expect("manifest_v03_target_disagreements", int((jm.comet_diagnosis.fillna("NA") != jm.v03.fillna("NA")).sum()), 0, chk)

    keys = sorted(set(v3.ccdi_diagnosis) | set(mm.ccdi_diagnosis) | set(s.ccdi_diagnosis))
    v3i = v3.set_index("ccdi_diagnosis"); s15i = s.drop_duplicates("ccdi_diagnosis").set_index("ccdi_diagnosis")
    mm = mm.assign(_unres=((mm.original_ffpe_546 == "True") & (mm.scope_status == "UNRESOLVED")),
                   _foxr2=((mm.original_ffpe_546 == "True") & (mm.scope_status == "EXCLUDE_KNOWN_FOXR2")))
    cnt = mm.groupby("ccdi_diagnosis").agg(n_staged_4160=("slide_id", "size"), n_evaluated_4154=("evaluated", "sum"),
                                           n_in_vocab_765=("population", lambda x: int((x == "INSCOPE").sum())),
                                           n_ffpe_546=("original_ffpe_546", lambda x: int((x == "True").sum())),
                                           n_scope_unresolved_546=("_unres", "sum"), n_foxr2_excluded_546=("_foxr2", "sum"),
                                           n_final_524=("ffpe_scope_clear", lambda x: int((x == "True").sum())),
                                           manifest_population=("population", lambda x: "/".join(sorted(set(x)))))
    unres_reason = mm[mm._unres].groupby("ccdi_diagnosis").scope_reason.agg(lambda x: "/".join(sorted(set(x)))).to_dict()
    PATH_FLAGS = {"Olfactory neuroblastoma": "pathology check: olfactory neuroblastoma (esthesioneuroblastoma) is not neuroblastoma — v03 maps to NB",
                  "Neurofibromatosis": "pathology check: neurofibromatosis is a syndrome, not a tumour type — v03 maps to NF"}

    def norm(k):
        return re.sub(r"[ \t]+", " ", k.replace("\xc2\xa0", " ").replace("\xa0", " ").replace("\xc2", "")).strip()
    out = []
    for k in keys:
        v_dx = v3i.comet_diagnosis.get(k) if k in v3i.index else None
        v_grp = v3i.comet_tumor_group.get(k) if k in v3i.index else None
        s_dx = s15i.comet_diagnosis.get(k) if k in s15i.index else None
        flags = []
        if "FOXR2" in k:
            status, target = "out of scope (CNS; corrected from NB — R1.2 scope decision)", "out of scope"
            flags.append("FOXR2 correction")
        elif v_grp is None and s_dx == "ATRT":
            status, target = "out of scope (CNS entity; no COMET reference class)", "out of scope"
            flags.append("ATRT row present in submitted S15 but absent from the v03 code input")
        elif v_grp is None:
            status, target = "unmapped (no mapping rule; slides excluded as unmapped)", "unmapped"
        elif v_grp == "NORMAL":
            status, target = "out of scope (no COMET reference class)", "out of scope"
        elif v_grp == "MISC":
            status, target = "excluded (COMET MISC: not a diagnostic group)", "MISC"
        elif v_grp == "Unmatched":
            status, target = "unmapped (no mapping rule; slides excluded as unmapped)", "unmapped"
        elif v_grp in groups14:
            status, target = "in scope", v_grp
        else:
            raise RuntimeError(f"unclassified v03 group {v_grp!r} for {k!r}")
        if v_dx == "Desmoid tumor" and "Desmoplastic fibroma" in k:
            flags.append("pathology check: 'Desmoplastic fibroma' is not desmoid fibromatosis")
        if "\xc2\xa0" in k or "\xa0" in k or "\xc2" in k:
            flags.append("source string contains mis-decoded no-break space; display form normalised")
        for needle, msg in PATH_FLAGS.items():
            if needle.lower() in k.lower() and status == "in scope":
                flags.append(msg)
        c = cnt.loc[k] if k in cnt.index else None
        n_unres = int(c.n_scope_unresolved_546) if c is not None else 0
        if n_unres:
            status = f"{status} (v03); {n_unres} FFPE slide(s) unresolved in R1.2 scope review ({unres_reason.get(k, '')}), excluded from 524"
        out.append(dict(ccdi_diagnosis_as_exported=k, display_string=norm(k),
                        compound_string=bool(";" in k or "#" in k), v03_comet_diagnosis=v_dx, v03_comet_tumor_group=v_grp,
                        submitted_S15_target=s_dx, revised_target=target, revised_status=status,
                        n_slides_staged_4160=int(c.n_staged_4160) if c is not None else 0,
                        n_slides_evaluated_4154=int(c.n_evaluated_4154) if c is not None else 0,
                        n_slides_in_vocabulary_765=int(c.n_in_vocab_765) if c is not None else 0,
                        n_slides_ffpe_546=int(c.n_ffpe_546) if c is not None else 0,
                        n_slides_scope_unresolved_546=n_unres, n_slides_foxr2_excluded_546=int(c.n_foxr2_excluded_546) if c is not None else 0,
                        n_slides_final_524=int(c.n_final_524) if c is not None else 0,
                        manifest_population=str(c.manifest_population) if c is not None else "(no staged slide)",
                        in_submitted_S15=k in s15i.index, in_v03_code_input=k in v3i.index, review_flags="; ".join(flags)))
    t = pd.DataFrame(out)
    t.insert(t.columns.get_loc("revised_status") + 1, "status_base", t.revised_status.str.replace(r" \(v03\);.*$", "", regex=True))
    expect("s15_status_base_values", int(t.status_base.nunique()), 6, chk)
    dupdisp = t[t.display_string.duplicated(keep=False)]
    for disp, g in dupdisp.groupby("display_string"):
        idx = list(g.index)
        for i in idx:
            others = "; ".join(repr(t.loc[j, "ccdi_diagnosis_as_exported"]) for j in idx if j != i)
            t.loc[i, "review_flags"] = (t.loc[i, "review_flags"] + "; " if t.loc[i, "review_flags"] else "") + \
                f"encoding variant of exported key {others}: same display string, distinct exported bytes; slides staged under this variant: {int(t.loc[i, 'n_slides_staged_4160'])}"
    expect("s15_display_collisions", int(len(dupdisp) // 2), 1, chk)
    expect("s15_identity_unmapped_plus_atrt", int(t[t.revised_target == "unmapped"].n_slides_staged_4160.sum() + t[t.review_flags.str.contains("ATRT row")].n_slides_staged_4160.sum()), 138, chk)
    expect("s15_identity_inscope_plus_foxr2", int(t[t.revised_target.isin(groups14)].n_slides_staged_4160.sum() + t[t.review_flags.str.contains("FOXR2")].n_slides_staged_4160.sum()), 765, chk)
    expect("s15_regen_total_unresolved", int(t.n_slides_scope_unresolved_546.sum()), 18, chk)
    expect("s15_regen_total_staged", int(t.n_slides_staged_4160.sum()), int(len(mm)), chk)
    expect("s15_regen_total_final", int(t.n_slides_final_524.sum()), 524, chk)
    expect("s15_regen_foxr2_final", int(t[t.review_flags.str.contains("FOXR2")].n_slides_final_524.sum()), 0, chk)
    summary = dict(n_strings=len(t), n_compound=int(t.compound_string.sum()), n_in_s15=int(t.in_submitted_S15.sum()),
                   n_in_v03=int(t.in_v03_code_input.sum()), n_manifest_only=int((~t.in_v03_code_input & ~t.in_submitted_S15).sum()),
                   status_base_counts=t.status_base.value_counts().to_dict(),
                   staged_by_status_base=t.groupby("status_base").n_slides_staged_4160.sum().to_dict(),
                   status_counts_full=t.revised_status.value_counts().to_dict(),
                   staged_by_status_full=t.groupby("revised_status").n_slides_staged_4160.sum().to_dict(),
                   writer_rule="report status_base (6 values); state once that 11 in-scope strings carry 18 FFPE slides unresolved in the R1.2 scope review (site-only 12, compound 6), excluded from the 524; do not print the suffixed statuses as categories",
                   no_metadata_slides_not_in_table=int(m.ccdi_diagnosis.isna().sum()), n_flagged=int((t.review_flags != "").sum()),
                   footnote_identities="S15 'unmapped' staged slides (105) + S15-only ATRT staged slides (33) = manifest UNMAPPED 138; S15 status_base 'in scope' staged (759; 549 under the bare status plus 210 under the 11 suffixed in-scope rows) + FOXR2 (6) = manifest INSCOPE 765; the ATRT status is a v02-provenance annotation (rows present in submitted S15, absent from the v03 code input), and the manifest treats those slides as UNMAPPED.")
    return t, summary


def s14_table(C, S, K, D, hist):
    cur = [
        dict(cohort="COMET (final library)", role="reference library and internal leave-one-patient-out queries",
             slides_delivered=f"{C['full']['slides']:,} [PM; MS L155]", slides_evaluable=f"{C['all_tissue']['slides']:,} all-tissue (−143 MISC) [PM]; {C['tumour_only']['slides']:,} tumour-only (−174 NORMAL) [PM]",
             slides_evaluated=f"{C['all_tissue']['slides']:,} queries vs {C['all_tissue']['slides']:,} references (all-tissue) [PM/T7]; tumour-only {C['tumour_only']['slides']:,} queries / 2,603 non-empty references (RB 16738 has zero tumour tiles) [PM/R2.1]; 2,766 tumour-only distance-matrix rows (12 slides — 11 NORMAL + RB 16738 — without finite tumour-only distance) [plan0002]",
             patients=f"{C['full']['patients']:,} library / {C['all_tissue']['patients']:,} all-tissue / {C['tumour_only']['patients']:,} tumour-only [PM]",
             groups="14 diagnostic groups + NORMAL (15 labels in 2,778) [PM]",
             specimen_types=f"FFPE archival; xenograft {C['xenograft_library']['slides']} slides / {C['xenograft_library']['patients']} patients in library, {C['xenograft_2778']['slides']} / {C['xenograft_2778']['patients']} in the 2,778 [RM]; normal tissue {C['sample_type_2778'].get('Normal', 0)}; {C['sample_type_2778'].get('not joined', 0)} slides without rich-metadata join [RM]",
             label_source="institutional pathology diagnosis (COMET tumour group) [MS L155–157]",
             submitted_to_current="Submitted S14 row 'COMET 2921 / COMET tumor group (16)': 2,921 unchanged; '16' → 15 labels in the 2,778 (14 diagnostic groups + NORMAL; MISC 143 removed) [PM]",
             exclusions=f"MISC {C['misc']} (not a diagnostic group) [PM]; NORMAL {C['normal']} in tumour-only arm [PM]; same-patient masking: {C['within_patient_pairs']:,} within-patient slide pairs in 2,778 [PM/ADR0002]; pre-library discrepant-pathology exclusion count: not sourced in any retained artefact [U]"),
        dict(cohort="STV (St. Jude extension)", role="additional query collection, same institution, later accession",
             slides_delivered=f"{S['labelled']['slides']} labelled [SM]", slides_evaluable=f"{S['after_export_exclusion']['slides']} after 6 truncated exports [SM/ZT]",
             slides_evaluated=f"{S['main']['slides']} (query-only) / {S['both_sided']['slides']} both-sided (−8 NORMAL) [C12/SY]",
             patients=f"{S['labelled']['patients']} → {S['after_export_exclusion']['patients']} → {S['main']['patients']} query-only → {S['both_sided']['patients']} both-sided (repeated ST_ IDs = same patient) [SM]",
             groups="15 labels incl. NORMAL [C12]", specimen_types="not recorded in manifest [SM]",
             submitted_to_current="not in submitted S14 (new cohort row)",
             label_source=f"curator label manifest; RET→RB provisional for {T7['stv_ret_to_rb_slides']} slides / {T7['stv_ret_to_rb_patients']} patients [T7]",
             exclusions=f"6 truncated exports ({', '.join(S['truncated_exports']['dx'])}) [ZT]; {S['unsupported_dx']['slides']} slides / {S['unsupported_dx']['patients']} patients with diagnoses outside the 14 groups ({prose(S['unsupported_dx']['dx'])}) [SM/C12]; patient overlap with COMET not verified"),
        dict(cohort="KidRare (provider release)", role="additional query collection, external institutions (per source publications)",
             slides_delivered=f"{K['source']:,} [KE]", slides_evaluable=f"{K['processed']:,} processed ({T7['kidrare_unreadable']} unreadable, {T7['kidrare_no_tissue']} no tissue [T7]) [KE]",
             slides_evaluated=f"{K['amended']['slides']:,} eligible; query-only returned {K['query_only']['returned']:,} of {K['query_only']['target']:,}; both-sided {K['both_sided']['returned']:,} of {K['both_sided']['target']:,} (−50 NORMAL; 1 LIVER query with no retained tumour tiles) [KE/SY]",
             patients=K["patients"], groups=f"LIVER {K['amended']['groups']['LIVER']}, NB {K['amended']['groups']['NB']}, WT {K['amended']['groups']['WT']}, NORMAL {K['amended']['groups']['NORMAL']} [KE]",
             specimen_types="not established [T7]", submitted_to_current="not in submitted S14 (new cohort row)", label_source=f"provider folder labels + PathPT archived subtype for {K['amended']['pathpt_integrated_labels']} WT slides (R1.3); legacy {K['amended']['legacy_subset']:,} subset retained as sensitivity [KE]",
             exclusions=f"MB {K['excluded_unsupported']['MB']} (no COMET reference class); {K['excluded_unsupported']['WT_UNVERIFIED']} unlabelled kidney slides [KE]"),
        dict(cohort="CCDI-MCI (external)", role="additional query collection, external institutions",
             slides_delivered=f"{D['downloaded_MS']:,} downloaded [MS L77/L163; no retained artefact]; {D['staged_with_features']:,} with UNI2 features [CM]; {D['evaluated_nonempty']:,} with historical retrieval output (6 without result: 1 out-of-scope, 5 no metadata) [CM/CP]",
             slides_evaluable=f"{D['in_vocabulary']['slides']} in COMET vocabulary (= 765 participants; 26 suffixed re-exports from 15 of them lacked metadata) [CM]; {D['ffpe_546']} verified FFPE [CM]",
             slides_evaluated=f"{D['final']['slides']} (FFPE, scope-clear) [CM/T1]", patients=f"{D['final']['participants']} participants [CM]",
             groups=f"{len(D['final']['groups'])} observed groups [CM]", specimen_types=f"FFPE {D['preparation_765']['VERIFIED_FFPE']}; frozen/OCT {D['preparation_765']['FROZEN_OCT']}, mixed {D['preparation_765']['MIXED']}, unclassified {D['preparation_765']['NONE_UNKNOWN']} excluded [CM]",
             label_source="CCDI diagnosis strings mapped to COMET groups (regenerated Table S15, v03 rule) [V03]",
             exclusions=f"of {D['evaluated_nonempty']:,}: out of scope {D['populations_4154']['OOS']:,}, no metadata {D['populations_4154']['NO_METADATA']}, unmapped {D['populations_4154']['UNMAPPED']}, MISC {D['populations_4154']['MISC']} [CM]; of 546 FFPE: {D['scope_546']['EXCLUDE_KNOWN_FOXR2']} CNS NB FOXR2, {D['scope_546']['UNRESOLVED']} slides with unresolved scope (site-only {D['unresolved_546_reasons'].get('site_only', 0)}, compound {D['unresolved_546_reasons'].get('compound_support', 0)}; {D['unresolved_546_n_strings']} distinct strings) [CM]; post-result restriction disclosed [T1]",
             submitted_to_current="Submitted S14 row 'CCDI-MCI full dataset 948 / 21 entities / Manual slide level mapping (Table S10)': 948 is not reconstructable from any retained artefact [U]; published Fig. 4C 867 = Fig. 4C notebook (v03 mapping; participant-level metadata join; all non-Normal/Unmatched comet_diagnosis labels, 20 entities) = 765 + 68 MISC-group slides + 34 suffixed re-exports (21 participants) [CP/CMD/V03/CM]; submitted '21 entities' = 23 S15 labels − Normal − Unmatched (includes ATRT, absent from v03), Fig. 4C showed 20; 21 entities → 12 observed COMET groups in the 524 [CM]; Table S10 → regenerated Table S15 [V03]"),
    ]
    return pd.DataFrame(cur), pd.DataFrame(hist)


def prose(d):
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))


def to_md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines) + "\n"


def flow_figure(C, S, K, D, hist, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, axes = plt.subplots(1, 4, figsize=(19, 9.5))
    fig.subplots_adjust(wspace=0.45)
    panels = [
        ("a  COMET final library", [
            (f"{C['full']['slides']:,} slides / {C['full']['patients']:,} patients\n(final library)", None),
            (f"{C['all_tissue']['slides']:,} / {C['all_tissue']['patients']:,}\nall-tissue queries and references", f"− MISC {C['misc']} (not a diagnostic group)"),
            (f"{C['tumour_only']['slides']:,} / {C['tumour_only']['patients']:,}\ntumour-only (14 groups)", f"− NORMAL {C['normal']}"),
            (f"2,603 non-empty references", "− 1 slide with zero tumour tiles (RB 16738)")],
         f"Same-patient masking: {C['within_patient_pairs']:,} within-patient slide pairs in 2,778.\n2,766 tumour-only matrix rows: 12 slides (11 NORMAL + RB 16738)\nhave no finite tumour-only distance.\nNormal tissue 171 (Sample.Type) + 3 NORMAL slides\nwithout rich-metadata join = NORMAL 174."),
        ("b  STV (St. Jude extension)", [
            (f"{S['labelled']['slides']} slides / {S['labelled']['patients']} patients\nlabelled manifest", None),
            (f"{S['after_export_exclusion']['slides']} / {S['after_export_exclusion']['patients']}", "− 6 truncated exports"),
            (f"{S['main']['slides']} / {S['main']['patients']}\nquery-only evaluated", f"− {S['unsupported_dx']['slides']} slides / {S['unsupported_dx']['patients']} patients\ndiagnoses outside the 14 groups"),
            (f"{S['both_sided']['slides']} / {S['both_sided']['patients']}\nboth-sided", "− 8 NORMAL queries")],
         "Repeated ST_ identifiers denote the same patient.\nRET→RB label provisional (32 slides / 19 patients)."),
        ("c  KidRare (provider release)", [
            (f"{K['source']:,} slides\n(patients not available)", None),
            (f"{K['processed']:,} processed", "− 1 unreadable, − 8 no tissue"),
            (f"{K['amended']['slides']:,} eligible\nLIVER 815 · NB 303\nWT 413 · NORMAL 50", f"− MB {K['excluded_unsupported']['MB']} (no reference class)\n− {K['excluded_unsupported']['WT_UNVERIFIED']} unlabelled kidney"),
            (f"query-only {K['query_only']['returned']:,} / {K['query_only']['target']:,}\nboth-sided {K['both_sided']['returned']:,} / {K['both_sided']['target']:,}", "query-only: 1 LIVER query with no\n  tumour tiles retained (1,581 → 1,580)\nboth-sided: − 50 NORMAL, same\n  LIVER query (1,531 → 1,530)")],
         "357 WT labels integrated from PathPT archive (R1.3);\nlegacy 1,227-slide subset retained as sensitivity."),
        ("d  CCDI-MCI (external)", [
            (f"{D['downloaded_MS']:,} downloaded\n(per manuscript;\nno retained artefact)", None),
            (f"{D['staged_with_features']:,} with features\n{D['evaluated_nonempty']:,} with historical\nretrieval output (6 without\nresult: 1 OOS, 5 no metadata)", "147 = 4,307 − 4,160 (difference,\n  not an observed exclusion)"),
            (f"{D['in_vocabulary']['slides']} in COMET vocabulary\n(= 765 participants)", f"− out of scope {D['populations_4154']['OOS']:,}\n− no metadata {D['populations_4154']['NO_METADATA']} · unmapped {D['populations_4154']['UNMAPPED']} · MISC {D['populations_4154']['MISC']}"),
            (f"{D['ffpe_546']} verified FFPE", f"− frozen/OCT {D['preparation_765']['FROZEN_OCT']} · mixed {D['preparation_765']['MIXED']} · unclassified {D['preparation_765']['NONE_UNKNOWN']}"),
            (f"{D['final']['slides']} evaluated\n({D['final']['participants']} participants, {len(D['final']['groups'])} groups)", f"− {D['scope_546']['EXCLUDE_KNOWN_FOXR2']} CNS NB FOXR2\n− {D['scope_546']['UNRESOLVED']} slides with unresolved scope\n  (site-only {D['unresolved_546_reasons'].get('site_only', 0)}, compound {D['unresolved_546_reasons'].get('compound_support', 0)}; {D['unresolved_546_n_strings']} strings)")],
         "Submitted S14 '948' is not reconstructable from retained artefacts. Published Fig. 4C 867\n= notebook route (v03 mapping, participant-level join, 20 entities) = 765 + 68 MISC-group\nslides + 34 suffixed re-exports (21 participants). 26 suffixed re-exports from 15\nin-vocabulary participants lacked metadata (within 'no metadata')."),
    ]
    for ax, (title, boxes, note) in zip(axes, panels):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        n = len(boxes); ys = np.linspace(0.90, 0.22, n)
        for k, ((txt, excl), y) in enumerate(zip(boxes, ys)):
            dashed = "no retained artefact" in txt
            ax.add_patch(FancyBboxPatch((0.05, y - 0.055), 0.62, 0.11, boxstyle="round,pad=0.01", fc="white" if dashed else "#e8eef7", ec="black", lw=0.8, ls="--" if dashed else "-"))
            ax.text(0.36, y, txt, ha="center", va="center", fontsize=7.5)
            if k < n - 1:
                ax.annotate("", xy=(0.36, ys[k + 1] + 0.056), xytext=(0.36, y - 0.056), arrowprops=dict(arrowstyle="->", lw=0.8))
            if excl:
                ax.text(0.70, (y + ys[k - 1]) / 2 if k else y, excl, ha="left", va="center", fontsize=6.5, color="#7a1f1f")
        ax.text(0.05, 0.06, note, ha="left", va="center", fontsize=6.5, style="italic")
    fig.text(0.5, 0.015, "Historical exploratory collections (not in the current comparison; submitted count / retained UNI2 output rows): " +
             "; ".join(f"{h['collection']} {h['submitted_count_S14']}/{h['retained_uni2_output_rows']}" for h in hist), ha="center", fontsize=7)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.time()
    out_root = os.environ.get("PROSPER_OUT")
    if not out_root:
        raise RuntimeError("PROSPER_OUT must be exported")
    OUT = os.path.join(out_root, "r3_3_cohort_accounting_20261008_v3")
    os.makedirs(OUT, exist_ok=False)
    paths = {
        "PM": f"{FSL}/final_scan_id_disease_map_with_sample_id_patient_id.csv", "RM": f"{FSL}/final_scan_id_disease_map_rich_data.csv",
        "SM": f"{REPO}/data/COMET_validation_set_deid_20260902.csv", "ZT": f"{out_root}/qc_stv/zero_tile_scan.csv",
        "C12": f"{out_root}/r1_3_external_correction_20260930/stv_main_696_whole_slide_ot_per_query.csv",
        "KE": f"{out_root}/r1_3_external_correction_20260930/kidrare_source_2331_eligibility.csv",
        "SY_cov": f"{out_root}/r1_3_external_synthesis_20260930/coverage_and_hit_yield.csv",
        "SY_loss": f"{out_root}/r1_3_external_synthesis_20260930/separate_coverage_losses.csv",
        "CM": f"{out_root}/r1_2_scope_correction_20260930/manifest_original_all_4160_scope_review_20260930.csv",
        "CP": f"{TESTSETS}/output/COG_CCDI_MCI_ALL/search_results/uni2/predicted_majority_disease_top_10.csv",
        "V03": f"{TESTSETS}/data/input/ccdi_comet_mapping_v03.csv",
        "CMD": f"{TESTSETS}/data/input/CCDI_metadata_all.csv",
        "S15": f"{REPO}/docs/prosper_submission/supplementary/supplementary_tables/Table_S15_CCDI_MCI_mapping.csv",
    }
    for k in ("NB_MycN_TMA", "NB_MycN_WSI", "NB_ADRN_vs_MES", "COMET_RMS_arispe", "COG_RMS_mutation"):
        paths[f"HX_{k}"] = f"{TESTSETS}/output/{k}/search_results/uni2/predicted_majority_disease_top_5.csv"
    hashes = {k: sha(v) for k, v in paths.items()}
    expect("PM_hash_prefix", hashes["PM"][:8], "45f1466c", {})
    expect("RM_hash_prefix", hashes["RM"][:8], "a611115d", {})
    chk = {}
    C = comet(paths, chk); S = stv(paths, chk); K = kidrare(paths, chk); D = ccdi(paths, chk); H = historical(paths, chk)
    m = D.pop("m")
    T15, s15_summary = s15(paths, chk, m, C["groups14"])
    cur, hist = s14_table(C, S, K, D, H)
    cur.to_csv(f"{OUT}/table_s14_replacement_current_cohorts.csv", index=False)
    hist.to_csv(f"{OUT}/table_s14_replacement_historical_collections.csv", index=False)
    with open(f"{OUT}/table_s14_replacement.md", "w") as fh:
        fh.write("# Replacement Table S14 — cohorts and evaluation populations\n\nSource tags: PM COMET patient map; RM COMET rich metadata; SM STV manifest; ZT STV zero-tile export scan; C12 accepted STV per-query output; KE KidRare eligibility table; SY R1.3 synthesis coverage; CM CCDI 4,160-slide scope manifest; CP CCDI historical prediction rows; V03 CCDI→COMET mapping v03; CMD CCDI participant metadata (Fig. 4C notebook input); T7 R1.3-T7 edit sheet; T1 R1.2-T1 edit sheet; MS manuscript line. Slide counts are slides; patient counts use the patient identifier of each source.\n\n## Current reference / evaluation populations\n\n" + to_md(cur) +
                 "\n## Historical exploratory collections\n\nThe submitted count is the source count stated in the submitted Table S14; the second column is the number of distinct scan identifiers in the retained UNI2 output. Neither is a reconciled current evaluation denominator; these collections are not part of the current comparison and patient overlap with other cohorts is not established.\n\n" + to_md(hist))
    T15.to_csv(f"{OUT}/table_s15_regenerated.csv", index=False)
    flow_figure(C, S, K, D, H, f"{OUT}/cohort_flow.png")
    rec = dict(script=os.path.basename(__file__), script_sha256=sha(os.path.abspath(__file__)), inputs=paths, input_sha256=hashes,
               constants=dict(T7=T7, T1=T1, MS=MS), assertions=chk, comet=C, stv=S, kidrare=K, ccdi=D, historical=H, s15_summary=s15_summary,
               library_versions={x: __import__(x).__version__ for x in ("numpy", "pandas", "matplotlib")}, python=sys.version,
               runtime_s=round(time.time() - t0, 1))
    json.dump(rec, open(f"{OUT}/counts.json", "w"), indent=1, default=str)
    with open(f"{OUT}/README.md", "w") as fh:
        fh.write("# R3.3 cohort accounting (batch 3 E4)\n\n" + __doc__ + f"\n\nOutputs: table_s14_replacement.md/.csv (current + historical), table_s15_regenerated.csv ({len(T15)} diagnosis strings; status_base counts {s15_summary['status_base_counts']}), cohort_flow.png, counts.json ({len(chk)} assertions, all passed).\n\nNewly computed, not previously accepted: the COMET rich-metadata `Discrepant` flag distribution within the 2,778 ({C['discrepant_2778_descriptive']}) is descriptive only and is not the pre-library discrepant-pathology exclusion count, which remains unsourced.\nRuntime {rec['runtime_s']} s.\n")
    with open(f"{OUT}/sha256.txt", "w") as fh:
        for n in sorted(os.listdir(OUT)):
            if n not in ("sha256.txt", "COMPLETE"):
                fh.write(f"{sha(os.path.join(OUT, n))}  {n}\n")
    open(f"{OUT}/COMPLETE", "w").write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print(f"assertions passed: {len(chk)}; S15 strings {len(T15)}; wrote {OUT} in {rec['runtime_s']} s")


if __name__ == "__main__":
    main()
