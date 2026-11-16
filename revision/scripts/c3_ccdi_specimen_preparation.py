"""C3 — specimen preparation (FFPE vs frozen/OCT) of every CCDI-MCI slide we evaluated.

Read-only. Two independent sources, cross-checked:

  A. CCDI study manifest (dbGaP phs002790, v3.1.0), sheet `pathology_file`, column
     `fixation_embedding_method`, joined pathology_file -> sample -> participant.
  B. The DICOM headers of the slides on disk: SpecimenDescriptionSequence ->
     SpecimenPreparationSequence ("Tissue Fixative", "Embedding medium"), read from the
     smallest .dcm of every series (headers only, stop_before_pixels).

Slide ids in `$PROSPER_OUT/external_ccdi/manifest_all.csv` are participant ids (USI), with
`_0/_1` suffixes for the 217 re-exports; the participant is the join key on both routes.

Outputs
  $PROSPER_OUT/external_ccdi/slide_preparation.csv          per slide (4,160 rows)
  $PROSPER_OUT/external_ccdi/series_preparation_dicom.csv   per DICOM series (4,204 rows)
  [internal record]          per class, in-scope 765
  [internal record]             all / in-scope / OOS / ...
  [internal record]       wrong@k / miss@k split FFPE vs frozen
  [internal record]      per class, all / FFPE / frozen
  $PROSPER_OUT/external_ccdi/per_query_ot_with_preparation.csv   C3 primary per-query + prep

Usage:  PROSPER_OUT=... python3 scripts/c3_ccdi_specimen_preparation.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pydicom

PROSPER_OUT = os.environ.get("PROSPER_OUT")
if not PROSPER_OUT:
    sys.exit("export PROSPER_OUT first (see CLAUDE.md)")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CCDI = "/PROSPER_GROUP_DATA/CCDI_MCI/common"
MANIFEST_XLSX = f"{CCDI}/ccdi_mci_v3/phs002790_CCDI_Study_Manifest_v3.1.0.xlsx"
DICOM_ROOT = f"{CCDI}/ccdi_mci"
OUT = f"{PROSPER_OUT}/external_ccdi"
RES = f"{REPO}/docs/results/c3"

NON_INSCOPE = {"OOS", "NO_METADATA", "UNMAPPED", "MISC"}
FIX_MAP = {"Formalin": "FFPE", "Tissue freezing medium": "FROZEN_OCT", "NONE": "NONE"}


def manifest_route():
    pf = pd.read_excel(MANIFEST_XLSX, sheet_name="pathology_file", dtype=str)
    pf = pf[pf["type"].notna()]
    sm = pd.read_excel(MANIFEST_XLSX, sheet_name="sample", dtype=str)
    sm = sm[sm["type"].notna()]
    j = pf.merge(sm[["sample_id", "participant.participant_id"]],
                 left_on="sample.sample_id", right_on="sample_id", how="left")
    assert j["participant.participant_id"].notna().all()
    per = j.groupby("participant.participant_id")["fixation_embedding_method"].agg(
        lambda s: ";".join(sorted(set(s))))
    return per, pf["fixation_embedding_method"].value_counts()


def dicom_route():
    rows = []
    for part in sorted(os.listdir(DICOM_ROOT)):
        for sd in sorted(glob.glob(f"{DICOM_ROOT}/{part}/*/SM_*")):
            fs = sorted(glob.glob(f"{sd}/*.dcm"), key=os.path.getsize)
            if not fs:
                rows.append(dict(participant=part, series=os.path.basename(sd),
                                 fixative="NO_DCM", embedding=""))
                continue
            ds = pydicom.dcmread(fs[0], stop_before_pixels=True)
            fx, em = set(), set()
            for s in ds.get("SpecimenDescriptionSequence", []):
                for step in s.get("SpecimenPreparationSequence", []):
                    for item in step.get("SpecimenPreparationStepContentItemSequence", []):
                        cn = item.get("ConceptNameCodeSequence")
                        cn = cn[0].CodeMeaning if cn else None
                        cc = item.get("ConceptCodeSequence")
                        val = cc[0].CodeMeaning if cc else item.get("TextValue")
                        if cn == "Tissue Fixative":
                            fx.add(str(val))
                        if cn == "Embedding medium":
                            em.add(str(val))
            rows.append(dict(participant=part, series=os.path.basename(sd),
                             container=ds.get("ContainerIdentifier"),
                             fixative=";".join(sorted(fx)) or "NONE",
                             embedding=";".join(sorted(em)) or "NONE"))
    d = pd.DataFrame(rows)
    d["prep"] = d.fixative.map(FIX_MAP).fillna("OTHER:" + d.fixative)
    return d


def main():
    os.makedirs(RES, exist_ok=True)
    man = pd.read_csv(f"{OUT}/manifest_all.csv")
    man["participant"] = man.slide_id.astype(str).str.split("_").str[0]
    man["suffixed"] = man.slide_id.astype(str).str.contains("_")

    per_manifest, pf_counts = manifest_route()
    print("manifest pathology_file fixation_embedding_method:\n", pf_counts.to_string(), "\n")

    series = dicom_route()
    series.to_csv(f"{OUT}/series_preparation_dicom.csv", index=False)
    print("DICOM series fixative x embedding:\n",
          pd.crosstab(series.fixative, series.embedding).to_string(), "\n")
    per_dicom = series.groupby("participant").agg(
        n_series=("series", "size"), prep_dicom=("prep", lambda s: ";".join(sorted(set(s)))))

    man["prep_manifest"] = man.participant.map(per_manifest).fillna("NOT_IN_MANIFEST_V3.1.0")
    man = man.merge(per_dicom, left_on="participant", right_index=True, how="left")
    man["prep_dicom"] = man.prep_dicom.fillna("NO_LOCAL_DICOM")
    man["population"] = man.tumor_group.where(man.tumor_group.isin(NON_INSCOPE), "INSCOPE")
    man.to_csv(f"{OUT}/slide_preparation.csv", index=False)

    print("DICOM route vs manifest route (all slides):\n",
          pd.crosstab(man.prep_dicom, man.prep_manifest).to_string(), "\n")

    pop = pd.crosstab(man.population, man.prep_dicom)
    pop["n"] = pop.sum(axis=1)
    pop["pct_frozen"] = (100 * pop.get("FROZEN_OCT", 0) / pop.n).round(1)
    pop.to_csv(f"{RES}/preparation_by_population.csv")
    print("by population (DICOM route):\n", pop.to_string(), "\n")

    ins = man[man.population == "INSCOPE"]
    ct = pd.crosstab(ins.tumor_group, ins.prep_dicom)
    ct["n"] = ct.sum(axis=1)
    ct["pct_frozen"] = (100 * ct.get("FROZEN_OCT", 0) / ct.n).round(1)
    ct = ct.sort_values("n", ascending=False)
    ct.to_csv(f"{RES}/inscope_preparation_by_class.csv")
    print(f"in-scope {len(ins)} by class (DICOM route):\n", ct.to_string())
    print("\nambiguous (unsuffixed id, >1 series, mixed prep):",
          int(((~man.suffixed) & man.prep_dicom.str.contains(";")).sum()))
    stratify_failures(man)


def stratify_failures(man):
    """Read-only re-read of the C3 primary per-query results by preparation. No re-run."""
    q = pd.read_csv(f"{OUT}_inscope/per_query_ccdi_uni2_ot_cosine.csv")
    d = q.merge(man[["slide_id", "prep_dicom"]], left_on="query", right_on="slide_id", how="left")
    assert d.prep_dicom.notna().all()
    d["prep"] = d.prep_dicom.replace({"FFPE;FROZEN_OCT": "MIXED", "NONE": "CYTOSPIN"})
    d["wrong_k5"] = d.generous_correct_k5 == 0
    d["miss_k5"] = d.hit_k5 == 0
    d["miss_k10"] = d.hit_k10 == 0
    d["wrong_k1"] = d.hit_k1 == 0
    d.to_csv(f"{OUT}/per_query_ot_with_preparation.csv", index=False)

    rows = []
    for fail in ["wrong_k1", "wrong_k5", "miss_k5", "miss_k10"]:
        tot = int(d[fail].sum())
        for k in ["FFPE", "FROZEN_OCT", "MIXED", "CYTOSPIN"]:
            s = d[d.prep == k]
            n_f = int(s[fail].sum())
            rows.append(dict(failure=fail, prep=k, n_slides=len(s), n_failed=n_f,
                             rate_within_prep=round(n_f / len(s), 3),
                             share_of_all_failures=round(n_f / tot, 3), all_failures=tot))
    fb = pd.DataFrame(rows)
    fb.to_csv(f"{RES}/inscope_failures_by_preparation.csv", index=False)
    print("\nfailures by preparation (OT primary, n = 765):\n", fb.to_string(index=False))

    f = d[d.prep == "FROZEN_OCT"].wrong_k5.values
    e = d[d.prep == "FFPE"].wrong_k5.values
    rng = np.random.default_rng(123)
    rr = [rng.choice(f, len(f)).mean() / rng.choice(e, len(e)).mean() for _ in range(5000)]
    print(f"\nwrong@5 frozen {f.mean():.3f} vs FFPE {e.mean():.3f}; RR {f.mean() / e.mean():.2f} "
          f"[{np.percentile(rr, 2.5):.2f}, {np.percentile(rr, 97.5):.2f}] (slide bootstrap, B = 5,000, seed 123)")

    ff, fz = d[d.prep == "FFPE"], d[d.prep == "FROZEN_OCT"]
    pc = pd.DataFrame({
        "n_all": d.groupby("true_label").size(),
        "hit5_all": d.groupby("true_label").hit_k5.mean().round(3),
        "wrong5_all": d.groupby("true_label").wrong_k5.sum().astype(int),
        "n_ffpe": ff.groupby("true_label").size(),
        "hit5_ffpe": ff.groupby("true_label").hit_k5.mean().round(3),
        "wrong5_ffpe": ff.groupby("true_label").wrong_k5.sum(),
        "n_frozen": fz.groupby("true_label").size(),
        "hit5_frozen": fz.groupby("true_label").hit_k5.mean().round(3),
        "wrong5_frozen": fz.groupby("true_label").wrong_k5.sum(),
    }).sort_values("n_all", ascending=False)
    pc.to_csv(f"{RES}/inscope_per_class_by_preparation.csv")
    print("\nper class:\n", pc.to_string())
    print(f"\nFFPE-only counterfactual (n = {len(ff)}): micro hit@5 {ff.hit_k5.mean():.3f}, "
          f"macro hit@5 {ff.groupby('true_label').hit_k5.mean().mean():.3f}, wrong@5 {int(ff.wrong_k5.sum())} "
          f"({100 * ff.wrong_k5.mean():.1f} %) | all 765: {d.hit_k5.mean():.3f} / "
          f"{d.groupby('true_label').hit_k5.mean().mean():.3f} / {int(d.wrong_k5.sum())} ({100 * d.wrong_k5.mean():.1f} %)")
    print("frozen wrong calls named:", fz[fz.wrong_k5].pred_generous_k5.value_counts().head(4).to_dict())
    print("frozen top-1 library slide:", fz.top1_id.value_counts().head(4).to_dict())


if __name__ == "__main__":
    main()
