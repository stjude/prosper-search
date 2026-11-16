#!/usr/bin/env python3
"""S6.0 gate ([analysis plan], [decision record]): label provenance + inventory for the
fine-grained subtype task family. Read-only over its inputs; writes manifests +
summary to $PROSPER_OUT/s6_0_label_inventory/.

Inputs (all read-only):
  - data/ALIdf3Ali.csv                      (authoritative COMET master metadata; untracked)
  - comet-llm/data/final_scan_id_disease_map_rich_data.csv  (slide-level entity table)
  - $PROSPER_OUT/external_ccdi/manifest_inscope.csv         (C3 evaluated CCDI population)
  - $PROSPER_OUT/ccdi_suffix_sweep/exclude_suffixed.csv     (SW1 exclusion list)
  - comet-search-analysis/data/similarity_matrix/combined_similarity_matrix_uni2_ot_cosine_tumor_only.pkl
    (tumour-only primary matrix; membership check only)
"""

import hashlib
import json
import os
import pickle
import re
import sys

import pandas as pd

CORE = "/PROSPER_WORKSPACE"
# The CSV is untracked data and lives only in the main checkout's docs/, not in
# worktrees; override with S6_ALI_CSV if it moves.
ALI_CSV = os.environ.get("S6_ALI_CSV",
                         f"{CORE}/prosper_revision/data/ALIdf3Ali.csv")
RICH_CSV = f"{CORE}/comet-llm/data/final_scan_id_disease_map_rich_data.csv"
PROSPER_OUT = os.environ.get("PROSPER_OUT", f"{CORE}/prosper_revision_output")
CCDI_MANIFEST = f"{PROSPER_OUT}/external_ccdi/manifest_inscope.csv"
CCDI_EXCLUDE = f"{PROSPER_OUT}/ccdi_suffix_sweep/exclude_suffixed.csv"
MATRIX = (f"{CORE}/comet-search-analysis/data/similarity_matrix/"
          "combined_similarity_matrix_uni2_ot_cosine_tumor_only.pkl")
OUT = f"{PROSPER_OUT}/s6_0_label_inventory"

RMS_KEEP = {"ERMS": "ERMS", "ARMS": "ARMS", "Spindle": "Spindle"}
RMS_EXCLUDE = {"RMS NOS", "Unknown"}
CCDI_RMS_MAP = {
    "Embryonal rhabdomyosarcoma, NOS": "ERMS",
    "Alveolar rhabdomyosarcoma": "ARMS",
    "Spindle cell rhabdomyosarcoma": "Spindle",
}
FUSION_COLS = ["PA3.fusion", "PA7.fusion", "FKHR..FOO1..FISH",
               "PA3..FOO1.PCR", "PA7..FOO1.PCR"]
SUBTYPE_ALT_COLS = ["Subtype.use.for.clustering", "OriginalSubtype.use.for.clustering",
                    "D.subtype.from.CC.sheet.061223"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_nmyc(v):
    if not isinstance(v, str):
        return None, None
    s = v.strip()
    if s.upper().startswith("Y"):
        return "amplified", (s if len(s) > 1 else None)
    if s.upper() == "N":
        return "not_amplified", None
    return None, None  # 'unknown', ''


def norm_inpc(v):
    if not isinstance(v, str):
        return None
    s = v.strip().lower()
    if s.startswith("unfavorable"):
        return "unfavorable"
    if s.startswith("favorable"):
        return "favorable"
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {"inputs": {}, "tasks": {}}

    for name, path in [("ALIdf3Ali", ALI_CSV), ("rich", RICH_CSV),
                       ("ccdi_manifest", CCDI_MANIFEST), ("ccdi_exclude", CCDI_EXCLUDE)]:
        summary["inputs"][name] = {"path": path, "sha256": sha256(path),
                                   "mtime": os.path.getmtime(path)}

    master = pd.read_csv(ALI_CSV, low_memory=False)
    rich = pd.read_csv(RICH_CSV, low_memory=False)
    rich["slide_id"] = rich["slide_id"].astype(str)

    # --- join master -> rich on Ali_Slides
    m = master.copy()
    m["slide_key"] = (m["Ali_Slides"].astype(str)
                      .str.replace(r"\.svs$", "", regex=True))
    m.loc[m["slide_key"].isin(["nan", "None", ""]), "slide_key"] = None
    joined = rich.merge(m, left_on="slide_id", right_on="slide_key",
                        how="left", suffixes=("", "_master"))
    dup = joined.duplicated("slide_id", keep=False)
    summary["join"] = {
        "rich_slides": len(rich),
        "rich_matched_master": int(joined.drop_duplicates("slide_id")["slide_key"].notna().sum()),
        "slides_with_multiple_master_rows": int(joined.loc[dup, "slide_id"].nunique()),
    }

    with open(MATRIX, "rb") as fh:
        mat = pickle.load(fh)
    mat_ids = {str(i) for i in mat.index}
    summary["matrix"] = {"path": MATRIX, "shape": list(mat.shape)}

    def finish(task, df, label_col="label"):
        df = df.copy()
        df["in_tumor_only_matrix"] = df["slide_id"].isin(mat_ids)
        path = f"{OUT}/manifest_{task}.csv"
        df.to_csv(path, index=False)
        pats = df["patient_id"].dropna()
        summary["tasks"][task] = {
            "manifest": path,
            "n_slides": len(df),
            "n_slides_in_matrix": int(df["in_tumor_only_matrix"].sum()),
            "n_patients": int(pats.nunique()) if len(pats) else None,
            "n_slides_no_patient_id": int(df["patient_id"].isna().sum()),
            "by_label_slides": df[label_col].value_counts().to_dict(),
            "by_label_slides_in_matrix":
                df[df["in_tumor_only_matrix"]][label_col].value_counts().to_dict(),
            "by_label_patients":
                df.dropna(subset=["patient_id"]).drop_duplicates(["patient_id", label_col])
                  .groupby(label_col)["patient_id"].nunique().to_dict(),
            "n_xenograft": int((df.get("sample_type") == "Xenograft").sum()),
        }
        return df

    base_cols = {"slide_id": "slide_id", "tumor_group": "tumor_group",
                 "UID.Subject": "patient_id", "Sample.Type_master": None,
                 "Sample.Type": "sample_type", "Evaluation": "evaluation",
                 "Met": "met", "Subtype": "master_subtype"}

    def slim(df, label, source, note=None):
        out = pd.DataFrame({
            "slide_id": df["slide_id"],
            "patient_id": df.get("UID.Subject"),
            "label": label,
            "label_source": source,
            "sample_type": df["Sample.Type"],
            "evaluation": df["Evaluation"],
            "met": df.get("Met"),
        })
        if note is not None:
            out["label_note"] = note
        return out

    j1 = joined.drop_duplicates("slide_id")

    # --- S6.1 internal: COMET RMS subtype
    rms = j1[(j1["tumor_group"] == "RMS") & j1["Subtype"].notna()].copy()
    rms["norm"] = rms["Subtype"].map(lambda s: RMS_KEEP.get(str(s).strip()))
    rms_lab = rms[rms["norm"].notna()]
    finish("s61_rms_comet", slim(rms_lab, rms_lab["norm"], "ALIdf3Ali.Subtype"))
    summary["tasks"]["s61_rms_comet"]["subtype_raw_all_rms_slides"] = (
        rms["Subtype"].value_counts().to_dict())
    # subtype-column consistency on labelled rows
    cons = {}
    for c in SUBTYPE_ALT_COLS:
        both = rms_lab[rms_lab[c].notna()]
        agree = (both[c].astype(str).str.strip() == both["Subtype"].astype(str).str.strip())
        cons[c] = {"n_both": len(both), "n_disagree": int((~agree).sum())}
    summary["tasks"]["s61_rms_comet"]["subtype_column_consistency"] = cons
    fus = {}
    for c in FUSION_COLS:
        v = rms_lab[c].dropna()
        fus[c] = {"n_nonnull": len(v),
                  "values": v.astype(str).str.strip().value_counts().head(6).to_dict()}
    summary["tasks"]["s61_rms_comet"]["fusion_columns"] = fus

    # --- S6.1 external: CCDI RMS subtype
    cc = pd.read_csv(CCDI_MANIFEST)
    excl = set(pd.read_csv(CCDI_EXCLUDE)["slide_id"].astype(str))
    cc = cc[~cc["slide_id"].astype(str).isin(excl)]
    ccr = cc[cc["tumor_group"] == "RMS"].copy()
    ccr["norm"] = ccr["ccdi_diagnosis"].map(CCDI_RMS_MAP)
    ccr["participant"] = ccr["slide_id"].astype(str).str.replace(r"_\d+$", "", regex=True)
    lab = ccr[ccr["norm"].notna()]
    out = pd.DataFrame({"slide_id": lab["slide_id"], "patient_id": lab["participant"],
                        "label": lab["norm"], "label_source": "CCDI Diagnosis (histologic)",
                        "sample_type": "Patient tumor", "evaluation": None, "met": None})
    out["in_tumor_only_matrix"] = False  # external; scored via CCDIxCOMET matrix
    out.to_csv(f"{OUT}/manifest_s61_rms_ccdi.csv", index=False)
    summary["tasks"]["s61_rms_ccdi"] = {
        "manifest": f"{OUT}/manifest_s61_rms_ccdi.csv",
        "n_rms_slides_after_sw1_exclusion": len(ccr),
        "n_slides_labelled": len(lab),
        "n_participants_labelled": int(lab["participant"].nunique()),
        "by_label_slides": lab["norm"].value_counts().to_dict(),
        "by_label_participants":
            lab.drop_duplicates(["participant", "norm"]).groupby("norm")["participant"]
               .nunique().to_dict(),
        "unlabelled_diagnoses":
            ccr[ccr["norm"].isna()]["ccdi_diagnosis"].value_counts().head(8).to_dict(),
    }

    # --- S6.2: COMET NB MYCN
    nb = j1[j1["tumor_group"] == "NB"].copy()
    nmyc = nb["NMYC.amplified..Y.N."].map(norm_nmyc)
    nb["norm"] = [t[0] for t in nmyc]
    nb["note"] = [t[1] for t in nmyc]
    nb_lab = nb[nb["norm"].notna()]
    finish("s62_mycn_comet", slim(nb_lab, nb_lab["norm"],
                                  "ALIdf3Ali.NMYC.amplified..Y.N.", nb_lab["note"]))
    v = nb_lab["MYCN_FISH"].dropna().astype(str).str.strip()
    summary["tasks"]["s62_mycn_comet"]["mycn_fish_nonnull"] = len(v)

    # --- S6.5 internal: COMET NB INPC
    nb["inpc"] = nb["INPC"].map(norm_inpc)
    inpc_lab = nb[nb["inpc"].notna()]
    finish("s65_inpc_comet", slim(inpc_lab, inpc_lab["inpc"], "ALIdf3Ali.INPC"))
    summary["tasks"]["s65_inpc_comet"]["inpc_raw_excluded"] = (
        nb[nb["inpc"].isna()]["INPC"].dropna().value_counts().head(8).to_dict())

    # --- S6.4: COMET LIVER HB vs HCC (labels from rich Diagnosis)
    liver = j1[(j1["tumor_group"] == "LIVER") & j1["Diagnosis"].isin(["HB", "HCC"])]
    finish("s64_hbhcc_comet", slim(liver, liver["Diagnosis"], "rich.Diagnosis"))

    # --- S6.5b: GNB vs NB entity contrast (rich Diagnosis within NB group)
    gnb = j1[(j1["tumor_group"] == "NB") &
             j1["Diagnosis"].isin(["NB", "Ganglioneuroblastoma"])]
    finish("s65b_gnb_comet", slim(gnb, gnb["Diagnosis"], "rich.Diagnosis"))

    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
