"""Bounded readback (D4, 2026-10-21): do the Fig. 7 ROI-search examples' top-10 hits include
slides from the query's own patient?

The c1b ROI pickles hold the top-10 library hits for every mosaic patch of a query slide (self-slide
already excluded). The Fig. 7 ROIs were drawn interactively and are not identified by patch id, so the
legend tallies cannot be re-derived exactly; this readback bounds the possible same-patient contribution
across all patches of each slide and tallies the labels of all hits with/without same-patient slides.
Read-only; writes a CSV + JSON record under $PROSPER_OUT/fig7_roi_same_patient_readback_20261021/.
"""
import glob, hashlib, json, os, pickle, sys
import pandas as pd

OUT = os.environ.get("PROSPER_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prosper_revision_output"))
D = "/PROSPER_WORKSPACE/comet-new-scans/output/top_k_hit_slides_final_slides_final"
META = ("/PROSPER_WORKSPACE/comet-feature-gen/comet-analysis/"
        "data/final_scan_id_disease_map_with_sample_id_patient_id.csv")
SLIDES = [("SJMPNST076015_D1", "12628", "7B"), ("SJHB075481_D1", "9937", "7C"), ("SJWLM056159_D2", "13093", "7D")]
dest = os.path.join(OUT, "fig7_roi_same_patient_readback_20261021"); os.makedirs(dest, exist_ok=True)
meta = pd.read_csv(META)
s2p = dict(zip(meta["Slide.Scan.File"].astype(str), meta["UID.Subject"]))
s2l = dict(zip(meta["Slide.Scan.File"].astype(str), meta["Tumor.group"]))
rows, rec = [], {"inputs": {"meta": {"path": META, "sha256": hashlib.sha256(open(META, "rb").read()).hexdigest()}}, "slides": {}}
for name, sid, panel in SLIDES:
    f = glob.glob(f"{D}/*_{sid}.pkl"); assert len(f) == 1, f
    df = pickle.load(open(f[0], "rb")); df["hit"] = df.hit_slide_id.astype(str); pat = s2p[sid]
    df["rel"] = df.hit.map(lambda h: "self" if h == sid else ("same_patient" if s2p.get(h) == pat else "other"))
    df["lab"] = df.hit.map(s2l)
    pp = df.groupby("query_patch_id").rel.apply(lambda s: int((s == "same_patient").sum()))
    own = meta[meta["UID.Subject"] == pat]
    r = dict(panel=panel, slide=name, scan_id=sid, patient=pat, library_label=s2l[sid], n_query_patches=int(df.query_patch_id.nunique()),
             n_hits=len(df), n_self_hits=int((df.rel == "self").sum()), n_same_patient_hits=int((df.rel == "same_patient").sum()),
             patient_slides_in_library=";".join(f"{a}:{b}" for a, b in zip(own["Slide.Scan.File"].astype(str), own["Tumor.group"])),
             same_patient_hit_slides=";".join(sorted(set(df[df.rel == "same_patient"].hit))),
             n_patches_with_any_same_patient_hit=int((pp > 0).sum()), max_same_patient_hits_in_one_patch=int(pp.max()),
             label_tally_all=json.dumps(df.lab.value_counts().to_dict()),
             label_tally_excl_same_patient=json.dumps(df[df.rel == "other"].lab.value_counts().to_dict()),
             pickle=os.path.basename(f[0]), pickle_sha256=hashlib.sha256(open(f[0], "rb").read()).hexdigest())
    rows.append(r); rec["slides"][name] = r
    print(f"{panel} {name}: {r['n_query_patches']} patches, self hits {r['n_self_hits']}, same-patient hits {r['n_same_patient_hits']} "
          f"(max/patch {r['max_same_patient_hits_in_one_patch']}); patient slides {r['patient_slides_in_library']}")
pd.DataFrame(rows).to_csv(os.path.join(dest, "fig7_roi_same_patient_readback.csv"), index=False)
rec["note"] = __doc__
json.dump(rec, open(os.path.join(dest, "readback.json"), "w"), indent=2)
