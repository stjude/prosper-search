"""Compare the 2026-09-28 single-ILP v3 pools with v2; audit CCDI OCT exposure.

export PROSPER_OUT=<repo>/prosper_revision_output
python3 scripts/ilp_representatives/compare_review_pools_v3.py

Reads the actual review workbooks, source outcomes, current manifest and fresh
DICOM headers (one smallest file per local series for every CCDI pool/alternate
participant). Never solves or changes a pool. Writes comparison/audit CSVs and
input hashes under [internal record]/.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pydicom

from build_review_pool import CFG, OUT, REPO, XLSX, get_top_k_similar_slides, selection_checks

sys.path.insert(0, str(REPO / "scripts"))
from c3_ccdi_specimen_preparation import DICOM_ROOT, MANIFEST_XLSX, manifest_route

RESULTS = REPO / "docs/results/ilp_representatives/v3_20260928"
PACKETS = REPO / "docs/pathologist_study"
PREP_FILE = OUT / "external_ccdi/slide_preparation.csv"


def read_pool(path, sheet="pool"):
    frame = pd.read_excel(path, sheet_name=sheet, dtype={"slide_id": str, "patient_id": str})
    if frame.slide_id.isna().any() or frame.slide_id.duplicated().any():
        raise ValueError(f"Missing or duplicate slide IDs in {path}:{sheet}")
    return frame.set_index("slide_id")


def metrics(pool, outcomes):
    q = outcomes.loc[pool.index]
    if q[["pred_closest_k5", "true_label", "hit_k5"]].isna().any().any():
        raise ValueError("Missing retrieval outcomes for a pool")
    vote = q.pred_closest_k5 == q.true_label
    if not q.hit_k5.isin([0, 1]).all():
        raise ValueError("Unknown hit_k5 values")
    if "prosper_hit@5" in pool and not pool["prosper_hit@5"].equals(vote):
        raise ValueError("Historical closest-vote column differs from source outcomes")
    if "prosper_vote_correct_at5" in pool:
        if not pool.prosper_vote_correct_at5.equals(vote):
            raise ValueError("v3 closest-vote column differs from source outcomes")
        if not pool.prosper_hit_at5.equals(q.hit_k5.astype(bool)):
            raise ValueError("v3 hit@5 column differs from source outcomes")
    return int(vote.sum()), int(q.hit_k5.sum())


def fresh_dicom_preparation(participant):
    series = sorted((Path(DICOM_ROOT) / participant).glob("*/SM_*"))
    if not series:
        raise ValueError(f"No local DICOM series for {participant}")
    rows = []
    for directory in series:
        files = list(directory.glob("*.dcm"))
        if not files:
            raise ValueError(f"No DICOM files in {directory}")
        path = min(files, key=lambda f: (f.stat().st_size, str(f)))
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        values = {"Tissue Fixative": set(), "Embedding medium": set()}
        for specimen in ds.get("SpecimenDescriptionSequence", []):
            for step in specimen.get("SpecimenPreparationSequence", []):
                for item in step.get("SpecimenPreparationStepContentItemSequence", []):
                    names = item.get("ConceptNameCodeSequence", [])
                    codes = item.get("ConceptCodeSequence", [])
                    name = names[0].CodeMeaning if names else None
                    if name in values:
                        val = codes[0].CodeMeaning if codes else item.get("TextValue")
                        values[name].add(str(val) if val is not None else "UNKNOWN")
        fixative = ";".join(sorted(values["Tissue Fixative"])) or "UNKNOWN"
        embedding = ";".join(sorted(values["Embedding medium"])) or "UNKNOWN"
        rows.append(dict(slide_id=participant, series=directory.name, header_file=str(path),
                         fixative=fixative, embedding=embedding,
                         confirmed_ffpe=fixative == "Formalin" and embedding == "Paraffin wax"))
    return rows


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    prep = pd.read_csv(PREP_FILE, dtype=str).set_index("slide_id")
    if prep.index.has_duplicates:
        raise ValueError("Duplicate slide preparation records")
    comparison, class_rows, series_rows, exposure = [], [], [], []
    hashes = {}

    def record(path):
        hashes[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    for path in (Path(__file__), Path(__file__).with_name("build_review_pool.py"),
                 XLSX, PREP_FILE, Path(MANIFEST_XLSX)):
        record(path)
    manifest, _ = manifest_route()
    for cohort, cfg in CFG.items():
        work = OUT / "ilp_representatives" / cfg["work_cohort"]
        pop = pd.read_csv(work / "population.csv", dtype=str).set_index("slide_id").sort_index()
        if cohort == "ccdi":
            pop["patient_id"] = pop.index
        tiles = pd.read_csv(work / "tile_counts.csv", dtype={"slide_id": str}).set_index("slide_id").n_tiles
        prov_path = work / "review_pool" / f"provenance_{cohort}_v3_20260928.json"
        prov = json.loads(prov_path.read_text())
        args = SimpleNamespace(**prov["arguments"])
        eligible = {s for s in pop.index if tiles[s] >= prov["tile_floor_by_class"][pop.loc[s, "tumor_group"]]}
        ali = pd.read_excel(XLSX, sheet_name=cfg["sheet"])
        ali["slide_id"] = cfg["ali_key"](ali)
        ali = ali[ali.slide_id.isin(pop.index)].set_index("slide_id")
        matrix = pd.read_pickle(cfg["matrix"]).loc[pop.index, pop.index]
        nb = get_top_k_similar_slides(matrix, top_k=5)
        outcomes = pd.read_csv(cfg["per_query"], dtype={"query": str}).set_index("query")
        old_path = PACKETS / f"review_pool_{cohort}_v2_20260925.xlsx"
        new_path = PACKETS / f"review_pool_{cohort}_v3_20260928.xlsx"
        old, new = read_pool(old_path), read_pool(new_path)
        alternates = read_pool(new_path, "alternates")
        if len(new) != 40 or len(alternates) != 8 or set(new.index) & set(alternates.index):
            raise ValueError(f"{cohort}: wrong output sizes or overlapping pool/alternates")
        checks = selection_checks(new.index, pop, eligible, ali, args)
        if not checks.passed.all() or not set(alternates.index) <= eligible:
            raise ValueError(f"{cohort}: pool/alternate constraint failure")
        checks.to_csv(RESULTS / f"constraints_{cohort}_v3.csv", index=False)
        saved_checks = pd.read_excel(new_path, sheet_name="constraints")
        if not saved_checks.equals(checks):
            raise ValueError(f"{cohort}: workbook constraint sheet disagrees with fresh checks")
        covered = set().union(*(nb[s] | {s} for s in new.index))
        if len(covered) != prov["n_covered"] or len(covered) != 240:
            raise ValueError(f"{cohort}: coverage disagrees with provenance or the 240-slide ceiling")
        for kind, frame in [("review_pool", new), ("alternates", alternates)]:
            csv_path = work / "review_pool" / f"{kind}_{cohort}_v3_20260928.csv"
            csv = pd.read_csv(csv_path, dtype={"slide_id": str}).set_index("slide_id")
            if list(csv.index) != list(frame.index):
                raise ValueError("CSV and workbook slide order disagree")
            record(csv_path)
        for path in (old_path, new_path, prov_path, cfg["matrix"], cfg["per_query"], work / "population.csv"):
            record(path)

        old_prov_path = (OUT / "ilp_representatives" / cohort / "review_pool" /
                         f"provenance_{cohort}_v2_20260925.json")
        old_prov = json.loads(old_prov_path.read_text())
        old_matrix_path = OUT / "similarity_matrix" / f"combined_similarity_matrix_{cohort}_inlib_uni2_ot_cosine.pkl"
        old_matrix = pd.read_pickle(old_matrix_path)
        old_nb = get_top_k_similar_slides(old_matrix, top_k=5)
        old_covered = set().union(*(old_nb[s] | {s} for s in old.index))
        if len(old_covered) != old_prov["n_covered"]:
            raise ValueError(f"{cohort}: v2 coverage disagrees with its provenance")
        pd.testing.assert_frame_equal(matrix, old_matrix.loc[pop.index, pop.index])
        record(old_prov_path)
        record(old_matrix_path)
        old_vote, old_hit = metrics(old, outcomes)
        new_vote, new_hit = metrics(new, outcomes)
        metrics(alternates, outcomes)
        old_ali = ali.reindex(old.index)
        new_ali = ali.reindex(new.index)
        row = dict(cohort=cohort, v2_n=len(old), v3_n=len(new),
                   retained=len(set(old.index) & set(new.index)),
                   removed=len(set(old.index) - set(new.index)), added=len(set(new.index) - set(old.index)),
                   v2_population=old_prov["population"], v3_population=len(pop),
                   v2_covered=old_prov["n_covered"], v3_covered=len(covered),
                   v2_ali_tier1=int((old_ali.selection_tier == "Tier1").sum()),
                   v3_ali_tier1=int((new_ali.selection_tier == "Tier1").sum()),
                   v2_ali_tier2=int((old_ali.selection_tier == "Tier2").sum()),
                   v3_ali_tier2=int((new_ali.selection_tier == "Tier2").sum()),
                   v2_classes=old.tumor_group.nunique(), v3_classes=new.tumor_group.nunique(),
                   v2_vote_correct_at5=old_vote, v3_vote_correct_at5=new_vote,
                   v2_retrieval_hit_at5=old_hit, v3_retrieval_hit_at5=new_hit)
        if cohort == "ccdi":
            row["v2_oct"] = int((prep.loc[old.index, "prep_dicom"] == "FROZEN_OCT").sum())
            row["v3_oct"] = int((prep.loc[new.index, "prep_dicom"] == "FROZEN_OCT").sum())
            row["v2_ffpe_selected"] = len(set(old.index) & set(pop.index))
            row["v2_ffpe_only_covered_on_v3_graph"] = len(set().union(*(
                nb[s] | {s} for s in old.index if s in nb)))
            earlier_path = PACKETS / "review_pool_ccdi_ffpe_v3_20260928.xlsx"
            earlier = read_pool(earlier_path)
            earlier_cov = set().union(*(nb[s] | {s} for s in earlier.index))
            row["earlier_ffpe_pool_retained"] = len(set(earlier.index) & set(new.index))
            row["earlier_ffpe_pool_covered"] = len(earlier_cov)
            row["earlier_ffpe_pool_vote_correct_at5"], row["earlier_ffpe_pool_hit_at5"] = metrics(earlier, outcomes)
            record(earlier_path)
            for kind, frame in [("pool", new), ("alternate", alternates)]:
                for slide, item in frame.iterrows():
                    raw = fresh_dicom_preparation(slide)
                    series_rows.extend(raw)
                    man = manifest.get(slide, "NOT_IN_MANIFEST_V3.1.0")
                    cached = prep.loc[slide, "prep_dicom"]
                    raw_ffpe = all(r["confirmed_ffpe"] for r in raw)
                    manifest_ok = man in {"Formalin fixed paraffin embedded (FFPE)", "NOT_IN_MANIFEST_V3.1.0"}
                    oct_evidence = ("OCT" in man or "FROZEN" in cached or
                                    any("freez" in r["fixative"].lower() for r in raw))
                    exposure.append(dict(slide_id=slide, kind=kind, tumor_group=item.tumor_group,
                                         source=item.source, prep_cached=cached, prep_manifest_current=man,
                                         n_local_series=len(raw),
                                         fixatives=";".join(sorted({r["fixative"] for r in raw})),
                                         embeddings=";".join(sorted({r["embedding"] for r in raw})),
                                         oct_evidence=oct_evidence,
                                         confirmed_ffpe=raw_ffpe and manifest_ok and cached == "FFPE"))
        comparison.append(row)
        changes = []
        for slide in sorted(set(old.index) | set(new.index)):
            in_old, in_new = slide in old.index, slide in new.index
            frame = new if in_new else old
            changes.append(dict(slide_id=slide, tumor_group=frame.loc[slide, "tumor_group"],
                                change="retained" if in_old and in_new else "added" if in_new else "removed",
                                v2_source=old.loc[slide, "source"] if in_old else "",
                                v3_source=new.loc[slide, "source"] if in_new else "",
                                preparation=prep.loc[slide, "prep_dicom"] if cohort == "ccdi" else "not independently audited"))
        pd.DataFrame(changes).to_csv(RESULTS / f"pool_changes_{cohort}.csv", index=False)
        for group in sorted(set(old.tumor_group) | set(new.tumor_group)):
            class_rows.append(dict(cohort=cohort, tumor_group=group,
                                   v2=int((old.tumor_group == group).sum()),
                                   v3=int((new.tumor_group == group).sum())))

    audit = pd.DataFrame(exposure)
    audit.to_csv(RESULTS / "ccdi_v3_preparation_audit.csv", index=False)
    pd.DataFrame(series_rows).to_csv(RESULTS / "ccdi_v3_dicom_headers.csv", index=False)
    if len(audit) != 48 or not audit.confirmed_ffpe.all() or audit.oct_evidence.any():
        raise ValueError("CCDI v3 preparation audit failed; inspect the saved audit rows")
    pd.DataFrame(comparison).to_csv(RESULTS / "comparison_v2_v3.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RESULTS / "class_comparison_v2_v3.csv", index=False)
    (RESULTS / "input_sha256.json").write_text(json.dumps(hashes, indent=2))
    print(pd.DataFrame(comparison).to_string(index=False))
    print("\nFresh CCDI audit:\n", audit.groupby(["kind", "confirmed_ffpe"]).size().to_string())
    print("\nManifest:\n", audit.groupby(["kind", "prep_manifest_current"]).size().to_string())
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
