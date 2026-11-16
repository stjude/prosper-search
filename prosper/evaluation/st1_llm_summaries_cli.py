"""ST1.7 (spec 0002): production LLM case summaries for the CCDI shortlist packets.

Same deployed pathway as S5 (`llm_grounding_cli`): the vendored summariser's
`attempt_llm_summary_from_payload` on payloads built by the deployed
`analysis_lib.build_json_for_slide`, audited by the S5 grounding auditor. Two
arms per query:

  packet : `blind_query(base)` -- the reader-study variant; the query's own
           tumor_group/diagnosis/discrepant/evaluation are nulled (the deployed
           UI does not know them either).
  qc     : the unblinded base payload, for internal reading only.

Differences from S5, all deliberate:
  * queries come from a manifest (the 22 shortlist cases), not stratified
    sampling;
  * the matrix is the RECTANGULAR external block (queries x COMET library),
    loaded directly with pandas -- `load_results_new` would misread a
    non-square frame as an embedding matrix and rebuild L1 distances;
  * `load_metadata(min_site_count=0)` -- the default 40 silently drops DSRCT
    (31 library slides) and with it three shortlist cases' neighbour metadata;
  * query-side metadata rows (CCDI cases, absent from the COMET rich table)
    are appended from --query-rows, already in the rich schema; their CCDI
    participant ids never go through the numeric-id cast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from .llm_grounding_cli import (SUMMARIZERS, _commit, _import_pipeline, audit, blind_query,
                                load_env_file, summarise, summarize_payload, summarizer_info)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=Path, required=True,
                   help="CSV with slide_id (+ tumor_group) -- the shortlist manifest")
    p.add_argument("--matrix", type=Path, required=True,
                   help="rectangular query x library distance block (.pkl DataFrame)")
    p.add_argument("--metadata", type=Path, required=True, help="the COMET rich metadata CSV")
    p.add_argument("--query-rows", type=Path, required=True,
                   help="rich-schema metadata rows for the query slides")
    p.add_argument("--backfill-labels", type=Path, default=None,
                   help="canonical Slide.Scan.File/Tumor.group CSV; library slides "
                        "absent from the rich table (28 of the 2,147 slides that "
                        "appear in any CCDI top-10) get a label-only row so a "
                        "packet neighbour is never entity-less")
    p.add_argument("--ali-map", type=Path, default=None,
                   help="Ali metadata CSV (Slide.Scan.File -> Comp.Bio.ID); gives the "
                        "backfilled label-only rows a citable sample id instead of the "
                        "numeric scan id (citation rule C9; same fallback the packet "
                        "builder uses, first row wins)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--arms", default="packet,qc")
    p.add_argument("--summarizer", choices=SUMMARIZERS, default="manuscript",
                   help="manuscript = single-shot prosper/llm/summarizer (default); agentic = comet-llm LangGraph branch")
    p.add_argument("--bayesian-n", type=int, default=None,
                   help="attach Bayesian disease probabilities (prosper.bayesian.compute_disease_probs "
                        "over the query's row of the rectangular block, library labels from the same "
                        "metadata/backfill map as the neighbour panel) using this many neighbours; "
                        "omitted = no 'probabilities' key, as in the v3 run")
    p.add_argument("--bayesian-t", type=float, default=20.0,
                   help="Bayesian temperature (T=20 reproduces Supplementary Tables S11/S13)")
    p.add_argument("--exclude-groups", nargs="*", default=[], metavar="GROUP",
                   help="drop library columns whose tumor_group is one of these (e.g. NORMAL MISC) "
                        "BEFORE the payload top-k and the Bayesian posterior, so both see the same "
                        "reference library as the reader-suite panel; every column must be labelled")
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def _norm_group(x) -> str:
    return str(x).upper().replace(" ", "_")


def exclude_library_groups(sim: pd.DataFrame, true_of: dict, groups: list) -> tuple[pd.DataFrame, dict]:
    """Drop the library columns of `sim` whose label is in `groups`. Returns (filtered sim, info).

    An unlabelled column cannot be classified as kept or dropped, so it is an error whenever
    an exclusion is requested (same rule as bayesian_probabilities)."""
    excl = {_norm_group(g) for g in groups}
    cols = [str(c) for c in sim.columns]
    if not excl:
        return sim, {"groups": [], "columns_total": len(cols), "columns_kept": len(cols), "dropped": {}}
    missing = [c for c in cols if c not in true_of or pd.isna(true_of[c])]
    if missing:
        raise SystemExit(f"{len(missing)} library columns have no tumor_group label "
                         f"(first: {missing[:5]}); cannot apply --exclude-groups")
    dropped: dict = {}
    keep = []
    for c in cols:
        g = _norm_group(true_of[c])
        if g in excl:
            dropped[g] = dropped.get(g, 0) + 1
        else:
            keep.append(c)
    unseen = sorted(excl - set(dropped))
    if unseen:
        raise SystemExit(f"--exclude-groups {unseen} match no library column; check the spelling "
                         f"(labels present: {sorted({_norm_group(true_of[c]) for c in cols})})")
    return sim.loc[:, keep], {"groups": sorted(excl), "columns_total": len(cols),
                              "columns_kept": len(keep), "dropped": dropped}


def bayesian_probabilities(sim: pd.DataFrame, queries: list, true_of: dict, n: int, T: float,
                           out: Path) -> dict:
    """Per-query {class: probability} (non-zero, descending) from the rectangular block.

    Every library column must carry a label: an unlabelled column would silently become
    its own 'UNKNOWN' class, so it is an error here rather than a default."""
    from prosper.bayesian import compute_disease_probs

    cols = [str(c) for c in sim.columns]
    missing = [c for c in cols if c not in true_of or pd.isna(true_of[c])]
    if missing:
        raise SystemExit(f"{len(missing)} library columns have no tumor_group label "
                         f"(first: {missing[:5]}); Bayesian probabilities need every column labelled")
    labels = [str(true_of[c]) for c in cols]
    dicts, df = compute_disease_probs(sim.loc[queries].to_numpy(dtype=float), labels,
                                      n_neighbors=n, T=T, test_slide_ids=queries)
    df.to_csv(out / "disease_probabilities.csv")
    # 3 dp in the payload: the model quotes whatever precision it is given (a 6-dp payload produced
    # "0.516809" in the text); full precision stays in the CSV.
    return {q: {d: round(p, 3) for d, p in sorted(dd.items(), key=lambda x: -x[1]) if round(p, 3) > 0}
            for q, dd in zip(queries, dicts)}


def resumable_done(runs_path: Path, settings: dict) -> set:
    """(query, arm) pairs already in runs.jsonl that may be skipped on resume. Every existing record
    must carry exactly the current `settings`; a record from another configuration (or a legacy
    record without `settings`) aborts, because resuming would keep its payload/text in place."""
    done = set()
    if runs_path.exists():
        for i, line in enumerate(runs_path.read_text().splitlines(), 1):
            r = json.loads(line)
            if r.get("settings") != settings:
                raise SystemExit(f"{runs_path} line {i} was produced under settings {r.get('settings')} "
                                 f"!= current {settings}; use a fresh --out-dir instead of resuming")
            done.add((r["query"], r["arm"]))
    return done


def load_label_map(analysis_lib, metadata: Path, query_rows: Path, backfill_labels: Path | None,
                   ali_map: Path | None) -> tuple[pd.DataFrame, dict]:
    """(meta, {slide_id: tumor_group}) exactly as the summaries use them: rich library metadata +
    the query rows + label-only backfill for library slides missing from the rich file."""
    meta = analysis_lib.load_metadata(str(metadata), min_site_count=0)
    # The rich file lists some scanned slides under several samples (one row each).
    # `enrich_metadata` builds a dict keyed by slide_id, so without this the LAST duplicate row
    # silently supplied the cited Comp.Bio.ID/site/age. Keep the first row: it matches the co-author
    # map's first-row rule used by the reader-suite panel manifests (2,871/2,873 agree).
    n_before = len(meta)
    meta = meta.drop_duplicates("slide_id", keep="first")
    if len(meta) != n_before:
        print(f"[st1.7] rich metadata: {n_before - len(meta)} duplicate slide rows dropped (first kept)")
    qrows = pd.read_csv(query_rows)
    qrows["slide_id"] = qrows["slide_id"].astype(str)
    qrows["tumor_group"] = qrows["tumor_group"].map(
        lambda x: str(x).upper().replace(" ", "_"))
    meta = pd.concat([meta, qrows], ignore_index=True)
    if backfill_labels:
        lab = pd.read_csv(backfill_labels).rename(
            columns={"Slide.Scan.File": "slide_id", "Tumor.group": "tumor_group"})
        lab["slide_id"] = lab["slide_id"].astype(str)
        lab["tumor_group"] = lab["tumor_group"].map(
            lambda x: str(x).upper().replace(" ", "_"))
        add = lab[~lab["slide_id"].isin(set(meta["slide_id"].astype(str)))]
        keep_cols = ["slide_id", "tumor_group"] + (["Sample.ID"] if "Sample.ID" in add.columns else [])
        add = add[keep_cols].copy()
        if ali_map:
            ali = pd.read_csv(ali_map, dtype=str, low_memory=False)
            sj_of = (ali.dropna(subset=["Comp.Bio.ID", "Slide.Scan.File"])
                     .drop_duplicates("Slide.Scan.File", keep="first")
                     .set_index("Slide.Scan.File")["Comp.Bio.ID"].to_dict())
            add["Comp.Bio.ID"] = add["slide_id"].map(sj_of)
            print(f"[st1.7] Comp.Bio.ID resolved for {int(add['Comp.Bio.ID'].notna().sum())}"
                  f"/{len(add)} backfilled rows via --ali-map")
        if "Sample.ID" in add.columns:
            # same fallback as the reader-suite manifests (comp_bio_id or Sample.ID),
            # applied whether or not --ali-map was given
            if "Comp.Bio.ID" not in add.columns:
                add["Comp.Bio.ID"] = pd.Series(index=add.index, dtype=object)
            add["Comp.Bio.ID"] = add["Comp.Bio.ID"].fillna(add["Sample.ID"])
        add = add.drop(columns=[c for c in ("Sample.ID",) if c in add.columns])
        meta = pd.concat([meta, add], ignore_index=True)
        print(f"[st1.7] backfilled {len(add)} label-only library rows")
    true_of = dict(zip(meta["slide_id"].astype(str), meta["tumor_group"]))
    return meta, true_of


def run(a) -> dict:
    t0 = time.time()
    if a.fake:
        os.environ["PATHO_LLM_FAKE"] = "1"
    env_keys = load_env_file(a.env_file) if a.env_file else []
    analysis_lib, pll = _import_pipeline(a.summarizer)

    meta, true_of = load_label_map(analysis_lib, a.metadata, a.query_rows, a.backfill_labels, a.ali_map)

    sim = pd.read_pickle(a.matrix)
    sim.index = sim.index.astype(str)
    sim.columns = sim.columns.astype(str)
    if sim.shape[0] == sim.shape[1]:
        raise SystemExit("expected the RECTANGULAR external block, got a square matrix")

    queries = pd.read_csv(a.queries)["slide_id"].astype(str).tolist()
    missing = [q for q in queries if q not in sim.index]
    if missing:
        raise SystemExit(f"queries not in matrix: {missing}")
    n_columns_total = int(sim.shape[1])
    sim, excl_info = exclude_library_groups(sim, true_of, a.exclude_groups)
    if excl_info["groups"]:
        print(f"[st1.7] library columns {excl_info['columns_total']} -> {excl_info['columns_kept']} "
              f"after excluding {excl_info['dropped']}")

    arms = [x for x in a.arms.split(",") if x]
    out = a.out_dir
    for sub in ("payloads", "packet", "qc"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    probs_of = (bayesian_probabilities(sim, queries, true_of, a.bayesian_n, a.bayesian_t, out)
                if a.bayesian_n else {})
    if a.bayesian_n:
        print(f"[st1.7] Bayesian probabilities attached (N={a.bayesian_n}, T={a.bayesian_t}) "
              f"for {len(probs_of)} queries")
    runs_path = out / "runs.jsonl"
    # Settings every run record carries; a resume into an existing out-dir must match them exactly,
    # otherwise skipped (query, arm) pairs would keep payloads/texts produced under other settings
    # (e.g. a --fake pass, another k/N/T or library) while audit_summary.json reports the new ones.
    settings = {"fake": bool(a.fake), "k": int(a.k),
                "bayesian_n": int(a.bayesian_n or 0), "bayesian_t": float(a.bayesian_t) if a.bayesian_n else None,
                "exclude_groups": sorted(excl_info["groups"]), "columns_used": int(sim.shape[1]),
                "summarizer": a.summarizer}
    done = resumable_done(runs_path, settings)

    with open(runs_path, "a") as fh:
        for q in queries:
            base = analysis_lib.build_json_for_slide(sim, meta, slide_id=q, k=a.k)
            if q in probs_of:
                base["probabilities"] = probs_of[q]
            payloads = {"packet": blind_query(base), "qc": base}
            for arm in arms:
                if (q, arm) in done:
                    continue
                payload = payloads[arm]
                ph = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
                with open(out / "payloads" / f"{q}_{arm}.json", "w") as pf:
                    json.dump(payload, pf, indent=1)
                t1 = time.time()
                try:
                    text, accepted, tele = summarize_payload(pll, payload)
                    err = None
                except Exception as e:                       # noqa: BLE001
                    text, accepted, tele, err = None, False, None, f"{type(e).__name__}: {str(e)[:200]}"
                claims = (tele or {}).get("claims") or [] if isinstance(tele, dict) else []
                rec = {"query": q, "true_group": true_of.get(q), "arm": arm, "repeat": 0,
                       "payload_sha": ph, "k": a.k, "counterfactual": None, "settings": settings,
                       "accepted": bool(accepted), "error": err,
                       "telemetry": {k: v for k, v in (tele or {}).items()
                                     if k != "tool_calls"} if isinstance(tele, dict) else None,
                       "wall_s": round(time.time() - t1, 2), "text": text,
                       "audit": audit(text, claims, payload, true_of.get(q))}
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                if text:
                    (out / arm / f"{q}.md").write_text(text)

    recs = [json.loads(line) for line in runs_path.read_text().splitlines()]
    summary = summarise(recs)
    summary["config"] = {"matrix": str(a.matrix), "metadata": str(a.metadata),
                         "query_rows": str(a.query_rows), "k": a.k,
                         "n_queries": len(queries), "arms": arms, "fake": bool(a.fake),
                         "library": {"columns_total": n_columns_total, "columns_used": int(sim.shape[1]),
                                     "exclude_groups": excl_info["groups"], "dropped": excl_info["dropped"]},
                         "bayesian": ({"n_neighbors": a.bayesian_n, "temperature": a.bayesian_t,
                                       "reference_columns": int(sim.shape[1]),
                                       "table": str(out / "disease_probabilities.csv")}
                                      if a.bayesian_n else None),
                         "env_keys_loaded": env_keys, "prosper_commit": _commit(),
                         "ali_map": str(a.ali_map) if a.ali_map else None,
                         **summarizer_info(pll, a.summarizer),
                         "elapsed_s": round(time.time() - t0, 1)}
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    print(json.dumps(run(a), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
