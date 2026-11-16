#!/usr/bin/env python3
"""Supplementary Table S30 (R3.2-T1): observed hit@5 beside the hypergeometric chance hit@5, by tumour group and
collection (internal, STV, KidRare, CCDI-MCI FFPE).

Specification: docs/manuscript_supplement_edits.md, block R3.2-T1 (batch 2, 10-06). This script only COPIES
observed values from the two accepted sources and ADDS the closed-form chance value; it runs no retrieval and no
resampling.

    chance hit@5 for group g = 1 - C(N - n_g, 5) / C(N, 5),   N = 2,778 evaluated library slides,

i.e. the probability that five slides drawn at random without replacement from the library include at least one
slide of group g. For internal queries the query's own patient is excluded from retrieval, so the full-library
value is an approximation (stated in the caption); external collections query the same 2,778-slide eligible
library.

Sources (full sha256 pinned; the run refuses to proceed on mismatch; each file is parsed from the hashed bytes):
  S23  $PROSPER_OUT/r2_3_internal_failure_cis_20261004/per_group_summary.csv   3248fd00...  (metric == hit5)
  E20  $PROSPER_OUT/r1_3_external_synthesis_20260930/broad_group_primary.csv   4cc93dfb...
       (configuration == whole_slide_ot, analysis_role == main, metric == hit_at_5)
  E20  $PROSPER_OUT/r1_3_external_synthesis_20260930/coverage_and_hit_yield.csv 39e40a3a...
       (same filter, metric in {coverage, hit_yield}; exactly one row of each is required per primary row)
Markers in the wide table: † = external sparse resampling units and/or degenerate interval (E20 flag);
‡ = internal interval with upper bound 1 (S23 boundary_flag = upper_eq_1; not sparse/degenerate in the E20 sense).
Outputs ($PROSPER_OUT/r3_2_s30_chance_<date>/): table_s30.csv (long), table_s30_wide.md (paste-ready),
run_record.json (command, environment, source hashes, N, per-group n_g, chance values, row counts, script hash).
Independent review: opus_s30_review_001/[internal record] (ACCEPT WITH FIXES F1-F10, all applied here except the
optional per-query masked chance; F3's bound "overstates by at most 0.005" is stated in the footnote).
"""
import argparse, csv, datetime as dt, hashlib, io, json, math, os, platform, socket, sys

GROUPS = ["ACT/ACC", "DSRCT", "EWS", "LIVER", "MEL", "MPNST", "MRT", "NB", "NF", "NORMAL", "OS", "RB", "RMS", "SS", "WT"]
N_LIBRARY = 2778
K = 5
SOURCES = {
    "s23": ("r2_3_internal_failure_cis_20261004/per_group_summary.csv",
            "3248fd009fcb082f04d1a437cbab25c844e95e88025f05a50e54759b6f1ce62f"),
    "e20_primary": ("r1_3_external_synthesis_20260930/broad_group_primary.csv",
                    "4cc93dfb9acbf6abfd879353e34609d73887669edabcd9c26a2bdeb3a98b3b74"),
    "e20_coverage": ("r1_3_external_synthesis_20260930/coverage_and_hit_yield.csv",
                     "39e40a3ad7dafab869a01ff6dcd841c37ef620928b897db41cd0f6f1465e0f62"),
}
COLLECTIONS = [
    # (label, cohort key in E20, population, resampling-unit wording for the caption/notes, unit noun for cells)
    ("internal", None, None, "whole patient (joint resampling, S23)", "patients"),
    ("STV", "stv", "main_696", "supplied ST_ patient cluster", "patients"),
    ("KidRare", "kidrare", "amended_1581", "nominal slide; no patient-level independence claim", "slides"),
    ("CCDI-MCI FFPE", "ccdi", "ffpe_scope_clear", "single supplied participant", "participants"),
]
NA = "not available"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def chance_hit_at_k(n_g, N=N_LIBRARY, k=K):
    """P(at least one of k draws without replacement is in group g) = 1 - C(N-n_g,k)/C(N,k)."""
    if n_g <= 0:
        return 0.0
    if N - n_g < k:
        return 1.0
    return 1.0 - math.comb(N - n_g, k) / math.comb(N, k)


def read_hashed_csv(p, expect):
    """Hash and parse the SAME bytes (no gap between verification and reading)."""
    with open(p, "rb") as f:
        data = f.read()
    h = hashlib.sha256(data).hexdigest()
    if h != expect:
        sys.exit(f"source hash mismatch for {p}: {h} != {expect}")
    return h, list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


def fmt(x, nd=3):
    return "" if x in (None, "") else f"{float(x):.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output dir (default $PROSPER_OUT/r3_2_s30_chance_<YYYYMMDD>)")
    a = ap.parse_args()
    root = os.environ.get("PROSPER_OUT")
    if not root:
        sys.exit("PROSPER_OUT not set")
    out = a.out or os.path.join(root, f"r3_2_s30_chance_{dt.date.today():%Y%m%d}")
    if os.path.exists(out):
        sys.exit(f"refusing to overwrite existing {out}")

    paths, hashes, tables = {}, {}, {}
    for key, (rel, expect) in SOURCES.items():
        p = os.path.join(root, rel)
        hashes[key], tables[key] = read_hashed_csv(p, expect)
        paths[key] = p

    s23 = [r for r in tables["s23"] if r["metric"] == "hit5"]
    n_g = {r["true_label"]: int(r["n_queries"]) for r in s23}
    if sorted(n_g) != sorted(GROUPS) or sum(n_g.values()) != N_LIBRARY:
        sys.exit(f"S23 hit5 rows do not define the {N_LIBRARY}-slide library: {n_g}")
    internal = {r["true_label"]: r for r in s23}

    def e20_filter(rows, metric):
        return [r for r in rows if r["configuration"] == "whole_slide_ot" and r["analysis_role"] == "main"
                and r["metric"] == metric and r["tumor_group"] in GROUPS]

    prim = e20_filter(tables["e20_primary"], "hit_at_5")
    cov_rows = tables["e20_coverage"]
    cov = e20_filter(cov_rows, "coverage"); hy = e20_filter(cov_rows, "hit_yield")

    def pick(rows, cohort, pop, g):
        m = [r for r in rows if r["cohort"] == cohort and r["population"] == pop and r["tumor_group"] == g]
        if len(m) > 1:
            sys.exit(f"ambiguous E20 rows for {cohort}/{pop}/{g}")
        return m[0] if m else None

    chance = {g: chance_hit_at_k(n_g[g]) for g in GROUPS}
    long_rows = []
    for label, cohort, pop, unit, noun in COLLECTIONS:
        for g in GROUPS:
            row = dict(collection=label, tumor_group=g, n_library_slides=n_g[g],
                       library_share=n_g[g] / N_LIBRARY, chance_hit_at_5=chance[g],
                       n_queries="", n_resampling_units="", resampling_unit_noun=noun, observed_hit_at_5="",
                       ci_lo="", ci_hi="", internal_boundary_flag="", external_sparse_units="",
                       external_degenerate_interval="", external_sparse_or_degenerate="",
                       coverage="", hit_yield="", hit_yield_ci_lo="", hit_yield_ci_hi="",
                       resampling_unit=unit, source="")
            if cohort is None:
                r = internal[g]
                row.update(n_queries=int(r["n_queries"]), n_resampling_units=int(r["n_patients"]),
                           observed_hit_at_5=float(r["point"]), ci_lo=float(r["ci_lo"]), ci_hi=float(r["ci_hi"]),
                           internal_boundary_flag=(r.get("boundary_flag") or ""),
                           source="S23 per_group_summary.csv hit5")
            else:
                r = pick(prim, cohort, pop, g)
                if r is None:
                    row.update(observed_hit_at_5=NA, source=f"E20 {cohort}/{pop}: no queries of this group")
                else:
                    c = pick(cov, cohort, pop, g); y = pick(hy, cohort, pop, g)
                    if c is None or y is None:
                        sys.exit(f"E20 primary row {cohort}/{pop}/{g} lacks its coverage or hit_yield row")
                    row.update(n_queries=int(r["n"]), n_resampling_units=int(r["n_resampling_units"]),
                               observed_hit_at_5=float(r["point"]), ci_lo=float(r["ci_lo"]), ci_hi=float(r["ci_hi"]),
                               external_sparse_units=r["sparse_units"],
                               external_degenerate_interval=r["degenerate_interval"],
                               external_sparse_or_degenerate=r["sparse_or_degenerate"],
                               coverage=float(c["point"]), hit_yield=float(y["point"]),
                               hit_yield_ci_lo=float(y["ci_lo"]), hit_yield_ci_hi=float(y["ci_hi"]),
                               source="E20 broad_group_primary.csv + coverage_and_hit_yield.csv (whole_slide_ot/main)")
            long_rows.append(row)
    if not all(r["coverage"] == 1.0 for r in long_rows if r["coverage"] != ""):
        sys.exit("external coverage is not 1 for every evaluated group; footnote wording must change")

    os.makedirs(out)
    with open(os.path.join(out, "table_s30.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0])); w.writeheader(); w.writerows(long_rows)

    # paste-ready wide markdown: one row per group; per collection "observed [lo, hi] (n queries; units)" or NA
    md = ["| Group | n in library | Share | Chance hit@5 | " + " | ".join(c[0] for c in COLLECTIONS) + " |",
          "|---|---:|---:|---:|" + "|".join(["---"] * len(COLLECTIONS)) + "|"]
    for g in GROUPS:
        cells = []
        for label, *_ in COLLECTIONS:
            r = next(x for x in long_rows if x["collection"] == label and x["tumor_group"] == g)
            if r["observed_hit_at_5"] == NA:
                cells.append(NA)
            else:
                flag = ""
                if r["external_sparse_or_degenerate"] == "True":
                    flag = "†"
                elif r["internal_boundary_flag"] == "upper_eq_1":
                    flag = "‡"
                nq = r["n_queries"]; nu = r["n_resampling_units"]; noun = r["resampling_unit_noun"]
                def pl(n, word):
                    return f"{n} {word if int(n) != 1 else word[:-1]}"
                units = pl(nq, "slides") if label == "KidRare" else f"{pl(nq, 'slides')}; {pl(nu, noun)}"
                cells.append(f"{fmt(r['observed_hit_at_5'])} [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] (n={units}){flag}")
        md.append(f"| {g} | {n_g[g]} | {fmt(n_g[g] / N_LIBRARY)} | {fmt(chance[g])} | " + " | ".join(cells) + " |")
    md.append("")
    bnd = [int(internal[g]["n_patients"]) for g in GROUPS if internal[g].get("boundary_flag") == "upper_eq_1"]
    bnd_txt = f"{min(bnd)}-{max(bnd)} patients" if bnd else "n/a"
    md.append("Observed hit@5 is copied from the accepted source tables and shown as point [95% interval] "
              "(n = query slides; resampling units). Intervals are nominal 95%, unadjusted, conditional on the fixed "
              "library; resampling unit per collection: " + "; ".join(f"{c[0]} = {c[3]}" for c in COLLECTIONS) + ". "
              "† external cell flagged in the source table as sparse resampling units and/or a degenerate (boundary) "
              "interval; ‡ internal interval whose upper bound is 1 (S23 boundary flag; these groups have " + bnd_txt +
              " and are not sparse). External cells with an upper bound of 1.000 but no † were not flagged "
              "in the source. "
              f"Chance hit@5 = 1 − C({N_LIBRARY:,} − n_g, {K}) / C({N_LIBRARY:,}, {K}), the probability that five slides "
              f"drawn at random without replacement from the {N_LIBRARY:,}-slide evaluated library include at least one "
              "slide of the group. For internal queries the query's own patient is excluded from retrieval, so this "
              "full-library value overstates each group's internal chance value by at most 0.005 on average over its "
              "queries (independent review bound 0.001-0.004; at most 0.007 for any single query). External "
              f"collections query the same {N_LIBRARY:,}-slide eligible library. Coverage "
              "was 1 for every evaluated external group, so hit yield equals hit@5 (see table_s30.csv). "
              "'not available' = no queries of this group in the evaluated population of that collection.")
    with open(os.path.join(out, "table_s30_wide.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    rec = dict(generated=dt.datetime.now().isoformat(timespec="seconds"), script=os.path.abspath(__file__),
               script_sha256=sha256_file(__file__), command=[sys.executable] + sys.argv, out=os.path.abspath(out),
               cwd=os.getcwd(), hostname=socket.gethostname(), python=platform.python_version(),
               platform=platform.platform(), prosper_out=root,
               n_library=N_LIBRARY, k=K, sources={k: dict(path=paths[k], sha256=hashes[k]) for k in paths},
               n_g=n_g, chance_hit_at_5=chance, n_rows=len(long_rows),
               n_not_available={c[0]: sum(1 for r in long_rows if r["collection"] == c[0] and r["observed_hit_at_5"] == NA) for c in COLLECTIONS},
               coverage_all_one={c[0]: all(r["coverage"] in ("", 1.0) for r in long_rows if r["collection"] == c[0]) for c in COLLECTIONS},
               outputs={n: sha256_file(os.path.join(out, n)) for n in ("table_s30.csv", "table_s30_wide.md")})
    with open(os.path.join(out, "run_record.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("n_rows", "n_not_available", "coverage_all_one")}))
    print("\n".join(md)); print("out", out)


if __name__ == "__main__":
    main()
