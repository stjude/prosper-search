# Revision analysis, figure and table code

This directory holds the analysis, figure and supplementary-table code that was
executed for the revised PROSPER manuscript, in addition to the `prosper`
package at the repository root. It is released so that readers can inspect how
each reported number, figure and table was produced. Most scripts need
controlled-access COMET/STV data or intermediate outputs that are not part of
this repository; see [`../docs/REPRODUCTION.md`](../docs/REPRODUCTION.md) for
the display-by-display map and [`../docs/KNOWN_GAPS.md`](../docs/KNOWN_GAPS.md)
for what cannot be rerun.

The files are the executed versions. The only edits are the mechanical,
logged substitutions listed in [`../docs/CODE_MANIFEST.tsv`](../docs/CODE_MANIFEST.tsv)
(workspace path prefixes, reader logins, a few slide identifiers moved to a
local lookup file, credentials/endpoints, internal notes in comments and
docstrings, and checksum pins updated to the released bytes of the files they
pin, listed in [`../docs/INTEGRITY_PINS.tsv`](../docs/INTEGRITY_PINS.tsv));
analysis logic was not changed.

## Layout

The directory mirrors the layout in which the code was run, because many
scripts locate siblings and inputs relative to their own position
(`Path(__file__).parents[N]`) or import neighbouring modules.

| Path | Role |
|---|---|
| `scripts/*.py` | Analyses: benchmark grid and comparators, patient-excluded metrics and whole-patient intervals, tumor-filtering held-out pipeline, Bayesian held-out evaluation, perturbation readbacks, additional collections (CCDI-MCI, KidRare), Optimal Transport localization, TITAN/CONCH comparisons, fine-label and cohort accounting |
| `scripts/figures/` | Builders for main and supplementary figures |
| `scripts/supp_tables/` | Builders for the supplementary tables and the grouped Supplementary Data workbooks |
| `scripts/reader_study/` | Reader-study analysis pipeline (derivation, endpoints, paired comparisons, QC sensitivity) and its tests |
| `scripts/llm_pilot_round2/` | LLM summary pilot: payload construction, generation harness (executed copy), readback and analysis |
| `scripts/llm_reader80/` | Rating of the 80 reader-study summaries and its report |
| `scripts/roi_search_6cfg/` | Region-of-interest search under six encoder/similarity configurations |
| `scripts/r2_2_all_eligible/` | All-eligible-tile versus tile-mosaic comparison, including resource profiling |
| `scripts/slide_selector/` | Integer linear programming representative-subset selection (top-k neighbor coverage) |
| `scripts/ilp_representatives/` | Two-stage representative selection of reader-study cases with the slide selector |
| `scripts/test_*.py`, `tests/` | Unit tests for the analysis scripts |
| `retained_executed/` | Byte-retained copies of code versions that differ from the current files but produced reported results (e.g. the stage-specific perturbation-suite modules); paths follow the original output-folder layout |
| `config/` | Local configuration templates (see below) |

## File naming

Analysis file names keep the work-package codes used during the revision
(for example `r2_1_`, `r3_5_`, `r4a_`, `e1_`, `c3_`, `s6_`, `st2_`). They are
internal identifiers kept so that the names match the execution records; they
carry no meaning for using the code. The mapping from displays to files is in
[`../docs/REPRODUCTION.md`](../docs/REPRODUCTION.md).

## Reader-study configuration

The reader-study code needs the study-server logins of the participating
pathologists. These are personal data and are not released. Create
`config/readers.local.json` (git-ignored) from
[`config/readers.example.json`](config/readers.example.json) as described in
[`config/README.md`](config/README.md), or point `PROSPER_READERS_CONFIG` at
such a file. Without it, the code falls back to the example file, which contains
fake logins and is sufficient only for the unit tests.

## Figure slide identifiers

A few figure builders check or print de-identified slide identifiers. These are
read from `config/figure_ids.local.json` (git-ignored) or the file named by
`PROSPER_FIGURE_IDS`; without it the builders fall back to
[`config/figure_ids.example.json`](config/figure_ids.example.json), which holds
synthetic identifiers. See [`config/README.md`](config/README.md).

## Output root

Scripts write under `$PROSPER_OUT`. Set it before running anything, e.g.
`export PROSPER_OUT="$PWD/prosper_revision_output"`. Absolute workspace paths
in the code were replaced by placeholders such as `/PROSPER_WORKSPACE`; see
[`../docs/PATHS.md`](../docs/PATHS.md).
