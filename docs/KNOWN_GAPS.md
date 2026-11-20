# Known gaps

This repository releases the code that produced the results of the revised
manuscript, but it is **not** an end-to-end reproduction package. This file
lists what is not released, what cannot be rerun from the repository, and what
could not be verified. [REPRODUCTION.md](REPRODUCTION.md) gives the per-display
consequences.

## 1. Files that are not released (held)

These files were executed but contain case-level identifiers, captured cluster
records or identifier lookups that cannot be removed by a purely mechanical
substitution without changing the executed content. They are withheld.

| Held file (repository-relative location it would have) | Reason | Consequence |
|---|---|---|
| `revision/scripts/llm_reader80/trace_readback_reader80.py` | A hand-transcribed check list pairing 22 slide identifiers (not printed in the manuscript) with payload values; removing them would remove the check itself | Its check output, read by `report_reader80.py`, cannot be regenerated. |
| `tests/test_submit_r2_1_route_b_uni2.py` and its three fixtures under `tests/fixtures/r2_1_route_b_uni2_readback/` | Captured cluster job listings and submission records (user, queue, hosts, internal paths, job IDs) | That submission-readback test is absent. |
| Reader-study entity map (`docs/pathologist_study/reader_study_misc_entity_map_20261027.csv` in the executed workspace, read by `revision/scripts/reader_study/analyze.py` and `build_readings.py`) | Reader-study material: diagnoses as written by readers and their mapping to tumor groups | The reader-study analysis needs it at that path; the reader-study tests that load it are skipped without it (see [TESTING.md](TESTING.md)). |
| Reader-comment code sheet (`docs/pathologist_study/reader_study_comment_codes_final_20261116.csv` in the executed workspace, read by `revision/scripts/supp_tables/build_reader_comment_coding.py`) | Reader-study material: one row per reader comment with the codes assigned to it | The coded-comment sheets of Table S47 cannot be rebuilt without it. |

One further candidate builder for an alternative Fig. 6B panel was not
released because it was not adopted and its docstring named case identifiers;
the published Fig. 6B panel is the submitted one.

**Identifiers replaced in released files.** The executed instruction text of
the LLM summarizer contained two study identifiers as format examples; they
are replaced by synthetic identifiers in this release. The same synthetic
identifiers replace them in the two summarizer test files. The figure
builders read slide identifiers shown in figure labels, and a scan-to-slide
lookup, from a local, git-ignored file (`revision/config/figure_ids.local.json`);
the shipped example contains synthetic identifiers only. The scan-to-slide
table itself is not released.

## 2. Steps not in the repository

- **Post-assembly packaging of the supplementary tables.** After
  `assemble_final_tables_20261110.py` wrote the per-table workbooks, further
  packaging steps edited cell text and labels and removed sheets before the
  six Supplementary Data files were written: provenance labels were rewritten
  as plain descriptions, internal codes, paths and checksums were removed,
  headers were made readable and some labels were renamed (for example
  "majority" to "plurality"). A recorded diff of the first step lists 8,434
  edited cells; four of them replaced numeric cells in one Table S25 sheet by
  descriptions of the populations. The second step changed 25 text cells and no
  numeric cells, and added the two reader-summary rating sheets of Table S48,
  which are built by the released `revision/scripts/supp_tables/reader80_sheets.py`.
  Later steps made further logged label and text edits (no numeric value
  changed), reduced the workbooks to
  the sheets that the Supplementary Information cites (rows and columns of
  some sheets were trimmed, and four Table S25 endpoint sheets were assembled
  by copying values from its key-value sheets), and regenerated the caption
  sheets from the Supplementary Information. These packaging scripts and their
  caption source are not released because they embed internal paths and
  process annotations in their assertions, or work on case-level sheets that
  are not distributed. The two steps that come after them are released under
  their executed names in `revision/retained_executed/`: the S49 builder
  (`collab_draft_20261102/checks_050/reproduction/make_s49_pass050.py`) and
  the grouping step that wrote the six delivered files
  (`collab_draft_20261102/checks_051/reproduction/build_supplementary_data_workbooks_051.py`
  with its layout and the two helpers under
  `e1_si_close_audit_20261115/scratch/si-tables/`). The released
  `revision/scripts/supp_tables/build_supplementary_data_workbooks.py` with
  `supplementary_data_layout.json` is the earlier grouping step (47 tables).
  Consequently the released code reproduces the table values but not the
  exact delivered workbook files.
- **Fig. 8 output filter.** The delivered Fig. 8 and Supplementary Fig. S14
  were rendered by the released `fig8_reader_study.py` through a small wrapper
  that only redirected the output folder, dropped the `reader_id` column from
  the source-data files and removed reader logins from the provenance note.
  The wrapper is not released; run directly, the figure script writes both.
- **Caption and specification documents.** The table builders and several
  figure builders read the Supplementary Information legend source and an
  internal table specification for captions and consistency checks. These
  documents are not released.
- **Earlier COMET pipelines.** Most similarity matrices of the encoder and
  aggregation grid, and the mosaics, were computed by the earlier COMET
  pipelines (mosaic generation, SAMPLER, Bag-of-Barcodes, Optimal Transport,
  median pooling and slide-feature similarity scripts). These are not
  included. Equivalent algorithms are in `prosper/`;
  `prosper/embeddings/aggregate_cli.py` reproduced the median-pooling
  slide-feature matrix exactly for one encoder (CTransPath, 2,921 slides,
  maximum absolute difference 0), but not every grid cell was re-derived.
- **Comparator input manifests.** A run-folder script that built the input
  manifests of the comparator feature banks is not included.
- **RetCCL tile encoder.** No RetCCL module exists in `prosper/embeddings/`;
  RetCCL results are evaluated by `revision/scripts/r3_1_retccl_whole_patient.py`
  from features computed elsewhere.
- **Cluster wrappers and checks.** Site-specific batch-job wrappers and
  run-folder verification/acceptance scripts that do not produce reported
  values are not included. The submission helpers in `scripts/lsf/` are
  specific to an LSF cluster.
- **Fig. 1** edits the submitted figure image, which is not redistributed
  as a source; the revised Fig. 1 schematic is included as
  `data/fig/overview.png`.

## 3. Data that are not released

No images, tile features, similarity matrices, ranked lists, labels, case
metadata, crosswalks, reader-study records, LLM payloads or responses, or
summaries shown to readers are in this repository. See
[DATA_ACCESS.md](DATA_ACCESS.md). The reader-study code needs a local reader
configuration (`revision/config/README.md`); the shipped example contains fake
logins and is sufficient only for the unit tests.

## 4. Things that could not be verified here

- No analysis was rerun against real data while preparing this release. The
  checks were: unit tests that need no restricted data, byte-compilation of
  all Python files, and a scan for identifiers, credentials and internal paths.
- Correspondence between released files and executed files is by checksum
  ([CODE_MANIFEST.tsv](CODE_MANIFEST.tsv)). Files with a logged substitution
  differ from the executed bytes only in those substitutions.
- Software environments are recorded only for some runs
  ([ENVIRONMENT.md](ENVIRONMENT.md)); there is no complete lock file for every
  analysis, and historical GPU environments are not reconstructed.
- LLM generation is not deterministic; temperature and seed were not set.
- **Checksum pins.** Many scripts and tests verify the sha256 of other
  executed files before running. Where the target was edited for the release
  (a logged substitution), the pin was updated to the released checksum so
  that the released scripts accept their released siblings.
  [INTEGRITY_PINS.tsv](INTEGRITY_PINS.tsv) lists each such pin with both the
  executed and the released checksum; the executed value can be compared with
  the `sha256_executed` column of [CODE_MANIFEST.tsv](CODE_MANIFEST.tsv) and
  with the original run records.
- **Absolute paths.** Many scripts and tests read inputs and their own
  sources at the executed absolute locations (shown with the placeholders of
  [PATHS.md](PATHS.md)); they fail unless those locations exist.
- **Test suites.** Without the controlled data many integration tests cannot
  run; they are skipped with a stated reason unless the site tests are
  enabled. [TESTING.md](TESTING.md) lists how to run the tests, every skip
  reason and the counts observed while preparing this release.
- Some test modules check properties of the original computing environment
  (for example a cluster mount point or pinned library versions) and are
  skipped elsewhere.
- The reader-study code is the version executed on the frozen study export;
  if that export is superseded, the reader-study tables and Fig. 8 must be
  rebuilt with this code.
