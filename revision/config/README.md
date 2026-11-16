# Local configuration

`readers.local.json` (git-ignored, never committed) maps the reader-study logins
used by the study server to the roles the analysis needs. The released code carries
no reader logins; `revision/scripts/reader_study/reader_config.py` loads this file,
or the file named by `PROSPER_READERS_CONFIG`, and falls back to
`readers.example.json` (fake logins, sufficient for the unit tests only).

Keys (all required except `export_checks`):

| key | meaning |
|---|---|
| `fig8_reader_order` | logins in the order shown as Reader 1–5 in Fig. 8 and the reader-study tables |
| `expertise_tier` | login → `"expert"` or `"general"`; readers missing here are not analysed |
| `excluded_readers` | logins excluded by name (developer and test accounts); logins starting with `demo-` are always excluded |
| `curator` | the login of the reader who curated the STV cohort metadata (named in the Fig. 8 caveat note) |
| `export_checks` | optional per-export expected submission counts (`counts`: phase A/B/C) and material matches, checked by `build_readings.py` |

Copy `readers.example.json` to `readers.local.json` and replace the fake logins
with the study's logins. Reader logins are personal data: do not commit them.

## Figure slide identifiers

`figure_ids.local.json` (git-ignored, never committed) holds the few slide and
patient identifiers that the figure builders check or print:
`revision/scripts/figures/_common.py` loads it, or the file named by
`PROSPER_FIGURE_IDS`, and falls back to `figure_ids.example.json` (synthetic
`SJEXAMPLE…` identifiers; the builders then run but their identity checks against
the restricted metadata fail, and any printed slide ID is synthetic).

| key | meaning |
|---|---|
| `scan_to_slide_id` | internal scan index (string) → de-identified slide ID (`Comp.Bio.ID`) for the scans that the figure builders check or label (Fig. 3, Fig. 5, Supplementary Figs. S11–S12 and the tile-classifier failure panel builder) |
| `fig3_example_patient` | patient identifier of the Fig. 3 example query (checked against the fold manifest, not printed in the figure) |
| `fig5_expected_ids` | scan index → slide ID for the five Fig. 5 slides (subset of `scan_to_slide_id`) |

`revision/scripts/figures/scan_to_slide_id.py` re-derives `scan_to_slide_id` from
the restricted COMET metadata. Do not commit the local file.
