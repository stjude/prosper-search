# Testing

The repository has four groups of tests:

| Command (from the repository root) | What it covers |
|---|---|
| `pytest` | package tests in `tests/` |
| `pytest revision` | revision analysis, figure, table, reader-study and LLM-rating code |
| `pytest prosper/llm/summarizer/tests` | LLM summarizer |
| `pytest scripts` | small package script tests |
| `pytest tests/test_r2_1_cpu_startup_probe.py` | start-up probe that needs a fresh interpreter (see below) |

Install the package first (`pip install -e .`) or run with the repository root
on `PYTHONPATH`; some tests start subprocesses that import `prosper`. The
revision tests import the analysis scripts as `scripts.<name>`;
`revision/conftest.py` puts `revision/` on the import path for this.

## Why many tests are skipped

Most tests were written to check the code against the data and the computing
environment in which the results were produced. On a fresh checkout without
that data or environment they cannot pass, so they are skipped with a stated
reason. The skip list is explicit: `tests/skip_registry.tsv` and
`revision/skip_registry.tsv` name every skipped test (or module), its marker,
its category, its requirement and the reason. The root `conftest.py` applies
them. Nothing else is skipped by the release configuration; tests that skip
themselves (`pytest.skip`/`skipif` in the test code) keep their own reasons.

| Marker | Category | Runs when |
|---|---|---|
| `site` | placeholder root: the test reads files under `/PROSPER_WORKSPACE` or `/HOME` ([PATHS.md](PATHS.md)) | `PROSPER_SITE_TESTS=1` and the documented roots exist |
| `site` | workspace layout: the test needs the directory layout of the original analysis workspace (sibling checkouts, the output root `revision/prosper_revision_output/`) or a retained run record | `PROSPER_SITE_TESTS=1` and that layout is provided |
| `site` | restricted data: COMET/STV embeddings, manifests or metadata ([DATA_ACCESS.md](DATA_ACCESS.md)) | `PROSPER_SITE_TESTS=1` and the data are provided |
| `site` | held record: the reader-study entity map ([KNOWN_GAPS.md](KNOWN_GAPS.md)) | `PROSPER_SITE_TESTS=1` and the file is provided |
| `pinned` | library-version drift: the code checks, or depends on, the exact versions of the executed environment (scikit-learn 1.7.0 / NumPy 2.2.4 / SciPy 1.16.0 for the fitted heads; PyTorch 2.5.1 / torchvision 0.20.1 / Pillow 11.3.0 for the pixel and stain pipelines; POT 0.9.6.post1 for exact optimal transport) | the installed versions match the `requires` column |
| `optional` | optional dependency: `torchstain`, `pulp`, or the LSF command `bqueues` | the dependency is installed |
| `slow` | two DataLoader extraction modules that did not finish within the release check | `PROSPER_SLOW_TESTS=1` |
| `isolated` | `tests/test_r2_1_cpu_startup_probe.py` asserts that `prosper` has not yet been imported | its file is the only pytest argument |
| `script` | `tests/test_r2_1_filter_cpu_profile_probe.py` is a standalone evidence check, not a pytest module | never under pytest; run `python tests/test_r2_1_filter_cpu_profile_probe.py --evidence <run folder>` on a retained run record |

Version-drift examples seen in the release check: POT 0.9.7 adds a `cost` key
to the exact-transport log, which the retrieval command rejects; scikit-learn
1.8 rejects the executed `SGDClassifier` setting `eta0=0.0`. Five tests in
`tests/test_r2_1_hoptimus_filter_retrieval_cli.py` (`test_actual_reader_*`)
also need `R21_FILTER_STABLE_READER_SHA` set to the sha256 of
`prosper/evaluation/r2_1_hoptimus_mosaic_reader.py`.

TensorFlow is not a dependency. If it is installed, POT imports it and its
start-up warnings on stderr break tests that require empty stderr;
`conftest.py` therefore sets `POT_BACKEND_DISABLE_TENSORFLOW=1`.

## Results of the release check

Clean clone of the release commit; Python 3.13.5 with the libraries listed in
[ENVIRONMENT.md](ENVIRONMENT.md) (not the executed environment); no restricted
data; `PROSPER_SITE_TESTS` and `PROSPER_SLOW_TESTS` unset.

| Command | Passed | Skipped | Failed | Errors |
|---|---:|---:|---:|---:|
| `pytest` | 3120 | 731 | 6 | 3 |
| `pytest revision` | 850 (+10 subtests) | 303 | 1 | 0 |
| `pytest prosper/llm/summarizer/tests` | 49 | 0 | 0 | 0 |
| `pytest scripts` | 1 | 1 | 0 | 0 |
| `pytest tests/test_r2_1_cpu_startup_probe.py` | 77 | 87 | 0 | 0 |
| **Total** | **4097** | **1122** | **7** | **3** |

Skips by reason:

| Reason | `pytest` | `pytest revision` | `scripts` | start-up probe | Total |
|---|---:|---:|---:|---:|---:|
| `pinned`: library-version drift | 660 | 17 | 0 | 0 | 677 |
| `site`: placeholder root | 16 | 133 | 0 | 87 | 236 |
| `site`: workspace layout | 1 module | 64 | 0 | 0 | 65 |
| `site`: restricted data | 3 | 22 | 1 | 0 | 26 |
| `site`: held record | 0 | 3 | 0 | 0 | 3 |
| `optional`: torchstain, bqueues, pulp | 8 | 3 modules | 0 | 0 | 11 |
| `slow` | 2 modules | 0 | 0 | 0 | 2 |
| `isolated` | 1 module | 0 | 0 | 0 | 1 |
| `script` | 1 module | 0 | 0 | 0 | 1 |
| skipped by the test itself (an input, run record or environment variable of the original run is absent; the reason is printed) | 39 | 61 | 0 | 0 | 100 |

A skipped module counts as one skip.

### Failures that are not skipped

- `tests/test_pipeline.py` (5 failures) and `tests/test_slide_search.py`
  (3 errors) are files of the previous public version, unchanged here, and
  fail there in the same way. Three `test_pipeline.py` tests write to `/test`
  (permission denied); two import `normalize_embeddings` and `get_slide_id`,
  which `prosper.utils` does not provide; the `test_slide_search.py` tests
  request a fixture `matrix` that does not exist.
- `tests/test_r2_1_perturb_eval13_comparator.py::test_frozen_modules_unchanged_and_imported_unchanged`
  fails only when it runs after `tests/test_r2_1_perturb_eval13.py`. That file
  calls `importlib.reload` on the module the comparator test then checks by
  object identity (`C.evaluate13 is A.evaluate13`), so the identity no longer
  holds. Run alone, `pytest tests/test_r2_1_perturb_eval13_comparator.py`
  passes (84 passed, 3 skipped); with the reload test deselected, both files
  pass together. This is a test-isolation problem of the original tests, not a
  change in the analysis code.
- `revision/scripts/roi_search_6cfg/test_roi_search.py::test_all`: all its
  assertions pass; the failure is `OSError: Directory not empty` while the
  temporary directory is removed on a network file system that keeps
  placeholder files for still-open memory maps. On a local temporary
  directory (`TMPDIR` on local disk) it is not expected to occur.

### Comparison with the previous public version

The tests present in the previous public version (`tests/` and `scripts/test_*`
there) gave, with the same interpreter: 9 failed, 8 passed and 3 errors for `pytest tests scripts`, and 35 passed
for the summarizer tests. Of the 9 failures, 5 are the `test_pipeline.py`
failures above and 4 need restricted data (`test_slide_search.py::test_load_precomputed_embeddings`,
two in `test_virchow2_sampler.py`, `scripts/test_ot_modules.py`); in the
release these 4 are skipped as `site` (restricted data). The 3 errors are the
`test_slide_search.py` errors above. No test that passed in the previous public
version fails in the release.
