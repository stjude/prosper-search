# Recorded environments

This file lists only software and hardware versions that were written into
retained run records at execution time. It is **not** a complete environment
lock. Most runs did not record the version of every transitive dependency, and
no `pip freeze` taken afterwards is presented as a historical environment.

## Versions found in run records

| Run family (manuscript use) | Recorded versions |
|---|---|
| Virchow2 FP32 saved-rank evaluation (encoder grid, Fig. 2 and Supplementary Data 1) | Python 3.12.4, NumPy 2.2.4, pandas 2.3.0; `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` set to 1 |
| Tumor-filtering patient-held-out evaluation (Fig. 3, Supplementary Data 2) | Python 3.12.4, NumPy 2.2.4, pandas 2.3.0 |
| Perturbation suite (Supplementary Fig. S6, Supplementary Data 2, S14), per-slide fingerprints | Python 3.12, PyTorch 2.5.1+cu124, NumPy 2.2.4, Pillow 11.3.0, openslide-python 1.4.3, OpenSlide library 3.4.1, asserted at run time |
| UNI2 all-eligible-tile versus mosaic comparison (Supplementary Data 1, S3), feature extraction | Python 3.10.18, PyTorch 2.7.1+cu126, timm 1.0.16, NumPy 2.2.6, openslide-python 1.4.2; GPU NVIDIA A100-SXM4-80GB |
| LLM summary pilot (Supplementary Note 2, Supplementary Data 5, S48) | Python 3.12.4, OpenAI Python client 3.20.0; model GPT-5 via Azure OpenAI, API-reported version `gpt-5-2025-08-07` |
| Reader-study analysis (Fig. 8, Supplementary Data 6) | Python 3.12.4, NumPy 2.2.4, pandas 2.3.0, Matplotlib 3.10.3 |

Encoder feature extraction for the other encoders, comparator pipelines
(CONCH v1.5 + TITAN, PRISM, SISH) and ROI search ran in separate per-model
environments. Their full package and CUDA versions are not recorded here.
Supplementary Note 4 of the manuscript describes the numerical precision and
software stacks of the encoder pipelines.

## Declared dependencies

`requirements.txt`, `environment.yml`, `pyproject.toml` and `setup.py`
declare the package dependencies, mostly as lower bounds. They are a starting
point for installing the package. They do not reproduce the historical
environments above. Revision scripts also use openpyxl (for the supplementary
workbooks), SciPy and Matplotlib. Foundation-model wrappers need the libraries
named by each model provider, such as timm and huggingface_hub.

## Interpreter used for the release checks

The pre-release tests (byte-compilation and unit tests) were run with Python
3.13.5, NumPy 2.2.6, pandas 2.3.3, SciPy 1.16.3, PyTorch 2.9.1, openpyxl 3.1.5,
Matplotlib 3.10.8, scikit-learn 1.8.0 and POT 0.9.7. That environment is
recorded only so the test results can be interpreted. It was not used to
produce any reported result.
