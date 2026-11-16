"""Artifact path resolution for the benchmark/evaluation layer.

This module is the *single owner* of every filesystem path used by the evaluation
code. It exists because the original scripts hardcoded
``/PROSPER_WORKSPACE/comet-feature-gen`` in 14
places across 7 files, and that mount does not exist on the HPC cluster -- so none
of the published analysis could be re-run without editing the sources.

Roots are resolved from the environment so the same code runs on the cluster, on a
workstation, and in tests:

- ``COMET_ROOT``      -> the ``comet-feature-gen`` tree (per-encoder artifacts)
- ``COMET_SCANS_ROOT``-> the ``comet-new-scans`` tree (ROI hit tables)
- ``PROSPER_OUT``     -> where *new* revision artifacts are written

New results must never be written back into the ``comet-*`` trees. Those hold the
artifacts behind the submitted manuscript and are treated as read-only inputs;
``PROSPER_OUT`` keeps the revision's outputs separable from them.
"""

from __future__ import annotations

import os
from pathlib import Path

# Cluster defaults. Both mount aliases resolve to the same filesystem; the
# ``/PROSPER_STORAGE`` form is shorter and is what the LSF nodes see.
_DEFAULT_COMET_ROOT = (
    "/PROSPER_WORKSPACE/comet-feature-gen"
)
_DEFAULT_SCANS_ROOT = (
    "/PROSPER_WORKSPACE/comet-new-scans"
)
# The WSI files themselves are not under COMET_ROOT or COMET_SCANS_ROOT -- they
# are a sibling tree. Discovered 2026-08-27 for S1a: this directory holds 2,921
# symlinks named ``<Slide.Scan.File>.svs``, one per row of ``slide_metadata()``
# (verified exact set equality, no missing/extra IDs), pointing into
# ``Core/COMET/SVS_RAW_SLIDES/<scan-date>/<id>.svs``. Both the symlink and its
# target are read-only inputs.
_DEFAULT_SLIDES_ROOT = (
    "/PROSPER_WORKSPACE/data/comet_final_HPC"
)


def comet_root() -> Path:
    """Root of the read-only per-encoder artifact tree."""
    return Path(os.environ.get("COMET_ROOT", _DEFAULT_COMET_ROOT))


def scans_root() -> Path:
    """Root of the read-only ROI/new-scans tree."""
    return Path(os.environ.get("COMET_SCANS_ROOT", _DEFAULT_SCANS_ROOT))


def slides_root() -> Path:
    """Root of the read-only WSI file tree (symlinks keyed by scan id)."""
    return Path(os.environ.get("COMET_SLIDES_ROOT", _DEFAULT_SLIDES_ROOT))


def slide_wsi_path(scan_id) -> Path:
    """Path to the WSI file for one slide id.

    All 2,921 COMET files in ``slides_root()`` are ``.svs`` (Aperio); there is
    no ``.ndpi`` or other extension in that cohort as of the 2026-08-27 audit.
    External cohorts are pointed at with ``COMET_SLIDES_ROOT`` plus
    ``COMET_SLIDES_EXT`` (e.g. ``.tiff`` for the generic-TIFF
    ``solid_tumor_validation`` cohort, 2026-08-29).
    """
    ext = os.environ.get("COMET_SLIDES_EXT", ".svs")
    return slides_root() / f"{scan_id}{ext}"


def output_root() -> Path:
    """Root for artifacts produced by the revision. Created on demand."""
    root = Path(os.environ.get("PROSPER_OUT", Path.cwd() / "revision_output"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def require(path: Path, what: str) -> Path:
    """Fail loudly and early on a missing input.

    The original drivers redirected stdout/stderr inside the job script, so a
    missing input surfaced only as an unread per-slide ``.err`` file while the LSF
    array reported success. Checking inputs up front is what makes a failed sweep
    visible.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            f"  COMET_ROOT={comet_root()}\n"
            f"  COMET_SCANS_ROOT={scans_root()}\n"
            f"Set the environment variable if the tree lives elsewhere."
        )
    return path


# --------------------------------------------------------------------------
# Slide-level benchmark artifacts
# --------------------------------------------------------------------------

def similarity_matrix(model: str, method: str, *, tumor_only: bool = False,
                      no_same_patient: bool = True) -> Path:
    """Path to a cached 2921x2921 distance matrix.

    ``no_same_patient=True`` selects the same-patient-masked matrix. That masking
    is applied *to the matrix* (same-patient off-diagonal cells demoted to the
    1e6 sentinel), not by a flag in the evaluator -- which is why the published
    CI JSONs record ``exclude_same_patient: false`` and still are
    patient-excluded. Keep the distinction explicit here so it cannot be
    misread again.
    """
    suffix = "_tumor_only" if tumor_only else ""
    if no_same_patient:
        suffix += "_no_same_patient"
    name = f"combined_similarity_matrix_{model}_{method}{suffix}.pkl"

    # Revision-produced matrices are checked FIRST, in $PROSPER_OUT. Standing
    # rule 6 forbids writing new results back into the read-only comet-* trees,
    # so a matrix for a new encoder (resnet50) or a new tile universe
    # (uni2_all_patches) has nowhere else it may legally live. The comet-* tree
    # stays the fallback and therefore still resolves every published
    # configuration byte-identically -- which is what keeps G2's reproduction
    # valid rather than quietly repointing the trust anchor.
    revision = output_root() / "similarity_matrix" / name
    if revision.exists():
        return revision
    return comet_root() / f"comet-{model}" / "output" / "similarity_matrix" / name


def slide_metadata() -> Path:
    """Slide -> label -> patient table. Carries ``UID.Subject``.

    Note there is a second, older map (``final_scan_id_disease_map.csv``) that has
    no patient column. The ROI evaluator loaded *that* one, which is the direct
    cause of the ROI-level patient leakage: it could not have excluded same-patient
    slides even in principle.
    """
    return (comet_root() / "comet-analysis" / "data"
            / "final_scan_id_disease_map_with_sample_id_patient_id.csv")


# --------------------------------------------------------------------------
# ROI-level artifacts
# --------------------------------------------------------------------------

def roi_hits_dir() -> Path:
    """Per-slide top-k ROI hit tables (2,921 pickles, ~1.2 GB)."""
    return scans_root() / "output" / "top_k_hit_slides_final_slides_final"


def roi_metadata_legacy() -> Path:
    """The patient-free map the published ROI analysis used. Kept for the
    fidelity check in G-ROI: reproducing the published 0.7903 macro F1 requires
    loading exactly this file."""
    return scans_root() / "data" / "final_slide_list" / "final_scan_id_disease_map.csv"


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def result_json(task: str, name: str) -> Path:
    """Where a new result goes: ``$PROSPER_OUT/<task>/<name>.json``.

    Every result is written with the config dict that produced it (see
    ``prosper.evaluation.cli``), so provenance travels with the number instead of
    being encoded in a filename.
    """
    d = output_root() / task
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.json"
