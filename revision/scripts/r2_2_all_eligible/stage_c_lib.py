"""R2.2-ALL Stage C shared helpers: bound inputs, imported production code, records.

Production algorithms are imported from the development repo (``prosper-search``) and called,
never re-implemented: ``aggregate_cli._pool``, ``bob_embedding.features_to_barcodes`` /
``hamming_min_median`` and ``OTSlideEmbeddingGenerator.process_single_slide``. The only new
numerical code in Stage C is the GPU Hamming kernel (stage_c_yottixel.py) and the GPU cosine
cost (stage_c_ot.py), each checked against the imported CPU path.
"""
import hashlib
import io
import json
import os
import pickle
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

H = Path("/PROSPER_WORKSPACE")
D_REPO = H / "prosper-search"
R = H / "prosper_revision"
RUN = R / "prosper_revision_output/r2_2_uni2_all_eligible_20261113/run_001"
STAGE_B = RUN / "stage_b"
STAGE_C = RUN / "stage_c"
BANK = STAGE_B / "bank"
BANK_SUMS = STAGE_B / "bank_SHA256SUMS.txt"
MOSAIC_BANK = H / "comet-feature-gen/comet-uni2/data/tmp_final"
FULL_BANK = R / "prosper_revision_output/features/uni2_all_patches"
LABELS = H / "comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv"
PUBLISHED = H / "comet-feature-gen/comet-uni2/output/similarity_matrix"
SELECTION = STAGE_C / "inputs/canary_selection.json"
DIM = 1536
NBITS = DIM - 1
CODE_MOSAIC, CODE_FULL, CODE_GAP = 0, 1, 2

if str(D_REPO) not in sys.path:
    sys.path.insert(0, str(D_REPO))


def production():
    """Import and return the production modules (bound by file sha in the records)."""
    from prosper.embeddings import aggregate_cli
    from prosper.embeddings.slide_embedding import bob_embedding, ot_embedding
    return aggregate_cli, bob_embedding, ot_embedding


def production_shas():
    files = ["prosper/embeddings/aggregate_cli.py", "prosper/embeddings/similarity_cli.py",
             "prosper/embeddings/slide_embedding/bob_embedding.py",
             "prosper/embeddings/slide_embedding/ot_embedding.py"]
    return {f: sha256_file(D_REPO / f) for f in files}


def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


_SUMS = None


def bank_sums():
    global _SUMS
    if _SUMS is None:
        _SUMS = {}
        for ln in BANK_SUMS.read_text().splitlines():
            if ln.strip():
                h, name = ln.split(None, 1)
                _SUMS[name.strip()] = h
    return _SUMS


def load_bank(slide, verify=True):
    """Admitted bank slide; the npz sha is computed from the bytes actually read."""
    raw = (BANK / f"{slide}.npz").read_bytes()
    if verify:
        got = hashlib.sha256(raw).hexdigest()
        want = bank_sums().get(f"{slide}.npz")
        if got != want:
            raise ValueError(f"bank sha mismatch for {slide}: {got} != {want}")
    with np.load(io.BytesIO(raw)) as z:
        out = {k: z[k] for k in ("feats", "xy", "coverage", "source")}
    if out["feats"].dtype != np.float32 or out["feats"].shape[1] != DIM:
        raise ValueError(f"{slide}: feats {out['feats'].dtype} {out['feats'].shape}")
    return out


def load_mosaic(slide):
    with open(MOSAIC_BANK / f"{slide}.pkl", "rb") as fh:
        feats, mosaic = pickle.load(fh)
    return np.asarray(feats)


def load_full_bank(slide):
    """fp16-autocast fully-tissue bank (feats float32 as stored, xy int64), or None."""
    p = FULL_BANK / f"{slide}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        feats, entries = pickle.load(fh)
    xy = np.asarray([e["wsi_loc"] for e in entries], dtype=np.int64).reshape(-1, 2)
    return np.asarray(feats), xy


def coord_key(xy):
    xy = np.asarray(xy, dtype=np.int64)
    return xy[:, 0] * (1 << 32) + xy[:, 1]


def partial_match(ref_xy, query_xy):
    """(query positions found, their row in ref) for coordinates present in both."""
    rk, qk = coord_key(ref_xy), coord_key(query_xy)
    if len(np.unique(rk)) != len(rk) or len(np.unique(qk)) != len(qk):
        raise ValueError("duplicate coordinates")
    if len(rk) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = np.argsort(rk, kind="stable")
    pos = np.clip(np.searchsorted(rk[order], qk), 0, len(rk) - 1)
    hit = rk[order[pos]] == qk
    return np.nonzero(hit)[0], order[pos[hit]]


def meta():
    """slide -> (patient, label) for all metadata slides; population = non-MISC bank slides."""
    import pandas as pd
    m = pd.read_csv(LABELS, dtype={"Slide.Scan.File": str})
    m["Slide.Scan.File"] = m["Slide.Scan.File"].str.replace(".svs", "", regex=False)
    return {s: (str(p), str(g)) for s, p, g in
            zip(m["Slide.Scan.File"], m["UID.Subject"], m["Tumor.group"])}


def bank_slides():
    return sorted((STAGE_B / "gap_d1/target_slides.txt").read_text().split(), key=int)


def selection():
    return json.loads(SELECTION.read_text())


def vmhwm_gb():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmHWM:"):
                return int(ln.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def env_record():
    rec = {"host": socket.gethostname(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "lsf_job": os.environ.get("LSB_JOBID"), "lsf_index": os.environ.get("LSB_JOBINDEX"),
           "lsf_queue": os.environ.get("LSB_QUEUE"),
           "threads_env": {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
           "numpy": np.__version__}
    try:
        rec["cpu_model"] = next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo")
                                if ln.startswith("model name"))
    except (OSError, StopIteration):
        rec["cpu_model"] = None
    try:
        import threadpoolctl
        rec["blas"] = [{k: i.get(k) for k in ("internal_api", "version", "num_threads",
                                                "architecture")}
                       for i in threadpoolctl.threadpool_info()]
    except Exception as exc:  # noqa: BLE001
        rec["blas"] = repr(exc)
    for mod in ("scipy", "ot", "torch"):
        if mod in sys.modules:
            rec[mod] = getattr(sys.modules[mod], "__version__", None)
    return rec


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=_jdefault) + "\n")
    os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def pack_barcodes(feats, bob):
    """Production barcodes (imported), packed 8 bits/byte along the bit axis."""
    bits = bob.features_to_barcodes(feats)
    if bits.shape[1] != NBITS:
        raise ValueError(bits.shape)
    return np.packbits(bits, axis=1)


def unpack_barcodes(packed):
    return np.unpackbits(packed, axis=1, count=NBITS)
