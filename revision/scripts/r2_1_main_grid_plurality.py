#!/usr/bin/env python3
"""RR-5: all-cell non-oracle plurality + hit@k recompute of Fig. 2 / Table S3 (repaired-matrix TITAN).

Implements [decision record] "### RR-5 - exact specification, candidate 002" (items 1-7b pinned inputs, 8/8b ranking and
order sensitivity, 9-16 rules/metrics/identities/contrasts/winners/reconciliation/hit gate/tie census/
main-configuration values/Fig. 2C, outputs, gates in order, FAILED/COMPLETE semantics).

Ranking: rows = queries, columns = library, both in populations.reference_index order; same-patient pairs
excluded by identity; ascending distance with exact float64 ties by ascending reference_index (canonical,
stable mergesort over the full eligible row); the reversed order breaks exact ties by descending
reference_index. Rules (plurality, legacy_generous ORACLE) are executed from the frozen bytes of
scripts/r2_3_internal_failure_tables.py, loaded by sha256. Uncertainty: the pinned whole-patient draws
(5000 x 1876); a slide's replicate weight is its patient's multiplicity; replicate statistics are weight x
one-hot matrix products; 95 % linear percentile intervals. Canonical intervals condition on the canonical
tie order and do not include tie-order uncertainty (OR-5).

Values are descriptive and unadjusted. Exit codes: 0 = COMPLETE (or gates-only pass), 1 = FAILED, 2 = usage.
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'

import argparse  # noqa: E402
import copy  # noqa: E402
import datetime  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import importlib.abc  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------------------------
# Specification constants (candidate 002)
# --------------------------------------------------------------------------------------------

K_VALUES = (1, 3, 5, 10)
RULES = ('plurality', 'oracle')
ORDERS = ('canonical', 'reversed')
QUANTILES = (0.025, 0.975)
QUANTILE_METHOD = 'linear'
SENTINEL = 1e6
RECON_TOL = 1e-9
PREC_TOL = 1e-12
XLSX_PER_CLASS_TOL = 1e-12  # see ambiguity A1: per-class sheets differ from the JSONs at the ~1e-17 level
DP = 6
GATE_ORDER = ('scope', 'binding', 'draws_binding', 'matrix_structure', 'tie_census', 'tie_logic', 'rule_identity',
              'reconciliation_partition', 'reconciliation', 'hit_gate', 'main_config_crosscheck',
              'fig2c_identification', 'bootstrap')
CI_LABEL = {'canonical': 'conditional on the canonical tie order (ascending reference_index); excludes tie-order '
                         'uncertainty (OR-5)',
            'reversed': 'conditional on the reversed tie order (descending reference_index); sensitivity only'}
SUBMITTED_CI_METHOD = 'pps_cluster_one_slide_class_stratified'
CONTRAST_LABEL = {
    'rule': 'rule effect, <= 0 by construction; interval is a magnitude range, not a test',
    'aggregation': 'descriptive, winner selected on the same data (selection-conditioned); no multiplicity '
                   'adjustment',
    'titan': 'TITAN repaired - TITAN published (paired, same draws); descriptive',
}

PUBLISHED_ENCODERS = ('ctranspath', 'exaonepath', 'gigapath', 'hibou', 'phikon-v2', 'simclr', 'uni2', 'virchow2')
METHODS = ('median_pooling', 'ot_cosine', 'sampler', 'yottixel')
ENC_DISPLAY = {'ctranspath': 'CTransPath', 'exaonepath': 'EXAONEPath', 'gigapath': 'GigaPath', 'hibou': 'Hibou-L',
               'phikon-v2': 'Phikon-v2', 'simclr': 'SimCLR', 'uni2': 'UNIv2', 'virchow2': 'Virchow2',
               'titan': 'Conchv1.5', 'h_optimus_1': 'H-Optimus-1'}
AGG_DISPLAY = {'median_pooling': 'Median Pooling', 'ot_cosine': 'Optimal Transport', 'sampler': 'SAMPLER',
               'yottixel': 'Yottixel', 'titan': 'TITAN'}

MATRIX_SHA = {
    'ctranspath/median_pooling': 'd7be568d0ed83ad4c9267f25c86590f92d29e981c4115fb302ad2fcfaefa130d',
    'ctranspath/ot_cosine': 'b88b03a61457753a5e8a2dfb1b2bc287fc7f85f8ba093a76cdbf159d4dc3aed1',
    'ctranspath/sampler': '361000dd31456f504f0758624daaea9ebe0b1b15dcd162d99906c7543c568a66',
    'ctranspath/yottixel': 'a66c021483eacdb35a4bd99be02bbebc51a802e70bb78dd4a0989ea80ed0a5cc',
    'exaonepath/median_pooling': 'c49b813da2f9e7546ceb71e785d208d6c433eeb3eb9d1e9078d8d5d92fe17ebf',
    'exaonepath/ot_cosine': 'ae515ce7986fd5bd734d9f59e3ee1dcb2168fc8ea481686df792f867c6df2cd1',
    'exaonepath/sampler': '92578ddf837cd291a99dc00e0c847f2f67bdbd9be22d2e350376e598763b775d',
    'exaonepath/yottixel': '45a9aabe0b49caf6c8b3df34cf9037cdba22faca0220000d5f4b692d4a746d64',
    'gigapath/median_pooling': 'b6187698af7141d0d3a4bf7a4062cca671af0e5d42c06192de62d20c3abc79f6',
    'gigapath/ot_cosine': 'dc825866d901385a5bef133894b35dce50ff35ae5b84231cb723d145c8cb8d08',
    'gigapath/sampler': 'dca4f524b7c1b9a5df1742c3ef8790bcd2338a1807a79a109cb50e6e6a9e52cb',
    'gigapath/yottixel': 'fddb2590555347d2a39057708a386f70cd77c894de3ae1a6db32b853fdc4e862',
    'h_optimus_1/median_pooling': '13a2c51ccdd5c6001927c20a3975580cc17b6d5d809d718855f7f92283fce4d4',
    'h_optimus_1/ot_cosine': '8240b0a4223fb95a55dfc4d95307683eb257c526240fc3c8ad7db30356b0f354',
    'h_optimus_1/sampler': 'b48cf91310cf8c5b8968d38291fd9e87cfe1b6ebbab87ec8c903735f769c1d23',
    'h_optimus_1/yottixel': 'c578fc7eb06096a0eb9b3a41e7147c17e3a7685b9514170edbfdeb79e5ce1d53',
    'hibou/median_pooling': '85f28defb936b795ee7ff1e6b61c2ada565626d31e7bc4dc16c190ae9123b1a3',
    'hibou/ot_cosine': '6756f4efbc88a9fe975ea09814d3b433bcf6a98c328f050f4b65717350577ff0',
    'hibou/sampler': '9b0992d544d7326980f126af525d773eaa3fa99e055ffcb4276ca2be96182e2f',
    'hibou/yottixel': '0762d829a59c0f4353639ed7beba82447d66fb3f22d6c102b28ab962617bd4fd',
    'phikon-v2/median_pooling': '02ea21ceb0a0e83d51f1a419bc372f518e2fdc36ac9f3d935bb0f8f9ae6be11a',
    'phikon-v2/ot_cosine': '986cea00b0cb8320fc1baee718c140ebdf2328fdf72fb004ca877b040a3f9381',
    'phikon-v2/sampler': '0b9601be510976c84d404bfe80abb62161e2b315f7dd97227e1fac893008ba62',
    'phikon-v2/yottixel': 'f61f35c883af22390c2b409cbb7dc503f24fb256aa347e121dd4d4029decaea2',
    'simclr/median_pooling': '9f0e013c81c520a1e850443a1a8ab4efe8e44e0fa143ae7f689b53ac8faf8570',
    'simclr/ot_cosine': '2ddb9a7aeaf53a2d3dc62b9ad47b826d3f0051b28bb4b0d1a55a1b0ebebad201',
    'simclr/sampler': '5d0dbb275e2373790ffbf857fe5111e900d7019ec9b531372ed8e60e3d3bcbf5',
    'simclr/yottixel': 'd80f19bfceac7e2b5e6be4c74ffbdd9d97e6543204b1bc7166f913c9abfb02f5',
    'titan/titan_published': 'c61aa382856359b86f186cf007be7c16362f5a3121852a55e6bb5328ff0e2a34',
    'titan/titan_repaired': 'f645d32f6414f03c99fa863a73c10d6ba6c767344268ee0120d843545b0010a0',
    'uni2/median_pooling': '7cf0c1e6e885395c6862e54da3fd07083368d9d145cac8f49d1d5e9d462274cf',
    'uni2/ot_cosine': 'a288d8926c8715c86ae3063ff68d9f7b0c6b365eaa77549f45aa02142d2e1714',
    'uni2/sampler': 'a133dcc8d66ead59dc9689a32a35c91373abe8697451aa3a6857d3f6f5adca23',
    'uni2/yottixel': '95b4bd13c4859c3e4460c4045f2a6b18a9a66cc894c070540f5ccc6dda1283e5',
    'virchow2/median_pooling': 'af100dae77fc928348e723830d5072f7ee6de48ce9873fb02691e70e120e8389',
    'virchow2/ot_cosine': 'cc08d4609c9aa666f56185eee504ff6f88730a7070c2c6142f3618d10210888c',
    'virchow2/sampler': '474eb1d9aece42fbd7058e0e6d0022ae8e2d18437a6450ed03d2c06801e9da35',
    'virchow2/yottixel': 'e797589beff0d2f5b4c8ec6594e5e6266181357c82afd8e604e63d34ea550f5f',
}

PIN = {
    'metadata': '45f1466c158f41ceea11abe06138356597a65ce1e97d2b072b780951864582b9',
    'population': '802b05aa6d84525d78500fe46bec9826eec78527d7fd06fb20ac45cebeb2ea5d',
    'draws': 'f81586230703cc578752a9b0a9363ddd4219f9cee28287b164b4115eb44a60a2',
    'pins_file': '1c55767720d344b4ae8de4790235be7097c894f64ab393c702ef78ef9ec16504',
    'xlsx': 'a183d54149c4c17ebf4580fbbaae73f2c51c444a98d030f215e35585ac8b9cac',
    'vote_module': 'afb1c4bd3220ece353268933203b80bb1cd9d5554bd88de9083ab0c27e126eb0',
    'r23_frame': '1ab4e8441a4cd6210b3b114428414d572819cc91020f8ff9f40361a45a3c25c1',
    'qm': {'uni2': '7311b6c0f97e8da2cfedad3cf878289cf42d8c638394005a0209b69245d49b5e',
           'virchow2': '918e759adda540ef263f47ebff8d781a8dd0627daa40d22102a7139ffcdc00c3',
           'h_optimus_1': '20d8e9b1995edbd04a5c175720ff02bf22cf7250b4b769111d26ed0de7fdb4e4'},
    'summary': {'uni2': 'c2ae16a7688fdab9d9b64912a661bb3717e4d970b0f4e888b123c8fef3856a37',
                'virchow2': 'd84549da672191e1bba686442a6dc71ce2e20738bed579c4b2ca6ab640f7ddf5',
                'h_optimus_1': '044c13536b7dc6e1065991547b7c03731fef27ffbdbbcafc61940e9e374c8c43'},
}

FIG2C_SUBMITTED_ORDER = ['RB', 'ACT/ACC', 'NB', 'NF', 'OS', 'WT', 'MEL', 'NORMAL', 'LIVER', 'SS', 'RMS', 'EWS',
                         'MRT', 'MPNST', 'DSRCT']

SCOPE_FORBIDDEN_TOKENS = ('feature', 'embedding', 'tile', 'model', 'checkpoint', '.h5', '.pt', '.pth',
                          '.safetensors', '.svs', '.tif', '.ndpi', '.mrxs', '.zarr')
SCOPE_PATTERNS = {  # role -> (required basename prefix, required basename suffix)
    'metadata': ('', '.csv'), 'population': ('populations', '.csv'), 'draws': ('patient_draws', '.npy'),
    'summary': ('summary', '.json'), 'qm': ('query_metrics', '.csv.gz'),
    'matrix': ('combined_similarity_matrix_', '_no_same_patient.pkl'),
    'hist': ('slide_level_CI_', '_no_same_patient.json'), 'xlsx': ('', '.xlsx'),
    'vote_module': ('r2_3_internal_failure_tables', '.py'), 'r23_frame': ('all_query_frame', '.csv'),
    'pins_file': ('pins_', '.json'),
}


def contract_spec(repo, cfg_root, prosper_out):
    """The exact candidate-002 specification: paths, pins, expected counts and anchors."""
    o = prosper_out
    uni_dir = os.path.join(o, 'r2_1_main_grid_comparators_20261011', 'main_uni2_attempt_001')
    v2_dir = os.path.join(o, 'r2_1_main_grid_comparators_20261011', 'main_virchow2_attempt_001')
    ho_dir = os.path.join(o, 'r2_1_cpu_20261002', 'main_attempt_001')
    cells = []
    for enc in PUBLISHED_ENCODERS:
        for m in METHODS:
            cid = f'{enc}/{m}'
            cells.append(dict(
                cell_id=cid, enc=enc, method=m, display_encoder=ENC_DISPLAY[enc], aggregation=AGG_DISPLAY[m],
                lineage='primary', submitted_lineage=True, reconcile=True,
                path=os.path.join(cfg_root, f'comet-{enc}', 'output', 'similarity_matrix',
                                  f'combined_similarity_matrix_{enc}_{m}_no_same_patient.pkl'),
                sha256=MATRIX_SHA[cid], hist_prefix=f'comet-{enc}/confidence_interval_tie_majority/'
                                                    f'slide_level_CI_{enc}_{m}',
                xlsx_key=[ENC_DISPLAY[enc], AGG_DISPLAY[m]]))
    for m in METHODS:
        cid = f'h_optimus_1/{m}'
        cells.append(dict(
            cell_id=cid, enc='h_optimus_1', method=m, display_encoder=ENC_DISPLAY['h_optimus_1'],
            aggregation=AGG_DISPLAY[m], lineage='h_optimus_new', submitted_lineage=False, reconcile=False,
            path=os.path.join(o, 'similarity_matrix', f'combined_similarity_matrix_h_optimus_1_{m}_no_same_patient.pkl'),
            sha256=MATRIX_SHA[cid], hist_prefix=None, xlsx_key=None))
    cells.append(dict(
        cell_id='titan/titan_repaired', enc='titan', method='titan', display_encoder='Conchv1.5', aggregation='TITAN',
        lineage='primary', submitted_lineage=False, reconcile=False,
        path=os.path.join(o, 'r2_6_titan_arm1_regen_20261004',
                          'combined_similarity_matrix_titan_titan_repaired_no_same_patient.pkl'),
        sha256=MATRIX_SHA['titan/titan_repaired'], hist_prefix=None, xlsx_key=None,
        published_counterpart='titan/titan_published'))
    cells.append(dict(
        cell_id='titan/titan_published', enc='titan', method='titan', display_encoder='Conchv1.5',
        aggregation='TITAN', lineage='titan_published_lineage', submitted_lineage=True, reconcile=True,
        path=os.path.join(cfg_root, 'comet-titan', 'output', 'similarity_matrix',
                          'combined_similarity_matrix_titan_titan_no_same_patient.pkl'),
        sha256=MATRIX_SHA['titan/titan_published'],
        hist_prefix='comet-titan/confidence_interval_tie_majority/slide_level_CI_titan_titan',
        xlsx_key=['Conchv1.5', 'TITAN']))
    zero = {'rank1_tie': 0, 'zero_top10': 0, 'zero_entries': 0}
    for k in K_VALUES:
        zero[f'boundary_tie_{k}'] = 0
        zero[f'order_sensitive_{k}'] = 0
    anchors = {'titan/titan_published': {'rank1_tie': 26, 'boundary_tie_1': 26, 'boundary_tie_3': 29,
                                         'boundary_tie_5': 34, 'boundary_tie_10': 57, 'zero_top10': 21,
                                         'zero_entries': 418},
               'titan/titan_repaired': zero,
               'uni2/yottixel': {'rank1_tie': 112, 'boundary_tie_5': 404}}
    for c in cells:
        if c['method'] in ('median_pooling', 'ot_cosine', 'sampler') and c['enc'] != 'titan':
            anchors.setdefault(c['cell_id'], {})['order_sensitive_10'] = 0
    invariant = [f'{c["cell_id"]}@{k}' for c in cells if c['reconcile'] and c['method'] != 'yottixel'
                 and c['enc'] != 'titan' for k in K_VALUES]
    sensitive = [f'{c["cell_id"]}@{k}' for c in cells if c['reconcile'] and (c['method'] == 'yottixel'
                 or c['enc'] == 'titan') for k in K_VALUES]
    return {
        'name': 'RR-5 candidate 002 (contract)',
        'paths': {
            'pins_file': os.path.join(o, 'r2_1_main_grid_plurality_20261012', 'contract_review_001',
                                      'pins_rr5_observed.json'),
            'metadata': os.path.join(cfg_root, 'comet-analysis', 'data',
                                     'final_scan_id_disease_map_with_sample_id_patient_id.csv'),
            'population': os.path.join(uni_dir, 'populations.csv'),
            'population_copy_virchow2': os.path.join(v2_dir, 'populations.csv'),
            'population_copy_h_optimus_1': os.path.join(ho_dir, 'populations.csv'),
            'draws': os.path.join(uni_dir, 'patient_draws.npy'),
            'draws_copy_virchow2': os.path.join(v2_dir, 'patient_draws.npy'),
            'draws_copy_h_optimus_1': os.path.join(ho_dir, 'patient_draws.npy'),
            'summary_uni2': os.path.join(uni_dir, 'summary.json'),
            'summary_virchow2': os.path.join(v2_dir, 'summary.json'),
            'summary_h_optimus_1': os.path.join(ho_dir, 'summary.json'),
            'qm_uni2': os.path.join(uni_dir, 'query_metrics.csv.gz'),
            'qm_virchow2': os.path.join(v2_dir, 'query_metrics.csv.gz'),
            'qm_h_optimus_1': os.path.join(ho_dir, 'query_metrics.csv.gz'),
            'xlsx': os.path.join(repo, 'docs', 'prosper_submission', 'supplementary', 'supplementary_tables',
                                 'Table_S3_raw_search_metrics.xlsx'),
            'vote_module': os.path.join(repo, 'scripts', 'r2_3_internal_failure_tables.py'),
            'r23_frame': os.path.join(o, 'r2_3_internal_failure_20261003', 'all_query_frame.csv'),
            'comet_root': cfg_root,
        },
        'pins': copy.deepcopy(PIN),
        'cells': cells,
        'k_values': list(K_VALUES),
        'methods': list(METHODS),
        'dropped_label': 'MISC',
        'n_queries': 2778, 'n_patients': 1876, 'n_groups': 15, 'n_cross_label': 45, 'n_reps': 5000,
        'n_hist_rows': 132,
        'qm_encoders': ['uni2', 'virchow2', 'h_optimus_1'],
        'contested_encoders': list(PUBLISHED_ENCODERS),
        'hopt_encoder': 'h_optimus_1',
        'titan_primary': 'titan/titan_repaired',
        'titan_published': 'titan/titan_published',
        'main_config_cells': ['uni2/ot_cosine', 'h_optimus_1/ot_cosine'],
        'r23_cell': 'uni2/ot_cosine',
        'anchors': anchors,
        'expected_partition': {'invariant': invariant, 'sensitive': sensitive},
        'fig2c': {'k': 5, 'pipelines': ['titan/titan_repaired', 'uni2/ot_cosine', 'virchow2/yottixel'],
                  'reversed_pipeline': 'virchow2/yottixel',
                  'xlsx_pipelines': [['Conchv1.5', 'TITAN'], ['UNIv2', 'Optimal Transport'], ['Virchow2', 'Yottixel']],
                  'submitted_order': list(FIG2C_SUBMITTED_ORDER),
                  'identification': 'contract review 001: submitted Figure_2.png + Table S3 k=5 per-class F1; '
                                    'groups ordered by the descending mean of the three pipelines (stable, ties by '
                                    'labels_order); plotting code not located'},
        'xlsx_per_class_tol': XLSX_PER_CLASS_TOL,
    }


# --------------------------------------------------------------------------------------------
# Errors, hashing, the frozen vote module
# --------------------------------------------------------------------------------------------

class GateError(Exception):
    """A specification gate failed; the run writes FAILED.json and exits 1."""

    def __init__(self, gate, reason, detail=None):
        super().__init__(f'{gate}: {reason}')
        self.gate, self.reason, self.detail = gate, reason, detail


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    size = 0
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


class _BytesLoader(importlib.abc.Loader):
    def __init__(self, source, origin):
        self.source, self.origin = source, origin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__dict__.setdefault('__file__', self.origin)
        exec(compile(self.source, self.origin, 'exec'), module.__dict__)


def load_vote_functions(path, expected_sha=PIN['vote_module'], source=None):
    """Verify the full sha256 of the bytes and exec those same bytes; return (plurality, legacy_generous, info)."""
    if source is None:
        with open(path, 'rb') as f:
            source = f.read()
    got = sha256_bytes(source)
    if got != expected_sha:
        raise GateError('binding', f'vote module {path} sha256 {got} != pinned {expected_sha}')
    loader = _BytesLoader(source, str(path))
    spec = importlib.util.spec_from_loader('r2_3_internal_failure_tables_frozen', loader, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.plurality, module.legacy_generous, {'path': str(path), 'sha256': got, 'size': len(source)}


# --------------------------------------------------------------------------------------------
# Scope (item 8): only declared inputs of the allowed kinds are ever opened
# --------------------------------------------------------------------------------------------

def hist_key(cell, k):
    return f'{cell["hist_prefix"]}_top_{k}_no_same_patient.json'


def declared_inputs(spec):
    """Ordered list of (name, role, path) for every file the run may open."""
    p = spec['paths']
    items = [('pins_file', 'pins_file', p['pins_file']), ('metadata', 'metadata', p['metadata'])]
    for n in ('population', 'population_copy_virchow2', 'population_copy_h_optimus_1'):
        items.append((n, 'population', p[n]))
    for n in ('draws', 'draws_copy_virchow2', 'draws_copy_h_optimus_1'):
        items.append((n, 'draws', p[n]))
    for e in spec['qm_encoders']:
        items.append((f'summary_{e}', 'summary', p[f'summary_{e}']))
        items.append((f'qm_{e}', 'qm', p[f'qm_{e}']))
    for c in spec['cells']:
        items.append((f'matrix:{c["cell_id"]}', 'matrix', p.get(f'matrix:{c["cell_id"]}', c['path'])))
    for c in spec['cells']:
        if c['reconcile']:
            for k in spec['k_values']:
                items.append((f'hist:{hist_key(c, k)}', 'hist', os.path.join(p['comet_root'], hist_key(c, k))))
    items += [('xlsx', 'xlsx', p['xlsx']), ('vote_module', 'vote_module', p['vote_module']),
              ('r23_frame', 'r23_frame', p['r23_frame'])]
    return items


def check_scope(spec):
    """FAILED `scope` if any declared input is not of an allowed kind (checked before any file is opened)."""
    seen = {}
    for name, role, path in declared_inputs(spec):
        base = os.path.basename(path).lower()
        pre, suf = SCOPE_PATTERNS[role]
        if not (base.startswith(pre.lower()) and base.endswith(suf.lower())):
            raise GateError('scope', f'{name}: basename {os.path.basename(path)!r} is not an allowed {role} input')
        stem = base[len(pre):len(base) - len(suf)] if role in ('matrix', 'hist') else base
        for tok in SCOPE_FORBIDDEN_TOKENS:
            if tok in stem and not (role == 'vote_module' and tok == 'model'):
                raise GateError('scope', f'{name}: basename {os.path.basename(path)!r} looks like a '
                                         f'feature/embedding/model/tile/image file ({tok!r})')
        if name in seen:
            raise GateError('scope', f'duplicate declared input {name}')
        seen[name] = path
    return [n for n, _, _ in declared_inputs(spec)]


class Inputs:
    """Reads declared inputs only; small inputs are read once and parsed from the hashed bytes."""

    def __init__(self, spec):
        self.spec = spec
        self.decl = {n: (r, pth) for n, r, pth in declared_inputs(spec)}
        self.bytes = {}
        self.bindings = {}

    def path(self, name):
        return self.decl[name][1]

    def read(self, name, keep=True):
        if name not in self.decl:
            raise GateError('scope', f'attempt to read undeclared input {name}')
        role, path = self.decl[name]
        if not os.path.isfile(path):
            raise GateError('binding', f'{name}: missing input file {path}')
        if role == 'matrix' and not keep:
            sha, size = sha256_file(path)
            data = None
        else:
            with open(path, 'rb') as f:
                data = f.read()
            sha, size = sha256_bytes(data), len(data)
        self.bindings[name] = {'role': role, 'path': os.path.abspath(path), 'sha256': sha, 'size': size}
        if keep:
            self.bytes[name] = data
        return data, sha

    def read_matrix_bytes(self, name):
        """Re-read a matrix for structure checks; the bytes must hash to the bound sha256."""
        role, path = self.decl[name]
        with open(path, 'rb') as f:
            data = f.read()
        got = sha256_bytes(data)
        if got != self.bindings[name]['sha256']:
            raise GateError('matrix_structure', f'{name}: bytes changed after binding ({got})')
        return data


# --------------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------------

def _round(x):
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), DP)
    return x


def _json_bytes(obj):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return [conv(v) for v in o.tolist()]
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (float, np.floating)):
            return _round(o)
        if isinstance(o, set):
            return sorted(o)
        return o
    return (json.dumps(conv(obj), indent=1, sort_keys=False, allow_nan=False) + '\n').encode()


def _csv_bytes(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(DP)
    return df.to_csv(index=False, lineterminator='\n', float_format=f'%.{DP}f').encode()


def _write(out, name, data):
    path = os.path.join(out, name)
    with open(path, 'wb') as f:
        f.write(data)
    return path


def write_sha256sums(out):
    names = sorted(n for n in os.listdir(out) if n != 'SHA256SUMS.txt' and os.path.isfile(os.path.join(out, n)))
    lines = [f'{sha256_file(os.path.join(out, n))[0]}  {n}\n' for n in names]
    _write(out, 'SHA256SUMS.txt', ''.join(lines).encode())


def write_failed(out, gate, reason, detail=None, bindings=None, gates_passed=None):
    partial = sorted(n for n in os.listdir(out) if os.path.isfile(os.path.join(out, n))
                     and n not in ('FAILED.json', 'SHA256SUMS.txt'))
    obj = {'status': 'FAILED', 'gate': gate, 'message': reason, 'detail': detail,
           'gates_passed_before_failure': list(gates_passed or []),
           'inputs_bound_so_far': {n: b['sha256'] for n, b in (bindings or {}).items()},
           'partial_outputs': partial, 'complete_written': False, 'automatic_retry': False,
           'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'lsf_job': os.environ.get('LSB_JOBID'), 'host': socket.gethostname()}
    _write(out, 'FAILED.json', _json_bytes(obj))
    write_sha256sums(out)


# --------------------------------------------------------------------------------------------
# Gate: binding (items 1-7b) and draws_binding (item 3)
# --------------------------------------------------------------------------------------------

OVERALL_METRIC_COLS = {'macro_f1': 'Macro F1', 'micro_f1': 'Micro F1', 'macro_precision': 'Macro Precision',
                       'macro_recall': 'Macro Recall'}
XLSX_EMB = {v: k for k, v in ENC_DISPLAY.items() if k != 'h_optimus_1'}
XLSX_AGG = {v: k for k, v in AGG_DISPLAY.items()}


def expected_shas(spec, pins_file):
    pn = spec['pins']
    exp = {'metadata': pn['metadata'], 'xlsx': pn['xlsx'], 'vote_module': pn['vote_module'],
           'r23_frame': pn['r23_frame']}
    for n in ('population', 'population_copy_virchow2', 'population_copy_h_optimus_1'):
        exp[n] = pn['population']
    for n in ('draws', 'draws_copy_virchow2', 'draws_copy_h_optimus_1'):
        exp[n] = pn['draws']
    for e in spec['qm_encoders']:
        exp[f'summary_{e}'] = pn['summary'][e]
        exp[f'qm_{e}'] = pn['qm'][e]
    for c in spec['cells']:
        exp[f'matrix:{c["cell_id"]}'] = c['sha256']
    hj = pins_file.get('historical_json', {})
    for c in spec['cells']:
        if c['reconcile']:
            for k in spec['k_values']:
                exp[f'hist:{hist_key(c, k)}'] = hj.get(hist_key(c, k))
    return exp


def cross_check_pins_file(spec, pf):
    """The frozen review pins file must agree with the specification pins (any disagreement -> FAILED binding)."""
    pn, bad = spec['pins'], []
    for key, want in (('metadata', pn['metadata']), ('populations', pn['population']),
                      ('table_s3_xlsx', pn['xlsx']), ('vote_module', pn['vote_module']),
                      ('r2_3_all_query_frame', pn['r23_frame'])):
        if pf.get(key) != want:
            bad.append(f'pins file {key}={pf.get(key)} != contract {want}')
    mats = pf.get('matrices', {})
    if set(mats) != {c['cell_id'] for c in spec['cells']}:
        bad.append('pins file matrices key set != the 38 contract cells')
    for c in spec['cells']:
        m = mats.get(c['cell_id'], {})
        if m.get('sha256') != c['sha256']:
            bad.append(f'pins file matrix {c["cell_id"]} sha256 != contract')
        if 'path' in m and os.path.realpath(m['path']) != os.path.realpath(
                spec['paths'].get(f'matrix:{c["cell_id"]}', c['path'])):
            bad.append(f'pins file matrix {c["cell_id"]} path != declared path')
    pk = pf.get('packages', {})
    for e in spec['qm_encoders']:
        p = pk.get(e, {})
        for key, want in (('draws_sha256', pn['draws']), ('populations_sha256', pn['population']),
                          ('query_metrics_sha256', pn['qm'][e]), ('summary_sha256', pn['summary'][e])):
            if p.get(key) != want:
                bad.append(f'pins file packages.{e}.{key} != contract')
    want_hist = {hist_key(c, k) for c in spec['cells'] if c['reconcile'] for k in spec['k_values']}
    if set(pf.get('historical_json', {})) != want_hist or len(want_hist) != spec['n_hist_rows']:
        bad.append(f'pins file historical_json keys ({len(pf.get("historical_json", {}))}) != the '
                   f'{spec["n_hist_rows"]} declared historical JSONs')
    if bad:
        raise GateError('binding', 'pins file disagrees with the contract', bad)


def parse_population(spec, inp):
    md = pd.read_csv(io.BytesIO(inp.bytes['metadata']), dtype=str)
    for col in ('Slide.Scan.File', 'Tumor.group', 'UID.Subject'):
        if col not in md.columns:
            raise GateError('binding', f'metadata lacks column {col}')
    if md[['Slide.Scan.File', 'Tumor.group', 'UID.Subject']].isna().any().any():
        raise GateError('binding', 'metadata has missing id/label/patient values')
    keep = (md['Tumor.group'] != spec['dropped_label']).to_numpy()
    mdk = md[keep]
    pop = pd.read_csv(io.BytesIO(inp.bytes['population']))
    want_cols = ['query_index', 'reference_index', 'metadata_index', 'slide_id', 'patient_id', 'diagnostic_group']
    if list(pop.columns) != want_cols:
        raise GateError('binding', f'population columns {list(pop.columns)} != {want_cols}')
    for n in ('population_copy_virchow2', 'population_copy_h_optimus_1'):
        if inp.bytes[n] != inp.bytes['population']:
            raise GateError('binding', f'{n} is not byte-identical to the pinned population')
    n = len(pop)
    ar = np.arange(n)
    for col in ('query_index', 'reference_index', 'metadata_index', 'slide_id'):
        if pop[col].dtype != np.int64:
            raise GateError('binding', f'population {col} dtype {pop[col].dtype} != int64')
    if n != spec['n_queries'] or not (pop.query_index.to_numpy() == ar).all() or \
            not (pop.reference_index.to_numpy() == ar).all():
        raise GateError('binding', f'population query_index/reference_index != 0..{spec["n_queries"] - 1}')
    ids = [str(int(x)) for x in pop.slide_id.tolist()]
    if ids != mdk['Slide.Scan.File'].tolist():
        raise GateError('binding', 'population slide_id (decimal strings) != metadata non-MISC Slide.Scan.File '
                                   'sequence')
    if not (pop.metadata_index.to_numpy() == np.flatnonzero(keep)).all():
        raise GateError('binding', 'population metadata_index != metadata non-MISC row positions')
    pat = pop.patient_id.astype(str).to_numpy()
    lab = pop.diagnostic_group.astype(str).to_numpy()
    if not (pat == mdk['UID.Subject'].to_numpy()).all() or not (lab == mdk['Tumor.group'].to_numpy()).all():
        raise GateError('binding', 'population patient/label != metadata UID.Subject/Tumor.group')
    if len(set(ids)) != n:
        raise GateError('binding', 'duplicate population slide ids')
    labels_order = sorted(set(lab.tolist()))
    n_pat = len(set(pat.tolist()))
    cross = int((pop.assign(p=pat).groupby('p')['diagnostic_group'].nunique() > 1).sum())
    if spec['dropped_label'] in labels_order or len(labels_order) != spec['n_groups'] or \
            n_pat != spec['n_patients'] or cross != spec['n_cross_label']:
        raise GateError('binding', f'population counts groups={len(labels_order)} patients={n_pat} '
                                   f'cross_label={cross} != contract')
    li = {g: i for i, g in enumerate(labels_order)}
    y = np.array([li[g] for g in lab], dtype=np.int64)
    # precision@k denominator check (item 10): every query has >= max(k) eligible same-group library slides
    grp_n = np.bincount(y, minlength=len(labels_order))
    pg = pd.Series(1, index=pd.MultiIndex.from_arrays([pat, y])).groupby(level=[0, 1]).transform('sum').to_numpy()
    support = grp_n[y] - pg
    if support.min() < max(spec['k_values']):
        raise GateError('binding', f'a query has only {int(support.min())} eligible same-group library slides '
                                   f'(< {max(spec["k_values"])})')
    return {'ids': ids, 'pat': pat, 'lab': lab, 'y': y, 'labels_order': labels_order, 'n': n,
            'n_patients': n_pat, 'n_cross_label': cross, 'group_counts': grp_n,
            'min_same_group_library': int(support.min())}


def parse_historical(spec, inp, popd):
    hist = {}
    for c in spec['cells']:
        if not c['reconcile']:
            continue
        for k in spec['k_values']:
            name = f'hist:{hist_key(c, k)}'
            try:
                d = json.loads(inp.bytes[name])
            except ValueError as e:
                raise GateError('binding', f'{name}: not JSON ({e})')
            cfg = d.get('config', {})
            cm = np.asarray(d.get('confusion_matrix'), dtype=np.int64)
            ok = (cfg.get('top_k') == k and cfg.get('dropped_labels') == [spec['dropped_label']]
                  and d.get('n_slides') == popd['n'] and d.get('labels_order') == popd['labels_order']
                  and cm.shape == (len(popd['labels_order']),) * 2)
            if not ok:
                raise GateError('binding', f'{name}: config/frame differs from the contract frame (top_k, MISC '
                                           f'dropped, n_slides, labels_order, 15x15 CM)')
            hist[(c['cell_id'], k)] = {'cm': cm, 'pe': d['point_estimates'], 'ci': d['confidence_intervals'],
                                       'config': cfg, 'key': hist_key(c, k)}
    return hist


def parse_xlsx(spec, inp, popd, hist):
    sheets = pd.read_excel(io.BytesIO(inp.bytes['xlsx']), sheet_name=None)
    need = ('Overall Metrics', 'Per-Class F1', 'Per-Class F1 CI', 'Per-Class Precision', 'Per-Class Recall',
            'Per-Class Support')
    for s in need:
        if s not in sheets:
            raise GateError('binding', f'xlsx lacks sheet {s!r}')
    by_xkey = {(c['xlsx_key'][0], c['xlsx_key'][1]): c['cell_id'] for c in spec['cells']
               if c['reconcile'] and c['xlsx_key']}
    rows = {}
    labels = popd['labels_order']
    max_diff = {'overall': 0.0, 'per_class': 0.0}
    for s in need:
        df = sheets[s]
        keys = []
        for _, r in df.iterrows():
            emb, agg, k = r['Embedding'], r['Aggregation'], int(r['Top K'])
            if emb not in XLSX_EMB or agg not in XLSX_AGG or (emb, agg) not in by_xkey:
                raise GateError('binding', f'xlsx {s}: unmapped row ({emb}, {agg}, {k})')
            cid = by_xkey[(emb, agg)]
            mapped = f'{XLSX_EMB[emb]}/{XLSX_AGG[agg]}'
            if cid != mapped and not (cid == spec['titan_published'] and mapped == 'titan/titan'):
                raise GateError('binding', f'xlsx {s}: key map inconsistency for ({emb}, {agg})')
            keys.append((cid, k))
            rows.setdefault((cid, k), {})[s] = r
        if len(set(keys)) != len(keys) or set(keys) != set(hist) or len(keys) != spec['n_hist_rows']:
            raise GateError('binding', f'xlsx {s}: (embedding, aggregation, k) map is not one-to-one onto the '
                                       f'{spec["n_hist_rows"]} historical rows')
    tol = spec.get('xlsx_per_class_tol', XLSX_PER_CLASS_TOL)
    bad = []
    for key, h in hist.items():
        r = rows[key]
        om = r['Overall Metrics']
        for m, col in OVERALL_METRIC_COLS.items():
            vals = [(float(om[col]), float(h['pe'][m])), (float(om[f'{col} CI Low']), float(h['ci'][m][0])),
                    (float(om[f'{col} CI High']), float(h['ci'][m][1]))]
            for a, b in vals:
                max_diff['overall'] = max(max_diff['overall'], abs(a - b))
                if a != b:
                    bad.append(f'{key} {col}: xlsx != JSON')
        pc = h['pe']['per_class']
        for g in labels:
            if int(r['Per-Class Support'][g]) != int(pc[g]['support']):
                bad.append(f'{key} support {g}: xlsx != JSON')
            pairs = [(r['Per-Class F1'][g], pc[g]['f1']), (r['Per-Class Precision'][g], pc[g]['precision']),
                     (r['Per-Class Recall'][g], pc[g]['recall']),
                     (r['Per-Class F1 CI'][f'{g} CI Low'], h['ci']['per_class_f1'][g][0]),
                     (r['Per-Class F1 CI'][f'{g} CI High'], h['ci']['per_class_f1'][g][1])]
            for a, b in pairs:
                d = abs(float(a) - float(b))
                max_diff['per_class'] = max(max_diff['per_class'], d)
                if not d <= tol:
                    bad.append(f'{key} per-class {g}: xlsx != JSON (|d|={d:.3g})')
    if bad:
        raise GateError('binding', 'xlsx values differ from the historical JSONs', bad[:50])
    return {'rows': rows, 'max_abs_diff_overall': max_diff['overall'],
            'max_abs_diff_per_class': max_diff['per_class'], 'per_class_tolerance': tol}


def gate_binding(spec, inp, out):
    data, sha = inp.read('pins_file')
    if sha != spec['pins']['pins_file']:
        _write(out, 'input_bindings.json', _json_bytes({'bindings': inp.bindings}))
        raise GateError('binding', f'pins file sha256 {sha} != pinned {spec["pins"]["pins_file"]}')
    pf = json.loads(data)
    for name, role, _ in declared_inputs(spec)[1:]:
        inp.read(name, keep=(role != 'matrix'))
    exp = expected_shas(spec, pf)
    _write(out, 'input_bindings.json', _json_bytes({
        'note': 'full sha256 of every declared input, bound before any computation', 'bindings': inp.bindings,
        'expected_sha256': exp, 'pins_file_sha256': sha}))
    mism = [f'{n}: {inp.bindings[n]["sha256"]} != {exp.get(n)}' for n in inp.bindings
            if n != 'pins_file' and inp.bindings[n]['sha256'] != exp.get(n)]
    if mism:
        raise GateError('binding', f'{len(mism)} input(s) differ from the pinned sha256', mism)
    cross_check_pins_file(spec, pf)
    popd = parse_population(spec, inp)
    hist = parse_historical(spec, inp, popd)
    xl = parse_xlsx(spec, inp, popd, hist)
    plur, gen, vinfo = load_vote_functions(inp.path('vote_module'), spec['pins']['vote_module'],
                                           source=inp.bytes['vote_module'])
    return popd, hist, xl, (plur, gen, vinfo)


def gate_draws(spec, inp, popd):
    for n in ('draws_copy_virchow2', 'draws_copy_h_optimus_1'):
        if inp.bytes[n] != inp.bytes['draws']:
            raise GateError('draws_binding', f'{n} is not byte-identical to the pinned draws')
    try:
        draws = np.load(io.BytesIO(inp.bytes['draws']), allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        raise GateError('draws_binding', f'draws not loadable without pickle: {e}')
    P = sorted(set(popd['pat'].tolist()))
    want_shape = (spec['n_reps'], spec['n_patients'])
    if draws.dtype != np.int64 or draws.shape != want_shape:
        raise GateError('draws_binding', f'draws dtype/shape {draws.dtype}/{draws.shape} != int64/{want_shape}')
    if (draws < 0).any() or not (draws.sum(axis=1) == len(P)).all():
        raise GateError('draws_binding', 'draws rows must be non-negative and sum to the number of patients')
    if len(P) != draws.shape[1]:
        raise GateError('draws_binding', 'patient count != draws columns')
    for e in spec['qm_encoders']:
        try:
            s = json.loads(inp.bytes[f'summary_{e}'])
            po = s['policy']['patient_order']
        except (ValueError, KeyError, TypeError):
            raise GateError('draws_binding', f'summary_{e} lacks policy.patient_order')
        if list(map(str, po)) != P or not all(isinstance(x, str) for x in po):
            raise GateError('draws_binding', f'summary_{e} policy.patient_order != sorted population patients')
    pidx = {p: j for j, p in enumerate(P)}
    pcol = np.array([pidx[p] for p in popd['pat']], dtype=np.int64)
    return draws, pcol, P


# --------------------------------------------------------------------------------------------
# Matrices: structure (item 8), ranking, order sensitivity (8b), tie census (14)
# --------------------------------------------------------------------------------------------

def _canonical_decimal(x):
    return isinstance(x, str) and x.isascii() and x.isdigit() and x == str(int(x))


def load_cell_matrix(spec, inp, cell, popd):
    """Unpickle the bound bytes, assert item-8 structure, return X (queries x library, reference_index order)."""
    cid = cell['cell_id']
    name = f'matrix:{cid}'
    data = inp.read_matrix_bytes(name)
    try:
        M = pickle.loads(data)
    except Exception as e:  # noqa: BLE001
        raise GateError('matrix_structure', f'{cid}: not unpicklable ({type(e).__name__})')
    del data
    if not isinstance(M, pd.DataFrame):
        raise GateError('matrix_structure', f'{cid}: not a pandas DataFrame ({type(M).__name__})')
    if not all(dt == np.float64 for dt in M.dtypes):
        raise GateError('matrix_structure', f'{cid}: not float64')
    idx, cols = list(M.index), list(M.columns)
    if idx != cols:
        raise GateError('matrix_structure', f'{cid}: index != columns')
    if len(set(idx)) != len(idx):
        raise GateError('matrix_structure', f'{cid}: duplicate labels')
    if not all(_canonical_decimal(x) for x in idx):
        raise GateError('matrix_structure', f'{cid}: labels are not canonical decimal strings')
    pos = M.index.get_indexer(popd['ids'])
    if (pos < 0).any():
        raise GateError('matrix_structure', f'{cid}: axis set does not contain all population ids '
                                            f'({int((pos < 0).sum())} missing)')
    A = M.to_numpy(dtype=np.float64, copy=False)
    X = np.array(A[np.ix_(pos, pos)], dtype=np.float64)
    axis_order_equals_population = bool((pos == np.arange(len(pos))).all())
    del M, A
    pat = popd['pat']
    same = pat[:, None] == pat[None, :]
    elig_vals = X[~same]
    if not np.isfinite(elig_vals).all() or (elig_vals < 0).any():
        raise GateError('matrix_structure', f'{cid}: an eligible entry is non-finite or negative')
    if (elig_vals == SENTINEL).any():
        raise GateError('matrix_structure', f'{cid}: an eligible (different-patient) entry equals the sentinel')
    if not (X[same] == SENTINEL).all():
        raise GateError('matrix_structure', f'{cid}: a diagonal/same-patient entry is not the sentinel {SENTINEL}')
    del elig_vals
    X[same] = np.inf
    return X, same, axis_order_equals_population


def rank_cell(X, same, y, depth=10):
    """Canonical and reversed orders over the full eligible row, order sensitivity and the tie census."""
    n = X.shape[0]
    o = np.argsort(X, axis=1, kind='mergesort')
    orev = (n - 1) - np.argsort(X[:, ::-1], axis=1, kind='mergesort')
    D = np.take_along_axis(X, o, axis=1)
    Drev = np.take_along_axis(X, orev, axis=1)
    if not np.array_equal(D, Drev):
        raise GateError('matrix_structure', 'reversed order does not preserve the sorted distances')
    del Drev
    eq = (D[:, 1:] == D[:, :-1]) & np.isfinite(D[:, 1:])
    L = y[o]
    diff = L[:, 1:] != L[:, :-1]
    newblock = np.ones(D.shape, dtype=bool)
    newblock[:, 1:] = ~eq
    bs = np.maximum.accumulate(np.where(newblock, np.arange(n)[None, :], 0), axis=1)
    big = np.iinfo(np.int64).max
    min_mixed_start = np.where(eq & diff, bs[:, 1:], big).min(axis=1)
    del L, diff, bs, newblock
    flags = {'rank1_tie': eq[:, 0].copy()}
    for k in K_VALUES:
        flags[f'boundary_tie_{k}'] = eq[:, k - 1].copy()
        flags[f'order_sensitive_{k}'] = min_mixed_start <= k - 1
    zero_top10 = (D[:, :depth] == 0.0).any(axis=1)
    census = {'rank1_tie': int(flags['rank1_tie'].sum())}
    for k in K_VALUES:
        census[f'boundary_tie_{k}'] = int(flags[f'boundary_tie_{k}'].sum())
    for k in K_VALUES:
        census[f'order_sensitive_{k}'] = int(flags[f'order_sensitive_{k}'].sum())
    census['zero_top10'] = int(zero_top10.sum())
    census['zero_entries'] = int(((X == 0.0) & ~same).sum())
    n_elig = (~same).sum(axis=1)
    census['min_eligible_per_query'] = int(n_elig.min())
    top = {'canonical': np.ascontiguousarray(o[:, :depth]), 'reversed': np.ascontiguousarray(orev[:, :depth])}
    return top, flags, census


def predict_cell(top, y, labels_order, plurality, legacy_generous):
    """Per order x k: plurality / ORACLE predictions (frozen vote module), module correct flag, hit, c_k."""
    lab = np.asarray(labels_order, dtype=object)
    li = {g: i for i, g in enumerate(labels_order)}
    truth = lab[y]
    res = {}
    for order in ORDERS:
        T10 = lab[y[top[order]]]
        rows = [list(r) for r in T10]
        for k in K_VALUES:
            pp = np.empty(len(rows), dtype=np.int64)
            po = np.empty(len(rows), dtype=np.int64)
            corr = np.empty(len(rows), dtype=bool)
            for i, r in enumerate(rows):
                pp[i] = li[plurality(r, k)]
                c, p = legacy_generous(r, truth[i], k)
                po[i] = li[p]
                corr[i] = bool(c)
            ck = (y[top[order][:, :k]] == y[:, None]).sum(axis=1).astype(np.int64)
            res[(order, k)] = {'plurality': pp, 'oracle': po, 'oracle_module_correct': corr,
                               'hit': (ck > 0).astype(np.int64), 'c': ck, 'top1': y[top[order][:, 0]]}
    return res


def check_tie_census(spec, census):
    bad = []
    for cid, anchors in spec['anchors'].items():
        if cid not in census:
            continue
        for key, want in anchors.items():
            if census[cid].get(key) != want:
                bad.append(f'{cid} {key}={census[cid].get(key)} != anchor {want}')
    if bad:
        raise GateError('tie_census', 'tie-census anchors not reproduced', bad)


def check_tie_logic(cid, preds, flags):
    for k in K_VALUES:
        inv = ~flags[f'order_sensitive_{k}']
        a, b = preds[('canonical', k)], preds[('reversed', k)]
        for key in ('plurality', 'oracle', 'hit', 'c'):
            if not np.array_equal(a[key][inv], b[key][inv]):
                raise GateError('tie_logic', f'{cid} k={k}: order-insensitive queries differ in {key} between '
                                             f'canonical and reversed orders')


def check_rule_identity(cid, preds, y):
    for order in ORDERS:
        p1 = preds[(order, 1)]
        if not (np.array_equal(p1['plurality'], p1['oracle']) and np.array_equal(p1['plurality'], p1['top1'])):
            raise GateError('rule_identity', f'{cid} {order}: k=1 plurality/ORACLE/top-1 differ')
        if int((p1['plurality'] == y).sum()) != int(p1['hit'].sum()):
            raise GateError('rule_identity', f'{cid} {order}: micro plurality@1 != micro hit@1')
        for k in K_VALUES:
            p = preds[(order, k)]
            pc, oc = p['plurality'] == y, p['oracle'] == y
            if (pc & ~oc).any():
                raise GateError('rule_identity', f'{cid} {order} k={k}: plurality-correct not within ORACLE-correct')
            nei = ~pc & ~oc
            if not np.array_equal(p['plurality'][nei], p['oracle'][nei]):
                raise GateError('rule_identity', f'{cid} {order} k={k}: both wrong but predictions differ')
            if not np.array_equal(p['oracle_module_correct'], oc):
                raise GateError('rule_identity', f'{cid} {order} k={k}: module correct flag != (pred == truth)')
            if (oc & (p['hit'] == 0)).any():
                raise GateError('rule_identity', f'{cid} {order} k={k}: ORACLE correct without a hit')


# --------------------------------------------------------------------------------------------
# Reconciliation (item 12), hit gate (13), main-configuration cross-check (15), Fig. 2C (16)
# --------------------------------------------------------------------------------------------

def confusion(y, pred, G, mask=None):
    if mask is not None:
        y, pred = y[mask], pred[mask]
    return np.bincount(y * G + pred, minlength=G * G).reshape(G, G).astype(np.int64)


def metrics_from_cm(cm):
    """sklearn-convention metrics (zero_division = 0) from a confusion matrix (rows truth)."""
    cm = np.asarray(cm, float)
    tp = np.diag(cm)
    s = cm.sum(1)
    p = cm.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        prec = np.where(p > 0, tp / np.where(p > 0, p, 1), 0.0)
        rec = np.where(s > 0, tp / np.where(s > 0, s, 1), 0.0)
        f1 = np.where(p + s > 0, 2 * tp / np.where(p + s > 0, p + s, 1), 0.0)
    return {'macro_f1': float(f1.mean()), 'micro_f1': float(tp.sum() / s.sum()),
            'macro_precision': float(prec.mean()), 'macro_recall': float(rec.mean())}


def reconcile(spec, cells_by_id, preds, flags, popd, hist, xl, on_partition=None):
    """Partition gate then reconciliation gate; returns (reconciliation dict, lineage rows)."""
    G = len(popd['labels_order'])
    y = popd['y']
    inv_rows, sens_rows = [], []
    for (cid, k) in sorted(hist):
        (inv_rows if flags[cid][f'order_sensitive_{k}'].sum() == 0 else sens_rows).append(f'{cid}@{k}')
    exp = spec['expected_partition']
    partition = {'n_order_invariant': len(inv_rows), 'n_order_sensitive': len(sens_rows),
                 'expected_n_order_invariant': len(exp['invariant']),
                 'expected_n_order_sensitive': len(exp['sensitive']),
                 'matches_expected': sorted(inv_rows) == sorted(exp['invariant'])
                 and sorted(sens_rows) == sorted(exp['sensitive'])}
    if not partition['matches_expected']:
        raise GateError('reconciliation_partition', 'order-invariant/sensitive partition differs from the contract',
                        {'unexpected_sensitive': sorted(set(sens_rows) - set(exp['sensitive'])),
                         'unexpected_invariant': sorted(set(inv_rows) - set(exp['invariant'])),
                         'n_order_invariant': len(inv_rows), 'n_order_sensitive': len(sens_rows)})
    if on_partition:
        on_partition()
    lineage, bad = [], []
    for (cid, k) in sorted(hist):
        h = hist[(cid, k)]
        cell = cells_by_id[cid]
        sens = flags[cid][f'order_sensitive_{k}']
        row = {'row': f'{cid}@{k}', 'cell_id': cid, 'k': k, 'matrix_path': cell['path'],
               'matrix_sha256': cell['sha256'], 'historical_json': h['key'], 'n_order_sensitive': int(sens.sum()),
               'partition': 'order_invariant' if sens.sum() == 0 else 'order_sensitive'}
        cm = {o: confusion(y, preds[cid][(o, k)]['oracle'], G) for o in ORDERS}
        hrs = h['cm'].sum(1)
        support_ok = bool((cm['canonical'].sum(1) == hrs).all() and (hrs == popd['group_counts']).all())
        row['support_gate'] = support_ok
        if not support_ok:
            bad.append(f'{cid}@{k}: support gate')
        xrow = xl['rows'][(cid, k)]['Overall Metrics']
        for o in ORDERS:
            m = metrics_from_cm(cm[o])
            row[f'cm_l1_{o}'] = int(np.abs(cm[o] - h['cm']).sum())
            for key, col in OVERALL_METRIC_COLS.items():
                row[f'abs_delta_json_{key}_{o}'] = abs(m[key] - float(h['pe'][key]))
                row[f'abs_delta_xlsx_{key}_{o}'] = abs(m[key] - float(xrow[col]))
        if sens.sum() == 0:
            ok = bool(np.array_equal(cm['canonical'], h['cm'])) and all(
                row[f'abs_delta_{src}_{key}_canonical'] <= RECON_TOL for src in ('json', 'xlsx')
                for key in OVERALL_METRIC_COLS)
            row['gate'] = 'cm_exact_and_metrics_within_1e-9'
        else:
            cm_inv = confusion(y, preds[cid][('canonical', k)]['oracle'], G, mask=~sens)
            resid = h['cm'] - cm_inv
            sens_counts = np.bincount(y[sens], minlength=G)
            ok = bool((resid >= 0).all() and (resid.sum(1) == sens_counts).all())
            row['gate'] = 'feasibility_hist_minus_cm_invariant'
            row['feasible_nonneg'] = bool((resid >= 0).all())
            row['feasible_rowsum'] = bool((resid.sum(1) == sens_counts).all())
        row['gate_passed'] = ok and support_ok
        if not ok:
            bad.append(f'{cid}@{k}: {row["gate"]}')
        lineage.append(row)
    rec = {'partition': partition, 'n_rows': len(lineage), 'n_rows_passed': sum(r['gate_passed'] for r in lineage),
           'xlsx_vs_json': {k: xl[k] for k in ('max_abs_diff_overall', 'max_abs_diff_per_class',
                                                'per_class_tolerance')},
           'titan_repaired_reconciled': False, 'h_optimus_1_reconciled': False,
           'note': 'historical per-query predictions do not exist; no per-query lineage comparison is claimed'}
    if bad:
        raise GateError('reconciliation', f'{len(bad)} submitted row(s) not reconciled', bad)
    return rec, lineage


def hit_gate(spec, inp, cells_by_id, preds, popd):
    ids = np.array(popd['ids'], dtype=object)
    n = popd['n']
    out = {}
    for e in spec['qm_encoders']:
        try:
            qm = pd.read_csv(io.BytesIO(inp.bytes[f'qm_{e}']), compression='gzip')
        except Exception as ex:  # noqa: BLE001
            raise GateError('hit_gate', f'qm_{e}: unreadable ({type(ex).__name__})')
        need = {'method', 'query_index', 'query', 'diagnostic_group', 'k', 'status', 'returned_depth', 'coverage',
                'conditional_hit', 'conditional_precision'}
        if not need <= set(qm.columns):
            raise GateError('hit_gate', f'qm_{e}: missing columns {sorted(need - set(qm.columns))}')
        n_rows = 0
        for m in spec['methods']:
            cid = f'{e}/{m}'
            for k in spec['k_values']:
                s = qm[(qm['method'] == m) & (qm['k'] == k)]
                if len(s) != n:
                    raise GateError('hit_gate', f'{cid} k={k}: {len(s)} rows != {n}')
                s = s.sort_values('query_index', kind='mergesort')
                qi = s['query_index'].to_numpy()
                if not np.array_equal(qi, np.arange(n)):
                    raise GateError('hit_gate', f'{cid} k={k}: query_index is not 0..{n - 1}')
                if not (s['query'].astype(str).to_numpy() == ids[qi]).all():
                    raise GateError('hit_gate', f'{cid} k={k}: query != ids[query_index]')
                if not (s['diagnostic_group'].astype(str).to_numpy() == popd['lab'][qi]).all():
                    raise GateError('hit_gate', f'{cid} k={k}: diagnostic_group != population label')
                if not ((s['status'] == 'returned').all() and (s['coverage'] == 1).all()
                        and (s['returned_depth'] == k).all()):
                    raise GateError('hit_gate', f'{cid} k={k}: status/coverage/returned_depth not returned/1/{k}')
                p = preds[cid][('canonical', k)]
                h = s['conditional_hit'].to_numpy(dtype=float)
                cp = s['conditional_precision'].to_numpy(dtype=float)
                nh = int((h != p['hit']).sum())
                npr = int((np.abs(cp - p['c'] / k) > PREC_TOL).sum())
                if nh or npr:
                    raise GateError('hit_gate', f'{cid} k={k}: {nh} hit and {npr} precision mismatches against the '
                                                f'accepted evaluator')
                n_rows += n
        out[e] = {'rows_checked': n_rows, 'hit_mismatches': 0, 'precision_mismatches': 0}
    return out


def main_config_crosscheck(spec, inp, preds, popd):
    fr = pd.read_csv(io.BytesIO(inp.bytes['r23_frame']), dtype={'query': str})
    n = popd['n']
    if len(fr) != n or fr['query'].duplicated().any() or set(fr['query']) != set(popd['ids']):
        raise GateError('main_config_crosscheck', 'r2_3 frame query set != population ids')
    fr = fr.set_index('query').loc[popd['ids']]
    lab = np.asarray(popd['labels_order'], dtype=object)
    y = popd['y']
    P = preds[spec['r23_cell']]
    checks = {
        'true_label': (fr['true_label'].astype(str).to_numpy(), lab[y]),
        'top1': (fr['top1'].astype(str).to_numpy(), lab[P[('canonical', 1)]['top1']]),
        'plurality_k3': (fr['plurality_k3'].astype(str).to_numpy(), lab[P[('canonical', 3)]['plurality']]),
        'plurality_k5': (fr['plurality_k5'].astype(str).to_numpy(), lab[P[('canonical', 5)]['plurality']]),
        'plurality_k10': (fr['plurality_k10'].astype(str).to_numpy(), lab[P[('canonical', 10)]['plurality']]),
        'legacy_generous_k5_pred': (fr['legacy_generous_k5_pred'].astype(str).to_numpy(),
                                    lab[P[('canonical', 5)]['oracle']]),
        'legacy_generous_k5_correct': (fr['legacy_generous_k5_correct'].astype(int).to_numpy(),
                                       (P[('canonical', 5)]['oracle'] == y).astype(int)),
    }
    for k in K_VALUES:
        checks[f'hit{k}'] = (fr[f'hit{k}'].astype(int).to_numpy(), P[('canonical', k)]['hit'])
    res = {}
    for col, (a, b) in checks.items():
        mm = int((a != b).sum())
        res[col] = mm
    for k in (5, 10):
        mm = int((np.abs(fr[f'precision{k}'].astype(float).to_numpy() - P[('canonical', k)]['c'] / k) > PREC_TOL).sum())
        res[f'precision{k}'] = mm
    bad = {c: v for c, v in res.items() if v}
    if bad:
        raise GateError('main_config_crosscheck', f'{spec["r23_cell"]} differs from the r2_3 interim frame', bad)
    return {'cell': spec['r23_cell'], 'queries': n, 'mismatches': res,
            'r2_3_ci_comparison': 'not performed (the r2_3 CI files are not declared inputs of candidate 002)'}


def fig2c_identification(spec, xl, popd):
    f = spec['fig2c']
    by_x = {tuple(c['xlsx_key']): c['cell_id'] for c in spec['cells'] if c['xlsx_key']}
    labels = popd['labels_order']
    vals = []
    for xk in f['xlsx_pipelines']:
        cid = by_x.get(tuple(xk))
        if cid is None or (cid, f['k']) not in xl['rows']:
            raise GateError('fig2c_identification', f'pipeline {xk} not in the submitted table')
        r = xl['rows'][(cid, f['k'])]['Per-Class F1']
        vals.append([float(r[g]) for g in labels])
    mean = pd.DataFrame(vals, columns=labels).mean(axis=0)
    order = sorted(labels, key=lambda g: (-mean[g], labels.index(g)))
    ok = order == f['submitted_order']
    if not ok:
        raise GateError('fig2c_identification', 'submitted S3 k=5 ORACLE per-class F1 of the three published-lineage '
                                                'pipelines does not reproduce the submitted Fig. 2C order')
    return {'k': f['k'], 'pipelines_rendered': f['pipelines'], 'xlsx_pipelines': f['xlsx_pipelines'],
            'submitted_order': f['submitted_order'], 'reproduced': True, 'identification': f['identification'],
            'note': 'TITAN rendered from the repaired matrix (footnoted); pipeline set is an open author decision'}


# --------------------------------------------------------------------------------------------
# Bootstrap (item 10): weights x one-hot products, statuses, intervals
# --------------------------------------------------------------------------------------------

STATUS_RANK = {'ok': 0, 'not_reportable': 1, 'zero_support': 2}


def percentile_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    lo, hi = np.quantile(x, QUANTILES, method=QUANTILE_METHOD)
    return float(lo), float(hi)


def max_zero_pred_reps(n_reps):
    """Reportability limit (item 10 / BC-2): <= 1 % of replicates (50 of 5000)."""
    return n_reps // 100


def onehot(idx, G):
    m = np.zeros((len(idx), G), dtype=np.float64)
    m[np.arange(len(idx)), idx] = 1.0
    return m


def class_metrics(tp, p, s):
    """Per-class precision, recall, F1 with zero_division = 0 (sklearn convention); arrays (..., G)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        prec = np.where(p > 0, tp / np.where(p > 0, p, 1.0), 0.0)
        rec = np.where(s > 0, tp / np.where(s > 0, s, 1.0), 0.0)
        f1 = np.where(p + s > 0, 2 * tp / np.where(p + s > 0, p + s, 1.0), 0.0)
    return prec, rec, f1


def worst_status(*s):
    return max(s, key=lambda x: STATUS_RANK[x])


def interval(reps, status):
    if status != 'ok':
        return np.nan, np.nan
    return percentile_ci(reps)


class Stat:
    """A point value, its replicate vector and interval status."""
    __slots__ = ('point', 'reps', 'status', 'lo', 'hi', 'n_zero_pred', 'n_zero_support')

    def __init__(self, point, reps, status, n_zero_pred=0, n_zero_support=0):
        self.point, self.reps, self.status = float(point), reps, status
        self.lo, self.hi = interval(reps, status)
        self.n_zero_pred, self.n_zero_support = int(n_zero_pred), int(n_zero_support)


def cell_statistics(W, T, S_rep, S_pt, y, preds_cell, G):
    """All bootstrap statistics for one cell, every order x k x rule, from one W @ B product."""
    R = W.shape[0]
    thr = max_zero_pred_reps(R)
    zs_g = (S_rep == 0).sum(axis=0)
    zs_any_n = int((S_rep == 0).any(axis=1).sum())
    zs_any = zs_any_n > 0
    blocks, layout = [], []
    for order in ORDERS:
        for k in K_VALUES:
            p = preds_cell[(order, k)]
            blocks.append(T * p['hit'][:, None])
            layout.append((order, k, 'hit'))
            blocks.append(T * (p['c'] / k)[:, None])
            layout.append((order, k, 'prec'))
            for rule in RULES:
                blocks.append(onehot(p[rule], G))
                layout.append((order, k, rule, 'p'))
                blocks.append(T * (p[rule] == y)[:, None])
                layout.append((order, k, rule, 'tp'))
    B = np.concatenate(blocks, axis=1)
    Rm = W @ B
    pt = B.sum(axis=0)
    del B
    sl = {key: (Rm[:, i * G:(i + 1) * G], pt[i * G:(i + 1) * G]) for i, key in enumerate(layout)}
    out = {}
    for order in ORDERS:
        for k in K_VALUES:
            hr, hp = sl[(order, k, 'hit')]
            cr, cp = sl[(order, k, 'prec')]
            with np.errstate(invalid='ignore', divide='ignore'):
                hm_r = (hr / S_rep).mean(axis=1)
                cm_r = (cr / S_rep).mean(axis=1)
            ms = 'zero_support' if zs_any else 'ok'
            out[(order, k, 'hp')] = {
                'hit_micro': Stat(hp.sum() / S_pt.sum(), hr.sum(1) / S_rep.sum(1), 'ok', 0, zs_any_n),
                'hit_macro': Stat((hp / S_pt).mean(), hm_r, ms, 0, zs_any_n),
                'precision_micro': Stat(cp.sum() / S_pt.sum(), cr.sum(1) / S_rep.sum(1), 'ok', 0, zs_any_n),
                'precision_macro': Stat((cp / S_pt).mean(), cm_r, ms, 0, zs_any_n)}
            for rule in RULES:
                pr, ppt = sl[(order, k, rule, 'p')]
                tr, tpt = sl[(order, k, rule, 'tp')]
                prec_r, rec_r, f1_r = class_metrics(tr, pr, S_rep)
                prec_p, rec_p, f1_p = class_metrics(tpt, ppt, S_pt)
                zp_g = (pr == 0).sum(axis=0)
                zp_any = int((pr == 0).any(axis=1).sum())
                m_fp = 'zero_support' if zs_any else ('not_reportable' if zp_any > thr else 'ok')
                m_r = 'zero_support' if zs_any else 'ok'
                d = {
                    'macro_f1': Stat(f1_p.mean(), f1_r.mean(1), m_fp, zp_any, zs_any_n),
                    'micro_f1': Stat(tpt.sum() / S_pt.sum(), tr.sum(1) / S_rep.sum(1), 'ok', zp_any, zs_any_n),
                    'macro_precision': Stat(prec_p.mean(), prec_r.mean(1), m_fp, zp_any, zs_any_n),
                    'macro_recall': Stat(rec_p.mean(), rec_r.mean(1), m_r, zp_any, zs_any_n),
                }
                pc = {}
                for g in range(G):
                    zsg = zs_g[g] > 0
                    st_fp = 'zero_support' if zsg else ('not_reportable' if zp_g[g] > thr else 'ok')
                    st_r = 'zero_support' if zsg else 'ok'
                    pc[g] = {'f1': Stat(f1_p[g], f1_r[:, g], st_fp, zp_g[g], zs_g[g]),
                             'precision': Stat(prec_p[g], prec_r[:, g], st_fp, zp_g[g], zs_g[g]),
                             'recall': Stat(rec_p[g], rec_r[:, g], st_r, zp_g[g], zs_g[g]),
                             'support': Stat(S_pt[g], S_rep[:, g], 'ok', zp_g[g], zs_g[g])}
                d['per_class'] = pc
                out[(order, k, rule)] = d
    return out


def contrast_stat(a, b):
    st = worst_status(a.status, b.status)
    return Stat(a.point - b.point, a.reps - b.reps, st, max(a.n_zero_pred, b.n_zero_pred),
                max(a.n_zero_support, b.n_zero_support))


def count_evaluated(st, n_reps):
    """IR5-4: (order, k, rule) combinations whose four overall metrics each carry exactly n_reps replicates."""
    return sum(1 for key, d in st.items() if len(key) == 3 and key[2] in RULES
               and all(np.shape(d[m].reps) == (n_reps,) for m in OVERALL_METRIC_COLS))


def post_bootstrap_checks(cid, st, hist, inv_rows, bad):
    """IR5-3: unrounded ORACLE canonical points of order-invariant submitted rows equal the JSON (<= 1e-9), and
    micro plurality@1 equals micro hit@1 (point and replicates). Appends failures to bad (deltas only); returns
    the number of order-invariant rows checked for this cell."""
    n = 0
    for k in K_VALUES:
        if f'{cid}@{k}' not in inv_rows:
            continue
        n += 1
        pe = hist[(cid, k)]['pe']
        d = st[('canonical', k, 'oracle')]
        for m in OVERALL_METRIC_COLS:
            delta = abs(d[m].point - float(pe[m]))
            if not delta <= RECON_TOL:
                bad.append(f'{cid}@{k} oracle canonical {m}: |table - submitted JSON| = {delta:.3e} > {RECON_TOL}')
    for order in ORDERS:
        a, h = st[(order, 1, 'plurality')]['micro_f1'], st[(order, 1, 'hp')]['hit_micro']
        if not (a.point == h.point and np.array_equal(a.reps, h.reps)):
            bad.append(f'{cid} {order}: micro plurality@1 != micro hit@1')
    return n


# --------------------------------------------------------------------------------------------
# Output rows
# --------------------------------------------------------------------------------------------

def _cell_cols(c):
    return {'display_encoder': c['display_encoder'], 'aggregation': c['aggregation'], 'cell_id': c['cell_id'],
            'matrix_sha256': c['sha256'], 'lineage': c['lineage']}


def _stat_cols(s, prefix=''):
    return {f'{prefix}value': s.point, f'{prefix}ci_low': s.lo, f'{prefix}ci_high': s.hi,
            f'{prefix}ci_status': s.status}


def cell_rows(spec, c, st, flags, labels_order, xl):
    """table_s3_revised / per_class / hit_precision / tie_sensitivity / rule_contrasts rows for one cell."""
    cid = c['cell_id']
    t3, pcl, hp, ts, rc = [], [], [], [], []
    pub = spec['titan_published']
    for k in K_VALUES:
        nos = int(flags[f'order_sensitive_{k}'].sum())
        sub = None
        if c['submitted_lineage']:
            sub = xl['rows'][(cid, k)]['Overall Metrics']
        elif c.get('published_counterpart') and (c['published_counterpart'], k) in xl['rows']:
            sub = xl['rows'][(pub, k)]['Overall Metrics']
        for order in ORDERS:
            for rule in RULES:
                d = st[(order, k, rule)]
                for m in OVERALL_METRIC_COLS:
                    s = d[m]
                    row = dict(_cell_cols(c), k=k, rule=rule, metric=m, tie_order=order, **_stat_cols(s),
                               ci_label=CI_LABEL[order], n_zero_support=s.n_zero_support,
                               n_zero_pred_any=s.n_zero_pred, n_order_sensitive_k=nos, submitted_value=np.nan,
                               submitted_value_from_published_matrix=np.nan, submitted_ci_low=np.nan,
                               submitted_ci_high=np.nan, submitted_ci_method='')
                    if rule == 'oracle' and sub is not None:
                        col = OVERALL_METRIC_COLS[m]
                        if c['submitted_lineage']:
                            row.update(submitted_value=float(sub[col]), submitted_ci_low=float(sub[f'{col} CI Low']),
                                       submitted_ci_high=float(sub[f'{col} CI High']),
                                       submitted_ci_method=SUBMITTED_CI_METHOD)
                        else:
                            row['submitted_value_from_published_matrix'] = float(sub[col])
                    t3.append(row)
                for g, gd in d['per_class'].items():
                    for m in ('f1', 'precision', 'recall', 'support'):
                        s = gd[m]
                        pcl.append(dict(_cell_cols(c), k=k, rule=rule, tie_order=order, group=labels_order[g],
                                        metric=m, **_stat_cols(s), n_zero_pred=s.n_zero_pred,
                                        n_zero_support=s.n_zero_support))
            for m, s in st[(order, k, 'hp')].items():
                hp.append(dict(_cell_cols(c), k=k, tie_order=order, metric=m, **_stat_cols(s),
                               n_zero_support=s.n_zero_support, n_order_sensitive_k=nos))
        # tie sensitivity: every metric's point value, reversed beside canonical
        for rule in RULES:
            a, b = st[('canonical', k, rule)], st[('reversed', k, rule)]
            for m in OVERALL_METRIC_COLS:
                ts.append(dict(cell_id=cid, k=k, rule=rule, metric=m, group='', canonical_value=a[m].point,
                               reversed_value=b[m].point, reversed_minus_canonical=b[m].point - a[m].point,
                               n_order_sensitive_k=nos))
            for g in a['per_class']:
                for m in ('f1', 'precision', 'recall'):
                    x, z = a['per_class'][g][m].point, b['per_class'][g][m].point
                    ts.append(dict(cell_id=cid, k=k, rule=rule, metric=m, group=labels_order[g], canonical_value=x,
                                   reversed_value=z, reversed_minus_canonical=z - x, n_order_sensitive_k=nos))
        for m in st[('canonical', k, 'hp')]:
            x, z = st[('canonical', k, 'hp')][m].point, st[('reversed', k, 'hp')][m].point
            ts.append(dict(cell_id=cid, k=k, rule='rule_independent', metric=m, group='', canonical_value=x,
                           reversed_value=z, reversed_minus_canonical=z - x, n_order_sensitive_k=nos))
        # (a) rule contrast, canonical order
        a, b = st[('canonical', k, 'plurality')], st[('canonical', k, 'oracle')]
        items = [('macro_f1', '', a['macro_f1'], b['macro_f1']), ('micro_f1', '', a['micro_f1'], b['micro_f1'])]
        items += [('f1', labels_order[g], a['per_class'][g]['f1'], b['per_class'][g]['f1']) for g in a['per_class']]
        for m, g, x, z in items:
            s = contrast_stat(x, z)
            rc.append(dict(_cell_cols(c), k=k, tie_order='canonical', metric=m, group=g,
                           contrast='plurality - oracle', **_stat_cols(s), label=CONTRAST_LABEL['rule']))
    return t3, pcl, hp, ts, rc


def slim_stats(st):
    """Keep only what the cross-cell contrasts/winners need (copies break views into the large product)."""
    keep = {}
    for order in ORDERS:
        for k in K_VALUES:
            for rule in RULES:
                for m in ('macro_f1', 'micro_f1'):
                    s = st[(order, k, rule)][m]
                    s.reps = np.array(s.reps, copy=True)
                    keep[(order, k, rule, m)] = s
            for m in ('hit_micro', 'hit_macro'):
                s = st[(order, k, 'hp')][m]
                s.reps = np.array(s.reps, copy=True)
                keep[(order, k, 'rule_independent', m)] = s
    return keep


def slots(spec):
    out = []
    for enc in spec['contested_encoders']:
        out.append(('C32', enc, ENC_DISPLAY[enc], [f'{enc}/{m}' for m in spec['methods']]))
    out.append(('D36_titan', 'titan', 'Conchv1.5', [spec['titan_primary']]))
    h = spec['hopt_encoder']
    out.append(('H36_h_optimus_1', h, ENC_DISPLAY[h], [f'{h}/{m}' for m in spec['methods']]))
    return out


def winners_and_panels(spec, cells_by_id, slim, xl):
    win, panel, agg = [], [], []
    counts = {}
    for sset, enc, disp, cids in slots(spec):
        for k in K_VALUES:
            for rule in RULES:
                for order in ORDERS:
                    vals = {cid: slim[cid][(order, k, rule, 'macro_f1')] for cid in cids}
                    best = max(v.point for v in vals.values())
                    winners = [cid for cid in cids if vals[cid].point == best]
                    others = [cid for cid in cids if cid not in winners]
                    rbest = max((vals[c].point for c in others), default=np.nan)
                    runners = [c for c in others if vals[c].point == rbest]
                    for w in winners:
                        ov = None  # undefined without a runner-up or without both intervals
                        cmp = [not (vals[w].lo > vals[r].hi or vals[r].lo > vals[w].hi) for r in runners
                               if np.isfinite(vals[w].lo) and np.isfinite(vals[r].lo)]
                        if cmp:
                            ov = any(cmp)
                        win.append(dict(row_type='slot', slot_set=sset, display_encoder=disp, k=k, rule=rule,
                                        tie_order=order, winner=cells_by_id[w]['aggregation'], winner_cell=w,
                                        n_tied_winners=len(winners), weight=1.0 / len(winners), value=best,
                                        ci_low=vals[w].lo, ci_high=vals[w].hi, ci_status=vals[w].status,
                                        runner_up=';'.join(cells_by_id[r]['aggregation'] for r in runners),
                                        runner_up_value=rbest, gap_to_runner_up=best - rbest if runners else np.nan,
                                        winner_ci_overlaps_runner_up=ov))
                        if sset in ('C32', 'H36_h_optimus_1'):
                            for cset in (('C32', 'H36') if sset == 'C32' else ('H36',)):
                                key = (cset, rule, order, cells_by_id[w]['method'])
                                counts[key] = counts.get(key, 0.0) + 1.0 / len(winners)
                                key_k = (f'{cset}@k={k}', rule, order, cells_by_id[w]['method'])
                                counts[key_k] = counts.get(key_k, 0.0) + 1.0 / len(winners)
                    ranked = sorted(cids, key=lambda c: -vals[c].point)
                    for cid in cids:
                        rank = 1 + sum(vals[o].point > vals[cid].point for o in cids)
                        panel.append(dict(slot_set=sset, display_encoder=disp, k=k, rule=rule, tie_order=order,
                                          aggregation=cells_by_id[cid]['aggregation'], cell_id=cid,
                                          value=vals[cid].point, ci_low=vals[cid].lo, ci_high=vals[cid].hi,
                                          ci_status=vals[cid].status, rank=rank, is_winner=cid in winners,
                                          n_tied_winners=len(winners)))
                    del ranked
                if sset == 'C32' or sset == 'H36_h_optimus_1':
                    if k == 5 and rule == 'plurality':
                        vals = {cid: slim[cid][('canonical', k, rule, 'macro_f1')] for cid in cids}
                        best = max(v.point for v in vals.values())
                        ws = [c for c in cids if vals[c].point == best]
                        for w in ws:
                            for cid in cids:
                                if cid in ws:
                                    continue
                                s = contrast_stat(vals[cid], vals[w])
                                agg.append(dict(display_encoder=disp, encoder=enc, k=k, rule=rule,
                                                tie_order='canonical', metric='macro_f1',
                                                aggregation=cells_by_id[cid]['aggregation'], cell_id=cid,
                                                winner_aggregation=cells_by_id[w]['aggregation'], winner_cell=w,
                                                n_tied_winners=len(ws), contrast='aggregation - winner',
                                                **_stat_cols(s), label=CONTRAST_LABEL['aggregation']))
    # submitted (xlsx, ORACLE) winner count over C32 -- diagnostic reproduction of the MS l.39 count
    sub_counts = {}
    for sset, enc, disp, cids in slots(spec):
        if sset != 'C32':
            continue
        for k in K_VALUES:
            v = {cid: float(xl['rows'][(cid, k)]['Overall Metrics']['Macro F1']) for cid in cids}
            best = max(v.values())
            ws = [c for c in cids if v[c] == best]
            for w in ws:
                key = cells_by_id[w]['method']
                sub_counts[key] = sub_counts.get(key, 0.0) + 1.0 / len(ws)
    for (cset, rule, order, method), n in sorted(counts.items()):
        win.append(dict(row_type='count', slot_set=cset, rule=rule, tie_order=order, winner=AGG_DISPLAY[method],
                        weight=n))
    for method, n in sorted(sub_counts.items()):
        win.append(dict(row_type='count', slot_set='C32', rule='oracle', tie_order='submitted_xlsx',
                        winner=AGG_DISPLAY[method], weight=n))
    recomputed = {m: counts.get(('C32', 'oracle', 'canonical', m), 0.0) for m in spec['methods']}
    count_check = {'submitted_xlsx_count': {m: sub_counts.get(m, 0.0) for m in spec['methods']},
                   'recomputed_oracle_canonical_count': recomputed,
                   'equal': all(abs(recomputed[m] - sub_counts.get(m, 0.0)) < 1e-12 for m in spec['methods'])}
    # arg-max k per cell (MS l.40)
    for cid, c in cells_by_id.items():
        for rule in RULES:
            for order in ORDERS:
                v = {k: slim[cid][(order, k, rule, 'macro_f1')].point for k in K_VALUES}
                best = max(v.values())
                ks = [k for k in K_VALUES if v[k] == best]
                win.append(dict(row_type='argmax_k', slot_set='', display_encoder=c['display_encoder'],
                                winner_cell=cid, rule=rule, tie_order=order, winner=';'.join(map(str, ks)),
                                n_tied_winners=len(ks), value=best))
    return win, panel, agg, count_check


def fig2b_rows(spec, cells_by_id, slim):
    rows = []
    k = 5
    for cid, c in cells_by_id.items():
        for rule in RULES:
            for order in ORDERS:
                s = slim[cid][(order, k, rule, 'macro_f1')]
                rows.append(dict(row_type='cell', display_encoder=c['display_encoder'], aggregation=c['aggregation'],
                                 cell_id=cid, lineage=c['lineage'], k=k, rule=rule, tie_order=order,
                                 metric='macro_f1', **_stat_cols(s), scope='', argmax_cell='', argmin_cell=''))
    pubs = spec['contested_encoders']
    h = spec['hopt_encoder']
    for rule in RULES:
        for order in ORDERS:
            def rng(cids, rtype, disp, agg, scope):
                v = {cid: slim[cid][(order, k, rule, 'macro_f1')].point for cid in cids}
                hi_c = max(v, key=v.get)
                lo_c = min(v, key=v.get)
                rows.append(dict(row_type=rtype, display_encoder=disp, aggregation=agg, cell_id='', lineage='',
                                 k=k, rule=rule, tie_order=order, metric='macro_f1_max_minus_min',
                                 value=v[hi_c] - v[lo_c], ci_low=np.nan, ci_high=np.nan, ci_status='point_only',
                                 scope=scope, argmax_cell=hi_c, argmin_cell=lo_c))
            for m in spec['methods']:
                rng([f'{e}/{m}' for e in pubs], 'range_across_encoders', '', AGG_DISPLAY[m], 'published_8')
                rng([f'{e}/{m}' for e in pubs] + [f'{h}/{m}'], 'range_across_encoders', '', AGG_DISPLAY[m],
                    'published_8_plus_h_optimus_1')
            for e in list(pubs) + [h]:
                rng([f'{e}/{m}' for m in spec['methods']], 'range_across_aggregations', ENC_DISPLAY[e], '',
                    'four_aggregations')
    return rows


def titan_rows(spec, slim):
    rep, pub = spec['titan_primary'], spec['titan_published']
    rows = []
    for k in K_VALUES:
        for porder in ORDERS:
            items = [(rule, m) for rule in RULES for m in ('macro_f1', 'micro_f1')]
            items += [('rule_independent', m) for m in ('hit_micro', 'hit_macro')]
            for rule, m in items:
                a = slim[rep][('canonical', k, rule, m)]
                b = slim[pub][(porder, k, rule, m)]
                s = contrast_stat(a, b)
                rows.append(dict(k=k, rule=rule, metric=m, repaired_tie_order='canonical',
                                 published_tie_order=porder, repaired_value=a.point, published_value=b.point,
                                 contrast='titan_repaired - titan_published', **_stat_cols(s),
                                 label=CONTRAST_LABEL['titan']))
    return rows


def fig2c_rows(spec, cells_by_id, full_stats_fig2c, labels_order, xl):
    f = spec['fig2c']
    k = f['k']
    mean = {}
    for g in range(len(labels_order)):
        mean[g] = float(np.mean([full_stats_fig2c[cid][('canonical', k, 'plurality')]['per_class'][g]['f1'].point
                                 for cid in f['pipelines']]))
    order = sorted(range(len(labels_order)), key=lambda g: (-mean[g], g))
    rank = {g: i + 1 for i, g in enumerate(order)}
    sub_rank = {g: i + 1 for i, g in enumerate(f['submitted_order'])}
    by_x = {tuple(c['xlsx_key']): c['cell_id'] for c in spec['cells'] if c['xlsx_key']}
    rows = []
    for cid, xk in zip(f['pipelines'], f['xlsx_pipelines']):
        c = cells_by_id[cid]
        st = full_stats_fig2c[cid]
        xr = xl['rows'][(by_x[tuple(xk)], k)]
        for g in order:
            gl = labels_order[g]
            p = st[('canonical', k, 'plurality')]['per_class'][g]['f1']
            o = st[('canonical', k, 'oracle')]['per_class'][g]['f1']
            pr = st[('reversed', k, 'plurality')]['per_class'][g]['f1']
            orr = st[('reversed', k, 'oracle')]['per_class'][g]['f1']
            rows.append(dict(pipeline=cid, display_encoder=c['display_encoder'], aggregation=c['aggregation'],
                             lineage=c['lineage'], k=k, group=gl, revised_rank=rank[g],
                             revised_order_mean_plurality_f1=mean[g], submitted_rank=sub_rank.get(gl),
                             plurality_f1=p.point, plurality_f1_ci_low=p.lo, plurality_f1_ci_high=p.hi,
                             plurality_f1_ci_status=p.status, oracle_f1=o.point, oracle_f1_ci_low=o.lo,
                             oracle_f1_ci_high=o.hi, oracle_f1_ci_status=o.status,
                             plurality_f1_reversed=pr.point, oracle_f1_reversed=orr.point,
                             submitted_oracle_f1=float(xr['Per-Class F1'][gl]),
                             submitted_from='published TITAN matrix' if c['cell_id'] != by_x[tuple(xk)] else
                             'same matrix',
                             footnote='TITAN repaired matrix (21 regenerated embeddings; LSF 327592210)'
                             if cid == spec['titan_primary'] else ''))
    return rows


def per_query_rows(cells, preds, flags, popd, labels_order):
    lab = np.asarray(labels_order, dtype=object)
    frames = []
    n = popd['n']
    for c in cells:
        cid = c['cell_id']
        for k in K_VALUES:
            a, b = preds[cid][('canonical', k)], preds[cid][('reversed', k)]
            frames.append(pd.DataFrame({
                'cell': cid, 'query_index': np.arange(n), 'slide_id': popd['ids'], 'patient_id': popd['pat'],
                'group': popd['lab'], 'k': k, 'pred_plurality': lab[a['plurality']], 'pred_oracle': lab[a['oracle']],
                'hit': a['hit'], 'c_k': a['c'], 'order_sensitive_k': flags[cid][f'order_sensitive_{k}'].astype(int),
                'boundary_tie_k': flags[cid][f'boundary_tie_{k}'].astype(int),
                'pred_plurality_reversed': lab[b['plurality']], 'pred_oracle_reversed': lab[b['oracle']],
                'hit_reversed': b['hit']}))
    df = pd.concat(frames, ignore_index=True)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0, filename='') as gz:
        gz.write(df.to_csv(index=False, lineterminator='\n').encode())
    return buf.getvalue()


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

def _resources(t0, c0):
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {'wall_seconds': time.time() - t0, 'cpu_seconds': time.process_time() - c0,
            'peak_rss_mib': ru.ru_maxrss / 1024.0, 'host': socket.gethostname(),
            'lsf': {k: v for k, v in os.environ.items() if k.startswith('LSB_') and k in (
                'LSB_JOBID', 'LSB_QUEUE', 'LSB_HOSTS', 'LSB_MCPU_HOSTS', 'LSB_JOBNAME', 'LSB_DJOB_NUMPROC')},
            'threads': {v: os.environ.get(v) for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__,
            'platform': platform.platform()}


def run(spec, out, n_reps=None, gates_only=False, spec_is_contract=True, log=print):
    """Execute every gate in specification order; returns the COMPLETE / GATES_ONLY record (raises GateError)."""
    t0, c0 = time.time(), time.process_time()
    passed = []
    state = {'passed': passed, 'inputs': None}
    n_reps = spec['n_reps'] if n_reps is None else int(n_reps)

    def ok(g):
        passed.append(g)
        log(f'[gate] {g}: pass')

    run.state = state
    check_scope(spec)
    ok('scope')
    inp = Inputs(spec)
    state['inputs'] = inp
    popd, hist, xl, (plurality, legacy_generous, vinfo) = gate_binding(spec, inp, out)
    ok('binding')
    draws, pcol, P = gate_draws(spec, inp, popd)
    ok('draws_binding')
    if not 1 <= n_reps <= draws.shape[0]:
        raise GateError('bootstrap', f'--n-reps {n_reps} outside 1..{draws.shape[0]}')
    # IR5-4: COMPLETE requires exactly the spec's replicate count; refuse before any computation (gates-only
    # runs no bootstrap, so the replicate count does not apply there)
    if not gates_only and n_reps != spec['n_reps']:
        raise GateError('bootstrap', f'--n-reps {n_reps} != the spec replicate count {spec["n_reps"]}; '
                                     'refused before computation')
    labels_order = popd['labels_order']
    y = popd['y']
    cells = spec['cells']
    cells_by_id = {c['cell_id']: c for c in cells}
    tops, flags, census, axis_flags, preds = {}, {}, {}, {}, {}
    for c in cells:
        X, same, axis_eq = load_cell_matrix(spec, inp, c, popd)
        top, fl, cen = rank_cell(X, same, y)
        del X
        tops[c['cell_id']], flags[c['cell_id']], census[c['cell_id']] = top, fl, cen
        axis_flags[c['cell_id']] = axis_eq
        log(f'[matrix] {c["cell_id"]}: ranked')
    ok('matrix_structure')
    check_tie_census(spec, census)
    ok('tie_census')
    for c in cells:
        preds[c['cell_id']] = predict_cell(tops[c['cell_id']], y, labels_order, plurality, legacy_generous)
    for c in cells:
        check_tie_logic(c['cell_id'], preds[c['cell_id']], flags[c['cell_id']])
    ok('tie_logic')
    for c in cells:
        check_rule_identity(c['cell_id'], preds[c['cell_id']], y)
    ok('rule_identity')
    rec, lineage = reconcile(spec, cells_by_id, preds, flags, popd, hist, xl,
                             on_partition=lambda: ok('reconciliation_partition'))
    ok('reconciliation')
    hits = hit_gate(spec, inp, cells_by_id, preds, popd)
    ok('hit_gate')
    mcc = main_config_crosscheck(spec, inp, preds, popd)
    ok('main_config_crosscheck')
    f2c = fig2c_identification(spec, xl, popd)
    ok('fig2c_identification')
    census_rows = [dict(_cell_cols(cells_by_id[cid]), **cen, matrix_axis_order_equals_population=axis_flags[cid])
                   for cid, cen in census.items()]
    if gates_only:
        rec_counts = {'partition': rec['partition'], 'n_rows': rec['n_rows'], 'n_rows_passed': rec['n_rows_passed'],
                      'rows': [{k: r[k] for k in ('row', 'n_order_sensitive', 'partition', 'support_gate',
                                                  'gate', 'gate_passed')} for r in lineage]}
        obj = {'status': 'GATES_ONLY_PASS', 'note': 'dry mode: no bootstrap, no metric values written',
               'gates_passed': passed, 'tie_census': census_rows, 'reconciliation': rec_counts,
               'hit_gate': hits, 'main_config_crosscheck': {'cell': mcc['cell'], 'mismatches': mcc['mismatches']},
               'fig2c_identification_reproduced': f2c['reproduced'],
               'min_same_group_library': popd['min_same_group_library'],
               'population': {'n': popd['n'], 'n_patients': popd['n_patients'], 'n_groups': len(labels_order),
                              'n_cross_label': popd['n_cross_label']},
               'spec_is_contract': spec_is_contract, 'resources': _resources(t0, c0)}
        _write(out, 'GATES_ONLY.json', _json_bytes(obj))
        write_sha256sums(out)
        return obj
    # ---- bootstrap ----
    W = draws[:n_reps][:, pcol].astype(np.float64)
    del draws
    T = onehot(y, len(labels_order))
    S_rep = W @ T
    S_pt = T.sum(axis=0)
    t3, pcl, hp, ts, rc = [], [], [], [], []
    slim, keep_full = {}, {}
    evaluated = 0
    post_bad = []
    inv_rows = {r['row'] for r in lineage if r['partition'] == 'order_invariant'}
    n_inv_checked = 0
    for c in cells:
        cid = c['cell_id']
        st = cell_statistics(W, T, S_rep, S_pt, y, preds[cid], len(labels_order))
        evaluated += count_evaluated(st, n_reps)
        n_inv_checked += post_bootstrap_checks(cid, st, hist, inv_rows, post_bad)
        a, b, h_, d, e = cell_rows(spec, c, st, flags[cid], labels_order, xl)
        t3 += a
        pcl += b
        hp += h_
        ts += d
        rc += e
        if cid in spec['fig2c']['pipelines']:
            keep_full[cid] = st
        slim[cid] = slim_stats(st)
        del st
        log(f'[bootstrap] {cid}: {n_reps} replicates')
    n_expected = len(cells) * len(K_VALUES) * len(RULES) * len(ORDERS)
    if evaluated != n_expected:
        raise GateError('bootstrap', f'{evaluated} of {n_expected} cell x k x rule x tie_order combinations have '
                                     f'{n_reps} replicates for every overall metric')
    # IR5-3: post-bootstrap checks, reported under the final `bootstrap` gate (reconciliation/rule_identity are
    # already passed at this point)
    if n_inv_checked != rec['partition']['n_order_invariant']:
        post_bad.append(f'{n_inv_checked} order-invariant ORACLE rows checked, expected '
                        f'{rec["partition"]["n_order_invariant"]}')
    if post_bad:
        raise GateError('bootstrap', f'post-bootstrap reconciliation/rule-identity check: {len(post_bad)} '
                                     'failure(s)', post_bad)
    win, panel, agg, count_check = winners_and_panels(spec, cells_by_id, slim, xl)
    outputs = {
        'table_s3_revised.csv': pd.DataFrame(t3), 'per_class.csv': pd.DataFrame(pcl),
        'hit_precision.csv': pd.DataFrame(hp), 'rule_contrasts.csv': pd.DataFrame(rc),
        'aggregation_contrasts.csv': pd.DataFrame(agg), 'titan_contrast.csv': pd.DataFrame(titan_rows(spec, slim)),
        'winners.csv': pd.DataFrame(win).astype({'k': 'Int64', 'n_tied_winners': 'Int64'}),
        'fig2a_panel.csv': pd.DataFrame(panel),
        'fig2b_matrix.csv': pd.DataFrame(fig2b_rows(spec, cells_by_id, slim)),
        'fig2c_groups.csv': pd.DataFrame(fig2c_rows(spec, cells_by_id, keep_full, labels_order, xl)),
        'tie_census.csv': pd.DataFrame(census_rows), 'tie_sensitivity.csv': pd.DataFrame(ts),
    }
    for name, df in outputs.items():
        _write(out, name, _csv_bytes(df))
    _write(out, 'per_query_predictions.csv.gz', per_query_rows(cells, preds, flags, popd, labels_order))
    rec.update({'hit_gate': hits, 'main_config_crosscheck': mcc, 'fig2c_identification': f2c,
                'winner_count_reproduction': count_check, 'min_same_group_library': popd['min_same_group_library'],
                'vote_module': vinfo})
    _write(out, 'reconciliation.json', _json_bytes(rec))
    _write(out, 'lineage.json', _json_bytes({'rows': lineage, 'note': 'ORACLE rule; submitted lineage only; '
                                             'TITAN repaired and H-Optimus-1 are not reconciled'}))
    mcv = main_config_values(spec, slim, hp, t3)
    _write(out, 'main_config_values.json', _json_bytes(mcv))
    res = _resources(t0, c0)
    _write(out, 'run_resources.json', _json_bytes(res))
    if n_reps != spec['n_reps'] or evaluated != n_expected:
        raise GateError('bootstrap', f'{n_reps} replicates evaluated for {evaluated} cell x k x rule x tie_order '
                                     f'combinations; COMPLETE requires {spec["n_reps"]} for all {n_expected}')
    check_main_config_tables(out, mcv)
    check_written_k1_identity(out)
    passed.append('bootstrap')
    log('[gate] bootstrap: pass')
    log(f'[diagnostic] ORACLE winner count reproduces submitted count: {bool(count_check["equal"])}')
    obj = {'status': 'COMPLETE', 'bootstrap': True, 'n_reps': n_reps,
           'n_cell_k_rule_order_evaluated': evaluated, 'gates_passed': passed, 'spec_is_contract': spec_is_contract,
           'spec_name': spec['name'], 'n_cells': len(cells),
           'oracle_winner_count_reproduces_submitted': bool(count_check['equal']),
           'outputs': sorted(n for n in os.listdir(out) if n not in ('COMPLETE.json', 'SHA256SUMS.txt')),
           'inputs_bound': {n: b['sha256'] for n, b in inp.bindings.items()},
           'interpretation': 'descriptive, unadjusted; canonical intervals exclude tie-order uncertainty (OR-5)',
           'lsf_job': os.environ.get('LSB_JOBID'), 'host': socket.gethostname(),
           'created': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    _write(out, 'COMPLETE.json', _json_bytes(obj))
    write_sha256sums(out)
    return obj


MAIN_CONFIG_ITEMS = (('plurality', 5, 'micro_f1'), ('plurality', 5, 'macro_f1'), ('oracle', 5, 'micro_f1'),
                     ('rule_independent', 5, 'hit_micro'), ('rule_independent', 5, 'hit_macro'),
                     ('rule_independent', 10, 'hit_micro'), ('rule_independent', 10, 'hit_macro'),
                     ('rule_independent', 5, 'precision_micro'))


def main_config_values(spec, slim, hp_rows, t3_rows):
    hp_idx = {(r['cell_id'], r['k'], r['tie_order'], r['metric']): r for r in hp_rows}
    t3_idx = {(r['cell_id'], r['k'], r['rule'], r['tie_order'], r['metric']): r for r in t3_rows}
    out = {'tie_order': 'canonical', 'ci_label': CI_LABEL['canonical'], 'cells': {}}
    for cid in spec['main_config_cells']:
        vals = {}
        for rule, k, m in MAIN_CONFIG_ITEMS:
            if rule == 'rule_independent':
                r = hp_idx[(cid, k, 'canonical', m)]
                src = 'hit_precision.csv'
            else:
                r = t3_idx[(cid, k, rule, 'canonical', m)]
                src = 'table_s3_revised.csv'
            vals[f'{rule}@{k}:{m}'] = {'value': r['value'], 'ci_low': r['ci_low'], 'ci_high': r['ci_high'],
                                       'ci_status': r['ci_status'], 'source': src, 'k': k, 'rule': rule,
                                       'metric': m}
        out['cells'][cid] = vals
    return out


def check_main_config_tables(out, mcv):
    """main_config_values.json must equal the written table rows exactly at 6 dp (IR5-5)."""
    strcols = {'value': str, 'ci_low': str, 'ci_high': str}
    t3 = pd.read_csv(os.path.join(out, 'table_s3_revised.csv'), dtype=strcols)
    hp = pd.read_csv(os.path.join(out, 'hit_precision.csv'), dtype=strcols)
    js = json.loads(open(os.path.join(out, 'main_config_values.json'), 'rb').read())
    expected_keys = [f'{rule}@{k}:{m}' for rule, k, m in MAIN_CONFIG_ITEMS]
    for cid, vals in js['cells'].items():
        if list(vals) != expected_keys:
            raise GateError('main_config_crosscheck', f'{cid}: main_config_values.json items differ from the contract')
        for key, v in vals.items():
            if v['source'] == 'hit_precision.csv':
                r = hp[(hp.cell_id == cid) & (hp.k == v['k']) & (hp.tie_order == 'canonical') & (hp.metric == v['metric'])]
            else:
                r = t3[(t3.cell_id == cid) & (t3.k == v['k']) & (t3.rule == v['rule']) &
                       (t3.tie_order == 'canonical') & (t3.metric == v['metric'])]
            if len(r) != 1:
                raise GateError('main_config_crosscheck', f'{cid} {key}: {len(r)} table rows')
            r = r.iloc[0]
            for f in ('value', 'ci_low', 'ci_high'):
                a, b = v[f], r[f]
                if not ((a is None and pd.isna(b)) or (a is not None and not pd.isna(b) and f'{a:.{DP}f}' == b)):
                    raise GateError('main_config_crosscheck', f'{cid} {key} {f}: main_config_values.json != table')
            if v['ci_status'] != r['ci_status']:
                raise GateError('main_config_crosscheck', f'{cid} {key}: ci_status differs from the table')


def check_written_k1_identity(out):
    """IR5-3: in the written tables, micro plurality@1 equals micro hit@1 (value, interval, status) per cell/order."""
    strcols = {'value': str, 'ci_low': str, 'ci_high': str, 'ci_status': str}
    cols = ['cell_id', 'tie_order', 'value', 'ci_low', 'ci_high', 'ci_status']
    t3 = pd.read_csv(os.path.join(out, 'table_s3_revised.csv'), dtype=strcols, keep_default_na=False)
    hp = pd.read_csv(os.path.join(out, 'hit_precision.csv'), dtype=strcols, keep_default_na=False)
    a = t3[(t3.k == 1) & (t3.rule == 'plurality') & (t3.metric == 'micro_f1')][cols]
    b = hp[(hp.k == 1) & (hp.metric == 'hit_micro')][cols]
    a = a.sort_values(['cell_id', 'tie_order']).reset_index(drop=True)
    b = b.sort_values(['cell_id', 'tie_order']).reset_index(drop=True)
    if len(a) == 0 or not a.equals(b):
        raise GateError('bootstrap', 'post-bootstrap rule-identity check: written micro plurality@1 != micro hit@1')


def load_spec(args):
    repo = os.path.abspath(args.repo)
    out_root = os.path.join(repo, 'prosper_revision_output')
    cfg = os.path.abspath(args.comet_root) if args.comet_root else os.path.join(os.path.dirname(repo),
                                                                                 'comet-feature-gen')
    spec = contract_spec(repo, cfg, out_root)
    if not args.spec:
        return spec, True
    with open(args.spec) as f:
        js = json.load(f)
    if 'full_spec' in js:
        return js['full_spec'], False
    unknown = set(js.get('paths', {})) - set(spec['paths']) - {f'matrix:{c["cell_id"]}' for c in spec['cells']}
    if unknown or set(js) - {'paths'}:
        raise SystemExit(f'--spec may only override declared paths; unknown: {sorted(unknown | (set(js) - {"paths"}))}')
    spec['paths'].update(js.get('paths', {}))
    return spec, not js.get('paths')


def parse_args(argv):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', required=True, help='new or empty run directory (run_NNN)')
    ap.add_argument('--n-reps', type=int, default=5000)
    ap.add_argument('--gates-only', action='store_true', help='run every gate except bootstrap; counts/booleans only')
    ap.add_argument('--spec', help='JSON with {"paths": {...}} overrides or {"full_spec": {...}} (tests)')
    ap.add_argument('--repo', default=here)
    ap.add_argument('--comet-root', default=None, help='default: <repo>/../comet-feature-gen')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = os.path.abspath(args.out_dir)
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        print(f'--out-dir {out} exists and is not empty', file=sys.stderr)
        return 2
    try:
        spec, is_contract = load_spec(args)
    except (OSError, ValueError, SystemExit) as e:
        print(f'spec error: {e}', file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    try:
        run(spec, out, n_reps=args.n_reps, gates_only=args.gates_only, spec_is_contract=is_contract,
            log=lambda m: print(m, flush=True))
    except GateError as e:
        st = getattr(run, 'state', {})
        inp = st.get('inputs')
        write_failed(out, e.gate, e.reason, e.detail, inp.bindings if inp else None, st.get('passed'))
        print(f'FAILED {e.gate}: {e.reason}', file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        st = getattr(run, 'state', {})
        inp = st.get('inputs')
        write_failed(out, 'unexpected_error', f'{type(e).__name__}: {e}', traceback.format_exc(),
                     inp.bindings if inp else None, st.get('passed'))
        print(f'FAILED unexpected_error: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
