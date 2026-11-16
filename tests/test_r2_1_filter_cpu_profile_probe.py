"""Filter37: stdlib records/source seams and labelled owned processes, never native capture."""
import argparse
import ast
import copy
import contextlib
import hashlib
import importlib.util
import json
import io
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import patch


class NativeTripwire:
    forbidden = ("numpy", "scipy", "sklearn", "pandas", "threadpoolctl", "torch", "ot", "triton", "prosper")

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.forbidden:
            raise AssertionError("Filter34 provider/native import tripwire: "+fullname)
        return None


sys.meta_path.insert(0, NativeTripwire())
SOURCE = Path(__file__).resolve().parents[1]/"scripts"/"r2_1_filter_cpu_profile_probe.py"
module_spec = importlib.util.spec_from_file_location("_filter_cpu_record_unit", SOURCE)
P = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = P
module_spec.loader.exec_module(P)
EVIDENCE = None
CONTEXT_INIT = P.Context.__init__
INITIAL_GATE = P.initial_gate
CAPTURE_WRAPPER = P.capture


def real_path_tripwire(event, args):
    if event != "open" or type(args[0]) is not str:
        return
    if not os.path.isabs(args[0]):
        if args[0] in ("owned", "record.json", "partial.json", "foreign-link", "FAILED.json",
                       "DIAGNOSTIC.json"):
            return
        raise AssertionError("Filter35 unapproved relative acquisition tripwire")
    path = os.path.abspath(args[0])
    if EVIDENCE is not None and path.startswith(str(EVIDENCE)+os.sep):
        return
    if EVIDENCE is not None and Path(path).parent == EVIDENCE.parent/"controls" and \
            Path(path).name in ("case_exercise_seed.json", "case_impossible_success_delta.json",
                "case_baseline_reservation_missing.json", "case_peak_below_observed_live.json",
                "case_software_seed.json", "generated_undeclared_cache_control.json",
                "case_generated_declared_absent_cache.json",
                "filter_cpu_preparation_implementation_repair002_release.json"):
        return
    allowed = {
        str(SOURCE), str(Path(__file__).resolve()),
        str(SOURCE.parent.parent/"prosper"/"evaluation"/"r2_1_hoptimus_filter_retrieval_cli.py"),
        str(SOURCE.parent.parent/"prosper"/"evaluation"/"r2_1_hoptimus_mosaic_reader.py"),
    }
    if path in allowed:
        return
    if "/prosper_revision/" in path or "/prosper-search/" in path or "/features/" in path:
        raise AssertionError("Filter35 real-input/source-scope tripwire")


sys.addaudithook(real_path_tripwire)


def native_denied(*args, **kwargs):
    raise AssertionError("labelled unit-only seam: actual startup/profile/capture forbidden")


for function in ("capture", "verify", "startup_gate", "initial_gate", "read_maps", "fixture", "exercise"):
    setattr(P, function, native_denied)
P.Context.__init__ = native_denied
P.SoftwareObservation.observe = native_denied
P.SoftwareObservation.runtime = native_denied

NULL_NAMES = """
_cython_3_0_12 _cython_3_1_1 _cython_3_1_2 _cython_3_1_4 _cython_3_2_4 cython_runtime
cv2.gapi.core cv2.gapi.core.cpu cv2.gapi.core.fluid cv2.gapi.core.ocl
cv2.gapi.ie cv2.gapi.ie.detail cv2.gapi.imgproc cv2.gapi.imgproc.fluid
cv2.gapi.oak cv2.gapi.onnx cv2.gapi.onnx.ep cv2.gapi.ot cv2.gapi.ot.cpu
cv2.gapi.ov cv2.gapi.own cv2.gapi.own.detail cv2.gapi.render cv2.gapi.render.ocv
cv2.gapi.streaming cv2.gapi.video cv2.gapi.wip cv2.gapi.wip.draw cv2.gapi.wip.gst cv2.gapi.wip.onevpl
cv2.utils.fs cv2.utils.nested pyexpat.errors pyexpat.model
scipy.optimize._highspy._core.cb scipy.optimize._highspy._core.simplex_constants
torch._C._aoti torch._C._autograd torch._C._cpu torch._C._cudart torch._C._cudnn torch._C._cusparselt
torch._C._distributed_autograd torch._C._distributed_c10d torch._C._distributed_rpc torch._C._dynamo
torch._C._dynamo.autograd_compiler torch._C._dynamo.eval_frame torch._C._dynamo.guards torch._C._dynamo.utils
torch._C._fft torch._C._functorch torch._C._instruction_counter torch._C._itt torch._C._jit
torch._C._jit_tree_views torch._C._lazy torch._C._lazy_ts_backend torch._C._linalg torch._C._monitor
torch._C._nested torch._C._nn torch._C._nvtx torch._C._onnx torch._C._profiler torch._C._return_types
torch._C._sparse torch._C._special torch._C._te torch._C._verbose torch._C.cpp torch._C.cpp.nn torch._VF
triton._C.libtriton.amd triton._C.libtriton.amd.passes triton._C.libtriton.amd.passes.ttgpuir
triton._C.libtriton.interpreter triton._C.libtriton.ir triton._C.libtriton.llvm triton._C.libtriton.nvidia
triton._C.libtriton.nvidia.passes triton._C.libtriton.nvidia.passes.ttgpuir
triton._C.libtriton.nvidia.passes.ttnvgpuir triton._C.libtriton.passes triton._C.libtriton.passes.analysis
triton._C.libtriton.passes.common triton._C.libtriton.passes.convert triton._C.libtriton.passes.llvmir
triton._C.libtriton.passes.ttgpuir triton._C.libtriton.passes.ttir typing.io typing.re
xml.parsers.expat.errors xml.parsers.expat.model
""".split()


def metadata_catalog():
    modules = {name: {"name": name.replace("xml.parsers.expat.", "pyexpat."),
                     "file": None, "origin": None, "search_locations": None} for name in NULL_NAMES}
    modules["six.moves"] = {"name": "six.moves", "file": None, "origin": None, "search_locations": []}
    return {"modules": modules}


def shape(key, **overrides):
    value = {"key": key, "name": key, "file": None, "origin": None,
             "search_locations": None, "cached_path": None}
    value.update(overrides)
    return value


def actual_owned(path, payload):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(fd, payload[offset:])
            if count <= 0:
                raise OSError("owned test writer")
            offset += count
        os.fsync(fd)
    finally:
        os.close(fd)


def persist(path, value):
    actual_owned(path, P.encoded(value, 8*P.MIB))


class OwnedCase(unittest.TestCase):
    def setUp(self):
        if EVIDENCE is None:
            self.fail("explicit --evidence directory required for retained Filter34 tests")
        self.directory = EVIDENCE/self.id().split(".")[-1]
        self.directory.mkdir(mode=0o700)
        self.deadline = time.monotonic_ns()+30*10**9


class Records(OwnedCase):
    def test_canonical_original(self):
        for value in (None, {}, [True, False, None, -2**255, 2**255, 0.5],
                      {"z": "\U0001f600\n\t\"\\", "a": ["é", 0]}):
            with self.subTest(value=type(value).__name__):
                data = P.encoded(value, 65536)
                self.assertEqual(data, (json.dumps(value, sort_keys=True, ensure_ascii=True,
                                 allow_nan=False, separators=(",", ":"))+"\n").encode())
                self.assertEqual(P.strict_json(data, canonical=True), value)

    def test_decode_rejections(self):
        for payload in (b'{"x":1,"x":2}', b'{"a":NaN}', b'{"a":1e999}', b'\xef\xbb\xbf{}',
                        b'{"a":[}', b'"unterminated', b'{}\n{}\n', b'[] trailing'):
            with self.subTest(payload=payload):
                with self.assertRaises((P.ProbeError, ValueError, UnicodeError)):
                    P.strict_json(payload)
        with self.assertRaises(P.ProbeError):
            P.strict_json(b"{} ", canonical=True)

    def test_predecode_depth_guard(self):
        with patch.object(P.json, "loads", side_effect=AssertionError("decode must not run")):
            with self.assertRaises(P.ProbeError):
                P.strict_json(b"["*33+b"0"+b"]"*33)
        self.assertEqual(P.strict_json(b"["*32+b"0"+b"]"*32), self.nested(32))

    @staticmethod
    def nested(count):
        value = 0
        for _ in range(count):
            value = [value]
        return value

    def test_preallocation_guard(self):
        huge = "\U0001f600"*100000
        with patch.object(P.json, "dumps", wraps=json.dumps) as encoder:
            with self.assertRaises(P.ProbeError):
                P.encoded({"value": huge}, 128)
            self.assertTrue(all(type(call.args[0]) is str and len(call.args[0]) == 1
                                for call in encoder.call_args_list))
        with self.assertRaises(P.ProbeError):
            P.encoded([2**256], 4096)
        cycle = []
        cycle.append(cycle)
        with self.assertRaises(P.ProbeError):
            P.encoded(cycle, 65536)
        with self.assertRaises(P.ProbeError):
            P.encoded((1, 2), 65536)

    def test_stream_cap_and_literal_bytes(self):
        destination = self.directory/"stream.json"
        fd = os.open(destination, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            binding = P.stream_json(fd, {"value": "\U0001f600"*6000}, 100000)
            os.fsync(fd)
        finally:
            os.close(fd)
        payload = destination.read_bytes()
        self.assertEqual(binding, {"size": len(payload), "sha256": P.sha(payload)})
        self.assertEqual(payload, P.encoded({"value": "\U0001f600"*6000}, 100000))

    def test_escape_note_budget_counts(self):
        for value in ("a"*4094, "a"*4095, "\U0001f600"*1000, "\0\n\"\\é"*2000, ""):
            d = P.bounded_detail(value)
            self.assertLessEqual(len(json.dumps(d["text"], ensure_ascii=True).encode()), 4096)
            self.assertEqual(d["original_characters"], d["retained_source_characters"]+d["omitted_source_characters"])
            prefix = value[:d["retained_source_characters"]]
            self.assertEqual(d["text"], prefix+(" [source text truncated]" if d["truncated"] else ""))

    def test_error_formatting_never_repr(self):
        class Bad:
            def __repr__(self):
                raise AssertionError("repr forbidden")
        error = ValueError("nonfinite scalar: nan", Bad())
        error.__notes__ = ["Bounded returned OT interface evidence: exact", "Captured retrieval warnings: "+"x"*10000, Bad()]
        record = P.error_record(error, "fixed")
        self.assertEqual(record["notes"][0]["text"], error.__notes__[0])
        self.assertTrue(record["notes"][1]["truncated"])
        self.assertTrue(record["notes"][2]["unavailable"])
        self.assertEqual(record["reason"]["unavailable_argument_count"], 1)
        error.__notes__ = Bad()
        self.assertTrue(P.error_record(error, "fixed")["notes_unavailable"])

    def test_secondary_original_and_limits(self):
        primary = ValueError("primary")
        for i in range(70):
            P.add_secondary(primary, OSError(5, "close "+str(i)), "owned close")
        values, omitted = P.secondary_records(primary)
        self.assertEqual(len(values), 64)
        self.assertEqual(omitted, 6)
        self.assertIs(P._SECONDARIES[id(primary)][0], primary)
        self.assertEqual(values[0]["reason"]["errno"], 5)

    def test_closed_error_record_schema_and_omissions(self):
        for error in (ValueError("nonfinite scalar: nan"), OSError(5, "owned synthetic failure"), SystemExit(256)):
            for i in range(70):
                error.add_note("synthetic note "+str(i))
            record = P.error_record(error, "fixed context")
            P.validate_error(record)
            self.assertEqual(record["omitted_note_count"], 6)
            record["notes"][0]["omitted_source_characters"] = 1
            with self.assertRaises(P.ProbeError):
                P.validate_error(record)

    def test_solver_failed_exportable_numbers(self):
        for number in (2, -1, 2**100, -(2**255)):
            record = P.partial_observation({"solver_result_code": number},
                    ValueError("unsuccessful solver status/warning"), "test")
            self.assertEqual(record["solver_result_code"], number)
            self.assertEqual(record["export_state"], "numeric")
            self.assertEqual(record["solver_success_validation"], "rejected")
        for number in (True, 2**256, "1"):
            with self.assertRaises(P.ProbeError):
                P.partial_observation({"solver_result_code": number}, ValueError("failure"), "test")

    def test_solver_absence_cost_and_notes(self):
        error = ValueError("OT distance: nonfinite scalar: nan")
        error.add_note("Bounded returned OT interface evidence: finite numeric code")
        result = P.partial_observation({"solver_result_code": None, "nonfinite_cost_tags": ["nan"]},
                                       error, "cell")
        self.assertEqual(result["export_state"], "capture_export_not_reached")
        self.assertEqual(result["nonfinite_cost_tags"], ["nan"])
        self.assertEqual(result["reason"]["fragments"][0]["text"], "OT distance: nonfinite scalar: nan")
        error.__notes__ = ["Bounded returned OT interface evidence: "+json.dumps({
            "request_id": "cell", "result_code_type": "builtins.int", "warning_type": "builtins.NoneType",
            "solver_result_code_export": "not exported; numeric cell=null; sign=positive; "
            "bit_length=257; diagnostic domain=256 bits"})+"; details: {}"]
        self.assertEqual(P.partial_observation({"solver_result_code": None}, error, "cell")["export_state"],
                         "outside_256_bits")
        unavailable = P.partial_observation({"solver_result_code": None}, ValueError("early"), "cell")
        self.assertEqual(unavailable["export_state"], "unavailable")
        self.assertIsNone(unavailable["nonfinite_cost_tags"])

    def test_catalog_exact94_and_aliases(self):
        catalog = P.catalog_from_metadata(metadata_catalog())
        self.assertEqual(len(catalog), 94)
        for name, reported in catalog.items():
            self.assertEqual(P.classify_module(shape(name, name=reported), catalog), "recorded_fileless_object")
        self.assertEqual(P.classify_module(shape("six.moves", search_locations=[]), catalog),
                         "recorded_empty_location_object")
        for entry in (shape("unlisted"), shape("six.moves"), shape("cython_runtime", search_locations=[]),
                      shape("xml.parsers.expat.model")):
            with self.assertRaises(P.ProbeError):
                P.classify_module(entry, catalog)

    def test_namespace_not_blanket_fileless(self):
        self.assertEqual(P.classify_module(shape("authorized.ns", search_locations=["/owned/ns"]), {},
                                          [{"name": "authorized.ns", "search_locations": ["/owned/ns"]}]), "namespace")
        for entry in (shape("unknown", search_locations=["/not/authorized"]),
                      shape("torch.ops", file="/fake/_ops.py"), shape("unknown", file="relative.py")):
            with self.assertRaises(P.ProbeError):
                P.classify_module(entry, {})

    def test_safe_exposure_no_dynamic_attribute(self):
        class Exploding(types.ModuleType):
            def __getattr__(self, key):
                raise AssertionError("dynamic attribute forbidden")
        source = self.directory/"provider.py"
        actual_owned(source, b"# explicitly synthetic provider file\n")
        provider = Exploding("typing")
        provider.__file__ = str(source)
        leaf = types.ModuleType("typing.io")
        inventory = P.Inventory(self.deadline)
        with patch.dict(P.sys.modules, {"typing": provider}):
            result = P.observe_exposure("typing.io", leaf, inventory)
            self.assertEqual(result["status"], "unresolved")
            provider.__dict__["io"] = leaf
            observed = P.observe_exposure("typing.io", leaf, inventory)
            self.assertEqual(observed["status"], "runtime_attribute_identity")
            self.assertEqual(observed["relation"], "exposes_same_object")

    def test_module_shape_avoids_deprecated_proxy_attribute(self):
        class DenyMeta(type):
            def __getattribute__(cls, name):
                raise AssertionError("deprecated class attribute hook must not run")
        proxy = DenyMeta("typing.io", (), {})
        self.assertEqual(P._module_shape("typing.io", proxy), shape("typing.io"))
        class DenyModule(types.ModuleType):
            def __getattribute__(self, name):
                raise AssertionError("module attribute hook must not run")
        self.assertEqual(P._module_shape("cython_runtime", DenyModule("cython_runtime")), shape("cython_runtime"))

    def test_runtime_typed_bridge_alias_content(self):
        source = self.directory/"owned.py"
        actual_owned(source, b"# synthetic\n")
        alias = self.directory/"alias.py"
        alias.symlink_to(source.name)
        inventory = P.Inventory(self.deadline)
        _, binding = inventory.read(str(source))
        inventory.current_phase = "before"
        _, other = inventory.read(str(alias))
        self.assertEqual(binding, other)
        bridge = P.runtime_bridge(binding, ["before", "after"])
        self.assertTrue(bridge["runtime_asset_id"].startswith("runtime:impl:"))
        self.assertTrue(bridge["software_asset_id"].startswith("sw:"))
        self.assertEqual(bridge["runtime_asset_id"].split(":")[-1], bridge["software_asset_id"].split(":")[-1])
        self.assertEqual(inventory.alias_phases[str(alias)], {"before"})
        with self.assertRaises(P.ProbeError):
            P.runtime_bridge({**binding, "sha256": "wrong"}, ["before"])

    def test_maps_old_predicate_and_cap(self):
        data = (b"0000-0001 r-xp 0 0 0 /owned/executable.so\n"
                b"0000-0001 rw-p 0 0 0 /not/a/selected/data\n"
                b"0000-0001 --xp 0 0 0 [vdso]\n")
        self.assertEqual(P.parse_maps(data), ["/owned/executable.so"])
        with self.assertRaises(P.ProbeError):
            P.parse_maps(b"x r-xp 0 0 0 /owned/deleted.so (deleted)\n")
        with self.assertRaises(P.ProbeError):
            P.parse_maps(b"x"*(8*P.MIB+1))

    def test_generated_adapter_closed_sources(self):
        adapter = P.ThreeSourceAdapter(P.Inventory(self.deadline), {"stable_sources": {}})
        with self.assertRaises(P.ProbeError):
            adapter.read("/not/a/released/source", 16*P.MIB, expected={}, retain=True)

    def test_fixed_fixture_counts_and_recipes(self):
        self.assertEqual(len(P.DATA_NAMES), 14)
        self.assertEqual(len(P.FIXTURE_NAMES), 18)
        self.assertEqual(len(set(P.FIXTURE_NAMES)), 18)
        self.assertEqual(len(P.CAPTURE_NAMES), 8)
        self.assertEqual(len(P.OPERATION_NAMES), 9)
        self.assertEqual([v["request_id"] for v in P.requests()], [
            "ot_literal", "ot_alltrue", "ot_filter5", "ot_filter100", "bob_literal", "bob_alltrue", "bob_filter5", "bob_filter100"])
        self.assertEqual(P.BASELINE_RECIPE["values_hex"], P.BASELINE_HEX)
        self.assertEqual(len(P.CODE_MODULES), 36)

    def test_stdlib_npy_exact_framing(self):
        for name, dtype, shape_value, data in (
            ("coordinates.npy", "<i8", (4, 2), struct.pack("<8q", 0, 0, 16, 0, 0, 0, 16, 0)),
            ("mask_f4_b5.npy", "|b1", (4,), b"\0\1\1\0"),
            ("mask_f0_b100.npy", "|b1", (4,), b"\1"*4)):
            header = repr({"descr": dtype, "fortran_order": False, "shape": shape_value}).encode()
            header += b" "*((16-(10+len(header)+1)%16)%16)+b"\n"
            payload = b"\x93NUMPY\x01\x00"+struct.pack("<H", len(header))+header+data
            P.check_npy(payload, name)
            with self.assertRaises(P.ProbeError):
                P.check_npy(payload+b"\0", name)

    def test_literal_array_recount_is_stdlib_only(self):
        for dtype, width in (("<f4", 1536), ("|u1", 1535)):
            data = bytearray(2*width*(4 if dtype == "<f4" else 1))
            for row, column in enumerate([0, 2]):
                if dtype == "<f4":
                    struct.pack_into("<f", data, 4*(row*width+column), 1)
                else:
                    data[row*width+column] = 1
            value = {"dtype": dtype, "shape": [2, width], "strides": [width*(4 if dtype == "<f4" else 1),
                    4 if dtype == "<f4" else 1], "c_contiguous": True, "f_contiguous": False,
                    "c_order_bytes": len(data), "c_order_sha256": P.sha(data), "nonzero": [
                    {"row": r, "column": c, "float_hex": "0x1.0000000000000p+0" if dtype == "<f4" else None,
                     "uint_value": None if dtype == "<f4" else 1} for r, c in enumerate([0, 2])]}
            P.array_record(value, dtype, [0, 2])
            value["nonzero"][0]["row"] = True
            with self.assertRaises(P.ProbeError):
                P.array_record(value, dtype, [0, 2])

    def test_strict_return_and_native_tripwires(self):
        valid = {"kind": "normal_success_return", "mode": "capture",
                 "handoff_sha256": "a"*64, "executing_source_sha256": "b"*64}
        P.validate_normal(valid, "capture", "b"*64, "a"*64)
        for value in (None, 0, 1, True, {**valid, "extra": 1}, {**valid, "handoff_sha256": "c"*64}):
            with self.assertRaises(P.ProbeError):
                P.validate_normal(value, "capture", "b"*64, "a"*64)
        for name in ("capture", "verify", "read_maps", "fixture", "exercise"):
            with self.assertRaises(AssertionError):
                getattr(P, name)(None)
        with self.assertRaises(AssertionError):
            importlib.util.find_spec("numpy")

    def test_fileless_cache_and_alias_payload_rejected(self):
        catalog = P.catalog_from_metadata(metadata_catalog())
        with self.assertRaises(P.ProbeError):
            P.classify_module(shape("cython_runtime", cached_path="/synthetic/cache.pyc"), catalog)
        path = self.directory/"synthetic.pkl"
        actual_owned(path, b"synthetic-not-a-pickle")
        alias = self.directory/"hidden.py"
        alias.symlink_to(path.name)
        with self.assertRaises(P.ProbeError):
            P.Inventory(self.deadline).read(str(alias))

    def test_unsupported_solver_type_not_silently_unavailable(self):
        error = ValueError("nonintegral/Boolean solver status")
        result = P.partial_observation({"solver_result_code": None}, error, "synthetic")
        self.assertEqual(result["export_state"], "unsupported_type")
        self.assertIsNone(result["solver_result_code"])


def runtime_record_fixture():
    """Fabricated records, not getter execution or current software observations."""
    rid = P.runtime_id("/synthetic/implementation.py")
    logical = ["python", "pandas", "threadpoolctl", "POT", "POT.dist", "POT.network_simplex",
               "POT.emd_wrap", "numpy_cpu_extension", "base:sklearn", "base:numpy", "base:scipy", "base:SGDClassifier"]
    pot = {"signature": "(synthetic_record_only)", "defaults": {
        "center_dual": True, "check_marginals": True, "processes": 1, "numThreads": 1},
        "call_arguments": {"dist": {"metric": "cosine"}, "emd2": {
            "numItermax": 1000000, "numThreads": 1, "log": True, "return_matrix": False}},
        "optimal_status_source": {"package": "POT", "release": "0.9.6.post1", "optimal_code": 1,
            "emd_h_git_blob": "b56f0601bbcc8f10a6671153ded6a23c408428e8",
            "emd_wrap_git_blob": "53df54fc36b804ff2eb7e179ae25d35071f7194b"},
        "numeric_dtype_recipe": {"status": "declared", "raw": "float32", "prepared_ot": "float32",
            "cost": "float32", "masses": "float64", "lp_cost": "float64", "distance": "float64",
            "barcodes": "uint8", "barcode_width": 1535}}
    versions = {"numpy": "2.2.4", "scipy": "1.16.0", "scikit-learn": "1.7.0"}
    profile = {"schema_version": 1, "runtime_scope": "hoptimus_filter_retrieval_pilot_environment_only",
        "python": {"implementation": "synthetic-record", "version": "synthetic-record",
                   "byteorder": "little", "executable_asset_id": rid},
        "base_profile": {"distribution_versions": versions, "implementation": {
            "numpy": {"version": "2.2.4", "path": rid}, "scipy": {"version": "1.16.0", "path": rid},
            "sklearn": {"version": "1.7.0", "path": rid}, "SGDClassifier": {"path": rid}}, "native_pools": []},
        "distribution_versions": {"POT": "0.9.6.post1", "pandas": "synthetic-record",
                                   "threadpoolctl": "synthetic-record", "pot_module_version": "0.9.6.post1"},
        "implementation_assets": [{"asset_id": rid, "size": 10, "sha256": "a"*64}],
        "native_environment": {**{n: "1" for n in P.NATIVE_NAMES}, "CUDA_VISIBLE_DEVICES": ""},
        "numpy_numerics": {"error_policy": {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"},
            "baseline": [], "dispatch": [], "features": {"SYNTHETIC": True}, "extension_asset_id": rid},
        "pot_interface": pot, "limitations": [
            "structural_and_unchanged_controls_only_not_cohort_inference",
            "fixed_reference_folds_heads_masks_and_saved_baselines",
            "no_historical_encoder_or_whole_matrix_lineage_certification",
            "exact_control_equality_not_equivalence_margin", "cooperative_file_custody_not_atomic_snapshot",
            "accounted_buffers_not_rss_or_solver_workspace_bound", "runtime_observation_not_full_executable_mapping",
            "no_next_stage_or_job_authorization"]}
    return {"profile": profile, "observed_paths": {
        n: {"path": "/synthetic/implementation.py", "asset_id": rid} for n in logical}}


class RuntimeRecords(OwnedCase):
    def test_exact_ten_keys_and_declared_numeric_domain(self):
        record = runtime_record_fixture()
        P.validate_runtime(record)
        self.assertEqual(len(record["profile"]), 10)
        record["profile"]["pot_interface"]["numeric_dtype_recipe"]["lp_cost"] = "observed_float64"
        with self.assertRaises(P.ProbeError):
            P.validate_runtime(record)

    def test_sw_prefix_not_filter_runtime_prefix(self):
        record = runtime_record_fixture()
        record["profile"]["implementation_assets"][0]["asset_id"] = P.software_id("/synthetic/implementation.py")
        with self.assertRaises(P.ProbeError):
            P.validate_runtime(record)

    def test_lazy_before_after_changes_preserved_as_records(self):
        before = runtime_record_fixture()
        after = runtime_record_fixture()
        after["profile"]["numpy_numerics"]["features"]["LAZY_SYNTHETIC"] = True
        P.validate_runtime(before)
        P.validate_runtime(after)
        self.assertNotEqual(before, after)
        self.assertEqual(P.strict_json(P.encoded(before, 65536)), before)
        self.assertEqual(P.strict_json(P.encoded(after, 65536)), after)

    def test_native_pool_multiplicity_retained(self):
        record = runtime_record_fixture()
        rid = record["profile"]["python"]["executable_asset_id"]
        pool = {"user_api": "openmp", "internal_api": "openmp", "num_threads": 1,
                "prefix": "synthetic", "filepath": rid, "version": None}
        record["profile"]["base_profile"]["native_pools"] = [dict(pool), dict(pool)]
        for i in range(2):
            record["observed_paths"]["native_pool:"+str(i)] = {"path": "/synthetic/implementation.py", "asset_id": rid}
        P.validate_runtime(record)
        self.assertEqual(len(record["profile"]["base_profile"]["native_pools"]), 2)
        record["profile"]["base_profile"]["native_pools"][0]["num_threads"] = True
        with self.assertRaises(P.ProbeError):
            P.validate_runtime(record)


class MemorySeams(OwnedCase):
    """Explicit fake bookkeeping only; this is not frozen WorkingMemory execution."""
    class Memory:
        def __init__(self):
            self.live, self.peak, self.retained = 0, 0, {}
            self.limit = 8*P.MIB
        def hold(self, token, size):
            self.retained[token] = size
            self.live += size
            self.peak = max(self.peak, self.live)
        def release(self, token):
            self.live -= self.retained.pop(token)

    def test_orphan_after_raw_prepare_baseline_failures(self):
        for stage in ("raw_reference", "prepare_reference", "baseline_load", "query_event"):
            memory = self.Memory()
            old, new = object(), object()
            memory.hold(old, 10)
            ledger = P.ReservationLedger(memory)
            primary = ValueError("owned operation failure")
            with self.assertRaises(ValueError) as caught:
                with ledger.operation("labelled "+stage, stage):
                    memory.hold(new, 20)
                    raise primary
            self.assertIs(caught.exception, primary)
            self.assertEqual(memory.retained, {old: 10})
            self.assertEqual(memory.live, 10)
            self.assertEqual(ledger.events[-1]["new_reservations"][0]["disposition"], "released")

    def test_success_ownership_and_foreign_release(self):
        memory = self.Memory()
        old, new = object(), object()
        memory.hold(old, 10)
        ledger = P.ReservationLedger(memory)
        with ledger.operation("query_event synthetic seam", "query_event"):
            memory.hold(new, 20)
        with self.assertRaises(P.ProbeError):
            ledger.release(old)
        ledger.cleanup()
        self.assertEqual(memory.retained, {old: 10})

    def test_release_error_preserves_primary(self):
        memory = self.Memory()
        ledger = P.ReservationLedger(memory)
        primary = ValueError("primary")
        with patch.object(memory, "release", side_effect=OSError("synthetic release failure")):
            with self.assertRaises(ValueError) as caught:
                with ledger.operation("raw synthetic seam", "raw_reference"):
                    memory.hold(object(), 20)
                    raise primary
        self.assertIs(caught.exception, primary)
        self.assertEqual(memory.live, 20)
        self.assertTrue(P.secondary_records(primary)[0])
        self.assertFalse(ledger.events[-1]["cleanup_complete"])
        self.assertIsNotNone(ledger.events[-1]["error"])
        P.validate_reservations(ledger.events)
        persist(self.directory/"failed_release.events.json", ledger.events)
        ledger.cleanup()
        self.assertEqual(memory.live, 0)


class Custody(OwnedCase):
    def owned(self):
        return P.OwnedDirectory(self.directory/"owned", os.getuid(), self.deadline, create=True)

    def test_exclusive_write_original_hash_and_close(self):
        d = self.owned()
        try:
            file = d.write("record.json", {"synthetic": True})
            self.assertEqual(d.read("record.json", file), b'{"synthetic":true}\n')
            self.assertEqual(d.names(), {"record.json"})
            self.assertIn(file["mode"], (0o600, 0o700))
            with self.assertRaises(FileExistsError):
                d.write("record.json", {"different": True})
        finally:
            d.release()
        self.assertIsNone(d.fd)
        self.assertIsNone(d.parent_fd)

    def test_replaced_namespace_rejected(self):
        d = self.owned()
        original = d.path
        original.rename(self.directory/"preserved-original")
        original.mkdir(mode=0o700)
        with self.assertRaises(P.ProbeError):
            d.check()
        with self.assertRaises(P.ProbeError):
            d.release()
        self.assertIsNone(d.fd)
        self.assertIsNone(d.parent_fd)

    def test_changed_owned_bytes_and_symlink(self):
        d = self.owned()
        file = d.write("record.json", {"x": 1})
        with open(d.path/"record.json", "r+b") as handle:
            handle.write(b'{"x":2}\n')
        with self.assertRaises(P.ProbeError):
            d.read("record.json", file)
        (d.path/"foreign-link").symlink_to("record.json")
        with self.assertRaises(OSError):
            d.read("foreign-link", {**file, "name": "foreign-link"})
        d.release()

    def test_sync_and_close_compound_preserve_primary(self):
        d = self.owned()
        primary = ValueError("same primary")
        real_close = os.close
        def closed_then_error(fd):
            real_close(fd)
            raise OSError("owned closed descriptor synthetic error")
        with patch.object(d, "sync", side_effect=OSError("synthetic sync failure")), \
             patch.object(P.os, "close", side_effect=closed_then_error):
            with self.assertRaises(ValueError) as caught:
                d.release(primary)
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(P.secondary_records(primary)[0]), 3)
        self.assertIsNone(d.fd)

    def test_software_payloads_and_caller_forbidden(self):
        inventory = P.Inventory(self.deadline)
        for name in ("real.npy", "real.pkl", "real.pt", "real.svs", "real.safetensors", "real.pth",
                     P.CALLER+".py"):
            with patch.object(P.os, "open", side_effect=AssertionError("no acquisition may occur")):
                with self.assertRaises(P.ProbeError):
                    inventory.read(str(self.directory/name))

    def test_same_run_source_mutation_rejected(self):
        path = self.directory/"owned.py"
        actual_owned(path, b"one\n")
        inventory = P.Inventory(self.deadline)
        inventory.read(str(path))
        with open(path, "r+b") as handle:
            handle.write(b"two\n")
        with self.assertRaises(P.ProbeError):
            inventory.sweep()


class ExitRecords(OwnedCase):
    def outcome(self, **updates):
        value = {"pid": 1234, "wait_state": "reaped_exit", "returncode": 0, "signal": None, "timed_out": False,
                 "started_monotonic_ns": 1, "last_observed_monotonic_ns": 9, "reaped_monotonic_ns": 8,
                 "deadline_monotonic_ns": 10, "deadline_met": True, "signal_attempts": []}
        value.update(updates)
        return value

    def test_unresolved_and_late_are_not_success(self):
        P.validate_outcome(self.outcome(), True)
        late = self.outcome(last_observed_monotonic_ns=12, reaped_monotonic_ns=11, deadline_met=False)
        P.validate_outcome(late)
        with self.assertRaises(P.ProbeError):
            P.validate_outcome(late, True)
        unresolved = self.outcome(wait_state="unresolved", returncode=None, reaped_monotonic_ns=None)
        P.validate_outcome(unresolved)
        with self.assertRaises(P.ProbeError):
            P.validate_outcome(unresolved, True)
        with self.assertRaises(P.ProbeError):
            P.validate_outcome(self.outcome(last_observed_monotonic_ns=11), True)

    def stream(self, **updates):
        file = {"name": "capture.stdout", "size": 0, "sha256": P.sha(b""), "st_dev": 1, "st_ino": 2,
                "uid": 1, "mode": 0o600}
        value = {"descriptor": {"path": "/owned/capture.stdout", "size": 0, "sha256": P.sha(b"")},
                 "identity": file, "limit_bytes": 16384, "received_bytes": 0, "retained_prefix_bytes": 0,
                 "written_bytes": 0, "eof_observed": True, "excess_observed": False, "reception_scope": "complete",
                 "received_prefix_sha256": P.sha(b""), "write_accounting": "exact_returned_counts",
                 "persistence": "verified_complete", "terminal_read": "eof", "error": None}
        value.update(updates)
        return value

    def test_actual_eof_excess_receive_write(self):
        P.validate_stream(self.stream(), True)
        for updates in ({"eof_observed": False}, {"excess_observed": True},
                        {"received_bytes": 1}, {"written_bytes": 1}, {"persistence": "verified_prefix"},
                        {"terminal_read": "drain_deadline"}, {"reception_scope": "known_prefix"}):
            with self.assertRaises(P.ProbeError):
                P.validate_stream(self.stream(**updates), True)

    def test_true_late_clock(self):
        now = time.monotonic_ns()
        value = P.clock((time.time_ns(), now-1000), now-1)
        self.assertFalse(value["deadline_met"])
        P.validate_clock(value)
        with self.assertRaises(P.ProbeError):
            P.validate_clock(value, True)


class BootProcesses(OwnedCase):
    """Actual owned stdlib children, with synthetic entry stubs explicitly labelled."""
    def child(self, name, body, expected, descriptor_change=None):
        directory = self.directory/name
        directory.mkdir(mode=0o700)
        source = directory/"synthetic_entry.py"
        text = ("import os,sys,hashlib\n"
                "_EMITTED=None\n"
                "def entry(argv,source):\n"+body)
        payload = text.encode()
        actual_owned(source, payload)
        actual_owned(directory/"source.snapshot.py", payload)
        exact = P.BOOT_LITERAL
        old = 'prescribed=os.path.join(cwd,"scripts","r2_1_filter_cpu_profile_probe.py")'
        self.assertEqual(exact.count(old), 1)
        literal = exact.replace(old, "prescribed="+repr(str(source)))
        actual_owned(directory/"BOOT.exact.literal.txt", exact.encode())
        actual_owned(directory/"BOOT.derived.unit.literal.txt", literal.encode())
        size, digest = len(payload), P.sha(payload)
        if descriptor_change == "bad_sha":
            digest = "0"*64
        args = [sys.executable, "-I", "-S", "-B", "-c", literal, str(source), str(size), digest, "capture"]
        env = {n: "1" for n in P.NATIVE_NAMES}
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        persist(directory/"launch.json", {"label": "derived_literal_synthetic_entry_not_exact_BOOT_or_capture",
                "derivation": "one prescribed-path assignment replaced with the owned stub path; no production option",
                "exact_BOOT_sha256": P.sha(exact.encode()), "derived_BOOT_sha256": P.sha(literal.encode()),
                "argv": args, "environment_exact": env, "source_size": size, "source_sha256": digest,
                "cwd": str(SOURCE.parent.parent)})
        stdout_path, stderr_path = directory/"stdout.bin", directory/"stderr.bin"
        actual_owned(stdout_path, b"")
        actual_owned(stderr_path, b"")
        started = time.monotonic_ns()
        with open(stdout_path, "r+b", buffering=0) as out, open(stderr_path, "r+b", buffering=0) as err:
            child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                     env=env, close_fds=True, cwd=SOURCE.parent.parent)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
                persist(directory/"timeout.json", {"owned_pid": child.pid, "actual_returncode": child.returncode})
                self.fail("owned synthetic BOOT child timeout")
            os.fsync(out.fileno())
            os.fsync(err.fileno())
        ended = time.monotonic_ns()
        stdout, stderr = stdout_path.read_bytes(), stderr_path.read_bytes()
        persist(directory/"outcome.json", {"owned_pid": child.pid, "returncode": child.returncode,
                "started_monotonic_ns": started, "ended_monotonic_ns": ended, "reaped": child.poll() is not None,
                "stdout_pipe_closed": None, "stderr_pipe_closed": None,
                "eof_witness_scope": "precreated_regular_files_and_reaped_process_not_raw_read_trace",
                "stdout_eof_observed": None, "stderr_eof_observed": None, "timed_out": False,
                "expected_returncode": expected, "stdout_bytes": len(stdout), "stderr_bytes": len(stderr),
                "source_original_sha256": P.sha(payload), "source_final_sha256": P.sha(source.read_bytes())})
        self.assertEqual(child.returncode, expected, stderr.decode(errors="replace"))
        self.assertLessEqual(len(stdout), 16384)
        self.assertLessEqual(len(stderr), 1048576)
        return stdout, stderr

    normal = ("    global _EMITTED\n"
              "    data=b'{}\\n'\n"
              "    os.write(1,data)\n"
              "    _EMITTED=hashlib.sha256(data).hexdigest()\n")
    returned = ("    return {'kind':'normal_success_return','mode':argv[0],"
                "'handoff_sha256':_EMITTED,'executing_source_sha256':source['sha256']}\n")

    def test_normal_stub_actual_exit(self):
        stdout, stderr = self.child("normal", self.normal+self.returned, 0)
        self.assertEqual(stdout, b"{}\n")
        self.assertEqual(stderr, b"")

    def test_invalid_returns_actual_exit(self):
        for value in ("None", "0", "1", "True", "{}"):
            self.child("return_"+value.replace("{", "dict").replace("}", ""), "    return "+value+"\n", 1)

    def test_systemexit_arbitration_actual_exit(self):
        for expression, expected in (("None", 1), ("0", 1), ("True", 1), ("False", 1), ("-1", 1),
                                     ("1", 1), ("125", 125), ("126", 1), ("256", 1), ("512", 1), ("'text'", 1)):
            self.child("exit_"+expression.replace("'", "").replace("-", "minus"),
                       "    raise SystemExit("+expression+")\n", expected)

    def test_normal_then_source_hash_failure_actual_exit(self):
        mutation = ("    with open(__file__,'r+b') as f:\n"
                    "        f.write(b'#')\n")
        stdout, _ = self.child("late_hash", self.normal+mutation+self.returned, 1)
        self.assertEqual(stdout, b"{}\n")

    def test_normal_then_close_failure_actual_exit(self):
        mutation = ("    real=os.close\n"
                    "    def close_then_fail(fd):\n"
                    "        real(fd)\n"
                    "        raise OSError('synthetic owned close failure')\n"
                    "    os.close=close_then_fail\n")
        self.child("late_close", self.normal+mutation+self.returned, 1)

    def test_normal_then_flush_failure_actual_exit(self):
        mutation = ("    class BadFlush:\n"
                    "        def flush(self):\n"
                    "            raise OSError('synthetic flush failure')\n"
                    "    sys.stdout=BadFlush()\n")
        self.child("late_flush", self.normal+mutation+self.returned, 1)

    def test_body_primary_compound_failure_actual_exit(self):
        self.child("primary_and_final_hash", "    with open(__file__,'r+b') as f:\n"
                   "        f.write(b'#')\n    raise SystemExit(125)\n", 125)

    def test_pre_envelope_bad_hash_actual_exit(self):
        stdout, stderr = self.child("bad_source_hash", self.normal+self.returned, 1, "bad_sha")
        self.assertEqual(stdout, b"")
        self.assertEqual(json.loads(stderr)["kind"], "pre_envelope_failure")


def software_graph():
    """Source-compatible fabricated graph; none of its paths/content were acquired."""
    envelope = {"schema_version": 1, "task": P.TASK, "run_id": "filter35-synthetic",
                "attempt_nonce": "a"*32, "spec_sha256": "b"*64, "private_sha256": "c"*64}
    value = {**envelope, "scope": "observed_startup_and_tiny_exercise_files_only",
             "files": [], "aliases": [], "roles": [], "modules": {p: [] for p in P.PHASES},
             "executable_mappings": [], "namespaces": [], "generated": {}, "native_pools": [],
             "version_metadata": [], "historical_comparisons": [], "historical_aliases": [],
             "historical_unobserved": [], "counts": {}, "limitations": list(P.NEW_LIMITATIONS),
             "fileless_objects": [], "runtime_bridge": []}

    def add_file(path):
        binding = {"path": path, "size": 10, "sha256": "a"*64, "st_dev": 1,
                   "st_ino": len(value["files"])+1}
        sid = P.software_id(path)
        value["files"].append({"asset_id": sid, "binding": binding, "first_phase": "bootstrap",
                              "phases": ["bootstrap"], "required": False, "final_binding": dict(binding)})
        value["aliases"].append({"lexical_path": path, "asset_id": sid, "phases": ["bootstrap"]})
        value["historical_comparisons"].append({"asset_id": sid, "canonical_path": path,
            "classification": "new", "prior_matches": [], "required": False})
        return sid

    for name, version in (("numpy", "2.2.4"), ("scipy", "1.16.0"), ("scikit-learn", "1.7.0"),
                          ("POT", "0.9.6.post1"), ("pandas", None), ("threadpoolctl", None)):
        root = "/synthetic/"+name+".dist-info"
        mid, rid = add_file(root+"/METADATA"), add_file(root+"/RECORD")
        value["version_metadata"].append({"distribution": name, "required_version": version,
            "observed_version": version or "synthetic", "metadata_asset_id": mid, "record_asset_id": rid,
            "metadata_path": root+"/METADATA", "record_path": root+"/RECORD", "distribution_root": root,
            "semantic_checks": ["single_wheel_metadata", "name_matches_distribution",
                               "metadata_version_matches_getter", "original_metadata_and_record_bytes"]})
    sid = add_file("/synthetic/module.py")
    value["modules"]["bootstrap"].append({**shape("synthetic", file="/synthetic/module.py",
        origin="/synthetic/module.py"), "phase": "bootstrap", "kind": "file",
        "file_asset_id": sid, "origin_asset_id": sid, "cached_asset_id": None, "cache_state": "not_declared"})
    value["roles"].append({"phase": "bootstrap", "role": "module:synthetic", "asset_id": sid,
                          "reported_path": "/synthetic/module.py", "occurrence": 0})
    return value, envelope


def recount_software(value):
    value["counts"] = {
        "canonical_files": len(value["files"]), "lexical_alias_paths": len(value["aliases"]),
        "role_occurrences": len(value["roles"]),
        "unique_content_bytes": sum(v["binding"]["size"] for v in value["files"]),
        **{p+"_modules": len(value["modules"][p]) for p in P.PHASES},
        **{p+"_maps": 0 for p in P.PHASES}, "before_native_pools": 0, "after_native_pools": 0,
        "required_files": sum(v["required"] for v in value["files"]),
        "new_files": len(value["files"]), "changed_files": 0, "matched_files": 0,
        "historical_unobserved_records": 0, "fileless_object_records": len(value["fileless_objects"]),
        "unresolved_creation_origins": len(value["fileless_objects"]),
        "runtime_bridge_rows": len(value["runtime_bridge"])}


def basis_record(dtype, dimensions, fortran=False):
    width, unit = (1536, 4) if dtype == "<f4" else (1535, 1)
    data = bytearray(len(dimensions)*width*unit)
    for row, column in enumerate(dimensions):
        if unit == 4:
            struct.pack_into("<f", data, (row*width+column)*unit, 1)
        else:
            data[row*width+column] = 1
    return {"dtype": dtype, "shape": [len(dimensions), width],
        "strides": [unit, unit*len(dimensions)] if fortran else [unit*width, unit],
        "c_contiguous": not fortran, "f_contiguous": fortran or len(dimensions) == 1,
        "c_order_bytes": len(data), "c_order_sha256": P.sha(data),
        "nonzero": [{"row": r, "column": c, "float_hex": 1.0.hex() if unit == 4 else None,
                     "uint_value": None if unit == 4 else 1} for r, c in enumerate(dimensions)]}


def exercise_records():
    """Entire Exercises schema made from stdlib records; no provider/state exercise."""
    _, envelope = software_graph()
    directory = {"path": "/synthetic/fixture", "st_dev": 1, "st_ino": 1, "uid": 1, "mode": 0o700}
    file = {"name": "fixture_manifest.json", "size": 10, "sha256": "a"*64,
            "st_dev": 1, "st_ino": 1, "uid": 1, "mode": 0o600}
    policy = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
    value = {**envelope, "scope": "tiny_generated_basis_vectors_only", "fixture_directory": directory,
        "fixture_manifest": file, "bundle_expected": {}, "requests": P.requests(), "decode_records": [],
        "preparations": [], "cells": [], "raw_consumptions": [], "numpy_policy_before": dict(policy),
        "numpy_policy_after": dict(policy), "dtype_evidence": {
            "directly_observed": {"raw_query": "<f4", "raw_reference": "<f4", "prepared_ot": "<f4",
                                 "prepared_bob": "|u1"},
            "frozen_call_validated": {"cost": "<f4", "masses": "<f8", "distance": "<f8"},
            "declared": {"lp_cost": "float64"}},
        "working_memory": {"peak_accounted_working_bytes": 100, "limit_bytes": 8*P.MIB,
            "basis": "conservative_buffers_and_known_reservations_not_rss"},
        "warnings": [], "checks": [{"name": "analytic_exact_values", "passed": True}],
        "limitations": list(P.NEW_LIMITATIONS)}
    for index, ordinal in [(0, 0), *((1, i) for i in range(8))]:
        raw = basis_record("<f4", [0, 1] if index == 0 else [0, 2], fortran=index == 1)
        consumption = {"asset_id": "feature:probe_q" if index == 0 else "feature:probe_r",
            "logical_name": "synthetic", "consumer_path": "/synthetic/fixture/"+(
                "query.pkl" if index == 0 else "reference.pkl"), "producer_size": 10,
            "producer_sha256": "a"*64, "semantic_checks": ["accepted_feature_bytes",
                "frozen_tuple_FP32_1536_decoder", "complete_ordered_coordinate_slice_match",
                "decoded_layout_preserved"]}
        value["decode_records"].append({"slide_index": index, "decode_ordinal": ordinal,
            "layout": {"dtype_str": "<f4", "shape": [2, 1536], "strides": raw["strides"],
                       "c_contiguous": index == 0, "f_contiguous": index == 1},
            "consumption": dict(consumption), "raw": raw})
        value["raw_consumptions"].append(consumption)
    for req in P.requests():
        filtered, ot = req["budget_percent"] == 5, req["method"] == "ot_cosine"
        expected = (1.0 if ot else 2.0) if filtered else (.5 if ot else 1.0)
        dimensions = ([1], [0]) if filtered else ([0, 1], [0, 2])
        arrays, ids = [], []
        for index in (0, 1):
            dtype = "<f4" if ot else "|u1"
            prepared = basis_record(dtype, dimensions[index])
            arrays.append(prepared)
            pid = f"p{len(value['preparations']):010d}"
            ids.append(pid)
            row = {name: None for name in P.PREPARATION_KEYS}
            row.update(preparation_id=pid, method=req["method"], selection=req["selection"],
                budget_percent=req["budget_percent"], query_fold=req["query_fold"], slide_index=index,
                slide_id="probe_q" if index == 0 else "probe_r", raw_rows=2,
                selected_rows=len(dimensions[index]))
            for prefix in ("raw", "selected", "prepared"):
                array = prepared if prefix == "prepared" else basis_record(
                    "<f4", [0, 1] if index == 0 else [0, 2]) if prefix == "raw" else basis_record(
                        "<f4", dimensions[index])
                row.update({prefix+"_dtype": array["dtype"],
                    prefix+"_shape": json.dumps(array["shape"], separators=(",", ":")),
                    prefix+"_strides": json.dumps(array["strides"], separators=(",", ":")),
                    prefix+"_c_contiguous": array["c_contiguous"],
                    prefix+"_f_contiguous": array["f_contiguous"]})
            value["preparations"].append(row)
        obs = {name: None for name in P.OBSERVATION_KEYS}
        obs.update(request_id=req["request_id"], query_index=0, reference_index=1, reference_id="probe_r",
            query_preparation_id=ids[0], reference_preparation_id=ids[1], same_patient_excluded=False,
            zero_query=False, zero_reference=False, observation_status="valid_distance", computed_valid=True,
            distance_hex=expected.hex(), cosine_min_hex=(1.0 if filtered else 0.0).hex() if ot else None,
            cosine_max_hex=1.0.hex() if ot else None, cosine_negative_count=0 if ot else None,
            solver_result_code=1 if ot else None, solver_warning_is_none=True if ot else None,
            captured_warning_count=0, scalar_distance_hex=expected.hex() if ot else None,
            scalar_log_comparison="equal" if ot else "not_applicable", baseline_comparison="not_applicable",
            literal_comparison="not_applicable", baseline_distance_comparison="not_applicable",
            baseline_rank_comparison="not_applicable")
        if ot:
            obs["solver_warning"] = None
        value["cells"].append({"request": dict(req), "observation": obs,
            "query_preparation_id": ids[0], "reference_preparation_id": ids[1],
            "query_prepared": arrays[0], "reference_prepared": arrays[1],
            "expected_distance_hex": expected.hex(), "actual_distance_hex": expected.hex(),
            "expected_equal": True, "selected_rows": {
                "query": [1] if filtered else [0, 1], "reference": [0] if filtered else [0, 1]},
            "literal_equal": True if req["selection"] == "all_true" or req["budget_percent"] == 100 else None,
            "scalar_equal": True if ot else None})
    descriptor = {"path": "/synthetic/fixture/baseline.pkl", "size": 10, "sha256": "a"*64}
    baseline_consumption = {"asset_id": "baseline:synthetic:ot_cosine", "logical_name": "baseline.pkl",
        "consumer_path": descriptor["path"], "producer_size": 10, "producer_sha256": "a"*64,
        "semantic_checks": ["generated_original_pickle_bytes", "unchanged_baseline_dataframe_decoder",
                           "exact_string_axes_2x2_FP64_values", "working_reservation_accounted"]}
    value["baseline"] = {"method": "ot_cosine", "asset_id": "baseline:synthetic:ot_cosine",
        "descriptor": descriptor, "consumption": baseline_consumption, "decoder": "unchanged_F.load_baseline",
        "load_calls": 1, "raw_ids": ["1", "2"], "index": ["1", "2"], "columns": ["1", "2"],
        "row_lookup": {"1": 0, "2": 1}, "column_lookup": {"1": 0, "2": 1},
        "shape": [2, 2], "dtype": "<f8", "strides": [16, 8], "c_contiguous": True,
        "f_contiguous": False, "values_hex": P.BASELINE_HEX, "expected_equal": True,
        "reservation_bytes": 4202, "reservation_released": True, "working_live_before": 100,
        "working_live_after": 100, "scope": "synthetic_decoder_only_not_baseline_equivalence"}
    memory = MemorySeams.Memory()
    ledger = P.ReservationLedger(memory)
    for ordinal, req in enumerate(P.requests()):
        references = []
        for stage in ("query_event", "raw_reference", "prepare_reference"):
            with ledger.operation("synthetic", stage):
                token = object()
                memory.hold(token, 1)
                if stage == "query_event" and ordinal == 0:
                    memory.hold(object(), 1)
                if stage != "query_event":
                    references.append(token)
        for token in references:
            ledger.release(token)
    with ledger.operation("synthetic", "baseline_load"):
        baseline_token = object()
        memory.hold(baseline_token, 4202)
    ledger.release(baseline_token)
    value["reservations"] = copy.deepcopy(ledger.events)
    value["working_memory"]["peak_accounted_working_bytes"] = memory.peak
    value["baseline"]["working_live_before"] = value["baseline"]["working_live_after"] = memory.live
    ledger.cleanup()
    return value, envelope


class RepairRegressions(OwnedCase):
    def test_unavailable_secondary_formatter_does_not_fabricate_arguments(self):
        primary = ValueError("original primary")
        additional = OSError("genuine secondary")
        with patch.object(P, "error_record", side_effect=ValueError("formatter failed")):
            P.add_secondary(primary, additional, "fixed")
        record = P.secondary_records(primary)[0][0]
        self.assertEqual(record["type"], "builtins.OSError")
        self.assertEqual(record["reason"]["kind"], "unavailable")
        self.assertEqual(record["reason"]["fragments"], [])
        self.assertEqual(record["formatting_failure_count"], 1)
        self.assertIs(P._SECONDARIES[id(primary)][0], primary)
        P.validate_error(record)
        persist(self.directory/"unavailable_original_secondary.json", record)

    def test_descriptor_safe_initial_and_exposure_seams(self):
        calls = []
        class Module(types.ModuleType):
            @property
            def __dict__(self):
                calls.append("dictionary")
                raise AssertionError("descriptor forbidden")
            @property
            def __spec__(self):
                calls.append("spec")
                raise AssertionError("descriptor forbidden")
        owned = Module("owned")
        namespace = P._MODULE_DICT.__get__(owned)
        namespace["__spec__"] = importlib.machinery.ModuleSpec("owned", None, origin="frozen")
        context = types.SimpleNamespace(initial_authority={"sys_path": ["/synthetic"],
            "imported_sources": [{"module": "owned", "origin": "frozen", "source": None,
                                 "associated_cache": None}]}, private={"sys_path_pre_site": ["/synthetic"]})
        with patch.object(P.sys, "path", ["/synthetic"]), patch.dict(P.sys.modules, {"owned": owned}, clear=True):
            INITIAL_GATE(context)
        provider = Module("typing")
        leaf = types.ModuleType("typing.io")
        actual_owned(self.directory/"provider.py", b"# owned stdlib exposure fixture\n")
        ns = P._MODULE_DICT.__get__(provider)
        ns.update(__file__=str(self.directory/"provider.py"), io=leaf)
        with patch.dict(P.sys.modules, {"typing": provider}):
            exposure = P.observe_exposure("typing.io", leaf, P.Inventory(self.deadline))
        self.assertEqual(exposure["status"], "runtime_attribute_identity")
        self.assertEqual(calls, [])
        persist(self.directory/"exposure.json", exposure)

    def test_original_primary_survives_serialization_and_cleanup_format_failures(self):
        primary = ValueError("original serialization failure")
        ctx = types.SimpleNamespace()
        def synthetic_capture(context):
            memory = MemorySeams.Memory()
            ledger = P.ReservationLedger(memory)
            with ledger.operation("synthetic", "query_event"):
                memory.hold(object(), 20)
            context.reservation_ledger = ledger
            context.exercise_state = types.SimpleNamespace(
                memory=memory, query_raw={0: object()}, query_prepared={0: object()})
            raise primary
        with patch.object(P, "_capture", new=synthetic_capture), \
                patch.object(P, "encoded", side_effect=OSError("actual owned formatting seam")):
            with self.assertRaises(ValueError) as caught:
                CAPTURE_WRAPPER(ctx)
        self.assertIs(caught.exception, primary)
        self.assertEqual(ctx.exercise_state.memory.live, 0)
        self.assertFalse(ctx.exercise_state.query_raw or ctx.exercise_state.query_prepared)
        self.assertTrue(P.secondary_records(primary)[0])
        persist(self.directory/"same_primary.json", {
            "primary": P.error_record(primary, "labelled wrapper and formatting seams"),
            "secondary": P.secondary_records(primary)[0], "fake_working_live": 0,
            "native_capture_executed": False})

    def test_actual_initial_fstat_and_close_compound_failure(self):
        d = P.OwnedDirectory(self.directory/"owned", os.getuid(), self.deadline, create=True)
        real_open, real_close, real_fstat = os.open, os.close, os.fstat
        opened = []
        primary = OSError("original initial fstat error")
        def tracked(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            if path == "partial.json":
                opened.append(fd)
            return fd
        def failed(fd):
            if fd in opened:
                raise primary
            return real_fstat(fd)
        def close_then_fail(fd):
            real_close(fd)
            if fd in opened:
                raise OSError("closed owned descriptor secondary")
        try:
            with patch.object(P.os, "open", new=tracked), patch.object(P.os, "fstat", new=failed), \
                    patch.object(P.os, "close", new=close_then_fail):
                with self.assertRaises(OSError) as caught:
                    d.write("partial.json", {"x": 1})
            self.assertIs(caught.exception, primary)
            with self.assertRaises(OSError) as closed:
                real_fstat(opened[0])
            self.assertEqual(closed.exception.errno, 9)
            self.assertEqual(len(P.secondary_records(primary)[0]), 1)
            persist(self.directory/"compound_fd.json", {
                "closed_EBADF": True, "same_primary": P.error_record(primary, "initial fstat"),
                "secondary": P.secondary_records(primary)[0]})
        finally:
            d.release()

    def test_source_safe_fallback_export_marker_and_note_format_accounting(self):
        note = ("Bounded returned OT interface evidence: formatting unavailable; "
                "request_id=ot_filter5; result_code_type=builtins.int; warning_type=builtins.NoneType; "
                "solver_result_code_export=not exported; numeric cell=null; sign=negative; "
                "bit_length=257; diagnostic domain=256 bits; formatting_error_type=builtins.ValueError")
        error = ValueError("OT distance: nonfinite scalar: nan")
        error.add_note(note)
        partial = P.partial_observation({"solver_result_code": None}, error, "ot_filter5")
        self.assertEqual(partial["export_state"], "outside_256_bits")
        self.assertEqual(partial["solver_raw_evidence"], note)
        original = P.bounded_detail
        def formatting(value):
            if value == note:
                raise ValueError("genuine formatter failure seam")
            return original(value)
        with patch.object(P, "bounded_detail", new=formatting):
            record = P.error_record(error, "fixed")
        self.assertEqual(record["formatting_failure_count"], 1)
        self.assertEqual((record["source_note_count"], record["omitted_note_count"]), (1, 1))
        P.validate_error(record)
        persist(self.directory/"fallback_partial.json", partial)
        persist(self.directory/"formatter_omissions.json", record)

    def test_complete_Exercises_schema_roundtrip_and_coordinated_negatives(self):
        value, envelope = exercise_records()
        P.validate_exercises(value, envelope)
        literal = P.encoded(value, 8*P.MIB)
        P.validate_exercises(P.strict_json(literal, canonical=True), envelope)
        actual_owned(self.directory/"positive_exercises.synthetic.json", literal)
        producer = next(n for n in ast.parse(SOURCE.read_bytes()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "exercise")
        emitted = next(n.value for n in ast.walk(producer) if isinstance(n, ast.Return)
                       and isinstance(n.value, ast.Dict) and any(isinstance(k, ast.Constant)
                           and k.value == "baseline" for k in n.value.keys))
        self.assertEqual(set(value), set(envelope)|{
            k.value for k in emitted.keys if isinstance(k, ast.Constant)})
        for index in (2, 6):
            bad = copy.deepcopy(value)
            bad["cells"][index]["reference_prepared"] = basis_record("<f4" if index == 2 else "|u1", [2])
            with self.assertRaises(P.ProbeError):
                P.validate_exercises(bad, envelope)
        bad = copy.deepcopy(value)
        del bad["reservations"]
        with self.assertRaises(P.ProbeError):
            P.validate_exercises(bad, envelope)
        for field, replacement in (("stage", "decode"), ("cleanup_complete", False),
                                   ("stage", "final_cleanup")):
            bad = copy.deepcopy(value)
            bad["reservations"][-1][field] = replacement
            with self.assertRaises(P.ProbeError):
                P.validate_exercises(bad, envelope)
    def test_exact_BOOT_rejects_arbitrary_path_and_leaf_symlink(self):
        source = self.directory/"unreleased_stub.py"
        payload = b"raise AssertionError('must never execute this source')\n"
        actual_owned(source, payload)
        alias = self.directory/"leaf_link.py"
        alias.symlink_to(source.name)
        for index, path in enumerate((source, alias)):
            directory = self.directory/("child_"+str(index))
            directory.mkdir(mode=0o700)
            actual_owned(directory/"source.snapshot.py", payload)
            actual_owned(directory/"BOOT.exact.literal.txt", P.BOOT_LITERAL.encode())
            argv = [sys.executable, "-I", "-S", "-B", "-c", P.BOOT_LITERAL,
                    str(path), str(len(payload)), P.sha(payload), "capture"]
            env = {**{n: "1" for n in P.NATIVE_NAMES}, "CUDA_VISIBLE_DEVICES": "",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
            persist(directory/"launch.json", {"argv": argv, "environment_exact": env,
                "cwd": str(SOURCE.parent.parent), "scope": "exact_BOOT_negative_path_no_entry_execution",
                "leaf_symlink_literal": source.name if index else None})
            actual_owned(directory/"stdout.bin", b"")
            actual_owned(directory/"stderr.bin", b"")
            with open(directory/"stdout.bin", "r+b", buffering=0) as out, \
                    open(directory/"stderr.bin", "r+b", buffering=0) as err:
                child = subprocess.Popen(argv, cwd=SOURCE.parent.parent, env=env, close_fds=True,
                                         stdin=subprocess.DEVNULL, stdout=out, stderr=err)
                timed_out = False
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    child.kill()
                    child.wait(timeout=5)
                os.fsync(out.fileno())
                os.fsync(err.fileno())
            persist(directory/"outcome.json", {"owned_pid": child.pid, "returncode": child.returncode,
                "reaped": child.poll() is not None, "timed_out": timed_out,
                "raw_EOF_observed": None, "scope": "precreated_regular_streams_not_raw_EOF_trace"})
            self.assertFalse(timed_out)
            self.assertEqual(child.returncode, 1)
            self.assertEqual((directory/"stdout.bin").read_bytes(), b"")
            failure = P.strict_json((directory/"stderr.bin").read_bytes(), canonical=True)
            self.assertEqual(failure["kind"], "pre_envelope_failure")
            self.assertIsNone(failure["validated_executing_source_sha256"])

    def test_source_handoff_path_not_just_hash_or_size(self):
        package = "/synthetic/prosper-search"
        path = package+"/scripts/r2_1_filter_cpu_profile_probe.py"
        spec = {"package_root": package, "probe_source": {
            "path": path, "size": 10, "sha256": "a"*64}}
        handoff = {**spec["probe_source"], "st_dev": 1, "st_ino": 1}
        P.source_handoff_agreement(spec, handoff)
        for key, value in (("path", "/synthetic/other.py"), ("size", 11), ("sha256", "b"*64)):
            with self.assertRaises(P.ProbeError):
                P.source_handoff_agreement(spec, {**handoff, key: value})

    def test_source_basis_and_reservation_schema_projection(self):
        tree = ast.parse(SOURCE.read_bytes())
        producer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "exercise")
        returned = next(n.value for n in ast.walk(producer) if isinstance(n, ast.Return)
                        and isinstance(n.value, ast.Dict) and any(
                            isinstance(k, ast.Constant) and k.value == "baseline" for k in n.value.keys))
        self.assertIn("reservations", [k.value for k in returned.keys if isinstance(k, ast.Constant)])
        for function, target in (("exercise", "qbits"), ("validate_exercises", "dims")):
            definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
            assignment = next(n for n in ast.walk(definition) if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == target or isinstance(t, ast.Tuple) and any(
                    isinstance(v, ast.Name) and v.id == target for v in t.elts) for t in n.targets))
            self.assertEqual(ast.literal_eval(assignment.value.body), ([1], [0]))
        for dtype, width, unit in (("<f4", 1536, 4), ("|u1", 1535, 1)):
            payload = bytearray(width*unit)
            if unit == 4:
                struct.pack_into("<f", payload, 0, 1)
            else:
                payload[0] = 1
            record = {"dtype": dtype, "shape": [1, width], "strides": [width*unit, unit],
                "c_contiguous": True, "f_contiguous": True, "c_order_bytes": len(payload),
                "c_order_sha256": P.sha(payload), "nonzero": [{
                    "row": 0, "column": 0, "float_hex": 1.0.hex() if unit == 4 else None,
                    "uint_value": None if unit == 4 else 1}]}
            P.array_record(record, dtype, [0])
            with self.assertRaises(P.ProbeError):
                P.array_record(record, dtype, [2])
            persist(self.directory/("basis_"+str(unit)+".json"), record)

    def test_exact_ledger_stages_and_failed_final_cleanup(self):
        memory = MemorySeams.Memory()
        ledger = P.ReservationLedger(memory)
        primary = ValueError("original helper failure")
        with patch.object(memory, "release", side_effect=OSError("release remains held")):
            with self.assertRaises(ValueError) as caught:
                with ledger.operation("synthetic raw", "raw_reference"):
                    memory.hold(object(), 20)
                    raise primary
            self.assertIs(caught.exception, primary)
            with self.assertRaises(ValueError) as final:
                ledger.cleanup(primary)
            self.assertIs(final.exception, primary)
        P.validate_reservations(ledger.events)
        event = ledger.events[-1]
        self.assertEqual(event["stage"], "final_cleanup")
        self.assertEqual((event["after_live"], event["after_retained"]), (20, 1))
        self.assertFalse(event["cleanup_complete"])
        self.assertIsNotNone(event["error"])
        persist(self.directory/"held_after_failure.json", ledger.events)
        # Test-owned fake accounting is released only after recording the failed attempt.
        ledger.cleanup()
        self.assertEqual(memory.live, 0)
        for stage in ("decode", "prepare", "raw", "unknown"):
            with self.assertRaises(P.ProbeError):
                with ledger.operation("invalid", stage):
                    self.fail("invalid stage entered")

    def test_query_cache_lives_through_actual_bounded_serialization(self):
        memory = MemorySeams.Memory()
        ledger = P.ReservationLedger(memory)
        with ledger.operation("synthetic composite", "query_event"):
            memory.hold(object(), 20)
        state = types.SimpleNamespace(memory=memory, query_raw={0: object()}, query_prepared={0: object()})
        record = {"reservations": ledger.events, "scope": "labelled serialization seam"}
        calls = []
        original = P.encoded
        def observe(value, cap):
            if value is record:
                calls.append((bool(state.query_raw), bool(state.query_prepared), memory.live))
            return original(value, cap)
        with patch.object(P, "encoded", new=observe):
            literal = P.serialize_exercises(record, state, ledger)
        self.assertEqual(calls, [(True, True, 20)])
        snapshot = P.strict_json(literal)["reservations"]
        self.assertEqual(snapshot[-1]["stage"], "query_event")
        self.assertEqual(snapshot[-1]["new_reservations"][0]["disposition"], "retained")
        self.assertEqual(ledger.events[-1]["stage"], "final_cleanup")
        self.assertFalse(state.query_raw or state.query_prepared)
        self.assertEqual(memory.live, 0)
        P.validate_reservations(snapshot, True)
        P.validate_reservations(ledger.events, True)
        actual_owned(self.directory/"serialized_ledger.json", literal)
        bad = copy.deepcopy(ledger.events)
        bad[0]["extra"] = 1
        with self.assertRaises(P.ProbeError):
            P.validate_reservations(bad)

    def test_quiet_actual_context_parser_and_canonical_early_route(self):
        base = ["capture", "--spec", "/unopened/spec", "--spec-sha256", "a"*64,
                "--private", "/unopened/private", "--private-sha256", "b"*64,
                "--deadline-monotonic-ns", str(self.deadline)]
        self.assertEqual(P.parse_entry_arguments(base).mode, "capture")
        variants = [[], ["unsupported", *base[1:]], base+["--unknown=RAW_CANARY"],
                    base+["--spec", "RAW_CANARY"], base[:-2],
                    [base[0], "--spec=/RAW_CANARY", *base[3:]],
                    base+["--capture-witness", "/RAW_CANARY"], base+["--help"],
                    base+["-h"], base+["--", "RAW_CANARY"]]
        for argv in variants:
            ctx = P.Context.__new__(P.Context)
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                    patch.object(P, "Inventory", side_effect=AssertionError("must not acquire")):
                with self.assertRaises(P.ProbeError):
                    CONTEXT_INIT(ctx, argv, {})
            self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
        error = P.ProbeError("fixed entry option rejection")
        ctx = types.SimpleNamespace(envelope=None, stage="bootstrap_args")
        fd = os.open(self.directory/"early.stderr", os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        real_write = os.write
        def redirect(fd_number, data):
            self.assertEqual(fd_number, 2)
            return real_write(fd, data)
        try:
            with patch.object(P, "_ACTIVE_CONTEXT", ctx), patch.object(P.os, "write", new=redirect):
                P.boot_error(error, None)
        finally:
            os.close(fd)
        literal = (self.directory/"early.stderr").read_bytes()
        self.assertEqual(P.strict_json(literal, canonical=True)["kind"], "pre_envelope_failure")
        self.assertNotIn(b"RAW_CANARY", literal)

    def test_descriptor_safe_modules_specs_types_and_errors(self):
        calls = []
        class Module(types.ModuleType):
            @property
            def __dict__(self):
                calls.append("module_dict")
                raise AssertionError("property forbidden")
        class Spec(importlib.machinery.ModuleSpec):
            @property
            def __dict__(self):
                calls.append("spec_dict")
                raise AssertionError("property forbidden")
        module = Module("owned")
        P._MODULE_DICT.__get__(module)["__spec__"] = Spec("owned", None, origin="frozen")
        self.assertEqual(P._module_shape("owned", module)["origin"], "frozen")
        class Error(Exception):
            @property
            def args(self):
                calls.append("args")
                return ("fabricated",)
            @property
            def __notes__(self):
                calls.append("notes")
                raise AssertionError("property forbidden")
        error = Error("genuine")
        P._ERROR_DICT.__get__(error)["__notes__"] = ["genuine note"]
        record = P.error_record(error, "actual base slots")
        self.assertEqual(record["reason"]["fragments"][0]["text"], "genuine")
        self.assertEqual(record["notes"][0]["text"], "genuine note")
        self.assertEqual(record["formatting_failure_count"], 0)
        self.assertEqual(calls, [])
        P.validate_error(record)
        with patch.object(P, "bounded_detail", side_effect=[ValueError("real formatter seam"),
                {"text": "fixed", "original_characters": 5, "retained_source_characters": 5,
                 "omitted_source_characters": 0, "truncated": False}]):
            formatted = P.error_record(ValueError("real original"), "fixed")
        self.assertEqual(formatted["formatting_failure_count"], 1)
        P.validate_error(formatted)
        persist(self.directory/"genuine_error.json", record)
        persist(self.directory/"actual_formatter_failure.json", formatted)

    def test_no_arbitrary_search_location_iterator(self):
        class Locations:
            def __iter__(self):
                raise AssertionError("arbitrary iterator forbidden")
        module = types.ModuleType("owned")
        module.__spec__ = importlib.machinery.ModuleSpec("owned", None)
        module.__spec__.submodule_search_locations = Locations()
        with self.assertRaises(P.ProbeError):
            P._module_shape("owned", module)
        # A stored stdlib namespace-path representation, without running its constructor/iterator.
        path = P._NAMESPACE_PATH.__new__(P._NAMESPACE_PATH)
        P._NAMESPACE_DICT.__get__(path)["_path"] = ["/synthetic/namespace"]
        module.__spec__.submodule_search_locations = path
        with patch.object(P._NAMESPACE_PATH, "__iter__", side_effect=AssertionError("no iteration")):
            self.assertEqual(P._module_shape("owned", module)["search_locations"], ["/synthetic/namespace"])

    def test_warning_marker_is_not_export_context(self):
        for words in ("diagnostic domain=256 bits", '"solver_result_code_export"'):
            context = {"request_id": "cell", "result_code_type": "builtins.int",
                       "warning_type": "builtins.str"}
            note = "Bounded returned OT interface evidence: "+json.dumps(context)+"; details: "+json.dumps({
                "result_code": {"type": "builtins.int", "representation": "1"}, "warning": words})
            error = ValueError("OT distance: nonfinite scalar: nan")
            error.add_note(note)
            value = P.partial_observation({"solver_result_code": None}, error, "cell")
            self.assertEqual(value["export_state"], "capture_export_not_reached")
            self.assertIsNone(value["solver_result_code"])
            self.assertEqual(value["solver_raw_evidence"], note)
            persist(self.directory/("marker_"+str(len(words))+".json"), value)

    def test_software_graph_joins_and_observed_only_fileless(self):
        value, envelope = software_graph()
        catalog = P.catalog_from_metadata(metadata_catalog())
        def validate(v):
            recount_software(v)
            return P.validate_software(v, envelope, catalog, {"ordinary_namespaces": []}, {"modules": {}})
        validate(value)
        persist(self.directory/"positive_software.json", value)
        for fields in (("file_asset_id", "origin_asset_id"), ("origin_asset_id",)):
            bad = copy.deepcopy(value)
            for field in fields:
                bad["modules"]["bootstrap"][0][field] = None
            with self.assertRaises(P.ProbeError):
                validate(bad)
        fileless = {**shape("cython_runtime"), "phase": "bootstrap", "kind": "recorded_fileless_object",
                    "file_asset_id": None, "origin_asset_id": None, "cached_asset_id": None,
                    "cache_state": "not_declared"}
        value["modules"]["bootstrap"].insert(0, fileless)
        with self.assertRaises(P.ProbeError):
            validate(value)
        sidecar = {"phase": "bootstrap", "key": "cython_runtime", "name": "cython_runtime",
            "object_id": "object:00000000", "representation": "known_all_null",
            "historical_catalog_sha256": P.CATALOG_HASH, "creation_origin": "unresolved",
            "defining_provider_asset_id": None, "exposure": {"status": "unresolved", "provider_key": None,
                "provider_source_asset_id": None, "attribute_path": [], "relation": None}}
        value["fileless_objects"].append(sidecar)
        validate(value)
        persist(self.directory/"positive_observed_fileless.json", value)
        for mutation in ("duplicate", "unobserved", "cache"):
            bad = copy.deepcopy(value)
            if mutation == "duplicate":
                bad["fileless_objects"].append(copy.deepcopy(sidecar))
            elif mutation == "unobserved":
                bad["modules"]["bootstrap"].pop(0)
            else:
                bad["modules"]["bootstrap"][0]["cached_path"] = "/synthetic/cache.pyc"
            with self.assertRaises(P.ProbeError):
                validate(bad)

    def test_Torch_namespace_observed_provenance_and_cache_joins(self):
        value, envelope = software_graph()
        catalog = P.catalog_from_metadata(metadata_catalog())
        def validate(v):
            recount_software(v)
            P.validate_software(v, envelope, catalog, {"ordinary_namespaces": []}, {"modules": {}})
        ids = {}
        for name, size, digest in (
            ("torch", 95190, "e1b5bf4151bd246a21e3ac5318f6edfcf2e6ef0fd4f233bfb77a264698ec1319"),
            ("torch._ops", 55360, "7df9e7998b6c2343f4ae5470a7b21d458236ba9e8d27035c14a670441308028c")):
            path = "/synthetic/"+name+".py"
            sid = P.software_id(path)
            ids[name] = sid
            binding = {"path": path, "size": size, "sha256": digest, "st_dev": 1,
                       "st_ino": len(value["files"])+1}
            value["files"].append({"asset_id": sid, "binding": binding, "first_phase": "bootstrap",
                "phases": ["bootstrap"], "required": False, "final_binding": dict(binding)})
            value["aliases"].append({"lexical_path": path, "asset_id": sid, "phases": ["bootstrap"]})
            value["roles"].append({"phase": "bootstrap", "role": "module:"+name, "asset_id": sid,
                                  "reported_path": path, "occurrence": 0})
            value["historical_comparisons"].append({"asset_id": sid, "canonical_path": path,
                "classification": "new", "prior_matches": [], "required": False})
            value["modules"]["bootstrap"].append({**shape(name, file=path, origin=path),
                "phase": "bootstrap", "kind": "file", "file_asset_id": sid, "origin_asset_id": sid,
                "cached_asset_id": None, "cache_state": "not_declared"})
        value["modules"]["bootstrap"].append({**shape("torch.ops", file="_ops.py"),
            "phase": "bootstrap", "kind": "torch_namespace", "file_asset_id": None,
            "origin_asset_id": None, "cached_asset_id": None, "cache_state": "not_declared"})
        value["modules"]["bootstrap"].sort(key=lambda row: row["key"])
        with self.assertRaises(P.ProbeError):
            validate(value)
        value["namespaces"].append({"name": "torch.ops", "reported_file": "_ops.py",
            "defining_module": "torch._ops", "defining_asset_id": ids["torch._ops"],
            "torch_asset_id": ids["torch"]})
        validate(value)
        persist(self.directory/"synthetic_namespace_positive.json", value)
        for kind in ("duplicate", "absent_module", "wrong_provider"):
            bad = copy.deepcopy(value)
            if kind == "duplicate":
                bad["namespaces"].append(copy.deepcopy(bad["namespaces"][0]))
            elif kind == "absent_module":
                bad["modules"]["bootstrap"] = [m for m in bad["modules"]["bootstrap"] if m["key"] != "torch.ops"]
            else:
                next(m for m in bad["modules"]["bootstrap"] if m["key"] == "torch._ops")["file_asset_id"] = ids["torch"]
            with self.assertRaises(P.ProbeError):
                validate(bad)
        value, envelope = software_graph()
        module = value["modules"]["bootstrap"][0]
        # Distinct lexical cache alias and role to an explicit synthetic content record.
        sid = value["files"][0]["asset_id"]
        cache = "/synthetic/cache.pyc"
        value["aliases"].append({"lexical_path": cache, "asset_id": sid, "phases": ["bootstrap"]})
        module.update(cached_path=cache, cached_asset_id=sid, cache_state="present")
        value["roles"].append({"phase": "bootstrap", "role": "cache:synthetic", "asset_id": sid,
                              "reported_path": cache, "occurrence": 0})
        validate(value)
        for mutation in ("missing_role", "wrong_alias", "absent_with_id"):
            bad = copy.deepcopy(value)
            if mutation == "missing_role":
                bad["roles"].pop()
            elif mutation == "wrong_alias":
                bad["modules"]["bootstrap"][0]["cached_asset_id"] = bad["files"][1]["asset_id"]
            else:
                bad["modules"]["bootstrap"][0]["cache_state"] = "absent"
            with self.assertRaises(P.ProbeError):
                validate(bad)

    def test_runtime_base_ids_join_real_record_roles(self):
        record = runtime_record_fixture()
        path = "/synthetic/implementation.py"
        binding = {"path": path, "size": 10, "sha256": "a"*64, "st_dev": 1, "st_ino": 1}
        bridge = P.runtime_bridge(binding, ["before", "after"])
        software = {"runtime_bridge": [bridge], "aliases": [{
            "lexical_path": path, "asset_id": P.software_id(path)}], "native_pools": [],
            "roles": [{"phase": "before", "role": name, "asset_id": P.software_id(path),
                       "reported_path": path, "occurrence": 0} for name in record["observed_paths"]]}
        P.runtime_links(record, "before", software)
        for name in ("numpy", "scipy", "sklearn", "SGDClassifier"):
            bad = copy.deepcopy(record)
            bad["profile"]["base_profile"]["implementation"][name]["path"] = "runtime:impl:"+"0"*64
            with self.assertRaises(P.ProbeError):
                P.validate_runtime(bad)
            with self.assertRaises(P.ProbeError):
                P.runtime_links(bad, "before", software)
        bad = copy.deepcopy(software)
        bad["roles"] = [r for r in bad["roles"] if r["role"] != "base:numpy"]
        with self.assertRaises(P.ProbeError):
            P.runtime_links(record, "before", bad)

    def test_exact_unique17_asset_role_path_guard_membership(self):
        directory = "/synthetic/fixture"
        manifest, bindings = {"files": {}}, []
        for ordinal, name in enumerate((*P.DATA_NAMES, *P.CONTROL_NAMES), 1):
            aid, role = P.fixture_identity(name)
            manifest["files"][name] = {"size": 10, "sha256": "a"*64, "st_dev": 1, "st_ino": ordinal}
            bindings.append({"asset_id": aid, "role": role, "path": directory+"/"+name,
                "resolved_path": directory+"/"+name, "size": 10, "sha256": "a"*64,
                "device": 1, "inode": ordinal})
        P.validate_fixture_bindings(bindings, manifest, directory)
        persist(self.directory/"unique17.json", {"manifest": manifest, "bindings": bindings})
        variants = [[bindings[0]]*17, bindings[:-1]]
        for key, value in (("role", "wrong"), ("path", directory+"/reference.pkl"),
                           ("asset_id", "fixture:fixture_manifest.json"), ("inode", 100)):
            bad = copy.deepcopy(bindings)
            bad[0][key] = value
            variants.append(bad)
        for bad in variants:
            with self.assertRaises(P.ProbeError):
                P.validate_fixture_bindings(bad, manifest, directory)

    def test_initial_fstat_failure_closes_actual_owned_fd(self):
        d = P.OwnedDirectory(self.directory/"owned", os.getuid(), self.deadline, create=True)
        real_open, real_fstat, opened = os.open, os.fstat, []
        primary = OSError("initial owned fstat failure")
        def tracked_open(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            if path == "partial.json":
                opened.append(fd)
            return fd
        def failed(fd):
            if fd in opened:
                raise primary
            return real_fstat(fd)
        try:
            with patch.object(P.os, "open", new=tracked_open), patch.object(P.os, "fstat", new=failed):
                with self.assertRaises(OSError) as caught:
                    d.write("partial.json", {"x": 1})
            self.assertIs(caught.exception, primary)
            self.assertEqual(len(opened), 1)
            with self.assertRaises(OSError) as closed:
                real_fstat(opened[0])
            self.assertEqual(closed.exception.errno, 9)
            self.assertEqual((d.path/"partial.json").read_bytes(), b"")
            persist(self.directory/"fd_outcome.json", {"descriptor_closed_EBADF": True,
                "partial_original_bytes": 0, "primary_preserved": True})
        finally:
            d.release()

    def test_every_stream_chunk_includes_lf_under65536(self):
        for size in (65533, 65534, 65535, 131070):
            writes, hasher = [], hashlib.sha256()
            def record_write(fd, data):
                self.assertEqual(fd, -35)
                writes.append(len(data))
                hasher.update(data)
                return len(data)
            with patch.object(P.os, "write", new=record_write):
                binding = P.stream_json(-35, "x"*size, 8*P.MIB)
            self.assertTrue(all(0 < n <= 65536 for n in writes))
            expected = P.encoded("x"*size, 8*P.MIB)
            self.assertEqual(binding, {"size": len(expected), "sha256": P.sha(expected)})
            self.assertEqual(hasher.hexdigest(), P.sha(expected))
            persist(self.directory/("chunks_"+str(size)+".json"), {"lengths": writes, "binding": binding,
                "callback": "plain function retains lengths/hash only, not memoryview"})


class FailureLedgerRepair(OwnedCase):
    """Actual P serializer/writer with fake accounting; no F/provider execution."""

    def context(self, ledger=None, deadline=None):
        ctx = P.Context.__new__(P.Context)
        _, ctx.envelope = software_graph()
        ctx.stage = "compute"
        ctx.deadline = self.deadline if deadline is None else deadline
        ctx.capture_dir = P.OwnedDirectory(self.directory/"owned", os.getuid(), ctx.deadline, create=True)
        ctx.fixture_dir = ctx.operations_dir = None
        ctx.reservation_ledger = ledger
        ctx.exercise_state = None
        ctx.failure_snapshot = None
        ctx.failure_snapshot_taken = False
        ctx.failure_observation_ordinal = 0
        ctx.observations = []
        ctx.private = {"owner_uid": os.getuid()}
        return ctx

    def held(self, size=20):
        memory = MemorySeams.Memory()
        ledger = P.ReservationLedger(memory)
        token = object()
        with ledger.operation("labelled query callback", "query_event"):
            memory.hold(token, size)
        return memory, ledger, token

    def failure_bytes(self, ctx):
        data = (ctx.capture_dir.path/"FAILED.json").read_bytes()
        record = P.decode_failure(data, ctx.envelope)
        diagnostic = P.strict_json((ctx.capture_dir.path/"DIAGNOSTIC.json").read_bytes(), canonical=True)
        P.validate_diagnostic(diagnostic, ctx.envelope)
        self.assertEqual(diagnostic["failed_record"], {"size": len(data), "sha256": P.sha(data)})
        return record, diagnostic, data

    def test_actual20byte_serializer_survivor_original_failure_binding(self):
        memory, ledger, token = self.held()
        state = types.SimpleNamespace(memory=memory, query_raw={0: object()}, query_prepared={0: object()})
        ctx = self.context(ledger)
        primary = OSError("actual configured final-release failure")
        try:
            with patch.object(memory, "release", side_effect=primary) as fault:
                with self.assertRaises(OSError) as caught:
                    P.serialize_exercises({"label": "synthetic serializer seam"}, state, ledger)
                self.assertIs(caught.exception, primary)
                self.assertEqual(fault.call_count, 1)
                # Actual capture unwind retries cleanup; the original primary is preserved.
                with self.assertRaises(OSError) as unwound:
                    P.cleanup_exercise(state, ledger, primary)
                self.assertIs(unwound.exception, primary)
                self.assertEqual(fault.call_count, 2)
            self.assertEqual((memory.live, list(memory.retained.values())), (20, [20]))
            ctx.failure(primary)
            record, diagnostic, original = self.failure_bytes(ctx)
            snapshot = record["reservation_ledger"]
            self.assertEqual(snapshot["state"], "observed")
            self.assertEqual((snapshot["live"], snapshot["peak"], snapshot["retained_sum"],
                              snapshot["owned_count"], snapshot["retained_count"]), (20, 20, 20, 1, 1))
            self.assertFalse(snapshot["cleanup_complete"])
            self.assertEqual(snapshot["owned_reservations"], [{
                "reservation_id": "reservation:00000001", "operation_id": "operation:00000000",
                "registered_size": 20, "retained": True, "retained_size": 20,
                "disposition": "release_failed"}])
            self.assertEqual([e["stage"] for e in snapshot["events"]],
                             ["query_event", "final_cleanup", "final_cleanup"])
            self.assertTrue(diagnostic["secondary"])
            self.assertNotIn("exercises.json", ctx.capture_dir.names())
            persist(self.directory/"actual_survivor_before_test_only_release.json", snapshot)
            epoch = copy.deepcopy(snapshot["epoch"])
            ledger.release(token)
            self.assertEqual(record["reservation_ledger"]["owned_reservations"][0]["disposition"],
                             "release_failed")
            ctx.failure(primary)
            self.assertEqual((ctx.capture_dir.path/"FAILED.json").read_bytes(), original)
            self.assertEqual(ctx.failure_snapshot["epoch"], epoch)
            self.assertEqual(ctx.failure_observation_ordinal, 1)
        finally:
            ctx.capture_dir.release()

    def test_initialized_and_uninitialized_source_observations(self):
        primary = ValueError("actual local primary")
        self.assertIsNone(P.failure_ledger_snapshot(None, "decode", 0, primary))
        memory = MemorySeams.Memory()
        ledger = P.ReservationLedger(memory)
        empty = P.failure_ledger_snapshot(ledger, "decode", 0, primary)
        self.assertEqual(empty["state"], "observed")
        self.assertEqual(empty["events"], [])
        self.assertEqual(empty["owned_reservations"], [])
        self.assertFalse(empty["cleanup_complete"])
        ledger.cleanup()
        complete = P.failure_ledger_snapshot(ledger, "write_capture", 1, primary)
        self.assertTrue(complete["cleanup_complete"])
        unavailable = P.failure_ledger_snapshot(P.ReservationLedger.__new__(P.ReservationLedger),
                                                "decode", 2, primary)
        self.assertEqual(unavailable["state"], "unavailable")
        self.assertTrue(all(unavailable[k] is None for k in P._LEDGER_FACTS))
        persist(self.directory/"local_states.json", {"null": None, "empty": empty,
                                                   "complete": complete, "unavailable": unavailable})

    def test_identity_count_mismatches_and_no_fabricated_ordinals(self):
        primary = ValueError("actual mismatch primary")
        memory, ledger, token = self.held()
        memory.hold(object(), 7)
        unknown = P.failure_ledger_snapshot(ledger, "compute", 0, primary)
        self.assertEqual(unknown["state"], "observed")
        self.assertEqual(unknown["unregistered_retained_count"], 1)
        self.assertEqual(len(unknown["owned_reservations"]), 1)
        self.assertFalse(unknown["registry_matches_retained"])
        memory.retained[token] = 19
        memory.live = 26
        mismatch = P.failure_ledger_snapshot(ledger, "compute", 1, primary)
        self.assertFalse(mismatch["registered_sizes_match"])
        self.assertEqual(mismatch["owned_reservations"][0]["retained_size"], 19)
        memory.release(token)
        disappeared = P.failure_ledger_snapshot(ledger, "compute", 2, primary)
        self.assertEqual(disappeared["owned_not_retained_count"], 1)
        self.assertFalse(disappeared["owned_reservations"][0]["retained"])
        self.assertIsNone(disappeared["owned_reservations"][0]["retained_size"])
        ledger.token_events.clear()
        missing = P.failure_ledger_snapshot(ledger, "compute", 3, primary)
        self.assertEqual(missing["state"], "incomplete")
        self.assertIsNone(missing["owned_reservations"][0]["operation_id"])
        persist(self.directory/"identity_observations.json",
                {"unknown": unknown, "mismatch": mismatch, "disappeared": disappeared, "missing": missing})

    def test_source_slots_never_invoke_properties_repr_or_equality(self):
        calls = []
        class HostileMemory:
            @property
            def live(self):
                calls.append("property")
                raise AssertionError("property forbidden")
        memory, ledger, token = self.held()
        hostile = HostileMemory()
        namespace = P.local_base_namespace(hostile)
        namespace.update(P.local_base_namespace(memory))
        ledger.memory = hostile
        class Token:
            def __repr__(self):
                calls.append("repr")
                raise AssertionError("repr forbidden")
            def __eq__(self, other):
                calls.append("equality")
                raise AssertionError("equality forbidden")
            __hash__ = object.__hash__
        foreign = Token()
        namespace["retained"][foreign] = 5
        namespace["live"] = 25
        snapshot = P.failure_ledger_snapshot(ledger, "compute", 0, ValueError("actual primary"))
        self.assertEqual(snapshot["live"], 25)
        self.assertEqual(snapshot["unregistered_retained_count"], 1)
        self.assertEqual(calls, [])
        ledger.events[0]["stage"] = foreign
        incomplete = P.failure_ledger_snapshot(ledger, "compute", 1, ValueError("actual malformed primary"))
        self.assertEqual(incomplete["state"], "incomplete")
        self.assertIsNone(incomplete["events"])
        self.assertEqual(calls, [])
        persist(self.directory/"safe_source.json", {"observed": snapshot, "incomplete": incomplete})

    def test_preallocation_limits_preserve_counts_without_clipped_prefixes(self):
        memory, ledger, _ = self.held()
        ledger.events = ledger.events*65
        for _ in range(128):
            memory.hold(object(), 1)
        snapshot = P.failure_ledger_snapshot(ledger, "compute", 0, ValueError("actual bound primary"))
        self.assertEqual(snapshot["event_count"], 65)
        self.assertIsNone(snapshot["events"])
        self.assertEqual(snapshot["retained_count"], 129)
        self.assertIsNone(snapshot["retained_sum"])
        self.assertEqual(snapshot["state"], "incomplete")
        persist(self.directory/"bounded_partial.json", snapshot)

    def test_all_helper_after_hold_faults_and_actual_cleanup_epochs(self):
        results = []
        for stage in ("query_event", "raw_reference", "prepare_reference", "baseline_load"):
            memory = MemorySeams.Memory()
            ledger = P.ReservationLedger(memory)
            primary = ValueError("actual configured helper after hold "+stage)
            with self.assertRaises(ValueError) as caught:
                with ledger.operation("stdlib callback; no F helper invocation", stage):
                    memory.hold(object(), 20)
                    if stage == "query_event":
                        memory.hold(object(), 10)
                    raise primary
            self.assertIs(caught.exception, primary)
            snapshot = P.failure_ledger_snapshot(ledger, "decode" if stage == "baseline_load" else "prepare",
                                                 0, primary)
            self.assertEqual(snapshot["live"], 0)
            self.assertEqual(snapshot["events"][0]["before_live"], snapshot["events"][0]["after_live"])
            self.assertEqual(sum(r["size"] for r in snapshot["events"][0]["new_reservations"]),
                             30 if stage == "query_event" else 20)
            self.assertFalse(snapshot["cleanup_attempted"])
            ledger.cleanup()
            cleaned = P.failure_ledger_snapshot(ledger, "write_capture", 1, primary)
            self.assertTrue(cleaned["cleanup_complete"])
            results.append({"stage": stage, "before_unwind": snapshot, "after_unwind": cleaned,
                            "fault_raised": True, "same_primary": caught.exception is primary})
        persist(self.directory/"helper_faults.labelled.json", results)

    def test_failed_persistence_boundaries_preserve_primary_partial_bytes_and_fd(self):
        for boundary in ("write", "zero", "fstat", "fsync", "close", "readback", "directory_sync", "encoding"):
            folder = self.directory/boundary
            folder.mkdir(mode=0o700)
            previous = self.directory
            self.directory = folder
            ctx = self.context()
            self.directory = previous
            primary = ValueError("original body primary")
            fault = OSError("actual configured "+boundary)
            real_open, real_write, real_fstat = os.open, os.write, os.fstat
            real_fsync, real_close = os.fsync, os.close
            opened, fired, writes = [], [], []
            def acquire(path, flags, *args, **kwargs):
                fd = real_open(path, flags, *args, **kwargs)
                if path == "FAILED.json" and flags & os.O_WRONLY:
                    opened.append(fd)
                return fd
            def writing(fd, data):
                if fd in opened and not fired:
                    writes.append(len(data))
                    if boundary == "zero":
                        fired.append(boundary)
                        return 0
                    if boundary == "write":
                        real_write(fd, data[:17])
                        fired.append(boundary)
                        raise fault
                return real_write(fd, data)
            def observing(fd):
                if boundary == "fstat" and fd in opened and not fired:
                    fired.append(boundary)
                    raise fault
                return real_fstat(fd)
            def syncing(fd):
                match = fd in opened if boundary == "fsync" else fd == ctx.capture_dir.fd
                if boundary in ("fsync", "directory_sync") and match and not fired:
                    fired.append(boundary)
                    raise fault
                return real_fsync(fd)
            def closing(fd):
                real_close(fd)
                if boundary == "close" and fd in opened and not fired:
                    fired.append(boundary)
                    raise fault
            def reading(*args):
                fired.append(boundary)
                raise fault
            def encoding(*args, **kwargs):
                fired.append(boundary)
                raise fault
            try:
                with contextlib.ExitStack() as stack:
                    for name, function in (("open", acquire), ("write", writing), ("fstat", observing),
                                           ("fsync", syncing), ("close", closing)):
                        stack.enter_context(patch.object(P.os, name, new=function))
                    if boundary == "readback":
                        stack.enter_context(patch.object(ctx.capture_dir, "read_failure", new=reading))
                    if boundary == "encoding":
                        original_stream = P.stream_json
                        def stream(fd, value, cap):
                            return encoding() if cap is None else original_stream(fd, value, cap)
                        stack.enter_context(patch.object(P, "stream_json", new=stream))
                    ctx.failure(primary)
                self.assertTrue(fired, boundary)
                self.assertNotIn("FAILED.json", ctx.capture_dir.completed_files, boundary)
                diagnostic = P.strict_json((ctx.capture_dir.path/"DIAGNOSTIC.json").read_bytes(), canonical=True)
                P.validate_diagnostic(diagnostic)
                self.assertIsNone(diagnostic["failed_record"], boundary)
                self.assertEqual(diagnostic["unavailable"],
                                 ["failed_record_unverified", "reservation_ledger_unpersisted"])
                self.assertTrue(diagnostic["secondary"])
                self.assertEqual(diagnostic["primary"]["reason"]["fragments"][0]["text"], "original body primary")
                for fd in opened:
                    with self.assertRaises(OSError) as caught:
                        real_fstat(fd)
                    self.assertEqual(caught.exception.errno, 9)
                partial = (ctx.capture_dir.path/"FAILED.json").read_bytes()
                original = partial
                ctx.failure(primary)
                self.assertEqual((ctx.capture_dir.path/"FAILED.json").read_bytes(), original)
                persist(folder/"actual_fault_outcome.json", {
                    "fault_fired": fired, "opened_fd_numbers": opened, "owned_fds_closed_EBADF": True,
                    "partial_original_size": len(partial), "partial_original_sha256": P.sha(partial),
                    "requested_write_lengths": writes, "completed_FAILED": False, "same_primary": True})
            finally:
                ctx.capture_dir.release()

    def test_actual_expired_deadline_no_relocation_or_completed_binding(self):
        ctx = self.context(deadline=time.monotonic_ns()+100_000_000)
        primary = ValueError("original expired primary")
        time.sleep(.12)
        try:
            ctx.failure(primary)
            self.assertEqual(os.listdir(ctx.capture_dir.fd), [])
            self.assertFalse(ctx.capture_dir.completed_files)
            self.assertTrue(P.secondary_records(primary)[0])
            self.assertGreater(time.monotonic_ns(), ctx.deadline)
            persist(self.directory/"expired.json", {"actual_deadline": ctx.deadline,
                "observed_monotonic_ns": time.monotonic_ns(), "no_completed_files": True})
        finally:
            # Deadline vetoes sync, but actual owned descriptors are still closed.
            with self.assertRaises(P.ProbeError):
                ctx.capture_dir.release()
            self.assertIsNone(ctx.capture_dir.fd)

    def test_repeated_entry_and_late_boot_error_preserve_bytes_and_epoch(self):
        ctx = self.context(P.ReservationLedger(MemorySeams.Memory()))
        primary = ValueError("original late primary")
        saved = P._ACTIVE_CONTEXT
        try:
            ctx.failure(primary)
            _, _, original = self.failure_bytes(ctx)
            epoch = copy.deepcopy(ctx.failure_snapshot["epoch"])
            P._ACTIVE_CONTEXT = ctx
            with self.assertRaises(P.ProbeError):
                P.entry([], {})
            self.assertIs(P._ACTIVE_CONTEXT, ctx)
            late = OSError("actual post-close synthetic finalizer")
            late_fd = ctx.capture_dir.fd
            original_close = os.close
            fired = []
            def close_then_fail(fd):
                original_close(fd)
                if fd == late_fd:
                    fired.append(fd)
                    raise late
            with patch.object(P.os, "close", new=close_then_fail):
                with self.assertRaises(ValueError) as caught:
                    P.release_context(ctx, primary)
            self.assertIs(caught.exception, primary)
            self.assertEqual(fired, [late_fd])
            self.assertTrue(P.secondary_records(primary)[0])
            stderr_path = self.directory/"boot_error.helper_stderr.bin"
            saved_stderr = os.dup(2)
            try:
                with open(stderr_path, "xb", buffering=0) as stderr:
                    os.dup2(stderr.fileno(), 2)
                    try:
                        P.boot_error(primary, "a"*64)
                    finally:
                        os.dup2(saved_stderr, 2)
                    os.fsync(stderr.fileno())
            finally:
                os.close(saved_stderr)
            self.assertEqual((ctx.capture_dir.path/"FAILED.json").read_bytes(), original)
            self.assertEqual(ctx.failure_snapshot["epoch"], epoch)
            self.assertIn(b"actual post-close synthetic finalizer", stderr_path.read_bytes())
        finally:
            P._ACTIVE_CONTEXT = saved
            if ctx.capture_dir.fd is not None:
                ctx.capture_dir.release()

    def test_actual_legal_FAILED_above8MiB_stream_writer_readback_and_decoder(self):
        ctx = self.context(deadline=time.monotonic_ns()+300_000_000_000)
        primary = ValueError("legal large failure")
        note = "x"*4094
        primary.add_note(note)
        # All notes and lists independently satisfy effective P33 bounds.
        for _ in range(32):
            error = ValueError("genuine bounded secondary")
            for _ in range(64):
                error.add_note(note)
            P.add_secondary(primary, error, "large stdlib Error fixture")
        original_write, original_read, lengths, reads = os.write, os.read, [], []
        def writing(fd, data):
            lengths.append(len(data))
            return original_write(fd, data)
        def reading(fd, count):
            reads.append(count)
            return original_read(fd, count)
        try:
            with patch.object(P.os, "write", new=writing), patch.object(P.os, "read", new=reading):
                ctx.failure(primary)
            bound = ctx.capture_dir.completed_files["FAILED.json"]
            self.assertGreater(bound["size"], 8*P.MIB)
            self.assertLessEqual(max(lengths), 65536)
            self.assertLessEqual(max(reads), 65536)
            # Fresh postmortem U-bounded allocation, distinct from streaming writer/readback.
            actual = (ctx.capture_dir.path/"FAILED.json").read_bytes()
            decoded = P.decode_failure(actual)
            self.assertEqual(len(decoded["secondary"]), 32)
            self.assertIsNone(decoded["reservation_ledger"])
            self.assertNotIn("DIAGNOSTIC.json", ctx.capture_dir.completed_files)
            self.assertTrue(P.secondary_records(primary)[0][-1]["context"] == "DIAGNOSTIC persistence")
            persist(self.directory/"large_actual_writer_readback.json", {
                "binding": bound, "actual_size": len(actual), "actual_sha256": P.sha(actual),
                "maximum_actual_write_request": max(lengths), "maximum_actual_read_request": max(reads),
                "fresh_decoder_ceiling": P.FAILURE_INPUT_BYTES, "FAILED_writer_cap": None,
                "DIAGNOSTIC_cap": 8*P.MIB, "DIAGNOSTIC_partial_preserved": True,
                "label": "actual stdlib writer/readback; synthetic Errors; not137 counting witness"})
        finally:
            ctx.capture_dir.release()

    def test_common_failure_generation_null_and_exact_authority(self):
        ctx = self.context()
        try:
            ctx.failure(ValueError("schema-only verifier/controller no local ledger"))
            record, diagnostic, _ = self.failure_bytes(ctx)
            self.assertIsNone(record["reservation_ledger"])
            self.assertEqual(P.OWNER_CONTRACT["size"], 10794)
            self.assertEqual(P.OWNER_CONTRACT["sha256"],
                             "b49d475f79fd021d68fb6bd9e78e95d15da427f40bdb567314ac5f834283b7d7")
            self.assertEqual(P.FAILURE_INPUT_BYTES, 49424896)
            for modification in ("missing_ledger", "extra", "bad_stage", "bad_marker"):
                bad = copy.deepcopy(record)
                if modification == "missing_ledger":
                    del bad["reservation_ledger"]
                elif modification == "extra":
                    bad["new_key"] = None
                elif modification == "bad_stage":
                    bad["stage"] = "baseline_load"
                else:
                    bad["marker_state"] = "present"
                with self.assertRaises(P.ProbeError):
                    P.validate_failure(bad)
            bad = copy.deepcopy(diagnostic)
            bad["failed_record"]["sha256"] = "0"*64
            with self.assertRaises(P.ProbeError):
                P.validate_diagnostic(bad)
        finally:
            ctx.capture_dir.release()

    def test_snapshot_errors_formatter_failure_and_actual_scalar_unavailability(self):
        memory, ledger, _ = self.held()
        memory.live = -1
        memory.limit = True
        primary = ValueError("actual scalar primary")
        original = P.error_record
        def formatting(error, context):
            if context == "working scalar acquisition":
                raise ValueError("actual snapshot formatter fault")
            return original(error, context)
        with patch.object(P, "error_record", new=formatting):
            snapshot = P.failure_ledger_snapshot(ledger, "compute", 0, primary)
        self.assertEqual(snapshot["state"], "incomplete")
        self.assertIsNone(snapshot["live"])
        self.assertIsNone(snapshot["limit_bytes"])
        self.assertEqual(snapshot["retained_sum"], 20)
        self.assertTrue(all(e["formatting_failure_count"] == 1 for e in snapshot["snapshot_errors"]))
        self.assertIs(P._SECONDARIES[id(primary)][0], primary)
        persist(self.directory/"actual_snapshot_formatting_failure.json", snapshot)

    def test_invalid_epoch_is_not_repaired_or_backdated(self):
        ctx = self.context(P.ReservationLedger(MemorySeams.Memory()))
        primary = ValueError("actual clock primary")
        try:
            with patch.object(P.time, "time_ns", return_value=-1):
                with self.assertRaises(P.ProbeError) as caught:
                    ctx.failure(primary)
            P.add_secondary(primary, caught.exception, "labelled caller epoch failure seam")
            self.assertFalse(ctx.failure_snapshot_taken)
            self.assertEqual(ctx.capture_dir.names(), set())
            self.assertIs(P._SECONDARIES[id(primary)][0], primary)
            persist(self.directory/"unencodable_epoch.json", {
                "fault_activated": True, "genuine_epoch_unavailable": True, "no_persisted_fields": True,
                "same_primary_held": True, "label": "negative clock seam, not a genuine negative UTC clock"})
        finally:
            ctx.capture_dir.release()

    def test_snapshot_schema_relations_positives_and_coordinated_negatives(self):
        memory, ledger, token = self.held()
        memory.retained[token] = 19
        memory.live = 21
        positive = P.failure_ledger_snapshot(ledger, "compute", 0, ValueError("actual inconsistent source"))
        self.assertEqual(positive["state"], "observed")
        self.assertFalse(positive["live_matches_sum"])
        self.assertFalse(positive["registered_sizes_match"])
        P.validate_failure_ledger(positive, "compute")
        variants = {}
        for name in ("live_flag", "size_flag", "count", "operation", "retained_size", "cleanup", "null_fact",
                     "epoch_stage", "extra", "coordinated_sum"):
            bad = copy.deepcopy(positive)
            if name == "live_flag":
                bad["live_matches_sum"] = True
            elif name == "size_flag":
                bad["registered_sizes_match"] = True
            elif name == "count":
                bad["retained_count"] += 1
            elif name == "operation":
                bad["owned_reservations"][0]["operation_id"] = "operation:00000099"
            elif name == "retained_size":
                bad["owned_reservations"][0]["retained_size"] = None
            elif name == "cleanup":
                bad["cleanup_complete"] = True
            elif name == "null_fact":
                bad["owned_reservations"] = None
            elif name == "epoch_stage":
                bad["epoch"]["failure_stage"] = "decode"
            elif name == "coordinated_sum":
                bad["retained_sum"] += 2
                bad["live_matches_sum"] = bad["live"] == bad["retained_sum"]
            else:
                bad["new_field"] = None
            with self.assertRaises(P.ProbeError):
                P.validate_failure_ledger(bad, "compute")
            variants[name] = bad
        persist(self.directory/"snapshot_schema_controls.json", {"positive_inconsistent": positive,
                                                                 "negative_inputs": variants})

    def test_original_fd_readback_detects_actual_changed_bytes(self):
        ctx = self.context()
        primary = ValueError("original modified-file primary")
        try:
            ctx.failure(primary)
            binding = ctx.capture_dir.completed_files["FAILED.json"]
            path = ctx.capture_dir.path/"FAILED.json"
            with open(path, "r+b", buffering=0) as stream:
                stream.seek(binding["size"]-1)
                stream.write(b" ")
                os.fsync(stream.fileno())
            with self.assertRaises(P.ProbeError) as caught:
                ctx.capture_dir.read_failure(binding, ctx.capture_dir.failure_records["FAILED.json"])
            P.add_secondary(primary, caught.exception, "actual changed original-FD readback")
            persist(self.directory/"changed_original_outcome.json", {
                "original_binding": binding, "changed_bytes_sha256": P.sha(path.read_bytes()),
                "actual_readback_rejected": True, "same_primary": True})
        finally:
            ctx.capture_dir.release()

    def test_short_actual_writes_complete_and_parent_sync_failure_does_not(self):
        ctx = self.context()
        primary = ValueError("original short-write primary")
        original_write = os.write
        requests = []
        def short(fd, data):
            requests.append(len(data))
            return original_write(fd, data[:7])
        try:
            with patch.object(P.os, "write", new=short):
                ctx.failure(primary)
            self.failure_bytes(ctx)
            self.assertGreater(len(requests), 2)
            self.assertLessEqual(max(requests), 65536)
            persist(self.directory/"actual_short_write.json", {"short_write_activated": True,
                "requested_lengths": requests, "completed_original_readback": True})
        finally:
            ctx.capture_dir.release()
        next_folder = self.directory/"parent_failure"
        next_folder.mkdir(mode=0o700)
        previous = self.directory
        self.directory = next_folder
        ctx = self.context()
        self.directory = previous
        original_sync = os.fsync
        fired = []
        def sync(fd):
            if fd == ctx.capture_dir.parent_fd and not fired:
                fired.append(fd)
                raise OSError("actual parent fsync fault")
            return original_sync(fd)
        try:
            with patch.object(P.os, "fsync", new=sync):
                ctx.failure(primary)
            self.assertTrue(fired)
            self.assertNotIn("FAILED.json", ctx.capture_dir.completed_files)
            diagnostic = P.strict_json((ctx.capture_dir.path/"DIAGNOSTIC.json").read_bytes(), canonical=True)
            self.assertIsNone(diagnostic["failed_record"])
            persist(next_folder/"parent_sync.json", {"fault_fired": True, "completed_FAILED": False})
        finally:
            ctx.capture_dir.release()

    def test_partial_context_initialization_fields_and_owner_pin_source_gate(self):
        tree = ast.parse(SOURCE.read_bytes())
        cls = next(v for v in tree.body if isinstance(v, ast.ClassDef) and v.name == "Context")
        init = next(v for v in cls.body if isinstance(v, ast.FunctionDef) and v.name == "__init__")
        parse = next(v for v in ast.walk(init) if isinstance(v, ast.Call)
                     and isinstance(v.func, ast.Name) and v.func.id == "parse_entry_arguments")
        for field in ("capture_dir", "fixture_dir", "operations_dir", "reservation_ledger",
                      "exercise_state", "failure_snapshot_taken", "observations"):
            stores = [v for v in ast.walk(init) if isinstance(v, ast.Attribute)
                      and v.attr == field and isinstance(v.ctx, ast.Store)]
            self.assertTrue(stores and min(v.lineno for v in stores) < parse.lineno)
        validator = next(v for v in tree.body if isinstance(v, ast.FunctionDef) and v.name == "validate_private")
        pin = next(v for v in ast.walk(validator) if isinstance(v, ast.Compare)
                   and isinstance(v.left, ast.Subscript) and isinstance(v.left.slice, ast.Constant)
                   and v.left.slice.value == "owner_contract")
        self.assertEqual(ast.unparse(pin), "value['owner_contract'] == OWNER_CONTRACT")
        self.assertEqual(P.OWNER_CONTRACT,
                         json.loads((EVIDENCE.parent/"controls/filter_cpu_preparation_implementation_repair002_release.json").
                                    read_bytes())["authority"]["adopted_owner_contract"])

    def test_partial_status_cells_notes_and_tags_validate_without_fabrication(self):
        controls = []
        for number in (2, -1, 2**100, -(2**255)):
            primary = ValueError("unsuccessful solver status/warning")
            primary.add_note("genuine numerical warning")
            value = P.partial_observation({"solver_result_code": number,
                "nonfinite_cost_tags": ["nan", "positive_infinity"]}, primary, "synthetic")
            P.validate_partial_observation(value)
            self.assertEqual(value["solver_result_code"], number)
            self.assertEqual(value["solver_success_validation"], "rejected")
            controls.append(value)
        for phrase in ("diagnostic domain=256 bits", '"solver_result_code_export"'):
            primary = ValueError("OT distance: nonfinite scalar: nan")
            note = "Bounded returned OT interface evidence: "+json.dumps({"request_id": "cell",
                "result_code_type": "builtins.int", "warning_type": "builtins.str"})+ \
                "; details: "+json.dumps({"warning": phrase})
            primary.add_note(note)
            value = P.partial_observation({"solver_result_code": None}, primary, "synthetic")
            P.validate_partial_observation(value)
            self.assertEqual(value["export_state"], "capture_export_not_reached")
            controls.append(value)
        persist(self.directory/"partial_source_fidelity.json", controls)

    def test_readback_close_crossing_actual_deadline_vetoes_completion(self):
        ctx = self.context(deadline=time.monotonic_ns()+2_000_000_000)
        primary = ValueError("original late readback-close primary")
        original_open, original_close = os.open, os.close
        readers, fired = [], []
        def opening(path, flags, *args, **kwargs):
            fd = original_open(path, flags, *args, **kwargs)
            if path == "FAILED.json" and not flags & os.O_WRONLY:
                readers.append(fd)
            return fd
        def closing(fd):
            original_close(fd)
            if fd in readers and not fired:
                fired.append(fd)
                while time.monotonic_ns() <= ctx.deadline:
                    time.sleep(.01)
        try:
            with patch.object(P.os, "open", new=opening), patch.object(P.os, "close", new=closing):
                ctx.failure(primary)
            self.assertTrue(fired)
            self.assertNotIn("FAILED.json", ctx.capture_dir.completed_files)
            self.assertGreater((ctx.capture_dir.path/"FAILED.json").stat().st_size, 0)
            self.assertIs(P._SECONDARIES[id(primary)][0], primary)
            for fd in readers:
                with self.assertRaises(OSError) as caught:
                    os.fstat(fd)
                self.assertEqual(caught.exception.errno, 9)
            persist(self.directory/"actual_close_deadline.json", {
                "readback_close_fault_fired": True, "actual_deadline": ctx.deadline,
                "observed_monotonic_ns": time.monotonic_ns(), "completed_FAILED": False,
                "readback_descriptors_closed_EBADF": True})
        finally:
            with self.assertRaises(P.ProbeError):
                ctx.capture_dir.release()


class AccountingAndCacheRepair(OwnedCase):
    def test_success_boundary_baseline_peak_full_validator(self):
        value, envelope = exercise_records()
        P.validate_exercises(value, envelope)
        higher = copy.deepcopy(value)
        higher["working_memory"]["peak_accounted_working_bytes"] += 100
        P.validate_exercises(higher, envelope)
        variants = {}
        for name in ("query_delta", "query_count", "missing_baseline", "baseline_size",
                     "reference_drift", "early_query_release", "peak", "baseline_release_live"):
            bad = copy.deepcopy(value)
            if name == "query_delta":
                bad["reservations"][0]["after_live"] = 0
            elif name == "query_count":
                bad["reservations"][0]["after_retained"] = 0
            elif name == "missing_baseline":
                bad["reservations"][-1]["new_reservations"] = []
            elif name == "baseline_size":
                bad["reservations"][-1]["new_reservations"][0]["size"] += 1
            elif name == "reference_drift":
                bad["reservations"][3]["before_live"] += 1
            elif name == "early_query_release":
                bad["reservations"][0]["new_reservations"][0]["disposition"] = "released"
            elif name == "peak":
                bad["working_memory"]["peak_accounted_working_bytes"] = 0
            else:
                bad["baseline"]["working_live_after"] += 1
            with self.assertRaises(P.ProbeError):
                P.validate_exercises(bad, envelope)
            variants[name] = bad
        persist(self.directory/"accounting_full_controls.json",
                {"positive": value, "higher_peak_positive": higher, "negative_inputs": variants})

    def test_generated_cache_absent_present_genuine_record_joins(self):
        value, envelope = software_graph()
        catalog = P.catalog_from_metadata(metadata_catalog())
        def validate(v):
            recount_software(v)
            return P.validate_software(v, envelope, catalog, {"ordinary_namespaces": []}, {"modules": {}})
        source_path = "/synthetic/"+P.GENERATED+".py"
        module = {**shape(P.GENERATED, file=source_path, origin=source_path),
                  "phase": "bootstrap", "kind": "generated", "file_asset_id": None,
                  "origin_asset_id": None, "cached_asset_id": None, "cache_state": "not_declared"}
        value["modules"]["bootstrap"].append(module)
        value["modules"]["bootstrap"].sort(key=lambda v: v["key"])
        validate(value)
        absent = copy.deepcopy(value)
        generated_index = next(i for i, v in enumerate(value["modules"]["bootstrap"]) if v["key"] == P.GENERATED)
        absent["modules"]["bootstrap"][generated_index]["cached_path"] = "/synthetic/associated.pyc"
        absent["modules"]["bootstrap"][generated_index]["cache_state"] = "absent"
        validate(absent)
        present = copy.deepcopy(absent)
        target = present["modules"]["bootstrap"][generated_index]
        binding = {"path": "/synthetic/associated.pyc", "size": 4, "sha256": "a"*64, "st_dev": 1, "st_ino": 99}
        sid = P.software_id(binding["path"])
        target.update(cached_asset_id=sid, cache_state="present")
        present["files"].append({"asset_id": sid, "binding": binding, "final_binding": dict(binding),
                                 "first_phase": "bootstrap", "phases": ["bootstrap"], "required": True})
        present["aliases"].append({"lexical_path": binding["path"], "asset_id": sid, "phases": ["bootstrap"]})
        present["roles"].append({"phase": "bootstrap", "role": "cache:"+target["key"], "asset_id": sid,
                                 "reported_path": binding["path"], "occurrence": 0})
        present["historical_comparisons"].append({"asset_id": sid, "canonical_path": binding["path"],
            "classification": "new", "prior_matches": [], "required": True})
        validate(present)
        negatives = {}
        for name in ("missing_role", "duplicate_role", "wrong_asset", "wrong_phase", "bad_state"):
            bad = copy.deepcopy(present)
            if name == "missing_role":
                bad["roles"].pop()
            elif name == "duplicate_role":
                bad["roles"].append(dict(bad["roles"][-1]))
            elif name == "wrong_asset":
                bad["modules"]["bootstrap"][generated_index]["cached_asset_id"] = bad["files"][0]["asset_id"]
            elif name == "wrong_phase":
                bad["files"][-1]["phases"] = ["after"]
                bad["files"][-1]["first_phase"] = "after"
            else:
                bad["modules"]["bootstrap"][generated_index]["cache_state"] = "not_declared"
            with self.assertRaises(P.ProbeError):
                validate(bad)
            negatives[name] = bad
        persist(self.directory/"generated_cache_controls.json",
                {"undeclared": value, "absent": absent, "present": present, "negatives": negatives})

    def test_original135_authenticated_record_controls(self):
        controls = EVIDENCE.parent/"controls"
        def load(name):
            return json.loads((controls/name).read_bytes())["input"]
        exercise = load("case_exercise_seed.json")
        _, envelope = exercise_records()
        envelope = {k: exercise[k] for k in envelope}
        P.validate_exercises(exercise, envelope)
        for name in ("case_impossible_success_delta.json", "case_baseline_reservation_missing.json",
                     "case_peak_below_observed_live.json"):
            with self.assertRaises(P.ProbeError):
                P.validate_exercises(load(name), envelope)
        catalog = P.catalog_from_metadata(metadata_catalog())
        for name in ("case_software_seed.json", "generated_undeclared_cache_control.json",
                     "case_generated_declared_absent_cache.json"):
            value = load(name)
            envelope = {k: value[k] for k in envelope}
            P.validate_software(value, envelope, catalog, {"ordinary_namespaces": []}, {"modules": {}})
        persist(self.directory/"original135_replay.json", {
            "actual_full_validators": True, "native_executed": False,
            "positive_exercise": True, "original_three_accounting_negatives_rejected": True,
            "software_seed_undeclared_absent_positives": True})


class SourceInterfaces(OwnedCase):
    def test_no_automatic_entry_tail(self):
        tree = ast.parse(SOURCE.read_bytes())
        self.assertFalse(any(isinstance(n, ast.If) for n in tree.body))
        self.assertTrue(all(not isinstance(n, ast.Expr) or isinstance(n.value, ast.Constant) for n in tree.body))

    def test_frozen_prepared_and_consumption_source_interfaces(self):
        root = SOURCE.parent.parent/"prosper"/"evaluation"
        fsource = root/"r2_1_hoptimus_filter_retrieval_cli.py"
        tree = ast.parse(fsource.read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Prepared")
        self.assertIn("reservation", [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)])
        header = next(n.value for n in tree.body if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == "PREPARATION_HEADER" for t in n.targets))
        self.assertEqual(tuple(ast.literal_eval(header)), P.PREPARATION_KEYS)
        reader = ast.parse((root/"r2_1_hoptimus_mosaic_reader.py").read_bytes())
        consumption = next(n for n in reader.body if isinstance(n, ast.ClassDef) and n.name == "ConsumptionRecord")
        checks = next(n for n in consumption.body if isinstance(n, ast.AnnAssign) and n.target.id == "semantic_checks")
        self.assertEqual(ast.unparse(checks.annotation), "tuple[str, ...]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    EVIDENCE = Path(args.evidence)
    if not EVIDENCE.is_absolute():
        parser.error("absolute owned evidence path required")
    EVIDENCE.mkdir(mode=0o700)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    persist(EVIDENCE/"unit_result.json", {
        "label": "stdlib_records_and_owned_processes_only",
        "native_capture_executed": False, "native_fixture_executed": False,
        "actual_profile_observed": False, "tests_run": result.testsRun,
        "failure_count": len(result.failures), "error_count": len(result.errors),
        "provider_tripwire_installed": True, "success": result.wasSuccessful(),
    })
    raise SystemExit(0 if result.wasSuccessful() else 1)
