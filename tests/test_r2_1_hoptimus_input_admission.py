"""Synthetic-only checks; never import the eager prosper package."""

from __future__ import annotations

import base64
import binascii
import builtins
from contextlib import ExitStack
import copy
from functools import wraps
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import types
from unittest.mock import patch

import pytest


SOURCE_PATH = Path(__file__).resolve().parents[1] / "prosper/evaluation/r2_1_hoptimus_input_admission.py"
SOURCE = SOURCE_PATH.read_bytes()
SOURCE_SHA = hashlib.sha256(SOURCE).hexdigest()
MODULE = types.ModuleType("verified_input_observation")
sys.modules[MODULE.__name__] = MODULE
exec(compile(SOURCE, str(SOURCE_PATH), "exec"), MODULE.__dict__)


@pytest.fixture(autouse=True)
def primitive_io_tripwire(monkeypatch):
    """Guard calls only, after trusted source/stdlib acquisition."""
    def forbidden_io(*args, **kwargs):
        raise AssertionError("primitive attempted forbidden IO")

    def guarded(function):
        @wraps(function)
        def call(*args, **kwargs):
            with ExitStack() as stack:
                stack.enter_context(patch.object(builtins, "open", forbidden_io))
                stack.enter_context(patch.object(io, "open", forbidden_io))
                for name in ("open", "stat", "lstat", "listdir", "scandir", "readlink",
                             "mkdir", "makedirs", "unlink", "remove", "rename", "replace"):
                    stack.enter_context(patch.object(os, name, forbidden_io))
                return function(*args, **kwargs)
        return call

    for name in ("classic_vit224_schema", "observe_png", "observe_safetensors",
                 "compare_vit224_schema", "inventory_targets"):
        monkeypatch.setattr(MODULE, name, guarded(getattr(MODULE, name)))


def chunk(name=b"IDAT", data=b"not a deflate stream"):
    return struct.pack(">I", len(data)) + name + data + struct.pack(
        ">I", binascii.crc32(name + data) & 0xffffffff)


def ihdr(width=512, height=512, depth=8, colour=2, compression=0, filter_method=0, interlace=0):
    return chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, colour,
                                      compression, filter_method, interlace))


def png(*extras, header=None, ending=None):
    return b"\x89PNG\r\n\x1a\n" + (ihdr() if header is None else header) + b"".join(
        extras) + chunk() + (chunk(b"IEND", b"") if ending is None else ending)


def observe_png(data):
    return MODULE.observe_png(data, declared_size=len(data))


def safe(records=None, payload_size=None, metadata=None, raw=None):
    if records is None:
        records = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    if raw is None:
        value = copy.deepcopy(records)
        if metadata is not None:
            value["__metadata__"] = metadata
        raw = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()
    if payload_size is None:
        payload_size = max((v["data_offsets"][1] for v in records.values()), default=0)
    return MODULE.observe_safetensors(struct.pack("<Q", len(raw)), raw,
                                      total_size=8 + len(raw) + payload_size)


def schema_observation(dtype="F32", edits=None):
    widths = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "F8_E4M3": 1,
              "F8_E5M2": 1, "I32": 4}
    shapes = {t["name"]: t["shape"] for t in MODULE.classic_vit224_schema()["tensors"]}
    if edits:
        edits(shapes)
    records, offset = {}, 0
    for name, shape in sorted(shapes.items()):
        numel = 1
        for dimension in shape:
            numel *= dimension
        end = offset + numel * widths[dtype]
        records[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        offset = end
    return safe(records, offset)


def inventory_fixture():
    metadata = [
        {"metadata_index": i, "slide_id": str(i + 10000), "patient_id": f"p{i // 2}",
         "diagnostic_group": ("NORMAL" if i % 17 == 0 else f"g{i % 15}")
         if i < 2778 else "MISC"}
        for i in range(2921)
    ]
    targets = [
        dict(row, population_index=i, expected_tile_rows=i + 1)
        for i, row in enumerate(metadata[:2778])
    ]
    probes = {
        row["slide_id"]: {
            "lexical_path": f"/synthetic/wsis/{row['slide_id']}.svs",
            "lexical_type": "symlink", "resolved_path": f"/synthetic/raw/{row['slide_id']}.svs",
            "resolved_type": "regular", "size_bytes": 42, "probe_coherent": True,
            "status": "observed_regular", "error_operation": None,
            "error_errno": None, "error_message": None,
        } for row in targets
    }
    prior = {"10000": {"resolved_path": "/synthetic/raw/10000.svs", "size_bytes": 42,
                       "sha256": "a" * 64, "receipt_reference": "accepted/prior.json"}}
    return targets, metadata, probes, prior


def inventory(fixture):
    return MODULE.inventory_targets(*fixture, lexical_root="/synthetic/wsis")


def rejected(call, kind=None, code=None):
    with pytest.raises(MODULE.AdmissionError) as caught:
        call()
    error = caught.value
    assert error.kind in ("malformed", "unsupported", "limit", "declaration")
    assert len(str(error)) <= 1024
    assert error.offset is None or type(error.offset) is int
    if kind:
        assert error.kind == kind
    if code:
        assert error.code == code
    return error


def test_schema_exact_names_shapes_counts_and_freshness():
    schema = MODULE.classic_vit224_schema()
    assert set(schema) == {"schema_version", "schema_id", "constructor_args", "tensors",
                           "tensor_count", "scalar_count", "source_fingerprints", "limitations"}
    assert schema["tensor_count"] == 567
    assert schema["scalar_count"] == 1134774272
    assert sum(t["numel"] for t in schema["tensors"]) == 1134774272
    names = [t["name"] for t in schema["tensors"]]
    assert names == sorted(names) and len(set(names)) == 567
    expected = {"cls_token": [1, 1, 1536], "reg_token": [1, 4, 1536],
                "pos_embed": [1, 256, 1536], "patch_embed.proj.weight": [1536, 3, 14, 14],
                "patch_embed.proj.bias": [1536], "norm.weight": [1536], "norm.bias": [1536]}
    block = {"norm1.weight": [1536], "norm1.bias": [1536], "norm2.weight": [1536],
             "norm2.bias": [1536], "attn.qkv.weight": [4608, 1536], "attn.qkv.bias": [4608],
             "attn.proj.weight": [1536, 1536], "attn.proj.bias": [1536],
             "ls1.gamma": [1536], "ls2.gamma": [1536], "mlp.fc1.weight": [8192, 1536],
             "mlp.fc1.bias": [8192], "mlp.fc2.weight": [1536, 4096], "mlp.fc2.bias": [1536]}
    for i in range(40):
        expected.update({f"blocks.{i}.{name}": shape for name, shape in block.items()})
    assert {t["name"]: t["shape"] for t in schema["tensors"]} == expected
    assert schema["constructor_args"]["norm_layer"] == {
        "name": "LayerNorm", "eps": 1e-6, "elementwise_affine": True}
    schema["constructor_args"]["norm_layer"]["eps"] = 9
    schema["tensors"][0]["shape"][0] = 0
    schema["source_fingerprints"].clear()
    fresh = MODULE.classic_vit224_schema()
    assert fresh["constructor_args"]["norm_layer"]["eps"] == 1e-6
    assert fresh["tensors"][0]["shape"][0] != 0
    assert len(fresh["source_fingerprints"]) == 5


def test_png_all_observations_and_opaque_invalid_compressed_metadata():
    data = png(
        chunk(b"gAMA", struct.pack(">I", 45455)),
        chunk(b"cHRM", struct.pack(">8I", *range(8))),
        chunk(b"sRGB", b"\x02"), chunk(b"iCCP", b"name\0\0NOT ZLIB"),
        chunk(b"PLTE", b"\x00\x01\x02"), chunk(b"tRNS", struct.pack(">3H", 1, 2, 3)),
        chunk(b"eXIf", b"NOT TIFF"), chunk(b"acTL", struct.pack(">II", 0, 0)),
        chunk(b"fcTL", b"!" * 26), chunk(b"fdAT", b"!" * 4),
        chunk(b"aaAA", b"opaque"), chunk(b"AAAB", b"opaque"),
    )
    result = observe_png(data)
    assert result["png_framing_valid"] is True
    assert result["colour"]["gama_u32"] == 45455
    assert result["colour"]["chrm_u32_values"] == list(range(8))
    assert result["colour"]["iccp_descriptor"]["compressed_sha256"] == hashlib.sha256(b"NOT ZLIB").hexdigest()
    assert result["transparency"]["raw_samples_or_alpha_table"] == [1, 2, 3]
    assert result["exif"]["orientation"] is None
    assert result["exif"]["orientation_status"] == "uninterpreted"
    assert result["apng"]["actl_raw_fields"] == {"num_frames": 0, "num_plays": 0}
    assert result["unsupported_properties"] == ["apng_semantics_uninterpreted", "unknown_critical_chunk"]
    assert result["consequential_properties"] == [
        "gamma_present", "chromaticity_present", "srgb_present", "icc_present",
        "srgb_iccp_conflict", "palette_present", "transparency_present",
        "exif_uninterpreted", "apng_uninterpreted", "unknown_ancillary", "unknown_critical",
    ]
    assert result["unknown_chunks"] == [11, 12]
    result["colour"]["chrm_u32_values"][0] = 900
    result["apng"]["chunk_descriptors"][0]["type"] = "changed"
    assert observe_png(data)["colour"]["chrm_u32_values"][0] == 0
    assert result["chunks"][8]["type"] == "acTL"
    assert data == bytes(data)


@pytest.mark.parametrize("colour,depth", [
    (0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (2, 8), (2, 16),
    (3, 1), (3, 2), (3, 4), (3, 8), (4, 8), (4, 16), (6, 8), (6, 16),
])
def test_png_legal_depths(colour, depth):
    extras = [chunk(b"PLTE", b"\0\0\0")] if colour == 3 else []
    result = observe_png(png(*extras, header=ihdr(depth=depth, colour=colour, interlace=1)))
    assert result["ihdr"]["bit_depth"] == depth
    assert result["ihdr"]["interlace_method"] == 1


@pytest.mark.parametrize("data", [
    b"wrong", png()[1:], png()[:-1], png() + b"x",
    b"\x89PNG\r\n\x1a\n" + chunk(),
    png(ihdr()), png(header=ihdr(width=0)), png(header=ihdr(height=2**31)),
    png(header=ihdr(depth=4)), png(header=ihdr(colour=1)),
    png(header=ihdr(compression=1)), png(header=ihdr(filter_method=1)),
    png(header=ihdr(interlace=2)), png(chunk(b"abca", b"")),
    png(chunk(b"1ABC", b"")), png(ending=chunk(b"IEND", b"x")),
    b"\x89PNG\r\n\x1a\n" + ihdr() + chunk(b"IEND", b""),
    b"\x89PNG\r\n\x1a\n" + ihdr() + chunk() + chunk(b"aaAA", b"") + chunk() + chunk(b"IEND", b""),
    png(chunk(b"eXIf", b"x"), chunk(b"eXIf", b"x")),
    png(chunk(b"gAMA", b"\0" * 4)), png(chunk(b"gAMA", b"\1")),
    png(chunk(b"cHRM", b"\1")), png(chunk(b"sRGB", b"\4")),
    png(chunk(b"acTL", b"\1")), png(chunk(b"fcTL", b"\1")),
    png(chunk(b"fdAT", b"\1")),
])
def test_png_malformed(data):
    rejected(lambda: observe_png(data), "malformed")


def test_png_crc_checks_unknown_and_opaque_chunks():
    for name in (b"IDAT", b"eXIf", b"aaAA", b"AAAB"):
        bad = bytearray(chunk(name, b"data"))
        bad[-1] ^= 1
        rejected(lambda: observe_png(png(bytes(bad))), "malformed", "png_crc")


@pytest.mark.parametrize("keyword", [b"", b" " + b"a", b"a ", b"a  b", b"a" * 80, b"\x7f", b"\xa0"])
def test_png_iccp_keyword_rejection(keyword):
    rejected(lambda: observe_png(png(chunk(b"iCCP", keyword + b"\0\0x"))), "malformed")


@pytest.mark.parametrize("payload", [b"name", b"name\0", b"name\0\0", b"name\0\1x"])
def test_png_iccp_separator_method_remainder(payload):
    rejected(lambda: observe_png(png(chunk(b"iCCP", payload))), "malformed")


def test_png_palette_transparency_order_and_ranges():
    for colour in (0, 4):
        rejected(lambda: observe_png(png(chunk(b"PLTE", b"\0\0\0"), header=ihdr(colour=colour))))
    for length in (0, 1, 2, 769):
        rejected(lambda: observe_png(png(chunk(b"PLTE", b"x" * length))))
    rejected(lambda: observe_png(png(header=ihdr(colour=3))))
    rejected(lambda: observe_png(png(chunk(b"PLTE", b"x" * 9), header=ihdr(colour=3, depth=1))))
    for colour in (0, 2):
        good = struct.pack(">H", 255) * (1 if colour == 0 else 3)
        result = observe_png(png(chunk(b"tRNS", good), header=ihdr(colour=colour)))
        assert result["transparency"]["present"] is True
        bad = struct.pack(">H", 256) * (1 if colour == 0 else 3)
        rejected(lambda: observe_png(png(chunk(b"tRNS", bad), header=ihdr(colour=colour))))
        rejected(lambda: observe_png(png(chunk(b"tRNS", good), chunk(b"PLTE", b"\0\0\0"),
                                        header=ihdr(colour=colour))))
    for colour in (4, 6):
        rejected(lambda: observe_png(png(chunk(b"tRNS", b"xx"), header=ihdr(colour=colour))))
    good = png(chunk(b"PLTE", b"\0\0\0" * 2), chunk(b"tRNS", b"\0\xff"),
               header=ihdr(colour=3, depth=1))
    assert observe_png(good)["transparency"]["raw_samples_or_alpha_table"] == [0, 255]
    for payload in (b"", b"\0" * 3):
        rejected(lambda: observe_png(png(chunk(b"PLTE", b"\0\0\0" * 2), chunk(b"tRNS", payload),
                                        header=ihdr(colour=3, depth=1))))
    for name, payload in ((b"gAMA", struct.pack(">I", 1)), (b"cHRM", b"\0" * 32),
                          (b"sRGB", b"\0"), (b"iCCP", b"n\0\0x")):
        rejected(lambda: observe_png(png(chunk(b"PLTE", b"\0\0\0"), chunk(name, payload))))
    late = b"\x89PNG\r\n\x1a\n" + ihdr() + chunk() + chunk(b"acTL", b"\0"*8) + chunk(b"IEND", b"")
    rejected(lambda: observe_png(late))


def test_png_caps_and_argument_types():
    rejected(lambda: MODULE.observe_png(bytearray(png()), declared_size=len(png())), "declaration")
    rejected(lambda: MODULE.observe_png(png(), declared_size=True), "declaration")
    rejected(lambda: MODULE.observe_png(png(), declared_size=0), "declaration")
    rejected(lambda: MODULE.observe_png(b"x" * (1024**2 + 1), declared_size=1024**2 + 1), "limit")
    good = png(*(chunk(b"aaAA", b"") for _ in range(1021)))
    assert len(observe_png(good)["chunks"]) == 1024
    rejected(lambda: observe_png(png(*(chunk(b"aaAA", b"") for _ in range(1022)))), "limit")
    oversized = b"\x89PNG\r\n\x1a\n" + ihdr() + struct.pack(">I", 1024**2 + 1) + b"IDAT" + b"xxxx"
    rejected(lambda: observe_png(oversized), "limit")


@pytest.mark.parametrize("dtype,width", [
    ("BOOL", 1), ("U8", 1), ("I8", 1), ("F8_E4M3", 1), ("F8_E5M2", 1),
    ("I16", 2), ("F16", 2), ("BF16", 2), ("I32", 4), ("F32", 4), ("I64", 8), ("F64", 8),
])
def test_safetensors_widths_scalars_metadata_and_fresh_returns(dtype, width):
    record = {"x": {"dtype": dtype, "shape": [], "data_offsets": [0, width]}}
    original = copy.deepcopy(record)
    result = safe(record, metadata={"unit": "synthetic"})
    assert result["scalar_count"] == 1
    assert result["payload_bytes_by_dtype"] == {dtype: width}
    assert result["metadata"] == {"unit": "synthetic"}
    assert safe(record)["metadata"] is None
    result["tensors"][0]["shape"].append(99)
    result["metadata"]["unit"] = "changed"
    assert record == original
    assert safe(record)["tensors"][0]["shape"] == []


@pytest.mark.parametrize("raw", [
    b" {}", b"\xef\xbb\xbf{}", b'{"x":', b'{"x":1,"x":2}',
    b'{"x":{"dtype":"F32","dtype":"F16","shape":[1],"data_offsets":[0,4]}}',
    b'{"x":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}}',
    b'{"x":{"dtype":"F32","shape":[true],"data_offsets":[0,4]}}',
    b'{"x":{"dtype":"F32","shape":[NaN],"data_offsets":[0,4]}}',
    b'{"x":{"dtype":"F32","shape":[Infinity],"data_offsets":[0,4]}}',
    b'{"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}',
    b'{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}',
    b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[false,4]}}',
    b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[-1,3]}}',
    b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":0}}',
    b'{"__metadata__":null}', b'{"__metadata__":{"key":true}}',
    b'{"__metadata__":{"key":"\\ud800"}}',
    b'{"\\udfff":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
    b'{"x":{"dtype":"\\ud800","shape":[1],"data_offsets":[0,4]}}',
    b"{}\0", b"{}\xff", b"{}{}",
])
def test_safetensors_json_and_type_rejections(raw):
    rejected(lambda: safe({}, payload_size=4, raw=raw), "malformed")


def test_safetensors_utf8_error_offset_is_bytes():
    raw = '{"__metadata__":{"x":"é"},BROKEN}'.encode()
    error = rejected(lambda: safe({}, 0, raw=raw), "malformed", "json_syntax")
    assert error.offset == raw.index(b"BROKEN")
    assert raw[error.offset:][:6] == b"BROKEN"


def test_safetensors_caps_prefix_and_checked_products():
    rejected(lambda: MODULE.observe_safetensors(b"", b"{}", total_size=10), "malformed")
    rejected(lambda: MODULE.observe_safetensors(bytearray(b"x"*8), b"{}", total_size=10), "declaration")
    rejected(lambda: MODULE.observe_safetensors(struct.pack("<Q", 3), b"{}", total_size=10), "declaration")
    rejected(lambda: MODULE.observe_safetensors(struct.pack("<Q", 2**20 + 1), b"", total_size=2**30), "limit")
    rejected(lambda: MODULE.observe_safetensors(struct.pack("<Q", 2), b"{}", total_size=9), "malformed")
    rejected(lambda: safe({}, 0, raw=b'{"x":' + b"[" * 16 + b"0" + b"]"*16 + b"}"), "limit")
    for raw in (b'{"x":' + b"9"*10000 + b"}", b'{"x":9223372036854775808}'):
        rejected(lambda: safe({}, 0, raw=raw), "limit", "json_integer_token")
    for shape in ([1]*9, [2147483647]*3):
        rejected(lambda: safe({"x": {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}}, 0), "limit")
    rejected(lambda: safe({"x": {"dtype": "F64", "shape": [2147483647, 2147483647],
                                 "data_offsets": [0, 0]}}, 0), "limit")
    zero = safe({"x": {"dtype": "F32", "shape": [2147483647]*3+[0], "data_offsets": [0, 0]}}, 0)
    assert zero["scalar_count"] == 0 and zero["tensors"][0]["numel"] == 0
    rejected(lambda: safe({"x": {"dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 4]}}),
             "unsupported")
    rejected(lambda: safe({"x"*257: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}), "limit")
    rejected(lambda: safe({}, 0, metadata={str(i): "x" for i in range(129)}), "limit")
    rejected(lambda: safe({}, 0, metadata={"x": "x"*4097}), "limit")
    records = {f"x{i}": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]} for i in range(4097)}
    rejected(lambda: safe(records, 0), "limit")


@pytest.mark.parametrize("records,payload", [
    ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 3]}}, 3),
    ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]}}, 5),
    ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, 5),
    ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
      "y": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, 4),
    ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
      "z": {"dtype": "F32", "shape": [0], "data_offsets": [2, 2]}}, 4),
    ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}, 4),
])
def test_safetensors_packing_negative_cases(records, payload):
    rejected(lambda: safe(records, payload), "malformed")


def test_safetensors_empty_boundary_and_padding():
    records = {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
               "z0": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
               "z4": {"dtype": "F32", "shape": [0], "data_offsets": [4, 4]}}
    result = safe(records)
    assert result["scalar_count"] == 1
    assert safe({}, 0, raw=b"{} \t\n\r")["tensor_count"] == 0


@pytest.mark.parametrize("dtype", ["F32", "F64", "F16", "BF16", "F8_E4M3", "F8_E5M2"])
def test_schema_exact_floating_types(dtype):
    observation = schema_observation(dtype)
    before = copy.deepcopy(observation)
    result = MODULE.compare_vit224_schema(observation)
    assert result == {"match": True, "missing_names": [], "unexpected_names": [],
                      "shape_mismatches": [], "dtype_issues": []}
    assert observation == before


def test_schema_reports_shapes_names_integer_and_coherent_zero():
    def edits(shapes):
        shapes["pos_embed"] = [1, 261, 1536]
        shapes["reg_tokens"] = shapes.pop("reg_token")
        shapes["head.weight"] = [3, 1536]
        shapes["norm.bias"] = [0]
    observation = schema_observation("I32", edits)
    before = copy.deepcopy(observation)
    result = MODULE.compare_vit224_schema(observation)
    assert result["missing_names"] == ["reg_token"]
    assert result["unexpected_names"] == ["head.weight", "reg_tokens"]
    assert [x["name"] for x in result["shape_mismatches"]] == ["norm.bias", "pos_embed"]
    assert len(result["dtype_issues"]) == 568
    result["shape_mismatches"][0]["observed_shape"].append(1)
    assert observation == before
    assert not MODULE.compare_vit224_schema(observation)["match"]


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("header_structure_valid", 1), ("tensor_count", True),
    ("scalar_count", False), ("total_size", -1), ("header_length", 0),
    ("payload_offset", 0), ("payload_size", 0), ("header_sha256", "X"*64),
    ("metadata", {"x": object()}), ("unsupported_properties", ["fake"]),
    ("payload_bytes_by_dtype", {"F32": True}),
])
def test_compare_revalidates_complete_observation(field, value):
    observation = schema_observation()
    observation[field] = value
    before = copy.deepcopy(observation) if field != "metadata" else None
    rejected(lambda: MODULE.compare_vit224_schema(observation), "declaration")
    if before is not None:
        assert observation == before


@pytest.mark.parametrize("field,value", [
    ("numel", True), ("nbytes", 1), ("dtype", "UNKNOWN"),
    ("shape", [True]), ("data_offsets", [0, -1]), ("name", "\ud800"),
])
def test_compare_revalidates_tensor_fields(field, value):
    observation = schema_observation()
    observation["tensors"][0][field] = value
    rejected(lambda: MODULE.compare_vit224_schema(observation), "declaration")


def test_compare_rejects_reordered_duplicate_incomplete_records():
    for action in (
        lambda o: o["tensors"].reverse(),
        lambda o: o["tensors"].append(copy.deepcopy(o["tensors"][0])),
        lambda o: o.pop("header_sha256"),
        lambda o: o["tensors"][0].pop("nbytes"),
        lambda o: o["tensors"][0]["data_offsets"].__setitem__(0, 1),
    ):
        observation = schema_observation()
        action(observation)
        rejected(lambda: MODULE.compare_vit224_schema(observation), "declaration")


def test_inventory_all_statuses_priors_aliases_and_freshness():
    fixture = inventory_fixture()
    targets, metadata, probes, prior = fixture
    statuses = MODULE._STATUSES
    for i, status in enumerate(statuses):
        p = probes[targets[i]["slide_id"]]
        p["status"] = status
        if status == "missing_lexical":
            p.update(lexical_type="absent", resolved_path=None, resolved_type=None,
                     size_bytes=None, probe_coherent=None)
        elif status == "broken_symlink":
            p.update(resolved_path=None, resolved_type=None, size_bytes=None, probe_coherent=None)
        elif status == "not_regular":
            p["resolved_type"] = "directory"
        elif status != "observed_regular":
            p.update(error_operation="stat", error_errno=5, error_message="synthetic failure",
                     probe_coherent=False if status == "changed_during_probe" else None)
    probes["10000"]["lexical_type"] = "regular"
    prior["10007"] = dict(prior["10000"], resolved_path="/literal/alias.svs", size_bytes=900)
    before = copy.deepcopy(fixture)
    result = inventory(fixture)
    assert len(result["rows"]) == 2778
    assert all(result["status_counts"][s] > 0 for s in statuses)
    assert result["status_counts"]["observed_regular"] == 2771
    assert result["rows"][0]["prior_path_match"] is True
    assert result["rows"][7]["prior_path_match"] is False
    assert result["rows"][7]["prior_size_match"] is False
    assert result["rows"][1]["prior_path_match"] is None
    assert result["content_admission"] == "not_performed"
    assert any(r["diagnostic_group"] == "NORMAL" for r in result["rows"])
    result["rows"][0]["size_bytes"] = 999
    result["limitations"].clear()
    assert fixture == before
    assert inventory(fixture)["rows"][0]["size_bytes"] == 42


@pytest.mark.parametrize("section,field,value", [
    (0, "population_index", True), (0, "metadata_index", 1.0), (0, "expected_tile_rows", 0),
    (0, "patient_id", " wrong "), (0, "diagnostic_group", "MISC"), (0, "slide_id", "../escape"),
    (1, "metadata_index", False), (1, "slide_id", ".."), (1, "patient_id", "p\nx"),
])
def test_inventory_record_rejections(section, field, value):
    fixture = inventory_fixture()
    fixture[section][0][field] = value
    before = copy.deepcopy(fixture)
    rejected(lambda: inventory(fixture), "declaration")
    assert fixture == before


@pytest.mark.parametrize("field,value", [
    ("lexical_path", "/elsewhere/10000.svs"), ("lexical_type", "absent"),
    ("lexical_type", "unknown"), ("resolved_path", None), ("resolved_path", "relative"),
    ("resolved_type", "directory"), ("size_bytes", True), ("size_bytes", None),
    ("probe_coherent", None), ("probe_coherent", 1), ("probe_coherent", False),
    ("error_operation", "stat"), ("error_errno", 5), ("error_message", "failure"),
    ("status", "success"),
])
def test_inventory_regular_incoherence(field, value):
    fixture = inventory_fixture()
    fixture[2]["10000"][field] = value
    rejected(lambda: inventory(fixture), "declaration")


@pytest.mark.parametrize("root", ["relative", "//double", "/a//b", "/a/../b", "/a/.", "/a/", "/a\\b", "/"+"x"*4096])
def test_inventory_root_rejections(root):
    fixture = inventory_fixture()
    rejected(lambda: MODULE.inventory_targets(*fixture, lexical_root=root), "declaration")


def test_inventory_missing_extra_duplicate_reordered_prior_and_types():
    actions = [
        lambda f: f[2].pop("10000"),
        lambda f: f[2].__setitem__("extra", dict(f[2]["10000"])),
        lambda f: f[0].reverse(),
        lambda f: f[1][1].__setitem__("slide_id", f[1][0]["slide_id"]),
        lambda f: f[3].__setitem__("outside", dict(f[3]["10000"])),
        lambda f: f[3]["10000"].__setitem__("sha256", "A"*64),
        lambda f: f[3]["10000"].__setitem__("size_bytes", True),
        lambda f: f[3]["10000"].__setitem__("receipt_reference", ""),
        lambda f: f[2]["10000"].__setitem__("unexpected", 1),
    ]
    for action in actions:
        fixture = inventory_fixture()
        action(fixture)
        rejected(lambda: inventory(fixture), "declaration")
    for section in (0, 1):
        fixture = list(inventory_fixture())
        fixture[section] = fixture[section][:-1]
        rejected(lambda: inventory(fixture), "declaration")
    rejected(lambda: MODULE.inventory_targets(iter([]), [], {}, {}, lexical_root="/synthetic"), "declaration")


def test_inventory_unavailable_current_prior_values_remain_null():
    fixture = inventory_fixture()
    p = fixture[2]["10000"]
    p.update(status="inaccessible", lexical_type="unknown", resolved_path=None,
             resolved_type=None, size_bytes=None, probe_coherent=None,
             error_operation="lstat", error_errno=13, error_message="denied")
    row = inventory(fixture)["rows"][0]
    assert row["size_bytes"] is row["prior_path_match"] is row["prior_size_match"] is None
    assert row["prior_sha256"] == "a"*64
    assert row["status"] == "inaccessible"


def test_exact_source_isolated_stdlib_child_all_functions_failures_and_io_tripwire():
    script = r'''
import __future__,_io,base64,binascii,builtins,hashlib,io,json,os,pathlib,struct,sys,types,typing,unicodedata
request=json.loads(sys.stdin.buffer.read())
source=base64.b64decode(request["source"],validate=True)
assert hashlib.sha256(source).hexdigest()==request["sha256"]
forbidden={"numpy","pandas","torch","timm","openslide","PIL","safetensors","prosper"}
assert not forbidden.intersection(sys.modules)
real_import=builtins.__import__
def guard_import(name,*args,**kwargs):
    assert name.split(".")[0] not in forbidden,name
    return real_import(name,*args,**kwargs)
def no_io(*args,**kwargs):
    raise AssertionError("primitive attempted forbidden IO")
builtins.__import__=guard_import
builtins.open=no_io
io.open=no_io
io.open_code=no_io
_io.open_code=no_io
for name in ("open","stat","lstat","listdir","scandir","readlink","mkdir","makedirs","unlink","remove","rename","replace"):
    setattr(os,name,no_io)
for name in ("open","read_bytes","read_text","write_bytes","write_text","stat","lstat","resolve","iterdir","glob","rglob"):
    setattr(pathlib.Path,name,no_io)
module=types.ModuleType("isolated_verified_observation")
sys.modules[module.__name__]=module
exec(compile(source,"verified-owned-source","exec"),module.__dict__)
def failure(fn,kind):
    try: fn()
    except module.AdmissionError as e:
        assert e.kind==kind,(e.kind,e.code)
        assert len(str(e))<=1024
        return
    raise AssertionError("expected rejection")
def c(name,data):
    return struct.pack(">I",len(data))+name+data+struct.pack(">I",binascii.crc32(name+data)&0xffffffff)
p=b"\x89PNG\r\n\x1a\n"+c(b"IHDR",struct.pack(">IIBBBBB",512,512,8,2,0,0,0))+c(b"iCCP",b"n\0\0INVALID")+c(b"eXIf",b"INVALID")+c(b"IDAT",b"INVALID")+c(b"IEND",b"")
schema=module.classic_vit224_schema()
assert schema["tensor_count"]==567 and schema["scalar_count"]==1134774272
schema["constructor_args"]["norm_layer"]["eps"]=9
assert module.classic_vit224_schema()["constructor_args"]["norm_layer"]["eps"]==1e-6
po=module.observe_png(p,declared_size=len(p))
assert po["exif"]["orientation_status"]=="uninterpreted"
po["colour"]["iccp_descriptor"]["keyword"]="changed"
assert module.observe_png(p,declared_size=len(p))["colour"]["iccp_descriptor"]["keyword"]=="n"
failure(lambda: module.observe_png(b"bad",declared_size=3),"malformed")
records={};offset=0
for tensor in module.classic_vit224_schema()["tensors"]:
    end=offset+tensor["numel"]*4
    records[tensor["name"]]={"dtype":"F32","shape":tensor["shape"],"data_offsets":[offset,end]}
    offset=end
header=json.dumps(records,separators=(",",":")).encode()
ob=module.observe_safetensors(struct.pack("<Q",len(header)),header,total_size=8+len(header)+offset)
assert module.compare_vit224_schema(ob)["match"]
tiny=dict(ob,header_length=2,payload_offset=10,total_size=10+ob["payload_size"],
          header_sha256=hashlib.sha256(b"{}").hexdigest())
failure(lambda: module.compare_vit224_schema(tiny),"declaration")
reserved_header=b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
reserved=module.observe_safetensors(struct.pack("<Q",len(reserved_header)),reserved_header,
                                  total_size=12+len(reserved_header))
reserved["tensors"][0]["name"]="__metadata__"
failure(lambda: module.compare_vit224_schema(reserved),"declaration")
large_dimension=b'{"x":{"dtype":"U8","shape":[2147483648],"data_offsets":[0,2147483648]}}'
failure(lambda: module.observe_safetensors(struct.pack("<Q",len(large_dimension)),large_dimension,
                                          total_size=8+len(large_dimension)+2147483648),"limit")
ob["tensor_count"]=True
failure(lambda: module.compare_vit224_schema(ob),"declaration")
for digits in (0,640):
    sys.set_int_max_str_digits(digits)
    huge=b'{"x":'+b"9"*10000+b"}"
    failure(lambda: module.observe_safetensors(struct.pack("<Q",len(huge)),huge,total_size=8+len(huge)),"limit")
zero=json.dumps({"z":{"dtype":"F32","shape":[2147483647]*3+[0],"data_offsets":[0,0]}}).encode()
z=module.observe_safetensors(struct.pack("<Q",len(zero)),zero,total_size=8+len(zero))
assert z["scalar_count"]==0
failure(lambda: module.observe_safetensors(b"",b"",total_size=0),"malformed")
metadata=[{"metadata_index":i,"slide_id":str(i),"patient_id":"same_patient","diagnostic_group":"NORMAL" if i<2778 else "MISC"} for i in range(2921)]
targets=[dict(row,population_index=i,expected_tile_rows=1) for i,row in enumerate(metadata[:2778])]
probes={str(i):{"lexical_path":"/synthetic/"+str(i)+".svs","lexical_type":"regular","resolved_path":"/other/"+str(i)+".svs","resolved_type":"regular","size_bytes":1,"probe_coherent":True,"status":"observed_regular","error_operation":None,"error_errno":None,"error_message":None} for i in range(2778)}
inv=module.inventory_targets(targets,metadata,probes,{},lexical_root="/synthetic")
assert inv["inventory_complete"] and inv["content_admission"]=="not_performed"
inv["rows"][0]["size_bytes"]=99
assert probes["0"]["size_bytes"]==1
original_probe=probes["0"]
for lexical_type in ("absent","regular"):
    probes["0"]=dict(original_probe,status="not_regular",lexical_type=lexical_type,
                     resolved_type="directory")
    failure(lambda: module.inventory_targets(targets,metadata,probes,{},lexical_root="/synthetic"),"declaration")
probes["0"]=original_probe
probes.pop("0")
failure(lambda: module.inventory_targets(targets,metadata,probes,{},lexical_root="/synthetic"),"declaration")
assert not forbidden.intersection(sys.modules)
print(json.dumps({"source_sha256":request["sha256"],"isolation":True,"io_tripwire":True,"all_public_functions":True,"failure_paths":True}))
'''
    request = json.dumps({"source": base64.b64encode(SOURCE).decode(), "sha256": SOURCE_SHA})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script],
                            input=request, text=True, capture_output=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["source_sha256"] == SOURCE_SHA
    assert all(report[key] for key in ("isolation", "io_tripwire", "all_public_functions", "failure_paths"))


def test_png_exact_encoded_ceiling_and_absent_fields():
    base = png()
    filler = chunk(b"aaAA", b"x" * (1024**2 - len(base) - 12))
    data = png(filler)
    assert len(data) == 1024**2
    result = observe_png(data)
    assert result["palette"] == {"present": False, "entry_count": None}
    assert result["transparency"] == {"present": False, "kind": None,
                                      "raw_samples_or_alpha_table": None}
    assert result["colour"] == {"gama_u32": None, "chrm_u32_values": None,
                                "srgb_intent": None, "iccp_descriptor": None}
    assert result["exif"]["orientation_status"] == "absent"
    assert result["apng"] == {"present": False, "actl_raw_fields": None,
                              "chunk_descriptors": [], "semantic_status": "absent"}
    assert result["consequential_properties"] == ["unknown_ancillary"]
    assert result["unsupported_properties"] == []
    assert set(result) == {
        "schema_version", "declared_size", "supplied_size", "png_framing_valid", "ihdr",
        "chunks", "palette", "transparency", "colour", "exif", "apng", "unknown_chunks",
        "consequential_properties", "unsupported_properties",
    }
    assert all(set(c) == {"index", "offset", "type", "length", "crc_valid", "data_sha256",
                          "critical", "recognised", "interpretation"} for c in result["chunks"])


def test_png_iccp_keyword_boundaries_colour_conflict_order_and_duplicate_properties():
    for keyword in (b"a", b"a"*79, b"\xa1\xff"):
        result = observe_png(png(chunk(b"iCCP", keyword+b"\0\0x")))
        assert result["colour"]["iccp_descriptor"]["keyword"] == keyword.decode("latin-1")
    result = observe_png(png(chunk(b"iCCP", b"x\0\0y"), chunk(b"sRGB", b"\0"),
                             chunk(b"aaAA", b""), chunk(b"abAB", b""),
                             chunk(b"AAAB", b""), chunk(b"BAAB", b"")))
    assert result["consequential_properties"] == [
        "icc_present", "srgb_present", "srgb_iccp_conflict", "unknown_ancillary", "unknown_critical",
    ]
    assert result["unsupported_properties"] == ["unknown_critical_chunk"]
    assert result["unknown_chunks"] == [3, 4, 5, 6]


@pytest.mark.parametrize("name,data", [
    (b"PLTE", b"\0\0\0"), (b"tRNS", b"\0"*6), (b"gAMA", struct.pack(">I", 1)),
    (b"cHRM", b"\0"*32), (b"sRGB", b"\0"), (b"iCCP", b"x\0\0y"),
    (b"eXIf", b""), (b"acTL", b"\0"*8),
])
def test_png_singleton_duplication(name, data):
    rejected(lambda: observe_png(png(chunk(name, data), chunk(name, data))), "malformed")


def test_png_transparency_low_depth_16bit_and_pre_idat_enforcement():
    for depth in (1, 2, 4, 16):
        maximum = 2**depth - 1
        good = png(chunk(b"tRNS", struct.pack(">H", maximum)),
                   header=ihdr(colour=0, depth=depth))
        assert observe_png(good)["transparency"]["raw_samples_or_alpha_table"] == [maximum]
        if depth != 16:
            rejected(lambda: observe_png(png(chunk(b"tRNS", struct.pack(">H", maximum+1)),
                                            header=ihdr(colour=0, depth=depth))), "malformed")
    for name, data in ((b"PLTE", b"\0"*3), (b"tRNS", b"\0"*6),
                       (b"sRGB", b"\0"), (b"gAMA", struct.pack(">I", 1))):
        late = b"\x89PNG\r\n\x1a\n"+ihdr()+chunk()+chunk(name, data)+chunk(b"IEND", b"")
        rejected(lambda: observe_png(late), "malformed")
    truncated_length = b"\x89PNG\r\n\x1a\n"+ihdr()+struct.pack(">I", 50)+b"IDAT"+b"xxxx"
    rejected(lambda: observe_png(truncated_length), "malformed", "png_chunk_truncated")


def test_safetensors_exact_resource_boundaries_and_rank():
    header = b"{}" + b" " * (1024**2 - 2)
    result = MODULE.observe_safetensors(struct.pack("<Q", len(header)), header,
                                       total_size=len(header)+8)
    assert result["header_length"] == 1024**2
    name = "x"*256
    result = safe({name: {"dtype": "F32", "shape": [0]*8, "data_offsets": [0, 0]}},
                  0, metadata={f"k{i}": "x"*4096 for i in range(128)})
    assert result["tensor_count"] == 1
    assert len(result["metadata"]) == 128
    records = {f"x{i}": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}
               for i in range(4096)}
    assert safe(records, 0)["tensor_count"] == 4096
    nested = b'{"x":'+b"["*15+b"0"+b"]"*15+b"}"
    rejected(lambda: safe({}, 0, raw=nested), "malformed", "record_type")
    for value in (True, 1.0, -1, 2**63):
        rejected(lambda: MODULE.observe_safetensors(struct.pack("<Q", 2), b"{}",
                                                   total_size=value), "declaration")
    rejected(lambda: safe({}, 0, metadata={"k"*257: "x"}), "limit")
    rejected(lambda: safe({}, 0, metadata={"x": "é"*2049}), "limit")
    rejected(lambda: safe({}, 0, raw=b'{"__metadata__":{"x":1e999}}'), "malformed")
    assert safe({}, 0, metadata={})["metadata"] == {}
    assert set(safe({}, 0)) == {
        "schema_version", "total_size", "header_length", "header_sha256", "payload_offset",
        "payload_size", "header_structure_valid", "metadata", "tensors", "tensor_count",
        "scalar_count", "payload_bytes_by_dtype", "unsupported_properties",
    }


def test_zero_shape_interior_extent_still_fails_and_comparison_totals_revalidated():
    record = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
              "z": {"dtype": "F32", "shape": [2147483647]*3+[0], "data_offsets": [2, 2]}}
    rejected(lambda: safe(record, 4), "malformed", "empty_extent_boundary")
    observation = safe({"pos_embed": {"dtype": "F32", "shape": [2147483647]*3+[0],
                                      "data_offsets": [0, 0]}}, 0)
    assert MODULE.compare_vit224_schema(observation)["shape_mismatches"] == [{
        "name": "pos_embed", "expected_shape": [1, 256, 1536],
        "observed_shape": [2147483647]*3+[0],
    }]
    for change in (
        lambda o: o["payload_bytes_by_dtype"].__setitem__("F32", 1),
        lambda o: o.__setitem__("scalar_count", 1),
        lambda o: o["tensors"][0].__setitem__("numel", 1),
        lambda o: o["tensors"][0].__setitem__("data_offsets", [1, 1]),
    ):
        invalid = copy.deepcopy(observation)
        change(invalid)
        rejected(lambda: MODULE.compare_vit224_schema(invalid), "declaration")
    copied = copy.deepcopy(observation)
    copied["header_sha256"] = "0"*64
    assert MODULE.compare_vit224_schema(copied)["match"] is False


@pytest.mark.parametrize("status,changes", [
    ("missing_lexical", {"lexical_type": "symlink", "resolved_path": None,
                        "resolved_type": None, "size_bytes": None}),
    ("broken_symlink", {"lexical_type": "regular", "resolved_path": None,
                       "resolved_type": None, "size_bytes": None}),
    ("missing_lexical", {"lexical_type": "absent"}),
    ("not_regular", {"resolved_type": "regular"}),
    ("changed_during_probe", {"probe_coherent": True, "error_operation": "stat",
                             "error_errno": 5, "error_message": "changed"}),
    ("io_error", {}),
    ("inaccessible", {"error_operation": "stat", "error_errno": 13, "error_message": None}),
    ("resolution_error", {"error_operation": None, "error_errno": 40, "error_message": "loop"}),
])
def test_inventory_inconsistent_unavailable_probes(status, changes):
    fixture = inventory_fixture()
    fixture[2]["10000"].update(status=status, **changes)
    rejected(lambda: inventory(fixture), "declaration")


def test_inventory_builtin_only_and_no_arbitrary_hooks():
    class BadDict(dict):
        pass
    class BadList(list):
        pass
    class Hook:
        def __deepcopy__(self, memo):
            raise AssertionError("unexpected metadata hook")
        def __str__(self):
            raise AssertionError("unexpected conversion")
    for section in (0, 1):
        fixture = list(inventory_fixture())
        fixture[section] = BadList(fixture[section])
        rejected(lambda: inventory(fixture), "declaration")
    fixture = inventory_fixture()
    fixture[0][0] = BadDict(fixture[0][0])
    rejected(lambda: inventory(fixture), "declaration")
    fixture = inventory_fixture()
    fixture[0][0]["patient_id"] = Hook()
    rejected(lambda: inventory(fixture), "declaration")
    fixture = list(inventory_fixture())
    fixture[2] = BadDict(fixture[2])
    rejected(lambda: inventory(fixture), "declaration")
    rejected(lambda: MODULE.observe_png(memoryview(png()), declared_size=len(png())), "declaration")
    rejected(lambda: MODULE.compare_vit224_schema(BadDict(safe())), "declaration")


def test_inventory_path_bounds_errors_and_return_fields():
    fixture = inventory_fixture()
    row = inventory(fixture)["rows"][0]
    assert set(row) == MODULE._TARGET_FIELDS | MODULE._PROBE_FIELDS | {
        "prior_stage1b", "prior_resolved_path", "prior_size_bytes", "prior_sha256",
        "prior_receipt_reference", "prior_path_match", "prior_size_match", "content_admission",
    }
    assert set(inventory(fixture)["status_counts"]) == set(MODULE._STATUSES)
    fixture[2]["10000"]["status"] = "io_error"
    fixture[2]["10000"].update(error_operation="stat", error_errno=5,
                             error_message="x"*1024, probe_coherent=None)
    assert inventory(fixture)["rows"][0]["error_message"] == "x"*1024
    fixture[2]["10000"]["error_message"] += "x"
    rejected(lambda: inventory(fixture), "declaration")
    for field, value in (("resolved_path", "//bad"), ("resolved_type", "invalid"),
                         ("error_errno", True), ("probe_coherent", 0)):
        fixture = inventory_fixture()
        fixture[2]["10000"][field] = value
        rejected(lambda: inventory(fixture), "declaration")
    for sid in (".", "..", "a/b", "a\\b", "a\0b"):
        fixture = inventory_fixture()
        fixture[1][0]["slide_id"] = sid
        rejected(lambda: inventory(fixture), "declaration")


def repair_independent_tensor_records():
    """Build the accepted state table independently of the candidate schema."""
    d = 1536
    shapes = {
        "cls_token": [1, 1, d], "reg_token": [1, 4, d],
        "pos_embed": [1, 16*16, d], "patch_embed.proj.weight": [d, 3, 14, 14],
        "patch_embed.proj.bias": [d], "norm.weight": [d], "norm.bias": [d],
    }
    for i in range(40):
        for stage in (1, 2):
            for parameter in ("weight", "bias"):
                shapes[f"blocks.{i}.norm{stage}.{parameter}"] = [d]
            shapes[f"blocks.{i}.ls{stage}.gamma"] = [d]
        for layer, output, input_width in (
            ("attn.qkv", 3*d, d), ("attn.proj", d, d),
            ("mlp.fc1", 8192, d), ("mlp.fc2", d, 4096),
        ):
            shapes[f"blocks.{i}.{layer}.weight"] = [output, input_width]
            shapes[f"blocks.{i}.{layer}.bias"] = [output]
    records, cursor = {}, 0
    for name, shape in sorted(shapes.items()):
        end = cursor + math.prod(shape)*4
        records[name] = {"dtype": "F32", "shape": shape, "data_offsets": [cursor, end]}
        cursor = end
    assert len(records) == 567 and cursor == 1134774272*4
    return records, cursor


def test_repair_obs1_impossible_two_byte_567_tensor_header():
    records, payload = repair_independent_tensor_records()
    observation = safe(records, payload)
    assert MODULE.compare_vit224_schema(observation)["match"]
    observation.update(header_length=2, payload_offset=10, total_size=10+payload,
                       header_sha256=hashlib.sha256(b"{}").hexdigest())
    before = copy.deepcopy(observation)
    rejected(lambda: MODULE.compare_vit224_schema(observation), "declaration",
             "observation_header_minimum")
    assert observation == before


def test_repair_obs1_reserved_metadata_tensor_name():
    observation = safe({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
    observation["tensors"][0]["name"] = "__metadata__"
    before = copy.deepcopy(observation)
    rejected(lambda: MODULE.compare_vit224_schema(observation), "declaration",
             "observation_reserved_name")
    assert observation == before


def test_repair_obs1_alternate_order_whitespace_escapes_unicode_and_metadata():
    records, payload = repair_independent_tensor_records()
    metadata = {"note": 'é"/\n', "版本": "中文"}
    reversed_root = {
        name: {"data_offsets": record["data_offsets"], "shape": record["shape"],
               "dtype": record["dtype"]}
        for name, record in reversed(list(records.items()))
    }
    reversed_root["__metadata__"] = metadata
    compact_root = dict(records, __metadata__=metadata)
    compact = json.dumps(compact_root, ensure_ascii=False, separators=(",", ":")).encode()
    variants = [
        compact,
        json.dumps(reversed_root, ensure_ascii=False, indent=2).encode()+b" \t\n\r",
        json.dumps(reversed_root, ensure_ascii=True, separators=(",", ":")).encode(),
        compact.replace(b'"cls_token"', b'"\\u0063ls_token"'),
    ]
    assert len({hashlib.sha256(raw).hexdigest() for raw in variants}) == 4
    for raw in variants:
        observation = safe({}, payload, raw=raw)
        before = copy.deepcopy(observation)
        assert MODULE.compare_vit224_schema(observation)["match"] is True
        assert observation == before
    for raw in (b"{}", b"{} \t\n", b'{"__metadata__":{}}',
                '{"__metadata__":{"字":"é"}}'.encode()):
        observation = safe({}, 0, raw=raw)
        observation["header_sha256"] = "0"*64
        result = MODULE.compare_vit224_schema(observation)
        assert result["match"] is False and len(result["missing_names"]) == 567
    ordinary = safe({"x": {"dtype": "U8", "shape": [], "data_offsets": [0, 1]}}, 1)
    result = MODULE.compare_vit224_schema(ordinary)
    assert result["unexpected_names"] == ["x"]
    assert result["dtype_issues"] == [{"name": "x", "dtype": "U8"}]


@pytest.mark.parametrize("lexical_type,resolved_type", [
    ("absent", "directory"), ("regular", "directory"),
    ("directory", "other"), ("other", "directory"),
])
def test_repair_obs2_coherent_contradictory_known_leaf_types(lexical_type, resolved_type):
    fixture = inventory_fixture()
    fixture[2]["10000"].update(status="not_regular", lexical_type=lexical_type,
                             resolved_type=resolved_type, probe_coherent=True)
    before = copy.deepcopy(fixture)
    rejected(lambda: inventory(fixture), "declaration", "probe_coherent_leaf_type")
    assert fixture == before


@pytest.mark.parametrize("lexical_type,resolved_type,status", [
    ("directory", "directory", "not_regular"),
    ("symlink", "directory", "not_regular"),
    ("other", "other", "not_regular"),
    ("regular", "regular", "observed_regular"),
])
def test_repair_obs2_consistent_leaves_and_ancestor_aliases(lexical_type, resolved_type, status):
    fixture = inventory_fixture()
    fixture[2]["10000"].update(status=status, lexical_type=lexical_type,
                             resolved_type=resolved_type, probe_coherent=True,
                             resolved_path="/ancestor/alias/10000.svs")
    result = inventory(fixture)
    assert len(result["rows"]) == 2778
    assert result["rows"][0]["resolved_path"] != result["rows"][0]["lexical_path"]
    assert result["rows"][0]["status"] == status
    assert result["content_admission"] == "not_performed"


@pytest.mark.parametrize("status", ["inaccessible", "changed_during_probe", "io_error"])
def test_repair_obs2_explicit_unavailable_changed_observations_preserved(status):
    fixture = inventory_fixture()
    probe = fixture[2]["10000"]
    probe.update(status=status, lexical_type="regular", resolved_type="directory",
                 probe_coherent=False if status == "changed_during_probe" else None,
                 error_operation="stat", error_errno=5, error_message="preserved observation")
    row = inventory(fixture)["rows"][0]
    assert all(row[key] == value for key, value in probe.items())
    assert row["content_admission"] == "not_performed"


def test_repair_obs3_dimension_ceiling_limit_with_consistent_extent():
    record = {"x": {"dtype": "U8", "shape": [2147483648],
                    "data_offsets": [0, 2147483648]}}
    rejected(lambda: safe(record, 2147483648), "limit", "shape_dimension_limit")
    permitted = {"x": {"dtype": "U8", "shape": [2147483647],
                       "data_offsets": [0, 2147483647]}}
    assert safe(permitted, 2147483647)["scalar_count"] == 2147483647


def test_repair_obs3_permitted_dimension_malformed_extent_and_declaration():
    inconsistent = {"x": {"dtype": "U8", "shape": [2147483647], "data_offsets": [0, 4]}}
    rejected(lambda: safe(inconsistent, 4), "malformed", "tensor_byte_length")
    for dimension in (-1, True, 1.0):
        record = {"x": {"dtype": "U8", "shape": [dimension], "data_offsets": [0, 1]}}
        rejected(lambda: safe(record, 1), "malformed")
    observation = safe({"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}, 1)
    observation["tensors"][0].update(shape=[2147483648], data_offsets=[0, 2147483648],
                                    numel=2147483648, nbytes=2147483648)
    observation.update(payload_size=2147483648, scalar_count=2147483648,
                       total_size=observation["payload_offset"]+2147483648,
                       payload_bytes_by_dtype={"U8": 2147483648})
    rejected(lambda: MODULE.compare_vit224_schema(observation), "declaration")
