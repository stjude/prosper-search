"""Pure encoded-input observations, not file custody or input/model admission."""

from __future__ import annotations

import binascii
import hashlib
import json
import struct
import unicodedata
from typing import TypedDict


_ENCODED_LIMIT = 1024 * 1024
_MAX_INT = 2**63 - 1
_MAX_DIM = 2**31 - 1
_WIDTHS = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4,
    "I64": 8, "F64": 8,
}
_FLOATS = frozenset(("F8_E4M3", "F8_E5M2", "F16", "BF16", "F32", "F64"))
_STATUSES = (
    "observed_regular", "missing_lexical", "broken_symlink", "inaccessible",
    "not_regular", "resolution_error", "changed_during_probe", "io_error",
)
_PATH_TYPES = frozenset(("regular", "symlink", "directory", "other", "absent", "unknown"))
_TARGET_FIELDS = frozenset((
    "population_index", "metadata_index", "slide_id", "patient_id",
    "diagnostic_group", "expected_tile_rows",
))
_METADATA_FIELDS = frozenset((
    "metadata_index", "slide_id", "patient_id", "diagnostic_group",
))
_PROBE_FIELDS = frozenset((
    "lexical_path", "lexical_type", "resolved_path", "resolved_type", "size_bytes",
    "probe_coherent", "status", "error_operation", "error_errno", "error_message",
))
_PRIOR_FIELDS = frozenset(("resolved_path", "size_bytes", "sha256", "receipt_reference"))
_SAFE_FIELDS = frozenset((
    "schema_version", "total_size", "header_length", "header_sha256",
    "payload_offset", "payload_size", "header_structure_valid", "metadata",
    "tensors", "tensor_count", "scalar_count", "payload_bytes_by_dtype",
    "unsupported_properties",
))


class AdmissionError(ValueError):
    """A bounded format, support, limit or caller-declaration rejection."""

    def __init__(self, kind: str, code: str, context: str, offset: int | None = None):
        self.kind = kind
        self.code = code
        self.offset = offset
        self.context = context[:900]
        super().__init__(f"{kind}/{code}: {self.context}"[:1024])


class TensorShape(TypedDict):
    name: str
    shape: list[int]
    numel: int


class StateSchema(TypedDict):
    schema_version: int
    schema_id: str
    constructor_args: dict[str, object]
    tensors: list[TensorShape]
    tensor_count: int
    scalar_count: int
    source_fingerprints: dict[str, str]
    limitations: list[str]


class TensorObservation(TensorShape):
    dtype: str
    data_offsets: list[int]
    nbytes: int


class SafetensorsObservation(TypedDict):
    schema_version: int
    total_size: int
    header_length: int
    header_sha256: str
    payload_offset: int
    payload_size: int
    header_structure_valid: bool
    metadata: dict[str, str] | None
    tensors: list[TensorObservation]
    tensor_count: int
    scalar_count: int
    payload_bytes_by_dtype: dict[str, int]
    unsupported_properties: list[str]


class SchemaComparison(TypedDict):
    match: bool
    missing_names: list[str]
    unexpected_names: list[str]
    shape_mismatches: list[dict[str, object]]
    dtype_issues: list[dict[str, str]]


class PNGObservation(TypedDict):
    schema_version: int
    declared_size: int
    supplied_size: int
    png_framing_valid: bool
    ihdr: dict[str, int]
    chunks: list[dict[str, object]]
    palette: dict[str, object]
    transparency: dict[str, object]
    colour: dict[str, object]
    exif: dict[str, object]
    apng: dict[str, object]
    unknown_chunks: list[int]
    consequential_properties: list[str]
    unsupported_properties: list[str]


class InventoryObservation(TypedDict):
    schema_version: int
    inventory_complete: bool
    rows: list[dict[str, object]]
    status_counts: dict[str, int]
    content_admission: str
    limitations: list[str]


class MetadataRecord(TypedDict):
    metadata_index: int
    slide_id: str
    patient_id: str
    diagnostic_group: str


class TargetRecord(MetadataRecord):
    population_index: int
    expected_tile_rows: int


class PathProbe(TypedDict):
    lexical_path: str
    lexical_type: str
    resolved_path: str | None
    resolved_type: str | None
    size_bytes: int | None
    probe_coherent: bool | None
    status: str
    error_operation: str | None
    error_errno: int | None
    error_message: str | None


class PriorBinding(TypedDict):
    resolved_path: str
    size_bytes: int
    sha256: str
    receipt_reference: str


def _need(condition: bool, kind: str, code: str, context: str,
          offset: int | None = None) -> None:
    if not condition:
        raise AdmissionError(kind, code, context, offset)


def _record(value: object, fields: frozenset[str], context: str,
            kind: str = "declaration") -> dict[str, object]:
    _need(type(value) is dict, kind, "record_type", context)
    _need(all(type(key) is str for key in value) and set(value) == fields,
          kind, "record_fields", context)
    return value


def _sequence(value: object, context: str,
              kind: str = "declaration") -> list[object] | tuple[object, ...]:
    _need(type(value) in (list, tuple), kind, "sequence_type", context)
    return value


def _integer(value: object, context: str, kind: str = "declaration",
             minimum: int = 0, maximum: int = _MAX_INT) -> int:
    _need(type(value) is int, kind, "integer_type", context)
    _need(minimum <= value <= maximum, kind, "integer_range", context)
    return value


def _utf8(value: object, context: str, limit: int,
          kind: str = "declaration") -> str:
    _need(type(value) is str, kind, "string_type", context)
    _need(len(value) <= limit, "limit" if kind != "declaration" else kind,
          "string_limit", context)
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AdmissionError(kind, "unicode_surrogate", context) from exc
    _need(len(encoded) <= limit, "limit" if kind != "declaration" else kind,
          "string_limit", context)
    return value


def _identity(value: object, context: str, maximum: int = 4096) -> str:
    text = _utf8(value, context, maximum * 4)
    _need(0 < len(text) <= maximum and text == text.strip(),
          "declaration", "identity_text", context)
    _need(not any(unicodedata.category(c) in ("Cc", "Cs") for c in text),
          "declaration", "identity_control", context)
    return text


def _path(value: object, context: str) -> str:
    path = _identity(value, context)
    _need(path.startswith("/") and "\\" not in path,
          "declaration", "absolute_posix_path", context)
    _need(path == "/" or all(p not in ("", ".", "..") for p in path.split("/")[1:]),
          "declaration", "canonical_path", context)
    return path


def _sha(value: object, context: str) -> str:
    _need(type(value) is str and len(value) == 64
          and all(c in "0123456789abcdef" for c in value),
          "declaration", "sha256_text", context)
    return value


def _product(shape: object, kind: str) -> tuple[list[int], int]:
    dims = _sequence(shape, "tensor shape", kind)
    _need(len(dims) <= 8, kind if kind == "declaration" else "limit",
          "shape_rank", "tensor shape")
    result = [_integer(d, "shape dimension", kind) for d in dims]
    _need(all(d <= _MAX_DIM for d in result),
          "declaration" if kind == "declaration" else "limit",
          "shape_dimension_limit", "tensor shape")
    # A zero dimension makes the product zero regardless of earlier dimensions.
    if 0 in result:
        return result, 0
    product = 1
    for dimension in result:
        _need(product <= _MAX_INT // dimension,
              kind if kind == "declaration" else "limit", "shape_product", "tensor shape")
        product *= dimension
    return result, product


def classic_vit224_schema() -> StateSchema:
    """Return a fresh descriptive source schema; instantiate nothing."""
    shapes = {
        "cls_token": [1, 1, 1536], "reg_token": [1, 4, 1536],
        "pos_embed": [1, 256, 1536], "patch_embed.proj.weight": [1536, 3, 14, 14],
        "patch_embed.proj.bias": [1536], "norm.weight": [1536], "norm.bias": [1536],
    }
    block = {
        "norm1.weight": [1536], "norm1.bias": [1536],
        "norm2.weight": [1536], "norm2.bias": [1536],
        "attn.qkv.weight": [4608, 1536], "attn.qkv.bias": [4608],
        "attn.proj.weight": [1536, 1536], "attn.proj.bias": [1536],
        "ls1.gamma": [1536], "ls2.gamma": [1536],
        "mlp.fc1.weight": [8192, 1536], "mlp.fc1.bias": [8192],
        "mlp.fc2.weight": [1536, 4096], "mlp.fc2.bias": [1536],
    }
    for i in range(40):
        for name, shape in block.items():
            shapes[f"blocks.{i}.{name}"] = list(shape)
    tensors: list[TensorShape] = []
    for name, shape in sorted(shapes.items()):
        dims, numel = _product(shape, "declaration")
        tensors.append({"name": name, "shape": dims, "numel": numel})
    return {
        "schema_version": 1, "schema_id": "hoptimus_classic_vit224_timm_1_0_16",
        "constructor_args": {
            "img_size": 224, "patch_size": 14, "in_chans": 3, "num_classes": 0,
            "global_pool": "token", "embed_dim": 1536, "depth": 40, "num_heads": 24,
            "mlp_ratio": 5.33334, "init_values": 1e-5, "class_token": True,
            "reg_tokens": 4, "pos_embed": "learn", "no_embed_class": True,
            "qkv_bias": True, "qk_norm": False, "proj_bias": True,
            "scale_attn_norm": False, "scale_mlp_norm": False, "pre_norm": False,
            "final_norm": True, "fc_norm": False, "pool_include_prefix": False,
            "dynamic_img_size": False, "dynamic_img_pad": False,
            "drop_rate": 0, "pos_drop_rate": 0, "patch_drop_rate": 0,
            "proj_drop_rate": 0, "attn_drop_rate": 0, "drop_path_rate": 0,
            "weight_init": "", "fix_init": False, "embed_layer": "PatchEmbed",
            "embed_norm_layer": None,
            "norm_layer": {"name": "LayerNorm", "eps": 1e-6, "elementwise_affine": True},
            "act_layer": "SiLU", "block_fn": "Block", "mlp_layer": "SwiGLUPacked",
        },
        "tensors": tensors, "tensor_count": len(tensors),
        "scalar_count": sum(t["numel"] for t in tensors),
        "source_fingerprints": {
            "timm/models/vision_transformer.py": "d362a7ad3f5676833e04d28d5e02220e57122bdffff30fbcb1d71180b4de4407",
            "timm/layers/attention.py": "55c7accd5cddc8512efa472962117395fa9ef04f81d387735e11dcda0eb6513f",
            "timm/layers/mlp.py": "463e1c4d6f5a2340ae631cd189b32282c59eb2010c4722ab75f3024e0e16c4fe",
            "timm/layers/patch_embed.py": "78f2d6afba3bf71a0b8a51cfb06e034b08af0708f4a49509511e4b419e448cda",
            "timm/layers/norm.py": "60676020bebb1fecebe6770d8534580c808fd84527920a4904f54982961b8712",
        },
        "limitations": [
            "source_schema_only_not_instantiated", "not_actual_checkpoint_header_observation",
            "not_historical_model_identity", "not_byte_or_model_admission",
            "compute_attention_device_profile_unselected",
        ],
    }


def observe_png(encoded: object, *, declared_size: object) -> PNGObservation:
    """Observe bounded PNG framing/metadata without decompressing any content."""
    _need(type(encoded) is bytes, "declaration", "bytes_type", "PNG encoded")
    size = _integer(declared_size, "PNG declared_size")
    _need(size <= _ENCODED_LIMIT and len(encoded) <= _ENCODED_LIMIT,
          "limit", "png_encoded_size", "PNG encoded")
    _need(size == len(encoded), "declaration", "png_size_mismatch", "PNG encoded")
    _need(encoded.startswith(b"\x89PNG\r\n\x1a\n"),
          "malformed", "png_signature", "PNG encoded", 0)
    recognised = frozenset((
        "IHDR", "PLTE", "IDAT", "IEND", "tRNS", "gAMA", "cHRM", "sRGB",
        "iCCP", "eXIf", "acTL", "fcTL", "fdAT",
    ))
    ihdr: dict[str, int] = {}
    chunks: list[dict[str, object]] = []
    palette: dict[str, object] = {"present": False, "entry_count": None}
    transparency: dict[str, object] = {
        "present": False, "kind": None, "raw_samples_or_alpha_table": None,
    }
    colour: dict[str, object] = {
        "gama_u32": None, "chrm_u32_values": None, "srgb_intent": None,
        "iccp_descriptor": None,
    }
    exif: dict[str, object] = {
        "present": False, "length": None, "data_sha256": None, "orientation": None,
        "orientation_status": "absent",
    }
    apng: dict[str, object] = {
        "present": False, "actl_raw_fields": None, "chunk_descriptors": [],
        "semantic_status": "absent",
    }
    unknown: list[int] = []
    consequential: list[str] = []
    unsupported: list[str] = []
    seen: set[str] = set()
    offset, idat_state, terminated = 8, 0, False

    def add(code: str, destination: list[str] = consequential) -> None:
        if code not in destination:
            destination.append(code)

    while offset < size:
        _need(len(chunks) < 1024, "limit", "png_chunk_count", "PNG encoded", offset)
        _need(size - offset >= 12, "malformed", "png_chunk_truncated", "PNG encoded", offset)
        length = struct.unpack_from(">I", encoded, offset)[0]
        _need(length <= _ENCODED_LIMIT, "limit", "png_chunk_length", "PNG encoded", offset)
        end = offset + 12 + length
        _need(end <= size, "malformed", "png_chunk_truncated", "PNG encoded", offset)
        raw_type = encoded[offset + 4:offset + 8]
        _need(all(65 <= c <= 90 or 97 <= c <= 122 for c in raw_type)
              and 65 <= raw_type[2] <= 90,
              "malformed", "png_chunk_type", "PNG encoded", offset + 4)
        name = raw_type.decode("ascii")
        data = encoded[offset + 8:offset + 8 + length]
        crc = struct.unpack_from(">I", encoded, offset + 8 + length)[0]
        _need(binascii.crc32(raw_type + data) & 0xffffffff == crc,
              "malformed", "png_crc", "PNG encoded", offset)
        _need(bool(chunks) or name == "IHDR",
              "malformed", "png_ihdr_first", "PNG encoded", offset)
        if name not in ("IDAT", "IHDR") and idat_state == 1:
            idat_state = 2
        singleton = name in recognised - {"IDAT", "fcTL", "fdAT"}
        _need(not singleton or name not in seen,
              "malformed", "png_duplicate_chunk", f"PNG {name}", offset)
        interpretation = "unknown"
        if name == "IHDR":
            _need(not chunks and length == 13,
                  "malformed", "png_ihdr", "PNG encoded", offset)
            values = struct.unpack(">IIBBBBB", data)
            ihdr = dict(zip((
                "width", "height", "bit_depth", "colour_type",
                "compression_method", "filter_method", "interlace_method",
            ), values))
            depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8),
                      4: (8, 16), 6: (8, 16)}
            _need(1 <= values[0] <= _MAX_DIM and 1 <= values[1] <= _MAX_DIM
                  and values[2] in depths.get(values[3], ())
                  and values[4] == values[5] == 0 and values[6] in (0, 1),
                  "malformed", "png_ihdr_fields", "PNG encoded", offset)
            interpretation = "parsed_critical"
        elif name == "PLTE":
            _need(idat_state == 0 and "tRNS" not in seen
                  and ihdr["colour_type"] not in (0, 4)
                  and 3 <= length <= 768 and length % 3 == 0,
                  "malformed", "png_palette", "PNG PLTE", offset)
            entries = length // 3
            _need(ihdr["colour_type"] != 3 or entries <= 2**ihdr["bit_depth"],
                  "malformed", "png_palette_depth", "PNG PLTE", offset)
            palette = {"present": True, "entry_count": entries}
            add("palette_present")
            interpretation = "parsed_critical"
        elif name == "IDAT":
            _need(idat_state != 2 and (ihdr["colour_type"] != 3 or "PLTE" in seen),
                  "malformed", "png_idat_order", "PNG IDAT", offset)
            idat_state = 1
            interpretation = "opaque_compressed_pixels"
        elif name == "IEND":
            _need(length == 0 and idat_state != 0 and end == size,
                  "malformed", "png_termination", "PNG IEND", offset)
            terminated = True
            interpretation = "framing_only"
        elif name == "tRNS":
            _need(idat_state == 0, "malformed", "png_transparency_order", "PNG tRNS", offset)
            ct = ihdr["colour_type"]
            if ct in (0, 2):
                _need(length == (2 if ct == 0 else 6),
                      "malformed", "png_transparency_length", "PNG tRNS", offset)
                samples = list(struct.unpack(">" + "H" * (length // 2), data))
                _need(all(v < 2**ihdr["bit_depth"] for v in samples),
                      "malformed", "png_transparency_sample", "PNG tRNS", offset)
                kind = "grayscale_key" if ct == 0 else "rgb_key"
            else:
                _need(ct == 3 and "PLTE" in seen
                      and 1 <= length <= palette["entry_count"],
                      "malformed", "png_transparency_palette", "PNG tRNS", offset)
                samples, kind = list(data), "palette_alpha"
            transparency = {"present": True, "kind": kind, "raw_samples_or_alpha_table": samples}
            add("transparency_present")
            interpretation = "parsed_metadata"
        elif name in ("gAMA", "cHRM", "sRGB", "iCCP"):
            _need(idat_state == 0 and "PLTE" not in seen,
                  "malformed", "png_colour_order", f"PNG {name}", offset)
            if name == "gAMA":
                _need(length == 4, "malformed", "png_gamma_length", "PNG gAMA", offset)
                value = struct.unpack(">I", data)[0]
                _need(value != 0, "malformed", "png_gamma_zero", "PNG gAMA", offset)
                colour["gama_u32"] = value
                add("gamma_present")
            elif name == "cHRM":
                _need(length == 32, "malformed", "png_chromaticity_length", "PNG cHRM", offset)
                colour["chrm_u32_values"] = list(struct.unpack(">8I", data))
                add("chromaticity_present")
            elif name == "sRGB":
                _need(length == 1 and data[0] <= 3,
                      "malformed", "png_srgb_intent", "PNG sRGB", offset)
                colour["srgb_intent"] = data[0]
                add("srgb_present")
            else:
                separator = data.find(b"\0")
                keyword = data[:separator] if separator >= 0 else b""
                _need(1 <= len(keyword) <= 79
                      and all(32 <= c <= 126 or 161 <= c <= 255 for c in keyword)
                      and not keyword.startswith(b" ") and not keyword.endswith(b" ")
                      and b"  " not in keyword and separator + 2 < length
                      and data[separator + 1] == 0,
                      "malformed", "png_iccp_framing", "PNG iCCP", offset)
                compressed = data[separator + 2:]
                colour["iccp_descriptor"] = {
                    "keyword": keyword.decode("latin-1"), "compression_method": 0,
                    "compressed_length": len(compressed),
                    "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                }
                add("icc_present")
            if name == "sRGB" and "iCCP" in seen or name == "iCCP" and "sRGB" in seen:
                add("srgb_iccp_conflict")
            interpretation = "parsed_metadata"
        elif name == "eXIf":
            exif = {
                "present": True, "length": length,
                "data_sha256": hashlib.sha256(data).hexdigest(), "orientation": None,
                "orientation_status": "uninterpreted",
            }
            add("exif_uninterpreted")
            interpretation = "uninterpreted_exif"
        elif name in ("acTL", "fcTL", "fdAT"):
            _need((name == "acTL" and length == 8 and idat_state == 0)
                  or (name == "fcTL" and length == 26)
                  or (name == "fdAT" and length >= 4),
                  "malformed", "png_apng_framing", f"PNG {name}", offset)
            apng["present"], apng["semantic_status"] = True, "uninterpreted"
            if name == "acTL":
                frames, plays = struct.unpack(">II", data)
                apng["actl_raw_fields"] = {"num_frames": frames, "num_plays": plays}
            add("apng_uninterpreted")
            add("apng_semantics_uninterpreted", unsupported)
            interpretation = "uninterpreted_apng"
        else:
            unknown.append(len(chunks))
            if raw_type[0] < 97:
                add("unknown_critical")
                add("unknown_critical_chunk", unsupported)
            else:
                add("unknown_ancillary")
        descriptor = {
            "index": len(chunks), "offset": offset, "type": name, "length": length,
            "crc_valid": True, "data_sha256": hashlib.sha256(data).hexdigest(),
            "critical": raw_type[0] < 97, "recognised": name in recognised,
            "interpretation": interpretation,
        }
        chunks.append(descriptor)
        if name in ("acTL", "fcTL", "fdAT"):
            apng["chunk_descriptors"].append(dict(descriptor))
        seen.add(name)
        offset = end
    _need(terminated, "malformed", "png_missing_iend", "PNG encoded", offset)
    return {
        "schema_version": 1, "declared_size": size, "supplied_size": size,
        "png_framing_valid": True, "ihdr": ihdr, "chunks": chunks, "palette": palette,
        "transparency": transparency, "colour": colour, "exif": exif, "apng": apng,
        "unknown_chunks": unknown, "consequential_properties": consequential,
        "unsupported_properties": unsupported,
    }


def _json_integer(token: str) -> int:
    _need(len(token.lstrip("-")) <= 19, "limit", "json_integer_token", "safetensors.header")
    value = int(token)
    _need(abs(value) <= _MAX_INT, "limit", "json_integer_token", "safetensors.header")
    return value


def _json_no_float(token: str) -> None:
    raise AdmissionError("malformed", "json_number", "safetensors.header floating/nonfinite token")


def _json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _utf8(key, "safetensors.header key", 256, "malformed")
        _need(key not in result, "malformed", "json_duplicate_key", "safetensors.header")
        result[key] = value
    return result


def _header_json(header: bytes) -> dict[str, object]:
    _need(header.startswith(b"{"), "malformed", "json_object_start", "safetensors.header", 0)
    try:
        text = header.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AdmissionError("malformed", "json_utf8", "safetensors.header", exc.start) from exc
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
            _need(depth <= 16, "limit", "json_nesting", "safetensors.header")
        elif char in "}]":
            depth -= 1
    try:
        result = json.loads(text, object_pairs_hook=_json_pairs, parse_int=_json_integer,
                            parse_float=_json_no_float, parse_constant=_json_no_float)
    except json.JSONDecodeError as exc:
        byte_offset = len(text[:exc.pos].encode("utf-8"))
        raise AdmissionError("malformed", "json_syntax", "safetensors.header", byte_offset) from exc
    _need(type(result) is dict, "malformed", "json_object", "safetensors.header")
    return result


def _metadata(value: object, kind: str) -> dict[str, str] | None:
    if value is None:
        return None
    _need(type(value) is dict, kind, "metadata_type", "safetensors metadata")
    _need(len(value) <= 128, kind if kind == "declaration" else "limit",
          "metadata_count", "safetensors metadata")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_utf8(key, "metadata key", 256, kind)] = _utf8(
            item, "metadata value", 4096, kind)
    return result


def _tensor(name: object, dtype: object, shape: object, offsets: object,
            payload_size: int, kind: str) -> TensorObservation:
    name = _utf8(name, "tensor name", 256, kind)
    dtype = _utf8(dtype, "tensor dtype", 256, kind)
    _need(dtype in _WIDTHS, kind if kind == "declaration" else "unsupported",
          "dtype_unsupported", "tensor dtype")
    dims, numel = _product(shape, kind)
    pair = _sequence(offsets, "tensor offsets", kind)
    _need(len(pair) == 2, kind, "offset_pair", "tensor offsets")
    start, end = (_integer(v, "tensor offset", kind) for v in pair)
    _need(start <= end <= payload_size, kind, "offset_bounds", "tensor offsets")
    width = _WIDTHS[dtype]
    _need(numel <= _MAX_INT // width, kind if kind == "declaration" else "limit",
          "tensor_byte_product", "tensor bytes")
    nbytes = numel * width
    _need(end - start == nbytes, kind, "tensor_byte_length", "tensor bytes")
    return {"name": name, "dtype": dtype, "shape": dims, "data_offsets": [start, end],
            "numel": numel, "nbytes": nbytes}


def _packing(tensors: list[TensorObservation], payload_size: int,
             kind: str) -> tuple[int, dict[str, int]]:
    cursor, scalar_count = 0, 0
    boundaries = {0}
    totals: dict[str, int] = {}
    nonempty = sorted((t for t in tensors if t["nbytes"]),
                      key=lambda t: t["data_offsets"][0])
    for tensor in nonempty:
        start, end = tensor["data_offsets"]
        _need(start == cursor, kind, "payload_packing", "safetensors extents")
        cursor = end
        boundaries.add(end)
    _need(cursor == payload_size, kind, "payload_unaccounted", "safetensors payload")
    for tensor in tensors:
        _need(tensor["nbytes"] != 0 or tensor["data_offsets"][0] in boundaries,
              kind, "empty_extent_boundary", "safetensors empty extent")
        _need(scalar_count <= _MAX_INT - tensor["numel"],
              kind if kind == "declaration" else "limit", "scalar_total", "safetensors tensors")
        scalar_count += tensor["numel"]
        dtype = tensor["dtype"]
        totals[dtype] = totals.get(dtype, 0) + tensor["nbytes"]
    return scalar_count, dict(sorted(totals.items()))


def observe_safetensors(prefix8: object, header: object, *,
                        total_size: object) -> SafetensorsObservation:
    """Check supplied header bytes and declared extents, not payload values."""
    _need(type(prefix8) is bytes and type(header) is bytes,
          "declaration", "bytes_type", "safetensors supplied regions")
    _need(len(prefix8) == 8, "malformed", "prefix_length", "safetensors.prefix8", 0)
    total = _integer(total_size, "safetensors total_size")
    length = struct.unpack("<Q", prefix8)[0]
    _need(length <= _ENCODED_LIMIT and len(header) <= _ENCODED_LIMIT,
          "limit", "header_length", "safetensors.header")
    _need(length == len(header), "declaration", "header_length_mismatch", "safetensors.header")
    _need(8 + length <= total, "malformed", "header_total_size", "safetensors.header")
    decoded = _header_json(header)
    metadata = _metadata(decoded["__metadata__"], "malformed") if "__metadata__" in decoded else None
    if "__metadata__" in decoded:
        _need(decoded["__metadata__"] is not None,
              "malformed", "metadata_type", "safetensors metadata")
    members = [(name, value) for name, value in decoded.items() if name != "__metadata__"]
    _need(len(members) <= 4096, "limit", "tensor_count", "safetensors.header")
    payload_size = total - 8 - length
    tensors: list[TensorObservation] = []
    for name, value in sorted(members):
        record = _record(value, frozenset(("dtype", "shape", "data_offsets")),
                         "safetensors tensor", "malformed")
        tensors.append(_tensor(name, record["dtype"], record["shape"],
                               record["data_offsets"], payload_size, "malformed"))
    scalar_count, totals = _packing(tensors, payload_size, "malformed")
    return {
        "schema_version": 1, "total_size": total, "header_length": length,
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "payload_offset": 8 + length, "payload_size": payload_size,
        "header_structure_valid": True, "metadata": metadata, "tensors": tensors,
        "tensor_count": len(tensors), "scalar_count": scalar_count,
        "payload_bytes_by_dtype": totals, "unsupported_properties": [],
    }


def _validate_observation(observation: object) -> list[TensorObservation]:
    record = _record(observation, _SAFE_FIELDS, "safetensors observation")
    _need(type(record["schema_version"]) is int and record["schema_version"] == 1
          and record["header_structure_valid"] is True,
          "declaration", "observation_version_flag", "safetensors observation")
    total = _integer(record["total_size"], "observation total_size")
    length = _integer(record["header_length"], "observation header_length",
                      minimum=2, maximum=_ENCODED_LIMIT)
    offset = _integer(record["payload_offset"], "observation payload_offset")
    payload = _integer(record["payload_size"], "observation payload_size")
    _need(offset == 8 + length and total == offset + payload,
          "declaration", "observation_layout", "safetensors observation")
    _sha(record["header_sha256"], "observation header SHA")
    metadata = _metadata(record["metadata"], "declaration")
    unsupported = _sequence(record["unsupported_properties"], "observation unsupported_properties")
    _need(not unsupported, "declaration", "observation_unsupported", "safetensors observation")
    items = _sequence(record["tensors"], "observation tensors")
    _need(len(items) <= 4096, "declaration", "tensor_count", "safetensors observation")
    tensors: list[TensorObservation] = []
    for value in items:
        item = _record(value, frozenset(("name", "dtype", "shape", "data_offsets", "numel", "nbytes")),
                       "observation tensor")
        tensor = _tensor(item["name"], item["dtype"], item["shape"],
                         item["data_offsets"], payload, "declaration")
        _need(tensor["name"] != "__metadata__", "declaration",
              "observation_reserved_name", "safetensors observation")
        _need(_integer(item["numel"], "observation numel") == tensor["numel"]
              and _integer(item["nbytes"], "observation nbytes") == tensor["nbytes"],
              "declaration", "observation_tensor_totals", "safetensors observation")
        tensors.append(tensor)
    names = [t["name"] for t in tensors]
    _need(names == sorted(names) and len(set(names)) == len(names),
          "declaration", "observation_tensor_names", "safetensors observation")
    count, totals = _packing(tensors, payload, "declaration")
    _need(_integer(record["tensor_count"], "observation tensor_count") == len(tensors)
          and _integer(record["scalar_count"], "observation scalar_count") == count,
          "declaration", "observation_counts", "safetensors observation")
    supplied = record["payload_bytes_by_dtype"]
    _need(type(supplied) is dict and all(type(key) is str for key in supplied)
          and set(supplied) == set(totals),
          "declaration", "observation_dtype_totals", "safetensors observation")
    for dtype, value in supplied.items():
        _need(_integer(value, "observation dtype total") == totals[dtype],
              "declaration", "observation_dtype_totals", "safetensors observation")
    contents = {
        t["name"]: {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": t["data_offsets"]}
        for t in tensors
    }
    if metadata is not None:
        contents["__metadata__"] = metadata
    # This is only a length lower bound; original order, escapes and hash are unavailable.
    minimum_length = len(json.dumps(contents, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    _need(length >= minimum_length, "declaration", "observation_header_minimum",
          "safetensors observation")
    return tensors


def compare_vit224_schema(observation: object) -> SchemaComparison:
    """Revalidate representation/packing, then report structural differences."""
    tensors = _validate_observation(observation)
    expected = {t["name"]: t["shape"] for t in classic_vit224_schema()["tensors"]}
    observed = {t["name"]: t for t in tensors}
    missing = sorted(expected.keys() - observed.keys())
    unexpected = sorted(observed.keys() - expected.keys())
    shapes = [
        {"name": name, "expected_shape": list(expected[name]),
         "observed_shape": list(observed[name]["shape"])}
        for name in sorted(expected.keys() & observed.keys())
        if expected[name] != observed[name]["shape"]
    ]
    dtypes = [{"name": t["name"], "dtype": t["dtype"]}
              for t in tensors if t["dtype"] not in _FLOATS]
    return {"match": not (missing or unexpected or shapes or dtypes),
            "missing_names": missing, "unexpected_names": unexpected,
            "shape_mismatches": shapes, "dtype_issues": dtypes}


def _slide(value: object) -> str:
    sid = _identity(value, "slide_id")
    _need(sid not in (".", "..") and "/" not in sid and "\\" not in sid,
          "declaration", "slide_component", "slide_id")
    return sid


def _probe(value: object, lexical_path: str) -> dict[str, object]:
    record = _record(value, _PROBE_FIELDS, "PathProbe")
    _need(_path(record["lexical_path"], "probe lexical_path") == lexical_path,
          "declaration", "probe_lexical_join", "PathProbe")
    _need(type(record["lexical_type"]) is str and record["lexical_type"] in _PATH_TYPES,
          "declaration", "probe_lexical_type", "PathProbe")
    resolved = record["resolved_path"]
    if resolved is not None:
        _path(resolved, "probe resolved_path")
    resolved_type = record["resolved_type"]
    _need(resolved_type is None or type(resolved_type) is str and resolved_type in _PATH_TYPES,
          "declaration", "probe_resolved_type", "PathProbe")
    size = record["size_bytes"]
    if size is not None:
        _integer(size, "probe size")
    coherence = record["probe_coherent"]
    _need(coherence is None or type(coherence) is bool,
          "declaration", "probe_coherence", "PathProbe")
    status = record["status"]
    _need(type(status) is str and status in _STATUSES,
          "declaration", "probe_status", "PathProbe")
    operation, errno, message = (record[k] for k in (
        "error_operation", "error_errno", "error_message"))
    if operation is not None:
        _identity(operation, "probe error operation", 1024)
    if errno is not None:
        _integer(errno, "probe errno")
    if message is not None:
        _need(type(message) is str and 0 < len(message) <= 1024,
              "declaration", "probe_error_message", "PathProbe")
        _utf8(message, "probe error message", 4096)
    if status == "observed_regular":
        _need(record["lexical_type"] in ("regular", "symlink")
              and resolved is not None and resolved_type == "regular"
              and size is not None and coherence is True
              and operation is errno is message is None,
              "declaration", "probe_regular_consistency", "PathProbe")
    elif status in ("missing_lexical", "broken_symlink"):
        _need(record["lexical_type"] == ("absent" if status == "missing_lexical" else "symlink")
              and resolved is resolved_type is size is None,
              "declaration", "probe_absence_consistency", "PathProbe")
    elif status == "not_regular":
        _need(resolved is not None and resolved_type in ("symlink", "directory", "other"),
              "declaration", "probe_nonregular_consistency", "PathProbe")
        _need(coherence is not True
              or record["lexical_type"] in ("symlink", "unknown")
              or record["lexical_type"] == resolved_type,
              "declaration", "probe_coherent_leaf_type", "PathProbe")
    else:
        _need(operation is not None and message is not None,
              "declaration", "probe_error_evidence", "PathProbe")
    if status == "changed_during_probe":
        _need(coherence is False, "declaration", "probe_changed_consistency", "PathProbe")
    if errno is not None or message is not None:
        _need(operation is not None, "declaration", "probe_error_evidence", "PathProbe")
    return dict(record)


def inventory_targets(targets: object, metadata: object, probes: object,
                      prior_stage1b: object, *, lexical_root: object) -> InventoryObservation:
    """Reconcile supplied cohort/probe records; never inspect a filesystem."""
    target_rows = _sequence(targets, "targets")
    meta_rows = _sequence(metadata, "metadata")
    _need(len(target_rows) == 2778 and len(meta_rows) == 2921,
          "declaration", "inventory_population_size", "inventory")
    root = _path(lexical_root, "lexical_root")
    _need(type(probes) is dict and type(prior_stage1b) is dict,
          "declaration", "inventory_mapping_type", "inventory")
    _need(all(type(key) is str for key in probes)
          and all(type(key) is str for key in prior_stage1b),
          "declaration", "inventory_mapping_keys", "inventory")
    meta_by_id: dict[str, dict[str, object]] = {}
    selected: list[dict[str, object]] = []
    for index, value in enumerate(meta_rows):
        row = _record(value, _METADATA_FIELDS, "MetadataRecord")
        _need(_integer(row["metadata_index"], "metadata index") == index,
              "declaration", "metadata_order", "inventory")
        sid = _slide(row["slide_id"])
        _identity(row["patient_id"], "metadata patient_id")
        group = _identity(row["diagnostic_group"], "metadata diagnostic_group")
        _need(sid not in meta_by_id, "declaration", "metadata_duplicate_slide", "inventory")
        meta_by_id[sid] = row
        if group != "MISC":
            selected.append(row)
    _need(len(selected) == 2778, "declaration", "inventory_non_misc_count", "inventory")
    expected_ids = {row["slide_id"] for row in selected}
    _need(set(probes) == expected_ids and set(prior_stage1b) <= expected_ids,
          "declaration", "inventory_probe_prior_keys", "inventory")
    prior_records: dict[str, dict[str, object]] = {}
    for sid, value in prior_stage1b.items():
        row = _record(value, _PRIOR_FIELDS, "PriorBinding")
        _path(row["resolved_path"], "prior resolved_path")
        _integer(row["size_bytes"], "prior size")
        _sha(row["sha256"], "prior SHA")
        _identity(row["receipt_reference"], "prior receipt_reference")
        prior_records[sid] = row
    rows: list[dict[str, object]] = []
    counts = {status: 0 for status in _STATUSES}
    seen: set[str] = set()
    for index, value in enumerate(target_rows):
        target = _record(value, _TARGET_FIELDS, "TargetRecord")
        _need(_integer(target["population_index"], "population index") == index,
              "declaration", "target_order", "inventory")
        _integer(target["metadata_index"], "target metadata index")
        _integer(target["expected_tile_rows"], "target tile count", minimum=1)
        sid = _slide(target["slide_id"])
        _identity(target["patient_id"], "target patient_id")
        _identity(target["diagnostic_group"], "target diagnostic_group")
        _need(sid not in seen and all(target[key] == selected[index][key]
                                     for key in _METADATA_FIELDS),
              "declaration", "target_metadata_join", "inventory")
        seen.add(sid)
        lexical = (root.rstrip("/") + "/" + sid + ".svs")
        _path(lexical, "joined lexical_path")
        probe = _probe(probes[sid], lexical)
        prior = prior_records.get(sid)
        row = dict(target)
        row.update(probe)
        row.update({
            "prior_stage1b": prior is not None,
            "prior_resolved_path": prior["resolved_path"] if prior else None,
            "prior_size_bytes": prior["size_bytes"] if prior else None,
            "prior_sha256": prior["sha256"] if prior else None,
            "prior_receipt_reference": prior["receipt_reference"] if prior else None,
            "prior_path_match": probe["resolved_path"] == prior["resolved_path"]
            if prior and probe["resolved_path"] is not None else None,
            "prior_size_match": probe["size_bytes"] == prior["size_bytes"]
            if prior and probe["size_bytes"] is not None else None,
            "content_admission": "not_performed",
        })
        rows.append(row)
        counts[probe["status"]] += 1
    return {
        "schema_version": 1, "inventory_complete": True, "rows": rows,
        "status_counts": counts, "content_admission": "not_performed",
        "limitations": [
            "caller_supplied_path_probes_not_filesystem_verification",
            "prior_stage1b_bindings_not_reread", "content_admission_not_performed",
            "complete_inventory_may_contain_unavailable_rows",
            "no_population_narrowing_or_substitution",
        ],
    }
