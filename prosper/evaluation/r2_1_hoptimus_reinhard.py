"""Prospective ordinary CPU FP32 Reinhard with source-mapped validity guards.

The caller binds encoded template bytes/decoding and the executing/native
closure. A supplied digest is recorded, not authenticated from decoded pixels.
No paths to images, implicit decoding, historical-equivalence claim or fallback
are supported. Native-one/thread constraints remain the caller's responsibility.

The ordinary orchestration and instrumented Lab-to-RGB below are minimal
derivatives of EIDOSLAB/torchstain at commit
3e10af2299998fe9e2e3e5072294bc08ffebb6c1:
torchstain/torch/normalizers/reinhard.py (SHA256
411d237be788a9f97bb6d31869ed826303fbfd3f663bd97eb57491f483e74031);
torchstain/torch/utils/lab2rgb.py (SHA256
aab1885c752fc270cfd1770e6f7feae1a7dce86fd3b820d674100911d2e9e0c0).
The original selected-branch arithmetic, clamp and cast are retained.

MIT License

Copyright (c) 2020 EIDOSlab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF SPECIFICATION, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module, metadata
from importlib.util import find_spec
import json
from pathlib import Path
import re
import sys
from types import CodeType, FunctionType, ModuleType
from typing import Callable

import numpy as np
from PIL import Image
import torch
from torch.overrides import _get_current_function_mode_stack
from torch.utils._device import DeviceContext

from .r2_1_hoptimus_pixels import ReinhardBridge, _materialized_image


SOURCE_COMMIT = "3e10af2299998fe9e2e3e5072294bc08ffebb6c1"
SOURCE_ARCHIVE_SHA256 = "6469c29d7dd8e231aff4bd76d23b138cd2ac97cc81128c979ec2d3468641ecdc"
SOURCE_AUDIT_SHA256 = "d5cc7a0697de96c529a35a24d023856a13520ac0bdd8f8c51afac90e3a6add73"
CORRESPONDENCE_SHA256 = "f0c7d8176787bd0c40d9bd6ab1233812da562f6a4cba88b64727b89a529a5d10"
PACKAGE_MEMBERS: tuple[tuple[str, int, str], ...] = (
    ("torchstain/__init__.py", 74, "b46e0ff3f0fe782c412cab3f4cdcee11cd86997796ec417ed40a29d2e891d407"),
    ("torchstain/base/__init__.py", 51, "f207c3bf634d861827de5374b92a497ea9c2a7f5fe5d483682ed365108852f9e"),
    ("torchstain/base/augmentors/__init__.py", 76, "8c88d4d4a3a5eb43f1d38247970dc7ea2bd74023723f74c9f495bbb6fd459c8a"),
    ("torchstain/base/augmentors/he_augmentor.py", 121, "feeb0e3eb7d8ca46a5fd766fdd744c2cbf1fa61b1e5d7e65f9feeed28842355e"),
    ("torchstain/base/augmentors/macenko.py", 635, "ba62cc251978fcdf557a22a7d2e1e6fb36e6d53f0998a78b43efb0d560b599bc"),
    ("torchstain/base/normalizers/__init__.py", 168, "26af76da209303da1c33a3506148f5fb42c7c1148f35b03ac8a33324af58028d"),
    ("torchstain/base/normalizers/he_normalizer.py", 142, "ba76a8f3d390c10a1d351abe6297f3edd1ae087acef3c7f698c3890342ee2cca"),
    ("torchstain/base/normalizers/macenko.py", 553, "dc1d0ace3a3d44eaf1440a62d51d5675be960443dc4758c563c358227ef0f4cf"),
    ("torchstain/base/normalizers/multitarget.py", 542, "03df7b09e5ceaf70f5d6f1ef025808d881f684ebe70a9e9e243a1e006e510c12"),
    ("torchstain/base/normalizers/reinhard.py", 596, "5cd24c9d6bfddb5fd741b6fb5b7695fdcd2bed8756679f3ab65adb33cacde4d1"),
    ("torchstain/numpy/__init__.py", 60, "24efe7c36413be710a5169e4ca12237eb36c9265ddc3c73acc7241d781ed0b51"),
    ("torchstain/numpy/augmentors/__init__.py", 42, "4441d1a164bc3aa687be2f4359531e5eaed26d8d8721a8c313f3af2cc937db54"),
    ("torchstain/numpy/augmentors/macenko.py", 3496, "7da07fea92864947127ec2c7c0edfb5c907172234e4ee22f5198dc75a5244955"),
    ("torchstain/numpy/normalizers/__init__.py", 89, "e51ed9635d8bcdc302997f85943b7c638c83d1540138c0eaf6c6e7d80031e339"),
    ("torchstain/numpy/normalizers/macenko.py", 4042, "689e9be892585dfe71fff12313995ce8435ea373cea665862497a492c5d65af2"),
    ("torchstain/numpy/normalizers/reinhard.py", 2551, "bdf7f2d990f6effebefdb53b09d2a25e34968d44396ee9501093ff5ec75e1f29"),
    ("torchstain/numpy/utils/__init__.py", 176, "c6fafd9d6d8e0781c3c9f26f8e1c4b6192a1d28bb06c65180041e28f8a13343d"),
    ("torchstain/numpy/utils/lab2rgb.py", 1272, "25e14a383151bc2ce8538ef3b9e77aea072fc8f4a4ba7ba99b14e8209332e1ed"),
    ("torchstain/numpy/utils/rgb2lab.py", 1415, "386dd68dce0b06f31ea4785b894025531f1d62a157b904a0a175188a061413c4"),
    ("torchstain/numpy/utils/split.py", 397, "c19d4c766010c2b3ed02c91ba9bd5cd0d7308b09f0ed39c001b5c76e254a3db1"),
    ("torchstain/numpy/utils/stats.py", 130, "59841a348a9bb4893128a3743bbebedef8cb06448d0a26fec97b722f2c53cf7d"),
    ("torchstain/tf/__init__.py", 57, "7d7429a397cd14af9c4104e539685acf74f61e859188ab69087e8454aed47c54"),
    ("torchstain/tf/augmentors/__init__.py", 48, "2d39307ec4981c5f9b080f91972d52ff593096f5de84122cc746bac96f56ef5f"),
    ("torchstain/tf/augmentors/macenko.py", 3652, "0705a9493f61be1987e28295a2f4d0c74ff5f27548535545ed0b70524c02c1d1"),
    ("torchstain/tf/normalizers/__init__.py", 150, "c233496c28b371b20e0fbc6d4e45e16e7015a064563f5c8c61d37fe802e405e6"),
    ("torchstain/tf/normalizers/macenko.py", 4512, "d86cfdce49b657b5147d35691091232261af18d860e2345f9ce09e13d1d3218e"),
    ("torchstain/tf/normalizers/reinhard.py", 2570, "ea692c02fa64dac4f35e49238d95a73bf69d34d99cde40cfc2ac6e2902db350c"),
    ("torchstain/tf/utils/__init__.py", 306, "25a2224e1360284a0e027d6d0daa1b2c4d6a104a3ef687dbc0cc9832d792d776"),
    ("torchstain/tf/utils/cov.py", 238, "7c68af61e93d9173efd40c90dbf6ed67dffc5053cf1998777de553ae0f248a82"),
    ("torchstain/tf/utils/lab2rgb.py", 1362, "0825d78c45335e8cf632e72d3b70c259e2cf488bf97b3bac9d4e32b74a24a732"),
    ("torchstain/tf/utils/percentile.py", 661, "1831ac3823d06a97de071317782b0560de865a3e70b622183be971fb3a1f83b6"),
    ("torchstain/tf/utils/rgb2lab.py", 1633, "f975e14f0cf0c2dd9b2a7477feebcd96de3eadd077c4ea9b5c8cb037e3c0449c"),
    ("torchstain/tf/utils/solveLS.py", 226, "a2fb8ec62711f8addc041935f83b20ec4c110965771795ebf3e988584e802def"),
    ("torchstain/tf/utils/split.py", 354, "2384af12262b175367c2e9949433ae965779f885f15b339ff977420dc9f82e44"),
    ("torchstain/tf/utils/stats.py", 159, "2ee69e5811475a601884fca5f1ca157eef48da38d10eec43ca54b862a5295e76"),
    ("torchstain/torch/__init__.py", 59, "833da4bb8d7fda936c0db135a397d4e6ee9bd88113e25d8695237ae413c2dbf4"),
    ("torchstain/torch/augmentors/__init__.py", 43, "c72c6c846be000599b7d9d0691f36f7612f6c02fdb37362785f138f0cbfbfc0c"),
    ("torchstain/torch/augmentors/macenko.py", 3773, "3adfb05f8ac9440e7ca914adc1486b5541788d99e90f0ad8551e9a08adf338fa"),
    ("torchstain/torch/normalizers/__init__.py", 227, "95ed1cf6c7d5423710050924dda0b3d154d8596c3d58ccb941f1c4fb62b18ae5"),
    ("torchstain/torch/normalizers/macenko.py", 4311, "eb0c140316710e81a7ce3319985d28d29f706e1362b105031d529e7064eaa1c1"),
    ("torchstain/torch/normalizers/multitarget.py", 4995, "810306abaddc80ca18194b0761e2daed32ec6ef1658994b946f79c912ad0629d"),
    ("torchstain/torch/normalizers/reinhard.py", 2549, "411d237be788a9f97bb6d31869ed826303fbfd3f663bd97eb57491f483e74031"),
    ("torchstain/torch/utils/__init__.py", 276, "5d5f6bf8aed8ab52718bf2c848a9b37a2e806449dac98f904d004b5e218e88f0"),
    ("torchstain/torch/utils/cov.py", 189, "1b30d1e4ad85659b5d3c53404c7e17ddc0c304926c2bcd3f68a6581895f3a416"),
    ("torchstain/torch/utils/lab2rgb.py", 1295, "aab1885c752fc270cfd1770e6f7feae1a7dce86fd3b820d674100911d2e9e0c0"),
    ("torchstain/torch/utils/percentile.py", 939, "05ef92b6dd216fa28ebd7721d1be861f0c473308c160b65afe389be9fb9efc3d"),
    ("torchstain/torch/utils/rgb2lab.py", 1527, "2e3214a2cf01a8d854c85b5e2265325291fc33c5814832d2a934cf37ce7bec2b"),
    ("torchstain/torch/utils/split.py", 337, "3ebc25b1e3f6555e396bd211486fb54809e14f5a201aeefa633dbd1cc7307bc1"),
    ("torchstain/torch/utils/stats.py", 130, "5fc037b61babd1ffc917ce14a903b51b1e813c2599562c14c2aa2c8b7b667353"),
)


class ReinhardValidationError(ValueError):
    """Pinned source, ambient context, image, state or selected arithmetic is invalid."""


def _need(valid: bool, message: str) -> None:
    if not valid:
        raise ReinhardValidationError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def _execution_context() -> tuple[tuple[str, str], ...]:
    versions = tuple((name, metadata.version(name)) for name in ("torch", "numpy", "Pillow"))
    _need(versions == (("torch", "2.5.1"), ("numpy", "2.2.4"), ("Pillow", "11.3.0")),
          "released CPU dependency distribution versions do not match")
    _need(torch.get_default_dtype() == torch.float32, "default dtype must be FP32")
    # Torch 2.5.1's public getter can allocate a probe. These stored-state getters
    # and overrides.py:2056 accessor allocate no Tensor and do not pop the stack.
    _need(torch._C._get_default_device() == "cpu", "underlying default device must be CPU")
    global_context = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
    _need(global_context is None or isinstance(global_context, DeviceContext),
          "invalid stored default DeviceContext")
    contexts = list(_get_current_function_mode_stack())
    if global_context is not None:
        contexts.append(global_context)
    for context in contexts:
        if isinstance(context, DeviceContext):
            device = context.device
            _need(type(device) is torch.device and device.type == "cpu",
                  "active/default DeviceContext must be CPU")
    _need(not torch.is_autocast_enabled("cpu"), "active CPU autocast is not allowed")
    return versions


def _member_locations() -> dict[str, Path]:
    try:
        distribution = metadata.distribution("torchstain")
    except metadata.PackageNotFoundError as exc:
        raise ReinhardValidationError("verified private torchstain dependency is required") from exc
    _need(distribution.version == "1.4.1", "torchstain distribution must be 1.4.1")
    declared = distribution.files
    _need(declared is not None, "torchstain distribution file inventory is missing")
    package_python = [str(item) for item in declared
                      if str(item).startswith("torchstain/") and str(item).endswith(".py")]
    expected = {name for name, _, _ in PACKAGE_MEMBERS}
    _need(len(package_python) == len(expected) and set(package_python) == expected,
          "torchstain Python member inventory does not match the source pin")
    locations: dict[str, Path] = {}
    for name, size, digest in PACKAGE_MEMBERS:
        path = Path(distribution.locate_file(name)).resolve(strict=True)
        _need(path.is_file() and path.stat().st_size == size, f"source member size: {name}")
        _need(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
              f"source member hash: {name}")
        locations[name] = path
    _loaded_locations(locations)
    return locations


def _module_member(name: str) -> str:
    tail = name.replace(".", "/")
    package = tail + "/__init__.py"
    return package if package in {p for p, _, _ in PACKAGE_MEMBERS} else tail + ".py"


def _loaded_locations(locations: dict[str, Path]) -> None:
    for name, module in tuple(sys.modules.items()):
        if name == "torchstain" or name.startswith("torchstain."):
            _need(isinstance(module, ModuleType), f"invalid loaded candidate module: {name}")
            member = _module_member(name)
            _need(member in locations, f"unbound loaded candidate module: {name}")
            file = getattr(module, "__file__", None)
            origin = getattr(getattr(module, "__spec__", None), "origin", None)
            _need(type(file) is str and type(origin) is str
                  and Path(file).resolve() == locations[member]
                  and Path(origin).resolve() == locations[member],
                  f"loaded candidate location mismatch: {name}")
            if member.endswith("/__init__.py"):
                paths = getattr(module, "__path__", ())
                _need(tuple(Path(p).resolve() for p in paths) == (locations[member].parent,),
                      f"loaded candidate package path mismatch: {name}")


def _effective_import_spec(locations: dict[str, Path]) -> None:
    specification = find_spec("torchstain")
    if specification is None or type(specification.origin) is not str:
        raise ReinhardValidationError("effective torchstain import specification is missing")
    package_init = locations["torchstain/__init__.py"]
    _need(Path(specification.origin).resolve() == package_init,
          "effective torchstain import specification origin mismatch")
    search_locations = specification.submodule_search_locations
    if search_locations is None:
        raise ReinhardValidationError("effective torchstain import specification is not a package")
    _need(tuple(Path(path).resolve() for path in search_locations) == (package_init.parent,),
          "effective torchstain import specification package search locations mismatch")


@dataclass(frozen=True)
class _Primitives:
    rgb2lab: Callable[[torch.Tensor], torch.Tensor]
    lab_split: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    lab_merge: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    get_mean_std: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    standardize: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    rgb_module: ModuleType
    inverse_module: ModuleType

    @property
    def functions(self) -> tuple[Callable[..., object], ...]:
        return self.rgb2lab, self.lab_split, self.lab_merge, self.get_mean_std, self.standardize


def _candidate() -> _Primitives:
    _execution_context()
    locations = _member_locations()
    _effective_import_spec(locations)
    root = import_module("torchstain")
    rgb = import_module("torchstain.torch.utils.rgb2lab")
    inverse = import_module("torchstain.torch.utils.lab2rgb")
    split = import_module("torchstain.torch.utils.split")
    stats = import_module("torchstain.torch.utils.stats")
    _loaded_locations(locations)
    _need(getattr(root, "__version__", None) == "1.3.0", "stale module version identity mismatch")
    primitives = _Primitives(rgb.rgb2lab, split.lab_split, split.lab_merge,
                             stats.get_mean_std, stats.standardize, rgb, inverse)
    for function, (name, function_name) in zip(primitives.functions, (
        ("torchstain.torch.utils.rgb2lab", "rgb2lab"),
        ("torchstain.torch.utils.split", "lab_split"),
        ("torchstain.torch.utils.split", "lab_merge"),
        ("torchstain.torch.utils.stats", "get_mean_std"),
        ("torchstain.torch.utils.stats", "standardize"),
    )):
        _need(isinstance(function, FunctionType) and function.__module__ == name
              and function.__name__ == function_name
              and Path(function.__code__.co_filename).resolve() == locations[_module_member(name)],
              f"candidate primitive location mismatch: {name}")
        member = _module_member(name)
        source = locations[member].read_bytes()
        digest = next(digest for path, _, digest in PACKAGE_MEMBERS if path == member)
        _need(hashlib.sha256(source).hexdigest() == digest, f"primitive source changed: {member}")
        # Compile only: never execute a second implementation or mutate globals.
        code = compile(source, function.__code__.co_filename, "exec", dont_inherit=True)
        definitions = [item for item in code.co_consts
                       if isinstance(item, CodeType) and item.co_name == function_name]
        _need(len(definitions) == 1 and function.__code__ == definitions[0]
              and function.__defaults__ is None and function.__kwdefaults__ is None,
              f"loaded candidate primitive code changed: {function_name}")
    return primitives


def _finite(value: torch.Tensor, context: str) -> torch.Tensor:
    _need(type(value) is torch.Tensor and value.device.type == "cpu"
          and value.dtype == torch.float32, f"{context}: require CPU FP32 tensor")
    _need(bool(torch.isfinite(value).all()), f"{context}: nonfinite selected arithmetic")
    return value


def _f32_bytes(value: torch.Tensor) -> bytes:
    return np.array(value.numpy(), dtype="<f4", copy=True, order="C").tobytes(order="C")


def _constants(primitives: _Primitives) -> str:
    matrix = primitives.rgb_module._rgb2xyz
    white = primitives.rgb_module._white
    inverse = primitives.inverse_module._xyz2rgb
    for value in (matrix, white, inverse):
        _finite(value, "conversion constant")
    _need(primitives.inverse_module._rgb2xyz is matrix
          and primitives.inverse_module._white is white, "conversion constant aliases changed")
    expected_matrix = torch.tensor(
        [[0.412453, 0.357580, 0.180423],
         [0.212671, 0.715160, 0.072169],
         [0.019334, 0.119193, 0.950227]], device="cpu", dtype=torch.float32,
    )
    expected_white = torch.tensor([0.95047, 1., 1.08883], device="cpu", dtype=torch.float32)
    _need(torch.equal(matrix, expected_matrix) and torch.equal(white, expected_white)
          and torch.equal(inverse, torch.linalg.inv(expected_matrix)),
          "pinned conversion constants changed")
    records = []
    for name, value, shape in (
        ("rgb2xyz", matrix, (3, 3)), ("white", white, (3,)), ("xyz2rgb", inverse, (3, 3)),
    ):
        _finite(value, f"conversion constant {name}")
        _need(tuple(value.shape) == shape, f"conversion constant geometry: {name}")
        records.append((name, shape, _f32_bytes(value).hex()))
    return hashlib.sha256(_json(records)).hexdigest()


def _image_data(image: Image.Image) -> np.ndarray:
    _need(type(image) is Image.Image, "require materialized base PIL.Image.Image")
    width, height = image.size
    _need(type(width) is int and type(height) is int and width > 0 and height > 0
          and width * height >= 2, "RGB image must have at least two pixels")
    _materialized_image(image, size=(width, height), modes=("RGB",), context="Reinhard RGB")
    data = np.array(image, dtype=np.uint8, copy=True, order="C")
    _need(data.shape == (height, width, 3), "materialized RGB data geometry changed")
    return data


def _working(data: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(data.copy(order="C")).permute(2, 0, 1).to(
        device="cpu", dtype=torch.float32, copy=True,
    )


def _lab_statistics(
    data: np.ndarray, primitives: _Primitives,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    # Normalizer fit/normalize: FP32/255 -> rgb2lab -> lab_split -> tensor(stats).
    lab = _finite(primitives.rgb2lab(_working(data) / 255), "RGB-to-Lab")
    channels = primitives.lab_split(lab)
    _need(len(channels) == 3, "Lab channel count mismatch")
    for channel in channels:
        _finite(channel, "Lab channel")
    statistics = torch.tensor([primitives.get_mean_std(channel) for channel in channels])
    _finite(statistics, "Lab mean/std")
    _need(tuple(statistics.shape) == (3, 2) and bool((statistics[:, 1] >= 0).all()),
          "invalid Lab statistics geometry or negative std")
    return channels, statistics


def _lab2rgb_checked(lab: torch.Tensor, primitives: _Primitives) -> torch.Tensor:
    """Upstream lab2rgb.py:6-35, with selected-path/pre-clamp guards only."""
    lab = _finite(lab.type(torch.float32), "transferred Lab")
    L, a, b = lab[0] / 2.55, lab[1] - 128, lab[2] - 128
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)
    out = _finite(torch.stack([x, y, z], axis=0), "inverse Lab coordinates")
    mask = out > 0.2068966
    not_mask = torch.logical_not(mask)
    out.masked_scatter_(mask, torch.pow(torch.masked_select(out, mask), 3))
    _finite(out, "selected inverse Lab cubic")
    out.masked_scatter_(not_mask, (torch.masked_select(out, not_mask) - 16 / 116) / 7.787)
    _finite(out, "selected inverse Lab linear")
    out = _finite(torch.mul(
        out, primitives.rgb_module._white.type(out.dtype).unsqueeze(dim=-1).unsqueeze(dim=-1),
    ), "XYZ white scaling")
    arr = _finite(torch.tensordot(
        out, torch.t(primitives.inverse_module._xyz2rgb).type(out.dtype), dims=([0], [0]),
    ), "linear RGB before gamma")
    mask = arr > 0.0031308
    not_mask = torch.logical_not(mask)
    arr.masked_scatter_(mask, 1.055 * torch.pow(torch.masked_select(arr, mask), 1 / 2.4) - 0.055)
    _finite(arr, "selected RGB nonlinear gamma")
    arr.masked_scatter_(not_mask, torch.masked_select(arr, not_mask) * 12.92)
    _finite(arr, "selected RGB linear gamma before original clamp")
    return torch.clamp(arr, 0, 1)


def _quantize(rgb: torch.Tensor) -> torch.Tensor:
    scaled = _finite(rgb * 255, "RGB before original uint8 quantization")
    return scaled.type(torch.uint8)


@dataclass(frozen=True)
class ReinhardProvenance:
    distribution_version: str
    module_version: str
    source_commit: str
    source_archive_sha256: str
    source_audit_sha256: str
    correspondence_sha256: str
    package_members: tuple[tuple[str, int, str], ...]
    library_versions: tuple[tuple[str, str], ...]
    device: str
    dtype: str
    backend: str
    method: None
    encoded_template_sha256: str
    encoded_digest_verified_by_adapter: bool
    decoded_pixel_sha256: str
    width: int
    height: int
    target_state_bytes: bytes
    target_state_sha256: str
    conversion_constants_sha256: str


def _state_hash(state: bytes) -> str:
    return hashlib.sha256(_json({
        "dtype": "<f4", "order": "L,a,b means then L,a,b stds", "std_correction": 1,
    }) + b"\n" + state).hexdigest()


@dataclass(frozen=True)
class _Transform:
    target_state: bytes
    target_fingerprint: str
    constants_fingerprint: str
    primitives: _Primitives

    def __call__(self, image: Image.Image) -> Image.Image:
        _execution_context()
        primitives = _candidate()
        _need(primitives.functions == self.primitives.functions, "candidate primitive identity changed")
        _need(_constants(primitives) == self.constants_fingerprint, "conversion constants changed")
        _need(type(self.target_state) is bytes and len(self.target_state) == 24
              and _state_hash(self.target_state) == self.target_fingerprint, "target state changed")
        target = torch.from_numpy(np.frombuffer(self.target_state, dtype="<f4").copy()).reshape(2, 3)
        _finite(target, "retained target statistics")
        _need(bool((target[1] >= 0).all()), "negative retained target std")
        data = _image_data(image)
        channels, statistics = _lab_statistics(data, primitives)
        means, stds = statistics[:, 0], statistics[:, 1]
        _need(bool((stds > 0).all()), "zero source-channel denominator")
        result = []
        for channel, mean, std, target_mean, target_std in zip(
            channels, means, stds, target[0], target[1],
        ):
            standardized = _finite(primitives.standardize(channel, mean, std), "source standardization")
            result.append(_finite(standardized * target_std + target_mean, "ordinary Lab transfer"))
        lab = _finite(primitives.lab_merge(*result), "rebuilt transferred Lab")
        rgb = _lab2rgb_checked(lab, primitives)
        _finite(rgb, "clamped RGB")
        quantized = _quantize(rgb)
        _need(type(quantized) is torch.Tensor and quantized.device.type == "cpu"
              and quantized.dtype == torch.uint8
              and tuple(quantized.shape) == tuple(data.shape), "invalid HWC RGB uint8 output")
        output = Image.fromarray(np.array(quantized.numpy(), dtype=np.uint8, copy=True, order="C"))
        _materialized_image(output, size=(data.shape[1], data.shape[0]),
                            modes=("RGB",), context="Reinhard output")
        return output


def make_reinhard_bridge(
    template: Image.Image, *, template_sha256: str,
) -> tuple[ReinhardBridge, ReinhardProvenance]:
    """Fit once to materialized RGB; retain immutable moments, never the image."""
    versions = _execution_context()
    _need(type(template_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", template_sha256) is not None,
          "encoded-template SHA must be lowercase64 hexadecimal")
    data = _image_data(template)
    primitives = _candidate()
    constants_fingerprint = _constants(primitives)
    _, statistics = _lab_statistics(data, primitives)
    state = _f32_bytes(torch.stack((statistics[:, 0], statistics[:, 1])))
    state_fingerprint = _state_hash(state)
    height, width = data.shape[:2]
    header = _json({"mode": "RGB", "width": width, "height": height})
    pixel_fingerprint = hashlib.sha256(header + b"\n" + data.tobytes(order="C")).hexdigest()
    provenance = ReinhardProvenance(
        "1.4.1", "1.3.0", SOURCE_COMMIT, SOURCE_ARCHIVE_SHA256, SOURCE_AUDIT_SHA256,
        CORRESPONDENCE_SHA256, PACKAGE_MEMBERS, versions, "cpu", "float32", "torch", None,
        template_sha256, False, pixel_fingerprint, width, height,
        state, state_fingerprint, constants_fingerprint,
    )
    transform = _Transform(state, state_fingerprint, constants_fingerprint, primitives)
    return ReinhardBridge(template_sha256, transform), provenance
