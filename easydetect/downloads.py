# Apache-2.0
"""Pretrained-weight download and cache — stdlib only (urllib), cached in ~/.easydetect.

``Detector("dfine-s")`` has to just work, so the first predict pulls the
OpenVINO IR for that name from the public mirror and keeps it under
``~/.easydetect/`` (override with ``$EASYDETECT_HOME``). The mirror base URL is
``$EASYDETECT_ASSETS_URL``-overridable, which is also how an offline site points
the package at an internal copy.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .errors import DownloadError, ModelNotFoundError

#: Where the released weights live. Layout: ``<base>/<name>/<file>``.
DEFAULT_ASSETS_URL = "https://huggingface.co/leeyunjai/easydetect/resolve/main"

#: Names the mirror knows, with the spellings users type mapped onto them.
MODEL_NAMES = ("dfine-n", "dfine-s", "dfine-m", "dfine-l", "dfine-x")

_TIMEOUT = 30.0


def normalize_name(name: str) -> str:
    """``dfine_s`` / ``D-FINE-S`` / ``DFINE-S`` / ``dfine-s.pt`` -> ``dfine-s``."""
    stem = str(name).strip().lower()
    for suffix in (".pt", ".xml", ".onnx"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    stem = stem.replace("_", "-")
    return "dfine-" + stem[len("d-fine-"):] if stem.startswith("d-fine-") else stem


def is_model_name(name: str) -> bool:
    return normalize_name(name) in MODEL_NAMES


def assets_url() -> str:
    return os.environ.get("EASYDETECT_ASSETS_URL", DEFAULT_ASSETS_URL).rstrip("/")


def cache_dir() -> Path:
    root = Path(os.environ.get("EASYDETECT_HOME", Path.home() / ".easydetect")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def read_url_bytes(url: str, timeout: float = _TIMEOUT) -> bytes:
    """Fetch a URL into memory (used for image URLs as well as weights)."""
    request = urllib.request.Request(url, headers={"User-Agent": "easydetect"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise DownloadError(f"could not fetch {url}: {exc}") from exc


def download(url: str, dest: Path, progress: bool = True) -> Path:
    """Download ``url`` to ``dest`` (atomically). Existing files are reused."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "easydetect"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response, open(tmp, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 16):
                out.write(chunk)
                done += len(chunk)
                if progress and total and sys.stderr.isatty():
                    pct = 100 * done / total
                    print(f"\rdownloading {dest.name} {pct:5.1f}%", end="", file=sys.stderr)
        if progress and sys.stderr.isatty():
            print(f"\rdownloading {dest.name}  done ", file=sys.stderr)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"could not fetch {url}: {exc}") from exc
    shutil.move(str(tmp), str(dest))
    return dest


def _asset(name: str, filename: str, required: bool = True) -> Path | None:
    dest = cache_dir() / name / filename
    if dest.exists() and dest.stat().st_size:
        return dest
    url = f"{assets_url()}/{name}/{filename}"
    try:
        return download(url, dest)
    except DownloadError:
        if required:
            raise
        return None


def download_ir(name: str) -> Path:
    """Fetch ``<name>.xml`` (+ ``.bin``, + ``labels.txt``); returns the .xml path."""
    name = normalize_name(name)
    if name not in MODEL_NAMES:
        raise ModelNotFoundError(_unknown_name_message(name))
    try:
        xml = _asset(name, f"{name}.xml")
        _asset(name, f"{name}.bin")
    except DownloadError as exc:
        raise ModelNotFoundError(
            f"'{name}' is not on the mirror yet ({exc}). Pass a .pt/.xml path, "
            f"point $EASYDETECT_ASSETS_URL at your own copy, or train it yourself: "
            f"Detector('{name}').train(data='data.yaml')"
        ) from exc
    _asset(name, "labels.txt", required=False)
    return xml


def download_onnx(name: str) -> Path:
    """Fetch ``<name>.onnx`` (+ ``labels.txt``) — what ONNX Runtime runs."""
    name = normalize_name(name)
    if name not in MODEL_NAMES:
        raise ModelNotFoundError(_unknown_name_message(name))
    try:
        onnx = _asset(name, f"{name}.onnx")
    except DownloadError as exc:
        raise ModelNotFoundError(
            f"'{name}'.onnx is not on the mirror ({exc}). Install openvino to use the IR, "
            f"or export one: Detector('{name}').export(format='onnx')"
        ) from exc
    _asset(name, "labels.txt", required=False)
    return onnx


def download_checkpoint(name: str) -> Path:
    """Fetch ``<name>.pt`` — the torch weights used to fine-tune or validate."""
    name = normalize_name(name)
    if name not in MODEL_NAMES:
        raise ModelNotFoundError(_unknown_name_message(name))
    try:
        return _asset(name, f"{name}.pt")
    except DownloadError as exc:
        raise ModelNotFoundError(
            f"no pretrained checkpoint for '{name}' on the mirror ({exc}). "
            f"Training starts from an ImageNet backbone instead: "
            f"Detector('{name}').train(data='data.yaml')"
        ) from exc


def download_segmenter() -> tuple[Path, Path]:
    """Fetch MobileSAM's ``encoder.onnx`` and ``decoder.onnx`` (task="segment")."""
    try:
        return _asset("mobile_sam", "encoder.onnx"), _asset("mobile_sam", "decoder.onnx")
    except DownloadError as exc:
        raise ModelNotFoundError(
            f"the segmenter (mobile_sam/) is not on the mirror yet ({exc}). Build it with "
            f"tools/convert_sam.py and point $EASYDETECT_ASSETS_URL at it, or put its two files "
            f"in {cache_dir() / 'mobile_sam'}"
        ) from exc


#: MobileSAM's release, pinned and checksummed: where the decoder's PyTorch
#: weights come from until the mirror has mobile_sam/decoder.pt
MOBILE_SAM_PT = ("https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/"
                 "f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt",
                 "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f")


def download_sam_decoder() -> Path:
    """MobileSAM's prompt encoder and mask decoder as PyTorch weights
    (``mobile_sam/decoder.pt``) — what fine-tuning the segmenter starts from."""
    try:
        return _asset("mobile_sam", "decoder.pt")
    except DownloadError:
        pass
    import hashlib

    import torch

    dest = cache_dir() / "mobile_sam" / "decoder.pt"
    full = download(MOBILE_SAM_PT[0], cache_dir() / "mobile_sam" / "mobile_sam.pt")
    digest = hashlib.sha256(full.read_bytes()).hexdigest()
    if digest != MOBILE_SAM_PT[1]:
        full.unlink()
        raise DownloadError(f"mobile_sam.pt: sha256 {digest}, expected {MOBILE_SAM_PT[1]}")
    state = torch.load(full, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in state.items()
                if k.startswith(("prompt_encoder.", "mask_decoder."))}, dest)
    full.unlink()  # the image encoder's weights are not needed: its ONNX file runs it
    return dest


def download_pose(size: str = "s") -> Path:
    """Fetch easydetect's keypoint model, ``pose/pose-<size>.onnx`` (task="pose")."""
    try:
        return _asset("pose", f"pose-{size}.onnx")
    except DownloadError as exc:
        raise ModelNotFoundError(
            f"the keypoint model (pose/pose-{size}.onnx) is not on the mirror yet ({exc}). "
            f"Train one with tools/train_pose.py and put its .onnx in {cache_dir() / 'pose'}"
        ) from exc


def _unknown_name_message(name: str) -> str:
    return (
        f"unknown model '{name}'. Known names: {', '.join(MODEL_NAMES)}. "
        f"You can also pass a path to your own .pt / .xml / .onnx."
    )
