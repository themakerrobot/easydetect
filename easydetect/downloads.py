# Apache-2.0
"""Pretrained-weight download and cache — stdlib only (urllib), cached in ~/.easydetect.

``Detector("ed-s")`` has to just work, so the first predict pulls the
OpenVINO IR for that name from the public mirror and keeps it under
``~/.easydetect/`` (override with ``$EASYDETECT_HOME``). The mirror base URL is
``$EASYDETECT_ASSETS_URL``-overridable, which is also how an offline site points
the package at an internal copy.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .errors import DownloadError, ModelNotFoundError

#: Where the released weights live. Layout: ``<base>/<name>/<file>``.
DEFAULT_ASSETS_URL = "https://huggingface.co/leeyunjai/easydetect/resolve/main"

#: easydetect-hub: models trained on public datasets, ``<base>/<dataset>/<model>/<file>``;
#: ``$EASYDETECT_HUB_URL`` points at another copy.
DEFAULT_HUB_URL = "https://huggingface.co/leeyunjai/easydetect-hub/resolve/main"
HUB_PREFIX = "hub:"

#: Names the mirror knows, with the spellings users type mapped onto them.
MODEL_NAMES = ("ed-n", "ed-s", "ed-m", "ed-l", "ed-x")

_TIMEOUT = 30.0


def normalize_name(name: str) -> str:
    """``ed_s`` / ``ED-S`` / ``ed-s.pt`` -> ``ed-s``."""
    stem = str(name).strip().lower()
    for suffix in (".pt", ".xml", ".onnx"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.replace("_", "-")


def is_model_name(name: str) -> bool:
    return normalize_name(name) in MODEL_NAMES


def assets_url() -> str:
    return os.environ.get("EASYDETECT_ASSETS_URL", DEFAULT_ASSETS_URL).rstrip("/")


def hub_url() -> str:
    return os.environ.get("EASYDETECT_HUB_URL", DEFAULT_HUB_URL).rstrip("/")


def is_hub_name(name: str | Path) -> bool:
    return str(name).strip().lower().startswith(HUB_PREFIX)


#: a hub folder named ...-seg / ...-pose: the dataset's detector with the network
#: of that task beside it, trained on the dataset's masks or keypoints
HUB_TASKS = {"-seg": ("segment", "mask_decoder.onnx"), "-pose": ("pose", "pose.onnx")}


def hub_task(folder: str) -> tuple[str, str | None]:
    """``ed-s-640-seg`` -> ``("segment", "mask_decoder.onnx")``; a detector's
    folder -> ``("detect", None)``."""
    return next((v for k, v in HUB_TASKS.items() if folder.endswith(k)), ("detect", None))


def beside(model: str | Path, head: str) -> Path | None:
    """The network a model keeps beside it (``mask_decoder.onnx``,
    ``pose.onnx``): ``<stem>.<head>`` first, the way easydetect hub names it so
    several models share a folder (an 8-bit file shares its float one's), then
    plain ``<head>``, the way training leaves it beside ``best.pt``."""
    path = Path(model)
    if not path.is_file():
        return None
    stem = path.stem
    for name in (f"{stem}.{head}", f"{stem.removesuffix('-int8')}.{head}", head):
        if (path.parent / name).is_file():
            return path.parent / name
    return None


def hub_parts(name: str) -> tuple[str, str, str]:
    """``hub:taco/ed-s-640`` -> ``("taco", "ed-s-640", "taco-ed-s-640.onnx")``:
    the dataset, the model's folder and the file, named after both so files
    from several datasets can sit in one folder. ``-int8`` names the 8-bit
    ONNX (``hub:bccd/ed-s-320-int8``); ``.xml`` the OpenVINO IR, ``.pt`` the
    checkpoint to fine-tune from (needs PyTorch); anything else the ONNX."""
    spec = str(name).strip()[len(HUB_PREFIX):].strip().strip("/")
    dataset, _, model = spec.partition("/")
    suffix = ""
    for ext in (".onnx", ".xml", ".pt"):
        if model.lower().endswith(ext):
            model, suffix = model[: -len(ext)], ext
    int8 = model.endswith("-int8")
    folder = model[: -len("-int8")] if int8 else model
    if not dataset or not folder or "/" in folder or (int8 and suffix not in ("", ".onnx")):
        raise ModelNotFoundError(
            f"{name!r}: a hub model is named hub:<dataset>/<model>, e.g. hub:taco/ed-s-640, "
            "hub:taco/ed-s-640.xml, hub:bccd/ed-s-320-int8. The list: easydetect hub")
    return dataset, folder, f"{dataset}-{model}{suffix or '.onnx'}"


def download_hub(name: str) -> Path:
    """The local copy of a hub model (downloaded once into ``~/.easydetect/hub``);
    an ``.xml`` comes with its ``.bin``, a ``-seg`` / ``-pose`` model with the
    network beside its detector (where task="segment" / "pose" finds it)."""
    dataset, folder, filename = hub_parts(name)
    base, dest = f"{hub_url()}/{dataset}/{folder}", cache_dir() / "hub" / dataset / folder
    try:
        try:
            path = download(f"{base}/{filename}", dest / filename)
        except DownloadError:
            # published before files were named after their dataset
            filename = filename[len(dataset) + 1:]
            path = download(f"{base}/{filename}", dest / filename)
        stem = Path(filename).stem
        if filename.endswith(".xml"):
            download(f"{base}/{stem}.bin", dest / f"{stem}.bin")
            _hub_names(base, dest, stem)
        head = hub_task(folder)[1]
        if head:
            own = f"{stem.removesuffix('-int8')}.{head}"
            try:
                download(f"{base}/{own}", dest / own)
            except DownloadError:
                download(f"{base}/{head}", dest / head)
    except DownloadError as exc:
        raise ModelNotFoundError(
            f"{name!r} is not on easydetect-hub ({exc}). See what is: easydetect hub") from exc
    return path


def _hub_names(base: str, dest: Path, stem: str) -> None:
    """``<stem>.names.json`` beside an IR (an IR holds no class names): the
    hub's own, or else written from the model's model.json."""
    import json

    names = dest / f"{stem}.names.json"
    if names.exists():
        return
    try:
        download(f"{base}/{names.name}", names, progress=False)
    except DownloadError:
        try:
            classes = json.loads(read_url_bytes(f"{base}/model.json")).get("classes", {})
        except (DownloadError, ValueError):
            return                            # the IR's own names, or class_<i>
        names.write_text(json.dumps({int(k): v for k, v in classes.items()},
                                    ensure_ascii=False), encoding="utf-8")


def is_model_url(name: str | Path) -> bool:
    return str(name).strip().lower().startswith(("http://", "https://"))


def model_url_to_path(url: str) -> Path:
    """A model given by its web address, downloaded once into
    ``~/.easydetect/url``: a file (``.onnx``, ``.xml`` with its ``.bin``,
    ``.pt``), its Hugging Face page (``/blob/`` is read as ``/resolve/``), or
    an easydetect-hub model folder (``.../easydetect-hub/tree/main/taco/ed-s-640``)."""
    import hashlib
    from urllib.parse import urlparse

    url = str(url).strip()
    hub_page = DEFAULT_HUB_URL.rsplit("/resolve/", 1)[0]
    for marker in ("/tree/main/", "/resolve/main/", "/blob/main/"):
        prefix = hub_page + marker
        rest = url[len(prefix):].strip("/") if url.startswith(prefix) else ""
        if rest.count("/") == 1 and not Path(rest).suffix:       # a hub model's folder
            return download_hub(HUB_PREFIX + rest)
    url = url.replace("/blob/", "/resolve/", 1)
    filename = Path(urlparse(url).path).name
    if Path(filename).suffix.lower() not in (".onnx", ".xml", ".pt"):
        raise ModelNotFoundError(f"{url}: a model address ends in .onnx, .xml or .pt")
    base = url[: -len(filename)]
    dest = cache_dir() / "url" / hashlib.sha1(base.encode()).hexdigest()[:16]
    try:
        path = download(url, dest / filename)
        if filename.lower().endswith(".xml"):
            stem = filename[: -len(".xml")]
            download(f"{base}{stem}.bin", dest / f"{stem}.bin")
            _hub_names(base.rstrip("/"), dest, stem)
    except DownloadError as exc:
        raise ModelNotFoundError(f"could not fetch the model at {url}: {exc}") from exc
    return path


def hub_index() -> dict:
    """``{"<dataset>/<model>": {...}}``: every model easydetect-hub holds, with
    its classes, size, score and dataset licence. The last copy read is kept
    for when there is no network."""
    import json

    cached = cache_dir() / "hub" / "index.json"
    try:
        data = read_url_bytes(f"{hub_url()}/index.json")
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    except DownloadError:
        if not cached.exists():
            raise
        data = cached.read_bytes()
    return json.loads(data)


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
    """Download ``url`` to ``dest`` (atomically). Existing files are reused.

    A transfer cut short is an error, not a file: what arrived is checked
    against ``Content-Length`` and, where the server publishes one (Hugging
    Face does, for every weights file), against its SHA-256 — so a dropped
    connection cannot leave a truncated model in the cache for good. Each
    process writes its own partial file, so two downloading at once do not
    write into each other's."""
    import hashlib
    import re

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "easydetect"})
    digest = hashlib.sha256()
    linked: dict[str, str] = {}

    class _Remember(urllib.request.HTTPRedirectHandler):
        """Hugging Face states a large file's SHA-256 in ``X-Linked-Etag`` on
        the redirect to its storage; the storage's own ``ETag`` is some other
        hash. Keep the first, never trust the second."""

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            if headers.get("X-Linked-Etag"):
                linked.setdefault("etag", headers["X-Linked-Etag"])
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(_Remember)
    try:
        with opener.open(request, timeout=_TIMEOUT) as response, open(tmp, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            tag = linked.get("etag") or response.headers.get("X-Linked-Etag") or ""
            expected = re.fullmatch(r'(?:W/)?"?([0-9a-f]{64})"?', tag.strip())
            done = 0
            while chunk := response.read(1 << 16):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress and total and sys.stderr.isatty():
                    pct = 100 * done / total
                    print(f"\rdownloading {dest.name} {pct:5.1f}%", end="", file=sys.stderr)
        if progress and sys.stderr.isatty():
            print(f"\rdownloading {dest.name}  done ", file=sys.stderr)
        if total and done != total:
            raise DownloadError(f"{url}: got {done} of {total} bytes; the connection dropped, "
                                "try again")
        if expected and digest.hexdigest() != expected.group(1):
            raise DownloadError(f"{url}: the file arrived damaged (SHA-256 does not match "
                                "the server's); try again")
    except DownloadError:
        tmp.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"could not fetch {url}: {exc}") from exc
    os.replace(tmp, dest)
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


#: COCO fine-tunes at a smaller input, with the queries they were trained
#: with. A run at that size with at most that many queries starts from one:
#: on the classroom datasets it scored higher and steadier (see ROADMAP.md).
SIZED_STARTS = {320: 100}


def download_sized_checkpoint(name: str, imgsz: int, queries: int | None) -> Path | None:
    """``<name>-<imgsz>.pt`` when the mirror has one that fits; None otherwise."""
    name = normalize_name(name)
    trained = SIZED_STARTS.get(int(imgsz))
    if name not in MODEL_NAMES or trained is None or queries is None or queries > trained:
        return None
    return _asset(name, f"{name}-{int(imgsz)}.pt", required=False)


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
        f"You can also pass a path to your own .pt / .xml / .onnx, or a model trained on "
        f"a public dataset: hub:taco/ed-s-640 (the list: easydetect hub)."
    )
