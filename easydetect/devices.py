# Apache-2.0
"""Which torch device training runs on: an NVIDIA card (CUDA), an Intel GPU
(XPU: Arc, and the Arc graphics in Core Ultra laptops), or the CPU.

PyTorch reaches an Intel GPU through its XPU build
(``pip install torch --index-url https://download.pytorch.org/whl/xpu``) and
the GPU's own driver. Everything here only asks; nothing needs that build.
"""

from __future__ import annotations

import torch

GPU_TYPES = ("cuda", "xpu")


def xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    try:
        return bool(xpu is not None and xpu.is_available())
    except Exception:  # an XPU build without a working driver raises instead of saying no
        return False


def default_device() -> str:
    """``cuda`` when there is an NVIDIA card, ``xpu`` for an Intel GPU, else ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    if xpu_available():
        return "xpu"
    return "cpu"


def is_gpu(device: torch.device) -> bool:
    return device.type in GPU_TYPES


def bf16_supported(device: torch.device) -> bool:
    """Whether autocast can run in bfloat16 here (float32's range: no loss scaling)."""
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    if device.type == "xpu":
        check = getattr(torch.xpu, "is_bf16_supported", None)
        return bool(check()) if check else True     # every Arc GPU has it
    return False


def device_name(device: torch.device) -> str | None:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "xpu":
        return torch.xpu.get_device_name(device.index or 0)
    return None


def memory_gb(device: torch.device) -> int | None:
    if device.type == "cuda":
        return round(torch.cuda.get_device_properties(device).total_memory / 2**30)
    if device.type == "xpu":
        return round(torch.xpu.get_device_properties(device.index or 0).total_memory / 2**30)
    return None


def seed_all(seed: int) -> None:
    torch.cuda.manual_seed_all(seed)
    if xpu_available():
        torch.xpu.manual_seed_all(seed)


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if xpu_available():
        torch.xpu.empty_cache()
