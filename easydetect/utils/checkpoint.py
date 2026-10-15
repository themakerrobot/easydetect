# Apache-2.0
"""Reading ``.pt`` files without running whatever is inside them."""

from __future__ import annotations

import os
import pickle
from pathlib import Path


def load_checkpoint(path: str | Path, map_location: str = "cpu"):
    """``torch.load`` with ``weights_only=True``: tensors, numbers, strings and
    containers of them — everything easydetect writes — and nothing that can
    run code. A ``.pt`` is a pickle, and a pickle from someone else can do
    anything when loaded; the lab, for one, loads files people upload.

    A file that needs more (written by other code) is refused with a clear
    error; ``$EASYDETECT_TRUST_CHECKPOINTS=1`` loads it anyway, for files
    you trust."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        if os.environ.get("EASYDETECT_TRUST_CHECKPOINTS") == "1":
            return torch.load(path, map_location=map_location, weights_only=False)
        raise ValueError(
            f"{path} holds more than weights, and loading it could run code from it. If it "
            "comes from a source you trust, set EASYDETECT_TRUST_CHECKPOINTS=1 and load it "
            f"again. ({exc})") from exc
