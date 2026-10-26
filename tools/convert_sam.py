#!/usr/bin/env python3
# Apache-2.0
"""Build the segmenter the mirror serves: MobileSAM, as two ONNX files.

    pip install torch timm onnx
    python tools/convert_sam.py --out mirror

MobileSAM (Apache-2.0, https://github.com/ChaoningZhang/MobileSAM) is SAM's
prompt encoder and mask decoder behind a small TinyViT image encoder. Given
the boxes a detector found, it outlines what is inside each one: easydetect's
``task="segment"``. This fetches its code at a pinned commit and its weights
by hash, then writes

    mobile_sam/encoder.onnx   RGB 0-255, 1x3x1024x1024 (longest side 1024,
                              zero-padded bottom/right) -> 1x256x64x64
    mobile_sam/decoder.onnx   embeddings + box corners (labels 2, 3) ->
                              mask logits at the picture's own size
    mobile_sam/decoder.pt     the prompt encoder and mask decoder in PyTorch, for
                              fine-tuning on your own masks
    mobile_sam/LICENSE        MobileSAM's licence, which these files carry
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path

REPO = "https://github.com/ChaoningZhang/MobileSAM"
COMMIT = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
WEIGHTS = f"https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/{COMMIT}/weights/mobile_sam.pt"
WEIGHTS_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"


def fetch(work: Path) -> tuple[Path, Path]:
    code = work / "MobileSAM"
    subprocess.run(["git", "clone", "-q", REPO, str(code)], check=True)
    subprocess.run(["git", "-C", str(code), "checkout", "-q", COMMIT], check=True)
    weights = work / "mobile_sam.pt"
    urllib.request.urlretrieve(WEIGHTS, weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise SystemExit(f"mobile_sam.pt: sha256 {digest}, expected {WEIGHTS_SHA256}")
    return code, weights


def export(code: Path, weights: Path, out: Path) -> None:
    import torch

    sys.path.insert(0, str(code))
    from mobile_sam import sam_model_registry
    from mobile_sam.utils.onnx import SamOnnxModel

    sam = sam_model_registry["vit_t"](checkpoint=str(weights)).eval()

    class Encoder(torch.nn.Module):
        """Normalisation inside the graph, so callers hand it plain RGB."""

        def __init__(self):
            super().__init__()
            self.encoder = sam.image_encoder
            self.register_buffer("mean", torch.tensor(sam.pixel_mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(sam.pixel_std).view(1, 3, 1, 1))

        def forward(self, x):
            return self.encoder((x - self.mean) / self.std)

    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.onnx.export(Encoder().eval(), torch.zeros(1, 3, 1024, 1024),
                          str(out / "encoder.onnx"), input_names=["image"],
                          output_names=["embeddings"], opset_version=17, dynamo=False)
        inputs = {
            "image_embeddings": torch.randn(1, 256, 64, 64),
            "point_coords": torch.tensor([[[100.0, 100.0], [400.0, 300.0]]]),
            "point_labels": torch.tensor([[2.0, 3.0]]),
            "mask_input": torch.zeros(1, 1, 256, 256),
            "has_mask_input": torch.zeros(1),
            "orig_im_size": torch.tensor([480.0, 640.0]),
        }
        torch.onnx.export(
            SamOnnxModel(sam, return_single_mask=True), tuple(inputs.values()),
            str(out / "decoder.onnx"), input_names=list(inputs),
            output_names=["masks", "iou_predictions", "low_res_masks"], opset_version=17,
            dynamo=False,
            dynamic_axes={"point_coords": {0: "boxes", 1: "points"},
                          "point_labels": {0: "boxes", 1: "points"}})
    # the decoder side as PyTorch weights: what fine-tuning on a segmentation
    # dataset of your own starts from (easydetect.nn.sam_decoder)
    torch.save({k: v for k, v in sam.state_dict().items()
                if k.startswith(("prompt_encoder.", "mask_decoder."))}, out / "decoder.pt")
    shutil.copyfile(code / "LICENSE", out / "LICENSE")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="mirror", help="the mirror folder; writes mobile_sam/")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        code, weights = fetch(Path(tmp))
        export(code, weights, Path(args.out) / "mobile_sam")
    written = sorted(p.name for p in (Path(args.out) / "mobile_sam").iterdir())
    print(f"mobile_sam: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
