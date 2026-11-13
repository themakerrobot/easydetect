#!/usr/bin/env python3
# Apache-2.0
"""ed-s with its encoder changed, random weights, as 8-bit .onnx files to time
on the robot before training any of them (the time depends on the structure
only). Each is written the way the trained one would be: 100 queries, every
convolution but the stem in 8 bits (easydetect/qat.py).

    python tools/arch_variants.py --sizes 416 448 --out ~/longrun/pi-kit-arch
    # on the robot
    python3 pi_profile.py ~/pi-kit-arch/

Variants (the encoder's settings; ``base`` is ed-s as it is):
  relu        ReLU instead of SiLU: the same weights fit, ONNX Runtime fuses it
  exp34       the FPN/PAN blocks at expansion 0.34 instead of 0.5 (as ed-n)
  relu-exp34  both
  h192        192 channels instead of 256 through the encoder and decoder
"""

from __future__ import annotations

import argparse
import copy

VARIANTS = {
    "base": {},
    "relu": {"act": "relu"},
    "exp34": {"expansion": 0.34},
    "relu-exp34": {"act": "relu", "expansion": 0.34},
    "h192": {"hidden_dim": 192},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[416, 448])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--out", default="pi-kit-arch")
    parser.add_argument("--float", action="store_true", help="the float files too")
    args = parser.parse_args()

    import torch

    import easydetect.nn.ednet as ednet
    from easydetect.exporter import export_onnx
    from easydetect.nn import EDNet
    from easydetect.qat import calibrate, export_qat_onnx, prepare_qat

    names = {i: str(i) for i in range(80)}
    for size in args.sizes:
        for tag in args.variants:
            change = VARIANTS[tag]
            saved = copy.deepcopy(ednet.SIZE_CFG["s"])
            try:
                ednet.SIZE_CFG["s"]["encoder"] = {**saved["encoder"], **change}
                if "hidden_dim" in change:
                    ednet.SIZE_CFG["s"]["decoder"] = {
                        **saved["decoder"], "feat_channels": [change["hidden_dim"]] * 3}
                torch.manual_seed(0)
                net = EDNet("s", 80, pretrained_backbone=False).eval()
            finally:
                ednet.SIZE_CFG["s"] = saved
            net.trim(queries=100)
            key = f"{tag}-{size}"
            if args.float:
                export_onnx(net, names, imgsz=size, out_dir=f"{args.out}/{key}", fname=key,
                            verbose=False)
            q = prepare_qat(copy.deepcopy(net))
            calibrate(q, [torch.rand(2, 3, size, size) for _ in range(2)], percentile=0.9999)
            print(export_qat_onnx(q, names, size, f"{args.out}/{key}-q8/{key}-q8.onnx"),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
