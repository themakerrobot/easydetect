#!/usr/bin/env python3
# Apache-2.0
"""RF-DETR (Roboflow, Apache-2.0 for Nano/Small/Medium) as a file Pibo runs
like an easydetect model: RGB 0..1 in, the ImageNet mean and spread inside,
outputs ``boxes`` (1,Q,4) cx cy w h 0..1 and ``scores`` (1,Q,80) in COCO's
80-class order, the class names in the metadata (``easydetect.names``).

    python -m venv ~/rfdetr && ~/rfdetr/bin/pip install rfdetr onnx pyyaml
    ~/rfdetr/bin/python tools/rfdetr_pibo.py --size 320 --out ~/longrun/pi-kit-rfdetr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: COCO's 80 classes as RF-DETR's 91 category ids
COCO80_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
              24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
              72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
MODELS = {"n": "RFDETRNano", "s": "RFDETRSmall", "m": "RFDETRMedium"}


def for_pibo(src: Path, dst: Path, names: list[str]) -> Path:
    """The exported RF-DETR graph with Pibo's input and outputs."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))
    g = m.graph
    old = g.input[0]
    shape = [d.dim_value for d in old.type.tensor_type.shape.dim]
    g.input.remove(old)
    g.input.insert(0, helper.make_tensor_value_info("images", TensorProto.FLOAT, shape))
    g.initializer.extend([
        numpy_helper.from_array(np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1),
                                "pibo_mean"),
        numpy_helper.from_array(np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1),
                                "pibo_std"),
        numpy_helper.from_array(np.array(COCO80_IDS, np.int64), "pibo_coco80")])
    g.node.insert(0, helper.make_node("Div", ["pibo_centred", "pibo_std"], [old.name]))
    g.node.insert(0, helper.make_node("Sub", ["images", "pibo_mean"], ["pibo_centred"]))
    q = {o.name: o for o in g.output}["dets"].type.tensor_type.shape.dim[1].dim_value
    g.node.extend([helper.make_node("Identity", ["dets"], ["boxes"]),
                   helper.make_node("Gather", ["labels", "pibo_coco80"], ["scores"], axis=2)])
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, q, 4]),
                     helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, q, 80])])
    del m.metadata_props[:]
    prop = m.metadata_props.add()
    prop.key = "easydetect.names"
    prop.value = json.dumps({str(i): n for i, n in enumerate(names)}, ensure_ascii=False)
    onnx.checker.check_model(m)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst))
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="n", choices=sorted(MODELS))
    parser.add_argument("--size", type=int, nargs="+", default=[320],
                        help="input sizes, multiples of 32")
    parser.add_argument("--out", default="pi-kit-rfdetr")
    parser.add_argument("--coco", default="~/datasets/coco/data.yaml",
                        help="COCO's data.yaml: the 80 class names")
    args = parser.parse_args()
    import rfdetr
    import yaml

    names = yaml.safe_load(Path(args.coco).expanduser().read_text())["names"]
    names = [names[i] for i in range(80)] if isinstance(names, dict) else list(names)

    out = Path(args.out).expanduser()
    net = getattr(rfdetr, MODELS[args.model])()
    for size in args.size:
        raw = Path(net.export(output_dir=str(out / f"raw{size}"), shape=(size, size),
                              verbose=False))
        print(for_pibo(raw, out / f"rfdetr-{args.model}-{size}.onnx", names),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
