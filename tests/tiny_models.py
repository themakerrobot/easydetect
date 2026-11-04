# Apache-2.0
"""Stand-ins for the models inference runs, built with the ``onnx`` package
alone: the same inputs and outputs as a detector, a keypoint model and
MobileSAM's encoder and decoder, a few kilobytes each, no torch. Every output
depends on the inputs (times zero), so no runtime folds them away.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OPSET = 17
NAMES = {0: "person", 1: "cat", 2: "cup"}
#: the one box the detector finds: a person in the middle of the picture
BOX = (0.5, 0.5, 0.4, 0.6)


def _save(nodes, inputs, outputs, inits, path: Path, meta: dict | None = None) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def vi(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    graph = helper.make_graph(nodes, path.stem, [vi(*i) for i in inputs],
                              [vi(*o) for o in outputs],
                              [numpy_helper.from_array(np.asarray(v, np.float32), k)
                               for k, v in inits.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    for k, v in (meta or {}).items():
        model.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _zero_of(node_in: str, out: str, axes):
    """``out`` = 0 × the mean of ``node_in`` over ``axes``, kept as dims of 1."""
    from onnx import helper

    return [helper.make_node("ReduceMean", [node_in], [out + "_m"], axes=list(axes), keepdims=1),
            helper.make_node("Mul", [out + "_m", "zero"], [out])]


def detector(path: Path, size: int = 64, queries: int = 10) -> Path:
    """``images (1,3,size,size)`` → boxes ``(1,Q,4)`` cxcywh 0..1, logits ``(1,Q,3)``."""
    from onnx import helper

    boxes = np.tile(np.array([[0.2, 0.2, 0.1, 0.1]], np.float32), (queries, 1))
    boxes[0] = BOX
    logits = np.full((queries, len(NAMES)), -6.0, np.float32)
    logits[0, 0] = 4.0
    nodes = _zero_of("images", "z4", (1, 2, 3)) + [
        helper.make_node("Reshape", ["z4", "shape3"], ["z"]),
        helper.make_node("Add", ["z", "boxes_c"], ["boxes"]),
        helper.make_node("Add", ["z", "logits_c"], ["logits"]),
    ]
    model = _save(nodes, [("images", [1, 3, size, size])],
                  [("logits", [1, queries, len(NAMES)]), ("boxes", [1, queries, 4])],
                  {"zero": 0.0, "shape3": np.zeros(3), "boxes_c": boxes[None],
                   "logits_c": logits[None]},
                  path, {"easydetect.names": json.dumps({str(k): v for k, v in NAMES.items()}),
                         "easydetect.conf": "0.5"})
    return _int64_axes(model, {"shape3": [1, 1, 1]})


def keypoints(path: Path, k: int = 17) -> Path:
    """``image (N,3,256,192)`` → x and y logits ``(N,K,384)``, ``(N,K,512)``: each
    keypoint peaks at the middle of the crop."""
    from onnx import helper

    x = np.full((1, k, 384), -4.0, np.float32)
    y = np.full((1, k, 512), -4.0, np.float32)
    x[..., 192], y[..., 256] = 8.0, 8.0
    nodes = _zero_of("image", "z4", (1, 2, 3)) + [
        helper.make_node("Squeeze", ["z4", "axes3"], ["z3"]),         # (N, 1, 1)
        helper.make_node("Add", ["z3", "x_c"], ["x"]),
        helper.make_node("Add", ["z3", "y_c"], ["y"]),
    ]
    model = _save(nodes, [("image", ["N", 3, 256, 192])], [("x", ["N", k, 384]),
                                                          ("y", ["N", k, 512])],
                  {"zero": 0.0, "x_c": x, "y_c": y, "axes3": np.zeros(1)}, path)
    return _int64_axes(model, {"axes3": [3]})


def _int64_axes(path: Path, axes: dict[str, list[int]]) -> Path:
    """Squeeze/Unsqueeze take int64 axes: swap those initializers in."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    keep = [t for t in model.graph.initializer if t.name not in axes]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    for name, value in axes.items():
        model.graph.initializer.append(numpy_helper.from_array(np.array(value, np.int64), name))
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def sam_encoder(path: Path) -> Path:
    """``image (1,3,1024,1024)`` → ``(1,256,64,64)`` embeddings."""
    from onnx import helper

    nodes = _zero_of("image", "z4", (1, 2, 3)) + [
        helper.make_node("Expand", ["z4", "emb_shape"], ["image_embeddings"])]
    model = _save(nodes, [("image", [1, 3, 1024, 1024])],
                  [("image_embeddings", [1, 256, 64, 64])],
                  {"zero": 0.0, "emb_shape": np.zeros(4)}, path)
    return _int64_axes(model, {"emb_shape": [1, 256, 64, 64]})


def sam_decoder(path: Path) -> Path:
    """MobileSAM's decoder inputs → ``masks``, ``iou_predictions (N,1)`` and
    ``low_res_masks (N,1,256,256)``: positive over the middle half."""
    from onnx import helper

    low = np.full((1, 1, 256, 256), -5.0, np.float32)
    low[..., 64:192, 64:192] = 5.0
    zeros = []
    for name, axes in (("image_embeddings", (0, 1, 2, 3)), ("mask_input", (0, 1, 2, 3)),
                       ("has_mask_input", (0,)), ("orig_im_size", (0,))):
        zeros += _zero_of(name, f"z_{name}", axes)
    nodes = zeros + _zero_of("point_coords", "zc", (1, 2)) + _zero_of("point_labels", "zl", (1,))
    nodes += [
        helper.make_node("Unsqueeze", ["zc", "axes3"], ["zc4"]),       # (N, 1, 1, 1)
        helper.make_node("Unsqueeze", ["zl", "axes23"], ["zl4"]),      # (N, 1, 1, 1)
        helper.make_node("Add", ["zc4", "zl4"], ["zn"]),
        helper.make_node("Sum", ["zn", "z_image_embeddings", "z_mask_input"], ["zall"]),
        helper.make_node("Add", ["zall", "low_c"], ["low_res_masks"]),
        helper.make_node("Identity", ["low_res_masks"], ["masks"]),
        helper.make_node("Squeeze", ["zn", "axes23"], ["zn2"]),        # (N, 1)
        helper.make_node("Sum", ["zn2", "z_has_mask_input", "z_orig_im_size", "iou_c"],
                         ["iou_predictions"]),
    ]
    model = _save(nodes,
                  [("image_embeddings", [1, 256, 64, 64]), ("point_coords", ["N", 2, 2]),
                   ("point_labels", ["N", 2]), ("mask_input", [1, 1, 256, 256]),
                   ("has_mask_input", [1]), ("orig_im_size", [2])],
                  [("masks", ["N", 1, 256, 256]), ("iou_predictions", ["N", 1]),
                   ("low_res_masks", ["N", 1, 256, 256])],
                  {"zero": 0.0, "low_c": low, "iou_c": np.array([[0.9]], np.float32),
                   "axes3": np.zeros(1), "axes23": np.zeros(2)}, path)
    return _int64_axes(model, {"axes3": [3], "axes23": [2, 3]})


def build(folder: Path) -> dict[str, Path]:
    """All of them in ``folder``, and the detector as an OpenVINO IR too."""
    import openvino as ov

    folder.mkdir(parents=True, exist_ok=True)
    onnx_path = detector(folder / "tiny.onnx")
    ir = folder / "ir" / "tiny.xml"
    ir.parent.mkdir(exist_ok=True)
    ov.save_model(ov.convert_model(str(onnx_path)), str(ir), compress_to_fp16=False)
    (ir.parent / "labels.txt").write_text("\n".join(NAMES.values()) + "\n")
    return {"onnx": onnx_path, "xml": ir, "pose": keypoints(folder / "pose.onnx"),
            "encoder": sam_encoder(folder / "encoder.onnx"),
            "decoder": sam_decoder(folder / "decoder.onnx")}
