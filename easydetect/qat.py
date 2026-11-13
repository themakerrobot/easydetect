# Apache-2.0
"""Quantization-aware training: an 8-bit network that keeps its accuracy.

Post-training 8 bits cost ed-s about 0.03 COCO mAP50-95; training a little
with the rounding in place wins most of it back. The convolutions train on
the weights they will run with (each BatchNorm folded in first), their inputs
rounded to uint8 per tensor and their weights to int8 per output channel, the
way ONNX Runtime's 8-bit convolutions take them. The export writes those
roundings as QuantizeLinear/DequantizeLinear pairs (QDQ) around float
convolutions, which ONNX Runtime runs as 8-bit ones.

    net = prepare_qat(detector.net)              # in place: folded, rounding
    calibrate(net, batches)                      # input ranges from pictures
    ... train a few epochs at a low learning rate ...
    export_qat_onnx(net, names, 416, "out/model-qat.onnx")

The backbone's stem stays in float (8 bits there cost the most), and so does
everything that is not a convolution: the decoder's attention and box
decoding, which take little of the time.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

#: modules left in float: the first layers see the raw picture
SKIP = ("backbone.stem",)
#: how fast the input range follows new pictures while training
MOMENTUM = 0.01


def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """``bn(conv(x))`` as one convolution (BatchNorm in eval form)."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True)
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    fused.weight.data = conv.weight.data * t.reshape(-1, 1, 1, 1)
    bias = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias.data = bn.bias.data + (bias - bn.running_mean) * t
    return fused.to(conv.weight.device)


def fold_batchnorm(net: nn.Module) -> nn.Module:
    """Every convolution followed by a BatchNorm merged with it (in place);
    the re-parameterised blocks of the encoder fused as for deployment."""
    from .nn.encoder import ConvNormLayer_fuse, VGGBlock

    for m in list(net.modules()):                 # the blocks that fuse themselves
        if isinstance(m, VGGBlock) and hasattr(m, "conv1"):
            m.convert_to_deploy()
    for m in list(net.modules()):
        if isinstance(m, ConvNormLayer_fuse) and hasattr(m, "conv"):
            m.convert_to_deploy()
    for m in list(net.modules()):
        kids = dict(m.named_children())
        for conv_name, bn_name in (("conv", "bn"), ("conv", "norm")):
            conv, bn = kids.get(conv_name), kids.get(bn_name)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(m, conv_name, _fold(conv, bn))
                setattr(m, bn_name, nn.Identity())
    return net


class QuantConv2d(nn.Module):
    """A convolution that rounds its input to uint8 and its weight to int8."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.stride, self.padding = conv.stride, conv.padding
        self.dilation, self.groups = conv.dilation, conv.groups
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(0.0))
        self.register_buffer("out_lo", torch.tensor(0.0))     # the output's range: ONNX
        self.register_buffer("out_hi", torch.tensor(0.0))     # Runtime fuses DQ-Conv-Q only
        self.observing = True                       # follow the input range in training
        self.fresh = True                           # no range yet: the next one sets it
        self.calibrating = False                    # follow it in eval mode too
        self.frozen: dict | None = None             # constants for the export

    # the input: uint8, one scale for the tensor (as tensors: no GPU sync)
    @staticmethod
    def _qparams(lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = lo.clamp(max=0.0), hi.clamp(min=0.0)
        scale = ((hi - lo) / 255.0).clamp(min=1e-8)
        return scale, (-lo / scale).round().clamp(0, 255).to(torch.int32)

    def _act_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._qparams(self.lo, self.hi)

    def _track(self, lo_buf, hi_buf, t: torch.Tensor, first: bool) -> None:
        lo, hi = t.detach().amin().float(), t.detach().amax().float()
        if first:
            lo_buf.copy_(lo)
            hi_buf.copy_(hi)
        else:
            lo_buf.lerp_(lo, MOMENTUM)
            hi_buf.lerp_(hi, MOMENTUM)

    # the weight: int8, symmetric, one scale an output channel
    def _weight_scale(self) -> torch.Tensor:
        return (self.weight.detach().abs().amax(dim=(1, 2, 3)) / 127.0).clamp(min=1e-8).float()

    def freeze(self) -> None:
        """Fix the scales as constants (for the export)."""
        scale, zp = self._act_qparams()
        out_scale, out_zp = self._qparams(self.out_lo, self.out_hi)
        self.frozen = {"scale": float(scale), "zp": int(zp),
                       "out_scale": float(out_scale), "out_zp": int(out_zp),
                       "w_scale": self._weight_scale().cpu(),
                       "w_zp": torch.zeros(self.weight.shape[0], dtype=torch.int32)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        observe = self.observing and (self.training or self.calibrating) or self.fresh
        first = self.fresh
        if observe:
            self._track(self.lo, self.hi, x, first)
        dtype = x.dtype
        x = x if dtype == torch.float32 else x.float()
        w = self.weight if self.weight.dtype == torch.float32 else self.weight.float()
        if self.frozen is not None:
            f = self.frozen
            xq = torch.fake_quantize_per_tensor_affine(x, f["scale"], f["zp"], 0, 255)
            wq = torch.fake_quantize_per_channel_affine(
                w, f["w_scale"].to(x.device), f["w_zp"].to(x.device), 0, -128, 127)
        else:
            scale, zp = self._act_qparams()
            xq = torch.fake_quantize_per_tensor_affine(x, scale, zp, 0, 255)
            ws = self._weight_scale()
            wq = torch.fake_quantize_per_channel_affine(
                w, ws, torch.zeros_like(ws, dtype=torch.int32), 0, -128, 127)
        bias = self.bias if self.bias is None or self.bias.dtype == torch.float32 \
            else self.bias.float()
        y = F.conv2d(xq, wq, bias, self.stride, self.padding, self.dilation, self.groups)
        if observe:
            self._track(self.out_lo, self.out_hi, y, first)
            self.fresh = False
        if self.frozen is not None:
            y = torch.fake_quantize_per_tensor_affine(
                y, self.frozen["out_scale"], self.frozen["out_zp"], 0, 255)
        else:
            out_scale, out_zp = self._qparams(self.out_lo, self.out_hi)
            y = torch.fake_quantize_per_tensor_affine(y, out_scale, out_zp, 0, 255)
        return y.to(dtype) if dtype in (torch.float16, torch.bfloat16) else y


def prepare_qat(net: nn.Module, skip: tuple[str, ...] = SKIP) -> nn.Module:
    """BatchNorms folded, then every convolution outside ``skip`` rounding (in place)."""
    fold_batchnorm(net)
    for name, m in list(net.named_modules()):
        for child_name, child in list(m.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Conv2d) and not any(full.startswith(s) for s in skip):
                setattr(m, child_name, QuantConv2d(child))
    net.qat = True
    return net


def quant_convs(net: nn.Module) -> list[QuantConv2d]:
    return [m for m in net.modules() if isinstance(m, QuantConv2d)]


def mark_calibrated(net: nn.Module) -> None:
    """After loading trained ranges: keep them (the next picture does not reset them)."""
    for c in quant_convs(net):
        c.fresh = False


@torch.no_grad()
def calibrate(net: nn.Module, batches, device="cpu", steps: int | None = None) -> None:
    """Input ranges from ``batches`` of pictures (B,3,H,W, 0..1)."""
    convs = quant_convs(net)
    for c in convs:
        c.fresh, c.calibrating = True, True
    was = net.training
    net.eval()                                      # no targets: the inference path
    try:
        for i, imgs in enumerate(batches):
            if steps is not None and i >= steps:
                break
            net(imgs.to(device))
    finally:
        for c in convs:
            c.calibrating = False
        net.train(was)


def export_qat_onnx(net: nn.Module, names: dict, imgsz: int, out: str | Path,
                    queries: int | None = None) -> Path:
    """The trained 8-bit network as a QDQ .onnx (opset 17), names inside."""
    import warnings

    from .exporter import _EXPORT_LOCK, _embed_names, _write_labels
    from .nn import DeployWrapper

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    deployed = copy.deepcopy(net).cpu().eval()
    if queries is not None:
        deployed.trim(queries=queries)
    for name, m in deployed.named_modules():        # the decoder's training heads go
        if name.startswith("decoder") and hasattr(m, "convert_to_deploy"):
            m.convert_to_deploy()
    for c in quant_convs(deployed):
        c.freeze()
    wrapper = DeployWrapper(deployed).eval()
    with _EXPORT_LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(wrapper, torch.zeros(1, 3, imgsz, imgsz), str(out),
                          input_names=["images"], output_names=["boxes", "scores"],
                          opset_version=17, dynamo=False)
        table = _write_labels(out.parent, out.stem, names)
        _embed_names(out, table)
    _store_int8_weights(out)
    return out


def _store_int8_weights(path: Path) -> None:
    """Each weight's QuantizeLinear done ahead: the int8 weight stored, only its
    DequantizeLinear left, the form ONNX Runtime fuses into 8-bit convolutions."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    consts = {n.output[0]: n for n in g.node if n.op_type == "Constant"}

    def value(name):
        if name in inits:
            return numpy_helper.to_array(inits[name])
        if name in consts:
            return numpy_helper.to_array(consts[name].attribute[0].t)
        return None

    dropped = []
    for n in g.node:
        if n.op_type != "QuantizeLinear":
            continue
        w, scale, zp = (value(name) for name in n.input[:3])
        if w is None or scale is None or zp is None or w.ndim != 4:
            continue                                  # an activation, quantized at run time
        axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
        shape = [1] * w.ndim
        shape[axis] = -1
        q = np.clip(np.round(w / scale.reshape(shape)) + zp.reshape(shape).astype(np.int32),
                    np.iinfo(zp.dtype).min, np.iinfo(zp.dtype).max).astype(zp.dtype)
        name = n.output[0] + "_int8"
        g.initializer.append(numpy_helper.from_array(q, name))
        for m in g.node:                              # its DequantizeLinear reads the int8 weight
            m.input[:] = [name if i == n.output[0] else i for i in m.input]
        dropped.append(n)
    for n in dropped:
        g.node.remove(n)
    used = {i for n in g.node for i in n.input}
    for init in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(init)
    onnx.save(model, str(path))


__all__ = ["QuantConv2d", "calibrate", "export_qat_onnx", "fold_batchnorm", "prepare_qat"]
