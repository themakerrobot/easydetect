# Apache-2.0
"""easydetect benchmark: every model, every runtime that reads it, one table."""

from __future__ import annotations


def test_benchmark_times_each_model_on_each_runtime(tiny_ir, capsys):
    from easydetect.benchmark import benchmark

    rows = benchmark(tiny_ir.parent, runs=2, warmup=1, threads=2)
    assert rows and all("error" not in r for r in rows)
    assert {r["backend"] for r in rows} >= {"openvino"}
    assert all(r["total"] > 0 and r["fps"] > 0 for r in rows)
    assert "fps" in capsys.readouterr().out


def test_the_command_line_needs_a_model():
    from easydetect import cli

    assert cli.main(["benchmark"]) == 2


def test_a_folder_scan_skips_helper_models_and_survives_one_that_is_no_detector(
        tiny_ir, tmp_path, monkeypatch):
    import shutil

    from easydetect import benchmark as bench
    from easydetect.predictor import OVPredictor

    folder = tmp_path / "models"
    shutil.copytree(tiny_ir.parent, folder)
    (folder / "pose.onnx").write_bytes(b"not read")           # beside a pose model
    (folder / "mask_decoder.onnx").write_bytes(b"not read")
    shutil.copy(folder / "tiny.xml", folder / "odd.xml")
    shutil.copy(folder / "tiny.bin", folder / "odd.bin")
    real = OVPredictor.__call__

    def odd_fails(self, img, **kw):
        if self.model_path.stem == "odd":
            raise RuntimeError("Exception from src/x.cpp:1:\nshape mismatch")
        return real(self, img, **kw)

    monkeypatch.setattr(OVPredictor, "__call__", odd_fails)
    rows = bench.benchmark(folder, runs=1, warmup=1, backend="openvino", verbose=False)
    names = {r["model"].split("/")[-1]: r for r in rows}
    assert "pose.onnx" not in names and "mask_decoder.onnx" not in names
    assert names["odd.xml"]["error"] == "shape mismatch"
    assert "error" not in names["tiny.xml"]
