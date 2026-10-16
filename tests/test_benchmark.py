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
