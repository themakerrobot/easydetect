# Apache-2.0
"""The key=value command line."""

from __future__ import annotations

import pytest

from easydetect.cli import HELP, main, parse_args, parse_value


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("true", True),
        ("False", False),
        ("none", None),
        ("100", 100),
        ("0.5", 0.5),
        ("openvino", "openvino"),
        ("0,1,2", [0, 1, 2]),
        ("[0,1]", [0, 1]),
        ("data.yaml", "data.yaml"),
    ],
)
def test_values_are_coerced_the_way_the_command_line_implies(text, expected):
    assert parse_value(text) == expected


def test_the_mode_can_be_positional_or_a_key():
    assert parse_args(["predict", "source=bus.jpg"]) == ("predict", {"source": "bus.jpg"})
    assert parse_args(["mode=val", "data=d.yaml"]) == ("val", {"data": "d.yaml"})


def test_train_style_arguments_land_as_python_types():
    mode, overrides = parse_args(
        ["train", "model=ed-s", "data=data.yaml", "epochs=100", "resume=true", "lr0=1e-4"]
    )
    assert mode == "train"
    assert overrides == {
        "model": "ed-s",
        "data": "data.yaml",
        "epochs": 100,
        "resume": True,
        "lr0": pytest.approx(1e-4),
    }


def test_a_stray_positional_argument_is_a_clear_error():
    with pytest.raises(SystemExit, match="key=value"):
        parse_args(["predict", "bus.jpg"])


def test_help_and_version_exit_cleanly(capsys):
    assert main(["--help"]) == 0
    assert "easydetect predict model=" in capsys.readouterr().out
    assert main(["version"]) == 0
    from easydetect import __version__

    assert capsys.readouterr().out.strip() == __version__


def test_an_unknown_mode_is_rejected_with_the_usage_text(capsys):
    assert main(["detect", "source=bus.jpg"]) == 2
    assert "unknown mode" in capsys.readouterr().err


def test_modes_that_need_data_or_source_say_which_key_is_missing(capsys):
    assert main(["val", "model=ed-s"]) == 2
    assert "data=" in capsys.readouterr().err
    assert main(["predict", "model=ed-s"]) == 2
    assert "source=" in capsys.readouterr().err


def test_the_help_text_documents_every_mode():
    for mode in ("predict", "track", "train", "val", "export"):
        assert mode in HELP
