# Apache-2.0
"""hub:<dataset>/<model> names: which file each means, where it comes from and goes."""

from __future__ import annotations

import json

import pytest

from easydetect import downloads
from easydetect.errors import DownloadError, ModelNotFoundError


@pytest.mark.parametrize("name, parts", [
    ("hub:taco/ed-s-640", ("taco", "ed-s-640", "ed-s-640.onnx")),
    ("hub:taco/ed-s-640.onnx", ("taco", "ed-s-640", "ed-s-640.onnx")),
    ("hub:taco/ed-s-640.xml", ("taco", "ed-s-640", "ed-s-640.xml")),
    ("hub:taco/ed-s-640.pt", ("taco", "ed-s-640", "ed-s-640.pt")),
    ("hub:bccd/ed-s-320-int8", ("bccd", "ed-s-320", "ed-s-320-int8.onnx")),
    (" HUB:rf100-marbles/ed-s-320/ ", ("rf100-marbles", "ed-s-320", "ed-s-320.onnx")),
])
def test_a_hub_name_says_which_file(name, parts):
    assert downloads.is_hub_name(name)
    assert downloads.hub_parts(name) == parts


@pytest.mark.parametrize("name", ["hub:", "hub:taco", "hub:taco/", "hub:bccd/ed-s-320-int8.xml",
                                  "hub:a/b/c"])
def test_a_malformed_hub_name_says_how_one_looks(name):
    with pytest.raises(ModelNotFoundError, match="hub:<dataset>/<model>"):
        downloads.hub_parts(name)


def test_a_hub_model_is_fetched_once_into_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path))
    monkeypatch.setenv("EASYDETECT_HUB_URL", "https://example.org/hub")
    fetched = []

    def fake(url, dest, progress=True):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(downloads, "download", fake)
    path = downloads.download_hub("hub:bccd/ed-s-320-int8")
    assert path == tmp_path / "hub" / "bccd" / "ed-s-320" / "ed-s-320-int8.onnx"
    assert fetched == ["https://example.org/hub/bccd/ed-s-320/ed-s-320-int8.onnx"]

    fetched.clear()
    xml = downloads.download_hub("hub:bccd/ed-s-640.xml")
    assert xml.name == "ed-s-640.xml"
    assert fetched == [f"https://example.org/hub/bccd/ed-s-640/ed-s-640.{ext}"
                       for ext in ("xml", "bin", "names.json")]


def test_an_ir_without_its_names_file_gets_them_from_model_json(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path))

    def fake(url, dest, progress=True):
        if url.endswith(".names.json"):
            raise DownloadError("HTTP 404")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(downloads, "download", fake)
    monkeypatch.setattr(downloads, "read_url_bytes",
                        lambda url, timeout=30: json.dumps({"classes": {"0": "RBC"}}).encode())
    xml = downloads.download_hub("hub:bccd/ed-s-640.xml")
    assert json.loads(xml.with_suffix(".names.json").read_text()) == {"0": "RBC"}


def test_a_name_missing_from_the_hub_says_where_the_list_is(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path))
    monkeypatch.setattr(downloads, "download",
                        lambda *a, **k: (_ for _ in ()).throw(DownloadError("HTTP 404")))
    with pytest.raises(ModelNotFoundError, match="easydetect hub"):
        downloads.download_hub("hub:nothing/ed-s-640")


def test_the_hub_list_is_kept_for_when_there_is_no_network(tmp_path, monkeypatch, capsys):
    from easydetect.cli import main

    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path))
    index = {"bccd/ed-s-320": {"title": "BCCD (blood cells)", "category": "Medical",
                               "classes": {"0": "a"}, "map50_95": 0.63, "map50_95_int8": 0.629},
             "taco/ed-s-640": {"title": "TACO", "classes": {}, "map50_95": 0.258}}
    monkeypatch.setattr(downloads, "read_url_bytes",
                        lambda url, timeout=30: json.dumps(index).encode())
    assert main(["hub", "search=blood"]) == 0
    out = capsys.readouterr().out
    assert "hub:bccd/ed-s-320" in out and "taco" not in out

    monkeypatch.setattr(downloads, "read_url_bytes",
                        lambda *a, **k: (_ for _ in ()).throw(DownloadError("offline")))
    assert downloads.hub_index() == index
    assert main(["hub", "search=nothing-like-it"]) == 1


def test_a_model_address_is_fetched_once(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path))
    fetched = []

    def fake(url, dest, progress=True):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(downloads, "download", fake)
    page = "https://huggingface.co/someone/models/blob/main/x/best.onnx"
    path = downloads.model_url_to_path(page)
    assert path.name == "best.onnx" and path.parent.parent == tmp_path / "url"
    assert fetched == ["https://huggingface.co/someone/models/resolve/main/x/best.onnx"]

    fetched.clear()      # a hub model's folder is that hub model
    hub = "https://huggingface.co/leeyunjai/easydetect-hub/tree/main/taco/ed-s-640"
    assert downloads.model_url_to_path(hub) == downloads.download_hub("hub:taco/ed-s-640")

    with pytest.raises(ModelNotFoundError, match=r"\.onnx, \.xml or \.pt"):
        downloads.model_url_to_path("https://example.org/model.zip")


def test_a_mask_or_keypoint_model_comes_with_its_network(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path))
    monkeypatch.setenv("EASYDETECT_HUB_URL", "https://example.org/hub")
    fetched = []

    def fake(url, dest, progress=True):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(downloads, "download", fake)
    assert downloads.hub_parts("hub:coco/ed-s-320-seg-int8") == \
        ("coco", "ed-s-320-seg", "ed-s-320-seg-int8.onnx")
    path = downloads.download_hub("hub:coco/ed-s-320-seg-int8")
    assert (path.parent / "mask_decoder.onnx").exists()
    assert fetched[-1] == "https://example.org/hub/coco/ed-s-320-seg/mask_decoder.onnx"
    downloads.download_hub("hub:coco/ed-s-640-pose")
    assert fetched[-1] == "https://example.org/hub/coco/ed-s-640-pose/pose.onnx"
    assert downloads.hub_task("ed-s-640-pose") == ("pose", "pose.onnx")
    assert downloads.hub_task("ed-s-640") == ("detect", None)
