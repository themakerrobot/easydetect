# Apache-2.0
"""YOLO-format datasets as they actually arrive: every layout people export."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

from easydetect.data.dataset import DetDataset, label_row_to_box  # noqa: E402


def _image(path, size=(40, 60)):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((*size, 3), 90, np.uint8))


def _label(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_the_standard_layout(tmp_path):
    for split in ("train", "val"):
        _image(tmp_path / "images" / split / "a.jpg")
        _label(tmp_path / "labels" / split / "a.txt", "0 .5 .5 .2 .2\n")
    (tmp_path / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: {0: can}\n"
    )
    ds = DetDataset(tmp_path / "data.yaml", "val", imgsz=64, augment=False)
    assert len(ds) == 1 and len(ds._load_labels(ds.files[0])) == 1


def test_a_split_per_folder_with_parent_relative_paths(tmp_path):
    """train/images + valid/images beside a yaml saying ``train: ../train/images``."""
    for split in ("train", "valid"):
        _image(tmp_path / split / "images" / "a.jpg")
        _label(tmp_path / split / "labels" / "a.txt", "1 .5 .5 .2 .2\n")
    (tmp_path / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\nnc: 2\nnames: ['can', 'bottle']\n"
    )
    for split in ("train", "val"):
        ds = DetDataset(tmp_path / "data.yaml", split, imgsz=64, augment=False)
        labels = ds._load_labels(ds.files[0])
        assert len(ds) == 1 and labels[0, 0] == 1


def test_a_path_from_someone_elses_machine_falls_back_to_the_yaml_folder(tmp_path):
    _image(tmp_path / "images" / "train" / "a.jpg")
    (tmp_path / "data.yaml").write_text(
        "path: /content/datasets/never-here\ntrain: images/train\nval: images/train\n"
        "names: [can]\n"
    )
    assert len(DetDataset(tmp_path / "data.yaml", "train", imgsz=64)) == 1


def test_a_split_can_list_several_folders(tmp_path):
    _image(tmp_path / "images" / "day" / "a.jpg")
    _image(tmp_path / "images" / "night" / "b.jpg")
    (tmp_path / "data.yaml").write_text(
        "train: [images/day, images/night]\nval: images/day\nnames: [can]\n"
    )
    ds = DetDataset(tmp_path / "data.yaml", "train", imgsz=64)
    assert sorted(f.name for f in ds.files) == ["a.jpg", "b.jpg"]


def test_a_missing_val_entry_says_so(tmp_path):
    _image(tmp_path / "images" / "a.jpg")
    (tmp_path / "data.yaml").write_text("train: images\nnames: [can]\n")
    with pytest.raises(ValueError, match="no 'val:' entry"):
        DetDataset(tmp_path / "data.yaml", "val", imgsz=64)


def test_a_missing_folder_names_every_place_it_looked(tmp_path):
    (tmp_path / "data.yaml").write_text("train: ../nowhere/images\nval: x\nnames: [can]\n")
    with pytest.raises(FileNotFoundError, match="nowhere"):
        DetDataset(tmp_path / "data.yaml", "train", imgsz=64)


def test_label_rows_boxes_confidences_and_polygons():
    assert label_row_to_box("2 .5 .4 .2 .1".split()) == pytest.approx([2, .5, .4, .2, .1])
    # a sixth value is a confidence from a pseudo-labeller, not geometry
    assert label_row_to_box("2 .5 .4 .2 .1 .93".split()) == pytest.approx([2, .5, .4, .2, .1])
    # a polygon becomes its bounding box instead of four misread numbers
    tri = "1 0.2 0.3 0.6 0.3 0.4 0.7".split()
    assert label_row_to_box(tri) == pytest.approx([1, 0.4, 0.5, 0.4, 0.4])
    assert label_row_to_box([]) is None and label_row_to_box("0 .5".split()) is None


def test_a_notepad_label_file_with_a_bom_and_a_stray_line_still_reads(tmp_path, capsys):
    _image(tmp_path / "images" / "val" / "a.jpg")
    lp = tmp_path / "labels" / "val" / "a.txt"
    lp.parent.mkdir(parents=True)
    lp.write_bytes("\ufeff0 .5 .5 .2 .2\r\n0 x y w h\r\n0 .3 .3 .1 .1\r\n".encode())
    (tmp_path / "data.yaml").write_text("path: .\ntrain: images/val\nval: images/val\n"
                                        "names: {0: can}\n")
    ds = DetDataset(tmp_path / "data.yaml", "val", imgsz=64, augment=False)
    assert len(ds._load_labels(ds.files[0])) == 2
    assert "a.txt: 1 line(s) are not numbers" in capsys.readouterr().out


def test_an_image_list_with_a_bom_finds_its_first_image(tmp_path):
    from easydetect.data.dataset import list_images

    _image(tmp_path / "images" / "a.jpg")
    (tmp_path / "list.txt").write_bytes("\ufeffimages/a.jpg\n".encode())
    assert list_images(tmp_path, "list.txt") == [tmp_path / "images" / "a.jpg"]
