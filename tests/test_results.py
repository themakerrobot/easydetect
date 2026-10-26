# Apache-2.0
"""Results/Boxes — the half of the parity checklist users touch every day."""

from __future__ import annotations

import numpy as np
import pytest

from easydetect.results import Boxes, Results

DET = np.array(
    [[10, 10, 50, 60, 0.9, 0], [0, 0, 20, 20, 0.8, 0], [5, 5, 30, 30, 0.7, 1]], np.float32
)


@pytest.fixture
def result() -> Results:
    img = np.zeros((100, 200, 3), np.uint8)
    return Results(img, "bus.jpg", {0: "person", 1: "car"}, DET)


def test_boxes_expose_every_coordinate_flavour(result):
    boxes = result.boxes
    assert boxes.xyxy.shape == (3, 4)
    assert boxes.xywh[0].tolist() == [30.0, 35.0, 40.0, 50.0]
    assert boxes.xyxyn[0].tolist() == pytest.approx([0.05, 0.1, 0.25, 0.6])
    assert boxes.xywhn[0].tolist() == pytest.approx([0.15, 0.35, 0.2, 0.5])
    assert boxes.conf.tolist() == pytest.approx([0.9, 0.8, 0.7])
    assert boxes.cls.tolist() == [0, 0, 1]
    assert boxes.id is None  # ids only exist after track()
    assert len(boxes) == 3


def test_boxes_are_indexable_and_stay_boxes(result):
    first = result.boxes[0]
    assert isinstance(first, Boxes) and len(first) == 1
    assert [len(b) for b in result.boxes] == [1, 1, 1]


def test_track_ids_appear_when_the_data_has_seven_columns():
    data = np.array([[0, 0, 10, 10, 7, 0.9, 1]], np.float32)
    boxes = Boxes(data, (100, 100))
    assert boxes.is_track and boxes.id.tolist() == [7]
    assert boxes.conf.tolist() == pytest.approx([0.9]) and boxes.cls.tolist() == [1]


def test_empty_predictions_are_a_normal_empty_result():
    result = Results(np.zeros((10, 10, 3), np.uint8))
    assert len(result) == 0 and result.boxes.xyxy.shape == (0, 4)
    assert "no detections" in result.verbose()


def test_the_log_line_counts_and_pluralises(result):
    assert result.verbose() == "2 persons, 1 car, "
    assert str(result) == "100x200 2 persons, 1 car"


def test_names_fall_back_to_a_readable_placeholder():
    result = Results(np.zeros((10, 10, 3), np.uint8), boxes=DET)
    assert result.name_of(1) == "class_1"


def test_plot_draws_without_touching_the_original(result):
    before = result.orig_img.copy()
    plotted = result.plot()
    assert plotted.shape == result.orig_img.shape
    assert plotted.any()  # something was drawn
    assert np.array_equal(result.orig_img, before)


def test_save_writes_the_annotated_image(result, tmp_path):
    out = result.save(tmp_path / "out" / "annotated.jpg")
    assert out.exists() and out.stat().st_size > 0


def test_summary_is_json_ready(result):
    rows = result.summary()
    assert rows[0]["name"] == "person" and rows[0]["confidence"] == pytest.approx(0.9)
    assert set(rows[0]["box"]) == {"x1", "y1", "x2", "y2"}


def test_malformed_box_data_is_rejected_early():
    with pytest.raises(ValueError, match=r"\(N, 6\)"):
        Boxes(np.zeros((3, 5), np.float32), (10, 10))


class TestLabelPlacement:
    """Labels have to stay inside the frame and off each other."""

    def test_a_label_at_the_right_edge_slides_back_inside(self):
        from easydetect.plotting import _label_position

        left, top = _label_position(x1=195, y1=50, y2=80, tw=60, th=12, shape=(100, 200), placed=[])
        assert left + 60 <= 200 and left >= 0
        assert 0 <= top <= 100 - 12

    def test_a_label_at_the_top_drops_below_the_line(self):
        from easydetect.plotting import _label_position

        _, top = _label_position(x1=10, y1=2, y2=40, tw=30, th=12, shape=(100, 200), placed=[])
        assert top >= 0

    def test_a_second_label_moves_instead_of_covering_the_first(self):
        from easydetect.plotting import _label_position

        first = _label_position(x1=10, y1=50, y2=90, tw=40, th=12, shape=(200, 200), placed=[])
        taken = [(first[0], first[1], first[0] + 40, first[1] + 12)]
        second = _label_position(x1=10, y1=50, y2=90, tw=40, th=12, shape=(200, 200), placed=taken)
        assert second != first

    def test_stacked_boxes_take_the_free_slots_before_colliding(self):
        """Three overlapping detections: each label finds its own row."""
        from easydetect.plotting import _label_position, _overlaps

        placed = []
        for i in range(3):
            left, top = _label_position(
                x1=10 + i, y1=40 + i, y2=90 + i, tw=50, th=12, shape=(300, 300), placed=placed
            )
            box = (left, top, left + 50, top + 12)
            assert all(not _overlaps(box, other) for other in placed)
            placed.append(box)

    def test_labels_never_leave_the_frame_however_crowded(self):
        from easydetect.plotting import _label_position

        placed = []
        for i in range(12):
            left, top = _label_position(
                x1=280, y1=2 + i, y2=40 + i, tw=50, th=12, shape=(100, 300), placed=placed
            )
            assert 0 <= left and left + 50 <= 300
            assert 0 <= top and top + 12 <= 100
            placed.append((left, top, left + 50, top + 12))


def test_a_fixed_colour_overrides_the_class_palette(result):
    """Overlays need "truth vs prediction" to read, not "class 0 vs class 1"."""
    import numpy as np

    from easydetect.plotting import color_for, draw_boxes

    palette = draw_boxes(result.orig_img, result.boxes, result.names)
    pinned = draw_boxes(result.orig_img, result.boxes, result.names, color=(0, 255, 0))
    assert not np.array_equal(palette, pinned)
    assert (pinned[:, :, 1] == 255).any()          # the green went on
    assert tuple(color_for(0)) != (0, 255, 0)      # and it is not the class colour
