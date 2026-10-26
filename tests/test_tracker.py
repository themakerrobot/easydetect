# Apache-2.0
"""The tracker: ids that survive movement, crossings and unsure frames, and
die when the object leaves."""

from __future__ import annotations

import numpy as np
import pytest

from easydetect.tracker import IoUTracker, greedy_assignment, iou_matrix, linear_assignment


def det(*boxes):
    return np.array([[*b, 0.9, 0] for b in boxes], np.float32)


def test_iou_matrix_handles_overlap_and_emptiness():
    a = np.array([[0, 0, 10, 10]], np.float32)
    assert iou_matrix(a, a)[0, 0] == 1.0
    assert iou_matrix(a, np.array([[20, 20, 30, 30]], np.float32))[0, 0] == 0.0
    assert iou_matrix(a, np.zeros((0, 4), np.float32)).shape == (1, 0)


def test_a_slowly_moving_box_keeps_its_id():
    tracker = IoUTracker()
    first = tracker.update(det([0, 0, 20, 20]))
    second = tracker.update(det([2, 2, 22, 22]))
    assert first.shape[1] == 7
    assert first[0, 4] == second[0, 4] == 1


def test_a_box_that_jumps_away_gets_a_new_id():
    tracker = IoUTracker()
    tracker.update(det([0, 0, 20, 20]))
    moved = tracker.update(det([100, 100, 120, 120]))
    assert moved[0, 4] == 2


def test_two_objects_keep_separate_ids_across_frames():
    tracker = IoUTracker()
    tracker.update(det([0, 0, 20, 20], [100, 100, 120, 120]))
    second = tracker.update(det([101, 101, 121, 121], [1, 1, 21, 21]))
    assert second[:, 4].tolist() == [2, 1]


def test_ids_are_not_shared_between_classes_when_asked():
    tracker = IoUTracker(match_class=True)
    tracker.update(np.array([[0, 0, 20, 20, 0.9, 0]], np.float32))
    other_class = tracker.update(np.array([[0, 0, 20, 20, 0.9, 1]], np.float32))
    assert other_class[0, 4] == 2


def test_a_track_expires_after_max_age_frames():
    tracker = IoUTracker(max_age=2)
    tracker.update(det([0, 0, 20, 20]))
    for _ in range(3):
        tracker.update(np.zeros((0, 6), np.float32))
    assert tracker.update(det([0, 0, 20, 20]))[0, 4] == 2


def test_an_empty_frame_is_an_empty_seven_column_array():
    assert IoUTracker().update(np.zeros((0, 6), np.float32)).shape == (0, 7)


def test_reset_starts_ids_over():
    tracker = IoUTracker()
    tracker.update(det([0, 0, 20, 20]))
    tracker.reset()
    assert tracker.update(det([0, 0, 20, 20]))[0, 4] == 1


def test_a_wavering_class_keeps_one_id_and_shows_the_class_seen_most():
    tracker = IoUTracker()
    for cls in (0, 0, 1, 0, 1):
        out = tracker.update(np.array([[0, 0, 20, 20, 0.9, cls]], np.float32))
    assert out[0, 4] == 1 and out[0, 6] == 0


def test_a_taken_best_track_does_not_cost_a_new_id():
    """The old greedy matcher gave up when a box's best track was taken,
    though another track overlapped it well enough."""
    tracker = IoUTracker()
    tracker.update(np.array([[0, 0, 100, 100, 0.9, 0], [60, 0, 160, 100, 0.8, 0]], np.float32))
    out = tracker.update(np.array([[62, 0, 162, 100, 0.9, 0], [40, 0, 140, 100, 0.8, 0]],
                                  np.float32))
    assert out[:, 4].tolist() == [2, 1] and len(tracker.tracks) == 2


@pytest.mark.parametrize("assign", [linear_assignment, greedy_assignment])
def test_assignment_pairs_each_box_with_one_track(assign):
    iou = np.array([[0.96, 0.23], [0.67, 0.43]])
    assert sorted(assign(iou, 0.2)) == [(0, 0), (1, 1)]
    assert assign(iou, 0.5) == [(0, 0)]
    assert assign(np.zeros((0, 3)), 0.2) == []


def test_without_scipy_the_best_pairs_go_first(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    assert linear_assignment(np.array([[0.9, 0.5], [0.8, 0.1]]), 0.2) == [(0, 0)]


@pytest.mark.parametrize("dy", [5, 20, 40])
def test_crossing_objects_keep_their_ids(dy):
    tracker, frames, speed = IoUTracker(), 16, 30
    for k in range(frames):
        a = [20 + speed * k, 0, 120 + speed * k, 100, 0.9, 0]
        x = 20 + speed * (frames - 1 - k)
        out = tracker.update(np.array([a, [x, dy, x + 100, 100 + dy, 0.85, 0]], np.float32))
        assert out[:, 4].tolist() == [1, 2], f"frame {k}"


def test_a_fast_box_keeps_its_id():
    """60 px a frame on a 100 px box: where it was overlaps where it is by
    only a quarter, and less once it speeds away from the first guess."""
    tracker = IoUTracker()
    for k in range(20):
        x = 60 * k
        out = tracker.update(np.array([[x, 0, x + 100, 100, 0.9, 0]], np.float32), conf=0.5)
        assert out[:, 4].tolist() == [1], f"frame {k}"


def test_an_unsure_frame_keeps_the_id_and_is_shown_only_when_asked():
    tracker = IoUTracker()
    for k, score in enumerate([0.9, 0.9, 0.9, 0.3, 0.3, 0.9]):
        x = 5 * k
        out = tracker.update(np.array([[x, 0, x + 50, 50, score, 0]], np.float32), conf=0.5)
        assert out[:, 4].tolist() == ([] if score < 0.5 else [1])
    shown = tracker.update(np.array([[30, 0, 80, 50, 0.3, 0]], np.float32), conf=0.25)
    assert shown[:, 4].tolist() == [1] and shown[0, 5] == pytest.approx(0.3)


def test_unsure_boxes_never_start_a_track():
    tracker = IoUTracker()
    for _ in range(3):
        out = tracker.update(np.array([[0, 0, 50, 50, 0.5, 0]], np.float32), conf=0.25)
    assert len(out) == 0 and not tracker.tracks


def test_a_new_object_shows_from_its_second_sighting():
    tracker = IoUTracker()
    first = tracker.update(det([0, 0, 20, 20]), conf=0.5)       # the first frame: at once
    assert first[:, 4].tolist() == [1]
    seen = tracker.update(det([1, 1, 21, 21], [100, 100, 140, 140]), conf=0.5)
    assert seen[:, 4].tolist() == [1]                           # the newcomer waits a frame
    again = tracker.update(det([2, 2, 22, 22], [102, 100, 142, 140]), conf=0.5)
    assert again[:, 4].tolist() == [1, 2]


def test_a_one_frame_blip_is_forgotten():
    tracker = IoUTracker()
    tracker.update(det([0, 0, 20, 20]), conf=0.5)
    tracker.update(det([0, 0, 20, 20], [100, 100, 140, 140]), conf=0.5)
    tracker.update(det([0, 0, 20, 20]), conf=0.5)
    assert [t.id for t in tracker.tracks] == [1]


def test_a_lost_track_waits_for_its_object():
    tracker = IoUTracker(max_age=5)
    tracker.update(det([0, 0, 50, 50]), conf=0.5)
    for _ in range(4):
        tracker.update(np.zeros((0, 6), np.float32), conf=0.5)
    assert tracker.update(det([1, 0, 51, 50]), conf=0.5)[:, 4].tolist() == [1]


def test_the_trackers_own_conf_is_used_when_update_is_not_given_one():
    tracker = IoUTracker()
    tracker.conf = 0.5
    assert len(tracker.update(np.array([[0, 0, 20, 20, 0.3, 0]], np.float32))) == 0
