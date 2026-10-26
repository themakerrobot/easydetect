# Apache-2.0
"""A segmented result's summary walks its masks once, not once a row; a
tracked frame with no boxes still has (no) ids."""

from __future__ import annotations

import numpy as np

from easydetect.results import Boxes, Masks, Results


def test_summary_takes_each_mask_property_once(monkeypatch):
    calls = {"xy": 0, "area": 0}
    for name in calls:
        prop = getattr(Masks, name)

        def counted(self, prop=prop, name=name):
            calls[name] += 1
            return prop.fget(self)

        monkeypatch.setattr(Masks, name, property(counted))
    masks = np.zeros((6, 40, 50), bool)
    for i in range(6):
        masks[i, 5 * i:5 * i + 10, 4 * i:4 * i + 12] = True
    boxes = np.array([[4 * i, 5 * i, 4 * i + 12, 5 * i + 10, 0.9, 0] for i in range(6)],
                     np.float32)
    rows = Results(np.zeros((40, 50, 3), np.uint8), boxes=boxes, masks=masks).summary()
    assert calls == {"xy": 1, "area": 1}
    assert [r["mask"]["area"] for r in rows] == [120] * 6
    assert rows[2]["mask"]["polygon"][0] == [8.0, 10.0]


def test_an_empty_tracked_frame_still_has_ids():
    boxes = Boxes(np.zeros((0, 7), np.float32), (10, 10))
    assert boxes.is_track and boxes.id.tolist() == []
    assert Boxes(np.zeros((0, 6), np.float32), (10, 10)).id is None
    assert Boxes(np.zeros(0, np.float32), (10, 10)).data.shape == (0, 6)
