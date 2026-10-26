# Apache-2.0
"""A ByteTrack-style tracker — enough to give boxes stable ``.id`` across frames.

Each track has a constant-velocity Kalman filter on its box (centre, aspect
ratio, height), so an object that moves fast is looked for where it is going,
not where it was. Detections are matched to tracks globally (the Hungarian
method when SciPy is installed, best pairs first otherwise), in two rounds as
ByteTrack does: confident boxes first, then unsure ones — a half-hidden person
scores 0.3 — only to carry on tracks that already exist, never to start one.
Classes do not split tracks by default: a detector that wavers between "car"
and "truck" keeps one id, and the track shows the class it has had most
often. A lost track waits ``max_age`` frames for its object.

The thresholds suit ed, whose unsure boxes score 0.3-0.5. No appearance
features: an object hidden for long, or two that swap places while hidden,
can come back with a new id.
"""

from __future__ import annotations

import numpy as np

#: a low-score box must overlap its track this much (ByteTrack's second round)
LOW_MATCH_IOU = 0.5


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy -> (N,M) IoU."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None] - inter, 1e-9, None)


def greedy_assignment(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """``(row, col)`` pairs, the most-overlapping first, each row and column
    used once, none under ``threshold``."""
    rows, cols = np.nonzero(iou >= threshold)
    pairs, used_r, used_c = [], set(), set()
    for k in np.argsort(-iou[rows, cols], kind="stable"):
        r, c = int(rows[k]), int(cols[k])
        if r not in used_r and c not in used_c:
            pairs.append((r, c))
            used_r.add(r)
            used_c.add(c)
    return pairs


def linear_assignment(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """``(row, col)`` pairs that match as many as possible at ``threshold``
    or more, then with the most overlap in all (the Hungarian method); best
    pairs first when SciPy is not installed."""
    if not iou.size:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return greedy_assignment(iou, threshold)
    ok = iou >= threshold
    # a pair under the threshold costs more than all allowed pairs together
    cost = np.where(ok, 1.0 - iou, 1.0 + min(iou.shape))
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if ok[r, c]]


class KalmanBox:
    """Constant-velocity Kalman filter on ``(cx, cy, w/h, h)`` and their
    velocities — ByteTrack's (from SORT), noise scaled by the box's height."""

    POS, VEL = 1 / 20, 1 / 160
    F = np.eye(8)
    F[:4, 4:] = np.eye(4)
    H = np.eye(4, 8)

    def __init__(self, box: np.ndarray) -> None:
        z = self.measure(box)
        h = z[3]
        self.mean = np.r_[z, np.zeros(4)]
        std = [2 * self.POS * h, 2 * self.POS * h, 1e-2, 2 * self.POS * h,
               10 * self.VEL * h, 10 * self.VEL * h, 1e-5, 10 * self.VEL * h]
        self.cov = np.diag(np.square(std))

    @staticmethod
    def measure(box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = (float(v) for v in box)
        w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
        return np.array([x1 + w / 2, y1 + h / 2, w / h, h])

    def predict(self) -> None:
        h = self.mean[3]
        q = [self.POS * h, self.POS * h, 1e-2, self.POS * h,
             self.VEL * h, self.VEL * h, 1e-5, self.VEL * h]
        self.mean = self.F @ self.mean
        self.cov = self.F @ self.cov @ self.F.T + np.diag(np.square(q))

    def update(self, box: np.ndarray) -> None:
        h = self.mean[3]
        r = np.diag(np.square([self.POS * h, self.POS * h, 1e-1, self.POS * h]))
        s = self.H @ self.cov @ self.H.T + r
        gain = np.linalg.solve(s, self.H @ self.cov).T
        self.mean = self.mean + gain @ (self.measure(box) - self.H @ self.mean)
        self.cov = self.cov - gain @ s @ gain.T

    @property
    def box(self) -> np.ndarray:
        cx, cy, a, h = self.mean[:4]
        h = max(h, 1e-3)
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


class Track:
    """One object followed across frames. ``box`` is its latest detection,
    ``age`` the frames since it was last seen, ``cls`` the class it has had
    most often."""

    __slots__ = ("id", "box", "cls", "age", "hits", "score", "confirmed", "kalman", "counts")

    def __init__(self, track_id: int, box: np.ndarray, cls: int, score: float = 1.0,
                 confirmed: bool = True) -> None:
        self.id = track_id
        self.box = box
        self.cls = cls
        self.age = 0
        self.hits = 1
        self.score = score
        self.confirmed = confirmed
        self.kalman = KalmanBox(box)
        self.counts = {cls: 1}

    def predict(self) -> np.ndarray:
        if self.age:  # lost: stop it growing or shrinking on its own
            self.kalman.mean[7] = 0.0
        self.kalman.predict()
        return self.kalman.box

    def hit(self, box: np.ndarray, cls: int, score: float) -> None:
        self.kalman.update(box)
        self.box, self.score, self.age = box, score, 0
        self.hits += 1
        self.confirmed = True
        self.counts[cls] = self.counts.get(cls, 0) + 1
        # the class seen most often (on a tie, the latest): only cls's count grew
        if self.counts[cls] >= self.counts.get(self.cls, 0):
            self.cls = cls


class IoUTracker:
    """``tracker.update(det) -> det with ids``, frame after frame.

    ``iou``: how much a confident box must overlap where a track is expected
    to carry its id on. ``max_age``: frames a lost track waits for its object.
    ``match_class``: only match boxes and tracks of one class. ``high``:
    boxes scoring this much are matched first and can be new objects if they
    also score ``new``; boxes from ``low`` up to ``high`` only carry tracks
    on. ``conf``: the score a box needs to be shown (see :meth:`update`).
    """

    def __init__(self, iou: float = 0.2, max_age: int = 30, match_class: bool = False,
                 high: float = 0.45, low: float = 0.15, new: float = 0.55,
                 conf: float | None = None) -> None:
        self.iou = iou
        self.max_age = max_age
        self.match_class = match_class
        self.high, self.low, self.new = high, low, new
        self.conf = conf
        self.tracks: list[Track] = []
        self._next_id = 1
        self._frames = 0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self._frames = 0

    def _ious(self, det: np.ndarray, tracks: list[Track], boxes: np.ndarray) -> np.ndarray:
        ious = iou_matrix(det[:, :4], boxes)
        if self.match_class and len(det) and len(tracks):
            same = det[:, 5][:, None] == np.array([t.cls for t in tracks], np.float32)
            ious = np.where(same, ious, 0.0)
        return ious

    def update(self, det: np.ndarray, conf: float | None = None) -> np.ndarray:
        """``(N,6)`` detections -> ``(M,7)`` with the track id at column 4.

        With ``conf`` (or the tracker's ``conf``), give it every box the
        detector has down to ``low``: the answer is the boxes of confirmed
        tracks that score ``conf`` or more — a track that is born after the
        first frame shows from its second sighting. Without, every box given
        is taken as one to show (the caller has filtered them): each gets an
        id at once, and the answer has a row for each, in order.
        """
        det = np.asarray(det, np.float32).reshape(-1, 6)
        conf = self.conf if conf is None else conf
        self._frames += 1
        if conf is None:
            high = low = new = -np.inf
            show = -np.inf
        else:
            high, low, new, show = self.high, min(self.low, conf), self.new, conf

        predicted = np.stack([t.predict() for t in self.tracks]) if self.tracks \
            else np.zeros((0, 4), np.float32)
        owner = np.full(len(det), -1)  # the track each detection went to
        scores = det[:, 4]
        free = list(range(len(self.tracks)))

        # round 1: confident boxes, with every track
        first = np.flatnonzero(scores >= high)
        for d, t in linear_assignment(self._ious(det[first], self.tracks, predicted), self.iou):
            owner[first[d]] = free[t]
        free = [t for t in free if t not in set(owner.tolist())]
        # round 2: unsure boxes, only with the tracks seen last frame
        second = np.flatnonzero((scores < high) & (scores >= low))
        seen = [t for t in free if self.tracks[t].age == 0 and self.tracks[t].confirmed]
        if len(second) and seen:
            ious = self._ious(det[second], [self.tracks[t] for t in seen], predicted[seen])
            for d, t in linear_assignment(ious, max(self.iou, LOW_MATCH_IOU)):
                owner[second[d]] = seen[t]

        matched = set(owner.tolist())
        for d in np.flatnonzero(owner >= 0):
            self.tracks[owner[d]].hit(det[d, :4], int(det[d, 5]), float(scores[d]))
        kept = []
        for t, track in enumerate(self.tracks):
            if t in matched:
                kept.append(track)
                continue
            track.age += 1
            # a track seen once and missed next is more likely noise than an object
            if track.confirmed and track.age <= self.max_age:
                kept.append(track)
        for d in np.flatnonzero((owner < 0) & (scores >= new)):
            track = Track(self._next_id, det[d, :4], int(det[d, 5]), float(scores[d]),
                          confirmed=conf is None or self._frames == 1)
            self._next_id += 1
            kept.append(track)
            owner[d] = len(self.tracks)
            self.tracks.append(track)
        rows = [d for d in range(len(det)) if owner[d] >= 0
                and self.tracks[owner[d]].confirmed and scores[d] >= show]
        out = np.zeros((len(rows), 7), np.float32)
        for k, d in enumerate(rows):
            track = self.tracks[owner[d]]
            out[k] = (*det[d, :4], track.id, scores[d], track.cls)
        self.tracks = kept
        return out
