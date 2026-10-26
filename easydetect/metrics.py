# Apache-2.0
"""Validation metrics: ``metrics.box.map50`` / ``metrics.box.map``."""

from __future__ import annotations

from typing import Any


class BoxMetrics:
    """mAP numbers for the detection head."""

    def __init__(self, map50: float, map: float, per_class: dict[int, float] | None = None) -> None:
        self.map50 = float(map50)
        self.map = float(map)
        self.maps = per_class or {}  # class index -> mAP50-95

    @property
    def map50_95(self) -> float:
        return self.map

    def __repr__(self) -> str:
        return f"BoxMetrics(map50={self.map50:.4f}, map={self.map:.4f})"


class DetMetrics:
    """What :meth:`Detector.val` answers with.

    ``metrics.box.map50`` / ``metrics.box.map`` are the headline numbers;
    ``metrics["map50"]`` and ``dict(metrics.results_dict)`` are there so the
    old dict-shaped call sites keep working.
    """

    def __init__(
        self,
        map50: float,
        map: float,
        per_class: dict[int, float] | None = None,
        names: dict[int, str] | None = None,
        speed: dict[str, float] | None = None,
    ) -> None:
        self.box = BoxMetrics(map50, map, per_class)
        self.names = names or {}
        self.speed = speed or {}

    @property
    def results_dict(self) -> dict[str, float]:
        return {"map50": self.box.map50, "map": self.box.map}

    def __getitem__(self, key: str) -> Any:
        return self.results_dict[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.results_dict.get(key, default)

    def keys(self):
        return self.results_dict.keys()

    def __iter__(self):
        return iter(self.results_dict)

    def __repr__(self) -> str:
        return f"DetMetrics(map50={self.box.map50:.4f}, map50-95={self.box.map:.4f})"
