"""T-shape geometry and the coverage (success) metric.

Local T frame: origin at the centroid of the T, +y points from the stem towards the bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


@dataclass(frozen=True)
class TShape:
    bar_length: float
    bar_width: float
    stem_length: float
    stem_width: float
    height: float

    @classmethod
    def from_config(cls, cfg: dict) -> "TShape":
        return cls(**{k: float(cfg[k]) for k in ("bar_length", "bar_width", "stem_length", "stem_width", "height")})

    @property
    def centroid_offset(self) -> float:
        """y of the centroid in a frame where the stem starts at y=0 and the bar sits on top."""
        a_stem = self.stem_width * self.stem_length
        a_bar = self.bar_length * self.bar_width
        return (a_stem * self.stem_length / 2 + a_bar * (self.stem_length + self.bar_width / 2)) / (a_stem + a_bar)

    def rects(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """[(center_xy, half_size_xy, area_fraction)] for bar and stem in the local T frame."""
        c = self.centroid_offset
        a_stem = self.stem_width * self.stem_length
        a_bar = self.bar_length * self.bar_width
        total = a_stem + a_bar
        bar = (np.array([0.0, self.stem_length + self.bar_width / 2 - c]),
               np.array([self.bar_length / 2, self.bar_width / 2]), a_bar / total)
        stem = (np.array([0.0, self.stem_length / 2 - c]),
                np.array([self.stem_width / 2, self.stem_length / 2]), a_stem / total)
        return [bar, stem]

    def local_polygon(self) -> Polygon:
        parts = [box(*(ctr - half), *(ctr + half)) for ctr, half, _ in self.rects()]
        return unary_union(parts)

    def polygon(self, x: float, y: float, yaw: float) -> Polygon:
        poly = affinity.rotate(self.local_polygon(), yaw, origin=(0, 0), use_radians=True)
        return affinity.translate(poly, x, y)


def coverage(shape: TShape, block_pose, target_pose) -> float:
    """Fraction of the target T area covered by the block T (Diffusion Policy metric)."""
    b = shape.polygon(*block_pose)
    t = shape.polygon(*target_pose)
    return float(b.intersection(t).area / t.area)


def yaw_from_quat(q) -> float:
    """Yaw (rotation about world z) from a MuJoCo quaternion [w, x, y, z]."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
