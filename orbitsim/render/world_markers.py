"""Pure layout math for readable world-space markers."""
from __future__ import annotations

import numpy as np


def apsis_indices(points_m) -> tuple[int, int]:
    """Return indices of minimum/maximum radius in an ``(N, 3)`` path."""
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_m must have shape (N, 3) with N > 0")
    radii_sq = np.einsum("ij,ij->i", points, points)
    return int(np.argmin(radii_sq)), int(np.argmax(radii_sq))


def apsis_points_on_path(points_m) -> tuple[np.ndarray, np.ndarray]:
    """Return sub-sample Pe/Ap positions that lie on the rendered polyline.

    Radial extrema normally fall between sampled vertices. A three-sample
    parabolic estimate finds the fractional index of each extremum, then the
    returned position is linearly interpolated on that adjacent line segment.
    This avoids marker hopping when a moving sample grid crosses an apsis.
    """
    points = np.asarray(points_m, dtype=np.float64)
    pe_index, ap_index = apsis_indices(points)
    radii_sq = np.einsum("ij,ij->i", points, points)

    def interpolate(index: int) -> np.ndarray:
        # At an open-path endpoint there is no adjacent segment on both sides;
        # the endpoint itself is already exactly on the rendered line.
        if index == 0 or index == len(points) - 1 or len(points) < 3:
            return points[index].copy()
        y_prev, y_here, y_next = radii_sq[index - 1:index + 2]
        curvature = y_prev - 2.0 * y_here + y_next
        if abs(curvature) <= np.finfo(np.float64).eps * max(abs(y_here), 1.0):
            return points[index].copy()
        offset = 0.5 * (y_prev - y_next) / curvature
        offset = float(np.clip(offset, -1.0, 1.0))
        neighbor = index + (1 if offset >= 0.0 else -1)
        return points[index] + abs(offset) * (points[neighbor] - points[index])

    return interpolate(pe_index), interpolate(ap_index)


def distance_fade(
    distance_m: float,
    near_m: float,
    far_m: float,
    *,
    minimum: float = 0.22,
) -> float:
    """Smoothly fade a distant item from one to ``minimum`` alpha."""
    if far_m <= near_m:
        raise ValueError("far_m must be greater than near_m")
    minimum = max(0.0, min(1.0, minimum))
    t = max(0.0, min(1.0, (distance_m - near_m) / (far_m - near_m)))
    smooth = t * t * (3.0 - 2.0 * t)
    return 1.0 + (minimum - 1.0) * smooth


def declutter_indices(
    points_px: list[tuple[float, float] | None],
    priorities: list[int],
    *,
    min_separation_px: float,
) -> set[int]:
    """Keep higher-priority labels, suppressing nearby lower-priority labels."""
    if len(points_px) != len(priorities):
        raise ValueError("points_px and priorities must have equal length")
    visible: set[int] = set()
    accepted: list[tuple[float, float]] = []
    order = sorted(range(len(points_px)), key=lambda i: (-priorities[i], i))
    min_sq = min_separation_px * min_separation_px
    for index in order:
        point = points_px[index]
        if point is None:
            continue
        if all((point[0] - x) ** 2 + (point[1] - y) ** 2 >= min_sq for x, y in accepted):
            visible.add(index)
            accepted.append(point)
    return visible
