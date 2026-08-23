from __future__ import annotations

import matplotlib.axes
import numpy as np

from point_cloud import PointCloud


def rotation_angle_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular distance between two rotation matrices in degrees.

    Args:
        R1: SO(3) matrix (3, 3).
        R2: SO(3) matrix (3, 3).

    Returns:
        Angle in degrees between R1 and R2.
    """
    cos = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def plot_point_clouds_xy(
    ax: matplotlib.axes.Axes,
    clouds: list[PointCloud],
    labels: list[str],
    colors: list[str],
    idx: np.ndarray | None = None,
) -> None:
    """Scatter-plot point clouds projected onto the XY plane.

    Args:
        ax:     Matplotlib axes to draw on.
        clouds: Point clouds to plot.
        labels: Legend label for each cloud.
        colors: Colour for each cloud.
        idx:    Optional index array to subsample all clouds consistently.
                Pass the same idx across multiple panels to track the same points.
    """
    for cloud, label, color in zip(clouds, labels, colors):
        pts = cloud.points if idx is None else cloud.points[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.4, color=color, label=label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
