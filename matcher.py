from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import KDTree

from point_cloud import PointCloud


class Matcher(ABC):
    """Abstract base class for point cloud correspondence algorithms (E-step)."""

    @abstractmethod
    def match(
        self, source: PointCloud, target: PointCloud
    ) -> tuple[PointCloud, PointCloud, np.ndarray]:
        """Return (source, target_matched, indices) where target_matched[i] is the
        correspondence for source[i] in target, and indices[i] is its index in target.

        Args:
            source: PointCloud of shape (N, 3).
            target: PointCloud of shape (M, 3).

        Returns:
            Tuple (source, target_matched, indices):
              - source:         PointCloud (N, 3), unchanged.
              - target_matched: PointCloud (N, 3), target points reordered to match source.
              - indices:        int ndarray (N,), index into target for each source point.
        """
        ...


class NearestNeighborMatcher(Matcher):
    """E-step via nearest-neighbor search using a KD-tree.

    For each point in source, assigns the closest point in target.
    Multiple source points may map to the same target point.
    """

    def match(
        self, source: PointCloud, target: PointCloud
    ) -> tuple[PointCloud, PointCloud, np.ndarray]:
        """Assign each source point to its nearest neighbor in target.

        Args:
            source: PointCloud of shape (N, 3).
            target: PointCloud of shape (M, 3).

        Returns:
            (source, target_matched, indices) where target_matched[i] is the nearest
            neighbor of source[i] in target, and indices[i] is its index in target.
        """
        tree = KDTree(target.points)
        _, indices = tree.query(source.points)
        return source, PointCloud(target.points[indices]), indices
