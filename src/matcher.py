from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from point_cloud import PointCloud


@dataclass
class Matching:
    """Correspondence between a source and a target point cloud.

    Attributes:
        source_points:    float (N, 3) source point positions.
        target_positions: float (N, 3) effective target position per source point.
                          Hard matching: direct target lookup.
                          Soft matching: Gaussian-weighted average of k neighbors.
        weights:          float (N,) per-point M-step confidence. Used by SoftMatching, target_positions that are
                          derived from a geometrically close cluster of points have higher confidence. None = uniform.
    """

    source_points: np.ndarray     # (N, 3)
    target_positions: np.ndarray  # (N, 3)
    weights: np.ndarray | None = None


class Matcher(ABC):
    """Abstract base class for point cloud correspondence algorithms (E-step)."""

    @abstractmethod
    def match(self, source: PointCloud, target: PointCloud) -> Matching:
        """Compute a correspondence from source to target.

        Args:
            source: PointCloud (N, 3).
            target: PointCloud (M, 3).

        Returns:
            Matching with source_points and target_positions always populated.
        """
        ...


class NearestNeighborMatcher(Matcher):
    """Hard E-step: each source point maps to its single nearest target point."""

    def match(self, source: PointCloud, target: PointCloud) -> Matching:
        """Assign each source point to its nearest neighbor in target.

        Args:
            source: PointCloud (N, 3).
            target: PointCloud (M, 3).

        Returns:
            Matching that assigns a target_point to each source point with uniform weights.
        """
        tree = KDTree(target.points)
        _, nn_indices = tree.query(source.points)
        return Matching(
            source_points=source.points,
            target_positions=target.points[nn_indices],
        )


class GaussianMatcher(Matcher):
    """
    Soft E-step: each source point maps to a Gaussian-weighted average of
    its k nearest target points.

    The bandwidth sigma controls softness: large sigma gives a blurred global
    view; small sigma approaches hard nearest-neighbor matching. sigma is a
    plain attribute and can be updated externally — use SigmaAnnealingCallback
    from icp.py to vary it over ICP iterations.
    """

    def __init__(self, sigma: float, k: int = 20):
        """
        Args:
            sigma: Bandwidth for exp(-d² / (2σ²)). This parameter can be annealed using the SigmaAnnealingCallback
            k:     Number of nearest neighbors used to approximate the
                   full Gaussian weight sum.
        """
        self.sigma = sigma
        self.k = k

    def match(self, source: PointCloud, target: PointCloud) -> Matching:
        """Soft Gaussian correspondence from source to target.

        Args:
            source: PointCloud (N, 3).
            target: PointCloud (M, 3).

        Returns:
            Matching with target_positions (N, 3) as the weighted-average
            target position and weights (N,) as total per-point confidence.
        """
        k = min(self.k, len(target))
        tree = KDTree(target.points)
        dists, nbr_idx = tree.query(source.points, k=k)  # (N, k)

        log_w = -0.5 * (dists / self.sigma) ** 2         # (N, k)
        log_w -= log_w.max(axis=1, keepdims=True)        # numerical stability
        w = np.exp(log_w)                                # (N, k)

        row_sums = w.sum(axis=1, keepdims=True)          # (N, 1)
        w_norm = w / row_sums                            # (N, k), rows sum to 1

        target_positions = (w_norm[:, :, None] * target.points[nbr_idx]).sum(axis=1)

        return Matching(
            source_points=source.points,
            target_positions=target_positions,
            weights=row_sums.squeeze(1),
        )
