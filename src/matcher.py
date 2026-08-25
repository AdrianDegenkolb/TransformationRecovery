from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import KDTree

from point_cloud import PointCloud
from feature_extractor import FeatureExtractor


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
    """Soft E-step: each source point maps to a Gaussian-weighted average of its k nearest
    target points, with optional feature-based augmentation.

    Two feature integration modes are supported (selected via ``feature_mode``):

    - ``'additive'``: k-NN candidates are found in 3D space; cosine similarity between
      feature vectors is added to the log-weights:
      ``log_w += alpha * cos_sim(feat_src[i], feat_tgt[j])``.

    - ``'append'``: feature vectors (z-scored, scaled by ``beta``) are appended to 3D
      coordinates before building the KDTree. Both candidate selection and Gaussian
      weights use the joint distance. True correspondences have zero feature distance
      (feature vectors match by index), so their joint distance equals the spatial
      distance; false correspondences pay a large feature penalty that drives their
      weight toward zero regardless of ``sigma``.

    ``sigma`` is a plain attribute and can be updated externally — use
    ``SigmaAnnealingCallback`` from ``icp.py`` to vary it over ICP iterations.
    """

    def __init__(
        self,
        sigma: float,
        k: int = 20,
        feature_extractor: FeatureExtractor | None = None,
        feature_mode: Literal['additive', 'append'] = 'additive',
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Args:
            sigma:             Gaussian bandwidth for exp(-d² / (2σ²)). Can be annealed
                               via SigmaAnnealingCallback.
            k:                 Number of nearest neighbors used per source point.
            feature_extractor: Optional extractor producing a (N, D) feature matrix per
                               cloud. When None, falls back to purely spatial matching.
            feature_mode:      How features are integrated:
                               - 'additive': cosine similarity added to log-weights.
                               - 'append':   features appended to coordinates for k-NN.
            alpha:             Cosine similarity weight (additive mode only).
                               Setting alpha=0 disables the feature term.
            beta:              Scale of feature dimensions relative to spatial coordinates
                               in the joint KDTree (append mode only).
        """
        self.sigma = sigma
        self.k = k
        self.feature_extractor = feature_extractor
        self.feature_mode = feature_mode
        self.alpha = alpha
        self.beta = beta

    def _zscored_features(
        self,
        source: PointCloud,
        target: PointCloud,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute z-scored feature matrices for source and target using target statistics.

        Z-scoring uses target mean and std so that beta (append mode) and alpha (additive
        mode) are interpretable regardless of the raw feature scale.

        Args:
            source: Source point cloud.
            target: Target point cloud.

        Returns:
            Tuple (feat_src_z, feat_tgt_z), each of shape (N, D) and (M, D) respectively.
        """
        feat_src = self.feature_extractor.get_features(source)   # (N, D)
        feat_tgt = self.feature_extractor.get_features(target)   # (M, D)
        feat_mean = feat_tgt.mean(axis=0)
        feat_std  = feat_tgt.std(axis=0) + 1e-8
        return (feat_src - feat_mean) / feat_std, (feat_tgt - feat_mean) / feat_std

    def match(self, source: PointCloud, target: PointCloud) -> Matching:
        """Soft Gaussian correspondence from source to target.

        Args:
            source: PointCloud (N, 3).
            target: PointCloud (M, 3).

        Returns:
            Matching with target_positions (N, 3) as the weighted-average target position
            and weights (N,) as total per-point confidence.
        """
        k = min(self.k, len(target))

        # --- Candidate selection ---
        feat_src_z = feat_tgt_z = None
        if self.feature_extractor is not None:
            feat_src_z, feat_tgt_z = self._zscored_features(source, target)

        n = len(source.points)
        if feat_src_z is not None and self.feature_mode == 'append':
            all_pts  = np.vstack([source.points, target.points])
            pos_mean = all_pts.mean()
            pos_std  = all_pts.std() + 1e-8
            pos_src_z = (source.points - pos_mean) / pos_std  # (N, 3)
            pos_tar_z = (target.points - pos_mean) / pos_std  # (M, 3)

            joint_src = np.hstack([pos_src_z, self.beta * feat_src_z])  # (N, 3+D)
            joint_tgt = np.hstack([pos_tar_z, self.beta * feat_tgt_z])  # (M, 3+D)
            dists, nbr_idx = KDTree(joint_tgt).query(joint_src, k=k)    # joint dist
            dists *= pos_std                                            # rescale to physical units
            dists = dists.reshape(n, k)                                 # ensure (N, k)
            nbr_idx = nbr_idx.reshape(n, k)
        else:
            dists, nbr_idx = KDTree(target.points).query(source.points, k=k)
            dists   = dists.reshape(n, k)                                    # ensure (N, k)
            nbr_idx = nbr_idx.reshape(n, k)

        # --- Log-weight computation ---
        log_w = -0.5 * (dists / self.sigma) ** 2                            # (N, k)

        if feat_src_z is not None and self.feature_mode == 'additive' and self.alpha != 0.0:
            norm = np.linalg.norm(feat_src_z, axis=1, keepdims=True) + 1e-8
            feat_src_unit = feat_src_z / norm                                # (N, D)
            norm = np.linalg.norm(feat_tgt_z, axis=1, keepdims=True) + 1e-8
            feat_tgt_unit = feat_tgt_z / norm                                # (M, D)
            feat_nbrs = feat_tgt_unit[nbr_idx]                               # (N, k, D)
            cos_sim   = (feat_src_unit[:, None, :] * feat_nbrs).sum(-1)     # (N, k)
            log_w    += self.alpha * cos_sim

        # --- Softmax normalization and weighted average ---
        log_w -= log_w.max(axis=1, keepdims=True)                           # numerical stability
        w = np.exp(log_w)                                                    # (N, k)

        row_sums = w.sum(axis=1, keepdims=True)                             # (N, 1)
        w_norm   = w / row_sums                                             # (N, k)

        target_positions = (w_norm[:, :, None] * target.points[nbr_idx]).sum(axis=1)

        return Matching(
            source_points=source.points,
            target_positions=target_positions,
            weights=row_sums.squeeze(1),
        )
