from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from point_cloud import PointCloud
from feature_extractor import FeatureExtractor, zscored_features


def _joint_knn(
    source_points: NDArray[np.float64],
    target_points: NDArray[np.float64],
    feat_src_z: NDArray[np.float64],
    feat_tgt_z: NDArray[np.float64],
    beta: float,
    k: int,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Find k nearest neighbors in a joint (position, feature) space.

    Positions are jointly z-scored across source and target; features (already
    z-scored by the caller) are scaled by beta. Both are concatenated into one
    vector per point before building the KDTree, so beta controls feature influence
    relative to spatial distance.

    Args:
        source_points: (N, 3) source coordinates.
        target_points: (M, 3) target coordinates.
        feat_src_z:    (N, D) z-scored source features.
        feat_tgt_z:    (M, D) z-scored target features.
        beta:          Scale of feature dimensions relative to spatial coordinates.
        k:             Number of neighbors to return per source point.

    Returns:
        Tuple (dists, nbr_idx), each of shape (N, k). Distances are rescaled back
        to physical (spatial) units.
    """
    n = len(source_points)
    all_pts  = np.vstack([source_points, target_points])
    pos_mean = all_pts.mean()
    pos_std  = all_pts.std() + 1e-8
    pos_src_z = (source_points - pos_mean) / pos_std  # (N, 3)
    pos_tar_z = (target_points - pos_mean) / pos_std  # (M, 3)

    joint_src = np.hstack([pos_src_z, beta * feat_src_z])  # (N, 3+D)
    joint_tgt = np.hstack([pos_tar_z, beta * feat_tgt_z])  # (M, 3+D)
    dists, nbr_idx = KDTree(joint_tgt).query(joint_src, k=k)    # joint dist
    dists *= pos_std                                            # rescale to physical units
    return dists.reshape(n, k), nbr_idx.reshape(n, k)


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

    source_points: NDArray[np.float64]     # (N, 3)
    target_positions: NDArray[np.float64]  # (N, 3)
    weights: NDArray[np.float64] | None = None


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
    """Hard E-step: each source point maps to its single nearest target point.

    Optionally augmented with a FeatureExtractor in 'append' mode: feature vectors
    (z-scored, scaled by ``beta``) are appended to 3D coordinates before the
    nearest-neighbor search, so the single nearest neighbor is chosen jointly on
    position and feature similarity — the hard-matching analog of GaussianMatcher's
    'append' mode. There is no 'additive' analog here: additive mode perturbs
    log-weights before a softmax over multiple candidates, which has no meaning
    when only a single nearest neighbor is kept.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor | None = None,
        beta: float = 1.0,
    ) -> None:
        """
        Args:
            feature_extractor: Optional extractor producing a (N, D) feature matrix per
                               cloud. When None, falls back to purely spatial matching.
            beta:              Scale of feature dimensions relative to spatial coordinates
                               in the joint KDTree. Only used when feature_extractor is set.
        """
        self.feature_extractor = feature_extractor
        self.beta = beta

    def match(self, source: PointCloud, target: PointCloud) -> Matching:
        """Assign each source point to its nearest neighbor in target.

        Args:
            source: PointCloud (N, 3).
            target: PointCloud (M, 3).

        Returns:
            Matching that assigns a target_point to each source point with uniform weights.
        """
        if self.feature_extractor is not None:
            feat_src_z, feat_tgt_z = zscored_features(self.feature_extractor, [source, target])
            _, nbr_idx = _joint_knn(
                source.points, target.points, feat_src_z, feat_tgt_z, self.beta, k=1,
            )
            nn_indices = nbr_idx[:, 0]
        else:
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
            feat_src_z, feat_tgt_z = zscored_features(self.feature_extractor, [source, target])

        n = len(source.points)
        if feat_src_z is not None and feat_tgt_z is not None and self.feature_mode == 'append':
            dists, nbr_idx = _joint_knn(
                source.points, target.points, feat_src_z, feat_tgt_z, self.beta, k=k,
            )
        else:
            dists, nbr_idx = KDTree(target.points).query(source.points, k=k)
            dists   = dists.reshape(n, k)                                    # ensure (N, k)
            nbr_idx = nbr_idx.reshape(n, k)

        # --- Log-weight computation ---
        log_w = -0.5 * (dists / self.sigma) ** 2                            # (N, k)

        if feat_src_z is not None and feat_tgt_z is not None and self.feature_mode == 'additive' and self.alpha != 0.0:
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
