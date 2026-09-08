from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from point_cloud import PointCloud
from feature_extractor import FeatureExtractor, zscored_features


def _joint_zscore(
    feat_src: NDArray[np.float64],
    feat_tgt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Z-score two feature matrices jointly using their pooled mean and std.

    Mirrors the normalisation performed by ``zscored_features`` so that cached
    raw features can be re-scored on demand without calling the extractor again.

    Args:
        feat_src: Raw source feature matrix of shape (N, D).
        feat_tgt: Raw target feature matrix of shape (M, D).

    Returns:
        Tuple ``(feat_src_z, feat_tgt_z)`` normalised with the pooled statistics.
    """
    all_raw = np.concatenate([feat_src, feat_tgt], axis=0)
    mean = all_raw.mean(axis=0)
    std  = all_raw.std(axis=0) + 1e-8
    return (feat_src - mean) / std, (feat_tgt - mean) / std


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

    def prepare(self, source: PointCloud, target: PointCloud) -> None:
        """Pre-compute and cache feature vectors before the ICP loop.

        Called once by ``ICP.fit()`` with the initial source cloud and the fixed
        target cloud before the iteration loop begins.  Subclasses that use a
        ``FeatureExtractor`` should override this method to:

        - Always cache target features (target never changes during ICP).
        - Cache source features only when
          ``feature_extractor.is_transformation_invariant`` is ``True``; otherwise
          source features must be recomputed from the current (transformed) cloud on
          every ``match()`` call.

        The default implementation is a no-op, which preserves the previous
        behaviour of computing features live on every ``match()`` call.

        Args:
            source: Initial source PointCloud (N, 3) before any ICP iteration.
            target: Fixed target PointCloud (M, 3).
        """

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
        self._feat_src_z: NDArray[np.float64] | None = None
        self._feat_tgt_z: NDArray[np.float64] | None = None
        self._feat_tgt_raw: NDArray[np.float64] | None = None
        self._prepared: bool = False

    def prepare(self, source: PointCloud, target: PointCloud) -> None:
        """Cache feature vectors for source and/or target before the ICP loop.

        Always caches target features. Caches source features only when the
        extractor declares ``is_transformation_invariant = True``, in which case
        ``match()`` skips ``get_features()`` entirely on every subsequent call.
        When only target is cached, ``match()`` still recomputes source features
        each iteration but avoids the target extraction cost.

        Args:
            source: Initial source PointCloud (N, 3).
            target: Fixed target PointCloud (M, 3).
        """
        if self.feature_extractor is None:
            return
        feat_tgt_raw = self.feature_extractor.get_features(target)
        if self.feature_extractor.is_transformation_invariant:
            feat_src_raw = self.feature_extractor.get_features(source)
            self._feat_src_z, self._feat_tgt_z = _joint_zscore(feat_src_raw, feat_tgt_raw)
            self._prepared = True
        else:
            self._feat_tgt_raw = feat_tgt_raw

    def match(self, source: PointCloud, target: PointCloud) -> Matching:
        """Assign each source point to its nearest neighbor in target.

        Uses cached feature vectors from ``prepare()`` when available.

        Args:
            source: PointCloud (N, 3).
            target: PointCloud (M, 3).

        Returns:
            Matching that assigns a target_point to each source point with uniform weights.
        """
        if self.feature_extractor is not None:
            if self._prepared:
                feat_src_z, feat_tgt_z = self._feat_src_z, self._feat_tgt_z
            elif self._feat_tgt_raw is not None:
                feat_src_raw = self.feature_extractor.get_features(source)
                feat_src_z, feat_tgt_z = _joint_zscore(feat_src_raw, self._feat_tgt_raw)
            else:
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
        self._feat_src_z: NDArray[np.float64] | None = None
        self._feat_tgt_z: NDArray[np.float64] | None = None
        self._feat_tgt_raw: NDArray[np.float64] | None = None
        self._prepared: bool = False

    def prepare(self, source: PointCloud, target: PointCloud) -> None:
        """Cache feature vectors for source and/or target before the ICP loop.

        Always caches target features. Caches source features only when the
        extractor declares ``is_transformation_invariant = True``, in which case
        ``match()`` skips ``get_features()`` entirely on every subsequent call.
        When only target is cached, ``match()`` still recomputes source features
        each iteration but avoids the target extraction cost.

        Args:
            source: Initial source PointCloud (N, 3).
            target: Fixed target PointCloud (M, 3).
        """
        if self.feature_extractor is None:
            return
        feat_tgt_raw = self.feature_extractor.get_features(target)
        if self.feature_extractor.is_transformation_invariant:
            feat_src_raw = self.feature_extractor.get_features(source)
            self._feat_src_z, self._feat_tgt_z = _joint_zscore(feat_src_raw, feat_tgt_raw)
            self._prepared = True
        else:
            self._feat_tgt_raw = feat_tgt_raw

    def match(self, source: PointCloud, target: PointCloud) -> Matching:
        """Soft Gaussian correspondence from source to target.

        Uses cached feature vectors from ``prepare()`` when available.

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
            if self._prepared:
                feat_src_z, feat_tgt_z = self._feat_src_z, self._feat_tgt_z
            elif self._feat_tgt_raw is not None:
                feat_src_raw = self.feature_extractor.get_features(source)
                feat_src_z, feat_tgt_z = _joint_zscore(feat_src_raw, self._feat_tgt_raw)
            else:
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
