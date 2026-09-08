from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from point_cloud import PointCloud


def zscored_features(
    feature_extractor: FeatureExtractor,
    point_clouds: list[PointCloud],
) -> list[NDArray[np.float64]]:
    """Compute z-scored feature matrices for a list of point clouds using aggregated statistics.

    Z-scoring uses target mean and std so that beta (append mode) and alpha (additive
    mode) are interpretable regardless of the raw feature scale.

    Args:
        feature_extractor: Extractor producing a (N, D) feature matrix per cloud.
        point_clouds: a list of point cloud.

    Returns:
        List of normalized feature arrays
    """
    features_per_point_cloud = [feature_extractor.get_features(cloud) for cloud in point_clouds]
    all_features = np.concatenate(features_per_point_cloud, axis=0)
    all_features_mean = all_features.mean(axis=0)
    all_features_std = all_features.std(axis=0) + 1e-8
    normalized_features = [(features - all_features_mean) / all_features_std for features in features_per_point_cloud]
    return normalized_features


class FeatureExtractor(ABC):
    """Base class for per-point geometric feature extractors.

    Implementations must return one feature vector per point.

    Subclasses must set the class variable ``is_transformation_invariant``:

    - ``True``: ``get_features()`` returns bit-identical results for any rigid
      transform (rotation, translation, uniform scale) applied to the input cloud.
      Instead of recomputing features repeatedly, feateures can be computed once and
      then be cached.  Non-invariant extractors will produce stale features after the first iteration.

    - ``False``: features depend on the absolute positions of the points. Whenever the PointCloud is
      transformed, features must be recomputed.

    Known caveat: k-NN tie-breaking on regular grid clouds (``lattice``,
    ``2d-lattice``) causes small numerical non-invariance (~1e-2 MAE) even for
    extractors that declare ``is_transformation_invariant = True``, because
    near-equal inter-point distances can swap neighbour rank order after a rotation.
    Benchmark measurements show no observable impact on matching quality, so caching
    remains safe in practice for these styles.
    """

    is_transformation_invariant: bool

    @abstractmethod
    def get_features(self, p: PointCloud) -> NDArray[np.float64]:
        """Compute a feature vector for each point in the cloud.

        Implementations with ``is_transformation_invariant = True`` must return
        the same array (up to floating-point precision) regardless of any rigid
        transform applied to ``p`` before calling this method.

        Args:
            p: Input point cloud with N points.

        Returns:
            Float64 array of shape (N, D) where D is the feature dimension.
        """
        ...


class GeometricFeatureExtractor(FeatureExtractor):
    """Similarity-invariant geometric feature extractor (9-dimensional).

    For each point p with k nearest neighbors at distances d_1 ≤ ... ≤ d_k:

    - d_1 / d_bar          — normalized nearest-neighbor distance
    - std(d) / d_bar       — coefficient of variation (regularity)
    - ||p - centroid|| / d_bar — normalized centroid offset (eccentricity)
    - linearity            — (λ1 - λ2) / λ1
    - planarity            — (λ2 - λ3) / λ1
    - sphericity           — λ3 / λ1
    - anisotropy           — (λ1 - λ3) / λ1
    - mean pairwise angle  — mean of angles between neighbor direction vectors
    - std pairwise angle   — std of angles between neighbor direction vectors

    All features are ratios or angles derived from local k-NN geometry and are
    invariant under similarity transformations (rotation, translation, uniform scale).
    ``Matcher.prepare()`` caches these features once before the ICP loop.
    """

    is_transformation_invariant: bool = True

    def __init__(self, k: int = 20) -> None:
        """
        Args:
            k: Number of nearest neighbors used to compute local geometry.
               Must be >= 2 for pairwise angles; clamped to N-1 if necessary.
        """
        self.k = k

    def get_features(self, p: PointCloud) -> NDArray[np.float64]:
        """Compute 9-dimensional geometric feature vectors for all points.

        Args:
            p: Input point cloud with N points (N >= 2).

        Returns:
            Float64 array of shape (N, 9).
        """
        points = p.points                                                   # (N, 3)
        n = len(points)
        k = min(self.k, n - 1)
        eps = 1e-8

        # k-NN excluding self (query k+1, drop index 0 which is the point itself)
        _, idx = KDTree(points).query(points, k=k + 1)
        idx = idx[:, 1:]                                                    # (N, k)
        nbr_pts = points[idx]                                               # (N, k, 3)

        # --- Distance-ratio features ---
        diff = nbr_pts - points[:, None, :]                                 # (N, k, 3)
        dists = np.linalg.norm(diff, axis=2)                                # (N, k)
        d_bar = dists.mean(axis=1, keepdims=True)                           # (N, 1)

        feat_d_min  = dists[:, :1] / (d_bar + eps)                          # (N, 1)
        feat_cv     = dists.std(axis=1, keepdims=True) / (d_bar + eps)      # (N, 1)
        centroid_offset = np.linalg.norm(
            points - nbr_pts.mean(axis=1), axis=1, keepdims=True
        ) / (d_bar + eps)                                                   # (N, 1)

        # --- PCA eigenvalue ratios ---
        cov = np.einsum('nki,nkj->nij', diff, diff) / k   # (N, 3, 3)
        eigvals = np.linalg.eigvalsh(cov)                                   # (N, 3) ascending
        lam1 = eigvals[:, 2:3]                                              # (N, 1) largest
        lam2 = eigvals[:, 1:2]
        lam3 = eigvals[:, 0:1]                                              # (N, 1) smallest
        l1 = np.maximum(lam1, eps)

        linearity  = (lam1 - lam2) / l1                                     # (N, 1)
        planarity  = (lam2 - lam3) / l1                                     # (N, 1)
        sphericity = lam3 / l1                                              # (N, 1)
        anisotropy = (lam1 - lam3) / l1                                     # (N, 1)

        # --- Pairwise angles between neighbor direction vectors ---
        dirs = diff / (np.linalg.norm(diff, axis=2, keepdims=True) + eps)   # (N, k, 3)
        cos_mat = np.einsum('nid,njd->nij', dirs, dirs)   # (N, k, k)
        cos_mat = np.clip(cos_mat, -1.0, 1.0)
        ti, tj = np.triu_indices(k, k=1)
        angles = np.arccos(cos_mat[:, ti, tj])                              # (N, n_pairs)

        if angles.shape[1] == 0:
            ang_mean = np.zeros((n, 1))
            ang_std  = np.zeros((n, 1))
        else:
            ang_mean = angles.mean(axis=1, keepdims=True)                   # (N, 1)
            ang_std  = angles.std(axis=1, keepdims=True)                    # (N, 1)

        return np.hstack([
            feat_d_min, feat_cv, centroid_offset,
            linearity, planarity, sphericity, anisotropy,
            ang_mean, ang_std,
        ]).astype(np.float64)                                               # (N, 9)


class RobustGeometricFeatureExtractor(FeatureExtractor):
    """Similarity-invariant geometric feature extractor robust to point dropout (11-dimensional).

    Refines GeometricFeatureExtractor in three ways, all aimed at keeping feature
    vectors similar for points in similar geometric context while being resilient
    to dropout (random loss of individual neighbor points, e.g. sensor misses):

    - Neighbor distances and pairwise angles are summarized by quantiles instead
      of single order statistics (nearest-neighbor distance) or raw moments
      (mean/std). Quantiles are aggregate statistics over all k neighbors (or all
      C(k, 2) pairs), so losing any one neighbor perturbs them only slightly, and
      they capture distribution shape (e.g. a bimodal vs. a spread-out angle
      distribution) that mean/std cannot.
    - Distances are normalized by one scale computed over the whole cloud
      (median of each point's local mean neighbor distance) instead of each
      point's own local mean neighbor distance. This keeps the descriptor
      invariant to the single global similarity transform being recovered,
      while preserving relative density differences between distinct regions
      of the same cloud as a discriminative signal.
    - The PCA covariance used for linearity/planarity/sphericity/anisotropy is
      computed about the neighbor centroid rather than about the query point,
      so these features describe pure local shape instead of being mixed with
      the point's offset within its own neighborhood (already captured
      separately by centroid_offset).

    For each point p with k nearest neighbors at distances d_1 ≤ ... ≤ d_k and
    the C(k, 2) pairwise angles between neighbor direction vectors:

    - dist quantiles (Q25, Q50, Q75 by default) — d_i / scale
    - centroid_offset       — ||p - neighbor_centroid|| / scale
    - linearity             — (λ1 - λ2) / λ1
    - planarity             — (λ2 - λ3) / λ1
    - sphericity            — λ3 / λ1
    - anisotropy            — (λ1 - λ3) / λ1
    - angle quantiles (Q25, Q50, Q75 by default) — pairwise neighbor-direction angles

    where scale is the median, over all points in the cloud, of each point's
    mean neighbor distance. All features are invariant under similarity
    transformations (rotation, translation, uniform scale) applied to the
    whole cloud. ``Matcher.prepare()`` caches these features once before the
    ICP loop.
    """

    is_transformation_invariant: bool = True

    def __init__(self, k: int = 20, quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)) -> None:
        """
        Args:
            k: Number of nearest neighbors used to compute local geometry.
               Must be >= 2 for pairwise angles; clamped to N-1 if necessary.
            quantiles: Quantile levels in [0, 1] used to summarize the neighbor
                       distance and pairwise angle distributions.
        """
        self.k = k
        self.quantiles = quantiles

    def get_features(self, p: PointCloud) -> NDArray[np.float64]:
        """Compute geometric feature vectors for all points.

        Args:
            p: Input point cloud with N points (N >= 2).

        Returns:
            Float64 array of shape (N, 5 + 2 * len(quantiles)).
        """
        points = p.points                                                   # (N, 3)
        n = len(points)
        k = min(self.k, n - 1)
        eps = 1e-8

        # k-NN excluding self (query k+1, drop index 0 which is the point itself)
        _, idx = KDTree(points).query(points, k=k + 1)
        idx = idx[:, 1:]                                                    # (N, k)
        nbr_pts = points[idx]                                               # (N, k, 3)

        # --- Point-relative geometry (distances, directions, centroid offset) ---
        diff = nbr_pts - points[:, None, :]                                 # (N, k, 3)
        dists = np.linalg.norm(diff, axis=2)                                # (N, k)
        d_bar = dists.mean(axis=1)                                          # (N,)
        scale = np.median(d_bar) + eps                                      # scalar, shared across the cloud

        dist_q = np.quantile(dists, self.quantiles, axis=1).T / scale       # (N, Q)

        nbr_centroid = nbr_pts.mean(axis=1)                                 # (N, 3)
        centroid_offset = np.linalg.norm(
            points - nbr_centroid, axis=1, keepdims=True
        ) / scale                                                           # (N, 1)

        # --- PCA eigenvalue ratios (covariance about the neighbor centroid) ---
        diff_c = nbr_pts - nbr_centroid[:, None, :]                         # (N, k, 3)
        cov = np.einsum('nki,nkj->nij', diff_c, diff_c) / k                 # (N, 3, 3)
        eigvals = np.linalg.eigvalsh(cov)                                   # (N, 3) ascending
        lam1 = eigvals[:, 2:3]                                              # (N, 1) largest
        lam2 = eigvals[:, 1:2]
        lam3 = eigvals[:, 0:1]                                              # (N, 1) smallest
        l1 = np.maximum(lam1, eps)

        linearity  = (lam1 - lam2) / l1                                     # (N, 1)
        planarity  = (lam2 - lam3) / l1                                     # (N, 1)
        sphericity = lam3 / l1                                              # (N, 1)
        anisotropy = (lam1 - lam3) / l1                                     # (N, 1)

        # --- Pairwise angles between (point-relative) neighbor direction vectors ---
        dirs = diff / (np.linalg.norm(diff, axis=2, keepdims=True) + eps)   # (N, k, 3)
        cos_mat = np.einsum('nid,njd->nij', dirs, dirs)   # (N, k, k)
        cos_mat = np.clip(cos_mat, -1.0, 1.0)
        ti, tj = np.triu_indices(k, k=1)
        angles = np.arccos(cos_mat[:, ti, tj])                              # (N, n_pairs)

        if angles.shape[1] == 0:
            angle_q = np.zeros((n, len(self.quantiles)))
        else:
            angle_q = np.quantile(angles, self.quantiles, axis=1).T  # (N, Q)

        return np.hstack([
            dist_q, centroid_offset,
            linearity, planarity, sphericity, anisotropy,
            angle_q,
        ]).astype(np.float64)                                               # (N, 5 + 2*len(quantiles))


class IdentityFeatureExtractor(FeatureExtractor):
    """Oracle feature extractor that returns the N×N identity matrix.

    Point i receives feature vector e_i (the i-th standard basis vector).
    Cosine similarity between point i in source and point i in target is 1;
    between any two distinct indices it is 0.

    This gives a GaussianMatcher with alpha > 0 a perfect correspondence
    signal, useful for validating the feature integration before any real
    features are implemented.

    ``is_transformation_invariant`` is ``False``: although ``get_features()``
    always returns ``eye(N)`` regardless of point positions, the oracle
    semantics (point i matches target point i) are only valid for the
    initial unrotated source.  Setting this to ``False`` prevents
    ``Matcher.prepare()`` from caching source features across iterations.

    Warning:
        Only meaningful when source and target have the same N and the
        correspondence is i↔i (true for all synthetic experiments).
        Not suitable for large clouds: feature dimension D=N causes
        high memory usage and slow cosine similarity computation.
    """

    is_transformation_invariant: bool = False

    def get_features(self, p: PointCloud) -> NDArray[np.float64]:
        """Return the N×N identity matrix as feature matrix.

        Args:
            p: Input point cloud with N points.

        Returns:
            Identity matrix of shape (N, N) and dtype float64.
        """
        n = len(p.points)
        return np.eye(n, dtype=np.float64)
