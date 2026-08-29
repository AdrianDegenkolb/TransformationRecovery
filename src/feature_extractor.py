from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from point_cloud import PointCloud

def zscored_features(
    feature_extractor: FeatureExtractor,
    source: PointCloud,
    target: PointCloud,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute z-scored feature matrices for source and target using target statistics.

    Z-scoring uses target mean and std so that beta (append mode) and alpha (additive
    mode) are interpretable regardless of the raw feature scale.

    Args:
        feature_extractor: Extractor producing a (N, D) feature matrix per cloud.
        source: Source point cloud.
        target: Target point cloud.

    Returns:
        Tuple (feat_src_z, feat_tgt_z), each of shape (N, D) and (M, D) respectively.
    """
    feat_src = feature_extractor.get_features(source)   # (N, D)
    feat_tgt = feature_extractor.get_features(target)   # (M, D)
    feat_mean = feat_tgt.mean(axis=0)
    feat_std  = feat_tgt.std(axis=0) + 1e-8
    return (feat_src - feat_mean) / feat_std, (feat_tgt - feat_mean) / feat_std


class FeatureExtractor(ABC):
    """Base class for per-point geometric feature extractors.

    Implementations must return one feature vector per point.
    Feature vectors should be invariant to the transformation being recovered
    (e.g. rotation, translation, and potentially scale).
    """

    @abstractmethod
    def get_features(self, p: PointCloud) -> NDArray[np.float64]:
        """Compute a feature vector for each point in the cloud.

        Args:
            p: Input point cloud with N points.

        Returns:
            Float64 array of shape (N, D) where D is the feature dimension (can be selected arbitrarily).
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

    All features are invariant under similarity transformations (rotation, translation,
    uniform scale). Features are computed once per cloud and can be cached across ICP
    iterations.
    """

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

        ang_mean = angles.mean(axis=1, keepdims=True)                       # (N, 1)
        ang_std  = angles.std(axis=1, keepdims=True)                        # (N, 1)

        return np.hstack([
            feat_d_min, feat_cv, centroid_offset,
            linearity, planarity, sphericity, anisotropy,
            ang_mean, ang_std,
        ]).astype(np.float64)                                               # (N, 9)


class IdentityFeatureExtractor(FeatureExtractor):
    """Oracle feature extractor that returns the N×N identity matrix.

    Point i receives feature vector e_i (the i-th standard basis vector).
    Cosine similarity between point i in source and point i in target is 1;
    between any two distinct indices it is 0.

    This gives a GaussianMatcher with alpha > 0 a perfect correspondence
    signal, useful for validating the feature integration before any real
    features are implemented.

    Warning:
        Only meaningful when source and target have the same N and the
        correspondence is i↔i (true for all synthetic experiments).
        Not suitable for large clouds: feature dimension D=N causes
        high memory usage and slow cosine similarity computation.
    """

    def get_features(self, p: PointCloud) -> NDArray[np.float64]:
        """Return the N×N identity matrix as feature matrix.

        Args:
            p: Input point cloud with N points.

        Returns:
            Identity matrix of shape (N, N) and dtype float64.
        """
        n = len(p.points)
        return np.eye(n, dtype=np.float64)
