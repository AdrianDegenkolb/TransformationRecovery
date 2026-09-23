from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from point_cloud import PointCloud


def zscore_jointly(
    feature_matrices: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    """Z-score several feature matrices against their pooled mean and standard deviation.

    Pooled rather than per-matrix: the matrices must land in one shared feature space
    to be comparable at all, and normalising each separately would erase genuine
    differences between the clouds while pretending their scales already agree.

    Normalisation is a consumer-side concern rather than part of a FeatureExtractor's
    output contract. It exists so that whatever weights a consumer applies downstream
    — ``beta`` in append mode, ``alpha`` in additive mode, DBSCAN's ``eps`` in the
    trimmer — mean the same thing regardless of the raw feature scale. A corollary
    worth knowing: any scaling applied to features *before* this step is cancelled
    exactly by it, so per-dimension weighting has to happen afterwards.

    Args:
        feature_matrices: One (N_i, D) raw feature matrix per cloud, all sharing D.

    Returns:
        One normalised matrix per input, in the same order and with the same shapes.

    Raises:
        ValueError: If no matrix is given, or if they disagree on the feature width D.
    """
    if not feature_matrices:
        raise ValueError("zscore_jointly needs at least one feature matrix.")
    widths = {features.shape[1] for features in feature_matrices}
    if len(widths) > 1:
        raise ValueError(f"Feature matrices must share a width, got {sorted(widths)}.")

    pooled = np.concatenate(feature_matrices, axis=0)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0) + 1e-8
    return [(features - mean) / std for features in feature_matrices]


def zscored_features(
    feature_extractor: FeatureExtractor,
    point_clouds: list[PointCloud],
) -> list[NDArray[np.float64]]:
    """Extract features for several point clouds and z-score them jointly.

    Convenience wrapper for callers holding clouds rather than cached feature
    matrices. See ``zscore_jointly`` for the normalisation itself and why it is
    pooled across clouds.

    Args:
        feature_extractor: Extractor producing a (N, D) feature matrix per cloud.
        point_clouds: Clouds to extract from.

    Returns:
        One normalised feature matrix per cloud, in the same order.
    """
    return zscore_jointly([feature_extractor.get_features(cloud) for cloud in point_clouds])


def angle_pair_indices(
    k: int,
    n_angle_pairs: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Pick which neighbor pairs the pairwise-angle statistics are computed from.

    Evaluating all C(k, 2) pairs costs O(k^2) in time and peak memory, which is what
    makes a large neighborhood unaffordable. Angle statistics only need enough
    samples, so above the cap a fixed number of pairs is drawn instead.

    Pairs index neighbor *ranks*, not point identities, and come from a fixed seed.
    Both matter: rank-indexed pairs are the same set for any rigid transform of the
    cloud, and a fixed seed makes source and target use identical pairs, so the
    sampling introduces no disagreement between two clouds being matched.

    Args:
        k:             Neighborhood size actually in use (already clamped to N-1).
        n_angle_pairs: Maximum number of pairs to evaluate. When C(k, 2) does not
                       exceed it, every pair is used and the result is exact.

    Returns:
        Tuple (ti, tj) of neighbor-rank index arrays with ti != tj elementwise.
        Both empty when k < 2, i.e. when no pair exists.
    """
    if k < 2:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)

    if k * (k - 1) // 2 <= n_angle_pairs:
        return np.triu_indices(k, k=1)

    # Sampled directly rather than by subsetting triu_indices, which would itself
    # allocate O(k^2).
    rng = np.random.default_rng(0)
    ti = rng.integers(0, k, size=n_angle_pairs)
    tj = rng.integers(0, k - 1, size=n_angle_pairs)
    tj = tj + (tj >= ti)  # uniform over the k-1 ranks other than ti
    return ti.astype(np.intp), tj.astype(np.intp)


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
    target_dim: int

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
    target_dim: int = 9

    def __init__(self, k: int = 20, n_angle_pairs: int = 2000) -> None:
        """
        Args:
            k: Number of nearest neighbors used to compute local geometry.
               Must be >= 2 for pairwise angles; clamped to N-1 if necessary.
            n_angle_pairs: Cap on how many neighbor pairs the angle mean/std are
                           computed from. Below the cap every pair is used, so the
                           default k is unaffected; the cap only bounds cost and
                           memory when k is large.
        """
        self.k = k
        self.n_angle_pairs = n_angle_pairs

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
        # Only the sampled pairs are evaluated, bounding this at O(n_angle_pairs)
        # per point instead of the O(k^2) a full (N, k, k) cosine matrix would need.
        dirs = diff / (np.linalg.norm(diff, axis=2, keepdims=True) + eps)   # (N, k, 3)
        ti, tj = angle_pair_indices(k, self.n_angle_pairs)

        if len(ti) == 0:
            ang_mean = np.zeros((n, 1))
            ang_std  = np.zeros((n, 1))
        else:
            cos = np.einsum('npd,npd->np', dirs[:, ti, :], dirs[:, tj, :])  # (N, P)
            angles = np.arccos(np.clip(cos, -1.0, 1.0))                     # (N, P)
            ang_mean = angles.mean(axis=1, keepdims=True)                   # (N, 1)
            ang_std  = angles.std(axis=1, keepdims=True)                    # (N, 1)

        return np.hstack([
            feat_d_min, feat_cv, centroid_offset,
            linearity, planarity, sphericity, anisotropy,
            ang_mean, ang_std,
        ]).astype(np.float64)                                               # (N, 9)


class RobustGeometricFeatureExtractor(FeatureExtractor):
    """Similarity-invariant geometric feature extractor robust to point dropout.

    Produces ``2 * len(quantiles) + 5`` features per point (11 with the default
    three quantile levels).

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
    target_dim: int

    def __init__(
        self,
        k: int = 160,
        quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
        n_angle_pairs: int = 2000,
    ) -> None:
        """
        Args:
            k: Number of nearest neighbors used to compute local geometry.
               Must be >= 2 for pairwise angles; clamped to N-1 if necessary.
               Every feature family is estimated from the same neighborhood; larger
               k makes all of them more stable under dropout and noise, since each
               is a statistic over the neighbor set and benefits from more samples.
               The default of 160 is a compromise across cloud sizes rather than a
               per-case optimum: at N=2000 a larger k is still better, but k beyond
               roughly a third of the cloud stops discriminating between points and
               degrades sharply (catastrophically once it approaches N). 160 was the
               largest value that was near-best at every N measured without ever
               collapsing. Tune it per cloud size if N is known and far from 2000.
            quantiles: Quantile levels in [0, 1] used to summarize the neighbor
                       distance and pairwise angle distributions.
            n_angle_pairs: Cap on how many neighbor pairs the angle quantiles are
                           estimated from. Computing all C(k, 2) pairs is O(k^2) in
                           both time and peak memory, which is what made large k
                           unaffordable; quantiles need only enough samples, not
                           every pair. When C(k, 2) <= n_angle_pairs all pairs are
                           used, so small k is exact and unchanged.
        """
        self.k = k
        self.quantiles = quantiles
        self.n_angle_pairs = n_angle_pairs
        # One dist quantile + one angle quantile per level, plus centroid_offset
        # and the four eigenvalue-ratio shape features.
        self.target_dim = 2 * len(quantiles) + 5

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
        # Only the sampled pairs are evaluated, so this costs O(n_angle_pairs) per
        # point instead of the O(k^2) a full (N, k, k) cosine matrix would need.
        dirs = diff / (np.linalg.norm(diff, axis=2, keepdims=True) + eps)   # (N, k, 3)
        ti, tj = angle_pair_indices(k, self.n_angle_pairs)

        if len(ti) == 0:
            angle_q = np.zeros((n, len(self.quantiles)))
        else:
            cos = np.einsum('npd,npd->np', dirs[:, ti, :], dirs[:, tj, :])  # (N, P)
            angles = np.arccos(np.clip(cos, -1.0, 1.0))                     # (N, P)
            angle_q = np.quantile(angles, self.quantiles, axis=1).T         # (N, Q)

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

    target_dim: int = -1
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
