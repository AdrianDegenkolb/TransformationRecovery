"""Point cloud trimming utilities for removing geometrically redundant points."""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from feature_extractor import FeatureExtractor
from point_cloud import PointCloud


class Clusterer(Protocol):
    """Protocol for sklearn-compatible clustering algorithms.

    Any algorithm exposing fit(X) and labels_ satisfies this protocol,
    including sklearn.cluster.DBSCAN, OPTICS, AgglomerativeClustering, etc.

    Note: the noise label -1 is only produced by density-based algorithms
    (DBSCAN, OPTICS). Partition-based algorithms (KMeans) assign every point
    to a cluster and will never produce -1 labels.
    """

    labels_: NDArray[np.int64]

    def fit(self, X: NDArray[np.float64]) -> "Clusterer":
        """Fit the clustering model to data.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Self, with labels_ populated.
        """
        ...


class Trimmer(ABC):
    """Abstract base class for point cloud trimmers.

    A trimmer reduces a point cloud by removing geometrically redundant points,
    lowering the cost of downstream matching and ICP without sacrificing accuracy.
    """

    @abstractmethod
    def trim(self, p: PointCloud) -> PointCloud:
        """Remove geometrically redundant points from a point cloud.

        Args:
            p: Input point cloud with N points.

        Returns:
            Trimmed PointCloud containing a subset of points from p.
        """
        ...


class ClusteringTrimmer(Trimmer):
    """Trims a point cloud by discarding points in large feature-space clusters.

    Points are mapped into feature space via a FeatureExtractor, then clustered.
    Clusters exceeding a size threshold are considered geometrically common
    (e.g. flat walls, uniform curvature) and their points are discarded.
    Points in small clusters and noise points (label -1) are retained, as these
    correspond to geometrically distinctive regions most useful for matching.

    Args:
        feature_extractor: Extracts per-point feature vectors of shape (N, D).
        clusterer: Clustering algorithm implementing the Clusterer protocol.
        min_cluster_fraction: Clusters whose size exceeds this fraction of total
            points are considered large and discarded. Default 0.05 (5%).
        min_cluster_size: Absolute minimum number of points for a cluster to be
            considered large. Both thresholds must be exceeded. Default 1.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        clusterer: Clusterer,
        min_cluster_fraction: float = 0.05,
        min_cluster_size: int = 1,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.clusterer = clusterer
        self.min_cluster_fraction = min_cluster_fraction
        self.min_cluster_size = min_cluster_size

    def _large_cluster_labels(self, labels: NDArray[np.int64], n_points: int) -> set[int]:
        """Return cluster labels that exceed both size thresholds.

        Args:
            labels: Per-point cluster assignment array of shape (N,).
            n_points: Total number of points in the cloud.

        Returns:
            Set of integer cluster labels considered large. Never includes -1.
        """
        unique, counts = np.unique(labels, return_counts=True)
        large: set[int] = set()
        for label, count in zip(unique, counts):
            if label == -1:
                continue
            if count / n_points > self.min_cluster_fraction and count >= self.min_cluster_size:
                large.add(int(label))
        return large

    def trim(self, p: PointCloud) -> PointCloud:
        """Remove points belonging to large feature-space clusters.

        Args:
            p: Input point cloud with N points.

        Returns:
            PointCloud containing only points not assigned to large clusters.
        """
        features = self.feature_extractor.get_features(p)
        self.clusterer.fit(features)
        labels = self.clusterer.labels_

        large = self._large_cluster_labels(labels, len(p))
        mask = ~np.isin(labels, list(large))

        return PointCloud(p.points[mask])
