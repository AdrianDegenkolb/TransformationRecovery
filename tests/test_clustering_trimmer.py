import numpy as np
import pytest
from numpy.typing import NDArray

from point_cloud import PointCloud
from feature_extractor import FeatureExtractor
from trimmer import ClusteringTrimmer


class _IdentityExtractor(FeatureExtractor):
    """Feature extractor that returns each point's own coordinates.

    Keeps clustering tests decoupled from GeometricFeatureExtractor's k-NN
    behavior; only ClusteringTrimmer's own logic is under test here.
    """

    def get_features(self, p: PointCloud) -> NDArray[np.float64]:
        return p.points.astype(np.float64)


class _FakeClusterer:
    """Deterministic stand-in for sklearn's DBSCAN.

    Labels are supplied up front (one label per point, in fit() call order),
    so tests can control exactly which points end up in which cluster without
    depending on DBSCAN's eps/min_samples tuning.
    """

    def __init__(self, labels: NDArray[np.int64]):
        self._labels = labels
        self.labels_: NDArray[np.int64] | None = None

    def fit(self, X: NDArray[np.float64]) -> "_FakeClusterer":
        assert len(X) == len(self._labels), "fake clusterer received unexpected number of points"
        self.labels_ = self._labels
        return self


def _cloud(n: int, offset: float = 0.0) -> PointCloud:
    rng = np.random.default_rng(0)
    return PointCloud(rng.uniform(-1, 1, size=(n, 3)) + offset)


def test_entirely_large_cluster_collapses_to_centroid() -> None:
    """A single cloud whose points are all one large cluster is replaced by its centroid."""
    cloud = _cloud(10)
    clusterer = _FakeClusterer(np.zeros(10, dtype=np.int64))
    trimmer = ClusteringTrimmer(_IdentityExtractor(), clusterer, min_cluster_fraction=0.5, min_points=1)

    result = trimmer.trim(cloud)

    assert len(result) == 1
    np.testing.assert_allclose(result.points[0], cloud.points.mean(axis=0))


def test_noise_only_cloud_is_unchanged() -> None:
    """A cloud entirely labeled as noise (-1) has nothing removed."""
    cloud = _cloud(10)
    clusterer = _FakeClusterer(-np.ones(10, dtype=np.int64))
    trimmer = ClusteringTrimmer(_IdentityExtractor(), clusterer, min_cluster_fraction=0.5)

    result = trimmer.trim(cloud)

    assert len(result) == len(cloud)
    np.testing.assert_allclose(np.sort(result.points, axis=0), np.sort(cloud.points, axis=0))


def test_small_cluster_below_fraction_threshold_is_retained() -> None:
    """A cluster below min_cluster_fraction is kept pointwise, not collapsed."""
    cloud = _cloud(10)
    # 3/10 = 0.3, below the 0.5 threshold -> not "large".
    labels = np.array([0, 0, 0] + [-1] * 7, dtype=np.int64)
    clusterer = _FakeClusterer(labels)
    trimmer = ClusteringTrimmer(_IdentityExtractor(), clusterer, min_cluster_fraction=0.5)

    result = trimmer.trim(cloud)

    assert len(result) == len(cloud)


def test_multi_cloud_clustering_is_joint_not_per_cloud() -> None:
    """Two clouds passed together are clustered on their concatenated features,
    so 'large' is determined from pooled counts and both clouds agree on it —
    not on two independent per-cloud fits that could disagree.
    """
    cloud_a = _cloud(6, offset=0.0)
    cloud_b = _cloud(4, offset=100.0)  # disjoint in space, but that's irrelevant to the fake clusterer

    # Joint label array as the fake clusterer will receive it: cloud_a's 6 points
    # then cloud_b's 4 points. Label 0 spans both clouds (4 from A, 2 from B) and
    # is 6/10 = 0.6 of the pooled total -> large under a 0.5 threshold, even
    # though it is only 4/6 = 0.67 of A and 2/4 = 0.5 of B individually.
    joint_labels = np.array([0, 0, 0, 0, -1, -1] + [0, 0, -1, -1], dtype=np.int64)
    clusterer = _FakeClusterer(joint_labels)
    trimmer = ClusteringTrimmer(_IdentityExtractor(), clusterer, min_cluster_fraction=0.5, min_points=1)

    result_a, result_b = trimmer.trim([cloud_a, cloud_b])

    # Cluster 0 collapses to one centroid per cloud; the 2 noise points per cloud survive.
    assert len(result_a) == 1 + 2
    assert len(result_b) == 1 + 2


def test_large_cluster_absent_from_one_cloud_does_not_crash() -> None:
    """A cluster that is globally large but has zero points in one specific cloud
    must be skipped for that cloud instead of producing a NaN centroid.
    """
    cloud_a = _cloud(10, offset=0.0)   # entirely label 0
    cloud_b = _cloud(10, offset=50.0)  # entirely label 1

    joint_labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    clusterer = _FakeClusterer(joint_labels)
    # Each label is 10/20 = 0.5 of the pooled total -> both "large" under a 0.4 threshold.
    trimmer = ClusteringTrimmer(_IdentityExtractor(), clusterer, min_cluster_fraction=0.4, min_points=1)

    result_a, result_b = trimmer.trim([cloud_a, cloud_b])

    assert len(result_a) == 1
    assert len(result_b) == 1
    assert np.all(np.isfinite(result_a.points))
    assert np.all(np.isfinite(result_b.points))
    np.testing.assert_allclose(result_a.points[0], cloud_a.points.mean(axis=0))
    np.testing.assert_allclose(result_b.points[0], cloud_b.points.mean(axis=0))


def test_large_cluster_labels_respects_both_thresholds() -> None:
    """_large_cluster_labels requires both the fraction and absolute-size thresholds."""
    trimmer = ClusteringTrimmer(
        _IdentityExtractor(), _FakeClusterer(np.zeros(1, dtype=np.int64)),
        min_cluster_fraction=0.2, min_cluster_size=5,
    )
    # label 0: 3/10 = 0.3 > 0.2 fraction, but count=3 < min_cluster_size=5 -> not large.
    # label 1: 4/10 = 0.4 > 0.2 fraction, count=4 < 5 -> not large.
    # label -1: noise, always excluded regardless of size.
    labels = np.array([0, 0, 0, 1, 1, 1, 1, -1, -1, -1], dtype=np.int64)

    large = trimmer._large_cluster_labels(labels, n_points=10)

    assert large == set()
