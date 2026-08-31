import numpy as np
import pytest

from point_cloud import PointCloud
from feature_extractor import GeometricFeatureExtractor, RobustGeometricFeatureExtractor


N_FEATURES = 11  # 3 dist quantiles + centroid_offset + 4 eigen ratios + 3 angle quantiles


@pytest.fixture
def cloud() -> PointCloud:
    rng = np.random.default_rng(42)
    return PointCloud(rng.uniform(-10, 10, size=(40, 3)))


def test_output_shape_and_dtype(cloud: PointCloud) -> None:
    """get_features returns (N, 11) float64 for the default 3 quantile levels."""
    feats = RobustGeometricFeatureExtractor(k=10).get_features(cloud)
    assert feats.shape == (len(cloud.points), N_FEATURES)
    assert feats.dtype == np.float64


def test_output_shape_follows_quantile_count(cloud: PointCloud) -> None:
    """Feature dimension is 5 + 2 * len(quantiles)."""
    feats = RobustGeometricFeatureExtractor(k=10, quantiles=(0.5,)).get_features(cloud)
    assert feats.shape == (len(cloud.points), 7)


def test_no_nans(cloud: PointCloud) -> None:
    """Feature vectors must be finite for a generic random cloud."""
    feats = RobustGeometricFeatureExtractor(k=10).get_features(cloud)
    assert np.all(np.isfinite(feats))


def test_rotation_invariance(cloud: PointCloud) -> None:
    """Features must be identical after an arbitrary rotation."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1  # ensure SO(3)

    rotated = PointCloud(cloud.points @ Q.T)
    ext = RobustGeometricFeatureExtractor(k=10)

    np.testing.assert_allclose(
        ext.get_features(cloud),
        ext.get_features(rotated),
        atol=1e-10,
        err_msg="Features changed under rotation.",
    )


def test_translation_invariance(cloud: PointCloud) -> None:
    """Features must be identical after an arbitrary translation."""
    shifted = PointCloud(cloud.points + np.array([5.0, -3.0, 12.0]))
    ext = RobustGeometricFeatureExtractor(k=10)

    np.testing.assert_allclose(
        ext.get_features(cloud),
        ext.get_features(shifted),
        atol=1e-10,
        err_msg="Features changed under translation.",
    )


def test_scale_invariance(cloud: PointCloud) -> None:
    """Features must be identical after a uniform scaling of the whole cloud."""
    scaled = PointCloud(cloud.points * 7.3)
    ext = RobustGeometricFeatureExtractor(k=10)

    np.testing.assert_allclose(
        ext.get_features(cloud),
        ext.get_features(scaled),
        atol=1e-10,
        err_msg="Features changed under uniform scaling.",
    )


def test_k_clamped_to_n_minus_one() -> None:
    """k larger than N-1 must not raise; features are still (N, 11)."""
    small = PointCloud(np.eye(5, 3))
    feats = RobustGeometricFeatureExtractor(k=100).get_features(small)
    assert feats.shape == (5, N_FEATURES)
    assert np.all(np.isfinite(feats))


def test_local_density_is_distinguishable() -> None:
    """Two clusters with identical shape but different local density/scale must
    NOT collapse to the same feature vector (unlike a purely locally-normalized
    extractor), since only a single global similarity transform should be
    normalized away, not arbitrary independent local rescaling.
    """
    rng = np.random.default_rng(1)
    base = rng.uniform(-1, 1, size=(30, 3))
    tight_cluster = PointCloud(np.vstack([base + [0, 0, 0], base * 0.1 + [20, 0, 0]]))
    ext = RobustGeometricFeatureExtractor(k=10)
    feats = ext.get_features(tight_cluster)

    interior_dense = feats[:30].mean(axis=0)
    interior_sparse = feats[30:].mean(axis=0)
    assert not np.allclose(interior_dense, interior_sparse, atol=1e-6)


def _relative_change_under_neighbor_dropout(
    extractor: GeometricFeatureExtractor | RobustGeometricFeatureExtractor,
    points: np.ndarray,
    query_idx: int,
) -> float:
    """Relative feature-vector change at `query_idx` after removing its nearest neighbor."""
    dists = np.linalg.norm(points - points[query_idx], axis=1)
    nearest_idx = np.argsort(dists)[1]  # closest point other than itself

    dropout_points = np.delete(points, nearest_idx, axis=0)
    dropout_cloud = PointCloud(dropout_points)
    query_point = points[query_idx]
    dropout_query_idx = int(np.where(np.all(dropout_points == query_point, axis=1))[0][0])

    before = extractor.get_features(PointCloud(points))[query_idx]
    after = extractor.get_features(dropout_cloud)[dropout_query_idx]
    return float(np.linalg.norm(after - before) / np.linalg.norm(before))


def test_dropout_more_resilient_than_baseline_on_average(cloud: PointCloud) -> None:
    """Averaged over many query points, single-neighbor dropout should perturb
    RobustGeometricFeatureExtractor's feature vectors less (relatively) than it
    perturbs the baseline GeometricFeatureExtractor's, since the baseline's
    nearest-neighbor-distance and mean/std features are more sensitive to the
    exact identity of individual neighbors than aggregate quantiles are.

    A single query point is too noisy to assert this reliably (the two
    extractors respond to the same neighbor swap differently depending on
    local configuration), so this test averages over many points instead.
    """
    points = cloud.points
    k = 10
    baseline = GeometricFeatureExtractor(k=k)
    robust = RobustGeometricFeatureExtractor(k=k)
    query_indices = range(20)

    base_changes = [_relative_change_under_neighbor_dropout(baseline, points, i) for i in query_indices]
    robust_changes = [_relative_change_under_neighbor_dropout(robust, points, i) for i in query_indices]

    assert np.mean(robust_changes) < np.mean(base_changes)
