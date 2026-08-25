import numpy as np
import pytest

from point_cloud import PointCloud
from feature_extractor import GeometricFeatureExtractor


N_FEATURES = 9


@pytest.fixture
def cloud() -> PointCloud:
    rng = np.random.default_rng(42)
    return PointCloud(rng.uniform(-10, 10, size=(40, 3)))


def test_output_shape_and_dtype(cloud: PointCloud) -> None:
    """get_features returns (N, 9) float64."""
    feats = GeometricFeatureExtractor(k=10).get_features(cloud)
    assert feats.shape == (len(cloud.points), N_FEATURES)
    assert feats.dtype == np.float64


def test_no_nans(cloud: PointCloud) -> None:
    """Feature vectors must be finite for a generic random cloud."""
    feats = GeometricFeatureExtractor(k=10).get_features(cloud)
    assert np.all(np.isfinite(feats))


def test_rotation_invariance(cloud: PointCloud) -> None:
    """Features must be identical after an arbitrary rotation."""
    rng = np.random.default_rng(0)
    # Random rotation via QR decomposition
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1  # ensure SO(3)

    rotated = PointCloud(cloud.points @ Q.T)
    ext = GeometricFeatureExtractor(k=10)

    np.testing.assert_allclose(
        ext.get_features(cloud),
        ext.get_features(rotated),
        atol=1e-10,
        err_msg="Features changed under rotation.",
    )


def test_translation_invariance(cloud: PointCloud) -> None:
    """Features must be identical after an arbitrary translation."""
    shifted = PointCloud(cloud.points + np.array([5.0, -3.0, 12.0]))
    ext = GeometricFeatureExtractor(k=10)

    np.testing.assert_allclose(
        ext.get_features(cloud),
        ext.get_features(shifted),
        atol=1e-10,
        err_msg="Features changed under translation.",
    )


def test_scale_invariance(cloud: PointCloud) -> None:
    """Features must be identical after uniform scaling."""
    scaled = PointCloud(cloud.points * 7.3)
    ext = GeometricFeatureExtractor(k=10)

    np.testing.assert_allclose(
        ext.get_features(cloud),
        ext.get_features(scaled),
        atol=1e-10,
        err_msg="Features changed under uniform scaling.",
    )


def test_k_clamped_to_n_minus_one() -> None:
    """k larger than N-1 must not raise; features are still (N, 9)."""
    small = PointCloud(np.eye(5, 3))
    feats = GeometricFeatureExtractor(k=100).get_features(small)
    assert feats.shape == (5, N_FEATURES)
    assert np.all(np.isfinite(feats))
