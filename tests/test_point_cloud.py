import numpy as np
import pytest

from point_cloud import PointCloud


@pytest.fixture
def cloud() -> PointCloud:
    return PointCloud(np.random.default_rng(0).normal(size=(300, 3)) * 7.0 + 5.0)


def test_normalize_gives_unit_median_spacing(cloud: PointCloud):
    assert cloud.normalize().median_spacing == pytest.approx(1.0)


def test_normalize_keeps_centroid_and_shape(cloud: PointCloud):
    normalized = cloud.normalize()
    np.testing.assert_allclose(normalized.points.mean(axis=0), cloud.points.mean(axis=0))
    # Pure scaling: every point's offset from the centroid shrinks by the same factor.
    offsets = cloud.points - cloud.points.mean(axis=0)
    np.testing.assert_allclose(normalized.points - normalized.points.mean(axis=0),
                               offsets / cloud.median_spacing)


def test_normalize_returns_a_new_cloud(cloud: PointCloud):
    original = cloud.points.copy()
    cloud.normalize()
    np.testing.assert_array_equal(cloud.points, original)


def test_normalize_rejects_cloud_without_spacing():
    with pytest.raises(ValueError):
        PointCloud(np.zeros((4, 3))).normalize()
