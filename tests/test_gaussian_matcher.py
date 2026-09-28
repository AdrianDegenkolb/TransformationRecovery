import numpy as np
import pytest

from matcher import GaussianMatcher
from point_cloud import PointCloud


@pytest.fixture
def clouds() -> tuple[PointCloud, PointCloud]:
    rng = np.random.default_rng(0)
    target = PointCloud(rng.normal(size=(200, 3)) * 5.0)
    source = PointCloud(target.points + rng.normal(scale=0.3, size=target.points.shape))
    return source, target


def test_neighbor_weights_rows_are_distributions(clouds):
    source, target = clouds
    weights, nbr_idx = GaussianMatcher(sigma=1.0, k=7).neighbor_weights(source, target)
    assert weights.shape == nbr_idx.shape == (200, 7)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_match_averages_targets_with_the_exposed_neighbor_weights(clouds):
    """neighbor_weights must describe exactly the distribution match() averages over."""
    source, target = clouds
    matcher = GaussianMatcher(sigma=1.0, k=7)
    weights, nbr_idx = matcher.neighbor_weights(source, target)
    expected = (weights[:, :, None] * target.points[nbr_idx]).sum(axis=1)
    np.testing.assert_allclose(matcher.match(source, target).target_positions, expected)


def test_neighbor_weights_k_is_capped_at_target_size():
    target = PointCloud(np.eye(3))
    weights, _ = GaussianMatcher(sigma=1.0, k=10).neighbor_weights(target, target)
    assert weights.shape == (3, 3)
