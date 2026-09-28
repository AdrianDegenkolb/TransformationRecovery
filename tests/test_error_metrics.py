import numpy as np
import pytest

from error_metrics import normalized_weight_entropy
from matcher import GaussianMatcher
from point_cloud import PointCloud


def test_entropy_is_one_for_uniform_weights():
    weights = np.full((4, 5), 0.2)
    np.testing.assert_allclose(normalized_weight_entropy(weights), 1.0)


def test_entropy_is_zero_for_one_hot_weights():
    weights = np.zeros((4, 5))
    weights[:, 2] = 1.0
    np.testing.assert_allclose(normalized_weight_entropy(weights), 0.0, atol=1e-12)


def test_entropy_is_zero_for_single_neighbor():
    np.testing.assert_allclose(normalized_weight_entropy(np.ones((3, 1))), 0.0)


def test_entropy_of_two_neighbors_matches_closed_form():
    p = 0.25
    expected = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)
    assert normalized_weight_entropy(np.array([[p, 1 - p]]))[0] == pytest.approx(expected)


@pytest.mark.parametrize("sigma, expected", [(1e-3, 0.0), (1e4, 1.0)])
def test_gaussian_matcher_entropy_limits(sigma: float, expected: float):
    """Tiny sigma collapses the soft match onto the nearest neighbour; huge sigma
    spreads it evenly over all k."""
    rng = np.random.default_rng(0)
    target = PointCloud(rng.normal(size=(100, 3)) * 5.0)
    source = PointCloud(target.points + 0.5)
    weights, _ = GaussianMatcher(sigma=sigma, k=6).neighbor_weights(source, target)
    np.testing.assert_allclose(normalized_weight_entropy(weights), expected, atol=1e-6)
