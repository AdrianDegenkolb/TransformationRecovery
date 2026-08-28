import numpy as np
import pytest

from point_cloud import PointCloud
from feature_extractor import IdentityFeatureExtractor
from matcher import GaussianMatcher, NearestNeighborMatcher


@pytest.fixture
def small_cloud() -> PointCloud:
    """20-point random cloud with fixed seed."""
    rng = np.random.default_rng(0)
    return PointCloud(rng.uniform(-10, 10, size=(20, 3)))


def test_identity_extractor_shape(small_cloud: PointCloud) -> None:
    """get_features returns (N, N) float64 identity matrix."""
    extractor = IdentityFeatureExtractor()
    features = extractor.get_features(small_cloud)
    n = len(small_cloud.points)
    assert features.shape == (n, n)
    assert features.dtype == np.float64
    np.testing.assert_array_equal(features, np.eye(n))


def test_gaussian_matcher_weights_correct_correspondence(small_cloud: PointCloud) -> None:
    """With IdentityFeatureExtractor and large alpha, the highest-weight candidate
    for each source point i should be target point i (the true correspondence).

    Target == source, so the spatial distance from point i to its own position is 0
    and cosine similarity to e_i is 1. With alpha=10 the feature term strongly
    dominates ties among spatially equidistant candidates.
    """
    matcher = GaussianMatcher(
        sigma=5.0,
        k=10,
        feature_extractor=IdentityFeatureExtractor(),
        alpha=50.0,
    )
    matching = matcher.match(small_cloud, small_cloud)

    # Recover the index of the highest-weight neighbor for each source point.
    # We need the raw per-neighbor weights; re-derive them from the match internals
    # by checking that target_positions == source_points (perfect self-match).
    np.testing.assert_allclose(
        matching.target_positions,
        small_cloud.points,
        atol=1e-6,
        err_msg="With identity features and alpha=10, each point should map to itself.",
    )


def test_append_mode_weights_by_joint_distance(small_cloud: PointCloud) -> None:
    """Append mode weights by joint distance, so false matches pay a large feature
    penalty (beta * ||feat_z[i] - feat_z[j]||) that drives their weight to ~0.

    With source == target, the true match for point i has joint distance = 0 while
    every other candidate has joint distance >= beta * feature_gap > 0. Even with
    k=10 and large sigma, the true match dominates and target_positions ≈ source_points.
    """
    matcher = GaussianMatcher(
        sigma=5.0,
        k=10,
        feature_extractor=IdentityFeatureExtractor(),
        feature_mode='append',
        beta=50.0,
    )
    matching = matcher.match(small_cloud, small_cloud)

    np.testing.assert_allclose(
        matching.target_positions,
        small_cloud.points,
        atol=1e-6,
        err_msg="In append mode, joint-distance weighting must recover the self-match even with k>1.",
    )


def test_nearest_neighbor_matcher_append_mode_recovers_correspondence(small_cloud: PointCloud) -> None:
    """NearestNeighborMatcher in append mode picks candidates by joint (position,
    feature) distance. With source == target, a slight positional offset makes the
    plain spatial nearest neighbor ambiguous/wrong for close points, but the identity
    feature term (scaled by a large beta) should still recover the true index-i match.
    """
    rng = np.random.default_rng(1)
    target = PointCloud(small_cloud.points + rng.uniform(-0.05, 0.05, size=small_cloud.points.shape))

    matcher = NearestNeighborMatcher(feature_extractor=IdentityFeatureExtractor(), beta=1000.0)
    matching = matcher.match(small_cloud, target)

    np.testing.assert_allclose(
        matching.target_positions,
        target.points,
        atol=1e-6,
        err_msg="With identity features and large beta, each point should map to its own index in target.",
    )


def test_nearest_neighbor_matcher_without_extractor_is_purely_spatial(small_cloud: PointCloud) -> None:
    """feature_extractor=None must behave exactly like the original spatial-only matcher."""
    target = PointCloud(small_cloud.points + 0.5)

    matching = NearestNeighborMatcher().match(small_cloud, target)

    expected_idx = np.argmin(
        np.linalg.norm(small_cloud.points[:, None, :] - target.points[None, :, :], axis=2), axis=1
    )
    np.testing.assert_allclose(matching.target_positions, target.points[expected_idx], atol=1e-10)


def test_alpha_zero_recovers_standard_gaussian(small_cloud: PointCloud) -> None:
    """alpha=0 makes the feature term vanish: output must match a plain GaussianMatcher."""
    target = PointCloud(small_cloud.points + 0.5)  # slight offset so NN is non-trivial

    plain = GaussianMatcher(sigma=3.0, k=8)
    with_features = GaussianMatcher(
        sigma=3.0,
        k=8,
        feature_extractor=IdentityFeatureExtractor(),
        alpha=0.0,
    )

    m_plain = plain.match(small_cloud, target)
    m_feat = with_features.match(small_cloud, target)

    np.testing.assert_allclose(m_plain.target_positions, m_feat.target_positions, atol=1e-10)
    np.testing.assert_allclose(m_plain.weights, m_feat.weights, atol=1e-10)
