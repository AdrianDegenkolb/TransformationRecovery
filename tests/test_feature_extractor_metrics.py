"""Tests for the feature-extractor characterisation metrics in error_metrics."""
import numpy as np
import pytest
from scipy.spatial import KDTree

from error_metrics import (
    feature_correspondence_correlation,
    mutual_nearest_neighbor_fraction,
)
from feature_extractor import RobustGeometricFeatureExtractor
from point_cloud import PointCloud
from synthetic import SyntheticExperiment, make_correspondence_pair


@pytest.fixture
def cloud() -> PointCloud:
    rng = np.random.default_rng(0)
    return PointCloud(rng.uniform(-10, 10, size=(200, 3)))


def test_correlation_is_one_for_identical_features(cloud: PointCloud) -> None:
    """An unperturbed pair agrees perfectly, so every dimension scores 1."""
    feats = RobustGeometricFeatureExtractor(k=20).get_features(cloud)
    r = feature_correspondence_correlation(feats, feats)
    np.testing.assert_allclose(r, np.ones(feats.shape[1]), atol=1e-10)


def test_correlation_is_scale_and_offset_invariant(cloud: PointCloud) -> None:
    """Correlation must ignore affine rescaling, since the matcher z-scores anyway."""
    feats = RobustGeometricFeatureExtractor(k=20).get_features(cloud)
    r = feature_correspondence_correlation(feats, 3.5 * feats + 7.0)
    np.testing.assert_allclose(r, np.ones(feats.shape[1]), atol=1e-10)


def test_correlation_is_zero_for_unrelated_features(cloud: PointCloud) -> None:
    """Independent noise carries no correspondence signal."""
    rng = np.random.default_rng(1)
    r = feature_correspondence_correlation(
        rng.standard_normal((500, 4)), rng.standard_normal((500, 4)),
    )
    assert np.all(np.abs(r) < 0.15)


def test_correlation_constant_dimension_scores_zero(cloud: PointCloud) -> None:
    """A dimension with no variance yields 0, not NaN."""
    feats = np.hstack([np.ones((50, 1)), np.arange(50).reshape(-1, 1)])
    r = feature_correspondence_correlation(feats, feats)
    assert r[0] == 0.0
    assert r[1] == pytest.approx(1.0)


def test_correlation_rejects_misaligned_inputs() -> None:
    """Rows must be correspondence-aligned; differing shapes are a usage error."""
    with pytest.raises(ValueError):
        feature_correspondence_correlation(np.zeros((10, 3)), np.zeros((9, 3)))


def test_mutual_nn_fraction_is_one_for_identical_clouds(cloud: PointCloud) -> None:
    """A cloud matched against itself has every point as its own mutual neighbour."""
    assert mutual_nearest_neighbor_fraction(cloud, cloud) == pytest.approx(1.0)


def test_mutual_nn_fraction_degrades_as_clouds_separate(cloud: PointCloud) -> None:
    """Shifting the target away breaks correspondences, so the fraction must fall."""
    shifted = PointCloud(cloud.points + np.array([4.0, 0.0, 0.0]))
    near = mutual_nearest_neighbor_fraction(cloud, PointCloud(cloud.points + 0.01))
    far = mutual_nearest_neighbor_fraction(cloud, shifted)
    assert near > far


def test_mutual_nn_fraction_honours_correspondence_argument(cloud: PointCloud) -> None:
    """A permuted target still scores 1 when the permutation is supplied."""
    rng = np.random.default_rng(2)
    perm = rng.permutation(len(cloud.points))
    permuted = PointCloud(cloud.points[perm])
    # source i sits at position perm^-1[i] in the permuted cloud
    correspondence = np.argsort(perm).astype(np.int64)
    assert mutual_nearest_neighbor_fraction(
        cloud, permuted, correspondence=correspondence,
    ) == pytest.approx(1.0)


def test_mutual_nn_fraction_ignores_unmatched_points(cloud: PointCloud) -> None:
    """Source points marked -1 are excluded rather than counted as failures."""
    correspondence = np.arange(len(cloud.points), dtype=np.int64)
    correspondence[:50] = -1
    assert mutual_nearest_neighbor_fraction(
        cloud, cloud, correspondence=correspondence,
    ) == pytest.approx(1.0)


def test_mutual_nn_fraction_rejects_length_mismatch_without_correspondence(cloud: PointCloud) -> None:
    """Assuming index alignment across different-length clouds is a usage error."""
    with pytest.raises(ValueError):
        mutual_nearest_neighbor_fraction(cloud, PointCloud(cloud.points[:100]))


def test_features_and_positions_win_in_opposite_regimes() -> None:
    """Positions win while the clouds are close; features win once they are not.

    Pins the trade-off the metric exists to expose. Below roughly one point spacing
    of misalignment, position alone already pairs almost everything correctly and
    adding features only injects the disagreement dropout causes between the two
    clouds' descriptors. Past one spacing, position-based pairing collapses to zero
    while features still recover a substantial fraction: they are what gives ICP a
    basin of attraction wider than the gap between neighbouring points.
    """
    experiment = SyntheticExperiment.generate(n=600, t_scale=8.0, style="muscle-fiber", seed=0)
    source, target, target_to_source = make_correspondence_pair(
        experiment.P, experiment.P, dropout_prob=0.1, rng=np.random.default_rng(0),
    )
    # make_correspondence_pair maps target -> source; this metric wants source -> target.
    correspondence = np.full(len(source.points), -1, dtype=np.int64)
    matched = target_to_source >= 0
    correspondence[target_to_source[matched]] = np.flatnonzero(matched)

    spacing = float(np.mean(np.linalg.norm(
        source.points - source.points[KDTree(source.points).query(source.points, k=2)[1][:, 1]], axis=1,
    )))
    extractor = RobustGeometricFeatureExtractor(k=40)

    def fraction(drift_in_spacings: float, beta: float, use_features: bool) -> float:
        drifted = PointCloud(target.points + np.array([drift_in_spacings * spacing, 0.0, 0.0]))
        return mutual_nearest_neighbor_fraction(
            source, drifted, feature_extractor=extractor if use_features else None,
            beta=beta, correspondence=correspondence,
        )

    assert fraction(0.1, 0.0, False) > fraction(0.1, 3.0, True), "Positions should win when nearly aligned."
    assert fraction(2.0, 3.0, True) > fraction(2.0, 0.0, False), "Features should win once misaligned."
