"""Tests for the feature-extractor characterisation metrics in error_metrics."""
import numpy as np
import pytest
from scipy.spatial import KDTree

from error_metrics import (
    correspondence_margin,
    feature_correspondence_correlation,
    mutual_nearest_neighbor_fraction,
)
from feature_extractor import GeometricFeatureExtractor, RobustGeometricFeatureExtractor
from point_cloud import PointCloud
from synthetic import SyntheticExperiment, invert_correspondence, make_correspondence_pair


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


def test_margin_is_zero_for_identical_clouds(cloud: PointCloud) -> None:
    """A point sits exactly on its own correspondent, so the numerator vanishes."""
    margins = correspondence_margin(cloud, cloud)
    assert margins.shape == (len(cloud.points),)
    np.testing.assert_allclose(margins, np.zeros(len(cloud.points)), atol=1e-10)


def test_margin_crosses_one_exactly_when_the_correspondent_stops_winning() -> None:
    """Below 1 the true pair is the nearest neighbour; above 1 an impostor is.

    This is the property that makes the margin a refinement of the mutual-NN
    fraction rather than an unrelated number: they share a decision threshold.
    """
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    source = PointCloud(points)
    # Nudge each point a little; every correspondent stays nearest.
    near = PointCloud(points + np.array([0.1, 0.0, 0.0]))
    # Shift by most of the spacing: an interior source point is now closer to the
    # target behind it than to its own correspondent ahead of it.
    far = PointCloud(points + np.array([0.9, 0.0, 0.0]))

    near_margins = correspondence_margin(source, near)
    far_margins = correspondence_margin(source, far)

    assert np.all(near_margins < 1.0)
    assert np.all(far_margins[1:] > 1.0), "Interior points lose to the target behind them."
    assert far_margins[0] < 1.0, "The leading point has no target behind it, so it still wins."
    assert mutual_nearest_neighbor_fraction(source, near) > mutual_nearest_neighbor_fraction(source, far)


def test_margin_ignores_unmatched_points(cloud: PointCloud) -> None:
    """Source points marked -1 are dropped from the result rather than scored."""
    correspondence = np.arange(len(cloud.points), dtype=np.int64)
    correspondence[:50] = -1
    margins = correspondence_margin(cloud, cloud, correspondence=correspondence)
    assert margins.shape == (len(cloud.points) - 50,)


def test_margin_still_separates_when_the_mnn_fraction_has_saturated() -> None:
    """The reason this metric exists: a gradient where the fraction reports a flat 1.0.

    Two alignments that both pair every point correctly are indistinguishable to the
    mutual-NN fraction, but the tighter one has a visibly larger margin.
    """
    rng = np.random.default_rng(5)
    points = rng.uniform(-10, 10, size=(200, 3))
    source = PointCloud(points)
    tight = PointCloud(points + rng.normal(0.0, 0.001, size=points.shape))
    loose = PointCloud(points + rng.normal(0.0, 0.05, size=points.shape))

    assert mutual_nearest_neighbor_fraction(source, tight) == pytest.approx(1.0)
    assert mutual_nearest_neighbor_fraction(source, loose) == pytest.approx(1.0)
    assert np.median(correspondence_margin(source, tight)) < np.median(correspondence_margin(source, loose))


def test_margin_rejects_length_mismatch_without_correspondence(cloud: PointCloud) -> None:
    """Assuming index alignment across different-length clouds is a usage error."""
    with pytest.raises(ValueError):
        correspondence_margin(cloud, PointCloud(cloud.points[:100]))


def test_invert_correspondence_round_trips() -> None:
    """Inverting twice returns the original mapping."""
    target_to_source = np.array([2, -1, 0, 3], dtype=np.int64)
    source_to_target = invert_correspondence(target_to_source, n_source=4)
    np.testing.assert_array_equal(source_to_target, np.array([2, -1, 0, 3]))
    np.testing.assert_array_equal(
        invert_correspondence(source_to_target, n_source=4), target_to_source,
    )


def test_invert_correspondence_marks_dropped_sources() -> None:
    """A source point no target claims must come back as -1, not as a stale index."""
    inverted = invert_correspondence(np.array([3, 1], dtype=np.int64), n_source=5)
    np.testing.assert_array_equal(inverted, np.array([-1, 1, -1, 0, -1]))


def test_correspondence_pair_noise_perturbs_both_sides_independently() -> None:
    """Noise must move the clouds apart, or it cannot degrade a descriptor."""
    experiment = SyntheticExperiment.generate(n=300, t_scale=0.0, style="clustered", seed=0)

    def paired_offsets(noise_std: float) -> np.ndarray:
        """Distance between each true pair, undoing the target permutation."""
        p_obs, q_obs, target_to_source = make_correspondence_pair(
            experiment.P, experiment.P, noise_std=noise_std, rng=np.random.default_rng(0),
        )
        matched = target_to_source >= 0
        return np.linalg.norm(
            q_obs.points[matched] - p_obs.points[target_to_source[matched]], axis=1,
        )

    # Without noise the two sides are the same cloud, so every pair coincides exactly.
    np.testing.assert_allclose(paired_offsets(0.0), 0.0, atol=1e-12)
    # With noise each side is drawn independently, so pairs separate.
    assert paired_offsets(0.1).mean() > 0.05


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
    correspondence = invert_correspondence(target_to_source, len(source.points))

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


def test_feature_names_match_output_width() -> None:
    """A name per column, or the per-dimension diagnostic mislabels its rows."""
    cloud = PointCloud(np.random.default_rng(0).uniform(-10, 10, size=(100, 3)))
    for extractor in (GeometricFeatureExtractor(k=10), RobustGeometricFeatureExtractor(k=10)):
        names = extractor.feature_names
        assert len(names) == extractor.target_dim
        assert len(names) == extractor.get_features(cloud).shape[1]
        assert len(set(names)) == len(names), "names must be unique to label a table"


def test_robust_feature_names_track_quantiles() -> None:
    """Names are derived from the configured levels, not hard-coded to the default."""
    extractor = RobustGeometricFeatureExtractor(k=10, quantiles=(0.1, 0.9))
    names = extractor.feature_names
    assert len(names) == extractor.target_dim == 7
    assert names[:2] == ["dist_q10%", "dist_q90%"]
    assert names[-2:] == ["angle_q10%", "angle_q90%"]


def test_rotation_with_angle_hits_the_requested_angle() -> None:
    """A misalignment sweep needs the angle held exactly, not merely bounded."""
    from algebra_utils import rotation_angle, rotation_with_angle

    rng = np.random.default_rng(0)
    for angle in (0.0, 15.0, 90.0, 179.0):
        R = rotation_with_angle(angle, rng)
        assert rotation_angle(R, np.eye(3)) == pytest.approx(angle, abs=1e-6)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_feature_space_metrics_ignore_position(cloud: PointCloud) -> None:
    """use_positions=False must score the descriptor alone, so moving a cloud changes nothing."""
    extractor = RobustGeometricFeatureExtractor(k=20)
    far = PointCloud(cloud.points + np.array([500.0, 0.0, 0.0]))

    near_frac = mutual_nearest_neighbor_fraction(cloud, cloud, extractor, use_positions=False)
    far_frac = mutual_nearest_neighbor_fraction(cloud, far, extractor, use_positions=False)
    assert near_frac == far_frac == pytest.approx(1.0)

    # The joint space, by contrast, is position-sensitive and degrades once separated.
    joint_near = mutual_nearest_neighbor_fraction(cloud, cloud, extractor, beta=1.0)
    joint_far = mutual_nearest_neighbor_fraction(cloud, far, extractor, beta=1.0)
    assert joint_far < joint_near == pytest.approx(1.0)


def test_feature_space_margin_ignores_beta(cloud: PointCloud) -> None:
    """Without positions there is no position block for beta to weigh against."""
    extractor = RobustGeometricFeatureExtractor(k=20)
    shifted = PointCloud(cloud.points + np.array([3.0, 0.0, 0.0]))
    a = correspondence_margin(cloud, shifted, extractor, beta=0.5, use_positions=False)
    b = correspondence_margin(cloud, shifted, extractor, beta=9.0, use_positions=False)
    np.testing.assert_allclose(a, b)


def test_heatmap_rejects_mismatched_labels() -> None:
    """Silently mislabelled axes would misreport which configuration won."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from visualization import SweepVisualizer

    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        SweepVisualizer.plot_heatmap(ax, np.zeros((3, 4)), x_labels=['a', 'b'], y_labels=[1, 2, 3])
    plt.close(fig)


def test_heatmap_annotates_every_cell() -> None:
    """The exact value is the point on these small grids; colour only shows the shape."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from visualization import SweepVisualizer

    fig, ax = plt.subplots()
    values = np.arange(12, dtype=float).reshape(3, 4)
    SweepVisualizer.plot_heatmap(ax, values, x_labels=list('wxyz'), y_labels=[1, 2, 3])
    assert len(ax.texts) == 12
    assert {t.get_text() for t in ax.texts} == {format(v, '.2f') for v in values.ravel()}
    plt.close(fig)
