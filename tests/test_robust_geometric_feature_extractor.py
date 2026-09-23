import numpy as np
import pytest

from point_cloud import PointCloud
from feature_extractor import (
    GeometricFeatureExtractor,
    RobustGeometricFeatureExtractor,
    angle_pair_indices,
)


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


@pytest.mark.parametrize("quantiles", [(0.5,), (0.25, 0.5, 0.75), tuple(np.linspace(0.1, 0.9, 8))])
def test_target_dim_matches_actual_feature_width(
    cloud: PointCloud,
    quantiles: tuple[float, ...],
) -> None:
    """target_dim must track the configured quantile count, not a fixed default.

    Consumers size derived parameters from target_dim (e.g. DBSCAN's eps in
    hpo._build_trimmer), so a stale value silently mis-configures them.
    """
    extractor = RobustGeometricFeatureExtractor(k=10, quantiles=quantiles)
    assert extractor.target_dim == extractor.get_features(cloud).shape[1]


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


def test_angle_pairs_exact_when_all_pairs_fit(cloud: PointCloud) -> None:
    """Below the cap every pair is used, so sampling cannot change small-k results."""
    k = 10  # C(10, 2) = 45, far below the default cap
    ti, tj = angle_pair_indices(k, n_angle_pairs=2000)
    expected_i, expected_j = np.triu_indices(k, k=1)
    np.testing.assert_array_equal(ti, expected_i)
    np.testing.assert_array_equal(tj, expected_j)


def test_angle_pairs_are_capped_and_distinct() -> None:
    """Above the cap exactly n_angle_pairs pairs are drawn, none pairing a rank with itself."""
    ti, tj = angle_pair_indices(200, n_angle_pairs=500)
    assert len(ti) == len(tj) == 500
    assert np.all(ti != tj)
    assert ti.max() < 200 and tj.max() < 200


def test_angle_pair_sampling_is_deterministic() -> None:
    """Repeated calls must draw the same pairs.

    Source and target features are computed by separate calls; if the pair set
    differed between them, the two clouds would estimate their angle quantiles from
    different samples and disagree even on a perfect correspondence.
    """
    a = angle_pair_indices(200, n_angle_pairs=300)
    b = angle_pair_indices(200, n_angle_pairs=300)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_large_k_is_still_similarity_invariant() -> None:
    """Sampled angle pairs must not break invariance at k above the pair cap.

    Pairs index neighbour ranks rather than point identities, so a transformed
    cloud selects the same pairs.
    """
    rng = np.random.default_rng(5)
    points = rng.uniform(-10, 10, size=(300, 3))
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    ext = RobustGeometricFeatureExtractor(k=120, n_angle_pairs=400)
    transformed = PointCloud(points @ Q.T * 3.5 + np.array([5.0, -3.0, 12.0]))

    np.testing.assert_allclose(
        ext.get_features(PointCloud(points)),
        ext.get_features(transformed),
        atol=1e-8,
        err_msg="Sampled angle pairs broke similarity invariance.",
    )


def test_sampled_angle_quantiles_converge_to_exhaustive_ones() -> None:
    """Sampled-pair angle quantiles must approach the all-pairs values as samples grow.

    This is the assumption the cap rests on: a quantile needs enough samples, not
    every pair. Asserted on the mean deviation and on its decrease with more
    samples; individual points have heavy-tailed error that a max-deviation bound
    would trip over without indicating a real problem.
    """
    rng = np.random.default_rng(6)
    points = PointCloud(rng.uniform(-10, 10, size=(400, 3)))
    k, n_q = 60, 3
    angle_cols = slice(n_q + 5, n_q + 5 + n_q)

    def angle_quantiles(n_pairs: int) -> np.ndarray:
        ext = RobustGeometricFeatureExtractor(k=k, n_angle_pairs=n_pairs)
        return ext.get_features(points)[:, angle_cols]

    exhaustive = angle_quantiles(k * (k - 1) // 2)
    coarse = float(np.mean(np.abs(angle_quantiles(200) - exhaustive)))
    fine = float(np.mean(np.abs(angle_quantiles(1000) - exhaustive)))

    assert fine < coarse, "More sampled pairs must reduce the deviation."
    assert fine < 0.05, f"Angle quantiles off by {fine:.3f} rad at 1000 sampled pairs."


def test_all_pairs_requested_reproduces_exhaustive_exactly() -> None:
    """Asking for every pair must take the exact path, not a sampled approximation."""
    rng = np.random.default_rng(7)
    points = PointCloud(rng.uniform(-10, 10, size=(200, 3)))
    k = 30

    capped = RobustGeometricFeatureExtractor(k=k, n_angle_pairs=k * (k - 1) // 2)
    generous = RobustGeometricFeatureExtractor(k=k, n_angle_pairs=10 * k * k)
    np.testing.assert_array_equal(capped.get_features(points), generous.get_features(points))
