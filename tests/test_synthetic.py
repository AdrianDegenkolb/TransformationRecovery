import numpy as np
import pytest

from point_cloud import PointCloud
from synthetic import (
    CloudStyle,
    SyntheticExperiment,
    _hexagonal_centers,
    _muscle_fiber_cloud,
    make_correspondence_pair,
)


ALL_STYLES: list[CloudStyle] = ["random", "clustered", "lattice", "2d-lattice", "muscle-fiber"]


@pytest.mark.parametrize("style", ALL_STYLES)
def test_generate_produces_finite_points_for_every_style(style: CloudStyle) -> None:
    """generate() should return a non-empty, finite (N, 3) source cloud for every style."""
    exp = SyntheticExperiment.generate(n=200, seed=0, style=style)
    assert exp.S.points.ndim == 2
    assert exp.S.points.shape[1] == 3
    assert len(exp.S.points) > 0
    assert np.all(np.isfinite(exp.S.points))


def test_muscle_fiber_cloud_is_centred() -> None:
    """The generated cloud should be centred at the origin."""
    points = _muscle_fiber_cloud(500, jitter_std=0.0)
    np.testing.assert_allclose(points.mean(axis=0), np.zeros(3), atol=1e-8)


def test_muscle_fiber_nuclei_sit_near_fiber_radius() -> None:
    """Without jitter, every nucleus lies at ~fiber_radius from its own fiber's axis.

    Nuclei are placed on the periphery of parallel fibers, so the xy-distance from
    each nucleus to the *nearest* hexagonally-packed fiber centre should equal
    fiber_radius (up to the small shift introduced by centring the whole cloud on
    the origin, since that shift is computed jointly with the small per-fiber
    angular sampling noise).
    """
    fiber_radius = 3.0
    fiber_spacing = 8.0
    np.random.seed(0)
    points = _muscle_fiber_cloud(
        500, fiber_radius=fiber_radius, fiber_spacing=fiber_spacing, jitter_std=0.0
    )

    n_fibers = max(1, round(500 / 20))  # mirrors the default nuclei_per_fiber=20
    centers = _hexagonal_centers(n_fibers, fiber_spacing)
    centers -= centers.mean(axis=0)

    dists_to_centers = np.linalg.norm(points[:, None, :2] - centers[None, :, :2], axis=-1)
    nearest_dist = dists_to_centers.min(axis=1)
    np.testing.assert_allclose(nearest_dist, fiber_radius, atol=0.15)


def test_muscle_fiber_jitter_increases_spread_around_fiber_radius() -> None:
    """Higher jitter_std should widen the spread of nucleus-to-fiber-axis distances."""
    np.random.seed(0)
    low_jitter = _muscle_fiber_cloud(500, jitter_std=0.05)
    np.random.seed(0)
    high_jitter = _muscle_fiber_cloud(500, jitter_std=1.5)

    def xy_radius_std(points: np.ndarray) -> float:
        radii = np.linalg.norm(points[:, :2], axis=-1)
        return float(radii.std())

    assert xy_radius_std(high_jitter) > xy_radius_std(low_jitter)


def test_generate_produces_noiseless_ground_truth_clouds() -> None:
    """P and Q must exactly satisfy T1(S) = P and T2(S) = Q with no noise, so
    they can serve as a noise-free reference (see MultiSeedSyntheticICPResult.true_residuals).
    """
    exp = SyntheticExperiment.generate(n=50, seed=0)
    np.testing.assert_allclose(exp.P.points, exp.T1.apply(exp.S).points)
    np.testing.assert_allclose(exp.Q.points, exp.T2.apply(exp.S).points)


@pytest.fixture
def aligned_pair() -> tuple[PointCloud, PointCloud]:
    exp = SyntheticExperiment.generate(n=100, seed=1)
    return exp.P, exp.Q


def test_make_correspondence_pair_no_dropout_is_a_bijection(aligned_pair: tuple[PointCloud, PointCloud]) -> None:
    """With dropout_prob=0, every target point keeps a valid, unique source match."""
    p, q = aligned_pair
    rng = np.random.default_rng(0)
    p_obs, q_obs, correspondence = make_correspondence_pair(p, q, dropout_prob=0.0, rng=rng)

    assert len(p_obs) == len(p)
    assert len(q_obs) == len(q)
    assert np.all(correspondence >= 0)
    assert sorted(correspondence.tolist()) == list(range(len(p)))


def test_make_correspondence_pair_matches_are_geometrically_consistent(
    aligned_pair: tuple[PointCloud, PointCloud],
) -> None:
    """correspondence[i] must index the p_obs point that truly corresponds to q_obs[i]."""
    p, q = aligned_pair
    rng = np.random.default_rng(3)
    p_obs, q_obs, correspondence = make_correspondence_pair(p, q, dropout_prob=0.3, rng=rng)

    matched = correspondence >= 0
    # p, q are index-aligned (p[i] <-> q[i]); q_obs points ARE q points (just reordered),
    # so looking up the original q point via p_obs's corresponding original p point
    # must recover q_obs's own point.
    original_p_idx = {tuple(pt): i for i, pt in enumerate(p.points)}
    for q_pt, src_idx in zip(q_obs.points[matched], correspondence[matched]):
        p_pt = p_obs.points[src_idx]
        orig_i = original_p_idx[tuple(p_pt)]
        np.testing.assert_array_equal(q.points[orig_i], q_pt)


def test_make_correspondence_pair_full_dropout_leaves_no_matches(aligned_pair: tuple[PointCloud, PointCloud]) -> None:
    """With dropout_prob=1 on the source side, every surviving target is unmatched."""
    p, q = aligned_pair
    rng = np.random.default_rng(0)
    p_obs, q_obs, correspondence = make_correspondence_pair(p, q, dropout_prob=1.0, rng=rng)

    assert len(p_obs) == 0
    assert np.all(correspondence == -1)


def test_make_correspondence_pair_rejects_misaligned_clouds() -> None:
    """p and q must have equal length (index-aligned)."""
    p = PointCloud(np.zeros((5, 3)))
    q = PointCloud(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        make_correspondence_pair(p, q)


@pytest.mark.parametrize("style", ["random", "clustered", "lattice", "muscle-fiber"])
def test_generate_normalize_spacing_gives_unit_spacing_for_every_style(style: CloudStyle) -> None:
    """With normalize_spacing, every style shares the same length unit."""
    exp = SyntheticExperiment.generate(n=300, seed=0, style=style, normalize_spacing=True)
    assert exp.S.median_spacing == pytest.approx(1.0)
    assert exp.P.median_spacing == pytest.approx(1.0)   # rigid transforms keep spacing
