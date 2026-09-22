import numpy as np
import pytest

from synthetic import CloudStyle, SyntheticExperiment, _hexagonal_centers, _muscle_fiber_cloud


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


def test_observe_point_clouds_no_noise_is_a_subset() -> None:
    """noise_std=0.0 (the default) must leave surviving points untouched."""
    exp = SyntheticExperiment.generate(n=100, seed=0)
    np.random.seed(0)
    p_obs, q_obs = exp.observe_point_clouds(dropout_prob=0.3)

    assert all(tuple(pt) in {tuple(x) for x in exp.P.points} for pt in p_obs.points)
    assert all(tuple(pt) in {tuple(x) for x in exp.Q.points} for pt in q_obs.points)


def test_observe_point_clouds_noise_perturbs_points_independently() -> None:
    """noise_std > 0 must perturb P and Q, and exp.P/exp.Q themselves stay untouched."""
    exp = SyntheticExperiment.generate(n=100, seed=0)
    clean_P, clean_Q = exp.P.points.copy(), exp.Q.points.copy()

    np.random.seed(0)
    p_obs, q_obs = exp.observe_point_clouds(dropout_prob=0.0, noise_std=1.0)

    assert not np.allclose(p_obs.points, exp.P.points)
    assert not np.allclose(q_obs.points, exp.Q.points)
    np.testing.assert_array_equal(exp.P.points, clean_P)
    np.testing.assert_array_equal(exp.Q.points, clean_Q)

