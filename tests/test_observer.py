import numpy as np
import pytest

from point_cloud import PointCloud
from synthetic import PerfectObserver, PointCloudObserver


def _cloud(n: int = 200) -> PointCloud:
    return PointCloud(np.random.default_rng(0).uniform(-10, 10, size=(n, 3)))


def test_perfect_observer_returns_the_cloud_unchanged() -> None:
    """The default observer for clean runs must not drop or perturb anything."""
    cloud = _cloud()
    observed = PerfectObserver().observe(cloud)

    np.testing.assert_array_equal(observed.points, cloud.points)


def test_dropout_removes_roughly_the_requested_fraction() -> None:
    cloud = _cloud(2000)
    observed = PointCloudObserver(seed=0, dropout_prob=0.3).observe(cloud)

    assert len(observed) < len(cloud)
    assert len(observed) / len(cloud) == pytest.approx(0.7, abs=0.05)


def test_noise_perturbs_every_point_but_leaves_the_input_untouched() -> None:
    cloud = _cloud()
    original = cloud.points.copy()
    observed = PointCloudObserver(seed=0, noise_std=1.0).observe(cloud)

    assert observed.points.shape == cloud.points.shape
    assert not np.allclose(observed.points, cloud.points)
    np.testing.assert_array_equal(cloud.points, original)


def test_same_seed_reproduces_the_same_observation() -> None:
    cloud = _cloud()
    first = PointCloudObserver(seed=7, noise_std=0.5, dropout_prob=0.2).observe(cloud)
    second = PointCloudObserver(seed=7, noise_std=0.5, dropout_prob=0.2).observe(cloud)

    np.testing.assert_array_equal(first.points, second.points)


def test_consecutive_observations_are_independent_draws() -> None:
    """Observing P then Q must mimic two separate measurements, not replay one."""
    cloud = _cloud()
    observer = PointCloudObserver(seed=0, noise_std=0.5, dropout_prob=0.2)

    first = observer.observe(cloud)
    second = observer.observe(cloud)

    assert not (len(first) == len(second) and np.allclose(first.points, second.points))


def test_spawned_observers_keep_settings_but_draw_independently() -> None:
    """Spawning is what keeps parallel workers from replaying identical noise."""
    cloud = _cloud()
    parent = PointCloudObserver(seed=0, noise_std=0.5, dropout_prob=0.2)
    first, second = parent.spawn(), parent.spawn()

    assert first.noise_std == parent.noise_std
    assert first.dropout_prob == parent.dropout_prob

    a, b = first.observe(cloud), second.observe(cloud)
    assert not (len(a) == len(b) and np.allclose(a.points, b.points))


def test_spawn_preserves_the_subclass() -> None:
    spawned = PerfectObserver().spawn()

    assert isinstance(spawned, PerfectObserver)
    assert spawned.noise_std == 0.0
    assert spawned.dropout_prob == 0.0
