import numpy as np
import pytest

import icp as icp_module
from algebra_utils import sample_dispersed_rotations, sample_uniform_rotations
from icp import ICP, MultiStartICP, SigmaAnnealingCallback, _windowed_delta
from point_cloud import PointCloud
from transformation import RigidTransformation


class _FakeMatcher:
    def __init__(self):
        self.sigma = None


class _FakeICP:
    def __init__(self, matcher):
        self.matcher = matcher


def _unit_x_translation() -> RigidTransformation:
    return RigidTransformation(np.eye(3), np.array([1.0, 0.0, 0.0]))


def test_windowed_delta_uses_accumulated_when_history_shorter_than_window():
    accumulated = _unit_x_translation()
    delta = _windowed_delta(transform_history=[], accumulated=accumulated, window=10)
    assert delta == pytest.approx(1.0)


def test_windowed_delta_is_zero_for_identity():
    accumulated = RigidTransformation.identity()
    history = [RigidTransformation.identity()] * 10
    assert _windowed_delta(history, accumulated, window=10) == pytest.approx(0.0, abs=1e-12)


def test_windowed_delta_composes_against_reference_window_steps_ago():
    """10 identical unit-x-translation steps: accumulated is translation-by-10;
    with window=10, ref is the 1st step (translation-by-1), so the windowed
    delta should reflect translation-by-9 (10 - 1).
    """
    step = _unit_x_translation()
    accumulated = RigidTransformation.identity()
    history = []
    for _ in range(10):
        accumulated = step.compose(accumulated)
        history.append(accumulated)

    delta = _windowed_delta(history, accumulated, window=10)
    assert delta == pytest.approx(9.0)


def test_windowed_delta_respects_custom_window_size():
    """Same setup, window=3: ref is the 8th step (translation-by-8), so the
    windowed delta should reflect translation-by-2 (10 - 8).
    """
    step = _unit_x_translation()
    accumulated = RigidTransformation.identity()
    history = []
    for _ in range(10):
        accumulated = step.compose(accumulated)
        history.append(accumulated)

    delta = _windowed_delta(history, accumulated, window=3)
    assert delta == pytest.approx(2.0)


def test_sigma_annealing_interpolates_from_init_to_final():
    matcher = _FakeMatcher()
    fake_icp = _FakeICP(matcher)
    cb = SigmaAnnealingCallback(sigma_init=4.0, sigma_final=0.5, anneal_steps=5)

    cb.on_iteration_start(0, fake_icp)
    assert matcher.sigma == pytest.approx(4.0)

    cb.on_iteration_start(4, fake_icp)
    assert matcher.sigma == pytest.approx(0.5)

    # Iterations beyond anneal_steps clamp to sigma_final rather than extrapolating.
    cb.on_iteration_start(100, fake_icp)
    assert matcher.sigma == pytest.approx(0.5)


def test_sigma_annealing_reads_icp_matcher_at_call_time():
    """Regression test: the callback must operate on whatever `icp.matcher` is
    passed to `on_iteration_start`, not a reference captured at construction
    (the earlier design held its own `self.matcher`, which could silently
    diverge from the matcher actually driving `icp`).
    """
    matcher_a = _FakeMatcher()
    matcher_b = _FakeMatcher()
    cb = SigmaAnnealingCallback(sigma_init=2.0, sigma_final=1.0, anneal_steps=2)

    cb.on_iteration_start(0, _FakeICP(matcher_a))
    cb.on_iteration_start(0, _FakeICP(matcher_b))

    assert matcher_a.sigma == pytest.approx(2.0)
    assert matcher_b.sigma == pytest.approx(2.0)


def test_multistart_icp_defaults_to_dispersed_rotation_sampler():
    multi = MultiStartICP(icp=ICP())
    assert multi.rotation_sampler is sample_dispersed_rotations


def test_multistart_icp_accepts_custom_rotation_sampler():
    multi = MultiStartICP(icp=ICP(), rotation_sampler=sample_uniform_rotations)
    assert multi.rotation_sampler is sample_uniform_rotations


def _small_clouds() -> tuple[PointCloud, PointCloud]:
    source = PointCloud(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    target = PointCloud(source.points + np.array([0.1, 0.0, 0.0]))
    return source, target


def test_icp_record_history_true_populates_cloud_and_matching_history():
    source, target = _small_clouds()
    result = ICP(max_iter=5, record_history=True).fit(source, target)
    assert len(result.cloud_history) == result.n_iterations
    assert len(result.matching_history) == result.n_iterations


def test_icp_record_history_false_skips_cloud_and_matching_history():
    source, target = _small_clouds()
    result = ICP(max_iter=5, record_history=False).fit(source, target)
    assert result.cloud_history == []
    assert result.matching_history == []
    # cheap per-iteration info is still recorded regardless
    assert len(result.mean_residuals) == result.n_iterations


def test_multistart_icp_n_jobs_1_runs_without_process_pool(monkeypatch):
    """n_jobs=1 must not fork a ProcessPoolExecutor: some wrapped .fit()
    implementations (e.g. probreg's CPD, which pulls in open3d) initialize native
    thread pools at import time, and forking such a process afterwards deadlocks
    the child on its first native call. Matches the n_jobs=1-is-sequential
    convention already used by experiment_runner.fit_multi_seed.
    """
    source, target = _small_clouds()

    def _boom(*args, **kwargs):
        raise AssertionError("ProcessPoolExecutor should not be used when n_jobs=1")

    monkeypatch.setattr(icp_module, "ProcessPoolExecutor", _boom)

    multi = MultiStartICP(ICP(max_iter=5), n_starts=3, n_jobs=1, verbose=False)
    result = multi.fit(source, target)
    assert len(result.all_results) == 3
    assert len(result.transform_history) == result.n_iterations
