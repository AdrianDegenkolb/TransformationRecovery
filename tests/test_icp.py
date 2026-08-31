import numpy as np
import pytest

from icp import SigmaAnnealingCallback, _windowed_delta
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
