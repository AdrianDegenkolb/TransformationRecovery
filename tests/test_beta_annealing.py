"""Tests for the beta-annealing callbacks in icp."""
import numpy as np
import pytest

from icp import (
    ICP,
    ICPProgress,
    IterationBetaAnnealing,
    ResidualBetaAnnealing,
    TwoPhaseBetaAnnealing,
)
from matcher import NearestNeighborMatcher
from feature_extractor import GeometricFeatureExtractor
from point_cloud import PointCloud
from synthetic import SyntheticExperiment


def _progress(iteration=0, residual=None, delta=None, spacing=1.0) -> ICPProgress:
    return ICPProgress(iteration=iteration, mean_residual=residual, delta=delta,
                       point_spacing=spacing)


class _FakeMatcher:
    def __init__(self, beta: float = 0.0) -> None:
        self.beta = beta


class _FakeICP:
    def __init__(self, matcher: object) -> None:
        self.matcher = matcher


def test_schedule_rejects_an_upward_anneal() -> None:
    """A schedule that rises would contradict what the callback exists to do."""
    with pytest.raises(ValueError):
        IterationBetaAnnealing(beta_start=1.0, beta_end=3.0, anneal_steps=10)


def test_iteration_schedule_spans_its_endpoints() -> None:
    """Open-loop decay must start at beta_start and finish at beta_end."""
    cb = IterationBetaAnnealing(beta_start=4.0, beta_end=0.5, anneal_steps=5)
    icp = _FakeICP(_FakeMatcher())
    cb.on_iteration_start(_progress(0), icp)
    assert icp.matcher.beta == pytest.approx(4.0)
    cb.on_iteration_start(_progress(4), icp)
    assert icp.matcher.beta == pytest.approx(0.5)
    # Past the schedule it holds, rather than continuing to fall.
    cb.on_iteration_start(_progress(99), icp)
    assert icp.matcher.beta == pytest.approx(0.5)


def test_iteration_schedule_reaches_zero_when_asked() -> None:
    """Geometric decay cannot reach 0, so a zero endpoint must be handled separately."""
    cb = IterationBetaAnnealing(beta_start=3.0, beta_end=0.0, anneal_steps=5)
    icp = _FakeICP(_FakeMatcher())
    cb.on_iteration_start(_progress(4), icp)
    assert icp.matcher.beta == pytest.approx(0.0)


def test_residual_schedule_is_scale_free() -> None:
    """The trigger is a multiple of point spacing, so absolute residuals must not matter."""
    cb_small = ResidualBetaAnnealing(3.0, 0.0, start_spacings=5.0, end_spacings=1.0)
    cb_large = ResidualBetaAnnealing(3.0, 0.0, start_spacings=5.0, end_spacings=1.0)
    small, large = _FakeICP(_FakeMatcher()), _FakeICP(_FakeMatcher())
    # Same residual in spacings, wildly different absolute values.
    cb_small.on_iteration_start(_progress(1, residual=3.0, spacing=1.0), small)
    cb_large.on_iteration_start(_progress(1, residual=300.0, spacing=100.0), large)
    assert small.matcher.beta == pytest.approx(large.matcher.beta)


def test_residual_schedule_holds_start_until_a_residual_exists() -> None:
    """Before the first M-step there is nothing to react to."""
    cb = ResidualBetaAnnealing(3.0, 0.0)
    icp = _FakeICP(_FakeMatcher())
    cb.on_iteration_start(_progress(0, residual=None), icp)
    assert icp.matcher.beta == pytest.approx(3.0)


def test_residual_schedule_saturates_outside_its_thresholds() -> None:
    """Far apart means beta_start; within a spacing means beta_end."""
    cb = ResidualBetaAnnealing(3.0, 0.25, start_spacings=5.0, end_spacings=1.0)
    icp = _FakeICP(_FakeMatcher())
    cb.on_iteration_start(_progress(1, residual=50.0, spacing=1.0), icp)
    assert icp.matcher.beta == pytest.approx(3.0)
    cb.on_iteration_start(_progress(2, residual=0.1, spacing=1.0), icp)
    assert icp.matcher.beta == pytest.approx(0.25)


def test_residual_schedule_rejects_inverted_thresholds() -> None:
    """end_spacings above start_spacings would invert the mapping."""
    with pytest.raises(ValueError):
        ResidualBetaAnnealing(3.0, 0.0, start_spacings=1.0, end_spacings=5.0)


def test_schedules_never_increase_beta() -> None:
    """A rising beta is the latch failure mode these schedules are clamped against."""
    cb = ResidualBetaAnnealing(3.0, 0.0, start_spacings=5.0, end_spacings=1.0)
    icp = _FakeICP(_FakeMatcher())
    cb.on_iteration_start(_progress(1, residual=1.0, spacing=1.0), icp)
    converged = icp.matcher.beta
    # The residual jumps back up, as a bad correspondence set would make it.
    cb.on_iteration_start(_progress(2, residual=50.0, spacing=1.0), icp)
    assert icp.matcher.beta <= converged


def test_two_phase_switches_on_a_stalled_transform() -> None:
    """The switch fires on movement stopping, not on the residual being small."""
    cb = TwoPhaseBetaAnnealing(beta_start=3.0, beta_end=0.0, stall_delta=1e-3)
    icp = _FakeICP(_FakeMatcher())
    cb.on_iteration_start(_progress(1, delta=5.0), icp)
    assert icp.matcher.beta == pytest.approx(3.0)
    assert cb.switched_at is None
    cb.on_iteration_start(_progress(2, delta=1e-6), icp)
    assert icp.matcher.beta == pytest.approx(0.0)
    assert cb.switched_at == 2


def test_reset_rewinds_a_schedule_for_reuse() -> None:
    """One callback instance is reused across seeds, so it must not carry state over."""
    cb = TwoPhaseBetaAnnealing(3.0, 0.0, stall_delta=1e-3)
    icp = _FakeICP(_FakeMatcher())
    cb.on_iteration_start(_progress(1, delta=1e-6), icp)
    assert cb.switched_at == 1
    cb.reset()
    assert cb.switched_at is None
    cb.on_iteration_start(_progress(0, delta=5.0), icp)
    assert icp.matcher.beta == pytest.approx(3.0)


def test_schedule_rejects_a_matcher_without_beta() -> None:
    """Annealing beta on a featureless matcher is a silent no-op otherwise."""
    cb = IterationBetaAnnealing(3.0, 0.0, 10)

    class _NoBeta:
        pass

    with pytest.raises(AttributeError):
        cb.on_iteration_start(_progress(0), _FakeICP(_NoBeta()))


def test_annealing_runs_end_to_end_and_moves_beta() -> None:
    """The callback must actually drive a real fit, not just a fake matcher."""
    experiment = SyntheticExperiment.generate(n=200, t_scale=8.0, style='clustered', seed=0)
    matcher = NearestNeighborMatcher(feature_extractor=GeometricFeatureExtractor(k=20), beta=3.0)
    cb = IterationBetaAnnealing(beta_start=3.0, beta_end=0.0, anneal_steps=20)
    icp = ICP(matcher=matcher, max_iter=20, tol=1e-10, callbacks=[cb])
    result = icp.fit(experiment.P, experiment.Q)
    assert result.n_iterations > 0
    assert matcher.beta < 3.0, 'beta should have been annealed down over the run'


def test_progress_reports_the_source_cloud_spacing() -> None:
    """A residual-driven schedule is only scale-free if the spacing it gets is right."""
    seen: list[ICPProgress] = []

    class _Recorder(IterationBetaAnnealing):
        def _target_beta(self, progress: ICPProgress) -> float:
            seen.append(progress)
            return super()._target_beta(progress)

    # A real experiment, so the fit takes several iterations and a residual appears.
    experiment = SyntheticExperiment.generate(n=200, t_scale=8.0, style='clustered', seed=0)
    matcher = NearestNeighborMatcher(feature_extractor=GeometricFeatureExtractor(k=20), beta=1.0)
    icp = ICP(matcher=matcher, max_iter=5, tol=1e-12, callbacks=[_Recorder(1.0, 0.0, 5)])
    icp.fit(experiment.P, experiment.Q)

    assert len(seen) >= 2
    assert seen[0].point_spacing == pytest.approx(experiment.P.median_spacing)
    assert seen[0].mean_residual is None, 'no M-step has run before the first iteration'
    assert seen[0].delta is None
    assert seen[1].mean_residual is not None and seen[1].delta is not None


def test_fit_resets_schedules_between_runs() -> None:
    """A monotone clamp that survives a run would start the next one already annealed."""
    experiment = SyntheticExperiment.generate(n=200, t_scale=8.0, style='clustered', seed=0)
    matcher = NearestNeighborMatcher(feature_extractor=GeometricFeatureExtractor(k=20), beta=3.0)
    cb = IterationBetaAnnealing(beta_start=3.0, beta_end=0.0, anneal_steps=10)
    icp = ICP(matcher=matcher, max_iter=10, tol=1e-12, callbacks=[cb])

    icp.fit(experiment.P, experiment.Q)
    annealed = matcher.beta
    assert annealed < 3.0

    seen: list[float] = []

    class _Recorder(IterationBetaAnnealing):
        def _target_beta(self, progress: ICPProgress) -> float:
            value = super()._target_beta(progress)
            seen.append(value)
            return value

    recorder = _Recorder(beta_start=3.0, beta_end=0.0, anneal_steps=10)
    icp.callbacks = [recorder]
    icp.fit(experiment.P, experiment.Q)
    icp.fit(experiment.P, experiment.Q)
    # Both runs must begin at the start value, not where the previous one ended.
    assert seen[0] == pytest.approx(3.0)
    assert max(seen) == pytest.approx(3.0), 'the second run restarted below beta_start'


def test_delta_schedule_is_scale_free() -> None:
    """Delta carries cloud units, so the trigger is a fraction of this run's own peak."""
    from icp import DeltaBetaAnnealing

    small, large = DeltaBetaAnnealing(3.0, 0.0), DeltaBetaAnnealing(3.0, 0.0)
    icp_small, icp_large = _FakeICP(_FakeMatcher()), _FakeICP(_FakeMatcher())
    for cb, icp, scale in ((small, icp_small, 1.0), (large, icp_large, 1000.0)):
        cb.on_iteration_start(_progress(1, delta=10.0 * scale), icp)
        cb.on_iteration_start(_progress(2, delta=1.0 * scale), icp)
    assert icp_small.matcher.beta == pytest.approx(icp_large.matcher.beta)


def test_delta_schedule_holds_start_while_the_transform_moves() -> None:
    """A delta at its peak means the search is still underway."""
    from icp import DeltaBetaAnnealing

    cb = DeltaBetaAnnealing(3.0, 0.0, end_fraction=0.05)
    icp = _FakeICP(_FakeMatcher())
    cb.on_iteration_start(_progress(0, delta=None), icp)
    assert icp.matcher.beta == pytest.approx(3.0)
    cb.on_iteration_start(_progress(1, delta=20.0), icp)
    assert icp.matcher.beta == pytest.approx(3.0)
    # Collapsed well past the threshold: the search is over.
    cb.on_iteration_start(_progress(2, delta=1e-6), icp)
    assert icp.matcher.beta == pytest.approx(0.0)


def test_delta_schedule_rejects_an_impossible_fraction() -> None:
    """A fraction outside (0, 1] could never be reached, or would trigger immediately."""
    from icp import DeltaBetaAnnealing

    with pytest.raises(ValueError):
        DeltaBetaAnnealing(3.0, 0.0, end_fraction=0.0)
    with pytest.raises(ValueError):
        DeltaBetaAnnealing(3.0, 0.0, end_fraction=1.5)


def test_delta_schedule_forgets_its_peak_between_runs() -> None:
    """A peak carried over would make the next run look converged from the start."""
    from icp import DeltaBetaAnnealing

    cb = DeltaBetaAnnealing(3.0, 0.0)
    icp = _FakeICP(_FakeMatcher())
    cb.on_iteration_start(_progress(1, delta=1000.0), icp)
    cb.reset()
    assert cb._peak_delta == 0.0
    cb.on_iteration_start(_progress(1, delta=10.0), icp)
    assert icp.matcher.beta == pytest.approx(3.0), 'a fresh run starts at its own peak'


def test_consistency_schedule_tracks_one_to_one_matching() -> None:
    """Degenerate correspondences mean keep searching; near one-to-one means refine."""
    from icp import ConsistencyBetaAnnealing

    cb = ConsistencyBetaAnnealing(3.0, 0.0, start_uniqueness=0.5, end_uniqueness=0.9)
    icp = _FakeICP(_FakeMatcher())
    cb.on_iteration_start(_progress(0, ), icp)
    assert icp.matcher.beta == pytest.approx(3.0), 'no matching has happened yet'
    icp2 = _FakeICP(_FakeMatcher())
    cb2 = ConsistencyBetaAnnealing(3.0, 0.0, start_uniqueness=0.5, end_uniqueness=0.9)
    progress = ICPProgress(iteration=1, mean_residual=None, delta=None,
                           point_spacing=1.0, match_uniqueness=0.3)
    cb2.on_iteration_start(progress, icp2)
    assert icp2.matcher.beta == pytest.approx(3.0)
    progress = ICPProgress(iteration=2, mean_residual=None, delta=None,
                           point_spacing=1.0, match_uniqueness=0.98)
    cb2.on_iteration_start(progress, icp2)
    assert icp2.matcher.beta == pytest.approx(0.0)


def test_consistency_schedule_rejects_inverted_thresholds() -> None:
    """end_uniqueness below start_uniqueness would invert the mapping."""
    from icp import ConsistencyBetaAnnealing

    with pytest.raises(ValueError):
        ConsistencyBetaAnnealing(3.0, 0.0, start_uniqueness=0.9, end_uniqueness=0.5)


def test_progress_reports_match_uniqueness_after_the_first_step() -> None:
    """The consistency schedule is only usable if fit actually supplies the signal."""
    from icp import ConsistencyBetaAnnealing

    seen: list[ICPProgress] = []

    class _Recorder(ConsistencyBetaAnnealing):
        def _target_beta(self, progress: ICPProgress) -> float:
            seen.append(progress)
            return super()._target_beta(progress)

    experiment = SyntheticExperiment.generate(n=200, t_scale=8.0, style='clustered', seed=0)
    matcher = NearestNeighborMatcher(feature_extractor=GeometricFeatureExtractor(k=20), beta=3.0)
    icp = ICP(matcher=matcher, max_iter=5, tol=1e-12, callbacks=[_Recorder(3.0, 0.0)])
    icp.fit(experiment.P, experiment.Q)

    assert seen[0].match_uniqueness is None
    assert seen[1].match_uniqueness is not None
    assert 0.0 < seen[1].match_uniqueness <= 1.0
