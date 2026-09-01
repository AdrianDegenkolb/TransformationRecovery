import pytest

from experiment_runner import MultiSeedSyntheticICPResult, _quiet, fit_multi_seed
from icp import ICP, ICPResult, MultiStartICP
from synthetic import SyntheticExperiment
from transformation import RigidTransformation


def _make_result(residual: float = 0.0) -> ICPResult:
    return ICPResult(
        transformation=RigidTransformation.identity(),
        n_iterations=1,
        duration_s=0.01,
        converged=True,
        mean_residuals=[residual],
    )


def test_getitem_raises_keyerror_for_missing_seed():
    result = MultiSeedSyntheticICPResult({})
    with pytest.raises(KeyError):
        result[0]


def test_getitem_returns_stored_entry():
    exp = SyntheticExperiment.generate(n=10, seed=0)
    entry = (exp, _make_result())
    result = MultiSeedSyntheticICPResult({0: entry})
    assert result[0] is entry


def test_closest_point_residuals_are_cached_not_recomputed():
    exp = SyntheticExperiment.generate(n=20, seed=0)
    icp_result = _make_result()
    icp_result.transformation = exp.T_gt
    ms_result = MultiSeedSyntheticICPResult({0: (exp, icp_result)})

    first = ms_result.closest_point_residuals
    second = ms_result.closest_point_residuals
    assert first is second


def test_mean_closest_point_residuals_reuses_cached_residuals():
    exp = SyntheticExperiment.generate(n=20, seed=0)
    icp_result = _make_result()
    icp_result.transformation = exp.T_gt
    ms_result = MultiSeedSyntheticICPResult({0: (exp, icp_result)})

    _ = ms_result.mean_closest_point_residuals
    assert 'closest_point_residuals' in ms_result.__dict__


def test_quiet_restores_verbose_after_normal_exit():
    icp = ICP(verbose=True)
    with _quiet(icp):
        assert icp.verbose is False
    assert icp.verbose is True


def test_quiet_restores_verbose_after_exception():
    icp = ICP(verbose=True)
    with pytest.raises(RuntimeError):
        with _quiet(icp):
            assert icp.verbose is False
            raise RuntimeError("boom")
    assert icp.verbose is True


def test_quiet_also_silences_wrapped_icp_for_multistart():
    inner = ICP(verbose=True)
    multi = MultiStartICP(icp=inner, verbose=True)
    with _quiet(multi):
        assert multi.verbose is False
        assert inner.verbose is False
    assert multi.verbose is True
    assert inner.verbose is True


def test_fit_multi_seed_runs_and_restores_verbose():
    icp = ICP(max_iter=3, verbose=True)
    result = fit_multi_seed(icp, seeds=[0, 1], verbose=False, experiment_kwargs={'n': 15, 'noise_std': 0.0})
    assert set(result.r.keys()) == {0, 1}
    assert icp.verbose is True


class _ExplodingTrimmer:
    def trim(self, clouds):
        raise RuntimeError("trim failed")


def test_fit_multi_seed_restores_verbose_on_exception():
    icp = ICP(max_iter=3, verbose=True)
    with pytest.raises(RuntimeError):
        fit_multi_seed(icp, seeds=[0], verbose=False, trimmer=_ExplodingTrimmer(), experiment_kwargs={'n': 15})
    assert icp.verbose is True
