import numpy as np
import pytest

pytest.importorskip("probreg")

from probreg_baselines import ProbregCPD, ProbregFilterReg, ProbregGMMTree
from point_cloud import PointCloud
from synthetic import SyntheticExperiment

_METHODS = [ProbregCPD, ProbregFilterReg, ProbregGMMTree]


def _clustered_cloud_with_offset(offset: np.ndarray) -> tuple[PointCloud, PointCloud]:
    """A clustered cloud and a small-translation copy of it.

    Deliberately avoids SyntheticExperiment's fully random rotations: without
    multi-start, none of these methods (like plain single-start ICP) are
    guaranteed to recover an arbitrary large rotation from an identity initial
    guess, so a rotation-only scenario here would be flaky. This isolates each
    adapter's correctness from that known local-optimum sensitivity, which the
    comparison notebook explores separately via MultiStartICP vs. single-run runs.
    """
    exp = SyntheticExperiment.generate(n=200, noise_std=0.01, t_scale=0, style="clustered", seed=0)
    source = exp.S
    target = PointCloud(source.points + offset)
    return source, target


@pytest.mark.parametrize("method_cls", _METHODS)
def test_probreg_method_recovers_small_translation(method_cls):
    """Checks each adapter is wired correctly (substantially closes the gap),
    not exact recovery: these EM methods converge to a slightly regularized/biased
    optimum even on easy data, so residual error stays nonzero by design.
    """
    offset = np.array([1.5, -0.5, 0.3])
    source, target = _clustered_cloud_with_offset(offset)
    initial_gap = float(np.linalg.norm(offset))

    result = method_cls(maxiter=150).fit(source, target)

    t_err = float(np.linalg.norm(result.transformation.t - offset))
    assert t_err < 0.5 * initial_gap
    assert float(np.linalg.norm(result.transformation.R - np.eye(3))) < 0.3


@pytest.mark.parametrize("method_cls", _METHODS)
def test_probreg_method_records_matching_length_history(method_cls):
    # n=200: GMMTree's lstsq-based M-step hits an edge case with very small point
    # counts (empty residuals array from an underdetermined per-node system),
    # a probreg-internal limitation unrelated to this adapter.
    exp = SyntheticExperiment.generate(n=200, noise_std=0.01, t_scale=8, style="clustered", seed=1)
    result = method_cls(maxiter=100).fit(exp.P, exp.Q)

    assert len(result.transform_history) == result.n_iterations
    assert len(result.mean_residuals) == result.n_iterations
    assert len(result.deltas) == result.n_iterations
    assert result.n_iterations > 0
