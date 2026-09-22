import numpy as np
import optuna
import pytest

from feature_extractor import GeometricFeatureExtractor, RobustGeometricFeatureExtractor
from hpo import (
    build_icp_factory,
    build_single_start_nn_icp_factory,
    build_trimmer,
    evaluate_icp,
    make_beta_sweep_objective,
    make_objective,
)
from icp import ICP, MultiStartICP, SigmaAnnealingCallback
from matcher import GaussianMatcher, NearestNeighborMatcher
from trimmer import ClusteringTrimmer


def _fixed_trial(params: dict) -> optuna.trial.FixedTrial:
    return optuna.trial.FixedTrial(params)


def test_hard_matching_without_features():
    trial = _fixed_trial({
        "matching": "hard", "feature_extractor": "none", "use_multistart": False,
    })
    icp = build_icp_factory(trial, max_iter=60, tol=1e-6)()

    assert isinstance(icp, ICP)
    assert isinstance(icp.matcher, NearestNeighborMatcher)
    assert icp.matcher.feature_extractor is None
    assert icp.callbacks == []


def test_hard_matching_with_robust_feature_extractor():
    trial = _fixed_trial({
        "matching": "hard", "feature_extractor": "robust", "fe_k": 12, "beta": 2.5,
        "use_multistart": False,
    })
    icp = build_icp_factory(trial, max_iter=60, tol=1e-6)()

    assert isinstance(icp.matcher, NearestNeighborMatcher)
    assert isinstance(icp.matcher.feature_extractor, RobustGeometricFeatureExtractor)
    assert icp.matcher.feature_extractor.k == 12
    assert icp.matcher.beta == pytest.approx(2.5)


def test_soft_matching_builds_gaussian_matcher_and_anneal_callback():
    trial = _fixed_trial({
        "matching": "soft", "feature_extractor": "none",
        "sigma_init": 10.0, "sigma_ratio": 0.1, "k": 10, "anneal_steps": 55,
        "use_multistart": False,
    })
    icp = build_icp_factory(trial, max_iter=60, tol=1e-6)()

    assert isinstance(icp.matcher, GaussianMatcher)
    assert icp.matcher.sigma == pytest.approx(10.0)
    assert len(icp.callbacks) == 1
    annealer = icp.callbacks[0]
    assert isinstance(annealer, SigmaAnnealingCallback)
    assert annealer.sigma_init == pytest.approx(10.0)
    assert annealer.sigma_final == pytest.approx(1.0)  # sigma_init * sigma_ratio


def test_soft_matching_with_geometric_extractor_additive_mode():
    trial = _fixed_trial({
        "matching": "soft", "feature_extractor": "geometric", "fe_k": 8,
        "feature_mode": "additive", "alpha": 3.0,
        "sigma_init": 5.0, "sigma_ratio": 0.2, "k": 15, "anneal_steps": 50,
        "use_multistart": False,
    })
    icp = build_icp_factory(trial, max_iter=60, tol=1e-6)()

    assert isinstance(icp.matcher.feature_extractor, GeometricFeatureExtractor)
    assert icp.matcher.feature_mode == "additive"
    assert icp.matcher.alpha == pytest.approx(3.0)


def test_multistart_wraps_icp_with_requested_n_starts():
    trial = _fixed_trial({
        "matching": "hard", "feature_extractor": "none",
        "use_multistart": True, "n_starts": 7, "multistart_mode": "disperse",
    })
    result = build_icp_factory(trial, max_iter=60, tol=1e-6, multistart_n_jobs=2)()

    assert isinstance(result, MultiStartICP)
    assert result.n_starts == 7
    assert result.n_jobs == 2
    assert isinstance(result.icp, ICP)


def test_build_trimmer_disabled_returns_none():
    trial = _fixed_trial({"use_trimmer": False})
    assert build_trimmer(trial, n=1000) is None


def test_build_trimmer_enabled_builds_clustering_trimmer():
    trial = _fixed_trial({
        "use_trimmer": True, "trimmer_extractor": "geometric",
        "trimmer_fe_k": 20, "min_cluster_fraction": 0.1,
        "eps_scaling": 0.2, "min_samples_scaling": 1.5,
    })
    trimmer = build_trimmer(trial, n=1000)

    assert isinstance(trimmer, ClusteringTrimmer)
    assert isinstance(trimmer.feature_extractor, GeometricFeatureExtractor)
    assert trimmer.feature_extractor.k == 20
    assert trimmer.min_cluster_fraction == pytest.approx(0.1)
    assert trimmer.clusterer.eps == pytest.approx((2 * 9) ** 0.5 * 0.2)
    assert trimmer.clusterer.min_samples == max(2, int(np.log(1000) * 1.5))


def test_evaluate_icp_returns_all_metrics():
    trial = _fixed_trial({
        "matching": "hard", "feature_extractor": "none", "use_multistart": False,
    })
    factory = build_icp_factory(trial, max_iter=20, tol=1e-6)
    metrics = evaluate_icp(factory, style="clustered", seeds=[0, 1], gen_kwargs=dict(n=100))

    assert set(metrics.keys()) == {
        "mean_true_residual", "mean_rot_err", "mean_t_err", "mean_duration_s", "reliability",
    }
    assert metrics["mean_duration_s"] > 0
    assert 0.0 <= metrics["reliability"] <= 1.0


def test_single_start_factory_builds_plain_icp_with_feature_matcher():
    trial = _fixed_trial({"feature_extractor": "geometric", "fe_k": 30, "beta": 4.0})
    icp = build_single_start_nn_icp_factory(trial, max_iter=60, tol=1e-6)()

    assert isinstance(icp, ICP)
    assert not isinstance(icp, MultiStartICP)
    assert isinstance(icp.matcher, NearestNeighborMatcher)
    assert isinstance(icp.matcher.feature_extractor, GeometricFeatureExtractor)
    assert icp.matcher.feature_extractor.k == 30
    assert icp.matcher.beta == pytest.approx(4.0)
    assert icp.callbacks == []
    assert icp.record_history is False


def test_single_start_factory_never_suggests_disabled_parameters():
    """The reduced space must not offer 'none', multi-start, or soft-matching params.

    A FixedTrial raises ValueError when a parameter it was not given is suggested,
    so supplying only the three intended parameters asserts the space is exactly those.
    """
    trial = _fixed_trial({"feature_extractor": "robust", "fe_k": 10, "beta": 1.0})
    build_single_start_nn_icp_factory(trial, max_iter=60, tol=1e-6)()

    assert set(trial.params) == {"feature_extractor", "fe_k", "beta"}


def test_single_start_factory_returns_fresh_instances():
    trial = _fixed_trial({"feature_extractor": "geometric", "fe_k": 8, "beta": 2.0})
    factory = build_single_start_nn_icp_factory(trial, max_iter=60, tol=1e-6)

    first, second = factory(), factory()
    assert first is not second
    assert first.matcher is not second.matcher
    assert first.matcher.beta == second.matcher.beta


def test_make_beta_sweep_objective_records_metrics_and_ignores_trimmer_params():
    trial = _fixed_trial({"feature_extractor": "geometric", "fe_k": 5, "beta": 3.0})
    objective = make_beta_sweep_objective(
        style="clustered", seeds=[0, 1], gen_kwargs=dict(n=100),
        max_iter=20, tol=1e-6,
    )
    value = objective(trial)

    assert isinstance(value, float)
    assert {"mean_rot_err", "mean_t_err", "mean_duration_s", "reliability"} <= trial.user_attrs.keys()
    # No trimmer parameters were suggested: trimming is fixed off for this study.
    assert set(trial.params) == {"feature_extractor", "fe_k", "beta"}


def test_make_objective_returns_single_true_residual_objective():
    trial = _fixed_trial({
        "matching": "hard", "feature_extractor": "none", "use_multistart": False,
        "use_trimmer": False,
    })
    objective = make_objective(
        style="clustered", seeds=[0, 1], gen_kwargs=dict(n=100),
        max_iter=20, tol=1e-6,
    )
    value = objective(trial)

    assert isinstance(value, float)
    assert {"mean_rot_err", "mean_t_err", "mean_duration_s", "reliability"} <= trial.user_attrs.keys()
