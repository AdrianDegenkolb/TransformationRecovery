from __future__ import annotations

from typing import Any, Callable, Literal, cast

import numpy as np
import optuna

from algebra_utils import rotation_angle
from feature_extractor import FeatureExtractor, GeometricFeatureExtractor
from icp import ICP, SigmaAnnealingCallback
from matcher import GaussianMatcher, NearestNeighborMatcher
from synthetic import CloudStyle, SyntheticExperiment


def build_icp_factory(
    trial: optuna.Trial,
    max_iter: int,
    tol: float,
) -> Callable[[], ICP]:
    """Suggest hyperparameters from trial and return a factory for fresh ICP instances.

    Tuned parameters:
        matching:      categorical ['hard', 'soft']
        sigma_init:    log-uniform [1.0, 50.0]          (soft only)
        sigma_ratio:   log-uniform [0.01, 0.5]          (soft only); sigma_final = sigma_init * sigma_ratio
        anneal_steps:  int [50, max_iter]               (soft only)
        k:             int [5, 30]                      (soft only)
        feature_extractor: categorical ['none', 'geometric']  (soft only)
        fe_k:          int [2, 50]  neighborhood size   (geometric extractor only)
        feature_mode:  categorical ['additive', 'append']     (geometric extractor only)
        alpha:         float [0.0, 10.0]                (additive mode only)
        beta:          float [0.0, 10.0]                (append mode only)

    Calling trial.suggest_* is idempotent within a trial, so the returned factory
    can be called multiple times (once per seed) and will always produce consistent
    hyperparameter values with fresh ICP/matcher instances.

    Args:
        trial:    Optuna trial for parameter suggestion.
        max_iter: Fixed ICP max_iter passed to each created instance.
        tol:      Fixed ICP convergence tolerance.

    Returns:
        Zero-argument callable that creates a fresh, configured ICP instance.
    """
    matching = trial.suggest_categorical("matching", ["hard", "soft"])

    if matching == "hard":
        return lambda: ICP(matcher=NearestNeighborMatcher(), max_iter=max_iter, tol=tol)

    # Feature extractor defaults (used when feature_extractor == 'none')
    feature_extractor: FeatureExtractor | None = None
    feature_mode: Literal['additive', 'append'] = 'additive'
    alpha = 1.0
    beta = 1.0

    feature_extractor_name = trial.suggest_categorical("feature_extractor", ["none", "geometric"])
    if feature_extractor_name == "geometric":
        fe_k = trial.suggest_int("fe_k", low=2, high=50)
        feature_extractor = GeometricFeatureExtractor(fe_k)
        feature_mode = cast(Literal['additive', 'append'], trial.suggest_categorical("feature_mode", ["additive", "append"]))
        if feature_mode == "additive":
            alpha = trial.suggest_float("alpha", low=0.0, high=10.0)
        else:
            beta = trial.suggest_float("beta", low=0.0, high=10.0)

    sigma_init = trial.suggest_float("sigma_init", 1.0, 50.0, log=True)
    sigma_ratio = trial.suggest_float("sigma_ratio", 0.01, 0.5, log=True)
    sigma_final = sigma_init * sigma_ratio
    anneal_steps = trial.suggest_int("anneal_steps", 50, max_iter)
    k = trial.suggest_int("k", 5, 30)
    trial.set_user_attr("sigma_final", sigma_final)

    def factory() -> ICP:
        matcher = GaussianMatcher(sigma=sigma_init, k=k, feature_extractor=feature_extractor,
                                  feature_mode=feature_mode, alpha=alpha, beta=beta)
        annealer = SigmaAnnealingCallback(sigma_init, sigma_final, anneal_steps)
        return ICP(matcher=matcher, max_iter=max_iter, tol=tol, callbacks=[annealer])

    return factory


def evaluate_icp(
    icp_factory: Callable[[], ICP],
    style: CloudStyle,
    n_seeds: int,
    gen_kwargs: dict[str, Any],
) -> dict[str, float]:
    """Evaluate an ICP configuration over multiple random seeds.

    Args:
        icp_factory: Callable returning a fresh ICP instance per call.
        style:       Cloud geometry ('random', 'clustered', 'lattice').
        n_seeds:     Number of seeds to average over. Seeds 0..n_seeds-1 are used.
        gen_kwargs:  Kwargs for SyntheticExperiment.generate (n, noise_std, t_scale).
                     Must not contain 'style' or 'seed'.

    Returns:
        Dictionary with:
            mean_rot_err: Mean rotation error in degrees across seeds.
            mean_t_err:   Mean translation error across seeds.
            reliability:  Fraction of seeds where rotation error < 5°.
    """
    rot_errs: list[float] = []
    t_errs: list[float] = []
    for seed in range(n_seeds):
        exp = SyntheticExperiment.generate(**gen_kwargs, style=style, seed=seed)
        result = icp_factory().fit(exp.P, exp.Q)
        T = result.transformation
        rot_errs.append(rotation_angle(exp.T_gt.R, T.R))
        t_errs.append(float(np.linalg.norm(exp.T_gt.t - T.t)))

    rot_arr, t_arr = np.array(rot_errs), np.array(t_errs)
    return {
        "mean_rot_err": float(rot_arr.mean()),
        "mean_t_err":   float(t_arr.mean()),
        "reliability":  float((rot_arr < 5.0).mean()),
    }


def make_objective(
    style: CloudStyle,
    n_seeds: int,
    gen_kwargs: dict[str, Any],
    max_iter: int,
    tol: float,
) -> Callable[[optuna.Trial], tuple[float, float]]:
    """Create an Optuna multi-objective function for a given cloud style.

    Both objectives are minimized: (mean_rot_err, mean_t_err).
    Reliability is stored as a trial user attribute for post-hoc inspection.

    Args:
        style:      Cloud geometry for SyntheticExperiment.generate.
        n_seeds:    Number of seeds per trial.
        gen_kwargs: Kwargs for SyntheticExperiment.generate (n, noise_std, t_scale).
        max_iter:   Fixed ICP max_iter.
        tol:        Fixed ICP tol.

    Returns:
        Callable (trial) -> (mean_rot_err, mean_t_err).
    """
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        factory = build_icp_factory(trial, max_iter, tol)
        metrics = evaluate_icp(factory, style, n_seeds, gen_kwargs)
        trial.set_user_attr("reliability", metrics["reliability"])
        return metrics["mean_rot_err"], metrics["mean_t_err"]

    return objective
