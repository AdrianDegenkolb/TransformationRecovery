from __future__ import annotations

from typing import Any, Callable, Literal, cast

import numpy as np
import optuna
from sklearn.cluster import DBSCAN

from experiment_runner import fit_multi_seed
from feature_extractor import FeatureExtractor, GeometricFeatureExtractor, RobustGeometricFeatureExtractor
from icp import ICP, ICPCallback, MultiStartICP, SigmaAnnealingCallback
from matcher import GaussianMatcher, Matcher, NearestNeighborMatcher
from synthetic import CloudStyle
from trimmer import ClusteringTrimmer, Trimmer

# Feature dimensionality per extractor, used to size the trimmer's DBSCAN eps
# via the (2*d)**0.5*0.2 heuristic used throughout the notebooks. Both
# extractors currently have a fixed dimension (RobustGeometricFeatureExtractor's
# only varies if `quantiles` is overridden, which HPO doesn't currently tune).
_FEATURE_EXTRACTOR_DIM: dict[str, int] = {"geometric": 9, "robust": 11}


def _build_feature_extractor(name: Literal["geometric", "robust"], k: int) -> FeatureExtractor:
    """Construct a feature extractor by name.

    Args:
        name: 'geometric' for GeometricFeatureExtractor, 'robust' for
              RobustGeometricFeatureExtractor.
        k:    Neighborhood size passed to the extractor.

    Returns:
        A configured FeatureExtractor instance.
    """
    if name == "geometric":
        return GeometricFeatureExtractor(k=k)
    return RobustGeometricFeatureExtractor(k=k)


def _build_matcher(trial: optuna.Trial, max_iter: int) -> tuple[Matcher, list[ICPCallback]]:
    """Suggest matching-strategy hyperparameters and build a Matcher.

    Feature extractor choice applies to both matching strategies: for 'hard'
    (NearestNeighborMatcher) it augments the single nearest-neighbor lookup via
    `beta`; for 'soft' (GaussianMatcher) it can additionally use additive
    cosine-similarity weighting via `alpha`, selected by `feature_mode`. Soft
    matching also anneals sigma over the run, returned as an ICPCallback.

    Args:
        trial:    Optuna trial for parameter suggestion.
        max_iter: Fixed ICP max_iter, used to bound anneal_steps (soft only).

    Returns:
        Tuple of (configured Matcher, callbacks to pass to ICP). callbacks is
        empty for hard matching.
    """
    matching = trial.suggest_categorical("matching", ["hard", "soft"])

    feature_extractor: FeatureExtractor | None = None
    feature_mode: Literal["additive", "append"] = "additive"
    alpha = 1.0
    beta = 1.0

    feature_extractor_name = trial.suggest_categorical("feature_extractor", ["none", "geometric", "robust"])
    if feature_extractor_name != "none":
        fe_k = trial.suggest_int("fe_k", low=2, high=50)
        feature_extractor = _build_feature_extractor(cast(Literal["geometric", "robust"], feature_extractor_name), fe_k)
        if matching == "soft":
            feature_mode = cast(Literal["additive", "append"], trial.suggest_categorical("feature_mode", ["additive", "append"]))
            if feature_mode == "additive":
                alpha = trial.suggest_float("alpha", low=0.0, high=10.0)
            else:
                beta = trial.suggest_float("beta", low=0.0, high=10.0)
        else:
            # NearestNeighborMatcher has no additive analog: features always
            # join the spatial KDTree via beta (see matcher.py).
            beta = trial.suggest_float("beta", low=0.0, high=10.0)

    if matching == "hard":
        return NearestNeighborMatcher(feature_extractor=feature_extractor, beta=beta), []

    sigma_init = trial.suggest_float("sigma_init", 1.0, 50.0, log=True)
    sigma_ratio = trial.suggest_float("sigma_ratio", 0.01, 0.5, log=True)
    sigma_final = sigma_init * sigma_ratio
    trial.set_user_attr("sigma_final", sigma_final)
    k = trial.suggest_int("k", 5, 30)
    anneal_steps = trial.suggest_int("anneal_steps", 50, max_iter)

    matcher = GaussianMatcher(
        sigma=sigma_init, k=k, feature_extractor=feature_extractor,
        feature_mode=feature_mode, alpha=alpha, beta=beta,
    )
    return matcher, [SigmaAnnealingCallback(sigma_init, sigma_final, anneal_steps)]


def build_icp_factory(
    trial: optuna.Trial,
    max_iter: int,
    tol: float,
    multistart_n_jobs: int = 1,
) -> Callable[[], ICP | MultiStartICP]:
    """Suggest hyperparameters from trial and return a factory for a fresh ICP/MultiStartICP.

    Tuned parameters:
        matching:           categorical ['hard', 'soft']
        feature_extractor:  categorical ['none', 'geometric', 'robust']
        fe_k:               int [2, 50]                        (feature_extractor != 'none')
        feature_mode:       categorical ['additive', 'append']  (soft + feature_extractor != 'none')
        alpha:              float [0.0, 10.0]                  (feature_mode == 'additive')
        beta:               float [0.0, 10.0]                  (feature_mode == 'append', or hard + feature_extractor != 'none')
        sigma_init:         log-uniform [1.0, 50.0]             (soft only)
        sigma_ratio:        log-uniform [0.01, 0.5]             (soft only); sigma_final = sigma_init * sigma_ratio
        anneal_steps:       int [50, max_iter]                  (soft only)
        k:                  int [5, 30]                         (soft only; GaussianMatcher candidate count)
        use_multistart:     categorical [True, False]
        n_starts:           log-int [2, 20]                     (use_multistart only)

    Calling trial.suggest_* is idempotent within a trial, so the returned factory
    can be called multiple times and will always produce consistent hyperparameter
    values with fresh instances.

    Args:
        trial:              Optuna trial for parameter suggestion.
        max_iter:           Fixed ICP max_iter passed to each created instance.
        tol:                Fixed ICP convergence tolerance.
        multistart_n_jobs:  Worker processes for MultiStartICP when use_multistart
                            is chosen. Defaults to 1 (no nested process pool) since
                            HPO trials may themselves already run in parallel.

    Returns:
        Zero-argument callable that creates a fresh, configured ICP or MultiStartICP instance.
    """
    matcher, callbacks = _build_matcher(trial, max_iter)

    use_multistart = trial.suggest_categorical("use_multistart", [True, False])
    n_starts = trial.suggest_int("n_starts", 2, 20, log=True) if use_multistart else None

    def factory() -> ICP | MultiStartICP:
        # evaluate_icp never reads cloud_history/matching_history.
        icp = ICP(matcher=matcher, max_iter=max_iter, tol=tol, callbacks=list(callbacks), record_history=False)
        if use_multistart:
            return MultiStartICP(icp=icp, n_starts=n_starts, n_jobs=multistart_n_jobs, verbose=False)
        return icp

    return factory


def build_trimmer(trial: optuna.Trial, n: int) -> Trimmer | None:
    """Suggest whether to apply a ClusteringTrimmer before ICP, and its hyperparameters.

    Tuned parameters:
        use_trimmer:          categorical [True, False]
        trimmer_extractor:    categorical ['geometric', 'robust']  (use_trimmer only)
        trimmer_fe_k:         int [2, 50]                          (use_trimmer only)
        min_cluster_fraction: log-uniform [0.01, 0.3]              (use_trimmer only)
        eps_scaling:          log-uniform [0.1, 0.5]               (use_trimmer only); DBSCAN eps = (2*d)**0.5 * eps_scaling
        min_samples_scaling:  log-uniform [0.5, 2.0]               (use_trimmer only); DBSCAN min_samples = log(n) * min_samples_scaling

    The trimmer's own feature extractor is independent of the matcher's: a
    matcher can run with no feature extractor while still being preceded by a
    feature-driven trim, since ClusteringTrimmer requires one regardless.

    Args:
        trial: Optuna trial for parameter suggestion.
        n:     Approximate point count per cloud (e.g. gen_kwargs['n']), used to
               size DBSCAN's min_samples via the same log(n) heuristic used in
               the notebooks.

    Returns:
        A configured ClusteringTrimmer, or None if trimming wasn't selected.
    """
    use_trimmer = trial.suggest_categorical("use_trimmer", [True, False])
    if not use_trimmer:
        return None

    extractor_name = cast(
        Literal["geometric", "robust"],
        trial.suggest_categorical("trimmer_extractor", ["geometric", "robust"]),
    )
    fe_k = trial.suggest_int("trimmer_fe_k", low=2, high=50)
    extractor = _build_feature_extractor(extractor_name, fe_k)

    min_cluster_fraction = trial.suggest_float("min_cluster_fraction", 0.01, 0.3, log=True)
    d = _FEATURE_EXTRACTOR_DIM[extractor_name]
    # eps/min_samples were previously fixed constants tuned for one clean-cloud
    # density; exposing their scale factors lets the search correct for point
    # density that shifts under dropout instead of assuming the notebook
    # heuristic still applies.
    eps_scaling = trial.suggest_float("eps_scaling", 0.1, 0.5, log=True)
    min_samples_scaling = trial.suggest_float("min_samples_scaling", 0.5, 2.0, log=True)
    eps = (2 * d) ** 0.5 * eps_scaling
    min_samples = max(2, int(np.log(n) * min_samples_scaling))

    return ClusteringTrimmer(
        feature_extractor=extractor,
        clusterer=DBSCAN(eps=eps, min_samples=min_samples),
        min_cluster_fraction=min_cluster_fraction,
    )


def evaluate_icp(
    icp_factory: Callable[[], ICP | MultiStartICP],
    style: CloudStyle,
    n_seeds: int,
    gen_kwargs: dict[str, Any],
    trimmer: Trimmer | None = None,
    dropout_prob: float = 0.0,
) -> dict[str, float]:
    """Evaluate an ICP configuration over multiple random seeds.

    Delegates the actual per-seed looping to `fit_multi_seed`, so trimming and
    dropout are handled consistently with the rest of the codebase.

    Args:
        icp_factory:  Callable returning a fresh ICP/MultiStartICP instance.
                      Called once; the returned instance is reused across seeds
                      (matches fit_multi_seed's contract elsewhere in the codebase).
        style:        Cloud geometry, see CloudStyle ('random', 'clustered',
                      'lattice', '2d-lattice', 'muscle-fiber').
        n_seeds:      Number of seeds to average over. Seeds 0..n_seeds-1 are used.
        gen_kwargs:   Kwargs for SyntheticExperiment.generate (n, noise_std, t_scale, ...).
                      Must not contain 'style' or 'seed'.
        trimmer:      Optional trimmer applied to (P, Q) before each ICP call.
        dropout_prob: Probability of dropping individual points from the observation.

    Returns:
        Dictionary with:
            mean_rot_err:   Mean rotation error in degrees across seeds.
            mean_t_err:     Mean translation error across seeds.
            mean_duration_s: Mean wall-clock fit duration across seeds.
            reliability:    Fraction of seeds where rotation error < 5°.
    """
    icp = icp_factory()
    result = fit_multi_seed(
        icp, seeds=list(range(n_seeds)), dropout_prob=dropout_prob, verbose=False,
        trimmer=trimmer, experiment_kwargs={**gen_kwargs, "style": style},
    )

    rot_arr = np.array(result.rotation_errors)
    t_arr = np.array(result.translation_errors)
    duration_arr = np.array(result.durations_s)
    return {
        "mean_rot_err":    float(rot_arr.mean()),
        "mean_t_err":      float(t_arr.mean()),
        "mean_duration_s": float(duration_arr.mean()),
        "reliability":     float((rot_arr < 5.0).mean()),
    }


def make_objective(
    style: CloudStyle,
    n_seeds: int,
    gen_kwargs: dict[str, Any],
    max_iter: int,
    tol: float,
    dropout_prob: float = 0.0,
    multistart_n_jobs: int = 1,
) -> Callable[[optuna.Trial], tuple[float, float, float]]:
    """Create an Optuna multi-objective function for a given cloud style.

    All three objectives are minimized: (mean_rot_err, mean_t_err, mean_duration_s).
    Including runtime as an objective (rather than only a tracked attribute) lets
    the Pareto front prefer the faster of two configurations that reach the same
    accuracy — e.g. fewer MultiStartICP starts when they're not needed for a
    given style. Reliability is stored as a trial user attribute for post-hoc
    inspection; it isn't itself optimized.

    Args:
        style:              Cloud geometry for SyntheticExperiment.generate.
        n_seeds:            Number of seeds per trial.
        gen_kwargs:         Kwargs for SyntheticExperiment.generate (n, noise_std, t_scale, ...).
        max_iter:           Fixed ICP max_iter.
        tol:                Fixed ICP tol.
        dropout_prob:       Probability of dropping individual points from the observation,
                            applied identically across all trials so the search optimizes
                            for this noise regime rather than only the clean case.
        multistart_n_jobs:  Worker processes for MultiStartICP trials. See build_icp_factory.

    Returns:
        Callable (trial) -> (mean_rot_err, mean_t_err, mean_duration_s).
    """
    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        factory = build_icp_factory(trial, max_iter, tol, multistart_n_jobs=multistart_n_jobs)
        trimmer = build_trimmer(trial, n=gen_kwargs.get("n", 2000))
        metrics = evaluate_icp(factory, style, n_seeds, gen_kwargs, trimmer=trimmer, dropout_prob=dropout_prob)
        trial.set_user_attr("reliability", metrics["reliability"])
        return metrics["mean_rot_err"], metrics["mean_t_err"], metrics["mean_duration_s"]

    return objective
