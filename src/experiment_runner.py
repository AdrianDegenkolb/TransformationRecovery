"""Wires synthetic experiments (dropout, optional trimming) to ICP for evaluation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate
from tqdm import tqdm

from algebra_utils import rotation_angle
from error_metrics import convergence_to_global_opt_ratio, nearest_neighbor_residuals, true_residuals as compute_true_residuals
from icp import ICP, ICPResult, MultiStartICP, MultiStartICPResult, quiet
from synthetic import PerfectObserver, PointCloudObserver, SyntheticExperiment
from transformation import RigidTransformation
from trimmer import Trimmer


@dataclass
class MultiSeedSyntheticICPResult:
    """
    Holds reference to a synthetic experiment and (MultiStart)ICPResult object each for a set of seeds.
    """
    r: dict[int, tuple[SyntheticExperiment, ICPResult | MultiStartICPResult]] = field(default_factory=dict)

    def __getitem__(self, seed: int) -> tuple[SyntheticExperiment, ICPResult | MultiStartICPResult]:
        return self.r[seed]

    @property
    def results(self) -> list[ICPResult | MultiStartICPResult]:
        """
        A list of (MultiStart)ICPResults, one per seed
        """
        return [t[1] for t in self.r.values()]

    @property
    def experiments(self) -> list[SyntheticExperiment]:
        """
        A list of SyntheticExperiments, one per seed
        """
        return [t[0] for t in self.r.values()]

    @property
    def ground_truths(self) -> list[RigidTransformation]:
        """
        A list of ground truth transformations, one per seed
        """
        return [t[0].T_gt for t in self.r.values()]

    @property
    def rotation_errors(self) -> list[float]:
        """
        A list of rotation errors, one per seed
        """
        return [rotation_angle(res.transformation.R, gt.R) for res, gt in zip(self.results, self.ground_truths)]

    @property
    def translation_errors(self) -> list[float]:
        """
        A list of translation errors, one per seed
        """
        return [float(np.linalg.norm(res.transformation.t - gt.t)) for res, gt in zip(self.results, self.ground_truths)]

    @cached_property
    def closest_point_residuals(self) -> list[NDArray[np.float64]]:
        """
        A list of closest-point (nearest-neighbor-matched) residuals, one array per seed.

        Cached: computing this re-matches every seed's aligned cloud against its
        target via a fresh KDTree, which isn't cheap. `mean_closest_point_residuals`
        depends on this property, so without caching, using both would redo the
        matching twice.
        """
        return [
            nearest_neighbor_residuals(result.transformation.apply(exp.P), exp.Q)
            for exp, result in self.r.values()
        ]

    @cached_property
    def mean_closest_point_residuals(self) -> list[float]:
        """
        A list of mean closest-point residual errors, one per seed
        """
        return [res.mean() for res in self.closest_point_residuals]

    @cached_property
    def true_residuals(self) -> list[NDArray[np.float64]]:
        """
        A list of per-point true residuals, one array per seed.

        Unlike `closest_point_residuals`, this needs no nearest-neighbor matching: P
        and Q share point-for-point ground-truth correspondence by construction (both
        are transformations of the same source cloud S, see
        SyntheticExperiment.generate), and that correspondence survives dropout and noise because
        the PointCloudObserver only degrades the copies used for fitting
        The true residual ||T_pred(exp.P)_i - exp.Q_i|| directly
        measures pure transformation-recovery error with no irreducible noise
        floor — it is 0 iff `result.transformation` exactly equals `exp.T_gt`.
        Only meaningful for synthetic experiments with known correspondence;
        real-world data would need `closest_point_residuals` instead.
        """
        return [
            compute_true_residuals(result.transformation.apply(exp.P), exp.Q)
            for exp, result in self.r.values()
        ]

    @cached_property
    def convergence_to_global_opt_ratio(self) -> float:
        """
        The fraction of seeds that converged to the global optimum (mean true residual < 1e-1).
        """
        return convergence_to_global_opt_ratio(self.mean_true_residuals, tol=0.1)

    @cached_property
    def mean_true_residuals(self) -> list[float]:
        """
        A list of mean true-residual errors, one per seed
        """
        return [res.mean() for res in self.true_residuals]

    @property
    def durations_s(self) -> list[float]:
        """
        A list of durations, one per seed
        """
        return [res.duration_s for res in self.results]

    @property
    def deltas(self) -> list[list[float]]:
        """
        A list of deltas, one per seed
        """
        return [res.deltas for res in self.results]


def _fit_one_seed(
    icp: ICP | MultiStartICP,
    seed: int,
    observer: PointCloudObserver,
    trimmer: Trimmer | None,
    experiment_kwargs: dict[str, Any],
) -> tuple[int, SyntheticExperiment, ICPResult | MultiStartICPResult]:
    """Generate and solve a single seed's experiment.

    Module-level (rather than a closure in fit_multi_seed) so it can be pickled
    and sent to worker processes by ProcessPoolExecutor.
    """
    exp = SyntheticExperiment.generate(**experiment_kwargs, seed=seed)
    p, q = observer.observe(exp.P), observer.observe(exp.Q)
    if trimmer is not None:
        p, q = trimmer.trim([p, q])
    result = icp.fit(p, q)
    return seed, exp, result


def fit_multi_seed(
    icp: ICP | MultiStartICP,
    seeds: list[int],
    experiment_kwargs: dict[str, Any] | None = None,
    observer: PointCloudObserver = PerfectObserver(),
    trimmer: Trimmer | None = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> MultiSeedSyntheticICPResult:
    """Generate and solve a synthetic experiment per seed and report the results.

    Args:
        icp:               ICP or MultiStartICP instance to use for fitting.
        seeds:             List of random seeds, one experiment per seed.
        experiment_kwargs: Keyword arguments forwarded to SyntheticExperiment.generate.
        observer:          Returns imperfect observations of each experiment's points clouds P and Q. Can be used to test more realistic scenarios. Defaults to PerfectObserver, which returns the original clouds without any dropout or noise.
        trimmer:           Optional trimmer applied to (P, Q) before each ICP call.
        n_jobs:            Worker processes for parallelizing across seeds. 1 (default)
                            runs sequentially in-process. -1
                            uses os.cpu_count(). If `icp` is a MultiStartICP, its own
                            `n_jobs` already parallelizes across starts within a single
                            seed; combining that with n_jobs != 1 here nests process
                            pools and oversubscribes CPU cores, so parallelize only one
                            of the two loops.
        verbose:           Show seed progress bar if True.

    Returns:
        MultiSeedSyntheticICPResult with one experiment and result per seed.
    """
    experiment_kwargs = experiment_kwargs or {}

    with quiet(icp):
        if n_jobs == 1:
            results: dict[int, tuple[SyntheticExperiment, ICPResult | MultiStartICPResult]] = {}
            pbar = tqdm(seeds, desc=f"Solving {len(seeds)} seeds", disable=not verbose)
            for seed in pbar:
                _, exp, result = _fit_one_seed(icp, seed, observer.spawn(), trimmer, experiment_kwargs)
                results[seed] = (exp, result)
                # Show the current reliability (fraction of seeds converged to the global opt) in the progress bar.
                reliability = MultiSeedSyntheticICPResult(results).convergence_to_global_opt_ratio
                pbar.set_postfix(reliability=f"{reliability:.0%}")
        else:
            max_workers = n_jobs if n_jobs > 0 else None
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                future_to_seed = {
                    pool.submit(_fit_one_seed, icp, seed, observer.spawn(), trimmer, experiment_kwargs): seed
                    for seed in seeds
                }
                unordered: dict[int, tuple[SyntheticExperiment, ICPResult | MultiStartICPResult]] = {}
                pbar = tqdm(
                    as_completed(future_to_seed), total=len(seeds), desc=f"Solving {len(seeds)} seeds", disable=not verbose
                )
                for future in pbar:
                    _, exp, result = future.result()
                    unordered[future_to_seed[future]] = (exp, result)
                    # Show the current reliability (fraction of seeds converged to the global opt) in the progress bar.
                    reliability = MultiSeedSyntheticICPResult(unordered).convergence_to_global_opt_ratio
                    pbar.set_postfix(reliability=f"{reliability:.0%}")
                    
            # Re-order to match the input seed order, independent of completion order.
            results = {seed: unordered[seed] for seed in seeds}

    return MultiSeedSyntheticICPResult(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run synthetic experiments and ICP on them.")
    parser.add_argument("--n-points", type=int, default=2000, help="Number of random seeds to run.")
    parser.add_argument("--multi-start", default=False,action="store_true", help="Use MultiStartICP instead of ICP.")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of random seeds to run.")
    parser.add_argument("--n-starts", type=int, default=20, help="Number of random seeds to run.")
    parser.add_argument("--verbose", default=True, action="store_true", help="Show progress bar.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of worker processes for parallelization. 1 (default) runs sequentially. -1 uses os.cpu_count().")
    args = parser.parse_args()

    experiment_kwargs: dict[str, Any] = {'n': args.n_points, 't_scale': 8.0, 'style': 'muscle-fiber'}
    icp = ICP()
    if args.multi_start:
        icp = icp.to_multi_start(n_starts=args.n_starts, n_jobs=args.n_jobs)
        
    result = fit_multi_seed(icp, seeds=list(range(args.n_seeds)), verbose=args.verbose, n_jobs=1, experiment_kwargs=experiment_kwargs)

    print("Results:")
    print(tabulate(
        zip(result.rotation_errors, result.translation_errors, result.mean_true_residuals, result.mean_closest_point_residuals, result.durations_s),
        headers=["Rotation Error (°)", "Translation Error", "Mean True Residual", "Mean Closest-Point Residual", "Duration (s)"],
        floatfmt=".2f", tablefmt="rounded_outline"
    ))
