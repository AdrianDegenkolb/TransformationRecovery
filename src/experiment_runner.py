"""Wires synthetic experiments (dropout, optional trimming) to ICP for evaluation."""

from __future__ import annotations

from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from algebra_utils import rotation_angle
from error_metrics import nearest_neighbor_residuals, true_residuals as compute_true_residuals
from icp import ICP, ICPResult, MultiStartICP, MultiStartICPResult
from synthetic import SyntheticExperiment
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
        SyntheticExperiment.generate), and that correspondence survives dropout because
        `observe_point_clouds` only drops points from the copies used for fitting,
        leaving exp.P/exp.Q full-length and index-aligned. So the true residual is
        directly ||T_pred(exp.P)_i - exp.Q_i||. Only meaningful for synthetic
        experiments with known correspondence; real-world data would need
        `closest_point_residuals` instead.
        """
        return [
            compute_true_residuals(result.transformation.apply(exp.P), exp.Q)
            for exp, result in self.r.values()
        ]

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


@contextmanager
def _quiet(icp: ICP | MultiStartICP) -> Generator[None, None, None]:
    """Temporarily disable progress bars on `icp` for the duration of the block.

    For a `MultiStartICP`, also silences the wrapped per-start `ICP` instance
    (`icp.icp`), since it has its own independent `verbose` flag. Restores the
    original value(s) even if the block raises.

    Args:
        icp: ICP or MultiStartICP instance to silence.
    """
    targets = [icp, icp.icp] if isinstance(icp, MultiStartICP) else [icp]
    originals = [t.verbose for t in targets]
    for t in targets:
        t.verbose = False
    try:
        yield
    finally:
        for t, original in zip(targets, originals):
            t.verbose = original


def _fit_one_seed(
    icp: ICP | MultiStartICP,
    seed: int,
    dropout_prob: float,
    trimmer: Trimmer | None,
    experiment_kwargs: dict[str, Any],
) -> tuple[int, SyntheticExperiment, ICPResult | MultiStartICPResult]:
    """Generate and solve a single seed's experiment.

    Module-level (rather than a closure in fit_multi_seed) so it can be pickled
    and sent to worker processes by ProcessPoolExecutor.
    """
    exp = SyntheticExperiment.generate(**experiment_kwargs, seed=seed)
    p, q = exp.observe_point_clouds(dropout_prob)
    if trimmer is not None:
        p, q = trimmer.trim([p, q])
    result = icp.fit(p, q)
    return seed, exp, result


def fit_multi_seed(
    icp: ICP | MultiStartICP,
    seeds: list[int],
    dropout_prob: float = 0.0,
    verbose: bool = True,
    trimmer: Trimmer | None = None,
    experiment_kwargs: dict[str, Any] | None = None,
    n_jobs: int = 1,
) -> MultiSeedSyntheticICPResult:
    """Generate and solve a synthetic experiment per seed and report the results.

    Args:
        icp:               ICP or MultiStartICP instance to use for fitting.
        seeds:             List of random seeds, one experiment per seed.
        dropout_prob:      Probability of dropping individual points from the observation.
        verbose:           Show seed progress bar if True.
        trimmer:           Optional trimmer applied to (P, Q) before each ICP call.
        experiment_kwargs: Keyword arguments forwarded to SyntheticExperiment.generate.
        n_jobs:            Worker processes for parallelizing across seeds. 1 (default)
                            runs sequentially in-process. -1
                            uses os.cpu_count(). If `icp` is a MultiStartICP, its own
                            `n_jobs` already parallelizes across starts within a single
                            seed; combining that with n_jobs != 1 here nests process
                            pools and oversubscribes CPU cores, so parallelize only one
                            of the two loops.

    Returns:
        MultiSeedSyntheticICPResult with one experiment and result per seed.
    """
    experiment_kwargs = experiment_kwargs or {}

    with _quiet(icp):
        if n_jobs == 1:
            results: dict[int, tuple[SyntheticExperiment, ICPResult | MultiStartICPResult]] = {}
            for seed in tqdm(seeds, desc=f"Solving {len(seeds)} seeds", disable=not verbose):
                _, exp, result = _fit_one_seed(icp, seed, dropout_prob, trimmer, experiment_kwargs)
                results[seed] = (exp, result)
        else:
            max_workers = n_jobs if n_jobs > 0 else None
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                future_to_seed = {
                    pool.submit(_fit_one_seed, icp, seed, dropout_prob, trimmer, experiment_kwargs): seed
                    for seed in seeds
                }
                unordered: dict[int, tuple[SyntheticExperiment, ICPResult | MultiStartICPResult]] = {}
                for future in tqdm(
                    as_completed(future_to_seed), total=len(seeds), desc=f"Solving {len(seeds)} seeds", disable=not verbose
                ):
                    _, exp, result = future.result()
                    unordered[future_to_seed[future]] = (exp, result)
            # Re-order to match the input seed order, independent of completion order.
            results = {seed: unordered[seed] for seed in seeds}

    return MultiSeedSyntheticICPResult(results)
