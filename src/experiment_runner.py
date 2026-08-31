"""Wires synthetic experiments (dropout, optional trimming) to ICP for evaluation."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from algebra_utils import rotation_angle
from icp import ICP, ICPResult, MultiStartICP, MultiStartICPResult
from matcher import NearestNeighborMatcher
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
    def residuals(self) -> list[NDArray[np.float64]]:
        """
        A list of residuals, one array per seed.

        Cached: computing this re-matches every seed's aligned cloud against its
        target via a fresh KDTree, which isn't cheap. `mean_residuals` depends on
        this property, so without caching, using both would redo the matching twice.
        """
        acc: list[NDArray[np.float64]] = []
        for exp, result in self.r.values():
            q_pred = result.transformation.apply(exp.P)
            nearest_matching = NearestNeighborMatcher().match(q_pred, exp.Q)
            residual = np.linalg.norm(nearest_matching.source_points - nearest_matching.target_positions, axis=1)
            acc.append(residual)
        return acc

    @cached_property
    def mean_residuals(self) -> list[float]:
        """
        A list of mean residual errors, one per seed
        """
        return [res.mean() for res in self.residuals]

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


def fit_multi_seed(
    icp: ICP | MultiStartICP,
    seeds: list[int],
    dropout_prob: float = 0.0,
    verbose: bool = True,
    trimmer: Trimmer | None = None,
    experiment_kwargs: dict[str, Any] | None = None,
) -> MultiSeedSyntheticICPResult:
    """Generate and solve a synthetic experiment per seed and report the results.

    Args:
        icp:               ICP or MultiStartICP instance to use for fitting.
        seeds:             List of random seeds, one experiment per seed.
        dropout_prob:      Probability of dropping individual points from the observation.
        verbose:           Show seed progress bar if True.
        trimmer:           Optional trimmer applied to (P, Q) before each ICP call.
        experiment_kwargs: Keyword arguments forwarded to SyntheticExperiment.generate.

    Returns:
        MultiSeedSyntheticICPResult with one experiment and result per seed.
    """
    results: dict[int, tuple[SyntheticExperiment, ICPResult | MultiStartICPResult]] = {}
    experiment_kwargs = experiment_kwargs or {}
    with _quiet(icp):
        for seed in tqdm(seeds, desc=f"Solving {len(seeds)} seeds", disable=not verbose):
            exp = SyntheticExperiment.generate(**experiment_kwargs, seed=seed)
            p, q = exp.observe_point_clouds(dropout_prob)
            if trimmer is not None:
                p, q = trimmer.trim([p, q])
            result = icp.fit(p, q)
            results[seed] = (exp, result)

    return MultiSeedSyntheticICPResult(results)
