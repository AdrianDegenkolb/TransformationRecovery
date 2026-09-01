from __future__ import annotations

import time
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from algebra_utils import sample_dispersed_rotations
from error_metrics import get_residuals
from matcher import Matcher, Matching, NearestNeighborMatcher
from point_cloud import PointCloud
from transformation import RigidTransformation


class ICPCallback(ABC):
    """Base class for hooks called at each ICP iteration."""

    @abstractmethod
    def on_iteration_start(self, iteration: int, icp: ICP) -> None:
        """Called at the start of each iteration, before the E-step.

        Args:
            iteration: Zero-based iteration index.
            icp:       The running ICP instance (access icp.matcher to update it).
        """
        ...


class SigmaAnnealingCallback(ICPCallback):
    """Exponentially anneals the sigma of a GaussianMatcher over ICP iterations.

    At the start of iteration i, sets:
        matcher.sigma = sigma_init * (sigma_final / sigma_init) ^ (i / anneal_steps)

    This transitions the matcher from a blurred global view (large sigma) to a
    near-hard assignment (small sigma), without any state inside the matcher itself.
    """

    def __init__(
        self,
        sigma_init: float,
        sigma_final: float,
        anneal_steps: int,
    ):
        """
        Args:
            sigma_init:   Starting bandwidth (large → soft/global).
            sigma_final:  Ending bandwidth (small → near-hard).
            anneal_steps: Number of iterations over which to anneal.
                          Typically, this is set equal to ICP max_iter.
        """
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.anneal_steps = anneal_steps

    def on_iteration_start(self, iteration: int, icp: ICP) -> None:
        """Update icp.matcher.sigma for the current iteration.

        Args:
            iteration: Zero-based iteration index.
            icp:       The running ICP instance; its matcher must be a GaussianMatcher.
        """
        t = min(iteration, self.anneal_steps - 1) / max(self.anneal_steps - 1, 1)
        icp.matcher.sigma = float(
            self.sigma_init * (self.sigma_final / self.sigma_init) ** t
        )


@dataclass
class ICPResult:
    """Result of an ICP run.

    Attributes:
        transformation:      Accumulated rigid transformation mapping source onto target.
        n_iterations:        Number of EM iterations performed.
        duration_s:          Duration in seconds until convergence or n_iterations reached.
        converged:           Whether the algorithm converged before max_iter.
        mean_residuals:      Mean point-to-point residual after each M-step.
        cloud_history:       Source cloud state at the start of each iteration.
                             Empty if the ICP instance was created with record_history=False.
        matching_history:    Matching from the E-step of each iteration.
                             Empty if the ICP instance was created with record_history=False.
        transform_history:   Accumulated transformation after each M-step.
        deltas:              Per step delta. ICP is considered converged if
                             delta = ||last_10_transformation.R - I||_F + ||last_10_transformation.t||_2 < tolerance
    """

    transformation: RigidTransformation
    n_iterations: int
    duration_s: float
    converged: bool
    mean_residuals: list[float] = field(default_factory=list)
    cloud_history: list[PointCloud] = field(default_factory=list)
    matching_history: list[Matching] = field(default_factory=list)
    transform_history: list[RigidTransformation] = field(default_factory=list)
    deltas: list[float] = field(default_factory=list)

    def __repr__(self):
        rows: list[tuple[str, int | RigidTransformation]] = [
            ("Converged",  self.converged),
            ("Iterations", self.n_iterations),
            ("Recovered",  self.transformation),
        ]
        return tabulate(rows, tablefmt="rounded_outline")

@dataclass
class MultiStartICPResult:
    """Result of a multi-start ICP run.

    Attributes:
        best:                   ICPResult with the lowest final residual.
        best_initial_rotation:  The SO(3) seed that produced the best result.
        all_results:            ICPResult for every starting rotation.
        all_initial_rotations:  All sampled starting rotations (3, 3) each.
        duration_s:             Duration in seconds until all workers convergence or reach n_iterations.
    """

    best: ICPResult
    best_initial_rotation: NDArray[np.float64]
    all_results: list[ICPResult]
    all_initial_rotations: list[NDArray[np.float64]]
    duration_s: float

    def __repr__(self) -> str:
        rows: list[tuple[str, int | RigidTransformation]] = [
            ("Starts",                  len(self.all_results)),
            ("Converged",               sum(r.converged for r in self.all_results)),
            ("Best Transformation",     self.best.transformation),
        ]
        return tabulate(rows, tablefmt="rounded_outline")

    @property
    def individual_durations_summed(self) -> float:
        return sum([result.duration_s for result in self.all_results])

    @property
    def num_workers(self) -> int:
        return len(self.all_results)

    @property
    def cpu_efficiency(self) -> float:
        return self.individual_durations_summed / self.duration_s * self.num_workers

    def __getattr__(self, name: str):
        # forward attribute access to the best result
        if hasattr(self.best, name):
            return getattr(self.best, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def _windowed_delta(
    transform_history: list[RigidTransformation],
    accumulated: RigidTransformation,
    window: int = 10,
) -> float:
    """Convergence delta: how far the last `window` steps' composition is from identity.

    Args:
        transform_history: Accumulated transformation after each prior M-step.
        accumulated:        Current accumulated transformation.
        window:              Number of trailing steps to compose over. If fewer
                             than `window` steps have elapsed, uses `accumulated`
                             directly (i.e. compares against the identity from
                             the very start of the run).

    Returns:
        ||R - I||_F + ||t||_2, where R, t come from composing `accumulated`
        with the inverse of the transformation from `window` steps ago.
    """
    if len(transform_history) >= window:
        ref = transform_history[-window]
        recent = accumulated.compose(ref.inverse())
    else:
        recent = accumulated
    return float(np.linalg.norm(recent.R - np.eye(3), ord="fro") + np.linalg.norm(recent.t))


class ICP:
    """EM algorithm for rigid point cloud registration without known correspondences.

    E-step: establish point correspondences via a Matcher.
    M-step: fit a RigidTransformation via weighted SVD Procrustes.
    Callbacks: called at the start of each iteration (e.g. for sigma annealing).

    Convergence is declared when the composed transformation of the last 10 steps is near identity:
        delta = ||R_step - I||_F + ||t_step|| < tol
    """

    def __init__(
        self,
        matcher: Matcher | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
        callbacks: list[ICPCallback] | None = None,
        record_history: bool = True,
    ):
        """
        Args:
            matcher:        Correspondence algorithm for the E-step.
                            Defaults to NearestNeighborMatcher.
            max_iter:       Maximum number of EM iterations.
            tol:            Convergence threshold on ||R_step - I||_F + ||t_step||.
            verbose:        Show a progress bar if True.
            callbacks:      Optional list of ICPCallback instances called before
                            each E-step (e.g. SigmaAnnealingCallback).
            record_history: If False, skip recording cloud_history and
                            matching_history (both O(n_points) per iteration).
                            Set to False for large sweeps that only need the
                            final transformation, to avoid retaining a full
                            point cloud + correspondence set per iteration per
                            trial. mean_residuals/transform_history/deltas are
                            always recorded (cheap, and needed for convergence).
        """
        self.matcher = matcher or NearestNeighborMatcher()
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.callbacks = callbacks or []
        self.record_history = record_history

    def fit(self, source: PointCloud, target: PointCloud) -> ICPResult:
        """Run ICP to find the rigid transformation mapping source onto target.

        Args:
            source: Source PointCloud (N, 3).
            target: Target PointCloud (M, 3).

        Returns:
            ICPResult with the accumulated transformation, convergence info,
            and per-iteration history.
        """
        current = source
        accumulated = RigidTransformation.identity()

        mean_residuals: list[float] = []
        cloud_history: list[PointCloud] = []
        matching_history: list[Matching] = []
        transform_history: list[RigidTransformation] = []
        deltas: list[float] = []

        t0 = time.perf_counter()
        pbar = tqdm(range(self.max_iter), desc="ICP", disable=not self.verbose)
        for i in pbar:
            for cb in self.callbacks:
                cb.on_iteration_start(i, self)

            matching = self.matcher.match(current, target)
            src_pc = PointCloud(matching.source_points)
            tgt_pc = PointCloud(matching.target_positions)
            transformation = RigidTransformation.fit(src_pc, tgt_pc, weights=matching.weights)
            accumulated = transformation.compose(accumulated)
            residual = float(get_residuals(matching, transformation.apply(src_pc)).mean())
            delta = _windowed_delta(transform_history, accumulated)

            pbar.set_postfix(residual=f"{residual:.4f}")
            if self.record_history:
                cloud_history.append(current)
                matching_history.append(matching)
            mean_residuals.append(residual)
            transform_history.append(accumulated)
            deltas.append(delta)
            current = transformation.apply(current)

            if delta < self.tol:
                pbar.set_description("ICP converged")
                return ICPResult(
                    transformation=accumulated, n_iterations=i + 1, converged=True,
                    mean_residuals=mean_residuals, cloud_history=cloud_history, matching_history=matching_history,
                    transform_history=transform_history, duration_s=time.perf_counter() - t0, deltas=deltas
                )

        pbar.set_description("ICP did not converge")
        return ICPResult(
            transformation=accumulated, n_iterations=self.max_iter, converged=False,
            mean_residuals=mean_residuals, cloud_history=cloud_history, matching_history=matching_history,
            transform_history=transform_history, duration_s=time.perf_counter() - t0, deltas=deltas
        )


class MultiStartICP:
    """Runs ICP from multiple dispersed starting rotations and returns the best result.

    Wraps an existing ICP instance. For each start, the source cloud is
    pre-rotated by one of a set of SO(3) rotations chosen via rotation_sampler
    (greedy farthest-point selection by default, see sample_dispersed_rotations)
    before running ICP, so the starts are spread across rotation space rather
    than left to chance. The recovered transformations are composed with the
    initial rotation so that all results refer to the original (un-rotated)
    source.

    Trials are executed in parallel via ProcessPoolExecutor.
    """

    def __init__(
        self,
        icp: ICP,
        n_starts: int = 20,
        n_jobs: int = -1,
        seed: int | None = None,
        verbose: bool = True,
        residual_threshold: float = 1e-3,
        rotation_sampler: Callable[[int, np.random.Generator], list[NDArray[np.float64]]] = sample_dispersed_rotations,
    ):
        """
        Args:
            icp:                 Configured ICP instance reused across all trials.
            n_starts:            Number of starting rotations to try.
            n_jobs:              Worker processes. -1 uses os.cpu_count().
            seed:                Optional random seed for reproducible rotation sampling.
            verbose:             Show a progress bar if True.
            residual_threshold:  Mean residual below which a converged trial
                                 triggers early stopping of remaining trials.
            rotation_sampler:    Callable(n, rng) -> list of n (3, 3) SO(3) rotations
                                 used to seed the starts. Defaults to greedy
                                 farthest-point sampling; pass e.g.
                                 sample_uniform_rotations for plain i.i.d. sampling.
        """
        self.icp = icp
        self.n_starts = n_starts
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose
        self.residual_threshold = residual_threshold
        self.rotation_sampler = rotation_sampler

    def fit(self, source: PointCloud, target: PointCloud) -> MultiStartICPResult:
        """Run ICP from n_starts rotations (via rotation_sampler) and return the best result.

        Args:
            source: Source PointCloud (N, 3).
            target: Target PointCloud (M, 3).

        Returns:
            MultiICPResult containing the best ICPResult and all trial results.
        """
        rng = np.random.default_rng(self.seed)
        rotations = self.rotation_sampler(self.n_starts, rng)

        all_results: list[ICPResult] = []
        all_rotations: list[NDArray[np.float64]] = []

        max_workers = self.n_jobs if self.n_jobs > 0 else None
        t0 = time.perf_counter()
        pbar = tqdm(total=self.n_starts, desc=f"Testing {self.n_starts} starting configurations", disable=not self.verbose)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._run_single, self.icp, source, target, R): R
                for R in rotations
            }
            for future in as_completed(futures):
                pbar.update(1)
                result, R_init = future.result()
                all_results.append(result)
                all_rotations.append(R_init)
                if result.converged and result.mean_residuals[-1] < self.residual_threshold:
                    pool.shutdown(cancel_futures=True)
                    break

        pbar.close()
        best_idx = int(np.argmin([r.mean_residuals[-1] for r in all_results]))
        return MultiStartICPResult(
            best=all_results[best_idx],
            best_initial_rotation=all_rotations[best_idx],
            all_results=all_results,
            all_initial_rotations=all_rotations,
            duration_s=time.perf_counter() - t0
        )

    @staticmethod
    def _run_single(
            icp: ICP,
            source: PointCloud,
            target: PointCloud,
            R_init: NDArray[np.float64],
    ) -> tuple[ICPResult, NDArray[np.float64]]:
        """Run one ICP trial from a pre-rotation R_init and compose the result.

        Args:
            icp:    Configured ICP instance.
            source: Original source PointCloud.
            target: Target PointCloud.
            R_init: (3, 3) initial rotation applied to source before ICP.

        Returns:
            Tuple of (ICPResult with composed transformation, R_init).
        """
        init_tf = RigidTransformation(R_init, np.zeros(3))
        rotated_source = init_tf.apply(source)
        result = icp.fit(rotated_source, target)
        result.transformation = result.transformation.compose(init_tf)
        result.transform_history = [T.compose(init_tf) for T in result.transform_history]
        return result, R_init
