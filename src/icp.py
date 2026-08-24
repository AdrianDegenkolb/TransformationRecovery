from __future__ import annotations
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from algebra_utils import sample_uniform_rotations
from point_cloud import PointCloud
from transformation import RigidTransformation
from matcher import Matcher, Matching, NearestNeighborMatcher, GaussianMatcher


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
        matcher: GaussianMatcher,
        sigma_init: float,
        sigma_final: float,
        anneal_steps: int,
    ):
        """
        Args:
            matcher:      GaussianMatcher whose sigma will be updated.
            sigma_init:   Starting bandwidth (large → soft/global).
            sigma_final:  Ending bandwidth (small → near-hard).
            anneal_steps: Number of iterations over which to anneal.
                          Typically set equal to ICP max_iter.
        """
        self.matcher = matcher
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.anneal_steps = anneal_steps

    def on_iteration_start(self, iteration: int, icp: ICP) -> None:
        """Update matcher.sigma for the current iteration."""
        t = min(iteration, self.anneal_steps - 1) / max(self.anneal_steps - 1, 1)
        self.matcher.sigma = float(
            self.sigma_init * (self.sigma_final / self.sigma_init) ** t
        )


@dataclass
class ICPResult:
    """Result of an ICP run.

    Attributes:
        transformation:      Accumulated rigid transformation mapping source onto target.
        n_iterations:        Number of EM iterations performed.
        converged:           Whether the algorithm converged before max_iter.
        mean_residuals:      Mean point-to-point residual after each M-step.
        cloud_history:       Source cloud state at the start of each iteration.
        matching_history:  Matching from the E-step of each iteration.
        transform_history:   Accumulated transformation after each M-step.
    """

    transformation: RigidTransformation
    n_iterations: int
    converged: bool
    mean_residuals: list[float] = field(default_factory=list)
    cloud_history: list[PointCloud] = field(default_factory=list)
    matching_history: list[Matching] = field(default_factory=list)
    transform_history: list[RigidTransformation] = field(default_factory=list)

    def __repr__(self):
        rows = [
            ["Converged",  self.converged],
            ["Iterations", self.n_iterations],
            ["Recovered",  self.transformation],
        ]
        return tabulate(rows, tablefmt="rounded_outline")


# ---------------------------------------------------------------------------
# ICP
# ---------------------------------------------------------------------------

class ICP:
    """EM algorithm for rigid point cloud registration without known correspondences.

    E-step: establish point correspondences via a Matcher.
    M-step: fit a RigidTransformation via weighted SVD Procrustes.
    Callbacks: called at the start of each iteration (e.g. for sigma annealing).

    Convergence is declared when the step transformation is near identity:
        ||R_step - I||_F + ||t_step|| < tol
    """

    def __init__(
        self,
        matcher: Matcher | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
        callbacks: list[ICPCallback] | None = None,
    ):
        """
        Args:
            matcher:   Correspondence algorithm for the E-step.
                       Defaults to NearestNeighborMatcher.
            max_iter:  Maximum number of EM iterations.
            tol:       Convergence threshold on ||R_step - I||_F + ||t_step||.
            verbose:   Show a progress bar if True.
            callbacks: Optional list of ICPCallback instances called before
                       each E-step (e.g. SigmaAnnealingCallback).
        """
        self.matcher = matcher or NearestNeighborMatcher()
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.callbacks = callbacks or []

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

        pbar = tqdm(range(self.max_iter), desc="ICP", disable=not self.verbose)
        for i in pbar:
            for cb in self.callbacks:
                cb.on_iteration_start(i, self)

            matching = self.matcher.match(current, target)
            src_pc = PointCloud(matching.source_points)
            tgt_pc = PointCloud(matching.target_positions)
            transformation = RigidTransformation.fit(src_pc, tgt_pc, weights=matching.weights)
            accumulated = transformation.compose(accumulated)
            residual = float(transformation.residuals(src_pc, tgt_pc).mean())

            pbar.set_postfix(residual=f"{residual:.4f}")
            cloud_history.append(current)
            matching_history.append(matching)
            mean_residuals.append(residual)
            transform_history.append(accumulated)

            current = transformation.apply(current)

            delta = np.linalg.norm(transformation.R - np.eye(3), ord="fro") + np.linalg.norm(transformation.t)
            if delta < self.tol:
                pbar.set_description("ICP converged")
                return ICPResult(
                    transformation=accumulated, n_iterations=i + 1, converged=True,
                    mean_residuals=mean_residuals, cloud_history=cloud_history, matching_history=matching_history,
                    transform_history=transform_history
                )

        pbar.set_description("ICP did not converge")
        return ICPResult(
            transformation=accumulated, n_iterations=self.max_iter, converged=False,
            mean_residuals=mean_residuals, cloud_history=cloud_history, matching_history=matching_history,
            transform_history=transform_history
        )


@dataclass
class MultiICPResult:
    """Result of a multi-start ICP run.

    Attributes:
        best:                   ICPResult with the lowest final residual.
        best_initial_rotation:  The SO(3) seed that produced the best result.
        all_results:            ICPResult for every starting rotation.
        all_initial_rotations:  All sampled starting rotations (3, 3) each.
    """

    best: ICPResult
    best_initial_rotation: np.ndarray
    all_results: list[ICPResult]
    all_initial_rotations: list[np.ndarray]

    def __repr__(self) -> str:
        rows = [
            ["Starts",    len(self.all_results)],
            ["Converged", sum(r.converged for r in self.all_results)],
            ["Best",      self.best],
        ]
        return tabulate(rows, tablefmt="rounded_outline")


class MultiStartICP:
    """Runs ICP from multiple random starting rotations and returns the best result.

    Wraps an existing ICP instance. For each start, the source cloud is
    pre-rotated by a uniformly sampled SO(3) rotation before running ICP.
    The recovered transformations are composed with the initial rotation so
    that all results refer to the original (un-rotated) source.

    Trials are executed in parallel via ProcessPoolExecutor.
    """

    def __init__(
        self,
        icp: ICP,
        n_starts: int = 20,
        n_jobs: int = -1,
        seed: int | None = None,
    ):
        """
        Args:
            icp:      Configured ICP instance reused across all trials.
            n_starts: Number of random starting rotations to try.
            n_jobs:   Worker processes. -1 uses os.cpu_count().
            seed:     Optional random seed for reproducible rotation sampling.
        """
        self.icp = icp
        self.n_starts = n_starts
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(self, source: PointCloud, target: PointCloud) -> MultiICPResult:
        """Run ICP from n_starts random rotations and return the best result.

        Args:
            source: Source PointCloud (N, 3).
            target: Target PointCloud (M, 3).

        Returns:
            MultiICPResult containing the best ICPResult and all trial results.
        """
        rng = np.random.default_rng(self.seed)
        rotations = sample_uniform_rotations(self.n_starts, rng=rng)

        all_results: list[ICPResult] = []
        all_rotations: list[np.ndarray] = []

        max_workers = self.n_jobs if self.n_jobs > 0 else None
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._run_single, self.icp, source, target, R): R
                for R in rotations
            }
            for future in as_completed(futures):
                result, R_init = future.result()
                all_results.append(result)
                all_rotations.append(R_init)

        best_idx = int(np.argmin([r.mean_residuals[-1] for r in all_results]))
        return MultiICPResult(
            best=all_results[best_idx],
            best_initial_rotation=all_rotations[best_idx],
            all_results=all_results,
            all_initial_rotations=all_rotations,
        )

    @staticmethod
    def _run_single(
            icp: ICP,
            source: PointCloud,
            target: PointCloud,
            R_init: np.ndarray,
    ) -> tuple[ICPResult, np.ndarray]:
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
        return result, R_init
