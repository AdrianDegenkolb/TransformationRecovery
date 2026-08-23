from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm
from tabulate import tabulate

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
