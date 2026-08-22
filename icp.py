from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from point_cloud import PointCloud
from transformation import RigidTransformation
from matcher import Matcher, NearestNeighborMatcher


@dataclass
class ICPResult:
    """Result of an ICP run.

    Attributes:
        transformation:      Accumulated rigid transformation mapping source onto target.
        n_iterations:        Number of EM iterations performed.
        converged:           Whether the algorithm converged before max_iter.
        mean_residuals:      Mean point-to-point residual after each M-step.
        cloud_history:       Source cloud state at the start of each iteration (before E-step).
        assignment_history:  NN index array (N,) from the E-step of each iteration.
        transform_history:   Accumulated transformation after the M-step of each iteration.
    """

    transformation: RigidTransformation
    n_iterations: int
    converged: bool
    mean_residuals: list[float] = field(default_factory=list)
    cloud_history: list[PointCloud] = field(default_factory=list)
    assignment_history: list[np.ndarray] = field(default_factory=list)
    transform_history: list[RigidTransformation] = field(default_factory=list)


class ICP:
    """EM algorithm for rigid point cloud registration without known correspondences.

    E-step: establish point correspondences via a Matcher (default: nearest neighbor).
    M-step: fit a RigidTransformation via SVD Procrustes on matched pairs.

    Convergence is declared when the step transformation is close to identity:
        ||R_step - I||_F + ||t_step|| < tol
    """

    def __init__(
        self,
        matcher: Matcher | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
    ):
        """
        Args:
            matcher:  Correspondence algorithm for the E-step.
                      Defaults to NearestNeighborMatcher.
            max_iter: Maximum number of EM iterations.
            tol:      Convergence threshold on ||R_step - I||_F + ||t_step||.
            verbose:  Print residual at each iteration if True.
        """
        self.matcher = matcher or NearestNeighborMatcher()
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, source: PointCloud, target: PointCloud) -> ICPResult:
        """Run ICP to find the rigid transformation mapping source onto target.

        Args:
            source: Source PointCloud (N, 3).
            target: Target PointCloud (M, 3).

        Returns:
            ICPResult with the accumulated transformation, convergence info, and
            per-iteration history (clouds, assignments, transformations).
        """
        current = source
        accumulated = _identity()

        mean_residuals: list[float] = []
        cloud_history: list[PointCloud] = []
        assignment_history: list[np.ndarray] = []
        transform_history: list[RigidTransformation] = []

        for i in range(self.max_iter):
            # Snapshot cloud state before this iteration's E-step
            cloud_history.append(current)

            # E-step: build correspondences
            _, target_matched, indices = self.matcher.match(current, target)
            assignment_history.append(indices)

            # M-step: fit rigid transformation from current cloud to matched target
            T_step = RigidTransformation.fit(current, target_matched)

            # Track mean point residual after applying this step
            residual = float(
                np.linalg.norm(
                    T_step.apply(current).points - target_matched.points, axis=1
                ).mean()
            )
            mean_residuals.append(residual)

            if self.verbose:
                print(f"Iter {i + 1:3d} | mean residual: {residual:.4f}")

            # Accumulate transformation and advance the current cloud
            accumulated = T_step.compose(accumulated)
            current = T_step.apply(current)
            transform_history.append(accumulated)

            # Convergence: step transformation is nearly identity
            delta = np.linalg.norm(T_step.R - np.eye(3), ord="fro") + np.linalg.norm(T_step.t)
            if delta < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i + 1} (delta={delta:.2e})")
                return ICPResult(
                    accumulated, i + 1, True,
                    mean_residuals, cloud_history, assignment_history, transform_history,
                )

        if self.verbose:
            print(f"Did not converge after {self.max_iter} iterations.")
        return ICPResult(
            accumulated, self.max_iter, False,
            mean_residuals, cloud_history, assignment_history, transform_history,
        )


def _identity() -> RigidTransformation:
    """Return the identity rigid transformation."""
    return RigidTransformation(np.eye(3), np.zeros(3))
