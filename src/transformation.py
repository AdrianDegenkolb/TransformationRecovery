from __future__ import annotations

from abc import ABC, abstractmethod, ABCMeta

import numpy as np
from numpy.typing import NDArray

from point_cloud import PointCloud


class Transformation(ABC, metaclass=ABCMeta):
    @abstractmethod
    def apply(self, p: PointCloud) -> PointCloud: ...

    @classmethod
    @abstractmethod
    def fit(cls, p1: PointCloud, p2: PointCloud, **kwargs) -> Transformation:
        """
        Fits the parameters of this transformation to minimize the residuals between f(p1) and p2
        """
        ...

    def residuals(self, p1: PointCloud, q2: PointCloud) -> NDArray[np.float64]:
        """
        Given two point clouds of identical size n returns an array of size n that contains the distance between
        f(p[i]) to q[i] in entry [i]
        """
        return np.linalg.norm(self.apply(p1).points - q2.points, axis=1)


class RigidTransformation(Transformation):
    """f(p) = R @ p + t, optionally with additive Gaussian noise."""

    def __init__(self, R: NDArray[np.float64], t: NDArray[np.float64], noise_std: float = 0.0):
        self.R = np.asarray(R, dtype=np.float64)          # (3, 3) in SO(3)
        self.t = np.asarray(t, dtype=np.float64)          # (3,)
        self.noise_std = noise_std

    def apply(self, p: PointCloud) -> PointCloud:
        pts = p.points @ self.R.T + self.t
        if self.noise_std > 0:
            pts = pts + np.random.randn(*pts.shape) * self.noise_std
        return PointCloud(pts)

    @classmethod
    def identity(cls) -> RigidTransformation:
        """Return the identity rigid transformation."""
        return RigidTransformation(np.eye(3), np.zeros(3))

    @classmethod
    def fit(
        cls,
        p1: PointCloud,
        p2: PointCloud,
        weights: NDArray[np.float64] | None = None,
    ) -> RigidTransformation:
        """Weighted least-squares rigid alignment via SVD (Procrustes).

        Minimises Σ_i w_i ||R p1_i + t − p2_i||².  When weights is None all
        points contribute equally, recovering the standard unweighted solution.

        Args:
            p1:      Source PointCloud (N, 3).
            p2:      Target PointCloud (N, 3).
            weights: Optional (N,) non-negative weight per point pair.
                     Need not be normalized.

        Returns:
            RigidTransformation minimising the weighted alignment error.
        """
        w = np.ones(len(p1)) / len(p1) if weights is None else weights / weights.sum()

        centroid_src = (w[:, None] * p1.points).sum(axis=0)
        centroid_tgt = (w[:, None] * p2.points).sum(axis=0)

        # Weighted cross-covariance of centred point clouds
        cross_cov = (p1.points - centroid_src).T @ (w[:, None] * (p2.points - centroid_tgt))

        U, _, Vt = np.linalg.svd(cross_cov)

        # Correct for reflections: force det(R) = +1
        det_sign = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1.0, 1.0, det_sign]) @ U.T
        t = centroid_tgt - R @ centroid_src
        return cls(R, t)

    @classmethod
    def random(cls, noise_std: float = 0.0, t_scale: float = 5.0) -> RigidTransformation:
        """Random rotation (via QR) and random translation."""
        Q, _ = np.linalg.qr(np.random.randn(3, 3))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        t = np.random.randn(3) * t_scale
        return cls(Q, t, noise_std)

    def inverse(self) -> RigidTransformation:
        """Return the inverse transformation: R⁻¹ = Rᵀ, t⁻¹ = −Rᵀ @ t."""
        R_inv = self.R.T
        t_inv = -(R_inv @ self.t)
        return RigidTransformation(R_inv, t_inv)

    def compose(self, other: RigidTransformation) -> RigidTransformation:
        """Return self ∘ other, i.e. f(p) = R_self @ (R_other @ p + t_other) + t_self."""
        R_out = self.R @ other.R
        t_out = self.R @ other.t + self.t
        return RigidTransformation(R_out, t_out)

    def __repr__(self) -> str:
        angle = np.degrees(np.arccos(np.clip((np.trace(self.R) - 1) / 2, -1, 1)))
        return f"RigidTransformation(rot={angle:.1f}°, t_norm={np.linalg.norm(self.t):.2f})"
