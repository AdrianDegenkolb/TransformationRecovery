from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from point_cloud import PointCloud


class Transformation(ABC):
    @abstractmethod
    def apply(self, p: PointCloud) -> PointCloud: ...

    @classmethod
    @abstractmethod
    def fit(cls, p1: PointCloud, p2: PointCloud, **kwargs) -> Transformation: ...


class RigidTransformation(Transformation):
    """f(p) = R @ p + t, optionally with additive Gaussian noise."""

    def __init__(self, R: np.ndarray, t: np.ndarray, noise_std: float = 0.0):
        self.R = np.asarray(R, dtype=np.float64)          # (3, 3) in SO(3)
        self.t = np.asarray(t, dtype=np.float64)          # (3,)
        self.noise_std = noise_std

    def apply(self, p: PointCloud) -> PointCloud:
        pts = p.points @ self.R.T + self.t
        if self.noise_std > 0:
            pts = pts + np.random.randn(*pts.shape) * self.noise_std
        return PointCloud(pts)

    @classmethod
    def fit(cls, p1: PointCloud, p2: PointCloud, **_) -> RigidTransformation:
        """Least-squares rigid alignment via SVD (Procrustes)."""
        c1 = p1.points.mean(axis=0)
        c2 = p2.points.mean(axis=0)
        H = (p1.points - c1).T @ (p2.points - c2)
        U, _, Vt = np.linalg.svd(H)
        # Ensure proper rotation (det = +1)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        t = c2 - R @ c1
        return cls(R, t)

    @classmethod
    def random(cls, noise_std: float = 0.0, t_scale: float = 5.0) -> RigidTransformation:
        """Random rotation (via QR) and random translation."""
        Q, _ = np.linalg.qr(np.random.randn(3, 3))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        t = np.random.randn(3) * t_scale
        return cls(Q, t, noise_std)

    def compose_with_inverse(self, other: RigidTransformation) -> RigidTransformation:
        """Return self ∘ other⁻¹, i.e. f(p) = R2 @ (R1ᵀ @ (p − t1)) + t2.

        Useful for computing ground-truth: if P = T1(S) and Q = T2(S), the
        ideal map from P to Q is T2 ∘ T1⁻¹.
        """
        R_inv = other.R.T
        t_inv = -(R_inv @ other.t)
        R_out = self.R @ R_inv
        t_out = self.R @ t_inv + self.t
        return RigidTransformation(R_out, t_out)

    def __repr__(self) -> str:
        angle = np.degrees(np.arccos(np.clip((np.trace(self.R) - 1) / 2, -1, 1)))
        return f"RigidTransformation(rot={angle:.1f}°, t_norm={np.linalg.norm(self.t):.2f})"
