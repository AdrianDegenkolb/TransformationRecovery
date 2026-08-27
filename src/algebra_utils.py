"""Rotation utilities shared across the project.

Provides:
- Uniform SO(3) sampling via random unit quaternions.
- Conversion between rotation matrices and the 6D representation
  (Zhou et al., 2019) used for gradient-based optimization.
"""
from __future__ import annotations

import numpy as np
import torch


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a rotation matrix.

    Args:
        q: Unit quaternion (4,) in [w, x, y, z] order.

    Returns:
        Rotation matrix (3, 3) in SO(3).
    """
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def sample_uniform_rotation(rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample a rotation matrix uniformly from SO(3) via a random unit quaternion.

    Uniformly distributed quaternions on the unit 3-sphere correspond to a
    uniform (Haar) measure on SO(3).

    Args:
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Rotation matrix (3, 3) in SO(3).
    """
    rng = rng or np.random.default_rng()
    q = rng.standard_normal(4)
    return quaternion_to_rotation(q)


def sample_uniform_rotations(
    n: int,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """
    Sample n rotation matrices uniformly from SO(3).

    Args:
        n:   Number of rotations to sample.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        List of n rotation matrices, each of shape (3, 3).
    """
    rng = rng or np.random.default_rng()
    return [sample_uniform_rotation(rng) for _ in range(n)]


def six_d_to_rotation(six_d: torch.Tensor) -> torch.Tensor:
    """6D representation → SO(3) via Gram-Schmidt (Zhou et al., 2019).

    Args:
        six_d: Tensor of shape (6,) = [a1 | a2], where a1, a2 are the first
               two columns of the target rotation matrix.

    Returns:
        Rotation matrix (3, 3) with orthonormal columns.
    """
    a1, a2 = six_d[:3], six_d[3:]
    b1 = a1 / a1.norm()
    b2 = a2 - (b1 @ a2) * b1
    b2 = b2 / b2.norm()
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack([b1, b2, b3], dim=1)  # columns → (3, 3)


def rotation_to_six_d(R: np.ndarray) -> np.ndarray:
    """Extract the 6D seed from a rotation matrix (first two columns, row-major).

    Args:
        R: Rotation matrix (3, 3) in SO(3).

    Returns:
        6D representation (6,) as a float64 array.
    """
    return R[:, :2].T.reshape(-1).astype(np.float64)


def rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular distance between two rotation matrices in degrees.

    Args:
        R1: SO(3) matrix (3, 3).
        R2: SO(3) matrix (3, 3).

    Returns:
        Angle in degrees between R1 and R2.
    """
    cos = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))
