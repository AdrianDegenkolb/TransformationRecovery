from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate

from point_cloud import PointCloud
from transformation import RigidTransformation

CloudStyle = Literal["random", "clustered", "lattice"]


@dataclass
class SyntheticExperiment:
    """Two rigid transformations of a shared source cloud with ground truth.

    S is transformed twice (with per-point Gaussian noise) to produce
    P = T1(S) and Q = T2(S). The ground-truth transformation mapping P to Q
    is T_gt = T2 ∘ T1⁻¹.

    Attributes:
        S:    Source point cloud (N, 3).
        T1:   First random rigid transformation (with noise).
        T2:   Second random rigid transformation (with noise).
        P:    T1(S) — first observed cloud.
        Q:    T2(S) — second observed cloud.
        T_gt: Ground-truth transformation T2 ∘ T1⁻¹ mapping P to Q.
    """

    S: PointCloud
    T1: RigidTransformation
    T2: RigidTransformation
    P: PointCloud
    Q: PointCloud
    T_gt: RigidTransformation

    def __repr__(self):
        rows: list[tuple[str, object]] = [
            ("Original Point Cloud", self.S),
            ("Transformation 1", self.T1),
            ("Transformation 2", self.T2),
            ("Point Cloud 1", self.P),
            ("Point Cloud 2", self.Q),
            ("GT transformation", self.T_gt)
        ]
        return tabulate(rows, tablefmt="rounded_outline")

    @staticmethod
    def generate(
        n: int = 2000,
        noise_std: float = 3.0,
        t_scale: float = 8.0,
        seed: int | None = 42,
        style: CloudStyle = "random",
    ) -> SyntheticExperiment:
        """Generate a synthetic point cloud experiment with two rigid transformations.

        Args:
            n:         Target number of points in the source cloud.
                       For 'lattice' the actual count may differ slightly due to integer grid dims.
            noise_std: Per-point Gaussian noise added when applying each transformation.
            t_scale:   Scale of the random translation component.
            seed:      Random seed for reproducibility. Pass None for a random run.
            style:     Shape of the source cloud — 'random', 'clustered', or 'lattice'.

        Returns:
            SyntheticExperiment with S, T1, T2, P, Q, and T_gt = T2 ∘ T1⁻¹.
        """
        if seed is not None:
            np.random.seed(seed)

        S = PointCloud(_make_cloud(n, style))
        T1 = RigidTransformation.random(noise_std=noise_std, t_scale=t_scale)
        T2 = RigidTransformation.random(noise_std=noise_std, t_scale=t_scale)
        P = T1.apply(S)
        Q = T2.apply(S)
        T_gt = T2.compose(T1.inverse())

        return SyntheticExperiment(S=S, T1=T1, T2=T2, P=P, Q=Q, T_gt=T_gt)

    def observe_point_clouds(self, dropout_prob: float) -> tuple[PointCloud, PointCloud]:
        """
        Returns the point clouds P and Q but omits individual points with probability dropout probability.

        Args:
            dropout_prob: The probability to miss individual points in the observation
        Returns:
            tuple containing observed and incomplete point clouds P and Q
        """
        indices_for_P = np.random.choice([True, False], size=len(self.P), replace=True, p=[1 - dropout_prob, dropout_prob])
        indices_for_Q = np.random.choice([True, False], size=len(self.Q), replace=True, p=[1 - dropout_prob, dropout_prob])
        return PointCloud(self.P.points[indices_for_P]), PointCloud(self.Q.points[indices_for_Q])


def _make_cloud(n: int, style: CloudStyle) -> NDArray[np.float64]:
    """Dispatch to the appropriate cloud generator.

    Args:
        n:     Target number of points.
        style: One of 'random', 'clustered', 'lattice'.

    Returns:
        (N, 3) float64 array of point positions.
    """
    if style == "random":
        return _random_cloud(n)
    if style == "clustered":
        return _clustered_cloud(n)
    if style == "lattice":
        return _lattice_cloud(n)
    raise ValueError(f"Unknown cloud style {style!r}. Choose from 'random', 'clustered', 'lattice'.")


def _random_cloud(n: int) -> NDArray[np.float64]:
    """Uniform random points in [-30, 30]^3.

    Args:
        n: Number of points.

    Returns:
        (n, 3) array.
    """
    return np.random.uniform(-30, 30, size=(n, 3))


def _clustered_cloud(n: int, n_clusters: int = 8, cluster_std: float = 4.0) -> NDArray[np.float64]:
    """Dense Gaussian clusters with centres spread across [-20, 20]^3.

    Points are distributed evenly across clusters.  Because cluster centers
    are placed far apart relative to the intra-cluster spread, each cluster
    acts as a distinct geometric anchor that breaks rotational symmetry.

    Args:
        n:           Total number of points.
        n_clusters:  Number of clusters.
        cluster_std: Standard deviation of each cluster (isotropic).

    Returns:
        (n, 3) array.
    """
    centers = np.random.uniform(-20, 20, size=(n_clusters, 3))
    sizes = np.random.multinomial(n, pvals=np.ones(n_clusters) / n_clusters)
    return np.vstack([
        np.random.normal(centers[k], cluster_std, size=(sizes[k], 3))
        for k in range(n_clusters)
    ])


def _lattice_cloud(
    n: int,
    spacing: tuple[float, float, float] = (2.0, 3.5, 6.0),
    jitter_std: float = 0.3,
) -> NDArray[np.float64]:
    """3-D lattice with anisotropic spacing and slight per-node jitter.

    The different spacing per axis makes rotations distinguishable: an ICP
    match that rotates x into z will produce much larger residuals than the
    correct alignment.  Jitter adds realistic noise while preserving structure.

    The actual number of returned points (nx * ny * nz) may differ slightly
    from n because grid dimensions are rounded to integers.

    Args:
        n:          Target number of points.
        spacing:    Grid spacing along (x, y, z).  Use distinct values to
                    break rotational symmetry.
        jitter_std: Std of per-node Gaussian jitter (should be ≪ min spacing).

    Returns:
        (nx*ny*nz, 3) array centred at the origin.
    """
    nz = max(1, round(n ** (1 / 3)))
    ny = max(1, round((n / nz) ** 0.5))
    nx = max(1, round(n / (ny * nz)))

    xs = np.arange(nx) * spacing[0]
    ys = np.arange(ny) * spacing[1]
    zs = np.arange(nz) * spacing[2]

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    grid -= grid.mean(axis=0)
    grid += np.random.normal(0, jitter_std, size=grid.shape)
    return grid
