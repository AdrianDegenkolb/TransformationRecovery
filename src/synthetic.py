from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate

from point_cloud import PointCloud
from transformation import RigidTransformation

CloudStyle = Literal["random", "clustered", "lattice", "2d-lattice", "muscle-fiber"]


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
        jitter_std: float = 0.0,
    ) -> SyntheticExperiment:
        """Generate a synthetic point cloud experiment with two rigid transformations.

        Args:
            n:          Target number of points in the source cloud.
                        For 'lattice' the actual count may differ slightly due to integer grid dims.
            noise_std:  Per-point Gaussian noise added when applying each transformation.
            t_scale:    Scale of the random translation component.
            seed:       Random seed for reproducibility. Pass None for a random run.
            style:      Shape of the source cloud — 'random', 'clustered', 'lattice', '2d-lattice',
                        or 'muscle-fiber'.
            jitter_std: Std of per-node Gaussian jitter for 'lattice'/'2d-lattice'/'muscle-fiber'
                        styles. Ignored otherwise.

        Returns:
            SyntheticExperiment with S, T1, T2, P, Q, and T_gt = T2 ∘ T1⁻¹.
        """
        if seed is not None:
            np.random.seed(seed)

        S = PointCloud(_make_cloud(n, style, jitter_std))
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


def _make_cloud(n: int, style: CloudStyle, jitter_std: float = 0.0) -> NDArray[np.float64]:
    """Dispatch to the appropriate cloud generator.

    Args:
        n:          Target number of points.
        style:      One of 'random', 'clustered', 'lattice', '2d-lattice', 'muscle-fiber'.
        jitter_std: Std of per-node Gaussian jitter, used by 'lattice'/'2d-lattice'/'muscle-fiber' only.

    Returns:
        (N, 3) float64 array of point positions.
    """
    if style == "random":
        return _random_cloud(n)
    if style == "clustered":
        return _clustered_cloud(n)
    if style == "lattice":
        return _lattice_cloud(n, jitter_std=jitter_std)
    if style == "2d-lattice":
        return _2d_lattice_cloud(n, jitter_std=jitter_std)
    if style == "muscle-fiber":
        return _muscle_fiber_cloud(n, jitter_std=jitter_std)
    raise ValueError(
        f"Unknown cloud style {style!r}. Choose from 'random', 'clustered', 'lattice', "
        "'2d-lattice', 'muscle-fiber'."
    )


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


def _2d_lattice_cloud(
    n: int,
    spacing: tuple[float, float] = (2.0, 3.5),
    jitter_std: float = 0.3,
) -> NDArray[np.float64]:
    """2-D lattice in the XY plane with anisotropic spacing and per-node jitter.

    All points lie approximately in z=0, giving every point a high-planarity
    geometric neighbourhood. This makes the 2D lattice a good test case for the
    trimmer: the whole cloud is one large geometrically common region.

    The actual count (nx * ny) may differ slightly from n because grid dimensions
    are rounded to integers.

    Args:
        n:          Target number of points.
        spacing:    Grid spacing along (x, y). Distinct values break x/y symmetry.
        jitter_std: Std of per-node Gaussian jitter in all 3 axes.

    Returns:
        (nx*ny, 3) array centred at the origin.
    """
    ny = max(1, round(n ** 0.5))
    nx = max(1, round(n / ny))

    xs = np.arange(nx) * spacing[0]
    ys = np.arange(ny) * spacing[1]

    grid_xy = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    z = np.zeros((len(grid_xy), 1))
    grid = np.hstack([grid_xy, z])
    grid -= grid.mean(axis=0)
    grid += np.random.normal(0, jitter_std, size=grid.shape)
    return grid


def _muscle_fiber_cloud(
    n: int,
    fiber_radius: float = 3.0,
    fiber_spacing: float = 8.0,
    nucleus_spacing: float = 4.0,
    nuclei_per_fiber: int = 20,
    jitter_std: float = 0.3,
) -> NDArray[np.float64]:
    """Nuclei-like points on the periphery of parallel, hexagonally packed fibers.

    Models skeletal muscle: fibers are parallel cylinders (long axis = z) packed
    in a hexagonal cross-section lattice (xy), mirroring how fibers pack into a
    fascicle. Nuclei sit on each fiber's surface at randomised angles around the
    circumference and are spaced roughly evenly along its length — matching the
    peripheral, sarcolemma-adjacent nuclear positioning seen in real muscle fibers.

    The actual number of returned points (n_fibers * nuclei_per_fiber) may differ
    slightly from n because the fiber count is rounded to an integer grid.

    Args:
        n:                Target number of points (nuclei).
        fiber_radius:      Radius of each fiber; nuclei are placed at this distance
                            from the fiber's central (z) axis.
        fiber_spacing:     Centre-to-centre distance between neighbouring fibers.
        nucleus_spacing:   Average distance between consecutive nuclei along a fiber.
        nuclei_per_fiber:  Number of nuclei placed per fiber.
        jitter_std:        Std of per-nucleus Gaussian jitter in all 3 axes.

    Returns:
        (n_fibers * nuclei_per_fiber, 3) array centred at the origin.
    """
    n_fibers = max(1, round(n / nuclei_per_fiber))
    centers = _hexagonal_centers(n_fibers, fiber_spacing)

    angles = np.random.uniform(0, 2 * np.pi, size=(len(centers), nuclei_per_fiber))
    z = np.arange(nuclei_per_fiber) * nucleus_spacing + np.random.normal(
        0, nucleus_spacing * 0.15, size=(len(centers), nuclei_per_fiber)
    )
    x = centers[:, 0:1] + fiber_radius * np.cos(angles)
    y = centers[:, 1:2] + fiber_radius * np.sin(angles)

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    points += np.random.normal(0, jitter_std, size=points.shape)
    points -= points.mean(axis=0)
    return points


def _hexagonal_centers(n: int, spacing: float) -> NDArray[np.float64]:
    """Roughly square grid of 2-D centres arranged in hexagonal (honeycomb) packing.

    Args:
        n:       Target number of centres.
        spacing: Centre-to-centre distance between neighbours.

    Returns:
        (N, 2) array of (x, y) centre positions; N may differ slightly from n
        because the row/column counts are rounded to integers.
    """
    n_cols = max(1, round(n ** 0.5))
    n_rows = max(1, round(n / n_cols))
    row_height = spacing * (3 ** 0.5 / 2)

    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    x = cols * spacing + (spacing / 2) * (rows % 2)
    y = rows * row_height
    return np.stack([x, y], axis=-1).reshape(-1, 2).astype(np.float64)
