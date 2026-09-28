from __future__ import annotations

import copy
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

    S is transformed twice to produce the noiseless ground-truth clouds
    P = T1(S) and Q = T2(S). The ground-truth transformation mapping P to Q
    is T_gt = T2 ∘ T1⁻¹. Dropout and measurement noise are simulated
    separately, on demand, by a PointCloudObserver — P and Q themselves stay
    exact so they can serve as a noise-free reference for evaluating recovered
    transformations.

    Attributes:
        S:    Source point cloud (N, 3).
        T1:   First random rigid transformation.
        T2:   Second random rigid transformation.
        P:    T1(S) — first ground-truth cloud.
        Q:    T2(S) — second ground-truth cloud.
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
        t_scale: float = 8.0,
        seed: int | None = 42,
        style: CloudStyle = "random",
        jitter_std: float = 0.0,
        normalize_spacing: bool = True,
    ) -> SyntheticExperiment:
        """Generate a synthetic point cloud experiment with two rigid transformations.

        Args:
            n:          Target number of points in the source cloud.
                        For 'lattice' the actual count may differ slightly due to integer grid dims.
            t_scale:    Scale of the random translation component.
            seed:       Random seed for reproducibility. Pass None for a random run.
            style:      Shape of the source cloud — 'random', 'clustered', 'lattice', '2d-lattice',
                        or 'muscle-fiber'.
            jitter_std: Std of per-node Gaussian jitter for 'lattice'/'2d-lattice'/'muscle-fiber'
                        styles. Ignored otherwise.
            normalize_spacing: If True, scale the source cloud to a median point spacing of 1
                        (see PointCloud.normalize), so that noise, sigma and t_scale are in
                        units of point spacing and mean the same thing for every style.

        Returns:
            SyntheticExperiment with S, T1, T2, P, Q, and T_gt = T2 ∘ T1⁻¹.
            P and Q are exact (noiseless); use a PointCloudObserver to simulate
            dropout and measurement noise.
        """
        if seed is not None:
            np.random.seed(seed)

        S = PointCloud(_make_cloud(n, style, jitter_std))
        if normalize_spacing:
            S = S.normalize()
        T1 = RigidTransformation.random(t_scale=t_scale)
        T2 = RigidTransformation.random(t_scale=t_scale)
        P = T1.apply(S)
        Q = T2.apply(S)
        T_gt = T2.compose(T1.inverse())

        return SyntheticExperiment(S=S, T1=T1, T2=T2, P=P, Q=Q, T_gt=T_gt)


class PointCloudObserver:
    """Turns an experiment's exact point cloud into a realistic observation of it.

    Applies per-point dropout (points the sensor missed entirely) followed by
    per-point Gaussian noise (measurement error on the points it did see).
    Owns its own random generator, so observations are reproducible from `seed`
    without touching global numpy random state.

    Each `observe` call consumes fresh randomness, so observing two clouds in
    sequence (e.g. an experiment's P and Q) yields independent dropout masks and
    noise — as two separate physical measurements would.
    """

    def __init__(self, seed: int = 42, noise_std: float = 0.0, dropout_prob: float = 0.0) -> None:
        """
        Args:
            seed:         Seed for this observer's random generator.
            noise_std:    Std of the per-point Gaussian noise added to observed points.
            dropout_prob: Probability of missing any individual point.
        """
        self.seed = seed
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.rng = np.random.default_rng(seed)

    def observe(self, point_cloud: PointCloud) -> PointCloud:
        """Produce a noisy, incomplete observation of `point_cloud`.

        Args:
            point_cloud: The exact PointCloud (N, 3) to observe.

        Returns:
            Observed PointCloud (M, 3) with M <= N, dropped points removed and
            the survivors perturbed by Gaussian noise. The input is not modified.
        """
        points = point_cloud.points
        if self.dropout_prob > 0:
            points = points[self.rng.random(len(points)) >= self.dropout_prob]
        if self.noise_std > 0:
            points = points + self.rng.normal(0.0, self.noise_std, size=points.shape)
        return PointCloud(points)

    def reset(self) -> PointCloudObserver:
        """Return an observer with the same settings and its randomness rewound.

        The complement of ``spawn``. Where spawn yields an *independent* stream, this
        yields the *same* stream from its beginning, so two callers see identical
        dropout masks and noise.

        Required whenever several methods are to be compared on the same data. An
        observer's generator advances with every ``spawn``, so passing one instance to
        several runs in sequence silently gives each run a different realisation of the
        degradation — turning a paired comparison into an unpaired one and letting the
        order methods are evaluated in change which of them looks better.

        Returns:
            A copy of this observer positioned at the start of its own seed's stream.
        """
        child = copy.copy(self)
        child.rng = np.random.default_rng(self.seed)
        return child

    def spawn(self) -> PointCloudObserver:
        """Return an observer with the same settings but an independent RNG stream.

        Required whenever observers cross a process boundary: pickling one
        observer into several worker processes copies its generator state, so
        every worker would otherwise replay the identical dropout mask and noise
        draws. Spawning per experiment also keeps results independent of whether
        the caller ran sequentially or in parallel.

        Returns:
            A copy of this observer whose randomness is statistically
            independent of this one's and of every other spawned child.
        """
        child = copy.copy(self)
        child.rng = self.rng.spawn(1)[0]
        return child


class PerfectObserver(PointCloudObserver):
    """An idealized observer: sees every point, exactly where it is.

    Equivalent to a PointCloudObserver with no dropout and no noise, so
    `observe` returns the cloud unchanged. Serves as the default for callers
    that want the clean case without constructing an observer themselves.
    """

    def __init__(self) -> None:
        super().__init__(seed=42, noise_std=0.0, dropout_prob=0.0)


def make_correspondence_pair(
    p: PointCloud,
    q: PointCloud,
    dropout_prob: float = 0.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[PointCloud, PointCloud, NDArray[np.int64]]:
    """Build a partial, shuffled correspondence pair from an index-aligned cloud pair.

    Independently drops points from ``p`` and ``q`` (simulating sensor dropout on
    each side), perturbs the survivors with independent Gaussian noise, then randomly
    permutes the surviving target points so that correspondence cannot be read off
    from point order. Yields realistic partial-overlap, order-agnostic correspondence
    problems with known ground-truth matches.

    Applies the same two degradations as ``PointCloudObserver`` but additionally
    tracks which points survived, which is what makes it usable for the
    feature-extractor metrics in ``error_metrics``: those need to know which row of
    one cloud corresponds to which row of the other, and dropout destroys the
    index alignment a ``PointCloudObserver`` output would otherwise be read with.

    Args:
        p:            Source cloud, index-aligned with q (p[i] <-> q[i]), e.g.
                      a SyntheticExperiment's P.
        q:            Target cloud, index-aligned with p, e.g. a
                      SyntheticExperiment's Q.
        dropout_prob: Per-point probability of dropping a point, applied
                      independently to each side.
        noise_std:    Std of the Gaussian noise added to the surviving points,
                      drawn independently for each side.
        rng:          Optional random generator for reproducibility.

    Returns:
        Tuple (p_obs, q_obs, correspondence):
            p_obs:          Observed source cloud after dropout and noise.
            q_obs:          Observed target cloud after dropout, noise and permutation.
            correspondence: (len(q_obs),) int64 array; correspondence[i] is the
                             index into p_obs.points of the true match for
                             q_obs.points[i], or -1 if that target point's
                             source correspondent was dropped. Note the direction:
                             target -> source. ``invert_correspondence`` flips it for
                             consumers that index by source.
    """
    rng = rng or np.random.default_rng()
    n = len(p)
    if len(q) != n:
        raise ValueError(f"p and q must be index-aligned (same length), got {len(p)} and {len(q)}")

    keep_p = rng.random(n) >= dropout_prob
    keep_q = rng.random(n) >= dropout_prob

    p_points = p.points[keep_p]
    p_obs_index = np.full(n, -1, dtype=np.int64)   # original index -> position in p_obs
    p_obs_index[keep_p] = np.arange(keep_p.sum())

    q_kept_original_idx = np.flatnonzero(keep_q)
    q_perm_original_idx = rng.permutation(q_kept_original_idx)
    q_points = q.points[q_perm_original_idx]

    if noise_std > 0:
        p_points = p_points + rng.normal(0.0, noise_std, size=p_points.shape)
        q_points = q_points + rng.normal(0.0, noise_std, size=q_points.shape)

    correspondence = p_obs_index[q_perm_original_idx]

    return PointCloud(p_points), PointCloud(q_points), correspondence


def invert_correspondence(
    correspondence: NDArray[np.int64],
    n_source: int,
) -> NDArray[np.int64]:
    """Flip a target -> source correspondence into a source -> target one.

    ``make_correspondence_pair`` reports, for each target point, which source point
    it came from. The feature-extractor metrics in ``error_metrics`` index the other
    way round, by source point. Converting between the two is easy to get subtly
    wrong, so it lives here rather than being re-derived at each call site.

    Args:
        correspondence: (M,) index into the source cloud per target point, or -1
                        where that target point has no counterpart.
        n_source:       Number of source points, i.e. the length of the result.

    Returns:
        (n_source,) int64 array giving the target index per source point, or -1
        where the source point has no counterpart.
    """
    inverted = np.full(n_source, -1, dtype=np.int64)
    matched = correspondence >= 0
    inverted[correspondence[matched]] = np.flatnonzero(matched)
    return inverted


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
