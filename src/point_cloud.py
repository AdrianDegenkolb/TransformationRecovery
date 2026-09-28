import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree


class PointCloud:
    """
    A 3D point cloud represented as an (N, 3) array of points.
    """
    def __init__(self, points: NDArray[np.float64]):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {self.points.shape}")

    @property
    def median_spacing(self) -> float:
        """Median distance from a point to its nearest neighbour.

        The natural unit for anything that has to judge whether a distance is "large"
        for this cloud — a residual, a misalignment, a noise level. An absolute
        threshold means something different on every cloud; a multiple of the spacing
        means the same thing on all of them.

        Returns:
            The median nearest-neighbour distance, or 0.0 for a cloud of one point.
        """
        if len(self.points) < 2:
            return 0.0
        distances, _ = KDTree(self.points).query(self.points, k=2)
        return float(np.median(distances[:, 1]))

    def normalize(self) -> "PointCloud":
        """Return a copy scaled about its centroid so that its median spacing is 1.

        Makes clouds of different styles share one length unit, so a noise level, a
        matching bandwidth or a residual threshold means the same thing on each of them.

        Returns:
            New PointCloud (N, 3) with median_spacing == 1 and the same centroid.

        Raises:
            ValueError: If the median spacing is 0 (fewer than two distinct points), so
                        there is no scale to normalise by.
        """
        spacing = self.median_spacing
        if spacing == 0.0:
            raise ValueError("Cannot normalize a cloud whose median spacing is 0.")
        centroid = self.points.mean(axis=0)
        return PointCloud(centroid + (self.points - centroid) / spacing)

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"PointCloud(n={len(self)})"
