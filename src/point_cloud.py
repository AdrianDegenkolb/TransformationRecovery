import numpy as np
from numpy.typing import NDArray


class PointCloud:
    """
    A 3D point cloud represented as an (N, 3) array of points.
    """
    def __init__(self, points: NDArray[np.float64]):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {self.points.shape}")

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"PointCloud(n={len(self)})"
