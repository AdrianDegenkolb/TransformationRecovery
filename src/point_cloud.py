import numpy as np


class PointCloud:
    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {self.points.shape}")

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"PointCloud(n={len(self)})"
