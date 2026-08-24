from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from tqdm import tqdm

from point_cloud import PointCloud
from algebra_utils import six_d_to_rotation, rotation_to_six_d
from transformation import Transformation, RigidTransformation


class ElasticTransformation(Transformation):
    """f(p) = R @ p + t + u(p), fitted by minimizing the objective from
    Transformation Recovery: data term + magnitude penalty on u + graph-Laplacian
    smoothness on u.
    """

    def __init__(
        self,
        R: np.ndarray,
        t: np.ndarray,
        u: np.ndarray,
        source_points: np.ndarray,
    ):
        self.R = R                        # (3, 3)
        self.t = t                        # (3,)
        self.u = u                        # (N, 3) per-source-point residuals
        self.source_points = source_points  # (N, 3) reference positions

    def apply(self, p: PointCloud) -> PointCloud:
        pts = p.points @ self.R.T + self.t + self.u
        return PointCloud(pts)

    @classmethod
    def fit(
        cls,
        p1: PointCloud,
        p2: PointCloud,
        lambda1: float = 0.1,
        lambda2: float = 0.01,
        n_iter: int = 2000,
        lr: float = 1e-2,
        k_neighbors: int = 10,
        sigma: float | None = None,
        verbose: bool = True,
    ) -> ElasticTransformation:
        """Fit f: p1 → p2 by gradient descent (Adam).

        Args:
            p1:          Source PointCloud (N, 3).
            p2:          Target PointCloud (N, 3).
            lambda1:     Weight of the per-point magnitude penalty on u.
            lambda2:     Weight of the graph-Laplacian smoothness penalty on u.
            n_iter:      Number of Adam optimization steps.
            lr:          Adam learning rate.
            k_neighbors: Neighbourhood size for the Laplacian graph.
            sigma:       Gaussian kernel bandwidth for edge weights;
                         defaults to the median nearest-neighbor distance.
            verbose:     Show tqdm progress bar with live loss values if True.

        Returns:
            Fitted ElasticTransformation mapping p1 onto p2.
        """
        pts1 = torch.tensor(p1.points, dtype=torch.float64)
        pts2 = torch.tensor(p2.points, dtype=torch.float64)
        N = len(pts1)

        # --- Neighbor graph for the Laplacian regulariser ---
        tree = KDTree(p1.points)
        dist, idx = tree.query(p1.points, k=k_neighbors + 1)
        dist, idx = dist[:, 1:], idx[:, 1:]   # drop self

        if sigma is None:
            sigma = float(np.median(dist[:, 0]))

        weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))  # (N, k)
        w_t = torch.tensor(weights, dtype=torch.float64)
        idx_t = torch.tensor(idx, dtype=torch.long)

        # --- Initialize from rigid alignment ---
        rigid = RigidTransformation.fit(p1, p2)
        six_d = nn.Parameter(torch.tensor(rotation_to_six_d(rigid.R), dtype=torch.float64))
        t_param = nn.Parameter(torch.tensor(rigid.t, dtype=torch.float64))
        u_param = nn.Parameter(torch.zeros(N, 3, dtype=torch.float64))

        optimizer = torch.optim.Adam([six_d, t_param, u_param], lr=lr)

        pbar = tqdm(range(n_iter), desc="ElasticFit", disable=not verbose)
        for it in pbar:
            optimizer.zero_grad()

            R = six_d_to_rotation(six_d)                   # (3, 3)
            pred = pts1 @ R.T + t_param + u_param           # (N, 3)

            data_loss = ((pred - pts2) ** 2).sum(dim=-1).mean()
            magnitude_loss = (u_param ** 2).sum(dim=-1).mean()

            # Laplacian: sum_{i,j in nbrs} w_ij * ||u_i - u_j||^2
            u_nbrs = u_param[idx_t]                         # (N, k, 3)
            diff = u_param.unsqueeze(1) - u_nbrs            # (N, k, 3)
            reg_loss = (w_t.unsqueeze(-1) * diff ** 2).sum()

            loss = data_loss + lambda1 * magnitude_loss + lambda2 * reg_loss
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                loss=f"{loss.item():.5f}",
                data=f"{data_loss.item():.5f}",
                u_mag=f"{magnitude_loss.item():.5f}",
                reg=f"{reg_loss.item():.5f}",
            )

        R_np = six_d_to_rotation(six_d).detach().numpy()
        t_np = t_param.detach().numpy()
        u_np = u_param.detach().numpy()
        return cls(R_np, t_np, u_np, p1.points.copy())
