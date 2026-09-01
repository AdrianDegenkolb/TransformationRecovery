"""Point-cloud residual and error-metric computations.

A residual is the distance between paired points, however the pairing was
established: reusing an already-computed Matching, freshly matching via
nearest-neighbor, or assuming known ground-truth correspondence. This module
is the one place that math lives, rather than being reimplemented separately
in icp.py, experiment_runner.py, and utils.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate

from algebra_utils import rotation_angle
from matcher import Matching, NearestNeighborMatcher
from point_cloud import PointCloud
from synthetic import SyntheticExperiment
from transformation import RigidTransformation


def get_residuals(matching: Matching, transformed_source: PointCloud) -> NDArray[np.float64]:
    """Per-point distance from transformed_source to matching's target positions.

    Takes an already-computed Matching so callers with a live correspondence
    (e.g. ICP's E-step) don't have to recompute it just to measure a residual.

    Args:
        matching:           A correspondence, e.g. from Matcher.match(...).
        transformed_source: Points to measure, index-aligned with matching's
                             source points (typically a transformed version of them).

    Returns:
        (N,) array of per-point distances.
    """
    return np.linalg.norm(transformed_source.points - matching.target_positions, axis=1)


def nearest_neighbor_residuals(p1: PointCloud, p2: PointCloud) -> NDArray[np.float64]:
    """Per-point distance from each p1 point to its nearest neighbor in p2."""
    matching = NearestNeighborMatcher().match(p1, p2)
    return get_residuals(matching, p1)


def true_residuals(p1: PointCloud, p2: PointCloud) -> NDArray[np.float64]:
    """Per-point distance assuming p1[i]/p2[i] are already the true corresponding pair.

    No matching step — e.g. p1 = transformation.apply(experiment.P), p2 = experiment.Q,
    relying on the known index-aligned correspondence synthetic experiments provide.
    """
    return np.linalg.norm(p1.points - p2.points, axis=1)


@dataclass
class ErrorMetrics:
    """Error metrics for one fitted transformation.

    Attributes:
        rotation_error:          Angle (deg) between the fitted and ground-truth rotation.
        translation_error:       Distance between the fitted and ground-truth translation.
        closest_point_residuals: Per-point NN-matched distance from transformation.apply(experiment.P)
                                  to experiment.Q. What you'd have on real data without ground truth.
        true_residuals:          Per-point ground-truth-correspondence distance from
                                  transformation.apply(experiment.P) to experiment.Q, using the
                                  known index-aligned correspondence instead of NN-matching.
    """
    rotation_error: float
    translation_error: float
    closest_point_residuals: NDArray[np.float64]
    true_residuals: NDArray[np.float64]

    def __repr__(self) -> str:
        rows = [
            ["Rotation error",             f"{self.rotation_error:.4f}°"],
            ["Translation error",          f"{self.translation_error:.4f}"],
            ["Mean closest-point residual", f"{self.closest_point_residuals.mean():.4f}"],
            ["Max closest-point residual",  f"{self.closest_point_residuals.max():.4f}"],
            ["Mean true residual",          f"{self.true_residuals.mean():.4f}"],
        ]
        return tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_outline")


def get_error_metrics(
        transformation: RigidTransformation,
        experiment: SyntheticExperiment,
) -> ErrorMetrics:
    """Computes rotation/translation error vs ground truth, plus two residual metrics.

    The error vs ground truth is what we want to have minimized. closest_point_residuals
    is the quantity ICP actually tries to minimize when matching hard — it may be small
    while rotation/translation error is large for experiments with lots of local optima.
    true_residuals uses the experiment's known ground-truth correspondence instead of
    nearest-neighbor matching. Both are computed from the full experiment.P/experiment.Q.

    Args:
        transformation: a fitted transformation
        experiment:     the SyntheticExperiment providing ground truth (T_gt) and the
                         full, index-aligned clouds (P, Q)
    Return:
        ErrorMetrics
    """
    rot_err = rotation_angle(experiment.T_gt.R, transformation.R)
    t_err = float(np.linalg.norm(experiment.T_gt.t - transformation.t))

    closest_point_residuals = nearest_neighbor_residuals(transformation.apply(experiment.P), experiment.Q)
    true_res = true_residuals(transformation.apply(experiment.P), experiment.Q)

    return ErrorMetrics(rot_err, t_err, closest_point_residuals, true_res)
