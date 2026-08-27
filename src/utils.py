import numpy as np
from numpy.typing import NDArray

from algebra_utils import rotation_angle
from icp import ICPResult
from matcher import NearestNeighborMatcher
from point_cloud import PointCloud
from transformation import RigidTransformation


def convergence_ratio(ICP_results: list[ICPResult]) -> float:
    """
    Returns the fraction of runs that have converged to a solution.
    """
    return sum(r.converged for r in ICP_results) / len(ICP_results)


def convergence_to_global_opt_ratio(ICP_results: list[ICPResult], tol: float = 1e-3) -> float:
    """
    Returns the fraction of runs that have converged to the globally optimal solution. This is measured by small residual errors.
    """
    return sum(r.mean_residuals[-1] < tol for r in ICP_results) / len(ICP_results)


def get_error_metrics(
        transformation: RigidTransformation,
        ground_truth_transformation: RigidTransformation,
        p: PointCloud,
        q: PointCloud
) -> tuple[float, float, NDArray[np.float64]]:
    """
    Computes three basic error metrics for a transformation:
    1. **Rotation error vs ground truth**: How much do transformation and the ground truth transformation
    differ in angle (deg)
    2. **Translation error vs ground truth**: How much do transformation and the ground truth transformation
    differ in translation direction and distance
    3. **Residuals**: The per point distances to the closest point in Q of f(p).

    The error vs ground truth is what we want to have minimized. The mean residuals is the quantity that ICP actually
    tries to minimize. It may achieve low residual errors while having large deviations from the ground truth for
    experiments with lots of local optima.

    Args:
        transformation: a transformation
        ground_truth_transformation: the ground truth transformation
        p: first PointCloud
        q: second PointCloud
    Return:
        rotation error, translation error, per point residuals
    """
    rot_err = rotation_angle(ground_truth_transformation.R, transformation.R)
    t_err = float(np.linalg.norm(ground_truth_transformation.t - transformation.t))

    q_pred = transformation.apply(p)
    nearest_matching = NearestNeighborMatcher().match(q_pred, q)
    residuals = np.linalg.norm(nearest_matching.source_points - nearest_matching.target_positions, axis=1)

    return rot_err, t_err, residuals
