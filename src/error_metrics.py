"""Point-cloud residual and error-metric computations.

For a single run several error metrics exist:
- Rotation error: The angle (in degrees) between the fitted and ground-truth rotation.
- Translation error: The distance between the fitted and ground-truth translation.
- Closest point residual: The per-point distance from the transformed source to the nearest neighbor in the target.
- True residual: The per-point distance from the transformed source to the corresponding point in the target, assuming a known ground-truth correspondence.

The most reliable of these is the true residual, since it uses the known ground-truth correspondence.
In order to measure the reliability of a method across multiple runs, we can compute the following:
- Convergence ratio: The fraction of runs that have converged to a solution.
- Convergence to global optimum ratio: The fraction of runs that have converged to the globally optimal solution.

All of the above score a *fitted transformation*. A second family of metrics scores a
FeatureExtractor instead, before and independently of any ICP run, because a feature
extractor is pulled between two objectives that trade off against each other:

1. **Robustness** — a feature must survive perturbation. Points are observed with noise
   and dropout, so a descriptor computed on the source and on the target is computed
   from two different realisations of the same neighbourhood. Measured by
   `feature_correspondence_correlation`.
2. **Locality** — a feature must describe the neighbourhood of *this* point and not of
   the cloud at large, or it cannot distinguish one point from another. Measured
   indirectly by `mutual_nearest_neighbor_fraction`, which collapses when a descriptor
   stops being discriminative.

The tension is direct: robustness is bought by estimating over a larger neighbourhood,
which averages perturbation away but also makes the descriptor less local. Pushed far
enough, every point gets a near-identical descriptor — maximally robust and completely
useless. Neither metric detects this alone, which is why both are needed: a feature can
score near 1.0 on the first while the second collapses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate

from algebra_utils import rotation_angle
from feature_extractor import FeatureExtractor, zscored_features
from matcher import Matching, NearestNeighborMatcher, joint_knn
from point_cloud import PointCloud
from synthetic import SyntheticExperiment
from transformation import RigidTransformation

if TYPE_CHECKING:
    from icp import ICPResult, MultiStartICPResult


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


# --- Feature-extractor characterisation -------------------------------------------
# The two metrics below score a FeatureExtractor rather than a fitted transformation.
# See the module docstring for the robustness/locality trade-off they measure.


def feature_correspondence_correlation(
    feat_source: NDArray[np.float64],
    feat_target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-dimension correlation of each feature across true correspondences.

    **Measures robustness.** The premise of feature-augmented matching is that a true
    correspondence has near-zero feature distance. Perturbing each cloud independently
    breaks that premise, because the descriptor at a surviving point is then computed
    from a different realisation of its neighbourhood on each side. This reports how
    much of the agreement each dimension retains.

    Correlation rather than an absolute error because the matcher z-scores every
    dimension before use (see ``matcher.joint_knn``), which discards scale and offset:
    only co-variation survives into the metric the KDTree actually searches.

    A dimension scoring near 1.0 is trustworthy under the perturbation applied; one
    near 0.0 contributes noise to every distance it participates in, since the joint
    tree weights all z-scored dimensions equally.

    Args:
        feat_source: (n_matched, D) source features.
        feat_target: (n_matched, D) target features, row-aligned with feat_source so
                     that row i of each is the same ground-truth correspondence.

    Returns:
        (D,) array of Pearson correlations. A dimension with no variance on either
        side yields 0.0 rather than NaN, since a constant feature carries no signal.

    Raises:
        ValueError: If the two feature matrices are not row-aligned.
    """
    if feat_source.shape != feat_target.shape:
        raise ValueError(
            f"Feature matrices must be row-aligned by correspondence, got "
            f"{feat_source.shape} and {feat_target.shape}."
        )

    source_centered = feat_source - feat_source.mean(axis=0)
    target_centered = feat_target - feat_target.mean(axis=0)
    denom = np.linalg.norm(source_centered, axis=0) * np.linalg.norm(target_centered, axis=0)
    numer = (source_centered * target_centered).sum(axis=0)
    return np.where(denom > 1e-12, numer / np.maximum(denom, 1e-12), 0.0)


def mutual_nearest_neighbor_fraction(
    source: PointCloud,
    target: PointCloud,
    feature_extractor: FeatureExtractor | None = None,
    beta: float = 1.0,
    correspondence: NDArray[np.int64] | None = None,
) -> float:
    """Fraction of true pairs that are each other's nearest neighbor in the joint space.

    **Measures locality, and is the most direct predictor of whether matching works.**
    A true pair counts only when the target point is the source point's nearest
    neighbor *and* the source point is that target point's nearest neighbor, in the
    same joint (position, feature) space ``NearestNeighborMatcher`` searches. Requiring
    the match to be mutual rejects hub points that many sources map onto.

    This is what features are supposed to buy: pulling corresponding points together
    relative to everything else. Comparing the value at ``beta=0`` (positions only)
    with ``beta>0`` isolates the contribution of the features themselves.

    It is also the measurement that catches an over-smoothed descriptor. A feature
    estimated over a very large neighborhood scores highly on
    ``feature_correspondence_correlation`` while describing the cloud rather than the
    point; every point then looks alike, and this fraction collapses.

    Depends on the current alignment, since positions are part of the space. Pass
    ground-truth-aligned clouds to ask whether the metric *preserves* correct
    correspondences; pass misaligned clouds to ask whether it can *find* them.

    Args:
        source:            Source PointCloud (N, 3).
        target:            Target PointCloud (M, 3).
        feature_extractor: Extractor to characterise. None scores positions alone,
                           which is the baseline the features have to beat.
        beta:              Feature influence relative to position, as in
                           ``NearestNeighborMatcher``.
        correspondence:    (N,) ground-truth target index per source point, or -1
                           where the source point has no counterpart. None assumes
                           the clouds are index-aligned, which requires equal lengths.

    Returns:
        Fraction in [0, 1] over the source points that have a counterpart. Returns
        0.0 when no source point has one.

    Raises:
        ValueError: If correspondence is None and the clouds differ in length.
    """
    n_source = len(source.points)
    if correspondence is None:
        if n_source != len(target.points):
            raise ValueError(
                f"Index-aligned correspondence needs equal lengths, got "
                f"{n_source} and {len(target.points)}; pass `correspondence` instead."
            )
        correspondence = np.arange(n_source, dtype=np.int64)

    if feature_extractor is None:
        feat_source = np.zeros((n_source, 0), dtype=np.float64)
        feat_target = np.zeros((len(target.points), 0), dtype=np.float64)
    else:
        feat_source, feat_target = zscored_features(feature_extractor, [source, target])

    # Nearest neighbour in both directions, through the matcher's own joint metric so
    # that the beta normalisation and position z-scoring match what ICP would see.
    _, forward = joint_knn(source.points, target.points, feat_source, feat_target, beta, k=1)
    _, backward = joint_knn(target.points, source.points, feat_target, feat_source, beta, k=1)
    forward, backward = forward[:, 0], backward[:, 0]

    has_match = correspondence >= 0
    if not np.any(has_match):
        return 0.0

    matched_source = np.flatnonzero(has_match)
    matched_target = correspondence[has_match]
    mutual = (forward[matched_source] == matched_target) & (backward[matched_target] == matched_source)
    return float(np.mean(mutual))


def convergence_ratio(icp_results: list[ICPResult | MultiStartICPResult]) -> float:
    """
    Returns the fraction of runs that have converged to a solution.
    """
    return sum(r.converged for r in icp_results) / len(icp_results)


def convergence_to_global_opt_ratio(mean_true_residuals: list[float], tol: float = 1e-3) -> float:
    """
    Returns the fraction of runs that have converged to the globally optimal solution.

    Measured via each run's mean *true* residual (ground-truth point correspondence,
    e.g. MultiSeedSyntheticICPResult.mean_true_residuals) rather than the matcher's own
    internal residual (ICPResult.mean_residuals): a wrong-but-locally-self-consistent
    match (e.g. a symmetric cluster swap) can have a small matcher residual while being
    far from the actual ground truth, which would falsely count as "converged".

    Args:
        mean_true_residuals: One mean true-residual value per run.
        tol:                 Threshold below which a run counts as globally optimal.
    """
    return sum(r < tol for r in mean_true_residuals) / len(mean_true_residuals)
