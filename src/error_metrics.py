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
   stops being discriminative, and on a continuous scale by `correspondence_margin`,
   which still separates two extractors once that fraction has saturated at 0 or 1.

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
from scipy.spatial import KDTree
from tabulate import tabulate

from algebra_utils import rotation_angle
from feature_extractor import FeatureExtractor, zscored_features
from matcher import Matching, NearestNeighborMatcher, joint_embedding, joint_knn
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


def normalized_weight_entropy(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """How evenly each source point spreads its soft correspondence over its k neighbours.

    Shannon entropy of each row divided by its maximum, log k. 0 means all weight sits
    on one neighbour (a hard match); 1 means it is spread evenly over all k.

    Args:
        weights: float (N, k) per-neighbour weights, each row summing to 1, e.g. from
                 GaussianMatcher.neighbor_weights.

    Returns:
        float (N,) normalised entropy in [0, 1]. All zeros for k = 1, where the
        match is hard by construction.
    """
    k = weights.shape[1]
    if k < 2:
        return np.zeros(weights.shape[0])
    # 0 * log 0 = 0 by convention; the where keeps log from ever seeing a zero.
    plogp = np.where(weights > 0, weights * np.log(np.where(weights > 0, weights, 1.0)), 0.0)
    return -plogp.sum(axis=1) / np.log(k)


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


def _correspondence_and_features(
    source: PointCloud,
    target: PointCloud,
    feature_extractor: FeatureExtractor | None,
    correspondence: NDArray[np.int64] | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Resolve the correspondence array and z-scored features both locality metrics need.

    Args:
        source:            Source PointCloud (N, 3).
        target:            Target PointCloud (M, 3).
        feature_extractor: Extractor to characterise, or None for positions alone.
        correspondence:    (N,) ground-truth target index per source point, -1 where
                           the source point has no counterpart. None assumes the clouds
                           are index-aligned.

    Returns:
        Tuple (correspondence, feat_source, feat_target). The feature matrices are
        (N, D) and (M, D), or width 0 when feature_extractor is None.

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

    return correspondence, feat_source, feat_target


def _embedding(
    source: PointCloud,
    target: PointCloud,
    feat_source: NDArray[np.float64],
    feat_target: NDArray[np.float64],
    beta: float,
    use_positions: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Place both clouds in the space the locality metrics should search.

    Args:
        source:        Source PointCloud.
        target:        Target PointCloud.
        feat_source:   (N, D) z-scored source features.
        feat_target:   (M, D) z-scored target features.
        beta:          Feature influence relative to position; unused without positions.
        use_positions: True for the matcher's joint (position, feature) space; False
                       for descriptor space alone.

    Returns:
        Tuple of the two embedded point sets.
    """
    if use_positions:
        embedded_source, embedded_target, _ = joint_embedding(
            source.points, target.points, feat_source, feat_target, beta
        )
        return embedded_source, embedded_target
    # Features arrive already jointly z-scored, so plain Euclidean distance on them
    # is the descriptor-space metric.
    return feat_source, feat_target


def correspondence_margin(
    source: PointCloud,
    target: PointCloud,
    feature_extractor: FeatureExtractor | None = None,
    beta: float = 1.0,
    correspondence: NDArray[np.int64] | None = None,
    use_positions: bool = True,
) -> NDArray[np.float64]:
    """Per-pair ratio of the true correspondent's distance to the nearest impostor's.

    **Measures locality on a continuous scale.** A refinement of
    ``mutual_nearest_neighbor_fraction``, which answers the same question as a yes/no
    per point and therefore saturates: once every pair matches (or none does), it
    returns 1.0 (or 0.0) and can no longer separate two configurations. This keeps a
    gradient on both sides of that threshold, because it reports *by how much* the
    true correspondent wins or loses rather than merely whether it does.

    For each source point with a counterpart, the ratio is

        ``d(source, true correspondent) / d(source, nearest non-correspondent)``

    measured in the same joint (position, feature) space ``NearestNeighborMatcher``
    searches. A value below 1 means the correspondent is the nearest target point,
    i.e. the one-directional half of what the mutual-NN fraction requires; the
    smaller the value, the larger the margin protecting that match from noise. The
    ratio is scale-free, so the position z-scoring inside the joint embedding cancels.

    Normalising by the nearest *impostor* rather than by the mean distance over all
    target points is deliberate. The mean is dominated by far-away points and is
    close to constant across configurations, so a mean-normalised ratio would mostly
    re-report the numerator and would not register the over-smoothing failure that
    matters here: when a descriptor stops discriminating, impostors crowd in around
    the query point and this ratio rises towards 1 even though absolute feature
    distances barely move.

    Summarise with the median rather than the mean — the distribution is skewed, and
    a handful of hopeless points would otherwise dominate.

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
        use_positions:     False measures in descriptor space alone, ignoring
                           position and therefore ``beta``. Use it to score a feature
                           extractor as such: the joint space mixes in how well the
                           clouds happen to be aligned, so a descriptor's own
                           discriminative power cannot be read off it.

    Returns:
        (n_matched,) array of ratios, one per source point that has a counterpart,
        in source order. Empty when no source point has one.

    Raises:
        ValueError: If correspondence is None and the clouds differ in length.
    """
    correspondence, feat_source, feat_target = _correspondence_and_features(
        source, target, feature_extractor, correspondence
    )

    matched_source = np.flatnonzero(correspondence >= 0)
    if matched_source.size == 0:
        return np.empty(0, dtype=np.float64)
    matched_target = correspondence[matched_source]

    joint_source, joint_target = _embedding(
        source, target, feat_source, feat_target, beta, use_positions
    )
    queries = joint_source[matched_source]
    true_distance = np.linalg.norm(queries - joint_target[matched_target], axis=1)

    # Two neighbours, because the nearest one may be the true correspondent itself;
    # the impostor is then the runner-up.
    distances, neighbours = KDTree(joint_target).query(queries, k=2)
    correspondent_is_nearest = neighbours[:, 0] == matched_target
    impostor_distance = np.where(correspondent_is_nearest, distances[:, 1], distances[:, 0])

    return true_distance / np.maximum(impostor_distance, 1e-12)


def mutual_nearest_neighbor_fraction(
    source: PointCloud,
    target: PointCloud,
    feature_extractor: FeatureExtractor | None = None,
    beta: float = 1.0,
    correspondence: NDArray[np.int64] | None = None,
    use_positions: bool = True,
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
        use_positions:     False measures in descriptor space alone, ignoring
                           position and therefore ``beta``. Use it to score a feature
                           extractor as such: the joint space mixes in how well the
                           clouds happen to be aligned, so a descriptor's own
                           discriminative power cannot be read off it.

    Returns:
        Fraction in [0, 1] over the source points that have a counterpart. Returns
        0.0 when no source point has one.

    Raises:
        ValueError: If correspondence is None and the clouds differ in length.
    """
    correspondence, feat_source, feat_target = _correspondence_and_features(
        source, target, feature_extractor, correspondence
    )

    # Nearest neighbour in both directions, through the matcher's own joint metric so
    # that the beta normalisation and position z-scoring match what ICP would see.
    embedded_source, embedded_target = _embedding(
        source, target, feat_source, feat_target, beta, use_positions
    )
    forward = KDTree(embedded_target).query(embedded_source, k=1)[1]
    backward = KDTree(embedded_source).query(embedded_target, k=1)[1]
    forward = np.asarray(forward).reshape(-1)
    backward = np.asarray(backward).reshape(-1)

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
