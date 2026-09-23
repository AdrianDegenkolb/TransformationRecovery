#!/usr/bin/env python3
"""
Diagnostic: which feature dimensions survive dropout?

Append-mode matching rests on one premise: for a true correspondence P[i] <-> Q[i],
the feature vectors agree, so the pair's joint distance equals its spatial distance
(see GaussianMatcher's docstring in matcher.py). Dropout breaks that premise at the
input, because it removes points from P and Q *independently* -- a point surviving
in both ends up with its descriptor computed from two different k-NN neighborhoods.

This script measures, per feature dimension, how much of that agreement survives.
For each dimension j we take the Pearson correlation between f(P)[:, j] and
f(Q)[:, j] over ground-truth-matched points, with and without dropout:

  r_clean  — correlation with no dropout. Should be ~1.0 for a transformation-
             invariant extractor; anything lower is baseline non-invariance
             (e.g. the k-NN tie-breaking caveat in FeatureExtractor's docstring).
  r_drop   — correlation under independent per-cloud dropout.
  retention — r_drop / r_clean. The fraction of usable signal the dimension keeps.

Why this matters for a learned compressor: a projection applied *after* extraction
cannot recover information dropout already destroyed, but it can down-weight the
dimensions that carry the most dropout-induced noise. That only helps if retention
varies across dimensions. If every dimension degrades uniformly, no linear
projection has anything to exploit and CMA-ES will find nothing.

Usage:
  uv run python scripts/feature_dropout_stability.py
  uv run python scripts/feature_dropout_stability.py --n-seeds 20 --style random
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feature_extractor import RobustGeometricFeatureExtractor  # noqa: E402
from point_cloud import PointCloud  # noqa: E402
from synthetic import SyntheticExperiment, make_correspondence_pair  # noqa: E402


def robust_feature_names(quantiles: tuple[float, ...]) -> list[str]:
    """Name each output column of RobustGeometricFeatureExtractor, in order.

    Mirrors the hstack order in RobustGeometricFeatureExtractor.get_features:
    distance quantiles, centroid offset, four eigenvalue-ratio shape features,
    then angle quantiles.

    Args:
        quantiles: Quantile levels the extractor was configured with.

    Returns:
        List of 2 * len(quantiles) + 5 column names.
    """
    return (
        [f"dist_q{int(q * 100):02d}" for q in quantiles]
        + ["centroid_offset", "linearity", "planarity", "sphericity", "anisotropy"]
        + [f"angle_q{int(q * 100):02d}" for q in quantiles]
    )


def _noise_feature_pairs(
    experiment: SyntheticExperiment,
    extractor: RobustGeometricFeatureExtractor,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Perturb both clouds with independent Gaussian noise and align by ground truth.

    The dropout counterpart removes points; this one keeps every point and jitters
    its position, so the neighbourhood *membership* is largely preserved while the
    geometry within it is perturbed. Correspondence is the identity, since
    SyntheticExperiment's P and Q are index-aligned and no point is removed.

    Args:
        experiment: Source of the index-aligned clean pair (P, Q).
        extractor:  Feature extractor under test.
        noise_std:  Standard deviation of the per-coordinate Gaussian noise,
                    applied independently to each cloud.
        rng:        Generator for the noise.

    Returns:
        Tuple (feat_p, feat_q), each (N, D), row-aligned by ground truth.
    """
    p_obs = PointCloud(experiment.P.points + rng.normal(0.0, noise_std, experiment.P.points.shape))
    q_obs = PointCloud(experiment.Q.points + rng.normal(0.0, noise_std, experiment.Q.points.shape))
    return extractor.get_features(p_obs), extractor.get_features(q_obs)


def _matched_feature_pairs(
    experiment: SyntheticExperiment,
    extractor: RobustGeometricFeatureExtractor,
    dropout_prob: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract features for one experiment and align them by ground-truth correspondence.

    Applies independent dropout to each side, recomputes features on the *observed*
    clouds (so each side's k-NN neighborhoods genuinely differ), then keeps only the
    point pairs whose correspondence survived on both sides.

    Args:
        experiment:   Source of the index-aligned clean pair (P, Q).
        extractor:    Feature extractor under test.
        dropout_prob: Per-point dropout probability, applied independently per side.
        rng:          Generator for the dropout and permutation.

    Returns:
        Tuple (feat_p, feat_q), each (n_matched, D), row-aligned so that row i of
        each is the same ground-truth correspondence.
    """
    p_obs, q_obs, correspondence = make_correspondence_pair(
        experiment.P, experiment.Q, dropout_prob=dropout_prob, rng=rng,
    )
    feat_p = extractor.get_features(p_obs)
    feat_q = extractor.get_features(q_obs)

    matched = correspondence >= 0
    return feat_p[correspondence[matched]], feat_q[matched]


def _per_dimension_correlation(
    feat_p: NDArray[np.float64],
    feat_q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pearson correlation between matched source and target values, per dimension.

    Args:
        feat_p: (n_matched, D) source features.
        feat_q: (n_matched, D) target features, row-aligned with feat_p.

    Returns:
        (D,) array of correlations. A dimension with zero variance on either side
        yields 0.0 rather than NaN, since a constant feature carries no signal.
    """
    p_centered = feat_p - feat_p.mean(axis=0)
    q_centered = feat_q - feat_q.mean(axis=0)
    denom = np.linalg.norm(p_centered, axis=0) * np.linalg.norm(q_centered, axis=0)
    numer = (p_centered * q_centered).sum(axis=0)
    return np.where(denom > 1e-12, numer / np.maximum(denom, 1e-12), 0.0)


def measure_stability(
    seeds: list[int],
    dropout_probs: tuple[float, ...],
    extractor: RobustGeometricFeatureExtractor,
    experiment_kwargs: dict[str, Any],
    mode: str = "dropout",
) -> dict[float, NDArray[np.float64]]:
    """Measure per-dimension correspondence correlation at several perturbation levels.

    Args:
        seeds:             One synthetic experiment per seed; correlations are
                           averaged over seeds.
        dropout_probs:     Perturbation levels to evaluate. Include 0.0 for the
                           baseline. Interpreted as dropout probabilities when
                           mode='dropout', and as noise standard deviations when
                           mode='noise'.
        extractor:         Feature extractor under test.
        experiment_kwargs: Forwarded to SyntheticExperiment.generate.
        mode:              'dropout' removes points independently per cloud;
                           'noise' jitters every point's position instead. The two
                           stress different things: dropout changes which points are
                           in a neighbourhood, noise changes where they are.

    Returns:
        Mapping from perturbation level to a (D,) array of mean correlations.
    """
    if mode not in ("dropout", "noise"):
        raise ValueError(f"mode must be 'dropout' or 'noise', got {mode!r}")
    pair_fn = _matched_feature_pairs if mode == "dropout" else _noise_feature_pairs

    out: dict[float, NDArray[np.float64]] = {}
    for level in dropout_probs:
        per_seed = []
        for seed in seeds:
            experiment = SyntheticExperiment.generate(**experiment_kwargs, seed=seed)
            rng = np.random.default_rng(seed)
            feat_p, feat_q = pair_fn(experiment, extractor, level, rng)
            per_seed.append(_per_dimension_correlation(feat_p, feat_q))
        out[level] = np.mean(per_seed, axis=0)
    return out


def main() -> None:
    """Parse arguments, run the diagnostic, and print correlations and retention."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-seeds", type=int, default=10, help="Experiments to average over.")
    parser.add_argument("--n-points", type=int, default=2000, help="Points per cloud.")
    parser.add_argument("--style", type=str, default="muscle-fiber", help="Cloud style.")
    parser.add_argument("--k", type=int, default=20, help="Extractor neighborhood size.")
    args = parser.parse_args()

    quantiles = (0.25, 0.5, 0.75)
    extractor = RobustGeometricFeatureExtractor(k=args.k, quantiles=quantiles)
    names = robust_feature_names(quantiles)
    dropout_probs = (0.0, 0.1, 0.2, 0.4)

    experiment_kwargs: dict[str, Any] = {"n": args.n_points, "t_scale": 8.0, "style": args.style}
    print(f"Dropout stability: {args.n_seeds} seeds, style={args.style}, "
          f"n={args.n_points}, k={args.k}, D={extractor.target_dim}")

    correlations = measure_stability(
        seeds=list(range(args.n_seeds)),
        dropout_probs=dropout_probs,
        extractor=extractor,
        experiment_kwargs=experiment_kwargs,
    )

    clean = correlations[0.0]
    rows = []
    for j, name in enumerate(names):
        retention = [
            correlations[d][j] / clean[j] if abs(clean[j]) > 1e-9 else 0.0
            for d in dropout_probs[1:]
        ]
        rows.append([name, *[correlations[d][j] for d in dropout_probs], *retention])

    rows.sort(key=lambda r: r[-1], reverse=True)
    print(tabulate(
        rows,
        headers=(
            ["Feature"]
            + [f"r @ drop={d:.0%}" for d in dropout_probs]
            + [f"retention @ {d:.0%}" for d in dropout_probs[1:]]
        ),
        floatfmt=".3f", tablefmt="rounded_outline",
    ))
    print("\nSorted by retention at the highest dropout level (most robust first).")


if __name__ == "__main__":
    main()
