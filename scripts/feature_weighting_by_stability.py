#!/usr/bin/env python3
"""
Test: does down-weighting dropout-fragile feature dimensions restore reliability?

scripts/feature_dropout_stability.py showed that per-dimension correspondence
correlation under dropout is strongly non-uniform (centroid_offset keeps ~0.90 of
its signal at 40% dropout, linearity keeps ~0.26). The joint KDTree weights every
z-scored dimension equally, so a fragile dimension injects noise into every
distance it participates in. This script asks whether re-weighting the dimensions
by their measured stability recovers any of the reliability lost to dropout.

Weighting must happen AFTER z-scoring. Scaling a dimension before z-scoring is
cancelled exactly by the z-scoring itself (zscore_jointly normalizes each dimension
to unit variance), so this cannot be implemented as a FeatureExtractor wrapper --
it has to reach into the matcher's cached, already-normalized features.

Schemes compared (r_j is the measured correlation at the dropout level under test):

  uniform      — baseline, every dimension weighted equally
  drop_eigen   — zero out the eigenvalue-ratio block except sphericity

(A former `drop_dup` scheme zeroed anisotropy as algebraically 1 - sphericity. The
extractors no longer emit anisotropy or planarity at all, since the four standard
shape ratios have rank 2, so that scheme is now the baseline.)
  linear_r     — w_j = r_j
  llr          — w_j = sqrt(r_j / (1 - r_j)), the likelihood-ratio-optimal weight
                 for separating true pairs from random pairs when each z-scored
                 difference is Gaussian with variance 2(1 - r_j) vs 2

Every weight vector is normalized to RMS 1 so the feature block's total norm is
unchanged. Without that, re-weighting would silently alter effective beta, which
the dimensionality sweep showed is itself a large effect -- the comparison would
measure a beta change rather than a re-allocation of weight.

Usage:
  uv run python scripts/feature_weighting_by_stability.py
  uv run python scripts/feature_weighting_by_stability.py --n-seeds 50
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_runner import fit_multi_seed  # noqa: E402
from feature_extractor import RobustGeometricFeatureExtractor  # noqa: E402
from icp import ICP  # noqa: E402
from matcher import NearestNeighborMatcher  # noqa: E402
from point_cloud import PointCloud  # noqa: E402
from synthetic import PointCloudObserver  # noqa: E402

from feature_dropout_stability import measure_stability, robust_feature_names  # noqa: E402


class WeightedNearestNeighborMatcher(NearestNeighborMatcher):
    """NearestNeighborMatcher applying per-dimension weights to z-scored features.

    Hooks ``prepare`` rather than ``match`` because the base class caches the
    jointly z-scored feature matrices there; scaling the cache once applies the
    weights to every subsequent ICP iteration at no per-iteration cost.

    Only valid for transformation-invariant extractors, which are the ones whose
    features the base class actually caches.
    """

    def __init__(self, weights: NDArray[np.float64], **kwargs: Any) -> None:
        """
        Args:
            weights:  (D,) per-dimension multipliers applied after z-scoring.
            **kwargs: Forwarded to NearestNeighborMatcher.
        """
        super().__init__(**kwargs)
        self.weights = weights

    def prepare(self, source: PointCloud, target: PointCloud) -> None:
        """Cache z-scored features as usual, then scale them by the weights.

        Args:
            source: Initial source PointCloud (N, 3).
            target: Fixed target PointCloud (M, 3).

        Raises:
            ValueError: If the extractor is not transformation-invariant, in which
                        case the base class does not populate the feature cache and
                        the weights would be silently ignored.
        """
        super().prepare(source, target)
        if not self._prepared:
            raise ValueError(
                "WeightedNearestNeighborMatcher requires a transformation-invariant "
                "extractor so that prepare() caches the z-scored features."
            )
        self._feat_src_z = self._feat_src_z * self.weights
        self._feat_tgt_z = self._feat_tgt_z * self.weights


def _normalize_rms(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale a weight vector to RMS 1, preserving the feature block's total norm.

    Args:
        weights: (D,) non-negative weights.

    Returns:
        (D,) weights with mean square 1.
    """
    rms = np.sqrt(np.mean(weights ** 2))
    return weights / max(rms, 1e-12)


def build_weight_schemes(
    correlations: NDArray[np.float64],
    names: list[str],
    r_clip: float = 0.99,
) -> dict[str, NDArray[np.float64]]:
    """Construct every weighting scheme from the measured per-dimension stabilities.

    Args:
        correlations: (D,) measured correspondence correlation per dimension at the
                      dropout level under test.
        names:        (D,) feature names, used for the hard-mask schemes.
        r_clip:       Upper clip on r before the llr weight, which diverges at r=1.

    Returns:
        Mapping from scheme name to an RMS-1 normalized (D,) weight vector.
    """
    d = len(correlations)
    r = np.clip(correlations, 0.0, r_clip)

    drop_eigen = np.ones(d)
    drop_eigen[names.index("linearity")] = 0.0

    return {
        "uniform": _normalize_rms(np.ones(d)),
        "drop_eigen": _normalize_rms(drop_eigen),
        "linear_r": _normalize_rms(r),
        "llr": _normalize_rms(np.sqrt(r / (1.0 - r))),
    }


def evaluate_scheme(
    weights: NDArray[np.float64],
    extractor: RobustGeometricFeatureExtractor,
    beta: float,
    dropout_prob: float,
    seeds: list[int],
    experiment_kwargs: dict[str, Any],
) -> NDArray[np.float64]:
    """Measure per-seed transformation error for one weighting scheme.

    Returns the full per-seed vector rather than a summary so that schemes can be
    compared pairwise on identical experiments. convergence_to_global_opt_ratio is
    unusable here: its 0.1 tolerance on the mean true residual is unreachable under
    any dropout at all (even 1% dropout lands near 0.5), so it reads 0% for every
    scheme and cannot discriminate.

    Args:
        weights:           (D,) per-dimension weights applied after z-scoring.
        extractor:         Feature extractor under test.
        beta:              Feature weight relative to position in the joint tree.
        dropout_prob:      Per-point dropout applied independently to each cloud.
        seeds:             Seed list, shared across schemes for paired comparison.
        experiment_kwargs: Forwarded to SyntheticExperiment.generate.

    Returns:
        (len(seeds),) array of mean true residual, one entry per seed.
    """
    icp = ICP(
        matcher=WeightedNearestNeighborMatcher(
            weights=weights, feature_extractor=extractor, beta=beta,
        ),
        init_align_centroids=True,
    )
    result = fit_multi_seed(
        icp, seeds=seeds, experiment_kwargs=experiment_kwargs,
        observer=PointCloudObserver(seed=1, noise_std=0.0, dropout_prob=dropout_prob),
        verbose=False, n_jobs=1,
    )
    return np.asarray(result.mean_true_residuals, dtype=np.float64)


def main() -> None:
    """Measure per-dimension stability, then test each weighting scheme against it."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-seeds", type=int, default=20, help="Seeds per configuration.")
    parser.add_argument("--n-points", type=int, default=2000, help="Points per cloud.")
    parser.add_argument("--style", type=str, default="muscle-fiber", help="Cloud style.")
    parser.add_argument("--k", type=int, default=20, help="Extractor neighborhood size.")
    args = parser.parse_args()

    quantiles = (0.25, 0.5, 0.75)
    extractor = RobustGeometricFeatureExtractor(k=args.k, quantiles=quantiles)
    names = robust_feature_names(quantiles)
    seeds = list(range(args.n_seeds))
    experiment_kwargs: dict[str, Any] = {"n": args.n_points, "t_scale": 8.0, "style": args.style}

    dropout_levels = (0.02, 0.05, 0.10)
    beta = 3.0

    print(f"Weighting test: {args.n_seeds} seeds, style={args.style}, n={args.n_points}, "
          f"k={args.k}, beta={beta:.0f}")
    stability = measure_stability(
        seeds=list(range(5)), dropout_probs=dropout_levels,
        extractor=extractor, experiment_kwargs=experiment_kwargs,
    )

    rows = []
    for dropout_prob in dropout_levels:
        schemes = build_weight_schemes(stability[dropout_prob], names)
        print(f"\nllr weights at dropout={dropout_prob:.0%}: "
              + ", ".join(f"{n}={w:.2f}" for n, w in zip(names, schemes["llr"])))

        per_seed = {
            name: evaluate_scheme(w, extractor, beta, dropout_prob, seeds, experiment_kwargs)
            for name, w in schemes.items()
        }
        baseline = per_seed["uniform"]
        for scheme_name, residuals in per_seed.items():
            # Paired comparison: same seeds, same experiments, so per-seed differences
            # cancel experiment difficulty and isolate the effect of the weighting.
            wins = float(np.mean(residuals < baseline)) if scheme_name != "uniform" else float("nan")
            rows.append([
                dropout_prob, scheme_name, float(np.median(residuals)),
                float(np.median(residuals - baseline)), wins,
            ])
            print(f"  drop={dropout_prob:.0%} {scheme_name:11s} median_res={np.median(residuals):.3f} "
                  f"median_delta={np.median(residuals - baseline):+.3f} wins={wins:.0%}")

    print()
    print(tabulate(
        rows,
        headers=["Dropout", "Scheme", "Median True Res", "Median Δ vs uniform", "Seeds beating uniform"],
        floatfmt=(".0%", "s", ".3f", "+.3f", ".0%"), tablefmt="rounded_outline",
    ))
    print("\nNegative Δ means the scheme beat the uniform baseline on the same experiments.")


if __name__ == "__main__":
    main()
