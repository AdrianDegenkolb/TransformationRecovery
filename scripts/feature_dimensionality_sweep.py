#!/usr/bin/env python3
"""
Sweep: does feature dimensionality cost ICP anything?

Motivation: append mode concatenates 3 position dims with D feature dims into one
KDTree (matcher.joint_knn). The concern is that large D degrades the tree, either
in speed (tree -> linear scan) or in quality (distance concentration). Before
building a dimensionality-reducing component, we measure whether D costs anything.

Two arms disentangle "more dimensions" from "more information":

  informative — sweep RobustGeometricFeatureExtractor's quantile count, D = 5 + 2Q.
                Information grows WITH D, so this arm alone cannot attribute a
                change to dimensionality.
  padded      — fix the default D=11 descriptor and append columns of pure noise.
                Information is constant by construction, so any degradation is
                purely dimensional. This is the arm that decides the question, and
                the gap between pad=0 and pad=max upper-bounds what a perfect
                compressor could recover.

SUPERSEDED (2026-09-22): this script's beta policies predate the fix in
matcher.joint_knn, which now scales the feature block by beta*sqrt(3/D) internally.
Under the current code the 'fixed' policy is already the corrected one, and 'sqrtD'
double-corrects. Kept as-is to reproduce the recorded numbers in the experiment note;
use 'fixed' only for new runs.

Beta confound: zscore_jointly normalizes each feature dim to unit variance, so the
feature block's norm grows like beta*sqrt(D). Raising D therefore silently raises
effective feature influence. Each arm is run under two beta policies:

  fixed  — beta held constant; D and effective feature weight move together
  sqrtD  — beta = beta_ref * sqrt(D_ref) / sqrt(D), holding beta*sqrt(D) constant

A dimensionality effect that appears under 'fixed' but vanishes under 'sqrtD' is a
beta artifact, not a tree problem.

Regime: clean clouds at beta_ref=1.0 give ~70% reliability at D=11, leaving headroom
in both directions. beta=3 saturates at 100% (ceiling) and any dropout collapses to
0% (floor); neither can discriminate between D values.

Usage:
  uv run python scripts/feature_dimensionality_sweep.py
  uv run python scripts/feature_dimensionality_sweep.py --n-seeds 20 --style random
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment_runner import fit_multi_seed  # noqa: E402
from feature_extractor import FeatureExtractor, RobustGeometricFeatureExtractor  # noqa: E402
from icp import ICP  # noqa: E402
from matcher import NearestNeighborMatcher  # noqa: E402
from point_cloud import PointCloud  # noqa: E402
from synthetic import PerfectObserver  # noqa: E402

BetaPolicy = Literal["fixed", "sqrtD"]


class NoisePaddedFeatureExtractor(FeatureExtractor):
    """Wraps an extractor and appends columns of pure Gaussian noise.

    Inflates the feature dimension without adding information, isolating the cost
    of dimensionality itself. The noise is drawn independently per cloud, which is
    what a genuinely useless feature looks like to the joint KDTree: it contributes
    variance to every distance but no correspondence signal.

    Declared transformation-invariant so Matcher.prepare() draws the noise exactly
    once per cloud and caches it. This matters for correctness, not just speed: the
    generator advances between the source and target calls, giving the two clouds
    independent noise, while caching keeps each cloud's noise fixed across the ICP
    loop. Recomputing per iteration would instead resample the source's noise every
    step, testing a flickering metric rather than a high-dimensional one.
    """

    is_transformation_invariant: bool = True

    def __init__(self, inner: FeatureExtractor, pad: int, seed: int = 0) -> None:
        """
        Args:
            inner: Extractor supplying the informative feature block.
            pad:   Number of pure-noise columns to append. 0 is a pass-through.
            seed:  Seed for the noise generator.
        """
        self.inner = inner
        self.pad = pad
        self.target_dim = inner.target_dim + pad
        self._rng = np.random.default_rng(seed)

    def get_features(self, p: PointCloud) -> NDArray[np.float64]:
        """Compute the inner features with `pad` noise columns appended.

        Args:
            p: Input point cloud with N points.

        Returns:
            Float64 array of shape (N, inner.target_dim + pad).
        """
        feats = self.inner.get_features(p)
        if self.pad == 0:
            return feats
        noise = self._rng.standard_normal((len(feats), self.pad))
        return np.hstack([feats, noise]).astype(np.float64)


@dataclass
class SweepRow:
    """One (arm, setting) measurement.

    Attributes:
        arm:          'informative' or 'padded'.
        beta_policy:  'fixed' or 'sqrtD'.
        d:            Feature dimension D (excluding the 3 position dims).
        beta:         Beta actually used for this run.
        reliability:  Fraction of seeds converged to the global optimum.
        true_residual: Median over seeds of the mean true residual.
        iterations:   Median ICP iteration count over seeds.
        s_per_seed:   Wall time per seed.
    """
    arm: str
    beta_policy: str
    d: int
    beta: float
    reliability: float
    true_residual: float
    iterations: float
    s_per_seed: float


def _beta_for(policy: BetaPolicy, d: int, beta_ref: float, d_ref: int) -> float:
    """Resolve the beta to use for a given feature dimension.

    Args:
        policy:   'fixed' keeps beta constant; 'sqrtD' holds beta*sqrt(D) constant.
        d:        Feature dimension of the run.
        beta_ref: Beta at the reference dimension.
        d_ref:    Reference dimension at which beta equals beta_ref.

    Returns:
        Beta for this run.
    """
    if policy == "fixed":
        return beta_ref
    return beta_ref * np.sqrt(d_ref) / np.sqrt(d)


def _run_setting(
    extractor: FeatureExtractor,
    beta: float,
    seeds: list[int],
    experiment_kwargs: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Run one extractor/beta setting across all seeds.

    Args:
        extractor:         Feature extractor under test.
        beta:              Feature weight in the joint KDTree.
        seeds:             Seed list, shared across settings for paired comparison.
        experiment_kwargs: Forwarded to SyntheticExperiment.generate.

    Returns:
        Tuple (reliability, median mean-true-residual, median iterations, s/seed).
    """
    icp = ICP(
        matcher=NearestNeighborMatcher(feature_extractor=extractor, beta=beta),
        init_align_centroids=True,
    )
    start = time.perf_counter()
    result = fit_multi_seed(
        icp, seeds=seeds, experiment_kwargs=experiment_kwargs,
        observer=PerfectObserver(), verbose=False, n_jobs=1,
    )
    elapsed = time.perf_counter() - start
    return (
        result.convergence_to_global_opt_ratio,
        float(np.median(result.mean_true_residuals)),
        float(np.median([len(d) for d in result.deltas])),
        elapsed / len(seeds),
    )


def run_sweep(
    seeds: list[int],
    experiment_kwargs: dict[str, Any],
    k: int,
    beta_ref: float,
    quantile_counts: tuple[int, ...],
    pads: tuple[int, ...],
) -> list[SweepRow]:
    """Run both arms under both beta policies.

    Args:
        seeds:             Seed list, shared across every setting (paired comparison).
        experiment_kwargs: Forwarded to SyntheticExperiment.generate.
        k:                 Neighborhood size, held fixed across all settings.
        beta_ref:          Beta at the default D=11 descriptor.
        quantile_counts:   Q values for the informative arm (D = 5 + 2Q).
        pads:              Noise-column counts for the padded arm.

    Returns:
        One SweepRow per (arm, beta policy, setting).
    """
    base = RobustGeometricFeatureExtractor(k=k)
    d_ref = base.target_dim
    rows: list[SweepRow] = []

    for policy in ("fixed", "sqrtD"):
        for q in quantile_counts:
            fe = RobustGeometricFeatureExtractor(k=k, quantiles=tuple(np.linspace(0.1, 0.9, q)))
            beta = _beta_for(policy, fe.target_dim, beta_ref, d_ref)
            rel, res, iters, sps = _run_setting(fe, beta, seeds, experiment_kwargs)
            rows.append(SweepRow("informative", policy, fe.target_dim, beta, rel, res, iters, sps))
            print(f"  [{policy:5s}] informative D={fe.target_dim:2d} beta={beta:.2f} "
                  f"reliability={rel:.0%} res={res:.3f} iters={iters:.0f} {sps:.3f}s/seed")

        for pad in pads:
            fe_p = NoisePaddedFeatureExtractor(RobustGeometricFeatureExtractor(k=k), pad)
            beta = _beta_for(policy, fe_p.target_dim, beta_ref, d_ref)
            rel, res, iters, sps = _run_setting(fe_p, beta, seeds, experiment_kwargs)
            rows.append(SweepRow("padded", policy, fe_p.target_dim, beta, rel, res, iters, sps))
            print(f"  [{policy:5s}] padded      D={fe_p.target_dim:2d} beta={beta:.2f} "
                  f"reliability={rel:.0%} res={res:.3f} iters={iters:.0f} {sps:.3f}s/seed")

    return rows


def main() -> None:
    """Parse arguments, run the sweep, and print both arms as tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=50, help="Seeds per setting.")
    parser.add_argument("--n-points", type=int, default=2000, help="Points per cloud.")
    parser.add_argument("--style", type=str, default="muscle-fiber", help="Cloud style.")
    parser.add_argument("--k", type=int, default=20, help="Neighborhood size, fixed across settings.")
    parser.add_argument("--beta-ref", type=float, default=1.0, help="Beta at the default D=11 descriptor.")
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))
    experiment_kwargs: dict[str, Any] = {"n": args.n_points, "t_scale": 8.0, "style": args.style}

    print(f"Sweep: {args.n_seeds} seeds, style={args.style}, n={args.n_points}, "
          f"k={args.k}, beta_ref={args.beta_ref}, clean clouds")
    rows = run_sweep(
        seeds=seeds,
        experiment_kwargs=experiment_kwargs,
        k=args.k,
        beta_ref=args.beta_ref,
        quantile_counts=(1, 2, 3, 4, 6, 8),
        pads=(0, 4, 8, 16, 32),
    )

    for policy in ("fixed", "sqrtD"):
        for arm in ("informative", "padded"):
            subset = [r for r in rows if r.arm == arm and r.beta_policy == policy]
            print(f"\n{arm} arm, beta policy = {policy}")
            print(tabulate(
                [[r.d, r.beta, r.reliability, r.true_residual, r.iterations, r.s_per_seed] for r in subset],
                headers=["D", "beta", "Reliability", "Median True Res", "Median Iters", "s/seed"],
                floatfmt=(".0f", ".2f", ".0%", ".3f", ".0f", ".3f"), tablefmt="rounded_outline",
            ))


if __name__ == "__main__":
    main()
