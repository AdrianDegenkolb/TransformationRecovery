#!/usr/bin/env python3
"""
Benchmark: feature vector caching in ICP.

Features are currently recomputed inside matcher.match() on every ICP iteration,
for both the (changing) source cloud and the (fixed) target cloud. Since features
are invariant to rigid transforms on most cloud styles (see notebook 14), caching
them may save significant compute without changing ICP behaviour.

Three modes are compared per (cloud style, extractor):

  no_cache      — baseline: features recomputed every iteration
  cache_target  — target features pre-computed once; source recomputed live
  cache_both    — both source and target features pre-computed once at iter 0

For each mode we measure:
  - Mean time per ICP iteration
  - Feature extraction time as a fraction of total iteration time
  - Feature drift: MAE between cached and live source features per iteration
    (only meaningful for cache_both; shows where caching accumulates error)
  - Final mean residual (convergence quality)

Usage:
  uv run python scripts/benchmark_feature_caching.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from tabulate import tabulate

sys.path.insert(0, 'src')

from feature_extractor import (
    FeatureExtractor,
    GeometricFeatureExtractor,
    RobustGeometricFeatureExtractor,
)
from icp import ICP
from matcher import Matcher, Matching, _joint_knn
from point_cloud import PointCloud
from synthetic import CloudStyle, SyntheticExperiment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STYLES: list[CloudStyle] = ['random', 'clustered', 'lattice', '2d-lattice', 'muscle-fiber']

EXTRACTORS: dict[str, FeatureExtractor] = {
    'Geometric (9-dim)': GeometricFeatureExtractor(k=20),
    'Robust (11-dim)':   RobustGeometricFeatureExtractor(k=20),
}

N_POINTS   = 1000
N_SEEDS    = 5
MAX_ITER   = 60   # fixed, so runtimes are comparable regardless of convergence
BETA       = 1.0
BASE_SEED  = 42


# ---------------------------------------------------------------------------
# Instrumented feature extractor: records time spent in get_features()
# ---------------------------------------------------------------------------

class TimedFeatureExtractor:
    """Wraps a FeatureExtractor and records cumulative time and call count.

    Args:
        extractor: The underlying FeatureExtractor to wrap.
    """

    def __init__(self, extractor: FeatureExtractor) -> None:
        self._inner = extractor
        self.total_time_s: float = 0.0
        self.n_calls: int = 0

    def get_features(self, p: PointCloud) -> NDArray[np.float64]:
        """Forward to inner extractor, recording elapsed time.

        Args:
            p: Input point cloud.

        Returns:
            Feature matrix of shape (N, D).
        """
        t0 = time.perf_counter()
        result = self._inner.get_features(p)
        self.total_time_s += time.perf_counter() - t0
        self.n_calls += 1
        return result

    def reset(self) -> None:
        """Reset accumulated timing stats."""
        self.total_time_s = 0.0
        self.n_calls = 0


# ---------------------------------------------------------------------------
# Matchers: live (baseline) and cached variants
# ---------------------------------------------------------------------------

class LiveFeatureMatcher(Matcher):
    """Recomputes features from scratch on every match() call (baseline).

    Args:
        extractor: Timed wrapper around a FeatureExtractor.
        beta:      Scale of feature dimensions in the joint KDTree.
    """

    def __init__(self, extractor: TimedFeatureExtractor, beta: float = 1.0) -> None:
        self.extractor = extractor
        self.beta = beta

    def match(self, source: PointCloud, target: PointCloud) -> Matching:
        """Compute features fresh for both clouds and find nearest neighbours.

        Args:
            source: Current (transformed) source cloud.
            target: Fixed target cloud.

        Returns:
            Hard nearest-neighbour matching in joint position+feature space.
        """
        feat_src_raw = self.extractor.get_features(source)
        feat_tgt_raw = self.extractor.get_features(target)
        feat_src_z, feat_tgt_z = _joint_zscore(feat_src_raw, feat_tgt_raw)
        _, nbr_idx = _joint_knn(source.points, target.points, feat_src_z, feat_tgt_z, self.beta, k=1)
        return Matching(source_points=source.points, target_positions=target.points[nbr_idx[:, 0]])


class CachedFeatureMatcher(Matcher):
    """Nearest-neighbour matcher with pre-cached raw feature vectors.

    Args:
        extractor:     Timed wrapper around a FeatureExtractor.
        source:        Initial source cloud used to pre-compute source features.
        target:        Target cloud used to pre-compute target features.
        cache_source:  If True, source features are cached from iteration 0
                       and never recomputed (even as source positions change).
                       If False, only target features are cached; source features
                       are recomputed live each iteration.
        measure_drift: If True, compute live source features alongside cached ones
                       each iteration and record MAE. Must not be set during timing
                       runs — it adds a hidden get_features() call that skews wall time.
        beta:          Scale of feature dimensions in the joint KDTree.
    """

    def __init__(
        self,
        extractor: TimedFeatureExtractor,
        source: PointCloud,
        target: PointCloud,
        cache_source: bool,
        measure_drift: bool = False,
        beta: float = 1.0,
    ) -> None:
        self.extractor = extractor
        self.beta = beta
        self.cache_source = cache_source
        self.measure_drift = measure_drift
        self._cached_target_raw = extractor.get_features(target)
        self._cached_source_raw = extractor.get_features(source) if cache_source else None
        self.source_drift: list[float] = []

    def match(self, source: PointCloud, target: PointCloud) -> Matching:
        """Match using cached features for target (and optionally source).

        Args:
            source: Current (transformed) source cloud (positions used regardless).
            target: Fixed target cloud.

        Returns:
            Hard nearest-neighbour matching in joint position+feature space.
        """
        feat_tgt_raw = self._cached_target_raw

        if self.cache_source:
            feat_src_raw = self._cached_source_raw
            if self.measure_drift:
                feat_src_live = self.extractor._inner.get_features(source)
                self.source_drift.append(float(np.abs(feat_src_raw - feat_src_live).mean()))
        else:
            feat_src_raw = self.extractor.get_features(source)

        feat_src_z, feat_tgt_z = _joint_zscore(feat_src_raw, feat_tgt_raw)
        _, nbr_idx = _joint_knn(source.points, target.points, feat_src_z, feat_tgt_z, self.beta, k=1)
        return Matching(source_points=source.points, target_positions=target.points[nbr_idx[:, 0]])


def _joint_zscore(
    feat_src: NDArray[np.float64],
    feat_tgt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Z-score two feature matrices jointly (mirrors zscored_features).

    Args:
        feat_src: Raw source features (N, D).
        feat_tgt: Raw target features (M, D).

    Returns:
        Tuple (feat_src_z, feat_tgt_z), each z-scored using pooled statistics.
    """
    all_raw = np.concatenate([feat_src, feat_tgt], axis=0)
    mean = all_raw.mean(axis=0)
    std  = all_raw.std(axis=0) + 1e-8
    return (feat_src - mean) / std, (feat_tgt - mean) / std


# ---------------------------------------------------------------------------
# Benchmark data container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Results for one (style, extractor, mode, seed) trial.

    Attributes:
        style:              Cloud style used.
        extractor_name:     Human-readable extractor name.
        mode:               Caching mode ('no_cache', 'cache_target', 'cache_both').
        total_time_s:       Wall time for all ICP iterations.
        feat_time_s:        Time spent inside get_features() across all iterations.
        n_calls:            Total get_features() calls made.
        final_residual:     Mean point-to-point residual after the last iteration.
        source_drift:       Per-iteration MAE between cached and live source features.
                            Empty list unless mode == 'cache_both'.
    """
    style: str
    extractor_name: str
    mode: str
    total_time_s: float
    feat_time_s: float
    n_calls: int
    final_residual: float
    source_drift: list[float] = field(default_factory=list)

    @property
    def feat_fraction(self) -> float:
        """Fraction of total time spent in feature extraction."""
        return self.feat_time_s / max(self.total_time_s, 1e-9)

    @property
    def time_per_iter_ms(self) -> float:
        return self.total_time_s / MAX_ITER * 1000


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------

def run_trial(
    style: CloudStyle,
    ext_name: str,
    extractor: FeatureExtractor,
    mode: str,
    seed: int,
) -> BenchmarkResult:
    """Run one timed ICP trial. Drift measurement is never performed here.

    Args:
        style:     Cloud style to generate.
        ext_name:  Human-readable extractor name.
        extractor: Feature extractor instance.
        mode:      One of 'no_cache', 'cache_target', 'cache_both'.
        seed:      Random seed for experiment generation.

    Returns:
        BenchmarkResult with timing and quality metrics. source_drift is always empty.
    """
    exp = SyntheticExperiment.generate(n=N_POINTS, style=style, seed=seed)
    timed_ext = TimedFeatureExtractor(extractor)

    if mode == 'no_cache':
        matcher = LiveFeatureMatcher(timed_ext, beta=BETA)
    elif mode == 'cache_target':
        matcher = CachedFeatureMatcher(timed_ext, exp.P, exp.Q, cache_source=False, beta=BETA)
    elif mode == 'cache_both':
        # measure_drift=False: no hidden get_features() call during the timed run
        matcher = CachedFeatureMatcher(timed_ext, exp.P, exp.Q, cache_source=True,
                                       measure_drift=False, beta=BETA)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Reset timing counters after pre-computation in CachedFeatureMatcher.__init__
    timed_ext.reset()

    icp = ICP(matcher=matcher, max_iter=MAX_ITER, tol=1e-9, verbose=False, record_history=False)
    t0 = time.perf_counter()
    result = icp.fit(exp.P, exp.Q)
    total_time = time.perf_counter() - t0

    return BenchmarkResult(
        style=style,
        extractor_name=ext_name,
        mode=mode,
        total_time_s=total_time,
        feat_time_s=timed_ext.total_time_s,
        n_calls=timed_ext.n_calls,
        final_residual=result.mean_residuals[-1],
        source_drift=[],
    )


def run_drift_trial(
    style: CloudStyle,
    extractor: FeatureExtractor,
    seed: int,
) -> list[float]:
    """Measure per-iteration source feature drift for cache_both, separately from timing.

    Runs ICP with cache_both and measure_drift=True. This is intentionally separate
    from run_trial so that drift measurement never contaminates timing runs.

    Args:
        style:     Cloud style to generate.
        extractor: Feature extractor instance.
        seed:      Random seed for experiment generation.

    Returns:
        List of per-iteration MAE between cached and live source features.
    """
    exp = SyntheticExperiment.generate(n=N_POINTS, style=style, seed=seed)
    timed_ext = TimedFeatureExtractor(extractor)
    matcher = CachedFeatureMatcher(timed_ext, exp.P, exp.Q, cache_source=True,
                                   measure_drift=True, beta=BETA)
    timed_ext.reset()
    icp = ICP(matcher=matcher, max_iter=MAX_ITER, tol=1e-9, verbose=False, record_history=False)
    icp.fit(exp.P, exp.Q)
    return matcher.source_drift


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    modes = ['no_cache', 'cache_target', 'cache_both']
    all_results: list[BenchmarkResult] = []

    # --- Timing runs (no drift measurement) ---
    total_trials = len(STYLES) * len(EXTRACTORS) * len(modes) * N_SEEDS
    done = 0
    for style in STYLES:
        for ext_name, extractor in EXTRACTORS.items():
            for mode in modes:
                for seed in range(N_SEEDS):
                    r = run_trial(style, ext_name, extractor, mode, BASE_SEED + seed)
                    all_results.append(r)
                    done += 1
                    print(f'  [{done}/{total_trials}] {style:12s} | {ext_name:20s} | {mode:14s} | seed {seed}', end='\r')

    print()

    # --- Drift measurement (separate pass, does not affect timing results) ---
    print('Measuring source feature drift (cache_both only) ...')
    drift_data: dict[tuple[str, str], list[float]] = {}  # (style, ext_name) -> mean drift per iter
    for style in STYLES:
        for ext_name, extractor in EXTRACTORS.items():
            per_seed = [run_drift_trial(style, extractor, BASE_SEED + s) for s in range(N_SEEDS)]
            max_len = max(len(d) for d in per_seed)
            padded = np.array([d + [d[-1]] * (max_len - len(d)) for d in per_seed])
            drift_data[(style, ext_name)] = padded.mean(axis=0).tolist()
    print()

    # ------------------------------------------------------------------
    # Aggregate: mean over seeds
    # ------------------------------------------------------------------
    def mean_results(style: str, ext_name: str, mode: str) -> dict:
        rs = [r for r in all_results if r.style == style and r.extractor_name == ext_name and r.mode == mode]
        return {
            'total_ms':      np.mean([r.time_per_iter_ms for r in rs]),
            'feat_frac':     np.mean([r.feat_fraction for r in rs]),
            'feat_calls':    np.mean([r.n_calls for r in rs]),
            'residual':      np.mean([r.final_residual for r in rs]),
        }

    # ------------------------------------------------------------------
    # Table: timing and quality per (style, extractor, mode)
    # ------------------------------------------------------------------
    header = ['style', 'extractor', 'mode', 'ms/iter', 'feat %', 'feat calls/run', 'residual']
    rows = []
    for style in STYLES:
        for ext_name in EXTRACTORS:
            for mode in modes:
                m = mean_results(style, ext_name, mode)
                rows.append([
                    style, ext_name, mode,
                    f'{m["total_ms"]:.2f}',
                    f'{m["feat_frac"] * 100:.1f}%',
                    f'{m["feat_calls"]:.0f}',
                    f'{m["residual"]:.4f}',
                ])

    print('\n' + tabulate(rows, headers=header, tablefmt='github'))

    # ------------------------------------------------------------------
    # Summary: speedup from caching (cache_target and cache_both vs no_cache)
    # ------------------------------------------------------------------
    print('\n--- Speedup summary (ms/iter relative to no_cache) ---')
    speedup_rows = []
    for style in STYLES:
        for ext_name in EXTRACTORS:
            base = mean_results(style, ext_name, 'no_cache')['total_ms']
            for mode in ['cache_target', 'cache_both']:
                m = mean_results(style, ext_name, mode)
                speedup = base / m['total_ms']
                speedup_rows.append([style, ext_name, mode, f'{speedup:.2f}x'])
    print(tabulate(speedup_rows, headers=['style', 'extractor', 'mode', 'speedup'], tablefmt='github'))

    # ------------------------------------------------------------------
    # Plot: feature drift over iterations (cache_both, per style)
    # ------------------------------------------------------------------
    style_colors = {
        'random': 'tab:blue', 'clustered': 'tab:orange', 'lattice': 'tab:green',
        '2d-lattice': 'tab:red', 'muscle-fiber': 'tab:purple',
    }
    ext_linestyles = {'Geometric (9-dim)': '-', 'Robust (11-dim)': '--'}

    fig, axes = plt.subplots(1, len(STYLES), figsize=(4 * len(STYLES), 4), sharey=False)
    for ax, style in zip(axes, STYLES):
        for ext_name in EXTRACTORS:
            mean_drift = drift_data[(style, ext_name)]
            ax.plot(
                range(1, len(mean_drift) + 1), mean_drift,
                linestyle=ext_linestyles[ext_name],
                color=style_colors[style],
                label=ext_name,
            )
        ax.set_yscale('log')
        ax.set_title(style)
        ax.set_xlabel('ICP iteration')
        if ax is axes[0]:
            ax.set_ylabel('Source feature drift (MAE, log)')
            ax.legend(fontsize=7)
        ax.axhline(1e-10, color='grey', linestyle=':', linewidth=0.8, label='~fp zero')

    fig.suptitle('Source feature drift when caching features from iter 0 (cache_both)', y=1.02)
    plt.tight_layout()
    out_path = 'results/benchmark_feature_caching_drift.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nDrift plot saved to {out_path}')
    plt.show()

    # ------------------------------------------------------------------
    # Plot: ms/iter per style, grouped by mode
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(EXTRACTORS), figsize=(6 * len(EXTRACTORS), 4), sharey=False)
    x = np.arange(len(STYLES))
    width = 0.25
    mode_colors = {'no_cache': 'tab:grey', 'cache_target': 'tab:blue', 'cache_both': 'tab:green'}

    for ax, ext_name in zip(axes, EXTRACTORS):
        for i, mode in enumerate(modes):
            times = [mean_results(s, ext_name, mode)['total_ms'] for s in STYLES]
            ax.bar(x + (i - 1) * width, times, width, label=mode, color=mode_colors[mode], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(STYLES, rotation=20, ha='right')
        ax.set_ylabel('ms / ICP iteration')
        ax.set_title(ext_name)
        ax.legend(fontsize=8)

    fig.suptitle('ICP iteration time by caching mode', y=1.02)
    plt.tight_layout()
    out_path = 'results/benchmark_feature_caching_timing.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Timing plot saved to {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
