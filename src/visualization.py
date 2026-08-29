"""Visualisation utilities for point cloud registration experiments.

Provides four static-method classes:
    PointCloudVisualizer   — raw point cloud scatter/projection plots.
    ErrorMetricsVisualizer — convergence curves, seed box plots, parameter sweeps.
    ResidualVisualizer     — histograms, field heatmaps, rotation bases.
    OptimizationVisualizer — HPO Pareto front and feature importance.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from algebra_utils import rotation_angle
from icp import ICPResult, MultiStartICPResult
from point_cloud import PointCloud
from transformation import RigidTransformation

_DEFAULT_COLORS: list[str] = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Common boxplot style used across all seed-distribution plots.
_BOX_STYLE: dict[str, object] = dict(
    patch_artist=True,
    whiskerprops=dict(color='#333333', linewidth=1.5),
    capprops=dict(color='#333333', linewidth=1.5),
    medianprops=dict(color='gold', linewidth=2),
    flierprops=dict(
        marker='o', markersize=5, markerfacecolor='#555555',
        markeredgewidth=0, alpha=0.6, linestyle='none',
    ),
)


class PointCloudVisualizer:
    """Visualization methods for 3D point clouds."""
    _PROJECTIONS: list[tuple[int, int, str, str, str]] = [
        (0, 1, 'x', 'y', 'XY'),
        (0, 2, 'x', 'z', 'XZ'),
        (1, 2, 'y', 'z', 'YZ'),
    ]

    @staticmethod
    def plot_xy(
        ax: Axes,
        clouds: list[PointCloud],
        labels: list[str],
        colors: list[str],
        idx: NDArray[np.intp] | None = None,
    ) -> None:
        """Scatter-plot one or more point clouds projected onto the XY plane.

        Args:
            ax:     Axes to draw on.
            clouds: Point clouds to plot.
            labels: Legend label per cloud.
            colors: Colour per cloud.
            idx:    Optional index array for consistent subsampling across clouds.
        """
        for cloud, label, color in zip(clouds, labels, colors):
            pts = cloud.points if idx is None else cloud.points[idx]
            ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.4, color=color, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    @staticmethod
    def plot_projections(
        axes: Sequence[Axes],
        cloud: PointCloud,
        color: str = 'tab:blue',
        point_size: int = 3,
        alpha: float = 0.4,
    ) -> None:
        """Plot XY, XZ, and YZ projections of a 3D point cloud side by side.

        Args:
            axes:       Sequence of exactly 3 Axes (XY, XZ, YZ).
            cloud:      Point cloud to visualize.
            color:      Scatter color.
            point_size: Scatter point size.
            alpha:      Scatter transparency.
        """
        pts = cloud.points
        for ax, (i, j, xl, yl, title) in zip(axes, PointCloudVisualizer._PROJECTIONS):
            ax.scatter(pts[:, i], pts[:, j], s=point_size, alpha=alpha, color=color)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(title)
            ax.set_aspect('equal')

    @staticmethod
    def plot_alignment_snapshots(
        axes: Sequence[Axes],
        cloud_history: list[PointCloud],
        target: PointCloud,
        final_cloud: PointCloud,
        snap_iters: Sequence[int],
        idx: NDArray[np.intp] | None = None,
        source_color: str = 'tab:blue',
        target_color: str = 'tab:red',
    ) -> None:
        """Plot source-vs-target alignment at selected iterations and at convergence.

        The first ``len(snap_iters)`` axes show intermediate snapshots; the last
        axis shows the final aligned state.

        Args:
            axes:          Axes to draw on. Must have ``len(snap_iters) + 1`` elements.
            cloud_history: Source cloud at the start of each ICP iteration.
            target:        Fixed target point cloud.
            final_cloud:   Source cloud after the final transformation.
            snap_iters:    0-based iteration indices to visualize.
            idx:           Optional subsampling index applied consistently to all clouds.
            source_color:  Colour for the source cloud.
            target_color:  Colour for the target cloud.
        """
        assert len(axes) == len(snap_iters) + 1, (
            f"Expected {len(snap_iters) + 1} axes (one per snapshot + final), got {len(axes)}"
        )
        for ax, it in zip(axes, snap_iters):
            PointCloudVisualizer.plot_xy(
                ax,
                [cloud_history[it], target],
                ['P (current)', 'Q (target)'],
                [source_color, target_color],
                idx=idx,
            )
            ax.set_title(f'Iter {it + 1}')
            ax.grid(False)

        axes[0].legend(markerscale=2, loc='upper left', fontsize=7)

        PointCloudVisualizer.plot_xy(
            axes[-1],
            [final_cloud, target],
            ['P (final)', 'Q (target)'],
            [source_color, target_color],
            idx=idx,
        )
        axes[-1].grid(False)
        axes[-1].set_title('Final (converged)')

    @staticmethod
    def plot_cluster_coloring(
        ax: Axes,
        cloud: PointCloud,
        labels: NDArray[np.int64],
    ) -> None:
        """3D scatter of a point cloud colored by feature-space cluster label.

        Noise points (label -1) are drawn in black. Each cluster gets a distinct
        color cycling through the tab20 colormap.

        Args:
            ax:     3D Axes to draw on (must be created with projection='3d').
            cloud:  Point cloud with N points.
            labels: Per-point cluster label array of shape (N,).
        """
        import matplotlib.pyplot as plt
        colors = _cluster_point_colors(labels, plt.cm.tab20)
        ax.scatter(cloud.points[:, 0], cloud.points[:, 1], cloud.points[:, 2], c=colors, s=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Feature-space clusters (black = noise)')

    @staticmethod
    def plot_trimmer_result(
        ax: Axes,
        cloud: PointCloud,
        labels: NDArray[np.int64],
        large_labels: set[int],
    ) -> None:
        """3D scatter showing trimmer outcome: discarded points in gray, kept points colored.

        Points in large clusters are drawn in gray with low alpha. All other points
        (small clusters and noise) retain their cluster color.

        Args:
            ax:           3D Axes to draw on (must be created with projection='3d').
            cloud:        Point cloud with N points.
            labels:       Per-point cluster label array of shape (N,).
            large_labels: Set of cluster labels considered large (to be discarded).
        """
        import matplotlib.pyplot as plt
        point_colors = _cluster_point_colors(labels, plt.cm.tab20)
        is_large = np.isin(labels, list(large_labels))

        if is_large.any():
            ax.scatter(
                cloud.points[is_large, 0],
                cloud.points[is_large, 1],
                cloud.points[is_large, 2],
                c='gray', alpha=0.15, s=10,
                label=f'discarded ({is_large.sum()})',
            )

        kept = ~is_large
        ax.scatter(
            cloud.points[kept, 0],
            cloud.points[kept, 1],
            cloud.points[kept, 2],
            c=point_colors[kept],
            s=10,
            label=f'kept ({kept.sum()})',
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'Trimmer: {kept.sum()} / {len(cloud)} kept')
        ax.legend(fontsize=7)


class ErrorMetricsVisualizer:
    """Visualisation methods for ICP error metrics and performance distributions."""

    @staticmethod
    def plot_rot_error_per_iterations(
        axis: Axes,
        results_per_method: list[list[ICPResult]],
        ground_truths_per_method: list[list[RigidTransformation]],
        method_labels: list[str],
        colors: list[str] | None = None,
        x_label: str = 'Iteration',
    ) -> None:
        """Plot per-iteration rotation error as mean ± 1 std shaded bands for each method.

        Args:
            axis:                     Axes to draw on.
            results_per_method:       Per-seed ICPResult lists, one per method.
            ground_truths_per_method: Per-seed ground-truth transformations, matching
                                      the structure of ``results_per_method``.
            method_labels:            Legend label per method.
            colors:                   Line/fill color per method. Defaults to tab palette.
            x_label:                  X-axis label.
        """
        if colors is None:
            colors = _DEFAULT_COLORS[:len(method_labels)]

        for results, ground_truths, label, color in zip(
            results_per_method, ground_truths_per_method, method_labels, colors
        ):
            trajectories = [
                np.array([rotation_angle(T_gt.R, T.R) for T in result.transform_history])
                for result, T_gt in zip(results, ground_truths)
            ]
            _plot_shaded_band(
                axis, trajectories, label, color,
                ylabel='Rotation error (°)',
                title='Rotation error vs. ground truth',
                x_label=x_label,
            )
        axis.legend()

    @staticmethod
    def plot_translation_error_per_iterations(
        axis: Axes,
        results_per_method: list[list[ICPResult]],
        ground_truths_per_method: list[list[RigidTransformation]],
        method_labels: list[str],
        colors: list[str] | None = None,
        x_label: str = 'Iteration',
    ) -> None:
        """Plot per-iteration translation error as mean ± 1 std shaded bands for each method.

        Args:
            axis:                     Axes to draw on.
            results_per_method:       Per-seed ICPResult lists, one per method.
            ground_truths_per_method: Per-seed ground-truth transformations, matching
                                      the structure of ``results_per_method``.
            method_labels:            Legend label per method.
            colors:                   Line/fill color per method. Defaults to tab palette.
            x_label:                  X-axis label.
        """
        if colors is None:
            colors = _DEFAULT_COLORS[:len(method_labels)]

        for results, ground_truths, label, color in zip(
            results_per_method, ground_truths_per_method, method_labels, colors
        ):
            trajectories = [
                np.array([float(np.linalg.norm(T_gt.t - T.t)) for T in result.transform_history])
                for result, T_gt in zip(results, ground_truths)
            ]
            _plot_shaded_band(
                axis, trajectories, label, color,
                ylabel='Translation error',
                title='Translation error vs. ground truth',
                x_label=x_label,
            )
        axis.legend()

    @staticmethod
    def plot_residual_errors_per_iteration(
        axis: Axes,
        results_per_method: list[list[ICPResult]],
        method_labels: list[str],
        colors: list[str] | None = None,
        x_label: str = 'Iteration',
    ) -> None:
        """Plot per-iteration mean residuals as mean ± 1 std shaded bands for each method.

        Args:
            axis:               Axes to draw on.
            results_per_method: Per-seed ICPResult lists, one per method.
            method_labels:      Legend label per method.
            colors:             Line/fill color per method. Defaults to tab palette.
            x_label:            X-axis label.
        """
        if colors is None:
            colors = _DEFAULT_COLORS[:len(method_labels)]

        for results, label, color in zip(results_per_method, method_labels, colors):
            trajectories = [result.mean_residuals for result in results]
            _plot_shaded_band(
                axis, trajectories, label, color,
                ylabel='Mean residual',
                title='Mean residuals',
                x_label=x_label,
            )
        axis.legend()

    @staticmethod
    def plot_delta_per_iterations(
        axis: Axes,
        results_per_method: list[list[ICPResult]],
        method_labels: list[str],
        colors: list[str] | None = None,
        x_label: str = 'Iteration',
    ) -> None:
        """Plot per-iteration convergence deltas as mean ± 1 std shaded bands for each method.

        Args:
            axis:               Axes to draw on.
            results_per_method: Per-seed ICPResult lists, one per method.
            method_labels:      Legend label per method.
            colors:             Line/fill color per method. Defaults to tab palette.
            x_label:            X-axis label.
        """
        if colors is None:
            colors = _DEFAULT_COLORS[:len(method_labels)]

        for results, label, color in zip(results_per_method, method_labels, colors):
            trajectories = [result.deltas for result in results]
            _plot_shaded_band(
                axis, trajectories, label, color,
                ylabel='Delta',
                title='Convergence delta',
                x_label=x_label,
            )
            axis.set_yscale("log")

        axis.legend()

    @classmethod
    def plot_convergence(
        cls,
        results_per_method: list[list[ICPResult | MultiStartICPResult]],
        ground_truths_per_method: list[list[RigidTransformation]],
        method_labels: list[str],
        colors: list[str] | None = None,
        x_label: str = 'Iteration',
    ) -> None:
        """Plot a 1×4 figure of per-iteration error trajectories as mean ± 1 std shaded bands.

        Creates subplots for rotation error, translation error, mean residuals, and
        convergence deltas, then delegates to the respective sub-methods.

        Args:
            results_per_method:       Iterable of per-seed ICPResult lists, one per method.
            ground_truths_per_method: Iterable of per-seed ground-truth RigidTransformation
                                      lists, matching ``results_per_method``.
            method_labels:            Legend label per method.
            colors:                   Line/fill color per method. Defaults to tab palette.
            x_label:                  Shared x-axis label.
        Return:
            fig, axes
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        cls.plot_rot_error_per_iterations(axes[0], results_per_method, ground_truths_per_method, method_labels, colors, x_label)
        cls.plot_translation_error_per_iterations(axes[1], results_per_method, ground_truths_per_method, method_labels, colors, x_label)
        cls.plot_residual_errors_per_iteration(axes[2], results_per_method, method_labels, colors, x_label)
        cls.plot_delta_per_iterations(axes[3], results_per_method, method_labels, colors, x_label)

        return fig, axes

    @staticmethod
    def plot_hists_over_seeds(
        rot_errors_per_method: list[NDArray[np.float64]],
        translation_errors_per_method: list[NDArray[np.float64]],
        residuals_per_method: list[NDArray[np.float64]],
        method_labels: list[str] | None = None,
        method_colors: list[str] | None = None,
    ) -> tuple[Figure, NDArray[np.float64]]:
        """Histograms of rotation error, translation error, and residual distributions.

        Creates a 1×3 figure with one overlaid histogram per metric.

        Args:
            rot_errors_per_method:         1D rotation error array per method.
            translation_errors_per_method: 1D translation error array per method.
            residuals_per_method:          1D residual array per method.
            method_labels:                 Legend label per method.
            method_colors:                 Bar color per method. Defaults to tab palette.

        Returns:
            fig, axes
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        _plot_hist(axes[0], rot_errors_per_method, 'Rotation error (°)', 'Rotation Error', method_labels, method_colors)
        _plot_hist(axes[1], translation_errors_per_method, 'Translation error', 'Translation Error', method_labels, method_colors)
        _plot_hist(axes[2], residuals_per_method, 'Residual', 'Residuals', method_labels, method_colors)
        return fig, axes

    @staticmethod
    def plot_parameter_sweep(
        axes: Sequence[Axes],
        x: Sequence[float | int],
        data_per_metric: list[NDArray[np.float64]],
        y_labels: list[str],
        x_label: str = '',
        colors: list[str] | None = None,
    ) -> None:
        """Plot mean ± 1 std shaded band for each metric over a parameter sweep.

        Args:
            axes:            Sequence of Axes, one per metric.
            x:               Parameter values on the x-axis.
            data_per_metric: For each metric, a 2D array of shape
                             ``(len(x), n_seeds)``. Mean and std are computed
                             over seeds (``axis=1``).
            y_labels:        Y-axis label per metric (also used as subplot title).
            x_label:         Shared x-axis label.
            colors:          Line color per metric. Defaults to tab palette.
        """
        if colors is None:
            colors = _DEFAULT_COLORS[:len(data_per_metric)]

        for ax, data, ylabel, color in zip(axes, data_per_metric, y_labels, colors):
            mean = data.mean(axis=1)
            std = data.std(axis=1)
            ax.plot(x, mean, marker='o', ms=5, color=color, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.25, color=color)
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.set_xticks(list(x))


class ResidualVisualizer:
    """Visualisation methods for residual field analysis and rotation recovery."""

    @staticmethod
    def plot_histogram(
        ax: Axes,
        residuals_per_method: list[NDArray[np.float64]],
        labels: list[str],
        bins: int = 30,
        colors: list[str] | None = None,
        alpha: float = 0.6,
    ) -> None:
        """Overlay residual histograms for one or more methods.

        Args:
            ax:                   Axes to draw on.
            residuals_per_method: 1D residual array per method.
            labels:               Legend label per method.
            bins:                 Number of histogram bins.
            colors:               Bar color per method. Defaults to tab palette.
            alpha:                Bar transparency.
        """
        if colors is None:
            colors = _DEFAULT_COLORS[:len(residuals_per_method)]
        for data, label, color in zip(residuals_per_method, labels, colors):
            ax.hist(data, bins=bins, alpha=alpha, label=label, color=color)
        ax.set_xlabel('Point residual')
        ax.set_ylabel('Count')
        ax.set_title('Point residuals P → Q')
        ax.legend()

    @staticmethod
    def plot_field_heatmap(
        ax: Axes,
        points: NDArray[np.float64],
        magnitudes: NDArray[np.float64],
        cmap: str = 'viridis',
        point_size: int = 15,
    ) -> None:
        """Scatter plot of XY point positions colored by scalar field magnitude.

        Args:
            ax:         Axes to draw on.
            points:     Point positions, shape ``(N, 2)`` or ``(N, 3)``.
                        Only the first two columns (x, y) are used.
            magnitudes: Scalar magnitude per point, shape ``(N,)``.
            cmap:       Matplotlib colormap name.
            point_size: Scatter point size.
        """
        sc = ax.scatter(points[:, 0], points[:, 1], c=magnitudes, cmap=cmap, s=point_size)
        ax.figure.colorbar(sc, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Residual field |u| (XY projection)')

class TransformationVisualizer:
    """Visualization methods for transformations"""

    @staticmethod
    def plot_rotation_bases(
        ax: Axes,
        rotations: list[NDArray[np.float64]],
        labels: list[str],
        colors: list[str]  | None = None,
        line_widths: list[float] | None = None,
    ) -> None:
        """Draw XY-projected basis vectors for one or more rotation matrices.

        Each rotation matrix contributes two arrows (its column vectors projected
        onto XY). By convention, the first entry is assumed to be the ground truth
        and is drawn with a thicker line.

        Args:
            ax:         Axes to draw on.
            rotations:  List of ``(3, 3)`` rotation matrices.
            labels:     Legend label per rotation.
            colors:     Arrow color per rotation.
            line_widths: Arrow line width per rotation. Defaults to ``1.5`` for
                         the first (ground-truth) entry and ``1.0`` for the rest.
        """
        if line_widths is None:
            line_widths = [1.5] + [1.0] * (len(rotations) - 1)

        if colors is None:
            colors = _DEFAULT_COLORS[:len(rotations)]

        origin = np.zeros(2)
        for R, label, color, lw in zip(rotations, labels, colors, line_widths):
            for axis_vec in R.T:
                ax.annotate(
                    '', xy=axis_vec[:2], xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                )
            ax.plot([], [], color=color, label=label)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title('Rotation basis vectors (XY projection)')
        ax.legend()
        ax.grid(True, alpha=0.3)


class TimeVisualizer:
    """Visualisation methods for ICP runtime distributions."""

    @staticmethod
    def plot_duration_distribution(
        ax: Axes,
        durations_per_method: list[NDArray[np.float64]],
        method_labels: list[str],
        colors: list[str] | None = None,
        unit: str = 's',
    ) -> None:
        """Box plot of wall-clock duration across seeds for each method.

        Args:
            ax:                   Axes to draw on.
            durations_per_method: 1D array of per-seed durations per method.
            method_labels:        Label per method (used as x-tick labels).
            colors:               Box fill color per method. Defaults to tab palette.
            unit:                 Time unit label shown on the y-axis (e.g. ``'s'`` or ``'ms'``).
        """
        if colors is None:
            colors = _DEFAULT_COLORS[:len(durations_per_method)]

        bp = ax.boxplot(
            durations_per_method,
            labels=method_labels,
            **_BOX_STYLE,
        )
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(f'Duration ({unit})')
        ax.set_title('Runtime distribution across seeds')


class OptimizationVisualizer:
    """Visualisation methods for multi-objective hyperparameter optimization results."""

    @staticmethod
    def plot_pareto_front(
        ax: Axes,
        x_values: NDArray[np.float64],
        y_values: NDArray[np.float64],
        group_ids: list[str],
        pareto_mask: NDArray[np.bool_],
        colors: dict[str, str] | None = None,
        x_label: str = 'Objective 1',
        y_label: str = 'Objective 2',
    ) -> None:
        """Scatter plot of optimization trials with Pareto-optimal points highlighted.

        Points are colored by group (e.g. matching strategy). Pareto-optimal
        points within each group receive a red edge.

        Args:
            ax:          Axes to draw on.
            x_values:    First objective value per trial, shape ``(N,)``.
            y_values:    Second objective value per trial, shape ``(N,)``.
            group_ids:   Group identifier per trial (e.g. ``'hard'`` / ``'soft'``).
            pareto_mask: Boolean mask marking Pareto-optimal trials, shape ``(N,)``.
            colors:      Mapping from group identifier to colour. Defaults to tab palette.
            x_label:     X-axis label.
            y_label:     Y-axis label.
        """
        unique_groups = sorted(set(group_ids))
        if colors is None:
            colors = {g: _DEFAULT_COLORS[i] for i, g in enumerate(unique_groups)}

        group_array = np.array(group_ids)
        for group in unique_groups:
            mask = group_array == group
            ax.scatter(x_values[mask], y_values[mask], color=colors[group], alpha=0.4, s=25, label=group)
            pareto_group = mask & pareto_mask
            if pareto_group.any():
                ax.scatter(
                    x_values[pareto_group], y_values[pareto_group],
                    color=colors[group], edgecolors='red', linewidths=1.5, s=60, zorder=5,
                )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(title='Group')

    @staticmethod
    def plot_feature_importance(
        ax: Axes,
        importances: dict[str, float],
        title: str = '',
        color: str = 'steelblue',
    ) -> None:
        """Horizontal barplot of hyperparameter relative importances.

        Args:
            ax:          Axes to draw on.
            importances: Mapping from parameter name to relative importance in ``[0, 1]``.
            title:       Axes title.
            color:       Bar color.
        """
        params = list(importances.keys())
        vals = list(importances.values())
        ax.barh(params[::-1], vals[::-1], color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Relative importance')
        ax.set_title(title)

def _plot_hist(
    ax: Axes,
    data_per_method: list[NDArray[np.float64]],
    xlabel: str,
    title: str,
    method_labels: list[str] | None = None,
    method_colors: list[str] | None = None,
    bins: int = 30,
    alpha: float = 0.6,
) -> None:
    """Draw overlaid histograms for one metric across methods onto a single axis.

    Args:
        ax:              Axes to draw on.
        data_per_method: 1D value array per method.
        xlabel:          X-axis label.
        title:           Axes title.
        method_labels:   Legend label per method.
        method_colors:   Bar color per method. Defaults to tab palette.
        bins:            Number of histogram bins.
        alpha:           Bar transparency.
    """
    if method_colors is None:
        method_colors = _DEFAULT_COLORS[:len(data_per_method)]
    labels = method_labels or [None] * len(data_per_method)
    all_data = np.concatenate(data_per_method)
    bin_edges = np.linspace(all_data.min(), all_data.max(), bins + 1)
    for data, label, color in zip(data_per_method, labels, method_colors):
        ax.hist(data, bins=bin_edges, alpha=alpha, label=label, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title)
    if method_labels is not None:
        ax.legend()


def _plot_shaded_band(
    ax: Axes,
    trajectories: list[NDArray[np.float64]],
    label: str,
    color: str,
    ylabel: str,
    title: str,
    x_label: str,
) -> None:
    """Plot mean ± 1 std shaded band for a list of trajectories onto a single axis.

    Args:
        ax:      Axes to draw on.
        trajectories:   Per-seed trajectory lists (variable length).
        label:   Legend label for this method.
        color:   Line and fill color.
        ylabel:  Y-axis label.
        title:   Axes title.
        x_label: X-axis label.
    """
    def _pad_trajectories(trajectories: list[NDArray[np.float64]], length: int) -> NDArray[np.float64]:
        """Pad variable-length trajectories to ``length`` by repeating the last value."""
        return np.array([
            np.concatenate([traj, np.full(length - len(traj), traj[-1])])
            for traj in trajectories
        ])

    max_len = max(len(t) for t in trajectories)
    arr = np.array(_pad_trajectories(trajectories, max_len)).T
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    iters = np.arange(max_len)
    ax.plot(iters, mean, color=color, label=label, linewidth=2)
    ax.fill_between(iters, mean - std, mean + std, alpha=0.25, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _cluster_point_colors(
    labels: NDArray[np.int64],
    cmap,
) -> NDArray[np.float64]:
    """Map cluster labels to RGBA colors.

    Args:
        labels: Per-point cluster label array. Label -1 (noise) maps to black.
        cmap:   Matplotlib colormap used for non-noise labels, cycled modulo 20.

    Returns:
        (N, 4) float64 array of RGBA colors.
    """
    return np.array([
        (0.0, 0.0, 0.0, 1.0) if label == -1 else cmap((label % 20) / 20)
        for label in labels
    ])