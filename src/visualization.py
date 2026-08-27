"""Visualisation utilities for point cloud registration experiments.

Provides four static-method classes:
    PointCloudVisualizer   — raw point cloud scatter/projection plots.
    ErrorMetricsVisualizer — convergence curves, seed box plots, parameter sweeps.
    ResidualVisualizer     — histograms, field heatmaps, rotation bases.
    OptimizationVisualizer — HPO Pareto front and feature importance.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from matplotlib.axes import Axes

from algebra_utils import rotation_angle
from icp import ICPResult
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
        idx: np.ndarray | None = None,
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
        idx: np.ndarray | None = None,
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

        axes[0].legend(markerscale=2, loc='upper left', fontsize=7)

        PointCloudVisualizer.plot_xy(
            axes[-1],
            [final_cloud, target],
            ['P (final)', 'Q (target)'],
            [source_color, target_color],
            idx=idx,
        )
        axes[-1].set_title('Final (converged)')


class ErrorMetricsVisualizer:
    """Visualisation methods for ICP error metrics and performance distributions."""

    @staticmethod
    def plot_convergence(
            axes: Sequence[Axes],
            rot_errors: list[Sequence[float]],
            t_errors: list[Sequence[float]],
            mean_residuals: list[Sequence[float]],
            colors: list[str] | None = None,
            x_label: str = 'Iteration') -> None:
        """Plot per-iteration rotation error, translation error, and mean residuals.

        Each argument is a list of sequences — one per method — so that multiple
        methods can be overlaid on the same axes. For a single method, wrap the
        sequence in a list, e.g. ``rot_errors=[my_rot_errors]``.

        Args:
            axes:          Sequence of 3 Axes (rotation, translation, residuals).
            rot_errors:    Rotation error (°) per iteration, per method.
            t_errors:      Translation error per iteration, per method.
            mean_residuals: Mean point residual per iteration, per method.
            colors:        Line color per method. Defaults to the tab palette.
            x_label:       Shared x-axis label.
        """
        n_methods = len(rot_errors)
        if n_methods == 1:
            # Single method: each subplot gets its own color (rot, t, residuals).
            if colors is None:
                colors = _DEFAULT_COLORS[:3]
            rot, t, res = rot_errors[0], t_errors[0], mean_residuals[0]
            axes[0].plot(rot, color=colors[0], marker='o', ms=4)
            axes[1].plot(t,   color=colors[1], marker='o', ms=4)
            axes[2].plot(res, color=colors[2], marker='o', ms=4)
        else:
            # Multi-method: each method gets its own color, consistent across subplots.
            if colors is None:
                colors = _DEFAULT_COLORS[:n_methods]
            for rot, t, res, color in zip(rot_errors, t_errors, mean_residuals, colors):
                axes[0].plot(rot, color=color, marker='o', ms=4)
                axes[1].plot(t,   color=color, marker='o', ms=4)
                axes[2].plot(res, color=color, marker='o', ms=4)

        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel('Rotation error (°)')
        axes[0].set_title('Rotation error vs. ground truth')

        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel('Translation error')
        axes[1].set_title('Translation error vs. ground truth')

        axes[2].set_xlabel(x_label)
        axes[2].set_ylabel('Mean residual')
        axes[2].set_title('Mean residuals')

    @staticmethod
    def plot_convergence_shaded(
        axes: Sequence[Axes],
        results_per_method: Iterable[list[ICPResult]],
        ground_truths_per_method: Iterable[list[RigidTransformation]],
        method_labels: Iterable[str],
        colors: list[str] | None = None,
        x_label: str = 'Iteration',
    ) -> None:
        """Plot per-iteration error trajectories as mean ± 1 std shaded bands.

        Rotation and translation errors are derived from each ICPResult's
        ``transform_history`` against the corresponding ground-truth transformation.
        Variable-length trajectories (from early convergence) are padded by repeating
        the final value.

        Args:
            axes:                     Sequence of 3 Axes: rotation error, translation error, residuals.
            results_per_method:       Iterable of per-seed ICPResult lists, one per method. [[seed 1, ...], [seed 1, ...], ...]
            ground_truths_per_method: Iterable of per-seed ground-truth RigidTransformation lists,
                                      matching the structure of ``results_per_method``.
            method_labels:            Legend label per method.
            colors:                   Line/fill color per method. Defaults to tab palette.
            x_label:                  Shared x-axis label.
        """
        methods = list(zip(results_per_method, ground_truths_per_method, method_labels))
        if colors is None:
            colors = _DEFAULT_COLORS[:len(methods)]

        _YLABELS = ['Rotation error (°)', 'Translation error', 'Mean residual']
        _TITLES  = ['Rotation error vs. ground truth', 'Translation error vs. ground truth', 'Mean residuals']

        for (results, ground_truths, label), color in zip(methods, colors):
            rot_trajs, t_trajs, res_trajs = [], [], []
            for result, T_gt in zip(results, ground_truths):
                rot_trajs.append([rotation_angle_error(T_gt.R, T.R) for T in result.transform_history])
                t_trajs.append([float(np.linalg.norm(T_gt.t - T.t)) for T in result.transform_history])
                res_trajs.append(result.mean_residuals)

            max_len = max(len(t) for t in rot_trajs)

            def _pad(trajs: list[list[float]], length: int) -> np.ndarray:
                """Pad trajectories to ``length`` by repeating the last value; returns (n_seeds, length)."""
                return np.array([t + [t[-1]] * (length - len(t)) for t in trajs])

            arrays = [
                _pad(rot_trajs, max_len).T,  # (max_len, n_seeds)
                _pad(t_trajs,   max_len).T,
                _pad(res_trajs, max_len).T,
            ]

            iters = np.arange(max_len)
            for ax, arr in zip(axes, arrays):
                mean = arr.mean(axis=1)
                std  = arr.std(axis=1)
                ax.plot(iters, mean, color=color, label=label, linewidth=2)
                ax.fill_between(iters, mean - std, mean + std, alpha=0.25, color=color)

        for ax, ylabel, title in zip(axes, _YLABELS, _TITLES):
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()

    @staticmethod
    def plot_seed_boxplots(
        axes: Sequence[Axes],
        data_per_metric: list[list[np.ndarray]],
        y_labels: list[str],
        titles: list[str],
        method_labels: list[str] | None = None,
        method_colors: list[str] | None = None,
    ) -> None:
        """Box-plots of error distributions across random seeds for one or more methods.

        Args:
            axes:            Sequence of Axes, one per metric.
            data_per_metric: ``data_per_metric[i][j]`` is the 1D array of values
                             for metric *i*, method *j*.
            y_labels:        Y-axis label per metric.
            titles:          Subplot title per metric.
            method_labels:   X-tick label per method. ``None`` hides x-ticks
                             (use for single-method plots).
            method_colors:   Box face color per method. Defaults to tab palette.
        """
        n_methods = len(data_per_metric[0])
        if method_colors is None:
            method_colors = _DEFAULT_COLORS[:n_methods]

        for ax, data_list, ylabel, title in zip(axes, data_per_metric, y_labels, titles):
            bp_kwargs: dict[str, object] = dict(**_BOX_STYLE)
            if method_labels is not None:
                bp_kwargs['tick_labels'] = method_labels
            bp = ax.boxplot(data_list, vert=True, **bp_kwargs)
            for patch, color in zip(bp['boxes'], method_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if method_labels is None:
                ax.set_xticks([])
            elif n_methods > 1:
                ax.tick_params(axis='x', rotation=10)

    @staticmethod
    def plot_parameter_sweep(
        axes: Sequence[Axes],
        x: Sequence[float | int],
        data_per_metric: list[np.ndarray],
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
        residuals_per_method: list[np.ndarray],
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
        points: np.ndarray,
        magnitudes: np.ndarray,
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
        rotations: list[np.ndarray],
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


class OptimizationVisualizer:
    """Visualisation methods for multi-objective hyperparameter optimization results."""

    @staticmethod
    def plot_pareto_front(
        ax: Axes,
        x_values: np.ndarray,
        y_values: np.ndarray,
        group_ids: list[str],
        pareto_mask: np.ndarray,
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
