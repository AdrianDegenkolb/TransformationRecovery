"""Adapters wrapping probreg's rigid registration algorithms behind the ICP.fit interface.

Requires the `probreg` package, which is not installable under this project's
main Python 3.13 environment (its dependency `open3d` has no cp313 wheels).
Run this module under the separate `.venv-probreg` (Python 3.12) environment,
e.g. via the "Python 3.12 (probreg)" Jupyter kernel.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from probreg.cpd import registration_cpd
from probreg.filterreg import registration_filterreg
from probreg.gmmtree import registration_gmmtree

from error_metrics import nearest_neighbor_residuals
from icp import ICPResult, _windowed_delta
from point_cloud import PointCloud
from transformation import RigidTransformation


def _run_probreg_registration(
    registration_fn: Callable[..., object],
    source: PointCloud,
    target: PointCloud,
    maxiter: int,
    **kwargs: object,
) -> ICPResult:
    """Runs a probreg `registration_*` function and packages its per-iteration
    callback history into an ICPResult.

    All of probreg's rigid registration_* entry points (CPD, FilterReg, GMMTree)
    share the same contract: a `callbacks` kwarg of `callback(transformation)`
    called once per EM iteration with the transformation accumulated from the
    original source, exposing `.rot`/`.t` — so one shared runner covers all of them.

    Args:
        registration_fn: A probreg `registration_*` function, e.g. `registration_cpd`.
        source:          Source PointCloud (N, 3).
        target:          Target PointCloud (M, 3).
        maxiter:         Maximum number of EM iterations, forwarded to registration_fn.
        **kwargs:        Extra keyword arguments forwarded to registration_fn
                         (e.g. `w`, `tol`, `tf_type_name`).

    Returns:
        ICPResult with the accumulated transformation and per-iteration history,
        for direct comparison against ICP/MultiStartICP results.
    """
    transform_history: list[RigidTransformation] = []

    def _record(tf: object) -> None:
        transform_history.append(RigidTransformation(np.asarray(tf.rot), np.asarray(tf.t)))

    t0 = time.perf_counter()
    registration_fn(source.points, target.points, maxiter=maxiter, callbacks=[_record], **kwargs)
    duration_s = time.perf_counter() - t0

    mean_residuals = [
        float(nearest_neighbor_residuals(T.apply(source), target).mean())
        for T in transform_history
    ]
    deltas = [
        _windowed_delta(transform_history[:i], transform_history[i])
        for i in range(len(transform_history))
    ]

    return ICPResult(
        transformation=transform_history[-1],
        n_iterations=len(transform_history),
        duration_s=duration_s,
        converged=len(transform_history) < maxiter,
        mean_residuals=mean_residuals,
        transform_history=transform_history,
        deltas=deltas,
    )


class ProbregCPD:
    """Rigid Coherent Point Drift (probreg) exposed like ICP.fit, for direct comparison.

    Produces an ICPResult so it can be dropped into fit_multi_seed, the
    visualization/error_metrics helpers, and MultiSeedSyntheticICPResult
    unchanged, alongside ICP/MultiStartICP results.
    """

    def __init__(self, w: float = 0.0, maxiter: int = 100, tol: float = 1e-4, verbose: bool = False):
        """
        Args:
            w:        Weight of the uniform (outlier) distribution component, in [0, 1).
            maxiter:  Maximum number of EM iterations.
            tol:      Termination tolerance on the change in CPD's likelihood criterion.
            verbose:  Unused internally; present only so experiment_runner._quiet()
                      (which toggles icp.verbose) works unmodified.
        """
        self.w = w
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose

    def fit(self, source: PointCloud, target: PointCloud) -> ICPResult:
        """Run rigid CPD to find the transformation mapping source onto target."""
        return _run_probreg_registration(
            registration_cpd, source, target, maxiter=self.maxiter,
            tf_type_name="rigid", w=self.w, tol=self.tol, update_scale=False,
        )


class ProbregFilterReg:
    """Rigid FilterReg (probreg) exposed like ICP.fit, for direct comparison.

    Same rigid Gaussian-mixture model as CPD, but replaces CPD's exact O(N*M)
    E-step with a permutohedral-lattice-filtered approximation — same registration
    quality target, much faster per iteration on larger point clouds.
    """

    def __init__(self, w: float = 0.0, maxiter: int = 100, tol: float = 1e-4, verbose: bool = False):
        """
        Args:
            w:        Weight of the uniform (outlier) distribution component, in [0, 1).
            maxiter:  Maximum number of EM iterations.
            tol:      Termination tolerance on the change in FilterReg's likelihood criterion.
            verbose:  Unused internally; present only so experiment_runner._quiet()
                      (which toggles icp.verbose) works unmodified.
        """
        self.w = w
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose

    def fit(self, source: PointCloud, target: PointCloud) -> ICPResult:
        """Run rigid FilterReg to find the transformation mapping source onto target."""
        return _run_probreg_registration(
            registration_filterreg, source, target, maxiter=self.maxiter,
            w=self.w, tol=self.tol, objective_type="pt2pt",
        )


class ProbregGMMTree:
    """Rigid GMMTree (probreg) exposed like ICP.fit, for direct comparison.

    Represents the source as a hierarchical (coarse-to-fine) Gaussian mixture
    tree, aligning coarse structure first before refining — unlike CPD/FilterReg's
    flat mixture, this may be less prone to large-rotation local optima.
    """

    def __init__(self, maxiter: int = 20, tol: float = 1e-4, verbose: bool = False):
        """
        Args:
            maxiter:  Maximum number of EM iterations.
            tol:      Termination tolerance on the change in GMMTree's likelihood criterion.
            verbose:  Unused internally; present only so experiment_runner._quiet()
                      (which toggles icp.verbose) works unmodified.
        """
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose

    def fit(self, source: PointCloud, target: PointCloud) -> ICPResult:
        """Run rigid GMMTree to find the transformation mapping source onto target."""
        return _run_probreg_registration(
            registration_gmmtree, source, target, maxiter=self.maxiter, tol=self.tol,
        )
