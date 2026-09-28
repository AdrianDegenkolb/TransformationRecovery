from __future__ import annotations

import time
from typing import Any
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from algebra_utils import sample_dispersed_rotations
from error_metrics import get_residuals
from matcher import Matcher, Matching, NearestNeighborMatcher
from point_cloud import PointCloud
from transformation import RigidTransformation


@dataclass(frozen=True)
class ICPProgress:
    """What a callback is told about the run so far, at the start of an iteration.

    A callback that only counts iterations needs nothing but ``iteration``; one that
    reacts to how the fit is going needs to see it. The per-iteration histories live in
    local variables inside ``ICP.fit`` rather than on the instance, deliberately — an
    ICP object is reused across seeds and across MultiStartICP trials, so storing run
    state on it would leak between runs. Passing a snapshot keeps the instance stateless
    while still letting a schedule close the loop.

    Attributes:
        iteration:      Zero-based index of the iteration about to run.
        mean_residual:  Mean point-to-point residual after the previous M-step. None on
                        the first iteration, when no M-step has happened yet.
        delta:          Windowed convergence delta after the previous M-step, i.e. how
                        far the last few steps moved the transform. None on the first
                        iteration. Small means the fit has stopped travelling, which is
                        not the same as being correct.
        point_spacing:  Median nearest-neighbour distance of the source cloud. Constant
                        for a run, supplied so a schedule can judge a residual against
                        the scale of the data rather than against an absolute number.
        match_uniqueness: Fraction of the previous E-step's correspondences that landed
                        on distinct target points. A correct alignment pairs points
                        roughly one-to-one and scores near 1; a wrong one collapses many
                        source points onto the same few targets and scores low. Unlike
                        the residual it says something about correspondence *quality*
                        rather than distance, and needs no ground truth. None on the
                        first iteration, and meaningless for soft matching, whose target
                        positions are weighted averages rather than actual points.
    """

    iteration: int
    mean_residual: float | None
    delta: float | None
    point_spacing: float
    match_uniqueness: float | None = None


class ICPCallback(ABC):
    """Base class for hooks called at each ICP iteration."""

    def reset(self) -> None:
        """Discard any state carried over from a previous run.

        Called by ``ICP.fit`` before the first iteration. A callback instance is reused
        across seeds and across MultiStartICP trials, so a schedule that remembers
        anything — a monotone clamp, a phase switch — would otherwise start run *n+1*
        wherever run *n* finished. Stateless callbacks need not override this.
        """

    @abstractmethod
    def on_iteration_start(self, progress: ICPProgress, icp: ICP) -> None:
        """Called at the start of each iteration, before the E-step.

        Args:
            progress: Snapshot of the run so far; see ICPProgress.
            icp:      The running ICP instance (access icp.matcher to update it).
        """
        ...


class SigmaAnnealingCallback(ICPCallback):
    """Exponentially anneals the sigma of a GaussianMatcher over ICP iterations.

    At the start of iteration i, sets:
        matcher.sigma = sigma_init * (sigma_final / sigma_init) ^ (i / anneal_steps)

    This transitions the matcher from a blurred global view (large sigma) to a
    near-hard assignment (small sigma), without any state inside the matcher itself.
    """

    def __init__(
        self,
        sigma_init: float,
        sigma_final: float,
        anneal_steps: int,
    ):
        """
        Args:
            sigma_init:   Starting bandwidth (large → soft/global).
            sigma_final:  Ending bandwidth (small → near-hard).
            anneal_steps: Number of iterations over which to anneal.
                          Typically, this is set equal to ICP max_iter.
        """
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.anneal_steps = anneal_steps

    def sigma_at(self, iteration: int) -> float:
        """Bandwidth the schedule assigns to a given iteration.

        Args:
            iteration: Zero-based ICP iteration index.

        Returns:
            sigma_init at iteration 0, decaying geometrically to sigma_final at
            anneal_steps - 1 and held there afterwards.
        """
        t = min(iteration, self.anneal_steps - 1) / max(self.anneal_steps - 1, 1)
        return float(self.sigma_init * (self.sigma_final / self.sigma_init) ** t)

    def on_iteration_start(self, progress: ICPProgress, icp: ICP) -> None:
        """Update icp.matcher.sigma for the current iteration.

        Args:
            progress: Snapshot of the run so far; only the iteration index is used.
            icp:      The running ICP instance; its matcher must be a GaussianMatcher.
        """
        if hasattr(icp.matcher, "sigma"):
            icp.matcher.sigma = self.sigma_at(progress.iteration)
        else:
            raise AttributeError(
                f"ICP matcher {type(icp.matcher).__name__} has no attribute 'sigma'; "
                "SigmaAnnealingCallback requires a matcher with a 'sigma' attribute."
            )


class BetaAnnealingCallback(ICPCallback):
    """Varies the feature weight ``beta`` over an ICP run.

    Notebook 4.2 measured two opposing pulls on ``beta``, the influence of the feature
    block relative to position in append-mode matching:

    - the **more misaligned** the clouds, the higher it should be, because position
      carries no usable signal until they roughly overlap (optimum 0 at 0 degrees,
      >= 8 at 150);
    - the **more degraded** the observation, the lower it should be, because a
      perturbed descriptor given full authority locks in wrong correspondences with no
      position signal left to overrule it (optimum 5 when clean, 0.25 when severe).

    Degradation is a property of the data and fixed for a run, so it sets a ceiling.
    Misalignment shrinks as ICP converges, so the right value *moves* underneath that
    ceiling — which no constant can follow. Subclasses differ only in what they use to
    decide where along that path the run currently is.

    The schedule is clamped to be non-increasing. Every closed-loop variant risks a
    latch — a high beta produces bad correspondences, which keeps the residual high,
    which keeps beta high — and forbidding increases removes that failure mode at the
    cost of never recovering from an overshoot.
    """

    def __init__(self, beta_start: float, beta_end: float) -> None:
        """
        Args:
            beta_start: Feature weight at the start of a run, when the clouds are still
                        misaligned. Bounded above by descriptor quality, not by the
                        misalignment: overshooting collapses reliability under
                        degradation far more sharply than undershooting does.
            beta_end:   Feature weight once the fit has converged. 0.0 hands the endgame
                        entirely to positions, which is optimal at perfect alignment.

        Raises:
            ValueError: If beta_end exceeds beta_start, which would invert the schedule.
        """
        if beta_end > beta_start:
            raise ValueError(
                f"beta must anneal downward: got start={beta_start}, end={beta_end}."
            )
        self.beta_start = beta_start
        self.beta_end = beta_end
        self._last = beta_start

    def _blend(self, t: float) -> float:
        """Interpolate between the endpoints, with t = 0 at the start and 1 at the end.

        Args:
            t: Progress along the schedule, clamped to [0, 1].

        Returns:
            The interpolated feature weight. Geometric between two positive endpoints,
            linear when the endpoint is 0, which a geometric curve cannot reach.
        """
        t = min(max(t, 0.0), 1.0)
        if self.beta_end <= 0.0:
            return self.beta_start * (1.0 - t)
        return self.beta_start * (self.beta_end / self.beta_start) ** t

    @abstractmethod
    def _target_beta(self, progress: ICPProgress) -> float:
        """Feature weight this schedule wants for the coming iteration.

        Args:
            progress: Snapshot of the run so far.

        Returns:
            The unclamped target; the caller enforces monotonicity.
        """
        ...

    def on_iteration_start(self, progress: ICPProgress, icp: ICP) -> None:
        """Set icp.matcher.beta for the coming iteration.

        Args:
            progress: Snapshot of the run so far.
            icp:      The running ICP instance; its matcher must accept a beta.

        Raises:
            AttributeError: If the matcher has no 'beta' attribute, i.e. it is not
                            doing append-mode feature matching.
        """
        if not hasattr(icp.matcher, "beta"):
            raise AttributeError(
                f"ICP matcher {type(icp.matcher).__name__} has no attribute 'beta'; "
                "beta annealing requires a matcher using append-mode features."
            )
        self._last = min(self._last, self._target_beta(progress))
        icp.matcher.beta = float(self._last)

    def reset(self) -> None:
        """Rewind to the start value so the schedule can be reused for another run."""
        self._last = self.beta_start


class IterationBetaAnnealing(BetaAnnealingCallback):
    """Geometric decay of beta over a fixed number of iterations.

    Open-loop: it never looks at how the fit is going, and simply assumes misalignment
    shrinks monotonically with iteration count. That assumption is exactly what makes it
    safe — there is no feedback path, so it cannot latch — and exactly what makes it
    blunt, since a run that converges early spends its remaining iterations at a beta
    lower than it needed and a run that stalls gets no extra help. It mirrors
    ``SigmaAnnealingCallback`` and is the baseline the closed-loop schedules must beat.
    """

    def __init__(self, beta_start: float, beta_end: float, anneal_steps: int) -> None:
        """
        Args:
            beta_start:   Feature weight at iteration 0.
            beta_end:     Feature weight at and after ``anneal_steps``.
            anneal_steps: Iterations to decay over, typically ICP's max_iter.
        """
        super().__init__(beta_start, beta_end)
        self.anneal_steps = anneal_steps

    def _target_beta(self, progress: ICPProgress) -> float:
        """Interpolate geometrically between the endpoints by iteration index.

        Args:
            progress: Snapshot of the run so far; only the iteration is used.

        Returns:
            The scheduled feature weight.
        """
        return self._blend(
            min(progress.iteration, self.anneal_steps - 1) / max(self.anneal_steps - 1, 1)
        )


class ResidualBetaAnnealing(BetaAnnealingCallback):
    """Sets beta from how far the fit still is from the data's own scale.

    Closed-loop. The residual is the only misalignment signal available at runtime, but
    it is ambiguous on its own: a large value can mean the clouds are far apart *or*
    that the observation is noisy, and those call for opposite changes to beta. Dividing
    by the cloud's median point spacing removes the part of that ambiguity which comes
    from units — a residual of 5 means nothing, a residual of five point spacings means
    the clouds are unambiguously misaligned, while anything at or below one spacing is
    within the range noise alone could explain.

    Above ``start_spacings`` the run is treated as misaligned and gets ``beta_start``;
    below ``end_spacings`` as converged and gets ``beta_end``; in between it interpolates
    geometrically.
    """

    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        start_spacings: float = 5.0,
        end_spacings: float = 1.0,
    ) -> None:
        """
        Args:
            beta_start:     Feature weight while the residual exceeds start_spacings.
            beta_end:       Feature weight once it falls below end_spacings.
            start_spacings: Residual, in point spacings, at or above which the clouds
                            count as fully misaligned.
            end_spacings:   Residual, in point spacings, at or below which they count as
                            aligned. One spacing is the natural floor: below it, points
                            are closer to their neighbours than the grid is wide.

        Raises:
            ValueError: If the thresholds are not ordered, or point spacing is unusable.
        """
        super().__init__(beta_start, beta_end)
        if end_spacings >= start_spacings:
            raise ValueError(
                f"start_spacings must exceed end_spacings, got {start_spacings} "
                f"and {end_spacings}."
            )
        self.start_spacings = start_spacings
        self.end_spacings = end_spacings

    def _target_beta(self, progress: ICPProgress) -> float:
        """Map the scale-normalised residual onto the beta range.

        Args:
            progress: Snapshot of the run so far.

        Returns:
            The scheduled feature weight; beta_start until a residual is available.
        """
        if progress.mean_residual is None or progress.point_spacing <= 0.0:
            return self.beta_start
        spacings = progress.mean_residual / progress.point_spacing
        if spacings >= self.start_spacings:
            return self.beta_start
        if spacings <= self.end_spacings:
            return self.beta_end
        return self._blend((self.start_spacings - spacings)
                           / (self.start_spacings - self.end_spacings))


class TwoPhaseBetaAnnealing(BetaAnnealingCallback):
    """Holds beta high until the fit stops moving, then drops it to refine.

    Not an anneal at all but a switch, and included as the baseline a smooth schedule
    has to justify itself against. Notebook 4.2's measurement is compatible with most of
    the benefit coming from just two regimes — search, then refine — in which case a
    graded schedule is complexity without payoff.

    The switch fires on ICP's own windowed convergence delta rather than on the residual.
    That is the more honest trigger for "has the search finished": delta measures how far
    the last few steps actually moved the transform, so it distinguishes a run that has
    stopped travelling from one that merely has a large residual because the data is
    noisy.
    """

    def __init__(self, beta_start: float, beta_end: float, stall_delta: float = 1e-3) -> None:
        """
        Args:
            beta_start:  Feature weight during the search phase.
            beta_end:    Feature weight after the switch.
            stall_delta: Windowed delta below which the search counts as finished. Well
                         above ICP's own convergence tol, since the point is to switch
                         *before* the run would otherwise stop.
        """
        super().__init__(beta_start, beta_end)
        self.stall_delta = stall_delta
        self.switched_at: int | None = None

    def _target_beta(self, progress: ICPProgress) -> float:
        """Return the search weight until the transform stalls, then the refine weight.

        Args:
            progress: Snapshot of the run so far.

        Returns:
            The scheduled feature weight.
        """
        if self.switched_at is None:
            if progress.delta is not None and progress.delta < self.stall_delta:
                self.switched_at = progress.iteration
            else:
                return self.beta_start
        return self.beta_end

    def reset(self) -> None:
        """Rewind the schedule and forget the recorded switch point."""
        super().reset()
        self.switched_at = None


class DeltaBetaAnnealing(BetaAnnealingCallback):
    """Sets beta from how much the transform is still moving, relative to its own peak.

    Closed-loop on ICP's windowed delta — the distance from the identity of the last ten
    steps composed. That asks "is the fit still travelling", which is a different and
    cleaner question than the residual's "is the fit far off": noise jitters the
    transform slightly, whereas genuine misalignment moves it a long way, so delta is far
    less confounded by the noise level than a residual is.

    The threshold is a fraction of the largest delta seen in this run rather than an
    absolute number, because delta carries units — it sums a dimensionless rotation term
    and a translation in cloud units, so any fixed cutoff would mean something different
    on every cloud. Self-normalising sidesteps that entirely.

    One artifact to be aware of: ``_windowed_delta`` compares against the transform from
    ten steps ago, or against the run's start when fewer than ten have elapsed. For the
    first ten iterations it therefore measures displacement from the origin, not recent
    motion, and reads large regardless. That is harmless here — it keeps beta at its
    start value through exactly the early iterations where the search is happening.
    """

    def __init__(self, beta_start: float, beta_end: float, end_fraction: float = 0.05) -> None:
        """
        Args:
            beta_start:   Feature weight while the transform is still moving freely.
            beta_end:     Feature weight once motion has all but stopped.
            end_fraction: Fraction of this run's peak delta at or below which the search
                          counts as finished.

        Raises:
            ValueError: If end_fraction is not in (0, 1].
        """
        super().__init__(beta_start, beta_end)
        if not 0.0 < end_fraction <= 1.0:
            raise ValueError(f"end_fraction must lie in (0, 1], got {end_fraction}.")
        self.end_fraction = end_fraction
        self._peak_delta = 0.0

    def _target_beta(self, progress: ICPProgress) -> float:
        """Map the delta, as a fraction of its peak, onto the beta range.

        Args:
            progress: Snapshot of the run so far.

        Returns:
            The scheduled feature weight; beta_start until a delta is available.
        """
        if progress.delta is None:
            return self.beta_start
        self._peak_delta = max(self._peak_delta, progress.delta)
        if self._peak_delta <= 0.0:
            return self.beta_end
        fraction = progress.delta / self._peak_delta
        if fraction <= self.end_fraction:
            return self.beta_end
        # Log-spaced, because delta falls by orders of magnitude rather than linearly.
        span = np.log(1.0) - np.log(self.end_fraction)
        return self._blend((np.log(1.0) - np.log(fraction)) / span)

    def reset(self) -> None:
        """Rewind the schedule and forget the peak delta of the previous run."""
        super().reset()
        self._peak_delta = 0.0


class ConsistencyBetaAnnealing(BetaAnnealingCallback):
    """Sets beta from how one-to-one the current correspondences are.

    Closed-loop on correspondence *quality* rather than on distance. A correct alignment
    pairs points roughly one-to-one, so almost every source point claims a different
    target; a wrong one collapses many source points onto the same few targets, because
    nearest-neighbour matching sends whole regions to whichever handful of points happens
    to lie nearest. The fraction of distinct targets therefore tracks whether the
    correspondences are trustworthy — which is what beta should respond to — and needs no
    ground truth to compute.

    It is the only signal here that is not a proxy for distance. Both the residual and
    the delta answer geometric questions and leave "are these correspondences any good"
    to be inferred; this measures it.

    Only meaningful with hard matching. A soft matcher's target positions are weighted
    averages of several neighbours, so they are distinct almost by construction and the
    fraction carries no information.
    """

    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        start_uniqueness: float = 0.5,
        end_uniqueness: float = 0.9,
    ) -> None:
        """
        Args:
            beta_start:       Feature weight while correspondences are still degenerate.
            beta_end:         Feature weight once they are near one-to-one.
            start_uniqueness: Distinct-target fraction at or below which the run counts
                              as still searching.
            end_uniqueness:   Distinct-target fraction at or above which it counts as
                              converged.

        Raises:
            ValueError: If the thresholds are not ordered.
        """
        super().__init__(beta_start, beta_end)
        if end_uniqueness <= start_uniqueness:
            raise ValueError(
                f"end_uniqueness must exceed start_uniqueness, got {end_uniqueness} "
                f"and {start_uniqueness}."
            )
        self.start_uniqueness = start_uniqueness
        self.end_uniqueness = end_uniqueness

    def _target_beta(self, progress: ICPProgress) -> float:
        """Map the distinct-target fraction onto the beta range.

        Args:
            progress: Snapshot of the run so far.

        Returns:
            The scheduled feature weight; beta_start until a matching has happened.
        """
        if progress.match_uniqueness is None:
            return self.beta_start
        uniqueness = progress.match_uniqueness
        if uniqueness <= self.start_uniqueness:
            return self.beta_start
        if uniqueness >= self.end_uniqueness:
            return self.beta_end
        return self._blend((uniqueness - self.start_uniqueness)
                           / (self.end_uniqueness - self.start_uniqueness))


@dataclass
class ICPResult:
    """Result of an ICP run.

    Attributes:
        transformation:      Accumulated rigid transformation mapping source onto target.
        n_iterations:        Number of EM iterations performed.
        duration_s:          Duration in seconds until convergence or n_iterations reached.
        converged:           Whether the algorithm converged before max_iter.
        mean_residuals:      Mean point-to-point residual after each M-step.
        cloud_history:       Source cloud state at the start of each iteration.
                             Empty if the ICP instance was created with record_history=False.
        matching_history:    Matching from the E-step of each iteration.
                             Empty if the ICP instance was created with record_history=False.
        transform_history:   Accumulated transformation after each M-step. One entry
                             per iteration, parallel to mean_residuals/deltas; any
                             init_align_centroids pre-alignment is folded into these
                             entries rather than recorded as a separate step.
        deltas:              Per step delta. ICP is considered converged if
                             delta = ||last_10_transformation.R - I||_F + ||last_10_transformation.t||_2 < tolerance
    """

    transformation: RigidTransformation
    n_iterations: int
    duration_s: float
    converged: bool
    mean_residuals: list[float] = field(default_factory=list)
    cloud_history: list[PointCloud] = field(default_factory=list)
    matching_history: list[Matching] = field(default_factory=list)
    transform_history: list[RigidTransformation] = field(default_factory=list)
    deltas: list[float] = field(default_factory=list)

    def __repr__(self):
        rows: list[tuple[str, int | RigidTransformation]] = [
            ("Converged",  self.converged),
            ("Iterations", self.n_iterations),
            ("Recovered",  self.transformation),
        ]
        return tabulate(rows, tablefmt="rounded_outline")

@dataclass
class MultiStartICPResult:
    """Result of a multi-start ICP run.

    Attributes:
        best:                   ICPResult with the lowest final residual.
        best_initial_rotation:  The SO(3) seed that produced the best result.
        all_results:            ICPResult for every starting rotation.
        all_initial_rotations:  All sampled starting rotations (3, 3) each.
        duration_s:             Duration in seconds until all workers convergence or reach n_iterations.
    """

    best: ICPResult
    best_initial_rotation: NDArray[np.float64]
    all_results: list[ICPResult]
    all_initial_rotations: list[NDArray[np.float64]]
    duration_s: float

    def __repr__(self) -> str:
        rows: list[tuple[str, int | RigidTransformation]] = [
            ("Starts",                  len(self.all_results)),
            ("Converged",               sum(r.converged for r in self.all_results)),
            ("Best Transformation",     self.best.transformation),
        ]
        return tabulate(rows, tablefmt="rounded_outline")

    @property
    def individual_durations_summed(self) -> float:
        return sum([result.duration_s for result in self.all_results])

    @property
    def num_workers(self) -> int:
        return len(self.all_results)

    @property
    def cpu_efficiency(self) -> float:
        return self.individual_durations_summed / self.duration_s * self.num_workers

    def __getattr__(self, name: str):
        # Forward attribute access to the best result. Look `best` up via
        # __dict__ directly (not self.best) to avoid infinite recursion: pickle
        # probes for dunder methods like __setstate__ on a bare instance before
        # `best` is set, and a plain `self.best` would re-enter __getattr__.
        best = self.__dict__.get("best")
        if best is not None and hasattr(best, name):
            return getattr(best, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def _windowed_delta(
    transform_history: list[RigidTransformation],
    accumulated: RigidTransformation,
    window: int = 10,
) -> float:
    """Convergence delta: how far the last `window` steps' composition is from identity.

    Args:
        transform_history: Accumulated transformation after each prior M-step.
        accumulated:        Current accumulated transformation.
        window:              Number of trailing steps to compose over. If fewer
                             than `window` steps have elapsed, uses `accumulated`
                             directly (i.e. compares against the identity from
                             the very start of the run).

    Returns:
        ||R - I||_F + ||t||_2, where R, t come from composing `accumulated`
        with the inverse of the transformation from `window` steps ago.
    """
    if len(transform_history) >= window:
        ref = transform_history[-window]
        recent = accumulated.compose(ref.inverse())
    else:
        recent = accumulated
    return float(np.linalg.norm(recent.R - np.eye(3), ord="fro") + np.linalg.norm(recent.t))


class ICP:
    """EM algorithm for rigid point cloud registration without known correspondences.

    E-step: establish point correspondences via a Matcher.
    M-step: fit a RigidTransformation via weighted SVD Procrustes.
    Callbacks: called at the start of each iteration (e.g. for sigma annealing).

    Convergence is declared when the composed transformation of the last 10 steps is near identity:
        delta = ||R_step - I||_F + ||t_step|| < tol
    """

    def __init__(
        self,
        init_align_centroids: bool = True,
        matcher: Matcher | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
        callbacks: list[ICPCallback] | None = None,
        record_history: bool = True,
    ):
        """
        Args:
            init_align_centroids: If True, initialize the transformation by aligning the 
                            centroids of the source and target point clouds.
            matcher:        Correspondence algorithm for the E-step.
                            Defaults to NearestNeighborMatcher.
            
            max_iter:       Maximum number of EM iterations.
            tol:            Convergence threshold on ||R_step - I||_F + ||t_step||.
            callbacks:      Optional list of ICPCallback instances called before
                            each E-step (e.g. SigmaAnnealingCallback).
            record_history: If False, skip recording cloud_history and
                            matching_history (both O(n_points) per iteration).
                            Set to False for large sweeps that only need the
                            final transformation, to avoid retaining a full
                            point cloud + correspondence set per iteration per
                            trial. mean_residuals/transform_history/deltas are
                            always recorded (cheap, and needed for convergence).
            verbose:        Show a progress bar if True.
        """
        self.matcher = matcher or NearestNeighborMatcher()
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.callbacks = callbacks or []
        self.record_history = record_history
        self.init_align_centroids = init_align_centroids

    def fit(self, source: PointCloud, target: PointCloud) -> ICPResult:
        """Run ICP to find the rigid transformation mapping source onto target.

        Args:
            source: Source PointCloud (N, 3).
            target: Target PointCloud (M, 3).

        Returns:
            ICPResult with the accumulated transformation, convergence info,
            and per-iteration history.
        """
        mean_residuals: list[float] = []
        cloud_history: list[PointCloud] = []
        matching_history: list[Matching] = []
        transform_history: list[RigidTransformation] = []
        deltas: list[float] = []

        current = source
        if self.init_align_centroids:
            # Pre-align the centroids of the source and target point clouds
            accumulated = self._fit_translation_only(current, target)
            current = accumulated.apply(current)
        else:
            accumulated = RigidTransformation.identity()

        # precompute feature vectors for source and target clouds, 
        # if the matcher is invariant to transformations.
        self.matcher.prepare(source, target)

        point_spacing = source.median_spacing if self.callbacks else 0.0
        match_uniqueness: float | None = None
        for cb in self.callbacks:
            cb.reset()
        t0 = time.perf_counter()
        pbar = tqdm(range(self.max_iter), desc="ICP", disable=not self.verbose)
        for i in pbar:
            if self.callbacks:
                progress = ICPProgress(
                    iteration=i,
                    mean_residual=mean_residuals[-1] if mean_residuals else None,
                    delta=deltas[-1] if deltas else None,
                    point_spacing=point_spacing,
                    match_uniqueness=match_uniqueness,
                )
                for cb in self.callbacks:
                    cb.on_iteration_start(progress, self)

            # E step: find correspondences between the current source and target clouds
            matching = self.matcher.match(current, target)
            src_pc = PointCloud(matching.source_points)
            tgt_pc = PointCloud(matching.target_positions)
            # M step: fit a rigid transformation from the matched source to target points
            transformation = RigidTransformation.fit(src_pc, tgt_pc, weights=matching.weights)

            accumulated = transformation.compose(accumulated)
            residual = float(get_residuals(matching, transformation.apply(src_pc)).mean())
            delta = _windowed_delta(transform_history, accumulated)
            pbar.set_postfix(residual=f"{residual:.4f}")
            if self.record_history:
                cloud_history.append(current)
                matching_history.append(matching)
            if self.callbacks:
                # Distinct targets claimed, as a fraction of source points. Cheap: no
                # extra matching, just a uniqueness count over the positions already
                # returned by the E-step.
                distinct = len(np.unique(matching.target_positions, axis=0))
                match_uniqueness = distinct / max(len(matching.target_positions), 1)
            mean_residuals.append(residual)
            transform_history.append(accumulated)
            deltas.append(delta)
            current = transformation.apply(current)

            if delta < self.tol:
                pbar.set_description("ICP converged")
                return ICPResult(
                    transformation=accumulated, n_iterations=i + 1, converged=True,
                    mean_residuals=mean_residuals, cloud_history=cloud_history, matching_history=matching_history,
                    transform_history=transform_history, duration_s=time.perf_counter() - t0, deltas=deltas
                )

        pbar.set_description("ICP did not converge")
        return ICPResult(
            transformation=accumulated, n_iterations=self.max_iter, converged=False,
            mean_residuals=mean_residuals, cloud_history=cloud_history, matching_history=matching_history,
            transform_history=transform_history, duration_s=time.perf_counter() - t0, deltas=deltas
        )

    def _fit_translation_only(self, source: PointCloud, target: PointCloud) -> RigidTransformation:
        """
        Fit a translation-only transformation from source to target computed as the difference between the centroids of the source and target point clouds.
        The hypothesis is that starting with aligned centroids will help the ICP algorithm converge faster and avoid local minima.

        Args:
            source: Source PointCloud (N, 3).
            target: Target PointCloud (M, 3).
        Returns:
            RigidTransformation with identity rotation and translation equal to the difference 
            between the centroids of the source and target point clouds.
        """
        source_centroid = source.points.mean(axis=0)
        target_centroid = target.points.mean(axis=0)
        translation = target_centroid - source_centroid
        return RigidTransformation(R=np.eye(3), t=translation)
    
    def to_multi_start(
        self, 
        rotation_sampler: Callable[[int, np.random.Generator], list[NDArray[np.float64]]] = sample_dispersed_rotations,
        n_starts: int = 20,
        n_jobs: int = -1, 
        residual_threshold: float = 1e-6, 
        seed: int = 42, 
        verbose: bool = True
        ) -> MultiStartICP:
        """Wrap this ICP instance in a MultiStartICP with the given parameters.

        Args:
            rotation_sampler:    Callable(n, rng) -> list of n (3, 3) SO(3) rotations
                                                used to seed the starts. Defaults to greedy
                                                farthest-point sampling; pass e.g.
                                                sample_uniform_rotations for plain i.i.d. sampling.
            n_starts:            Number of starting rotations to try.
            residual_threshold:  Mean residual below which a converged trial triggers early stopping of remaining trials.
            n_jobs:              Worker processes. -1 uses os.cpu_count().
            seed:                Optional random seed for reproducible rotation sampling.
            verbose:             Show a progress bar if True.
        """
        return MultiStartICP(
            icp=self,
            rotation_sampler=rotation_sampler,
            residual_threshold=residual_threshold,
            n_starts=n_starts,
            n_jobs=n_jobs,
            seed=seed,
            verbose=verbose,
        )


class MultiStartICP:
    """Runs ICP from multiple dispersed starting rotations and returns the best result.

    Wraps an existing ICP instance. For each start, the source cloud is
    pre-rotated by one of a set of SO(3) rotations chosen via rotation_sampler
    (greedy farthest-point selection by default, see sample_dispersed_rotations)
    before running ICP, so the starts are spread across rotation space rather
    than left to chance. The recovered transformations are composed with the
    initial rotation so that all results refer to the original (un-rotated)
    source.

    Trials are executed in parallel via ProcessPoolExecutor.
    """

    def __init__(
        self,
        icp: ICP,
        rotation_sampler: Callable[[int, np.random.Generator], list[NDArray[np.float64]]] = sample_dispersed_rotations,
        residual_threshold: float = 1e-6,
        n_starts: int = 20,
        n_jobs: int = -1,
        seed: int = 42,
        verbose: bool = True,
    ):
        """
        Args:
            icp:                 Configured ICP instance reused across all trials.
            rotation_sampler:    Callable(n, rng) -> list of n (3, 3) SO(3) rotations
                                 used to seed the starts. Defaults to greedy
                                 farthest-point sampling; pass e.g.
                                 sample_uniform_rotations for plain i.i.d. sampling.
            n_starts:            Number of starting rotations to try.
            residual_threshold:  Mean residual below which a converged trial
                                 triggers early stopping of remaining trials.
            n_jobs:              Worker processes. -1 uses os.cpu_count().
            seed:                Optional random seed for reproducible rotation sampling.
            verbose:             Show a progress bar if True.
        """
        self.icp = icp
        self.n_starts = n_starts
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose
        self.residual_threshold = residual_threshold
        self.rotation_sampler = rotation_sampler

    def fit(self, source: PointCloud, target: PointCloud) -> MultiStartICPResult:
        """Run ICP from n_starts rotations (via rotation_sampler) and return the best result.

        Args:
            source: Source PointCloud (N, 3).
            target: Target PointCloud (M, 3).

        Returns:
            MultiICPResult containing the best ICPResult and all trial results.
        """
        rng = np.random.default_rng(self.seed)
        rotations = self.rotation_sampler(self.n_starts, rng)

        all_results: list[ICPResult] = []
        all_rotations: list[NDArray[np.float64]] = []

        t0 = time.perf_counter()
        pbar = tqdm(total=self.n_starts, desc=f"Testing {self.n_starts} starting configurations", disable=not self.verbose)
        if self.n_jobs == 1:
            # Run in-process rather than via a single-worker ProcessPoolExecutor: some
            # wrapped fit() implementations (e.g. probreg's CPD, which pulls in open3d)
            # initialize native thread pools at import time, and forking such a process
            # (ProcessPoolExecutor's default start method on Linux) deadlocks the child
            # the moment it touches a lock held by a thread that didn't survive the fork.
            for R in rotations:
                result, R_init = self._run_single(self.icp, source, target, R)
                pbar.update(1)
                all_results.append(result)
                all_rotations.append(R_init)
                if result.converged and result.mean_residuals[-1] < self.residual_threshold:
                    break
        else:
            max_workers = self.n_jobs if self.n_jobs > 0 else None
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self._run_single, self.icp, source, target, R): R
                    for R in rotations
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    result, R_init = future.result()
                    all_results.append(result)
                    all_rotations.append(R_init)
                    if result.converged and result.mean_residuals[-1] < self.residual_threshold:
                        pool.shutdown(cancel_futures=True)
                        break

        pbar.close()
        best_idx = int(np.argmin([r.mean_residuals[-1] for r in all_results]))
        return MultiStartICPResult(
            best=all_results[best_idx],
            best_initial_rotation=all_rotations[best_idx],
            all_results=all_results,
            all_initial_rotations=all_rotations,
            duration_s=time.perf_counter() - t0
        )

    @staticmethod
    def _run_single(
            icp: ICP,
            source: PointCloud,
            target: PointCloud,
            R_init: NDArray[np.float64],
    ) -> tuple[ICPResult, NDArray[np.float64]]:
        """Run one ICP trial from a pre-rotation R_init and compose the result.

        Args:
            icp:    Configured ICP instance.
            source: Original source PointCloud.
            target: Target PointCloud.
            R_init: (3, 3) initial rotation applied to source before ICP.

        Returns:
            Tuple of (ICPResult with composed transformation, R_init).
        """
        init_tf = RigidTransformation(R_init, np.zeros(3))
        rotated_source = init_tf.apply(source)
        result = icp.fit(rotated_source, target)
        result.transformation = result.transformation.compose(init_tf)
        result.transform_history = [T.compose(init_tf) for T in result.transform_history]
        return result, R_init
