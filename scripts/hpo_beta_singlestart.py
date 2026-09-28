#!/usr/bin/env python3
"""
HPO study: is the optimal `beta` high when ICP runs from a single start?

Background
----------
Notebook 5 tuned the full ICP search space and selected beta = 0.005395 — i.e. it
almost entirely disabled the feature contribution to the joint (position, feature)
KDTree. That result is suspect, because it was obtained with `n_starts=20`: with
20 dispersed starting rotations, MultiStartICP can brute-force past a poor `beta`
and the search never feels the penalty.

A separate single-start experiment (5 seeds, muscle-fiber and clustered, no
dropout) found the opposite: rotation error < 5° on 10/10 runs at beta >= 3, but
only 2/10 at beta = 0.005. High beta freezes correspondences across iterations
(feature distances do not change under rigid motion), which turns matching into
feature-based correspondence and makes it immune to the spatial local minima that
trap standard ICP.

Those two results disagree, but they differ in more than `n_starts`. Notebook 5
also ran with `dropout_prob=0.2`, and dropout perturbs each point's k-NN
neighborhood and therefore its features — degrading exactly the invariance a high
`beta` relies on. Dropout is a competing explanation for the low tuned beta.

This script therefore runs one study per dropout probability, identical in every
other respect, so the result can be attributed rather than merely observed.

Search space (see hpo.build_single_start_nn_icp_factory):
    feature_extractor  categorical ['geometric', 'robust']
    fe_k               int   [2, 50]
    beta               float [0.0, 10.0]

Fixed: hard matching, no multi-start, no trimming.

Usage:
  uv run python scripts/hpo_beta_singlestart.py
  uv run python scripts/hpo_beta_singlestart.py --n-trials 300 --style clustered
"""
from __future__ import annotations

import argparse
import sys

import optuna
from optuna.trial import FixedTrial
from tabulate import tabulate

sys.path.insert(0, 'src')

from hpo import build_single_start_nn_icp_factory, evaluate_icp, make_beta_sweep_objective
from synthetic import CloudStyle, PointCloudObserver

# ---------------------------------------------------------------------------
# Configuration — mirrors notebook 7 so results are directly comparable
# ---------------------------------------------------------------------------

N_SEEDS               = 30
N_HOLDOUT_SEEDS       = 15
MAX_ITER              = 400
TOL                   = 0.1
GEN_KWARGS            = dict(n=500, t_scale=8.0)
NOISE_STD             = 0.1
OBSERVER_SEED         = 42
RELIABILITY_THRESHOLD = 0.8

TUNING_SEEDS  = list(range(N_SEEDS))
HOLDOUT_SEEDS = list(range(10_000, 10_000 + N_HOLDOUT_SEEDS))

DEFAULT_DROPOUTS = [0.0, 0.2]
DEFAULT_STORAGE  = 'sqlite:///results/icp_hpo.db'


def study_name(style: CloudStyle, dropout_prob: float) -> str:
    """Build the Optuna study name for one (style, dropout) combination.

    Deliberately distinct from notebook 7's `icp_hpo_{style}_true_residual`: that
    study holds trials from the full 17-parameter space, and resuming it under this
    reduced space would have TPE model a mixture of two different search spaces.

    Args:
        style:        Cloud geometry the study is run for.
        dropout_prob: Dropout probability the study is run at.

    Returns:
        Study name, unique per (style, dropout).
    """
    return f'icp_hpo_{style}_beta_singlestart_dropout{dropout_prob:g}'


def build_observer(dropout_prob: float) -> PointCloudObserver:
    """Build the observer for one study, at this script's fixed noise level.

    Args:
        dropout_prob: Dropout probability the study is run at.

    Returns:
        A PointCloudObserver combining `dropout_prob` with NOISE_STD, seeded so
        tuning and holdout runs observe reproducibly.
    """
    return PointCloudObserver(seed=OBSERVER_SEED, noise_std=NOISE_STD, dropout_prob=dropout_prob)


def run_study(
    style: CloudStyle,
    dropout_prob: float,
    n_trials: int,
    storage: str,
    n_jobs: int,
) -> optuna.Study:
    """Create or resume one study and append `n_trials` trials to it.

    Args:
        style:        Cloud geometry for SyntheticExperiment.generate.
        dropout_prob: Probability of dropping individual points from the observation.
        n_trials:     Number of new trials to append. 0 loads without optimizing.
        storage:      Optuna storage URL (SQLite path).
        n_jobs:       Worker processes for parallelizing across seeds within a trial.

    Returns:
        The Optuna study, after optimization.
    """
    study = optuna.create_study(
        direction='minimize',  # mean_true_residual
        sampler=optuna.samplers.TPESampler(seed=0),
        study_name=study_name(style, dropout_prob),
        storage=storage,
        load_if_exists=True,
    )
    n_before = len(study.trials)
    print(f"\n=== style={style!r}  dropout_prob={dropout_prob}  "
          f"({n_before} trial(s) already stored) ===")
    study.optimize(
        make_beta_sweep_objective(
            style, TUNING_SEEDS, GEN_KWARGS, MAX_ITER, TOL,
            observer=build_observer(dropout_prob), n_jobs=n_jobs,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"  {len(study.trials)} total trials ({len(study.trials) - n_before} new).")
    return study


def select_best_reliable_trial(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    """Pick the lowest-residual trial that also clears the reliability threshold.

    Optimizing mean true residual alone can favour a configuration that is very
    accurate on most seeds but fails catastrophically on a few. Filtering by
    reliability first matches how notebook 7 reports its best config.

    Args:
        study: A completed (or partially completed) study.

    Returns:
        The best reliable trial, the best trial overall if none clears the
        threshold, or None if the study has no completed trials.
    """
    trials = [t for t in study.trials if t.value is not None]
    if not trials:
        return None
    reliable = [t for t in trials if t.user_attrs.get('reliability', 0.0) >= RELIABILITY_THRESHOLD]
    if not reliable:
        print(f"  WARNING: no trial reaches reliability >= {RELIABILITY_THRESHOLD:.0%}; "
              "falling back to the best trial overall.")
        reliable = trials
    return min(reliable, key=lambda t: t.value)


def validate_on_holdout(
    trial: optuna.trial.FrozenTrial,
    style: CloudStyle,
    dropout_prob: float,
    n_jobs: int,
) -> dict[str, float]:
    """Replay a trial's hyperparameters on seeds the search never saw.

    The tuning-seed metrics are an optimistic estimate, since the search picked the
    configuration that does best on exactly those seeds.

    Args:
        trial:        Trial whose params should be replayed.
        style:        Cloud geometry for SyntheticExperiment.generate.
        dropout_prob: Dropout probability, matching the study the trial came from.
        n_jobs:       Worker processes for parallelizing across seeds.

    Returns:
        Metrics dictionary as returned by `evaluate_icp`.
    """
    factory = build_single_start_nn_icp_factory(FixedTrial(trial.params), MAX_ITER, TOL)
    return evaluate_icp(
        factory, style, HOLDOUT_SEEDS, GEN_KWARGS, trimmer=None,
        observer=build_observer(dropout_prob), n_jobs=n_jobs,
    )


def print_comparison(
    style: CloudStyle,
    bests: dict[float, optuna.trial.FrozenTrial],
    holdouts: dict[float, dict[str, float]],
) -> None:
    """Print the tuned configuration per dropout probability side by side.

    Args:
        style:    Cloud geometry the studies were run for.
        bests:    Best trial per dropout probability.
        holdouts: Holdout metrics per dropout probability.
    """
    dropouts = sorted(bests)
    rows = [
        ['beta',                     *[f"{bests[d].params['beta']:.4g}" for d in dropouts]],
        ['fe_k',                     *[bests[d].params['fe_k'] for d in dropouts]],
        ['feature_extractor',        *[bests[d].params['feature_extractor'] for d in dropouts]],
        ['Mean true residual',       *[f"{bests[d].value:.4g}" for d in dropouts]],
        ['  (holdout)',              *[f"{holdouts[d]['mean_true_residual']:.4g}" for d in dropouts]],
        ['Rotation error (°)',       *[f"{bests[d].user_attrs['mean_rot_err']:.3f}" for d in dropouts]],
        ['  (holdout)',              *[f"{holdouts[d]['mean_rot_err']:.3f}" for d in dropouts]],
        ['Reliability',              *[f"{bests[d].user_attrs['reliability']:.0%}" for d in dropouts]],
        ['  (holdout)',              *[f"{holdouts[d]['reliability']:.0%}" for d in dropouts]],
        ['Mean duration (s)',        *[f"{bests[d].user_attrs['mean_duration_s']:.3f}" for d in dropouts]],
    ]
    print(f"\n\n=== Tuned single-start config for style={style!r} ===")
    print(tabulate(rows, headers=['Parameter', *[f'dropout={d:g}' for d in dropouts]],
                   tablefmt='rounded_outline'))
    print("\nNotebook 5 (full space, n_starts=20, dropout=0.2) selected beta = 0.005395.")


def main() -> None:
    """Run one beta study per dropout probability and report the tuned configs."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-trials', type=int, default=150,
                        help='New trials to append per study (default: 150).')
    parser.add_argument('--style', type=str, default='muscle-fiber',
                        help='Cloud style (default: muscle-fiber, matching notebook 7).')
    parser.add_argument('--dropout', type=float, nargs='+', default=DEFAULT_DROPOUTS,
                        help='Dropout probabilities, one study each (default: 0.0 0.2).')
    parser.add_argument('--storage', type=str, default=DEFAULT_STORAGE,
                        help=f'Optuna storage URL (default: {DEFAULT_STORAGE}).')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Workers parallelizing seeds within a trial (default: -1, all cores).')
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    style: CloudStyle = args.style

    bests: dict[float, optuna.trial.FrozenTrial] = {}
    holdouts: dict[float, dict[str, float]] = {}
    for dropout_prob in args.dropout:
        study = run_study(style, dropout_prob, args.n_trials, args.storage, args.n_jobs)
        best = select_best_reliable_trial(study)
        if best is None:
            print(f"  No completed trials for dropout={dropout_prob}; skipping.")
            continue
        bests[dropout_prob] = best
        holdouts[dropout_prob] = validate_on_holdout(best, style, dropout_prob, args.n_jobs)

    if bests:
        print_comparison(style, bests, holdouts)


if __name__ == '__main__':
    main()
