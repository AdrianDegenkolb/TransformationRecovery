# Transformation Recovery

Recovering the rigid (and elastic) transformation between two unordered, noisy point clouds that both originate from a shared but unknown source cloud.

<video src="https://github.com/user-attachments/assets/85f458bf-f2f5-4f45-9736-40ab21ace136" controls></video>

## Problem setting

Given a source point cloud `S`, two transformations `T1` and `T2` (each with additive noise) produce two observed clouds:

```
P = T1(S)
Q = T2(S)
```

Neither `S` nor `T1`/`T2` are known at recovery time — only `P` and `Q` are observed, and their points are not in correspondence. The goal is to recover the transformation

```
T_gt = T2 ∘ T1⁻¹
```

that maps `P` onto `Q`, using variants of Iterative Closest Point (ICP) registration.

## Approach

The pipeline alternates between two steps until convergence:

1. **Matching (E-step)** — find correspondences between points in `P` and `Q`, either via hard nearest-neighbor assignment or soft (Gaussian-weighted) assignment, optionally in a joint position+feature space.
2. **Fitting (M-step)** — fit a transformation (rigid or elastic) that minimizes the residuals given the current correspondences.

On top of vanilla ICP, the project explores several extensions:

- **Feature-augmented matching** — per-point geometric features (e.g. neighborhood curvature/shape descriptors) that are invariant to the transformation, used to disambiguate matches beyond spatial proximity, including a dropout-resilient robust variant.
- **Trimming / clustering** — removing or collapsing geometrically redundant points (e.g. dense clusters represented by their centroid) before registration.
- **Elastic transformations** — a rigid component plus a smooth per-point residual field, fitted with a data term, magnitude penalty, and graph-Laplacian smoothness term (implemented with PyTorch).
- **Multi-start ICP** — running ICP from multiple random rotation initializations to avoid local optima.
- **Hyperparameter optimization** — tuning matcher/ICP/feature-extractor settings with Optuna.

## Project structure

```
src/
  point_cloud.py           PointCloud data structure
  transformation.py        Rigid transformation (fit/apply/residuals)
  elastic_transformation.py Elastic transformation (rigid + smooth residual field)
  matcher.py                Correspondence matchers (nearest-neighbor, Gaussian/soft, feature-joint)
  icp.py                     ICP loop, callbacks (e.g. sigma annealing), multi-start variant
  trimmer.py                 Point cloud trimming/clustering utilities
  feature_extractor.py       Per-point geometric feature extractors
  synthetic.py                Synthetic experiment generation (source cloud + ground-truth transforms)
  algebra_utils.py           Rotation utilities (SO(3) sampling, 6D rotation representation)
  hpo.py                      Optuna-based hyperparameter search
  visualization.py            Plotting utilities for clouds, convergence, residuals, HPO results
  utils.py                     Convergence/accuracy metrics over ICP results

notebooks/    Numbered experiments demonstrating each component (rigid vs. elastic,
              matching strategies, cloud styles, clustering, trimming, feature extractors,
              hyperparameter optimization, multi-start, incomplete observations, ...)
tests/        Unit tests for matcher, trimmer, feature extractors, synthetic data
results/      Generated plots referenced by the notebooks
```

## Setup

Requires Python 3.13. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Running tests

```bash
uv run pytest
```

## Notebooks

The `notebooks/` directory contains numbered, self-contained experiments — start with `1_rigid_vs_elastic_transformation.ipynb` and `2_icp.ipynb` for the core registration loop, then explore matching, cloud styles, clustering/trimming, feature extraction, and hyperparameter optimization in later notebooks.
