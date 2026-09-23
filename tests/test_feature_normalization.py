"""Tests for joint feature z-scoring in feature_extractor."""
import numpy as np
import pytest

from feature_extractor import (
    RobustGeometricFeatureExtractor,
    zscore_jointly,
    zscored_features,
)
from point_cloud import PointCloud


@pytest.fixture
def clouds() -> list[PointCloud]:
    rng = np.random.default_rng(0)
    return [
        PointCloud(rng.uniform(-10, 10, size=(60, 3))),
        PointCloud(rng.uniform(-4, 4, size=(40, 3))),
    ]


def test_pooled_result_has_zero_mean_and_unit_variance() -> None:
    """Statistics are pooled, so the concatenation is standardised, not each part."""
    rng = np.random.default_rng(1)
    matrices = [rng.normal(5.0, 3.0, size=(200, 4)), rng.normal(-2.0, 7.0, size=(150, 4))]

    pooled = np.concatenate(zscore_jointly(matrices), axis=0)

    np.testing.assert_allclose(pooled.mean(axis=0), np.zeros(4), atol=1e-8)
    np.testing.assert_allclose(pooled.std(axis=0), np.ones(4), atol=1e-6)


def test_matrices_are_not_standardised_individually() -> None:
    """Each matrix keeps its offset relative to the others.

    Normalising per matrix would erase genuine differences between clouds, which is
    exactly the signal a matcher needs to tell them apart.
    """
    low, high = np.zeros((50, 1)), np.ones((50, 1))
    normalised_low, normalised_high = zscore_jointly([low, high])

    assert normalised_low.mean() < normalised_high.mean()
    assert not np.isclose(normalised_low.mean(), 0.0)


def test_shapes_and_order_are_preserved() -> None:
    """Outputs line up one-to-one with inputs."""
    rng = np.random.default_rng(2)
    matrices = [rng.standard_normal((n, 3)) for n in (10, 25, 7)]
    out = zscore_jointly(matrices)
    assert [m.shape for m in out] == [m.shape for m in matrices]


def test_constant_dimension_does_not_divide_by_zero() -> None:
    """A dimension with no spread must stay finite rather than blow up."""
    constant = np.hstack([np.full((30, 1), 2.5), np.arange(30).reshape(-1, 1).astype(float)])
    out = zscore_jointly([constant])[0]
    assert np.all(np.isfinite(out))


def test_single_matrix_is_allowed() -> None:
    """One matrix is a valid degenerate case, used by the trimmer on a single cloud."""
    rng = np.random.default_rng(3)
    out = zscore_jointly([rng.normal(4.0, 2.0, size=(100, 2))])[0]
    np.testing.assert_allclose(out.mean(axis=0), np.zeros(2), atol=1e-8)


def test_empty_input_is_rejected() -> None:
    """Nothing to pool over is a usage error, not an empty result."""
    with pytest.raises(ValueError):
        zscore_jointly([])


def test_mismatched_widths_are_rejected() -> None:
    """Matrices from different extractors cannot share a feature space."""
    with pytest.raises(ValueError):
        zscore_jointly([np.zeros((10, 3)), np.zeros((10, 4))])


def test_zscored_features_matches_extract_then_normalise(clouds: list[PointCloud]) -> None:
    """The convenience wrapper must be exactly extraction followed by zscore_jointly."""
    extractor = RobustGeometricFeatureExtractor(k=10)

    via_wrapper = zscored_features(extractor, clouds)
    via_parts = zscore_jointly([extractor.get_features(cloud) for cloud in clouds])

    for wrapped, manual in zip(via_wrapper, via_parts):
        np.testing.assert_array_equal(wrapped, manual)


def test_prescaling_is_cancelled_by_normalisation(clouds: list[PointCloud]) -> None:
    """Scaling a dimension before z-scoring has no effect.

    This is why per-dimension weighting cannot be implemented as a FeatureExtractor
    wrapper and has to be applied downstream of normalisation.
    """
    extractor = RobustGeometricFeatureExtractor(k=10)
    raw = [extractor.get_features(cloud) for cloud in clouds]

    weights = np.linspace(0.5, 4.0, raw[0].shape[1])
    baseline = zscore_jointly(raw)
    prescaled = zscore_jointly([features * weights for features in raw])

    for plain, scaled in zip(baseline, prescaled):
        np.testing.assert_allclose(plain, scaled, atol=1e-8)
