import numpy as np

from algebra_utils import rotation_angle, sample_dispersed_rotations, sample_uniform_rotations


def _min_pairwise_angle(rotations: list[np.ndarray]) -> float:
    angles = [
        rotation_angle(rotations[i], rotations[j])
        for i in range(len(rotations))
        for j in range(i + 1, len(rotations))
    ]
    return min(angles)


def test_sample_dispersed_rotations_returns_valid_rotation_matrices():
    rotations = sample_dispersed_rotations(8, rng=np.random.default_rng(0))
    assert len(rotations) == 8
    for R in rotations:
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)


def test_sample_dispersed_rotations_is_reproducible():
    a = sample_dispersed_rotations(6, rng=np.random.default_rng(42))
    b = sample_dispersed_rotations(6, rng=np.random.default_rng(42))
    for Ra, Rb in zip(a, b):
        np.testing.assert_array_equal(Ra, Rb)


def test_sample_dispersed_rotations_handles_n_zero_and_one():
    assert sample_dispersed_rotations(0, rng=np.random.default_rng(0)) == []
    single = sample_dispersed_rotations(1, rng=np.random.default_rng(0))
    assert len(single) == 1
    assert single[0].shape == (3, 3)


def test_sample_dispersed_rotations_are_more_spread_than_uniform_sampling():
    n = 10
    dispersed = sample_dispersed_rotations(n, rng=np.random.default_rng(7))
    uniform = sample_uniform_rotations(n, rng=np.random.default_rng(7))
    assert _min_pairwise_angle(dispersed) >= _min_pairwise_angle(uniform)
