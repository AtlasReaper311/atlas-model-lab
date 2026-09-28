from __future__ import annotations

import numpy as np
import pytest

from atlas_model_lab import (
    cross_entropy,
    softmax,
    softmax_cross_entropy,
    softmax_cross_entropy_backward,
)


def test_cross_entropy_matches_known_probability_result() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float64)
    targets = np.array([0, 1], dtype=np.int64)

    loss = cross_entropy(probabilities, targets)

    assert loss == pytest.approx((-np.log(0.7) - np.log(0.8)) / 2.0)


def test_cross_entropy_accepts_integer_targets() -> None:
    probabilities = np.array([[0.25, 0.75]], dtype=np.float64)

    assert cross_entropy(probabilities, np.array([1], dtype=np.int64)) == pytest.approx(
        -np.log(0.75)
    )


@pytest.mark.parametrize("target", [np.array([-1]), np.array([2])])
def test_cross_entropy_rejects_out_of_range_targets(target: np.ndarray) -> None:
    with pytest.raises(ValueError, match="range"):
        cross_entropy(np.array([[0.5, 0.5]], dtype=np.float64), target)


def test_cross_entropy_rejects_non_integer_or_wrong_shape_targets() -> None:
    probabilities = np.array([[0.5, 0.5], [0.2, 0.8]], dtype=np.float64)

    with pytest.raises(ValueError, match="integer"):
        cross_entropy(probabilities, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        cross_entropy(probabilities, np.array([[0], [1]], dtype=np.int64))
    with pytest.raises(ValueError, match="length"):
        cross_entropy(probabilities, np.array([0], dtype=np.int64))


def test_cross_entropy_rejects_empty_and_nonfinite_probabilities() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cross_entropy(np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int64))
    with pytest.raises(ValueError, match="finite"):
        cross_entropy(np.array([[0.5, np.nan]], dtype=np.float64), np.array([0], dtype=np.int64))


def test_cross_entropy_rejects_non_normalised_or_negative_probabilities() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        cross_entropy(np.array([[1.1, -0.1]], dtype=np.float64), np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="sum to 1"):
        cross_entropy(np.array([[0.5, 0.4]], dtype=np.float64), np.array([0], dtype=np.int64))


def test_softmax_cross_entropy_is_finite_and_matches_probability_path() -> None:
    logits = np.array([[2.0, 0.5, -1.0], [-1.0, 1.5, 0.25]], dtype=np.float64)
    targets = np.array([0, 2], dtype=np.int64)

    logits_loss = softmax_cross_entropy(logits, targets)
    probability_loss = cross_entropy(softmax(logits), targets)

    assert np.isfinite(logits_loss)
    assert logits_loss == pytest.approx(probability_loss)


def test_softmax_cross_entropy_backward_has_expected_shape_and_row_sums() -> None:
    logits = np.array([[2.0, 0.5, -1.0], [-1.0, 1.5, 0.25]], dtype=np.float64)
    targets = np.array([0, 2], dtype=np.int64)

    gradient = softmax_cross_entropy_backward(logits, targets)

    assert gradient.shape == logits.shape
    np.testing.assert_allclose(np.sum(gradient, axis=1), np.zeros(2), atol=1e-15)
    assert gradient[0, 0] < 0.0
    assert gradient[1, 2] < 0.0


def test_softmax_cross_entropy_rejects_invalid_logits() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        softmax_cross_entropy(np.ones(3, dtype=np.float64), np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="finite"):
        softmax_cross_entropy(
            np.array([[0.0, np.inf]], dtype=np.float64),
            np.array([0], dtype=np.int64),
        )
