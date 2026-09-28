from __future__ import annotations

import numpy as np
import pytest

from atlas_model_lab import Dense, ReLU, softmax


def _example_dense() -> Dense:
    return Dense(
        2,
        2,
        weights=np.array([[0.5, -1.0], [2.0, 3.0]], dtype=np.float64),
        bias=np.array([0.25, -0.5], dtype=np.float64),
    )


def test_dense_forward_has_expected_shape_and_values() -> None:
    layer = _example_dense()
    inputs = np.array([[1.0, 2.0], [-1.0, 3.0]], dtype=np.float64)

    output = layer.forward(inputs)

    np.testing.assert_allclose(output, np.array([[4.75, 4.5], [5.75, 9.5]]))
    assert output.shape == (2, 2)


def test_dense_backward_returns_input_weight_and_bias_gradients() -> None:
    layer = _example_dense()
    inputs = np.array([[1.0, 2.0], [-1.0, 3.0]], dtype=np.float64)
    upstream = np.array([[2.0, -1.0], [0.5, 4.0]], dtype=np.float64)
    layer.forward(inputs)

    input_gradient, weight_gradient, bias_gradient = layer.backward(upstream)

    np.testing.assert_allclose(input_gradient, np.array([[2.0, 1.0], [-3.75, 13.0]]))
    np.testing.assert_allclose(weight_gradient, np.array([[1.5, -5.0], [5.5, 10.0]]))
    np.testing.assert_allclose(bias_gradient, np.array([2.5, 3.0]))


def test_dense_rejects_invalid_input_rank() -> None:
    layer = Dense(2, 2)

    with pytest.raises(ValueError, match="two-dimensional"):
        layer.forward(np.array([1.0, 2.0], dtype=np.float64))


def test_dense_rejects_incompatible_feature_count() -> None:
    layer = Dense(2, 2)

    with pytest.raises(ValueError, match="feature count"):
        layer.forward(np.ones((3, 3), dtype=np.float64))


def test_dense_rejects_backward_before_forward() -> None:
    layer = Dense(2, 2)

    with pytest.raises(RuntimeError, match="forward pass"):
        layer.backward(np.ones((1, 2), dtype=np.float64))


def test_dense_rejects_incompatible_upstream_gradient() -> None:
    layer = Dense(2, 2)
    layer.forward(np.ones((3, 2), dtype=np.float64))

    with pytest.raises(ValueError, match="upstream_gradient must have shape"):
        layer.backward(np.ones((2, 2), dtype=np.float64))


def test_dense_rejects_nonfinite_input() -> None:
    layer = Dense(2, 2)

    with pytest.raises(ValueError, match="finite"):
        layer.forward(np.array([[1.0, np.inf]], dtype=np.float64))


def test_relu_forward_preserves_shape_and_clips_at_zero() -> None:
    layer = ReLU()
    inputs = np.array([[-2.0, 0.0, 3.0], [4.0, -1.0, 0.0]], dtype=np.float64)

    output = layer.forward(inputs)

    np.testing.assert_array_equal(output, np.array([[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]))
    assert output.shape == inputs.shape


def test_relu_backward_passes_only_positive_inputs() -> None:
    layer = ReLU()
    inputs = np.array([[-2.0, 0.0, 3.0], [4.0, -1.0, 0.0]], dtype=np.float64)
    upstream = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    layer.forward(inputs)

    gradient = layer.backward(upstream)

    np.testing.assert_array_equal(gradient, np.array([[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]))


def test_relu_rejects_backward_before_forward() -> None:
    with pytest.raises(RuntimeError, match="forward pass"):
        ReLU().backward(np.ones((2, 2), dtype=np.float64))


def test_relu_rejects_wrong_upstream_shape() -> None:
    layer = ReLU()
    layer.forward(np.ones((2, 2), dtype=np.float64))

    with pytest.raises(ValueError, match="same shape"):
        layer.backward(np.ones((4,), dtype=np.float64))


def test_softmax_rows_are_probabilities_and_preserve_ordering() -> None:
    probabilities = softmax(np.array([[1.0, 2.0, 3.0], [3.0, 1.0, -1.0]]))

    np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))
    assert np.all(probabilities >= 0.0)
    assert probabilities[0, 2] > probabilities[0, 1] > probabilities[0, 0]
    assert probabilities[1, 0] > probabilities[1, 1] > probabilities[1, 2]


def test_softmax_is_translation_invariant_per_row() -> None:
    logits = np.array([[1.0, -2.0], [4.0, 5.0]], dtype=np.float64)
    translated = logits + np.array([[1000.0], [-999.0]], dtype=np.float64)

    np.testing.assert_allclose(softmax(logits), softmax(translated))


def test_softmax_handles_extreme_logits_without_nonfinite_values() -> None:
    probabilities = softmax(
        np.array([[1000.0, 0.0, -1000.0], [-1000.0, 0.0, 1000.0]], dtype=np.float64)
    )

    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))


def test_softmax_rejects_invalid_rank_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        softmax(np.ones(3, dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        softmax(np.array([[0.0, np.nan]], dtype=np.float64))
