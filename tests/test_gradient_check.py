from __future__ import annotations

import numpy as np
import pytest

from atlas_model_lab import (
    Dense,
    ReLU,
    check_gradient,
    softmax_cross_entropy,
    softmax_cross_entropy_backward,
)

EPSILON = 1e-7
TOLERANCE = 1e-7


def test_gradient_checker_accepts_correct_quadratic_gradient_and_restores_array() -> None:
    parameter = np.array([0.4, -1.2, 2.0], dtype=np.float64)
    original = parameter.copy()

    result = check_gradient(
        2.0 * parameter,
        parameter,
        lambda: float(np.sum(parameter**2)),
        epsilon=EPSILON,
        tolerance=TOLERANCE,
    )

    assert result.passed
    assert result.maximum_error < TOLERANCE
    assert result.checked_elements == parameter.size
    np.testing.assert_array_equal(parameter, original)


def test_gradient_checker_reports_incorrect_gradient_and_worst_element() -> None:
    parameter = np.array([0.4, -1.2, 2.0], dtype=np.float64)

    result = check_gradient(
        np.array([0.8, 99.0, 4.0], dtype=np.float64),
        parameter,
        lambda: float(np.sum(parameter**2)),
        epsilon=EPSILON,
        tolerance=TOLERANCE,
    )

    assert not result.passed
    assert result.worst_index == (1,)
    assert result.analytical_value == pytest.approx(99.0)
    assert result.numerical_value == pytest.approx(-2.4)
    assert result.maximum_error > TOLERANCE


def test_gradient_checker_validates_epsilon_and_tolerance() -> None:
    parameter = np.array([1.0], dtype=np.float64)
    gradient = np.array([2.0], dtype=np.float64)

    def function() -> float:
        return float(np.sum(parameter**2))

    with pytest.raises(ValueError, match="epsilon"):
        check_gradient(gradient, parameter, function, epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        check_gradient(gradient, parameter, function, epsilon=np.inf)
    with pytest.raises(ValueError, match="tolerance"):
        check_gradient(gradient, parameter, function, tolerance=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        check_gradient(gradient, parameter, function, tolerance=np.nan)


def test_dense_gradients_match_finite_differences() -> None:
    layer = Dense(
        2,
        2,
        weights=np.array([[0.2, -0.4], [0.7, 0.3]], dtype=np.float64),
        bias=np.array([0.1, -0.2], dtype=np.float64),
    )
    inputs = np.array([[0.5, -1.0], [1.25, 0.75]], dtype=np.float64)
    upstream = np.array([[0.4, -0.8], [1.1, 0.2]], dtype=np.float64)
    layer.forward(inputs)
    input_gradient, weight_gradient, bias_gradient = layer.backward(upstream)

    def objective() -> float:
        return float(np.sum(layer.forward(inputs) * upstream))

    weight_result = check_gradient(
        weight_gradient,
        layer.weights,
        objective,
        epsilon=EPSILON,
        tolerance=TOLERANCE,
    )
    bias_result = check_gradient(
        bias_gradient,
        layer.bias,
        objective,
        epsilon=EPSILON,
        tolerance=TOLERANCE,
    )
    input_result = check_gradient(
        input_gradient,
        inputs,
        objective,
        epsilon=EPSILON,
        tolerance=TOLERANCE,
    )

    assert weight_result.passed
    assert bias_result.passed
    assert input_result.passed


def test_relu_gradient_matches_finite_differences_away_from_zero() -> None:
    layer = ReLU()
    inputs = np.array([[-2.0, -0.5, 0.25, 2.0]], dtype=np.float64)
    upstream = np.array([[0.4, -0.8, 1.1, 0.2]], dtype=np.float64)
    layer.forward(inputs)
    analytical_gradient = layer.backward(upstream)

    result = check_gradient(
        analytical_gradient,
        inputs,
        lambda: float(np.sum(layer.forward(inputs) * upstream)),
        epsilon=EPSILON,
        tolerance=TOLERANCE,
    )

    assert result.passed


def test_softmax_cross_entropy_gradient_matches_finite_differences() -> None:
    logits = np.array([[0.8, -0.2, 1.1], [-0.5, 0.25, 0.4]], dtype=np.float64)
    targets = np.array([2, 1], dtype=np.int64)
    analytical_gradient = softmax_cross_entropy_backward(logits, targets)

    result = check_gradient(
        analytical_gradient,
        logits,
        lambda: softmax_cross_entropy(logits, targets),
        epsilon=EPSILON,
        tolerance=TOLERANCE,
    )

    assert result.passed


def test_composed_dense_relu_dense_path_has_correct_first_layer_gradient() -> None:
    first = Dense(
        2,
        3,
        weights=np.array([[0.4, -0.3, 0.2], [0.1, 0.5, -0.6]], dtype=np.float64),
        bias=np.array([0.2, -0.1, 0.3], dtype=np.float64),
    )
    activation = ReLU()
    second = Dense(
        3,
        2,
        weights=np.array([[0.5, -0.2], [-0.4, 0.6], [0.3, 0.7]], dtype=np.float64),
        bias=np.array([0.1, -0.2], dtype=np.float64),
    )
    inputs = np.array([[0.5, -0.75], [1.0, 0.25]], dtype=np.float64)
    targets = np.array([0, 1], dtype=np.int64)

    hidden = first.forward(inputs)
    activated = activation.forward(hidden)
    logits = second.forward(activated)
    output_gradient = softmax_cross_entropy_backward(logits, targets)
    activated_gradient, _, _ = second.backward(output_gradient)
    hidden_gradient = activation.backward(activated_gradient)
    _, first_weight_gradient, _ = first.backward(hidden_gradient)

    def objective() -> float:
        hidden_values = first.forward(inputs)
        activated_values = activation.forward(hidden_values)
        return softmax_cross_entropy(second.forward(activated_values), targets)

    result = check_gradient(
        first_weight_gradient,
        first.weights,
        objective,
        epsilon=EPSILON,
        tolerance=TOLERANCE,
    )

    assert result.passed
