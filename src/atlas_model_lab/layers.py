"""Small neural-network layers with explicit NumPy forward and backward passes."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class Dense:
    """A fully connected layer implementing ``Y = X @ W + b``.

    Parameters default to zero arrays so that construction is deterministic. A
    caller can provide explicit arrays when a non-zero educational example is
    useful. Initialisation experiments are intentionally outside this class.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        *,
        weights: FloatArray | None = None,
        bias: FloatArray | None = None,
    ) -> None:
        if (
            isinstance(input_features, bool)
            or not isinstance(input_features, int)
            or input_features <= 0
        ):
            raise ValueError("input_features must be a positive integer")
        if (
            isinstance(output_features, bool)
            or not isinstance(output_features, int)
            or output_features <= 0
        ):
            raise ValueError("output_features must be a positive integer")

        self.input_features = input_features
        self.output_features = output_features
        self.weights = _as_weights(
            np.zeros((input_features, output_features), dtype=np.float64)
            if weights is None
            else weights,
            input_features=input_features,
            output_features=output_features,
        )
        self.bias = _as_bias(
            np.zeros(output_features, dtype=np.float64) if bias is None else bias,
            output_features=output_features,
        )
        self._cached_input: FloatArray | None = None

    def forward(self, inputs: FloatArray) -> FloatArray:
        """Return the dense layer output for a two-dimensional batch."""

        values = _as_batch(inputs, name="inputs")
        if values.shape[1] != self.input_features:
            raise ValueError(
                "inputs feature count must match input_features "
                f"({self.input_features}), got {values.shape[1]}"
            )

        self._cached_input = values.copy()
        return values @ self.weights + self.bias

    def backward(self, upstream_gradient: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``dX``, ``dW``, and ``db`` for the last forward pass.

        The formulas are kept next to the code so the implementation mirrors
        the mathematics:

        ``dX = dY @ W.T``; ``dW = X.T @ dY``; ``db = sum(dY, axis=0)``.
        """

        if self._cached_input is None:
            raise RuntimeError("backward requires a compatible forward pass first")

        gradient = _as_batch(upstream_gradient, name="upstream_gradient")
        expected_shape = (self._cached_input.shape[0], self.output_features)
        if gradient.shape != expected_shape:
            raise ValueError(
                f"upstream_gradient must have shape {expected_shape}, got {gradient.shape}"
            )

        input_gradient = gradient @ self.weights.T
        weight_gradient = self._cached_input.T @ gradient
        bias_gradient = np.sum(gradient, axis=0, dtype=np.float64)
        return input_gradient, weight_gradient, bias_gradient


class ReLU:
    """Rectified linear unit with a manual derivative.

    At exactly zero the derivative is defined as zero. That convention keeps
    the backward pass deterministic at ReLU's non-differentiable point.
    """

    def __init__(self) -> None:
        self._cached_input: FloatArray | None = None

    def forward(self, inputs: FloatArray) -> FloatArray:
        """Return ``max(0, inputs)`` while caching the input mask."""

        values = _as_array(inputs, name="inputs")
        self._cached_input = values.copy()
        return np.maximum(values, 0.0)

    def backward(self, upstream_gradient: FloatArray) -> FloatArray:
        """Return the upstream gradient where the cached input was positive."""

        if self._cached_input is None:
            raise RuntimeError("backward requires a compatible forward pass first")

        gradient = _as_array(upstream_gradient, name="upstream_gradient")
        if gradient.shape != self._cached_input.shape:
            raise ValueError(
                "upstream_gradient must have the same shape as the cached input "
                f"{self._cached_input.shape}, got {gradient.shape}"
            )

        return np.where(self._cached_input > 0.0, gradient, 0.0)


def softmax(logits: FloatArray) -> FloatArray:
    """Return row-wise softmax probabilities using a stable shifted exponent."""

    values = _as_batch(logits, name="logits")
    row_maximum = np.max(values, axis=1, keepdims=True)
    shifted_logits = values - row_maximum
    exponentials = np.exp(shifted_logits)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _as_array(values: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_batch(values: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have a non-empty batch and feature dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_weights(
    values: FloatArray,
    *,
    input_features: int,
    output_features: int,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    expected_shape = (input_features, output_features)
    if array.shape != expected_shape:
        raise ValueError(f"weights must have shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("weights must contain only finite values")
    return array.copy()


def _as_bias(values: FloatArray, *, output_features: int) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    expected_shape = (output_features,)
    if array.shape != expected_shape:
        raise ValueError(f"bias must have shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("bias must contain only finite values")
    return array.copy()
