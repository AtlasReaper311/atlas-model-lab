"""The smallest useful trainable model in Atlas Model Lab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass
class LinearUnit:
    """One trainable linear unit with a weight and bias."""

    weight: float = 0.0
    bias: float = 0.0

    def predict(self, x: FloatArray) -> FloatArray:
        """Return predictions for a one-dimensional batch."""

        values = _as_vector(x, name="x")
        return self.weight * values + self.bias

    def train_step(
        self,
        x: FloatArray,
        y: FloatArray,
        *,
        learning_rate: float,
    ) -> float:
        """Run one full-batch gradient-descent update and return pre-update loss."""

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        inputs = _as_vector(x, name="x")
        targets = _as_vector(y, name="y")

        if inputs.shape != targets.shape:
            raise ValueError("x and y must have the same shape")

        predictions = self.predict(inputs)
        error = predictions - targets
        loss = float(np.mean(error**2))

        weight_gradient = float(np.mean(2.0 * error * inputs))
        bias_gradient = float(np.mean(2.0 * error))

        self.weight -= learning_rate * weight_gradient
        self.bias -= learning_rate * bias_gradient

        return loss


@dataclass(frozen=True)
class TrainingHistory:
    """Immutable training result for one linear-unit experiment."""

    losses: tuple[float, ...]

    @property
    def first_loss(self) -> float:
        return self.losses[0]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]


def train_linear_unit(
    x: FloatArray,
    y: FloatArray,
    *,
    steps: int = 300,
    learning_rate: float = 0.05,
) -> tuple[LinearUnit, TrainingHistory]:
    """Train a linear unit from zero-initialised parameters."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    inputs = _as_vector(x, name="x")
    targets = _as_vector(y, name="y")

    if inputs.shape != targets.shape:
        raise ValueError("x and y must have the same shape")

    unit = LinearUnit()
    losses: list[float] = []

    for _ in range(steps):
        losses.append(
            unit.train_step(
                inputs,
                targets,
                learning_rate=learning_rate,
            )
        )

    return unit, TrainingHistory(losses=tuple(losses))


def _as_vector(values: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array
