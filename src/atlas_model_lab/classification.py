"""Deterministic synthetic classification built from the Stage 2 primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from atlas_model_lab.layers import Dense, FloatArray, ReLU, softmax
from atlas_model_lab.losses import softmax_cross_entropy, softmax_cross_entropy_backward

IntArray = npt.NDArray[np.int64]


def tiny_classification_dataset() -> tuple[FloatArray, IntArray]:
    """Return the bounded two-class dataset used by the Stage 3 experiment."""

    inputs = np.array(
        [
            [0.10, 0.20],
            [0.20, 0.10],
            [0.15, 0.25],
            [0.25, 0.15],
            [0.75, 0.85],
            [0.85, 0.75],
            [0.80, 0.90],
            [0.90, 0.80],
        ],
        dtype=np.float64,
    )
    targets = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    return inputs, targets


@dataclass(frozen=True, slots=True)
class ClassificationTrainingHistory:
    """Immutable measurements from one deterministic classifier training run."""

    losses: tuple[float, ...]
    accuracies: tuple[float, ...]

    @property
    def steps(self) -> int:
        """Return the number of parameter updates represented by the history."""

        return len(self.losses) - 1

    @property
    def initial_loss(self) -> float:
        return self.losses[0]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]

    @property
    def initial_accuracy(self) -> float:
        return self.accuracies[0]

    @property
    def final_accuracy(self) -> float:
        return self.accuracies[-1]


class TinyClassifier:
    """A fixed 2 -> 4 -> 2 classifier for the bounded Stage 3 experiment."""

    def __init__(self) -> None:
        self.hidden = Dense(
            2,
            4,
            weights=np.array(
                [[0.20, 0.10, 0.30, 0.15], [0.10, 0.25, 0.15, 0.30]],
                dtype=np.float64,
            ),
            bias=np.full(4, 0.05, dtype=np.float64),
        )
        self.activation = ReLU()
        self.output = Dense(
            4,
            2,
            weights=np.array(
                [[0.10, -0.10], [-0.05, 0.05], [0.08, -0.08], [-0.02, 0.02]],
                dtype=np.float64,
            ),
            bias=np.zeros(2, dtype=np.float64),
        )

    def logits(self, inputs: FloatArray) -> FloatArray:
        """Return class logits for a two-feature batch."""

        hidden_values = self.hidden.forward(inputs)
        activated_values = self.activation.forward(hidden_values)
        return self.output.forward(activated_values)

    def probabilities(self, inputs: FloatArray) -> FloatArray:
        """Return class probabilities for a two-feature batch."""

        return softmax(self.logits(inputs))

    def predict(self, inputs: FloatArray) -> IntArray:
        """Return the most likely class index for each row."""

        return np.argmax(self.logits(inputs), axis=1).astype(np.int64, copy=False)

    def evaluate(self, inputs: FloatArray, targets: npt.ArrayLike) -> tuple[float, float]:
        """Return mean cross-entropy loss and classification accuracy."""

        logits = self.logits(inputs)
        loss = softmax_cross_entropy(logits, targets)
        labels = np.asarray(targets, dtype=np.int64)
        predictions = np.argmax(logits, axis=1)
        accuracy = float(np.mean(predictions == labels))
        return loss, accuracy

    def train_step(
        self,
        inputs: FloatArray,
        targets: npt.ArrayLike,
        *,
        learning_rate: float,
    ) -> float:
        """Run one full-batch gradient-descent update and return pre-update loss."""

        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be a finite positive number")

        logits = self.logits(inputs)
        loss = softmax_cross_entropy(logits, targets)
        output_gradient = softmax_cross_entropy_backward(logits, targets)

        activated_gradient, output_weight_gradient, output_bias_gradient = self.output.backward(
            output_gradient
        )
        hidden_gradient = self.activation.backward(activated_gradient)
        _, hidden_weight_gradient, hidden_bias_gradient = self.hidden.backward(hidden_gradient)

        self.output.weights -= learning_rate * output_weight_gradient
        self.output.bias -= learning_rate * output_bias_gradient
        self.hidden.weights -= learning_rate * hidden_weight_gradient
        self.hidden.bias -= learning_rate * hidden_bias_gradient

        return loss


def train_tiny_classifier(
    inputs: FloatArray,
    targets: npt.ArrayLike,
    *,
    steps: int = 500,
    learning_rate: float = 0.2,
) -> tuple[TinyClassifier, ClassificationTrainingHistory]:
    """Train a fresh classifier with deterministic full-batch gradient descent."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a finite positive number")

    model = TinyClassifier()
    initial_loss, initial_accuracy = model.evaluate(inputs, targets)
    losses = [initial_loss]
    accuracies = [initial_accuracy]

    for _ in range(steps):
        model.train_step(inputs, targets, learning_rate=learning_rate)
        loss, accuracy = model.evaluate(inputs, targets)
        losses.append(loss)
        accuracies.append(accuracy)

    history = ClassificationTrainingHistory(
        losses=tuple(losses),
        accuracies=tuple(accuracies),
    )
    return model, history
