from __future__ import annotations

import numpy as np
import pytest

from atlas_model_lab import (
    TinyClassifier,
    tiny_classification_dataset,
    train_tiny_classifier,
)


def test_tiny_dataset_is_deterministic_and_returns_fresh_arrays() -> None:
    first_inputs, first_targets = tiny_classification_dataset()
    second_inputs, second_targets = tiny_classification_dataset()

    np.testing.assert_array_equal(first_inputs, second_inputs)
    np.testing.assert_array_equal(first_targets, second_targets)
    assert first_inputs.shape == (8, 2)
    assert first_targets.shape == (8,)
    assert set(first_targets.tolist()) == {0, 1}

    first_inputs[0, 0] = 99.0
    first_targets[0] = 1

    assert second_inputs[0, 0] == pytest.approx(0.10)
    assert second_targets[0] == 0


def test_classifier_probabilities_and_predictions_have_expected_shapes() -> None:
    inputs, _ = tiny_classification_dataset()
    model = TinyClassifier()

    probabilities = model.probabilities(inputs)
    predictions = model.predict(inputs)

    assert probabilities.shape == (8, 2)
    assert predictions.shape == (8,)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(8))


def test_one_training_step_updates_parameters_and_reduces_dataset_loss() -> None:
    inputs, targets = tiny_classification_dataset()
    model = TinyClassifier()
    initial_loss, _ = model.evaluate(inputs, targets)
    hidden_before = model.hidden.weights.copy()
    output_before = model.output.weights.copy()

    returned_loss = model.train_step(inputs, targets, learning_rate=0.2)
    next_loss, _ = model.evaluate(inputs, targets)

    assert returned_loss == pytest.approx(initial_loss)
    assert next_loss < initial_loss
    assert not np.array_equal(model.hidden.weights, hidden_before)
    assert not np.array_equal(model.output.weights, output_before)


def test_training_memorises_the_bounded_dataset() -> None:
    inputs, targets = tiny_classification_dataset()

    model, history = train_tiny_classifier(inputs, targets)

    assert history.steps == 500
    assert len(history.losses) == 501
    assert len(history.accuracies) == 501
    assert history.final_loss < history.initial_loss
    assert history.final_loss < 0.02
    assert history.initial_accuracy == pytest.approx(0.5)
    assert history.final_accuracy == pytest.approx(1.0)
    np.testing.assert_array_equal(model.predict(inputs), targets)


def test_training_is_reproducible() -> None:
    inputs, targets = tiny_classification_dataset()

    first_model, first_history = train_tiny_classifier(inputs, targets, steps=200)
    second_model, second_history = train_tiny_classifier(inputs, targets, steps=200)

    assert first_history == second_history
    np.testing.assert_array_equal(first_model.hidden.weights, second_model.hidden.weights)
    np.testing.assert_array_equal(first_model.hidden.bias, second_model.hidden.bias)
    np.testing.assert_array_equal(first_model.output.weights, second_model.output.weights)
    np.testing.assert_array_equal(first_model.output.bias, second_model.output.bias)


def test_classifier_rejects_wrong_feature_count() -> None:
    model = TinyClassifier()

    with pytest.raises(ValueError, match="feature count"):
        model.predict(np.ones((4, 3), dtype=np.float64))


def test_classifier_rejects_invalid_targets() -> None:
    inputs, _ = tiny_classification_dataset()
    model = TinyClassifier()

    with pytest.raises(ValueError, match="range"):
        model.evaluate(inputs, np.full(8, 2, dtype=np.int64))

    with pytest.raises(ValueError, match="integer"):
        model.evaluate(inputs, np.zeros(8, dtype=np.float64))


@pytest.mark.parametrize("learning_rate", [0.0, -0.1, np.inf, np.nan])
def test_training_step_rejects_invalid_learning_rate(learning_rate: float) -> None:
    inputs, targets = tiny_classification_dataset()

    with pytest.raises(ValueError, match="learning_rate"):
        TinyClassifier().train_step(inputs, targets, learning_rate=learning_rate)


@pytest.mark.parametrize("steps", [0, -1, True])
def test_train_tiny_classifier_rejects_invalid_steps(steps: int) -> None:
    inputs, targets = tiny_classification_dataset()

    with pytest.raises(ValueError, match="steps"):
        train_tiny_classifier(inputs, targets, steps=steps)


def test_train_tiny_classifier_rejects_invalid_learning_rate() -> None:
    inputs, targets = tiny_classification_dataset()

    with pytest.raises(ValueError, match="learning_rate"):
        train_tiny_classifier(inputs, targets, learning_rate=0.0)
