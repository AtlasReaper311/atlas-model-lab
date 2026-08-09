from __future__ import annotations

import numpy as np
import pytest

from atlas_model_lab import LinearUnit, train_linear_unit


def test_prediction_uses_weight_and_bias() -> None:
    unit = LinearUnit(weight=2.0, bias=3.0)
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float64)

    prediction = unit.predict(x)

    np.testing.assert_allclose(prediction, np.array([1.0, 3.0, 7.0]))


def test_training_learns_linear_relationship() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    y = 2.0 * x + 3.0

    unit, history = train_linear_unit(x, y)

    assert history.final_loss < history.first_loss
    assert unit.weight == pytest.approx(2.0, abs=1e-9)
    assert unit.bias == pytest.approx(3.0, abs=1e-9)


def test_training_rejects_shape_mismatch() -> None:
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="same shape"):
        train_linear_unit(x, y)


def test_training_rejects_nonpositive_learning_rate() -> None:
    unit = LinearUnit()
    x = np.array([1.0], dtype=np.float64)
    y = np.array([2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="positive"):
        unit.train_step(x, y, learning_rate=0.0)
