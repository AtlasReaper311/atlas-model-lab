"""Classification losses with explicit, numerically stable derivatives."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from atlas_model_lab.layers import FloatArray, softmax


def cross_entropy(probabilities: FloatArray, targets: npt.ArrayLike) -> float:
    """Return mean multiclass cross-entropy for probability rows.

    ``probabilities`` must already be a finite row-normalised matrix and
    ``targets`` must be integer class indices. Zero probability is handled by
    taking the logarithm of the smallest positive float instead of producing
    ``log(0)``.
    """

    values = _as_probability_matrix(probabilities)
    labels = _as_targets(targets, batch_size=values.shape[0], classes=values.shape[1])
    selected_probabilities = values[np.arange(values.shape[0]), labels]
    safe_probabilities = np.maximum(selected_probabilities, np.finfo(np.float64).tiny)
    return float(-np.mean(np.log(safe_probabilities), dtype=np.float64))


def softmax_cross_entropy(logits: FloatArray, targets: npt.ArrayLike) -> float:
    """Return mean cross-entropy from logits using a stable log-sum-exp path."""

    values = _as_logits(logits)
    labels = _as_targets(targets, batch_size=values.shape[0], classes=values.shape[1])

    row_maximum = np.max(values, axis=1)
    shifted_logits = values - row_maximum[:, np.newaxis]
    log_sum_exp = row_maximum + np.log(np.sum(np.exp(shifted_logits), axis=1))
    target_logits = values[np.arange(values.shape[0]), labels]
    loss = np.mean(log_sum_exp - target_logits, dtype=np.float64)
    return float(loss)


def softmax_cross_entropy_backward(logits: FloatArray, targets: npt.ArrayLike) -> FloatArray:
    """Return the manual derivative of mean softmax cross-entropy.

    Starting from ``d_logits = probabilities``, subtract one at each target
    class and divide by the batch size. This is the simplified derivative of
    the combined softmax and cross-entropy operation.
    """

    values = _as_logits(logits)
    labels = _as_targets(targets, batch_size=values.shape[0], classes=values.shape[1])
    probabilities = softmax(values)
    logits_gradient = probabilities.copy()
    logits_gradient[np.arange(values.shape[0]), labels] -= 1.0
    logits_gradient /= values.shape[0]
    return logits_gradient


def _as_logits(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("logits must be two-dimensional")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("logits must have a non-empty batch and class dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError("logits must contain only finite values")
    return array


def _as_probability_matrix(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("probabilities must have a non-empty batch and class dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError("probabilities must contain only finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("probabilities must be between 0 and 1")
    if not np.allclose(
        np.sum(array, axis=1, dtype=np.float64),
        1.0,
        rtol=1e-7,
        atol=1e-12,
    ):
        raise ValueError("each probability row must sum to 1")
    return array


def _as_targets(
    targets: npt.ArrayLike,
    *,
    batch_size: int,
    classes: int,
) -> npt.NDArray[np.int64]:
    array = np.asarray(targets)
    if array.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if array.shape[0] != batch_size:
        raise ValueError(f"targets must have length {batch_size}, got {array.shape[0]}")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError("targets must contain integer class labels")
    if np.any(array < 0) or np.any(array >= classes):
        raise ValueError(f"targets must be in the range [0, {classes})")
    return array.astype(np.int64, copy=False)
