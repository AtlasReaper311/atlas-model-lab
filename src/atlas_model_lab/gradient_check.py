"""Finite-difference checks for first-principles analytical gradients."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ScalarFunction = Callable[[], float]


@dataclass(frozen=True, slots=True)
class GradientCheckResult:
    """Evidence from comparing one analytical gradient with finite differences."""

    passed: bool
    maximum_error: float
    worst_index: tuple[int, ...]
    analytical_value: float
    numerical_value: float
    epsilon: float
    tolerance: float
    checked_elements: int


def finite_difference_gradient(
    parameter: FloatArray,
    function: ScalarFunction,
    *,
    epsilon: float = 1e-7,
) -> FloatArray:
    """Approximate ``df/dparameter`` with centred differences.

    The parameter is changed one element at a time and restored in a ``finally``
    block, so even a failing objective cannot leave the caller's array altered.
    """

    _validate_parameter(parameter)
    _validate_epsilon(epsilon)

    original = parameter.copy()
    numerical_gradient = np.empty_like(parameter, dtype=np.float64)
    try:
        for index in np.ndindex(parameter.shape):
            original_value = float(original[index])

            parameter[index] = original_value + epsilon
            plus_value = _as_scalar(function())

            parameter[index] = original_value - epsilon
            minus_value = _as_scalar(function())

            numerical_gradient[index] = (plus_value - minus_value) / (2.0 * epsilon)
    finally:
        np.copyto(parameter, original)

    return numerical_gradient


def check_gradient(
    analytical_gradient: FloatArray,
    parameter: FloatArray,
    function: ScalarFunction,
    *,
    epsilon: float = 1e-7,
    tolerance: float = 1e-7,
) -> GradientCheckResult:
    """Compare an analytical gradient with finite differences.

    The comparison is ``abs(a - n) / max(1, abs(a), abs(n))``. The denominator
    keeps zero-valued gradients meaningful without dividing by zero.
    """

    _validate_parameter(parameter)
    _validate_epsilon(epsilon)
    _validate_tolerance(tolerance)

    analytical = np.asarray(analytical_gradient, dtype=np.float64)
    if analytical.shape != parameter.shape:
        raise ValueError(
            "analytical_gradient must have the same shape as parameter "
            f"{parameter.shape}, got {analytical.shape}"
        )
    if not np.all(np.isfinite(analytical)):
        raise ValueError("analytical_gradient must contain only finite values")

    numerical = finite_difference_gradient(parameter, function, epsilon=epsilon)
    denominator = np.maximum(
        1.0,
        np.maximum(np.abs(analytical), np.abs(numerical)),
    )
    errors = np.abs(analytical - numerical) / denominator
    worst_flat_index = int(np.argmax(errors))
    worst_index = tuple(int(value) for value in np.unravel_index(worst_flat_index, errors.shape))
    maximum_error = float(errors[worst_index])

    return GradientCheckResult(
        passed=maximum_error <= tolerance,
        maximum_error=maximum_error,
        worst_index=worst_index,
        analytical_value=float(analytical[worst_index]),
        numerical_value=float(numerical[worst_index]),
        epsilon=epsilon,
        tolerance=tolerance,
        checked_elements=parameter.size,
    )


def _validate_parameter(parameter: FloatArray) -> None:
    if not isinstance(parameter, np.ndarray):
        raise TypeError("parameter must be a NumPy array")
    if parameter.ndim == 0 or parameter.size == 0:
        raise ValueError("parameter must be a non-empty array")
    if not np.issubdtype(parameter.dtype, np.floating):
        raise ValueError("parameter must use a floating-point dtype")
    if not parameter.flags.writeable:
        raise ValueError("parameter must be writeable for finite differences")
    if not np.all(np.isfinite(parameter)):
        raise ValueError("parameter must contain only finite values")


def _validate_epsilon(epsilon: float) -> None:
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite positive number")


def _validate_tolerance(tolerance: float) -> None:
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a finite positive number")


def _as_scalar(value: float) -> float:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError("gradient-check function must return one scalar")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError("gradient-check function must return a finite scalar")
    return scalar
