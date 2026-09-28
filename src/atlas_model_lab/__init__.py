"""First-principles machine-learning experiments for Atlas Systems."""

from atlas_model_lab.gradient_check import (
    GradientCheckResult,
    check_gradient,
    finite_difference_gradient,
)
from atlas_model_lab.layers import Dense, ReLU, softmax
from atlas_model_lab.linear import LinearUnit, TrainingHistory, train_linear_unit
from atlas_model_lab.losses import (
    cross_entropy,
    softmax_cross_entropy,
    softmax_cross_entropy_backward,
)

__all__ = [
    "Dense",
    "GradientCheckResult",
    "LinearUnit",
    "ReLU",
    "TrainingHistory",
    "check_gradient",
    "cross_entropy",
    "finite_difference_gradient",
    "softmax",
    "softmax_cross_entropy",
    "softmax_cross_entropy_backward",
    "train_linear_unit",
]
__version__ = "0.1.0"
