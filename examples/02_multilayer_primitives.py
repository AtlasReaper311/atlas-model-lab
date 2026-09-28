"""Demonstrate Stage 2 layers, loss, backward passes, and gradient checking."""

from __future__ import annotations

import numpy as np

from atlas_model_lab import (
    Dense,
    ReLU,
    check_gradient,
    softmax,
    softmax_cross_entropy,
    softmax_cross_entropy_backward,
)


def main() -> None:
    inputs = np.array([[0.5, -0.75], [1.0, 0.25]], dtype=np.float64)
    targets = np.array([0, 1], dtype=np.int64)

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

    hidden = first.forward(inputs)
    activated = activation.forward(hidden)
    logits = second.forward(activated)
    probabilities = softmax(logits)
    loss = softmax_cross_entropy(logits, targets)

    output_gradient = softmax_cross_entropy_backward(logits, targets)
    activated_gradient, second_weight_gradient, second_bias_gradient = second.backward(
        output_gradient
    )
    hidden_gradient = activation.backward(activated_gradient)
    input_gradient, first_weight_gradient, first_bias_gradient = first.backward(hidden_gradient)

    def objective() -> float:
        return softmax_cross_entropy(
            second.forward(activation.forward(first.forward(inputs))),
            targets,
        )

    gradient_result = check_gradient(
        first_weight_gradient,
        first.weights,
        objective,
        epsilon=1e-7,
        tolerance=1e-7,
    )

    print("ATLAS MODEL LAB // 02 MULTILAYER PRIMITIVES")
    print()
    print(f"hidden activations:\n{activated}")
    print(f"logits:\n{logits}")
    print(f"softmax probabilities:\n{probabilities}")
    print(f"cross-entropy loss: {loss:.6f}")
    print(f"manual input gradient:\n{input_gradient}")
    print(f"second-layer dW:\n{second_weight_gradient}")
    print(f"second-layer db: {second_bias_gradient}")
    print(f"first-layer db: {first_bias_gradient}")
    print()
    print(
        "first-layer dW gradient check: "
        f"{'PASS' if gradient_result.passed else 'FAIL'} "
        f"(max error={gradient_result.maximum_error:.3e}, "
        f"tolerance={gradient_result.tolerance:.1e})"
    )


if __name__ == "__main__":
    main()
