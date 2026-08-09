"""Play with the smallest trainable model in Atlas Model Lab."""

from __future__ import annotations

import numpy as np

from atlas_model_lab import train_linear_unit


def main() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([-1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float64)

    unit, history = train_linear_unit(
        x,
        y,
        steps=300,
        learning_rate=0.05,
    )

    print("ATLAS MODEL LAB // 01 LINEAR UNIT")
    print()
    print("Training examples:")
    for input_value, target in zip(x, y, strict=True):
        print(f"  x={input_value:>4.1f} -> y={target:>4.1f}")

    print()
    print(f"first loss: {history.first_loss:.6f}")
    print(f"final loss: {history.final_loss:.12f}")
    print(f"learned weight: {unit.weight:.6f}")
    print(f"learned bias:   {unit.bias:.6f}")

    prediction = unit.predict(np.array([10.0], dtype=np.float64))[0]
    print()
    print(f"prediction for x=10: {prediction:.6f}")


if __name__ == "__main__":
    main()
