"""Train the bounded Stage 3 classifier and show its learning evidence."""

from __future__ import annotations

from atlas_model_lab import TinyClassifier, tiny_classification_dataset, train_tiny_classifier


def main() -> None:
    inputs, targets = tiny_classification_dataset()
    initial_model = TinyClassifier()
    initial_loss, initial_accuracy = initial_model.evaluate(inputs, targets)

    model, history = train_tiny_classifier(inputs, targets)
    predictions = model.predict(inputs)

    print("ATLAS MODEL LAB // 03 SYNTHETIC CLASSIFICATION")
    print()
    print(f"initial loss:     {initial_loss:.6f}")
    print(f"final loss:       {history.final_loss:.6f}")
    print(f"initial accuracy: {initial_accuracy:.0%}")
    print(f"final accuracy:   {history.final_accuracy:.0%}")
    print(f"updates:          {history.steps}")
    print()
    print("sample predictions:")
    for inputs_row, target, prediction in zip(inputs, targets, predictions, strict=True):
        print(
            f"  x=({inputs_row[0]:.2f}, {inputs_row[1]:.2f}) "
            f"target={target} prediction={prediction}"
        )


if __name__ == "__main__":
    main()
