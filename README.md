<div align="center">
  <img src="https://raw.githubusercontent.com/AtlasReaper311/AtlasReaper311/main/atlas-icon-dark-256.png" width="88" alt="Atlas Systems"/>
</div>

# atlas-model-lab

```text
┌─────────────────────────────────────────────┐
│  ATLAS SYSTEMS // atlas-model-lab           │
│  neural networks from first principles      │
└─────────────────────────────────────────────┘
```

![Python](https://img.shields.io/badge/python-3.12-f5a623?style=flat-square&labelColor=0a0a0f)
![NumPy](https://img.shields.io/badge/numpy-2.5%2B-aaa9a0?style=flat-square&labelColor=0a0a0f)
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-aaa9a0?style=flat-square&labelColor=0a0a0f)
![Tests](https://img.shields.io/badge/tests-pytest-4ade80?style=flat-square&labelColor=0a0a0f)
![Cost](https://img.shields.io/badge/cost-%C2%A30-aaa9a0?style=flat-square&labelColor=0a0a0f)

`atlas-model-lab` is an experimental learning repository for implementing neural networks and transformer models from first principles. The first lab is deliberately tiny: one trainable linear unit that learns a relationship from examples using gradient descent.

Stage 2 adds the multilayer-network primitives needed to make manual backpropagation observable. Stage 3 composes those primitives into a deterministic two-layer classifier that can memorise a bounded synthetic dataset. The implementation remains NumPy-only.

The project keeps infrastructure out of the way. There is no API, deployment, database, hosted model, or production runtime. The point is to understand the mathematics and software mechanics beneath the abstractions used elsewhere in Atlas Systems.

## Start here

Create a virtual environment and install the project.

macOS, Linux, or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/01_linear_neuron.py
python examples/02_multilayer_primitives.py
python examples/03_synthetic_classification.py
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python examples/01_linear_neuron.py
python examples/02_multilayer_primitives.py
python examples/03_synthetic_classification.py
```

The first example learns the relationship `y = 2x + 3` without being given that equation directly.

## Repository layout

```text
atlas-model-lab/
├── examples/
│   ├── 01_linear_neuron.py
│   ├── 02_multilayer_primitives.py
│   └── 03_synthetic_classification.py
├── src/
│   └── atlas_model_lab/
│       ├── __init__.py
│       ├── classification.py
│       ├── gradient_check.py
│       ├── layers.py
│       ├── linear.py
│       └── losses.py
├── tests/
│   ├── test_classification.py
│   ├── test_gradient_check.py
│   ├── test_layers.py
│   ├── test_linear.py
│   └── test_losses.py
├── docs/
│   ├── REPOSITORY-DECISION.md
│   └── ROADMAP.md
├── scripts/
│   ├── publish-repository.sh
│   ├── publish-repository.ps1
│   └── validate.py
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       ├── ci.yml
│       ├── codeql.yml
│       └── scorecard.yml
├── pyproject.toml
└── LICENSE
```

## What the first lab contains

The model has only two trainable values:

```text
prediction = weight * x + bias
```

Training repeatedly:

1. predicts an output;
2. measures mean squared error;
3. calculates the gradients for the weight and bias;
4. moves both parameters a small amount in the direction that reduces error.

The implementation is intentionally readable before it is general.

## What Stage 2 contains

Stage 2 is a first-principles forward and backward pass for a tiny multilayer network:

- `Dense` computes `Y = X @ W + b` and its manual `dX`, `dW`, and `db` formulas;
- `ReLU` keeps positive values and defines its derivative as zero at exactly zero;
- `softmax` subtracts each row's maximum logit before exponentiation, preventing overflow for extreme logits;
- `cross_entropy` measures the negative log probability assigned to the correct class;
- `softmax_cross_entropy` uses a stable log-sum-exp calculation for logits;
- `softmax_cross_entropy_backward` exposes the simplified `(probabilities - one_hot_targets) / batch_size` derivative;
- `check_gradient` compares those analytical derivatives with centred finite differences.

Finite-difference checking is useful evidence because it estimates the slope of the complete forward calculation by nudging one value at a time. When that independent estimate agrees with the hand-written derivative, the backward formula is being checked against the behaviour it is meant to describe.

The Stage 2 example performs one forward pass, one manual backward pass, and a gradient check:

```bash
python examples/02_multilayer_primitives.py
```

## What Stage 3 contains

Stage 3 turns the Stage 2 primitives into a complete learning loop.

The bounded experiment uses eight deterministic two-dimensional examples and a fixed `2 -> 4 -> 2` network. Each training update:

1. runs a forward pass through Dense, ReLU, and Dense layers;
2. measures softmax cross-entropy loss;
3. backpropagates the loss through the hand-written Stage 2 derivatives;
4. applies explicit full-batch gradient descent to weights and biases;
5. records the resulting loss and accuracy.

The purpose is intentional overfitting. Memorising all eight examples proves that the forward path, loss, backward path, and parameter updates work together as one trainable system. It is not evidence of generalisation and it is not a production classifier.

Run the Stage 3 experiment with:

```bash
python examples/03_synthetic_classification.py
```

## Validation

After installing the development dependencies:

```bash
python scripts/validate.py
```

The validation script runs:

- Python bytecode compilation;
- Ruff linting;
- Ruff formatting checks;
- mypy strict type checking;
- pytest;
- package build verification.

CI runs the same validation on pull requests and pushes to `main`.

## Boundaries

This repository is:

- experimental;
- public;
- original work;
- non-runtime;
- local-only;
- free to run with the existing development environment.

It does not contain secrets, production credentials, model-serving infrastructure, private training data, or automatic deployment.

The intended public classification must be registered through current `atlas-infra` policy after repository creation. This README does not replace that authority.

## Roadmap

The full learning and applied-ML programme is in [docs/ROADMAP.md](docs/ROADMAP.md). Stages 1-3 are implemented; Stage 4 (MNIST) is next. Later stages cover optimisers, controlled experiments, a tiny Transformer and a GPT-2-style decoder. The roadmap specifies data provenance, reproducibility and acceptance gates rather than treating a successful training run as production suitability.

A separate [SPECULAR Neural State Engine roadmap](docs/SPECULAR-NEURAL-STATE-ENGINE-ROADMAP.md) describes a proposed long-term application: a small, evidence-backed model that interprets machine telemetry over time for Ramone while **preserving raw SPECULAR readings and the existing deterministic detector**. The proposed model would not replace Ramone's LLM. It is **planning only**: no training data capture, private data, inference service, tool integration, model deployment or live verification is implemented by these documents.

The roadmap also describes a possible interactive Model Lab experience using real recorded experiment results, explicit evidence labels and accessible comparisons. That public experience is optional presentation work, not a claim of present availability.

## How it fits into Atlas Systems

Atlas Systems already evaluates, retrieves from, and operates language models. This repository works at the opposite end of the stack by rebuilding the mechanics beneath those systems. Stages 1 through 3 now cover scalar optimisation, manual multilayer derivatives, and a complete deterministic classifier training loop; later stages progress through reproducible dataset experiments, optimisers, transformer blocks, and GPT-style architecture experiments.

Part of [atlas-systems.uk](https://atlas-systems.uk) · MIT License
