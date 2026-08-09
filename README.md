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

The project keeps infrastructure out of the way. There is no API, deployment, database, hosted model, or production runtime. The point is to understand the mathematics and software mechanics beneath the abstractions used elsewhere in Atlas Systems.

## Start here

Create a virtual environment and install the project.

macOS, Linux, or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/01_linear_neuron.py
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python examples/01_linear_neuron.py
```

The first example learns the relationship `y = 2x + 3` without being given that equation directly.

## Repository layout

```text
atlas-model-lab/
├── examples/
│   └── 01_linear_neuron.py
├── src/
│   └── atlas_model_lab/
│       ├── __init__.py
│       └── linear.py
├── tests/
│   └── test_linear.py
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

The learning sequence is kept in [`docs/ROADMAP.md`](docs/ROADMAP.md). The order is intentional: prove each mathematical layer before adding another abstraction.

## How it fits into Atlas Systems

Atlas Systems already evaluates, retrieves from, and operates language models. This repository works at the opposite end of the stack by rebuilding the mechanics beneath those systems. The first stage covers gradients and optimisation directly with NumPy; later stages progress through multilayer networks, transformer blocks, and GPT-style architecture experiments.

Part of [atlas-systems.uk](https://atlas-systems.uk) · MIT License
