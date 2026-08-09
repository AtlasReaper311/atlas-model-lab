# Repository Decision

## Purpose

`atlas-model-lab` is the experimental home for first-principles machine-learning work in Atlas Systems.

The repository exists to study implementation mechanics that production AI libraries normally abstract away. Its initial scope is a NumPy neural network; later scope includes a directly implemented transformer and GPT-style architecture.

## Intended classification

The intended Atlas Systems classification at creation is:

| Axis | Value |
|---|---|
| Lifecycle | `experimental` |
| Scope | `public` |
| Provenance | `original` |
| Runtime status | non-runtime |

`atlas-infra` remains the classification authority. This file records intent and does not replace the current policy projection.

## Non-goals

The repository does not:

- serve models;
- expose a public API;
- deploy to Cloudflare or another provider;
- store secrets;
- ingest private Atlas data;
- replace Ramone runtime models;
- claim production model suitability;
- publish generated model output automatically.

## Technology

Initial implementation:

- Python 3.12 or newer;
- NumPy;
- pytest;
- Ruff;
- mypy;
- Python package build validation.

PyTorch is deliberately deferred until the transformer stage.

## Data

The first stages use synthetic data and later public learning datasets.

No private personal, employer, healthcare, production telemetry, or credential-bearing data belongs in this repository.

## Cost

The repository is designed for zero incremental monthly cost.

## Release and deployment

There is no deployment path.

`main` represents reviewed source state only. A successful pull request or CI run is not a deployment claim.

## Validation

Repository-native validation is:

```text
python scripts/validate.py
```

The script compiles source, checks formatting and linting, runs strict typing, executes tests, and proves the package builds.

## Retirement

If the experiments become obsolete, the repository can be archived without runtime migration because it owns no production service or authoritative operational data.
