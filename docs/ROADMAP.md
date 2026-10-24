# Atlas Model Lab: engineering and applied-ML roadmap

Status: **planning document**. Last source inspection: **2026-10-24**.
This document is not a runtime inventory, deployment record, architecture decision record (ADR), release approval, or instruction to execute later stages.

## Purpose and destination

Atlas Model Lab has two related but independently useful destinations:

1. **Track A, first-principles ML:** implement a neuron, manual backpropagation, trainable classifiers, optimisation experiments, a tiny Transformer, and a GPT-2-style decoder. Prove each abstraction with reproducible tests before relying on it.
2. **Track B, applied machine perception:** investigate a specialist **Neural State Engine (NSE)** that learns patterns in SPECULAR-CORE telemetry and offers bounded, evidence-backed interpretations to Ramone. Keep the existing LLM responsible for conversation/reasoning; keep SPECULAR telemetry responsible for factual measurements.

Neither track alone proves the other. A trained tiny GPT does not establish that a temporal telemetry detector is effective. A working temporal model does not require first implementing a Transformer.

The public-facing **interactive Model Lab experience** is a third, presentation-oriented output built only on measured or genuinely recorded experiments. It is not a prerequisite for core ML correctness or a substitute for evidence.

Start here for the specialised plan: [SPECULAR Neural State Engine roadmap](SPECULAR-NEURAL-STATE-ENGINE-ROADMAP.md).

## Status, authority and boundaries

| Area | Source-supported status at inspection | Next evidence |
| --- | --- | --- |
| Stage 1: linear unit | Implemented on main | Preserve regression tests |
| Stage 2: multilayer primitives | Implemented on main | Preserve gradient checks |
| Stage 3: bounded synthetic classifier | Implemented on main | Preserve deterministic memorisation gate |
| Stage 4: MNIST | Planned; no Stage 4 implementation identified | Dataset and experiment design, then PR |
| Stages 5-8 | Planned; no implementation claimed | Stage-specific evidence below |
| Interactive Model Lab | Concept only | Design, accessibility, data/publication boundary |
| NSE-0 to NSE-7 | Proposed applied track, not implemented here | Current source/runtime audit and separate approvals |

At inspection, this repository remains experimental, public, original, NumPy-only for its implemented runtime dependencies, and **non-runtime**. The repository decision at [REPOSITORY-DECISION.md](REPOSITORY-DECISION.md) excludes private telemetry, model serving, production credentials, and deployments. This roadmap does not override those boundaries.

Inspect current source, relevant Atlas Infra accepted ADRs and classifications, actual SPECULAR telemetry contracts, current Ramone integration, and runtime/provider evidence before starting cross-repository work. Records below describe verified source *at inspection* or future proposals; do not treat documentation as proof of present hardware access, deployed identity, telemetry availability or model performance.

**Workflow:** inspect current repo and policies -> agree bounded task and authority -> plan exact files -> branch -> implement -> run native validation -> review exact-head diff and checks -> authorised merge -> **separately authorised** install/deploy/tool assignment -> verify running and user-visible behaviour. A green PR is not deployment or live proof (Atlas Infra ADR-0013).

## Dependency diagram

~~~text
TRACK A: FIRST PRINCIPLES
Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 -> Stage 6
                                                   |
                                  +----------------+----------------+
                                  |                                 |
                               Stage 7                           NSE-0/1
                                  |                                 |
                               Stage 8                           NSE-2/3
                                  |                                 |
                        Interactive Model Lab                    NSE-4
                  (may begin earlier with real data)               |
                                                                  NSE-5
                                                                  |
                                                                  NSE-6 -> NSE-7

NSE-0 data discovery can begin independently; NSE-3 needs measured data,
an evaluation design, and appropriate ML competence, not a completed GPT.
NSE-5 depends on evidence from shadow-mode comparisons, not UI work.
~~~

The sequence is an **ordering of dependencies, not an automatic schedule or approval**. Do not displace the estate's mandatory year-end audit/freeze and January handoff to rush a runtime model. Plan GPU-intensive experiments around local resources and other SPECULAR workloads.

## Track A: research-to-implementation gates

### Stage 1: linear unit [IMPLEMENTED]

**Objective:** learn a known linear relationship from examples with one weight and one bias.

**Implementation:** mean squared error, analytical derivatives, gradient descent, deterministic training, small example and tests.

**Exit gate:** a reproducible experiment learns the known relationship and loss falls. Existing implementation and tests form a regression baseline for later stages.

### Stage 2: multilayer primitives [IMPLEMENTED]

**Objective:** implement the mathematics needed for a neural network without automatic differentiation.

**Implementation:** NumPy Dense forward/backward, ReLU, stable softmax, cross-entropy, composed gradients and centred finite-difference gradient checking.

**Exit gate:** analytical versus numerical gradients agree within the documented float64 tolerance, with failures revealing the incorrect derivative. Preserve shape, finite-input and stability tests.

### Stage 3: bounded synthetic classification [IMPLEMENTED]

**Objective:** prove a complete training loop, not generalisation.

**Implementation:** fixed 2 -> 4 -> 2 NumPy classifier on eight two-dimensional examples, manual backward passes, explicit full-batch parameter updates, recorded loss and accuracy.

**Exit gate:** deterministic runs intentionally memorise the bounded dataset and reduce loss. **100% training accuracy on eight examples is not proof of unseen-data performance.**

### Stage 4: MNIST image classification [NEXT]

**Question:** can a classifier trained on examples correctly identify **unseen** handwritten digit images?

**Scope:**
- Obtain MNIST through a documented, licence-aware, integrity-checked acquisition path. Do not commit the dataset to Git unless policy explicitly allows it.
- Implement 28 x 28 preprocessing, 784-feature input representation and a small multilayer perceptron with a fixed, reported architecture.
- Split train/validation/test by a documented policy. Do not evaluate model choices on the held-out test set.
- Implement explicit training, validation and inference; keep Stage 2/3 tests green and preserve deterministic seeding.
- Record initialisation, preprocessing version, model configuration, learning rate, batch size, epochs, elapsed time and library/runtime versions.
- Export evaluation results and reproducible inference evidence without misrepresenting a local file as a deployed model.

**Evidence:** training/validation loss and accuracy per epoch, final held-out evaluation, confusion matrix, representative correct/incorrect predictions, seed/configuration and runtime cost.

**Exit gate:** repository-native CI plus a reproducible local run that can load its own model artifact and classify untouched images; validation and held-out results are labelled separately. Declare acceptance criteria before running comparisons, not after seeing scores.

**Explicit exclusions:** webcam/OCR product, a public API, inference deployment, automatic model promotion, hidden data leakage, new ML framework if a bounded NumPy baseline suffices.

### Stage 5: optimisers

**Question:** how do update algorithms change convergence, stability and compute cost under equivalent conditions?

**Scope:** implement vanilla SGD, momentum and Adam; introduce AdamW only where its weight-decay semantics are meaningfully distinguishable. Separate optimiser state from model/loss code while keeping equations inspectable. Validate first and repeated steps against hand-computed reference cases and deterministic tiny problems.

**Evidence:** matched model/data/seed experiments, learning curves, parameter-update reference tests, per-run configuration and elapsed time.

**Exit gate:** fair, reproducible optimiser comparisons without tuning one method on a different split or claiming an observed winner universally generalises.

### Stage 6: controlled experiments

**Question:** which changes measurably affect convergence and generalisation?

**Scope:** compare pre-registered, bounded matrices across learning rate, batch size, depth, hidden width, activation, initialisation and optimiser; change one factor at a time or explicitly document a factorial design. Repeat seeds; isolate data splits; capture compute cost and failures.

**Evidence:** machine-readable run manifest, environment, seed, source revision, dataset fingerprint, metrics, variability, charts and written interpretation of limits. Detect regression against a chosen accepted baseline.

**Exit gate:** each experiment answers an explicit question, is repeatable, and separates a hypothesis, observed result and unsupported speculation.

**Shared application gate:** after Stage 6, Track B's temporal-model experiment can proceed if data quality and evaluation gates also pass. This is a learning milestone, **not** permission to use private telemetry in the public repository.

### Stage 7: tiny Transformer

**Question:** how does an autoregressive decoder construct next-token predictions?

**Scope:** introduce PyTorch for tensor execution/autodiff while directly implementing token/positional embeddings, Q/K/V projections, masked self-attention, multi-head attention, feed-forward blocks, residuals, normalisation and a bounded sampling loop. Test causal masking, shapes, numerical stability and deterministic generation. Compare selected hand-written reference operations with trusted library primitives.

**Evidence:** attention-mask tests, known tensor fixtures, parameter counts, small training/inference records and a documented model-card boundary.

**Exit gate:** correct causal behaviour and reproducible bounded autoregressive generation. Do not claim GPT equivalence.

### Stage 8: GPT-2-style architecture

**Question:** can a GPT-2-style decoder architecture be reproduced and evaluated on local resources?

**Scope:** implement a documented decoder configuration from primary architecture references; version tokenizer/data, train/evaluate on a legally usable bounded corpus, produce constrained sample generation and measure throughput, memory, overfit behaviour and validation loss.

**Evidence:** architecture/source comparison, model/tokenizer versions, training logs and checkpoints, evaluation protocol, generated examples marked as generated, reproducibility limits, hardware/time/cost.

**Exit gate:** independently reproducible architecture and bounded generation, with honest separation of architecture reproduction, pipeline reproduction and historical-paper-result reproduction.

**Explicit exclusions:** claiming parity with commercial LLMs, deploying a Ramone replacement, or conflating a working demo with general-purpose model quality.

## Interactive Model Lab: public explanation track

**Concept:** a portfolio-grade educational interface that lets a viewer inspect **real, versioned** results from the lab rather than fabricated live training activity.

**Candidate milestones:**
1. **Read-only evidence pages:** show Stage 1-3 training history, gradient checks and explanations from checked artifacts.
2. **MNIST explorer:** draw a digit locally or inspect selected fixed examples; display predictions and per-class scores only from an actually loaded model; label model identity and preprocessing. Browser preprocessing is a measurable train/serve compatibility risk.
3. **Learning and error explorer:** display recorded epochs, train/validation curves, a confusion matrix, prediction failures and run-to-run variance. Playback of recorded training must be labelled **recorded replay**, not live learning.
4. **Experiment comparator:** compare Stage 5-6 configurations under matched dataset, seed policy, measured timing and honest uncertainty.
5. **Architecture explainer:** step through genuine attention masks, token flows and model outputs from Stage 7-8; make a live demo a separate, explicitly authorised operation.

**Design gates:** WCAG-minded keyboard/focus support, text alternatives to charts, reduced motion, mobile/desktop behaviour, payload size, security review, no unapproved model weights or source data published. The public site remains a presentation consumer and does not become the authority for training, evaluation or runtime identity. Inspect atlas-systems interface policy and repository instructions before any site changes.

**Public claim rule:** differentiate simulated diagrams, recorded evaluation, running inference and verified live service. Never show invented accuracy, temperature, incident, benchmark or deployment figures.

## Track B: SPECULAR Neural State Engine

**Objective:** a specialised **machine-perception** model alongside SPECULAR's factual telemetry and existing deterministic anomaly detector. It may classify observed workload state, detect temporal deviations and return uncertainty plus raw evidence references to Ramone. Ramone's LLM remains the language/reasoning/orchestration layer.

**Why separate:** a hardware measurement is a fact from a sensor with a timestamp; a neural label such as "thermal pressure" is a probabilistic **interpretation**. One must never replace or silently overwrite the other.

**Approved planning direction, not approved implementation architecture:** the full proposal, candidate models, data boundaries, owner map and eight gated milestones are in [SPECULAR-NEURAL-STATE-ENGINE-ROADMAP.md](SPECULAR-NEURAL-STATE-ENGINE-ROADMAP.md).

| Milestone | Output | Prerequisite and gate |
| --- | --- | --- |
| NSE-0: source/runtime data audit | Actual signal inventory, sampling, missingness, security and ownership map | Fresh repository and runtime inspection |
| NSE-1: private dataset foundation | Bounded local recording, labels, provenance, versioned datasets | Consent/privacy, retention and capture review |
| NSE-2: evaluation and baselines | Detector baseline, simple ML baselines, pre-registered scoring protocol | Held-out, time-separated real sessions |
| NSE-3: specialist model selection | Reproducible classical/ML/temporal comparison | Evidence justifies model choice, not preference for GRU |
| NSE-4: shadow inference | Local neural and existing detector side by side | Bounded overhead, stale/missing-data behaviour, no authority transfer |
| NSE-5: Ramone perception tool | Authenticated read-only machine_state capability | Proven tool owner, grounding, fallback and source references |
| NSE-6: governed rollout | Versioning, evaluation approval, separate local install and runtime verification | Operator rollout approval and rollback rehearsal |
| NSE-7: optional expansion | More states, transitions, embeddings, history and optional visual/sonic surfaces | Independent evidence of added value |

The first useful deployment, if later approved, is deliberately modest: a few **validated** state labels and an honest uncertain/unavailable state. Rich categories such as gaming versus rendering, or inference versus unrelated GPU load, cannot be asserted from the current five anomaly metrics alone.

### Ramone and estate boundaries

~~~text
SPECULAR-CORE sensors
        |
SPECULAR telemetry collector  <- factual values + timestamps
        |
        +-- existing deterministic anomaly detector
        |
        +-- proposed local neural inference, initially shadow-only
                         |
                interpreted state + evidence IDs + uncertainty
                         |
              governed, read-only machine_state()
                         |
                       Ramone
              (LLM + existing memory + corpus)
~~~

No GRU replaces Ramone's LLM. No neural model replaces the collection of raw sensor readings. No neural verdict may be promoted to a confirmed hardware fault, causal explanation or production health fact solely because a classifier produced it.

The public experimental lab must not ingest raw private SPECULAR data; any private recording, service integration, retraining, provider mutation, release or deployment requires separate reviewed implementation and authority in its eventual owning repository.

## Programme management and review gates

- **Start immediately next:** Stage 4 MNIST, as a separate PR; run native validation and held-out-data gate.
- **Parallel planning only:** NSE-0 source/runtime audit may begin after separately agreed scope, without recording private telemetry or changing hardware collectors.
- **After Stage 6:** begin NSE-1/2/3 experiments when measurement and retention are approved; Transformer development can proceed independently if resources permit.
- **After observed shadow success:** propose NSE-5/6 integration and rollout under a new architecture/contract review. Do not automatically retire the existing detector.
- **Optional presentation:** build the interactive lab on validated static experiment artifacts before adding any live integration.
- **Stop or simplify when:** collection quality is inadequate, ground truth cannot be obtained, the simple baseline matches/defeats neural results, false alarms are excessive, or inference disrupts SPECULAR/LLM workloads.

For each milestone retain: exact base/head, task authorization, owners, changed files, commands and validation, experiment/fixture provenance, model/data versions, metrics with limitations, review outcome, delivery stage, separately observed runtime stage if any, and the next independently required evidence.

## Sources and governing references

- [Repository decision](REPOSITORY-DECISION.md)
- [SPECULAR Neural State Engine long-term roadmap](SPECULAR-NEURAL-STATE-ENGINE-ROADMAP.md)
- [SPECULAR telemetry repository](https://github.com/AtlasReaper311/specular-telemetry), source evidence for sampling and existing detector
- [Ramone RAG runtime](https://github.com/AtlasReaper311/ollama-rag-kit), potential orchestration integration; live assignment not inferred
- [Atlas evaluation harness](https://github.com/AtlasReaper311/atlas-eval-harness), language-model-specific harness needing compatibility review
- [Atlas Infra ADRs](https://github.com/AtlasReaper311/atlas-infra/tree/main/docs/adrs), particularly ADR-0013 and ADR-0019
- [Atlas Systems 2026 completion roadmap](https://github.com/AtlasReaper311/atlas-infra/blob/main/docs/2026-completion-roadmap.md), independent estate programme

**Reinspection rule:** these links identify owners and source locations, not present provider/runtime state. Source changes do not authorise downstream deployments.
