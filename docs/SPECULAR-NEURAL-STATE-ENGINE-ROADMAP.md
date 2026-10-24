# SPECULAR Neural State Engine: long-term implementation roadmap

**Programme:** Atlas Model Lab, applied machine perception
**Status:** proposed, documentation only
**Inspection baseline:** 2026-10-24; source-only observations, not live measurements
**Governing overview:** [Atlas Model Lab roadmap](ROADMAP.md)
**Decision boundary:** this file proposes a capability and stages. It is not an accepted Atlas Infra ADR, change approval, service contract, model promotion, training-data authorisation, release instruction or live-verified operating procedure.

## 1. Outcome and principles

The intended outcome is a small, specialised machine-state interpretation service for **SPECULAR-CORE**. It receives factual readings from existing SPECULAR telemetry, interprets their time evolution, and exposes a bounded, explainable **read-only machine_state()** capability to Ramone. Ramone uses its existing language/reasoning model to communicate what the specialist detected, optionally reads supporting measurements, and maintains explicit uncertainty.

This is **not** a GRU replacing the Ramone LLM. Ramone's LLM continues doing language, tool use, retrieval and reasoning. It is **not** a replacement telemetry collector. The neural system starts as a parallel interpretation path and does not supersede the current deterministic detector without separate evidence and an operator decision.

Non-negotiable invariants:

1. **Measurements remain facts, model outputs remain hypotheses.** Never alter or replace measured sensor samples with a prediction.
2. **Local-first and private by default.** Private raw PC telemetry, associated session labels, host/process data and training artifacts stay off the public repo and public edge unless explicitly sanitised and approved.
3. **Provenance and freshness.** Every prediction binds to exact input-window timestamps, input schema, data quality, model identity and available source evidence.
4. **Fail closed on uncertainty.** Missing, stale, invalid or insufficient history must not yield an apparently confident "normal".
5. **Compare rather than assume.** The current detector is the operational baseline. A neural approach must demonstrate measurable value for named tasks.
6. **Resource coexistence.** Training/inference cannot be allowed to disrupt SPECULAR's current local LLM workloads, telemetry sampler, Ramone or other services.
7. **Separated authority.** Source merge, evaluation approval, installing weights, assigning a Ramone tool and live rollout are separate decisions and evidence stages.
8. **No autonomous control in v1.** Read-only classification and explanation support only; no fan curves, clock/power changes, process killing or alert/automation authority.

## 2. Actual architecture inspected versus future hypothesis

### 2.1 Current source-supported observations

| Component | Verified from current source | Gap or qualification |
| --- | --- | --- |
| specular-telemetry sampler | Local FastAPI gathers CPU, GPU, RAM and Ollama snapshots. Default sampling interval is 30 seconds. | Do not assume the service is running now or the effective environment uses the default. |
| CPU collector | Overall/per-core utilisation; reported current/max frequency and core counts. | No CPU temperature in the inspected collector. Frequency is not itself proof of throttling. |
| GPU collector | NVIDIA NVML name, VRAM used/total, utilisation and temperature when available. | No GPU power, clocks or fan speed in current source; GPU may be null when NVML is unavailable. |
| RAM collector | Used/total GB and percentage. | No per-process allocations or memory-pressure event source. |
| Ollama collector | Reachability, models available and currently loaded. | Loaded model is not proof that a model is actively generating tokens; unavailable is an expected state. |
| deterministic anomaly detector | Five monitored paths: cpu.overall_pct, ram.pct, gpu.utilisation_pct, gpu.temperature_c and gpu.vram_used_mb. Robust shape features, constrained derivative DTW, continuity decoding and persisted bounded state. | Verdicts normal/watch/warning/critical and anomaly scores are *algorithm outputs*, not ground-truth fault labels or calibrated fault probabilities. |
| local anomaly endpoints | GET /anomaly and GET /anomaly/history; default local history cap 192 responses. | History is detector evidence, not yet a versioned ML dataset. |
| public edge | GET /specular, /specular/anomaly and /specular/anomaly/history via Worker; offline last-known-good state exists. | Edge cache and KV history cannot be assumed to be a complete, uniformly sampled training series. |
| atlas-model-lab | Stage 1-3 implemented; experimental, public, non-runtime. | This repository's decision explicitly prohibits private Atlas telemetry and model serving. |
| Ramone path | ramone-edge is a public gateway; ollama-rag-kit implements local retrieval/generation; ramone-memory is a separate memory boundary. | Verify the *actual* active orchestrator, any Open WebUI tools, Home Assistant or voice integration on the target machine before choosing the machine_state tool owner. |
| atlas-eval-harness | Private capability-scoped Ollama/generative-model evaluation and source-level model-promotion evidence. | Current scoring/promotion is not automatically suitable for time-series classification. A compatible extension or distinct governed ML evaluator requires review. |

Sources to recheck when starting: [specular-telemetry](https://github.com/AtlasReaper311/specular-telemetry), [ollama-rag-kit](https://github.com/AtlasReaper311/ollama-rag-kit), [ramone-edge](https://github.com/AtlasReaper311/ramone-edge), [ramone-memory](https://github.com/AtlasReaper311/ramone-memory), [atlas-corpus](https://github.com/AtlasReaper311/atlas-corpus), [atlas-eval-harness](https://github.com/AtlasReaper311/atlas-eval-harness), [atlas-infra](https://github.com/AtlasReaper311/atlas-infra). Current accepted decisions and estate-policy inputs override this roadmap.

### 2.2 Intended logical data flow

~~~text
                 SPECULAR-CORE hardware
                           |
                    Sensors / OS APIs
                           |
                    SPECULAR telemetry
               raw immutable observation stream
                           |
                 +---------+-----------+
                 |                     |
       deterministic detector    historical recording
        (existing authority)      (private, approved)
                 |                     |
                 |              versioned train/eval data
                 |                     |
                 |               candidate ML models
                 |                     |
                 +--------+------------+
                          |
                  side-by-side evidence
                          |
                 future local inference
                (read-only, fallible view)
                          |
          machine_state(): provenance and uncertainty
                          |
                        Ramone
               existing LLM and tool routing
                          |
             bounded, evidence-backed answer
~~~

The first neural consumer takes **post-collection snapshots**. It never sits in the sensor acquisition path. Initially it must not generate alerts, mutate detector state, affect raw sample retention, publish public telemetry, trigger control operations or cause the LLM to make unsupported causal claims.

### 2.3 Separation of responsibilities

- **Sensory layer:** what did the hardware/OS collector return, at what time, with what missing fields and provenance?
- **Deterministic layer:** what does the existing shape-aware detector report under its current rule set?
- **Neural layer:** how does the recent observed sequence compare with modelled known states? How uncertain is the classifier or anomaly score?
- **Reasoning layer:** what can Ramone responsibly conclude, when should it abstain, and which raw observations support the explanation?
- **Memory/knowledge:** ramone-memory and atlas-corpus remain their own owners; do not reclassify model interpretations as evergreen knowledge or unqualified memories.

## 3. Current telemetry capability and required gap closure

### 3.1 Candidate signals and evidence strength

| Signal | Source state at inspection | Potential contribution | Claim limitation |
| --- | --- | --- | --- |
| CPU overall utilisation | Existing | CPU-heavy patterns | Alone cannot identify the responsible workload |
| CPU per-core utilisation | Existing | Thread saturation and changing parallelism | Depends on sampling and core topology |
| CPU frequency | Existing | Temporal frequency changes | Not evidence of throttling without context |
| CPU temperature | Not observed in collector | Thermal-pressure evidence | Must first verify safe supported source |
| GPU utilisation | Existing if NVML works | Sustained GPU load | Does not by itself distinguish games, renders or inference |
| GPU temperature | Existing if NVML works | Thermal trend | High value alone does not prove a fault |
| GPU VRAM use/total | Existing if NVML works | VRAM saturation | Memory allocated is not necessarily active processing |
| GPU clocks | Not observed in collector | Clock drop under sustained demand | Correlate with power/temperature and throttle reasons |
| GPU power draw | Not observed in collector | Load versus energy trend | Driver/permission support requires testing |
| GPU fan speed | Not observed in collector | "Why are my fans loud?" question | Sensor may report null, manual fan modes may exist |
| RAM utilisation | Existing | Memory pressure trend | Percentage alone does not prove swapping or OOM |
| Disk I/O and pressure | Not observed in inspected snapshot | Storage bottleneck interpretation | Sampling/units and attribution need validation |
| Network activity | Not observed in inspected snapshot | Data transfer and burst patterns | Avoid collecting payloads, IPs or identifiers |
| Ollama loaded/reachable | Existing | Context signal about local inference availability | Loaded != actively inferring or generating |
| Workload/session label | Not present as training truth | Separates gaming, render, AI and idle | Requires operator notes or separately approved observation |

**NSE-0 must measure actual source availability on the target OS/WSL/driver and chosen privacy boundary** before adding sensors. Do not widen collection just to make a demo label plausible. Prefer raw values plus quality flags over fabricated zeroes.

### 3.2 Temporal sampling questions

The existing local default is 30 seconds/sample. At that default, a 24-sample window covers around 12 minutes of nominal time, not a guaranteed 12-minute continuous record. A missing sample, restart, sleep or effective interval override breaks that equivalence.

Before model selection:
- inspect effective sample intervals and monotonic/UTC time handling;
- measure dropouts, duplicated/out-of-order samples, host sleep and reboots;
- distinguish observed offline from historical last-known-good snapshots;
- report window duration, actual point count, coverage and gap distribution;
- decide whether native 30-second cadence suffices for target events; faster acquisition is a separate collector/runtime authority decision;
- define resampling, interpolation and imputation rules without concealing missingness;
- avoid treating edge cache snapshots as an independent high-frequency sensor series.

## 4. Ground truth, datasets and security

### 4.1 Dataset ownership and collection

Raw historical telemetry belongs in an **approved private local data store**, not this repository or an automatically public project artifact. The dataset recorder should be proposed to the current telemetry/service owner after a bounded source/runtime audit. Stage 1-8 public examples remain synthetic or legitimately licensed learning data.

Record, at minimum, in a reviewed private schema:
- event/sample timestamp in UTC and acquisition timestamp where different;
- device/sensor family identity using safe stable identifiers, source schema/version and collector build ID;
- measured feature names, units and original values plus explicit null/missingness;
- source availability, sampling errors, gaps, sampler interval and host sleep/reboot boundaries;
- workload session ID, operator-confirmed label if supplied, label source and adjudication status;
- optional detector version/verdict for comparison, **never as ground truth by default**;
- dataset/feature-transform version, normalisation parameters and integrity fingerprint.

Do not log secrets, raw requests/prompts, chat bodies, email bodies, filenames, window titles, usernames, browsing activity, personal IPs or per-process metadata by default. If any such signal later becomes genuinely necessary, perform a separate privacy/security review and minimise or aggregate it. Prevent raw payloads from leaking into public CI logs, checkpoints, exceptions, metrics or the portfolio.

### 4.2 Labelling plan

1. **Operator-labelled normal sessions:** deliberate idle, CPU load, GPU load, actual gaming, actual local-model inference or rendering, with start/end windows and checks for confounders.
2. **Observed abnormal sessions:** genuine, independently corroborated degradation, sensor faults or resource pressure with timestamps and source of confirmation.
3. **Unknown/unlabelled:** no inference of "abnormal" from the old detector's warning output alone.
4. **Safe fixtures:** synthetic series for unit tests and boundary simulations, strictly labelled as simulated and excluded from claims of real-world performance.
5. **Rare events:** if thermal throttling or failure examples do not occur naturally, abstain on those labels or use only separately governed non-production experiments. Never manufacture failures and later call them real-world validation.
6. **Weak labels:** may support exploratory research, but must be separately marked and must not contaminate the independently reviewed test labels.

An initial NORMAL/ABNORMAL task can be useful only if ABNORMAL has a defensible definition and reference examples. Otherwise start with known workload states plus **UNKNOWN**, and anomaly *scores* that are not presented as calibrated probabilities.

### 4.3 Dataset validity

- Split **by independent session/time block** before constructing overlapping windows; adjacent overlapping windows must never cross train/validation/test boundaries.
- Include machine-off, sleep, collector outage and null GPU observations as explicit **data availability** states rather than abnormal hardware labels.
- Version feature construction, scaling, category schema, code revision and data fingerprints.
- Fit scaling and thresholds only on training data. Lock the test set before tuning and predeclare success criteria.
- Track class imbalance and label disagreement. Report class-specific recall and false positive rates, not just aggregate accuracy.
- Identify hardware/driver/config drift. A model trained on one machine is not automatically portable to different GPU generations.
- Retain only what is needed for the approved experiment. Define retention/rotation, encryption, backup scope and irreversible deletion before collecting.
- Keep private artifacts, model weights and metadata behind a reviewed export gate; trained weights may encode properties of the private training data.

**Hard stop:** if safe retention, training labels, privacy boundary or independent evaluation cannot be established, do not proceed to deployment.

## 5. Candidate modelling approaches and selection

Do not select a GRU simply because it is a recurrent neural network. Use task-appropriate, reproducible baselines first.

| Candidate | Strength | Main weakness / gate |
| --- | --- | --- |
| Existing SPECULAR detector | Real current baseline; deterministic, low dependency burden, temporal shape detection | Its heuristic score is not calibrated fault probability or independent ground truth |
| Fixed thresholds/state machine | Transparent first comparator for known thermal or resource conditions | Poor adaptation to complex temporal variation |
| Simple linear/logistic or small MLP | Fast, explainable baseline over window summary features | Limited raw sequence modelling; test before increasing complexity |
| Classical anomaly scoring on windows | Can operate with abundant normal data and few true failures | Evaluation against real events and false alarms is difficult |
| Autoencoder | Learns representations of frequent patterns | Reconstruction error may ignore faults or flag benign changes; requires independent validation |
| GRU/LSTM | Captures sequence dependencies and variable temporal context | More parameters/training cost; not inherently better than engineered features |
| Temporal convolutional network | Parallel temporal pattern modelling with bounded receptive field | Receptive field/window choice must match event duration |
| Hybrid | Rule-supported safety labels plus learned workload/anomaly interpretation | Clear precedence needed; never disguise a heuristic output as model probability |

**Comparison protocol:** same held-out sessions, comparable input evidence, pre-registered metrics, fixed seed policy, repeated runs and measured resource cost. A more complex model proceeds only if it improves a meaningful endpoint without violating latency/privacy/CPU/GPU limits.

### 5.1 Desired outputs: distinguish types

- **Workload class:** e.g. idle, heavy GPU usage, CPU-heavy, unknown. Labels must be observable and independently distinguishable.
- **Anomaly score:** a bounded ranking/severity or deviation measure; it is **not** a probability unless calibrated and evaluated as one.
- **Class confidence:** only if calibration is measured on representative held-out sessions. Softmax alone is not a reliability certificate.
- **Thermal status:** only if independently supported by appropriate measurements, not inferred from high GPU usage alone.
- **Memory pressure:** requires evidence beyond a single percentage where a stronger claim is made.
- **Recent transitions:** time-stamped transitions with hysteresis or persistence validation; do not call one noisy sample a state change.
- **Latent vector:** optional future internal representation; not an interpretation, diagnosis or automatically public interface.

State taxonomy must be **versioned** and derived from available signals, real workload examples and user value. Labels such as GAMING, RENDERING, AI_INFERENCE and THERMAL_PRESSURE are hypotheses until validated. The first useful classifier can expose a very small set of supportable states.

## 6. Evaluation and acceptance protocol

### 6.1 Predeclare evaluation tasks

| Evaluation task | Evidence and measurements |
| --- | --- |
| Known workload classification | Macro-F1, per-class recall/precision, confusion matrix, unknown/abstain frequency |
| Anomaly detection | Precision/recall for adjudicated incidents, false alarms per operating hour, time-to-detect, missed events |
| State transitions | Detection delay, state flapping, unstable transitions, minimum sustained duration |
| Prediction uncertainty | Calibration error/reliability diagram, abstention quality, out-of-distribution behaviour |
| Missingness/staleness | Correct unavailability/unknown result; no false "normal" |
| Robustness | Null GPU, restart, clock skew, sample gaps, changed sampling rate, sensor spikes |
| Resources | CPU/RAM/VRAM footprint, inference latency p50/p95, GPU contention, startup/stop time |
| Ramone interpretation | Grounding to raw evidence, no unsupported causal assertions, correct abstention, tool error handling |

For each capability write down thresholds and allowable regression **before** candidate comparisons. Do not invent target pass rates, latency budgets or historical fault counts in this roadmap. The approved runtime owner must establish budgets from the current machine and workloads.

### 6.2 Comparator design

Run the existing detector and candidates against the **same** time-aligned windows. Record both outputs and source identities. Do not train candidate labels from the existing detector and then claim independent superiority because they agree. Evaluate with blinded/adjudicated ground truth where possible; say explicitly when no reliable anomaly labels exist.

Record disagreements and trace them back to input windows. Compare per-task performance rather than producing one global "AI better" score. Test whether a simple ruleset achieves the user-facing value before retaining a temporal model.

### 6.3 Model identity, reproducibility and promotion

Every candidate run should bind:
- exact training/evaluation source revision and environment;
- feature/schema/scaling and dataset fingerprints;
- training parameters and seeds;
- architecture, weights/checkpoint digest, preprocessing contract and library versions;
- evaluation split and benchmark protocol;
- signed-off acceptance/rejection rationale and known regressions;
- runtime compatibility and rollback information where relevant.

The current private atlas-eval-harness evaluates **LLM-generated answers** and has model promotion evidence. It is not a drop-in time-series classifier evaluator. Investigate whether to extend its governance abstractions via a separate reviewed interface or keep a dedicated ML evaluator and project only safe summary evidence. Any new promotion contract belongs under current Atlas Infra authority.

**Promotion approved is not installed, deployed, assigned to Ramone or live verified.** Runtime identity must be independently observed after an authorised install.

## 7. Proposed machine_state tool contract

This is an **illustrative response**, not a published schema or live API:

~~~json
{
  "schema": "specular-machine-state/proposal-v0",
  "availability": "available",
  "generated_at": "example-timestamp-only",
  "input_window": {
    "start": "example-timestamp-only",
    "end": "example-timestamp-only",
    "observed_samples": 24,
    "coverage": "measured-by-collector"
  },
  "machine_state": "gpu_heavy",
  "confidence": null,
  "anomaly_score": null,
  "thermal_state": "unknown",
  "memory_state": "unknown",
  "model": {
    "identifier": "example-candidate",
    "version": "example-version"
  },
  "supporting_evidence": [
    { "ref": "local-read-only-evidence-reference" }
  ],
  "interpretation_status": "model_prediction_not_measurement"
}
~~~

All values above are **illustrative placeholders**, not actual telemetry, measured confidence or approved identifiers. The actual contract should reject unknown fields, specify units/ranges/allowed enums, cap arrays, validate UTC timestamps, carry freshness TTL and explicit data-quality status, and prevent direct exposure of private local paths or endpoint names to untrusted/public consumers.

Required operations, subject to interface design:
- **machine_state()**: bounded latest interpretation, state confidence only when calibrated, status (available/stale/unavailable/unknown), input window and evidence references.
- **machine_state_history(window)**: later optional controlled bounded state-transition history after retention and privacy review.
- **read_raw_evidence(reference)**: privileged, read-only, allowlisted lookup into original timestamped SPECULAR readings; no arbitrary file/path or URL fetch.

**Ramone grounding rule:** if output is unknown/unavailable/stale, Ramone says so. For a thermal or memory explanation, cite or report actual relevant sensor values with timestamps, distinguish observed correlations from diagnosis, and avoid asserting thermal throttling without suitable indicators. Never claim "nothing is wrong" without defining what sensors, time window and fault classes were evaluated.

### 7.1 Security and orchestration boundary

- Keep v1 local/internal. **Do not** publish a raw machine_state endpoint via the public telemetry Worker by default.
- Select the actual active tool/orchestrator (Ramone RAG service, Open WebUI, Home Assistant/voice or approved connector) only after current runtime reinspection. ramone-edge is a gateway, not proof of where private tools execute.
- Authenticate/authorise all local tool access with approved mechanisms; never put credentials in source or prompts.
- Allowlist tool routes and result fields, bound query windows and response size, enforce rate/timeout budgets; avoid server-side tool-generated outbound fetches.
- Treat both ML prediction text and source metadata as untrusted tool data. Tool output may inform Ramone, never instruct it to bypass authority.
- Separate public Ramone questions from privileged host-specific introspection. Authorise per surface; do not assume public users may query private PC behaviour.
- Prevent observability/LLM transcripts from automatically storing detailed raw readings, private host identities or model checkpoints.

### 7.2 Failure and fallback table

| Condition | Required safe behaviour |
| --- | --- |
| No fresh sample / computer off | Return unavailable/stale with last observation explicitly historical, never present cached state as current |
| Insufficient warm-up / history | Return unknown/insufficient_history; optionally show raw latest values separately |
| Missing GPU/fan/clock signals | Mark unsupported fields unknown; avoid thermal or throttling claims requiring them |
| Wrong schema or corrupt sequence | Reject inference or downgrade to unavailable with reason code and evidence for operator |
| Model service timeout/crash/OOM | Telemetry collector remains independent; existing detector continues; no forced restart loop |
| Model disagreement with deterministic detector | Preserve both readings and a disagreement indicator; do not silently suppress either |
| Model out of distribution | Abstain or return uncertain; no forced best-class assertion |
| Ramone tool routing failure | Ramone explains it cannot verify current machine state, may use separately authorised raw telemetry |
| Resource contention with LLM | Enforce CPU/latency budgets or stop neural inference; LLM and telemetry take priority |
| Candidate regression after upgrade | Restore previously accepted model or disable neural tool; verify fallback and actual running identity |
| Prediction suggests safety-critical hardware fault | Surface as a conditional indicator; require independent measurements/operator triage |

## 8. Programme milestones, exact deliverables and stop gates

### NSE-0: authoritative inspection, telemetry suitability and ownership

**Inputs:** current SPECULAR collector/anomaly sources, actual SPECULAR-CORE operating system/drivers/sampling environment, existing persistence and edge contracts, Ramone tool routing, Atlas Infra classification/ADRs. Current source inspection is a navigation baseline, not live proof.

**Actions:**
1. Make a matrix of available fields, units, missingness, update frequency, safe access and source authority.
2. Inspect raw local history and present detector persistence; identify why the existing 512 metric samples / 192 verdict events do not constitute an independent long-term training corpus.
3. Verify whether clocks, power, fans, CPU temperature and disk/network stats are accessible safely; do not implement collectors yet.
4. Map data sensitivity and deployment ownership and identify any required ADR/schema changes.
5. Define initial useful questions Ramone should answer and which observations are sufficient to answer them.
6. Document resource constraints and LLM co-scheduling on the *actual* target hardware/runtime.

**Deliverables:** dated source/runtime audit, versioned candidate feature inventory, owner/authority map, collector enhancement proposals, privacy review inputs and go/no-go for collection.

**Exit gate:** measurements and data gaps demonstrated, initial tasks bounded, no unjustified labels or live claims. **Stop** if signal access is insufficient for credible answers.

### NSE-1: private recording and labelled sessions

**Dependencies:** NSE-0 and approval of collection/storage changes.

**Actions:**
1. Define a minimal versioned private record schema and local append-only recording approach separate from the public edge.
2. Add hard retention/rotation, storage/CPU cost controls, secure local access and opt-in/operator-started labelled sessions.
3. Build deterministic parsing/quality validators: duplicates, gaps, clock changes, missing sensors and range/units.
4. Record representative normal workload sessions without collecting user content; mark session start/end and concurrent workloads.
5. Create clean train/validation/test splits at session level with hashes and feature transforms.
6. Create separate synthetic tests for failure handling, clearly marked simulated.

**Deliverables:** private versioned dataset(s), collection runbook and provenance report, sample-quality and label audits, retention/cleanup verification.

**Exit gate:** reproducible training and genuinely held-out evaluation data, authorised private handling, no repository leakage. **Stop** if labels or privacy controls cannot be trusted.

### NSE-2: comparator and pre-registered evaluation

**Dependencies:** NSE-1 real data and valid labels for each asserted task.

**Actions:**
1. Replay identical windows through the current deterministic detector.
2. Implement transparent rules and fixed-feature/simple-classifier baselines.
3. Define target categories, abstention and explicit data-availability states from what is distinguishable.
4. Lock held-out sessions and pre-register metrics, resource limits and acceptable false-alarm tradeoffs.
5. Compare performance per workload/event; record disagreements and missing evidence.

**Deliverables:** accepted comparison protocol, baseline metrics with denominators, confusion/error analysis and reproducible evaluator script.

**Exit gate:** scientifically meaningful benchmark and justified reason to attempt a neural temporal model. **Stop** if independent abnormal examples are unavailable: proceed only with supported workload classification or unsupervised research labelled exploratory.

### NSE-3: neural and temporal model comparison

**Dependencies:** NSE-2; Track A Stage 6 experimental discipline is recommended but Stage 7/8 are not required.

**Actions:**
1. Build a small MLP on temporal summary features; establish parameter count and training cost.
2. Evaluate one sequence model (candidate GRU or TCN) with fixed window/missingness handling; add alternatives only if useful.
3. Compare calibration and false-positive behaviour across multiple time-separated sessions.
4. Investigate feature ablation to see which sensors truly contribute.
5. Evaluate model drift after representative software, driver and workload changes.
6. Select the simplest model that passes the relevant task budget; reject complexity that has no demonstrated gain.

**Deliverables:** experiment manifest and checkpoints in approved private storage, evaluator results, model card, limitations and selection decision.

**Exit gate:** demonstrable, repeatable benefit versus selected simpler baselines, acceptable false alarms and resource cost, unknown/abstention behaviour. **Stop** if no useful neural advantage exists; retaining the deterministic detector is a successful evidence-led outcome.

### NSE-4: local shadow inference

**Dependencies:** NSE-3 model accepted as a *candidate* and separate operator authorisation for installation.

**Actions:**
1. Design a **read-only** local inference consumer that observes snapshot copies after collection; never interfere with sampling.
2. Enforce versioned feature preprocessing, bounded window queue, backpressure/timeout and no automatic public exposure.
3. Run neural inference alongside the existing anomaly detector without routing it to Ramone as authoritative.
4. Capture head-to-head predictions, latency, memory/CPU/GPU costs, model identity, quality flags and source window IDs.
5. Test power-off, stale edge snapshots, WSL restarts, model crash, unavailable NVML, host sleep and LLM GPU contention.
6. Roll back cleanly by disabling only inference, preserving existing detector and raw collection.

**Deliverables:** independently authorised local deployment proposal, shadow comparison report with real sessions, resource/fallback evidence and rollback drill.

**Exit gate:** positive observed value with acceptable operator cost; collector and existing detector remain unaffected. Deployment evidence is separate from merged source.

### NSE-5: Ramone read-only machine perception

**Dependencies:** NSE-4 evidence, approved tool owner and reviewed private contract.

**Actions:**
1. Establish the exact active Ramone orchestration and authentication boundary.
2. Define machine_state schema, permissions, freshness/uncertainty semantics, model/input identity and raw evidence references.
3. Implement read-only tool access without process controls or broader filesystem/network capabilities.
4. Evaluate Ramone on questions including fan noise, heavy load, suspected throttle, memory pressure, unknown readings and unavailable service.
5. Create grounding regressions: no fabricated sensor values, no unsupported fault causality, no certainty from stale results.
6. Verify voice/web/UI paths only if explicitly in scope and authorised. Public and private Ramone views have different permissions.

**Deliverables:** reviewed contract and provenance fixtures, tool tests, failure-mode tests, Ramone language-grounding evaluation, auth/permission review.

**Exit gate:** Ramone reliably distinguishes prediction from fact, can inspect actual supporting measurements within authority and abstains correctly. No tool assignment/rollout implied by checked source.

### NSE-6: reviewed promotion, controlled rollout and rollback

**Dependencies:** NSE-5 source/eval gates plus separate model/inference deployment authority.

**Actions:**
1. Decide evaluator and promotion policy ownership with atlas-infra; avoid misusing the current LLM-oriented promotion contract.
2. Bind candidate model, features, evaluation corpus, schema, thresholds, supported labels and known limitations to immutable evidence.
3. Require explicit human review and approval of candidate; then separately approve install/service changes.
4. Install only to reviewed local endpoints, record exact model digest, verify startup/health, actual identity, latency, raw-source freshness and tool availability.
5. Validate operator-facing Ramone answers with independent runtime evidence and a rollback exercise.
6. Retain current deterministic anomaly detector unless a further reviewed comparative retirement gate is explicitly passed.

**Deliverables:** promotion record, release/rollback runbook, installed identity evidence, post-rollout verification and exception/incident handling.

**Exit gate:** successful **runtime verification** and safe user-visible behaviour for named capabilities. No generic "AI health verified" claim.

### NSE-7: evidence-led extensions, not launch blockers

Candidates, evaluated independently:
- richer machine-state taxonomy and new sensor families;
- learned internal embeddings with measured value;
- longer temporal context and state-transition tracking;
- careful, reviewed retraining triggered by detected drift, not silent continuous self-training;
- operator-authorised bounded machine-state history in Ramone memory;
- model-error explanations with supporting sensor windows;
- recorded model-activation visualisations within the public Model Lab;
- System SYMPHONY sonification of **explicitly tagged interpreted** machine state;
- forecasting experiments clearly distinguished from fault diagnosis.

Each candidate needs its own owner, privacy/resource/evaluation gate, source review, rollout authority and current runtime observation. Latent embeddings and confident labels must never be treated as equivalent to sensor evidence.

## 9. Ownership, changes and authority

| Repository / system | Current role | Proposed future responsibility | Mandatory boundary |
| --- | --- | --- | --- |
| atlas-model-lab | Public educational ML experiments | Synthetic/approved public baselines, training code, reproducibility protocols, planning docs | No raw private telemetry, private checkpoints, API or production deployment |
| specular-telemetry | Collector, local anomaly detector, public bounded projection | Owner candidate for reviewed local private capture and read-only inference integration | Raw measurement path independent of ML; no automatic edge exposure |
| atlas-infra | Architecture, lifecycle, classification, contracts | New ADR and private tool/model contracts if cross-repo integration proceeds | Accepted ADRs/policy are authority; this document is not |
| atlas-eval-harness | Private LLM evaluation and promotion evidence | Potential governance/evidence interface, subject to compatibility review | Do not assert existing LLM scoring can evaluate neural telemetry |
| ollama-rag-kit / actual Ramone runtime | Local retrieval/generation/orchestration | Potential read-only machine_state tool consumer | Verify actual runtime/tool route and assignment first |
| ramone-edge | Public Ramone gateway | No privileged machine telemetry by default | Public boundary cannot expose private host data by assumption |
| ramone-memory | Private conversational memory | Optional later reviewed state-history consumer | Predictions never silently become persistent factual memories |
| atlas-corpus | Estate knowledge/retrieval | Architecture docs and non-sensitive approved source evidence | No private telemetry/corpus ingestion by default |
| atlas-systems | Portfolio presentation | Optional governed public display of aggregated, sanitised experiment results | No invented or unreviewed live claims |
| SPECULAR-CORE operator | Runtime authority | Approved sensor access, local storage, inference install and resource budget | Separate approval per mutation and measured validation |

Ownership above is **proposed** when labelled future. Establish actual service owner, runtime route, machine constraints, and current policy before implementing.

## 10. Security, cost, retention and operational risk

- **Threat surface:** sensor spoofing/errors, schema changes, adversarial input, stale last-known-good data, inference model drift, unexpected subprocess/tool permissions, overexposure of host telemetry, accidental public dataset export, model weight leakage, unsafe autogenerated causal claims.
- **Access:** least-privilege, local-only default, read-only machine-state queries, bounded request frequency/response size, secret handling exclusively via approved interactive mechanisms.
- **Storage:** explicit estimate from measured sample size/frequency and retention. Rotate and delete. Encryption/access and backup scope must be reviewed; never assume Cloudflare KV is the canonical training store.
- **Compute:** establish budgets with local benchmarks while the LLM is in use. Prioritise normal telemetry and Ramone over optional inference. Stop training on contention instead of silently starving other services.
- **Update risk:** model/feature/schema version mismatch must fail safely. Revisit training after hardware/OS/driver changes instead of pretending identical distributions.
- **Incident response:** log only safe diagnostics, preserve source evidence IDs, prefer operator-controlled rollback, and retain the existing detector as fallback.
- **Operating cost:** target zero incremental recurring cost, but measure disk, power, wear, time and GPU contention rather than declaring them free.

## 11. Example user journeys and proof obligations

### Why are my fans loud?

**Raw evidence required:** GPU/CPU utilisation, temperatures, available fan measurements, time-window freshness and potentially power/clocks. If fan telemetry is unavailable, Ramone must say that it cannot directly measure fan behaviour.

**Safe outcome:** "Recent GPU load and temperature are elevated together; that may explain a cooling response. Fan RPM was not measured." Do not assert a specific fan controller action from missing inputs.

### Is my GPU thermally throttling?

**Raw evidence required:** appropriate clock and thermal/performance-limit indicators, load, temperatures and time correlation. Current inspected GPU collector does not provide a sufficient direct throttling evidence set.

**Safe outcome before instrumentation:** "I cannot confirm throttling from the available temperature/utilisation alone."

### What happened during the last 20 minutes?

**Evidence required:** timestamped sequence with sufficient coverage, validated transitions and gap reporting. Never extrapolate continuously through sleep/offline intervals. A history feature additionally requires reviewed retention.

### Is something wrong?

**Evidence required:** defined detectable fault classes and current independent measurements. Unknown anomalies and out-of-distribution windows must not be forced into "normal." The system should report what was and was not observed.

## 12. Programme acceptance and stop checklist

Do **not** proceed from a gate until these questions are answered with recorded evidence:

- **Evidence:** are sensor fields actually present, versioned, fresh and attributable to the current machine?
- **Privacy:** where is data stored, for how long, with what access and deletion mechanism?
- **Ground truth:** which labels were independently verified rather than inferred from existing detector output?
- **Validity:** are train/validation/test sessions isolated without overlapping-window leakage?
- **Comparison:** does the neural model beat an appropriate simple/current baseline on at least one defined useful dimension without unacceptable regressions?
- **Uncertainty:** is confidence calibrated where claimed, and are unavailable/out-of-distribution inputs handled safely?
- **Provenance:** does every result bind to input window, feature version, model digest, eval revision and evidence reference?
- **Resources:** has cost been measured alongside real SPECULAR workloads, including local LLM use?
- **Authority:** is there separately granted permission for collection, private storage, training, model installation, tool assignment, deployment and operator-visible integration?
- **Failure:** can the system be stopped without losing raw telemetry or existing anomaly detection?
- **Claims:** can Ramone consistently distinguish observed measurements, model classifications and unsupported diagnoses?
- **Release:** did independent runtime/live checks prove the exact expected identity rather than just a merged commit?

**Success does not require a GRU.** If the best evidence-backed solution is a smaller classifier, a hybrid with the current detector, or no neural replacement at all, document that result and retain the trusted system.

## 13. Versioning and next bounded work item

The next executable task is **NSE-0 inspection only**, separately scoped and authorised. Produce a dated cross-repository/current-runtime audit, a field-by-field sensor coverage table, a private-data risk register and the first set of measurable Ramone use cases. That inspection informs a potential Atlas Infra architecture decision and any eventual training-data capture contract.

Do not jump from this roadmap to collecting private data, expanding the public Worker, building an inference server, changing Home Assistant/Open WebUI tools, retiring the old detector, automatically retraining, or publishing machine-state values.

Related reading:
- [Model Lab programme](ROADMAP.md)
- [Model Lab repository purpose and prohibitions](REPOSITORY-DECISION.md)
- [Atlas Infra ADR-0013](https://github.com/AtlasReaper311/atlas-infra/blob/main/docs/adrs/ADR-0013-estate-wide-evidence-lifecycle.md) source merge is not live verification
- [Atlas Infra ADR-0019](https://github.com/AtlasReaper311/atlas-infra/blob/main/docs/adrs/ADR-0019-public-model-promotion-projection.md) promotion is capability-specific and not runtime deployment
- [SPECULAR telemetry](https://github.com/AtlasReaper311/specular-telemetry)
- [Private evaluation harness](https://github.com/AtlasReaper311/atlas-eval-harness)
