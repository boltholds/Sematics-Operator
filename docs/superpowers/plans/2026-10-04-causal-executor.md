# Inspectable causal executor implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Train a small inspectable causal executor and evaluate structural generalization.
**Architecture:** Explicit probability memory, local shared transformer or MLP, supplied graph and clamps. Separate oracle, model, experiment and offline reporting modules.
**Tech Stack:** Existing Python, PyTorch and safetensors; no new dependencies.
**Spec:** `docs/executor-protocol.md`

## Global constraints

- Work on main as authorized; preserve existing experiments and .env.
- No LLM load for the executor command.
- Intermediate oracle states never feed a model rollout after state 0.
- Topology-disjoint splits; fixed budgets; validation-only checkpoint selection.
- State and graph assumptions must be disclosed in reports.

## Review focus

- Non-topological node order: validate and execute arbitrary acyclic relabelings.
- Multiple intervention targets: persistent clamps override their normal mechanisms.
- Empty denominators: changed-node and eligible-pulse metrics must expose counts.
- Padded batches: no loss or scoring on padding, roots or forced targets.
- End-to-end CLI: no accidental pretrained model loading, restorable checkpoints.

### Task 1: Oracle and structural splits

Files: `executor_tasks.py`, `tests/test_executor.py`.
Interfaces: `Circuit`, `Episode`, `oracle_step`, `oracle_trace`, `make_dataset`.
- [x] Test a literal COPY/NOT chain, protected root, simultaneous clamps and permutation.
- [x] Observe missing-feature failures, implement immutable data and canonical topology keys.
- [x] Test disjoint/reproducible graph splits and invalid/cyclic input rejection.

### Task 2: Learned local transition and audits

Files: `executor_model.py`, `executor_experiment.py`, `tests/test_executor.py`.
Interfaces: `LocalExecutor`, `pack`, `rollout`, `evaluate`, `run_executor`.
- [x] Test one-hop dependence, clamp/padding behavior and gradient flow through recurrence.
- [x] Implement own-prediction rollouts, BCE training and validation checkpoint selection.
- [x] Test metrics against hand-derived errors and memory-pulse oracle traces.
- [x] Implement paired, long-graph and double-clamp evaluation plus all controls.

### Task 3: CLI, artifacts and actual experiment

Files: `executor_reporting.py`, `configs/executor.toml`, `cli.py`, `README.md`.
Interfaces: `load_executor_config`, `save_executor`, `semop executor`.
- [x] Test CLI with an absent LLM, checkpoint reload and finite serialized evidence.
- [x] Implement offline HTML with explicit graph rules and per-step state tables.
- [x] Run the new tests, full repository suite, lint and independent read-only review.
- [x] Run the predeclared 3-seed experiment; save successes and failures without tuning on test.
- [x] Publish code and make the real experiment archive available.
