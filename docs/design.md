# First experiment: supervised operators in weight space

The research aim is to investigate reasoning as controlled transformations of an LLM's
learned semantic organization. A holographic interpretation motivates the experiment;
it is not assumed as an established property or inferred from successful adapter training.

Version 0.1 tests the narrower prerequisite: can a temporary internal weight change
implement an intervention, preserve unrelated answers, and compose with another change?
It intentionally does not claim a learned general transition controller.

## Components

- `config`: machine-local model root in `.env`, reusable named TOML profiles.
- `world`: external Boolean structural equations, interventions, disjoint prompt splits.
- `model`: offline causal-LM loading, exact continuation likelihoods, layer-output probes.
- `weights`: low-rank effective matrices, immutable base parameters, snapshots and branches.
- `experiment`: primitive training, independent evaluation, composition, revision and rollback.
- `artifacts`: JSON observations, Markdown comparison, safetensors factors and activations.
- `cli`: inspect models and run the complete experiment.

## Algorithm

Keep all pretrained parameters frozen. Replace one internal `nn.Linear` with the
algebraically equivalent computation using W_effective = W_base + B A / sqrt(rank).
Optimize only A and B using oracle-labeled training questions. The objective is binary
candidate cross entropy, plus KL to original predictions on unchanged training facts,
plus a mean-square effective-delta penalty. Candidate probabilities are conditional on
the two supplied answers; they are not calibrated probabilities over all generated text.

Each primitive starts from the same seeded A and zero B. Its factors are copied to CPU
at the end of a fixed training budget. No held-out question affects optimization or
selection. Compose independent patches by adding their effective matrices, with no
cross terms. Same-node revision explicitly replaces the earlier patch. This replacement
is application logic, not learned conflict detection. Longer/compositional training and
a learned transition operator are future experiments.

## First world and evidence

source, switch and independent flag are binary exogenous values.
relay = source AND switch; lamp = relay. do(relay=v) replaces the relay equation;
do(lamp=v) replaces the lamp equation. Train relay=0, relay=1 and lamp=1 separately.
Evaluate primitives, relay=0 + lamp=1 (both orders), relay=0 then relay=1, and empty branch.

Train/test each contain all eight input assignments with different circuit names and
prompt templates. This tests paraphrase/name transfer, not unseen truth tables, causal
discovery or new mechanisms. All five nodes are queried to expose collateral changes.

Controls: unchanged base, explicit intervention in the prompt, and independently sampled
low-rank patches matched to each learned patch's effective Frobenius norm. Report raw
scores, oracle labels, per-node correctness, changed-fact accuracy, unchanged-fact
accuracy and preservation of originally correct answers. Report baseline competence so
that a weak base model cannot look successful merely by adopting a constant answer.

## Acceptance and limits

Engineering acceptance: local HF reload, differentiable multi-token scoring, actual weight
update equivalence, frozen base tensors, no aliasing of snapshots, exception rollback,
no train/test overlap, persisted artifacts and a full tiny-Transformer integration run.
Scientific acceptance is deliberately not a pass/fail flag. A useful result must beat
controls on changed consequences while retaining unrelated answers, across seeds/models.
A random tiny Transformer only verifies the pipeline. User-supplied pretrained models
must be run before making any semantic or reasoning claim.

Local safetensors models only; no remote model code or automatic downloads; no GGUF,
bitsandbytes or sharded multi-device editing in v0.1. One active session per model/thread.
Original checkpoint files are never written. Reports and weights stay out of Git.
