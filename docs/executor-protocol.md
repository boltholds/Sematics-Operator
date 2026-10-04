# Inspectable causal executor v1

User-approved scope: first train and test a small neural causal transition module
on structured inputs, before adding an LLM representation adapter. The additional
constraint is inspectability rather than two opaque neural components.

## Representation and priors

The persistent state is one P(value=1) per node. No opaque recurrent memory bypasses
this state. A local two-token transformer sees the current node value, its parent's
value, and a ROOT/COPY/NOT rule. Two transformer blocks process this local pair;
one recurrence can therefore move information only one graph edge. The same
parameters are reused at every node and computational step. A two-layer MLP with
the same input information is trained as a simpler comparator; parameter counts
are reported and are not claimed to match.

Known graph edges, node identity routing, source values, and do-intervention clamps
are supplied explicitly. Roots and intervention targets are clamped by the harness.
COPY/NOT transitions for the remaining nodes are learned. This is supervised neural
execution of supplied SCMs, not causal discovery or unrestricted latent reasoning.
The structural priors themselves enforce much of locality; report this explicitly.

## Data and training

Random unary acyclic forests contain COPY/NOT mechanisms and an isolated root.
Short graphs have 6–9 nodes and depth 2–3; long test graphs have 11–14 nodes and
depth 6–8. Canonical unlabelled topology keys (ignoring root values and rule labels)
are disjoint across train, validation and both tests. Node indices are permuted.
For each evaluation graph both do(target=0) and do(target=1) are evaluated; at least
one descendant of the target exists. Composition evaluates two simultaneous clamps.

The oracle updates all free nodes synchronously from the preceding state. State 0
is the observed equilibrium with the intervention already applied to its targets.
Subsequent steps propagate its consequences. Targets are training labels only;
rollouts feed the model's own continuous predictions back through all steps.
Loss is BCE over free nodes at every step, with an extra weight on nodes outside
the intervention's descendant set. Padding, roots and forced targets are excluded.
Validation selects the checkpoint; test results do not change training or selection.
Default: width 64, 2 blocks, 4 heads, 300 updates, 32 episodes per batch, 3 seeds.

## Evidence

Report final whole-state and learned-node accuracy, changed-node accuracy excluding
forced targets, all-step exactness, preservation damage, and paired correctness on
both intervention values. Compare oracle, clamp-only, one-step truncation,
parent-disconnected ablation, transformer and MLP. Evaluate node permutation
equivariance. Read the complete binary local COPY/NOT truth table.

For a causal memory audit, run two steps, force a non-root, non-intervened internal
state coordinate to the opposite value once, then resume the model. Compare with
the oracle receiving the same pulse. Report eligibility (correct state before the
pulse), descendants affected, whole-state trajectory agreement, and a self-patch
control. This is a transient pulse, not a persistent do intervention.

Save configuration, dataset topology keys and episodes, validation history,
per-step probabilities and oracle values, exact selected weights, and an offline
HTML report. The command must run without an LLM model file. Fixed training budget;
no retry-until-success search. A failed experiment is an informative saved result.
