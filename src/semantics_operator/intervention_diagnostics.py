"""Read-only comparison of override wording and surgical equation replacement."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import torch

from .answer_protocol import BARE_CANDIDATES, parse_generation, protocol_metadata
from .causal_tasks import InterventionMode, Scheme, circuit_questions, metrics
from .generation_evaluation import attach_candidate_agreement, generation_metrics
from .world import OPERATORS


@torch.no_grad()
def run_intervention_diagnostics(
    lm, cfg, *, max_new_tokens=16, schemes=tuple(Scheme), progress=lambda _: None
):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    tests = {}
    for scheme in schemes:
        scheme = Scheme(scheme)
        samples = circuit_questions("test", scheme)

        def run(
            sequence=(), mode=InterventionMode.REPLACE_EQUATION, samples=samples, scheme=scheme
        ):
            prompts = [q.prompt(sequence, mode=mode) for q in samples]
            scores = torch.cat(
                [
                    lm.scores(prompts[i : i + 2], candidates=BARE_CANDIDATES).cpu()
                    for i in range(0, len(prompts), 2)
                ]
            )
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Non-finite intervention audit scores")
            outputs = []
            for i, prompt in enumerate(prompts):
                generated = lm.generate_greedy([prompt], max_new_tokens=max_new_tokens)[0]
                outputs.append({"prompt": prompt, **parse_generation(generated)})
                if i == 0 or (i + 1) % 20 == 0 or i + 1 == len(prompts):
                    progress(f"Audit {scheme.value}/{mode.value}: {i + 1}/{len(prompts)}")
            return scores, outputs

        base_scores, base_outputs = run()
        tests[scheme.value] = {}
        for op in OPERATORS:
            tests[scheme.value][op.key] = {}
            for name in ("base", *(mode.value for mode in InterventionMode)):
                progress(f"Intervention audit: {scheme.value}/{op.key}/{name}")
                scores, outputs = (
                    (base_scores, base_outputs)
                    if name == "base"
                    else run((op,), InterventionMode(name))
                )
                ranked = metrics(scores, samples, (op,), base_scores)
                generated = generation_metrics(outputs, samples, (op,), base_outputs)
                attach_candidate_agreement(generated, ranked)
                tests[scheme.value][op.key][name] = {"candidates": ranked, "greedy": generated}
    return {
        "experiment": "intervention_wording_diagnostics_v1",
        "model": {"profile": cfg.profile, "device": str(lm.device)},
        "answer_protocol": protocol_metadata(lm),
        "max_new_tokens": max_new_tokens,
        "test": tests,
        "limitations": [
            "Read-only model audit, without activation or weight changes; no operator is trained or selected.",
            "All modes use the same symbolic do-targets and the same default-wording test questions. Base predictions are evaluated against those counterfactual targets.",
            "replace_equation replaces only the intervened equations in the printed system; legacy_override retains the original equations and appends a force instruction.",
            "Complete digit plus stopping is required; invalid and incomplete responses remain in accuracy denominators. Protected damage is relative to correct baseline generations.",
        ],
    }


def save_intervention_diagnostics(root, report):
    folder = root / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-intervention-audit-" + uuid4().hex[:8]
    )
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# Intervention wording audit",
        "",
        "| Scheme | Operation | Mode | Greedy accuracy | Relay | Lamp | All nodes | New protected errors | Format errors | Incomplete |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scheme, ops in report["test"].items():
        for op, modes in ops.items():
            for name, values in modes.items():
                m = values["greedy"]
                lines.append(
                    f"| {scheme} | {op} | {name} | {m['overall']['accuracy']:.3f} | "
                    f"{m['by_node']['relay']['accuracy']:.3f} | {m['by_node']['lamp']['accuracy']:.3f} | "
                    f"{m['all_nodes_correct']:.3f} | {m['protected_damage']['damaged']} | "
                    f"{m['overall']['format_errors']} | {m['overall']['incomplete']} |"
                )
    lines += ["", "## Limits", "", *["- " + x for x in report["limitations"]]]
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder
