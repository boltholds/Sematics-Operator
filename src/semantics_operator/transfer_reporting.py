"""Compact transfer summaries; full question-level evidence stays in report.json."""

import json
from datetime import UTC, datetime
from uuid import uuid4

from safetensors.torch import save_file


def _fraction(row):
    return f"{row['correct']}/{row['count']}" if row["count"] else "n/a (0 eligible)"


def save_transfer(root, report, tensors):
    folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-transfer-" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    save_file(
        tensors,
        folder / "operators.safetensors",
        metadata={
            "experiment": report["experiment"],
            "site": json.dumps(report["site"]),
            "train_representation": report["train_representation"],
            "selected": json.dumps(report["selected"]),
        },
    )
    lines = [
        "# Cross-representation transfer",
        "",
        (
            f"Source: `{report['train_representation']}`; fixed site: `{report['site']['path']}`; "
            f"position: `{report['site']['position']}`; seed: {report['seed']}."
        ),
        "",
        (
            "All choices use source validation only. Natural understanding is measured before training; "
            "target results do not change fitting or selection."
        ),
        "",
        (
            "Pairs require complete correctness on BOTH COPY and NOT. Source/direct/seen is an "
            "in-domain control; renamed and chain conditions are separate transfer tests."
        ),
        "",
        "## Natural understanding (free generation)",
        "",
        "| Representation / names / topology | Answer accuracy | Both worlds correct | Invalid / incomplete |",
        "|---|---:|---:|---:|",
    ]
    for key, group in report["test"].items():
        m = group["understanding"]["natural"]["greedy"]
        lines.append(
            f"| {key} | {m['overall']['accuracy']:.3f} | "
            f"{_fraction(m['paired']['all_nodes_correct'])} | "
            f"{m['overall']['invalid']} / {m['overall']['incomplete']} |"
        )
    lines += ["", "## Frozen strengths", "", "| Operator | Method | Alpha |", "|---|---|---:|"]
    for op, methods in report["selected"].items():
        for method, choice in methods.items():
            lines.append(f"| {op} | {method} | {choice['alpha']} |")
    lines += [
        "",
        "## Transfer (free generation)",
        "",
        "| Condition | Operator | Method | Both worlds | Affected pair | On baseline-correct pairs | Protected damage | Equations |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, group in report["test"].items():
        for op, modes in group["operators"].items():
            for method, result in modes.items():
                if "greedy" not in result:
                    continue
                m, p = result["greedy"], result["greedy"]["paired"]
                d = m["protected_damage"]
                lines.append(
                    f"| {key} | {op} | {method} | {_fraction(p['all_nodes_correct'])} | "
                    f"{_fraction(p['affected_nodes_correct'])} | "
                    f"{_fraction(p['baseline_correct_pairs'])} | {d['damaged']}/{d['eligible']} | "
                    f"{m['equation_consistency']['all_satisfied']:.3f} |"
                )
    lines += [
        "",
        "## Joint source/target success (free generation)",
        "",
        "| Target condition | Operator | Method | Both representations | Target when source correct | On pairs understood in both |",
        "|---|---|---|---:|---:|---:|",
    ]
    for key, ops in report["cross_representation"].items():
        for op, modes in ops.items():
            for method, result in modes.items():
                if "greedy" in result:
                    m = result["greedy"]
                    lines.append(
                        f"| {key} | {op} | {method} | {_fraction(m['joint_all_nodes_correct'])} | "
                        f"{_fraction(m['target_given_source_correct'])} | "
                        f"{_fraction(m['baseline_correct_in_both'])} |"
                    )
    lines += [
        "",
        (
            "Full candidate scores, 0/1 output controls, per-node protected damage, prompts, "
            "generations, validation trials and prefix positions are in `report.json`."
        ),
        "",
        "## Limits",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ]
    (folder / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return folder
