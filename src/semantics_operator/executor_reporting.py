"""Portable report: all predictions, transition tables, and reloadable checkpoints."""

import html
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from safetensors.torch import save_file


def ratio(score, numerator="correct"):
    n = score["count"]
    return f"{score[numerator]}/{n} ({score[numerator] / n:.1%})" if n else "n/a (0 cases)"


def summary(report):
    lines = [
        "# Inspectable causal executor",
        "",
        "Known COPY/NOT circuits; frozen graph routing and hard source/intervention clamps.",
        "The model learns the free-node update. No LLM is loaded.",
        "",
        "| Model / seed | Parameters | Selected step | Unseen graphs, all nodes | Long chains, all nodes | Two clamps, all nodes | Pulse, full trajectory |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for name, run in report["runs"].items():
        scores = [
            ratio(run["evaluation"][s]["metrics"]["all_nodes_final"])
            for s in ("test", "long", "composition")
        ]
        lines.append(
            f"| {name} | {run['parameter_count']} | {run['selected_step']} | "
            + " | ".join(scores + [ratio(run["pulse_audit"]["all_steps_exact"])])
            + " |"
        )
    lines += [
        "",
        "Pulse scores are conditional on a correct decoded pre-pulse state; see eligible/considered counts in report.json.",
        "All initial states are supplied equilibrium states with the intervention clamped.",
        "Composition evaluates two simultaneous interventions, not planning.",
        "A correct oracle control is a code ceiling. Clamp-only / one-step / disconnected-parent controls test shortcuts.",
        "",
        "## Supplied by the experiment",
        "",
    ]
    lines.extend(f"- {value}" for value in report["supplied"])
    lines += [
        "",
        report["limits"],
        "",
        "Safetensors store selected validation checkpoints, not optimizer state. Recreate LocalExecutor with the saved width/blocks/heads and load strictly.",
    ]
    return "\n".join(lines) + "\n"


def _trace_table(record):
    cells = ["<table><tr><th>Step</th>"]
    for i in range(len(record["gold"][0])):
        cells.append(f"<th>n{i}</th>")
    cells.append("</tr>")
    for t, (predicted, expected) in enumerate(zip(record["predicted"], record["gold"])):
        cells.append(f"<tr><th>{t}</th>")
        for p, bit in zip(predicted, expected):
            mismatch = int(p >= 0.5) != bit
            color = f"hsl({215 - 190 * p:.0f} 65% 84%)"
            cls = ' class="wrong"' if mismatch else ""
            cells.append(
                f'<td{cls} style="background:{color}" title="P(1)={p:.6f}; expected={bit}">'
                f"{p:.3f}<small> expected {bit}</small></td>"
            )
        cells.append("</tr>")
    cells.append("</table>")
    return "".join(cells)


def save_executor(output_dir, report, weights):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_dir) / f"{stamp}-executor-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (folder / "summary.md").write_text(summary(report), encoding="utf-8")
    for name, state in weights.items():
        save_file(
            {k: v.contiguous() for k, v in state.items()}, str(folder / f"{name}.safetensors")
        )
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Causal executor</title>',
        (
            "<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px;background:#f7f8fa;color:#192130}"
            "table{border-collapse:collapse;margin:12px 0;font-variant-numeric:tabular-nums}td,th{padding:7px;border:1px solid #bbc4d0}"
            "small{display:block;font-size:10px}.wrong{outline:3px solid #bd1837;outline-offset:-3px}"
            "details{margin:12px 0;padding:10px;background:white}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        ),
        "<h1>Inspectable causal executor</h1><p>Each cell: predicted P(1), expected bit. Red border: wrong decoded value.</p>",
        (
            "<p>Graph routing, sources and intervention clamps are supplied. Only free-node transitions are learned. "
            "This experiment does not use language representations.</p>"
        ),
        '<p><a href="report.json">Full report</a> · <a href="summary.md">Summary</a></p>',
        "<pre>" + html.escape(summary(report)) + "</pre>",
    ]
    for name, run in report["runs"].items():
        parts.append(f"<h2>{html.escape(name)}</h2><p>Checkpoint step {run['selected_step']}</p>")
        parts.append(
            "<h3>Learned local truth table</h3><table><tr><th>Rule</th><th>Own</th><th>Parent</th><th>P(1)</th><th>Expected</th></tr>"
        )
        for row in run["truth_table"]:
            parts.append(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(row[k]))}</td>"
                    for k in ("rule", "own", "parent", "probability_one", "expected")
                )
                + "</tr>"
            )
        parts.append("</table>")
        for split, result in run["evaluation"].items():
            parts.append(
                f"<details><summary>{split}: "
                f"{ratio(result['metrics']['all_nodes_final'])} complete final states</summary>"
            )
            parts.append(
                "<pre>"
                + html.escape(
                    json.dumps(
                        {"metrics": result["metrics"], "controls": result["controls"]}, indent=2
                    )
                )
                + "</pre>"
            )
            for record in result["trajectories"]:
                equations = [
                    f"n{i}: {r}, parent=n{p}"
                    for i, (p, r) in enumerate(zip(record["parents"], record["rules"]))
                ]
                parts.append(
                    f"<details><summary>{html.escape(record['id'])}; do={record['interventions']}</summary>"
                    "<p>"
                    + html.escape("; ".join(equations))
                    + "</p>"
                    + _trace_table(record)
                    + "</details>"
                )
            parts.append("</details>")
        audit = run["pulse_audit"]
        parts.append(
            f"<details><summary>Memory pulse: {ratio(audit['all_steps_exact'])}; "
            f"eligible {audit['eligible']}/{audit['considered']}</summary>"
        )
        for record in audit["trajectories"]:
            parts.append(
                f"<details><summary>{html.escape(record['id'])}; pulse n{record['node']}; "
                f"eligible={record['eligible']}</summary>" + _trace_table(record) + "</details>"
            )
        parts.append("</details>")
    parts.append("</html>")
    (folder / "index.html").write_text("\n".join(parts), encoding="utf-8")
    return folder
