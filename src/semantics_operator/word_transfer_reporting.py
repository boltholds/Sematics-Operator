"""Self-contained, inspectable word-transfer evidence and saved directions."""

import html
import json
from datetime import UTC, datetime
from uuid import uuid4

from safetensors.torch import save_file


def save_word_transfer(root, report, tensors):
    folder = root / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-word-transfer-" + uuid4().hex[:8]
    )
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    save_file(
        tensors,
        folder / "directions.safetensors",
        metadata={"experiment": report["experiment"], "site": json.dumps(report["site"])},
    )

    def fraction(value):
        return f"{value['correct']}/{value['count']}" if value["count"] else "n/a (0 eligible)"

    lines = [
        "# RU → EN word-direction transfer",
        "",
        f"Block {report['site']['layer']}; primary: ru_delta@1; raw prompts; frozen weights.",
        "",
        "Baseline understanding (original temperature and protected properties):",
        "",
    ]
    for mode in ("candidate", "greedy"):
        m = report["baseline_understanding"][mode]
        lines.append(
            f"- {mode}: temperature {fraction(m['temperature'])}; protected {fraction(m['protected'])}; joint {fraction(m['joint'])}."
        )
    lines += [
        "",
        "All rows below target the OPPOSITE temperature, including baseline.",
        "",
        "| Condition | Mode | Temperature | Protected | Joint both directions | On baseline-correct pairs | Protected damage | Invalid/incomplete |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, condition in report["conditions"].items():
        for mode in ("candidate", "greedy"):
            m = condition["metrics"][mode]
            d = m["protected_damage"]
            lines.append(
                f"| {key} | {mode} | {fraction(m['temperature'])} | {fraction(m['protected'])} | {fraction(m['paired_joint'])} | {fraction(m['baseline_correct_paired_joint'])} | {d['damaged']}/{d['eligible']} | {m['invalid_or_incomplete']} |"
            )
    lines += [
        "",
        "## Full-vocabulary first-answer KL on protected questions",
        "",
        "| Condition | Color | Object | Count |",
        "|---|---:|---:|---:|",
    ]
    for key, c in report["conditions"].items():
        kl = c["metrics"]["protected_first_token_kl"]
        lines.append(
            f"| {key} | "
            + " | ".join(
                f"{kl[k]:.6g}" if kl[k] is not None else "n/a" for k in ("color", "object", "count")
            )
            + " |"
        )
    lines += ["", "## Raw completions of the original minimal heatmap prompts", ""]
    for key, c in report["conditions"].items():
        for word, result in c["raw_completions"].items():
            g = result["generation"]
            lines += [f"{key} / {word} ({g['stop_reason']}):", "", "    " + repr(g["text"]), ""]
    lines += ["## Limits", "", *["- " + s for s in report["limitations"]]]
    summary = "\n".join(lines) + "\n"
    (folder / "summary.md").write_text(summary, encoding="utf-8")
    details = []
    for key, c in report["conditions"].items():
        rows = []
        for case, questions in c["cases"].items():
            for name, result in questions.items():
                g = result["generation"]
                rows.append(
                    "<tr>"
                    + "".join(
                        f"<td>{html.escape(str(v))}</td>"
                        for v in (case, name, g["text"], g["stop_reason"], result["candidate_logp"])
                    )
                    + "</tr>"
                )
        details.append(
            f"<details><summary>{html.escape(key)} — all answers</summary><table><tr><th>Input</th><th>Property</th><th>Generation</th><th>Stop</th><th>Candidate log P</th></tr>{''.join(rows)}</table></details>"
        )
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Word direction transfer</title><style>body{font:15px system-ui;max-width:1500px;margin:32px auto;padding:20px}pre{white-space:pre-wrap}td,th{border:1px solid #ccc;padding:8px;text-align:left}table{border-collapse:collapse}details{margin:20px 0}summary{cursor:pointer;font-weight:bold}</style>'
    page += '<h1>Word direction transfer</h1><p><a href="report.json">Full JSON</a> · <a href="summary.md">Summary</a> · <a href="directions.safetensors">Directions</a></p>'
    page += "<pre>" + html.escape(summary) + "</pre>" + "".join(details) + "</html>"
    (folder / "index.html").write_text(page, encoding="utf-8")
    return folder
