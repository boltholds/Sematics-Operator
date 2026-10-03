"""Machine-readable results, portable low-rank factors and captured layer vectors."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from safetensors.torch import save_file


def save_run(root: Path, report: dict, patches: dict, representations: dict) -> Path:
    folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    tensors = {
        f"{key}.{factor}": getattr(patch, factor).contiguous()
        for key, patch in patches.items()
        for factor in ("a", "b")
    }
    metadata = {
        "target_module": report["model"]["target_module"],
        "target_weight_sha256": report["model"]["target_weight_sha256"],
        "format": "delta_W = B @ A / sqrt(rank)",
        "schema_version": "1",
    }
    save_file(tensors, folder / "operators.safetensors", metadata=metadata)
    save_file(
        {key: value.contiguous() for key, value in representations.items()},
        folder / "representations.safetensors",
        metadata={
            "position": "last non-padding prompt token",
            "target_module": report["model"]["target_module"],
            "samples": "held-out lamp questions, in dataset order",
        },
    )
    (folder / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Weight operator experiment",
        "",
        report["claim"],
        "",
        f"Target: `{report['model']['target_module']}`",
        "",
        "| Scenario | Original | Weight edit | Random control | Explicit prompt |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, scenario in report["scenarios"].items():
        values = [
            scenario[method]["overall"]["accuracy"]
            for method in ("base", "weight_edit", "random_norm_matched", "explicit_prompt")
        ]
        lines.append("| " + name + " | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += [
        "",
        (
            "These are overall restricted-choice accuracies. See report.json for changed "
            "facts, unchanged facts, preservation of originally correct answers and per-node scores."
        ),
        "",
        "## Rollback",
        "",
        json.dumps(report["rollback"]),
        "",
        "## Limitations",
        "",
    ]
    lines += ["- " + item for item in report["limitations"]]
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder
