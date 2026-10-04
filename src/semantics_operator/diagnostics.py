"""Baseline answer audit: unconstrained decoding versus two candidate formats."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import torch

from .answer_protocol import BARE_CANDIDATES, parse_generation, protocol_metadata
from .causal_tasks import PromptStyle, Scheme, answer_metrics, circuit_questions


@torch.no_grad()
def run_diagnostics(lm, cfg, *, max_new_tokens=16, progress=lambda _: None):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    formats = {"spaced_candidates": (" 0", " 1"), "bare_candidates": BARE_CANDIDATES}
    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = {}
        for scheme in Scheme if split == "test" else (Scheme.AND_COPY,):
            samples = circuit_questions(split, scheme, styles=tuple(PromptStyle))
            prompts = [q.prompt() for q in samples]
            common = [
                {
                    "key": q.key,
                    "node": q.node.value,
                    "style": q.style.value,
                    "expected": q.answer(),
                    "prompt": p,
                }
                for q, p in zip(samples, prompts, strict=True)
            ]
            modes = {}
            for name, candidates in formats.items():
                progress(f"Baseline {split}/{scheme.value}: {name}")
                scores = torch.cat(
                    [
                        lm.scores(prompts[i : i + 2], candidates=candidates).cpu()
                        for i in range(0, len(prompts), 2)
                    ]
                )
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("Non-finite baseline candidate scores")
                modes[name] = answer_metrics(
                    [
                        {
                            **record,
                            "prediction": int(score.argmax()),
                            "candidate_logp": score.tolist(),
                            "p1": float(score.softmax(-1)[1]),
                        }
                        for record, score in zip(common, scores, strict=True)
                    ]
                )
            records = []
            for i, (record, prompt) in enumerate(zip(common, prompts, strict=True)):
                generated = lm.generate_greedy([prompt], max_new_tokens=max_new_tokens)[0]
                records.append({**record, **parse_generation(generated)})
                if i == 0 or (i + 1) % 20 == 0 or i + 1 == len(prompts):
                    progress(f"Baseline {split}/{scheme.value}: greedy {i + 1}/{len(prompts)}")
            modes["greedy"] = answer_metrics(records)
            modes["greedy"]["token_limit_count"] = sum(
                r["stop_reason"] == "max_new_tokens" for r in records
            )
            valid = [i for i, r in enumerate(records) if r["prediction"] is not None]
            modes["agreement"] = {
                "count": len(records),
                "valid_generated_count": len(valid),
                "candidate_format_disagreements": sum(
                    a["prediction"] != b["prediction"]
                    for a, b in zip(
                        modes["spaced_candidates"]["records"],
                        modes["bare_candidates"]["records"],
                        strict=True,
                    )
                ),
                "greedy_disagreements_on_valid": {
                    name: sum(
                        records[i]["prediction"] != modes[name]["records"][i]["prediction"]
                        for i in valid
                    )
                    for name in formats
                },
            }
            splits[split][scheme.value] = modes
    return {
        "experiment": "base_answer_diagnostics_v2",
        "answer_protocol": protocol_metadata(lm),
        "model": {"profile": cfg.profile, "device": str(lm.device)},
        "max_new_tokens": max_new_tokens,
        "chat_template_applied": bool(getattr(lm.tokenizer, "chat_template", None)),
        "candidate_token_ids": {
            name: [lm.tokenizer.encode(c, add_special_tokens=False) for c in candidates]
            for name, candidates in formats.items()
        },
        "splits": splits,
        "limitations": [
            "Read-only baseline audit; shifted graphs are never used for training or operator selection.",
            "Greedy decoding uses raw argmax, no forced answer prefix, no sampling, no logits processors, and no KV cache.",
            "Only a complete decoded response of 0 or 1 (ignoring surrounding whitespace and special tokens) is parsed. Other outputs count as errors.",
            "Responses reaching the token budget are incomplete, even if the partial text is a digit; they are counted separately from completed format errors and are not accepted as complete answers.",
            "Candidate scores are separately tokenized continuation log-probabilities; normalized 0/1 probabilities are not vocabulary-wide confidence.",
            "The three wording styles share equations and variable names. Rephrased worlds are correlated observations.",
        ],
    }


def save_diagnostics(root, report):
    folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-diagnose-" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# Baseline answer diagnostics",
        "",
        "| Split | Scheme | Mode | Accuracy | Errors for 0 | Errors for 1 | Format errors | Incomplete |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for split, schemes in report["splits"].items():
        for scheme, modes in schemes.items():
            for name in ("spaced_candidates", "bare_candidates", "greedy"):
                m = modes[name]
                lines.append(
                    f"| {split} | {scheme} | {name} | {m['overall']['accuracy']:.3f} | "
                    f"{m['by_label']['0']['errors']}/{m['by_label']['0']['count']} | "
                    f"{m['by_label']['1']['errors']}/{m['by_label']['1']['count']} | {m['overall']['format_errors']} | {m['overall']['incomplete']} |"
                )
    lines += [
        "",
        "Raw prompts, generated text, token IDs, stopping reasons, per-node/per-style metrics and agreement counts are in report.json.",
        "",
        "## Limits",
        "",
    ]
    lines.extend("- " + s for s in report["limitations"])
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder
