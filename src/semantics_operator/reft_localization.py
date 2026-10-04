"""Recheck decoder blocks on the strict causal validation prompts before training."""

import torch

from .answer_protocol import BARE_CANDIDATES
from .causal_tasks import circuit_questions, metrics, selection
from .localization import capture_tail, discover_sites, patched_scores
from .world import OPERATORS


@torch.no_grad()
def locate_block(lm, layers, preservation_weight, progress=lambda _: None):
    catalog = discover_sites(lm)
    layers = sorted(set(layers))
    if not layers or any(type(i) is not int or i not in catalog for i in layers):
        raise ValueError(f"Provide existing decoder layer indices: {sorted(catalog)}")
    sites = [catalog[i]["block"] for i in layers]
    samples = circuit_questions("validation")
    prompts = [q.prompt() for q in samples]
    base = torch.cat(
        [
            lm.scores(prompts[i : i + 2], candidates=BARE_CANDIDATES).cpu()
            for i in range(0, len(samples), 2)
        ]
    )

    def capture(prompts):
        chunks = {site: [] for site in sites}
        for i in range(0, len(samples), 2):
            values = capture_tail(lm, prompts[i : i + 2], sites, 1, [])
            for site in sites:
                chunks[site].append(values[site])
        return {site: torch.cat(rows) for site, rows in chunks.items()}

    natural = capture(prompts)
    donors = {op.key: capture([q.prompt((op,)) for q in samples]) for op in OPERATORS}

    def patch(site, donor):
        return torch.cat(
            [
                patched_scores(
                    lm,
                    prompts[i : i + 2],
                    {site: donor[site][i : i + 2]},
                    [site],
                    1,
                    [],
                    candidates=BARE_CANDIDATES,
                )
                for i in range(0, len(samples), 2)
            ]
        )

    def objective(by_operator):
        scores = [selection(m, preservation_weight) for m in by_operator.values()]
        return [sum(s[k] for s in scores) / len(scores) for k in range(3)]

    baseline = {op.key: metrics(base, samples, (op,), base) for op in OPERATORS}
    trials = []
    for layer, site in zip(layers, sites, strict=True):
        progress(f"Relocalize bare digits: block {layer}")
        own = patch(site, natural)
        error = float((own - base).abs().max())
        if not torch.allclose(own, base, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Bare-digit localization self-patch failed")
        by_operator = {
            op.key: metrics(patch(site, donors[op.key]), samples, (op,), base) for op in OPERATORS
        }
        trials.append(
            {
                "layer": layer,
                "site": site,
                "objective": objective(by_operator),
                "by_operator": by_operator,
                "self_patch_max_score_difference": error,
            }
        )
    chosen = max(trials, key=lambda t: tuple(t["objective"]))
    base_objective = objective(baseline)
    beats_base = chosen["objective"] > base_objective
    progress(f"Selected block {chosen['layer']}; donor validation beats no patch: {beats_base}")
    return chosen["site"], {
        "candidate_layers": layers,
        "selected_layer": chosen["layer"],
        "boundary": "prompt",
        "forced_prefix": [],
        "trials": trials,
        "baseline_objective": base_objective,
        "beats_no_patch": beats_base,
        "selection": "Lexicographic mean of per-operator preservation-penalized objectives on validation AND/copy; ties prefer lower block index.",
    }
