import pytest
from test_model import tiny_model


def test_equation_replacement_preserves_other_equations_and_inputs():
    from semantics_operator.causal_tasks import InterventionMode, Scheme, circuit_questions
    from semantics_operator.world import OPERATORS

    for scheme, lamp_rule in (
        (Scheme.AND_COPY, "lamp = relay"),
        (Scheme.AND_INVERTED, "lamp = NOT relay"),
        (Scheme.AND_XOR, "lamp = relay XOR flag"),
    ):
        q = circuit_questions("test", scheme)[0]
        prompt = q.prompt((OPERATORS[1],))
        assert "relay = 1;" in prompt
        assert "relay = source AND switch" not in prompt
        assert lamp_rule in prompt
        assert "source=0; switch=0; flag=0" in prompt
        assert "all other equations and inputs are unchanged" in prompt
        legacy = q.prompt((OPERATORS[1],), mode=InterventionMode.LEGACY_OVERRIDE)
        assert "relay = source AND switch" in legacy and "force relay=1" in legacy
        lamp_only = q.prompt((OPERATORS[2],))
        assert "relay = source AND switch" in lamp_only and "lamp = 1." in lamp_only
        assert lamp_rule + "." not in lamp_only
        composed = q.prompt((OPERATORS[0], OPERATORS[1], OPERATORS[2]))
        assert "relay = 1; lamp = 1." in composed


def test_opposite_training_mechanisms_and_held_out_chains():
    from semantics_operator.causal_tasks import (
        PromptStyle,
        Scheme,
        circuit_questions,
        training_questions,
    )
    from semantics_operator.reft_experiment import train_batches
    from semantics_operator.world import OPERATORS

    samples = training_questions("train", styles=tuple(PromptStyle))
    assert len(samples) == 240
    assert {q.world.scheme for q in samples} == {Scheme.AND_COPY, Scheme.AND_INVERTED}
    for op in OPERATORS:
        batches = train_batches(samples, op, 80, 42)
        assert {i for batch in batches for i in batch} == set(range(240))
        for start in range(0, 80, 2):
            assert {samples[batches[i][0]].world.scheme for i in (start, start + 1)} == {
                Scheme.AND_COPY,
                Scheme.AND_INVERTED,
            }
    for scheme in (Scheme.AND_INVERTED_CHAIN, Scheme.AND_XOR_CHAIN):
        for split in ("train", "validation"):
            with pytest.raises(ValueError, match="test-only"):
                circuit_questions(split, scheme)
        for q in circuit_questions("test", scheme):
            if q.node.value in ("bridge", "lamp"):
                assert q.answer((OPERATORS[1],)) == (
                    0 if scheme == Scheme.AND_INVERTED_CHAIN else 1 - q.world.flag
                )
                assert q.answer((OPERATORS[0],)) == (
                    1 if scheme == Scheme.AND_INVERTED_CHAIN else q.world.flag
                )
            prompt = q.prompt((OPERATORS[2],))
            assert "bridge = " in prompt and "lamp = 1." in prompt


def test_intervention_audit_compares_both_prompt_modes_and_saves(tmp_path):
    from semantics_operator.causal_tasks import Scheme
    from semantics_operator.config import Settings
    from semantics_operator.intervention_diagnostics import (
        run_intervention_diagnostics,
        save_intervention_diagnostics,
    )

    report = run_intervention_diagnostics(
        tiny_model(),
        Settings("tiny", tmp_path),
        max_new_tokens=1,
        schemes=(Scheme.AND_INVERTED,),
    )
    for op, modes in report["test"]["and_inverted"].items():
        assert set(modes) == {"base", "legacy_override", "replace_equation"}
        for name, m in modes.items():
            assert m["greedy"]["overall"]["count"] == 40
            assert "protected_damage" in m["greedy"]
            records = m["greedy"]["records"]
            if op == "relay_1":
                assert all(x["expected"] == 0 for x in records if x["node"] == "lamp")
            if name == "replace_equation" and op == "relay_1":
                assert all("relay = 1;" in x["prompt"] for x in records)
    folder = save_intervention_diagnostics(tmp_path, report)
    assert (folder / "report.json").is_file() and (folder / "summary.md").is_file()
