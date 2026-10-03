from semantics_operator.sampling import coverage, training_batches
from semantics_operator.world import OPERATORS, questions


def test_short_run_has_effect_locality_and_distinguishing_examples():
    samples = questions("train")
    for seed in (0, 42, 99):
        for op in OPERATORS:
            batches = training_batches(samples, op, steps=5, seed=seed)
            assert batches == training_batches(samples, op, steps=5, seed=seed)
            for batch in batches:
                assert len(batch) == 2
                assert {samples[i].answer((op,)) != samples[i].answer() for i in batch} == {
                    True,
                    False,
                }
            stats = coverage(samples, op, batches)
            assert stats["changed_presentations"] == 5
            assert stats["unchanged_presentations"] == 5
            assert all(v > 0 for v in stats["distinguishing_presentations"].values())
            assert all(samples[i].world.name.startswith("training_") for b in batches for i in b)


def test_long_run_covers_every_training_question():
    samples = questions("train")
    for op in OPERATORS:
        batches = training_batches(samples, op, steps=40, seed=42)
        assert {i for b in batches for i in b} == set(range(40))
