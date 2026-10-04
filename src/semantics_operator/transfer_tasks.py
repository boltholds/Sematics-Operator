"""Matched COPY/NOT worlds with independent language and naming factors.

SCM identities are metadata only. The two halves share names, input values,
wording, and rule order; only the downstream equation changes.
"""

import random
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from .causal_tasks import (
    CircuitNode,
    CircuitQuestion,
    CircuitWorld,
    PromptLayout,
    PromptStyle,
    Scheme,
)
from .positions import NEUTRAL_STATE_MARKER


class Representation(StrEnum):
    EN = "en"
    RU = "ru"
    SYMBOLIC = "symbolic"


class Names(StrEnum):
    SEEN = "seen"
    RENAMED = "renamed"


class Family(StrEnum):
    DIRECT = "direct"
    CHAIN = "chain"


PAIR_SCHEMES = {
    Family.DIRECT: (Scheme.AND_COPY, Scheme.AND_INVERTED),
    Family.CHAIN: (Scheme.AND_CHAIN, Scheme.AND_INVERTED_CHAIN),
}


@dataclass(frozen=True)
class TransferQuestion(CircuitQuestion):
    representation: Representation = Representation.EN
    names: Names = Names.SEEN
    symbols: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
    pair_id: str = ""

    @property
    def aliases(self):
        return dict(zip(CircuitNode, self.symbols, strict=True))

    def prompt(self, interventions=()):
        names = self.aliases
        s, w, r, b, l, f = (names[n] for n in CircuitNode)
        symbolic = self.representation == Representation.SYMBOLIC
        russian = self.representation == Representation.RU
        conjunction = "∧" if symbolic else "И" if russian else "AND"
        negation = "¬" if symbolic else "НЕ " if russian else "NOT "
        chain = self.world.scheme in PAIR_SCHEMES[Family.CHAIN]
        inverted = self.world.scheme in (Scheme.AND_INVERTED, Scheme.AND_INVERTED_CHAIN)
        expressions = {r: f"{s} {conjunction} {w}"}
        expressions[b if chain else l] = f"{negation if inverted else ''}{r}"
        if chain:
            expressions[l] = b
        for op in interventions:
            expressions[names[CircuitNode(op.node.value)]] = str(op.value)
        facts = [
            f"{name} = {value}"
            for name, value in (
                (s, self.world.source),
                (w, self.world.switch),
                (f, self.world.flag),
            )
        ]
        rules = [f"{name} = {value}" for name, value in expressions.items()]
        if symbolic:
            legend = "{0,1}; 0∧0=0; 0∧1=0; 1∧0=0; 1∧1=1; ¬0=1; ¬1=0."
            inputs, equations = "", ""
            question = f"{names[self.node]} = ?\n{names[self.node]} ="
        elif russian:
            legend = (
                "Все переменные равны 0 или 1. И даёт 1, только когда оба входа равны 1. "
                "НЕ меняет 0 на 1 и 1 на 0."
            )
            inputs, equations = "Входы:", "Уравнения:"
            question = (
                f"Чему равно {names[self.node]}? Ответь ровно одной цифрой: 0 или 1. "
                "Без слов, знаков препинания и объяснения.\nОтвет:"
            )
        else:
            legend = (
                "All variables are 0 or 1. AND gives 1 only when both inputs are 1. "
                "NOT changes 0 to 1 and 1 to 0."
            )
            inputs, equations = "Inputs:", "Equations:"
            question = (
                f"What is {names[self.node]}? Reply with exactly one digit: 0 or 1. "
                "Do not add words, punctuation, or an explanation.\nAnswer:"
            )
        if self.style == PromptStyle.QUERY_FIRST:
            facts, rules = facts[::-1], rules[::-1]
        sections = [[inputs, *facts], [equations, *rules]]
        if self.style == PromptStyle.VERBAL:
            sections.reverse()
        body = "\n".join([legend, *(line for section in sections for line in section if line)])
        return body + NEUTRAL_STATE_MARKER + "\n" + question


def transfer_questions(split, representation, *, names=Names.SEEN, family=Family.DIRECT, seed=42):
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    representation, names, family = Representation(representation), Names(names), Family(family)
    if split != "test" and (family != Family.DIRECT or names != Names.SEEN):
        raise ValueError("Renamed and chain conditions are test-only")
    # Held-out identifiers for source validation; a third alphabet for renamed tests.
    alphabet = (
        "GHIJKL" if split == "validation" else "UVWXYZ" if names == Names.RENAMED else "ABCDEF"
    )
    symbols = list(alphabet)
    random.Random(seed).shuffle(symbols)
    styles = tuple(PromptStyle) if split == "train" else (PromptStyle.DEFAULT,)
    result = []
    for scheme in PAIR_SCHEMES[family]:
        for i, values in enumerate(product((0, 1), repeat=3)):
            world = CircuitWorld(
                f"{split}/{representation}/{names}/{family}/{scheme}/{i}", *values, scheme
            )
            for style in styles:
                result.extend(
                    TransferQuestion(
                        world,
                        node,
                        split,
                        style,
                        layout=PromptLayout.STATE_FIRST,
                        representation=representation,
                        names=names,
                        symbols=tuple(symbols),
                        pair_id=f"{family}/{i}/{style.value}",
                    )
                    for node in world.values()
                )
    return tuple(result)
