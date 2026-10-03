"""An external Boolean structural causal model used only as an experiment oracle."""

from dataclasses import dataclass
from enum import StrEnum
from itertools import product


class Node(StrEnum):
    SOURCE = "source"
    SWITCH = "switch"
    RELAY = "relay"
    LAMP = "lamp"
    FLAG = "flag"


@dataclass(frozen=True)
class Intervention:
    node: Node
    value: int

    def __post_init__(self):
        if self.node not in (Node.RELAY, Node.LAMP) or self.value not in (0, 1):
            raise ValueError("Interventions support relay/lamp with binary values")

    @property
    def key(self) -> str:
        return f"{self.node.value}_{self.value}"


@dataclass(frozen=True)
class World:
    name: str
    source: int
    switch: int
    flag: int

    def __post_init__(self):
        if any(v not in (0, 1) for v in (self.source, self.switch, self.flag)):
            raise ValueError("World values must be binary")

    def values(self, interventions: tuple[Intervention, ...] = ()) -> dict[Node, int]:
        overrides = {i.node: i.value for i in interventions}
        relay = overrides.get(Node.RELAY, self.source & self.switch)
        lamp = overrides.get(Node.LAMP, relay)
        return {
            Node.SOURCE: self.source,
            Node.SWITCH: self.switch,
            Node.RELAY: relay,
            Node.LAMP: lamp,
            Node.FLAG: self.flag,
        }

    def prompt(
        self, node: Node, *, paraphrase: bool = False, interventions: tuple[Intervention, ...] = ()
    ) -> str:
        if paraphrase:
            text = (
                f"Consider device {self.name}. The lamp copies the relay. "
                "The relay is 1 exactly when both source and switch are 1; otherwise it is 0. "
                f"The source is {self.source}, the switch is {self.switch}, "
                f"and an unrelated flag is {self.flag}. "
            )
        else:
            text = (
                f"Circuit {self.name}: source={self.source}; switch={self.switch}; "
                f"flag={self.flag}. Rules: relay = source AND switch; lamp = relay. "
                "The flag is independent. "
            )
        if interventions:
            overrides = {i.node: i.value for i in interventions}
            text += (
                "Override these rules: "
                + "; ".join(f"force {n.value}={v}" for n, v in overrides.items())
                + ". "
            )
        return text + f"What is {node.value}? Answer with only 0 or 1.\nAnswer:"


@dataclass(frozen=True)
class Question:
    world: World
    node: Node
    paraphrase: bool = False

    @property
    def key(self) -> str:
        return f"{self.world.name}/{self.node.value}/{int(self.paraphrase)}"

    def prompt(self, interventions: tuple[Intervention, ...] = ()) -> str:
        return self.world.prompt(self.node, paraphrase=self.paraphrase, interventions=interventions)

    def answer(self, interventions: tuple[Intervention, ...] = ()) -> int:
        return self.world.values(interventions)[self.node]


def dataset(split: str) -> tuple[World, ...]:
    prefixes = {"train": "training", "test": "unseen"}
    if split not in prefixes:
        raise ValueError(f"Unknown split: {split}")
    return tuple(
        World(f"{prefixes[split]}_{i}", *values)
        for i, values in enumerate(product((0, 1), repeat=3))
    )


def questions(split: str) -> tuple[Question, ...]:
    return tuple(Question(w, n, split == "test") for w in dataset(split) for n in Node)


OPERATORS = (Intervention(Node.RELAY, 0), Intervention(Node.RELAY, 1), Intervention(Node.LAMP, 1))
