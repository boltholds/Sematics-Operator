"""Explicit Boolean SCMs: train on AND/copy, test on held-out mechanisms/topology."""

from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from .world import Intervention, Node


class Scheme(StrEnum):
    AND_COPY = "and_copy"
    OR_COPY = "or_copy"
    AND_GATED = "and_gated"
    AND_CHAIN = "and_chain"


class CircuitNode(StrEnum):
    SOURCE = "source"
    SWITCH = "switch"
    RELAY = "relay"
    BRIDGE = "bridge"
    LAMP = "lamp"
    FLAG = "flag"


@dataclass(frozen=True)
class CircuitWorld:
    name: str
    source: int
    switch: int
    flag: int
    scheme: Scheme

    def values(self, interventions=()):
        overrides = {op.node.value: op.value for op in interventions}
        relay = (
            self.source | self.switch
            if self.scheme == Scheme.OR_COPY
            else self.source & self.switch
        )
        relay = overrides.get("relay", relay)
        lamp = relay & self.flag if self.scheme == Scheme.AND_GATED else relay
        values = {
            CircuitNode.SOURCE: self.source,
            CircuitNode.SWITCH: self.switch,
            CircuitNode.RELAY: relay,
            CircuitNode.LAMP: overrides.get("lamp", lamp),
            CircuitNode.FLAG: self.flag,
        }
        if self.scheme == Scheme.AND_CHAIN:
            values[CircuitNode.BRIDGE] = relay
        return values

    def affected(self, op):
        if op.node == Node.LAMP:
            return {CircuitNode.LAMP}
        nodes = {CircuitNode.RELAY, CircuitNode.LAMP}
        if self.scheme == Scheme.AND_CHAIN:
            nodes.add(CircuitNode.BRIDGE)
        return nodes


@dataclass(frozen=True)
class CircuitQuestion:
    world: CircuitWorld
    node: CircuitNode
    split: str

    @property
    def key(self):
        return f"{self.world.name}/{self.node.value}"

    def answer(self, interventions=()):
        return self.world.values(interventions)[self.node]

    def prompt(self, interventions=()):
        w = self.world
        gate = "OR" if w.scheme == Scheme.OR_COPY else "AND"
        rules = f"relay = source {gate} switch; "
        if w.scheme == Scheme.AND_GATED:
            rules += "lamp = relay AND flag. "
        elif w.scheme == Scheme.AND_CHAIN:
            rules += "bridge = relay; lamp = bridge. "
        else:
            rules += "lamp = relay. "
        facts = f"source={w.source}; switch={w.switch}; flag={w.flag}. "
        if self.split == "train":
            text = f"Circuit {w.name}: {facts}Rules: {rules}"
        elif self.split == "validation":
            text = f"Evaluate circuit {w.name}. Rules: {rules}Given inputs: {facts}"
        else:
            text = f"Consider device {w.name}. Its equations are: {rules}Input settings: {facts}"
        if interventions:
            overrides = {op.node: op.value for op in interventions}
            text += (
                "Override these rules: "
                + "; ".join(f"force {n.value}={v}" for n, v in overrides.items())
                + ". "
            )
        return text + f"What is {self.node.value}? Answer with only 0 or 1.\nAnswer:"


def circuit_questions(split, scheme=Scheme.AND_COPY):
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    if split != "test" and scheme != Scheme.AND_COPY:
        raise ValueError("Shifted schemes are test-only")
    result = []
    for i, values in enumerate(product((0, 1), repeat=3)):
        w = CircuitWorld(f"{split}_{scheme.value}_{i}", *values, scheme)
        result.extend(CircuitQuestion(w, node, split) for node in w.values())
    return tuple(result)


@dataclass(frozen=True)
class InterchangePair:
    base: CircuitQuestion
    source: CircuitQuestion
    node: Node

    @property
    def intervention(self):
        return Intervention(self.node, self.source.world.values()[self.node.value])

    @property
    def expected(self):
        return self.base.answer((self.intervention,))


def interchange_pairs(samples):
    """All ordered worlds, aligned query roles; donors contain no override instruction."""
    return tuple(
        InterchangePair(base, source, node)
        for node in (Node.RELAY, Node.LAMP)
        for base in samples
        for source in samples
        if source.node == base.node
    )


def metrics(scores, samples, sequence, baseline):
    from .conditional import protected_damage
    from .experiment import summarize

    result = summarize(scores, samples, sequence, baseline)
    result["by_node"] = {}
    for node in dict.fromkeys(q.node.value for q in samples):
        records = [r for r in result["records"] if r["node"] == node]
        result["by_node"][node] = {
            "count": len(records),
            "accuracy": sum(r["prediction"] == r["expected"] for r in records) / len(records),
        }
    groups = {}
    for q, pred in zip(samples, scores.argmax(-1).tolist(), strict=True):
        groups.setdefault(q.world.name, []).append((q, pred))
    complete, pair = [], []
    for group in groups.values():
        complete.append(all(pred == q.answer(sequence) for q, pred in group))
        pair.append(
            all(pred == q.answer(sequence) for q, pred in group if q.node in ("relay", "lamp"))
        )
    result["all_nodes_correct"] = sum(complete) / len(complete)
    result["relay_lamp_correct"] = sum(pair) / len(pair)
    result["protected_damage"] = protected_damage(scores, samples, baseline)
    return result


def selection(metrics, weight):
    return (
        metrics["all_nodes_correct"] - weight * metrics["protected_damage"]["rate"],
        metrics["relay_lamp_correct"],
        metrics["overall"]["accuracy"],
    )
