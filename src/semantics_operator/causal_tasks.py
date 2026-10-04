"""Explicit Boolean SCMs: train on AND/copy, test on held-out mechanisms/topology."""

from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from .answer_protocol import ANSWER_INSTRUCTION
from .world import Intervention, Node


class Scheme(StrEnum):
    AND_COPY = "and_copy"
    OR_COPY = "or_copy"
    AND_GATED = "and_gated"
    AND_CHAIN = "and_chain"
    AND_INVERTED = "and_inverted"
    AND_XOR = "and_xor"


class CircuitNode(StrEnum):
    SOURCE = "source"
    SWITCH = "switch"
    RELAY = "relay"
    BRIDGE = "bridge"
    LAMP = "lamp"
    FLAG = "flag"


class PromptStyle(StrEnum):
    DEFAULT = "default"
    VERBAL = "verbal"
    QUERY_FIRST = "query_first"


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
        if self.scheme == Scheme.AND_INVERTED:
            lamp = 1 - relay
        elif self.scheme == Scheme.AND_XOR:
            lamp = relay ^ self.flag
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
    style: PromptStyle = PromptStyle.DEFAULT

    @property
    def key(self):
        style = "" if self.style == PromptStyle.DEFAULT else f"/{self.style.value}"
        return f"{self.world.name}{style}/{self.node.value}"

    @property
    def state_key(self):
        return self.world.name, self.style

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
        elif w.scheme == Scheme.AND_INVERTED:
            rules += "lamp = NOT relay (NOT 0 = 1; NOT 1 = 0). "
        elif w.scheme == Scheme.AND_XOR:
            rules += "lamp = relay XOR flag (XOR is 1 exactly when its inputs differ). "
        else:
            rules += "lamp = relay. "
        facts = f"source={w.source}; switch={w.switch}; flag={w.flag}. "
        if self.split == "train":
            text = f"Circuit {w.name}: {facts}Rules: {rules}"
        elif self.split == "validation":
            text = f"Evaluate circuit {w.name}. Rules: {rules}Given inputs: {facts}"
        else:
            text = f"Consider device {w.name}. Its equations are: {rules}Input settings: {facts}"
        if self.style == PromptStyle.VERBAL:
            header = {
                "train": "Recorded circuit",
                "validation": "Evaluate this recorded system",
                "test": "Inspect the following device",
            }[self.split]
            text = (
                f"{header} {w.name}. The source has value {w.source}, the switch has value "
                f"{w.switch}, and the flag has value {w.flag}. The equations are: {rules}"
            )
        elif self.style == PromptStyle.QUERY_FIRST:
            header = {
                "train": "Find",
                "validation": "Determine",
                "test": "Report",
            }[self.split]
            text = f"{header} {self.node.value} for {w.name}. Equations: {rules}Inputs: {facts}"
        if interventions:
            overrides = {op.node: op.value for op in interventions}
            text += (
                "Override these rules: "
                + "; ".join(f"force {n.value}={v}" for n, v in overrides.items())
                + ". "
            )
        return text + f"What is {self.node.value}? " + ANSWER_INSTRUCTION + "\nAnswer:"


def circuit_questions(split, scheme=Scheme.AND_COPY, *, styles=(PromptStyle.DEFAULT,)):
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    if split != "test" and scheme != Scheme.AND_COPY:
        raise ValueError("Shifted schemes are test-only")
    styles = tuple(PromptStyle(s) for s in styles)
    if not styles or len(set(styles)) != len(styles):
        raise ValueError("Provide unique prompt styles")
    result = []
    for i, values in enumerate(product((0, 1), repeat=3)):
        w = CircuitWorld(f"{split}_{scheme.value}_{i}", *values, scheme)
        result.extend(
            CircuitQuestion(w, node, split, style) for style in styles for node in w.values()
        )
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
    for record, q in zip(result["records"], samples, strict=True):
        record["style"] = q.style.value
    breakdown = answer_metrics(result["records"])
    for field in ("by_label", "by_style", "by_node_label", "prediction_counts"):
        result[field] = breakdown[field]
    result["by_node"] = {}
    for node in dict.fromkeys(q.node.value for q in samples):
        records = [r for r in result["records"] if r["node"] == node]
        result["by_node"][node] = {
            "count": len(records),
            "accuracy": sum(r["prediction"] == r["expected"] for r in records) / len(records),
        }
    groups = {}
    for q, pred in zip(samples, scores.argmax(-1).tolist(), strict=True):
        groups.setdefault(q.state_key, []).append((q, pred))
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


def answer_metrics(records):
    """Invalid/missing generated digits count as errors, never disappear from denominators."""

    def group(rows):
        correct = sum(r["prediction"] == r["expected"] for r in rows)
        return {
            "count": len(rows),
            "accuracy": correct / len(rows) if rows else None,
            "errors": len(rows) - correct,
            "invalid": sum(r["prediction"] is None for r in rows),
            "incomplete": sum(r.get("generation_status") == "incomplete" for r in rows),
            "format_errors": sum(r.get("generation_status") == "format_error" for r in rows),
        }

    nodes = dict.fromkeys(r["node"] for r in records)
    return {
        "overall": group(records),
        "by_label": {str(v): group([r for r in records if r["expected"] == v]) for v in (0, 1)},
        "by_node": {n: group([r for r in records if r["node"] == n]) for n in nodes},
        "by_node_label": {
            n: {
                str(v): group([r for r in records if r["node"] == n and r["expected"] == v])
                for v in (0, 1)
            }
            for n in nodes
        },
        "by_style": {
            s: group([r for r in records if r["style"] == s])
            for s in dict.fromkeys(r["style"] for r in records)
        },
        "prediction_counts": {
            str(v) if v is not None else "invalid": sum(r["prediction"] == v for r in records)
            for v in (0, 1, None)
        },
        "records": records,
    }


def selection(metrics, weight):
    return (
        metrics["all_nodes_correct"] - weight * metrics["protected_damage"]["rate"],
        metrics["relay_lamp_correct"],
        metrics["overall"]["accuracy"],
    )
