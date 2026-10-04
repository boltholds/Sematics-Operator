"""Known unary Boolean mechanisms; synthetic supervision, not causal discovery."""

import random
from dataclasses import dataclass
from enum import IntEnum


class Rule(IntEnum):
    ROOT = 0
    COPY = 1
    NOT = 2


@dataclass(frozen=True)
class Circuit:
    parents: tuple[int, ...]
    rules: tuple[Rule, ...]
    source_values: tuple[int, ...]

    def __post_init__(self):
        n = len(self.parents)
        if not n or len(self.rules) != n or len(self.source_values) != n:
            raise ValueError("Circuit arrays must have the same nonzero length")
        for i, (p, rule, value) in enumerate(zip(self.parents, self.rules, self.source_values)):
            if p < 0 or p >= n or rule not in tuple(Rule) or value not in (0, 1):
                raise ValueError("Invalid parent, rule, or source bit")
            if rule == Rule.ROOT and p != i:
                raise ValueError("ROOT must point to itself")
        self.depths()  # also rejects cycles in arbitrarily numbered graphs

    def depths(self):
        memo, active = {}, set()

        def visit(i):
            if i in active:
                raise ValueError("Circuit contains a cycle")
            if i not in memo:
                active.add(i)
                memo[i] = 0 if self.rules[i] == Rule.ROOT else visit(self.parents[i]) + 1
                active.remove(i)
            return memo[i]

        return tuple(visit(i) for i in range(len(self.parents)))

    def depth(self):
        return max(self.depths())

    def equilibrium(self):
        values = list(self.source_values)
        for i in sorted(range(len(values)), key=self.depths().__getitem__):
            if self.rules[i] != Rule.ROOT:
                values[i] = values[self.parents[i]] ^ (self.rules[i] == Rule.NOT)
        return tuple(map(int, values))

    def descendants(self, node):
        found = set()
        frontier = [node]
        while frontier:
            parent = frontier.pop()
            children = [
                i for i, p in enumerate(self.parents) if p == parent and self.rules[i] != Rule.ROOT
            ]
            found.update(children)
            frontier.extend(children)
        return found

    def topology_key(self):
        """Unlabelled rooted-forest signature, ignoring rules AND root values."""

        def encode(i):
            children = [
                encode(j)
                for j, p in enumerate(self.parents)
                if p == i and self.rules[j] != Rule.ROOT
            ]
            return "(" + "".join(sorted(children)) + ")"

        return "".join(sorted(encode(i) for i, r in enumerate(self.rules) if r == Rule.ROOT))

    def permute(self, order):
        if sorted(order) != list(range(len(self.parents))):
            raise ValueError("Order must be a permutation of all nodes")
        inverse = {old: new for new, old in enumerate(order)}
        return Circuit(
            tuple(inverse[self.parents[i]] for i in order),
            tuple(self.rules[i] for i in order),
            tuple(self.source_values[i] for i in order),
        )


@dataclass(frozen=True)
class Episode:
    id: str
    graph: Circuit
    interventions: tuple[tuple[int, int], ...]

    def __post_init__(self):
        targets = [i for i, _ in self.interventions]
        if len(set(targets)) != len(targets):
            raise ValueError("Duplicate intervention target")
        if any(
            i < 0 or i >= len(self.graph.parents) or v not in (0, 1) for i, v in self.interventions
        ):
            raise ValueError("Invalid intervention target or value")

    def permute(self, order):
        graph = self.graph.permute(order)
        inverse = {old: new for new, old in enumerate(order)}
        return Episode(self.id, graph, tuple((inverse[i], v) for i, v in self.interventions))

    def initial(self):
        values = list(self.graph.equilibrium())
        for i, v in self.interventions:
            values[i] = v
        return tuple(values)

    def free(self):
        targets = dict(self.interventions)
        return tuple(r != Rule.ROOT and i not in targets for i, r in enumerate(self.graph.rules))

    def protected(self):
        affected = set(dict(self.interventions))
        for i, _ in self.interventions:
            affected.update(self.graph.descendants(i))
        return tuple(i not in affected for i in range(len(self.graph.parents)))


def oracle_step(episode, state):
    if len(state) != len(episode.graph.parents):
        raise ValueError("State size does not match graph")
    values = list(episode.graph.source_values)
    for i, rule in enumerate(episode.graph.rules):
        if rule != Rule.ROOT:
            values[i] = int(state[episode.graph.parents[i]]) ^ (rule == Rule.NOT)
    for i, v in episode.interventions:
        values[i] = v
    return tuple(map(int, values))


def oracle_trace(episode, steps):
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    states = [episode.initial()]
    for _ in range(steps):
        states.append(oracle_step(episode, states[-1]))
    return states


def _graph(rng, long=False):
    n = rng.randint(11, 14) if long else rng.randint(6, 9)
    depth = rng.randint(6, 8) if long else rng.randint(2, 3)
    parents, rules, depths = [0], [Rule.ROOT], [0]
    for i in range(1, n):
        if i == n - 1 or (i > depth and rng.random() < 0.25):
            p, rule, d = i, Rule.ROOT, 0
        else:
            p = i - 1 if i <= depth else rng.choice([j for j, d in enumerate(depths) if d < depth])
            rule, d = rng.choice([Rule.COPY, Rule.NOT]), depths[p] + 1
        parents.append(p)
        rules.append(rule)
        depths.append(d)
    g = Circuit(tuple(parents), tuple(rules), tuple(rng.randrange(2) for _ in parents))
    order = list(range(n))
    rng.shuffle(order)
    return g.permute(order)


def make_dataset(seed=42, train_graphs=64, eval_graphs=16):
    if train_graphs < 1 or eval_graphs < 1:
        raise ValueError("Graph counts must be positive")
    rng, used, result = random.Random(seed), set(), {}
    for split, count in (
        ("train", train_graphs),
        ("validation", eval_graphs),
        ("test", eval_graphs),
        ("long", eval_graphs),
    ):
        graphs = []
        for _ in range(100000):
            graph = _graph(rng, split == "long")
            key = graph.topology_key()
            if key not in used:
                used.add(key)
                graphs.append(graph)
            if len(graphs) == count:
                break
        if len(graphs) != count:
            raise ValueError("Cannot generate enough disjoint topologies; reduce graph counts")
        result[split] = tuple(graphs)
    return result


def episodes_for(graphs, prefix, composition=False):
    result = []
    for k, g in enumerate(graphs):
        targets = [i for i, r in enumerate(g.rules) if r != Rule.ROOT and g.descendants(i)]
        # First internal node by causal depth; IDs are already randomly permuted.
        target = min(targets, key=lambda i: (g.depths()[i], i))
        second = next(
            (i for i in targets if i != target),
            next(i for i, r in enumerate(g.rules) if r != Rule.ROOT and i != target),
        )
        for value in (0, 1):
            edits = ((target, value), (second, 1 - value)) if composition else ((target, value),)
            result.append(Episode(f"{prefix}/{k}/{value}", g, edits))
    return result
