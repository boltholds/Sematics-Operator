import unittest


class WorldTests(unittest.TestCase):
    def test_intervention_breaks_incoming_edge_preserves_other_nodes(self):
        from semantics_operator.world import Intervention, Node, World

        w = World("circuit", 1, 1, 0)
        self.assertEqual(
            w.values(), {Node.SOURCE: 1, Node.SWITCH: 1, Node.RELAY: 1, Node.LAMP: 1, Node.FLAG: 0}
        )
        values = w.values((Intervention(Node.RELAY, 0),))
        self.assertEqual(values[Node.LAMP], 0)
        self.assertEqual(values[Node.SOURCE], 1)
        self.assertEqual(values[Node.SWITCH], 1)
        self.assertEqual(values[Node.FLAG], 0)

    def test_composition_and_revision(self):
        from semantics_operator.world import Intervention, Node, World

        w = World("x", 1, 1, 0)
        a, b = Intervention(Node.RELAY, 0), Intervention(Node.LAMP, 1)
        self.assertEqual(w.values((a, b))[Node.LAMP], 1)
        self.assertEqual(w.values((a, Intervention(Node.RELAY, 1)))[Node.RELAY], 1)
        self.assertEqual(w.values((a, b)), w.values((b, a)))

    def test_split_has_disjoint_world_names_and_both_target_values(self):
        from semantics_operator.world import Node, dataset

        train, test = dataset("train"), dataset("test")
        self.assertFalse({w.name for w in train} & {w.name for w in test})
        self.assertEqual({w.values()[Node.LAMP] for w in test}, {0, 1})
