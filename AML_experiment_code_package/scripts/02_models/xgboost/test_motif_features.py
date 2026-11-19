import unittest
from collections import defaultdict

from build_motif_features import (
    directed_cycles_4_10,
    timed_fan_in_then_out,
    timed_in_then_fan_out,
)


class MotifFeatureTests(unittest.TestCase):
    def test_five_node_directed_cycle_and_noncycle(self):
        outgoing = defaultdict(set)
        for source, destination in zip("abcde", "bcdea"):
            outgoing[source].add(destination)
        outgoing["f"].add("a")
        for node in "abcde":
            self.assertEqual(directed_cycles_4_10(node, outgoing), (1, 5))
        self.assertEqual(directed_cycles_4_10("f", outgoing), (0, 0))

    def test_three_node_cycle_is_outside_target_range(self):
        outgoing = defaultdict(set, {"a": {"b"}, "b": {"c"}, "c": {"a"}})
        self.assertEqual(directed_cycles_4_10("a", outgoing), (0, 0))

    def test_timed_fan_patterns_obey_direction_and_window(self):
        incoming = [(1, "s1"), (3, "s2"), (20, "old")]
        outgoing = [(4, "r1"), (4, "r2"), (40, "late")]
        self.assertEqual(timed_fan_in_then_out(incoming, outgoing), 2)
        self.assertEqual(timed_in_then_fan_out(incoming, outgoing), 2)


if __name__ == "__main__":
    unittest.main()
