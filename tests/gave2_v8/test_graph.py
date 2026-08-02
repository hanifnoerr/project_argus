import unittest

import numpy as np

from experiments.gave2_v8.graph import GraphParameters, graph_refine_probability


class GraphTests(unittest.TestCase):
    def test_crossing_keeps_both_classes(self):
        probability = np.full((3, 65, 65), 0.02, dtype=np.float32)
        probability[0, 32, 5:60] = 0.90
        probability[2, 5:60, 32] = 0.90
        probability[1] = np.maximum(probability[0], probability[2])
        probability[0, 30:35, 30:35] = np.maximum(probability[0, 30:35, 30:35], 0.52)
        probability[2, 30:35, 30:35] = np.maximum(probability[2, 30:35, 30:35], 0.52)
        refined, diagnostics = graph_refine_probability(
            probability,
            GraphParameters(min_component_size=4, grow_threshold=0.10),
        )
        self.assertGreater(diagnostics["segments"], 0)
        self.assertGreaterEqual(float(refined[0, 32, 32]), 0.5)
        self.assertGreaterEqual(float(refined[2, 32, 32]), 0.5)
        self.assertTrue(np.all(refined[1] >= np.maximum(refined[0], refined[2])))

    def test_single_branch_uses_pooled_artery_evidence(self):
        probability = np.full((3, 33, 65), 0.02, dtype=np.float32)
        probability[0, 16, 4:61] = 0.75
        probability[2, 16, 4:61] = 0.30
        probability[1, 16, 4:61] = 0.90
        refined, _ = graph_refine_probability(
            probability,
            GraphParameters(min_component_size=4, grow_threshold=0.10),
        )
        self.assertTrue(np.all(refined[0, 16, 8:57] >= 0.5))
        self.assertTrue(np.all(refined[2, 16, 8:57] < 0.5))


if __name__ == "__main__":
    unittest.main()

