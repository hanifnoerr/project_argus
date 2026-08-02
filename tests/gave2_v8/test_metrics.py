import unittest

import numpy as np

from experiments.gave2_v8.metrics import _score_components, path_counts


class MetricTests(unittest.TestCase):
    def test_perfect_line_paths_are_correct(self):
        target = np.zeros((48, 64), dtype=bool)
        target[24, 5:59] = True
        counts = path_counts(target, target, paths=60, case_id="perfect", seed=7)
        self.assertEqual(counts.total, 60)
        self.assertEqual(counts.correct, 60)
        self.assertEqual(counts.infeasible, 0)

    def test_empty_prediction_is_infeasible(self):
        target = np.zeros((48, 64), dtype=bool)
        target[24, 5:59] = True
        counts = path_counts(np.zeros_like(target), target, paths=40, case_id="empty", seed=9)
        self.assertEqual(counts.infeasible, 40)
        self.assertEqual(counts.correct, 0)

    def test_disconnected_prediction_has_infeasible_pairs(self):
        target = np.zeros((48, 64), dtype=bool)
        target[24, 5:59] = True
        prediction = target.copy()
        prediction[24, 30:35] = False
        counts = path_counts(prediction, target, paths=100, case_id="broken", seed=11)
        self.assertGreater(counts.infeasible, 0)
        self.assertLess(counts.correct, counts.total)

    def test_observed_score_weights_classification_and_topology_at_40_percent(self):
        score = _score_components(0.72, 0.9, 0.97, 0.97, 0.58, 0.38)
        self.assertAlmostEqual(score["classification"], 0.949)
        self.assertAlmostEqual(score["topology"], 0.60)
        self.assertAlmostEqual(score["score_observed"], 7.636)


if __name__ == "__main__":
    unittest.main()
