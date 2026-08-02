from __future__ import annotations

import unittest

import numpy as np

from experiments.gave2_v12.gate import structural_counts


class GateTests(unittest.TestCase):
    def test_prune_mode_rejects_new_geometry_and_detects_lost_path(self):
        teacher = np.zeros((3, 15, 19), dtype=np.float32)
        teacher[0, 7, 2:17] = 0.9
        teacher[1] = teacher[0]
        candidate = teacher.copy()
        candidate[0, 2, 2] = 0.9
        candidate[0, 7, 9] = 0.0
        counts = structural_counts(candidate, teacher, "prune")
        self.assertEqual(counts["off_support_additions"], 1)
        self.assertGreater(counts["protected_skeleton_missing"], 0)

    def test_task2_reclassification_is_limited_to_teacher_vessel_support(self):
        teacher = np.zeros((3, 11, 13), dtype=np.float32)
        teacher[0, 5, 2:11] = 0.9
        teacher[1, 5, 2:11] = 0.9
        candidate = teacher.copy()
        candidate[2, 5, 6] = 0.9
        counts = structural_counts(candidate, teacher, "vessel_support")
        self.assertEqual(counts["off_support_additions"], 0)
        self.assertEqual(counts["class_reassignment_additions"], 1)


if __name__ == "__main__":
    unittest.main()
