from __future__ import annotations

import unittest

import numpy as np

from experiments.gave2_v12.metrics import teacher_path_recall
from experiments.gave2_v12.selection import calibrated_residual


class SelectionTests(unittest.TestCase):
    def test_zero_alpha_reproduces_teacher(self):
        rng = np.random.default_rng(7)
        teacher = rng.uniform(0.01, 0.99, size=(3, 19, 23)).astype(np.float32)
        teacher[1] = np.maximum.reduce((teacher[0], teacher[1], teacher[2]))
        candidate = 1.0 - teacher
        result = calibrated_residual(
            candidate,
            teacher,
            alpha=0.0,
            decision_threshold=0.5,
            temperature=1.0,
            corridor_radius=4,
        )
        np.testing.assert_allclose(result, teacher, atol=2e-6)

    def test_teacher_paths_are_preserved_and_crossings_remain_legal(self):
        teacher = np.zeros((3, 17, 17), dtype=np.float32)
        teacher[0, 8, 2:15] = 0.9
        teacher[2, 2:15, 8] = 0.9
        teacher[1] = np.maximum(teacher[0], teacher[2])
        candidate = np.zeros_like(teacher)
        result = calibrated_residual(
            candidate,
            teacher,
            alpha=1.0,
            decision_threshold=0.5,
            temperature=1.0,
            corridor_radius=2,
        )
        self.assertEqual(teacher_path_recall(result, teacher), 1.0)
        self.assertGreaterEqual(result[0, 8, 8], 0.5)
        self.assertGreaterEqual(result[2, 8, 8], 0.5)
        self.assertTrue(np.all(result[1] >= np.maximum(result[0], result[2])))

    def test_prune_mode_cannot_add_a_new_positive_pixel(self):
        teacher = np.zeros((3, 13, 17), dtype=np.float32)
        teacher[0, 6, 3:14] = 0.9
        teacher[1] = teacher[0]
        candidate = np.ones_like(teacher)
        result = calibrated_residual(
            candidate,
            teacher,
            alpha=1.0,
            decision_threshold=0.45,
            temperature=0.9,
            corridor_radius=2,
            correction_mode="prune",
        )
        self.assertFalse(bool(((result >= 0.5) & (teacher < 0.5)).any()))

    def test_task2_mode_can_reclassify_only_inside_teacher_vessel_support(self):
        teacher = np.zeros((3, 13, 17), dtype=np.float32)
        teacher[0, 6, 3:14] = 0.9
        teacher[1, 6, 3:14] = 0.9
        candidate = np.ones_like(teacher)
        result = calibrated_residual(
            candidate,
            teacher,
            alpha=1.0,
            decision_threshold=0.45,
            temperature=0.9,
            corridor_radius=2,
            correction_mode="vessel_support",
        )
        support = teacher[1] >= 0.5
        self.assertFalse(bool(((result[2] >= 0.5) & ~support).any()))


if __name__ == "__main__":
    unittest.main()
