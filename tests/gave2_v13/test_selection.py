from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from experiments.gave2_v13.selection import (
    PathSettings,
    TopologySafeSettings,
    geodesic_hysteresis,
    parse_args,
    topology_safe_prune,
)


class SelectionTests(unittest.TestCase):
    def test_fine_search_grid_is_configurable(self):
        args = parse_args(
            [
                "search",
                "--data-root", ".",
                "--fold-manifest", "folds.json",
                "--teacher-store", "teacher",
                "--raw-store", "raw",
                "--output-config", "selection.json",
                "--task", "task2",
                "--alpha-values", "1.0",
                "--decision-threshold-values", "0.575", "0.59", "0.605",
                "--temperature-values", "0.85", "0.9", "0.95",
            ]
        )
        self.assertEqual(args.alpha_values, [1.0])
        self.assertEqual(args.decision_threshold_values, [0.575, 0.59, 0.605])
        self.assertEqual(args.temperature_values, [0.85, 0.9, 0.95])

    def test_topology_safe_prune_never_adds_class_pixels_and_preserves_centerline(self):
        teacher = np.zeros((3, 64, 96), dtype=np.float32)
        teacher[1, 25:39, 8:88] = 0.92
        teacher[0, 28:33, 8:48] = 0.72
        teacher[2, 32:37, 48:88] = 0.72
        raw = teacher.copy()
        raw[0, 28:33, 8:48] = 0.20
        raw[2, 32:37, 48:88] = 0.20
        raw[0, 28:33, 52:80] = 0.95

        result = topology_safe_prune(
            raw,
            teacher,
            TopologySafeSettings(
                alpha=1.0,
                decision_threshold=0.55,
                temperature=1.0,
                corridor_radius=1,
            ),
        )

        teacher_positive = teacher[(0, 2), ...] >= 0.5
        result_positive = result[(0, 2), ...] >= 0.5
        self.assertFalse(bool((result_positive & ~teacher_positive).any()))
        self.assertGreater(int((teacher_positive & ~result_positive).sum()), 0)
        for channel in (0, 2):
            centerline = skeletonize(teacher[channel] >= 0.5)
            self.assertTrue(bool((result[channel, centerline] >= 0.5).all()))
        self.assertTrue(np.all(result[1] >= np.maximum(result[0], result[2])))

    def test_geodesic_reassignment_stays_on_vessel_support(self):
        teacher = np.zeros((3, 64, 96), dtype=np.float32)
        teacher[1, 27:37, 8:88] = 0.95
        teacher[0, 29:32, 8:48] = 0.90
        teacher[2, 33:36, 48:88] = 0.90
        raw = teacher.copy()
        raw[0, 29:32, 42:78] = 0.90
        raw[2, 33:36, 42:55] = 0.12
        settings = PathSettings(1.0, 0.62, 0.30, 0.85, support_radius=2)
        result = geodesic_hysteresis(raw, teacher, settings)

        support = ndimage.binary_dilation(teacher[1] >= settings.support_threshold, iterations=2)
        additions = (result[(0, 2), ...] >= 0.5) & ~(teacher[(0, 2), ...] >= 0.5)
        self.assertFalse(bool((additions & ~support[None]).any()))
        self.assertGreater(int((result[0] >= 0.5).sum()), int((teacher[0] >= 0.5).sum()))
        self.assertTrue(np.all(result[1] >= np.maximum(result[0], result[2])))

    def test_non_crossing_overlap_is_resolved_by_raw_confidence(self):
        teacher = np.zeros((3, 32, 48), dtype=np.float32)
        teacher[1, 14:18, 4:44] = 0.9
        teacher[0, 15:17, 4:44] = 0.7
        teacher[2, 15:17, 4:44] = 0.7
        raw = teacher.copy()
        raw[0, 15:17, 4:24] = 0.8
        raw[2, 15:17, 4:24] = 0.4
        raw[0, 15:17, 24:44] = 0.4
        raw[2, 15:17, 24:44] = 0.8
        result = geodesic_hysteresis(raw, teacher, PathSettings(1.0, 0.62, 0.3, 0.65))
        overlap = (result[0] >= 0.5) & (result[2] >= 0.5)
        self.assertFalse(bool(overlap.any()))


if __name__ == "__main__":
    unittest.main()
