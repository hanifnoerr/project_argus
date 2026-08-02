from __future__ import annotations

import unittest

import numpy as np

from experiments.gave2_v12.registration import fit_registration


class RegistrationTests(unittest.TestCase):
    def test_similarity_fit_maps_moving_ffa_to_fixed_cfp(self):
        rng = np.random.default_rng(77)
        moving = rng.uniform((100, 100), (1436, 924), size=(80, 2))
        angle = np.deg2rad(1.2)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        fixed = moving @ rotation.T + np.array((13.0, -8.0))
        fixed += rng.normal(0.0, 0.3, fixed.shape)
        matrix, qa = fit_registration(moving, fixed, np.ones(len(moving)), (1024, 1536))
        homogeneous = np.column_stack((moving, np.ones(len(moving))))
        projected = (homogeneous @ matrix.T)[:, :2]
        self.assertTrue(qa.accepted)
        self.assertEqual(qa.model, "similarity")
        self.assertLess(float(np.median(np.linalg.norm(projected - fixed, axis=1))), 1.0)

    def test_implausible_sparse_matches_fall_back_to_identity(self):
        moving = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        fixed = moving + 100.0
        matrix, qa = fit_registration(moving, fixed, None, (1024, 1536))
        np.testing.assert_array_equal(matrix, np.eye(3))
        self.assertFalse(qa.accepted)


if __name__ == "__main__":
    unittest.main()

