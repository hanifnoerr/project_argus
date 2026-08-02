from __future__ import annotations

import unittest

import numpy as np

from experiments.gave2_v13.task3 import Candidate, _domain_audit, _fit, _predict


class Task3Tests(unittest.TestCase):
    def test_ridge_prediction_tracks_positive_target(self):
        x = np.linspace(-1.0, 1.0, 50)[:, None]
        features = np.concatenate((x, x**2), axis=1)
        target = np.exp(0.2 + 0.35 * x[:, 0])
        model = _fit(features, target, Candidate("ffa_only", (0,), 1.0, 1.0))
        prediction = _predict(model, features)
        self.assertLess(float(np.mean(np.abs(prediction - target))), 0.03)
        self.assertTrue(bool((prediction > 0).all()))

    def test_domain_audit_rejects_large_shift(self):
        training = np.linspace(-1.0, 1.0, 50)[:, None]
        target = np.exp(0.2 + 0.1 * training[:, 0])
        model = _fit(training, target, Candidate("ffa_only", (0,), 1.0, 1.0))
        report, accepted = _domain_audit(
            training,
            training + 8.0,
            {"AVR": model},
            max_mean_shift=1.0,
            max_abs_z=8.0,
        )
        self.assertEqual(accepted, [])
        self.assertFalse(report["targets"]["AVR"]["passed"])


if __name__ == "__main__":
    unittest.main()
