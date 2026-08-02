from __future__ import annotations

import unittest
from pathlib import Path

from experiments.gave2_v12.task3 import audit_frozen_task3, read_biomarker


ROOT = Path(__file__).resolve().parents[2]


class Task3Tests(unittest.TestCase):
    def test_proven_v8_task3_payload_is_valid(self):
        source = ROOT / "experiments/gave2_v8/assets/proven_task3"
        report = audit_frozen_task3(ROOT / "GAVE2_preliminary", source)
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["cases"], 50)

    def test_biomarker_avr_is_consistent(self):
        values = read_biomarker(ROOT / "experiments/gave2_v8/assets/proven_task3/g_051.txt")
        self.assertAlmostEqual(values["AVR"], values["CRAE"] / values["CRVE"], places=4)


if __name__ == "__main__":
    unittest.main()

