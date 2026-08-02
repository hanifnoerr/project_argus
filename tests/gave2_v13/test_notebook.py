from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from experiments.gave2_v13 import RUNTIME_BUILD_ID


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "submission/GAVE2_Channel_Path_FFA_V13_Colab.ipynb"


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join("".join(cell.get("source", [])) for cell in cls.notebook["cells"])

    def test_notebook_is_clean_and_code_parses(self):
        self.assertGreaterEqual(len(self.notebook["cells"]), 20)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))

    def test_release_contract_is_explicit(self):
        self.assertIn('ARCHIVE_PATH = DRIVE_BASE / "miccai_v13.zip"', self.source)
        self.assertIn('"submission/GAVE2_R2V2_FFA_Residual_V12_Colab.ipynb"', self.source)
        self.assertIn('V12_RUN_DIR = DRIVE_BASE / "runs/gave2_v12_safe_3fold"', self.source)
        self.assertIn('V12_SELECTED = V12_RUN_DIR / "predictions/selected"', self.source)
        self.assertIn(f'EXPECTED_RUNTIME_BUILD_ID = "{RUNTIME_BUILD_ID}"', self.source)
        self.assertIn("torch.cuda.is_bf16_supported()", self.source)
        self.assertNotIn('assert "L4"', self.source)
        self.assertIn('train_task("task2", epochs=80, minimum_epochs=25)', self.source)
        self.assertIn('train_task("task1", epochs=55, minimum_epochs=20)', self.source)
        self.assertIn('"--early-stopping-patience", 7', self.source)
        self.assertIn('"--paths-per-case", 100', self.source)
        self.assertIn('"--search-mode", "topology_safe"', self.source)
        self.assertIn('"--maximum-reassignment-fraction", 0.0', self.source)
        self.assertIn('SELECTION_ROOT = RUN_DIR / "selection_r51"', self.source)
        self.assertIn('"--decision-threshold-values", 0.575, 0.585, 0.595', self.source)
        self.assertIn('RELEASE_TARGET = 7.7', self.source)
        self.assertIn('"--required-task3-targets", "vein_density"', self.source)
        self.assertIn('"--segmentation-transfer-scale", 1.0', self.source)
        self.assertIn("keep V12", self.source)
        self.assertIn("experiments.gave2_v13.task3", self.source)
        self.assertIn("experiments.gave2_v13.release", self.source)
        self.assertIn("READY_FOR_ONE_CAUTIOUS_SUBMISSION", self.source)
        self.assertIn("PORTAL_MAXIMUM_BYTES = 100_000_000", self.source)
        self.assertIn('submission["threshold_mismatch_pixels"] == 0', self.source)
        self.assertIn('decision["compact_valid"]', self.source)
        self.assertIn("stop_and_disconnect", self.source)
        self.assertIn("runtime.unassign()", self.source)
        self.assertIn('TEAM_ID = "梯度不下降队"', self.source)
        self.assertNotIn("hotfix", self.source.lower())
        self.assertNotIn("apply embedded", self.source.lower())


if __name__ == "__main__":
    unittest.main()
