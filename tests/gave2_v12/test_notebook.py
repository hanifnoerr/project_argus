from __future__ import annotations

import ast
import hashlib
import json
import unittest
import zipfile
from pathlib import Path

from experiments.gave2_v12 import RUNTIME_BUILD_ID


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "submission/GAVE2_R2V2_FFA_Residual_V12_Colab.ipynb"
ARCHIVE = ROOT / "miccai_v12.zip"


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join("".join(cell.get("source", [])) for cell in cls.notebook["cells"])

    def test_notebook_is_clean_and_all_code_parses(self):
        self.assertGreaterEqual(len(self.notebook["cells"]), 20)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))

    def test_notebook_has_v12_release_contract(self):
        self.assertIn('ARCHIVE_PATH = DRIVE_BASE / "miccai_v12.zip"', self.source)
        self.assertIn('RUN_DIR = DRIVE_BASE / "runs/gave2_v12_safe_3fold"', self.source)
        self.assertIn("torch.cuda.is_bf16_supported()", self.source)
        self.assertIn('"kornia": kornia.__version__', self.source)
        self.assertNotIn('assert "L4"', self.source)
        self.assertNotIn("hotfix", self.source.lower())
        self.assertNotIn("apply embedded", self.source.lower())
        self.assertIn('train_task("task2", epochs=60, minimum_epochs=25)', self.source)
        self.assertIn('train_task("task1", epochs=40, minimum_epochs=15)', self.source)
        self.assertIn('"--early-stopping-patience", 7', self.source)
        self.assertIn('"--corridor-radius", 2', self.source)
        self.assertIn("experiments.gave2_v12.gate", self.source)
        self.assertIn("experiments.gave2_v12.release", self.source)
        self.assertIn("RELEASE_TARGET = 7.95", self.source)
        self.assertIn("MINIMA_MIN_SUCCESS_FRACTION = 0.90", self.source)
        self.assertIn(f'EXPECTED_RUNTIME_BUILD_ID = "{RUNTIME_BUILD_ID}"', self.source)
        self.assertIn("Runtime ZIP does not match this notebook", self.source)
        self.assertIn('"--fold-manifest", FOLD_MANIFEST', self.source)
        self.assertIn('"--correction-mode", "prune"', self.source)
        self.assertIn('"--phase", phase', self.source)
        self.assertIn('"--failure-policy", "identity"', self.source)
        self.assertIn("Running one-case MINIMA import and inference preflight", self.source)
        self.assertIn('"--failure-policy", "error"', self.source)
        self.assertIn('"--limit-cases", "1"', self.source)
        self.assertIn("MINIMA extraction gate failed; stop before training", self.source)
        self.assertIn("subprocess.Popen", self.source)
        self.assertIn('stdout="\\n".join(tail)', self.source)
        self.assertIn("memory test failed for a non-memory reason", self.source)
        self.assertIn('decision["status"] == "DO_NOT_SUBMIT"', self.source)
        self.assertNotIn('"submit_first"', self.source)
        self.assertIn("stdout=subprocess.PIPE", self.source)
        self.assertIn("test_result.check_returncode()", self.source)
        self.assertIn("runtime.unassign()", self.source)

    def test_notebook_builds_one_root_layout_candidate_and_source_release(self):
        self.assertIn("v12_safe", self.source)
        self.assertNotIn("v12_task2_probe", self.source)
        self.assertNotIn("v12_full", self.source)
        self.assertIn("GAVE2_V12_source_code.zip", self.source)
        self.assertIn('TEAM_ID = "梯度不下降队"', self.source)
        self.assertIn('"--force"', self.source)

    def test_runtime_pins_the_audited_kornia_release(self):
        requirements = (ROOT / "experiments/gave2_v12/requirements.txt").read_text(encoding="utf-8")
        self.assertIn("kornia==0.8.3", requirements.splitlines())

    @unittest.skipUnless(ARCHIVE.exists(), "runtime archive has not been built")
    def test_runtime_archive_manifest_covers_payload(self):
        with zipfile.ZipFile(ARCHIVE) as archive:
            self.assertIsNone(archive.testzip())
            manifest = json.loads(archive.read("archive_manifest.json"))
            self.assertEqual(manifest["runtime_build_id"], RUNTIME_BUILD_ID)
            self.assertEqual(set(manifest["members"]), set(archive.namelist()) - {"archive_manifest.json"})
            for member, expected in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256(archive.read(member)).hexdigest(), expected, member)
            self.assertTrue(any(name.startswith("GAVE2_preliminary/") for name in archive.namelist()))
            self.assertFalse(any(name.endswith((".pt", ".pth", ".ckpt")) for name in archive.namelist()))
            required_dependencies = {
                "experiments/gave2_ensemble/biomarkers.py",
                "experiments/gave2_ensemble/biomarkers_v2.py",
                "experiments/gave2_ensemble/submission.py",
            }
            self.assertTrue(required_dependencies.issubset(archive.namelist()))


if __name__ == "__main__":
    unittest.main()
