import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "experiments/gave2_v8/GAVE2_R2V2_Graph_V8_Colab.ipynb"


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join("".join(cell.get("source", [])) for cell in cls.notebook["cells"])

    def test_notebook_is_clean_and_all_code_parses(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertFalse(cell["outputs"])
                ast.parse("".join(cell["source"]))

    def test_notebook_uses_clean_archive_without_hardware_brand_assertion(self):
        self.assertIn('ARCHIVE_PATH = DRIVE_BASE / "miccai_v8.zip"', self.source)
        self.assertIn("4003158544825cdcfe2b2ac8f7e2a2240055fe7ed0e4857a812fc237760e521e", self.source)
        self.assertIn("torch.cuda.is_bf16_supported()", self.source)
        self.assertNotIn('assert "L4"', self.source)
        self.assertNotIn("hotfix", self.source.lower())
        self.assertNotIn("apply embedded", self.source.lower())

    def test_direct_submission_precedes_graph_search(self):
        direct_position = self.source.index('DIRECT_SUBMISSION_ID = "GAVE2-S006"')
        direct_build_position = self.source.index('"--version", "v8-r2v2-direct"')
        graph_search_position = self.source.index('"experiments.gave2_v8.graph", "search"')
        graph_build_position = self.source.index('"--version", "v8-r2v2-graph"')
        self.assertLess(direct_position, direct_build_position)
        self.assertLess(direct_build_position, graph_search_position)
        self.assertLess(graph_search_position, graph_build_position)

    def test_notebook_has_resume_and_safety_controls(self):
        self.assertIn('RUN_DIR = DRIVE_BASE / "runs/gave2_r2v2_v8"', self.source)
        self.assertIn("RUN_GRAPH_SEARCH = True", self.source)
        self.assertIn('"--minimum-gain", 0.10', self.source)
        self.assertIn('"--maximum-dice-drop", 0.03', self.source)
        self.assertIn('"--maximum-sensitivity-drop", 0.02', self.source)
        self.assertIn("runtime.unassign()", self.source)


if __name__ == "__main__":
    unittest.main()
