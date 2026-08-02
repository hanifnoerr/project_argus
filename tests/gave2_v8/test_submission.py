import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.gave2_v8.store import ProbabilityStore
from experiments.gave2_v8.submission import build_team_directory, certify_root_task_submission_atomic


ROOT = Path(__file__).resolve().parents[2]
PROVEN_TASK3 = ROOT / "experiments/gave2_v8/assets/proven_task3"


class SubmissionTests(unittest.TestCase):
    def test_proven_task3_asset_is_complete(self):
        files = sorted(PROVEN_TASK3.glob("*.txt"))
        self.assertEqual(len(files), 50)
        self.assertEqual(files[0].name, "g_051.txt")
        self.assertEqual(files[-1].name, "g_100.txt")

    def test_build_team_directory_uses_native_store_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            images = data_root / "validation/images"
            masks = data_root / "validation/masks"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            task1 = ProbabilityStore(root / "task1", namespace="r2v2_direct", split="validation")
            task2 = ProbabilityStore(root / "task2", namespace="r2v2_direct", split="validation")
            yy, xx = np.mgrid[:8, :12]
            probability = np.stack(
                (
                    0.1 + 0.7 * xx / 11.0,
                    0.2 + 0.7 * np.maximum(xx / 11.0, yy / 7.0),
                    0.1 + 0.7 * yy / 7.0,
                ),
                axis=0,
            ).astype(np.float32)
            for index in range(51, 101):
                case_id = f"g_{index:03d}"
                Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8), mode="RGB").save(images / f"{case_id}.png")
                Image.fromarray(np.full((8, 12), 255, dtype=np.uint8), mode="L").save(masks / f"{case_id}.png")
                task1.write_case(case_id, probability, {"case": case_id})
                task2.write_case(case_id, probability, {"case": case_id})
            team_root = build_team_directory(
                data_root=data_root,
                task1_store=task1,
                task2_store=task2,
                task3_source=PROVEN_TASK3,
                team_root=root / "candidate/team",
            )
            self.assertEqual(len(list((team_root / "Task1").glob("*.png"))), 50)
            self.assertEqual(len(list((team_root / "Task2").glob("*.png"))), 50)
            self.assertEqual(len(list((team_root / "Task3").glob("*.txt"))), 50)
            with Image.open(team_root / "Task1/g_051.png") as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (12, 8))

            output_zip = root / "submission.zip"
            certification = certify_root_task_submission_atomic(
                team_root,
                data_root,
                output_zip,
                root / "certification.json",
                expected_size=(8, 12),
            )
            self.assertEqual(certification["readback"]["layout"], "tasks_at_zip_root")
            with zipfile.ZipFile(output_zip) as archive:
                names = archive.namelist()
                self.assertEqual(archive.testzip(), None)
            self.assertEqual({name.split("/")[0] for name in names}, {"Task1", "Task2", "Task3"})
            self.assertNotIn(team_root.name, {name.split("/")[0] for name in names})


if __name__ == "__main__":
    unittest.main()
