from __future__ import annotations

import argparse
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from experiments.gave2_v12.utils import sha256_file
from experiments.gave2_v13.compact import EXPECTED_CASES, PORTAL_MAXIMUM_BYTES
from experiments.gave2_v13.release import decide


class ReleaseTests(unittest.TestCase):
    def test_release_requires_projection_and_all_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = {
                "accepted": True,
                "selected": {"score": 8.4, "score_gain": 1.1, "minimum_fold_score_gain": 1.0},
            }
            for task in ("task1", "task2"):
                (root / f"{task}.json").write_text(json.dumps(selection), encoding="utf-8")
            task3 = {
                "accepted_targets": ["vein_density"],
                "nested_audit": {
                    "targets": {
                        "vein_density": {"nested_relative_gain": 0.90},
                    }
                },
            }
            (root / "task3.json").write_text(json.dumps(task3), encoding="utf-8")
            zip_path = root / "team.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                for task, suffix in (("Task1", ".png"), ("Task2", ".png"), ("Task3", ".txt")):
                    for case_id in EXPECTED_CASES:
                        archive.writestr(f"{task}/{case_id}{suffix}", b"certified")
            compact = {
                "output_sha256": sha256_file(zip_path),
                "output_bytes": zip_path.stat().st_size,
                "maximum_bytes": PORTAL_MAXIMUM_BYTES,
                "threshold_mismatch_pixels": 0,
                "threshold_masks_equivalent": True,
                "task3_byte_identical": True,
                "layout": "tasks_at_zip_root",
                "counts": {"Task1": 50, "Task2": 50, "Task3": 50},
            }
            compact_path = root / "compact.json"
            compact_path.write_text(json.dumps(compact), encoding="utf-8")
            manifest = {
                "zip": str(zip_path),
                "zip_sha256": sha256_file(zip_path),
                "zip_bytes": zip_path.stat().st_size,
                "maximum_submission_bytes": PORTAL_MAXIMUM_BYTES,
                "layout": "tasks_at_zip_root",
                "counts": {"Task1": 50, "Task2": 50, "Task3": 50},
                "compact_certification": str(compact_path),
                "compact_certification_sha256": sha256_file(compact_path),
            }
            (root / "submission.json").write_text(json.dumps(manifest), encoding="utf-8")
            args = argparse.Namespace(
                task1_selection=root / "task1.json",
                task2_selection=root / "task2.json",
                task3_audit=root / "task3.json",
                submission_manifest=root / "submission.json",
                output=root / "release.json",
                release_target=7.7,
                minimum_local_task_score=8.0,
                segmentation_transfer_scale=1.0,
                required_task3_targets=("vein_density",),
            )
            report = decide(args)
            self.assertEqual(report["status"], "READY_FOR_ONE_CAUTIOUS_SUBMISSION")
            self.assertTrue(report["zip_valid"])
            self.assertTrue(report["compact_valid"])
            self.assertLess(report["zip_bytes"], PORTAL_MAXIMUM_BYTES)


if __name__ == "__main__":
    unittest.main()
