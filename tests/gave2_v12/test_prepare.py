from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from experiments.gave2_v12.prepare import prepare_split
from experiments.gave2_v12.registration import RegistrationQA


def accepted_qa(model: str) -> RegistrationQA:
    return RegistrationQA(True, model, 20, 20, 1.0, 0.1, 0.2, 1.0, 1.0, 1.0, 0.0, 0.1, "accepted")


class PrepareTests(unittest.TestCase):
    def test_early_and_late_frames_use_independent_registrations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data/training"
            for folder in ("images", "masks", "FFA_A", "FFA_AV"):
                (data / folder).mkdir(parents=True, exist_ok=True)
            rgb = np.zeros((8, 12, 3), dtype=np.uint8)
            gray = np.full((8, 12), 127, dtype=np.uint8)
            mask = np.full((8, 12), 255, dtype=np.uint8)
            Image.fromarray(rgb, mode="RGB").save(data / "images/g_001.png")
            Image.fromarray(mask, mode="L").save(data / "masks/g_001.png")
            Image.fromarray(gray, mode="L").save(data / "FFA_A/g_001.png")
            Image.fromarray(gray, mode="L").save(data / "FFA_AV/g_001.png")
            for phase in ("FFA_A", "FFA_AV"):
                match = root / "matches/training" / phase / "g_001.npz"
                match.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(match, moving_xy=np.zeros((20, 2)), fixed_xy=np.zeros((20, 2)))

            early_matrix = np.eye(3)
            late_matrix = np.eye(3)
            late_matrix[0, 2] = 3.0
            fits = [(early_matrix, accepted_qa("early")), (late_matrix, accepted_qa("late"))]
            args = SimpleNamespace(
                data_root=root / "data",
                matches_root=root / "matches",
                output_root=root / "prepared",
                split="training",
                fallback="error",
                seed=77,
                limit_cases=1,
            )
            with mock.patch("experiments.gave2_v12.prepare.fit_registration", side_effect=fits):
                report = prepare_split(args)

            metadata = json.loads((root / "prepared/training/metadata/g_001.json").read_text())
            self.assertEqual(metadata["registration"]["FFA_A"]["model"], "early")
            self.assertEqual(metadata["registration"]["FFA_AV"]["model"], "late")
            self.assertEqual(metadata["matrix_moving_to_fixed"]["FFA_AV"][0][2], 3.0)
            self.assertEqual(report["accepted_registrations"], {"FFA_A": 1, "FFA_AV": 1})

    def test_corrupt_match_archive_uses_recorded_identity_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data/training"
            for folder in ("images", "masks", "FFA_A", "FFA_AV"):
                (data / folder).mkdir(parents=True, exist_ok=True)
            rgb = np.zeros((8, 12, 3), dtype=np.uint8)
            gray = np.full((8, 12), 127, dtype=np.uint8)
            mask = np.full((8, 12), 255, dtype=np.uint8)
            Image.fromarray(rgb, mode="RGB").save(data / "images/g_001.png")
            Image.fromarray(mask, mode="L").save(data / "masks/g_001.png")
            Image.fromarray(gray, mode="L").save(data / "FFA_A/g_001.png")
            Image.fromarray(gray, mode="L").save(data / "FFA_AV/g_001.png")

            corrupt = root / "matches/training/FFA_A/g_001.npz"
            corrupt.parent.mkdir(parents=True, exist_ok=True)
            corrupt.write_bytes(b"truncated-npz")
            args = SimpleNamespace(
                data_root=root / "data",
                matches_root=root / "matches",
                output_root=root / "prepared",
                split="training",
                fallback="identity",
                seed=77,
                limit_cases=1,
            )

            report = prepare_split(args)

            metadata = json.loads((root / "prepared/training/metadata/g_001.json").read_text())
            early = metadata["registration"]["FFA_A"]
            late = metadata["registration"]["FFA_AV"]
            self.assertFalse(early["accepted"])
            self.assertIn("invalid MINIMA FFA_A matches", early["reason"])
            self.assertFalse(late["accepted"])
            self.assertIn("matches absent", late["reason"])
            self.assertEqual(report["identity_fallbacks"], {"FFA_A": 1, "FFA_AV": 1})
            self.assertEqual(list(np.load(root / "prepared/training/arrays/g_001.npz")["features"].shape), [5, 8, 12])


if __name__ == "__main__":
    unittest.main()
