from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from experiments.gave2_v12.minima_adapter import _enable_pinned_minima_kornia_compatibility, run


class MinimaAdapterTests(unittest.TestCase):
    def test_current_kornia_public_grid_api_is_exposed_at_legacy_path(self):
        utils = ModuleType("kornia.utils")

        def create_meshgrid(*_args, **_kwargs):
            return "grid"

        utils.create_meshgrid = create_meshgrid
        missing = ModuleNotFoundError("No module named 'kornia.utils.grid'", name="kornia.utils.grid")
        previous = sys.modules.pop("kornia.utils.grid", None)
        try:
            with mock.patch(
                "experiments.gave2_v12.minima_adapter.importlib.import_module",
                side_effect=(missing, utils),
            ):
                _enable_pinned_minima_kornia_compatibility()
            compatibility = sys.modules["kornia.utils.grid"]
            self.assertIs(compatibility.create_meshgrid, create_meshgrid)
            self.assertIs(utils.grid, compatibility)
        finally:
            sys.modules.pop("kornia.utils.grid", None)
            if previous is not None:
                sys.modules["kornia.utils.grid"] = previous

    def test_official_load_model_callable_contract_is_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ("training/images", "training/FFA_A"):
                (root / "data" / folder).mkdir(parents=True, exist_ok=True)
            image = np.zeros((8, 12, 3), dtype=np.uint8)
            Image.fromarray(image, mode="RGB").save(root / "data/training/images/g_001.png")
            Image.fromarray(image, mode="RGB").save(root / "data/training/FFA_A/g_001.png")
            checkpoint = root / "minima_loftr.ckpt"
            checkpoint.write_bytes(b"mock-checkpoint")
            moving = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            fixed = moving + np.array([5.0, -2.0], dtype=np.float32)

            def matcher(first: str, second: str):
                self.assertIn("float", np.__dict__)
                self.assertTrue(first.endswith("FFA_A\\g_001.png") or first.endswith("FFA_A/g_001.png"))
                self.assertTrue(second.endswith("images\\g_001.png") or second.endswith("images/g_001.png"))
                return {"mkpts0": moving, "mkpts1": fixed, "mconf": np.array([0.8, 0.9])}

            args = SimpleNamespace(
                source_dir=root / "source",
                checkpoint=checkpoint,
                threshold=0.2,
                output_root=root / "matches",
                split="training",
                data_root=root / "data",
                limit_cases=1,
                phase="FFA_A",
                failure_policy="error",
            )
            with mock.patch("experiments.gave2_v12.minima_adapter.load_minima_matcher", return_value=matcher):
                report = run(args)

            self.assertEqual(report["cases"], 1)
            with np.load(root / "matches/training/FFA_A/g_001.npz", allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["moving_xy"], moving)
                np.testing.assert_array_equal(payload["fixed_xy"], fixed)
                np.testing.assert_allclose(payload["confidence"], (0.8, 0.9))

    def test_identity_policy_records_loader_failure_without_aborting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ("training/images", "training/FFA_AV"):
                (root / "data" / folder).mkdir(parents=True, exist_ok=True)
            image = np.zeros((8, 12, 3), dtype=np.uint8)
            Image.fromarray(image, mode="RGB").save(root / "data/training/images/g_001.png")
            Image.fromarray(image, mode="RGB").save(root / "data/training/FFA_AV/g_001.png")
            checkpoint = root / "minima_loftr.ckpt"
            checkpoint.write_bytes(b"mock-checkpoint")
            args = SimpleNamespace(
                source_dir=root / "source",
                checkpoint=checkpoint,
                threshold=0.2,
                output_root=root / "matches",
                split="training",
                data_root=root / "data",
                limit_cases=1,
                phase="FFA_AV",
                failure_policy="identity",
            )
            with mock.patch(
                "experiments.gave2_v12.minima_adapter.load_minima_matcher",
                side_effect=ImportError("mock external incompatibility"),
            ):
                report = run(args)

            self.assertEqual(report["successful_cases"], 0)
            self.assertEqual(report["failed_cases"], 1)
            self.assertIn("mock external incompatibility", report["load_error"])
            self.assertEqual(report["cases_report"], {})
            self.assertFalse((root / "matches/training/FFA_AV/g_001.npz").exists())


if __name__ == "__main__":
    unittest.main()
