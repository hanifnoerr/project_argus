from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.gave2_v12.release import decide_release


def gate(accepted: bool, gain: float, fold_gain: float) -> dict[str, object]:
    return {
        "accepted": accepted,
        "pixel_score_gain": gain,
        "minimum_fold_pixel_score_gain": fold_gain,
    }


class ReleaseTests(unittest.TestCase):
    def test_rejects_candidate_that_cannot_clear_release_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = decide_release(
                gate(True, 0.20, 0.10),
                gate(True, 0.30, 0.20),
                output_root=Path(temporary),
                team_id="team",
            )
        self.assertEqual(report["status"], "DO_NOT_SUBMIT")
        self.assertIsNone(report["recommended"])

    def test_releases_only_the_safe_candidate_when_it_clears_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            safe = root / "v12_safe/team.zip"
            safe.parent.mkdir(parents=True)
            safe.write_bytes(b"certified-placeholder")
            report = decide_release(
                gate(False, 0.0, 0.0),
                gate(True, 1.30, 1.20),
                output_root=root,
                team_id="team",
            )
        self.assertEqual(report["status"], "READY_FOR_ONE_CAUTIOUS_SUBMISSION")
        self.assertEqual(report["recommended"]["variant"], "v12_safe")


if __name__ == "__main__":
    unittest.main()
