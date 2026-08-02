from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.gave2_v12.assets import _ensure_checkpoint
from experiments.gave2_v12.constants import MINIMA_LOFTR_SHA256


class MinimaAssetTests(unittest.TestCase):
    def test_verified_existing_checkpoint_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "minima_loftr.ckpt"
            with mock.patch("experiments.gave2_v12.assets.sha256_file", return_value=MINIMA_LOFTR_SHA256), mock.patch(
                "experiments.gave2_v12.assets.urllib.request.urlopen"
            ) as download:
                checkpoint.write_bytes(b"verified-checkpoint")
                self.assertEqual(_ensure_checkpoint(checkpoint), MINIMA_LOFTR_SHA256)
            download.assert_not_called()

    def test_download_with_wrong_digest_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "minima_loftr.ckpt"
            checkpoint.write_bytes(b"previous-file")
            with mock.patch(
                "experiments.gave2_v12.assets.urllib.request.urlopen",
                return_value=io.BytesIO(b"wrong-download"),
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    _ensure_checkpoint(checkpoint)
            self.assertEqual(checkpoint.read_bytes(), b"previous-file")
            self.assertFalse(checkpoint.with_suffix(".ckpt.partial").exists())


if __name__ == "__main__":
    unittest.main()
