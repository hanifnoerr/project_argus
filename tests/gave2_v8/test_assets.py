import unittest

from experiments.gave2_v8 import R2V2_SOURCE_COMMIT
from experiments.gave2_v8.assets import ASSETS


class AssetTests(unittest.TestCase):
    def test_release_assets_are_pinned(self):
        self.assertEqual(len(R2V2_SOURCE_COMMIT), 40)
        self.assertEqual(set(ASSETS), {"av.pth", "av_config.json", "bv.pth", "bv_config.json"})
        for record in ASSETS.values():
            self.assertEqual(len(record["sha256"]), 64)
            self.assertGreater(record["size"], 0)


if __name__ == "__main__":
    unittest.main()

