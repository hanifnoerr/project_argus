import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.gave2_v8.fuse import fuse_direct_probabilities
from experiments.gave2_v8.store import ProbabilityStore


class StoreAndFusionTests(unittest.TestCase):
    def test_store_roundtrip_and_provenance_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProbabilityStore(Path(temporary), namespace="test", split="validation")
            probability = np.linspace(0, 1, 3 * 7 * 11, dtype=np.float32).reshape(3, 7, 11)
            provenance = {"revision": "one", "input": "abc"}
            store.write_case("g_051", probability, provenance)
            self.assertTrue(store.is_complete("g_051", provenance))
            self.assertFalse(store.is_complete("g_051", {"revision": "two"}))
            np.testing.assert_allclose(store.read_case("g_051"), probability, atol=5e-4)

    def test_direct_fusion_keeps_av_classes_and_crossing_overlap(self):
        av = np.zeros((3, 5, 7), dtype=np.float32)
        bv = np.zeros_like(av)
        av[0, 2, :] = 0.8
        av[2, :, 3] = 0.9
        av[1] = np.maximum(av[0], av[2])
        bv[1] = np.maximum(av[1], 0.6)
        fused = fuse_direct_probabilities(av, bv)
        self.assertGreaterEqual(float(fused[0, 2, 3]), 0.5)
        self.assertGreaterEqual(float(fused[2, 2, 3]), 0.5)
        np.testing.assert_allclose(fused[0], av[0])
        np.testing.assert_allclose(fused[2], av[2])
        self.assertTrue(np.all(fused[1] >= np.maximum(fused[0], fused[2])))


if __name__ == "__main__":
    unittest.main()

