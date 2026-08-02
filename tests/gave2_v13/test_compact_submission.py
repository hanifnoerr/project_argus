from __future__ import annotations

import unittest

import numpy as np

from experiments.gave2_v13.compact import DEFAULT_BIT_CANDIDATES, quantize_threshold_safe


class CompactSubmissionTests(unittest.TestCase):
    def test_every_release_precision_preserves_threshold_exactly(self):
        source = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
        for bits in DEFAULT_BIT_CANDIDATES:
            with self.subTest(bits=bits):
                compact = quantize_threshold_safe(source, bits)
                np.testing.assert_array_equal(source >= 128, compact >= 128)
                self.assertLessEqual(
                    int(np.abs(source.astype(int) - compact.astype(int)).max()),
                    (1 << (8 - bits)) - 1,
                )


if __name__ == "__main__":
    unittest.main()
