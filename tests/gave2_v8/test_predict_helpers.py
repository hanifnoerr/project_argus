import unittest


class PredictHelperTests(unittest.TestCase):
    def test_rectangular_tta_round_trip(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from experiments.gave2_v8.predict_r2v2 import _invert, _transform

        tensor = torch.arange(1 * 3 * 32 * 64, dtype=torch.float32).reshape(1, 3, 32, 64)
        for rotation in range(4):
            for flip in (False, True):
                restored = _invert(_transform(tensor, rotation, flip), rotation, flip)
                self.assertTrue(torch.equal(restored, tensor))


if __name__ == "__main__":
    unittest.main()

