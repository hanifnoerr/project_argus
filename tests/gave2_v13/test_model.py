from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed")
class ModelTests(unittest.TestCase):
    def test_zero_delta_nearly_reproduces_valid_teacher(self):
        import torch

        from experiments.gave2_v13.model import ModelConfig, build_model

        torch.manual_seed(13)
        teacher = torch.rand(1, 3, 32, 48) * 0.75 + 0.1
        teacher[:, 1] = torch.maximum(teacher[:, 1], torch.maximum(teacher[:, 0], teacher[:, 2]))
        model = build_model(ModelConfig(input_channels=8, base_channels=4))
        output = model(torch.randn(1, 8, 32, 48), teacher, torch.ones(1, 1, 32, 48))
        torch.testing.assert_close(output["probability"], teacher, atol=5e-5, rtol=0.0)
        self.assertEqual(tuple(output["state_probability"].shape), (1, 5, 32, 48))

    def test_five_state_channels_allow_only_explicit_crossings(self):
        import torch

        from experiments.gave2_v13.model import states_to_channels

        states = torch.zeros(1, 5, 8, 8)
        states[:, 0] = 1.0
        states[:, 0, 3, 3] = 0.0
        states[:, 3, 3, 3] = 1.0
        channels = states_to_channels(states)
        self.assertEqual(float(channels[0, 0, 3, 3]), 1.0)
        self.assertEqual(float(channels[0, 2, 3, 3]), 1.0)
        self.assertEqual(float(channels[0, 1, 3, 3]), 1.0)


if __name__ == "__main__":
    unittest.main()
