from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed")
class ModelTests(unittest.TestCase):
    def test_zero_initialized_residual_reproduces_valid_teacher(self):
        import torch

        from experiments.gave2_v12.model import ModelConfig, build_model

        torch.manual_seed(7)
        teacher = torch.rand(1, 3, 32, 48) * 0.8 + 0.1
        teacher[:, 1] = torch.maximum(teacher[:, 1], torch.maximum(teacher[:, 0], teacher[:, 2]))
        features = torch.randn(1, 8, 32, 48)
        mask = torch.ones(1, 1, 32, 48)
        corridor = torch.zeros_like(teacher)
        model = build_model(ModelConfig(input_channels=8, base_channels=8, corridor_radius=2))
        with torch.inference_mode():
            output = model(features, teacher, mask, corridor)
        torch.testing.assert_close(output["probability"], teacher, atol=2e-6, rtol=0.0)
        self.assertEqual(tuple(output["crossing_logit"].shape), (1, 1, 32, 48))
        self.assertEqual(tuple(output["crossing_probability"].shape), (1, 1, 32, 48))

    def test_full_resolution_shape_is_stable(self):
        import torch

        from experiments.gave2_v12.model import ModelConfig, build_model

        model = build_model(ModelConfig(input_channels=13, base_channels=8, corridor_radius=2))
        features = torch.randn(1, 13, 64, 96)
        teacher = torch.rand(1, 3, 64, 96)
        teacher[:, 1] = torch.maximum(teacher[:, 1], torch.maximum(teacher[:, 0], teacher[:, 2]))
        mask = torch.ones(1, 1, 64, 96)
        output = model(features, teacher, mask)
        self.assertEqual(tuple(output["probability"].shape), (1, 3, 64, 96))

    def test_task1_positive_support_cannot_expand(self):
        import torch

        from experiments.gave2_v12.model import ModelConfig, build_model

        teacher = torch.full((1, 3, 32, 48), 0.1)
        teacher[:, 0, 15, 5:43] = 0.9
        teacher[:, 1] = torch.maximum(teacher[:, 1], teacher[:, 0])
        model = build_model(ModelConfig(input_channels=8, base_channels=8, correction_mode="prune"))
        with torch.no_grad():
            model.residual_head.bias.fill_(4.0)
            output = model(torch.randn(1, 8, 32, 48), teacher, torch.ones(1, 1, 32, 48))
        additions = (output["probability"] >= 0.5) & (teacher < 0.5)
        self.assertFalse(bool(additions.any()))


if __name__ == "__main__":
    unittest.main()
