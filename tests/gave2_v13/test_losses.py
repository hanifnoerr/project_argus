from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed")
class ChannelPathLossTests(unittest.TestCase):
    def test_all_components_are_finite_and_backward_succeeds(self):
        import torch

        from experiments.gave2_v13.losses import ChannelPathLoss

        torch.manual_seed(13)
        batch, height, width = 2, 16, 20
        probability_logits = torch.randn(batch, 3, height, width, requires_grad=True)
        state_logits = torch.randn(batch, 5, height, width, requires_grad=True)
        bounded_delta = torch.randn(batch, 5, height, width, requires_grad=True)
        artery = (torch.rand(batch, height, width) > 0.82).float()
        vein = (torch.rand(batch, height, width) > 0.80).float()
        vessel = torch.maximum(artery, vein)
        target = torch.stack((artery, vessel, vein), dim=1)
        state_target = torch.zeros(batch, height, width, dtype=torch.long)
        state_target[(artery > 0.5) & (vein <= 0.5)] = 1
        state_target[(vein > 0.5) & (artery <= 0.5)] = 2
        state_target[(artery > 0.5) & (vein > 0.5)] = 3
        centerline = target.index_select(1, torch.tensor((0, 2)))
        teacher = torch.rand(batch, 3, height, width)
        mask = torch.ones(batch, 1, height, width)
        output = {
            "probability": torch.sigmoid(probability_logits),
            "state_logits": state_logits,
            "bounded_delta": bounded_delta,
        }
        criterion = ChannelPathLoss(
            positive_weights=torch.tensor((2.0, 1.0, 3.0)),
            state_weights=torch.ones(5),
        )
        loss, components = criterion(output, target, state_target, centerline, teacher, mask)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(
            set(components),
            {"classification", "state", "dice", "topology", "centerline", "teacher", "residual"},
        )
        self.assertTrue(all(bool(torch.isfinite(value)) for value in components.values()))
        loss.backward()
        self.assertIsNotNone(probability_logits.grad)
        self.assertIsNotNone(state_logits.grad)
        self.assertIsNotNone(bounded_delta.grad)

    def test_model_output_and_loss_backward_are_compatible(self):
        import torch

        from experiments.gave2_v13.losses import ChannelPathLoss
        from experiments.gave2_v13.model import ModelConfig, build_model

        torch.manual_seed(17)
        height, width = 32, 32
        teacher = torch.rand(1, 3, height, width) * 0.8 + 0.05
        teacher[:, 1] = torch.maximum(teacher[:, 1], torch.maximum(teacher[:, 0], teacher[:, 2]))
        mask = torch.ones(1, 1, height, width)
        model = build_model(ModelConfig(input_channels=8, base_channels=4))
        output = model(torch.randn(1, 8, height, width), teacher, mask)
        artery = (torch.rand(1, height, width) > 0.82).float()
        vein = (torch.rand(1, height, width) > 0.80).float()
        vessel = torch.maximum(artery, vein)
        target = torch.stack((artery, vessel, vein), dim=1)
        state_target = torch.zeros(1, height, width, dtype=torch.long)
        state_target[(artery > 0.5) & (vein <= 0.5)] = 1
        state_target[(vein > 0.5) & (artery <= 0.5)] = 2
        state_target[(artery > 0.5) & (vein > 0.5)] = 3
        centerline = target.index_select(1, torch.tensor((0, 2)))
        criterion = ChannelPathLoss(torch.tensor((2.0, 1.0, 3.0)), torch.ones(5))
        loss, _ = criterion(output, target, state_target, centerline, teacher, mask)
        loss.backward()
        gradient = model.state_delta_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))


if __name__ == "__main__":
    unittest.main()
