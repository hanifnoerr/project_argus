from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class LossSourceTests(unittest.TestCase):
    def test_crossing_loss_uses_autocast_safe_logits_api(self):
        source = (ROOT / "experiments/gave2_v12/losses.py").read_text(encoding="utf-8")
        self.assertIn("F.binary_cross_entropy_with_logits", source)
        self.assertNotIn("F.binary_cross_entropy(", source)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed")
class LossRuntimeTests(unittest.TestCase):
    def test_cuda_bf16_forward_and_backward_are_finite(self):
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 is unavailable")
        from experiments.gave2_v12.losses import ResidualChallengeLoss

        device = "cuda"
        raw = torch.randn(1, 3, 32, 48, device=device, requires_grad=True)
        crossing_logit = torch.randn(1, 1, 32, 48, device=device, requires_grad=True)
        target = (torch.rand(1, 3, 32, 48, device=device) > 0.8).float()
        teacher = torch.rand(1, 3, 32, 48, device=device)
        mask = torch.ones(1, 1, 32, 48, device=device)
        output = {
            "probability": torch.sigmoid(raw),
            "bounded_delta": raw,
            "crossing_logit": crossing_logit,
            "crossing_probability": torch.sigmoid(crossing_logit),
        }
        criterion = ResidualChallengeLoss(torch.tensor((20.0, 12.0, 20.0), device=device))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, components = criterion(output, target, teacher, mask)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(bool(torch.isfinite(value)) for value in components.values()))
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertTrue(bool(torch.isfinite(crossing_logit.grad).all()))


if __name__ == "__main__":
    unittest.main()
