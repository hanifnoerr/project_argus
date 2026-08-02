from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _in_channels_for_task(task: str) -> int:
    task = task.lower()
    if task == "task1":
        return 3
    if task == "task2":
        return 5
    raise ValueError(f"Unsupported task {task!r}")


class ConvAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvAct(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.bn(self.conv2(self.conv1(x))))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = ConvAct(in_ch, out_ch, stride=2)
        self.res = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvAct(in_ch + skip_ch, out_ch),
            ResidualBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SemanticUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.stem = nn.Sequential(ConvAct(in_channels, b), ResidualBlock(b))
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)
        self.down3 = DownBlock(b * 4, b * 8)
        self.bottleneck = nn.Sequential(ConvAct(b * 8, b * 8), ResidualBlock(b * 8))
        self.up2 = UpBlock(b * 8, b * 4, b * 4)
        self.up1 = UpBlock(b * 4, b * 2, b * 2)
        self.up0 = UpBlock(b * 2, b, b)
        self.head = nn.Conv2d(b, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        x = self.down3(s2)
        x = self.bottleneck(x)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)


class RecursiveRefinementNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        base_channels: int = 32,
        num_iterations: int = 2,
    ) -> None:
        super().__init__()
        self.first = SemanticUNet(in_channels, out_channels, base_channels)
        self.refiner = SemanticUNet(out_channels, out_channels, max(4, base_channels // 2))
        self.num_iterations = max(1, num_iterations)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        predictions = [self.first(x)]
        refined = predictions[0]
        for _ in range(self.num_iterations):
            refined = self.refiner(torch.sigmoid(refined))
            predictions.append(refined)
        return predictions


class SAM3NativeSeg(nn.Module):
    """SAM-style native-resolution semantic branch.

    This branch keeps the SAM idea of a strong image encoder plus task decoder,
    but exposes a plain semantic output for GAVE2. Official SAM3 checkpoint
    wiring should live in the separate `gave2-sam3` environment; this class is
    intentionally trainable without requiring that dependency.
    """

    def __init__(self, in_channels: int, out_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.patch = nn.Sequential(
            ConvAct(in_channels, b, kernel_size=7),
            ConvAct(b, b, kernel_size=3),
        )
        self.encoder = nn.Sequential(
            DownBlock(b, b * 2),
            DownBlock(b * 2, b * 4),
            ResidualBlock(b * 4),
            ResidualBlock(b * 4),
        )
        self.low = ConvAct(b, b, 1)
        self.mid = ConvAct(b * 2, b * 2, 1)
        self.high = ConvAct(b * 4, b * 4, 1)
        self.up1 = UpBlock(b * 4, b * 2, b * 2)
        self.up0 = UpBlock(b * 2, b, b)
        self.head = nn.Sequential(ConvAct(b, b), nn.Conv2d(b, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.patch(x)
        mid = self.encoder[0](low)
        high = self.encoder[1](mid)
        high = self.encoder[2](high)
        high = self.encoder[3](high)
        x = self.up1(self.high(high), self.mid(mid))
        x = self.up0(x, self.low(low))
        return self.head(x)


class CSPBlock(nn.Module):
    def __init__(self, channels: int, repeats: int = 2) -> None:
        super().__init__()
        hidden = max(4, channels // 2)
        self.left = ConvAct(channels, hidden, 1)
        self.right = ConvAct(channels, hidden, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(repeats)])
        self.out = ConvAct(hidden * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.cat([self.blocks(self.left(x)), self.right(x)], dim=1))


class YOLONativeSeg(nn.Module):
    """YOLO-style native semantic branch.

    This uses CSP/PAN-style feature fusion, but returns dense GAVE2 semantic
    probability logits instead of YOLO instance masks.
    """

    def __init__(self, in_channels: int, out_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.stem = ConvAct(in_channels, b)
        self.p1 = nn.Sequential(ConvAct(b, b * 2, stride=2), CSPBlock(b * 2))
        self.p2 = nn.Sequential(ConvAct(b * 2, b * 4, stride=2), CSPBlock(b * 4))
        self.p3 = nn.Sequential(ConvAct(b * 4, b * 8, stride=2), CSPBlock(b * 8))
        self.spp = nn.Sequential(
            ConvAct(b * 8, b * 8, 1),
            nn.MaxPool2d(5, stride=1, padding=2),
            ConvAct(b * 8, b * 8, 1),
        )
        self.lat2 = ConvAct(b * 8, b * 4, 1)
        self.fuse2 = CSPBlock(b * 4)
        self.lat1 = ConvAct(b * 4, b * 2, 1)
        self.fuse1 = CSPBlock(b * 2)
        self.out = nn.Sequential(ConvAct(b * 2, b), nn.Conv2d(b, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        p1 = self.p1(s0)
        p2 = self.p2(p1)
        p3 = self.spp(self.p3(p2))
        x = F.interpolate(self.lat2(p3), size=p2.shape[-2:], mode="nearest")
        x = self.fuse2(x + p2)
        x = F.interpolate(self.lat1(x), size=p1.shape[-2:], mode="nearest")
        x = self.fuse1(x + p1)
        x = F.interpolate(x, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(x)


def _load_official_cmrrwnet(task: str, in_channels: int, base_channels: int, num_iterations: int):
    if base_channels < 16:
        return None
    project_root = Path(__file__).resolve().parents[2]
    official = project_root / "knowledge_base" / "sources" / "github" / "Peng2004_CMRRWNet" / "train" / "models.py"
    if not official.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("gave2_official_cmrrwnet_models", official)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = module.RRWNet if task == "task1" else module.CMRRWNet
        return cls(
            input_ch=in_channels,
            output_ch=3,
            base_ch=base_channels,
            num_iterations=num_iterations,
        )
    except Exception:
        return None


def create_model(
    branch: str,
    task: str,
    base_channels: int = 32,
    num_iterations: int = 2,
    use_official_cmrrwnet: bool = True,
) -> nn.Module:
    branch = branch.lower()
    task = task.lower()
    in_channels = _in_channels_for_task(task)
    if branch == "cmrrwnet":
        if use_official_cmrrwnet:
            official = _load_official_cmrrwnet(task, in_channels, base_channels, num_iterations)
            if official is not None:
                return official
        return RecursiveRefinementNet(in_channels, 3, base_channels, num_iterations)
    if branch == "sam3":
        return SAM3NativeSeg(in_channels, 3, base_channels)
    if branch == "yolo_native":
        return YOLONativeSeg(in_channels, 3, base_channels)
    raise ValueError(f"Unsupported branch {branch!r}")
