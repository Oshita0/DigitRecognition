import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """
    Multi-scale inception block with 1x1, 3x3, and 5x5 convolution branches.
    Output channels = out_ch * 4.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1), nn.GELU())
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GELU())
        self.b5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 5, padding=2), nn.GELU())
        self.b4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1), nn.GELU())
        self.bn = nn.BatchNorm2d(out_ch * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.b4(x)], dim=1)
        return F.relu(self.bn(x))


class DigitNet(nn.Module):
    """
    CNN classifier for spoken digit recognition.

    Architecture:
        Stem  : 2× (Conv → BN → GELU → MaxPool)
        Trunk : 4× InceptionBlock with intermediate MaxPool
        Head  : Global Average Pool → 3-layer MLP → 10-class logits
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2),
        )
        self.inc1     = InceptionBlock(128, 64)   # → 256 ch
        self.inc2     = InceptionBlock(256, 64)   # → 256 ch
        self.mid_pool = nn.MaxPool2d(2)
        self.inc3     = InceptionBlock(256, 64)   # → 256 ch
        self.inc4     = InceptionBlock(256, 64)   # → 256 ch
        self.fc = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.mid_pool(x)
        x = self.inc3(x)
        x = self.inc4(x)
        x = x.mean(dim=[2, 3])   # Global Average Pooling
        return self.fc(x)
