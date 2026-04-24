"""
MSDR-Net: Multi-Scale Dilated Residual Network with Efficient Channel Attention
Official PyTorch implementation.
"""

import math
import torch
import torch.nn as nn


class MBRB(nn.Module):
    """
    Multi-Branch Dilated Residual Block (MBRB).
    As described in Section 2.4.1 of the manuscript.
    """
    def __init__(self, in_channels, out_channels):
        super(MBRB, self).__init__()
        # Channel alignment via 1x1 convolution if dimensions differ
        self.align = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

        # Pathway 1: Fine Details (D=1, receptive field 3x3)
        self.path1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Pathway 2: Local Textures (D=2, receptive field 7x7)
        self.path2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Pathway 3: Global Semantics (D=3, receptive field 13x13)
        self.path3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 1x1 convolution for dimensionality reduction back to C channels
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        identity = self.align(x)
        p1 = self.path1(identity)
        p2 = self.path2(identity)
        p3 = self.path3(identity)
        out = torch.cat([p1, p2, p3], dim=1)  # Concatenate along channel dimension
        out = self.fuse(out)
        out = out + identity  # Residual connection
        return out


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA).
    As described in Section 2.4.2 of the manuscript.
    """
    def __init__(self, channel):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Adaptive kernel size: k = floor(log2(C)) + 1
        k = int(math.log2(channel)) + 1
        # Ensure odd kernel size for symmetric padding
        k = k if k % 2 == 1 else k + 1
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate channel descriptor
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).permute(0, 2, 1)  # (B, 1, C)
        # Cross-channel interaction
        y = self.conv1d(y)  # (B, 1, C)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)  # (B, C, 1, 1)
        # Adaptive feature recalibration
        return x * y.expand_as(x)


class MSDRNet(nn.Module):
    """
    Multi-Scale Dilated Residual Network (MSDR-Net).
    Four-level encoder with MBRB+ECA modules.
    As described in Section 2.4.3 and Table 1 of the manuscript.
    """
    def __init__(self, num_classes=2, blocks_per_stage=(2, 2, 2, 2)):
        super(MSDRNet, self).__init__()

        # Initial convolution: 3 channels -> 64 channels
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 224x224, 64 channels
        layers1 = []
        for _ in range(blocks_per_stage[0]):
            layers1.append(MBRB(64, 64))
            layers1.append(ECA(64))
        self.stage1 = nn.Sequential(*layers1)

        # Transition to Stage 2: 112x112, 128 channels
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Stage 2: 112x112, 128 channels
        layers2 = []
        for _ in range(blocks_per_stage[1]):
            layers2.append(MBRB(128, 128))
            layers2.append(ECA(128))
        self.stage2 = nn.Sequential(*layers2)

        # Transition to Stage 3: 56x56, 256 channels
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Stage 3: 56x56, 256 channels
        layers3 = []
        for _ in range(blocks_per_stage[2]):
            layers3.append(MBRB(256, 256))
            layers3.append(ECA(256))
        self.stage3 = nn.Sequential(*layers3)

        # Transition to Stage 4: 28x28, 512 channels
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Stage 4: 28x28, 512 channels
        layers4 = []
        for _ in range(blocks_per_stage[3]):
            layers4.append(MBRB(512, 512))
            layers4.append(ECA(512))
        self.stage4 = nn.Sequential(*layers4)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # MLP Classification Head
        # FC1: 512 -> 256, ReLU, BatchNorm, Dropout(p=0.5)
        # FC2: 256 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
            # Softmax is applied externally during loss computation or inference
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_param_count(self):
        """Return the number of trainable parameters in millions."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


if __name__ == "__main__":
    # Quick sanity check
    model = MSDRNet(num_classes=2, blocks_per_stage=(2, 2, 2, 2))
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total trainable parameters: {model.get_param_count():.2f}M")
    # Expected: ~23.6M parameters (adjust blocks_per_stage to match exact paper configuration)
