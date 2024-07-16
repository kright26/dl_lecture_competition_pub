# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class WaveNetClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, hidden_channels: int = 128, dropout_prob: float = 0.5):
        super(WaveNetClassifier, self).__init__()

        self.input_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, dilation=2**i, dropout_prob=dropout_prob) for i in range(10)
        ])
        self.attention = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=8)

        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        for block in self.residual_blocks:
            x = block(x)
        x = x.permute(2, 0, 1)  # (Batch, Channels, Seq) -> (Seq, Batch, Channels)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0)  # (Seq, Batch, Channels) -> (Batch, Channels, Seq)
        x = self.output_conv(x)
        x = x.mean(dim=-1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout_prob: float):
        super(ResidualBlock, self).__init__()

        kernel_size = 2
        padding = (kernel_size - 1) * dilation  # パディングを計算

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)

        # 残差接続の前にサイズを確認してトリミング
        if x.size(-1) != residual.size(-1):
            min_size = min(x.size(-1), residual.size(-1))
            x = x[:, :, :min_size]
            residual = residual[:, :, :min_size]

        return self.relu(x + residual)