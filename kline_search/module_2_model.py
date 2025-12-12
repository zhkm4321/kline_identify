# module_2_model.py
"""
模块2：模型定义
功能：定义K线编码器(KlineEncoder)和自编码器(KlineAutoEncoder)网络结构
"""

import torch
import torch.nn as nn


class KlineEncoder(nn.Module):
    """
    K线编码器：将K线序列编码为固定维度的向量表示
    使用1D卷积网络提取时序特征
    """
    def __init__(self, in_channels=5, latent_dim=32):
        """
        参数:
            in_channels: 输入特征数量（对应FEATS的长度）
            latent_dim: 输出embedding的维度
        """
        super().__init__()

        # Conv1D期望输入格式: (batch, channels, seq_len)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 自适应平均池化 -> (batch, 128, 1)
        )
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量，形状 (batch, seq_len, in_channels)
        返回:
            z: 编码向量，形状 (batch, latent_dim)
        """
        x = x.permute(0, 2, 1)     # 转换为 (batch, in_channels, seq_len)
        h = self.conv(x)           # 卷积处理 -> (batch, 128, 1)
        h = h.squeeze(-1)          # 压缩最后一维 -> (batch, 128)
        z = self.fc(h)             # 全连接层 -> (batch, latent_dim)
        return z


class KlineAutoEncoder(nn.Module):
    """
    K线自编码器：用于无监督学习K线的向量表示
    通过重建任务学习有意义的embedding
    """
    def __init__(self, in_channels=5, latent_dim=32, seq_len=30):
        """
        参数:
            in_channels: 输入特征数量
            latent_dim: 隐层embedding维度
            seq_len: 序列长度（窗口大小）
        """
        super().__init__()
        self.encoder = KlineEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len * in_channels),
        )
        self.in_channels = in_channels
        self.seq_len = seq_len

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量，形状 (batch, seq_len, in_channels)
        返回:
            recon: 重建的输出，形状 (batch, seq_len, in_channels)
            z: 编码向量，形状 (batch, latent_dim)
        """
        z = self.encoder(x)
        recon = self.decoder(z).view(-1, self.seq_len, self.in_channels)
        return recon, z
