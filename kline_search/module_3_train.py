# module_3_train.py
"""
模块3：模型训练
功能：训练自编码器模型，学习K线窗口的向量表示
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module_2_model import KlineAutoEncoder

# ===== 路径配置 =====
WINDOWS_PATH = "index/windows.npy"
MODEL_PATH = "index/embedding_model.pt"
os.makedirs("index", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 超参数设置 =====
batch_size = 256
epochs = 50
latent_dim = 64   # embedding维度，推荐32/64/128，可调
lr = 1e-3         # 学习率

# ===== 加载窗口数据 =====
windows = np.load(WINDOWS_PATH)   # shape [N, seq_len, F]
N, seq_len, F = windows.shape
print("加载窗口数据:", windows.shape)

# ===== 数据质量检查 =====
print("\n检查数据质量...")
nan_count = np.isnan(windows).sum()
inf_count = np.isinf(windows).sum()
print(f"  NaN 数量: {nan_count}")
print(f"  Inf 数量: {inf_count}")

if nan_count > 0 or inf_count > 0:
    print("  正在清理 NaN/Inf 值...")
    # 将 NaN 和 Inf 替换为 0
    windows = np.nan_to_num(windows, nan=0.0, posinf=0.0, neginf=0.0)
    print("  清理完成")

# 检查数据范围
print(f"  数据范围: [{windows.min():.4f}, {windows.max():.4f}]")
print(f"  数据均值: {windows.mean():.4f}")
print(f"  数据标准差: {windows.std():.4f}")

# 如果数据范围过大，进行裁剪以防止梯度爆炸
if np.abs(windows).max() > 100:
    print("  警告: 数据范围过大，进行裁剪 [-100, 100]")
    windows = np.clip(windows, -100, 100)

dataset = TensorDataset(torch.tensor(windows, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# ===== 构建模型（自适应输入通道数）=====
model = KlineAutoEncoder(in_channels=F, latent_dim=latent_dim, seq_len=seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

# ===== 训练循环 =====
print("\n开始训练...")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    nan_batch_count = 0
    
    for batch_idx, (batch,) in enumerate(loader):
        batch = batch.to(device)
        
        # 检查输入是否有 NaN
        if torch.isnan(batch).any():
            nan_batch_count += 1
            continue
        
        recon, _ = model(batch)
        loss = loss_fn(recon, batch)  # 重建损失
        
        # 检查损失是否为 NaN
        if torch.isnan(loss):
            nan_batch_count += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    avg_loss = total_loss / N
    
    # 打印训练信息
    if nan_batch_count > 0:
        print(f"Epoch {epoch}/{epochs}  平均损失: {avg_loss:.6f}  (跳过 {nan_batch_count} 个 NaN batch)")
    else:
        print(f"Epoch {epoch}/{epochs}  平均损失: {avg_loss:.6f}")

# ===== 保存模型（包含state_dict和元数据）=====
save_obj = {
    'state_dict': model.state_dict(),
    'in_channels': F,
    'latent_dim': latent_dim,
    'seq_len': seq_len
}
torch.save(save_obj, MODEL_PATH)
print("模型已保存至", MODEL_PATH)
