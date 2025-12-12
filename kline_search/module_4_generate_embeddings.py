# module_4_generate_embeddings.py
"""
模块4：生成Embeddings并存入向量数据库
功能：使用训练好的编码器为每个窗口生成embedding，存入LanceDB
"""

import sys
import numpy as np
import json
import torch
import lancedb
import pyarrow as pa
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module_2_model import KlineEncoder

# ===== 路径配置 =====
WINDOWS_PATH = "index/windows.npy"
META_PATH = "index/windows_meta.json"
MODEL_PATH = "index/embedding_model.pt"
OUT_MAPPING = "index/code_row_mapping.json"  # 单表模式时保存的映射文件
LANCEDB_PATH = "E:/data/kline_lance"         # LanceDB存储目录

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 加载数据 =====
windows = np.load(WINDOWS_PATH)   # (N, seq_len, F)
meta = json.load(open(META_PATH, 'r', encoding='utf-8'))
N, seq_len, F = windows.shape
print(f"加载窗口数据: {N} 个窗口, 序列长度: {seq_len}, 特征数: {F}")

# ===== 加载模型（module_3保存的格式）=====
checkpoint = torch.load(MODEL_PATH, map_location=device)
in_channels = checkpoint.get('in_channels', F)
latent_dim = checkpoint.get('latent_dim', 32)
seq_len_m = checkpoint.get('seq_len', seq_len)
print(f"模型参数: in_channels={in_channels}, latent_dim={latent_dim}, seq_len={seq_len_m}")

# ===== 构建编码器并加载权重 =====
encoder = KlineEncoder(in_channels=in_channels, latent_dim=latent_dim)
# state_dict保存在'state_dict'键下
encoder_state = {}
for k, v in checkpoint['state_dict'].items():
    # encoder的权重键以'encoder.'开头，需要去掉前缀
    if k.startswith('encoder.'):
        encoder_state[k[len('encoder.'):]] = v
# 加载到encoder（允许部分匹配）
encoder.load_state_dict(encoder_state, strict=False)
encoder = encoder.to(device)
encoder.eval()
print("编码器加载完成")

# ===== 连接LanceDB =====
os.makedirs(LANCEDB_PATH, exist_ok=True)
db = lancedb.connect(LANCEDB_PATH)

# ===== 决定分表策略 =====
codes = [m['code'] for m in meta]
unique_codes = sorted(set(codes))
print("股票数量:", len(unique_codes))

# 阈值：股票数量不多时，为每只股票创建独立表（便于按股票索引和检索）
SINGLE_TABLE_THRESHOLD = 200
BATCH = 512


def generate_embeddings_batch(batch_windows, batch_meta, encoder, device):
    """批量生成 embedding 和记录"""
    batch_t = torch.tensor(batch_windows, dtype=torch.float32).to(device)
    with torch.no_grad():
        emb = encoder(batch_t).cpu().numpy()
    
    records = []
    for j, e in enumerate(emb):
        m = batch_meta[j]
        records.append({
            'code': m['code'],
            'start_date': m['start_date'],
            'end_date': m['end_date'],
            'vector': e.tolist(),
            'raw': batch_windows[j].flatten().tolist()  # 展平存储
        })
    return records


if len(unique_codes) <= SINGLE_TABLE_THRESHOLD:
    # ===== 多表模式：每只股票一个表 =====
    print("创建多表模式（股票数量较少）")
    
    # 先生成所有 embeddings，按股票代码分组
    all_records_by_code = {code: [] for code in unique_codes}
    
    for i in range(0, N, BATCH):
        batch = windows[i:i + BATCH]
        batch_meta = meta[i:i + BATCH]
        records = generate_embeddings_batch(batch, batch_meta, encoder, device)
        
        for rec in records:
            all_records_by_code[rec['code']].append(rec)
        
        if (i // BATCH + 1) % 100 == 0:
            print(f"  已处理: {min(i + BATCH, N)}/{N}")
    
    # 为每只股票创建表并插入数据
    for idx, code in enumerate(unique_codes):
        tbl_name = f'kline_embeddings_{code}'
        recs = all_records_by_code[code]
        
        if len(recs) > 0:
            # 使用数据直接创建表，LanceDB 会自动推断 schema
            db.create_table(tbl_name, data=recs, mode='overwrite')
        
        if (idx + 1) % 500 == 0:
            print(f"  已创建表: {idx + 1}/{len(unique_codes)}")

    print("所有股票表已插入完成。")
    # 保存表名列表索引文件
    json.dump({'tables': [f'kline_embeddings_{c}' for c in unique_codes]}, 
              open(OUT_MAPPING, 'w'), indent=2)

else:
    # ===== 单表模式：所有数据放在一个表中（推荐用于大规模数据）=====
    print("创建单表模式（股票数量较多）")
    
    # 先生成第一批数据来创建表
    first_batch = windows[:BATCH]
    first_meta = meta[:BATCH]
    first_records = generate_embeddings_batch(first_batch, first_meta, encoder, device)
    
    # 创建表（使用第一批数据，LanceDB 自动推断 schema）
    tbl = db.create_table('kline_embeddings', data=first_records, mode='overwrite')
    print(f"  表已创建，正在插入数据...")

    # 保存股票代码到行号的映射，便于快速定位
    code_mapping = {}
    current_row = 0
    
    # 更新第一批的映射
    for m in first_meta:
        code = m['code']
        if code not in code_mapping:
            code_mapping[code] = {'start_row': current_row, 'count': 0}
        code_mapping[code]['count'] += 1
        current_row += 1

    # 处理剩余批次
    for i in range(BATCH, N, BATCH):
        batch = windows[i:i + BATCH]
        batch_meta = meta[i:i + BATCH]
        records = generate_embeddings_batch(batch, batch_meta, encoder, device)
        
        tbl.add(records)

        # 更新映射
        for m in batch_meta:
            code = m['code']
            if code not in code_mapping:
                code_mapping[code] = {'start_row': current_row, 'count': 0}
            code_mapping[code]['count'] += 1
            current_row += 1
        
        # 打印进度
        if (i // BATCH + 1) % 100 == 0:
            print(f"  已插入: {min(i + BATCH, N)}/{N} ({(i + BATCH) * 100 / N:.1f}%)")

    # 保存映射文件以便快速查找
    json.dump(code_mapping, open(OUT_MAPPING, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"单表已插入完成，共 {current_row} 条记录")
    print(f"映射文件已保存至 {OUT_MAPPING}")

print("完成！")
