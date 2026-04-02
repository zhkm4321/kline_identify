基于深度学习的 K 线形态相似性搜索工具，可在全市场历史数据中快速找到相似的 K 线走势形态。
![8e4689fcd7de48f39cd559008375208d](https://github.com/user-attachments/assets/a6475604-b4fe-4061-b06b-e44d240de3bb)

## 📋 功能概述

- **形态编码**：使用 CNN 自编码器将 K 线形态压缩为固定维度的向量表示
- **相似搜索**：基于向量数据库（LanceDB）实现毫秒级相似形态检索
- **跨股票搜索**：支持在全市场所有股票的历史数据中搜索相似形态
- **可视化对比**：搜索结果支持图表可视化，直观对比形态差异

## 🏗️ 系统架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Module 1       │     │  Module 3       │     │  Module 4       │
│  数据加载       │────▶│  模型训练       │────▶│  生成Embedding  │
│  特征工程       │     │  学习形态表示   │     │  存入向量库     │
│  滑动窗口       │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ windows.npy     │     │ embedding_      │     │ LanceDB         │
│ window_meta.csv │     │ model.pt        │     │ 向量数据库      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  Module 5       │
                                                │  相似性搜索     │
                                                │  可视化对比     │
                                                └─────────────────┘
```

## 📁 文件结构

```
kline_search/
├── module_1_load_windows.py    # 数据加载与滑动窗口生成
├── module_2_model.py           # 模型定义（编码器 + 自编码器）
├── module_3_train.py           # 模型训练
├── module_4_generate_embeddings.py  # 生成 embedding 并存入 LanceDB
├── module_5_search_similar.py  # 相似性搜索与可视化
├── index/                      # 生成的数据目录
│   ├── windows.npy             # 滑动窗口数据
│   ├── window_meta.csv         # 窗口元数据
│   ├── embedding_model.pt      # 训练好的模型
│   └── kline_lancedb/          # LanceDB 向量数据库
└── readme.md                   # 本文档
```

## 🔧 模块说明

### Module 1：数据加载与特征工程

**文件**：`module_1_load_windows.py`

**功能**：
- 加载原始 K 线数据
- 计算技术指标特征（共 21 个特征）
- 生成固定长度的滑动窗口

**特征列表（共21个）**：
| 索引 | 类别 | 特征 | 说明 |
|------|------|------|------|
| 0-4 | 收益率 | `return_open`, `return_close`, `return_high`, `return_low`, `logret_close` | 价格收益率 & 对数收益率 |
| 5 | 成交量 | `volume_chg` | 成交量变化率（裁剪到[-5,10]） |
| 6-10 | Z-score | `open_z`, `close_z`, `high_z`, `low_z`, `volume_z` | Z-score 标准化 |
| 11-13 | MACD | `macd_line`, `macd_signal`, `macd_hist` | 归一化 MACD（除以收盘价） |
| 14 | RSI | `rsi_14` | 14日 RSI（0-1归一化） |
| 15-16 | 布林带 | `boll_width`, `boll_dist` | 布林带宽度 & 到中轨距离 |
| 17-19 | 均线斜率 | `ma5_slope`, `ma10_slope`, `ma20_slope` | 均线变化率 |
| 20 | 振幅 | `hl_range` | (最高价-最低价)/收盘价 |

**输出文件**：
- `index/windows.npy`：形状 `(N, window_size, num_features)`，即 `(N, 30, 21)`
- `index/windows_meta.json`：窗口元数据（股票代码、起止日期等）

---

### Module 2：模型定义

**文件**：`module_2_model.py`

**模型架构**：

```
KlineAutoEncoder
├── KlineEncoder（编码器）
│   ├── Conv1D(in_channels → 64)
│   ├── Conv1D(64 → 128)
│   ├── Conv1D(128 → 128)
│   ├── AdaptiveAvgPool1d(1)
│   └── Linear(128 → latent_dim)
│
└── Decoder（解码器）
    ├── Linear(latent_dim → 128)
    └── Linear(128 → seq_len × in_channels)
```

**参数**：
- `in_channels`：输入特征数量（默认 21）
- `latent_dim`：embedding 维度（默认 64）
- `seq_len`：序列长度/窗口大小（默认 30）

---

### Module 3：模型训练

**文件**：`module_3_train.py`

**训练方式**：自监督学习（重建任务）

```
原始窗口 ──▶ 编码器 ──▶ 64维向量 ──▶ 解码器 ──▶ 重建窗口
     │                                            │
     └─────────────── MSE Loss ───────────────────┘
```

**超参数**：
- `batch_size`：256
- `epochs`：50
- `latent_dim`：64
- `learning_rate`：1e-3

**输出文件**：`index/embedding_model.pt`

---

### Module 4：生成 Embedding

**文件**：`module_4_generate_embeddings.py`

**功能**：
1. 加载训练好的模型（只使用编码器部分）
2. 对所有滑动窗口生成 embedding 向量
3. 将向量和元数据存入 LanceDB

**存储字段**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `vector` | float[64] | embedding 向量 |
| `code` | string | 股票代码 |
| `start_date` | string | 窗口起始日期 |
| `end_date` | string | 窗口结束日期 |
| `raw` | float[630] | 原始特征数据（展平存储，30×21=630） |

---

### Module 5：相似性搜索

**文件**：`module_5_search_similar.py`

**功能**：
- 根据窗口 ID 或股票代码+日期查询相似形态
- 支持限定同一股票内搜索
- 可视化搜索结果

## 🚀 使用方法

### 1. 准备工作

确保已安装依赖：

```bash
pip install torch numpy pandas lancedb matplotlib
```

### 2. 完整流程

```bash
# Step 1: 生成滑动窗口数据
python module_1_load_windows.py

# Step 2: 训练模型
python module_3_train.py

# Step 3: 生成 embedding 并存入 LanceDB
python module_4_generate_embeddings.py

# Step 4: 相似性搜索
python module_5_search_similar.py -q <window_id> -k 5
```

### 3. 搜索脚本参数详解

**文件**：`module_5_search_similar.py`

#### 命令行参数

| 参数 | 短参数 | 类型 | 必填 | 默认值 | 说明 |
|------|--------|------|------|--------|------|
| `--query` | `-q` | int | 二选一* | - | 查询窗口ID，对应 `windows.npy` 中的索引值 |
| `--code` | `-c` | str | 二选一* | - | 股票代码，如 `000001`、`600519` |
| `--date` | `-d` | str | 与 `-c` 配合 | - | 查询日期，格式 `YYYY-MM-DD`，如 `2024-01-15` |
| `--k` | `-k` | int | 否 | 5 | 返回的相似结果数量（Top-K） |
| `--feat` | - | str | 否 | `close_z` | 可视化时使用的特征名称 |
| `--same-stock` | - | flag | 否 | False | 只在同一股票的历史数据中搜索 |

> **注意**：`-q` 和 `-c`/`-d` 二选一，必须指定其中一种查询方式。如果同时指定，优先使用 `-q`（窗口ID）。

#### 查询方式说明

**方式1：通过窗口ID查询（`-q`）**

直接使用 `windows.npy` 中的索引作为查询条件。适用于：
- 已知具体窗口ID的情况
- 程序化批量搜索
- 调试和测试

```bash
# 查询第1000个窗口的相似形态
python module_5_search_similar.py -q 1000 -k 5
```

**方式2：通过股票代码+日期查询（`-c` + `-d`）**

更直观的查询方式，系统会自动匹配最接近的窗口。适用于：
- 查看某股票某日附近的形态
- 不知道窗口ID时使用

```bash
# 查询 000001 在 2024-01-15 附近的形态
python module_5_search_similar.py -c 000001 -d 2024-01-15 -k 10
```

日期匹配规则（优先级从高到低）：
1. 精确匹配窗口的 `end_date`
2. 精确匹配窗口的 `start_date`
3. 匹配该股票中 ≤ 指定日期的最近窗口

#### `--feat` 可用特征列表

可视化时可以选择绘制不同的特征，以下是所有可用选项：

| 特征名 | 索引 | 说明 | 推荐场景 |
|--------|------|------|----------|
| `close_z` | 7 | 收盘价 Z-score | **默认，推荐用于形态对比** |
| `open_z` | 6 | 开盘价 Z-score | 关注开盘形态 |
| `high_z` | 8 | 最高价 Z-score | 关注压力位形态 |
| `low_z` | 9 | 最低价 Z-score | 关注支撑位形态 |
| `return_close` | 1 | 收盘价收益率 | 关注涨跌幅 |
| `logret_close` | 4 | 对数收益率 | 专业量化分析 |
| `volume_z` | 10 | 成交量 Z-score | 量价配合分析 |
| `volume_chg` | 5 | 成交量变化率 | 放量/缩量分析 |
| `rsi_14` | 14 | 14日 RSI | 超买超卖分析 |
| `macd_line` | 11 | MACD 线 | MACD 形态 |
| `macd_signal` | 12 | MACD 信号线 | MACD 金叉死叉 |
| `macd_hist` | 13 | MACD 柱状图 | MACD 背离分析 |
| `boll_width` | 15 | 布林带宽度 | 波动率分析 |
| `boll_dist` | 16 | 到布林带中轨距离 | 偏离度分析 |
| `ma5_slope` | 17 | 5日均线斜率 | 短期趋势 |
| `ma10_slope` | 18 | 10日均线斜率 | 中短期趋势 |
| `ma20_slope` | 19 | 20日均线斜率 | 中期趋势 |
| `hl_range` | 20 | 振幅 | 波动幅度分析 |
| `return_open` | 0 | 开盘价收益率 | - |
| `return_high` | 2 | 最高价收益率 | - |
| `return_low` | 3 | 最低价收益率 | - |

#### `--same-stock` 参数说明

| 模式 | 搜索范围 | 适用场景 |
|------|----------|----------|
| 不加该参数 | 全市场所有股票 | 发现跨股票的相似形态 |
| 加 `--same-stock` | 仅同一股票历史 | 分析个股历史走势规律 |

#### 搜索示例

```bash
# 示例1：基础搜索 - 通过窗口ID搜索最相似的5个形态
python module_5_search_similar.py -q 1000 -k 5

# 示例2：股票+日期搜索 - 查询000001在2024-01-15的相似形态
python module_5_search_similar.py -c 000001 -d 2024-01-15 -k 10

# 示例3：同股票内搜索 - 只在000001的历史中找相似走势
python module_5_search_similar.py -c 000001 -d 2024-01-15 -k 5 --same-stock

# 示例4：指定可视化特征 - 使用RSI特征绘图
python module_5_search_similar.py -q 100 -k 10 --feat rsi_14

# 示例5：多参数组合 - 同股票搜索+MACD特征可视化
python module_5_search_similar.py -c 600519 -d 2024-06-01 -k 8 --same-stock --feat macd_hist

# 示例6：量价分析 - 使用成交量特征
python module_5_search_similar.py -q 500 -k 5 --feat volume_z
```

#### 输出说明

搜索完成后，脚本会输出：

1. **控制台信息**：
   - 模型加载状态
   - 查询窗口信息（股票代码、起止日期）
   - 搜索结果表格（包含股票代码、日期、相似距离）

2. **可视化图表**：
   - 红色粗线：查询窗口的形态
   - 其他颜色：相似窗口的形态
   - 图例显示距离值（越小越相似）

## 📊 应用场景

### 1. 形态识别与预判

```
场景：某股票最近30天走出一个形态，想看历史上类似形态的后续走势
使用：搜索相似形态，观察这些形态之后的价格变化
```

### 2. 跨股票形态发现

```
场景：股票A正在走的形态，其他股票有没有走过？
使用：全市场搜索，找到其他股票的相似历史形态
```

### 3. 量化策略验证

```
场景：验证某个技术形态的历史胜率
使用：搜索所有相似形态，统计后续涨跌比例
```

## ⚙️ 参数调优建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `window_size` | 20-60 | 窗口太小捕捉不到形态，太大会引入噪音 |
| `latent_dim` | 32-128 | 维度越高表达能力越强，但可能过拟合 |
| `epochs` | 30-100 | 观察损失曲线，收敛后可停止 |

## ⚠️ 注意事项

1. **不是预测系统**：本系统只找相似形态，不预测未来走势
2. **历史不代表未来**：相似形态的后续走势仅供参考
3. **需要结合基本面**：技术形态只是决策的一部分
4. **数据质量很重要**：确保原始 K 线数据完整准确

## 📈 训练效果参考

正常训练曲线示例：

```
Epoch 1/50   损失: 0.16   ← 起始
Epoch 10/50  损失: 0.07   ← 快速下降
Epoch 25/50  损失: 0.04   ← 逐渐收敛
Epoch 50/50  损失: 0.03   ← 稳定收敛
```

如果损失出现 NaN，检查：
- 数据是否包含 NaN/Inf
- 特征数值范围是否过大
- 学习率是否过高

## 🔗 依赖说明

| 依赖 | 用途 |
|------|------|
| PyTorch | 深度学习框架 |
| NumPy | 数值计算 |
| Pandas | 数据处理 |
| LanceDB | 向量数据库 |
| Matplotlib | 可视化 |
