# module_1_load_windows.py
"""
模块1：数据加载与滑动窗口生成
功能：从CSV加载股票数据，计算技术指标特征，生成滑动窗口用于后续训练
"""

import pandas as pd
import numpy as np
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foundation.kline_data import KLineDataService

CSV_PATH = r"E:\company\cursor_py_work\finance\db\all_stocks_180days.csv"
WINDOW = 30      # 滑动窗口大小（天数）
STRIDE = 1       # 滑动步长

OUT_DIR = "index"
OUT_WINDOWS = f"{OUT_DIR}/windows.npy"
OUT_META = f"{OUT_DIR}/windows_meta.json"

os.makedirs(OUT_DIR, exist_ok=True)

# ===== 技术指标计算函数 =====
def compute_macd(df, fast=12, slow=26, signal=9):
    """
    计算MACD指标
    参数:
        fast: 快线周期（默认12）
        slow: 慢线周期（默认26）
        signal: 信号线周期（默认9）
    返回:
        macd_line: MACD线（快线-慢线）
        signal_line: 信号线（MACD线的EMA）
        hist: MACD柱状图（MACD线-信号线）
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.fillna(0), signal_line.fillna(0), hist.fillna(0)

def compute_rsi(series, period=14):
    """
    计算RSI相对强弱指标
    参数:
        series: 价格序列
        period: 计算周期（默认14）
    返回:
        rsi: RSI值（0-100之间）
    """
    delta = series.diff()
    up = delta.clip(lower=0)           # 上涨幅度
    down = -1 * delta.clip(upper=0)    # 下跌幅度
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)      # 相对强度
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # 用中性值50填充缺失值

def compute_bollinger(df, period=20, n_std=2):
    """
    计算布林带指标
    参数:
        period: 移动平均周期（默认20）
        n_std: 标准差倍数（默认2）
    返回:
        upper: 上轨
        lower: 下轨
        width: 相对宽度（上轨-下轨）/中轨
        dist_middle: 收盘价到中轨的距离（正数表示在中轨上方，负数表示下方）
    """
    ma = df['close'].rolling(window=period, min_periods=1).mean()
    std = df['close'].rolling(window=period, min_periods=1).std().fillna(0)
    upper = ma + n_std * std
    lower = ma - n_std * std
    width = (upper - lower) / (ma + 1e-9)  # 相对宽度
    dist_middle = (df['close'] - ma) / (std + 1e-9)  # 到中轨的标准化距离
    return upper.fillna(df['close']), lower.fillna(df['close']), width.fillna(0), dist_middle.fillna(0)

# ===== 单只股票特征构建 =====
def build_features_for_stock(sub, chart_days: int):
    """
    为单只股票构建所有特征
    参数:
        sub: 单只股票的DataFrame，已按交易日期排序
    返回:
        sub: 添加了特征列的DataFrame
    """
    # 1) 基础收益率
    sub['return_close'] = sub['close'].pct_change().fillna(0)   # 收盘价收益率
    sub['return_open'] = sub['open'].pct_change().fillna(0)     # 开盘价收益率
    sub['return_high'] = sub['high'].pct_change().fillna(0)     # 最高价收益率
    sub['return_low'] = sub['low'].pct_change().fillna(0)       # 最低价收益率

    # 2) 对数收益率
    sub['logret_close'] = np.log(sub['close'] / sub['close'].shift(1)).replace([np.inf, -np.inf], 0).fillna(0)

    # 3) 成交量变化率（裁剪极端值，防止放量时数值过大）
    sub['volume_chg'] = sub['volume'].pct_change().fillna(0).clip(-5, 10)

    # 4) Z-score标准化（使用累计均值/标准差，避免前视偏差） 涉及属性：'open_z', 'close_z', 'high_z', 'low_z', 'volume_z'
    #    对于短历史数据使用expanding更安全
    for col in ['open', 'close', 'high', 'low', 'volume']:
        mean = sub[col].expanding(min_periods=1).mean()
        std = sub[col].expanding(min_periods=1).std().fillna(0)
        sub[f'{col}_z'] = (sub[col] - mean) / (std + 1e-9)

    # 5) MACD指标（归一化为相对值，除以收盘价）
    macd_line, macd_signal, macd_hist = compute_macd(sub)
    sub['macd_line'] = macd_line / (sub['close'] + 1e-9)
    sub['macd_signal'] = macd_signal / (sub['close'] + 1e-9)
    sub['macd_hist'] = macd_hist / (sub['close'] + 1e-9)

    # 6) RSI指标（归一化到0-1）
    sub['rsi_14'] = compute_rsi(sub['close'], period=14) / 100

    # 7) 布林带指标
    b_up, b_low, b_width, b_dist = compute_bollinger(sub)
    sub['boll_upper'] = b_up
    sub['boll_lower'] = b_low
    sub['boll_width'] = b_width
    sub['boll_dist'] = b_dist

    # 8) 振幅：(最高价-最低价)/收盘价
    sub['hl_range'] = (sub['high'] - sub['low']) / (sub['close'] + 1e-9)

    # 9) 均线及其斜率
    for period in [5, 10, 20]:
        ma = sub['close'].rolling(window=period, min_periods=1).mean()
        # 均线斜率：(当前MA - 前一日MA) / 前一日MA，表示均线的变化率
        ma_slope = (ma - ma.shift(1)) / (ma.shift(1) + 1e-9)
        sub[f'ma{period}_slope'] = ma_slope.fillna(0)

    # 填充剩余的缺失值
    sub = sub.fillna(0)
    return sub

# ===== 选择最终使用的特征 =====
FEATS = [
    # 收益率/对数收益率
    'return_open', 'return_close', 'return_high', 'return_low', 'logret_close',
    # 成交量
    'volume_chg',
    # 原始值的Z-score
    'open_z', 'close_z', 'high_z', 'low_z', 'volume_z',
    # MACD
    'macd_line', 'macd_signal', 'macd_hist',
    # RSI
    'rsi_14',
    # 布林带
    'boll_width', 'boll_dist',
    # 均线斜率
    'ma5_slope', 'ma10_slope', 'ma20_slope',
    # 其他
    'hl_range'
]

# 初始化 K 线数据服务
kline_service = KLineDataService(CSV_PATH, min_trading_days=120, use_cache=True)
print(f"数据最新日期: {kline_service.max_trade_date}")
total_codes = len(kline_service.codes)
windows = []
meta = []

print("股票总数:", total_codes)

for idx, code in enumerate(kline_service.codes, 1):
    # 每500只股票打印一次进度
    if idx % 500 == 0 or idx == 1:
        print(f"正在处理: {idx}/{total_codes} ({idx * 100 / total_codes:.1f}%)")
    
    sub = kline_service.get_stock_data(code=code, chart_days=90, signal_processor=build_features_for_stock, force_update=True)
    if len(sub) < WINDOW:
        continue  # 数据不足一个窗口，跳过

    values = sub[FEATS].values  # shape [T, F]

    # 滑动窗口切片
    for i in range(0, len(sub) - WINDOW + 1, STRIDE):
        win = values[i:i + WINDOW]
        windows.append(win.astype('float32'))
        meta.append({
            'code': str(code),
            'start_date': str(sub.loc[i, 'trade_date']),
            'end_date': str(sub.loc[i + WINDOW - 1, 'trade_date']),
            'start_idx': int(i),
        })

print(f"处理完成: {total_codes}/{total_codes}")

# ===== 转换并保存 =====
if len(windows) == 0:
    raise RuntimeError("未生成任何窗口 - 请检查数据长度或WINDOW参数")

windows = np.stack(windows)  # shape [N, WINDOW, F]
np.save(OUT_WINDOWS, windows)
json.dump(meta, open(OUT_META, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

print("窗口数据已保存:", OUT_WINDOWS)
print("元数据已保存:", OUT_META)
print("窗口总数:", len(windows))
print("单个窗口形状:", windows[0].shape)
print("特征维度:", windows.shape[2])
