"""
技术信号预计算服务
用于定时任务周期性预计算每天的技术指标并缓存

使用方式:
    # 计算今天的技术信号
    python visual/tech_signals_service.py
    
    # 计算指定日期的技术信号
    python visual/tech_signals_service.py --date 2025-12-05
    
    # 计算日期范围内的技术信号
    python visual/tech_signals_service.py --start-date 2025-12-01 --end-date 2025-12-05
    
    # 指定最小交易天数和图表天数
    python visual/tech_signals_service.py --date 2025-12-05 --min-trading-days 100 --chart-days 80
"""
import os
import sys
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

# 添加项目根目录到path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_ema(n, data, field=None):
    """
    计算EMA指数平滑移动平均线，用于MACD
    :param n: 时间窗口
    :param data: 输入数据（一维数组或包含字典的列表）
    :param field: 计算字段配置，如果data是字典列表则需要指定字段名
    :return: EMA数组
    """
    a = 2 / (n + 1)
    ema = []
    
    if field:
        # 二维数组（字典列表）
        ema.append(data[0][field])
        for i in range(1, len(data)):
            ema.append(a * data[i][field] + (1 - a) * ema[i - 1])
    else:
        # 普通一维数组
        ema.append(data[0])
        for i in range(1, len(data)):
            ema.append(a * data[i] + (1 - a) * ema[i - 1])
    
    return ema


def calculate_dif(short, long, data, field=None):
    """
    计算DIF快线，用于MACD
    :param short: 快速EMA时间窗口
    :param long: 慢速EMA时间窗口
    :param data: 输入数据
    :param field: 计算字段配置
    :return: DIF数组
    """
    ema_short = calculate_ema(short, data, field)
    ema_long = calculate_ema(long, data, field)
    
    dif = []
    for i in range(len(data)):
        dif.append(ema_short[i] - ema_long[i])
    
    return dif


def calculate_macd(short=12, long=26, mid=9, data=None, field=None):
    """
    计算MACD指标
    :param short: 快速EMA时间窗口，默认12
    :param long: 慢速EMA时间窗口，默认26
    :param mid: dea时间窗口，默认9
    :param data: 输入数据（价格数组或包含价格字段的字典列表）
    :param field: 计算字段配置，如果data是字典列表则需要指定字段名
    :return: 包含dif、dea、macd的字典
    """
    if data is None:
        raise ValueError("data 参数不能为空")
    
    dif = calculate_dif(short, long, data, field)
    dea = calculate_ema(mid, dif)
    
    macd = []
    for i in range(len(data)):
        macd.append((dif[i] - dea[i]) * 2)
    
    return {
        'dif': dif,
        'dea': dea,
        'macd': macd
    }

def compute_boll(df, window=20):
    """
    计算BOLL布林线指标
    BOLL_MID = N日收盘价的移动平均线
    BOLL_UP  = BOLL_MID + 2 * N日的标准差
    BOLL_LOW = BOLL_MID - 2 * N日的标准差
    """
    close = df['close']
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    df['BOLL_MID'] = mid
    df['BOLL_UP'] = mid + 2 * std
    df['BOLL_LOW'] = mid - 2 * std
    return df

def precompute_ma_signals(df: pd.DataFrame, window_ma5: int = 5, window_ma10: int = 10) -> pd.DataFrame:
    df = df.sort_values('trade_date')

    df['ma5'] = df['close'].rolling(window_ma5).mean()
    df['ma10'] = df['close'].rolling(window_ma10).mean()

    df['golden_cross'] = (df['ma5'] > df['ma10']) & (df['ma5'].shift(1) <= df['ma10'].shift(1))
    return df

def precompute_macd_signals(df: pd.DataFrame, short: int = 12, long: int = 26, mid: int = 9) -> pd.DataFrame:
    """预计算 MACD 信号"""
    df = df.sort_values(['code', 'trade_date'])
    macd_result = calculate_macd(short=short, long=long, mid=mid, data=df['close'].tolist())
    df['macd'] = macd_result['macd']
    df['dif'] = macd_result['dif']
    df['dea'] = macd_result['dea']
    return df

def precompute_macd_cross_signals(df: pd.DataFrame) -> pd.DataFrame:
    """预计算 MACD 金叉信号"""
    df = df.sort_values(['code', 'trade_date'])
    df['macd_cross'] = (df['dif'] > df['dea']) & (df['dif'].shift(1) <= df['dea'].shift(1))
    return df

def precompute_boll_cross_signal(df: pd.DataFrame) -> pd.DataFrame:
    """预计算 BOLL 交叉信号"""
    df = df.sort_values(['code', 'trade_date'])
    df['boll_cross'] = (df['close'] > df['BOLL_MID']) & (df['high'] < df['BOLL_UP'])
    return df

def precompute_boll_mid_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """预计算 BOLL_MID 涨幅"""
    df = df.sort_values(['code', 'trade_date'])
    df['boll_mid_ratio'] = df['BOLL_MID'].pct_change(periods=1, fill_method=None)
    return df

def compute_signals_for_code(group: pd.DataFrame, chart_days: int):
    # 按trade_date升序排序，确保rolling和shift计算正确
    group = group.sort_values('trade_date', ascending=True)
    group = compute_boll(group, window=20)
    group = precompute_macd_signals(group, short=12, long=26, mid=9)
    group = precompute_macd_cross_signals(group)
    group = precompute_ma_signals(group, window_ma5=5, window_ma10=10)
    group = precompute_boll_cross_signal(group)
    group = precompute_boll_mid_ratio(group)
    group = group.tail(chart_days)
    return group