#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
from typing import Any, Dict, List, Tuple
import pandas as pd
import mplfinance as mpf
import matplotlib

from foundation.stock_cache import stock_cache

matplotlib.use('Agg')  # 使用非交互式后端，避免多线程问题
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_kline_charts(
    df: pd.DataFrame,
    result:List[Dict],
    left_align_days: int = 10,
    right_align_days: int = 10,
    output_dir: str = None, 
    max_workers: int = 4
):
    """
    生成K线图
    
    Args:
        df: 数据DataFrame
        result: 搜索结果字典
        left_align_days: 左对齐天数，默认为10天
        right_align_days: 右对齐天数，默认为10天
        output_dir: 输出目录路径，如果为None则抛出异常
        max_workers: 最大线程/进程数，默认为4
    
    Returns:
        bool: 是否成功生成K线图
    """
    try:
        if output_dir is None:
            raise ValueError("output_dir不能为None,请指定输出目录")
        print("=" * 80)
        print("开始生成K线图")
        print("=" * 80)
        
        # 设置中文字体支持
        setup_chinese_font()
        
        # 统计成功生成的K线图数量
        success_count = 0

        # 循环遍历搜索结果
        for idx, r in enumerate(result):
            code = r.get('code')
            start_date = r.get('start_date')
            end_date = r.get('end_date')

            # 计算所需区间（将字符串转换为日期类型）
            chart_start = pd.to_datetime(start_date) - pd.Timedelta(days=left_align_days)
            chart_end = pd.to_datetime(end_date) + pd.Timedelta(days=right_align_days)

            # 筛选对应股票和区间的数据
            code_df = df[df['code'] == code].copy()
            if code_df.empty:
                print(f"❌ 股票{code}无数据，跳过")
                continue

            code_df['trade_date'] = pd.to_datetime(code_df['trade_date'])
            code_df = code_df.set_index('trade_date', drop=False)
            date_range_df = code_df.loc[(code_df.index >= chart_start) & (code_df.index <= chart_end)]
            if date_range_df.empty:
                print(f"❌ 股票{code}在区间{chart_start.date()}~{chart_end.date()}无数据，跳过")
                continue
            # 处理数据格式
            df_processed = prepare_data_for_chart(date_range_df)
            if df_processed.empty:
                print("❌ 数据处理失败")
                return False
            r['kline_data'] = df_processed

        
        # 生成K线图
        success_count = generate_charts_for_stocks(result, output_dir, max_workers)
        
        print(f"\n✅ K线图生成完成！")
        print(f"   成功生成: {success_count} 张K线图")
        print(f"   输出目录: {output_dir}")
        
        return success_count > 0, output_dir
        
    except Exception as e:
        print(f"❌ 生成K线图时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def setup_chinese_font():
    """设置中文字体支持"""
    try:
        # Windows 常见字体路径
        font_path = "C:\\Windows\\Fonts\\msyh.ttc"
        if os.path.exists(font_path):
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ 中文字体设置完成")
    except Exception as e:
        print(f"⚠️ 字体设置警告: {e}")


def prepare_data_for_chart(df):
    """准备用于绘制K线图的数据"""
    try:
        print("正在处理数据格式...")
        
        # 如果 trade_date 同时是索引和列，先重置索引避免歧义
        if df.index.name == 'trade_date' and 'trade_date' in df.columns:
            df = df.reset_index(drop=True)
        # 如果 trade_date 只是索引，将其转为列
        elif df.index.name == 'trade_date':
            df = df.reset_index()
        
        # 确保日期格式正确
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 检查必要的列
        colmap = {c.lower(): c for c in df.columns}
        needed = {}
        for want in ('code', 'open', 'high', 'low', 'price'):
            if want in colmap:
                needed[want] = colmap[want]
            else:
                raise ValueError(f"缺少必要列: {want}")
        
        # 创建新的DataFrame并重命名列
        df2 = df[['trade_date', needed['code'], needed['open'], 
                 needed['high'], needed['low'], needed['price']]].copy()
        
        df2 = df2.rename(columns={
            needed['code']: 'code',
            needed['open']: 'Open',
            needed['high']: 'High',
            needed['low']: 'Low',
            needed['price']: 'Close'
        })
        
        # 添加成交量列（如果存在）
        if 'volume' in colmap:
            df2['Volume'] = df[colmap['volume']]
        
        # 按时间排序并设为索引
        df2 = df2.sort_values('trade_date')
        df2.set_index('trade_date', inplace=True)
        
        # 确保数值类型正确
        df2[['Open', 'High', 'Low', 'Close']] = df2[['Open', 'High', 'Low', 'Close']].apply(
            pd.to_numeric, errors='coerce'
        )
        
        # 删除包含NaN的行
        df2 = df2.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        print(f"✅ 数据处理完成，共 {len(df2)} 条记录")
        return df2
        
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        return pd.DataFrame()


def generate_charts_for_stocks(result: List[Dict], output_dir: str, max_workers: int = 4) -> Tuple[int, str]:
    """
    并发生成多只股票的K线图

    线程利用效率为何不高的分析：
    ----------------------------------------------------------
    1. 每个K线图生成任务的计算量很小，且主要瓶颈在于matplotlib的CPU绘图与图片IO保存，不是密集型的长时间CPU计算；
    2. GIL（全局解释器锁）限制了多线程对CPU的并行利用，matplotlib、pandas、numpy等操作不会真正做到多核并行，尤其在单进程多线程模式下；
    3. 大量任务很快结束或在IO（写磁盘图片）等待，导致线程未饱和；
    4. 线程创建和切换本身有不小的开销，对于K线图这种“轻量+大量”任务，ThreadPoolExecutor分配的线程不会长时间占满CPU，很多时间其实在等待matplotlib绘制、pandas处理、磁盘写入，空闲线程等待队列；
    5. matplotlib非线程安全，多线程环境下也可能被GIL/锁机制序列化，实际上几乎像是串行执行。
    6. 部分股票已存在图片被跳过，真实需要处理的任务数很少，也导致线程利用不满；
    ----------------------------------------------------------
    结论：CPU线程利用率低，多为“线程等待/IO等待/被GIL串行”。如要最佳利用多核，建议用多进程模式（ProcessPoolExecutor），或者将真正的CPU密集绘图任务通过多进程分发。

    """
    try:
        # --- 样式 ---
        mc = mpf.make_marketcolors(up='r', down='g', edge='inherit', wick='inherit', volume='in')
        style = mpf.make_mpf_style(
            base_mpf_style='yahoo',
            marketcolors=mc,
            rc={'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei']}
        )
        # 遍历result将df数据合并到一个DataFrame，同时记录每个code对应的distance
        dfs = []
        code_distance_map = {}  # 存储 code -> distance 的映射
        code_window_map = {}    # 存储 code -> (start_date, end_date) 的映射（用于标注）
        for item in result:
            df = item.get("kline_data")
            if df is not None and not df.empty:
                # 确保每个df包含 'code' 一列（若未包含则尝试从item键推断）
                code_val = item.get("code")
                if 'code' not in df.columns:
                    if code_val is not None:
                        df = df.copy()  # 避免 SettingWithCopyWarning
                        df['code'] = code_val
                # 保存 distance 信息
                if code_val is not None:
                    distance = item.get("_distance")
                    if distance is not None:
                        code_distance_map[code_val] = distance
                    # 保存窗口日期范围信息（同一 code 只记录第一条即可）
                    if code_val not in code_window_map:
                        code_window_map[code_val] = (item.get("start_date"), item.get("end_date"))
                dfs.append(df)
        if not dfs:
            print("❌ 没有可用的数据可供合并")
            return 0
        dfs_reset = [d.reset_index() for d in dfs]
        df = pd.concat(dfs_reset, axis=0, ignore_index=True)
        df_grouped = df.groupby('code')

        tasks = []
        for code, group in df_grouped:
            stock = stock_cache.get_stock_by_ts_code(code)
            if stock is None:
                print(f"⚠️ 跳过未知股票代码: {code}")
                continue
            save_path_jpg = os.path.join(output_dir, f'{stock.exchange}_{stock.symbol}.jpg')
            save_path_png = os.path.join(output_dir, f'{stock.exchange}_{stock.symbol}.png')
            if os.path.exists(save_path_jpg) or os.path.exists(save_path_png):
                continue
            # 获取该股票的 distance 值
            distance = code_distance_map.get(code)
            win_start, win_end = code_window_map.get(code, (None, None))
            tasks.append((code, group, output_dir, None, style, stock, distance, win_start, win_end))

        if not tasks:
            print("⚠️ 没有需要生成的任务")
            return 0

        print(f"开始为 {len(tasks)} 只股票生成K线图...")

        success_count = 0
        success_lock = threading.Lock()

        # 提前提交所有任务，使用 as_completed 处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_single_stock_chart, *t): t[0] for t in tasks}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    if future.result():
                        with success_lock:
                            success_count += 1
                except Exception as e:
                    print(f"❌ 股票 {code} 任务失败: {e}")

        print(f"🎯 所有任务完成，共成功 {success_count}/{len(tasks)} 个")
        return success_count

    except Exception as e:
        print(f"❌ 批量生成K线图失败: {e}")
        return 0


def generate_single_stock_chart(code, group, output_dir, chart_days, style, stock, distance=None, win_start=None, win_end=None):
    fig = None
    try:
        # 将 trade_date 列转为 DatetimeIndex（mplfinance 要求）
        if 'trade_date' in group.columns:
            group = group.copy()
            group['trade_date'] = pd.to_datetime(group['trade_date'])
            group = group.set_index('trade_date')
        
        group = group.sort_index().drop_duplicates()
        if len(group) < 20:  # 最少保证可以计算 BOLL
            return False

        # ================= 计算 BOLL =================
        close = group['Close']
        mid = close.rolling(20, min_periods=1).mean()
        std = close.rolling(20, min_periods=1).std()
        upper = mid + 2*std
        lower = mid - 2*std

        group['BOLL_MID'] = mid
        group['BOLL_UP'] = upper
        group['BOLL_LOW'] = lower

        # ================= 判断上升通道 =================
        cond1 = group['BOLL_MID'] > group['BOLL_MID'].shift(1)
        cond2 = group['BOLL_UP']  > group['BOLL_UP'].shift(1)
        cond3 = group['Close'] > group['BOLL_MID']
        cond4 = group['High'] >= group['BOLL_UP'].shift(1)
        group['UP_CHANNEL'] = cond1 & cond2 & cond3 & cond4

        # ================= 截取最后 chart_days =================
        if chart_days is not None:
            group = group.tail(chart_days)
            if len(group) < chart_days:
                return False

        if 'Volume' not in group.columns:
            group['Volume'] = 0

        # ================= 构建 addplot =================
        # 普通 BOLL 线
        add_plots = [
            mpf.make_addplot(group['BOLL_MID'], color='blue', width=0.6),
            mpf.make_addplot(group['BOLL_UP'],  color='red',  width=0.6),
            mpf.make_addplot(group['BOLL_LOW'], color='green', width=0.6)
        ]

        # 高亮上升通道中轨（只保留 UP_CHANNEL=True，其余 NaN）
        highlighted_mid = group['BOLL_MID'].where(group['UP_CHANNEL'])
        if highlighted_mid.notna().any():
            add_plots.append(
                mpf.make_addplot(highlighted_mid, color='limegreen', width=1.2)
            )
        # 构建标题，如果有距离值则拼接显示
        if distance is not None:
            title_str = f'{stock.name}({stock.ts_code}) DST:{distance:.4f}'
        else:
            title_str = f'{stock.name}({stock.ts_code})'
        # ================= 绘图 =================
        save_path = os.path.join(output_dir, f'{stock.exchange}_{stock.symbol}.png')
        fig, axlist = mpf.plot(
            group,
            type='candle',
            style=style,
            volume=True,
            returnfig=True,
            addplot=add_plots,
            title=title_str,
            ylabel='价格',
            ylabel_lower='成交量',
            datetime_format='%m-%d'
        )

        # ================= 标注窗口日期范围 + 背景高亮 =================
        try:
            # 整体背景色（轻微提升可读性）
            fig.patch.set_facecolor('#f6f8ff')
            for ax in axlist:
                ax.set_facecolor('#ffffff')

            if win_start and win_end:
                start_dt = pd.to_datetime(win_start)
                end_dt = pd.to_datetime(win_end)

                # 主图轴一般是第一个
                ax0 = axlist[0] if axlist else None
                if ax0 is not None:
                    # mplfinance 的 x 轴通常是“整数位置(0..N)”而不是 datetime。
                    # 因此这里把 start/end 映射到 group.index 的整数位置，避免把 x 轴范围拉爆。
                    xlim_before = ax0.get_xlim()

                    # 兼容：若 start/end 超出当前数据范围，则裁剪到可见范围
                    x_min_dt = group.index.min()
                    x_max_dt = group.index.max()
                    if pd.notna(x_min_dt) and pd.notna(x_max_dt):
                        start_dt = max(start_dt, x_min_dt)
                        end_dt = min(end_dt, x_max_dt)

                    if start_dt <= end_dt and len(group.index) > 0:
                        # searchsorted: 不要求日期必须是交易日
                        start_loc = int(group.index.searchsorted(start_dt, side='left'))
                        end_loc = int(group.index.searchsorted(end_dt, side='right')) - 1

                        # 裁剪到 [0, len-1]
                        start_loc = max(0, min(start_loc, len(group.index) - 1))
                        end_loc = max(0, min(end_loc, len(group.index) - 1))
                        if end_loc < start_loc:
                            start_loc, end_loc = end_loc, start_loc

                        # 区间背景色（高亮“这段K线”）
                        ax0.axvspan(start_loc - 0.5, end_loc + 0.5, facecolor='#fff2b2', alpha=0.35, zorder=0)
                        # 起止竖线
                        ax0.axvline(start_loc, color='#ff8c00', linestyle='--', linewidth=0.9, alpha=0.9)
                        ax0.axvline(end_loc, color='#ff8c00', linestyle='--', linewidth=0.9, alpha=0.9)

                        # 顶部左侧文字标注（原始字符串更直观）
                        label = f"范围: {win_start} ~ {win_end}"
                        ax0.text(
                            0.01, 0.99, label,
                            transform=ax0.transAxes,
                            ha='left', va='top',
                            fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.25')
                        )

                    # 防止 axvspan/axvline 触发 autoscale 影响 x 轴范围
                    ax0.set_xlim(xlim_before)
        except Exception:
            # 标注失败不影响出图
            pass

        # ================= 保存并关闭 =================
        fig.savefig(save_path, dpi=80, bbox_inches='tight', format='png')
        plt.close(fig)
        return True

    except Exception as e:
        if fig is not None:
            plt.close(fig)
        print(f"❌ 生成股票 {code} 的K线图失败: {e}")
        return False

