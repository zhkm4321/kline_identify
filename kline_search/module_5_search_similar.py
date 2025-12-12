# module_5_search_similar.py
"""
模块5：相似走势检索与可视化
功能：基于向量检索找到相似的K线走势，并可视化对比结果

依赖:
 - index/windows.npy              (模块1生成)
 - index/windows_meta.json        (模块1生成)
 - index/embedding_model.pt       (模块3生成)
 - LanceDB向量表                   (模块4生成)
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lancedb
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foundation.kline_data import KLineDataService, StockQuoteDBService, get_kline_service
from foundation.kline_image import generate_kline_charts
from module_2_model import KlineEncoder

# ----------------------------
# 配置matplotlib中文字体支持
# ----------------------------
# Windows系统常用中文字体：SimHei(黑体), Microsoft YaHei(微软雅黑), SimSun(宋体)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ----------------------------
# 配置（按需修改）
# ----------------------------
WINDOWS_PATH = "index/windows.npy"
META_PATH = "index/windows_meta.json"
MODEL_PATH = "index/embedding_model.pt"
MAPPING_PATH = "index/code_row_mapping.json"  # 模块4生成的映射文件
LANCEDB_PATH = "E:/data/kline_lance"          # LanceDB目录（与模块4一致）
SEARCH_RESULT_DIR = "kline_search/search_result"           # 搜索结果输出目录
TOP_K = 5                                      # 默认检索数量
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模块1中FEATS的索引映射（用于可视化时选择正确的特征）
# 必须与 module_1_load_windows.py 中的 FEATS 列表顺序完全一致！
FEAT_INDEX = {
    # 收益率/对数收益率
    'return_open': 0,      # 开盘价收益率
    'return_close': 1,     # 收盘价收益率
    'return_high': 2,      # 最高价收益率
    'return_low': 3,       # 最低价收益率
    'logret_close': 4,     # 对数收益率
    # 成交量
    'volume_chg': 5,       # 成交量变化率
    # 原始值的Z-score
    'open_z': 6,           # 开盘价Z-score
    'close_z': 7,          # 收盘价Z-score（推荐用于可视化）
    'high_z': 8,           # 最高价Z-score
    'low_z': 9,            # 最低价Z-score
    'volume_z': 10,        # 成交量Z-score
    # MACD
    'macd_line': 11,       # MACD线
    'macd_signal': 12,     # MACD信号线
    'macd_hist': 13,       # MACD柱状图
    # RSI
    'rsi_14': 14,          # 14日RSI
    # 布林带
    'boll_width': 15,      # 布林带宽度
    'boll_dist': 16,       # 到布林带中轨距离
    # 均线斜率
    'ma5_slope': 17,       # 5日均线斜率
    'ma10_slope': 18,      # 10日均线斜率
    'ma20_slope': 19,      # 20日均线斜率
    # 其他
    'hl_range': 20,        # 振幅
}
DEFAULT_PLOT_FEAT = 'close_z'  # 默认绘制收盘价Z-score


# ----------------------------
# 辅助函数：加载checkpoint并恢复encoder
# 兼容module_3_train.py保存的checkpoint格式
# ----------------------------
def load_encoder_from_checkpoint(checkpoint_path, device=DEVICE):
    """
    从checkpoint加载编码器

    参数:
        checkpoint_path: 模型文件路径
        device: 计算设备

    返回:
        encoder: 已加载权重的KlineEncoder实例（eval模式）
        latent_dim: 隐层维度
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 读取元数据，如不存在则使用默认值
    in_channels = int(ckpt.get('in_channels', 5))
    latent_dim = int(ckpt.get('latent_dim', 32))

    # 构造encoder并加载权重
    encoder = KlineEncoder(in_channels=in_channels, latent_dim=latent_dim)
    state = ckpt.get('state_dict', ckpt)

    # state可能是完整模型的state_dict，键名格式为'encoder.<layer>...'
    encoder_state = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            # 去掉'encoder.'前缀
            new_k = k[len("encoder."):]
            encoder_state[new_k] = v
        elif k.startswith("module.encoder."):
            # 如果使用DataParallel保存的模型
            new_k = k[len("module.encoder."):]
            encoder_state[new_k] = v
        else:
            # 如果键名已经匹配encoder层，直接使用
            encoder_state[k] = v

    # 加载权重（允许部分匹配，避免严格模式下的报错）
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.to(device)
    encoder.eval()
    return encoder, latent_dim


# ----------------------------
# LanceDB表模式检测
# ----------------------------
def detect_table_mode(db):
    """
    检测模块4使用的表模式

    返回:
        mode: 'single'(单表模式) 或 'multi'(多表模式)
        table_names: 表名列表
    """
    # 先检查映射文件判断模式
    if os.path.exists(MAPPING_PATH):
        try:
            mapping = json.load(open(MAPPING_PATH, 'r', encoding='utf-8'))
            if 'tables' in mapping:
                # 多表模式
                return 'multi', mapping['tables']
            else:
                # 单表模式
                return 'single', ['kline_embeddings']
        except Exception:
            pass

    # 如果没有映射文件，尝试自动检测
    try:
        db.get_table("kline_embeddings")
        return 'single', ['kline_embeddings']
    except Exception:
        pass

    # 尝试列出所有表
    try:
        table_names = []
        if hasattr(db, "table_names"):
            table_names = db.table_names()
        elif hasattr(db, "list_tables"):
            table_names = db.list_tables()

        # 过滤出kline_embeddings_开头的表
        kline_tables = [t for t in table_names if t.startswith('kline_embeddings_')]
        if kline_tables:
            return 'multi', kline_tables
    except Exception:
        pass

    raise RuntimeError("未找到可用的向量表。请先运行模块4生成数据。")


def pick_vector_table(db, target_code=None):
    """
    选择向量表

    参数:
        db: LanceDB连接
        target_code: 目标股票代码（可选）

    返回:
        table: 表对象
        table_name: 表名
        mode: 表模式
    """
    mode, table_names = detect_table_mode(db)

    if mode == 'single':
        tbl = db.get_table("kline_embeddings")
        return tbl, "kline_embeddings", mode
    else:
        # 多表模式
        if target_code:
            tbl_name = f'kline_embeddings_{target_code}'
            if tbl_name in table_names:
                tbl = db.get_table(tbl_name)
                return tbl, tbl_name, mode
            else:
                raise ValueError(f"表 {tbl_name} 不存在。可用表: {table_names[:5]}...")
        else:
            # 返回第一个表（仅用于单股票搜索）
            tbl_name = table_names[0]
            tbl = db.get_table(tbl_name)
            return tbl, tbl_name, mode


def search_all_tables(db, query_emb, top_k):
    """
    在多表模式下搜索所有表，合并结果返回top_k

    参数:
        db: LanceDB连接
        query_emb: 查询向量
        top_k: 返回数量

    返回:
        DataFrame: 合并后的搜索结果
    """
    mode, table_names = detect_table_mode(db)

    if mode == 'single':
        tbl = db.open_table("kline_embeddings")
        return tbl.search(query_emb).limit(top_k).to_pandas()

    # 多表模式：搜索所有表
    all_hits = []
    for tbl_name in table_names:
        try:
            tbl = db.open_table(tbl_name)
            hits = tbl.search(query_emb).limit(top_k).to_pandas()
            all_hits.append(hits)
        except Exception as e:
            print(f"警告: 搜索表 {tbl_name} 失败: {e}")
            continue

    if not all_hits:
        return pd.DataFrame()

    # 合并并按距离排序
    combined = pd.concat(all_hits, ignore_index=True)
    if '_distance' in combined.columns:
        combined = combined.nsmallest(top_k, '_distance')

    return combined


# ----------------------------
# 主流程：搜索 + 可视化
# ----------------------------
def compute_embedding(encoder, window_array, device=DEVICE):
    """
    计算单个窗口的embedding

    参数:
        encoder: 编码器模型
        window_array: 窗口数据，形状 (seq_len, in_channels)
        device: 计算设备

    返回:
        一维numpy数组，embedding向量
    """
    x = torch.tensor(window_array, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = encoder(x)
    return emb.cpu().numpy().reshape(-1)


def search_similar_by_window_id(window_id, top_k=TOP_K, same_stock_only=False):
    """
    根据窗口ID搜索相似走势

    参数:
        window_id: 查询窗口的索引
        top_k: 返回的相似结果数量
        same_stock_only: 是否只在同一股票内搜索

    返回:
        windows: 窗口数据数组
        meta: 元数据列表
        hits_df: 搜索结果DataFrame
        window_id: 查询窗口ID
        table_info: 使用的表信息
    """
    # 加载窗口数据和元数据
    windows = np.load(WINDOWS_PATH)
    meta = json.load(open(META_PATH, 'r', encoding='utf-8'))

    if window_id < 0 or window_id >= len(windows):
        raise IndexError(f"window_id {window_id} 超出范围 (0..{len(windows)-1})")

    # 加载编码器
    encoder, latent_dim = load_encoder_from_checkpoint(MODEL_PATH, device=DEVICE)

    # 计算查询向量
    q_win = windows[window_id]
    q_emb = compute_embedding(encoder, q_win, device=DEVICE)
    query_code = meta[window_id]['code']

    # 连接LanceDB
    db = lancedb.connect(LANCEDB_PATH)

    # 执行搜索
    try:
        if same_stock_only:
            # 只在同一股票内搜索
            mode, _ = detect_table_mode(db)
            if mode == 'single':
                tbl = db.get_table("kline_embeddings")
                hits = tbl.search(q_emb).where(f"code = '{query_code}'").limit(top_k).to_pandas()
                tbl_name = "kline_embeddings"
            else:
                # 多表模式，直接搜索该股票的表
                tbl, tbl_name, _ = pick_vector_table(db, target_code=query_code)
                hits = tbl.search(q_emb).limit(top_k).to_pandas()
        else:
            # 全局搜索所有股票
            hits = search_all_tables(db, q_emb, top_k)
            tbl_name = "all_tables"
    except Exception as e:
        # 降级处理：尝试使用list格式
        try:
            hits = search_all_tables(db, q_emb.tolist(), top_k)
            tbl_name = "all_tables (fallback)"
        except Exception as e2:
            raise RuntimeError(f"LanceDB搜索失败: {e}; 降级处理也失败: {e2}")

    return windows, meta, hits, window_id, tbl_name

# ----------------------------
# 辅助函数：根据股票代码和日期查找窗口ID
# ----------------------------
def find_window_by_code_and_date(meta, code, date):
    """
    根据股票代码和日期查找最匹配的窗口ID
    
    参数:
        meta: 元数据列表
        code: 股票代码
        date: 查询日期（窗口结束日期）
        
    返回:
        window_id: 匹配的窗口索引，未找到返回None
    """
    code = str(code)
    date = str(date)
    
    # 优先精确匹配 end_date
    for idx, m in enumerate(meta):
        if m.get('code') == code and m.get('end_date') == date:
            return idx
    
    # 如果没有精确匹配，尝试匹配 start_date
    for idx, m in enumerate(meta):
        if m.get('code') == code and m.get('start_date') == date:
            return idx
    
    # 如果还是没有，找该股票最接近该日期的窗口
    candidates = []
    for idx, m in enumerate(meta):
        if m.get('code') == code:
            candidates.append((idx, m.get('end_date', '')))
    
    if not candidates:
        return None
    
    # 按日期排序，找最接近的
    candidates.sort(key=lambda x: abs(hash(x[1]) - hash(date)))
    
    # 返回日期最接近的（简单实现，实际应该用日期比较）
    # 找到 <= date 的最大日期
    valid = [(idx, d) for idx, d in candidates if d <= date]
    if valid:
        valid.sort(key=lambda x: x[1], reverse=True)
        return valid[0][0]
    
    # 如果没有 <= date 的，返回最早的
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


# ----------------------------
# 辅助函数：保存搜索结果到CSV
# ----------------------------
def save_search_result_to_csv(hits_df, query_meta, search_params, output_dir=SEARCH_RESULT_DIR):
    """
    将搜索结果保存到CSV文件
    
    参数:
        hits_df: 搜索结果DataFrame
        query_meta: 查询窗口的元数据
        search_params: 搜索参数字典，包含:
            - window_id: 窗口ID (可选)
            - code: 股票代码 (可选)
            - date: 日期 (可选)
            - k: Top-K数量
            - same_stock: 是否同股票搜索
        output_dir: 输出目录
        
    返回:
        csv_path: 保存的CSV文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 构建文件名
    if search_params.get('code'):
        # 通过股票代码+日期搜索
        code = search_params['code']
        date = search_params.get('date', '').replace('-', '')
        k = search_params.get('k', TOP_K)
        same_stock_flag = "_same" if search_params.get('same_stock') else ""
        filename = f"{code}_{date}_k{k}{same_stock_flag}_{timestamp}.csv"
    else:
        # 通过窗口ID搜索
        window_id = search_params.get('window_id', 0)
        k = search_params.get('k', TOP_K)
        same_stock_flag = "_same" if search_params.get('same_stock') else ""
        filename = f"window_{window_id}_k{k}{same_stock_flag}_{timestamp}.csv"
    
    csv_path = os.path.join(output_dir, filename)
    
    # 准备输出数据
    output_df = hits_df.copy()
    
    # 移除不需要保存的列（如向量数据）
    cols_to_drop = ['vector', 'raw']
    for col in cols_to_drop:
        if col in output_df.columns:
            output_df = output_df.drop(columns=[col])
    
    # 添加查询信息列
    output_df.insert(0, 'query_code', query_meta.get('code', ''))
    output_df.insert(1, 'query_start_date', query_meta.get('start_date', ''))
    output_df.insert(2, 'query_end_date', query_meta.get('end_date', ''))
    
    # 添加排名列
    output_df.insert(0, 'rank', range(1, len(output_df) + 1))
    
    # 保存CSV
    output_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return csv_path


# ----------------------------
# 命令行入口
# ----------------------------
if __name__ == "__main__":
    import json as _json
    from contextlib import redirect_stdout
    import argparse

    # 可用的特征名称
    available_feats = list(FEAT_INDEX.keys())

    parser = argparse.ArgumentParser(
        description="搜索相似K线走势并可视化结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法:
  # 方式1：通过窗口ID搜索
  python module_5_search_similar.py -q 100 -k 5
  
  # 方式2：通过股票代码和日期搜索
  python module_5_search_similar.py -c 000001 -d 2024-01-15 -k 10
  
  # 只在同一股票历史中搜索
  python module_5_search_similar.py -c 000001 -d 2024-01-15 -k 5 --same-stock
  
  # 指定可视化特征
  python module_5_search_similar.py -q 100 -k 10 --feat close_z

可用特征: {', '.join(available_feats)}
        """
    )
    parser.add_argument("--query", "-q", type=int, default=None,
                        help="查询窗口ID（index/windows.npy中的索引）")
    parser.add_argument("--code", "-c", type=str, default=None,
                        help="股票代码（与 -d 配合使用）")
    parser.add_argument("--date", "-d", type=str, default=None,
                        help="查询日期，格式如 2024-01-15（窗口结束日期）")
    parser.add_argument("--k", "-k", type=int, default=TOP_K,
                        help=f"返回的相似结果数量（默认: {TOP_K}）")
    parser.add_argument("--feat", type=str, default=DEFAULT_PLOT_FEAT,
                        choices=available_feats,
                        help=f"可视化时使用的特征（默认: {DEFAULT_PLOT_FEAT}）")
    parser.add_argument("--same-stock", "--same_stock", action="store_true",
                        dest="same_stock",
                        help="只在同一股票内搜索相似走势")
    args = parser.parse_args()

    # 约定：stdout 只输出 JSON（方便调用方解析）；其它输出全部走 stderr
    _stdout = sys.stdout
    _result_json = {
        "ok": False,
        "query": {
            "window_id": None,
            "code": args.code,
            "date": args.date,
        },
        "top_k": args.k,
        "same_stock": bool(args.same_stock),
        "csv_path": None,
        "image_dir": None,
        "hits": 0,
        "error": None,
    }

    try:
        with redirect_stdout(sys.stderr):
            # 参数校验
            if args.query is None and (args.code is None or args.date is None):
                raise ValueError("请指定 -q (窗口ID) 或 -c (股票代码) 和 -d (日期)")
            
            if args.query is not None and args.code is not None:
                print("警告: 同时指定了 -q 和 -c，将优先使用 -q (窗口ID)")

            k = args.k
            feat = args.feat
            same_stock = args.same_stock

            # 检查必要文件是否存在
            for path, desc in [(WINDOWS_PATH, "窗口数据"), (META_PATH, "元数据"), (MODEL_PATH, "模型")]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{desc}文件不存在: {path}")

            # 加载元数据（用于根据code+date查找window_id）
            meta = json.load(open(META_PATH, 'r', encoding='utf-8'))
            
            # 确定查询窗口ID
            if args.query is not None:
                qid = args.query
            else:
                # 根据股票代码和日期查找
                qid = find_window_by_code_and_date(meta, args.code, args.date)
                if qid is None:
                    print(f"错误: 未找到股票 {args.code} 在日期 {args.date} 附近的窗口")
                    print(f"提示: 请确认股票代码正确，且日期在数据范围内")
                    # 显示该股票可用的日期范围
                    code_windows = [(m['start_date'], m['end_date']) for m in meta if m['code'] == args.code]
                    if code_windows:
                        dates = sorted(set([d for pair in code_windows for d in pair]))
                        print(f"该股票可用日期范围: {dates[0]} ~ {dates[-1]}")
                    raise ValueError("未找到匹配窗口")
                print(f"找到匹配窗口: 股票={args.code}, 日期={args.date} -> 窗口ID={qid}")

            _result_json["query"]["window_id"] = qid

            print("=" * 50)
            print("K线相似走势检索")
            print("=" * 50)
            print(f"模型路径: {MODEL_PATH}")
            print(f"查询窗口ID: {qid}")
            if args.code:
                print(f"查询股票: {args.code}")
                print(f"查询日期: {args.date}")
            print(f"Top-K: {k}")
            print(f"只在同一股票内搜索: {same_stock}")
            print(f"可视化特征: {feat}")
            print("-" * 50)

            # 加载编码器（保持原行为，日志重定向到stderr）
            encoder, latent_dim = load_encoder_from_checkpoint(MODEL_PATH, device=DEVICE)
            print(f"编码器加载完成. 隐层维度: {latent_dim}")

            # 搜索
            windows, meta, hits, query_id, table_name = search_similar_by_window_id(
                qid, top_k=k, same_stock_only=same_stock
            )

            print(f"\n搜索表: {table_name}")
            print(f"查询窗口: {meta[query_id]}")
            print(f"\n找到 {len(hits)} 个相似走势:")
            print("-" * 50)

            _result_json["hits"] = int(len(hits))

            # 显示结果
            display_cols = ['code', 'start_date', 'end_date', '_distance']
            display_cols = [c for c in display_cols if c in hits.columns]
            if display_cols:
                print(hits[display_cols].to_string(index=False))
            else:
                print(hits)

            # 保存搜索结果到CSV + 生成图片
            if not hits.empty:
                search_params = {
                    'window_id': qid,
                    'code': args.code,
                    'date': args.date,
                    'k': k,
                    'same_stock': same_stock
                }
                csv_path = save_search_result_to_csv(hits, meta[query_id], search_params)
                csv_abs = os.path.abspath(csv_path)
                _result_json["csv_path"] = csv_abs
                print(f"\n搜索结果已保存到: {csv_abs}")

                stock_quote_service = StockQuoteDBService(use_cache=False)
                # 遍历hits，获取code，并取对应df
                df_list = []
                for code in hits['code'].tolist():
                    df = stock_quote_service.get_stock_data(code=code, start_date=None, end_date=None, with_tech_signals=True)
                    df_list.append(df)
                merged_df = pd.concat(df_list)

                # 输出目录名：从 CSV 文件名前三段组成
                file_name = os.path.basename(csv_path)
                parts = file_name.split('_')
                if len(parts) >= 3:
                    prefix = '_'.join(parts[:3])
                else:
                    prefix = file_name.rsplit('.', 1)[0]
                img_output_dir = os.path.join(SEARCH_RESULT_DIR, prefix).replace('.', '_')
                os.makedirs(img_output_dir, exist_ok=True)
                img_abs = os.path.abspath(img_output_dir)
                _result_json["image_dir"] = img_abs

                # 从hits这个df中取出code,start_date,end_date,_distance这几列，转换为list[Dict]
                hits_list = hits[['code', 'start_date', 'end_date', '_distance']].to_dict(orient='records')
                ret = generate_kline_charts(df=merged_df, result=hits_list, left_align_days=30, right_align_days=30, output_dir=img_output_dir)
                # ret 可能是 (bool, dir) 或 bool
                if isinstance(ret, tuple) and len(ret) >= 1:
                    _result_json["ok"] = bool(ret[0])
                else:
                    _result_json["ok"] = bool(ret)

                print(f"图片已保存到: {img_abs}")
            else:
                _result_json["ok"] = True
                print("\n未找到相似走势。")

    except Exception as e:
        _result_json["ok"] = False
        _result_json["error"] = str(e)

    # 最终：stdout 只输出 JSON（一行）
    _stdout.write(_json.dumps(_result_json, ensure_ascii=False) + "\n")
