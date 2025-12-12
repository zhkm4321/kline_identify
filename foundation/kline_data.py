"""
K线数据服务 - 提供股票数据加载、预处理和技术指标计算

提供功能：
1. CSV 读取和数据预处理
2. 技术指标计算支持（通过 tech_signals 模块）
3. Redis 缓存支持（带日期验证）
4. 数据库查询支持（pymysql，查询 t_stock_quote 表）

使用示例:
    from foundation.kline_data import KLineDataService
    
    # 创建服务实例
    service = KLineDataService(csv_path, use_cache=True)
    
    # 获取单只股票数据（带技术指标）
    df = service.get_stock_data("000001.SZ")
    
    # 获取所有股票数据
    all_data = service.get_all_stock_data()
    
    # 也可以直接使用底层函数
    from foundation.kline_data import read_stock_data_csv
    df = read_stock_data_csv(csv_path)
    
    # 使用数据库查询服务
    from foundation.kline_data import StockQuoteDBService
    db_service = StockQuoteDBService()
    df = db_service.get_stock_data("000001.SZ", start_date="2024-01-01", end_date="2024-12-31")
"""
import os
import sys
from typing import Dict, List, Optional, Tuple, Callable
from io import StringIO
import pandas as pd
import pymysql

# 定义 signal_processor 函数的类型签名
# 签名: (df: pd.DataFrame, chart_days: int) -> pd.DataFrame
SignalProcessorType = Callable[[pd.DataFrame, int], pd.DataFrame]
from pymysql.cursors import DictCursor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foundation.redis_client import get_redis_client
from foundation.stock_cache import stock_cache
from foundation.tech_signals import compute_signals_for_code
from foundation.app_env import db_config

# Redis 缓存配置
REDIS_CACHE_KEY_PREFIX = "kline:tech_signals:"
REDIS_CACHE_EXPIRE_SECONDS = 60 * 60 * 24  # 缓存 24 小时


# ==================== 数据处理工具函数 ====================

def remove_empty_columns(df: pd.DataFrame, min_data_ratio: float = 0.0) -> pd.DataFrame:
    """
    移除没有数据或数据不足的列
    
    Args:
        df: 原始DataFrame
        min_data_ratio: 最小数据比例阈值 (0.0-1.0)，低于此比例的列将被移除
        
    Returns:
        pd.DataFrame: 清理后的DataFrame
    """
    print(f"\n正在移除数据比例低于 {min_data_ratio*100:.1f}% 的列...")
    
    original_columns = len(df.columns)
    total_rows = len(df)
    
    # 计算每列的非空比例
    columns_to_keep = []
    removed_columns = []
    
    for col in df.columns:
        non_null_ratio = df[col].count() / total_rows
        if non_null_ratio > min_data_ratio:
            columns_to_keep.append(col)
        else:
            removed_columns.append((col, non_null_ratio * 100))
    
    # 创建清理后的DataFrame
    df_cleaned = df[columns_to_keep].copy()
    
    print(f"✅ 列清理完成:")
    print(f"   原始列数: {original_columns}")
    print(f"   保留列数: {len(columns_to_keep)}")
    print(f"   移除列数: {len(removed_columns)}")
    
    if removed_columns:
        print(f"\n移除的列:")
        for col, ratio in removed_columns:
            print(f"   {col} (数据比例: {ratio:.1f}%)")
    
    return df_cleaned


def preprocess_stock_data(df: pd.DataFrame, min_trading_days: int = 90) -> pd.DataFrame:
    """
    预处理股票数据
    
    Args:
        df: 原始DataFrame
        min_trading_days: 最小交易天数
    Returns:
        pd.DataFrame: 预处理后的DataFrame
    """
    print("正在预处理数据...")
    
    # 移除完全没有数据的列
    df = remove_empty_columns(df, min_data_ratio=0.0)
    
    # 检查交易天数
    if len(df) < min_trading_days:
        print(f"❌ 交易天数不足: {len(df)} < {min_trading_days}")
        return pd.DataFrame()
    
    # 转换日期列
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # 转换数值列
    numeric_columns = ['price', 'open', 'close', 'last_close', 'high', 'low', 
                      'avg_price', 'chg_amt', 'chg_pct', 'volume', 'amount', 'turnover_rate', 
                      'pb', 'eps', 'market_cap', 'float_market_cap']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 处理涨跌幅百分比
    if 'chg_pct' in df.columns:
        # 如果涨跌幅已经是百分比形式，转换为小数
        if df['chg_pct'].max() > 100 or df['chg_pct'].min() < -100:
            df['chg_pct'] = df['chg_pct'] / 100
    
    print("✅ kline数据预处理完成")
    return df


def read_stock_data_csv(file_path: str = None, min_trading_days: int = 90) -> pd.DataFrame:
    """
    读取股票数据CSV文件到pandas DataFrame
    
    Args:
        file_path: CSV文件路径，如果为None则使用默认路径
        min_trading_days: 最小交易天数
        
    Returns:
        pd.DataFrame: 包含股票数据的DataFrame
    """
    if file_path is None:
        # 默认文件路径
        default_file = r"db\all_stocks_180days.csv"
        file_path = default_file
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return pd.DataFrame()
        
        print(f"正在读取CSV文件: {file_path}")

        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        stocks = stock_cache.get_all_stocks()
        # 提取出stocks中ts_code列表
        ts_codes = [stock.ts_code for stock in stocks]
        # 从df中排除ts_codes中不存在的代码
        df = df[df['code'].isin(ts_codes)]
        
        # 数据预处理
        df = preprocess_stock_data(df, min_trading_days=min_trading_days)
        print(f"   股票代码数量: {df['code'].nunique()}")
        print(f"   交易日期范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
        
        # 检查是否有缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\n缺失值统计:")
            print(missing_values[missing_values > 0])
        # 将trade_date转换为date字符串类型，如果不是str类型才转换
        if not pd.api.types.is_string_dtype(df['trade_date']):
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        return df
        
    except Exception as e:
        print(f"❌ 读取CSV文件失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ==================== K线数据服务类 ====================

class KLineDataService:
    """K线数据服务 - 提供带缓存的股票技术指标数据"""
    
    def __init__(
        self, 
        csv_path: str,
        min_trading_days: int = 90,
        use_cache: bool = True,
        signal_processor: Optional[SignalProcessorType] = None
    ):
        """
        初始化 K 线数据服务
        
        Args:
            csv_path: CSV 数据文件路径
            min_trading_days: 最小交易天数过滤
            use_cache: 是否使用 Redis 缓存
            signal_processor: 技术指标处理函数，签名为 (df: pd.DataFrame, chart_days: int) -> pd.DataFrame
                             如果为 None，则使用默认的 compute_signals_for_code 函数
        """
        self.csv_path = csv_path
        self.min_trading_days = min_trading_days
        self.use_cache = use_cache
        self.signal_processor: SignalProcessorType = signal_processor or compute_signals_for_code
        
        # 初始化 Redis 客户端
        self.redis = get_redis_client() if use_cache else None
        
        # 加载原始 CSV 数据
        self.df = read_stock_data_csv(csv_path, min_trading_days=min_trading_days)
        self.df = self.df.sort_values(["code", "trade_date"])
        self.df["trade_date"] = self.df["trade_date"].astype(str)
        
        # 获取数据最新日期
        self.max_trade_date = self.df["trade_date"].max()
        
        # 股票代码列表
        self.codes = self.df["code"].unique().tolist()
        
        # 内存索引缓存：{股票代码: redis_key}，只保存映射关系，实际数据从 Redis 获取
        self._cache_index: Dict[str, str] = {}
        
        # 缓存统计
        self.cache_hit_count = 0
        self.cache_miss_count = 0
    
    def _get_cache_key(self, code: str) -> str:
        """生成 Redis 缓存键"""
        return f"{REDIS_CACHE_KEY_PREFIX}{code}"
    
    def _load_from_redis(self, code: str, required_latest_date: str) -> Optional[pd.DataFrame]:
        """
        从 Redis 缓存加载技术指标数据（带日期验证）
        
        Args:
            code: 股票代码
            required_latest_date: 要求的最新日期，如果缓存中的最新日期 >= 此日期则有效
            
        Returns:
            DataFrame 或 None（缓存未命中或已过期）
        """
        if not self.use_cache or self.redis is None:
            return None
        
        try:
            cache_key = self._get_cache_key(code)
            cached_json = self.redis.get(cache_key)
            if cached_json:
                df = pd.read_json(StringIO(cached_json), orient='records')
                
                # 日期验证：检查缓存数据的最新日期是否满足要求
                if required_latest_date and "trade_date" in df.columns:
                    # 确保 trade_date 是字符串格式进行比较
                    df["trade_date"] = df["trade_date"].astype(str)
                    cached_latest_date = df["trade_date"].max()
                    
                    # 如果缓存中的最新日期 < 要求的日期，则缓存无效
                    if cached_latest_date < str(required_latest_date):
                        return None
                
                return df
        except Exception:
            # 缓存读取失败，静默处理
            pass
        
        return None
    
    def _save_to_redis(self, code: str, df: pd.DataFrame) -> bool:
        """
        将技术指标数据保存到 Redis 缓存
        
        Args:
            code: 股票代码
            df: 技术指标 DataFrame
            
        Returns:
            是否保存成功
        """
        if not self.use_cache or self.redis is None:
            return False
        
        try:
            cache_key = self._get_cache_key(code)
            json_str = df.to_json(orient='records', date_format='iso')
            self.redis.set(cache_key, json_str, ex=REDIS_CACHE_EXPIRE_SECONDS)
            return True
        except Exception:
            # 缓存写入失败，静默处理
            return False
    
    def get_stock_data(
        self, 
        code: str, 
        chart_days: int = None,
        signal_processor: Optional[SignalProcessorType] = None,
        force_update: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        获取单只股票的 K 线数据（带技术指标）
        
        缓存策略：
        - 内存索引缓存只保存 {code: redis_key} 映射
        - 实际数据存储在 Redis 中
        - 获取数据时先检查索引，再从 Redis 读取
        
        Args:
            code: 股票代码
            chart_days: 图表天数，如果为 None 则返回全部数据
            signal_processor: 技术指标处理函数，签名为 (df: pd.DataFrame, chart_days: int) -> pd.DataFrame
                             如果为 None，则使用实例默认的处理函数
            force_update: 是否强制更新缓存，默认 False。为 True 时跳过缓存检查，重新计算并更新缓存
            
        Returns:
            包含技术指标的 DataFrame 或 None
        """
        code_str = str(code)
        
        # 确定使用的处理函数
        processor: SignalProcessorType = signal_processor or self.signal_processor
        
        # 获取原始数据（用于获取最新日期）
        sub = self.df[self.df["code"] == code].reset_index(drop=True)
        if len(sub) == 0:
            return None
        
        # 获取当前股票的最新日期
        required_latest_date = sub["trade_date"].max()
        
        # 如果不是强制更新，则尝试从缓存获取
        if not force_update:
            # 检查内存索引缓存，如果存在则从 Redis 获取数据
            if code_str in self._cache_index and self.use_cache:
                cached_data = self._load_from_redis(code_str, required_latest_date)
                if cached_data is not None:
                    self.cache_hit_count += 1
                    if chart_days:
                        return cached_data.tail(chart_days).reset_index(drop=True)
                    return cached_data
                else:
                    # Redis 缓存已过期或无效，移除内存索引
                    del self._cache_index[code_str]
            
            # 尝试从 Redis 缓存加载（内存索引不存在时也尝试一下）
            if self.use_cache and code_str not in self._cache_index:
                cached_data = self._load_from_redis(code_str, required_latest_date)
                if cached_data is not None:
                    # 添加到内存索引
                    self._cache_index[code_str] = self._get_cache_key(code_str)
                    self.cache_hit_count += 1
                    if chart_days:
                        return cached_data.tail(chart_days).reset_index(drop=True)
                    return cached_data
        
        # 使用处理函数计算技术指标
        sub = processor(sub, chart_days=len(sub))
        
        # 保存到 Redis 缓存，并更新内存索引
        if self.use_cache:
            if self._save_to_redis(code_str, sub):
                self._cache_index[code_str] = self._get_cache_key(code_str)
        
        self.cache_miss_count += 1
        
        if chart_days:
            return sub.tail(chart_days).reset_index(drop=True)
        return sub
    
    def get_all_stock_data(
        self, 
        show_progress: bool = True,
        signal_processor: Optional[SignalProcessorType] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        获取所有股票的 K 线数据（带技术指标）
        
        注意：此方法会从 Redis 加载所有数据到内存，数据量大时请谨慎使用
        
        Args:
            show_progress: 是否显示进度
            signal_processor: 技术指标处理函数，签名为 (df: pd.DataFrame, chart_days: int) -> pd.DataFrame
                             如果为 None，则使用实例默认的处理函数
            
        Returns:
            {code: DataFrame} 字典
        """
        total_codes = len(self.codes)
        result: Dict[str, pd.DataFrame] = {}
        
        for idx, code in enumerate(self.codes, 1):
            if show_progress and (idx % 500 == 0 or idx == 1):
                print(f"正在处理技术指标: {idx}/{total_codes}")
            
            df = self.get_stock_data(str(code), signal_processor=signal_processor)
            if df is not None:
                result[str(code)] = df
        
        if show_progress:
            print(f"加载完成，共 {len(result)} 只股票")
            if self.use_cache:
                print(f"  缓存命中: {self.cache_hit_count}, 缓存未命中/过期: {self.cache_miss_count}")
                print(f"  内存索引数量: {len(self._cache_index)}")
        
        return result
    
    def clear_cache(self, code: str = None) -> int:
        """
        清除缓存（同时清除内存索引和 Redis 数据）
        
        Args:
            code: 股票代码，如果为 None 则清除所有缓存
            
        Returns:
            清除的 Redis 缓存数量
        """
        # 清除内存索引缓存
        if code:
            if code in self._cache_index:
                del self._cache_index[code]
        else:
            self._cache_index.clear()
        
        # 清除 Redis 缓存
        if not self.use_cache or self.redis is None:
            return 0
        
        try:
            if code:
                cache_key = self._get_cache_key(code)
                return self.redis.delete(cache_key)
            else:
                # 清除所有技术指标缓存
                pattern = f"{REDIS_CACHE_KEY_PREFIX}*"
                keys = self.redis.keys(pattern)
                if keys:
                    return self.redis.delete(*keys)
                return 0
        except Exception as e:
            print(f"清除缓存失败: {e}")
            return 0
    
    def reset_stats(self):
        """重置缓存统计"""
        self.cache_hit_count = 0
        self.cache_miss_count = 0


# ==================== 数据库查询服务类 ====================

class StockQuoteDBService:
    """
    股票报价数据库查询服务 - 从 t_stock_quote 表查询数据
    
    使用 pymysql 连接数据库，支持按股票代码和日期范围查询
    
    使用示例:
        service = StockQuoteDBService()
        
        # 查询单只股票数据
        df = service.get_stock_data("000001.SZ", "2024-01-01", "2024-12-31")
        
        # 查询多只股票数据
        df = service.get_stocks_data(["000001.SZ", "000002.SZ"], "2024-01-01", "2024-12-31")
        
        # 查询指定日期的所有股票
        df = service.get_all_stocks_by_date("2024-01-15")
    """
    
    # 数据库字段到 DataFrame 列名的映射
    COLUMN_MAPPING = {
        'ts_code': 'code',
        'trade_date': 'trade_date',
        'current_price': 'price',
        'open_price': 'open',
        'close_price': 'close',
        'last_close': 'last_close',
        'high_price': 'high',
        'low_price': 'low',
        'avg_price': 'avg_price',
        'chg': 'chg_amt',
        'percent': 'chg_pct',
        'volume': 'volume',
        'amount': 'amount',
        'turnover_rate': 'turnover_rate',
        'pb': 'pb',
        'eps': 'eps',
        'market_capital': 'market_cap',
        'float_market_capital': 'float_market_cap',
        'amplitude': 'amplitude',
        'volume_ratio': 'volume_ratio',
        'buy': 'buy',
        'sell': 'sell',
        'float_shares': 'float_shares',
        'total_shares': 'total_shares',
    }
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        database: str = None,
        charset: str = None,
        use_cache: bool = True
    ):
        """
        初始化数据库查询服务
        
        Args:
            host: 数据库主机地址，默认从配置读取
            port: 数据库端口，默认从配置读取
            user: 数据库用户名，默认从配置读取
            password: 数据库密码，默认从配置读取
            database: 数据库名，默认从配置读取
            charset: 字符集，默认从配置读取
            use_cache: 是否使用 Redis 缓存
        """
        # 使用传入的参数或从配置读取
        self.db_config = {
            'host': host or db_config.get('host', 'localhost'),
            'port': port or db_config.get('port', 3306),
            'user': user or db_config.get('user', 'root'),
            'password': password or db_config.get('password', 'root'),
            'database': database or db_config.get('database', 'zstock_db'),
            'charset': charset or db_config.get('charset', 'utf8mb4'),
        }
        
        self.use_cache = use_cache
        self.redis = get_redis_client() if use_cache else None
        
        # 缓存统计
        self.cache_hit_count = 0
        self.cache_miss_count = 0
    
    def _get_connection(self) -> pymysql.Connection:
        """获取数据库连接"""
        return pymysql.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            database=self.db_config['database'],
            charset=self.db_config['charset'],
            cursorclass=DictCursor
        )
    
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """将数据库列名映射为标准列名"""
        return df.rename(columns=self.COLUMN_MAPPING)
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理查询结果 DataFrame
        
        包括列名映射、数据类型转换等
        """
        if df.empty:
            return df
        
        # 重命名列
        df = self._rename_columns(df)
        
        # 将 Decimal 类型转换为 float（数据库中的 decimal 字段）
        numeric_columns = [
            'price', 'open', 'close', 'last_close', 'high', 'low', 'avg_price',
            'chg_amt', 'chg_pct', 'volume', 'amount', 'turnover_rate',
            'pb', 'eps', 'market_cap', 'float_market_cap',
            'amplitude', 'volume_ratio', 'buy', 'sell', 'float_shares', 'total_shares'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 确保 trade_date 为字符串格式
        if 'trade_date' in df.columns:
            df['trade_date'] = df['trade_date'].astype(str)
        
        # 涨跌幅转换（如果需要）
        if 'chg_pct' in df.columns:
            # 数据库中存储的是百分比值，转换为小数
            df['chg_pct'] = df['chg_pct'] / 100
        
        # 按日期排序
        if 'trade_date' in df.columns:
            df = df.sort_values('trade_date').reset_index(drop=True)
        
        return df
    
    def _get_cache_key(self, ts_code: str, start_date: str = None, end_date: str = None) -> str:
        """生成缓存键"""
        return f"{REDIS_CACHE_KEY_PREFIX}db:{ts_code}:{start_date or ''}:{end_date or ''}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从 Redis 缓存加载数据"""
        if not self.use_cache or self.redis is None:
            return None
        
        try:
            cached_json = self.redis.get(cache_key)
            if cached_json:
                df = pd.read_json(StringIO(cached_json), orient='records')
                return df
        except Exception:
            pass
        
        return None
    
    def _save_to_cache(self, cache_key: str, df: pd.DataFrame) -> bool:
        """保存数据到 Redis 缓存"""
        if not self.use_cache or self.redis is None:
            return False
        
        try:
            json_str = df.to_json(orient='records', date_format='iso')
            self.redis.set(cache_key, json_str, ex=REDIS_CACHE_EXPIRE_SECONDS)
            return True
        except Exception:
            return False
    
    def get_stock_data(
        self,
        code: str,
        start_date: str = None,
        end_date: str = None,
        with_tech_signals: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        查询单只股票的报价数据
        
        Args:
            code: 股票代码（如 "000001.SZ"）
            start_date: 开始日期（如 "2024-01-01"），为 None 则不限制
            end_date: 结束日期（如 "2024-12-31"），为 None 则不限制
            with_tech_signals: 是否计算技术指标
            
        Returns:
            包含报价数据的 DataFrame，查询失败返回 None
        """
        # 构建 SQL 查询
        sql = "SELECT * FROM t_stock_quote WHERE ts_code = %s"
        params = [code]
        
        if start_date:
            sql += " AND trade_date >= %s"
            params.append(start_date)
        
        if end_date:
            sql += " AND trade_date <= %s"
            params.append(end_date)
        
        sql += " ORDER BY trade_date ASC"
        
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    results = cursor.fetchall()
                    
                    if not results:
                        return None
                    
                    df = pd.DataFrame(results)
                    df = self._process_dataframe(df)
                    
                    if with_tech_signals:
                        df = compute_signals_for_code(df, chart_days=len(df))
                    
                    return df
                    
        except Exception as e:
            print(f"❌ 查询股票 {code} 数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_stocks_data(
        self,
        ts_codes: List[str],
        start_date: str = None,
        end_date: str = None,
        with_tech_signals: bool = False,
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        查询多只股票的报价数据
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            with_tech_signals: 是否计算技术指标
            show_progress: 是否显示进度
            
        Returns:
            {ts_code: DataFrame} 字典
        """
        result = {}
        total = len(ts_codes)
        
        for idx, ts_code in enumerate(ts_codes, 1):
            if show_progress and (idx % 100 == 0 or idx == 1):
                print(f"正在查询: {idx}/{total}")
            
            df = self.get_stock_data(code=ts_code, start_date=start_date, end_date=end_date, with_tech_signals=with_tech_signals)
            if df is not None:
                result[ts_code] = df
        
        if show_progress:
            print(f"查询完成，共 {len(result)} 只股票")
            if self.use_cache:
                print(f"  缓存命中: {self.cache_hit_count}, 缓存未命中: {self.cache_miss_count}")
        
        return result
    
    def get_stocks_data_batch(
        self,
        ts_codes: List[str],
        start_date: str = None,
        end_date: str = None,
        with_tech_signals: bool = False
    ) -> pd.DataFrame:
        """
        批量查询多只股票的报价数据（使用 IN 查询）
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            with_tech_signals: 是否计算技术指标
            
        Returns:
            包含所有股票数据的 DataFrame
        """
        if not ts_codes:
            return pd.DataFrame()
        
        # 构建 IN 查询
        placeholders = ', '.join(['%s'] * len(ts_codes))
        sql = f"SELECT * FROM t_stock_quote WHERE ts_code IN ({placeholders})"
        params = list(ts_codes)
        
        if start_date:
            sql += " AND trade_date >= %s"
            params.append(start_date)
        
        if end_date:
            sql += " AND trade_date <= %s"
            params.append(end_date)
        
        sql += " ORDER BY ts_code, trade_date ASC"
        
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    results = cursor.fetchall()
                    
                    if not results:
                        return pd.DataFrame()
                    
                    df = pd.DataFrame(results)
                    df = self._process_dataframe(df)
                    
                    if with_tech_signals:
                        # 按股票分组计算技术指标
                        dfs = []
                        for code in df['code'].unique():
                            sub_df = df[df['code'] == code].copy()
                            sub_df = compute_signals_for_code(sub_df, chart_days=len(sub_df))
                            dfs.append(sub_df)
                        df = pd.concat(dfs, ignore_index=True)
                    
                    return df
                    
        except Exception as e:
            print(f"❌ 批量查询股票数据失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_all_stocks_by_date(
        self,
        trade_date: str
    ) -> pd.DataFrame:
        """
        查询指定日期的所有股票数据
        
        Args:
            trade_date: 交易日期（如 "2024-01-15"）
            
        Returns:
            包含所有股票数据的 DataFrame
        """
        sql = "SELECT * FROM t_stock_quote WHERE trade_date = %s ORDER BY ts_code"
        
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, [trade_date])
                    results = cursor.fetchall()
                    
                    if not results:
                        return pd.DataFrame()
                    
                    df = pd.DataFrame(results)
                    df = self._process_dataframe(df)
                    return df
                    
        except Exception as e:
            print(f"❌ 查询日期 {trade_date} 数据失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_date_range(self, ts_code: str = None) -> Tuple[Optional[str], Optional[str]]:
        """
        获取数据的日期范围
        
        Args:
            ts_code: 股票代码，为 None 则查询所有数据的日期范围
            
        Returns:
            (min_date, max_date) 元组
        """
        if ts_code:
            sql = "SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date FROM t_stock_quote WHERE ts_code = %s"
            params = [ts_code]
        else:
            sql = "SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date FROM t_stock_quote"
            params = []
        
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    result = cursor.fetchone()
                    
                    if result:
                        return result['min_date'], result['max_date']
                    return None, None
                    
        except Exception as e:
            print(f"❌ 查询日期范围失败: {e}")
            return None, None
    
    def get_stock_count(self, trade_date: str = None) -> int:
        """
        获取股票数量
        
        Args:
            trade_date: 交易日期，为 None 则统计所有股票
            
        Returns:
            股票数量
        """
        if trade_date:
            sql = "SELECT COUNT(DISTINCT ts_code) as cnt FROM t_stock_quote WHERE trade_date = %s"
            params = [trade_date]
        else:
            sql = "SELECT COUNT(DISTINCT ts_code) as cnt FROM t_stock_quote"
            params = []
        
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    result = cursor.fetchone()
                    return result['cnt'] if result else 0
                    
        except Exception as e:
            print(f"❌ 查询股票数量失败: {e}")
            return 0
    
    def reset_stats(self):
        """重置缓存统计"""
        self.cache_hit_count = 0
        self.cache_miss_count = 0


# ==================== 便捷函数 ====================

_default_service: Optional[KLineDataService] = None


def get_kline_service(
    csv_path: str = None,
    min_trading_days: int = 90,
    use_cache: bool = True
) -> KLineDataService:
    """
    获取 K 线数据服务实例（单例模式）
    
    Args:
        csv_path: CSV 数据文件路径（首次调用必须提供）
        min_trading_days: 最小交易天数过滤
        use_cache: 是否使用 Redis 缓存
        
    Returns:
        KLineDataService 实例
    """
    global _default_service
    
    if _default_service is None:
        if csv_path is None:
            raise ValueError("首次调用必须提供 csv_path 参数")
        _default_service = KLineDataService(csv_path, min_trading_days, use_cache)
    
    return _default_service


def reset_kline_service():
    """重置全局服务实例"""
    global _default_service
    _default_service = None


# 数据库服务单例
_default_db_service: Optional[StockQuoteDBService] = None


def get_stock_quote_db_service(
    use_cache: bool = True,
    **db_kwargs
) -> StockQuoteDBService:
    """
    获取股票报价数据库查询服务实例（单例模式）
    
    Args:
        use_cache: 是否使用 Redis 缓存
        **db_kwargs: 数据库连接参数（host, port, user, password, database, charset）
        
    Returns:
        StockQuoteDBService 实例
    """
    global _default_db_service
    
    if _default_db_service is None:
        _default_db_service = StockQuoteDBService(use_cache=use_cache, **db_kwargs)
    
    return _default_db_service


def reset_stock_quote_db_service():
    """重置数据库服务实例"""
    global _default_db_service
    _default_db_service = None


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试用例
    CSV_PATH = r"E:\company\cursor_py_work\finance\db\all_stocks_180days.csv"
    
    print("=" * 60)
    print("测试 KLineDataService（CSV 数据源）")
    print("=" * 60)
    
    # 创建服务
    service = KLineDataService(CSV_PATH, use_cache=True)
    print(f"数据最新日期: {service.max_trade_date}")
    print(f"股票数量: {len(service.codes)}")
    
    # 测试获取单只股票数据
    print("\n--- 测试获取单只股票数据 ---")
    df = service.get_stock_data("000001.SZ")
    if df is not None:
        print(f"000001.SZ 数据行数: {len(df)}")
        print(f"列名: {df.columns.tolist()}")
    
    # 测试缓存统计
    print(f"\n缓存统计 - 命中: {service.cache_hit_count}, 未命中: {service.cache_miss_count}")
    
    # 再次获取同一只股票（应该命中内存缓存）
    df2 = service.get_stock_data("000001.SZ")
    print(f"再次获取后 - 命中: {service.cache_hit_count}, 未命中: {service.cache_miss_count}")
    
    print("\n" + "=" * 60)
    print("测试 StockQuoteDBService（数据库数据源）")
    print("=" * 60)
    
    # 创建数据库服务
    db_service = StockQuoteDBService(use_cache=True)
    
    # 查询日期范围
    min_date, max_date = db_service.get_date_range()
    print(f"数据日期范围: {min_date} ~ {max_date}")
    
    # 查询股票数量
    stock_count = db_service.get_stock_count()
    print(f"股票总数量: {stock_count}")
    
    # 测试查询单只股票数据
    print("\n--- 测试查询单只股票数据 ---")
    db_df = db_service.get_stock_data("000001.SZ", start_date="2024-01-01", end_date="2024-12-31")
    if db_df is not None:
        print(f"000001.SZ 数据行数: {len(db_df)}")
        print(f"列名: {db_df.columns.tolist()}")
        print(f"前3行数据:\n{db_df.head(3)}")
    else:
        print("未查询到数据")
    
    # 测试带技术指标查询
    print("\n--- 测试带技术指标查询 ---")
    db_df_with_signals = db_service.get_stock_data(
        "000001.SZ", 
        start_date="2024-01-01", 
        end_date="2024-12-31",
        with_tech_signals=True
    )
    if db_df_with_signals is not None:
        print(f"带技术指标列名: {db_df_with_signals.columns.tolist()}")
    
    # 缓存统计
    print(f"\n缓存统计 - 命中: {db_service.cache_hit_count}, 未命中: {db_service.cache_miss_count}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
