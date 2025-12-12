import os
import sys
import pandas as pd
import pymysql
from datetime import datetime
from typing import List, Optional, Dict, Any
import threading
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foundation.app_env import db_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TradingDayCache:
    """
    交易日缓存服务
    从t_trading_day_record表中读取交易日数据并缓存，提供高效的交易日查询服务
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TradingDayCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化缓存服务"""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.db_config = db_config
        self.trading_dates = []  # 缓存所有交易日
        self.trading_dates_set = set()  # 用于快速查找
        self.cache_time = None  # 缓存时间
        self.cache_duration = 3600  # 缓存有效期（秒），默认1小时
        self._lock = threading.RLock()  # 读写锁
        
        # 初始化时加载数据
        self._load_trading_dates()
    
    def _get_db_connection(self):
        """
        获取数据库连接
        
        返回:
        pymysql.Connection: 数据库连接对象
        """
        try:
            connection = pymysql.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                charset=self.db_config['charset'],
                autocommit=True,
                cursorclass=pymysql.cursors.DictCursor
            )
            return connection
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return None
    
    def _load_trading_dates(self):
        """
        从数据库加载交易日数据到缓存
        """
        with self._lock:
            try:
                connection = self._get_db_connection()
                if not connection:
                    logger.error("无法建立数据库连接")
                    return
                
                # 查询所有有效的交易日
                sql = """
                SELECT trade_date 
                FROM t_trading_day_record 
                WHERE valid = 1 
                ORDER BY trade_date ASC
                """
                
                with connection.cursor() as cursor:
                    cursor.execute(sql)
                    results = cursor.fetchall()
                
                # 更新缓存
                self.trading_dates = [row['trade_date'] for row in results]
                self.trading_dates_set = set(self.trading_dates)
                self.cache_time = datetime.now()
                
                logger.info(f"✅ 成功加载 {len(self.trading_dates)} 个交易日到缓存")
                
            except Exception as e:
                logger.error(f"❌ 加载交易日数据失败: {e}")
            finally:
                if connection:
                    connection.close()
    
    def _is_cache_valid(self) -> bool:
        """
        检查缓存是否有效
        
        返回:
        bool: 缓存是否有效
        """
        if not self.cache_time or not self.trading_dates:
            return False
        
        # 检查缓存是否过期
        if (datetime.now() - self.cache_time).seconds > self.cache_duration:
            return False
        
        return True
    
    def _ensure_cache_valid(self):
        """
        确保缓存有效，如果无效则重新加载
        """
        if not self._is_cache_valid():
            logger.info("缓存已过期，重新加载交易日数据...")
            self._load_trading_dates()
    
    def get_trading_dates(self, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None, 
                         valid_only: bool = True) -> List[str]:
        """
        获取交易日列表
        
        参数:
        start_date: 开始日期 (YYYY-MM-DD)，可选
        end_date: 结束日期 (YYYY-MM-DD)，可选
        valid_only: 是否只返回有效的交易日，默认True
        
        返回:
        List[str]: 交易日列表，格式为YYYY-MM-DD
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.trading_dates:
                logger.warning("交易日缓存为空")
                return []
            
            # 复制原始数据
            result_dates = self.trading_dates.copy()
            
            # 过滤日期范围
            if start_date:
                start_dt = pd.to_datetime(start_date)
                result_dates = [date for date in result_dates if pd.to_datetime(date) >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                result_dates = [date for date in result_dates if pd.to_datetime(date) <= end_dt]
            
            return result_dates
    
    def get_trading_dates_range(self, start_date: str, end_date: str, 
                               valid_only: bool = True) -> List[str]:
        """
        获取指定日期范围内的交易日
        
        参数:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        valid_only: 是否只返回有效的交易日，默认True
        
        返回:
        List[str]: 交易日列表
        """
        return self.get_trading_dates(start_date, end_date, valid_only)
    
    def get_latest_trading_date(self, valid_only: bool = True) -> Optional[str]:
        """
        获取最新的交易日
        
        参数:
        valid_only: 是否只返回有效的交易日，默认True
        
        返回:
        Optional[str]: 最新交易日，格式为YYYY-MM-DD
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.trading_dates:
                logger.warning("交易日缓存为空")
                return None
            # 早于或等于今天的交易日
            today = datetime.now().strftime('%Y-%m-%d')
            latest_date = self.trading_dates[-1]
            if pd.to_datetime(latest_date) > pd.to_datetime(today):
                latest_date = today
            logger.info(f"✅ 最新交易日: {latest_date}")
            return latest_date
    
    def is_trading_day(self, date: str) -> bool:
        """
        判断指定日期是否为交易日
        
        参数:
        date: 日期 (YYYY-MM-DD)
        
        返回:
        bool: 是否为交易日
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.trading_dates_set:
                logger.warning("交易日缓存为空")
                return False
            
            return date in self.trading_dates_set
    
    def is_trading_time(self):
        now = datetime.now()
        start_morning = now.replace(hour=9, minute=25, second=0, microsecond=0)
        end_morning = now.replace(hour=11, minute=32, second=0, microsecond=0)
        start_afternoon = now.replace(hour=13, minute=0, second=0, microsecond=0)
        end_afternoon = now.replace(hour=15, minute=5, second=0, microsecond=0)

        if start_morning <= now <= end_morning or start_afternoon <= now <= end_afternoon:
            return True
        else:
            return False
    
    def is_hk_trading_time(self):
        now = datetime.now()
        start_morning = now.replace(hour=9, minute=25, second=0, microsecond=0)
        end_morning = now.replace(hour=12, minute=1, second=0, microsecond=0)
        start_afternoon = now.replace(hour=13, minute=0, second=0, microsecond=0)
        end_afternoon = now.replace(hour=16, minute=5, second=0, microsecond=0)

        if start_morning <= now <= end_morning or start_afternoon <= now <= end_afternoon:
            return True
        else:
            return False
    
    def get_next_trading_day(self, date: str, valid_only: bool = True) -> Optional[str]:
        """
        获取指定日期后的下一个交易日
        
        参数:
        date: 指定日期 (YYYY-MM-DD)
        valid_only: 是否只返回有效的交易日，默认True
        
        返回:
        Optional[str]: 下一个交易日，格式为YYYY-MM-DD
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.trading_dates:
                logger.warning("交易日缓存为空")
                return None
            
            target_date = pd.to_datetime(date)
            
            for trading_date in self.trading_dates:
                if pd.to_datetime(trading_date) > target_date:
                    logger.info(f"✅ {date} 的下一个交易日: {trading_date}")
                    return trading_date
            
            logger.warning(f"未找到 {date} 之后的交易日")
            return None
    
    def get_previous_trading_day(self, date: str, valid_only: bool = True) -> Optional[str]:
        """
        获取指定日期前的上一个交易日
        
        参数:
        date: 指定日期 (YYYY-MM-DD)
        valid_only: 是否只返回有效的交易日，默认True
        
        返回:
        Optional[str]: 上一个交易日，格式为YYYY-MM-DD
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.trading_dates:
                logger.warning("交易日缓存为空")
                return None
            
            target_date = pd.to_datetime(date)
            previous_date = None
            
            for trading_date in self.trading_dates:
                if pd.to_datetime(trading_date) >= target_date:
                    break
                previous_date = trading_date
            
            if previous_date:
                logger.info(f"✅ {date} 的上一个交易日: {previous_date}")
            else:
                logger.warning(f"未找到 {date} 之前的交易日")
            
            return previous_date
    
    def get_trading_days_count(self, start_date: str, end_date: str) -> int:
        """
        获取指定日期范围内的交易日数量
        
        参数:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        
        返回:
        int: 交易日数量
        """
        trading_dates = self.get_trading_dates_range(start_date, end_date)
        return len(trading_dates)
    
    def get_trading_days_by_count(self, end_date: str, count: int) -> List[str]:
        """
        从指定日期开始获取指定数量的交易日
        
        参数:
        end_date: 结束日期 (YYYY-MM-DD)
        count: 交易日数量
        
        返回:
        List[str]: 交易日列表
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.trading_dates:
                logger.warning("交易日缓存为空")
                return []
            
            end_dt = pd.to_datetime(end_date)
            result_dates = []
            reversed_dates = self.trading_dates.copy();
            reversed_dates.reverse()
            for trading_date in reversed_dates:
                if pd.to_datetime(trading_date) <= end_dt:
                    result_dates.append(trading_date)
                    if len(result_dates) >= count:
                        break
            
            return result_dates
    
    def refresh_cache(self):
        """
        手动刷新缓存
        """
        logger.info("手动刷新交易日缓存...")
        self._load_trading_dates()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        返回:
        Dict[str, Any]: 缓存信息
        """
        with self._lock:
            return {
                'trading_dates_count': len(self.trading_dates),
                'cache_time': self.cache_time,
                'cache_duration': self.cache_duration,
                'is_valid': self._is_cache_valid(),
                'date_range': {
                    'start': self.trading_dates[0] if self.trading_dates else None,
                    'end': self.trading_dates[-1] if self.trading_dates else None
                }
            }
    
    def set_cache_duration(self, duration: int):
        """
        设置缓存有效期
        
        参数:
        duration: 缓存有效期（秒）
        """
        with self._lock:
            self.cache_duration = duration
            logger.info(f"缓存有效期设置为 {duration} 秒")


# 全局单例实例
trading_day_cache = TradingDayCache()

# 便捷函数
def get_trading_dates(start_date: Optional[str] = None, 
                     end_date: Optional[str] = None, 
                     valid_only: bool = True) -> List[str]:
    """
    获取交易日列表的便捷函数
    
    参数:
    start_date: 开始日期 (YYYY-MM-DD)，可选
    end_date: 结束日期 (YYYY-MM-DD)，可选
    valid_only: 是否只返回有效的交易日，默认True
    
    返回:
    List[str]: 交易日列表，格式为YYYY-MM-DD
    """
    return trading_day_cache.get_trading_dates(start_date, end_date, valid_only)

def get_trading_dates_range(start_date: str, end_date: str, 
                           valid_only: bool = True) -> List[str]:
    """
    获取指定日期范围内的交易日的便捷函数
    
    参数:
    start_date: 开始日期 (YYYY-MM-DD)
    end_date: 结束日期 (YYYY-MM-DD)
    valid_only: 是否只返回有效的交易日，默认True
    
    返回:
    List[str]: 交易日列表
    """
    return trading_day_cache.get_trading_dates_range(start_date, end_date, valid_only)

def get_latest_trading_date(valid_only: bool = True) -> Optional[str]:
    """
    获取最新交易日的便捷函数
    
    参数:
    valid_only: 是否只返回有效的交易日，默认True
    
    返回:
    Optional[str]: 最新交易日，格式为YYYY-MM-DD
    """
    return trading_day_cache.get_latest_trading_date(valid_only)

def is_trading_day(date: str) -> bool:
    """
    判断指定日期是否为交易日的便捷函数
    
    参数:
    date: 日期 (YYYY-MM-DD)
    
    返回:
    bool: 是否为交易日
    """
    return trading_day_cache.is_trading_day(date)

def get_next_trading_day(date: str, valid_only: bool = True) -> Optional[str]:
    """
    获取指定日期后的下一个交易日的便捷函数
    
    参数:
    date: 指定日期 (YYYY-MM-DD)
    valid_only: 是否只返回有效的交易日，默认True
    
    返回:
    Optional[str]: 下一个交易日，格式为YYYY-MM-DD
    """
    return trading_day_cache.get_next_trading_day(date, valid_only)

def get_previous_trading_day(date: str, valid_only: bool = True) -> Optional[str]:
    """
    获取指定日期前的上一个交易日的便捷函数
    
    参数:
    date: 指定日期 (YYYY-MM-DD)
    valid_only: 是否只返回有效的交易日，默认True
    
    返回:
    Optional[str]: 上一个交易日，格式为YYYY-MM-DD
    """
    return trading_day_cache.get_previous_trading_day(date, valid_only)

def get_trading_days_count(start_date: str, end_date: str) -> int:
    """
    获取指定日期范围内的交易日数量的便捷函数
    
    参数:
    start_date: 开始日期 (YYYY-MM-DD)
    end_date: 结束日期 (YYYY-MM-DD)
    
    返回:
    int: 交易日数量
    """
    return trading_day_cache.get_trading_days_count(start_date, end_date)

def get_trading_days_by_count(end_date: str, count: int) -> List[str]:
    """
    从指定日期开始获取指定数量的交易日的便捷函数
    
    参数:
    end_date: 结束日期 (YYYY-MM-DD)
    count: 交易日数量
    
    返回:
    List[str]: 交易日列表
    """
    return trading_day_cache.get_trading_days_by_count(end_date, count)

def refresh_trading_days_cache():
    """
    手动刷新交易日缓存的便捷函数
    """
    trading_day_cache.refresh_cache()

def get_trading_days_cache_info() -> Dict[str, Any]:
    """
    获取交易日缓存信息的便捷函数
    
    返回:
    Dict[str, Any]: 缓存信息
    """
    return trading_day_cache.get_cache_info()


# 使用示例
if __name__ == "__main__":
    # 测试缓存服务
    print("=" * 60)
    print("交易日缓存服务测试")
    print("=" * 60)
    
    # 获取缓存信息
    cache_info = get_trading_days_cache_info()
    print(f"缓存信息: {cache_info}")
    
    # 获取所有交易日
    all_dates = get_trading_dates()
    print(f"\n总交易日数: {len(all_dates)}")
    if all_dates:
        print(f"最早交易日: {all_dates[0]}")
        print(f"最新交易日: {all_dates[-1]}")
    
    # 获取指定范围的交易日
    dates_2024 = get_trading_dates_range("2024-01-01", "2024-12-31")
    print(f"\n2024年交易日数: {len(dates_2024)}")
    
    # 测试特定日期
    test_date = "2024-01-15"
    print(f"\n{test_date} 是否为交易日: {is_trading_day(test_date)}")
    
    # 获取下一个交易日
    next_day = get_next_trading_day("2024-01-15")
    print(f"{test_date} 的下一个交易日: {next_day}")
    
    # 获取上一个交易日
    prev_day = get_previous_trading_day("2024-01-15")
    print(f"{test_date} 的上一个交易日: {prev_day}")
    
    # 获取指定数量的交易日
    recent_days = get_trading_days_by_count("2024-01-01", 10)
    print(f"\n从2024-01-01开始的10个交易日: {recent_days}")

