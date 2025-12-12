import pymysql
from datetime import datetime
from typing import List, Optional, Dict, Any
import threading
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foundation.app_env import db_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class StockInfo:
    """股票信息数据类"""
    def __init__(self, ts_code: str, symbol: str, name: str, area: str, 
                 industry: str, fullname: str = None, enname: str = None,
                 market: str = None, exchange: str = None, list_status: str = None,
                 list_date: str = None, delist_date: str = None, is_hs: str = None):
        self.ts_code = ts_code
        self.symbol = symbol
        self.name = name
        self.area = area
        self.industry = industry
        self.fullname = fullname
        self.enname = enname
        self.market = market
        self.exchange = exchange
        self.list_status = list_status
        self.list_date = list_date
        self.delist_date = delist_date
        self.is_hs = is_hs
    
    def __repr__(self):
        return f"StockInfo(ts_code='{self.ts_code}', symbol='{self.symbol}', name='{self.name}')"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'ts_code': self.ts_code,
            'symbol': self.symbol,
            'name': self.name,
            'area': self.area,
            'industry': self.industry,
            'fullname': self.fullname,
            'enname': self.enname,
            'market': self.market,
            'exchange': self.exchange,
            'list_status': self.list_status,
            'list_date': self.list_date,
            'delist_date': self.delist_date,
            'is_hs': self.is_hs
        }

class StockCache:
    """
    股票代码缓存服务
    从t_stock表中读取股票数据并缓存，提供高效的股票查询服务
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(StockCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化缓存服务"""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.db_config = db_config
        self.stocks = {}  # 缓存所有股票信息，key为ts_code
        self.stocks_by_symbol = {}  # 按symbol索引
        self.stocks_by_name = {}  # 按name索引
        self.cache_time = None  # 缓存时间
        self.cache_duration = 3600  # 缓存有效期（秒），默认1小时
        self._lock = threading.RLock()  # 读写锁
        
        # 初始化时加载数据
        self._load_stocks()
    
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
    
    def _load_stocks(self):
        """
        从数据库加载股票数据到缓存
        """
        with self._lock:
            try:
                connection = self._get_db_connection()
                if not connection:
                    logger.error("无法建立数据库连接")
                    return
                
                # 查询所有股票信息
                sql = """
                SELECT ts_code, symbol, name, area, industry, fullname, enname,
                       market, exchange, list_status, list_date, delist_date, is_hs
                FROM t_stock where list_status='L'
                ORDER BY ts_code ASC
                """
                
                with connection.cursor() as cursor:
                    cursor.execute(sql)
                    results = cursor.fetchall()
                
                # 清空现有缓存
                self.stocks.clear()
                self.stocks_by_symbol.clear()
                self.stocks_by_name.clear()
                
                # 更新缓存
                for row in results:
                    stock = StockInfo(
                        ts_code=row['ts_code'],
                        symbol=row['symbol'],
                        name=row['name'],
                        area=row['area'],
                        industry=row['industry'],
                        fullname=row['fullname'],
                        enname=row['enname'],
                        market=row['market'],
                        exchange=row['exchange'],
                        list_status=row['list_status'],
                        list_date=str(row['list_date']) if row['list_date'] else None,
                        delist_date=str(row['delist_date']) if row['delist_date'] else None,
                        is_hs=row['is_hs']
                    )
                    
                    # 按ts_code索引
                    self.stocks[stock.ts_code] = stock
                    # 按symbol索引
                    self.stocks_by_symbol[stock.symbol] = stock
                    # 按name索引
                    self.stocks_by_name[stock.name] = stock
                
                self.cache_time = datetime.now()
                
                logger.info(f"✅ 成功加载 {len(self.stocks)} 只股票到缓存")
                
            except Exception as e:
                logger.error(f"❌ 加载股票数据失败: {e}")
            finally:
                if connection:
                    connection.close()
    
    def _is_cache_valid(self) -> bool:
        """
        检查缓存是否有效
        
        返回:
        bool: 缓存是否有效
        """
        if not self.cache_time or not self.stocks:
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
            logger.info("缓存已过期，重新加载股票数据...")
            self._load_stocks()
    
    def get_stock_by_ts_code(self, ts_code: str) -> Optional[StockInfo]:
        """
        根据ts_code获取股票信息
        
        参数:
        ts_code: 股票TS代码
        
        返回:
        Optional[StockInfo]: 股票信息对象，如果未找到返回None
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.stocks:
                logger.warning("股票缓存为空")
                return None
            
            stock = self.stocks.get(ts_code)
            if stock:
                logger.debug(f"✅ 找到股票: {stock.name} ({stock.ts_code})")
            else:
                logger.warning(f"未找到ts_code为 {ts_code} 的股票")
            
            return stock
    
    def get_stock_by_symbol(self, symbol: str) -> Optional[StockInfo]:
        """
        根据symbol获取股票信息
        
        参数:
        symbol: 股票代码
        
        返回:
        Optional[StockInfo]: 股票信息对象，如果未找到返回None
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.stocks_by_symbol:
                logger.warning("股票缓存为空")
                return None
            
            stock = self.stocks_by_symbol.get(symbol)
            if stock:
                logger.debug(f"✅ 找到股票: {stock.name} ({stock.symbol})")
            else:
                logger.warning(f"未找到symbol为 {symbol} 的股票")
            
            return stock
    
    def get_stock_by_name(self, name: str) -> Optional[StockInfo]:
        """
        根据股票名称获取股票信息
        
        参数:
        name: 股票名称
        
        返回:
        Optional[StockInfo]: 股票信息对象，如果未找到返回None
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.stocks_by_name:
                logger.warning("股票缓存为空")
                return None
            
            stock = self.stocks_by_name.get(name)
            if stock:
                logger.debug(f"✅ 找到股票: {stock.name} ({stock.ts_code})")
            else:
                logger.warning(f"未找到名称为 {name} 的股票")
            
            return stock
    
    def search_stocks(self, keyword: str, search_fields: List[str] = None) -> List[StockInfo]:
        """
        根据关键词搜索股票
        
        参数:
        keyword: 搜索关键词
        search_fields: 搜索字段列表，默认为['name', 'symbol', 'ts_code']
        
        返回:
        List[StockInfo]: 匹配的股票列表
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.stocks:
                logger.warning("股票缓存为空")
                return []
            
            if search_fields is None:
                search_fields = ['name', 'symbol', 'ts_code']
            
            results = []
            keyword_lower = keyword.lower()
            
            for stock in self.stocks.values():
                for field in search_fields:
                    field_value = getattr(stock, field, '')
                    if field_value and keyword_lower in str(field_value).lower():
                        results.append(stock)
                        break  # 避免重复添加
            
            logger.info(f"✅ 搜索关键词 '{keyword}' 找到 {len(results)} 只股票")
            return results
    
    def get_stocks_by_industry(self, industry: str) -> List[StockInfo]:
        """
        根据行业获取股票列表
        
        参数:
        industry: 行业名称
        
        返回:
        List[StockInfo]: 该行业的股票列表
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.stocks:
                logger.warning("股票缓存为空")
                return []
            
            results = [stock for stock in self.stocks.values() if stock.industry == industry]
            logger.info(f"✅ 行业 '{industry}' 共有 {len(results)} 只股票")
            return results
    
    def get_stocks_by_market(self, market: str) -> List[StockInfo]:
        """
        根据市场类型获取股票列表
        
        参数:
        market: 市场类型（主板/中小板/创业板/科创板）
        
        返回:
        List[StockInfo]: 该市场的股票列表
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.stocks:
                logger.warning("股票缓存为空")
                return []
            
            results = [stock for stock in self.stocks.values() if stock.market == market]
            logger.info(f"✅ 市场 '{market}' 共有 {len(results)} 只股票")
            return results
    
    def get_stocks_by_exchange(self, exchange: str) -> List[StockInfo]:
        """
        根据交易所获取股票列表
        
        参数:
        exchange: 交易所代码
        
        返回:
        List[StockInfo]: 该交易所的股票列表
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.stocks:
                logger.warning("股票缓存为空")
                return []
            
            results = [stock for stock in self.stocks.values() if stock.exchange == exchange]
            logger.info(f"✅ 交易所 '{exchange}' 共有 {len(results)} 只股票")
            return results
    
    def get_all_stocks(self) -> List[StockInfo]:
        """
        获取所有股票列表
        
        返回:
        List[StockInfo]: 所有股票列表
        """
        with self._lock:
            self._ensure_cache_valid()
            
            if not self.stocks:
                logger.warning("股票缓存为空")
                return []
            
            return list(self.stocks.values())
    
    def get_stocks_count(self) -> int:
        """
        获取股票总数
        
        返回:
        int: 股票总数
        """
        with self._lock:
            self._ensure_cache_valid()
            return len(self.stocks)
    
    def refresh_cache(self):
        """
        手动刷新缓存
        """
        logger.info("手动刷新股票缓存...")
        self._load_stocks()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        返回:
        Dict[str, Any]: 缓存信息
        """
        with self._lock:
            return {
                'stocks_count': len(self.stocks),
                'cache_time': self.cache_time,
                'cache_duration': self.cache_duration,
                'is_valid': self._is_cache_valid(),
                'indexes': {
                    'by_ts_code': len(self.stocks),
                    'by_symbol': len(self.stocks_by_symbol),
                    'by_name': len(self.stocks_by_name)
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
stock_cache = StockCache()

# 便捷函数
def get_stock_by_ts_code(ts_code: str) -> Optional[StockInfo]:
    """
    根据ts_code获取股票信息的便捷函数
    
    参数:
    ts_code: 股票TS代码
    
    返回:
    Optional[StockInfo]: 股票信息对象，如果未找到返回None
    """
    return stock_cache.get_stock_by_ts_code(ts_code)

def get_stock_by_symbol(symbol: str) -> Optional[StockInfo]:
    """
    根据symbol获取股票信息的便捷函数
    
    参数:
    symbol: 股票代码
    
    返回:
    Optional[StockInfo]: 股票信息对象，如果未找到返回None
    """
    return stock_cache.get_stock_by_symbol(symbol)

def get_stock_by_name(name: str) -> Optional[StockInfo]:
    """
    根据股票名称获取股票信息的便捷函数
    
    参数:
    name: 股票名称
    
    返回:
    Optional[StockInfo]: 股票信息对象，如果未找到返回None
    """
    return stock_cache.get_stock_by_name(name)

def search_stocks(keyword: str, search_fields: List[str] = None) -> List[StockInfo]:
    """
    根据关键词搜索股票的便捷函数
    
    参数:
    keyword: 搜索关键词
    search_fields: 搜索字段列表，默认为['name', 'symbol', 'ts_code']
    
    返回:
    List[StockInfo]: 匹配的股票列表
    """
    return stock_cache.search_stocks(keyword, search_fields)

def get_stocks_by_industry(industry: str) -> List[StockInfo]:
    """
    根据行业获取股票列表的便捷函数
    
    参数:
    industry: 行业名称
    
    返回:
    List[StockInfo]: 该行业的股票列表
    """
    return stock_cache.get_stocks_by_industry(industry)

def get_stocks_by_market(market: str) -> List[StockInfo]:
    """
    根据市场类型获取股票列表的便捷函数
    
    参数:
    market: 市场类型（主板/中小板/创业板/科创板）
    
    返回:
    List[StockInfo]: 该市场的股票列表
    """
    return stock_cache.get_stocks_by_market(market)

def get_stocks_by_exchange(exchange: str) -> List[StockInfo]:
    """
    根据交易所获取股票列表的便捷函数
    
    参数:
    exchange: 交易所代码
    
    返回:
    List[StockInfo]: 该交易所的股票列表
    """
    return stock_cache.get_stocks_by_exchange(exchange)

def get_all_stocks() -> List[StockInfo]:
    """
    获取所有股票列表的便捷函数
    
    返回:
    List[StockInfo]: 所有股票列表
    """
    return stock_cache.get_all_stocks()

def get_stocks_count() -> int:
    """
    获取股票总数的便捷函数
    
    返回:
    int: 股票总数
    """
    return stock_cache.get_stocks_count()

def refresh_stocks_cache():
    """
    手动刷新股票缓存的便捷函数
    """
    stock_cache.refresh_cache()

def get_stocks_cache_info() -> Dict[str, Any]:
    """
    获取股票缓存信息的便捷函数
    
    返回:
    Dict[str, Any]: 缓存信息
    """
    return stock_cache.get_cache_info()


# 使用示例
if __name__ == "__main__":
    # 测试缓存服务
    print("=" * 60)
    print("股票缓存服务测试")
    print("=" * 60)
    
    # 获取缓存信息
    cache_info = get_stocks_cache_info()
    print(f"缓存信息: {cache_info}")
    
    # 获取股票总数
    total_count = get_stocks_count()
    print(f"\n股票总数: {total_count}")
    
    # 测试根据ts_code查询
    test_ts_code = "000001.SZ"
    stock = get_stock_by_ts_code(test_ts_code)
    if stock:
        print(f"\n根据ts_code查询 {test_ts_code}:")
        print(f"  股票名称: {stock.name}")
        print(f"  股票代码: {stock.symbol}")
        print(f"  所属行业: {stock.industry}")
        print(f"  市场类型: {stock.market}")
        print(f"  交易所: {stock.exchange}")
    
    # 测试根据symbol查询
    test_symbol = "000001"
    stock = get_stock_by_symbol(test_symbol)
    if stock:
        print(f"\n根据symbol查询 {test_symbol}:")
        print(f"  股票名称: {stock.name}")
        print(f"  TS代码: {stock.ts_code}")
    
    # 测试根据名称查询
    test_name = "平安银行"
    stock = get_stock_by_name(test_name)
    if stock:
        print(f"\n根据名称查询 {test_name}:")
        print(f"  TS代码: {stock.ts_code}")
        print(f"  股票代码: {stock.symbol}")
    
    # 测试搜索功能
    search_results = search_stocks("银行")
    print(f"\n搜索关键词 '银行' 找到 {len(search_results)} 只股票")
    for stock in search_results[:5]:  # 只显示前5个
        print(f"  {stock.name} ({stock.symbol}) - {stock.industry}")
    
    # 测试按行业查询
    industry_stocks = get_stocks_by_industry("银行")
    print(f"\n银行业股票数量: {len(industry_stocks)}")
    
    # 测试按市场查询
    market_stocks = get_stocks_by_market("主板")
    print(f"主板股票数量: {len(market_stocks)}") 