#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Redis工具类 - 单例模式
提供Redis基本数据结构的操作方法
"""
__author__ = "zhenghang(chibizhenghang@gmail.com)"

import os
import json
import sys
import threading
from typing import Any, Dict, List, Optional, Union
import redis
from redis import ConnectionPool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foundation.app_env import env_app_config as config


class RedisClient:
    """
    Redis客户端单例类
    提供String、Hash、List、Set、ZSet等数据结构的操作方法
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, custom_config: Optional[Dict] = None):
        """
        初始化Redis连接
        
        Args:
            custom_config: 自定义配置，如果不传则使用app_config中的配置
        """
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            redis_config = custom_config or config.get('redis', {})
            
            # 构建连接池
            pool_kwargs = {
                'host': redis_config.get('host', 'localhost'),
                'port': redis_config.get('port', 6379),
                'db': redis_config.get('db', 0),
                'decode_responses': redis_config.get('decode_responses', True),
                'max_connections': redis_config.get('max_connections', 10),
                'socket_timeout': redis_config.get('socket_timeout', 5),
                'socket_connect_timeout': redis_config.get('socket_connect_timeout', 5),
            }
            
            # 如果有密码则添加
            password = redis_config.get('password')
            if password:
                pool_kwargs['password'] = password
            
            self._pool = ConnectionPool(**pool_kwargs)
            self._client = redis.Redis(connection_pool=self._pool)
            self._initialized = True

    @property
    def client(self) -> redis.Redis:
        """获取Redis客户端实例"""
        return self._client

    def ping(self) -> bool:
        """测试连接是否正常"""
        try:
            return self._client.ping()
        except Exception as e:
            print(f"Redis连接失败: {e}")
            return False

    def close(self):
        """关闭连接池"""
        if self._pool:
            self._pool.disconnect()

    # ==================== String 操作 ====================
    
    def set(self, key: str, value: Any, ex: Optional[int] = None, 
            px: Optional[int] = None, nx: bool = False, xx: bool = False) -> bool:
        """
        设置字符串值
        
        Args:
            key: 键名
            value: 值（如果是dict/list会自动转为JSON）
            ex: 过期时间（秒）
            px: 过期时间（毫秒）
            nx: 仅当key不存在时设置
            xx: 仅当key存在时设置
        """
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        return self._client.set(key, value, ex=ex, px=px, nx=nx, xx=xx)

    def get(self, key: str, as_json: bool = False) -> Optional[Any]:
        """
        获取字符串值
        
        Args:
            key: 键名
            as_json: 是否将结果解析为JSON
        """
        value = self._client.get(key)
        if value and as_json:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def mset(self, mapping: Dict[str, Any]) -> bool:
        """批量设置多个键值对"""
        processed = {}
        for k, v in mapping.items():
            if isinstance(v, (dict, list)):
                processed[k] = json.dumps(v, ensure_ascii=False)
            else:
                processed[k] = v
        return self._client.mset(processed)

    def mget(self, keys: List[str], as_json: bool = False) -> List[Optional[Any]]:
        """批量获取多个键的值"""
        values = self._client.mget(keys)
        if as_json:
            result = []
            for v in values:
                if v:
                    try:
                        result.append(json.loads(v))
                    except json.JSONDecodeError:
                        result.append(v)
                else:
                    result.append(v)
            return result
        return values

    def incr(self, key: str, amount: int = 1) -> int:
        """自增"""
        return self._client.incr(key, amount)

    def decr(self, key: str, amount: int = 1) -> int:
        """自减"""
        return self._client.decr(key, amount)

    def incrbyfloat(self, key: str, amount: float) -> float:
        """浮点数自增"""
        return self._client.incrbyfloat(key, amount)

    # ==================== Hash 操作 ====================
    
    def hset(self, name: str, key: str = None, value: Any = None, 
             mapping: Dict = None) -> int:
        """
        设置Hash字段
        
        Args:
            name: Hash名称
            key: 字段名
            value: 字段值
            mapping: 批量设置的字典
        """
        if mapping:
            processed = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list)):
                    processed[k] = json.dumps(v, ensure_ascii=False)
                else:
                    processed[k] = v
            return self._client.hset(name, mapping=processed)
        
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        return self._client.hset(name, key, value)

    def hget(self, name: str, key: str, as_json: bool = False) -> Optional[Any]:
        """获取Hash字段值"""
        value = self._client.hget(name, key)
        if value and as_json:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def hmget(self, name: str, keys: List[str], as_json: bool = False) -> List[Optional[Any]]:
        """批量获取Hash字段值"""
        values = self._client.hmget(name, keys)
        if as_json:
            result = []
            for v in values:
                if v:
                    try:
                        result.append(json.loads(v))
                    except json.JSONDecodeError:
                        result.append(v)
                else:
                    result.append(v)
            return result
        return values

    def hgetall(self, name: str, as_json: bool = False) -> Dict:
        """获取Hash所有字段和值"""
        data = self._client.hgetall(name)
        if as_json:
            result = {}
            for k, v in data.items():
                try:
                    result[k] = json.loads(v)
                except json.JSONDecodeError:
                    result[k] = v
            return result
        return data

    def hdel(self, name: str, *keys) -> int:
        """删除Hash字段"""
        return self._client.hdel(name, *keys)

    def hexists(self, name: str, key: str) -> bool:
        """检查Hash字段是否存在"""
        return self._client.hexists(name, key)

    def hkeys(self, name: str) -> List[str]:
        """获取Hash所有字段名"""
        return self._client.hkeys(name)

    def hvals(self, name: str) -> List[Any]:
        """获取Hash所有值"""
        return self._client.hvals(name)

    def hlen(self, name: str) -> int:
        """获取Hash字段数量"""
        return self._client.hlen(name)

    def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        """Hash字段整数自增"""
        return self._client.hincrby(name, key, amount)

    def hincrbyfloat(self, name: str, key: str, amount: float) -> float:
        """Hash字段浮点数自增"""
        return self._client.hincrbyfloat(name, key, amount)

    # ==================== List 操作 ====================
    
    def lpush(self, name: str, *values) -> int:
        """从左侧插入列表"""
        processed = []
        for v in values:
            if isinstance(v, (dict, list)):
                processed.append(json.dumps(v, ensure_ascii=False))
            else:
                processed.append(v)
        return self._client.lpush(name, *processed)

    def rpush(self, name: str, *values) -> int:
        """从右侧插入列表"""
        processed = []
        for v in values:
            if isinstance(v, (dict, list)):
                processed.append(json.dumps(v, ensure_ascii=False))
            else:
                processed.append(v)
        return self._client.rpush(name, *processed)

    def lpop(self, name: str, as_json: bool = False) -> Optional[Any]:
        """从左侧弹出元素"""
        value = self._client.lpop(name)
        if value and as_json:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def rpop(self, name: str, as_json: bool = False) -> Optional[Any]:
        """从右侧弹出元素"""
        value = self._client.rpop(name)
        if value and as_json:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def lrange(self, name: str, start: int, end: int, as_json: bool = False) -> List[Any]:
        """获取列表指定范围的元素"""
        values = self._client.lrange(name, start, end)
        if as_json:
            result = []
            for v in values:
                try:
                    result.append(json.loads(v))
                except json.JSONDecodeError:
                    result.append(v)
            return result
        return values

    def llen(self, name: str) -> int:
        """获取列表长度"""
        return self._client.llen(name)

    def lindex(self, name: str, index: int, as_json: bool = False) -> Optional[Any]:
        """获取列表指定索引的元素"""
        value = self._client.lindex(name, index)
        if value and as_json:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def lset(self, name: str, index: int, value: Any) -> bool:
        """设置列表指定索引的值"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        return self._client.lset(name, index, value)

    def lrem(self, name: str, count: int, value: Any) -> int:
        """移除列表中的元素"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        return self._client.lrem(name, count, value)

    def ltrim(self, name: str, start: int, end: int) -> bool:
        """修剪列表，只保留指定范围的元素"""
        return self._client.ltrim(name, start, end)

    # ==================== Set 操作 ====================
    
    def sadd(self, name: str, *values) -> int:
        """添加集合成员"""
        processed = []
        for v in values:
            if isinstance(v, (dict, list)):
                processed.append(json.dumps(v, ensure_ascii=False))
            else:
                processed.append(v)
        return self._client.sadd(name, *processed)

    def srem(self, name: str, *values) -> int:
        """移除集合成员"""
        processed = []
        for v in values:
            if isinstance(v, (dict, list)):
                processed.append(json.dumps(v, ensure_ascii=False))
            else:
                processed.append(v)
        return self._client.srem(name, *processed)

    def smembers(self, name: str, as_json: bool = False) -> set:
        """获取集合所有成员"""
        values = self._client.smembers(name)
        if as_json:
            result = set()
            for v in values:
                try:
                    result.add(json.loads(v) if isinstance(json.loads(v), str) else v)
                except json.JSONDecodeError:
                    result.add(v)
            return result
        return values

    def sismember(self, name: str, value: Any) -> bool:
        """检查是否是集合成员"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        return self._client.sismember(name, value)

    def scard(self, name: str) -> int:
        """获取集合成员数量"""
        return self._client.scard(name)

    def sinter(self, keys: List[str]) -> set:
        """获取多个集合的交集"""
        return self._client.sinter(keys)

    def sunion(self, keys: List[str]) -> set:
        """获取多个集合的并集"""
        return self._client.sunion(keys)

    def sdiff(self, keys: List[str]) -> set:
        """获取多个集合的差集"""
        return self._client.sdiff(keys)

    def spop(self, name: str, count: int = None) -> Union[str, List, None]:
        """随机弹出集合成员"""
        return self._client.spop(name, count)

    def srandmember(self, name: str, number: int = None) -> Union[str, List, None]:
        """随机获取集合成员（不删除）"""
        return self._client.srandmember(name, number)

    # ==================== Sorted Set (ZSet) 操作 ====================
    
    def zadd(self, name: str, mapping: Dict[str, float], 
             nx: bool = False, xx: bool = False, 
             gt: bool = False, lt: bool = False) -> int:
        """
        添加有序集合成员
        
        Args:
            name: 有序集合名称
            mapping: {member: score} 字典
            nx: 仅当成员不存在时添加
            xx: 仅当成员存在时更新
            gt: 仅当新score大于当前score时更新
            lt: 仅当新score小于当前score时更新
        """
        return self._client.zadd(name, mapping, nx=nx, xx=xx, gt=gt, lt=lt)

    def zrem(self, name: str, *values) -> int:
        """移除有序集合成员"""
        return self._client.zrem(name, *values)

    def zscore(self, name: str, value: str) -> Optional[float]:
        """获取成员分数"""
        return self._client.zscore(name, value)

    def zrank(self, name: str, value: str) -> Optional[int]:
        """获取成员排名（从小到大，从0开始）"""
        return self._client.zrank(name, value)

    def zrevrank(self, name: str, value: str) -> Optional[int]:
        """获取成员排名（从大到小，从0开始）"""
        return self._client.zrevrank(name, value)

    def zrange(self, name: str, start: int, end: int, 
               withscores: bool = False, desc: bool = False) -> List:
        """
        获取有序集合指定范围的成员
        
        Args:
            name: 有序集合名称
            start: 起始索引
            end: 结束索引
            withscores: 是否返回分数
            desc: 是否降序
        """
        return self._client.zrange(name, start, end, withscores=withscores, desc=desc)

    def zrevrange(self, name: str, start: int, end: int, 
                  withscores: bool = False) -> List:
        """获取有序集合指定范围的成员（降序）"""
        return self._client.zrevrange(name, start, end, withscores=withscores)

    def zrangebyscore(self, name: str, min_score: float, max_score: float,
                      start: int = None, num: int = None, 
                      withscores: bool = False) -> List:
        """按分数范围获取成员"""
        return self._client.zrangebyscore(name, min_score, max_score, 
                                          start=start, num=num, withscores=withscores)

    def zcard(self, name: str) -> int:
        """获取有序集合成员数量"""
        return self._client.zcard(name)

    def zcount(self, name: str, min_score: float, max_score: float) -> int:
        """统计分数范围内的成员数量"""
        return self._client.zcount(name, min_score, max_score)

    def zincrby(self, name: str, amount: float, value: str) -> float:
        """增加成员分数"""
        return self._client.zincrby(name, amount, value)

    def zremrangebyrank(self, name: str, min_rank: int, max_rank: int) -> int:
        """按排名范围删除成员"""
        return self._client.zremrangebyrank(name, min_rank, max_rank)

    def zremrangebyscore(self, name: str, min_score: float, max_score: float) -> int:
        """按分数范围删除成员"""
        return self._client.zremrangebyscore(name, min_score, max_score)

    # ==================== 通用操作 ====================
    
    def delete(self, *keys) -> int:
        """删除键"""
        return self._client.delete(*keys)

    def exists(self, *keys) -> int:
        """检查键是否存在，返回存在的键的数量"""
        return self._client.exists(*keys)

    def expire(self, key: str, seconds: int) -> bool:
        """设置键的过期时间（秒）"""
        return self._client.expire(key, seconds)

    def expireat(self, key: str, timestamp: int) -> bool:
        """设置键在指定时间戳过期"""
        return self._client.expireat(key, timestamp)

    def ttl(self, key: str) -> int:
        """获取键的剩余过期时间（秒）"""
        return self._client.ttl(key)

    def pttl(self, key: str) -> int:
        """获取键的剩余过期时间（毫秒）"""
        return self._client.pttl(key)

    def persist(self, key: str) -> bool:
        """移除键的过期时间"""
        return self._client.persist(key)

    def keys(self, pattern: str = '*') -> List[str]:
        """查找匹配的键"""
        return self._client.keys(pattern)

    def scan(self, cursor: int = 0, match: str = None, count: int = None) -> tuple:
        """增量迭代键"""
        return self._client.scan(cursor=cursor, match=match, count=count)

    def scan_iter(self, match: str = None, count: int = None):
        """迭代所有匹配的键"""
        return self._client.scan_iter(match=match, count=count)

    def type(self, key: str) -> str:
        """获取键的类型"""
        return self._client.type(key)

    def rename(self, src: str, dst: str) -> bool:
        """重命名键"""
        return self._client.rename(src, dst)

    def renamenx(self, src: str, dst: str) -> bool:
        """仅当新键不存在时重命名"""
        return self._client.renamenx(src, dst)

    def dbsize(self) -> int:
        """获取当前数据库的键数量"""
        return self._client.dbsize()

    def flushdb(self, asynchronous: bool = False) -> bool:
        """清空当前数据库"""
        return self._client.flushdb(asynchronous=asynchronous)

    # ==================== 管道操作 ====================
    
    def pipeline(self, transaction: bool = True):
        """
        创建管道，用于批量执行命令
        
        使用示例:
            pipe = redis_client.pipeline()
            pipe.set('key1', 'value1')
            pipe.set('key2', 'value2')
            pipe.get('key1')
            results = pipe.execute()
        """
        return self._client.pipeline(transaction=transaction)

    # ==================== 发布订阅 ====================
    
    def publish(self, channel: str, message: Any) -> int:
        """发布消息到频道"""
        if isinstance(message, (dict, list)):
            message = json.dumps(message, ensure_ascii=False)
        return self._client.publish(channel, message)

    def pubsub(self):
        """获取发布订阅对象"""
        return self._client.pubsub()


# 全局单例实例
redis_client = RedisClient()


def get_redis_client(custom_config: Optional[Dict] = None) -> RedisClient:
    """
    获取Redis客户端实例
    
    Args:
        custom_config: 自定义配置（仅在首次初始化时生效）
    
    Returns:
        RedisClient单例实例
    """
    if custom_config:
        return RedisClient(custom_config)
    return redis_client


# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 获取客户端实例
    client = get_redis_client()
    
    # 测试连接
    if client.ping():
        print("Redis连接成功!")
    else:
        print("Redis连接失败!")
        exit(1)
    
    # String 操作示例
    print("\n=== String 操作 ===")
    client.set('test_str', 'Hello Redis!')
    print(f"get test_str: {client.get('test_str')}")
    
    # 存储JSON
    client.set('test_json', {'name': '张三', 'age': 25})
    print(f"get test_json: {client.get('test_json', as_json=True)}")
    
    # 自增
    client.set('counter', 0)
    client.incr('counter')
    client.incr('counter', 5)
    print(f"counter: {client.get('counter')}")
    
    # Hash 操作示例
    print("\n=== Hash 操作 ===")
    client.hset('user:1001', mapping={'name': '李四', 'age': '30', 'city': '北京'})
    print(f"hgetall user:1001: {client.hgetall('user:1001')}")
    print(f"hget user:1001 name: {client.hget('user:1001', 'name')}")
    
    # List 操作示例
    print("\n=== List 操作 ===")
    client.delete('mylist')
    client.rpush('mylist', 'a', 'b', 'c')
    client.lpush('mylist', 'first')
    print(f"lrange mylist 0 -1: {client.lrange('mylist', 0, -1)}")
    print(f"llen mylist: {client.llen('mylist')}")
    
    # Set 操作示例
    print("\n=== Set 操作 ===")
    client.delete('myset')
    client.sadd('myset', 'apple', 'banana', 'orange')
    print(f"smembers myset: {client.smembers('myset')}")
    print(f"sismember myset apple: {client.sismember('myset', 'apple')}")
    
    # ZSet 操作示例
    print("\n=== ZSet 操作 ===")
    client.delete('leaderboard')
    client.zadd('leaderboard', {'player1': 100, 'player2': 85, 'player3': 92})
    print(f"zrange leaderboard 0 -1 (升序): {client.zrange('leaderboard', 0, -1, withscores=True)}")
    print(f"zrevrange leaderboard 0 -1 (降序): {client.zrevrange('leaderboard', 0, -1, withscores=True)}")
    print(f"zscore leaderboard player1: {client.zscore('leaderboard', 'player1')}")
    
    # 管道操作示例
    print("\n=== 管道操作 ===")
    pipe = client.pipeline()
    pipe.set('pipe_key1', 'value1')
    pipe.set('pipe_key2', 'value2')
    pipe.get('pipe_key1')
    pipe.get('pipe_key2')
    results = pipe.execute()
    print(f"pipeline results: {results}")
    
    # 清理测试数据
    client.delete('test_str', 'test_json', 'counter', 'user:1001', 
                  'mylist', 'myset', 'leaderboard', 'pipe_key1', 'pipe_key2')
    
    print("\n测试完成!")

