"""
LanceDB 向量数据库通用工具服务

提供向量的写入、读取、搜索等功能
"""
import os
import numpy as np
import pandas as pd
import lancedb
from typing import List, Optional, Dict, Any, Union


class LanceVectorDB:
    """LanceDB 向量数据库服务类"""
    
    def __init__(self, db_path: str = "E:/data/kline_lance"):
        """
        初始化 LanceDB 连接
        
        Args:
            db_path: 数据库存储路径
        """
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self.db_path = db_path
    
    def create_table(
        self, 
        table_name: str, 
        vectors: np.ndarray,
        metadata: Optional[Dict[str, List]] = None,
        mode: str = "overwrite"
    ) -> lancedb.table.Table:
        """
        创建表并写入向量数据
        
        Args:
            table_name: 表名
            vectors: 向量数组，形状为 (N, dim)
            metadata: 元数据字典，如 {"code": [...], "trade_date": [...]}
            mode: 写入模式，"overwrite" 覆盖 或 "append" 追加
            
        Returns:
            LanceDB 表对象
        """
        if vectors.ndim != 2:
            raise ValueError(f"向量数组必须是2维的，当前维度: {vectors.ndim}")
        
        # 构建 DataFrame
        df_dict = {"vector": vectors.tolist()}
        
        # 添加元数据列
        if metadata:
            for key, values in metadata.items():
                if len(values) != len(vectors):
                    raise ValueError(f"元数据 '{key}' 长度 ({len(values)}) 与向量数量 ({len(vectors)}) 不匹配")
                df_dict[key] = values
        
        df = pd.DataFrame(df_dict)
        
        if mode == "append" and table_name in self.list_tables():
            table = self.db.open_table(table_name)
            table.add(df)
        else:
            table = self.db.create_table(table_name, df, mode="overwrite")
        
        print(f"表 '{table_name}' 创建/更新成功，共 {len(vectors)} 条向量，维度: {vectors.shape[1]}")
        return table
    
    def add_vectors(
        self,
        table_name: str,
        vectors: np.ndarray,
        metadata: Optional[Dict[str, List]] = None
    ) -> None:
        """
        向已有表追加向量
        
        Args:
            table_name: 表名
            vectors: 向量数组
            metadata: 元数据字典
        """
        self.create_table(table_name, vectors, metadata, mode="append")
    
    def get_table(self, table_name: str) -> lancedb.table.Table:
        """获取表对象"""
        return self.db.open_table(table_name)
    
    def list_tables(self) -> List[str]:
        """列出所有表名"""
        return self.db.table_names()
    
    def load_vectors(
        self, 
        table_name: str,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        读取表中的向量数据
        
        Args:
            table_name: 表名
            columns: 要读取的列名列表，None 表示读取全部
            
        Returns:
            包含向量和元数据的 DataFrame
        """
        table = self.db.open_table(table_name)
        if columns:
            return table.to_pandas()[columns]
        return table.to_pandas()
    
    def load_vectors_as_numpy(self, table_name: str) -> np.ndarray:
        """
        以 numpy 数组形式读取向量
        
        Args:
            table_name: 表名
            
        Returns:
            向量数组，形状为 (N, dim)
        """
        df = self.load_vectors(table_name, columns=["vector"])
        vectors = np.array(df["vector"].tolist())
        return vectors
    
    def search(
        self,
        table_name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        metric: str = "L2",
        filter_expr: Optional[str] = None
    ) -> pd.DataFrame:
        """
        向量相似性搜索
        
        Args:
            table_name: 表名
            query_vector: 查询向量，形状为 (dim,) 或 (1, dim)
            top_k: 返回最相似的 K 个结果
            metric: 距离度量，"L2" 或 "cosine"
            filter_expr: 过滤表达式，如 "code = '600000'"
            
        Returns:
            搜索结果 DataFrame，包含 _distance 列
        """
        table = self.db.open_table(table_name)
        
        # 确保查询向量是 1D
        if query_vector.ndim == 2:
            query_vector = query_vector.flatten()
        
        query = table.search(query_vector.tolist()).limit(top_k).metric(metric)
        
        if filter_expr:
            query = query.where(filter_expr)
        
        return query.to_pandas()
    
    def batch_search(
        self,
        table_name: str,
        query_vectors: np.ndarray,
        top_k: int = 10,
        metric: str = "L2"
    ) -> List[pd.DataFrame]:
        """
        批量向量搜索
        
        Args:
            table_name: 表名
            query_vectors: 查询向量数组，形状为 (N, dim)
            top_k: 每个查询返回的结果数
            metric: 距离度量
            
        Returns:
            搜索结果列表
        """
        results = []
        for vec in query_vectors:
            result = self.search(table_name, vec, top_k, metric)
            results.append(result)
        return results
    
    def delete_table(self, table_name: str) -> bool:
        """
        删除表
        
        Args:
            table_name: 表名
            
        Returns:
            是否删除成功
        """
        try:
            self.db.drop_table(table_name)
            print(f"表 '{table_name}' 已删除")
            return True
        except Exception as e:
            print(f"删除表 '{table_name}' 失败: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        获取表信息
        
        Args:
            table_name: 表名
            
        Returns:
            表信息字典
        """
        table = self.db.open_table(table_name)
        df = table.to_pandas()
        
        # 获取向量维度
        if len(df) > 0:
            vector_dim = len(df["vector"].iloc[0])
        else:
            vector_dim = 0
        
        return {
            "table_name": table_name,
            "row_count": len(df),
            "vector_dim": vector_dim,
            "columns": list(df.columns),
            "schema": str(table.schema)
        }


# 创建默认实例，方便直接导入使用
_default_db: Optional[LanceVectorDB] = None


def get_default_db(db_path: str = "E:/data/kline_lance") -> LanceVectorDB:
    """获取默认的数据库实例（单例模式）"""
    global _default_db
    if _default_db is None:
        _default_db = LanceVectorDB(db_path)
    return _default_db


# ==================== 便捷函数 ====================

def save_embeddings(
    table_name: str,
    vectors: np.ndarray,
    metadata: Optional[Dict[str, List]] = None,
    db_path: str = "E:/data/kline_lance"
) -> None:
    """
    保存 embedding 向量到 LanceDB
    
    Args:
        table_name: 表名
        vectors: 向量数组
        metadata: 元数据
        db_path: 数据库路径
    """
    db = get_default_db(db_path)
    db.create_table(table_name, vectors, metadata)


def load_embeddings(
    table_name: str,
    db_path: str = "E:/data/kline_lance"
) -> np.ndarray:
    """
    从 LanceDB 加载 embedding 向量
    
    Args:
        table_name: 表名
        db_path: 数据库路径
        
    Returns:
        向量数组
    """
    db = get_default_db(db_path)
    return db.load_vectors_as_numpy(table_name)


def search_similar(
    table_name: str,
    query_vector: np.ndarray,
    top_k: int = 10,
    metric: str = "L2",
    db_path: str = "E:/data/kline_lance"
) -> pd.DataFrame:
    """
    搜索相似向量
    
    Args:
        table_name: 表名
        query_vector: 查询向量
        top_k: 返回数量
        metric: 距离度量
        db_path: 数据库路径
        
    Returns:
        搜索结果
    """
    db = get_default_db(db_path)
    return db.search(table_name, query_vector, top_k, metric)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试用例
    db = LanceVectorDB()
    
    # 1. 创建测试数据
    print("=" * 60)
    print("测试 LanceVectorDB 服务")
    print("=" * 60)
    
    embeddings = np.random.rand(100, 128).astype(np.float32)
    metadata = {
        "code": [f"60000{i % 10}" for i in range(100)],
        "trade_date": [f"2024{i:04d}" for i in range(100)]
    }
    
    # 2. 创建表
    db.create_table("test_vectors", embeddings, metadata)
    
    # 3. 获取表信息
    info = db.get_table_info("test_vectors")
    print(f"\n表信息: {info}")
    
    # 4. 搜索测试
    query = np.random.rand(128).astype(np.float32)
    results = db.search("test_vectors", query, top_k=5)
    print(f"\n搜索结果 (top 5):")
    print(results[["code", "trade_date", "_distance"]])
    
    # 5. 列出所有表
    print(f"\n所有表: {db.list_tables()}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
