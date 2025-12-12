# app_env.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foundation import app_config

# 根据当前目录路径判断环境
def get_environment():
    """
    根据当前目录路径判断环境
    - cursor_py_work: 开发环境 (dev)
    - python_work: 生产环境 (prod)
    - 其他: 默认使用开发环境
    """
    # 使用脚本文件本身的路径来判断，更可靠
    current_file_path = os.path.abspath(__file__)
    current_path = os.path.abspath(os.getcwd())
    
    # 优先使用脚本所在路径，如果不行则使用工作目录
    check_path = current_file_path if ('cursor_py_work' in current_file_path or 'develop' in current_file_path) else current_path
    
    if 'cursor_py_work' in check_path:
        return 'dev'
    elif 'develop' in check_path:
        return 'prod'
    else:
        # 默认使用开发环境
        return 'dev'

db_config = {}
env_app_config = {}
# 导入 app_config 中的配置
try:
    # 尝试导入 app_config（可能需要向上查找路径）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # 根据环境获取对应的配置
    env = get_environment()
    if env == 'dev':
        env_app_config = app_config.development_config
    elif env == 'prod':
        env_app_config = app_config.production_config
    else:
        env_app_config = app_config.testing_config
    
    # 从配置中提取数据库配置
    db_config_dict = env_app_config.get('database', {})
    db_config = {
        'host': db_config_dict.get('host', 'localhost'),
        'port': db_config_dict.get('port', 3306),
        'database': db_config_dict.get('database', 'zstock_db'),
        'user': db_config_dict.get('user', 'root'),
        'password': db_config_dict.get('password', 'root'),
        'charset': db_config_dict.get('charset', 'utf8mb4')
    }
    print(f"[db_config] 检测到环境: {env}")
    
except ImportError as e:
    # 如果无法导入 app_config，使用默认配置
    print(f"[db_config] 警告: 无法导入 app_config ({e}), 使用默认配置")
    db_config = {
        'host': 'localhost',
        'port': 3306,
        'database': 'zstock_db',
        'user': 'root',
        'password': 'root',
        'charset': 'utf8mb4'
    }
