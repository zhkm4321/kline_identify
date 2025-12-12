#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配置信息"""
__author__ = "zhenghang(zhkm_cb@163.com)"

development_config = {
    "DEBUG": True,
    "schedulerTaskEnable": False,
    "database": {
        'host': 'localhost',
        'port': 3306,
        'user': 'xxxxx',
        'password': 'xxxxx',
        'database': 'zstock_db',
        'charset': 'utf8mb4'
    },
    "redis": {
        'host': '127.0.0.1',
        'port': 6379,
        'password': 'xxxxxx',
        'db': 1,
        'decode_responses': True,  # 自动解码响应为字符串
        'max_connections': 10,  # 连接池最大连接数
        'socket_timeout': 5,  # 套接字超时（秒）
        'socket_connect_timeout': 5  # 连接超时（秒）
    }
}

testing_config = {
    "DEBUG": True,
    "schedulerTaskEnable": False,
    "database": {
        'host': 'localhost',
        'port': 3306,
        'user': 'xxxxx',
        'password': 'xxxxxx',
        'database': 'zstock_db',
        'charset': 'utf8mb4'
    },
    "redis": {
        'host': '127.0.0.1',
        'port': 6379,
        'password': 'xxxxxx',
        'db': 1,
        'decode_responses': True,
        'max_connections': 10,
        'socket_timeout': 5,
        'socket_connect_timeout': 5
    }
}

production_config = {
    "DEBUG": True,
    "schedulerTaskEnable": False,
    "database": {
        'host': 'localhost',
        'port': 3306,
        'user': 'xxxxx',
        'password': 'xxxxxx',
        'database': 'zstock_db',
        'charset': 'utf8mb4'
    },
    "redis": {
        'host': '127.0.0.1',
        'port': 6379,
        'password': 'xxxxxx',
        'db': 1,
        'decode_responses': True,
        'max_connections': 10,
        'socket_timeout': 5,
        'socket_connect_timeout': 5
    }
}
