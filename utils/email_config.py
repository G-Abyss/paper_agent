#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮箱配置管理模块
"""

import os
import json
import logging
from typing import Dict, Optional

CONFIG_FILE = 'email_config.json'

def get_config_path():
    """获取配置文件路径"""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), CONFIG_FILE)

def load_email_config() -> Dict:
    """
    从配置文件加载邮箱配置
    
    Returns:
        Dict: 邮箱配置字典，包含 user 和 password
    """
    config_path = get_config_path()
    
    if not os.path.exists(config_path):
        # 如果配置文件不存在，返回None（使用环境变量）
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            config = {
                'user': data.get('user', ''),
                'password': data.get('password', ''),
                'imap_server': data.get('imap_server', 'imap.qq.com'),
                'imap_port': data.get('imap_port', 993)
            }
            # 验证配置
            if config['user'] and config['password']:
                return config
            return None
    except Exception as e:
        logging.error(f"加载邮箱配置失败: {e}")
        return None

def save_email_config(user: str, password: str, imap_server: str = 'imap.qq.com', imap_port: int = 993) -> bool:
    """
    保存邮箱配置到配置文件
    
    Args:
        user: 邮箱账号
        password: 邮箱密码/授权码
        imap_server: IMAP服务器地址（默认：imap.qq.com）
        imap_port: IMAP端口（默认：993）
    
    Returns:
        bool: 是否保存成功
    """
    config_path = get_config_path()
    
    try:
        config = {
            'user': user,
            'password': password,
            'imap_server': imap_server,
            'imap_port': imap_port
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logging.info(f"邮箱配置已保存到: {config_path}")
        return True
    except Exception as e:
        logging.error(f"保存邮箱配置失败: {e}")
        return False

def get_email_config() -> Optional[Dict]:
    """
    获取邮箱配置（优先从配置文件读取，否则从环境变量读取）
    
    Returns:
        Dict: 邮箱配置字典，如果都不存在则返回None
    """
    # 优先从配置文件读取
    config = load_email_config()
    if config:
        return config
    
    # 如果配置文件不存在，从环境变量读取
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    user = os.getenv('QMAIL_USER')
    password = os.getenv('QMAIL_PASSWORD')
    
    if user and password:
        return {
            'user': user,
            'password': password,
            'imap_server': os.getenv('QMAIL_IMAP_SERVER', 'imap.qq.com'),
            'imap_port': int(os.getenv('QMAIL_IMAP_PORT', '993'))
        }
    
    return None

