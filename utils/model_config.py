#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型配置管理模块
"""

import os
import json
import logging
from typing import List, Dict, Optional

CONFIG_FILE = 'model_config.json'

def get_config_path():
    """获取配置文件路径"""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), CONFIG_FILE)

def load_models() -> List[Dict]:
    """
    从配置文件加载模型列表
    
    Returns:
        List[Dict]: 模型配置列表
    """
    config_path = get_config_path()
    
    if not os.path.exists(config_path):
        # 如果配置文件不存在，返回默认配置
        return get_default_models()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            models = data.get('models', [])
            # 验证模型配置格式
            validated_models = []
            for model in models:
                if validate_model_config(model):
                    validated_models.append(model)
            return validated_models if validated_models else get_default_models()
    except Exception as e:
        logging.error(f"加载模型配置失败: {e}")
        return get_default_models()

def save_models(models: List[Dict]) -> bool:
    """
    保存模型列表到配置文件
    
    Args:
        models: 模型配置列表
        
    Returns:
        bool: 是否保存成功
    """
    config_path = get_config_path()
    
    try:
        # 验证所有模型配置
        validated_models = []
        for model in models:
            if validate_model_config(model):
                validated_models.append(model)
            else:
                logging.warning(f"跳过无效的模型配置: {model.get('name', 'Unknown')}")
        
        data = {
            'models': validated_models,
            'version': '1.0'
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"成功保存 {len(validated_models)} 个模型配置到 {config_path}")
        return True
    except Exception as e:
        logging.error(f"保存模型配置失败: {e}")
        return False

def validate_model_config(model: Dict) -> bool:
    """
    验证模型配置是否有效
    
    Args:
        model: 模型配置字典
        
    Returns:
        bool: 是否有效
    """
    required_fields = ['name', 'type']
    
    # 检查必需字段
    for field in required_fields:
        if field not in model or not model[field]:
            return False
    
    # 检查类型
    if model['type'] not in ['local', 'remote']:
        return False
    
    # 本地模型需要 base_url 和 model_name
    if model['type'] == 'local':
        if 'base_url' not in model or not model['base_url']:
            return False
        if 'model_name' not in model or not model['model_name']:
            return False
    
    # 远程模型需要 api_key 和 base_url（或 api_base）
    if model['type'] == 'remote':
        if 'api_key' not in model or not model['api_key']:
            return False
        if 'base_url' not in model and 'api_base' not in model:
            return False
    
    return True

def get_default_models() -> List[Dict]:
    """
    获取默认模型配置
    
    Returns:
        List[Dict]: 默认模型配置列表
    """
    return [
        {
            'id': 'default_ollama',
            'name': '默认 Ollama',
            'type': 'local',
            'base_url': 'http://192.168.2.169:11434',
            'model_name': 'qwen2.5:32b',
            'api_key': 'ollama',
            'description': '默认本地 Ollama 模型',
            'is_default': True,  # 标记为默认模型
            'enabled': True  # 默认启用，参与评审
        }
    ]

def get_active_model() -> Optional[Dict]:
    """
    获取当前激活的模型配置（默认模型）
    
    Returns:
        Optional[Dict]: 激活的模型配置，如果没有则返回None
    """
    models = load_models()
    if not models:
        return None
    
    # 查找标记为 is_default 的模型
    for model in models:
        if model.get('is_default', False):
            return model
    
    # 如果没有默认模型，返回第一个
    return models[0] if models else None

def get_selected_models() -> List[Dict]:
    """
    获取所有启用的模型配置（用于多模型并行处理，如评审）
    
    Returns:
        List[Dict]: 启用的模型配置列表
    """
    models = load_models()
    if not models:
        return []
    
    # 返回所有启用的模型（enabled为True或未设置，默认启用）
    enabled_models = []
    for model in models:
        # enabled字段为True或未设置时，视为启用
        if model.get('enabled', True):
            enabled_models.append(model)
    
    return enabled_models

def set_active_model(model_id: str) -> bool:
    """
    设置激活的模型
    
    Args:
        model_id: 模型ID
        
    Returns:
        bool: 是否设置成功
    """
    models = load_models()
    
    # 清除所有模型的 active 标记
    for model in models:
        model['active'] = False
    
    # 设置指定模型为激活状态
    for model in models:
        if model.get('id') == model_id:
            model['active'] = True
            return save_models(models)
    
    return False

