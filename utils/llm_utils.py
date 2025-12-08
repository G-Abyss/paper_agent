#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM工具函数 - 根据模型配置创建LLM实例
"""

import os
import logging
from crewai import LLM
from typing import Dict, Optional


def create_llm_from_model_config(model_config: Dict) -> Optional[LLM]:
    """
    根据模型配置创建LLM实例
    
    Args:
        model_config: 模型配置字典，包含type, base_url, model_name, api_key等
        
    Returns:
        Optional[LLM]: LLM实例，如果创建失败则返回None
    """
    try:
        model_type = model_config.get('type')
        base_url = model_config.get('base_url') or model_config.get('api_base', '')
        model_name = model_config.get('model_name', '')
        api_key = model_config.get('api_key', '')
        
        if not base_url:
            logging.error(f"模型配置缺少base_url: {model_config.get('name', 'Unknown')}")
            return None
        
        if model_type == 'local':
            # 本地模型（Ollama）
            if not model_name:
                logging.error(f"本地模型配置缺少model_name: {model_config.get('name', 'Unknown')}")
                return None
            
            # 设置环境变量
            os.environ['OLLAMA_API_BASE'] = base_url
            if not os.getenv('OPENAI_API_KEY'):
                os.environ['OPENAI_API_KEY'] = 'ollama'
            
            # 模型名称必须包含 "ollama/" 前缀
            llm_model_name = f"ollama/{model_name}" if not model_name.startswith("ollama/") else model_name
            
            logging.info(f"创建本地LLM: model={llm_model_name}, base_url={base_url}")
            
            return LLM(
                model=llm_model_name,
                base_url=base_url,
                api_key="ollama"  # Ollama 不需要真实的 API key
            )
        
        elif model_type == 'remote':
            # 远程模型（API）
            if not api_key:
                logging.error(f"远程模型配置缺少api_key: {model_config.get('name', 'Unknown')}")
                return None
            
            # 根据base_url判断模型类型
            # 这里可以根据不同的API提供商进行适配
            # 例如：OpenAI、Anthropic、DeepSeek等
            
            # 如果base_url包含deepseek，使用deepseek模型
            if 'deepseek' in base_url.lower():
                if not model_name:
                    model_name = 'deepseek-chat'  # 默认模型
                llm_model_name = f"deepseek/{model_name}" if not model_name.startswith("deepseek/") else model_name
            elif 'openai' in base_url.lower() or 'api.openai.com' in base_url.lower():
                if not model_name:
                    model_name = 'gpt-4'  # 默认模型
                llm_model_name = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name
            else:
                # 通用API，使用openai格式
                if not model_name:
                    model_name = 'gpt-4'
                llm_model_name = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name
            
            logging.info(f"创建远程LLM: model={llm_model_name}, base_url={base_url}")
            
            return LLM(
                model=llm_model_name,
                base_url=base_url,
                api_key=api_key
            )
        
        else:
            logging.error(f"不支持的模型类型: {model_type}")
            return None
    
    except Exception as e:
        logging.error(f"创建LLM实例失败: {str(e)}")
        return None

