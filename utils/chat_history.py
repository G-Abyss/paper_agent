#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话历史管理系统
负责保存、加载和压缩对话历史
"""

import json
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
import tiktoken

# 最大上下文token数
MAX_CONTEXT_TOKENS = 6000

def get_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """估算文本的token数量"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # 如果tiktoken不支持该模型，使用简单估算：1 token ≈ 4 字符
        return len(text) // 4

def get_chat_history_path() -> str:
    """获取对话历史文件路径"""
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, 'chat_history.json')

def load_chat_history() -> List[Dict]:
    """加载对话历史"""
    history_path = get_chat_history_path()
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载对话历史失败: {e}")
            return []
    return []

def save_chat_history(history: List[Dict]):
    """保存对话历史"""
    history_path = get_chat_history_path()
    try:
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"保存对话历史失败: {e}")

def add_message(role: str, content: str, metadata: Optional[Dict] = None):
    """添加一条消息到历史记录"""
    history = load_chat_history()
    message = {
        'role': role,  # 'user' 或 'assistant'
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    history.append(message)
    save_chat_history(history)
    return message

def get_context_string(max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """获取格式化的上下文字符串，不超过指定token数"""
    history = load_chat_history()
    if not history:
        return ""
    
    # 从最新消息开始，逐步添加直到达到token限制
    context_parts = []
    total_tokens = 0
    
    for message in reversed(history):
        role = message.get('role', 'user')
        content = message.get('content', '')
        msg_text = f"{role}: {content}\n"
        msg_tokens = get_token_count(msg_text)
        
        if total_tokens + msg_tokens > max_tokens:
            break
        
        context_parts.insert(0, msg_text)
        total_tokens += msg_tokens
    
    return "".join(context_parts)

def compress_context(compressor_agent=None):
    """
    压缩对话历史上下文
    如果上下文超过限制，调用压缩器agent进行压缩
    
    Args:
        compressor_agent: 压缩器agent（系统管家/优化器），如果为None则使用简单策略
    """
    history = load_chat_history()
    if not history:
        return
    
    context_str = get_context_string(max_tokens=MAX_CONTEXT_TOKENS * 2)  # 检查是否超过2倍限制
    current_tokens = get_token_count(context_str)
    
    if current_tokens <= MAX_CONTEXT_TOKENS:
        return  # 不需要压缩
    
    logging.info(f"对话历史超过限制 ({current_tokens} tokens)，开始压缩...")
    
    if compressor_agent:
        # 使用压缩器agent进行智能压缩
        # TODO: 实现压缩器agent调用
        logging.warning("压缩器agent尚未实现，使用简单策略")
        simple_compress_history()
    else:
        # 简单策略：保留最近的N条消息
        simple_compress_history()

def simple_compress_history(keep_recent: int = 20):
    """简单压缩策略：只保留最近的N条消息"""
    history = load_chat_history()
    if len(history) > keep_recent:
        # 保留最近的N条消息
        compressed_history = history[-keep_recent:]
        save_chat_history(compressed_history)
        logging.info(f"已压缩对话历史：保留最近 {keep_recent} 条消息")

def clear_chat_history():
    """清空对话历史"""
    save_chat_history([])
    logging.info("对话历史已清空")
