#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话历史管理工具
用于存储和检索用户与知识库助手的对话历史
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
from utils.vector_db import get_db_connection, return_db_connection
from psycopg2.extras import RealDictCursor, Json


def save_chat_message(user_message: str, assistant_message: str, session_id: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
    """
    保存对话消息到数据库
    
    Args:
        user_message: 用户消息
        assistant_message: 助手回复
        session_id: 会话ID（可选）
        metadata: 额外元数据（可选）
    
    Returns:
        是否成功保存
    """
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO chat_history (session_id, user_message, assistant_message, metadata)
                VALUES (%s, %s, %s, %s)
            """, (
                session_id,
                user_message,
                assistant_message,
                Json(metadata) if metadata else None
            ))
            conn.commit()
            return True
        finally:
            return_db_connection(conn)
    except Exception as e:
        logging.error(f"保存对话历史失败: {str(e)}", exc_info=True)
        return False


def get_chat_history(limit: int = 20, session_id: Optional[str] = None) -> List[Dict]:
    """
    获取对话历史
    
    Args:
        limit: 返回的最大记录数，默认20
        session_id: 会话ID（可选，如果提供则只返回该会话的记录）
    
    Returns:
        对话历史列表，格式为 [{'user_message': ..., 'assistant_message': ..., 'created_at': ...}, ...]
    """
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            if session_id:
                cur.execute("""
                    SELECT user_message, assistant_message, created_at, metadata
                    FROM chat_history
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (session_id, limit))
            else:
                cur.execute("""
                    SELECT user_message, assistant_message, created_at, metadata
                    FROM chat_history
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
            
            results = cur.fetchall()
            # 转换为字典列表并反转顺序（最旧的在前）
            history = []
            for row in reversed(results):
                history.append({
                    'user_message': row['user_message'],
                    'assistant_message': row['assistant_message'],
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                    'metadata': row['metadata'] if row['metadata'] else {}
                })
            
            return history
        finally:
            return_db_connection(conn)
    except Exception as e:
        logging.error(f"获取对话历史失败: {str(e)}", exc_info=True)
        return []


def format_chat_history_for_context(history: List[Dict], max_length: int = 4000) -> str:
    """
    将对话历史格式化为上下文字符串，用于传递给LLM
    
    Args:
        history: 对话历史列表
        max_length: 最大字符长度（避免超出上下文限制）
    
    Returns:
        格式化的对话历史字符串
    """
    if not history:
        return ""
    
    formatted_parts = ["## 历史对话记录\n"]
    total_length = len(formatted_parts[0])
    
    # 从最旧的开始，但只取最近的几条
    for i, entry in enumerate(history):
        user_msg = entry.get('user_message', '')
        assistant_msg = entry.get('assistant_message', '')
        
        entry_text = f"\n[对话 {i+1}]\n用户: {user_msg}\n助手: {assistant_msg}\n"
        
        # 检查是否会超出长度限制
        if total_length + len(entry_text) > max_length:
            formatted_parts.append(f"\n... (还有 {len(history) - i} 条历史记录未显示)")
            break
        
        formatted_parts.append(entry_text)
        total_length += len(entry_text)
    
    return "".join(formatted_parts)

