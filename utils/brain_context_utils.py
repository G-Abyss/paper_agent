import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from utils.vector_db import get_db_connection, return_db_connection
from psycopg2.extras import RealDictCursor

BRAIN_CONTEXT_PATH = os.path.join("data", "brain_cache_context.json")

def get_brain_context() -> Dict[str, Any]:
    """获取大脑缓存上下文"""
    if os.path.exists(BRAIN_CONTEXT_PATH):
        try:
            with open(BRAIN_CONTEXT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"读取大脑上下文失败: {e}")
    
    return {
        "known_topics": [],
        "core_beliefs": [],
        "blind_spots": [],
        "last_updated": None,
        "summary": "尚未进行知识盘点。"
    }

def save_brain_context(context: Dict[str, Any]):
    """保存大脑缓存上下文"""
    os.makedirs(os.path.dirname(BRAIN_CONTEXT_PATH), exist_ok=True)
    try:
        context["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(BRAIN_CONTEXT_PATH, "w", encoding="utf-8") as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"保存大脑上下文失败: {e}")

def update_brain_context_from_notes():
    """从所有笔记（source='note'）中重新生成认知边界"""
    logging.info("正在根据笔记更新大脑认知边界...")
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT title, content, metadata FROM papers WHERE source = 'note'")
        notes = cur.fetchall()
        
        if not notes:
            logging.info("没有找到笔记，跳过上下文更新。")
            return
        
        # 这里实际上应该调用一个 LLM 来总结，但先构建框架
        # 暂时合并所有笔记内容作为输入
        combined_notes = "\n".join([f"标题: {n['title']}\n内容: {n['content']}" for n in notes])
        
        # 在这里我们暂时手动构建一个提示词，后续由 Brain Agent 调用
        # 返回更新后的上下文结构
        return combined_notes
    finally:
        return_db_connection(conn)

def get_think_points(paper_id: str) -> Optional[List[str]]:
    """从数据库获取论文的 think 点"""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT think_points FROM papers WHERE paper_id = %s", (paper_id,))
        row = cur.fetchone()
        return row['think_points'] if row else None
    finally:
        return_db_connection(conn)

def save_think_results(paper_id: str, think_points: List[str], summary: str):
    """保存 think 点和总结到数据库"""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE papers SET think_points = %s, contextual_summary = %s, updated_at = NOW() WHERE paper_id = %s",
            (json.dumps(think_points, ensure_ascii=False), summary, paper_id)
        )
        conn.commit()
    finally:
        return_db_connection(conn)

