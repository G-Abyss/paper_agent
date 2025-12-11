#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速修复脚本：创建 chat_history 表

如果数据库中没有 chat_history 表，运行此脚本可以快速创建它。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector_db import get_db_connection, return_db_connection
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_chat_history_table():
    """创建 chat_history 表"""
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            
            # 检查表是否已存在
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'chat_history'
                )
            """)
            exists = cur.fetchone()[0]
            
            if exists:
                print("✅ chat_history 表已存在，无需创建")
                return True
            
            # 创建表
            print("正在创建 chat_history 表...")
            cur.execute("""
                CREATE TABLE chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB
                )
            """)
            
            # 创建索引
            print("正在创建索引...")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_history_created_at 
                ON chat_history(created_at DESC)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_history_session_id 
                ON chat_history(session_id)
            """)
            
            conn.commit()
            print("✅ chat_history 表创建成功！")
            return True
            
        finally:
            return_db_connection(conn)
    except Exception as e:
        logging.error(f"创建 chat_history 表失败: {str(e)}")
        print(f"❌ 创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 80)
    print("创建 chat_history 表")
    print("=" * 80)
    print()
    
    if create_chat_history_table():
        print("\n✅ 完成！现在可以使用对话历史功能了。")
    else:
        print("\n❌ 失败！请检查数据库连接和权限。")
        sys.exit(1)

