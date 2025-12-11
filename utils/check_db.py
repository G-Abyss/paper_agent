#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库状态检查脚本

检查PostgreSQL数据库是否正确初始化
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector_db import get_db_connection, return_db_connection
from psycopg2.extras import RealDictCursor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_database():
    """检查数据库状态"""
    print("=" * 80)
    print("PostgreSQL 数据库状态检查")
    print("=" * 80)
    print()
    
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 1. 检查pgvector扩展
            print("1. 检查pgvector扩展...")
            cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            ext = cur.fetchone()
            if ext:
                print("   ✅ pgvector扩展已安装")
            else:
                print("   ❌ pgvector扩展未安装")
                print("   请运行: CREATE EXTENSION vector;")
                return False
            
            # 2. 检查papers表
            print("\n2. 检查papers表...")
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'papers'
                )
            """)
            if cur.fetchone()['exists']:
                print("   ✅ papers表存在")
                
                # 检查表结构
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'papers'
                    ORDER BY ordinal_position
                """)
                columns = cur.fetchall()
                print(f"   表结构: {len(columns)} 个字段")
                for col in columns[:5]:  # 只显示前5个
                    print(f"     - {col['column_name']}: {col['data_type']}")
            else:
                print("   ❌ papers表不存在")
                print("   请运行: python utils/init_db.py")
                return False
            
            # 3. 检查paper_chunks表
            print("\n3. 检查paper_chunks表...")
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'paper_chunks'
                )
            """)
            if cur.fetchone()['exists']:
                print("   ✅ paper_chunks表存在")
                
                # 检查是否有embedding列
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'paper_chunks' AND column_name = 'embedding'
                """)
                if cur.fetchone():
                    print("   ✅ embedding列存在（向量类型）")
                else:
                    print("   ❌ embedding列不存在")
                    return False
            else:
                print("   ❌ paper_chunks表不存在")
                print("   请运行: python utils/init_db.py")
                return False
            
            # 4. 检查数据
            print("\n4. 检查数据...")
            cur.execute("SELECT COUNT(*) as count FROM papers")
            paper_count = cur.fetchone()['count']
            print(f"   论文数量: {paper_count}")
            
            cur.execute("SELECT COUNT(*) as count FROM paper_chunks")
            chunk_count = cur.fetchone()['count']
            print(f"   向量块数量: {chunk_count}")
            
            if paper_count > 0:
                # 显示前3篇论文
                cur.execute("SELECT paper_id, title, source FROM papers LIMIT 3")
                papers = cur.fetchall()
                print("\n   前3篇论文:")
                for paper in papers:
                    print(f"     - {paper['paper_id'][:16]}... | {paper['title'][:50]} | {paper['source']}")
            
            # 5. 检查向量索引（可选）
            print("\n5. 检查向量索引...")
            cur.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'paper_chunks' 
                AND indexname LIKE '%embedding%'
            """)
            indexes = cur.fetchall()
            if indexes:
                print(f"   ✅ 找到 {len(indexes)} 个向量索引")
                for idx in indexes:
                    print(f"     - {idx['indexname']}")
            else:
                print("   ⚠️  未找到向量索引（可选，用于优化搜索性能）")
                print("   可以运行以下命令创建索引:")
                print("   python utils/create_vector_index.py")
                print("   或者手动执行SQL:")
                print("   CREATE INDEX idx_chunks_embedding ON paper_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            
            print("\n" + "=" * 80)
            print("✅ 数据库状态正常！")
            print("=" * 80)
            return True
            
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        print(f"\n❌ 检查失败: {str(e)}")
        print("\n请检查：")
        print("1. PostgreSQL是否已安装并运行")
        print("2. 数据库连接配置是否正确（.env文件）")
        print("3. 数据库和用户是否存在")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = check_database()
    sys.exit(0 if success else 1)

