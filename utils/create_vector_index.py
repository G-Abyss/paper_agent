#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建向量索引脚本

为paper_chunks表的embedding列创建IVFFlat索引，优化向量搜索性能
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

def create_vector_index(lists: int = 100):
    """
    创建向量索引
    
    Args:
        lists: IVFFlat索引的列表数量，默认100
               - 数据量小（<10万）：50-100
               - 数据量中等（10-100万）：100-200
               - 数据量大（>100万）：200-500
    """
    print("=" * 80)
    print("创建向量索引")
    print("=" * 80)
    print()
    
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            
            # 1. 检查是否已有索引
            print("1. 检查现有索引...")
            cur.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'paper_chunks' 
                AND indexname LIKE '%embedding%'
            """)
            existing = cur.fetchall()
            if existing:
                print(f"   ⚠️  已存在 {len(existing)} 个向量索引:")
                for idx in existing:
                    print(f"     - {idx[0]}")
                response = input("\n是否删除旧索引并重新创建？(y/N): ").strip().lower()
                if response != 'y':
                    print("已取消操作")
                    return False
                
                # 删除旧索引
                for idx in existing:
                    print(f"   删除索引: {idx[0]}")
                    cur.execute(f"DROP INDEX IF EXISTS {idx[0]}")
            
            # 2. 检查数据量
            print("\n2. 检查数据量...")
            cur.execute("SELECT COUNT(*) FROM paper_chunks")
            chunk_count = cur.fetchone()[0]
            print(f"   向量块数量: {chunk_count}")
            
            if chunk_count == 0:
                print("   ⚠️  没有数据，无法创建索引")
                print("   建议：先存储一些论文数据后再创建索引")
                return False
            
            # 3. 根据数据量调整lists参数
            if chunk_count < 10000:
                recommended_lists = 50
            elif chunk_count < 100000:
                recommended_lists = 100
            else:
                recommended_lists = min(200, max(100, chunk_count // 1000))
            
            if lists != recommended_lists:
                print(f"   建议lists参数: {recommended_lists}（当前: {lists}）")
                response = input(f"是否使用建议值 {recommended_lists}？(Y/n): ").strip().lower()
                if response != 'n':
                    lists = recommended_lists
            
            # 4. 创建索引
            print(f"\n3. 创建向量索引（lists={lists}）...")
            print("   这可能需要一些时间，请耐心等待...")
            
            index_name = "idx_chunks_embedding"
            create_sql = f"""
                CREATE INDEX {index_name} 
                ON paper_chunks 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = {lists})
            """
            
            cur.execute(create_sql)
            conn.commit()
            
            print(f"   ✅ 索引创建成功: {index_name}")
            
            # 5. 验证索引
            print("\n4. 验证索引...")
            cur.execute("""
                SELECT 
                    indexname,
                    indexdef
                FROM pg_indexes 
                WHERE indexname = %s
            """, (index_name,))
            index_info = cur.fetchone()
            if index_info:
                print(f"   索引名称: {index_info[0]}")
                print(f"   索引定义: {index_info[1][:100]}...")
                print("   ✅ 索引验证成功")
            
            print("\n" + "=" * 80)
            print("✅ 向量索引创建完成！")
            print("=" * 80)
            print("\n提示：")
            print("- 索引已创建，向量搜索性能将得到提升")
            print("- 如果后续数据量大幅增加，可以重新运行此脚本调整lists参数")
            return True
            
        except Exception as e:
            conn.rollback()
            print(f"\n❌ 创建索引失败: {str(e)}")
            print("\n可能的原因：")
            print("1. 数据量太少（IVFFlat需要至少一些数据）")
            print("2. pgvector扩展未正确安装")
            print("3. 数据库权限不足")
            import traceback
            traceback.print_exc()
            return False
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        print(f"\n❌ 连接数据库失败: {str(e)}")
        print("\n请检查：")
        print("1. PostgreSQL是否已安装并运行")
        print("2. 数据库连接配置是否正确（.env文件）")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='创建向量索引')
    parser.add_argument('--lists', type=int, default=100, 
                       help='IVFFlat索引的列表数量（默认100）')
    args = parser.parse_args()
    
    success = create_vector_index(lists=args.lists)
    sys.exit(0 if success else 1)

