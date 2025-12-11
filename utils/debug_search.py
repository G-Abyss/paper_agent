#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索调试脚本

用于调试向量搜索问题，检查：
1. 数据库中实际存储了什么内容
2. 搜索查询返回了什么结果
3. 相似度分数是多少
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector_db import (
    get_db_connection, return_db_connection, 
    search_similar_chunks, get_paper_list
)
from psycopg2.extras import RealDictCursor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_database_content():
    """检查数据库中的实际内容"""
    print("=" * 80)
    print("检查数据库内容")
    print("=" * 80)
    print()
    
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 1. 检查论文列表
            print("1. 论文列表:")
            cur.execute("SELECT paper_id, title, source FROM papers")
            papers = cur.fetchall()
            for paper in papers:
                print(f"   - {paper['paper_id'][:16]}... | {paper['title']} | {paper['source']}")
            
            # 2. 检查向量块数量
            print(f"\n2. 向量块统计:")
            cur.execute("""
                SELECT 
                    paper_id,
                    COUNT(*) as chunk_count,
                    SUM(LENGTH(chunk_text)) as total_chars
                FROM paper_chunks
                GROUP BY paper_id
            """)
            chunk_stats = cur.fetchall()
            for stat in chunk_stats:
                print(f"   - {stat['paper_id'][:16]}... | {stat['chunk_count']} 块 | {stat['total_chars']} 字符")
            
            # 3. 检查是否有embedding
            print(f"\n3. 检查向量嵌入:")
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(embedding) as with_embedding,
                    COUNT(*) - COUNT(embedding) as without_embedding
                FROM paper_chunks
            """)
            embedding_stats = cur.fetchone()
            print(f"   总块数: {embedding_stats['total']}")
            print(f"   有向量: {embedding_stats['with_embedding']}")
            print(f"   无向量: {embedding_stats['without_embedding']}")
            
            # 4. 显示一些示例文本块
            print(f"\n4. 示例文本块（前3个块的前200字符）:")
            cur.execute("""
                SELECT 
                    paper_id,
                    chunk_index,
                    LEFT(chunk_text, 200) as text_preview
                FROM paper_chunks
                ORDER BY paper_id, chunk_index
                LIMIT 3
            """)
            samples = cur.fetchall()
            for sample in samples:
                print(f"\n   论文ID: {sample['paper_id'][:16]}...")
                print(f"   块索引: {sample['chunk_index']}")
                print(f"   内容预览: {sample['text_preview']}...")
            
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        print(f"检查失败: {str(e)}")
        import traceback
        traceback.print_exc()

def test_search(query: str, n_results: int = 10):
    """测试搜索功能"""
    print("\n" + "=" * 80)
    print(f"测试搜索: '{query}'")
    print("=" * 80)
    print()
    
    try:
        results = search_similar_chunks(query, n_results=n_results)
        
        if not results:
            print("❌ 未找到任何结果")
            print("\n可能的原因：")
            print("1. 数据库中没有向量嵌入数据")
            print("2. 查询文本与论文内容不匹配")
            print("3. 语言不匹配（中文查询 vs 英文论文）")
            return
        
        print(f"✅ 找到 {len(results)} 个结果:\n")
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            paper_title = metadata.get('paper_title', '未知标题')
            distance = result.get('distance', 0)
            similarity = 1 - distance  # distance越小，similarity越大
            
            print(f"[结果 {i}]")
            print(f"  论文: {paper_title}")
            print(f"  相似度: {similarity:.4f} (距离: {distance:.4f})")
            print(f"  内容预览: {result['content'][:200]}...")
            print()
        
        # 分析相似度分布
        similarities = [1 - r.get('distance', 1) for r in results]
        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        max_sim = max(similarities) if similarities else 0
        min_sim = min(similarities) if similarities else 0
        
        print(f"相似度统计:")
        print(f"  最高: {max_sim:.4f}")
        print(f"  最低: {min_sim:.4f}")
        print(f"  平均: {avg_sim:.4f}")
        
        if max_sim < 0.3:
            print("\n⚠️  警告: 最高相似度很低，可能的原因：")
            print("1. 查询文本与论文内容确实不相关")
            print("2. 语言不匹配（中文查询 vs 英文论文）")
            print("3. 关键词不匹配")
            print("\n建议：")
            print("- 尝试使用英文关键词搜索")
            print("- 尝试搜索论文中可能存在的术语")
            print("- 先查看论文列表，了解有哪些论文")
        
    except Exception as e:
        print(f"搜索测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

def test_keyword_search():
    """测试关键词搜索"""
    print("\n" + "=" * 80)
    print("关键词搜索测试")
    print("=" * 80)
    print()
    
    # 尝试不同的查询方式
    test_queries = [
        "CycleManip",
        "CycleManip framework",
        "framework",
        "cycle manipulation",
        "机器人",
        "robot",
        "control",
        "控制"
    ]
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        try:
            results = search_similar_chunks(query, n_results=3)
            if results:
                max_sim = max([1 - r.get('distance', 1) for r in results])
                print(f"  找到 {len(results)} 个结果，最高相似度: {max_sim:.4f}")
            else:
                print("  未找到结果")
        except Exception as e:
            print(f"  错误: {str(e)}")

if __name__ == '__main__':
    # 1. 检查数据库内容
    check_database_content()
    
    # 2. 测试原始查询
    test_search("CycleManip的框架", n_results=10)
    
    # 3. 测试关键词搜索
    test_keyword_search()
    
    print("\n" + "=" * 80)
    print("调试完成")
    print("=" * 80)

