#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新处理PDF文件脚本

从papers表中读取已存储的论文，重新提取文本、生成向量并存储到paper_chunks表
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector_db import (
    get_db_connection, return_db_connection, 
    store_pdf_to_vector_db
)
from psycopg2.extras import RealDictCursor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def reprocess_pdfs():
    """重新处理所有已存储的PDF文件"""
    print("=" * 80)
    print("重新处理PDF文件")
    print("=" * 80)
    print()
    
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 1. 获取所有论文
            print("1. 查找需要处理的论文...")
            cur.execute("""
                SELECT 
                    paper_id,
                    title,
                    attachment_path,
                    source
                FROM papers
                ORDER BY created_at
            """)
            papers = cur.fetchall()
            
            if not papers:
                print("   未找到任何论文")
                return True
            
            print(f"   找到 {len(papers)} 篇论文\n")
            
            # 2. 检查哪些论文缺少向量块
            papers_to_process = []
            for paper in papers:
                paper_id = paper['paper_id']
                pdf_path = paper['attachment_path']
                
                # 检查是否有向量块
                cur.execute("SELECT COUNT(*) as count FROM paper_chunks WHERE paper_id = %s", (paper_id,))
                chunk_count = cur.fetchone()['count']
                
                # 检查PDF文件是否存在
                pdf_exists = pdf_path and os.path.exists(pdf_path) if pdf_path else False
                
                if chunk_count == 0:
                    if pdf_exists:
                        papers_to_process.append(paper)
                        print(f"   ⚠️  {paper['title'][:50]}... - 缺少向量块，PDF存在")
                    else:
                        print(f"   ❌ {paper['title'][:50]}... - 缺少向量块，PDF不存在: {pdf_path}")
                else:
                    print(f"   ✅ {paper['title'][:50]}... - 已有 {chunk_count} 个向量块")
            
            if not papers_to_process:
                print("\n✅ 所有论文都已正确处理，无需重新处理")
                return True
            
            print(f"\n2. 需要处理 {len(papers_to_process)} 篇论文\n")
            
            # 3. 重新处理每篇论文
            success_count = 0
            fail_count = 0
            
            for i, paper in enumerate(papers_to_process, 1):
                paper_id = paper['paper_id']
                pdf_path = paper['attachment_path']
                title = paper['title']
                
                print(f"[{i}/{len(papers_to_process)}] 处理: {title[:50]}...")
                print(f"   PDF路径: {pdf_path}")
                
                try:
                    # 删除旧的向量块（如果存在）
                    cur.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (paper_id,))
                    conn.commit()
                    
                    # 重新存储PDF
                    success = store_pdf_to_vector_db(
                        pdf_path=pdf_path,
                        paper_title=title,
                        metadata={
                            'source': paper['source'],
                            'reprocessed': True
                        }
                    )
                    
                    if success:
                        print(f"   ✅ 处理成功")
                        success_count += 1
                    else:
                        print(f"   ❌ 处理失败（查看日志了解详情）")
                        fail_count += 1
                        
                except Exception as e:
                    print(f"   ❌ 处理失败: {str(e)}")
                    logging.error(f"处理论文失败: {paper_id}, 错误: {str(e)}", exc_info=True)
                    fail_count += 1
                
                print()
            
            # 4. 总结
            print("=" * 80)
            print(f"处理完成！")
            print(f"  成功: {success_count} 篇")
            print(f"  失败: {fail_count} 篇")
            print("=" * 80)
            
            return fail_count == 0
            
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = reprocess_pdfs()
    sys.exit(0 if success else 1)

