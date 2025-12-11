#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据迁移脚本：从ChromaDB迁移到PostgreSQL + pgvector

使用方法：
1. 确保PostgreSQL已安装并运行
2. 确保已安装pgvector扩展
3. 运行此脚本：python utils/migrate_chroma_to_postgres.py
"""

import os
import sys
import logging
from typing import List, Dict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def migrate_from_chromadb():
    """从ChromaDB迁移数据到PostgreSQL"""
    try:
        # 导入ChromaDB相关模块
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            logging.error("chromadb未安装，无法执行迁移")
            return False
        
        # 导入PostgreSQL相关模块
        try:
            from utils.vector_db import get_db_connection, return_db_connection, init_database, get_embedding_model
            from psycopg2.extras import execute_values, RealDictCursor
        except ImportError as e:
            logging.error(f"PostgreSQL相关模块未安装: {e}")
            return False
        
        # 初始化PostgreSQL数据库
        logging.info("初始化PostgreSQL数据库...")
        init_database()
        
        # 连接ChromaDB
        chroma_path = "data/chroma_db"
        if not os.path.exists(chroma_path):
            logging.warning(f"ChromaDB路径不存在: {chroma_path}，可能没有需要迁移的数据")
            return True
        
        logging.info(f"连接ChromaDB: {chroma_path}")
        chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            collection = chroma_client.get_collection(name="papers")
        except:
            logging.warning("ChromaDB中没有找到'papers'集合，可能没有需要迁移的数据")
            return True
        
        # 获取所有数据
        logging.info("从ChromaDB读取数据...")
        all_data = collection.get()
        
        if not all_data['ids'] or len(all_data['ids']) == 0:
            logging.info("ChromaDB中没有数据需要迁移")
            return True
        
        logging.info(f"找到 {len(all_data['ids'])} 个文档需要迁移")
        
        # 按paper_id分组
        papers_dict = {}
        for i, doc_id in enumerate(all_data['ids']):
            metadata = all_data['metadatas'][i] if all_data['metadatas'] else {}
            document = all_data['documents'][i] if all_data['documents'] else ""
            embedding = all_data['embeddings'][i] if all_data['embeddings'] else None
            
            paper_id = metadata.get('paper_id', '')
            if not paper_id:
                continue
            
            if paper_id not in papers_dict:
                papers_dict[paper_id] = {
                    'paper_id': paper_id,
                    'title': metadata.get('paper_title', '未知标题'),
                    'path': metadata.get('paper_path', ''),
                    'source': metadata.get('source', 'chromadb_migration'),
                    'chunks': []
                }
            
            papers_dict[paper_id]['chunks'].append({
                'chunk_index': metadata.get('chunk_index', len(papers_dict[paper_id]['chunks'])),
                'chunk_text': document,
                'embedding': embedding,
                'chunk_size': len(document)
            })
        
        logging.info(f"找到 {len(papers_dict)} 篇论文需要迁移")
        
        # 连接到PostgreSQL
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            
            migrated_papers = 0
            migrated_chunks = 0
            
            for paper_id, paper_data in papers_dict.items():
                try:
                    # 插入论文元数据
                    cur.execute("""
                        INSERT INTO papers (paper_id, title, attachment_path, source, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (paper_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            attachment_path = EXCLUDED.attachment_path,
                            source = EXCLUDED.source,
                            updated_at = NOW()
                    """, (
                        paper_data['paper_id'],
                        paper_data['title'],
                        paper_data['path'],
                        paper_data['source'],
                        '{}'  # 空的JSONB
                    ))
                    
                    # 检查是否已有chunks
                    cur.execute("SELECT COUNT(*) FROM paper_chunks WHERE paper_id = %s", (paper_id,))
                    existing_count = cur.fetchone()[0]
                    
                    if existing_count == 0:
                        # 批量插入chunks
                        chunk_data = []
                        for chunk in paper_data['chunks']:
                            if chunk['embedding']:
                                chunk_data.append((
                                    paper_id,
                                    chunk['chunk_index'],
                                    chunk['chunk_text'],
                                    chunk['embedding'],  # 已经是列表格式
                                    chunk['chunk_size'],
                                    '{}'  # 空的JSONB
                                ))
                        
                        if chunk_data:
                            execute_values(
                                cur,
                                """
                                INSERT INTO paper_chunks (paper_id, chunk_index, chunk_text, embedding, chunk_size, metadata)
                                VALUES %s
                                """,
                                chunk_data,
                                template=None,
                                page_size=100
                            )
                            migrated_chunks += len(chunk_data)
                    
                    migrated_papers += 1
                    
                    if migrated_papers % 10 == 0:
                        logging.info(f"已迁移 {migrated_papers}/{len(papers_dict)} 篇论文...")
                        conn.commit()
                        
                except Exception as e:
                    logging.error(f"迁移论文 {paper_id} 失败: {str(e)}")
                    conn.rollback()
                    continue
            
            conn.commit()
            logging.info(f"迁移完成！共迁移 {migrated_papers} 篇论文，{migrated_chunks} 个文本块")
            return True
            
        finally:
            return_db_connection(conn)
        
    except Exception as e:
        logging.error(f"迁移失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    print("=" * 80)
    print("ChromaDB 到 PostgreSQL 数据迁移工具")
    print("=" * 80)
    print()
    
    success = migrate_from_chromadb()
    
    if success:
        print("\n✅ 迁移成功完成！")
        print("\n提示：")
        print("1. 请验证PostgreSQL中的数据是否正确")
        print("2. 确认无误后，可以删除ChromaDB数据目录（data/chroma_db）")
    else:
        print("\n❌ 迁移失败，请查看日志了解详情")
        sys.exit(1)

