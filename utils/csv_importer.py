#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV导入工具
支持表头映射和动态列创建
"""

import pandas as pd
import logging
import os
from typing import Dict, List, Optional, Tuple
from utils.vector_db import get_db_connection, return_db_connection
from psycopg2.extras import RealDictCursor, Json
# 表头映射规则
HEADER_MAPPING = {
    'title': ['论文标题', '标题', 'title'],
    'link': ['论文链接', '链接', 'link'],
    'abstract': ['论文摘要', '摘要', 'abstract'],
    'keywords': ['关键词', 'keywords'],
    'authors': ['作者', 'authors']
}


def normalize_header(header: str) -> Optional[str]:
    """
    将CSV表头标准化为数据库字段名
    
    Args:
        header: CSV原始表头
        
    Returns:
        标准化的字段名，如果不在映射规则中则返回None（表示自定义字段）
    """
    header_lower = header.strip().lower()
    
    for standard_field, possible_headers in HEADER_MAPPING.items():
        if header_lower in [h.lower() for h in possible_headers]:
            return standard_field
    
    # 不在映射规则中，返回原始表头（作为自定义字段）
    return header.strip()


def import_csv_to_database(csv_path: str) -> Dict:
    """
    将CSV文件导入数据库
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        {
            'success': bool,
            'imported_count': int,
            'error': str (如果失败)
        }
    """
    try:
        # 读取CSV文件
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'cp936']
        df = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                used_encoding = encoding
                logging.info(f"成功使用 {encoding} 编码读取CSV文件")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                if 'Unicode' in str(type(e).__name__):
                    continue
                if encoding == encodings[0]:
                    raise
        
        if df is None:
            return {
                'success': False,
                'error': '无法使用常见编码格式读取CSV文件'
            }
        
        # 获取表头映射
        header_mapping = {}
        custom_headers = []
        
        for col in df.columns:
            normalized = normalize_header(col)
            if normalized in HEADER_MAPPING.keys():
                header_mapping[col] = normalized
            else:
                # 自定义字段
                custom_headers.append(col)
                header_mapping[col] = col  # 使用原始表头作为字段名
        
        logging.info(f"表头映射: {header_mapping}")
        logging.info(f"自定义字段: {custom_headers}")
        
        # 获取导入tag
        try:
            import json
            import os
            settings_file = os.path.join(os.getcwd(), 'tag_settings.json')
            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    tag_mode = settings.get('tag_mode', 'default')
                    if tag_mode == 'custom':
                        import_tag = settings.get('custom_tag', 'csv')
                    else:
                        import_tag = 'csv'
            else:
                import_tag = 'csv'
        except Exception as e:
            logging.warning(f"获取导入tag失败，使用默认值: {str(e)}")
            import_tag = 'csv'
        
        # 连接数据库
        conn = get_db_connection()
        imported_count = 0
        
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 遍历每一行
            for idx, row in df.iterrows():
                try:
                    # 提取标准字段
                    title = None
                    link = None
                    abstract = None
                    keywords = None
                    authors = None
                    
                    # 根据映射提取字段
                    for csv_col, db_field in header_mapping.items():
                        value = str(row[csv_col]).strip() if pd.notna(row[csv_col]) else ''
                        
                        if db_field == 'title':
                            title = value
                        elif db_field == 'link':
                            link = value
                        elif db_field == 'abstract':
                            abstract = value
                        elif db_field == 'keywords':
                            keywords = [k.strip() for k in value.split(';') if k.strip()] if value else []
                        elif db_field == 'authors':
                            authors = [a.strip() for a in value.split(';') if a.strip()] if value else []
                    
                    # 跳过没有标题的行
                    if not title:
                        continue
                    
                    # 生成paper_id（基于标题）
                    import hashlib
                    paper_id = hashlib.md5(title.encode('utf-8')).hexdigest()
                    
                    # 构建metadata（包含自定义字段）
                    metadata = {
                        'tag': import_tag,
                        'source': 'csv_import'
                    }
                    
                    # 添加自定义字段到metadata（只添加非空值）
                    for csv_col in custom_headers:
                        value = str(row[csv_col]).strip() if pd.notna(row[csv_col]) else ''
                        if value:
                            metadata[csv_col] = value
                    
                    # 检查是否有title重名（通过title匹配，而不是paper_id）
                    cur.execute("""
                        SELECT paper_id, title, abstract, keywords, authors, metadata 
                        FROM papers 
                        WHERE title = %s
                    """, (title,))
                    existing = cur.fetchone()
                    
                    if existing:
                        # 存在重名论文，检查tag
                        existing_metadata = existing['metadata'] or {}
                        if isinstance(existing_metadata, str):
                            import json
                            existing_metadata = json.loads(existing_metadata) if existing_metadata else {}
                        elif not isinstance(existing_metadata, dict):
                            existing_metadata = {}
                        
                        existing_tag = existing_metadata.get('tag', '')
                        existing_paper_id = existing['paper_id']
                        
                        if existing_tag == 'mail':
                            # tag是mail，只覆盖空缺部分
                            logging.info(f"检测到重名论文（tag=mail），只覆盖空缺字段: {title[:50]}")
                            
                            update_fields = []
                            update_values = []
                            
                            # 检查并更新title（通常不会空缺，但检查一下）
                            if not existing.get('title'):
                                update_fields.append("title = %s")
                                update_values.append(title)
                            
                            # 检查并更新abstract（只覆盖空缺的）
                            if not existing.get('abstract'):
                                if abstract:  # 导入数据不为空才覆盖
                                    update_fields.append("abstract = %s")
                                    update_values.append(abstract)
                            
                            # 检查并更新keywords（只覆盖空缺的）
                            existing_keywords = existing.get('keywords', [])
                            if not existing_keywords or (isinstance(existing_keywords, list) and len(existing_keywords) == 0):
                                if keywords:  # 导入数据不为空才覆盖
                                    update_fields.append("keywords = %s")
                                    update_values.append(keywords)
                            
                            # 检查并更新authors（只覆盖空缺的）
                            existing_authors = existing.get('authors', [])
                            if not existing_authors or (isinstance(existing_authors, list) and len(existing_authors) == 0):
                                if authors:  # 导入数据不为空才覆盖
                                    update_fields.append("authors = %s")
                                    update_values.append(authors)
                            
                            # 更新metadata（保留原有数据，只更新tag和新的自定义字段）
                            merged_metadata = existing_metadata.copy()
                            merged_metadata['tag'] = import_tag  # 更新tag
                            # 只添加新的自定义字段（如果导入数据不为空）
                            for csv_col in custom_headers:
                                value = str(row[csv_col]).strip() if pd.notna(row[csv_col]) else ''
                                if value and csv_col not in merged_metadata:  # 只添加新的字段
                                    merged_metadata[csv_col] = value
                            
                            if update_fields or True:  # 至少更新metadata
                                update_fields.append("metadata = %s")
                                update_values.append(Json(merged_metadata))
                                update_fields.append("updated_at = NOW()")
                                update_values.append(existing_paper_id)
                                
                                query = f"UPDATE papers SET {', '.join(update_fields)} WHERE paper_id = %s"
                                cur.execute(query, update_values)
                        else:
                            # tag不是mail，全部覆盖（但只覆盖导入数据中不为空的数据）
                            logging.info(f"检测到重名论文（tag={existing_tag}），全部覆盖（仅非空字段）: {title[:50]}")
                            
                            update_fields = []
                            update_values = []
                            
                            # 更新title（总是更新）
                            update_fields.append("title = %s")
                            update_values.append(title)
                            
                            # 更新abstract（只覆盖导入数据中不为空的数据）
                            if abstract:
                                update_fields.append("abstract = %s")
                                update_values.append(abstract)
                            
                            # 更新keywords（只覆盖导入数据中不为空的数据）
                            if keywords:
                                update_fields.append("keywords = %s")
                                update_values.append(keywords)
                            
                            # 更新authors（只覆盖导入数据中不为空的数据）
                            if authors:
                                update_fields.append("authors = %s")
                                update_values.append(authors)
                            
                            # 更新metadata（保留原有数据，更新tag和自定义字段）
                            merged_metadata = existing_metadata.copy()
                            merged_metadata['tag'] = import_tag
                            # 只更新导入数据中不为空的自定义字段
                            for csv_col in custom_headers:
                                value = str(row[csv_col]).strip() if pd.notna(row[csv_col]) else ''
                                if value:  # 只更新非空值
                                    merged_metadata[csv_col] = value
                            
                            update_fields.append("metadata = %s")
                            update_values.append(Json(merged_metadata))
                            update_fields.append("updated_at = NOW()")
                            update_values.append(existing_paper_id)
                            
                            query = f"UPDATE papers SET {', '.join(update_fields)} WHERE paper_id = %s"
                            cur.execute(query, update_values)
                    else:
                        # 没有重名，正常导入
                        cur.execute("""
                            INSERT INTO papers (paper_id, title, abstract, keywords, authors, source, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            paper_id,
                            title,
                            abstract or '',
                            keywords or [],
                            authors or [],
                            'csv_import',
                            Json(metadata)
                        ))
                    
                    imported_count += 1
                    
                except Exception as e:
                    logging.error(f"导入CSV行 {idx + 2} 失败: {str(e)}", exc_info=True)
                    continue
            
            conn.commit()
            
            return {
                'success': True,
                'imported_count': imported_count
            }
            
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        logging.error(f"导入CSV到数据库失败: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def update_paper_processing_results(paper_id: str, translated_content: str, review: str, score: float) -> bool:
    """
    更新论文的处理结果（中英文摘要覆盖Abstract、评审结果、评审总分）
    
    Args:
        paper_id: 论文ID
        translated_content: 中英文摘要（将覆盖abstract字段）
        review: 评审结果
        score: 评审总分
        
    Returns:
        是否成功更新
    """
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 获取现有metadata
            cur.execute("SELECT metadata FROM papers WHERE paper_id = %s", (paper_id,))
            result = cur.fetchone()
            
            if not result:
                logging.warning(f"论文不存在: {paper_id}")
                return False
            
            metadata = result['metadata'] or {}
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata) if metadata else {}
            elif not isinstance(metadata, dict):
                metadata = {}
            
            # 更新处理结果字段（评审结果和评审总分存储在metadata中）
            metadata['评审结果'] = review
            metadata['评审总分'] = score
            
            # 更新数据库：abstract字段用中英文摘要覆盖，metadata存储评审结果和总分
            cur.execute("""
                UPDATE papers 
                SET abstract = %s, metadata = %s, updated_at = NOW()
                WHERE paper_id = %s
            """, (translated_content, Json(metadata), paper_id))
            
            conn.commit()
            return True
            
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        logging.error(f"更新论文处理结果失败: {str(e)}", exc_info=True)
        return False


def import_papers_from_email_csv(csv_path: str) -> Dict:
    """
    从邮箱模式生成的CSV文件导入论文到数据库
    
    Args:
        csv_path: CSV文件路径（表头格式：title, link, abstract, 评审结果）
        
    Returns:
        {
            'success': bool,
            'imported_count': int,
            'error': str (如果失败)
        }
    """
    try:
        import pandas as pd
        import hashlib
        import json
        
        # 读取CSV文件
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'cp936']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                if 'Unicode' in str(type(e).__name__):
                    continue
                if encoding == encodings[0]:
                    raise
        
        if df is None:
            return {
                'success': False,
                'error': '无法使用常见编码格式读取CSV文件'
            }
        
        # 获取导入tag
        try:
            import os
            settings_file = os.path.join(os.getcwd(), 'tag_settings.json')
            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    tag_mode = settings.get('tag_mode', 'default')
                    if tag_mode == 'custom':
                        import_tag = settings.get('custom_tag', 'mail')
                    else:
                        import_tag = 'mail'
            else:
                import_tag = 'mail'
        except Exception as e:
            logging.warning(f"获取导入tag失败，使用默认值: {str(e)}")
            import_tag = 'mail'
        
        # 连接数据库
        conn = get_db_connection()
        imported_count = 0
        
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 遍历每一行
            for idx, row in df.iterrows():
                try:
                    # 提取字段（支持表头映射）
                    title = None
                    link = None
                    abstract = None
                    review = None
                    
                    # 查找title字段
                    for col in df.columns:
                        col_lower = col.lower().strip()
                        if col_lower in ['title', '论文标题', '标题']:
                            title = str(row[col]).strip() if pd.notna(row[col]) else ''
                            break
                    
                    # 查找link字段
                    for col in df.columns:
                        col_lower = col.lower().strip()
                        if col_lower in ['link', '论文链接', '链接']:
                            link = str(row[col]).strip() if pd.notna(row[col]) else ''
                            break
                    
                    # 查找abstract字段
                    for col in df.columns:
                        col_lower = col.lower().strip()
                        if col_lower in ['abstract', '摘要', '论文摘要']:
                            abstract = str(row[col]).strip() if pd.notna(row[col]) else ''
                            break
                    
                    # 查找评审结果字段
                    for col in df.columns:
                        col_lower = col.lower().strip()
                        if col_lower in ['评审结果', 'review']:
                            review = str(row[col]).strip() if pd.notna(row[col]) else ''
                            break
                    
                    # 跳过没有标题的行
                    if not title:
                        continue
                    
                    # 生成paper_id（基于标题）
                    paper_id = hashlib.md5(title.encode('utf-8')).hexdigest()
                    
                    # 构建metadata
                    metadata = {
                        'tag': import_tag,
                        'source': 'email_import'
                    }
                    
                    if review:
                        metadata['评审结果'] = review
                    
                    # 检查是否有title重名（通过title匹配）
                    cur.execute("""
                        SELECT paper_id, title, abstract, metadata 
                        FROM papers 
                        WHERE title = %s
                    """, (title,))
                    existing = cur.fetchone()
                    
                    if existing:
                        # 存在重名论文，检查tag
                        existing_metadata = existing['metadata'] or {}
                        if isinstance(existing_metadata, str):
                            existing_metadata = json.loads(existing_metadata) if existing_metadata else {}
                        elif not isinstance(existing_metadata, dict):
                            existing_metadata = {}
                        
                        existing_tag = existing_metadata.get('tag', '')
                        existing_paper_id = existing['paper_id']
                        
                        if existing_tag == 'mail':
                            # tag是mail，只覆盖空缺部分
                            logging.info(f"检测到重名论文（tag=mail），只覆盖空缺字段: {title[:50]}")
                            
                            update_fields = []
                            update_values = []
                            
                            # 检查并更新title（通常不会空缺，但检查一下）
                            if not existing.get('title'):
                                update_fields.append("title = %s")
                                update_values.append(title)
                            
                            # 检查并更新abstract（只覆盖空缺的，且导入数据不为空）
                            if not existing.get('abstract'):
                                if abstract:  # 导入数据不为空才覆盖
                                    update_fields.append("abstract = %s")
                                    update_values.append(abstract)
                            
                            # 更新metadata（保留原有数据，只更新tag和新的字段）
                            merged_metadata = existing_metadata.copy()
                            merged_metadata['tag'] = import_tag  # 更新tag
                            if review:  # 只更新导入数据中不为空的评审结果
                                merged_metadata['评审结果'] = review
                            
                            if update_fields or True:  # 至少更新metadata
                                update_fields.append("metadata = %s")
                                update_values.append(Json(merged_metadata))
                                update_fields.append("updated_at = NOW()")
                                update_values.append(existing_paper_id)
                                
                                query = f"UPDATE papers SET {', '.join(update_fields)} WHERE paper_id = %s"
                                cur.execute(query, update_values)
                        else:
                            # tag不是mail，全部覆盖（但只覆盖导入数据中不为空的数据）
                            logging.info(f"检测到重名论文（tag={existing_tag}），全部覆盖（仅非空字段）: {title[:50]}")
                            
                            update_fields = []
                            update_values = []
                            
                            # 更新title（总是更新）
                            update_fields.append("title = %s")
                            update_values.append(title)
                            
                            # 更新abstract（只覆盖导入数据中不为空的数据）
                            if abstract:
                                update_fields.append("abstract = %s")
                                update_values.append(abstract)
                            
                            # 更新metadata（保留原有数据，只更新tag和导入数据中不为空的字段）
                            merged_metadata = existing_metadata.copy()
                            merged_metadata['tag'] = import_tag
                            if review:  # 只更新导入数据中不为空的评审结果
                                merged_metadata['评审结果'] = review
                            
                            update_fields.append("metadata = %s")
                            update_values.append(Json(merged_metadata))
                            update_fields.append("updated_at = NOW()")
                            update_values.append(existing_paper_id)
                            
                            query = f"UPDATE papers SET {', '.join(update_fields)} WHERE paper_id = %s"
                            cur.execute(query, update_values)
                    else:
                        # 没有重名，正常导入
                        cur.execute("""
                            INSERT INTO papers (paper_id, title, abstract, source, metadata)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (
                            paper_id,
                            title,
                            abstract or '',
                            'email_import',
                            Json(metadata)
                        ))
                    
                    imported_count += 1
                    
                except Exception as e:
                    logging.error(f"导入CSV行 {idx + 2} 失败: {str(e)}", exc_info=True)
                    continue
            
            conn.commit()
            
            return {
                'success': True,
                'imported_count': imported_count
            }
            
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        logging.error(f"导入邮箱CSV到数据库失败: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

