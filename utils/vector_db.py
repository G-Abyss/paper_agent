#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量数据库工具模块 - 使用PostgreSQL + pgvector存储和检索论文PDF内容

主要功能：
- 存储论文元数据和向量嵌入
- 支持语义搜索（RAG）
- 支持结构化查询（SQL）
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
import hashlib
import numpy as np
from datetime import datetime

# 延迟导入，避免在模块加载时出错
try:
    import psycopg2
    from psycopg2 import extras
    from psycopg2.extras import execute_values, RealDictCursor
    from psycopg2.pool import ThreadedConnectionPool
except ImportError:
    psycopg2 = None
    extras = None
    execute_values = None
    RealDictCursor = None
    ThreadedConnectionPool = None
    logging.warning("psycopg2未安装，请运行: pip install psycopg2-binary")

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logging.warning("PyMuPDF未正确安装，请运行: pip install PyMuPDF")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logging.warning("sentence-transformers未安装，请运行: pip install sentence-transformers")

# 全局变量
_db_pool = None
_embedding_model = None
_embedding_dimension = 384  # paraphrase-multilingual-MiniLM-L12-v2的维度

def get_embedding_model():
    """获取或初始化嵌入模型"""
    global _embedding_model
    if _embedding_model is None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers未安装，请运行: pip install sentence-transformers")
        try:
            # 使用多语言模型，支持中英文
            model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            logging.info(f"正在加载嵌入模型: {model_name}...")
            
            import os
            
            # 优先使用离线模式，避免每次检查远程更新
            # 只有在环境变量明确设置为在线模式时才允许网络请求
            use_offline = os.environ.get('TRANSFORMERS_OFFLINE', '1') == '1'
            use_hf_offline = os.environ.get('HF_HUB_OFFLINE', '1') == '1'
            
            # 如果环境变量未设置，默认使用离线模式（避免不必要的网络请求）
            if 'TRANSFORMERS_OFFLINE' not in os.environ:
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
            if 'HF_HUB_OFFLINE' not in os.environ:
                os.environ['HF_HUB_OFFLINE'] = '1'
            
            # 首先尝试离线模式（使用本地缓存）
            # 方法1: 尝试直接使用本地缓存路径（完全避免网络请求）
            model_loaded = False
            try:
                from pathlib import Path
                import platform
                
                # 获取缓存目录
                if platform.system() == 'Windows':
                    cache_base = Path(os.environ.get('USERPROFILE', '')) / '.cache' / 'huggingface' / 'hub'
                else:
                    cache_base = Path.home() / '.cache' / 'huggingface' / 'hub'
                
                # 查找模型目录
                model_dir_name = f'models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2'
                model_dir = cache_base / model_dir_name
                
                if model_dir.exists():
                    # 查找快照目录
                    snapshots_dir = model_dir / 'snapshots'
                    if snapshots_dir.exists():
                        snapshots = list(snapshots_dir.iterdir())
                        if snapshots:
                            local_model_path = snapshots[0]
                            logging.info(f"找到本地模型缓存: {local_model_path}")
                            # 直接使用本地路径加载，避免任何网络请求
                            _embedding_model = SentenceTransformer(str(local_model_path), device='cpu')
                            logging.info(f"已成功从本地缓存加载嵌入模型: {model_name}（无网络请求）")
                            model_loaded = True
                        else:
                            raise FileNotFoundError("快照目录为空")
                    else:
                        raise FileNotFoundError("未找到快照目录")
                else:
                    raise FileNotFoundError("未找到模型缓存目录")
            except (FileNotFoundError, ImportError) as local_path_error:
                logging.debug(f"直接路径加载失败: {local_path_error}，尝试其他方法...")
            
            # 方法2: 如果直接路径加载失败，尝试使用模型名称 + local_files_only
            if not model_loaded:
                try:
                    logging.info("尝试从本地缓存加载模型（强制离线模式）...")
                    # 设置更严格的离线模式
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'
                    os.environ['HF_HUB_OFFLINE'] = '1'
                    os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
                    
                    # 使用 local_files_only 参数（如果 SentenceTransformer 支持）
                    # 注意：某些版本可能不支持此参数，需要捕获异常
                    try:
                        _embedding_model = SentenceTransformer(model_name, device='cpu', local_files_only=True)
                        logging.info(f"已成功从本地缓存加载嵌入模型: {model_name}（使用 local_files_only）")
                    except TypeError:
                        # 如果不支持 local_files_only 参数，使用普通方式
                        # 但此时环境变量应该已经生效（可能仍会尝试网络请求）
                        logging.warning("当前版本不支持 local_files_only 参数，使用环境变量模式（可能仍会尝试网络请求）")
                        _embedding_model = SentenceTransformer(model_name, device='cpu')
                        logging.info(f"已成功从本地缓存加载嵌入模型: {model_name}")
                    model_loaded = True
                except Exception as offline_error:
                    # 如果离线模式失败（本地缓存不存在），尝试在线下载
                    error_str = str(offline_error).lower()
                    if 'not found' in error_str or 'cache' in error_str or 'local' in error_str:
                        logging.warning("本地缓存中未找到模型，尝试在线下载...")
                        try:
                            # 临时允许在线模式
                            os.environ['TRANSFORMERS_OFFLINE'] = '0'
                            os.environ['HF_HUB_OFFLINE'] = '0'
                            _embedding_model = SentenceTransformer(model_name, device='cpu')
                            logging.info(f"已成功下载并加载嵌入模型: {model_name}")
                            # 下载成功后，恢复离线模式设置
                            os.environ['TRANSFORMERS_OFFLINE'] = '1'
                            os.environ['HF_HUB_OFFLINE'] = '1'
                            model_loaded = True
                        except (OSError, ConnectionError) as network_error:
                            # 网络错误
                            error_msg = (
                                f"加载嵌入模型失败。\n"
                                f"原因: 本地缓存中未找到模型，且无法连接到 Hugging Face 下载模型。\n"
                                f"解决方案:\n"
                                f"1. 检查网络连接，确保可以访问 https://huggingface.co\n"
                                f"2. 手动下载模型: python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')\"\n"
                                f"3. 使用代理或镜像站（如果在中国大陆）\n"
                                f"4. 设置环境变量允许在线下载: TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0"
                            )
                            logging.error(error_msg)
                            raise OSError(error_msg) from network_error
                    else:
                        # 其他错误，直接抛出
                        error_msg = (
                            f"加载嵌入模型失败: {str(offline_error)}\n"
                            f"如果这是首次使用，请确保网络连接正常，或手动下载模型。"
                        )
                        logging.error(error_msg)
                        raise OSError(error_msg) from offline_error
            
            if not model_loaded:
                raise RuntimeError("未能加载嵌入模型")
        except Exception as e:
            if isinstance(e, (OSError, ConnectionError, ImportError, RuntimeError)):
                raise
            logging.error(f"加载嵌入模型时发生未知错误: {str(e)}")
            raise
    return _embedding_model

def get_db_connection():
    """获取数据库连接（从连接池）"""
    global _db_pool
    if _db_pool is None:
        if psycopg2 is None:
            raise ImportError("psycopg2未安装，请运行: pip install psycopg2-binary")
        try:
            from config import DB_URL
            # 创建连接池
            _db_pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=DB_URL,
                client_encoding='utf8'
            )
            logging.info("已创建PostgreSQL连接池")
        except Exception as e:
            logging.error(f"创建数据库连接池失败: {str(e)}")
            raise
    return _db_pool.getconn()

def return_db_connection(conn):
    """归还数据库连接到连接池"""
    global _db_pool
    if _db_pool:
        _db_pool.putconn(conn)

def init_database():
    """初始化数据库（创建表和扩展）"""
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            
            # 读取并执行初始化SQL
            sql_file = os.path.join(os.path.dirname(__file__), 'db_init.sql')
            if os.path.exists(sql_file):
                with open(sql_file, 'r', encoding='utf-8') as f:
                    sql = f.read()
                
                # --- 修复视图冲突：先删除可能存在冲突的视图 ---
                try:
                    cur.execute("DROP VIEW IF EXISTS paper_details CASCADE;")
                    cur.execute("DROP VIEW IF EXISTS paper_stats CASCADE;")
                    conn.commit()
                except Exception as e:
                    logging.warning(f"删除旧视图失败 (可能不存在): {e}")
                    conn.rollback()
                # ------------------------------------------

                # 移除注释行（以--开头的行）
                lines = []
                for line in sql.split('\n'):
                    # 保留包含--的行，但只移除纯注释行
                    stripped = line.strip()
                    if stripped and not stripped.startswith('--'):
                        lines.append(line)
                    elif not stripped.startswith('--'):
                        lines.append('')  # 保留空行以维持格式
                
                sql_cleaned = '\n'.join(lines)
                
                # 智能分割SQL语句
                # 匹配 $$...$$ 块（用于函数定义）
                # 然后按分号分割，但要排除 $$ 块内的分号
                statements = []
                in_dollar_quote = False
                dollar_quote_tag = None
                buffer = ''
                
                i = 0
                while i < len(sql_cleaned):
                    char = sql_cleaned[i]
                    
                    # 检查是否进入或退出 $$ 引号块
                    if char == '$' and i + 1 < len(sql_cleaned):
                        # 检查是否是 $$ 的开始或结束
                        j = i + 1
                        tag = '$'
                        while j < len(sql_cleaned) and sql_cleaned[j] == '$':
                            tag += '$'
                            j += 1
                        
                        if tag == dollar_quote_tag:
                            # 结束 $$ 块
                            buffer += sql_cleaned[i:j]
                            in_dollar_quote = False
                            dollar_quote_tag = None
                            i = j - 1
                        elif not in_dollar_quote and j > i + 1:
                            # 开始 $$ 块
                            dollar_quote_tag = tag
                            in_dollar_quote = True
                            buffer += sql_cleaned[i:j]
                            i = j - 1
                        else:
                            buffer += char
                    elif char == ';' and not in_dollar_quote:
                        # 在 $$ 块外遇到分号，这是一个语句的结束
                        buffer += char
                        statement = buffer.strip()
                        if statement and not statement.startswith('--'):
                            statements.append(statement)
                        buffer = ''
                    else:
                        buffer += char
                    
                    i += 1
                
                # 添加最后一个语句（如果有）
                if buffer.strip() and not buffer.strip().startswith('--'):
                    statements.append(buffer.strip())
                
                # 执行每个语句
                for statement in statements:
                    try:
                        if statement.strip():
                            cur.execute(statement)
                    except Exception as stmt_error:
                        # 如果是已存在的错误，忽略它
                        error_msg = str(stmt_error)
                        if ('already exists' in error_msg.lower() or 
                            'duplicate' in error_msg.lower() or
                            '触发器已经存在' in error_msg or
                            'does not exist' in error_msg.lower() and 'DROP' in statement.upper()):
                            logging.debug(f"跳过已存在的对象或DROP不存在的对象: {error_msg[:100]}")
                            continue
                        else:
                            # 其他错误继续抛出
                            raise
                
                # 检查并添加 content 列（如果不存在）
                try:
                    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='papers' AND column_name='content';")
                    if not cur.fetchone():
                        logging.info("正在为 papers 表添加 content 列...")
                        cur.execute("ALTER TABLE papers ADD COLUMN content TEXT;")
                        conn.commit()
                        logging.info("已成功添加 content 列")
                except Exception as e:
                    logging.warning(f"检查或添加 content 列失败: {str(e)}")
                    conn.rollback()

                conn.commit()
                logging.info("数据库初始化成功")
            else:
                logging.warning(f"未找到数据库初始化文件: {sql_file}")
        finally:
            return_db_connection(conn)
    except Exception as e:
        # 如果是触发器已存在的错误，只记录警告，不抛出异常
        error_msg = str(e)
        if '触发器已经存在' in error_msg or 'already exists' in error_msg.lower() or 'duplicate' in error_msg.lower():
            logging.warning(f"数据库初始化检查失败（可能已初始化）: {error_msg}")
        else:
            logging.error(f"数据库初始化失败: {error_msg}")
            raise

def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    从PDF文件提取文本内容
    
    Args:
        pdf_path: PDF文件路径
        max_pages: 最大提取页数，如果为None则提取所有页面
    
    Returns:
        提取的文本内容
    """
    if fitz is None:
        raise ImportError("PyMuPDF未正确安装，请运行: pip install PyMuPDF")
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        pages_to_extract = min(max_pages, len(doc)) if max_pages else len(doc)
        for page_num in range(pages_to_extract):
            page = doc[page_num]
            full_text += page.get_text()
        doc.close()
        logging.info(f"从PDF提取文本成功: {pdf_path}, 共 {pages_to_extract} 页, {len(full_text)} 字符")
        return full_text
    except Exception as e:
        logging.error(f"提取PDF文本失败: {pdf_path}, 错误: {str(e)}")
        raise

def extract_pdf_title(pdf_path: str) -> str:
    """
    从PDF文件提取论文标题
    
    Args:
        pdf_path: PDF文件路径
    
    Returns:
        提取的论文标题，如果提取失败则返回None
    """
    try:
        # 提取前3页文本（通常标题在前3页）
        text_preview = extract_text_from_pdf(pdf_path, max_pages=3)
        
        # 限制预览文本长度（避免超出模型上下文）
        preview_length = 3000
        if len(text_preview) > preview_length:
            text_preview = text_preview[:preview_length] + "..."
        
        # 使用agent提取标题
        from crewai import Crew
        from agents.pdf_title_agent import create_pdf_title_extractor_agent, create_pdf_title_extraction_task
        
        agent = create_pdf_title_extractor_agent()
        task = create_pdf_title_extraction_task(text_preview)
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False  # 不显示详细日志
        )
        
        result = crew.kickoff()
        
        # 提取标题文本
        if hasattr(result, 'raw'):
            title = result.raw.strip()
        elif isinstance(result, str):
            title = result.strip()
        else:
            title = str(result).strip()
        
        # 清理标题
        title = title.strip('"\'')  # 去除引号
        title = title.strip()  # 去除首尾空白
        
        # 如果标题为空或为"未知标题"，返回None
        if not title or title.lower() in ['未知标题', 'unknown title', 'untitled']:
            return None
        
        logging.info(f"从PDF提取标题成功: {title[:50]}...")
        return title
        
    except Exception as e:
        logging.warning(f"提取PDF标题失败: {pdf_path}, 错误: {str(e)}")
        return None

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    将文本分割成块
    
    Args:
        text: 原始文本
        chunk_size: 每块的大小（字符数），默认1000
        chunk_overlap: 块之间的重叠字符数，默认200（20%重叠）
    
    Returns:
        文本块列表
    """
    if not text:
        return []
    
    # 清理文本中的NUL字符（PostgreSQL不允许）
    text = text.replace('\x00', '').replace('\0', '')
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 移动到下一个块，考虑重叠
        start = end - chunk_overlap
        if start >= text_length:
            break
    
    logging.info(f"文本已分割为 {len(chunks)} 个块")
    return chunks

def generate_paper_id(pdf_path: str) -> str:
    """生成论文的唯一ID（基于文件路径的哈希）"""
    return hashlib.md5(pdf_path.encode()).hexdigest()

def copy_and_rename_pdf(source_path: str, target_title: str, database_dir: str = "database") -> Optional[str]:
    """
    复制PDF文件到database目录并重命名
    
    Args:
        source_path: 源PDF文件路径
        target_title: 目标文件名（论文标题）
        database_dir: 目标目录，默认"database"
    
    Returns:
        新的文件路径，如果失败则返回None
    """
    try:
        import shutil
        from utils.file_utils import sanitize_filename
        
        # 确保目标目录存在
        os.makedirs(database_dir, exist_ok=True)
        
        # 清理标题作为文件名
        safe_filename = sanitize_filename(target_title, max_length=200)
        target_path = os.path.join(database_dir, f"{safe_filename}.pdf")
        
        # 如果文件已存在，添加时间戳
        if os.path.exists(target_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = os.path.join(database_dir, f"{safe_filename}_{timestamp}.pdf")
        
        # 复制文件
        shutil.copy2(source_path, target_path)
        
        # 验证文件是否成功复制
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            logging.info(f"PDF已复制并重命名: {source_path} -> {target_path}")
            return target_path
        else:
            logging.error(f"PDF复制失败: {target_path}")
            return None
            
    except Exception as e:
        logging.error(f"复制PDF文件失败: {str(e)}")
        return None

def process_and_store_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[bool, str, Optional[Dict]]:
    """
    处理PDF文件：复制、分析元数据、重命名，然后存储到数据库
    
    Args:
        pdf_path: 源PDF文件路径
        chunk_size: 文本块大小（字符数），默认1000
        chunk_overlap: 文本块重叠大小（字符数），默认200
    
    Returns:
        (success, message, metadata_dict)
        - success: 是否成功
        - message: 成功或错误消息
        - metadata_dict: 提取的元数据（如果成功），包含title、authors、abstract、keywords
    """
    try:
        # 步骤1：先复制PDF到database目录（使用临时文件名）
        logging.info("步骤1：复制PDF到database目录...")
        temp_filename = f"temp_{os.path.basename(pdf_path)}"
        database_dir = "database"
        os.makedirs(database_dir, exist_ok=True)
        temp_path = os.path.join(database_dir, temp_filename)
        
        import shutil
        shutil.copy2(pdf_path, temp_path)
        
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return False, "PDF复制失败：文件不存在或大小为0", None
        
        logging.info(f"PDF已复制到临时位置: {temp_path}")
        
        # 步骤2：使用agent分析PDF元数据
        logging.info("步骤2：使用agent分析PDF元数据...")
        from agents.pdf_metadata_agent import extract_pdf_metadata
        
        pdf_metadata = extract_pdf_metadata(temp_path)
        
        if not pdf_metadata:
            # 清理临时文件
            try:
                os.remove(temp_path)
            except:
                pass
            return False, "PDF元数据提取失败：无法从PDF中提取标题、摘要等信息", None
        
        # 验证标题是否提取成功
        if not pdf_metadata.get('title') or pdf_metadata['title'] == '未知标题':
            # 清理临时文件
            try:
                os.remove(temp_path)
            except:
                pass
            return False, "PDF元数据提取失败：无法提取论文标题", None
        
        # 确保所有必需字段都存在（如果agent没有检测到，添加默认值）
        if 'abstract' not in pdf_metadata or not pdf_metadata.get('abstract'):
            pdf_metadata['abstract'] = ''
        if 'keywords' not in pdf_metadata or not pdf_metadata.get('keywords'):
            pdf_metadata['keywords'] = ''
        if 'authors' not in pdf_metadata or not pdf_metadata.get('authors'):
            pdf_metadata['authors'] = ''
        
        logging.info(f"PDF元数据提取成功: 标题={pdf_metadata['title'][:50]}...")
        
        # 步骤3：使用提取的标题重命名PDF
        logging.info("步骤3：使用提取的标题重命名PDF...")
        from utils.file_utils import sanitize_filename
        
        safe_filename = sanitize_filename(pdf_metadata['title'], max_length=200)
        final_path = os.path.join(database_dir, f"{safe_filename}.pdf")
        
        # 如果文件已存在，添加时间戳
        if os.path.exists(final_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(database_dir, f"{safe_filename}_{timestamp}.pdf")
        
        # 重命名文件
        try:
            os.rename(temp_path, final_path)
            logging.info(f"PDF已重命名: {temp_path} -> {final_path}")
        except Exception as e:
            # 清理临时文件
            try:
                os.remove(temp_path)
            except:
                pass
            return False, f"PDF重命名失败: {str(e)}", None
        
        # 验证重命名后的文件
        if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            return False, "PDF重命名失败：重命名后的文件不存在或大小为0", None
        
        # 步骤4：检查数据库是否有重名论文
        logging.info("步骤4：检查数据库是否有重名论文...")
        conn = get_db_connection()
        try:
            from psycopg2.extras import RealDictCursor
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 检查是否有重名论文（通过title匹配）
            title = pdf_metadata.get('title', '').strip()
            cur.execute("""
                SELECT paper_id, title, abstract, keywords, authors, metadata, attachment_path
                FROM papers 
                WHERE title = %s
            """, (title,))
            existing_paper = cur.fetchone()
            
            # 获取导入tag
            try:
                from web_server import get_import_tag
                import_tag = get_import_tag('pdf')
            except:
                import_tag = 'pdf'
            
            # 如果存在重名论文，根据tag规则处理
            if existing_paper:
                existing_metadata = existing_paper.get('metadata', {})
                if isinstance(existing_metadata, str):
                    import json
                    existing_metadata = json.loads(existing_metadata) if existing_metadata else {}
                elif not isinstance(existing_metadata, dict):
                    existing_metadata = {}
                
                existing_tag = existing_metadata.get('tag', '')
                
                if existing_tag == 'mail':
                    # tag是mail，只覆盖空缺字段
                    logging.info(f"检测到重名论文（tag=mail），只覆盖空缺字段")
                    
                    # 准备更新的字段（只更新空缺的）
                    update_fields = []
                    update_values = []
                    
                    # 检查并更新title（通常不会空缺，但检查一下）
                    if not existing_paper.get('title'):
                        update_fields.append("title = %s")
                        update_values.append(title)
                    
                    # 检查并更新abstract（只覆盖空缺的，且导入数据不为空）
                    if not existing_paper.get('abstract'):
                        abstract_value = pdf_metadata.get('abstract', '')
                        if abstract_value:  # 导入数据不为空才覆盖
                            update_fields.append("abstract = %s")
                            update_values.append(abstract_value)
                    
                    # 检查并更新keywords
                    existing_keywords = existing_paper.get('keywords', [])
                    if not existing_keywords or (isinstance(existing_keywords, list) and len(existing_keywords) == 0):
                        keywords_str = pdf_metadata.get('keywords', '')
                        keywords_list = [k.strip() for k in keywords_str.split(',') if k.strip()] if keywords_str else []
                        if keywords_list:
                            update_fields.append("keywords = %s")
                            update_values.append(keywords_list)
                    
                    # 检查并更新authors
                    existing_authors = existing_paper.get('authors', [])
                    if not existing_authors or (isinstance(existing_authors, list) and len(existing_authors) == 0):
                        authors_str = pdf_metadata.get('authors', '')
                        authors_list = [a.strip() for a in authors_str.split(',') if a.strip()] if authors_str else []
                        if authors_list:
                            update_fields.append("authors = %s")
                            update_values.append(authors_list)
                    
                    # 更新attachment_path（总是更新，因为PDF是新上传的）
                    update_fields.append("attachment_path = %s")
                    update_values.append(final_path)
                    
                    # 更新metadata（保留原有metadata，只更新tag）
                    merged_metadata = existing_metadata.copy()
                    merged_metadata['tag'] = import_tag
                    update_fields.append("metadata = %s")
                    update_values.append(extras.Json(merged_metadata) if extras else merged_metadata)
                    update_fields.append("updated_at = NOW()")
                    update_values.append(existing_paper['paper_id'])
                    
                    if update_fields:
                        query = f"UPDATE papers SET {', '.join(update_fields)} WHERE paper_id = %s"
                        cur.execute(query, update_values)
                        conn.commit()
                        logging.info(f"已更新重名论文的空缺字段: {title[:50]}")
                    
                    # 存储PDF文本块（删除旧的，添加新的）
                    paper_id = existing_paper['paper_id']
                    cur.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (paper_id,))
                    
                    # 提取PDF文本并存储
                    text = extract_text_from_pdf(final_path)
                    if text:
                        chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        if chunks:
                            try:
                                embedding_model = get_embedding_model()
                                embeddings = embedding_model.encode(chunks, show_progress_bar=True, batch_size=32)
                                
                                chunk_data = []
                                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                                    cleaned_chunk = chunk.replace('\x00', '').replace('\0', '')
                                    chunk_data.append((
                                        paper_id,
                                        i,
                                        cleaned_chunk,
                                        embedding.tolist(),
                                        len(cleaned_chunk),
                                        extras.Json({}) if extras else {}
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
                                    conn.commit()
                                    logging.info(f"已更新PDF文本块: {len(chunks)} 个块")
                            except Exception as e:
                                logging.warning(f"生成嵌入向量失败: {str(e)}")
                    
                    return_db_connection(conn)
                    return True, f"PDF已成功处理并更新（只覆盖空缺字段）: {title}", pdf_metadata
                else:
                    # tag不是mail，全部覆盖（但只覆盖导入数据中不为空的数据）
                    logging.info(f"检测到重名论文（tag={existing_tag}），全部覆盖（仅非空字段）")
                    paper_id = existing_paper['paper_id']
                    
                    # 删除旧的文本块
                    cur.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (paper_id,))
                    
                    # 准备更新的字段（只覆盖导入数据中不为空的数据）
                    update_fields = []
                    update_values = []
                    
                    # 更新title（总是更新）
                    update_fields.append("title = %s")
                    update_values.append(title)
                    
                    # 更新abstract（只覆盖导入数据中不为空的数据）
                    abstract_value = pdf_metadata.get('abstract', '')
                    if abstract_value:
                        update_fields.append("abstract = %s")
                        update_values.append(abstract_value)
                    
                    # 更新keywords（只覆盖导入数据中不为空的数据）
                    keywords_str = pdf_metadata.get('keywords', '')
                    if keywords_str:
                        keywords_list = [k.strip() for k in keywords_str.split(',') if k.strip()] if keywords_str else []
                        if keywords_list:
                            update_fields.append("keywords = %s")
                            update_values.append(keywords_list)
                    
                    # 更新authors（只覆盖导入数据中不为空的数据）
                    authors_str = pdf_metadata.get('authors', '')
                    if authors_str:
                        authors_list = [a.strip() for a in authors_str.split(',') if a.strip()] if authors_str else []
                        if authors_list:
                            update_fields.append("authors = %s")
                            update_values.append(authors_list)
                    
                    # 更新attachment_path（总是更新，因为PDF是新上传的）
                    update_fields.append("attachment_path = %s")
                    update_values.append(final_path)
                    
                    # 更新metadata（保留原有数据，只更新tag和导入数据中不为空的字段）
                    merged_metadata = existing_metadata.copy()
                    merged_metadata['tag'] = import_tag
                    merged_metadata['source'] = 'web_upload'
                    merged_metadata['upload_time'] = str(datetime.now())
                    # 只更新导入数据中不为空的字段
                    if abstract_value:
                        merged_metadata['abstract'] = abstract_value
                    if keywords_str:
                        merged_metadata['keywords'] = keywords_str
                    if authors_str:
                        merged_metadata['authors'] = authors_str
                    
                    update_fields.append("metadata = %s")
                    update_values.append(extras.Json(merged_metadata) if extras else merged_metadata)
                    update_fields.append("updated_at = NOW()")
                    update_values.append(paper_id)
                    
                    query = f"UPDATE papers SET {', '.join(update_fields)} WHERE paper_id = %s"
                    cur.execute(query, update_values)
                    conn.commit()
                    
                    # 提取PDF文本并存储
                    text = extract_text_from_pdf(final_path)
                    if text:
                        chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        if chunks:
                            try:
                                embedding_model = get_embedding_model()
                                embeddings = embedding_model.encode(chunks, show_progress_bar=True, batch_size=32)
                                
                                chunk_data = []
                                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                                    cleaned_chunk = chunk.replace('\x00', '').replace('\0', '')
                                    chunk_data.append((
                                        paper_id,
                                        i,
                                        cleaned_chunk,
                                        embedding.tolist(),
                                        len(cleaned_chunk),
                                        extras.Json({}) if extras else {}
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
                                    conn.commit()
                                    logging.info(f"已更新PDF文本块: {len(chunks)} 个块")
                            except Exception as e:
                                logging.warning(f"生成嵌入向量失败: {str(e)}")
                    
                    return_db_connection(conn)
                    return True, f"PDF已成功处理并覆盖: {title}", pdf_metadata
            else:
                # 没有重名论文，正常存储
                return_db_connection(conn)
                logging.info("步骤5：存储PDF到向量数据库...")
                
                success = store_pdf_to_vector_db(
                    pdf_path=final_path,
                    paper_title=pdf_metadata['title'],
                    metadata={
                        'authors': pdf_metadata.get('authors', ''),
                        'abstract': pdf_metadata.get('abstract', ''),
                        'keywords': pdf_metadata.get('keywords', ''),
                        'source': 'web_upload',
                        'upload_time': str(datetime.now()),
                        'tag': import_tag
                    },
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    extract_title=False,  # 已经提取了，不需要再次提取
                    copy_to_database=False  # 已经在database目录了，不需要再次复制
                )
                
                if success:
                    return True, f"PDF已成功处理并存储: {pdf_metadata['title']}", pdf_metadata
                else:
                    return False, "PDF存储到数据库失败", None
                    
        except Exception as e:
            logging.error(f"检查重名论文失败: {str(e)}", exc_info=True)
            return_db_connection(conn)
            # 如果检查失败，尝试正常存储
            try:
                from web_server import get_import_tag
                import_tag = get_import_tag('pdf')
            except:
                import_tag = 'pdf'
            
            success = store_pdf_to_vector_db(
                pdf_path=final_path,
                paper_title=pdf_metadata['title'],
                metadata={
                    'authors': pdf_metadata.get('authors', ''),
                    'abstract': pdf_metadata.get('abstract', ''),
                    'keywords': pdf_metadata.get('keywords', ''),
                    'source': 'web_upload',
                    'upload_time': str(datetime.now()),
                    'tag': import_tag
                },
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                extract_title=False,
                copy_to_database=False
            )
            
            if success:
                return True, f"PDF已成功处理并存储: {pdf_metadata['title']}", pdf_metadata
            else:
                return False, "PDF存储到数据库失败", None
            
    except Exception as e:
        logging.error(f"处理PDF失败: {str(e)}", exc_info=True)
        # 清理临时文件
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass
        return False, f"处理PDF失败: {str(e)}", None


def store_pdf_to_vector_db(pdf_path: str, paper_title: Optional[str] = None, metadata: Optional[Dict] = None, chunk_size: int = 1000, chunk_overlap: int = 200, extract_title: bool = False, copy_to_database: bool = False) -> bool:
    """
    将PDF论文存储到向量数据库
    
    Args:
        pdf_path: PDF文件路径
        paper_title: 论文标题（可选，如果为None且extract_title=True，则从PDF提取）
        metadata: 额外的元数据（可选）
        chunk_size: 文本块大小（字符数），默认1000
        chunk_overlap: 文本块重叠大小（字符数），默认200
        extract_title: 是否从PDF提取标题，默认True
        copy_to_database: 是否复制PDF到database目录并重命名，默认True
    
    Returns:
        是否成功存储
    """
    try:
        # 初始化数据库（如果尚未初始化）
        try:
            init_database()
        except Exception as init_error:
            logging.warning(f"数据库初始化检查失败（可能已初始化）: {str(init_error)}")
        
        # 1. 提取论文标题（如果需要）
        final_paper_title = paper_title
        if extract_title and not final_paper_title:
            logging.info("正在从PDF提取论文标题...")
            extracted_title = extract_pdf_title(pdf_path)
            if extracted_title:
                final_paper_title = extracted_title
                logging.info(f"已提取论文标题: {final_paper_title[:50]}...")
            else:
                logging.warning("无法从PDF提取标题，使用文件名作为标题")
                final_paper_title = os.path.splitext(os.path.basename(pdf_path))[0]
        elif not final_paper_title:
            final_paper_title = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 2. 复制PDF到database目录并重命名（如果需要）
        final_pdf_path = pdf_path
        if copy_to_database and final_paper_title:
            logging.info(f"正在复制PDF到database目录并重命名...")
            database_path = copy_and_rename_pdf(pdf_path, final_paper_title)
            if database_path:
                final_pdf_path = database_path
                logging.info(f"PDF已复制到: {database_path}")
            else:
                logging.warning("PDF复制失败，使用原始路径")
        
        # 3. 提取文本（使用最终的文件路径）
        text = extract_text_from_pdf(final_pdf_path)
        if not text:
            logging.warning(f"PDF文件为空或无法提取文本: {final_pdf_path}")
            return False
        
        # 分割文本
        chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            logging.warning(f"文本分割后为空: {pdf_path}")
            return False
        
        # 生成嵌入向量
        try:
            embedding_model = get_embedding_model()
            logging.info(f"开始生成嵌入向量，共 {len(chunks)} 个文本块...")
            embeddings = embedding_model.encode(chunks, show_progress_bar=True, batch_size=32)
            logging.info(f"嵌入向量生成完成，形状: {embeddings.shape}")
        except (OSError, ConnectionError) as e:
            error_msg = (
                f"生成嵌入向量失败: {str(e)}\n"
                f"这通常是由于无法连接到 Hugging Face 下载模型导致的。\n"
                f"请检查网络连接或手动下载模型。"
            )
            logging.error(error_msg)
            raise OSError(error_msg) from e
        
        # 4. 准备数据（使用最终的文件路径生成ID）
        paper_id = generate_paper_id(final_pdf_path)
        conn = get_db_connection()
        
        try:
            cur = conn.cursor()
            
            # 检查论文是否已存在
            cur.execute("SELECT id FROM papers WHERE paper_id = %s", (paper_id,))
            existing_paper = cur.fetchone()
            
            if existing_paper:
                # 如果已存在，先删除旧的块
                logging.warning(f"论文已存在（ID: {paper_id}），删除旧数据后重新添加")
                cur.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (paper_id,))
                cur.execute("DELETE FROM papers WHERE paper_id = %s", (paper_id,))
            
            # 插入论文元数据（使用最终的文件路径和标题）
            # 清理标题和路径中的NUL字符（PostgreSQL不允许）
            cleaned_title = final_paper_title.replace('\x00', '').replace('\0', '') if final_paper_title else None
            cleaned_path = final_pdf_path.replace('\x00', '').replace('\0', '') if final_pdf_path else None
            
            # 准备metadata，添加tag
            final_metadata = metadata.copy() if metadata else {}
            # 如果metadata中没有tag，添加默认tag（由调用者传入）
            if 'tag' not in final_metadata:
                final_metadata['tag'] = metadata.get('tag', 'pdf') if metadata else 'pdf'
            
            # 如果论文已存在，检查是否需要更新tag（用最新的覆盖）
            if existing_paper:
                # 保留原有的metadata，但更新tag
                cur.execute("SELECT metadata FROM papers WHERE paper_id = %s", (paper_id,))
                old_result = cur.fetchone()
                if old_result and old_result[0]:
                    old_metadata = old_result[0] if isinstance(old_result[0], dict) else {}
                    # 合并metadata，但tag使用新的
                    final_metadata = {**old_metadata, **final_metadata, 'tag': final_metadata.get('tag', 'pdf')}
            
            cur.execute("""
                INSERT INTO papers (paper_id, title, attachment_path, source, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (paper_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    attachment_path = EXCLUDED.attachment_path,
                    source = EXCLUDED.source,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """, (
                paper_id,
                cleaned_title,
                cleaned_path,
                final_metadata.get('source', 'web_upload'),
                extras.Json(final_metadata) if extras else final_metadata
            ))
            
            # 批量插入向量块
            chunk_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # 清理文本中的NUL字符（PostgreSQL不允许）
                cleaned_chunk = chunk.replace('\x00', '').replace('\0', '')
                
                chunk_data.append((
                    paper_id,
                    i,
                    cleaned_chunk,
                    embedding.tolist(),  # 转换为列表
                    len(cleaned_chunk),
                    extras.Json({}) if extras else {}  # 空的元数据
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
            else:
                logging.error(f"没有向量块数据需要插入: {pdf_path}")
                conn.rollback()
                return False
            
            conn.commit()
            logging.info(f"PDF已成功存储到向量数据库: {final_pdf_path}, 标题: {final_paper_title}, 共 {len(chunks)} 个块")
            return True
            
        except Exception as db_error:
            conn.rollback()
            logging.error(f"数据库操作失败: {str(db_error)}")
            raise
        finally:
            return_db_connection(conn)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"存储PDF到向量数据库失败: {pdf_path}")
        logging.error(f"错误类型: {type(e).__name__}")
        logging.error(f"错误信息: {str(e)}")
        logging.error(f"详细堆栈:\n{error_details}")
        return False

def search_similar_chunks(query: str, n_results: int = 5, paper_id: Optional[str] = None, source: Optional[str] = None) -> List[Dict]:
    """
    在向量数据库中搜索相似的文本块
    
    Args:
        query: 查询文本
        n_results: 返回结果数量
        paper_id: 可选的论文ID，用于限制搜索范围
        source: 可选的来源过滤，例如 'note' 表示只搜索笔记，'!note' 表示排除笔记
    
    Returns:
        相似文本块列表，每个包含：content, metadata, distance
    """
    try:
        embedding_model = get_embedding_model()
        
        # 生成查询向量
        query_embedding = embedding_model.encode([query], show_progress_bar=False)[0]
        
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 构建SQL查询
            where_conditions = []
            params = [query_embedding.tolist()]
            
            if paper_id:
                where_conditions.append("pc.paper_id = %s")
                params.append(paper_id)
            
            # 处理source过滤
            if source == 'note':
                where_conditions.append("p.source = 'note'")
            elif source == '!note':
                where_conditions.append("p.source != 'note'")
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            sql = f"""
                SELECT 
                    pc.chunk_text as content,
                    pc.chunk_index,
                    pc.paper_id,
                    p.title as paper_title,
                    p.attachment_path as paper_path,
                    p.source,
                    1 - (pc.embedding <=> %s::vector) as distance
                FROM paper_chunks pc
                JOIN papers p ON pc.paper_id = p.paper_id
                {where_clause}
                ORDER BY pc.embedding <=> %s::vector
                LIMIT %s
            """
            params.extend([query_embedding.tolist(), n_results])
            cur.execute(sql, params)
            
            results = cur.fetchall()
            
            # 格式化结果
            similar_chunks = []
            for row in results:
                chunk = {
                    'content': row['content'],
                    'metadata': {
                        'paper_id': row['paper_id'],
                        'paper_title': row['paper_title'],
                        'paper_path': row['paper_path'],
                        'chunk_index': row['chunk_index'],
                        'source': row.get('source', 'unknown')
                    },
                    'distance': float(row['distance'])
                }
                similar_chunks.append(chunk)
            
            # 记录详细的搜索信息
            if similar_chunks:
                similarities = [1 - chunk['distance'] for chunk in similar_chunks]
                logging.info(f"搜索完成，查询: {query[:50]}..., 返回 {len(similar_chunks)} 个结果")
                logging.info(f"相似度范围: {min(similarities):.4f} - {max(similarities):.4f}, 平均: {sum(similarities)/len(similarities):.4f}")
            else:
                logging.warning(f"搜索完成，查询: {query[:50]}..., 但未返回任何结果")
                # 检查是否有数据
                cur.execute("SELECT COUNT(*) as count FROM paper_chunks")
                total_chunks = cur.fetchone()['count']
                logging.warning(f"数据库中总共有 {total_chunks} 个向量块")
                if total_chunks == 0:
                    logging.error("数据库中没有向量块数据，可能未正确存储PDF")
            
            return similar_chunks
            
        finally:
            return_db_connection(conn)
        
    except Exception as e:
        logging.error(f"向量数据库搜索失败: {str(e)}")
        return []

def get_paper_list() -> List[Dict]:
    """获取所有已存储的论文列表（包含完整元数据）"""
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT DISTINCT ON (paper_id)
                    paper_id,
                    title as paper_title,
                    authors,
                    abstract,
                    content,
                    keywords,
                    source,
                    attachment_path as paper_path,
                    metadata,
                    think_points,
                    contextual_summary,
                    created_at,
                    updated_at
                FROM papers
                ORDER BY paper_id, created_at DESC
            """)
            results = cur.fetchall()
            
            paper_list = []
            for row in results:
                # 解析metadata（如果是JSONB）
                metadata = row.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                elif not isinstance(metadata, dict):
                    metadata = {}
                
                paper_list.append({
                    'paper_id': row['paper_id'],
                    'paper_title': row['paper_title'] or '未知标题',
                    'title': row['paper_title'] or '未知标题',
                    'authors': ', '.join(row['authors']) if isinstance(row.get('authors'), list) else (row.get('authors') or ''),
                    'abstract': row.get('abstract') or metadata.get('abstract', ''),
                    'content': row.get('content') or '',
                    'keywords': ', '.join(row['keywords']) if isinstance(row.get('keywords'), list) else (row.get('keywords') or ''),
                    'source': row.get('source') or '未知来源',
                    'paper_path': row['paper_path'] or '',
                    'metadata': metadata,
                    'think_points': row.get('think_points'),
                    'contextual_summary': row.get('contextual_summary') or '',
                    'created_at': str(row['created_at']) if row.get('created_at') else '',
                    'updated_at': str(row['updated_at']) if row.get('updated_at') else ''
                })
            
            logging.info(f"获取到 {len(paper_list)} 篇论文")
            return paper_list
            
        finally:
            return_db_connection(conn)
        
    except Exception as e:
        logging.error(f"获取论文列表失败: {str(e)}")
        return []

def delete_paper(paper_id: str) -> bool:
    """从向量数据库中删除指定论文的所有块"""
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            # 由于外键约束，删除papers记录会自动删除相关的chunks
            cur.execute("DELETE FROM papers WHERE paper_id = %s", (paper_id,))
            deleted_count = cur.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logging.info(f"已删除论文: {paper_id}")
                return True
            else:
                logging.warning(f"未找到论文: {paper_id}")
                return False
                
        except Exception as e:
            conn.rollback()
            logging.error(f"删除论文失败: {paper_id}, 错误: {str(e)}")
            return False
        finally:
            return_db_connection(conn)
    except Exception as e:
        logging.error(f"删除论文失败: {paper_id}, 错误: {str(e)}")
        return False

