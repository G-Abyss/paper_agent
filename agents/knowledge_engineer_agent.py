#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识工程师Agent

职责：
1. 接收来自数据摄取Agent的原始文本和元数据
2. 执行关键信息抽取（摘要、关键词、方法论、核心结论）
3. 将文本分块（Chunking）
4. 调用嵌入模型生成向量
5. 将向量、元数据、关联的原始文件路径一并存入PostgreSQL + pgVector
6. 为知识条目打上初步的来源标签（如"email"、"pdf"、"csv"）
"""

from crewai import Agent, Task, Crew
from crewai.tools import tool
from agents.base import get_llm
import os
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List
from utils.vector_db import (
    get_db_connection, 
    return_db_connection, 
    get_embedding_model,
    split_text_into_chunks
)
from psycopg2.extras import RealDictCursor, Json, execute_values


@tool("关键信息抽取工具")
def extract_key_information_tool(raw_text: str, metadata: Dict[str, Any], source_type: str) -> str:
    """
    从原始文本中抽取关键信息（摘要、关键词、方法论、核心结论）。
    输入参数：
    - raw_text: 原始文本内容
    - metadata: 已有的元数据字典（可能包含标题、作者等）
    - source_type: 数据源类型（'csv'、'pdf'、'email'）
    返回：JSON格式字符串，包含抽取的关键信息
    """
    import re
    
    result = {
        'abstract': metadata.get('abstract', ''),
        'keywords': metadata.get('keywords', ''),
        'methodology': '',
        'core_conclusions': '',
        'extracted_metadata': {}
    }
    
    try:
        # 如果元数据中已有摘要和关键词，直接使用
        if not result['abstract'] and raw_text:
            # 尝试从文本中提取摘要（前500字符作为摘要）
            result['abstract'] = raw_text[:500].strip()
        
        if not result['keywords'] and raw_text:
            # 尝试提取关键词（简单实现，后续可以用AI增强）
            # 查找常见的"Keywords:"、"关键词："等标记
            keywords_match = re.search(r'(?:keywords|关键词)[:：]\s*(.+?)(?:\n|$)', raw_text, re.I)
            if keywords_match:
                result['keywords'] = keywords_match.group(1).strip()
        
        # 提取方法论和核心结论（简单实现，后续可以用AI增强）
        if raw_text:
            # 查找"Method"、"方法"等部分
            method_match = re.search(r'(?:method|方法|methodology)[:：]?\s*(.+?)(?:\n\n|结论|conclusion|$)', raw_text, re.I | re.DOTALL)
            if method_match:
                result['methodology'] = method_match.group(1).strip()[:500]
            
            # 查找"Conclusion"、"结论"等部分
            conclusion_match = re.search(r'(?:conclusion|结论|summary|总结)[:：]?\s*(.+?)(?:\n\n|$)', raw_text, re.I | re.DOTALL)
            if conclusion_match:
                result['core_conclusions'] = conclusion_match.group(1).strip()[:500]
        
        result['extracted_metadata'] = {
            'source_type': source_type,
            'text_length': len(raw_text),
            'has_abstract': bool(result['abstract']),
            'has_keywords': bool(result['keywords']),
            'has_methodology': bool(result['methodology']),
            'has_conclusions': bool(result['core_conclusions'])
        }
        
    except Exception as e:
        logging.error(f"关键信息抽取失败: {str(e)}")
        result['error'] = str(e)
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool("文本分块工具")
def chunk_text_tool(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
    """
    将文本分块（Chunking），为向量化做准备。
    输入参数：
    - text: 要分块的文本
    - chunk_size: 每块的大小（字符数，默认1000）
    - chunk_overlap: 块之间的重叠大小（字符数，默认200）
    返回：JSON格式字符串，包含分块后的文本列表
    """
    try:
        chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        result = {
            'chunks': chunks,
            'chunk_count': len(chunks),
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logging.error(f"文本分块失败: {str(e)}")
        return json.dumps({
            'chunks': [],
            'chunk_count': 0,
            'error': str(e)
        }, ensure_ascii=False)


@tool("生成向量嵌入工具")
def generate_embeddings_tool(chunks: List[str]) -> str:
    """
    调用嵌入模型生成向量。
    输入参数：
    - chunks: 文本块列表
    返回：JSON格式字符串，包含向量列表
    """
    import numpy as np
    
    try:
        embedding_model = get_embedding_model()
        embeddings = embedding_model.encode(chunks, show_progress_bar=False, batch_size=32)
        
        # 转换为列表格式
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        
        result = {
            'embeddings': embeddings_list,
            'embedding_count': len(embeddings_list),
            'embedding_dimension': len(embeddings_list[0]) if embeddings_list else 0
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logging.error(f"生成向量嵌入失败: {str(e)}")
        return json.dumps({
            'embeddings': [],
            'embedding_count': 0,
            'error': str(e)
        }, ensure_ascii=False)


@tool("存储到向量数据库工具")
def store_to_vector_db_tool(
    paper_id: str,
    title: str,
    metadata: Dict[str, Any],
    file_path: str,
    source_tag: str,
    raw_text: str = ""
) -> str:
    """
    将论文或笔记的内容和元数据存入数据库。
    
    参数：
    - paper_id: 唯一ID
    - title: 标题
    - metadata: 元数据字典
    - file_path: 文件路径
    - source_tag: 来源标签
    - raw_text: 完整的原始文本内容 (入库必填)
    """
    try:
        from typing import List
        
        # 增加兜底：如果 raw_text 为空，尝试从 metadata 中获取
        if not raw_text and metadata:
            raw_text = metadata.get('raw_text', metadata.get('content', ''))

        # 内部处理 chunks 和 embeddings
        final_chunks: List[str] = []
        final_embeddings: List[List[float]] = []

        # --- 增强：自动从 raw_text 生成向量 ---
        if raw_text and not (final_chunks and final_embeddings):
            logging.info(f"--- [工具内部] 为 '{title}' 自动生成向量 ---")
            final_chunks = split_text_into_chunks(raw_text, chunk_size=1000, chunk_overlap=200)
            embedding_model = get_embedding_model()
            embeddings_raw = embedding_model.encode(final_chunks, show_progress_bar=False, batch_size=32)
            final_embeddings = [emb.tolist() for emb in embeddings_raw]
            logging.info(f"--- [工具内部] 生成了 {len(final_chunks)} 个分块 ---")
        elif not raw_text:
            logging.info(f"--- [工具内部] 笔记 '{title}' 正文为空，仅导入标题和元数据 ---")
            final_chunks = []
            final_embeddings = []

        conn = get_db_connection()
        cur = conn.cursor()
        
        # 准备元数据和 URL
        final_metadata = {
            **metadata,
            'tag': source_tag,
            'file_path': file_path,
            'source': source_tag
        }
        paper_url = metadata.get('link', metadata.get('url', ''))
        
        # --- 查重逻辑：根据 title 检查是否已存在 ---
        # 如果已存在相同标题的记录，获取其 paper_id 以便更新
        existing_paper_id = None
        if title:
            cur.execute("SELECT paper_id FROM papers WHERE title = %s LIMIT 1", (title,))
            row = cur.fetchone()
            if row:
                existing_paper_id = row[0]
                logging.info(f"检测到重复标题: '{title}'，将更新现有记录 (ID: {existing_paper_id})")

        # 如果标题存在，则使用现有的 paper_id 执行更新
        target_paper_id = existing_paper_id if existing_paper_id else paper_id

        # 插入或更新论文主表
        cur.execute("""
            INSERT INTO papers (paper_id, title, attachment_path, source, url, content, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (paper_id) DO UPDATE SET
                title = EXCLUDED.title,
                attachment_path = EXCLUDED.attachment_path,
                source = EXCLUDED.source,
                url = EXCLUDED.url,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """, (
            target_paper_id,
            title,
            file_path,
            source_tag,
            paper_url,
            raw_text,
            Json(final_metadata)
        ))
        
        # 删除旧的向量块（如果存在，使用 target_paper_id）
        cur.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (target_paper_id,))
        
        # 批量插入向量块
        if final_chunks and final_embeddings and len(final_chunks) == len(final_embeddings):
            chunk_data = []
            for i, (chunk, embedding) in enumerate(zip(final_chunks, final_embeddings)):
                # 清理文本中的NUL字符（PostgreSQL不允许）
                cleaned_chunk = chunk.replace('\x00', '').replace('\0', '')
                
                chunk_data.append((
                    target_paper_id,
                    i,
                    cleaned_chunk,
                    embedding,  # 已经是列表格式
                    len(cleaned_chunk),
                    Json({})  # 空的块元数据
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
        cur.close()
        return_db_connection(conn)
        
        result = {
            'success': True,
            'paper_id': target_paper_id,
            'chunks_stored': len(final_chunks),
            'message': f'成功存储到数据库: {title[:50]}...'
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logging.error(f"存储到向量数据库失败: {str(e)}", exc_info=True)
        return json.dumps({
            'success': False,
            'error': str(e)
        }, ensure_ascii=False)
        
    except Exception as e:
        logging.error(f"存储到向量数据库失败: {str(e)}", exc_info=True)
        return json.dumps({
            'success': False,
            'error': str(e)
        }, ensure_ascii=False)


def create_knowledge_engineer_agent():
    """
    创建知识工程师Agent
    
    Returns:
        Agent实例
    """
    return Agent(
        role="知识工程师",
        goal=(
            "接收来自数据摄取Agent的原始文本和元数据，执行关键信息抽取（摘要、关键词、方法论、核心结论），"
            "将文本分块（Chunking），调用嵌入模型生成向量，将向量、元数据、关联的原始文件路径一并存入PostgreSQL + pgVector，"
            "为知识条目打上初步的来源标签（如'email'、'pdf'、'csv'、'note'）。"
        ),
        backstory=(
            "你是一位专业的知识工程师，擅长从原始文本中提取关键信息，并将其结构化存储到知识库中。"
            "你了解学术论文和个人笔记的结构特点，能够准确提取关键信息。"
            "你熟悉文本分块技术，能够将长文本合理分割为适合向量化的块。"
            "你掌握向量嵌入技术，能够将文本转换为高维向量表示。"
            "你的工作是将非结构化的文本数据转换为结构化的、可检索的知识条目。"
        ),
        tools=[
            extract_key_information_tool,
            chunk_text_tool,
            generate_embeddings_tool,
            store_to_vector_db_tool
        ],
        allow_delegation=False,
        verbose=True,
        llm=get_llm(),
        max_iter=8,
        max_execution_time=1200
    )


def create_knowledge_engineering_task(
    raw_text: str,
    metadata: Dict[str, Any],
    file_path: str,
    source_type: str
) -> Task:
    """
    创建知识工程任务
    
    Args:
        raw_text: 原始文本内容
        metadata: 元数据字典
        file_path: 文件路径
        source_type: 数据源类型（'csv'、'pdf'、'email'）
        
    Returns:
        Task实例
    """
    file_name = os.path.basename(file_path)
    
    # 生成paper_id（基于文件路径的哈希）
    paper_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
    
    # 获取标题
    title = metadata.get('title', file_name)
    
    return Task(
        description=(
            f"## 任务目标\n"
            f"处理文件：{file_name}\n"
            f"文件路径：{file_path}\n"
            f"数据源类型：{source_type}\n\n"
            f"## 原始数据\n"
            f"### 元数据\n"
            f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
            f"### 原始文本（前2000字符）\n"
            f"{raw_text[:2000]}{'...' if len(raw_text) > 2000 else ''}\n\n"
            f"## 执行步骤\n"
            f"### 步骤1：关键信息抽取\n"
            f"- 使用'关键信息抽取工具'从原始文本中提取关键信息\n"
            f"- 提取摘要、关键词、方法论、核心结论\n"
            f"- 如果元数据中已有这些信息，优先使用元数据中的信息\n\n"
            f"### 步骤2：文本分块\n"
            f"- 使用'文本分块工具'将原始文本分块\n"
            f"- 块大小：1000字符，重叠：200字符\n"
            f"- 确保分块后的文本适合向量化\n\n"
            f"### 步骤3：生成向量嵌入\n"
            f"- 使用'生成向量嵌入工具'为每个文本块生成向量\n"
            f"- 确保向量数量与文本块数量一致\n\n"
            f"### 步骤4：存储到向量数据库\n"
            f"- 使用'存储到向量数据库工具'将数据存入PostgreSQL + pgVector\n"
            f"- Paper ID: {paper_id}\n"
            f"- 标题: {title}\n"
            f"- 来源标签: {source_type}\n"
            f"- 文件路径: {file_path}\n"
            f"- 确保向量、元数据、文件路径都正确存储\n\n"
            f"## 输出要求\n"
            f"输出格式必须是JSON，包含以下字段：\n"
            f"- success: 是否成功（true/false）\n"
            f"- paper_id: 论文唯一标识符\n"
            f"- chunks_stored: 存储的块数量\n"
            f"- message: 处理结果消息\n"
            f"- error: 错误信息（如果有）\n"
        ),
        agent=None,  # 将在创建Crew时指定
        expected_output="JSON格式的处理结果，包含存储状态和统计信息"
    )


def process_with_knowledge_engineer(
    raw_text: str,
    metadata: Dict[str, Any],
    file_path: str,
    source_type: str
) -> Dict[str, Any]:
    """
    使用知识工程师Agent处理数据
    """
    try:
        from crewai import Crew
        agent = create_knowledge_engineer_agent()
        
        # 预先生成 ID 和标题，确保一致性
        paper_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
        title = metadata.get('title', os.path.basename(file_path))
        
        if source_type in ['email', 'note']:
            # --- 使用 Agent 推理模式处理邮件或笔记数据 ---
            logging.info(f"--- [Agent 模式] 知识工程师开始工作 (类型: {source_type}, ID: {paper_id}) ---")
            
            task = create_knowledge_engineering_task(
                raw_text=raw_text,
                metadata=metadata,
                file_path=file_path,
                source_type=source_type
            )
            
            # 为不同源定制化任务描述
            if source_type == 'email':
                task.description = (
                    f"## 核心任务：邮件内容入库\n"
                    f"请直接将以下邮件信息存入数据库。\n\n"
                    f"## 1. 严格参数 (严禁修改)\n"
                    f"- **paper_id**: {paper_id}\n"
                    f"- **title**: {title}\n"
                    f"- **file_path**: {file_path}\n"
                    f"- **source_tag**: 'email'\n"
                    f"- **raw_text**: {raw_text}\n"
                    f"- **metadata**: {json.dumps(metadata, ensure_ascii=False)}\n\n"
                    f"## 2. 执行指令\n"
                    f"- 直接调用 '存储到向量数据库工具'。\n"
                    f"- 不要尝试分块或生成向量。\n\n"
                    f"## 3. 输出要求\n"
                    f"完成后返回 success: true 的 JSON。"
                )
            else:  # source_type == 'note'
                task.description = (
                    f"## 核心任务：笔记一键入库\n"
                    f"你现在需要将笔记原文存入数据库。不得仅以文字形式回答，必须调用工具执行！\n\n"
                    f"## 1. 待处理笔记原文 (必须作为 raw_text 传入)\n"
                    f"--- 内容开始 ---\n"
                    f"{raw_text}\n"
                    f"--- 内容结束 ---\n\n"
                    f"## 2. 严格要求的入库参数 (严禁修改)\n"
                    f"- **paper_id**: {paper_id}\n"
                    f"- **title**: {title}\n"
                    f"- **file_path**: {file_path}\n"
                    f"- **source_tag**: 'note'\n"
                    f"- **metadata**: {json.dumps(metadata, ensure_ascii=False)}\n\n"
                    f"## 3. 操作指南\n"
                    f"- **必须**调用 '存储到向量数据库工具'。\n"
                    f"- **raw_text**: 请务必提取上方 --- 内容开始 --- 之间的所有笔记正文，不得缺失，更不得抄写本指令中的提示语。\n"
                    f"- 不要进行分块或生成向量，工具内部会自动处理。\n\n"
                    f"## 4. 输出要求\n"
                    f"- 请使用 **中文** 进行最终总结。\n"
                    f"- 执行成功后，返回包含 success: true 的 JSON 结果。"
                )
            
            task.agent = agent
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                process='sequential'
            )
            
            result = crew.kickoff()
            
            # 解析 Agent 返回的结果
            try:
                output = result.raw if hasattr(result, 'raw') else str(result)
                
                # 尝试更健壮地解析 JSON
                def robust_json_parse(text):
                    import re
                    # 1. 尝试匹配 Markdown 代码块或最外层 JSON
                    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
                    if json_match:
                        try: 
                            res = json.loads(json_match.group(1).strip())
                            if isinstance(res, dict): return res
                        except: pass
                    return None

                parsed_res = robust_json_parse(output)
                if parsed_res and isinstance(parsed_res, dict) and parsed_res.get('success'):
                    return parsed_res
                
                # 宽容处理：只要 Agent 提到了成功执行（兼容中英文）
                success_keywords = ["存储到向量数据库工具", "成功", "success", "successful", "completed", "stored", "chunks"]
                if any(kw in output.lower() for kw in success_keywords):
                    return {'success': True, 'message': 'Agent 已完成处理', 'raw_output': output[:100]}
                
                return {'success': False, 'error': f'Agent 未能按预期返回结果: {output[:200]}...', 'raw_output': output}
            except Exception as e:
                logging.error(f"解析 Agent 结果时出错: {str(e)}")
                return {'success': False, 'error': f'解析 Agent 结果时出错: {str(e)}'}

        elif source_type == 'csv':
            # CSV文件处理：保持直接导入（通常CSV包含大量数据，Agent逐条处理太慢）
            logging.info(f"处理 CSV 数据: {file_path}")
            return process_csv_file(file_path, metadata)
            
        elif source_type == 'pdf':
            # PDF文件处理：目前由 process_and_store_pdf 逻辑处理
            logging.info(f"处理 PDF 文件: {file_path}")
            return process_pdf_file(file_path, metadata)
            
        else:
            logging.error(f"不支持的文件类型: {source_type}")
            return {
                'success': False,
                'error': f'不支持的文件类型: {source_type}',
                'paper_id': '',
                'title': metadata.get('title', ''),
                'chunks_stored': 0
            }
            
    except Exception as e:
        logging.error(f"知识工程师处理失败: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'paper_id': hashlib.md5(file_path.encode('utf-8')).hexdigest() if file_path else '',
            'title': metadata.get('title', ''),
            'chunks_stored': 0
        }


def process_csv_file(file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理CSV文件：直接将数据写入papers表
    
    Args:
        file_path: CSV文件路径
        metadata: 元数据字典（包含columns、rows等信息）
        
    Returns:
        处理结果字典
    """
    try:
        from utils.csv_importer import import_csv_to_database
        
        # 使用现有的CSV导入函数
        result = import_csv_to_database(file_path)
        
        if result.get('success', False):
            imported_count = result.get('imported_count', 0)
            return {
                'success': True,
                'paper_id': '',  # CSV导入可能包含多篇论文
                'title': os.path.basename(file_path),
                'chunks_stored': 0,  # CSV不需要分块
                'papers_imported': imported_count,
                'message': f'CSV文件处理成功，已导入 {imported_count} 篇论文'
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'CSV导入失败'),
                'paper_id': '',
                'title': os.path.basename(file_path),
                'chunks_stored': 0
            }
            
    except Exception as e:
        logging.error(f"处理CSV文件失败: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'paper_id': '',
            'title': os.path.basename(file_path),
            'chunks_stored': 0
        }


def process_pdf_file(file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理PDF文件：使用process_and_store_pdf流程
    
    Args:
        file_path: PDF文件路径
        metadata: 元数据字典（包含已提取的元数据）
        
    Returns:
        处理结果字典
    """
    try:
        from utils.vector_db import process_and_store_pdf, get_db_connection, return_db_connection, generate_paper_id
        
        # 使用现有的PDF处理函数
        success, message, pdf_metadata = process_and_store_pdf(
            pdf_path=file_path,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        if success and pdf_metadata:
            # 查找最终的文件路径（process_and_store_pdf会将文件复制到database目录并重命名）
            database_dir = 'database'
            title = pdf_metadata.get('title', '')
            
            # 尝试查找最终的文件路径
            final_pdf_path = None
            if title:
                from utils.file_utils import sanitize_filename
                safe_filename = sanitize_filename(title, max_length=200)
                # 先尝试不带时间戳的文件名
                potential_path = os.path.join(database_dir, f"{safe_filename}.pdf")
                if os.path.exists(potential_path):
                    final_pdf_path = potential_path
                else:
                    # 查找带时间戳的文件
                    if os.path.exists(database_dir):
                        for f in os.listdir(database_dir):
                            if f.startswith(safe_filename) and f.endswith('.pdf'):
                                final_pdf_path = os.path.join(database_dir, f)
                                break
            
            # 如果找不到，使用原始路径
            if not final_pdf_path:
                final_pdf_path = file_path
            
            # 生成paper_id（基于最终的文件路径）
            paper_id = generate_paper_id(final_pdf_path)
            
            # 获取存储的块数量
            conn = get_db_connection()
            chunk_count = 0
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM paper_chunks WHERE paper_id = %s", (paper_id,))
                result = cur.fetchone()
                chunk_count = result[0] if result else 0
            finally:
                return_db_connection(conn)
            
            return {
                'success': True,
                'paper_id': paper_id,
                'title': title,
                'chunks_stored': chunk_count,
                'message': message or f'PDF处理成功，已存储 {chunk_count} 个文本块',
                'metadata': pdf_metadata
            }
        else:
            return {
                'success': False,
                'error': message or 'PDF处理失败',
                'paper_id': '',
                'title': metadata.get('title', os.path.basename(file_path)),
                'chunks_stored': 0
            }
            
    except Exception as e:
        logging.error(f"处理PDF文件失败: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'paper_id': '',
            'title': metadata.get('title', os.path.basename(file_path)),
            'chunks_stored': 0
        }

