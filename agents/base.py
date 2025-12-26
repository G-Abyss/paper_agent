#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent基础配置和工具
"""

from crewai.tools import tool
from config import llm
from utils.file_utils import extract_real_url_from_redirect, is_pdf_url
from typing import Optional, List, Dict
import requests
import re
import os
import logging
from datetime import datetime

def get_llm():
    """获取当前激活的 LLM 实例"""
    try:
        from utils.model_config import get_active_model
        from utils.llm_utils import create_llm_from_model_config
        
        active_model = get_active_model()
        if active_model:
            llm_instance = create_llm_from_model_config(active_model)
            if llm_instance:
                return llm_instance
    except Exception as e:
        logging.error(f"动态获取 LLM 失败: {e}")
    
    # 备选方案：返回 config.py 中的默认 llm
    return llm


@tool("网页内容获取工具")
def fetch_webpage_content(url: str) -> str:
    """
    从论文URL获取网页内容。当邮件片段信息不足时，可以使用此工具获取论文网页的详细内容（包括摘要、介绍等）来辅助判断相关性。输入参数：url (字符串，论文的URL)
    """
    try:
        # 先提取真实URL（处理跳转链接）
        from utils.file_utils import extract_real_url_from_redirect
        real_url = extract_real_url_from_redirect(url)
        if real_url:
            url = real_url
        
        # 使用requests获取网页内容
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        try:
            # 缩短超时时间到 15 秒，避免长时间阻塞
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logging.warning(f"获取网页内容超时（15秒）: {url}")
            return f"获取失败：连接超时（15秒）。可能是由于网络限制或目标网站无法访问。"
        except requests.exceptions.RequestException as e:
            logging.warning(f"获取网页内容失败: {url}, 错误: {str(e)}")
            return f"获取失败：网络错误 {str(e)}"
        
        # 使用BeautifulSoup提取文本
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        # 移除script和style标签
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        
        # 限制长度，返回前5000字符
        return text[:5000] if len(text) > 5000 else text
    except Exception as e:
        return f"获取网页内容失败: {str(e)}"


@tool("arXiv论文下载工具")
def arxiv_download_tool(paper_url: str, paper_title: str = "") -> str:
    """
    检查论文URL是否为arXiv论文，如果是则自动下载PDF并提取文本内容（前3页）。当检测到论文可能来自arXiv时，可以使用此工具获取论文的完整内容来辅助判断相关性。输入参数：paper_url (字符串，论文的URL)，paper_title (可选，字符串，论文标题，用于保存PDF文件)
    
    注意：此工具会自动处理Google Scholar等跳转链接，提取真实的arXiv URL。
    """
    try:
        # 提取真实URL（处理跳转链接，特别是Google Scholar的跳转链接）
        from utils.file_utils import extract_real_url_from_redirect, sanitize_filename
        real_url = extract_real_url_from_redirect(paper_url)
        if real_url:
            logging.info(f"从跳转链接提取到真实URL: {real_url}")
            paper_url = real_url
        
        # 检查是否为arXiv URL
        arxiv_pattern = r'arxiv\.org/(?:pdf|abs)/(\d+\.\d+)'
        match = re.search(arxiv_pattern, paper_url, re.I)
        
        if not match:
            return "不是arXiv论文"
        
        arxiv_id = match.group(1)
        pdf_bytes = None
        arxiv_title = ""
        
        # 直接使用requests下载，避免arxiv库可能的卡住问题
        arxiv_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_bytes = None
        arxiv_title = ""
        
        # 尝试获取论文元数据（用于获取标题），但设置超时保护
        try:
            import arxiv
            import threading
            import queue
            
            # 使用线程和队列来实现超时保护
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def fetch_metadata():
                try:
                    client = arxiv.Client()
                    search = arxiv.Search(id_list=[arxiv_id])
                    paper = next(client.results(search), None)
                    if paper:
                        result_queue.put(paper)
                    else:
                        result_queue.put(None)
                except Exception as e:
                    exception_queue.put(e)
            
            # 启动线程获取元数据
            metadata_thread = threading.Thread(target=fetch_metadata, daemon=True)
            metadata_thread.start()
            metadata_thread.join(timeout=10)  # 10秒超时
            
            if metadata_thread.is_alive():
                logging.warning(f"获取arXiv元数据超时（超过10秒）: {arxiv_id}")
            else:
                # 检查是否有结果
                try:
                    paper = result_queue.get_nowait()
                    if paper:
                        arxiv_title = paper.title if hasattr(paper, 'title') else ""
                except queue.Empty:
                    pass
                # 检查是否有异常
                try:
                    exc = exception_queue.get_nowait()
                    logging.debug(f"获取arXiv元数据时出错: {str(exc)}")
                except queue.Empty:
                    pass
        except ImportError:
            # arxiv库未安装，跳过元数据获取
            pass
        except Exception as e:
            logging.debug(f"获取arXiv元数据失败: {str(e)}")
        
        # 直接使用requests下载PDF（更可靠，支持超时设置）
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            # 缩短PDF下载超时到 30 秒
            response = requests.get(arxiv_pdf_url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            if response.content[:4] == b'%PDF':
                pdf_bytes = response.content
            else:
                logging.warning(f"下载的arXiv PDF文件无效（不是PDF格式）: {arxiv_pdf_url}")
                return "下载的PDF文件无效"
        except requests.exceptions.Timeout:
            logging.warning(f"下载arXiv PDF超时（超过180秒）: {arxiv_pdf_url}")
            return "下载PDF失败: 下载超时（超过180秒），已跳过"
        except requests.exceptions.RequestException as e:
            logging.warning(f"下载arXiv PDF失败: {arxiv_pdf_url}, 错误: {str(e)}")
            return f"下载PDF失败: {str(e)}"
        
        # 保存PDF到downloads文件夹（使用统一的保存函数）
        pdf_filename = None
        if pdf_bytes:
            from utils.file_utils import save_pdf_to_downloads
            # 优先使用传入的论文标题，否则使用arXiv标题
            save_title = paper_title if paper_title else arxiv_title
            pdf_filename = save_pdf_to_downloads(pdf_bytes, paper_title=save_title, url=paper_url)
        
        # 提取文本（添加超时保护，避免卡住）
        if pdf_bytes:
            try:
                import fitz  # PyMuPDF
                import threading
                import queue
                
                # 使用线程和队列来实现超时保护
                text_queue = queue.Queue()
                exception_queue = queue.Queue()
                
                def extract_text():
                    try:
                        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        full_text = ""
                        for page_num in range(min(3, len(pdf_doc))):  # 只提取前3页
                            page = pdf_doc[page_num]
                            full_text += page.get_text()
                        pdf_doc.close()
                        text_queue.put(full_text)
                    except Exception as e:
                        exception_queue.put(e)
                
                # 启动线程提取文本
                extract_thread = threading.Thread(target=extract_text, daemon=True)
                extract_thread.start()
                extract_thread.join(timeout=30)  # 30秒超时
                
                if extract_thread.is_alive():
                    logging.warning(f"提取PDF文本超时（超过30秒）: {arxiv_id}")
                    full_text = "PDF文本提取超时，但PDF文件已下载"
                else:
                    # 检查是否有结果
                    try:
                        full_text = text_queue.get_nowait()
                    except queue.Empty:
                        # 检查是否有异常
                        try:
                            exc = exception_queue.get_nowait()
                            logging.warning(f"提取PDF文本时出错: {str(exc)}")
                            full_text = f"PDF文本提取失败: {str(exc)}"
                        except queue.Empty:
                            full_text = "PDF文本提取失败"
                
                # 限制长度，并在返回文本中包含PDF文件路径信息（如果保存成功）
                result_text = full_text[:5000] if len(full_text) > 5000 else full_text
                if pdf_filename:
                    # 在文本末尾添加PDF路径信息（agent可以读取）
                    result_text += f"\n[PDF已保存到: {pdf_filename}]"
                return result_text
            except Exception as e:
                logging.warning(f"提取PDF文本时出错: {str(e)}")
                result_text = f"PDF文本提取失败: {str(e)}"
                if pdf_filename:
                    result_text += f"\n[PDF已保存到: {pdf_filename}]"
                return result_text
        else:
            return "无法获取PDF内容"
    except Exception as e:
        return f"处理arXiv论文失败: {str(e)}"


# 创建工具对象引用（保持向后兼容）
fetch_webpage_tool = fetch_webpage_content


@tool("RAG论文查询工具")
def rag_paper_query_tool(query: str, n_results: Optional[int] = None, paper_id: str = "") -> str:
    """
    使用RAG（检索增强生成）在已存储的论文中查询相关信息。
    
    输入参数：
    - query (字符串，必需): 用户的问题或查询内容
    - n_results (整数，可选): 返回结果数量，默认5。如果不提供，将使用默认值5。
    - paper_id (字符串，可选): 指定论文ID，如果为空字符串则搜索所有论文
    """
    try:
        from utils.vector_db import search_similar_chunks
        
        # 如果未提供n_results，使用默认值5
        if n_results is None:
            n_results = 5
        
        # 搜索相似文本块（如果paper_id为空字符串，则搜索所有论文）
        results = search_similar_chunks(query, n_results=n_results, paper_id=paper_id if paper_id else None)
        
        if not results:
            # 提供更详细的错误信息
            from utils.vector_db import get_paper_list
            papers = get_paper_list()
            if not papers:
                return f"未找到与查询 '{query}' 相关的论文内容。数据库中没有存储任何论文，请先上传PDF论文。"
            else:
                return f"未找到与查询 '{query}' 相关的论文内容。\n\n提示：\n- 当前数据库中有 {len(papers)} 篇论文\n- 可以尝试使用英文关键词搜索\n- 或者先使用'获取论文列表工具'查看有哪些论文"
        
        # 格式化结果
        response_parts = [f"找到 {len(results)} 个相关片段：\n"]
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            paper_title = metadata.get('paper_title', '未知标题')
            paper_path = metadata.get('paper_path', '')
            chunk_index = metadata.get('chunk_index', 0)
            distance = result.get('distance', 0)
            
            response_parts.append(f"\n[片段 {i}] (相关性: {1-distance:.3f})")
            response_parts.append(f"论文: {paper_title}")
            response_parts.append(f"内容: {result['content'][:500]}...")
            if i < len(results):
                response_parts.append("---")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logging.error(f"RAG查询失败: {str(e)}")
        return f"RAG查询出错: {str(e)}"


@tool("查询已知知识工具")
def query_known_knowledge_tool(query: str, n_results: Optional[int] = None) -> str:
    """
    在"已知知识"（标签为note的笔记）中查询相关信息。
    这是用户已经掌握的知识，应该优先使用。
    
    输入参数：
    - query (字符串，必需): 用户的问题或查询内容
    - n_results (整数，可选): 返回结果数量，默认5。如果不提供，将使用默认值5。
    """
    try:
        from utils.vector_db import search_similar_chunks
        
        # 如果未提供n_results，使用默认值5
        if n_results is None:
            n_results = 5
        
        # 只在source='note'的笔记中搜索
        results = search_similar_chunks(query, n_results=n_results, source='note')
        
        if not results:
            return f"在已知知识（笔记）中未找到与查询 '{query}' 相关的内容。"
        
        # 格式化结果
        response_parts = [f"在已知知识中找到 {len(results)} 个相关片段：\n"]
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            paper_title = metadata.get('paper_title', '未知标题')
            chunk_index = metadata.get('chunk_index', 0)
            distance = result.get('distance', 0)
            
            response_parts.append(f"\n[片段 {i}] (相关性: {1-distance:.3f})")
            response_parts.append(f"笔记: {paper_title}")
            response_parts.append(f"内容: {result['content'][:500]}...")
            if i < len(results):
                response_parts.append("---")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logging.error(f"查询已知知识失败: {str(e)}")
        return f"查询已知知识出错: {str(e)}"


@tool("查询未知知识工具")
def query_unknown_knowledge_tool(query: str, n_results: Optional[int] = None) -> str:
    """
    在"未知知识"（非note标签的论文）中查询相关信息。
    这是用户尚未掌握的知识，用于补充和扩展理解。
    
    输入参数：
    - query (字符串，必需): 用户的问题或查询内容
    - n_results (整数，可选): 返回结果数量，默认5。如果不提供，将使用默认值5。
    """
    try:
        from utils.vector_db import search_similar_chunks
        
        # 如果未提供n_results，使用默认值5
        if n_results is None:
            n_results = 5
        
        # 只在source!='note'的论文中搜索
        results = search_similar_chunks(query, n_results=n_results, source='!note')
        
        if not results:
            return f"在未知知识（论文库）中未找到与查询 '{query}' 相关的内容。"
        
        # 格式化结果
        response_parts = [f"在未知知识中找到 {len(results)} 个相关片段：\n"]
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            paper_title = metadata.get('paper_title', '未知标题')
            paper_path = metadata.get('paper_path', '')
            chunk_index = metadata.get('chunk_index', 0)
            distance = result.get('distance', 0)
            
            response_parts.append(f"\n[片段 {i}] (相关性: {1-distance:.3f})")
            response_parts.append(f"论文: {paper_title}")
            if paper_path:
                response_parts.append(f"路径: {paper_path}")
            response_parts.append(f"内容: {result['content'][:500]}...")
            if i < len(results):
                response_parts.append("---")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logging.error(f"查询未知知识失败: {str(e)}")
        return f"查询未知知识出错: {str(e)}"


@tool("获取论文Think点工具")
def get_papers_think_points_tool(max_results: Optional[int] = None) -> str:
    """
    获取数据库中所有笔记（已知知识）的think点（思考点）。
    这些think点反映了用户已知知识中可能感兴趣的研究方向。
    
    输入参数：
    - max_results (整数，可选): 返回结果数量，默认返回所有。如果不提供，将返回所有笔记的think点。
    
    返回：
    - 包含笔记标题和think点的格式化字符串
    """
    try:
        from utils.vector_db import get_db_connection
        from psycopg2.extras import RealDictCursor
        
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 查询所有note来源的笔记，且think_points不为空的记录
            sql = """
                SELECT paper_id, title, think_points 
                FROM papers 
                WHERE source = 'note' 
                  AND think_points IS NOT NULL 
                  AND think_points != '[]'::jsonb
                  AND jsonb_array_length(think_points) > 0
                ORDER BY updated_at DESC
            """
            
            if max_results:
                sql += f" LIMIT {max_results}"
            
            cur.execute(sql)
            papers = cur.fetchall()
            
            if not papers:
                return "数据库中暂无包含think点的笔记（已知知识）。"
            
            # 格式化结果
            response_parts = [f"找到 {len(papers)} 条包含think点的笔记（已知知识）：\n"]
            for i, paper in enumerate(papers, 1):
                title = paper.get('title', '未知标题')
                think_points = paper.get('think_points', [])
                
                if isinstance(think_points, str):
                    import json
                    try:
                        think_points = json.loads(think_points)
                    except:
                        think_points = []
                
                response_parts.append(f"\n[{i}] {title}")
                if think_points and isinstance(think_points, list):
                    for j, point in enumerate(think_points, 1):
                        if isinstance(point, str):
                            response_parts.append(f"   Think点 {j}: {point[:200]}...")
                response_parts.append("---")
            
            return "\n".join(response_parts)
            
        finally:
            from utils.vector_db import return_db_connection
            return_db_connection(conn)
            
    except Exception as e:
        logging.error(f"获取think点失败: {str(e)}")
        return f"获取think点出错: {str(e)}"

@tool("获取论文列表工具")
def get_paper_list_tool() -> str:
    """
    获取向量数据库中所有已存储的论文列表，包括论文数量、标题和ID。
    这个工具用于回答关于论文库中论文数量、论文列表等问题。
    
    不需要输入参数。
    """
    try:
        from utils.vector_db import get_paper_list
        
        papers = get_paper_list()
        
        if not papers:
            return "当前论文库中没有存储任何论文。"
        
        # 格式化结果
        response_parts = [f"论文库中共有 {len(papers)} 篇论文：\n"]
        for i, paper in enumerate(papers, 1):
            paper_id = paper.get('paper_id', '未知ID')
            paper_title = paper.get('paper_title', '未知标题')
            paper_path = paper.get('paper_path', '')
            
            response_parts.append(f"\n[{i}] {paper_title}")
            response_parts.append(f"   ID: {paper_id}")
            if paper_path:
                response_parts.append(f"   路径: {paper_path}")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logging.error(f"获取论文列表失败: {str(e)}")
        return f"获取论文列表出错: {str(e)}"


@tool("获取论文详细信息工具")
def get_paper_details_tool(paper_id: str) -> str:
    """
    获取指定论文的详细信息，包括所有文本块的内容摘要。
    这个工具用于深入了解某篇论文的完整内容。
    
    输入参数：
    - paper_id (字符串，必需): 论文ID，可以通过获取论文列表工具获取
    """
    try:
        from utils.vector_db import get_db_connection, return_db_connection
        from psycopg2.extras import RealDictCursor
        
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 查询论文基本信息
            cur.execute("""
                SELECT title, attachment_path, source, metadata
                FROM papers
                WHERE paper_id = %s
            """, (paper_id,))
            paper_info = cur.fetchone()
            
            if not paper_info:
                return f"未找到ID为 '{paper_id}' 的论文。"
            
            # 查询该论文的所有块
            cur.execute("""
                SELECT chunk_index, chunk_text
                FROM paper_chunks
                WHERE paper_id = %s
                ORDER BY chunk_index
            """, (paper_id,))
            chunks = cur.fetchall()
            
            # 格式化结果
            paper_title = paper_info['title'] or '未知标题'
            paper_path = paper_info['attachment_path'] or ''
            
            response_parts = [
                f"论文信息：\n",
                f"标题: {paper_title}\n",
                f"ID: {paper_id}\n",
                f"文本块数量: {len(chunks)}\n"
            ]
            
            if paper_path:
                response_parts.append(f"路径: {paper_path}\n")
            
            response_parts.append(f"\n内容概览（前3个块）：\n")
            
            # 显示前3个块的内容摘要
            for i, chunk in enumerate(chunks[:3], 1):
                chunk_index = chunk['chunk_index']
                content_preview = chunk['chunk_text'][:300] + "..." if len(chunk['chunk_text']) > 300 else chunk['chunk_text']
                response_parts.append(f"\n[块 {chunk_index + 1}]")
                response_parts.append(f"{content_preview}\n")
            
            if len(chunks) > 3:
                response_parts.append(f"\n... 还有 {len(chunks) - 3} 个文本块未显示")
            
            return "\n".join(response_parts)
            
        finally:
            return_db_connection(conn)
        
    except Exception as e:
        logging.error(f"获取论文详细信息失败: {str(e)}")
        return f"获取论文详细信息出错: {str(e)}"


@tool("读取PDF文件工具")
def read_pdf_file_tool(pdf_path: str, max_pages: int = 10) -> str:
    """
    直接从文件路径读取PDF文件内容。
    这个工具用于读取PDF文件的原始文本内容，特别是当需要查看完整PDF内容时。
    
    输入参数：
    - pdf_path (字符串，必需): PDF文件的完整路径
    - max_pages (整数，可选): 最大读取页数，默认10页。如果为0或负数，则读取所有页面。
    """
    try:
        import fitz  # PyMuPDF
        
        if not os.path.exists(pdf_path):
            return f"PDF文件不存在: {pdf_path}"
        
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            # 获取总页数
            total_pages = len(doc)
            
            # 确定要读取的页数
            if max_pages <= 0:
                pages_to_read = total_pages
            else:
                pages_to_read = min(max_pages, total_pages)
            
            # 提取文本
            for page_num in range(pages_to_read):
                page = doc[page_num]
                page_text = page.get_text()
                full_text += f"\n\n--- 第 {page_num + 1} 页 ---\n\n"
                full_text += page_text
            
            doc.close()
            
            # 清理NUL字符
            full_text = full_text.replace('\x00', '').replace('\0', '')
            
            # 限制返回长度（避免超出上下文限制）
            max_length = 10000
            original_length = len(full_text)
            if original_length > max_length:
                full_text = full_text[:max_length] + f"\n\n... (已截断，总长度: {original_length} 字符，仅显示前 {max_length} 字符)"
            
            return f"PDF文件内容（共读取 {pages_to_read} 页，总页数: {total_pages} 页）：\n\n{full_text}"
            
        except Exception as e:
            logging.error(f"读取PDF文件失败: {pdf_path}, 错误: {str(e)}")
            return f"读取PDF文件失败: {str(e)}"
            
    except ImportError:
        return "PyMuPDF未安装，无法读取PDF文件。请运行: pip install PyMuPDF"
    except Exception as e:
        logging.error(f"读取PDF文件工具出错: {str(e)}")
        return f"读取PDF文件出错: {str(e)}"


@tool("按作者查询论文工具")
def search_papers_by_author_tool(author_name: str, limit: int = 10) -> str:
    """
    根据作者姓名查询论文列表。
    这个工具用于查找特定作者的所有论文。
    
    输入参数：
    - author_name (字符串，必需): 作者姓名（支持部分匹配）
    - limit (整数，可选): 返回的最大结果数，默认10
    """
    try:
        from utils.vector_db import get_db_connection, return_db_connection
        from psycopg2.extras import RealDictCursor
        
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 查询包含指定作者的论文（authors是数组字段）
            cur.execute("""
                SELECT paper_id, title, authors, abstract, year, journal, keywords, attachment_path
                FROM papers
                WHERE authors IS NOT NULL 
                AND EXISTS (
                    SELECT 1 FROM unnest(authors) AS author 
                    WHERE LOWER(author) LIKE LOWER(%s)
                )
                ORDER BY created_at DESC
                LIMIT %s
            """, (f'%{author_name}%', limit))
            
            papers = cur.fetchall()
            
            if not papers:
                return f"未找到作者 '{author_name}' 的论文。"
            
            response_parts = [f"找到 {len(papers)} 篇作者包含 '{author_name}' 的论文：\n"]
            for i, paper in enumerate(papers, 1):
                response_parts.append(f"\n[{i}] {paper['title'] or '未知标题'}")
                response_parts.append(f"   ID: {paper['paper_id']}")
                if paper['authors']:
                    authors_str = ', '.join(paper['authors']) if isinstance(paper['authors'], list) else str(paper['authors'])
                    response_parts.append(f"   作者: {authors_str}")
                if paper['year']:
                    response_parts.append(f"   年份: {paper['year']}")
                if paper['journal']:
                    response_parts.append(f"   期刊: {paper['journal']}")
                if paper['abstract']:
                    abstract_preview = paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']
                    response_parts.append(f"   摘要: {abstract_preview}")
            
            return "\n".join(response_parts)
            
        finally:
            return_db_connection(conn)
    except Exception as e:
        logging.error(f"按作者查询论文失败: {str(e)}")
        return f"按作者查询论文出错: {str(e)}"


@tool("按关键词查询论文工具")
def search_papers_by_keywords_tool(keywords: str, limit: int = 10) -> str:
    """
    根据关键词查询论文列表。关键词可以匹配论文的标题、摘要或关键词字段。
    
    输入参数：
    - keywords (字符串，必需): 关键词（支持多个关键词，用空格或逗号分隔）
    - limit (整数，可选): 返回的最大结果数，默认10
    """
    try:
        from utils.vector_db import get_db_connection, return_db_connection
        from psycopg2.extras import RealDictCursor
        
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 分割关键词
            keyword_list = [k.strip() for k in keywords.replace(',', ' ').split() if k.strip()]
            if not keyword_list:
                return "请输入有效的关键词。"
            
            # 构建查询条件（在标题、摘要、关键词字段中搜索）
            conditions = []
            params = []
            
            for keyword in keyword_list:
                conditions.append("""
                    (LOWER(title) LIKE LOWER(%s) 
                     OR LOWER(abstract) LIKE LOWER(%s)
                     OR EXISTS (
                         SELECT 1 FROM unnest(keywords) AS kw 
                         WHERE LOWER(kw) LIKE LOWER(%s)
                     ))
                """)
                keyword_pattern = f'%{keyword}%'
                params.extend([keyword_pattern, keyword_pattern, keyword_pattern])
            
            # 组合所有条件（AND关系）
            where_clause = " AND ".join(conditions)
            
            query = f"""
                SELECT paper_id, title, authors, abstract, year, journal, keywords, attachment_path
                FROM papers
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(limit)
            
            cur.execute(query, params)
            papers = cur.fetchall()
            
            if not papers:
                return f"未找到包含关键词 '{keywords}' 的论文。"
            
            response_parts = [f"找到 {len(papers)} 篇包含关键词 '{keywords}' 的论文：\n"]
            for i, paper in enumerate(papers, 1):
                response_parts.append(f"\n[{i}] {paper['title'] or '未知标题'}")
                response_parts.append(f"   ID: {paper['paper_id']}")
                if paper['authors']:
                    authors_str = ', '.join(paper['authors']) if isinstance(paper['authors'], list) else str(paper['authors'])
                    response_parts.append(f"   作者: {authors_str}")
                if paper['year']:
                    response_parts.append(f"   年份: {paper['year']}")
                if paper['abstract']:
                    abstract_preview = paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']
                    response_parts.append(f"   摘要: {abstract_preview}")
            
            return "\n".join(response_parts)
            
        finally:
            return_db_connection(conn)
    except Exception as e:
        logging.error(f"按关键词查询论文失败: {str(e)}")
        return f"按关键词查询论文出错: {str(e)}"


@tool("按条件查询论文工具")
def search_papers_by_conditions_tool(year: Optional[int] = None, journal: Optional[str] = None, source: Optional[str] = None, limit: int = 10) -> str:
    """
    根据多个条件查询论文列表（年份、期刊、来源等）。
    
    输入参数：
    - year (整数，可选): 论文发表年份
    - journal (字符串，可选): 期刊名称（支持部分匹配）
    - source (字符串，可选): 论文来源（'csv', 'email', 'pdf', 'zotero', 'obsidian'）
    - limit (整数，可选): 返回的最大结果数，默认10
    """
    try:
        from utils.vector_db import get_db_connection, return_db_connection
        from psycopg2.extras import RealDictCursor
        
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 构建查询条件
            conditions = []
            params = []
            
            if year:
                conditions.append("year = %s")
                params.append(year)
            
            if journal:
                conditions.append("LOWER(journal) LIKE LOWER(%s)")
                params.append(f'%{journal}%')
            
            if source:
                conditions.append("source = %s")
                params.append(source)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT paper_id, title, authors, abstract, year, journal, keywords, source, attachment_path
                FROM papers
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(limit)
            
            cur.execute(query, params)
            papers = cur.fetchall()
            
            if not papers:
                conditions_str = []
                if year:
                    conditions_str.append(f"年份={year}")
                if journal:
                    conditions_str.append(f"期刊包含'{journal}'")
                if source:
                    conditions_str.append(f"来源={source}")
                return f"未找到满足条件（{', '.join(conditions_str)}）的论文。"
            
            response_parts = [f"找到 {len(papers)} 篇满足条件的论文：\n"]
            for i, paper in enumerate(papers, 1):
                response_parts.append(f"\n[{i}] {paper['title'] or '未知标题'}")
                response_parts.append(f"   ID: {paper['paper_id']}")
                if paper['authors']:
                    authors_str = ', '.join(paper['authors']) if isinstance(paper['authors'], list) else str(paper['authors'])
                    response_parts.append(f"   作者: {authors_str}")
                if paper['year']:
                    response_parts.append(f"   年份: {paper['year']}")
                if paper['journal']:
                    response_parts.append(f"   期刊: {paper['journal']}")
                if paper['source']:
                    response_parts.append(f"   来源: {paper['source']}")
            
            return "\n".join(response_parts)
            
        finally:
            return_db_connection(conn)
    except Exception as e:
        logging.error(f"按条件查询论文失败: {str(e)}")
        return f"按条件查询论文出错: {str(e)}"


@tool("获取论文全文工具")
def get_paper_full_text_tool(paper_id: str) -> str:
    """
    获取指定论文的完整文本内容（优先从数据库块读取，如果不存在则从PDF文件读取）。
    这个工具用于需要查看论文完整内容的情况。
    
    注意：优先从数据库块读取，因为这样更高效。只有在数据库块不存在时才会尝试读取PDF文件。
    
    输入参数：
    - paper_id (字符串，必需): 论文ID
    """
    try:
        from utils.vector_db import get_db_connection, return_db_connection
        from psycopg2.extras import RealDictCursor
        
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 查询论文信息和附件路径
            cur.execute("""
                SELECT title, attachment_path, abstract, authors, year, journal, keywords
                FROM papers
                WHERE paper_id = %s
            """, (paper_id,))
            paper_info = cur.fetchone()
            
            if not paper_info:
                return f"未找到ID为 '{paper_id}' 的论文。"
            
            paper_title = paper_info['title'] or '未知标题'
            attachment_path = paper_info['attachment_path']
            
            response_parts = [
                f"论文信息：\n",
                f"标题: {paper_title}\n",
                f"ID: {paper_id}\n"
            ]
            
            # 添加元数据
            if paper_info['authors']:
                authors_str = ', '.join(paper_info['authors']) if isinstance(paper_info['authors'], list) else str(paper_info['authors'])
                response_parts.append(f"作者: {authors_str}\n")
            if paper_info['year']:
                response_parts.append(f"年份: {paper_info['year']}\n")
            if paper_info['journal']:
                response_parts.append(f"期刊: {paper_info['journal']}\n")
            if paper_info['abstract']:
                response_parts.append(f"\n摘要:\n{paper_info['abstract']}\n")
            if paper_info['keywords']:
                keywords_str = ', '.join(paper_info['keywords']) if isinstance(paper_info['keywords'], list) else str(paper_info['keywords'])
                response_parts.append(f"\n关键词: {keywords_str}\n")
            
            # 优先从数据库块中获取（更高效）
            cur.execute("""
                SELECT chunk_text
                FROM paper_chunks
                WHERE paper_id = %s
                ORDER BY chunk_index
            """, (paper_id,))
            chunks = cur.fetchall()
            
            if chunks and len(chunks) > 0:
                # 从数据库块中获取全文（高效方式）
                full_text = "\n\n".join([chunk['chunk_text'] for chunk in chunks])
                # 限制长度
                max_length = 15000
                if len(full_text) > max_length:
                    full_text = full_text[:max_length] + f"\n\n... (已截断，总长度: {len(full_text)} 字符，仅显示前 {max_length} 字符)"
                response_parts.append(f"\n\n论文全文（从数据库块读取，共 {len(chunks)} 个块）：\n{full_text}")
            elif attachment_path and os.path.exists(attachment_path):
                # 如果数据库块不存在，尝试从PDF文件读取（备用方案）
                try:
                    logging.info(f"数据库块不存在，尝试从PDF读取: {attachment_path}")
                    # 使用read_pdf_file_tool读取PDF
                    full_text = read_pdf_file_tool(attachment_path, max_pages=0)  # 读取所有页面
                    response_parts.append(f"\n\n论文全文（从PDF读取，数据库块不存在）：\n{full_text}")
                except Exception as e:
                    logging.warning(f"从PDF读取全文失败: {str(e)}")
                    response_parts.append(f"\n\n未找到论文的文本内容（数据库块不存在，且PDF读取失败: {str(e)}）。")
            else:
                response_parts.append("\n\n未找到论文的文本内容（数据库块不存在，且PDF文件路径无效）。")
            
            return "".join(response_parts)
            
        finally:
            return_db_connection(conn)
    except Exception as e:
        logging.error(f"获取论文全文失败: {str(e)}")
        return f"获取论文全文出错: {str(e)}"


@tool("总结论文内容工具")
def summarize_paper_tool(paper_id: str, max_chunks: int = 10) -> str:
    """
    总结指定论文的主要内容，通过检索最相关的文本块并生成摘要。
    这个工具用于快速了解论文的核心内容。
    
    输入参数：
    - paper_id (字符串，必需): 论文ID
    - max_chunks (整数，可选): 用于总结的最大文本块数量，默认10
    """
    try:
        from utils.vector_db import get_db_connection, return_db_connection
        from psycopg2.extras import RealDictCursor
        
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 查询论文标题
            cur.execute("SELECT title FROM papers WHERE paper_id = %s", (paper_id,))
            paper_info = cur.fetchone()
            
            if not paper_info:
                return f"未找到ID为 '{paper_id}' 的论文。"
            
            paper_title = paper_info['title'] or '未知标题'
            
            # 查询该论文的所有块（按索引排序，前面的块通常包含摘要、介绍等重要内容）
            cur.execute("""
                SELECT chunk_text
                FROM paper_chunks
                WHERE paper_id = %s
                ORDER BY chunk_index
                LIMIT %s
            """, (paper_id, max_chunks))
            selected_chunks = cur.fetchall()
            
            # 查询总块数
            cur.execute("SELECT COUNT(*) as total FROM paper_chunks WHERE paper_id = %s", (paper_id,))
            total_result = cur.fetchone()
            total_chunks = total_result['total'] if total_result else 0
            
            # 合并文本
            combined_text = "\n\n".join([chunk['chunk_text'] for chunk in selected_chunks])
            
            # 限制总长度，避免超出模型上下文
            max_length = 8000
            if len(combined_text) > max_length:
                combined_text = combined_text[:max_length] + "..."
            
            return (
                f"论文: {paper_title}\n"
                f"论文ID: {paper_id}\n"
                f"已检索 {len(selected_chunks)} 个文本块（共 {total_chunks} 个）\n\n"
                f"内容摘要：\n{combined_text}"
            )
            
        finally:
            return_db_connection(conn)
        
    except Exception as e:
        logging.error(f"总结论文内容失败: {str(e)}")
        return f"总结论文内容出错: {str(e)}"

