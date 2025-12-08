#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent基础配置和工具
"""

from crewai.tools import tool
from config import llm
from utils.file_utils import extract_real_url_from_redirect, is_pdf_url
import requests
import re
import os
import logging
from datetime import datetime


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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=180, allow_redirects=True)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logging.warning(f"获取网页内容超时（超过180秒）: {url}")
            return f"获取网页内容失败: 下载超时（超过180秒），已跳过"
        except requests.exceptions.RequestException as e:
            logging.warning(f"获取网页内容失败: {url}, 错误: {str(e)}")
            return f"获取网页内容失败: {str(e)}"
        
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
        
        # 直接使用requests下载PDF（更可靠，支持超时）
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(arxiv_pdf_url, headers=headers, timeout=180, allow_redirects=True)
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

