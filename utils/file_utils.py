#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件处理工具（PDF、网页、CSV）
"""

import os
import re
import logging
import pandas as pd
import requests
import asyncio
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from crawl4ai import AsyncWebCrawler


def load_papers_from_csv(csv_path, title_col=0, abstract_col=2, link_col=None):
    """
    从CSV文件加载论文信息
    
    Args:
        csv_path: CSV文件路径
        title_col: 论文标题列索引（从0开始）
        abstract_col: 摘要列索引（从0开始）
        link_col: 论文链接列索引（可选，从0开始，如果为None则不读取链接）
    
    Returns:
        papers: 论文列表，格式与extract_paper_info返回的格式一致
    """
    papers = []
    
    try:
        # 尝试多种编码格式读取CSV文件（WPS可能使用GBK编码保存）
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'cp936']
        df = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                used_encoding = encoding
                print(f"  成功使用 {encoding} 编码读取CSV文件")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # 如果是其他错误（如CSV格式错误），也继续尝试下一个编码
                if 'Unicode' in str(type(e).__name__):
                    continue
                # 如果不是编码错误，可能是CSV格式问题，使用第一个编码再试一次
                if encoding == encodings[0]:
                    raise
        
        if df is None:
            raise Exception("无法使用常见编码格式读取CSV文件，请检查文件编码。支持的编码：utf-8-sig, utf-8, gbk, gb2312, gb18030")
        
        # 检查列索引是否有效
        if title_col >= len(df.columns):
            raise ValueError(f"标题列索引 {title_col} 超出CSV列数 {len(df.columns)}")
        if abstract_col >= len(df.columns):
            raise ValueError(f"摘要列索引 {abstract_col} 超出CSV列数 {len(df.columns)}")
        if link_col is not None and link_col != '':
            link_col_int = int(link_col)
            if link_col_int >= len(df.columns):
                raise ValueError(f"链接列索引 {link_col_int} 超出CSV列数 {len(df.columns)}")
        
        # 获取列名
        title_col_name = df.columns[title_col]
        abstract_col_name = df.columns[abstract_col]
        link_col_name = df.columns[int(link_col)] if link_col and link_col != '' else None
        
        print(f"  从CSV读取论文信息:")
        print(f"    - 标题列: {title_col_name} (索引 {title_col})")
        print(f"    - 摘要列: {abstract_col_name} (索引 {abstract_col})")
        if link_col_name:
            print(f"    - 链接列: {link_col_name} (索引 {link_col})")
        else:
            print(f"    - 链接列: 未指定")
        
        # 遍历每一行
        for idx, row in df.iterrows():
            title = str(row[title_col_name]).strip() if pd.notna(row[title_col_name]) else ''
            abstract = str(row[abstract_col_name]).strip() if pd.notna(row[abstract_col_name]) else ''
            link = str(row[link_col_name]).strip() if link_col_name and pd.notna(row[link_col_name]) else ''
            
            # 跳过空标题
            if not title:
                continue
            
            paper = {
                'title': title,
                'link': link,
                'snippet': abstract if abstract else ''  # 保存完整摘要到snippet（本地模式下会直接使用）
            }
            papers.append(paper)
        
        print(f"  成功读取 {len(papers)} 篇论文")
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    except Exception as e:
        raise Exception(f"读取CSV文件时出错: {str(e)}")
    
    return papers


def is_pdf_url(url):
    """
    判断URL是否为PDF文件
    检查多种情况：
    1. URL路径以.pdf结尾
    2. URL参数中包含PDF链接（如Google Scholar跳转链接）
    3. URL中包含.pdf字符串
    """
    if not url:
        return False
    
    url_lower = url.lower()
    
    # 方法1: 检查路径是否以.pdf结尾
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith('.pdf'):
        return True
    
    # 方法2: 检查查询参数中是否包含PDF链接（处理Google Scholar等跳转链接）
    if parsed.query:
        try:
            params = parse_qs(parsed.query)
            # 检查常见的URL参数名（如url, link, href等）
            for param_name in ['url', 'link', 'href', 'target', 'redirect']:
                if param_name in params:
                    param_value = params[param_name][0]
                    # URL解码
                    decoded = unquote(param_value)
                    if '.pdf' in decoded.lower() or decoded.lower().endswith('.pdf'):
                        return True
        except:
            pass
    
    # 方法3: 检查整个URL中是否包含.pdf（但排除查询参数中的误判）
    if '.pdf' in url_lower:
        # 确保不是误判（如包含"pdf"的域名）
        # 检查是否在路径或文件名中
        if '/pdf' in path or path.endswith('.pdf') or '.pdf' in parsed.fragment.lower():
            return True
        # 检查查询参数值中
        if parsed.query and '.pdf' in parsed.query.lower():
            return True
    
    return False


def extract_real_url_from_redirect(url):
    """
    从跳转链接中提取真正的目标URL
    例如：Google Scholar的scholar_url参数
    """
    try:
        parsed = urlparse(url)
        if parsed.query:
            params = parse_qs(parsed.query)
            # 检查常见的URL参数名
            for param_name in ['url', 'link', 'href', 'target', 'redirect']:
                if param_name in params:
                    real_url = params[param_name][0]
                    # URL解码
                    decoded = unquote(real_url)
                    # 验证是否为有效URL
                    if decoded.startswith('http://') or decoded.startswith('https://'):
                        return decoded
    except:
        pass
    return None


def check_content_type_pdf(url, timeout=10):
    """
    通过HEAD请求检查URL的Content-Type是否为PDF
    返回: (is_pdf, real_url) - real_url是最终跳转后的URL
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # 使用HEAD请求检查Content-Type
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '').lower()
        real_url = response.url  # 获取最终跳转后的URL
        
        # 检查Content-Type
        if 'application/pdf' in content_type or 'pdf' in content_type:
            return True, real_url
        
        # 也检查最终URL是否为PDF
        if is_pdf_url(real_url):
            return True, real_url
            
    except Exception as e:
        logging.debug(f"HEAD请求检查失败 {url}: {str(e)}")
    return False, None


async def fetch_fulltext_from_url_async(url, timeout=30):
    """
    使用 Crawl4AI 从URL获取全文内容（异步版本）
    返回: (content, is_pdf, pdf_bytes)
    """
    try:
        # 步骤1: 检查URL参数中是否包含PDF链接（处理跳转链接）
        real_url = extract_real_url_from_redirect(url)
        if real_url and is_pdf_url(real_url):
            url = real_url  # 使用真正的PDF URL
            logging.info(f"从跳转链接提取到PDF URL: {real_url}")
        
        # 步骤2: 通过HEAD请求检查Content-Type（更可靠的方法）
        is_pdf_by_type, final_url = check_content_type_pdf(url, timeout=min(timeout, 10))
        if is_pdf_by_type:
            if final_url and final_url != url:
                url = final_url  # 使用最终跳转后的URL
            # PDF文件使用requests下载
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                response.raise_for_status()
                # 再次确认Content-Type
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' in content_type or 'pdf' in content_type:
                    return None, True, response.content
                # 检查内容开头是否为PDF文件头（%PDF）
                if response.content[:4] == b'%PDF':
                    return None, True, response.content
            except Exception as e:
                logging.error(f"下载PDF失败 {url}: {str(e)}")
                return None, False, None
        
        # 步骤3: 检查URL路径是否为PDF
        if is_pdf_url(url):
            # PDF文件使用requests下载
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                response.raise_for_status()
                # 检查Content-Type
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' in content_type or 'pdf' in content_type:
                    return None, True, response.content
                # 检查内容开头是否为PDF文件头（%PDF）
                if response.content[:4] == b'%PDF':
                    return None, True, response.content
            except Exception as e:
                logging.error(f"下载PDF失败 {url}: {str(e)}")
                return None, False, None
        
        # 使用 Crawl4AI 获取网页内容
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(
                    url=url,
                    wait_for="css:body",  # 等待页面加载
                    timeout=timeout * 1000,  # Crawl4AI使用毫秒
                    bypass_cache=True
                )
                
                if result and hasattr(result, 'success') and result.success:
                    if hasattr(result, 'html') and result.html:
                        # 检查返回的内容类型
                        content_type = ''
                        if hasattr(result, 'headers') and result.headers:
                            content_type = result.headers.get('Content-Type', '').lower()
                        
                        # 如果Content-Type是PDF，或者HTML内容很少（可能是PDF被误判为HTML）
                        if 'pdf' in content_type or 'application/pdf' in content_type:
                            # 如果返回的是PDF，尝试下载
                            try:
                                pdf_response = requests.get(url, timeout=timeout, allow_redirects=True)
                                pdf_response.raise_for_status()
                                # 再次确认是PDF
                                if pdf_response.content[:4] == b'%PDF':
                                    return None, True, pdf_response.content
                            except:
                                pass
                        
                        # 如果HTML内容很少，可能是PDF被误判，尝试直接下载检查
                        if len(result.html) < 1000:  # HTML内容异常少
                            try:
                                test_response = requests.get(url, timeout=timeout, allow_redirects=True, stream=True)
                                test_response.raise_for_status()
                                # 读取前几个字节检查是否为PDF
                                chunk = test_response.raw.read(4)
                                if chunk == b'%PDF':
                                    # 确实是PDF，完整下载
                                    full_response = requests.get(url, timeout=timeout, allow_redirects=True)
                                    full_response.raise_for_status()
                                    return None, True, full_response.content
                            except:
                                pass
                        
                        # 返回HTML内容
                        return result.html, False, None
                    else:
                        logging.warning(f"Crawl4AI获取成功但无HTML内容 {url}")
                        return await fetch_fulltext_fallback(url, timeout)
                else:
                    error_msg = '未知错误'
                    if hasattr(result, 'error_message'):
                        error_msg = result.error_message
                    elif hasattr(result, 'error'):
                        error_msg = str(result.error)
                    logging.warning(f"Crawl4AI获取失败 {url}: {error_msg}")
                    # 如果Crawl4AI失败，尝试使用requests作为备选
                    return await fetch_fulltext_fallback(url, timeout)
        except ImportError:
            # 如果Crawl4AI未安装，使用备选方案
            logging.warning(f"Crawl4AI未安装，使用备选方案获取 {url}")
            return await fetch_fulltext_fallback(url, timeout)
        except Exception as e:
            logging.warning(f"Crawl4AI异常 {url}: {str(e)}")
            # 如果Crawl4AI异常，尝试使用requests作为备选
            return await fetch_fulltext_fallback(url, timeout)
                
    except Exception as e:
        logging.error(f"获取URL内容失败 {url}: {str(e)}")
        # 尝试使用requests作为备选
        return await fetch_fulltext_fallback(url, timeout)


async def fetch_fulltext_fallback(url, timeout=30):
    """
    备选方案：使用requests获取网页内容
    """
    try:
        # 先检查是否为跳转链接
        real_url = extract_real_url_from_redirect(url)
        if real_url and is_pdf_url(real_url):
            url = real_url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        # 检查Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        is_pdf = 'application/pdf' in content_type or 'pdf' in content_type
        
        # 如果Content-Type不是PDF，但内容开头是PDF文件头，也认为是PDF
        if not is_pdf and response.content[:4] == b'%PDF':
            is_pdf = True
        
        if is_pdf:
            return None, True, response.content
        else:
            return response.text, False, None
    except Exception as e:
        logging.error(f"备选方案也失败 {url}: {str(e)}")
        return None, False, None


def fetch_fulltext_from_url(url, timeout=30):
    """
    从URL获取全文内容（同步包装器）
    返回: (content, is_pdf, pdf_bytes)
    """
    try:
        # 运行异步函数
        # 检查是否已有事件循环
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # 如果没有事件循环，创建新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(fetch_fulltext_from_url_async(url, timeout))
        
        # 清理未完成的任务
        pending = asyncio.all_tasks(loop)
        if pending:
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        # 关闭事件循环
        if not loop.is_closed():
            loop.close()
        
        return result
    except Exception as e:
        logging.error(f"同步包装器错误 {url}: {str(e)}")
        # 如果异步失败，尝试同步备选方案
        try:
            # 先检查是否为跳转链接
            real_url = extract_real_url_from_redirect(url)
            if real_url and is_pdf_url(real_url):
                url = real_url
            
            # 检查Content-Type
            is_pdf_by_type, final_url = check_content_type_pdf(url, timeout=min(timeout, 10))
            if is_pdf_by_type and final_url:
                url = final_url
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            # 检查Content-Type
            content_type = response.headers.get('Content-Type', '').lower()
            is_pdf = 'application/pdf' in content_type or 'pdf' in content_type
            
            # 如果Content-Type不是PDF，但内容开头是PDF文件头，也认为是PDF
            if not is_pdf and response.content[:4] == b'%PDF':
                is_pdf = True
            
            # 如果还不是PDF，检查URL路径
            if not is_pdf:
                is_pdf = is_pdf_url(url) or is_pdf_url(response.url)
            
            if is_pdf:
                return None, True, response.content
            else:
                return response.text, False, None
        except Exception as e2:
            logging.error(f"所有方法都失败 {url}: {str(e2)}")
            return None, False, None


def extract_abstract_from_html(html_content):
    """从HTML中提取摘要"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 尝试多种方式查找摘要
        # 方法1: 查找包含"Abstract"或"摘要"的标签
        abstract_keywords = ['abstract', '摘要', 'summary', '概述']
        for keyword in abstract_keywords:
            # 查找包含关键词的标签
            for tag in soup.find_all(['div', 'section', 'p', 'span'], 
                                    string=re.compile(keyword, re.I)):
                # 查找相邻的文本内容
                parent = tag.parent
                if parent:
                    text = parent.get_text(separator=' ', strip=True)
                    if len(text) > 100:  # 摘要通常较长
                        return text
        
        # 方法2: 查找常见的摘要类名或ID
        abstract_selectors = [
            'div.abstract', 'div#abstract', 'section.abstract',
            'div[class*="abstract"]', 'div[id*="abstract"]',
            'p.abstract', 'span.abstract'
        ]
        for selector in abstract_selectors:
            elements = soup.select(selector)
            if elements:
                text = elements[0].get_text(separator=' ', strip=True)
                if len(text) > 100:
                    return text
        
        # 方法3: 如果找不到，返回所有文本（可能包含摘要）
        text = soup.get_text(separator=' ', strip=True)
        # 尝试找到Abstract之后的内容
        abstract_match = re.search(r'abstract[:\s]+(.{200,2000})', text, re.I)
        if abstract_match:
            return abstract_match.group(1)
        
        return text[:2000]  # 返回前2000字符
    except Exception as e:
        logging.error(f"从HTML提取摘要失败: {str(e)}")
        return None


def extract_text_from_pdf(pdf_bytes):
    """
    从PDF字节流中提取文本
    返回: (full_text, abstract, body)
    """
    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        abstract = ""
        body = ""
        
        # 提取所有页面的文本
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            page_text = page.get_text()
            full_text += f"\n=== 第 {page_num + 1} 页 ===\n{page_text}\n"
        
        # 尝试分离摘要和正文
        # 查找Abstract部分
        abstract_patterns = [
            r'abstract[:\s]+(.{200,2000}?)(?=\n\s*(?:introduction|1\.|keywords|key words))',
            r'摘要[:\s]+(.{200,2000}?)(?=\n\s*(?:引言|1\.|关键词))',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, full_text, re.I | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                break
        
        # 如果没有找到摘要，使用前几页作为摘要
        if not abstract:
            first_pages = full_text.split("===")[:3]  # 前3页
            abstract = " ".join(first_pages)[:2000]
        
        # 正文是剩余部分
        if abstract:
            body_start = full_text.lower().find(abstract.lower())
            if body_start > 0:
                body = full_text[body_start + len(abstract):]
            else:
                body = full_text
        else:
            body = full_text
        
        pdf_doc.close()
        return full_text, abstract, body
    except Exception as e:
        logging.error(f"从PDF提取文本失败: {str(e)}")
        return None, None, None

