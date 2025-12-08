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


def normalize_title_for_dedup(title):
    """
    规范化论文标题用于查重，处理空格差异和大小写差异
    
    Args:
        title: 原始标题
    
    Returns:
        规范化后的标题（小写、去除多余空格、去除标点符号）
    """
    if not title:
        return ""
    
    # 转换为小写
    normalized = title.lower()
    
    # 去除标点符号和特殊字符（保留字母、数字和空格）
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # 将多个连续空格替换为单个空格
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # 去除首尾空格
    normalized = normalized.strip()
    
    return normalized


def deduplicate_papers_by_title(papers):
    """
    根据论文标题去除重复论文，处理空格差异和大小写差异
    
    Args:
        papers: 论文列表，每个元素是包含 'title' 键的字典
    
    Returns:
        (去重后的论文列表, 被删除的重复论文列表)
    """
    if not papers:
        return [], []
    
    seen_titles = {}  # {规范化标题: 原始标题}
    deduplicated = []
    duplicates = []
    
    for paper in papers:
        title = paper.get('title', '')
        normalized_title = normalize_title_for_dedup(title)
        
        if not normalized_title:
            # 标题为空，保留（可能是特殊情况）
            deduplicated.append(paper)
            continue
        
        if normalized_title in seen_titles:
            # 发现重复
            original_title = seen_titles[normalized_title]
            duplicates.append({
                'paper': paper,
                'original_title': original_title,
                'duplicate_title': title,
                'normalized': normalized_title
            })
        else:
            # 首次出现，保留
            seen_titles[normalized_title] = title
            deduplicated.append(paper)
    
    return deduplicated, duplicates


def sanitize_filename(filename, max_length=200):
    """
    清理文件名，去除不允许的字符
    
    Args:
        filename: 原始文件名
        max_length: 最大长度
    
    Returns:
        清理后的文件名
    """
    # 去除不允许的字符（Windows和Linux都不允许的字符）
    # 不允许的字符: < > : " / \ | ? *
    # 同时去除控制字符
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
    
    # 去除首尾空格和点
    sanitized = sanitized.strip(' .')
    
    # 将多个空格替换为单个空格
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # 限制长度
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # 如果文件名为空，使用默认名称
    if not sanitized:
        sanitized = "paper"
    
    return sanitized


def save_pdf_to_downloads(pdf_bytes, paper_title=None, url=None):
    """
    保存PDF文件到downloads文件夹
    
    Args:
        pdf_bytes: PDF文件的字节内容
        paper_title: 论文标题（用于生成文件名）
        url: 论文URL（如果标题不可用，从URL提取文件名）
    
    Returns:
        pdf_filename: 保存的PDF文件路径，如果保存失败则返回None
    """
    try:
        # 验证pdf_bytes是否有效
        if not pdf_bytes:
            logging.warning("PDF字节内容为空，跳过保存")
            return None
        
        if not isinstance(pdf_bytes, bytes):
            logging.warning(f"PDF字节内容类型错误: {type(pdf_bytes)}，跳过保存")
            return None
        
        if len(pdf_bytes) == 0:
            logging.warning("PDF字节内容长度为0，跳过保存")
            return None
        
        # 验证是否为有效的PDF文件（检查PDF文件头）
        if len(pdf_bytes) < 4 or pdf_bytes[:4] != b'%PDF':
            logging.warning("PDF字节内容不是有效的PDF文件（缺少PDF文件头），跳过保存")
            return None
        
        downloads_dir = 'downloads'
        os.makedirs(downloads_dir, exist_ok=True)
        
        # 生成文件名
        if paper_title:
            filename_base = sanitize_filename(paper_title)
        elif url:
            # 从URL提取文件名
            from urllib.parse import urlparse, unquote
            parsed = urlparse(url)
            filename_from_url = os.path.basename(unquote(parsed.path))
            if filename_from_url and filename_from_url.endswith('.pdf'):
                filename_base = sanitize_filename(filename_from_url[:-4])
            else:
                # 如果无法从URL提取，使用时间戳
                filename_base = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            filename_base = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        pdf_filename = os.path.join(downloads_dir, f"{filename_base}.pdf")
        
        # 如果文件已存在，添加时间戳
        if os.path.exists(pdf_filename):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = os.path.join(downloads_dir, f"{filename_base}_{timestamp}.pdf")
        
        with open(pdf_filename, 'wb') as f:
            f.write(pdf_bytes)
        
        # 验证文件是否成功写入
        if not os.path.exists(pdf_filename):
            logging.warning(f"PDF文件保存后不存在: {pdf_filename}")
            return None
        
        file_size = os.path.getsize(pdf_filename)
        if file_size == 0:
            logging.warning(f"PDF文件保存后大小为0: {pdf_filename}")
            try:
                os.remove(pdf_filename)  # 删除空文件
            except:
                pass
            return None
        
        logging.info(f"PDF文件已保存到: {pdf_filename} (大小: {file_size} 字节)")
        return pdf_filename
    except Exception as e:
        logging.warning(f"保存PDF文件失败: {str(e)}")
        return None


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


def check_content_type_pdf(url, timeout=30):
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
            
    except requests.exceptions.Timeout:
        logging.debug(f"HEAD请求检查超时（超过{timeout}秒）: {url}")
    except Exception as e:
        logging.debug(f"HEAD请求检查失败 {url}: {str(e)}")
    return False, None


async def fetch_fulltext_from_url_async(url, timeout=180, paper_title=None, save_pdf=True):
    """
    使用 Crawl4AI 从URL获取全文内容（异步版本）
    返回: (content, is_pdf, pdf_bytes, pdf_filename)
    
    Args:
        url: 论文URL
        timeout: 超时时间
        paper_title: 论文标题（用于保存PDF文件）
        save_pdf: 是否保存PDF到downloads文件夹
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
                    pdf_bytes = response.content
                    pdf_filename = None
                    if save_pdf:
                        pdf_filename = save_pdf_to_downloads(pdf_bytes, paper_title, url)
                    return None, True, pdf_bytes, pdf_filename
                # 检查内容开头是否为PDF文件头（%PDF）
                if response.content[:4] == b'%PDF':
                    pdf_bytes = response.content
                    pdf_filename = None
                    if save_pdf:
                        pdf_filename = save_pdf_to_downloads(pdf_bytes, paper_title, url)
                    return None, True, pdf_bytes, pdf_filename
            except requests.exceptions.Timeout:
                logging.warning(f"下载PDF超时（超过{timeout}秒）: {url}")
                return None, False, None, None
            except Exception as e:
                logging.error(f"下载PDF失败 {url}: {str(e)}")
                return None, False, None, None
        
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
                    pdf_bytes = response.content
                    pdf_filename = None
                    if save_pdf:
                        pdf_filename = save_pdf_to_downloads(pdf_bytes, paper_title, url)
                    return None, True, pdf_bytes, pdf_filename
                # 检查内容开头是否为PDF文件头（%PDF）
                if response.content[:4] == b'%PDF':
                    pdf_bytes = response.content
                    pdf_filename = None
                    if save_pdf:
                        pdf_filename = save_pdf_to_downloads(pdf_bytes, paper_title, url)
                    return None, True, pdf_bytes, pdf_filename
            except requests.exceptions.Timeout:
                logging.warning(f"下载PDF超时（超过{timeout}秒）: {url}")
                return None, False, None, None
            except Exception as e:
                logging.error(f"下载PDF失败 {url}: {str(e)}")
                return None, False, None, None
        
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
                                    pdf_bytes = pdf_response.content
                                    pdf_filename = None
                                    if save_pdf:
                                        pdf_filename = save_pdf_to_downloads(pdf_bytes, paper_title, url)
                                    return None, True, pdf_bytes, pdf_filename
                            except requests.exceptions.Timeout:
                                logging.warning(f"下载PDF超时（超过{timeout}秒）: {url}")
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
                                    pdf_bytes = full_response.content
                                    pdf_filename = None
                                    if save_pdf:
                                        pdf_filename = save_pdf_to_downloads(pdf_bytes, paper_title, url)
                                    return None, True, pdf_bytes, pdf_filename
                            except:
                                pass
                        
                        # 返回HTML内容
                        return result.html, False, None, None
                    else:
                        logging.warning(f"Crawl4AI获取成功但无HTML内容 {url}")
                        return await fetch_fulltext_fallback(url, timeout, paper_title, save_pdf)
                else:
                    error_msg = '未知错误'
                    if hasattr(result, 'error_message'):
                        error_msg = result.error_message
                    elif hasattr(result, 'error'):
                        error_msg = str(result.error)
                    logging.warning(f"Crawl4AI获取失败 {url}: {error_msg}")
                    # 如果Crawl4AI失败，尝试使用requests作为备选
                    return await fetch_fulltext_fallback(url, timeout, paper_title, save_pdf)
        except ImportError:
            # 如果Crawl4AI未安装，使用备选方案
            logging.warning(f"Crawl4AI未安装，使用备选方案获取 {url}")
            return await fetch_fulltext_fallback(url, timeout, paper_title, save_pdf)
        except Exception as e:
            logging.warning(f"Crawl4AI异常 {url}: {str(e)}")
            # 如果Crawl4AI异常，尝试使用requests作为备选
            return await fetch_fulltext_fallback(url, timeout, paper_title, save_pdf)
                
    except Exception as e:
        logging.error(f"获取URL内容失败 {url}: {str(e)}")
        # 尝试使用requests作为备选
        return await fetch_fulltext_fallback(url, timeout, paper_title, save_pdf)


async def fetch_fulltext_fallback(url, timeout=180, paper_title=None, save_pdf=True):
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
            pdf_bytes = response.content
            pdf_filename = None
            if save_pdf:
                pdf_filename = save_pdf_to_downloads(pdf_bytes, paper_title, url)
            return None, True, pdf_bytes, pdf_filename
        else:
            return response.text, False, None, None
    except requests.exceptions.Timeout:
        logging.warning(f"备选方案下载超时（超过{timeout}秒）: {url}")
        return None, False, None, None
    except Exception as e:
        logging.error(f"备选方案也失败 {url}: {str(e)}")
        return None, False, None, None


def fetch_fulltext_from_url(url, timeout=180, paper_title=None, save_pdf=True):
    """
    从URL获取全文内容（同步包装器）
    返回: (content, is_pdf, pdf_bytes, pdf_filename)
    
    Args:
        url: 论文URL
        timeout: 超时时间
        paper_title: 论文标题（用于保存PDF文件）
        save_pdf: 是否保存PDF到downloads文件夹
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
        
        result = loop.run_until_complete(fetch_fulltext_from_url_async(url, timeout, paper_title, save_pdf))
        
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
                pdf_bytes = response.content
                pdf_filename = None
                if save_pdf:
                    pdf_filename = save_pdf_to_downloads(pdf_bytes, paper_title, url)
                return None, True, pdf_bytes, pdf_filename
            else:
                return response.text, False, None, None
        except Exception as e2:
            logging.error(f"所有方法都失败 {url}: {str(e2)}")
            return None, False, None, None


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

