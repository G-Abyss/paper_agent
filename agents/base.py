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
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        
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
def arxiv_download_tool(paper_url: str) -> str:
    """
    检查论文URL是否为arXiv论文，如果是则自动下载PDF并提取文本内容（前3页）。当检测到论文可能来自arXiv时，可以使用此工具获取论文的完整内容来辅助判断相关性。输入参数：paper_url (字符串，论文的URL)
    """
    try:
        # 提取真实URL
        from utils.file_utils import extract_real_url_from_redirect
        real_url = extract_real_url_from_redirect(paper_url)
        if real_url:
            paper_url = real_url
        
        # 检查是否为arXiv URL
        arxiv_pattern = r'arxiv\.org/(?:pdf|abs)/(\d+\.\d+)'
        match = re.search(arxiv_pattern, paper_url, re.I)
        
        if not match:
            return "不是arXiv论文"
        
        arxiv_id = match.group(1)
        pdf_bytes = None
        
        # 尝试使用arxiv库下载
        try:
            import arxiv
            # 使用新的Client API（避免弃用警告）
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(client.results(search), None)
            
            if paper:
                # 下载PDF
                pdf_bytes = paper.download_pdf()
            else:
                return f"未找到arXiv论文: {arxiv_id}"
        except ImportError:
            # 如果arxiv库未安装，尝试直接下载PDF
            arxiv_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.get(arxiv_pdf_url, timeout=15)
            response.raise_for_status()
            
            if response.content[:4] == b'%PDF':
                pdf_bytes = response.content
            else:
                return "下载的PDF文件无效"
        
        # 保存PDF到downloads文件夹
        if pdf_bytes:
            try:
                downloads_dir = 'downloads'
                os.makedirs(downloads_dir, exist_ok=True)
                
                # 生成安全的文件名
                safe_filename = re.sub(r'[^\w\s-]', '', arxiv_id).strip()
                safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
                pdf_filename = os.path.join(downloads_dir, f"{safe_filename}.pdf")
                
                # 如果文件已存在，添加时间戳
                if os.path.exists(pdf_filename):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pdf_filename = os.path.join(downloads_dir, f"{safe_filename}_{timestamp}.pdf")
                
                with open(pdf_filename, 'wb') as f:
                    f.write(pdf_bytes)
                logging.info(f"PDF文件已保存到: {pdf_filename}")
            except Exception as e:
                logging.warning(f"保存PDF文件失败: {str(e)}")
        
        # 提取文本
        if pdf_bytes:
            import fitz  # PyMuPDF
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = ""
            for page_num in range(min(3, len(pdf_doc))):  # 只提取前3页
                page = pdf_doc[page_num]
                full_text += page.get_text()
            pdf_doc.close()
            
            # 限制长度
            return full_text[:5000] if len(full_text) > 5000 else full_text
        else:
            return "无法获取PDF内容"
    except Exception as e:
        return f"处理arXiv论文失败: {str(e)}"


# 创建工具对象引用（保持向后兼容）
fetch_webpage_tool = fetch_webpage_content

