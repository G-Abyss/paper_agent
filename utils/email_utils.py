#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件处理工具

主要修改点：
- 优化代码结构和注释
"""

import ssl
import imaplib
import time
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup
from config import QMAIL_USER, QMAIL_PASSWORD, QMAIL_IMAP_SERVER, QMAIL_IMAP_PORT

# 北京时间时区（UTC+8）
BEIJING_TZ = timezone(timedelta(hours=8))


def connect_gmail(max_retries=3, retry_delay=5):
    """连接Gmail IMAP服务器，带重试机制"""
    print("正在连接Gmail...")
    
    for attempt in range(max_retries):
        try:
            # 创建 SSL 上下文
            context = ssl.create_default_context()
            
            mail = imaplib.IMAP4_SSL(QMAIL_IMAP_SERVER, port=QMAIL_IMAP_PORT, ssl_context=context)
            mail.sock.settimeout(30)  # 设置30秒超时
            
            mail.login(QMAIL_USER, QMAIL_PASSWORD)
            print("✓ QQmail连接成功")
            return mail
            
        except (imaplib.IMAP4.error, ssl.SSLError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"连接失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"✗ QQmail连接失败，已重试 {max_retries} 次")
                raise Exception(f"无法连接到Gmail: {str(e)}")
    
    raise Exception("无法连接到Gmail")


def parse_email_date(date_str):
    """
    解析邮件日期字符串为datetime对象，并转换为北京时间
    
    Args:
        date_str: 邮件日期字符串（通常是UTC时间）
    
    Returns:
        datetime: 北京时间的datetime对象，如果解析失败返回None
    """
    try:
        # 使用email.utils的标准方法解析邮件日期（通常是UTC时间）
        email_date = parsedate_to_datetime(date_str)
        
        # 如果解析出的日期没有时区信息，假设它是UTC时间
        if email_date.tzinfo is None:
            email_date = email_date.replace(tzinfo=timezone.utc)
        
        # 转换为北京时间（UTC+8）
        beijing_date = email_date.astimezone(BEIJING_TZ)
        
        return beijing_date
    except (ValueError, TypeError, AttributeError) as e:
        logging.warning(f"解析邮件日期失败: {date_str}, 错误: {str(e)}")
        # 如果标准方法失败，返回None
        return None


def is_email_in_date_range(msg, start_days=1, end_days=0):
    """
    检查邮件是否在指定的日期范围内
    
    Args:
        msg: 邮件对象
        start_days: 开始日期（前start_days天，例如start_days=3表示前3天）
        end_days: 结束日期（前end_days天，例如end_days=0表示今天，end_days=1表示昨天）
    
    Returns:
        bool: 如果邮件在日期范围内返回True，否则返回False
    """
    try:
        # 获取邮件日期
        date_str = msg.get('Date')
        if not date_str:
            return False
        
        email_date = parse_email_date(date_str)
        if not email_date:
            return False
        
        # 计算日期范围（前start_days天到前end_days天）
        # 使用北京时间的当前时间
        now = datetime.now(BEIJING_TZ)
        # 结束日期：前end_days天（不包含下一天）
        end_date = (now - timedelta(days=end_days)).date()
        end_date_exclusive = end_date + timedelta(days=1)
        # 开始日期：前start_days天
        start_date = (now - timedelta(days=start_days)).date()
        
        # 只比较日期部分，忽略时间
        # email_date 已经是北京时间，直接取日期部分
        email_date_only = email_date.date()
        
        return start_date <= email_date_only < end_date_exclusive
    except Exception as e:
        logging.warning(f"检查邮件日期时出错: {str(e)}")
        return True  # 如果无法解析日期，默认包含该邮件


def fetch_scholar_emails(mail, start_days=1, end_days=0):
    """
    获取Google学术推送邮件
    
    Args:
        mail: IMAP邮件连接对象
        start_days: 开始日期（前start_days天，例如start_days=3表示从前3天开始）
        end_days: 结束日期（前end_days天，例如end_days=0表示到今天，end_days=1表示到昨天）
    
    Returns:
        list: 邮件ID列表
    """
    # 使用北京时间的当前时间
    now = datetime.now(BEIJING_TZ)
    start_date_obj = now - timedelta(days=start_days)
    end_date_obj = now - timedelta(days=end_days)
    end_date_exclusive = end_date_obj + timedelta(days=1)
    
    start_date_str = start_date_obj.strftime("%d-%b-%Y")
    end_date_str = end_date_exclusive.strftime("%d-%b-%Y")
    
    if start_days == end_days:
        print(f"\n正在获取前{start_days}天的Google学术推送...")
    else:
        print(f"\n正在获取从前{start_days}天到前{end_days}天的Google学术推送...")
    
    # 选择收件箱
    mail.select("inbox")
    
    # 搜索Google学术邮件，使用SINCE和BEFORE限制日期范围
    search_criteria = f'(FROM "scholaralerts-noreply@google.com" SINCE {start_date_str} BEFORE {end_date_str})'
    status, messages = mail.search(None, search_criteria)
    
    email_ids = messages[0].split()
    # 反转列表，使邮件从最新到最旧排序（IMAP默认返回的是从旧到新）
    email_ids = list(reversed(email_ids))
    date_range_str = f"{start_date_obj.strftime('%Y-%m-%d')} 到 {end_date_obj.strftime('%Y-%m-%d')}"
    print(f"✓ 找到 {len(email_ids)} 封邮件（日期范围: {date_range_str}），将从最新邮件开始处理")
    
    return email_ids


def extract_paper_info(email_body):
    """
    从邮件中提取论文信息，包括标题、链接和摘要内容
    
    Google学术邮件结构：
    - h3标签包含论文标题和链接（class="gse_alrt_title"）
    - div style="color:#006621" 包含作者和期刊信息
    - div class="gse_alrt_sni" 包含摘要内容（这是关键！）
    """
    soup = BeautifulSoup(email_body, 'html.parser')
    
    papers = []
    
    # Google学术推送的结构通常包含多篇论文
    # 查找所有论文标题和链接
    for h3 in soup.find_all('h3'):
        title_link = h3.find('a')
        if not title_link:
            continue
            
        title = title_link.get_text(strip=True)
        link = title_link.get('href', '')
        
        if not title:
            continue
        
        # 查找该论文的摘要内容
        # Google学术邮件中，摘要在 class="gse_alrt_sni" 的div中
        snippet = ''
        
        # 方法1: 查找h3后面的 class="gse_alrt_sni" 的div（这是摘要）
        current = h3.next_sibling
        while current:
            if hasattr(current, 'get') and current.get('class') and 'gse_alrt_sni' in current.get('class', []):
                # 找到摘要div，提取文本
                snippet = current.get_text(separator=' ', strip=True)
                # 清理HTML标签和特殊字符
                snippet = snippet.replace('\n', ' ').replace('\r', ' ')
                # 移除多余的空白
                snippet = re.sub(r'\s+', ' ', snippet).strip()
                break
            elif hasattr(current, 'find'):
                # 在当前元素中查找gse_alrt_sni
                snippet_div = current.find('div', class_='gse_alrt_sni')
                if snippet_div:
                    snippet = snippet_div.get_text(separator=' ', strip=True)
                    snippet = snippet.replace('\n', ' ').replace('\r', ' ')
                    snippet = re.sub(r'\s+', ' ', snippet).strip()
                    break
            current = current.next_sibling
        
        # 方法2: 如果方法1失败，查找h3父元素的下一个兄弟元素中的gse_alrt_sni
        if not snippet:
            parent = h3.parent
            if parent:
                next_sibling = parent.next_sibling
                if next_sibling and hasattr(next_sibling, 'find'):
                    snippet_div = next_sibling.find('div', class_='gse_alrt_sni')
                    if snippet_div:
                        snippet = snippet_div.get_text(separator=' ', strip=True)
                        snippet = snippet.replace('\n', ' ').replace('\r', ' ')
                        snippet = re.sub(r'\s+', ' ', snippet).strip()
        
        # 方法3: 如果还是找不到，尝试在整个文档中查找h3后面的gse_alrt_sni
        if not snippet:
            # 找到h3在文档中的位置，然后查找后面的gse_alrt_sni
            all_elements = soup.find_all(['h3', 'div'])
            h3_found = False
            for elem in all_elements:
                if elem == h3:
                    h3_found = True
                    continue
                if h3_found:
                    if hasattr(elem, 'get') and elem.get('class') and 'gse_alrt_sni' in elem.get('class', []):
                        snippet = elem.get_text(separator=' ', strip=True)
                        snippet = snippet.replace('\n', ' ').replace('\r', ' ')
                        snippet = re.sub(r'\s+', ' ', snippet).strip()
                        break
                    # 如果遇到下一个h3，停止查找
                    if elem.name == 'h3' and elem.find('a'):
                        break
        
        # 限制长度，但保留足够内容用于相关性分析
        if len(snippet) > 1000:
            snippet = snippet[:1000] + '...'
        
        paper = {
            'title': title,
            'link': link,
            'snippet': snippet  # 保存完整的摘要内容，用于后续相关性分析
        }
        papers.append(paper)
    
    return papers
