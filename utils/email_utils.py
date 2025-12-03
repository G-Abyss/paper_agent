#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件处理工具
"""

import ssl
import imaplib
import time
import logging
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup
from config import QMAIL_USER, QMAIL_PASSWORD


def connect_gmail(max_retries=3, retry_delay=5):
    """连接Gmail IMAP服务器，带重试机制"""
    print("正在连接Gmail...")
    
    for attempt in range(max_retries):
        try:
            # 创建 SSL 上下文
            context = ssl.create_default_context()
            
            mail = imaplib.IMAP4_SSL("imap.qq.com", port=993, ssl_context=context)
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
    """解析邮件日期字符串为datetime对象"""
    try:
        # 使用email.utils的标准方法解析邮件日期
        return parsedate_to_datetime(date_str)
    except (ValueError, TypeError, AttributeError):
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
        now = datetime.now()
        # 结束日期：前end_days天（不包含下一天）
        end_date = (now - timedelta(days=end_days)).date()
        end_date_exclusive = end_date + timedelta(days=1)
        # 开始日期：前start_days天
        start_date = (now - timedelta(days=start_days)).date()
        
        # 只比较日期部分，忽略时间
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
    now = datetime.now()
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
    """从邮件中提取论文信息"""
    soup = BeautifulSoup(email_body, 'html.parser')
    
    papers = []
    
    # Google学术推送的结构通常包含多篇论文
    # 查找所有论文标题和链接
    for h3 in soup.find_all('h3'):
        title_link = h3.find('a')
        if title_link:
            title = title_link.get_text(strip=True)
            link = title_link.get('href', '')
            
            # 查找作者和摘要信息
            parent = h3.find_parent()
            if parent:
                text_content = parent.get_text()
                
                paper = {
                    'title': title,
                    'link': link,
                    'snippet': text_content[:500]  # 获取前500字符作为片段
                }
                papers.append(paper)
    
    return papers
