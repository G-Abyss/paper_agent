#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件存储工具
用于将IMAP读取的邮件信息存储到本地JSON文件
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from email.message import Message

# 北京时间时区（UTC+8）
BEIJING_TZ = timezone(timedelta(hours=8))

# 邮件存储文件路径
EMAIL_STORAGE_FILE = 'data/emails.json'
EMAIL_STORAGE_DIR = 'data'


def ensure_storage_dir():
    """确保存储目录存在"""
    if not os.path.exists(EMAIL_STORAGE_DIR):
        os.makedirs(EMAIL_STORAGE_DIR)
        logging.info(f"创建邮件存储目录: {EMAIL_STORAGE_DIR}")


def load_emails() -> Dict:
    """
    从本地文件加载邮件信息
    
    Returns:
        dict: 邮件信息字典，格式为 {
            'last_update': '2024-01-01 12:00:00',
            'total_emails': 10,
            'emails': [
                {
                    'id': 'email_id',
                    'subject': '邮件主题',
                    'date': '2024-01-01 12:00:00',
                    'from': 'sender@example.com',
                    'paper_count': 5,
                    'papers': [...]
                },
                ...
            ]
        }
    """
    ensure_storage_dir()
    
    if not os.path.exists(EMAIL_STORAGE_FILE):
        return {
            'last_update': None,
            'total_emails': 0,
            'emails': []
        }
    
    try:
        with open(EMAIL_STORAGE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 按时间降序排序（最新的在前）
            emails = data.get('emails', [])
            if emails:
                try:
                    emails.sort(key=lambda x: x.get('date', ''), reverse=True)
                except Exception as e:
                    logging.warning(f"排序邮件时出错: {str(e)}")
                data['emails'] = emails
            return data
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"加载邮件存储文件失败: {str(e)}")
        return {
            'last_update': None,
            'total_emails': 0,
            'emails': []
        }


def save_emails(emails_data: Dict):
    """
    保存邮件信息到本地文件
    
    Args:
        emails_data: 邮件信息字典
    """
    ensure_storage_dir()
    
    try:
        with open(EMAIL_STORAGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(emails_data, f, ensure_ascii=False, indent=2)
        logging.info(f"邮件信息已保存到: {EMAIL_STORAGE_FILE}")
    except IOError as e:
        logging.error(f"保存邮件存储文件失败: {str(e)}")
        raise


def parse_email_message(msg: Message, email_id: str, papers: List[Dict]) -> Dict:
    """
    解析邮件消息对象为存储格式
    
    Args:
        msg: 邮件消息对象
        email_id: 邮件ID
        papers: 从邮件中提取的论文列表
    
    Returns:
        dict: 邮件信息字典
    """
    # 为每个paper添加处理状态（初始为待处理）
    for paper in papers:
        if 'processing_status' not in paper:
            paper['processing_status'] = 'pending'  # pending: 待处理, processing: 处理中, processed: 已处理
        if 'paper_id' not in paper:
            # 生成唯一ID：email_id + title的hash
            import hashlib
            title = paper.get('title', '')
            paper_id_str = f"{email_id}_{title}"
            paper['paper_id'] = hashlib.md5(paper_id_str.encode('utf-8')).hexdigest()[:16]
    # 获取邮件日期（转换为北京时间）
    date_str = msg.get('Date', '')
    email_date = None
    if date_str:
        try:
            from email.utils import parsedate_to_datetime
            email_date_obj = parsedate_to_datetime(date_str)
            if email_date_obj.tzinfo is None:
                email_date_obj = email_date_obj.replace(tzinfo=timezone.utc)
            email_date_obj = email_date_obj.astimezone(BEIJING_TZ)
            email_date = email_date_obj.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logging.warning(f"解析邮件日期失败: {str(e)}")
            email_date = datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
    
    # 获取邮件主题
    subject = msg.get('Subject', '无主题')
    # 解码邮件主题（可能包含编码）
    try:
        from email.header import decode_header
        decoded_parts = decode_header(subject)
        decoded_subject = ''
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                if encoding:
                    decoded_subject += part.decode(encoding)
                else:
                    decoded_subject += part.decode('utf-8', errors='ignore')
            else:
                decoded_subject += part
        subject = decoded_subject
    except Exception:
        pass
    
    # 获取发件人
    from_addr = msg.get('From', '未知发件人')
    
    return {
        'id': email_id,
        'subject': subject,
        'date': email_date or datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S'),
        'from': from_addr,
        'paper_count': len(papers),
        'papers': papers
    }


def update_email_storage(email_list: List[Dict]):
    """
    更新邮件存储
    
    Args:
        email_list: 邮件列表，每个元素包含 'msg', 'email_id', 'papers'
    """
    ensure_storage_dir()
    
    # 加载现有邮件
    existing_data = load_emails()
    
    # 创建现有paper_id到处理状态的映射（用于保留处理状态）
    existing_paper_status_map = {}
    for email in existing_data.get('emails', []):
        for paper in email.get('papers', []):
            paper_id = paper.get('paper_id')
            if paper_id:
                existing_paper_status_map[paper_id] = paper.get('processing_status', 'pending')
    
    # 创建邮件ID到邮件的映射（用于去重和更新）
    existing_email_map = {email['id']: email for email in existing_data.get('emails', [])}
    
    # 解析新邮件
    new_emails = []
    updated_emails = []
    for email_info in email_list:
        email_id = email_info.get('email_id', '')
        msg = email_info.get('msg')
        papers = email_info.get('papers', [])
        
        if not msg:
            continue
        
        if email_id in existing_email_map:
            # 如果邮件已存在，更新论文列表（保留已有论文的处理状态）
            existing_email = existing_email_map[email_id]
            existing_papers = existing_email.get('papers', [])
            existing_paper_map = {p.get('paper_id'): p for p in existing_papers if p.get('paper_id')}
            
            # 合并新论文（保留已有论文的处理状态）
            for paper in papers:
                paper_id = paper.get('paper_id')
                if paper_id and paper_id in existing_paper_map:
                    # 保留已有论文的处理状态和其他信息
                    existing_paper = existing_paper_map[paper_id]
                    paper['processing_status'] = existing_paper.get('processing_status', 'pending')
                    # 保留其他处理信息
                    if 'relevance_score' in existing_paper:
                        paper['relevance_score'] = existing_paper['relevance_score']
                    if 'relevance_explanation' in existing_paper:
                        paper['relevance_explanation'] = existing_paper['relevance_explanation']
                    if 'title_cn' in existing_paper:
                        paper['title_cn'] = existing_paper['title_cn']
                elif paper_id and paper_id in existing_paper_status_map:
                    # 如果paper_id存在于其他邮件中，保留其处理状态
                    paper['processing_status'] = existing_paper_status_map[paper_id]
            
            # 更新邮件
            parsed_email = parse_email_message(msg, email_id, papers)
            parsed_email['papers'] = papers  # 使用更新后的papers列表
            updated_emails.append(parsed_email)
        else:
            # 新邮件
            parsed_email = parse_email_message(msg, email_id, papers)
            # 检查是否有已存在的paper_id，保留其处理状态
            for paper in parsed_email.get('papers', []):
                paper_id = paper.get('paper_id')
                if paper_id and paper_id in existing_paper_status_map:
                    paper['processing_status'] = existing_paper_status_map[paper_id]
            new_emails.append(parsed_email)
    
    # 合并新旧邮件（更新已存在的邮件）
    all_emails = []
    updated_email_ids = {e['id'] for e in updated_emails}
    for email in existing_data.get('emails', []):
        if email['id'] not in updated_email_ids:
            all_emails.append(email)
    all_emails.extend(updated_emails)
    all_emails.extend(new_emails)
    
    # 按时间降序排序（最新的在前）
    try:
        all_emails.sort(key=lambda x: x.get('date', ''), reverse=True)
    except Exception as e:
        logging.warning(f"排序邮件时出错: {str(e)}")
    
    # 更新数据
    updated_data = {
        'last_update': datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S'),
        'total_emails': len(all_emails),
        'emails': all_emails
    }
    
    # 保存到文件
    save_emails(updated_data)
    
    return updated_data


def get_email_summary() -> Dict:
    """
    获取邮件摘要信息（用于前端显示）
    
    Returns:
        dict: 邮件摘要信息
    """
    data = load_emails()
    emails = data.get('emails', [])
    
    # 确保邮件已按时间降序排序（load_emails已排序，这里再次确认）
    if emails:
        try:
            emails.sort(key=lambda x: x.get('date', ''), reverse=True)
        except Exception as e:
            logging.warning(f"排序邮件摘要时出错: {str(e)}")
    
    return {
        'last_update': data.get('last_update'),
        'total_emails': data.get('total_emails', 0),
        'total_papers': sum(email.get('paper_count', 0) for email in emails),
        'recent_emails': emails[:10]  # 最近10封邮件（已排序，最新的在前）
    }


def filter_emails_by_date_range(start_days: int, end_days: int) -> List[Dict]:
    """
    从本地存储中根据时间范围筛选邮件
    
    Args:
        start_days: 开始日期（前start_days天，例如start_days=3表示前3天）
        end_days: 结束日期（前end_days天，例如end_days=0表示今天，end_days=1表示昨天）
    
    Returns:
        list: 符合条件的邮件列表
    """
    data = load_emails()
    all_emails = data.get('emails', [])
    
    if not all_emails:
        return []
    
    # 计算日期范围（前start_days天到前end_days天）
    now = datetime.now(BEIJING_TZ)
    end_date = (now - timedelta(days=end_days)).date()
    end_date_exclusive = end_date + timedelta(days=1)
    start_date = (now - timedelta(days=start_days)).date()
    
    filtered_emails = []
    for email_item in all_emails:
        try:
            # 解析邮件日期
            email_date_str = email_item.get('date', '')
            if not email_date_str:
                continue
            
            email_date = datetime.strptime(email_date_str, '%Y-%m-%d %H:%M:%S')
            email_date_only = email_date.date()
            
            # 检查是否在日期范围内
            if start_date <= email_date_only < end_date_exclusive:
                filtered_emails.append(email_item)
        except Exception as e:
            logging.warning(f"筛选邮件日期时出错: {str(e)}")
            continue
    
    # 按时间降序排序（最新的在前）
    try:
        filtered_emails.sort(key=lambda x: x.get('date', ''), reverse=True)
    except Exception as e:
        logging.warning(f"排序筛选后的邮件时出错: {str(e)}")
    
    return filtered_emails


def sync_remote_emails_to_local(mail, start_days: int = 30, end_days: int = 0) -> Dict:
    """
    同步远程邮箱到本地存储（不限制数量，获取所有符合条件的邮件）
    
    Args:
        mail: IMAP邮件连接对象
        start_days: 开始日期（前start_days天，默认30天）
        end_days: 结束日期（前end_days天，默认0表示今天）
    
    Returns:
        dict: 同步结果，包含 'updated_count', 'total_count' 等信息
    """
    from utils.email_utils import fetch_scholar_emails, extract_paper_info, is_email_in_date_range
    import email
    
    # 获取邮件ID列表（不限制数量）
    email_ids = fetch_scholar_emails(mail, start_days=start_days, end_days=end_days)
    
    if not email_ids:
        return {
            'updated_count': 0,
            'total_count': 0,
            'message': '没有找到邮件'
        }
    
    # 处理所有邮件
    email_storage_list = []
    
    for email_id in email_ids:
        email_id_str = email_id.decode() if isinstance(email_id, bytes) else str(email_id)
        
        try:
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            
            if status != 'OK':
                continue
            
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    # 验证邮件日期是否在指定范围内
                    if not is_email_in_date_range(msg, start_days=start_days, end_days=end_days):
                        continue
                    
                    # 获取邮件正文
                    body = None
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/html":
                                body = part.get_payload(decode=True).decode()
                                break
                    else:
                        body = msg.get_payload(decode=True).decode()
                    
                    if not body:
                        continue
                    
                    # 提取论文信息
                    papers = extract_paper_info(body)
                    
                    # 保存邮件信息到存储列表
                    email_storage_list.append({
                        'msg': msg,
                        'email_id': email_id_str,
                        'papers': papers
                    })
        except Exception as e:
            logging.warning(f"处理邮件 {email_id_str} 时出错: {str(e)}")
            continue
    
    # 更新邮件存储
    updated_count = 0
    if email_storage_list:
        updated_data = update_email_storage(email_storage_list)
        updated_count = len(email_storage_list)
    
    return {
        'updated_count': updated_count,
        'total_count': len(email_ids),
        'message': f'成功同步 {updated_count} 封邮件'
    }


def get_pending_papers() -> List[Dict]:
    """
    获取所有待处理的文章列表
    
    Returns:
        List[Dict]: 待处理文章列表，每个元素包含文章信息和所属邮件信息
    """
    emails_data = load_emails()
    pending_papers = []
    
    for email in emails_data.get('emails', []):
        for paper in email.get('papers', []):
            if paper.get('processing_status', 'pending') == 'pending':
                paper_with_email = paper.copy()
                paper_with_email['email_id'] = email.get('id')
                paper_with_email['email_subject'] = email.get('subject')
                paper_with_email['email_date'] = email.get('date')
                pending_papers.append(paper_with_email)
    
    return pending_papers


def get_processing_papers() -> List[Dict]:
    """
    获取所有处理中的文章列表（状态为processing的文章）
    
    Returns:
        List[Dict]: 处理中文章列表，每个元素包含文章信息和所属邮件信息
    """
    emails_data = load_emails()
    processing_papers = []
    
    for email in emails_data.get('emails', []):
        for paper in email.get('papers', []):
            if paper.get('processing_status', 'pending') == 'processing':
                paper_with_email = paper.copy()
                paper_with_email['email_id'] = email.get('id')
                paper_with_email['email_subject'] = email.get('subject')
                paper_with_email['email_date'] = email.get('date')
                processing_papers.append(paper_with_email)
    
    return processing_papers


def update_paper_processing_status(paper_id: str, status: str, **kwargs) -> bool:
    """
    更新文章的处理状态
    
    Args:
        paper_id: 文章ID
        status: 处理状态 ('pending': 待处理, 'processing': 处理中, 'processed': 已处理)
        **kwargs: 其他要更新的字段（如relevance_score, relevance_explanation, title_cn等）
    
    Returns:
        bool: 是否更新成功
    """
    try:
        emails_data = load_emails()
        updated = False
        
        for email in emails_data.get('emails', []):
            for paper in email.get('papers', []):
                if paper.get('paper_id') == paper_id:
                    paper['processing_status'] = status
                    # 更新其他字段
                    for key, value in kwargs.items():
                        paper[key] = value
                    updated = True
                    break
            if updated:
                break
        
        if updated:
            save_emails(emails_data)
        
        return updated
    except Exception as e:
        logging.error(f"更新文章处理状态失败: {str(e)}", exc_info=True)
        return False


def get_paper_by_id(paper_id: str) -> Optional[Dict]:
    """
    根据paper_id获取文章信息
    
    Args:
        paper_id: 文章ID
    
    Returns:
        Optional[Dict]: 文章信息，如果不存在则返回None
    """
    emails_data = load_emails()
    
    for email in emails_data.get('emails', []):
        for paper in email.get('papers', []):
            if paper.get('paper_id') == paper_id:
                paper_with_email = paper.copy()
                paper_with_email['email_id'] = email.get('id')
                paper_with_email['email_subject'] = email.get('subject')
                paper_with_email['email_date'] = email.get('date')
                return paper_with_email
    
    return None

