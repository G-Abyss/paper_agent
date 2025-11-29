#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Summarizer - 自动总结Google学术邮件推送
"""

import os
import imaplib
import email
from email.header import decode_header
import re
from datetime import datetime, timedelta
import yaml
from dotenv import load_dotenv
import ollama
from bs4 import BeautifulSoup
import time
import ssl
from crewai import Agent, Task, Crew, LLM
import logging
import pandas as pd
import requests
from urllib.parse import urlparse
import fitz  # PyMuPDF
from io import BytesIO
from crawl4ai import AsyncWebCrawler
import asyncio
import warnings
import atexit
import sys
from contextlib import contextmanager

# 禁用 CrewAI 遥测（可选）
os.environ['CREWAI_TELEMETRY_OPT_OUT'] = 'true'
os.environ['OTEL_SDK_DISABLED'] = 'true'

# 注册退出时的清理函数
def cleanup_litellm_on_exit():
    """程序退出时清理 LiteLLM 异步客户端"""
    try:
        import litellm
        if hasattr(litellm, 'close_litellm_async_clients'):
            try:
                # 创建新的事件循环来执行清理
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(litellm.close_litellm_async_clients())
                finally:
                    # 清理并关闭事件循环
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.close()
            except Exception:
                pass
    except (ImportError, AttributeError):
        pass

# 注册退出处理函数
atexit.register(cleanup_litellm_on_exit)

# 抑制 LiteLLM 异步客户端清理警告（如果清理失败）
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       message=".*coroutine 'close_litellm_async_clients' was never awaited.*")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置 CrewAI 专用日志输出到文件
CREWAI_LOG_DIR = 'crewai_logs'  # CrewAI 日志目录
os.makedirs(CREWAI_LOG_DIR, exist_ok=True)
crewai_log_file = os.path.join(CREWAI_LOG_DIR, f'crewai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# 创建 CrewAI 日志文件处理器
crewai_file_handler = logging.FileHandler(crewai_log_file, encoding='utf-8')
crewai_file_handler.setLevel(logging.DEBUG)
crewai_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
crewai_file_handler.setFormatter(crewai_formatter)

# 配置 CrewAI 相关的日志记录器
crewai_logger = logging.getLogger('crewai')
crewai_logger.setLevel(logging.DEBUG)
crewai_logger.addHandler(crewai_file_handler)
crewai_logger.propagate = False  # 防止日志传播到根日志记录器（避免重复输出到控制台）

# 同时配置 CrewAI 依赖库的日志
for logger_name in ['crewai.agent', 'crewai.task', 'crewai.crew', 'litellm']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(crewai_file_handler)
    logger.propagate = False

logging.info(f"CrewAI 处理过程日志将保存到: {crewai_log_file}")


# ANSI转义码清理函数
def remove_ansi_codes(text):
    """
    移除ANSI转义码（终端颜色代码）
    例如：[32m, [0m, [1;32m 等
    """
    import re
    # 匹配ANSI转义序列：\x1b[ 或 \033[ 开头，后面跟着数字、分号、字母，以m结尾
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m|\033\[[0-9;]*m')
    return ansi_escape.sub('', text)


# 创建 CrewAI 输出捕获类
class CrewAILogWriter:
    """捕获 CrewAI 的 print 输出并写入文件"""
    def __init__(self, log_file, log_callback=None):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.file = open(log_file, 'a', encoding='utf-8')
        self.log_callback = log_callback  # 用于实时发送日志的回调函数
        self.buffer = ''  # 累积消息缓冲区
        self.last_flush_time = 0  # 上次刷新时间
    
    def write(self, message):
        """写入消息到文件和终端"""
        # 终端输出保持原样（带颜色）
        self.terminal.write(message)
        # 文件输出移除ANSI转义码（更易读）
        cleaned_message = remove_ansi_codes(message)
        self.file.write(cleaned_message)
        self.file.flush()
        
        # 累积消息到缓冲区
        if cleaned_message:
            self.buffer += cleaned_message
    
    def flush(self):
        """刷新缓冲区"""
        self.terminal.flush()
        self.file.flush()
        # 不再发送任何日志到前端
        self.buffer = ''
    
    def close(self):
        """关闭文件"""
        if self.file:
            self.file.close()


# 全局 CrewAI 输出捕获器
crewai_output_capture = None
crewai_log_callback = None  # 全局日志回调函数
agent_status_callback = None  # 全局agent状态回调函数


@contextmanager
def capture_crewai_output(log_callback=None):
    """上下文管理器：捕获 CrewAI 的输出到文件"""
    global crewai_output_capture, crewai_log_callback
    
    # 使用传入的回调或全局回调
    callback = log_callback if log_callback is not None else crewai_log_callback
    
    if crewai_output_capture is None or (callback and crewai_output_capture.log_callback != callback):
        crewai_output_capture = CrewAILogWriter(crewai_log_file, log_callback=callback)
    elif callback:
        # 更新回调函数
        crewai_output_capture.log_callback = callback
    
    # 保存原始 stdout
    original_stdout = sys.stdout
    
    try:
        # 重定向 stdout 到文件
        sys.stdout = crewai_output_capture
        yield
    finally:
        # 恢复原始 stdout
        sys.stdout = original_stdout

# 加载环境变量
load_dotenv()

# 配置
# GMAIL_USER = os.getenv('GMAIL_USER')
# GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
QMAIL_USER = os.getenv('QMAIL_USER')
QMAIL_PASSWORD = os.getenv('QMAIL_PASSWORD')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:32b')
MAX_EMAILS = int(os.getenv('MAX_EMAILS', 20))
# MAX_EMAILS = 23
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://192.168.2.169:11434')
# 日期范围配置：从前START_DAYS天到前END_DAYS天
# 例如：START_DAYS=3, END_DAYS=0 表示从前3天到今天
#      START_DAYS=7, END_DAYS=3 表示从前7天到前3天
START_DAYS = int(os.getenv('START_DAYS', 1))  # 默认从前1天开始
END_DAYS = int(os.getenv('END_DAYS', 0))  # 默认到今天（前0天）
# START_DAYS = 4  # 默认从前1天开始
# END_DAYS = 0  # 默认到今天（前0天）
# 备份路径配置（可选）：如果设置了此路径，报告会同时保存到该路径
BACKUP_DIR = os.getenv('BACKUP_DIR', '')  # 默认为空，不进行备份
# 调试模式配置
DEBUG_MODE = os.getenv('DEBUG_MODE', '0') == '1'  # 从环境变量读取，默认为False
# DEBUG_MODE = True
DEBUG_DIR = 'debug'  # 调试文件输出目录

# 本地处理模式配置
LOCAL_MODE = os.getenv('LOCAL', '0') == '1'  # 从环境变量读取，默认为False
CSV_FILE_PATH = os.getenv('CSV_FILE_PATH', 'papers.csv')  # CSV文件路径
CSV_TITLE_COLUMN = int(os.getenv('CSV_TITLE_COLUMN', '0'))  # 论文标题列索引（从0开始，默认第1列）
CSV_ABSTRACT_COLUMN = int(os.getenv('CSV_ABSTRACT_COLUMN', '2'))  # 摘要列索引（从0开始，默认第3列）
CSV_LINK_COLUMN = os.getenv('CSV_LINK_COLUMN', '')  # 论文链接列索引（可选，从0开始，如果为空则不读取链接）

# 设置环境变量（CrewAI 通过 LiteLLM 连接 Ollama 需要这些）
os.environ['OLLAMA_API_BASE'] = OLLAMA_BASE_URL
if not os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = 'ollama'  # 占位符，实际不使用

# 初始化 CrewAI LLM
# 关键：模型名称必须包含 "ollama/" 前缀
llm_model_name = f"ollama/{OLLAMA_MODEL}" if not OLLAMA_MODEL.startswith("ollama/") else OLLAMA_MODEL

logging.info(f"初始化 CrewAI LLM: model={llm_model_name}, base_url={OLLAMA_BASE_URL}")

llm = LLM(
    model=llm_model_name,
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"  # Ollama 不需要真实的 API key
)

# 加载关键词
with open('keywords.yaml', 'r', encoding='utf-8') as f:
    KEYWORDS = yaml.safe_load(f)

HIGH_PRIORITY_KEYWORDS = [kw.lower() for kw in KEYWORDS['high_priority']]
RELATED_KEYWORDS = [kw.lower() for kw in KEYWORDS['related']]


# 调试功能
class DebugLogger:
    """调试日志记录器"""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.debug_file = None
        self.debug_dir = DEBUG_DIR
        if self.enabled:
            # 创建调试目录
            os.makedirs(self.debug_dir, exist_ok=True)
            # 创建调试文件（带时间戳）
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            debug_filename = os.path.join(self.debug_dir, f'debug_{timestamp}.log')
            self.debug_file = open(debug_filename, 'w', encoding='utf-8')
            self.log("=" * 80)
            self.log(f"调试模式已启用 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log("=" * 80)
            self.log("")
    
    def log(self, message, level="INFO"):
        """记录调试信息"""
        if self.enabled and self.debug_file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_message = f"[{timestamp}] [{level}] {message}\n"
            self.debug_file.write(log_message)
            self.debug_file.flush()  # 立即写入文件
    
    def log_separator(self, title=""):
        """记录分隔线"""
        if self.enabled:
            if title:
                self.log("")
                self.log("=" * 80)
                self.log(f"  {title}")
                self.log("=" * 80)
            else:
                self.log("-" * 80)
    
    def log_paper_info(self, paper, index=None):
        """记录论文信息"""
        if self.enabled:
            if index is not None:
                self.log_separator(f"论文 #{index}: {paper.get('title', 'Unknown')[:60]}")
            else:
                self.log_separator(f"论文: {paper.get('title', 'Unknown')[:60]}")
            self.log(f"标题: {paper.get('title', 'N/A')}")
            self.log(f"链接: {paper.get('link', 'N/A')}")
            self.log(f"原始片段: {paper.get('snippet', 'N/A')[:200]}...")
            self.log("")
    
    def log_abstract_extraction(self, paper_title, url, source_type, raw_content=None, extracted_abstract=None, agent_result=None):
        """记录摘要提取过程"""
        if self.enabled:
            self.log_separator(f"摘要提取 - {source_type}")
            self.log(f"论文标题: {paper_title}")
            self.log(f"URL: {url}")
            self.log(f"来源类型: {source_type}")
            
            if raw_content:
                self.log("")
                self.log("原始内容预览:")
                self.log("-" * 80)
                content_preview = raw_content[:2000] if len(raw_content) > 2000 else raw_content
                self.log(content_preview)
                if len(raw_content) > 2000:
                    self.log(f"... (共 {len(raw_content)} 字符，已截断)")
                self.log("-" * 80)
            
            if agent_result:
                self.log("")
                self.log("Agent处理结果:")
                self.log("-" * 80)
                self.log(agent_result)
                self.log("-" * 80)
            
            if extracted_abstract:
                self.log("")
                self.log("最终提取的摘要:")
                self.log("-" * 80)
                self.log(extracted_abstract)
                self.log(f"摘要长度: {len(extracted_abstract)} 字符")
                self.log("-" * 80)
            
            self.log("")
    
    def close(self):
        """关闭调试文件"""
        if self.enabled and self.debug_file:
            self.log("")
            self.log("=" * 80)
            self.log(f"调试日志结束 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log("=" * 80)
            self.debug_file.close()
            self.debug_file = None


# 创建全局调试日志记录器
debug_logger = DebugLogger(enabled=DEBUG_MODE)


def connect_gmail(max_retries=3, retry_delay=5):
    """连接Gmail IMAP服务器，带重试机制"""
    print("正在连接Gmail...")
    
    for attempt in range(max_retries):
        try:
            # 创建 SSL 上下文
            context = ssl.create_default_context()
            
            # 使用超时设置连接
            # mail = imaplib.IMAP4_SSL("imap.gmail.com", port=993, ssl_context=context)
            # mail.sock.settimeout(30)  # 设置30秒超时
            
            # mail.login(GMAIL_USER, GMAIL_PASSWORD)
            # print("✓ Gmail连接成功")

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
        from email.utils import parsedate_to_datetime
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
    # search_criteria = f'(FROM "ligen4073187@gmail.com" SINCE {since_date}) AND (HEADER FROM "scholaralerts-noreply@google.com")'
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


def check_relevance(paper):
    """检查论文相关性"""
    text = (paper['title'] + ' ' + paper['snippet']).lower()
    
    # 检查高优先级关键词
    high_priority_matches = sum(1 for kw in HIGH_PRIORITY_KEYWORDS if kw in text)
    
    # 检查相关关键词
    related_matches = sum(1 for kw in RELATED_KEYWORDS if kw in text)
    
    # 计算相关性分数
    relevance_score = high_priority_matches * 2 + related_matches
    
    return relevance_score, high_priority_matches > 0


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
        from urllib.parse import parse_qs, unquote
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
            from urllib.parse import parse_qs, unquote
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


def get_full_abstract(paper):
    """
    获取论文的完整摘要
    返回: (abstract_text, is_pdf)
    """
    url = paper.get('link', '')
    paper_title = paper.get('title', '')
    
    if not url:
        logging.warning(f"论文 {paper_title} 没有链接")
        debug_logger.log(f"论文 {paper_title} 没有链接，无法提取摘要", "WARNING")
        return None, False  # 没有URL，无法提取摘要，返回None
    
    print(f"    正在访问论文网址: {url[:80]}...")
    debug_logger.log(f"开始获取摘要 - URL: {url}")
    
    html_content, is_pdf, pdf_bytes = fetch_fulltext_from_url(url)
    
    if is_pdf and pdf_bytes:
        print(f"    检测到PDF文件，正在提取文本...")
        debug_logger.log(f"检测到PDF文件，开始提取文本...")
        
        # 保存PDF文件到downloads文件夹
        try:
            downloads_dir = 'downloads'
            os.makedirs(downloads_dir, exist_ok=True)
            # 使用论文标题生成文件名（清理非法字符）
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', paper_title[:100])  # 限制长度并替换非法字符
            pdf_filename = os.path.join(downloads_dir, f"{safe_title}.pdf")
            # 如果文件已存在，添加时间戳
            if os.path.exists(pdf_filename):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                pdf_filename = os.path.join(downloads_dir, f"{safe_title}_{timestamp}.pdf")
            with open(pdf_filename, 'wb') as f:
                f.write(pdf_bytes)
            print(f"    ✓ PDF文件已保存到: {pdf_filename}")
            debug_logger.log(f"PDF文件已保存到: {pdf_filename}")
        except Exception as e:
            logging.warning(f"保存PDF文件失败: {str(e)}")
            debug_logger.log(f"保存PDF文件失败: {str(e)}", "WARNING")
        
        full_text, abstract, body = extract_text_from_pdf(pdf_bytes)
        
        if full_text:
            debug_logger.log(f"PDF文本提取成功，总长度: {len(full_text)} 字符")
            debug_logger.log(f"正则提取的摘要长度: {len(abstract) if abstract else 0} 字符")
            
            # 使用PDF处理agent来准确提取摘要
            print(f"    使用AI agent清洗和提取摘要...")
            debug_logger.log("开始使用PDF处理agent提取摘要...")
            try:
                # 提取前几页的文本（通常摘要在前3页）
                # 限制文本长度，避免超出token限制
                pdf_text_for_agent = full_text[:8000]  # 前8000字符，通常足够包含摘要
                
                debug_logger.log_abstract_extraction(
                    paper_title=paper_title,
                    url=url,
                    source_type="PDF",
                    raw_content=pdf_text_for_agent,
                    extracted_abstract=None
                )
                
                # 创建PDF摘要提取任务
                pdf_task = create_pdf_abstract_extraction_task(pdf_text_for_agent, paper_title)
                pdf_agent = create_pdf_processor_agent()
                
                # 发送agent工作开始状态
                send_agent_status("PDF摘要提取专家", "start", task=pdf_task)
                
                with capture_crewai_output():
                    pdf_crew = Crew(
                        agents=[pdf_agent],
                        tasks=[pdf_task],
                        verbose=True,
                        share_crew=False
                    )
                    result = pdf_crew.kickoff()
                
                # 发送agent工作结束状态
                send_agent_status("PDF摘要提取专家", "end", result=result)
                
                # 记录 Agent 的输入和输出
                if crewai_log_callback:
                    log_agent_io("PDF摘要提取专家", pdf_task, result, crewai_log_callback)
                
                agent_output = result.raw.strip()
                
                debug_logger.log(f"Agent返回结果长度: {len(agent_output)} 字符")
                debug_logger.log_abstract_extraction(
                    paper_title=paper_title,
                    url=url,
                    source_type="PDF",
                    raw_content=pdf_text_for_agent,
                    extracted_abstract=None,
                    agent_result=agent_output
                )
                
                # 解析agent输出：提取结果（0或1）和摘要内容
                extraction_result = None
                abstract_content = ""
                
                # 查找"提取结果："后面的数字
                result_match = re.search(r'提取结果[：:]\s*([01])', agent_output, re.I)
                if result_match:
                    extraction_result = int(result_match.group(1))
                
                # 如果没找到，尝试查找独立的0或1
                if extraction_result is None:
                    # 查找行首的0或1
                    first_line_match = re.match(r'^\s*([01])\s*$', agent_output.split('\n')[0] if agent_output else '')
                    if first_line_match:
                        extraction_result = int(first_line_match.group(1))
                
                # 如果还是没找到，检查是否包含"提取结果：0"或"提取结果：1"
                if extraction_result is None:
                    if re.search(r'提取结果[：:]\s*0|结果[：:]\s*0|^0\s*$', agent_output, re.I | re.M):
                        extraction_result = 0
                    elif re.search(r'提取结果[：:]\s*1|结果[：:]\s*1|^1\s*$', agent_output, re.I | re.M):
                        extraction_result = 1
                
                # 如果提取结果为0，直接返回None
                if extraction_result == 0:
                    print(f"    ✗ Agent反馈：提取结果=0（摘要提取失败）")
                    debug_logger.log(f"Agent反馈提取结果=0，摘要提取失败", "WARNING")
                    return None, True
                
                # 如果提取结果为1，提取摘要内容
                if extraction_result == 1:
                    # 检查是否包含生成内容的标志性关键词（如果包含，说明是生成的内容，不是提取的）
                    generation_keywords = [
                        'simulated', 'generated', 'based on the', 'not provided', 'not available',
                        'cannot extract', 'unable to', 'assumed', 'presumed', 'inferred',
                        '推测', '生成', '模拟', '假设', '推断', '未提供', '无法提取',
                        'Note:', 'note:', '注意', '说明', '备注'
                    ]
                    agent_output_lower = agent_output.lower()
                    is_generated = any(keyword in agent_output_lower for keyword in generation_keywords)
                    
                    if is_generated:
                        print(f"    ✗ Agent反馈：检测到生成内容的关键词，视为提取失败")
                        debug_logger.log(f"检测到生成内容的关键词，视为提取失败", "WARNING")
                        return None, True
                    
                    # 查找"摘要内容："后面的内容
                    abstract_match = re.search(r'摘要内容[：:]\s*(.+?)(?:\n\n|\n提取结果|$)', agent_output, re.DOTALL | re.I)
                    if abstract_match:
                        abstract_content = abstract_match.group(1).strip()
                    else:
                        # 如果没有找到"摘要内容："标记，尝试提取"提取结果：1"之后的所有内容
                        after_result = re.split(r'提取结果[：:]\s*1\s*\n?', agent_output, flags=re.I)
                        if len(after_result) > 1:
                            abstract_content = after_result[1].strip()
                            # 移除可能的"摘要内容："标记
                            abstract_content = re.sub(r'^摘要内容[：:]\s*', '', abstract_content, flags=re.I)
                    
                    # 再次检查提取的内容是否包含生成标志
                    if abstract_content:
                        abstract_lower = abstract_content.lower()
                        if any(keyword in abstract_lower for keyword in generation_keywords):
                            print(f"    ✗ Agent反馈：摘要内容中包含生成标志，视为提取失败")
                            debug_logger.log(f"摘要内容中包含生成标志，视为提取失败", "WARNING")
                            return None, True
                
                    if abstract_content and len(abstract_content) > 50:
                        print(f"    ✓ AI agent成功提取摘要 ({len(abstract_content)} 字符)")
                        debug_logger.log(f"✓ 使用Agent提取的摘要（长度: {len(abstract_content)} 字符）", "SUCCESS")
                        return abstract_content, True
                    else:
                        print(f"    ✗ Agent反馈：提取结果=1但摘要内容为空或太短")
                        debug_logger.log(f"Agent反馈提取结果=1但摘要内容为空或太短", "WARNING")
                        return None, True
                
                # 如果无法解析提取结果，视为提取失败
                print(f"    ✗ 无法解析Agent提取结果，视为提取失败")
                debug_logger.log(f"无法解析Agent提取结果，视为提取失败", "WARNING")
                return None, True  # 无法解析，视为提取失败，返回None
            except Exception as e:
                logging.error(f"PDF处理agent失败: {str(e)}")
                debug_logger.log(f"PDF处理agent失败: {str(e)}", "ERROR")
                # Agent失败，视为提取失败，返回None
                return None, True
        else:
            logging.warning(f"无法从PDF提取文本: {url}")
            debug_logger.log(f"无法从PDF提取文本: {url}", "ERROR")
            return None, True  # 无法提取文本，返回None
    elif html_content:
        print(f"    正在从网页提取摘要...")
        debug_logger.log(f"检测到网页内容，开始提取摘要...")
        debug_logger.log(f"HTML内容长度: {len(html_content)} 字符")
        
        # 先尝试使用BeautifulSoup快速提取（作为备选）
        quick_abstract = extract_abstract_from_html(html_content)
        debug_logger.log(f"BeautifulSoup快速提取结果长度: {len(quick_abstract) if quick_abstract else 0} 字符")
        
        # 使用网页摘要提取agent来准确提取摘要
        print(f"    使用AI agent清洗和提取摘要...")
        debug_logger.log("开始使用网页摘要提取agent提取摘要...")
        try:
            # 限制HTML内容长度，避免超出token限制
            html_for_agent = html_content[:15000] if len(html_content) > 15000 else html_content
            
            debug_logger.log_abstract_extraction(
                paper_title=paper_title,
                url=url,
                source_type="网页",
                raw_content=html_for_agent,
                extracted_abstract=None
            )
            
            # 创建网页摘要提取任务
            web_task = create_web_abstract_extraction_task(html_for_agent, paper_title, url)
            web_agent = create_web_abstract_extractor_agent()
            
            # 发送agent工作开始状态
            send_agent_status("网页摘要提取专家", "start", task=web_task)
            
            with capture_crewai_output():
                web_crew = Crew(
                    agents=[web_agent],
                    tasks=[web_task],
                    verbose=True,
                    share_crew=False
                )
                result = web_crew.kickoff()
            
            # 发送agent工作结束状态
            send_agent_status("网页摘要提取专家", "end", result=result)
            
            # 记录 Agent 的输入和输出
            if crewai_log_callback:
                log_agent_io("网页摘要提取专家", web_task, result, crewai_log_callback)
            
            agent_output = result.raw.strip()
            
            debug_logger.log(f"Agent返回结果长度: {len(agent_output)} 字符")
            debug_logger.log_abstract_extraction(
                paper_title=paper_title,
                url=url,
                source_type="网页",
                raw_content=html_for_agent,
                extracted_abstract=None,
                agent_result=agent_output
            )
            
            # 解析agent输出：提取结果（0或1）和摘要内容
            extraction_result = None
            abstract_content = ""
            
            # 查找"提取结果："后面的数字
            result_match = re.search(r'提取结果[：:]\s*([01])', agent_output, re.I)
            if result_match:
                extraction_result = int(result_match.group(1))
            
            # 如果没找到，尝试查找独立的0或1
            if extraction_result is None:
                # 查找行首的0或1
                first_line_match = re.match(r'^\s*([01])\s*$', agent_output.split('\n')[0] if agent_output else '')
                if first_line_match:
                    extraction_result = int(first_line_match.group(1))
            
            # 如果还是没找到，检查是否包含"提取结果：0"或"提取结果：1"
            if extraction_result is None:
                if re.search(r'提取结果[：:]\s*0|结果[：:]\s*0|^0\s*$', agent_output, re.I | re.M):
                    extraction_result = 0
                elif re.search(r'提取结果[：:]\s*1|结果[：:]\s*1|^1\s*$', agent_output, re.I | re.M):
                    extraction_result = 1
            
            # 如果提取结果为0，直接返回None
            if extraction_result == 0:
                print(f"    ✗ Agent反馈：提取结果=0（摘要提取失败）")
                debug_logger.log(f"Agent反馈提取结果=0，摘要提取失败", "WARNING")
                return None, False
            
            # 如果提取结果为1，提取摘要内容
            if extraction_result == 1:
                # 检查是否包含生成内容的标志性关键词（如果包含，说明是生成的内容，不是提取的）
                generation_keywords = [
                    'simulated', 'generated', 'based on the', 'not provided', 'not available',
                    'cannot extract', 'unable to', 'assumed', 'presumed', 'inferred',
                    '推测', '生成', '模拟', '假设', '推断', '未提供', '无法提取',
                    'Note:', 'note:', '注意', '说明', '备注'
                ]
                agent_output_lower = agent_output.lower()
                is_generated = any(keyword in agent_output_lower for keyword in generation_keywords)
                
                if is_generated:
                    print(f"    ✗ Agent反馈：检测到生成内容的关键词，视为提取失败")
                    debug_logger.log(f"检测到生成内容的关键词，视为提取失败", "WARNING")
                    return None, False
                
                # 查找"摘要内容："后面的内容
                abstract_match = re.search(r'摘要内容[：:]\s*(.+?)(?:\n\n|\n提取结果|$)', agent_output, re.DOTALL | re.I)
                if abstract_match:
                    abstract_content = abstract_match.group(1).strip()
                else:
                    # 如果没有找到"摘要内容："标记，尝试提取"提取结果：1"之后的所有内容
                    after_result = re.split(r'提取结果[：:]\s*1\s*\n?', agent_output, flags=re.I)
                    if len(after_result) > 1:
                        abstract_content = after_result[1].strip()
                        # 移除可能的"摘要内容："标记
                        abstract_content = re.sub(r'^摘要内容[：:]\s*', '', abstract_content, flags=re.I)
                
                # 再次检查提取的内容是否包含生成标志
                if abstract_content:
                    abstract_lower = abstract_content.lower()
                    if any(keyword in abstract_lower for keyword in generation_keywords):
                        print(f"    ✗ Agent反馈：摘要内容中包含生成标志，视为提取失败")
                        debug_logger.log(f"摘要内容中包含生成标志，视为提取失败", "WARNING")
                        return None, False
                
                if abstract_content and len(abstract_content) > 50:
                    print(f"    ✓ AI agent成功提取摘要 ({len(abstract_content)} 字符)")
                    debug_logger.log(f"✓ 使用Agent提取的摘要（长度: {len(abstract_content)} 字符）", "SUCCESS")
                    return abstract_content, False
                else:
                    print(f"    ✗ Agent反馈：提取结果=1但摘要内容为空或太短")
                    debug_logger.log(f"Agent反馈提取结果=1但摘要内容为空或太短", "WARNING")
                    return None, False
            
            # 如果无法解析提取结果，视为提取失败
            print(f"    ✗ 无法解析Agent提取结果，视为提取失败")
            debug_logger.log(f"无法解析Agent提取结果，视为提取失败", "WARNING")
            return None, False  # 无法解析，视为提取失败，返回None
        except Exception as e:
            logging.error(f"网页摘要提取agent失败: {str(e)}")
            debug_logger.log(f"网页摘要提取agent失败: {str(e)}", "ERROR")
            # Agent失败，视为提取失败，返回None
            return None, False
    else:
        logging.warning(f"无法获取URL内容: {url}")
        debug_logger.log(f"无法获取URL内容: {url}", "ERROR")
        return None, False  # 无法获取URL内容，返回None


def create_relevance_analyzer_agent():
    """创建相关性分析 Agent"""
    return Agent(
        role="论文相关性分析专家",
        goal="基于论文标题和邮件片段信息，准确判断论文是否符合遥操作、力控、灵巧手、机器人动力学和机器学习等研究方向",
        backstory="你是一位在机器人学、控制理论、遥操作、机器人动力学、力控和机器学习领域拥有丰富研究经验的专家。你能够基于论文标题和邮件中的片段信息，准确判断论文的研究方向是否与目标领域相关。你严格基于提供的信息进行分析，不会虚构或推测论文内容，确保判断的准确性和可靠性。",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=2,
        max_execution_time=300
    )


def extract_task_input(task_description):
    """从任务描述中提取输入信息"""
    input_section = ""
    # 查找 "## 输入信息" 或 "**论文标题**" 等标记
    if "## 输入信息" in task_description:
        # 提取输入信息部分
        start_idx = task_description.find("## 输入信息")
        # 找到下一个 ## 开头的部分作为结束
        remaining = task_description[start_idx:]
        lines = remaining.split('\n')
        input_lines = []
        for line in lines:
            if line.startswith('## ') and line != '## 输入信息':
                break
            input_lines.append(line)
        input_section = '\n'.join(input_lines).replace('## 输入信息', '').strip()
    else:
        # 如果没有明确的输入信息部分，尝试提取关键信息
        # 查找论文标题
        if "**论文标题**" in task_description:
            title_match = re.search(r'\*\*论文标题\*\*[：:]\s*(.+?)(?:\n|$)', task_description)
            if title_match:
                input_section += f"**论文标题**：{title_match.group(1).strip()}\n\n"
        
        # 查找邮件片段信息
        if "**邮件片段信息**" in task_description:
            snippet_match = re.search(r'\*\*邮件片段信息\*\*[：:]?\s*```\s*(.+?)```', task_description, re.DOTALL)
            if snippet_match:
                snippet_text = snippet_match.group(1).strip()
                # 限制显示长度，避免过长
                if len(snippet_text) > 500:
                    snippet_text = snippet_text[:500] + "...\n[内容已截断]"
                input_section += f"**邮件片段信息**：\n```\n{snippet_text}\n```\n\n"
        
        # 查找PDF文本内容
        if "**PDF文本内容**" in task_description:
            pdf_match = re.search(r'\*\*PDF文本内容\*\*[^`]*```\s*(.+?)```', task_description, re.DOTALL)
            if pdf_match:
                pdf_text = pdf_match.group(1).strip()
                # 限制显示长度
                if len(pdf_text) > 1000:
                    pdf_text = pdf_text[:1000] + "...\n[内容已截断]"
                input_section += f"**PDF文本内容**：\n```\n{pdf_text}\n```\n\n"
        
        # 查找网页文本内容
        if "**网页文本内容**" in task_description:
            web_match = re.search(r'\*\*网页文本内容\*\*[^`]*```\s*(.+?)```', task_description, re.DOTALL)
            if web_match:
                web_text = web_match.group(1).strip()
                # 限制显示长度
                if len(web_text) > 1000:
                    web_text = web_text[:1000] + "...\n[内容已截断]"
                input_section += f"**网页文本内容**：\n```\n{web_text}\n```\n\n"
        
        # 查找提取的摘要内容
        if "**提取的摘要内容**" in task_description:
            abstract_match = re.search(r'\*\*提取的摘要内容\*\*[：:]?\s*```\s*(.+?)```', task_description, re.DOTALL)
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
                if len(abstract_text) > 1000:
                    abstract_text = abstract_text[:1000] + "...\n[内容已截断]"
                input_section += f"**提取的摘要内容**：\n```\n{abstract_text}\n```\n\n"
        
        # 查找原始摘要内容
        if "**原始摘要内容**" in task_description:
            abstract_match = re.search(r'\*\*原始摘要内容\*\*[：:]?\s*```\s*(.+?)```', task_description, re.DOTALL)
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
                if len(abstract_text) > 1000:
                    abstract_text = abstract_text[:1000] + "...\n[内容已截断]"
                input_section += f"**原始摘要内容**：\n```\n{abstract_text}\n```\n\n"
        
        # 查找论文摘要（英文原文）
        if "**论文摘要（英文原文）**" in task_description:
            abstract_match = re.search(r'\*\*论文摘要（英文原文）\*\*[：:]?\s*```\s*(.+?)```', task_description, re.DOTALL)
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
                if len(abstract_text) > 1000:
                    abstract_text = abstract_text[:1000] + "...\n[内容已截断]"
                input_section += f"**论文摘要（英文原文）**：\n```\n{abstract_text}\n```\n\n"
        
        # 查找论文内容（已翻译为中文）
        if "**论文内容（已翻译为中文）**" in task_description:
            content_match = re.search(r'\*\*论文内容（已翻译为中文）\*\*[：:]?\s*```\s*(.+?)```', task_description, re.DOTALL)
            if content_match:
                content_text = content_match.group(1).strip()
                if len(content_text) > 1000:
                    content_text = content_text[:1000] + "...\n[内容已截断]"
                input_section += f"**论文内容（已翻译为中文）**：\n```\n{content_text}\n```\n\n"
        
        # 查找来源类型
        if "**来源类型**" in task_description:
            source_match = re.search(r'\*\*来源类型\*\*[：:]\s*(.+?)(?:\n|$)', task_description)
            if source_match:
                input_section += f"**来源类型**：{source_match.group(1).strip()}\n\n"
    
    return input_section.strip()


def log_agent_io(agent_name, task, result, log_callback):
    """记录 Agent 的输入和输出（已禁用，不再发送任何日志）"""
    # 不再发送任何日志到前端
    return


def extract_paper_title_from_task(task_description):
    """
    从任务描述中提取论文标题
    
    Args:
        task_description: 任务描述字符串
        
    Returns:
        论文标题字符串，如果未找到则返回None
    """
    import re
    
    # 尝试匹配 "**论文标题**：" 或 "**论文标题（英文）**："
    patterns = [
        r'\*\*论文标题[（(]英文[）)]?\*\*[：:]\s*([^\n]+)',
        r'\*\*论文标题\*\*[：:]\s*([^\n]+)',
        r'论文标题[（(]英文[）)]?[：:]\s*([^\n]+)',
        r'论文标题[：:]\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, task_description)
        if match:
            title = match.group(1).strip()
            # 移除可能的markdown格式标记
            title = re.sub(r'^`+|`+$', '', title)
            if title:
                return title
    
    return None


def send_agent_status(agent_name, status, task=None, result=None):
    """
    发送agent工作状态信息
    
    Args:
        agent_name: agent名称
        status: 状态，"start"或"end"
        task: Task对象（用于提取目标信息）
        result: 任务执行结果（用于提取输出信息）
    """
    global agent_status_callback
    
    if not agent_status_callback:
        return
    
    target = None
    output = None
    
    if status == 'start' and task:
        # 从任务描述中提取论文标题
        task_description = task.description if hasattr(task, 'description') else str(task)
        target = extract_paper_title_from_task(task_description)
    
    if status == 'end' and result:
        # 提取输出信息的前200个字符作为摘要
        if hasattr(result, 'raw'):
            output = result.raw.strip()
        elif isinstance(result, str):
            output = result.strip()
        else:
            output = str(result).strip()
    
    # 调用回调函数
    agent_status_callback(agent_name, status, target, output)


def create_relevance_analysis_task(paper):
    """创建相关性分析任务"""
    paper_title = paper.get('title', '')
    paper_snippet = paper.get('snippet', '')
    
    return Task(
        description=(
            f"## 任务目标\n"
            f"基于论文标题和邮件片段信息，判断论文是否符合以下研究方向：\n"
            f"- 遥操作（Teleoperation）\n"
            f"- 力控（Force Control）\n"
            f"- 灵巧手（Dexterous Manipulation/Hand）\n"
            f"- 机器人动力学（Robot Dynamics）\n"
            f"- 机器学习（Machine Learning）\n\n"
            f"## 输入信息\n"
            f"**论文标题**：{paper_title}\n\n"
            f"**邮件片段信息**：\n```\n{paper_snippet}\n```\n\n"
            f"## 研究方向定义\n"
            f"### 方向1：遥操作（Teleoperation）\n"
            f"涉及远程操作、远程控制、主从控制、双边控制、远程操作机器人、人机遥操作等相关研究。\n"
            f"关键特征：远程控制、主从系统、操作者与远程机器人之间的控制与反馈。\n\n"
            f"### 方向2：力控（Force Control）\n"
            f"涉及力控制、力反馈、阻抗控制、导纳控制、混合力位控制、柔顺控制、接触力控制等相关研究。\n"
            f"关键特征：力控制策略、力感知、力反馈、力/力矩控制、接触力处理。\n\n"
            f"### 方向3：灵巧手（Dexterous Manipulation/Hand）\n"
            f"涉及灵巧手、灵巧操作、多指手、手指控制、抓取、手内操作、精细操作等相关研究。\n"
            f"关键特征：多指手、抓取策略、手内操作、精细操作、手指协调控制。\n\n"
            f"### 方向4：机器人动力学（Robot Dynamics）\n"
            f"涉及机器人动力学建模、动力学参数辨识、系统辨识、动态建模、动力学模型、参数识别等相关研究。\n"
            f"关键特征：动力学模型、参数辨识、系统辨识、动态特性分析、动力学建模。\n\n"
            f"### 方向5：机器学习（Machine Learning）\n"
            f"涉及强化学习、深度强化学习、模仿学习、演示学习、学习控制、自适应控制、神经网络控制、深度学习控制、模型学习、动力学学习、数据驱动控制、基于学习的控制、策略学习、技能学习、端到端学习、迁移学习、元学习等相关研究。\n"
            f"关键特征：机器学习方法应用于机器人控制、强化学习控制、学习策略、数据驱动方法、基于学习的控制算法。\n"
            f"**注意**：仅当机器学习方法应用于遥操作、力控、灵巧手、机器人动力学等机器人控制领域时，才视为符合方向。纯机器学习理论研究或应用于其他领域（如计算机视觉、自然语言处理等）的论文不符合此方向。\n\n"
            f"## 判断标准\n"
            f"### 输出1（符合方向）的条件\n"
            f"- 论文标题或邮件片段信息明确表明论文研究内容与上述任一方向相关\n"
            f"- 论文涉及的技术、方法、应用场景与上述方向的核心特征匹配\n"
            f"- 论文的研究目标、技术路线或应用领域与上述方向高度相关\n"
            f"- **重要**：判断必须基于提供的标题和片段信息，不能虚构或推测论文内容\n\n"
            f"### 输出0（不符合方向）的条件\n"
            f"- 论文标题和邮件片段信息无法明确判断与上述方向相关\n"
            f"- 论文研究内容明显属于其他领域（如纯计算机视觉、自动驾驶、导航等）\n"
            f"- 论文虽然涉及机器人或机器学习，但研究重点不在上述五个方向\n"
            f"- 信息不足，无法准确判断论文研究方向\n\n"
            f"## 分析要求\n"
            f"1. **严格基于提供信息**：只能基于论文标题和邮件片段信息进行分析，不能虚构、推测或添加未提供的信息\n"
            f"2. **准确理解研究方向**：准确理解五个研究方向的核心特征和关键要素\n"
            f"3. **客观判断**：基于客观证据进行判断，避免主观臆测\n"
            f"4. **明确输出**：明确输出1或0，并说明判断依据\n\n"
            f"## 输出格式（严格遵循）\n"
            f"```\n"
            f"相关性判断：1\n"
            f"判断依据：[简要说明论文与哪个方向相关，以及判断的关键依据]\n"
            f"```\n"
            f"或\n"
            f"```\n"
            f"相关性判断：0\n"
            f"判断依据：[简要说明为什么不符合上述方向]\n"
            f"```\n\n"
            f"## 注意事项\n"
            f"- 必须基于提供的标题和片段信息进行判断，不能虚构论文内容\n"
            f"- 如果信息不足，应输出0并说明信息不足\n"
            f"- 判断依据应具体、明确，指出与哪个方向相关以及关键特征\n"
            f"- 避免过度解读或推测，确保判断的准确性"
        ),
        agent=create_relevance_analyzer_agent(),
        expected_output="首先输出相关性判断（1表示符合方向，0表示不符合），然后输出判断依据"
    )


def create_abstract_validator_agent():
    """创建摘要验证 Agent"""
    return Agent(
        role="摘要真实性检查专家",
        goal="检查提取的摘要内容是否真实可靠，判断是否为AI模型虚构生成的内容",
        backstory="你是一位专业的学术内容验证专家。你擅长识别AI生成的内容和真实提取的内容。你能够通过分析文本特征、关键词、逻辑连贯性等来判断内容是否来自原文提取，还是AI模型虚构生成的。你严格遵循验证标准，确保只有真实可靠的摘要才能通过验证。",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=2,
        max_execution_time=300
    )


def create_abstract_cleaner_agent():
    """创建摘要清洗 Agent"""
    return Agent(
        role="摘要清洗专家",
        goal="清洗摘要内容，剔除无关符号、引用标记、图表引用和无意义的格式字符，保持摘要的完整性和可读性",
        backstory="你是一位专业的文本清洗专家。你擅长识别和清理学术文本中的无关内容，包括引用文献标记（如[1]、[2-5]）、图表引用（如图1、Table 2）、无意义的换行符、多余的空格等。你能够在不改变原文意思的前提下，清理这些干扰内容，使文本更加简洁易读。",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=2,
        max_execution_time=300
    )


def create_translator_agent():
    """创建专业翻译 Agent"""
    return Agent(
        role="专业翻译专家",
        goal="将英文论文内容准确、专业地翻译成中文，确保专业术语的准确性和技术表达的清晰性",
        backstory="你是一位在机器人学、控制理论、遥操作、机器人动力学和力控领域拥有深厚专业背景的翻译专家。你擅长将英文学术论文翻译成中文，能够准确处理专业术语，保持技术描述的完整性和逻辑结构的清晰性。",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=3,
        max_execution_time=300
    )


def create_reviewer_agent():
    """创建专业评审 Agent"""
    return Agent(
        role="专业评审专家",
        goal="对论文进行专业评审，生成结构化总结并给出简洁的4分制评分（只输出一次）",
        backstory="你是一位在机器人学、控制理论、遥操作、机器人动力学和力控领域拥有丰富研究经验的评审专家。你能够从创新性、技术深度、相关性、实用性等多个维度对论文进行客观、专业的评价。你总是简洁明了地输出结果，不会重复说明。",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=2,  # 减少迭代次数，避免重复
        max_execution_time=300
    )


def create_web_abstract_extractor_agent():
    """创建网页摘要提取 Agent"""
    return Agent(
        role="网页摘要提取专家",
        goal="从学术论文网页的HTML内容中准确提取摘要部分，排除导航、广告、正文等其他内容。对于摘要与引言融合的情况，可以将引言开篇部分一起提取。严禁生成、创造或修改摘要，只能提取现有内容。",
        backstory="你是一位专业的学术论文网页处理专家。你擅长从复杂的HTML网页中准确识别和提取摘要部分。你能够识别各种网页结构中的摘要内容，包括不同网站的设计风格，准确提取摘要并排除导航栏、广告、作者信息、正文等其他无关内容。你特别了解某些学术会议论文的格式特点：摘要内容可能会融入引言的开篇部分，如果识别到摘要直接连接到引言且引言开篇在逻辑上延续了摘要内容，你会将摘要和引言开篇部分（通常1-2段）一起提取。你严格遵守一个原则：只能提取网页中现有的摘要内容，绝对不能自己生成、创造、改写或总结摘要。如果网页中没有明确的摘要，你会如实报告提取失败。你只输出摘要内容，不包含任何其他信息。",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=2,
        max_execution_time=300
    )


def create_pdf_processor_agent():
    """创建PDF处理 Agent"""
    return Agent(
        role="PDF摘要提取专家",
        goal="从PDF文档的文本内容中准确提取摘要部分，排除正文内容。对于摘要与引言融合的情况，可以将引言开篇部分一起提取。严禁生成、创造或修改摘要，只能提取现有内容。",
        backstory="你是一位专业的学术论文PDF处理专家。你擅长从PDF提取的文本中准确识别和提取摘要部分。你能够识别'Abstract'、'摘要'等标记，准确提取摘要内容，并排除Introduction、正文等其他部分。你特别了解某些学术会议论文的格式特点：摘要内容可能会融入引言的开篇部分，如果识别到摘要直接连接到引言且引言开篇在逻辑上延续了摘要内容，你会将摘要和引言开篇部分（通常1-2段）一起提取。你严格遵守一个原则：只能提取PDF文本中现有的摘要内容，绝对不能自己生成、创造、改写或总结摘要。如果PDF文本中没有明确的摘要，你会如实报告提取失败。你只输出摘要内容，不包含任何正文。",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=2,
        max_execution_time=300
    )


def create_pdf_abstract_extraction_task(pdf_text, paper_title):
    """创建PDF摘要提取任务"""
    # 限制文本长度，避免超出token限制（通常前8000字符足够包含摘要）
    text_preview = pdf_text[:8000] if len(pdf_text) > 8000 else pdf_text
    
    return Task(
        description=(
            f"## 任务目标\n"
            f"从PDF文档的文本内容中准确提取摘要部分，仅提取现有内容，严禁生成或修改。\n\n"
            f"## 输入信息\n"
            f"**论文标题**：{paper_title}\n\n"
            f"**PDF文本内容**（前8000字符，通常包含摘要）：\n```\n{text_preview}\n```\n\n"
            f"## 核心约束（必须严格遵守）\n"
            f"1. **严格禁止生成内容**：只能提取文本中已存在的摘要，禁止生成、创造、改写、总结或补充任何内容。\n"
            f"2. **严格禁止修改内容**：提取的摘要必须与原文完全一致，禁止任何修改、补充、改写或重新表述。\n"
            f"3. **严格禁止推测**：如果文本中没有明确的摘要部分，必须输出0，禁止根据标题或其他信息推测。\n"
            f"4. **逐字提取原则**：提取的内容必须与原文措辞和表达方式完全一致。\n"
            f"5. **内容不完整处理**：如果PDF文本内容不完整、无法识别摘要部分或文本提取失败，必须输出0。\n"
            f"6. **禁止添加元信息**：输出中禁止包含任何Note、说明、注释或解释性文字。\n\n"
            f"## 执行步骤\n"
            f"### 步骤1：判断提取可行性\n"
            f"- **输出1的条件**：文本中明确存在摘要内容（通常以'Abstract'、'摘要'等标记开始，或论文开头的描述性段落）。\n"
            f"- **输出0的条件**：文本中没有明确的摘要部分、内容不完整、文本提取失败或无法识别摘要。\n"
            f"- **判断标准**：仅在文本中明确存在摘要内容时输出1，禁止根据标题或其他信息推测。\n\n"
            f"### 步骤2：提取摘要内容（仅在步骤1输出1时执行）\n"
            f"- **识别范围**：摘要通常在Introduction、Keywords、Key Words或正文开始之前结束。\n"
            f"- **特殊处理：摘要与引言融合**：\n"
            f"  - 某些学术会议论文中，摘要可能融入引言的开篇部分。\n"
            f"  - 如果摘要直接连接到引言（无明显分隔标记，或引言开篇延续摘要内容），可将摘要和引言开篇部分（通常1-2段）一起提取。\n"
            f"  - **判断标准**：引言开篇在逻辑上延续摘要内容且无明显章节分隔时，可视为摘要的一部分；\n"
            f"    当引言开始讨论具体研究背景、相关工作等详细内容时，应停止提取。\n"
            f"- **提取要求**：\n"
            f"  - 完全按照原文逐字提取，禁止任何修改。\n"
            f"  - 仅输出摘要内容（如与引言融合，则包含引言开篇部分）。\n"
            f"  - 禁止包含：论文标题、作者信息、Keywords、正文详细内容、参考文献。\n"
            f"  - 保持摘要的完整性和连贯性（必须是原文内容）。\n\n"
            f"### 步骤3：输出结果（仅在步骤1输出0时执行）\n"
            f"- 输出0，不输出任何摘要内容，不添加任何说明。\n\n"
            f"## 输出格式（严格遵循）\n"
            f"```\n"
            f"提取结果：1\n"
            f"摘要内容：[提取的摘要文本]\n"
            f"```\n"
            f"或\n"
            f"```\n"
            f"提取结果：0\n"
            f"摘要内容：\n"
            f"```\n\n"
            f"## 注意事项\n"
            f"- 严格按照上述格式输出，先输出提取结果（1或0），然后根据结果决定是否输出摘要内容。\n"
            f"- 只能提取现有内容，严禁生成、创造或修改摘要。\n"
            f"- 如果无法提取，必须输出0，禁止添加任何说明或注释。"
        ),
        agent=create_pdf_processor_agent(),
        expected_output="首先输出提取结果（1表示成功，0表示失败），如果成功则输出从原文中逐字提取的摘要内容（严禁生成或修改，禁止添加任何说明或注释）"
    )


def create_web_abstract_extraction_task(html_text, paper_title, url=""):
    """创建网页摘要提取任务"""
    # 限制文本长度，避免超出token限制
    # 先尝试提取主要文本内容（去除HTML标签后的纯文本）
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        # 移除script和style标签
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        # 获取纯文本
        text_content = soup.get_text(separator=' ', strip=True)
        # 限制长度
        text_preview = text_content[:10000] if len(text_content) > 10000 else text_content
    except:
        # 如果解析失败，直接使用原始HTML的前10000字符
        text_preview = html_text[:10000] if len(html_text) > 10000 else html_text
    
    return Task(
        description=(
            f"## 任务目标\n"
            f"从学术论文网页内容中准确提取摘要部分，仅提取现有内容，严禁生成或修改。\n\n"
            f"## 输入信息\n"
            f"**论文标题**：{paper_title}\n\n"
            f"**网页URL**：{url}\n\n"
            f"**网页文本内容**（前10000字符）：\n```\n{text_preview}\n```\n\n"
            f"## 核心约束（必须严格遵守）\n"
            f"1. **严格禁止生成内容**：只能提取网页文本中已存在的摘要，禁止生成、创造、改写、总结或补充任何内容。\n"
            f"2. **严格禁止修改内容**：提取的摘要必须与原文完全一致，禁止任何修改、补充、改写或重新表述。\n"
            f"3. **严格禁止推测**：如果网页中没有明确的摘要部分，必须输出0，禁止根据标题或其他信息推测。\n"
            f"4. **逐字提取原则**：提取的内容必须与原文措辞和表达方式完全一致。\n"
            f"5. **内容不完整处理**：如果网页文本内容不完整、无法识别摘要部分或HTML解析失败，必须输出0。\n"
            f"6. **禁止添加元信息**：输出中禁止包含任何Note、说明、注释或解释性文字。\n\n"
            f"## 执行步骤\n"
            f"### 步骤1：判断提取可行性\n"
            f"- **输出1的条件**：网页中明确存在摘要内容（通常包含'Abstract'、'摘要'、'Summary'等标记，或论文描述部分）。\n"
            f"- **输出0的条件**：网页中没有明确的摘要部分、内容不完整、HTML解析失败或无法识别摘要。\n"
            f"- **判断标准**：仅在网页中明确存在摘要内容时输出1，禁止根据标题或其他信息推测。\n\n"
            f"### 步骤2：提取摘要内容（仅在步骤1输出1时执行）\n"
            f"- **识别范围**：摘要通常在Introduction、Keywords、正文开始之前结束。\n"
            f"- **特殊处理：摘要与引言融合**：\n"
            f"  - 某些学术会议论文中，摘要可能融入引言的开篇部分。\n"
            f"  - 如果摘要直接连接到引言（无明显分隔标记，或引言开篇延续摘要内容），可将摘要和引言开篇部分（通常1-2段）一起提取。\n"
            f"  - **判断标准**：引言开篇在逻辑上延续摘要内容且无明显章节分隔时，可视为摘要的一部分；\n"
            f"    当引言开始讨论具体研究背景、相关工作等详细内容时，应停止提取。\n"
            f"- **提取要求**：\n"
            f"  - 完全按照原文逐字提取，禁止任何修改。\n"
            f"  - 仅输出摘要内容（如与引言融合，则包含引言开篇部分）。\n"
            f"  - 禁止包含：导航栏、广告、标题、作者、关键词、正文详细内容、参考文献。\n"
            f"  - 去除HTML标签、特殊字符和格式标记，仅保留纯文本摘要内容（文本内容必须与原文一致）。\n"
            f"  - 保持摘要的完整性和连贯性（必须是原文内容）。\n\n"
            f"### 步骤3：输出结果（仅在步骤1输出0时执行）\n"
            f"- 输出0，不输出任何摘要内容，不添加任何说明。\n\n"
            f"## 输出格式（严格遵循）\n"
            f"```\n"
            f"提取结果：1\n"
            f"摘要内容：[提取的摘要文本]\n"
            f"```\n"
            f"或\n"
            f"```\n"
            f"提取结果：0\n"
            f"摘要内容：\n"
            f"```\n\n"
            f"## 注意事项\n"
            f"- 严格按照上述格式输出，先输出提取结果（1或0），然后根据结果决定是否输出摘要内容。\n"
            f"- 只能提取现有内容，严禁生成、创造或修改摘要。\n"
            f"- 如果无法提取，必须输出0，禁止添加任何说明或注释。"
        ),
        agent=create_web_abstract_extractor_agent(),
        expected_output="首先输出提取结果（1表示成功，0表示失败），如果成功则输出从原文中逐字提取的摘要内容（严禁生成或修改，禁止添加任何说明或注释）"
    )


def create_abstract_validation_task(paper, abstract_text, source_type="网页"):
    """创建摘要验证任务"""
    return Task(
        description=(
            f"## 任务目标\n"
            f"检查提取的摘要内容是否真实可靠，判断是否为AI模型虚构生成的内容。\n\n"
            f"## 输入信息\n"
            f"**论文标题**：{paper['title']}\n\n"
            f"**提取的摘要内容**：\n```\n{abstract_text}\n```\n\n"
            f"**来源类型**：{source_type}\n\n"
            f"## 验证维度\n"
            f"### 维度1：生成标志检测\n"
            f"检查摘要中是否包含明显的AI生成标志：\n"
            f"- 关键词标志：'simulated'、'generated'、'based on'、'not provided'、'assumed'、'presumed'、'inferred'等\n"
            f"- 元信息标志：'Note:'、'注意'、'说明'、'备注'等解释性文字\n"
            f"- 不确定性表述：'推测'、'假设'、'推断'、'可能'等不确定性的表述\n\n"
            f"### 维度2：内容相关性验证\n"
            f"检查摘要内容与论文标题的相关性：\n"
            f"- 摘要是否明确提到标题中的关键概念和主题\n"
            f"- 摘要内容是否与标题主题高度一致\n"
            f"- 是否存在明显的主题偏离或无关内容\n\n"
            f"### 维度3：文本特征分析\n"
            f"检查摘要是否具有真实学术摘要的特征：\n"
            f"- 是否包含具体的技术细节、方法、结果、贡献等实质性内容\n"
            f"- 是否过于笼统、模糊或缺乏具体信息\n"
            f"- 文本特征是否表明来自原文逐字提取（而非重新表述）\n\n"
            f"### 维度4：逻辑连贯性评估\n"
            f"检查摘要的逻辑连贯性：\n"
            f"- 句子之间是否存在清晰的逻辑关系\n"
            f"- 是否包含完整的学术摘要结构（研究背景、方法、结果、结论等）\n"
            f"- 段落结构是否合理，信息流是否顺畅\n\n"
            f"## 判断标准\n"
            f"- **输出1（真实可靠）**：摘要内容来自原文提取，通过所有验证维度，无明显AI生成标志。\n"
            f"- **输出0（可能是虚构）**：摘要内容可能是AI虚构生成，或包含明显的生成标志，或未通过验证维度。\n\n"
            f"## 输出格式（严格遵循）\n"
            f"```\n"
            f"验证结果：1\n"
            f"验证说明：[简要说明验证依据，包括各维度的检查结果]\n"
            f"```\n"
            f"或\n"
            f"```\n"
            f"验证结果：0\n"
            f"验证说明：[详细说明检测到的虚构标志和未通过的验证维度]\n"
            f"```\n\n"
            f"## 注意事项\n"
            f"- 必须基于客观证据进行判断，避免主观臆测。\n"
            f"- 验证说明应具体、明确，指出具体的检测依据。\n"
            f"- 如果存在任何可疑标志，应输出0并详细说明。"
        ),
        agent=create_abstract_validator_agent(),
        expected_output="首先输出验证结果（1表示真实可靠，0表示可能是虚构的），然后输出详细的验证说明"
    )


def create_abstract_cleaning_task(abstract_text):
    """创建摘要清洗任务"""
    return Task(
        description=(
            f"## 任务目标\n"
            f"清洗摘要内容，剔除无关符号、引用标记和格式字符，保持文本的完整性和可读性。\n\n"
            f"## 输入信息\n"
            f"**原始摘要内容**：\n```\n{abstract_text}\n```\n\n"
            f"## 清洗规则\n"
            f"### 规则1：删除引用文献标记\n"
            f"- 删除所有引用标记，包括但不限于：\n"
            f"  - 单引用：[1]、[2]、[3]等\n"
            f"  - 范围引用：[1-5]、[2-10]等\n"
            f"  - 多引用：[1,2,3]、[1, 2, 3]等\n"
            f"  - 混合引用：[1,3-5,7]等\n\n"
            f"### 规则2：删除图表引用\n"
            f"- 删除所有图表引用，包括但不限于：\n"
            f"  - 中文格式：'图1'、'图2'、'表1'、'表2'等\n"
            f"  - 英文格式：'Figure 1'、'Fig. 1'、'Table 2'、'Tab. 2'等\n"
            f"  - 带括号格式：'(图1)'、'(Figure 1)'等\n\n"
            f"### 规则3：清理无意义的换行\n"
            f"- 将多个连续的空行（≥2个）合并为单个空行\n"
            f"- 删除段落中间不必要的换行（保持段落内句子连续）\n"
            f"- 保留段落之间的合理分隔（单个空行）\n\n"
            f"### 规则4：清理多余空格\n"
            f"- 删除行首和行尾的所有空格\n"
            f"- 将多个连续空格（≥2个）合并为单个空格\n"
            f"- 保留句子之间的单个空格\n\n"
            f"### 规则5：保持内容完整性\n"
            f"- 仅删除标记和格式字符，禁止删除或修改实际的文本内容\n"
            f"- 禁止修改单词、句子或段落的实际内容\n"
            f"- 禁止添加、删除或重新组织文本内容\n\n"
            f"### 规则6：保持逻辑连贯性\n"
            f"- 确保清洗后的文本逻辑连贯，句子完整\n"
            f"- 保持原文的段落结构和句子顺序\n"
            f"- 确保删除引用标记后，句子仍然语法正确\n\n"
            f"## 输出要求\n"
            f"- 直接输出清洗后的摘要内容，不添加任何说明或注释\n"
            f"- 保持原文的段落结构和句子顺序\n"
            f"- 输出应为纯文本，无额外的格式标记\n\n"
            f"## 示例\n"
            f"**输入**：This paper presents a novel method [1,2]. As shown in Figure 1, the results are promising.\n\n"
            f"**输出**：This paper presents a novel method. As shown in, the results are promising.\n\n"
            f"（注意：删除引用标记和图表引用，但保持句子结构）"
        ),
        agent=create_abstract_cleaner_agent(),
        expected_output="清洗后的摘要内容，已删除引用标记、图表引用和无意义的格式字符，保持文本完整性和逻辑连贯性"
    )


def create_translation_task(paper, abstract_text):
    """创建翻译任务"""
    return Task(
        description=(
            f"## 任务目标\n"
            f"将英文论文摘要准确、专业地翻译成中文，确保专业术语准确性和技术描述完整性。\n\n"
            f"## 输入信息\n"
            f"**论文标题**：{paper['title']}\n\n"
            f"**论文摘要（英文原文）**：\n```\n{abstract_text}\n```\n\n"
            f"## 翻译原则\n"
            f"### 原则1：专业术语准确性\n"
            f"- 使用机器人学、控制理论、遥操作、机器人动力学和力控领域的标准中文术语\n"
            f"- 对于专有名词和术语，优先使用该领域公认的中文译名\n"
            f"- 保持术语翻译的一致性（同一术语在全文中的翻译应一致）\n\n"
            f"### 原则2：技术描述完整性\n"
            f"- 确保技术描述的准确性和完整性，不遗漏关键信息\n"
            f"- 准确传达技术方法、实验结果、创新点等核心内容\n"
            f"- 保持技术概念的精确性，避免模糊或歧义表达\n\n"
            f"### 原则3：逻辑结构保持\n"
            f"- 保持原文的逻辑结构和段落组织\n"
            f"- 保持句子之间的逻辑关系和连接\n"
            f"- 保持原文的表达风格和语气\n\n"
            f"### 原则4：不确定术语处理\n"
            f"- 如果遇到不确定的术语，提供最可能的专业翻译\n"
            f"- 优先考虑该领域的常用译法和标准术语\n"
            f"- 避免直译或字面翻译，注重专业表达的准确性\n\n"
            f"## 输出要求\n"
            f"- 直接输出翻译后的中文内容，不添加任何说明或注释\n"
            f"- 保持原文的结构和逻辑\n"
            f"- 确保翻译流畅、自然，符合中文表达习惯"
        ),
        agent=create_translator_agent(),
        expected_output="翻译后的中文摘要内容，保持原文的结构和逻辑，专业术语准确，技术描述完整"
    )


def create_review_task(paper, translated_content):
    """创建评审任务"""
    return Task(
        description=(
            f"## 任务目标\n"
            f"对论文进行专业评审，生成结构化总结并给出4个维度的分项评分（每个维度0.0-1.0分）。\n\n"
            f"## 输入信息\n"
            f"**论文标题（英文）**：{paper['title']}\n\n"
            f"**论文内容（已翻译为中文）**：\n```\n{translated_content}\n```\n\n"
            f"## 评审维度\n"
            f"### 维度1：创新性（0.0-1.0分）\n"
            f"评估论文的创新程度：\n"
            f"- 是否提出了新的理论、方法或技术\n"
            f"- 是否解决了现有方法无法解决的问题\n"
            f"- 创新点的显著性和重要性\n\n"
            f"### 维度2：技术深度（0.0-1.0分）\n"
            f"评估论文的技术深度：\n"
            f"- 技术方法的复杂度和先进性\n"
            f"- 理论分析的深度和严谨性\n"
            f"- 实验验证的充分性和可靠性\n\n"
            f"### 维度3：相关性（0.0-1.0分）\n"
            f"评估论文与目标领域的相关性：\n"
            f"- 与遥操作、机器人动力学、力控、机器人控制等领域的关联程度\n"
            f"- 对目标领域研究的贡献和影响\n"
            f"- 研究主题的契合度\n\n"
            f"### 维度4：实用性（0.0-1.0分）\n"
            f"评估论文的实用价值：\n"
            f"- 方法的可应用性和可实施性\n"
            f"- 潜在的应用场景和价值\n"
            f"- 对实际问题的解决能力\n\n"
            f"## 输出格式（严格遵循）\n"
            f"```\n"
            f"**核心贡献**：[1-2句话说明主要创新点和贡献]\n\n"
            f"**技术方法**：[简述主要技术路线和方法]\n\n"
            f"**相关性分析**：[详细说明与遥操作/机器人动力学/力控/机器人控制的关系]\n\n"
            f"**技术价值**：[评估该论文的技术价值和潜在应用]\n\n"
            f"**值得关注的原因**：[为什么这篇论文重要，有哪些亮点]\n\n"
            f"**评分详情**：\n"
            f"```json\n"
            f'{{"创新性": 0.0-1.0, "技术深度": 0.0-1.0, "相关性": 0.0-1.0, "实用性": 0.0-1.0, "评分理由": "简要说明评分依据"}}\n'
            f"```\n"
            f"```\n\n"
            f"## 重要约束\n"
            f"1. **评分详情格式**：必须使用Markdown代码块格式（```json ... ```），仅输出一次\n"
            f"2. **评分范围**：每个维度评分范围为0.0-1.0分，使用浮点数（如0.85）\n"
            f"3. **禁止输出总分**：只输出4个维度的分项得分，不要计算或输出总分\n"
            f"4. **评分理由**：必须在JSON中包含'评分理由'字段，简要说明评分依据\n"
            f"5. **禁止重复**：不要重复说明评分规则，不要多次输出评分详情\n"
            f"6. **直接输出**：直接给出评审结果，不要添加额外的说明或解释\n\n"
            f"## 注意事项\n"
            f"- 评分应客观、公正，基于论文的实际内容进行评估\n"
            f"- 各维度的评分应相互独立，避免相互影响\n"
            f"- 评分理由应简洁明了，说明关键评分依据"
        ),
        agent=create_reviewer_agent(),
        expected_output=(
            "评审报告包含：核心贡献、技术方法、相关性分析、技术价值、值得关注的原因，"
            "以及一个Markdown代码块格式的JSON评分详情（包含4个维度的分项得分和评分理由，不包含总分，仅输出一次）。"
        )
    )


def process_paper_with_crewai(paper, full_abstract, source_type="网页"):
    """
    使用 CrewAI 框架处理论文：验证 + 翻译 + 评审
    返回处理结果字典
    
    Args:
        paper: 论文信息字典
        full_abstract: 完整摘要文本
        source_type: 摘要来源类型（"PDF"或"网页"）
    """
    try:
        # 步骤0: 验证摘要真实性
        print("  [步骤0/4] 验证摘要真实性...")
        validation_task = create_abstract_validation_task(paper, full_abstract, source_type)
        validation_agent = create_abstract_validator_agent()
        
        # 发送agent工作开始状态
        send_agent_status("摘要真实性检查专家", "start", task=validation_task)
        
        with capture_crewai_output():
            validation_crew = Crew(
                agents=[validation_agent],
                tasks=[validation_task],
                verbose=True,
                share_crew=False
            )
            validation_result = validation_crew.kickoff()
        
        # 发送agent工作结束状态
        send_agent_status("摘要真实性检查专家", "end", result=validation_result)
        
        # 记录 Agent 的输入和输出
        if crewai_log_callback:
            log_agent_io("摘要真实性检查专家", validation_task, validation_result, crewai_log_callback)
        
        validation_output = validation_result.raw.strip()
        
        # 解析验证结果
        validation_result_code = None
        validation_explanation = ""
        
        # 查找"验证结果："后面的数字
        result_match = re.search(r'验证结果[：:]\s*([01])', validation_output, re.I)
        if result_match:
            validation_result_code = int(result_match.group(1))
        
        # 如果没找到，尝试查找独立的0或1
        if validation_result_code is None:
            first_line_match = re.match(r'^\s*([01])\s*$', validation_output.split('\n')[0] if validation_output else '')
            if first_line_match:
                validation_result_code = int(first_line_match.group(1))
        
        # 如果还是没找到，检查是否包含"验证结果：0"或"验证结果：1"
        if validation_result_code is None:
            if re.search(r'验证结果[：:]\s*0|结果[：:]\s*0|^0\s*$', validation_output, re.I | re.M):
                validation_result_code = 0
            elif re.search(r'验证结果[：:]\s*1|结果[：:]\s*1|^1\s*$', validation_output, re.I | re.M):
                validation_result_code = 1
        
        # 提取验证说明
        explanation_match = re.search(r'验证说明[：:]\s*(.+?)(?:\n\n|$)', validation_output, re.DOTALL | re.I)
        if explanation_match:
            validation_explanation = explanation_match.group(1).strip()
        
        # 如果验证结果为0（可能是虚构的），返回失败
        if validation_result_code == 0:
            print(f"    ✗ 摘要验证失败：检测到可能是AI虚构生成的内容")
            debug_logger.log(f"摘要验证失败：{validation_explanation}", "WARNING")
            return {
                'translated_content': "摘要验证失败：检测到可能是AI虚构生成的内容",
                'review': "摘要验证失败：检测到可能是AI虚构生成的内容",
                'score': 0.0,
                'score_details': {},
                'is_high_value': False,
                'full_abstract': full_abstract
            }
        
        # 如果验证结果为1（真实可靠），继续处理
        if validation_result_code == 1:
            print(f"    ✓ 摘要验证通过：内容真实可靠")
            debug_logger.log(f"摘要验证通过：{validation_explanation}", "SUCCESS")
        else:
            # 如果无法解析验证结果，使用关键词检测作为备选
            print(f"    ⚠ 无法解析验证结果，使用关键词检测...")
            generation_keywords = [
                'simulated', 'generated', 'based on the', 'not provided', 'not available',
                'cannot extract', 'unable to', 'assumed', 'presumed', 'inferred',
                '推测', '生成', '模拟', '假设', '推断', '未提供', '无法提取',
                'Note:', 'note:', '注意', '说明', '备注'
            ]
            abstract_lower = full_abstract.lower()
            is_generated = any(keyword in abstract_lower for keyword in generation_keywords)
            
            if is_generated:
                print(f"    ✗ 关键词检测失败：检测到生成标志")
                debug_logger.log(f"关键词检测失败：检测到生成标志", "WARNING")
                return {
                    'translated_content': "摘要验证失败：关键词检测发现生成标志",
                    'review': "摘要验证失败：关键词检测发现生成标志",
                    'score': 0.0,
                    'score_details': {},
                    'is_high_value': False,
                    'full_abstract': full_abstract
                }
            else:
                print(f"    ✓ 关键词检测通过：未发现生成标志")
                debug_logger.log(f"关键词检测通过：未发现生成标志", "INFO")
        
        # 步骤1: 清洗摘要
        print("  [步骤1/4] 清洗摘要中...")
        cleaning_task = create_abstract_cleaning_task(full_abstract)
        cleaning_agent = create_abstract_cleaner_agent()
        
        # 发送agent工作开始状态（清洗任务没有论文标题，直接从paper字典获取）
        if agent_status_callback:
            agent_status_callback("摘要清洗专家", "start", paper.get('title', ''), None)
        
        with capture_crewai_output():
            cleaning_crew = Crew(
                agents=[cleaning_agent],
                tasks=[cleaning_task],
                verbose=True,
                share_crew=False
            )
            cleaning_result = cleaning_crew.kickoff()
        
        # 发送agent工作结束状态
        send_agent_status("摘要清洗专家", "end", result=cleaning_result)
        
        # 记录 Agent 的输入和输出
        if crewai_log_callback:
            log_agent_io("摘要清洗专家", cleaning_task, cleaning_result, crewai_log_callback)
        
        cleaned_abstract = cleaning_result.raw.strip()
        
        # 如果清洗失败，使用原始摘要
        if not cleaned_abstract or len(cleaned_abstract) < 50:
            print(f"    ⚠ 摘要清洗结果异常，使用原始摘要")
            debug_logger.log(f"摘要清洗结果异常，使用原始摘要", "WARNING")
            cleaned_abstract = full_abstract
        else:
            print(f"    ✓ 摘要清洗完成 ({len(cleaned_abstract)} 字符)")
            debug_logger.log(f"摘要清洗完成，长度: {len(cleaned_abstract)} 字符", "SUCCESS")
        
        # 步骤2: 翻译
        print("  [步骤2/4] 专业翻译中...")
        translation_task = create_translation_task(paper, cleaned_abstract)
        translation_agent = create_translator_agent()
        
        # 发送agent工作开始状态
        send_agent_status("专业翻译专家", "start", task=translation_task)
        
        with capture_crewai_output():
            translation_crew = Crew(
                agents=[translation_agent],
                tasks=[translation_task],
                verbose=True,
                share_crew=False
            )
            translation_result = translation_crew.kickoff()
        
        # 发送agent工作结束状态
        send_agent_status("专业翻译专家", "end", result=translation_result)
        
        # 记录 Agent 的输入和输出
        if crewai_log_callback:
            log_agent_io("专业翻译专家", translation_task, translation_result, crewai_log_callback)
        
        translated_content = translation_result.raw.strip()
        
        # 步骤3: 评审
        print("  [步骤3/4] 专业评审和评分中...")
        review_task = create_review_task(paper, translated_content)
        review_agent = create_reviewer_agent()
        
        # 发送agent工作开始状态
        send_agent_status("专业评审专家", "start", task=review_task)
        
        with capture_crewai_output():
            review_crew = Crew(
                agents=[review_agent],
                tasks=[review_task],
                verbose=True,
                share_crew=False
            )
            review_result = review_crew.kickoff()
        
        # 发送agent工作结束状态
        send_agent_status("专业评审专家", "end", result=review_result)
        
        # 记录 Agent 的输入和输出
        if crewai_log_callback:
            log_agent_io("专业评审专家", review_task, review_result, crewai_log_callback)
        
        review_text = review_result.raw.strip()
        
        # 提取评分
        score_data = extract_score_from_review(review_text)
        
        return {
            'translated_content': translated_content,
            'review': review_text,
            'score': score_data.get('总分', 0.0),
            'score_details': score_data,
            'is_high_value': score_data.get('总分', 0.0) > 3.0,
            'full_abstract': full_abstract
        }
    except Exception as e:
        logging.error(f"处理论文时出错: {str(e)}")
        return {
            'translated_content': f"翻译失败: {str(e)}",
            'review': f"评审失败: {str(e)}",
            'score': 0.0,
            'score_details': {},
            'is_high_value': False,
            'full_abstract': full_abstract
        }


def extract_score_from_review(review_text):
    """
    从评审文本中提取评分信息
    只提取分项得分，总分通过计算得出（避免agent幻觉）
    """
    import json
    import re
    
    # 定义评分维度（4个维度，总分4分）
    dimensions = ['创新性', '技术深度', '相关性', '实用性']
    
    score_data = {
        '创新性': 0.0,
        '技术深度': 0.0,
        '相关性': 0.0,
        '实用性': 0.0,
        '总分': 0.0,  # 将通过计算得出（4个维度之和，满分4分）
        '评分理由': ''
    }
    
    # 方法1: 尝试提取完整的JSON对象（支持多行和嵌套）
    # 查找JSON对象的开始和结束
    json_start = review_text.find('{')
    json_end = review_text.rfind('}')
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            json_str = review_text[json_start:json_end+1]
            # 尝试解析JSON
            parsed = json.loads(json_str)
            # 只提取分项得分和评分理由，忽略总分（即使agent输出了也忽略）
            for dim in dimensions:
                if dim in parsed and isinstance(parsed[dim], (int, float)):
                    score = float(parsed[dim])
                    # 验证分数范围（0.0-1.0）
                    if 0.0 <= score <= 1.0:
                        score_data[dim] = score
            # 提取评分理由
            if '评分理由' in parsed and isinstance(parsed['评分理由'], str):
                score_data['评分理由'] = parsed['评分理由']
            
            # 计算总分（各维度之和）
            total = sum([score_data[dim] for dim in dimensions])
            score_data['总分'] = total
            return score_data
        except (json.JSONDecodeError, ValueError):
            pass
    
    # 方法2: 尝试提取JSON格式的评分（更宽松的匹配）
    json_patterns = [
        r'\{[^{}]*"创新性"[^{}]*"技术深度"[^{}]*\}',
        r'\{[^{}]*"创新性"[^{}]*\}',
    ]
    for pattern in json_patterns:
        json_match = re.search(pattern, review_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                # 只提取分项得分，忽略总分
                for dim in dimensions:
                    if dim in parsed and isinstance(parsed[dim], (int, float)):
                        score = float(parsed[dim])
                        if 0.0 <= score <= 1.0:
                            score_data[dim] = score
                if '评分理由' in parsed and isinstance(parsed['评分理由'], str):
                    score_data['评分理由'] = parsed['评分理由']
                
                # 计算总分
                total = sum([score_data[dim] for dim in dimensions])
                if total > 0:
                    score_data['总分'] = total
                    return score_data
            except (json.JSONDecodeError, ValueError):
                continue
    
    # 方法3: 如果JSON提取失败，尝试从文本中提取各个维度的分数
    for dim in dimensions:
        patterns = [
            rf'{dim}[：:]\s*([0-9.]+)',
            rf'"{dim}"[：:]\s*([0-9.]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, review_text)
            if match:
                try:
                    score = float(match.group(1))
                    # 验证分数范围（0.0-1.0）
                    if 0.0 <= score <= 1.0:
                        score_data[dim] = score
                        break
                except (ValueError, IndexError):
                    continue
    
    # 计算总分（各维度之和）
    total = sum([score_data[dim] for dim in dimensions])
    score_data['总分'] = total
    
    return score_data


def generate_daily_report(relevant_papers):
    """生成原始日报（Markdown 格式，与理想格式一致）"""
    import json
    import re
    report = []
    
    # 按评分分类论文
    high_value_papers = [p for p in relevant_papers if p.get('is_high_value', False)]
    other_papers = [p for p in relevant_papers if not p.get('is_high_value', False)]
    
    # 按评分排序
    high_value_papers.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    other_papers.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    
    def has_score_details_in_review(review_content):
        """检查review内容中是否已经包含JSON代码块格式的评分详情"""
        if not review_content:
            return False
        # 检查是否包含```json代码块，并且代码块中包含评分相关的字段（如"创新性"、"技术深度"等）
        pattern = r'```json\s*.*?(?:"创新性"|"技术深度"|"相关性"|"实用性").*?```'
        return bool(re.search(pattern, review_content, re.DOTALL | re.IGNORECASE))
    
    def extract_and_replace_score_details(review_content, score_details):
        """从review内容中提取评分详情JSON代码块，并替换为带总分的版本"""
        if not review_content or not score_details:
            return review_content
        
        # 匹配```json开始到```结束之间的内容
        pattern = r'(```json\s*)(.*?)(\s*```)'
        
        def replace_json(match):
            json_content = match.group(2).strip()
            # 检查是否包含评分相关字段
            if re.search(r'"(创新性|技术深度|相关性|实用性)"', json_content):
                # 替换为带总分的版本
                new_json_str = json.dumps(score_details, ensure_ascii=False, indent=2)
                return f"{match.group(1)}{new_json_str}{match.group(3)}"
            return match.group(0)
        
        # 替换所有匹配的JSON代码块
        result = re.sub(pattern, replace_json, review_content, flags=re.DOTALL | re.IGNORECASE)
        return result
    
    # 高价值论文（评分>3.0，需要进一步研究）
    if high_value_papers:
        report.append("## 🔥 高价值论文（评分>3.0，建议下载原文深入研究）")
        
        for i, paper in enumerate(high_value_papers, 1):
            report.append(f"### {i}. {paper['title']}")
            report.append("")
            
            # 添加翻译后的摘要（提到最前面）
            translated_content = paper.get('translated_content', '')
            # 如果翻译内容存在且不是错误信息，则显示翻译后的摘要
            if translated_content and translated_content not in ["摘要提取失败，无法处理", "摘要验证失败：检测到可能是AI虚构生成的内容", "摘要验证失败：关键词检测发现生成标志"]:
                report.append("**摘要**：")
                report.append("")
                report.append(translated_content)
                report.append("")
            
            # 添加评审内容
            review_content = paper.get('review', paper.get('summary', '')).strip()
            score_details = paper.get('score_details', {})
            
            # 如果review中已经包含评分详情，替换为带总分的版本
            if review_content and has_score_details_in_review(review_content) and score_details:
                review_content = extract_and_replace_score_details(review_content, score_details)
            
            if review_content:
                report.append(review_content)
                report.append("")
            
            # 检查review中是否已经包含评分详情，如果没有才添加
            if not has_score_details_in_review(review_content) and score_details:
                report.append("**评分详情**：")
                report.append("")
                report.append("```json")
                # 格式化JSON，确保美观
                json_str = json.dumps(score_details, ensure_ascii=False, indent=2)
                report.append(json_str)
                report.append("```")
                report.append("")
            
            # 添加论文链接
            report.append(f"🔗 [论文链接]({paper['link']})")
            report.append("")
            
            # 添加分隔符（最后一个论文后不添加）
            if i < len(high_value_papers):
                report.append("---")
                report.append("")
    
    # 其他相关论文
    if other_papers:
        report.append("## 📖 相关论文")
        
        for i, paper in enumerate(other_papers, 1):
            report.append(f"### {i}. {paper['title']}")
            report.append("")
            
            # 添加翻译后的摘要（提到最前面）
            translated_content = paper.get('translated_content', '')
            # 如果翻译内容存在且不是错误信息，则显示翻译后的摘要
            if translated_content and translated_content not in ["摘要提取失败，无法处理", "摘要验证失败：检测到可能是AI虚构生成的内容", "摘要验证失败：关键词检测发现生成标志"]:
                report.append("**摘要**：")
                report.append("")
                report.append(translated_content)
                report.append("")
            
            # 添加评审内容
            review_content = paper.get('review', paper.get('summary', '')).strip()
            score_details = paper.get('score_details', {})
            
            # 如果review中已经包含评分详情，替换为带总分的版本
            if review_content and has_score_details_in_review(review_content) and score_details:
                review_content = extract_and_replace_score_details(review_content, score_details)
            
            if review_content:
                report.append(review_content)
                report.append("")
            
            # 检查review中是否已经包含评分详情，如果没有才添加
            if not has_score_details_in_review(review_content):
                # 添加评分
                report.append(f"**评分：** {paper.get('score', 0.0):.2f}/4.0")
                report.append("")
                
                # 添加评分详情（JSON格式）
                if score_details:
                    report.append("**评分详情**：")
                    report.append("")
                    report.append("```json")
                    json_str = json.dumps(score_details, ensure_ascii=False, indent=2)
                    report.append(json_str)
                    report.append("```")
                    report.append("")
            
            # 添加论文链接
            report.append(f"🔗 [论文链接]({paper['link']})")
            report.append("")
            
            # 添加分隔符（最后一个论文后不添加）
            if i < len(other_papers):
                report.append("---")
                report.append("")
    
    # 统计信息（使用表格，格式与理想文件一致）
    report.append("## 📊 统计信息")
    report.append("")
    report.append("| 类别            | 数量       |")
    report.append("| ------------- | -------- |")
    report.append(f"| 高价值论文（评分>3.0） | {len(high_value_papers)} 篇     |")
    report.append(f"| 其他相关论文        | {len(other_papers)} 篇      |")
    report.append(f"| **总计**        | **{len(relevant_papers)} 篇** |")
    
    return "\n".join(report)


def export_all_papers_to_csv(relevant_papers, output_dir="reports"):
    """
    将所有符合关键词的论文导出到CSV文件（包含处理结果）
    
    Args:
        relevant_papers: 所有相关论文列表（符合关键词的论文）
        output_dir: 输出目录
    """
    if not relevant_papers:
        print("\n没有相关论文需要导出")
        return None
    
    # 按评分排序（处理成功的排在前面）
    relevant_papers_sorted = sorted(relevant_papers, 
                                   key=lambda x: (
                                       x.get('score', 0.0) if (x.get('review') and x.get('score_details')) else -1
                                   ), 
                                   reverse=True)
    
    # 准备数据
    csv_data = []
    
    for paper in relevant_papers_sorted:
        # 检查是否成功读取摘要（full_abstract不为None表示成功）
        # 成功读取摘要：full_abstract存在且不为None，且长度>0
        full_abstract = paper.get('full_abstract')
        is_success = 1 if (full_abstract is not None and len(full_abstract) > 0) else 0
        
        # 获取翻译后的摘要（优先使用翻译后的摘要）
        translated_content = paper.get('translated_content', '')
        # 如果翻译后的摘要存在且不是错误信息，使用翻译后的摘要；否则使用原始摘要
        if translated_content and translated_content not in [
            "摘要提取失败，无法处理", 
            "摘要验证失败：检测到可能是AI虚构生成的内容", 
            "摘要验证失败：关键词检测发现生成标志",
            "翻译失败:"
        ]:
            abstract_to_save = translated_content
        elif is_success == 1:
            # 如果翻译失败但摘要提取成功，使用原始摘要
            abstract_to_save = full_abstract
        else:
            # 摘要提取失败
            abstract_to_save = ''
        
        # 构建数据行
        row = {
            '论文标题': paper.get('title', ''),
            '论文链接': paper.get('link', ''),
            '摘要': abstract_to_save,  # 保存翻译后的摘要（如果翻译成功），否则保存原始摘要
            '处理结果': is_success,
        }
        
        csv_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(csv_data)
    
    # 生成CSV文件名
    csv_filename = f"{output_dir}/相关论文_{datetime.now().strftime('%Y%m%d')}.csv"
    
    # 保存到CSV
    try:
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')  # 使用utf-8-sig以支持Excel正确显示中文
        
        successful_count = sum(1 for p in relevant_papers if p.get('full_abstract') is not None and 
                              len(p.get('full_abstract', '')) > 0)
        failed_count = len(relevant_papers) - successful_count
        
        print(f"\n✓ 相关论文已导出到CSV: {csv_filename}")
        print(f"  - 共导出 {len(relevant_papers)} 篇相关论文（符合关键词的论文）")
        print(f"  - 成功读取摘要: {successful_count} 篇")
        print(f"  - 摘要读取失败: {failed_count} 篇")
        return csv_filename
    except Exception as e:
        logging.error(f"导出CSV时出错: {str(e)}")
        print(f"\n✗ 导出CSV失败: {str(e)}")
        return None



def main(on_log=None, on_paper_added=None, on_paper_updated=None, on_file_generated=None, on_agent_status=None, on_waiting_confirmation=None, get_confirmed_abstracts=None):
    """
    主程序
    
    Args:
        on_log: 日志回调函数 (level, message)
        on_paper_added: 论文添加回调函数 (paper)
        on_paper_updated: 论文更新回调函数 (paper_id, paper)
        on_file_generated: 文件生成回调函数 (file_info)
        on_agent_status: agent状态回调函数 (agent_name, status, target, output)
        on_waiting_confirmation: 等待确认回调函数 (papers_data) -> 返回确认后的摘要字典
        get_confirmed_abstracts: 获取确认后的摘要函数 () -> 返回 {paper_id: abstract_text}
    """
    print("=" * 80)
    print("Paper Summarizer - 学术论文自动总结系统")
    print("=" * 80)
    print()
    
    # 设置全局日志回调（用于CrewAI输出捕获）
    global crewai_log_callback, agent_status_callback
    crewai_log_callback = on_log
    agent_status_callback = on_agent_status
    
    # 辅助函数：发送日志
    def log(level, message):
        print(message)
        if on_log:
            on_log(level, message)
    
    # 辅助函数：发送论文添加
    def paper_added(paper):
        if on_paper_added:
            on_paper_added(paper)
    
    # 辅助函数：发送论文更新
    def paper_updated(paper_id, paper):
        if on_paper_updated:
            on_paper_updated(paper_id, paper)
    
    # 辅助函数：发送文件生成
    def file_generated(file_info):
        if on_file_generated:
            on_file_generated(file_info)
    
    # 1. 根据模式选择数据源
    if LOCAL_MODE:
        print("=" * 80)
        print("本地处理模式 (LOCAL=1)")
        print("=" * 80)
        print(f"CSV文件路径: {CSV_FILE_PATH}")
        print(f"标题列索引: {CSV_TITLE_COLUMN}")
        print(f"摘要列索引: {CSV_ABSTRACT_COLUMN}")
        if CSV_LINK_COLUMN:
            print(f"链接列索引: {CSV_LINK_COLUMN}")
        else:
            print("链接列索引: 未指定（将使用空字符串）")
        print()
        
        # 从CSV文件读取论文信息
        try:
            all_papers = load_papers_from_csv(
                csv_path=CSV_FILE_PATH,
                title_col=CSV_TITLE_COLUMN,
                abstract_col=CSV_ABSTRACT_COLUMN,
                link_col=CSV_LINK_COLUMN if CSV_LINK_COLUMN else None
            )
        except Exception as e:
            print(f"\n✗ 错误: {str(e)}")
            logging.error(f"从CSV读取论文信息失败: {str(e)}")
            return
        
        if not all_papers:
            print("\n没有从CSV文件中读取到论文信息")
            return
        
        print(f"\n总共从CSV读取到 {len(all_papers)} 篇论文")
        debug_logger.log(f"本地模式：从CSV文件 {CSV_FILE_PATH} 读取到 {len(all_papers)} 篇论文")
    else:
        print("=" * 80)
        print("邮件处理模式 (LOCAL=0)")
        print("=" * 80)
        print()
        
        # 1. 连接Gmail
        mail = connect_gmail()
        
        # 2. 获取邮件（从前START_DAYS天到前END_DAYS天）
        email_ids = fetch_scholar_emails(mail, start_days=START_DAYS, end_days=END_DAYS)
        
        if not email_ids:
            print("\n没有找到新的学术推送邮件")
            mail.close()
            mail.logout()
            return
            
        # 3. 处理邮件
        all_papers = []
        
        try:
            for email_id in email_ids[:MAX_EMAILS]:
                print(f"\n处理邮件 {email_id.decode()}...")
                
                try:
                    status, msg_data = mail.fetch(email_id, "(RFC822)")
                    
                    if status != 'OK':
                        print(f"  警告: 无法获取邮件内容 (状态: {status})")
                        continue
                    
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            
                            # 验证邮件日期是否在指定范围内
                            if not is_email_in_date_range(msg, start_days=START_DAYS, end_days=END_DAYS):
                                email_date_str = msg.get('Date', '未知')
                                print(f"  跳过: 邮件日期不在范围内 ({email_date_str})")
                                continue
                            
                            # 获取邮件正文
                            if msg.is_multipart():
                                for part in msg.walk():
                                    if part.get_content_type() == "text/html":
                                        body = part.get_payload(decode=True).decode()
                                        break
                            else:
                                body = msg.get_payload(decode=True).decode()
                            
                            # 提取论文信息
                            papers = extract_paper_info(body)
                            all_papers.extend(papers)
                            print(f"  提取到 {len(papers)} 篇论文")
                            
                except (imaplib.IMAP4.error, ssl.SSLError, OSError) as e:
                    print(f"  警告: 处理邮件时出错: {str(e)}")
                    # 尝试重新连接
                    try:
                        mail.close()
                    except:
                        pass
                    try:
                        mail = connect_gmail()
                        mail.select("inbox")
                    except Exception as reconnect_error:
                        print(f"  错误: 重新连接失败: {str(reconnect_error)}")
                        break
                    continue
                except Exception as e:
                    print(f"  警告: 处理邮件时出现未知错误: {str(e)}")
                    continue
        finally:
            # 确保连接被正确关闭
            try:
                mail.close()
            except:
                pass
            try:
                mail.logout()
            except:
                pass
        
        print(f"\n总共提取到 {len(all_papers)} 篇论文")
        debug_logger.log(f"总共提取到 {len(all_papers)} 篇论文")
    
    # 4. 使用相关性分析Agent筛选相关论文
    print("\n正在使用相关性分析Agent分析论文相关性...")
    debug_logger.log_separator("相关性分析（Agent）")
    relevant_papers = []
    
    for i, paper in enumerate(all_papers, 1):
        print(f"\n分析论文 {i}/{len(all_papers)}: {paper.get('title', '')[:60]}...")
        debug_logger.log(f"分析论文: {paper.get('title', '')[:60]}")
        
        try:
            # 使用CrewAI进行相关性分析
            relevance_task = create_relevance_analysis_task(paper)
            
            # 发送agent工作开始状态
            send_agent_status("相关性分析专家", "start", task=relevance_task)
            
            with capture_crewai_output():
                relevance_crew = Crew(
                    agents=[create_relevance_analyzer_agent()],
                    tasks=[relevance_task],
                    verbose=True,
                    share_crew=False
                )
                relevance_result = relevance_crew.kickoff()
            
            # 发送agent工作结束状态
            send_agent_status("相关性分析专家", "end", result=relevance_result)
            
            relevance_output = relevance_result.raw.strip()
            debug_logger.log(f"相关性分析输出: {relevance_output[:200]}...")
            
            # 解析相关性判断结果
            relevance_code = None
            
            # 查找"相关性判断："后面的数字
            result_match = re.search(r'相关性判断[：:]\s*([01])', relevance_output, re.I)
            if result_match:
                relevance_code = int(result_match.group(1))
            
            # 如果没找到，尝试查找独立的0或1
            if relevance_code is None:
                first_line_match = re.match(r'^\s*([01])\s*$', relevance_output.split('\n')[0] if relevance_output else '')
                if first_line_match:
                    relevance_code = int(first_line_match.group(1))
            
            # 如果还是没找到，检查是否包含"相关性判断：0"或"相关性判断：1"
            if relevance_code is None:
                if re.search(r'相关性判断[：:]\s*0|判断[：:]\s*0|^0\s*$', relevance_output, re.I | re.M):
                    relevance_code = 0
                elif re.search(r'相关性判断[：:]\s*1|判断[：:]\s*1|^1\s*$', relevance_output, re.I | re.M):
                    relevance_code = 1
            
            # 提取判断依据
            explanation_match = re.search(r'判断依据[：:]\s*(.+?)(?:\n\n|\n*$)', relevance_output, re.DOTALL | re.I)
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            
            if relevance_code == 1:
                paper['relevance_score'] = 1
                paper['relevance_explanation'] = explanation
                relevant_papers.append(paper)
                print(f"  ✓ 符合研究方向")
                debug_logger.log(f"相关论文: {paper.get('title', '')[:60]} (判断依据: {explanation[:100]})", "SUCCESS")
                # 发送论文添加回调
                if on_paper_added:
                    paper_id = f"paper_{len(relevant_papers)}"
                    paper['_paper_id'] = paper_id  # 保存ID到paper对象中
                    on_paper_added({
                        'id': paper_id,
                        'title': paper.get('title', ''),
                        'link': paper.get('link', ''),
                        'status': 'pending'
                    })
            else:
                print(f"  ✗ 不符合研究方向")
                debug_logger.log(f"不符合方向: {paper.get('title', '')[:60]} (判断依据: {explanation[:100] if explanation else '未提供'})")
        
        except Exception as e:
            print(f"  ✗ 相关性分析出错: {str(e)}")
            debug_logger.log(f"相关性分析出错: {paper.get('title', '')[:60]} - {str(e)}", "ERROR")
            # 出错时跳过该论文，不加入相关论文列表
            continue
    
    print(f"\n✓ 找到 {len(relevant_papers)} 篇相关论文")
    debug_logger.log(f"找到 {len(relevant_papers)} 篇相关论文")
    
    if not relevant_papers:
        print("\n没有找到相关论文")
        debug_logger.log("没有找到相关论文")
        debug_logger.close()
        return
    
    # 5. 获取完整摘要
    if LOCAL_MODE:
        # 本地模式：直接使用CSV中的摘要
        print("\n本地模式：使用CSV中的摘要信息...")
        debug_logger.log_separator("使用CSV摘要")
        for i, paper in enumerate(relevant_papers, 1):
            print(f"\n处理摘要 {i}/{len(relevant_papers)}: {paper['title'][:50]}...")
            debug_logger.log_paper_info(paper, index=i)
            
            # 从snippet中获取完整摘要（CSV中读取的摘要）
            snippet = paper.get('snippet', '')
            if snippet and len(snippet) > 0:
                # 如果snippet长度超过500，说明可能是完整摘要
                # 否则尝试从CSV的abstract列获取（如果有的话）
                full_abstract = snippet
                paper['full_abstract'] = full_abstract
                paper['is_pdf'] = False
                print(f"  ✓ 使用CSV中的摘要 ({len(full_abstract)} 字符)")
                debug_logger.log(f"✓ 使用CSV中的摘要 ({len(full_abstract)} 字符)", "SUCCESS")
            else:
                paper['full_abstract'] = None
                paper['is_pdf'] = False
                print(f"  ✗ CSV中未找到摘要信息")
                debug_logger.log(f"CSV中未找到摘要信息", "WARNING")
    else:
        # 邮件模式：从URL获取完整摘要
        print("\n正在获取论文完整摘要...")
        debug_logger.log_separator("获取完整摘要")
        for i, paper in enumerate(relevant_papers, 1):
            print(f"\n获取摘要 {i}/{len(relevant_papers)}: {paper['title'][:50]}...")
            debug_logger.log_paper_info(paper, index=i)
            try:
                full_abstract, is_pdf = get_full_abstract(paper)
                paper['full_abstract'] = full_abstract
                paper['is_pdf'] = is_pdf
                if full_abstract is None:
                    print(f"  ✗ 摘要提取失败（Agent反馈不包含摘要信息）")
                    debug_logger.log(f"摘要提取失败（Agent反馈不包含摘要信息）", "WARNING")
                elif is_pdf:
                    print(f"  ✓ 已从PDF提取摘要 ({len(full_abstract)} 字符)")
                    debug_logger.log(f"✓ 已从PDF提取摘要 ({len(full_abstract)} 字符)", "SUCCESS")
                else:
                    print(f"  ✓ 已从网页提取摘要 ({len(full_abstract)} 字符)")
                    debug_logger.log(f"✓ 已从网页提取摘要 ({len(full_abstract)} 字符)", "SUCCESS")
            except Exception as e:
                logging.error(f"获取摘要失败: {str(e)}")
                debug_logger.log(f"获取摘要失败: {str(e)}", "ERROR")
                # 如果获取失败，标记为None（提取失败）
                paper['full_abstract'] = None
                paper['is_pdf'] = False
                print(f"  ✗ 获取摘要失败")
                debug_logger.log(f"获取摘要失败，将跳过处理", "WARNING")
    
    # 5.5. 人工确认摘要（如果启用了确认功能）
    if on_waiting_confirmation and get_confirmed_abstracts:
        print("\n等待人工确认摘要...")
        debug_logger.log_separator("等待人工确认摘要")
        
        # 准备等待确认的论文数据
        confirmation_papers_data = {}
        for paper in relevant_papers:
            paper_id = paper.get('_paper_id', '')
            if paper_id:
                confirmation_papers_data[paper_id] = {
                    'title': paper.get('title', ''),
                    'abstract': paper.get('full_abstract', ''),
                    'link': paper.get('link', '')
                }
        
        # 发送等待确认消息（这会阻塞等待用户确认）
        confirmed_abstracts = on_waiting_confirmation(confirmation_papers_data)
        log('info', f'已发送 {len(confirmation_papers_data)} 篇论文等待人工确认')
        
        # 更新论文摘要（使用确认后的摘要）
        for paper in relevant_papers:
            paper_id = paper.get('_paper_id', '')
            if paper_id and paper_id in confirmed_abstracts:
                confirmed_abstract = confirmed_abstracts[paper_id].strip()
                if confirmed_abstract:
                    paper['full_abstract'] = confirmed_abstract
                    log('info', f'论文 {paper.get("title", "")[:50]} 摘要已确认')
                else:
                    # 如果确认后的摘要为空，标记为失败
                    paper['full_abstract'] = None
                    log('warning', f'论文 {paper.get("title", "")[:50]} 确认后的摘要为空，将跳过处理')
        
        print("\n人工确认完成，继续处理...")
        debug_logger.log("人工确认完成，继续处理")
    
    # 6. 使用 CrewAI 处理论文：验证 + 翻译 + 评审（仅处理摘要提取成功的论文）
    print("\n正在使用AI处理论文（验证 + 翻译 + 评审）...")
    debug_logger.log_separator("翻译和评审处理")
    
    # 筛选出摘要提取成功的论文
    papers_with_abstract = [p for p in relevant_papers if p.get('full_abstract') is not None and p.get('full_abstract').strip()]
    papers_without_abstract = [p for p in relevant_papers if not (p.get('full_abstract') and p.get('full_abstract').strip())]
    
    print(f"  摘要提取成功: {len(papers_with_abstract)} 篇，将进行翻译和评审")
    print(f"  摘要提取失败: {len(papers_without_abstract)} 篇，将跳过处理")
    debug_logger.log(f"摘要提取成功: {len(papers_with_abstract)} 篇")
    debug_logger.log(f"摘要提取失败: {len(papers_without_abstract)} 篇")
    
    # 只处理摘要提取成功的论文
    for i, paper in enumerate(papers_with_abstract, 1):
        print(f"\n处理 {i}/{len(papers_with_abstract)}: {paper['title'][:50]}...")
        debug_logger.log_paper_info(paper, index=i)
        
        # 使用完整摘要进行处理
        full_abstract = paper.get('full_abstract')
        
        # 确定摘要来源类型
        source_type = "PDF" if paper.get('is_pdf', False) else "网页"
        
        # 使用 CrewAI 框架处理（包含验证、翻译、评审）
        result = process_paper_with_crewai(paper, full_abstract, source_type)
        
        # 更新论文信息
        paper['translated_content'] = result['translated_content']
        paper['review'] = result['review']
        paper['score'] = result['score']
        paper['score_details'] = result['score_details']
        paper['is_high_value'] = result['is_high_value']
        
        print(f"  ✓ 完成 - 评分: {paper['score']:.2f}/4.0", end="")
        if paper['is_high_value']:
            print(" [高价值论文 ⭐]")
        else:
            print()
        
        # 发送论文更新回调
        if on_paper_updated:
            paper_id = paper.get('_paper_id', f"paper_{i}")
            
            # 判断处理是否成功：检查是否包含错误信息
            translated_content = paper.get('translated_content', '')
            review_content = paper.get('review', '')
            
            # 定义错误关键词
            error_keywords = [
                '摘要验证失败',
                '翻译失败',
                '评审失败',
                '摘要提取失败',
                '无法处理'
            ]
            
            # 检查是否包含错误信息
            is_error = any(keyword in translated_content or keyword in review_content 
                          for keyword in error_keywords)
            
            # 根据处理结果设置状态
            paper_status = 'error' if is_error else 'success'
            
            on_paper_updated(paper_id, {
                'status': paper_status,
                'abstract': translated_content,
                'score': paper.get('score', 0.0)
            })
    
    # 标记摘要提取失败的论文
    for idx, paper in enumerate(papers_without_abstract, 1):
        paper['translated_content'] = "摘要提取失败，无法处理"
        paper['review'] = "摘要提取失败，无法处理"
        paper['score'] = 0.0
        paper['score_details'] = {}
        paper['is_high_value'] = False
        
        # 发送论文更新回调（摘要提取失败，状态为error）
        if on_paper_updated:
            paper_id = paper.get('_paper_id', f"paper_failed_{idx}")
            on_paper_updated(paper_id, {
                'status': 'error',
                'abstract': "摘要提取失败，无法处理",
                'score': 0.0
            })
    
    # 6. 生成日报（只包含成功处理的论文）
    # 只包含成功处理的论文（有review和score_details的）
    successful_papers = [p for p in relevant_papers if p.get('review') and p.get('score_details') and 
                        p.get('review') != "摘要提取失败，无法处理"]
    
    if successful_papers:
        print("\n生成日报...")
        report = generate_daily_report(successful_papers)
        
        # 7. 保存报告（Markdown 格式，命名方式与理想格式一致）
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        
        # 文件命名格式：Robotics_Academic_Daily_YYYYMMDD .md（注意有空格）
        filename = f"{output_dir}/Robotics_Academic_Daily_{datetime.now().strftime('%Y%m%d')}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ 报告已保存到: {filename}")
        # 发送文件生成回调
        if on_file_generated:
            file_size = os.path.getsize(filename) if os.path.exists(filename) else 0
            on_file_generated({
                'name': os.path.basename(filename),
                'path': filename,
                'type': 'md',
                'size': f"{file_size / 1024:.1f} KB",
                'time': datetime.now().strftime('%H:%M:%S')
            })
    else:
        print("\n没有成功处理的论文，不生成日报")
        debug_logger.log("没有成功处理的论文，不生成日报", "INFO")
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        report = None  # 标记没有报告
    
    # 7.1. 如果配置了备份路径，同时保存到备份目录（仅当有报告时）
    if report and BACKUP_DIR:
        try:
            os.makedirs(BACKUP_DIR, exist_ok=True)
            backup_filename = os.path.join(BACKUP_DIR, f"Robotics_Academic_Daily_{datetime.now().strftime('%Y%m%d')}.md")
            with open(backup_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✓ 报告已另存到: {backup_filename}")
        except Exception as e:
            logging.warning(f"保存到备份目录失败: {str(e)}")
            print(f"⚠ 警告: 无法保存到备份目录: {str(e)}")
    
    # 8. 导出所有符合关键词的论文到CSV（包含处理结果）
    csv_file = export_all_papers_to_csv(relevant_papers, output_dir)
    # 发送CSV文件生成回调
    if csv_file and on_file_generated:
        file_size = os.path.getsize(csv_file) if os.path.exists(csv_file) else 0
        on_file_generated({
            'name': os.path.basename(csv_file),
            'path': csv_file,
            'type': 'csv',
            'size': f"{file_size / 1024:.1f} KB",
            'time': datetime.now().strftime('%H:%M:%S')
        })
    
    # 关闭调试日志
    debug_logger.close()
    
    # 关闭 CrewAI 日志文件处理器和输出捕获
    if 'crewai_file_handler' in globals():
        crewai_file_handler.close()
        # 从所有日志记录器中移除处理器
        for logger_name in ['crewai', 'crewai.agent', 'crewai.task', 'crewai.crew', 'litellm']:
            logger = logging.getLogger(logger_name)
            if crewai_file_handler in logger.handlers:
                logger.removeHandler(crewai_file_handler)
    
    # 关闭 CrewAI 输出捕获器
    global crewai_output_capture
    if crewai_output_capture:
        crewai_output_capture.close()
        crewai_output_capture = None
    
    logging.info(f"CrewAI 处理过程日志已保存到: {crewai_log_file}")
    
    # 输出最终报告（如果有成功处理的论文）
    if 'report' in locals() and report:
        print("\n" + "=" * 80)
        print(report)
    
    # 清理异步资源（修复 LiteLLM 异步客户端警告）
    try:
        import litellm
        # 尝试关闭 LiteLLM 的异步客户端
        if hasattr(litellm, 'close_litellm_async_clients'):
            try:
                # 尝试获取现有的事件循环
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("Event loop is closed")
                except RuntimeError:
                    # 如果没有事件循环或已关闭，创建新的
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # 运行清理协程
                if not loop.is_closed():
                    loop.run_until_complete(litellm.close_litellm_async_clients())
                    # 清理未完成的任务
                    pending = asyncio.all_tasks(loop)
                    if pending:
                        for task in pending:
                            task.cancel()
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    # 关闭事件循环
                    loop.close()
            except Exception as e:
                # 如果清理失败，记录但不影响程序退出
                logging.debug(f"清理 LiteLLM 异步客户端时出错: {str(e)}")
    except (ImportError, AttributeError):
        # 如果 litellm 不可用，忽略
        pass
    
    print("\n" + "=" * 80)
    print(report)


if __name__ == "__main__":
    try:
        main()
    finally:
        # 确保在程序退出前清理所有异步资源
        try:
            import litellm
            if hasattr(litellm, 'close_litellm_async_clients'):
                try:
                    # 创建新的事件循环来执行清理
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(litellm.close_litellm_async_clients())
                    finally:
                        # 清理并关闭事件循环
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        loop.close()
                except Exception:
                    # 忽略清理错误，确保程序能正常退出
                    pass
        except (ImportError, AttributeError):
            pass