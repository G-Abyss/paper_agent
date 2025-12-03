#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置和初始化模块
"""

import os
import asyncio
import warnings
import atexit
from datetime import datetime
from dotenv import load_dotenv
from crewai import LLM
import logging

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

# 加载环境变量
load_dotenv()

# 邮件配置
QMAIL_USER = os.getenv('QMAIL_USER')
QMAIL_PASSWORD = os.getenv('QMAIL_PASSWORD')

# Ollama配置
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:32b')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://192.168.2.169:11434')

# 邮件处理配置
MAX_EMAILS = int(os.getenv('MAX_EMAILS', 20))
# 日期范围配置：从前START_DAYS天到前END_DAYS天
# 例如：START_DAYS=3, END_DAYS=0 表示从前3天到今天
#      START_DAYS=7, END_DAYS=3 表示从前7天到前3天
START_DAYS = int(os.getenv('START_DAYS', 1))  # 默认从前1天开始
END_DAYS = int(os.getenv('END_DAYS', 0))  # 默认到今天（前0天）

# 备份路径配置（可选）：如果设置了此路径，报告会同时保存到该路径
BACKUP_DIR = os.getenv('BACKUP_DIR', '')  # 默认为空，不进行备份

# 调试模式配置
DEBUG_MODE = os.getenv('DEBUG_MODE', '0') == '1'  # 从环境变量读取，默认为False
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

