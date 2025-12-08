#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrewAI回调

主要修改点：
- 删除未使用的导入（DEBUG_DIR）
- 优化代码结构和注释
"""

import os
import sys
import re
import logging
from contextlib import contextmanager
from datetime import datetime
from utils.text_utils import remove_ansi_codes


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

# 全局 CrewAI 输出捕获器
crewai_output_capture = None


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


@contextmanager
def capture_crewai_output(log_callback=None):
    """上下文管理器：捕获 CrewAI 的输出到文件"""
    global crewai_output_capture
    
    # 延迟导入，避免循环导入
    from callbacks.frontend_callbacks import crewai_log_callback
    
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


def extract_paper_title_from_task(task_description):
    """
    从任务描述中提取论文标题
    
    Args:
        task_description: 任务描述字符串
        
    Returns:
        论文标题字符串，如果未找到则返回None
    """
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

