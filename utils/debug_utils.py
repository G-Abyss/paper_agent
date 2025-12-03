#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试工具
"""

import os
from datetime import datetime
from config import DEBUG_MODE, DEBUG_DIR


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


# 创建全局调试日志记录器实例
debug_logger = DebugLogger(enabled=DEBUG_MODE)
