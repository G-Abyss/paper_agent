#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF标题提取Agent - 从PDF前几页提取论文标题
"""

from crewai import Agent, Task
from config import llm


def create_pdf_title_extractor_agent():
    """创建PDF标题提取 Agent"""
    return Agent(
        role="论文标题提取专家",
        goal="从PDF文件的前几页文本中准确识别和提取论文标题。标题通常是PDF开头最显眼的文本，可能是第一页的第一行或前几行，也可能是\"Title:\"、\"题目：\"等标记后的内容。",
        backstory="你是一位专业的学术论文标题识别专家。你擅长从PDF文档的前几页中识别论文标题。你了解学术论文的常见格式：标题通常出现在第一页的顶部，可能是加粗、大字体或特殊格式。你能够识别各种格式的标题标记，如'Title:'、'题目：'、'论文标题：'等，也能识别没有明确标记但格式明显的标题。你只提取标题文本，不添加任何额外内容。",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=3,
        max_execution_time=60
    )


def create_pdf_title_extraction_task(pdf_text_preview: str) -> Task:
    """创建PDF标题提取任务"""
    return Task(
        description=(
            f"## 任务目标\n"
            f"从PDF文件的前几页文本中准确提取论文标题。\n\n"
            f"## PDF文本预览（前3000字符）\n"
            f"{pdf_text_preview}\n\n"
            f"## 执行步骤\n"
            f"### 步骤1：识别标题位置\n"
            f"- 查找PDF开头最显眼的文本（通常是第一页的前几行）\n"
            f"- 查找标题标记（如'Title:'、'题目：'、'论文标题：'等）\n"
            f"- 识别格式明显的标题（大字体、加粗、居中显示等）\n\n"
            f"### 步骤2：提取标题文本\n"
            f"- 提取完整的标题文本\n"
            f"- 去除多余的空白字符和换行\n"
            f"- 保持标题的原始格式（但去除明显的格式标记）\n"
            f"- 如果标题跨多行，合并为一行\n\n"
            f"### 步骤3：验证标题\n"
            f"- 确保提取的是标题，而不是作者、摘要或其他内容\n"
            f"- 标题通常比较简短（通常不超过200字符）\n"
            f"- 标题通常不包含\"Abstract\"、\"Introduction\"等章节标题\n\n"
            f"## 输出要求\n"
            f"- 只输出论文标题，不要添加任何前缀、后缀或说明\n"
            f"- 如果无法确定标题，输出\"未知标题\"\n"
            f"- 标题应该是完整的，去除多余的空白字符\n"
            f"- 如果标题包含特殊字符，保留它们（但去除明显的格式标记）"
        ),
        agent=create_pdf_title_extractor_agent(),
        expected_output="论文标题（纯文本，无额外说明）"
    )

