#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译Agent
"""

from crewai import Agent, Task
from agents.base import get_llm


def create_translator_agent(expanded_keywords=None):
    """
    创建专业翻译 Agent
    
    Args:
        expanded_keywords: 扩写后的研究方向描述（可选），用于定制agent的专业领域背景
    """
    # 根据扩写关键词生成backstory
    if expanded_keywords and expanded_keywords.strip():
        # 提取研究方向列表（从扩写结果中提取）
        import re
        # 尝试提取研究方向列表部分
        directions_match = re.search(r'## 研究方向列表\s*\n(.*?)(?=\n##|$)', expanded_keywords, re.DOTALL)
        if directions_match:
            directions_text = directions_match.group(1).strip()
            # 提取研究方向名称
            directions = re.findall(r'-\s*(.+?)(?:\n|$)', directions_text)
            if directions:
                research_fields = '、'.join([d.strip() for d in directions[:5]])  # 最多取5个
                backstory = f"你是一位在{research_fields}等领域拥有深厚专业背景的翻译专家。你擅长将英文学术论文翻译成中文，能够准确处理专业术语，保持技术描述的完整性和逻辑结构的清晰性。你熟悉这些领域的标准中文术语和表达方式。"
            else:
                backstory = "你是一位在学术研究领域拥有深厚专业背景的翻译专家。你擅长将英文学术论文翻译成中文，能够准确处理专业术语，保持技术描述的完整性和逻辑结构的清晰性。"
        else:
            backstory = "你是一位在学术研究领域拥有深厚专业背景的翻译专家。你擅长将英文学术论文翻译成中文，能够准确处理专业术语，保持技术描述的完整性和逻辑结构的清晰性。"
    else:
        backstory = "你是一位在机器人学、控制理论、遥操作、机器人动力学和力控领域拥有深厚专业背景的翻译专家。你擅长将英文学术论文翻译成中文，能够准确处理专业术语，保持技术描述的完整性和逻辑结构的清晰性。"
    
    return Agent(
        role="专业翻译专家",
        goal="将英文论文内容准确、专业地翻译成中文，确保专业术语的准确性和技术表达的清晰性",
        backstory=backstory,
        allow_delegation=False,
        verbose=True,
        llm=get_llm(),
        max_iter=3,
        max_execution_time=300
    )


def create_translation_task(paper, abstract_text, expanded_keywords=None):
    """
    创建翻译任务（输出包含中英文双语）
    
    Args:
        paper: 论文信息字典
        abstract_text: 摘要文本
        expanded_keywords: 扩写后的研究方向描述（可选），用于定制专业领域
    """
    # 根据扩写关键词生成专业领域描述
    if expanded_keywords and expanded_keywords.strip():
        # 提取研究方向列表
        import re
        directions_match = re.search(r'## 研究方向列表\s*\n(.*?)(?=\n##|$)', expanded_keywords, re.DOTALL)
        if directions_match:
            directions_text = directions_match.group(1).strip()
            directions = re.findall(r'-\s*(.+?)(?:\n|$)', directions_text)
            if directions:
                research_fields = '、'.join([d.strip() for d in directions[:5]])
                field_description = f"使用{research_fields}等领域的标准中文术语"
            else:
                field_description = "使用相关研究领域的标准中文术语"
        else:
            field_description = "使用相关研究领域的标准中文术语"
    else:
        field_description = "使用机器人学、控制理论、遥操作、机器人动力学和力控领域的标准中文术语"
    
    return Task(
        description=(
            f"## 任务目标\n"
            f"将英文论文摘要准确、专业地翻译成中文，确保专业术语准确性和技术描述完整性。输出时需同时包含中文翻译和英文原文。\n\n"
            f"## 输入信息\n"
            f"**论文标题**：{paper['title']}\n\n"
            f"**论文摘要（英文原文）**：\n```\n{abstract_text}\n```\n\n"
            f"## 翻译原则\n"
            f"### 原则1：专业术语准确性\n"
            f"- {field_description}\n"
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
            f"## 输出格式（严格遵循）\n"
            f"```\n"
            f"中文翻译：\n[翻译后的中文摘要内容]\n\n"
            f"英文原文：\n[输入的英文原文摘要内容]\n"
            f"```\n\n"
            f"## 输出要求\n"
            f"- 必须同时输出中文翻译和英文原文\n"
            f"- 中文翻译应保持原文的结构和逻辑\n"
            f"- 英文原文必须与输入内容完全一致，不得修改\n"
            f"- 确保翻译流畅、自然，符合中文表达习惯\n"
            f"- 不添加任何额外的说明或注释"
        ),
        agent=create_translator_agent(),
        expected_output="输出包含两部分：1) 中文翻译（翻译后的中文摘要内容）；2) 英文原文（与输入完全一致的英文摘要内容）"
    )

