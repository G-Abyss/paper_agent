#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件处理解析主管Agent

负责：
1. 从邮件中提取论文标题和摘要片段
2. 将标题翻译成中文（双语输出）
3. 进行相关性分析（根据用户提供的关键词）
4. 判断是否可以将文章交给知识工程师处理
"""

from crewai import Agent, Task
from agents.base import get_llm
import logging
import re
from typing import Dict, Any, Optional


def create_email_processing_agent(research_keywords: str = None, expanded_keywords: str = None):
    """
    创建邮件处理解析主管Agent
    
    Args:
        research_keywords: 研究方向关键词字符串
        expanded_keywords: 扩写后的研究方向描述（可选）
    """
    # 根据扩写关键词生成backstory
    if expanded_keywords and expanded_keywords.strip():
        import re
        directions_match = re.search(r'## 研究方向列表\s*\n(.*?)(?=\n##|$)', expanded_keywords, re.DOTALL)
        if directions_match:
            directions_text = directions_match.group(1).strip()
            directions = re.findall(r'-\s*(.+?)(?:\n|$)', directions_text)
            if directions:
                research_fields = '、'.join([d.strip() for d in directions[:5]])
                backstory = (
                    f"你是一位在{research_fields}等领域拥有丰富研究经验的邮件处理专家。"
                    f"你能够从邮件中准确提取论文标题和摘要片段，将标题翻译成中文（输出双语），"
                    f"并基于论文标题和摘要片段信息，准确判断论文的研究方向是否与目标领域相关。"
                    f"你严格基于提供的信息进行分析，不会虚构或推测论文内容，确保判断的准确性和可靠性。"
                )
            else:
                backstory = (
                    "你是一位在学术研究领域拥有丰富研究经验的邮件处理专家。"
                    "你能够从邮件中准确提取论文标题和摘要片段，将标题翻译成中文（输出双语），"
                    "并基于论文标题和摘要片段信息，准确判断论文的研究方向是否与目标领域相关。"
                    "你严格基于提供的信息进行分析，不会虚构或推测论文内容，确保判断的准确性和可靠性。"
                )
        else:
            backstory = (
                "你是一位在学术研究领域拥有丰富研究经验的邮件处理专家。"
                "你能够从邮件中准确提取论文标题和摘要片段，将标题翻译成中文（输出双语），"
                "并基于论文标题和摘要片段信息，准确判断论文的研究方向是否与目标领域相关。"
                "你严格基于提供的信息进行分析，不会虚构或推测论文内容，确保判断的准确性和可靠性。"
            )
    else:
        backstory = (
            "你是一位在学术研究领域拥有丰富研究经验的邮件处理专家。"
            "你能够从邮件中准确提取论文标题和摘要片段，将标题翻译成中文（输出双语），"
            "并基于论文标题和摘要片段信息，准确判断论文的研究方向是否与目标领域相关。"
            "你严格基于提供的信息进行分析，不会虚构或推测论文内容，确保判断的准确性和可靠性。"
        )
    
    return Agent(
        role="邮件处理解析主管",
        goal=(
            "从邮件中提取论文标题和摘要片段，将标题翻译成中文（输出双语），"
            "并基于论文标题和摘要片段信息，准确判断论文的研究方向是否与目标领域相关。"
        ),
        backstory=backstory,
        allow_delegation=False,
        verbose=True,
        llm=get_llm(),
        max_iter=3,
        max_execution_time=300
    )


def create_email_processing_task(
    paper_title: str,
    paper_snippet: str,
    paper_link: str,
    research_keywords: str = None,
    expanded_keywords: str = None
) -> Task:
    """
    创建邮件处理任务
    
    Args:
        paper_title: 论文标题（英文）
        paper_snippet: 邮件中的摘要片段
        paper_link: 论文链接
        research_keywords: 研究方向关键词字符串
        expanded_keywords: 扩写后的研究方向描述（可选）
    """
    # 如果提供了扩写后的关键词描述，优先使用
    if expanded_keywords and expanded_keywords.strip():
        research_directions_section = expanded_keywords
    else:
        # 否则使用原始关键词列表
        if research_keywords:
            # 将关键词字符串按逗号、顿号、分号等分隔符分割
            keywords_list = re.split(r'[，,、;；\s]+', research_keywords.strip())
            keywords_list = [kw.strip() for kw in keywords_list if kw.strip()]
        else:
            # 默认研究方向
            keywords_list = ['遥操作（Teleoperation）', '力控（Force Control）', '灵巧手（Dexterous Manipulation/Hand）', 
                            '机器人动力学（Robot Dynamics）', '机器学习（Machine Learning）']
        
        # 构建研究方向列表字符串
        research_directions_section = "\n".join([f"- {kw}" for kw in keywords_list])
    
    return Task(
        description=(
            f"## 任务目标\n"
            f"从邮件中提取论文标题和摘要片段，将标题翻译成中文（输出双语），"
            f"并基于论文标题和摘要片段信息，判断论文是否符合以下研究方向：\n\n"
            f"{research_directions_section}\n\n"
            f"## 输入信息\n"
            f"**论文标题（英文）**：{paper_title}\n\n"
            f"**论文链接**：{paper_link}\n\n"
            f"**邮件摘要片段**：\n```\n{paper_snippet}\n```\n\n"
            f"## 执行步骤\n"
            f"### 步骤1：提取和规范化论文标题\n"
            f"- 从输入的论文标题中提取完整的标题文本\n"
            f"- 规范化标题格式（单词之间用单个空格分隔，去除多余空格）\n"
            f"- 确保标题格式规范\n\n"
            f"### 步骤2：翻译标题（输出双语）\n"
            f"- 将英文标题准确翻译成中文\n"
            f"- 使用专业术语，确保翻译准确性\n"
            f"- 输出格式：英文标题 | 中文标题（或：中文标题 (英文标题)）\n\n"
            f"### 步骤3：相关性分析\n"
            f"- 基于论文标题和邮件摘要片段信息，判断论文是否与上述研究方向相关\n"
            f"- 判断时应该：\n"
            f"  1. 仔细阅读每个研究方向的核心含义、关键特征和判断标准\n"
            f"  2. 基于论文标题和邮件摘要片段信息，判断论文是否涉及这些研究方向\n"
            f"  3. 如果论文的研究内容、技术方法、应用场景与研究方向相关，则视为符合\n"
            f"  4. 如果论文明显属于其他领域（如纯计算机视觉、自动驾驶、导航等），则视为不符合\n"
            f"  5. 参考研究方向描述中的判断标准，进行准确判断\n"
            f"- **重要**：判断必须基于提供的标题和摘要片段信息，不能虚构或推测论文内容\n\n"
            f"## 判断标准\n"
            f"### 输出1（符合方向）的条件\n"
            f"- 论文标题或邮件摘要片段信息明确表明论文研究内容与上述任一方向相关\n"
            f"- 论文涉及的技术、方法、应用场景与上述方向的核心特征匹配\n"
            f"- 论文的研究目标、技术路线或应用领域与上述方向高度相关\n"
            f"- **重要**：判断必须基于提供的标题和摘要片段信息，不能虚构或推测论文内容\n\n"
            f"### 输出0（不符合方向）的条件\n"
            f"- 论文标题和邮件摘要片段信息无法明确判断与上述研究方向相关\n"
            f"- 论文研究内容明显属于其他领域（如纯计算机视觉、自动驾驶、导航等）\n"
            f"- 论文虽然涉及相关技术，但研究重点不在上述研究方向范围内\n"
            f"- 信息不足，无法准确判断论文研究方向\n\n"
            f"## 输出格式（严格遵循）\n"
            f"```\n"
            f"论文标题（英文）：[规范化后的英文标题]\n"
            f"论文标题（中文）：[翻译后的中文标题]\n"
            f"论文标题（双语）：[英文标题 | 中文标题]\n"
            f"相关性判断：1\n"
            f"相关性说明：[简要说明为什么符合或不符合研究方向]\n"
            f"```\n"
            f"或\n"
            f"```\n"
            f"论文标题（英文）：[规范化后的英文标题]\n"
            f"论文标题（中文）：[翻译后的中文标题]\n"
            f"论文标题（双语）：[英文标题 | 中文标题]\n"
            f"相关性判断：0\n"
            f"相关性说明：[简要说明为什么不符合研究方向]\n"
            f"```\n\n"
            f"## 输出要求\n"
            f"- 必须输出规范化后的英文标题、中文标题和双语标题\n"
            f"- 必须输出相关性判断（1或0）和相关性说明\n"
            f"- 标题翻译应准确、专业\n"
            f"- 相关性说明应简洁明了，基于提供的标题和摘要片段信息\n"
        ),
        agent=create_email_processing_agent(research_keywords, expanded_keywords),
        expected_output="包含论文标题（英文、中文、双语）、相关性判断和相关性说明的结构化输出"
    )


def process_email_paper_with_agent(
    paper: Dict[str, Any],
    research_keywords: str = None,
    expanded_keywords: str = None
) -> Dict[str, Any]:
    """
    使用邮件处理解析主管Agent处理文章
    
    Args:
        paper: 文章信息字典，包含title, snippet, link等字段
        research_keywords: 研究方向关键词字符串
        expanded_keywords: 扩写后的研究方向描述（可选）
    
    Returns:
        Dict: 处理结果，包含：
            - title_en: 规范化后的英文标题
            - title_cn: 中文标题
            - title_bilingual: 双语标题
            - relevance_score: 相关性分数（1或0）
            - relevance_explanation: 相关性说明
            - success: 是否处理成功
    """
    from crewai import Crew
    from callbacks.crewai_callbacks import capture_crewai_output
    
    try:
        paper_title = paper.get('title', '')
        paper_snippet = paper.get('snippet', '')
        paper_link = paper.get('link', '')
        
        if not paper_title:
            return {
                'success': False,
                'error': '论文标题为空'
            }
        
        # 创建任务和Agent
        task = create_email_processing_task(
            paper_title=paper_title,
            paper_snippet=paper_snippet or '',
            paper_link=paper_link or '',
            research_keywords=research_keywords,
            expanded_keywords=expanded_keywords
        )
        agent = create_email_processing_agent(research_keywords, expanded_keywords)
        
        # 执行任务
        with capture_crewai_output():
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                share_crew=False
            )
            result = crew.kickoff()
        
        # 解析结果
        output = result.raw.strip() if hasattr(result, 'raw') else str(result).strip()
        
        # 提取标题信息
        title_en_match = re.search(r'论文标题（英文）[：:]\s*(.+?)(?:\n|$)', output, re.I)
        title_cn_match = re.search(r'论文标题（中文）[：:]\s*(.+?)(?:\n|$)', output, re.I)
        title_bilingual_match = re.search(r'论文标题（双语）[：:]\s*(.+?)(?:\n|$)', output, re.I)
        
        title_en = title_en_match.group(1).strip() if title_en_match else paper_title
        title_cn = title_cn_match.group(1).strip() if title_cn_match else ''
        title_bilingual = title_bilingual_match.group(1).strip() if title_bilingual_match else f"{title_en} | {title_cn}"
        
        # 如果没有找到中文标题，尝试从双语标题中提取
        if not title_cn and title_bilingual:
            if '|' in title_bilingual:
                parts = title_bilingual.split('|')
                if len(parts) == 2:
                    title_cn = parts[1].strip()
            elif '(' in title_bilingual and ')' in title_bilingual:
                # 格式：中文标题 (英文标题)
                match = re.match(r'(.+?)\s*\((.+?)\)', title_bilingual)
                if match:
                    title_cn = match.group(1).strip()
                    title_en = match.group(2).strip()
        
        # 提取相关性判断
        relevance_match = re.search(r'相关性判断[：:]\s*([01])', output, re.I)
        relevance_score = int(relevance_match.group(1)) if relevance_match else 0
        
        # 提取相关性说明
        explanation_match = re.search(r'相关性说明[：:]\s*(.+?)(?:\n\n|$)', output, re.DOTALL | re.I)
        relevance_explanation = explanation_match.group(1).strip() if explanation_match else ''
        
        return {
            'success': True,
            'title_en': title_en,
            'title_cn': title_cn,
            'title_bilingual': title_bilingual,
            'relevance_score': relevance_score,
            'relevance_explanation': relevance_explanation
        }
        
    except Exception as e:
        logging.error(f"处理邮件文章失败: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

