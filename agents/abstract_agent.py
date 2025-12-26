#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摘要提取Agent
"""

from crewai import Agent, Task
from bs4 import BeautifulSoup
from agents.base import get_llm
from agents.base import fetch_webpage_tool


def extract_abstract_task_logic(paper_title, paper_link):
    """直接执行摘要提取逻辑（包装函数），跳过 Agent 的工具决策环"""
    from agents.base import fetch_webpage_content
    import logging
    import re

    logging.info(f"--- [直接执行爬取] 尝试获取: {paper_link} ---")
    # 直接调用工具函数，获取原始网页文本
    web_content = fetch_webpage_content.run(paper_link)
    
    if "获取失败" in web_content or "Timeout" in web_content:
        logging.warning(f"--- [爬取失败] 无法获取网页内容，将返回空摘要 ---")
        return "提取结果：0\n摘要内容："

    # 如果爬到了内容，让 Agent 进行一次快速格式化/提取
    agent = create_abstract_extractor_agent()
    task = Task(
        description=(
            f"## 任务描述\n"
            f"从以下网页文本中提取论文《{paper_title}》的摘要（Abstract）部分。\n"
            f"网页文本：\n{web_content[:10000]}\n\n"
            f"## 输出规范\n"
            f"提取结果：1\n"
            f"摘要内容：[提取的文本]"
        ),
        agent=agent,
        expected_output="输出提取结果和摘要内容"
    )
    
    from crewai import Crew
    from callbacks.crewai_callbacks import capture_crewai_output
    
    try:
        with capture_crewai_output():
            crew = Crew(agents=[agent], tasks=[task], verbose=True)
            result = crew.kickoff()
        return str(result)
    except Exception as e:
        logging.error(f"Agent格式化摘要失败: {str(e)}")
        return "提取结果：0\n摘要内容："

def create_abstract_extractor_agent():
    """创建格式化 Agent"""
    return Agent(
        role="信息提取员",
        goal="从杂乱的网页文本中准确提取摘要内容",
        backstory="你擅长从网页HTML转化后的纯文本中定位'Abstract'部分并提取。你的响应必须极其简练。",
        allow_delegation=False,
        verbose=True,
        llm=get_llm(),
        max_iter=1
    )

# 保持旧函数签名兼容性，但内部调用逻辑重构
def create_abstract_extraction_task(paper_title, paper_url):
    """(已弃用工具调用模式) 请改用 extract_abstract_task_logic"""
    pass



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
        expected_output="首先通过 Thought 过程分析输入文本。最终通过 Final Answer 按照指定格式输出结果：首先输出提取结果（1表示成功，0表示失败），如果成功则输出从原文中逐字提取的摘要内容（严禁生成或修改，禁止添加任何说明或注释）"
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
        expected_output="首先通过 Thought 过程分析输入的网页文本。最终通过 Final Answer 按照指定格式输出结果：首先输出提取结果（1表示成功，0表示失败），如果成功则输出从原文中逐字提取的摘要内容（严禁生成或修改，禁止添加任何说明或注释）"
    )

