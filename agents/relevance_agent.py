#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相关性分析Agent
"""

from crewai import Agent, Task
from agents.base import get_llm


def create_relevance_analyzer_agent(expanded_keywords=None):
    """
    创建相关性分析 Agent（仅通过邮件内容判断，不使用工具，具有记忆功能）
    
    Args:
        expanded_keywords: 扩写后的研究方向描述（可选），用于定制agent的专业领域背景
    """
    # 根据扩写关键词生成backstory
    if expanded_keywords and expanded_keywords.strip():
        # 提取研究方向列表
        import re
        directions_match = re.search(r'## 研究方向列表\s*\n(.*?)(?=\n##|$)', expanded_keywords, re.DOTALL)
        if directions_match:
            directions_text = directions_match.group(1).strip()
            directions = re.findall(r'-\s*(.+?)(?:\n|$)', directions_text)
            if directions:
                research_fields = '、'.join([d.strip() for d in directions[:5]])
                backstory = f"你是一位在{research_fields}等领域拥有丰富研究经验的专家。你能够基于论文标题和邮件中的片段信息，准确判断论文的研究方向是否与目标领域相关。你严格基于提供的信息进行分析，不会虚构或推测论文内容，确保判断的准确性和可靠性。你具有记忆功能，能够记住已经处理过的论文（通过标题和URL识别），如果发现是重复论文，会直接跳过处理。"
            else:
                backstory = "你是一位在学术研究领域拥有丰富研究经验的专家。你能够基于论文标题和邮件中的片段信息，准确判断论文的研究方向是否与目标领域相关。你严格基于提供的信息进行分析，不会虚构或推测论文内容，确保判断的准确性和可靠性。你具有记忆功能，能够记住已经处理过的论文（通过标题和URL识别），如果发现是重复论文，会直接跳过处理。"
        else:
            backstory = "你是一位在学术研究领域拥有丰富研究经验的专家。你能够基于论文标题和邮件中的片段信息，准确判断论文的研究方向是否与目标领域相关。你严格基于提供的信息进行分析，不会虚构或推测论文内容，确保判断的准确性和可靠性。你具有记忆功能，能够记住已经处理过的论文（通过标题和URL识别），如果发现是重复论文，会直接跳过处理。"
    else:
        backstory = "你是一位在机器人学、控制理论、遥操作、机器人动力学、力控和机器学习领域拥有丰富研究经验的专家。你能够基于论文标题和邮件中的片段信息，准确判断论文的研究方向是否与目标领域相关。你严格基于提供的信息进行分析，不会虚构或推测论文内容，确保判断的准确性和可靠性。你具有记忆功能，能够记住已经处理过的论文（通过标题和URL识别），如果发现是重复论文，会直接跳过处理。"
    
    return Agent(
        role="论文相关性分析专家",
        goal="基于论文标题和邮件片段信息，准确判断论文是否符合指定的研究方向。同时能够识别重复论文，避免重复处理。",
        backstory=backstory,
        allow_delegation=False,
        verbose=True,
        llm=get_llm(),
        max_iter=2,
        max_execution_time=300
    )


def create_relevance_analysis_task(paper, processed_papers=None, research_keywords=None, expanded_keywords=None):
    """
    创建相关性分析任务
    
    Args:
        paper: 论文信息字典
        processed_papers: 已处理的论文列表，格式为 [{'title': ..., 'link': ...}, ...]
        research_keywords: 研究方向关键词字符串，例如"机器人学、控制理论、遥操作、机器人动力学、力控、机器学习"
        expanded_keywords: 扩写后的研究方向描述（如果提供了，将优先使用此描述）
    """
    paper_title = paper.get('title', '')
    paper_snippet = paper.get('snippet', '')
    paper_link = paper.get('link', '')
    
    # 如果提供了扩写后的关键词描述，优先使用
    if expanded_keywords and expanded_keywords.strip():
        # 使用扩写后的详细描述
        research_directions_section = expanded_keywords
    else:
        # 否则使用原始关键词列表
        if research_keywords:
            # 将关键词字符串按逗号、顿号、分号等分隔符分割
            import re
            keywords_list = re.split(r'[，,、;；\s]+', research_keywords.strip())
            keywords_list = [kw.strip() for kw in keywords_list if kw.strip()]
        else:
            # 默认研究方向
            keywords_list = ['遥操作（Teleoperation）', '力控（Force Control）', '灵巧手（Dexterous Manipulation/Hand）', 
                            '机器人动力学（Robot Dynamics）', '机器学习（Machine Learning）']
        
        # 构建研究方向列表字符串
        research_directions_section = "\n".join([f"- {kw}" for kw in keywords_list])
    
    # 构建已处理论文信息（用于重复检测）
    processed_info = ""
    if processed_papers and len(processed_papers) > 0:
        processed_info = "\n\n## 已处理论文列表（用于重复检测）\n"
        processed_info += "以下论文已经处理过，如果当前论文与其中任何一篇重复（标题相同或URL相同），应直接输出0并说明是重复论文。\n\n"
        processed_info += "**重要提示**：在判断重复时，需要考虑以下情况：\n"
        processed_info += "1. 标题完全相同（忽略大小写）\n"
        processed_info += "2. 标题内容相同但空格不同（例如：'Title A' 和 'TitleA' 应视为相同）\n"
        processed_info += "3. URL完全相同\n"
        processed_info += "4. 标题高度相似（核心词汇相同，仅格式略有不同）\n"
        processed_info += "5. 比较时应该去除空格后对比，或者比较核心词汇是否一致\n\n"
        for i, proc_paper in enumerate(processed_papers[-20:], 1):  # 只显示最近20篇，避免过长
            proc_title = proc_paper.get('title', '')[:80]  # 限制长度
            proc_link = proc_paper.get('link', '')[:80]
            # 同时显示规范化后的标题（去除空格）用于对比
            proc_title_normalized = ''.join(proc_title.split())  # 去除所有空格
            processed_info += f"{i}. 标题：{proc_title}\n"
            processed_info += f"   标题（无空格）：{proc_title_normalized}\n"
            processed_info += f"   链接：{proc_link}\n\n"
        if len(processed_papers) > 20:
            processed_info += f"（共{len(processed_papers)}篇已处理论文，仅显示最近20篇）\n\n"
    
    return Task(
        description=(
            f"## 任务目标\n"
            f"基于论文标题和邮件片段信息，判断论文是否符合以下研究方向：\n\n"
            f"{research_directions_section}\n\n"
            f"## 输入信息\n"
            f"**论文标题**：{paper_title}\n\n"
            f"**论文链接**：{paper_link}\n\n"
            f"**邮件片段信息**：\n```\n{paper_snippet}\n```\n\n"
            f"## 研究方向说明\n"
            f"请根据上述研究方向描述，判断论文是否与这些研究方向相关。\n"
            f"判断时应该：\n"
            f"1. 仔细阅读每个研究方向的核心含义、关键特征和判断标准\n"
            f"2. 基于论文标题和邮件片段信息，判断论文是否涉及这些研究方向\n"
            f"3. 如果论文的研究内容、技术方法、应用场景与研究方向相关，则视为符合\n"
            f"4. 如果论文明显属于其他领域（如纯计算机视觉、自动驾驶、导航等），则视为不符合\n"
            f"5. 参考研究方向描述中的判断标准，进行准确判断\n\n"
            f"{processed_info}"
            f"## 判断标准\n"
            f"### 输出1（符合方向）的条件\n"
            f"- 论文标题或邮件片段信息明确表明论文研究内容与上述任一方向相关\n"
            f"- 论文涉及的技术、方法、应用场景与上述方向的核心特征匹配\n"
            f"- 论文的研究目标、技术路线或应用领域与上述方向高度相关\n"
            f"- **重要**：判断必须基于提供的标题和片段信息，不能虚构或推测论文内容\n"
            f"- **重要**：当前论文不是重复论文（标题和URL与已处理论文列表中的任何一篇都不相同）\n\n"
            f"### 输出0（不符合方向）的条件\n"
            f"- 论文标题和邮件片段信息无法明确判断与上述研究方向相关\n"
            f"- 论文研究内容明显属于其他领域（如纯计算机视觉、自动驾驶、导航等）\n"
            f"- 论文虽然涉及相关技术，但研究重点不在上述研究方向范围内\n"
            f"- 信息不足，无法准确判断论文研究方向\n"
            f"- **重复论文**：当前论文的标题或URL与已处理论文列表中的任何一篇相同或高度相似（视为重复论文，直接输出0）\n"
            f"  - 判断重复时，需要比较标题的核心内容，即使空格不同也应视为相同（例如：'Title A' 和 'TitleA'）\n"
            f"  - 比较时应该去除空格后对比，或者比较核心词汇是否一致\n"
            f"  - 如果标题的核心内容相同，即使格式略有不同（如空格、大小写），也应视为重复论文\n\n"
            f"## 分析要求\n"
            f"1. **严格基于提供信息**：只能基于论文标题和邮件片段信息进行分析，不能虚构、推测或添加未提供的信息\n"
            f"2. **准确理解研究方向**：准确理解五个研究方向的核心特征和关键要素\n"
            f"3. **客观判断**：基于客观证据进行判断，避免主观臆测\n"
            f"4. **明确输出**：明确输出相关性判断、论文名称和网址\n"
            f"5. **标题规范化**：输出论文名称时，确保标题格式规范（单词之间用单个空格分隔，去除多余空格）\n\n"
            f"## 输出格式（严格遵循）\n"
            f"```\n"
            f"相关性判断：1\n"
            f"是否重复：0\n"
            f"论文名称：[论文的完整标题]\n"
            f"论文网址：[论文的完整URL]\n"
            f"判断依据：[简要说明论文与哪个方向相关，以及判断的关键依据]\n"
            f"```\n"
            f"或\n"
            f"```\n"
            f"相关性判断：0\n"
            f"是否重复：1\n"
            f"论文名称：[论文的完整标题]\n"
            f"论文网址：[论文的完整URL]\n"
            f"判断依据：[说明是重复论文，指出与哪篇已处理论文重复]\n"
            f"```\n"
            f"或\n"
            f"```\n"
            f"相关性判断：0\n"
            f"是否重复：0\n"
            f"论文名称：[论文的完整标题]\n"
            f"论文网址：[论文的完整URL]\n"
            f"判断依据：[简要说明为什么不符合上述方向]\n"
            f"```\n\n"
            f"**重要说明**：\n"
            f"- 如果论文与已处理论文列表中的任何一篇重复（标题相同或URL相同），必须设置 `是否重复：1`，并设置 `相关性判断：0`\n"
            f"- 如果论文不重复但不符合研究方向，设置 `是否重复：0`，`相关性判断：0`\n"
            f"- 如果论文不重复且符合研究方向，设置 `是否重复：0`，`相关性判断：1`\n\n"
            f"## 注意事项\n"
            f"- 必须基于提供的标题和片段信息进行判断，不能虚构论文内容\n"
            f"- 如果信息不足，应输出0并说明信息不足\n"
            f"- 判断依据应具体、明确，指出与哪个方向相关以及关键特征\n"
            f"- 论文名称和网址必须准确输出，即使判断为不相关也要输出\n"
            f"- 避免过度解读或推测，确保判断的准确性"
        ),
        agent=create_relevance_analyzer_agent(expanded_keywords=expanded_keywords),
        expected_output="首先输出相关性判断（1表示符合方向，0表示不符合），然后输出论文名称、论文网址和判断依据"
    )

