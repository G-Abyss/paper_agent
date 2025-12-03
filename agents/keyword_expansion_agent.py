#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键词扩写Agent - 根据用户提供的关键词，生成详细的研究方向描述
"""

from crewai import Agent, Task, Crew
from config import llm
from callbacks.crewai_callbacks import capture_crewai_output
from callbacks.frontend_callbacks import send_agent_status


def create_keyword_expansion_agent():
    """创建关键词扩写 Agent"""
    return Agent(
        role="研究方向关键词扩写专家",
        goal="根据用户提供的研究方向关键词，生成详细、专业的研究方向描述，包括每个方向的核心含义、关键特征、相关技术、应用场景等，以便其他Agent能够准确理解研究方向并做出判断。",
        backstory="你是一位在学术研究领域拥有深厚经验的专家，擅长理解和扩写研究方向关键词。你能够根据简短的关键词，深入分析其学术含义，识别相关的技术领域、研究方法、应用场景，并生成结构化的研究方向描述。你的扩写结果将用于指导论文相关性分析，因此需要准确、全面、专业。",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=3,
        max_execution_time=300
    )


def create_keyword_expansion_task(user_keywords):
    """
    创建关键词扩写任务
    
    Args:
        user_keywords: 用户提供的关键词字符串，例如"机器人学、控制理论、遥操作、机器人动力学、力控、机器学习"
    
    Returns:
        Task: 关键词扩写任务
    """
    return Task(
        description=(
            f"## 任务目标\n"
            f"根据用户提供的研究方向关键词，生成详细、专业的研究方向描述。\n\n"
            f"## 用户提供的关键词\n"
            f"{user_keywords}\n\n"
            f"## 扩写要求\n"
            f"1. **解析关键词**：将用户提供的关键词字符串解析为独立的研究方向列表\n"
            f"2. **深入扩写**：为每个研究方向生成详细描述，包括：\n"
            f"   - 研究方向的核心含义和定义\n"
            f"   - 关键特征和技术要素\n"
            f"   - 相关技术、方法和应用场景\n"
            f"   - 与其他研究方向的关系\n"
            f"   - 判断论文是否属于该方向的标准\n"
            f"3. **专业准确**：确保扩写内容准确反映该研究方向的学术内涵，使用专业术语\n"
            f"4. **结构化输出**：按照指定格式输出，便于其他Agent使用\n\n"
            f"## 输出格式（严格遵循）\n"
            f"请按照以下格式输出扩写结果：\n\n"
            f"```\n"
            f"## 研究方向列表\n"
            f"[列出所有研究方向，每行一个，格式：- 研究方向名称]\n\n"
            f"## 研究方向详细描述\n"
            f"### 方向1：[研究方向名称]\n"
            f"**核心含义**：[该研究方向的核心定义和学术含义]\n\n"
            f"**关键特征**：\n"
            f"- [特征1]\n"
            f"- [特征2]\n"
            f"- [特征3]\n\n"
            f"**相关技术和方法**：\n"
            f"- [技术/方法1]\n"
            f"- [技术/方法2]\n"
            f"- [技术/方法3]\n\n"
            f"**应用场景**：\n"
            f"- [应用场景1]\n"
            f"- [应用场景2]\n\n"
            f"**判断标准**：论文符合此方向的条件包括：\n"
            f"- [判断标准1]\n"
            f"- [判断标准2]\n\n"
            f"---\n\n"
            f"### 方向2：[研究方向名称]\n"
            f"[按照相同格式继续描述其他方向]\n"
            f"```\n\n"
            f"## 注意事项\n"
            f"- 必须基于用户提供的关键词进行扩写，不能添加用户未提及的方向\n"
            f"- 扩写内容应该专业、准确，反映该研究方向的真实学术内涵\n"
            f"- 每个方向的描述应该详细但不过于冗长（每个方向约200-400字）\n"
            f"- 确保判断标准清晰明确，便于后续的论文相关性分析\n"
            f"- 如果关键词中包含英文或缩写，应同时提供中文解释\n"
        ),
        agent=create_keyword_expansion_agent(),
        expected_output="结构化的研究方向描述，包括研究方向列表和每个方向的详细描述（核心含义、关键特征、相关技术、应用场景、判断标准）"
    )


def expand_keywords(user_keywords, on_log=None):
    """
    执行关键词扩写任务
    
    Args:
        user_keywords: 用户提供的关键词字符串
        on_log: 日志回调函数（可选）
    
    Returns:
        str: 扩写后的研究方向描述，如果失败则返回None
    """
    if not user_keywords or not user_keywords.strip():
        if on_log:
            on_log('warning', '关键词为空，跳过扩写')
        return None
    
    try:
        # 创建任务
        expansion_task = create_keyword_expansion_task(user_keywords)
        
        # 发送agent工作开始状态
        send_agent_status("关键词扩写专家", "start", task=expansion_task)
        
        # 创建Crew并执行任务
        with capture_crewai_output():
            crew = Crew(
                agents=[create_keyword_expansion_agent()],
                tasks=[expansion_task],
                verbose=True
            )
            result = crew.kickoff()
        
        # 发送agent工作结束状态
        expanded_content = str(result) if result else None
        send_agent_status("关键词扩写专家", "end", task=expansion_task, result=result)
        
        if on_log:
            on_log('info', '关键词扩写完成')
        
        return expanded_content
        
    except Exception as e:
        if on_log:
            on_log('error', f'关键词扩写失败: {str(e)}')
        # 如果任务已创建，传递任务；否则只传递错误信息
        try:
            expansion_task = create_keyword_expansion_task(user_keywords)
            send_agent_status("关键词扩写专家", "end", task=expansion_task, result=str(e))
        except:
            # 如果创建任务也失败，只传递错误信息
            send_agent_status("关键词扩写专家", "end", result=str(e))
        return None

