#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个人知识库Agent - 基于RAG技术的智能问答系统

这个Agent负责根据用户的提问，查询现有资源（向量数据库中的PDF论文）来回答问题。
它能够：
1. 查询论文库中的论文列表和元数据
2. 使用RAG技术检索相关论文内容
3. 获取论文详细信息
4. 总结论文内容
5. 综合多个信息源提供准确、全面的回答
"""

from crewai import Agent, Task
from config import llm
from typing import Optional
from agents.base import (
    rag_paper_query_tool,
    get_paper_list_tool,
    get_paper_details_tool,
    summarize_paper_tool,
    search_papers_by_author_tool,
    search_papers_by_keywords_tool,
    search_papers_by_conditions_tool,
    get_paper_full_text_tool,
    read_pdf_file_tool
)


def create_knowledge_base_agent():
    """创建个人知识库 Agent"""
    return Agent(
        role="个人知识库助手",
        goal=(
            "根据用户的提问，智能查询知识库中的资源（PDF论文）来回答问题。"
            "能够理解用户意图，选择合适的工具查询相关信息，并综合多个信息源提供准确、全面、有条理的回答。"
        ),
        backstory=(
            "你是一位专业的个人知识库助手，擅长从结构化的知识库中检索和整合信息。"
            "你的知识库包含用户上传的学术论文PDF，这些论文已经经过处理并存储在向量数据库中。"
            "\n\n"
            "你的核心能力包括：\n"
            "1. **理解用户意图**：准确理解用户问题的核心，判断需要查询什么类型的信息\n"
            "2. **智能工具选择**：根据问题类型选择合适的工具：\n"
            "   - 询问论文数量或列表 → 使用'获取论文列表工具'\n"
            "   - 按作者查询论文 → 使用'按作者查询论文工具'\n"
            "   - 按关键词查询论文 → 使用'按关键词查询论文工具'\n"
            "   - 按年份、期刊、来源等条件查询 → 使用'按条件查询论文工具'\n"
            "   - 询问论文具体内容 → 使用'RAG论文查询工具'检索相关内容\n"
            "   - 需要了解某篇论文的详细信息 → 使用'获取论文详细信息工具'或'获取论文全文工具'\n"
            "   - 需要快速了解论文核心内容 → 使用'总结论文内容工具'\n"
            "   - 需要读取PDF文件 → 使用'读取PDF文件工具'\n"
            "3. **信息整合**：能够综合多个工具的结果，提供全面、准确的回答\n"
            "4. **上下文理解**：理解论文的学术内容，能够解释技术细节、方法、实验结果等\n"
            "5. **清晰表达**：使用清晰、易懂的语言回答问题，避免过于技术化的术语（除非用户明确要求）\n"
            "\n"
            "回答原则：\n"
            "- 基于检索到的实际内容回答，不编造信息\n"
            "- 如果信息不足，如实说明，不要猜测\n"
            "- 引用论文内容时，说明来源（论文标题）\n"
            "- 对于复杂问题，分步骤、分方面回答\n"
            "- 保持回答的准确性和专业性"
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=8,  # 允许更多迭代，以便使用多个工具
        max_execution_time=600,
        tools=[
            rag_paper_query_tool,              # RAG查询工具：检索相关论文内容
            get_paper_list_tool,                # 获取论文列表：回答论文数量、列表等问题
            get_paper_details_tool,            # 获取论文详情：深入了解某篇论文
            summarize_paper_tool,              # 总结论文内容：快速了解论文核心
            search_papers_by_author_tool,      # 按作者查询论文
            search_papers_by_keywords_tool,    # 按关键词查询论文
            search_papers_by_conditions_tool,  # 按条件查询论文（年份、期刊、来源等）
            get_paper_full_text_tool,          # 获取论文全文
            read_pdf_file_tool                 # 读取PDF文件工具
        ]
    )


def create_knowledge_base_task(user_question: str, chat_history: Optional[str] = None) -> Task:
    """创建个人知识库任务"""
    # 构建任务描述
    task_description = (
        f"## 任务目标\n"
        f"根据用户的提问，查询个人知识库中的资源（PDF论文），提供准确、全面、有条理的回答。\n\n"
    )
    
    # 如果有历史对话，添加到任务描述中
    if chat_history:
        task_description += f"{chat_history}\n\n"
    
    task_description += (
        f"## 用户问题\n"
        f"{user_question}\n\n"
        f"## 执行步骤\n"
        f"### 步骤1：理解用户问题\n"
        f"- 仔细分析用户的问题，确定问题的类型和需要的信息\n"
        f"- 识别关键概念和关键词\n"
        f"- 参考历史对话（如果有）来理解上下文和用户意图\n"
        f"- 判断是否需要查询论文列表、具体内容、论文详情或总结\n\n"
        f"### 步骤2：选择合适的工具\n"
        f"根据问题类型选择工具：\n"
        f"- **论文库元数据问题**（如\"有多少篇论文\"、\"列出所有论文\"）：使用'获取论文列表工具'\n"
        f"- **按作者查询**（如\"查找某位作者的论文\"）：使用'按作者查询论文工具'\n"
        f"- **按关键词查询**（如\"查找包含某个关键词的论文\"）：使用'按关键词查询论文工具'\n"
        f"- **按条件查询**（如\"查找2024年的论文\"、\"查找某期刊的论文\"）：使用'按条件查询论文工具'\n"
        f"- **论文内容查询**（如\"某篇论文讲了什么\"、\"关于某个主题的论文\"）：使用'RAG论文查询工具'\n"
        f"- **论文详细信息**（如\"某篇论文的完整内容\"）：使用'获取论文详细信息工具'或'获取论文全文工具'\n"
        f"- **论文快速总结**（如\"总结某篇论文\"）：使用'总结论文内容工具'\n"
        f"- **读取PDF文件**（如\"读取某个PDF文件\"）：使用'读取PDF文件工具'\n"
        f"- **复杂问题**：可能需要使用多个工具，先获取列表，再查询相关内容\n\n"
        f"### 步骤3：执行查询\n"
        f"- 使用选定的工具查询相关信息\n"
        f"- 如果第一次查询结果不足，可以尝试：\n"
        f"  - 调整查询关键词\n"
        f"  - 使用不同的工具\n"
        f"  - 查询更多结果（增加limit或n_results参数）\n"
        f"- 如果问题涉及多篇论文，可以分别查询每篇论文\n\n"
        f"### 步骤4：整合和分析信息\n"
        f"- 综合多个工具的结果\n"
        f"- 理解检索到的论文内容\n"
        f"- 识别关键信息和要点\n"
        f"- 组织回答的结构\n\n"
        f"### 步骤5：生成回答\n"
        f"- 基于检索到的实际内容回答，不编造信息\n"
        f"- 如果信息不足，如实说明\n"
        f"- 引用论文内容时，说明来源（论文标题）\n"
        f"- 使用清晰、有条理的语言\n"
        f"- 对于复杂问题，分步骤、分方面回答\n"
        f"- 如果用户的问题与历史对话相关，可以参考历史对话来提供更准确的回答\n\n"
        f"## 输出要求\n"
        f"- 回答要准确、全面、有条理\n"
        f"- 基于实际检索到的内容，不编造信息\n"
        f"- 引用论文时说明来源\n"
        f"- 使用清晰的语言，避免过于技术化的术语（除非用户明确要求）\n"
        f"- 如果问题无法回答，要说明原因和已尝试的方法"
    )
    
    return Task(
        description=task_description,
        agent=create_knowledge_base_agent(),
        expected_output="基于知识库检索的准确、全面、有条理的回答"
    )

