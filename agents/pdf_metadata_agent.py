#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF元数据提取Agent - 从PDF提取标题、摘要、作者等信息
"""

from crewai import Agent, Task, Crew
from config import llm
import logging


def create_pdf_metadata_extractor_agent():
    """创建PDF元数据提取 Agent"""
    return Agent(
        role="论文元数据提取专家",
        goal="从PDF文件的前几页文本中准确提取论文的元数据信息，包括标题、摘要、作者、关键词等。",
        backstory=(
            "你是一位专业的学术论文元数据提取专家。你擅长从PDF文档的前几页中识别和提取论文的关键信息。"
            "你了解学术论文的常见格式：\n"
            "1. **标题**：通常出现在第一页的顶部，可能是加粗、大字体或特殊格式\n"
            "2. **作者**：通常在标题下方，可能包含多个作者，格式如\"Author1, Author2\"或\"Author1 and Author2\"\n"
            "3. **摘要**：通常在\"Abstract\"或\"摘要\"标记后，是一段描述论文主要内容的文字\n"
            "4. **关键词**：通常在摘要后，标记为\"Keywords\"或\"关键词\"\n\n"
            "你能够识别各种格式的标记，如'Title:'、'题目：'、'Abstract'、'摘要'、'Author:'、'作者：'等，"
            "也能识别没有明确标记但格式明显的元数据。你只提取现有内容，不生成或创造信息。"
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=5,
        max_execution_time=120
    )


def create_pdf_metadata_extraction_task(pdf_text_preview: str) -> Task:
    """创建PDF元数据提取任务"""
    return Task(
        description=(
            f"## 任务目标\n"
            f"从PDF文件的前几页文本中准确提取论文的元数据信息。\n\n"
            f"## PDF文本预览（前5000字符）\n"
            f"{pdf_text_preview}\n\n"
            f"## 需要提取的元数据\n"
            f"1. **标题（title）**：论文的完整标题\n"
            f"2. **摘要（abstract）**：论文的摘要内容\n"
            f"3. **作者（authors）**：所有作者的姓名，用逗号分隔\n"
            f"4. **关键词（keywords）**：论文的关键词，用逗号分隔（可选）\n\n"
            f"## 执行步骤\n"
            f"### 步骤1：识别标题\n"
            f"- 查找PDF开头最显眼的文本（通常是第一页的前几行）\n"
            f"- 查找标题标记（如\"Title:\"、\"题目：\"等）\n"
            f"- 识别格式明显的标题（大字体、加粗、居中显示等）\n"
            f"- 提取完整的标题文本，去除多余的空白字符和换行\n\n"
            f"### 步骤2：识别作者\n"
            f"- 查找标题下方的作者信息\n"
            f"- 查找作者标记（如\"Author:\"、\"作者：\"等）\n"
            f"- 提取所有作者的姓名，用逗号分隔\n"
            f"- 如果作者信息跨多行，合并为一行\n\n"
            f"### 步骤3：识别摘要\n"
            f"- 查找\"Abstract\"或\"摘要\"标记后的内容\n"
            f"- 提取摘要的完整文本\n"
            f"- 去除多余的空白字符和换行\n"
            f"- 如果摘要与引言融合，只提取明确的摘要部分\n\n"
            f"### 步骤4：识别关键词（可选）\n"
            f"- 查找\"Keywords\"或\"关键词\"标记后的内容\n"
            f"- 提取关键词，用逗号分隔\n\n"
            f"## 输出要求\n"
            f"请以JSON格式输出提取的元数据，格式如下：\n"
            f"```json\n"
            f"{{\n"
            f"  \"title\": \"论文标题\",\n"
            f"  \"authors\": \"作者1, 作者2, 作者3\",\n"
            f"  \"abstract\": \"摘要内容\",\n"
            f"  \"keywords\": \"关键词1, 关键词2, 关键词3\"\n"
            f"}}\n"
            f"```\n"
            f"- 如果某个字段无法提取，使用空字符串\"\"\n"
            f"- 标题和摘要必须提取，如果无法提取则使用\"未知标题\"或\"无摘要\"\n"
            f"- 只输出JSON，不要添加任何其他说明文字"
        ),
        agent=create_pdf_metadata_extractor_agent(),
        expected_output="JSON格式的元数据，包含title、authors、abstract、keywords字段"
    )


def extract_pdf_metadata(pdf_path: str) -> dict:
    """
    从PDF文件提取元数据（标题、摘要、作者等）
    
    Args:
        pdf_path: PDF文件路径
    
    Returns:
        包含title、authors、abstract、keywords的字典，如果提取失败则返回None
    """
    try:
        from utils.vector_db import extract_text_from_pdf
        import json
        import re
        
        # 提取前5页文本（通常元数据在前5页）
        text_preview = extract_text_from_pdf(pdf_path, max_pages=5)
        
        # 限制预览文本长度（避免超出模型上下文）
        preview_length = 5000
        if len(text_preview) > preview_length:
            text_preview = text_preview[:preview_length] + "..."
        
        # 使用agent提取元数据
        agent = create_pdf_metadata_extractor_agent()
        task = create_pdf_metadata_extraction_task(text_preview)
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False  # 不显示详细日志
        )
        
        result = crew.kickoff()
        
        # 提取JSON文本
        if hasattr(result, 'raw'):
            result_text = result.raw.strip()
        elif isinstance(result, str):
            result_text = result.strip()
        else:
            result_text = str(result).strip()
        
        # 尝试从结果中提取JSON
        # 查找JSON代码块
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 查找直接的JSON对象
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = result_text
        
        # 解析JSON
        metadata = json.loads(json_str)
        
        # 验证必需字段
        if 'title' not in metadata or not metadata['title']:
            metadata['title'] = '未知标题'
        if 'abstract' not in metadata:
            metadata['abstract'] = ''
        if 'authors' not in metadata:
            metadata['authors'] = ''
        if 'keywords' not in metadata:
            metadata['keywords'] = ''
        
        # 清理字段
        metadata['title'] = metadata['title'].strip()
        metadata['abstract'] = metadata['abstract'].strip()
        metadata['authors'] = metadata['authors'].strip()
        metadata['keywords'] = metadata['keywords'].strip()
        
        logging.info(f"从PDF提取元数据成功: 标题={metadata['title'][:50]}...")
        return metadata
        
    except json.JSONDecodeError as e:
        result_text_str = result_text if 'result_text' in locals() else str(result)
        logging.warning(f"解析PDF元数据JSON失败: {pdf_path}, 错误: {str(e)}")
        logging.warning(f"原始结果: {result_text_str[:500]}")
        return None
    except Exception as e:
        logging.warning(f"提取PDF元数据失败: {pdf_path}, 错误: {str(e)}")
        return None

