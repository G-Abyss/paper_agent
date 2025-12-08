#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评审Agent
"""

import json
import re
from crewai import Agent, Task
from config import llm


def create_reviewer_agent(expanded_keywords=None, llm=None):
    """
    创建专业评审 Agent
    
    Args:
        expanded_keywords: 扩写后的研究方向描述（可选），用于定制agent的专业领域背景
        llm: LLM实例（可选），如果不提供则使用默认的llm
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
                backstory = f"你是一位在{research_fields}等领域拥有丰富研究经验的评审专家。你能够从创新性、技术深度、相关性、实用性等多个维度对论文进行客观、专业的评价。你总是简洁明了地输出结果，不会重复说明。"
            else:
                backstory = "你是一位在学术研究领域拥有丰富研究经验的评审专家。你能够从创新性、技术深度、相关性、实用性等多个维度对论文进行客观、专业的评价。你总是简洁明了地输出结果，不会重复说明。"
        else:
            backstory = "你是一位在学术研究领域拥有丰富研究经验的评审专家。你能够从创新性、技术深度、相关性、实用性等多个维度对论文进行客观、专业的评价。你总是简洁明了地输出结果，不会重复说明。"
    else:
        backstory = "你是一位在机器人学、控制理论、遥操作、机器人动力学和力控领域拥有丰富研究经验的评审专家。你能够从创新性、技术深度、相关性、实用性等多个维度对论文进行客观、专业的评价。你总是简洁明了地输出结果，不会重复说明。"
    
    # 如果没有提供llm，使用默认的llm
    from config import llm as default_llm
    agent_llm = llm if llm is not None else default_llm
    
    return Agent(
        role="专业评审专家",
        goal="对论文进行专业评审，生成结构化总结并给出简洁的4分制评分（只输出一次）",
        backstory=backstory,
        allow_delegation=False,
        verbose=True,
        llm=agent_llm,
        max_iter=2,  # 减少迭代次数，避免重复
        max_execution_time=300
    )


def create_review_task(paper, translated_content, original_english_abstract="", expanded_keywords=None):
    """
    创建评审任务（支持中英文双语评审）
    
    Args:
        paper: 论文信息字典
        translated_content: 翻译后的内容
        original_english_abstract: 原始英文摘要（可选）
        expanded_keywords: 扩写后的研究方向描述（可选），用于定制专业领域
    """
    # 根据扩写关键词生成相关性描述
    if expanded_keywords and expanded_keywords.strip():
        import re
        directions_match = re.search(r'## 研究方向列表\s*\n(.*?)(?=\n##|$)', expanded_keywords, re.DOTALL)
        if directions_match:
            directions_text = directions_match.group(1).strip()
            directions = re.findall(r'-\s*(.+?)(?:\n|$)', directions_text)
            if directions:
                research_fields = '、'.join([d.strip() for d in directions[:5]])
                relevance_description = f"与{research_fields}等领域的关联程度"
            else:
                relevance_description = "与目标研究领域的关联程度"
        else:
            relevance_description = "与目标研究领域的关联程度"
    else:
        relevance_description = "与遥操作、机器人动力学、力控、机器人控制等领域的关联程度"
    
    return Task(
        description=(
            f"## 任务目标\n"
            f"对论文进行专业评审，生成结构化总结并给出4个维度的分项评分（每个维度0.0-1.0分）。评审时可参考中英文双语内容，以提高评审准确性。\n\n"
            f"## 输入信息\n"
            f"**论文标题（英文）**：{paper['title']}\n\n"
            f"**论文内容（已翻译为中文）**：\n```\n{translated_content}\n```\n\n"
            f"{f'**论文摘要（英文原文）**：\n```\n{original_english_abstract}\n```\n\n' if original_english_abstract else ''}"
            f"## 评审维度\n"
            f"### 维度1：创新性（0.0-1.0分）\n"
            f"评估论文的创新程度：\n"
            f"- 是否提出了新的理论、方法或技术\n"
            f"- 是否解决了现有方法无法解决的问题\n"
            f"- 创新点的显著性和重要性\n\n"
            f"### 维度2：技术深度（0.0-1.0分）\n"
            f"评估论文的技术深度：\n"
            f"- 技术方法的复杂度和先进性\n"
            f"- 理论分析的深度和严谨性\n"
            f"- 实验验证的充分性和可靠性\n\n"
            f"### 维度3：相关性（0.0-1.0分）\n"
            f"评估论文与目标领域的相关性：\n"
            f"- {relevance_description}\n"
            f"- 对目标领域研究的贡献和影响\n"
            f"- 研究主题的契合度\n\n"
            f"### 维度4：实用性（0.0-1.0分）\n"
            f"评估论文的实用价值：\n"
            f"- 方法的可应用性和可实施性\n"
            f"- 潜在的应用场景和价值\n"
            f"- 对实际问题的解决能力\n\n"
            f"## 输出格式（严格遵循）\n"
            f"```\n"
            f"**核心贡献**：[1-2句话说明主要创新点和贡献]\n\n"
            f"**技术方法**：[简述主要技术路线和方法]\n\n"
            f"**相关性分析**：[详细说明与目标研究领域的关系]\n\n"
            f"**技术价值**：[评估该论文的技术价值和潜在应用]\n\n"
            f"**值得关注的原因**：[为什么这篇论文重要，有哪些亮点]\n\n"
            f"**评分详情**：\n"
            f"```json\n"
            f'{{"创新性": 0.0-1.0, "技术深度": 0.0-1.0, "相关性": 0.0-1.0, "实用性": 0.0-1.0, "评分理由": "简要说明评分依据"}}\n'
            f"```\n"
            f"```\n\n"
            f"## 重要约束\n"
            f"1. **评分详情格式**：必须使用Markdown代码块格式（```json ... ```），仅输出一次\n"
            f"2. **评分范围**：每个维度评分范围为0.0-1.0分，使用浮点数（如0.85）\n"
            f"3. **禁止输出总分**：只输出4个维度的分项得分，不要计算或输出总分\n"
            f"4. **评分理由**：必须在JSON中包含'评分理由'字段，简要说明评分依据\n"
            f"5. **禁止重复**：不要重复说明评分规则，不要多次输出评分详情\n"
            f"6. **直接输出**：直接给出评审结果，不要添加额外的说明或解释\n\n"
            f"## 评审建议\n"
            f"- 如果提供了英文原文，建议同时参考中英文双语内容进行评审，以提高评审准确性\n"
            f"- 英文原文可以帮助理解专业术语的准确含义和技术细节\n"
            f"- 中文翻译可以帮助快速理解论文的整体内容\n"
            f"- 当翻译可能存在歧义时，优先参考英文原文\n\n"
            f"## 注意事项\n"
            f"- 评分应客观、公正，基于论文的实际内容进行评估\n"
            f"- 各维度的评分应相互独立，避免相互影响\n"
            f"- 评分理由应简洁明了，说明关键评分依据"
        ),
        agent=create_reviewer_agent(expanded_keywords=expanded_keywords),
        expected_output=(
            "评审报告包含：核心贡献、技术方法、相关性分析、技术价值、值得关注的原因，"
            "以及一个Markdown代码块格式的JSON评分详情（包含4个维度的分项得分和评分理由，不包含总分，仅输出一次）。"
        )
    )


def extract_score_from_review(review_text):
    """
    从评审文本中提取评分信息
    只提取分项得分，总分通过计算得出（避免agent幻觉）
    """
    # 定义评分维度（4个维度，总分4分）
    dimensions = ['创新性', '技术深度', '相关性', '实用性']
    
    score_data = {
        '创新性': 0.0,
        '技术深度': 0.0,
        '相关性': 0.0,
        '实用性': 0.0,
        '总分': 0.0,  # 将通过计算得出（4个维度之和，满分4分）
        '评分理由': ''
    }
    
    # 方法1: 尝试提取完整的JSON对象（支持多行和嵌套）
    # 查找JSON对象的开始和结束
    json_start = review_text.find('{')
    json_end = review_text.rfind('}')
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            json_str = review_text[json_start:json_end+1]
            # 尝试解析JSON
            parsed = json.loads(json_str)
            # 只提取分项得分和评分理由，忽略总分（即使agent输出了也忽略）
            for dim in dimensions:
                if dim in parsed and isinstance(parsed[dim], (int, float)):
                    score = float(parsed[dim])
                    # 验证分数范围（0.0-1.0）
                    if 0.0 <= score <= 1.0:
                        score_data[dim] = score
            # 提取评分理由
            if '评分理由' in parsed and isinstance(parsed['评分理由'], str):
                score_data['评分理由'] = parsed['评分理由']
            
            # 计算总分（各维度之和）
            total = sum([score_data[dim] for dim in dimensions])
            score_data['总分'] = total
            return score_data
        except (json.JSONDecodeError, ValueError):
            pass
    
    # 方法2: 尝试提取JSON格式的评分（更宽松的匹配）
    json_patterns = [
        r'\{[^{}]*"创新性"[^{}]*"技术深度"[^{}]*\}',
        r'\{[^{}]*"创新性"[^{}]*\}',
    ]
    for pattern in json_patterns:
        json_match = re.search(pattern, review_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                # 只提取分项得分，忽略总分
                for dim in dimensions:
                    if dim in parsed and isinstance(parsed[dim], (int, float)):
                        score = float(parsed[dim])
                        if 0.0 <= score <= 1.0:
                            score_data[dim] = score
                if '评分理由' in parsed and isinstance(parsed['评分理由'], str):
                    score_data['评分理由'] = parsed['评分理由']
                
                # 计算总分
                total = sum([score_data[dim] for dim in dimensions])
                if total > 0:
                    score_data['总分'] = total
                    return score_data
            except (json.JSONDecodeError, ValueError):
                continue
    
    # 方法3: 如果JSON提取失败，尝试从文本中提取各个维度的分数
    for dim in dimensions:
        patterns = [
            rf'{dim}[：:]\s*([0-9.]+)',
            rf'"{dim}"[：:]\s*([0-9.]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, review_text)
            if match:
                try:
                    score = float(match.group(1))
                    # 验证分数范围（0.0-1.0）
                    if 0.0 <= score <= 1.0:
                        score_data[dim] = score
                        break
                except (ValueError, IndexError):
                    continue
    
    # 计算总分（各维度之和）
    total = sum([score_data[dim] for dim in dimensions])
    score_data['总分'] = total
    
    return score_data

