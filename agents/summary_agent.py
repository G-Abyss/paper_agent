#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总Agent - 用于汇总多个模型的评审结果
"""

import json
import re
import logging
from crewai import Agent, Task
from agents.base import get_llm


def create_summary_agent(expanded_keywords=None):
    """
    创建汇总分析 Agent
    
    Args:
        expanded_keywords: 扩写后的研究方向描述（可选），用于定制agent的专业领域背景
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
                backstory = f"你是一位在{research_fields}等领域拥有丰富研究经验的资深评审专家。你擅长综合分析多个评审专家的意见，识别共识和分歧，并基于多角度分析得出最客观、最合理的综合评审结论。你能够评估不同评审结果的一致性和可信度。"
            else:
                backstory = "你是一位在学术研究领域拥有丰富研究经验的资深评审专家。你擅长综合分析多个评审专家的意见，识别共识和分歧，并基于多角度分析得出最客观、最合理的综合评审结论。你能够评估不同评审结果的一致性和可信度。"
        else:
            backstory = "你是一位在学术研究领域拥有丰富研究经验的资深评审专家。你擅长综合分析多个评审专家的意见，识别共识和分歧，并基于多角度分析得出最客观、最合理的综合评审结论。你能够评估不同评审结果的一致性和可信度。"
    else:
        backstory = "你是一位在机器人学、控制理论、遥操作、机器人动力学和力控领域拥有丰富研究经验的资深评审专家。你擅长综合分析多个评审专家的意见，识别共识和分歧，并基于多角度分析得出最客观、最合理的综合评审结论。你能够评估不同评审结果的一致性和可信度。"
    
    return Agent(
        role="综合评审汇总专家",
        goal="综合分析多个模型的评审结果，识别共识和分歧，生成最客观、最合理的综合评审报告和最终评分",
        backstory=backstory,
        allow_delegation=False,
        verbose=True,
        llm=get_llm(),
        max_iter=3,
        max_execution_time=300
    )


def create_summary_task(paper, multi_model_results, expanded_keywords=None):
    """
    创建汇总分析任务
    
    Args:
        paper: 论文信息字典
        multi_model_results: 多模型评审结果字典，格式为 {model_id: {review, score, score_details, ...}, ...}
        expanded_keywords: 扩写后的研究方向描述（可选），用于定制专业领域
    """
    # 构建模型结果摘要
    model_results_summary = []
    for model_id, result in multi_model_results.items():
        model_name = result.get('model_name', model_id)
        review = result.get('review', '')
        score = result.get('score', 0.0)
        score_details = result.get('score_details', {})
        
        model_summary = f"**模型 {model_name}**：\n"
        model_summary += f"- 评审报告：{review[:500]}...\n" if len(review) > 500 else f"- 评审报告：{review}\n"
        model_summary += f"- 总分：{score:.2f}/4.0\n"
        if score_details:
            model_summary += f"- 分项评分：\n"
            for dim, dim_score in score_details.items():
                if dim != '总分' and dim != '评分理由':
                    model_summary += f"  - {dim}：{dim_score:.2f}/1.0\n"
        model_results_summary.append(model_summary)
    
    model_results_text = "\n\n".join(model_results_summary)
    
    # 计算评分统计信息
    scores = [result.get('score', 0.0) for result in multi_model_results.values()]
    if scores:
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        score_range = f"{min_score:.2f} - {max_score:.2f}"
    else:
        avg_score = 0.0
        score_range = "N/A"
    
    # 计算各维度评分统计
    dimensions = ['创新性', '技术深度', '相关性', '实用性']
    dimension_stats = {}
    for dim in dimensions:
        dim_scores = []
        for result in multi_model_results.values():
            score_details = result.get('score_details', {})
            if dim in score_details:
                dim_scores.append(score_details[dim])
        if dim_scores:
            dimension_stats[dim] = {
                'avg': sum(dim_scores) / len(dim_scores),
                'min': min(dim_scores),
                'max': max(dim_scores),
                'scores': dim_scores
            }
    
    dimension_stats_text = ""
    for dim, stats in dimension_stats.items():
        dimension_stats_text += f"- {dim}：平均 {stats['avg']:.2f}，范围 {stats['min']:.2f} - {stats['max']:.2f}\n"
    
    return Task(
        description=(
            f"## 任务目标\n"
            f"综合分析多个模型的评审结果，识别共识和分歧，生成最客观、最合理的综合评审报告和最终评分。\n\n"
            f"## 输入信息\n"
            f"**论文标题**：{paper.get('title', 'Unknown')}\n\n"
            f"**多模型评审结果**（共 {len(multi_model_results)} 个模型）：\n\n"
            f"{model_results_text}\n\n"
            f"**评分统计**：\n"
            f"- 平均总分：{avg_score:.2f}/4.0\n"
            f"- 评分范围：{score_range}\n"
            f"- 各维度统计：\n"
            f"{dimension_stats_text}\n"
            f"## 分析要求\n"
            f"### 1. 共识识别\n"
            f"- 识别所有模型都认同的观点和评价\n"
            f"- 识别评分相近的维度（差异小于0.2分）\n"
            f"- 总结共同认可的核心贡献和技术方法\n\n"
            f"### 2. 分歧分析\n"
            f"- 识别各模型存在明显分歧的观点（评分差异大于0.3分）\n"
            f"- 分析分歧产生的原因（可能是不同模型关注点不同）\n"
            f"- 评估分歧对最终结论的影响\n\n"
            f"### 3. 综合评估\n"
            f"- 基于共识和分歧，生成综合评审报告\n"
            f"- 计算最终评分（考虑模型一致性和评分分布）\n"
            f"- 评估结果的可信度（模型一致性越高，可信度越高）\n\n"
            f"## 输出格式（严格遵循）\n"
            f"```\n"
            f"**共识要点**：\n"
            f"[列出所有模型都认同的核心观点，包括：核心贡献、技术方法、相关性分析等]\n\n"
            f"**分歧分析**：\n"
            f"[列出各模型存在分歧的观点，并分析分歧原因]\n\n"
            f"**综合评审报告**：\n"
            f"[基于共识和分歧，生成综合评审报告，包括：核心贡献、技术方法、相关性分析、技术价值、值得关注的原因]\n\n"
            f"**最终评分详情**：\n"
            f"```json\n"
            f'{{"创新性": 0.0-1.0, "技术深度": 0.0-1.0, "相关性": 0.0-1.0, "实用性": 0.0-1.0, "评分依据": "基于多模型共识和分歧分析的综合评分", "模型一致性": 0.0-1.0, "结果可信度": 0.0-1.0}}\n'
            f"```\n"
            f"```\n\n"
            f"## 评分计算规则\n"
            f"1. **最终评分**：基于各模型评分的加权平均，权重考虑模型一致性\n"
            f"2. **模型一致性**：计算各模型评分之间的标准差，标准差越小，一致性越高（0-1，1表示完全一致）\n"
            f"3. **结果可信度**：综合考虑模型一致性和评分分布，一致性越高、评分越集中，可信度越高（0-1）\n"
            f"4. **评分范围**：每个维度评分范围为0.0-1.0分，使用浮点数\n\n"
            f"## 重要约束\n"
            f"1. **评分详情格式**：必须使用Markdown代码块格式（```json ... ```），仅输出一次\n"
            f"2. **禁止输出总分**：只输出4个维度的分项得分，不要计算或输出总分\n"
            f"3. **评分依据**：必须在JSON中包含'评分依据'字段，说明综合评分的依据\n"
            f"4. **模型一致性**：必须在JSON中包含'模型一致性'字段（0-1，表示各模型评分的一致性）\n"
            f"5. **结果可信度**：必须在JSON中包含'结果可信度'字段（0-1，表示最终结果的可信程度）\n"
            f"6. **禁止重复**：不要重复说明评分规则，不要多次输出评分详情\n"
            f"7. **直接输出**：直接给出汇总结果，不要添加额外的说明或解释\n\n"
            f"## 注意事项\n"
            f"- 综合评审报告应客观、公正，基于多模型的实际评审结果\n"
            f"- 当模型之间存在分歧时，应分析分歧原因，而不是简单地取平均值\n"
            f"- 最终评分应反映多模型的综合判断，而不是单一模型的意见\n"
            f"- 模型一致性和结果可信度应基于实际的评分分布计算"
        ),
        agent=create_summary_agent(expanded_keywords=expanded_keywords),
        expected_output=(
            "汇总报告包含：共识要点、分歧分析、综合评审报告，"
            "以及一个Markdown代码块格式的JSON评分详情（包含4个维度的分项得分、评分依据、模型一致性、结果可信度，不包含总分，仅输出一次）。"
        )
    )


def extract_summary_from_output(summary_output):
    """
    从汇总Agent的输出中提取信息
    
    Returns:
        dict: 包含共识要点、分歧分析、综合评审报告、最终评分等
    """
    result = {
        'consensus_points': '',
        'divergence_points': '',
        'final_review': '',
        'final_score_details': {},
        'model_consistency': 0.0,
        'confidence': 0.0,
        'consensus_score': 0.0
    }
    
    summary_text = summary_output.raw.strip() if hasattr(summary_output, 'raw') else str(summary_output)
    
    # 提取共识要点
    consensus_match = re.search(r'\*\*共识要点\*\*[：:]\s*\n(.*?)(?=\n\*\*|$)', summary_text, re.DOTALL)
    if consensus_match:
        result['consensus_points'] = consensus_match.group(1).strip()
    
    # 提取分歧分析
    divergence_match = re.search(r'\*\*分歧分析\*\*[：:]\s*\n(.*?)(?=\n\*\*|$)', summary_text, re.DOTALL)
    if divergence_match:
        result['divergence_points'] = divergence_match.group(1).strip()
    
    # 提取综合评审报告
    review_match = re.search(r'\*\*综合评审报告\*\*[：:]\s*\n(.*?)(?=\n\*\*最终评分详情\*\*|$)', summary_text, re.DOTALL)
    if review_match:
        result['final_review'] = review_match.group(1).strip()
    
    # 提取JSON格式的评分详情
    json_start = summary_text.find('{')
    json_end = summary_text.rfind('}')
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            json_str = summary_text[json_start:json_end+1]
            parsed = json.loads(json_str)
            
            # 提取各维度评分
            dimensions = ['创新性', '技术深度', '相关性', '实用性']
            score_data = {}
            for dim in dimensions:
                if dim in parsed and isinstance(parsed[dim], (int, float)):
                    score = float(parsed[dim])
                    if 0.0 <= score <= 1.0:
                        score_data[dim] = score
            
            result['final_score_details'] = score_data
            
            # 提取模型一致性和结果可信度
            if '模型一致性' in parsed and isinstance(parsed['模型一致性'], (int, float)):
                result['model_consistency'] = float(parsed['模型一致性'])
            
            if '结果可信度' in parsed and isinstance(parsed['结果可信度'], (int, float)):
                result['confidence'] = float(parsed['结果可信度'])
            
            # 计算共识评分（各维度之和）
            if score_data:
                result['consensus_score'] = sum(score_data.values())
            
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"解析汇总评分JSON失败: {e}")
    
    return result


def calculate_model_consistency(multi_model_results):
    """
    计算模型一致性（基于评分标准差）
    
    Args:
        multi_model_results: 多模型评审结果字典
        
    Returns:
        float: 模型一致性（0-1，1表示完全一致）
    """
    if len(multi_model_results) < 2:
        return 1.0  # 只有一个模型时，一致性为1
    
    # 收集所有模型的总分
    scores = []
    for result in multi_model_results.values():
        score = result.get('score', 0.0)
        scores.append(score)
    
    if not scores:
        return 0.0
    
    # 计算标准差
    import statistics
    if len(scores) == 1:
        return 1.0
    
    try:
        std_dev = statistics.stdev(scores)
        # 将标准差转换为一致性（标准差越小，一致性越高）
        # 假设标准差为0时一致性为1，标准差为1时一致性为0.5，标准差为2时一致性为0
        # 使用指数衰减函数
        consistency = max(0.0, min(1.0, 1.0 - (std_dev / 2.0)))
        return consistency
    except statistics.StatisticsError:
        return 0.5  # 如果无法计算，返回中等一致性

