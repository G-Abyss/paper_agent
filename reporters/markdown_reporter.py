#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown报告生成

主要修改点：
- 优化代码结构和注释
"""

import json
import re
from datetime import datetime


def has_score_details_in_review(review_content):
    """检查review内容中是否已经包含JSON代码块格式的评分详情"""
    if not review_content:
        return False
    # 检查是否包含```json代码块，并且代码块中包含评分相关的字段（如"创新性"、"技术深度"等）
    pattern = r'```json\s*.*?(?:"创新性"|"技术深度"|"相关性"|"实用性").*?```'
    return bool(re.search(pattern, review_content, re.DOTALL | re.IGNORECASE))


def extract_and_replace_score_details(review_content, score_details):
    """从review内容中提取评分详情JSON代码块，并替换为带总分的版本"""
    if not review_content or not score_details:
        return review_content
    
    # 匹配```json开始到```结束之间的内容
    pattern = r'(```json\s*)(.*?)(\s*```)'
    
    def replace_json(match):
        json_content = match.group(2).strip()
        # 检查是否包含评分相关字段
        if re.search(r'"(创新性|技术深度|相关性|实用性)"', json_content):
            # 替换为带总分的版本
            new_json_str = json.dumps(score_details, ensure_ascii=False, indent=2)
            return f"{match.group(1)}{new_json_str}{match.group(3)}"
        return match.group(0)
    
    # 替换所有匹配的JSON代码块
    result = re.sub(pattern, replace_json, review_content, flags=re.DOTALL | re.IGNORECASE)
    return result


def generate_daily_report(relevant_papers):
    """生成原始日报（Markdown 格式，与理想格式一致）"""
    report = []
    
    # 按评分分类论文
    # 只保留评分>=3.0的论文（评分<3.0的论文不输出）
    papers_to_output = [p for p in relevant_papers if p.get('score', 0.0) >= 3.0]
    
    # 高价值论文：评分>=3.5
    high_value_papers = [p for p in papers_to_output if p.get('score', 0.0) >= 3.5]
    # 其他相关论文：评分>=3.0 且 <3.5
    other_papers = [p for p in papers_to_output if 3.0 <= p.get('score', 0.0) < 3.5]
    
    # 按评分排序
    high_value_papers.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    other_papers.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    
    # 统计信息（使用表格，格式与理想文件一致）
    report.append("## 📊 统计信息")
    report.append("")
    report.append("| 类别            | 数量       |")
    report.append("| ------------- | -------- |")
    report.append(f"| 高价值论文（评分≥3.5） | {len(high_value_papers)} 篇     |")
    report.append(f"| 其他相关论文（3.0≤评分<3.5） | {len(other_papers)} 篇      |")
    report.append(f"| **总计**        | **{len(papers_to_output)} 篇** |")

    # 高价值论文（评分≥3.5，需要进一步研究）
    if high_value_papers:
        report.append("## 🔥 高价值论文（评分≥3.5，建议下载原文深入研究）")
        
        for i, paper in enumerate(high_value_papers, 1):
            report.append(f"### {i}. {paper['title']}")
            report.append("")
            
            # 添加翻译后的摘要（提到最前面）
            translated_content = paper.get('translated_content', '')
            # 如果翻译内容存在且不是错误信息，则显示翻译后的摘要
            if translated_content and translated_content not in ["摘要提取失败，无法处理", "摘要验证失败：检测到可能是AI虚构生成的内容", "摘要验证失败：关键词检测发现生成标志"]:
                report.append(f"**摘要**：{translated_content}")
                report.append("")
            
            # 添加评审内容
            review_content = paper.get('review', paper.get('summary', '')).strip()
            score_details = paper.get('score_details', {})
            
            # 如果review中已经包含评分详情，替换为带总分的版本
            if review_content and has_score_details_in_review(review_content) and score_details:
                review_content = extract_and_replace_score_details(review_content, score_details)
            
            if review_content:
                report.append(review_content)
                report.append("")
            
            # 检查review中是否已经包含评分详情，如果没有才添加
            if not has_score_details_in_review(review_content) and score_details:
                report.append("**评分详情**：")
                report.append("")
                report.append("```json")
                # 格式化JSON，确保美观
                json_str = json.dumps(score_details, ensure_ascii=False, indent=2)
                report.append(json_str)
                report.append("```")
                report.append("")
            
            # 添加论文链接
            report.append(f"🔗 [论文链接]({paper['link']})")
            report.append("")
            
            # 添加分隔符（最后一个论文后不添加）
            if i < len(high_value_papers):
                report.append("---")
                report.append("")
    
    # 其他相关论文（3.0≤评分<3.5）
    if other_papers:
        report.append("## 📖 相关论文（3.0≤评分<3.5）")
        
        for i, paper in enumerate(other_papers, 1):
            report.append(f"### {i}. {paper['title']}")
            report.append("")
            
            # 添加翻译后的摘要（提到最前面）
            translated_content = paper.get('translated_content', '')
            # 如果翻译内容存在且不是错误信息，则显示翻译后的摘要
            if translated_content and translated_content not in ["摘要提取失败，无法处理", "摘要验证失败：检测到可能是AI虚构生成的内容", "摘要验证失败：关键词检测发现生成标志"]:
                report.append("**摘要**：")
                report.append("")
                report.append(translated_content)
                report.append("")
            
            # 添加评审内容
            review_content = paper.get('review', paper.get('summary', '')).strip()
            score_details = paper.get('score_details', {})
            
            # 如果review中已经包含评分详情，替换为带总分的版本
            if review_content and has_score_details_in_review(review_content) and score_details:
                review_content = extract_and_replace_score_details(review_content, score_details)
            
            if review_content:
                report.append(review_content)
                report.append("")
            
            # 检查review中是否已经包含评分详情，如果没有才添加
            if not has_score_details_in_review(review_content):
                # 添加评分
                report.append(f"**评分：** {paper.get('score', 0.0):.2f}/4.0")
                report.append("")
                
                # 添加评分详情（JSON格式）
                if score_details:
                    report.append("**评分详情**：")
                    report.append("")
                    report.append("```json")
                    json_str = json.dumps(score_details, ensure_ascii=False, indent=2)
                    report.append(json_str)
                    report.append("```")
                    report.append("")
            
            # 添加论文链接
            report.append(f"🔗 [论文链接]({paper['link']})")
            report.append("")
            
            # 添加分隔符（最后一个论文后不添加）
            if i < len(other_papers):
                report.append("---")
                report.append("")
    
    return "\n".join(report)

