#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV报告生成

主要修改点：
- 优化代码结构和注释
"""

import os
import pandas as pd
from datetime import datetime
import logging


def export_all_papers_to_csv(relevant_papers, output_dir="reports"):
    """
    将所有符合关键词的论文导出到CSV文件（包含处理结果）
    
    Args:
        relevant_papers: 所有相关论文列表（符合关键词的论文）
        output_dir: 输出目录
    """
    if not relevant_papers:
        print("\n没有相关论文需要导出")
        return None
    
    # 按评分排序（处理成功的排在前面）
    relevant_papers_sorted = sorted(relevant_papers, 
                                   key=lambda x: (
                                       x.get('score', 0.0) if (x.get('review') and x.get('score_details')) else -1
                                   ), 
                                   reverse=True)
    
    # 准备数据
    csv_data = []
    
    for paper in relevant_papers_sorted:
        # 检查是否成功读取摘要（full_abstract不为None表示成功）
        # 成功读取摘要：full_abstract存在且不为None，且长度>0
        full_abstract = paper.get('full_abstract')
        is_success = 1 if (full_abstract is not None and len(full_abstract) > 0) else 0
        
        # 获取翻译后的摘要（优先使用翻译后的摘要）
        translated_content = paper.get('translated_content', '')
        # 如果翻译后的摘要存在且不是错误信息，使用翻译后的摘要；否则使用原始摘要
        if translated_content and translated_content not in [
            "摘要提取失败，无法处理", 
            "摘要验证失败：检测到可能是AI虚构生成的内容", 
            "摘要验证失败：关键词检测发现生成标志",
            "翻译失败:"
        ]:
            abstract_chinese = translated_content
        elif is_success == 1:
            # 如果翻译失败但摘要提取成功，使用原始摘要
            abstract_chinese = full_abstract
        else:
            # 摘要提取失败
            abstract_chinese = ''
        
        # 获取英文原文摘要
        original_english_abstract = paper.get('original_english_abstract', '')
        # 如果没有保存的英文原文，使用full_abstract作为英文原文
        if not original_english_abstract:
            original_english_abstract = full_abstract if is_success == 1 else ''
        
        # 合并中文摘要和英文原文摘要
        abstract_combined = ''
        if abstract_chinese and original_english_abstract:
            abstract_combined = f"{abstract_chinese}\n\n[English Original]\n{original_english_abstract}"
        elif abstract_chinese:
            abstract_combined = abstract_chinese
        elif original_english_abstract:
            abstract_combined = f"[English Original]\n{original_english_abstract}"
        
        # 获取评审结果
        review = paper.get('review', '')
        
        # 构建数据行（使用新的表头格式：title, link, abstract, 评审结果）
        row = {
            'title': paper.get('title', ''),
            'link': paper.get('link', ''),
            'abstract': abstract_combined,  # 中文摘要和英文原文的合集
            '评审结果': review,
        }
        
        csv_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(csv_data)
    
    # 生成CSV文件名
    csv_filename = f"{output_dir}/相关论文_{datetime.now().strftime('%Y%m%d')}.csv"
    
    # 保存到CSV
    try:
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')  # 使用utf-8-sig以支持Excel正确显示中文
        
        successful_count = sum(1 for p in relevant_papers if p.get('full_abstract') is not None and 
                              len(p.get('full_abstract', '')) > 0)
        failed_count = len(relevant_papers) - successful_count
        
        print(f"\n✓ 相关论文已导出到CSV: {csv_filename}")
        print(f"  - 共导出 {len(relevant_papers)} 篇相关论文（符合关键词的论文）")
        print(f"  - 成功读取摘要: {successful_count} 篇")
        print(f"  - 摘要读取失败: {failed_count} 篇")
        return csv_filename
    except Exception as e:
        logging.error(f"导出CSV时出错: {str(e)}")
        print(f"\n✗ 导出CSV失败: {str(e)}")
        return None

