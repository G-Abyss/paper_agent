#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相关性处理（分析+提取）
"""

import re
from agents.relevance_agent import create_relevance_analyzer_agent, create_relevance_analysis_task
from agents.abstract_agent import create_abstract_extractor_agent, create_abstract_extraction_task
from agents.abstract_agent import create_pdf_processor_agent, create_pdf_abstract_extraction_task
from agents.abstract_agent import create_web_abstract_extractor_agent, create_web_abstract_extraction_task
from callbacks.crewai_callbacks import capture_crewai_output, log_agent_io
from callbacks.frontend_callbacks import send_agent_status, agent_status_callback, crewai_log_callback
from crewai import Crew
from utils.file_utils import is_pdf_url
from utils.debug_utils import debug_logger


def process_relevance_and_extraction(paper, processed_papers=None, on_paper_updated=None, on_paper_removed=None, on_paper_ready_for_confirmation=None, research_keywords=None, expanded_keywords=None):
    """
    处理相关性分析和摘要提取
    
    Args:
        paper: 论文信息字典
        processed_papers: 已处理的论文列表
        on_paper_updated: 论文更新回调函数
        on_paper_removed: 论文删除回调函数
        on_paper_ready_for_confirmation: 论文准备确认回调函数
        research_keywords: 研究方向关键词字符串，例如"机器人学、控制理论、遥操作"
        expanded_keywords: 扩写后的研究方向描述（如果提供了，将优先使用此描述）
    
    Returns:
        paper: 处理后的论文信息字典，如果不相关则返回None
    """
    paper_id = paper.get('_paper_id', '')
    
    try:
        # 更新状态：相关性分析中
        if on_paper_updated:
            on_paper_updated(paper_id, {
                'status': 'analyzing_relevance',
                'status_text': '相关性分析中...',
                'title': paper.get('title', '')
            })
        
        # 步骤1: 相关性分析（传入已处理论文列表用于重复检测）
        relevance_task = create_relevance_analysis_task(paper, processed_papers=processed_papers or [], research_keywords=research_keywords, expanded_keywords=expanded_keywords)
        
        # 发送agent工作开始状态
        send_agent_status("相关性分析专家", "start", task=relevance_task)
        
        with capture_crewai_output():
            relevance_crew = Crew(
                agents=[create_relevance_analyzer_agent()],
                tasks=[relevance_task],
                verbose=True,
                share_crew=False
            )
            relevance_result = relevance_crew.kickoff()
        
        # 发送agent工作结束状态
        send_agent_status("相关性分析专家", "end", result=relevance_result)
        
        relevance_output = relevance_result.raw.strip()
        debug_logger.log(f"相关性分析输出: {relevance_output[:200]}...")
        
        # 解析相关性判断结果
        relevance_code = None
        is_duplicate = False
        extracted_title = None
        extracted_url = None
        explanation = ""
        
        # 首先检查是否重复（优先判断，使用明确的重复标识）
        duplicate_match = re.search(r'是否重复[：:]\s*([01])', relevance_output, re.I)
        if duplicate_match:
            is_duplicate = int(duplicate_match.group(1)) == 1
        
        # 查找"相关性判断："后面的数字
        result_match = re.search(r'相关性判断[：:]\s*([01])', relevance_output, re.I)
        if result_match:
            relevance_code = int(result_match.group(1))
        
        # 如果没找到，尝试查找独立的0或1
        if relevance_code is None:
            first_line_match = re.match(r'^\s*([01])\s*$', relevance_output.split('\n')[0] if relevance_output else '')
            if first_line_match:
                relevance_code = int(first_line_match.group(1))
        
        # 如果还是没找到，检查是否包含"相关性判断：0"或"相关性判断：1"
        if relevance_code is None:
            if re.search(r'相关性判断[：:]\s*0|判断[：:]\s*0|^0\s*$', relevance_output, re.I | re.M):
                relevance_code = 0
            elif re.search(r'相关性判断[：:]\s*1|判断[：:]\s*1|^1\s*$', relevance_output, re.I | re.M):
                relevance_code = 1
        
        # 提取论文名称（规范化处理：保留空格，去除多余空格）
        title_match = re.search(r'论文名称[：:]\s*(.+?)(?:\n|$)', relevance_output, re.I)
        if title_match:
            extracted_title = ' '.join(title_match.group(1).strip().split())  # 规范化空格
        else:
            extracted_title = None
        
        # 提取论文网址
        url_match = re.search(r'论文网址[：:]\s*(.+?)(?:\n|$)', relevance_output, re.I)
        if url_match:
            extracted_url = url_match.group(1).strip()
        else:
            extracted_url = None
        
        # 提取判断依据
        explanation_match = re.search(r'判断依据[：:]\s*(.+?)(?:\n\n|\n*$)', relevance_output, re.DOTALL | re.I)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        # 如果没有明确的是否重复标识，通过判断依据中的关键词判断（作为备选）
        if not is_duplicate and explanation:
            is_duplicate = "重复" in explanation or "duplicate" in explanation.lower() or "已处理" in explanation
        
        # 使用提取的标题和URL（如果agent输出了），否则使用原始值
        # 规范化标题：去除多余空格，但保留单词之间的单个空格
        if extracted_title:
            final_title = ' '.join(extracted_title.split())  # 规范化空格
        else:
            original_title = paper.get('title', '')
            final_title = ' '.join(original_title.split()) if original_title else ''
        
        final_url = extracted_url if extracted_url else paper.get('link', '')
        
        # 优先检查是否重复（无论相关性判断结果如何）
        if is_duplicate:
            print(f"  ✗ 重复论文，跳过处理")
            debug_logger.log(f"重复论文: {final_title[:60]}", "INFO")
            if on_paper_removed:
                on_paper_removed(paper_id)
            return None
        
        # 如果相关性分析失败或不相关，删除论文并返回
        if relevance_code != 1:
            if relevance_code == 0:
                print(f"  ✗ 不符合研究方向")
                debug_logger.log(f"不符合方向: {final_title[:60]} (判断依据: {explanation[:100] if explanation else '未提供'})")
            else:
                print(f"  ✗ 相关性分析出错: 无法解析结果")
                debug_logger.log(f"相关性分析出错: {final_title[:60]} - 无法解析结果", "ERROR")
            
            # 删除论文（从来源栏移除）
            if on_paper_removed:
                on_paper_removed(paper_id)
            return None
        
        # 步骤2: 如果相关，进行摘要提取
        print(f"  ✓ 符合研究方向，开始提取摘要...")
        debug_logger.log(f"相关论文: {final_title[:60]} (判断依据: {explanation[:100]})", "SUCCESS")
        
        # 更新论文信息
        paper['title'] = final_title
        paper['link'] = final_url
        paper['relevance_score'] = 1
        paper['relevance_explanation'] = explanation
        
        # 更新状态：相关性分析通过，开始摘要提取
        if on_paper_updated:
            on_paper_updated(paper_id, {
                'status': 'extracting_abstract',
                'status_text': '摘要提取中...',
                'title': final_title,
                'link': final_url
            })
        
        if not final_url:
            print(f"  ✗ 论文没有URL，无法提取摘要")
            debug_logger.log(f"论文没有URL，无法提取摘要", "WARNING")
            paper['full_abstract'] = None
            paper['is_pdf'] = False
            # 更新状态：摘要提取失败，但允许用户手动添加
            if on_paper_updated:
                on_paper_updated(paper_id, {
                    'status': 'abstract_extraction_failed',
                    'status_text': '摘要提取失败：无URL，可以手动添加摘要',
                    'title': final_title
                })
            # 即使没有URL，也显示可编辑摘要框，让用户手动添加
            if on_paper_ready_for_confirmation:
                on_paper_ready_for_confirmation(paper_id, {
                    'title': final_title,
                    'abstract': '',  # 空摘要，让用户手动输入
                    'link': ''
                })
            return paper
        
        # 创建摘要提取任务
        abstract_task = create_abstract_extraction_task(final_title, final_url)
        
        # 发送agent工作开始状态
        send_agent_status("摘要提取专家", "start", task=abstract_task)
        
        with capture_crewai_output():
            abstract_crew = Crew(
                agents=[create_abstract_extractor_agent()],
                tasks=[abstract_task],
                verbose=True,
                share_crew=False
            )
            abstract_result = abstract_crew.kickoff()
        
        # 发送agent工作结束状态
        send_agent_status("摘要提取专家", "end", result=abstract_result)
        
        # 记录 Agent 的输入和输出
        if crewai_log_callback:
            log_agent_io("摘要提取专家", abstract_task, abstract_result, crewai_log_callback)
        
        abstract_output = abstract_result.raw.strip()
        debug_logger.log(f"摘要提取输出: {abstract_output[:200]}...")
        
        # 解析摘要提取结果
        extraction_result = None
        abstract_content = ""
        
        # 查找"提取结果："后面的数字
        result_match = re.search(r'提取结果[：:]\s*([01])', abstract_output, re.I)
        if result_match:
            extraction_result = int(result_match.group(1))
        
        # 如果没找到，尝试查找独立的0或1
        if extraction_result is None:
            first_line_match = re.match(r'^\s*([01])\s*$', abstract_output.split('\n')[0] if abstract_output else '')
            if first_line_match:
                extraction_result = int(first_line_match.group(1))
        
        # 如果还是没找到，检查是否包含"提取结果：0"或"提取结果：1"
        if extraction_result is None:
            if re.search(r'提取结果[：:]\s*0|结果[：:]\s*0|^0\s*$', abstract_output, re.I | re.M):
                extraction_result = 0
            elif re.search(r'提取结果[：:]\s*1|结果[：:]\s*1|^1\s*$', abstract_output, re.I | re.M):
                extraction_result = 1
        
        # 如果提取结果为1，提取摘要内容
        if extraction_result == 1:
            # 查找"摘要内容："后面的内容
            abstract_match = re.search(r'摘要内容[：:]\s*(.+?)(?:\n\n|\n提取结果|$)', abstract_output, re.DOTALL | re.I)
            if abstract_match:
                abstract_content = abstract_match.group(1).strip()
            else:
                # 如果没有找到"摘要内容："标记，尝试提取"提取结果：1"之后的所有内容
                after_result = re.split(r'提取结果[：:]\s*1\s*\n?', abstract_output, flags=re.I)
                if len(after_result) > 1:
                    abstract_content = after_result[1].strip()
                    # 移除可能的"摘要内容："标记
                    abstract_content = re.sub(r'^摘要内容[：:]\s*', '', abstract_content, flags=re.I)
            
            if abstract_content and len(abstract_content) > 50:
                paper['full_abstract'] = abstract_content
                paper['is_pdf'] = is_pdf_url(final_url) or 'arxiv.org' in final_url.lower()
                print(f"  ✓ 摘要提取成功 ({len(abstract_content)} 字符)")
                debug_logger.log(f"✓ 摘要提取成功 ({len(abstract_content)} 字符)", "SUCCESS")
                # 更新状态：摘要提取成功，可以编辑摘要
                if on_paper_updated:
                    on_paper_updated(paper_id, {
                        'status': 'abstract_extracted',
                        'status_text': '摘要提取成功，可以编辑摘要',
                        'title': final_title,
                        'link': final_url
                    })
                # 立即显示可编辑摘要框（不阻塞）
                if on_paper_ready_for_confirmation:
                    on_paper_ready_for_confirmation(paper_id, {
                        'title': final_title,
                        'abstract': abstract_content,
                        'link': final_url
                    })
            else:
                paper['full_abstract'] = None
                paper['is_pdf'] = False
                print(f"  ✗ 提取结果=1但摘要内容为空或太短")
                debug_logger.log(f"提取结果=1但摘要内容为空或太短", "WARNING")
                # 更新状态：摘要提取失败，但允许用户手动添加
                if on_paper_updated:
                    on_paper_updated(paper_id, {
                        'status': 'abstract_extraction_failed',
                        'status_text': '摘要提取失败，可以手动添加摘要',
                        'title': final_title,
                        'link': final_url
                    })
                # 即使提取失败，也显示可编辑摘要框，让用户手动添加
                if on_paper_ready_for_confirmation:
                    on_paper_ready_for_confirmation(paper_id, {
                        'title': final_title,
                        'abstract': '',  # 空摘要，让用户手动输入
                        'link': final_url
                    })
        else:
            paper['full_abstract'] = None
            paper['is_pdf'] = False
            print(f"  ✗ 摘要提取失败（Agent反馈提取结果=0）")
            debug_logger.log(f"摘要提取失败（Agent反馈提取结果=0）", "WARNING")
            # 更新状态：摘要提取失败，但允许用户手动添加
            if on_paper_updated:
                on_paper_updated(paper_id, {
                    'status': 'abstract_extraction_failed',
                    'status_text': '摘要提取失败，可以手动添加摘要',
                    'title': final_title,
                    'link': final_url
                })
            # 即使提取失败，也显示可编辑摘要框，让用户手动添加
            if on_paper_ready_for_confirmation:
                on_paper_ready_for_confirmation(paper_id, {
                    'title': final_title,
                    'abstract': '',  # 空摘要，让用户手动输入
                    'link': final_url
                })
        
        return paper
        
    except Exception as e:
        print(f"  ✗ 处理论文时出错: {str(e)}")
        debug_logger.log(f"处理论文时出错: {paper.get('title', '')[:60]} - {str(e)}", "ERROR")
        return None

