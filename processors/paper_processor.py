#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文处理流程（验证+翻译+评审）

主要修改点：
- 优化代码结构和注释
"""

import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.validation_agent import create_abstract_validation_and_cleaning_agent, create_abstract_validation_and_cleaning_task
from agents.translation_agent import create_translator_agent, create_translation_task
from agents.review_agent import create_reviewer_agent, create_review_task, extract_score_from_review
from agents.summary_agent import create_summary_agent, create_summary_task, extract_summary_from_output
from callbacks.crewai_callbacks import capture_crewai_output, log_agent_io
from callbacks.frontend_callbacks import send_agent_status, crewai_log_callback
from crewai import Crew
from utils.debug_utils import debug_logger
from utils.model_config import get_active_model, get_selected_models
from utils.llm_utils import create_llm_from_model_config


def process_paper_with_crewai(paper, full_abstract, source_type="网页", on_paper_updated=None, expanded_keywords=None):
    """
    使用 CrewAI 框架处理论文：验证 + 翻译 + 评审
    返回处理结果字典
    
    Args:
        paper: 论文信息字典
        full_abstract: 完整摘要文本
        source_type: 摘要来源类型（"PDF"或"网页"）
        on_paper_updated: 论文更新回调函数（可选）
        expanded_keywords: 扩写后的研究方向描述（可选），用于定制agent的专业领域
    """
    paper_id = paper.get('_paper_id', '')
    
    try:
        # 更新状态：验证与清洗中
        if on_paper_updated and paper_id:
            on_paper_updated(paper_id, {
                'status': 'validating_cleaning',
                'status_text': '验证与清洗摘要中...',
                'title': paper.get('title', '')
            })
        
        # 步骤0: 验证与清洗摘要（合并步骤）
        print("  [步骤0/3] 验证与清洗摘要中...")
        validation_cleaning_task = create_abstract_validation_and_cleaning_task(paper, full_abstract, source_type)
        validation_cleaning_agent = create_abstract_validation_and_cleaning_agent()
        
        # 发送agent工作开始状态
        send_agent_status("摘要验证与清洗专家", "start", task=validation_cleaning_task)
        
        with capture_crewai_output():
            validation_cleaning_crew = Crew(
                agents=[validation_cleaning_agent],
                tasks=[validation_cleaning_task],
                verbose=True,
                share_crew=False
            )
            validation_cleaning_result = validation_cleaning_crew.kickoff()
        
        # 发送agent工作结束状态
        send_agent_status("摘要验证与清洗专家", "end", result=validation_cleaning_result)
        
        # 记录 Agent 的输入和输出
        if crewai_log_callback:
            log_agent_io("摘要验证与清洗专家", validation_cleaning_task, validation_cleaning_result, crewai_log_callback)
        
        validation_cleaning_output = validation_cleaning_result.raw.strip()
        
        # 解析验证与清洗结果
        validation_result_code = None
        validation_explanation = ""
        cleaned_abstract = full_abstract  # 默认使用原始摘要
        
        # 查找"验证结果："后面的数字
        result_match = re.search(r'验证结果[：:]\s*([01])', validation_cleaning_output, re.I)
        if result_match:
            validation_result_code = int(result_match.group(1))
        
        # 如果没找到，尝试查找独立的0或1
        if validation_result_code is None:
            first_line_match = re.match(r'^\s*([01])\s*$', validation_cleaning_output.split('\n')[0] if validation_cleaning_output else '')
            if first_line_match:
                validation_result_code = int(first_line_match.group(1))
        
        # 如果还是没找到，检查是否包含"验证结果：0"或"验证结果：1"
        if validation_result_code is None:
            if re.search(r'验证结果[：:]\s*0|结果[：:]\s*0|^0\s*$', validation_cleaning_output, re.I | re.M):
                validation_result_code = 0
            elif re.search(r'验证结果[：:]\s*1|结果[：:]\s*1|^1\s*$', validation_cleaning_output, re.I | re.M):
                validation_result_code = 1
        
        # 提取验证说明
        explanation_match = re.search(r'验证说明[：:]\s*(.+?)(?:\n清洗后的摘要|$)', validation_cleaning_output, re.DOTALL | re.I)
        if explanation_match:
            validation_explanation = explanation_match.group(1).strip()
        
        # 如果验证结果为0（可能是虚构的），返回失败
        if validation_result_code == 0:
            print(f"    ✗ 摘要验证失败：检测到可能是AI虚构生成的内容")
            debug_logger.log(f"摘要验证失败：{validation_explanation}", "WARNING")
            return {
                'translated_content': "摘要验证失败：检测到可能是AI虚构生成的内容",
                'review': "摘要验证失败：检测到可能是AI虚构生成的内容",
                'score': 0.0,
                'score_details': {},
                'is_high_value': False,
                'full_abstract': full_abstract,
                'original_english_abstract': full_abstract  # 保存原始英文摘要
            }
        
        # 如果验证结果为1（真实可靠），提取清洗后的摘要
        if validation_result_code == 1:
            print(f"    ✓ 摘要验证通过：内容真实可靠")
            debug_logger.log(f"摘要验证通过：{validation_explanation}", "SUCCESS")
            
            # 提取清洗后的摘要
            cleaned_match = re.search(r'清洗后的摘要[：:]\s*(.+?)(?:\n\n|$)', validation_cleaning_output, re.DOTALL | re.I)
            if cleaned_match:
                cleaned_abstract = cleaned_match.group(1).strip()
                # 如果清洗后的摘要太短，使用原始摘要
                if len(cleaned_abstract) < 50:
                    print(f"    ⚠ 清洗后的摘要太短，使用原始摘要")
                    cleaned_abstract = full_abstract
                else:
                    print(f"    ✓ 摘要清洗完成 ({len(cleaned_abstract)} 字符)")
                    debug_logger.log(f"摘要清洗完成，长度: {len(cleaned_abstract)} 字符", "SUCCESS")
            else:
                # 如果没有找到清洗后的摘要，使用原始摘要
                print(f"    ⚠ 未找到清洗后的摘要，使用原始摘要")
                cleaned_abstract = full_abstract
        else:
            # 如果无法解析验证结果，使用关键词检测作为备选
            print(f"    ⚠ 无法解析验证结果，使用关键词检测...")
            generation_keywords = [
                'simulated', 'generated', 'based on the', 'not provided', 'not available',
                'cannot extract', 'unable to', 'assumed', 'presumed', 'inferred',
                '推测', '生成', '模拟', '假设', '推断', '未提供', '无法提取',
                'Note:', 'note:', '注意', '说明', '备注'
            ]
            abstract_lower = full_abstract.lower()
            is_generated = any(keyword in abstract_lower for keyword in generation_keywords)
            
            if is_generated:
                print(f"    ✗ 关键词检测失败：检测到生成标志")
                debug_logger.log(f"关键词检测失败：检测到生成标志", "WARNING")
                return {
                    'translated_content': "摘要验证失败：关键词检测发现生成标志",
                    'review': "摘要验证失败：关键词检测发现生成标志",
                    'score': 0.0,
                    'score_details': {},
                    'is_high_value': False,
                    'full_abstract': full_abstract,
                    'original_english_abstract': full_abstract
                }
            else:
                print(f"    ✓ 关键词检测通过：未发现生成标志")
                debug_logger.log(f"关键词检测通过：未发现生成标志", "INFO")
        
        # 更新状态：翻译中
        if on_paper_updated and paper_id:
            on_paper_updated(paper_id, {
                'status': 'translating',
                'status_text': '翻译中...',
                'title': paper.get('title', '')
            })
        
        # 步骤1: 翻译（输出包含中英文双语）
        print("  [步骤1/3] 专业翻译中...")
        translation_task = create_translation_task(paper, cleaned_abstract, expanded_keywords=expanded_keywords)
        translation_agent = create_translator_agent(expanded_keywords=expanded_keywords)
        
        # 发送agent工作开始状态
        send_agent_status("专业翻译专家", "start", task=translation_task)
        
        with capture_crewai_output():
            translation_crew = Crew(
                agents=[translation_agent],
                tasks=[translation_task],
                verbose=True,
                share_crew=False
            )
            translation_result = translation_crew.kickoff()
        
        # 发送agent工作结束状态
        send_agent_status("专业翻译专家", "end", result=translation_result)
        
        # 记录 Agent 的输入和输出
        if crewai_log_callback:
            log_agent_io("专业翻译专家", translation_task, translation_result, crewai_log_callback)
        
        translation_output = translation_result.raw.strip()
        
        # 解析翻译结果，提取中文翻译和英文原文
        chinese_translation = ""
        original_english_abstract = cleaned_abstract  # 默认使用清洗后的摘要
        
        # 查找"中文翻译："后面的内容
        chinese_match = re.search(r'中文翻译[：:]\s*(.+?)(?:\n\n英文原文|$)', translation_output, re.DOTALL | re.I)
        if chinese_match:
            chinese_translation = chinese_match.group(1).strip()
        else:
            # 如果没有找到标记，尝试提取第一段作为中文翻译
            lines = translation_output.split('\n')
            if lines:
                chinese_translation = lines[0].strip()
        
        # 查找"英文原文："后面的内容
        english_match = re.search(r'英文原文[：:]\s*(.+?)(?:\n\n|$)', translation_output, re.DOTALL | re.I)
        if english_match:
            original_english_abstract = english_match.group(1).strip()
        
        # 如果中文翻译为空，使用整个输出作为中文翻译
        if not chinese_translation or len(chinese_translation) < 10:
            chinese_translation = translation_output
        
        print(f"    ✓ 翻译完成 (中文: {len(chinese_translation)} 字符, 英文: {len(original_english_abstract)} 字符)")
        debug_logger.log(f"翻译完成，中文长度: {len(chinese_translation)} 字符", "SUCCESS")
        
        # 更新状态：评审中
        if on_paper_updated and paper_id:
            on_paper_updated(paper_id, {
                'status': 'reviewing',
                'status_text': '评审和评分中...',
                'title': paper.get('title', '')
            })
        
        # 步骤2: 多模型并行评审（使用中英文双语）
        print("  [步骤2/3] 多模型并行评审和评分中...")
        
        # 获取启用的模型列表
        enabled_models = get_selected_models()
        
        if not enabled_models:
            logging.warning("没有启用的模型，使用默认模型进行评审")
            active_model = get_active_model()
            enabled_models = [active_model] if active_model else []
        
        if not enabled_models:
            logging.error("无法获取模型配置，评审失败")
            return {
                'translated_content': chinese_translation,
                'review': "评审失败：无法获取模型配置",
                'score': 0.0,
                'score_details': {},
                'is_high_value': False,
                'full_abstract': full_abstract,
                'original_english_abstract': original_english_abstract,
                'multi_model_results': {}
            }
        
        # 多模型并行评审
        multi_model_results = {}
        
        def process_single_model_review(model_config):
            """处理单个模型的评审"""
            model_id = model_config.get('id', 'unknown')
            model_name = model_config.get('name', 'Unknown')
            
            try:
                # 为每个模型创建LLM实例
                model_llm = create_llm_from_model_config(model_config)
                if not model_llm:
                    logging.error(f"无法为模型 {model_name} 创建LLM实例")
                    return model_id, None
                
                # 创建评审任务和Agent
                review_task = create_review_task(paper, chinese_translation, original_english_abstract, expanded_keywords=expanded_keywords)
                review_agent = create_reviewer_agent(expanded_keywords=expanded_keywords, llm=model_llm)
                
                # 发送agent工作开始状态
                send_agent_status(f"评审专家-{model_name}", "start", task=review_task)
                
                with capture_crewai_output():
                    review_crew = Crew(
                        agents=[review_agent],
                        tasks=[review_task],
                        verbose=True,
                        share_crew=False
                    )
                    review_result = review_crew.kickoff()
                
                # 发送agent工作结束状态
                send_agent_status(f"评审专家-{model_name}", "end", result=review_result)
                
                # 记录 Agent 的输入和输出
                if crewai_log_callback:
                    log_agent_io(f"评审专家-{model_name}", review_task, review_result, crewai_log_callback)
                
                review_text = review_result.raw.strip()
                
                # 提取评分
                score_data = extract_score_from_review(review_text)
                
                return model_id, {
                    'model_name': model_name,
                    'model_id': model_id,
                    'review': review_text,
                    'score': score_data.get('总分', 0.0),
                    'score_details': score_data,
                    'is_high_value': score_data.get('总分', 0.0) >= 3.5
                }
            
            except Exception as e:
                logging.error(f"模型 {model_name} 评审失败: {str(e)}")
                return model_id, {
                    'model_name': model_name,
                    'model_id': model_id,
                    'review': f"评审失败: {str(e)}",
                    'score': 0.0,
                    'score_details': {},
                    'is_high_value': False,
                    'error': str(e)
                }
        
        # 使用线程池并行处理多个模型
        print(f"    使用 {len(enabled_models)} 个模型并行评审...")
        with ThreadPoolExecutor(max_workers=len(enabled_models)) as executor:
            # 提交所有任务
            future_to_model = {
                executor.submit(process_single_model_review, model): model 
                for model in enabled_models
            }
            
            # 收集结果
            for future in as_completed(future_to_model):
                model_config = future_to_model[future]
                try:
                    model_id, result = future.result()
                    if result:
                        multi_model_results[model_id] = result
                        print(f"    ✓ 模型 {result.get('model_name', 'Unknown')} 评审完成 (评分: {result.get('score', 0.0):.2f}/4.0)")
                    else:
                        print(f"    ✗ 模型 {model_config.get('name', 'Unknown')} 评审失败")
                except Exception as e:
                    logging.error(f"收集模型评审结果时出错: {str(e)}")
                    model_id = model_config.get('id', 'unknown')
                    multi_model_results[model_id] = {
                        'model_name': model_config.get('name', 'Unknown'),
                        'model_id': model_id,
                        'review': f"评审失败: {str(e)}",
                        'score': 0.0,
                        'score_details': {},
                        'is_high_value': False,
                        'error': str(e)
                    }
        
        # 计算平均评分（用于兼容性）
        if multi_model_results:
            avg_score = sum(r.get('score', 0.0) for r in multi_model_results.values()) / len(multi_model_results)
            # 使用第一个模型的结果作为主要结果（用于兼容现有代码）
            first_result = list(multi_model_results.values())[0]
        else:
            avg_score = 0.0
            first_result = {
                'review': "所有模型评审失败",
                'score': 0.0,
                'score_details': {}
            }
        
        # 如果有多个模型评审结果，调用汇总agent进行综合分析
        summary_result = None
        if len(multi_model_results) > 1:
            try:
                print("    [汇总阶段] 综合分析多个模型的评审结果...")
                
                # 更新状态
                if on_paper_updated and paper_id:
                    on_paper_updated(paper_id, {
                        'status': 'reviewing',
                        'status_text': '汇总分析中...',
                        'title': paper.get('title', '')
                    })
                
                # 创建汇总任务和Agent
                summary_task = create_summary_task(paper, multi_model_results, expanded_keywords=expanded_keywords)
                summary_agent = create_summary_agent(expanded_keywords=expanded_keywords)
                
                # 发送agent工作开始状态
                send_agent_status("综合评审汇总专家", "start", task=summary_task)
                
                with capture_crewai_output():
                    summary_crew = Crew(
                        agents=[summary_agent],
                        tasks=[summary_task],
                        verbose=True,
                        share_crew=False
                    )
                    summary_output = summary_crew.kickoff()
                
                # 发送agent工作结束状态
                send_agent_status("综合评审汇总专家", "end", result=summary_output)
                
                # 记录 Agent 的输入和输出
                if crewai_log_callback:
                    log_agent_io("综合评审汇总专家", summary_task, summary_output, crewai_log_callback)
                
                # 提取汇总结果
                summary_result = extract_summary_from_output(summary_output)
                
                print(f"    ✓ 汇总分析完成 (一致性: {summary_result.get('model_consistency', 0.0):.2f}, 可信度: {summary_result.get('confidence', 0.0):.2f})")
                debug_logger.log(f"汇总分析完成: 一致性={summary_result.get('model_consistency', 0.0):.2f}", "SUCCESS")
                
            except Exception as e:
                logging.error(f"汇总分析失败: {str(e)}")
                debug_logger.log(f"汇总分析失败: {str(e)}", "ERROR")
                summary_result = None
        
        return {
            'translated_content': chinese_translation,  # 只保存中文翻译
            'review': first_result.get('review', ''),
            'score': avg_score,  # 使用平均评分
            'score_details': first_result.get('score_details', {}),
            'is_high_value': avg_score >= 3.5,
            'full_abstract': full_abstract,
            'original_english_abstract': original_english_abstract,  # 保存英文原文
            'multi_model_results': multi_model_results,  # 保存所有模型的结果
            'summary_result': summary_result  # 保存汇总结果（如果有多个模型）
        }
    except Exception as e:
        logging.error(f"处理论文时出错: {str(e)}")
        return {
            'translated_content': f"翻译失败: {str(e)}",
            'review': f"评审失败: {str(e)}",
            'score': 0.0,
            'score_details': {},
            'is_high_value': False,
            'full_abstract': full_abstract,
            'original_english_abstract': full_abstract  # 保存原始英文摘要
        }

