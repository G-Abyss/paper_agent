#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序入口

主要修改点：
- 删除未使用的导入（email, ssl, imaplib, crewai_log_callback, agent_status_callback, capture_crewai_output）
- 删除注释掉的代码
- 删除重复的print语句
- 优化代码结构和注释
"""

import os
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging

from config import (
    CSV_FILE_PATH, CSV_TITLE_COLUMN, CSV_ABSTRACT_COLUMN, CSV_LINK_COLUMN,
    START_DAYS, END_DAYS, BACKUP_DIR
)
# LOCAL_MODE 在函数内部动态读取，确保使用最新的环境变量
from utils.email_utils import connect_gmail
from utils.file_utils import load_papers_from_csv, deduplicate_papers_by_title
from utils.debug_utils import debug_logger
from utils.email_storage import sync_remote_emails_to_local, filter_emails_by_date_range
from processors.relevance_processor import process_relevance_and_extraction
from processors.paper_processor import process_paper_with_crewai
from agents.keyword_expansion_agent import expand_keywords
from reporters.markdown_reporter import generate_daily_report
from reporters.csv_reporter import export_all_papers_to_csv
from callbacks.frontend_callbacks import set_callbacks


def main(
    on_log=None, 
    on_paper_added=None, 
    on_paper_updated=None, 
    on_paper_removed=None, 
    on_file_generated=None, 
    on_agent_status=None, 
    on_waiting_confirmation=None, 
    get_confirmed_abstracts=None, 
    on_paper_ready_for_confirmation=None, 
    get_confirmed_abstract=None
):
    """
    主程序
    
    Args:
        on_log: 日志回调函数 (level, message)
        on_paper_added: 论文添加回调函数 (paper)
        on_paper_updated: 论文更新回调函数 (paper_id, paper)
        on_paper_removed: 论文删除回调函数 (paper_id)
        on_file_generated: 文件生成回调函数 (file_info)
        on_agent_status: agent状态回调函数 (agent_name, status, target, output)
        get_confirmed_abstracts: 获取确认后的摘要函数 () -> 返回 {paper_id: abstract_text}（批量模式）
        on_paper_ready_for_confirmation: 单个论文准备确认回调函数 (paper_id, paper_data)
        on_waiting_confirmation: 等待确认回调函数（已弃用，保留以兼容旧代码）
        get_confirmed_abstract: 获取单个论文确认后的摘要函数（已弃用，保留以兼容旧代码）
    """
    print("=" * 80)
    print("Paper Summarizer - 学术论文自动总结系统")
    print("=" * 80)
    print()
    
    # 设置回调函数
    set_callbacks(on_log=on_log, on_agent_status=on_agent_status)
    
    # 获取研究方向关键词（从环境变量读取，如果没有则使用默认值）
    research_keywords = os.environ.get('RESEARCH_KEYWORDS', '机器人学、控制理论、遥操作、机器人动力学、力控、机器学习')
    
    # 辅助函数：发送日志
    def log(level, message):
        print(message)
        if on_log:
            on_log(level, message)
    
    # 0. 关键词扩写（在开始处理论文之前执行，这是第一步）
    log('info', f'用户提供的关键词: {research_keywords}')
    log('info', '正在执行关键词扩写...')
    expanded_keywords = expand_keywords(research_keywords, on_log=log)
    if expanded_keywords:
        log('info', '✓ 关键词扩写完成')
        log('info', f'扩写结果预览: {expanded_keywords[:200]}...' if len(expanded_keywords) > 200 else f'扩写结果: {expanded_keywords}')
        log('info', '扩写后的研究方向描述将用于后续的论文相关性分析、翻译和评审')
    else:
        log('warning', '⚠ 关键词扩写失败或跳过，将使用原始关键词')
        expanded_keywords = None
    
    # 1. 根据模式选择数据源
    # 动态读取LOCAL_MODE，确保使用最新的环境变量
    LOCAL_MODE = os.getenv('LOCAL', '0') == '1'
    if LOCAL_MODE:
        print("=" * 80)
        print("本地处理模式 (LOCAL=1)")
        print("=" * 80)
        print(f"CSV文件路径: {CSV_FILE_PATH}")
        print(f"标题列索引: {CSV_TITLE_COLUMN}")
        print(f"摘要列索引: {CSV_ABSTRACT_COLUMN}")
        if CSV_LINK_COLUMN:
            print(f"链接列索引: {CSV_LINK_COLUMN}")
        else:
            print("链接列索引: 未指定（将使用空字符串）")
        print()
        
        # 从CSV文件读取论文信息
        try:
            all_papers = load_papers_from_csv(
                csv_path=CSV_FILE_PATH,
                title_col=CSV_TITLE_COLUMN,
                abstract_col=CSV_ABSTRACT_COLUMN,
                link_col=CSV_LINK_COLUMN if CSV_LINK_COLUMN else None
            )
        except Exception as e:
            print(f"\n✗ 错误: {str(e)}")
            logging.error(f"从CSV读取论文信息失败: {str(e)}")
            return
        
        if not all_papers:
            print("\n没有从CSV文件中读取到论文信息")
            return
        
        print(f"\n总共从CSV读取到 {len(all_papers)} 篇论文")
        debug_logger.log(f"本地模式：从CSV文件 {CSV_FILE_PATH} 读取到 {len(all_papers)} 篇论文")
    else:
        print("=" * 80)
        print("邮件处理模式 (LOCAL=0)")
        print("=" * 80)
        print()
        
        # 步骤1: 同步远程邮箱到本地
        print("步骤1: 同步远程邮箱到本地...")
        mail = None
        try:
            mail = connect_gmail()
            # 同步最近30天的邮件（确保覆盖用户可能选择的时间范围）
            sync_result = sync_remote_emails_to_local(mail, start_days=30, end_days=0)
            print(f"✓ {sync_result['message']}")
            print(f"  共处理 {sync_result['total_count']} 封邮件，新增/更新 {sync_result['updated_count']} 封")
        except Exception as e:
            print(f"✗ 同步远程邮箱失败: {str(e)}")
            logging.error(f"同步远程邮箱失败: {str(e)}")
            if mail:
                try:
                    mail.close()
                    mail.logout()
                except:
                    pass
            return
        finally:
            if mail:
                try:
                    mail.close()
                    mail.logout()
                except:
                    pass
        
        # 步骤2: 从本地存储中根据时间范围筛选邮件
        print(f"\n步骤2: 从本地存储筛选邮件（前{START_DAYS}天到前{END_DAYS}天）...")
        filtered_emails = filter_emails_by_date_range(START_DAYS, END_DAYS)
        
        if not filtered_emails:
            print("\n没有找到符合条件的邮件")
            return
        
        print(f"✓ 找到 {len(filtered_emails)} 封符合条件的邮件")
        
        # 步骤3: 从筛选的邮件中提取论文信息
        print("\n步骤3: 提取论文信息...")
        all_papers = []
        for email_item in filtered_emails:
            papers = email_item.get('papers', [])
            all_papers.extend(papers)
            print(f"  邮件 [{email_item.get('subject', '无主题')[:50]}...] 包含 {len(papers)} 篇论文")
        
        print(f"\n✓ 总共提取到 {len(all_papers)} 篇论文")
        debug_logger.log(f"总共提取到 {len(all_papers)} 篇论文")
    
    # 2. 查重处理：去除重复论文（在相关性分析之前）
    print("\n步骤4: 查重处理（去除重复论文）...")
    original_count = len(all_papers)
    all_papers, duplicate_papers = deduplicate_papers_by_title(all_papers)
    duplicate_count = len(duplicate_papers)
    
    if duplicate_count > 0:
        print(f"✓ 发现 {duplicate_count} 篇重复论文，已去除")
        debug_logger.log(f"查重：从 {original_count} 篇论文中去除 {duplicate_count} 篇重复论文")
        for dup in duplicate_papers[:10]:  # 只显示前10个重复项
            print(f"  重复: \"{dup['original_title'][:60]}...\" 与 \"{dup['duplicate_title'][:60]}...\"")
            debug_logger.log(f"  重复论文: \"{dup['original_title']}\" 与 \"{dup['duplicate_title']}\"")
        if duplicate_count > 10:
            print(f"  ... 还有 {duplicate_count - 10} 篇重复论文未显示")
    else:
        print(f"✓ 未发现重复论文")
        debug_logger.log(f"查重：未发现重复论文")
    
    print(f"✓ 查重后剩余 {len(all_papers)} 篇论文")
    debug_logger.log(f"查重后剩余 {len(all_papers)} 篇论文")
    
    # 3. 立即将所有论文显示在"来源"栏中（状态为待处理）
    if on_paper_added:
        print(f"\n正在添加 {len(all_papers)} 篇论文到来源栏...")
        for i, paper in enumerate(all_papers, 1):
            paper_id = f"paper_pending_{i}"
            paper['_paper_id'] = paper_id
            on_paper_added({
                'id': paper_id,
                'title': paper.get('title', ''),
                'link': paper.get('link', ''),
                'status': 'pending'
            })
        print(f"✓ 已添加 {len(all_papers)} 篇论文到来源栏")
        debug_logger.log(f"已添加 {len(all_papers)} 篇论文到来源栏")
    
    # 4. 并行处理多篇论文（每篇论文内部：相关性分析 -> 摘要提取，串联执行）
    print("\n正在并行处理论文（不同论文之间并行，单篇论文内部串联）...")
    debug_logger.log_separator("并行处理论文")
    
    # 确保downloads文件夹存在
    downloads_dir = 'downloads'
    os.makedirs(downloads_dir, exist_ok=True)
    
    # 创建已处理论文的记忆列表（用于重复检测）
    processed_papers_memory = []
    memory_lock = threading.Lock()  # 线程锁，确保线程安全
    
    def process_single_paper(paper, index, total):
        """处理单篇论文的完整流程：相关性分析 -> 摘要提取（串联）"""
        paper_id = paper.get('_paper_id', f"paper_{index}")
        
        try:
            print(f"\n处理论文 {index}/{total}: {paper.get('title', '')[:60]}...")
            debug_logger.log(f"处理论文 {index}/{total}: {paper.get('title', '')[:60]}")
            
            # 获取当前已处理的论文列表（创建副本，避免在分析过程中列表被修改）
            with memory_lock:
                current_processed = list(processed_papers_memory)
            
            # 使用相关性处理器处理论文
            # 动态读取LOCAL_MODE，确保使用最新的环境变量
            current_local_mode = os.getenv('LOCAL', '0') == '1'
            result = process_relevance_and_extraction(
                paper, 
                processed_papers=current_processed,
                on_paper_updated=on_paper_updated,
                on_paper_removed=on_paper_removed,
                on_paper_ready_for_confirmation=on_paper_ready_for_confirmation,
                research_keywords=research_keywords,
                expanded_keywords=expanded_keywords,
                is_local_mode=current_local_mode  # 传递本地模式标志
            )
            
            # 如果处理成功，添加到已处理列表
            if result:
                final_title = result.get('title', paper.get('title', ''))
                final_url = result.get('link', paper.get('link', ''))
                with memory_lock:
                    processed_papers_memory.append({
                        'title': final_title,
                        'link': final_url
                    })
            
            return result
            
        except Exception as e:
            print(f"  ✗ 处理论文时出错: {str(e)}")
            debug_logger.log(f"处理论文时出错: {paper.get('title', '')[:60]} - {str(e)}", "ERROR")
            
            # 即使出错，也添加到已处理列表（避免重复处理）
            try:
                with memory_lock:
                    processed_papers_memory.append({
                        'title': paper.get('title', ''),
                        'link': paper.get('link', '')
                    })
            except:
                pass
            
            return None
    
    # 并行处理多篇论文（每篇论文内部串联执行）
    relevant_papers = []
    
    # 使用线程池并行处理多篇论文
    # max_workers设置为3，避免过多线程
    max_workers = 3
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for i, paper in enumerate(all_papers, 1):
            # 确保每篇论文都有_paper_id（使用pending时的ID）
            if '_paper_id' not in paper:
                paper['_paper_id'] = f"paper_pending_{i}"
            future = executor.submit(process_single_paper, paper, i, len(all_papers))
            futures.append((future, paper['_paper_id']))  # 保存paper_id用于匹配
        
        # 收集结果
        for future, original_paper_id in futures:
            try:
                result = future.result()
                if result:  # 如果返回了论文对象，说明是相关论文
                    # 使用原始的paper_id，确保状态更新能匹配到正确的论文
                    if '_paper_id' not in result:
                        result['_paper_id'] = original_paper_id
                    relevant_papers.append(result)
            except Exception as e:
                print(f"  ✗ 获取论文处理结果时出错: {str(e)}")
                debug_logger.log(f"获取论文处理结果时出错: {str(e)}", "ERROR")
    
    print(f"\n✓ 找到 {len(relevant_papers)} 篇相关论文")
    debug_logger.log(f"找到 {len(relevant_papers)} 篇相关论文")
    
    if not relevant_papers:
        print("\n没有找到相关论文")
        debug_logger.log("没有找到相关论文")
        debug_logger.close()
        return
    
    # 4. 处理摘要（本地模式或已提取的摘要）
    # 动态读取LOCAL_MODE，确保使用最新的环境变量
    current_local_mode = os.getenv('LOCAL', '0') == '1'
    if current_local_mode:
        # 本地模式：摘要已在相关性分析阶段从CSV读取并显示确认框，这里只需要检查是否有遗漏
        print("\n本地模式：检查CSV摘要读取结果...")
        debug_logger.log_separator("CSV摘要检查")
        for i, paper in enumerate(relevant_papers, 1):
            if 'full_abstract' not in paper or not paper.get('full_abstract'):
                print(f"\n论文 {i}/{len(relevant_papers)}: {paper['title'][:50]}... CSV中未找到摘要，可以手动添加摘要")
                debug_logger.log(f"论文 {i}: {paper['title'][:50]} CSV中未找到摘要", "WARNING")
    else:
        # 邮件模式：摘要已在并行处理中提取，这里只需要检查是否有遗漏
        print("\n检查摘要提取结果...")
        debug_logger.log_separator("摘要提取检查")
        for i, paper in enumerate(relevant_papers, 1):
            if 'full_abstract' not in paper or not paper.get('full_abstract'):
                print(f"\n论文 {i}/{len(relevant_papers)}: {paper['title'][:50]}... 摘要提取失败，可以手动添加摘要")
                debug_logger.log(f"论文 {i}: {paper['title'][:50]} 摘要提取失败", "WARNING")
    
    # 5. 等待用户确认所有论文的摘要（批量确认）
    if get_confirmed_abstracts:
        log('info', '所有论文的摘要提取已完成，等待用户确认...')
        log('info', f'共有 {len(relevant_papers)} 篇论文需要确认摘要')
        log('info', '用户可以在前端编辑摘要，确认无误后点击"等待人工确认"按钮继续处理')
        
        # 批量获取所有确认后的摘要（阻塞等待用户点击确认按钮）
        confirmed_abstracts_dict = get_confirmed_abstracts()  # 返回 {paper_id: abstract_text}
        
        log('info', f'用户已确认 {len(confirmed_abstracts_dict)} 篇论文的摘要')
        
        # 更新每篇论文的摘要（使用确认后的摘要）
        for paper in relevant_papers:
            paper_id = paper.get('_paper_id', '')
            if paper_id in confirmed_abstracts_dict:
                confirmed_abstract = confirmed_abstracts_dict[paper_id]
                if confirmed_abstract and confirmed_abstract.strip():
                    paper['full_abstract'] = confirmed_abstract.strip()
                    log('info', f'论文 "{paper.get("title", "")[:50]}" 使用确认后的摘要')
                else:
                    # 用户确认的摘要为空，设置为None（后续会被跳过）
                    paper['full_abstract'] = None
                    log('warning', f'论文 "{paper.get("title", "")[:50]}" 确认的摘要为空，将跳过处理')
            else:
                # 如果论文不在确认字典中，使用原始摘要（如果有）
                original_abstract = paper.get('full_abstract')
                if original_abstract and original_abstract.strip():
                    log('info', f'论文 "{paper.get("title", "")[:50]}" 未在确认列表中，使用原始摘要')
                else:
                    paper['full_abstract'] = None
                    log('warning', f'论文 "{paper.get("title", "")[:50]}" 未确认且无原始摘要，将跳过处理')
    
    # 6. 使用 CrewAI 处理论文：验证 + 翻译 + 评审
    print("\n正在使用AI处理论文（验证 + 翻译 + 评审）...")
    debug_logger.log_separator("翻译和评审处理")
    
    # 筛选出有摘要的论文（包括确认后的和原始的）
    papers_with_abstract = [p for p in relevant_papers if p.get('full_abstract') is not None and p.get('full_abstract').strip()]
    papers_without_abstract = [p for p in relevant_papers if not (p.get('full_abstract') and p.get('full_abstract').strip())]
    
    print(f"  有摘要的论文: {len(papers_with_abstract)} 篇，将进行翻译和评审")
    print(f"  无摘要的论文: {len(papers_without_abstract)} 篇，将跳过处理")
    debug_logger.log(f"有摘要的论文: {len(papers_with_abstract)} 篇")
    debug_logger.log(f"无摘要的论文: {len(papers_without_abstract)} 篇")
    
    # 只处理有摘要的论文
    for i, paper in enumerate(papers_with_abstract, 1):
        print(f"\n处理 {i}/{len(papers_with_abstract)}: {paper['title'][:50]}...")
        debug_logger.log_paper_info(paper, index=i)
        
        paper_id = paper.get('_paper_id', f"paper_{i}")
        
        # 使用论文的摘要（已经过确认或使用原始摘要）
        full_abstract = paper.get('full_abstract')
        
        if not full_abstract or not full_abstract.strip():
            log('warning', f'论文 {paper.get("title", "")[:50]} 没有摘要，跳过处理')
            continue
        
        # 确定摘要来源类型
        source_type = "PDF" if paper.get('is_pdf', False) else "网页"
        
        # 使用 CrewAI 框架处理（包含验证、翻译、评审）
        result = process_paper_with_crewai(paper, full_abstract, source_type, on_paper_updated=on_paper_updated, expanded_keywords=expanded_keywords)
        
        # 更新论文信息
        paper['translated_content'] = result['translated_content']
        paper['review'] = result['review']
        paper['score'] = result['score']
        paper['score_details'] = result['score_details']
        paper['is_high_value'] = result['is_high_value']
        paper['original_english_abstract'] = result.get('original_english_abstract', full_abstract)  # 保存英文原文
        paper['multi_model_results'] = result.get('multi_model_results', {})  # 保存多模型评审结果
        paper['summary_result'] = result.get('summary_result')  # 保存汇总结果（如果有多个模型）
        
        print(f"  ✓ 完成 - 评分: {paper['score']:.2f}/4.0", end="")
        if paper['is_high_value']:
            print(" [高价值论文 ⭐]")
        else:
            print()
        
        # 发送论文更新回调
        if on_paper_updated:
            paper_id = paper.get('_paper_id', f"paper_{i}")
            
            # 判断处理是否成功：检查是否包含错误信息
            translated_content = paper.get('translated_content', '')
            review_content = paper.get('review', '')
            
            # 定义错误关键词
            error_keywords = [
                '摘要验证失败',
                '翻译失败',
                '评审失败',
                '摘要提取失败',
                '无法处理'
            ]
            
            # 检查是否包含错误信息
            is_error = any(keyword in translated_content or keyword in review_content 
                          for keyword in error_keywords)
            
            # 根据处理结果设置状态
            paper_status = 'error' if is_error else 'success'
            
            on_paper_updated(paper_id, {
                'status': paper_status,
                'abstract': translated_content,
                'review': review_content,
                'score': paper.get('score', 0.0),
                'score_details': paper.get('score_details', {}),
                'is_high_value': paper.get('is_high_value', False),
                'title': paper.get('title', ''),
                'link': paper.get('link', ''),
                'multi_model_results': paper.get('multi_model_results', {}),  # 包含多模型评审结果
                'summary_result': paper.get('summary_result')  # 包含汇总结果（如果有多个模型）
            })
    
    # 为摘要提取失败的论文也显示可编辑摘要框（让用户手动添加）
    for paper in papers_without_abstract:
        paper_id = paper.get('_paper_id', '')
        if not paper_id:
            continue
        # 发送准备确认消息，让用户手动添加摘要
        if on_paper_ready_for_confirmation:
            on_paper_ready_for_confirmation(paper_id, {
                'title': paper.get('title', ''),
                'abstract': '',  # 空摘要，让用户手动输入
                'link': paper.get('link', '')
            })
        # 更新状态
        if on_paper_updated:
            on_paper_updated(paper_id, {
                'status': 'abstract_extraction_failed',
                'status_text': '摘要提取失败，可以手动添加摘要',
                'title': paper.get('title', '')
            })
    
    # 标记摘要提取失败的论文
    for idx, paper in enumerate(papers_without_abstract, 1):
        paper['translated_content'] = "摘要提取失败，无法处理"
        paper['review'] = "摘要提取失败，无法处理"
        paper['score'] = 0.0
        paper['score_details'] = {}
        paper['is_high_value'] = False
        
        # 发送论文更新回调（摘要提取失败，状态为error）
        if on_paper_updated:
            paper_id = paper.get('_paper_id', f"paper_failed_{idx}")
            on_paper_updated(paper_id, {
                'status': 'error',
                'abstract': "摘要提取失败，无法处理",
                'score': 0.0
            })
    
    # 6. 生成日报（只包含成功处理的论文）
    # 只包含成功处理的论文（有review和score_details的）
    successful_papers = [p for p in relevant_papers if p.get('review') and p.get('score_details') and 
                        p.get('review') != "摘要提取失败，无法处理"]
    
    if successful_papers:
        print("\n生成日报...")
        report = generate_daily_report(successful_papers)
        
        # 7. 保存报告（Markdown 格式，命名方式与理想格式一致）
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        
        # 文件命名格式：Robotics_Academic_Daily_YYYYMMDD .md（注意有空格）
        filename = f"{output_dir}/Robotics_Academic_Daily_{datetime.now().strftime('%Y%m%d')}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ 报告已保存到: {filename}")
        # 发送文件生成回调
        if on_file_generated:
            file_size = os.path.getsize(filename) if os.path.exists(filename) else 0
            on_file_generated({
                'name': os.path.basename(filename),
                'path': filename,
                'type': 'md',
                'size': f"{file_size / 1024:.1f} KB",
                'time': datetime.now().strftime('%H:%M:%S')
            })
    else:
        print("\n没有成功处理的论文，不生成日报")
        debug_logger.log("没有成功处理的论文，不生成日报", "INFO")
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        report = None  # 标记没有报告
    
    # 7.1. 如果配置了备份路径，同时保存到备份目录（仅当有报告时）
    if report and BACKUP_DIR:
        try:
            os.makedirs(BACKUP_DIR, exist_ok=True)
            backup_filename = os.path.join(BACKUP_DIR, f"Robotics_Academic_Daily_{datetime.now().strftime('%Y%m%d')}.md")
            with open(backup_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✓ 报告已另存到: {backup_filename}")
        except Exception as e:
            logging.warning(f"保存到备份目录失败: {str(e)}")
            print(f"⚠ 警告: 无法保存到备份目录: {str(e)}")
    
    # 8. 导出所有符合关键词的论文到CSV（包含处理结果）
    csv_file = export_all_papers_to_csv(relevant_papers, output_dir)
    # 发送CSV文件生成回调
    if csv_file and on_file_generated:
        file_size = os.path.getsize(csv_file) if os.path.exists(csv_file) else 0
        on_file_generated({
            'name': os.path.basename(csv_file),
            'path': csv_file,
            'type': 'csv',
            'size': f"{file_size / 1024:.1f} KB",
            'time': datetime.now().strftime('%H:%M:%S')
        })
    
    # 关闭调试日志
    debug_logger.close()
    
    # 清理异步资源（修复 LiteLLM 异步客户端警告）
    try:
        import litellm
        import asyncio
        # 尝试关闭 LiteLLM 的异步客户端
        if hasattr(litellm, 'close_litellm_async_clients'):
            try:
                # 尝试获取现有的事件循环
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("Event loop is closed")
                except RuntimeError:
                    # 如果没有事件循环或已关闭，创建新的
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # 运行清理协程
                if not loop.is_closed():
                    loop.run_until_complete(litellm.close_litellm_async_clients())
                    # 清理未完成的任务
                    pending = asyncio.all_tasks(loop)
                    if pending:
                        for task in pending:
                            task.cancel()
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    # 关闭事件循环
                    loop.close()
            except Exception as e:
                # 如果清理失败，记录但不影响程序退出
                logging.debug(f"清理 LiteLLM 异步客户端时出错: {str(e)}")
    except (ImportError, AttributeError):
        # 如果 litellm 不可用，忽略
        pass
    
    # 输出最终报告（如果有成功处理的论文）
    if 'report' in locals() and report:
        print("\n" + "=" * 80)
        print(report)


if __name__ == "__main__":
    try:
        main()
    finally:
        # 确保在程序退出前清理所有异步资源
        try:
            import litellm
            import asyncio
            if hasattr(litellm, 'close_litellm_async_clients'):
                try:
                    # 创建新的事件循环来执行清理
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(litellm.close_litellm_async_clients())
                    finally:
                        # 清理并关闭事件循环
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        loop.close()
                except Exception:
                    # 忽略清理错误，确保程序能正常退出
                    pass
        except (ImportError, AttributeError):
            pass

