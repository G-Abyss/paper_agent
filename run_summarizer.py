#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Summarizer - 自动总结Google学术邮件推送
"""

import os
import imaplib
import email
from email.header import decode_header
import re
from datetime import datetime, timedelta
import yaml
from dotenv import load_dotenv
import ollama
from bs4 import BeautifulSoup
import time
import ssl
from crewai import Agent, Task, Crew, LLM
import logging

# 禁用 CrewAI 遥测（可选）
os.environ['CREWAI_TELEMETRY_OPT_OUT'] = 'true'
os.environ['OTEL_SDK_DISABLED'] = 'true'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 加载环境变量
load_dotenv()

# 配置
# GMAIL_USER = os.getenv('GMAIL_USER')
# GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
QMAIL_USER = os.getenv('QMAIL_USER')
QMAIL_PASSWORD = os.getenv('QMAIL_PASSWORD')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:32b')
MAX_EMAILS = int(os.getenv('MAX_EMAILS', 20))
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

# 设置环境变量（CrewAI 通过 LiteLLM 连接 Ollama 需要这些）
os.environ['OLLAMA_API_BASE'] = OLLAMA_BASE_URL
if not os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = 'ollama'  # 占位符，实际不使用

# 初始化 CrewAI LLM
# 关键：模型名称必须包含 "ollama/" 前缀
llm_model_name = f"ollama/{OLLAMA_MODEL}" if not OLLAMA_MODEL.startswith("ollama/") else OLLAMA_MODEL

logging.info(f"初始化 CrewAI LLM: model={llm_model_name}, base_url={OLLAMA_BASE_URL}")

llm = LLM(
    model=llm_model_name,
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"  # Ollama 不需要真实的 API key
)

# 加载关键词
with open('keywords.yaml', 'r', encoding='utf-8') as f:
    KEYWORDS = yaml.safe_load(f)

HIGH_PRIORITY_KEYWORDS = [kw.lower() for kw in KEYWORDS['high_priority']]
RELATED_KEYWORDS = [kw.lower() for kw in KEYWORDS['related']]


def connect_gmail(max_retries=3, retry_delay=5):
    """连接Gmail IMAP服务器，带重试机制"""
    print("正在连接Gmail...")
    
    for attempt in range(max_retries):
        try:
            # 创建 SSL 上下文
            context = ssl.create_default_context()
            
            # 使用超时设置连接
            # mail = imaplib.IMAP4_SSL("imap.gmail.com", port=993, ssl_context=context)
            # mail.sock.settimeout(30)  # 设置30秒超时
            
            # mail.login(GMAIL_USER, GMAIL_PASSWORD)
            # print("✓ Gmail连接成功")

            mail = imaplib.IMAP4_SSL("imap.qq.com", port=993, ssl_context=context)
            mail.sock.settimeout(30)  # 设置30秒超时
            
            mail.login(QMAIL_USER, QMAIL_PASSWORD)
            print("✓ QQmail连接成功")
            return mail
            
        except (imaplib.IMAP4.error, ssl.SSLError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"连接失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"✗ Gmail连接失败，已重试 {max_retries} 次")
                raise Exception(f"无法连接到Gmail: {str(e)}")
    
    raise Exception("无法连接到Gmail")


def fetch_scholar_emails(mail, days=1):
    """获取Google学术推送邮件"""
    print(f"\n正在获取最近{days}天的Google学术推送...")
    
    # 选择收件箱
    mail.select("inbox")
    
    # 计算日期
    since_date = (datetime.now() - timedelta(days=days)).strftime("%d-%b-%Y")
    
    # 搜索Google学术邮件，以转发邮件发件人为ligen4073187@gmail.com，且转发邮件中的原始发件人地址为scholaralerts-noreply@google.com
    search_criteria = f'(FROM "ligen4073187@gmail.com" SINCE {since_date}) AND (HEADER FROM "scholaralerts-noreply@google.com")'
    status, messages = mail.search(None, search_criteria)
    
    email_ids = messages[0].split()
    print(f"✓ 找到 {len(email_ids)} 封邮件")
    
    return email_ids


def extract_paper_info(email_body):
    """从邮件中提取论文信息"""
    soup = BeautifulSoup(email_body, 'html.parser')
    
    papers = []
    
    # Google学术推送的结构通常包含多篇论文
    # 查找所有论文标题和链接
    for h3 in soup.find_all('h3'):
        title_link = h3.find('a')
        if title_link:
            title = title_link.get_text(strip=True)
            link = title_link.get('href', '')
            
            # 查找作者和摘要信息
            parent = h3.find_parent()
            if parent:
                text_content = parent.get_text()
                
                paper = {
                    'title': title,
                    'link': link,
                    'snippet': text_content[:500]  # 获取前500字符作为片段
                }
                papers.append(paper)
    
    return papers


def check_relevance(paper):
    """检查论文相关性"""
    text = (paper['title'] + ' ' + paper['snippet']).lower()
    
    # 检查高优先级关键词
    high_priority_matches = sum(1 for kw in HIGH_PRIORITY_KEYWORDS if kw in text)
    
    # 检查相关关键词
    related_matches = sum(1 for kw in RELATED_KEYWORDS if kw in text)
    
    # 计算相关性分数
    relevance_score = high_priority_matches * 2 + related_matches
    
    return relevance_score, high_priority_matches > 0


def create_translator_agent():
    """创建专业翻译 Agent"""
    return Agent(
        role="专业翻译专家",
        goal="将英文论文内容准确、专业地翻译成中文，确保专业术语的准确性和技术表达的清晰性",
        backstory="你是一位在机器人学、控制理论、遥操作、机器人动力学和力控领域拥有深厚专业背景的翻译专家。你擅长将英文学术论文翻译成中文，能够准确处理专业术语，保持技术描述的完整性和逻辑结构的清晰性。",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=3,
        max_execution_time=300
    )


def create_reviewer_agent():
    """创建专业评审 Agent"""
    return Agent(
        role="专业评审专家",
        goal="对论文进行专业评审，生成结构化总结并给出简洁的5分制评分（只输出一次）",
        backstory="你是一位在机器人学、控制理论、遥操作、机器人动力学和力控领域拥有丰富研究经验的评审专家。你能够从创新性、技术深度、相关性、实用性和研究质量等多个维度对论文进行客观、专业的评价。你总是简洁明了地输出结果，不会重复说明。",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        max_iter=2,  # 减少迭代次数，避免重复
        max_execution_time=300
    )


def create_translation_task(paper):
    """创建翻译任务"""
    return Task(
        description=(
            f"请将以下英文论文信息准确、专业地翻译成中文。\n\n"
            f"论文标题：\n{paper['title']}\n\n"
            f"论文片段（英文原文）：\n{paper['snippet']}\n\n"
            f"翻译要求：\n"
            f"1. 保持专业术语的准确性，使用该领域标准的中文术语\n"
            f"2. 确保技术描述的准确性和完整性\n"
            f"3. 保持原文的逻辑结构和表达风格\n"
            f"4. 如果遇到不确定的术语，请提供最可能的专业翻译\n\n"
            f"请直接输出翻译结果，不需要额外说明。"
        ),
        agent=create_translator_agent(),
        expected_output="翻译后的中文内容，保持原文的结构和逻辑"
    )


def create_review_task(paper, translated_content):
    """创建评审任务"""
    return Task(
        description=(
            f"请对以下论文进行专业评审，生成结构化总结并给出评分。\n\n"
            f"论文标题（英文）：\n{paper['title']}\n\n"
            f"论文内容（已翻译为中文）：\n{translated_content}\n\n"
            f"请按以下格式输出（不要重复说明评分规则，直接给出结果）：\n\n"
            f"**核心贡献**：（1-2句话说明主要创新点和贡献）\n\n"
            f"**技术方法**：（简述主要技术路线和方法）\n\n"
            f"**相关性分析**：（详细说明与遥操作/机器人动力学/力控/机器人控制的关系）\n\n"
            f"**技术价值**：（评估该论文的技术价值和潜在应用）\n\n"
            f"**值得关注的原因**：（为什么这篇论文重要，有哪些亮点）\n\n"
            f"**评分详情**（仅输出一次，JSON格式，包含评分理由）：\n"
            f'{{"创新性": 0.0-1.0, "技术深度": 0.0-1.0, "相关性": 0.0-1.0, "实用性": 0.0-1.0, "研究质量": 0.0-1.0, "总分": 0.0-5.0, "评分理由": "简要说明评分依据"}}\n\n'
            f"重要：只输出一次评分详情，评分理由必须包含在JSON中，不要重复说明评分规则或多次输出评分。"
        ),
        agent=create_reviewer_agent(),
        expected_output=(
            "评审报告包含：核心贡献、技术方法、相关性分析、技术价值、值得关注的原因，"
            "以及一个JSON格式的评分详情（包含各维度分数、总分和评分理由，不要重复输出）。"
        )
    )


def process_paper_with_crewai(paper):
    """
    使用 CrewAI 框架处理论文：翻译 + 评审
    返回处理结果字典
    """
    try:
        # 步骤1: 翻译
        print("  [步骤1/2] 专业翻译中...")
        translation_crew = Crew(
            agents=[create_translator_agent()],
            tasks=[create_translation_task(paper)],
            verbose=True,
            share_crew=False
        )
        translation_result = translation_crew.kickoff()
        translated_content = translation_result.raw.strip()
        
        # 步骤2: 评审
        print("  [步骤2/2] 专业评审和评分中...")
        review_crew = Crew(
            agents=[create_reviewer_agent()],
            tasks=[create_review_task(paper, translated_content)],
            verbose=True,
            share_crew=False
        )
        review_result = review_crew.kickoff()
        review_text = review_result.raw.strip()
        
        # 提取评分
        score_data = extract_score_from_review(review_text)
        
        return {
            'translated_content': translated_content,
            'review': review_text,
            'score': score_data.get('总分', 0.0),
            'score_details': score_data,
            'is_high_value': score_data.get('总分', 0.0) > 4.0
        }
    except Exception as e:
        logging.error(f"处理论文时出错: {str(e)}")
        return {
            'translated_content': f"翻译失败: {str(e)}",
            'review': f"评审失败: {str(e)}",
            'score': 0.0,
            'score_details': {},
            'is_high_value': False
        }


def extract_score_from_review(review_text):
    """从评审文本中提取评分信息"""
    import json
    import re
    
    score_data = {
        '创新性': 0.0,
        '技术深度': 0.0,
        '相关性': 0.0,
        '实用性': 0.0,
        '研究质量': 0.0,
        '总分': 0.0,
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
            # 更新分数数据，只更新存在的字段
            for key in score_data.keys():
                if key in parsed:
                    if isinstance(parsed[key], (int, float)):
                        score_data[key] = float(parsed[key])
                    elif isinstance(parsed[key], str) and key == '评分理由':
                        score_data[key] = parsed[key]
            return score_data
        except (json.JSONDecodeError, ValueError):
            pass
    
    # 方法2: 尝试提取JSON格式的评分（更宽松的匹配）
    json_patterns = [
        r'\{[^{}]*"总分"[^{}]*\}',
        r'\{[^{}]*"创新性"[^{}]*"技术深度"[^{}]*\}',
    ]
    for pattern in json_patterns:
        json_match = re.search(pattern, review_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                for key in score_data.keys():
                    if key in parsed:
                        if isinstance(parsed[key], (int, float)):
                            score_data[key] = float(parsed[key])
                        elif isinstance(parsed[key], str) and key == '评分理由':
                            score_data[key] = parsed[key]
                if score_data['总分'] > 0:
                    return score_data
            except (json.JSONDecodeError, ValueError):
                continue
    
    # 方法3: 如果JSON提取失败，尝试从文本中提取数字
    # 查找总分（支持多种格式）
    total_score_patterns = [
        r'总分[：:]\s*([0-9.]+)',
        r'综合评分[：:]\s*([0-9.]+)',
        r'评分[：:]\s*([0-9.]+)\s*[/／]\s*5',
        r'([0-9.]+)\s*[/／]\s*5\.0',
    ]
    for pattern in total_score_patterns:
        match = re.search(pattern, review_text)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 5:
                    score_data['总分'] = score
                    break
            except (ValueError, IndexError):
                continue
    
    # 查找各个维度的分数
    dimensions = ['创新性', '技术深度', '相关性', '实用性', '研究质量']
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
                    if 0 <= score <= 1:
                        score_data[dim] = score
                        break
                except (ValueError, IndexError):
                    continue
    
    # 如果没有找到总分，计算各维度之和
    if score_data['总分'] == 0.0:
        total = sum([score_data[dim] for dim in dimensions])
        if total > 0:
            score_data['总分'] = total
    
    return score_data


def generate_daily_report(relevant_papers):
    """生成原始日报（Markdown 格式，简洁版）"""
    report = []
    
    # 标题
    report.append(f"# 📚 机器人学术论文日报")
    report.append(f"**日期：** {datetime.now().strftime('%Y年%m月%d日')}")
    report.append("")
    
    # 按评分分类论文
    high_value_papers = [p for p in relevant_papers if p.get('is_high_value', False)]
    other_papers = [p for p in relevant_papers if not p.get('is_high_value', False)]
    
    # 按评分排序
    high_value_papers.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    other_papers.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    
    # 高价值论文（评分>4.0，需要进一步研究）
    if high_value_papers:
        report.append("## 🔥 高价值论文（评分>4.0，建议下载原文深入研究）")
        report.append("")
        
        for i, paper in enumerate(high_value_papers, 1):
            report.append(f"### {i}. {paper['title']}")
            report.append("")
            report.append(paper.get('review', paper.get('summary', '')))
            report.append("")
            
            # # 简洁的评分展示（合并评分详情和评分理由）
            # score = paper.get('score', 0.0)
            # score_details = paper.get('score_details', {})
            
            # # 使用表格展示评分（更直观），评分理由合并到表格最后
            # if score_details:
            #     report.append("**评分详情：**")
            #     report.append("")
            #     report.append("| 维度 | 分数 |")
            #     report.append("|------|------|")
            #     for dim in ['创新性', '技术深度', '相关性', '实用性', '研究质量']:
            #         if dim in score_details:
            #             dim_score = score_details[dim]
            #             # 用星星表示分数
            #             stars = "⭐" * int(dim_score * 5)
            #             report.append(f"| {dim} | {dim_score:.2f}/1.0 {stars} |")
            #     report.append(f"| **总分** | **{score:.2f}/5.0** |")
            #     # 将评分理由合并到表格最后
            #     if '评分理由' in score_details and score_details['评分理由']:
            #         report.append(f"| **评分理由** | {score_details['评分理由']} |")
            #     report.append("")
            
            report.append(f"🔗 [论文链接]({paper['link']})")
            report.append("")
            report.append("---")
            report.append("")
    
    # 其他相关论文
    if other_papers:
        report.append("## 📖 相关论文")
        report.append("")
        
        for i, paper in enumerate(other_papers, 1):
            report.append(f"### {i}. {paper['title']}")
            report.append("")
            report.append(paper.get('review', paper.get('summary', '')))
            report.append("")
            report.append(f"**评分：** {paper.get('score', 0.0):.2f}/5.0")
            report.append(f"🔗 [论文链接]({paper['link']})")
            report.append("")
            report.append("---")
            report.append("")
    
    # 统计信息（使用表格）
    report.append("## 📊 统计信息")
    report.append("")
    report.append("| 类别 | 数量 |")
    report.append("|------|------|")
    report.append(f"| 高价值论文（评分>4.0） | {len(high_value_papers)} 篇 |")
    report.append(f"| 其他相关论文 | {len(other_papers)} 篇 |")
    report.append(f"| **总计** | **{len(relevant_papers)} 篇** |")
    
    if high_value_papers:
        avg_score = sum(p.get('score', 0.0) for p in high_value_papers) / len(high_value_papers)
        report.append(f"| 高价值论文平均评分 | {avg_score:.2f}/5.0 |")
    
    report.append("")
    
    return "\n".join(report)




def main():
    """主程序"""
    print("=" * 80)
    print("Paper Summarizer - 学术论文自动总结系统")
    print("=" * 80)
    print()
    
    # 1. 连接Gmail
    mail = connect_gmail()
    
    # 2. 获取邮件
    email_ids = fetch_scholar_emails(mail, days=1)
    
    if not email_ids:
        print("\n没有找到新的学术推送邮件")
        mail.close()
        mail.logout()
        return
    
    # 3. 处理邮件
    all_papers = []
    
    try:
        for email_id in email_ids[:MAX_EMAILS]:
            print(f"\n处理邮件 {email_id.decode()}...")
            
            try:
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                
                if status != 'OK':
                    print(f"  警告: 无法获取邮件内容 (状态: {status})")
                    continue
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        # 获取邮件正文
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/html":
                                    body = part.get_payload(decode=True).decode()
                                    break
                        else:
                            body = msg.get_payload(decode=True).decode()
                        
                        # 提取论文信息
                        papers = extract_paper_info(body)
                        all_papers.extend(papers)
                        print(f"  提取到 {len(papers)} 篇论文")
                        
            except (imaplib.IMAP4.error, ssl.SSLError, OSError) as e:
                print(f"  警告: 处理邮件时出错: {str(e)}")
                # 尝试重新连接
                try:
                    mail.close()
                except:
                    pass
                try:
                    mail = connect_gmail()
                    mail.select("inbox")
                except Exception as reconnect_error:
                    print(f"  错误: 重新连接失败: {str(reconnect_error)}")
                    break
                continue
            except Exception as e:
                print(f"  警告: 处理邮件时出现未知错误: {str(e)}")
                continue
    finally:
        # 确保连接被正确关闭
        try:
            mail.close()
        except:
            pass
        try:
            mail.logout()
        except:
            pass
    
    print(f"\n总共提取到 {len(all_papers)} 篇论文")
    
    # 4. 筛选相关论文
    print("\n正在分析论文相关性...")
    relevant_papers = []
    
    for paper in all_papers:
        relevance_score, is_high_priority = check_relevance(paper)
        
        if relevance_score > 0:  # 至少匹配一个关键词
            paper['relevance_score'] = relevance_score
            paper['is_high_priority'] = is_high_priority
            relevant_papers.append(paper)
    
    print(f"✓ 找到 {len(relevant_papers)} 篇相关论文")
    
    if not relevant_papers:
        print("\n没有找到相关论文")
        return
    
    # 5. 使用 CrewAI 处理论文：翻译 + 评审
    print("\n正在使用AI处理论文（翻译 + 评审）...")
    for i, paper in enumerate(relevant_papers, 1):
        print(f"\n处理 {i}/{len(relevant_papers)}: {paper['title'][:50]}...")
        
        # 使用 CrewAI 框架处理
        result = process_paper_with_crewai(paper)
        
        # 更新论文信息
        paper['translated_content'] = result['translated_content']
        paper['review'] = result['review']
        paper['score'] = result['score']
        paper['score_details'] = result['score_details']
        paper['is_high_value'] = result['is_high_value']
        
        print(f"  ✓ 完成 - 评分: {paper['score']:.2f}/5.0", end="")
        if paper['is_high_value']:
            print(" [高价值论文 ⭐]")
        else:
            print()
    
    # 6. 生成日报
    print("\n生成日报...")
    report = generate_daily_report(relevant_papers)
    
    # 7. 保存报告（Markdown 格式）
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{output_dir}/paper_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ 报告已保存到: {filename}")
    print("\n" + "=" * 80)
    print(report)


if __name__ == "__main__":
    main()