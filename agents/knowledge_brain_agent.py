import json
import logging
from crewai import Agent, Task
from agents.base import get_llm
from typing import List, Dict, Any, Optional
from crewai.tools import tool
from agents.base import rag_paper_query_tool, get_paper_list_tool
from utils.brain_context_utils import get_brain_context, save_brain_context, save_think_results

@tool("获取认知边界上下文工具")
def get_cognitive_context_tool() -> str:
    """获取用户当前的认知边界（缓存上下文），包含已知主题、核心观点和关注盲点。"""
    context = get_brain_context()
    return json.dumps(context, ensure_ascii=False, indent=2)

@tool("更新认知边界工具")
def update_cognitive_context_tool(known_topics: List[str], core_beliefs: List[str], blind_spots: List[str], summary: str) -> str:
    """更新用户的认知边界缓存。直接传入结构化的字段内容。"""
    try:
        new_context = {
            "known_topics": known_topics,
            "core_beliefs": core_beliefs,
            "blind_spots": blind_spots,
            "summary": summary
        }
        save_brain_context(new_context)
        return "认知边界更新成功。"
    except Exception as e:
        return f"更新失败: {e}"

@tool("论文深度初始化分析工具")
def paper_depth_analysis_tool(paper_id: str, think_points: List[str], contextual_summary: str) -> str:
    """将论文的深度分析结果（3个think点和上下文总结）存入数据库。"""
    try:
        save_think_results(paper_id, think_points, contextual_summary)
        return f"论文 {paper_id} 深度分析结果已存储。"
    except Exception as e:
        return f"存储失败: {e}"

def create_knowledge_brain_agent():
    """创建知识库大脑 Agent (Librarian)"""
    return Agent(
        role="知识库大脑管理员",
        goal=(
            "作为知识库的核心枢纽，管理用户的认知边界，为其他 Agent 提供精准的知识索引。"
            "维护‘已知’（笔记）与‘未知’（论文/邮件）的动态平衡，通过深度分析发现知识间的关联。"
        ),
        backstory=(
            "你是一位博学且严谨的数字图书馆馆长。你不仅管理着成千上万的文档，更重要的是你理解读者的心智模型。"
            "你通过阅读读者的笔记（已知知识）来构建一个‘认知缓存’。当你看到新的论文（未知知识）时，"
            "你会思考：‘这篇论文如何补充读者的现有知识？’、‘它是否挑战了读者的某个核心观点？’。"
            "你的任务是让知识有序，让搜索高效，让新旧知识产生化学反应。你从不生成新笔记，只负责整理和深度索引。"
        ),
        tools=[
            get_cognitive_context_tool,
            update_cognitive_context_tool,
            paper_depth_analysis_tool,
            rag_paper_query_tool,
            get_paper_list_tool
        ],
        allow_delegation=True, # 允许大脑调度其他 Agent 执行具体任务
        verbose=True,
        llm=get_llm()
    )

def create_brain_maintenance_task(notes_summary: str) -> Task:
    """创建大脑维护任务：更新认知边界"""
    return Task(
        description=(
            f"根据以下用户笔记的汇总内容，重新梳理并更新‘认知边界’缓存：\n\n"
            f"{notes_summary}\n\n"
            f"请识别并提取以下内容：\n"
            f"1. known_topics: 用户已经熟悉的领域或主题列表\n"
            f"2. core_beliefs: 用户在笔记中表现出的核心观点或立场\n"
            f"3. blind_spots: 用户提到感兴趣但尚未深入的领域\n"
            f"4. summary: 对用户当前知识体系的简短描述\n\n"
            f"完成后请调用‘更新认知边界工具’，直接传入上述四个字段的内容。"
        ),
        agent=create_knowledge_brain_agent(),
        expected_output="更新后的认知边界 JSON 结果"
    )

def create_absorb_note_task(note_title: str, note_content: str, current_context: str) -> Task:
    """创建吸收单篇笔记的任务：增量更新认知边界"""
    return Task(
        description=(
            f"你现在需要吸收一篇新的笔记内容，并将其整合到用户现有的‘认知边界’中。\n\n"
            f"当前已知认知边界：\n{current_context}\n\n"
            f"待吸收的新笔记：\n标题: {note_title}\n内容: {note_content}\n\n"
            f"请执行以下操作：\n"
            f"1. 分析新笔记是否引入了新的主题、改变了核心观点或填补了盲点。\n"
            f"2. 整合后调用‘更新认知边界工具’，**必须同时传入以下四个更新后的参数**：\n"
            f"   - known_topics: 更新后的已知主题列表\n"
            f"   - core_beliefs: 更新后的核心观点列表\n"
            f"   - blind_spots: 更新后的知识盲点列表\n"
            f"   - summary: 对整合新笔记后用户知识体系的最新简短描述\n\n"
            f"注意：保持信息的精炼，不要让列表无限膨胀，合并同类项。遇到 LaTeX 数学公式等特殊字符请保留原样。"
        ),
        agent=create_knowledge_brain_agent(),
        expected_output="包含四个必要字段的增量更新认知边界结果"
    )

def create_paper_think_task(paper_id: str, combined_summaries: str, cognitive_context: str) -> Task:
    """创建论文深度综合分析任务（基于各段总结）"""
    return Task(
        description=(
            f"针对论文 (ID: {paper_id}) 的各段要点进行深度初始化综合分析。\n\n"
            f"当前用户认知背景：\n{cognitive_context}\n\n"
            f"论文各章节/段落要点汇总：\n{combined_summaries}\n\n"
            f"请执行以下操作：\n"
            f"1. 结合全文要点，总结 3 个用户可能最感兴趣的‘think点’。这些点应该是该论文核心价值与用户现有知识最相关的结合部。\n"
            f"2. 生成一份高质量的‘上下文总结’ (contextual_summary)，突出该论文如何填补用户的知识盲点、挑战现有观点或提供了哪些值得参考的实验/方法论。\n"
            f"3. 调用‘论文深度初始化分析工具’保存结果。"
        ),
        agent=create_knowledge_brain_agent(),
        expected_output="包含 think 点和最终上下文总结的分析结果"
    )

def create_segment_summary_task(segment_text: str, segment_index: int, total_segments: int) -> Task:
    """创建论文段落提炼任务"""
    return Task(
        description=(
            f"你正在协助处理一篇长论文。这是第 {segment_index}/{total_segments} 个文本段落。\n\n"
            f"段落内容：\n{segment_text}\n\n"
            f"请精炼地总结该段落的核心内容（如提出的方法、实验结果、关键结论等）。"
            f"输出应保持客观、简练，为后续的综合分析提供高质量素材。"
        ),
        agent=create_knowledge_brain_agent(),
        expected_output="该段落的核心要点总结"
    )

