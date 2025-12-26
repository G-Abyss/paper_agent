#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话与推理引擎 Agent
支持"研究员"和"导师"两种模式
"""

import logging
import json
from typing import Dict, List, Optional
from crewai import Agent, Task, Crew
from agents.base import (
    get_llm, 
    query_known_knowledge_tool, 
    query_unknown_knowledge_tool,
    get_paper_list_tool, 
    get_paper_details_tool, 
    get_paper_full_text_tool,
    get_papers_think_points_tool
)
from utils.web_search import search_web, crawl_url
from crewai.tools import tool

@tool("网络搜索工具")
def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    使用Firecrawl进行网络搜索，获取最新的网络信息。
    当知识库中没有相关信息或信息不完整时，可以使用此工具搜索补充信息。
    
    Args:
        query: 搜索查询关键词
        max_results: 最大返回结果数量（默认5）
    
    Returns:
        格式化的搜索结果字符串，包含标题、URL和摘要
    """
    results = search_web(query, max_results)
    if not results:
        return "未找到相关网络信息。"
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(
            f"{i}. {result.get('title', '无标题')}\n"
            f"   URL: {result.get('url', '')}\n"
            f"   摘要: {result.get('snippet', '')[:200]}...\n"
        )
    
    return "\n".join(formatted_results)

def create_researcher_agent(llm=None, web_search_enabled: bool = False):
    """创建研究员模式的Agent"""
    if llm is None:
        llm = get_llm()
    
    # 基础工具：知识库查询（按优先级排序）
    tools = [
        query_known_knowledge_tool,  # 优先：查询已知知识（笔记）
        query_unknown_knowledge_tool,  # 其次：查询未知知识（论文）
        get_paper_list_tool,
        get_paper_details_tool,
        get_paper_full_text_tool
    ]
    
    # 如果允许联网，添加网络搜索工具
    if web_search_enabled:
        tools.append(web_search_tool)
    
    agent = Agent(
        role="知识库研究员",
        goal="作为知识管理员，准确回答用户问题，提供真实可靠的信息并引用来源",
        backstory="""你是一位严谨的学术研究员，专门负责从知识库中检索和整理信息。

**重要：查询顺序必须严格遵守以下逻辑**：
1. **首先查询"已知知识"**：使用"查询已知知识工具"在标签为note的笔记中查找相关信息
   - 这些是用户已经掌握的知识，应该优先使用
   - 如果找到相关信息，优先基于这些内容回答
2. **如果已知知识中没有相关信息或信息不完整**：
   - 使用"查询未知知识工具"在非note标签的论文中查找补充信息
   - 这些是用户尚未掌握的知识，用于扩展理解
3. **如果知识库中都没有相关信息**：
   - 如果允许联网，进行网络查询补充信息
   - 如果无法联网，进行合理推理（基于已有知识）
4. **所有回答必须基于真实信息，并明确引用来源**：
   - 已知知识：标注为 (来源：笔记《标题》)
   - 未知知识：标注为 (来源：论文《标题》，第X页)
   - 网络信息：标注为 (来源：网页URL)
5. 如果无法找到可靠信息，明确告知用户，不要编造答案""",
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=llm
    )
    return agent

def create_mentor_agent(llm=None):
    """创建导师模式的Agent"""
    if llm is None:
        llm = get_llm()
    
    # 导师模式需要访问知识库，特别是think点
    tools = [
        query_known_knowledge_tool,  # 查询已知知识（笔记）
        query_unknown_knowledge_tool,  # 查询未知知识（论文，包含think点）
        get_papers_think_points_tool,  # 获取论文think点
        get_paper_list_tool,
        get_paper_details_tool,
        get_paper_full_text_tool
    ]
    
    agent = Agent(
        role="资深科研导师",
        goal="通过苏格拉底提问与深度反馈，评估用户认知，纠正知识错误，引导用户构建准确的科研思维",
        backstory="""你是一位经验丰富的科研导师。你的核心教学方法是：
1. **深度评估**：不仅提问，更要对用户的每一个回答进行专业评估（评判对错）。
2. **精准纠错**：如果用户的回答存在偏差、错误或理解不透彻，必须明确指出，并使用类比或已知知识（笔记）进行引导。
3. **知识内化**：通过总结用户已表达的观点，肯定其正确部分，分析其缺失部分。
4. **渐进式引导**：在用户回答后，先评判，再总结，最后提出针对性的改进问题。

**核心原则**：
- 严禁直接提供完整答案，必须让用户自己思考。
- 必须使用中文进行对话。
- **必须立即执行任务并输出结果，不要只停留在思考阶段**。
- 如果用户回答“不知道”，提供暗示（Hint）或相关概念的对比，而非直接给答案。

**每次回复必须包含三个部分：【导师点评】、【思维总结】、【引导提问】。**

### 响应结构要求：
1. **【导师点评】**：评价用户的回答，指出闪光点和具体的错误/不足。如果用户回答正确，给予鼓励；如果错误，明确指出错误点。
2. **【思维总结】**：总结用户目前的知识掌握情况，分析其掌握程度，指出其“认知盲点”。
3. **【引导提问】**：提出1-2个新的引导性问题，引导用户修正错误或深入探索。

**注意**：
- 不要直接给答案，即使是纠错也应通过引导和对比来进行。
- 当知识点梳理完整后，输出【完整报告】并提示可进行下一次“agent自我探索”。""",
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=get_llm()
    )
    return agent

def create_chat_task(user_message: str, mode: str = 'query', context: str = '', web_search_enabled: bool = False, is_first_topic: bool = False):
    """
    创建对话任务
    
    Args:
        user_message: 用户消息
        mode: 模式 ('query' 研究员模式 或 'explore' 导师模式)
        context: 上下文信息（历史对话、知识库内容等）
        web_search_enabled: 是否允许联网搜索
        is_first_topic: 是否是首次探索该话题（仅探索模式有效）
    """
    if mode == 'explore':
        # 导师模式：苏格拉底式提问
        # 判断是否是第一次对话或用户消息太短（可能是切换模式后的首次消息）
        is_first_message = is_first_topic or not context or len(context.strip()) < 50 or len(user_message.strip()) < 10
        
        if is_first_message:
            task_description = f"""用户刚刚切换到探索模式，用户消息是：{user_message}

**立即执行**：用中文询问用户："你对该知识点（{user_message}）已经有了哪些认知？" 或者 "你以前使用过这个知识点吗？你是如何理解的？"

**要求**：
- 必须使用中文
- 只问一个问题
- 不要给出任何答案、列表或资源
- 立即输出问题

**输出格式**：直接输出问题，例如："你对该知识点（{user_message}）已经有了哪些认知？" """
        else:
            task_description = f"""用户消息：{user_message}

**你的导师任务**：

1. **评判与评估**：
   - 仔细审阅用户的回答。
   - 在【导师点评】中明确指出回答中哪些是准确的，哪些是错误或模糊的。
   - **纠错重点**：如果用户理解有误，必须通过逻辑引导或类比来纠正他，而不是直接给答案。

2. **知识总结**：
   - 在【思维总结】中根据用户回答和已知知识（{context}），分析用户目前的知识盲区。

3. **深度启发**：
   - 在【引导提问】中基于评估结果提出 1-2 个新问题。

**重要规则**：
- 严禁直接给出完整答案（除非是整理后的【完整报告】）。
- 如果用户回答得很好，可以考虑输出【完整报告】。
- 必须使用中文。

**现在立即按以下格式输出反馈：**
### 🎓 导师点评
[评价并纠错]

### 📝 思维总结
[总结现状与盲点]

### 💡 引导提问
[提出新的启发性问题]"""
            expected_output = "输出包含导师点评、思维总结和引导提问的反馈，严禁直接给答案。"
    else:
        # 研究员模式：知识检索与回答
        search_hint = "如果知识库信息不足，可以使用网络搜索工具进行补充。" if web_search_enabled else "如果知识库信息不足，请明确告知用户。"
        
        task_description = f"""用户问题：{user_message}

请作为知识库研究员回答用户问题，**必须严格遵守以下查询顺序**：

**第一步：查询已知知识（优先）**
- 使用"查询已知知识工具"在标签为note的笔记中查找相关信息
- 这些是用户已经掌握的知识，应该优先使用
- 如果找到相关信息，优先基于这些内容回答，并标注为 (来源：笔记《标题》)

**第二步：查询未知知识（补充）**
- 如果已知知识中没有相关信息或信息不完整，使用"查询未知知识工具"在非note标签的论文中查找
- 这些是用户尚未掌握的知识，用于扩展和补充理解
- 如果找到相关信息，标注为 (来源：论文《标题》，第X页)

**第三步：网络搜索（可选）**
- 如果知识库中都没有相关信息或信息不完整：
  {search_hint}

**回答要求**：
- **所有回答必须基于真实信息，以夹注形式标注来源**
- 如果无法找到可靠信息，明确告知用户，不要编造答案
- 优先使用已知知识，其次使用未知知识，最后考虑网络搜索

上下文信息：
{context}"""
    
    agent = create_researcher_agent(web_search_enabled=web_search_enabled) if mode == 'query' else create_mentor_agent()
    
    if mode == 'explore':
        expected_output = """使用中文输出：
- 如果是第一次对话该话题：直接询问"你对该知识点已经有了哪些认知？"
- 否则：总结用户知识架构（1-2句话），然后提出1-2个引导性问题
必须立即输出结果，不要只停留在思考阶段。不要直接给出答案、论文列表或资源推荐。"""
    else:
        expected_output = "返回带引用的答案，优先使用已知知识，其次使用未知知识"
    
    task = Task(
        description=task_description,
        agent=agent,
        expected_output=expected_output
    )
    return task

def _get_papers_think_points(max_results: Optional[int] = None) -> str:
    """
    获取数据库中所有笔记（已知知识）的think点（内部函数，不通过tool装饰器）
    """
    try:
        from utils.vector_db import get_db_connection
        from psycopg2.extras import RealDictCursor
        
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 查询所有note来源的笔记，且think_points不为空的记录
            sql = """
                SELECT paper_id, title, think_points 
                FROM papers 
                WHERE source = 'note' 
                  AND think_points IS NOT NULL 
                  AND think_points != '[]'::jsonb
                  AND jsonb_array_length(think_points) > 0
                ORDER BY updated_at DESC
            """
            
            if max_results:
                sql += f" LIMIT {max_results}"
            
            cur.execute(sql)
            papers = cur.fetchall()
            
            if not papers:
                return "数据库中暂无包含think点的笔记（已知知识）。"
            
            # 格式化结果
            response_parts = [f"找到 {len(papers)} 条包含think点的笔记（已知知识）：\n"]
            for i, paper in enumerate(papers, 1):
                title = paper.get('title', '未知标题')
                think_points = paper.get('think_points', [])
                
                if isinstance(think_points, str):
                    try:
                        think_points = json.loads(think_points)
                    except:
                        think_points = []
                
                response_parts.append(f"\n[{i}] {title}")
                if think_points and isinstance(think_points, list):
                    for j, point in enumerate(think_points, 1):
                        if isinstance(point, str):
                            response_parts.append(f"   Think点 {j}: {point[:200]}...")
                response_parts.append("---")
            
            return "\n".join(response_parts)
            
        finally:
            from utils.vector_db import return_db_connection
            return_db_connection(conn)
            
    except Exception as e:
        logging.error(f"获取think点失败: {str(e)}")
        return f"获取think点出错: {str(e)}"

def agent_self_exploration() -> Dict:
    """
    Agent自我探索：基于缓存上下文，从已知知识（笔记）的think点中选择5个用户最感兴趣的点
    
    Returns:
        包含5个备选think点的字典
    """
    from utils.brain_context_utils import get_brain_context
    
    # 获取缓存上下文
    brain_context = get_brain_context()
    context_str = json.dumps(brain_context, ensure_ascii=False) if brain_context else "暂无认知边界"
    
    # 获取所有think点（从已知知识/笔记中提取，使用内部函数，不通过tool）
    think_points_data = _get_papers_think_points(max_results=50)  # 获取最多50条笔记的think点
    
    # 创建探索任务
    agent = create_mentor_agent()
    
    task_description = f"""你正在以用户的视角进行自我探索，需要从已知知识（笔记）的think点中选择5个用户最感兴趣的研究方向。

**用户当前的认知边界**：
{context_str}

**可用的think点数据（来自已知知识/笔记）**：
{think_points_data}

**你的任务**：
1. 分析用户的认知边界，了解用户已经掌握的知识领域
2. 浏览所有think点（这些think点来自用户的已知知识/笔记），找出与用户认知边界相关但用户尚未深入理解的方向
3. 选择5个用户最有可能感兴趣的研究方向（think点）
4. 将这5个方向整理成简洁明了的表述，作为备选选择提供给用户

**输出格式**：
请使用以下格式输出5个备选选择：

【备选研究方向】
1. [研究方向1的简洁描述]
2. [研究方向2的简洁描述]
3. [研究方向3的简洁描述]
4. [研究方向4的简洁描述]
5. [研究方向5的简洁描述]

请确保：
- 每个方向都是基于think点提炼的
- 方向描述简洁明了，易于理解
- 与用户认知边界相关，但又是用户尚未深入掌握的
- 使用中文输出"""
    
    task = Task(
        description=task_description,
        agent=agent,
        expected_output="使用中文输出5个备选研究方向，格式为【备选研究方向】+ 5个编号的选项"
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        max_iter=3
    )
    
    try:
        result = crew.kickoff()
        exploration_result = str(result)
        
        return {
            'success': True,
            'message': exploration_result,
            'type': 'self_exploration'
        }
    except Exception as e:
        logging.error(f"Agent自我探索失败: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

def process_chat(user_message: str, mode: str = 'query', web_search_enabled: bool = False) -> Dict:
    """
    处理用户对话
    
    Args:
        user_message: 用户消息
        mode: 模式 ('query' 研究员模式 或 'explore' 导师模式)
        web_search_enabled: 是否允许联网搜索
    
    Returns:
        包含回答和元数据的字典
    """
    from utils.chat_history import (
        add_message, get_context_string, compress_context,
        get_token_count, MAX_CONTEXT_TOKENS
    )
    from utils.brain_context_utils import get_brain_context
    
    # 对于探索模式，检查是否是首次启动探索模式
    is_first_explore_session = False
    is_first_explore_topic = False
    
    if mode == 'explore':
        from utils.chat_history import load_chat_history
        history = load_chat_history()
        # 检查历史记录中是否有探索模式的对话
        explore_messages = [msg for msg in history if msg.get('metadata', {}).get('mode') == 'explore']
        is_first_explore_session = len(explore_messages) == 0
        
        # 检查最后一条消息是否是自我探索结果（避免重复触发）
        last_message_is_exploration = False
        if history:
            last_msg = history[-1]
            if (last_msg.get('role') == 'assistant' and 
                last_msg.get('metadata', {}).get('type') == 'self_exploration'):
                last_message_is_exploration = True
        
        logging.info(f"探索模式检查: 历史记录总数={len(history)}, 探索模式消息数={len(explore_messages)}, 用户消息长度={len(user_message.strip())}, 是否首次探索={is_first_explore_session}, 最后一条是探索={last_message_is_exploration}")
        
        # 如果消息为空或很短，且不是重复触发，则进行agent自我探索
        if (not user_message or len(user_message.strip()) < 5) and not last_message_is_exploration:
            logging.info("触发agent自我探索（消息为空或很短）")
            # 先不添加用户消息到历史（因为这是系统自动触发的）
            try:
                exploration_result = agent_self_exploration()
                if exploration_result.get('success'):
                    # 将自我探索结果添加到历史
                    add_message('assistant', exploration_result['message'], {'mode': mode, 'type': 'self_exploration'})
                    logging.info("Agent自我探索成功")
                    return {
                        'success': True,
                        'message': exploration_result['message'],
                        'mode': mode,
                        'type': 'self_exploration',
                        'is_first_explore': True
                    }
                else:
                    # 如果自我探索失败，继续正常流程
                    logging.warning(f"Agent自我探索失败，继续正常对话流程: {exploration_result.get('error')}")
            except Exception as e:
                logging.error(f"Agent自我探索异常: {e}", exc_info=True)
                # 如果异常，继续正常流程
    
    # 对于探索模式，检测用户是否请求进行下一次自我探索
    if mode == 'explore':
        # 检测关键词：用户想探索新方向、进行下一次探索等
        exploration_keywords = ['探索新方向', '下一次探索', '新研究方向', '继续探索', '下一个', '新方向', '自我探索']
        user_msg_lower = user_message.lower()
        if any(keyword in user_message for keyword in exploration_keywords):
            # 用户请求进行下一次自我探索
            exploration_result = agent_self_exploration()
            if exploration_result.get('success'):
                # 将自我探索结果添加到历史
                add_message('assistant', exploration_result['message'], {'mode': mode, 'type': 'self_exploration'})
                return {
                    'success': True,
                    'message': exploration_result['message'],
                    'mode': mode,
                    'type': 'self_exploration',
                    'is_new_exploration': True
                }
            else:
                logging.warning(f"Agent自我探索失败: {exploration_result.get('error')}")
                # 继续正常流程
    
    # 添加用户消息到历史
    add_message('user', user_message, {'mode': mode})
    
    # 获取上下文
    context = get_context_string(max_tokens=MAX_CONTEXT_TOKENS - 500)  # 留500 tokens给回答
    
    # 获取认知边界（用于导师模式）
    brain_context = get_brain_context()
    brain_context_str = ""
    if mode == 'explore' and brain_context:
        brain_context_str = f"\n用户认知边界：{json.dumps(brain_context, ensure_ascii=False)}"
    
    # 组合上下文，添加首次话题标记
    full_context = context + brain_context_str
    if mode == 'explore' and is_first_explore_topic:
        full_context += "\n\n注意：这是用户首次在探索模式下提问，请先询问用户对该知识点的已有认知。"
    
    # 创建任务
    task = create_chat_task(
        user_message=user_message,
        mode=mode,
        context=full_context,
        web_search_enabled=web_search_enabled,
        is_first_topic=is_first_explore_topic if mode == 'explore' else False
    )
    
    # 创建Crew并执行
    agent = create_researcher_agent(web_search_enabled=web_search_enabled) if mode == 'query' else create_mentor_agent()
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        max_iter=3,  # 限制最大迭代次数，避免无限思考
        max_rpm=10  # 限制每分钟请求数
    )
    
    try:
        result = crew.kickoff()
        assistant_message = str(result)
        
        # 如果结果为空或只包含思考过程，尝试提取实际输出
        if not assistant_message or len(assistant_message.strip()) < 10:
            # 尝试从crew的执行结果中提取
            if hasattr(result, 'raw') and result.raw:
                assistant_message = str(result.raw)
            elif hasattr(result, 'output') and result.output:
                assistant_message = str(result.output)
            else:
                assistant_message = "抱歉，我还在思考中。请重新提问。"
        
        # 对于探索模式，检查是否是完整报告，如果是则调用生成工程师
        generated_note_path = None
        if mode == 'explore':
            # 检测完整报告的标志
            if '【完整报告】' in assistant_message or '完整报告' in assistant_message:
                import re
                
                # 尝试解析完整报告格式
                topic = None
                content = assistant_message
                tags = None
                
                # 尝试提取主题
                topic_match = re.search(r'主题[：:]\s*(.+?)(?:\n|$)', assistant_message)
                if topic_match:
                    topic = topic_match.group(1).strip()
                
                # 尝试提取标签建议
                tags_match = re.search(r'标签建议[：:]\s*(.+?)(?:\n|$)', assistant_message)
                if tags_match:
                    tags = tags_match.group(1).strip()
                
                # 如果没有提取到主题，使用用户消息
                if not topic:
                    topic = user_message[:50]
                    if len(user_message) > 50:
                        topic = user_message[:47] + "..."
                
                # 调用生成工程师
                try:
                    from agents.content_generator_agent import generate_note_from_content
                    note_result = generate_note_from_content(
                        topic=topic,
                        content=content,
                        tags=tags
                    )
                    
                    if note_result.get('success'):
                        generated_note_path = note_result.get('file_path', '')
                        assistant_message += f"\n\n✅ 已自动生成笔记文件：{generated_note_path}"
                        logging.info(f"自动生成笔记成功: {generated_note_path}")
                    else:
                        logging.warning(f"自动生成笔记失败: {note_result.get('error')}")
                except Exception as e:
                    logging.error(f"调用生成工程师失败: {e}", exc_info=True)
        
        # 添加助手回答到历史
        add_message('assistant', assistant_message, {'mode': mode})
        
        # 检查是否需要压缩上下文
        compress_context()
        
        return {
            'success': True,
            'message': assistant_message,
            'mode': mode,
            'generated_note': generated_note_path if generated_note_path else None
        }
    except Exception as e:
        logging.error(f"对话处理失败: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

