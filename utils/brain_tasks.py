import logging
import json
from crewai import Crew, Task
from agents.knowledge_brain_agent import (
    create_knowledge_brain_agent, 
    create_brain_maintenance_task, 
    create_paper_think_task
)
from utils.brain_context_utils import update_brain_context_from_notes, get_brain_context
from utils.vector_db import get_db_connection, return_db_connection
from psycopg2.extras import RealDictCursor

def trigger_brain_context_update(paper_id: str = None):
    """
    触发大脑认知边界更新。
    如果提供了 paper_id，则只同步该篇笔记；否则同步所有状态为 'processed' 的笔记。
    """
    try:
        conn = get_db_connection()
        notes = []
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            if paper_id:
                # 获取笔记基本信息，并检查 chunk 数量（识别 PDF 笔记）
                cur.execute("""
                    SELECT p.paper_id, p.title, p.content, p.source,
                    (SELECT count(*) FROM paper_chunks pc WHERE pc.paper_id = p.paper_id) as chunk_count
                    FROM papers p WHERE p.paper_id = %s
                """, (paper_id,))
                note = cur.fetchone()
                if note:
                    notes = [note]
            else:
                # 获取所有已处理但未同步的笔记
                cur.execute("""
                    SELECT p.paper_id, p.title, p.content, p.source,
                    (SELECT count(*) FROM paper_chunks pc WHERE pc.paper_id = p.paper_id) as chunk_count
                    FROM papers p WHERE source = 'note' ORDER BY updated_at ASC
                """)
                notes = cur.fetchall()
        finally:
            return_db_connection(conn)

        if not notes:
            return {"success": False, "message": "没有待同步的笔记"}

        agent = create_knowledge_brain_agent()
        
        logging.info(f"--- [大脑 Agent] 开始增量吸收笔记 (共 {len(notes)} 篇) ---")
        
        success_count = 0
        for i, note in enumerate(notes):
            # 每次循环都实时获取最新的上下文
            current_context = json.dumps(get_brain_context(), ensure_ascii=False)
            
            note_content = note['content'] or ""
            note_title = note['title']
            
            # 关键：如果是 PDF 笔记（内容为空但有 chunks），先执行深度提炼作为吸收内容
            if (not note_content or len(note_content.strip()) < 20) and note['chunk_count'] > 0:
                logging.info(f"检测到 PDF 类型笔记: {note_title}，正在进行多段预分析...")
                # 复用或改造深度分析逻辑，只需获取总结，不写 think_points
                summary_res = perform_pdf_note_distillation(note['paper_id'])
                if summary_res:
                    note_content = summary_res
                    logging.info(f"PDF 笔记深度提炼成功，长度: {len(note_content)}")
                else:
                    logging.warning(f"PDF 笔记提炼失败，将尝试直接吸收原始内容。")

            # 创建增量吸收任务
            from agents.knowledge_brain_agent import create_absorb_note_task
            task = create_absorb_note_task(note_title, note_content, current_context)
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True
            )
            
            logging.info(f"正在吸收第 {i+1}/{len(notes)} 篇笔记: {note_title}")
            result = crew.kickoff()
            
            success_count += 1
            
        logging.info("--- [大脑 Agent] 认知边界同步完成 ---")
        return {"success": True, "message": f"成功吸收了 {success_count} 篇笔记", "count": success_count}
    except Exception as e:
        logging.error(f"大脑增量更新失败: {e}")
        return {"success": False, "error": str(e)}

def perform_pdf_note_distillation(paper_id: str) -> str:
    """专门为笔记同步设计的 PDF 内容提炼逻辑（Map-Reduce）"""
    try:
        conn = get_db_connection()
        segments = []
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT chunk_text FROM paper_chunks WHERE paper_id = %s ORDER BY chunk_index", (paper_id,))
            chunks = cur.fetchall()
            chunk_size = 10 # 笔记提炼可以稍微大一点，加快速度
            for i in range(0, len(chunks), chunk_size):
                segment_text = "\n\n".join([c['chunk_text'] for c in chunks[i:i + chunk_size]])
                segments.append(segment_text)
        finally:
            return_db_connection(conn)
            
        if not segments: return None
            
        agent = create_knowledge_brain_agent()
        from agents.knowledge_brain_agent import create_segment_summary_task
        
        # 1. Map: 总结各段
        map_tasks = []
        for i, seg in enumerate(segments):
            map_tasks.append(create_segment_summary_task(seg, i + 1, len(segments)))
            
        # 2. Reduce: 产出最终全文提炼（专为吸收设计）
        reduce_task = Task(
            description="请汇总之前所有段落的总结，为这篇 PDF 笔记产出一份详尽的全文综述。这份综述将被用于更新用户的知识库‘认知边界’，请确保涵盖所有核心观点和技术细节。",
            agent=agent,
            context=map_tasks,
            expected_output="高质量的全文综述内容"
        )
        
        crew = Crew(agents=[agent], tasks=map_tasks + [reduce_task], verbose=True)
        result = crew.kickoff()
        
        # CrewAI 2.0+ 返回的是 CrewOutput 对象，直接转字符串
        return str(result)
    except Exception as e:
        logging.error(f"PDF 笔记提炼异常: {e}")
        return None

def trigger_paper_depth_analysis(paper_id: str):
    """触发论文的深度分析（多段理解架构）"""
    try:
        # 1. 获取认知上下文
        context = get_brain_context()
        context_str = json.dumps(context, ensure_ascii=False)
        
        # 2. 获取论文全量内容并分段
        conn = get_db_connection()
        segments = []
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            # 获取所有块
            cur.execute("SELECT chunk_text FROM paper_chunks WHERE paper_id = %s ORDER BY chunk_index", (paper_id,))
            chunks = cur.fetchall()
            
            # 每 8 个块分为一个段落 (约 4000-6000 tokens)
            chunk_size = 8
            for i in range(0, len(chunks), chunk_size):
                segment_text = "\n\n".join([c['chunk_text'] for c in chunks[i:i + chunk_size]])
                segments.append(segment_text)
                
            logging.info(f"读取到论文 {paper_id} 的 {len(chunks)} 个文本块，划分为 {len(segments)} 个段落进行分段理解")
        finally:
            return_db_connection(conn)
            
        if not segments:
            return {"success": False, "message": "未找到论文内容"}
            
        agent = create_knowledge_brain_agent()
        tasks = []
        
        # 3. 创建分段总结任务 (Map)
        from agents.knowledge_brain_agent import create_segment_summary_task
        for i, segment_text in enumerate(segments):
            task = create_segment_summary_task(segment_text, i + 1, len(segments))
            tasks.append(task)
            
        # 4. 创建最终综合任务 (Reduce)
        from agents.knowledge_brain_agent import create_paper_think_task
        final_analysis_task = create_paper_think_task(paper_id, "请参考之前所有任务的总结内容", context_str)
        final_analysis_task.context = tasks # 关键：引入之前所有总结任务作为上下文
        tasks.append(final_analysis_task)
        
        # 5. 执行 Crew
        crew = Crew(
            agents=[agent],
            tasks=tasks,
            verbose=True
        )
        
        logging.info(f"--- [大脑 Agent] 开始多段分析论文 {paper_id} (共 {len(segments)} 段) ---")
        result = crew.kickoff()
        logging.info(f"--- [大脑 Agent] 论文 {paper_id} 深度分析完成 ---")
        
        return {"success": True, "result": result}
    except Exception as e:
        logging.error(f"论文深度分析失败: {e}")
        return {"success": False, "error": str(e)}

