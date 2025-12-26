#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成工程师 Agent
负责将对话Agent的完整知识点整理成Obsidian格式的markdown笔记
"""

import logging
import json
import os
from typing import Dict, Optional
from datetime import datetime
from crewai import Agent, Task, Crew
from crewai.tools import tool
from agents.base import get_llm
from utils.file_utils import sanitize_filename

def get_note_path() -> Optional[str]:
    """获取笔记路径"""
    try:
        import json
        settings_file = os.path.join(os.getcwd(), 'data', 'note_settings.json')
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                return settings.get('note_path')
    except Exception as e:
        logging.error(f"读取笔记路径失败: {e}")
    return None

@tool("创建Obsidian笔记工具")
def create_obsidian_note_tool(topic: str, content: str, tags: Optional[str] = None) -> str:
    """
    在笔记路径的paper_agent文件夹中创建Obsidian格式的markdown笔记。
    
    输入参数：
    - topic (字符串，必需): 笔记主题/标题
    - content (字符串，必需): 笔记内容（markdown格式）
    - tags (字符串，可选): 标签，多个标签用逗号分隔，例如："机器人学,控制理论,力控"
    
    返回：
    - 创建的文件路径，如果失败则返回错误信息
    """
    try:
        note_path = get_note_path()
        if not note_path or not os.path.exists(note_path):
            return f"错误：笔记路径未设置或不存在。请先在设置中配置笔记路径。"
        
        # 创建paper_agent子文件夹
        paper_agent_dir = os.path.join(note_path, 'paper_agent')
        os.makedirs(paper_agent_dir, exist_ok=True)
        
        # 清理文件名，确保符合文件系统要求
        safe_filename = sanitize_filename(topic, max_length=100)
        if not safe_filename:
            safe_filename = "未命名笔记"
        
        # 确保文件名以.md结尾
        if not safe_filename.endswith('.md'):
            safe_filename += '.md'
        
        file_path = os.path.join(paper_agent_dir, safe_filename)
        
        # 如果文件已存在，添加时间戳
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_part = safe_filename[:-3]  # 去掉.md
            safe_filename = f"{name_part}_{timestamp}.md"
            file_path = os.path.join(paper_agent_dir, safe_filename)
        
        # 构建Obsidian格式的markdown内容
        # 添加YAML frontmatter（Obsidian支持）
        frontmatter = f"---\n"
        frontmatter += f"title: {topic}\n"
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',')]
            frontmatter += f"tags: {json.dumps(tag_list, ensure_ascii=False)}\n"
        frontmatter += f"created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        frontmatter += f"source: paper_agent_generated\n"
        frontmatter += f"---\n\n"
        
        # 组合完整内容
        full_content = frontmatter + content
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        # 返回相对路径（相对于笔记根目录）
        rel_path = os.path.relpath(file_path, note_path).replace('\\', '/')
        
        logging.info(f"成功创建Obsidian笔记: {rel_path}")
        
        # 尝试自动触发笔记扫描（如果笔记导入系统正在运行）
        try:
            # 这里可以添加自动触发扫描的逻辑
            # 目前先记录日志，后续可以通过API或消息队列触发
            logging.info(f"笔记已创建，建议在笔记系统中手动触发扫描以导入到知识库: {rel_path}")
        except Exception as e:
            logging.warning(f"自动触发笔记扫描失败（可手动触发）: {e}")
        
        return f"成功创建笔记文件: {rel_path}。请在笔记系统中触发扫描以导入到知识库。"
        
    except Exception as e:
        error_msg = f"创建笔记失败: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return error_msg

def create_content_generator_agent(llm=None):
    """创建生成工程师Agent"""
    if llm is None:
        llm = get_llm()
    
    agent = Agent(
        role="生成工程师",
        goal="将完整的知识点内容整理成符合Obsidian规范的markdown笔记，并保存到paper_agent文件夹",
        backstory="""你是一位专业的笔记生成工程师，专门负责将知识内容整理成结构化的Obsidian格式笔记。

**你的职责**：
1. 接收来自对话Agent的完整知识点内容
2. 将其整理成结构清晰、符合Obsidian规范的markdown格式
3. 在笔记路径的paper_agent文件夹中创建笔记文件
4. 确保生成的内容可以作为"已知知识"存储到系统中

**Obsidian笔记规范**：
1. **YAML Frontmatter**：必须包含title、tags、created、source等元数据
2. **标题层级**：使用#、##、###等合理组织内容
3. **双向链接**：使用[[链接]]格式创建内部链接
4. **标签**：使用#标签格式或YAML frontmatter中的tags
5. **代码块**：使用```代码块```格式
6. **列表**：使用-或*创建无序列表，使用1.创建有序列表
7. **引用**：使用>创建引用块
8. **数学公式**：使用$或$$包裹LaTeX公式

**内容要求**：
- 结构清晰，层次分明
- 使用中文编写
- 包含必要的概念解释、原理说明、应用案例
- 添加相关的双向链接，连接到其他相关概念
- 添加合适的标签，便于分类和检索
- 确保内容完整、准确、易于理解

**文件命名**：
- 使用主题名称作为文件名
- 文件名要简洁明了，避免特殊字符
- 如果文件已存在，自动添加时间戳

**重要**：
- 必须使用create_obsidian_note_tool工具来创建笔记
- 确保内容格式正确，能在Obsidian中正常显示
- 生成的内容将成为系统的"已知知识"，要保证质量""",
        verbose=True,
        allow_delegation=False,
        tools=[create_obsidian_note_tool],
        llm=llm
    )
    return agent

def generate_note_from_content(topic: str, content: str, tags: Optional[str] = None) -> Dict:
    """
    从知识点内容生成Obsidian笔记
    
    Args:
        topic: 笔记主题/标题
        content: 知识点内容（可能是对话Agent的完整报告）
        tags: 标签（可选）
    
    Returns:
        包含成功状态和文件路径的字典
    """
    agent = create_content_generator_agent()
    
    task_description = f"""请将以下知识点内容整理成Obsidian格式的markdown笔记：

**主题**：{topic}

**原始内容**：
{content}

**整理要求**：
1. **结构组织**：
   - 使用#作为主标题（主题名称）
   - 使用##作为主要章节标题（如：概念定义、原理说明、应用案例、相关概念等）
   - 使用###作为小节标题
   - 确保层次清晰，逻辑连贯

2. **内容处理**：
   - 保留原始内容的核心信息
   - 如果内容中提到其他概念（如"阻抗控制"、"导纳控制"等），使用[[概念名]]格式创建双向链接
   - 使用列表（- 或 1.）组织要点
   - 使用代码块（```）包裹代码或公式
   - 使用引用块（>）突出重要观点

3. **标签处理**：
   - 如果提供了tags参数：{tags if tags else "未提供"}
   - 如果未提供tags：根据主题和内容自动生成3-5个相关标签
   - 标签示例：机器人学、控制理论、力控、阻抗控制、动力学等

4. **Obsidian特性**：
   - 在内容末尾添加相关链接：使用## 相关链接章节，列出[[相关概念]]
   - 确保所有双向链接使用正确的格式：[[概念名]]
   - 可以使用#标签格式在内容中添加行内标签

5. **文件创建**：
   - 使用create_obsidian_note_tool工具创建笔记文件
   - 传递整理后的markdown内容（不包括YAML frontmatter，工具会自动添加）
   - 传递主题和标签

**输出格式示例**：
```markdown
# {topic}

## 概念定义
[概念解释]

## 原理说明
[原理内容]

## 应用案例
[应用场景]

## 相关概念
- [[相关概念1]]
- [[相关概念2]]

## 相关链接
- [[概念A]]
- [[概念B]]
```

请立即执行，整理内容并创建笔记文件。"""
    
    task = Task(
        description=task_description,
        agent=agent,
        expected_output="成功创建Obsidian笔记文件，返回文件路径"
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        result_str = str(result)
        
        # 检查结果中是否包含成功信息
        if "成功创建笔记" in result_str or "成功创建" in result_str:
            # 提取文件路径
            import re
            path_match = re.search(r'paper_agent[/\\][^\s]+\.md', result_str)
            if path_match:
                file_path = path_match.group(0)
            else:
                file_path = result_str
            
            return {
                'success': True,
                'message': result_str,
                'file_path': file_path
            }
        else:
            return {
                'success': False,
                'error': result_str
            }
            
    except Exception as e:
        logging.error(f"生成笔记失败: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

