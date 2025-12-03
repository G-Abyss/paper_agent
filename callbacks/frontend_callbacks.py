#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前端通信回调
"""

import re

# 全局回调函数
crewai_log_callback = None  # 全局日志回调函数
agent_status_callback = None  # 全局agent状态回调函数


def send_agent_status(agent_name, status, task=None, result=None):
    """
    发送agent工作状态信息
    
    Args:
        agent_name: agent名称
        status: 状态，"start"或"end"
        task: Task对象（用于提取目标信息）
        result: 任务执行结果（用于提取输出信息）
    """
    global agent_status_callback
    
    if not agent_status_callback:
        return
    
    # 延迟导入，避免循环导入
    from callbacks.crewai_callbacks import extract_paper_title_from_task
    
    target = None
    output = None
    
    if status == 'start' and task:
        # 从任务描述中提取论文标题
        task_description = task.description if hasattr(task, 'description') else str(task)
        target = extract_paper_title_from_task(task_description)
    
    if status == 'end' and result:
        # 提取输出信息的前200个字符作为摘要
        if hasattr(result, 'raw'):
            output = result.raw.strip()
        elif isinstance(result, str):
            output = result.strip()
        else:
            output = str(result).strip()
        
        # 如果任务存在，从任务描述中提取论文标题，用于前端显示
        if task:
            task_description = task.description if hasattr(task, 'description') else str(task)
            target = extract_paper_title_from_task(task_description)
    
    # 调用回调函数
    agent_status_callback(agent_name, status, target, output)


def set_callbacks(on_log=None, on_agent_status=None):
    """
    设置回调函数
    
    Args:
        on_log: 日志回调函数 (level, message)
        on_agent_status: agent状态回调函数 (agent_name, status, target, output)
    """
    global crewai_log_callback, agent_status_callback
    crewai_log_callback = on_log
    agent_status_callback = on_agent_status

