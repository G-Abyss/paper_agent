#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web服务器 - 为Paper Summarizer提供Web界面和实时通信
"""

from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
import os
import sys
import subprocess
import threading
import json
import time
from datetime import datetime
import webbrowser

app = Flask(__name__)
app.config['SECRET_KEY'] = 'paper-summarizer-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局状态
current_process = None
process_thread = None

# 存储当前运行状态
running_status = {
    'is_running': False,
    'papers': {},
    'files': [],
    'waiting_confirmation': False,  # 是否等待人工确认（批量模式，已弃用）
    'confirmed_abstracts': {},  # 已确认的摘要 {paper_id: abstract_text}
    'confirmation_papers': {}  # 等待确认的论文数据
}

# 用于线程间通信的事件
confirmation_event = threading.Event()
confirmed_papers = {}  # 确认后的论文摘要数据

def emit_log(level, message):
    """发送日志消息到前端"""
    socketio.emit('message', {
        'type': 'log',
        'level': level,
        'message': message
    })

def emit_paper_added(paper):
    """发送论文添加消息"""
    socketio.emit('message', {
        'type': 'paper_added',
        'paper': paper
    })

def emit_paper_updated(paper_id, paper):
    """发送论文更新消息"""
    socketio.emit('message', {
        'type': 'paper_updated',
        'paper_id': paper_id,
        'paper': paper
    })

def emit_paper_removed(paper_id):
    """发送论文删除消息"""
    socketio.emit('message', {
        'type': 'paper_removed',
        'paper_id': paper_id
    })

def emit_file_generated(file_info):
    """发送文件生成消息"""
    socketio.emit('message', {
        'type': 'file_generated',
        'file': file_info
    })

def emit_status(status):
    """发送状态更新"""
    socketio.emit('message', {
        'type': 'status',
        'status': status
    })

def emit_waiting_confirmation(papers_data):
    """
    发送等待人工确认消息（批量模式，已弃用）
    
    Args:
        papers_data: 论文数据字典，格式为 {paper_id: {'title': ..., 'abstract': ..., 'link': ...}}
    """
    socketio.emit('message', {
        'type': 'waiting_confirmation',
        'papers': papers_data
    })

def emit_paper_ready_for_confirmation(paper_id, paper_data):
    """
    发送单个论文准备确认消息（实时模式）
    
    Args:
        paper_id: 论文ID
        paper_data: 论文数据字典，格式为 {'title': ..., 'abstract': ..., 'link': ...}
    """
    socketio.emit('message', {
        'type': 'paper_ready_for_confirmation',
        'paper_id': paper_id,
        'paper': paper_data
    })

def emit_agent_status(agent_name, status, target=None, output=None):
    """
    发送agent工作状态信息到前端
    
    Args:
        agent_name: agent名称，如"摘要清洗专家"
        status: 状态，"start"或"end"
        target: 目标信息（任务开始时），如论文标题
        output: 输出信息（任务结束时），如处理结果摘要
    """
    if status == 'start':
        message = f"{agent_name}工作开始"
        if target:
            # 如果target是论文标题，截取前60个字符
            target_display = target[:60] + "..." if len(target) > 60 else target
            message += f"：目标文章是{target_display}"
    elif status == 'end':
        message = f"{agent_name}工作结束"
        if output:
            # 输出信息截取前200个字符
            output_display = output[:200] + "..." if len(output) > 200 else output
            message += f"：输出是{output_display}"
    else:
        message = f"{agent_name}状态：{status}"
    
    socketio.emit('message', {
        'type': 'agent_status',
        'agent_name': agent_name,
        'status': status,
        'target': target,
        'output': output,
        'message': message
    })

@app.route('/')
def index():
    """返回主页面"""
    return send_from_directory('.', 'web_interface.html')

@app.route('/api/start', methods=['POST'])
def start_task():
    """启动处理任务"""
    global current_process, process_thread, running_status
    
    if running_status['is_running']:
        return jsonify({'success': False, 'error': '任务正在运行中'})
    
    try:
        config = request.json
        running_status['is_running'] = True
        running_status['papers'] = {}
        running_status['files'] = []
        
        # 在后台线程中运行任务
        process_thread = threading.Thread(
            target=run_summarizer_task,
            args=(config,),
            daemon=True
        )
        process_thread.start()
        
        emit_status('running')
        return jsonify({'success': True})
    except Exception as e:
        running_status['is_running'] = False
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/open_file', methods=['GET'])
def open_file():
    """打开文件"""
    try:
        file_path = request.args.get('path')
        if not file_path:
            return jsonify({'success': False, 'error': '未指定文件路径'})
        
        # 转换为绝对路径
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件不存在'})
        
        # 使用系统默认程序打开文件
        if sys.platform == 'win32':
            os.startfile(file_path)
        elif sys.platform == 'darwin':
            subprocess.run(['open', file_path])
        else:
            subprocess.run(['xdg-open', file_path])
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取当前状态"""
    return jsonify(running_status)

@app.route('/api/confirm_abstracts', methods=['POST'])
def confirm_abstracts():
    """确认论文摘要，继续处理（批量模式，已弃用）"""
    global confirmed_papers, confirmation_event, running_status
    
    try:
        data = request.json
        confirmed_papers = data.get('papers', {})  # {paper_id: abstract_text}
        
        # 更新运行状态
        running_status['waiting_confirmation'] = False
        
        # 通知等待的线程继续执行
        confirmation_event.set()
        
        emit_status('running')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@socketio.on('connect')
def handle_connect():
    """Socket.IO连接"""
    emit('message', {
        'type': 'log',
        'level': 'info',
        'message': '已连接到服务器'
    })

@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket断开"""
    pass

def run_summarizer_task(config):
    """在后台线程中运行summarizer任务"""
    global running_status
    
    try:
        # 设置环境变量
        if config.get('mode') == 'local':
            os.environ['LOCAL'] = '1'
            os.environ['CSV_FILE_PATH'] = config.get('csv_path', 'local_papers.csv')
        else:
            os.environ['LOCAL'] = '0'
            os.environ['START_DAYS'] = str(config.get('start_days', 1))
            os.environ['END_DAYS'] = str(config.get('end_days', 0))
        
        emit_log('info', '开始处理任务...')
        
        # 导入并运行主程序
        import run_summarizer as summarizer
        
        # 运行主程序（带回调）
        try:
            def wait_for_confirmation(papers_data):
                """等待人工确认"""
                global running_status, confirmation_event, confirmed_papers
                
                # 更新状态
                running_status['waiting_confirmation'] = True
                running_status['confirmation_papers'] = papers_data
                
                # 发送等待确认消息
                emit_waiting_confirmation(papers_data)
                emit_status('waiting_confirmation')
                
                # 清空之前确认的数据
                confirmed_papers = {}
                
                # 等待确认事件
                confirmation_event.clear()
                confirmation_event.wait()  # 阻塞等待确认
                
                # 返回确认后的摘要
                return confirmed_papers
            
            def get_confirmed_abstracts():
                """获取确认后的摘要"""
                global confirmed_papers
                return confirmed_papers
            
            def paper_ready_for_confirmation(paper_id, paper_data):
                """单个论文准备确认（立即显示可编辑摘要框，不阻塞）"""
                emit_paper_ready_for_confirmation(paper_id, paper_data)
            
            summarizer.main(
                on_log=emit_log,
                on_paper_added=emit_paper_added,
                on_paper_updated=emit_paper_updated,
                on_paper_removed=emit_paper_removed,
                on_file_generated=emit_file_generated,
                on_agent_status=emit_agent_status,
                on_waiting_confirmation=wait_for_confirmation,
                get_confirmed_abstracts=get_confirmed_abstracts,
                on_paper_ready_for_confirmation=paper_ready_for_confirmation,
                get_confirmed_abstract=None  # 不再需要实时确认
            )
            emit_log('success', '任务完成！')
            emit_status('completed')
        except Exception as e:
            emit_log('error', f'任务执行出错: {str(e)}')
            emit_status('error')
        
        running_status['is_running'] = False
        
    except Exception as e:
        emit_log('error', f'任务启动失败: {str(e)}')
        emit_status('error')
        running_status['is_running'] = False

if __name__ == '__main__':
    print("=" * 80)
    print("Paper Summarizer Web Server")
    print("=" * 80)
    print(f"访问地址: http://localhost:5000")
    print("=" * 80)
    
    # 自动打开浏览器
    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://localhost:5000')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 启动服务器
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

