#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web服务器 - 为Paper Summarizer提供Web界面和实时通信

主要修改点：
- 优化代码结构和注释
- 清理冗余代码
"""

from flask import Flask, send_from_directory, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import os
import sys
import subprocess
import threading
import json
import time
import logging
from datetime import datetime
import webbrowser
import requests
from utils.model_config import load_models, save_models, validate_model_config, get_active_model
from utils.email_storage import get_email_summary, load_emails
from utils.email_utils import connect_gmail
from agents.data_parser_agent import parse_file_with_agent, prepare_for_delivery
from agents.knowledge_engineer_agent import process_with_knowledge_engineer
from utils.agent_task_queue import get_task_queue
from utils.brain_tasks import trigger_brain_context_update, trigger_paper_depth_analysis

app = Flask(__name__)
app.config['SECRET_KEY'] = 'paper-summarizer-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False, engineio_logger=False)

# 全局状态
current_process = None
process_thread = None

# 存储当前运行状态
running_status = {
    'is_running': False,
    'papers': {},
    'files': [],
    'waiting_confirmation': False,  # 是否等待人工确认
    'confirmed_abstracts': {},  # 已确认的摘要 {paper_id: abstract_text}
    'confirmation_papers': {}  # 等待确认的论文数据
}

# 用于线程间通信的事件
confirmation_event = threading.Event()
confirmed_papers = {}  # 确认后的论文摘要数据 {paper_id: abstract_text}（批量模式）
paper_confirmation_papers = {}  # 需要确认的论文数据 {paper_id: paper_data}（用于批量确认）

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

def emit_queue_status():
    """发送任务队列状态更新"""
    try:
        task_queue = get_task_queue()
        # 获取活跃任务数（等待中 + 运行中）
        active_count = 0
        with task_queue.queue_lock:
            for task in task_queue.tasks.values():
                if task.status.value in ['pending', 'running']:
                    active_count += 1
        
        socketio.emit('message', {
            'type': 'queue_status',
            'count': active_count
        })
    except Exception as e:
        logging.error(f"发送队列状态失败: {str(e)}")

def queue_status_monitor():
    """后台监控队列状态的线程"""
    while True:
        emit_queue_status()
        time.sleep(2)  # 每2秒刷新一次

def emit_waiting_confirmation(papers_data):
    """
    发送等待人工确认消息（批量模式）
    
    Args:
        papers_data: 论文数据字典，格式为 {paper_id: {'title': ..., 'abstract': ..., 'link': ...}}
    """
    socketio.emit('message', {
        'type': 'waiting_confirmation',
        'papers': papers_data
    })

# --- 笔记功能相关辅助函数 ---
def get_root_dir():
    """获取项目根目录绝对路径"""
    return os.path.dirname(os.path.abspath(__file__))

def get_note_settings():
    """获取笔记设置"""
    try:
        settings_file = os.path.join(get_root_dir(), 'data', 'note_settings.json')
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logging.error(f"加载笔记设置失败: {str(e)}")
        return {}

def save_note_settings(settings):
    """保存笔记设置"""
    try:
        # 确保目录存在
        os.makedirs(os.path.join(get_root_dir(), 'data'), exist_ok=True)
        settings_file = os.path.join(get_root_dir(), 'data', 'note_settings.json')
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"保存笔记设置失败: {str(e)}")
        return False

def get_note_file_status():
    """获取笔记文件处理状态"""
    try:
        status_file = os.path.join(get_root_dir(), 'data', 'note_file_status.json')
        if os.path.exists(status_file):
            with open(status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logging.error(f"加载笔记文件状态失败: {str(e)}")
        return {}

def save_note_file_status(status):
    """保存笔记文件处理状态"""
    try:
        # 确保目录存在
        os.makedirs(os.path.join(get_root_dir(), 'data'), exist_ok=True)
        status_file = os.path.join(get_root_dir(), 'data', 'note_file_status.json')
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"保存笔记文件状态失败: {str(e)}")
        return False
# -----------------------------

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

@app.route('/database')
def database_management():
    """返回数据库系统页面"""
    return send_from_directory('.', 'database_management.html')

@app.route('/api/papers/list', methods=['GET'])
def get_papers_list():
    """获取论文列表"""
    try:
        from utils.vector_db import get_paper_list
        
        papers = get_paper_list()
        
        # 格式化论文数据
        formatted_papers = []
        for paper in papers:
            # 解析metadata中的信息
            metadata = paper.get('metadata', {}) if isinstance(paper.get('metadata'), dict) else {}
            
            formatted_papers.append({
                'paper_id': paper.get('paper_id', ''),
                'title': paper.get('paper_title', paper.get('title', '未知标题')),
                'tag': metadata.get('tag', '') or '',
                'authors': paper.get('authors', metadata.get('authors', '')) or '',
                'source': paper.get('source', '未知来源'),
                'abstract': paper.get('abstract', metadata.get('abstract', '')) or '',
                'keywords': ', '.join(paper.get('keywords', [])) if isinstance(paper.get('keywords'), list) else (metadata.get('keywords', '') or ''),
                'created_at': paper.get('created_at', ''),
                'attachment_path': paper.get('paper_path', paper.get('attachment_path', '')),
                'url': paper.get('url', metadata.get('link', '')),
                'content': paper.get('content', ''),
                'think_points': paper.get('think_points'),
                'contextual_summary': paper.get('contextual_summary', ''),
                'metadata': metadata  # 包含所有自定义字段
            })
        
        return jsonify({
            'success': True,
            'papers': formatted_papers,
            'total': len(formatted_papers)
        })
        
    except Exception as e:
        logging.error(f"获取论文列表失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/settings/tag', methods=['GET'])
def get_tag_settings():
    """获取tag设置"""
    try:
        import json
        import os
        
        settings_file = os.path.join(os.getcwd(), 'tag_settings.json')
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                return jsonify({
                    'success': True,
                    'tag_mode': settings.get('tag_mode', 'default'),
                    'custom_tag': settings.get('custom_tag', '')
                })
        else:
            return jsonify({
                'success': True,
                'tag_mode': 'default',
                'custom_tag': ''
            })
    except Exception as e:
        logging.error(f"获取tag设置失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/settings/tag', methods=['POST'])
def save_tag_settings():
    """保存tag设置"""
    try:
        import json
        import os
        
        data = request.json
        tag_mode = data.get('tag_mode', 'default')
        custom_tag = data.get('custom_tag', '')
        
        settings = {
            'tag_mode': tag_mode,
            'custom_tag': custom_tag
        }
        
        settings_file = os.path.join(os.getcwd(), 'tag_settings.json')
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True
        })
    except Exception as e:
        logging.error(f"保存tag设置失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_import_tag(source_type='csv'):
    """获取导入tag（根据设置返回default或custom tag）"""
    try:
        import json
        import os
        
        settings_file = os.path.join(os.getcwd(), 'tag_settings.json')
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                tag_mode = settings.get('tag_mode', 'default')
                if tag_mode == 'custom':
                    return settings.get('custom_tag', '')
        
        # 默认tag
        default_tags = {
            'csv': 'csv',
            'mail': 'mail',
            'pdf': 'pdf'
        }
        return default_tags.get(source_type, 'csv')
    except Exception as e:
        logging.error(f"获取导入tag失败: {str(e)}", exc_info=True)
        # 返回默认tag
        default_tags = {
            'csv': 'csv',
            'mail': 'mail',
            'pdf': 'pdf'
        }
        return default_tags.get(source_type, 'csv')

@app.route('/api/papers/update', methods=['POST'])
def update_paper():
    """更新论文字段"""
    try:
        data = request.json
        paper_id = data.get('paper_id')
        field = data.get('field')
        value = data.get('value', '')
        
        if not paper_id or not field:
            return jsonify({
                'success': False,
                'error': '缺少必要参数'
            }), 400
        
        from utils.vector_db import get_db_connection, return_db_connection
        from psycopg2.extras import RealDictCursor, Json
        
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 获取当前论文数据
            cur.execute("SELECT metadata FROM papers WHERE paper_id = %s", (paper_id,))
            result = cur.fetchone()
            
            if not result:
                return jsonify({
                    'success': False,
                    'error': '论文不存在'
                }), 404
            
            # 解析metadata
            metadata = result['metadata'] or {}
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata)
            elif not isinstance(metadata, dict):
                metadata = {}
            
            # 更新字段
            if field in ['title', 'authors', 'abstract', 'keywords']:
                # 直接更新表字段
                if field == 'keywords':
                    # keywords是数组类型
                    keywords_list = [k.strip() for k in value.split(',') if k.strip()] if value else []
                    cur.execute(f"UPDATE papers SET {field} = %s WHERE paper_id = %s", 
                              (keywords_list, paper_id))
                elif field == 'authors':
                    # authors是数组类型
                    authors_list = [a.strip() for a in value.split(',') if a.strip()] if value else []
                    cur.execute(f"UPDATE papers SET {field} = %s WHERE paper_id = %s", 
                              (authors_list, paper_id))
                else:
                    cur.execute(f"UPDATE papers SET {field} = %s WHERE paper_id = %s", 
                              (value, paper_id))
            else:
                # 自定义字段存储在metadata中
                metadata[field] = value
                cur.execute("UPDATE papers SET metadata = %s WHERE paper_id = %s", 
                          (Json(metadata), paper_id))
            
            conn.commit()
            logging.info(f"更新论文字段成功: paper_id={paper_id}, field={field}")
            
            return jsonify({
                'success': True
            })
            
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        logging.error(f"更新论文失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/papers/delete', methods=['POST'])
def delete_paper():
    """删除论文"""
    try:
        data = request.json
        paper_id = data.get('paper_id')
        
        if not paper_id:
            return jsonify({
                'success': False,
                'error': '缺少paper_id参数'
            }), 400
        
        from utils.vector_db import delete_paper
        
        success = delete_paper(paper_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': '论文已删除'
            })
        else:
            return jsonify({
                'success': False,
                'error': '删除失败'
            }), 500
            
    except Exception as e:
        logging.error(f"删除论文失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/papers/compare_csv', methods=['POST'])
def compare_csv():
    """比较CSV文件与数据库，返回差异信息（不更新数据库）"""
    try:
        import csv
        import io
        import json
        import os
        from utils.vector_db import get_db_connection, return_db_connection, get_paper_list
        from psycopg2.extras import RealDictCursor
        
        if 'csv_file' not in request.files:
            return jsonify({
                'success': False,
                'error': '未找到CSV文件'
            }), 400
        
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '未选择文件'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'success': False,
                'error': '文件必须是CSV格式'
            }), 400
        
        # 读取上传的CSV文件内容
        uploaded_stream = io.StringIO(file.stream.read().decode('utf-8-sig'))  # 处理BOM
        uploaded_reader = csv.DictReader(uploaded_stream)
        uploaded_headers = uploaded_reader.fieldnames
        uploaded_rows = list(uploaded_reader)
        
        if not uploaded_headers:
            return jsonify({
                'success': False,
                'error': '上传的CSV文件格式错误：缺少表头'
            }), 400
        
        # 获取数据库中的所有论文
        db_papers = get_paper_list()
        
        # 收集所有可能的字段
        all_db_field_keys = set()
        standard_db_fields = [
            'paper_id', 'title', 'authors', 'abstract', 'keywords', 'year', 'journal', 'doi', 'url',
            'source', 'source_id', 'attachment_path', 'zotero_key', 'obsidian_note_path', 'created_at', 'updated_at'
        ]
        for field in standard_db_fields:
            all_db_field_keys.add(field)
        
        for paper in db_papers:
            if paper.get('metadata'):
                for key in paper['metadata'].keys():
                    all_db_field_keys.add(key)
        
        # 创建标准字段的小写映射（用于忽略大小写比较）
        standard_fields_lower_map = {field.lower(): field for field in all_db_field_keys}
        upload_headers_lower_set = {h.lower() for h in uploaded_headers}
        
        # 识别新列
        new_columns = []
        for upload_header in uploaded_headers:
            upload_header_lower = upload_header.lower()
            if upload_header_lower not in standard_fields_lower_map:
                new_columns.append(upload_header)
        
        # 创建上传表头到标准字段名的映射
        upload_to_standard_map = {}
        for upload_header in uploaded_headers:
            upload_header_lower = upload_header.lower()
            if upload_header_lower in standard_fields_lower_map:
                upload_to_standard_map[upload_header] = standard_fields_lower_map[upload_header_lower]
            else:
                upload_to_standard_map[upload_header] = upload_header
        
        # 比较差异
        differences = []
        db_paper_map = {p['paper_id']: p for p in db_papers}
        
        for uploaded_row_idx, uploaded_row in enumerate(uploaded_rows):
            try:
                # 获取paper_id
                paper_id_header = None
                for upload_h in uploaded_headers:
                    if upload_h.lower() == 'paper_id':
                        paper_id_header = upload_h
                        break
                
                paper_id = uploaded_row.get(paper_id_header or 'paper_id', '').strip()
                if not paper_id:
                    continue
                
                existing_paper = db_paper_map.get(paper_id)
                if not existing_paper:
                    # 尝试通过title匹配
                    title_header = None
                    for upload_h in uploaded_headers:
                        if upload_h.lower() == 'title':
                            title_header = upload_h
                            break
                    
                    title = uploaded_row.get(title_header or 'title', '').strip()
                    if title:
                        for p in db_papers:
                            if p.get('title') == title:
                                existing_paper = p
                                paper_id = p['paper_id']
                                break
                
                if not existing_paper:
                    continue
                
                # 比较字段差异
                paper_differences = []
                metadata = existing_paper.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata) if metadata else {}
                elif not isinstance(metadata, dict):
                    metadata = {}
                
                for header in uploaded_headers:
                    standard_field = upload_to_standard_map.get(header, header)
                    upload_value = uploaded_row.get(header, '').strip()
                    
                    # 获取数据库值
                    db_value = None
                    if standard_field in ['title', 'abstract', 'authors', 'keywords']:
                        if standard_field == 'authors' or standard_field == 'keywords':
                            db_value = '; '.join(existing_paper.get(standard_field, [])) if isinstance(existing_paper.get(standard_field), list) else str(existing_paper.get(standard_field, ''))
                        else:
                            db_value = str(existing_paper.get(standard_field, ''))
                    elif standard_field in standard_db_fields:
                        db_value = str(existing_paper.get(standard_field, ''))
                    else:
                        db_value = str(metadata.get(standard_field, ''))
                    
                    if upload_value != (db_value or ''):
                        paper_differences.append({
                            'field': header,
                            'old_value': db_value or '',
                            'new_value': upload_value
                        })
                
                if paper_differences:
                    differences.append({
                        'paper_id': paper_id,
                        'title': existing_paper.get('title', ''),
                        'differences': paper_differences
                    })
            except Exception as e:
                logging.warning(f"比较CSV行 {uploaded_row_idx + 2} 失败: {str(e)}")
                continue
        
        return jsonify({
            'success': True,
            'new_columns': [{'name': col, 'display': col} for col in sorted(new_columns)],
            'differences': differences,
            'total_differences': len(differences),
            'total_rows': len(uploaded_rows)
        })
        
    except Exception as e:
        logging.error(f"比较CSV失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'处理CSV文件时出错: {str(e)}'
        }), 500

@app.route('/api/papers/cleanup_unused_pdfs', methods=['POST'])
def cleanup_unused_pdfs():
    """清理database目录中不在数据库attachment_path中的PDF文件"""
    try:
        from utils.vector_db import get_db_connection, return_db_connection
        from pathlib import Path
        
        # 获取database目录路径
        database_dir = Path(os.getcwd()) / 'database'
        if not database_dir.exists():
            return jsonify({
                'success': False,
                'error': 'database目录不存在'
            }), 404
        
        # 从数据库获取所有attachment_path
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT attachment_path FROM papers WHERE attachment_path IS NOT NULL AND attachment_path != ''")
            db_paths_raw = [row[0] for row in cur.fetchall()]
            cur.close()
        finally:
            return_db_connection(conn)
        
        # 标准化数据库中的路径（统一使用正斜杠，并转换为绝对路径进行比较）
        db_paths = set()
        for path in db_paths_raw:
            if path:
                # 统一路径分隔符为正斜杠
                normalized_path = path.replace('\\', '/')
                # 如果是相对路径，转换为绝对路径
                if not os.path.isabs(normalized_path):
                    abs_path = os.path.abspath(os.path.join(os.getcwd(), normalized_path))
                else:
                    abs_path = os.path.abspath(normalized_path)
                db_paths.add(abs_path)
                # 同时保留相对路径格式（以防数据库存储的是相对路径）
                db_paths.add(normalized_path)
        
        # 扫描database目录中的所有PDF文件
        pdf_files = list(database_dir.glob('*.pdf'))
        
        # 找出不在数据库中的PDF文件
        deleted_files = []
        total_size = 0
        
        for pdf_file in pdf_files:
            # 获取绝对路径
            abs_path = str(pdf_file.resolve())
            # 获取相对路径（统一使用正斜杠）
            relative_path = str(pdf_file.relative_to(os.getcwd())).replace('\\', '/')
            
            # 检查是否在数据库中（检查绝对路径和相对路径）
            if abs_path not in db_paths and relative_path not in db_paths:
                try:
                    # 获取文件大小
                    file_size = pdf_file.stat().st_size
                    total_size += file_size
                    
                    # 删除文件
                    pdf_file.unlink()
                    deleted_files.append(relative_path)
                    logging.info(f'已删除无用PDF文件: {relative_path}')
                except Exception as e:
                    logging.error(f'删除PDF文件失败 {relative_path}: {str(e)}')
        
        return jsonify({
            'success': True,
            'deleted_files': deleted_files,
            'deleted_count': len(deleted_files),
            'total_size_mb': total_size / (1024 * 1024),
            'message': f'清理完成，删除了 {len(deleted_files)} 个无用PDF文件'
        })
        
    except Exception as e:
        logging.error(f'清理无用PDF失败: {str(e)}', exc_info=True)
        return jsonify({
            'success': False,
            'error': f'清理失败: {str(e)}'
        }), 500

@app.route('/api/upload/parse', methods=['POST'])
def upload_and_parse_file():
    """上传文件并解析（数据社区与解析主管）"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '未找到上传文件'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '未选择文件'
            }), 400
        
        # 检查文件类型
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.csv', '.pdf']:
            return jsonify({
                'success': False,
                'error': '不支持的文件类型，仅支持 CSV 和 PDF 文件'
            }), 400
        
        # 保存文件到临时目录
        upload_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # 生成唯一文件名（避免冲突）
        timestamp = int(time.time() * 1000)
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(upload_dir, safe_filename)
        file.save(file_path)
        
        # 步骤1：使用数据摄取与解析主管Agent解析文件（通过任务队列）
        logging.info(f"开始解析文件: {file.filename}, 路径: {file_path}")
        
        try:
            # 将解析任务添加到队列
            task_queue = get_task_queue()
            parse_task_id = task_queue.add_task(
                f"parse_{int(time.time() * 1000)}_{os.path.basename(file_path)}",
                'file_parse',
                parse_file_with_agent,
                file_path
            )
            
            # 等待任务完成（最多等待5分钟）
            parse_task_result = task_queue.wait_for_task(parse_task_id, timeout=300)
            
            if not parse_task_result.get('success'):
                error_msg = parse_task_result.get('error', '解析任务执行失败')
                logging.error(f"文件解析任务失败: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': f'文件解析失败: {error_msg}'
                }), 500
            
            parse_result = parse_task_result.get('result')
            logging.info(f"解析结果: {parse_result}")
        except Exception as e:
            logging.error(f"解析文件时发生异常: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'文件解析异常: {str(e)}'
            }), 500
        
        # 检查解析结果
        if not parse_result:
            return jsonify({
                'success': False,
                'error': '文件解析失败: 未返回结果'
            }), 500
        
        # 提取解析数据 - 处理不同的返回格式
        parse_data = None
        source_type = 'unknown'
        
        # 情况1: parse_result包含parse_result字段
        if 'parse_result' in parse_result:
            parse_data = parse_result['parse_result']
            if isinstance(parse_data, str):
                try:
                    parse_data = json.loads(parse_data)
                except:
                    logging.warning(f"parse_result是字符串但无法解析为JSON: {parse_data[:100]}")
            source_type = parse_data.get('file_type', parse_result.get('file_type', 'unknown'))
        # 情况2: parse_result直接就是解析数据
        elif 'file_type' in parse_result:
            parse_data = parse_result
            source_type = parse_result.get('file_type', 'unknown')
        # 情况3: 回退处理
        else:
            logging.warning(f"无法识别解析结果格式: {list(parse_result.keys())}")
            # 尝试直接使用parse_result
            parse_data = parse_result
            source_type = parse_result.get('file_type', file_ext.lstrip('.'))
        
        # 检查是否成功
        if not parse_result.get('success', True):
            error_msg = parse_result.get('error', '解析失败')
            return jsonify({
                'success': False,
                'error': f'文件解析失败: {error_msg}'
            }), 500
        
        if not parse_data:
            return jsonify({
                'success': False,
                'error': '文件解析失败: 无法提取解析数据'
            }), 500
        
        logging.info(f"提取的解析数据 - 文件类型: {source_type}, 数据键: {list(parse_data.keys())}")
        
        # 步骤2：准备传递给知识工程师的数据
        try:
            if source_type == 'csv':
                # CSV文件：传递文件路径和基本元数据
                metadata = {
                    'file_name': parse_data.get('file_name', os.path.basename(file_path)),
                    'columns': parse_data.get('columns', []),
                    'total_rows': parse_data.get('total_rows', 0),
                    'file_type': 'csv'
                }
                raw_text = ''  # CSV不需要原始文本
                logging.info(f"CSV文件准备完成: {metadata.get('total_rows', 0)} 行数据")
            elif source_type == 'pdf':
                # PDF文件：传递文件路径和已提取的元数据
                metadata = parse_data.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}
                
                metadata['file_type'] = 'pdf'
                metadata['page_count'] = parse_data.get('page_count', 0)
                metadata['file_size'] = parse_data.get('file_size', 0)
                metadata['title'] = metadata.get('title', parse_data.get('file_name', os.path.basename(file_path)))
                raw_text = parse_data.get('raw_text', '')  # PDF的原始文本（用于后续处理）
                logging.info(f"PDF文件准备完成: {len(raw_text)} 字符, {metadata.get('page_count', 0)} 页")
            else:
                logging.error(f"不支持的文件类型: {source_type}")
                return jsonify({
                    'success': False,
                    'error': f'不支持的文件类型: {source_type}'
                }), 400
        except Exception as e:
            logging.error(f"准备知识工程师数据时发生异常: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'准备数据失败: {str(e)}'
            }), 500
        
        # 步骤3：使用知识工程师Agent处理数据（通过任务队列）
        logging.info(f"开始知识工程处理: {file.filename} (类型: {source_type})")
        try:
            # 将知识工程任务添加到队列
            task_queue = get_task_queue()
            knowledge_task_id = task_queue.add_task(
                f"knowledge_{int(time.time() * 1000)}_{os.path.basename(file_path)}",
                'knowledge_engine',
                process_with_knowledge_engineer,
                raw_text,
                metadata,
                file_path,
                source_type
            )
            
            # 等待任务完成（最多等待10分钟）
            knowledge_task_result = task_queue.wait_for_task(knowledge_task_id, timeout=600)
            
            if not knowledge_task_result.get('success'):
                error_msg = knowledge_task_result.get('error', '知识工程任务执行失败')
                logging.error(f"知识工程处理任务失败: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': f'知识工程处理失败: {error_msg}'
                }), 500
            
            knowledge_result = knowledge_task_result.get('result')
            logging.info(f"知识工程处理结果: success={knowledge_result.get('success', False)}, chunks={knowledge_result.get('chunks_stored', 0)}")
        except Exception as e:
            logging.error(f"知识工程处理时发生异常: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'知识工程处理异常: {str(e)}'
            }), 500
        
        # 返回处理结果
        if knowledge_result and knowledge_result.get('success', False):
            # 根据文件类型生成不同的成功消息
            if source_type == 'csv':
                papers_count = knowledge_result.get('papers_imported', 0)
                success_message = f'CSV文件处理成功: {file.filename}，已导入 {papers_count} 篇论文到数据库'
            else:  # pdf
                chunks_count = knowledge_result.get('chunks_stored', 0)
                success_message = f'PDF文件处理成功: {file.filename}，已存储 {chunks_count} 个文本块到知识库'
                
                # 检查是否需要触发大脑深度分析
                settings = get_note_settings()
                if settings.get('brain_organizer_active', False):
                    paper_id = knowledge_result.get('paper_id')
                    if paper_id:
                        task_queue.add_task(f"paper_think_{paper_id}", "paper_think", trigger_paper_depth_analysis, paper_id)
                        emit_log('info', f'已触发论文深度分析任务: {paper_id}')
            
            return jsonify({
                'success': True,
                'data': {
                    'parse_result': parse_data,
                    'knowledge_result': knowledge_result
                },
                'message': success_message
            })
        else:
            error_msg = knowledge_result.get('error', '未知错误') if knowledge_result else '知识工程处理返回空结果'
            return jsonify({
                'success': False,
                'error': f'知识工程处理失败: {error_msg}',
                'data': {
                    'parse_result': parse_data,
                    'knowledge_result': knowledge_result
                }
            }), 500
        
    except Exception as e:
        logging.error(f"文件上传解析失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'文件解析失败: {str(e)}'
        }), 500

@app.route('/api/papers/upload_csv', methods=['POST'])
def upload_csv():
    """上传CSV文件并更新数据库（后端生成标准CSV并比较更新）"""
    try:
        import csv
        import io
        import json
        import os
        from utils.vector_db import get_db_connection, return_db_connection
        from psycopg2.extras import RealDictCursor, Json
        
        if 'csv_file' not in request.files:
            return jsonify({
                'success': False,
                'error': '未找到CSV文件'
            }), 400
        
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '未选择文件'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'success': False,
                'error': '文件必须是CSV格式'
            }), 400
        
        conn = get_db_connection()
        updated_count = 0
        skipped_count = 0
        error_count = 0
        new_columns_info = []
        
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 第一步：生成标准CSV（当前数据库的完整内容）
            cur.execute("""
                SELECT 
                    paper_id, title, authors, abstract, keywords, year, journal, 
                    doi, url, source, source_id, attachment_path, zotero_key, 
                    obsidian_note_path, metadata, created_at, updated_at
                FROM papers
                ORDER BY paper_id
            """)
            db_papers = cur.fetchall()
            
            # 收集所有字段名（标准字段 + metadata中的自定义字段）
            standard_fields = ['paper_id', 'title', 'authors', 'abstract', 'keywords', 
                             'year', 'journal', 'doi', 'url', 'source', 'source_id', 
                             'attachment_path', 'zotero_key', 'obsidian_note_path', 
                             'created_at', 'updated_at']
            custom_fields_set = set()
            for paper in db_papers:
                metadata = paper.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata) if metadata else {}
                elif not isinstance(metadata, dict):
                    metadata = {}
                custom_fields_set.update(metadata.keys())
            
            # 合并所有字段（标准字段在前，自定义字段按字母顺序）
            all_fields = standard_fields + sorted(custom_fields_set)
            
            # 构建标准CSV数据（用于比较）
            standard_csv_data = {}
            for paper in db_papers:
                paper_id = paper['paper_id']
                row_data = {}
                
                # 处理标准字段
                for field in standard_fields:
                    value = paper.get(field)
                    if field == 'authors' and isinstance(value, list):
                        row_data[field] = '; '.join(str(v) for v in value)
                    elif field == 'keywords' and isinstance(value, list):
                        row_data[field] = '; '.join(str(v) for v in value)
                    elif field == 'attachment_path' and value:
                        # 转换为相对路径
                        path_str = str(value).replace('\\', '/')
                        if 'database' in path_str:
                            parts = path_str.split('/')
                            db_index = parts.index('database')
                            row_data[field] = '/'.join(parts[db_index:])
                        else:
                            row_data[field] = value
                    else:
                        row_data[field] = str(value) if value is not None else ''
                
                # 处理metadata中的自定义字段
                metadata = paper.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata) if metadata else {}
                elif not isinstance(metadata, dict):
                    metadata = {}
                
                for field in sorted(custom_fields_set):
                    value = metadata.get(field)
                    if isinstance(value, (dict, list)):
                        row_data[field] = json.dumps(value, ensure_ascii=False)
                    else:
                        row_data[field] = str(value) if value is not None else ''
                
                standard_csv_data[paper_id] = row_data
            
            # 第二步：读取上传的CSV文件
            file.stream.seek(0)
            stream = io.StringIO(file.stream.read().decode('utf-8-sig'))  # 处理BOM
            reader = csv.DictReader(stream)
            
            # 获取上传CSV的表头
            upload_headers = reader.fieldnames
            if not upload_headers:
                return jsonify({
                    'success': False,
                    'error': 'CSV文件格式错误：缺少表头'
                }), 400
            
            # 第三步：比较表头，识别新列（忽略大小写）
            # 创建标准字段的小写映射（用于忽略大小写比较）
            standard_fields_lower_map = {field.lower(): field for field in all_fields}
            upload_headers_lower_set = {h.lower() for h in upload_headers}
            
            # 识别新列：上传CSV中的列，在标准字段中不存在（忽略大小写）
            new_columns = []
            for upload_header in upload_headers:
                upload_header_lower = upload_header.lower()
                if upload_header_lower not in standard_fields_lower_map:
                    new_columns.append(upload_header)
            
            # 创建上传表头到标准字段名的映射（忽略大小写）
            upload_to_standard_map = {}
            for upload_header in upload_headers:
                upload_header_lower = upload_header.lower()
                if upload_header_lower in standard_fields_lower_map:
                    # 找到对应的标准字段名
                    upload_to_standard_map[upload_header] = standard_fields_lower_map[upload_header_lower]
                else:
                    # 新列，使用原始列名
                    upload_to_standard_map[upload_header] = upload_header
            
            if new_columns:
                new_columns_info = [{'name': col, 'display': col} for col in sorted(new_columns)]
            
            # 第四步：读取上传CSV的所有行数据
            upload_csv_rows = []
            for row in reader:
                upload_csv_rows.append(dict(row))
            
            # 第五步：比较单元格内容并更新数据库
            for upload_row in upload_csv_rows:
                try:
                    # 通过paper_id匹配（优先）或title匹配（忽略大小写）
                    # 使用映射表找到对应的标准字段名
                    paper_id_header = None
                    title_header = None
                    for upload_h in upload_headers:
                        upload_h_lower = upload_h.lower()
                        if upload_h_lower == 'paper_id':
                            paper_id_header = upload_h
                        elif upload_h_lower == 'title':
                            title_header = upload_h
                    
                    paper_id = upload_row.get(paper_id_header or 'paper_id', '').strip()
                    title = upload_row.get(title_header or 'title', '').strip()
                    
                    # 查找对应的数据库记录
                    db_paper = None
                    if paper_id:
                        cur.execute("SELECT paper_id, metadata FROM papers WHERE paper_id = %s", (paper_id,))
                        result = cur.fetchone()
                        if result:
                            db_paper = result
                            paper_id = result['paper_id']
                    
                    if not db_paper and title:
                        cur.execute("SELECT paper_id, metadata FROM papers WHERE title = %s ORDER BY created_at DESC LIMIT 1", (title,))
                        result = cur.fetchone()
                        if result:
                            db_paper = result
                            paper_id = result['paper_id']
                    
                    if not db_paper:
                        skipped_count += 1
                        continue
                    
                    # 获取现有数据
                    cur.execute("""
                        SELECT paper_id, title, authors, abstract, keywords, year, journal, 
                               doi, url, source, source_id, attachment_path, zotero_key, 
                               obsidian_note_path, metadata
                        FROM papers WHERE paper_id = %s
                    """, (paper_id,))
                    existing = cur.fetchone()
                    
                    if not existing:
                        skipped_count += 1
                        continue
                    
                    # 解析metadata
                    metadata = existing.get('metadata', {})
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata) if metadata else {}
                    elif not isinstance(metadata, dict):
                        metadata = {}
                    
                    # 比较并构建更新字段
                    update_fields = {}
                    has_changes = False
                    
                    # 处理所有上传CSV中的字段（只处理上传CSV中存在的字段，不删除已有列）
                    for header in upload_headers:
                        upload_value = upload_row.get(header, '').strip()
                        
                        # 获取对应的标准字段名（忽略大小写）
                        standard_field = upload_to_standard_map.get(header, header)
                        
                        # 获取数据库中的对应值
                        db_value = None
                        if standard_field in standard_fields:
                            if standard_field == 'authors':
                                db_value = '; '.join(existing.get('authors', [])) if isinstance(existing.get('authors'), list) else str(existing.get('authors', ''))
                            elif standard_field == 'keywords':
                                db_value = '; '.join(existing.get('keywords', [])) if isinstance(existing.get('keywords'), list) else str(existing.get('keywords', ''))
                            elif standard_field == 'attachment_path':
                                db_path = existing.get('attachment_path', '')
                                if db_path:
                                    path_str = str(db_path).replace('\\', '/')
                                    if 'database' in path_str:
                                        parts = path_str.split('/')
                                        db_index = parts.index('database')
                                        db_value = '/'.join(parts[db_index:])
                                    else:
                                        db_value = str(db_path)
                                else:
                                    db_value = ''
                            else:
                                db_value = str(existing.get(standard_field, '')) if existing.get(standard_field) is not None else ''
                        else:
                            # 自定义字段从metadata中获取（使用标准字段名）
                            meta_value = metadata.get(standard_field)
                            if isinstance(meta_value, (dict, list)):
                                db_value = json.dumps(meta_value, ensure_ascii=False)
                            else:
                                db_value = str(meta_value) if meta_value is not None else ''
                        
                        # 比较值是否不同
                        if upload_value != db_value:
                            has_changes = True
                            
                            # 根据字段类型处理更新（使用标准字段名）
                            if standard_field == 'paper_id':
                                # paper_id不更新
                                pass
                            elif standard_field == 'authors':
                                authors_list = [a.strip() for a in upload_value.split(';') if a.strip()]
                                update_fields['authors'] = authors_list
                            elif standard_field == 'keywords':
                                keywords_list = [k.strip() for k in upload_value.split(';') if k.strip()]
                                update_fields['keywords'] = keywords_list
                            elif standard_field == 'attachment_path':
                                # attachment_path不更新（由系统管理）
                                pass
                            elif standard_field in ['title', 'abstract', 'source', 'year', 'journal', 'doi', 'url', 'source_id', 'zotero_key', 'obsidian_note_path']:
                                update_fields[standard_field] = upload_value
                            else:
                                # 自定义字段或新列存储在metadata中（使用标准字段名）
                                try:
                                    # 尝试解析JSON
                                    metadata[standard_field] = json.loads(upload_value)
                                except:
                                    # 如果不是JSON，直接存储字符串
                                    metadata[standard_field] = upload_value
                    
                    # 添加tag到metadata（CSV导入）
                    if 'tag' not in metadata:
                        import_tag = get_import_tag('csv')
                        metadata['tag'] = import_tag
                    
                    # 执行更新
                    if has_changes:
                        set_clauses = []
                        values = []
                        
                        for field, value in update_fields.items():
                            set_clauses.append(f"{field} = %s")
                            values.append(value)
                        
                        if metadata:
                            set_clauses.append("metadata = %s")
                            values.append(Json(metadata))
                        
                        if set_clauses:
                            set_clauses.append("updated_at = NOW()")
                            values.append(paper_id)
                            
                            query = f"UPDATE papers SET {', '.join(set_clauses)} WHERE paper_id = %s"
                            cur.execute(query, values)
                            updated_count += 1
                        else:
                            skipped_count += 1
                    else:
                        # 即使没有其他变化，也要更新tag
                        if 'tag' in metadata:
                            cur.execute("""
                                UPDATE papers 
                                SET metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{tag}', %s::jsonb),
                                    updated_at = NOW()
                                WHERE paper_id = %s
                            """, (Json(metadata['tag']), paper_id))
                            updated_count += 1
                        else:
                            skipped_count += 1
                        
                except Exception as e:
                    error_count += 1
                    logging.error(f"处理CSV行失败: {str(e)}", exc_info=True)
                    continue
            
            conn.commit()
            
            return jsonify({
                'success': True,
                'updated': updated_count,
                'skipped': skipped_count,
                'errors': error_count,
                'new_columns': new_columns_info,
                'message': f'成功更新 {updated_count} 条记录，跳过 {skipped_count} 条，{error_count} 条处理失败'
            })
            
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        logging.error(f"上传CSV失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/start', methods=['POST'])
def start_task():
    """启动处理任务"""
    global current_process, process_thread, running_status
    
    if running_status['is_running']:
        return jsonify({'success': False, 'error': '任务正在运行中'})
    
    try:
        # 获取配置（支持JSON和FormData两种方式）
        if request.content_type and 'application/json' in request.content_type:
            config = request.json
        else:
            # FormData方式（用于文件上传）
            config_str = request.form.get('config', '{}')
            try:
                config = json.loads(config_str)
            except json.JSONDecodeError as e:
                return jsonify({'success': False, 'error': f'配置解析失败: {str(e)}'}), 400
            
            # 处理文件上传（本地CSV模式或论文PDF模式）
            if 'csv_file' in request.files:
                csv_file = request.files['csv_file']
                if csv_file.filename:
                    # 保存文件到工作目录
                    upload_dir = os.getcwd()
                    file_path = os.path.join(upload_dir, csv_file.filename)
                    csv_file.save(file_path)
                    # 设置文件路径到配置中
                    config['csv_path'] = csv_file.filename
            elif 'pdf_files' in request.files:
                # 支持多文件上传
                pdf_files = request.files.getlist('pdf_files')
                if pdf_files and len(pdf_files) > 0:
                    # 保存PDF文件到downloads目录
                    upload_dir = os.path.join(os.getcwd(), 'downloads')
                    os.makedirs(upload_dir, exist_ok=True)
                    pdf_paths = []
                    for pdf_file in pdf_files:
                        if pdf_file.filename:
                            file_path = os.path.join(upload_dir, pdf_file.filename)
                            pdf_file.save(file_path)
                            pdf_paths.append(file_path)
                    # 设置文件路径列表到配置中
                    config['pdf_paths'] = pdf_paths
            elif 'pdf_file' in request.files:
                # 兼容单文件上传（向后兼容）
                pdf_file = request.files['pdf_file']
                if pdf_file.filename:
                    # 保存PDF文件到downloads目录
                    upload_dir = os.path.join(os.getcwd(), 'downloads')
                    os.makedirs(upload_dir, exist_ok=True)
                    file_path = os.path.join(upload_dir, pdf_file.filename)
                    pdf_file.save(file_path)
                    # 设置文件路径到配置中（兼容旧代码）
                    config['pdf_path'] = file_path
                    config['pdf_paths'] = [file_path]
        
        # 验证配置中的mode字段
        if 'mode' not in config:
            return jsonify({'success': False, 'error': '配置中缺少mode字段'}), 400
        
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

@app.route('/api/download_file', methods=['GET'])
def download_file():
    """下载文件"""
    try:
        file_path = request.args.get('path')
        if not file_path:
            return jsonify({'success': False, 'error': '未指定文件路径'}), 400
        
        # 转换为绝对路径
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件不存在'}), 404
        
        # 安全检查：只允许下载 reports/ 和 downloads/ 目录下的文件
        base_dir = os.path.abspath(os.getcwd())
        abs_file_path = os.path.abspath(file_path)
        
        reports_dir = os.path.abspath(os.path.join(base_dir, 'reports'))
        downloads_dir = os.path.abspath(os.path.join(base_dir, 'downloads'))
        
        if not (abs_file_path.startswith(reports_dir) or abs_file_path.startswith(downloads_dir)):
            return jsonify({'success': False, 'error': '不允许访问该路径'}), 403
        
        return send_file(file_path, as_attachment=True, download_name=os.path.basename(abs_file_path))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取当前状态"""
    return jsonify(running_status)

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取所有模型配置"""
    try:
        models = load_models()
        return jsonify({'success': True, 'models': models})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models', methods=['POST'])
def save_models_api():
    """保存模型配置"""
    try:
        data = request.json
        models = data.get('models', [])
        
        if not isinstance(models, list):
            return jsonify({'success': False, 'error': '模型配置必须是列表格式'})
        
        # 验证所有模型配置
        for model in models:
            if not validate_model_config(model):
                return jsonify({'success': False, 'error': f'模型配置无效: {model.get("name", "Unknown")}'})
        
        # 生成ID（如果没有）
        for i, model in enumerate(models):
            if 'id' not in model or not model['id']:
                model['id'] = f"model_{i}_{int(time.time())}"
        
        if save_models(models):
            return jsonify({'success': True, 'message': '模型配置保存成功'})
        else:
            return jsonify({'success': False, 'error': '保存模型配置失败'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models/test', methods=['POST'])
def test_model():
    """测试模型连通性"""
    try:
        data = request.json
        model = data.get('model')
        
        if not model:
            return jsonify({'success': False, 'error': '未提供模型配置'})
        
        if not validate_model_config(model):
            return jsonify({'success': False, 'error': '模型配置无效'})
        
        model_type = model.get('type')
        base_url = model.get('base_url') or model.get('api_base', '')
        
        if model_type == 'local':
            # 测试本地Ollama连接
            try:
                # 尝试访问Ollama的API端点
                test_url = f"{base_url.rstrip('/')}/api/tags"
                response = requests.get(test_url, timeout=5)
                if response.status_code == 200:
                    return jsonify({'success': True, 'message': '本地模型连接成功'})
                else:
                    return jsonify({'success': False, 'error': f'连接失败: HTTP {response.status_code}'})
            except requests.exceptions.Timeout:
                return jsonify({'success': False, 'error': '连接超时，请检查URL和网络'})
            except requests.exceptions.ConnectionError:
                return jsonify({'success': False, 'error': '无法连接到服务器，请检查URL'})
            except Exception as e:
                return jsonify({'success': False, 'error': f'连接失败: {str(e)}'})
        
        elif model_type == 'remote':
            # 测试远程API连接
            api_key = model.get('api_key', '')
            api_base = base_url or model.get('api_base', '')
            
            # 这里可以根据不同的API提供商进行测试
            # 例如OpenAI、Anthropic等
            try:
                # 简单的连接测试：尝试访问API基础URL
                if api_base:
                    test_url = f"{api_base.rstrip('/')}/v1/models"
                    headers = {'Authorization': f'Bearer {api_key}'}
                    response = requests.get(test_url, headers=headers, timeout=10)
                    if response.status_code in [200, 401, 403]:  # 401/403表示连接成功但认证失败
                        return jsonify({'success': True, 'message': '远程模型连接成功（认证可能需要检查）'})
                    else:
                        return jsonify({'success': False, 'error': f'连接失败: HTTP {response.status_code}'})
                else:
                    return jsonify({'success': False, 'error': '未提供API基础URL'})
            except requests.exceptions.Timeout:
                return jsonify({'success': False, 'error': '连接超时，请检查URL和网络'})
            except requests.exceptions.ConnectionError:
                return jsonify({'success': False, 'error': '无法连接到服务器，请检查URL'})
            except Exception as e:
                return jsonify({'success': False, 'error': f'连接失败: {str(e)}'})
        
        else:
            return jsonify({'success': False, 'error': '不支持的模型类型'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/confirm_abstracts', methods=['POST'])
def confirm_abstracts():
    """确认所有论文摘要，继续处理（批量模式）"""
    global confirmed_papers, confirmation_event, running_status, paper_confirmation_papers
    
    try:
        data = request.json
        confirmed_papers_data = data.get('papers', {})  # {paper_id: abstract_text}
        
        # 更新确认后的摘要（批量模式）
        confirmed_papers = confirmed_papers_data
        
        # 更新运行状态
        running_status['waiting_confirmation'] = False
        
        # 清空需要确认的论文列表
        paper_confirmation_papers.clear()
        
        # 通知等待的线程继续执行
        confirmation_event.set()
        
        emit_status('running')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/chat', methods=['POST'])
def chat():
    """处理用户对话请求，使用个人知识库agent回答（支持对话历史）"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')  # 会话ID，用于区分不同会话
        
        if not user_message:
            return jsonify({'success': False, 'error': '消息不能为空'})
        
        # 检查是否有存储的论文
        try:
            from utils.vector_db import get_paper_list
            papers = get_paper_list()
            if not papers:
                return jsonify({
                    'success': False,
                    'error': '当前没有已存储的论文。请先使用"论文PDF"模式上传论文。'
                })
        except Exception as e:
            logging.error(f"检查论文列表失败: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'无法访问论文数据库: {str(e)}'
            })
        
        # 获取对话历史（最近10条）
        try:
            from utils.chat_history import get_chat_history, format_chat_history_for_context
            history = get_chat_history(limit=10, session_id=session_id)
            chat_history_context = format_chat_history_for_context(history, max_length=3000)
        except Exception as e:
            logging.warning(f"获取对话历史失败: {str(e)}")
            chat_history_context = None
        
        # 使用个人知识库agent处理用户问题
        try:
            from crewai import Crew
            from agents.knowledge_base_agent import create_knowledge_base_agent, create_knowledge_base_task
            
            # 创建agent和任务（传入对话历史）
            agent = create_knowledge_base_agent()
            task = create_knowledge_base_task(user_message, chat_history=chat_history_context)
            
            # 创建crew并执行
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True
            )
            
            # 执行任务
            result = crew.kickoff()
            
            # 提取回答（result可能是字符串或对象）
            if hasattr(result, 'raw'):
                response = result.raw
            elif isinstance(result, str):
                response = result
            else:
                response = str(result)
            
            # 保存对话历史
            try:
                from utils.chat_history import save_chat_message
                save_chat_message(
                    user_message=user_message,
                    assistant_message=response,
                    session_id=session_id,
                    metadata={'papers_count': len(papers)}
                )
            except Exception as e:
                logging.warning(f"保存对话历史失败: {str(e)}")
            
            return jsonify({
                'success': True,
                'response': response
            })
            
        except ImportError as import_error:
            missing_module = str(import_error).split("'")[1] if "'" in str(import_error) else "未知模块"
            error_msg = f'缺少必要的依赖模块: {missing_module}。请运行: pip install crewai'
            logging.error(error_msg)
            return jsonify({'success': False, 'error': error_msg})
        except Exception as e:
            error_msg = f'处理对话请求时出错: {str(e)}'
            logging.error(error_msg, exc_info=True)
            return jsonify({'success': False, 'error': error_msg})
            
    except Exception as e:
        logging.error(f"对话API错误: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/emails', methods=['GET'])
def get_emails():
    """获取邮件摘要信息（保留用于兼容性）"""
    try:
        from utils.email_storage import get_email_summary
        summary = get_email_summary()
        return jsonify({'success': True, 'data': summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/emails/all', methods=['GET'])
def get_all_emails():
    """获取所有邮件详细信息"""
    try:
        from utils.email_storage import load_emails
        emails_data = load_emails()
        
        # 计算论文总数
        total_papers = sum(email.get('paper_count', 0) for email in emails_data.get('emails', []))
        emails_data['total_papers'] = total_papers
        
        return jsonify({'success': True, 'data': emails_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/emails/update', methods=['POST'])
def update_emails():
    """手动更新邮件信息（同步远程邮箱到本地）"""
    try:
        data = request.get_json() or {}
        start_days = int(data.get('start_days', 30))  # 默认前30天，确保覆盖足够的时间范围
        end_days = int(data.get('end_days', 0))  # 默认到今天
        
        # 连接邮箱
        mail = connect_gmail()
        
        try:
            # 使用同步函数（不限制数量，获取所有符合条件的邮件）
            from utils.email_storage import sync_remote_emails_to_local, get_pending_papers
            sync_result = sync_remote_emails_to_local(mail, start_days=start_days, end_days=end_days)
            
            # 更新邮件后，检查是否有新的待处理文章
            # 如果有且Agent导入已开启，记录日志提示后台线程开始处理
            if agent_import_enabled:
                pending_papers = get_pending_papers()
                unprocessed_count = len([p for p in pending_papers if 'relevance_score' not in p or p.get('relevance_score') is None])
                if unprocessed_count > 0:
                    logging.info(f"邮件更新完成，发现 {unprocessed_count} 篇待处理文章，后台Agent将开始处理...")
            
            return jsonify({
                'success': True,
                'message': sync_result['message'],
                'updated_count': sync_result['updated_count'],
                'total_count': sync_result['total_count']
            })
        finally:
            # 关闭连接
            try:
                mail.close()
                mail.logout()
            except:
                pass
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/email/config', methods=['GET'])
def get_email_config():
    """获取邮箱配置"""
    try:
        from utils.email_config import load_email_config
        config = load_email_config()
        if config:
            # 返回配置信息（密码字段为空，让用户重新输入）
            return jsonify({
                'success': True,
                'config': {
                    'user': config.get('user', ''),
                    'imap_server': config.get('imap_server', 'imap.qq.com'),
                    'imap_port': config.get('imap_port', 993),
                    'password': ''  # 不返回真实密码，前端显示为空
                }
            })
        else:
            return jsonify({
                'success': True,
                'config': None
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/email/config', methods=['POST'])
def save_email_config():
    """保存邮箱配置"""
    try:
        data = request.get_json() or {}
        user = data.get('user', '').strip()
        password = data.get('password', '').strip()
        imap_server = data.get('imap_server', 'imap.qq.com').strip()
        imap_port = int(data.get('imap_port', 993))
        
        if not user:
            return jsonify({'success': False, 'error': '邮箱账号不能为空'}), 400
        
        # 如果密码为空或为占位符，尝试从现有配置中获取密码
        if not password or password == '********':
            from utils.email_config import load_email_config
            existing_config = load_email_config()
            if existing_config and existing_config.get('user') == user:
                # 使用现有密码
                password = existing_config.get('password', '')
            else:
                return jsonify({'success': False, 'error': '授权码/密码不能为空'}), 400
        
        from utils.email_config import save_email_config
        if save_email_config(user, password, imap_server, imap_port):
            return jsonify({'success': True, 'message': '配置已保存'})
        else:
            return jsonify({'success': False, 'error': '保存配置失败'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/email/test', methods=['POST'])
def test_email_connection():
    """测试邮箱连接"""
    try:
        data = request.get_json() or {}
        user = data.get('user', '').strip()
        password = data.get('password', '').strip()
        imap_server = data.get('imap_server', 'imap.qq.com').strip()
        imap_port = int(data.get('imap_port', 993))
        
        if not user:
            return jsonify({'success': False, 'error': '邮箱账号不能为空'}), 400
        
        # 如果密码为空或为占位符，尝试从现有配置中获取密码
        if not password or password == '********':
            from utils.email_config import load_email_config
            existing_config = load_email_config()
            if existing_config and existing_config.get('user') == user:
                # 使用现有密码
                password = existing_config.get('password', '')
            else:
                return jsonify({'success': False, 'error': '授权码/密码不能为空'}), 400
        
        # 临时使用提供的配置进行连接测试
        import ssl
        import imaplib
        
        try:
            context = ssl.create_default_context()
            mail = imaplib.IMAP4_SSL(imap_server, port=imap_port, ssl_context=context)
            mail.sock.settimeout(10)  # 设置10秒超时
            mail.login(user, password)
            mail.close()
            mail.logout()
            return jsonify({'success': True, 'message': '连接成功'})
        except Exception as e:
            return jsonify({'success': False, 'error': f'连接失败: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Agent导入开关状态（全局变量）
agent_import_enabled = False


@app.route('/api/email/agent-import-status', methods=['GET'])
def get_agent_import_status():
    """获取Agent导入开关状态"""
    global agent_import_enabled
    return jsonify({'success': True, 'enabled': agent_import_enabled})


@app.route('/api/email/agent-import-status', methods=['POST'])
def set_agent_import_status():
    """设置Agent导入开关状态"""
    global agent_import_enabled
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled', False)
        agent_import_enabled = enabled
        
        # 记录日志
        if enabled:
            from utils.email_storage import get_pending_papers
            pending_papers = get_pending_papers()
            unprocessed_count = len([p for p in pending_papers if 'relevance_score' not in p or p.get('relevance_score') is None])
            logging.info(f"Agent导入已开启，发现 {unprocessed_count} 篇待处理文章，后台线程将开始处理...")
        else:
            logging.info("Agent导入已关闭")
        
        return jsonify({'success': True, 'enabled': agent_import_enabled})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/email/pending-papers', methods=['GET'])
def get_pending_papers():
    """获取所有待处理的文章列表（状态为processing的文章，等待用户选择）"""
    try:
        from utils.email_storage import get_processing_papers
        processing_papers = get_processing_papers()
        
        # 只返回相关性为1的文章（等待用户选择）
        relevant_papers = []
        for paper in processing_papers:
            # 如果文章相关性为1，则显示
            if paper.get('relevance_score') == 1:
                relevant_papers.append(paper)
        
        return jsonify({
            'success': True,
            'papers': relevant_papers
        })
    except Exception as e:
        logging.error(f"获取待处理文章失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/email/handle-paper-interest', methods=['POST'])
def handle_paper_interest():
    """处理用户对文章的兴趣选择（感兴趣/不感兴趣）"""
    try:
        data = request.get_json() or {}
        paper_id = data.get('paper_id', '').strip()
        interested = data.get('interested', False)
        
        if not paper_id:
            return jsonify({'success': False, 'error': '文章ID不能为空'}), 400
        
        from utils.email_storage import get_paper_by_id, update_paper_processing_status
        
        # 获取文章信息
        paper = get_paper_by_id(paper_id)
        if not paper:
            return jsonify({'success': False, 'error': '未找到文章'}), 404
        
        if interested:
            # 如果感兴趣，尝试提取摘要/PDF，然后存入数据库
            paper_link = paper.get('link', '')
            paper_title = paper.get('title_en') or paper.get('title', '')
            paper_title_cn = paper.get('title_cn', '')
            paper_title_bilingual = paper.get('title_bilingual', '')
            
            logging.info(f"开始处理感兴趣文章 {paper_id}: title={paper_title}, link={paper_link}")
            
            if paper_link:
                # --- [流程简化] 暂时禁用摘要提取 Agent，直接进入知识处理流程 ---
                logging.info(f"--- [快速入库] 已禁用摘要爬取，直接移交给知识工程师 ---")
                
                # 直接使用邮件中的 snippet 作为摘要
                final_abstract = paper.get('snippet', '')
                
                knowledge_metadata = {
                    'title_en': paper_title,
                    'title_cn': paper_title_cn,
                    'title_bilingual': paper_title_bilingual,
                    'abstract': final_abstract,
                    'authors': paper.get('authors', []),
                    'journal': paper.get('journal', ''),
                    'year': paper.get('year', ''),
                    'link': paper_link,
                    'source': 'email',
                    'tag': 'email'
                }

                def knowledge_engineering_task():
                    """知识工程师处理任务包装函数"""
                    from agents.knowledge_engineer_agent import process_with_knowledge_engineer
                    logging.info(f"--- [知识工程师启动] 正在处理文章快速入库: {paper_title} ---")
                    # 使用知识工程师 Agent 处理，内部根据 source_type='email' 或 'pdf' 执行不同逻辑
                    return process_with_knowledge_engineer(
                        raw_text=final_abstract,
                        metadata=knowledge_metadata,
                        file_path=paper_link,
                        source_type='email'
                    )

                try:
                    # 将知识工程师任务添加到队列
                    task_queue = get_task_queue()
                    ke_task_id = task_queue.add_task(
                        f"ke_{paper_id}_{int(time.time() * 1000)}",
                        'knowledge_engine',
                        knowledge_engineering_task
                    )
                    
                    logging.info(f"知识工程师任务 {ke_task_id} 已添加到队列，开始等待完成...")
                    # 增加等待时间到 120 秒，因为现在是真实的 Agent 在推理入库
                    ke_result = task_queue.wait_for_task(ke_task_id, timeout=120)
                    
                    if ke_result.get('success'):
                        logging.info(f"✓ 快速入库处理完成: {paper_id}")
                        update_paper_processing_status(paper_id, 'processed')
                        return jsonify({
                            'success': True,
                            'message': '文章已通过知识工程师成功处理并入库'
                        })
                    else:
                        raise Exception(f"知识工程师执行失败: {ke_result.get('error')}")
                except Exception as e:
                    logging.error(f"快速入库执行阶段失败: {str(e)}", exc_info=True)
                    return jsonify({'success': False, 'error': f'快速处理失败: {str(e)}'}), 500
            
            else:
                # 如果没有链接，也交给知识工程师直接入库
                logging.info(f"--- [流程流转] 无链接文章，直接提交给知识工程师 ---")
                knowledge_metadata = {
                    'title_en': paper_title,
                    'title_cn': paper_title_cn,
                    'title_bilingual': paper_title_bilingual,
                    'abstract': paper.get('snippet', ''),
                    'authors': paper.get('authors', []),
                    'journal': paper.get('journal', ''),
                    'year': paper.get('year', ''),
                    'link': '',
                    'source': 'email',
                    'tag': 'email'
                }
                
                # 重复调用知识工程师逻辑
                from agents.knowledge_engineer_agent import process_with_knowledge_engineer
                res = process_with_knowledge_engineer(
                    raw_text=knowledge_metadata['abstract'],
                    metadata=knowledge_metadata,
                    file_path=f"email_{paper_id}",
                    source_type='email'
                )
                
                if res.get('success'):
                    update_paper_processing_status(paper_id, 'processed')
                    return jsonify({'success': True, 'message': '文章已成功入库（无链接）'})
                else:
                    return jsonify({'success': False, 'error': res.get('error')}), 500
        else:
            # 不感兴趣，直接标记为已处理
            logging.info(f"文章 {paper_id} 标记为不感兴趣")
            update_paper_processing_status(paper_id, 'processed')
            return jsonify({
                'success': True,
                'message': '已标记为不感兴趣'
            })
        
    except Exception as e:
        logging.error(f"处理文章兴趣失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


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


# 后台处理线程
import threading
import time

# 全局变量存储研究方向关键词
research_keywords_global = '机器人学、控制理论、遥操作、机器人动力学、力控、机器学习'
expanded_keywords_global = None


def background_paper_processor():
    """后台处理待处理文章的线程"""
    global agent_import_enabled, research_keywords_global, expanded_keywords_global
    
    logging.info("后台文章处理线程已启动")
    
    while True:
        try:
            if agent_import_enabled:
                # 获取待处理文章
                from utils.email_storage import get_pending_papers, update_paper_processing_status
                pending_papers = get_pending_papers()
                
                # 只处理还没有relevance_score的文章（未处理过的）
                unprocessed_papers = [p for p in pending_papers if 'relevance_score' not in p or p.get('relevance_score') is None]
                
                if unprocessed_papers:
                    logging.info(f"[Agent处理] 发现 {len(unprocessed_papers)} 篇待处理文章，开始处理...")
                    # 处理第一篇文章
                    paper = unprocessed_papers[0]
                    paper_id = paper.get('paper_id')
                    paper_title = paper.get('title', '')[:50]
                    
                    logging.info(f"开始处理文章: {paper_title}... (ID: {paper_id})")
                    
                    if paper_id:
                        try:
                            # --- 接入任务队列，防止资源竞争 ---
                            logging.info(f"使用关键词: {research_keywords_global[:50]}...")
                            
                            def email_agent_task():
                                """邮件处理 Agent 任务包装函数"""
                                from agents.email_processing_agent import process_email_paper_with_agent
                                return process_email_paper_with_agent(
                                    paper=paper,
                                    research_keywords=research_keywords_global,
                                    expanded_keywords=expanded_keywords_global
                                )

                            # 将后台邮件分析任务添加到全局任务队列
                            task_queue = get_task_queue()
                            task_id = task_queue.add_task(
                                f"email_bg_{paper_id}",
                                'email_process',
                                email_agent_task
                            )
                            
                            # 等待分析完成（最长300秒）
                            task_res = task_queue.wait_for_task(task_id, timeout=300)
                            
                            if task_res.get('success'):
                                result = task_res.get('result')
                                relevance_score = result.get('relevance_score', 0)
                                
                                # 更新文章信息
                                # 相关性为1时标记为processing（等待用户选择），否则标记为processed
                                update_paper_processing_status(
                                    paper_id,
                                    'processing' if relevance_score == 1 else 'processed',
                                    title_en=result.get('title_en'),
                                    title_cn=result.get('title_cn'),
                                    title_bilingual=result.get('title_bilingual'),
                                    relevance_score=relevance_score,
                                    relevance_explanation=result.get('relevance_explanation', '')
                                )
                                
                                logging.info(f"✓ 文章 {paper_id} 队列分析完成，相关性: {relevance_score}")
                            else:
                                logging.error(f"✗ 邮件文章队列分析失败: {task_res.get('error')}")
                                # 分析失败也标记为已处理，避免重复处理
                                update_paper_processing_status(paper_id, 'processed')
                        except Exception as e:
                            logging.error(f"✗ 处理文章 {paper_id} 时出错: {str(e)}", exc_info=True)
                            # 出错也标记为已处理，避免重复处理
                            try:
                                update_paper_processing_status(paper_id, 'processed')
                            except:
                                pass
                else:
                    # 没有待处理文章时，每30秒记录一次日志（避免日志过多）
                    import random
                    if random.randint(1, 6) == 1:  # 约每30秒记录一次（5秒*6）
                        logging.info(f"[Agent处理] 没有待处理的文章（共 {len(pending_papers)} 篇已处理），继续等待...")
            else:
                # Agent导入未开启时，每60秒记录一次日志
                import random
                if random.randint(1, 12) == 1:  # 约每60秒记录一次（5秒*12）
                    logging.info("Agent导入未开启，等待中...")
            
            # 等待5秒后再次检查
            time.sleep(5)
            
        except Exception as e:
            logging.error(f"后台处理线程出错: {str(e)}", exc_info=True)
            time.sleep(10)  # 出错后等待更长时间


# 启动后台处理线程
background_thread = threading.Thread(target=background_paper_processor, daemon=True)
background_thread.start()
logging.info("后台文章处理线程已创建并启动")


@app.route('/api/settings/research-keywords', methods=['POST'])
def update_research_keywords():
    """更新研究方向关键词（用于后台处理）"""
    global research_keywords_global, expanded_keywords_global
    try:
        data = request.get_json() or {}
        keywords = data.get('keywords', '').strip()
        expanded = data.get('expanded_keywords', None)
        
        if keywords:
            research_keywords_global = keywords
            logging.info(f"研究方向关键词已更新: {keywords[:50]}...")
        if expanded:
            expanded_keywords_global = expanded
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/email/processor-status', methods=['GET'])
def get_processor_status():
    """获取后台处理线程状态（用于调试）"""
    global agent_import_enabled, research_keywords_global
    try:
        from utils.email_storage import get_pending_papers, get_processing_papers
        
        pending_count = len(get_pending_papers())
        processing_count = len(get_processing_papers())
        unprocessed_count = len([p for p in get_pending_papers() if 'relevance_score' not in p or p.get('relevance_score') is None])
        
        return jsonify({
            'success': True,
            'agent_import_enabled': agent_import_enabled,
            'research_keywords': research_keywords_global[:50] + '...' if len(research_keywords_global) > 50 else research_keywords_global,
            'pending_count': pending_count,
            'processing_count': processing_count,
            'unprocessed_count': unprocessed_count,
            'thread_alive': background_thread.is_alive() if background_thread is not None else False
        })
    except Exception as e:
        logging.error(f"获取处理线程状态失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

def run_summarizer_task(config):
    """在后台线程中运行summarizer任务"""
    global running_status, paper_confirmation_papers, confirmed_papers
    
    try:
        # 清空之前的确认状态
        paper_confirmation_papers.clear()
        confirmed_papers.clear()
        
        # 设置环境变量（必须在导入main之前设置）
        if config.get('mode') == 'local':
            os.environ['LOCAL'] = '1'
            csv_path = config.get('csv_path', 'local_papers.csv')
            os.environ['CSV_FILE_PATH'] = csv_path
            emit_log('info', f'本地CSV模式：CSV文件路径 = {csv_path}')
            
            # 先导入CSV数据到数据库
            try:
                from utils.csv_importer import import_csv_to_database
                import_result = import_csv_to_database(csv_path)
                if import_result['success']:
                    emit_log('info', f'成功导入 {import_result["imported_count"]} 篇论文到数据库')
                else:
                    emit_log('warning', f'CSV导入数据库时出现问题: {import_result.get("error", "未知错误")}')
            except Exception as e:
                emit_log('warning', f'CSV导入数据库失败，将继续处理: {str(e)}')
                logging.warning(f"CSV导入数据库失败: {str(e)}", exc_info=True)
        elif config.get('mode') == 'pdf':
            # PDF模式：将PDF存储到向量数据库（支持多文件）
            pdf_paths = config.get('pdf_paths', [])
            # 兼容旧代码：如果只有pdf_path，转换为列表
            if not pdf_paths:
                pdf_path = config.get('pdf_path', '')
                if pdf_path:
                    pdf_paths = [pdf_path]
            
            if not pdf_paths:
                error_msg = '未找到PDF文件'
                emit_log('error', error_msg)
                logging.error(error_msg)
                running_status['is_running'] = False
                emit_status('completed')
                return
            
            # 过滤存在的文件
            existing_pdf_paths = [p for p in pdf_paths if os.path.exists(p)]
            if not existing_pdf_paths:
                error_msg = '所有PDF文件都不存在'
                emit_log('error', error_msg)
                logging.error(error_msg)
                running_status['is_running'] = False
                emit_status('completed')
                return
            
            emit_log('info', f'论文PDF模式：共 {len(existing_pdf_paths)} 个PDF文件')
            
            # 检查依赖
            try:
                import sentence_transformers
            except ImportError as import_error:
                missing_module = str(import_error).split("'")[1] if "'" in str(import_error) else "未知模块"
                error_msg = f'缺少必要的依赖模块: {missing_module}。请运行: pip install sentence-transformers'
                emit_log('error', error_msg)
                logging.error(error_msg)
                running_status['is_running'] = False
                emit_status('completed')
                return
            
            try:
                from utils.vector_db import process_and_store_pdf
                
                # 逐一处理每个PDF文件
                success_count = 0
                fail_count = 0
                
                for idx, pdf_path in enumerate(existing_pdf_paths, 1):
                    filename = os.path.basename(pdf_path)
                    emit_log('info', f'[{idx}/{len(existing_pdf_paths)}] 正在处理: {filename}')
                    emit_log('info', '正在将PDF存储到向量数据库...')
                    emit_log('info', '正在从PDF提取论文标题...')
                    
                    # 使用新的处理流程：复制、分析、重命名、存储
                    success, message, pdf_metadata = process_and_store_pdf(
                        pdf_path=pdf_path,
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    
                    if success:
                        success_count += 1
                        emit_log('info', f'✓ {filename} 处理成功')
                        emit_log('info', message)
                        if pdf_metadata:
                            title = pdf_metadata.get('title', '未知标题')
                            authors = pdf_metadata.get('authors', '')
                            if authors:
                                emit_log('info', f'  标题: {title}')
                                emit_log('info', f'  作者: {authors}')
                            else:
                                emit_log('info', f'  标题: {title}')
                    else:
                        fail_count += 1
                        emit_log('error', f'✗ {filename} 处理失败: {message}')
                
                # 总结
                emit_log('info', f'处理完成：成功 {success_count} 个，失败 {fail_count} 个')
                if success_count > 0:
                    emit_log('info', '现在可以使用RAG功能查询论文内容了')
                
                # PDF处理完成，更新状态
                running_status['is_running'] = False
                emit_log('success', 'PDF处理任务完成！')
                emit_status('completed')
                return
            except ImportError as import_error:
                missing_module = str(import_error).split("'")[1] if "'" in str(import_error) else "未知模块"
                error_msg = f'导入模块失败: {missing_module}。请确保已安装所有依赖: pip install -r requirements.txt'
                emit_log('error', error_msg)
                logging.error(f"导入错误: {error_msg}", exc_info=True)
                running_status['is_running'] = False
                emit_status('completed')
                return
            except OSError as os_error:
                # 网络连接错误（无法下载模型）
                error_msg = str(os_error)
                if 'huggingface' in error_msg.lower() or 'connection' in error_msg.lower():
                    detailed_msg = (
                        '无法连接到 Hugging Face 下载嵌入模型。\n'
                        '解决方案：\n'
                        '1. 检查网络连接，确保可以访问 https://huggingface.co\n'
                        '2. 手动下载模型（在有网络的机器上运行）：\n'
                        '   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(\'paraphrase-multilingual-MiniLM-L12-v2\')"\n'
                        '3. 如果在中国大陆，可能需要使用代理或镜像站\n'
                        '4. 将下载的模型复制到当前机器的缓存目录'
                    )
                    emit_log('error', detailed_msg)
                else:
                    emit_log('error', f'PDF存储失败: {error_msg[:200]}')
                logging.error(f"PDF存储异常详情: {error_msg}", exc_info=True)
                running_status['is_running'] = False
                emit_status('completed')
                return
            except Exception as e:
                error_msg = str(e)
                emit_log('error', f'PDF存储到向量数据库时发生异常: {error_msg[:200]}')
                logging.error(f"PDF存储异常详情: {error_msg}", exc_info=True)
                running_status['is_running'] = False
                emit_status('completed')
                return
        else:
            os.environ['LOCAL'] = '0'
            os.environ['START_DAYS'] = str(config.get('start_days', 1))
            os.environ['END_DAYS'] = str(config.get('end_days', 0))
            emit_log('info', f'邮箱模式：日期范围 = 前{config.get("start_days", 1)}天到前{config.get("end_days", 0)}天')
        
        # 设置研究方向关键词
        keywords = config.get('keywords', '机器人学、控制理论、遥操作、机器人动力学、力控、机器学习')
        os.environ['RESEARCH_KEYWORDS'] = keywords
        
        emit_log('info', '开始处理任务...')
        emit_log('info', f'配置信息：mode={config.get("mode")}, csv_path={config.get("csv_path", "N/A")}')
        
        # 重新加载config模块以确保使用最新的环境变量
        import importlib
        if 'config' in sys.modules:
            importlib.reload(sys.modules['config'])
        
        # 重新加载main模块以确保使用最新的config值
        # 注意：需要先删除已导入的main模块，然后重新导入
        if 'main' in sys.modules:
            del sys.modules['main']
        
        # 重新导入main（会使用重新加载后的config模块）
        from main import main
        
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
            
            # get_confirmed_abstracts 已在上面定义
            
            def paper_ready_for_confirmation(paper_id, paper_data):
                """单个论文准备确认（立即显示可编辑摘要框，不阻塞）"""
                global paper_confirmation_papers
                # 记录需要确认的论文（不创建阻塞事件，只记录）
                paper_confirmation_papers[paper_id] = paper_data
                # 立即显示可编辑摘要框（不阻塞）
                emit_paper_ready_for_confirmation(paper_id, paper_data)
            
            def get_confirmed_abstracts():
                """获取所有确认后的摘要（批量模式，阻塞等待用户确认所有论文）"""
                global paper_confirmation_papers, confirmation_event, confirmed_papers, running_status
                
                # 更新状态为等待人工确认
                running_status['waiting_confirmation'] = True
                running_status['confirmation_papers'] = paper_confirmation_papers.copy()
                
                # 发送等待确认消息（批量模式）
                # 即使paper_confirmation_papers为空，也发送消息，让前端知道可以继续
                if paper_confirmation_papers:
                    emit_waiting_confirmation(paper_confirmation_papers)
                else:
                    # 如果没有需要确认的论文，发送空字典，但仍然等待确认
                    emit_waiting_confirmation({})
                
                emit_status('waiting_confirmation')
                
                # 清空之前确认的数据
                confirmed_papers = {}
                
                # 等待确认事件（阻塞等待用户点击确认按钮）
                confirmation_event.clear()
                confirmation_event.wait()  # 阻塞等待确认
                
                # 返回确认后的摘要字典 {paper_id: abstract_text}
                # 如果用户没有确认任何论文，返回空字典，系统会使用原始摘要
                return confirmed_papers
            
            main(
                on_log=emit_log,
                on_paper_added=emit_paper_added,
                on_paper_updated=emit_paper_updated,
                on_paper_removed=emit_paper_removed,
                on_file_generated=emit_file_generated,
                on_agent_status=emit_agent_status,
                on_waiting_confirmation=wait_for_confirmation,
                get_confirmed_abstracts=get_confirmed_abstracts,  # 使用批量确认
                on_paper_ready_for_confirmation=paper_ready_for_confirmation,
                get_confirmed_abstract=None  # 不使用单篇确认
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

# --- 笔记系统接口 ---

@app.route('/api/utils/select-folder', methods=['GET'])
def api_select_folder():
    """打开原生对话框选择文件夹"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes('-topmost', True)  # 确保对话框在最前面
        
        folder_path = filedialog.askdirectory()
        root.destroy()
        
        if folder_path:
            return jsonify({'success': True, 'path': folder_path})
        return jsonify({'success': False, 'message': '取消选择'})
    except Exception as e:
        logging.error(f"打开文件夹选择对话框失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/note/settings', methods=['GET'])
def api_get_note_settings():
    """获取笔记设置"""
    return jsonify(get_note_settings())

@app.route('/api/note/settings', methods=['POST'])
def api_save_note_settings():
    """保存笔记设置（合并现有设置）"""
    new_settings = request.json
    current_settings = get_note_settings()
    
    # 合并设置
    current_settings.update(new_settings)
    
    if save_note_settings(current_settings):
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': '保存设置失败'}), 500

@app.route('/api/note/files', methods=['GET'])
def api_get_note_files():
    """扫描笔记文件夹并返回文件列表，同时监控文件变化"""
    try:
        settings = get_note_settings()
        note_path = settings.get('note_path')
        
        if not note_path or not os.path.exists(note_path):
            return jsonify({'success': False, 'error': '笔记路径未设置或不存在'})
            
        # 支持的文件类型
        supported_exts = {'.txt', '.md', '.docx', '.doc', '.json', '.yaml', '.csv', '.xls', '.xlsx', '.pdf'}
        
        files_list = []
        file_status_data = get_note_file_status() # 格式: {file_id: {"status": "...", "mtime": ...}}
        
        # 记录本次扫描发现的文件 ID
        current_scan_ids = set()
        status_changed = False
        
        for root, dirs, files in os.walk(note_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in supported_exts:
                    # 在 Windows 上，确保文件名和路径是正确的 Unicode
                    try:
                        # 尝试通过系统编码纠正（针对部分旧环境乱码）
                        if isinstance(file, bytes):
                            file = file.decode('utf-8')
                    except:
                        pass
                        
                    rel_path = os.path.relpath(file_path, note_path)
                    mod_time = os.path.getmtime(file_path)
                    # 使用秒级整数进行对比，避免浮点数精度误差
                    mod_time_int = int(mod_time)
                    
                    # 使用相对路径作为唯一ID
                    file_id = rel_path.replace('\\', '/')
                    current_scan_ids.add(file_id)
                    
                    # 获取旧记录
                    old_record = file_status_data.get(file_id)
                    
                    if isinstance(old_record, str):
                        # 兼容极旧版本格式 (只有字符串状态)
                        old_record = {"status": old_record, "mtime": 0}
                    
                    # 检查是否发生变化
                    if not old_record:
                        # 只有在完全没有记录时才设为 pending
                        file_status_data[file_id] = {"status": "pending", "mtime": mod_time_int}
                        status_changed = True
                    else:
                        # 获取已有的状态和时间（兼容处理）
                        old_mtime = old_record.get('mtime', 0)
                        # 如果 old_mtime 是浮点数，也转为整数对比
                        if abs(int(old_mtime) - mod_time_int) > 1:
                            # 只有时间偏差超过1秒才认为文件已更改
                            file_status_data[file_id]["status"] = "pending"
                            file_status_data[file_id]["mtime"] = mod_time_int
                            status_changed = True
                            emit_log('info', f"检测到文件内容更新: {file}")
                    
                    current_record = file_status_data[file_id]
                    
                    files_list.append({
                        'id': file_id,
                        'name': file,
                        'path': file_path,
                        'rel_path': rel_path,
                        'type': file_ext.replace('.', ''),
                        'mod_time': datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S'),
                        'status': current_record.get('status', 'pending')
                    })
        
        # 清理已删除的文件记录
        for fid in list(file_status_data.keys()):
            if fid not in current_scan_ids:
                del file_status_data[fid]
                status_changed = True
        
        if status_changed:
            save_note_file_status(file_status_data)
        
        return jsonify({
            'success': True, 
            'files': files_list,
            'note_path': note_path
        })
    except Exception as e:
        logging.error(f"扫描笔记文件失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# 全局变量，用于控制笔记后台处理线程
note_feeder_thread = None
note_feeder_stop_event = threading.Event()

def start_note_feeder():
    """启动笔记分发器线程"""
    global note_feeder_thread
    if note_feeder_thread is None or not note_feeder_thread.is_alive():
        note_feeder_stop_event.clear()
        note_feeder_thread = threading.Thread(target=note_feeder_worker, daemon=True)
        note_feeder_thread.start()
        logging.info("笔记分发器后台线程已启动")

def note_feeder_worker():
    """笔记分发器工作循环：按需向队列添加任务"""
    task_queue = get_task_queue()
    while not note_feeder_stop_event.is_set():
        try:
            # 1. 检查开关状态
            settings = get_note_settings()
            if not settings.get('agent_import_active', False):
                # 如果开关关闭，稍微休息后继续检查（不退出线程，以便随时响应重新开启）
                time.sleep(5)
                continue

            # 2. 检查队列中是否已经有笔记导入任务
            # 我们通过任务ID前缀来判断，允许每种类型同时存在 2 个任务
            active_note_tasks_count = 0
            with task_queue.queue_lock:
                for tid, task in task_queue.tasks.items():
                    if tid.startswith("note_import_") and task.status.value in ["pending", "running"]:
                        active_note_tasks_count += 1
            
            if active_note_tasks_count >= 2:
                # 队列中已有 2 个任务，等待处理
                time.sleep(5)
                continue

            # 3. 寻找一个待处理文件
            note_path = settings.get('note_path')
            if not note_path or not os.path.exists(note_path):
                time.sleep(10)
                continue

            file_status_data = get_note_file_status()
            supported_exts = {'.txt', '.md', '.docx', '.doc', '.json', '.yaml', '.csv', '.xls', '.xlsx', '.pdf'}
            
            target_file = None
            # 注意：我们需要过滤掉已经在队列中的文件，避免重复添加
            queued_file_ids = set()
            with task_queue.queue_lock:
                for tid in task_queue.tasks.keys():
                    if tid.startswith("note_import_"):
                        queued_file_ids.add(tid.replace("note_import_", ""))

            for root, dirs, files in os.walk(note_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for file in files:
                    if file.startswith('.'): continue
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in supported_exts:
                        rel_path = os.path.relpath(file_path, note_path)
                        file_id = rel_path.replace('\\', '/')
                        
                        # 只有不在队列中 且 状态为 unprocessed 的文件才会被选中
                        if file_id not in queued_file_ids:
                            record = file_status_data.get(file_id)
                            status = record.get('status') if isinstance(record, dict) else record
                            
                            # 自动检测文件修改：如果 mtime 变了，重置为 unprocessed
                            file_mtime = os.path.getmtime(file_path)
                            if isinstance(record, dict) and record.get('mtime') != file_mtime:
                                status = 'unprocessed'
                                record['status'] = 'unprocessed'
                                record['mtime'] = file_mtime
                                file_status_data[file_id] = record
                                save_note_file_status(file_status_data)
                                logging.info(f"检测到笔记修改，重置状态: {file_id}")
                            
                            if status == 'unprocessed' or status == 'pending': # 兼容旧的 pending 状态
                                target_file = (file_id, file_path)
                                break
                if target_file: break

            if target_file:
                file_id, file_path = target_file
                
                # 定义任务逻辑（闭包）
                def make_note_task(f_id, f_path):
                    def note_task_logic():
                        try:
                            emit_log('info', f'正在解析笔记: {os.path.basename(f_path)}')
                            parse_result = parse_file_with_agent(f_path)
                            if not parse_result.get('success', False):
                                raise Exception(parse_result.get('error', '解析失败'))
                            
                            p_res = parse_result.get('parse_result', {})
                            if isinstance(p_res, str):
                                try: p_res = json.loads(p_res)
                                except: pass
                            
                            # 获取元数据，并确保有标题
                            metadata = p_res.get('metadata', {})
                            if not metadata.get('title'):
                                # 使用文件名（去除后缀）作为标题
                                filename = os.path.basename(f_path)
                                metadata['title'] = os.path.splitext(filename)[0]
                            
                            raw_text = p_res.get('content') or p_res.get('raw_text') or ""
                            emit_log('info', f'正在将笔记导入数据库: {metadata["title"]}')
                            
                            ke_result = process_with_knowledge_engineer(
                                raw_text=raw_text,
                                metadata=metadata,
                                file_path=f_path,
                                source_type='note'
                            )
                            
                            if ke_result.get('success', False):
                                # 只有成功时才更新状态为已处理
                                current_status = get_note_file_status()
                                current_status[f_id] = {"status": "processed", "mtime": os.path.getmtime(f_path)}
                                save_note_file_status(current_status)
                                emit_log('success', f'笔记处理完成: {metadata["title"]}，等待大脑同步')
                                socketio.emit('message', {'type': 'note_file_updated', 'file_id': f_id, 'status': 'processed'})
                            else:
                                error_msg = ke_result.get('error', '未知错误')
                                emit_log('error', f'笔记导入失败 ({metadata["title"]}): {error_msg}')
                                raise Exception(f"知识工程师导入失败: {error_msg}")
                        except Exception as e:
                            logging.error(f"处理笔记失败: {str(e)}")
                            emit_log('error', f"处理笔记失败 ({os.path.basename(f_path)}): {str(e)}")
                    return note_task_logic

                # 添加到队列（一次只添加一个）
                task_queue.add_task(f"note_import_{file_id}", "note_import", make_note_task(file_id, file_path))
                logging.info(f"笔记分发器：已将 {file_id} 添加到队列")
            else:
                # 没有待处理文件了，休息久一点
                time.sleep(30)

        except Exception as e:
            logging.error(f"笔记分发器异常: {str(e)}")
            time.sleep(10)

@app.route('/api/note/agent-import', methods=['POST'])
def api_note_agent_import():
    """启动/停止笔记 Agent 导入任务（仅更新设置并确保分发器运行）"""
    try:
        data = request.json
        is_active = data.get('active', False)
        
        # 更新设置以持久化开关状态
        current_settings = get_note_settings()
        current_settings['agent_import_active'] = is_active
        save_note_settings(current_settings)
        
        if is_active:
            start_note_feeder()
            return jsonify({'success': True, 'message': '笔记 Agent 导入已开启（按需分发任务）'})
        else:
            return jsonify({'success': True, 'message': '笔记 Agent 导入已停止（剩余任务将继续执行）'})
            
    except Exception as e:
        logging.error(f"操作笔记导入开关失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        logging.error(f"启动笔记导入失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/brain/update', methods=['POST'])
def api_brain_update():
    """手动触发大脑认知边界更新"""
    try:
        task_queue = get_task_queue()
        task_id = task_queue.add_task(f"brain_update_{int(time.time())}", "brain_update", trigger_brain_context_update)
        return jsonify({'success': True, 'task_id': task_id, 'message': '大脑更新任务已添加到队列'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/brain/think/<paper_id>', methods=['POST'])
def api_brain_think(paper_id):
    """手动触发单篇论文深度分析"""
    try:
        task_queue = get_task_queue()
        task_id = task_queue.add_task(f"paper_think_{paper_id}_{int(time.time())}", "paper_think", trigger_paper_depth_analysis, paper_id)
        return jsonify({'success': True, 'task_id': task_id, 'message': f'论文 {paper_id} 深度分析任务已添加到队列'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/brain/think_all', methods=['POST'])
def api_brain_think_all():
    """触发全库未分析论文的深度分析"""
    try:
        from utils.vector_db import get_db_connection, return_db_connection
        conn = get_db_connection()
        try:
            from psycopg2.extras import RealDictCursor
            cur = conn.cursor(cursor_factory=RealDictCursor)
            # 查找有正文但没有 think_points 的论文
            cur.execute("SELECT paper_id FROM papers WHERE (content IS NOT NULL AND content != '') AND (think_points IS NULL OR think_points = 'null'::jsonb)")
            papers = cur.fetchall()
            
            task_queue = get_task_queue()
            count = 0
            for p in papers:
                paper_id = p['paper_id']
                task_queue.add_task(f"paper_think_{paper_id}_{int(time.time())}", "paper_think", trigger_paper_depth_analysis, paper_id)
                count += 1
            
            return jsonify({'success': True, 'count': count, 'message': f'已将 {count} 篇论文的分析任务添加到队列'})
        finally:
            return_db_connection(conn)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def brain_analysis_loop():
    """后台循环：当开关开启时，自动将待分析文章加入队列"""
    while True:
        try:
            settings = get_note_settings()
            # 1. 处理论文深度分析 (think 点和 summary)
            if settings.get('brain_analysis_active', False):
                task_queue = get_task_queue()
                
                # 检查当前已经在运行的分析任务数
                running_count = 0
                with task_queue.queue_lock:
                    for t in task_queue.tasks.values():
                        if t.task_type == 'paper_think' and t.status.value in ['pending', 'running']:
                            running_count += 1
                
                if running_count < 2:
                    from utils.vector_db import get_db_connection, return_db_connection
                    conn = get_db_connection()
                    try:
                        from psycopg2.extras import RealDictCursor
                        cur = conn.cursor(cursor_factory=RealDictCursor)
                        # 找一篇在 paper_chunks 中有内容但没 think 点的文章 (PDF/Email等)
                        cur.execute("""
                            SELECT p.paper_id FROM papers p
                            WHERE p.source != 'note'
                            AND (p.think_points IS NULL OR p.think_points = 'null'::jsonb OR p.think_points = '[]'::jsonb)
                            AND EXISTS (SELECT 1 FROM paper_chunks pc WHERE pc.paper_id = p.paper_id)
                            LIMIT 1
                        """)
                        paper = cur.fetchone()
                        if paper:
                            p_id = paper['paper_id']
                            task_queue.add_task(f"paper_think_{p_id}", "paper_think", trigger_paper_depth_analysis, p_id)
                    finally:
                        return_db_connection(conn)

            # 2. 处理笔记大脑同步 (已处理 -> 已同步)
            if settings.get('brain_sync_active', False):
                task_queue = get_task_queue()
                
                # 检查当前已经在运行的同步任务数
                running_sync_count = 0
                with task_queue.queue_lock:
                    for t in task_queue.tasks.values():
                        if t.task_type == 'brain_sync' and t.status.value in ['pending', 'running']:
                            running_sync_count += 1
                
                if running_sync_count < 2:
                    file_status_data = get_note_file_status()
                    target_note_file = None
                    
                    # 寻找一个状态为 'processed' 的笔记进行同步
                    for f_id, info in file_status_data.items():
                        status = info.get('status') if isinstance(info, dict) else info
                        if status == 'processed':
                            # 获取对应的 paper_id 和内容
                            from utils.vector_db import get_db_connection, return_db_connection
                            conn = get_db_connection()
                            try:
                                from psycopg2.extras import RealDictCursor
                                cur = conn.cursor(cursor_factory=RealDictCursor)
                                # 这里的 f_id 使用正斜杠，但数据库中可能存的是反斜杠，进行兼容性匹配
                                f_id_normalized = f_id.replace('/', '%')
                                cur.execute("""
                                    SELECT paper_id, content FROM papers 
                                    WHERE REPLACE(obsidian_note_path, '\\', '/') LIKE %s 
                                    OR REPLACE(attachment_path, '\\', '/') LIKE %s
                                """, (f'%{f_id}%', f'%{f_id}%'))
                                row = cur.fetchone()
                                
                                if not row:
                                    # 进一步宽松匹配：只匹配文件名部分
                                    filename_only = f_id.split('/')[-1]
                                    cur.execute("""
                                        SELECT paper_id, content FROM papers 
                                        WHERE (obsidian_note_path LIKE %s OR attachment_path LIKE %s OR title LIKE %s)
                                        AND source = 'note'
                                    """, (f'%{filename_only}%', f'%{filename_only}%', f'%{filename_only.rsplit(".", 1)[0]}%'))
                                    row = cur.fetchone()
                                    if row:
                                        logging.info(f"路径精确匹配失败，已通过文件名宽松匹配成功: {filename_only} -> {row['paper_id']}")

                                if row:
                                    p_id = row['paper_id']
                                    content = row['content']
                                    
                                    # 如果内容为空，直接标记为已同步
                                    if not content or not content.strip():
                                        logging.info(f"笔记 {f_id} 内容为空，跳过同步过程，直接标记为‘已同步’")
                                        current_status = get_note_file_status()
                                        if f_id in current_status:
                                            if isinstance(current_status[f_id], dict):
                                                current_status[f_id]['status'] = 'synchronized'
                                            else:
                                                current_status[f_id] = 'synchronized'
                                            save_note_file_status(current_status)
                                        socketio.emit('message', {'type': 'note_file_updated', 'file_id': f_id, 'status': 'synchronized'})
                                        continue # 继续找下一个
                                        
                                    target_note_file = (f_id, p_id)
                                    break
                            finally:
                                return_db_connection(conn)
                    
                    if target_note_file:
                        f_id, p_id = target_note_file
                        
                        def make_sync_task(file_id, paper_id):
                            def sync_task_logic():
                                try:
                                    emit_log('info', f'正在将笔记同步至大脑: {file_id}')
                                    res = trigger_brain_context_update(paper_id)
                                    if res.get('success'):
                                        # 更新状态为 synchronized
                                        current_status = get_note_file_status()
                                        if file_id in current_status:
                                            if isinstance(current_status[file_id], dict):
                                                current_status[file_id]['status'] = 'synchronized'
                                            else:
                                                current_status[file_id] = 'synchronized'
                                            save_note_file_status(current_status)
                                        emit_log('success', f'笔记同步完成: {file_id}')
                                        socketio.emit('message', {'type': 'note_file_updated', 'file_id': file_id, 'status': 'synchronized'})
                                        return True
                                    else:
                                        raise Exception(res.get('error', '同步失败'))
                                except Exception as e:
                                    emit_log('error', f'同步笔记 {file_id} 失败: {str(e)}')
                                    raise e
                            return sync_task_logic

                        task_queue.add_task(f"brain_sync_{p_id}", "brain_sync", make_sync_task(f_id, p_id))

        except Exception as e:
            logging.error(f"论文分析/同步轮询异常: {e}")
            
        time.sleep(10) # 每10秒检查一次

@app.route('/api/brain/context', methods=['GET'])
def api_get_brain_context():
    """获取大脑当前认知上下文"""
    try:
        from utils.brain_context_utils import get_brain_context
        return jsonify(get_brain_context())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# --------------------

@app.route('/api/settings/external_services', methods=['GET'])
def get_external_services():
    """获取外部服务设置"""
    try:
        import json
        import os
        settings_file = os.path.join(os.getcwd(), 'external_services.json')
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                return jsonify({'success': True, **settings})
        return jsonify({'success': True, 'firecrawl_api_key': ''})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/settings/external_services', methods=['POST'])
def save_external_services():
    """保存外部服务设置"""
    try:
        import json
        import os
        data = request.json
        settings_file = os.path.join(os.getcwd(), 'external_services.json')
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/system/open_folder', methods=['POST'])
def open_system_folder():
    """在文件管理器中打开并定位文件"""
    try:
        data = request.json
        file_path = data.get('path', '')
        if not file_path:
            return jsonify({'success': False, 'error': '未提供路径'})
            
        import subprocess
        import os
        
        # 转换路径为绝对路径
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return jsonify({'success': False, 'error': f'文件不存在: {abs_path}'})
            
        # Windows 下打开并定位文件
        subprocess.run(['explorer', '/select,', os.path.normpath(abs_path)])
        
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"打开文件夹失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/send', methods=['POST'])
def chat_send():
    """发送对话消息"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        mode = data.get('mode', 'query')  # 'query' 或 'explore'
        web_search_enabled = data.get('web_search_enabled', False)
        
        # 探索模式下允许空消息触发自我探索
        if not message and mode != 'explore':
            return jsonify({'success': False, 'error': '消息不能为空'})
        
        from agents.chat_agent import process_chat
        
        result = process_chat(
            user_message=message,
            mode=mode,
            web_search_enabled=web_search_enabled
        )
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"对话处理失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """获取对话历史"""
    try:
        from utils.chat_history import load_chat_history
        history = load_chat_history()
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        logging.error(f"获取对话历史失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/history', methods=['DELETE'])
def clear_chat_history():
    """清空对话历史"""
    try:
        from utils.chat_history import clear_chat_history
        clear_chat_history()
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"清空对话历史失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/content/generate', methods=['POST'])
def generate_content_note():
    """生成工程师：从知识点内容创建Obsidian笔记"""
    try:
        data = request.json
        topic = data.get('topic', '').strip()
        content = data.get('content', '').strip()
        tags = data.get('tags', '').strip()
        
        if not topic:
            return jsonify({'success': False, 'error': '主题不能为空'})
        if not content:
            return jsonify({'success': False, 'error': '内容不能为空'})
        
        from agents.content_generator_agent import generate_note_from_content
        
        result = generate_note_from_content(
            topic=topic,
            content=content,
            tags=tags if tags else None
        )
        
        # 如果成功创建笔记，尝试自动导入到知识库
        if result.get('success') and result.get('file_path'):
            try:
                # 触发笔记导入系统重新扫描
                # 这里可以添加自动导入逻辑，或者让用户手动触发扫描
                logging.info(f"笔记已创建，文件路径: {result.get('file_path')}")
            except Exception as e:
                logging.warning(f"自动导入笔记失败（可手动触发扫描）: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"生成笔记失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 80)
    print("Paper Summarizer Web Server")
    print("=" * 80)
    
    # 启动后台处理线程（如果还未启动）
    if background_thread is None or not background_thread.is_alive():
        background_thread = threading.Thread(target=background_paper_processor, daemon=True)
        background_thread.start()
        logging.info("后台文章处理线程已创建并启动（在main中）")
    
    # 初始化数据库（确保所有表都存在）
    try:
        from utils.vector_db import init_database
        print("正在初始化数据库...")
        init_database()
        print("✅ 数据库初始化完成")
    except Exception as e:
        logging.warning(f"数据库初始化警告: {str(e)}")
        print(f"⚠️  数据库初始化警告: {str(e)}")
        print("   如果这是首次运行，请确保PostgreSQL已安装并运行")
        print("   可以手动运行: python utils/init_db.py")
    
    print(f"访问地址: http://localhost:5000")
    print("=" * 80)
    
    # 自动打开浏览器
    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://localhost:5000')
    
    threading.Thread(target=open_browser, daemon=True).start()

    # 检查笔记导入状态，如果开启则启动分发器
    if get_note_settings().get('agent_import_active', False):
        start_note_feeder()
    
    # 启动任务队列监控线程
    threading.Thread(target=queue_status_monitor, daemon=True).start()
    logging.info("任务队列监控线程已启动")
    
    # 启动论文分析与同步轮询线程
    threading.Thread(target=brain_analysis_loop, daemon=True).start()
    logging.info("论文分析与同步轮询线程已启动")
    
    # 启动服务器
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True, use_reloader=False)

