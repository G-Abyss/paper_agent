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
    """返回数据库管理页面"""
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
            from utils.email_storage import sync_remote_emails_to_local
            sync_result = sync_remote_emails_to_local(mail, start_days=start_days, end_days=end_days)
            
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

if __name__ == '__main__':
    print("=" * 80)
    print("Paper Summarizer Web Server")
    print("=" * 80)
    
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
    
    # 启动服务器
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True, use_reloader=False)

