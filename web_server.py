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
from datetime import datetime
import webbrowser
import requests
from utils.model_config import load_models, save_models, validate_model_config, get_active_model
from utils.email_storage import get_email_summary, load_emails
from utils.email_utils import connect_gmail

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
            
            # 处理文件上传（本地模式）
            if 'csv_file' in request.files:
                csv_file = request.files['csv_file']
                if csv_file.filename:
                    # 保存文件到工作目录
                    upload_dir = os.getcwd()
                    file_path = os.path.join(upload_dir, csv_file.filename)
                    csv_file.save(file_path)
                    # 设置文件路径到配置中
                    config['csv_path'] = csv_file.filename
        
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
            os.environ['CSV_FILE_PATH'] = config.get('csv_path', 'local_papers.csv')
            emit_log('info', f'本地模式：CSV文件路径 = {config.get("csv_path", "local_papers.csv")}')
        else:
            os.environ['LOCAL'] = '0'
            os.environ['START_DAYS'] = str(config.get('start_days', 1))
            os.environ['END_DAYS'] = str(config.get('end_days', 0))
            emit_log('info', f'远程模式：日期范围 = 前{config.get("start_days", 1)}天到前{config.get("end_days", 0)}天')
        
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
    print(f"访问地址: http://localhost:5000")
    print("=" * 80)
    
    # 自动打开浏览器
    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://localhost:5000')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 启动服务器
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

