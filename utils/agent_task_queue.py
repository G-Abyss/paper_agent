#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent任务队列管理器

负责管理所有Agent任务的执行，确保任务按队列顺序执行，避免资源冲突。
"""

import threading
import queue
import logging
import time
from typing import Callable, Dict, Any, Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentTask:
    """Agent任务类"""
    def __init__(self, task_id: str, task_type: str, task_func: Callable, *args, **kwargs):
        self.task_id = task_id
        self.task_type = task_type  # 'file_parse', 'knowledge_engine', 'email_process', 'note_import' 等
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.lock = threading.Lock()
    
    def execute(self):
        """执行任务"""
        with self.lock:
            if self.status == TaskStatus.RUNNING:
                return
            self.status = TaskStatus.RUNNING
            self.started_at = time.time()
        
        try:
            logging.info(f"--- [任务队列] 开始执行任务 {self.task_id} (类型: {self.task_type}) ---")
            self.result = self.task_func(*self.args, **self.kwargs)
            with self.lock:
                self.status = TaskStatus.COMPLETED
                self.completed_at = time.time()
            logging.info(f"--- [任务队列] 任务 {self.task_id} 执行成功，耗时 {self.completed_at - self.started_at:.2f}s ---")
            return self.result
        except Exception as e:
            with self.lock:
                self.status = TaskStatus.FAILED
                self.error = str(e)
                self.completed_at = time.time()
            logging.error(f"--- [任务队列] 任务 {self.task_id} 执行失败: {str(e)} ---")
            raise e


class AgentTaskQueue:
    """Agent任务队列管理器（单例模式）"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AgentTaskQueue, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.task_queue = queue.Queue()
        self.tasks = {}  # task_id -> AgentTask
        self.executor = ThreadPoolExecutor(max_workers=10) # 允许最大10个总线程
        self.dispatcher_thread = None
        self.is_running = False
        self.queue_lock = threading.Lock()
        self._initialized = True
        
        # 每种类型允许的最大并发数
        self.MAX_CONCURRENT_PER_TYPE = 2
        
        # 启动调度线程
        self.start_dispatcher()
    
    def start_dispatcher(self):
        """启动调度线程"""
        if self.dispatcher_thread is None or not self.dispatcher_thread.is_alive():
            self.is_running = True
            self.dispatcher_thread = threading.Thread(target=self._dispatcher_loop, daemon=True)
            self.dispatcher_thread.start()
            logging.info("[任务队列] 任务调度线程已启动")
    
    def _dispatcher_loop(self):
        """核心调度循环：从队列取任务，并根据类型限流并行执行"""
        waiting_tasks = [] # 暂时由于限流无法执行的任务
        
        while self.is_running:
            try:
                # 1. 首先尝试处理之前因为限流而等待的任务
                still_waiting = []
                for task in waiting_tasks:
                    if self._can_run_task(task.task_type):
                        self.executor.submit(task.execute)
                    else:
                        still_waiting.append(task)
                waiting_tasks = still_waiting

                # 2. 从主队列中获取新任务
                try:
                    # 降低轮询频率，避免CPU空转
                    new_task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    if not waiting_tasks:
                        time.sleep(1) # 如果完全没任务，多睡一会儿
                    continue
                
                # 3. 检查新任务是否可以执行
                if self._can_run_task(new_task.task_type):
                    self.executor.submit(new_task.execute)
                else:
                    waiting_tasks.append(new_task)
                    
            except Exception as e:
                logging.error(f"[任务队列] 调度线程异常: {str(e)}", exc_info=True)
                time.sleep(1)

    def _can_run_task(self, task_type: str) -> bool:
        """检查特定类型的任务当前运行数是否小于阈值"""
        running_count = 0
        with self.queue_lock:
            for task in self.tasks.values():
                if task.task_type == task_type and task.status == TaskStatus.RUNNING:
                    running_count += 1
        return running_count < self.MAX_CONCURRENT_PER_TYPE

    def add_task(self, task_id: str, task_type: str, task_func: Callable, *args, **kwargs) -> str:
        """添加任务到队列"""
        if not task_id:
            task_id = f"{task_type}_{int(time.time() * 1000)}"
        
        task = AgentTask(task_id, task_type, task_func, *args, **kwargs)
        
        with self.queue_lock:
            self.tasks[task_id] = task
            self.task_queue.put(task)
        
        logging.info(f"[任务队列] 任务 {task_id} (类型: {task_type}) 已添加到队列")
        self.start_dispatcher()
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        with self.queue_lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            with task.lock:
                return {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'status': task.status.value,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'error': task.error
                }
    
    def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """
        获取任务结果（阻塞等待直到任务完成）
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            任务结果
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")
        
        # 等待任务完成
        start_time = time.time()
        while True:
            with task.lock:
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise Exception(f"任务执行失败: {task.error}")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"等待任务 {task_id} 超时")
            
            time.sleep(0.1)
    
    def wait_for_task(self, task_id: str, timeout: float = None) -> Dict[str, Any]:
        """
        等待任务完成并返回结果
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            
        Returns:
            包含任务结果的字典
        """
        try:
            result = self.get_task_result(task_id, timeout)
            return {
                'success': True,
                'result': result
            }
        except TimeoutError as e:
            return {
                'success': False,
                'error': str(e)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.task_queue.qsize()
    
    def get_running_tasks_count(self) -> int:
        """获取正在运行的任务数量"""
        with self.queue_lock:
            return sum(1 for task in self.tasks.values() if task.status == TaskStatus.RUNNING)


# 全局任务队列实例
_global_queue = None
_queue_lock = threading.Lock()

def get_task_queue() -> AgentTaskQueue:
    """获取全局任务队列实例"""
    global _global_queue
    if _global_queue is None:
        with _queue_lock:
            if _global_queue is None:
                _global_queue = AgentTaskQueue()
    return _global_queue

