#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Firecrawl 联网搜索工具
"""

import logging
import requests
import json
import os
from typing import List, Dict, Optional

def get_firecrawl_api_key() -> Optional[str]:
    """从配置文件获取Firecrawl API Key"""
    try:
        config_path = os.path.join(os.getcwd(), 'external_services.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('firecrawl_api_key', '').strip()
    except Exception as e:
        logging.error(f"读取Firecrawl API Key失败: {e}")
    return None

def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    使用Firecrawl进行网络搜索
    
    Args:
        query: 搜索查询
        max_results: 最大结果数量
    
    Returns:
        搜索结果列表，每个结果包含 title, url, content, snippet
    """
    api_key = get_firecrawl_api_key()
    if not api_key:
        logging.warning("Firecrawl API Key未配置，无法进行网络搜索")
        return []
    
    try:
        # Firecrawl API endpoint
        url = "https://api.firecrawl.dev/v0/search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "limit": max_results
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if data.get('success') and 'data' in data:
            for item in data['data']:
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'snippet': item.get('description', ''),
                    'content': item.get('content', '')[:500] if item.get('content') else ''  # 限制内容长度
                })
        
        logging.info(f"Firecrawl搜索完成，找到 {len(results)} 个结果")
        return results
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Firecrawl搜索请求失败: {e}")
        return []
    except Exception as e:
        logging.error(f"Firecrawl搜索失败: {e}")
        return []

def crawl_url(url: str) -> Optional[Dict]:
    """
    使用Firecrawl抓取指定URL的内容
    
    Args:
        url: 要抓取的URL
    
    Returns:
        包含 title, content, url 的字典，失败返回None
    """
    api_key = get_firecrawl_api_key()
    if not api_key:
        logging.warning("Firecrawl API Key未配置，无法抓取网页")
        return None
    
    try:
        # Firecrawl API endpoint
        api_url = "https://api.firecrawl.dev/v0/scrape"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "url": url
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('success') and 'data' in data:
            return {
                'title': data['data'].get('title', ''),
                'content': data['data'].get('markdown', '') or data['data'].get('content', ''),
                'url': url
            }
        
        return None
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Firecrawl抓取失败: {e}")
        return None
    except Exception as e:
        logging.error(f"Firecrawl抓取失败: {e}")
        return None

