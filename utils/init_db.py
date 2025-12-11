#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库初始化脚本

运行此脚本可以手动初始化PostgreSQL数据库（创建表和扩展）
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector_db import init_database
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    print("=" * 80)
    print("PostgreSQL 数据库初始化")
    print("=" * 80)
    print()
    
    try:
        print("正在初始化数据库...")
        init_database()
        print("\n✅ 数据库初始化成功！")
        print("\n提示：")
        print("1. 数据库表和扩展已创建")
        print("2. 现在可以开始使用系统了")
        print("3. 如果有ChromaDB数据，可以运行迁移脚本：python utils/migrate_chroma_to_postgres.py")
    except Exception as e:
        print(f"\n❌ 数据库初始化失败: {str(e)}")
        print("\n请检查：")
        print("1. PostgreSQL是否已安装并运行")
        print("2. pgvector扩展是否已安装")
        print("3. 数据库连接配置是否正确（.env文件）")
        import traceback
        traceback.print_exc()
        sys.exit(1)

