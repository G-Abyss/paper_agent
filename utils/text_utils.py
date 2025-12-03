#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本处理工具
"""

import re


def remove_ansi_codes(text):
    """
    移除ANSI转义码（终端颜色代码）
    例如：[32m, [0m, [1;32m 等
    """
    # 匹配ANSI转义序列：\x1b[ 或 \033[ 开头，后面跟着数字、分号、字母，以m结尾
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m|\033\[[0-9;]*m')
    return ansi_escape.sub('', text)
