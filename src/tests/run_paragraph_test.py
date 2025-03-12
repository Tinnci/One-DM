#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专门运行段落数据集测试的脚本
"""

import os
import sys
import unittest

# 确保项目根目录在 Python 路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from tests.test_paragraph_dataset import TestParagraphDataset, run_tests
except ImportError:
    print("无法导入段落数据集测试模块，请确保项目安装正确")
    sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("开始运行段落数据集测试")
    print("=" * 60)
    
    # 运行测试
    success = run_tests()
    
    print("\n" + "=" * 60)
    print(f"测试结果: {'通过' if success else '失败'}")
    print("=" * 60)
    
    sys.exit(0 if success else 1) 