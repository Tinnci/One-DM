#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行 one_dm 项目的所有测试
"""

import os
import sys
import importlib
import unittest

def run_all_tests():
    """运行所有测试文件，返回测试是否全部通过"""
    print("=" * 60)
    print("开始运行 one_dm 项目的所有测试")
    print("=" * 60)
    
    # 收集所有测试用例
    test_suite = unittest.TestSuite()
    
    # 导入各个测试模块
    test_modules = [
        'tests.test_import',
        'tests.test_components',
        'tests.test_data_loader'
    ]
    
    # 跟踪每个测试模块的结果
    results = {}
    
    for module_name in test_modules:
        try:
            # 尝试导入测试模块
            module = importlib.import_module(module_name)
            
            # 检查模块是否有run_tests函数
            if hasattr(module, 'run_tests'):
                print(f"\n{'-' * 60}")
                print(f"运行 {module_name} 测试...")
                print(f"{'-' * 60}")
                result = module.run_tests()
                results[module_name] = result
            else:
                print(f"警告: {module_name} 模块中未找到 run_tests 函数，跳过")
                results[module_name] = False
        except ImportError as e:
            print(f"错误: 无法导入测试模块 {module_name}: {str(e)}")
            results[module_name] = False
        except Exception as e:
            print(f"错误: 运行 {module_name} 测试时出现异常: {str(e)}")
            results[module_name] = False
    
    # 汇总测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("-" * 60)
    
    all_passed = True
    for module_name, result in results.items():
        status = "通过" if result else "失败"
        all_passed = all_passed and result
        print(f"{module_name}: {status}")
    
    print("-" * 60)
    print(f"总体结果: {'全部通过' if all_passed else '部分失败'}")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    # 确保项目根目录在 Python 路径中
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    success = run_all_tests()
    sys.exit(0 if success else 1) 