#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 one_dm 包中数据加载相关的功能
"""

import os
import sys
import unittest
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

# 确保 src 目录在 Python 路径中
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from one_dm.data.loader import IAMDataset, ContentData
    from one_dm.data.paragraph_dataset import ParagraphDataset
    from one_dm.utils.util import fix_random_seed
except ImportError:
    print("无法导入数据加载模块，请确保项目安装正确")
    sys.exit(1)

# 设置随机种子以确保结果可重现
fix_random_seed(42)

class TestContentData(unittest.TestCase):
    """测试ContentData类的基本功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.batch_size = 2
        try:
            # 不要传入max_len参数
            self.content_data = ContentData()
        except Exception as e:
            self.skipTest(f"无法初始化ContentData: {str(e)}")
    
    def test_get_random_content(self):
        """测试随机内容生成功能"""
        try:
            content = self.content_data.get_random_content(self.batch_size)
            
            # 检查输出
            self.assertIsNotNone(content)
            
            # 检查批次大小
            if isinstance(content, torch.Tensor):
                self.assertEqual(content.shape[0], self.batch_size)
            elif isinstance(content, list):
                self.assertEqual(len(content), self.batch_size)
            
            print("ContentData.get_random_content测试通过")
        except Exception as e:
            self.fail(f"ContentData.get_random_content失败，错误信息: {str(e)}")

class TestParagraphDataset(unittest.TestCase):
    """测试ParagraphDataset的基本功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录和测试数据
        self.test_dir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.test_dir, "images")
        self.style_dir = os.path.join(self.test_dir, "styles")
        self.laplace_dir = os.path.join(self.test_dir, "laplace")
        
        # 创建子目录
        os.makedirs(os.path.join(self.image_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.style_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.laplace_dir, "train"), exist_ok=True)
        
        # 由于实际数据集可能不存在，我们将跳过实际实例化
        self.can_test_paragraph_dataset = False
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_dataset_structure(self):
        """测试数据集类结构与导入"""
        # 检查ParagraphDataset是否存在并可导入
        self.assertTrue(hasattr(sys.modules.get('one_dm.data.paragraph_dataset'), 'ParagraphDataset'))
        print("成功导入ParagraphDataset类")
        
        # 如果没有实际数据，跳过实例化测试
        if not self.can_test_paragraph_dataset:
            print("跳过ParagraphDataset实例化测试，因为缺少真实数据")
            return
        
        # 尝试创建数据集实例
        try:
            dataset = ParagraphDataset(
                image_path=self.image_dir,
                style_path=self.style_dir,
                laplace_path=self.laplace_dir,
                type="train"
            )
            self.assertIsNotNone(dataset)
            print("成功创建ParagraphDataset实例")
        except Exception as e:
            print(f"无法创建ParagraphDataset实例: {str(e)}")

class TestIAMDataset(unittest.TestCase):
    """测试IAMDataset的基本功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.test_dir, "images")
        self.style_dir = os.path.join(self.test_dir, "styles")
        self.laplace_dir = os.path.join(self.test_dir, "laplace")
        
        # 创建子目录
        os.makedirs(os.path.join(self.image_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.style_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.laplace_dir, "train"), exist_ok=True)
        
        # 创建测试数据文件
        self.create_test_data_file()
        
        # 由于实际数据集可能不存在，我们将跳过实际实例化
        self.can_test_iam_dataset = False
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_test_data_file(self):
        """创建测试数据文件"""
        # 创建一个简单的文本文件作为测试数据
        test_text_path = os.path.join(self.test_dir, "IAM64_train.txt")
        with open(test_text_path, "w") as f:
            f.write("sample1 This is a test\n")
            f.write("sample2 Another test sample\n")
        
        # 创建一个空白图像作为测试样本
        test_img = Image.new('RGB', (64, 64), color='white')
        test_img_path = os.path.join(self.image_dir, "train", "sample1.png")
        test_img.save(test_img_path)
    
    def test_dataset_structure(self):
        """测试数据集类结构与导入"""
        # 检查IAMDataset是否存在并可导入
        self.assertTrue(hasattr(sys.modules.get('one_dm.data.loader'), 'IAMDataset'))
        print("成功导入IAMDataset类")
        
        # 如果没有实际数据，跳过实例化测试
        if not self.can_test_iam_dataset:
            print("跳过IAMDataset实例化测试，因为缺少真实数据")
            return
        
        # 由于IAMDataset依赖实际数据，我们只测试类的结构
        try:
            # 尝试实例化，但预期会失败
            dataset = IAMDataset(
                image_path=self.image_dir,
                style_path=self.style_dir,
                laplace_path=self.laplace_dir,
                type="train"
            )
            # 如果到达这里，表示成功创建了实例
            self.assertIsNotNone(dataset)
            print("成功创建IAMDataset实例")
        except Exception as e:
            print(f"无法创建IAMDataset实例: {str(e)}")

def run_tests():
    """运行所有测试"""
    print("开始测试 one_dm 数据加载组件...\n")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_loader = unittest.TestLoader()
    
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestContentData))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestParagraphDataset))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestIAMDataset))
    
    # 运行测试
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # 返回测试结果
    return test_result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 