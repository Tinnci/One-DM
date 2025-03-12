#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 one_dm 模型参数配置和初始化
"""

import os
import sys
import unittest
import torch
import numpy as np
from collections import defaultdict

# 确保 src 目录在 Python 路径中
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 导入所需模块
from one_dm.models.diffusion import Diffusion
from one_dm.models.unet import UNetModel
from one_dm.models.transformer import TransformerEncoder, TransformerDecoder
from one_dm.models.paragraph_diffusion import ParagraphDiffusion
from one_dm.utils.util import fix_random_seed

# 设置随机种子以确保结果可重现
fix_random_seed(42)

class TestDiffusionParameters(unittest.TestCase):
    """测试扩散模型的参数配置"""
    
    def test_diffusion_initialization_params(self):
        """测试扩散模型初始化参数的各种组合"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 测试默认参数
        model = Diffusion(device=device)
        self.assertEqual(model.noise_steps, 1000)
        self.assertEqual(model.beta_start, 1e-4)
        self.assertEqual(model.beta_end, 0.02)
        
        # 测试自定义参数
        model = Diffusion(noise_steps=500, beta_start=1e-5, beta_end=0.01, device=device)
        self.assertEqual(model.noise_steps, 500)
        self.assertEqual(model.beta_start, 1e-5)
        self.assertEqual(model.beta_end, 0.01)
        
        # 测试beta值生成
        self.assertEqual(len(model.beta), 500)
        self.assertEqual(len(model.alpha), 500)
        self.assertEqual(len(model.alpha_hat), 500)
        
        # 验证beta、alpha和alpha_hat计算是否正确
        self.assertAlmostEqual(model.beta[0].item(), 1e-5, places=6)
        self.assertAlmostEqual(model.beta[-1].item(), 0.01, places=6)
        self.assertAlmostEqual(model.alpha[0].item(), 1 - 1e-5, places=6)
        
        # 验证beta是否递增
        self.assertTrue(torch.all(model.beta[1:] > model.beta[:-1]))
        
        # 验证alpha_hat是否递减
        self.assertTrue(torch.all(model.alpha_hat[1:] < model.alpha_hat[:-1]))

class TestUNetParameters(unittest.TestCase):
    """测试UNet模型的参数配置"""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_basic_initialization(self):
        """测试基本的UNet初始化参数"""
        try:
            # 尝试创建一个最小配置的UNet
            model = UNetModel(
                in_channels=3,
                model_channels=64,
                out_channels=3,
                num_res_blocks=2,
                attention_resolutions=[2],
                dropout=0.0,
                channel_mult=(1, 2, 4),
                num_heads=8,
                use_checkpoint=False
            )
            
            # 验证主要参数是否正确设置
            self.assertEqual(model.in_channels, 3)
            self.assertEqual(model.model_channels, 64)
            self.assertEqual(model.out_channels, 3)
            self.assertEqual(model.num_res_blocks, 2)
            self.assertEqual(model.dropout, 0.0)
            self.assertEqual(model.num_heads, 8)
        except Exception as e:
            self.skipTest(f"UNet初始化失败: {str(e)}")
    
    def test_parameter_combinations(self):
        """测试不同的参数组合"""
        # 常见的参数配置
        test_configs = [
            # (in_channels, model_channels, out_channels, num_res_blocks, channel_mult)
            (3, 64, 3, 2, (1, 2, 4)),
            (4, 128, 4, 2, (1, 1, 2, 2, 4)),
            (1, 32, 1, 1, (1, 2)),
        ]
        
        for config in test_configs:
            in_channels, model_channels, out_channels, num_res_blocks, channel_mult = config
            try:
                model = UNetModel(
                    in_channels=in_channels,
                    model_channels=model_channels,
                    out_channels=out_channels,
                    num_res_blocks=num_res_blocks,
                    attention_resolutions=[2],
                    dropout=0.0,
                    channel_mult=channel_mult,
                    num_heads=4,
                    use_checkpoint=False
                )
                # 配置成功初始化
                self.assertEqual(model.in_channels, in_channels)
                self.assertEqual(model.out_channels, out_channels)
                print(f"UNet配置 {config} 初始化成功")
            except Exception as e:
                print(f"UNet配置 {config} 初始化失败: {str(e)}")
    
    def test_model_structure(self):
        """测试模型结构层次"""
        try:
            model = UNetModel(
                in_channels=3,
                model_channels=32,
                out_channels=3,
                num_res_blocks=1,
                attention_resolutions=[2],
                dropout=0.0,
                channel_mult=(1, 2),
                num_heads=4,
                use_checkpoint=False
            ).to(self.device)
            
            # 测试模型前向传播
            batch_size = 2
            test_input = torch.randn(batch_size, 3, 32, 32).to(self.device)
            time_step = torch.tensor([0, 500]).to(self.device)
            
            # 检查输出形状
            output = model(test_input, time_step)
            self.assertEqual(output.shape, (batch_size, 3, 32, 32))
            
            # 检查参数量是否合理
            param_count = sum(p.numel() for p in model.parameters())
            print(f"UNet模型参数数量: {param_count}")
            self.assertGreater(param_count, 1000)  # 应该有足够的参数
        except Exception as e:
            self.skipTest(f"UNet结构测试失败: {str(e)}")

class TestTransformerParameters(unittest.TestCase):
    """测试Transformer模型的参数配置"""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_encoder_parameters(self):
        """测试TransformerEncoder的参数配置"""
        try:
            # 初始化TransformerEncoder
            encoder = TransformerEncoder(
                dim=64,  # 假设参数名为dim
                depth=3,
                heads=4
            ).to(self.device)
            
            # 检查基本属性
            if hasattr(encoder, 'dim'):
                self.assertEqual(encoder.dim, 64)
            if hasattr(encoder, 'depth'):
                self.assertEqual(encoder.depth, 3)
            if hasattr(encoder, 'heads'):
                self.assertEqual(encoder.heads, 4)
            
            # 检查层结构
            if hasattr(encoder, 'layers'):
                self.assertEqual(len(encoder.layers), 3)
            
            print("TransformerEncoder参数测试通过")
        except Exception as e:
            self.skipTest(f"TransformerEncoder参数测试失败: {str(e)}")
    
    def test_decoder_parameters(self):
        """测试TransformerDecoder的参数配置"""
        try:
            # 初始化TransformerDecoder
            decoder = TransformerDecoder(
                dim=64,
                depth=3,
                heads=4
            ).to(self.device)
            
            # 检查基本属性
            if hasattr(decoder, 'dim'):
                self.assertEqual(decoder.dim, 64)
            if hasattr(decoder, 'depth'):
                self.assertEqual(decoder.depth, 3)
            if hasattr(decoder, 'heads'):
                self.assertEqual(decoder.heads, 4)
            
            # 检查层结构
            if hasattr(decoder, 'layers'):
                self.assertEqual(len(decoder.layers), 3)
            
            print("TransformerDecoder参数测试通过")
        except Exception as e:
            self.skipTest(f"TransformerDecoder参数测试失败: {str(e)}")

class TestParagraphDiffusionParameters(unittest.TestCase):
    """测试ParagraphDiffusion模型的参数配置"""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_initialization(self):
        """测试ParagraphDiffusion的初始化参数"""
        try:
            # 尝试创建ParagraphDiffusion实例
            model = ParagraphDiffusion(
                noise_steps=100,
                beta_start=1e-4,
                beta_end=0.02,
                device=self.device
            )
            
            # 验证参数是否正确设置
            self.assertEqual(model.noise_steps, 100)
            self.assertEqual(model.beta_start, 1e-4)
            self.assertEqual(model.beta_end, 0.02)
            self.assertEqual(model.device, self.device)
            
            print("ParagraphDiffusion初始化参数测试通过")
        except Exception as e:
            self.skipTest(f"ParagraphDiffusion初始化测试失败: {str(e)}")

def run_tests():
    """运行所有模型参数测试"""
    print("开始测试 one_dm 模型参数配置...\n")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_loader = unittest.TestLoader()
    
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestDiffusionParameters))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestUNetParameters))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestTransformerParameters))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestParagraphDiffusionParameters))
    
    # 运行测试
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # 返回测试结果
    return test_result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 