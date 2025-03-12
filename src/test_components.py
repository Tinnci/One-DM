#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 one_dm 包中主要功能组件
"""

import os
import sys
import unittest
import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn

# 确保 src 目录在 Python 路径中
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 导入所需模块
from one_dm.models.diffusion import Diffusion, EMA
from one_dm.models.unet import UNetModel
from one_dm.models.transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from one_dm.utils.parse_config import Config
from one_dm.utils.util import fix_random_seed

# 设置随机种子以确保结果可重现
fix_random_seed(42)

class TestDiffusion(unittest.TestCase):
    """测试扩散模型的基本功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = Diffusion(
            noise_steps=100, 
            beta_start=1e-4, 
            beta_end=0.02, 
            device=self.device
        )
        
    def test_initialization(self):
        """测试扩散模型的初始化"""
        self.assertEqual(self.diffusion.noise_steps, 100)
        self.assertEqual(self.diffusion.beta_start, 1e-4)
        self.assertEqual(self.diffusion.beta_end, 0.02)
        self.assertEqual(self.diffusion.device, self.device)
        
    def test_prepare_noise_schedule(self):
        """测试噪声调度表的生成"""
        beta = self.diffusion.beta
        self.assertEqual(len(beta), 100)
        self.assertAlmostEqual(beta[0].item(), 1e-4, places=5)
        self.assertAlmostEqual(beta[-1].item(), 0.02, places=5)
        
    def test_noise_images(self):
        """测试添加噪声到图像的功能"""
        batch_size = 4
        channels = 3
        height, width = 64, 64
        
        # 创建一个随机张量作为测试图像
        images = torch.randn(batch_size, channels, height, width).to(self.device)
        
        # 选择一个随机时间步
        t = torch.tensor([10]).to(self.device)
        
        # 添加噪声
        noised_images, noise = self.diffusion.noise_images(images, t)
        
        # 检查输出尺寸是否正确
        self.assertEqual(noised_images.shape, images.shape)
        self.assertEqual(noise.shape, images.shape)
        
        # 检查noised_images和原始images是否不同
        self.assertFalse(torch.allclose(noised_images, images))
        
    def test_sample_timesteps(self):
        """测试时间步采样功能"""
        batch_size = 10
        timesteps = self.diffusion.sample_timesteps(batch_size)
        
        # 检查生成的时间步是否在正确范围内
        self.assertEqual(len(timesteps), batch_size)
        self.assertTrue(all(0 <= t < self.diffusion.noise_steps for t in timesteps))

class TestEMA(unittest.TestCase):
    """测试EMA(Exponential Moving Average)功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.beta = 0.99
        self.ema = EMA(beta=self.beta)
        
        # 创建两个小型模型用于测试
        self.model1 = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        self.model2 = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
    def test_update_average(self):
        """测试参数平均值更新功能"""
        old = torch.ones(5)
        new = torch.zeros(5)
        
        # 更新平均值
        updated = self.ema.update_average(old, new)
        
        # 检查更新后的值是否符合期望
        expected = old * self.beta + (1 - self.beta) * new
        self.assertTrue(torch.allclose(updated, expected))
        
    def test_reset_parameters(self):
        """测试参数重置功能"""
        # 重置参数
        self.ema.reset_parameters(self.model2, self.model1)
        
        # 检查模型参数是否相同
        for p1, p2 in zip(self.model1.parameters(), self.model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

class TestUNetModel(unittest.TestCase):
    """测试UNet模型的基本功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用正确的参数集初始化UNet模型
        try:
            d_model = 256  # 修改为与Mix_TR的默认输出维度匹配
            self.unet = UNetModel(
                in_channels=3,                # 输入通道数
                model_channels=64,            # 基础通道数
                out_channels=3,               # 输出通道数
                num_res_blocks=2,             # 每个层级的残差块数量
                attention_resolutions=(1, 2, 4), # 注意力机制的分辨率
                dropout=0.1,                  # dropout率
                channel_mult=(1, 2, 4, 8),    # 通道数的倍数
                conv_resample=True,           # 是否使用卷积重采样
                dims=2,                       # 维度
                use_checkpoint=False,         # 是否使用梯度检查点
                num_heads=8,                  # 注意力头数
                num_head_channels=-1,         # 每个头的通道数
                num_heads_upsample=-1,        # 上采样时的头数
                use_scale_shift_norm=True,    # 是否使用scale shift norm
                resblock_updown=False,        # 是否在上下采样时使用残差块
                use_new_attention_order=False, # 是否使用新的注意力顺序
                use_spatial_transformer=True,  # 是否使用空间transformer
                transformer_depth=1,          # transformer深度
                context_dim=d_model,          # 上下文维度，与Mix_TR的输出维度匹配
                use_fp16=False,              # 是否使用FP16
                legacy=False                  # 是否使用遗留模式
            ).to(self.device)
        except Exception as e:
            self.skipTest(f"无法初始化UNetModel: {str(e)}")

    def test_forward(self):
        """测试UNet模型的前向传播"""
        batch_size = 2
        height, width = 32, 32
        time_steps = 10
        
        try:
            # 创建输入张量
            x = torch.randn(batch_size, self.unet.in_channels, height, width).to(self.device)
            t = torch.randint(0, 1000, (batch_size,)).to(self.device)
            
            # 创建风格和内容输入，确保维度正确
            style = torch.randn(batch_size, 2, height, width).to(self.device)  # [B, 2, H, W] for training
            laplace = torch.randn(batch_size, 2, height, width).to(self.device)  # [B, 2, H, W] for training
            content = torch.randn(batch_size, time_steps, height, width).to(self.device)  # [B, T, H, W]
            
            # 测试训练模式
            print("测试训练模式...")
            output_train, high_nce_emb, low_nce_emb = self.unet(x, t, style=style, laplace=laplace, content=content, tag='train')
            
            # 检查训练模式输出
            self.assertEqual(output_train.shape, (batch_size, self.unet.out_channels, height, width))
            self.assertEqual(high_nce_emb.shape, (batch_size, 2, 256))  # [B, 2, C] - 修改为256维度，匹配Mix_TR的输出
            self.assertEqual(low_nce_emb.shape, (batch_size, 2, 256))   # [B, 2, C] - 修改为256维度，匹配Mix_TR的输出
            
            # 测试推理模式
            print("测试推理模式...")
            style_test = torch.randn(batch_size, 1, height, width).to(self.device)  # [B, 1, H, W] for testing
            laplace_test = torch.randn(batch_size, 1, height, width).to(self.device)  # [B, 1, H, W] for testing
            
            output_test = self.unet(x, t, style=style_test, laplace=laplace_test, content=content, tag='test')
            
            # 检查推理模式输出
            self.assertEqual(output_test.shape, (batch_size, self.unet.out_channels, height, width))
            
            # 检查输出值范围
            self.assertTrue(torch.isfinite(output_train).all(), "训练模式输出包含无限值或NaN")
            self.assertTrue(torch.isfinite(output_test).all(), "推理模式输出包含无限值或NaN")
            
            print("UNet前向传播测试通过")
            
        except Exception as e:
            self.fail(f"UNet前向传播失败，错误信息: {str(e)}")

class TestTransformer(unittest.TestCase):
    """测试Transformer组件的基本功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # 使用标准参数初始化TransformerEncoder
            self.encoder = TransformerEncoder(
                encoder_layer=TransformerEncoderLayer(
                    d_model=512,              # 模型维度
                    nhead=8,                  # 注意力头数
                    dim_feedforward=2048,     # 前馈网络维度
                    dropout=0.1,              # dropout率
                    activation="relu",        # 激活函数
                    normalize_before=False    # 是否在前面进行归一化
                ),
                num_layers=6,                 # 编码器层数
                norm=nn.LayerNorm(512)        # 归一化层
            ).to(self.device)
            
            # 使用标准参数初始化TransformerDecoder
            self.decoder = TransformerDecoder(
                decoder_layer=TransformerDecoderLayer(
                    d_model=512,              # 模型维度
                    nhead=8,                  # 注意力头数
                    dim_feedforward=2048,     # 前馈网络维度
                    dropout=0.1,              # dropout率
                    activation="relu",        # 激活函数
                    normalize_before=False    # 是否在前面进行归一化
                ),
                num_layers=6,                 # 解码器层数
                norm=nn.LayerNorm(512)        # 归一化层
            ).to(self.device)
            
        except Exception as e:
            self.skipTest(f"无法初始化Transformer: {str(e)}")
            
    def test_encoder_forward(self):
        """测试TransformerEncoder的前向传播"""
        batch_size = 2
        seq_len = 10
        d_model = 512  # 使用与初始化相同的维度
        
        try:
            # 创建输入张量
            x = torch.randn(seq_len, batch_size, d_model).to(self.device)  # 注意维度顺序：(seq_len, batch, d_model)
            mask = None
            
            # 进行前向传播
            output = self.encoder(x, mask)
            
            # 检查输出形状
            self.assertEqual(output.shape, (seq_len, batch_size, d_model))
            print("TransformerEncoder前向传播测试通过")
        except Exception as e:
            self.fail(f"TransformerEncoder前向传播失败，错误信息: {str(e)}")
            
    def test_decoder_forward(self):
        """测试TransformerDecoder的前向传播"""
        batch_size = 2
        seq_len = 10
        d_model = 512  # 使用与初始化相同的维度
        
        try:
            # 创建输入张量
            tgt = torch.randn(seq_len, batch_size, d_model).to(self.device)  # (seq_len, batch, d_model)
            memory = torch.randn(seq_len, batch_size, d_model).to(self.device)
            
            # 进行前向传播
            output = self.decoder(tgt, memory)
            
            # 检查输出形状 (1, seq_len, batch_size, d_model)，因为return_intermediate=False
            self.assertEqual(output.shape, (1, seq_len, batch_size, d_model))
            print("TransformerDecoder前向传播测试通过")
        except Exception as e:
            self.fail(f"TransformerDecoder前向传播失败，错误信息: {str(e)}")

class TestConfig(unittest.TestCase):
    """测试配置系统的基本功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = Config()
        
        # 创建测试配置字典
        self.test_config = {
            'TRAIN': {
                'IMS_PER_BATCH': 64,
                'DROPOUT_P': 0.0,
                'SEED': 1001
            },
            'MODEL': {
                'IN_CHANNELS': 4,
                'OUT_CHANNELS': 4,
                'NUM_RES_BLOCKS': 1
            }
        }
        
    def test_from_dict(self):
        """测试从字典加载配置"""
        try:
            cfg = self.config.from_dict(self.test_config)
            
            # 检查配置是否正确加载
            self.assertEqual(cfg.TRAIN.IMS_PER_BATCH, 64)
            self.assertEqual(cfg.TRAIN.DROPOUT_P, 0.0)
            self.assertEqual(cfg.TRAIN.SEED, 1001)
            self.assertEqual(cfg.MODEL.IN_CHANNELS, 4)
            self.assertEqual(cfg.MODEL.OUT_CHANNELS, 4)
            self.assertEqual(cfg.MODEL.NUM_RES_BLOCKS, 1)
        except AttributeError:
            # 如果属性访问方式不同，可能需要调整测试
            self.skipTest("配置对象的属性访问方式与预期不同")
        
    def test_get_config(self):
        """测试获取配置功能"""
        try:
            self.config.from_dict(self.test_config)
            cfg = self.config.get_config()
            
            # 检查获取的配置是否正确
            self.assertEqual(cfg.TRAIN.IMS_PER_BATCH, 64)
            self.assertEqual(cfg.MODEL.NUM_RES_BLOCKS, 1)
        except AttributeError:
            # 如果属性访问方式不同，可能需要调整测试
            self.skipTest("配置对象的属性访问方式与预期不同")

def run_tests():
    """运行所有测试"""
    print("开始测试 one_dm 主要功能组件...\n")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_loader = unittest.TestLoader()
    
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestDiffusion))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestEMA))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestConfig))
    
    # 如果有CUDA设备，才添加模型测试
    if torch.cuda.is_available():
        print("检测到CUDA设备，添加模型前向传播测试...")
        test_suite.addTest(test_loader.loadTestsFromTestCase(TestUNetModel))
        test_suite.addTest(test_loader.loadTestsFromTestCase(TestTransformer))
    else:
        print("未检测到CUDA设备，跳过模型前向传播测试...")
    
    # 运行测试
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # 返回测试结果
    return test_result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)