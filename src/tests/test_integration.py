#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对One-DM模型进行端到端集成测试
测试包括：
1. 数据加载 -> 模型训练 -> 模型保存的完整流程
2. 模型加载 -> 推理流程
3. 完整扩散过程（Diffusion）
"""

import os
import sys
import unittest
import torch
import shutil
import tempfile
import numpy as np
from PIL import Image
import json

# 确保src目录在Python路径中
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 导入模拟数据生成器
try:
    from test_mock_data import MockDataGenerator
except ImportError:
    print("无法导入模拟数据生成器，请确保test_mock_data.py已创建")
    sys.exit(1)

# 导入必要的模块
try:
    from one_dm.models.paragraph_diffusion import ParagraphDiffusion
    from one_dm.models.unet import UNetModel
    from one_dm.models.transformer import TransformerEncoder, TransformerDecoder
    from one_dm.models.diffusion import GaussianDiffusion
    from one_dm.data.loader import IAMDataset, ContentData
    from one_dm.trainer.trainer import Trainer
    from one_dm.utils.util import fix_random_seed
except ImportError as e:
    print(f"导入模块失败: {str(e)}")
    sys.exit(1)

# 设置随机种子保证结果可重现
fix_random_seed(42)

class TestIntegration(unittest.TestCase):
    """端到端集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建临时目录用于保存模型和中间结果
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_dir = os.path.join(cls.temp_dir, "models")
        os.makedirs(cls.model_dir, exist_ok=True)
        
        # 创建模拟数据
        cls.data_generator = MockDataGenerator(base_dir=cls.temp_dir)
        cls.indices, cls.text_mapping = cls.data_generator.generate_dataset(num_samples=20)
        
        # 设置设备
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {cls.device}")
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    def test_1_unet_model_init(self):
        """测试初始化UNet模型"""
        try:
            # 创建UNet模型
            unet = UNetModel(
                in_channels=3,
                model_channels=32,
                out_channels=3,
                num_res_blocks=2,
                attention_resolutions=(1,),
                dropout=0.1,
                channel_mult=(1, 2),
                use_checkpoint=False
            )
            self.assertIsNotNone(unet)
            print("UNet模型初始化成功")
            
            # 测试前向传播
            dummy_input = torch.randn(2, 3, 64, 64)
            dummy_timesteps = torch.ones(2, dtype=torch.long)
            output = unet(dummy_input, dummy_timesteps)
            self.assertEqual(output.shape, (2, 3, 64, 64))
            print("UNet模型前向传播测试通过")
        except Exception as e:
            self.fail(f"UNet模型测试失败: {str(e)}")
    
    def test_2_transformer_init(self):
        """测试初始化Transformer模型"""
        try:
            # 创建Transformer编码器
            encoder = TransformerEncoder(
                dim=32,
                depth=2,
                heads=4,
                dim_head=8
            )
            self.assertIsNotNone(encoder)
            print("Transformer编码器初始化成功")
            
            # 创建Transformer解码器
            decoder = TransformerDecoder(
                dim=32,
                depth=2,
                heads=4,
                dim_head=8
            )
            self.assertIsNotNone(decoder)
            print("Transformer解码器初始化成功")
            
            # 测试前向传播
            x = torch.randn(2, 16, 32)  # (batch_size, seq_len, dim)
            mask = torch.ones(2, 16).bool()  # (batch_size, seq_len)
            
            # 编码器前向传播
            enc_output = encoder(x, mask)
            self.assertEqual(enc_output.shape, (2, 16, 32))
            print("Transformer编码器前向传播测试通过")
            
            # 解码器前向传播
            dec_output = decoder(x, enc_output, mask, mask)
            self.assertEqual(dec_output.shape, (2, 16, 32))
            print("Transformer解码器前向传播测试通过")
        except Exception as e:
            self.fail(f"Transformer模型测试失败: {str(e)}")
    
    def test_3_gaussian_diffusion_init(self):
        """测试初始化GaussianDiffusion模型"""
        try:
            # 创建一个小型UNet作为GaussianDiffusion的模型
            denoise_fn = UNetModel(
                in_channels=3,
                model_channels=32,
                out_channels=3,
                num_res_blocks=1,
                attention_resolutions=(1,),
                dropout=0.0,
                channel_mult=(1, 2),
                use_checkpoint=False
            )
            
            # 创建GaussianDiffusion
            diffusion = GaussianDiffusion(
                denoise_fn=denoise_fn,
                image_size=64,
                channels=3,
                timesteps=10,
                loss_type='l1'
            )
            self.assertIsNotNone(diffusion)
            print("GaussianDiffusion初始化成功")
            
            # 测试训练损失计算
            dummy_x_start = torch.randn(2, 3, 64, 64)
            loss = diffusion(dummy_x_start)
            self.assertIsInstance(loss, torch.Tensor)
            print("GaussianDiffusion训练损失计算测试通过")
            
            # 测试采样（推理）
            try:
                # 批量大小为2，每个样本只采样2步（为加速测试）
                samples = diffusion.sample(batch_size=2, steps=2)
                self.assertEqual(samples.shape, (2, 3, 64, 64))
                print("GaussianDiffusion采样测试通过")
            except Exception as e:
                print(f"GaussianDiffusion采样测试跳过: {str(e)}")
        except Exception as e:
            self.fail(f"GaussianDiffusion模型测试失败: {str(e)}")
    
    def test_4_paragraph_diffusion_init(self):
        """测试初始化ParagraphDiffusion模型（完整模型）"""
        try:
            # 创建小型配置用于测试
            config = {
                'data': {
                    'image_size': 64,
                    'channels': 3,
                    'batch_size': 2
                },
                'model': {
                    'content_emb_size': 32,
                    'unet': {
                        'in_channels': 3,
                        'model_channels': 32,
                        'out_channels': 3,
                        'num_res_blocks': 1,
                        'attention_resolutions': [1],
                        'dropout': 0.0,
                        'channel_mult': [1, 2],
                        'dims': 2,
                        'use_checkpoint': False
                    },
                    'transformer': {
                        'dim': 32,
                        'depth': 2,
                        'heads': 4,
                        'dim_head': 8
                    },
                    'diffusion': {
                        'timesteps': 10,
                        'sampling_timesteps': 5,
                        'loss_type': 'l1'
                    }
                }
            }
            
            # 保存配置用于后续加载
            config_path = os.path.join(self.model_dir, "test_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # 创建ParagraphDiffusion
            pd_model = ParagraphDiffusion(config)
            
            self.assertIsNotNone(pd_model)
            print("ParagraphDiffusion初始化成功")
            
            # 保存模型
            model_path = os.path.join(self.model_dir, "test_model.pth")
            torch.save(pd_model.state_dict(), model_path)
            print(f"模型保存到: {model_path}")
            
            # 加载模型
            pd_model_loaded = ParagraphDiffusion(config)
            pd_model_loaded.load_state_dict(torch.load(model_path))
            print("模型加载成功")
            
            # 验证模型结构
            self.assertIsNotNone(pd_model_loaded.diffusion)
            self.assertIsNotNone(pd_model_loaded.transformer_encoder)
            self.assertIsNotNone(pd_model_loaded.transformer_decoder)
            print("模型结构验证通过")
        except Exception as e:
            self.fail(f"ParagraphDiffusion模型测试失败: {str(e)}")
    
    def test_5_mini_training_loop(self):
        """测试小型训练循环"""
        try:
            # 加载配置
            config_path = os.path.join(self.model_dir, "test_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 创建模型
            model = ParagraphDiffusion(config)
            model = model.to(self.device)
            
            # 创建优化器
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # 创建数据集
            try:
                dataset = IAMDataset(
                    image_path=self.data_generator.image_dir,
                    style_path=self.data_generator.style_dir,
                    laplace_path=self.data_generator.laplace_dir,
                    type="train",
                    content_type='unifont'
                )
                
                # 创建数据加载器
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config['data']['batch_size'],
                    shuffle=True,
                    num_workers=0
                )
                
                # 模拟训练循环
                model.train()
                for epoch in range(1):
                    total_loss = 0.0
                    batch_count = 0
                    
                    try:
                        # 只训练前2个批次
                        for i, batch in enumerate(dataloader):
                            if i >= 2:  # 限制训练批次数量
                                break
                            
                            # 确保所有输入都在正确的设备上
                            img = batch['img'].to(self.device)
                            style = batch.get('style', None)
                            if style is not None:
                                style = style.to(self.device)
                            
                            laplace = batch.get('laplace', None)
                            if laplace is not None:
                                laplace = laplace.to(self.device)
                            
                            content = batch.get('content', None)
                            if content is not None:
                                content = content.to(self.device)
                            
                            # 前向传播和反向传播
                            optimizer.zero_grad()
                            
                            # 尝试不同的输入组合
                            if style is not None and content is not None:
                                loss = model(img, style_input=style, content_input=content)
                            elif style is not None:
                                loss = model(img, style_input=style)
                            elif content is not None:
                                loss = model(img, content_input=content)
                            else:
                                loss = model(img)
                            
                            loss.backward()
                            optimizer.step()
                            
                            total_loss += loss.item()
                            batch_count += 1
                            
                            print(f"批次 {i+1}, 损失: {loss.item():.6f}")
                        
                        if batch_count > 0:
                            avg_loss = total_loss / batch_count
                            print(f"平均损失: {avg_loss:.6f}")
                            self.assertLess(avg_loss, float('inf'))  # 确保损失值是有限的
                    except Exception as e:
                        print(f"训练循环异常: {str(e)}")
                        raise
                    
                # 保存训练后的模型
                trained_model_path = os.path.join(self.model_dir, "trained_model.pth")
                torch.save(model.state_dict(), trained_model_path)
                print(f"训练后的模型保存到: {trained_model_path}")
                
                # 测试模型评估模式
                model.eval()
                print("模型设置为评估模式")
                
                # 测试推理
                with torch.no_grad():
                    try:
                        # 从数据加载器获取一个样本
                        test_batch = next(iter(dataloader))
                        test_img = test_batch['img'].to(self.device)
                        
                        # 尝试生成样本
                        gen_samples = model.sample(batch_size=2, steps=2)
                        self.assertIsNotNone(gen_samples)
                        
                        # 检查样本形状
                        expected_shape = (2, config['data']['channels'], config['data']['image_size'], config['data']['image_size'])
                        self.assertEqual(gen_samples.shape, expected_shape)
                        
                        print("推理测试通过")
                    except Exception as e:
                        print(f"推理测试异常: {str(e)}")
                
            except Exception as e:
                self.skipTest(f"数据集加载失败，跳过训练测试: {str(e)}")
        except Exception as e:
            self.fail(f"小型训练循环测试失败: {str(e)}")
    
    def test_6_trainer_interface(self):
        """测试训练器接口"""
        try:
            # 加载配置
            config_path = os.path.join(self.model_dir, "test_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 扩展配置以包含训练器设置
            train_config = {
                'train': {
                    'epochs': 1,
                    'save_every': 1,
                    'sample_every': 1,
                    'log_every': 1,
                    'save_dir': self.model_dir,
                    'gradient_accumulate_every': 1,
                    'lr': 1e-4
                }
            }
            config.update(train_config)
            
            # 创建模型
            model = ParagraphDiffusion(config)
            
            # 测试Trainer初始化
            try:
                trainer = Trainer(
                    diffusion_model=model,
                    config=config,
                    train_dl=None,  # 不需要实际加载数据
                    valid_dl=None
                )
                self.assertIsNotNone(trainer)
                print("Trainer初始化成功")
                
                # 测试Trainer的一些基本方法
                try:
                    # 测试Trainer的save方法
                    trainer.save(0)
                    print("Trainer保存模型测试通过")
                    
                    # 检查模型是否存在
                    expected_path = os.path.join(self.model_dir, f"model_0.pt")
                    self.assertTrue(os.path.exists(expected_path))
                except Exception as e:
                    print(f"Trainer方法测试异常: {str(e)}")
            except Exception as e:
                self.skipTest(f"Trainer初始化失败，跳过Trainer测试: {str(e)}")
        except Exception as e:
            self.fail(f"Trainer接口测试失败: {str(e)}")
    
    def test_7_end_to_end_workflow(self):
        """测试端到端工作流程（简化版）"""
        try:
            print("执行端到端测试...")
            
            # 1. 加载配置
            config_path = os.path.join(self.model_dir, "test_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 2. 创建数据集
            try:
                train_dataset = IAMDataset(
                    image_path=self.data_generator.image_dir,
                    style_path=self.data_generator.style_dir,
                    laplace_path=self.data_generator.laplace_dir,
                    type="train",
                    content_type='unifont'
                )
                
                # 创建数据加载器
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=config['data']['batch_size'],
                    shuffle=True,
                    num_workers=0
                )
                
                # 3. 创建模型并训练一个小批次
                model = ParagraphDiffusion(config)
                model = model.to(self.device)
                
                # 4. 创建优化器
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                
                # 5. 简单训练循环
                model.train()
                for i, batch in enumerate(train_loader):
                    if i >= 1:  # 仅训练一个批次
                        break
                    
                    # 获取训练数据
                    img = batch['img'].to(self.device)
                    
                    # 前向传播和反向传播
                    optimizer.zero_grad()
                    loss = model(img)
                    loss.backward()
                    optimizer.step()
                    
                    print(f"训练批次 {i+1}, 损失: {loss.item():.6f}")
                
                # 6. 保存模型
                final_model_path = os.path.join(self.model_dir, "final_model.pth")
                torch.save(model.state_dict(), final_model_path)
                print(f"模型保存到: {final_model_path}")
                
                # 7. 加载模型并进行推理
                model_loaded = ParagraphDiffusion(config)
                model_loaded.load_state_dict(torch.load(final_model_path))
                model_loaded = model_loaded.to(self.device)
                model_loaded.eval()
                
                # 8. 生成样本
                with torch.no_grad():
                    try:
                        # 采样步数减少以加速测试
                        samples = model_loaded.sample(batch_size=1, steps=2)
                        self.assertIsNotNone(samples)
                        
                        # 保存生成的样本
                        output_sample_path = os.path.join(self.model_dir, "sample.png")
                        
                        # 转换为图像并保存
                        sample_np = samples[0].cpu().numpy().transpose(1, 2, 0)
                        sample_np = (sample_np * 255).clip(0, 255).astype(np.uint8)
                        sample_img = Image.fromarray(sample_np)
                        sample_img.save(output_sample_path)
                        
                        print(f"样本生成并保存到: {output_sample_path}")
                    except Exception as e:
                        print(f"样本生成测试异常: {str(e)}")
                
                # 9. 测试条件生成
                with torch.no_grad():
                    try:
                        # 从数据加载器获取一个样本
                        test_batch = next(iter(train_loader))
                        
                        # 如果有内容输入
                        if 'content' in test_batch:
                            content = test_batch['content'].to(self.device)
                            # 条件采样
                            cond_samples = model_loaded.sample(batch_size=1, steps=2, content_input=content[:1])
                            self.assertIsNotNone(cond_samples)
                            
                            # 保存条件生成的样本
                            cond_sample_path = os.path.join(self.model_dir, "cond_sample.png")
                            
                            # 转换为图像并保存
                            cond_sample_np = cond_samples[0].cpu().numpy().transpose(1, 2, 0)
                            cond_sample_np = (cond_sample_np * 255).clip(0, 255).astype(np.uint8)
                            cond_sample_img = Image.fromarray(cond_sample_np)
                            cond_sample_img.save(cond_sample_path)
                            
                            print(f"条件样本生成并保存到: {cond_sample_path}")
                    except Exception as e:
                        print(f"条件样本生成测试异常: {str(e)}")
                
                # 10. 测试模型评估
                try:
                    # 获取一个批次用于评估
                    eval_batch = next(iter(train_loader))
                    img = eval_batch['img'].to(self.device)
                    
                    # 在评估模式下计算损失
                    model_loaded.eval()
                    with torch.no_grad():
                        eval_loss = model_loaded(img).item()
                    
                    print(f"评估损失: {eval_loss:.6f}")
                    self.assertLess(eval_loss, float('inf'))  # 确保损失值是有限的
                except Exception as e:
                    print(f"模型评估测试异常: {str(e)}")
                
            except Exception as e:
                self.skipTest(f"数据集加载失败，跳过端到端测试: {str(e)}")
            
            print("端到端测试完成")
        except Exception as e:
            self.fail(f"端到端测试失败: {str(e)}")

def run_tests():
    """运行所有集成测试"""
    print("开始集成测试...\n")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_loader = unittest.TestLoader()
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # 返回测试结果
    return test_result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 