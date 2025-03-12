#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对One-DM模型进行性能测试
测试包括：
1. 模型推理速度
2. 内存使用情况
3. GPU利用率（如果可用）
4. 批处理性能
5. 模型参数量统计
"""

import os
import sys
import unittest
import torch
import numpy as np
import time
import psutil
import json
from collections import defaultdict
from contextlib import contextmanager

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
    from one_dm.models.diffusion import Diffusion
    from one_dm.data.loader import IAMDataset, ContentData
    from one_dm.utils.util import fix_random_seed
except ImportError as e:
    print(f"导入模块失败: {str(e)}")
    sys.exit(1)

# 设置随机种子保证结果可重现
fix_random_seed(42)

@contextmanager
def timer(name=""):
    """计时器上下文管理器"""
    start = time.time()
    yield
    end = time.time()
    print(f"{name} 耗时: {end - start:.4f} 秒")

def get_gpu_memory_usage():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # 转换为MB
    return 0

def get_cpu_memory_usage():
    """获取CPU内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # 转换为MB

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())

def profile_forward_pass(model, input_data, num_runs=10):
    """测量前向传播的性能"""
    times = []
    memory_usage = []
    
    # 预热
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_data)
    
    # 正式测量
    for _ in range(num_runs):
        start_mem = get_gpu_memory_usage() if torch.cuda.is_available() else get_cpu_memory_usage()
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(input_data)
        
        end_time = time.time()
        end_mem = get_gpu_memory_usage() if torch.cuda.is_available() else get_cpu_memory_usage()
        
        times.append(end_time - start_time)
        memory_usage.append(end_mem - start_mem)
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'avg_memory': np.mean(memory_usage),
        'std_memory': np.std(memory_usage)
    }

class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {cls.device}")
        
        # 创建基本配置
        cls.config = {
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
                    'use_checkpoint': False,
                    'num_heads': 4,
                    'use_spatial_transformer': True,
                    'transformer_depth': 1,
                    'context_dim': 32
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
                    'loss_type': 'l1',
                    'noise_offset': 0,
                    'beta_start': 1e-4,
                    'beta_end': 0.02
                }
            }
        }
    
    def test_1_model_size(self):
        """测试模型大小和参数量"""
        try:
            # 从配置中提取diffusion参数
            diffusion_config = self.config['model']['diffusion']
            noise_steps = diffusion_config.get('timesteps', 1000)
            noise_offset = diffusion_config.get('noise_offset', 0)
            beta_start = diffusion_config.get('beta_start', 1e-4)
            beta_end = diffusion_config.get('beta_end', 0.02)
            
            # 创建模型
            model = ParagraphDiffusion(
                noise_steps=noise_steps,
                noise_offset=noise_offset,
                beta_start=beta_start,
                beta_end=beta_end,
                device=self.device
            ).to(self.device)
            
            # 统计总参数量
            total_params = count_parameters(model)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\n模型参数统计:")
            print(f"总参数量: {total_params:,}")
            print(f"可训练参数量: {trainable_params:,}")
            
            # 分组统计参数量
            param_groups = defaultdict(int)
            for name, param in model.named_parameters():
                group = name.split('.')[0]
                param_groups[group] += param.numel()
            
            print("\n各组件参数量:")
            for group, count in param_groups.items():
                print(f"{group}: {count:,} ({count/total_params*100:.2f}%)")
            
            # 估算模型大小（以MB为单位）
            model_size_mb = total_params * 4 / (1024 * 1024)  # 假设每个参数占用4字节
            print(f"\n估计模型大小: {model_size_mb:.2f} MB")
            
            # 验证参数量是否在合理范围内
            self.assertGreater(total_params, 1000)  # 确保模型不是太小
            self.assertEqual(total_params, trainable_params)  # 确保所有参数都是可训练的
        except Exception as e:
            self.fail(f"模型大小测试失败: {str(e)}")
    
    def test_2_inference_speed(self):
        """测试推理速度"""
        try:
            # 从配置中提取diffusion参数
            diffusion_config = self.config['model']['diffusion']
            noise_steps = diffusion_config.get('timesteps', 1000)
            noise_offset = diffusion_config.get('noise_offset', 0)
            beta_start = diffusion_config.get('beta_start', 1e-4)
            beta_end = diffusion_config.get('beta_end', 0.02)
            
            # 创建模型
            model = ParagraphDiffusion(
                noise_steps=noise_steps,
                noise_offset=noise_offset,
                beta_start=beta_start,
                beta_end=beta_end,
                device=self.device
            ).to(self.device)
            model.eval()
            
            # 准备不同大小的输入批次
            batch_sizes = [1, 2, 4]
            results = {}
            
            for batch_size in batch_sizes:
                print(f"\n测试批次大小 {batch_size}:")
                
                # 创建输入数据
                dummy_input = torch.randn(batch_size, 3, 64, 64).to(self.device)
                style = torch.randn(batch_size, 1, 64, 64).to(self.device)  # 添加style参数
                laplace = torch.randn(batch_size, 1, 64, 64).to(self.device)  # 添加laplace参数
                content = torch.randn(batch_size, 1, 64, 64).to(self.device)  # 添加content参数，注意这里的1是时间步
                
                # 测量前向传播性能
                with timer(f"批次大小 {batch_size} 的推理"):
                    perf_metrics = profile_forward_pass(model, (dummy_input, style, laplace, content))
                
                results[batch_size] = perf_metrics
                print(f"平均推理时间: {perf_metrics['avg_time']*1000:.2f} ms")
                print(f"时间标准差: {perf_metrics['std_time']*1000:.2f} ms")
                print(f"平均内存使用: {perf_metrics['avg_memory']:.2f} MB")
                print(f"内存使用标准差: {perf_metrics['std_memory']:.2f} MB")
            
            # 计算吞吐量（每秒处理的样本数）
            for batch_size, metrics in results.items():
                throughput = batch_size / metrics['avg_time']
                print(f"\n批次大小 {batch_size} 的吞吐量: {throughput:.2f} 样本/秒")
            
            # 验证性能指标
            self.assertLess(results[1]['avg_time'], 1.0)  # 单样本推理应该小于1秒
            
            # 验证批处理效率
            efficiency = results[4]['avg_time'] / (4 * results[1]['avg_time'])
            print(f"\n批处理效率: {efficiency:.2f}")
            self.assertLess(efficiency, 1.2)  # 批处理应该有一定的效率提升
        except Exception as e:
            self.fail(f"推理速度测试失败: {str(e)}")
    
    def test_3_memory_profile(self):
        """测试内存使用情况"""
        try:
            print("\n开始内存分析...")
            
            # 记录初始内存使用
            initial_memory = get_cpu_memory_usage()
            initial_gpu_memory = get_gpu_memory_usage() if torch.cuda.is_available() else 0
            
            print(f"初始CPU内存使用: {initial_memory:.2f} MB")
            if torch.cuda.is_available():
                print(f"初始GPU内存使用: {initial_gpu_memory:.2f} MB")
            
            # 从配置中提取diffusion参数
            diffusion_config = self.config['model']['diffusion']
            noise_steps = diffusion_config.get('timesteps', 1000)
            noise_offset = diffusion_config.get('noise_offset', 0)
            beta_start = diffusion_config.get('beta_start', 1e-4)
            beta_end = diffusion_config.get('beta_end', 0.02)
            
            # 创建模型
            with timer("模型创建"):
                model = ParagraphDiffusion(
                    noise_steps=noise_steps,
                    noise_offset=noise_offset,
                    beta_start=beta_start,
                    beta_end=beta_end,
                    device=self.device
                ).to(self.device)
            
            post_model_memory = get_cpu_memory_usage()
            post_model_gpu_memory = get_gpu_memory_usage() if torch.cuda.is_available() else 0
            
            print(f"模型创建后CPU内存使用: {post_model_memory:.2f} MB")
            print(f"模型创建导致的CPU内存增长: {post_model_memory - initial_memory:.2f} MB")
            
            if torch.cuda.is_available():
                print(f"模型创建后GPU内存使用: {post_model_gpu_memory:.2f} MB")
                print(f"模型创建导致的GPU内存增长: {post_model_gpu_memory - initial_gpu_memory:.2f} MB")
            
            # 测试不同批次大小的内存使用
            batch_sizes = [1, 2, 4, 8]
            for batch_size in batch_sizes:
                print(f"\n测试批次大小 {batch_size}:")
                
                # 创建输入数据
                dummy_input = torch.randn(batch_size, 3, 64, 64).to(self.device)
                style = torch.randn(batch_size, 1, 64, 64).to(self.device)  # 添加style参数
                laplace = torch.randn(batch_size, 1, 64, 64).to(self.device)  # 添加laplace参数
                content = torch.randn(batch_size, 1, 64, 64).to(self.device)  # 添加content参数，注意这里的1是时间步
                
                # 测量内存使用
                _ = model((dummy_input, style, laplace, content))
            
            # 测试内存释放
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            final_memory = get_cpu_memory_usage()
            final_gpu_memory = get_gpu_memory_usage() if torch.cuda.is_available() else 0
            
            print(f"\n清理后CPU内存使用: {final_memory:.2f} MB")
            if torch.cuda.is_available():
                print(f"清理后GPU内存使用: {final_gpu_memory:.2f} MB")
        except Exception as e:
            self.fail(f"内存分析测试失败: {str(e)}")
    
    def test_4_training_performance(self):
        """测试训练性能"""
        try:
            print("\n开始训练性能测试...")
            
            # 从配置中提取diffusion参数
            diffusion_config = self.config['model']['diffusion']
            noise_steps = diffusion_config.get('timesteps', 1000)
            noise_offset = diffusion_config.get('noise_offset', 0)
            beta_start = diffusion_config.get('beta_start', 1e-4)
            beta_end = diffusion_config.get('beta_end', 0.02)
            
            # 创建模型
            model = ParagraphDiffusion(
                noise_steps=noise_steps,
                noise_offset=noise_offset,
                beta_start=beta_start,
                beta_end=beta_end,
                device=self.device
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # 准备模拟数据
            batch_sizes = [1, 2, 4]
            training_steps = 5
            
            for batch_size in batch_sizes:
                print(f"\n测试批次大小 {batch_size}:")
                
                # 记录训练时间和内存使用
                training_times = []
                memory_usage = []
                
                for step in range(training_steps):
                    # 创建训练数据
                    train_data = (
                        torch.randn(batch_size, 3, 64, 64).to(self.device),  # dummy_input
                        torch.randn(batch_size, 1, 64, 64).to(self.device),  # style
                        torch.randn(batch_size, 1, 64, 64).to(self.device),  # laplace
                        torch.randn(batch_size, 1, 64, 64).to(self.device)   # content，注意这里的1是时间步
                    )
                    
                    # 记录开始时间和内存
                    start_time = time.time()
                    start_mem = get_gpu_memory_usage() if torch.cuda.is_available() else get_cpu_memory_usage()
                    
                    # 训练步骤
                    optimizer.zero_grad()
                    loss = model(train_data)
                    # 确保损失是标量值
                    if isinstance(loss, tuple):
                        loss = sum(l.mean() if isinstance(l, torch.Tensor) else l for l in loss)
                    else:
                        loss = loss.mean()
                    loss.backward()  # 使用普通的backward()
                    optimizer.step()
                    
                    # 记录结束时间和内存
                    end_time = time.time()
                    end_mem = get_gpu_memory_usage() if torch.cuda.is_available() else get_cpu_memory_usage()
                    
                    training_times.append(end_time - start_time)
                    memory_usage.append(end_mem - start_mem)
                    
                    print(f"步骤 {step + 1}, 损失: {loss.item():.6f}, "
                          f"时间: {training_times[-1]*1000:.2f} ms, "
                          f"内存: {memory_usage[-1]:.2f} MB")
                
                # 计算平均值和标准差
                avg_time = np.mean(training_times)
                std_time = np.std(training_times)
                avg_memory = np.mean(memory_usage)
                std_memory = np.std(memory_usage)
                
                print(f"\n批次大小 {batch_size} 的统计:")
                print(f"平均训练时间: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
                print(f"平均内存使用: {avg_memory:.2f} ± {std_memory:.2f} MB")
                print(f"训练吞吐量: {batch_size/avg_time:.2f} 样本/秒")
            
            # 清理内存
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.fail(f"训练性能测试失败: {str(e)}")
    
    def test_5_model_complexity(self):
        """测试模型复杂度"""
        try:
            print("\n开始模型复杂度分析...")
            
            # 从配置中提取diffusion参数
            diffusion_config = self.config['model']['diffusion']
            noise_steps = diffusion_config.get('timesteps', 1000)
            noise_offset = diffusion_config.get('noise_offset', 0)
            beta_start = diffusion_config.get('beta_start', 1e-4)
            beta_end = diffusion_config.get('beta_end', 0.02)
            
            # 创建模型
            model = ParagraphDiffusion(
                noise_steps=noise_steps,
                noise_offset=noise_offset,
                beta_start=beta_start,
                beta_end=beta_end,
                device=self.device
            ).to(self.device)
            
            # 分析模型结构
            print("\n模型结构分析:")
            total_params = 0
            trainable_params = 0
            
            # 按层统计参数
            layer_stats = defaultdict(lambda: {'params': 0, 'trainable': 0, 'size_mb': 0})
            
            for name, param in model.named_parameters():
                layer_name = name.split('.')[0]
                param_count = param.numel()
                
                total_params += param_count
                if param.requires_grad:
                    trainable_params += param_count
                
                layer_stats[layer_name]['params'] += param_count
                layer_stats[layer_name]['trainable'] += param_count if param.requires_grad else 0
                layer_stats[layer_name]['size_mb'] += param_count * 4 / (1024 * 1024)
            
            # 打印层统计信息
            print("\n各层参数统计:")
            for layer_name, stats in layer_stats.items():
                print(f"\n{layer_name}:")
                print(f"参数量: {stats['params']:,}")
                print(f"可训练参数: {stats['trainable']:,}")
                print(f"大小: {stats['size_mb']:.2f} MB")
                print(f"占总参数比例: {stats['params']/total_params*100:.2f}%")
            
            # 计算模型总大小
            total_size_mb = total_params * 4 / (1024 * 1024)
            
            print(f"\n模型总统计:")
            print(f"总参数量: {total_params:,}")
            print(f"可训练参数量: {trainable_params:,}")
            print(f"模型总大小: {total_size_mb:.2f} MB")
            print(f"参数利用率: {trainable_params/total_params*100:.2f}%")
            
            # 验证复杂度指标
            self.assertGreater(total_params, 1000)  # 确保模型不是太小
            self.assertEqual(total_params, trainable_params)  # 确保所有参数都是可训练的
            
            # 清理内存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.fail(f"模型复杂度分析失败: {str(e)}")

def run_tests():
    """运行所有性能测试"""
    print("开始性能测试...\n")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_loader = unittest.TestLoader()
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestPerformance))
    
    # 运行测试
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # 返回测试结果
    return test_result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 