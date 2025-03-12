"""
One-DM项目诊断工具

该模块提供了一套用于诊断和验证设备一致性及其他问题的工具函数
"""

import torch
import os
import logging
import time
import random
import numpy as np
from typing import Optional, Dict, List, Any, Tuple, Union
from src.one_dm.utils.device_utils import DeviceManager, check_model_devices, verify_model_on_device
from src.one_dm.models.paragraph_diffusion import ParagraphDiffusion
from src.one_dm.data.loader import ContentData

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def diagnose_one_dm():
    """One-DM项目诊断函数"""
    logger.info("===== One-DM 项目诊断工具 =====")
    
    # 检查CUDA可用性
    logger.info("\n1. 设备信息:")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")
        logger.info(f"当前设备: {torch.cuda.current_device()}")
        logger.info(f"设备名称: {torch.cuda.get_device_name(0)}")
    
    # 加载模型并检查设备一致性
    logger.info("\n2. 模型设备一致性检查:")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ParagraphDiffusion(device=device)
        
        # 检查模型设备
        devices = check_model_devices(model)
        for dev, params in devices.items():
            logger.info(f"设备 {dev}：{len(params)} 个参数")
        
        # 验证模型参数是否都在正确的设备上
        incorrect_params = verify_model_on_device(model, device)
        if incorrect_params:
            logger.warning(f"发现 {len(incorrect_params)} 个参数不在目标设备 {device} 上")
            for name, dev in incorrect_params[:10]:  # 只显示前10个
                logger.warning(f"  - {name}: {dev}")
        else:
            logger.info(f"所有参数都在目标设备 {device} 上")
            
        # 详细检查UNetModel
        logger.info("\n2.1 UNetModel设备一致性检查:")
        if hasattr(model, 'unet'):
            unet = model.unet
            unet_devices = check_model_devices(unet)
            for dev, params in unet_devices.items():
                logger.info(f"UNet设备 {dev}：{len(params)} 个参数")
            
            # 检查mix_net
            if hasattr(unet, 'mix_net'):
                logger.info("\n2.1.1 Mix_TR设备一致性检查:")
                mix_net = unet.mix_net
                mix_net_devices = check_model_devices(mix_net)
                for dev, params in mix_net_devices.items():
                    logger.info(f"Mix_TR设备 {dev}：{len(params)} 个参数")
                
                # 尝试修复Mix_TR设备问题
                if len(mix_net_devices) > 1:
                    logger.info("尝试修复Mix_TR设备不一致问题...")
                    mix_net.to(device)
                    mix_net_devices_after = check_model_devices(mix_net)
                    for dev, params in mix_net_devices_after.items():
                        logger.info(f"修复后Mix_TR设备 {dev}：{len(params)} 个参数")
            
            # 尝试修复UNet设备问题
            if len(unet_devices) > 1:
                logger.info("尝试修复UNet设备不一致问题...")
                unet.to(device)
                unet_devices_after = check_model_devices(unet)
                for dev, params in unet_devices_after.items():
                    logger.info(f"修复后UNet设备 {dev}：{len(params)} 个参数")
        
        # 再次检查整个模型
        logger.info("\n2.2 修复后的模型设备一致性检查:")
        model.to(device)  # 再次调用to方法
        devices_after = check_model_devices(model)
        for dev, params in devices_after.items():
            logger.info(f"修复后设备 {dev}：{len(params)} 个参数")
        
        incorrect_params_after = verify_model_on_device(model, device)
        if incorrect_params_after:
            logger.warning(f"修复后仍有 {len(incorrect_params_after)} 个参数不在目标设备 {device} 上")
            for name, dev in incorrect_params_after[:10]:  # 只显示前10个
                logger.warning(f"  - {name}: {dev}")
        else:
            logger.info(f"修复后所有参数都在目标设备 {device} 上")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 检查数据加载
    logger.info("\n3. 数据加载测试:")
    try:
        content_data = ContentData()
        sample_content = content_data.get_content("testing")
        logger.info(f"ContentData测试成功")
        logger.info(f"样本内容形状: {sample_content.shape}, 设备: {sample_content.device}")
        
        # 测试设备迁移
        if torch.cuda.is_available():
            cuda_content = content_data.get_content("testing", device=torch.device("cuda"))
            logger.info(f"内容已成功移至CUDA: {cuda_content.device}")
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
    
    # 检查文件结构
    logger.info("\n4. 目录结构检查:")
    data_dir = "data"
    if not os.path.exists(data_dir):
        logger.warning(f"数据目录不存在: {data_dir}")
    else:
        logger.info(f"数据目录存在: {data_dir}")
        unifont_path = os.path.join(data_dir, "unifont.pickle")
        if os.path.exists(unifont_path):
            logger.info(f"字体文件存在: {unifont_path}")
        else:
            logger.warning(f"字体文件不存在: {unifont_path}")
    
    # 检查TensorBoard日志
    logger.info("\n5. TensorBoard日志检查:")
    logs_dir = "logs/tensorboard"
    if os.path.exists(logs_dir):
        event_files = [f for f in os.listdir(logs_dir) if f.startswith("events.out.tfevents")]
        logger.info(f"发现 {len(event_files)} 个TensorBoard日志文件")
    else:
        logger.info(f"TensorBoard日志目录不存在: {logs_dir}")
    
    logger.info("\n===== 诊断完成 =====")

def test_device_consistency():
    """
    测试模型在不同设备间的一致性
    
    1. 创建两个相同的模型
    2. 一个放在CPU上，一个放在CUDA上
    3. 使用相同的输入运行，确保输出一致
    """
    logger.info("测试模型在不同设备间的一致性...")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，跳过测试")
        return
    
    # 设置随机种子确保一致性
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    try:
        # 创建两个模型
        cpu_model = ParagraphDiffusion(device=torch.device("cpu"))
        cuda_model = ParagraphDiffusion(device=torch.device("cuda"))
        
        # 确保模型具有相同的权重
        cpu_state = cpu_model.state_dict()
        cuda_model.load_state_dict(cpu_state)
        
        # 创建相同的输入数据
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)
        t = torch.randint(0, 1000, (batch_size,))
        styles = torch.randn(batch_size, 2, 64, 64)
        laplace = torch.randn(batch_size, 2, 64, 64)
        content = torch.randn(batch_size, 1, 16, 16)
        
        # CPU前向传播
        logger.info("在CPU上运行模型...")
        with torch.no_grad():
            cpu_output = cpu_model(x, t, styles, laplace, content)
        
        # CUDA前向传播
        logger.info("在CUDA上运行模型...")
        x_cuda = x.to("cuda")
        t_cuda = t.to("cuda")
        styles_cuda = styles.to("cuda")
        laplace_cuda = laplace.to("cuda")
        content_cuda = content.to("cuda")
        
        with torch.no_grad():
            cuda_output = cuda_model(x_cuda, t_cuda, styles_cuda, laplace_cuda, content_cuda)
            
        # 比较输出
        cuda_output_cpu = cuda_output.to("cpu")
        tolerance = 1e-5
        max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
        
        if max_diff < tolerance:
            logger.info(f"测试通过! CPU和CUDA输出一致 (最大差异: {max_diff})")
        else:
            logger.warning(f"测试失败! CPU和CUDA输出不一致 (最大差异: {max_diff})")
            
        logger.info(f"CPU输出: 形状={cpu_output.shape}, 均值={cpu_output.mean():.6f}, 标准差={cpu_output.std():.6f}")
        logger.info(f"CUDA输出: 形状={cuda_output.shape}, 均值={cuda_output.mean():.6f}, 标准差={cuda_output.std():.6f}")
        
    except Exception as e:
        logger.error(f"设备一致性测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

def test_content_handling():
    """
    测试ContentData类处理不同类型内容的能力
    """
    logger.info("测试内容处理...")
    
    content_loader = ContentData()
    
    # 测试字符串输入
    logger.info("测试字符串输入...")
    string_content = content_loader.get_content("testing")
    logger.info(f"字符串内容: 形状={string_content.shape}, 类型={string_content.dtype}")
    
    # 测试字符串列表输入
    logger.info("测试字符串列表输入...")
    string_list_content = content_loader.get_content(["hello", "world"])
    logger.info(f"字符串列表内容: 形状={string_list_content.shape}, 类型={string_list_content.dtype}")
    
    # 测试设备参数
    if torch.cuda.is_available():
        logger.info("测试设备参数...")
        cuda_content = content_loader.get_content("testing", device=torch.device("cuda"))
        logger.info(f"CUDA内容: 形状={cuda_content.shape}, 设备={cuda_content.device}")
    
    # 测试缓存
    logger.info("测试缓存...")
    start_time = time.time()
    first_call = content_loader.get_content("performance test")
    first_time = time.time() - start_time
    
    start_time = time.time()
    second_call = content_loader.get_content("performance test")
    second_time = time.time() - start_time
    
    logger.info(f"首次调用时间: {first_time:.6f}秒")
    logger.info(f"缓存后调用时间: {second_time:.6f}秒")
    logger.info(f"加速比: {first_time/second_time:.1f}倍")
    
    # 测试边界情况
    logger.info("测试边界情况...")
    try:
        empty_content = content_loader.get_content("")
        logger.info(f"空字符串处理: 形状={empty_content.shape}")
    except Exception as e:
        logger.warning(f"空字符串处理出错: {str(e)}")
        
    try:
        special_chars = content_loader.get_content("!@#$%^&*()")
        logger.info(f"特殊字符处理: 形状={special_chars.shape}")
    except Exception as e:
        logger.warning(f"特殊字符处理出错: {str(e)}")
        
    logger.info("内容处理测试完成")
        
if __name__ == "__main__":
    diagnose_one_dm()
    test_device_consistency()
    test_content_handling() 