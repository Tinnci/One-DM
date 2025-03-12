"""
设备管理工具 - 处理模型和张量在设备间的一致性
"""

import torch
import logging
from typing import Union, List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def get_default_device() -> torch.device:
    """获取默认设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """确保张量在指定设备上"""
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor

def move_tensors_to_device(data: Any, device: torch.device) -> Any:
    """将数据中的所有张量移动到指定设备
    
    支持嵌套字典、列表和元组结构
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_tensors_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_device(item, device) for item in data)
    return data

def check_model_devices(model: torch.nn.Module) -> Dict[torch.device, List[str]]:
    """检查模型参数在哪些设备上
    
    返回:
        设备到参数名列表的映射
    """
    devices = {}
    for name, param in model.named_parameters():
        if param.device not in devices:
            devices[param.device] = []
        devices[param.device].append(name)
    
    if len(devices) > 1:
        logger.warning(f"模型参数分布在{len(devices)}个不同设备上!")
        for device, params in devices.items():
            logger.warning(f"设备 {device}：{len(params)} 个参数")
    
    return devices

def verify_model_on_device(model, device):
    """
    验证模型的所有参数是否都在指定的设备上，并返回不在指定设备上的参数列表。
    
    Args:
        model (nn.Module): 待验证的模型
        device (str 或 torch.device): 目标设备
        
    Returns:
        list: 不在指定设备上的参数名称列表
    """
    incorrect_params = []
    
    # 将设备转换为字符串以便处理
    device_str = str(device)
    
    # 提取设备类型（cuda 或 cpu），忽略设备索引
    target_type = device_str.split(':')[0] if ':' in device_str else device_str
    
    for name, param in model.named_parameters():
        # 获取参数的设备类型
        param_device = str(param.device)
        param_type = param_device.split(':')[0] if ':' in param_device else param_device
        
        # 特殊处理cuda设备：cuda等同于cuda:0
        if (target_type == 'cuda' and param_device == 'cuda:0') or (device_str == 'cuda:0' and param_device == 'cuda'):
            # 这两种情况视为相同设备，不添加到incorrect_params
            continue
            
        # 其他情况比较设备类型
        if param_type != target_type:
            incorrect_params.append(name)
    
    if incorrect_params:
        logger.warning(f"模型参数在错误的设备上. 预期设备: {device_str}")
        for param in incorrect_params:
            param_device = next(p.device for n, p in model.named_parameters() if n == param)
            logger.warning(f"  - {param}: {param_device}")
    
    return incorrect_params

def move_model_to_device(model: torch.nn.Module, device: torch.device, recursive: bool = True) -> torch.nn.Module:
    """移动模型到指定设备
    
    Args:
        model: 要移动的模型
        device: 目标设备
        recursive: 是否递归处理所有子模块
        
    Returns:
        移动后的模型
    """
    if not recursive:
        return model.to(device)
    
    # 递归处理所有子模块
    model = model.to(device)
    for child in model.children():
        move_model_to_device(child, device, recursive=True)
    
    return model

def print_tensor_info(tensor: torch.Tensor, name: str) -> None:
    """打印张量信息"""
    print(f"{name} - 形状: {tensor.shape}, 设备: {tensor.device}, 类型: {tensor.dtype}")

def content_type_checker(content: Any) -> Optional[torch.Tensor]:
    """检查并处理内容数据类型
    
    确保内容可以表示为张量，或返回None
    """
    if content is None:
        logger.warning("警告：content为None")
        return None
        
    if isinstance(content, torch.Tensor):
        logger.debug(f"Content是张量 - 形状: {content.shape}, 设备: {content.device}, 类型: {content.dtype}")
        return content
        
    if isinstance(content, (list, tuple)):
        logger.debug(f"Content是{type(content).__name__} - 长度: {len(content)}, 元素类型: {type(content[0]) if content else 'unknown'}")
        try:
            content_tensor = torch.tensor(content, device=get_default_device())
            logger.debug(f"已转换为张量 - 形状: {content_tensor.shape}")
            return content_tensor
        except Exception as e:
            logger.error(f"无法将content转换为张量: {str(e)}")
            
            # 尝试替代转换方法
            if all(isinstance(x, str) for x in content):
                logger.warning("检测到全是字符串，应使用ContentData类处理")
            
    logger.error(f"未知content类型: {type(content)}")
    return None

class DeviceManager:
    """设备管理器 - 管理模型和张量设备一致性"""
    
    def __init__(self, device=None):
        self.device = device if device is not None else get_default_device()
        logger.info(f"DeviceManager初始化在设备: {self.device}")
    
    def prepare_batch(self, batch):
        """准备批次数据，确保所有张量都在相同设备上"""
        return move_tensors_to_device(batch, self.device)
    
    def prepare_model(self, model):
        """准备模型，确保所有参数都在相同设备上"""
        model = move_model_to_device(model, self.device)
        # 验证所有参数是否都移到了目标设备
        incorrect_params = verify_model_on_device(model, self.device)
        if incorrect_params:
            logger.warning(f"模型移动后仍有 {len(incorrect_params)} 个参数不在目标设备上")
        return model
    
    def diagnose_batch(self, batch):
        """诊断批次数据，检查设备一致性"""
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: 形状={v.shape}, 设备={v.device}, 类型={v.dtype}")
        elif isinstance(batch, torch.Tensor):
            print(f"批次是张量: 形状={batch.shape}, 设备={batch.device}, 类型={batch.dtype}")
        else:
            print(f"批次类型: {type(batch)}")
            
    def diagnose_model(self, model):
        """诊断模型，检查设备一致性"""
        devices = check_model_devices(model)
        for device, params in devices.items():
            print(f"设备 {device}：{len(params)} 个参数")
            if len(devices) > 1 and len(params) < 10:
                for p in params:
                    print(f"  - {p}") 