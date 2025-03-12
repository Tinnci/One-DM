import torch
import numpy as np
import random
import os
import logging
import sys

# 添加setup_logger函数
def setup_logger(name, save_dir, distributed_rank=0, filename="log.txt"):
    """配置日志记录器
    
    Args:
        name: 日志记录器名称
        save_dir: 日志文件保存目录
        distributed_rank: 分布式训练中的进程rank
        filename: 日志文件名
    
    Returns:
        配置好的logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    
    # 控制台处理器
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 文件处理器
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

# 添加setup_determinism函数
def setup_determinism(seed=42, benchmark=False):
    """设置随机种子和确定性策略
    
    Args:
        seed: 随机种子
        benchmark: 是否启用cudnn benchmark模式
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark

# fix random seeds for reproducibility
def fix_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.manual_seed(random_seed)

# 添加fix_random_seed作为fix_seed的别名
fix_random_seed = fix_seed

### model loads specific parameters (i.e., par) from pretrained_model 
def load_specific_dict(model, pretrained_model, par):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model)
    if par in list(pretrained_dict.keys())[0]:
        count = len(par) + 1
        pretrained_dict = {k[count:]: v for k, v in pretrained_dict.items() if k[count:] in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    if len(pretrained_dict) > 0:
        model_dict.update(pretrained_dict)
    else:
        return ValueError
    return model_dict


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def move_to_device(obj, device):
    """
    将张量或包含张量的数据结构递归地移动到指定设备
    
    参数:
        obj: 需要移动的对象（张量、列表、元组、字典等）
        device: 目标设备（torch.device 对象或字符串，如 'cuda', 'cpu'）
        
    返回:
        移动到目标设备的对象
    """
    # 处理无效设备
    if device is None:
        # 如果未指定设备，则检查是否有可用的CUDA设备
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    # 如果设备是字符串，转换为torch.device
    if isinstance(device, str):
        device = torch.device(device)
    
    # 设备类型检查
    try:
        # 递归处理不同类型的对象
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_device(x, device) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(move_to_device(x, device) for x in obj)
        else:
            return obj
    except RuntimeError as e:
        # 处理常见的设备错误
        if "CUDA out of memory" in str(e):
            print(f"警告: CUDA内存不足，尝试在CPU上处理")
            # 如果CUDA内存不足，尝试在CPU上处理
            if device.type == 'cuda':
                return move_to_device(obj, 'cpu')
            else:
                raise  # 如果已经在CPU上，则重新抛出异常
        elif "Expected all tensors to be on the same device" in str(e):
            print(f"警告: 设备不匹配错误，确保所有张量在同一设备上")
            raise
        else:
            # 其他错误，打印信息并重新抛出
            print(f"移动到设备 {device} 时发生错误: {str(e)}")
            raise

# 添加一个检查张量设备的实用函数
def check_tensor_device(tensor_or_container):
    """
    检查张量或包含张量的容器中所有张量的设备
    
    参数:
        tensor_or_container: 张量或包含张量的容器(列表、字典等)
        
    返回:
        包含所有设备的列表
    """
    devices = []
    
    # 递归检查容器中的张量
    def _check_device(obj):
        if isinstance(obj, torch.Tensor):
            devices.append(obj.device)
        elif isinstance(obj, dict):
            for v in obj.values():
                _check_device(v)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                _check_device(x)
    
    _check_device(tensor_or_container)
    return list(set(devices))  # 返回唯一的设备列表

def ensure_same_device(inputs, target_device=None):
    """
    确保所有输入张量都在同一设备上
    
    参数:
        inputs: 单个张量或包含张量的字典、列表或元组
        target_device: 目标设备。如果为None，则使用第一个找到的张量的设备
    
    返回:
        处理后的输入，所有张量都在同一设备上
    """
    # 如果输入为None，直接返回
    if inputs is None:
        return None
    
    # 找出输入中的所有设备
    devices = check_tensor_device(inputs)
    
    # 如果没有张量，直接返回
    if not devices:
        return inputs
    
    # 确定目标设备
    if target_device is None:
        target_device = devices[0]  # 使用第一个找到的设备
    else:
        if isinstance(target_device, str):
            target_device = torch.device(target_device)
    
    # 如果所有张量已经在同一设备上，直接返回
    if len(devices) == 1 and devices[0] == target_device:
        return inputs
    
    # 移动张量到目标设备
    return move_to_device(inputs, target_device)

# 示例使用：在模型的forward方法开始处
# def forward(self, x, *args, **kwargs):
#     # 确保所有输入在同一设备上
#     device = x.device  # 或者 self.device
#     x = ensure_same_device(x, device)
#     args = [ensure_same_device(arg, device) for arg in args]
#     kwargs = {k: ensure_same_device(v, device) for k, v in kwargs.items()}