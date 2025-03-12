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