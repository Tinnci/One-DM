import time
import os
import logging
import sys

class AverageMeter(object):
    """用于计算和存储训练过程中的平均值和当前值
    
    用于记录训练过程中的各种指标（如损失、准确率等）的平均值和累计值
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    """日志记录器类
    
    用于记录训练过程中的日志信息
    """
    def __init__(self, log_file=None, log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_format = logging.Formatter('[%(asctime)s] - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # 文件处理器（如果提供了日志文件）
        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_format = logging.Formatter('[%(asctime)s] - %(message)s')
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)

""" prepare logdir for tensorboard and logging output"""
def set_log(output_dir, cfg_file, log_name):
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base_name = os.path.basename(cfg_file).split('.')[0]
    log_dir = os.path.join(output_dir, base_name, log_name + "-" + t)
    logs = {}
    for temp in ['tboard', 'model', 'sample']:
        temp_dir = os.path.join(log_dir, temp)
        os.makedirs(temp_dir, exist_ok=True)
        logs[temp] = temp_dir
    return logs