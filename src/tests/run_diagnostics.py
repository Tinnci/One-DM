#!/usr/bin/env python3
"""
One-DM项目诊断运行脚本

此脚本运行所有诊断工具，帮助检测和定位项目中的问题
"""

import os
import sys
import logging
import argparse
import time

# 确保能够导入one_dm模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.one_dm.utils.diagnose import diagnose_one_dm, test_device_consistency, test_content_handling
from src.one_dm.utils.device_utils import DeviceManager
import torch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('diagnostic_results.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='运行One-DM项目诊断工具')
    parser.add_argument('--basic', action='store_true', help='只运行基本诊断')
    parser.add_argument('--device', action='store_true', help='运行设备一致性测试')
    parser.add_argument('--content', action='store_true', help='运行内容处理测试')
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    parser.add_argument('--device-type', choices=['cuda', 'cpu'], default=None, 
                        help='指定要使用的设备类型')
    parser.add_argument('--log-file', type=str, default='diagnostic_results.log',
                        help='诊断结果日志文件路径')
    return parser.parse_args()

def run_diagnostics(args):
    """运行诊断工具"""
    logger.info("=" * 60)
    logger.info("One-DM项目诊断开始")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # 设置设备
    if args.device_type:
        if args.device_type == 'cuda' and not torch.cuda.is_available():
            logger.warning("请求使用CUDA，但CUDA不可用。将使用CPU。")
            device = torch.device('cpu')
        else:
            device = torch.device(args.device_type)
        logger.info(f"使用指定设备: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用默认设备: {device}")
    
    # 运行基本诊断
    if args.basic or args.all:
        logger.info("\n运行基本诊断...")
        diagnose_one_dm()
        
    # 运行设备一致性测试
    if args.device or args.all:
        logger.info("\n运行设备一致性测试...")
        test_device_consistency()
        
    # 运行内容处理测试
    if args.content or args.all:
        logger.info("\n运行内容处理测试...")
        test_content_handling()
    
    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"One-DM项目诊断完成，耗时: {elapsed_time:.2f}秒")
    logger.info("=" * 60)

if __name__ == "__main__":
    args = parse_args()
    
    # 如果没有指定任何测试，则运行所有测试
    if not (args.basic or args.device or args.content or args.all):
        args.all = True
        
    run_diagnostics(args) 