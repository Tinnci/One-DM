#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 one_dm 包中所有模块的导入是否正确
"""

import sys
import os
import importlib
from collections import defaultdict

def test_imports():
    """测试所有模块的导入"""
    print("开始测试 one_dm 模块的导入...")
    
    # 记录导入结果
    results = defaultdict(lambda: "成功")
    
    # 测试主模块导入
    try:
        import one_dm
        print(f"导入 one_dm 成功, 版本: {one_dm.__version__}")
    except ImportError as e:
        results["one_dm"] = f"失败: {str(e)}"
    except Exception as e:
        results["one_dm"] = f"失败 (未知错误): {str(e)}"
    
    # 测试子模块导入
    submodules = [
        "one_dm.data",
        "one_dm.models",
        "one_dm.trainer",
        "one_dm.utils",
    ]
    
    # 测试具体模块导入
    specific_modules = [
        # data 模块
        "one_dm.data.loader",
        "one_dm.data.paragraph_dataset",
        
        # models 模块
        "one_dm.models.diffusion",
        "one_dm.models.fusion",
        "one_dm.models.loss",
        "one_dm.models.paragraph_diffusion",
        "one_dm.models.paragraph_generator",
        "one_dm.models.paragraph_processing",
        "one_dm.models.paragraph_unet",
        "one_dm.models.recognition",
        "one_dm.models.resnet_dilation",
        "one_dm.models.transformer",
        "one_dm.models.unet",
        
        # trainer 模块
        "one_dm.trainer.paragraph_trainer",
        "one_dm.trainer.trainer",
        
        # utils 模块
        "one_dm.utils.logger",
        "one_dm.utils.parse_config",
        "one_dm.utils.util",
    ]
    
    # 测试子模块
    for module_name in submodules:
        try:
            module = importlib.import_module(module_name)
            print(f"导入 {module_name} 成功")
        except ImportError as e:
            results[module_name] = f"失败: {str(e)}"
        except Exception as e:
            results[module_name] = f"失败 (未知错误): {str(e)}"
    
    # 测试具体模块
    for module_name in specific_modules:
        try:
            module = importlib.import_module(module_name)
            print(f"导入 {module_name} 成功")
        except ImportError as e:
            results[module_name] = f"失败: {str(e)}"
        except Exception as e:
            results[module_name] = f"失败 (未知错误): {str(e)}"
    
    # 测试导入特定的类
    classes_to_import = {
        # data 模块
        "one_dm.data": ["IAMDataset", "ContentData", "ParagraphDataset"],
        
        # models 模块
        "one_dm.models": [
            "Diffusion", "Mix_TR", "LatentLoss", "StyleLoss", 
            "BaseRecognitionModel", "ResnetDilated", 
            "TransformerEncoder", "TransformerDecoder", 
            "UNetModel", "ParagraphDiffusion", 
            "ParagraphGenerator", "ParagraphProcessor", 
            "ParagraphUNetModel"
        ],
        
        # trainer 模块
        "one_dm.trainer": ["Trainer", "ParagraphTrainer"],
        
        # utils 模块
        "one_dm.utils": [
            "AverageMeter", "Logger", "setup_logger", 
            "fix_random_seed", "setup_determinism", 
            "parse_config", "Config"
        ],
    }
    
    for module_name, class_list in classes_to_import.items():
        try:
            module = importlib.import_module(module_name)
            for class_name in class_list:
                try:
                    cls = getattr(module, class_name)
                    print(f"从 {module_name} 导入 {class_name} 成功")
                except AttributeError as e:
                    results[f"{module_name}.{class_name}"] = f"失败: {str(e)}"
        except ImportError as e:
            for class_name in class_list:
                results[f"{module_name}.{class_name}"] = f"失败: 无法导入模块 {module_name}"
    
    # 输出失败的导入
    print("\n导入测试结果摘要:")
    failed_imports = {k: v for k, v in results.items() if v != "成功"}
    
    if not failed_imports:
        print("所有模块导入成功!")
    else:
        print("以下模块导入失败:")
        for module_name, error in failed_imports.items():
            print(f"  - {module_name}: {error}")
    
    return len(failed_imports) == 0

def run_tests():
    """为所有测试脚本提供统一接口"""
    # 确保项目根目录在 Python 路径中
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 运行导入测试
    return test_imports()

if __name__ == "__main__":
    # 确保项目根目录在 Python 路径中
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    success = test_imports()
    sys.exit(0 if success else 1) 