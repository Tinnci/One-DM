# One-DM 测试指南

本目录包含 One-DM 项目的所有测试文件，用于确保项目各组件的正确性。

## 测试文件说明

- `test_import.py`: 测试模块导入
- `test_components.py`: 测试项目核心组件
- `test_data_loader.py`: 测试数据加载功能
- `test_performance.py`: 测试性能相关指标
- `test_integration.py`: 集成测试
- `test_model_params.py`: 测试模型参数
- `test_mock_data.py`: 使用模拟数据进行测试
- `test_paragraph_dataset.py`: 专门测试段落数据集功能

## 如何运行测试

### 运行所有测试

```bash
cd src/tests
python run_all_tests.py
```

### 运行特定测试

可以单独运行特定的测试文件，例如：

```bash
python test_import.py
python test_components.py
```

### 运行段落数据集测试

我们提供了一个专门的脚本来运行段落数据集测试：

```bash
python run_paragraph_test.py
```

## 测试段落数据集

`test_paragraph_dataset.py` 提供了对 `paragraph_dataset.py` 的全面测试，包括：

1. **基础功能测试**
   - 数据集初始化
   - 样本获取
   - 批处理功能

2. **设备处理测试**
   - 确保所有张量在同一设备上
   - 测试数据在不同设备间移动

3. **段落处理器测试**
   - 段落特征提取
   - 段落数据集创建

## 调试测试失败

如果测试失败，请检查以下几点：

1. **设备不匹配问题**
   - 确保所有张量和模型参数在同一设备上
   - 使用 `.to(device)` 方法确保张量设备一致
   - 修改模型加载代码，确保参数加载到正确设备

2. **数据格式问题**
   - 确保元数据文件格式正确
   - 确保图像文件存在且可读取

3. **依赖问题**
   - 确保所有必要的依赖已安装
   - 检查项目路径是否正确设置

## 添加新测试

如需添加新的测试，请遵循以下步骤：

1. 创建新的测试文件，命名为 `test_*.py`
2. 实现 `run_tests()` 函数作为统一接口
3. 将测试模块添加到 `run_all_tests.py` 的 `test_modules` 列表中

## 模拟数据

测试使用的模拟数据会在临时目录中创建，测试完成后自动清理。如果需要保留模拟数据进行调试，可以修改 `tearDownClass` 方法中的清理逻辑。 