# 从conda迁移到uv的指南

本指南将帮助您将One-DM项目从conda环境迁移到使用uv进行依赖管理。

## 什么是uv？

uv是一个新的Python包管理器和安装工具，它比pip/conda更快，并解决了它们的一些问题。uv支持从requirements.txt、pyproject.toml和setup.py安装依赖，同时保持与pip兼容。

## 迁移步骤

### 1. 安装uv

```bash
pip install uv
```

### 2. 创建和激活虚拟环境

```bash
# 创建一个使用Python 3.8的虚拟环境（必须使用3.8版本）
uv venv --python=3.8
```

### 3. 安装项目依赖

有多种方式安装依赖:

```bash
# 从项目根目录安装
uv pip install -e .

# 或者从requirements.txt安装
uv pip install -r requirements.txt

# 安装带CUDA支持的PyTorch（如果上面的安装没有包含CUDA支持）
uv pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
```

## 与conda环境的区别

1. **环境激活**: 不需要显式激活环境，uv管理的虚拟环境自动与命令一起使用。
2. **安装速度**: uv通常比conda和pip快得多。
3. **依赖解析**: uv提供更快且更准确的依赖解析。

## 常见问题解答

### Q: 我可以同时使用uv和conda吗？
A: 是的，您可以在同一个系统上同时安装两者，但不建议在同一个项目中混用。

### Q: 如何确认我的PyTorch已经正确安装并支持CUDA？
A: 运行以下Python代码：

```python
import torch
print(torch.cuda.is_available())  # 如果返回True，则CUDA已正确配置
```

### Q: uv是否支持所有平台？
A: uv支持Windows、macOS和Linux。

### Q: 为什么必须使用Python 3.8？
A: 项目依赖的某些库（尤其是PyTorch 1.13.1和相关库）与Python 3.8最兼容。使用其他版本的Python可能会导致兼容性问题。

### Q: 我使用uv venv命令时报错怎么办？
A: 确保使用`--python=3.8`参数显式指定Python版本，例如：`uv venv --python=3.8`

## 更多资源

- [uv官方文档](https://github.com/astral-sh/uv)
- [uv与PyTorch集成指南](https://docs.astral.sh/uv/guides/integration/pytorch/) 