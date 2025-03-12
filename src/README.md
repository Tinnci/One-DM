# One-DM 目录结构重构

## 目录结构

这个新的目录结构遵循Python最佳实践，使用src布局：

```
One-DM/
├── src/                   # 源代码目录
│   └── one_dm/           # 主要包
│       ├── __init__.py
│       ├── data/          # 数据加载模块
│       ├── models/        # 模型定义
│       ├── trainer/       # 训练逻辑
│       └── utils/         # 工具函数
├── scripts/               # 训练和测试脚本
├── examples/              # 示例代码
├── tests/                 # 测试代码
├── config/                # 配置文件
├── data/                  # 数据目录
├── outputs/               # 输出目录
└── setup.py              # 安装脚本
```

## 迁移计划

目前，为了保持兼容性，我们同时保留了旧目录结构和新目录结构。这使得现有的导入语句继续正常工作，同时允许逐步迁移到新结构。

迁移步骤：

1. 对于新代码，使用 `src.one_dm.xxx` 导入路径
2. 逐步更新现有文件中的导入语句，从旧路径改为新路径
3. 完成所有文件的迁移后，删除旧目录结构

## 导入示例

旧的导入方式：
```python
from models.unet import UNetModel
from trainer.trainer import Trainer
from utils.util import fix_random_seed
```

新的导入方式：
```python
from src.one_dm.models.unet import UNetModel
from src.one_dm.trainer.trainer import Trainer
from src.one_dm.utils.util import fix_random_seed
```

或者在安装项目后：
```python
from one_dm.models.unet import UNetModel
from one_dm.trainer.trainer import Trainer
from one_dm.utils.util import fix_random_seed
``` 