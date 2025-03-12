# One-DM 段落级手写文本生成功能

本文档介绍如何使用One-DM模型的段落级手写文本生成功能，包括安装、训练和使用方法。

## 一、功能介绍

One-DM模型现在支持段落级手写文本生成，可以在保持一致风格的情况下生成完整的文本段落。新增的主要功能包括：

1. **风格一致性**：整个段落保持统一的手写风格，包括笔迹粗细、倾斜度、压力等特征。
2. **空间布局控制**：支持调整行间距、对齐方式（左对齐、居中、右对齐）。
3. **全局风格转移**：从单个参考图像中提取风格，并应用于整个段落。

## 二、安装依赖

在使用段落级功能前，确保已安装以下依赖：

```bash
pip install torch torchvision diffusers==0.8.0 opencv-python einops pillow
```

## 三、代码结构

段落级生成的主要文件包括：

- `models/paragraph_processing.py`: 段落级特征提取和处理
- `models/paragraph_unet.py`: 扩展UNetModel以支持段落级特征
- `models/paragraph_diffusion.py`: 扩展Diffusion类以支持段落生成
- `models/paragraph_generator.py`: 段落生成器
- `datasets/paragraph_dataset.py`: 段落级数据集处理
- `trainer/paragraph_trainer.py`: 段落级训练器
- `train_paragraph.py`: 段落级训练脚本
- `test_paragraph.py`: 段落级测试脚本

## 四、数据集准备

要训练段落级生成模型，首先需要准备段落数据集：

1. 将现有的笔记或手写图像收集到`data/paragraph`目录下。
2. 使用以下命令创建段落级数据集：

```python
from datasets.paragraph_dataset import ParagraphProcessor
from data_loader.loader import IAMDataset

# 加载基础IAM数据集
base_dataset = IAMDataset(data_path="data/IAM", split="train")

# 创建段落数据集
para_processor = ParagraphProcessor()
para_processor.create_paragraph_dataset(
    base_dataset=base_dataset,
    output_dir="data/paragraph",
    n_paragraphs=1000  # 生成的段落数量
)
```

## 五、模型训练

### 1. 基础训练

首先，使用以下命令进行基础训练：

```bash
python train_paragraph.py \
  --cfg configs/train/base.yaml \
  --data_path data/paragraph \
  --batch_size 8 \
  --ep 50 \
  --unet_path logs/model/unet_xxxx.pth  # 原始单字级模型权重
```

如果需要在多GPU上进行分布式训练，请添加`--dist`参数：

```bash
python -m torch.distributed.launch --nproc_per_node=4 train_paragraph.py \
  --cfg configs/train/base.yaml \
  --data_path data/paragraph \
  --batch_size 8 \
  --ep 50 \
  --unet_path logs/model/unet_xxxx.pth \
  --dist
```

### 2. 可读性微调（可选）

为了提高生成文本的可读性，可以进行额外的微调：

```bash
python train_paragraph.py \
  --cfg configs/train/base.yaml \
  --data_path data/paragraph \
  --batch_size 8 \
  --ep 20 \
  --unet_path logs/paragraph/model/paragraph_unet_final.pth \
  --finetune
```

## 六、文本生成

### 1. 单段文本生成

使用以下命令生成单个段落：

```bash
python test_paragraph.py \
  --unet_path logs/paragraph/model/paragraph_unet_final.pth \
  --style_image examples/styles/style1.png \
  --text_file examples/content/para1.txt \
  --output_dir outputs \
  --alignment 0  # 0=左对齐，1=居中，2=右对齐
```

### 2. 批量生成

```python
from test_paragraph import batch_generate

# 批量生成多个风格和文本组合
batch_generate(
    style_dir="examples/styles",
    text_dir="examples/content",
    output_dir="outputs/batch",
    unet_path="logs/paragraph/model/paragraph_unet_final.pth",
    stable_dif_path="stabilityai/sd-vae-ft-mse",
    alignment=0,
    line_spacing=30,
    paragraph_width=800
)
```

### 3. 在Python代码中使用

```python
import torch
from PIL import Image
from models.paragraph_unet import ParagraphUNetModel
from diffusers import AutoencoderKL
from models.paragraph_diffusion import ParagraphDiffusion
from models.paragraph_generator import ParagraphGenerator

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = ParagraphUNetModel(...).to(device)
unet.load_state_dict(torch.load("logs/paragraph/model/paragraph_unet_final.pth"))
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", subfolder="vae").to(device)
diffusion = ParagraphDiffusion(device=device)

# 创建生成器
generator = ParagraphGenerator(diffusion, unet, vae, device)

# 生成段落
style_image = Image.open("examples/styles/style1.png").convert("L")
content_texts = ["这是第一行文本。", "这是第二行文本。", "这是第三行文本。"]

result = generator.generate_paragraph(
    style_image=style_image,
    content_texts=content_texts,
    output_path="outputs/para_example.png",
    alignment=0,  # 左对齐
    line_spacing=30,
    paragraph_width=800
)
```

## 七、参数调整

### 1. 生成参数

在生成段落时，可以调整以下参数：

- `alignment`: 对齐方式（0=左对齐，1=居中，2=右对齐）
- `line_spacing`: 行间距（像素）
- `paragraph_width`: 段落宽度（像素）
- `font_size`: 字体大小（像素高度），如果为None则自动从风格图像中估计
- `slant_angle`: 字体倾斜角度，如果为None则自动从风格图像中估计

### 2. 模型参数

训练时，可以调整以下模型参数以适应不同的场景：

- `paragraph_features_dim`: 段落特征向量的维度，默认为12
- `enable_paragraph_consistency`: 是否启用段落级一致性机制，默认为True

## 八、技术细节

### 段落特征

模型分析以下段落级特征来指导生成过程：

1. **行高特征**：分析行的平均高度及变化
2. **行间距特征**：行之间的平均距离
3. **对齐方式**：文本的左、中、右对齐
4. **倾斜角度**：文字的整体倾斜角度
5. **单词间距**：单词之间的平均间距

### 全局风格一致性

使用全局风格一致性模块（GlobalStyleConsistency）确保段落中所有生成的文本保持一致的风格：

1. 使用风格一致性tokens捕获全局风格信息
2. 通过交叉注意力机制将风格信息应用到文本内容上
3. 基于位置信息计算权重，使相邻位置的风格更相似

## 九、已知问题与限制

1. 在极长段落上可能会出现风格逐渐漂移的问题
2. 中文和英文混合段落处理时可能会有字符间距不一致的现象
3. 对背景复杂的风格参考图像处理效果可能不佳

## 十、未来改进

1. 增强多语言支持，特别是中文段落处理
2. 改进段落中的标点符号处理
3. 支持更多样化的排版格式，如项目符号、缩进等
4. 提高长段落的一致性 