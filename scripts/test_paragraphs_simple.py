import os
import torch
import sys
from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np
import random

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.one_dm.models.paragraph_unet import ParagraphUNetModel
from diffusers import AutoencoderKL
from src.one_dm.models.paragraph_diffusion import ParagraphDiffusion
from src.one_dm.models.paragraph_generator import ParagraphGenerator

def create_simple_dataset():
    """创建简单的测试数据集"""
    print("创建简单的测试数据集...")
    
    # 创建目录
    dataset_dir = "data/english_test"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "styles"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "content"), exist_ok=True)
    
    # 创建风格参考图像
    try:
        img = Image.new('L', (400, 64), color=255)  # 白底
        draw = ImageDraw.Draw(img)
        try:
            # 尝试加载系统字体
            font = ImageFont.load_default()
            if os.name == 'nt':  # Windows
                try:
                    font = ImageFont.truetype("arial.ttf", 36)
                except:
                    print("无法加载Arial字体，使用默认字体")
        except:
            print("无法加载字体，使用默认方法绘制")
        
        # 绘制文本
        draw.text((20, 10), "Sample Text Style", fill=0)  # 黑色文本
        
        # 保存参考风格图像
        style_path = os.path.join(dataset_dir, "styles/style_1.png")
        img.save(style_path)
        print(f"已创建风格参考图像: {style_path}")
    except Exception as e:
        print(f"创建风格图像时出错: {e}")
    
    # 创建内容文本文件
    test_paragraphs = [
        # 段落1：简单文本
        [
            "This is a sample paragraph.",
            "It contains multiple lines of text.",
            "Each line has a different length.",
            "We can test the alignment and spacing."
        ]
    ]
    
    # 保存段落文本
    for i, paragraph in enumerate(test_paragraphs):
        content_path = os.path.join(dataset_dir, f"content/paragraph_{i+1}.txt")
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(paragraph))
        print(f"已创建内容文本文件: {content_path}")
    
    return dataset_dir

def generate_paragraphs(dataset_dir):
    """生成段落文本"""
    print("开始生成段落...")
    
    # 设置输出目录
    output_dir = "outputs/english_test_simple"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    print("创建模型...")
    unet = ParagraphUNetModel(
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=(4, 2, 1),
        dropout=0.0,
        channel_mult=(1, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=8,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=768,
        paragraph_features_dim=12,
        enable_paragraph_consistency=True,
    )
    unet = unet.to(device)
    unet.eval()
    
    # 加载VAE模型
    try:
        print("加载VAE模型...")
        # 首先尝试从官方来源加载
        try:
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", subfolder="vae")
        except:
            # 如果失败，尝试从本地加载
            print("从官方来源加载VAE失败，尝试从本地加载...")
            if os.path.exists("model_zoo/sd-vae-ft-mse"):
                vae = AutoencoderKL.from_pretrained("model_zoo/sd-vae-ft-mse")
            else:
                # 如果本地也没有，使用随机初始化的VAE
                print("无法加载预训练VAE，使用随机初始化模型")
                vae = AutoencoderKL(
                    in_channels=3,
                    out_channels=3,
                    down_block_types=["DownEncoderBlock2D"] * 4,
                    up_block_types=["UpDecoderBlock2D"] * 4,
                    block_out_channels=[64, 128, 256, 512],
                    latent_channels=4,
                )
        vae.requires_grad_(False)
        vae = vae.to(device)
        vae.eval()
    except Exception as e:
        print(f"加载VAE模型时出错: {e}")
        return
    
    # 创建扩散模型
    diffusion = ParagraphDiffusion(device=device)
    
    # 创建段落生成器
    print("创建段落生成器...")
    generator = ParagraphGenerator(diffusion, unet, vae, device)
    
    # 加载样式和内容
    styles_dir = os.path.join(dataset_dir, "styles")
    content_dir = os.path.join(dataset_dir, "content")
    
    # 获取所有风格和内容文件
    style_files = [f for f in os.listdir(styles_dir) if f.endswith(('.png', '.jpg'))]
    content_files = [f for f in os.listdir(content_dir) if f.endswith('.txt')]
    
    print(f"找到 {len(style_files)} 个风格图像和 {len(content_files)} 个内容文件")
    
    # 生成段落
    for style_file in style_files:
        style_path = os.path.join(styles_dir, style_file)
        style_name = os.path.splitext(style_file)[0]
        
        # 加载风格图像
        try:
            style_image = Image.open(style_path).convert('L')
        except Exception as e:
            print(f"加载风格图像时出错: {e}")
            continue
        
        for content_file in content_files:
            content_path = os.path.join(content_dir, content_file)
            content_name = os.path.splitext(content_file)[0]
            
            # 读取内容文本
            try:
                with open(content_path, 'r', encoding='utf-8') as f:
                    content_lines = f.read().strip().split('\n')
            except Exception as e:
                print(f"读取内容文件时出错: {e}")
                continue
            
            print(f"生成段落: 风格 {style_name} + 内容 {content_name}")
            
            # 测试所有对齐方式
            for alignment, alignment_name in [(0, "left"), (1, "center"), (2, "right")]:
                output_path = os.path.join(output_dir, f"{style_name}_{content_name}_{alignment_name}.png")
                
                try:
                    # 生成段落
                    result = generator.generate_paragraph(
                        style_image=style_image,
                        content_texts=content_lines,
                        output_path=output_path,
                        alignment=alignment,
                        line_spacing=30,
                        paragraph_width=800
                    )
                    print(f"  - 已生成 {alignment_name} 对齐段落: {output_path}")
                except Exception as e:
                    print(f"  - 生成 {alignment_name} 对齐段落时出错: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    try:
        # 创建简单数据集
        dataset_dir = create_simple_dataset()
        
        # 生成段落
        generate_paragraphs(dataset_dir)
        
        print("测试完成！")
    except Exception as e:
        print(f"程序出现错误: {e}")
        import traceback
        traceback.print_exc() 