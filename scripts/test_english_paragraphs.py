import os
import sys
import torch
import argparse
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from create_english_dataset import create_paragraph_dataset
from models.paragraph_unet import ParagraphUNetModel
from diffusers import AutoencoderKL
from models.paragraph_diffusion import ParagraphDiffusion
from models.paragraph_generator import ParagraphGenerator

def setup_models(unet_path, device):
    """
    加载模型
    Args:
        unet_path: UNet模型路径
        device: 设备
    Returns:
        generator: 段落生成器
    """
    print("加载模型...")
    
    # 检查模型路径
    if unet_path and not os.path.exists(unet_path):
        print(f"警告: 模型路径 {unet_path} 不存在，将使用随机初始化模型")
    
    # 创建UNet模型
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
    
    # 加载模型权重
    if unet_path and os.path.exists(unet_path):
        print(f"从 {unet_path} 加载权重...")
        state_dict = torch.load(unet_path, map_location="cpu")
        missing_keys, unexpected_keys = unet.load_state_dict(state_dict, strict=False)
        print(f"缺失键: {len(missing_keys)}, 意外键: {len(unexpected_keys)}")
    
    unet = unet.to(device)
    unet.eval()
    
    # 加载VAE模型
    try:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", subfolder="vae")
        vae.requires_grad_(False)
        vae = vae.to(device)
        vae.eval()
    except Exception as e:
        print(f"加载VAE模型时出错: {e}")
        print("使用本地VAE路径...")
        # 尝试使用本地检查点
        vae = AutoencoderKL.from_pretrained("model_zoo/sd-vae-ft-mse")
        vae.requires_grad_(False)
        vae = vae.to(device)
        vae.eval()
    
    # 创建扩散模型和段落生成器
    diffusion = ParagraphDiffusion(device=device)
    generator = ParagraphGenerator(diffusion, unet, vae, device)
    
    return generator

def test_paragraph_generation(generator, dataset_dir, output_dir):
    """
    测试段落生成
    Args:
        generator: 段落生成器
        dataset_dir: 数据集目录
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查风格图像和内容文件
    styles_dir = os.path.join(dataset_dir, 'styles')
    content_dir = os.path.join(dataset_dir, 'content')
    
    if not os.path.exists(styles_dir) or not os.path.exists(content_dir):
        print(f"错误: 风格目录或内容目录不存在: {styles_dir}, {content_dir}")
        return
    
    # 获取所有风格和内容文件
    style_files = [f for f in os.listdir(styles_dir) if f.endswith(('.png', '.jpg'))]
    content_files = [f for f in os.listdir(content_dir) if f.endswith('.txt')]
    
    if not style_files or not content_files:
        print("错误: 未找到风格图像或内容文本文件")
        return
    
    print(f"找到 {len(style_files)} 个风格图像和 {len(content_files)} 个内容文件")
    
    # 对每个组合进行测试
    for style_file in style_files:
        style_path = os.path.join(styles_dir, style_file)
        style_name = os.path.splitext(style_file)[0]
        
        # 加载风格图像
        style_image = Image.open(style_path).convert('L')
        
        for content_file in content_files:
            content_path = os.path.join(content_dir, content_file)
            content_name = os.path.splitext(content_file)[0]
            
            # 读取内容文本
            with open(content_path, 'r', encoding='utf-8') as f:
                content_lines = f.read().strip().split('\n')
            
            # 设置输出路径
            output_file = f"{style_name}_{content_name}.png"
            output_path = os.path.join(output_dir, output_file)
            
            print(f"生成段落: 风格 {style_name} + 内容 {content_name}")
            
            # 测试所有对齐方式
            for alignment, alignment_name in [(0, "left"), (1, "center"), (2, "right")]:
                alignment_output = os.path.join(output_dir, f"{style_name}_{content_name}_{alignment_name}.png")
                
                # 生成段落
                try:
                    result = generator.generate_paragraph(
                        style_image=style_image,
                        content_texts=content_lines,
                        output_path=alignment_output,
                        alignment=alignment,
                        line_spacing=30,
                        paragraph_width=800
                    )
                    print(f"  - 已生成 {alignment_name} 对齐段落，保存至 {alignment_output}")
                except Exception as e:
                    print(f"  - 生成 {alignment_name} 对齐段落时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="测试英文段落数据集和生成")
    parser.add_argument("--unet_path", type=str, default="", help="UNet模型路径")
    parser.add_argument("--dataset_dir", type=str, default="data/english_paragraph", help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="outputs/english_test", help="输出目录")
    parser.add_argument("--create_dataset", action="store_true", help="是否创建数据集")
    parser.add_argument("--num_writers", type=int, default=5, help="作者数量")
    parser.add_argument("--samples_per_writer", type=int, default=3, help="每个作者的样本数量")
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据集
    if args.create_dataset:
        print(f"创建英文段落数据集，保存至 {args.dataset_dir}")
        create_paragraph_dataset(
            args.dataset_dir,
            args.num_writers,
            args.samples_per_writer
        )
    
    # 设置模型
    generator = setup_models(args.unet_path, device)
    
    # 测试段落生成
    test_paragraph_generation(generator, args.dataset_dir, args.output_dir)

if __name__ == "__main__":
    main() 