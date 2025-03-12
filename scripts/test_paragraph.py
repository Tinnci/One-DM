import argparse
import os
import torch
import numpy as np
from PIL import Image
from src.one_dm.models.paragraph_unet import ParagraphUNetModel
from diffusers import AutoencoderKL
from src.one_dm.models.paragraph_diffusion import ParagraphDiffusion
from src.one_dm.models.paragraph_generator import ParagraphGenerator
import glob

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_path", type=str, required=True, help="Path to the paragraph UNet model checkpoint")
    parser.add_argument("--style_image", type=str, required=True, help="Path to the style reference image")
    parser.add_argument("--text_file", type=str, required=True, help="Path to the text file with content to generate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save generated outputs")
    parser.add_argument("--alignment", type=int, default=0, help="Text alignment (0=left, 1=center, 2=right)")
    parser.add_argument("--line_spacing", type=int, default=30, help="Line spacing in pixels")
    parser.add_argument("--paragraph_width", type=int, default=800, help="Width of the paragraph in pixels")
    parser.add_argument("--stable_dif_path", type=str, default="stabilityai/sd-vae-ft-mse", help="stable diffusion path")
    opt = parser.parse_args()
    return opt

def main(opt):
    """设置设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    """创建输出目录"""
    os.makedirs(opt.output_dir, exist_ok=True)
    
    """加载模型"""
    # 创建段落级UNet模型
    print("加载ParagraphUNetModel...")
    unet = ParagraphUNetModel(
        in_channels=4,  # 潜在通道数
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=(4, 2, 1),
        dropout=0.0,  # 推理时不需要dropout
        channel_mult=(1, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,  # 推理时不需要梯度检查点
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
    
    # 加载预训练权重
    print(f"从 {opt.unet_path} 加载权重...")
    state_dict = torch.load(opt.unet_path, map_location="cpu")
    unet.load_state_dict(state_dict, strict=True)
    unet = unet.to(device)
    unet.eval()  # 设置为评估模式
    
    # 加载VAE模型
    print("加载VAE模型...")
    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device)
    vae.eval()  # 设置为评估模式
    
    # 创建扩散模型
    print("创建ParagraphDiffusion模型...")
    diffusion = ParagraphDiffusion(device=device)
    
    # 创建段落生成器
    print("创建ParagraphGenerator...")
    generator = ParagraphGenerator(diffusion, unet, vae, device)
    
    # 读取文本内容
    print(f"从 {opt.text_file} 读取文本内容...")
    with open(opt.text_file, "r", encoding="utf-8") as f:
        content_texts = f.read().strip().split("\n")
    
    # 加载风格参考图像
    print(f"加载风格参考图像: {opt.style_image}...")
    style_image = Image.open(opt.style_image).convert("L")
    
    # 生成手写文本段落
    print("开始生成段落文本...")
    output_path = os.path.join(opt.output_dir, "generated_paragraph.png")
    
    # 提取文件名做为输出的前缀
    style_filename = os.path.splitext(os.path.basename(opt.style_image))[0]
    text_filename = os.path.splitext(os.path.basename(opt.text_file))[0]
    output_path = os.path.join(opt.output_dir, f"{style_filename}_{text_filename}.png")
    
    # 生成段落
    result = generator.generate_paragraph(
        style_image=style_image,
        content_texts=content_texts,
        output_path=output_path,
        alignment=opt.alignment,
        line_spacing=opt.line_spacing,
        paragraph_width=opt.paragraph_width
    )
    
    print(f"段落生成完成，已保存到: {result}")

def batch_generate(style_dir, text_dir, output_dir, **kwargs):
    """批量生成手写文本"""
    # 查找所有风格图像和文本文件
    style_images = glob.glob(os.path.join(style_dir, "*.png")) + glob.glob(os.path.join(style_dir, "*.jpg"))
    text_files = glob.glob(os.path.join(text_dir, "*.txt"))
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个风格图像和文本文件组合生成段落
    for style_image in style_images:
        for text_file in text_files:
            # 设置命令行参数
            args = argparse.Namespace(
                style_image=style_image,
                text_file=text_file,
                output_dir=output_dir,
                **kwargs
            )
            
            # 调用主函数
            main(args)

if __name__ == "__main__":
    opt = get_args()
    main(opt) 