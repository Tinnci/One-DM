import os
import sys
import torch
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.one_dm.models.paragraph_unet import ParagraphUNetModel
from diffusers import AutoencoderKL
from src.one_dm.models.paragraph_diffusion import ParagraphDiffusion
from src.one_dm.models.paragraph_generator import ParagraphGenerator

def main():
    # 设置参数
    unet_path = "logs/paragraph/model/paragraph_unet_final.pth"  # 模型路径
    style_image_path = "examples/styles/style1.jpg"  # 风格参考图像路径
    output_dir = "outputs/examples"  # 输出目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    print("加载模型...")
    
    # 检查文件是否存在
    if not os.path.exists(unet_path):
        print(f"错误：模型文件 {unet_path} 不存在")
        print("请先训练模型或者提供正确的模型路径")
        return
    
    if not os.path.exists(style_image_path):
        print(f"错误：风格图片 {style_image_path} 不存在")
        print("请提供有效的风格参考图像")
        return
    
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
    
    try:
        # 加载权重
        state_dict = torch.load(unet_path, map_location="cpu")
        unet.load_state_dict(state_dict, strict=True)
        unet = unet.to(device)
        unet.eval()
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        return
    
    try:
        # 加载VAE模型
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", subfolder="vae")
        vae.requires_grad_(False)
        vae = vae.to(device)
        vae.eval()
    except Exception as e:
        print(f"加载VAE模型时出错: {e}")
        return
    
    # 创建扩散模型和生成器
    diffusion = ParagraphDiffusion(device=device)
    generator = ParagraphGenerator(diffusion, unet, vae, device)
    
    # 加载风格参考图像
    try:
        style_image = Image.open(style_image_path).convert("L")
    except Exception as e:
        print(f"加载风格图像时出错: {e}")
        return
    
    # 示例1：生成简单段落（左对齐）
    print("生成左对齐段落示例...")
    content_texts_1 = [
        "这是一个左对齐段落的例子。",
        "我们可以使用One-DM模型生成手写文本。",
        "该模型支持多行文本生成。",
        "并且保持整个段落的风格一致性。",
    ]
    
    try:
        generator.generate_paragraph(
            style_image=style_image,
            content_texts=content_texts_1,
            output_path=os.path.join(output_dir, "paragraph_left_aligned.png"),
            alignment=0,  # 左对齐
            line_spacing=30,
            paragraph_width=800
        )
        print("左对齐段落生成成功！")
    except Exception as e:
        print(f"生成左对齐段落时出错: {e}")
    
    # 示例2：生成居中段落
    print("生成居中段落示例...")
    content_texts_2 = [
        "居中对齐的段落",
        "One-DM模型",
        "支持不同的对齐方式",
        "这是居中对齐的效果",
    ]
    
    try:
        generator.generate_paragraph(
            style_image=style_image,
            content_texts=content_texts_2,
            output_path=os.path.join(output_dir, "paragraph_center_aligned.png"),
            alignment=1,  # 居中
            line_spacing=40,
            paragraph_width=800
        )
        print("居中段落生成成功！")
    except Exception as e:
        print(f"生成居中段落时出错: {e}")
    
    # 示例3：生成右对齐段落
    print("生成右对齐段落示例...")
    content_texts_3 = [
        "右对齐段落示例",
        "文本靠右排列",
        "展示不同的排版效果",
        "体验多样化的布局",
    ]
    
    try:
        generator.generate_paragraph(
            style_image=style_image,
            content_texts=content_texts_3,
            output_path=os.path.join(output_dir, "paragraph_right_aligned.png"),
            alignment=2,  # 右对齐
            line_spacing=35,
            paragraph_width=800
        )
        print("右对齐段落生成成功！")
    except Exception as e:
        print(f"生成右对齐段落时出错: {e}")
    
    # 示例4：生成长段落
    print("生成长段落示例...")
    content_texts_4 = [
        "One-DM是一个强大的手写文本生成模型，",
        "它现在支持段落级文本生成功能。",
        "通过这个模型，我们可以生成风格一致的完整段落。",
        "这对于模拟真实手写笔记、创建个性化内容非常有用。",
        "模型可以保持整个段落的风格一致性，包括笔迹粗细、倾斜度等特征。",
        "同时，它还支持调整行间距和对齐方式等布局参数。",
        "这使得生成的文本看起来更加自然和真实。",
        "我们希望这个功能能够帮助到更多的用户！",
    ]
    
    try:
        generator.generate_paragraph(
            style_image=style_image,
            content_texts=content_texts_4,
            output_path=os.path.join(output_dir, "paragraph_long.png"),
            alignment=0,  # 左对齐
            line_spacing=30,
            paragraph_width=800
        )
        print("长段落生成成功！")
    except Exception as e:
        print(f"生成长段落时出错: {e}")
    
    print(f"所有示例已生成完成，请查看 {output_dir} 目录")

if __name__ == "__main__":
    main() 