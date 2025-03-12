import os
import numpy as np
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import random
import uuid
import cv2
from tqdm import tqdm
import sys

def generate_style_sample(text, font_path, font_size, color, bg_color, width, height, angle=0):
    """生成单个手写风格样本"""
    # 创建图像
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"无法加载字体: {font_path}，使用默认字体")
        font = ImageFont.load_default()
    
    # 获取文本大小
    try:
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
    except Exception as e:
        print(f"获取文本大小时出错: {e}")
        text_width, text_height = width // 2, height // 2  # 使用默认值
    
    # 计算居中位置
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    # 绘制文本
    try:
        draw.text(position, text, font=font, fill=color)
    except Exception as e:
        print(f"绘制文本时出错: {e}")
        # 使用简单的文本绘制
        draw.text((10, 10), text, fill=color)
    
    # 如果需要旋转
    if angle != 0:
        try:
            img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
            
            # 调整为原始大小
            new_img = Image.new('RGB', (width, height), bg_color)
            new_w, new_h = img.size
            paste_x = (width - new_w) // 2
            paste_y = (height - new_h) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
        except Exception as e:
            print(f"旋转图像时出错: {e}")
    
    # 转为灰度图
    img = img.convert('L')
    
    return img

def create_paragraph_dataset(output_dir, num_writers=10, samples_per_writer=5, 
                           font_paths=None, word_list_path=None):
    """创建英文段落数据集"""
    print(f"创建英文段落数据集到 {output_dir}")
    print(f"设置：作者数量 = {num_writers}, 每个作者样本数量 = {samples_per_writer}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # 默认英文字体列表
    if font_paths is None:
        print("查找系统字体...")
        # 使用常见的手写风格字体
        font_names = [
            'arial.ttf',                  # 基本字体
            'times.ttf',                  # 衬线字体
            'comic.ttf',                  # 漫画风格
            'calibri.ttf',                # 现代字体
            'georgia.ttf',                # 优雅衬线字体
            'verdana.ttf',                # 无衬线字体
            'tahoma.ttf',                 # 清晰字体
            'trebuchet_ms.ttf',           # 网页字体
            'courier.ttf',                # 等宽字体
            'arial.ttf',                  # 备用字体
        ]
        
        # 查找系统字体目录
        system_font_dirs = []
        if os.name == 'nt':  # Windows
            system_font_dirs = [os.path.join(os.environ['WINDIR'], 'Fonts')]
        elif os.name == 'posix':  # Linux/Mac
            system_font_dirs = [
                '/usr/share/fonts',
                '/usr/local/share/fonts',
                os.path.expanduser('~/.fonts')
            ]
        
        print(f"系统字体目录: {system_font_dirs}")
        
        # 找到可用字体
        available_fonts = []
        for font_dir in system_font_dirs:
            if os.path.exists(font_dir):
                for font in font_names:
                    font_path = os.path.join(font_dir, font)
                    if os.path.exists(font_path):
                        available_fonts.append(font_path)
                        print(f"找到字体: {font_path}")
        
        # 如果找不到预定义字体，使用系统中的任何字体
        if not available_fonts:
            print("未找到预定义字体，搜索系统中的任何字体...")
            # 搜索系统中的所有字体文件
            for font_dir in system_font_dirs:
                if os.path.exists(font_dir):
                    for root, _, files in os.walk(font_dir):
                        for file in files:
                            if file.lower().endswith(('.ttf', '.otf')):
                                available_fonts.append(os.path.join(root, file))
                                print(f"找到字体: {os.path.join(root, file)}")
                                if len(available_fonts) >= num_writers * 2:
                                    break  # 找到足够的字体就停止
                        if len(available_fonts) >= num_writers * 2:
                            break
                if len(available_fonts) >= num_writers * 2:
                    break
        
        if not available_fonts:
            print("警告：找不到任何可用字体，使用默认字体")
            available_fonts = ["default"] * num_writers
        
        # 确保有足够的字体
        if len(available_fonts) < num_writers:
            print(f"警告：找到的字体数量({len(available_fonts)})少于请求的作者数量({num_writers})，将重复使用字体")
            # 重复字体使其数量足够
            available_fonts = (available_fonts * ((num_writers // len(available_fonts)) + 1))[:num_writers]
        
        font_paths = available_fonts[:num_writers]
        print(f"使用 {len(font_paths)} 种字体作为作者风格")
    
    # 加载或创建单词列表
    if word_list_path and os.path.exists(word_list_path):
        print(f"从 {word_list_path} 加载单词列表")
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [word.strip() for word in f.readlines()]
    else:
        print("使用默认英文单词列表")
        # 默认英文单词列表
        words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
        ]
    
    # 创建数据集
    train_data = []
    val_data = []
    test_data = []
    
    # 为每个"作者"创建样本
    for writer_idx, font_path in enumerate(tqdm(font_paths[:num_writers], desc="生成作者样本")):
        writer_id = f"writer_{writer_idx}"
        writer_samples = []
        
        # 每个作者的风格参数
        font_size = random.randint(24, 36)
        color = (0, 0, 0)  # 黑色文本
        bg_color = (255, 255, 255)  # 白色背景
        slant_angle = random.uniform(-10, 10)  # 随机倾斜角度
        
        print(f"创建作者 {writer_id} 的样本，使用字体: {font_path}")
        
        # 为每个作者创建多个段落样本
        for sample_idx in range(samples_per_writer):
            paragraph_id = f"para_{writer_id}_{sample_idx}"
            
            # 随机段落长度(行数)
            num_lines = random.randint(3, 7)
            lines = []
            
            # 为段落创建每一行
            for line_idx in range(num_lines):
                # 随机选择单词组成一行
                num_words = random.randint(4, 10)
                line_words = random.sample(words, num_words)
                line_text = " ".join(line_words)
                
                # 生成行图像
                img_width = max(len(line_text) * font_size, 320)  # 根据文本长度调整宽度
                img_height = font_size * 2  # 足够的高度容纳文本
                
                # 为每行添加一些随机变化
                line_font_size = font_size + random.randint(-2, 2)
                line_angle = slant_angle + random.uniform(-2, 2)
                
                # 生成图像
                try:
                    line_img = generate_style_sample(
                        line_text, font_path, line_font_size, color, bg_color, 
                        img_width, img_height, line_angle
                    )
                    
                    # 保存图像
                    img_path = f"images/{paragraph_id}_line_{line_idx}.png"
                    line_img.save(os.path.join(output_dir, img_path))
                    
                    # 创建行信息
                    word_positions = list(range(num_words))
                    line_info = {
                        "image_path": img_path,
                        "text": line_text,
                        "word_positions": word_positions
                    }
                    lines.append(line_info)
                except Exception as e:
                    print(f"生成行图像时出错: {e}")
                    continue  # 跳过出错的行
            
            if lines:  # 只有当有行成功生成时才添加段落
                # 创建段落信息
                paragraph = {
                    "id": paragraph_id,
                    "writer_id": writer_id,
                    "lines": lines
                }
                writer_samples.append(paragraph)
        
        # 分割训练/验证/测试集
        n_samples = len(writer_samples)
        if n_samples == 0:
            print(f"警告: 作者 {writer_id} 没有成功生成样本")
            continue
            
        n_train = max(1, int(n_samples * 0.7))
        n_val = max(1, int(n_samples * 0.15))
        
        train_data.extend(writer_samples[:n_train])
        val_data.extend(writer_samples[n_train:n_train+n_val])
        test_data.extend(writer_samples[n_train+n_val:])
    
    print("保存数据集文件...")
    # 保存数据集文件
    with open(os.path.join(output_dir, 'train_paragraph_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'val_paragraph_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'test_paragraph_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 生成风格参考图像和测试文本文件
    create_test_resources(output_dir, font_paths, words)
    
    print(f"数据集创建完成: 训练集 {len(train_data)}个段落, 验证集 {len(val_data)}个段落, 测试集 {len(test_data)}个段落")
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

def create_test_resources(output_dir, font_paths, words):
    """创建测试所需的风格参考图像和文本文件"""
    print("创建测试资源...")
    # 创建目录
    os.makedirs(os.path.join(output_dir, 'styles'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'content'), exist_ok=True)
    
    # 为每种字体风格创建参考图像
    for i, font_path in enumerate(font_paths[:5]):  # 只使用前5种字体
        # 使用常见单词作为参考
        reference_text = "The quick brown fox jumps over the lazy dog"
        
        # 生成参考图像
        font_size = random.randint(28, 36)
        color = (0, 0, 0)  # 黑色文本
        bg_color = (255, 255, 255)  # 白色背景
        img_width = 400
        img_height = 64
        
        try:
            img = generate_style_sample(
                reference_text, font_path, font_size, color, bg_color, 
                img_width, img_height
            )
            
            # 保存参考图像
            style_path = os.path.join(output_dir, f'styles/style_{i+1}.png')
            img.save(style_path)
            print(f"已创建风格参考图像: {style_path}")
        except Exception as e:
            print(f"创建风格参考图像时出错: {e}")
    
    # 创建测试用的段落文本
    test_paragraphs = [
        # 段落1：短句子
        [
            "This is a sample paragraph.",
            "It contains multiple lines of text.",
            "Each line has a different length.",
            "We can test the alignment and spacing."
        ],
        # 段落2：引述
        [
            "The greatest glory in living lies not in never falling,",
            "but in rising every time we fall.",
            "- Nelson Mandela"
        ],
        # 段落3：技术内容
        [
            "Python is a programming language that lets you work quickly",
            "and integrate systems more effectively.",
            "It supports multiple programming paradigms.",
            "It features a dynamic type system and automatic memory management."
        ],
        # 段落4：长段落
        [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum.",
            "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia."
        ]
    ]
    
    # 保存段落文本
    for i, paragraph in enumerate(test_paragraphs):
        content_path = os.path.join(output_dir, f'content/paragraph_{i+1}.txt')
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(paragraph))
        print(f"已创建内容文本文件: {content_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成英文段落数据集")
    parser.add_argument("--output_dir", type=str, default="data/english_paragraph", 
                       help="输出目录")
    parser.add_argument("--num_writers", type=int, default=10, 
                       help="作者(字体)数量")
    parser.add_argument("--samples_per_writer", type=int, default=5, 
                       help="每个作者的样本数量")
    parser.add_argument("--word_list", type=str, default="", 
                       help="单词列表文件路径，如不指定则使用内置单词列表")
    
    args = parser.parse_args()
    
    try:
        create_paragraph_dataset(
            args.output_dir,
            args.num_writers,
            args.samples_per_writer,
            word_list_path=args.word_list
        )
    except Exception as e:
        print(f"程序出现错误: {e}")
        import traceback
        traceback.print_exc() 