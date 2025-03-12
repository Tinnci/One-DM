import os
from PIL import Image, ImageDraw, ImageFont
import random

def generate_sample_paragraph(text_lines, font_path, font_size, output_path, alignment=0, line_spacing=30):
    """
    使用PIL库生成一个样本段落图像
    Args:
        text_lines: 文本行列表
        font_path: 字体路径
        font_size: 字体大小
        output_path: 输出路径
        alignment: 对齐方式 (0=左对齐, 1=居中, 2=右对齐)
        line_spacing: 行间距
    """
    # 段落参数
    width = 800
    padding = 50
    text_color = (0, 0, 0)  # 黑色
    bg_color = (255, 255, 255)  # 白色
    
    # 计算段落高度
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        print(f"无法加载字体 {font_path}，使用默认字体")
        font = ImageFont.load_default()
    
    # 计算总高度
    height = padding * 2
    for line in text_lines:
        # 估计行高
        try:
            text_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), line, font=font)
            line_height = text_bbox[3] - text_bbox[1]
        except:
            # 如果无法计算，使用字体大小作为行高
            line_height = font_size + 10
        
        height += line_height + line_spacing
    
    # 最后一行不需要额外的行间距
    height -= line_spacing
    
    # 创建图像
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 绘制文本
    y_offset = padding
    for line in text_lines:
        try:
            # 计算文本大小
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            line_height = text_bbox[3] - text_bbox[1]
            
            # 计算水平位置
            if alignment == 0:  # 左对齐
                x_offset = padding
            elif alignment == 1:  # 居中
                x_offset = (width - text_width) // 2
            else:  # 右对齐
                x_offset = width - text_width - padding
            
            # 绘制文本
            draw.text((x_offset, y_offset), line, font=font, fill=text_color)
            
            # 更新垂直位置
            y_offset += line_height + line_spacing
        except Exception as e:
            print(f"绘制文本行时出错: {e}")
            # 使用简单绘制
            draw.text((padding, y_offset), line, fill=text_color)
            y_offset += font_size + line_spacing
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"已生成样本段落图像: {output_path}")
    
    return output_path

def main():
    # 输出目录
    output_dir = "outputs/sample_paragraphs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试数据
    paragraphs = [
        # 段落1：简单文本
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
        ]
    ]
    
    # 生成不同对齐方式的段落
    alignment_names = ["left", "center", "right"]
    
    # 找到系统字体
    font_path = None
    system_font_paths = []
    
    if os.name == 'nt':  # Windows
        font_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
        for font_name in ['arial.ttf', 'times.ttf', 'calibri.ttf', 'verdana.ttf']:
            font_path = os.path.join(font_dir, font_name)
            if os.path.exists(font_path):
                system_font_paths.append(font_path)
    
    # 如果找不到系统字体，使用默认字体
    if not system_font_paths:
        print("找不到系统字体，使用默认字体")
        system_font_paths = ["default"]
    
    # 为每种字体生成段落样本
    for font_idx, font_path in enumerate(system_font_paths[:2]):  # 只使用前两种字体
        font_name = os.path.basename(font_path).split('.')[0] if font_path != "default" else "default"
        font_size = random.randint(24, 36)
        
        for para_idx, paragraph in enumerate(paragraphs):
            for align_idx, alignment_name in enumerate(alignment_names):
                output_path = os.path.join(output_dir, f"{font_name}_paragraph{para_idx+1}_{alignment_name}.png")
                
                try:
                    generate_sample_paragraph(
                        paragraph,
                        font_path,
                        font_size,
                        output_path,
                        alignment=align_idx,
                        line_spacing=30
                    )
                except Exception as e:
                    print(f"生成段落时出错: {e}")
    
    print("样本段落生成完成，请查看输出目录：", output_dir)

if __name__ == "__main__":
    try:
        main()
        # 将完成标记写入文件
        with open("sample_generation_complete.txt", "w") as f:
            f.write("样本段落生成完成！")
    except Exception as e:
        print(f"程序出现错误: {e}")
        import traceback
        traceback.print_exc()
        # 将错误写入文件
        with open("sample_generation_error.txt", "w") as f:
            f.write(f"程序运行出错: {e}\n")
            import traceback
            traceback.print_exc(file=f) 