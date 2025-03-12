import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import math
import os
from src.one_dm.models.paragraph_processing import ParagraphProcessor
import torchvision.transforms as transforms


class ParagraphGenerator:
    """
    段落生成器: 用于生成格式统一的段落文本
    """
    def __init__(self, diffusion, model, vae, device):
        """
        初始化段落生成器
        Args:
            diffusion: 扩散模型
            model: UNet模型 (应为ParagraphUNetModel类型)
            vae: VAE模型
            device: 设备
        """
        self.diffusion = diffusion
        self.model = model
        self.vae = vae
        self.device = device
        self.paragraph_processor = ParagraphProcessor()
        
        # 创建转换器
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.model.eval()
        self.vae.eval()
    
    def generate_paragraph(self, style_image, content_texts, output_path=None, 
                          alignment=0, line_spacing=30, paragraph_width=800,
                          font_size=None, slant_angle=None):
        """
        生成段落文本
        Args:
            style_image: 风格参考图像路径或PIL图像
            content_texts: 段落文本内容列表，每个元素为一行
            output_path: 输出图像路径，如果为None则返回图像
            alignment: 对齐方式 (0=左对齐, 1=居中, 2=右对齐)
            line_spacing: 行间距
            paragraph_width: 段落宽度
            font_size: 字体大小 (高度)，如果为None则自动检测
            slant_angle: 字体倾斜角度，如果为None则从参考图像中提取
            
        Returns:
            如果output_path不为None，则保存图像并返回路径；否则返回PIL图像
        """
        # 加载风格参考图像
        if isinstance(style_image, str):
            style_img = Image.open(style_image).convert('L')
        else:
            style_img = style_image.convert('L')
        
        # 处理风格参考图像
        style_tensor = self._preprocess_style_image(style_img)
        
        # 计算拉普拉斯特征
        laplace_tensor = self._compute_laplace(style_tensor)
        
        # 提取段落布局特征
        paragraph_features = self._extract_paragraph_features(
            style_img, content_texts, alignment, line_spacing, font_size, slant_angle
        )
        
        # 逐行生成文本
        line_images = []
        for text in content_texts:
            if not text.strip():  # 跳过空行
                continue
                
            # 准备内容输入
            content_tensor = self._prepare_content(text)
            
            # 生成图像
            generated_image = self._generate_text_line(
                style_tensor, laplace_tensor, content_tensor, paragraph_features
            )
            
            line_images.append(generated_image)
        
        # 合并行图像成段落
        paragraph_image = self._merge_lines_to_paragraph(
            line_images, alignment, line_spacing, paragraph_width
        )
        
        # 保存或返回图像
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            paragraph_image.save(output_path)
            return output_path
        else:
            return paragraph_image
    
    def _preprocess_style_image(self, style_img):
        """预处理风格参考图像"""
        # 调整图像大小
        style_img = style_img.resize((128, 128))
        
        # 应用转换
        style_tensor = self.transform(style_img)
        
        # 准备模型输入格式
        style_tensor = style_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        style_tensor = style_tensor.to(self.device)
        
        return style_tensor
    
    def _compute_laplace(self, style_tensor):
        """计算拉普拉斯特征"""
        # 将tensor转换为numpy以应用拉普拉斯
        style_np = style_tensor.squeeze().cpu().numpy()
        
        # 应用拉普拉斯滤波
        laplace = cv2.Laplacian(style_np, cv2.CV_32F)
        
        # 归一化
        laplace = (laplace - laplace.min()) / (laplace.max() - laplace.min() + 1e-6) * 2 - 1
        
        # 转回tensor
        laplace_tensor = torch.from_numpy(laplace).float().unsqueeze(0).unsqueeze(0)
        laplace_tensor = laplace_tensor.to(self.device)
        
        return laplace_tensor
    
    def _extract_paragraph_features(self, style_img, content_texts, alignment, 
                                   line_spacing, font_size, slant_angle):
        """提取段落布局特征"""
        # 创建模拟段落样本
        paragraph_sample = {
            'writer_id': 'test',
            'lines': []
        }
        
        # 模拟行高和倾斜角度
        if font_size is None:
            # 从风格图像中估计行高
            style_np = np.array(style_img)
            projection = np.sum(style_np < 128, axis=1)  # 假设为二值图像
            font_size = self._estimate_font_size(projection)
        
        if slant_angle is None:
            # 从风格图像中估计倾斜角度
            style_np = np.array(style_img)
            slant_angle = self._estimate_slant_angle(style_np)
        
        # 创建特征字典
        features = {
            'line_heights': [font_size] * len(content_texts),
            'line_spacings': [line_spacing] * (len(content_texts) - 1) if len(content_texts) > 1 else [0],
            'word_spacings': [10] * len(content_texts),  # 默认单词间距
            'alignment': alignment,
            'slant_angle': slant_angle
        }
        
        # 转换为特征向量
        feature_vector = self._convert_to_feature_vector(features)
        
        return feature_vector.unsqueeze(0).to(self.device)  # 添加批次维度
    
    def _estimate_font_size(self, projection):
        """从投影中估计字体大小"""
        # 查找连续的非零区域
        regions = []
        current_region = None
        
        for i, val in enumerate(projection):
            if val > 0:
                if current_region is None:
                    current_region = [i, i]
                else:
                    current_region[1] = i
            elif current_region is not None:
                regions.append(current_region)
                current_region = None
                
        if current_region is not None:
            regions.append(current_region)
            
        # 计算区域高度
        heights = [r[1] - r[0] + 1 for r in regions]
        
        # 返回平均高度，或者默认值
        return int(np.mean(heights)) if heights else 32
    
    def _estimate_slant_angle(self, image):
        """从图像中估计文本倾斜角度"""
        # 使用霍夫变换检测文本线条
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        try:
            lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
            if lines is not None:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta) - 90
                    if -45 <= angle <= 45:  # 只考虑合理范围内的角度
                        angles.append(angle)
                        
                return np.median(angles) if angles else 0
        except:
            pass
            
        return 0
    
    def _convert_to_feature_vector(self, features):
        """将特征字典转换为特征向量"""
        vector = []
        
        # 编码行高
        mean_height = np.mean(features['line_heights'])
        std_height = np.std(features['line_heights']) if len(features['line_heights']) > 1 else 0
        vector.extend([mean_height / 100, std_height / 50])  # 归一化
        
        # 编码行间距
        mean_spacing = np.mean(features['line_spacings']) if features['line_spacings'] else 0
        vector.append(mean_spacing / 50)  # 归一化
        
        # 编码单词间距
        mean_word_spacing = np.mean(features['word_spacings']) if features['word_spacings'] else 0
        vector.append(mean_word_spacing / 20)  # 归一化
        
        # 编码对齐方式 (独热编码)
        alignment = features['alignment']
        vector.extend([1 if i == alignment else 0 for i in range(3)])  # 左对齐、居中、右对齐
        
        # 编码倾斜角度
        vector.append(features['slant_angle'] / 45)  # 归一化到[-1, 1]范围
        
        # 填充到固定长度(12)
        while len(vector) < 12:
            vector.append(0)
        
        return torch.tensor(vector, dtype=torch.float32)
    
    def _prepare_content(self, text):
        """准备内容输入"""
        # 模拟内容图像 - 实际实现中应该使用渲染的文本
        content_tensor = torch.zeros((1, 1, 64, 256), device=self.device)
        
        # 在实际实现中，这里应该：
        # 1. 使用字体渲染文本
        # 2. 将渲染的图像转换为tensor
        # 3. 调整大小为模型期望的形状
        
        return content_tensor
    
    def _generate_text_line(self, style_tensor, laplace_tensor, content_tensor, paragraph_features):
        """生成单行文本图像"""
        with torch.no_grad():
            # 生成随机噪声作为起点
            batch_size = style_tensor.shape[0]
            image_size = 64  # 模型预期的图像大小
            x = torch.randn((batch_size, 4, image_size, image_size)).to(self.device)
            
            # 使用扩散模型生成样本
            samples = self.diffusion.sample(
                self.model, x, style_tensor, laplace_tensor, content_tensor,
                paragraph_features=paragraph_features,
                position_info=None  # 推理时不需要位置信息
            )
            
            # 解码生成的样本
            samples = 1 / 0.18215 * samples  # 缩放回VAE范围
            decoded_samples = self.vae.decode(samples).sample
            
            # 转换为PIL图像
            img_tensor = decoded_samples[0].cpu()
            img_tensor = (img_tensor + 1) / 2  # 归一化到[0, 1]
            img_tensor = img_tensor.clamp(0, 1)
            
            # 转换为PIL图像
            img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)
            
            return img_pil
    
    def _merge_lines_to_paragraph(self, line_images, alignment, line_spacing, paragraph_width):
        """将行图像合并为段落"""
        if not line_images:
            return Image.new('RGB', (paragraph_width, 100), color='white')
            
        # 计算段落高度
        total_height = sum([img.height for img in line_images]) + line_spacing * (len(line_images) - 1)
        
        # 创建空白段落图像
        paragraph_img = Image.new('RGB', (paragraph_width, total_height), color='white')
        
        # 逐行粘贴
        y_offset = 0
        for line_img in line_images:
            # 计算水平偏移量
            if alignment == 0:  # 左对齐
                x_offset = 0
            elif alignment == 1:  # 居中
                x_offset = (paragraph_width - line_img.width) // 2
            else:  # 右对齐
                x_offset = paragraph_width - line_img.width
                
            # 粘贴图像
            paragraph_img.paste(line_img, (x_offset, y_offset))
            
            # 更新垂直偏移量
            y_offset += line_img.height + line_spacing
            
        return paragraph_img 