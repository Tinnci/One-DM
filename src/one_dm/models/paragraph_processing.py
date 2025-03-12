import numpy as np
import torch
import torch.nn as nn
import cv2
import json
import os
from PIL import Image
from einops import rearrange

class ParagraphProcessor:
    """段落级特征提取和处理"""
    def __init__(self):
        self.line_height_stats = {}  # 保存每个作者的行高统计信息
        self.spacing_stats = {}      # 保存每个作者的间距统计信息
    
    def extract_paragraph_features(self, paragraph_sample):
        """提取段落级特征"""
        writer_id = paragraph_sample['writer_id']
        line_images = [self._load_image(line_path) 
                      for line_path in paragraph_sample['lines']]
        
        # 计算行高
        line_heights = [img.shape[0] for img in line_images]
        
        # 计算行间距 (如果有多行)
        line_spacings = []
        if len(line_images) > 1:
            for i in range(len(line_images)-1):
                # 使用投影分析计算实际行间距
                projection_i = np.sum(line_images[i], axis=1)
                projection_i_plus_1 = np.sum(line_images[i+1], axis=1)
                # 简化的行间距计算
                line_spacings.append(10)  # 默认值，实际应基于投影计算
        
        # 计算单词间距
        word_spacings = []
        for img in line_images:
            # 使用水平投影分析识别单词间隔
            projection = np.sum(img, axis=0)
            # 简化的间距计算
            word_spacings.append(5)  # 默认值，实际应基于投影计算
        
        # 更新统计信息
        if writer_id not in self.line_height_stats:
            self.line_height_stats[writer_id] = []
        self.line_height_stats[writer_id].extend(line_heights)
        
        # 返回段落特征
        features = {
            'line_heights': line_heights,
            'line_spacings': line_spacings,
            'word_spacings': word_spacings,
            'alignment': self._detect_alignment(line_images),
            'slant_angle': self._calculate_slant(line_images)
        }
        
        # 转换为向量形式
        feature_vector = self._convert_to_feature_vector(features)
        
        return feature_vector
    
    def _load_image(self, path):
        """加载图像并转换为灰度"""
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    def _detect_alignment(self, line_images):
        """检测文本对齐方式：左对齐、居中、右对齐等"""
        # 简化实现：检测行首位置的一致性
        left_positions = []
        for img in line_images:
            # 计算每行文本开始的位置
            projection = np.sum(img, axis=0)
            threshold = np.max(projection) * 0.1
            for i, val in enumerate(projection):
                if val > threshold:
                    left_positions.append(i)
                    break
        
        # 计算左边界位置的标准差
        if len(left_positions) > 1:
            std_dev = np.std(left_positions)
            if std_dev < 5:  # 阈值可调
                return 0  # 左对齐
            else:
                # 检查是否右对齐
                right_positions = []
                for img in line_images:
                    projection = np.sum(img, axis=0)
                    threshold = np.max(projection) * 0.1
                    for i in range(len(projection) - 1, -1, -1):
                        if projection[i] > threshold:
                            right_positions.append(i)
                            break
                
                if np.std(right_positions) < 5:  # 阈值可调
                    return 2  # 右对齐
                else:
                    return 1  # 可能是居中
        
        return 0  # 默认左对齐
    
    def _calculate_slant(self, line_images):
        """计算文本倾斜角度"""
        # 简化实现：使用Hough变换检测主要线条角度
        angles = []
        for img in line_images:
            # 边缘检测
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            
            # 使用霍夫变换检测线条
            try:
                lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
                if lines is not None:
                    for line in lines:
                        rho, theta = line[0]
                        # 将角度转换为度数并限制范围
                        angle = np.degrees(theta) - 90
                        if -45 <= angle <= 45:  # 只考虑合理范围内的角度
                            angles.append(angle)
            except:
                pass  # 忽略可能的错误
        
        # 计算平均倾斜角度
        if angles:
            return np.median(angles)  # 使用中位数减小离群值的影响
        else:
            return 0  # 默认无倾斜
    
    def _convert_to_feature_vector(self, features):
        """将段落特征转换为神经网络可用的特征向量"""
        # 将不同特征编码为固定长度的向量
        vector = []
        
        # 编码行高
        mean_height = np.mean(features['line_heights']) if features['line_heights'] else 0
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

class SpatialLayoutTransformer(nn.Module):
    """处理段落空间布局的Transformer模块"""
    def __init__(self, dim, depth, heads, dim_head):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dim_head),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            ]))
            
    def forward(self, x, layout_features):
        """
        x: 内容特征 [batch, seq_len, dim]
        layout_features: 布局特征 [batch, layout_features_dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # 将布局特征整合到序列中
        layout_tokens = layout_features.unsqueeze(1).expand(-1, 1, dim)  # [batch, 1, dim]
        x_with_layout = torch.cat([layout_tokens, x], dim=1)  # [batch, seq_len+1, dim]
        
        # 应用Transformer层
        for norm1, attn, norm2, ff in self.layers:
            # Self-attention
            x_norm = norm1(x_with_layout)
            x_norm = x_norm.permute(1, 0, 2)  # [seq_len+1, batch, dim]
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            attn_out = attn_out.permute(1, 0, 2)  # [batch, seq_len+1, dim]
            x_with_layout = x_with_layout + attn_out
            
            # Feed-forward
            x_norm = norm2(x_with_layout)
            x_with_layout = x_with_layout + ff(x_norm)
            
        # 分离布局tokens和内容tokens
        layout_enhanced = x_with_layout[:, 1:, :]  # [batch, seq_len, dim]
        
        return layout_enhanced

class GlobalStyleConsistency(nn.Module):
    """确保整个段落的风格一致性"""
    def __init__(self, dim, consistency_tokens=4):
        super().__init__()
        # 创建全局一致性tokens，用于捕获全局风格信息
        self.consistency_tokens = nn.Parameter(torch.randn(1, consistency_tokens, dim))
        self.cross_attention = nn.MultiheadAttention(dim, heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.position_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, dim)
        )
        
    def forward(self, x, position_info):
        """
        x: 特征 [batch, seq_len, dim]
        position_info: 位置信息 [batch, seq_len, 2] (行索引, 在行中的位置)
        """
        batch_size, seq_len, dim = x.shape
        
        # 扩展一致性tokens
        tokens = self.consistency_tokens.expand(batch_size, -1, -1)  # [batch, n_tokens, dim]
        
        # 交叉注意力以捕获全局风格
        tokens_norm = self.norm1(tokens)
        x_norm = self.norm1(x)
        style_context, _ = self.cross_attention(tokens_norm, x_norm, x_norm)
        
        # 计算位置编码
        position_embeddings = self.position_mlp(position_info)  # [batch, seq_len, dim]
        
        # 基于位置计算权重，使相邻位置风格更相似
        weights = self._get_position_weights(position_info)  # [batch, seq_len, seq_len]
        
        # 应用权重融合全局和局部信息
        global_style = style_context.mean(dim=1, keepdim=True)  # [batch, 1, dim]
        
        # 应用位置感知的风格调整
        x_adj = x + position_embeddings + global_style
        
        return self.norm2(x_adj)
        
    def _get_position_weights(self, position_info):
        """基于位置信息计算权重，确保相邻位置风格更相似"""
        # 提取行索引和行内位置
        batch_size, seq_len, _ = position_info.shape
        
        # 计算每对位置之间的相似度
        weights = torch.zeros(batch_size, seq_len, seq_len, device=position_info.device)
        
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    # 计算位置相似度: 同一行且位置接近的元素有更高权重
                    row_i, pos_i = position_info[b, i]
                    row_j, pos_j = position_info[b, j]
                    
                    # 相同行给予高权重
                    if row_i == row_j:
                        # 位置越接近权重越高
                        pos_diff = torch.abs(pos_i - pos_j)
                        weights[b, i, j] = torch.exp(-pos_diff * 5)  # 指数衰减
                    else:
                        # 不同行但行索引接近的也有一定权重
                        row_diff = torch.abs(row_i - row_j)
                        weights[b, i, j] = torch.exp(-row_diff * 2) * 0.3  # 较低权重
        
        # 归一化
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)
        
        return weights

class ParagraphConsistencyLoss(nn.Module):
    """确保段落内风格一致性的损失函数"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, position_info):
        """
        features: 生成内容的特征 [batch*paragraph_length, feature_dim]
        position_info: 位置信息 [batch*paragraph_length, 2]
        """
        # 计算特征相似度矩阵
        features_norm = nn.functional.normalize(features, dim=1)
        sim_matrix = torch.mm(features_norm, features_norm.transpose(0, 1)) / self.temperature
        
        # 创建位置相关性矩阵
        position_sim = self._calculate_position_similarity(position_info)
        
        # 计算损失: 特征相似度应与位置相关性匹配
        loss = nn.functional.mse_loss(
            nn.functional.softmax(sim_matrix, dim=1), 
            position_sim
        )
        
        return loss
        
    def _calculate_position_similarity(self, position_info):
        """计算位置相似性矩阵"""
        n_samples = position_info.shape[0]
        
        # 提取行索引和行内位置
        row_indices = position_info[:, 0].unsqueeze(1)  # [N, 1]
        positions = position_info[:, 1].unsqueeze(1)    # [N, 1]
        
        # 计算行索引差的绝对值
        row_diff = torch.abs(row_indices - row_indices.transpose(0, 1))  # [N, N]
        
        # 计算位置差的绝对值
        pos_diff = torch.abs(positions - positions.transpose(0, 1))      # [N, N]
        
        # 综合相似度计算: 同一行的元素相似度高，相邻行次之，远距离行最低
        row_sim = torch.exp(-row_diff / 0.5)  # 行相似度
        pos_sim = torch.exp(-pos_diff / 0.3)  # 位置相似度
        
        # 同一行的位置相似度权重更高
        same_row = (row_diff == 0).float()
        similarity = 0.7 * same_row * pos_sim + 0.3 * row_sim
        
        # 归一化
        similarity = similarity / (similarity.sum(dim=1, keepdim=True) + 1e-6)
        
        return similarity 