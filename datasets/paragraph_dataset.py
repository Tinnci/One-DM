import torch
import numpy as np
import os
import json
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from models.paragraph_processing import ParagraphProcessor
import random
from torch.nn.utils.rnn import pad_sequence
import re

class ParagraphDataset(Dataset):
    """段落级手写文本数据集"""
    def __init__(self, data_dir, split='train', transform=None, max_paragraph_length=5):
        """
        参数:
            data_dir: 数据目录
            split: 'train', 'val', 或 'test'
            transform: 图像预处理
            max_paragraph_length: 段落最大行数
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_paragraph_length = max_paragraph_length
        self.paragraph_processor = ParagraphProcessor()
        
        # 加载metadata文件
        meta_path = os.path.join(data_dir, f'{split}_paragraph_meta.json')
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            # 如果没有段落元数据，则创建模拟数据
            print(f"段落元数据文件不存在: {meta_path}，创建模拟数据")
            self.metadata = self._create_mock_data()
        
        # 设置图像预处理转换器
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        paragraph_data = self.metadata[idx]
        paragraph_id = paragraph_data['id']
        writer_id = paragraph_data['writer_id']
        
        # 加载段落行图像
        line_images = []
        line_texts = []
        
        for line_info in paragraph_data['lines'][:self.max_paragraph_length]:
            img_path = os.path.join(self.data_dir, line_info['image_path'])
            text = line_info['text']
            
            try:
                img = Image.open(img_path).convert('L')  # 灰度图像
                if self.transform:
                    img = self.transform(img)
                line_images.append(img)
                line_texts.append(text)
            except Exception as e:
                print(f"加载图像时出错 {img_path}: {e}")
                # 使用空白图像作为替代
                img = torch.zeros((1, 64, 256))
                line_images.append(img)
                line_texts.append("") 
        
        # 提取段落布局特征
        layout_sample = {
            'writer_id': writer_id,
            'lines': [os.path.join(self.data_dir, line['image_path']) 
                     for line in paragraph_data['lines']]
        }
        try:
            paragraph_features = self.paragraph_processor.extract_paragraph_features(layout_sample)
        except Exception as e:
            print(f"提取段落特征时出错: {e}")
            paragraph_features = torch.zeros(12)  # 使用默认特征向量
        
        # 创建位置信息
        position_info = []
        for i, line_info in enumerate(paragraph_data['lines'][:self.max_paragraph_length]):
            # 将文本拆分为单词
            if isinstance(line_info.get('word_positions'), list) and line_info['word_positions']:
                # 使用已有的单词位置
                word_positions = line_info['word_positions']
            else:
                # 简单地根据空格分割位置
                text = line_info['text']
                word_count = len(re.findall(r'\S+', text))
                word_positions = list(range(word_count))
            
            # 为每个单词创建行索引和单词位置
            for pos in word_positions:
                position_info.append([i, pos])  # [行索引, 在行中的位置]
        
        # 确保位置信息是tensor
        if not position_info:
            position_info = torch.zeros((1, 2))
        else:
            position_info = torch.tensor(position_info, dtype=torch.float32)
        
        # 返回样本
        return {
            'paragraph_id': paragraph_id,
            'writer_id': writer_id,
            'line_images': line_images,  # List of tensors
            'line_texts': line_texts,    # List of strings
            'paragraph_features': paragraph_features,  # Tensor of shape [feature_dim]
            'position_info': position_info,  # Tensor of shape [n_words, 2]
        }
    
    def _create_mock_data(self):
        """创建模拟段落数据，仅用于开发测试"""
        mock_data = []
        
        # 假设我们有10个作者，每个作者有5个段落
        for writer_id in range(10):
            for para_id in range(5):
                paragraph = {
                    'id': f"para_{writer_id}_{para_id}",
                    'writer_id': f"writer_{writer_id}",
                    'lines': []
                }
                
                # 每个段落有1-5行
                n_lines = random.randint(1, self.max_paragraph_length)
                for line_id in range(n_lines):
                    line = {
                        'image_path': f"mock_images/writer_{writer_id}/line_{para_id}_{line_id}.png",
                        'text': f"This is line {line_id} of paragraph {para_id} by writer {writer_id}",
                        'word_positions': list(range(8))  # 假设每行8个单词
                    }
                    paragraph['lines'].append(line)
                
                mock_data.append(paragraph)
        
        return mock_data
    
    def collate_fn(self, batch):
        """将批次样本整合成张量"""
        paragraph_ids = [item['paragraph_id'] for item in batch]
        writer_ids = [item['writer_id'] for item in batch]
        
        # 处理可变长度的图像列表
        line_images_batch = []
        line_counts = []
        for item in batch:
            line_images_batch.extend(item['line_images'])
            line_counts.append(len(item['line_images']))
        
        # 将所有行的图像stack成一个张量
        if line_images_batch:
            # 假设所有图像都有相同的大小
            line_images_tensor = torch.stack(line_images_batch)
        else:
            line_images_tensor = torch.zeros((1, 1, 64, 256))
        
        # 处理段落特征
        paragraph_features = torch.stack([item['paragraph_features'] for item in batch])
        
        # 处理位置信息 - 因为每个样本的单词数不同，需要特殊处理
        position_info_list = []
        position_info_lengths = []
        
        # 添加批次偏移量
        offset = 0
        for i, item in batch:
            pos_info = item['position_info']
            position_info_lengths.append(pos_info.shape[0])
            
            # 批次中的样本索引
            batch_indices = torch.full((pos_info.shape[0], 1), i)
            
            # 连接批次索引和原始位置信息
            pos_info_with_batch = torch.cat([batch_indices, pos_info], dim=1)
            position_info_list.append(pos_info_with_batch)
            
            # 更新偏移量
            offset += pos_info.shape[0]
        
        # 将位置信息整合为单个张量
        if position_info_list:
            position_info_tensor = torch.cat(position_info_list, dim=0)
        else:
            position_info_tensor = torch.zeros((1, 3))  # [batch_idx, row_idx, position]
        
        return {
            'paragraph_ids': paragraph_ids,
            'writer_ids': writer_ids,
            'line_images': line_images_tensor,
            'line_counts': torch.tensor(line_counts),
            'paragraph_features': paragraph_features,
            'position_info': position_info_tensor,
            'position_info_lengths': torch.tensor(position_info_lengths)
        }


class ParagraphProcessor:
    """简化版段落处理器，用于创建和处理段落数据集"""
    @staticmethod
    def create_paragraph_dataset(base_dataset, output_dir, n_paragraphs=1000):
        """
        从现有单词或行级数据集创建段落数据集
        
        参数:
            base_dataset: 基础数据集对象，需要提供获取文本行的方法
            output_dir: 输出目录
            n_paragraphs: 要创建的段落数量
        """
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 将现有数据按作者分组
        writer_data = {}
        for item in base_dataset:
            writer_id = item['writer_id']
            if writer_id not in writer_data:
                writer_data[writer_id] = []
            writer_data[writer_id].append(item)
        
        # 创建段落
        paragraph_data = []
        para_id = 0
        
        for writer_id, items in writer_data.items():
            if len(items) < 3:  # 需要至少3个样本构建一个段落
                continue
                
            # 为该作者创建多个段落
            n_writer_paragraphs = min(len(items) // 3, n_paragraphs // len(writer_data))
            for _ in range(n_writer_paragraphs):
                # 随机选择3-5个样本作为段落中的行
                n_lines = random.randint(3, 5)
                selected_items = random.sample(items, min(n_lines, len(items)))
                
                paragraph = {
                    'id': f"para_{para_id}",
                    'writer_id': writer_id,
                    'lines': []
                }
                
                for line_item in selected_items:
                    # 处理行数据
                    line = {
                        'image_path': line_item['image_path'],
                        'text': line_item['text'],
                    }
                    
                    # 如果有单词位置信息，添加它
                    if 'word_positions' in line_item:
                        line['word_positions'] = line_item['word_positions']
                    
                    paragraph['lines'].append(line)
                
                paragraph_data.append(paragraph)
                para_id += 1
        
        # 拆分为训练/验证/测试集
        random.shuffle(paragraph_data)
        n_total = len(paragraph_data)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        
        train_data = paragraph_data[:n_train]
        val_data = paragraph_data[n_train:n_train+n_val]
        test_data = paragraph_data[n_train+n_val:]
        
        # 保存数据集
        with open(os.path.join(output_dir, 'train_paragraph_meta.json'), 'w') as f:
            json.dump(train_data, f)
            
        with open(os.path.join(output_dir, 'val_paragraph_meta.json'), 'w') as f:
            json.dump(val_data, f)
            
        with open(os.path.join(output_dir, 'test_paragraph_meta.json'), 'w') as f:
            json.dump(test_data, f)
        
        print(f"创建了段落数据集: 训练集 {len(train_data)}个段落, 验证集 {len(val_data)}个段落, 测试集 {len(test_data)}个段落")
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        } 