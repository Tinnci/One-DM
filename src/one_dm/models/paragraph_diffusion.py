import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.one_dm.models.diffusion import Diffusion
from src.one_dm.models.unet import UNetModel
from src.one_dm.models.transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from src.one_dm.utils.device_utils import move_model_to_device, verify_model_on_device

class ParagraphDiffusion(Diffusion):
    """
    扩展Diffusion类以支持段落级特征的生成
    """
    def __init__(self, config=None, noise_steps=1000, noise_offset=0, beta_start=1e-4, beta_end=0.02, device=None):
        # 从配置中获取参数
        if config is None:
            config = {
                'data': {
                    'image_size': 64,
                    'channels': 3,
                    'batch_size': 2
                },
                'model': {
                    'content_emb_size': 32,
                    'unet': {
                        'in_channels': 3,
                        'model_channels': 32,
                        'out_channels': 3,
                        'num_res_blocks': 1,
                        'attention_resolutions': [1],
                        'dropout': 0.0,
                        'channel_mult': [1, 2],
                        'dims': 2,
                        'use_checkpoint': False
                    },
                    'transformer': {
                        'dim': 32,
                        'depth': 2,
                        'heads': 4,
                        'dim_head': 8
                    }
                }
            }
        
        # 先调用父类的初始化方法
        super().__init__(noise_steps, noise_offset, beta_start, beta_end, device)
        
        # 保存一些配置参数
        self.content_emb_size = config['model']['content_emb_size']
        self.image_size = config['data']['image_size']
        self.channels = config['data']['channels']
        
        # 创建模型组件
        unet_config = config['model']['unet']
        self.unet = UNetModel(
            in_channels=unet_config['in_channels'],
            model_channels=unet_config['model_channels'],
            out_channels=unet_config['out_channels'],
            num_res_blocks=unet_config['num_res_blocks'],
            attention_resolutions=unet_config['attention_resolutions'],
            dropout=unet_config['dropout'],
            channel_mult=unet_config['channel_mult'],
            dims=unet_config['dims'],
            use_checkpoint=unet_config['use_checkpoint'],
            num_heads=unet_config.get('num_heads', 4),
            use_spatial_transformer=unet_config.get('use_spatial_transformer', True),
            transformer_depth=unet_config.get('transformer_depth', 1),
            context_dim=unet_config.get('context_dim', 32),
            num_head_channels=unet_config.get('num_head_channels', -1)
        )
        
        transformer_config = config['model']['transformer']
        d_model = transformer_config['dim']
        nhead = transformer_config['heads']
        dim_feedforward = d_model * 4  # 常见的设置
        dropout = 0.1
        
        # 创建encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_config['depth'],
            norm=encoder_norm
        )
        
        # 创建decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=transformer_config['depth'],
            norm=decoder_norm
        )
        
        # 将模型移动到指定设备
        if device:
            self.to(device)
            
    def to(self, device):
        """将所有模型组件移至同一设备"""
        # 先调用父类的to方法
        super().to(device)
        self.device = device
        
        # 使用move_model_to_device递归地移动所有子模块
        if hasattr(self, 'unet'):
            self.unet = move_model_to_device(self.unet, device)
        if hasattr(self, 'encoder'):
            self.encoder = move_model_to_device(self.encoder, device)
        if hasattr(self, 'decoder'):
            self.decoder = move_model_to_device(self.decoder, device)
            
        # 移动所有tensor属性
        for attr_name in dir(self):
            if attr_name.startswith('__'):
                continue
                
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))
        
        # 验证所有参数是否都在正确的设备上
        incorrect_params = verify_model_on_device(self, device)
        if incorrect_params:
            print(f"警告: 在移动ParagraphDiffusion模型到{device}后，仍有{len(incorrect_params)}个参数在错误的设备上")
        
        return self
    
    def forward(self, x, t=None, styles=None, laplace=None, content=None, 
                paragraph_features=None, position_info=None, tag=None):
        """
        模型的前向传播
        Args:
            x: 输入图像或噪声，或者是包含(x, styles, laplace, content)的元组
            t: 时间步
            styles: 风格参考
            laplace: 拉普拉斯特征
            content: 内容参考
            paragraph_features: 段落特征
            position_info: 位置信息
            tag: 标记（用于训练或推理）
        """
        # 确定当前设备
        device = self.device if self.device is not None else x.device
        
        # 处理元组输入
        if isinstance(x, tuple):
            x, styles, laplace, content = x
            tag = 'test'  # 默认为测试模式
        
        # 确保所有输入都在同一设备上
        x = x.to(device)
        
        # 如果没有提供时间步，使用随机时间步
        if t is None:
            batch_size = x.shape[0]
            t = self.sample_timesteps(batch_size).to(device)
        else:
            t = t.to(device)
        
        # 确保其他输入也在同一设备上
        if styles is not None:
            styles = styles.to(device)
        if laplace is not None:
            laplace = laplace.to(device)
        if content is not None:
            # 检查content类型，确保是张量
            if isinstance(content, torch.Tensor):
                content = content.to(device)
            else:
                try:
                    content = torch.tensor(content, device=device)
                except Exception as e:
                    print(f"无法将content转换为张量: {str(e)}")
                    # 提供错误处理的回退方案
                    content = None
        
        # 处理其他特殊输入
        if paragraph_features is not None:
            paragraph_features = paragraph_features.to(device)
        if position_info is not None:
            position_info = position_info.to(device)
        
        # 使用UNet进行特征提取
        if tag == 'train':
            features, high_nce_emb, low_nce_emb = self.unet(x, timesteps=t, style=styles, laplace=laplace, content=content, tag=tag)
            # 计算MSE损失
            mse_loss = ((features - x) ** 2).mean()
            # 计算NCE损失
            high_nce_loss = (high_nce_emb[:, 0] * high_nce_emb[:, 1]).mean()
            low_nce_loss = (low_nce_emb[:, 0] * low_nce_emb[:, 1]).mean()
            nce_loss = high_nce_loss + low_nce_loss
            # 总损失
            total_loss = mse_loss + 0.1 * nce_loss
            # 确保返回标量
            return total_loss.squeeze()  # 使用squeeze确保返回标量
        else:
            # 优化批处理
            batch_size = x.shape[0]
            if batch_size > 1:
                # 并行处理多个样本
                features = []
                for i in range(0, batch_size, 2):
                    # 每次处理2个样本
                    end_idx = min(i + 2, batch_size)
                    batch_features = self.unet(
                        x[i:end_idx],
                        timesteps=t[i:end_idx] if t is not None else None,
                        style=styles[i:end_idx] if styles is not None else None,
                        laplace=laplace[i:end_idx] if laplace is not None else None,
                        content=content[i:end_idx] if content is not None else None,
                        tag=tag
                    )
                    features.append(batch_features)
                features = torch.cat(features, dim=0)
            else:
                features = self.unet(x, timesteps=t, style=styles, laplace=laplace, content=content, tag=tag)
            return features
    
    @torch.no_grad()
    def sample(self, model, x, styles, laplace, content, paragraph_features=None, position_info=None, 
              sampling_timesteps=50, eta=0):
        """
        使用DDIM采样方法生成样本，支持段落特征
        Args:
            model: 扩展的UNet模型
            x: 输入噪声
            styles: 风格参考
            laplace: 拉普拉斯特征
            content: 内容参考
            paragraph_features: 段落特征
            position_info: 位置信息
            sampling_timesteps: 采样步数
            eta: 随机性参数
        """
        # 确保模型在评估模式
        model.eval()
        
        # 确定设备并确保所有输入在同一设备上
        device = self.device if self.device is not None else x.device
        x = x.to(device)
        
        if styles is not None:
            styles = styles.to(device)
        if laplace is not None:
            laplace = laplace.to(device)
        if content is not None:
            # 确保content是张量
            if isinstance(content, torch.Tensor):
                content = content.to(device)
            else:
                try:
                    content = torch.tensor(content, device=device)
                except:
                    # 如果转换失败，提供备用方案
                    print("警告: 无法将content转换为张量，使用None代替")
                    content = None
        
        if paragraph_features is not None:
            paragraph_features = paragraph_features.to(device)
        if position_info is not None:
            position_info = position_info.to(device)
        
        n = x.shape[0]  # 批次大小
        
        total_timesteps, sampling_timesteps = self.noise_steps, sampling_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        x_start = None
        
        for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
            time = (torch.ones(n) * time).long().to(device)
            time_next = (torch.ones(n) * time_next).long().to(device)
            
            # 前向传播，加入段落特征
            predicted_noise = model(
                x, time, styles, laplace, content,
                paragraph_features=paragraph_features,
                position_info=position_info
            )
            
            beta = self.beta[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None, None]
            
            x_start = (x - (1 - alpha_hat).sqrt() * predicted_noise) / (alpha_hat.sqrt())
            
            if time_next[0] < 0:
                x = x_start
                continue
            
            sigma = eta * (beta * (1 - alpha_hat_next) / (1 - alpha_hat)).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x)
            
            x = x_start * alpha_hat_next.sqrt() + \
                c * predicted_noise + \
                sigma * noise
        
        model.train()
        return x
    
    def sample_ddim(self, model, x, styles, laplace, content, paragraph_features=None, position_info=None,
                   sampling_timesteps=50, eta=0):
        """
        使用DDIM采样方法生成样本，支持段落特征，保留计算图以便计算梯度
        用于段落微调时的可读性训练
        
        Args:
            与sample方法相同，但不使用torch.no_grad()
        """
        # 设置模型为训练模式
        model.train()
        
        # 确定设备并确保所有输入在同一设备上
        device = self.device if self.device is not None else x.device
        x = x.to(device)
        
        if styles is not None:
            styles = styles.to(device)
        if laplace is not None:
            laplace = laplace.to(device)
        if content is not None:
            # 确保content是张量
            if isinstance(content, torch.Tensor):
                content = content.to(device)
            else:
                try:
                    content = torch.tensor(content, device=device)
                except:
                    print("警告: 无法将content转换为张量，使用None代替")
                    content = None
        
        if paragraph_features is not None:
            paragraph_features = paragraph_features.to(device)
        if position_info is not None:
            position_info = position_info.to(device)
        
        n = x.shape[0]  # 批次大小
        
        total_timesteps, sampling_timesteps = self.noise_steps, sampling_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        x_start = None
        
        for time, time_next in time_pairs:
            time = (torch.ones(n) * time).long().to(device)
            time_next = (torch.ones(n) * time_next).long().to(device)
            
            # 前向传播，加入段落特征
            predicted_noise = model(
                x, time, styles, laplace, content,
                paragraph_features=paragraph_features,
                position_info=position_info
            )
            
            beta = self.beta[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None, None]
            
            x_start = (x - (1 - alpha_hat).sqrt() * predicted_noise) / (alpha_hat.sqrt())
            
            if time_next[0] < 0:
                x = x_start
                continue
            
            sigma = eta * (beta * (1 - alpha_hat_next) / (1 - alpha_hat)).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x)
            
            x = x_start * alpha_hat_next.sqrt() + \
                c * predicted_noise + \
                sigma * noise
        
        return x
    
    def train_paragraph(self, model, x, styles, laplace, content, 
                        paragraph_features=None, position_info=None,
                        total_t=1000, sampling_timesteps=50, eta=0):
        """
        段落级训练，使用DDIM进行采样，同时保留计算图以便计算梯度
        Args:
            model: 扩展的UNet模型
            x: 输入噪声
            styles: 风格参考
            laplace: 拉普拉斯特征
            content: 内容参考
            paragraph_features: 段落特征
            position_info: 位置信息
            total_t: 总时间步
            sampling_timesteps: 采样步数
            eta: 随机性参数
        """
        # 确保模型处于训练模式
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        # 确定设备并确保所有输入在同一设备上
        device = self.device if self.device is not None else x.device
        x = x.to(device)
        
        if styles is not None:
            styles = styles.to(device)
        if laplace is not None:
            laplace = laplace.to(device)
        if content is not None:
            # 确保content是张量
            if isinstance(content, torch.Tensor):
                content = content.to(device)
            else:
                try:
                    content = torch.tensor(content, device=device)
                except:
                    print("警告: 无法将content转换为张量，使用None代替")
                    content = None
        
        if paragraph_features is not None:
            paragraph_features = paragraph_features.to(device)
        if position_info is not None:
            position_info = position_info.to(device)
        
        total_timesteps, sampling_timesteps = total_t, sampling_timesteps
        times = torch.linspace(0, 1, steps=sampling_timesteps + 1, device=device)
        times = list(reversed(times.tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        x_start = None
        noise_list = []
        
        for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
            time = (total_timesteps * time).long().to(device)
            time_next = (total_timesteps * time_next).long().to(device)
            
            # 前向传播，加入段落特征
            predicted_noise, high_nce_emb, low_nce_emb = model(
                x, time, styles, laplace, content,
                paragraph_features=paragraph_features,
                position_info=position_info,
                tag='train'
            )
            
            noise_list.append(predicted_noise)
            beta = self.beta[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None, None]
            
            x_start = (x - (1 - alpha_hat).sqrt() * predicted_noise) / (alpha_hat.sqrt())
            if time_next[0] < 0:
                x = x_start
                continue
            
            sigma = eta * (beta * (1 - alpha_hat_next) / (1 - alpha_hat)).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x)
            
            x = x_start * alpha_hat_next.sqrt() + \
                c * predicted_noise + \
                sigma * noise
        
        return x, noise_list[0], high_nce_emb, low_nce_emb 