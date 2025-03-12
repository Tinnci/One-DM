import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.one_dm.models.diffusion import Diffusion
from src.one_dm.models.unet import UNetModel
from src.one_dm.models.transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

class ParagraphDiffusion(Diffusion):
    """
    扩展Diffusion类以支持段落级特征的生成
    """
    def __init__(self, config=None, noise_steps=1000, noise_offset=0, beta_start=1e-4, beta_end=0.02, device=None):
        super().__init__(noise_steps, noise_offset, beta_start, beta_end, device)
        
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
        
        # 其他参数
        self.content_emb_size = config['model']['content_emb_size']
        self.image_size = config['data']['image_size']
        self.channels = config['data']['channels']
    
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
        # 处理元组输入
        if isinstance(x, tuple):
            x, styles, laplace, content = x
            tag = 'test'  # 默认为测试模式
        
        # 如果没有提供时间步，使用随机时间步
        if t is None:
            batch_size = x.shape[0]
            t = self.sample_timesteps(batch_size).to(x.device)
        
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
        model.eval()
        n = x.shape[0]  # 批次大小
        
        total_timesteps, sampling_timesteps = self.noise_steps, sampling_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        x_start = None
        
        for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
            time = (torch.ones(n) * time).long().to(self.device)
            time_next = (torch.ones(n) * time_next).long().to(self.device)
            
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
        model.train()  # 确保模型处于训练模式
        n = x.shape[0]  # 批次大小
        
        total_timesteps, sampling_timesteps = self.noise_steps, sampling_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        x_start = None
        
        for time, time_next in time_pairs:
            time = (torch.ones(n) * time).long().to(self.device)
            time_next = (torch.ones(n) * time_next).long().to(self.device)
            
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
        训练时采样过程，支持段落特征，用于计算损失
        Args:
            model: 扩展的UNet模型
            x: 噪声图像
            styles: 风格参考
            laplace: 拉普拉斯特征
            content: 内容参考
            paragraph_features: 段落特征
            position_info: 位置信息
            total_t: 总时间步
            sampling_timesteps: 采样步数
            eta: 随机性参数
        
        Returns:
            x: 生成的样本
            noise_list[0]: 预测的噪声
            high_nce_emb, low_nce_emb: 风格特征
        """
        total_timesteps, sampling_timesteps = total_t, sampling_timesteps
        times = [-1] + [i/sampling_timesteps for i in range(1, sampling_timesteps + 1)]
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))
        x_start = None
        noise_list = []
        
        for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
            time = (total_timesteps * time).long().to(self.device)
            time_next = (total_timesteps * time_next).long().to(self.device)
            
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