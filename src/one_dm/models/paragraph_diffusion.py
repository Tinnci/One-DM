import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.one_dm.models.diffusion import Diffusion

class ParagraphDiffusion(Diffusion):
    """
    扩展Diffusion类以支持段落级特征的生成
    """
    def __init__(self, noise_steps=1000, noise_offset=0, beta_start=1e-4, beta_end=0.02, device=None):
        super().__init__(noise_steps, noise_offset, beta_start, beta_end, device)
    
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