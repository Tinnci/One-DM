import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by 
    computing a moving average of the parameters, which can help to reduce 
    the noise in the gradients and improve the performance of the model.
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=20000000000000000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class Diffusion(nn.Module):
    def __init__(self, noise_steps=1000, noise_offset=0, beta_start=1e-4, beta_end=0.02, device=None):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_offset = noise_offset
        
        # 先不把beta移动到设备上，在to方法中统一处理
        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.device = device
        
        # 如果提供了设备，则移动到该设备
        if device is not None:
            self.to(device)

    def to(self, device):
        """将所有模型组件移至同一设备"""
        super().to(device)
        self.device = device
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        return self

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def predict_start_from_noise(self, x, t, noise):
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        x_start = (x - (1 - alpha_hat).sqrt()*noise) / (alpha_hat.sqrt())
        return x_start


    def noise_images(self, x, t):
        # 确保输入在正确的设备上
        device = x.device
        x = x.to(device)
        t = t.to(device)
        
        # 确保alpha_hat在正确的设备上
        alpha_hat = self.alpha_hat.to(device)
        
        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
        
        # 在与x相同的设备上创建噪声
        Ɛ = torch.randn_like(x)
        if self.noise_offset > 0:
            noise_offset = torch.randn(x.shape[0], x.shape[1], 1, 1, device=device) * self.noise_offset
            Ɛ = Ɛ + noise_offset
            
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n, finetune=False):
        # 确保张量在正确的设备上创建
        device = self.device if self.device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        if finetune:
            return torch.randint(low=6, high=self.noise_steps, size=(n,), device=device)
        else:
            return torch.randint(low=0, high=self.noise_steps, size=(n,), device=device)
    
    ## output
    def train_ddim(self, model, x, styles, laplace, content, total_t, sampling_timesteps=50, eta=0):
        # 确保所有输入在同一设备上
        device = self.device if self.device is not None else x.device
        x = x.to(device)
        if styles is not None:
            styles = styles.to(device)
        if laplace is not None:
            laplace = laplace.to(device)
        if content is not None:
            if isinstance(content, torch.Tensor):
                content = content.to(device)
            else:
                try:
                    content = torch.tensor(content, device=device)
                except Exception as e:
                    print(f"无法将content转换为张量: {str(e)}")
                    content = None
        
        total_timesteps, sampling_timesteps = total_t, sampling_timesteps
        times = torch.linspace(-1, 1, steps=sampling_timesteps + 1, device=device)
        times = list(reversed(times.tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        x_start = None
        noise_list = []
        
        for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
            time = (total_timesteps * time).long().to(device)
            time_next = (total_timesteps * time_next).long().to(device)
            
            predicted_noise, high_nce_emb, low_nce_emb = model(x, time, styles, laplace, content, tag='train')
            noise_list.append(predicted_noise)
            
            # 确保beta和alpha_hat在正确的设备上
            beta = self.beta.to(device)[time][:, None, None, None]
            alpha_hat = self.alpha_hat.to(device)[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat.to(device)[time_next][:, None, None, None]
            
            x_start = (x - (1 - alpha_hat).sqrt()*predicted_noise) / (alpha_hat.sqrt())
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

    @torch.no_grad()
    def ddim_sample(self, model, vae, n, x, styles, laplace, content, sampling_timesteps=50, eta=0):
        # 确保模型在评估模式
        model.eval()
        
        # 确定设备和确保输入在同一设备上
        device = self.device if self.device is not None else x.device
        x = x.to(device)
        if styles is not None:
            styles = styles.to(device)
        if laplace is not None:
            laplace = laplace.to(device)
        if content is not None:
            if isinstance(content, torch.Tensor):
                content = content.to(device)
            else:
                try:
                    content = torch.tensor(content, device=device)
                except:
                    print("警告: 无法将content转换为张量，使用None代替")
                    content = None

        total_timesteps, sampling_timesteps = self.noise_steps, sampling_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1, device=device)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        x_start = None

        for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
            time = (torch.ones(n, device=device) * time).long()
            time_next = (torch.ones(n, device=device) * time_next).long()
            predicted_noise = model(x, time, styles, laplace, content)

            # 确保beta和alpha_hat在正确的设备上
            beta = self.beta.to(device)[time][:, None, None, None]
            alpha_hat = self.alpha_hat.to(device)[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat.to(device)[time_next][:, None, None, None]
            
            x_start = (x - (1 - alpha_hat).sqrt()*predicted_noise) / (alpha_hat.sqrt())

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
        
        # vae处理部分也需要确保设备一致
        latents = 1 / 0.18215 * x
        try:
            image = vae.decode(latents.to(vae.device)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).contiguous().numpy()
            image = torch.from_numpy(image).to(device)
            x = image.permute(0, 3, 1, 2).contiguous()
        except Exception as e:
            print(f"VAE处理时出错: {str(e)}")
            # 如果VAE处理失败，直接返回最后的x
            
        return x
    
    @torch.no_grad()
    def ddpm_sample(self, model, vae, n, x, styles, laplace, content):
        # 确保模型在评估模式
        model.eval()

        # 确定设备和确保输入在同一设备上
        device = self.device if self.device is not None else x.device
        x = x.to(device)
        if styles is not None:
            styles = styles.to(device)
        if laplace is not None:
            laplace = laplace.to(device)
        if content is not None:
            if isinstance(content, torch.Tensor):
                content = content.to(device)
            else:
                try:
                    content = torch.tensor(content, device=device)
                except:
                    print("警告: 无法将content转换为张量，使用None代替")
                    content = None

        for i in tqdm(reversed(range(0, self.noise_steps)), position=1, leave=False, desc='sampling'):
            time = (torch.ones(n, device=device) * i).long()
            predicted_noise = model(x, time, styles, laplace, content)
            
            # 确保beta和alpha_hat在正确的设备上
            alpha = self.alpha.to(device)[time][:, None, None, None]
            alpha_hat = self.alpha_hat.to(device)[time][:, None, None, None]
            beta = self.beta.to(device)[time][:, None, None, None]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        
        # vae处理部分也需要确保设备一致
        try:
            latents = 1 / 0.18215 * x
            image = vae.decode(latents.to(vae.device)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).contiguous().numpy()
            image = torch.from_numpy(image).to(device)
            x = image.permute(0, 3, 1, 2).contiguous()
        except Exception as e:
            print(f"VAE处理时出错: {str(e)}")
            # 如果VAE处理失败，直接返回最后的x
            
        return x