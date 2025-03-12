import torch
from src.one_dm.trainer.trainer import Trainer
from src.one_dm.models.paragraph_processing import ParagraphConsistencyLoss
import torch.distributed as dist
from tqdm import tqdm
import os
from PIL import Image
import torchvision
import torch.nn.functional as F

class ParagraphTrainer(Trainer):
    """
    扩展训练器以支持段落级手写文本生成训练
    """
    def __init__(self, diffusion, unet, vae, criterion, optimizer, data_loader, 
                logs, valid_data_loader=None, device=None, ocr_model=None, ctc_loss=None):
        """
        初始化段落级训练器
        Args:
            diffusion: 扩散模型
            unet: UNet模型，应该是ParagraphUNetModel类型
            vae: VAE编码器
            criterion: 损失函数字典
            optimizer: 优化器
            data_loader: 数据加载器
            logs: 日志配置
            valid_data_loader: 验证数据加载器
            device: 设备
            ocr_model: OCR模型（用于可读性微调）
            ctc_loss: CTC损失函数（用于可读性微调）
        """
        super().__init__(diffusion, unet, vae, criterion, optimizer, data_loader, 
                         logs, valid_data_loader, device, ocr_model, ctc_loss)
        
        # 添加段落一致性损失
        self.consistency_criterion = ParagraphConsistencyLoss()
        
    def _train_paragraph_iter(self, data, step, pbar):
        """
        段落级训练迭代
        Args:
            data: 包含段落数据的批次
            step: 当前训练步数
            pbar: 进度条
        """
        # 确保模型在训练模式
        self.model.train()
        
        try:
            # 准备输入数据，确保所有数据在正确的设备上
            images = data['line_images'].to(self.device)
            style_ref = data['style'].to(self.device)
            laplace_ref = data['laplace'].to(self.device)
            
            # 确保content是张量且在正确的设备上
            if isinstance(data['content'], torch.Tensor):
                content_ref = data['content'].to(self.device)
            else:
                try:
                    content_ref = torch.tensor(data['content'], device=self.device)
                except Exception as e:
                    print(f"无法将content转换为张量: {str(e)}")
                    content_ref = None
                    
            wid = data['writer_ids'].to(self.device)
            paragraph_features = data['paragraph_features'].to(self.device)
            position_info = data['position_info'].to(self.device)
            
            # 确保VAE在正确的设备上
            self.vae = self.vae.to(self.device)
            
            # 通过VAE编码图像
            images = self.vae.encode(images).latent_dist.sample()
            images = images * 0.18215
            
            # 确保diffusion在正确的设备上
            self.diffusion = self.diffusion.to(self.device)
            
            # 向图像添加噪声
            t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
            x_t, noise = self.diffusion.noise_images(images, t)
            
            # 确保model在正确的设备上
            self.model = self.model.to(self.device)
            
            # 前向传播，注意这里我们需要传递段落特征和位置信息
            predicted_noise, high_nce_emb, low_nce_emb = self.model(
                x_t, t, style_ref, laplace_ref, content_ref, 
                paragraph_features=paragraph_features,
                position_info=position_info,
                tag='train'
            )
            
            # 计算基本损失
            recon_loss = self.recon_criterion(predicted_noise, noise)
            high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
            low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
            
            # 计算段落一致性损失
            # 从输出中提取特征用于一致性损失
            feature_dim = high_nce_emb.shape[-1]
            features = high_nce_emb.reshape(-1, feature_dim)
            
            # 计算段落一致性损失
            consistency_loss = self.consistency_criterion(features, position_info[:, 1:])
            
            # 总损失
            loss = recon_loss + high_nce_loss + low_nce_loss + 0.5 * consistency_loss
            
            # 反向传播和参数更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if dist.get_rank() == 0:
                # 记录损失
                loss_dict = {
                    "reconstruct_loss": recon_loss.item(), 
                    "high_nce_loss": high_nce_loss.item(),
                    "low_nce_loss": low_nce_loss.item(),
                    "consistency_loss": consistency_loss.item()
                }
                self.tb_summary.add_scalars("loss", loss_dict, step)
                self._progress(recon_loss.item(), pbar)
                
        except Exception as e:
            print(f"段落训练迭代中出错: {str(e)}")
        finally:
            # 无论成功与否，都释放内存
            if 'data' in locals():
                del data
            if 'loss' in locals():
                del loss
            torch.cuda.empty_cache()
    
    def train_paragraph(self, n_epochs):
        """
        执行段落级训练
        Args:
            n_epochs: 训练轮数
        """
        total_step = 0
        for epoch in range(n_epochs):
            if dist.get_rank() == 0:
                print(f"Epoch {epoch+1}/{n_epochs}")
                pbar = tqdm(total=len(self.data_loader))
            else:
                pbar = None
                
            for batch_idx, data in enumerate(self.data_loader):
                total_step += 1
                self._train_paragraph_iter(data, total_step, pbar)
                
                # 保存样本和检查点
                if dist.get_rank() == 0 and total_step % 2000 == 0:
                    self._save_paragraph_sample(data, total_step)
                    torch.save(self.model.module.state_dict(), 
                              os.path.join(self.save_model_dir, f'paragraph_unet_{total_step}.pth'))
                    
            if dist.get_rank() == 0:
                pbar.close()
                
        # 保存最终模型
        if dist.get_rank() == 0:
            torch.save(self.model.module.state_dict(), 
                      os.path.join(self.save_model_dir, 'paragraph_unet_final.pth'))
    
    def _save_paragraph_sample(self, data, step):
        """
        保存生成的段落样本
        Args:
            data: 输入数据批次
            step: 当前步数
        """
        self.model.eval()
        
        # 准备输入
        style_ref = data['style'].to(self.device)[:4]  # 仅使用4个样本
        laplace_ref = data['laplace'].to(self.device)[:4]
        content_ref = data['content'].to(self.device)[:4]
        paragraph_features = data['paragraph_features'].to(self.device)[:4]
        position_info = None  # 在推理时不需要位置信息
        
        # 生成样本
        n_samples = 4
        batch_size = style_ref.shape[0]
        image_size = 64  # 假设图像大小为64x64
        
        with torch.no_grad():
            # 生成随机噪声作为起点
            x = torch.randn((batch_size, 4, image_size, image_size)).to(self.device)
            
            # 使用DDIM采样法从噪声生成图像
            samples = self.diffusion.sample(self.model, x, style_ref, laplace_ref, content_ref,
                                           paragraph_features=paragraph_features,
                                           position_info=position_info)
            
            # 通过VAE解码生成的样本
            samples = 1 / 0.18215 * samples
            samples = self.vae.decode(samples).sample
            
            # 保存生成的样本图像
            grid = torchvision.utils.make_grid(samples, nrow=2)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(os.path.join(self.save_sample_dir, f'paragraph_sample_{step}.png'))
            
        self.model.train()
    
    def _finetune_paragraph_iter(self, data, step, pbar):
        """
        段落级微调迭代（包含OCR可读性训练）
        Args:
            data: 包含段落数据的批次
            step: 当前训练步数
            pbar: 进度条
        """
        # 确保模型在训练模式
        self.model.train()
        
        try:
            # 准备输入数据，确保所有数据在正确的设备上
            images = data['line_images'].to(self.device)
            style_ref = data['style'].to(self.device)
            laplace_ref = data['laplace'].to(self.device)
            
            # 确保content是张量且在正确的设备上
            if isinstance(data['content'], torch.Tensor):
                content_ref = data['content'].to(self.device)
            else:
                try:
                    content_ref = torch.tensor(data['content'], device=self.device)
                except Exception as e:
                    print(f"无法将content转换为张量: {str(e)}")
                    content_ref = None
                    
            wid = data['writer_ids'].to(self.device)
            paragraph_features = data['paragraph_features'].to(self.device)
            position_info = data['position_info'].to(self.device)
            texts = data['line_texts']  # 文本内容，用于CTC损失
            
            # 确保所有模型在正确的设备上
            self.model = self.model.to(self.device)
            self.diffusion = self.diffusion.to(self.device)
            self.vae = self.vae.to(self.device)
            if self.ocr_model is not None:
                self.ocr_model = self.ocr_model.to(self.device)
            
            # 生成样本并计算可读性损失
            batch_size = style_ref.shape[0]
            image_size = 64  # 假设图像大小为64x64
            
            # 生成随机噪声作为起点
            x = torch.randn((batch_size, 4, image_size, image_size), device=self.device)
            
            # 使用DDIM采样法生成样本，但保留计算图
            samples = self.diffusion.sample_ddim(
                self.model, x, style_ref, laplace_ref, content_ref,
                paragraph_features=paragraph_features,
                position_info=position_info
            )
            
            # 通过VAE解码生成的样本
            samples = 1 / 0.18215 * samples
            decoded_samples = self.vae.decode(samples).sample
            
            # 计算基本训练的损失
            t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
            x_t, noise = self.diffusion.noise_images(images, t)
            
            predicted_noise, high_nce_emb, low_nce_emb = self.model(
                x_t, t, style_ref, laplace_ref, content_ref,
                paragraph_features=paragraph_features,
                position_info=position_info,
                tag='train'
            )
            
            recon_loss = self.recon_criterion(predicted_noise, noise)
            high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
            low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
            
            # 计算段落一致性损失
            feature_dim = high_nce_emb.shape[-1]
            features = high_nce_emb.reshape(-1, feature_dim)
            consistency_loss = self.consistency_criterion(features, position_info[:, 1:])
            
            # 初始化CTC损失
            ctc_loss = torch.tensor(0.0, device=self.device)
            
            # 计算OCR可读性损失（CTC损失）
            if self.ocr_model is not None and self.ctc_criterion is not None:
                try:
                    # 调整生成的图像以适合OCR模型
                    ocr_input = F.interpolate(decoded_samples, size=(32, 128))
                    ocr_input = ocr_input.repeat(1, 3, 1, 1)  # 如果OCR需要RGB输入
                    
                    # 通过OCR模型获取预测
                    ocr_pred = self.ocr_model(ocr_input)
                    
                    # 准备CTC损失的目标
                    target_lengths = []
                    targets = []
                    for text in texts:
                        targets.extend([ord(c) - ord('a') + 1 for c in text.lower() if c.isalpha()])
                        target_lengths.append(len(text))
                    
                    targets = torch.tensor(targets, device=self.device)
                    target_lengths = torch.tensor(target_lengths, device=self.device)
                    input_lengths = torch.full((ocr_pred.size(1),), ocr_pred.size(0), device=self.device)
                    
                    # 计算CTC损失
                    ctc_loss = self.ctc_criterion(
                        ocr_pred.log_softmax(2).permute(1, 0, 2),
                        targets,
                        input_lengths,
                        target_lengths
                    )
                except Exception as e:
                    print(f"计算CTC损失时出错: {str(e)}")
                    ctc_loss = torch.tensor(0.0, device=self.device)
            
            # 总损失 - 微调时加大OCR损失权重
            loss = recon_loss + high_nce_loss + low_nce_loss + 0.5 * consistency_loss + 2.0 * ctc_loss
            
            # 反向传播和参数更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if dist.get_rank() == 0:
                # 记录损失
                loss_dict = {
                    "reconstruct_loss": recon_loss.item(), 
                    "high_nce_loss": high_nce_loss.item(),
                    "low_nce_loss": low_nce_loss.item(),
                    "consistency_loss": consistency_loss.item(),
                    "ctc_loss": ctc_loss.item()
                }
                self.tb_summary.add_scalars("loss", loss_dict, step)
                self._progress(recon_loss.item(), pbar)
                
        except Exception as e:
            print(f"段落微调迭代中出错: {str(e)}")
        finally:
            # 无论成功与否，都释放内存
            if 'data' in locals():
                del data
            if 'loss' in locals():
                del loss
            torch.cuda.empty_cache()
    
    def finetune_paragraph(self, n_epochs):
        """
        执行段落级微调（包含OCR可读性训练）
        Args:
            n_epochs: 训练轮数
        """
        total_step = 0
        for epoch in range(n_epochs):
            if dist.get_rank() == 0:
                print(f"Finetune Epoch {epoch+1}/{n_epochs}")
                pbar = tqdm(total=len(self.data_loader))
            else:
                pbar = None
                
            for batch_idx, data in enumerate(self.data_loader):
                total_step += 1
                self._finetune_paragraph_iter(data, total_step, pbar)
                
                # 保存样本和检查点
                if dist.get_rank() == 0 and total_step % 1000 == 0:
                    self._save_paragraph_sample(data, total_step)
                    torch.save(self.model.module.state_dict(), 
                              os.path.join(self.save_model_dir, f'paragraph_unet_finetune_{total_step}.pth'))
                    
            if dist.get_rank() == 0:
                pbar.close()
                
        # 保存最终模型
        if dist.get_rank() == 0:
            torch.save(self.model.module.state_dict(), 
                      os.path.join(self.save_model_dir, 'paragraph_unet_finetune_final.pth')) 