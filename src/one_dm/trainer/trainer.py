import torch
from tensorboardX import SummaryWriter
import time
from parse_config import cfg
import os
import sys
from PIL import Image
import torchvision
from tqdm import tqdm
from src.one_dm.data.loader import ContentData
import torch.distributed as dist
import torch.nn.functional as F
from src.one_dm.utils.device_utils import DeviceManager, move_tensors_to_device, content_type_checker

class Trainer:
    def __init__(self, diffusion, unet, vae, criterion, optimizer, data_loader, 
                logs, valid_data_loader=None, device=None, ocr_model=None, ctc_loss=None):
        # 创建设备管理器
        self.device_manager = DeviceManager(device)
        self.device = self.device_manager.device
        
        # 确保所有模型都在同一设备上
        self.model = self.device_manager.prepare_model(unet)
        self.diffusion = self.device_manager.prepare_model(diffusion)
        
        if vae is not None:
            self.vae = self.device_manager.prepare_model(vae)
        else:
            self.vae = None
            
        if ocr_model is not None:
            self.ocr_model = self.device_manager.prepare_model(ocr_model)
        else:
            self.ocr_model = None
            
        # 损失函数和优化器
        self.recon_criterion = criterion['recon']
        self.nce_criterion = criterion['nce']
        self.ctc_criterion = ctc_loss
        self.optimizer = optimizer
        
        # 数据加载器和日志
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']
        
        # 内容数据加载器
        self.content_loader = ContentData()
      
    def _train_iter(self, data, step, pbar):
        # 确保模型在训练模式
        self.model.train()
        
        # 确保所有输入都在同一设备上
        try:
            # 使用设备管理器准备批次数据
            data = self.device_manager.prepare_batch(data)
            
            # 提取数据
            images = data['img']
            style_ref = data['style']
            laplace_ref = data['laplace']
            
            # 处理content数据
            if 'content' in data:
                content_ref = content_type_checker(data['content'])
                if content_ref is None:
                    # 如果无法转换为张量，尝试使用ContentData处理
                    try:
                        if isinstance(data['content'], str):
                            content_ref = self.content_loader.get_content(data['content'], device=self.device)
                        elif isinstance(data['content'], list) and len(data['content']) > 0:
                            if isinstance(data['content'][0], str):
                                # 处理字符串列表
                                content_ref = self.content_loader.get_content(data['content'], device=self.device)
                            else:
                                # 未知列表类型
                                print(f"警告: 无法识别的content列表类型: {type(data['content'][0])}")
                                content_ref = None
                        else:
                            content_ref = None
                    except Exception as e:
                        print(f"处理content时出错: {str(e)}")
                        content_ref = None
            else:
                content_ref = None
                
            wid = data['wid']
            
            # VAE编码
            if self.vae is not None:
                images = self.vae.encode(images).latent_dist.sample()
                images = images * 0.18215
    
            # 前向传播
            t = self.diffusion.sample_timesteps(images.shape[0])
            x_t, noise = self.diffusion.noise_images(images, t)
            
            predicted_noise, high_nce_emb, low_nce_emb = self.model(x_t, t, style_ref, laplace_ref, content_ref, tag='train')
            
            # 计算损失
            recon_loss = self.recon_criterion(predicted_noise, noise)
            high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
            low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
            loss = recon_loss + high_nce_loss + low_nce_loss
    
            # 反向传播和参数更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            if dist.get_rank() == 0:
                # 记录损失
                loss_dict = {
                    "reconstruct_loss": recon_loss.item(), 
                    "high_nce_loss": high_nce_loss.item(),
                    "low_nce_loss": low_nce_loss.item()
                }
                self.tb_summary.add_scalars("loss", loss_dict, step)
                self._progress(recon_loss.item(), pbar)
    
        except Exception as e:
            print(f"训练迭代中出错: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # 无论成功与否，都释放内存
            if 'data' in locals():
                del data
            if 'loss' in locals():
                del loss
            torch.cuda.empty_cache()

    def _finetune_iter(self, data, step, pbar):
        # 确保模型在训练模式
        self.model.train()
        
        try:
            # 使用设备管理器准备批次数据
            data = self.device_manager.prepare_batch(data)
            
            # 提取数据
            images = data['img']
            style_ref = data['style']
            laplace_ref = data['laplace']
            
            # 处理content数据
            if 'content' in data:
                content_ref = content_type_checker(data['content'])
                if content_ref is None:
                    # 尝试使用ContentData处理
                    try:
                        content_ref = self.content_loader.get_content(data['content'], device=self.device)
                    except Exception as e:
                        print(f"处理content时出错: {str(e)}")
                        content_ref = None
            else:
                content_ref = None
                
            wid = data['wid']
            target = data['target']
            target_lengths = data['target_lengths']
            
            # VAE编码
            if self.vae is not None:
                latent_images = self.vae.encode(images).latent_dist.sample()
                latent_images = latent_images * 0.18215
    
            # 前向传播
            t = self.diffusion.sample_timesteps(latent_images.shape[0], finetune=True)
            x_t, noise = self.diffusion.noise_images(latent_images, t)
            
            x_start, predicted_noise, high_nce_emb, low_nce_emb = self.diffusion.train_ddim(
                self.model, x_t, style_ref, laplace_ref, content_ref, t, sampling_timesteps=5
            )
     
            # 计算损失
            recon_loss = self.recon_criterion(predicted_noise, noise)
            
            # 计算OCR损失
            ctc_loss = torch.tensor(0.0, device=self.device)
            if self.ocr_model is not None:
                try:
                    rec_out = self.ocr_model(x_start)
                    input_lengths = torch.IntTensor(x_start.shape[0]*[rec_out.shape[0]]).to(self.device)
                    ctc_loss = self.ctc_criterion(F.log_softmax(rec_out, dim=2), target, input_lengths, target_lengths)
                except Exception as e:
                    print(f"计算OCR损失时出错: {str(e)}")
                
            high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
            low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
            loss = recon_loss + high_nce_loss + low_nce_loss + 0.1*ctc_loss
    
            # 反向传播和参数更新
            self.optimizer.zero_grad()
            loss.backward()
            if cfg.SOLVER.GRAD_L2_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
            self.optimizer.step()
    
            if dist.get_rank() == 0:
                # 记录损失
                loss_dict = {
                    "reconstruct_loss": recon_loss.item(), 
                    "high_nce_loss": high_nce_loss.item(),
                    "low_nce_loss": low_nce_loss.item(), 
                    "ctc_loss": ctc_loss.item()
                }
                self.tb_summary.add_scalars("loss", loss_dict, step)
                self._progress(recon_loss.item(), pbar)
        
        except Exception as e:
            print(f"微调迭代中出错: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # 无论成功与否，都释放内存
            if 'data' in locals():
                del data
            if 'loss' in locals():
                del loss
            torch.cuda.empty_cache()

    def _save_images(self, images, path):
        grid = torchvision.utils.make_grid(images)
        im = torchvision.transforms.ToPILImage()(grid)
        im.save(path)
        return im

    @torch.no_grad()
    def _valid_iter(self, epoch):
        print('loading test dataset, the number is', len(self.valid_data_loader))
        self.model.eval()
        # use the first batch of dataloader in all validations for better visualization comparisons
        test_loader_iter = iter(self.valid_data_loader)
        test_data = next(test_loader_iter)
        # prepare input
        images, style_ref, laplace_ref, content_ref = test_data['img'].to(self.device), \
            test_data['style'].to(self.device), \
            test_data['laplace'].to(self.device), \
            test_data['content'].to(self.device)
    
        load_content = ContentData()
        # forward
        texts = ['getting', 'both', 'success']
        for text in texts:
            rank = dist.get_rank()
            text_ref = load_content.get_content(text)
            text_ref = text_ref.to(self.device).repeat(style_ref.shape[0], 1, 1, 1)
            x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2]//8, (text_ref.shape[1]*32)//8)).to(self.device)
            preds = self.diffusion.ddim_sample(self.model, self.vae, images.shape[0], x, style_ref, laplace_ref, text_ref)
            out_path = os.path.join(self.save_sample_dir, f"epoch-{epoch}-{text}-process-{rank}.png")
            self._save_images(preds, out_path)

    def train(self):
        """start training iterations"""
        for epoch in range(cfg.SOLVER.EPOCHS):
            self.data_loader.sampler.set_epoch(epoch)
            print(f"Epoch:{epoch} of process {dist.get_rank()}")
            dist.barrier()
            if dist.get_rank() == 0:
                pbar = tqdm(self.data_loader, leave=False)
            else:
                pbar = self.data_loader

            for step, data in enumerate(pbar):
                total_step = epoch * len(self.data_loader) + step
                if self.ocr_model is not None:
                    self._finetune_iter(data, total_step, pbar)
                    if (total_step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (total_step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                        if dist.get_rank() == 0:
                            self._save_checkpoint(total_step)
                    else:
                        pass
                    if self.valid_data_loader is not None:
                        if (total_step+1) > cfg.TRAIN.VALIDATE_BEGIN  and (total_step+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                            self._valid_iter(total_step)
                        else:
                            pass 
                else:
                    self._train_iter(data, total_step, pbar)

            if (epoch+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (epoch+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                if dist.get_rank() == 0:
                    self._save_checkpoint(epoch)
                else:
                    pass
            if self.valid_data_loader is not None:
                if (epoch+1) > cfg.TRAIN.VALIDATE_BEGIN  and (epoch+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                    self._valid_iter(epoch)
            else:
                pass

            if dist.get_rank() == 0:
                pbar.close()

    def _progress(self, loss, pbar):
        pbar.set_postfix(mse='%.6f' % (loss))

    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_model_dir, str(epoch)+'-'+"ckpt.pt"))