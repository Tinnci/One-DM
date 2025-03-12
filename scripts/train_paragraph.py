import argparse
import os
import torch
import random
import numpy as np
from parse_config import cfg
from torch import distributed as dist
from torch.utils.data import DataLoader
from models.unet import UNetModel
from models.paragraph_unet import ParagraphUNetModel
from diffusers import AutoencoderKL
import torch.optim as optim
from torch import nn
from models.loss import SupConLoss
from models.paragraph_diffusion import ParagraphDiffusion
from datasets.paragraph_dataset import ParagraphDataset
from trainer.paragraph_trainer import ParagraphTrainer
import warnings

warnings.filterwarnings("ignore")

def fix_seeds(seed=10086):
    """固定随机种子以便结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/train/base.yaml", help="yaml file path")
    parser.add_argument("--local_rank", type=int, default=0, help="get rank in distributed training")
    parser.add_argument("--dist", action="store_true", help="use distributed training")
    parser.add_argument("--ep", type=int, default=150, help="total epochs")
    parser.add_argument("--dist_url", type=str, default="env://", help="distributed url")
    parser.add_argument("--world_size", type=int, default=1, help="samples for evaluation")
    parser.add_argument("--stable_dif_path", type=str, default="stabilityai/sd-vae-ft-mse", help="stable diffusion path")
    parser.add_argument("--unet_path", type=str, default="", help="unet checkpoint path")
    parser.add_argument("--data_path", type=str, default="data/paragraph", help="Path to paragraph dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for paragraph training")
    parser.add_argument("--noise_offset", type=float, default=0, help="noise offset")
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune with OCR loss")
    opt = parser.parse_args()
    return opt

def main(opt):
    """加载配置文件到cfg"""
    print("Local rank: {}".format(opt.local_rank))
    cfg.merge_from_file(opt.cfg)
    cfg.freeze()
    
    """设置分布式训练"""
    if opt.dist:
        if opt.local_rank == 0:
            print("Distributed training enabled")
        opt.world_size = torch.cuda.device_count()
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend="nccl", init_method=opt.dist_url, rank=opt.local_rank, world_size=opt.world_size)
    
    """设置随机种子"""
    fix_seeds()
    logs = {}
    logs['tboard'] = 'logs/paragraph/board'
    logs['model'] = 'logs/paragraph/model'
    logs['sample'] = 'logs/paragraph/sample'
    os.makedirs(logs['tboard'], exist_ok=True)
    os.makedirs(logs['model'], exist_ok=True)
    os.makedirs(logs['sample'], exist_ok=True)
    
    """设置设备"""
    device = torch.device("cuda:" + str(opt.local_rank) if torch.cuda.is_available() else "cpu")

    """加载段落数据集"""
    train_dataset = ParagraphDataset(
        data_dir=opt.data_path,
        split='train',
        max_paragraph_length=5
    )
    
    val_dataset = ParagraphDataset(
        data_dir=opt.data_path,
        split='val',
        max_paragraph_length=5
    )
    
    """创建数据加载器"""
    if opt.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    
    """构建模型"""
    # 创建段落级UNet模型
    unet = ParagraphUNetModel(
        in_channels=4,  # 潜在通道数
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=(4, 2, 1),
        dropout=0.1,
        channel_mult=(1, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=True,
        use_fp16=False,
        num_heads=8,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        use_spatial_transformer=True,
        transformer_depth=1, 
        context_dim=768,
        paragraph_features_dim=12,
        enable_paragraph_consistency=True,
    )
    
    """加载预训练权重"""
    if opt.unet_path:
        # 加载基础模型权重
        if opt.local_rank == 0:
            print(f"加载预训练UNet权重: {opt.unet_path}")
        state_dict = torch.load(opt.unet_path, map_location="cpu")
        
        # 过滤掉paragraph_feature_proj、layout_transformer和global_consistency层的权重
        # 因为这些是我们新增的层
        for key in list(state_dict.keys()):
            if any(x in key for x in ['paragraph_feature_proj', 'layout_transformer', 'global_consistency']):
                del state_dict[key]
                
        # 加载过滤后的权重
        missing_keys, unexpected_keys = unet.load_state_dict(state_dict, strict=False)
        if opt.local_rank == 0:
            print(f"缺失键: {missing_keys}")
            print(f"意外键: {unexpected_keys}")
    
    """分布式模型包装"""
    if opt.dist:
        unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)
        unet = unet.to(device)
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[opt.local_rank], output_device=opt.local_rank, 
            broadcast_buffers=False, find_unused_parameters=True
        )
    else:
        unet = unet.to(device)
    
    """构建损失函数和优化器"""
    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.SOLVER.BASE_LR)
    
    """构建扩散模型和VAE编码器"""
    diffusion = ParagraphDiffusion(device=device, noise_offset=opt.noise_offset)
    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device)
    
    """构建OCR模型（用于微调）"""
    ocr_model = None
    ctc_loss = None
    if opt.finetune:
        try:
            from models.recognition import CRNN
            ocr_model = CRNN(32, 1, 37, 256)
            ocr_model = ocr_model.to(device)
            ocr_model.requires_grad_(False)  # 冻结OCR模型
            ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
            if opt.local_rank == 0:
                print("OCR模型加载成功，将进行可读性微调")
        except ImportError:
            print("CRNN模型导入失败，将不使用OCR可读性微调")
    
    """构建训练器"""
    trainer = ParagraphTrainer(
        diffusion, unet, vae, criterion, optimizer, train_loader,
        logs, val_loader, device, ocr_model, ctc_loss
    )
    
    """执行训练"""
    if opt.finetune:
        if opt.local_rank == 0:
            print("开始段落级微调...")
        trainer.finetune_paragraph(opt.ep)
    else:
        if opt.local_rank == 0:
            print("开始段落级训练...")
        trainer.train_paragraph(opt.ep)
    
    """清理分布式进程"""
    if opt.dist:
        dist.destroy_process_group()

if __name__ == "__main__":
    opt = get_args()
    main(opt) 