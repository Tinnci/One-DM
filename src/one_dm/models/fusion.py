import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from src.one_dm.models.transformer import *
from einops import rearrange, repeat
import math
from src.one_dm.models.resnet_dilation import resnet18 as resnet18_dilation
from src.one_dm.utils.device_utils import move_model_to_device, verify_model_on_device

### merge the handwriting style and printed content
class Mix_TR(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        
        # 保存d_model参数
        self.d_model = d_model
        
        # 位置编码
        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model)
        self.add_position1D = PositionalEncoding1D(d_model=d_model)
        
        # 特征降维
        self.style_dim_reduction = nn.Conv2d(512, d_model, kernel_size=1)
        self.content_dim_reduction = nn.Linear(512, d_model)  # 修改为从512降到32
        
        # Transformer编码器和解码器层
        encoder_layer = TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128, dropout=0.1)
        encoder_norm = nn.LayerNorm(d_model)
        self.style_encoder = TransformerEncoder(encoder_layer, num_layers=2, norm=encoder_norm)
        
        fre_encoder_layer = TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128, dropout=0.1)
        fre_encoder_norm = nn.LayerNorm(d_model)
        self.fre_encoder = TransformerEncoder(fre_encoder_layer, num_layers=2, norm=fre_encoder_norm)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead=4, dim_feedforward=128, dropout=0.1)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=2, norm=decoder_norm)
        
        fre_decoder_layer = TransformerDecoderLayer(d_model, nhead=4, dim_feedforward=128, dropout=0.1)
        fre_decoder_norm = nn.LayerNorm(d_model)
        self.fre_decoder = TransformerDecoder(fre_decoder_layer, num_layers=2, norm=fre_decoder_norm)
        
        # MLP层保持512维度的输入输出
        self.high_pro_mlp = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 512))
        self.low_pro_mlp = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 512))
        self.low_feature_filter = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

        # NCE投影层，将512维降到256维
        self.nce_projection = nn.Linear(512, d_model)

        self._reset_parameters()

        ### low frequency style encoder
        self.Feat_Encoder = self.initialize_resnet18()
        self.style_dilation_layer = resnet18_dilation().conv5_x
        
        ### hig frequency style encoder
        self.freq_encoder = self.initialize_resnet18()
        self.freq_dilation_layer = resnet18_dilation().conv5_x

        ### content encoder
        self.content_encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2]))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def initialize_resnet18(self,):
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.layer4 = nn.Identity()
        resnet.fc = nn.Identity()
        resnet.avgpool = nn.Identity()
        return resnet

    def process_style_feature(self, encoder, dilation_layer, style, add_position2D, style_encoder):
        style = encoder(style)
        style = rearrange(style, 'n (c h w) ->n c h w', c=256, h=4).contiguous()
        style = dilation_layer(style)  # 输出 512 通道
        style_emb = style  # 保存512维的特征用于MLP
        style = self.style_dim_reduction(style)  # 降维到 256 用于transformer
        style = add_position2D(style)
        # 修改维度排列: (n, c, h, w) -> (h*w, n, c)
        style = rearrange(style, 'n c h w -> (h w) n c').contiguous()
        style = style_encoder(style)
        return style, style_emb

    def get_low_style_feature(self, style):
        return self.process_style_feature(self.Feat_Encoder, self.style_dilation_layer, style, self.add_position2D, self.style_encoder)

    def get_high_style_feature(self, laplace):
        return self.process_style_feature(self.freq_encoder, self.freq_dilation_layer, laplace, self.add_position2D, self.fre_encoder)

    def forward(self, style, laplace, content):
        # 处理高频特征
        anchor_style = style[:, 0, :, :].clone().unsqueeze(1).contiguous()
        anchor_high = laplace[:, 0, :, :].clone().unsqueeze(1).contiguous()
        anchor_high_feature, anchor_high_emb = self.get_high_style_feature(anchor_high)

        # 使用512维特征进行MLP处理
        B, C, H, W = anchor_high_emb.shape
        anchor_high_nce = self.high_pro_mlp(anchor_high_emb.view(B, C, -1).permute(0, 2, 1).reshape(-1, C))
        anchor_high_nce = self.nce_projection(anchor_high_nce)  # 投影到256维
        anchor_high_nce = torch.mean(anchor_high_nce.view(B, H*W, -1), dim=1)  # (B, 256)

        pos_style = style[:, 1, :, :].clone().unsqueeze(1).contiguous()
        pos_high = laplace[:, 1, :, :].clone().unsqueeze(1).contiguous()
        pos_high_feature, pos_high_emb = self.get_high_style_feature(pos_high)

        pos_high_nce = self.high_pro_mlp(pos_high_emb.view(B, C, -1).permute(0, 2, 1).reshape(-1, C))
        pos_high_nce = self.nce_projection(pos_high_nce)  # 投影到256维
        pos_high_nce = torch.mean(pos_high_nce.view(B, H*W, -1), dim=1)  # (B, 256)

        # 调整维度: (batch_size, num_samples=2, features=256)
        high_nce_emb = torch.stack([anchor_high_nce, pos_high_nce], dim=1)
        high_nce_emb = nn.functional.normalize(high_nce_emb, p=2, dim=-1)

        # 处理低频特征
        anchor_low = anchor_style
        anchor_low_feature, anchor_low_emb = self.get_low_style_feature(anchor_low)
        
        # 正确处理mask的维度
        B, C, H, W = anchor_low_emb.shape
        anchor_mask = self.low_feature_filter(anchor_low_emb.view(B, C, -1).permute(0, 2, 1).reshape(-1, C))
        anchor_mask = anchor_mask.view(H*W, B, 1)  # 调整为 (H*W, B, 1) 以匹配feature维度
        # anchor_low_feature已经是 (H*W, B, C) 形状
        anchor_low_feature = anchor_low_feature * anchor_mask
        
        anchor_low_nce = self.low_pro_mlp(anchor_low_emb.view(B, C, -1).permute(0, 2, 1).reshape(-1, C))
        anchor_low_nce = self.nce_projection(anchor_low_nce)  # 投影到256维
        anchor_low_nce = torch.mean(anchor_low_nce.view(B, H*W, -1), dim=1)  # (B, 256)

        pos_low = pos_style
        pos_low_feature, pos_low_emb = self.get_low_style_feature(pos_low)
        
        pos_mask = self.low_feature_filter(pos_low_emb.view(B, C, -1).permute(0, 2, 1).reshape(-1, C))
        pos_mask = pos_mask.view(H*W, B, 1)  # 调整为 (H*W, B, 1) 以匹配feature维度
        # pos_low_feature已经是 (H*W, B, C) 形状
        pos_low_feature = pos_low_feature * pos_mask
        
        pos_low_nce = self.low_pro_mlp(pos_low_emb.view(B, C, -1).permute(0, 2, 1).reshape(-1, C))
        pos_low_nce = self.nce_projection(pos_low_nce)  # 投影到256维
        pos_low_nce = torch.mean(pos_low_nce.view(B, H*W, -1), dim=1)  # (B, 256)

        # 调整维度: (batch_size, num_samples=2, features=256)
        low_nce_emb = torch.stack([anchor_low_nce, pos_low_nce], dim=1)
        low_nce_emb = nn.functional.normalize(low_nce_emb, p=2, dim=-1)

        # 处理内容特征
        B = style.shape[0]
        # 确保content是4D张量，并且有正确的维度
        if isinstance(content, torch.Tensor):
            if content.dim() == 4:  # 已经是4D张量 [B, C, H, W]
                content_h, content_w = content.shape[-2], content.shape[-1]
            elif content.dim() == 3:  # 3D张量 [B, H, W]
                content = content.unsqueeze(1)  # 添加通道维度
                content_h, content_w = content.shape[-2], content.shape[-1]
            else:
                # 处理其他维度情况
                raise ValueError(f"Content tensor must be 3D or 4D, got shape {content.shape}")
            
            content = content.view(-1, 1, content_h, content_w)  # 展平batch和time维度
        else:
            raise TypeError(f"Content must be a tensor, got {type(content)}")
            
        content = self.content_encoder(content)  # 输出512通道
        _, C, H, W = content.shape
        content = content.permute(0, 2, 3, 1).reshape(-1, C)  # 重新组织维度为 (N, C)
        content = self.content_dim_reduction(content)  # 降维到32
        content = content.view(-1, B, self.d_model)  # 恢复维度为 (T, B, d_model)
        content = self.add_position1D(content)

        # Transformer处理
        style_hs = self.decoder(content, anchor_low_feature, tgt_mask=None)
        hs = self.fre_decoder(style_hs[0], anchor_high_feature, tgt_mask=None)

        return hs[0].permute(1, 0, 2).contiguous(), high_nce_emb, low_nce_emb
    
    def generate(self, style, laplace, content):
        # 检查和处理输入维度
        if style is None or laplace is None:
            raise ValueError("Style and laplace inputs must not be None")
            
        # 确保在同一设备上
        device = style.device
        if laplace.device != device:
            laplace = laplace.to(device)
        if isinstance(content, torch.Tensor) and content.device != device:
            content = content.to(device)
            
        # 输入形状检查和预处理
        # 打印详细输入形状信息以帮助诊断
        if hasattr(self, 'shape_check_count'):
            self.shape_check_count += 1
        else:
            self.shape_check_count = 1
            
        # 每10次调用打印一次形状信息，用于调试
        if self.shape_check_count % 10 == 1:
            print(f"Generate调用 #{self.shape_check_count}")
            print(f"  Style形状: {style.shape}, 设备: {style.device}")
            print(f"  Laplace形状: {laplace.shape}, 设备: {laplace.device}")
            if isinstance(content, torch.Tensor):
                print(f"  Content形状: {content.shape}, 设备: {content.device}")
            else:
                print(f"  Content类型: {type(content)}")
            
        # 确保style和laplace至少是3维的 [batch, channels, ...]
        if style.dim() < 3:
            raise ValueError(f"Style tensor must have at least 3 dimensions, got {style.dim()}")
        if laplace.dim() < 3:
            raise ValueError(f"Laplace tensor must have at least 3 dimensions, got {laplace.dim()}")
            
        # 处理4维输入 [batch, seq_len, height, width]
        if style.dim() == 4 and style.shape[1] != 1:
            # 有多个样本时，取第一个
            anchor_style = style[:, 0:1].contiguous()
            anchor_high = laplace[:, 0:1].contiguous()
        else:
            # 处理3维输入 [batch, height, width] 或已经正确的4维输入 [batch, 1, height, width]
            if style.dim() == 3:
                # 增加一个维度使其成为 [batch, 1, height, width]
                anchor_style = style.unsqueeze(1).contiguous()
                anchor_high = laplace.unsqueeze(1).contiguous()
            else:
                # 已经是正确的4维
                anchor_style = style
                anchor_high = laplace
        
        # 获取高频和风格特征
        try:
            anchor_high_feature, anchor_high_emb = self.get_high_style_feature(anchor_high)
        except Exception as e:
            print(f"获取高频特征时出错: {str(e)}")
            raise
        
        # 获取低频和风格特征
        try:
            anchor_low = anchor_style
            anchor_low_feature, anchor_low_emb = self.get_low_style_feature(anchor_low)
            
            # 正确处理mask的维度
            B, C, H, W = anchor_low_emb.shape
            anchor_mask = self.low_feature_filter(anchor_low_emb.view(B, C, -1).permute(0, 2, 1).reshape(-1, C))
            anchor_mask = anchor_mask.view(H*W, B, 1)  # 调整为 (H*W, B, 1) 以匹配feature维度
            anchor_low_feature = anchor_low_feature * anchor_mask
        except Exception as e:
            print(f"获取低频特征时出错: {str(e)}")
            raise

        # 处理内容特征
        try:
            B = style.shape[0]
            # 确保content是4D张量，并且有正确的维度
            if not isinstance(content, torch.Tensor):
                raise TypeError(f"Content must be a tensor, got {type(content)}")
                
            # 处理不同维度的content
            if content.dim() == 4:  # 已经是4D张量 [B, C, H, W]
                content_h, content_w = content.shape[-2], content.shape[-1]
                # 确保通道数为1
                if content.shape[1] != 1:
                    # 如果有多个通道，取第一个通道或者平均所有通道
                    content_tensor = content[:, 0:1].contiguous()
                    print(f"警告: content有{content.shape[1]}个通道，已取第一个通道")
                else:
                    content_tensor = content
            elif content.dim() == 3:  # 3D张量 [B, H, W]
                content_tensor = content.unsqueeze(1)  # 添加通道维度
                content_h, content_w = content_tensor.shape[-2], content_tensor.shape[-1]
            elif content.dim() == 2:  # 2D张量 [H, W]
                # 添加批次和通道维度
                content_tensor = content.unsqueeze(0).unsqueeze(0)
                content_h, content_w = content_tensor.shape[-2], content_tensor.shape[-1]
                
                # 如果批次大小大于1，复制内容
                if B > 1:
                    content_tensor = content_tensor.expand(B, -1, -1, -1)
            else:
                # 处理其他维度情况
                raise ValueError(f"Content tensor must be 2D, 3D or 4D, got shape {content.shape}")
            
            # 确保content_tensor是单通道的
            if content_tensor.shape[1] != 1:
                print(f"警告: 调整content通道数从{content_tensor.shape[1]}到1")
                # 如果有多个通道，取第一个通道
                content_tensor = content_tensor[:, 0:1].contiguous()
                
            # 编码内容特征
            content_encoded = self.content_encoder(content_tensor)  # 输出512通道
            _, C, H, W = content_encoded.shape
            content_flat = content_encoded.permute(0, 2, 3, 1).reshape(-1, C)  # 重新组织维度为 (N, C)
            content_reduced = self.content_dim_reduction(content_flat)  # 降维到32
            content_reshaped = content_reduced.view(-1, B, self.d_model)  # 恢复维度为 (T, B, d_model)
            content_positioned = self.add_position1D(content_reshaped)
        except Exception as e:
            print(f"处理内容特征时出错: {str(e)}")
            print(f"Content信息: 类型={type(content)}")
            if isinstance(content, torch.Tensor):
                print(f"  形状={content.shape}, 设备={content.device}, 类型={content.dtype}")
            raise
        
        # 融合内容和风格特征
        try:
            style_hs = self.decoder(content_positioned, anchor_low_feature, tgt_mask=None)
            hs = self.fre_decoder(style_hs[0], anchor_high_feature, tgt_mask=None)
            return hs[0].permute(1, 0, 2).contiguous()
        except Exception as e:
            print(f"特征融合时出错: {str(e)}")
            raise

    def to(self, device):
        """将模型及其所有子模块移动到指定设备"""
        # 先调用父类的to方法
        super().to(device)
        
        # 处理位置编码
        if hasattr(self, 'add_position2D'):
            self.add_position2D = move_model_to_device(self.add_position2D, device)
        if hasattr(self, 'add_position1D'):
            self.add_position1D = move_model_to_device(self.add_position1D, device)
            
        # 处理特征降维
        if hasattr(self, 'style_dim_reduction'):
            self.style_dim_reduction = move_model_to_device(self.style_dim_reduction, device)
        if hasattr(self, 'content_dim_reduction'):
            self.content_dim_reduction = move_model_to_device(self.content_dim_reduction, device)
            
        # 处理Transformer编码器和解码器
        if hasattr(self, 'style_encoder'):
            self.style_encoder = move_model_to_device(self.style_encoder, device)
        if hasattr(self, 'fre_encoder'):
            self.fre_encoder = move_model_to_device(self.fre_encoder, device)
        if hasattr(self, 'decoder'):
            self.decoder = move_model_to_device(self.decoder, device)
        if hasattr(self, 'fre_decoder'):
            self.fre_decoder = move_model_to_device(self.fre_decoder, device)
            
        # 处理MLP层
        if hasattr(self, 'high_pro_mlp'):
            self.high_pro_mlp = move_model_to_device(self.high_pro_mlp, device)
        if hasattr(self, 'low_pro_mlp'):
            self.low_pro_mlp = move_model_to_device(self.low_pro_mlp, device)
        if hasattr(self, 'low_feature_filter'):
            self.low_feature_filter = move_model_to_device(self.low_feature_filter, device)
            
        # 处理NCE投影层
        if hasattr(self, 'nce_projection'):
            self.nce_projection = move_model_to_device(self.nce_projection, device)
            
        # 处理特征编码器
        if hasattr(self, 'Feat_Encoder'):
            self.Feat_Encoder = move_model_to_device(self.Feat_Encoder, device)
        if hasattr(self, 'style_dilation_layer'):
            self.style_dilation_layer = move_model_to_device(self.style_dilation_layer, device)
        if hasattr(self, 'freq_encoder'):
            self.freq_encoder = move_model_to_device(self.freq_encoder, device)
        if hasattr(self, 'freq_dilation_layer'):
            self.freq_dilation_layer = move_model_to_device(self.freq_dilation_layer, device)
        if hasattr(self, 'content_encoder'):
            self.content_encoder = move_model_to_device(self.content_encoder, device)
            
        # 验证所有参数是否都在正确的设备上
        incorrect_params = verify_model_on_device(self, device)
        if incorrect_params:
            print(f"警告: 在移动Mix_TR到{device}后，仍有{len(incorrect_params)}个参数在错误的设备上")
            
        return self