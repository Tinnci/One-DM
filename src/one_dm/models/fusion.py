import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from src.one_dm.models.transformer import *
from einops import rearrange, repeat
import math
from src.one_dm.models.resnet_dilation import resnet18 as resnet18_dilation

### merge the handwriting style and printed content
class Mix_TR(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 normalize_before=True):
        super(Mix_TR, self).__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        style_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.style_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, style_norm)

        fre_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.fre_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, fre_norm)

        ### fusion the content and style in the transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        
        fre_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.fre_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, fre_decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        
        self.add_position1D = PositionalEncoding(dropout=0.1, dim=d_model)
        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model)
        
        # 维度转换层
        self.style_dim_reduction = nn.Conv2d(512, d_model, kernel_size=1)  # 风格特征降维
        self.content_dim_reduction = nn.Linear(512, d_model)  # 内容特征降维
        
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
        content = rearrange(content, 'n t h w ->(n t) 1 h w').contiguous()
        content = self.content_encoder(content)  # 输出512维
        content = rearrange(content, '(n t) c h w -> t n (c h w)', n=style.shape[0]).contiguous()
        content = self.content_dim_reduction(content)  # 降维到256
        content = self.add_position1D(content)

        # Transformer处理
        style_hs = self.decoder(content, anchor_low_feature, tgt_mask=None)
        hs = self.fre_decoder(style_hs[0], anchor_high_feature, tgt_mask=None)

        return hs[0].permute(1, 0, 2).contiguous(), high_nce_emb, low_nce_emb
    
    def generate(self, style, laplace, content):
        if style.shape[1] == 1:
            anchor_style = style
            anchor_high = laplace
        else:
            anchor_style = style[:, 0, :, :].unsqueeze(1).contiguous()
            anchor_high = laplace[:, 0, :, :].unsqueeze(1).contiguous()
        
        # get the highg frequency and style feature
        anchor_high_feature, anchor_high_emb = self.get_high_style_feature(anchor_high)
        
        # get the low frequency and style feature
        anchor_low = anchor_style
        anchor_low_feature, anchor_low_emb = self.get_low_style_feature(anchor_low)
        
        # 正确处理mask的维度
        B, C, H, W = anchor_low_emb.shape
        anchor_mask = self.low_feature_filter(anchor_low_emb.view(B, C, -1).permute(0, 2, 1).reshape(-1, C))
        anchor_mask = anchor_mask.view(H*W, B, 1)  # 调整为 (H*W, B, 1) 以匹配feature维度
        # anchor_low_feature已经是 (H*W, B, C) 形状
        anchor_low_feature = anchor_low_feature * anchor_mask

        # content encoder
        content = rearrange(content, 'n t h w ->(n t) 1 h w').contiguous()
        content = self.content_encoder(content)
        content = rearrange(content, '(n t) c h w -> t n (c h w)', n=style.shape[0]).contiguous()
        content = self.content_dim_reduction(content)  # 降维到256
        content = self.add_position1D(content)
        
        # fusion of content and style features
        style_hs = self.decoder(content, anchor_low_feature, tgt_mask=None)
        hs = self.fre_decoder(style_hs[0], anchor_high_feature, tgt_mask=None)
        
        return hs[0].permute(1, 0, 2).contiguous()