import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import repeat
from src.one_dm.models.unet import UNetModel
from src.one_dm.models.paragraph_processing import SpatialLayoutTransformer, GlobalStyleConsistency

class ParagraphUNetModel(UNetModel):
    """
    扩展UNetModel以支持段落级风格特征
    添加段落级布局感知和风格一致性机制
    """
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=True,     # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=768,                  # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=False,
        paragraph_features_dim=12,        # 段落特征向量的维度
        enable_paragraph_consistency=True,# 是否启用段落一致性
    ):
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            n_embed=n_embed,
            legacy=legacy,
        )
        
        # 用于处理段落特征的模块
        self.enable_paragraph_consistency = enable_paragraph_consistency
        
        # 段落特征投影层
        self.paragraph_feature_proj = nn.Sequential(
            nn.Linear(paragraph_features_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, context_dim)
        )
        
        # 空间布局Transformer，用于根据布局调整风格
        self.layout_transformer = SpatialLayoutTransformer(
            dim=context_dim,
            depth=2,
            heads=8,
            dim_head=64
        )
        
        # 全局风格一致性模块
        if enable_paragraph_consistency:
            self.global_consistency = GlobalStyleConsistency(
                dim=context_dim,
                consistency_tokens=4
            )
    
    def forward(self, x, timesteps=None, style=None, laplace=None, content=None, 
                paragraph_features=None, position_info=None, tag='test', **kwargs):
        """
        扩展后的前向传播方法，增加段落特征支持
        Args:
            x: 输入噪声图像
            timesteps: 时间步
            style: 风格参考图像
            laplace: 拉普拉斯特征
            content: 内容参考
            paragraph_features: 段落特征，形状为 [batch_size, paragraph_features_dim]
            position_info: 词的位置信息，形状为 [batch_size, n_words, 2]
            tag: 标志是训练还是测试
        """
        # 确保所有输入在同一设备上
        device = x.device
        
        if timesteps is not None:
            timesteps = timesteps.to(device)
        if style is not None:
            style = style.to(device)
        if laplace is not None:
            laplace = laplace.to(device)
        if content is not None:
            # 确保content是张量
            if isinstance(content, torch.Tensor):
                content = content.to(device)
            else:
                try:
                    content = torch.tensor(content, device=device)
                except Exception as e:
                    print(f"无法将content转换为张量: {str(e)}")
                    content = None
        
        if paragraph_features is not None:
            paragraph_features = paragraph_features.to(device)
        if position_info is not None:
            position_info = position_info.to(device)
        
        hs = []
        t_emb = self.timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        
        # 获取基本的风格内容特征
        if tag == 'train':
            context, high_nce_emb, low_nce_emb = self.mix_net(style, laplace, content)
        else:
            context = self.mix_net.generate(style, laplace, content)
        
        # 处理段落特征
        if paragraph_features is not None:
            # 投影段落特征到上下文空间
            para_context = self.paragraph_feature_proj(paragraph_features)  # [batch_size, context_dim]
            
            # 根据段落布局调整风格
            if position_info is not None:
                # [batch_size, seq_len, context_dim]
                context = self.layout_transformer(context, para_context)
                
                # 应用全局风格一致性
                if self.enable_paragraph_consistency:
                    context = self.global_consistency(context, position_info)
        
        h = x.type(self.dtype)
        
        # 以下代码与原始UNetModel相同
        # INPUT BLOCKS
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        
        # MIDDLE BLOCK
        h = self.middle_block(h, emb, context)
        
        # OUTPUT BLOCKS
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            
        h = h.type(x.dtype)
        
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            if tag == 'train':
                return self.out(h), high_nce_emb, low_nce_emb
            else:
                return self.out(h)
    
    @staticmethod
    def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
        """
        从UNetModel中提取出来的时间步嵌入方法
        """
        if timesteps is None:
            return None
            
        device = timesteps.device
        
        if not repeat_only:
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=device) / half
            )
            args = timesteps[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            embedding = repeat(timesteps, 'b -> b d', d=dim)
        return embedding 