# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from Embed import DataEmbedding, DataEmbedding2, TokenEmbedding, get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_with_resolution, get_1d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid_with_resolution
import copy
import random
def modulate(x, shift, scale):
    return x * (1 + scale) + shift

# def Conv1d_with_init(in_channels, out_channels, kernel_size):
#     layer = nn.Conv1d(in_channels, out_channels, kernel_size)
#     nn.init.kaiming_normal_(layer.weight)
#     return layer
class Conv1d_with_init(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None):
        super(Conv1d_with_init, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        nn.init.kaiming_normal_(self.conv.weight)  # Kaiming 初始化
        
        # 可选的激活函数
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x):
        x = x.to(torch.float32)

        x = self.conv(x)
        x = self.activation(x)  # 应用激活函数（如果有的话）
        return x

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)
#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size1, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size1, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size1, num_heads=num_heads, qkv_bias=True, attn_drop=0, proj_drop=0,**block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size1, elementwise_affine=False, eps=1e-6)
        #--------------
        self.time_layer = get_torch_trans(heads=num_heads, layers=1, channels=hidden_size1)
        #------------

        mlp_hidden_dim = int(hidden_size1 * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size1, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size1, 6 * hidden_size1, bias=True)
        )
        self.hide = hidden_size1



    def forward(self, x, c):
        # c = c.type_as(self.adaLN_modulation.weight)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        # x = self.time_layer((x+c).permute(1,0,2)).permute(1,0,2)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.linear = nn.Linear(hidden_size,  out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        args = None,
        input_size=32,
        patch_size=1,
        in_channels=2,
        prompt_channels = 32,
        # emb_channels = 256,
        cond_channels = 256,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if self.learn_sigma else in_channels
        self.out_channels = 1
        # self.patch_size = patch_size
        self.num_heads = num_heads
        self.args = args
        self.hidden_size = hidden_size
        #added_test-------------------------------------------------
        self.Embedding = DataEmbedding(1, self.hidden_size, args=self.args)

        self.Embedding_plus_mask = DataEmbedding2(1, hidden_size, args=self.args)

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, hidden_size)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50, hidden_size)
        )

        #---------------------------------------------------
        self.prompt_channels = prompt_channels
        if self.args.use_cond == True:
            self.cond_channels = cond_channels
        else:
            self.cond_channels = 0

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.input_projection = Conv1d_with_init(2 * hidden_size + self.prompt_channels + self.cond_channels, hidden_size, 1)
        self.input_projection = Conv1d_with_init(2 * (hidden_size + 3 * self.prompt_channels + self.cond_channels), hidden_size, kernel_size=3, activation=nn.ReLU())
        self.cond_projection = Conv1d_with_init(self.cond_channels, self.cond_channels, kernel_size=3, activation=nn.ReLU())
        self.cond_layerNorm = nn.LayerNorm(args.feature_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights_trivial()

        # 初始化prompt
        self.time_length = args.time_length
        self.prompt = nn.Parameter(torch.zeros(1, self.time_length, self.prompt_channels))  # Initialize learnable prompt
        nn.init.trunc_normal_(self.prompt, std=0.02)  # Optional: Initialize with small random values

        # 初始化三个 prompt 生成器 =============================================================================================
        self.has_periodicity_extractor_init = True
        self.periodicity_projection = Conv1d_with_init(in_channels=self.hidden_size, out_channels=self.prompt_channels, kernel_size=3)

        self.has_temporal_attention_init = True
        # 定义时间维度自注意力模块，这里使用TransformerEncoder
        self.encoder_temporal_x = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, dim_feedforward=64, activation="gelu"),
            num_layers=1
        )
        
        # 假设Conv1d_with_init是一个已经定义好的Conv1d层或函数
        self.temporal_attention_projection = Conv1d_with_init(in_channels=self.hidden_size, out_channels=self.prompt_channels)

        self.has_feature_attention_init = True
        # 初始化 Transformer 编码器和解码器
        self.encoder_feature_x = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, dim_feedforward=64, activation="gelu"),
            num_layers=1
        )

        self.feature_attention_projection = Conv1d_with_init(in_channels=self.hidden_size, out_channels=self.prompt_channels)

    def initialize_weights_trivial(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.weekday_embed.weight.data, std=0.02)

        w = self.Embedding.value_embedding.tokenConv.weight.data

        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))



        # # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        # torch.nn.init.normal_(self.mask_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=0.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:  # Check if elementwise_affine is True
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        T = self.args.seq_len
        t = T//self.args.t_patch_size
        sigma_split = 2 if self.learn_sigma else 1

        x = x.reshape(x.shape[0], t, self.args.t_patch_size, sigma_split).permute(0, 3, 1, 2)
        imgs = x.reshape(x.shape[0],sigma_split, T)
        return imgs

    def get_weights_sincos(self, num_t_patch):

        pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_t_patch, self.hidden_size)
        )

        pos_temporal_emb = get_1d_sincos_pos_embed_from_grid(pos_embed_temporal.shape[-1], np.arange(num_t_patch, dtype=np.float32))

        pos_embed_temporal.data.copy_(torch.from_numpy(pos_temporal_emb).float().unsqueeze(0))

        pos_embed_temporal.requires_grad = False

        return pos_embed_temporal, copy.deepcopy(pos_embed_temporal)

    def pos_embed_enc(self, batch, input_size):

        if self.args.pos_emb == 'trivial':
            pos_embed = self.args.pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                self.args.pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        elif self.args.pos_emb == 'SinCos':
            pos_embed_temporal, _ = self.get_weights_sincos(input_size)

        pos_embed = pos_embed_temporal

        pos_embed = pos_embed.expand(batch, -1, -1)


        return pos_embed

    def forward(self, x, cond, mask_origin, t, datatype, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        N, _, T= x.shape

        # TimeEmb = self.Embedding(x, y, is_time=True)
        T = T // self.args.t_patch_size
        input_size = T
        pos_embed_sort = self.pos_embed_enc(N, input_size)
        #####-----------------------------------------------------------------------####

        x_obs = x[:,0].unsqueeze(1) # 掩码真实数据
        x_noise_mask = x[:,1].unsqueeze(1) # 噪声，与真实数据无关

        x_mask_emb, obs_embed, mask_embed = self.Embedding_plus_mask(x_noise_mask, x_obs, mask_origin)

        _, L, C = x_mask_emb.shape
        assert x_mask_emb.shape == pos_embed_sort.shape

        # load prompt
        if self.args.prompt_state == 'loading':
            prompt_path = './pretrained_prompts/prompt_' + datatype + '.pkl'
            self.prompt = torch.load(prompt_path)
            self.prompt.requires_grad = False # 冻结 prompt 的梯度
        # elif self.args.prompt_state == 'test':
        #     prompt_path = './experiments/' + self.args.save_folder + '/model_save/prompt_' + args.datatype + '.pkl'
        #     self.prompt = torch.load(prompt_path)
        #     self.prompt.requires_grad = False # 冻结 prompt 的梯度
            
        # 这里要定义三类prompt ----------------------------------------------------------------------####
        # 第一类是 FFT
        prompt_periodical = self.periodicity_extractor(x_mask_emb.permute(0, 2, 1)) # (B, x_channels, L) -> (B, prompt_channels, L)
            
        # 第二类是时序 attention
        prompt_temporal = self.temporal_attention(x_mask_emb.permute(0, 2, 1), d_model=32) # (B, x_channels, L) -> (B, prompt_channels, L)
        
        # 第三类是特征维 attention
        prompt_feature = self.feature_attention(x_mask_emb.permute(0, 2, 1), d_model=32) # (B, x_channels, L) -> (B, prompt_channels, L)
        
        # Expand prompt
        # prompt_expanded = self.prompt.repeat(N, 1, 1)  # (B, K, L, Cp)
        prompt_expanded = torch.concat((prompt_periodical, prompt_temporal, prompt_feature), dim=1).permute(0, 2, 1)

        # 拼接 prompt
        if self.args.use_cond == True:
            # Expand condition
            # cond = self.cond_layerNorm(cond.permute(0, 2, 1)).permute(0, 2, 1)
            cond_expanded = self.cond_projection(cond).permute(0, 2, 1)

            x_mask_emb = torch.cat([prompt_expanded, x_mask_emb, cond_expanded], dim=-1)
            obs_embed = torch.cat([prompt_expanded, obs_embed, cond_expanded], dim=-1)
            mask_embed = torch.cat([prompt_expanded, mask_embed, cond_expanded], dim=-1) # (B, T, C+Cp)
        else:
            x_mask_emb = torch.cat([prompt_expanded, x_mask_emb], dim=-1)
            obs_embed = torch.cat([prompt_expanded, obs_embed], dim=-1)
            mask_embed = torch.cat([prompt_expanded, mask_embed], dim=-1) # (B, T, C+Cp)

        x_mask_emb_comb = x_mask_emb + obs_embed
        x_mask_emb_comb = x_mask_emb_comb.to(torch.float32)
        mask_embed = mask_embed.to(torch.float32)

        x_mask_emb = self.input_projection(torch.cat((x_mask_emb_comb, mask_embed), dim=-1).permute(0, 2, 1)).permute(0, 2, 1)

        t = self.t_embedder(t)                   # (N, D)
        x_mask_emb = x_mask_emb + pos_embed_sort.to(device = t.device)

        c = t.unsqueeze(1).repeat(1,x_mask_emb.shape[1],1)
        #c = self.layer_norm1(c)
        #####-----------------------------------------------------------------------####
        for block in self.blocks:
            x_mask_emb = block(x_mask_emb, c)                      # (N, T, D)
        # x_mask_emb = x_mask_emb + pos_embed_sort.to(device = t.device)+  t.unsqueeze(1)
        x = self.final_layer(x_mask_emb, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x, mask_origin

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    def periodicity_extractor_init(self, x_channels, device):

        return

    def periodicity_extractor(self, x):

        self.periodicity_projection = self.periodicity_projection.to(x.device)

        X = torch.fft.fft(x, dim=-1)
        _, top_indices = torch.topk(X.real, k=4, dim=-1)

        mask = torch.zeros_like(X, dtype=torch.bool)

        mask.scatter_(-1, top_indices, True)
        X_lpf = X * mask.float()
        x = torch.fft.ifft(X_lpf, dim=-1).real.to(x.dtype)
        output = self.periodicity_projection(x)  # (B, C1,L)

        return output
    
    def temporal_attention_init(self, x_channels, d_model, device, num_encoder_layers=1):
        
        return

    def temporal_attention(self, x, d_model, num_encoder_layers=1):

        device = x.device
        self.encoder_temporal_x = self.encoder_temporal_x.to(device)
        self.temporal_attention_projection = self.temporal_attention_projection.to(device)

        B, x_channels, L = x.shape
        # 对 x 进行编码 (B, C1, L)
        x_encoded = self.encoder_temporal_x(x.permute(2, 0, 1)).permute(1, 2, 0) # (B * x_channels, d_model, L)
        x_encoded = x_encoded.reshape(B, -1, L)
        
        # 将两者在C维度拼接，然后映射到输出维度
        output = self.temporal_attention_projection(x_encoded)  # (B, C_3, L)
        
        return output
    
    def feature_attention_init(self, x_channels, d_model, device, num_encoder_layers=1):
        
        return

    def feature_attention(self, x, d_model, num_encoder_layers=1):

        device = x.device
        self.encoder_feature_x = self.encoder_feature_x.to(device)
        self.feature_attention_projection = self.feature_attention_projection.to(device)
        
        B, x_channels, L = x.shape
        # 对 x 进行编码 (B, C1, L)

        x_encoded = self.encoder_feature_x(x.permute(2, 0, 1)).permute(1, 2, 0) # (B * L, d_model, x_channels)

        # 解码器的输出需要重新映射回原始的 C_3 通道数
        # output = x_encoded.reshape(B, L, d_model, x_channels).permute(0, 2, 3, 1).reshape(B, -1, L) # (B, d_model * x_channels, L)
        output = self.feature_attention_projection(x_encoded)  # 形状 (B, C3, L)

        return output



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(args=None,**kwargs):
    return DiT(args = args,depth=6, hidden_size=256, patch_size=1, num_heads=8,  **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


