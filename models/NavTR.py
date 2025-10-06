import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp

class PositionalEncoding1D(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        T = x.size(1)
        return x + self.pos_embed[:, :T, :]

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        causal=False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=self.causal,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        nhead,
        mlp_ratio=4.0,
        drop=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Mlp_block=Mlp,
        causal=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mixer = Attention(
            dim, num_heads=nhead, qkv_bias=True, proj_drop=drop, causal=causal
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class NavTR(nn.Module):
    def __init__(self, hidden_size=128, nhead=4, mlp_ratio=2.0, num_layers=2, max_seq_len=50):
        super().__init__()
        self.imu_proj = nn.Linear(6, hidden_size)
        self.pos_embed = PositionalEncoding1D(hidden_size, max_len=max_seq_len)

        self.imu_blocks = nn.ModuleList([
            Block(hidden_size, nhead, mlp_ratio) for _ in range(num_layers)
        ])
        self.gnss_fc = nn.Sequential(nn.Linear(3, hidden_size), nn.ReLU(inplace=True))

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 3)
        )

    def forward(self, imu_seq, gnss_seq):
        imu_feat = self.imu_proj(imu_seq)
        imu_feat = self.pos_embed(imu_feat)     
        for blk in self.imu_blocks:
            imu_feat = blk(imu_feat)
        gnss_feat = self.gnss_fc(gnss_seq)
        fused = torch.cat([imu_feat, gnss_feat], dim=-1)
        out = self.fc(fused)
        return out
