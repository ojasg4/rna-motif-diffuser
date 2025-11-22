# RotaryPositionalEmbedding, FlashMultiHeadAttention, TransformerBlock, MotifAwareEmbedder

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RMDConfig


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]


class FlashMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, use_flash: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_flash:
            attn_mask = None
            if mask is not None:
                attn_mask = mask[:, None, None, :] == 0
                attn_mask = attn_mask.expand(b, self.num_heads, n, n)
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, use_flash: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashMultiHeadAttention(dim, num_heads, dropout, use_flash)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MotifAwareEmbedder(nn.Module):
    def __init__(self, config: RMDConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=4)
        conv_layers = []
        in_ch = config.embed_dim
        for ch in config.cnn_channels:
            conv_layers.append(nn.Conv1d(in_ch, ch, kernel_size=config.cnn_kernel_size, padding=config.cnn_kernel_size // 2))
            conv_layers.append(nn.GELU())
            in_ch = ch
        self.conv = nn.Sequential(*conv_layers)
        self.proj = nn.Linear(in_ch, config.embed_dim)
        self.pos_encoder = RotaryPositionalEmbedding(config.embed_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(config.embed_dim, config.transformer_heads, config.dropout, config.flash_attention)
             for _ in range(config.transformer_layers)]
        )

    def forward(self, seq_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(seq_ids)  # [B, N, D]
        b, n, d = x.shape
        conv_in = x.transpose(1, 2)
        conv_out = self.conv(conv_in).transpose(1, 2)
        x = self.proj(conv_out)
        pos = self.pos_encoder(x)
        x = x + pos[:, :n, :]
        for layer in self.transformer_layers:
            x = layer(x, mask)
        return x
