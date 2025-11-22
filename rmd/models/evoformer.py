# Evoformer2D blocks / motif interaction stack

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import RMDConfig
from .embedder import MotifAwareEmbedder


class Evoformer2DBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.row_attn = nn.MultiheadAttention(in_dim, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(in_dim)
        self.col_attn = nn.MultiheadAttention(in_dim, 8, batch_first=True)
        self.norm3 = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, out_dim * 4),
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim),
        )
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, m, d = x.shape
        x_norm = self.norm1(x)
        x_row = x_norm.reshape(b * n, m, d)
        x_row, _ = self.row_attn(x_row, x_row, x_row)
        x = x + x_row.reshape(b, n, m, d)
        x_norm = self.norm2(x)
        x_col = x_norm.transpose(1, 2).reshape(b * m, n, d)
        x_col, _ = self.col_attn(x_col, x_col, x_col)
        x = x + x_col.reshape(b, m, n, d).transpose(1, 2)
        x_norm = self.norm3(x)
        x = self.proj(x) + self.ffn(x_norm)
        return x


class Evoformer2D(nn.Module):
    def __init__(self, config: RMDConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Evoformer2DBlock(config.embed_dim, config.embed_dim) for _ in range(config.num_evoformer_blocks)]
        )
        self.out_proj = nn.Linear(config.embed_dim, 1)

    def forward(self, seq_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, d = seq_embeddings.shape
        x_i = seq_embeddings.unsqueeze(2)
        x_j = seq_embeddings.unsqueeze(1)
        pair = 0.5 * (x_i + x_j)
        for block in self.blocks:
            pair = block(pair, mask=None)
        logits = self.out_proj(pair).squeeze(-1)  # [B, N, N]
        if mask is not None:
            mask2d = (mask[:, :, None] * mask[:, None, :])
            logits = logits * mask2d
        return logits


class MotifInteractionNetwork(nn.Module):
    def __init__(self, config: RMDConfig):
        super().__init__()
        self.embedder = MotifAwareEmbedder(config)
        self.interaction_predictor = Evoformer2D(config)

    def forward(self, seq_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedder(seq_ids, mask)
        binding_logits = self.interaction_predictor(embeddings, mask)
        return embeddings, binding_logits
