# SE3EquivariantLayer, DiffusionModel

from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RMDConfig
from ..data import CoordinateNormalizer


class SE3EquivariantLayer(nn.Module):
    def __init__(self, dim: int, edge_dim: int = 32, max_neighbors: int = 64,
                 normalizer: Optional[CoordinateNormalizer] = None):
        super().__init__()
        self.dim = dim
        self.max_neighbors = max_neighbors
        self.normalizer = normalizer
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1 + edge_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1, bias=False),
        )
        self.edge_embed = nn.Linear(1, edge_dim)

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = coords.shape
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        dist2 = (diff ** 2).sum(-1, keepdim=True)
        edge_feat = self.edge_embed(dist2)
        feat_i = features[:, :, None, :].expand(b, n, n, self.dim)
        feat_j = features[:, None, :, :].expand(b, n, n, self.dim)
        edge_input = torch.cat([feat_i, feat_j, dist2, edge_feat], dim=-1)
        edge_msg = self.edge_mlp(edge_input)
        if mask is not None:
            mask2d = mask[:, :, None] * mask[:, None, :]
            edge_msg = edge_msg * mask2d.unsqueeze(-1)
        agg_msg = edge_msg.sum(dim=2)
        node_input = torch.cat([features, agg_msg], dim=-1)
        new_features = self.node_mlp(node_input)
        scalar = self.coord_mlp(edge_msg)
        norm = torch.sqrt(dist2 + 1e-8)
        direction = diff / (norm + 1e-8)
        coord_update = (scalar * direction).sum(dim=2)
        if mask is not None:
            coord_update = coord_update * mask.unsqueeze(-1)
        new_coords = coords + coord_update
        return new_coords, new_features


class DiffusionModel(nn.Module):
    def __init__(self, config: RMDConfig, normalizer: Optional[CoordinateNormalizer] = None):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.time_embed = nn.Sequential(
            nn.Linear(config.diffusion_dim, config.diffusion_dim * 4),
            nn.SiLU(),
            nn.Linear(config.diffusion_dim * 4, config.diffusion_dim),
        )
        self.cond_proj = nn.Linear(config.embed_dim, config.diffusion_dim)
        self.layers = nn.ModuleList(
            [SE3EquivariantLayer(config.diffusion_dim, max_neighbors=config.max_neighbors, normalizer=normalizer)
             for _ in range(config.gnn_layers)]
        )
        self.output_proj = nn.Linear(config.diffusion_dim, 3)
        betas = self._get_noise_schedule(config)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_noise_schedule(self, config: RMDConfig) -> torch.Tensor:
        if config.noise_schedule == "linear":
            return torch.linspace(1e-4, 0.02, config.diffusion_steps)
        if config.noise_schedule == "cosine":
            steps = config.diffusion_steps
            s = 0.008
            x = torch.linspace(0, steps, steps + 1)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 1e-4, 0.9999)
        raise ValueError(f"Unknown noise schedule: {config.noise_schedule}")

    def _get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.config.diffusion_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb)

    def forward(
        self,
        noisy_coords: torch.Tensor,
        timesteps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self._get_time_embedding(timesteps).unsqueeze(1)
        cond_emb = self.cond_proj(seq_embeddings)
        features = cond_emb + 0.1 * t_emb
        coords = noisy_coords
        for layer in self.layers:
            coords, features = layer(coords, features, mask)
        noise = self.output_proj(features)
        if self.normalizer is None or not getattr(self.normalizer, "is_fitted", False):
            noise = torch.clamp(noise, -5.0, 5.0)
        else:
            noise = noise * 0.1
            noise = torch.clamp(noise, -0.1, 0.1)
        if torch.isnan(noise).any() or torch.isinf(noise).any():
            noise = torch.zeros_like(noise)
        if mask is not None:
            noise = noise * mask.unsqueeze(-1)
        return noise

    def add_noise(self, coords: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps][:, None, None]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None]
        return sqrt_alpha_prod * coords + sqrt_one_minus_alpha_prod * noise
