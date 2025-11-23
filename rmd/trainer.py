from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import RMDConfig


class RMDTrainer:
    def __init__(
        self,
        config: RMDConfig,
        model: nn.Module,
        device: str = "cuda",
        normalizer: Optional["CoordinateNormalizer"] = None,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.normalizer = normalizer

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.lr_t_max,
        )
        self.scaler = torch.amp.GradScaler(enabled=config.mixed_precision)
        self.global_step = 0
    def save_checkpoint(self, path: str | Path) -> None:
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }
        torch.save(state, str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)

    # Module 1: contact prediction
    def _train_step_module1(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq_ids = batch["seq_ids"].to(self.device)          # [B, N]
        contact_map = batch["contact_map"].to(self.device)  # [B, N, N]
        mask = batch["mask"].to(self.device)                # [B, N]

        self.model.train()
        embeddings, logits = self.model(seq_ids, mask)

        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            contact_map,
            reduction="none",
        )

        B, N = seq_ids.shape
        idx = torch.arange(N, device=self.device)
        seq_dist = torch.abs(idx[None, :, None] - idx[None, None, :])
        seq_dist_mask = (seq_dist > 3).float()
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        loss_mask = seq_dist_mask * pair_mask

        loss = (bce_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        return loss

    def _val_step_module1(self, batch: Dict[str, torch.Tensor]) -> Optional[float]:
        seq_ids = batch["seq_ids"].to(self.device)
        contact_map = batch["contact_map"].to(self.device)
        mask = batch["mask"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            embeddings, logits = self.model(seq_ids, mask)

            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                contact_map,
                reduction="none",
            )

            B, N = seq_ids.shape
            idx = torch.arange(N, device=self.device)
            seq_dist = torch.abs(idx[None, :, None] - idx[None, None, :])
            seq_dist_mask = (seq_dist > 3).float()
            pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            loss_mask = seq_dist_mask * pair_mask

            loss = (bce_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
            return float(loss.item())

    def train_module1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> None:
        for epoch in range(num_epochs):
            train_losses: list[float] = []

            for batch in train_loader:
                if not batch:
                    continue

                with torch.amp.autocast(
                    device_type="cuda" if "cuda" in self.device else "cpu",
                    enabled=self.config.mixed_precision,
                ):
                    loss = self._train_step_module1(batch)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.scheduler.step()
                self.global_step += 1

                train_losses.append(loss.item())

            if len(val_loader) > 0:
                val_losses = []
                for batch in val_loader:
                    if not batch:
                        continue
                    val_loss = self._val_step_module1(batch)
                    if val_loss is not None and np.isfinite(val_loss):
                        val_losses.append(val_loss)

    # Module 2: diffusion (coords)
    def _train_step_module2(
        self,
        batch: Dict[str, torch.Tensor],
        frozen_embedder: nn.Module,
    ) -> Optional[torch.Tensor]:
        seq_ids = batch["seq_ids"].to(self.device)    # [B, N]
        coords = batch["coords"].to(self.device)      # [B, N, 3]
        mask = batch["mask"].to(self.device)          # [B, N]

        if torch.isnan(coords).any() or torch.isinf(coords).any():
            return None

        frozen_embedder.eval()
        with torch.no_grad():
            seq_embeddings, _ = frozen_embedder(seq_ids, mask)

        batch_size = seq_ids.shape[0]
        timesteps = torch.randint(
            low=0,
            high=self.config.diffusion_steps,
            size=(batch_size,),
            device=self.device,
        )

        if (
            self.config.use_coordinate_normalization
            and self.normalizer is not None
            and getattr(self.normalizer, "is_fitted", False)
            and not self.config.per_sample_normalization
        ):
            coords_proc = self.normalizer.normalize(coords, mask)
        else:
            coords_proc = coords

        noise = torch.randn_like(coords_proc)
        noisy_coords = self.model.add_noise(coords_proc, noise, timesteps)

        self.model.train()
        noise_pred = self.model(
            noisy_coords=noisy_coords,
            timesteps=timesteps,
            seq_embeddings=seq_embeddings,
            mask=mask,
        )

        if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
            return None

        mse = torch.nn.functional.mse_loss(
            noise_pred,
            noise,
            reduction="none",
        )
        mse = mse * mask.unsqueeze(-1)
        loss = mse.sum() / mask.sum().clamp_min(1.0)
        return loss

    def train_module2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        frozen_embedder: nn.Module,
        num_epochs: int,
    ) -> None:
        frozen_embedder.to(self.device)
        for p in frozen_embedder.parameters():
            p.requires_grad = False

        for epoch in range(num_epochs):
            train_losses: list[float] = []

            for batch in train_loader:
                if not batch:
                    continue

                with torch.amp.autocast(
                    device_type="cuda" if "cuda" in self.device else "cpu",
                    enabled=self.config.mixed_precision,
                ):
                    loss = self._train_step_module2(batch, frozen_embedder)

                if loss is None or not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.scheduler.step()
                self.global_step += 1

                train_losses.append(loss.item())
