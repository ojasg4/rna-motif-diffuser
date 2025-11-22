from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import RMDConfig
from .data import CoordinateNormalizer


class RMDTrainer:
    def __init__(self, config: RMDConfig, model: torch.nn.Module, device: str = "cuda",
                 normalizer: Optional[CoordinateNormalizer] = None):
        self.config = config
        self.device = device
        self.normalizer = normalizer
        model = model.to(device)
        if config.distributed and dist.is_initialized():
            self.model = DDP(model, device_ids=[int(device.split(":")[-1])])
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.model = model
            self.rank = 0
            self.world_size = 1
        effective_lr = config.learning_rate * 0.5
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=effective_lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=100000,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        self.scaler = torch.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
        self.grad_accumulation_steps = config.grad_accumulation_steps
        self.global_step = 0

    def train_module1(self, train_loader: DataLoader, val_loader: DataLoader,
                      num_epochs: int = 10, log_interval: int = 10) -> None:
        criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(num_epochs):
            if self.rank == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
            self.model.train()
            for step, batch in enumerate(train_loader):
                seq_ids = batch["seq_ids"].to(self.device)
                contact_map = batch["contact_map"].to(self.device)
                mask = batch["mask"].to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.scaler is not None):
                    _, binding_logits = self.model(seq_ids, mask)
                    loss = criterion(binding_logits, contact_map)
                    loss = loss / self.grad_accumulation_steps
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % self.grad_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1
                    if self.rank == 0 and self.global_step % log_interval == 0:
                        print(f"Step {self.global_step}: loss={loss.item():.4f}")
            val_loss = self.evaluate_module1(val_loader, criterion)
            if self.rank == 0:
                print(f"Validation loss: {val_loss:.4f}")

    @torch.no_grad()
    def evaluate_module1(self, val_loader: DataLoader, criterion: torch.nn.Module) -> float:
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        for batch in val_loader:
            seq_ids = batch["seq_ids"].to(self.device)
            contact_map = batch["contact_map"].to(self.device)
            mask = batch["mask"].to(self.device)
            _, binding_logits = self.model(seq_ids, mask)
            loss = criterion(binding_logits, contact_map)
            total_loss += loss.item() * seq_ids.size(0)
            total_count += seq_ids.size(0)
        return total_loss / max(total_count, 1)
