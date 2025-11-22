from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import torch
import logging

logger = logging.getLogger(__name__)


def setup_pace_gpu() -> int:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_devices is not None:
        logger.info(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
        num_gpus = len(visible_devices.split(","))
    else:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPUs")
    return num_gpus


@dataclass
class RMDConfig:
    vocab_size: int = 5
    max_seq_len: int = 512

    embed_dim: int = 512
    cnn_channels: List[int] = field(default_factory=lambda: [128, 256, 512])
    cnn_kernel_size: int = 7
    transformer_layers: int = 12
    transformer_heads: int = 16
    dropout: float = 0.1
    num_evoformer_blocks: int = 4

    diffusion_steps: int = 1000
    diffusion_dim: int = 512
    gnn_layers: int = 12
    noise_schedule: str = "cosine"
    use_coordinate_normalization: bool = False
    per_sample_normalization: bool = False
    recenter_coordinates: bool = True

    batch_size: int = 32
    grad_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10_000
    max_grad_norm: float = 1.0

    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    distributed: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    compile_model: bool = False

    max_neighbors: int = 64
    flash_attention: bool = True
    chunk_size: int = 64

    data_dir: str = "./data/preprocessed"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    def __post_init__(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @classmethod
    def kaggle_config(cls) -> "RMDConfig":
        return cls(
            max_seq_len=256,
            batch_size=4,
            embed_dim=256,
            diffusion_dim=256,
            cnn_channels=[64, 128, 256],
            transformer_layers=6,
            transformer_heads=8,
            gnn_layers=6,
            num_evoformer_blocks=2,
            grad_accumulation_steps=8,
            gradient_checkpointing=True,
            mixed_precision=True,
            distributed=False,
            compile_model=False,
            num_workers=2,
            flash_attention=True,
            max_neighbors=48,
            chunk_size=64,
            use_coordinate_normalization=False,
            per_sample_normalization=False,
            recenter_coordinates=True,
            data_dir="/kaggle/input/diff-kin-gem-preprocessing/data/preprocessed",
        )

    @classmethod
    def hpc_config(cls, num_gpus: int = 8) -> "RMDConfig":
        home_dir = os.path.expanduser("~")
        scratch_dir = os.path.join(home_dir, "scratch")
        return cls(
            max_seq_len=256,
            batch_size=8,
            embed_dim=384,
            diffusion_dim=384,
            cnn_channels=[96, 192, 384],
            transformer_layers=8,
            transformer_heads=8,
            gnn_layers=8,
            num_evoformer_blocks=2,
            grad_accumulation_steps=4,
            gradient_checkpointing=True,
            mixed_precision=True,
            distributed=True,
            compile_model=False,
            num_workers=4,
            flash_attention=True,
            max_neighbors=48,
            chunk_size=64,
            data_dir=os.path.join(scratch_dir, "archive"),
            checkpoint_dir=os.path.join(scratch_dir, "rna-diffusion", "checkpoints"),
            log_dir=os.path.join(scratch_dir, "rna-diffusion", "logs"),
        )

    @classmethod
    def debug_config(cls) -> "RMDConfig":
        return cls(
            max_seq_len=64,
            batch_size=2,
            embed_dim=128,
            diffusion_dim=128,
            cnn_channels=[64, 128],
            transformer_layers=2,
            transformer_heads=4,
            gnn_layers=2,
            num_evoformer_blocks=1,
            grad_accumulation_steps=1,
            gradient_checkpointing=False,
            mixed_precision=False,
            distributed=False,
            compile_model=False,
            num_workers=0,
            flash_attention=False,
            max_neighbors=16,
            chunk_size=32,
            data_dir="./data/preprocessed",
        )
