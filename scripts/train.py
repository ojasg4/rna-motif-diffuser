# CLI entrypoint

from __future__ import annotations

import argparse
import logging

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rmd.config import RMDConfig, setup_pace_gpu
from rmd.data import RNAStructureDataset, collate_fn, CoordinateNormalizer
from rmd.models.evoformer import MotifInteractionNetwork
from rmd.training import RMDTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rmd.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RNA-Motif-Diffuser training")
    parser.add_argument("--mode", choices=["kaggle", "hpc", "debug"], default="kaggle")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "kaggle":
        config = RMDConfig.kaggle_config()
    elif args.mode == "hpc":
        num_gpus = setup_pace_gpu()
        config = RMDConfig.hpc_config(num_gpus=num_gpus)
    else:
        config = RMDConfig.debug_config()

    distributed = config.distributed and args.num_gpus > 1
    device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"

    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)

    train_dataset = RNAStructureDataset(config.data_dir, max_seq_len=config.max_seq_len, split="train")
    val_dataset = RNAStructureDataset(config.data_dir, max_seq_len=config.max_seq_len, split="val")

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )

    normalizer = CoordinateNormalizer() if config.use_coordinate_normalization else None
    model = MotifInteractionNetwork(config)
    trainer = RMDTrainer(config, model, device=device, normalizer=normalizer)
    trainer.train_module1(train_loader, val_loader, num_epochs=args.epochs)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
