# CLI entrypoint

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rmd.config import RMDConfig
from rmd.data import RNADataset, collate_fn, CoordinateNormalizer
from rmd.models.module1 import MotifInteractionNetwork
from rmd.models.module2 import DiffusionModel
from rmd.trainer import RMDTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNA Motif Diffuser")

    parser.add_argument("--train-json", type=str, required=True,
                        help="Path to training data JSON")
    parser.add_argument("--val-json", type=str, required=True,
                        help="Path to validation data JSON")

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for logs")

    parser.add_argument(
        "--module",
        type=str,
        choices=["1", "2", "both"],
        default="1",
        help="Which module to train: 1 (contacts), 2 (diffusion), or both"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume from (for Module 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RMDConfig:
    cfg = RMDConfig()
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.log_dir = args.log_dir
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def create_dataloaders(
    train_json: str,
    val_json: str,
    config: RMDConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = RNADataset(train_json, max_seq_len=config.max_seq_len)
    val_dataset = RNADataset(val_json, max_seq_len=config.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def fit_normalizer(
    train_dataset: RNADataset,
    config: RMDConfig,
) -> CoordinateNormalizer | None:
    if not config.use_coordinate_normalization:
        return None

    normalizer = CoordinateNormalizer(
        method="minmax",
        clip_range=(-0.1, 1.1),
    )

    if config.per_sample_normalization:
        # Per-sample normalization is handled inside the trainer
        return None

    coords_list = []
    mask_list = []
    sample_size = min(1000, len(train_dataset))

    for i in range(sample_size):
        sample = train_dataset[i]
        coords_list.append(sample["coords"])
        mask_list.append(sample["mask"])

    normalizer.fit(coords_list, mask_list, use_percentiles=True)
    return normalizer


def main() -> None:
    args = parse_args()
    config = build_config(args)

    device = args.device

    # Data
    train_dataset = RNADataset(args.train_json, max_seq_len=config.max_seq_len)
    val_dataset = RNADataset(args.val_json, max_seq_len=config.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    normalizer = fit_normalizer(train_dataset, config)
    if normalizer is not None:
        normalizer_path = Path(config.checkpoint_dir) / "normalizer.pt"
        torch.save(normalizer.get_state_dict(), normalizer_path)

    # Phase 1: Module 1 (contact prediction)
    trainer1 = None
    if args.module in ["1", "both"]:
        module1 = MotifInteractionNetwork(config)
        trainer1 = RMDTrainer(config, module1, device=device, normalizer=None)

        if args.resume and os.path.exists(args.resume):
            trainer1.load_checkpoint(args.resume)

        trainer1.train_module1(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs_module1,
        )

        best_path = Path(config.checkpoint_dir) / "module1_best.pt"
        trainer1.save_checkpoint(best_path)

    # Phase 2: Module 2 (diffusion model)
    if args.module in ["2", "both"]:
        module2 = DiffusionModel(config, normalizer=normalizer)
        trainer2 = RMDTrainer(config, module2, device=device, normalizer=normalizer)

        # Use a frozen embedder
        if args.module == "both":
            if trainer1 is None:
                raise RuntimeError("Trainer for Module 1 is not available.")
            frozen_embedder = trainer1.model.embedder
        else:
            module1_path = Path(config.checkpoint_dir) / "module1_best.pt"
            if not module1_path.exists():
                print(
                    f"Module 1 checkpoint not found at {module1_path}. "
                    "Train Module 1 first or use --module both.",
                    file=sys.stderr,
                )
                sys.exit(1)

            module1 = MotifInteractionNetwork(config)
            ckpt = torch.load(module1_path, map_location=device)
            module1.load_state_dict(ckpt["model_state_dict"])
            module1.to(device)
            frozen_embedder = module1.embedder

        trainer2.train_module2(
            train_loader=train_loader,
            val_loader=val_loader,
            frozen_embedder=frozen_embedder,
            num_epochs=config.num_epochs_module2,
        )

        best_path = Path(config.checkpoint_dir) / "module2_best.pt"
        trainer2.save_checkpoint(best_path)


if __name__ == "__main__":
    main()
