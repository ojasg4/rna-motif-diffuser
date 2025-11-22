from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class CoordinateNormalizer:
    def __init__(self, method: str = "minmax", clip_range: Tuple[float, float] = (-5.0, 5.0)):
        self.method = method
        self.clip_range = clip_range
        self.coord_mean: Optional[torch.Tensor] = None
        self.coord_std: Optional[torch.Tensor] = None
        self.coord_min: Optional[torch.Tensor] = None
        self.coord_max: Optional[torch.Tensor] = None
        self.coord_range: Optional[torch.Tensor] = None
        self.scale_factor: Optional[torch.Tensor] = None
        self.is_fitted = False

    def fit(
        self,
        coords_list: List[torch.Tensor],
        mask_list: Optional[List[torch.Tensor]] = None,
        use_percentiles: bool = True,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
    ) -> None:
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info("Computing coordinate normalization statistics")
        all_coords = []
        for i, coords in enumerate(coords_list):
            if mask_list is not None and i < len(mask_list):
                mask = mask_list[i]
                valid_coords = coords[mask.bool()]
            else:
                valid_coords = coords
            all_coords.append(valid_coords)
        stacked_coords = torch.cat(all_coords, dim=0)
        if self.method == "minmax":
            if use_percentiles:
                self.coord_min = torch.quantile(stacked_coords, percentile_low / 100.0, dim=0)
                self.coord_max = torch.quantile(stacked_coords, percentile_high / 100.0, dim=0)
            else:
                self.coord_min = stacked_coords.min(dim=0)[0]
                self.coord_max = stacked_coords.max(dim=0)[0]
            self.coord_range = self.coord_max - self.coord_min
            self.coord_range = torch.clamp(self.coord_range, min=1e-6)
            self.scale_factor = self.coord_range.max()
        else:
            self.coord_mean = stacked_coords.mean(dim=0)
            self.coord_std = stacked_coords.std(dim=0)
            self.coord_std = torch.clamp(self.coord_std, min=1e-6)
            self.scale_factor = (self.coord_std * 3).max()
        self.is_fitted = True

    def normalize(self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before use")
        is_batched = coords.dim() == 3
        if not is_batched:
            coords = coords.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        device = coords.device
        if self.method == "minmax":
            coord_min = self.coord_min.to(device).unsqueeze(0).unsqueeze(0)
            coord_range = self.coord_range.to(device).unsqueeze(0).unsqueeze(0)
            normalized = (coords - coord_min) / coord_range
        else:
            coord_mean = self.coord_mean.to(device).unsqueeze(0).unsqueeze(0)
            scale_factor = self.scale_factor.to(device)
            normalized = ((coords - coord_mean) / scale_factor + 1.0) / 2.0
        normalized = torch.clamp(normalized, self.clip_range[0], self.clip_range[1])
        if mask is not None:
            normalized = normalized * mask.unsqueeze(-1)
        if not is_batched:
            normalized = normalized.squeeze(0)
        return normalized

    def denormalize(self, normalized_coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before use")
        is_batched = normalized_coords.dim() == 3
        if not is_batched:
            normalized_coords = normalized_coords.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        device = normalized_coords.device
        if self.method == "minmax":
            coord_min = self.coord_min.to(device).unsqueeze(0).unsqueeze(0)
            coord_range = self.coord_range.to(device).unsqueeze(0).unsqueeze(0)
            coords = normalized_coords * coord_range + coord_min
        else:
            coord_mean = self.coord_mean.to(device).unsqueeze(0).unsqueeze(0)
            scale_factor = self.scale_factor.to(device)
            coords = ((normalized_coords * 2.0 - 1.0) * scale_factor) + coord_mean
        if mask is not None:
            coords = coords * mask.unsqueeze(-1)
        if not is_batched:
            coords = coords.squeeze(0)
        return coords

    def get_state_dict(self) -> Dict:
        return {
            "method": self.method,
            "clip_range": self.clip_range,
            "coord_mean": self.coord_mean,
            "coord_std": self.coord_std,
            "coord_min": self.coord_min,
            "coord_max": self.coord_max,
            "coord_range": self.coord_range,
            "scale_factor": self.scale_factor,
            "is_fitted": self.is_fitted,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.method = state_dict["method"]
        self.clip_range = state_dict["clip_range"]
        self.coord_mean = state_dict["coord_mean"]
        self.coord_std = state_dict["coord_std"]
        self.coord_min = state_dict["coord_min"]
        self.coord_max = state_dict["coord_max"]
        self.coord_range = state_dict["coord_range"]
        self.scale_factor = state_dict["scale_factor"]
        self.is_fitted = state_dict["is_fitted"]


class RNAStructureDataset(Dataset):
    def __init__(self, data_dir: str, max_seq_len: int = 512, split: str = "train", cache_in_memory: bool = False):
        data_path = Path(data_dir)
        if (data_path / split).exists():
            self.data_dir = data_path / split
        else:
            self.data_dir = data_path
        self.max_seq_len = max_seq_len
        self.split = split
        self.cache_in_memory = cache_in_memory
        self.data_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self.samples = self._load_index()
        if self.samples and cache_in_memory and len(self.samples) < 1000:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info("Caching dataset in memory")
            for idx in range(len(self.samples)):
                self.data_cache[idx] = self._load_npz(idx)

    def _load_index(self) -> List[Path]:
        if not self.data_dir.exists():
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.error(f"Data directory does not exist: {self.data_dir}")
            return []
        npz_files = sorted([f for f in self.data_dir.glob("*.npz") if f.name != "index.npz"])
        if not npz_files and (not dist.is_initialized() or dist.get_rank() == 0):
            logger.error(f"No .npz files found in {self.data_dir}")
        return npz_files

    def _load_npz(self, idx: int) -> Dict[str, np.ndarray]:
        filepath = self.samples[idx]
        try:
            data = np.load(filepath, allow_pickle=True)
            return {
                "seq_ids": data["seq_ids"],
                "coords": data["coords"],
                "contact_map": data["contact_map"],
                "sequence": str(data["sequence"]) if "sequence" in data else "",
                "target_id": str(data["target_id"]) if "target_id" in data else filepath.stem,
            }
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self.data_cache:
            data = self.data_cache[idx]
        else:
            try:
                data = self._load_npz(idx)
            except Exception:
                return self.__getitem__((idx + 1) % len(self))
        seq_ids = torch.from_numpy(data["seq_ids"]).long()
        coords = torch.from_numpy(data["coords"]).float()
        contact_map = torch.from_numpy(data["contact_map"]).float()
        original_seq_len = len(seq_ids)
        if original_seq_len > self.max_seq_len:
            seq_ids = seq_ids[: self.max_seq_len]
            coords = coords[: self.max_seq_len]
            contact_map = contact_map[: self.max_seq_len, : self.max_seq_len]
            seq_len = self.max_seq_len
            mask = torch.ones(seq_len)
        elif original_seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - original_seq_len
            seq_ids = F.pad(seq_ids, (0, pad_len), value=4)
            coords = F.pad(coords, (0, 0, 0, pad_len), value=0.0)
            contact_map = F.pad(contact_map, (0, pad_len, 0, pad_len), value=0.0)
            mask = torch.cat([torch.ones(original_seq_len), torch.zeros(pad_len)])
            seq_len = original_seq_len
        else:
            mask = torch.ones(original_seq_len)
            seq_len = original_seq_len
        return {
            "seq_ids": seq_ids,
            "coords": coords,
            "contact_map": contact_map,
            "mask": mask,
            "seq_len": torch.tensor(seq_len),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    if not batch:
        return {}
    batch = [item for item in batch if item]
    if not batch:
        return {}
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0].keys()}
