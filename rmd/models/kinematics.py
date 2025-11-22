# KinematicLoopRefiner

from __future__ import annotations

from typing import Optional

import numpy as np
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)


class KinematicLoopRefiner:
    def __init__(self, bond_length: float = 1.5, max_iterations: int = 50):
        self.bond_length = bond_length
        self.max_iterations = max_iterations

    def _get_rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c
        return np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ]
        )

    def _refine_loop_ccd(self, coords: np.ndarray, start: int, end: int) -> np.ndarray:
        loop_indices = list(range(start + 1, end))
        for _ in range(self.max_iterations):
            moved = False
            for pivot_idx in range(start, end - 1):
                pivot_pos = coords[pivot_idx]
                effector_idx = end
                effector_pos = coords[effector_idx]
                target_pos = coords[start] + (end - start) * (coords[end] - coords[start]) / (
                    np.linalg.norm(coords[end] - coords[start]) + 1e-8
                )
                vec_eff = effector_pos - pivot_pos
                vec_target = target_pos - pivot_pos
                if np.linalg.norm(vec_eff) < 1e-6 or np.linalg.norm(vec_target) < 1e-6:
                    continue
                vec_eff_norm = vec_eff / np.linalg.norm(vec_eff)
                vec_target_norm = vec_target / np.linalg.norm(vec_target)
                cos_angle = np.clip(np.dot(vec_eff_norm, vec_target_norm), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                if angle < 1e-3:
                    continue
                rot_axis = np.cross(vec_eff_norm, vec_target_norm)
                if np.linalg.norm(rot_axis) < 1e-6:
                    continue
                R = self._get_rotation_matrix(rot_axis, angle)
                for j in range(pivot_idx + 1, end + 1):
                    vec = coords[j] - pivot_pos
                    coords[j] = np.dot(R, vec) + pivot_pos
                moved = True
            if not moved:
                break
        current_pos = coords[start]
        for i in loop_indices:
            vec = coords[i] - current_pos
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            coords[i] = current_pos + vec_norm * self.bond_length
            current_pos = coords[i]
        return coords

    def refine(self, coarse_coords: np.ndarray, binding_probs: np.ndarray,
               threshold: float = 0.9, sequence: Optional[str] = None) -> np.ndarray:
        coords = coarse_coords.copy()
        n_nodes = len(coords)
        anchors = set()
        for i in range(n_nodes):
            for j in range(i + 4, n_nodes):
                if binding_probs[i, j] > threshold:
                    anchors.add((min(i, j), max(i, j)))
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f"Found {len(anchors)} anchor pairs")
        anchor_indices = sorted(set(sum(anchors, ()))) if anchors else []
        loops = []
        for i in range(len(anchor_indices) - 1):
            start_anchor = anchor_indices[i]
            end_anchor = anchor_indices[i + 1]
            if end_anchor - start_anchor > 1:
                loops.append((start_anchor, end_anchor))
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f"Refining {len(loops)} loops")
        for start, end in loops:
            coords = self._refine_loop_ccd(coords, start, end)
        return coords
