"""Utilities for loading and sampling recovery initialization datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import quat_from_euler_xyz, quat_mul


@dataclass
class RecoverySampleBatch:
    root_states: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor


class RecoveryInitDataset:
    """Load and sample gravity-settled recovery states from an NPZ file."""

    def __init__(self, dataset_path: str, device: str):
        resolved_path = resolve_data_file_path(dataset_path)
        with np.load(resolved_path) as data:
            self.root_states = torch.tensor(data["root_states"], dtype=torch.float32, device=device)
            self.dof_pos = torch.tensor(data["dof_pos"], dtype=torch.float32, device=device)
            self.dof_vel = torch.tensor(data["dof_vel"], dtype=torch.float32, device=device)

        if self.root_states.ndim != 2 or self.root_states.shape[1] != 13:
            raise ValueError("Recovery dataset `root_states` must have shape [N, 13].")
        if self.dof_pos.ndim != 2 or self.dof_vel.ndim != 2:
            raise ValueError("Recovery dataset `dof_pos` and `dof_vel` must have shape [N, num_dofs].")
        if self.root_states.shape[0] != self.dof_pos.shape[0] or self.root_states.shape[0] != self.dof_vel.shape[0]:
            raise ValueError("Recovery dataset arrays must contain the same number of samples.")

        self.device = device
        self.num_samples = self.root_states.shape[0]

    def sample(self, batch_size: int, *, yaw_augmentation: bool) -> RecoverySampleBatch:
        if self.num_samples == 0:
            raise ValueError("Recovery dataset is empty.")

        sample_ids = torch.randint(self.num_samples, (batch_size,), device=self.device)
        root_states = self.root_states[sample_ids].clone()
        dof_pos = self.dof_pos[sample_ids].clone()
        dof_vel = self.dof_vel[sample_ids].clone()

        if yaw_augmentation:
            yaw = (torch.rand(batch_size, device=self.device) * 2.0 - 1.0) * torch.pi
            yaw_delta = quat_from_euler_xyz(
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                yaw,
            )
            root_states[:, 3:7] = quat_mul(yaw_delta, root_states[:, 3:7], w_last=True)
            root_states[:, 10:13] = _rotate_yaw_vectors(root_states[:, 10:13], yaw)

        return RecoverySampleBatch(root_states=root_states, dof_pos=dof_pos, dof_vel=dof_vel)


def _rotate_yaw_vectors(vectors: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotated = vectors.clone()
    rotated[:, 0] = cos_yaw * vectors[:, 0] - sin_yaw * vectors[:, 1]
    rotated[:, 1] = sin_yaw * vectors[:, 0] + cos_yaw * vectors[:, 1]
    return rotated
