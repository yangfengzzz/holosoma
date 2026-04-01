"""Utilities for loading and sampling recovery initialization datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from holosoma.config_types.command import MotionConfig
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import get_euler_xyz, quat_from_euler_xyz, quat_inverse, quat_mul, quat_unit, yaw_quat


@dataclass
class RecoverySampleBatch:
    root_states: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor


@dataclass(frozen=True)
class RecoveryDatasetMetadata:
    dataset_kind: str
    augmentation_mode: str
    preset: str
    robot_type: str
    friction_range: tuple[float, float]
    settle_steps: int
    seed: int
    batch_size: int
    num_samples: int
    source_path: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset_kind": self.dataset_kind,
                "augmentation_mode": self.augmentation_mode,
                "preset": self.preset,
                "robot_type": self.robot_type,
                "friction_range": list(self.friction_range),
                "settle_steps": self.settle_steps,
                "seed": self.seed,
                "batch_size": self.batch_size,
                "num_samples": self.num_samples,
                "source_path": self.source_path,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> RecoveryDatasetMetadata:
        data = json.loads(payload)
        return cls(
            dataset_kind=str(data.get("dataset_kind", MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT)),
            augmentation_mode=str(
                data.get(
                    "augmentation_mode",
                    MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
                )
            ),
            preset=str(data.get("preset", "")),
            robot_type=str(data.get("robot_type", "")),
            friction_range=tuple(float(v) for v in data.get("friction_range", [1.0, 1.0])),
            settle_steps=int(data.get("settle_steps", 0)),
            seed=int(data.get("seed", 0)),
            batch_size=int(data.get("batch_size", 0)),
            num_samples=int(data.get("num_samples", 0)),
            source_path=data.get("source_path"),
        )


class RecoveryInitDataset:
    """Load and sample gravity-settled recovery states from an NPZ file."""

    def __init__(self, dataset_path: str, device: str):
        resolved_path = resolve_data_file_path(dataset_path)
        with np.load(resolved_path) as data:
            self.root_states = torch.tensor(data["root_states"], dtype=torch.float32, device=device)
            self.dof_pos = torch.tensor(data["dof_pos"], dtype=torch.float32, device=device)
            self.dof_vel = torch.tensor(data["dof_vel"], dtype=torch.float32, device=device)
            metadata_json = data["metadata_json"].item() if "metadata_json" in data else ""

        if self.root_states.ndim != 2 or self.root_states.shape[1] != 13:
            raise ValueError("Recovery dataset `root_states` must have shape [N, 13].")
        if self.dof_pos.ndim != 2 or self.dof_vel.ndim != 2:
            raise ValueError("Recovery dataset `dof_pos` and `dof_vel` must have shape [N, num_dofs].")
        if self.root_states.shape[0] != self.dof_pos.shape[0] or self.root_states.shape[0] != self.dof_vel.shape[0]:
            raise ValueError("Recovery dataset arrays must contain the same number of samples.")

        self.device = device
        self.num_samples = self.root_states.shape[0]
        if metadata_json:
            self.metadata = RecoveryDatasetMetadata.from_json(metadata_json)
        else:
            self.metadata = RecoveryDatasetMetadata(
                dataset_kind=MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT,
                augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.YAW,
                preset="",
                robot_type="",
                friction_range=(1.0, 1.0),
                settle_steps=0,
                seed=0,
                batch_size=self.num_samples,
                num_samples=self.num_samples,
            )

    def sample(
        self,
        batch_size: int,
        *,
        augmentation_mode: str | MotionConfig.RecoveryInitDatasetConfig.AugmentationMode | None = None,
    ) -> RecoverySampleBatch:
        if self.num_samples == 0:
            raise ValueError("Recovery dataset is empty.")

        sample_ids = torch.randint(self.num_samples, (batch_size,), device=self.device)
        root_states = self.root_states[sample_ids].clone()
        dof_pos = self.dof_pos[sample_ids].clone()
        dof_vel = self.dof_vel[sample_ids].clone()

        mode = augmentation_mode or self.metadata.augmentation_mode
        if mode == MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.YAW:
            root_states = _apply_yaw_augmentation(root_states)
        elif mode == MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION:
            root_states = _apply_rotation_recombination(root_states, self.root_states, self.device)
        elif mode != MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.NONE:
            raise ValueError(f"Unsupported recovery augmentation mode: {mode}")

        return RecoverySampleBatch(root_states=root_states, dof_pos=dof_pos, dof_vel=dof_vel)


def _rotate_yaw_vectors(vectors: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotated = vectors.clone()
    rotated[:, 0] = cos_yaw * vectors[:, 0] - sin_yaw * vectors[:, 1]
    rotated[:, 1] = sin_yaw * vectors[:, 0] + cos_yaw * vectors[:, 1]
    return rotated


def _apply_yaw_augmentation(root_states: torch.Tensor) -> torch.Tensor:
    batch_size = root_states.shape[0]
    yaw = (torch.rand(batch_size, device=root_states.device) * 2.0 - 1.0) * torch.pi
    yaw_delta = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    augmented = root_states.clone()
    augmented[:, 3:7] = quat_mul(yaw_delta, root_states[:, 3:7], w_last=True)
    augmented[:, 7:10] = _rotate_yaw_vectors(root_states[:, 7:10], yaw)
    augmented[:, 10:13] = _rotate_yaw_vectors(root_states[:, 10:13], yaw)
    return augmented


def _apply_rotation_recombination(
    root_states: torch.Tensor,
    source_root_states: torch.Tensor,
    device: str,
) -> torch.Tensor:
    donor_ids = torch.randint(source_root_states.shape[0], (root_states.shape[0],), device=device)
    donor_states = source_root_states[donor_ids]

    primary_quat = quat_unit(root_states[:, 3:7])
    donor_quat = quat_unit(donor_states[:, 3:7])

    primary_yaw_quat = yaw_quat(primary_quat, w_last=True)
    donor_yaw_quat = yaw_quat(donor_quat, w_last=True)
    donor_tilt_quat = quat_mul(quat_inverse(donor_yaw_quat, w_last=True), donor_quat, w_last=True)

    recombined = root_states.clone()
    recombined[:, 3:7] = quat_unit(quat_mul(primary_yaw_quat, donor_tilt_quat, w_last=True))

    _, _, primary_yaw = get_euler_xyz(primary_yaw_quat, w_last=True)
    _, _, donor_yaw = get_euler_xyz(donor_yaw_quat, w_last=True)
    yaw_delta = primary_yaw - donor_yaw

    recombined[:, 7:10] = _rotate_yaw_vectors(donor_states[:, 7:10], yaw_delta)
    recombined[:, 10:12] = _rotate_yaw_vectors(donor_states[:, 10:13], yaw_delta)[:, :2]
    recombined[:, 12] = root_states[:, 12]
    return recombined
