from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from holosoma.config_types.command import MotionConfig
from holosoma.generate_recovery_dataset import (
    build_dataset_metadata,
    derive_raw_output_path,
    write_recovery_dataset,
)
from holosoma.utils.recovery_init_dataset import RecoveryInitDataset


def test_derive_raw_output_path_defaults_to_raw_suffix(tmp_path: Path):
    output_path = tmp_path / "g1_ground_v1.npz"
    assert derive_raw_output_path(str(output_path), None) == tmp_path / "g1_ground_v1.raw.npz"


def test_write_recovery_dataset_persists_metadata(tmp_path: Path):
    args = Namespace(
        exp="g1_29dof_wbt_recovery_fast_sac",
        friction_range=(0.3, 1.2),
        settle_steps=180,
        seed=7,
        batch_size=4,
    )
    config = Namespace(robot=Namespace(asset=Namespace(robot_type="g1_29dof")))
    metadata = build_dataset_metadata(
        args=args,
        config=config,
        dataset_kind=MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT,
        augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
        num_samples=4,
        source_path="raw.npz",
    )

    output_path = tmp_path / "dataset.npz"
    write_recovery_dataset(
        output_path,
        root_states=np.zeros((4, 13), dtype=np.float32),
        dof_pos=np.zeros((4, 2), dtype=np.float32),
        dof_vel=np.zeros((4, 2), dtype=np.float32),
        metadata=metadata,
    )

    dataset = RecoveryInitDataset(str(output_path), "cpu")
    assert dataset.metadata.dataset_kind == MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT
    assert dataset.metadata.augmentation_mode == MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION
    assert dataset.metadata.source_path == "raw.npz"
