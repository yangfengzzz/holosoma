#!/usr/bin/env python3
"""Validate that a recovery-init dataset can drive real WBT recovery resets."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from contextlib import contextmanager

from holosoma.config_types.command import MotionConfig
from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_values.experiment import DEFAULTS as EXPERIMENT_DEFAULTS
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.helpers import get_class
from holosoma.utils.recovery_init_dataset import RecoveryInitDataset
from holosoma.utils.safe_torch_import import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="g1_29dof_wbt_recovery_fast_sac")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


@contextmanager
def _isolated_argv() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        yield
    finally:
        sys.argv = original_argv


def build_validation_config(args: argparse.Namespace):
    from holosoma.train_agent import training_context  # noqa: F401

    if args.exp not in EXPERIMENT_DEFAULTS:
        raise KeyError(f"Unknown experiment preset: {args.exp}")

    base = EXPERIMENT_DEFAULTS[args.exp]
    motion_term = base.command.setup_terms["motion_command"]
    motion_cfg = motion_term.params["motion_config"]
    if not isinstance(motion_cfg, MotionConfig):
        motion_cfg = MotionConfig(**motion_cfg)

    recovery_cfg = dataclasses.replace(
        motion_cfg.recovery_init_dataset,
        enabled=True,
        dataset_path=args.dataset_path,
        sample_probability=1.0,
        augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
    )
    patched_motion_cfg = dataclasses.replace(motion_cfg, recovery_init_dataset=recovery_cfg)
    patched_command = dataclasses.replace(
        base.command,
        setup_terms={
            **base.command.setup_terms,
            "motion_command": dataclasses.replace(
                motion_term,
                params={**motion_term.params, "motion_config": patched_motion_cfg},
            ),
        },
    )

    return dataclasses.replace(
        base,
        training=dataclasses.replace(base.training, num_envs=args.batch_size, headless=args.headless),
        logger=dataclasses.replace(base.logger, video=dataclasses.replace(base.logger.video, enabled=False)),
        command=patched_command,
    )


def validate_dataset_contract(dataset: RecoveryInitDataset, *, expected_num_dofs: int) -> None:
    if dataset.root_states.shape[1] != 13:
        raise ValueError("Recovery dataset root_states must have 13 columns.")
    if dataset.dof_pos.shape[1] != expected_num_dofs or dataset.dof_vel.shape[1] != expected_num_dofs:
        raise ValueError(
            f"Recovery dataset DOF shape mismatch: expected {expected_num_dofs}, "
            f"got dof_pos={dataset.dof_pos.shape[1]} dof_vel={dataset.dof_vel.shape[1]}"
        )
    if dataset.metadata.dataset_kind != MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT:
        raise ValueError(f"Expected a training-ready recovery dataset, got {dataset.metadata.dataset_kind!r}")


def main() -> None:
    init_eval_logging()
    args = parse_args()
    config = build_validation_config(args)

    from holosoma.train_agent import training_context

    with _isolated_argv(), training_context(config):
        tyro_env_config = get_tyro_env_config(config)
        env = get_class(config.env_class)(tyro_env_config, device="cuda:0" if torch.cuda.is_available() else "cpu")
        env.reset_all()

        motion_command = env.command_manager.get_state("motion_command")
        if motion_command is None or motion_command.recovery_init_dataset is None:
            raise RuntimeError("Recovery dataset was not wired into MotionCommand.")

        dataset = motion_command.recovery_init_dataset
        validate_dataset_contract(dataset, expected_num_dofs=env.num_dofs)

        env_ids = torch.arange(env.num_envs, device=env.device)
        motion_command.reset(env_ids)
        env.simulator.set_actor_root_state_tensor_robots(env_ids, env.simulator.robot_root_states)
        env.simulator.set_dof_state_tensor_robots(env_ids, env.simulator.dof_state)
        env.simulator.refresh_sim_tensors()

        if not torch.all(motion_command.last_reset_used_recovery[env_ids]):
            raise RuntimeError("Expected all validation resets to come from the recovery dataset.")
        if not torch.isfinite(env.simulator.robot_root_states[env_ids]).all():
            raise RuntimeError("Non-finite robot root states produced by recovery reset.")
        if not torch.isfinite(env.simulator.dof_pos[env_ids]).all():
            raise RuntimeError("Non-finite DOF positions produced by recovery reset.")

        print(
            "validated_recovery_dataset",
            {
                "dataset_path": args.dataset_path,
                "num_envs": int(env.num_envs),
                "num_dofs": int(env.num_dofs),
                "augmentation_mode": dataset.metadata.augmentation_mode,
            },
        )


if __name__ == "__main__":
    main()
