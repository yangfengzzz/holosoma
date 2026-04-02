#!/usr/bin/env python3
"""Generate gravity-settled recovery initialization datasets for WBT recovery training."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
from pathlib import Path
import random
import sys

import numpy as np

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.command import MotionConfig
from holosoma.config_values.experiment import DEFAULTS as EXPERIMENT_DEFAULTS
from holosoma.managers.randomization.terms.locomotion import randomize_friction_startup
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.helpers import get_class
from holosoma.utils.recovery_init_dataset import RecoveryDatasetMetadata, apply_recovery_root_state_augmentation
from holosoma.utils.safe_torch_import import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="g1_29dof_wbt_recovery_fast_sac")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--raw-output-path", default=None)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--settle-steps", type=int, default=180)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--drop-root-height-range", type=float, nargs=2, default=(0.35, 1.05))
    parser.add_argument("--drop-roll-pitch-range", type=float, nargs=2, default=(-1.2, 1.2))
    parser.add_argument("--drop-yaw-range", type=float, nargs=2, default=(-3.1415926535, 3.1415926535))
    parser.add_argument("--friction-range", type=float, nargs=2, default=(0.3, 1.2))
    parser.add_argument("--joint-noise-scale", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--processed-augmentation-mode",
        default=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
        choices=[mode.value for mode in MotionConfig.RecoveryInitDatasetConfig.AugmentationMode],
    )
    return parser.parse_args()


def build_experiment_config(args: argparse.Namespace):
    from holosoma.train_agent import training_context  # noqa: F401

    if args.exp not in EXPERIMENT_DEFAULTS:
        raise KeyError(f"Unknown experiment preset: {args.exp}")

    base = EXPERIMENT_DEFAULTS[args.exp]
    return dataclasses.replace(
        base,
        training=dataclasses.replace(
            base.training,
            num_envs=args.batch_size,
            headless=args.headless,
        ),
        logger=dataclasses.replace(base.logger, video=dataclasses.replace(base.logger.video, enabled=False)),
    )


def derive_raw_output_path(output_path: str, raw_output_path: str | None) -> Path:
    if raw_output_path:
        return Path(raw_output_path).expanduser().resolve()
    processed = Path(output_path).expanduser().resolve()
    stem = processed.stem
    suffix = processed.suffix or ".npz"
    return processed.with_name(f"{stem}.raw{suffix}")


def build_dataset_metadata(
    *,
    args: argparse.Namespace,
    config,
    dataset_kind: MotionConfig.RecoveryInitDatasetConfig.DatasetKind,
    augmentation_mode: MotionConfig.RecoveryInitDatasetConfig.AugmentationMode,
    num_samples: int,
    source_path: str | None = None,
) -> RecoveryDatasetMetadata:
    return RecoveryDatasetMetadata(
        dataset_kind=dataset_kind,
        augmentation_mode=augmentation_mode,
        preset=args.exp,
        robot_type=config.robot.asset.robot_type,
        friction_range=(float(args.friction_range[0]), float(args.friction_range[1])),
        settle_steps=int(args.settle_steps),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        num_samples=int(num_samples),
        source_path=source_path,
    )


def write_recovery_dataset(
    output_path: Path,
    *,
    root_states: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    metadata: RecoveryDatasetMetadata,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        root_states=root_states,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        metadata_json=np.array(metadata.to_json()),
    )


def randomize_drop_states(env, env_ids: torch.Tensor, args: argparse.Namespace) -> None:
    num = env_ids.numel()
    simulator = env.simulator
    device = env.device

    env_origins = simulator.scene.env_origins[env_ids]
    low_z, high_z = args.drop_root_height_range
    low_rp, high_rp = args.drop_roll_pitch_range
    low_yaw, high_yaw = args.drop_yaw_range

    root_pos = env_origins.clone()
    root_pos[:, 2] = torch.rand(num, device=device) * (high_z - low_z) + low_z

    roll = torch.rand(num, device=device) * (high_rp - low_rp) + low_rp
    pitch = torch.rand(num, device=device) * (high_rp - low_rp) + low_rp
    yaw = torch.rand(num, device=device) * (high_yaw - low_yaw) + low_yaw
    root_quat = _quat_from_euler_xyz(roll, pitch, yaw)

    dof_noise = (torch.rand((num, env.num_dofs), device=device) * 2.0 - 1.0) * args.joint_noise_scale
    dof_pos = env.default_dof_pos[env_ids] + dof_noise
    hard_limits = simulator.dof_pos_limits
    dof_pos = torch.clip(dof_pos, hard_limits[:, 0], hard_limits[:, 1])

    simulator.robot_root_states[env_ids, :3] = root_pos
    simulator.robot_root_states[env_ids, 3:7] = root_quat
    simulator.robot_root_states[env_ids, 7:13] = 0.0
    simulator.dof_pos[env_ids] = dof_pos
    simulator.dof_vel[env_ids] = 0.0

    simulator.set_actor_root_state_tensor_robots(env_ids, simulator.robot_root_states)
    simulator.set_dof_state_tensor_robots(env_ids, simulator.dof_state)
    simulator.refresh_sim_tensors()


def randomize_settle_friction(env, env_ids: torch.Tensor, args: argparse.Namespace) -> None:
    randomize_friction_startup(env, env_ids, friction_range=args.friction_range, enabled=True)


def settle_with_zero_torque(env, settle_steps: int) -> None:
    zero_torques = torch.zeros((env.num_envs, env.num_dofs), device=env.device)
    for _ in range(settle_steps):
        env.render(sync_frame_time=False)
        env.simulator.apply_torques_at_dof(zero_torques)
        env.simulator.simulate_at_each_physics_step()
        env.simulator.refresh_sim_tensors()


def _quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    quat = torch.zeros((roll.shape[0], 4), device=roll.device)
    quat[:, 0] = sr * cp * cy - cr * sp * sy
    quat[:, 1] = cr * sp * cy + sr * cp * sy
    quat[:, 2] = cr * cp * sy - sr * sp * cy
    quat[:, 3] = cr * cp * cy + sr * sp * sy
    return quat


@contextmanager
def _isolated_argv() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        yield
    finally:
        sys.argv = original_argv


def main() -> None:
    init_eval_logging()
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = build_experiment_config(args)
    raw_output_path = derive_raw_output_path(args.output_path, args.raw_output_path)
    processed_output_path = Path(args.output_path).expanduser().resolve()

    collected_root_states: list[np.ndarray] = []
    collected_dof_pos: list[np.ndarray] = []
    collected_dof_vel: list[np.ndarray] = []

    from holosoma.train_agent import training_context

    with _isolated_argv(), training_context(config):
        tyro_env_config = get_tyro_env_config(config)
        env = get_class(config.env_class)(tyro_env_config, device="cuda:0" if torch.cuda.is_available() else "cpu")
        env.reset_all()

        while sum(batch.shape[0] for batch in collected_root_states) < args.num_samples:
            env_ids = torch.arange(env.num_envs, device=env.device)
            randomize_drop_states(env, env_ids, args)
            randomize_settle_friction(env, env_ids, args)
            settle_with_zero_torque(env, args.settle_steps)

            remaining = args.num_samples - sum(batch.shape[0] for batch in collected_root_states)
            keep = min(remaining, env.num_envs)
            keep_ids = env_ids[:keep]

            collected_root_states.append(env.simulator.robot_root_states[keep_ids].detach().cpu().numpy())
            collected_dof_pos.append(env.simulator.dof_pos[keep_ids].detach().cpu().numpy())
            collected_dof_vel.append(env.simulator.dof_vel[keep_ids].detach().cpu().numpy())

        root_states = np.concatenate(collected_root_states, axis=0)
        dof_pos = np.concatenate(collected_dof_pos, axis=0)
        dof_vel = np.concatenate(collected_dof_vel, axis=0)
        processed_root_states = apply_recovery_root_state_augmentation(
            torch.tensor(root_states, dtype=torch.float32, device=env.device),
            augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode(args.processed_augmentation_mode),
        ).detach().cpu().numpy()

        raw_metadata = build_dataset_metadata(
            args=args,
            config=config,
            dataset_kind=MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RAW_GRSI,
            augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.NONE,
            num_samples=root_states.shape[0],
        )
        processed_metadata = build_dataset_metadata(
            args=args,
            config=config,
            dataset_kind=MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT,
            augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode(args.processed_augmentation_mode),
            num_samples=root_states.shape[0],
            source_path=str(raw_output_path),
        )
        write_recovery_dataset(
            raw_output_path,
            root_states=root_states,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            metadata=raw_metadata,
        )
        write_recovery_dataset(
            processed_output_path,
            root_states=processed_root_states,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            metadata=processed_metadata,
        )


if __name__ == "__main__":
    main()
