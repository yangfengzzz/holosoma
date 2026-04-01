from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.managers.command.terms.wbt import LowKineticAnchorSampler
from holosoma.managers.reward.terms import wbt as wbt_reward_terms
from holosoma.managers.termination.terms import wbt as wbt_termination_terms
from holosoma.utils.recovery_init_dataset import RecoveryInitDataset


def test_low_kinetic_anchor_sampler_extracts_and_updates_weights():
    joint_vel = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.1, 0.1],
            [1.2, 1.2],
            [0.2, 0.2],
            [0.8, 0.8],
            [0.05, 0.05],
        ],
        dtype=torch.float32,
    )
    sampler = LowKineticAnchorSampler(
        joint_vel,
        "cpu",
        window_size=1,
        min_anchor_spacing=1,
        ema_alpha=0.5,
        uniform_ratio=0.1,
        failure_weight=1.0,
    )

    assert sampler.anchor_timesteps.tolist() == [0, 3, 5, 7]
    sampler.update_failed_timesteps(torch.tensor([4, 7], dtype=torch.long))
    assert sampler.anchor_weights[1].item() > 1.0
    assert sampler.anchor_weights[-1].item() > 1.0


def test_recovery_dataset_sampling_and_yaw_augmentation(tmp_path):
    dataset_path = tmp_path / "recovery_init.npz"
    np.savez_compressed(
        dataset_path,
        root_states=np.array([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        dof_pos=np.zeros((1, 4), dtype=np.float32),
        dof_vel=np.zeros((1, 4), dtype=np.float32),
    )

    torch.manual_seed(0)
    dataset = RecoveryInitDataset(str(dataset_path), "cpu")
    batch = dataset.sample(1, yaw_augmentation=True)

    assert batch.root_states.shape == (1, 13)
    assert batch.dof_pos.shape == (1, 4)
    assert torch.isclose(torch.norm(batch.root_states[0, 3:7]), torch.tensor(1.0), atol=1e-5)


def test_recovery_action_rate_penalty_is_gated_by_recovery_state():
    motion_command = SimpleNamespace(recovery_active_mask=lambda threshold: torch.tensor([True, False]))
    env = SimpleNamespace(
        action_manager=SimpleNamespace(
            action=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            prev_action=torch.zeros((2, 2)),
        ),
    )

    with patch.object(wbt_reward_terms, "_get_motion_command_and_assert_type", return_value=motion_command):
        penalty = wbt_reward_terms.recovery_action_rate_penalty(env, shoulder_height_threshold=0.2)
    assert penalty.tolist() == [5.0, 0.0]


def test_recovery_aware_bad_tracking_hysteresis():
    motion_command = SimpleNamespace(recovery_active_mask=lambda threshold: torch.tensor([True, False]))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        command_manager=SimpleNamespace(get_state=lambda name: motion_command),
    )
    cfg = TerminationTermCfg(
        func="unused",
        params={
            "bad_ref_pos_threshold": 0.5,
            "bad_ref_ori_threshold": 0.8,
            "bad_motion_body_pos_threshold": 0.25,
            "body_names_to_track": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "bad_motion_body_pos_body_names": ["left_ankle_roll_link"],
            "bad_object_pos_threshold": 0.25,
            "bad_object_ori_threshold": 0.8,
            "shoulder_height_threshold": 0.2,
            "max_consecutive_bad_tracking_steps": 2,
        },
    )
    term = wbt_termination_terms.RecoveryAwareBadTracking(cfg, env)

    with patch.object(wbt_termination_terms.BadTrackingZOnly, "__call__", return_value=torch.tensor([True, True])):
        first = term(env)
        second = term(env)

    assert first.tolist() == [False, True]
    assert second.tolist() == [True, True]
