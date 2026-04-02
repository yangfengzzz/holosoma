from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.termination import TerminationTermCfg
from holosoma.managers.command.terms.wbt import LowKineticAnchorSampler, MotionCommand
from holosoma.managers.reward.terms import wbt as wbt_reward_terms
from holosoma.managers.termination.terms import wbt as wbt_termination_terms
from holosoma.config_types.command import MotionConfig
from holosoma.utils.recovery_init_dataset import (
    RecoveryDatasetMetadata,
    RecoveryInitDataset,
    apply_recovery_root_state_augmentation,
)


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
    assert sampler.anchor_weights.tolist() == pytest.approx([1.0, 1.5, 1.0, 1.5])
    sampler.update_failed_timesteps(torch.tensor([5], dtype=torch.long))
    assert sampler.anchor_weights.tolist() == pytest.approx([1.0, 1.5, 1.5, 1.5])


def test_recovery_dataset_sampling_and_augmentation_modes(tmp_path):
    dataset_path = tmp_path / "recovery_init.npz"
    np.savez_compressed(
        dataset_path,
        root_states=np.array(
            [
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.995, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dof_pos=np.zeros((2, 4), dtype=np.float32),
        dof_vel=np.zeros((2, 4), dtype=np.float32),
        metadata_json=np.array(
            RecoveryDatasetMetadata(
                dataset_kind=MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT,
                augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
                preset="g1_29dof_wbt_recovery_fast_sac",
                robot_type="g1_29dof",
                friction_range=(0.3, 1.2),
                settle_steps=180,
                seed=7,
                batch_size=2,
                num_samples=2,
            ).to_json()
        ),
    )

    torch.manual_seed(0)
    dataset = RecoveryInitDataset(str(dataset_path), "cpu")
    no_aug = dataset.sample(1, augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.NONE)
    yaw_aug = dataset.sample(1, augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.YAW)
    recombined = dataset.sample(
        2,
        augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
    )

    assert dataset.metadata.dataset_kind == MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT
    assert no_aug.root_states.shape == (1, 13)
    assert yaw_aug.dof_pos.shape == (1, 4)
    assert torch.isclose(torch.norm(yaw_aug.root_states[0, 3:7]), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(torch.norm(recombined.root_states[0, 3:7]), torch.tensor(1.0), atol=1e-5)
    assert recombined.root_states.shape == (2, 13)
    assert recombined.dof_vel.shape == (2, 4)


def test_processed_recovery_dataset_is_not_augmented_again_on_sample(tmp_path):
    dataset_path = tmp_path / "processed_recovery_init.npz"
    stored_root_states = np.array(
        [
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        dataset_path,
        root_states=stored_root_states,
        dof_pos=np.zeros((1, 4), dtype=np.float32),
        dof_vel=np.zeros((1, 4), dtype=np.float32),
        metadata_json=np.array(
            RecoveryDatasetMetadata(
                dataset_kind=MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT,
                augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
                preset="g1_29dof_wbt_recovery_fast_sac",
                robot_type="g1_29dof",
                friction_range=(0.3, 1.2),
                settle_steps=180,
                seed=7,
                batch_size=1,
                num_samples=1,
            ).to_json()
        ),
    )

    dataset = RecoveryInitDataset(str(dataset_path), "cpu")
    sampled = dataset.sample(
        1,
        augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
    )

    assert torch.allclose(sampled.root_states, torch.tensor(stored_root_states))


def test_recovery_dataset_rotation_recombination_preserves_quaternion_norm(tmp_path):
    dataset_path = tmp_path / "raw_recovery_init.npz"
    np.savez_compressed(
        dataset_path,
        root_states=np.array(
            [
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.1, 0.2, 0.3],
                [0.0, 0.0, 0.6, 0.2, 0.1, 0.0, 0.97, 0.0, 0.1, 0.0, 0.4, 0.5, 0.6],
            ],
            dtype=np.float32,
        ),
        dof_pos=np.zeros((2, 3), dtype=np.float32),
        dof_vel=np.zeros((2, 3), dtype=np.float32),
        metadata_json=np.array(
            RecoveryDatasetMetadata(
                dataset_kind=MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RAW_GRSI,
                augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.NONE,
                preset="g1_29dof_wbt_recovery_fast_sac",
                robot_type="g1_29dof",
                friction_range=(0.3, 1.2),
                settle_steps=180,
                seed=7,
                batch_size=2,
                num_samples=2,
            ).to_json()
        ),
    )

    dataset = RecoveryInitDataset(str(dataset_path), "cpu")
    batch = dataset.sample(2, augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION)
    assert torch.allclose(torch.norm(batch.root_states[:, 3:7], dim=1), torch.ones(2), atol=1e-4)


def test_processed_recovery_dataset_materialization_differs_from_raw_states():
    raw_root_states = torch.tensor(
        [
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.1, 0.2, 0.3],
            [0.0, 0.0, 0.6, 0.2, 0.1, 0.0, 0.97, 0.0, 0.1, 0.0, 0.4, 0.5, 0.6],
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(0)
    processed_root_states = apply_recovery_root_state_augmentation(
        raw_root_states,
        augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
    )

    assert not torch.allclose(processed_root_states, raw_root_states)
    assert torch.allclose(torch.norm(processed_root_states[:, 3:7], dim=1), torch.ones(2), atol=1e-4)


def test_recovery_action_rate_penalty_is_gated_by_recovery_state():
    motion_command = SimpleNamespace(
        motion_cfg=SimpleNamespace(recovery_shoulder_height_threshold=1.0),
        recovery_active_mask=lambda threshold: torch.tensor([True, False]),
    )
    env = SimpleNamespace(
        action_manager=SimpleNamespace(
            action=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            prev_action=torch.zeros((2, 2)),
        ),
    )

    with patch.object(wbt_reward_terms, "_get_motion_command_and_assert_type", return_value=motion_command):
        penalty = wbt_reward_terms.recovery_action_rate_penalty(env, shoulder_height_threshold=None)
    assert penalty.tolist() == [5.0, 0.0]


def test_recovery_shoulder_height_threshold_defaults_to_motion_config():
    motion_command = SimpleNamespace(
        motion_cfg=SimpleNamespace(recovery_shoulder_height_threshold=1.0),
        recovery_active_mask=lambda threshold: torch.tensor([threshold == 1.0, False]),
    )
    env = SimpleNamespace(
        action_manager=SimpleNamespace(
            action=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            prev_action=torch.zeros((2, 2)),
        ),
    )

    with patch.object(wbt_reward_terms, "_get_motion_command_and_assert_type", return_value=motion_command):
        penalty = wbt_reward_terms.recovery_action_rate_penalty(env, shoulder_height_threshold=None)
    assert penalty.tolist() == [1.0, 0.0]


def test_motion_com_support_alignment_uses_support_foot():
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        com_body_indices=torch.tensor([0, 1]),
        rigid_body_masses=torch.tensor([3.0, 1.0, 0.0, 0.0]),
        feet_indices=torch.tensor([2, 3]),
        simulator=SimpleNamespace(
            _rigid_body_pos=torch.tensor(
                [[[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [2.0, 0.0, 0.1]]], dtype=torch.float32
            ),
            contact_forces=torch.tensor(
                [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
            ),
        ),
    )

    reward = wbt_reward_terms.motion_com_support_alignment_exp(env, sigma=0.15, contact_force_threshold=1.0)
    assert reward.item() == pytest.approx(1.0, rel=1e-6)


def test_feet_slip_penalty_uses_contact_force_stance_detection():
    env = SimpleNamespace(
        feet_indices=torch.tensor([0, 1]),
        simulator=SimpleNamespace(
            contact_forces=torch.tensor(
                [[[0.0, 0.0, 5.0], [0.0, 0.0, 0.2]]],
                dtype=torch.float32,
            ),
            _rigid_body_vel=torch.tensor(
                [[[3.0, 4.0, 0.0], [6.0, 8.0, 0.0]]],
                dtype=torch.float32,
            ),
        ),
    )

    penalty = wbt_reward_terms.feet_slip_penalty(env, contact_force_threshold=1.0)
    assert penalty.item() == pytest.approx(5.0)


def test_close_feet_penalty_only_applies_while_standing():
    motion_command = SimpleNamespace(
        motion_cfg=SimpleNamespace(recovery_shoulder_height_threshold=1.0),
        recovery_active_mask=lambda threshold: torch.tensor([False, True]),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        feet_indices=torch.tensor([0, 1]),
        simulator=SimpleNamespace(
            _rigid_body_pos=torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
            contact_forces=torch.tensor(
                [
                    [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
                    [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
                ],
                dtype=torch.float32,
            ),
        ),
    )

    with patch.object(wbt_reward_terms, "_get_motion_command_and_assert_type", return_value=motion_command):
        penalty = wbt_reward_terms.close_feet_penalty(env, close_feet_threshold=0.12, shoulder_height_threshold=None)
    assert penalty.tolist() == [1.0, 0.0]


def test_joint_action_rate_penalty_targets_requested_group():
    env = SimpleNamespace(
        knee_joint_indices=torch.tensor([0, 2]),
        ankle_joint_indices=torch.tensor([1, 3]),
        action_manager=SimpleNamespace(
            action=torch.tensor([[1.0, 10.0, 2.0, 20.0]]),
            prev_action=torch.zeros((1, 4)),
        ),
    )

    knee_penalty = wbt_reward_terms.joint_action_rate_penalty(env, joint_group="knee")
    ankle_penalty = wbt_reward_terms.joint_action_rate_penalty(env, joint_group="ankle")

    assert knee_penalty.item() == pytest.approx(5.0)
    assert ankle_penalty.item() == pytest.approx(500.0)


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
            "shoulder_height_threshold": 1.0,
            "max_consecutive_bad_tracking_steps": 2,
        },
    )
    term = wbt_termination_terms.RecoveryAwareBadTracking(cfg, env)

    with patch.object(wbt_termination_terms.BadTrackingZOnly, "__call__", return_value=torch.tensor([True, True])):
        first = term(env)
        second = term(env)

    assert first.tolist() == [False, True]
    assert second.tolist() == [True, True]


def test_bad_tracking_orientation_uses_radian_quaternion_error():
    motion_command = SimpleNamespace(
        motion_cfg=SimpleNamespace(body_names_to_track=["left_ankle_roll_link"]),
        ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.5, 0.8660254]], dtype=torch.float32),
        ref_pos_w=torch.zeros((1, 3), dtype=torch.float32),
        robot_ref_pos_w=torch.zeros((1, 3), dtype=torch.float32),
        body_pos_relative_w=torch.zeros((1, 1, 3), dtype=torch.float32),
        robot_body_pos_w=torch.zeros((1, 1, 3), dtype=torch.float32),
        motion=SimpleNamespace(has_object=False),
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        command_manager=SimpleNamespace(get_state=lambda name: motion_command),
    )
    cfg = TerminationTermCfg(
        func="unused",
        params={
            "bad_ref_pos_threshold": 0.5,
            "bad_ref_ori_threshold": 0.8,
            "bad_motion_body_pos_threshold": 0.25,
            "body_names_to_track": ["left_ankle_roll_link"],
            "bad_motion_body_pos_body_names": ["left_ankle_roll_link"],
            "bad_object_pos_threshold": 0.25,
            "bad_object_ori_threshold": 0.8,
        },
    )
    term = wbt_termination_terms.BadTrackingZOnly(cfg, env)

    assert term.bad_ref_ori(motion_command).tolist() == [True]


def test_motion_command_low_kinetic_sampler_uses_pre_transition_motion():
    original_joint_vel = torch.tensor(
        [
            [0.0],
            [2.0],
            [0.2],
            [3.0],
        ],
        dtype=torch.float32,
    )

    class FakeMotionLoader:
        def __init__(self, *args, **kwargs):
            self._joint_vel = original_joint_vel.clone()
            self._joint_pos = torch.zeros((4, 1), dtype=torch.float32)
            self._body_indexes = torch.arange(3, dtype=torch.long)
            self._joint_indexes = torch.arange(1, dtype=torch.long)
            self.time_step_total = 4
            self.has_object = False

        @property
        def joint_vel(self):
            return self._joint_vel

    def mutate_motion_with_transition(self, prepend: bool) -> None:
        extra = torch.full((2, 1), 0.01 if prepend else 0.02, dtype=torch.float32)
        self.motion._joint_vel = torch.cat([extra, self.motion._joint_vel], dim=0) if prepend else torch.cat(
            [self.motion._joint_vel, extra], dim=0
        )
        self.motion.time_step_total = self.motion._joint_vel.shape[0]

    motion_cfg = MotionConfig(
        motion_file="unused.npz",
        body_name_ref=["torso_link"],
        body_names_to_track=["left_shoulder_roll_link", "right_shoulder_roll_link"],
        sampling_strategy=MotionConfig.MotionSamplingStrategy.LOW_KINETIC,
        enable_default_pose_prepend=True,
        enable_default_pose_append=True,
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        dt=0.02,
        viewer=False,
        simulator=SimpleNamespace(
            _body_list=["torso_link", "left_shoulder_roll_link", "right_shoulder_roll_link"],
            dof_names=["joint0"],
        ),
    )
    cfg = CommandTermCfg(func="unused", params={"motion_config": motion_cfg})

    with patch("holosoma.managers.command.terms.wbt.MotionLoader", FakeMotionLoader), patch.object(
        MotionCommand,
        "_maybe_add_default_pose_transition",
        mutate_motion_with_transition,
    ):
        term = MotionCommand(cfg, env)
        term.setup()

    assert term.motion.time_step_total == 8
    assert term.low_kinetic_anchor_sampler is not None
    assert term.low_kinetic_anchor_sampler.anchor_timesteps.max().item() < original_joint_vel.shape[0]
