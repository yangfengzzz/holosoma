from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import PropertyMock

import numpy as np
import pytest
import torch

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.termination import TerminationTermCfg
from holosoma.managers.command.terms.wbt import (
    LowKineticAnchorSampler,
    MotionCommand,
    ReleaseLowKineticEnergySampler,
    select_most_diverse_quaternions,
)
from holosoma.managers.reward.terms import wbt as wbt_reward_terms
from holosoma.managers.termination.terms import wbt as wbt_termination_terms
from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
from holosoma.config_values.experiment import DEFAULTS as EXPERIMENT_DEFAULTS
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


def test_release_low_kinetic_energy_sampler_prefers_low_energy_frames():
    joint_vel = torch.tensor([[0.0], [2.0], [0.1]], dtype=torch.float32)
    body_lin_vel = torch.tensor(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    body_ang_vel = torch.tensor(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    sampler = ReleaseLowKineticEnergySampler(joint_vel, body_lin_vel, body_ang_vel, "cpu")
    probs = sampler.sampling_probabilities

    assert probs.shape == (3,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert probs[0].item() > probs[1].item()


def test_select_most_diverse_quaternions_uses_random_seed_index():
    quats = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    with patch("torch.randint", return_value=torch.tensor([1])):
        selected = select_most_diverse_quaternions(quats, 2)
    assert selected[0].item() == 1


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


def test_self_collision_cost_counts_all_bodies_without_filter():
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        simulator=SimpleNamespace(
            _body_list=["pelvis", "arm"],
            contact_forces_history=torch.tensor(
                [[[[0.0, 0.0, 0.0], [11.0, 0.0, 0.0]], [[12.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
                dtype=torch.float32,
            ),
        ),
    )

    penalty = wbt_reward_terms.self_collision_cost(env, force_threshold=10.0)

    assert penalty.item() == pytest.approx(2.0)


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


def test_feet_slip_penalty_uses_z_force_threshold_only():
    env = SimpleNamespace(
        feet_indices=torch.tensor([0, 1]),
        simulator=SimpleNamespace(
            contact_forces=torch.tensor(
                [[[9.0, 0.0, 0.2], [0.0, 0.0, 9.0]]],
                dtype=torch.float32,
            ),
            _rigid_body_vel=torch.tensor(
                [[[3.0, 4.0, 0.0], [6.0, 8.0, 0.0]]],
                dtype=torch.float32,
            ),
        ),
    )

    penalty = wbt_reward_terms.feet_slip_penalty(env, contact_force_threshold=8.0)
    assert penalty.item() == pytest.approx(10.0)


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
    motion_command = SimpleNamespace(
        motion_cfg=SimpleNamespace(body_names_to_track=["left_ankle_roll_link", "right_ankle_roll_link"]),
        recovery_active_mask=lambda threshold: torch.tensor([True, False]),
    )
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

    with patch.object(wbt_termination_terms.BadTracking, "__call__", return_value=torch.tensor([True, True])):
        first = term(env)
        second = term(env)

    assert first.tolist() == [False, True]
    assert second.tolist() == [True, True]


def test_release_parity_tolerant_tracking_uses_grace_window_only_for_standing_tasks():
    motion_command = SimpleNamespace(
        is_standing_task=torch.tensor([True, False], dtype=torch.bool),
        ref_pos_w=torch.zeros((2, 3), dtype=torch.float32),
        robot_ref_pos_w=torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=torch.float32),
        ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=torch.float32),
        body_pos_relative_w=torch.zeros((2, 1, 3), dtype=torch.float32),
        robot_body_pos_w=torch.zeros((2, 1, 3), dtype=torch.float32),
        joint_pos=torch.zeros((2, 8), dtype=torch.float32),
        robot_joint_pos=torch.zeros((2, 8), dtype=torch.float32),
        motion_cfg=SimpleNamespace(body_names_to_track=["left_ankle_roll_link"]),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        dt=0.1,
        extras={"log": {}},
        command_manager=SimpleNamespace(get_state=lambda name: motion_command),
    )
    cfg = TerminationTermCfg(
        func="unused",
        params={
            "bad_tracking_time_threshold_s": 0.2,
            "predicate_specs": [
                {"name": "anchor_pos", "kind": "anchor_pos", "params": {"threshold": 0.5}},
            ],
        },
    )
    term = wbt_termination_terms.ReleaseParityTolerantTracking(cfg, env)

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
    term = wbt_termination_terms.BadTracking(cfg, env)

    assert term.bad_ref_ori(motion_command).tolist() == [True]


def test_bad_tracking_position_uses_full_vector_distance():
    motion_command = SimpleNamespace(
        motion_cfg=SimpleNamespace(body_names_to_track=["left_ankle_roll_link"]),
        ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        ref_pos_w=torch.tensor([[0.4, 0.4, 0.0]], dtype=torch.float32),
        robot_ref_pos_w=torch.zeros((1, 3), dtype=torch.float32),
        body_pos_relative_w=torch.tensor([[[0.2, 0.2, 0.0]]], dtype=torch.float32),
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
    term = wbt_termination_terms.BadTracking(cfg, env)

    assert term.bad_ref_pos(motion_command).tolist() == [True]
    assert term.bad_motion_body_pos(motion_command).tolist() == [True]


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


def test_motion_command_recovery_reset_mask_respects_sample_probability_extremes():
    term = object.__new__(MotionCommand)
    term.device = "cpu"
    term.recovery_init_dataset = object()
    term.motion_cfg = SimpleNamespace(
        recovery_init_dataset=SimpleNamespace(enabled=True, sample_probability=0.0),
    )
    term._env = SimpleNamespace(is_evaluating=False)
    env_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    assert not term._sample_recovery_reset_mask(env_ids).any()

    term.motion_cfg.recovery_init_dataset.sample_probability = 1.0
    assert term._sample_recovery_reset_mask(env_ids).all()


def test_motion_command_setup_uses_explicit_standing_body_groups():
    class FakeMotionLoader:
        def __init__(self, *args, **kwargs):
            self._joint_vel = torch.zeros((4, 1), dtype=torch.float32)
            self._joint_pos = torch.zeros((4, 1), dtype=torch.float32)
            self._body_indexes = torch.arange(4, dtype=torch.long)
            self._joint_indexes = torch.arange(1, dtype=torch.long)
            self.time_step_total = 4
            self.has_object = False

        @property
        def joint_vel(self):
            return self._joint_vel

    motion_cfg = MotionConfig(
        motion_file="unused.npz",
        body_name_ref=["torso_link"],
        body_names_to_track=["right_ankle_roll_link", "torso_link", "pelvis", "left_shoulder_roll_link"],
        root_body_names=["pelvis"],
        shoulders_body_names=["left_shoulder_roll_link"],
        feet_body_names=["right_ankle_roll_link"],
        sampling_strategy=MotionConfig.MotionSamplingStrategy.UNIFORM,
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        dt=0.02,
        viewer=False,
        simulator=SimpleNamespace(
            _body_list=["right_ankle_roll_link", "torso_link", "pelvis", "left_shoulder_roll_link"],
            dof_names=["joint0"],
        ),
    )
    cfg = CommandTermCfg(func="unused", params={"motion_config": motion_cfg})

    with patch("holosoma.managers.command.terms.wbt.MotionLoader", FakeMotionLoader), patch.object(
        MotionCommand,
        "_maybe_add_default_pose_transition",
        lambda self, prepend: None,
    ):
        term = MotionCommand(cfg, env)
        term.setup()

    assert term.root_index == 2
    assert term.shoulders_indexes == [3]
    assert term.feet_indexes == [0]


def test_motion_command_standing_like_reset_mask_uses_reset_mode_weights():
    term = object.__new__(MotionCommand)
    term.device = "cpu"
    term.recovery_init_dataset = object()
    term.motion_cfg = SimpleNamespace(
        standing_like_reset_enabled=True,
        reset_mode_weights=(1.0, 3.0),
        recovery_init_dataset=SimpleNamespace(enabled=True),
    )
    term._env = SimpleNamespace(is_evaluating=False)
    env_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    with patch("torch.rand", return_value=torch.tensor([0.2, 0.8, 0.7], dtype=torch.float32)):
        mask = term._sample_standing_like_reset_mask(env_ids)

    assert mask.tolist() == [True, False, True]


def test_motion_command_recovery_reset_mask_uses_random_draws_for_mixed_probability():
    term = object.__new__(MotionCommand)
    term.device = "cpu"
    term.recovery_init_dataset = object()
    term.motion_cfg = SimpleNamespace(
        recovery_init_dataset=SimpleNamespace(enabled=True, sample_probability=0.5),
    )
    term._env = SimpleNamespace(is_evaluating=False)
    env_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    with patch("torch.rand", return_value=torch.tensor([0.2, 0.8, 0.49], dtype=torch.float32)):
        mask = term._sample_recovery_reset_mask(env_ids)

    assert mask.tolist() == [True, False, True]


def test_motion_command_reset_applies_recovery_batch_to_selected_envs():
    term = object.__new__(MotionCommand)
    term.num_envs = 2
    term.device = "cpu"
    term.time_steps = torch.zeros(2, dtype=torch.long)
    term.last_reset_used_recovery = torch.zeros(2, dtype=torch.bool)
    term._reset_recovery_count = torch.tensor(0, dtype=torch.long)
    term._reset_motion_count = torch.tensor(0, dtype=torch.long)
    term._terminated_in_recovery_count = torch.tensor(0, dtype=torch.long)
    term._terminated_outside_recovery_count = torch.tensor(0, dtype=torch.long)
    term._clip_end_reset_count = torch.tensor(0, dtype=torch.long)
    term._regular_reset_count = torch.tensor(0, dtype=torch.long)
    term._standing_like_diverse_indices = torch.zeros(0, dtype=torch.long)
    term.motion = SimpleNamespace(time_step_total=4, has_object=False)
    term.init_pose_cfg = NoiseToInitialPoseConfig()
    term.motion_cfg = SimpleNamespace(
        start_at_timestep_zero_prob=0.0,
        recovery_init_dataset=SimpleNamespace(
            enabled=True,
            sample_probability=0.5,
            augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
        )
    )
    simulator = SimpleNamespace(
        dof_pos_limits=torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32),
        dof_pos=torch.zeros((2, 2), dtype=torch.float32),
        dof_vel=torch.zeros((2, 2), dtype=torch.float32),
        robot_root_states=torch.zeros((2, 13), dtype=torch.float32),
        scene=SimpleNamespace(env_origins=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float32)),
    )
    term._env = SimpleNamespace(simulator=simulator)
    term.recovery_init_dataset = SimpleNamespace(
        sample=lambda batch_size, augmentation_mode: SimpleNamespace(
            dof_pos=torch.tensor([[0.7, -0.7]], dtype=torch.float32),
            dof_vel=torch.tensor([[0.3, -0.3]], dtype=torch.float32),
            root_states=torch.tensor(
                [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]],
                dtype=torch.float32,
            ),
        )
    )

    with patch.object(MotionCommand, "_update_sampling_failures", return_value=None), patch.object(
        MotionCommand, "_sample_reference_timesteps", return_value=torch.zeros(2, dtype=torch.long)
    ), patch.object(
        MotionCommand, "_sample_standing_like_reset_mask", return_value=torch.tensor([False, False], dtype=torch.bool)
    ), patch.object(
        MotionCommand, "_sample_recovery_reset_mask", return_value=torch.tensor([False, True], dtype=torch.bool)
    ), patch.object(
        MotionCommand, "root_pos_w", new_callable=PropertyMock, return_value=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    ), patch.object(
        MotionCommand,
        "root_quat_w",
        new_callable=PropertyMock,
        return_value=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
    ), patch.object(
        MotionCommand, "root_lin_vel_w", new_callable=PropertyMock, return_value=torch.zeros((2, 3), dtype=torch.float32)
    ), patch.object(
        MotionCommand, "root_ang_vel_w", new_callable=PropertyMock, return_value=torch.zeros((2, 3), dtype=torch.float32)
    ), patch.object(
        MotionCommand, "joint_pos", new_callable=PropertyMock, return_value=torch.tensor([[0.1, -0.1], [0.2, -0.2]], dtype=torch.float32)
    ), patch.object(
        MotionCommand, "joint_vel", new_callable=PropertyMock, return_value=torch.zeros((2, 2), dtype=torch.float32)
    ):
        term.reset(torch.tensor([0, 1], dtype=torch.long))

    assert term.last_reset_used_recovery.tolist() == [False, True]
    assert torch.allclose(simulator.dof_pos[0], torch.tensor([0.1, -0.1]))
    assert torch.allclose(simulator.dof_pos[1], torch.tensor([0.7, -0.7]))
    assert torch.allclose(simulator.dof_vel[1], torch.tensor([0.3, -0.3]))
    assert torch.allclose(simulator.robot_root_states[0, :3], torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(simulator.robot_root_states[1, :3], torch.tensor([11.0, 2.0, 3.0]))
    assert torch.allclose(simulator.robot_root_states[1, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_motion_command_standing_like_reset_preserves_motion_xy_and_swaps_standing_state():
    term = object.__new__(MotionCommand)
    term.num_envs = 1
    term.device = "cpu"
    term.time_steps = torch.zeros(1, dtype=torch.long)
    term.last_reset_used_recovery = torch.zeros(1, dtype=torch.bool)
    term.is_standing_task = torch.zeros(1, dtype=torch.bool)
    term._reset_recovery_count = torch.tensor(0, dtype=torch.long)
    term._reset_motion_count = torch.tensor(0, dtype=torch.long)
    term._reset_standing_like_count = torch.tensor(0, dtype=torch.long)
    term._terminated_in_recovery_count = torch.tensor(0, dtype=torch.long)
    term._terminated_outside_recovery_count = torch.tensor(0, dtype=torch.long)
    term._clip_end_reset_count = torch.tensor(0, dtype=torch.long)
    term._regular_reset_count = torch.tensor(0, dtype=torch.long)
    term._standing_like_diverse_indices = torch.zeros(0, dtype=torch.long)
    term.motion = SimpleNamespace(time_step_total=4, has_object=False)
    term.init_pose_cfg = NoiseToInitialPoseConfig()
    term.motion_cfg = SimpleNamespace(
        start_at_timestep_zero_prob=0.0,
        recovery_init_dataset=SimpleNamespace(
            enabled=True,
            sample_probability=0.0,
            augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.NONE,
        ),
    )
    simulator = SimpleNamespace(
        dof_pos_limits=torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32),
        dof_pos=torch.zeros((1, 2), dtype=torch.float32),
        dof_vel=torch.zeros((1, 2), dtype=torch.float32),
        robot_root_states=torch.zeros((1, 13), dtype=torch.float32),
        scene=SimpleNamespace(env_origins=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)),
    )
    term._env = SimpleNamespace(simulator=simulator)
    term.recovery_init_dataset = SimpleNamespace(
        root_states=torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]], dtype=torch.float32),
        dof_pos=torch.tensor([[0.7, -0.7]], dtype=torch.float32),
        sample=lambda batch_size, augmentation_mode: SimpleNamespace(
            dof_pos=torch.tensor([[0.7, -0.7]], dtype=torch.float32),
            dof_vel=torch.tensor([[0.3, -0.3]], dtype=torch.float32),
            root_states=torch.tensor(
                [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]],
                dtype=torch.float32,
            ),
        ),
    )

    with patch.object(MotionCommand, "_update_sampling_failures", return_value=None), patch.object(
        MotionCommand, "_sample_reference_timesteps", return_value=torch.zeros(1, dtype=torch.long)
    ), patch.object(
        MotionCommand, "_sample_standing_like_reset_mask", return_value=torch.tensor([True], dtype=torch.bool)
    ), patch.object(
        MotionCommand, "_sample_recovery_reset_mask", return_value=torch.tensor([True], dtype=torch.bool)
    ), patch.object(
        MotionCommand, "_apply_standing_like_reset_noise", side_effect=lambda *args: args[1:]
    ), patch.object(
        MotionCommand, "root_pos_w", new_callable=PropertyMock, return_value=torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32)
    ), patch.object(
        MotionCommand, "root_quat_w", new_callable=PropertyMock, return_value=torch.tensor([[0.1, 0.0, 0.0, 0.995]], dtype=torch.float32)
    ), patch.object(
        MotionCommand, "root_lin_vel_w", new_callable=PropertyMock, return_value=torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float32)
    ), patch.object(
        MotionCommand, "root_ang_vel_w", new_callable=PropertyMock, return_value=torch.tensor([[0.5, 0.6, 0.7]], dtype=torch.float32)
    ), patch.object(
        MotionCommand, "joint_pos", new_callable=PropertyMock, return_value=torch.tensor([[0.1, -0.1]], dtype=torch.float32)
    ), patch.object(
        MotionCommand, "joint_vel", new_callable=PropertyMock, return_value=torch.zeros((1, 2), dtype=torch.float32)
    ):
        term.reset(torch.tensor([0], dtype=torch.long))

    assert torch.allclose(simulator.robot_root_states[0, :2], torch.tensor([0.4, 0.5]))
    assert torch.allclose(simulator.robot_root_states[0, 2:3], torch.tensor([3.0]))
    assert torch.allclose(simulator.robot_root_states[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(simulator.robot_root_states[0, 7:10], torch.tensor([4.0, 5.0, 6.0]))
    assert torch.allclose(simulator.dof_pos[0], torch.tensor([0.7, -0.7]))
    assert torch.allclose(simulator.dof_vel[0], torch.tensor([0.0, 0.0]))


def test_motion_command_recovery_reset_mask_approx_matches_mixed_probability():
    term = object.__new__(MotionCommand)
    term.device = "cpu"
    term.recovery_init_dataset = object()
    term.motion_cfg = SimpleNamespace(
        recovery_init_dataset=SimpleNamespace(enabled=True, sample_probability=0.5),
    )
    term._env = SimpleNamespace(is_evaluating=False)
    env_ids = torch.arange(1024, dtype=torch.long)

    torch.manual_seed(0)
    mask = term._sample_recovery_reset_mask(env_ids)

    assert abs(mask.float().mean().item() - 0.5) < 0.08


def test_recovery_reset_is_disabled_during_evaluation():
    term = object.__new__(MotionCommand)
    term.device = "cpu"
    term.recovery_init_dataset = object()
    term.motion_cfg = SimpleNamespace(
        recovery_init_dataset=SimpleNamespace(enabled=True, sample_probability=1.0),
    )
    term._env = SimpleNamespace(is_evaluating=True)
    env_ids = torch.arange(8, dtype=torch.long)

    assert not term._sample_recovery_reset_mask(env_ids).any()


def test_motion_command_update_metrics_reports_reset_event_rates():
    term = object.__new__(MotionCommand)
    term.device = "cpu"
    term.metrics = {}
    term.last_reset_used_recovery = torch.tensor([True, False, True, False], dtype=torch.bool)
    term._reset_recovery_count = torch.tensor(3, dtype=torch.long)
    term._reset_motion_count = torch.tensor(1, dtype=torch.long)
    term._terminated_in_recovery_count = torch.tensor(2, dtype=torch.long)
    term._terminated_outside_recovery_count = torch.tensor(2, dtype=torch.long)
    term._clip_end_reset_count = torch.tensor(1, dtype=torch.long)
    term._regular_reset_count = torch.tensor(4, dtype=torch.long)
    term.adaptive_timesteps_sampler = None
    term.low_kinetic_anchor_sampler = None
    term.release_lke_sampler = None
    term.is_standing_task = torch.tensor([True, False, True, False], dtype=torch.bool)
    term.recovery_active_mask = lambda threshold=None: torch.tensor([True, False, True, False], dtype=torch.bool)
    term.recovery_shoulder_height_gap = lambda: torch.tensor([0.2, 0.4, 1.4, 1.8], dtype=torch.float32)
    term.body_pos_relative_w = torch.zeros((4, 2, 3))
    term.body_quat_relative_w = torch.tensor([[[0.0, 0.0, 0.0, 1.0]] * 2] * 4)

    with patch.object(MotionCommand, "ref_pos_w", new_callable=PropertyMock, return_value=torch.zeros((4, 3))), patch.object(
        MotionCommand, "robot_ref_pos_w", new_callable=PropertyMock, return_value=torch.zeros((4, 3))
    ), patch.object(
        MotionCommand, "ref_quat_w", new_callable=PropertyMock, return_value=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 4)
    ), patch.object(
        MotionCommand, "robot_ref_quat_w", new_callable=PropertyMock, return_value=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 4)
    ), patch.object(
        MotionCommand, "ref_lin_vel_w", new_callable=PropertyMock, return_value=torch.zeros((4, 3))
    ), patch.object(
        MotionCommand, "robot_ref_lin_vel_w", new_callable=PropertyMock, return_value=torch.zeros((4, 3))
    ), patch.object(
        MotionCommand, "ref_ang_vel_w", new_callable=PropertyMock, return_value=torch.zeros((4, 3))
    ), patch.object(
        MotionCommand, "robot_ref_ang_vel_w", new_callable=PropertyMock, return_value=torch.zeros((4, 3))
    ), patch.object(
        MotionCommand, "robot_body_pos_w", new_callable=PropertyMock, return_value=torch.zeros((4, 2, 3))
    ), patch.object(
        MotionCommand, "robot_body_quat_w", new_callable=PropertyMock, return_value=torch.tensor([[[0.0, 0.0, 0.0, 1.0]] * 2] * 4)
    ), patch.object(
        MotionCommand, "body_lin_vel_w", new_callable=PropertyMock, return_value=torch.zeros((4, 2, 3))
    ), patch.object(
        MotionCommand, "robot_body_lin_vel_w", new_callable=PropertyMock, return_value=torch.zeros((4, 2, 3))
    ), patch.object(
        MotionCommand, "body_ang_vel_w", new_callable=PropertyMock, return_value=torch.zeros((4, 2, 3))
    ), patch.object(
        MotionCommand, "robot_body_ang_vel_w", new_callable=PropertyMock, return_value=torch.zeros((4, 2, 3))
    ), patch.object(
        MotionCommand, "joint_pos", new_callable=PropertyMock, return_value=torch.zeros((4, 2))
    ), patch.object(
        MotionCommand, "robot_joint_pos", new_callable=PropertyMock, return_value=torch.zeros((4, 2))
    ), patch.object(
        MotionCommand, "joint_vel", new_callable=PropertyMock, return_value=torch.zeros((4, 2))
    ), patch.object(
        MotionCommand, "robot_joint_vel", new_callable=PropertyMock, return_value=torch.zeros((4, 2))
    ):
        term.update_metrics()

    assert term.metrics["motion/reset_recovery_flag_fraction"].item() == pytest.approx(0.5)
    assert term.metrics["motion/reset_recovery_rate_on_reset"].item() == pytest.approx(0.75)
    assert term.metrics["motion/reset_recovery_count"].item() == 3.0
    assert term.metrics["motion/reset_motion_count"].item() == 1.0
    assert term.metrics["motion/terminated_in_recovery_fraction"].item() == pytest.approx(0.5)
    assert term.metrics["motion/clip_end_reset_fraction"].item() == pytest.approx(0.25)


def test_reward_center_of_mass_only_rewards_single_support():
    motion_command = object.__new__(MotionCommand)
    motion_command.feet_indexes = [0, 1]
    robot_body_pos = torch.tensor(
        [
            [[0.1, 0.0, 0.30], [0.0, 0.0, 0.10]],
            [[0.1, 0.0, 0.20], [0.0, 0.0, 0.19]],
        ],
        dtype=torch.float32,
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        robot_com_pos_w=torch.tensor(
            [
                [0.0, 0.0, 0.20],
                [0.0, 0.0, 0.20],
            ],
            dtype=torch.float32,
        ),
        rigid_body_masses=torch.tensor([1.0, 1.0], dtype=torch.float32),
        simulator=SimpleNamespace(
            _rigid_body_pos=torch.tensor(
                [
                    [[0.0, 0.0, 0.30], [0.2, 0.0, 0.10]],
                    [[0.0, 0.0, 0.20], [0.2, 0.0, 0.19]],
                ],
                dtype=torch.float32,
            )
        ),
        command_manager=SimpleNamespace(get_state=lambda name: motion_command),
    )
    with patch.object(MotionCommand, "robot_body_pos_w", new_callable=PropertyMock, return_value=robot_body_pos):
        reward = wbt_reward_terms.reward_center_of_mass(env, sigma_com=0.1)

    assert reward[0].item() > 0.0
    assert reward[1].item() == pytest.approx(0.0)


def test_recovery_debug_and_low_kinetic_presets_are_staged_from_recovery():
    debug_cfg = EXPERIMENT_DEFAULTS["g1_29dof_wbt_recovery_debug_base_fast_sac"]
    low_kinetic_cfg = EXPERIMENT_DEFAULTS["g1_29dof_wbt_recovery_low_kinetic_fast_sac"]
    full_cfg = EXPERIMENT_DEFAULTS["g1_29dof_wbt_recovery_fast_sac"]

    debug_motion_cfg = debug_cfg.command.setup_terms["motion_command"].params["motion_config"]
    low_kinetic_motion_cfg = low_kinetic_cfg.command.setup_terms["motion_command"].params["motion_config"]
    full_motion_cfg = full_cfg.command.setup_terms["motion_command"].params["motion_config"]

    assert debug_motion_cfg.sampling_strategy == MotionConfig.MotionSamplingStrategy.UNIFORM
    assert debug_motion_cfg.start_at_timestep_zero_prob == 0.0
    assert debug_motion_cfg.freeze_at_timestep_zero_prob == 0.0
    assert debug_motion_cfg.recovery_init_dataset.enabled is True
    assert debug_motion_cfg.recovery_init_dataset.sample_probability == pytest.approx(0.1)
    assert low_kinetic_motion_cfg.sampling_strategy == MotionConfig.MotionSamplingStrategy.LOW_KINETIC
    assert low_kinetic_motion_cfg.recovery_init_dataset.enabled is True
    assert low_kinetic_motion_cfg.recovery_init_dataset.sample_probability == pytest.approx(0.25)
    assert full_motion_cfg.recovery_init_dataset.enabled is True
    assert full_motion_cfg.recovery_init_dataset.sample_probability == pytest.approx(0.5)
