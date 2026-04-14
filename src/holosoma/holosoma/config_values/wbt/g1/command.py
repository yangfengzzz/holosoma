"""Whole Body Tracking command presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig
from holosoma.utils.kfa_parity import (
    KFA_OPTIONAL_REGRESSION_CLIP_IDS,
    KFA_PRIMARY_REGRESSION_CLIP_ID,
    KFA_RELEASED_MOTION_ROOT,
    KFA_TRAINING_EXAMPLE_CLIP_ID,
    get_kfa_released_motion_path,
)


kfa_motion_file_placeholder = get_kfa_released_motion_path(KFA_TRAINING_EXAMPLE_CLIP_ID)

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    body_names_to_track=[
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=True,
    noise_to_initial_pose=init_pose_config,
)

motion_config_w_object = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
)

motion_config_recovery = replace(
    motion_config,
    motion_file=kfa_motion_file_placeholder,
    sampling_strategy=MotionConfig.MotionSamplingStrategy.LOW_KINETIC,
    start_at_timestep_zero_prob=0.2,
    freeze_at_timestep_zero_prob=0.95,
    enable_default_pose_prepend=True,
    enable_default_pose_append=True,
    recovery_shoulder_height_threshold=1.0,
    recovery_init_dataset=MotionConfig.RecoveryInitDatasetConfig(
        enabled=True,
        dataset_path="./artifacts/recovery_init/g1_ground_v1.npz",
        sample_probability=0.5,
        augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
        dataset_kind=MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT,
    ),
    low_kinetic_sampling=MotionConfig.LowKineticSamplingConfig(
        anchor_window_size=15,
        min_anchor_spacing=10,
        ema_alpha=0.05,
        uniform_ratio=0.1,
        failure_weight=1.0,
    ),
)

motion_config_stand = replace(
    motion_config,
    motion_file=kfa_motion_file_placeholder,
    # Keep the paper-facing low-dynamic Ground clip and reward surface, but
    # start from a gentler reset curriculum so the stand policy can first solve
    # the tracking problem before we remove easy-start scaffolding in ablations.
    sampling_strategy=MotionConfig.MotionSamplingStrategy.LOW_KINETIC,
    start_at_timestep_zero_prob=0.2,
    freeze_at_timestep_zero_prob=0.95,
    enable_default_pose_prepend=True,
    enable_default_pose_append=True,
    recovery_shoulder_height_threshold=1.0,
    low_kinetic_sampling=MotionConfig.LowKineticSamplingConfig(
        anchor_window_size=15,
        min_anchor_spacing=10,
        ema_alpha=0.05,
        uniform_ratio=0.1,
        failure_weight=1.0,
    ),
)

motion_config_recovery_debug_base = replace(
    motion_config_stand,
    sampling_strategy=MotionConfig.MotionSamplingStrategy.UNIFORM,
    start_at_timestep_zero_prob=0.0,
    freeze_at_timestep_zero_prob=0.0,
    enable_default_pose_prepend=False,
    enable_default_pose_append=False,
    recovery_init_dataset=MotionConfig.RecoveryInitDatasetConfig(
        enabled=True,
        dataset_path="./artifacts/recovery_init/g1_ground_v1.npz",
        sample_probability=0.1,
        augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
        dataset_kind=MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT,
    ),
)

motion_config_recovery_low_kinetic = replace(
    motion_config_recovery,
    recovery_init_dataset=replace(
        motion_config_recovery.recovery_init_dataset,
        sample_probability=0.25,
    ),
)

motion_config_kfa_1307_release_base = replace(
    motion_config,
    motion_file=get_kfa_released_motion_path(KFA_PRIMARY_REGRESSION_CLIP_ID),
    root_body_names=["pelvis"],
    shoulders_body_names=["left_shoulder_roll_link", "right_shoulder_roll_link"],
    feet_body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
    use_adaptive_timesteps_sampler=False,
    start_at_timestep_zero_prob=0.0,
    freeze_at_timestep_zero_prob=0.0,
    enable_default_pose_prepend=False,
    enable_default_pose_append=False,
    sampling_strategy=MotionConfig.MotionSamplingStrategy.ADAPTIVE,
    standing_like_reset_enabled=True,
    reset_mode_weights=(1.0, 1.0),
    standing_like_reset_joint_noise_scale=0.0,
    standing_like_reset_root_noise_scale=0.0,
    release_standing_relative_target=True,
    standing_like_reset_use_diverse_quaternions=True,
    standing_like_reset_diverse_candidate_count=2048,
    recovery_shoulder_height_threshold=1.0,
    recovery_init_dataset=MotionConfig.RecoveryInitDatasetConfig(
        enabled=True,
        dataset_path="./artifacts/recovery_init/g1_ground_v1.npz",
        sample_probability=0.0,
        augmentation_mode=MotionConfig.RecoveryInitDatasetConfig.AugmentationMode.ROTATION_RECOMBINATION,
        dataset_kind=MotionConfig.RecoveryInitDatasetConfig.DatasetKind.RECOVERY_INIT,
    ),
)

motion_config_kfa_1307_stage1 = replace(
    motion_config_kfa_1307_release_base,
)

motion_config_kfa_1307_stage2 = replace(
    motion_config_kfa_1307_release_base,
    sampling_strategy=MotionConfig.MotionSamplingStrategy.ADAPTIVE,
)

motion_config_kfa_1307_stage3 = replace(
    motion_config_kfa_1307_stage2,
    noise_to_initial_pose=replace(
        init_pose_config,
        root_pos=[0.15, 0.15, 0.15],
        root_lin_vel=[0.75, 0.75, 0.3],
        root_ang_vel=[0.78, 0.78, 1.17],
    ),
    standing_like_reset_joint_noise_scale=0.0,
    standing_like_reset_root_noise_scale=0.0,
)

g1_29dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)

g1_29dof_wbt_command_w_object = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object,
            },
        )
    },
)

g1_29dof_wbt_recovery_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_recovery,
            },
        )
    },
)

g1_29dof_wbt_stand_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_stand,
            },
        )
    },
)

g1_29dof_wbt_recovery_debug_base_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_recovery_debug_base,
            },
        )
    },
)

g1_29dof_wbt_recovery_low_kinetic_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_recovery_low_kinetic,
            },
        )
    },
)

g1_29dof_kfa_1307_stage1_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": motion_config_kfa_1307_stage1},
        )
    },
)

g1_29dof_kfa_1307_stage2_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": motion_config_kfa_1307_stage2},
        )
    },
)

g1_29dof_kfa_1307_stage3_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": motion_config_kfa_1307_stage3},
        )
    },
)

__all__ = [
    "KFA_OPTIONAL_REGRESSION_CLIP_IDS",
    "KFA_PRIMARY_REGRESSION_CLIP_ID",
    "KFA_RELEASED_MOTION_ROOT",
    "KFA_TRAINING_EXAMPLE_CLIP_ID",
    "g1_29dof_wbt_command",
    "g1_29dof_kfa_1307_stage1_command",
    "g1_29dof_kfa_1307_stage2_command",
    "g1_29dof_kfa_1307_stage3_command",
    "g1_29dof_wbt_recovery_command",
    "g1_29dof_wbt_recovery_debug_base_command",
    "g1_29dof_wbt_recovery_low_kinetic_command",
    "g1_29dof_wbt_stand_command",
    "g1_29dof_wbt_command_w_object",
    "get_kfa_released_motion_path",
    "kfa_motion_file_placeholder",
]
