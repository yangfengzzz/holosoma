"""Whole Body Tracking command presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

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
    sampling_strategy=MotionConfig.MotionSamplingStrategy.LOW_KINETIC,
    recovery_shoulder_height_threshold=1.0,
    recovery_init_dataset=MotionConfig.RecoveryInitDatasetConfig(
        enabled=False,
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

kfa_motion_file_placeholder = "./datasets/KungFuAthleteBot/org_smoothed_mj/1317_mj.npz"

motion_config_stand = replace(
    motion_config,
    motion_file=kfa_motion_file_placeholder,
    recovery_shoulder_height_threshold=1.0,
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

__all__ = [
    "g1_29dof_wbt_command",
    "g1_29dof_wbt_recovery_command",
    "g1_29dof_wbt_stand_command",
    "g1_29dof_wbt_command_w_object",
    "kfa_motion_file_placeholder",
]
