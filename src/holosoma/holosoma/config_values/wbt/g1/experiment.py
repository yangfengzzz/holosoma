from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

g1_29dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.3, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.4, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.85, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.7, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.60, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.45, "inf"],
        },
    ),
)

g1_29dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fast_sac_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,  # For motion tracking, high gamma + high num_steps is better
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

g1_29dof_wbt_w_object = replace(
    g1_29dof_wbt,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(
            robot.g1_29dof_w_object.asset,
            enable_self_collisions=True,
        ),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_fast_sac_w_object = replace(
    g1_29dof_wbt_fast_sac,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_recovery_fast_sac = replace(
    g1_29dof_wbt_fast_sac,
    training=replace(
        g1_29dof_wbt_fast_sac.training,
        project="WholeBodyTrackingRecovery",
        name="g1_29dof_wbt_recovery_fast_sac_manager",
    ),
    robot=replace(
        robot.g1_29dof_kfa,
        control=replace(
            robot.g1_29dof_kfa.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof_kfa.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof_kfa.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    command=command.g1_29dof_wbt_recovery_command,
    termination=termination.g1_29dof_wbt_recovery_termination,
    reward=reward.g1_29dof_wbt_recovery_fast_sac_reward,
)

g1_29dof_wbt_recovery_debug_base_fast_sac = replace(
    g1_29dof_wbt_recovery_fast_sac,
    training=replace(
        g1_29dof_wbt_recovery_fast_sac.training,
        name="g1_29dof_wbt_recovery_debug_base_fast_sac_manager",
    ),
    command=command.g1_29dof_wbt_recovery_debug_base_command,
    reward=reward.g1_29dof_wbt_recovery_debug_fast_sac_reward,
)

g1_29dof_wbt_recovery_low_kinetic_fast_sac = replace(
    g1_29dof_wbt_recovery_fast_sac,
    training=replace(
        g1_29dof_wbt_recovery_fast_sac.training,
        name="g1_29dof_wbt_recovery_low_kinetic_fast_sac_manager",
    ),
    command=command.g1_29dof_wbt_recovery_low_kinetic_command,
    reward=reward.g1_29dof_wbt_recovery_debug_fast_sac_reward,
)

g1_29dof_wbt_recovery_only_fast_sac = replace(
    g1_29dof_wbt_recovery_fast_sac,
    training=replace(
        g1_29dof_wbt_recovery_fast_sac.training,
        name="g1_29dof_wbt_recovery_only_fast_sac_manager",
    ),
    reward=reward.g1_29dof_wbt_recovery_only_fast_sac_reward,
)

g1_29dof_wbt_recovery_slip3_fast_sac = replace(
    g1_29dof_wbt_recovery_fast_sac,
    training=replace(
        g1_29dof_wbt_recovery_fast_sac.training,
        name="g1_29dof_wbt_recovery_slip3_fast_sac_manager",
    ),
    reward=reward.g1_29dof_wbt_recovery_slip3_fast_sac_reward,
)

g1_29dof_wbt_recovery_slip5_fast_sac = replace(
    g1_29dof_wbt_recovery_fast_sac,
    training=replace(
        g1_29dof_wbt_recovery_fast_sac.training,
        name="g1_29dof_wbt_recovery_slip5_fast_sac_manager",
    ),
    reward=reward.g1_29dof_wbt_recovery_slip5_fast_sac_reward,
)

g1_29dof_wbt_stand_fast_sac = replace(
    g1_29dof_wbt_fast_sac,
    training=replace(
        g1_29dof_wbt_fast_sac.training,
        name="g1_29dof_wbt_stand_fast_sac_manager",
    ),
    robot=replace(
        robot.g1_29dof_kfa,
        control=replace(
            robot.g1_29dof_kfa.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof_kfa.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof_kfa.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    command=command.g1_29dof_wbt_stand_command,
    termination=termination.g1_29dof_wbt_stand_termination,
    reward=reward.g1_29dof_wbt_stand_fast_sac_reward,
)

g1_29dof_kfa_1307_stage1_fast_sac = replace(
    g1_29dof_wbt_fast_sac,
    training=replace(
        g1_29dof_wbt_fast_sac.training,
        project="WholeBodyTrackingReleaseParity",
        name="g1_29dof_kfa_1307_stage1_fast_sac_manager",
    ),
    robot=replace(
        robot.g1_29dof_kfa,
        control=replace(
            robot.g1_29dof_kfa.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof_kfa.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof_kfa.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    command=command.g1_29dof_kfa_1307_stage1_command,
    termination=termination.g1_29dof_kfa_1307_stage1_termination,
    randomization=randomization.g1_29dof_kfa_1307_stage1_randomization,
    reward=reward.g1_29dof_kfa_1307_stage1_fast_sac_reward,
    observation=observation.g1_29dof_kfa_release_standing_observation,
)

g1_29dof_kfa_1307_stage2_fast_sac = replace(
    g1_29dof_kfa_1307_stage1_fast_sac,
    training=replace(
        g1_29dof_kfa_1307_stage1_fast_sac.training,
        name="g1_29dof_kfa_1307_stage2_fast_sac_manager",
    ),
    command=command.g1_29dof_kfa_1307_stage2_command,
    termination=termination.g1_29dof_kfa_1307_stage2_termination,
    randomization=randomization.g1_29dof_kfa_1307_stage2_randomization,
    reward=reward.g1_29dof_kfa_1307_stage2_fast_sac_reward,
)

g1_29dof_kfa_1307_stage3_fast_sac = replace(
    g1_29dof_kfa_1307_stage2_fast_sac,
    training=replace(
        g1_29dof_kfa_1307_stage2_fast_sac.training,
        name="g1_29dof_kfa_1307_stage3_fast_sac_manager",
    ),
    command=command.g1_29dof_kfa_1307_stage3_command,
    termination=termination.g1_29dof_kfa_1307_stage3_termination,
    randomization=randomization.g1_29dof_kfa_1307_stage3_randomization,
    reward=reward.g1_29dof_kfa_1307_stage3_fast_sac_reward,
    terrain=terrain.terrain_locomotion_mix,
)

__all__ = [
    "g1_29dof_kfa_1307_stage1_fast_sac",
    "g1_29dof_kfa_1307_stage2_fast_sac",
    "g1_29dof_kfa_1307_stage3_fast_sac",
    "g1_29dof_wbt",
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_recovery_debug_base_fast_sac",
    "g1_29dof_wbt_recovery_fast_sac",
    "g1_29dof_wbt_recovery_low_kinetic_fast_sac",
    "g1_29dof_wbt_recovery_only_fast_sac",
    "g1_29dof_wbt_recovery_slip3_fast_sac",
    "g1_29dof_wbt_recovery_slip5_fast_sac",
    "g1_29dof_wbt_stand_fast_sac",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_29dof_wbt_w_object",
]

"""
Example 1: Robot only:
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt

Example 2: Robot+Object:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object

Example 3: Robot+Terrain:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj" \
  --command.setup_terms.motion_command.params.motion_config.motion_file\
="holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz" \
  --simulator.config.scene.env_spacing=0.0
"""
