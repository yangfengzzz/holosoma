"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude, quat_rotate_inverse

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def _recovery_mask(env: WholeBodyTrackingManager, shoulder_height_threshold: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.recovery_active_mask(shoulder_height_threshold)


def _resolve_recovery_threshold(
    env: WholeBodyTrackingManager, shoulder_height_threshold: float | None = None
) -> tuple[MotionCommand, float]:
    motion_command = _get_motion_command_and_assert_type(env)
    threshold = (
        motion_command.motion_cfg.recovery_shoulder_height_threshold
        if shoulder_height_threshold is None
        else shoulder_height_threshold
    )
    return motion_command, threshold


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_com_support_alignment_exp(
    env: WholeBodyTrackingManager,
    sigma: float = 0.15,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    body_positions = env.simulator._rigid_body_pos
    masses = env.rigid_body_masses.unsqueeze(0)
    body_mask = torch.zeros_like(masses)
    body_mask[:, env.com_body_indices] = 1.0
    weighted_masses = masses * body_mask
    total_mass = torch.clamp(weighted_masses.sum(dim=1, keepdim=True), min=1e-6)
    robot_com_xy = (body_positions[:, :, :2] * weighted_masses.unsqueeze(-1)).sum(dim=1) / total_mass

    foot_forces = torch.norm(env.simulator.contact_forces[:, env.feet_indices, :], dim=-1)
    foot_heights = env.simulator._rigid_body_pos[:, env.feet_indices, 2]
    in_contact = foot_forces > contact_force_threshold
    lowest_foot_idx = torch.argmin(foot_heights, dim=1)
    support_foot_idx = torch.where(
        torch.any(in_contact, dim=1),
        torch.argmax(torch.where(in_contact, foot_forces, torch.full_like(foot_forces, -1.0)), dim=1),
        lowest_foot_idx,
    )
    support_xy = env.simulator._rigid_body_pos[:, env.feet_indices, :2][
        torch.arange(env.num_envs, device=env.device), support_foot_idx
    ]
    error = torch.sum(torch.square(robot_com_xy - support_xy), dim=-1)
    return torch.exp(-error / sigma**2)


def feet_slip_penalty(env: WholeBodyTrackingManager, contact_force_threshold: float = 1.0) -> torch.Tensor:
    contact_forces = torch.norm(env.simulator.contact_forces[:, env.feet_indices, :], dim=-1)
    foot_vel_xy = env.simulator._rigid_body_vel[:, env.feet_indices, :2]
    contact_mask = contact_forces > contact_force_threshold
    return torch.sum(torch.norm(foot_vel_xy, dim=-1) * contact_mask.float(), dim=1)


def close_feet_penalty(
    env: WholeBodyTrackingManager,
    close_feet_threshold: float = 0.12,
    shoulder_height_threshold: float | None = None,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    left_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[0], :2]
    right_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[1], :2]
    feet_distance = torch.norm(left_foot_xy - right_foot_xy, dim=-1)
    _, resolved_threshold = _resolve_recovery_threshold(env, shoulder_height_threshold)
    recovery_mask = _get_motion_command_and_assert_type(env).recovery_active_mask(resolved_threshold)
    foot_force = torch.norm(env.simulator.contact_forces[:, env.feet_indices, :], dim=-1)
    stance_mask = torch.all(foot_force > contact_force_threshold, dim=1)
    return ((feet_distance < close_feet_threshold) & stance_mask & ~recovery_mask).float()


def root_orientation_penalty(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    gravity = torch.tensor([[0.0, 0.0, -1.0]], device=env.device).repeat(env.num_envs, 1)
    motion_gravity = quat_rotate_inverse(motion_command.ref_quat_w, gravity, w_last=True)
    robot_gravity = quat_rotate_inverse(motion_command.robot_ref_quat_w, gravity, w_last=True)
    return torch.sum(torch.square(motion_gravity[:, :2] - robot_gravity[:, :2]), dim=-1)


def joint_action_rate_penalty(
    env: WholeBodyTrackingManager,
    joint_group: str,
    shoulder_height_threshold: float | None = None,
    recovery_only: bool = False,
) -> torch.Tensor:
    joint_indices = getattr(env, f"{joint_group}_joint_indices")
    actions = env.action_manager.action[:, joint_indices]
    prev_actions = env.action_manager.prev_action[:, joint_indices]
    penalty = torch.sum(torch.square(prev_actions - actions), dim=1)
    if recovery_only:
        _, resolved_threshold = _resolve_recovery_threshold(env, shoulder_height_threshold)
        penalty = penalty * _recovery_mask(env, resolved_threshold).float()
    return penalty


def recovery_relative_shoulder_height_penalty(
    env: WholeBodyTrackingManager,
    shoulder_height_threshold: float | None = None,
) -> torch.Tensor:
    motion_command, resolved_threshold = _resolve_recovery_threshold(env, shoulder_height_threshold)
    shoulder_gap = motion_command.recovery_shoulder_height_gap()
    recovery_mask = motion_command.recovery_active_mask(resolved_threshold)
    return torch.square(torch.clamp(shoulder_gap, min=0.0)) * recovery_mask.float()


def recovery_xy_root_movement_penalty(
    env: WholeBodyTrackingManager,
    shoulder_height_threshold: float | None = None,
) -> torch.Tensor:
    motion_command, resolved_threshold = _resolve_recovery_threshold(env, shoulder_height_threshold)
    recovery_mask = motion_command.recovery_active_mask(resolved_threshold)
    error = torch.sum(torch.square(motion_command.robot_root_pos_w[:, :2] - motion_command.ref_pos_w[:, :2]), dim=-1)
    return error * recovery_mask.float()


def recovery_action_rate_penalty(
    env: WholeBodyTrackingManager,
    shoulder_height_threshold: float | None = None,
) -> torch.Tensor:
    _, resolved_threshold = _resolve_recovery_threshold(env, shoulder_height_threshold)
    recovery_mask = _recovery_mask(env, resolved_threshold)
    return penalty_action_rate(env) * recovery_mask.float()


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
