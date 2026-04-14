"""Whole Body Tracking-specific termination terms."""

from __future__ import annotations

import math
from typing import Any, List

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import quat_error_magnitude
from holosoma.utils.safe_torch_import import torch


#########################################################################################################
## Termination terms
#########################################################################################################
def motion_ends(env, **_) -> torch.Tensor:
    """Terminate if the motion ends."""
    motion_command = env.command_manager.get_state("motion_command")
    return motion_command.time_steps >= motion_command.motion.time_step_total - 2


def bad_anchor_pos(
    env: WholeBodyTrackingManager,
    threshold: float,
    z_only: bool = False,
) -> torch.Tensor:
    motion_command = env.command_manager.get_state("motion_command")
    diff = motion_command.ref_pos_w - motion_command.robot_ref_pos_w
    if z_only:
        return torch.abs(diff[:, 2]) > threshold
    return torch.norm(diff, dim=1) > threshold


def bad_anchor_ori(
    env: WholeBodyTrackingManager,
    threshold: float,
) -> torch.Tensor:
    motion_command = env.command_manager.get_state("motion_command")
    return quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) > threshold


def bad_motion_body_pos(
    env: WholeBodyTrackingManager,
    threshold: float,
    body_names: list[str] | None = None,
    z_only: bool = False,
) -> torch.Tensor:
    motion_command = env.command_manager.get_state("motion_command")
    tracked_names = motion_command.motion_cfg.body_names_to_track
    if body_names:
        body_indexes = torch.tensor([tracked_names.index(name) for name in body_names], device=env.device, dtype=torch.long)
    else:
        body_indexes = torch.arange(len(tracked_names), device=env.device, dtype=torch.long)
    if z_only:
        error = torch.abs(
            motion_command.body_pos_relative_w[:, body_indexes, 2] - motion_command.robot_body_pos_w[:, body_indexes, 2]
        )
    else:
        error = torch.norm(
            motion_command.body_pos_relative_w[:, body_indexes] - motion_command.robot_body_pos_w[:, body_indexes], dim=-1
        )
    return torch.any(error > threshold, dim=-1)


def bad_hip_dof(
    env: WholeBodyTrackingManager,
    threshold: float,
    joint_indices: list[int] | None = None,
) -> torch.Tensor:
    motion_command = env.command_manager.get_state("motion_command")
    hip_joint_indices = joint_indices or [0, 1, 2, 6, 7, 8]
    error = torch.abs(
        motion_command.joint_pos[:, hip_joint_indices] - motion_command.robot_joint_pos[:, hip_joint_indices]
    )
    return torch.any(error > threshold, dim=-1)


class BadTracking(TerminationTermBase):
    """Terminate if the tracking is bad.

    - bad ref pos
    - bad ref ori
    - bad motion body pos
    if has object:
        - bad object pos
        - bad object ori

    When bad tracking is detected, the motion_commmand.AdaptiveTimestepsSampler will be updated.
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self.bad_ref_pos_threshold = cfg.params["bad_ref_pos_threshold"]
        self.bad_ref_ori_threshold = cfg.params["bad_ref_ori_threshold"]

        self.bad_motion_body_pos_body_names = cfg.params["bad_motion_body_pos_body_names"]

        # NOTE: body_names_to_track is shared with command_manager
        self.body_names_to_track = cfg.params["body_names_to_track"]
        self.bad_motion_body_pos_threshold = cfg.params["bad_motion_body_pos_threshold"]
        self.bad_motion_body_pos_body_indexes = self._get_index_of_a_in_b(
            self.bad_motion_body_pos_body_names, self.body_names_to_track, self.env.device
        )

        self.bad_object_pos_threshold = cfg.params["bad_object_pos_threshold"]
        self.bad_object_ori_threshold = cfg.params["bad_object_ori_threshold"]

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        # return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        bad_ref_pos = self.bad_ref_pos(motion_command)
        bad_ref_ori = self.bad_ref_ori(motion_command)
        bad_motion_body_pos = self.bad_motion_body_pos(motion_command)
        bad_tracking = bad_ref_pos | bad_ref_ori | bad_motion_body_pos

        if motion_command.motion.has_object:
            bad_object_pos = self.bad_object_pos(motion_command)
            bad_object_ori = self.bad_object_ori(motion_command)
            bad_tracking |= bad_object_pos | bad_object_ori

        return bad_tracking

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference position is too far from the robot's position."""
        return torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1) > self.bad_ref_pos_threshold

    def bad_ref_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference orientation error exceeds the configured radian threshold."""
        return quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) > self.bad_ref_ori_threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the motion body position is too far from the robot's body position."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.norm(
            motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx], dim=-1
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)

    def bad_object_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object position is too far from the simulator's object position."""
        return (
            torch.norm(motion_command.object_pos_w - motion_command.simulator_object_pos_w, dim=-1)
            > self.bad_object_pos_threshold
        )

    def bad_object_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object orientation is too far from the simulator's object orientation."""
        return (
            quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)
            > self.bad_object_ori_threshold
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class BadTrackingZOnly(BadTracking):
    """BadTracking variant using z-axis-only position checks for parity with BM Wo-State-Estimation."""

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference z position is too far from the robot's z position."""
        z_err = torch.abs(motion_command.ref_pos_w[:, -1] - motion_command.robot_ref_pos_w[:, -1])
        return z_err > self.bad_ref_pos_threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if tracked bodies have too much z-axis position error."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.abs(
            motion_command.body_pos_relative_w[:, body_idx, -1] - motion_command.robot_body_pos_w[:, body_idx, -1]
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)


class RecoveryAwareBadTracking(BadTracking):
    """Bad tracking term with hysteresis while the robot is in recovery mode."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.shoulder_height_threshold = cfg.params.get("shoulder_height_threshold")
        self.max_consecutive_bad_tracking_steps = int(cfg.params.get("max_consecutive_bad_tracking_steps", 8))
        self._recovery_bad_tracking_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        bad_tracking = super().__call__(env, **kwargs)
        motion_command = self.env.command_manager.get_state("motion_command")
        recovery_mask = motion_command.recovery_active_mask(self.shoulder_height_threshold)

        self._recovery_bad_tracking_counter[~recovery_mask] = 0
        self._recovery_bad_tracking_counter[recovery_mask & ~bad_tracking] = 0
        self._recovery_bad_tracking_counter[recovery_mask & bad_tracking] += 1

        recovery_exceeded = self._recovery_bad_tracking_counter >= self.max_consecutive_bad_tracking_steps
        return (bad_tracking & ~recovery_mask) | (bad_tracking & recovery_mask & recovery_exceeded)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._recovery_bad_tracking_counter.zero_()
        else:
            self._recovery_bad_tracking_counter[env_ids] = 0


class ReleaseParityTolerantTracking(TerminationTermBase):
    """Release-style tolerant bad-tracking term with standing-task grace window."""

    _PREDICATE_MAP = {
        "anchor_pos": bad_anchor_pos,
        "anchor_ori": bad_anchor_ori,
        "motion_body_pos": bad_motion_body_pos,
        "hip_dof": bad_hip_dof,
    }

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.bad_tracking_time_threshold_s = float(cfg.params["bad_tracking_time_threshold_s"])
        self.predicate_specs = list(cfg.params["predicate_specs"])
        self.bad_tracking_episode_length = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.final_trigger_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_triggered_terms: dict[str, torch.Tensor] = {}

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        standing_mask = getattr(
            motion_command,
            "is_standing_task",
            torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device),
        )
        combined_bad = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for spec in self.predicate_specs:
            name = spec["name"]
            kind = spec["kind"]
            params = dict(spec.get("params", {}))
            raw_bad = self._PREDICATE_MAP[kind](self.env, **params)
            self.last_triggered_terms[name] = raw_bad
            combined_bad |= raw_bad

        max_bad_steps = max(1, math.ceil(self.bad_tracking_time_threshold_s / self.env.dt))
        self.bad_tracking_episode_length = torch.where(
            combined_bad,
            self.bad_tracking_episode_length + 1,
            torch.zeros_like(self.bad_tracking_episode_length),
        )
        self.final_trigger_mask = torch.where(
            standing_mask,
            self.bad_tracking_episode_length >= max_bad_steps,
            combined_bad,
        )
        return self.final_trigger_mask

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        target_ids = slice(None) if env_ids is None else env_ids
        if env_ids is None:
            self.bad_tracking_episode_length.zero_()
            mask = self.final_trigger_mask
        else:
            self.bad_tracking_episode_length[env_ids] = 0
            mask = self.final_trigger_mask[env_ids]

        extras = getattr(self.env, "extras", None)
        if isinstance(extras, dict):
            log_dict = extras.setdefault("log", {})
            for name, value in self.last_triggered_terms.items():
                count = torch.count_nonzero(value[target_ids] & mask).item()
                extras[f"Episode_Termination/ReleaseParityTolerantTracking/{name}"] = count
                if isinstance(log_dict, dict):
                    log_dict[f"Episode_Termination/ReleaseParityTolerantTracking/{name}"] = count
