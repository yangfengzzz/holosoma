"""Whole Body Tracking termination presets for the G1 robot."""

import math
from dataclasses import replace

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

g1_29dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTracking",
            params={
                # robot tracking
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                # NOTE: body_names_to_track is shared with command_manager
                "body_names_to_track": [
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
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                # object tracking
                # only triggered when has_object=True
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

g1_29dof_wbt_recovery_termination = TerminationManagerCfg(
    terms={
        **g1_29dof_wbt_termination.terms,
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:RecoveryAwareBadTracking",
            params={
                **g1_29dof_wbt_termination.terms["bad_tracking"].params,
                "shoulder_height_threshold": 1.0,
                "max_consecutive_bad_tracking_steps": 8,
            },
        ),
    }
)

g1_29dof_wbt_stand_termination = replace(g1_29dof_wbt_termination)

_release_stage_base_predicates = [
    {
        "name": "anchor_pos_z",
        "kind": "anchor_pos",
        "params": {"threshold": 0.25, "z_only": True},
    },
    {
        "name": "anchor_ori",
        "kind": "anchor_ori",
        "params": {"threshold": 0.8},
    },
    {
        "name": "ee_body_pos_z",
        "kind": "motion_body_pos",
        "params": {
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
            "z_only": True,
        },
    },
]

g1_29dof_kfa_1307_stage1_termination = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination.terms["timeout"],
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:ReleaseParityTolerantTracking",
            params={
                "bad_tracking_time_threshold_s": 3.0,
                "predicate_specs": [
                    {
                        **_release_stage_base_predicates[0],
                        "params": {"threshold": 0.5, "z_only": True},
                    },
                    _release_stage_base_predicates[1],
                    {
                        **_release_stage_base_predicates[2],
                        "params": {
                            **_release_stage_base_predicates[2]["params"],
                            "threshold": 0.4,
                        },
                    },
                ],
            },
        ),
    }
)

g1_29dof_kfa_1307_stage2_termination = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination.terms["timeout"],
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:ReleaseParityTolerantTracking",
            params={
                "bad_tracking_time_threshold_s": 3.0,
                "predicate_specs": [
                    {
                        "name": "anchor_ori",
                        "kind": "anchor_ori",
                        "params": {"threshold": 0.6},
                    },
                    {
                        "name": "anchor_pos",
                        "kind": "anchor_pos",
                        "params": {"threshold": 1.0, "z_only": False},
                    },
                    {
                        "name": "hip_dof",
                        "kind": "hip_dof",
                        "params": {"threshold": math.pi / 6},
                    },
                ],
            },
        ),
    }
)

g1_29dof_kfa_1307_stage3_termination = g1_29dof_kfa_1307_stage2_termination

__all__ = [
    "g1_29dof_kfa_1307_stage1_termination",
    "g1_29dof_kfa_1307_stage2_termination",
    "g1_29dof_kfa_1307_stage3_termination",
    "g1_29dof_wbt_recovery_termination",
    "g1_29dof_wbt_stand_termination",
    "g1_29dof_wbt_termination",
]
