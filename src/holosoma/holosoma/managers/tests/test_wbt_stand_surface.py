from __future__ import annotations

import numpy as np
import torch

from holosoma.managers.command.terms.wbt import MotionLoader


def test_motion_loader_accepts_head_link_body_layout(tmp_path):
    motion_path = tmp_path / "kfa_motion.npz"
    body_names = np.array(
        [
            "pelvis",
            "left_hip_pitch_link",
            "left_hip_roll_link",
            "left_hip_yaw_link",
            "left_knee_link",
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "left_foot_contact_point",
            "right_hip_pitch_link",
            "right_hip_roll_link",
            "right_hip_yaw_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
            "right_foot_contact_point",
            "waist_yaw_link",
            "waist_roll_link",
            "torso_link",
            "head_link",
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "left_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_roll_link",
            "right_wrist_pitch_link",
            "right_wrist_yaw_link",
        ],
        dtype="<U64",
    )
    joint_names = np.array(
        [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        dtype="<U64",
    )
    np.savez_compressed(
        motion_path,
        fps=np.array(30, dtype=np.int32),
        body_names=body_names,
        joint_names=joint_names,
        joint_pos=np.zeros((2, 36), dtype=np.float32),
        joint_vel=np.zeros((2, 35), dtype=np.float32),
        body_pos_w=np.zeros((2, len(body_names), 3), dtype=np.float32),
        body_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, len(body_names), 1)),
        body_lin_vel_w=np.zeros((2, len(body_names), 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, len(body_names), 3), dtype=np.float32),
    )

    loader = MotionLoader(
        str(motion_path),
        robot_body_names=body_names.tolist(),
        robot_joint_names=joint_names.tolist(),
        device="cpu",
    )

    assert loader.body_pos_w.shape == (2, len(body_names), 3)
    assert loader.joint_pos.shape == (2, len(joint_names))
    assert torch.equal(loader._body_indexes, torch.arange(len(body_names), dtype=torch.long))
