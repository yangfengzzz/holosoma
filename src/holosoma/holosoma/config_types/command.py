"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""

    func: str
    """Import path for the command hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across command hooks."""

    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)


########################################################################################################################
# Motion command configuration
########################################################################################################################
@dataclass(frozen=True)
class NoiseToInitialPoseConfig:
    """Initial pose of the robot and object to those in the motion file."""

    overall_noise_scale: float = 0.0
    """Overall noise scale for the initial pose."""

    dof_pos: float = 0.0
    """Noise scale for the initial dof position."""

    root_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root position x, y, z."""

    root_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root rotation roll, pitch, yaw."""

    root_lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root linear velocity vx, vy, vz."""

    root_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root angular velocity wx, wy, wz."""

    object_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for object position x, y, z."""


@dataclass(frozen=True)
class MotionConfig:
    """Motion related configuration for Whole Body Tracking.

    NOTE:
    - Motion file is assumed to be in the format of:
      - joint_pos: (T, J)
      - joint_vel: (T, J)

      - body_pos_w: (T, B, 3)
      - body_quat_w: (T, B, 4) # wxyz -> xyzw
      - body_lin_vel_w: (T, B, 3)
      - body_ang_vel_w: (T, B, 3)

      If object is present in the motion file, it is assumed to be in the format of:
      - object_pos_w: (T, 3)
      - object_quat_w: (T, 4)
      - object_lin_vel_w: (T, 3)
      - object_ang_vel_w: (T, 3)

      If the motion clip assumes a terrain, the terrain has to be specified in holosoma/config/terrain/terrain_wbt.yaml
    """

    motion_file: str
    """Motion file (.npz) that contains motion_clips to track. """

    body_name_ref: list[str]
    """Body name of the reference frame (in general, torso_link). """
    body_names_to_track: list[str]
    """Key body names to track, used for reward/termination computation."""

    # motion sampling related
    class MotionSamplingStrategy(str, Enum):
        START = "start"
        UNIFORM = "uniform"
        ADAPTIVE = "adaptive"
        LKE = "lke"
        LOW_KINETIC = "low_kinetic"

    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    sampling_strategy: MotionSamplingStrategy = MotionSamplingStrategy.UNIFORM
    """How reference timesteps are sampled during reset.

    ``use_adaptive_timesteps_sampler`` is preserved for backward compatibility and
    maps to ``adaptive`` when this field remains ``uniform``.
    """

    start_at_timestep_zero_prob: float = 0.2
    """Probability of starting at timestep zero."""

    freeze_at_timestep_zero_prob: float = 0.95
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    enable_default_pose_prepend: bool = True
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True."""

    enable_default_pose_append: bool = True
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)

    @dataclass(frozen=True)
    class LowKineticSamplingConfig:
        anchor_window_size: int = 15
        """Half-window used to detect local minima of joint kinetic proxy."""

        min_anchor_spacing: int = 10
        """Minimum spacing between selected anchor timesteps."""

        ema_alpha: float = 0.05
        """EMA coefficient for updating anchor failure weights."""

        uniform_ratio: float = 0.1
        """Uniform exploration mass mixed into anchor sampling probabilities."""

        failure_weight: float = 1.0
        """Multiplier applied to new failure counts before EMA update."""

    low_kinetic_sampling: LowKineticSamplingConfig = field(default_factory=LowKineticSamplingConfig)
    """Configuration for low-kinetic-energy anchor sampling."""

    reset_mode_weights: tuple[float, float] = (1.0, 0.0)
    """Relative sampling weights for (tracking_like, standing_like) reset branches."""

    standing_like_reset_enabled: bool = False
    """Whether a release-parity standing-like reset branch should be sampled on reset."""

    standing_like_reset_joint_noise_scale: float = 0.0
    """Extra scale multiplier applied to joint reset noise for standing-like resets."""

    standing_like_reset_root_noise_scale: float = 0.0
    """Extra scale multiplier applied to root reset noise for standing-like resets."""

    @dataclass(frozen=True)
    class RecoveryInitDatasetConfig:
        class AugmentationMode(str, Enum):
            NONE = "none"
            YAW = "yaw"
            ROTATION_RECOMBINATION = "rotation_recombination"

        class DatasetKind(str, Enum):
            RAW_GRSI = "raw_grsi"
            RECOVERY_INIT = "recovery_init"

        enabled: bool = False
        """Whether recovery-state resets are enabled."""

        dataset_path: str = ""
        """Path to an `.npz` file produced by the recovery dataset generator."""

        sample_probability: float = 0.5
        """Probability of using a sampled recovery state instead of motion-state reset."""

        augmentation_mode: AugmentationMode = AugmentationMode.ROTATION_RECOMBINATION
        """How sampled recovery states should be augmented before reset."""

        dataset_kind: DatasetKind = DatasetKind.RECOVERY_INIT
        """Expected kind of recovery dataset consumed at training time."""

    recovery_init_dataset: RecoveryInitDatasetConfig = field(default_factory=RecoveryInitDatasetConfig)
    """Optional dataset of gravity-settled recovery initial states."""

    recovery_shoulder_height_threshold: float = 1.0
    """Shared shoulder-height threshold used to detect recovery mode."""
