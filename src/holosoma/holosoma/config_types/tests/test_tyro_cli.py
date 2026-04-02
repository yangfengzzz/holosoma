import tyro

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.kfa_parity import (
    KFA_OPTIONAL_REGRESSION_CLIP_IDS,
    KFA_PRIMARY_REGRESSION_CLIP_ID,
    KFA_TRAINING_EXAMPLE_CLIP_ID,
    get_kfa_released_motion_path,
)
from holosoma.config_values.experiment import AnnotatedExperimentConfig, DEFAULTS as EXPERIMENT_DEFAULTS
from holosoma.utils.tyro_utils import TYRO_CONIFG


def test_experiment_config():
    assert isinstance(tyro.cli(ExperimentConfig, args=(), config=TYRO_CONIFG), ExperimentConfig)


def test_kfa_stand_experiment_cli_surface():
    config = tyro.cli(AnnotatedExperimentConfig, args=("exp:g1-29dof-wbt-stand-fast-sac",), config=TYRO_CONIFG)
    motion_cfg = config.command.setup_terms["motion_command"].params["motion_config"]

    assert config.training.name == "g1_29dof_wbt_stand_fast_sac_manager"
    assert config.robot.asset.xml_file.endswith("g1/g1_29dof_kfa.xml")
    assert config.robot.asset.urdf_file.endswith("g1/g1_29dof_kfa.urdf")
    assert config.robot.asset.robot_type == "g1_29dof"
    assert "head_link" in config.robot.body_names
    assert motion_cfg.recovery_shoulder_height_threshold == 1.0
    assert motion_cfg.start_at_timestep_zero_prob == 0.0
    assert motion_cfg.freeze_at_timestep_zero_prob == 0.0
    assert motion_cfg.enable_default_pose_prepend is False
    assert motion_cfg.enable_default_pose_append is False
    assert config.reward.terms["motion_relative_body_position_error_exp"].weight == 4.0
    assert config.reward.terms["motion_relative_body_position_error_exp"].tags == ["r_mtr", "tracking"]
    assert config.reward.terms["feet_slip_penalty"].params["contact_force_threshold"] == 8.0
    assert config.termination.terms["bad_tracking"].func.endswith(":BadTracking")
    assert config.randomization.setup_terms["mass_randomizer"].params["enable_base_mass"] is True
    assert config.randomization.setup_terms["setup_action_delay_buffers"].params["enabled"] is True
    assert config.randomization.setup_terms["actuator_randomizer_state"].params["enable_pd_gain"] is True
    assert "undesired_contacts" in config.reward.terms
    assert "head_link" not in config.reward.terms["undesired_contacts"].params["undesired_contacts_body_names"]
    assert "recovery_action_rate_penalty" not in config.reward.terms


def test_existing_wbt_fast_sac_surface_is_unchanged():
    config = EXPERIMENT_DEFAULTS["g1_29dof_wbt_fast_sac"]

    assert config.robot.asset.xml_file.endswith("g1/g1_29dof.xml")
    assert "head_link" not in config.robot.body_names
    assert "action_rate_l2" in config.reward.terms


def test_recovery_preset_uses_explicit_grsi_augmentation_mode():
    config = tyro.cli(AnnotatedExperimentConfig, args=("exp:g1-29dof-wbt-recovery-fast-sac",), config=TYRO_CONIFG)
    recovery_cfg = config.command.setup_terms["motion_command"].params["motion_config"].recovery_init_dataset
    motion_cfg = config.command.setup_terms["motion_command"].params["motion_config"]
    assert recovery_cfg.augmentation_mode == "rotation_recombination"
    assert recovery_cfg.dataset_kind == "recovery_init"
    assert motion_cfg.start_at_timestep_zero_prob == 0.0
    assert motion_cfg.freeze_at_timestep_zero_prob == 0.0
    assert motion_cfg.enable_default_pose_prepend is False
    assert motion_cfg.enable_default_pose_append is False
    assert config.reward.terms["feet_slip_penalty"].params["contact_force_threshold"] == 8.0
    assert config.termination.terms["bad_tracking"].func.endswith(":RecoveryAwareBadTracking")
    assert config.randomization.setup_terms["mass_randomizer"].params["link_mass_range"] == [0.9, 1.2]
    assert config.randomization.setup_terms["setup_action_delay_buffers"].params["enabled"] is True
    assert "undesired_contacts" in config.reward.terms


def test_kfa_released_clip_helpers():
    assert get_kfa_released_motion_path(KFA_TRAINING_EXAMPLE_CLIP_ID).endswith("/1317.npz")
    assert get_kfa_released_motion_path(KFA_PRIMARY_REGRESSION_CLIP_ID).endswith("/1307.npz")
    assert KFA_OPTIONAL_REGRESSION_CLIP_IDS == ("969", "203")


def test_recovery_ablation_presets_resolve_with_expected_slip_weights():
    recovery_only = tyro.cli(
        AnnotatedExperimentConfig,
        args=("exp:g1-29dof-wbt-recovery-only-fast-sac",),
        config=TYRO_CONIFG,
    )
    slip3 = tyro.cli(
        AnnotatedExperimentConfig,
        args=("exp:g1-29dof-wbt-recovery-slip3-fast-sac",),
        config=TYRO_CONIFG,
    )
    slip5 = tyro.cli(
        AnnotatedExperimentConfig,
        args=("exp:g1-29dof-wbt-recovery-slip5-fast-sac",),
        config=TYRO_CONIFG,
    )

    assert recovery_only.training.name == "g1_29dof_wbt_recovery_only_fast_sac_manager"
    assert "feet_slip_penalty" not in recovery_only.reward.terms

    assert slip3.training.name == "g1_29dof_wbt_recovery_slip3_fast_sac_manager"
    assert slip3.reward.terms["feet_slip_penalty"].weight == -3.0

    assert slip5.training.name == "g1_29dof_wbt_recovery_slip5_fast_sac_manager"
    assert slip5.reward.terms["feet_slip_penalty"].weight == -5.0

    assert slip3.command.setup_terms["motion_command"].params["motion_config"].motion_file.endswith("/1317.npz")
    assert slip5.termination.terms["bad_tracking"].params["shoulder_height_threshold"] == 1.0
