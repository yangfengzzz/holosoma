import tyro

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig, DEFAULTS as EXPERIMENT_DEFAULTS
from holosoma.utils.tyro_utils import TYRO_CONIFG


def test_experiment_config():
    assert isinstance(tyro.cli(ExperimentConfig, args=(), config=TYRO_CONIFG), ExperimentConfig)


def test_kfa_stand_experiment_cli_surface():
    config = tyro.cli(AnnotatedExperimentConfig, args=("exp:g1-29dof-wbt-stand-fast-sac",), config=TYRO_CONIFG)

    assert config.training.name == "g1_29dof_wbt_stand_fast_sac_manager"
    assert config.robot.asset.xml_file.endswith("g1/g1_29dof_kfa.xml")
    assert config.robot.asset.urdf_file.endswith("g1/g1_29dof_kfa.urdf")
    assert config.robot.asset.robot_type == "g1_29dof"
    assert "head_link" in config.robot.body_names


def test_existing_wbt_fast_sac_surface_is_unchanged():
    config = EXPERIMENT_DEFAULTS["g1_29dof_wbt_fast_sac"]

    assert config.robot.asset.xml_file.endswith("g1/g1_29dof.xml")
    assert "head_link" not in config.robot.body_names
