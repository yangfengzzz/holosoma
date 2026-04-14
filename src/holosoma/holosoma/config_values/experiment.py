import tyro
from typing_extensions import Annotated

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof, t1_29dof_fast_sac
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_kfa_1307_stage1_fast_sac,
    g1_29dof_kfa_1307_stage2_fast_sac,
    g1_29dof_kfa_1307_stage3_fast_sac,
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_recovery_debug_base_fast_sac,
    g1_29dof_wbt_recovery_fast_sac,
    g1_29dof_wbt_recovery_low_kinetic_fast_sac,
    g1_29dof_wbt_recovery_only_fast_sac,
    g1_29dof_wbt_recovery_slip3_fast_sac,
    g1_29dof_wbt_recovery_slip5_fast_sac,
    g1_29dof_wbt_stand_fast_sac,
    g1_29dof_wbt_w_object,
)

DEFAULTS = {
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "g1_29dof_kfa_1307_stage1_fast_sac": g1_29dof_kfa_1307_stage1_fast_sac,
    "g1_29dof_kfa_1307_stage2_fast_sac": g1_29dof_kfa_1307_stage2_fast_sac,
    "g1_29dof_kfa_1307_stage3_fast_sac": g1_29dof_kfa_1307_stage3_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,
    "g1_29dof_wbt_recovery_debug_base_fast_sac": g1_29dof_wbt_recovery_debug_base_fast_sac,
    "g1_29dof_wbt_recovery_fast_sac": g1_29dof_wbt_recovery_fast_sac,
    "g1_29dof_wbt_recovery_low_kinetic_fast_sac": g1_29dof_wbt_recovery_low_kinetic_fast_sac,
    "g1_29dof_wbt_recovery_only_fast_sac": g1_29dof_wbt_recovery_only_fast_sac,
    "g1_29dof_wbt_recovery_slip3_fast_sac": g1_29dof_wbt_recovery_slip3_fast_sac,
    "g1_29dof_wbt_recovery_slip5_fast_sac": g1_29dof_wbt_recovery_slip5_fast_sac,
    "g1_29dof_wbt_stand_fast_sac": g1_29dof_wbt_stand_fast_sac,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]
