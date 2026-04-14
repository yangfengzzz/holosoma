"""Default reward manager configurations."""

from holosoma.config_values.loco.g1.reward import g1_29dof_loco, g1_29dof_loco_fast_sac
from holosoma.config_values.loco.t1.reward import t1_29dof_loco, t1_29dof_loco_fast_sac
from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_kfa_1307_stage1_fast_sac_reward,
    g1_29dof_kfa_1307_stage2_fast_sac_reward,
    g1_29dof_kfa_1307_stage3_fast_sac_reward,
    g1_29dof_wbt_fast_sac_reward,
    g1_29dof_wbt_recovery_debug_fast_sac_reward,
    g1_29dof_wbt_recovery_fast_sac_reward,
    g1_29dof_wbt_recovery_only_fast_sac_reward,
    g1_29dof_wbt_recovery_slip3_fast_sac_reward,
    g1_29dof_wbt_recovery_slip5_fast_sac_reward,
    g1_29dof_wbt_reward,
    g1_29dof_wbt_stand_fast_sac_reward,
    g1_29dof_wbt_reward_w_object,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_loco": t1_29dof_loco,
    "t1_29dof_loco_fast_sac": t1_29dof_loco_fast_sac,
    "g1_29dof_loco": g1_29dof_loco,
    "g1_29dof_loco_fast_sac": g1_29dof_loco_fast_sac,
    "g1_29dof_kfa_1307_stage1_fast_sac": g1_29dof_kfa_1307_stage1_fast_sac_reward,
    "g1_29dof_kfa_1307_stage2_fast_sac": g1_29dof_kfa_1307_stage2_fast_sac_reward,
    "g1_29dof_kfa_1307_stage3_fast_sac": g1_29dof_kfa_1307_stage3_fast_sac_reward,
    "g1_29dof_wbt": g1_29dof_wbt_reward,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_reward_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac_reward,
    "g1_29dof_wbt_recovery_debug_fast_sac": g1_29dof_wbt_recovery_debug_fast_sac_reward,
    "g1_29dof_wbt_recovery_fast_sac": g1_29dof_wbt_recovery_fast_sac_reward,
    "g1_29dof_wbt_recovery_only_fast_sac": g1_29dof_wbt_recovery_only_fast_sac_reward,
    "g1_29dof_wbt_recovery_slip3_fast_sac": g1_29dof_wbt_recovery_slip3_fast_sac_reward,
    "g1_29dof_wbt_recovery_slip5_fast_sac": g1_29dof_wbt_recovery_slip5_fast_sac_reward,
    "g1_29dof_wbt_stand_fast_sac": g1_29dof_wbt_stand_fast_sac_reward,
}
