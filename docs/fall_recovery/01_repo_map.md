# Repo Map

## Core Runtime
- `holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager`
  Owns WBT env buffers and body/joint index caches for recovery rewards.
- `holosoma.managers.command.terms.wbt.MotionCommand`
  Owns motion loading, timestep sampling, mixed reset-source logic, and recovery-state signals.
- `holosoma.managers.reward.terms.wbt`
  Contains imitation rewards plus recovery and balance terms.
- `holosoma.managers.termination.terms.wbt`
  Contains bad-tracking logic and the recovery-aware hysteresis term.

## Config Composition
- `config_values/wbt/g1/command.py`
  Recovery sampling strategy and recovery dataset fields.
- `config_values/wbt/g1/reward.py`
  Recovery and stability reward preset.
- `config_values/wbt/g1/termination.py`
  Recovery-aware termination preset.
- `config_values/wbt/g1/experiment.py`
  Public entrypoint: `exp:g1-29dof-wbt-recovery-fast-sac`.

## Tooling
- `holosoma/generate_recovery_dataset.py`
  Generates gravity-settled recovery initialization datasets.
- `holosoma/utils/recovery_init_dataset.py`
  Loads and samples those datasets during training.
