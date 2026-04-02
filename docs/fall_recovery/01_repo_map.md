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
  Generates raw gravity-settled recovery initialization datasets and materialized processed variants for Ground training.
- `holosoma/utils/recovery_init_dataset.py`
  Loads, samples, and applies the documented recovery-state augmentation policy during training.

## Scope Notes
- Ground-only is the current paper-facing implementation target in this repo.
- Section 3 preprocessing remains externalized to the local `KungFuAthleteBot` checkout; Holosoma consumes the resulting `org_smoothed_mj` clips rather than reproducing that pipeline.
