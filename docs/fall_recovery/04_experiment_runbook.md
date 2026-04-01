# Experiment Runbook

## Environment
```bash
source scripts/source_isaacsim_setup.sh
```

## 1. Generate Recovery States
```bash
python src/holosoma/holosoma/generate_recovery_dataset.py \
  --exp g1_29dof_wbt_recovery_fast_sac \
  --output-path ./artifacts/recovery_init/g1_ground_v1.npz
```

## 2. Train
```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-recovery-fast-sac \
  logger:wandb \
  --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.enabled=True \
  --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.dataset_path=./artifacts/recovery_init/g1_ground_v1.npz
```

## 3. Evaluate
```bash
python src/holosoma/holosoma/eval_agent.py --checkpoint=<checkpoint>
```

## Useful Overrides
- Force pure anchor resets:
  `--command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.sample_probability=0.0`
- Force pure recovery-state resets:
  `--command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.sample_probability=1.0`
- Swap motion clip:
  `--command.setup_terms.motion_command.params.motion_config.motion_file=<path>`

## Failure Triage
- Immediate collapse:
  check recovery dataset quality and shoulder-height threshold
- High reset churn during recovery:
  increase `max_consecutive_bad_tracking_steps`
- Stable standing but poor imitation:
  reduce recovery penalties and inspect anchor distribution
