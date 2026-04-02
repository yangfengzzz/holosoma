# Experiment Runbook

## Environment
```bash
source scripts/source_isaacsim_setup.sh
```

## Released Clip Defaults
- Training example clip: `1317`
- Primary hard regression clip: `1307`
- Optional parity follow-ups: `969`, `0203`
- Expected dataset root: `./datasets/KungFuAthleteBot/org_smoothed_mj`

## 1. Generate Recovery States
```bash
python src/holosoma/holosoma/generate_recovery_dataset.py \
  --exp g1_29dof_wbt_recovery_fast_sac \
  --output-path ./artifacts/recovery_init/g1_ground_v1.npz \
  --num-samples 1024 \
  --friction-range 0.3 1.2 \
  --processed-augmentation-mode rotation_recombination
```

## 2. Train Paper-Facing Presets
```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-stand-fast-sac \
  logger:wandb \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./datasets/KungFuAthleteBot/org_smoothed_mj/1317_mj.npz
```

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-recovery-fast-sac \
  logger:wandb \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./datasets/KungFuAthleteBot/org_smoothed_mj/1317_mj.npz \
  --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.enabled=True \
  --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.dataset_path=./artifacts/recovery_init/g1_ground_v1.npz
```

## 3. Run Ablations
```bash
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-recovery-only-fast-sac logger:wandb
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-recovery-slip3-fast-sac logger:wandb
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-recovery-slip5-fast-sac logger:wandb
```

## 4. Same-Sim Evaluation and ONNX Export
```bash
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=<checkpoint> \
  --training.headless=True \
  --training.max-eval-steps=300 \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./datasets/KungFuAthleteBot/org_smoothed_mj/1307_mj.npz
```

- `eval_agent.py` writes same-sim artifacts under the eval log directory and exports ONNX beside the checkpoint when `training.export_onnx=True`.
- Re-run the same command with clip `969` or `0203` by changing only the `motion_file` override.

## 5. MuJoCo Sim-to-Sim Evaluation
Terminal 1:
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof
```

Terminal 2:
```bash
source scripts/source_inference_setup.sh
python src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
  --task.model-path <checkpoint_dir>/exported/<model>.onnx \
  --task.no-use-joystick \
  --task.use-sim-time \
  --task.rl-rate 50 \
  --task.interface lo
```

- Use `lo` for same-machine sim-to-sim.
- The MuJoCo WBT path replays the exported policy directly; there is no separate clip-file override in `run_policy.py`.

## Useful Overrides
- Force pure anchor resets:
  `--command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.sample_probability=0.0`
- Force pure recovery-state resets:
  `--command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.sample_probability=1.0`
- Swap same-sim regression clip:
  `--command.setup_terms.motion_command.params.motion_config.motion_file=./datasets/KungFuAthleteBot/org_smoothed_mj/<clip>_mj.npz`

## Failure Triage
- Immediate collapse:
  check recovery dataset quality and shoulder-height threshold
- High reset churn during recovery:
  increase `max_consecutive_bad_tracking_steps`
- Same-sim good but MuJoCo unstable:
  verify the exported ONNX came from the exact checkpoint under test and that inference uses `inference:g1-29dof-wbt`
- Stable standing but poor imitation:
  compare `1307` against `969` and `0203` before changing reward terms
