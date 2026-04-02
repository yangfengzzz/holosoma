# Ground Fall-Recovery Training Roadmap

This roadmap is the recommended step-by-step path for reproducing the paper-style Ground workflow in this repo.

Use only:
- dataset root: `./KungFuAthleteBot/collection_g129dof/org_smoothed_mj`
- training clip: `1317.npz`
- hard evaluation clip: `1307.npz`
- optional local follow-up clips: `969.npz`, `203.npz`

Do not train directly from:
- `./KungFuAthleteBot/collection_g129dof/org_smoothed`
- `./KungFuAthleteBot/kungfu_gvhmr_video_demo/...`

## Stage 0: Environment And Dataset Sanity Check

Command:
```bash
conda run -n hssim python - <<'PY'
from pathlib import Path
root = Path("./KungFuAthleteBot/collection_g129dof/org_smoothed_mj").resolve()
print("dataset_root", root)
print("exists", root.exists())
for clip in ("1317", "1307", "969", "203"):
    print(clip, (root / f"{clip}.npz").exists())
PY
```

Expected artifacts:
- none

Pass criterion:
- dataset root exists
- `1317.npz` and `1307.npz` exist

If it fails:
- confirm the clone is at `./KungFuAthleteBot`
- do not continue until the `org_smoothed_mj` root is present

## Stage 1: Single-Clip Path Validation

Command:
```bash
conda run -n hssim python - <<'PY'
from holosoma.utils.kfa_parity import resolve_kfa_motion_path
for clip in ("1317", "1307", "969", "203"):
    print(clip, "->", resolve_kfa_motion_path(clip, "./KungFuAthleteBot/collection_g129dof/org_smoothed_mj"))
PY
```

Expected artifacts:
- none

Pass criterion:
- `1317`, `1307`, and `969` resolve directly
- `203` resolves directly

If it fails:
- fix the dataset root or path helper first
- do not start training with manual path guessing

## Stage 2: Motion Replay Smoke

Command:
```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
  exp:g1-29dof-wbt-stand-fast-sac \
  --training.headless=False \
  --training.num_envs=1 \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1317.npz
```

Expected artifacts:
- none required

Pass criterion:
- IsaacSim boots
- the motion clip loads without body-name or file-path errors

If it fails:
- confirm the clip path is from `org_smoothed_mj`
- if body-name mismatch appears, stop and fix the robot or motion contract before training

## Stage 3: Stand Training Smoke

Command:
```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-stand-fast-sac \
  logger:disabled \
  --training.num-envs=2 \
  --algo.config.num-learning-iterations=1 \
  --training.seed=1 \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1317.npz
```

Expected artifacts:
- `logs/WholeBodyTracking/<run>/model_0000001.pt`

Pass criterion:
- training completes one iteration
- a checkpoint is written

If it fails:
- first remove path/config issues
- only then investigate simulator or reward issues

## Stage 4: Stand Same-Sim Eval On Hard Clip

Command:
```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=<stand_checkpoint>.pt \
  --training.headless=True \
  --training.max-eval-steps=300 \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1307.npz
```

Expected artifacts:
- eval log directory beside the checkpoint
- exported ONNX under `exported/`

Pass criterion:
- eval starts and finishes without config/runtime errors
- ONNX file is exported

If it fails:
- verify the checkpoint came from the stand preset
- verify the eval clip path is `1307.npz`

## Stage 5: MuJoCo Launch Smoke

Terminal 1:
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof-kfa --training.headless=True
```

Terminal 2:
```bash
conda run -n hsinference python src/holosoma_inference/holosoma_inference/run_policy.py \
  inference:g1-29dof-wbt \
  --task.model-path=<stand_exported_model>.onnx \
  --task.no-use-joystick \
  --task.use-sim-time \
  --task.rl-rate=50 \
  --task.interface=lo \
  --secondary none
```

Expected artifacts:
- none required

Pass criterion:
- MuJoCo loads `robot:g1-29dof-kfa`
- inference process starts without immediate ONNX or robot mismatch errors

If it fails:
- treat it as implementation debt, not tuning debt
- fix sim-to-sim packaging before broader training

## Stage 6: Recovery Dataset Generation

Command:
```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/generate_recovery_dataset.py \
  --exp g1_29dof_wbt_recovery_fast_sac \
  --output-path ./artifacts/recovery_init/g1_ground_v1.npz \
  --num-samples 1024 \
  --batch-size 128 \
  --settle-steps 300 \
  --friction-range 0.3 1.2 \
  --processed-augmentation-mode rotation_recombination \
  --seed 1
```

Expected artifacts:
- `./artifacts/recovery_init/g1_ground_v1.npz`
- `./artifacts/recovery_init/g1_ground_v1.raw.npz`

Pass criterion:
- both files are written
- generator exits without IsaacSim CLI parsing errors

If it fails:
- fix generator/runtime issues before recovery training

## Stage 7: Recovery Training Smoke

Command:
```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-recovery-fast-sac \
  logger:disabled \
  --training.num-envs=2 \
  --algo.config.num-learning-iterations=1 \
  --training.seed=1 \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1317.npz \
  --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.enabled=True \
  --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.dataset_path=./artifacts/recovery_init/g1_ground_v1.npz
```

Expected artifacts:
- `logs/WholeBodyTrackingRecovery/<run>/model_0000001.pt`

Pass criterion:
- recovery training completes one iteration
- checkpoint is written

If it fails:
- verify the recovery dataset path first
- then inspect reset mixing and recovery-init loading

## Stage 8: Recovery Eval On Hard Clip

Command:
```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=<recovery_checkpoint>.pt \
  --training.headless=True \
  --training.max-eval-steps=300 \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1307.npz
```

Expected artifacts:
- eval logs
- exported ONNX

Pass criterion:
- recovery checkpoint evaluates cleanly on `1307`

If it fails:
- compare against the stand checkpoint on the same clip before changing rewards

## Stage 9: Recovery Ablations

Commands:
```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-recovery-only-fast-sac logger:wandb --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1317.npz --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.enabled=True --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.dataset_path=./artifacts/recovery_init/g1_ground_v1.npz

python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-recovery-slip3-fast-sac logger:wandb --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1317.npz --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.enabled=True --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.dataset_path=./artifacts/recovery_init/g1_ground_v1.npz

python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-recovery-slip5-fast-sac logger:wandb --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1317.npz --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.enabled=True --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.dataset_path=./artifacts/recovery_init/g1_ground_v1.npz
```

Expected artifacts:
- one Wandb or local log directory per ablation

Pass criterion:
- all three presets launch with identical dataset paths and only the intended reward difference

If it fails:
- fix preset wiring before comparing results

## Stage 10: Multi-Seed Parity Harness

Dry-run first:
```bash
python src/holosoma/holosoma/run_kfa_ground_parity.py \
  --dataset-root=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj \
  --output-dir=./artifacts/parity_ground
```

Execute one-seed readiness gate:
```bash
python src/holosoma/holosoma/run_kfa_ground_parity.py \
  --dataset-root=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj \
  --output-dir=./artifacts/parity_ground_execute \
  --preset-keys stand recovery \
  --seeds 1 \
  --execute=True
```

Expand to full parity sweep:
```bash
python src/holosoma/holosoma/run_kfa_ground_parity.py \
  --dataset-root=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj \
  --output-dir=./artifacts/parity_ground_full \
  --preset-keys stand recovery recovery_only slip3 slip5 \
  --seeds 1 2 3 \
  --execute=True
```

Expected artifacts:
- `suite_manifest.yaml`
- one `run_manifest.yaml` per preset and seed
- train/eval logs in each artifact directory

Pass criterion:
- one-seed `stand` and `recovery` pass end-to-end before broader training
- manifests record resolved local clip paths correctly

If it fails:
- fix the failing stage before scaling to more seeds

## Evaluation Progression
- Start with `1307`
- Then add `969`
- Then add local `203`
- Do not start Jump experiments until Ground tracking and recovery are stable across the Ground suite
