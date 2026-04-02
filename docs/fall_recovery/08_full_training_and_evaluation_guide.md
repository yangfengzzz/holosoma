# Full Training And Evaluation Guide

This guide is the practical end-to-end workflow for running the released Ground stand and recovery experiments in this repo.

Use this document when you want to run real training and evaluation. For smoke checks, path validation, and readiness debugging, use `06_training_roadmap.md` instead.

This guide is intentionally Ground-only. Section 3 preprocessing remains external to Holosoma; this repo consumes the prepared `org_smoothed_mj` clips.

## 1. Prerequisites

Use these defaults throughout the workflow:
- dataset root: `./KungFuAthleteBot/collection_g129dof/org_smoothed_mj`
- training clip: `1317.npz`
- hard evaluation clip: `1307.npz`
- optional local follow-up clip in this workspace: `203.npz`
- recovery dataset path: `./artifacts/recovery_init/g1_ground_v1.npz`

Do not train directly from:
- `./KungFuAthleteBot/collection_g129dof/org_smoothed`
- `./KungFuAthleteBot/kungfu_gvhmr_video_demo/...`

IsaacSim shells in this guide assume:

```bash
source scripts/source_isaacsim_setup.sh
```

MuJoCo evaluation shells assume:

```bash
source scripts/source_mujoco_setup.sh
```

Inference shells assume:

```bash
source scripts/source_inference_setup.sh
```

If you have not validated the environment, clip paths, or simulator boot sequence yet, run the smoke workflow in `06_training_roadmap.md` before starting long training runs.

## 2. Recovery Dataset Preparation

Generate the recovery initialization dataset once before recovery training:

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

Validate that `MotionCommand` can consume the processed dataset for real recovery resets:

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/validate_recovery_dataset.py \
  --exp g1_29dof_wbt_recovery_fast_sac \
  --dataset-path ./artifacts/recovery_init/g1_ground_v1.npz \
  --batch-size 32
```

Expected result:
- the validator prints `validated_recovery_dataset`
- the reset path uses recovery states with finite root and DOF tensors

Do not start recovery training until this validation passes.

## 3. Stand Training

Train the paper-facing stand preset on the released Ground training clip:

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-stand-fast-sac \
  logger:wandb \
  --training.seed=1 \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1317.npz
```

Expected artifacts:
- a run directory under `logs/WholeBodyTracking/`
- training logs and saved configs
- one or more checkpoints such as `model_*.pt`

Use the resulting stand checkpoint for both same-sim evaluation and ONNX export. There is no need to retrain stand just to evaluate a different regression clip.

## 4. Stand Same-Sim Evaluation And ONNX Export

Evaluate the trained stand checkpoint in the same simulator configuration used for training:

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=<stand_checkpoint>.pt \
  --training.headless=True \
  --training.max-eval-steps=300 \
  --training.export_onnx=True \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1307.npz
```

Expected artifacts:
- an eval run directory under `logs/WholeBodyTracking/`
- ONNX export under `<checkpoint_dir>/exported/`

Use `1307.npz` as the default hard regression clip. For a local follow-up check, rerun the same command with only the clip override changed to `203.npz`.

## 5. Stand MuJoCo Sim-To-Sim Evaluation

Launch MuJoCo in one terminal:

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof-kfa --training.headless=True
```

Run the exported stand policy in a second terminal:

```bash
source scripts/source_inference_setup.sh
python src/holosoma_inference/holosoma_inference/run_policy.py \
  inference:g1-29dof-wbt \
  --task.model-path=<stand_exported_model>.onnx \
  --task.no-use-joystick \
  --task.use-sim-time \
  --task.rl-rate=50 \
  --task.interface=lo \
  --secondary none
```

Expected result:
- MuJoCo loads `robot:g1-29dof-kfa`
- the inference process accepts the exported ONNX without immediate mismatch errors

This step reuses the ONNX exported from the stand evaluation step. You do not need a separate export pipeline.

## 6. Recovery Training

Train the paper-facing recovery preset after the recovery dataset has been generated and validated:

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-recovery-fast-sac \
  logger:wandb \
  --training.seed=1 \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1317.npz \
  --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.enabled=True \
  --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.dataset_path=./artifacts/recovery_init/g1_ground_v1.npz
```

Expected artifacts:
- a run directory under `logs/WholeBodyTrackingRecovery/`
- training logs and saved configs
- one or more checkpoints such as `model_*.pt`

Recovery training is a separate run from stand training. Reuse the prepared recovery dataset, but do not reuse the stand checkpoint as a substitute for the recovery run.

## 7. Recovery Same-Sim Evaluation And ONNX Export

Evaluate the recovery checkpoint on the hard Ground regression clip:

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=<recovery_checkpoint>.pt \
  --training.headless=True \
  --training.max-eval-steps=300 \
  --training.export_onnx=True \
  --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1307.npz
```

Expected artifacts:
- an eval run directory for the recovery checkpoint
- ONNX export under the recovery checkpoint directory's `exported/` folder

As with stand evaluation, use `1307.npz` as the default hard clip and `203.npz` as the local optional follow-up clip.

## 8. Recovery MuJoCo Sim-To-Sim Evaluation

Launch MuJoCo:

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof-kfa --training.headless=True
```

Run the exported recovery policy:

```bash
source scripts/source_inference_setup.sh
python src/holosoma_inference/holosoma_inference/run_policy.py \
  inference:g1-29dof-wbt \
  --task.model-path=<recovery_exported_model>.onnx \
  --task.no-use-joystick \
  --task.use-sim-time \
  --task.rl-rate=50 \
  --task.interface=lo \
  --secondary none
```

Expected result:
- the recovery ONNX launches in MuJoCo without immediate packaging or robot-contract failures

## 9. Recommended Run Order

Run the main workflow in this order:
1. generate the recovery dataset
2. validate the recovery dataset
3. train stand
4. evaluate stand in the same simulator and export ONNX
5. launch stand in MuJoCo
6. train recovery
7. evaluate recovery in the same simulator and export ONNX
8. launch recovery in MuJoCo

This keeps the recovery workflow grounded on a validated reset source and ensures both paper-facing presets pass the same-sim and sim-to-sim path.

## 10. Next Experiments

After the main stand and recovery runs are stable, the next paper-facing ablations are:
- `exp:g1-29dof-wbt-recovery-only-fast-sac`
- `exp:g1-29dof-wbt-recovery-slip3-fast-sac`
- `exp:g1-29dof-wbt-recovery-slip5-fast-sac`

Treat those as follow-up experiments rather than part of the primary training roadmap in this document.
