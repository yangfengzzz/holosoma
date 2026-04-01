# AGENTS Guide

## Purpose
This repository is a humanoid robotics mono-repo with three main packages:
- `src/holosoma`: training and simulator-side RL code
- `src/holosoma_inference`: policy inference and deployment
- `src/holosoma_retargeting`: motion retargeting and dataset conversion

When changing behavior, prefer extending the existing manager/config system instead of adding parallel one-off code paths.

## Preferred Environments
- Use `hssim` for simulator-side Python checks and unit tests in `src/holosoma`.
- Use `hsinference` for inference-only work when a lighter environment is enough.
- Use `hsretargeting` for retargeting/data pipeline work.

For this repo, the default test command should usually be run as:

```bash
conda run -n hssim pytest -q <target>
```

## Key Workflows

### Training
```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py exp:<preset> logger:wandb
```

### Evaluation
```bash
python src/holosoma/holosoma/eval_agent.py --checkpoint=<checkpoint>
```

### Recovery Dataset Generation
```bash
python src/holosoma/holosoma/generate_recovery_dataset.py \
  --exp g1_29dof_wbt_recovery_fast_sac \
  --output-path ./artifacts/recovery_init/g1_ground_v1.npz
```

## Repo Conventions
- Use `rg` and `rg --files` for search.
- Keep new behavior configurable through `config_types` and `config_values`.
- For WBT work, the main extension points are:
  - `holosoma.managers.command.terms.wbt`
  - `holosoma.managers.reward.terms.wbt`
  - `holosoma.managers.termination.terms.wbt`
  - `holosoma.envs.wbt.wbt_manager`
- Keep simulator-specific assumptions isolated. If a feature is IsaacSim-only, say so in code comments or docs.
- Add small targeted tests near the subsystem you changed before broad E2E coverage.

## Editing Guardrails
- Do not revert unrelated user changes in a dirty worktree.
- Prefer `apply_patch` for manual edits.
- Avoid destructive git commands unless explicitly requested.
- If a test requires a specific conda env, run it in that env rather than weakening the test.

## Fall-Recovery Notes
- The recovery training preset is `exp:g1-29dof-wbt-recovery-fast-sac`.
- The paper-facing stand preset is `exp:g1-29dof-wbt-stand-fast-sac`.
- The KungFuAthlete-specific robot key is `robot:g1-29dof-kfa`, backed by `g1_29dof_kfa.{xml,urdf}` while keeping `asset.robot_type="g1_29dof"` for bridge compatibility.
- The recovery docs live under `docs/fall_recovery/`.
- Mixed reset-source behavior is owned by `MotionCommand`, not by the environment class directly.
