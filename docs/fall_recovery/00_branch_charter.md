# Fall Recovery Branch Charter

## Goal
Implement the paper-style fall-recovery training path in Holosoma for `G1 + IsaacSim + FastSAC + WBT`, starting from Ground motions.

Implementation readiness is blocked on the Ground parity gate rather than assumed from code structure alone.

## V1 Scope
- Low-kinetic anchor sampling for reference reset timesteps
- Mixed reset sources: motion states or recovery-init dataset states
- Recovery-aware WBT rewards and termination
- Recovery dataset generator CLI
- Training and evaluation runbook

## Non-Goals
- Jump subset parity
- Real-robot deployment changes
- Head-link-specific ablations
- Re-implementing the paper's Section 3 dataset-cleaning pipeline inside Holosoma

## Success Criteria
- `exp:g1-29dof-wbt-recovery-fast-sac` is a valid preset
- Recovery dataset generation produces a reusable `.npz`
- Same-simulator evaluation runs end-to-end
- Unit tests cover sampler, recovery gating, hysteresis, and dataset loading
- One seed of the Ground parity gate passes train, same-sim eval, ONNX export, and MuJoCo launch-command generation

## Baseline Commands
```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-recovery-fast-sac logger:wandb

python src/holosoma/holosoma/eval_agent.py --checkpoint=<checkpoint>

python src/holosoma/holosoma/generate_recovery_dataset.py \
  --output-path ./artifacts/recovery_init/g1_ground_v1.npz
```
