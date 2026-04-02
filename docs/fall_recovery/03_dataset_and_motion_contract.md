# Dataset And Motion Contract

## Default Training Dataset
- Use `./KungFuAthleteBot/collection_g129dof/org_smoothed_mj` as the default dataset root for all paper-facing training and evaluation commands in this repo.
- This is the only dataset family that should be treated as the default WBT training input.
- The local clone in this workspace currently resolves these Ground clips directly:
  - `1317.npz`
  - `1307.npz`
  - `969.npz`
  - `203.npz`

## Clip Naming Rules
- Preferred local clip ids:
  - train smoke and baseline: `1317`
  - hard regression eval: `1307`
  - optional follow-up eval: `969`
  - optional follow-up eval in this workspace: `203`
- Supported training path form:
  - flat: `org_smoothed_mj/<clip>.npz`

## Dataset Families In The Clone
- `KungFuAthleteBot/collection_g129dof/org_smoothed_mj`
  - default training and eval input for Holosoma WBT
  - contains processed motion clips with `joint_pos`, `joint_vel`, body states, `joint_names`, and `body_names`
- `KungFuAthleteBot/collection_g129dof/org_smoothed`
  - source qpos-style retargeted motions
  - use this only for provenance, conversion, or debugging data preparation
- `KungFuAthleteBot/kungfu_gvhmr_video_demo/collection_gvhmr`
  - visualization and mocap-source data
  - do not use this tree as a Holosoma training input

## Motion File Expectations
- Motion files under `org_smoothed_mj` should provide:
  - `fps`
  - `joint_pos`
  - `joint_vel`
  - `body_pos_w`
  - `body_quat_w`
  - `body_lin_vel_w`
  - `body_ang_vel_w`
  - `joint_names`
  - `body_names`
- The local sampled clips in this workspace include `head_link` in `body_names`, which matches the paper-facing KFA robot preset.

## Recovery Dataset Contract
- Raw GRSI dataset:
  `.raw.npz` gravity-settled fallen states
- Training-ready recovery dataset:
  `.npz` file with:
  - `root_states`
  - `dof_pos`
  - `dof_vel`
  - `metadata_json`
- Default local output path:
  `./artifacts/recovery_init/g1_ground_v1.npz`

## Helper Behavior
- `src/holosoma/holosoma/utils/kfa_parity.py` owns paper-facing dataset discovery for parity runs.
- The helper should:
  - use `./KungFuAthleteBot/collection_g129dof/org_smoothed_mj` as the canonical dataset root
  - accept direct absolute and repo-relative motion-file overrides unchanged
  - treat `203` as the actual local optional follow-up clip id
  - record the clip id and resolved file path in parity manifests

## Ground Scope
- Ground-only is the default and documented path for fall-recovery work.
- Jump clips remain out of scope for the training roadmap until Ground tracking plus recovery is stable.
