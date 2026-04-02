# Dataset And Motion Contract

## Motion Inputs
- Existing WBT motion files remain `.npz`
- Required arrays:
  - `joint_pos`, `joint_vel`
  - `body_pos_w`, `body_quat_w`
  - `body_lin_vel_w`, `body_ang_vel_w`
- Optional object arrays are unchanged

## Recovery Dataset Format
- Generator output is a compressed `.npz`
- Required arrays:
  - `root_states`: `[N, 13]`
  - `dof_pos`: `[N, num_dofs]`
  - `dof_vel`: `[N, num_dofs]`

## Default Location
- Recommended path: `./artifacts/recovery_init/g1_ground_v1.npz`

## Ground Subset Assumption
- V1 should start from a single hard Ground clip or a curated Ground-only subset.
- Jump motions stay out of scope until recovery is stable on Ground.

## Reset Semantics
- Motion reference and reset source are decoupled.
- Motion anchors define what the policy should recover back to.
- Recovery dataset states define where some episodes begin.
# Dataset And Motion Contract

## Phase A Surface
- The paper-facing stand preset is `exp:g1-29dof-wbt-stand-fast-sac`.
- Its default motion path is auto-resolved from the first existing dataset root:
  `./KungFuAthleteBot/collection_g129dof/org_smoothed_mj`
  `./datasets/KungFuAthleteBot/org_smoothed_mj`
- File naming accepts both:
  `<clip>.npz`
  `<clip>_mj.npz`
- The local repo checkout is preferred when it exists because it matches the data layout currently used in this workspace.

## Supported Motion Path Forms
- Package data path:
  `holosoma/data/motions/g1_29dof/whole_body_tracking/<clip>.npz`
- Absolute path:
  `/data/kungfu/org_smoothed_mj/1317.npz`
- Repo-relative or cwd-relative path:
  `./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1317.npz`

Holosoma resolves these forms through `resolve_data_file_path()`, and the Ground parity helper further normalizes the two common local dataset layouts plus both clip filename conventions.

## Recovery Dataset Contract
- Raw GRSI dataset:
  direct gravity-settled fallen states saved as `.raw.npz`
- Training-ready recovery dataset:
  NPZ with the same `root_states`, `dof_pos`, and `dof_vel` payload plus `metadata_json`
- Required metadata captures:
  dataset kind, default augmentation mode, preset, robot type, friction range, settle steps, seed, batch size, and sample count
- Supported augmentation modes:
  `none`, `yaw`, `rotation_recombination`

## KFA Robot Surface
- The KungFuAthlete-specific robot preset is `robot:g1-29dof-kfa`.
- It uses dedicated asset files:
  `src/holosoma/holosoma/data/robots/g1/g1_29dof_kfa.xml`
  `src/holosoma/holosoma/data/robots/g1/g1_29dof_kfa.urdf`
- The KFA body layout adds `head_link` immediately after `torso_link` to match the dedicated asset indexing.

## Motion File Expectations
- Motion `.npz` files still use the repo’s WBT contract:
  - `fps`
  - `body_names`
  - `joint_names`
  - `joint_pos`
  - `joint_vel`
  - `body_pos_w`
  - `body_quat_w`
  - `body_lin_vel_w`
  - `body_ang_vel_w`
- KungFuAthlete Ground clips are expected to expose the same G1 body and joint names used by the KFA preset, including `head_link` when body-level alignment needs it.
