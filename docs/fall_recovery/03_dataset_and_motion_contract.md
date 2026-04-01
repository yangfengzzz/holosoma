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
