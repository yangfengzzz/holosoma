# Paper To Code Spec

## Reset Sampling
- Reference timestep sampling is selected by `MotionConfig.sampling_strategy`.
- `adaptive` keeps the old failure-driven timestep sampler.
- `low_kinetic` extracts local minima of joint-velocity energy and samples anchors with failure-weight updates.
- Repo paper-parity note:
  the public paper does not expose the exact Eq. 17 weight-update rule, so Holosoma uses one explicit interpretation:
  each failure updates only the nearest preceding anchor and leaves all other anchors unchanged.

## Recovery Reset Source
- `MotionConfig.recovery_init_dataset` enables a second reset source.
- The reference timestep still comes from the motion clip.
- The robot state may instead come from a sampled recovery-init state with configurable probability.
- Raw `.raw.npz` files store gravity-settled states without augmentation.
- Processed `.npz` files store materialized augmented root states when the selected augmentation mode is not `none`.

## Recovery Signal
- Recovery mode is defined by shoulder-height gap:
  `reference_shoulder_height - robot_shoulder_height > shoulder_height_threshold`
- Reward gating and termination hysteresis both consume this shared signal from `MotionCommand`.
- The default threshold source is `MotionConfig.recovery_shoulder_height_threshold`; paper-facing Ground presets keep this at `1.0`.

## Hybrid Objective
- Reward phase selection is now explicit in `RewardManager`, not just documented by tags.
- Terms tagged `r_mtr` are active only while `recovery_active_mask == False`.
- Terms tagged `r_rc` are active only while `recovery_active_mask == True`.
- Untagged terms remain active in all phases.
- Repo paper-parity decision:
  the recovery preset follows a strict phase switch between motion-tracking (`r_mtr`) and recovery (`r_rc`) terms, matching the paper-facing hybrid-objective interpretation more closely than the previous additive implementation.

## Recovery Rewards
- Shoulder-height penalty while recovering
- XY root drift penalty while recovering
- Action-rate penalty while recovering

## Stability Terms
- Root/support alignment reward
- Feet-slip penalty
- Close-feet penalty
- Root orientation penalty
- Knee and ankle action-rate penalties

## Recovery Termination
- Outside recovery: bad tracking terminates immediately
- During recovery: bad tracking increments a counter
- Termination occurs only after `max_consecutive_bad_tracking_steps`
