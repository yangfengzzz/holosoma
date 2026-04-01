# Paper To Code Spec

## Reset Sampling
- Reference timestep sampling is selected by `MotionConfig.sampling_strategy`.
- `adaptive` keeps the old failure-driven timestep sampler.
- `low_kinetic` extracts local minima of joint-velocity energy and samples anchors with failure-weight updates.

## Recovery Reset Source
- `MotionConfig.recovery_init_dataset` enables a second reset source.
- The reference timestep still comes from the motion clip.
- The robot state may instead come from a sampled recovery-init state with configurable probability.

## Recovery Signal
- Recovery mode is defined by shoulder-height gap:
  `reference_shoulder_height - robot_shoulder_height > shoulder_height_threshold`
- Reward gating and termination hysteresis both consume this shared signal from `MotionCommand`.

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
