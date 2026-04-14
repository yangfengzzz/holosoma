# Research Process Reconstruction: From Baseline Tracking to Released 1307 Stages

This document reconstructs the likely research path behind the released KungFuAthlete Ground standing tasks by comparing three evidence sources:

- the public paper: [A Kung Fu Athlete Bot That Can Do It All Day](https://arxiv.org/html/2602.13656v1)
- the released MJLab code under `KungFuAthleteBot/unitree_rl_mjlab`
- the Holosoma implementation history in this repo

The goal is not to invent a private timeline for the author. The goal is to explain, with references, how the final released `1307` Stage I/II/III tasks can be understood as an evolution from a baseline whole-body tracking task toward a more operational, curriculum-driven standing and robustness recipe.

## Reading Standard

This document uses three evidence labels:

- `direct_evidence`: stated directly in the paper or visible directly in released code
- `strong_inference`: not explicitly stated, but strongly suggested by code structure and task design
- `open_inference`: plausible interpretation, but not provable from public artifacts alone

When a claim is not directly supported, it is marked as an inference.

## Source Map

Primary sources used in this reconstruction:

- Paper method and experiments:
  - [Section 4: End-to-End Motion Tracking with Autonomous Fall Recovery](https://arxiv.org/html/2602.13656v1#S4)
  - [Section 4.2: Low Kinetic Energy Sampling](https://arxiv.org/html/2602.13656v1#S4.SS2)
  - [Section 4.3: Reward Design](https://arxiv.org/html/2602.13656v1#S4.SS3)
  - [Section 4.4: Termination Conditions](https://arxiv.org/html/2602.13656v1#S4.SS4)
  - [Section 4.5: GRSI](https://arxiv.org/html/2602.13656v1#S4.SS5)
  - [Section 5.2: Motion Tracking Performance / Ablation](https://arxiv.org/html/2602.13656v1#S5.SS2)

- Released MJLab code:
  - [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)
  - [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)
  - [rewards.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/rewards.py>)
  - [env_cfgs.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/env_cfgs.py>)
  - [__init__.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/__init__.py>)
  - [rl_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/rl_cfg.py>)

- Holosoma paper-facing implementation:
  - [02_paper_to_code_spec.md](./02_paper_to_code_spec.md)
  - [07_paper_alignment_evidence.md](./07_paper_alignment_evidence.md)
  - [src/holosoma/holosoma/config_values/wbt/g1/experiment.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/experiment.py>)
  - [src/holosoma/holosoma/config_values/wbt/g1/command.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/command.py>)
  - [src/holosoma/holosoma/config_values/wbt/g1/reward.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/reward.py>)
  - [src/holosoma/holosoma/config_values/wbt/g1/termination.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/termination.py>)
  - [src/holosoma/holosoma/config_values/wbt/g1/observation.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/observation.py>)

## Short Conclusion

The most defensible reconstruction is:

1. start from a baseline whole-body tracking stack
2. identify that highly dynamic Ground clips fail under standard tracking
3. introduce low-kinetic reset sampling and stability-biased rewards
4. introduce recovery-state initialization and hybrid recovery logic
5. operationalize the hardest Ground case into a dedicated standing curriculum with Stage I, II, III
6. migrate the final training recipe into the released MJLab codebase

This is `strong_inference`, not `direct_evidence`, but it is well supported by the structure of the paper and the released code.

## Step 0: Baseline Whole-Body Tracking Existed First

Evidence:

- The released MJLab repo still registers a generic tracking task and a standing variant side by side:
  - `Unitree-G1-Tracking`
  - `Unitree-G1-Tracking-Standing`
  - `Unitree-G1-1307-Stage-I`
  - `Unitree-G1-1307-Stage-II`
  - `Unitree-G1-1307-Stage-III`
  in [__init__.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/__init__.py>)

- The released standing config explicitly says it is:
  “Based on https://github.com/HybridRobotics/whole_body_tracking”
  in [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)

- Holosoma already had a generic FastSAC whole-body tracking preset:
  `exp:g1-29dof-wbt-fast-sac`
  in [experiment.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/experiment.py>)

Interpretation:

- `direct_evidence`: a baseline tracking task existed before the special `1307` stages.
- `strong_inference`: the released work likely grew from a standard tracking pipeline rather than being designed as Stage I/II/III from day one.

## Step 1: High-Dynamic Ground Clips Exposed a Baseline Failure Mode

Evidence:

- The paper motivates the dataset as pushing beyond AMASS/LAFAN-style difficulty and highlights failure during high-dynamic execution in [Introduction](https://arxiv.org/html/2602.13656v1#S1) and [Section 2.2](https://arxiv.org/html/2602.13656v1#S2.SS2).
- The paper frames fall recovery as necessary because high-dynamic motions increase failure rates in [Section 1](https://arxiv.org/html/2602.13656v1#S1) and [Section 2.3](https://arxiv.org/html/2602.13656v1#S2.SS3).
- The paper’s hard-case ablation in [Section 5.2](https://arxiv.org/html/2602.13656v1#S5.SS2) reports a severe baseline failure on the hardest sequence before adding recovery-oriented components.

Interpretation:

- `direct_evidence`: the authors viewed some clips as too difficult for a standard tracking setup.
- `strong_inference`: clips like `1307` likely became special because they exposed the gap between generic tracking and robust long-horizon execution.

## Step 2: Sampling Was Changed Before the Final Release Task Shape

Evidence from the paper:

- The paper introduces low-kinetic energy sampling in [Section 4.2](https://arxiv.org/html/2602.13656v1#S4.SS2) as a specific remedy for difficult motion learning.

Evidence from released code:

- Generic tracking uses `MotionCommandCfg`.
- Standing tasks switch to `MotionStandingCommandCfg`.
- `MotionStandingCommand` supports multiple sampling modes:
  - `start`
  - `uniform`
  - `adaptive`
  - `lke`
  in [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)

- The released `lke` implementation computes a softmin distribution over kinetic energy in `MotionLoader` and `compute_sampling_prob_softmin(...)`, then samples from that distribution in `_lke_sampling(...)` in [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)

Interpretation:

- `direct_evidence`: sampling became an explicit design axis in the released system.
- `strong_inference`: one of the earliest research changes after baseline tracking was likely “do not sample all timesteps equally.”

Why this matters:

- This is one of the clearest examples where the paper gives the idea, but the release code gives the operational implementation.
- Our original Holosoma paper-facing path implemented low-kinetic sampling differently, because the paper did not disclose all implementation details; see [02_paper_to_code_spec.md](./02_paper_to_code_spec.md).

## Step 3: The Problem Split Into Two Related Directions

At this point, the public artifacts suggest two connected but non-identical research directions.

### Direction A: Paper-Facing Recovery Logic

Evidence:

- The paper explicitly proposes a unified paradigm for motion tracking plus autonomous fall recovery in [Section 4](https://arxiv.org/html/2602.13656v1#S4).
- Recovery reward terms and GRSI are described in:
  - [Section 4.3.2](https://arxiv.org/html/2602.13656v1#S4.SS3.SSS2)
  - [Section 4.5](https://arxiv.org/html/2602.13656v1#S4.SS5)
- Our Holosoma implementation of this interpretation is documented in:
  - [02_paper_to_code_spec.md](./02_paper_to_code_spec.md)
  - [07_paper_alignment_evidence.md](./07_paper_alignment_evidence.md)

This leads to the paper-facing Holosoma recovery preset:

- `exp:g1-29dof-wbt-recovery-fast-sac`

Its defining characteristics are:

- a recovery initialization dataset
- recovery phase switching
- recovery-specific penalties
- recovery-aware bad-tracking hysteresis

See:

- [command.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/command.py>)
- [reward.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/reward.py>)
- [termination.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/termination.py>)

### Direction B: Released Standing Curriculum

Evidence:

- The released code defines a separate standing task family with:
  - dedicated standing observations
  - mixed standing/tracking resets
  - tolerant termination
  - Stage I/II/III variants

See:

- [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)
- [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)
- [env_cfgs.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/env_cfgs.py>)

Interpretation:

- `direct_evidence`: the release operationalizes a standing curriculum rather than exposing only a single hybrid recovery task.
- `strong_inference`: the final released training recipe for the hardest Ground case is not just “the paper reward plus recovery dataset.” It is a more specialized task family.

## Step 4: Standing-Like Reset Mixing Was a Major Operational Upgrade

Evidence:

- The released standing command loads `robot_init_states_8192.pth` and uses `tracking_standing_weight=(1.0, 1.0)` in [env_cfgs.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/env_cfgs.py>)
- `MotionStandingCommand` samples whether each reset is a standing task and, if so, injects root/joint state from the standing-init dataset in [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)

Interpretation:

- `direct_evidence`: released standing tasks are not pure motion-timestep resets.
- `strong_inference`: once the authors saw that difficult clips required more than timestep sampling, they added a second reset source to stabilize learning from upright-capable states.

Important note:

- This mechanism is barely visible from the paper alone.
- It is one of the clearest places where release code contains important training logic that the paper does not fully spell out.

## Step 5: Reward Design Shifted From Generic Tracking Toward Standing Stability

Evidence from the paper:

- Section 4.3 splits reward design into:
  - high-speed motion stability reward
  - autonomous fall recovery reward
  [Section 4.3](https://arxiv.org/html/2602.13656v1#S4.SS3)

Evidence from released code:

- The standing base reward includes:
  - global root position/orientation tracking
  - relative body position/orientation tracking
  - body linear/angular velocity tracking
  - action rate
  - joint limit
  - self collision
  - electrical power cost
  - shoulder-height penalty
  - root-orientation penalty
  - xy-rate-before-stand penalty
  in [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)

- Stage II and Stage III add `reward_center_of_mass` in the same file.

Interpretation:

- `direct_evidence`: the final standing recipe is a stability-aware imitation objective, not just a generic tracking reward.
- `strong_inference`: the authors likely found that pure imitation terms were not enough for the hardest Ground clips, so the reward surface was gradually biased toward stable standing/tracking.

## Step 6: Termination Became Tolerant Instead of Immediate

Evidence:

- The paper gives termination logic in [Section 4.4](https://arxiv.org/html/2602.13656v1#S4.SS4), but not the exact task-specific implementation details.
- The released standing task uses `TolerantTermination` with `bad_tracking_time_threshold_s=3.0` in [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)

Stage progression:

- Stage I uses looser z-only failure predicates.
- Stage II removes the z-only anchor/body terms, tightens orientation, and adds `anchor_pos` and `hip_dof`.
- Stage III keeps Stage II failure logic.

See:

- [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)

Interpretation:

- `direct_evidence`: the released recipe uses a tolerant standing-specific failure mechanism.
- `strong_inference`: this likely came after the authors observed that hard clips need a grace window instead of immediate failure.

## Step 7: The Hardest Case Was Frozen Into a Three-Stage Curriculum

Evidence:

- The release registers three separate `1307` tasks in [__init__.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/__init__.py>)
- Stage differences are explicit in [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)

Stage I:

- looser `anchor_pos_z`
- looser `ee_body_pos_z`
- no COM reward

Stage II:

- adds COM reward
- removes z-only body/anchor failure terms
- tightens `anchor_ori`
- adds `anchor_pos`
- adds `hip_dof`

Stage III:

- keeps Stage II reward/termination
- adds:
  - `terrain`
  - `reset_base`
  - stronger `push_robot`
  - `reset_robot_joints`

Interpretation:

- `direct_evidence`: the released code treats `1307` as a staged curriculum problem.
- `strong_inference`: this is likely the end state of multiple rounds of task shaping rather than the initial design.

## Step 8: The Final Public Release Was Migrated Into MJLab

Evidence:

- The released repo registers tasks in MJLab’s task registry in [__init__.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/__init__.py>)
- The released runner uses PPO, not FastSAC, in [rl_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/rl_cfg.py>)
- The standing configs and commands are rewritten for MJLab types and sensors in:
  - [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)
  - [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)

Interpretation:

- `direct_evidence`: the released system is a MJLab-native implementation.
- `strong_inference`: the research recipe likely existed conceptually before this final migration, but the public release exposes it only in the migrated MJLab form.

## What Probably Happened, Step by Step

This is the most defensible reconstruction of the research path.

### Phase A: Start from baseline tracking

Evidence level: `strong_inference`

- Begin with a generic whole-body tracking formulation.
- Use standard imitation rewards and standard reset sampling.
- Observe that hard Ground clips are unstable or fail to complete.

Why this is likely:

- both Holosoma and the released repo still contain a generic tracking task beside the specialized tasks
- the released standing code explicitly cites a prior whole-body tracking codebase

### Phase B: Change timestep sampling

Evidence level: `direct_evidence`

- Introduce low-kinetic or non-uniform sampling to avoid wasting capacity on difficult transitions too early.
- This is formalized in the paper and concretely implemented in release code.

### Phase C: Add stability-aware standing reward terms

Evidence level: `direct_evidence`

- Retain imitation terms
- add stability-biased penalties and regularizers
- eventually add COM support reward on harder stages

### Phase D: Add alternate reset sources

Evidence level: `direct_evidence`

- Introduce a second reset source for standing-capable initialization
- mix tracking-like and standing-like resets via `tracking_standing_weight`

This is a large practical step beyond what the paper alone makes obvious.

### Phase E: Add tolerant termination

Evidence level: `direct_evidence`

- stop terminating immediately on hard but recoverable transient errors
- allow bad tracking for a grace window while training standing robustness

### Phase F: Freeze the hardest clip into Stage I/II/III

Evidence level: `direct_evidence`

- Stage I: solvability
- Stage II: tighter tracking plus COM and posture constraints
- Stage III: robustness under stronger disturbances and reset events

### Phase G: Release the final recipe in MJLab

Evidence level: `direct_evidence`

- register the three stages as top-level tasks
- expose the final operational recipe in MJLab config files

## Where Holosoma Recovery Fits In

Our `exp:g1-29dof-wbt-recovery-fast-sac` should be understood as a paper-facing implementation of the paper’s recovery idea, not as a literal copy of the later released standing curriculum.

That explains why it differs from the released `1307` Stage III path:

- it follows the paper’s hybrid recovery framing more directly
- it uses a recovery-init dataset rather than the release standing-init dataset
- it uses recovery-phase reward/termination logic
- it was written from paper evidence before the author released the MJLab training recipe

See:

- [02_paper_to_code_spec.md](./02_paper_to_code_spec.md)
- [07_paper_alignment_evidence.md](./07_paper_alignment_evidence.md)

## Why The Paper And Release Code Are Not 1:1

The evidence suggests this is mainly an issue of specification depth, not contradiction.

The paper gives:

- the method
- the motivation
- the main reward/termination/recovery ideas
- the experimental story

The release code adds:

- exact observation surface
- exact reset mixing
- exact tolerant termination structure
- explicit `1307` stage curriculum
- standing-init dataset consumption
- exact event-level robustness configuration

Conclusion:

- `direct_evidence`: the release code contains important training logic not fully specified in the paper.
- `strong_inference`: the release code is best viewed as a more operational and more complete realization of the paper, not a contradiction of it.

## Practical Takeaway For This Repo

If the goal is:

- **paper-facing reproduction**
  use the Holosoma stand and recovery presets documented in [08_full_training_and_evaluation_guide.md](./08_full_training_and_evaluation_guide.md)

- **release-facing reproduction**
  use the additive:
  - `exp:g1-29dof-kfa-1307-stage1-fast-sac`
  - `exp:g1-29dof-kfa-1307-stage2-fast-sac`
  - `exp:g1-29dof-kfa-1307-stage3-fast-sac`

The important lesson is that these are not the same thing:

- the former reconstructs the paper’s recovery logic
- the latter reconstructs the released operational standing curriculum

## Final Assessment

The hypothesis

“the author likely started from a baseline whole-body tracking task, then built the `1307` staged standing tasks, then released the final recipe in MJLab”

is not directly proven, but it is the best evidence-supported reconstruction currently available from public artifacts.

What is proven:

- baseline tracking existed
- the paper introduced low-kinetic sampling, recovery reward logic, recovery termination ideas, and GRSI
- the released code implements a staged standing curriculum for `1307`
- the released code contains additional concrete machinery not fully described in the paper

What remains inference:

- the exact chronological order in the private research process
- whether the author first implemented this in a Holosoma-like FastSAC codebase before the MJLab migration
- whether the recovery paper path and the standing release path were developed sequentially or partly in parallel

Until the authors publish more implementation notes or earlier internal code, this is the strongest reconstruction that can be made without overclaiming.
