# Holosoma Training Framework

Core training framework for humanoid robot reinforcement learning with support for locomotion (velocity tracking) and whole-body tracking tasks.

| **Category** | **Supported Options** |
|-------------|----------------------|
| **Simulators** | IsaacGym, IsaacSim, MJWarp (training) \| Mujoco (evaluation) |
| **Algorithms** | PPO, FastSAC |
| **Robots** | Unitree G1, Booster T1 |

## Training

All training/eval scripts support `--help` for discovering available flags, e.g. `python src/holosoma/holosoma/train_agent.py --help`.

> **Note:** Video recording is enabled by default with `logger:wandb`. On headless servers, you may need to disable video or configure rendering. See [Video Recording](#video-recording) below.

### Locomotion (Velocity Tracking)

Train robots to track velocity commands.

```bash
# G1 with FastSAC on IsaacGym
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1

# T1 with PPO on IsaacSim
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1
```

Once checkpoints are saved, you can evaluate policies using [In-Training Evaluation](#in-training-evaluation) (same simulator as training) or cross-simulator evaluation in MuJoCo (see [holosoma_inference](../holosoma_inference/README.md)).

### MJWarp Training for Locomotion (Velocity Tracking)

Train using the MJWarp simulator (GPU-accelerated MuJoCo). **Note: MJWarp support is in beta.**

```bash
# G1 with FastSAC
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:mjwarp \
    logger:wandb

# G1 with PPO
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof \
    simulator:mjwarp \
    logger:wandb

# T1 with FastSAC
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof-fast-sac \
    simulator:mjwarp \
    logger:wandb

# T1 with PPO
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof \
    simulator:mjwarp \
    logger:wandb \
    --terrain.terrain-term.scale-factor=0.5  # required to avoid training instabilities

```

> **Note:**
> - MJWarp uses `nconmax=96` (maximum contacts per environment) by default. This can be adjusted via `--simulator.config.mujoco-warp.nconmax-per-env=96` if needed.
> - These examples use `--training.num-envs=4096`, but you may need to adjust this value based on your hardware.
> - When training T1 with PPO on mixed terrain, use `--terrain.terrain-term.scale-factor=0.5` to avoid training instabilities.


### Whole-Body Tracking

Train robots to track full-body motion sequences.

**Note**: Currently only supported for Unitree G1 / IsaacSim.

```bash
# G1 with FastSAC
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-fast-sac \
    logger:wandb

# KungFuAthleteBot / paper-facing Ground stand preset
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-stand-fast-sac \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1317.npz

# Recovery parity preset with GRSI resets
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-recovery-fast-sac \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1317.npz \
    --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.enabled=True \
    --command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.dataset_path=./artifacts/recovery_init/g1_ground_v1.npz

# G1 with PPO
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb

# Custom motion file
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/<your file>.npz"

# Visualize the motion file in isaacsim before training
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    --training.headless=False \
    --training.num_envs=1
```

Once checkpoints are saved, you can evaluate policies using [In-Training Evaluation](#in-training-evaluation) (same simulator as training) or cross-simulator evaluation in MuJoCo (see [holosoma_inference](../holosoma_inference/README.md)).

The `exp:g1-29dof-wbt-stand-fast-sac` preset is the paper-facing entrypoint for the released KungFuAthlete Ground workflow. It switches to the dedicated `robot:g1-29dof-kfa` asset/config surface, which exposes `head_link` while leaving the generic WBT presets unchanged.

Paper-facing recovery ablations are also available:
- `exp:g1-29dof-wbt-recovery-only-fast-sac`
- `exp:g1-29dof-wbt-recovery-slip3-fast-sac`
- `exp:g1-29dof-wbt-recovery-slip5-fast-sac`

For reproducible Ground parity validation, generate the suite manifest and per-run artifact layout with:

```bash
python src/holosoma/holosoma/run_kfa_ground_parity.py \
    --dataset-root=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj \
    --output-dir=./artifacts/parity_ground
```

This helper defaults to seeds `1,2,3` and the full paper-facing preset family, and writes one `run_manifest.yaml` per preset/seed pair.
The generated `suite_manifest.yaml` also records the one-seed Ground readiness gate; implementation is only considered complete after `stand` and `recovery` both pass train, same-sim eval, ONNX export, and MuJoCo launch.
Treat the local KungFuAthlete clone as:
- `collection_g129dof/org_smoothed_mj`: default training and evaluation motions
- `collection_g129dof/org_smoothed`: qpos source data and conversion provenance
- `kungfu_gvhmr_video_demo/...`: mocap and visualization data only

For the staged Ground-only workflow, see [06_training_roadmap.md](/home/yangfengzzz/Desktop/holosoma/docs/fall_recovery/06_training_roadmap.md).

---

## Evaluation

### In-Training Evaluation

For evaluating policies with the exact same configuration used during training (same simulator, environment settings, etc.):

```bash
# Evaluate checkpoint from Wandb
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=wandb://<ENTITY>/<PROJECT>/<RUN_ID>/<CHECKPOINT_NAME>
# e.g., --checkpoint=wandb://username/fastsac-t1-locomotion/abcdefgh/model_0010000.pt

# Evaluate local checkpoint
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=<CHECKPOINT_PATH>
# e.g., --checkpoint=/home/username/checkpoints/fastsac-t1-locomotion/model_0010000.pt
```

This evaluation mode:
- Automatically loads the training configuration from the checkpoint
- Runs evaluation in the same simulator and environment as training
- Can export policies to ONNX format (via `--training.export_onnx=True`)
- For locomotion evaluation, supports interactive velocity commands via keyboard (when simulator window is active):
  - `w`/`a`/`s`/`d`: linear velocity commands
  - `q`/`e`: angular velocity commands
  - `z`: zero velocity command

For released KungFuAthlete Ground parity runs, use clip `1307` as the default hard regression clip:

```bash
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=<CHECKPOINT_PATH> \
    --command.setup_terms.motion_command.params.motion_config.motion_file=./KungFuAthleteBot/collection_g129dof/org_smoothed_mj/1307.npz
```

In this workspace, the optional follow-up clip is `203.npz`.

### Cross-Simulator Evaluation (MuJoCo)

For testing trained policies in MuJoCo simulation or deploying to real robots, see the [holosoma_inference documentation](../holosoma_inference/README.md). This covers:
- Sim-to-sim evaluation (IsaacGym/IsaacSim → MuJoCo)
- Real robot deployment (both locomotion and WBT)

**Note**: ONNX policies are typically exported alongside `.pt` checkpoints during training, but can also be generated using the in-training evaluation script above.

## Advanced Configuration

The training system uses a hierarchical configuration system. The `exp` config serves as the main entry point with default configurations tuned for each algorithm and robot. You can customize training by overriding parameters on the command line.

> **Tip**: When composing Tyro configs, pass the `exp:<name>` preset before any other config fragments (e.g., `logger:wandb`). Tyro expects the base experiment to be declared first, and reversing the order can lead to confusing resolution errors.

### Logging with Weights & Biases

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof \
    simulator:isaacsim \
    --training.seed 1 \
    --algo.config.use-symmetry=False \
    logger:wandb \
    --logger.project locomotion-g1-29dof-ppo \
    --logger.name ppo-without-symmetry-seed1
```

### Video Recording

Video recording is **enabled by default** when using `logger:wandb`. Videos are recorded periodically and uploaded to Weights & Biases.

**Configuration:**
```bash
# Disable video recording
--logger.video.enabled False

# Adjust recording interval (episodes)
--logger.video.interval 10

# Change resolution
--logger.video.width 640 --logger.video.height 360
```

**Troubleshooting Headless Environments:**

If training fails on headless servers with display/rendering errors (e.g., `GLXBadFBConfig`, `eglInitialize failed`, `GLFW initialization failed`):

- **IsaacSim:** Disable video with `--logger.video.enabled False`, or force EGL with `DISPLAY= python ...`, or use virtual display with `xvfb-run -a python ...`
- **MJWarp/MuJoCo:** Set environment variable before training: `export MUJOCO_GL=egl`. See [MuJoCo docs](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl)
- **IsaacGym:** Usually works in headless environments. If issues occur, disable video with `--logger.video.enabled False`

### Terrain

```bash
# Use plane terrain instead of mixed terrain
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    terrain:terrain-locomotion-plane
```

### Multi-GPU Training

```bash
source scripts/source_isaacgym_setup.sh
torchrun --nproc_per_node=4 src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof-fast-sac \
    simulator:isaacgym \
    --training.num-envs 16384  # global/total number of environments
```

### Custom Reward Weights

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --reward.terms.tracking-lin-vel.weight=2.5 \
    --reward.terms.feet-phase.params.swing-height=0.12
```

### Observation Noise

```bash
# Disable observation noise
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --observation.groups.actor-obs.enable-noise=False
```

### Observation History Length

Some policies benefit from stacking multiple timesteps of observations. You can increase the history length used during training with:

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --observation.groups.actor_obs.history-length 4
```

Make sure to pass the same history length when running inference so the exported ONNX policy receives inputs with the correct shape.

### Curriculum Learning

```bash
# Disable curriculum
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --curriculum.setup-terms.penalty-curriculum.params.enabled=False

# Custom curriculum threshold (for shorter episodes)
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --simulator.config.sim.max-episode-length-s=10.0 \
    --curriculum.setup-terms.penalty-curriculum.params.level-up-threshold=350
```

### Domain Randomization

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --randomization.setup-terms.push-randomizer-state.params.enabled=False \
    --randomization.setup-terms.randomize-base-com-startup.params.enabled=True \
    --randomization.setup-terms.mass-randomizer.params.added-mass-range=[-1.0,3.0]
```
