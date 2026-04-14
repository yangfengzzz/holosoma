# 研究过程重建：从基础 Tracking 到公开发布的 1307 三阶段任务

本文档尝试重建 KungFuAthlete Ground 站立任务从“基础全身 tracking”演化到公开发布 `1307` Stage I / II / III 的可能研究过程。重建依据来自三类公开材料：

- 论文原文：[A Kung Fu Athlete Bot That Can Do It All Day](https://arxiv.org/html/2602.13656v1)
- 作者公开发布的 MJLab 代码：`KungFuAthleteBot/unitree_rl_mjlab`
- 本仓库中 Holosoma 的论文导向实现历史

本文目标不是凭空想象作者内部的真实研发时间线，而是尽量基于可验证证据，解释最终公开的 `1307` 三阶段任务是如何从一个基础 whole-body tracking 框架逐步演化成更工程化、更课程化的 standing / robustness 训练配方的。

## 阅读标准

本文使用三种证据标签：

- `direct_evidence`：论文中直接写明，或公开代码中直接可见
- `strong_inference`：材料里没有直接明说，但从代码结构和任务设计上高度可推断
- `open_inference`：合理猜测，但目前公开材料不足以严格证明

凡是不是直接证据的结论，本文都会明确标成推断，而不是当作事实陈述。

## 资料来源

主要参考来源如下：

- 论文方法与实验部分：
  - [Section 4: End-to-End Motion Tracking with Autonomous Fall Recovery](https://arxiv.org/html/2602.13656v1#S4)
  - [Section 4.2: Low Kinetic Energy Sampling](https://arxiv.org/html/2602.13656v1#S4.SS2)
  - [Section 4.3: Reward Design](https://arxiv.org/html/2602.13656v1#S4.SS3)
  - [Section 4.4: Termination Conditions](https://arxiv.org/html/2602.13656v1#S4.SS4)
  - [Section 4.5: GRSI](https://arxiv.org/html/2602.13656v1#S4.SS5)
  - [Section 5.2: Motion Tracking Performance / Ablation](https://arxiv.org/html/2602.13656v1#S5.SS2)

- 作者公开发布的 MJLab 代码：
  - [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)
  - [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)
  - [rewards.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/rewards.py>)
  - [env_cfgs.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/env_cfgs.py>)
  - [__init__.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/__init__.py>)
  - [rl_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/rl_cfg.py>)

- Holosoma 中按论文理解实现的版本：
  - [02_paper_to_code_spec.md](./02_paper_to_code_spec.md)
  - [07_paper_alignment_evidence.md](./07_paper_alignment_evidence.md)
  - [experiment.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/experiment.py>)
  - [command.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/command.py>)
  - [reward.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/reward.py>)
  - [termination.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/termination.py>)
  - [observation.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/observation.py>)

## 简短结论

当前最合理、也最符合证据的重建路径是：

1. 先有一个基础的 whole-body tracking 框架
2. 发现高动态 Ground 片段在标准 tracking 下很容易失败
3. 引入低动能采样与更偏稳定性的 reward
4. 引入 recovery state 初始化与 hybrid recovery 逻辑
5. 把最难的 Ground case 进一步固化为 Stage I / II / III 的 standing curriculum
6. 最后把成型后的训练配方迁移并发布在 MJLab 中

这一定性判断属于 `strong_inference`，不是论文或代码里逐句明说的事实；但它和现有论文结构、公开代码结构是高度一致的。

## Step 0：基础 Whole-Body Tracking 一定先存在

证据：

- 作者公开的 MJLab 仓库里，generic tracking task 和 standing / 1307 staged task 是并存的：
  - `Unitree-G1-Tracking`
  - `Unitree-G1-Tracking-Standing`
  - `Unitree-G1-1307-Stage-I`
  - `Unitree-G1-1307-Stage-II`
  - `Unitree-G1-1307-Stage-III`
  见 [__init__.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/__init__.py>)

- 公开 standing 配置文件顶部明确写了：
  “Based on https://github.com/HybridRobotics/whole_body_tracking”
  见 [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)

- Holosoma 里本来就有 generic FastSAC WBT 预设：
  `exp:g1-29dof-wbt-fast-sac`
  见 [experiment.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/experiment.py>)

解释：

- `direct_evidence`：`1307` 三阶段之前，基础 tracking 任务一定已经存在。
- `strong_inference`：作者后续工作大概率是从 standard tracking pipeline 继续演化出来的，而不是一开始就直接设计成三阶段 standing 任务。

## Step 1：高动态 Ground 片段暴露了基础 Tracking 的失效模式

证据：

- 论文在 [Introduction](https://arxiv.org/html/2602.13656v1#S1) 和 [Section 2.2](https://arxiv.org/html/2602.13656v1#S2.SS2) 中都在强调现有数据和方法对高动态动作不够，尤其容易在高动态执行中失败。
- 论文在 [Section 2.3](https://arxiv.org/html/2602.13656v1#S2.SS3) 里进一步说明：高动态动作导致 failure rate 上升，因此需要 recovery。
- 论文 [Section 5.2](https://arxiv.org/html/2602.13656v1#S5.SS2) 的 hardest sequence ablation 说明，基线方法在最难序列上会明显失败。

解释：

- `direct_evidence`：作者确实把某些高动态片段视为“标准 tracking 不足以解决”的 hard case。
- `strong_inference`：类似 `1307` 这样的 Ground clip，很可能就是因为暴露出 generic tracking 和真正长时鲁棒执行之间的差距，才被重点拿出来做专项课程化设计。

## Step 2：Sampling 很可能是最早被修改的机制之一

论文证据：

- 论文在 [Section 4.2](https://arxiv.org/html/2602.13656v1#S4.SS2) 明确提出了 low-kinetic energy sampling。

公开代码证据：

- generic tracking 使用的是 `MotionCommandCfg`
- standing task 换成了 `MotionStandingCommandCfg`
- `MotionStandingCommand` 支持多种采样模式：
  - `start`
  - `uniform`
  - `adaptive`
  - `lke`
  见 [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)

- 公开 release 中的 `lke` 不是简单口头概念，而是有具体实现：
  - 先计算 kinetic energy
  - 再经过 softmin 概率
  - 最后按这个分布采样
  同样见 [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)

解释：

- `direct_evidence`：sampling 在最终系统里是一个明确、重要的设计轴。
- `strong_inference`：作者在 baseline tracking 之后，最早改的很可能就是“不要再平均采样所有 timestep”。

为什么这一点重要：

- 这是论文和 release code 差异最明显的地方之一。
- 论文告诉我们“有 low-kinetic sampling 这个思路”，但 release code 才给出真正训练时用的 operational implementation。
- 我们当初在 Holosoma 里按论文实现 low-kinetic 时，就不得不自己补足这部分细节，见 [02_paper_to_code_spec.md](./02_paper_to_code_spec.md)。

## Step 3：问题后来分成了两条相关但不完全相同的路径

从公开材料看，后续研究至少分化成了两条方向。

### 方向 A：论文导向的 Recovery 路径

证据：

- 论文在 [Section 4](https://arxiv.org/html/2602.13656v1#S4) 中明确提出了“tracking + autonomous fall recovery”统一训练范式。
- recovery reward 和 GRSI 的描述主要在：
  - [Section 4.3.2](https://arxiv.org/html/2602.13656v1#S4.SS3.SSS2)
  - [Section 4.5](https://arxiv.org/html/2602.13656v1#S4.SS5)
- Holosoma 中我们按论文解读实现的结果整理在：
  - [02_paper_to_code_spec.md](./02_paper_to_code_spec.md)
  - [07_paper_alignment_evidence.md](./07_paper_alignment_evidence.md)

它对应的是现在这个 paper-facing recovery 预设：

- `exp:g1-29dof-wbt-recovery-fast-sac`

这个预设的核心特征是：

- recovery 初始化数据集
- recovery phase switching
- recovery-specific penalties
- recovery-aware bad-tracking hysteresis

相关本地实现见：

- [command.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/command.py>)
- [reward.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/reward.py>)
- [termination.py](</home/yangfengzzz/Desktop/holosoma/src/holosoma/holosoma/config_values/wbt/g1/termination.py>)

### 方向 B：公开 release 的 Standing Curriculum 路径

证据：

- 公开 release 不是只给一个 hybrid recovery task，而是给了专门的 standing family：
  - standing-specific observations
  - tracking / standing mixed resets
  - tolerant termination
  - Stage I / II / III

见：

- [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)
- [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)
- [env_cfgs.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/env_cfgs.py>)

解释：

- `direct_evidence`：release 公开出来的是一个更像 standing curriculum 的系统，而不是只公开 paper 中那个单一 hybrid recovery 任务。
- `strong_inference`：也就是说，最终对 hardest Ground case 的公开训练配方，已经不是“论文 reward + recovery dataset”这么简单，而是进一步发展成了更专门化的 staged standing family。

## Step 4：Standing-Like Reset Mixing 是非常关键的工程升级

证据：

- 公开 standing command 会加载 `robot_init_states_8192.pth`，并设置 `tracking_standing_weight=(1.0, 1.0)`，见 [env_cfgs.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/env_cfgs.py>)
- `MotionStandingCommand` 在 reset 时会先决定当前 env 是 tracking task 还是 standing task；如果是 standing task，就直接从 standing-init dataset 写入 root / joint 状态，见 [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)

解释：

- `direct_evidence`：release 中的 standing task 并不是纯粹从 motion timestep reset。
- `strong_inference`：作者很可能是在发现“只改 timestep sampling 还不够”之后，进一步加入了第二种 reset source，用 upright-capable states 稳定 hardest clips 的训练。

这一步尤其重要，因为：

- 从论文正文几乎无法严格重建这一机制
- 但在 release code 里，这其实是影响训练行为的关键设计之一

## Step 5：Reward 从 generic tracking 逐步偏向 standing stability

论文证据：

- [Section 4.3](https://arxiv.org/html/2602.13656v1#S4.SS3) 明确把 reward 分成：
  - high-speed motion stability reward
  - autonomous fall recovery reward

公开代码证据：

- standing base reward 包含：
  - global root pos / ori tracking
  - relative body pos / ori tracking
  - body linear / angular velocity tracking
  - action rate
  - joint limit
  - self collision
  - electrical power cost
  - shoulder-height penalty
  - root-orientation penalty
  - xy-rate-before-stand penalty
  见 [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)

- Stage II 和 Stage III 进一步加入了 `reward_center_of_mass`

解释：

- `direct_evidence`：最终 release 的 standing recipe 已经不是纯 imitation reward，而是明显偏向“稳定站住并跟踪”的 reward surface。
- `strong_inference`：这意味着作者大概率经历过一个阶段，发现 hardest Ground clips 仅靠 generic imitation terms 不够，必须往 stable standing / support / posture 方向加更多结构化约束。

## Step 6：Termination 从“立即失败”变成了“容错式失败”

证据：

- 论文 [Section 4.4](https://arxiv.org/html/2602.13656v1#S4.SS4) 给出了 termination 的总体思路，但没有把 standing task 的具体 task-level 细节完整展开。
- 公开 standing task 使用的是 `TolerantTermination`，并给了 `bad_tracking_time_threshold_s=3.0`，见 [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)

并且三阶段是分层次的：

- Stage I：更松的 z-only failure predicates
- Stage II：删除 z-only anchor/body terms，收紧 orientation，再加 `anchor_pos` 和 `hip_dof`
- Stage III：沿用 Stage II 的 failure logic

解释：

- `direct_evidence`：release 采用的是 standing-specific tolerant failure 机制，而不是立即 terminate。
- `strong_inference`：作者很可能是观察到 hardest clips 上很多“瞬时偏差”其实是可恢复的，因此才把 termination 改成了带宽限时间的形式。

## Step 7：最难 case 最终被固化成了 1307 三阶段课程

证据：

- release 在 [__init__.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/__init__.py>) 中注册了三个独立的 `1307` task
- 每个 stage 的差异都在 [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>) 里写得很明确

Stage I：

- 放宽 `anchor_pos_z`
- 放宽 `ee_body_pos_z`
- 不加 COM reward

Stage II：

- 加入 COM reward
- 删掉 z-only body / anchor failure term
- 收紧 `anchor_ori`
- 增加 `anchor_pos`
- 增加 `hip_dof`

Stage III：

- 保持 Stage II 的 reward / termination
- 再加：
  - `terrain`
  - `reset_base`
  - 更强的 `push_robot`
  - `reset_robot_joints`

解释：

- `direct_evidence`：release 把 `1307` 当成了需要 staged curriculum 的难任务，而不是一个单独静态任务。
- `strong_inference`：这非常像是作者经过多轮试验后，把 hardest Ground case 的经验总结为“先学可解性，再学精度，再学鲁棒性”的固定训练配方。

## Step 8：最终公开版本被迁移到了 MJLab

证据：

- release 最终是以 MJLab task registry 的形式注册在 [__init__.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/__init__.py>)
- release 使用的是 PPO，不是 FastSAC，见 [rl_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/config/g1/rl_cfg.py>)
- standing config / command / reward / event 都被重写成了 MJLab 的类型系统，见：
  - [tracking_standing_env_cfg.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/tracking_standing_env_cfg.py>)
  - [commands.py](</home/yangfengzzz/Desktop/KungFuAthleteBot/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py>)

解释：

- `direct_evidence`：最终公开版本就是一个 MJLab-native implementation。
- `strong_inference`：研究思路本身大概率先形成，再在公开前被迁移并整理成 MJLab 结构。

## 最可能的完整研究过程

下面是目前最符合证据的完整重建。

### Phase A：先从 baseline tracking 开始

证据等级：`strong_inference`

- 先有 generic whole-body tracking
- 用标准 imitation rewards 和标准 reset sampling
- 发现 hardest Ground clips 上很难长期稳定跟踪

为什么这个推断强：

- generic tracking task 和 standing task 在 release 中是并存的
- standing code 顶部还显式引用了更早的 whole_body_tracking 来源

### Phase B：先动 sampling

证据等级：`direct_evidence`

- 引入 low-kinetic 或至少非均匀 timestep sampling
- 目标是避免一开始就在 hardest transitions 上浪费大量训练容量

### Phase C：再引入稳定性更强的 standing reward

证据等级：`direct_evidence`

- 保留 imitation 主体
- 但加入 standing / posture / support / power / self-collision 这些稳定性约束
- 在更难阶段再加 COM reward

### Phase D：加入第二类 reset source

证据等级：`direct_evidence`

- 除了 motion-based reset，还引入了 standing-init reset
- 并通过 `tracking_standing_weight` 把两种 reset 混合

这一步是 release code 中非常关键、但论文不容易完整还原的训练逻辑。

### Phase E：把 failure 变成 tolerant failure

证据等级：`direct_evidence`

- 不再对短时 tracking error 立刻 terminate
- 给 standing task 一个容错窗口，让策略有机会从瞬时偏差里拉回来

### Phase F：把 hardest case 固化成 Stage I / II / III

证据等级：`direct_evidence`

- Stage I：先学“别马上失败”
- Stage II：再学更紧的 tracking 和 posture / COM 约束
- Stage III：最后再学更强扰动和 reset-level robustness

### Phase G：把最终训练配方迁移到 MJLab 并公开

证据等级：`direct_evidence`

- 把最终 recipe 注册成 `Unitree-G1-1307-Stage-I/II/III`
- 用 MJLab 的 env / command / reward / event / PPO runner 形式公开

## Holosoma 的 Recovery-Fast-SAC 在这个故事里是什么位置

我们这里的 `exp:g1-29dof-wbt-recovery-fast-sac` 更适合被理解为：

**按论文 recovery 思路重建出来的 paper-facing implementation**

而不是：

**后期公开的 standing curriculum 的逐行复刻**

所以它和 `stage3` 不一样是正常的。它的差异来自于：

- 它更直接遵循论文里的 hybrid recovery framing
- 它使用的是 recovery-init dataset，而不是 release standing-init dataset
- 它有 recovery-phase reward / termination 逻辑
- 它是在 release code 公布之前，按论文证据构建出来的

相关文档见：

- [02_paper_to_code_spec.md](./02_paper_to_code_spec.md)
- [07_paper_alignment_evidence.md](./07_paper_alignment_evidence.md)

## 为什么论文和 release code 不是 1:1

根据现有证据，更像是“说明粒度不同”，而不是“彼此矛盾”。

论文负责讲清楚：

- 方法动机
- 核心思路
- recovery / reward / termination 的高层设计
- 实验结论

release code 则补上了很多真正影响训练结果的 operational 细节：

- observation surface
- reset mixing
- standing-init dataset 消费方式
- tolerant termination 的具体结构
- `1307` 三阶段课程
- event-level robustness 配置

所以更准确的说法是：

- `direct_evidence`：release code 包含很多论文没有完整展开的训练细节
- `strong_inference`：release code 更像是论文思想的更完整、更工程化实现，而不是和论文原则相冲突的另一套方法

## 对本仓库的实际意义

如果目标是：

- **按论文复现**
  使用 Holosoma 里的 paper-facing stand / recovery 预设，见 [08_full_training_and_evaluation_guide.md](./08_full_training_and_evaluation_guide.md)

- **按 release 复现**
  使用新增的：
  - `exp:g1-29dof-kfa-1307-stage1-fast-sac`
  - `exp:g1-29dof-kfa-1307-stage2-fast-sac`
  - `exp:g1-29dof-kfa-1307-stage3-fast-sac`

最重要的是要意识到：

- 前者是在复现论文中的 recovery 叙事
- 后者是在复现作者最终公开出来的 standing curriculum recipe

二者相关，但不相同。

## 最终判断

你提出的假设：

“作者很可能先从 baseline whole-body tracking 出发，然后逐步写出三阶段任务，最后再迁移到 MJLab”

目前不能算 `direct_evidence`，但它是当前公开材料下最合理、最有证据支撑的重建版本。

已经可以确定的事实：

- baseline tracking 先存在
- 论文引入了 low-kinetic sampling、recovery reward、recovery termination 和 GRSI
- release code 确实实现了 `1307` 的 staged standing curriculum
- release code 确实包含了论文里没有完全展开的很多训练逻辑

仍然只是推断的部分：

- 作者私下真实的开发时间顺序
- 他是否真的先在 Holosoma/FastSAC 风格代码里完成过这一套，再迁移到 MJLab
- paper-facing recovery 路线和 standing release 路线到底是串行演化还是部分并行开发

在作者公开更多中间版本代码或额外说明之前，这已经是目前最稳妥、也最不容易误导的研究过程重建。
