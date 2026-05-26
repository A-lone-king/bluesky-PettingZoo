# 训练指南

本文档介绍如何使用项目提供的脚本进行各种强化学习训练。

## 目录

- [概述](#概述)
- [环境准备](#环境准备)
- [快速验证：烟雾测试](#快速验证烟雾测试)
- [单场景训练](#单场景训练)
- [批量训练](#批量训练)
- [评估模型](#评估模型)
- [配置文件说明](#配置文件说明)
- [模型输出结构](#模型输出结构)
- [常见问题](#常见问题)

---

## 概述

### 支持的算法

| 算法 | 类型 | 动作空间 | 说明 |
|------|------|----------|------|
| **PPO** | On-policy | discrete / continuous | 默认算法，适合快速验证 |
| **SAC** | Off-policy | continuous（自动切换） | 最大熵策略，探索能力强 |
| **TD3** | Off-policy | continuous（自动切换） | 双 Q 网络，减少过估计 |
| **DDPG** | Off-policy | continuous（自动切换） | 确定性策略梯度 |

> SAC/TD3/DDPG 会自动将动作空间切换为 `continuous`，无需手动指定。

### 支持的场景

| 场景名 | 飞机数 | 控制模式 | 动作维度 | 说明 |
|--------|--------|----------|----------|------|
| `HorizontalCR` | 5 | MULTI_RL | heading | 同高度对头冲突解脱 |
| `VerticalCR` | 5 | MULTI_RL | altitude | 垂直高度层冲突解脱 |
| `SectorCR` | 5 | MULTI_RL | heading+speed | 扇区边界冲突解脱 |
| `WaypointNav` | 3 | MULTI_RL | heading | 无冲突航路点导航（基线） |
| `Merge` | 20 | SINGLE_RL | all | 汇合进近（1 可控 + 19 背景） |
| `Descent` | 3 | SINGLE_RL | altitude | 下降阶段冲突解脱 |
| `StaticObstacle` | 1 | MULTI_RL | heading+speed | 静态障碍物避让 |
| `SectorCapacity` | 6 | MULTI_RL | heading+speed | 扇区容量限制 |
| `RouteNav` | 3 | MULTI_RL | heading+speed | 交叉航路导航 |
| `PlanWaypoint` | 1 | MULTI_RL | heading | 5 航路点顺序到达 |

### 仿真后端

所有训练和测试均使用真实 BlueSky 仿真引擎（BlueSkyWrapper）。需要先安装 BlueSky：

```bash
pip install "bluesky-simulator[full]"
```

---

## 环境准备

```bash
# 安装项目依赖
pip install -e .

# 验证安装
python -c "from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv; print('OK')"
```

---

## 快速验证：烟雾测试

在正式训练前，先运行烟雾测试验证训练管线是否正常工作：

```bash
python scripts/train_smoke_test.py
```

该脚本会：
1. 创建 WaypointNav 场景（1 架飞机，无冲突）
2. 使用 PPO 训练 10,000 步
3. 对比训练前后的平均奖励
4. 输出 `SUCCESS`（奖励提升）或 `WARNING`（未检测到提升）

预期输出：
```
[Before training] Mean reward: -XX.XX
Training for 10,000 timesteps...
[After training]  Mean reward: -XX.XX
Improvement: +XX.XX
SUCCESS: PPO learned to improve rewards.
```

---

## 单场景训练

### 基本命令格式

```bash
python scripts/train_ppo_scenarios.py --scenario <场景名> --algorithm <算法> --timesteps <步数>
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scenario` | `HorizontalCR` | 场景名称（见上表） |
| `--algorithm` | `PPO` | 算法：PPO / SAC / TD3 / DDPG |
| `--action-space` | `discrete` | 动作空间：discrete / continuous |
| `--timesteps` | `500000` | 总训练步数 |
| `--num-aircraft` | `3` | 飞机数量 |
| `--max-steps` | `50` | 每回合最大步数 |
| `--seed` | `42` | 随机种子 |
| `--save-dir` | `models` | 模型保存目录 |
| `--resume` | `None` | 从检查点恢复（路径） |
| `--render` | `False` | 启用 Pygame 渲染 |

### 训练示例

**PPO 离散动作（默认）：**

```bash
python scripts/train_ppo_scenarios.py \
    --scenario HorizontalCR \
    --algorithm PPO \
    --timesteps 100000 \
    --num-aircraft 5
```

**SAC 连续动作（自动切换）：**

```bash
python scripts/train_ppo_scenarios.py \
    --scenario VerticalCR \
    --algorithm SAC \
    --timesteps 200000
```

**TD3 连续动作：**

```bash
python scripts/train_ppo_scenarios.py \
    --scenario SectorCR \
    --algorithm TD3 \
    --timesteps 200000
```

**DDPG 连续动作：**

```bash
python scripts/train_ppo_scenarios.py \
    --scenario Merge \
    --algorithm DDPG \
    --timesteps 200000
```

**启用渲染（可视化训练过程）：**

```bash
python scripts/train_ppo_scenarios.py \
    --scenario PlanWaypoint \
    --timesteps 50000 \
    --render
```

---

## 批量训练

### 全场景全算法批量训练

```bash
python scripts/train_all_algos.py --timesteps 200000
```

默认在所有 10 个场景上运行 SAC、TD3、DDPG 三种算法（共 30 个任务），每个任务超时 1800 秒。

> PPO 不在批量脚本中，因为 PPO 通常作为基线单独运行。

### 自定义算法和场景

```bash
# 只运行 SAC 和 TD3
python scripts/train_all_algos.py --timesteps 200000 --algos SAC TD3

# 只在指定场景上训练
python scripts/train_all_algos.py --timesteps 200000 --scenarios HorizontalCR VerticalCR

# 组合使用
python scripts/train_all_algos.py --timesteps 100000 --algos SAC --scenarios HorizontalCR PlanWaypoint
```

### 断点续训

批量脚本会自动跳过已有 `checkpoint_final.zip` 的任务。如需重新训练，删除对应的模型目录即可。

---

## 评估模型

### 基线评估（Random / RuleBased / PPO）

```bash
python scripts/evaluate_baselines.py --scenario HorizontalCR
```

指定已训练模型：

```bash
python scripts/evaluate_baselines.py \
    --scenario HorizontalCR \
    --model models/HorizontalCR/PPO/checkpoint_final.zip
```

### 全算法评估

评估一个场景下所有已训练的算法：

```bash
python scripts/evaluate_all.py --scenario HorizontalCR
```

可选参数：

```bash
python scripts/evaluate_all.py \
    --scenario HorizontalCR \
    --model-dir models \
    --num-aircraft 5 \
    --episodes 20
```

> 未找到模型的算法会被自动跳过，不会报错。

### 一站式：训练 + 评估

```bash
python scripts/run_baselines.py \
    --scenario HorizontalCR \
    --timesteps 50000 \
    --episodes 10
```

该脚本先训练 PPO，再评估 Random / RuleBased / PPO 三个策略。

---

## 配置文件说明

### 算法超参数：`config/algorithms.yaml`

```yaml
PPO:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2

SAC:
  learning_rate: 3.0e-4
  buffer_size: 100000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  learning_starts: 100

TD3:
  learning_rate: 3.0e-4
  buffer_size: 100000
  batch_size: 100
  tau: 0.005
  gamma: 0.99
  policy_delay: 2

DDPG:
  learning_rate: 3.0e-4
  buffer_size: 100000
  batch_size: 128
  tau: 0.005
  gamma: 0.99
```

### 奖励配置：`config/rewards.yaml`

| 组件 | 默认权重 | 说明 |
|------|----------|------|
| `conflict` | 1.0 | NMAC/告警/间隔惩罚 |
| `drift_penalty` | 0.5 | 航向偏移惩罚 |
| `smoothness` | 0.0 | 动作平滑惩罚 |
| `efficiency` | 0.3 | 到达奖励 + 偏离惩罚 |
| `obstacle_intrusion` | 1.0 | 障碍物入侵惩罚 |
| `capacity` | 1.0 | 扇区容量超限惩罚 |
| `delay` | 0.2 | 延误惩罚 |
| `flow_efficiency` | 0.2 | 流量效率奖励 |
| `fairness` | 0.1 | 公平性惩罚 |

### 场景配置：`config/scenarios/*.yaml`

场景配置覆盖默认奖励权重，例如：

```yaml
# config/scenarios/horizontal_cr.yaml
scenario: HorizontalCR
num_aircraft: 5
seed: 42
action_space: discrete
control_mode: MULTI_RL
conflict_generation: creconfs
reward_overrides:
  conflict:
    weight: 0.7
  efficiency:
    weight: 0.3
```

可用场景配置：`horizontal_cr.yaml`、`vertical_cr.yaml`、`sector_cr.yaml`、`plan_waypoint.yaml`

### 仿真默认参数：`config/default.yaml`

包含仿真步长、空域定义、飞机生成范围、观测空间、动作空间、归一化参数等。

---

## 模型输出结构

```
models/
  {Scenario}/
    {Algorithm}/
      checkpoint_10000.zip      # 中间检查点
      checkpoint_10000.json     # 检查点元数据
      checkpoint_20000.zip
      checkpoint_20000.json
      ...
      checkpoint_final.zip      # 最终模型
      checkpoint_final.json     # 最终模型元数据
      logs/
        training_log.csv        # 训练日志（每回合一行）
```

`training_log.csv` 列：`timestep`, `episode`, `reward`, `episode_length`, `conflicts`, `arrivals`, `algorithm`, `action_space`, `timestamp`

---

## 常见问题

### 训练速度慢

- 减少 `--num-aircraft` 和 `--max-steps` 可加速单回合
- 减少 `--timesteps` 进行快速验证
- BlueSky 仿真引擎有初始化开销，首次运行较慢

### SAC/TD3/DDPG 报错 action space 相关

这三个算法自动切换为 `continuous` 动作空间，无需手动指定 `--action-space continuous`。

### 如何复现结果

使用相同的 `--seed`（默认 42）可保证可复现性。

### 检查点太多磁盘空间不足

`CheckpointManager` 默认保留最近 5 个检查点，自动轮转删除旧的。可通过修改 `train_ppo_scenarios.py` 中的 `max_checkpoints` 参数调整。

### 如何自定义奖励权重

编辑 `config/rewards.yaml` 修改全局权重，或编辑 `config/scenarios/*.yaml` 中的 `reward_overrides` 为特定场景覆盖权重。

### 如何添加新场景

1. 在 `src/bluesky_pettingzoo/envs/scenarios/` 下创建新场景类，继承 `BaseScenario`
2. 在 `scripts/train_ppo_scenarios.py` 的 `SCENARIO_MAP` 中注册场景名
3. 在 `scripts/train_ppo_scenarios.py` 的 `_resolve_scenario()` 中添加导入映射
4. （可选）在 `config/scenarios/` 下创建 YAML 配置
