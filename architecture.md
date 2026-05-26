# bluesky-pettingzoo 架构文档

## 项目概述

bluesky-pettingzoo 是一个面向空中交通管理（ATM）领域的多智能体强化学习环境。项目基于 BlueSky 仿真引擎提供真实的飞行动力学模拟，通过 PettingZoo ParallelEnv 标准实现多智能体接口，并集成 Stable-Baselines3 进行策略训练。

**核心特性：**
- 10 个 ATM 场景（冲突解脱、航路导航、进近汇合等）
- 9 个可插拔奖励组件
- 离散/连续双动作空间
- 真实 BlueSky 仿真引擎集成
- PPO/SAC/TD3/DDPG 四种算法支持
- Pygame 可视化渲染

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    训练脚本层 (scripts/)                      │
│  train_ppo_scenarios.py │ evaluate_baselines.py │ ...       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               Stable-Baselines3 算法层                       │
│              PPO / SAC / TD3 / DDPG                          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│            SingleAgentGymWrapper (单智能体包装)               │
│         将 ParallelEnv 转为 gymnasium.Env                     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              BlueSkyMARLEnv (核心环境)                        │
│                  PettingZoo ParallelEnv                      │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │ 场景     │ 观测     │ 动作     │ 奖励     │ 渲染     │   │
│  │ Scenario │ ObsMgr   │ Action   │ Reward   │ Renderer │   │
│  │          │          │ Trans    │ Calc     │          │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              BlueSkyWrapper (仿真引擎接口)                    │
│           单位转换 │ 命令批处理 │ 状态查询                     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                BlueSky 仿真引擎                              │
│        飞行动力学 │ 冲突检测 │ 导航数据库 │ 大气模型           │
└─────────────────────────────────────────────────────────────┘
```

---

## 模块详解

### 1. 环境核心 (`envs/`)

#### `parallel_env.py` — BlueSkyMARLEnv

系统的核心入口，继承 PettingZoo `ParallelEnv`，负责：

- **生命周期管理**：`reset()` 初始化场景和飞机，`step()` 推进仿真，`close()` 清理资源
- **动作分发**：收集所有 Agent 的动作，通过 `ActionTranslator` 转为 BlueSky 命令，批量发送
- **观测生成**：通过 `ObservationManager` 为每个 Agent 生成局部观测
- **奖励计算**：通过 `RewardCalculator` 汇总所有奖励组件
- **终止判断**：检测到达目标、NMAC、飞机离开空域、障碍物入侵等终止条件
- **动态进入**：支持仿真过程中新飞机加入

```python
# 典型使用流程
env = BlueSkyMARLEnv(config, wrapper, obs_manager, action_translator, 
                     reward_calculator, rewards_config, scenario=scenario)
obs, info = env.reset()
for _ in range(max_steps):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
```

---

### 2. BlueSky 仿真接口 (`bluesky/`)

#### `wrapper.py` — BlueSkyWrapper

封装 BlueSky 仿真引擎的 headless 模式操作：

| 方法 | 功能 |
|------|------|
| `init_simulation()` | 初始化 BlueSky（`bs.init(mode='sim', detached=True)`） |
| `create_aircraft(acid, actype, lat, lon, alt, hdg, spd)` | 创建飞机（自动 TAS→CAS 转换） |
| `remove_aircraft(acid)` | 删除飞机 |
| `step()` / `step_n(n)` | 推进仿真 1/n 步 |
| `get_aircraft_state(acid)` | 查询单架飞机状态（返回英尺/节） |
| `get_all_aircraft_states()` | 查询所有飞机状态 |
| `send_command(cmd)` / `send_commands_batch(cmds)` | 发送 BlueSky 命令 |
| `create_conflict_aircraft(...)` | 使用 `creconfs` 生成冲突场景 |
| `is_aircraft_in_airspace(acid)` | 检查飞机是否在空域内 |
| `close()` | 清理托管飞机 |

**单位约定**：外部接口使用英尺（altitude）、节（speed）、英尺/分钟（vs）；BlueSky 内部使用米和米/秒，转换在 wrapper 层完成。

---

### 3. 场景系统 (`envs/scenarios/`)

#### `base.py` — BaseScenario

所有场景的抽象基类，定义统一接口：

| 抽象方法 | 功能 |
|----------|------|
| `setup(wrapper)` | 在 BlueSky 中创建飞机、设置初始状态 |
| `get_spawn_config()` | 返回飞机生成参数（高度/速度/航向范围） |
| `get_conflict_config()` | 返回冲突生成参数 |
| `should_truncate(aircraft_states)` | 判断是否应截断（如离开空域） |
| `get_waypoint(acid)` | 返回指定飞机的目标航路点 |

可选方法：`update()`（每步更新）、`reset()`（重置场景状态）、`create_intruders()`（创建入侵飞机）、`get_background_actions()`（背景交通动作）、`get_priority()`（飞机优先级）。

支持 YAML 配置加载：`BaseScenario.from_config("config/scenarios/horizontal_cr.yaml")`。

#### 10 个场景

| 场景 | 飞机数 | 控制模式 | 动作维度 | 描述 |
|------|--------|----------|----------|------|
| **HorizontalCR** | 5 | MULTI_RL | heading | 同高度巡航，航向机动避让对头冲突 |
| **VerticalCR** | 5 | MULTI_RL | altitude | 相似水平位置，垂直机动避让 |
| **SectorCR** | 5 | MULTI_RL | heading+speed | 扇区内冲突解脱，离开扇区截断 |
| **WaypointNav** | 3 | MULTI_RL | heading | 无冲突航路点导航（基线场景） |
| **PlanWaypoint** | 1 | MULTI_RL | heading | 单飞机顺序访问 5 个航路点 |
| **StaticObstacle** | 1 | MULTI_RL | heading+speed | 禁飞区规避 |
| **SectorCapacity** | 6 | MULTI_RL | heading+speed | 扇区容量约束下的导航 |
| **RouteNav** | 3 | MULTI_RL | heading+speed | 交叉航路导航 |
| **Merge** | 20 | SINGLE_RL | all | 进近汇合（1 可控 + 19 背景） |
| **Descent** | 3 | SINGLE_RL | altitude | 下降阶段冲突解脱 |

**控制模式**：
- `MULTI_RL`：所有飞机均由 RL 策略控制
- `SINGLE_RL`：仅 ego 飞机由 RL 控制，其余使用预设动作

---

### 4. 奖励系统 (`rewards/`)

#### 架构

```
RewardCalculator
├── ConflictPenalty (weight=1.0)
├── DriftPenalty (weight=0.5)
├── SmoothnessPenalty (weight=0.5)
├── EfficiencyReward (weight=0.3)
├── CapacityPenalty (weight=1.0)
├── DelayPenalty (weight=0.2)
├── FlowEfficiencyReward (weight=0.2)
├── FairnessReward (weight=0.1)
└── ObstacleIntrusion (weight=1.0)
```

`RewardCalculator` 管理组件注册和加权求和。每个组件继承 `RewardComponent`，实现 `compute(aircraft_states, actions, prev_states)` 和 `reset()`。

#### 9 个奖励组件

| 组件 | 类 | 功能 |
|------|-----|------|
| **ConflictPenalty** | `ConflictPenalty` | 三级冲突惩罚：NMAC（<5NM/<1000ft）、告警（<10NM/<2000ft）、间隔违规。支持前瞻预测和链式冲突检测（BFS） |
| **EfficiencyReward** | `EfficiencyReward` | 航路效率：偏离惩罚（距离偏差×步长）、步进成本、到达奖励 |
| **SmoothnessPenalty** | `SmoothnessPenalty` | 动作平滑：任何非零调整均惩罚 |
| **DriftPenalty** | `DriftPenalty` | 航向偏离：`scale × |heading - bearing_to_goal|`（弧度） |
| **DelayPenalty** | `DelayPenalty` | 延误惩罚：超出预期到达时间后，每步按比例惩罚 |
| **CapacityPenalty** | `CapacityPenalty` | 容量惩罚：全局阈值或逐扇区容量超限惩罚 |
| **FlowEfficiencyReward** | `FlowEfficiencyReward` | 流量效率：单位时间通过扇区的飞机数越多，奖励越高 |
| **FairnessReward** | `FairnessReward** | 公平性：延误标准差越大，惩罚越重 |
| **ObstacleIntrusion** | `ObstacleIntrusion` | 障碍物入侵：进入禁飞区多边形的固定惩罚 |

---

### 5. 观测处理 (`observations/`)

#### 观测空间结构

```python
Dict({
    "self_state": Box(9,),          # 自身状态：lat, lon, alt, hdg, tas, vs, goal_dist, goal_bearing, sector_id
    "other_aircraft": Box(N, 10,),   # 他机状态：相对位置/速度 + mask
    "goal": Box(4,),                 # 目标：goal_lat, goal_lon, goal_dist, goal_bearing
    "obstacles": Box(M, 4,),         # 障碍物边界（可选）
    "waypoints": Box(K, 3,),         # 航路点序列（可选）
})
```

#### 模块

| 模块 | 类 | 功能 |
|------|-----|------|
| `filters.py` | `PerceptionFilter` | 按水平半径、垂直范围、最大可观测数过滤飞机，按距离排序 |
| `normalizer.py` | `Normalizer` | 将原始值映射到 [-1, 1]，支持航向/方位的 circular 编码（cos/sin） |
| `manager.py` | `ObservationManager` | 集成过滤和归一化，生成完整观测包 |

---

### 6. 动作翻译 (`actions/`)

#### 离散动作空间

`MultiDiscrete([5, 5, 5])` — 航向/高度/速度各 5 个调整档位：

```
航向：[-20°, -10°, 0°, +10°, +20°]
高度：[-2000ft, -1000ft, 0ft, +1000ft, +2000ft]
速度：[-20kts, -10kts, 0kts, +10kts, +20kts]
```

#### 连续动作空间

`Box(-1, 1, shape=(3,))` — 归一化值通过可配置缩放映射到实际调整量。

| 模块 | 类 | 功能 |
|------|-----|------|
| `translator.py` | `ActionTranslator` | 离散索引 → BlueSky HDG/ALT/SPD 命令 |
| `continuous_translator.py` | `ContinuousActionTranslator` | 连续值 → BlueSky 命令 |

---

### 7. 环境包装器 (`wrappers/`)

| 包装器 | 功能 |
|--------|------|
| `SingleAgentGymWrapper` | 将 ParallelEnv 转为单智能体 gymnasium.Env。ego 飞机由 RL 策略控制，其余使用 noop `[2,2,2]` |
| `NoisyObservationWrapper` | 对 ndarray 观测添加高斯噪声 |
| `WindFieldWrapper` | 注入均匀风场，可选在观测中添加机体坐标系风分量 |

---

### 8. 训练基础设施 (`training/`)

| 模块 | 类 | 功能 |
|------|-----|------|
| `algorithm_factory.py` | `AlgorithmFactory` | 创建 SB3 算法实例（PPO/SAC/TD3/DDPG），支持 YAML 配置 |
| `evaluator.py` | `ModelEvaluator` | 评估模型和基线策略，输出 `EvalResult`（均值/标准差/最小/最大奖励、到达率、NMAC率） |
| `logger.py` | `CSVLoggerCallback` | SB3 回调，记录每回合指标到 CSV |
| `checkpoint.py` | `CheckpointManager` | 模型检查点保存/加载/轮转（.zip + .json 元数据） |

---

### 9. 渲染系统 (`rendering/`)

基于 Pygame 的可视化，支持 3 个场景：

| 渲染器 | 场景 | 特殊元素 |
|--------|------|----------|
| `HorizontalCRRenderer` | HorizontalCR | 飞机、航路点、NMAC 圆 |
| `VerticalCRRenderer` | VerticalCR | 飞机、航路点、NMAC 圆、高度标签 |
| `SectorCrRenderer` | SectorCR | 飞机、航路点、NMAC 圆、扇区多边形 |

`BaseRenderer` 管理 Pygame 初始化、HUD 叠加和资源清理，子类实现 `render_frame()`。

---

### 10. 工具模块 (`utils/`)

#### `geometry.py` — 几何计算

| 函数 | 功能 |
|------|------|
| `haversine_distance(lat1, lon1, lat2, lon2)` | 大圆距离（NM） |
| `bearing(lat1, lon1, lat2, lon2)` | 方位角（度） |
| `point_in_polygon(lat, lon, polygon)` | 射线法判断点是否在多边形内 |
| `segments_intersect(p1, p2, p3, p4)` | 线段相交检测 |
| `generate_polygon(center, num_vertices, area_nm2)` | 生成随机多边形 |
| `assign_sector(lat, lon, sectors)` | 判断点所属扇区 |

#### `types.py` — 类型定义

- `AircraftState`：飞机状态（支持字典式访问）
- `DiscreteAction` / `ContinuousAction`：动作类型
- `ConflictLevel`：冲突等级枚举（NONE/SEPARATION/WARNING/NMAC）
- `Route`：航路（航路点序列）
- 各种配置 dataclass：`SimulationConfig`、`SectorConfig`、`SpawnConfig` 等

---

### 11. 流量管理 (`flow/`)

#### `scheduler.py` — FlowScheduler

管理出发/到达间隔，追踪每架飞机的扇区移交次数。用于容量约束场景。

---

## 配置系统

### 配置文件结构

```
config/
├── default.yaml          # 默认环境参数
├── rewards.yaml          # 奖励组件权重和阈值
├── algorithms.yaml       # 算法超参数
└── scenarios/            # 场景特定配置
    ├── horizontal_cr.yaml
    ├── vertical_cr.yaml
    ├── sector_cr.yaml
    └── plan_waypoint.yaml
```

### 配置优先级

```
场景 YAML 配置 > rewards.yaml > default.yaml
```

场景配置中的 `reward_overrides` 可覆盖全局奖励权重。

### 关键配置项

**default.yaml**：
- `simulation.dt`：仿真步长（5 秒）
- `simulation.max_episode_steps`：每回合最大步数（360）
- `aircraft.initial_count`：初始飞机数（5）
- `aircraft.spawn`：生成范围（高度/速度/航向）
- `observation`：感知半径、垂直范围、最大可观测数
- `action`：离散动作调整档位
- `normalization`：观测归一化参数

**rewards.yaml**：
- 9 个组件的 `enabled`、`weight`、惩罚值和阈值

---

## 训练流程

### 快速验证

```bash
python scripts/train_smoke_test.py
```

### 单场景训练

```bash
python scripts/train_ppo_scenarios.py --scenario HorizontalCR --algorithm PPO --timesteps 100000
```

### 批量训练

```bash
python scripts/train_all_algos.py --timesteps 200000
```

### 评估

```bash
python scripts/evaluate_baselines.py --scenario HorizontalCR
```

---

## 测试体系

### 测试分层

| 层级 | 数量 | 覆盖范围 |
|------|------|----------|
| 单元测试 | ~83 | 各模块独立功能 |
| 集成测试 | ~16 | 场景端到端、组件交互 |
| 性能测试 | ~2 | 步时间、内存使用 |

### 测试运行

```bash
pytest tests/ -v                    # 全部测试
pytest tests/test_bluesky_wrapper.py # 指定模块
pytest tests/integration/           # 集成测试
```

### 测试基础设施

- `conftest.py`：共享 fixture（`default_config`、`rewards_config`、`bluesky_wrapper`）
- `env_factory.py`：环境工厂函数（`make_config()`、`make_env()`）
- 所有测试使用真实 BlueSky 仿真引擎

---

## 依赖关系

```
bluesky-pettingzoo
├── bluesky-simulator[full]  # BlueSky 仿真引擎（pip install）
├── pettingzoo>=1.24.0       # 多智能体框架
├── gymnasium>=0.29.0        # 单智能体接口
├── numpy>=1.24.0            # 数值计算
├── pyyaml>=6.0              # 配置解析
└── stable-baselines3        # RL 算法（训练时）
```
