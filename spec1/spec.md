# bluesky-marl 项目需求规范

> 版本：v1.0 MVP
> 日期：2026-05-20
> 状态：草案

---

## 1. 项目概述

### 1.1 项目名称

`bluesky-marl` — 基于 BlueSky 的多智能体强化学习空管仿真环境

### 1.2 项目目标

构建一个符合 PettingZoo `ParallelEnv` 标准的多智能体强化学习环境，用于空中交通管理（ATM）领域的研究。第一版聚焦于**飞机级冲突解脱与航线重构**，为后续扇区动态管理、LLM 集成奠定基础。

### 1.3 目标用户

空中交通管理方向研究人员（博士/硕士），具备 Python 和强化学习基础知识。

### 1.4 长期愿景

- V1.0：飞机级多智能体冲突解脱（当前版本）
- V2.0：扇区级智能体 + 扇区动态合并/拆分
- V3.0：LLM/RAG 集成，混合决策架构

---

## 2. 系统架构

### 2.1 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    Training Framework                     │
│              (Ray/RLlib, SB3, CleanRL, etc.)             │
└─────────────────────────┬───────────────────────────────┘
                          │ env.step(actions)
                          ▼
┌─────────────────────────────────────────────────────────┐
│              BlueSkyMARLEnv (ParallelEnv)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Observation  │  │   Reward    │  │  Action         │  │
│  │ Manager      │  │ Calculator  │  │  Translator     │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────┬───────────────────────────────┘
                          │ Python API (in-process)
                          ▼
┌─────────────────────────────────────────────────────────┐
│                BlueSky Simulator (Headless)               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Traffic      │  │ Simulation  │  │ Stack           │  │
│  │ Manager      │  │ Clock       │  │ Processor       │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 仿真引擎 | BlueSky (headless mode) | 通过 Python API 直接调用，无网络 I/O |
| MARL 接口 | PettingZoo ParallelEnv | 所有 Agent 同时观测、同时动作 |
| 动作空间 | gymnasium.spaces.MultiDiscrete | 离散化指令，便于 MARL 和 LLM 对齐 |
| 观测空间 | gymnasium.spaces.Dict | 支持异构、变长观测 |
| 配置管理 | YAML + dataclasses | 环境参数与代码解耦 |

---

## 3. 智能体设计

### 3.1 智能体类型（V1.0）

V1.0 仅支持**飞机级 Agent**，每个 Agent 代表一架飞机。

| 属性 | 定义 |
|------|------|
| Agent ID | 字符串格式，如 `AC001`, `AC002` |
| 决策内容 | 航向调整、高度调整、速度调整 |
| 可观测范围 | 以自身为中心，半径 R 海里内的其他飞机 |

### 3.2 智能体类型（V2.0 预留）

V2.0 将引入**扇区级 Agent**，形成分层多智能体架构：

- 高层 Agent（扇区控制器）：决策扇区合并/拆分、容量调整
- 底层 Agent（飞机）：执行具体航线调整和冲突解脱

V1.0 的架构设计必须为此预留扩展点，但不实现。

### 3.3 动态增减

飞机进入/离开仿真区域时，`agents` 列表动态变化：

```python
def step(self, actions):
    # ... 仿真推进 ...
    # 移除已离开空域的 Agent
    self.agents = [a for a in self.agents if self._is_active(a)]
    # 新进入空域的 Agent 在下一步自动加入
```

---

## 4. 仿真配置

### 4.1 空域场景

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 扇区数量 | 2 | 两个相邻扇区 |
| 初始飞机数量 | 5 | 随机生成，存在潜在交汇冲突 |
| 空域范围 | 可配置 | 定义扇区边界坐标 |

### 4.2 仿真时序

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 仿真步长 Δt | 5 秒 | `env.step()` 推进的仿真时间 |
| 最大仿真步数 | 可配置 | 单个 episode 最大步数 |
| 冲突检测间隔 | 每步 | 每个 step 都检测冲突 |

### 4.3 飞机生成规则

- 在空域边界随机生成
- 初始航向指向空域内部或对侧边界
- 初始高度在标准飞行高度层（FL）中随机选择
- 速度在典型巡航速度范围内随机

---

## 5. 观测空间

### 5.1 设计原则

- **部分可观测**：每个 Agent 只能观测到感知范围内的信息
- **异构支持**：使用 `Dict` 空间，支持不同类型的信息字段
- **双轨输出**：数值轨道用于 MARL，文本轨道用于未来 LLM

### 5.2 单个 Agent 的观测结构

```python
observation_space = spaces.Dict({
    # 自身状态（固定维度）
    "self_state": spaces.Box(
        low=-1.0, high=1.0, shape=(6,),
        # [归一化航向, 归一化高度, 归一化速度, 归一化纬度, 归一化经度, 归一化垂直速率]
    ),
    # 感知范围内的其他飞机（变长，padding 到最大数量）
    "other_aircraft": spaces.Box(
        low=-1.0, high=1.0, shape=(MAX_OBSERVABLE_AIRCRAFT, 7),
        # 每架: [相对方位, 相对距离, 相对高度, 相对速度, 航向, 高度, 速度]
    ),
    # 其他飞机掩码（标识哪些位置是真实数据 vs padding）
    "other_aircraft_mask": spaces.Box(
        low=0, high=1, shape=(MAX_OBSERVABLE_AIRCRAFT,), dtype=np.int8
    ),
    # 目标信息
    "goal": spaces.Box(
        low=-1.0, high=1.0, shape=(4,),
        # [目标航路点纬度, 目标航路点经度, 目标高度, 剩余距离]
    ),
})
```

### 5.3 感知范围

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 水平感知半径 | 20 海里 | 超出此范围的飞机不可观测 |
| 垂直感知范围 | ±3000 英尺 | 高度差超出此范围的飞机不可观测 |
| 最大可观测飞机数 | 10 | 用于 padding 维度 |

### 5.4 归一化方法

所有数值归一化到 [-1, 1] 范围：

| 字段 | 归一化方法 |
|------|-----------|
| 航向 | `(heading - 180) / 180` |
| 高度 | `(alt - ALT_MID) / ALT_RANGE` |
| 速度 | `(tas - TAS_MID) / TAS_RANGE` |
| 距离 | `distance / MAX_DISTANCE` |
| 方位 | `bearing / 360` |

---

## 6. 动作空间

### 6.1 设计原则

- **离散化**：符合真实 ATC 指令模式，便于 MARL 收敛和 LLM 对齐
- **组合动作**：每个 step 同时决策航向、高度、速度

### 6.2 动作定义

```python
action_space = spaces.MultiDiscrete([
    len(HDG_ADJUSTMENTS),  # 航向调整选项数
    len(ALT_ADJUSTMENTS),  # 高度调整选项数
    len(SPD_ADJUSTMENTS),  # 速度调整选项数
])
```

### 6.3 离散化选项

| 动作维度 | 选项 | 说明 |
|---------|------|------|
| 航向调整 | [-20°, -10°, 0°, +10°, +20°] | 5 个离散选项 |
| 高度调整 | [-2000ft, -1000ft, 0, +1000ft, +2000ft] | 5 个离散选项 |
| 速度调整 | [-20kt, -10kt, 0, +10kt, +20kt] | 5 个离散选项 |

动作空间大小：5 × 5 × 5 = 125 种组合

### 6.4 动作执行

动作在每个 step 开始时批量发送给 BlueSky：

```python
def _send_actions_to_bluesky(self, actions):
    commands = []
    for agent_id, action in actions.items():
        hdg_adj = HDG_ADJUSTMENTS[action[0]]
        alt_adj = ALT_ADJUSTMENTS[action[1]]
        spd_adj = SPD_ADJUSTMENTS[action[2]]

        if hdg_adj != 0:
            new_hdg = (self._get_heading(agent_id) + hdg_adj) % 360
            commands.append(f"HDG {agent_id} {new_hdg}")
        if alt_adj != 0:
            new_alt = self._get_altitude(agent_id) + alt_adj
            commands.append(f"ALT {agent_id} {new_alt}")
        if spd_adj != 0:
            new_spd = self._get_tas(agent_id) + spd_adj
            commands.append(f"SPD {agent_id} {new_spd}")

    # 批量执行
    for cmd in commands:
        bs.stack.stack(cmd)
```

---

## 7. 奖励函数

### 7.1 设计原则

- **模块化**：独立 `RewardCalculator` 类，支持动态注册奖励组件
- **可配置**：权重通过 YAML 配置文件管理
- **优先级**：安全 > 平稳性 > 效率

### 7.2 奖励组件（V1.0）

#### 7.2.1 冲突惩罚（安全）

| 事件 | 惩罚值 | 说明 |
|------|--------|------|
| NMAC（近距空中相撞） | -100 | 水平 < 5NM 且垂直 < 1000ft |
| 冲突预警 | -10 | 水平 < 10NM 且垂直 < 2000ft |
| 安全间隔违反 | -5 | 水平 < 5NM 或垂直 < 1000ft |

NMAC 事件触发时，该 Agent 的 episode 终止（`termination = True`）。

#### 7.2.2 平稳性惩罚

| 事件 | 惩罚值 | 说明 |
|------|--------|------|
| 发布任何指令 | -0.1 | 鼓励保持当前状态 |

#### 7.2.3 效率奖励/惩罚

| 指标 | 计算方法 | 说明 |
|------|---------|------|
| 航线偏离 | `-偏离距离 / MAX_DEVIATION * 5` | 偏离目标航线越远，惩罚越大 |
| 到达奖励 | `+10` | 到达目标航路点 |
| 步数惩罚 | `-0.01` | 每步微小惩罚，鼓励尽快完成 |

### 7.3 奖励组件注册机制

```python
class RewardCalculator:
    def __init__(self):
        self.components = []

    def register(self, component: RewardComponent, weight: float):
        self.components.append((component, weight))

    def compute(self, agent_id, state, action, next_state) -> float:
        total = 0.0
        for component, weight in self.components:
            total += weight * component.compute(agent_id, state, action, next_state)
        return total

# 配置示例
reward_calculator = RewardCalculator()
reward_calculator.register(ConflictPenalty(), weight=1.0)
reward_calculator.register(SmoothnessPenalty(), weight=0.5)
reward_calculator.register(EfficiencyReward(), weight=0.3)
```

### 7.4 奖励配置文件

```yaml
# config/rewards.yaml
rewards:
  conflict:
    nmac_penalty: -100
    warning_penalty: -10
    separation_penalty: -5
    nmac_horizontal_nm: 5
    nmac_vertical_ft: 1000
    warning_horizontal_nm: 10
    warning_vertical_ft: 2000

  smoothness:
    action_penalty: -0.1

  efficiency:
    max_deviation_nm: 50
    deviation_penalty_scale: 5
    arrival_reward: 10
    step_penalty: -0.01
```

---

## 8. 终止与截断条件

### 8.1 终止条件（Termination）

满足以下任一条件时，该 Agent 的 episode 终止：

| 条件 | 说明 |
|------|------|
| NMAC | 发生近距空中相撞 |
| 到达目标 | 成功到达目标航路点 |

### 8.2 截断条件（Truncation）

满足以下任一条件时，整个 episode 截断：

| 条件 | 说明 |
|------|------|
| 最大步数 | 达到 `max_episode_steps` |
| 超出空域 | 飞机飞出仿真区域边界 |

---

## 9. 双轨输出接口

### 9.1 设计目标

在 `infos` 字典中预留结构化/文本化状态，为未来 LLM/RAG 集成做准备。V1.0 仅生成数据，不接入真实 LLM。

### 9.2 infos 结构

```python
infos = {
    # 标准 PettingZoo infos
    agent_id: {
        # 该 Agent 的额外信息
    },
    # 双轨输出：文本/结构化状态
    "__common__": {
        "textual_state": {
            "agent_id": "AC001",
            "position": {"lat": 39.123, "lon": 116.456},
            "heading": 90,
            "altitude": 35000,
            "speed": 450,
            "observable_aircraft": [
                {"id": "AC002", "distance_nm": 15.3, "relative_alt_ft": -1000}
            ],
            "conflict_status": "warning",
            "text": "Flight AC001 is flying at heading 090, altitude FL350, speed 450kt. 1 aircraft observable. Conflict warning with AC002 at 15.3 NM."
        },
        "airspace_snapshot": {
            "sectors": [...],
            "waypoints": [...],
            "weather": [...]
        }
    }
}
```

### 9.3 文本生成模板

```python
def _generate_text_state(self, agent_id):
    state = self._get_agent_state(agent_id)
    observables = self._get_observable_aircraft(agent_id)

    text = f"Flight {agent_id} is flying at heading {state['heading']:.0f}, "
    text += f"altitude {state['altitude']:.0f}ft, speed {state['speed']:.0f}kt. "
    text += f"{len(observables)} aircraft observable."

    if self._has_conflict_warning(agent_id):
        text += f" Conflict warning with nearest aircraft."
    elif self._has_nmac(agent_id):
        text += f" NMAC detected!"

    return text
```

---

## 10. 基线 Agent

### 10.1 RandomAgent

随机从动作空间中采样，用于验证环境接口正确性。

```python
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return {agent: self.action_space.sample() for agent in observation}
```

### 10.2 RuleBasedAgent（直飞 Agent）

不进行任何避让动作，保持当前航向直飞，用于：

- 测试冲突检测逻辑
- 作为性能基线

```python
class RuleBasedAgent:
    def act(self, observation):
        # 选择"无调整"动作（索引 2 对应 0 调整）
        return {agent: [2, 2, 2] for agent in observation}
```

---

## 11. 测试要求

### 11.1 PettingZoo API 测试

必须通过 PettingZoo 官方提供的 API 测试套件：

```python
from pettingzoo.test import parallel_api_test

def test_bluesky_env_api():
    env = BlueSkyMARLEnv(config="test_config.yaml")
    parallel_api_test(env, num_cycles=100)
```

### 11.2 单元测试覆盖

| 模块 | 最低覆盖率 | 测试重点 |
|------|-----------|---------|
| 环境核心 (env.py) | 90% | reset/step 循环、Agent 增减 |
| 观测管理 (observation.py) | 90% | 归一化、感知范围过滤 |
| 奖励计算 (reward.py) | 95% | 各组件独立测试 |
| 动作转换 (action.py) | 90% | 离散到连续映射 |
| BlueSky 接口 (bluesky_wrapper.py) | 85% | 命令生成、状态读取 |

### 11.3 集成测试

| 测试场景 | 验证内容 |
|---------|---------|
| 无冲突场景 | 所有飞机直飞到达目标 |
| 单冲突场景 | 两架飞机冲突，验证避让 |
| 多冲突场景 | 5 架飞机交叉冲突 |
| 边界条件 | 飞机进入/离开空域 |

---

## 12. 工程规范

### 12.1 项目结构

```
bluesky-PettingZoo/
├── src/
│   └── bluesky_pettingzoo/
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── bluesky_marl_env.py      # 主环境类
│       │   └── scenarios/
│       │       └── two_sector_crossing.py
│       ├── observations/
│       │   ├── __init__.py
│       │   └── observation_manager.py
│       ├── rewards/
│       │   ├── __init__.py
│       │   ├── calculator.py
│       │   └── components.py
│       ├── actions/
│       │   ├── __init__.py
│       │   └── action_translator.py
│       ├── bluesky/
│       │   ├── __init__.py
│       │   └── wrapper.py               # BlueSky API 封装
│       └── agents/
│           ├── __init__.py
│           ├── random_agent.py
│           └── rule_based_agent.py
├── tests/
│   ├── test_env.py
│   ├── test_observations.py
│   ├── test_rewards.py
│   └── test_actions.py
├── config/
│   ├── default.yaml
│   └── rewards.yaml
├── docs/
│   └── spec.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

### 12.2 依赖管理

```
# requirements.txt
bluesky-sim>=1.0.0
pettingzoo>=1.24.0
gymnasium>=0.29.0
numpy>=1.24.0
pyyaml>=6.0
```

### 12.3 代码质量

- 类型注解：所有公开函数必须有完整类型注解
- Linting：使用 ruff
- 格式化：使用 ruff format
- 类型检查：使用 mypy
- 提交规范：Conventional Commits

---

## 13. V1.0 验收标准

### 13.1 功能验收

- [ ] 环境可通过 PettingZoo `parallel_api_test`
- [ ] `reset()` 返回正确格式的观测和 infos
- [ ] `step()` 返回 5 元组 (obs, rewards, terminations, truncations, infos)
- [ ] 飞机进入/离开空域时 `agents` 列表正确更新
- [ ] 冲突检测和 NMAC 检测正常工作
- [ ] 奖励函数各组件正确计算
- [ ] `infos` 中包含 `textual_state` 和 `airspace_snapshot`
- [ ] RandomAgent 和 RuleBasedAgent 可正常运行

### 13.2 性能验收

- [ ] 单个 `step()` 执行时间 < 100ms（5 架飞机场景）
- [ ] `reset()` 执行时间 < 500ms
- [ ] 内存占用 < 1GB（单环境）

### 13.3 测试验收

- [ ] 单元测试覆盖率 ≥ 90%
- [ ] 所有测试用例通过
- [ ] 集成测试覆盖所有验收场景

---

## 14. 未来扩展预留

V1.0 不实现，但架构必须支持：

| 扩展方向 | 预留点 |
|---------|--------|
| 扇区 Agent | Agent ID 命名空间、观测空间扩展 |
| GNN 集成 | `airspace_snapshot` 提供图结构数据 |
| LLM 集成 | `textual_state` 提供自然语言描述 |
| 连续动作空间 | ActionTranslator 可扩展 |
| 多场景 | Scenario 基类 + 配置文件驱动 |
| 分布式训练 | 环境无状态设计，支持 `ray.EnvRunner` |

---

## 附录 A：术语表

| 术语 | 定义 |
|------|------|
| NMAC | Near Mid-Air Collision，近距空中相撞 |
| ATC | Air Traffic Control，空中交通管制 |
| ATC | Air Traffic Management，空中交通管理 |
| FL | Flight Level，飞行高度层（1FL ≈ 100ft） |
| NM | Nautical Mile，海里 |
| TAS | True Airspeed，真空速 |
| LNAV | Lateral Navigation，水平导航 |
| VNAV | Vertical Navigation，垂直导航 |
| POMDP | Partially Observable Markov Decision Process，部分可观测马尔可夫决策过程 |
| MARL | Multi-Agent Reinforcement Learning，多智能体强化学习 |

---

## 附录 B：参考实现

| 项目 | 说明 | 参考点 |
|------|------|--------|
| bluesky-gym | BlueSky 单智能体 Gym 环境 | BlueSky API 调用方式 |
| PettingZoo | 多智能体环境标准 | ParallelEnv 接口规范 |
| pettingzoo-test | API 测试套件 | 环境合规性验证 |
