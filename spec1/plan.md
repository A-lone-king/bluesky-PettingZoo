# bluesky-marl 技术实现方案

> 版本：v1.0
> 日期：2026-05-20
> 基于：spec.md v1.0 MVP

---

## 1. 技术选型

### 1.1 核心依赖

| 组件 | 选型 | 版本 | 选型理由 |
|------|------|------|---------|
| Python | CPython | 3.11.x | 性能提升约 10-60%，错误信息更精确 |
| 仿真引擎 | BlueSky | latest | 开源 ATM 仿真器，Python API 支持良好 |
| MARL 框架 | PettingZoo | ≥1.24.0 | 多智能体环境标准，生态成熟 |
| Gym 接口 | Gymnasium | ≥0.29.0 | PettingZoo 底层依赖，维护活跃 |
| 数值计算 | NumPy | ≥1.24.0 | 向量化计算，性能优异 |
| 配置管理 | PyYAML | ≥6.0 | 配置文件解析 |
| 类型检查 | mypy | ≥1.0 | 静态类型检查 |
| 代码规范 | ruff | ≥0.1.0 | 快速 linter + formatter |
| 测试框架 | pytest | ≥7.0 | 灵活的测试框架 |
| 覆盖率 | pytest-cov | ≥4.0 | 测试覆盖率统计 |

### 1.2 可选依赖（V2.0 预留）

| 组件 | 选型 | 用途 |
|------|------|------|
| Ray/RLlib | ≥2.0 | 分布式 MARL 训练 |
| Stable-Baselines3 | ≥2.0 | 单智能体 baseline |
| CleanRL | - | 参考实现 |
| torch | ≥2.0 | 深度学习框架 |

### 1.3 虚拟环境管理（Windows）

```powershell
# 创建虚拟环境（已存在）
python -m venv venv

# 激活虚拟环境（PowerShell）
.\venv\Scripts\Activate.ps1

# 如果遇到执行策略错误，先运行：
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 升级 pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
pip install -e .  # 开发模式安装
```

---

## 2. 目录结构

### 2.1 完整目录树

```
bluesky-PettingZoo/
├── venv/                          # Python 虚拟环境（已存在）
├── bluesky/                       # BlueSky 官方仓库（submodule/reference）
├── bluesky-gym/                   # BlueSky-Gym 参考实现
├── PettingZoo/                    # PettingZoo 源码（参考）
│
├── src/
│   └── bluesky_pettingzoo/        # 主包
│       ├── __init__.py            # 包初始化，版本信息
│       ├── py.typed               # PEP 561 类型标记
│       │
│       ├── envs/                  # 环境实现
│       │   ├── __init__.py
│       │   ├── base.py            # 基类 BlueSkyBaseEnv
│       │   ├── parallel_env.py    # 主环境 ParallelEnv 实现
│       │   └── scenarios/         # 场景定义
│       │       ├── __init__.py
│       │       ├── base_scenario.py
│       │       └── two_sector_crossing.py
│       │
│       ├── spaces/                # 自定义空间
│       │   ├── __init__.py
│       │   └── variable_length.py # 变长观测空间
│       │
│       ├── observations/          # 观测处理
│       │   ├── __init__.py
│       │   ├── manager.py         # ObservationManager
│       │   ├── normalizer.py      # 归一化处理
│       │   └── filters.py         # 感知范围过滤
│       │
│       ├── actions/               # 动作处理
│       │   ├── __init__.py
│       │   ├── translator.py      # 离散→连续动作转换
│       │   └── definitions.py     # 动作空间定义
│       │
│       ├── rewards/               # 奖励系统
│       │   ├── __init__.py
│       │   ├── calculator.py      # RewardCalculator
│       │   ├── base.py            # RewardComponent 基类
│       │   └── components/        # 奖励组件
│       │       ├── __init__.py
│       │       ├── conflict.py    # 冲突惩罚
│       │       ├── smoothness.py  # 平稳性惩罚
│       │       └── efficiency.py  # 效率奖励
│       │
│       ├── bluesky/               # BlueSky 接口封装
│       │   ├── __init__.py
│       │   ├── wrapper.py         # BlueSky API 封装
│       │   ├── traffic.py         # 飞机状态管理
│       │   └── commands.py        # 命令构建器
│       │
│       ├── agents/                # 基线 Agent
│       │   ├── __init__.py
│       │   ├── base.py            # BaseAgent
│       │   ├── random_agent.py    # RandomAgent
│       │   └── rule_based_agent.py # RuleBasedAgent
│       │
│       └── utils/                 # 工具函数
│           ├── __init__.py
│           ├── geometry.py        # 几何计算
│           ├── types.py           # 类型定义
│           └── logging.py         # 日志配置
│
├── tests/                         # 测试套件
│   ├── conftest.py                # pytest fixtures
│   ├── test_env.py                # 环境核心测试
│   ├── test_api_compliance.py     # PettingZoo API 合规测试
│   ├── test_observations.py       # 观测模块测试
│   ├── test_actions.py            # 动作模块测试
│   ├── test_rewards.py            # 奖励模块测试
│   ├── test_bluesky_wrapper.py    # BlueSky 接口测试
│   └── integration/               # 集成测试
│       ├── test_no_conflict.py
│       ├── test_single_conflict.py
│       └── test_multi_conflict.py
│
├── config/                        # 配置文件
│   ├── default.yaml               # 默认配置
│   ├── rewards.yaml               # 奖励配置
│   ├── scenarios/                 # 场景配置
│   │   └── two_sector.yaml
│   └── test.yaml                  # 测试配置
│
├── docs/                          # 文档
│   ├── spec.md                    # 需求规范
│   ├── plan.md                    # 技术方案（本文件）
│   └── api/                       # API 文档
│
├── scripts/                       # 脚本工具
│   ├── setup_dev.bat              # 开发环境初始化（Windows）
│   └── run_tests.bat              # 测试运行脚本（Windows）
│
├── pyproject.toml                 # 项目配置
├── requirements.txt               # 生产依赖
├── requirements-dev.txt           # 开发依赖
├── .gitignore
├── .editorconfig
└── README.md
```

### 2.2 模块依赖关系

```
                    ┌─────────────────┐
                    │   parallel_env  │
                    │   (主入口)       │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ observations  │    │    rewards    │    │    actions    │
│   manager     │    │  calculator   │    │  translator   │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        │                    ▼                    │
        │            ┌───────────────┐            │
        │            │  components/  │            │
        │            │  conflict     │            │
        │            │  smoothness   │            │
        │            │  efficiency   │            │
        │            └───────────────┘            │
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     bluesky     │
                    │    wrapper      │
                    └─────────────────┘
```

---

## 3. 数据模型

### 3.1 核心类型定义

```python
# src/bluesky_pettingzoo/utils/types.py

from typing import TypedDict, NamedTuple
from enum import IntEnum
import numpy as np
from numpy.typing import NDArray

# Agent 标识符
AgentID = str

# 飞机状态
class AircraftState(TypedDict):
    """BlueSky 飞机状态"""
    id: str
    lat: float
    lon: float
    alt: float          # 英尺
    hdg: float          # 度
    tas: float          # 节
    vs: float           # 英尺/分钟

# 归一化观测
class NormalizedObservation(TypedDict):
    """归一化后的观测数据"""
    self_state: NDArray[np.float32]      # shape=(6,)
    other_aircraft: NDArray[np.float32]  # shape=(MAX_OBS, 7)
    other_aircraft_mask: NDArray[np.int8] # shape=(MAX_OBS,)
    goal: NDArray[np.float32]            # shape=(4,)

# 文本状态（双轨输出）
class TextualState(TypedDict):
    """文本化状态，用于 LLM/RAG"""
    agent_id: str
    position: dict[str, float]
    heading: float
    altitude: float
    speed: float
    observable_aircraft: list[dict]
    conflict_status: str  # "safe" | "warning" | "nmac"
    text: str

# 空域快照
class AirspaceSnapshot(TypedDict):
    """空域拓扑快照"""
    sectors: list[dict]
    waypoints: list[dict]
    aircraft_positions: dict[str, dict]

# 动作类型
class DiscreteAction(NamedTuple):
    """离散动作"""
    heading_idx: int    # 0-4
    altitude_idx: int   # 0-4
    speed_idx: int      # 0-4

# 连续动作（BlueSky 命令）
class ContinuousAction(NamedTuple):
    """连续动作（实际发送给 BlueSky）"""
    heading: float      # 0-360
    altitude: float     # 英尺
    speed: float        # 节

# 冲突状态
class ConflictLevel(IntEnum):
    """冲突等级"""
    SAFE = 0
    WARNING = 1
    NMAC = 2

# 场景配置
class ScenarioConfig(TypedDict):
    """场景配置"""
    name: str
    num_sectors: int
    initial_aircraft: int
    airspace_bounds: dict[str, float]
    dt: float
    max_steps: int
```

### 3.2 配置数据模型

```python
# config/default.yaml 结构

simulation:
  dt: 5.0                    # 仿真步长（秒）
  max_episode_steps: 360     # 最大步数（30分钟）
  headless: true

airspace:
  name: "two_sector_crossing"
  sectors:
    - id: "sector_a"
      bounds: [[39.0, 116.0], [39.5, 116.5]]
    - id: "sector_b"
      bounds: [[39.0, 116.5], [39.5, 117.0]]

aircraft:
  initial_count: 5
  spawn:
    altitude_range: [29000, 37000]  # FL290-FL370
    speed_range: [400, 500]         # 节
    heading_range: [0, 360]

observation:
  perception_radius_nm: 20
  perception_alt_diff_ft: 3000
  max_observable_aircraft: 10

action:
  heading_adjustments: [-20, -10, 0, 10, 20]
  altitude_adjustments: [-2000, -1000, 0, 1000, 2000]
  speed_adjustments: [-20, -10, 0, 10, 20]

normalization:
  heading:
    mid: 180
    range: 180
  altitude:
    mid: 33000
    range: 10000
  speed:
    mid: 450
    range: 100
  distance:
    max: 20  # 海里
```

### 3.3 奖励配置数据模型

```python
# config/rewards.yaml 结构

components:
  conflict:
    enabled: true
    weight: 1.0
    nmac_penalty: -100
    warning_penalty: -10
    separation_penalty: -5
    thresholds:
      nmac_horizontal_nm: 5
      nmac_vertical_ft: 1000
      warning_horizontal_nm: 10
      warning_vertical_ft: 2000

  smoothness:
    enabled: true
    weight: 0.5
    action_penalty: -0.1

  efficiency:
    enabled: true
    weight: 0.3
    max_deviation_nm: 50
    deviation_penalty_scale: 5
    arrival_reward: 10
    step_penalty: -0.01
```

---

## 4. 接口设计

### 4.1 主环境接口

```python
# src/bluesky_pettingzoo/envs/parallel_env.py

from pettingzoo import ParallelEnv
from gymnasium import spaces
from bluesky_pettingzoo.utils.types import AgentID

class BlueSkyMARLEnv(ParallelEnv[AgentID, spaces.Dict, dict]):
    """
    基于 BlueSky 的多智能体强化学习环境

    符合 PettingZoo ParallelEnv 标准，支持：
    - 飞机级 Agent 动态增减
    - 部分可观测（POMDP）
    - 离散动作空间
    - 双轨输出（数值 + 文本）
    """

    metadata = {
        "name": "bluesky_marl_v0",
        "render_modes": ["human", "ansi"],
    }

    def __init__(
        self,
        config_path: str = "config/default.yaml",
        render_mode: str | None = None,
    ) -> None:
        """
        初始化环境

        Args:
            config_path: 配置文件路径
            render_mode: 渲染模式
        """
        ...

    @property
    def observation_space(self) -> spaces.Dict:
        """观测空间定义"""
        ...

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        """动作空间定义"""
        ...

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, dict], dict[AgentID, dict]]:
        """
        重置环境

        Args:
            seed: 随机种子
            options: 额外选项

        Returns:
            observations: 各 Agent 的观测
            infos: 各 Agent 的信息
        """
        ...

    def step(
        self,
        actions: dict[AgentID, dict],
    ) -> tuple[
        dict[AgentID, dict],  # observations
        dict[AgentID, float], # rewards
        dict[AgentID, bool],  # terminations
        dict[AgentID, bool],  # truncations
        dict[AgentID, dict],  # infos
    ]:
        """
        执行一步仿真

        Args:
            actions: 各 Agent 的动作 {agent_id: [hdg_idx, alt_idx, spd_idx]}

        Returns:
            observations: 各 Agent 的观测
            rewards: 各 Agent 的奖励
            terminations: 各 Agent 是否终止
            truncations: 各 Agent 是否截断
            infos: 各 Agent 的额外信息
        """
        ...

    def render(self) -> None:
        """渲染环境（V1.0 仅支持文本模式）"""
        ...

    def close(self) -> None:
        """关闭环境，释放资源"""
        ...
```

### 4.2 BlueSky 封装接口

```python
# src/bluesky_pettingzoo/bluesky/wrapper.py

class BlueSkyWrapper:
    """
    BlueSky 仿真器封装

    提供无头模式下的 BlueSky API 访问
    """

    def __init__(self, config: dict) -> None:
        """初始化 BlueSky 仿真器"""
        ...

    def init_simulation(self) -> None:
        """初始化仿真环境"""
        ...

    def step(self, dt: float) -> None:
        """
        推进仿真一步

        Args:
            dt: 仿真时间步长（秒）
        """
        ...

    def create_aircraft(
        self,
        acid: str,
        actype: str,
        lat: float,
        lon: float,
        alt: float,
        hdg: float,
        spd: float,
    ) -> None:
        """创建飞机"""
        ...

    def remove_aircraft(self, acid: str) -> None:
        """移除飞机"""
        ...

    def send_command(self, command: str) -> None:
        """
        发送命令

        Args:
            command: BlueSky 命令，如 "HDG AC001 90"
        """
        ...

    def send_commands_batch(self, commands: list[str]) -> None:
        """
        批量发送命令

        Args:
            commands: 命令列表
        """
        ...

    def get_aircraft_state(self, acid: str) -> AircraftState:
        """获取飞机状态"""
        ...

    def get_all_aircraft_states(self) -> dict[str, AircraftState]:
        """获取所有飞机状态"""
        ...

    def get_active_aircraft_ids(self) -> list[str]:
        """获取活跃飞机 ID 列表"""
        ...

    def is_aircraft_in_airspace(self, acid: str) -> bool:
        """检查飞机是否在空域内"""
        ...

    def reset(self) -> None:
        """重置仿真器"""
        ...

    def close(self) -> None:
        """关闭仿真器"""
        ...
```

### 4.3 观测管理接口

```python
# src/bluesky_pettingzoo/observations/manager.py

class ObservationManager:
    """
    观测管理器

    负责：
    - 从 BlueSky 读取原始状态
    - 感知范围过滤
    - 归一化处理
    - 生成双轨输出
    """

    def __init__(self, config: dict) -> None:
        """初始化观测管理器"""
        ...

    def get_observation(
        self,
        agent_id: str,
        all_states: dict[str, AircraftState],
    ) -> NormalizedObservation:
        """
        获取单个 Agent 的归一化观测

        Args:
            agent_id: Agent ID
            all_states: 所有飞机状态

        Returns:
            归一化观测数据
        """
        ...

    def get_textual_state(
        self,
        agent_id: str,
        all_states: dict[str, AircraftState],
    ) -> TextualState:
        """
        获取文本化状态（双轨输出）

        Args:
            agent_id: Agent ID
            all_states: 所有飞机状态

        Returns:
            文本化状态
        """
        ...

    def get_airspace_snapshot(
        self,
        all_states: dict[str, AircraftState],
    ) -> AirspaceSnapshot:
        """
        获取空域快照

        Args:
            all_states: 所有飞机状态

        Returns:
            空域拓扑快照
        """
        ...

    def normalize_value(self, value: float, mid: float, range_val: float) -> float:
        """归一化单个值到 [-1, 1]"""
        ...

    def filter_by_perception(
        self,
        agent_id: str,
        all_states: dict[str, AircraftState],
    ) -> list[AircraftState]:
        """
        根据感知范围过滤飞机

        Args:
            agent_id: 观测者 Agent ID
            all_states: 所有飞机状态

        Returns:
            感知范围内的飞机状态列表
        """
        ...
```

### 4.4 奖励计算接口

```python
# src/bluesky_pettingzoo/rewards/calculator.py

class RewardCalculator:
    """
    奖励计算器

    支持动态注册奖励组件，组合计算总奖励
    """

    def __init__(self, config: dict) -> None:
        """初始化奖励计算器"""
        ...

    def register_component(
        self,
        component: RewardComponent,
        weight: float,
    ) -> None:
        """
        注册奖励组件

        Args:
            component: 奖励组件实例
            weight: 权重
        """
        ...

    def compute_reward(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
    ) -> float:
        """
        计算总奖励

        Args:
            agent_id: Agent ID
            prev_state: 前一状态
            action: 执行的动作
            curr_state: 当前状态
            all_states: 所有飞机状态

        Returns:
            总奖励值
        """
        ...
```

```python
# src/bluesky_pettingzoo/rewards/base.py

from abc import ABC, abstractmethod

class RewardComponent(ABC):
    """奖励组件基类"""

    @abstractmethod
    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
    ) -> float:
        """
        计算该组件的奖励

        Returns:
            奖励值（未加权）
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """重置组件状态"""
        ...
```

### 4.5 动作翻译接口

```python
# src/bluesky_pettingzoo/actions/translator.py

class ActionTranslator:
    """
    动作翻译器

    将离散动作转换为 BlueSky 连续命令
    """

    def __init__(self, config: dict) -> None:
        """
        初始化动作翻译器

        Args:
            config: 包含离散化选项的配置
        """
        ...

    def translate(
        self,
        agent_id: str,
        discrete_action: DiscreteAction,
        current_state: AircraftState,
    ) -> list[str]:
        """
        翻译离散动作为 BlueSky 命令

        Args:
            agent_id: Agent ID
            discrete_action: 离散动作 [hdg_idx, alt_idx, spd_idx]
            current_state: 当前飞机状态

        Returns:
            BlueSky 命令列表，如 ["HDG AC001 90", "ALT AC001 35000"]
        """
        ...

    def translate_batch(
        self,
        actions: dict[str, DiscreteAction],
        states: dict[str, AircraftState],
    ) -> list[str]:
        """
        批量翻译动作为命令列表

        Args:
            actions: {agent_id: discrete_action}
            states: {agent_id: aircraft_state}

        Returns:
            所有命令的列表
        """
        ...
```

### 4.6 基线 Agent 接口

```python
# src/bluesky_pettingzoo/agents/base.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(self, action_space: spaces.MultiDiscrete) -> None:
        """初始化 Agent"""
        ...

    @abstractmethod
    def act(
        self,
        observations: dict[AgentID, dict],
    ) -> dict[AgentID, list[int]]:
        """
        根据观测选择动作

        Args:
            observations: 各 Agent 的观测

        Returns:
            各 Agent 的动作
        """
        ...

    def reset(self) -> None:
        """重置 Agent 状态"""
        ...
```

---

## 5. 实现阶段

### 5.1 阶段划分

```
Phase 1: 基础框架 (Week 1-2)
    ├── 项目初始化
    ├── BlueSky 封装
    └── 基础环境骨架

Phase 2: 核心功能 (Week 3-4)
    ├── 观测系统
    ├── 动作系统
    └── 奖励系统

Phase 3: 集成测试 (Week 5)
    ├── PettingZoo API 测试
    ├── 基线 Agent
    └── 集成测试

Phase 4: 文档完善 (Week 6)
    ├── API 文档
    ├── 使用示例
    └── README
```

### 5.2 Phase 1: 基础框架（Week 1-2）

**目标**：搭建项目骨架，实现 BlueSky 无头模式封装

**任务清单**：

| 任务 | 优先级 | 预估工时 | 依赖 |
|------|--------|---------|------|
| 1.1 项目初始化（pyproject.toml, requirements） | P0 | 2h | - |
| 1.2 虚拟环境配置 | P0 | 1h | 1.1 |
| 1.3 代码规范配置（ruff, mypy） | P1 | 1h | 1.1 |
| 1.4 BlueSky 无头模式初始化 | P0 | 4h | - |
| 1.5 BlueSkyWrapper 基础实现 | P0 | 8h | 1.4 |
| 1.6 配置系统实现 | P1 | 4h | - |
| 1.7 基础环境骨架（reset/step 循环） | P0 | 8h | 1.5, 1.6 |

**验收标准**：
- [ ] 项目可安装（`pip install -e .`）
- [ ] BlueSky 可在无头模式初始化
- [ ] 可创建/删除飞机
- [ ] 可发送基本命令（HDG, ALT, SPD）
- [ ] 空的 `reset()` 和 `step()` 可运行

### 5.3 Phase 2: 核心功能（Week 3-4）

**目标**：实现观测、动作、奖励三大系统

**任务清单**：

| 任务 | 优先级 | 预估工时 | 依赖 |
|------|--------|---------|------|
| 2.1 观测空间定义 | P0 | 4h | - |
| 2.2 感知范围过滤 | P0 | 4h | 2.1 |
| 2.3 观测归一化 | P0 | 4h | 2.1 |
| 2.4 ObservationManager 完整实现 | P0 | 8h | 2.1-2.3 |
| 2.5 动作空间定义 | P0 | 2h | - |
| 2.6 ActionTranslator 实现 | P0 | 6h | 2.5 |
| 2.7 RewardComponent 基类 | P0 | 2h | - |
| 2.8 ConflictPenalty 实现 | P0 | 4h | 2.7 |
| 2.9 SmoothnessPenalty 实现 | P1 | 2h | 2.7 |
| 2.10 EfficiencyReward 实现 | P1 | 4h | 2.7 |
| 2.11 RewardCalculator 完整实现 | P0 | 4h | 2.7-2.10 |
| 2.12 双轨输出实现 | P1 | 4h | 2.4 |
| 2.13 终止/截断条件 | P0 | 4h | - |

**验收标准**：
- [ ] 观测空间符合 spec 定义
- [ ] 感知范围过滤正确工作
- [ ] 归一化值在 [-1, 1] 范围
- [ ] 动作正确转换为 BlueSky 命令
- [ ] 奖励组件独立计算正确
- [ ] infos 包含 textual_state

### 5.4 Phase 3: 集成测试（Week 5）

**目标**：通过 PettingZoo API 测试，实现基线 Agent

**任务清单**：

| 任务 | 优先级 | 预估工时 | 依赖 |
|------|--------|---------|------|
| 3.1 RandomAgent 实现 | P0 | 2h | - |
| 3.2 RuleBasedAgent 实现 | P0 | 2h | - |
| 3.3 PettingZoo API 合规测试 | P0 | 4h | Phase 2 |
| 3.4 无冲突场景集成测试 | P0 | 4h | Phase 2 |
| 3.5 单冲突场景集成测试 | P0 | 4h | 3.4 |
| 3.6 多冲突场景集成测试 | P1 | 4h | 3.5 |
| 3.7 边界条件测试 | P1 | 4h | - |
| 3.8 性能基准测试 | P1 | 4h | - |

**验收标准**：
- [ ] 通过 `parallel_api_test`
- [ ] RandomAgent 可运行 100 步
- [ ] RuleBasedAgent 可运行到 episode 结束
- [ ] 冲突检测正确触发惩罚
- [ ] NMAC 正确终止 episode
- [ ] 单步执行时间 < 100ms

### 5.5 Phase 4: 文档完善（Week 6）

**目标**：完善文档和示例

**任务清单**：

| 任务 | 优先级 | 预估工时 | 依赖 |
|------|--------|---------|------|
| 4.1 README.md 编写 | P0 | 4h | - |
| 4.2 API 文档生成 | P1 | 4h | - |
| 4.3 使用示例脚本 | P0 | 4h | - |
| 4.4 配置说明文档 | P1 | 2h | - |
| 4.5 CHANGELOG 初始化 | P2 | 1h | - |

**验收标准**：
- [ ] README 包含安装、使用、配置说明
- [ ] 所有公开 API 有 docstring
- [ ] 至少一个完整示例脚本

---

## 6. 风险评估

### 6.1 技术风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| BlueSky API 不稳定 | 高 | 中 | 深入阅读 bluesky-gym 源码，参考其实现；封装隔离变化 |
| BlueSky 无头模式性能瓶颈 | 中 | 低 | 批量命令处理；性能分析后针对性优化 |
| PettingZoo API 变更 | 低 | 低 | 锁定版本；关注 changelog |
| 动态 Agent 增减导致索引混乱 | 高 | 中 | 使用字符串 ID 而非索引；充分测试边界情况 |
| 观测空间异构性处理困难 | 中 | 中 | 使用 Dict 空间 + padding；参考 PettingZoo 官方示例 |

### 6.2 项目风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 需求变更 | 中 | 中 | spec.md 明确 MVP 边界；V2.0 功能不提前实现 |
| 开发周期超期 | 中 | 中 | 每周 review 进度；优先实现 P0 任务 |
| 测试覆盖不足 | 高 | 低 | TDD 流程；CI 自动检查覆盖率 |
| 依赖冲突 | 低 | 低 | 使用 requirements.txt 锁定版本；定期更新 |

### 6.3 缓解策略汇总

**高风险项专项缓解**：

1. **BlueSky API 不稳定**
   - 第一步：深入阅读 `bluesky-gym` 的 `horizontal_cr_env.py`
   - 第二步：封装 `BlueSkyWrapper` 隔离 API 变化
   - 第三步：编写 API 使用的单元测试，尽早发现问题

2. **动态 Agent 增减**
   - 设计：使用字符串 ID，不依赖列表索引
   - 测试：专门测试飞机进入/离开场景
   - 监控：step() 中添加 Agent 数量变化的日志

---

## 7. 代码规范

### 7.1 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 包名 | 小写 + 下划线 | `bluesky_pettingzoo` |
| 模块名 | 小写 + 下划线 | `observation_manager.py` |
| 类名 | PascalCase | `BlueSkyMARLEnv` |
| 函数名 | snake_case | `get_observation()` |
| 常量 | 大写 + 下划线 | `MAX_OBSERVABLE_AIRCRAFT` |
| 私有方法 | 前缀下划线 | `_normalize_value()` |

### 7.2 类型注解规范

```python
# 函数签名必须有完整类型注解
def get_observation(
    self,
    agent_id: str,
    all_states: dict[str, AircraftState],
) -> NormalizedObservation:
    ...

# 复杂类型使用 TypeAlias
AgentID = str
Observation = dict[str, Any]
```

### 7.3 Docstring 规范

使用 Google 风格：

```python
def compute_reward(
    self,
    agent_id: str,
    prev_state: AircraftState,
    action: DiscreteAction,
    curr_state: AircraftState,
) -> float:
    """计算单步奖励

    Args:
        agent_id: Agent 标识符
        prev_state: 前一步的飞机状态
        action: 执行的离散动作
        curr_state: 当前飞机状态

    Returns:
        计算得到的奖励值

    Raises:
        ValueError: 当 agent_id 不存在时
    """
    ...
```

### 7.4 提交规范

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

类型：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档
- `style`: 格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 工具

示例：
```
feat(observations): implement perception range filtering

- Add ObservationManager.filter_by_perception()
- Support horizontal radius and vertical range
- Unit tests with 95% coverage

Closes #12
```

---

## 8. 配置文件模板

### 8.1 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "bluesky-pettingzoo"
version = "0.1.0"
description = "Multi-agent RL environment for ATM based on BlueSky"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "pettingzoo>=1.24.0",
    "gymnasium>=0.29.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src/bluesky_pettingzoo --cov-report=term-missing"
```

### 8.2 scripts/setup_dev.bat

```batch
@echo off
REM 开发环境初始化脚本（Windows）

echo [1/4] 创建虚拟环境...
python -m venv venv

echo [2/4] 激活虚拟环境...
call .\venv\Scripts\activate.bat

echo [3/4] 升级 pip...
python -m pip install --upgrade pip

echo [4/4] 安装依赖...
pip install -r requirements-dev.txt
pip install -e .

echo.
echo 开发环境初始化完成！
echo 请运行: .\venv\Scripts\activate.bat 激活虚拟环境
pause
```

### 8.3 scripts/run_tests.bat

```batch
@echo off
REM 测试运行脚本（Windows）

call .\venv\Scripts\activate.bat

echo 运行代码检查...
ruff check src/ tests/

echo 运行类型检查...
mypy src/bluesky_pettingzoo/

echo 运行测试...
pytest tests/ -v --cov=src/bluesky_pettingzoo --cov-report=term-missing

pause
```

### 8.2 requirements.txt

```
pettingzoo>=1.24.0
gymnasium>=0.29.0
numpy>=1.24.0
pyyaml>=6.0
```

### 8.3 requirements-dev.txt

```
-r requirements.txt
pytest>=7.0
pytest-cov>=4.0
ruff>=0.1.0
mypy>=1.0
```

---

## 9. 下一步行动

### 9.1 立即行动项

1. **确认 BlueSky 安装方式**
   - 作为 submodule 引用？
   - 还是 pip 安装 `bluesky-sim`？

2. **确认虚拟环境状态**
   - 已安装的包版本
   - Python 版本确认

3. **确认代码仓库位置**
   - BlueSky 源码位置
   - bluesky-gym 参考代码位置

### 9.2 本周计划

| 日期 | 任务 | 产出 |
|------|------|------|
| Day 1 | 项目初始化 + 配置文件 | 可安装的包 |
| Day 2 | BlueSky 无头模式封装 | BlueSkyWrapper 基础版 |
| Day 3 | 观测空间定义 | ObservationManager 骨架 |
| Day 4 | 动作空间定义 | ActionTranslator 骨架 |
| Day 5 | 基础环境骨架 | reset/step 可运行 |

---

## 附录：术语对照

| 英文 | 中文 | 说明 |
|------|------|------|
| Agent | 智能体 | 决策主体 |
| Observation | 观测 | 环境状态的观测值 |
| Action | 动作 | Agent 的决策输出 |
| Reward | 奖励 | 环境反馈信号 |
| Termination | 终止 | 自然结束（到达目标/碰撞） |
| Truncation | 截断 | 强制结束（超时/超出边界） |
| Episode | 回合 | 一次完整的交互过程 |
| Step | 步 | 一次动作-观测循环 |
