# bluesky-marl 技术实现方案 v2.0

> 版本：v2.0
> 日期：2026-05-21
> 基于：spec2/spec.md v2.0 + 现有 V1.0 代码库

---

## 1. 目录结构

### 1.1 现有结构（V1.0 已完成）

```
src/bluesky_pettingzoo/
├── __init__.py
├── envs/
│   ├── __init__.py
│   └── parallel_env.py          # 主环境 BlueSkyMARLEnv
├── bluesky/
│   ├── __init__.py
│   └── wrapper.py               # BlueSky API 封装
├── observations/
│   ├── __init__.py
│   ├── manager.py               # ObservationManager
│   ├── normalizer.py            # 归一化处理
│   └── filters.py               # 感知范围过滤
├── actions/
│   ├── __init__.py
│   └── translator.py            # ActionTranslator
├── rewards/
│   ├── __init__.py
│   ├── base.py                  # RewardComponent ABC
│   ├── calculator.py            # RewardCalculator
│   └── components/
│       ├── __init__.py
│       ├── conflict.py          # ConflictPenalty
│       ├── smoothness.py        # SmoothnessPenalty
│       └── efficiency.py        # EfficiencyReward
├── agents/
│   ├── __init__.py
│   ├── base.py                  # BaseAgent ABC
│   ├── random_agent.py
│   └── rule_based_agent.py
└── utils/
    ├── __init__.py
    ├── types.py                 # 数据模型定义
    └── geometry.py              # 几何计算工具
```

### 1.2 V2.0 扩展结构

```
src/bluesky_pettingzoo/
├── envs/
│   ├── parallel_env.py          # [修改] 支持场景系统、动态进入
│   └── scenarios/               # [新增] 场景模块
│       ├── __init__.py
│       ├── base.py              # BaseScenario 场景基类
│       ├── horizontal_cr.py     # 水平冲突解脱
│       ├── vertical_cr.py       # 垂直冲突解脱
│       ├── sector_cr.py         # 扇区冲突管理
│       ├── merge.py             # 进近汇合
│       └── waypoint_nav.py      # 航路点导航
├── bluesky/
│   └── wrapper.py               # [修改] 支持 ACTION_FREQUENCY、多边形空域
├── observations/
│   ├── manager.py               # [修改] 增强观测空间
│   ├── normalizer.py            # [修改] 支持 cos/sin 分解
│   └── filters.py               # [不变]
├── actions/
│   └── translator.py            # [修改] 支持场景自定义动作维度
├── rewards/
│   └── components/
│       ├── conflict.py          # [修改] 增强冲突检测
│       ├── smoothness.py        # [不变]
│       ├── efficiency.py        # [不变]
│       ├── delay.py             # [新增] 延误惩罚
│       └── capacity.py          # [新增] 容量违反惩罚
└── utils/
    ├── types.py                 # [修改] 新增场景相关类型
    └── geometry.py              # [修改] 新增多边形包含检测
```

### 1.3 配置文件扩展

```
config/
├── default.yaml                 # [不变] 默认配置
├── rewards.yaml                 # [不变] 奖励配置
└── scenarios/                   # [新增] 场景配置目录
    ├── horizontal_cr.yaml
    ├── vertical_cr.yaml
    ├── sector_cr.yaml
    ├── merge.yaml
    └── waypoint_nav.yaml
```

### 1.4 测试文件扩展

```
tests/
├── [现有测试文件不变]
├── test_scenario_base.py        # [新增] 场景基类测试
├── test_dynamic_entry.py        # [新增] 动态进入测试
├── test_observation_enhanced.py # [新增] 增强观测测试
└── integration/
├── [现有集成测试不变]
├── test_horizontal_cr.py        # [新增] 水平冲突场景集成测试
├── test_vertical_cr.py          # [新增] 垂直冲突场景集成测试
└── test_sector_cr.py            # [新增] 扇区冲突场景集成测试
```

---

## 2. 核心数据模型

### 2.1 现有模型（V1.0）

| 模型 | 类型 | 用途 |
|------|------|------|
| `AircraftState` | `__slots__` class | 飞机状态载体（id, lat, lon, alt, hdg, tas, vs） |
| `DiscreteAction` | `NamedTuple` | 离散动作（heading_idx, altitude_idx, speed_idx） |
| `NormalizedObservation` | `TypedDict` | 归一化观测（self_state, other_aircraft, mask, goal） |
| `TextualState` | `TypedDict` | 文本状态（双轨输出） |
| `AirspaceSnapshot` | `TypedDict` | 空域快照 |
| `ConflictLevel` | `IntEnum` | 冲突等级（SAFE, WARNING, NMAC） |

### 2.2 新增模型（V2.0）

#### ScenarioConfig（场景配置）

扩展现有 `ScenarioConfig`，增加场景级参数：

| 字段 | 类型 | 说明 |
|------|------|------|
| name | str | 场景名称 |
| scenario_type | str | 场景类型标识 |
| airspace | AirspaceConfig | 空域配置 |
| aircraft | AircraftConfig | 飞机生成配置 |
| conflict | ConflictConfig | 冲突参数配置 |
| rewards | RewardsConfig | 奖励参数配置 |
| simulation | SimulationConfig | 仿真参数配置 |

#### AirspaceConfig（空域配置）

| 字段 | 类型 | 说明 |
|------|------|------|
| type | str | 空域类型：`rectangular` / `polygon` |
| bounds | list[list[float]] | 矩形边界坐标或多边形顶点 |
| sectors | list[SectorConfig] | 扇区划分（可选） |
| waypoints | list[WaypointConfig] | 航路点定义（可选） |
| no_fly_zones | list[NoFlyZoneConfig] | 禁飞区定义（可选） |

#### SectorConfig（扇区配置）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | str | 扇区标识 |
| bounds | list[list[float]] | 多边形顶点坐标 |
| capacity | int | 最大同时在场飞机数 |

#### WaypointConfig（航路点配置）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | str | 航路点标识 |
| lat | float | 纬度 |
| lon | float | 经度 |
| alt | float \| None | 推荐高度（可选） |

#### AircraftConfig（飞机生成配置）

| 字段 | 类型 | 说明 |
|------|------|------|
| type | str | 飞行器类型（如 A320, B737） |
| initial_count | int | 初始飞机数量 |
| spawn | SpawnConfig | 生成位置和状态范围 |
| dynamic_entry | DynamicEntryConfig \| None | 动态进入配置（可选） |

#### SpawnConfig（生成配置）

| 字段 | 类型 | 说明 |
|------|------|------|
| altitude_range | list[float] | 高度范围 [min, max]（ft） |
| speed_range | list[float] | 速度范围 [min, max]（kt） |
| heading_range | list[float] | 航向范围 [min, max]（度） |
| mode | str | 生成模式：`random` / `boundary` / `waypoint` |

#### DynamicEntryConfig（动态进入配置）

| 字段 | 类型 | 说明 |
|------|------|------|
| enabled | bool | 是否启用动态进入 |
| max_total | int | episode 中最大飞机总数 |
| entry_interval | list[int] | 进入间隔步数范围 [min, max] |
| entry_boundary | str | 进入边界：`random` / `specific` |

#### ConflictConfig（冲突参数配置）

| 字段 | 类型 | 说明 |
|------|------|------|
| horizontal_nm | float | 水平冲突距离（NM） |
| vertical_ft | float | 垂直冲突距离（ft） |
| detection_mode | str | 检测模式：`horizontal_only` / `vertical_only` / `both` |
| warning_horizontal_nm | float | 预警水平距离 |
| warning_vertical_ft | float | 预警垂直距离 |

#### SimulationConfig（仿真参数配置）

| 字段 | 类型 | 说明 |
|------|------|------|
| dt | float | BlueSky 仿真步长（秒） |
| action_frequency | int | 每个 env.step() 内的仿真步进次数 |
| max_episode_steps | int | 最大 episode 步数 |
| headless | bool | 是否无头模式 |

---

## 3. 接口定义

### 3.1 场景基类接口（新增）

```
class BaseScenario(ABC):

    @abstractmethod
    def setup(self, wrapper: BlueSkyWrapper, rng: np.random.RandomState) -> list[str]:
        """初始化场景：创建飞机、定义空域。返回初始 agent ID 列表。"""

    @abstractmethod
    def get_spawn_config(self) -> SpawnConfig:
        """返回飞机生成配置。"""

    @abstractmethod
    def get_conflict_config(self) -> ConflictConfig:
        """返回冲突参数配置。"""

    @abstractmethod
    def should_truncate(self, agent_id: str, state: AircraftState) -> bool:
        """判断是否应截断（如飞机离开空域）。"""

    @abstractmethod
    def get_waypoint(self, agent_id: str) -> WaypointConfig | None:
        """返回指定 agent 的目标航路点。"""

    def update(self, wrapper: BlueSkyWrapper, step: int) -> list[str]:
        """每步更新：处理动态进入等。返回新进入的 agent ID 列表。默认无操作。"""

    def reset(self) -> None:
        """重置场景内部状态。默认无操作。"""
```

### 3.2 BlueSkyWrapper 接口扩展

现有接口不变，新增以下方法：

| 方法 | 签名 | 说明 |
|------|------|------|
| `step_n` | `(self, n: int) -> float` | 执行 n 次仿真步进 |
| `create_polygon_area` | `(self, name: str, points: list[list[float]]) -> None` | 创建多边形区域 |
| `delete_polygon_area` | `(self, name: str) -> None` | 删除多边形区域 |
| `check_inside_polygon` | `(self, name: str, lat: float, lon: float) -> bool` | 检查点是否在多边形内 |
| `get_aircraft_gs` | `(self, acid: str) -> float` | 获取地速（knots） |

### 3.3 ObservationManager 接口扩展

现有接口不变，观测空间维度调整：

**self_state**（shape 变化：6 → 8）：

| 索引 | 字段 | 说明 |
|------|------|------|
| 0 | heading_cos | 航向 cos 分量 |
| 1 | heading_sin | 航向 sin 分量 |
| 2 | altitude | 归一化高度 |
| 3 | speed | 归一化速度 |
| 4 | lat | 纬度 |
| 5 | lon | 经度 |
| 6 | vs | 垂直速率 |
| 7 | ground_speed | 归一化地速 |

**other_aircraft**（shape 变化：(MAX_OBS, 7) → (MAX_OBS, 9)）：

| 索引 | 字段 | 说明 |
|------|------|------|
| 0 | relative_bearing_cos | 相对方位 cos |
| 1 | relative_bearing_sin | 相对方位 sin |
| 2 | distance | 归一化距离 |
| 3 | relative_altitude | 归一化相对高度 |
| 4 | relative_speed_x | 归一化相对速度 x 分量 |
| 5 | relative_speed_y | 归一化相对速度 y 分量 |
| 6 | heading | 归一化航向 |
| 7 | altitude | 归一化高度 |
| 8 | speed | 归一化速度 |

**goal**（shape 不变：4，但语义调整）：

| 索引 | 字段 | 说明 |
|------|------|------|
| 0 | distance | 归一化距离 |
| 1 | bearing_cos | 目标方位 cos |
| 2 | bearing_sin | 目标方位 sin |
| 3 | alt_diff | 归一化高度差 |

### 3.4 ActionTranslator 接口扩展

现有接口不变，新增支持场景自定义动作空间：

| 方法 | 签名 | 说明 |
|------|------|------|
| `action_space` | `(self) -> spaces.MultiDiscrete` | 返回当前场景的动作空间 |

动作空间维度由场景配置决定，不再固定为 `[5, 5, 5]`。

### 3.5 RewardComponent 接口扩展

现有接口不变，新增两个奖励组件：

#### DelayPenalty（延误惩罚）

```
class DelayPenalty(RewardComponent):
    def compute(self, agent_id, prev_state, action, curr_state, all_states) -> float:
        """每步给予固定惩罚 -0.01，鼓励尽快到达。"""
```

#### CapacityPenalty（容量违反惩罚）

```
class CapacityPenalty(RewardComponent):
    def compute(self, agent_id, prev_state, action, curr_state, all_states) -> float:
        """扇区内飞机数超过容量上限时给予惩罚。"""
```

### 3.6 BlueSkyMARLEnv 接口扩展

现有接口不变，构造函数新增可选 `scenario` 参数：

```
def __init__(
    self,
    config: dict[str, Any],
    wrapper: BlueSkyWrapper,
    observation_manager: ObservationManager,
    action_translator: ActionTranslator,
    reward_calculator: RewardCalculator,
    rewards_config: dict[str, Any],
    scenario: BaseScenario | None = None,  # [新增] 场景实例
) -> None
```

`step()` 方法变更：
- 每步调用 `scenario.update()` 获取新进入的飞机
- 每步调用 `scenario.should_truncate()` 判断截断
- 到达目标航路点时触发 termination

---

## 4. 实施阶段

### Phase 1: V1.0 缺失功能补齐

**目标**：补齐 spec2 第 2 章定义的 5 项缺失功能。

#### 任务 1.1：到达目标终止

| 字段 | 值 |
|------|-----|
| 修改文件 | `parallel_env.py` |
| 依赖 | 无 |
| 说明 | 在 step() 中检查飞机是否到达目标航路点，到达则 termination=True |

验收：
- 飞机到达目标后从 agents 列表移除
- terminations 字典中该 agent 为 True
- 其他 agent 不受影响

#### 任务 1.2：仿真时间步长细化

| 字段 | 值 |
|------|-----|
| 修改文件 | `wrapper.py`, `parallel_env.py`, `config/default.yaml` |
| 依赖 | 无 |
| 说明 | BlueSkyWrapper 新增 step_n() 方法，parallel_env 每步调用多次仿真步进 |

验收：
- ACTION_FREQUENCY 可配置
- 每个 env.step() 内执行多次 bs.sim.step()
- 飞机状态在多次步进后正确更新

#### 任务 1.3：观测空间增强 — 方位角分解

| 字段 | 值 |
|------|-----|
| 修改文件 | `normalizer.py`, `manager.py`, `filters.py` |
| 依赖 | 无 |
| 说明 | 将 bearing 拆分为 cos/sin，heading 拆分为 cos/sin |

验收：
- 方位角在 0/360 度边界无跳变
- 观测空间维度与配置一致
- 现有测试更新后通过

#### 任务 1.4：观测空间增强 — 相对速度分量

| 字段 | 值 |
|------|-----|
| 修改文件 | `manager.py`, `normalizer.py`, `types.py` |
| 依赖 | 任务 1.3 |
| 说明 | 增加与其他飞机的相对速度 x/y 分量 |

验收：
- 相对速度信息包含在 other_aircraft 观测中
- 观测空间维度从 (MAX_OBS, 7) 变为 (MAX_OBS, 9)
- 相对速度可正确反映冲突趋势

#### 任务 1.5：飞机动态进入空域

| 字段 | 值 |
|------|-----|
| 修改文件 | `parallel_env.py`, `types.py` |
| 依赖 | 无 |
| 说明 | 支持 episode 过程中新的飞机从空域边界进入 |

验收：
- 新飞机可从空域边界进入
- 进入后自动出现在 agents 列表中
- 进入的飞机获得正确的初始观测
- 进入时机和参数可配置

#### 任务 1.6：真实 BlueSky 集成验证

| 字段 | 值 |
|------|-----|
| 修改文件 | `wrapper.py`（如需要） |
| 依赖 | 任务 1.1-1.5 |
| 说明 | 使用真实 BlueSky 运行完整的 reset/step/close 循环 |

验收：
- 使用真实 BlueSky 可完成 reset/step/close
- 飞机状态读取与 BlueSky 内部状态一致
- 无内存泄漏或资源未释放

---

### Phase 2: 场景系统基础

**目标**：实现场景基类和配置系统，为具体场景奠定基础。

#### 任务 2.1：场景配置数据模型

| 字段 | 值 |
|------|-----|
| 修改文件 | `types.py`, `config/scenarios/` |
| 依赖 | 无 |
| 说明 | 定义场景配置相关的数据模型（ScenarioConfig, AirspaceConfig 等） |

验收：
- 所有配置类型可通过 mypy 检查
- YAML 配置文件可正确加载为配置对象

#### 任务 2.2：场景基类

| 字段 | 值 |
|------|-----|
| 新增文件 | `envs/scenarios/base.py` |
| 依赖 | 任务 2.1 |
| 说明 | 实现 BaseScenario 抽象基类 |

验收：
- 基类定义清晰的抽象接口
- 可被具体场景继承

#### 任务 2.3：parallel_env 集成场景系统

| 字段 | 值 |
|------|-----|
| 修改文件 | `parallel_env.py` |
| 依赖 | 任务 2.2 |
| 说明 | 主环境支持可选的场景实例，场景驱动飞机生成和生命周期管理 |

验收：
- 无场景时行为与 V1.0 一致
- 有场景时由场景驱动飞机生成

---

### Phase 3: 具体场景实现

**目标**：实现 3 个核心场景（水平冲突、垂直冲突、扇区冲突）。

#### 任务 3.1：水平冲突解脱场景

| 字段 | 值 |
|------|-----|
| 新增文件 | `envs/scenarios/horizontal_cr.py`, `config/scenarios/horizontal_cr.yaml` |
| 依赖 | 任务 2.2 |
| 参考 | bluesky-gym horizontal_cr_env |
| 说明 | 多架飞机巡航，航向机动避碰，到达目标终止 |

关键参数：
- 飞行器：A320，150 kt
- 冲突距离：5 NM
- 航路点距离：100-150 NM
- 动作维度：航向

验收：
- 场景可正常初始化和运行
- 冲突检测正确
- 到达目标正确终止

#### 任务 3.2：垂直冲突解脱场景

| 字段 | 值 |
|------|-----|
| 新增文件 | `envs/scenarios/vertical_cr.py`, `config/scenarios/vertical_cr.yaml` |
| 依赖 | 任务 2.2 |
| 参考 | bluesky-gym vertical_cr_env |
| 说明 | 多架飞机在不同高度层，升降机动避碰 |

关键参数：
- 初始高度：2000-4000 ft
- 垂直间隔：1000 ft
- 水平冲突距离：5 NM
- 升降速率：±2500 ft/min
- 动作维度：升降速率

验收：
- 同时检测水平和垂直间隔违反
- 两个间隔必须同时违反才算冲突

#### 任务 3.3：扇区冲突管理场景

| 字段 | 值 |
|------|-----|
| 新增文件 | `envs/scenarios/sector_cr.py`, `config/scenarios/sector_cr.yaml` |
| 依赖 | 任务 2.2 |
| 参考 | bluesky-gym sector_cr_env |
| 说明 | 多边形扇区内多架飞机，航向+速度机动 |

关键参数：
- 扇区面积：2400-3750 NM²
- 飞行密度：0.003-0.007 架/NM²
- 巡航高度：FL350
- 冲突距离：5 NM
- 动作维度：航向 + 速度

验收：
- 扇区边界为多边形
- 飞机离开扇区边界时截断（非终止）
- 飞机密度可配置

---

### Phase 4: 增强冲突检测与流量管理

**目标**：增强冲突检测能力，新增流量管理相关组件。

#### 任务 4.1：冲突检测增强

| 字段 | 值 |
|------|-----|
| 修改文件 | `rewards/components/conflict.py`, `parallel_env.py` |
| 依赖 | 无 |
| 说明 | 支持预测性冲突检测、多机冲突关联 |

验收：
- 基于速度和航向预测未来冲突
- 识别连锁冲突

#### 任务 4.2：延误惩罚组件

| 字段 | 值 |
|------|-----|
| 新增文件 | `rewards/components/delay.py` |
| 依赖 | 无 |
| 说明 | 每步给予固定惩罚，鼓励尽快到达 |

验收：
- 每步惩罚值可配置
- 与现有奖励组件正确叠加

#### 任务 4.3：容量违反惩罚组件

| 字段 | 值 |
|------|-----|
| 新增文件 | `rewards/components/capacity.py` |
| 依赖 | 任务 3.3 |
| 说明 | 扇区内飞机数超过容量上限时给予惩罚 |

验收：
- 容量阈值可配置
- 超限时正确计算惩罚

#### 任务 4.4：航路点导航场景

| 字段 | 值 |
|------|-----|
| 新增文件 | `envs/scenarios/waypoint_nav.py`, `config/scenarios/waypoint_nav.yaml` |
| 依赖 | 任务 2.2 |
| 参考 | bluesky-gym plan_waypoint_env |
| 说明 | 无冲突的纯导航任务，测试制导逻辑 |

验收：
- 飞机按航路点飞行
- 到达目标正确终止

#### 任务 4.5：进近汇合场景

| 字段 | 值 |
|------|-----|
| 新增文件 | `envs/scenarios/merge.py`, `config/scenarios/merge.yaml` |
| 依赖 | 任务 2.2 |
| 参考 | bluesky-gym merge_env |
| 说明 | 飞机进近降落，与背景交通保持间隔 |

关键参数：
- 进近速度：100 kt
- 汇合冲突距离：4 NM
- 飞机总数：20 架（1 主控 + 19 背景交通）
- 可观测邻居数：5 架

验收：
- 背景交通不可控，按预设航路飞行
- 主控飞机需找到安全间隙汇入

---

### Phase 5: 测试与集成

**目标**：为所有新增功能编写测试，确保整体质量。

#### 任务 5.1：场景基类测试

| 字段 | 值 |
|------|-----|
| 新增文件 | `tests/test_scenario_base.py` |
| 依赖 | 任务 2.2 |
| 说明 | 测试场景基类的接口和生命周期 |

#### 任务 5.2：动态进入测试

| 字段 | 值 |
|------|-----|
| 新增文件 | `tests/test_dynamic_entry.py` |
| 依赖 | 任务 1.5 |
| 说明 | 测试飞机动态进入空域的完整流程 |

#### 任务 5.3：增强观测测试

| 字段 | 值 |
|------|-----|
| 新增文件 | `tests/test_observation_enhanced.py` |
| 依赖 | 任务 1.3, 1.4 |
| 说明 | 测试 cos/sin 分解和相对速度分量 |

#### 任务 5.4：场景集成测试

| 字段 | 值 |
|------|-----|
| 新增文件 | `tests/integration/test_horizontal_cr.py`, `test_vertical_cr.py`, `test_sector_cr.py` |
| 依赖 | 任务 3.1-3.3 |
| 说明 | 每个场景的端到端集成测试 |

---

## 5. 依赖关系图

```
Phase 1（V1.0 补齐）:
  1.1 到达终止 ──┐
  1.2 步长细化 ──┤
  1.3 方位分解 ──┼──→ 1.6 真实 BlueSky 验证
  1.4 相对速度 ──┤
  1.5 动态进入 ──┘

Phase 2（场景基础）:
  2.1 配置模型 ──→ 2.2 场景基类 ──→ 2.3 env 集成

Phase 3（具体场景）:
  2.2 ──→ 3.1 水平冲突
  2.2 ──→ 3.2 垂直冲突
  2.2 ──→ 3.3 扇区冲突

Phase 4（增强功能）:
  3.3 ──→ 4.3 容量惩罚
  2.2 ──→ 4.4 航路点导航
  2.2 ──→ 4.5 进近汇合

Phase 5（测试）:
  各任务完成后并行编写测试
```

---

## 6. 关键设计决策

### 6.1 场景与环境的关系

场景（Scenario）是环境的可插拔组件，通过依赖注入传入 BlueSkyMARLEnv。无场景时环境行为与 V1.0 完全一致，保证向后兼容。

### 6.2 动作空间的场景化

不同场景可定义不同的动作维度。水平冲突只需航向，垂直冲突只需升降速率，扇区冲突需要航向+速度。ActionTranslator 根据场景配置动态调整。

### 6.3 观测空间的向后兼容

观测空间增强（cos/sin 分解、相对速度）会改变维度，需要同步更新所有依赖观测空间的代码和测试。通过配置开关控制是否启用增强观测。

### 6.4 BlueSky API 复用

优先使用 BlueSky 内置工具：
- `bs.tools.geo.kwikqdrdist` — 方位和距离计算
- `bs.tools.geo.kwikpos` — 从方位距离推算位置
- `bs.tools.areafilter.defineArea` — 定义多边形区域
- `bs.tools.areafilter.checkInside` — 点在多边形内检测
- `bs.traf.creconfs` — 自动生成冲突场景
