# V3.0 环境健壮性改进计划

> 制定日期：2026-06-11
> 范围：基于 env_comparison.md 评审发现的环境实现缺陷，制定系统性改进方案
> 原则：先规划、后实现，每个 feature 独立可验证

---

## 一、问题总览

| 编号 | 严重度 | 问题 | 影响范围 | 涉及文件 |
|------|--------|------|----------|----------|
| F1 | **CRITICAL** | 观测空间缺少 conflict_state | 所有场景的 RL 训练 | observation_manager.py, observation_builder.py |
| F2 | **CRITICAL** | step() 无异常处理 | 所有场景的鲁棒性 | parallel_env.py |
| F3 | **HIGH** | EfficiencyReward 忽略高度维度 | 垂直冲突场景 | efficiency.py |
| F4 | **HIGH** | DelayPenalty 预期步数静态不变 | 所有场景的延迟惩罚 | delay.py |
| F5 | **MEDIUM** | 观测零填充产生虚假航空器 | 高密度场景 | observation_manager.py, perception filter |
| F6 | **MEDIUM** | max_observable_aircraft 固定为 10 | 场景密度自适应 | observation manager, config |
| F7 | **MEDIUM** | 渲染器直接访问内部状态 | 渲染器可维护性 | renderer 各文件, parallel_env.py |
| F8 | **LOW** | 部分场景初始位置固定 | 训练泛化性 | vertical_cr.yaml, horizontal_cr.yaml |

---

## 二、各 Feature 详细设计

### F1: 观测空间添加 conflict_state（CRITICAL）

**问题分析**：

`observation_builder.py:120` 计算了 `conflict_status`（"nmac"/"warning"/"safe"），但只传给了 `ObservationManager.generate()` 的 `textual_state`，没有进入 `observation` dict。Agent 无法直接观察到自身是否处于冲突状态，只能通过其他机的位置间接推断。

在 `conflict.py:200-221` 的 `get_conflict_status()` 已经实现了完整的冲突检测逻辑，但结果被浪费了。

**改进方案**：

在 `observation` dict 中添加 `conflict_state` 字段：

```python
# observation_manager.py generate() 方法中
# 在构建 observation dict 时添加：
conflict_vec = self._encode_conflict_status(conflict_status)
observation["conflict_state"] = conflict_vec  # shape: (3,), one-hot
```

编码方式：one-hot 向量 `[is_nmac, is_warning, is_safe]`，与 observation space 的 `Box(-1, 1)` 范围兼容。

**影响文件**：
- `src/bluesky_pettingzoo/observations/manager.py` — 添加 `_encode_conflict_status()` 方法，修改 `observation_space()` 和 `generate()`
- `tests/test_observation_manager.py` — 添加 conflict_state 维度测试

**验证标准**：
- `observation_space()["conflict_state"]` shape 为 `(3,)`，dtype 为 `float32`
- NMAC 状态下 `conflict_state[0] == 1.0`
- Warning 状态下 `conflict_state[1] == 1.0`
- Safe 状态下 `conflict_state[2] == 1.0`
- 所有现有测试不退化

---

### F2: step() 异常处理与安全回退（CRITICAL）

**问题分析**：

`parallel_env.py:333-602` 的 `step()` 方法中，`self._wrapper.step_n()`、`self._wrapper.send_commands_batch()` 等 BlueSky 调用没有 try-catch。如果引擎内部崩溃（航空器飞出边界、冲突检测死循环、数值溢出），整个环境会抛异常，训练进程中断。

**改进方案**：

```python
def step(self, actions):
    try:
        # ... 现有逻辑 ...
        self._wrapper.send_commands_batch(commands)
        self._wrapper.step_n(self._action_frequency, on_substep=_on_substep)
    except Exception as e:
        # 安全回退：返回全零观测 + 负奖励 + done=True
        logging.error(f"BlueSky engine error at step {self._step_count}: {e}")
        return self._safe_termination_fallback(actions)
```

新增 `_safe_termination_fallback()` 方法：
- 所有 agent 的 reward 设为 `config["simulation"].get("crash_penalty", -100.0)`
- 所有 agent 的 terminated 设为 `True`
- observations 返回 `default_observation()`
- infos 中记录错误信息

**影响文件**：
- `src/bluesky_pettingzoo/envs/parallel_env.py` — step() 添加 try-catch，新增 `_safe_termination_fallback()`
- `tests/test_parallel_env.py` — 添加异常回退测试

**验证标准**：
- 模拟 BlueSky 抛异常时，step() 返回正确的 (obs, rewards, terms, truncs, infos) 结构
- rewards 全部为 crash_penalty 负值
- terms 全部为 True
- 日志记录错误信息
- 现有测试不退化

---

### F3: EfficiencyReward 增加高度维度（HIGH）

**问题分析**：

`efficiency.py:61-66` 的距离计算只用了 `haversine_distance(lat, lon)`，完全忽略高度差。在 VerticalCR 场景中，5 架飞机在不同高度层（29000-37000 ft），RL agent 需要调整垂直速度到达目标高度，但 reward 不感知高度进展。

**改进方案**：

在 `EfficiencyReward.compute()` 中添加高度偏差惩罚：

```python
# 效率奖励 = 航路偏差 + 到达奖励 + 步惩罚 + 高度偏差
alt_penalty = 0.0
if goal.get("alt") is not None:
    alt_diff = abs(curr_state.alt - goal["alt"])
    alt_penalty = -(alt_diff / self._max_alt_deviation) * self._alt_deviation_scale
    alt_penalty = max(alt_penalty, -self._alt_deviation_scale)
reward += alt_penalty
```

新增配置项：
- `max_alt_deviation_ft`: 最大高度偏差参考值（默认 10000 ft）
- `alt_deviation_penalty_scale`: 高度偏差惩罚系数（默认 2.0）

`set_goal()` 方法签名扩展：`set_goal(agent_id, lat, lon, alt=None)`

**影响文件**：
- `src/bluesky_pettingzoo/rewards/components/efficiency.py` — 添加高度偏差计算
- `config/rewards.yaml` — 添加新配置项
- `tests/test_reward_efficiency.py` — 添加高度偏差测试

**验证标准**：
- 有高度目标时，高度偏差被正确计算和惩罚
- 无高度目标时（alt=None），行为与现有逻辑一致
- 配置项可从 rewards.yaml 读取
- 所有现有奖励测试不退化

---

### F4: DelayPenalty 动态更新预期步数（HIGH）

**问题分析**：

`delay.py:36-55` 的 `set_goal()` 在 reset 时一次性计算预期步数，之后不再更新。如果 agent 中途减速（让行冲突）或改变航路，预期步数不会调整，导致不合理的延迟惩罚。

**改进方案**：

在 `compute()` 中动态调整预期步数：

```python
def compute(self, agent_id, prev_state, action, curr_state, all_states, step_count=0):
    expected = self._expected_steps.get(agent_id)
    if expected is None:
        return 0.0

    # 动态调整：如果当前速度与初始速度偏差大，按比例缩放预期步数
    initial_speed = self._initial_speeds.get(agent_id)
    if initial_speed is not None and initial_speed > 0:
        speed_ratio = initial_speed / max(curr_state.tas, 1.0)
        adjusted_expected = int(expected * speed_ratio)
    else:
        adjusted_expected = expected

    overdue = step_count - adjusted_expected
    if overdue <= 0:
        return 0.0
    return overdue * self._penalty_per_step
```

新增 `_initial_speeds` 字典，在 `set_goal()` 时记录初始速度。

**影响文件**：
- `src/bluesky_pettingzoo/rewards/components/delay.py` — 添加动态调整逻辑
- `tests/test_reward_delay.py` — 添加速度变化场景测试

**验证标准**：
- agent 减速时，预期步数按比例增加，不产生不合理惩罚
- agent 保持巡航速度时，行为与现有逻辑一致
- 所有现有延迟测试不退化

---

### F5: 观测零填充优化（MEDIUM）

**问题分析**：

`observation_manager.py:146` 中 `other_aircraft` 数组用零填充未使用的槽位。虽然 `other_aircraft_mask` 标记了真实/填充，但网络仍需处理零向量，可能学到无意义的响应模式。

**改进方案**：

方案 A（推荐）：保持零填充 + mask，但在训练文档中明确说明 mask 的使用方式。这是 PettingZoo 标准做法，改动最小。

方案 B：使用 padding_value = -1（表示"无数据"），与真实航空器的归一化范围 [0, 1] 区分更明显。

**选择方案 A**，因为：
- mask 已经正确实现
- RL 网络（PPO/SAC）通常能学会忽略 mask=0 的槽位
- 改动最小，风险最低

但仍需改进：在 `PerceptionFilter.filter()` 中，当航空器不足 `max_observable` 时，当前返回截断后的列表，由 `ObservationManager` 负责填充。这个流程是正确的，无需修改。

**影响文件**：无需修改代码，仅更新文档说明

**验证标准**：
- 文档明确说明 mask 机制
- 现有测试不退化

---

### F6: max_observable_aircraft 动态配置（MEDIUM）

**问题分析**：

`default.yaml:26` 中 `max_observable_aircraft: 10` 是固定的。在 5 架飞机的场景中，10 个槽位有 5 个是零填充；如果未来扩展到 20 架飞机，10 个槽位又不够。

**改进方案**：

支持从场景配置覆盖：

```python
# observation_manager.py __init__ 中
# 优先使用场景级别的配置
def update_max_observable(self, max_obs: int) -> None:
    """Update max observable aircraft (called by scenario setup)."""
    self._max_obs = max_obs
    # 重建 observation_space
    self._cached_space = self._observation_space()
```

在 `BaseScenario` 中添加 `max_observable_aircraft` 属性：

```python
@property
def max_observable_aircraft(self) -> int | None:
    """Override to set scenario-specific max observable aircraft."""
    return None
```

**影响文件**：
- `src/bluesky_pettingzoo/observations/manager.py` — 添加 `update_max_observable()`
- `src/bluesky_pettingzoo/envs/scenarios/base.py` — 添加属性
- `src/bluesky_pettingzoo/envs/parallel_env.py` — reset() 中调用更新

**验证标准**：
- 场景可覆盖 max_observable_aircraft
- 未覆盖时保持默认值 10
- observation_space 随配置变化
- 所有现有测试不退化

---

### F7: 渲染器接口解耦（MEDIUM）

**问题分析**：

渲染器通过 `self.env.agents` 和 `self.env.pz_env` 直接访问环境内部状态，违反封装原则。环境重构时渲染器会第一个崩溃。

**改进方案**：

定义 `RendererDataSource` Protocol：

```python
class RendererDataSource(Protocol):
    """渲染器所需的数据源接口。"""
    def get_aircraft_states(self) -> dict[str, Any]: ...
    def get_waypoints(self) -> dict[str, dict[str, float]] | None: ...
    def get_step_count(self) -> int: ...
    def get_active_agents(self) -> list[str]: ...
```

修改渲染器构造函数：

```python
class HorizontalCRRenderer:
    def __init__(self, data_source: RendererDataSource, **kwargs): ...
```

在 `parallel_env.py` 中创建 `EnvRendererAdapter` 实现此 Protocol。

**影响文件**：
- `src/bluesky_pettingzoo/utils/protocols.py` — 添加 `RendererDataSource` Protocol
- `src/bluesky_pettingzoo/envs/parallel_env.py` — 添加 `EnvRendererAdapter`
- 所有渲染器文件 — 修改构造函数签名
- `tests/test_renderers.py` — 更新渲染器测试

**验证标准**：
- 渲染器不再直接访问 `env.agents` 或 `env.pz_env`
- 所有渲染测试通过
- 渲染效果不变

---

### F8: 场景初始位置随机化（LOW）

**问题分析**：

`vertical_cr.yaml` 等配置文件中，航空器初始位置是固定数组。每次 reset 时，agent 面对完全相同的初始构型，学到的是特定排列的最优解。

**改进方案**：

在场景 YAML 中支持分布采样：

```yaml
# vertical_cr.yaml 改进后
initial_positions:
  x:
    dist: uniform
    low: 2.0
    high: 8.0
  y:
    dist: uniform
    low: 4.0
    high: 6.0
  alt:
    values: [29000, 31000, 33000, 35000, 37000]
    shuffle: true
```

在 `BaseScenario` 中添加 `_sample_initial_positions()` 工具方法。

**影响文件**：
- `src/bluesky_pettingzoo/envs/scenarios/base.py` — 添加采样工具方法
- `src/bluesky_pettingzoo/envs/scenarios/vertical_cr.py` — 使用采样替代固定值
- `config/scenarios/vertical_cr.yaml` — 更新配置格式
- `tests/test_scenario_enhance.py` — 添加随机化测试

**验证标准**：
- 每次 reset 生成不同的初始位置
- 配置格式向后兼容（固定数组仍可用）
- 所有现有场景测试不退化

---

## 三、实施顺序

```
Phase 1（基础健壮性）—— 预计 2-3 天
├── F1: 观测空间添加 conflict_state [CRITICAL]
└── F2: step() 异常处理 [CRITICAL]

Phase 2（奖励函数修正）—— 预计 2-3 天
├── F3: EfficiencyReward 高度维度 [HIGH]
└── F4: DelayPenalty 动态调整 [HIGH]

Phase 3（观测与配置优化）—— 预计 1-2 天
├── F5: 零填充文档说明 [MEDIUM]
└── F6: max_observable 动态配置 [MEDIUM]

Phase 4（架构解耦）—— 预计 2-3 天
├── F7: 渲染器接口解耦 [MEDIUM]
└── F8: 场景初始位置随机化 [LOW]
```

**总工作量**：7-11 天

---

## 四、Feature ID 分配

| Feature ID | 标题 | 优先级 | 状态 |
|------------|------|--------|------|
| robust-001 | 观测空间添加 conflict_state | P0 | not_started |
| robust-002 | step() 异常处理与安全回退 | P0 | not_started |
| reward-002 | EfficiencyReward 高度维度 | P1 | not_started |
| reward-003 | DelayPenalty 动态预期步数 | P1 | not_started |
| obs-002 | 观测零填充文档说明 | P2 | not_started |
| obs-003 | max_observable 动态配置 | P2 | not_started |
| arch-003 | 渲染器接口解耦 | P2 | not_started |
| scenario-002 | 场景初始位置随机化 | P3 | not_started |

---

## 五、风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| F1 改动 observation space shape | 所有现有训练模型失效 | 提供兼容模式，新字段可选启用 |
| F2 异常处理掩盖真实 bug | 问题被静默吞掉 | 日志记录完整 traceback，不 swallow |
| F3 高度惩罚权重不当 | 垂直场景训练不稳定 | 默认权重 0，渐进调整 |
| F7 渲染器重构工作量大 | 影响 10 个渲染器文件 | 分批迁移，先添加 adapter 再改渲染器 |

---

## 六、完成门槛

每个 feature 完成必须满足：

1. 单元测试通过（`pytest tests/ -v --ignore=tests/integration`）
2. 代码检查通过（`ruff check src/ tests/`）
3. 格式检查通过（`ruff format --check src/ tests/`）
4. 类型检查通过（`mypy src/bluesky_pettingzoo/`）
5. feature_list.json 状态更新为 `passing`
6. 测试覆盖率 >= 90%
