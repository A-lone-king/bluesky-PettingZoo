# bluesky-marl V2.0 原子任务列表

> 基于 spec2/spec.md v2.0 和 spec2/plan.md v2.0
> 规则：每个任务只改一个文件，奇数任务写测试，偶数任务写实现
> 编号规则：`S-XX` 脚手架，`T-XX` 测试，`I-XX` 实现，`G-XX` 集成测试

---

## Phase 1: V1.0 缺失功能补齐

> TDD 循环：到达终止 → 步长细化 → 观测增强 → 动态进入 → 集成验证

### T-V01: 测试 — 到达目标终止 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_arrival_termination.py` |
| 依赖 | 无 |
| 说明 | 测试飞机到达目标航路点后 termination=True 并从 agents 移除 |

测试用例：
1. `test_arrival_triggers_termination` — 飞机到达目标后 terminations 为 True
2. `test_arrival_removes_from_agents` — 到达后飞机从 agents 列表移除
3. `test_arrival_other_agents_unaffected` — 其他未到达 agent 不受影响
4. `test_arrival_threshold_configurable` — 到达阈值可配置
5. `test_arrival_not_reached_no_termination` — 未到达时不触发终止

验收：
- [x] 所有用例通过（5/5）
- [x] `pytest tests/test_arrival_termination.py` 可运行

---

### I-V01: 实现 — 到达目标终止 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/envs/parallel_env.py` |
| 依赖 | T-V01 |
| 说明 | 在 step() 中检查飞机是否到达目标航路点，到达则 termination=True |

实现要点：
- 复用 EfficiencyReward 中的到达判定逻辑（距离 < arrival_threshold_nm）
- 从 efficiency 组件获取目标航路点坐标
- 到达后从 self.agents 移除，调用 wrapper.remove_aircraft()
- 在 terminations 字典中标记为 True

验收：
- [x] T-V01 所有用例通过

---

### T-V02: 测试 — 仿真时间步长细化 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_action_frequency.py` |
| 依赖 | 无 |
| 说明 | 测试每个 env.step() 内执行多次仿真步进 |

测试用例：
1. `test_step_n_executes_n_times` — step_n(n) 执行 n 次仿真步进
2. `test_action_frequency_configurable` — ACTION_FREQUENCY 可通过配置设定
3. `test_state_after_multiple_steps` — 多次步进后飞机状态正确更新
4. `test_default_frequency_is_1` — 默认频率为 1（向后兼容）
5. `test_time_advances_correctly` — 仿真时间按 dt * frequency 推进

验收：
- [x] 所有用例通过（11/11）

---

### I-V02: 实现 — 仿真时间步长细化 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/bluesky/wrapper.py` |
| 依赖 | T-V02 |
| 说明 | BlueSkyWrapper 新增 step_n() 方法 |

实现要点：
- 新增 `step_n(self, n: int) -> float` 方法，内部调用 n 次 `bs.sim.step()`
- 修改 `step()` 方法调用 `step_n(1)` 保持向后兼容
- config 中新增 `action_frequency` 参数

验收：
- [x] T-V02 所有用例通过

---

### T-V03: 测试 — 观测空间增强（方位角分解） ✅

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_observation_enhanced.py` |
| 依赖 | 无 |
| 说明 | 测试方位角 cos/sin 分解和航向 cos/sin 分解 |

测试用例：
1. `test_heading_cos_sin_0` — 航向 0° → cos=1, sin=0
2. `test_heading_cos_sin_90` — 航向 90° → cos=0, sin=1
3. `test_heading_cos_sin_180` — 航向 180° → cos=-1, sin=0
4. `test_heading_cos_sin_270` — 航向 270° → cos=0, sin=-1
5. `test_bearing_cos_sin_north` — 正北方位 → cos=1, sin=0
6. `test_bearing_cos_sin_east` — 正东方位 → cos=0, sin=1
7. `test_bearing_no_discontinuity` — 359° 和 1° 的 cos/sin 连续
8. `test_self_state_shape` — self_state 维度为 8
9. `test_goal_bearing_cos_sin` — goal 中方位为 cos/sin 两个分量

验收：
- [x] 所有用例通过（9/9）

---

### I-V03: 实现 — 观测空间增强（方位角分解） ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/observations/normalizer.py` |
| 依赖 | T-V03 |
| 说明 | 将 bearing 和 heading 拆分为 cos/sin 分量 |

实现要点：
- 新增 `normalize_heading_cos(heading)` 和 `normalize_heading_sin(heading)` 函数
- 新增 `normalize_bearing_cos(bearing)` 和 `normalize_bearing_sin(bearing)` 函数
- heading cos/sin：`cos(hdg * π/180)`, `sin(hdg * π/180)`
- bearing cos/sin：同上，但基于相对方位角

验收：
- [x] T-V03 所有用例通过

---

### T-V04: 测试 — 观测空间增强（相对速度分量）

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_observation_enhanced.py`（追加到 T-V03 文件） |
| 依赖 | T-V03 |
| 说明 | 测试相对速度 x/y 分量的计算 |

测试用例：
1. `test_relative_speed_head_on` — 对头飞行：相对速度 = 两机速度之和
2. `test_relative_speed_parallel` — 平行飞行：相对速度 ≈ 0
3. `test_relative_speed_crossing` — 交叉飞行：x/y 分量非零
4. `test_other_aircraft_shape` — other_aircraft 维度为 (MAX_OBS, 9)
5. `test_relative_speed_normalized` — 相对速度已归一化

验收：
- [x] 所有用例通过（5/5）

---

### I-V04: 实现 — 观测空间增强（相对速度分量）

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/observations/manager.py` |
| 依赖 | T-V04, I-V03 |
| 说明 | 在 other_aircraft 中增加相对速度 x/y 分量 |

实现要点：
- 计算相对速度：将两机的速度和航向分解为 x/y 分量后相减
- other_aircraft 从 7 维扩展到 9 维
- 更新 observation_space 定义
- 更新 self_state 从 6 维到 8 维（heading → cos/sin, 新增 ground_speed）
- goal 从 [distance, bearing, alt_diff, hdg_diff] 改为 [distance, bearing_cos, bearing_sin, alt_diff]

验收：
- [x] T-V04 所有用例通过

---

### T-V05: 测试 — 飞机动态进入空域

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_dynamic_entry.py` |
| 依赖 | 无 |
| 说明 | 测试 episode 过程中新的飞机从空域边界进入 |

测试用例：
1. `test_dynamic_entry_adds_agent` — 新飞机进入后出现在 agents 列表中
2. `test_dynamic_entry_gets_observation` — 进入的飞机获得正确初始观测
3. `test_dynamic_entry_configurable_interval` — 进入间隔可配置
4. `test_dynamic_entry_max_total` — 飞机总数不超过 max_total
5. `test_dynamic_entry_from_boundary` — 新飞机从空域边界进入
6. `test_dynamic_entry_disabled_by_default` — 默认不启用动态进入

验收：
- [x] 所有用例通过（6/6）

---

### I-V05: 实现 — 飞机动态进入空域

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/envs/parallel_env.py` |
| 依赖 | T-V05, I-V01 |
| 说明 | 支持 episode 过程中新的飞机从空域边界进入 |

实现要点：
- 在 step() 中检查是否需要生成新飞机
- 新飞机从空域边界随机位置生成，航向指向空域内部
- 生成后调用 wrapper.create_aircraft()，加入 self.agents
- 使用 DynamicEntryConfig 控制进入时机和数量
- 在构建观测时包含新进入的飞机

验收：
- [x] T-V05 所有用例通过

---

### G-V01: 集成测试 — 真实 BlueSky 验证

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_bluesky_real.py` |
| 依赖 | I-V01, I-V02, I-V03, I-V04, I-V05 |
| 说明 | 使用真实 BlueSky 运行完整的 reset/step/close 循环 |

测试用例：
1. `test_real_bluesky_reset` — 使用真实 BlueSky 完成 reset
2. `test_real_bluesky_step` — 使用真实 BlueSky 完成 step
3. `test_real_bluesky_state_consistency` — 飞机状态读取与 BlueSky 内部一致
4. `test_real_bluesky_close` — 使用真实 BlueSky 完成 close
5. `test_real_bluesky_full_episode` — 使用真实 BlueSky 运行完整 episode

验收：
- [x] 所有用例通过（5/5）

---

## Phase 2: 场景系统基础

> TDD 循环：配置模型 → 场景基类 → env 集成

### T-V06: 测试 — 场景配置数据模型

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_scenario_config.py` |
| 依赖 | 无 |
| 说明 | 测试场景配置数据模型的创建和验证 |

测试用例：
1. `test_scenario_config_creation` — 创建 ScenarioConfig 实例
2. `test_airspace_config_rectangular` — 矩形空域配置
3. `test_airspace_config_polygon` — 多边形空域配置
4. `test_sector_config_creation` — 创建 SectorConfig 实例
5. `test_waypoint_config_creation` — 创建 WaypointConfig 实例
6. `test_aircraft_config_creation` — 创建 AircraftConfig 实例
7. `test_dynamic_entry_config_creation` — 创建 DynamicEntryConfig 实例
8. `test_conflict_config_creation` — 创建 ConflictConfig 实例
9. `test_simulation_config_creation` — 创建 SimulationConfig 实例
10. `test_config_from_yaml` — 从 YAML 文件加载配置

验收：
- [x] 所有用例通过（13/13）

---

### I-V06: 实现 — 场景配置数据模型

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/utils/types.py` |
| 依赖 | T-V06 |
| 说明 | 定义场景配置相关的数据模型 |

实现要点：
- 新增 ScenarioConfig、AirspaceConfig、SectorConfig、WaypointConfig 类
- 新增 AircraftConfig、SpawnConfig、DynamicEntryConfig 类
- 新增 ConflictConfig、SimulationConfig 类
- 所有类支持 dict-style 访问（与现有 AircraftState 一致）

验收：
- [x] T-V06 所有用例通过

---

### T-V07: 测试 — 场景基类

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_scenario_base.py` |
| 依赖 | 无 |
| 说明 | 测试场景基类的接口和生命周期 |

测试用例：
1. `test_base_scenario_is_abstract` — BaseScenario 不能直接实例化
2. `test_setup_returns_agent_ids` — setup() 返回 agent ID 列表
3. `test_get_spawn_config` — get_spawn_config() 返回 SpawnConfig
4. `test_get_conflict_config` — get_conflict_config() 返回 ConflictConfig
5. `test_should_truncate` — should_truncate() 正确判断截断
6. `test_get_waypoint` — get_waypoint() 返回正确航路点
7. `test_update_default_noop` — update() 默认无操作
8. `test_reset_default_noop` — reset() 默认无操作

验收：
- [x] 所有用例通过（11/11）

---

### I-V07: 实现 — 场景基类

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/envs/scenarios/base.py` |
| 依赖 | T-V07, I-V06 |
| 说明 | 实现 BaseScenario 抽象基类 |

实现要点：
- 定义 ABC 基类，包含 5 个抽象方法和 2 个可选方法
- 抽象方法：setup, get_spawn_config, get_conflict_config, should_truncate, get_waypoint
- 可选方法：update（默认返回空列表）, reset（默认无操作）

验收：
- [x] T-V07 所有用例通过

---

### T-V08: 测试 — parallel_env 集成场景系统

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_env.py`（追加） |
| 依赖 | 无 |
| 说明 | 测试主环境支持可选的场景实例 |

测试用例：
1. `test_env_without_scenario` — 无场景时行为与 V1.0 一致
2. `test_env_with_scenario` — 有场景时由场景驱动飞机生成
3. `test_scenario_setup_called` — reset 时调用 scenario.setup()
4. `test_scenario_update_called` — step 时调用 scenario.update()
5. `test_scenario_should_truncate` — step 时调用 scenario.should_truncate()

验收：
- [x] 所有用例通过（5/5）

---

### I-V08: 实现 — parallel_env 集成场景系统 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/envs/parallel_env.py` |
| 依赖 | T-V08, I-V07 |
| 说明 | 主环境支持可选的场景实例 |

实现要点：
- 构造函数新增可选 `scenario: BaseScenario | None = None` 参数
- reset() 中：有场景时调用 scenario.setup()，无场景时使用现有逻辑
- step() 中：有场景时调用 scenario.update() 和 scenario.should_truncate()
- 无场景时行为与 V1.0 完全一致（向后兼容）

验收：
- [x] T-V08 所有用例通过

---

## Phase 3: 具体场景实现

> TDD 循环：水平冲突 → 垂直冲突 → 扇区冲突

### T-V09: 测试 — 水平冲突解脱场景

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_horizontal_cr.py` |
| 依赖 | 无 |
| 说明 | 测试水平冲突解脱场景的完整流程 |

测试用例：
1. `test_horizontal_cr_setup` — 场景初始化成功，飞机数量正确
2. `test_horizontal_cr_conflict_detected` — 对头飞行的飞机被检测为冲突
3. `test_horizontal_cr_no_conflict_safe` — 平行飞行的飞机无冲突
4. `test_horizontal_cr_arrival_termination` — 到达目标后 termination
5. `test_horizontal_cr_action_heading_only` — 动作空间仅包含航向
6. `test_horizontal_cr_full_episode` — 完整 episode 可正常运行

验收：
- [x] 所有用例通过（6/6）

---

### I-V09: 实现 — 水平冲突解脱场景 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/envs/scenarios/horizontal_cr.py` |
| 依赖 | T-V09, I-V07 |
| 参考 | bluesky-gym horizontal_cr_env |
| 说明 | 多架飞机巡航，航向机动避碰，到达目标终止 |

实现要点：
- 继承 BaseScenario
- setup()：在空域内随机生成 N 架飞机，每架分配目标航路点
- 使用 haversine_distance 计算航路点距离（100-150 NM）
- should_truncate()：飞机离开空域时截断
- get_waypoint()：返回分配的航路点
- 动作空间：仅航向调整

验收：
- [x] T-V09 所有用例通过

---

### T-V10: 测试 — 垂直冲突解脱场景

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_vertical_cr.py` |
| 依赖 | 无 |
| 说明 | 测试垂直冲突解脱场景的完整流程 |

测试用例：
1. `test_vertical_cr_setup` — 场景初始化成功
2. `test_vertical_cr_conflict_both` — 水平+垂直同时违反 = 冲突
3. `test_vertical_cr_horizontal_only` — 仅水平违反 ≠ 冲突
4. `test_vertical_cr_vertical_only` — 仅垂直违反 ≠ 冲突
5. `test_vertical_cr_action_vs_only` — 动作空间仅包含升降速率
6. `test_vertical_cr_full_episode` — 完整 episode 可正常运行

验收：
- [x] 所有用例通过（6/6）

---

### I-V10: 实现 — 垂直冲突解脱场景 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/envs/scenarios/vertical_cr.py` |
| 依赖 | T-V10, I-V07 |
| 参考 | bluesky-gym vertical_cr_env |
| 说明 | 多架飞机在不同高度层，升降机动避碰 |

实现要点：
- 继承 BaseScenario
- setup()：在相近水平位置、不同高度生成飞机
- 冲突检测：水平 < 5 NM 且垂直 < 1000 ft 同时满足
- 动作空间：仅升降速率调整
- 使用 ALT 命令设置目标高度

验收：
- [x] T-V10 所有用例通过

---

### T-V11: 测试 — 扇区冲突管理场景

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_sector_cr.py` |
| 依赖 | 无 |
| 说明 | 测试扇区冲突管理场景的完整流程 |

测试用例：
1. `test_sector_cr_setup` — 场景初始化成功
2. `test_sector_cr_polygon_boundary` — 扇区边界为多边形
3. `test_sector_cr_exit_truncation` — 飞机离开扇区时截断（非终止）
4. `test_sector_cr_conflict_detection` — 冲突检测正确
5. `test_sector_cr_action_hdg_spd` — 动作空间包含航向+速度
6. `test_sector_cr_density_configurable` — 飞机密度可配置
7. `test_sector_cr_full_episode` — 完整 episode 可正常运行

验收：
- [x] 所有用例通过（7/7）

---

### I-V11: 实现 — 扇区冲突管理场景 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/envs/scenarios/sector_cr.py` |
| 依赖 | T-V11, I-V07 |
| 参考 | bluesky-gym sector_cr_env |
| 说明 | 多边形扇区内多架飞机，航向+速度机动 |

实现要点：
- 继承 BaseScenario
- setup()：在多边形扇区内按密度生成飞机
- 使用 BlueSky areafilter 定义多边形扇区
- should_truncate()：使用 areafilter.checkInside 判断是否在扇区内
- 动作空间：航向 + 速度

验收：
- [x] T-V11 所有用例通过

---

## Phase 4: 增强冲突检测与流量管理

> TDD 循环：冲突增强 → 延误惩罚 → 容量惩罚 → 航路点导航 → 进近汇合

### T-V12: 测试 — 冲突检测增强 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_reward_conflict.py`（追加） |
| 依赖 | 无 |
| 说明 | 测试预测性冲突检测和多机冲突关联 |

测试用例：
1. `test_predictive_conflict_converging` — 相向飞行预测未来冲突
2. `test_predictive_conflict_diverging` — 背向飞行预测无冲突
3. `test_multi_aircraft_chain_conflict` — 三机连锁冲突检测
4. `test_conflict_configurable_distance` — 冲突距离可配置

验收：
- [x] 所有用例通过（4/4）

---

### I-V12: 实现 — 冲突检测增强 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/rewards/components/conflict.py` |
| 依赖 | T-V12 |
| 说明 | 支持预测性冲突检测、多机冲突关联 |

实现要点：
- 基于速度和航向预测未来 N 秒内的最近距离
- 识别涉及多架飞机的连锁冲突
- 冲突距离从配置读取

验收：
- [x] T-V12 所有用例通过

---

### T-V13: 测试 — 延误惩罚组件 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_reward_delay.py` |
| 依赖 | 无 |
| 说明 | 测试延误惩罚的计算 |

测试用例：
1. `test_delay_penalty_per_step` — 每步给予固定惩罚
2. `test_delay_penalty_configurable` — 惩罚值可配置
3. `test_delay_penalty_weight` — 权重正确生效
4. `test_delay_penalty_reset` — reset 后状态清零

验收：
- [x] 所有用例通过（4/4）

---

### I-V13: 实现 — 延误惩罚组件 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/rewards/components/delay.py` |
| 依赖 | T-V13, 现有 RewardComponent 基类 |
| 说明 | 每步给予固定惩罚，鼓励尽快到达 |

实现要点：
- 继承 RewardComponent
- compute() 返回固定惩罚值（默认 -0.01）
- 惩罚值从配置读取

验收：
- [x] T-V13 所有用例通过

---

### T-V14: 测试 — 容量违反惩罚组件 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_reward_capacity.py` |
| 依赖 | 无 |
| 说明 | 测试扇区容量违反惩罚 |

测试用例：
1. `test_capacity_no_penalty_under_limit` — 未超限时无惩罚
2. `test_capacity_penalty_over_limit` — 超限时给予惩罚
3. `test_capacity_penalty_proportional` — 惩罚与超限数量成正比
4. `test_capacity_threshold_configurable` — 容量阈值可配置
5. `test_capacity_penalty_reset` — reset 后状态清零

验收：
- [x] 所有用例通过（5/5）

---

### I-V14: 实现 — 容量违反惩罚组件 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/rewards/components/capacity.py` |
| 依赖 | T-V14, 现有 RewardComponent 基类 |
| 说明 | 扇区内飞机数超过容量上限时给予惩罚 |

实现要点：
- 继承 RewardComponent
- compute() 检查 all_states 中同一扇区的飞机数
- 超限时返回惩罚值（与超限数量成正比）
- 容量阈值从配置读取

验收：
- [x] T-V14 所有用例通过

---

### T-V15: 测试 — 航路点导航场景 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_waypoint_nav.py` |
| 依赖 | 无 |
| 说明 | 测试航路点导航场景（无冲突基线） |

测试用例：
1. `test_waypoint_nav_setup` — 场景初始化成功
2. `test_waypoint_nav_no_conflict` — 无冲突场景
3. `test_waypoint_nav_arrival` — 到达目标后终止
4. `test_waypoint_nav_guidance` — 飞机朝目标航路点飞行
5. `test_waypoint_nav_full_episode` — 完整 episode 可正常运行

验收：
- [x] 所有用例通过（5/5）

---

### I-V15: 实现 — 航路点导航场景 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/envs/scenarios/waypoint_nav.py` |
| 依赖 | T-V15, I-V07 |
| 参考 | bluesky-gym plan_waypoint_env |
| 说明 | 无冲突的纯导航任务 |

实现要点：
- 继承 BaseScenario
- setup()：生成飞机并分配航路点
- 飞机间距离足够远，不会冲突
- 用于测试制导逻辑和到达终止

验收：
- [x] T-V15 所有用例通过

---

### T-V16: 测试 — 进近汇合场景 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_merge.py` |
| 依赖 | 无 |
| 说明 | 测试进近汇合场景 |

测试用例：
1. `test_merge_setup` — 场景初始化成功（20 架飞机）
2. `test_merge_background_traffic_uncontrollable` — 背景交通不可控
3. `test_merge_conflict_distance_4nm` — 汇合冲突距离 4 NM
4. `test_merge_observable_neighbors` — 可观测邻居数限制为 5
5. `test_merge_arrival_at_faf` — 到达 FAF 后终止
6. `test_merge_full_episode` — 完整 episode 可正常运行

验收：
- [x] 所有用例通过（6/6）

---

### I-V16: 实现 — 进近汇合场景 ✅

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/envs/scenarios/merge.py` |
| 依赖 | T-V16, I-V07 |
| 参考 | bluesky-gym merge_env |
| 说明 | 飞机进近降落，与背景交通保持间隔 |

实现要点：
- 继承 BaseScenario
- setup()：生成 1 架主控飞机 + 19 架背景交通
- 背景交通按预设航路飞行（不可控）
- 主控飞机需要找到安全间隙汇入
- 冲突距离：4 NM（比巡航更严格）
- 可观测邻居数：5 架

验收：
- [x] T-V16 所有用例通过

---

## Phase 5: 测试与集成

> 为所有新增功能编写集成测试

### G-V02: 集成测试 — V1.0 向后兼容

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_backward_compat.py` |
| 依赖 | Phase 1 全部 |
| 说明 | 验证 V2.0 改动不影响 V1.0 已有功能 |

测试用例：
1. `test_v1_env_still_works` — 无场景时环境行为不变
2. `test_v1_observation_compatible` — 观测空间增强后旧代码可适配
3. `test_v1_reward_compatible` — 奖励计算不受影响
4. `test_v1_api_compliance` — PettingZoo API 合规测试仍通过

验收：
- [x] 所有用例通过

---

### G-V03: 集成测试 — 场景端到端

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_scenario_e2e.py` |
| 依赖 | Phase 3 全部 |
| 说明 | 所有场景的端到端集成测试 |

测试用例：
1. `test_horizontal_cr_e2e` — 水平冲突场景完整 episode
2. `test_vertical_cr_e2e` — 垂直冲突场景完整 episode
3. `test_sector_cr_e2e` — 扇区冲突场景完整 episode
4. `test_waypoint_nav_e2e` — 航路点导航场景完整 episode
5. `test_merge_e2e` — 进近汇合场景完整 episode

验收：
- [x] 所有用例通过

---

### G-V04: 集成测试 — 性能基准

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_performance.py` |
| 依赖 | Phase 1-4 全部 |
| 说明 | 建立性能基准 |

测试用例：
1. `test_step_time_5_aircraft` — 5 架飞机单步 < 100ms
2. `test_step_time_20_aircraft` — 20 架飞机单步 < 200ms
3. `test_reset_time` — reset < 500ms
4. `test_episode_memory` — 单 episode 内存 < 1GB

验收：
- [x] 所有用例通过

---

## 任务依赖图

```
Phase 1（V1.0 补齐）:
  T-V01 ──→ I-V01 ──┐
  T-V02 ──→ I-V02 ──┤
  T-V03 ──→ I-V03 ──┼──→ G-V01
  T-V04 ──→ I-V04 ──┤
  T-V05 ──→ I-V05 ──┘

Phase 2（场景基础）:
  T-V06 ──→ I-V06 ──→ T-V07 ──→ I-V07 ──→ T-V08 ──→ I-V08

Phase 3（具体场景）:
  I-V07 ──→ T-V09 ──→ I-V09
  I-V07 ──→ T-V10 ──→ I-V10
  I-V07 ──→ T-V11 ──→ I-V11

Phase 4（增强功能）:
  T-V12 ──→ I-V12
  T-V13 ──→ I-V13
  T-V14 ──→ I-V14
  I-V07 ──→ T-V15 ──→ I-V15
  I-V07 ──→ T-V16 ──→ I-V16

Phase 5（集成）:
  G-V02  (依赖 Phase 1)
  G-V03  (依赖 Phase 3)
  G-V04  (依赖 Phase 1-4)
```

---

## 并行执行策略

| 并行组 | 任务 |
|--------|------|
| Group A | T-V01, T-V02, T-V03, T-V05, T-V06 |
| Group B | I-V01, I-V02, I-V03, I-V05 |
| Group C | T-V09, T-V10, T-V11, T-V12, T-V13, T-V14, T-V15, T-V16 |
| Group D | I-V09, I-V10, I-V11, I-V12, I-V13, I-V14, I-V15, I-V16 |

---

## 统计

| 类别 | 数量 |
|------|------|
| 测试任务 | 16 |
| 实现任务 | 16 |
| 集成测试 | 4 |
| **总计** | **36** |
