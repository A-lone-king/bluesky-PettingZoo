# bluesky-marl 原子任务列表

> 基于 spec.md v1.0 MVP 和 plan.md v1.0
> 规则：每个任务只改一个文件，奇数任务写测试，偶数任务写实现
> 编号规则：`S-XX` 脚手架，`T-XX` 测试，`I-XX` 实现，`G-XX` 集成测试

---

## Phase 0: 项目脚手架

> 不遵循 TDD，纯搭建。后续所有模块的 `__init__.py` 在该模块首个任务中一并创建。

### S-01: 项目配置

| 字段 | 值 |
|------|-----|
| 文件 | `pyproject.toml` |
| 依赖 | 无 |
| 说明 | 项目元数据、构建配置、工具配置（ruff/mypy/pytest） |

验收：
- [x] `pip install -e .` 成功
- [x] `ruff check --help` 可用
- [x] `mypy --help` 可用

---

### S-02: 依赖清单

| 字段 | 值 |
|------|-----|
| 文件 | `requirements.txt` |
| 依赖 | S-01 |
| 说明 | 生产依赖：pettingzoo, gymnasium, numpy, pyyaml |

验收：
- [x] `pip install -r requirements.txt` 成功

---

### S-03: 开发依赖

| 字段 | 值 |
|------|-----|
| 文件 | `requirements-dev.txt` |
| 依赖 | S-02 |
| 说明 | 开发依赖：pytest, pytest-cov, ruff, mypy |

验收：
- [x] `pip install -r requirements-dev.txt` 成功

---

### S-04: 默认配置

| 字段 | 值 |
|------|-----|
| 文件 | `config/default.yaml` |
| 依赖 | 无 |
| 说明 | 仿真参数、空域定义、观测参数、动作空间参数、归一化参数 |

验收：
- [x] YAML 格式合法（`python -c "import yaml; yaml.safe_load(open('config/default.yaml'))"`）

---

### S-05: 奖励配置

| 字段 | 值 |
|------|-----|
| 文件 | `config/rewards.yaml` |
| 依赖 | 无 |
| 说明 | 奖励组件权重和阈值参数 |

验收：
- [x] YAML 格式合法

---

### S-06: 类型定义

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/utils/types.py` |
| 依赖 | 无 |
| 说明 | AgentID, AircraftState, DiscreteAction, ContinuousAction, TextualState, AirspaceSnapshot, ConflictLevel 等类型定义 |

验收：
- [x] `mypy src/bluesky_pettingzoo/utils/types.py` 无错误

---

### S-07: 测试 fixtures

| 字段 | 值 |
|------|-----|
| 文件 | `tests/conftest.py` |
| 依赖 | S-04, S-06 |
| 说明 | 共享 fixtures：默认配置加载、模拟飞机状态生成器、动作空间实例 |

验收：
- [x] `pytest tests/ --collect-only` 可发现 fixtures

---

### S-08: 奖励组件基类

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/rewards/base.py` |
| 依赖 | S-06 |
| 说明 | RewardComponent 抽象基类，定义 `compute()` 和 `reset()` 接口 |

验收：
- [x] mypy 无错误
- [x] 无法直接实例化（ABC）

---

### S-09: 工具函数 — 几何计算

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/utils/geometry.py` |
| 依赖 | 无 |
| 说明 | `haversine_distance()`、`bearing()`、`relative_position()` 等纯函数 |

验收：
- [x] mypy 无错误

---

## Phase 1: BlueSky 封装

> TDD 循环 #1：wrapper

### T-01: 测试 — BlueSky Wrapper

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_bluesky_wrapper.py` |
| 依赖 | S-07 |
| 优先级 | P0 |
| 说明 | 测试 BlueSky 无头模式封装的所有公开方法 |

测试用例：
1. `test_init_simulation` — 无头模式初始化成功
2. `test_create_aircraft` — 创建飞机后可在状态中找到
3. `test_remove_aircraft` — 移除飞机后状态中消失
4. `test_send_command` — 发送 HDG/ALT/SPD 命令不报错
5. `test_send_commands_batch` — 批量命令全部执行
6. `test_get_aircraft_state` — 返回完整状态字典
7. `test_get_all_aircraft_states` — 返回所有飞机状态
8. `test_get_active_aircraft_ids` — 返回活跃 ID 列表
9. `test_is_aircraft_in_airspace` — 空域内外判断正确
10. `test_step_advances_time` — step 后仿真时钟推进 dt
11. `test_reset_clears_state` — reset 后无飞机

验收：
- [x] 所有用例 `xfail` 或 `skip`（实现前）
- [x] `pytest tests/test_bluesky_wrapper.py` 可运行

---

### I-01: 实现 — BlueSky Wrapper

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/bluesky/wrapper.py` |
| 依赖 | T-01, S-06 |
| 优先级 | P0 |
| 说明 | BlueSky 无头模式 API 封装，内存级变量读写，无网络 I/O |

实现要点：
- `bs.init(mode='sim', detached=True)` 初始化
- 通过 `bs.traf` 直接读写飞机状态
- `bs.stack.stack()` 发送命令
- `bs.sim.step()` 推进仿真
- 批量命令用 `for cmd in commands: bs.stack.stack(cmd)`

验收：
- [x] T-01 所有用例通过
- [x] 单步执行时间 < 50ms

---

## Phase 2: 观测系统

> TDD 循环 #2-4：normalizer → filters → manager

### T-02: 测试 — 观测归一化器

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_normalizer.py` |
| 依赖 | S-07 |
| 优先级 | P0 |
| 说明 | 测试数值归一化到 [-1, 1] 的正确性 |

测试用例：
1. `test_normalize_heading_0` — 航向 0° → -1.0
2. `test_normalize_heading_180` — 航向 180° → 0.0
3. `test_normalize_heading_360` — 航向 360° → 1.0
4. `test_normalize_altitude_low` — 低高度 → 负值
5. `test_normalize_altitude_mid` — 中间高度 → 0.0
6. `test_normalize_altitude_high` — 高高度 → 正值
7. `test_normalize_speed_range` — 速度范围边界
8. `test_normalize_distance_zero` — 距离 0 → 0.0
9. `test_normalize_distance_max` — 最大距离 → 1.0
10. `test_normalize_bearing` — 方位角归一化
11. `test_output_clipping` — 超范围值被裁剪到 [-1, 1]
12. `test_normalize_aircraft_state` — 完整飞机状态归一化
13. `test_normalize_relative_position` — 相对位置归一化

验收：
- [x] 所有用例通过（13/13）

---

### I-02: 实现 — 观测归一化器

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/observations/normalizer.py` |
| 依赖 | T-02, S-06 |
| 优先级 | P0 |
| 说明 | 将原始观测值归一化到 [-1, 1]，支持航向/高度/速度/距离/方位 |

实现要点：
- 使用配置中的 mid/range 参数
- 公式：`(value - mid) / range`
- 输出用 `np.clip` 裁剪到 [-1, 1]
- 所有函数纯函数，无副作用

验收：
- [x] T-02 所有用例通过（13/13）

---

### T-03: 测试 — 感知范围过滤器

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_filters.py` |
| 依赖 | S-07, S-09 |
| 优先级 | P0 |
| 说明 | 测试感知范围（FOV）过滤逻辑 |

测试用例：
1. `test_filter_no_aircraft` — 无其他飞机时返回空列表
2. `test_filter_within_radius` — 水平距离 10NM（<20NM）→ 可观测
3. `test_filter_outside_radius` — 水平距离 30NM（>20NM）→ 不可观测
4. `test_filter_at_boundary` — 水平距离恰好 20NM → 可观测
5. `test_filter_within_alt_range` — 高度差 2000ft（<3000ft）→ 可观测
6. `test_filter_outside_alt_range` — 高度差 5000ft（>3000ft）→ 不可观测
7. `test_filter_at_alt_boundary` — 高度差恰好 3000ft → 可观测
8. `test_filter_combined` — 水平在范围内但垂直超范围 → 不可观测
9. `test_filter_max_observable` — 超过最大可观测数时截断
10. `test_filter_sorted_by_distance` — 返回结果按距离升序排列

验收：
- [x] 所有用例通过（12/12）

---

### I-03: 实现 — 感知范围过滤器

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/observations/filters.py` |
| 依赖 | T-03, S-09 |
| 优先级 | P0 |
| 说明 | 根据水平半径和垂直范围过滤可观测飞机 |

实现要点：
- 使用 `haversine_distance` 计算水平距离
- 双阈值过滤：水平半径 + 垂直范围
- 按距离排序后截断到 MAX_OBSERVABLE
- 返回过滤后的飞机状态列表

验收：
- [x] T-03 所有用例通过（12/12）

---

### T-04: 测试 — 观测管理器

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_observation_manager.py` |
| 依赖 | S-07 |
| 优先级 | P0 |
| 说明 | 测试观测管理器的完整观测生成流程 |

测试用例：
1. `test_observation_space_shape` — 返回的观测符合 Dict 空间定义
2. `test_self_state_fields` — self_state 包含 6 个归一化值
3. `test_other_aircraft_fields` — other_aircraft 每行 7 个值
4. `test_other_aircraft_mask` — 掩码正确标识有效/填充位置
5. `test_goal_fields` — goal 包含 4 个归一化值
6. `test_padding_with_mask` — 不足 MAX_OBS 时用零填充，掩码为 0
7. `test_full_observable` — 达到 MAX_OBS 时掩码全为 1
8. `test_textual_state_structure` — textual_state 包含所有必要字段
9. `test_textual_state_text_content` — 生成的文本包含关键信息
10. `test_textual_state_conflict_status` — 冲突状态正确标记
11. `test_airspace_snapshot_structure` — 空域快照结构正确
12. `test_observation_consistency` — 连续两次调用结果一致（确定性）

验收：
- [x] 所有用例通过（18/18）

---

### I-04: 实现 — 观测管理器

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/observations/manager.py` |
| 依赖 | T-04, I-02, I-03 |
| 优先级 | P0 |
| 说明 | 整合归一化器和过滤器，生成完整观测 |

实现要点：
- 调用 filters 过滤可观测飞机
- 调用 normalizer 归一化所有数值
- 构造 Dict 格式观测（self_state + other_aircraft + mask + goal）
- 生成 textual_state 和 airspace_snapshot

验收：
- [x] T-04 所有用例通过（18/18）

---

## Phase 3: 动作系统

> TDD 循环 #5：translator

### T-05: 测试 — 动作翻译器

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_actions.py` |
| 依赖 | S-07 |
| 优先级 | P0 |
| 说明 | 测试离散动作到 BlueSky 命令的转换 |

测试用例：
1. `test_no_adjustment` — 索引 [2,2,2]（全零调整）→ 无命令
2. `test_heading_positive` — 航向 +10° → `HDG AC001 {current+10}`
3. `test_heading_negative` — 航向 -20° → `HDG AC001 {current-20}`
4. `test_heading_wraparound` — 航向 350° +20° → 10°
5. `test_altitude_up` — 高度 +1000ft → `ALT AC001 {current+1000}`
6. `test_altitude_down` — 高度 -2000ft → `ALT AC001 {current-2000}`
7. `test_speed_up` — 速度 +10kt → `SPD AC001 {current+10}`
8. `test_speed_down` — 速度 -20kt → `SPD AC001 {current-20}`
9. `test_combined_action` — 同时调整航向+高度+速度 → 3 条命令
10. `test_translate_batch` — 批量翻译多个 Agent → 所有命令合并
11. `test_command_format` — 命令格式符合 BlueSky 规范

验收：
- [x] 所有用例通过（11/11）

---

### I-05: 实现 — 动作翻译器

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/actions/translator.py` |
| 依赖 | T-05, S-06 |
| 优先级 | P0 |
| 说明 | 将 MultiDiscrete 离散动作转换为 BlueSky 文本命令 |

实现要点：
- 从配置读取离散化选项数组
- 索引查表得到调整量
- 与当前状态相加得到目标值
- 航向取模 360，高度/速度保持合理范围
- 零调整不生成命令

验收：
- [x] T-05 所有用例通过（11/11）

---

## Phase 4: 奖励系统

> TDD 循环 #6-9：conflict → smoothness → efficiency → calculator

### T-06: 测试 — 冲突惩罚组件

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_reward_conflict.py` |
| 依赖 | S-07 |
| 优先级 | P0 |
| 说明 | 测试冲突检测和惩罚计算 |

测试用例：
1. `test_no_conflict` — 无冲突时奖励为 0
2. `test_nmac_horizontal_only` — 水平 <5NM 但垂直 >1000ft → 不算 NMAC
3. `test_nmac_vertical_only` — 垂直 <1000ft 但水平 >5NM → 不算 NMAC
4. `test_nmac_both` — 水平 <5NM 且垂直 <1000ft → NMAC 惩罚 -100
5. `test_warning_horizontal_only` — 水平 <10NM 但垂直 >2000ft → 不算预警
6. `test_warning_both` — 水平 <10NM 且垂直 <2000ft → 预警惩罚 -10
7. `test_separation_violation` — 水平 <5NM 或垂直 <1000ft → 间隔违反 -5
8. `test_multiple_conflicts` — 同时与多架飞机冲突 → 取最严重惩罚
9. `test_conflict_at_exact_boundary` — 精确边界值测试
10. `test_conflict_level_enum` — ConflictLevel 枚举值正确

验收：
- [x] 所有用例通过（10/10）

---

### I-06: 实现 — 冲突惩罚组件

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/rewards/components/conflict.py` |
| 依赖 | T-06, S-08, S-09 |
| 优先级 | P0 |
| 说明 | 检测 NMAC、冲突预警、安全间隔违反并计算惩罚 |

实现要点：
- 使用 haversine_distance 计算水平距离
- 计算垂直距离绝对值
- 三级判定：NMAC > 预警 > 间隔违反
- 多冲突取最严重等级

验收：
- [x] T-06 所有用例通过（10/10）

---

### T-07: 测试 — 平稳性惩罚组件

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_reward_smoothness.py` |
| 依赖 | S-07 |
| 优先级 | P1 |
| 说明 | 测试动作频率惩罚 |

测试用例：
1. `test_no_action_penalty` — 零调整动作 → 惩罚 0
2. `test_heading_action_penalty` — 航向调整 → 惩罚 -0.1
3. `test_altitude_action_penalty` — 高度调整 → 惩罚 -0.1
4. `test_speed_action_penalty` — 速度调整 → 惩罚 -0.1
5. `test_combined_action_penalty` — 多维调整 → 惩罚仍为 -0.1（按动作次数）

验收：
- [x] 所有用例通过（5/5）

---

### I-07: 实现 — 平稳性惩罚组件

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/rewards/components/smoothness.py` |
| 依赖 | T-07, S-08 |
| 优先级 | P1 |
| 说明 | 每次发布指令给予固定惩罚 |

实现要点：
- 检查动作是否包含非零调整
- 有调整则返回 `action_penalty`，否则返回 0

验收：
- [x] T-07 所有用例通过（5/5）

---

### T-08: 测试 — 效率奖励组件

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_reward_efficiency.py` |
| 依赖 | S-07 |
| 优先级 | P1 |
| 说明 | 测试航线偏离惩罚和到达奖励 |

测试用例：
1. `test_step_penalty` — 每步都有微小惩罚 -0.01
2. `test_no_deviation` — 在目标航线上无偏离惩罚
3. `test_deviation_penalty` — 偏离航线 10NM → 惩罚按比例计算
4. `test_max_deviation_penalty` — 偏离达到 MAX_DEVIATION → 最大惩罚 -5
5. `test_arrival_reward` — 到达目标航路点 → 奖励 +10
6. `test_not_arrived` — 未到达时无到达奖励
7. `test_deviation_proportional` — 惩罚与偏离距离成正比

验收：
- [x] 所有用例通过（7/7）

---

### I-08: 实现 — 效率奖励组件

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/rewards/components/efficiency.py` |
| 依赖 | T-08, S-08, S-09 |
| 优先级 | P1 |
| 说明 | 航线偏离惩罚 + 到达奖励 + 步数惩罚 |

实现要点：
- 使用 haversine_distance 计算偏离距离
- 偏离惩罚：`-distance / max_deviation * scale`
- 到达判定：距离 < 到达阈值
- 叠加每步固定惩罚

验收：
- [x] T-08 所有用例通过（7/7）

---

### T-09: 测试 — 奖励计算器

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_reward_calculator.py` |
| 依赖 | S-07 |
| 优先级 | P0 |
| 说明 | 测试组件注册和加权求和 |

测试用例：
1. `test_register_component` — 注册组件后 components 列表增长
2. `test_empty_calculator` — 无组件时返回 0
3. `test_single_component` — 单组件返回其值 × 权重
4. `test_multiple_components` — 多组件返回加权和
5. `test_zero_weight` — 权重为 0 时该组件无贡献
6. `test_negative_weight` — 负权重正确计算
7. `test_reset_calls_components` — reset() 调用所有组件的 reset()
8. `test_compute_passes_correct_args` — compute 参数正确传递给组件

验收：
- [x] 所有用例通过（8/8）

---

### I-09: 实现 — 奖励计算器

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/rewards/calculator.py` |
| 依赖 | T-09, S-08 |
| 优先级 | P0 |
| 说明 | 管理奖励组件注册和加权求和 |

实现要点：
- `register(component, weight)` 添加组件
- `compute()` 遍历组件，累加 `weight * component.compute()`
- `reset()` 调用所有组件的 `reset()`

验收：
- [x] T-09 所有用例通过（8/8）

---

## Phase 5: 基线 Agent

> TDD 循环 #10-11：random → rule_based

### T-10: 测试 — RandomAgent

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_random_agent.py` |
| 依赖 | S-07 |
| 优先级 | P0 |
| 说明 | 测试随机 Agent 的行为 |

测试用例：
1. `test_act_returns_dict` — 返回字典类型
2. `test_act_keys_match_agents` — 键与输入 agents 一致
3. `test_action_in_space` — 每个动作在 action_space 内
4. `test_different_observations_different_actions` — 不同观测产生不同随机动作（概率性，多次采样验证）
5. `test_reset_no_error` — reset() 无报错

验收：
- [x] 所有用例通过（5/5）

---

### I-10: 实现 — RandomAgent

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/agents/random_agent.py` |
| 依赖 | T-10 |
| 优先级 | P0 |
| 说明 | 随机采样动作空间的 Agent |

实现要点：
- `act()` 对每个 agent 调用 `action_space.sample()`
- 继承 BaseAgent

验收：
- [x] T-10 所有用例通过（5/5）

---

### T-11: 测试 — RuleBasedAgent

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_rule_based_agent.py` |
| 依赖 | S-07 |
| 优先级 | P0 |
| 说明 | 测试规则 Agent（直飞，不避让） |

测试用例：
1. `test_act_returns_dict` — 返回字典类型
2. `test_act_keys_match_agents` — 键与输入 agents 一致
3. `test_always_no_adjustment` — 动作始终为 [2, 2, 2]（零调整）
4. `test_deterministic` — 相同输入产生相同输出
5. `test_reset_no_error` — reset() 无报错

验收：
- [x] 所有用例通过（5/5）

---

### I-11: 实现 — RuleBasedAgent

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/agents/rule_based_agent.py` |
| 依赖 | T-11 |
| 优先级 | P0 |
| 说明 | 保持当前航向直飞的规则 Agent |

实现要点：
- `act()` 对每个 agent 返回 `[2, 2, 2]`（零调整索引）
- 继承 BaseAgent

验收：
- [x] T-11 所有用例通过（5/5）

---

## Phase 6: 环境核心

> TDD 循环 #12：parallel_env

### T-12: 测试 — 主环境

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_env.py` |
| 依赖 | S-07 |
| 优先级 | P0 |
| 说明 | 测试 BlueSkyMARLEnv 的核心功能 |

测试用例：

**reset 系列：**
1. `test_reset_returns_tuple` — 返回 (observations, infos) 元组
2. `test_reset_observations_keys` — 观测键与 agents 列表一致
3. `test_reset_infos_keys` — infos 键与 agents 列表一致
4. `test_reset_observation_in_space` — 观测值在 observation_space 内
5. `test_reset_agents_populated` — reset 后 agents 非空
6. `test_reset_with_seed` — 相同 seed 产生相同初始状态
7. `test_reset_clears_previous_state` — 二次 reset 清除旧状态

**step 系列：**
8. `test_step_returns_five_tuple` — 返回 (obs, rewards, terms, truncs, infos)
9. `test_step_rewards_keys` — rewards 键与 agents 一致
10. `test_step_terminations_keys` — terminations 键与 agents 一致
11. `test_step_truncations_keys` — truncations 键与 agents 一致
12. `test_step_observation_in_space` — step 后观测在空间内
13. `test_step_agents_update` — step 后 agents 列表可能变化

**空间系列：**
14. `test_observation_space_type` — 返回 Dict 空间
15. `test_action_space_type` — 返回 MultiDiscrete 空间
16. `test_action_space_sample_valid` — sample() 产生的动作合法

**生命周期系列：**
17. `test_episode_ends_on_max_steps` — 超过最大步数时 truncations 全 True
18. `test_agent_removal_on_exit` — 飞机离开空域后从 agents 中移除
19. `test_infos_has_textual_state` — infos 包含 textual_state
20. `test_infos_has_airspace_snapshot` — infos 包含 airspace_snapshot

验收：
- [x] 所有用例通过（20/20）

---

### I-12: 实现 — 主环境

| 字段 | 值 |
|------|-----|
| 文件 | `src/bluesky_pettingzoo/envs/parallel_env.py` |
| 依赖 | T-12, I-01, I-04, I-05, I-09 |
| 优先级 | P0 |
| 说明 | 继承 PettingZoo ParallelEnv 的主环境类 |

实现要点：
- 组合 BlueSkyWrapper + ObservationManager + ActionTranslator + RewardCalculator
- `reset()`：重置 BlueSky、生成飞机、返回观测
- `step()`：批量发送命令 → 推进仿真 → 读取状态 → 计算观测/奖励/终止
- 动态更新 `self.agents` 列表
- infos 包含双轨输出

验收：
- [x] T-12 所有用例通过（20/20）

---

## Phase 7: PettingZoo 合规与集成测试

### T-13: 测试 — PettingZoo API 合规

| 字段 | 值 |
|------|-----|
| 文件 | `tests/test_api_compliance.py` |
| 依赖 | I-12 |
| 优先级 | P0 |
| 说明 | 使用 PettingZoo 官方测试套件验证环境合规性 |

测试用例：
1. `test_parallel_api` — 调用 `parallel_api_test(env, num_cycles=100)`
2. `test_render_modes` — 验证 render_mode 支持
3. `test_close` — 验证 close() 无资源泄漏

验收：
- [x] `parallel_api_test` 全部通过（6/6）

---

### T-14: 集成测试 — 无冲突场景

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_no_conflict.py` |
| 依赖 | I-12, I-10, I-11 |
| 优先级 | P0 |
| 说明 | 所有飞机直飞无冲突的完整 episode 测试 |

测试用例：
1. `test_rule_based_agent_full_episode` — RuleBasedAgent 跑完整 episode
2. `test_random_agent_runs_100_steps` — RandomAgent 运行 100 步不报错
3. `test_no_nmac_in_safe_scenario` — 安全场景无 NMAC
4. `test_rewards_bounded` — 奖励值在合理范围内

验收：
- [x] 所有用例通过（4/4）

---

### T-15: 集成测试 — 单冲突场景

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_single_conflict.py` |
| 依赖 | I-12 |
| 优先级 | P0 |
| 说明 | 两架飞机对头冲突的场景测试 |

测试用例：
1. `test_conflict_detected` — 冲突被正确检测
2. `test_nmac_triggers_termination` — NMAC 触发终止
3. `test_conflict_penalty_applied` — 冲突惩罚正确应用
4. `test_reward_component_weights` — 各组件权重正确生效

验收：
- [x] 所有用例通过（4/4）

---

### T-16: 集成测试 — 多冲突场景

| 字段 | 值 |
|------|-----|
| 文件 | `tests/integration/test_multi_conflict.py` |
| 依赖 | I-12 |
| 优先级 | P1 |
| 说明 | 5 架飞机交叉冲突的场景测试 |

测试用例：
1. `test_multi_conflict_detection` — 多对冲突同时检测
2. `test_agent_lifecycle` — 飞机进入/离开空域的完整生命周期
3. `test_infos_completeness` — infos 包含所有必要字段
4. `test_episode_completion` — episode 可正常完成

验收：
- [x] 所有用例通过（5/5）

---

## 任务依赖图

```
S-01 ─→ S-02 ─→ S-03
S-04    S-05    S-06 ─→ S-07 ─→ S-08
        S-09

Phase 1:  T-01 ──→ I-01
Phase 2:  T-02 ──→ I-02
          T-03 ──→ I-03
          T-04 ──→ I-04  (依赖 I-02, I-03)
Phase 3:  T-05 ──→ I-05
Phase 4:  T-06 ──→ I-06
          T-07 ──→ I-07
          T-08 ──→ I-08
          T-09 ──→ I-09
Phase 5:  T-10 ──→ I-10
          T-11 ──→ I-11
Phase 6:  T-12 ──→ I-12  (依赖 I-01, I-04, I-05, I-09)
Phase 7:  T-13  (依赖 I-12)
          T-14  (依赖 I-12, I-10, I-11)
          T-15  (依赖 I-12)
          T-16  (依赖 I-12)
```

---

## 并行执行策略

以下任务组内可并行开发（无依赖）：

| 并行组 | 任务 |
|--------|------|
| Group A | S-04, S-05, S-06, S-09 |
| Group B | T-01, T-02, T-03, T-05, T-06, T-07, T-08, T-09, T-10, T-11 |
| Group C | I-02, I-03, I-05, I-06, I-07, I-08, I-10, I-11 |
| Group D | T-14, T-15, T-16 |

---

## 统计

| 类别 | 数量 |
|------|------|
| 脚手架任务 | 9 |
| 测试任务 | 16 |
| 实现任务 | 12 |
| **总计** | **37** |
| 测试文件 | 16 |
| 实现文件 | 12 |
| 配置文件 | 5 |
