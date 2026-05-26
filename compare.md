# bluesky-PettingZoo vs bluesky-gym 功能对比

本文档逐模块对比两个项目的实现细节。数据来源为源码直接阅读。

---

## 1. 项目架构

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 定位 | 单智能体 RL 环境集合 | 多智能体 RL 环境（PettingZoo ParallelEnv） |
| 环境数量 | 7 个独立 `gym.Env` 类 | 10 个 `BaseScenario` 子类，共享 1 个 `BlueSkyMARLEnv` |
| 代码量 | ~2000 行（7 个 env 文件 + main.py） | ~5000+ 行（51 个源文件） |
| 测试 | 无 | 1026 个测试用例 |
| 配置 | 全部硬编码 | YAML 外部配置 |
| 封装 | 直接耦合 BlueSky 内部 API | `BlueSkyWrapper` 抽象层 |

---

## 2. 环境/场景对比

### 2.1 HorizontalCR — 水平冲突解脱

| 维度 | bluesky-gym `HorizontalCREnv` | bluesky-PettingZoo `HorizontalCRScenario` |
|------|-------------------------------|------------------------------------------|
| 智能体数 | 1（ownship + 5 intruders） | 5（全部可控） |
| 动作空间 | `Box(-1,1, shape=(1,))` 连续，仅航向 | `MultiDiscrete([5,5,5])` 离散，航向+高度+速度 |
| 动作缩放 | `action * 45°` | `[-20,-10,0,10,20]°` / `[-2000,-1000,0,1000,2000]ft` / `[-20,-10,0,10,20]kts` |
| 仿真子步 | 10 步/RL step (dt=5s) | `action_frequency=10` 步/RL step (dt=5s) |
| 冲突生成 | `bs.traf.creconfs()` 原生命令，生成真实冲突轨迹 | `BlueSkyWrapper.create_conflict_aircraft()` 封装 `creconfs` |
| 入侵距离 | 5 NM（仅水平） | 5 NM 水平 + 1000 ft 垂直（NMAC） |
| 观测：自身 | 无自身状态观测 | `self_state` 9 维（航向sin/cos、高度、速度、经纬度、VS、地速、优先级） |
| 观测：入侵者 | 固定 5 个：距离、bearing sin/cos、相对速度 x/y | 最多 10 个（带 mask）：航向、高度、速度、距离、bearing sin/cos、相对高度、相对速度 x/y、优先级 |
| 观测：目标 | waypoint 距离 + drift sin/cos | `goal` 4 维（距离、bearing sin/cos、高度差） |
| 感知过滤 | 无（始终观测全部 5 个入侵者） | 可配置半径（20 NM）、高度差（3000 ft）、最大观测数（10） |
| 归一化 | 硬编码除法常量 | `Normalizer` 类，mid/range 归一化到 [-1,1] |
| 奖励：到达 | +1（waypoint 距离 < 5 km） | +10（`EfficiencyReward`，到达阈值 2 NM） |
| 奖励：偏航 | `-0.1 * abs(偏航角弧度)` | `-0.1 * 动作调整量`（`SmoothnessPenalty`） |
| 奖励：入侵 | `-1 * 入侵入侵者数` | `-100`（NMAC）/ `-10`（Warning）/ `-5`（Separation），三级 |
| 奖励：步惩罚 | 无 | `-0.01`（`EfficiencyReward.step_penalty`） |
| 预测冲突检测 | 无 | 有，向前投影位置 |
| 链式冲突检测 | 无 | 有，BFS 图搜索 |
| 终止条件 | 所有 waypoint 到达 | NMAC / 到达 / 最大步数 |
| 最大步数 | 无显式限制 | 360 步（30 分钟） |
| 渲染 | Pygame 512×512 | Pygame 渲染（`HorizontalCRRenderer`） |

### 2.2 VerticalCR — 垂直冲突解脱

| 维度 | bluesky-gym `VerticalCREnv` | bluesky-PettingZoo `VerticalCRScenario` |
|------|----------------------------|----------------------------------------|
| 智能体数 | 1（ownship + 5 intruders） | 5（全部可控） |
| 动作空间 | `Box(-1,1, shape=(1,))` 连续，仅垂直速度 | `MultiDiscrete([5,5,5])` 离散，航向+高度+速度 |
| 动作缩放 | `action * 12.5 m/s`（≈2500 ft/min） | `[-2000,-1000,0,1000,2000]ft` 高度调整 |
| 动作执行 | 直接写 `bs.traf.selalt[0]` 和 `bs.traf.selvs[0]` | 通过 `ALT` 栈命令 |
| 仿真子步 | 30 步/RL step | `action_frequency=10` |
| 冲突生成 | `bs.traf.creconfs()` + 手动设置高度和 `dH`/`tlosv` | 随机位置 + 交错高度（2000 ft 间距） |
| 入侵检查 | 水平 5 NM **且** 垂直 304.8 m（≈1000 ft） | 水平 5 NM **且** 垂直 1000 ft |
| 观测：自身 | 高度、垂直速度、目标高度、跑道距离（各 1 维） | `self_state` 9 维 |
| 观测：入侵者 | 固定 5 个：距离、bearing、高度差、相对速度 x/y/z | 最多 10 个（带 mask）：完整 10 维特征 |
| 奖励：高度误差 | `abs(目标高度-当前高度) * (-5/3000)` | 通过 `EfficiencyReward` 航路偏离惩罚 |
| 奖坠：入侵 | `-50` / 入侵者 | `-100`（NMAC）/ `-10`（Warning）/ `-5`（Separation） |
| 奖励：坠毁 | `-100`（高度 ≤ 0） | NMAC 触发终止 |
| 奖励：跑道到达 | `高度 * (-50/3000)`（到达时剩余高度越低越好） | +10 到达奖励 |
| 终止条件 | 坠毁或到达跑道 | NMAC / 到达 / 最大步数 |
| VNAV 控制 | `bs.traf.swvnav[0] = False` 禁用 | 不涉及 |

### 2.3 SectorCR — 扇区冲突解脱

| 维度 | bluesky-gym `SectorCREnv` | bluesky-PettingZoo `SectorCRScenario` |
|------|--------------------------|--------------------------------------|
| 智能体数 | 1（ownship + 可变密度入侵者） | 5（全部可控） |
| 动作空间 | `Box(-1,1, shape=(2,))` 连续，航向+速度 | `MultiDiscrete([5,5,5])` 离散，航向+高度+速度 |
| 动作缩放 | `dh * 22.5°`, `dv * (20/3) kts` | `[-20,-10,0,10,20]°`, `[-20,-10,0,10,20]kts` |
| 入侵者数量 | 可变，基于密度（0.003-0.007 AC/NM²） | 固定 5 |
| 观测入侵者数 | 最近 4 个（`NUM_AC_STATE=4`） | 最多 10 个（带 mask） |
| 扇区面积 | 2400-3750 NM²（随机多边形） | 由 `generate_polygon()` 生成 |
| 扇区检测 | `bs.tools.areafilter.checkInside()` | `point_in_polygon()` Python 实现 |
| 坐标系 | 笛卡尔相对坐标（米） | 经纬度 + haversine 距离 |
| 观测：自身 | drift sin/cos + 空速 | `self_state` 9 维 |
| 观测：入侵者 | 相对位置 x/y（米）、相对速度 vx/vy（m/s）、track sin/cos、距离（米） | 完整 10 维特征 |
| 归一化常量 | 距离 ÷13000m, 速度 ÷32/66 m/s, 距离 `(d-50000)/15000` | `Normalizer` 类统一处理 |
| 奖励 | `-0.1*偏航` + `-1*入侵` | 多组件加权：冲突(-100/-10/-5) + 平滑(-0.1) + 效率(-0.01) |
| 终止 | 从不终止（`terminated=False`），飞出扇区截断 | NMAC / 飞出扇区 / 最大步数 |
| 最大步数 | 无限制 | 360 步 |
| 高度交错 | 无（单高度层 350 FL） | 有（31000-39000 ft，2000 ft 间距） |

### 2.4 Descent — 下降阶段

| 维度 | bluesky-gym `DescentEnv` | bluesky-PettingZoo `DescentScenario` |
|------|-------------------------|-------------------------------------|
| 智能体数 | 1（仅 ownship，无入侵者） | 3（全部可控） |
| 动作空间 | `Box(-1,1, shape=(1,))` 连续，仅垂直速度 | `MultiDiscrete([5,5,5])` 离散，高度控制 |
| 动作缩放 | `action * 12.5 m/s` | `[-2000,-1000,0,1000,2000]ft` |
| 观测 | 4 维：高度、VS、目标高度、跑道距离 | `self_state` 9 维 + `goal` 4 维 + 周围飞机 |
| 奖励：高度误差 | `abs(目标-当前) * (-5/3000)` | 通过 `EfficiencyReward` |
| 奖励：坠毁 | `-100` | NMAC 检测 |
| 奖励：跑道到达 | `高度 * (-50/3000)` | +10 到达奖励 |
| 跑道位置 | 固定 (52°N, 4°E) | 场景配置 |
| VNAV | `swvnav[0] = False` | 不涉及 |
| 终止 | 坠毁或到达 | 到达 / NMAC / 最大步数 |

### 2.5 Merge — 进近汇合

| 维度 | bluesky-gym `MergeEnv` | bluesky-PettingZoo `MergeScenario` |
|------|-----------------------|-----------------------------------|
| 智能体数 | 1（ownship + 19 NPC） | 20（1 可控 + 19 背景） |
| 动作空间 | `Box(-1,1, shape=(2,))` 连续，航向+速度 | `MultiDiscrete([5,5,5])` 离散 |
| 动作缩放 | `dh * 15°`, `dv * 20 kts` | `[-20,-10,0,10,20]°` |
| 速度 | 100 kts | 400-500 kts |
| NPC 行为 | 有 FAF waypoint 和跑道目的地，BlueSky 自动导航 | 背景飞机按航路飞行 |
| 冲突解决 | `reso off`（NPC 不做冲突解脱） | NPC 使用 noop 动作 |
| 观测入侵者 | 最近 5 个：笛卡尔相对位置/速度、track sin/cos、距离 | 最多 10 个（带 mask） |
| FAF 逻辑 | 到达 FAF 后切换目标为跑道 | 无 FAF 概念 |
| 奖励 | +1 到达 + `-0.1*偏航` + `-1*入侵` | 多组件加权 |
| 终止 | FAF 到达且接近跑道 | 到达 / NMAC / 最大步数 |
| 跑道 | Schiphol (52.36°N, 4.71°E) | 场景配置 |

### 2.6 StaticObstacle — 静态障碍物

| 维度 | bluesky-gym `StaticObstacleEnv` | bluesky-PettingZoo `StaticObstacleScenario` |
|------|-------------------------------|---------------------------------------------|
| 智能体数 | 1（仅 ownship） | 1（仅 ownship） |
| 动作空间 | `Box(-1,1, shape=(2,))` 连续，航向+速度 | `MultiDiscrete([5,5,5])` 离散，航向+速度 |
| 障碍物数量 | 10 个多边形 | 可配置 |
| 障碍物面积 | 50-1000 NM² | 场景配置 |
| 入侵检测 | `bs.tools.areafilter.checkInside()` | `point_in_polygon()` Python 实现 |
| 观测：障碍物 | 半径、距离、bearing sin/cos（各 10 维） | `obstacles` 位置 + mask |
| 奖励：入侵 | `-5`（多边形内）+ 立即终止 | `-5`（`ObstacleIntrusion`） |
| 奖励：偏航 | `-0.01 * abs(偏航弧度)` | `-0.1 * 动作调整`（`SmoothnessPenalty`） |
| 奖励：到达 | +1 | +10（`EfficiencyReward`） |
| 终止 | 到达或入侵 | 到达 / 入侵 / 最大步数 |
| 特殊逻辑 | step 内子步循环中检查入侵，可中途终止 | 子步中途检查（NMAC/入侵/越界），通过 `on_substep` 回调实现 |

### 2.7 PlanWaypoint — 多航路点导航

| 维度 | bluesky-gym `PlanWaypointEnv` | bluesky-PettingZoo `PlanWaypointScenario` |
|------|-------------------------------|------------------------------------------|
| 智能体数 | 1（仅 ownship） | 1（仅 ownship） |
| 航路点数 | 5 | 5 |
| 动作空间 | `Box(-1,1, shape=(1,))` 仅航向 | `MultiDiscrete([5,5,5])` 航向控制 |
| 观测 | 航路点距离(5)、drift sin/cos(5)、到达状态(5) | `self_state` 9 维 + `goal` 4 维 |
| 奖励 | 仅 +1 到达奖励（无任何惩罚项） | 多组件加权（效率+平滑+偏航） |
| 已到达航路点 | 观测值清零（mask 技巧） | 通过 mask 系统 |
| 终止 | 全部 5 个航路点到达 | 全部到达 / 最大步数 |
| 复杂度 | 最简单（无入侵者、无惩罚） | 最简单（无入侵者、无冲突） |

---

## 3. 观测系统

### 3.1 特征维度对比

| 特征 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 自身航向 | 无（隐含在 drift 中） | `heading_cos`, `heading_sin`（sin/cos 分解） |
| 自身高度 | `altitude`（1 维，部分场景） | `altitude`（1 维，所有场景） |
| 自身速度 | `airspeed`（1 维，部分场景） | `speed`（1 维） |
| 自身经纬度 | 无 | `lat`, `lon` |
| 垂直速度 | `vz`（1 维，仅 Vertical/Descent） | `vs`（1 维） |
| 地速 | 无 | `ground_speed`（1 维） |
| 优先级 | 无 | `priority`（1 维） |
| 入侵者距离 | `intruder_distance` (N,) | 合并在 `other_aircraft` 中 |
| 入侵者 bearing | `cos/sin_difference_pos` (N,) | `bearing_cos`, `bearing_sin` |
| 入侵者相对速度 | `x/y_difference_speed` (N,) | `relative_speed_x`, `relative_speed_y` |
| 入侵者相对高度 | `altitude_difference` (N,)（仅 Vertical） | `relative_altitude`（所有场景） |
| 入侵者航向 | 无 | `heading`（1 维） |
| 入侵者高度 | 无 | `altitude`（1 维） |
| 入侵者速度 | 无 | `speed`（1 维） |
| 入侵者优先级 | 无 | `priority`（1 维） |
| 航路点/目标 | `waypoint_distance` + `drift sin/cos` | `goal` 4 维（距离、bearing sin/cos、高度差） |
| 障碍物 | `restricted_area_radius/distance/bearing`（仅 StaticObstacle） | `obstacles` 位置 + mask |
| mask | 无 | `other_aircraft_mask` (max_obs,) |
| TextualState | 无 | 人类可读状态文本（用于 LLM/RAG） |
| AirspaceSnapshot | 无 | 扇区、航路点、全部飞机位置快照 |

### 3.2 归一化方法

| 方法 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 实现 | 硬编码除法常量（每个环境不同） | `Normalizer` 类，统一 mid/range 归一化 |
| 距离 | ÷150 或 ÷200 或 ÷250（环境不同） | ÷ `max_distance`（默认 20 NM） |
| 速度 | ÷150 或 ÷32/66 或不归一化 | `(v - mid) / range`，mid=450, range=100 |
| 高度 | `(alt - 1500) / 3000` | `(alt - 33000) / 10000` |
| 航向 | sin/cos 分解（已归一化） | sin/cos 分解 + `(angle - 180) / 180` |
| 输出范围 | 不统一 | 统一裁剪到 [-1, 1] |

### 3.3 感知过滤

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 感知半径 | 无（始终观测全部入侵者） | 20 NM（可配置） |
| 高度过滤 | 无 | 3000 ft（可配置） |
| 最大观测数 | 固定（5 或 4，环境决定） | 10（可配置，带 mask） |
| 排序 | 部分场景按距离排序（SectorCR 最近 4） | 按距离排序，最近的优先 |

---

## 4. 动作系统

### 4.1 动作空间

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 类型 | 连续 `Box(-1, 1, shape=(N,))` | 离散 `MultiDiscrete([5,5,5])`（PPO）+ 连续 `Box(-1,1,shape=(3,))`（SAC/TD3/DDPG） |
| 维数 | 1 或 2（环境决定） | 3（航向 + 高度 + 速度） |
| 航向控制 | `action * D_HEADING`（D_HEADING=15°~45°） | `[-20,-10,0,10,20]°` 离散档位 |
| 高度控制 | `action * 12.5 m/s`（仅 VS） | `[-2000,-1000,0,1000,2000]ft` 离散档位 |
| 速度控制 | `action * D_SPEED`（D_SPEED=6.67~20 kts） | `[-20,-10,0,10,20]kts` 离散档位 |
| 算法适配 | 连续动作 → SAC/TD3/DDPG/PPO 均可 | 离散（PPO 默认）+ 连续（SAC/TD3/DDPG 自动切换） |

### 4.2 命令翻译

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 航向命令 | `bs.stack.stack(f"HDG KL001 {heading}")` | `ActionTranslator.translate()` → `HDG AC001 {heading}` |
| 高度命令 | 直接写 `bs.traf.selalt[0]` 和 `bs.traf.selvs[0]` | `ALT AC001 {altitude}` 栈命令 |
| 速度命令 | `bs.stack.stack(f"SPD KL001 {speed_kts}")` | `SPD AC001 {speed}` 栈命令 |
| 动作频率 | 每个环境不同（5/10/30 子步） | 统一 `action_frequency=10` |

---

## 5. 奖励系统

### 5.1 bluesky-gym 奖励（硬编码，无模块化）

| 环境 | 到达奖励 | 偏航惩罚 | 入侵惩罚 | 坠毁惩罚 | 步惩罚 |
|------|---------|---------|---------|---------|--------|
| HorizontalCR | +1 | `-0.1*rad` | `-1*数` | — | — |
| VerticalCR | `高度*(-50/3000)` | — | `-50*数` | `-100` | — |
| SectorCR | — | `-0.1*rad` | `-1*数` | — | — |
| Descent | `高度*(-50/3000)` | — | — | `-100` | — |
| Merge | +1 | `-0.1*rad` | `-1*数` | — | — |
| StaticObstacle | +1 | `-0.01*rad` | `-5`（多边形） | — | — |
| PlanWaypoint | +1/点 | — | — | — | — |

### 5.2 bluesky-PettingZoo 奖励（模块化，YAML 配置）

| 组件 | 文件 | 权重 | 默认参数 |
|------|------|------|---------|
| `ConflictPenalty` | `conflict.py` | 1.0 | NMAC: -100, Warning: -10, Separation: -5 |
| `DriftPenalty` | `drift.py` | 0.5 | scale × |heading - bearing_to_goal| |
| `SmoothnessPenalty` | `smoothness.py` | 0.5 | action_penalty: -0.1 |
| `EfficiencyReward` | `efficiency.py` | 0.3 | arrival: +10, step: -0.01, deviation_scale: 5 |
| `ObstacleIntrusion` | `obstacle_intrusion.py` | 1.0 | intrusion: -5 |
| `CapacityPenalty` | `capacity.py` | 1.0 | per_excess: -10, warning_threshold: 0.8 |
| `DelayPenalty` | `delay.py` | 0.2 | delay_per_step: -0.05 |
| `FlowEfficiencyReward` | `flow_efficiency.py` | 0.2 | reward_per_aircraft: 0.1 |
| `FairnessReward` | `fairness.py` | 0.1 | penalty_factor: 0.1 |
| `AltitudeReward` | `altitude_reward.py` | 0.5 | enroute: -5/3000, runway: -50/3000, crash: -100 |

### 5.3 冲突检测机制对比

| 机制 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 当前冲突 | 水平距离 < 5 NM | 水平 < 5 NM **且** 垂直 < 1000 ft |
| 预测冲突 | 无 | 向前投影位置，检测未来冲突 |
| 链式冲突 | 无 | BFS 图搜索，检测多机冲突链 |
| 三级严重性 | 无（单级入侵） | NMAC / Warning / Separation |

---

## 6. BlueSky 集成

### 6.1 初始化与生命周期

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 初始化 | 每个 env 的 `__init__` 中调用 `bs.init()` | `BlueSkyWrapper.init_simulation()` |
| 模式 | `bs.init(mode='sim', detached=True)` | 相同 |
| 时间步 | 各环境不同：`DT 1;FF` 或 `DT 5;FF` | `DT {config.dt};FF`，默认 5.0s |
| 冲突解决 | Merge 中 `reso off`，其余默认 | `reso off`（全局禁用） |
| 清理 | `bs.traf.delete(idx)` 逐个删除 | `DELETE {acid}` 栈命令，管理集合跟踪 |
| 封装 | 无封装，直接调用 bs 内部 API | `BlueSkyWrapper` 类封装所有交互 |

### 6.2 飞机创建

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 创建方式 | `bs.traf.cre(acid, actype, aclat, aclon, achdg, acalt, acspd)` | 相同，通过 `BlueSkyWrapper.create_aircraft()` |
| 冲突生成 | `bs.traf.creconfs()` — BlueSky 原生冲突生成 | `create_conflict_aircraft()` 封装 `creconfs` |
| 单位 | 直接传值（部分隐式转换） | 英尺→米、节→米/秒 显式转换 |
| 飞机类型 | 全部 "A320" | 可配置 `_AIRCRAFT_TYPE` |

### 6.3 状态读取

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 方式 | 直接读 `bs.traf.lat[i]`, `bs.traf.alt[i]` 等 | `BlueSkyWrapper.get_aircraft_state()` |
| 单位 | 原始值（米/米每秒） | 转换为英尺/节/英尺每分钟 |
| 索引 | `bs.traf.id2idx('KL001')` | `BlueSkyWrapper._resolve_idx()` |
| 批量读取 | 无（循环逐个读） | `get_all_aircraft_states()` 一次获取全部 |

### 6.4 命令发送

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 航向 | `bs.stack.stack(f"HDG {acid} {hdg}")` | `send_command()` / `send_commands_batch()` |
| 高度 | 直接写 `bs.traf.selalt[0]` 和 `bs.traf.selvs[0]` | `ALT {acid} {alt}` 栈命令 |
| 速度 | `bs.stack.stack(f"SPD {acid} {spd}")` | `SPD {acid} {spd}` 栈命令 |
| 批量 | 无（逐条发送） | `send_commands_batch(commands)` |

### 6.5 几何计算

| 工具 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 距离 | `bs.tools.geo.kwikdist()` | `haversine_distance()` Python 实现 |
| 方位 | `bs.tools.geo.kwikqdrdist()` | `bearing()` Python 实现 |
| 点在多边形 | `bs.tools.areafilter.checkInside()` | `point_in_polygon()` Python 实现 |
| 多边形定义 | `bs.tools.areafilter.defineArea()` | `generate_polygon()` Python 实现 |
| 位置推算 | `bs.tools.geo.kwikpos()` | 无 |
| 距离矩阵 | `bs.tools.geo.kwikdist_matrix()` | `haversine_distance_matrix()` numpy 向量化实现 |

### 6.6 测试基础设施

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 仿真后端 | 直接使用 BlueSky | BlueSkyWrapper（真实 BlueSky 引擎） |
| 测试方式 | 需要 BlueSky | 所有测试使用真实 BlueSky |
| 物理模拟 | BlueSky 原生 | BlueSky 原生（完整飞行动力学） |

---

## 7. 训练系统

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 脚本 | `main.py`（单一文件，`TRAIN=True/False` 切换） | `train_ppo_scenarios.py`（CLI 专用） |
| CLI 参数 | 无（改源码） | `--scenario`, `--timesteps`, `--resume`, `--seed`, `--num-aircraft`, `--max-steps` |
| 算法 | PPO, SAC, TD3, DDPG（4 种） | PPO, SAC, TD3, DDPG（4 种） |
| 策略 | `MultiInputPolicy` | `MultiInputPolicy` |
| 超参数 | lr=3e-4, 2M 步 | YAML 配置（`config/algorithms.yaml`），默认 lr=3e-4 |
| 检查点 | 无 | `CheckpointManager`：定期保存、轮转（最多 5 个）、元数据 JSON |
| 断点续训 | 无 | `--resume` 从检查点恢复 |
| 日志 | 无结构化日志 | `CSVLoggerCallback`：timestep, episode, reward, length, conflicts, arrivals, timestamp |
| 烟雾测试 | 无 | `train_smoke_test.py`（10k 步快速验证） |
| 性能基准 | 无 | `benchmark_performance.py`（步时间、内存） |
| 基线对比 | 无 | `evaluate_baselines.py` + `run_baselines.py` |
| 多场景支持 | 手动改 `env_name` | `--scenario` 参数选择 10 个场景 |

---

## 8. 评估系统

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 评估脚本 | 内嵌在 `main.py`（`TRAIN=False` 时） | 专用 `evaluate_baselines.py` + `ModelEvaluator` 类 |
| 评估指标 | 仅总奖励 | mean_reward, std_reward, min, max, mean_steps, arrival_rate, nmac_rate |
| 基线对比 | 无 | Random vs RuleBased vs PPO 表格输出 |
| 评估 episodes | 硬编码 10 | 可配置 `--episodes` |
| 确定性策略 | `deterministic=True` | `deterministic=True` |
| 结果格式 | 打印每 episode 奖励 | 结构化 `EvalResult` 数据类 + `format_table()` |
| 结果保存 | 无 | 可扩展（当前仅打印） |
| RuleBased Agent | 无 | `RuleBasedAgent`：TCAS 右转规则 + 优先级让行 + 目标引导 |
| Random Agent | 无 | `RandomAgent` |

---

## 9. 配置系统

| 维度 | bluesky-gym | bluesky-PettingZoo |
|------|------------|-------------------|
| 配置方式 | 全部硬编码为模块级常量 | YAML 外部配置文件 |
| 仿真参数 | 各环境文件内 `DT`, `ACTION_FREQUENCY` | `config/default.yaml` |
| 奖励参数 | 各环境文件内常量 | `config/rewards.yaml`（10 组件、权重、阈值） |
| 场景参数 | 各环境文件内常量 | `config/scenarios/*.yaml` |
| 观测参数 | 硬编码归一化常量 | `config/default.yaml` 的 `normalization` 和 `observation` 段 |
| 动作参数 | `D_HEADING`, `D_SPEED` 等 | `config/default.yaml` 的 `action` 段 |
| 修改方式 | 改源码 | 改 YAML 文件，无需改代码 |
| 场景加载 | 无（环境类直接使用） | `BaseScenario.from_config()` + `_SCENARIO_REGISTRY` |

---

## 10. 独有功能

### 10.1 bluesky-gym 独有

| 功能 | 说明 |
|------|------|
| **7 个环境渲染器** | 各环境定制 Pygame 渲染（飞机图标、航路点、冲突圈、跑道、航向线） |
| **`kwikdist_matrix`** | 批量距离矩阵计算（MergeEnv 中使用） |
| **`areafilter`** | BlueSky 原生多边形定义和检测 |
| **子步中途终止** | StaticObstacleEnv 在仿真子步循环中检查入侵并立即终止 |

### 10.2 bluesky-PettingZoo 独有

| 功能 | 说明 |
|------|------|
| **多智能体** | PettingZoo ParallelEnv，所有智能体同时观测和动作 |
| **SingleAgentGymWrapper** | 将多智能体环境包装为标准 gym.Env，兼容 SB3 |
| **4 种 RL 算法** | PPO, SAC, TD3, DDPG — SAC/TD3/DDPG 自动切换连续动作空间 |
| **连续动作空间** | `Box(-1, 1, shape=(3,))` — SAC/TD3/DDPG 自动切换，step() 双路径分发 |
| **航路点流式更新** | `BaseScenario.update_waypoint()` — 到达后生成新航路点，延长 episode |
| **高度奖励分段缩放** | `AltitudeReward` — 航路/跑道到达/坠毁三段式惩罚 |
| **NPC 导航行为** | `configure_npc_navigation()` — 背景飞机 LNAV + `reso off` |
| **`creconfs` 冲突生成** | `BlueSkyWrapper.create_conflict_aircraft()` 封装原生 `creconfs` |
| **Pygame 渲染** | 10 个场景渲染器，覆盖全部场景 |
| **模块化奖励** | 10 个可插拔组件（含 AltitudeReward 三段式），加权求和，YAML 配置 |
| **预测冲突检测** | 向前投影位置，检测未来冲突 |
| **链式冲突检测** | BFS 图搜索，检测多机冲突链 |
| **三级冲突严重性** | NMAC (-100) / Warning (-10) / Separation (-5) |
| **优先级系统** | 每个智能体有优先级值，用于冲突解决让行 |
| **感知过滤** | 可配置半径、高度差、最大观测数 |
| **mask 系统** | 处理可变数量的周围飞机 |
| **TextualState** | 人类可读状态文本（为 LLM/RAG 预留） |
| **AirspaceSnapshot** | 扇区、航路点、全部飞机位置快照 |
| **FlowScheduler** | 离场/到场间隔和扇区移交跟踪 |
| **容量管理** | 每扇区容量约束 + 警告阈值 |
| **延迟惩罚** | 按时间比例惩罚逾期到达 |
| **流量效率奖励** | 奖励扇区吞吐量 |
| **公平性奖励** | 惩罚不均等延迟分布 |
| **动态飞机注入** | 可配置周期性从边界注入新飞机 |
| **SectorCapacity 场景** | 容量约束多扇区管理 |
| **RouteNav 场景** | 航路网络交叉 |
| **检查点管理** | 定期保存 + 元数据 + 轮转 + 断点续训 |
| **CSV 日志** | 结构化训练日志 |
| **基线对比** | Random vs RuleBased vs PPO |
| **RuleBased Agent** | TCAS 右转规则 + 优先级让行 |
| **风场包装器** | `WindFieldWrapper`：均匀风场注入 + 机体坐标系风观测 |
| **噪声观测包装器** | `NoisyObservationWrapper`：可复现噪声注入 |
| **类型系统** | `AircraftState`, `DiscreteAction`, `SpawnConfig`, `ConflictConfig` 等 |
| **YAML 场景加载** | `BaseScenario.from_config()` + 注册表 |
| **1026 个测试** | 全面的单元/集成测试套件 |
| **性能基准** | 步时间和内存分析脚本 |
| **几何工具** | 独立于 BlueSky 的 Python 实现 |

---

## 11. 总结

### 功能覆盖矩阵

| 功能类别 | bluesky-gym | bluesky-PettingZoo | 差距 |
|---------|------------|-------------------|------|
| 环境数量 | 7 | 10 | PettingZoo 多 3 个（SectorCapacity, RouteNav, PlanWaypoint） |
| 多智能体 | ✗ | ✓ | PettingZoo 独有 |
| 训练算法 | 4 (PPO/SAC/TD3/DDPG) | 4 (PPO/SAC/TD3/DDPG) | 持平 |
| 连续动作 | ✓ | ✓ | 持平（SAC/TD3/DDPG 自动切换） |
| 检查点管理 | ✗ | ✓ | PettingZoo 独有 |
| YAML 配置 | ✗ | ✓ | PettingZoo 独有 |
| 模块化奖励 | ✗ | ✓ | PettingZoo 独有 |
| 预测冲突检测 | ✗ | ✓ | PettingZoo 独有 |
| 可视化渲染 | ✓ (7 场景) | ✓ (10 场景) | PettingZoo 覆盖全部场景 |
| `creconfs` 冲突生成 | ✓ | ✓（支持 per-intruder 参数） | PettingZoo 更灵活 |
| 子步中途终止 | ✓ | ✓（on_substep 回调） | 持平 |
| 距离矩阵 | ✓ (C/Fortran) | ✓ (numpy 向量化) | 持平 |
| 测试套件 | ✗ | 1026 tests | PettingZoo 独有 |
| 评估基线 | ✗ | Random/RuleBased/PPO | PettingZoo 独有 |
| 训练效果 | 25 个可用模型 | PPO 烟雾测试通过（+19.03） | 训练已可用 |

### 关键差距（按优先级）

1. **长期训练效果待验证** — 烟雾测试通过，但大规模训练（50k+ 步）效果未充分验证
2. **性能优化** — 距离矩阵已向量化，但尚未集成到冲突检测、感知过滤等热路径中
