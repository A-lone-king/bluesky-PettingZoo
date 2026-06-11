# bluesky-gym vs bluesky-pettingzoo 环境对比

> bluesky-gym（单智能体）7 种环境 vs bluesky-pettingzoo（多智能体）12 种环境的实现差异分析。

---

## 一、架构差异概述

| 维度 | bluesky-gym | bluesky-pettingzoo |
|------|-------------|-------------------|
| RL 框架 | Gymnasium 单智能体 | PettingZoo ParallelEnv 多智能体 |
| 控制模式 | 1 架 RL + N-1 架 rule-based（RVO2） | 全部/部分 RL（MULTI_RL 或 SINGLE_RL） |
| API 标准 | `env.step(action)` → 单个 obs/reward | `env.step(actions_dict)` → 每个 agent 独立 obs/reward |
| 优先级系统 | 无 | 按高度/速度/距离归一化，用于多智能体决策排序 |
| 仿真引擎 | BlueSky | BlueSky（通过 BlueSkyWrapper 封装） |

---

## 二、7 个共有环境逐项对比

### 1. HorizontalCR（水平冲突消解）

| 维度 | bluesky-gym（单智能体） | bluesky-pettingzoo（多智能体） |
|------|------------------------|-------------------------------|
| 智能体数 | 5 架 | 5 架（支持 `num_aircraft_range` 动态范围） |
| 控制模式 | 单架 RL + 4 架 rule-based | **全部 RL**（MULTI_RL） |
| 动作空间 | heading ±20° 5 离散动作 | heading ±20° 5 离散动作 |
| 优先级 | 无 | 按高度归一化到 [-1, 1]（越高优先级越高） |
| 航路点 | 左右两侧随机 | 东/西交替放置，刻意制造迎头冲突 |
| 扩展功能 | 无 | `num_altitude_layers` 多高度层、`create_intruders` 冲突创建 |

**相同点**：常量完全匹配（NMAC 5 NM / 1000 ft、警告 10 NM / 2000 ft、速度 240-340 kt、空域 ±2°）

### 2. VerticalCR（垂直冲突消解）

| 维度 | bluesky-gym（单智能体） | bluesky-pettingzoo（多智能体） |
|------|------------------------|-------------------------------|
| 智能体数 | 5 架 | 5 架（支持动态范围） |
| 控制模式 | 单架 RL + 4 架 rule-based | **全部 RL**（MULTI_RL） |
| 动作空间 | vs ±500 ft/min 5 离散动作 | vs ±500 ft/min 5 离散动作 |
| 优先级 | 无 | 按速度归一化（更快的飞机优先级更高） |
| 扩展功能 | 无 | **新增进近剖面模式**：3° 下滑道、250→140 kt 减速、最终高度 3000 ft |

**相同点**：水平聚簇（~2 NM）、垂直交错（20-40 kft）、冲突阈值相同

### 3. SectorCR（扇区冲突消解）

| 维度 | bluesky-gym（单智能体） | bluesky-pettingzoo（多智能体） |
|------|------------------------|-------------------------------|
| 智能体数 | 5 架 | 5 架（支持动态范围） |
| 控制模式 | 单架 RL + 4 架 rule-based | **全部 RL**（MULTI_RL） |
| 动作空间 | heading + vs + speed 全三轴 | heading + speed（**缺少 vs**） |
| 扇区 | 随机凸多边形 | 随机凸多边形 |
| 优先级 | 无 | 按距航路点距离归一化（越近优先级越高） |
| 扩展功能 | 无 | **新增动态容量调度**（高峰/低谷交替、4-6 时间段） |

**相同点**：扇区生成逻辑、边界检测、5 NM / 1000 ft 阈值

### 4. WaypointNav（航路点导航）

| 维度 | bluesky-gym（单智能体） | bluesky-pettingzoo（多智能体） |
|------|------------------------|-------------------------------|
| 智能体数 | 5 架 | 3 架 |
| 控制模式 | 单架 RL + 4 架 rule-based | **全部 RL**（MULTI_RL） |
| 动作空间 | heading + speed | heading（**缺少 speed**） |
| 速度 | 150 kt 固定 | 150 kt 固定 |
| 到达阈值 | 5 NM | 2 NM |
| 安全分离 | 无 | **新增**：30 NM 最小分离（拒绝采样） |

**相同点**：随机方向航路点、空域中心起始、矩形空域、到达终止

### 5. Merge（进近合并）

| 维度 | bluesky-gym（单智能体） | bluesky-pettingzoo（多智能体） |
|------|------------------------|-------------------------------|
| 智能体数 | 20 架 | 20 架（1 可控 + 19 背景交通） |
| 控制模式 | 单架 RL + 19 架 rule-based | **SINGLE_RL**（1 架 RL + 19 架背景交通） |
| 动作空间 | heading + vs + speed 全三轴 | heading + altitude + speed 全三轴 |
| FAF 距离 | 20-30 NM | 15-30 NM |
| NMAC 阈值 | 4 NM（进近更严格） | 4 NM（进近更严格） |
| 背景交通 | 规则生成 | 方位角 0-360° 均匀分布 |

**相同点**：单可控飞机、FAF 汇聚、禁用冲突消解、严格 NMAC 标准

### 6. Descent（下降进近）

| 维度 | bluesky-gym（单智能体） | bluesky-pettingzoo（多智能体） |
|------|------------------------|-------------------------------|
| 智能体数 | 3 架 | 3 架（1 可控 + 2 背景） |
| 控制模式 | 单架 RL + 2 架 rule-based | **SINGLE_RL** |
| 动作空间 | vs ±500 ft/min 5 离散动作 | vs ±500 ft/min 5 离散动作 |
| 目标高度 | 2000-6000 ft | 2000-6000 ft |
| 进近速度 | 150 kt | 150 kt |
| 撞地检测 | alt ≤ 50 ft | alt ≤ 0 |
| 跑道 | 空域中心 | 空域中心 |

**相同点**：巡航高度随机、目标高度随机、下降优先

### 7. FlightPlan（飞行计划）

| 维度 | bluesky-gym（单智能体） | bluesky-pettingzoo（多智能体） |
|------|------------------------|-------------------------------|
| 智能体数 | 1-3 架 | 3+ 架（取决于计划文件） |
| 控制模式 | 单架 RL + rule-based | **SINGLE_RL** |
| 飞行计划格式 | CSV/JSON | CSV/JSON（使用 FlightPlanParser） |
| 航路点导航 | BlueSky LNAV | BlueSky LNAV |
| 动态进入 | 有（entry_time） | 有（entry_time） |
| 扩展功能 | 无 | **新增航路点流式更新**（距当前航路点 < 2 NM 自动推进） |

**相同点**：多航路点导航、CSV/JSON 解析、LNAV 命令

---

## 三、bluesky-pettingzoo 独有的 5 个环境

### WaypointNav（航路点导航基线）

纯导航任务，无冲突机会。飞机之间保持 30 NM 最小分离（拒绝采样），用于测试引导逻辑和到达终止。

### PlanWaypoint（顺序航路点导航）

单架飞机按顺序访问 5 个航路点（间隔约 30 NM），仅航向控制。提供 `check_arrival`、`mark_reached`、`all_reached` 等状态管理方法。

### RouteNav（航路导航）

5-7 个航路点组成网络，每架飞机分配 2-4 个连续航路点。航路点共享和交叉产生自然冲突。禁用 BlueSky 冲突消解。

### SectorCapacity（扇区容量管理）

多扇区（默认 2 个，每个容量 4 架）沿经度排列。飞机穿越扇区时不能超出容量限制。结合冲突消解和交通流量管理。

### StarApproach（STAR 标准终端到场）

仿阿姆斯特丹史基浦机场 3 条 STAR 程序（ARTIP3C/RIVER4M/SOBTU3G），每条 5 个航路点带高度约束（24000→6000 ft），加 IAF 和 ILS 27 跑道入口。更严格的 NMAC 标准（3 NM）。

---

## 四、核心差异总结

### 架构层面

- bluesky-gym 是**单智能体 RL**：1 架可控 + N-1 架 rule-based（RVO2 跟驰/避让）
- bluesky-pettingzoo 是**多智能体 RL**：全部或部分飞机由独立 RL 策略控制
- PettingZoo ParallelEnv 标准 vs Gymnasium 单智能体标准

### 实现层面

- **优先级系统**：bluesky-gym 无，pettingzoo 用于多智能体决策排序（按高度/速度/距离归一化）
- **动态容量调度**：SectorCR 新增高峰/低谷交替（bluesky-gym 无）
- **进近剖面**：VerticalCR 新增 3° 下滑道模式（bluesky-gym 无）
- **STAR 程序**：StarApproach 为全新场景（bluesky-gym 无）
- **飞行计划导入**：FlightPlan 支持 CSV/JSON 格式（bluesky-gym 无）

### 兼容性

- 核心常量（NMAC 阈值、速度范围、空域尺寸）与 bluesky-gym **完全匹配**
- 共有环境保持了与 bluesky-gym 的**行为兼容性**
- bluesky-pettingzoo 额外支持 `num_aircraft_range` 参数进行**过程化生成**

---

## 五、全部 12 场景速查表

| 场景 | 智能体数 | 控制模式 | 动作维度 | 动态进入 | 核心特征 |
|------|----------|----------|----------|----------|----------|
| HorizontalCR | 5 | MULTI_RL | 航向 | 否 | 同高度迎头冲突，东/西交替 |
| VerticalCR | 5 | MULTI_RL | 垂直速度 | 否 | 水平聚簇+垂直交错，支持进近剖面 |
| SectorCR | 5 | MULTI_RL | 航向+速度 | 否 | 随机凸多边形扇区，动态容量 |
| WaypointNav | 3 | MULTI_RL | 航向 | 否 | 纯导航基线，无冲突 |
| Merge | 20 | SINGLE_RL | 全三轴 | 否 | 1可控+19背景，严格进近标准 |
| Descent | 3 | SINGLE_RL | 垂直速度 | 否 | 下降到目标高度，撞地检测 |
| FlightPlan | 3+ | SINGLE_RL | 默认 | 是 | 飞行计划导入，动态进入 |
| StarApproach | 3 | SINGLE_RL | 默认 | 否 | 仿史基浦 STAR 程序 |
| StaticObstacle | 1 | MULTI_RL | 航向+速度 | 否 | 随机多边形禁飞区 |
| SectorCapacity | 6 | MULTI_RL | 航向+速度 | 否 | 多扇区容量限制 |
| RouteNav | 3 | MULTI_RL | 航向+速度 | 否 | 航路点网络，交叉航路 |
| PlanWaypoint | 1 | MULTI_RL | 航向 | 否 | 单机5航路点顺序访问 |
