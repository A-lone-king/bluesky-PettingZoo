# Bluesky-PettingZoo 优化路线图

**创建日期**：2026-06-05
**状态**：V1.0 已完成，规划下一阶段

---

## 一、当前状态总结

### 已完成功能（19/19）

| 模块 | 功能 | 状态 |
|------|------|------|
| envs | BlueSkyMARLEnv ParallelEnv 核心 | ✅ passing |
| bluesky | BlueSkyWrapper 仿真引擎接口 | ✅ passing |
| bluesky | 几何计算改用 bs.tools.geo | ✅ passing |
| bluesky | 面积检测改用 bs.tools.areafilter | ✅ passing |
| bluesky | 垂直控制改用 selalt/selvs | ✅ passing |
| envs | 动作频率配置 | ✅ passing |
| envs | 10 个 ATM 场景实现 | ✅ passing |
| rewards | 9 个奖励组件实现 | ✅ passing |
| observations | 观测管理与归一化 | ✅ passing |
| actions | 离散/连续动作翻译 | ✅ passing |
| envs | 过程式场景随机生成 | ✅ passing |
| bluesky | LNAV 航路跟随集成 | ✅ passing |
| bluesky | 性能模型集成（OpenAP） | ✅ passing |
| training | 训练基础设施 | ✅ passing |
| wrappers | 环境包装器 | ✅ passing |
| tests | 测试体系（943 个测试） | ✅ passing |
| envs | parallel_env.py 架构拆分 | ✅ passing |
| envs | Protocol 接口替换 hasattr | ✅ passing |
| rendering | Pygame 可视化渲染 | ✅ passing |

### 测试覆盖率

- 单元测试：943 个
- 通过率：100%
- 代码风格：ruff + mypy 通过

---

## 二、下一阶段优化方向

### 方向 1：STAR/SID 进近程序场景（高价值）

**目标**：创建标准终端到场（STAR）和标准仪表离场（SID）场景，模拟真实机场进近流程。

**技术分析**：

BlueSky 的航路系统支持以下航路点类型：
```python
wplatlon = 0   # 经纬度航路点
wpnav    = 1   # VOR/nav 数据库航路点
orig     = 2   # 出发机场
dest     = 3   # 目的地机场
calcwp   = 4   # 计算航路点（T/C, T/D, A/C）
runway   = 5   # 跑道
```

虽然 BlueSky 没有直接的 STAR/SID 命令，但可以通过 **航路点序列 + LNAV** 实现：

```python
# 模拟 STAR 程序示例
wrapper.set_origin(acid, entry_lat, entry_lon)  # STAR 入口点
wrapper.add_waypoint(acid, wpt1_lat, wpt1_lon)  # 中间航路点
wrapper.add_waypoint(acid, wpt2_lat, wpt2_lon)  # 下降点
wrapper.set_destination(acid, airport_lat, airport_lon)  # 机场
wrapper.enable_lnav(acid)
```

**场景设计**：

| 场景名 | 描述 | 飞机数 | 动作空间 |
|--------|------|--------|----------|
| StarApproach | 多架飞机同时进近，需排序间隔 | 3-5 | 航向+速度+高度 |
| SidDeparture | 多架飞机离场，需保持间隔 | 3-5 | 航向+速度+高度 |
| StarMerge | STAR 末端汇合，需合并排序 | 4-6 | 航向+速度 |

**实现步骤**：

1. 创建 `src/bluesky_pettingzoo/envs/scenarios/star_approach.py`
2. 实现 STAR 程序航路点生成逻辑
3. 添加进近排序奖励组件
4. 编写单元测试和集成测试
5. 添加 Pygame 渲染器

**预估工作量**：3-5 天

---

### 方向 2：航班计划导入（高价值）

**目标**：支持从 CSV/JSON 文件导入真实航班计划，使训练环境更接近真实运行。

**数据格式设计**：

```csv
flight_id,aircraft_type,origin,destination,entry_time,entry_lat,entry_lon,entry_alt,entry_hdg,entry_spd,waypoints
AC001,B737,ZBAA,ZSPD,08:00,40.08,116.58,35000,180,450,"WPT1:40.2,116.6,30000;WPT2:40.5,116.8,25000"
AC002,A320,ZBAA,ZSPD,08:05,40.10,116.60,33000,175,440,"WPT1:40.2,116.6,28000;WPT2:40.5,116.8,23000"
```

**JSON 格式**：

```json
{
  "flights": [
    {
      "flight_id": "AC001",
      "aircraft_type": "B737",
      "origin": "ZBAA",
      "destination": "ZSPD",
      "entry_time": "08:00",
      "entry": {"lat": 40.08, "lon": 116.58, "alt": 35000, "hdg": 180, "spd": 450},
      "waypoints": [
        {"name": "WPT1", "lat": 40.2, "lon": 116.6, "alt": 30000},
        {"name": "WPT2", "lat": 40.5, "lon": 116.8, "alt": 25000}
      ]
    }
  ]
}
```

**实现步骤**：

1. 创建 `src/bluesky_pettingzoo/envs/scenarios/flight_plan.py`
2. 实现 CSV/JSON 解析器
3. 创建 `FlightPlanScenario` 类
4. 添加航班计划验证逻辑
5. 编写测试用例
6. 提供示例数据文件

**预估工作量**：2-3 天

---

### 方向 3：增强现有场景（中价值）

**目标**：改进现有 10 个场景的真实性和训练效果。

**改进点**：

| 场景 | 当前问题 | 改进方案 |
|------|----------|----------|
| HorizontalCR | 只有同高度对头 | 添加多高度层冲突 |
| VerticalCR | 简单爬升/下降 | 添加真实进近剖面 |
| Merge | 简单汇合 | 添加 STAR 程序 |
| SectorCapacity | 静态容量 | 添加动态容量变化 |
| StaticObstacle | 单个障碍物 | 添加多个动态障碍物 |

**预估工作量**：5-7 天（全部改进）

---

### 方向 4：数据记录与分析（中价值）

**目标**：记录训练和评估过程中的详细数据，支持离线分析。

**功能**：

1. **飞行轨迹记录**：记录每架飞机的位置、速度、高度时间序列
2. **冲突事件记录**：记录冲突发生时间、位置、严重程度
3. **奖励分解记录**：记录每个奖励组件的贡献
4. **回放功能**：支持事后回放和可视化

**数据格式**：

```json
{
  "episode_id": "ep_001",
  "timestamp": "2026-06-05T10:00:00",
  "scenario": "HorizontalCR",
  "duration": 120.5,
  "aircraft": {
    "AC001": {
      "trajectory": [[40.0, 116.5, 35000, 180, 450], ...],
      "conflicts": [{"time": 45.2, "severity": "warning"}]
    }
  },
  "rewards": {
    "total": -62.5,
    "conflict": -50.0,
    "efficiency": 10.5,
    "smoothness": -23.0
  }
}
```

**预估工作量**：3-4 天

---

## 三、优先级排序

| 优先级 | 方向 | 价值 | 工作量 | 依赖 |
|--------|------|------|--------|------|
| P0 | STAR/SID 进近程序场景 | 高 | 3-5 天 | 无 |
| P1 | 航班计划导入 | 高 | 2-3 天 | 无 |
| P2 | 增强现有场景 | 中 | 5-7 天 | P0 |
| P3 | 数据记录与分析 | 中 | 3-4 天 | 无 | ✅ |

**推荐执行顺序**：P0 → P1 → P3 → P2

---

## 四、技术风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| BlueSky STAR/SID 命令不明确 | 中 | 使用航路点序列 + LNAV 模拟 |
| 航班计划数据格式不兼容 | 低 | 设计灵活的解析器，支持多种格式 |
| OpenAP 性能模型精度不足 | 低 | 已有 BADA 备选方案（需许可证） |
| 渲染器需要更新 | 低 | 逐步添加新场景渲染器 |

---

## 五、文档更新计划

完成优化后需更新：

1. **README.md** - 添加新场景和功能说明
2. **feature_list.json** - 添加新功能条目
3. **ROADMAP.md** - 更新进度
4. **session-handoff.md** - 记录会话交接信息

---

## 六、参考资源

- BlueSky 文档：https://github.com/TUDelft-CNS-ATM/bluesky
- PettingZoo 文档：https://pettingzoo.farama.org/
- bluesky-gym 参考：https://github.com/jfink87/bluesky-gym
- OpenAP 文档：https://github.com/udp/openap
