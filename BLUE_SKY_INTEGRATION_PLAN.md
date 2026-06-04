# BlueSky 深度集成实施计划

> 基于 bluesky-gym 对比分析，提升 BlueSky 底层能力复用率。

## 目标

将 bluesky-pettingzoo 从"BlueSky 作为黑盒仿真器"升级为"深度利用 BlueSky 内置能力"，
与 bluesky-gym 的集成深度对齐，同时保持多智能体模块化架构优势。

---

## Phase 1: BlueSky 深度集成（基础层改造）

### 1.1 几何计算改用 bs.tools.geo

**现状**: 自实现 `haversine_distance()`, `bearing()`, `point_at_distance()`

**目标**: 优先使用 `bs.tools.geo`，保留自实现作为 fallback

**改造文件**:
- `src/bluesky_pettingzoo/utils/geometry.py` — 添加 `bs.tools.geo` 包装函数
- `src/bluesky_pettingzoo/bluesky/wrapper.py` — 暴露 geo 工具给外部调用

**API 映射**:
| 自实现 | BlueSky 对等 |
|--------|-------------|
| `haversine_distance(lat1, lon1, lat2, lon2)` | `bs.tools.geo.kwikdist(lat1, lon1, lat2, lon2)` |
| `bearing(lat1, lon1, lat2, lon2)` | `bs.tools.geo.kwikqdrdist(lat1, lon1, lat2, lon2)` → (qdr, dist) |
| `point_at_distance(lat, lon, dist, bearing)` | `bs.tools.geo.kwikpos(lat, lon, bearing, dist)` |
| 无 | `bs.tools.geo.kwikdist_matrix(lat0, lon0, lats, lons)` — 距离矩阵 |

**实现方案**:
```python
# utils/geometry.py
try:
    from bluesky.tools.geo import kwikdist, kwikqdrdist, kwikpos
    _HAS_BLUESKY_GEO = True
except ImportError:
    _HAS_BLUESKY_GEO = False

def haversine_distance(lat1, lon1, lat2, lon2):
    if _HAS_BLUESKY_GEO:
        return kwikdist(lat1, lon1, lat2, lon2)
    return _haversine_fallback(lat1, lon1, lat2, lon2)
```

**验证**: 现有几何测试全部通过

---

### 1.2 面积检测改用 bs.tools.areafilter

**现状**: 自实现 `point_in_polygon()`，在 Python 侧判断点是否在多边形内

**目标**: 用 BlueSky 原生面积系统，支持 defineArea + checkInside

**改造文件**:
- `src/bluesky_pettingzoo/utils/geometry.py` — 添加 area filter 包装
- `src/bluesky_pettingzoo/bluesky/wrapper.py` — 添加 `define_area()`, `check_inside()`, `delete_area()`
- `src/bluesky_pettingzoo/rewards/components/capacity.py` — 用 BlueSky 面积系统替代 assign_sector
- `src/bluesky_pettingzoo/rewards/components/obstacle_intrusion.py` — 用 BlueSky 检测障碍区入侵
- `src/bluesky_pettingzoo/envs/scenarios/` — 扇区/障碍物定义改用 defineArea

**API**:
```python
# wrapper.py
def define_area(self, name: str, area_type: str, points: list[float]) -> None:
    """定义多边形区域。points: [lat1, lon1, lat2, lon2, ...] 扁平化格式"""
    bs.tools.areafilter.defineArea(name, area_type, points)

def check_inside(self, name: str, lats, lons, alts) -> np.ndarray:
    """检查点是否在区域内。返回布尔数组"""
    return bs.tools.areafilter.checkInside(name, lats, lons, alts)

def delete_area(self, name: str) -> None:
    bs.tools.areafilter.deleteArea(name)
```

**局限性处理**:
- 2D 检查（高度被忽略）→ 需要在 Python 侧额外做高度过滤
- 无距离信息 → 保留 haversine 做距离计算
- 边界精度有限 → 可接受

**验证**: 扇区容量测试、障碍入侵测试通过

---

### 1.3 垂直控制改用 selalt/selvs

**现状**: 垂直场景用 `ALT {acid} {alt}` stack 命令

**目标**: 直接写 `bs.traf.selalt[idx]` 和 `bs.traf.selvs[idx]`，与 bluesky-gym 一致

**改造文件**:
- `src/bluesky_pettingzoo/actions/translator.py` — 新增 `translate_vertical()` 方法
- `src/bluesky_pettingzoo/bluesky/wrapper.py` — 新增 `set_vertical_control()` 方法

**实现逻辑** (参考 bluesky-gym VerticalCR):
```python
# wrapper.py
def set_vertical_control(self, acid: str, vertical_speed_ms: float) -> None:
    """直接设置垂直速度控制（绕过 stack 命令）

    前置条件: 已关闭 VNAV (bs.traf.swvnav[idx] = False)
    爬升时: selalt = 1_000_000 ft, selvs = vs
    下降时: selalt = 0 ft, selvs = vs
    """
    idx = self._resolve_idx(acid)
    if vertical_speed_ms >= 0:
        bs.traf.selalt[idx] = 1_000_000  # 极高目标 → 强制爬升
    else:
        bs.traf.selalt[idx] = 0           # 极低目标 → 强制下降
    bs.traf.selvs[idx] = vertical_speed_ms
```

**动作值转换**: action * 12.5 = 垂直速度 (m/s)

**VNAV 关闭**: 在 `init_simulation()` 或场景 `setup()` 中:
```python
for i in range(len(bs.traf.id)):
    bs.traf.swvnav[i] = False
```

**验证**: VerticalCR、Descent 场景测试通过

---

### 1.4 动作频率配置

**现状**: 每个 RL step = 1 个 sim.step()（1:1）

**目标**: 支持可配置的 `action_frequency`（每个 RL step 执行 N 个 sim.step()）

**改造文件**:
- `src/bluesky_pettingzoo/envs/parallel_env.py` — `step()` 中循环 N 次
- `src/bluesky_pettingzoo/envs/scenarios/base.py` — 添加 `action_frequency` 属性

**实现**:
```python
# parallel_env.py step() 中
action_freq = getattr(self._scenario, 'action_frequency', 1)
for _ in range(action_freq):
    self._wrapper.step()
```

**bluesky-gym 参考值**: `ACTION_FREQUENCY = 5`（HorizontalCR/PlanWaypoint）

**验证**: 训练收敛速度对比

---

## Phase 2: 场景与观测改进

### 2.1 过程式 reset() 场景生成

**现状**: `setup()` 固定参数，`reset()` 只清理

**目标**: 每次 `reset()` 随机化场景参数（bluesky-gym 核心设计原则）

**改造文件**: 每个 scenario 的 `reset()` 方法

**随机化内容** (参考 bluesky-gym):
- HorizontalCR/VerticalCR: 随机 intercept angle, CPA, time-to-separation
- SectorCR: 随机多边形顶点、随机飞机数量（基于密度采样）
- Merge: 随机 spawn 位置（bearing range）、随机 intruder 位置
- StaticObstacle: 随机障碍多边形（数量/大小/位置）

**实现**:
```python
# base.py
def reset(self, rng: np.random.RandomState) -> None:
    """重置场景，用 rng 重新随机化参数"""
    pass  # 默认空操作，子类覆盖
```

---

### 2.2 N-nearest neighbors 固定窗口

**现状**: PerceptionFilter 返回距离最近的 N 个

**目标**: 与 bluesky-gym 一致，固定 `NUM_AC_STATE = 4` 近邻

**改造文件**: `src/bluesky_pettingzoo/observations/filters.py`

---

## Phase 3: 性能与导航

### 3.1 BADA/OpenAP 性能模型

**改造文件**:
- `src/bluesky_pettingzoo/bluesky/wrapper.py` — 初始化时激活性能模型

**BlueSky 命令**:
```
PERF OpenAP         # 激活 OpenAP 性能模型（默认，开源免费）
PERF BADA           # 激活 BADA 性能模型（需要 EUROCONTROL 许可证）
PERF OFF            # 关闭性能模型
```

**约束效果**: 爬升/下降率受限、速度受限、燃油消耗

**已知问题**: BADA 模式在无数据文件时静默失败

当前 `init_simulation()` 发送 `PERF BADA` 后，BlueSky 内部 `coeff_bada.check()` 检测到文件缺失时，错误被 `try/except` 吞掉（只 print 警告），性能模型实际未激活但 wrapper 无法感知。

**修复方案**:

```python
# wrapper.py init_simulation() 中
def init_simulation(self) -> None:
    # ... 现有初始化代码 ...

    # Activate performance model
    perf_model = self.config.get("simulation", {}).get("performance_model", "openap")
    if perf_model and perf_model.lower() != "off":
        bs.stack.stack(f"PERF {perf_model}")
        bs.sim.step()  # 执行命令

        # 验证性能模型是否真正激活
        active_model = getattr(bs.traf, 'perfmodel', None)
        if active_model is None and perf_model.lower() != "off":
            import warnings
            warnings.warn(
                f"Performance model '{perf_model}' failed to activate. "
                f"Check if data files are available. Falling back to no performance model.",
                UserWarning,
                stacklevel=2,
            )

    # ... 后续代码 ...
```

**修复优先级**: 中（默认 OpenAP 不会触发，仅 BADA 模式需要）

**修复时机**: 下次修改 wrapper.py 时顺便添加

---

### 3.2 LNAV 航路跟随

**改造文件**:
- `src/bluesky_pettingzoo/bluesky/wrapper.py` — 添加 LNAV 相关方法
- `src/bluesky_pettingzoo/actions/translator.py` — 支持 LNAV 命令

**BlueSky LNAV 工作流**:
```
1. CRE 创建飞机
2. ORIG {acid} {lat} {lon}     # 起点
3. DEST {acid} {lat} {lon}     # 目的地
4. ADDWPT {acid} {lat} {lon}   # 添加途经航路点
5. LNAV {acid} ON              # 激活横向导航
6. VNAV {acid} ON              # 激活垂直导航（可选）
```

**验证**: RouteNav、WaypointNav 场景测试

---

### 3.3 VNAV 垂直导航模式

**与 1.3 配合**: VNAV ON 后飞机自动管理垂直剖面，但需手动写 selalt/selvs 的场景应先关闭 VNAV。

---

## Phase 4: 架构优化

### 4.1 parallel_env.py 拆分

**当前**: 778 行单体

**拆分为**:
- `envs/lifecycle.py` — reset/step 生命周期管理
- `envs/observation_builder.py` — 观测构建逻辑
- `envs/sector_tracker.py` — 扇区变化检测

---

### 4.2 Protocol 接口

替换 `hasattr()` duck-typing:
```python
from typing import Protocol

class EfficiencyComponent(Protocol):
    def set_goal(self, agent_id: str, lat: float, lon: float) -> None: ...

class ConflictComponent(Protocol):
    def get_conflict_status(self, own, others) -> str: ...
```

---

## 验证计划

每个 Phase 完成后运行:
```bash
# 单元测试
pytest tests/ -v --ignore=tests/integration

# 代码风格
ruff check src/ tests/
ruff format --check src/ tests/

# 类型检查
mypy src/bluesky_pettingzoo/

# 集成测试（需要 BlueSky）
pytest tests/integration/ -v
```
