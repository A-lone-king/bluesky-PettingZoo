# envs/

PettingZoo ParallelEnv 核心实现模块，定义多智能体环境和各 ATM 场景。

## 文件说明

| 文件 | 用途 |
|------|------|
| `parallel_env.py` | `BlueSkyMARLEnv` 核心环境类，实现 ParallelEnv 接口 |
| `scenarios/` | 各场景实现目录 |

## 已实现场景

| 场景 | 类名 | 说明 |
|------|------|------|
| `horizontal_cr` | `HorizontalCRScenario` | 水平冲突解脱 |
| `vertical_cr` | `VerticalCRScenario` | 垂直冲突解脱 |
| `sector_cr` | `SectorCRScenario` | 扇区冲突解脱 |
| `descent` | `DescentScenario` | 下降阶段 |
| `merge` | `MergeScenario` | 汇合冲突 |
| `static_obstacle` | `StaticObstacleScenario` | 禁飞区规避 |
| `plan_waypoint` | `PlanWaypointScenario` | 顺序航路点导航 |
| `route_nav` | `RouteNavScenario` | 航路导航 |
| `sector_capacity` | `SectorCapacityScenario` | 扇区容量管理 |
| `waypoint_nav` | `WaypointNavScenario` | 航路点导航 |

## ParallelEnv 接口

```python
class BlueSkyMARLEnv(ParallelEnv):
    agents: list[AgentID]
    possible_agents: list[AgentID]

    def reset(self, seed=None, options=None) -> tuple[dict, dict]: ...
    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]: ...
    def observation_space(self, agent) -> Space: ...
    def action_space(self, agent) -> Space: ...
```

## 扩展方式

1. 在 `scenarios/` 下创建新文件，继承 `BaseScenario`
2. 实现 `setup_scenario()`, `get_observation()`, `get_reward()` 等方法
3. 在 `config/scenarios/` 下创建对应的 YAML 配置文件
4. 在 `scenarios/__init__.py` 中注册新场景类
