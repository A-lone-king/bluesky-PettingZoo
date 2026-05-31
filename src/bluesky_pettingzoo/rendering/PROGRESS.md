# rendering 开发进度

## 核心模块

- [x] `base_renderer.py` — BaseRenderer 基类（通用渲染逻辑和 bounds 管理）
- [x] `common.py` — 通用渲染工具

## 已完成场景渲染器

| 渲染器 | 文件 | 对应场景 |
|--------|------|----------|
| HorizontalCR | `horizontal_cr_renderer.py` | 水平冲突解脱 |
| VerticalCR | `vertical_cr_renderer.py` | 垂直冲突解脱 |
| SectorCR | `sector_cr_renderer.py` | 扇区冲突解脱 |
| Descent | `descent_renderer.py` | 下降阶段 |
| Merge | `merge_renderer.py` | 汇合冲突 |
| StaticObstacle | `static_obstacle_renderer.py` | 禁飞区规避 |
| PlanWaypoint | `plan_waypoint_renderer.py` | 顺序航路点导航 |
| RouteNav | `route_nav_renderer.py` | 航路导航 |
| SectorCapacity | `sector_capacity_renderer.py` | 扇区容量管理 |
| WaypointNav | `waypoint_nav_renderer.py` | 航路点导航 |

## 待开发

无待开发渲染器。
