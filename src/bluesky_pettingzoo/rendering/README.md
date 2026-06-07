# rendering/

可视化渲染模块，使用 Pygame 绘制仿真状态。

## 文件说明

| 文件 | 用途 |
|------|------|
| `base_renderer.py` | `BaseRenderer` 基类，提供通用渲染逻辑和 bounds 管理 |
| `common.py` | 通用渲染工具函数（颜色、坐标转换等） |
| `horizontal_cr_renderer.py` | 水平冲突解脱场景渲染器 |
| `vertical_cr_renderer.py` | 垂直冲突解脱场景渲染器 |
| `sector_cr_renderer.py` | 扇区冲突解脱场景渲染器 |
| `descent_renderer.py` | 下降阶段场景渲染器 |
| `merge_renderer.py` | 汇合冲突场景渲染器 |
| `static_obstacle_renderer.py` | 禁飞区规避场景渲染器 |
| `plan_waypoint_renderer.py` | 顺序航路点导航场景渲染器 |
| `route_nav_renderer.py` | 航路导航场景渲染器 |
| `sector_capacity_renderer.py` | 扇区容量管理场景渲染器 |
| `waypoint_nav_renderer.py` | 航路点导航场景渲染器 |

## 设计要点

- 每个场景有专用渲染器，继承 `BaseRenderer`
- 渲染器与场景解耦，通过接口绑定
- 支持 headless 模式下跳过渲染，仅在需要可视化时启用

## 渲染改进计划 (render-enhance-001)

参考 bluesky-gym 的渲染风格，改进渲染效果：

### Phase 1: 基础渲染增强 ✅
- [x] 添加天空渐变背景（浅蓝色）
- [x] 添加地面渲染（绿色区域）
- [x] 添加跑道渲染（灰色矩形）

### Phase 2: 场景特定渲染 ✅
- [x] VerticalCR：添加高度层方框、垂直冲突标记
- [x] SectorCR：改进多边形边界渲染
- [x] Merge：添加汇合线渲染
- [x] StaticObstacle：添加三角形障碍物渲染
- [x] PlanWaypoint：改进航路点渲染（白色圆圈）

### Phase 3: 视觉效果优化 ✅
- [x] 添加飞机尾迹
- [x] 添加冲突预警闪烁效果
- [x] 改进 HUD 信息显示

### bluesky-gym 渲染参考

| 环境 | 渲染特点 |
|------|----------|
| DescentEnv | 浅蓝色天空 + 绿色地面 + 灰色跑道 |
| VerticalCREnv | 多高度层飞机 + 方框冲突区域 + 红色冲突标记 |
| HorizontalCREnv | 同高度飞机 + 圆形保护区域 + 黑色/红色冲突圆 |
| SectorCREnv | 白色多边形边界 + 扇区内飞机 |
| MergeEnv | 白色汇合线 + 汇聚飞机 |
| StaticObstacleEnv | 黑色三角形障碍物 + 飞机绕行 |
| PlanWaypointEnv | 白色圆圈航路点 + 飞机依次访问 |

## 扩展方式

继承 `BaseRenderer`，实现 `render(observation, info)` 方法，并在场景中注册渲染器。
