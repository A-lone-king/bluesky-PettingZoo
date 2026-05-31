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

## 扩展方式

继承 `BaseRenderer`，实现 `render(observation, info)` 方法，并在场景中注册渲染器。
