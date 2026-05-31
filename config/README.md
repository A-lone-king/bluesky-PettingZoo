# config/

YAML 配置文件目录，管理所有仿真环境参数，实现代码与配置彻底解耦。

## 目录结构

```
config/
├── default.yaml          # 全局默认参数（仿真步长、时间限制等）
├── rewards.yaml          # 奖励函数权重配置（所有 reward component 共享）
├── algorithms.yaml       # RL 算法超参数配置
└── scenarios/            # 各场景独立配置
    ├── horizontal_cr.yaml
    ├── vertical_cr.yaml
    ├── sector_cr.yaml
    ├── descent.yaml
    ├── merge.yaml
    ├── static_obstacle.yaml
    ├── plan_waypoint.yaml
    ├── route_nav.yaml
    ├── sector_capacity.yaml
    └── waypoint_nav.yaml
```

## 配置层级

配置按优先级从低到高覆盖：`default.yaml` → `scenarios/<name>.yaml` → 命令行参数。

## 修改指南

- 新增场景时，在 `scenarios/` 下创建对应的 YAML 文件
- 奖励权重变更统一在 `rewards.yaml` 中调整
- 算法参数变更在 `algorithms.yaml` 中调整
- 所有配置项必须有注释说明含义和单位
