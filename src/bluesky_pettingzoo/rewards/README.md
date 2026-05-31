# rewards/

奖励函数模块，提供模块化的奖励计算框架。

## 文件说明

| 文件 | 用途 |
|------|------|
| `base.py` | `RewardComponent` 基类，提供 `get_config()` 和自动 `reset()` |
| `calculator.py` | `RewardCalculator` 奖励计算器，动态注册和组合奖励分量 |
| `components/` | 各奖励分量实现 |

## 已实现奖励分量

| 分量 | 文件 | 说明 |
|------|------|------|
| `ConflictPenalty` | `conflict.py` | 冲突惩罚 |
| `EfficiencyReward` | `efficiency.py` | 效率奖励（航迹偏差） |
| `DelayPenalty` | `delay.py` | 延误惩罚 |
| `DriftPenalty` | `drift.py` | 偏航惩罚 |
| `SmoothnessPenalty` | `smoothness.py` | 动作平滑性惩罚 |
| `AltitudeReward` | `altitude_reward.py` | 高度奖励 |
| `CapacityPenalty` | `capacity.py` | 扇区容量惩罚 |
| `FairnessReward` | `fairness.py` | 公平性奖励 |
| `FlowEfficiencyReward` | `flow_efficiency.py` | 流量效率奖励 |
| `ObstacleIntrusion` | `obstacle_intrusion.py` | 障碍物入侵惩罚 |

## 设计要点

- 每个奖励分量是独立的 `RewardComponent` 子类
- `RewardCalculator` 支持动态注册分量，通过 `rewards.yaml` 配置权重
- 所有分量自动 `reset()`，无需手动管理状态

## 扩展方式

1. 继承 `RewardComponent`，实现 `calculate()` 方法
2. 在 `components/__init__.py` 中注册
3. 在 `rewards.yaml` 中添加权重配置
