# rewards 开发进度

## 核心模块

- [x] `base.py` — RewardComponent 基类（含 `get_config()` 和自动 `reset()`）
- [x] `calculator.py` — RewardCalculator 奖励计算器

## 已完成奖励分量

| 分量 | 文件 | 说明 |
|------|------|------|
| ConflictPenalty | `conflict.py` | 冲突惩罚 |
| EfficiencyReward | `efficiency.py` | 效率奖励 |
| DelayPenalty | `delay.py` | 延误惩罚 |
| DriftPenalty | `drift.py` | 偏航惩罚 |
| SmoothnessPenalty | `smoothness.py` | 平滑性惩罚 |
| AltitudeReward | `altitude_reward.py` | 高度奖励 |
| CapacityPenalty | `capacity.py` | 容量惩罚 |
| FairnessReward | `fairness.py` | 公平性奖励 |
| FlowEfficiencyReward | `flow_efficiency.py` | 流量效率奖励 |
| ObstacleIntrusion | `obstacle_intrusion.py` | 障碍物入侵惩罚 |

## 待开发

无待开发奖励分量。
