# scripts/

训练、评估和工具脚本目录。

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `train_ppo_scenarios.py` | PPO 多场景训练主脚本，支持多环境并行和 GPU 加速 |
| `train_all_algos.py` | 多算法批量训练脚本 |
| `train_smoke_test.py` | 训练流程冒烟测试（验证环境和算法可运行） |
| `evaluate_baselines.py` | 评估基线 Agent 性能（RandomAgent, RuleBasedAgent） |
| `evaluate_all.py` | 批量评估所有场景和算法 |
| `run_baselines.py` | 运行基线 Agent 并保存结果 |
| `benchmark_performance.py` | 性能基准测试（FPS、内存等） |
| `refactor_tests.py` | 测试代码重构辅助工具 |

## 使用方式

```bash
# 训练单个场景
python scripts/train_ppo_scenarios.py --scenario horizontal_cr --algo ppo

# 训练所有场景
python scripts/train_all_algos.py

# 评估基线
python scripts/evaluate_baselines.py

# 冒烟测试
python scripts/train_smoke_test.py
```

