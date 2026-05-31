# training/

训练基础设施模块，提供算法工厂、评估、检查点管理和日志记录。

## 文件说明

| 文件 | 用途 |
|------|------|
| `algorithm_factory.py` | 算法工厂，根据配置创建 PPO/DQN 等算法实例 |
| `evaluator.py` | `ModelEvaluator` 评估器，支持多场景批量评估 |
| `checkpoint.py` | `CheckpointManager` 检查点管理，保存/加载模型和元数据 |
| `logger.py` | `CSVLoggerCallback` CSV 日志记录，持久化训练指标 |
| `progress.py` | 进度跟踪，训练进度可视化 |

## 核心类

- `AlgorithmFactory` — 根据 `algorithms.yaml` 创建 RL 算法
- `ModelEvaluator` — 批量评估模型在各场景下的性能
- `CheckpointManager` — 管理 checkpoint 保存/加载，支持元数据
- `CSVLoggerCallback` — 将训练指标写入 CSV 文件

## 使用方式

```python
from bluesky_pettingzoo.training import AlgorithmFactory, ModelEvaluator

factory = AlgorithmFactory()
algo = factory.create("ppo", env)

evaluator = ModelEvaluator()
results = evaluator.evaluate_all(algo, scenarios=["horizontal_cr", "vertical_cr"])
```
