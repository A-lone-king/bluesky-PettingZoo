# wrappers/

环境包装器模块，提供观测增强、噪声注入和接口转换功能。

## 文件说明

| 文件 | 用途 |
|------|------|
| `base.py` | `EnvWrapperMixin` 基类，统一包装器委托实现 |
| `single_agent.py` | `SingleAgentGymWrapper` 单智能体包装器（ParallelEnv → 单 Agent 接口） |
| `noisy_observation.py` | `NoisyObservationWrapper` 噪声观测包装器（模拟传感器误差） |
| `wind_field.py` | `WindFieldWrapper` 风场包装器（添加风场扰动） |

## 使用方式

```python
from bluesky_pettingzoo.wrappers import SingleAgentGymWrapper, NoisyObservationWrapper

# 包装为单智能体接口（对接 SB3）
env = SingleAgentGymWrapper(parallel_env)

# 添加噪声观测
env = NoisyObservationWrapper(env, noise_std=0.1)
```

## 设计要点

- `EnvWrapperMixin` 提供统一的委托实现，减少重复代码
- 包装器支持链式组合，可同时使用多个包装器
- `SingleAgentGymWrapper` 是对接 Stable-Baselines3 的关键接口

## 扩展方式

继承 `EnvWrapperMixin` 或 `gymnasium.ObservationWrapper`，实现需要的包装逻辑。
