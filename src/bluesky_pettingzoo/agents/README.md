# agents/

基线 Agent 模块，提供用于对比评估的预置智能体。

## 文件说明

| 文件 | 用途 |
|------|------|
| `base.py` | Agent 基类，定义 `select_action()` 接口 |
| `random_agent.py` | 随机动作 Agent，在动作空间内均匀采样 |
| `rule_based_agent.py` | 基于 BlueSky 内置 LNAV/VNAV 的规则 Agent |

## 使用方式

```python
from bluesky_pettingzoo.agents import RandomAgent, RuleBasedAgent

random_agent = RandomAgent(action_space)
rule_agent = RuleBasedAgent()
```

## 扩展方式

继承 `BaseAgent`，实现 `select_action(observation)` 方法即可创建自定义基线 Agent。
