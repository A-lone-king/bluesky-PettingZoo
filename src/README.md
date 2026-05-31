# src/

项目核心源码目录，包含 `bluesky_pettingzoo` 包的全部实现。

## 包结构

```
src/bluesky_pettingzoo/
├── envs/               # PettingZoo ParallelEnv 实现
│   ├── parallel_env.py # 核心环境类
│   └── scenarios/      # 各场景实现（horizontal_cr, vertical_cr 等）
├── bluesky/            # BlueSky 仿真引擎封装
│   └── wrapper.py      # 同步 headless 模式封装
├── observations/       # 观测空间管理
│   ├── manager.py      # 观测管理器
│   ├── filters.py      # 感知范围过滤
│   └── normalizer.py   # 观测归一化
├── actions/            # 动作空间管理
│   └── translator.py   # 动作翻译器（RL 动作 → BlueSky 命令）
├── rewards/            # 奖励函数系统
│   ├── base.py         # RewardComponent 基类
│   ├── calculator.py   # RewardCalculator 奖励计算器
│   └── components/     # 各奖励分量实现
├── agents/             # 基线 Agent
│   ├── base.py         # Agent 基类
│   ├── random_agent.py # 随机 Agent
│   └── rule_based_agent.py  # 基于规则的 Agent
├── wrappers/           # 环境包装器
│   ├── single_agent.py # 单智能体包装
│   ├── noisy_observation.py  # 噪声观测包装
│   └── wind_field.py   # 风场包装
├── rendering/          # 可视化渲染器
│   ├── base_renderer.py
│   └── *_renderer.py   # 各场景专用渲染器
├── flow/               # 流量管理
│   └── scheduler.py    # 航班调度器
├── training/           # 训练工具
│   ├── algorithm_factory.py  # 算法工厂
│   ├── evaluator.py    # 评估器
│   ├── checkpoint.py   # 检查点管理
│   ├── logger.py       # CSV 日志记录
│   └── progress.py     # 进度跟踪
└── utils/              # 通用工具
    ├── geometry.py     # 几何计算
    ├── mixin.py        # Mixin 类
    └── types.py        # 类型定义
```

## 开发指引

- 新增场景：在 `envs/scenarios/` 下创建新模块，继承 `BaseScenario`
- 新增奖励分量：在 `rewards/components/` 下创建新模块，继承 `RewardComponent`
- 新增渲染器：在 `rendering/` 下创建新模块，继承 `BaseRenderer`
- 所有模块必须有对应的测试文件在 `tests/` 目录下
