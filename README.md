# bluesky-pettingzoo

基于 [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) 空中交通仿真平台与 [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) 多智能体强化学习框架，构建支持多智能体强化学习（MARL）的空管仿真环境。

目标是为低空经济空中交通管理研究提供长期可用的多智能体实验平台。

## 特性

- **PettingZoo ParallelEnv** — 所有智能体同时观测、同时动作，原生支持主流 MARL 框架
- **7 个场景** — 水平/垂直冲突解脱、扇区冲突、航路导航、进近汇合、下降阶段
- **模块化奖励函数** — 冲突惩罚、效率奖励、平滑惩罚，支持动态注册和权重配置
- **SB3 集成** — 通过 `SingleAgentGymWrapper` 无缝对接 Stable-Baselines3
- **配置驱动** — YAML 管理环境参数、奖励函数、观测空间
- **449 个测试** — 严格的 TDD 开发流程，覆盖率 >90%
- **真实 BlueSky 集成** — 支持 headless 模式运行真实仿真器

## 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/bluesky-pettingzoo.git
cd bluesky-pettingzoo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -e .

# 安装开发依赖
pip install -r requirements-dev.txt
```

## 快速开始

### 基础环境使用

```python
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario

# 创建环境
env = BlueSkyMARLEnv(config, wrapper, obs_manager, action_translator, reward_calculator, rewards_config,
                     scenario=WaypointNavScenario(num_aircraft=3, seed=42))

# 重置环境
observations, infos = env.reset(seed=42)

# 交互循环
for _ in range(100):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    if not env.agents:
        break
```

### PPO 训练

```python
from stable_baselines3 import PPO
from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper

# 包装为单智能体环境
env = SingleAgentGymWrapper(marl_env, ego_agent="AC000")

# 训练 PPO
model = PPO("MultiInputPolicy", env, n_steps=128, batch_size=64, verbose=1)
model.learn(total_timesteps=50_000)
```

### 运行训练脚本

```bash
# PPO 多场景训练与基线对比
python scripts/train_ppo_scenarios.py

# 基线评估（Random + RuleBased）
python scripts/evaluate_baselines.py

# 训练 smoke test（快速验证）
python scripts/train_smoke_test.py
```

## 场景

| 场景 | 文件 | 飞机数 | 动作维度 | 说明 |
|------|------|--------|----------|------|
| HorizontalCR | `horizontal_cr.py` | 3 | 航向 | 水平冲突解脱，同高度对飞 |
| VerticalCR | `vertical_cr.py` | 3 | 高度 | 垂直冲突解脱，集群爬升/下降 |
| SectorCR | `sector_cr.py` | 3 | 航向+速度 | 扇区内冲突，多边形边界约束 |
| WaypointNav | `waypoint_nav.py` | 3 | 航向 | 纯导航任务，航路点追踪 |
| Merge | `merge.py` | 5 | 全部 | 进近汇合，1 可控 + 4 背景 |
| Descent | `descent.py` | 3 | 高度 | 下降阶段，从巡航高度下降 |

所有场景继承自 `BaseScenario`，可自定义：
- `setup()` — 初始化飞机位置和航路点
- `get_spawn_config()` — 生成参数（高度、速度、航向范围）
- `should_truncate()` — 自定义截断条件
- `get_initial_positions()` — 指定初始位置（如多边形内）

## 项目结构

```
bluesky-PettingZoo/
├── src/bluesky_pettingzoo/
│   ├── actions/          # 动作翻译器
│   ├── agents/           # 基线智能体（Random, RuleBased）
│   ├── bluesky/          # BlueSky wrapper
│   ├── envs/
│   │   ├── parallel_env.py    # PettingZoo ParallelEnv 核心
│   │   └── scenarios/         # 6 个场景实现
│   ├── observations/     # 观测管理器
│   ├── rewards/
│   │   ├── calculator.py      # 模块化奖励计算器
│   │   └── components/        # 冲突/效率/平滑奖励组件
│   ├── utils/            # 类型定义、几何工具
│   └── wrappers/
│       ├── single_agent.py    # SB3 单智能体包装
│       ├── noisy_observation.py
│       └── wind_field.py
├── config/               # YAML 配置文件
├── scripts/              # 训练和评估脚本
└── tests/                # 449 个测试用例
```

## 配置

环境参数通过 YAML 配置管理：

```yaml
# config/default.yaml
simulation:
  dt: 5.0
  max_episode_steps: 50
  headless: true

aircraft:
  initial_count: 5
  spawn:
    altitude_range: [29000, 37000]
    speed_range: [400, 500]
    heading_range: [0, 360]

# config/rewards.yaml
components:
  conflict:
    weight: 1.0
    nmac_penalty: -100
    warning_penalty: -10
  efficiency:
    weight: 0.3
    arrival_reward: 10
    step_penalty: -0.01
  smoothness:
    weight: 0.5
    action_penalty: -0.1
```

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单元测试
pytest tests/ -v --ignore=tests/integration

# 运行集成测试
pytest tests/integration/ -v

# 代码检查
ruff check src/ tests/
ruff format src/ tests/

# 类型检查
mypy src/bluesky_pettingzoo/
```

## 训练结果

PPO（50k timesteps）在所有场景上显著优于基线：

| 场景 | PPO | Random | RuleBased |
|------|-----|--------|-----------|
| HorizontalCR | **-62** | -255 | -236 |
| VerticalCR | **-20** | -89 | -72 |
| SectorCR | **-87** | -287 | -290 |
| WaypointNav | **-64** | -212 | -281 |
| Merge | **-50** | -613 | -250 |
| Descent | **-36** | -173 | -138 |

## 依赖

- Python >= 3.11
- PettingZoo >= 1.24.0
- Gymnasium >= 0.29.0
- NumPy >= 1.24.0
- PyYAML >= 6.0

可选：
- Stable-Baselines3（PPO 训练）
- BlueSky（真实仿真器集成）

## 引用

如果这个项目对你的研究有帮助，请引用：

```bibtex
@software{bluesky_pettingzoo,
  title={bluesky-pettingzoo: Multi-Agent RL Environment for Air Traffic Management},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/bluesky-pettingzoo}
}
```

## 许可证

MIT License
