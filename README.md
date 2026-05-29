# bluesky-pettingzoo

将 [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) 空中交通仿真平台从单智能体环境（[bluesky-gym](https://github.com/jfink87/bluesky-gym)）扩展为多智能体环境，基于 [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) ParallelEnv 标准，专注于空中交通管理（ATM）领域的多智能体强化学习研究。

BlueSky 是底层仿真引擎，负责飞行动力学、冲突检测和空域管理；本项目在其上构建多智能体 RL 接口，使多个飞机 Agent 能够同时观测、同时决策。

## 特性

- **BlueSky 多智能体扩展** — 基于 PettingZoo ParallelEnv，将 BlueSky 从单智能体扩展为多智能体，原生支持主流 MARL 框架
- **10 个场景** — 水平/垂直冲突解脱、扇区冲突、航路导航、进近汇合、下降阶段、禁飞区规避、扇区容量、航路网络、顺序航路点
- **模块化奖励函数** — 冲突惩罚、效率奖励、平滑惩罚，支持动态注册和权重配置
- **SB3 集成** — 通过 `SingleAgentGymWrapper` 无缝对接 Stable-Baselines3
- **配置驱动** — YAML 管理环境参数、奖励函数、观测空间
- **1026 个测试** — 严格的 TDD 开发流程，覆盖率 >90%
- **真实 BlueSky 集成** — 支持 headless 模式运行真实仿真器
- **模块化架构** — 通过 Mixin 和基类消除重复代码，提供统一的扩展接口

## 安装

```bash
# 克隆仓库
git clone https://github.com/A-lone-king/bluesky-PettingZoo.git
cd bluesky-pettingzoo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -e .

# 安装 BlueSky 仿真引擎
pip install "bluesky-simulator[full]"

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
# PPO 训练（默认 HorizontalCR，50k timesteps）
rtk python scripts/train_ppo_scenarios.py

# 指定场景和训练步数
rtk python scripts/train_ppo_scenarios.py --scenario WaypointNav --timesteps 100000

# 可视化训练（启用 Pygame 渲染）
rtk python scripts/train_ppo_scenarios.py --scenario HorizontalCR --render

# 基线评估（Random + RuleBased）
rtk python scripts/evaluate_baselines.py --scenario HorizontalCR --episodes 20

# 训练模型评估
rtk python scripts/evaluate_baselines.py --scenario HorizontalCR --model models/ppo_HorizontalCR.zip

# 训练 smoke test（快速验证，约 1 分钟）
rtk python scripts/train_smoke_test.py

# 性能基准
rtk python scripts/benchmark_performance.py
```

**可用场景**：HorizontalCR, VerticalCR, SectorCR, WaypointNav, Merge, Descent, StaticObstacle, SectorCapacity, RouteNav, PlanWaypoint

**可用算法**：PPO, SAC, TD3, DDPG

**注意**：首次运行 BlueSky 仿真器需要加载导航数据库，可能需要 1-2 分钟。训练模型会保存到 `models/` 目录。

## 场景

| 场景 | 文件 | 飞机数 | 动作维度 | 说明 |
|------|------|--------|----------|------|
| HorizontalCR | `horizontal_cr.py` | 3 | 航向 | 水平冲突解脱，同高度对飞 |
| VerticalCR | `vertical_cr.py` | 3 | 高度 | 垂直冲突解脱，集群爬升/下降 |
| SectorCR | `sector_cr.py` | 3 | 航向+速度 | 扇区内冲突，多边形边界约束 |
| WaypointNav | `waypoint_nav.py` | 3 | 航向 | 纯导航任务，航路点追踪 |
| Merge | `merge.py` | 5 | 全部 | 进近汇合，1 可控 + 4 背景 |
| Descent | `descent.py` | 3 | 高度 | 下降阶段，从巡航高度下降 |
| StaticObstacle | `static_obstacle.py` | 1 | 航向+速度 | 禁飞区规避，多边形障碍物检测 |
| SectorCapacity | `sector_capacity.py` | 6 | 航向+速度 | 扇区容量管理，per-sector 容量约束 |
| RouteNav | `route_nav.py` | 4 | 航向+速度 | 航路网络导航，交叉路线冲突检测 |
| PlanWaypoint | `plan_waypoint.py` | 1 | 航向 | 顺序航路点导航，逐一到达 5 个航路点 |

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
│   │   └── scenarios/         # 10 个场景实现（继承 BaseScenario）
│   ├── observations/     # 观测管理器
│   ├── rewards/
│   │   ├── base.py            # RewardComponent 基类（get_config, 自动 reset）
│   │   ├── calculator.py      # 模块化奖励计算器
│   │   └── components/        # 冲突/效率/平滑/障碍物奖励组件
│   ├── flow/             # 流量管理（扇区容量调度）
│   ├── training/         # 训练框架（PPO、检查点、评估）
│   ├── rendering/
│   │   ├── base_renderer.py   # BaseRenderer（通用渲染逻辑）
│   │   └── ...                # 场景渲染器
│   ├── utils/
│   │   ├── types.py           # 类型定义（DictBackedMixin）
│   │   ├── mixin.py           # DictBackedMixin 基类
│   │   └── geometry.py        # 几何工具
│   └── wrappers/
│       ├── base.py            # EnvWrapperMixin（统一委托）
│       ├── single_agent.py    # SB3 单智能体包装
│       ├── noisy_observation.py
│       └── wind_field.py
├── config/               # YAML 配置文件
├── scripts/              # 训练和评估脚本
└── tests/                # 1026 个测试用例
    └── helpers/
        └── state_factory.py   # 共享测试工厂函数
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
| StaticObstacle | **-28** | -145 | -120 |
| SectorCapacity | **-246** | -2032 | -1572 |
| RouteNav | **-48** | -216 | -202 |

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
  url={https://github.com/A-lone-king/bluesky-PettingZoo}
}
```

## 许可证

MIT License
