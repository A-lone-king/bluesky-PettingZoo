# CLAUDE.md — bluesky-pettingzoo

## 项目概览与目的

将 BlueSky 空中交通仿真平台从单智能体环境扩展为多智能体环境，基于 PettingZoo ParallelEnv 标准，专注于空中交通管理（ATM）领域的多智能体强化学习研究。

核心能力：多架飞机同时观测、同时决策、同时执行动作，支持 10 个 ATM 场景（冲突解脱、航路导航、汇合、下降、禁飞区规避、扇区容量管理等）。

## 技术栈与版本

| 组件 | 版本要求 | 用途 |
|------|---------|------|
| Python | >= 3.11 | 运行环境 |
| PettingZoo | >= 1.24.0 | 多智能体 RL 接口标准 |
| Gymnasium | >= 0.29.0 | RL 环境规范 |
| NumPy | >= 1.24.0 | 数值计算 |
| PyYAML | >= 6.0 | 配置管理 |
| BlueSky Simulator | >= 1.0.7 | 空中交通仿真引擎（可选） |
| pytest | >= 7.0 | 测试框架 |
| ruff | >= 0.1.0 | Lint 和格式化 |
| mypy | >= 1.0 | 静态类型检查 |

## 首次运行

```bash
# 1. 克隆并进入项目
git clone https://github.com/A-lone-king/bluesky-PettingZoo.git
cd bluesky-pettingzoo

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. 安装依赖
pip install -e .                # 核心依赖
pip install "bluesky-simulator[full]"  # BlueSky 引擎
pip install -r requirements-dev.txt     # 开发工具

# 4. 验证安装
pytest tests/ -v --ignore=tests/integration  # 跑单元测试
ruff check src/ tests/                       # 代码检查
```

## 不可违反的硬约束

1. **TDD 强制**：必须先写失败测试，再写实现。未写测试的代码不得合入。
2. **覆盖率 >= 90%**：每个功能模块的测试覆盖率不得低于 90%。
3. **禁止猜测**：不得猜测 API、版本号、commit SHA、包名。必须先读代码或文档再断言。
4. **类型注解**：所有公开函数必须有完整类型注解，mypy strict 模式通过。
5. **配置解耦**：环境参数必须通过 YAML 管理，禁止硬编码在 Python 代码中。
6. **中文交互**：所有交流和文档使用简体中文。
7. **Conventional Commits**：提交信息必须遵循 `<type>(<scope>): <description>` 格式。
8. **不修改已有测试**：测试文件一旦通过，不得为让实现通过而修改测试。只改实现。

## Reference Files

所有项目文档按目录组织。每个目录下的 README.md 说明用途和结构，PROGRESS.md 记录开发进度。

### 根目录文档

| File | Description |
|------|-------------|
| [architecture.md](architecture.md) | 技术架构、核心设计决策、BlueSky 接口、PettingZoo 标准、模块化设计 |
| [README.md](README.md) | 项目概述、安装、快速开始、场景列表 |

### 一级目录

| Directory | README | Description |
|-----------|--------|-------------|
| [config/](config/) | [README](config/README.md) | YAML 配置文件（default, rewards, algorithms, scenarios） |
| [docs/](docs/) | [README](docs/README.md) | 非代码类参考资料 |
| [models/](models/) | [README](models/README.md) | 预训练模型 checkpoint 存储 |
| [models_test/](models_test/) | [README](models_test/README.md) | 测试用模型 |
| [scripts/](scripts/) | [README](scripts/README.md) | 训练、评估和工具脚本 |
| [src/](src/) | [README](src/README.md) | 核心源码（bluesky_pettingzoo 包） |
| [tests/](tests/) | [README](tests/README.md) | 单元测试、集成测试、端到端测试 |

### src/bluesky_pettingzoo 子模块

| Directory | README | PROGRESS | Description |
|-----------|--------|----------|-------------|
| [actions/](src/bluesky_pettingzoo/actions/) | [README](src/bluesky_pettingzoo/actions/README.md) | [PROGRESS](src/bluesky_pettingzoo/actions/PROGRESS.md) | RL 动作 → BlueSky 命令翻译 |
| [agents/](src/bluesky_pettingzoo/agents/) | [README](src/bluesky_pettingzoo/agents/README.md) | [PROGRESS](src/bluesky_pettingzoo/agents/PROGRESS.md) | 基线 Agent（Random, RuleBased） |
| [bluesky/](src/bluesky_pettingzoo/bluesky/) | [README](src/bluesky_pettingzoo/bluesky/README.md) | [PROGRESS](src/bluesky_pettingzoo/bluesky/PROGRESS.md) | BlueSky 仿真引擎封装 |
| [envs/](src/bluesky_pettingzoo/envs/) | [README](src/bluesky_pettingzoo/envs/README.md) | [PROGRESS](src/bluesky_pettingzoo/envs/PROGRESS.md) | ParallelEnv 核心 + 10 个场景 |
| [flow/](src/bluesky_pettingzoo/flow/) | [README](src/bluesky_pettingzoo/flow/README.md) | [PROGRESS](src/bluesky_pettingzoo/flow/PROGRESS.md) | 航班调度器 |
| [observations/](src/bluesky_pettingzoo/observations/) | [README](src/bluesky_pettingzoo/observations/README.md) | [PROGRESS](src/bluesky_pettingzoo/observations/PROGRESS.md) | 观测管理、过滤、归一化 |
| [rendering/](src/bluesky_pettingzoo/rendering/) | [README](src/bluesky_pettingzoo/rendering/README.md) | [PROGRESS](src/bluesky_pettingzoo/rendering/PROGRESS.md) | 各场景 Pygame 渲染器 |
| [rewards/](src/bluesky_pettingzoo/rewards/) | [README](src/bluesky_pettingzoo/rewards/README.md) | [PROGRESS](src/bluesky_pettingzoo/rewards/PROGRESS.md) | 奖励计算器 + 奖励分量 |
| [training/](src/bluesky_pettingzoo/training/) | [README](src/bluesky_pettingzoo/training/README.md) | [PROGRESS](src/bluesky_pettingzoo/training/PROGRESS.md) | 算法工厂、评估器、检查点、日志 |
| [utils/](src/bluesky_pettingzoo/utils/) | [README](src/bluesky_pettingzoo/utils/README.md) | [PROGRESS](src/bluesky_pettingzoo/utils/PROGRESS.md) | 几何计算、类型定义、Mixin |
| [wrappers/](src/bluesky_pettingzoo/wrappers/) | [README](src/bluesky_pettingzoo/wrappers/README.md) | [PROGRESS](src/bluesky_pettingzoo/wrappers/PROGRESS.md) | 包装器（单智能体、噪声、风场） |

### 开发工作流

**开始任务前**：先读取目标目录的 README.md 了解模块职责，再读 PROGRESS.md 了解当前进度。

**完成任务后**：更新目标目录的 PROGRESS.md（标记完成/进行中/阻塞），如有结构变更则同步更新 README.md。

## 工程规范

### 开发流程（TDD）

严格遵循测试驱动开发（TDD）：先写失败测试 -> 编写最小实现 -> 重构。每个功能模块必须有对应测试，覆盖率不低于 90%。

### 代码质量

- 类型注解：所有公开函数必须有完整类型注解
- 文档规范：公开类和方法必须有 Google 风格 docstring
- 代码风格：ruff lint + format
- 静态检查：mypy 类型检查
- 配置驱动：环境参数使用 YAML 管理，代码和配置彻底解耦

### 提交规范

使用 Conventional Commits 格式：`feat(scope): description`。类型包括 feat, fix, docs, style, refactor, test, chore。

### 基线 Agent

附带基线 Agent：`RandomAgent` 和基于 BlueSky 内置 LNAV/VNAV 的 `RuleBasedAgent`。

## 常用命令

```bash
pip install -e .              # 安装依赖
pytest tests/ -v              # 运行测试
ruff check src/ tests/        # 代码检查
ruff format src/ tests/       # 代码格式化
mypy src/bluesky_pettingzoo/  # 类型检查
```

## 项目验证命令

每次提交前必须通过以下全部检查：

```bash
# 1. 单元测试（不依赖 BlueSky 引擎）
pytest tests/ -v --ignore=tests/integration

# 2. 集成测试（需要 BlueSky 引擎，首次运行或环境变更后执行）
pytest tests/integration/ -v

# 3. 全量测试 + 覆盖率报告
pytest tests/ --cov=bluesky_pettingzoo --cov-report=term-missing

# 4. 代码风格检查（零 warning 标准）
ruff check src/ tests/

# 5. 代码格式化（dry-run 检查是否需要格式化）
ruff format --check src/ tests/

# 6. 静态类型检查
mypy src/bluesky_pettingzoo/

# 7. 一键全部验证（推荐在提交前执行）
ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/bluesky_pettingzoo/ && pytest tests/ --ignore=tests/integration -v
```

验证失败时的处理：
- 测试失败：先读测试文件理解期望行为，再修改实现代码
- ruff 报错：直接 `ruff format src/ tests/` 自动修复格式问题
- mypy 报错：根据类型提示修复，不使用 `# type: ignore` 绕过

## 文档自动维护

代码开发过程中，必须自动维护以下文档，不需要用户额外指令。

### 触发条件

每次代码变更（新增功能、修复 bug、重构）完成后，按以下规则更新文档：

| 变更类型 | 必须更新的文件 |
|---------|---------------|
| 修改/新增 `src/bluesky_pettingzoo/<module>/` 下的文件 | 该模块的 `PROGRESS.md` |
| 新增/删除/重命名目录 | 该目录的 `README.md` + 父目录的 `PROGRESS.md` |
| 修改训练流程、新增算法 | `train.md` |
| 新增/修改场景 | `architecture.md`（场景列表）+ 场景对应的 `README.md` |
| 修改奖励函数 | `rewards.yaml` + `architecture.md`（奖励系统部分） |
| 修改依赖 | `requirements.txt` / `pyproject.toml` + 根 `README.md` |
| 影响全局架构 | `architecture.md` + `CLAUDE.md` |

### 更新时机

- **每个功能完成时**：更新对应模块的 `PROGRESS.md`
- **每次 commit 前**：检查本次变更涉及的所有目录，更新对应的 `PROGRESS.md` 和 `README.md`
- **会话结束前**：扫描本次所有变更，确保文档同步。读取 `git diff --name-only` 识别变更文件，逐一更新

### PROGRESS.md 格式标准

每个代码目录的 `PROGRESS.md` 必须包含以下结构：

```markdown
# <模块名> 开发进度

## 状态总览

| 子模块/功能 | 状态 | 说明 |
|------------|------|------|
| xxx | ✅ 完成 | 简要说明 |
| xxx | 🔄 进行中 | 当前进度 |
| xxx | ⏸️ 阻塞 | 阻塞原因 |

## 待开发

- [ ] 功能描述
```

状态标记：
- `✅ 完成` — 功能已实现且测试通过
- `🔄 进行中` — 正在开发，未完成
- `⏸️ 阻塞` — 被依赖或技术问题阻塞，注明原因
- `❌ 废弃` — 已废弃的功能，注明废弃原因

### README.md 标准

每个目录的 `README.md` 必须包含：
1. 目录用途一句话说明
2. 目录结构树
3. 关键文件的用途说明
4. 使用/开发指引（如适用）

### 会话开始时

每次新会话开始，读取各代码目录的 `PROGRESS.md` 获取当前进度，作为后续工作的上下文。

## 交互规范

- 始终使用中文交流和生成文档
- 除删除文件和修改系统重要文件外，其他命令无需重复征求同意
- 每次执行完任务后简短总结本次对话做了什么
- 遇到多个任务和复杂任务时，自动拆分为多个子任务顺序执行
- 遇到复杂推理问题，像多个专业讨论一样思考，一步一步推理最后给出答案
- 用户提供的文档和建议需要辩证性批判性思考，不盲从
- 在代码迭代过程中，自动更新根目录下的 md 文件和代码配置文件
