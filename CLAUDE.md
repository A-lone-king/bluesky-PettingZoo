# CLAUDE.md — bluesky-pettingzoo

## remember

Read existing files before writing. Don't re-read unless changed.

Thorough in reasoning, concise in output.

Skip files over 100KB unless required.

No sycophantic openers or closing fluff.

No emojis or em-dashes.

Do not guess APIs, versions, flags, commit SHAs, or package names. Verify by reading code or docs before asserting.

## 项目概述

将 BlueSky 空中交通仿真平台从单智能体环境（bluesky-gym）扩展为多智能体环境，基于 PettingZoo ParallelEnv 标准，专注于空中交通管理（ATM）领域的多智能体强化学习研究。BlueSky 是底层仿真引擎，本项目在其上构建多智能体 RL 接口。

## 用户背景

- 空中交通管理方向博士在读，本科计算机科学
- 研究方向：低空经济规划，特别是空中交通管理
- 长期维护此仓库，作为研究成果的基础设施

## 仓库结构

```
bluesky-PettingZoo/
├── bluesky-gym/      # BlueSky-Gym 单智能体环境（clone，参考实现）
├── PettingZoo/       # PettingZoo 多智能体框架（clone）
├── bluesky.wiki/     # BlueSky 官方文档
├── src/              # 本项目核心代码
├── config/           # YAML 配置文件（default, rewards, scenarios）
├── scripts/          # 训练和评估脚本
└── tests/            # 测试套件（1026 个用例）
```

BlueSky 仿真引擎通过 `pip install "bluesky-simulator[full]"` 安装，无需本地 clone。

## 技术架构

### 核心设计决策

1. **基于 PettingZoo ParallelEnv**：空管场景是时空并行系统，所有 Agent 在同一仿真步长内同时观测、同时动作，使用 `ParallelEnv` 而非 AEC。

2. **Headless 同步模式**：BlueSky 以无 UI 模式运行，`env.step(actions)` 推进固定仿真时间 Δt 后阻塞等待 Python 端读取状态。

3. **动作批处理**：所有 Agent 的 action 在 `step()` 中打包为一条命令流一次性写入 BlueSky。

4. **异构观测空间**：使用 `gymnasium.spaces.Dict` 定义观测空间，支持动态变化的飞机数量，通过唯一字符串 ID 索引。

5. **奖励函数模块化**：独立 `RewardCalculator` 类，支持动态注册不同惩罚项。

### 重构后的模块化设计

- **RewardComponent 基类**：提供 `get_config()` 辅助方法和自动 `reset()` 机制
- **EnvWrapperMixin**：统一包装器委托实现，减少重复代码
- **BaseScenario**：提供默认冲突配置和工具方法（`generate_agent_ids()`, `get_center_point()`）
- **DictBackedMixin**：统一字典兼容接口，消除配置类中的重复模式
- **BaseRenderer**：提供通用渲染逻辑和 bounds 管理

### BlueSky 关键接口

- 初始化：`bs.init(mode='sim', detached=True)`
- 飞机状态：`bs.traf.lat/lon/alt/hdg/tas/vs`（numpy 数组）
- 创建飞机：`bs.traf.cre(acid, actype, aclat, aclon, acalt, achdg, acspd)`
- 发送命令：`bs.stack.stack('HDG KL001 90')`
- 推进仿真：`bs.sim.step()`
- 索引查找：`bs.traf.id2idx('KL001')` → 数组索引

### PettingZoo ParallelEnv 接口

```python
class ParallelEnv:
    agents: list[AgentID]
    possible_agents: list[AgentID]

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        ...

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
        # returns (observations, rewards, terminations, truncations, infos)
        ...

    def observation_space(self, agent) -> Space: ...
    def action_space(self, agent) -> Space: ...
```

### 参考：bluesky-gym 现有环境

- `horizontal_cr_env.py` — 水平冲突解脱
- `vertical_cr_env.py` — 垂直冲突解脱
- `sector_cr_env.py` — 扇区冲突解脱
- `descent_env.py` — 下降阶段
- `merge_env.py` — 汇合冲突
- `static_obstacle_env.py` — 禁飞区规避
- `plan_waypoint_env.py` — 顺序航路点导航

## 工程规范

### 开发流程（TDD）

本项目严格采用测试驱动开发（TDD）流程：

1. **红（Red）**：先写失败的测试用例，定义期望行为
2. **绿（Green）**：编写最小实现使测试通过
3. **重构（Refactor）**：优化代码结构，保持测试通过

TDD 要求：
- 每个功能模块必须有对应的测试文件
- 测试覆盖率不低于 90%
- 新功能必须先有测试，再有实现
- Bug 修复必须先写复现测试

### 代码质量标准

- **类型注解**：所有公开函数必须有完整的类型注解
- **文档规范**：公开类和方法必须有 docstring（Google 风格）
- **代码风格**：使用 ruff 进行 lint 和格式化
- **静态检查**：使用 mypy 进行类型检查
- **配置驱动**：环境参数使用 YAML 管理
- **代码和配置彻底解耦**

### 提交规范

使用 Conventional Commits 格式：

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

类型（type）：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档变更
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具变更

示例：
- `feat(envs): add conflict resolution environment`
- `fix(wrapper): handle aircraft not found error`
- `test(rewards): add unit tests for conflict penalty`

### 基线 Agent

附带基线 Agent：至少提供 `RandomAgent` 和基于 BlueSky 内置 LNAV/VNAV 的 `RuleBasedAgent`

## 常用命令

```bash
# 安装依赖
pip install -e .

# 运行测试
pytest tests/ -v

# 代码检查
ruff check src/ tests/

# 代码格式化
ruff format src/ tests/

# 类型检查
mypy src/bluesky_pettingzoo/
```

## 交互规范

- 始终使用中文交流和生成文档
- 除删除文件和修改系统重要文件外，其他命令无需重复征求同意
- 每次执行完任务后简短总结本次对话做了什么
- 遇到多个任务和复杂任务时，自动拆分为多个子任务顺序执行
- 遇到复杂推理问题，像多个专业讨论一样思考，一步一步推理最后给出答案
- 用户提供的文档和建议需要辩证性批判性思考，不盲从
- 在代码迭代过程中，自动更新根目录下的 md 文件和代码配置文件

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (60-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk go test             # Go test failures only (90%)
rtk jest                # Jest failures only (99.5%)
rtk vitest              # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk pytest              # Python test failures only (90%)
rtk rake test           # Ruby test failures only (90%)
rtk rspec               # RSpec test failures only (60%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%). Format flags (-c, -l, -L, -o, -Z) run raw.
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->