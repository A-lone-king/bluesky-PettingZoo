# CLAUDE.md — bluesky-pettingzoo

你正在一个面向空中交通管理（ATM）领域的多智能体强化学习环境中工作。项目基于 BlueSky 仿真引擎 + PettingZoo ParallelEnv 标准，优先保证可靠完成、跨会话连续性和显式验证。

## 固定工作循环

每轮会话开始时：

1. 运行 `pwd`，确认当前在正确的仓库根目录
2. 读取 `claude-progress.md`
3. 读取 `feature_list.json`
4. 用 `git log --oneline -5` 查看最近提交
5. 运行 `./init.sh`（或手动执行 `pip install -e . && pytest tests/ -v --ignore=tests/integration`）
6. 检查单元测试 smoke test 是否已经损坏

然后只选择一个未完成功能，围绕它工作，直到它被验证通过，或者被明确记录为 blocked。

## 规则

- 同一时间只能有一个 active feature
- 没有可运行证据时，不要声称完成
- 不要通过重写功能清单来隐藏未完成工作
- 不要为了"看起来完成"而删除或削弱测试
- 以仓库内文件作为唯一事实来源
- 遵循 TDD：先写失败测试，再写实现，覆盖率 >= 90%
- 所有公开函数必须有完整类型注解
- 环境参数通过 YAML 配置管理，禁止硬编码
- 提交信息遵循 Conventional Commits 格式：`<type>(<scope>): <description>`
- 中文交流和文档

## 必需文件

- `feature_list.json`
- `claude-progress.md`
- `init.sh`
- `session-handoff.md` — 会话交接摘要，让下一轮快速了解现状
- `clean-state-checklist.md` — 收尾检查清单，确保仓库处于可开工状态
- `evaluator-rubric.md` — 评审评分表，评估 agent 输出质量
- `quality-document.md` — 质量快照，跟踪代码库健康度变化

## 项目验证命令

```bash
# 单元测试（不依赖 BlueSky 引擎）
pytest tests/ -v --ignore=tests/integration

# 集成测试（需要 BlueSky 引擎）
pytest tests/integration/ -v

# 代码风格检查
ruff check src/ tests/

# 代码格式化
ruff format --check src/ tests/

# 静态类型检查
mypy src/bluesky_pettingzoo/

# 一键全部验证
ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/bluesky_pettingzoo/ && pytest tests/ --ignore=tests/integration -v
```

## 完成门槛

只有在要求的验证成功且结果被记录后，功能状态才可以切换到 `passing`。

## 结束前

1. 更新 `claude-progress.md` 进度日志
2. 更新 `feature_list.json` 功能状态
3. 更新 `session-handoff.md` 交接信息
4. 更新 `quality-document.md` 质量快照（如有重要变更）
5. 按 `clean-state-checklist.md` 检查仓库状态
6. 记录仍然损坏或未验证的内容
7. 在仓库可安全恢复后提交
8. 给下一轮会话留下干净的重启路径
