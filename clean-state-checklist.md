# 干净状态检查清单

每次会话结束前过一遍，确保仓库处于下一轮可以直接开工的状态。

## 检查什么

- [ ] `pip install -e .` 安装路径仍然可用
- [ ] `pytest tests/ -v --ignore=tests/integration` 单元测试仍然通过
- [ ] `pytest tests/integration/ -v` 集成测试仍然通过（如已验证）
- [ ] `ruff check src/ tests/` 代码检查通过
- [ ] `ruff format --check src/ tests/` 格式检查通过
- [ ] `mypy src/bluesky_pettingzoo/` 类型检查通过
- [ ] 当前进度已记录到 `claude-progress.md`
- [ ] `feature_list.json` 状态真实反映了 passing 和未验证的边界
- [ ] 没有任何半成品代码处于未记录状态
- [ ] `session-handoff.md` 已更新交接信息
- [ ] 下一轮会话无需人工修复即可继续
