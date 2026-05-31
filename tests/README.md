# tests/

测试套件目录，包含单元测试、集成测试和端到端测试。

## 测试结构

```
tests/
├── conftest.py                    # 全局 fixture 定义
├── helpers/
│   └── env_factory.py            # 测试环境工厂
├── integration/                   # 集成测试（需要 BlueSky 引擎）
│   ├── test_bluesky_real.py       # BlueSky 真实引擎集成测试
│   ├── test_component_integration.py  # 组件集成测试
│   ├── test_backward_compat.py    # 向后兼容性测试
│   ├── test_scenario_e2e.py       # 场景端到端测试
│   ├── test_performance.py        # 性能测试
│   └── test_*.py                  # 各场景集成测试
└── test_*.py                      # 单元测试
```

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单元测试（不依赖 BlueSky）
pytest tests/ -v --ignore=tests/integration

# 运行集成测试（需要 BlueSky 引擎）
pytest tests/integration/ -v

# 运行特定测试文件
pytest tests/test_reward_calculator.py -v

# 查看覆盖率
pytest tests/ --cov=bluesky_pettingzoo --cov-report=html
```

## 测试规范

- 测试覆盖率不低于 90%
- 每个功能模块必须有对应测试文件
- 新功能必须先写测试，再写实现（TDD）
- Bug 修复必须先写复现测试
- 测试文件命名：`test_<module_name>.py`
- 测试类命名：`Test<ClassName>`
- 测试方法命名：`test_<behavior>`
