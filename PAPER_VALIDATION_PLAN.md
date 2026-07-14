# 论文级实验验证方案

> **目标**：ATM 领域研讨会/Workshop 投稿级别
> **创建日期**：2026-07-14
> **预计总时间**：GPU ~12 小时 / CPU ~55 小时

---

## 一、前置条件检查

### 1.1 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | 任意 CUDA GPU | RTX 3060+ (12GB VRAM) |
| RAM | 16 GB | 32 GB |
| 存储 | 20 GB | 50 GB (模型+结果) |

### 1.2 环境检查

```bash
# 1. 检查 GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 2. 检查 BlueSky
python -c "import bluesky; print('BlueSky OK')"

# 3. 检查 Stable-Baselines3
python -c "import stable_baselines3; print('SB3 OK')"

# 4. 运行冒烟测试（约 1 分钟）
python scripts/train_smoke_test.py
```

**成功标准**：冒烟测试无错误完成。

---

## 二、分阶段验证计划

### Phase 0: 环境验证（10 分钟）

**目标**：确认所有依赖正常工作。

```bash
# 0.1 运行全部单元测试
pytest tests/ --ignore=tests/integration -v

# 0.2 代码质量检查
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/bluesky_pettingzoo/
```

**成功标准**：
- 测试通过 >= 1000 passed
- 无 ruff/mypy 错误

---

### Phase 1: 快速训练验证（30 分钟）

**目标**：验证训练流程可用，获得初步结果。

```bash
# 1.1 单场景快速训练（10k 步）
python scripts/train_ppo_scenarios.py `
    --scenario HorizontalCR `
    --timesteps 10000 `
    --num-aircraft 3

# 1.2 多算法快速训练（各 10k 步）
python scripts/train_multi_algo.py `
    --scenario HorizontalCR `
    --timesteps 10000

# 1.3 基线评估
python scripts/run_baselines.py `
    --scenario HorizontalCR `
    --episodes 20
```

**预期输出**：
- `models/HorizontalCR/PPO/final_model.zip` 存在
- `results/training_log.csv` 包含训练曲线
- 基线评估输出对比表

**成功标准**：
- 训练无错误完成
- 奖励值有提升趋势（不必显著）

---

### Phase 2: 多种子训练（GPU 约 2 小时）

**目标**：生成统计显著性数据。

```bash
# 2.1 HorizontalCR 多种子训练（5 seeds × 100k）
python scripts/train_multi_seed.py `
    --scenario HorizontalCR `
    --algorithm PPO `
    --timesteps 100000 `
    --seeds 42 123 456 789 1024 `
    --device cuda `
    --eval-episodes 20

# 2.2 VerticalCR 多种子训练
python scripts/train_multi_seed.py `
    --scenario VerticalCR `
    --algorithm PPO `
    --timesteps 100000 `
    --seeds 42 123 456 789 1024 `
    --device cuda `
    --eval-episodes 20
```

**预期输出**：
- `models/HorizontalCR/PPO/seed_{seed}/final_model.zip` (5 个模型)
- `results/multi_seed/HorizontalCR_PPO_summary.json`
- `results/multi_seed/VerticalCR_PPO_summary.json`

**成功标准**：
- mean_reward 标准差 < 30% 均值
- mean_arrival_rate > 0.3

---

### Phase 3: 多算法对比（GPU 约 4 小时）

**目标**：验证算法普适性。

```bash
# 3.1 4 种算法 × 2 场景训练
python scripts/train_all_algos.py `
    --scenarios HorizontalCR VerticalCR `
    --timesteps 200000 `
    --device cuda

# 3.2 MAPPO/IPPO 训练（如果 Ray 可用）
python scripts/train_multi_algo.py `
    --scenario HorizontalCR `
    --algorithm MAPPO `
    --timesteps 200000 `
    --device cuda
```

**预期输出**：
- `models/{scenario}/{algorithm}/final_model.zip` (8+ 模型)
- `results/comparison/{scenario}_{algorithm}_eval.json`

**成功标准**：
- 所有算法均能完成训练
- 至少 2 个算法 mean_reward > Random baseline

---

### Phase 4: 可扩展性实验（GPU 约 1 小时）

**目标**：验证不同飞机数量下的性能。

```bash
# 4.1 运行可扩展性实验
python -c "
from bluesky_pettingzoo.experiments.scalability_experiment import ScalabilityExperiment
from scripts.train_ppo_scenarios import make_scenario_env_factory

exp = ScalabilityExperiment(
    scenario='HorizontalCR',
    algorithms=['PPO', 'Random'],
    aircraft_counts=[3, 5, 10, 15, 20],
    seeds=[42],
    eval_episodes=10,
    save_dir='results/scalability',
    device='cuda'
)
results = exp.run()
print(results.summary())
"
```

**预期输出**：
- `results/scalability/HorizontalCR_PPO_scalability.json`
- `results/scalability/HorizontalCR_Random_scalability.json`

**成功标准**：
- 所有飞机数量配置均能完成评估
- 生成可扩展性曲线数据

---

### Phase 5: 消融实验（GPU 约 4 小时）

**目标**：验证动作空间和奖励组件的影响。

```bash
# 5.1 列出所有实验
python scripts/run_ablation.py --list

# 5.2 运行动作空间消融（选择关键实验）
python scripts/run_ablation.py `
    --experiment discrete_heading_only `
    --timesteps 50000 `
    --device cuda

python scripts/run_ablation.py `
    --experiment continuous_full `
    --timesteps 50000 `
    --device cuda

# 5.3 运行所有实验（完整版）
python scripts/run_ablation.py `
    --config config/ablation_experiments.yaml `
    --timesteps 50000 `
    --device cuda
```

**预期输出**：
- `results/ablation/{experiment}_results.json` (14 个文件)
- `results/ablation/ablation_report.md`

**成功标准**：
- 所有 14 个实验完成
- 报告包含对比表和结论

---

### Phase 6: 增强指标计算（10 分钟）

**目标**：补充论文级评估指标。

```bash
# 6.1 使用已有模型计算增强指标
python -c "
from bluesky_pettingzoo.training.metrics import MetricsCalculator
from pathlib import Path
import json

# 加载模型评估结果
results_path = Path('results/multi_seed/HorizontalCR_PPO_summary.json')
if results_path.exists():
    with open(results_path) as f:
        data = json.load(f)

    print('Extended metrics ready for calculation')
    print(f'Mean reward: {data[\"mean_reward\"]:.2f} +/- {data[\"std_reward\"]:.2f}')
    print(f'Arrival rate: {data[\"mean_arrival_rate\"]:.2%}')
    print(f'NMAC rate: {data[\"mean_nmac_rate\"]:.2%}')
else:
    print('No results found. Run Phase 2-5 first.')
"
```

---

### Phase 7: 可视化与论文素材生成（20 分钟）

**目标**：生成论文可直接使用的图表。

```bash
# 7.1 创建输出目录
New-Item -ItemType Directory -Force -Path results/figures, results/statistics

# 7.2 生成训练曲线
python -c "
from bluesky_pettingzoo.visualization.paper_figures import PaperFigureGenerator
import json
from pathlib import Path

gen = PaperFigureGenerator()

# 加载多种子结果
results = {}
for scenario in ['HorizontalCR', 'VerticalCR']:
    path = Path(f'results/multi_seed/{scenario}_PPO_summary.json')
    if path.exists():
        with open(path) as f:
            results[scenario] = json.load(f)

if results:
    print('Generating paper figures...')
    # gen.plot_training_curves(results, 'results/figures/training_curves.png')
    # gen.plot_comparison_bars(results, 'results/figures/comparison.png')
    print('Figures generation framework ready')
else:
    print('No results found. Run Phase 2-5 first.')
"
```

**预期输出**：
- `results/figures/training_curves.png`
- `results/figures/comparison.png`
- `results/figures/scalability.png`
- `results/figures/ablation.png`

**成功标准**：
- 图表清晰可读
- 适合论文排版（PDF 矢量图）

---

## 三、时间估算

| Phase | CPU 时间 | GPU 时间 | 优先级 |
|-------|----------|----------|--------|
| 0: 环境验证 | 10 分钟 | 10 分钟 | P0 |
| 1: 快速训练 | 1 小时 | 30 分钟 | P0 |
| 2: 多种子训练 | 10 小时 | 2 小时 | P1 |
| 3: 多算法对比 | 20 小时 | 4 小时 | P1 |
| 4: 可扩展性 | 3 小时 | 1 小时 | P2 |
| 5: 消融实验 | 20 小时 | 4 小时 | P2 |
| 6: 增强指标 | 10 分钟 | 10 分钟 | P2 |
| 7: 可视化 | 20 分钟 | 20 分钟 | P2 |

**总时间**：
- GPU: ~12 小时
- CPU: ~55 小时（不推荐）

---

## 四、依赖关系

```
Phase 0 ──┬──> Phase 1 ──┬──> Phase 2 ──┐
          │              │              │
          │              └──> Phase 3 ──┼──> Phase 7
          │                             │
          └──> Phase 4 ─────────────────┤
                                        │
          └──> Phase 5 ─────────────────┤
                                        │
          └──> Phase 6 ─────────────────┘
```

**并行建议**：
- Phase 2-6 可并行执行（如果有多个 GPU）
- Phase 7 依赖 Phase 2-6 的结果

---

## 五、成功标准汇总

| 指标 | 目标值 |
|------|--------|
| 多种子训练 | 5 seeds 完成，std < 30% mean |
| 多算法对比 | 4 算法完成，>= 2 算法优于 Random |
| 可扩展性 | 5 个飞机数量配置完成 |
| 消融实验 | 14 个实验完成 |
| 训练模型 | >= 10 个检查点 |
| 论文图表 | 5 类图表生成 |

---

## 六、排错指南

### 6.1 CUDA 内存不足

```bash
# 降低 batch size
python scripts/train_multi_seed.py `
    --timesteps 100000 `
    --batch-size 64  # 默认 256
```

### 6.2 BlueSky 初始化超时

```bash
# 首次运行需加载导航数据库（1-2 分钟）
# 后续运行会缓存，速度正常
$env:BLUESKY_CACHE_DIR = "$env:USERPROFILE\.bluesky_cache"
```

### 6.3 训练不收敛

```bash
# 检查奖励配置
Get-Content config/rewards.yaml

# 调整学习率
python scripts/train_ppo_scenarios.py `
    --learning-rate 0.0003 `
    --timesteps 200000
```

### 6.4 无 GPU 环境

```bash
# 使用 CPU（慢，仅用于调试）
python scripts/train_ppo_scenarios.py `
    --device cpu `
    --timesteps 10000
```

### 6.5 PowerShell 命令语法

Windows PowerShell 使用反引号（`）作为换行符，而非 bash 的反斜杠（\）：

```powershell
# 正确示例
python scripts/train_multi_seed.py `
    --scenario HorizontalCR `
    --timesteps 100000

# 或使用单行
python scripts/train_multi_seed.py --scenario HorizontalCR --timesteps 100000
```

---

## 七、验证清单

完成以下检查后，项目达到论文级别：

- [ ] Phase 0: 单元测试全通过
- [ ] Phase 1: 快速训练无错误
- [ ] Phase 2: 5 seeds x 2 scenarios 完成
- [ ] Phase 3: 4 算法对比完成
- [ ] Phase 4: 可扩展性数据生成
- [ ] Phase 5: 消融实验报告生成
- [ ] Phase 6: 增强指标计算完成
- [ ] Phase 7: 论文图表生成
- [ ] `results/` 目录包含完整实验数据
- [ ] `models/` 目录包含训练模型
- [ ] 所有 feature_list.json 状态为 passing

---

## 八、下一步建议

1. **如果只有 CPU**：先完成 Phase 0-1，确认流程正确，然后在云 GPU 上执行 Phase 2-7
2. **如果有云 GPU**：建议使用 tmux/screen 后台运行长时间任务
3. **存储管理**：定期清理中间检查点，只保留最终模型

---

## 九、参考文档

- [论文级升级计划](.trae/documents/paper-level-upgrade-plan.md)
- [功能清单](feature_list.json)
- [进度日志](Codex-progress.md)
- [README](README.md)