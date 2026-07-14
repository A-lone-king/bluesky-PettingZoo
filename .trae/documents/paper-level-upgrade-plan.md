# 论文级升级计划：bluesky-pettingzoo

> **目标**：ATM 领域研讨会/Workshop 投稿级别
> **硬件**：云 GPU（可用）
> **基线对比**：bluesky-gym 单智能体 + MARL 算法（MAPPO）
> **创建日期**：2026-07-14

---

## 一、现状分析

### 已完成（工程基础 90 分）
- 12 个 ATM 场景，943 个测试，覆盖率 >90%
- 4 种 RL 算法支持（PPO/SAC/TD3/DDPG）
- 完整的训练/评估/消融框架
- V4.0 奖励函数调优（23/24 features passing）

### 缺失（论文实验 50 分）
1. **训练规模不足**：当前仅 50k 步，论文需 500k+
2. **无多种子统计**：仅 seed=42，论文需 3-5 个种子
3. **缺关键基线**：无 bluesky-gym 单智能体对比，无 MARL 算法对比
4. **评估指标单薄**：仅 reward/arrival_rate/nmac_rate，缺安全性/效率指标
5. **无可视化结果**：无训练曲线图、对比表、统计检验
6. **消融实验无结果**：框架完整但未实际运行

---

## 二、实施计划（7 个 Phase）

### Phase 1: 多种子训练基础设施 (paper-001)

**目标**：支持多随机种子训练，生成统计显著性结果

**修改文件**：
- `scripts/train_ppo_scenarios.py` — 添加 `--seeds` 参数支持多种子批量训练
- `scripts/train_all_algos.py` — 添加多种子循环
- `src/bluesky_pettingzoo/training/checkpoint.py` — 检查点路径包含种子信息

**新增文件**：
- `scripts/train_multi_seed.py` — 多种子批量训练脚本

**具体改动**：
1. `train_ppo_scenarios.py` 的 `parse_args()` 添加 `--seeds` 参数（如 `--seeds 42 123 456 789 1024`）
2. `train_multi_seed.py` 调用 `train_scenario()` 循环每个种子，保存到 `models/{scenario}/{algorithm}/seed_{seed}/`
3. 每次训练保存 `training_log.csv` 和最终奖励到 JSON
4. 训练完成后自动计算 mean ± std 并保存到 `results/multi_seed/{scenario}_{algorithm}_summary.json`

**验证标准**：
- 5 个种子 × PPO × HorizontalCR 训练完成
- `results/multi_seed/HorizontalCR_PPO_summary.json` 包含 mean_reward, std_reward, mean_arrival_rate, mean_nmac_rate
- 单元测试：`test_multi_seed.py` 验证种子路径和汇总逻辑

---

### Phase 2: bluesky-gym 基线对比 (paper-002)

**目标**：将 bluesky-gym 单智能体方法作为基线，证明多智能体扩展的优势

**新增文件**：
- `src/bluesky_pettingzoo/baselines/bluesky_gym_adapter.py` — bluesky-gym 适配器
- `src/bluesky_pettingzoo/baselines/__init__.py`
- `scripts/run_bluesky_gym_baseline.py` — bluesky-gym 基线运行脚本
- `tests/test_bluesky_gym_adapter.py` — 适配器测试

**具体改动**：
1. 安装 bluesky-gym：`pip install bluesky-gym`（或 git clone）
2. `bluesky_gym_adapter.py` 实现：
   - `BlueSkyGymAdapter` 类，包装 bluesky-gym 的 7 个共有场景
   - 将 bluesky-gym 的单智能体 obs/reward 转换为统一格式
   - 支持 RVO2 rule-based 背景 agent（bluesky-gym 原生方式）
3. `run_bluesky_gym_baseline.py` 运行 bluesky-gym 的 PPO 训练（同等步数）
4. 评估脚本输出对比表：bluesky-pettingzoo MARL vs bluesky-gym 单智能体

**验证标准**：
- bluesky-gym 适配器可在 HorizontalCR、VerticalCR、WaypointNav 等 5 个共有场景上运行
- 对比表包含：mean_reward, arrival_rate, nmac_rate（双方统一指标）
- 单元测试：适配器 obs/reward 格式转换正确

---

### Phase 3: MAPPO 多智能体算法基线 (paper-003)

**目标**：添加 MAPPO（Multi-Agent PPO）作为 MARL 专用算法基线

**新增文件**：
- `src/bluesky_pettingzoo/baselines/mappo_agent.py` — MAPPO agent 实现
- `scripts/train_mappo.py` — MAPPO 训练脚本
- `tests/test_mappo_agent.py` — MAPPO agent 测试

**具体改动**：
1. 基于 PettingZoo 的 `pettingzoo.benchmark` 或自行实现 MAPPO：
   - 集中式 critic + 分散式 actor
   - 共享参数策略
   - 全局状态 = 所有 agent obs 拼接
2. `mappo_agent.py` 实现 `MAPPOAgent` 类：
   - `act(observations, action_spaces)` — 多智能体同时决策
   - `update(trajectories)` — 集中式训练
   - 支持 discrete 和 continuous 动作空间
3. `train_mappo.py` 训练 MAPPO 在 HorizontalCR/VerticalCR 上
4. 输出与 SB3 相同格式的评估结果

**替代方案**（如 MAPPO 实现复杂度过高）：
- 使用 `tianshou` 或 `pettingzoo.benchmark` 中的 MAPPO 实现
- 或退而使用 IPPO（Independent PPO，每个 agent 独立 PPO），实现更简单

**验证标准**：
- MAPPO/IPOPO 在 HorizontalCR 上训练 500k 步完成
- 评估结果与 PPO/Random/RuleBased 格式一致
- 单元测试：MAPPO agent act/update 逻辑正确

---

### Phase 4: 增强评估指标 (paper-004)

**目标**：从论文审稿角度补充关键评估指标

**修改文件**：
- `src/bluesky_pettingzoo/training/evaluator.py` — 扩展 `EvalResult` 数据类
- `scripts/evaluate_baselines.py` — 采集新指标

**新增文件**：
- `src/bluesky_pettingzoo/training/metrics.py` — 指标计算模块
- `tests/test_metrics.py` — 指标测试

**新增指标**：
```python
# metrics.py 新增指标计算
@dataclass
class ExtendedMetrics:
    # 安全性指标
    conflict_resolution_rate: float    # 冲突解脱成功率（冲突→安全分离）
    separation_violation_duration: float  # 平均分离违规持续时间（步）
    min_separation_distance: float    # 最小分离距离（NM）

    # 效率指标
    trajectory_efficiency: float       # 实际轨迹/最优轨迹长度比
    mean_deviation_distance: float    # 平均偏航距离（NM）
    fuel_consumption_estimate: float  # 燃油消耗估算（基于 OpenAP）

    # 时间指标
    mean_episode_length: float         # 平均 episode 长度
    mean_time_to_resolve: float       # 平均冲突解脱时间（步）
```

**具体改动**：
1. `metrics.py` 实现 `ExtendedMetrics` 计算逻辑
2. `evaluator.py` 的 `EvalResult` 添加 `extended_metrics` 字段
3. `_run_episodes()` 在每个 episode 中采集额外数据：
   - 每步记录所有飞机间距离（用于 min_separation）
   - 记录冲突状态变化（用于 conflict_resolution_rate）
   - 记录轨迹点（用于 trajectory_efficiency）
4. `evaluate_baselines.py` 输出扩展指标表

**验证标准**：
- 5 个新指标在 HorizontalCR 场景上正确计算
- 对比表包含 8+ 指标
- 单元测试：`test_metrics.py` 覆盖每个指标的计算逻辑

---

### Phase 5: 可扩展性实验 (paper-005)

**目标**：验证不同飞机数量下的性能表现

**新增文件**：
- `scripts/run_scalability_test.py` — 可扩展性实验脚本
- `tests/test_scalability.py` — 测试

**具体改动**：
1. `run_scalability_test.py` 实现：
   - 在 3/5/10/15/20 架飞机下分别评估 PPO 和基线
   - 每个配置运行 20 个 episode
   - 输出性能 vs 飞机数量曲线数据
2. 场景需支持动态飞机数量（已有 `num_aircraft_range`）
3. 输出 JSON 包含：
   ```json
   {
     "3_ac": {"mean_reward": ..., "nmac_rate": ..., "arrival_rate": ...},
     "5_ac": {...},
     "10_ac": {...},
     "15_ac": {...},
     "20_ac": {...}
   }
   ```

**验证标准**：
- 5 个飞机数量配置 × PPO + Random = 10 个评估完成
- 结果 JSON 保存到 `results/scalability/scalability_results.json`

---

### Phase 6: 消融实验执行 (paper-006)

**目标**：实际运行已有的消融实验框架，生成结果数据

**修改文件**：
- `scripts/run_ablation.py` — 确保脚本可运行
- `config/ablation_experiments.yaml` — 确认实验配置

**新增文件**：
- `scripts/run_reward_ablation.py` — 奖励组件消融脚本
- `tests/test_reward_ablation.py` — 消融测试

**具体改动**：
1. 运行 `run_ablation.py` 执行 14 个消融实验配置：
   - 7 个离散动作空间实验
   - 7 个连续动作空间实验
   - 每个实验训练 100k 步
2. `run_reward_ablation.py` 新增奖励组件消融：
   - 去掉 ConflictPenalty → 观察冲突率变化
   - 去掉 EfficiencyReward → 观察到达率变化
   - 去掉 SmoothnessPenalty → 观察动作平滑度变化
   - 去掉距离引导奖励 → 观察收敛速度变化
3. 生成 Markdown 和 JSON 报告到 `results/ablation/`

**验证标准**：
- 14 个动作空间消融实验完成，报告生成
- 4 个奖励组件消融实验完成，报告生成
- `results/ablation/ablation_report.md` 包含对比表

---

### Phase 7: 可视化与论文素材 (paper-007)

**目标**：生成论文可直接使用的图表和统计检验

**新增文件**：
- `scripts/generate_paper_figures.py` — 论文图表生成脚本
- `scripts/statistical_analysis.py` — 统计显著性检验

**具体改动**：
1. `generate_paper_figures.py` 生成以下图表：
   - 训练曲线图（reward vs timesteps，含多种子置信区间）
   - 算法对比柱状图（PPO/SAC/TD3/DDPG/MAPPO/Random/RuleBased）
   - bluesky-gym vs bluesky-pettingzoo 对比图
   - 可扩展性曲线图（性能 vs 飞机数量）
   - 消融实验对比图
2. `statistical_analysis.py` 实现：
   - Wilcoxon rank-sum test（非参数检验）
   - 效应量计算（Cohen's d）
   - 95% 置信区间
3. 输出到 `results/figures/` 和 `results/statistics/`

**图表规格**：
- 格式：PDF（矢量图）+ PNG（预览）
- 字体：Times New Roman 或 Computer Modern
- 尺寸：单栏 3.5" × 2.8"，双栏 7" × 4"
- 配色：色盲友好（避免红绿对比）

**验证标准**：
- 5 类图表全部生成到 `results/figures/`
- 统计检验结果保存到 `results/statistics/`
- 图表清晰可读，适合论文排版

---

## 三、feature_list.json 新增条目

以下 7 个新 feature 将添加到 `feature_list.json`：

| ID | 优先级 | 标题 | Phase |
|----|--------|------|-------|
| paper-001 | 1 | 多种子训练基础设施 | Phase 1 |
| paper-002 | 2 | bluesky-gym 基线对比 | Phase 2 |
| paper-003 | 3 | MAPPO 多智能体算法基线 | Phase 3 |
| paper-004 | 4 | 增强评估指标 | Phase 4 |
| paper-005 | 5 | 可扩展性实验 | Phase 5 |
| paper-006 | 6 | 消融实验执行 | Phase 6 |
| paper-007 | 7 | 可视化与论文素材 | Phase 7 |

---

## 四、执行顺序与依赖

```
Phase 1 (多种子训练) ─┐
                       ├──> Phase 7 (可视化)
Phase 2 (bluesky-gym) ─┤
                       │
Phase 3 (MAPPO) ───────┤
                       │
Phase 4 (增强指标) ────┤
                       │
Phase 5 (可扩展性) ────┤
                       │
Phase 6 (消融实验) ────┘
```

- Phase 1-6 可并行启动，Phase 7 依赖前 6 个的结果
- 建议执行顺序：Phase 1 → Phase 4 → Phase 2 → Phase 3 → Phase 5 → Phase 6 → Phase 7

---

## 五、假设与决策

1. **MAPPO 实现方式**：优先使用现有库（如 pettingzoo.benchmark 或 tianshou），如不可用则自行实现 IPPO（Independent PPO）作为简化替代
2. **bluesky-gym 安装**：通过 `pip install bluesky-gym` 或 git clone 安装，需要确认兼容性
3. **训练步数**：ATM 研讨会级别，500k 步足够；如后续投稿期刊再扩展到 1M+
4. **种子数量**：5 个种子（42, 123, 456, 789, 1024），足以计算置信区间
5. **统计检验**：使用 Wilcoxon rank-sum test（非参数），因为样本量小
6. **不修改核心环境代码**：所有改动限于 scripts/、baselines/、training/ 新增模块，不影响现有 943 个测试

---

## 六、验证步骤

### 每个 Phase 的通用验证
1. `ruff check src/ tests/` — 代码风格
2. `ruff format --check src/ tests/` — 格式化
3. `mypy src/bluesky_pettingzoo/` — 类型检查
4. `pytest tests/ --ignore=tests/integration -v` — 单元测试不退化
5. 新增测试全部通过

### 最终整体验证
1. 完整测试套件通过：`pytest tests/ -v --ignore=tests/integration`
2. `results/` 目录包含所有实验结果数据
3. `results/figures/` 包含 5 类论文图表
4. `results/statistics/` 包含统计检验结果
5. `feature_list.json` 中 7 个新 feature 全部标记为 passing
