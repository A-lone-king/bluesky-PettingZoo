# 奖励函数调优报告

## 1. 训练结果分析

### 1.1 HorizontalCR（水平冲突场景）

| 指标 | 值 | 评价 |
|------|-----|------|
| Episodes | 118,792 | - |
| Total Timesteps | 5,001,193 | - |
| Final Reward Mean | -56.17 | 严重负奖励 |
| Final Reward Std | 38.97 | 高方差 |
| Max Reward | -0.63 | 从未获得正奖励 |
| Avg Episode Length | 42.1 | 接近 max_steps=50 |

**诊断**：
- Agent 学会了"拖延"策略（episode_length=42，接近上限50）
- 从未成功到达目标（max_reward=-0.63）
- 可能原因：碰撞惩罚太重，导致 agent 不敢采取行动

### 1.2 VerticalCR（垂直冲突场景）

| 指标 | 值 | 评价 |
|------|-----|------|
| Episodes | 83,144 | - |
| Total Timesteps | 9,999,792 | - |
| Final Reward Mean | -0.87 | 接近零 |
| Final Reward Std | 10.19 | 中等方差 |
| Max Reward | 2.99 | 有成功 episode |
| Avg Episode Length | 0.0 | 数据异常 |

**诊断**：
- 基本收敛，有正奖励出现
- 垂直冲突相对容易解决（改变高度即可）
- Avg Episode Length=0 可能是数据记录问题

## 2. 问题根因分析

### 2.1 奖励函数配置（config/rewards.yaml）

```yaml
components:
  conflict:
    nmac_penalty: -500.0      # 碰撞一次 = -500
    warning_penalty: -50.0     # 警告 = -50
    separation_penalty: -20.0  # 间隔不足 = -20

  efficiency:
    arrival_reward: 10.0       # 到达目标 = +10
    step_penalty: -0.01        # 每步惩罚 = -0.01
```

### 2.2 奖励失衡问题

| 事件 | 奖励 | 相对比值 |
|------|------|----------|
| NMAC 碰撞 | -500 | 1x（基准） |
| Warning 警告 | -50 | 0.1x |
| 到达目标 | +10 | 0.02x |
| 每步拖延 | -0.01 | 0.00002x |

**问题**：
1. **惩罚/奖励比 = 50:1** — 碰撞一次的惩罚等于成功到达50次的奖励
2. **Agent 学会"不行动"** — 采取任何动作都有风险，不如原地等待
3. **缺少距离引导** — 没有"靠近目标"的渐进奖励

### 2.3 对比 bluesky-gym 的奖励设计

| 组件 | bluesky-gym | bluesky-pettingzoo | 建议 |
|------|-------------|-------------------|------|
| 碰撞惩罚 | -1.0 | -500.0 | 降低100倍 |
| 到达奖励 | +1.0 | +10.0 | 保持或提高 |
| 距离引导 | 有 | 无 | 需要添加 |
| 每步惩罚 | -0.01 | -0.01 | 保持 |

## 3. V4.0 改进计划

### 3.1 Phase 1：调整奖励平衡（P0）

**目标**：让 agent 敢于采取行动

| 改进项 | 当前值 | 目标值 | 理由 |
|--------|--------|--------|------|
| nmac_penalty | -500 | -50 | 降低10倍，减少恐惧 |
| warning_penalty | -50 | -10 | 降低5倍 |
| separation_penalty | -20 | -5 | 降低4倍 |
| arrival_reward | +10 | +100 | 提高10倍，增强正向引导 |
| step_penalty | -0.01 | -0.005 | 降低2倍，减少拖延惩罚 |

### 3.2 Phase 2：添加距离引导奖励（P1）

**目标**：让 agent 学会"靠近目标"

新增配置项：
```yaml
efficiency:
  distance_reward_scale: 0.5  # 每靠近目标 1nm 奖励 0.5
  distance_threshold_nm: 50   # 距离超过此值不给奖励
```

### 3.3 Phase 3：简化场景验证（P1）

**目标**：用最简单场景快速验证调参效果

```bash
# 用 2 架飞机、30 步限制快速测试
python scripts/train_ppo_scenarios.py \
    --scenario HorizontalCR \
    --timesteps 100000 \
    --num-aircraft 2 \
    --max-steps 30
```

### 3.4 Phase 4：多算法对比验证（P2）

**目标**：确认调参后 PPO/SAC/TD3 都能收敛

```bash
python scripts/train_all_algos.py \
    --timesteps 500000 \
    --scenarios HorizontalCR VerticalCR
```

## 4. 验证标准

| 指标 | 目标值 | 说明 |
|------|--------|------|
| HorizontalCR Final Reward | > -10 | 从 -56 提升到 -10 以内 |
| VerticalCR Final Reward | > 0 | 保持正奖励 |
| Arrival Rate | > 10% | 至少 10% 的 episode 成功到达 |
| NMAC Rate | < 5% | 碰撞率低于 5% |
| Training Curve | 上升趋势 | reward 随 timestep 增加 |

## 5. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 惩罚降太低导致 agent 不学避碰 | 高 | 保留 warning 作为安全网 |
| 奖励提太高导致 agent 不顾安全 | 中 | arrival_reward 上限 100 |
| 距离引导引入局部最优 | 中 | 距离阈值 50nm |
| 调参后原有测试失败 | 低 | 更新测试断言值 |

## 6. 执行顺序

1. **Phase 1**：修改 rewards.yaml 调整奖励平衡
2. **Phase 3**：用简化场景快速验证（10万步）
3. **Phase 2**：添加距离引导奖励
4. **Phase 4**：多算法对比验证（50万步）

---

**制定日期**：2026-06-12
**基于数据**：HorizontalCR 5M steps, VerticalCR 10M steps 训练结果
