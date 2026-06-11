# observations/

观测空间管理模块，负责构建、过滤和归一化 Agent 的观测数据。

## 文件说明

| 文件 | 用途 |
|------|------|
| `manager.py` | 观测管理器，构建 Dict 空间，支持动态 Agent 数量 |
| `filters.py` | 感知范围过滤器，只保留 Agent 可观测范围内的目标 |
| `normalizer.py` | 观测归一化，将各维度缩放到统一范围 |

## 设计要点

- 使用 `gymnasium.spaces.Dict` 定义观测空间，通过唯一字符串 ID 索引
- 支持动态变化的飞机数量（部分可观测场景）
- 感知范围过滤：根据配置的最大感知距离裁剪观测
- 归一化：将经纬度、高度、航向等不同量纲的值缩放到 [0, 1]

## 零填充与 Mask 机制

由于不同场景的飞机数量不同，且同一场景在不同 episode 中可观测飞机数量也会变化，观测空间采用**零填充（zero-padding）** + **mask** 的标准做法：

### 观测结构

```python
observation = {
    "self_state": np.array([9], dtype=np.float32),              # 自身状态
    "other_aircraft": np.array([max_obs, 12], dtype=np.float32), # 他机状态
    "other_aircraft_mask": np.array([max_obs], dtype=np.int8),   # 他机 mask
    "goal": np.array([4], dtype=np.float32),                     # 目标航路点
    "conflict_state": np.array([3], dtype=np.float32),           # 冲突状态 one-hot
}
```

### Mask 含义

- `mask[i] = 1`：第 `i` 个槽位包含真实飞机数据
- `mask[i] = 0`：第 `i` 个槽位是零填充，应被忽略

### 为什么需要 Mask

1. **固定形状要求**：Gymnasium 空间要求固定形状，但实际可观测飞机数量动态变化
2. **批量处理友好**：固定形状便于 GPU 并行处理
3. **PettingZoo 标准**：多智能体环境的通用做法

### 训练时使用 Mask

```python
# 方式 1：元素级乘法（推荐）
masked_obs = observation["other_aircraft"] * observation["other_aircraft_mask"][:, None]

# 方式 2：PyTorch/TensorFlow 掩码
mask_tensor = torch.from_numpy(observation["other_aircraft_mask"]).bool()
masked_obs = other_aircraft_tensor[mask_tensor]

# 方式 3：直接输入网络（网络内部处理 mask）
# 某些网络架构（如 Transformer）可以在注意力层处理 mask
```

### 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `observation.max_observable_aircraft` | 10 | 最大可观测飞机数量，决定 `other_aircraft` 形状 |
| `observation.perception_radius_nm` | 20.0 | 感知半径（海里），超出此距离的飞机不进入观测 |

### 障碍物观测

类似地，障碍物观测也使用零填充 + mask：

```python
obstacles = {
    "position": np.array([max_obstacles, 4], dtype=np.float32),  # [dist, bear_cos, bear_sin, radius]
    "mask": np.array([max_obstacles], dtype=np.int8),
}
```

## 扩展方式

新增观测维度时，在对应场景的 `get_observation()` 中添加字段，并在 `manager.py` 中注册新的空间定义。
