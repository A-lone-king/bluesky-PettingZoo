# utils/

通用工具模块，提供几何计算、类型定义和 Mixin 类。

## 文件说明

| 文件 | 用途 |
|------|------|
| `geometry.py` | 几何计算工具（距离、方位角、位置投影） |
| `types.py` | 类型定义（AgentID, AircraftState, ConflictLevel 等） |
| `mixin.py` | Mixin 类（DictBackedMixin 等，统一字典兼容接口） |

## 核心函数

### geometry.py

| 函数 | 说明 |
|------|------|
| `haversine_distance(lat1, lon1, lat2, lon2)` | 两点间大圆距离（米） |
| `bearing(lat1, lon1, lat2, lon2)` | 两点间方位角（度） |
| `project_position(lat, lon, hdg, dist)` | 沿航向投影位置 |
| `relative_position(lat, lon, ref_lat, ref_lon)` | 相对参考点位置 |
| `point_at_distance(lat, lon, hdg, dist)` | 沿航向指定距离处的坐标 |

### types.py

| 类型 | 说明 |
|------|------|
| `AgentID` | Agent 唯一标识符（字符串） |
| `AircraftState` | 飞机状态数据结构 |
| `ConflictLevel` | 冲突等级枚举 |
| `DiscreteAction` | 离散动作类型 |
