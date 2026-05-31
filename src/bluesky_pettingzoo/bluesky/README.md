# bluesky/

BlueSky 仿真引擎封装模块，提供 Python 接口与 BlueSky 仿真器的同步通信。

## 文件说明

| 文件 | 用途 |
|------|------|
| `wrapper.py` | BlueSky headless 同步模式封装 |

## 核心接口

| 方法 | 说明 |
|------|------|
| `bs.init(mode='sim', detached=True)` | 初始化 BlueSky 为无 UI 模式 |
| `bs.traf.cre(acid, actype, lat, lon, alt, hdg, spd)` | 创建飞机 |
| `bs.stack.stack('HDG KL001 90')` | 发送命令流 |
| `bs.sim.step()` | 推进仿真一步 |
| `bs.traf.id2idx('KL001')` | 字符串 ID 转数组索引 |

## 状态读取

飞机状态通过 numpy 数组访问：`bs.traf.lat`, `bs.traf.lon`, `bs.traf.alt`, `bs.traf.hdg`, `bs.traf.tas`, `bs.traf.vs`。

## 设计要点

- Headless 同步模式：`env.step()` 推进固定仿真时间 Δt 后阻塞等待 Python 端读取状态
- 通过 `bs.stack.stack()` 批量写入命令，保证所有 Agent 动作在同一仿真步长内生效
