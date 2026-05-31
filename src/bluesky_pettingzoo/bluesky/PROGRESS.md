# bluesky 开发进度

## 已完成

- [x] `wrapper.py` — BlueSky 同步 headless 模式封装
  - [x] 初始化：`bs.init(mode='sim', detached=True)`
  - [x] 飞机状态读取：`bs.traf.lat/lon/alt/hdg/tas/vs`
  - [x] 创建飞机：`bs.traf.cre()`
  - [x] 发送命令：`bs.stack.stack()`
  - [x] 推进仿真：`bs.sim.step()`
  - [x] 索引查找：`bs.traf.id2idx()`

## 待开发

无待开发项。
