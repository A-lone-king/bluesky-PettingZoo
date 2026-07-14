# 远程训练部署指南

> **给朋友的使用文档**
> **项目**：bluesky-pettingzoo - 空中交通管理多智能体强化学习环境
> **创建日期**：2026-07-14

---

## 一、前置要求

### 1.1 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | CUDA 兼容 GPU | RTX 3060+ (12GB VRAM) |
| RAM | 16 GB | 32 GB |
| 存储 | 30 GB 可用空间 | 50 GB |
| CPU | 4 核 | 8 核+ |

### 1.2 软件要求

- **操作系统**：Ubuntu 20.04/22.04 LTS 或 Windows 10/11
- **Python**：3.11 或更高版本
- **Git**：用于克隆仓库
- **CUDA**：12.0+（如果使用 GPU）

### 1.3 检查 GPU 环境

**Linux:**
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version
```

**Windows:**
```powershell
nvidia-smi
```

**预期输出**：应显示 GPU 型号和 CUDA 版本

---

## 二、克隆项目

### 2.1 克隆仓库

```bash
# 克隆项目
git clone https://github.com/A-lone-king/bluesky-PettingZoo.git
cd bluesky-PettingZoo
```

### 2.2 确认分支

```bash
# 确保在正确的分支
git branch
git status
```

---

## 三、环境配置

### 3.1 创建虚拟环境

**Linux/Mac:**
```bash
# 创建虚拟环境
python3.11 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

**Windows:**
```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1
```

### 3.2 安装项目依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装项目
pip install -e .

# 安装 BlueSky 仿真引擎
pip install "bluesky-simulator[full]"

# 安装开发依赖（可选，用于测试）
pip install -r requirements-dev.txt
```

### 3.3 安装 CUDA 版 PyTorch（重要！）

**默认安装的是 CPU 版本，需要手动安装 CUDA 版本：**

```bash
# 卸载 CPU 版本
pip uninstall torch torchvision torchaudio -y

# 安装 CUDA 12.8 版本（RTX 50 系列）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 或 CUDA 12.6 版本（RTX 40/30 系列）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 或 CUDA 12.1 版本（RTX 20/30 系列）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3.4 验证 GPU 可用

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**预期输出**：
```
CUDA available: True
Device: NVIDIA GeForce RTX 4090
```

---

## 四、环境验证

### 4.1 运行冒烟测试

```bash
# 快速验证（约 1 分钟）
python scripts/train_smoke_test.py
```

### 4.2 运行单元测试（可选）

```bash
# 运行单元测试（约 5 分钟）
pytest tests/ --ignore=tests/integration -v
```

**预期输出**：应显示 1000+ tests passed

---

## 五、运行训练

### 5.1 快速训练验证（推荐先运行）

**目的**：验证训练流程正常工作（约 30 分钟 GPU）

```bash
# Linux/Mac
python scripts/train_ppo_scenarios.py \
    --scenario HorizontalCR \
    --timesteps 10000 \
    --num-aircraft 3 \
    --device cuda

# Windows PowerShell
python scripts/train_ppo_scenarios.py `
    --scenario HorizontalCR `
    --timesteps 10000 `
    --num-aircraft 3 `
    --device cuda
```

**预期输出**：
- 训练进度条显示
- `models/HorizontalCR/PPO/` 目录生成

### 5.2 多种子训练（核心实验）

**目的**：生成统计显著性数据

```bash
# Linux/Mac
python scripts/train_multi_seed.py \
    --scenario HorizontalCR \
    --algorithm PPO \
    --timesteps 100000 \
    --seeds 42 123 456 789 1024 \
    --device cuda \
    --eval-episodes 20

# Windows PowerShell
python scripts/train_multi_seed.py `
    --scenario HorizontalCR `
    --algorithm PPO `
    --timesteps 100000 `
    --seeds 42 123 456 789 1024 `
    --device cuda `
    --eval-episodes 20

# 同样运行 VerticalCR
python scripts/train_multi_seed.py `
    --scenario VerticalCR `
    --algorithm PPO `
    --timesteps 100000 `
    --seeds 42 123 456 789 1024 `
    --device cuda `
    --eval-episodes 20
```

**预计时间**：每种场景约 2 小时（共 4 小时）

### 5.3 多算法对比（可选）

```bash
# Linux/Mac
python scripts/train_all_algos.py \
    --scenarios HorizontalCR VerticalCR \
    --timesteps 200000 \
    --device cuda

# Windows PowerShell
python scripts/train_all_algos.py `
    --scenarios HorizontalCR VerticalCR `
    --timesteps 200000 `
    --device cuda
```

**预计时间**：约 4 小时

### 5.4 消融实验（可选）

```bash
# 列出所有实验
python scripts/run_ablation.py --list

# 运行所有消融实验（约 4 小时）
python scripts/run_ablation.py `
    --config config/ablation_experiments.yaml `
    --timesteps 50000 `
    --device cuda
```

---

## 六、后台运行训练（推荐）

### 6.1 使用 nohup（Linux）

```bash
# 创建日志目录
mkdir -p logs

# 后台运行
nohup python scripts/train_multi_seed.py \
    --scenario HorizontalCR \
    --algorithm PPO \
    --timesteps 100000 \
    --seeds 42 123 456 789 1024 \
    --device cuda \
    > logs/train_horizontal_cr.log 2>&1 &

# 查看日志
tail -f logs/train_horizontal_cr.log

# 查看进程
ps aux | grep train_multi_seed
```

### 6.2 使用 screen（Linux）

```bash
# 创建新 screen 会话
screen -S training

# 运行训练
python scripts/train_multi_seed.py \
    --scenario HorizontalCR \
    --algorithm PPO \
    --timesteps 100000 \
    --seeds 42 123 456 789 1024 \
    --device cuda

# 按 Ctrl+A+D 分离会话
# 重新连接：screen -r training
```

### 6.3 Windows 后台运行

```powershell
# 使用 Start-Process
Start-Process -NoNewWindow python -ArgumentList "scripts/train_multi_seed.py --scenario HorizontalCR --algorithm PPO --timesteps 100000 --seeds 42 123 456 789 1024 --device cuda" -RedirectStandardOutput logs/train.log -RedirectStandardError logs/error.log
```

---

## 七、监控训练进度

### 7.1 查看训练日志

```bash
# 查看最新日志
tail -f results/training_log.csv

# 或查看完整日志
cat results/training_log.csv
```

### 7.2 查看模型文件

```bash
# 检查模型目录
ls -la models/

# 检查特定场景
ls -la models/HorizontalCR/PPO/
```

### 7.3 查看结果文件

```bash
# 检查结果目录
ls -la results/

# 检查多种子汇总
cat results/multi_seed/HorizontalCR_PPO_summary.json
```

---

## 八、收集训练结果

### 8.1 需要收集的文件

训练完成后，请打包以下目录：

```bash
# 创建结果打包目录
mkdir -p training_results

# 复制模型文件
cp -r models/* training_results/

# 复制结果文件
cp -r results/* training_results/

# 打包（Linux）
tar -czvf training_results.tar.gz training_results/

# 或打包（Windows）
Compress-Archive -Path training_results -DestinationPath training_results.zip
```

### 8.2 重要文件清单

| 文件/目录 | 说明 |
|----------|------|
| `models/{scenario}/{algorithm}/seed_{seed}/final_model.zip` | 训练好的模型 |
| `results/multi_seed/{scenario}_{algorithm}_summary.json` | 多种子汇总结果 |
| `results/training_log.csv` | 训练曲线数据 |
| `results/ablation/*.json` | 消融实验结果 |

### 8.3 结果文件示例

**多种子汇总（JSON 格式）：**
```json
{
  "scenario": "HorizontalCR",
  "algorithm": "PPO",
  "seeds": [42, 123, 456, 789, 1024],
  "mean_reward": -45.2,
  "std_reward": 12.3,
  "mean_arrival_rate": 0.65,
  "mean_nmac_rate": 0.08,
  "seed_results": [...]
}
```

---

## 九、常见问题

### 9.1 CUDA 内存不足

```bash
# 降低 batch size
python scripts/train_multi_seed.py \
    --timesteps 100000 \
    --n-steps 2048 \
    --batch-size 64 \
    --device cuda
```

### 9.2 BlueSky 初始化慢

首次运行 BlueSky 需要加载导航数据库，可能需要 1-2 分钟。后续运行会自动缓存。

```bash
# 设置缓存目录（可选）
export BLUESKY_CACHE_DIR=~/.bluesky_cache  # Linux
$env:BLUESKY_CACHE_DIR = "$env:USERPROFILE\.bluesky_cache"  # Windows
```

### 9.3 训练中断恢复

如果训练中断，可以从检查点恢复：

```bash
python scripts/train_ppo_scenarios.py \
    --scenario HorizontalCR \
    --resume models/HorizontalCR/PPO/checkpoint_50000.zip \
    --timesteps 100000
```

### 9.4 模块导入错误

```bash
# 确保在项目根目录
cd bluesky-PettingZoo

# 确保虚拟环境激活
source venv/bin/activate  # Linux
.\venv\Scripts\Activate.ps1  # Windows

# 重新安装
pip install -e .
```

### 9.5 权限错误（Windows）

```powershell
# 如果遇到执行策略错误
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 十、联系方式

如有问题，请联系项目负责人：
- GitHub: https://github.com/A-lone-king/bluesky-PettingZoo
- 提交 Issue: https://github.com/A-lone-king/bluesky-PettingZoo/issues

---

## 十一、快速命令参考

### 验证环境
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python scripts/train_smoke_test.py
```

### 快速训练（10k 步，约 5 分钟）
```bash
python scripts/train_ppo_scenarios.py --scenario HorizontalCR --timesteps 10000 --device cuda
```

### 多种子训练（100k 步，约 2 小时）
```bash
python scripts/train_multi_seed.py --scenario HorizontalCR --algorithm PPO --timesteps 100000 --seeds 42 123 456 789 1024 --device cuda
```

### 收集结果
```bash
tar -czvf results.tar.gz models/ results/
```

---

**祝训练顺利！**