#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INSTALL_CMD=(pip install -e .)
VERIFY_CMD=(pytest tests/ -v --ignore=tests/integration)
LINT_CMD=(ruff check src/ tests/)
FORMAT_CMD=(ruff format --check src/ tests/)
TYPE_CMD=(mypy src/bluesky_pettingzoo/)
TRAIN_CMD=(python scripts/train_smoke_test.py)

echo "==> 当前目录: $PWD"
echo "==> 安装依赖"
"${INSTALL_CMD[@]}"

echo "==> 运行单元测试"
"${VERIFY_CMD[@]}"

echo "==> 代码风格检查"
"${LINT_CMD[@]}"

echo "==> 代码格式化检查"
"${FORMAT_CMD[@]}"

echo "==> 静态类型检查"
"${TYPE_CMD[@]}"

echo "==> 训练 smoke test"
"${TRAIN_CMD[@]}"

echo ""
echo "所有验证通过。可用命令："
echo "  pytest tests/ -v --ignore=tests/integration    # 单元测试"
echo "  pytest tests/integration/ -v                   # 集成测试"
echo "  python scripts/train_ppo_scenarios.py --scenario HorizontalCR --timesteps 100000"
echo "  python scripts/evaluate_baselines.py --scenario HorizontalCR"

if [ "${RUN_START_COMMAND:-0}" = "1" ]; then
  echo "==> 启动训练 smoke test"
  exec "${TRAIN_CMD[@]}"
fi

echo "如果希望 init.sh 直接启动训练，请设置 RUN_START_COMMAND=1。"
