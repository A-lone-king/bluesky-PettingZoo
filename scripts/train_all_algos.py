"""Batch training script: SAC/TD3/DDPG on all 10 scenarios.

Usage:
    python scripts/train_all_algos.py --timesteps 200000
    python scripts/train_all_algos.py --timesteps 200000 --algos SAC TD3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCENARIOS = [
    "HorizontalCR", "VerticalCR", "SectorCR",
    "WaypointNav", "Merge", "Descent",
    "StaticObstacle", "SectorCapacity", "RouteNav", "PlanWaypoint",
]

ALGORITHMS = ["SAC", "TD3", "DDPG"]


def run_training(scenario: str, algorithm: str, timesteps: int, save_dir: str) -> bool:
    """Run a single training job via subprocess with generous timeout."""
    import subprocess
    cmd = [
        sys.executable, "scripts/train_ppo_scenarios.py",
        "--scenario", scenario,
        "--algorithm", algorithm,
        "--timesteps", str(timesteps),
        "--save-dir", save_dir,
        "--max-steps", "50",
        "--num-aircraft", "3",
    ]
    print(f"\n{'='*60}")
    print(f"Training {algorithm} on {scenario} ({timesteps} steps)...")
    print(f"{'='*60}")
    start = time.time()

    try:
        # Stream output in real-time instead of capturing
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # Print subprocess output line by line
        for line in process.stdout:
            print(f"  {line}", end="")
        process.wait(timeout=1800)
        elapsed = time.time() - start
        if process.returncode == 0:
            print(f"  OK {algorithm}/{scenario} done in {elapsed:.0f}s")
            return True
        else:
            print(f"  FAIL {algorithm}/{scenario} FAILED ({elapsed:.0f}s)")
            return False
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  TIMEOUT {algorithm}/{scenario} after {elapsed:.0f}s")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch train SAC/TD3/DDPG on all scenarios")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--algos", nargs="+", default=ALGORITHMS, choices=ALGORITHMS)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    args = parser.parse_args()

    total = len(args.algos) * len(args.scenarios)
    success = 0
    failed_jobs: list[str] = []

    print(f"Starting batch training: {len(args.algos)} algorithms × {len(args.scenarios)} scenarios = {total} jobs")
    print(f"{'='*60}")

    job_idx = 0
    for algo in args.algos:
        for scenario in args.scenarios:
            job_idx += 1
            # Check if model already exists
            model_path = Path(args.save_dir) / scenario / algo / "checkpoint_final.zip"
            if model_path.exists():
                print(f"\n  [{job_idx}/{total}] Skipping {algo}/{scenario} — model already exists")
                success += 1
                continue

            print(f"\n  [{job_idx}/{total}] Starting {algo}/{scenario}...")
            ok = run_training(scenario, algo, args.timesteps, args.save_dir)
            if ok:
                success += 1
            else:
                failed_jobs.append(f"{algo}/{scenario}")

    print(f"\n{'='*60}")
    print(f"Batch training complete: {success}/{total} succeeded")
    if failed_jobs:
        print(f"Failed: {', '.join(failed_jobs)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
