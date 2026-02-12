from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot saved bandit results from results/*.npz")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--outdir", type=str, default="plots")
    return p.parse_args()


def _load_npz_files(results_dir: Path) -> list[tuple[str, dict[str, np.ndarray]]]:
    files = sorted(results_dir.glob("*.npz"))
    data: list[tuple[str, dict[str, np.ndarray]]] = []
    for f in files:
        arrays = dict(np.load(f))
        data.append((f.stem, arrays))
    return data


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = _load_npz_files(results_dir)
    if not runs:
        raise SystemExit(f"No .npz files found in {results_dir}")

    steps = len(runs[0][1]["avg_reward"])
    x = np.arange(1, steps + 1)

    def plot_metric(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 5))
        for name, arrays in runs:
            y = arrays[metric]
            plt.plot(x, y, linewidth=1.2, label=name)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=200)
        plt.close()

    plot_metric("avg_reward", "Average Reward", "avg_reward.png")
    plot_metric("pct_optimal_action", "% Optimal Action", "pct_optimal_action.png")
    plot_metric("cumulative_regret", "Cumulative Regret", "cumulative_regret.png")


if __name__ == "__main__":
    main()
