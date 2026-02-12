from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bandits import AgentSpec, BanditSpec, RunSpec, simulate_eps_greedy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-Armed Bandits (ε-greedy) experiments")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Directory to write metrics (.npz) and a summary.json",
    )

    p.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["disabled", "offline", "online"],
        help="W&B mode. Use 'online' to log to your public project.",
    )
    p.add_argument("--wandb_project", type=str, default="k-armed-bandits")
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Your W&B entity/team (optional).",
    )
    p.add_argument(
        "--wandb_group",
        type=str,
        default="grid",
        help="Group name used to compare runs in W&B UI.",
    )
    p.add_argument(
        "--wandb_tags",
        type=str,
        default="pa2,k-armed-bandits",
        help="Comma-separated tags.",
    )
    return p.parse_args()


def _maybe_init_wandb(mode: str) -> Any:
    if mode == "disabled":
        return None
    import wandb

    if mode == "offline":
        # equivalent to setting WANDB_MODE=offline
        wandb.setup(settings=wandb.Settings(mode="offline"))
    else:
        wandb.setup()
    return wandb


def _config_name(agent: AgentSpec) -> str:
    alpha_str = "const0.1" if agent.alpha_type == "constant" else "sampleavg"
    return f"eps={agent.epsilon}_qinit={agent.q_init}_alpha={alpha_str}"


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bandit = BanditSpec(k=10)
    base_run = RunSpec(steps=args.steps, runs=args.runs, seed=args.seed)

    grid: list[AgentSpec] = []
    for epsilon in (0.0, 0.01, 0.1):
        for q_init in (0.0, 5.0):
            grid.append(AgentSpec(epsilon=epsilon, q_init=q_init, alpha_type="constant", alpha_constant=0.1))
            grid.append(AgentSpec(epsilon=epsilon, q_init=q_init, alpha_type="sample_average"))

    wandb = _maybe_init_wandb(args.wandb_mode)
    tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]

    summary: dict[str, Any] = {
        "bandit": asdict(bandit),
        "run": asdict(base_run),
        "grid_size": len(grid),
        "artifacts": [],
    }

    for i, agent in enumerate(grid):
        run = RunSpec(steps=base_run.steps, runs=base_run.runs, seed=base_run.seed + 1000 * i)
        name = _config_name(agent)
        metrics = simulate_eps_greedy(bandit=bandit, agent=agent, run=run)

        out_path = outdir / f"{name}.npz"
        np.savez_compressed(out_path, **metrics)

        record = {
            "name": name,
            "agent": asdict(agent),
            "run": asdict(run),
            "file": str(out_path),
            "final_avg_reward": float(metrics["avg_reward"][-1]),
            "final_pct_optimal": float(metrics["pct_optimal_action"][-1]),
            "final_cumulative_regret": float(metrics["cumulative_regret"][-1]),
        }
        summary["artifacts"].append(record)

        if wandb is not None:
            try:
                wb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    group=args.wandb_group,
                    name=name,
                    tags=tags,
                    config={
                        "bandit": asdict(bandit),
                        "agent": asdict(agent),
                        "run": asdict(run),
                    },
                    reinit="finish_previous",
                )
                wandb.define_metric("step")
                wandb.define_metric("avg_reward", step_metric="step")
                wandb.define_metric("pct_optimal_action", step_metric="step")
                wandb.define_metric("avg_regret", step_metric="step")
                wandb.define_metric("cumulative_regret", step_metric="step")

                for t in range(run.steps):
                    wandb.log(
                        {
                            "step": t + 1,
                            "avg_reward": float(metrics["avg_reward"][t]),
                            "pct_optimal_action": float(metrics["pct_optimal_action"][t]),
                            "avg_regret": float(metrics["avg_regret"][t]),
                            "cumulative_regret": float(metrics["cumulative_regret"][t]),
                        },
                        step=t + 1,
                    )
                wb_run.finish()
            except Exception as e:
                # Keep the assignment runnable even if WandB permissions/networking fail.
                # Results are still saved under outdir.
                print(f"[wandb] logging failed for {name}: {e}")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
