# K-Armed Bandits (PA2)

Project Assignment 2: The Gambler’s Dilemma (K-Armed Bandits)

## LLM Usage

Used GitHub Copilot (model: GPT-5.3-Codex) for implementation and documentation assistance.

## Overview

This repo implements a stationary 10-armed bandit environment and an ε-greedy agent. It runs a grid of experiments and logs:

- Average Reward (per step, averaged over runs)
- % Optimal Action (per step)
- Cumulative Regret (expected regret: $\mu^* - \mu_{a_t}$)

## Setup

Create a virtual environment and install dependencies:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Run Experiments (saves results to disk)

Runs the required hyperparameter combinations:

- ε ∈ {0.0, 0.01, 0.1}
- Q1 ∈ {0, 5}
- α ∈ {0.1 (constant), 1/n (sample average)}

Minimum 2,000 steps per run is the default.

```bash
python3 run_experiments.py --steps 2000 --runs 200 --seed 0 --wandb_mode disabled
```

Outputs:

- `results/*.npz` per configuration
- `results/summary.json`

## Log to Weights & Biases (WandB)

1. Log in:

```bash
wandb login
```

2. Run with online logging:

```bash
python3 run_experiments.py --wandb_mode online --wandb_project k-armed-bandits --wandb_entity <YOUR_ENTITY>
```

If you see an error like "Personal entities are disabled", you need to set `--wandb_entity` to your *team* entity (the one shown in `wandb status` / your `wandb.ai/<entity>` URL), not your personal username.

Notes:

- Set the WandB project visibility to **public** to satisfy submission requirements.
- In WandB, create a **Report** with at least these two charts comparing runs:
	- `avg_reward` vs `step`
	- `pct_optimal_action` vs `step`

## Local Plots (no WandB required)

After running experiments:

```bash
python3 plot_results.py --results_dir results --outdir plots
```

This produces:

- `plots/avg_reward.png`
- `plots/pct_optimal_action.png`
- `plots/cumulative_regret.png`

## Files

- `bandits.py`: Bandit environment + ε-greedy simulation
- `run_experiments.py`: Required grid runner + optional WandB logging
- `plot_results.py`: Local plotting from saved results